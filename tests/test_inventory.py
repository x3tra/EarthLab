from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import rasterio
from rasterio.transform import from_origin

from nasa_agri_intelligence.inventory import build_inventory, write_inventory


PRODUCT_ID = "MOD13Q1.061"
LAYER_ID = "_250m_16_days_NDVI"


def _write_raster(path: Path) -> bytes:
    path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        width=2,
        height=1,
        count=1,
        dtype="int16",
        crs="EPSG:4326",
        transform=from_origin(69, 42, 0.01, 0.01),
        nodata=-3000,
    ) as dataset:
        dataset.write(np.array([[[4265, -3000]]], dtype=np.int16))
        dataset.scales = (0.0001,)
        dataset.offsets = (0.0,)
        dataset.set_band_description(1, LAYER_ID)
    return path.read_bytes()


def test_build_inventory_joins_actual_raster_metadata_and_manifest(
    tmp_path: Path,
) -> None:
    data_dir = tmp_path / "data"
    raster_path = (
        data_dir
        / "raw/MODIS/MOD13Q1/MOD13Q1.061__250m_16_days_NDVI_20250712T000000_aid0001.tif"
    )
    raster_bytes = _write_raster(raster_path)
    checksum = hashlib.sha256(raster_bytes).hexdigest()

    metadata_dir = data_dir / "metadata"
    metadata_dir.mkdir(parents=True)
    (metadata_dir / "MOD13Q1.json").write_text(
        json.dumps(
            {
                "available_products": [
                    {
                        "catalog": {
                            "Product": "MOD13Q1",
                            "ProductAndVersion": PRODUCT_ID,
                        },
                        "layers": {
                            LAYER_ID: {
                                "ScaleFactor": 0.0001,
                                "AddOffset": 0,
                                "FillValue": -3000,
                                "Units": "NDVI",
                                "ValidMin": -0.2,
                                "ValidMax": 1.0,
                            }
                        },
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    (metadata_dir / "availability_report.json").write_text(
        json.dumps(
            {
                "generated_at": "2026-11-09T00:00:00+00:00",
                "catalog_product_count": 193,
                "datasets": [
                    {
                        "dataset": "MOD13Q1",
                        "status": "available",
                        "available_product_ids": [PRODUCT_ID],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    manifest_dir = data_dir / "manifests"
    manifest_dir.mkdir()
    (manifest_dir / "task-id.json").write_text(
        json.dumps(
            {
                "task_id": "task-id",
                "product_id": PRODUCT_ID,
                "layers": [LAYER_ID],
                "start_date": "2025-07-01",
                "end_date": "2025-07-16",
                "task_status": "completed",
                "download_status": "completed",
                "downloads": [
                    {
                        "local_path": str(raster_path.relative_to(tmp_path)),
                        "filename": raster_path.name,
                        "status": "validated",
                        "expected_size_bytes": len(raster_bytes),
                        "checksum": checksum,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    inventory = build_inventory(data_dir)

    assert inventory["file_count"] == 1
    assert inventory["valid_file_count"] == 1
    dataset = inventory["datasets"][0]
    assert dataset["product_id"] == PRODUCT_ID
    assert dataset["availability"]["status"] == "available"
    assert dataset["layers"][0]["layer_id"] == LAYER_ID
    file_record = dataset["files"][0]
    assert file_record["task_id"] == "task-id"
    assert file_record["acquisition_date"] == "2025-07-12"
    assert file_record["validation"]["sha256"] == checksum
    assert file_record["raster"]["width"] == 2
    assert file_record["raster"]["valid_pixel_count"] == 1
    assert file_record["product_layer_metadata"]["ScaleFactor"] == 0.0001


def test_inventory_marks_checksum_mismatch_invalid(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    raster_path = data_dir / "raw/MODIS/sample.tif"
    _write_raster(raster_path)
    manifest_dir = data_dir / "manifests"
    manifest_dir.mkdir(parents=True)
    (manifest_dir / "task-id.json").write_text(
        json.dumps(
            {
                "task_id": "task-id",
                "product_id": PRODUCT_ID,
                "downloads": [
                    {
                        "local_path": str(raster_path.relative_to(tmp_path)),
                        "status": "validated",
                        "expected_size_bytes": raster_path.stat().st_size,
                        "checksum": "0" * 64,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    inventory = build_inventory(data_dir)

    assert inventory["valid_file_count"] == 0
    assert inventory["invalid_file_count"] == 1
    assert "SHA-256 checksum" in inventory["datasets"][0]["files"][0]["validation"]["errors"][0]


def test_inventory_reports_missing_manifest_file(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    manifest_dir = data_dir / "manifests"
    manifest_dir.mkdir(parents=True)
    missing_path = data_dir / "raw/MODIS/MOD13Q1/missing.tif"
    (manifest_dir / "task-id.json").write_text(
        json.dumps(
            {
                "task_id": "task-id",
                "product_id": PRODUCT_ID,
                "downloads": [
                    {
                        "local_path": str(missing_path.relative_to(tmp_path)),
                        "filename": missing_path.name,
                        "status": "validated",
                        "expected_size_bytes": 100,
                        "checksum": "0" * 64,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    inventory = build_inventory(data_dir)

    assert inventory["file_count"] == 1
    assert inventory["invalid_file_count"] == 1
    assert inventory["datasets"][0]["files"][0]["validation"]["exists"] is False


def test_inventory_does_not_count_uninspected_formats_as_valid(
    tmp_path: Path,
) -> None:
    data_dir = tmp_path / "data"
    file_path = data_dir / "raw/MODIS/unknown.bin"
    file_path.parent.mkdir(parents=True)
    file_path.write_bytes(b"opaque")
    manifest_dir = data_dir / "manifests"
    manifest_dir.mkdir(parents=True)
    (manifest_dir / "task-id.json").write_text(
        json.dumps(
            {
                "task_id": "task-id",
                "product_id": PRODUCT_ID,
                "downloads": [
                    {
                        "local_path": str(file_path.relative_to(tmp_path)),
                        "filename": file_path.name,
                        "status": "validated",
                        "expected_size_bytes": file_path.stat().st_size,
                        "checksum": hashlib.sha256(file_path.read_bytes()).hexdigest(),
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    inventory = build_inventory(data_dir)

    validation = inventory["datasets"][0]["files"][0]["validation"]
    assert inventory["valid_file_count"] == 0
    assert validation["format_valid"] is None
    assert "No format validator" in validation["errors"][0]


def test_layer_fallback_preserves_leading_underscore(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    raster_path = (
        data_dir
        / "raw/MODIS/MOD13Q1.061__250m_16_days_NDVI_20250712T000000_aid0001.tif"
    )
    raster_bytes = _write_raster(raster_path)
    manifest_dir = data_dir / "manifests"
    manifest_dir.mkdir(parents=True)
    (manifest_dir / "task-id.json").write_text(
        json.dumps(
            {
                "task_id": "task-id",
                "product_id": PRODUCT_ID,
                "downloads": [
                    {
                        "local_path": str(raster_path.relative_to(tmp_path)),
                        "status": "validated",
                        "expected_size_bytes": len(raster_bytes),
                        "checksum": hashlib.sha256(raster_bytes).hexdigest(),
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    inventory = build_inventory(data_dir)

    assert inventory["datasets"][0]["files"][0]["layer_id"] == LAYER_ID


def test_write_inventory_creates_json_and_markdown_outputs(tmp_path: Path) -> None:
    inventory = build_inventory(tmp_path / "data")
    json_path, report_path = write_inventory(
        inventory,
        json_path=tmp_path / "data/metadata/dataset_inventory.json",
        report_path=tmp_path / "reports/data_inventory.md",
    )

    assert json.loads(json_path.read_text(encoding="utf-8"))["file_count"] == 0
    assert "# Local Satellite Data Inventory" in report_path.read_text(encoding="utf-8")
