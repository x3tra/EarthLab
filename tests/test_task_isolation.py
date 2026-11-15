from __future__ import annotations

import json
from datetime import date
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import rasterio

from nasa_agri_intelligence.collection.manifest import write_manifest
from nasa_agri_intelligence.collection.task_manager import (
    AOISettings,
    CollectionSettings,
    SpatialAnalysisSettings,
)
from nasa_agri_intelligence.collection.task_registry import TaskRegistry
from nasa_agri_intelligence.standardization import standardize_task
from nasa_agri_intelligence.suitability import spatial
from nasa_agri_intelligence.suitability.spatial import run_spatial_suitability

PRODUCT_ID = "MOD13Q1.061"
LAYER_ID = "_250m_16_days_NDVI"

AOI = {
    "type": "FeatureCollection",
    "features": [
        {
            "type": "Feature",
            "properties": {},
            "geometry": {
                "type": "Polygon",
                "coordinates": [[
                    [69.2797, 41.3111],
                    [69.2807, 41.3111],
                    [69.2807, 41.3121],
                    [69.2797, 41.3121],
                    [69.2797, 41.3111],
                ]],
            },
        }
    ],
}


def _create_task(
    tmp_path: Path,
    task_name: str,
    task_id: str,
    value: float,
) -> tuple[TaskRegistry, Path, Path]:
    registry = TaskRegistry(tmp_path / "data/tasks")
    task_root = tmp_path / "data/tasks" / task_name
    registry.reserve(
        task_name,
        {"task_id": task_id, "task_directory": str(task_root)},
    )
    raster_path = task_root / "processed/rasters/identical-name.tif"
    with rasterio.open(
        raster_path,
        "w",
        driver="GTiff",
        width=2,
        height=2,
        count=1,
        dtype="float32",
        crs="EPSG:4326",
        transform=rasterio.transform.from_origin(69.30, 41.33, 0.01, 0.01),
        nodata=float("nan"),
    ) as raster:
        raster.write(np.full((2, 2), value, dtype=np.float32), 1)
        raster.update_tags(source_product_id=PRODUCT_ID, source_layer_id=LAYER_ID)

    task_manifest = {
        "task_name": task_name,
        "task_id": task_id,
        "created_at": "2026-11-12T00:00:00+00:00",
        "aoi": {"name": task_name, "crs": "EPSG:4326", "geojson": AOI},
        "product_id": PRODUCT_ID,
        "layers": [LAYER_ID],
        "start_date": "2025-01-01",
        "end_date": "2025-01-31",
        "spatial_analysis": {
            "analysis_crs": "EPSG:32642",
            "requested_cell_size_m": 1000,
        },
        "download_status": "completed",
        "preprocessing_status": "completed",
        "analysis_runs": [],
    }
    write_manifest(task_root / "task_manifest.json", task_manifest)
    write_manifest(
        task_root / "task_metadata.json",
        {
            "task_name": task_name,
            "task_id": task_id,
            "created_at": task_manifest["created_at"],
            "aoi": task_manifest["aoi"],
            "requested_products": [PRODUCT_ID],
            "requested_layers": [LAYER_ID],
            "start_date": task_manifest["start_date"],
            "end_date": task_manifest["end_date"],
            "local_task_directory": str(task_root),
        },
    )
    standardized = task_root / "processed/standardized_manifest.json"
    write_manifest(
        standardized,
        {
            "task_name": task_name,
            "task_id": task_id,
            "created_at": "2026-11-12T00:00:00+00:00",
            "source_directory": "raw",
            "rasters": [
                {
                    "product_id": PRODUCT_ID,
                    "layer_id": LAYER_ID,
                    "variable_kind": "continuous",
                    "output_path": "processed/rasters/identical-name.tif",
                }
            ],
        },
    )
    return registry, task_root, raster_path


def _settings() -> CollectionSettings:
    return CollectionSettings(
        aoi=AOISettings(name="config-aoi", geojson=AOI),
        start_date=date(2025, 1, 1),
        end_date=date(2025, 1, 31),
        spatial_analysis=SpatialAnalysisSettings(
            analysis_crs="EPSG:32642",
            cell_size_m=1000,
            max_grid_cells=10000,
        ),
    )


def test_tasks_can_store_same_filename_without_collision(tmp_path: Path) -> None:
    _, task_a, raster_a = _create_task(tmp_path, "task-a", "nasa-a", 0.2)
    _, task_b, raster_b = _create_task(tmp_path, "task-b", "nasa-b", 0.8)

    assert raster_a.name == raster_b.name
    assert raster_a != raster_b
    with rasterio.open(raster_a) as source_a, rasterio.open(raster_b) as source_b:
        assert source_a.read(1)[0, 0] == pytest.approx(0.2)
        assert source_b.read(1)[0, 0] == pytest.approx(0.8)
    assert task_a != task_b


def test_requested_science_layer_may_retain_only_its_documented_qa_layer(
    tmp_path: Path,
) -> None:
    registry, task_root, _ = _create_task(tmp_path, "task-a", "nasa-a", 0.3)
    quality_layer = "_250m_16_days_VI_Quality"
    other_quality_layer = "_250m_16_days_pixel_reliability"
    metadata_directory = tmp_path / "data/metadata"
    metadata_directory.mkdir(parents=True)
    write_manifest(
        metadata_directory / "MOD13Q1.json",
        {
            "available_products": [
                {
                    "catalog": {"ProductAndVersion": PRODUCT_ID},
                    "layers": {
                        LAYER_ID: {"QualityLayers": f"['{quality_layer}']"},
                        quality_layer: {"IsQA": True},
                        other_quality_layer: {"IsQA": True},
                    },
                }
            ]
        },
    )
    quality_path = task_root / "processed/rasters/quality.tif"
    with rasterio.open(
        quality_path,
        "w",
        driver="GTiff",
        width=2,
        height=2,
        count=1,
        dtype="uint8",
        crs="EPSG:4326",
        transform=rasterio.transform.from_origin(69.30, 41.33, 0.01, 0.01),
        nodata=255,
    ) as raster:
        raster.write(np.zeros((2, 2), dtype=np.uint8), 1)
        raster.update_tags(
            source_product_id=PRODUCT_ID,
            source_layer_id=quality_layer,
        )

    manifest_path = task_root / "processed/standardized_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["rasters"].append(
        {
            "product_id": PRODUCT_ID,
            "layer_id": quality_layer,
            "variable_kind": "quality",
            "output_path": "processed/rasters/quality.tif",
        }
    )
    write_manifest(manifest_path, manifest)

    _, _, records = registry.validate_raster_manifest("task-a", manifest_path)
    assert records[-1]["layer_id"] == quality_layer

    manifest["rasters"][-1]["layer_id"] = other_quality_layer
    write_manifest(manifest_path, manifest)
    with pytest.raises(ValueError, match="was not requested"):
        registry.validate_raster_manifest("task-a", manifest_path)


def test_requested_quality_layer_is_detected_across_multiple_metadata_files(
    tmp_path: Path,
) -> None:
    registry, task_root, _ = _create_task(tmp_path, "task-b", "nasa-b", 0.3)
    quality_layer = "_250m_16_days_VI_Quality"
    metadata_directory = tmp_path / "data/metadata"
    metadata_directory.mkdir(parents=True)
    write_manifest(
        metadata_directory / "other_metadata.json",
        {"available_products": [{"catalog": {"ProductAndVersion": "OTHER.001"}, "layers": {}}]},
    )
    write_manifest(
        metadata_directory / "MOD13Q1.json",
        {
            "available_products": [
                {
                    "catalog": {"ProductAndVersion": PRODUCT_ID},
                    "layers": {
                        LAYER_ID: {"QualityLayers": f"['{quality_layer}']"},
                        quality_layer: {"IsQA": True},
                    },
                }
            ]
        },
    )

    quality_path = task_root / "processed/rasters/quality.tif"
    with rasterio.open(
        quality_path,
        "w",
        driver="GTiff",
        width=2,
        height=2,
        count=1,
        dtype="uint8",
        crs="EPSG:4326",
        transform=rasterio.transform.from_origin(69.30, 41.33, 0.01, 0.01),
        nodata=255,
    ) as raster:
        raster.write(np.zeros((2, 2), dtype=np.uint8), 1)
        raster.update_tags(source_product_id=PRODUCT_ID, source_layer_id=quality_layer)

    manifest_path = task_root / "processed/standardized_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["rasters"].append(
        {
            "product_id": PRODUCT_ID,
            "layer_id": quality_layer,
            "variable_kind": "quality",
            "output_path": "processed/rasters/quality.tif",
            "source_product_id": PRODUCT_ID,
            "source_layer_id": quality_layer,
        }
    )
    write_manifest(manifest_path, manifest)

    _, _, records = registry.validate_raster_manifest("task-b", manifest_path)
    assert records[-1]["layer_id"] == quality_layer


def test_standardization_uses_only_selected_task_download_manifest(tmp_path: Path) -> None:
    metadata_directory = tmp_path / "data/metadata"
    metadata_directory.mkdir(parents=True)
    write_manifest(
        metadata_directory / "MOD13Q1.json",
        {
            "available_products": [
                {
                    "catalog": {"ProductAndVersion": PRODUCT_ID},
                    "layers": {
                        LAYER_ID: {
                            "ScaleFactor": 0.0001,
                            "FillValue": -3000,
                            "OrigValidMin": -2000,
                            "OrigValidMax": 10000,
                            "ValidMin": -0.2,
                            "ValidMax": 1.0,
                            "Units": "NDVI",
                            "QualityLayers": [],
                        }
                    },
                }
            ]
        },
    )
    registry = TaskRegistry(tmp_path / "data/tasks")
    task_directories: dict[str, Path] = {}
    for task_name, task_id, raw_value in (
        ("task-a", "nasa-a", 1000),
        ("task-b", "nasa-b", 8000),
    ):
        task_root = registry.reserve(
            task_name,
            {"task_id": task_id, "task_directory": str(tmp_path / "data/tasks" / task_name)},
        )
        task_directories[task_name] = task_root
        filename = f"MOD13Q1.061__{LAYER_ID}_20250626T000000_.tif"
        raw_path = task_root / "raw/MODIS/MOD13Q1" / filename
        raw_path.parent.mkdir(parents=True, exist_ok=True)
        with rasterio.open(
            raw_path,
            "w",
            driver="GTiff",
            width=1,
            height=1,
            count=1,
            dtype="int16",
            crs="EPSG:4326",
            transform=rasterio.transform.from_origin(69.3, 41.3, 0.01, 0.01),
            nodata=-3000,
        ) as raw:
            raw.write(np.array([[raw_value]], dtype=np.int16), 1)
            raw.update_tags(scale_factor="0.0001", add_offset="0")
        write_manifest(
            task_root / "task_manifest.json",
            {
                "task_name": task_name,
                "task_id": task_id,
                "download_status": "completed",
                "downloads": [
                    {
                        "filename": filename,
                        "local_path": str(raw_path),
                        "status": "validated",
                    }
                ],
            },
        )
        write_manifest(task_root / "task_metadata.json", {"task_name": task_name, "task_id": task_id})

    unrelated = task_directories["task-a"] / "raw/unlisted.tif"
    unrelated.write_bytes(b"not-a-raster-and-not-in-the-task-manifest")
    standardized = standardize_task(
        task_directories["task-a"],
        task_name="task-a",
        task_id="nasa-a",
        metadata_directory=metadata_directory,
    )

    assert standardized["task_name"] == "task-a"
    assert standardized["task_id"] == "nasa-a"
    assert standardized["raster_count"] == 1
    standardized_path = task_directories["task-a"] / standardized["rasters"][0]["output_path"]
    with rasterio.open(standardized_path) as result:
        assert result.read(1)[0, 0] == pytest.approx(0.1)
    assert not list((task_directories["task-b"] / "processed/rasters").rglob("*.tif"))


def test_derived_slope_is_validated_against_its_task_owned_dem(
    tmp_path: Path,
) -> None:
    task_name = "dem-task"
    task_id = "nasa-dem-task"
    product_id = "ASTGTM_NC.003"
    layer_id = "ASTER_GDEM_DEM"
    registry = TaskRegistry(tmp_path / "data/tasks")
    task_root = registry.reserve(
        task_name,
        {"task_id": task_id, "task_directory": str(tmp_path / "data/tasks" / task_name)},
    )
    write_manifest(
        task_root / "task_manifest.json",
        {
            "task_name": task_name,
            "task_id": task_id,
            "product_id": product_id,
            "layers": [layer_id],
        },
    )
    slope_path = task_root / "processed/rasters/derived/slope.tif"
    slope_path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(
        slope_path,
        "w",
        driver="GTiff",
        width=2,
        height=2,
        count=1,
        dtype="float32",
        crs="EPSG:4326",
        transform=rasterio.transform.from_origin(69.3, 41.3, 0.01, 0.01),
        nodata=float("nan"),
    ) as slope:
        slope.write(np.ones((2, 2), dtype=np.float32), 1)
        slope.update_tags(
            source_product_id=product_id,
            source_layer_id=layer_id,
            derived_from="digital_elevation_model",
        )
    manifest_path = task_root / "processed/standardized_manifest.json"
    manifest = {
        "task_name": task_name,
        "task_id": task_id,
        "created_at": "2026-11-12T00:00:00+00:00",
        "source_directory": "raw",
        "rasters": [
            {
                "product_id": "_DERIVED",
                "layer_id": "slope",
                "derived": True,
                "source_product_id": product_id,
                "source_layer_id": layer_id,
                "output_path": "processed/rasters/derived/slope.tif",
            }
        ],
    }
    write_manifest(manifest_path, manifest)

    _, _, validated = registry.validate_raster_manifest(task_name, manifest_path)

    assert len(validated) == 1
    assert validated[0]["output_path"] == str(slope_path)
    manifest["rasters"][0]["source_product_id"] = PRODUCT_ID
    write_manifest(manifest_path, manifest)
    with pytest.raises(ValueError, match="source product .* does not belong"):
        registry.validate_raster_manifest(task_name, manifest_path)


@pytest.mark.parametrize("task_name,task_id,other_name,other_id", [
    ("task-a", "nasa-a", "task-b", "nasa-b"),
    ("task-b", "nasa-b", "task-a", "nasa-a"),
])
def test_analysis_only_samples_selected_task_rasters(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    task_name: str,
    task_id: str,
    other_name: str,
    other_id: str,
) -> None:
    registry, _, selected_raster = _create_task(tmp_path, task_name, task_id, 0.4)
    _create_task(tmp_path, other_name, other_id, 0.9)
    reference = Path(__file__).resolve().parents[1] / "data/reference/crop_requirements.json"
    target_reference = tmp_path / "data/reference/crop_requirements.json"
    target_reference.parent.mkdir(parents=True, exist_ok=True)
    target_reference.write_bytes(reference.read_bytes())

    sampled: list[Path] = []
    actual_sample = spatial._sample_raster_value

    def track(path: str | Path, x: float, y: float, *, target_crs: str) -> float | None:
        sampled.append(Path(path).resolve())
        return actual_sample(path, x, y, target_crs=target_crs)

    monkeypatch.setattr(spatial, "_sample_raster_value", track)
    with pytest.raises(ValueError, match="no verified environmental suitability criteria"):
        run_spatial_suitability(
            _settings(),
            task_directory=registry.resolve(task_name)["task_directory"],
            task_name=task_name,
            task_id=task_id,
            crop_ids=["cotton"],
        )

    assert sampled == []
    assert selected_raster.is_file()


def test_wrong_task_name_is_rejected(tmp_path: Path) -> None:
    registry, task_root, _ = _create_task(tmp_path, "task-a", "nasa-a", 0.3)
    manifest_path = task_root / "processed/standardized_manifest.json"
    manifest = {
        **{
            "task_name": "task-b",
            "task_id": "nasa-a",
            "rasters": [
                {
                    "product_id": PRODUCT_ID,
                    "layer_id": LAYER_ID,
                    "output_path": "processed/rasters/identical-name.tif",
                }
            ],
        }
    }
    write_manifest(manifest_path, manifest)

    with pytest.raises(ValueError, match="task name"):
        registry.validate_raster_manifest("task-a", manifest_path)


def test_wrong_nasa_task_id_is_rejected(tmp_path: Path) -> None:
    registry, task_root, _ = _create_task(tmp_path, "task-a", "nasa-a", 0.3)
    manifest_path = task_root / "processed/standardized_manifest.json"
    manifest = {
        "task_name": "task-a",
        "task_id": "nasa-b",
        "rasters": [
            {
                "product_id": PRODUCT_ID,
                "layer_id": LAYER_ID,
                "output_path": "processed/rasters/identical-name.tif",
            }
        ],
    }
    write_manifest(manifest_path, manifest)

    with pytest.raises(ValueError, match="NASA task ID"):
        registry.validate_raster_manifest("task-a", manifest_path)


def test_raster_path_cannot_escape_task_root(tmp_path: Path) -> None:
    registry, task_root, _ = _create_task(tmp_path, "task-a", "nasa-a", 0.3)
    manifest_path = task_root / "processed/standardized_manifest.json"
    write_manifest(
        manifest_path,
        {
            "task_name": "task-a",
            "task_id": "nasa-a",
            "created_at": "test",
            "source_directory": "raw",
            "rasters": [
                {
                    "product_id": PRODUCT_ID,
                    "layer_id": LAYER_ID,
                    "output_path": "../../task-b/processed/rasters/identical-name.tif",
                }
            ],
        },
    )
    with pytest.raises(ValueError, match="escapes"):
        registry.validate_raster_manifest("task-a", manifest_path)


def test_missing_task_manifest_fails_explicitly(tmp_path: Path) -> None:
    registry = TaskRegistry(tmp_path / "data/tasks")
    task_root = registry.reserve(
        "task-a",
        {"task_id": "nasa-a", "task_directory": str(tmp_path / "data/tasks/task-a")},
    )
    with pytest.raises(FileNotFoundError, match="Task manifest is missing"):
        registry.validate_raster_manifest(
            "task-a",
            task_root / "processed/standardized_manifest.json",
        )


def test_duplicate_task_name_never_overwrites_existing_directory(tmp_path: Path) -> None:
    _, task_root, raster_path = _create_task(tmp_path, "task-a", "nasa-a", 0.3)
    before = raster_path.read_bytes()
    registry = TaskRegistry(task_root.parent)
    with pytest.raises(FileExistsError, match="already exists"):
        registry.reserve("task-a", {"task_id": "nasa-b"})
    assert raster_path.read_bytes() == before
    assert registry.resolve("task-a")["task_id"] == "nasa-a"
