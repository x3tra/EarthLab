from __future__ import annotations

import json
import shutil
from datetime import date
from pathlib import Path

import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin

from nasa_agri_intelligence.collection.task_registry import TaskRegistry
from nasa_agri_intelligence.temporal import (
    aggregate_monthly,
    build_temporal_inventory,
)


def _write_raster(
    path: Path,
    values: np.ndarray,
    *,
    quality_layers: list[str] | None = None,
    quality_filter_applied: bool = False,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        width=values.shape[1],
        height=values.shape[0],
        count=1,
        dtype=values.dtype,
        crs="EPSG:4326",
        transform=from_origin(0, 2, 1, 1),
        nodata=(
            float("nan")
            if np.issubdtype(values.dtype, np.floating)
            else None
        ),
    ) as output:
        output.write(values, 1)
        output.update_tags(
            source_product_id="MOD13Q1.061",
            source_layer_id="_250m_16_days_NDVI",
            standardized_units="NDVI",
            quality_layers=json.dumps(quality_layers or []),
            quality_filter_applied=str(quality_filter_applied).lower(),
        )


def _dated_path(folder: Path, day: str) -> Path:
    return folder / f"MOD13Q1.061_NDVI_{day}T000000_aid0001.tif"


def _register_task(tmp_path: Path, raster_paths: list[Path]) -> Path:
    registry = TaskRegistry(tmp_path / "data/tasks")
    task_root = registry.reserve(
        "temporal-test",
        {"task_id": "task-1", "task_directory": str(tmp_path / "data/tasks/temporal-test")},
    )
    (task_root / "task_metadata.json").write_text(
        json.dumps({"task_name": "temporal-test", "task_id": "task-1"}),
        encoding="utf-8",
    )
    owned_paths = []
    for path in raster_paths:
        owned_path = task_root / "processed/rasters" / path.name
        owned_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, owned_path)
        owned_paths.append(owned_path)
    product_id = "MOD13Q1.061"
    layer_ids = sorted(
        {rasterio.open(path).tags()["source_layer_id"] for path in owned_paths}
    )
    (task_root / "task_manifest.json").write_text(
        json.dumps(
            {
                "task_name": "temporal-test",
                "task_id": "task-1",
                "product_id": product_id,
                "layers": layer_ids,
                "start_date": "2025-01-01",
                "end_date": "2025-03-31",
            }
        ),
        encoding="utf-8",
    )
    records = []
    for path in owned_paths:
        records.append(
            {
                "product_id": product_id,
                "layer_id": rasterio.open(path).tags()["source_layer_id"],
                "output_path": str(path),
            }
        )
    (task_root / "processed/standardized_manifest.json").write_text(
        json.dumps(
            {
                "task_name": "temporal-test",
                "task_id": "task-1",
                "created_at": "2025-04-01T00:00:00+00:00",
                "source_directory": str(task_root / "raw"),
                "rasters": records,
            }
        ),
        encoding="utf-8",
    )
    return task_root


def test_temporal_inventory_reports_real_dates_requested_range_and_internal_gap(
    tmp_path: Path,
) -> None:
    task_root = TaskRegistry(tmp_path / "data/tasks").reserve(
        "temporal-test",
        {"task_id": "task-1", "task_directory": str(tmp_path / "data/tasks/temporal-test")},
    )
    (task_root / "task_metadata.json").write_text(
        json.dumps({"task_name": "temporal-test", "task_id": "task-1"}),
        encoding="utf-8",
    )
    (task_root / "task_manifest.json").write_text(
        json.dumps(
            {
                "task_name": "temporal-test",
                "task_id": "task-1",
                "product_id": "MOD13Q1.061",
                "layers": ["_250m_16_days_NDVI"],
                "start_date": "2025-01-01",
                "end_date": "2025-02-28",
            }
        ),
        encoding="utf-8",
    )
    observations = ["20250101", "20250117", "20250218"]
    records = []
    paths = []
    for day in observations:
        output_path = task_root / "processed/rasters" / f"MOD13Q1.061_NDVI_{day}T000000_aid0001.tif"
        _write_raster(
            output_path,
            np.ones((1, 1), dtype=np.float32),
        )
        paths.append(output_path)
        records.append(
            {
                "product_id": "MOD13Q1.061",
                "layer_id": "_250m_16_days_NDVI",
                "output_path": str(output_path),
                "valid_pixel_count": 1,
                "masked_pixel_count": 0,
            }
        )
    (task_root / "processed/standardized_manifest.json").write_text(
        json.dumps(
            {
                "task_name": "temporal-test",
                "task_id": "task-1",
                "created_at": "2025-04-01T00:00:00+00:00",
                "source_directory": str(task_root / "raw"),
                "records": records,
            }
        ),
        encoding="utf-8",
    )
    metadata_dir = tmp_path / "data/metadata"
    metadata_dir.mkdir(parents=True)
    (metadata_dir / "MOD13Q1.json").write_text(
        json.dumps(
            {
                "available_products": [
                    {
                        "catalog": {
                            "ProductAndVersion": "MOD13Q1.061",
                            "TemporalGranularity": "16 day",
                        }
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    inventory = build_temporal_inventory(
        task_root,
        task_name="temporal-test",
        task_id="task-1",
        metadata_dir=metadata_dir,
    )

    series = inventory["series"][0]
    assert series["observation_count"] == 3
    assert series["requested_date_ranges"] == [
        {
            "task_id": "task-1",
            "start_date": "2025-01-01",
            "end_date": "2025-02-28",
        }
    ]
    assert series["missing_cadence_slots"] == ["2025-02-02"]
    assert series["interpolated_observation_count"] == 0


def test_monthly_aggregation_records_missing_month_and_pixel_counts(
    tmp_path: Path,
) -> None:
    first = _dated_path(tmp_path / "input", "20250105")
    second = _dated_path(tmp_path / "input", "20250305")
    _write_raster(first, np.array([[2.0, np.nan]], dtype=np.float32))
    _write_raster(second, np.array([[8.0, 10.0]], dtype=np.float32))
    task_root = _register_task(tmp_path, [first, second])
    first = task_root / "processed/rasters" / first.name
    second = task_root / "processed/rasters" / second.name

    manifest = aggregate_monthly(
        [first, second],
        task_directory=task_root,
        task_name="temporal-test",
        task_id="task-1",
        requested_start=date(2025, 1, 1),
        requested_end=date(2025, 3, 31),
    )

    assert [period["status"] for period in manifest["periods"]] == [
        "aggregated",
        "missing",
        "aggregated",
    ]
    assert manifest["missing_period_count"] == 1
    assert manifest["interpolation"] == "none"
    with rasterio.open(task_root / "processed/temporal/monthly/2025-01-mean.tif") as output:
        assert output.read(1)[0, 0] == pytest.approx(2.0)
    with rasterio.open(task_root / "processed/temporal/monthly/2025-01-observation-count.tif") as output:
        assert output.read(1).tolist() == [[1, 0]]
    with rasterio.open(task_root / "processed/temporal/monthly/2025-03-mean.tif") as output:
        assert output.read(1).tolist() == [[8.0, 10.0]]
    assert Path(manifest["manifest_path"]).is_file()


def test_monthly_aggregation_requires_explicit_qa_override(tmp_path: Path) -> None:
    raster = _dated_path(tmp_path / "input", "20250105")
    _write_raster(
        raster,
        np.array([[0.5]], dtype=np.float32),
        quality_layers=["_250m_16_days_VI_Quality"],
    )
    task_root = _register_task(tmp_path, [raster])
    raster = task_root / "processed/rasters" / raster.name

    with pytest.raises(ValueError, match="quality layers that have not been applied"):
        aggregate_monthly(
            [raster],
            task_directory=task_root,
            task_name="temporal-test",
            task_id="task-1",
        )

    manifest = aggregate_monthly(
        [raster],
        task_directory=task_root,
        task_name="temporal-test",
        task_id="task-1",
        allow_unfiltered_quality=True,
    )

    period = manifest["periods"][0]
    assert period["quality_filter_applied"] is False
    assert period["unfiltered_quality_override"] is True
    with rasterio.open(period["output_path"]) as output:
        assert output.tags()["quality_filter_applied"] == "false"
        assert output.tags()["unfiltered_quality_override"] == "true"


def test_monthly_aggregation_rejects_categorical_and_qa_rasters(
    tmp_path: Path,
) -> None:
    raster = _dated_path(tmp_path / "input", "20250105")
    _write_raster(raster, np.array([[3]], dtype=np.uint16))
    task_root = _register_task(tmp_path, [raster])
    raster = task_root / "processed/rasters" / raster.name

    with pytest.raises(ValueError, match="continuous floating-point"):
        aggregate_monthly(
            [raster],
            task_directory=task_root,
            task_name="temporal-test",
            task_id="task-1",
        )
