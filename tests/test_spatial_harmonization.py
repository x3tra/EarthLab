from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin

from nasa_agri_intelligence.collection.task_registry import TaskRegistry
from nasa_agri_intelligence.spatial_harmonization import (
    build_grid,
    coarsest_native_resolution,
    harmonize_raster,
)


AOI = {
    "type": "Polygon",
    "coordinates": [[
        [0.0, 0.0],
        [2.0, 0.0],
        [2.0, 2.0],
        [0.0, 2.0],
        [0.0, 0.0],
    ]],
}
PRODUCT_ID = "MOD13Q1.061"
LAYER_ID = "_250m_16_days_NDVI"


def _write_raster(
    path: Path,
    values: np.ndarray,
    *,
    nodata: int | None = None,
    pixel_size: float = 0.5,
) -> None:
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        width=values.shape[1],
        height=values.shape[0],
        count=1,
        dtype=values.dtype,
        crs="EPSG:4326",
        transform=from_origin(0, 2, pixel_size, pixel_size),
        nodata=nodata,
    ) as dataset:
        dataset.write(values, 1)
        dataset.update_tags(
            source_product_id=PRODUCT_ID,
            source_layer_id=LAYER_ID,
        )


def _task_root(tmp_path: Path, task_name: str = "test-task") -> Path:
    registry = TaskRegistry(tmp_path / "tasks")
    task_root = registry.reserve(task_name, {"task_id": task_name})
    (task_root / "task_metadata.json").write_text(
        json.dumps({"task_name": task_name, "task_id": task_name}),
        encoding="utf-8",
    )
    return task_root


def _write_task_manifests(
    task_root: Path,
    raster_paths: list[Path],
    task_name: str = "test-task",
) -> None:
    (task_root / "task_manifest.json").write_text(
        json.dumps(
            {
                "task_name": task_name,
                "task_id": task_name,
                "product_id": PRODUCT_ID,
                "layers": [LAYER_ID],
            }
        ),
        encoding="utf-8",
    )
    (task_root / "processed/standardized_manifest.json").write_text(
        json.dumps(
            {
                "task_name": task_name,
                "task_id": task_name,
                "created_at": "2025-01-01T00:00:00+00:00",
                "source_directory": str(task_root / "raw"),
                "rasters": [
                    {
                        "product_id": PRODUCT_ID,
                        "layer_id": LAYER_ID,
                        "output_path": str(path),
                    }
                    for path in raster_paths
                ],
            }
        ),
        encoding="utf-8",
    )


def test_build_grid_snaps_and_bounds_grid_to_aoi() -> None:
    grid = build_grid(AOI, target_crs="EPSG:4326", resolution=1.0)

    assert grid.width == 2
    assert grid.height == 2
    assert grid.transform == from_origin(0, 2, 1, 1)


def test_coarsest_native_resolution_selects_largest_source_pixels(
    tmp_path: Path,
) -> None:
    task_root = _task_root(tmp_path)
    fine = task_root / "processed/rasters/fine.tif"
    coarse = task_root / "processed/rasters/coarse.tif"
    _write_raster(fine, np.ones((4, 4), dtype=np.int16), pixel_size=0.5)
    _write_raster(coarse, np.ones((2, 2), dtype=np.int16), pixel_size=1.0)
    _write_task_manifests(task_root, [fine, coarse])

    resolution = coarsest_native_resolution(
        [fine, coarse],
        task_directory=task_root,
        task_name="test-task",
        task_id="test-task",
        target_crs="EPSG:4326",
    )

    assert resolution == pytest.approx(1.0)


def test_continuous_downsampling_averages_and_preserves_aoi_and_nodata(
    tmp_path: Path,
) -> None:
    task_root = _task_root(tmp_path)
    source_path = task_root / "processed/rasters/source.tif"
    output_path = task_root / "processed/harmonized/downsampled.tif"
    values = np.arange(1, 17, dtype=np.int16).reshape(4, 4)
    values[0, 0] = -9999
    _write_raster(source_path, values, nodata=-9999)
    _write_task_manifests(task_root, [source_path])
    grid = build_grid(AOI, target_crs="EPSG:4326", resolution=1.0)

    record = harmonize_raster(
        source_path,
        output_path,
        task_directory=task_root,
        task_name="test-task",
        task_id="test-task",
        grid=grid,
        aoi_geojson=AOI,
        kind="continuous",
    )

    with rasterio.open(output_path) as output:
        result = output.read(1, masked=True)
        assert output.shape == (2, 2)
        assert output.dtypes == ("float32",)
        assert result[0, 0] == pytest.approx((2 + 5 + 6) / 3)
        assert result[0, 1] == pytest.approx(5.5)
        assert np.ma.getmaskarray(result).tolist() == [[False, False], [False, False]]
        assert output.crs.to_epsg() == 4326
        assert output.transform == grid.transform
    assert record["resampling"] == "average"
    assert record["valid_pixel_count"] == 4


def test_quality_uses_nearest_neighbor_and_keeps_categorical_values(
    tmp_path: Path,
) -> None:
    task_root = _task_root(tmp_path)
    source_path = task_root / "processed/rasters/quality.tif"
    output_path = task_root / "processed/harmonized/quality-aligned.tif"
    _write_raster(
        source_path,
        np.array([[0, 1, 2, 3], [1, 1, 3, 3], [0, 2, 2, 3], [0, 0, 3, 3]], dtype=np.uint16),
    )
    _write_task_manifests(task_root, [source_path])
    grid = build_grid(AOI, target_crs="EPSG:4326", resolution=1.0)

    record = harmonize_raster(
        source_path,
        output_path,
        task_directory=task_root,
        task_name="test-task",
        task_id="test-task",
        grid=grid,
        aoi_geojson=AOI,
        kind="quality",
    )

    with rasterio.open(output_path) as output:
        result = output.read(1)
        assert output.dtypes == ("uint16",)
        assert set(np.unique(result)).issubset({0, 1, 2, 3})
    assert record["resampling"] == "nearest"


def test_harmonization_refuses_upsampling_by_default(tmp_path: Path) -> None:
    task_root = _task_root(tmp_path)
    source_path = task_root / "processed/rasters/source.tif"
    _write_raster(source_path, np.ones((4, 4), dtype=np.int16))
    _write_task_manifests(task_root, [source_path])
    finer_grid = build_grid(AOI, target_crs="EPSG:4326", resolution=0.25)

    with pytest.raises(ValueError, match="refusing to create false detail"):
        harmonize_raster(
            source_path,
            task_root / "processed/harmonized/upsampled.tif",
            task_directory=task_root,
            task_name="test-task",
            task_id="test-task",
            grid=finer_grid,
            aoi_geojson=AOI,
            kind="continuous",
        )


def test_harmonization_rejects_raster_owned_by_another_task(tmp_path: Path) -> None:
    task_a = _task_root(tmp_path, "task-a")
    task_b = _task_root(tmp_path, "task-b")
    raster_a = task_a / "processed/rasters/source.tif"
    raster_b = task_b / "processed/rasters/source.tif"
    _write_raster(raster_a, np.ones((4, 4), dtype=np.int16))
    _write_raster(raster_b, np.ones((4, 4), dtype=np.int16))
    _write_task_manifests(task_a, [raster_a], "task-a")
    _write_task_manifests(task_b, [raster_b], "task-b")

    with pytest.raises(ValueError, match="not listed in the selected task manifest"):
        harmonize_raster(
            raster_b,
            task_a / "processed/harmonized/output.tif",
            task_directory=task_a,
            task_name="task-a",
            task_id="task-a",
            grid=build_grid(AOI, target_crs="EPSG:4326", resolution=1.0),
            aoi_geojson=AOI,
            kind="continuous",
        )
