from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import numpy as np
import pytest
import rasterio
from pyproj import Transformer
from rasterio.transform import from_origin

from nasa_agri_intelligence.cell_statistics import run_cell_statistics
from nasa_agri_intelligence.collection.task_manager import (
    AOISettings,
    CollectionSettings,
    SpatialAnalysisSettings,
)
from nasa_agri_intelligence.collection.manifest import write_manifest
from nasa_agri_intelligence.collection.task_registry import TaskRegistry


def _settings(
    bounds: tuple[float, float, float, float],
    *,
    cell_size_m: float,
    output_directory: Path,
    max_grid_cells: int = 500_000,
    polygon_vertices: list[tuple[float, float]] | None = None,
) -> CollectionSettings:
    min_x, min_y, max_x, max_y = bounds
    transformer = Transformer.from_crs("EPSG:32642", "EPSG:4326", always_xy=True)
    vertices = polygon_vertices or [
        (min_x, min_y),
        (max_x, min_y),
        (max_x, max_y),
        (min_x, max_y),
        (min_x, min_y),
    ]
    if vertices[0] != vertices[-1]:
        vertices = [*vertices, vertices[0]]
    coordinates = [transformer.transform(x, y) for x, y in vertices]
    return CollectionSettings(
        aoi=AOISettings(
            name="test-area",
            crs="EPSG:4326",
            geojson={
                "type": "FeatureCollection",
                "features": [
                    {
                        "type": "Feature",
                        "properties": {},
                        "geometry": {
                            "type": "Polygon",
                            "coordinates": [coordinates],
                        },
                    }
                ],
            },
        ),
        start_date=date(2025, 1, 1),
        end_date=date(2025, 1, 31),
        spatial_analysis=SpatialAnalysisSettings(
            analysis_crs="EPSG:32642",
            cell_size_m=cell_size_m,
            max_grid_cells=max_grid_cells,
            output_directory=output_directory,
        ),
    )


def _run(
    tmp_path: Path,
    values: np.ndarray,
    *,
    pixel_size: int,
    cell_size_m: int,
    kind: str = "continuous",
    nodata: int | None = None,
    max_grid_cells: int = 500_000,
    aoi_bounds: tuple[float, float, float, float] | None = None,
    polygon_vertices: list[tuple[float, float]] | None = None,
) -> tuple[dict, dict]:
    root = tmp_path
    task_name = "test-task"
    task_id = "test-task-id"
    task_root = root / "data/tasks" / task_name
    registry = TaskRegistry(task_root.parent)
    registry.reserve(
        task_name,
        {"task_id": task_id, "task_directory": str(task_root)},
    )
    raster_path = task_root / "processed/rasters/input.tif"
    raster_path.parent.mkdir(parents=True, exist_ok=True)
    raster_transform = from_origin(500_000, 4_601_000, pixel_size, pixel_size)
    with rasterio.open(
        raster_path,
        "w",
        driver="GTiff",
        width=values.shape[1],
        height=values.shape[0],
        count=1,
        dtype=values.dtype,
        crs="EPSG:32642",
        transform=raster_transform,
        nodata=nodata,
    ) as output:
        output.write(values, 1)
        output.update_tags(
            source_product_id="TEST.001",
            source_layer_id="_test_layer",
            variable_kind=kind,
            is_quality=str(kind == "quality").lower(),
            quality_layers=json.dumps(["_test_qa"] if kind == "continuous" else []),
            quality_filter_applied="false",
            standardized_units="index" if kind == "continuous" else "",
        )
    settings = _settings(
        aoi_bounds or (
            500_000,
            4_601_000 - values.shape[0] * pixel_size,
            500_000 + values.shape[1] * pixel_size,
            4_601_000,
        ),
        cell_size_m=cell_size_m,
        output_directory=Path("data/processed/cell_statistics"),
        max_grid_cells=max_grid_cells,
        polygon_vertices=polygon_vertices,
    )
    task_manifest = {
        "task_name": task_name,
        "task_id": task_id,
        "aoi": settings.aoi.model_dump(mode="json"),
        "spatial_analysis": {
            "analysis_crs": settings.spatial_analysis.analysis_crs,
            "requested_cell_size_m": settings.spatial_analysis.cell_size_m,
        },
        "product_id": "TEST.001",
        "layers": ["_test_layer"],
        "preprocessing_status": "completed",
        "analysis_runs": [],
    }
    write_manifest(task_root / "task_manifest.json", task_manifest)
    write_manifest(
        task_root / "task_metadata.json",
        {"task_name": task_name, "task_id": task_id},
    )
    write_manifest(
        task_root / "processed/standardized_manifest.json",
        {
            "task_name": task_name,
            "task_id": task_id,
            "created_at": "test",
            "source_directory": "raw",
            "records": [
                {
                    "product_id": "TEST.001",
                    "layer_id": "_test_layer",
                    "variable_kind": kind,
                    "quality_layers": ["_test_qa"] if kind == "continuous" else [],
                    "quality_filter_applied": False,
                    "output_path": "processed/rasters/input.tif",
                }
            ],
        },
    )
    result = run_cell_statistics(
        settings,
        task_directory=task_root,
        task_name=task_name,
        task_id=task_id,
    )
    geojson_path = Path(result["files"][0]["output_path"])
    geojson = json.loads(geojson_path.read_text(encoding="utf-8"))
    return result, geojson


def test_native_continuous_values_are_summarized_per_cell_with_coverage(
    tmp_path: Path,
) -> None:
    values = np.arange(1, 17, dtype=np.float32).reshape(4, 4)
    values[3, 3] = np.nan
    _, geojson = _run(
        tmp_path,
        values,
        pixel_size=250,
        cell_size_m=500,
        nodata=None,
    )

    assert geojson["metadata"]["effective_cell_size_m"] == 500
    assert geojson["metadata"]["quality_status"] == "unfiltered; QA retained separately"
    by_id = {feature["properties"]["cell_id"]: feature["properties"] for feature in geojson["features"]}
    first = by_id["r000000_c000000"]
    assert first["mean"] == pytest.approx(3.5)
    assert first["median"] == pytest.approx(3.5)
    assert first["minimum"] == pytest.approx(1.0)
    assert first["maximum"] == pytest.approx(6.0)
    assert first["valid_data_coverage"] == pytest.approx(1.0)
    last = by_id["r000001_c000001"]
    assert last["valid_pixel_count"] == 3
    assert last["native_pixel_center_count"] == 4
    assert last["valid_data_coverage"] == pytest.approx(0.75)
    assert last["mean"] == pytest.approx(38.0 / 3)
    assert "not exact area coverage" in last["coverage_definition"]


def test_requested_grid_never_upsamples_coarse_native_data(tmp_path: Path) -> None:
    values = np.array([[1.0, 3.0], [5.0, 7.0]], dtype=np.float32)
    result, geojson = _run(
        tmp_path,
        values,
        pixel_size=1000,
        cell_size_m=250,
    )

    assert result["files"][0]["requested_cell_size_m"] == 250
    assert result["files"][0]["effective_cell_size_m"] == 1000
    assert len(geojson["features"]) == 4
    assert all(
        feature["properties"]["mean"] == feature["properties"]["minimum"]
        for feature in geojson["features"]
    )


def test_quality_layers_are_reported_as_categories_not_averaged(
    tmp_path: Path,
) -> None:
    _, geojson = _run(
        tmp_path,
        np.array([[1, 1], [2, 3]], dtype=np.uint16),
        pixel_size=250,
        cell_size_m=500,
        kind="quality",
    )

    assert geojson["metadata"]["statistics"] == [
        "category_counts",
        "valid_data_coverage",
    ]
    properties = geojson["features"][0]["properties"]
    assert properties["mean"] is None
    assert properties["category_counts"] == {"1": 2, "2": 1, "3": 1}


def test_polygon_aoi_clips_boundary_grid_cell_geometry(tmp_path: Path) -> None:
    _, geojson = _run(
        tmp_path,
        np.ones((2, 2), dtype=np.float32),
        pixel_size=250,
        cell_size_m=500,
        polygon_vertices=[
            (500_000, 4_600_500),
            (500_500, 4_600_500),
            (500_000, 4_601_000),
        ],
    )

    assert len(geojson["features"]) == 1
    assert geojson["features"][0]["properties"]["aoi_overlap_area_m2"] == pytest.approx(
        125_000,
        abs=1,
    )


def test_config_rejects_geographic_or_non_meter_analysis_crs() -> None:
    with pytest.raises(ValueError, match="must be projected"):
        SpatialAnalysisSettings(analysis_crs="EPSG:4326")
    with pytest.raises(ValueError, match="must use meter units"):
        SpatialAnalysisSettings(analysis_crs="EPSG:2263")


def test_grid_cell_limit_fails_with_actionable_message(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="Increase cell_size_m or reduce the AOI"):
        _run(
            tmp_path,
            np.ones((1, 1), dtype=np.float32),
            pixel_size=250,
            cell_size_m=250,
            max_grid_cells=10,
            aoi_bounds=(500_000, 4_600_000, 502_500, 4_602_500),
        )
