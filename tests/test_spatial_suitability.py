from __future__ import annotations

from datetime import date
from pathlib import Path

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
from nasa_agri_intelligence.suitability.engine import evaluate_cell
from nasa_agri_intelligence.suitability.spatial import run_spatial_suitability


def test_spatial_suitability_rejects_unverified_crop_requirements(tmp_path: Path) -> None:
    task_name = "suitability-test"
    task_id = "suitability-test-id"
    task_root = tmp_path / "data/tasks" / task_name
    registry = TaskRegistry(task_root.parent)
    registry.reserve(
        task_name,
        {"task_id": task_id, "task_directory": str(task_root)},
    )
    requirements = Path(__file__).resolve().parents[1] / "data/reference/crop_requirements.json"
    (tmp_path / "data/reference").mkdir(parents=True, exist_ok=True)
    (tmp_path / "data/reference/crop_requirements.json").write_bytes(requirements.read_bytes())

    aoi = {
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
    task_settings = CollectionSettings(
        aoi=AOISettings(name="demo", geojson=aoi),
        start_date=date(2025, 1, 1),
        end_date=date(2025, 1, 2),
        spatial_analysis=SpatialAnalysisSettings(
            analysis_crs="EPSG:32642",
            cell_size_m=1000,
            max_grid_cells=10000,
        ),
    )
    transform = rasterio.transform.from_origin(524000, 4571000, 1000, 1000)
    raster_path = task_root / "processed/rasters/sample_ndvi.tif"
    with rasterio.open(
        raster_path,
        "w",
        driver="GTiff",
        height=2,
        width=2,
        count=1,
        dtype="float32",
        crs="EPSG:32642",
        transform=transform,
        nodata=np.nan,
    ) as dataset:
        dataset.write(np.array([[0.5, 0.7], [0.6, 0.8]], dtype=np.float32), 1)
        dataset.update_tags(
            source_product_id="MOD13Q1.061",
            source_layer_id="_250m_16_days_NDVI",
        )
    write_manifest(
        task_root / "task_manifest.json",
        {
            "task_name": task_name,
            "task_id": task_id,
            "aoi": task_settings.aoi.model_dump(mode="json"),
            "spatial_analysis": {
                "analysis_crs": "EPSG:32642",
                "requested_cell_size_m": 1000,
            },
            "product_id": "MOD13Q1.061",
            "layers": ["_250m_16_days_NDVI"],
            "preprocessing_status": "completed",
            "analysis_runs": [],
        },
    )
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
                    "output_path": "processed/rasters/sample_ndvi.tif",
                    "layer_id": "_250m_16_days_NDVI",
                    "product_id": "MOD13Q1.061",
                    "variable_kind": "continuous",
                }
            ],
        },
    )

    with pytest.raises(ValueError, match="no verified environmental suitability criteria"):
        run_spatial_suitability(
            task_settings,
            task_directory=task_root,
            task_name=task_name,
            task_id=task_id,
        )

    assert not any((task_root / "suitability/runs").iterdir())
    assert registry.load_task_manifest(task_name)[2]["analysis_runs"] == []


def test_repeated_unsupported_analysis_does_not_create_runs(tmp_path: Path) -> None:
    test_spatial_suitability_rejects_unverified_crop_requirements(tmp_path)
    task_name = "suitability-test"
    task_id = "suitability-test-id"
    task_root = tmp_path / "data/tasks" / task_name
    settings = CollectionSettings(
        aoi=AOISettings(
            name="demo",
            geojson=TaskRegistry(task_root.parent).load_task_manifest(task_name)[2]["aoi"]["geojson"],
        ),
        start_date=date(2025, 1, 1),
        end_date=date(2025, 1, 2),
        spatial_analysis=SpatialAnalysisSettings(
            analysis_crs="EPSG:32642",
            cell_size_m=1000,
            max_grid_cells=10000,
        ),
    )

    with pytest.raises(ValueError, match="no verified environmental suitability criteria"):
        run_spatial_suitability(
            settings,
            task_directory=task_root,
            task_name=task_name,
            task_id=task_id,
        )

    assert not any((task_root / "suitability/runs").iterdir())


def test_evaluate_cell_returns_none_when_no_real_guidance_matches() -> None:
    result = evaluate_cell({"ndvi": 0.4}, crop_ids=["cotton", "wheat"])
    assert result["best_crop"] is None
    assert result["crop_count"] == 2
