from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

from fastapi.testclient import TestClient
import numpy as np
import pytest
import rasterio
import requests
from pyproj import Transformer

from nasa_agri_intelligence import web
from nasa_agri_intelligence.collection.manifest import write_manifest
from nasa_agri_intelligence.collection.task_registry import TaskRegistry

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PRODUCT_ID = "MOD13Q1.061"
LAYER_ID = "_250m_16_days_NDVI"


def _client() -> TestClient:
    return TestClient(web.create_app(project_root=PROJECT_ROOT))


def test_datasets_only_exposes_confirmed_products_and_layers() -> None:
    response = _client().get("/api/datasets")

    assert response.status_code == 200
    products = response.json()["products"]
    product_ids = [product["product_id"] for product in products]
    assert len(product_ids) == len(set(product_ids))
    modis = next(product for product in products if product["product_id"] == PRODUCT_ID)
    assert any(layer["layer_id"] == LAYER_ID for layer in modis["layers"])
    elevation_product = next(
        product for product in products if product["product_id"] == "ASTGTM_NC.003"
    )
    elevation = next(
        layer
        for layer in elevation_product["layers"]
        if layer["layer_id"] == "ASTER_GDEM_DEM"
    )
    assert elevation["quality_layers"] == []
    assert elevation["auxiliary_layers"] == ["ASTER_GDEM_NUM"]
    assert all(
        layer["available"] is True
        for product in products
        for layer in product["layers"]
    )
    assert all("NASA_APP_EARS_TOKEN" not in str(product) for product in products)


def test_vite_fallback_port_is_allowed_by_api_cors() -> None:
    response = _client().options(
        "/api/essential-layers",
        headers={
            "Origin": "http://127.0.0.1:5175",
            "Access-Control-Request-Method": "GET",
            "Access-Control-Request-Headers": "content-type",
        },
    )

    assert response.status_code == 200
    assert response.headers["access-control-allow-origin"] == "http://127.0.0.1:5175"


def test_essential_catalog_is_curated_and_reports_verified_metadata() -> None:
    response = _client().get("/api/essential-layers")

    assert response.status_code == 200
    catalog = response.json()
    layers = catalog["layers"]
    assert catalog["logical_layer_count"] == 10
    assert [item["id"] for item in layers] == [
        "ndvi",
        "lst",
        "evapotranspiration",
        "potential_evapotranspiration",
        "soil_properties",
        "elevation",
        "slope",
        "land_cover",
        "air_temperature",
        "precipitation",
    ]

    ndvi = layers[0]["variables"][0]
    assert ndvi["product_id"] == "MOD13Q1.061"
    assert ndvi["layer_id"] == LAYER_ID
    assert ndvi["spatial_resolution"] == "250m"
    assert ndvi["temporal_resolution"] == "16 day"
    assert ndvi["units"] == "NDVI"
    assert ndvi["scale_factor"] == 0.0001
    assert ndvi["qa_required"] is True
    assert ndvi["quality_layers"] == ["_250m_16_days_VI_Quality"]
    assert ndvi["fill_values"] == ["-3000"]

    lst = layers[1]
    assert {item["name"] for item in lst["variables"]} == {
        "Daytime LST",
        "Nighttime LST",
    }
    assert all(item["spatial_resolution"] == "1000m" for item in lst["variables"])
    et = layers[2]["variables"][0]
    pet = layers[3]["variables"][0]
    assert et["name"] == "Actual ET" and et["layer_id"] == "ET_500m"
    assert pet["name"] == "Potential ET" and pet["layer_id"] == "PET_500m"
    assert et["product_id"] == pet["product_id"] == "MOD16A2GF.061"
    assert et["scale_factor"] == pet["scale_factor"] == 0.1
    assert et["temporal_resolution"] == pet["temporal_resolution"] == "8 day"
    elevation = layers[5]["variables"][0]
    assert elevation["spatial_resolution"] == "30m"
    assert elevation["temporal_resolution"] == "Static"
    assert elevation["quality_layers"] == []
    assert elevation["auxiliary_layers"] == ["ASTER_GDEM_NUM"]
    assert elevation["qa_required"] is False
    assert layers[7]["variables"][0]["spatial_resolution"] == "500m"
    assert layers[7]["variables"][0]["temporal_resolution"] == "Yearly"

    soil_properties = layers[4]
    assert soil_properties["source"] == "ISRIC SoilGrids 2.0"
    assert {variable["id"] for variable in soil_properties["variables"]} == {
        "soil_ph",
        "soil_sand",
        "soil_silt",
        "soil_clay",
        "soil_organic_carbon",
    }
    assert all(
        variable["product_id"] == "ISRIC_SOILGRIDS"
        and variable["external_source"] == "ISRIC/SoilGrids"
        and variable["depth_interval"] == "0-5cm"
        and variable["availability_status"] == "available_on_demand"
        for variable in soil_properties["variables"]
    )
    slope = layers[6]["variables"][0]
    assert slope["derived"] is True
    assert slope["product_id"] == "_DERIVED"
    assert slope["layer_id"] == "slope"
    assert slope["requestable_from_saved_metadata"] is False
    assert slope["availability_status"] == "derived_when_source_is_processed"
    assert "SoilGrids" in layers[4]["source"]
    air_temperature_variables = {
        variable["layer_id"]: variable for variable in layers[8]["variables"]
    }
    air_temperature = air_temperature_variables["T2M"]
    precipitation = layers[9]["variables"][0]
    assert layers[8]["source"] == layers[9]["source"] == "NASA POWER (MERRA-2)"
    assert air_temperature["product_id"] == precipitation["product_id"] == "NASA_POWER_DAILY"
    assert air_temperature["layer_id"] == "T2M"
    assert {"T2M", "T2M_MIN", "T2M_MAX"} == set(air_temperature_variables)
    assert precipitation["layer_id"] == "PRECTOTCORR"
    assert air_temperature["units"] == "°C"
    assert precipitation["units"] == "mm"
    assert air_temperature["spatial_resolution"] == "0.5° × 0.625°"
    assert all(
        variable["availability_status"] == "available_on_demand"
        for variable in (*air_temperature_variables.values(), precipitation)
    )
    assert all(not item["required_for_suitability"] for item in layers)
    all_layer_ids = {
        variable.get("layer_id")
        for item in layers
        for variable in item["variables"]
    }
    assert "QC_Day" not in all_layer_ids
    assert "ET_QC_500m" not in all_layer_ids
    suitability_inputs = catalog["suitability_inputs"]
    assert suitability_inputs["map_visibility_changes_analysis_inputs"] is False
    assert suitability_inputs["crop_reference_scoring_criteria"] == [
        "depletion_fraction_p",
        "effective_root_depth_m",
        "kc_end",
        "kc_initial",
        "kc_mid",
        "season_length_days",
    ]
    assert suitability_inputs["matching_raster_scoring_criteria"] == []
    assert "not scientifically supported" in suitability_inputs["status"]


def test_suitability_api_does_not_score_unverified_crop_requirements() -> None:
    client = _client()

    crops = client.get("/api/crops")
    response = client.post(
        "/api/suitability/evaluate",
        json={"environment": {"ndvi": 0.8}, "crops": ["cotton"]},
    )

    assert crops.status_code == 200
    assert crops.json()["suitability_available"] is False
    assert response.status_code == 422
    assert "no verified environmental suitability criteria" in response.json()["detail"]


def test_essential_catalog_distinguishes_task_data_and_map_hides_quality_rasters(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    science_path = _register_raster_task(
        tmp_path, "task-a", "nasa-task-a", value=0.42
    )
    task_root = science_path.parents[2]
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
    ) as quality:
        quality.write(np.zeros((2, 2), dtype=np.uint8), 1)
        quality.update_tags(
            source_product_id=PRODUCT_ID,
            source_layer_id="_250m_16_days_VI_Quality",
        )
    task_manifest_path = task_root / "task_manifest.json"
    task_manifest = json.loads(task_manifest_path.read_text(encoding="utf-8"))
    task_manifest["layers"].append("_250m_16_days_VI_Quality")
    write_manifest(task_manifest_path, task_manifest)
    standardized_path = task_root / "processed/standardized_manifest.json"
    standardized = json.loads(standardized_path.read_text(encoding="utf-8"))
    standardized["records"].append(
        {
            "product_id": PRODUCT_ID,
            "layer_id": "_250m_16_days_VI_Quality",
            "variable_kind": "quality",
            "output_path": quality_path.relative_to(task_root).as_posix(),
            "output_units": None,
        }
    )
    write_manifest(standardized_path, standardized)
    client = _isolated_raster_app(tmp_path, monkeypatch)

    catalog_response = client.get(
        "/api/essential-layers",
        params={"task_name": "task-a"},
    )
    assert catalog_response.status_code == 200, catalog_response.text
    catalog = catalog_response.json()
    ndvi = next(item for item in catalog["layers"] if item["id"] == "ndvi")
    assert catalog["task_data_status"] == "processed"
    assert ndvi["processed_on_selected_task"] is True
    assert ndvi["variables"][0]["availability_status"] == "processed_on_task"
    assert ndvi["variables"][0]["quality_status"] == "missing_quality_companion"
    assert ndvi["variables"][0]["quality_filter_applied"] is False
    assert ndvi["variables"][0]["quality_missing_layers"] == [
        "_250m_16_days_VI_Quality"
    ]
    assert all(
        not item["processed_on_selected_task"]
        for item in catalog["layers"]
        if item["id"] != "ndvi"
    )

    map_layers = client.get(
        "/api/layers",
        params={"task_name": "task-a"},
    ).json()["layers"]
    assert [item["variable"] for item in map_layers] == [LAYER_ID]


def test_aoi_statistics_transform_geometry_and_reject_invalid_geometry() -> None:
    client = _client()
    aoi = {
        "type": "Polygon",
        "coordinates": [[
            [69.2797, 41.3111],
            [69.2807, 41.3111],
            [69.2807, 41.3121],
            [69.2797, 41.3121],
            [69.2797, 41.3111],
        ]],
    }

    response = client.post(
        "/api/aoi/statistics",
        json={
            "name": "Tashkent test",
            "geojson": aoi,
            "analysis_crs": "EPSG:32642",
            "resolution_m": 250,
        },
    )

    assert response.status_code == 200
    stats = response.json()
    assert stats["geometry_type"] == "Polygon"
    assert stats["area_km2"] > 0
    assert stats["estimated_grid_cells"] > 0
    assert stats["analysis_crs"] == "EPSG:32642"
    assert stats["bounds"] == {
        "west": 69.2797,
        "south": 41.3111,
        "east": 69.2807,
        "north": 41.3121,
    }
    second_aoi = {
        "type": "Polygon",
        "coordinates": [[
            [64.4200, 39.6500],
            [64.4210, 39.6500],
            [64.4210, 39.6510],
            [64.4200, 39.6510],
            [64.4200, 39.6500],
        ]],
    }
    second_stats = client.post(
        "/api/aoi/statistics",
        json={"geojson": second_aoi},
    ).json()
    assert second_stats["bounds"] == {
        "west": 64.42,
        "south": 39.65,
        "east": 64.421,
        "north": 39.651,
    }

    invalid = {
        "type": "Polygon",
        "coordinates": [[
            [69, 41],
            [70, 42],
            [69, 42],
            [70, 41],
            [69, 41],
        ]],
    }
    assert client.post(
        "/api/aoi/statistics",
        json={"geojson": invalid},
    ).status_code == 422

    antimeridian_aoi = {
        "type": "Polygon",
        "coordinates": [[
            [179, 10],
            [-179, 10],
            [-179, 11],
            [179, 11],
            [179, 10],
        ]],
    }
    antimeridian_response = client.post(
        "/api/aoi/statistics",
        json={"geojson": antimeridian_aoi},
    )
    assert antimeridian_response.status_code == 422
    assert "crossing the antimeridian are not supported" in antimeridian_response.json()["detail"]

    polar_aoi = {
        "type": "Polygon",
        "coordinates": [[
            [0, 88],
            [1, 88],
            [1, 89],
            [0, 89],
            [0, 88],
        ]],
    }
    polar_response = client.post(
        "/api/aoi/statistics",
        json={"geojson": polar_aoi},
    )
    assert polar_response.status_code == 422
    assert "outside the documented area of use" in polar_response.json()["detail"]


@pytest.mark.parametrize(
    ("longitude", "latitude"),
    [(-46.0, -23.0), (37.0, -1.0), (2.0, 48.0), (151.0, -33.0), (78.0, 22.0)],
)
def test_aoi_statistics_default_crs_supports_multiple_global_regions(
    longitude: float,
    latitude: float,
) -> None:
    delta = 0.01
    aoi = {
        "type": "Polygon",
        "coordinates": [[
            [longitude, latitude],
            [longitude + delta, latitude],
            [longitude + delta, latitude + delta],
            [longitude, latitude + delta],
            [longitude, latitude],
        ]],
    }

    response = _client().post(
        "/api/aoi/statistics",
        json={"name": "Global test area", "geojson": aoi, "resolution_m": 1000},
    )

    assert response.status_code == 200
    assert response.json()["analysis_crs"] == "EPSG:6933"
    assert response.json()["area_km2"] > 0
    assert response.json()["estimated_grid_cells"] > 0


def _register_raster_task(
    root: Path,
    task_name: str,
    task_id: str,
    *,
    value: float,
) -> Path:
    task_root = root / "data/tasks" / task_name
    registry = TaskRegistry(task_root.parent)
    registry.reserve(
        task_name,
        {"task_id": task_id, "task_directory": str(task_root)},
    )
    raster_path = task_root / "processed/rasters/MOD13Q1.061__250m_16_days_NDVI_20250626T000000_.tif"
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
    ) as dataset:
        dataset.write(np.full((2, 2), value, dtype=np.float32), 1)
        dataset.update_tags(
            source_product_id=PRODUCT_ID,
            source_layer_id=LAYER_ID,
        )
    aoi = {
        "name": task_name,
        "crs": "EPSG:4326",
        "geojson": {
            "type": "Polygon",
            "coordinates": [[[69.27, 41.29], [69.33, 41.29], [69.33, 41.35], [69.27, 41.35], [69.27, 41.29]]],
        },
    }
    write_manifest(
        task_root / "task_manifest.json",
        {
            "task_name": task_name,
            "task_id": task_id,
            "aoi": aoi,
            "product_id": PRODUCT_ID,
            "layers": [LAYER_ID],
            "preprocessing_status": "completed",
            "analysis_runs": [],
        },
    )
    write_manifest(task_root / "task_metadata.json", {"task_name": task_name, "task_id": task_id})
    write_manifest(
        task_root / "processed/standardized_manifest.json",
        {
            "task_name": task_name,
            "task_id": task_id,
            "created_at": "test",
            "source_directory": "raw",
            "records": [
                {
                    "product_id": PRODUCT_ID,
                    "layer_id": LAYER_ID,
                    "variable_kind": "continuous",
                    "quality_layers": ["_250m_16_days_VI_Quality"],
                    "quality_filter_applied": False,
                    "quality_status": "missing_quality_companion",
                    "quality_missing_layers": ["_250m_16_days_VI_Quality"],
                    "output_path": raster_path.relative_to(task_root).as_posix(),
                    "output_units": "NDVI",
                }
            ],
        },
    )
    return raster_path


def _isolated_raster_app(root: Path, monkeypatch: Any) -> TestClient:
    (root / "config").mkdir(parents=True, exist_ok=True)
    (root / "data/metadata").mkdir(parents=True, exist_ok=True)
    (root / "data/reference").mkdir(parents=True, exist_ok=True)
    shutil.copyfile(PROJECT_ROOT / "config/settings.yaml", root / "config/settings.yaml")
    shutil.copyfile(PROJECT_ROOT / "config/datasets.yaml", root / "config/datasets.yaml")
    shutil.copyfile(PROJECT_ROOT / "data/metadata/MOD13Q1.json", root / "data/metadata/MOD13Q1.json")
    shutil.copyfile(
        PROJECT_ROOT / "data/reference/crop_requirements.json",
        root / "data/reference/crop_requirements.json",
    )
    monkeypatch.setattr(web, "AppEEARSClient", lambda: FakeAppEEARSClient())
    return TestClient(web.create_app(project_root=root))


def test_processed_rasters_are_exposed_as_task_owned_tiles_and_values(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    raster_path = _register_raster_task(
        tmp_path, "task-a", "nasa-task-a", value=0.42
    )
    client = _isolated_raster_app(tmp_path, monkeypatch)
    response = client.get("/api/layers", params={"task_name": "task-a"})

    assert response.status_code == 200
    layers = response.json()["layers"]
    ndvi = next(layer for layer in layers if layer["variable"] == LAYER_ID)
    assert ndvi["observation_date"] == "2025-06-26"
    assert "path" not in ndvi

    tile = client.get(
        f"/api/layers/{ndvi['id']}/tiles/0/0/0.png",
        params={"task_name": "task-a"},
    )
    assert tile.status_code == 200
    assert tile.headers["content-type"] == "image/png"
    assert tile.content.startswith(b"\x89PNG\r\n\x1a\n")

    with rasterio.open(raster_path) as raster:
        x_coord, y_coord = raster.xy(0, 0)
        longitude, latitude = Transformer.from_crs(
            raster.crs, "EPSG:4326", always_xy=True
        ).transform(x_coord, y_coord)
    sample = client.get(
        f"/api/layers/{ndvi['id']}/value",
        params={"task_name": "task-a", "lat": latitude, "lon": longitude},
    )
    assert sample.status_code == 200
    assert sample.json()["value"] is not None


def test_non_default_soil_depth_is_listed_in_catalog_and_map(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    raster_path = _register_raster_task(
        tmp_path, "task-a", "nasa-task-a", value=7.2
    )
    task_root = raster_path.parents[2]
    soil_layer_id = "phh2o_5-15cm_Q0.5"
    with rasterio.open(raster_path, "r+") as raster:
        raster.update_tags(
            source_product_id="ISRIC_SOILGRIDS",
            source_layer_id=soil_layer_id,
        )

    task_manifest_path = task_root / "task_manifest.json"
    task_manifest = json.loads(task_manifest_path.read_text(encoding="utf-8"))
    task_manifest.update(
        product_id="ISRIC_SOILGRIDS",
        layers=[soil_layer_id],
    )
    write_manifest(task_manifest_path, task_manifest)

    standardized_path = task_root / "processed/standardized_manifest.json"
    standardized = json.loads(standardized_path.read_text(encoding="utf-8"))
    standardized["records"] = [
        {
            "product_id": "ISRIC_SOILGRIDS",
            "layer_id": soil_layer_id,
            "variable": "soil_ph",
            "depth_interval": "5-15cm",
            "variable_kind": "continuous",
            "output_path": raster_path.relative_to(task_root).as_posix(),
            "output_units": "pH",
        }
    ]
    write_manifest(standardized_path, standardized)
    client = _isolated_raster_app(tmp_path, monkeypatch)

    catalog = client.get(
        "/api/essential-layers", params={"task_name": "task-a"}
    ).json()
    soil = next(layer for layer in catalog["layers"] if layer["id"] == "soil_properties")
    ph = next(variable for variable in soil["variables"] if variable["id"] == "soil_ph")
    assert ph["layer_id"] == soil_layer_id
    assert ph["depth_interval"] == "5-15cm"
    assert ph["availability_status"] == "processed_on_task"

    map_layers = client.get(
        "/api/layers", params={"task_name": "task-a"}
    ).json()["layers"]
    assert [layer["variable"] for layer in map_layers] == [soil_layer_id]


def test_suitability_api_rejects_unverified_crop_requirements(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    _register_raster_task(tmp_path, "task-a", "nasa-task-a", value=0.42)
    client = _isolated_raster_app(tmp_path, monkeypatch)

    response = client.post(
        "/api/suitability/analyze",
        json={"task_name": "task-a"},
    )

    assert response.status_code == 422
    assert "no verified environmental suitability criteria" in response.json()["detail"]
    assert not any((tmp_path / "data/tasks/task-a/suitability/runs").iterdir())


def test_switching_api_tasks_cannot_return_stale_raster_layers(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    raster_a = _register_raster_task(tmp_path, "task-a", "nasa-task-a", value=0.25)
    _register_raster_task(tmp_path, "task-b", "nasa-task-b", value=0.75)
    client = _isolated_raster_app(tmp_path, monkeypatch)

    layers_a = client.get("/api/layers", params={"task_name": "task-a"}).json()["layers"]
    layers_b = client.get("/api/layers", params={"task_name": "task-b"}).json()["layers"]
    id_a = next(layer["id"] for layer in layers_a if layer["variable"] == LAYER_ID)
    id_b = next(layer["id"] for layer in layers_b if layer["variable"] == LAYER_ID)
    assert id_a != id_b
    assert client.get(
        f"/api/layers/{id_a}",
        params={"task_name": "task-b"},
    ).status_code == 404

    with rasterio.open(raster_a) as source:
        longitude, latitude = source.xy(0, 0)
    sample = client.get(
        f"/api/layers/{id_b}/value",
        params={"task_name": "task-b", "lat": latitude, "lon": longitude},
    )
    assert sample.status_code == 200
    assert sample.json()["value"] == 0.75


class FakeAppEEARSClient:
    base_url = "https://appeears.earthdatacloud.nasa.gov/api"

    def __init__(self) -> None:
        self.submitted_request: dict[str, Any] | None = None
        self.remote_tasks: list[dict[str, Any]] = []

    def get_json(
        self,
        endpoint: str,
        *,
        params: dict[str, Any] | None = None,
    ) -> Any:
        del params
        if endpoint == "task":
            return self.remote_tasks
        raise AssertionError(f"Unexpected endpoint: {endpoint}")

    def post_json(
        self,
        endpoint: str,
        payload: dict[str, Any],
    ) -> tuple[dict[str, str], Any]:
        assert endpoint == "task"
        self.submitted_request = payload
        response = type("Response", (), {"headers": {}})()
        return {"task_id": "web-test-task", "status": "pending"}, response


def _isolated_web_app(tmp_path: Path, monkeypatch: Any) -> TestClient:
    root = tmp_path
    (root / "config").mkdir()
    (root / "data/metadata").mkdir(parents=True)
    shutil.copyfile(PROJECT_ROOT / "config/settings.yaml", root / "config/settings.yaml")
    shutil.copyfile(PROJECT_ROOT / "config/datasets.yaml", root / "config/datasets.yaml")
    shutil.copyfile(
        PROJECT_ROOT / "data/metadata/MOD13Q1.json",
        root / "data/metadata/MOD13Q1.json",
    )
    shutil.copyfile(
        PROJECT_ROOT / "data/metadata/ASTER.json",
        root / "data/metadata/ASTER.json",
    )
    fake_client = FakeAppEEARSClient()
    monkeypatch.setattr(web, "AppEEARSClient", lambda: fake_client)
    app = web.create_app(project_root=root)
    app.state.fake_appeears_client = fake_client
    return TestClient(app)


def test_task_list_returns_existing_tasks_and_warns_about_missing_registry_directories(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    client = _isolated_web_app(tmp_path, monkeypatch)
    registry = TaskRegistry(tmp_path / "data/tasks")
    task_id = "valid-task-id"
    task_root = registry.reserve(
        "valid-task",
        {
            "task_id": task_id,
            "task_type": "aoi_analysis",
            "task_status": "completed",
            "download_status": "completed",
            "created_at": "2026-11-14T00:00:00+00:00",
            "task_directory": str(tmp_path / "data/tasks/valid-task"),
        },
    )
    write_manifest(
        task_root / "task_manifest.json",
        {
            "task_name": "valid-task",
            "task_id": task_id,
            "task_type": "aoi_analysis",
            "aoi": {"name": "Valid AOI"},
            "task_status": "completed",
            "download_status": "completed",
            "submitted_at": "2026-11-14T00:00:00+00:00",
            "downloads": [],
            "analysis_runs": [],
        },
    )
    write_manifest(
        registry.path,
        {
            "tasks": [
                *registry.records(),
                {
                    "task_name": "missing-task",
                    "task_id": "missing-task-id",
                    "task_type": "aoi_analysis",
                    "task_status": "failed",
                    "download_status": "failed",
                    "created_at": "2026-11-13T00:00:00+00:00",
                },
            ]
        },
    )

    response = client.get("/api/tasks")

    assert response.status_code == 200
    result = response.json()
    assert [task["task_name"] for task in result["tasks"]] == ["valid-task"]
    assert len(result["warnings"]) == 1
    assert "missing-task" in result["warnings"][0]


def test_task_submission_requires_confirmation_and_reuses_task_manager(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    client = _isolated_web_app(tmp_path, monkeypatch)
    request = {
        "aoi_name": "Web AOI",
        "task_name": "web-test",
        "aoi_geojson": {
            "type": "Polygon",
            "coordinates": [[
                [69.2797, 41.3111],
                [69.2807, 41.3111],
                [69.2807, 41.3121],
                [69.2797, 41.3121],
                [69.2797, 41.3111],
            ]],
        },
        "start_date": "2025-07-01",
        "end_date": "2025-07-16",
        "analysis_crs": "EPSG:32642",
        "resolution_m": 250,
        "products": [{"product_id": PRODUCT_ID, "layer_ids": [LAYER_ID]}],
    }

    assert client.post("/api/tasks", json=request).status_code == 400
    excessive_grid = client.post(
        "/api/tasks",
        json={
            **request,
            "resolution_m": 0.1,
            "confirm_real_submission": True,
        },
    )
    assert excessive_grid.status_code == 422
    response = client.post(
        "/api/tasks",
        json={**request, "confirm_real_submission": True},
    )

    assert response.status_code == 200
    result = response.json()
    assert result["submitted"][0]["task_id"] == "web-test-task"
    assert result["failures"] == []
    app_task = client.get("/api/tasks").json()["tasks"][0]
    assert app_task["product_id"] == PRODUCT_ID
    manifest_path = tmp_path / "data/tasks/web-test/task_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["aoi"]["name"] == "Web AOI"
    assert manifest["layers"] == [LAYER_ID]
    assert manifest["spatial_analysis"] == {
        "analysis_crs": "EPSG:32642",
        "requested_cell_size_m": 250,
    }

    duplicate = client.post(
        "/api/tasks",
        json={**request, "confirm_real_submission": True},
    )
    assert duplicate.status_code == 409
    assert len(list((tmp_path / "data/tasks").glob("*/task_manifest.json"))) == 1


def test_static_elevation_uses_its_available_period_not_analysis_dates(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    client = _isolated_web_app(tmp_path, monkeypatch)
    request = {
        "aoi_name": "Web AOI",
        "task_name": "static-elevation",
        "aoi_geojson": {
            "type": "Polygon",
            "coordinates": [[
                [69.2797, 41.3111],
                [69.2807, 41.3111],
                [69.2807, 41.3121],
                [69.2797, 41.3121],
                [69.2797, 41.3111],
            ]],
        },
        "start_date": "2025-07-01",
        "end_date": "2025-07-16",
        "analysis_crs": "EPSG:32642",
        "resolution_m": 250,
        "products": [
            {"product_id": "ASTGTM_NC.003", "layer_ids": ["ASTER_GDEM_DEM"]}
        ],
        "confirm_real_submission": True,
    }

    response = client.post("/api/tasks", json=request)

    assert response.status_code == 200, response.text
    payload = client.app.state.fake_appeears_client.submitted_request
    assert payload is not None
    assert payload["params"]["dates"] == [
        {
            "startDate": "03-01-2000",
            "endDate": "11-30-2013",
            "recurring": False,
        }
    ]


def test_aoi_download_all_stores_standardized_child_rasters_on_analysis_server(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    client = _isolated_web_app(tmp_path, monkeypatch)
    registry = TaskRegistry(tmp_path / "data/tasks")
    parent_root = registry.reserve(
        "aoi-parent",
        {
            "task_id": "parent-task-id",
            "task_directory": str(tmp_path / "data/tasks/aoi-parent"),
        },
    )
    child_root = registry.reserve(
        "appeears-child",
        {
            "task_id": "child-task-id",
            "task_directory": str(tmp_path / "data/tasks/appeears-child"),
        },
    )
    parent_manifest = {
        "task_name": "aoi-parent",
        "task_id": "parent-task-id",
        "task_type": "aoi_analysis",
        "appeears_tasks": [
            {"task_name": "appeears-child", "task_id": "child-task-id"}
        ],
    }
    child_manifest = {
        "task_name": "appeears-child",
        "task_id": "child-task-id",
        "task_type": "appeears",
        "product_id": PRODUCT_ID,
        "layers": [LAYER_ID],
        "task_status": "completed",
        "download_status": "completed",
        "preprocessing_status": "pending",
        "parent_aoi_task": {
            "task_name": "aoi-parent",
            "task_id": "parent-task-id",
        },
    }
    write_manifest(parent_root / "task_manifest.json", parent_manifest)
    write_manifest(child_root / "task_manifest.json", child_manifest)
    write_manifest(
        parent_root / "processed/standardized_manifest.json",
        {
            "task_name": "aoi-parent",
            "task_id": "parent-task-id",
            "created_at": "test",
            "source_directory": str(parent_root / "raw"),
            "rasters": [],
            "records": [],
        },
    )
    child_raster_path = child_root / "processed/rasters/ndvi.tif"
    with rasterio.open(
        child_raster_path,
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
        raster.write(np.full((2, 2), 0.4, dtype=np.float32), 1)
        raster.update_tags(
            source_product_id=PRODUCT_ID,
            source_layer_id=LAYER_ID,
        )

    download_calls: list[str] = []
    class FakeDownloadManager:
        def __init__(self, _manager: Any) -> None:
            pass

        def download_task(self, _task_id: str) -> dict[str, Any]:
            download_calls.append(_task_id)
            return {}

    monkeypatch.setattr(web, "DownloadManager", FakeDownloadManager)

    def fake_standardize_task(
        task_directory: Path,
        *,
        task_name: str,
        task_id: str,
        metadata_directory: Path,
    ) -> dict[str, Any]:
        del metadata_directory
        assert task_directory == child_root
        record = {
            "product_id": PRODUCT_ID,
            "layer_id": LAYER_ID,
            "output_path": "processed/rasters/ndvi.tif",
            "variable_kind": "continuous",
            "output_units": "NDVI",
        }
        write_manifest(
            child_root / "processed/standardized_manifest.json",
            {
                "task_name": task_name,
                "task_id": task_id,
                "created_at": "test",
                "source_directory": str(child_root / "raw"),
                "rasters": [record],
                "records": [record],
            },
        )
        return {"rasters": [record]}

    monkeypatch.setattr(web, "standardize_task", fake_standardize_task)
    response = client.post("/api/aoi-tasks/parent-task-id/download")
    assert response.status_code == 200
    result = response.json()
    assert result["status"] == "ready"
    assert result["newly_integrated_raster_count"] == 1
    assert download_calls == ["child-task-id"]
    imported_raster = (
        parent_root / "processed/rasters/appeears/appeears-child/ndvi.tif"
    )
    assert imported_raster.is_file()
    parent_raster_manifest = json.loads(
        (parent_root / "processed/standardized_manifest.json").read_text(
            encoding="utf-8"
        )
    )
    imported_record = parent_raster_manifest["rasters"][0]
    assert imported_record["source_task_id"] == "child-task-id"
    assert imported_record["task_id"] == "parent-task-id"
    assert client.get(
        "/api/layers", params={"task_name": "aoi-parent"}
    ).json()["layers"][0]["variable"] == LAYER_ID
    assert client.get("/api/tasks/parent-task-id").json()["analysis_data"]["status"] == "ready"

    repeated = client.post("/api/aoi-tasks/parent-task-id/download")
    assert repeated.status_code == 200
    assert repeated.json()["newly_integrated_raster_count"] == 0
    assert download_calls == ["child-task-id", "child-task-id"]

    child_manifest["parent_aoi_task"] = {
        "task_name": "some-other-parent",
        "task_id": "other-parent-id",
    }
    write_manifest(child_root / "task_manifest.json", child_manifest)
    isolated_response = client.post("/api/aoi-tasks/parent-task-id/download")
    assert isolated_response.status_code == 422


def _aoi_task_request(**updates: Any) -> dict[str, Any]:
    return {
        "aoi_name": "New AOI",
        "task_name": "new-aoi-task",
        "aoi_geojson": {
            "type": "Polygon",
            "coordinates": [[
                [69.2797, 41.3111],
                [69.2807, 41.3111],
                [69.2807, 41.3121],
                [69.2797, 41.3121],
                [69.2797, 41.3111],
            ]],
        },
        "start_date": "2024-03-01",
        "end_date": "2024-03-03",
        "analysis_crs": "EPSG:6933",
        "resolution_m": 250,
        "soil_properties": True,
        "soil_depth": "5-15cm",
        "nasa_power_temperature": True,
        "nasa_power_precipitation": True,
        "products": [],
        "confirm_acquisition": True,
        **updates,
    }


def test_new_aoi_acquisition_creates_local_task_without_existing_tasks(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    client = _isolated_web_app(tmp_path, monkeypatch)
    observed: dict[str, Any] = {}

    def collect_soil(registry: TaskRegistry, identifier: str, *, depth: str) -> dict[str, Any]:
        task, _, manifest = registry.load_task_manifest(identifier)
        observed["soil"] = (task["task_name"], depth, manifest["aoi"]["geojson"])
        return {"new_variables": ["soil_ph"], "already_integrated": []}

    def collect_power(
        registry: TaskRegistry,
        identifier: str,
        *,
        variables: tuple[str, ...],
    ) -> dict[str, Any]:
        task, _, manifest = registry.load_task_manifest(identifier)
        observed["power"] = (task["task_name"], variables, manifest["aoi"]["geojson"])
        return {"new_variables": list(variables), "already_integrated": []}

    monkeypatch.setattr(web, "collect_soilgrids", collect_soil)
    monkeypatch.setattr(web, "collect_power_climate", collect_power)

    response = client.post("/api/aoi-tasks", json=_aoi_task_request())

    assert response.status_code == 202
    result = response.json()
    task = client.get("/api/tasks").json()["tasks"][0]
    assert task["task_name"] == result["task_name"] == "new-aoi-task"
    assert task["task_id"] == result["task_id"]
    assert task["acquisition"]["status"] == "completed"
    assert task["acquisition"]["providers"]["soilgrids"]["status"] == "completed"
    assert task["acquisition"]["providers"]["nasa_power"]["status"] == "completed"
    assert observed["soil"][0:2] == ("new-aoi-task", "5-15cm")
    assert observed["power"][0:2] == (
        "new-aoi-task",
        ("T2M", "T2M_MIN", "T2M_MAX", "PRECTOTCORR"),
    )
    assert observed["soil"][2] == observed["power"][2]

    task_root = tmp_path / "data/tasks/new-aoi-task"
    manifest = json.loads((task_root / "task_manifest.json").read_text(encoding="utf-8"))
    submitted_geometry = _aoi_task_request()["aoi_geojson"]
    expected_geometry = {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "properties": {},
                "geometry": submitted_geometry,
            }
        ],
    }
    assert result["aoi"]["crs"] == "EPSG:4326"
    assert result["aoi"]["geojson"] == manifest["aoi"]["geojson"] == expected_geometry
    assert observed["soil"][2] == observed["power"][2] == expected_geometry
    raster_manifest = json.loads(
        (task_root / "processed/standardized_manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["aoi"]["name"] == "New AOI"
    assert manifest["selected_datasets"]["soil_depth"] == "5-15cm"
    assert manifest["spatial_analysis"]["requested_cell_size_m"] == 250
    assert raster_manifest["task_id"] == result["task_id"]
    assert client.get(
        "/api/layers", params={"task_name": "new-aoi-task"}
    ).json()["layers"] == []


def test_new_aoi_tasks_keep_distinct_submitted_geometries_isolated(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    client = _isolated_web_app(tmp_path, monkeypatch)
    observed: dict[str, Any] = {}

    def collect_soil(
        registry: TaskRegistry,
        identifier: str,
        *,
        depth: str,
    ) -> dict[str, Any]:
        task, _, manifest = registry.load_task_manifest(identifier)
        observed[task["task_name"]] = manifest["aoi"]["geojson"]
        return {"new_variables": [depth], "already_integrated": []}

    def collect_power(
        registry: TaskRegistry,
        identifier: str,
        *,
        variables: tuple[str, ...],
    ) -> dict[str, Any]:
        del variables
        task, _, manifest = registry.load_task_manifest(identifier)
        observed[task["task_name"]] = manifest["aoi"]["geojson"]
        return {"new_variables": ["T2M"], "already_integrated": []}

    monkeypatch.setattr(web, "collect_soilgrids", collect_soil)
    monkeypatch.setattr(web, "collect_power_climate", collect_power)
    first_request = _aoi_task_request(task_name="first-area")
    second_geometry = {
        "type": "Polygon",
        "coordinates": [[
            [64.4200, 39.6500],
            [64.4210, 39.6500],
            [64.4210, 39.6510],
            [64.4200, 39.6510],
            [64.4200, 39.6500],
        ]],
    }
    second_request = _aoi_task_request(
        aoi_name="Second area",
        task_name="second-area",
        aoi_geojson=second_geometry,
    )

    first_response = client.post("/api/aoi-tasks", json=first_request)
    second_response = client.post("/api/aoi-tasks", json=second_request)

    assert first_response.status_code == second_response.status_code == 202
    first_expected = {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "properties": {},
                "geometry": first_request["aoi_geojson"],
            }
        ],
    }
    second_expected = {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "properties": {},
                "geometry": second_geometry,
            }
        ],
    }
    first_manifest_path = tmp_path / "data/tasks/first-area/task_manifest.json"
    first_manifest_before = json.loads(first_manifest_path.read_text(encoding="utf-8"))
    second_manifest = json.loads(
        (tmp_path / "data/tasks/second-area/task_manifest.json").read_text(
            encoding="utf-8"
        )
    )
    first_manifest_after = json.loads(first_manifest_path.read_text(encoding="utf-8"))

    assert first_response.json()["aoi"]["geojson"] == first_expected
    assert second_response.json()["aoi"]["geojson"] == second_expected
    assert first_manifest_before["aoi"]["geojson"] == first_expected
    assert first_manifest_after["aoi"]["geojson"] == first_expected
    assert second_manifest["aoi"]["geojson"] == second_expected
    assert observed["first-area"] == first_expected
    assert observed["second-area"] == second_expected


def test_new_aoi_provider_failure_does_not_cancel_other_provider(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    client = _isolated_web_app(tmp_path, monkeypatch)

    def fail_soil(*args: Any, **kwargs: Any) -> dict[str, Any]:
        raise requests.RequestException("ISRIC is unavailable")

    def collect_power(
        registry: TaskRegistry,
        identifier: str,
        *,
        variables: tuple[str, ...],
    ) -> dict[str, Any]:
        del registry, identifier
        return {"new_variables": list(variables), "already_integrated": []}

    monkeypatch.setattr(web, "collect_soilgrids", fail_soil)
    monkeypatch.setattr(web, "collect_power_climate", collect_power)

    response = client.post("/api/aoi-tasks", json=_aoi_task_request())

    assert response.status_code == 202
    task = client.get("/api/tasks").json()["tasks"][0]
    assert task["acquisition"]["status"] == "partial"
    assert task["acquisition"]["providers"]["soilgrids"]["status"] == "failed"
    assert "ISRIC is unavailable" in task["acquisition"]["providers"]["soilgrids"]["error"]
    assert task["acquisition"]["providers"]["nasa_power"]["status"] == "completed"
    assert task["acquisition"]["providers"]["nasa_power"]["variables"] == [
        "T2M",
        "T2M_MIN",
        "T2M_MAX",
        "PRECTOTCORR",
    ]
    assert (tmp_path / "data/tasks/new-aoi-task/task_manifest.json").is_file()


def test_retry_retries_only_failed_provider_without_repeating_successful_provider(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    client = _isolated_web_app(tmp_path, monkeypatch)
    soil_calls = 0
    power_calls = 0

    def collect_soil(
        registry: TaskRegistry,
        identifier: str,
        *,
        depth: str,
    ) -> dict[str, Any]:
        nonlocal soil_calls
        soil_calls += 1
        if soil_calls == 1:
            raise requests.RequestException("ISRIC is temporarily unavailable")
        return {"new_variables": [depth], "already_integrated": []}

    def collect_power(
        registry: TaskRegistry,
        identifier: str,
        *,
        variables: tuple[str, ...],
    ) -> dict[str, Any]:
        nonlocal power_calls
        power_calls += 1
        return {"new_variables": list(variables), "already_integrated": []}

    monkeypatch.setattr(web, "collect_soilgrids", collect_soil)
    monkeypatch.setattr(web, "collect_power_climate", collect_power)

    created = client.post("/api/aoi-tasks", json=_aoi_task_request())
    assert created.status_code == 202
    first_state = client.get("/api/tasks").json()["tasks"]
    assert len(first_state) == 1
    assert first_state[0]["acquisition"]["status"] == "partial"

    retried = client.post(
        f"/api/aoi-tasks/{created.json()['task_id']}/retry",
    )

    assert retried.status_code == 202
    final_state = client.get("/api/tasks").json()["tasks"]
    assert len(final_state) == 1
    assert final_state[0]["acquisition"]["status"] == "completed"
    assert final_state[0]["acquisition"]["providers"]["soilgrids"]["status"] == "completed"
    assert final_state[0]["acquisition"]["providers"]["nasa_power"]["status"] == "completed"
    assert soil_calls == 2
    assert power_calls == 1


def test_new_aoi_app_eeears_uses_the_same_aoi_configuration(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    client = _isolated_web_app(tmp_path, monkeypatch)
    fake_client = client.app.state.fake_appeears_client

    def collect_soil(
        registry: TaskRegistry,
        identifier: str,
        *,
        depth: str,
    ) -> dict[str, Any]:
        task, _, manifest = registry.load_task_manifest(identifier)
        assert task["task_name"] == "new-aoi-task"
        assert manifest["aoi"]["geojson"]["features"]
        return {"new_variables": [depth], "already_integrated": []}

    def collect_power(
        registry: TaskRegistry,
        identifier: str,
        *,
        variables: tuple[str, ...],
    ) -> dict[str, Any]:
        task, _, manifest = registry.load_task_manifest(identifier)
        assert task["task_name"] == "new-aoi-task"
        assert manifest["aoi"]["geojson"]["features"]
        return {"new_variables": list(variables), "already_integrated": []}

    monkeypatch.setattr(web, "collect_soilgrids", collect_soil)
    monkeypatch.setattr(web, "collect_power_climate", collect_power)

    response = client.post(
        "/api/aoi-tasks",
        json=_aoi_task_request(
            products=[{"product_id": PRODUCT_ID, "layer_ids": [LAYER_ID]}],
        ),
    )

    result = response.json()
    task_manifest = json.loads(
        (
            tmp_path
            / "data/tasks/new-aoi-task/task_manifest.json"
        ).read_text(encoding="utf-8")
    )
    assert fake_client.submitted_request is not None
    assert result["aoi"] == task_manifest["aoi"]
    assert fake_client.submitted_request["params"]["geo"] == task_manifest["aoi"]["geojson"]
    assert fake_client.submitted_request["params"]["dates"][0]["startDate"] == "03-01-2024"
    assert task_manifest["acquisition"]["status"] == "running"
    assert task_manifest["acquisition"]["providers"]["appeears"]["status"] == "submitted"
    assert task_manifest["acquisition"]["providers"]["soilgrids"]["status"] == "completed"
    assert task_manifest["acquisition"]["providers"]["nasa_power"]["status"] == "completed"
    assert task_manifest["appeears_tasks"][0]["task_id"] == "web-test-task"
    assert task_manifest["acquisition"]["providers"]["appeears"]["task_ids"] == [
        "web-test-task"
    ]
    assert task_manifest["acquisition"]["providers"]["appeears"]["operations"][0][
        "task_id"
    ] == "web-test-task"
    child_manifest = json.loads(
        (
            tmp_path
            / f"data/tasks/{task_manifest['appeears_tasks'][0]['task_name']}/task_manifest.json"
        ).read_text(encoding="utf-8")
    )
    assert child_manifest["parent_aoi_task"] == {
        "task_name": "new-aoi-task",
        "task_id": response.json()["task_id"],
    }
    assert client.app.state.fake_appeears_client.remote_tasks == []
    visible_tasks = client.get("/api/tasks").json()["tasks"]
    assert len(visible_tasks) == 1
    assert visible_tasks[0]["task_type"] == "aoi_analysis"
    assert visible_tasks[0]["task_id"] == response.json()["task_id"]
    client.app.state.fake_appeears_client.remote_tasks = [
        {
            "task_id": "web-test-task",
            "status": "pending",
            "params": fake_client.submitted_request["params"],
        }
    ]
    visible_tasks = client.get("/api/tasks?include_remote=true").json()["tasks"]
    assert len(visible_tasks) == 1
    assert visible_tasks[0]["task_type"] == "aoi_analysis"
    assert visible_tasks[0]["task_id"] == response.json()["task_id"]
