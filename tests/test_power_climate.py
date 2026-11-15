from __future__ import annotations

from pathlib import Path
from datetime import date
from typing import Any

import numpy as np
import pytest
import rasterio

from nasa_agri_intelligence.collection.manifest import write_manifest
from nasa_agri_intelligence.collection.power_climate import collect_power_climate
from nasa_agri_intelligence.collection.power_climate import (
    _monthly_periods,
    _seasonal_periods,
)
from nasa_agri_intelligence.collection.task_registry import TaskRegistry


def _create_task(tmp_path: Path) -> tuple[TaskRegistry, Path]:
    task_name = "climate-task"
    task_id = "nasa-climate-task"
    product_id = "MOD13Q1.061"
    layer_id = "_250m_16_days_NDVI"
    registry = TaskRegistry(tmp_path / "data/tasks")
    task_root = registry.reserve(
        task_name,
        {"task_id": task_id, "task_directory": str(tmp_path / "data/tasks" / task_name)},
    )
    aoi = {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "properties": {},
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [[
                        [69.27, 41.30],
                        [69.29, 41.30],
                        [69.29, 41.32],
                        [69.27, 41.32],
                        [69.27, 41.30],
                    ]],
                },
            }
        ],
    }
    write_manifest(
        task_root / "task_manifest.json",
        {
            "task_name": task_name,
            "task_id": task_id,
            "product_id": product_id,
            "layers": [layer_id],
            "aoi": {"name": task_name, "crs": "EPSG:4326", "geojson": aoi},
            "start_date": "2024-07-01",
            "end_date": "2024-07-02",
            "download_status": "completed",
            "preprocessing_status": "completed",
        },
    )
    raster_path = task_root / "processed/rasters/ndvi.tif"
    raster_path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(
        raster_path,
        "w",
        driver="GTiff",
        width=2,
        height=2,
        count=1,
        dtype="float32",
        crs="EPSG:4326",
        transform=rasterio.transform.from_origin(69.3, 41.33, 0.01, 0.01),
        nodata=float("nan"),
    ) as raster:
        raster.write(np.full((2, 2), 0.4, dtype=np.float32), 1)
        raster.update_tags(source_product_id=product_id, source_layer_id=layer_id)
    (task_root / "raw").mkdir(parents=True, exist_ok=True)
    write_manifest(
        task_root / "processed/standardized_manifest.json",
        {
            "task_name": task_name,
            "task_id": task_id,
            "created_at": "2026-11-12T00:00:00+00:00",
            "source_directory": "raw",
            "rasters": [
                {
                    "product_id": product_id,
                    "layer_id": layer_id,
                    "output_path": "processed/rasters/ndvi.tif",
                    "variable_kind": "continuous",
                }
            ],
        },
    )
    return registry, task_root


class _PowerResponse:
    def __init__(self, layer_id: str) -> None:
        self.layer_id = layer_id

    def raise_for_status(self) -> None:
        return None

    def json(self) -> dict[str, Any]:
        longitudes = [69.375, 70.0, 70.625]
        latitudes = [40.5, 41.0, 41.5]
        features = []
        for row, latitude in enumerate(latitudes):
            for column, longitude in enumerate(longitudes):
                if self.layer_id == "T2M":
                    values = {"20240701": 27 + row + column, "20240702": 29 + row + column}
                elif self.layer_id == "T2M_MIN":
                    values = {"20240701": 12 + row + column, "20240702": 14 + row + column}
                elif self.layer_id == "T2M_MAX":
                    values = {"20240701": 32 + row + column, "20240702": 34 + row + column}
                else:
                    values = {"20240701": 0.1 * (row + column + 1), "20240702": 0.2 * (row + column + 1)}
                if row == 0 and column == 0:
                    values = {"20240701": -999, "20240702": -999}
                features.append(
                    {
                        "type": "Feature",
                        "geometry": {"type": "Point", "coordinates": [longitude, latitude, 0]},
                        "properties": {"parameter": {self.layer_id: values}},
                    }
                )
        return {"features": features}


def test_real_climate_provider_records_task_scoped_daily_rasters(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    registry, task_root = _create_task(tmp_path)
    calls: list[dict[str, Any]] = []

    def fake_get(url: str, *, params: dict[str, Any], timeout: float) -> _PowerResponse:
        calls.append({"url": url, "params": params, "timeout": timeout})
        return _PowerResponse(str(params["parameters"]))

    monkeypatch.setattr(
        "nasa_agri_intelligence.collection.power_climate.requests.get",
        fake_get,
    )

    result = collect_power_climate(registry, "climate-task")

    assert result["new_variables"] == [
        "air_temperature",
        "air_temperature_min",
        "air_temperature_max",
        "precipitation",
    ]
    assert len(calls) == 4


    assert all(call["params"]["start"] == "20240701" for call in calls)
    assert all(call["params"]["end"] == "20240702" for call in calls)
    manifest_path = task_root / "processed/standardized_manifest.json"
    _, _, records = registry.validate_raster_manifest("climate-task", manifest_path)
    climate = {
        record["layer_id"]: record
        for record in records
        if record["product_id"] == "NASA_POWER_DAILY"
    }
    assert {"T2M", "T2M_MIN", "T2M_MAX", "PRECTOTCORR"} <= set(climate)
    assert "T2M_202407_MONTHLY" in climate
    assert "PRECTOTCORR_202407_MONTHLY" in climate
    assert climate["T2M"]["data_scope"] == "task_scoped"
    assert climate["PRECTOTCORR"]["output_units"] == "mm"
    assert climate["PRECTOTCORR"]["temporal_aggregation"] == (
        "sum of valid daily precipitation amounts"
    )
    with rasterio.open(climate["T2M"]["output_path"]) as raster:
        data = raster.read(1, masked=True)
        assert raster.count == 1
        assert raster.crs.to_epsg() == 4326
        assert data[1, 1] == pytest.approx(30.0)
        assert np.ma.getmaskarray(data)[-1, 0]
    with rasterio.open(climate["T2M_MIN"]["output_path"]) as raster:
        assert raster.read(1)[1, 1] == pytest.approx(15.0)
    with rasterio.open(climate["T2M_MAX"]["output_path"]) as raster:
        assert raster.read(1)[1, 1] == pytest.approx(35.0)
    with rasterio.open(climate["PRECTOTCORR"]["output_path"]) as raster:
        assert raster.read(1)[1, 1] == pytest.approx(0.9)
    with rasterio.open(climate["PRECTOTCORR_202407_MONTHLY"]["output_path"]) as raster:
        assert raster.read(1)[1, 1] == pytest.approx(0.9)

    repeated = collect_power_climate(registry, "climate-task")

    assert repeated["new_variables"] == []
    assert repeated["already_integrated"] == [
        "PRECTOTCORR",
        "T2M",
        "T2M_MAX",
        "T2M_MIN",
    ]
    assert len(calls) == 4


def test_power_periods_include_monthly_statistics_and_complete_seasons() -> None:
    monthly = _monthly_periods("PRECTOTCORR", date(2024, 1, 15), date(2024, 3, 10))
    assert [period["layer_id"] for period in monthly] == [
        "PRECTOTCORR_202401_MONTHLY",
        "PRECTOTCORR_202402_MONTHLY",
        "PRECTOTCORR_202403_MONTHLY",
    ]
    assert monthly[0]["start_date"] == date(2024, 1, 15)
    assert monthly[0]["end_date"] == date(2024, 1, 31)
    assert monthly[1]["temporal_aggregation"].startswith("monthly sum")

    seasons = _seasonal_periods(
        "PRECTOTCORR", date(2024, 1, 1), date(2024, 12, 31)
    )
    assert [period["layer_id"] for period in seasons] == [
        "PRECTOTCORR_2024_MAM",
        "PRECTOTCORR_2024_JJA",
        "PRECTOTCORR_2024_SON",
    ]
    assert all(
        period["temporal_aggregation"].startswith("seasonal sum")
        for period in seasons
    )


def test_power_provider_only_fetches_requested_variables(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    registry, task_root = _create_task(tmp_path)
    calls: list[str] = []

    def fake_get(url: str, *, params: dict[str, Any], timeout: float) -> _PowerResponse:
        del url, timeout
        calls.append(str(params["parameters"]))
        return _PowerResponse(str(params["parameters"]))

    monkeypatch.setattr(
        "nasa_agri_intelligence.collection.power_climate.requests.get",
        fake_get,
    )

    result = collect_power_climate(
        registry,
        "climate-task",
        variables=("PRECTOTCORR",),
    )

    assert calls == ["PRECTOTCORR"]
    assert result["new_variables"] == ["precipitation"]
    _, _, records = registry.validate_raster_manifest(
        "climate-task",
        task_root / "processed/standardized_manifest.json",
    )
    power_layers = {
        record["layer_id"]
        for record in records
        if record.get("external_source") == "NASA/POWER"
    }
    assert power_layers
    assert all(layer.startswith("PRECTOTCORR") for layer in power_layers)
