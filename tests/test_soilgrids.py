from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import rasterio
import requests
from pyproj import CRS, Transformer
from shapely.geometry import box
from rasterio.io import MemoryFile
from rasterio.transform import from_origin

from nasa_agri_intelligence.collection.manifest import write_manifest
from nasa_agri_intelligence.collection.soilgrids import (
    SOILGRIDS_GEOREFERENCING_VERSION,
    SOILGRIDS_PROJECTION,
    SOILGRIDS_DEPTHS,
    collect_soilgrids,
)
from nasa_agri_intelligence.collection.task_registry import TaskRegistry


_COVERAGE_VALUES = {
    "phh2o": 72,
    "sand": 250,
    "silt": 450,
    "clay": 300,
    "soc": 20,
}


def _create_task(tmp_path: Path) -> tuple[TaskRegistry, Path]:
    task_name = "soil-task"
    task_id = "nasa-soil-task"
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
    with rasterio.open(
        raster_path,
        "w",
        driver="GTiff",
        width=2,
        height=2,
        count=1,
        dtype="float32",
        crs="EPSG:4326",
        transform=from_origin(69.3, 41.33, 0.01, 0.01),
        nodata=float("nan"),
    ) as raster:
        raster.write(np.full((2, 2), 0.4, dtype=np.float32), 1)
        raster.update_tags(source_product_id=product_id, source_layer_id=layer_id)
    write_manifest(
        task_root / "processed/standardized_manifest.json",
        {
            "task_name": task_name,
            "task_id": task_id,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "source_directory": str(task_root / "raw"),
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


def _geotiff_bytes(property_id: str) -> bytes:
    aoi_bounds = box(69.27, 41.30, 69.29, 41.32).bounds
    transformed_bounds = Transformer.from_crs(
        "EPSG:4326",
        CRS.from_proj4(SOILGRIDS_PROJECTION),
        always_xy=True,
    ).transform_bounds(*aoi_bounds, densify_pts=21)
    memory_file = MemoryFile()
    with memory_file.open(
        driver="GTiff",
        width=64,
        height=64,
        count=1,
        dtype="int16",
        crs=CRS.from_proj4(SOILGRIDS_PROJECTION),
        transform=from_origin(
            transformed_bounds[0] - 1000,
            transformed_bounds[3] + 1000,
            250,
            250,
        ),
    ) as dataset:
        dataset.write(
            np.full(
                (64, 64),
                _COVERAGE_VALUES[property_id],
                dtype=np.int16,
            ),
            1,
        )
    return memory_file.read()


class _Response:
    def __init__(
        self,
        *,
        content: bytes,
        content_type: str,
    ) -> None:
        self.content = content
        self.headers = {"content-type": content_type}
        self.status_code = 200
        self.response = None

    def raise_for_status(self) -> None:
        return None

    def iter_content(self, chunk_size: int) -> Any:
        for offset in range(0, len(self.content), chunk_size):
            yield self.content[offset : offset + chunk_size]

    def __enter__(self) -> _Response:
        return self

    def __exit__(self, *_: object) -> None:
        return None


def test_soilgrids_metadata_and_supported_depths() -> None:
    assert SOILGRIDS_DEPTHS == (
        "0-5cm",
        "5-15cm",
        "15-30cm",
        "30-60cm",
        "60-100cm",
        "100-200cm",
    )
    assert set(_COVERAGE_VALUES) == {"phh2o", "sand", "silt", "clay", "soc"}


def test_soilgrids_download_standardizes_and_registers_task_owned_rasters(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    registry, task_root = _create_task(tmp_path)
    coverage_requests: list[str] = []
    fail_once = True

    def fake_get(
        url: str,
        *,
        params: dict[str, str] | list[tuple[str, str]],
        timeout: float,
        **kwargs: Any,
    ) -> _Response:
        nonlocal fail_once
        if isinstance(params, dict):
            property_id = url.split("/map/")[1].removesuffix(".map")
            coverage_ids = [
                f"{property_id}_{depth}_Q0.5" for depth in SOILGRIDS_DEPTHS
            ]
            payload = (
                "<wcs:Capabilities xmlns:wcs='http://www.opengis.net/wcs/2.0'>"
                + "".join(f"<wcs:CoverageId>{item}</wcs:CoverageId>" for item in coverage_ids)
                + "</wcs:Capabilities>"
            ).encode()
            return _Response(content=payload, content_type="text/xml")

        coverage_id = next(value for key, value in params if key == "COVERAGEID")
        coverage_requests.append(coverage_id)
        if fail_once:
            fail_once = False
            raise requests.ConnectionError("temporary network failure")
        property_id = coverage_id.split("_", maxsplit=1)[0]
        return _Response(content=_geotiff_bytes(property_id), content_type="image/tiff")

    monkeypatch.setattr(
        "nasa_agri_intelligence.collection.soilgrids.requests.get", fake_get
    )
    monkeypatch.setattr("nasa_agri_intelligence.collection.soilgrids.time.sleep", lambda _: None)

    result = collect_soilgrids(registry, "soil-task")

    assert len(coverage_requests) == 6
    assert result["downloaded_count"] == 5
    assert result["standardized_count"] == 5
    assert set(result["new_variables"]) == {
        "soil_ph",
        "soil_sand",
        "soil_silt",
        "soil_clay",
        "soil_organic_carbon",
    }
    manifest_path = task_root / "processed/standardized_manifest.json"
    _, _, records = registry.validate_raster_manifest("soil-task", manifest_path)
    soil = {
        record["variable"]: record
        for record in records
        if record.get("external_source") == "ISRIC/SoilGrids"
    }
    assert len(soil) == 5
    assert soil["soil_ph"]["output_units"] == "pH"
    assert soil["soil_sand"]["source_units"] == "g/kg"
    assert soil["soil_sand"]["output_units"] == "%"
    assert soil["soil_organic_carbon"]["output_units"] == "g/kg"
    assert all(record["data_scope"] == "task_scoped" for record in soil.values())
    assert all(
        record["georeferencing_version"] == SOILGRIDS_GEOREFERENCING_VERSION
        for record in soil.values()
    )
    with rasterio.open(soil["soil_ph"]["output_path"]) as raster:
        values = raster.read(1, masked=True)
        assert raster.count == 1
        assert raster.crs is not None and raster.crs.is_projected
        assert raster.res[0] == pytest.approx(250)
        assert np.allclose(values.compressed(), 7.2)
        assert raster.tags()["task_name"] == "soil-task"
        assert raster.tags()["external_source"] == "ISRIC/SoilGrids"
        footprint = Transformer.from_crs(
            raster.crs, "EPSG:4326", always_xy=True
        ).transform_bounds(*raster.bounds, densify_pts=21)
        assert box(*footprint).intersects(box(69.27, 41.30, 69.29, 41.32))
    with rasterio.open(soil["soil_sand"]["output_path"]) as raster:
        assert np.allclose(raster.read(1, masked=True).compressed(), 25.0)
    with rasterio.open(soil["soil_organic_carbon"]["output_path"]) as raster:
        assert np.allclose(raster.read(1, masked=True).compressed(), 20.0)

    second_result = collect_soilgrids(registry, "soil-task")
    assert second_result["downloaded_count"] == 0
    assert second_result["standardized_count"] == 0
    assert second_result["already_integrated"] == sorted(
        ["soil_ph", "soil_sand", "soil_silt", "soil_clay", "soil_organic_carbon"]
    )
    assert len(coverage_requests) == 6


def test_soilgrids_rejects_unsupported_depth(tmp_path: Path) -> None:
    registry, _ = _create_task(tmp_path)
    with pytest.raises(ValueError, match="Unsupported SoilGrids depth"):
        collect_soilgrids(registry, "soil-task", depth="0-10cm")
