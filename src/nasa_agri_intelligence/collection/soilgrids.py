from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import math
from pathlib import Path
import time
from typing import Any
import xml.etree.ElementTree as ET

import numpy as np
import rasterio
import requests
from pyproj import CRS, Transformer
from rasterio.warp import transform_bounds
from shapely import transform as transform_geometry
from rasterio.mask import mask
from shapely.geometry import box, mapping, shape
from shapely.ops import unary_union

from .manifest import write_manifest
from .task_registry import TaskRegistry

SOILGRIDS_VERSION = "2.0"
SOILGRIDS_WCS_BASE_URL = "https://maps.isric.org/mapserv?map=/map/"
SOILGRIDS_CRS = "http://www.opengis.net/def/crs/EPSG/0/152160"
SOILGRIDS_PROJECTION = (
    "+proj=igh +lon_0=0 +x_0=0 +y_0=0 +datum=WGS84 +units=m +no_defs"
)
SOILGRIDS_GEOREFERENCING_VERSION = "EPSG:152160-v2"
SOILGRIDS_DEPTHS = ("0-5cm", "5-15cm", "15-30cm", "30-60cm", "60-100cm", "100-200cm")
_MAX_DOWNLOAD_BYTES = 256 * 1024 * 1024
_MAX_RASTER_PIXELS = 20_000_000
_VARIABLES: dict[str, dict[str, Any]] = {
    "soil_ph": {
        "property": "phh2o",
        "name": "Soil pH in water",
        "source_units": "pH × 10",
        "units": "pH",
        "scale_factor": 0.1,
        "raw_valid_range": (10, 140),
        "valid_range": (1.0, 14.0),
    },
    "soil_sand": {
        "property": "sand",
        "name": "Sand fraction",
        "source_units": "g/kg",
        "units": "%",
        "scale_factor": 0.1,
        "raw_valid_range": (0, 1000),
        "valid_range": (0.0, 100.0),
    },
    "soil_silt": {
        "property": "silt",
        "name": "Silt fraction",
        "source_units": "g/kg",
        "units": "%",
        "scale_factor": 0.1,
        "raw_valid_range": (0, 1000),
        "valid_range": (0.0, 100.0),
    },
    "soil_clay": {
        "property": "clay",
        "name": "Clay fraction",
        "source_units": "g/kg",
        "units": "%",
        "scale_factor": 0.1,
        "raw_valid_range": (0, 1000),
        "valid_range": (0.0, 100.0),
    },
    "soil_organic_carbon": {
        "property": "soc",
        "name": "Soil organic carbon content",
        "source_units": "g/kg",
        "units": "g/kg",
        "scale_factor": 1.0,
        "raw_valid_range": (0, 1000),
        "valid_range": (0.0, 1000.0),
    },
}


def collect_soilgrids(
    registry: TaskRegistry,
    identifier: str,
    *,
    depth: str = "0-5cm",
    timeout: float = 120,
    max_retries: int = 3,
) -> dict[str, Any]:
    """Fetch and standardize SoilGrids median predictions for a task AOI."""
    if depth not in SOILGRIDS_DEPTHS:
        raise ValueError(f"Unsupported SoilGrids depth interval: {depth}")
    task, task_manifest_path, task_manifest = registry.load_task_manifest(identifier)
    if not isinstance(task.get("task_id"), str):
        raise ValueError("Selected task has no NASA task ID.")
    if task_manifest.get("preprocessing_status") != "completed":
        raise ValueError("Task preprocessing must be complete before soil collection.")
    aoi = task_manifest.get("aoi")
    if not isinstance(aoi, dict) or not isinstance(aoi.get("geojson"), dict):
        raise ValueError("Selected task does not contain a valid AOI.")
    try:
        geometry = unary_union(
            [
                shape(feature["geometry"])
                for feature in aoi["geojson"]["features"]
            ]
        )
    except (IndexError, KeyError, TypeError, ValueError) as error:
        raise ValueError("Selected task AOI has no valid polygon geometry.") from error
    if (
        geometry.is_empty
        or not geometry.is_valid
        or geometry.geom_type not in {"Polygon", "MultiPolygon"}
        or not all(math.isfinite(value) for value in geometry.bounds)
    ):
        raise ValueError("Selected task AOI must be a valid geographic polygon.")

    projected_geometry_transformer = Transformer.from_crs(
        "EPSG:4326", CRS.from_proj4(SOILGRIDS_PROJECTION), always_xy=True
    )
    projected_geometry = transform_geometry(
        geometry,
        projected_geometry_transformer.transform,
        interleaved=False,
    )
    if projected_geometry.is_empty or not projected_geometry.is_valid:
        raise ValueError("Task AOI could not be transformed to the SoilGrids grid.")
    west, south, east, north = projected_geometry.bounds
    if east <= west or north <= south:
        raise ValueError("Task AOI has invalid projected bounds.")

    external_downloads = task_manifest.setdefault("external_downloads", [])
    if not isinstance(external_downloads, list):
        raise ValueError("Task external download records are invalid.")
    task_root = task["task_directory"]
    output_directory = task_root / "raw" / "external" / "soilgrids"
    output_directory.mkdir(parents=True, exist_ok=True)
    existing_downloads = {
        (record.get("product_id"), record.get("layer_id")): record
        for record in external_downloads
        if isinstance(record, dict)
        and record.get("external_source") == "ISRIC/SoilGrids"
        and record.get("depth_interval") == depth
    }

    new_records: list[dict[str, Any]] = []
    for variable_id, variable in _VARIABLES.items():
        property_id = str(variable["property"])
        coverage_id = f"{property_id}_{depth}_Q0.5"
        existing = existing_downloads.get(("ISRIC_SOILGRIDS", coverage_id))
        if existing is not None:
            try:
                _validate_existing_download(
                    existing,
                    task_root,
                    task,
                    aoi_geometry=geometry,
                )
            except ValueError as error:
                existing.update(
                    status="invalid_coverage",
                    validation_error=str(error),
                )
                _remove_invalid_standardized_soil_layer(
                    registry,
                    task,
                    coverage_id,
                    validation_error=str(error),
                )
            else:
                if existing.get("georeferencing_version") == SOILGRIDS_GEOREFERENCING_VERSION:
                    continue

        service_url = f"{SOILGRIDS_WCS_BASE_URL}{property_id}.map"
        _verify_coverage(service_url, coverage_id, timeout=timeout, max_retries=max_retries)
        query = [
            ("SERVICE", "WCS"),
            ("VERSION", "2.0.1"),
            ("REQUEST", "GetCoverage"),
            ("COVERAGEID", coverage_id),
            ("FORMAT", "GEOTIFF_INT16"),
            ("SUBSET", f"X({west:.6f},{east:.6f})"),
            ("SUBSET", f"Y({south:.6f},{north:.6f})"),
            ("SUBSETTINGCRS", SOILGRIDS_CRS),
            ("OUTPUTCRS", SOILGRIDS_CRS),
            ("RESX", "250"),
            ("RESY", "250"),
        ]
        filename = (
            f"ISRIC_SOILGRIDS_{coverage_id}_"
            f"{SOILGRIDS_GEOREFERENCING_VERSION.replace(':', '-')}.tif"
        )
        destination = output_directory / filename
        if destination.exists():
            raise FileExistsError(
                f"Unregistered SoilGrids raster already exists: {destination}"
            )
        temporary_path = destination.with_name(f".{destination.name}.part")
        try:
            _download_coverage(
                service_url,
                query,
                temporary_path,
                timeout=timeout,
                max_retries=max_retries,
            )
            raster_metadata = _clip_and_tag(
                temporary_path,
                destination,
                projected_geometry,
                aoi_geometry=geometry,
                variable=variable,
                coverage_id=coverage_id,
                depth=depth,
                task_name=task["task_name"],
                task_id=task["task_id"],
            )
        finally:
            temporary_path.unlink(missing_ok=True)

        relative_path = destination.relative_to(task_root).as_posix()
        record = {
            "status": "validated",
            "filename": filename,
            "file_type": "geotiff",
            "local_path": relative_path,
            "product_id": "ISRIC_SOILGRIDS",
            "layer_id": coverage_id,
            "variable": variable_id,
            "external_source": "ISRIC/SoilGrids",
            "source_url": service_url,
            "source_version": SOILGRIDS_VERSION,
            "georeferencing_version": SOILGRIDS_GEOREFERENCING_VERSION,
            "data_scope": "task_scoped",
            "task_name": task["task_name"],
            "task_id": task["task_id"],
            "depth_interval": depth,
            "quantile": "Q0.5",
            "temporal_resolution": "static prediction",
            "source_units": variable["source_units"],
            "output_units": variable["units"],
            "scale_factor": variable["scale_factor"],
            "raw_valid_range": list(variable["raw_valid_range"]),
            "valid_range": list(variable["valid_range"]),
            "spatial_resolution_m": 250,
            "retrieved_at": datetime.now(timezone.utc).isoformat(),
            "sha256": _sha256(destination),
            **raster_metadata,
        }
        provenance_path = destination.with_suffix(".json")
        write_manifest(
            provenance_path,
            {
                "provider": "ISRIC - World Soil Information",
                "dataset": "SoilGrids 2.0",
                "source_url": service_url,
                "coverage_id": coverage_id,
                "depth_interval": depth,
                "quantile": "Q0.5",
                "source_units": variable["source_units"],
                "output_units": variable["units"],
                "scale_factor": variable["scale_factor"],
                "native_resolution_m": 250,
                "crs": raster_metadata["crs"],
                "task_name": task["task_name"],
                "task_id": task["task_id"],
                "retrieved_at": record["retrieved_at"],
                "license": "CC-BY-4.0",
                "citation": (
                    "Poggio et al. (2021), SoilGrids 2.0: producing soil "
                    "information for the globe with quantified spatial uncertainty, "
                    "SOIL 7, 217–240, doi:10.5194/soil-7-217-2021."
                ),
            },
        )
        record["source_metadata_file"] = provenance_path.relative_to(task_root).as_posix()
        external_downloads.append(record)
        new_records.append(record)

    task_manifest["external_downloads"] = external_downloads
    task_manifest["external_data_sources"] = [
        *[
            record
            for record in task_manifest.get("external_data_sources", [])
            if not isinstance(record, dict)
            or record.get("source") != "ISRIC/SoilGrids"
            or record.get("depth_interval") != depth
        ],
        {
            "source": "ISRIC/SoilGrids",
            "version": SOILGRIDS_VERSION,
            "task_name": task["task_name"],
            "task_id": task["task_id"],
            "depth_interval": depth,
            "variables": [
                record["layer_id"]
                for record in external_downloads
                if record.get("external_source") == "ISRIC/SoilGrids"
                and record.get("depth_interval") == depth
                and record.get("status") == "validated"
            ],
        },
    ]
    write_manifest(task_manifest_path, task_manifest)

    from ..standardization import standardize_external_rasters

    standardized = standardize_external_rasters(registry, task["task_name"])
    return {
        "task_name": task["task_name"],
        "task_id": task["task_id"],
        "source": "ISRIC/SoilGrids",
        "version": SOILGRIDS_VERSION,
        "depth_interval": depth,
        "new_variables": [record["variable"] for record in new_records],
        "already_integrated": sorted(
            record["variable"]
            for record in external_downloads
            if record.get("external_source") == "ISRIC/SoilGrids"
            and record.get("depth_interval") == depth
            and record.get("status") == "validated"
            and record.get("georeferencing_version")
            == SOILGRIDS_GEOREFERENCING_VERSION
            and record not in new_records
        ),
        "downloaded_count": len(new_records),
        "standardized_count": len(standardized),
        "layers": standardized,
        "manifest_path": str(task_root / "processed" / "standardized_manifest.json"),
    }


def _verify_coverage(
    service_url: str,
    coverage_id: str,
    *,
    timeout: float,
    max_retries: int,
) -> None:
    response = _request_with_retries(
        service_url,
        params={"SERVICE": "WCS", "VERSION": "2.0.1", "REQUEST": "GetCapabilities"},
        timeout=timeout,
        max_retries=max_retries,
    )
    try:
        root = ET.fromstring(response.content)
    except ET.ParseError as error:
        raise ValueError(f"ISRIC returned invalid WCS capabilities for {service_url}.") from error
    if not any(element.text == coverage_id for element in root.iter() if element.tag.endswith("CoverageId")):
        raise ValueError(f"SoilGrids coverage is not available: {coverage_id}")


def _download_coverage(
    service_url: str,
    params: list[tuple[str, str]],
    destination: Path,
    *,
    timeout: float,
    max_retries: int,
) -> None:
    last_error: Exception | None = None
    for attempt in range(max_retries + 1):
        offset = destination.stat().st_size if destination.is_file() else 0
        try:
            with requests.get(
                service_url,
                params=params,
                timeout=timeout,
                stream=True,
                headers={"Range": f"bytes={offset}-"} if offset else {},
            ) as response:
                response.raise_for_status()
                append = (
                    offset > 0
                    and response.status_code == 206
                    and response.headers.get("content-range", "").startswith(
                        f"bytes {offset}-"
                    )
                )
                if offset and response.status_code == 206 and not append:
                    destination.unlink(missing_ok=True)
                    raise ValueError("ISRIC returned an invalid partial-content range.")
                content_type = response.headers.get("content-type", "").casefold()
                if "tiff" not in content_type and "geotiff" not in content_type:
                    first_chunk = next(response.iter_content(chunk_size=64), b"")
                    if not first_chunk.startswith((b"II*\\x00", b"MM\\x00*")):
                        detail = first_chunk.decode("utf-8", errors="replace")[:400]
                        raise ValueError(
                            f"ISRIC WCS did not return GeoTIFF data: {detail}"
                        )
                    chunks = [first_chunk, *response.iter_content(chunk_size=1024 * 1024)]
                else:
                    chunks = response.iter_content(chunk_size=1024 * 1024)
                total = 0
                with destination.open("ab" if append else "wb") as file:
                    for chunk in chunks:
                        if not chunk:
                            continue
                        total += len(chunk)
                        if total > _MAX_DOWNLOAD_BYTES:
                            raise ValueError(
                                f"SoilGrids AOI raster exceeds {_MAX_DOWNLOAD_BYTES} bytes."
                            )
                        file.write(chunk)
                if total < 128:
                    raise ValueError("ISRIC WCS returned an empty or truncated GeoTIFF.")
                with rasterio.open(destination) as dataset:
                    if dataset.count != 1 or dataset.crs is None:
                        raise ValueError("ISRIC GeoTIFF must have one band and a CRS.")
                    if dataset.width * dataset.height > _MAX_RASTER_PIXELS:
                        raise ValueError(
                            f"SoilGrids raster exceeds {_MAX_RASTER_PIXELS} pixels."
                        )
                    values = dataset.read(1, masked=True)
                    if values.count() == 0:
                        raise ValueError("SoilGrids raster contains no valid pixels.")
                return
        except requests.RequestException as error:
            last_error = error
            if attempt >= max_retries or (
                isinstance(error, requests.HTTPError)
                and error.response is not None
                and error.response.status_code < 500
                and error.response.status_code != 429
            ):
                raise
            time.sleep(min(2**attempt, 8))
        except (OSError, rasterio.errors.RasterioError, ValueError) as error:
            destination.unlink(missing_ok=True)
            last_error = error
            if attempt >= max_retries:
                raise
            time.sleep(min(2**attempt, 8))
    if last_error is not None:
        raise last_error


def _request_with_retries(
    url: str,
    *,
    params: dict[str, str],
    timeout: float,
    max_retries: int,
) -> requests.Response:
    last_error: requests.RequestException | None = None
    for attempt in range(max_retries + 1):
        try:
            response = requests.get(url, params=params, timeout=timeout)
            response.raise_for_status()
            return response
        except requests.RequestException as error:
            last_error = error
            if attempt >= max_retries or (
                isinstance(error, requests.HTTPError)
                and error.response is not None
                and error.response.status_code < 500
                and error.response.status_code != 429
            ):
                raise
            time.sleep(min(2**attempt, 8))
    if last_error is not None:
        raise last_error
    raise RuntimeError("ISRIC WCS request did not complete.")


def _clip_and_tag(
    source_path: Path,
    destination_path: Path,
    projected_geometry: Any,
    *,
    aoi_geometry: Any,
    variable: dict[str, Any],
    coverage_id: str,
    depth: str,
    task_name: str,
    task_id: str,
) -> dict[str, Any]:
    property_id = str(variable["property"])
    raw_min, raw_max = variable["raw_valid_range"]
    with rasterio.open(source_path) as source:
        if source.crs is None or not source.crs.is_projected:
            raise ValueError(f"SoilGrids coverage has an unsupported CRS: {source_path}")
        expected_crs = CRS.from_proj4(SOILGRIDS_PROJECTION)
        if not source.crs.equals(expected_crs):
            raise ValueError(
                "SoilGrids returned a raster whose CRS does not match the requested "
                f"EPSG:152160 grid: {source.crs}"
            )
        if any(not 200 <= abs(float(value)) <= 300 for value in source.res):
            raise ValueError(
                f"SoilGrids returned an implausible native resolution: {source.res}"
            )
        geographic_bounds = transform_bounds(
            source.crs,
            "EPSG:4326",
            *source.bounds,
            densify_pts=21,
        )
        if not box(*geographic_bounds).intersects(aoi_geometry):
            raise ValueError(
                "SoilGrids raster footprint does not intersect the requested AOI "
                "when interpreted in its declared CRS."
            )
        source_crs = source.crs.to_string()
        source_resolution = [float(value) for value in source.res]
        values, transform = mask(
            source,
            [mapping(projected_geometry)],
            crop=True,
            filled=True,
            nodata=-32768,
        )
        raw = values[0]
        valid = np.isfinite(raw) & (raw >= raw_min) & (raw <= raw_max)
        if not valid.any() or not np.any(raw[valid] != 0):
            raise ValueError(
                f"SoilGrids returned no plausible {property_id} values inside the AOI."
            )
        profile = source.profile.copy()
        profile.pop("blockxsize", None)
        profile.pop("blockysize", None)
        profile.update(
            driver="GTiff",
            width=raw.shape[1],
            height=raw.shape[0],
            count=1,
            dtype="int16",
            transform=transform,
            nodata=-32768,
            compress="deflate",
        )
    destination_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = destination_path.with_name(f".{destination_path.name}.tmp")
    try:
        with rasterio.Env(GDAL_TIFF_INTERNAL_MASK=True):
            with rasterio.open(temporary_path, "w", **profile) as output:
                output.write(raw.astype(np.int16), 1)
                output.write_mask(valid.astype(np.uint8) * 255)
                output.set_band_description(1, str(variable["name"]))
                output.set_band_unit(1, str(variable["source_units"]))
                output.update_tags(
                    source_product_id="ISRIC_SOILGRIDS",
                    source_layer_id=coverage_id,
                    external_source="ISRIC/SoilGrids",
                    source_version=SOILGRIDS_VERSION,
                    georeferencing_version=SOILGRIDS_GEOREFERENCING_VERSION,
                    task_name=task_name,
                    task_id=task_id,
                    variable_kind="continuous",
                    depth_interval=depth,
                    quantile="Q0.5",
                    source_units=str(variable["source_units"]),
                    native_resolution_m="250",
                    source_coverage_id=coverage_id,
                )
        temporary_path.replace(destination_path)
    finally:
        temporary_path.unlink(missing_ok=True)

    return {
        "width": int(raw.shape[1]),
        "height": int(raw.shape[0]),
        "crs": source_crs,
        "resolution": source_resolution,
        "valid_pixel_count": int(valid.sum()),
        "masked_pixel_count": int(valid.size - valid.sum()),
        "raw_valid_range": [int(raw[valid].min()), int(raw[valid].max())],
    }


def _validate_existing_download(
    record: dict[str, Any],
    task_root: Path,
    task: dict[str, Any],
    *,
    aoi_geometry: Any,
) -> None:
    path_value = record.get("local_path")
    if record.get("status") != "validated" or not isinstance(path_value, str):
        raise ValueError("Existing SoilGrids download record is not validated.")
    path = (task_root / path_value).resolve()
    if not path.is_relative_to((task_root / "raw").resolve()) or not path.is_file():
        raise ValueError("Existing SoilGrids download is outside this task or missing.")
    with rasterio.open(path) as dataset:
        tags = dataset.tags()
        if (
            tags.get("external_source") != "ISRIC/SoilGrids"
            or tags.get("task_name") != task["task_name"]
            or tags.get("task_id") != task["task_id"]
            or tags.get("source_layer_id") != record.get("layer_id")
        ):
            raise ValueError("Existing SoilGrids raster provenance does not match this task.")
        expected_crs = CRS.from_proj4(SOILGRIDS_PROJECTION)
        if dataset.crs is None or not dataset.crs.equals(expected_crs):
            raise ValueError(
                "Existing SoilGrids raster uses an unverified CRS and must be reacquired."
            )
        if any(not 200 <= abs(float(value)) <= 300 for value in dataset.res):
            raise ValueError("Existing SoilGrids raster has an implausible resolution.")
        geographic_bounds = transform_bounds(
            dataset.crs,
            "EPSG:4326",
            *dataset.bounds,
            densify_pts=21,
        )
        if not box(*geographic_bounds).intersects(aoi_geometry):
            raise ValueError(
                "Existing SoilGrids raster footprint does not intersect this task AOI."
            )


def _remove_invalid_standardized_soil_layer(
    registry: TaskRegistry,
    task: dict[str, Any],
    layer_id: str,
    *,
    validation_error: str,
) -> None:
    manifest_path = task["task_directory"] / "processed" / "standardized_manifest.json"
    if not manifest_path.is_file():
        return
    _, raster_manifest, records = registry.validate_raster_manifest(
        str(task["task_name"]), manifest_path
    )
    retained: list[dict[str, Any]] = []
    superseded = raster_manifest.setdefault("superseded_rasters", [])
    for record in records:
        if (
            record.get("external_source") == "ISRIC/SoilGrids"
            and record.get("layer_id") == layer_id
        ):
            superseded.append(
                {
                    **record,
                    "status": "invalid_coverage",
                    "validation_error": validation_error,
                }
            )
        else:
            retained.append(record)
    raster_manifest["rasters"] = retained
    raster_manifest["records"] = retained
    raster_manifest["raster_count"] = len(retained)
    statuses = raster_manifest.setdefault("derived_layer_statuses", [])
    if not isinstance(statuses, list):
        raise ValueError("Task raster manifest has invalid layer status metadata.")
    statuses[:] = [
        item
        for item in statuses
        if not (
            isinstance(item, dict)
            and item.get("product_id") == "ISRIC_SOILGRIDS"
            and item.get("layer_id") == layer_id
        )
    ]
    statuses.append(
        {
            "product_id": "ISRIC_SOILGRIDS",
            "layer_id": layer_id,
            "status": "invalid_coverage",
            "detail": validation_error,
        }
    )
    write_manifest(manifest_path, raster_manifest)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
