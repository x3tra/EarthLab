from __future__ import annotations

import calendar
from datetime import date, timedelta
import math
from pathlib import Path
from typing import Any

import numpy as np
import rasterio
import requests
from rasterio.transform import from_origin
from shapely.geometry import shape
from shapely.ops import unary_union

from .manifest import load_manifest, write_manifest
from .task_registry import TaskRegistry

POWER_REGIONAL_DAILY_URL = (
    "https://power.larc.nasa.gov/api/temporal/daily/regional"
)
_VARIABLES = {
    "T2M": ("air_temperature", "°C"),
    "T2M_MIN": ("air_temperature_min", "°C"),
    "T2M_MAX": ("air_temperature_max", "°C"),
    "PRECTOTCORR": ("precipitation", "mm"),
}
_FILL_VALUE = -999.0
_MAX_PERIOD_DAYS = 366
_MIN_VALID_FRACTION = 0.9


def collect_power_climate(
    registry: TaskRegistry,
    identifier: str,
    *,
    variables: tuple[str, ...] | None = None,
    timeout: float = 120,
) -> dict[str, Any]:
    """Fetch real NASA POWER daily climate summaries into one registered task."""
    task, task_manifest_path, task_manifest = registry.load_task_manifest(identifier)
    if not isinstance(task.get("task_id"), str):
        raise ValueError("Selected task has no NASA task ID.")
    if task_manifest.get("preprocessing_status") != "completed":
        raise ValueError("Task preprocessing must be complete before climate collection.")
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
        west, south, east, north = geometry.bounds
    except (IndexError, KeyError, TypeError, ValueError) as error:
        raise ValueError("Selected task AOI has no valid geographic bounds.") from error
    if geometry.is_empty or not all(math.isfinite(value) for value in (west, south, east, north)):
        raise ValueError("Selected task AOI has invalid geographic bounds.")

    start_date = _task_date(task_manifest.get("start_date"), "start_date")
    end_date = _task_date(task_manifest.get("end_date"), "end_date")
    if end_date < start_date:
        raise ValueError("Task end date is earlier than its start date.")
    period_days = (end_date - start_date).days + 1
    if period_days > _MAX_PERIOD_DAYS:
        raise ValueError(
            f"NASA POWER daily summaries are limited to {_MAX_PERIOD_DAYS} days per request."
        )
    selected_variables = tuple(_VARIABLES) if variables is None else variables
    if not selected_variables or any(variable not in _VARIABLES for variable in selected_variables):
        raise ValueError("Select one or more supported NASA POWER variables.")

    standardized_path = task["task_directory"] / "processed" / "standardized_manifest.json"
    _, raster_manifest, existing_records = registry.validate_raster_manifest(
        task["task_name"], standardized_path
    )
    existing_keys = {
        (
            record.get("layer_id"),
            record.get("temporal_start"),
            record.get("temporal_end"),
        )
        for record in existing_records
        if record.get("external_source") == "NASA/POWER"
    }

    lon_center = (west + east) / 2
    lat_center = (south + north) / 2
    lon_half_range = max(1.25, (east - west) / 2 + 0.4)
    lat_half_range = max(1.25, (north - south) / 2 + 0.3)
    params_base = {
        "community": "AG",
        "longitude-min": lon_center - lon_half_range,
        "longitude-max": lon_center + lon_half_range,
        "latitude-min": lat_center - lat_half_range,
        "latitude-max": lat_center + lat_half_range,
        "start": start_date.strftime("%Y%m%d"),
        "end": end_date.strftime("%Y%m%d"),
        "format": "JSON",
    }
    task_root = task["task_directory"]
    output_directory = task_root / "processed" / "rasters" / "climate"
    new_records: list[dict[str, Any]] = []
    created_paths: list[Path] = []
    try:
        for layer_id in selected_variables:
            variable, units = _VARIABLES[layer_id]
            periods = [
                {
                    "layer_id": layer_id,
                    "start_date": start_date,
                    "end_date": end_date,
                    "aggregation_period": "task_period",
                    "temporal_aggregation": (
                        "sum of valid daily precipitation amounts"
                        if layer_id == "PRECTOTCORR"
                        else "mean of valid daily values"
                    ),
                },
                *_monthly_periods(layer_id, start_date, end_date),
            ]
            periods.extend(_seasonal_periods(layer_id, start_date, end_date))
            needed_periods = [
                period
                for period in periods
                if (
                    period["layer_id"],
                    period["start_date"].isoformat(),
                    period["end_date"].isoformat(),
                )
                not in existing_keys
            ]
            if not needed_periods:
                continue
            response = requests.get(
                POWER_REGIONAL_DAILY_URL,
                params={"parameters": layer_id, **params_base},
                timeout=timeout,
            )
            response.raise_for_status()
            payload = response.json()
            features = payload.get("features") if isinstance(payload, dict) else None
            if not isinstance(features, list) or not features:
                messages = payload.get("messages", []) if isinstance(payload, dict) else []
                detail = "; ".join(str(item) for item in messages) or "No regional cells returned."
                raise ValueError(f"NASA POWER returned no {layer_id} cells: {detail}")
            for period in needed_periods:
                try:
                    record = _write_climate_period(
                        task=task,
                        features=features,
                        parent_layer_id=layer_id,
                        variable=variable,
                        units=units,
                        period=period,
                    )
                except ValueError as error:
                    if "no valid daily values" not in str(error):
                        raise
                    continue
                new_records.append(record)
                created_paths.append(task_root / str(record["output_path"]))
                existing_keys.add(
                    (
                        record["layer_id"],
                        record["temporal_start"],
                        record["temporal_end"],
                    )
                )
    except (OSError, requests.RequestException, ValueError, rasterio.errors.RasterioError):
        for path in created_paths:
            path.unlink(missing_ok=True)
        raise

    if not new_records:
        return {
            "task_name": task["task_name"],
            "task_id": task["task_id"],
            "source": "NASA/POWER",
            "temporal_start": start_date.isoformat(),
            "temporal_end": end_date.isoformat(),
            "new_variables": [],
            "already_integrated": sorted(
                layer_id
                for layer_id in selected_variables
                if (layer_id, start_date.isoformat(), end_date.isoformat())
                in existing_keys
            ),
            "layers": [],
            "manifest_path": str(standardized_path),
        }

    all_records = [*existing_records, *new_records]
    raster_manifest["rasters"] = all_records
    raster_manifest["records"] = all_records
    raster_manifest["raster_count"] = len(all_records)
    raster_manifest["updated_at"] = date.today().isoformat()
    raster_manifest.setdefault("external_integrations", []).append(
        {
            "source": "NASA/POWER",
            "task_name": task["task_name"],
            "task_id": task["task_id"],
            "temporal_start": start_date.isoformat(),
            "temporal_end": end_date.isoformat(),
            "variables": [record["layer_id"] for record in new_records],
        }
    )
    write_manifest(standardized_path, raster_manifest)
    registry.validate_raster_manifest(task["task_name"], standardized_path)
    task_manifest.setdefault("external_data_sources", []).append(
        {
            "source": "NASA/POWER",
            "manifest_path": str(standardized_path),
            "collected_at": date.today().isoformat(),
            "variables": [record["layer_id"] for record in new_records],
        }
    )
    write_manifest(task_manifest_path, task_manifest)
    return {
        "task_name": task["task_name"],
        "task_id": task["task_id"],
        "source": "NASA/POWER",
        "temporal_start": start_date.isoformat(),
        "temporal_end": end_date.isoformat(),
        "new_variables": list(
            dict.fromkeys(record["variable"] for record in new_records)
        ),
        "already_integrated": sorted(
            layer_id
            for layer_id in selected_variables
            if (layer_id, start_date.isoformat(), end_date.isoformat())
            in existing_keys
        ),
        "layers": new_records,
        "manifest_path": str(standardized_path),
    }


def _task_date(value: Any, field: str) -> date:
    if not isinstance(value, str):
        raise ValueError(f"Task {field} is missing or invalid.")
    try:
        return date.fromisoformat(value)
    except ValueError as error:
        raise ValueError(f"Task {field} is missing or invalid.") from error


def _monthly_periods(
    layer_id: str,
    first_date: date,
    last_date: date,
) -> list[dict[str, Any]]:
    periods = []
    current = date(first_date.year, first_date.month, 1)
    while current <= last_date:
        month_end = date(
            current.year,
            current.month,
            calendar.monthrange(current.year, current.month)[1],
        )
        period_start = max(first_date, current)
        period_end = min(last_date, month_end)
        is_precipitation = layer_id == "PRECTOTCORR"
        periods.append(
            {
                "layer_id": f"{layer_id}_{current:%Y%m}_MONTHLY",
                "start_date": period_start,
                "end_date": period_end,
                "aggregation_period": "monthly",
                "temporal_aggregation": (
                    "monthly sum of valid daily precipitation amounts"
                    if is_precipitation
                    else "monthly mean of valid daily values"
                ),
            }
        )
        if current.month == 12:
            current = date(current.year + 1, 1, 1)
        else:
            current = date(current.year, current.month + 1, 1)
    return periods


def _seasonal_periods(
    layer_id: str,
    first_date: date,
    last_date: date,
) -> list[dict[str, Any]]:
    seasons = (
        ("DJF", 12, 2, -1),
        ("MAM", 3, 5, 0),
        ("JJA", 6, 8, 0),
        ("SON", 9, 11, 0),
    )
    periods = []
    for season_year in range(first_date.year, last_date.year + 1):
        for season, start_month, end_month, year_offset in seasons:
            season_start_year = season_year + year_offset
            season_start = date(season_start_year, start_month, 1)
            season_end_year = season_start_year + (1 if end_month < start_month else 0)
            season_end = date(
                season_end_year,
                end_month,
                calendar.monthrange(season_end_year, end_month)[1],
            )
            if first_date > season_start or last_date < season_end:
                continue
            is_precipitation = layer_id == "PRECTOTCORR"
            periods.append(
                {
                    "layer_id": f"{layer_id}_{season_year}_{season}",
                    "start_date": season_start,
                    "end_date": season_end,
                    "aggregation_period": "seasonal",
                    "temporal_aggregation": (
                        "seasonal sum of valid daily precipitation amounts"
                        if is_precipitation
                        else "seasonal mean of valid daily values"
                    ),
                }
            )
    return periods


def _write_climate_period(
    *,
    task: dict[str, Any],
    features: list[Any],
    parent_layer_id: str,
    variable: str,
    units: str,
    period: dict[str, Any],
) -> dict[str, Any]:
    layer_id = str(period["layer_id"])
    first_date = period["start_date"]
    last_date = period["end_date"]
    raster, transform, crs, grid_spacing = _aggregate_features(
        features,
        layer_id=parent_layer_id,
        first_date=first_date,
        last_date=last_date,
    )
    task_root = task["task_directory"]
    output_directory = task_root / "processed" / "rasters" / "climate"
    output_path = output_directory / (
        f"NASA_POWER_DAILY_{layer_id}_{first_date:%Y%m%d}_{last_date:%Y%m%d}.tif"
    )
    if output_path.exists():
        raise FileExistsError(
            f"Climate raster already exists but is not in the task manifest: {output_path}"
        )
    output_directory.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_name(f".{output_path.name}.tmp")
    try:
        with rasterio.Env(GDAL_TIFF_INTERNAL_MASK=True):
            with rasterio.open(
                temporary_path,
                "w",
                driver="GTiff",
                width=raster.shape[1],
                height=raster.shape[0],
                count=1,
                dtype="float32",
                crs=crs,
                transform=transform,
                nodata=float("nan"),
                compress="deflate",
            ) as dataset:
                dataset.write(raster, 1)
                dataset.write_mask(np.isfinite(raster).astype(np.uint8) * 255)
                dataset.set_band_description(1, variable)
                dataset.set_band_unit(1, units)
                dataset.update_tags(
                    source_product_id="NASA_POWER_DAILY",
                    source_layer_id=layer_id,
                    external_source="NASA/POWER",
                    task_name=task["task_name"],
                    task_id=task["task_id"],
                    variable_kind="continuous",
                    temporal_start=first_date.isoformat(),
                    temporal_end=last_date.isoformat(),
                    temporal_aggregation=period["temporal_aggregation"],
                    native_resolution_degrees=(
                        f"{grid_spacing[0]:g} latitude x "
                        f"{grid_spacing[1]:g} longitude"
                    ),
                )
        temporary_path.replace(output_path)
    finally:
        temporary_path.unlink(missing_ok=True)
    return {
        "product_id": "NASA_POWER_DAILY",
        "layer_id": layer_id,
        "variable": variable,
        "output_path": output_path.relative_to(task_root).as_posix(),
        "external_source": "NASA/POWER",
        "data_scope": "task_scoped",
        "task_name": task["task_name"],
        "task_id": task["task_id"],
        "variable_kind": "continuous",
        "is_quality": False,
        "output_units": units,
        "temporal_start": first_date.isoformat(),
        "temporal_end": last_date.isoformat(),
        "temporal_resolution": "daily",
        "temporal_aggregation": period["temporal_aggregation"],
        "aggregation_period": period["aggregation_period"],
        "native_resolution_degrees": list(grid_spacing),
        "quality_filter_applied": False,
        "quality_layers": [],
        "physical_valid_range": [
            float(np.nanmin(raster)),
            float(np.nanmax(raster)),
        ],
        "valid_pixel_count": int(np.isfinite(raster).sum()),
        "masked_pixel_count": int(raster.size - np.isfinite(raster).sum()),
    }


def _aggregate_features(
    features: list[Any],
    *,
    layer_id: str,
    first_date: date,
    last_date: date,
) -> tuple[np.ndarray, Any, str, tuple[float, float]]:
    points: dict[tuple[float, float], list[float]] = {}
    date_keys = {
        (first_date + timedelta(days=offset)).strftime("%Y%m%d")
        for offset in range((last_date - first_date).days + 1)
    }
    for feature in features:
        if not isinstance(feature, dict):
            continue
        geometry = feature.get("geometry")
        properties = feature.get("properties")
        coordinates = geometry.get("coordinates") if isinstance(geometry, dict) else None
        parameters = properties.get("parameter") if isinstance(properties, dict) else None
        layer_data = parameters.get(layer_id) if isinstance(parameters, dict) else None
        if (
            not isinstance(coordinates, list)
            or len(coordinates) < 2
            or not isinstance(layer_data, dict)
        ):
            continue
        lon, lat = float(coordinates[0]), float(coordinates[1])
        values = []
        for key in date_keys:
            value = layer_data.get(key)
            if value is None:
                continue
            parsed = float(value)
            if math.isfinite(parsed) and parsed != _FILL_VALUE:
                values.append(parsed)
        points[(lon, lat)] = values
    if not points:
        raise ValueError(f"NASA POWER response contains no valid {layer_id} observations.")

    longitudes = sorted({point[0] for point in points})
    latitudes = sorted({point[1] for point in points})
    if len(longitudes) < 2 or len(latitudes) < 2:
        raise ValueError(f"NASA POWER returned an incomplete regional grid for {layer_id}.")
    longitude_steps = np.diff(longitudes)
    latitude_steps = np.diff(latitudes)
    dx = float(np.median(longitude_steps))
    dy = float(np.median(latitude_steps))
    if (
        dx <= 0
        or dy <= 0
        or not np.allclose(longitude_steps, dx, rtol=0, atol=1e-6)
        or not np.allclose(latitude_steps, dy, rtol=0, atol=1e-6)
    ):
        raise ValueError(f"NASA POWER returned a non-regular grid for {layer_id}.")

    array = np.full((len(latitudes), len(longitudes)), np.nan, dtype=np.float32)
    longitude_indices = {value: index for index, value in enumerate(longitudes)}
    latitude_indices = {value: index for index, value in enumerate(latitudes)}
    for (longitude, latitude), values in points.items():
        minimum_valid_values = max(1, math.ceil(len(date_keys) * _MIN_VALID_FRACTION))
        if len(values) >= minimum_valid_values:
            aggregate = (
                sum(values)
                if layer_id == "PRECTOTCORR"
                else float(np.mean(values))
            )
            array[
                len(latitudes) - 1 - latitude_indices[latitude],
                longitude_indices[longitude],
            ] = aggregate
    if not np.isfinite(array).any():
        raise ValueError(f"NASA POWER returned no valid daily values for {layer_id}.")
    transform = from_origin(
        longitudes[0] - dx / 2,
        latitudes[-1] + dy / 2,
        dx,
        dy,
    )
    return array, transform, "EPSG:4326", (dy, dx)
