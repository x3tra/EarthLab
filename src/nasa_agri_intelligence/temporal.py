from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Literal

import numpy as np
import rasterio

from .collection.manifest import write_manifest
from .collection.task_registry import TaskRegistry

_DATE_IN_NAME = re.compile(r"_(\d{8})T\d{6}_")
_CADENCE_DAYS = re.compile(r"(\d+)\s*[- ]?day", re.IGNORECASE)


def build_temporal_inventory(
    task_directory: str | Path,
    *,
    task_name: str,
    task_id: str,
    metadata_dir: str | Path = "data/metadata",
) -> dict[str, Any]:
    """Describe one task's raster dates and cadence gaps without interpolation."""

    task_root = Path(task_directory).resolve()
    registry = TaskRegistry(task_root.parent)
    task, _, task_manifest = registry.load_task_manifest(task_name)
    if task["task_directory"] != task_root or task.get("task_id") != task_id:
        raise ValueError("Selected temporal inventory task identity does not match the registry.")
    manifest_path = task_root / "processed" / "standardized_manifest.json"
    _, standardized, raster_records = registry.validate_raster_manifest(
        task_name,
        manifest_path,
    )
    product_metadata = _load_product_metadata(Path(metadata_dir))
    task_ranges: dict[tuple[str, str], list[dict[str, str]]] = {}
    product_id = task_manifest.get("product_id")
    layers = task_manifest.get("layers", [])
    if isinstance(product_id, str) and isinstance(layers, list):
        for layer_id in layers:
            if isinstance(layer_id, str):
                task_ranges[(product_id, layer_id)] = [
                    {
                        "task_id": task_id,
                        "start_date": task_manifest.get("start_date", ""),
                        "end_date": task_manifest.get("end_date", ""),
                    }
                ]
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)

    for record in raster_records:
        product_id = record.get("product_id")
        layer_id = record.get("layer_id")
        output_path = Path(record["output_path"]).resolve()
        acquisition_date = _acquisition_date(output_path)
        raster_details = _raster_details(output_path)
        grouped[(product_id, layer_id)].append(
            {
                "date": acquisition_date.isoformat() if acquisition_date else None,
                "path": _relative_path(output_path, task_root),
                "exists": True,
                "valid_pixel_count": record.get("valid_pixel_count"),
                "masked_pixel_count": record.get("masked_pixel_count"),
                "raster": raster_details,
            }
        )

    series = []
    for (product_id, layer_id), observations in sorted(grouped.items()):
        observations.sort(key=lambda item: item["date"] or "")
        product = product_metadata.get(product_id, {})
        cadence_text = product.get("catalog", {}).get("TemporalGranularity")
        cadence_days = _cadence_days(cadence_text)
        dates = [
            date.fromisoformat(item["date"])
            for item in observations
            if item.get("date") and item["exists"]
        ]
        missing_slots = _missing_cadence_slots(dates, cadence_days)
        series.append(
            {
                "product_id": product_id,
                "layer_id": layer_id,
                "temporal_granularity": cadence_text,
                "nominal_cadence_days": cadence_days,
                "requested_date_ranges": task_ranges.get((product_id, layer_id), []),
                "observations": observations,
                "observation_count": len(dates),
                "missing_file_count": sum(not item["exists"] for item in observations),
                "missing_cadence_slots": missing_slots,
                "interpolated_observation_count": 0,
            }
        )

    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "task_name": task_name,
        "task_id": task_id,
        "source_manifest": str(manifest_path),
        "series_count": len(series),
        "observation_count": sum(item["observation_count"] for item in series),
        "series": series,
        "policy": (
            "Original file dates are retained. Missing observations are recorded, "
            "not interpolated or replaced with zero."
        ),
    }


def aggregate_monthly(
    raster_paths: list[str | Path],
    output_dir: str | Path | None = None,
    *,
    task_directory: str | Path,
    task_name: str,
    task_id: str,
    method: Literal["mean", "median", "min", "max"] = "mean",
    requested_start: date | None = None,
    requested_end: date | None = None,
    allow_unfiltered_quality: bool = False,
) -> dict[str, Any]:
    """Aggregate continuous rasters explicitly owned by one task."""

    if not raster_paths:
        raise ValueError("At least one raster path is required for aggregation.")
    task_root = Path(task_directory).resolve()
    registry = TaskRegistry(task_root.parent)
    task, _, _ = registry.load_task_manifest(task_name)
    if task["task_directory"] != task_root or task.get("task_id") != task_id:
        raise ValueError("Selected aggregation task identity does not match the registry.")
    standardization_manifest = task_root / "processed" / "standardized_manifest.json"
    _, _, raster_records = registry.validate_raster_manifest(
        task_name,
        standardization_manifest,
    )
    owned_rasters = {Path(item["output_path"]).resolve() for item in raster_records}
    selected_rasters = [Path(path).resolve() for path in raster_paths]
    if any(path not in owned_rasters for path in selected_rasters):
        raise ValueError("Monthly aggregation input is not listed in the selected task manifest.")
    destination_dir = (
        Path(output_dir).resolve()
        if output_dir is not None
        else task_root / "processed" / "temporal" / "monthly"
    )
    if not destination_dir.is_relative_to(task_root):
        raise ValueError("Monthly aggregation output must stay inside the selected task directory.")
    if method not in {"mean", "median", "min", "max"}:
        raise ValueError(f"Unsupported monthly aggregation method: {method}")
    if requested_start and requested_end and requested_end < requested_start:
        raise ValueError("requested_end must be on or after requested_start.")

    observations = []
    reference_grid: tuple[Any, ...] | None = None
    reference_identity: tuple[str, str, str] | None = None
    seen_dates: set[date] = set()
    reference_quality_layers: list[str] | None = None
    for source_path in selected_rasters:
        observation_date = _acquisition_date(source_path)
        if observation_date is None:
            raise ValueError(f"No observation date can be read from raster name: {source_path}")
        if observation_date in seen_dates:
            raise ValueError(f"Duplicate observation date supplied: {observation_date}")
        seen_dates.add(observation_date)
        with rasterio.open(source_path) as source:
            if source.count != 1:
                raise ValueError(f"Expected a single-band raster: {source_path}")
            tags = source.tags()
            layer_id = tags.get("source_layer_id", "")
            if not np.issubdtype(np.dtype(source.dtypes[0]), np.floating):
                raise ValueError(
                    f"Monthly aggregation accepts continuous floating-point rasters, "
                    f"not categorical or quality data: {source_path}"
                )
            grid = (
                source.crs,
                source.transform,
                source.width,
                source.height,
                tags.get("standardized_units", ""),
            )
            if reference_grid is None:
                reference_grid = grid
                reference_identity = (
                    tags.get("source_product_id", ""),
                    layer_id,
                    tags.get("standardized_units", ""),
                )
            elif grid[:4] != reference_grid[:4] or grid[4] != reference_grid[4]:
                raise ValueError(
                    "Monthly aggregation requires spatially aligned rasters with "
                    "identical CRS, transform, dimensions, and units."
                )
            identity = (
                tags.get("source_product_id", ""),
                layer_id,
                tags.get("standardized_units", ""),
            )
            if identity != reference_identity:
                raise ValueError(
                    "Monthly aggregation inputs must represent the same product "
                    "layer and units."
                )
            try:
                quality_layers = json.loads(tags.get("quality_layers", "[]"))
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"Invalid quality-layer metadata in {source_path}"
                ) from error
            if not isinstance(quality_layers, list):
                raise ValueError(f"Invalid quality-layer metadata in {source_path}")
            if not all(isinstance(layer, str) for layer in quality_layers):
                raise ValueError(f"Invalid quality-layer metadata in {source_path}")
            if reference_quality_layers is None:
                reference_quality_layers = quality_layers
            elif quality_layers != reference_quality_layers:
                raise ValueError(
                    "Monthly aggregation inputs have inconsistent quality-layer metadata."
                )
            quality_applied = tags.get("quality_filter_applied") == "true"
            if quality_layers and not quality_applied and not allow_unfiltered_quality:
                raise ValueError(
                    f"{source_path} references quality layers that have not been "
                    "applied. Configure a documented QA rule or explicitly allow "
                    "unfiltered aggregation."
                )
            values = np.ma.masked_invalid(
                source.read(1, masked=True).astype(np.float32)
            )
        observations.append(
            {
                "date": observation_date,
                "path": source_path,
                "values": values,
                "quality_filter_applied": not quality_layers or quality_applied,
            }
        )

    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for observation in observations:
        groups[observation["date"].strftime("%Y-%m")].append(observation)
    first_date = min(item["date"] for item in observations)
    last_date = max(item["date"] for item in observations)
    start_month = min(
        date(first_date.year, first_date.month, 1),
        date(requested_start.year, requested_start.month, 1)
        if requested_start
        else date(first_date.year, first_date.month, 1),
    )
    end_month = max(
        date(last_date.year, last_date.month, 1),
        date(requested_end.year, requested_end.month, 1)
        if requested_end
        else date(last_date.year, last_date.month, 1),
    )
    months = _months_in_range(start_month, end_month)
    destination_dir.mkdir(parents=True, exist_ok=True)
    periods = []
    for month in months:
        items = groups.get(month, [])
        if not items:
            periods.append({"month": month, "status": "missing", "observation_count": 0})
            continue
        if len(items) > np.iinfo(np.uint16).max:
            raise ValueError(
                f"Month {month} exceeds the UInt16 per-pixel observation-count limit."
            )
        stack = np.ma.stack([item["values"] for item in items])
        if method == "mean":
            result = np.ma.mean(stack, axis=0)
        elif method == "median":
            result = np.ma.median(stack, axis=0)
        elif method == "min":
            result = np.ma.min(stack, axis=0)
        else:
            result = np.ma.max(stack, axis=0)
        count = np.ma.count(stack, axis=0).astype(np.uint16)
        quality_filter_applied = all(
            item["quality_filter_applied"] for item in items
        )
        output_path, count_path = _write_monthly_rasters(
            destination_dir,
            month,
            result,
            count,
            reference_grid,
            method,
            quality_filter_applied,
            reference_identity,
            reference_quality_layers or [],
            allow_unfiltered_quality and not quality_filter_applied,
        )
        periods.append(
            {
                "month": month,
                "status": "aggregated",
                "observation_count": len(items),
                "input_dates": [item["date"].isoformat() for item in items],
                "output_path": str(output_path),
                "count_path": str(count_path),
                "method": method,
                "quality_filter_applied": quality_filter_applied,
                "unfiltered_quality_override": (
                    allow_unfiltered_quality
                    and not quality_filter_applied
                ),
            }
        )
    result = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "task_name": task_name,
        "task_id": task_id,
        "source_manifest": str(standardization_manifest),
        "aggregation": "calendar_month",
        "method": method,
        "input_paths": [str(item["path"]) for item in observations],
        "periods": periods,
        "missing_period_count": sum(item["status"] == "missing" for item in periods),
        "interpolation": "none",
    }
    manifest_path = write_manifest(
        destination_dir / "monthly_aggregation_manifest.json",
        result,
    )
    result["manifest_path"] = str(manifest_path)
    return result


def write_temporal_inventory(
    inventory: dict[str, Any],
    *,
    json_path: str | Path,
    report_path: str | Path,
) -> tuple[Path, Path]:
    json_destination = Path(json_path)
    report_destination = Path(report_path)
    json_destination.parent.mkdir(parents=True, exist_ok=True)
    report_destination.parent.mkdir(parents=True, exist_ok=True)
    write_manifest(json_destination, inventory)
    report_destination.write_text(_format_temporal_report(inventory), encoding="utf-8")
    return json_destination, report_destination


def _write_monthly_rasters(
    output_dir: Path,
    month: str,
    values: np.ma.MaskedArray,
    count: np.ndarray,
    grid: tuple[Any, ...] | None,
    method: str,
    quality_filter_applied: bool,
    source_identity: tuple[str, str, str] | None,
    quality_layers: list[str],
    unfiltered_quality_override: bool,
) -> tuple[Path, Path]:
    if grid is None:
        raise ValueError("No reference raster grid was established.")
    crs, transform, width, height, units = grid
    output_path = output_dir / f"{month}-{method}.tif"
    count_path = output_dir / f"{month}-observation-count.tif"
    result = values.filled(np.nan).astype(np.float32)
    result[count == 0] = np.nan
    profile = {
        "driver": "GTiff",
        "width": width,
        "height": height,
        "count": 1,
        "crs": crs,
        "transform": transform,
        "dtype": "float32",
        "nodata": float("nan"),
        "compress": "deflate",
    }
    mask = count > 0
    with rasterio.Env(GDAL_TIFF_INTERNAL_MASK=True):
        with rasterio.open(output_path, "w", **profile) as output:
            output.write(result, 1)
            output.write_mask(mask.astype(np.uint8) * 255)
            if units:
                output.set_band_unit(1, units)
            output.update_tags(
                temporal_aggregation="calendar_month",
                aggregation_method=method,
                aggregation_period=month,
                interpolation="none",
                quality_filter_applied=str(quality_filter_applied).lower(),
                unfiltered_quality_override=str(unfiltered_quality_override).lower(),
                source_product_id=source_identity[0] if source_identity else "",
                source_layer_id=source_identity[1] if source_identity else "",
                standardized_units=source_identity[2] if source_identity else "",
                quality_layers=json.dumps(quality_layers),
            )
    count_profile = {**profile, "dtype": "uint16", "nodata": 0}
    with rasterio.Env(GDAL_TIFF_INTERNAL_MASK=True):
        with rasterio.open(count_path, "w", **count_profile) as output:
            output.write(count, 1)
            output.write_mask(mask.astype(np.uint8) * 255)
            output.update_tags(
                measure="number of valid observations per pixel",
                aggregation_period=month,
            )
    return output_path, count_path


def _load_product_metadata(metadata_dir: Path) -> dict[str, dict[str, Any]]:
    metadata = {}
    for path in sorted(metadata_dir.glob("*.json")):
        if path.name in {"availability_report.json", "dataset_inventory.json"}:
            continue
        content = _read_json(path)
        for product in content.get("available_products", []):
            catalog = product.get("catalog", {})
            product_id = catalog.get("ProductAndVersion")
            if isinstance(product_id, str):
                metadata[product_id] = product
    return metadata


def _raster_details(path: Path) -> dict[str, Any]:
    with rasterio.open(path) as source:
        valid_pixel_count = 0
        masked_pixel_count = 0
        for _, window in source.block_windows(1):
            mask = source.read_masks(1, window=window)
            valid_pixel_count += int(np.count_nonzero(mask))
            masked_pixel_count += int(mask.size - np.count_nonzero(mask))
        return {
            "width": source.width,
            "height": source.height,
            "crs": source.crs.to_string() if source.crs else None,
            "resolution": list(source.res),
            "valid_pixel_count": valid_pixel_count,
            "masked_pixel_count": masked_pixel_count,
        }


def _acquisition_date(path: Path) -> date | None:
    match = _DATE_IN_NAME.search(path.name)
    if not match:
        return None
    try:
        return datetime.strptime(match.group(1), "%Y%m%d").date()
    except ValueError:
        return None


def _cadence_days(value: Any) -> int | None:
    if not isinstance(value, str):
        return None
    if "daily" in value.casefold():
        return 1
    match = _CADENCE_DAYS.search(value)
    return int(match.group(1)) if match else None


def _missing_cadence_slots(
    dates: list[date],
    cadence_days: int | None,
) -> list[str]:
    if cadence_days is None or cadence_days <= 0 or len(dates) < 2:
        return []
    missing = []
    unique_dates = sorted(set(dates))
    for current, following in zip(unique_dates, unique_dates[1:]):
        expected = current + timedelta(days=cadence_days)
        while expected < following:
            missing.append(expected.isoformat())
            expected += timedelta(days=cadence_days)
    return missing


def _months_in_range(start: date, end: date) -> list[str]:
    months = []
    current = date(start.year, start.month, 1)
    last = date(end.year, end.month, 1)
    while current <= last:
        months.append(current.strftime("%Y-%m"))
        current = date(current.year + (current.month == 12), current.month % 12 + 1, 1)
    return months


def _relative_path(path: Path, root: Path) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return path.resolve().as_posix()


def _read_json(path: Path) -> Any:
    with path.open(encoding="utf-8") as file:
        return json.load(file)


def _format_temporal_report(inventory: dict[str, Any]) -> str:
    lines = [
        "# Temporal Data Inventory",
        "",
        f"- Generated: {inventory['generated_at']}",
        f"- Series: {inventory['series_count']}",
        f"- Dated raster observations: {inventory['observation_count']}",
        "",
    ]
    for series in inventory["series"]:
        lines.extend(
            [
                f"## {series['product_id']} / `{series['layer_id']}`",
                "",
                f"- Product cadence metadata: {series['temporal_granularity'] or 'not recorded'}",
                f"- Requested date ranges: {series['requested_date_ranges'] or 'not recorded'}",
                f"- Dated observations: {series['observation_count']}",
                f"- Missing files: {series['missing_file_count']}",
                f"- Missing expected cadence slots between observations: "
                f"{series['missing_cadence_slots'] or 'none detected'}",
            ]
        )
        for observation in series["observations"]:
            lines.append(
                f"- {observation['date'] or 'date unknown'}: `{observation['path']}`, "
                f"{'present' if observation['exists'] else 'MISSING'}."
            )
        lines.append("")
    lines.extend(["## Temporal policy", "", inventory["policy"], ""])
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Inventory actual raster observation dates without interpolation."
    )
    task_selection = parser.add_mutually_exclusive_group(required=True)
    task_selection.add_argument("--task-name")
    task_selection.add_argument("--task-id")
    parser.add_argument("--tasks-directory", default="data/tasks")
    parser.add_argument("--metadata-dir", default="data/metadata")
    args = parser.parse_args()
    task = TaskRegistry(args.tasks_directory).resolve(args.task_name or args.task_id)
    task_name = task["task_name"]
    task_id = task["task_id"]
    task_root = task["task_directory"]
    inventory = build_temporal_inventory(
        task_root,
        task_name=task_name,
        task_id=task_id,
        metadata_dir=args.metadata_dir,
    )
    json_path, report_path = write_temporal_inventory(
        inventory,
        json_path=task_root / "processed" / "temporal_inventory.json",
        report_path=task_root / "processed" / "temporal_inventory.md",
    )
    print(f"Temporal inventory: {json_path}")
    print(f"Report: {report_path}")
    print(
        f"Found {inventory['series_count']} series and "
        f"{inventory['observation_count']} dated observations."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
