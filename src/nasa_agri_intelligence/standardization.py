from __future__ import annotations

import argparse
import ast
import json
import logging
import math
import os
import re
import secrets
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import rasterio
from pyproj import CRS, Geod

from .collection.manifest import write_manifest
from .collection.task_registry import TaskRegistry

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class LayerMetadata:
    product_id: str
    layer_id: str
    scale_factor: float | None
    add_offset: float
    fill_values: tuple[float, ...]
    raw_valid_min: float | None
    raw_valid_max: float | None
    valid_min: float | None
    valid_max: float | None
    units: str | None
    is_quality: bool
    quality_layers: tuple[str, ...]
    description: str | None
    metadata_file: str
    auxiliary_layers: tuple[str, ...] = ()

    @classmethod
    def from_appeears(
        cls,
        product_id: str,
        layer_id: str,
        layer: dict[str, Any],
        metadata_file: Path,
    ) -> LayerMetadata:
        scale_factor = _number(layer.get("ScaleFactor"))
        offset = _number(layer.get("AddOffset"))
        fill_values = layer.get("FillValueAll")
        if not isinstance(fill_values, list):
            fill_values = [layer.get("FillValue")]
        quality_layers = layer.get("QualityLayers")
        if isinstance(quality_layers, str):
            try:
                quality_layers = ast.literal_eval(quality_layers) if quality_layers else []
            except (SyntaxError, ValueError) as error:
                raise ValueError(
                    f"Invalid QualityLayers metadata for {product_id}/{layer_id}."
                ) from error
        if not isinstance(quality_layers, list) or not all(
            isinstance(value, str) for value in quality_layers
        ):
            raise ValueError(f"Invalid QualityLayers metadata for {product_id}/{layer_id}.")
        auxiliary_layers = (
            tuple(quality_layers)
            if product_id == "ASTGTM_NC.003" and layer_id == "ASTER_GDEM_DEM"
            else ()
        )
        if auxiliary_layers:
            quality_layers = []
        units = layer.get("Units")
        return cls(
            product_id=product_id,
            layer_id=layer_id,
            scale_factor=scale_factor,
            add_offset=offset or 0.0,
            fill_values=tuple(
                value for item in fill_values if (value := _number(item)) is not None
            ),
            raw_valid_min=_number(layer.get("OrigValidMin")),
            raw_valid_max=_number(layer.get("OrigValidMax")),
            valid_min=_number(layer.get("ValidMin")),
            valid_max=_number(layer.get("ValidMax")),
            units=units if isinstance(units, str) and units else None,
            is_quality=layer.get("IsQA") is True,
            quality_layers=tuple(quality_layers),
            description=(
                layer.get("Description")
                if isinstance(layer.get("Description"), str)
                else None
            ),
            metadata_file=str(metadata_file),
            auxiliary_layers=auxiliary_layers,
        )


def standardize_raster(
    source_path: str | Path,
    destination_path: str | Path,
    metadata: LayerMetadata,
    *,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Apply AppEEARS scale/offset and validity metadata; preserve native grid and mask."""

    source_path = Path(source_path)
    destination_path = Path(destination_path)
    if destination_path.exists() and not overwrite:
        raise FileExistsError(
            f"Standardized output already exists: {destination_path}. "
            "Use overwrite=True to replace it."
        )
    destination_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = _temporary_raster_path(destination_path)

    try:
        with rasterio.open(source_path) as source:
            if source.count != 1:
                raise ValueError(
                    f"Expected a single-layer raster for {metadata.product_id}/"
                    f"{metadata.layer_id}; found {source.count} bands."
                )
            raw = source.read(1)
            valid_mask = source.read_masks(1) > 0
            for fill_value in metadata.fill_values:
                valid_mask &= raw != fill_value

            use_raw_range = (
                metadata.raw_valid_min is not None
                and metadata.raw_valid_max is not None
            )
            if use_raw_range:
                valid_mask &= raw >= metadata.raw_valid_min
                valid_mask &= raw <= metadata.raw_valid_max

            effective_scale = (
                None if metadata.is_quality else metadata.scale_factor
            )
            effective_offset = 0.0 if metadata.is_quality else metadata.add_offset
            if effective_scale is not None or effective_offset != 0:
                effective_scale = 1.0 if effective_scale is None else effective_scale
                _validate_raster_scale_tags(source, effective_scale, effective_offset)
                converted = raw.astype(np.float32) * effective_scale + effective_offset
            else:
                converted = raw.copy()

            if not use_raw_range:
                if metadata.valid_min is not None:
                    valid_mask &= converted >= metadata.valid_min
                if metadata.valid_max is not None:
                    valid_mask &= converted <= metadata.valid_max

            if np.issubdtype(converted.dtype, np.floating):
                converted = converted.astype(np.float32, copy=False)
                converted[~valid_mask] = np.nan
                output_nodata: float | int | None = float("nan")
            else:
                output_nodata = source.nodata
                if output_nodata is None and metadata.fill_values:
                    output_nodata = metadata.fill_values[0]
                if output_nodata is not None:
                    converted[~valid_mask] = output_nodata

            profile = source.profile.copy()
            profile.pop("blockxsize", None)
            profile.pop("blockysize", None)
            profile.update(
                driver="GTiff",
                dtype=converted.dtype,
                nodata=output_nodata,
                compress="deflate",
            )
            raster_info = {
                "crs": source.crs.to_string() if source.crs else None,
                "transform": list(source.transform),
                "resolution": list(source.res),
                "shape": [source.height, source.width],
            }
            with rasterio.Env(GDAL_TIFF_INTERNAL_MASK=True):
                with rasterio.open(temporary_path, "w", **profile) as output:
                    output.write(converted, 1)
                    output.write_mask(valid_mask.astype(np.uint8) * 255)
                    output.set_band_description(1, metadata.layer_id)
                    output.scales = (1.0,)
                    output.offsets = (0.0,)
                    if metadata.units:
                        output.set_band_unit(1, metadata.units)
                    output.update_tags(
                        **_without_scale_tags(source.tags())
                    )
                    output.update_tags(
                        1,
                        **_without_scale_tags(source.tags(1)),
                    )
                    output.update_tags(
                        source_product_id=metadata.product_id,
                        source_layer_id=metadata.layer_id,
                        is_quality=str(metadata.is_quality).lower(),
                        variable_kind=(
                            "quality"
                            if metadata.is_quality
                            else (
                                "continuous"
                                if np.issubdtype(converted.dtype, np.floating)
                                else "categorical"
                            )
                        ),
                        source_metadata_file=metadata.metadata_file,
                        original_units=metadata.units or "",
                        standardized_units=metadata.units or "",
                        raw_scale_factor=(
                            "" if effective_scale is None else str(effective_scale)
                        ),
                        raw_add_offset=str(effective_offset),
                        transform=(
                            "identity; QA layer preserved"
                            if metadata.is_quality
                            else _transform_description(effective_scale, effective_offset)
                        ),
                        quality_layers=json.dumps(metadata.quality_layers),
                        auxiliary_layers=json.dumps(metadata.auxiliary_layers),
                        quality_filter_applied="false",
                    )
        temporary_path.replace(destination_path)
    finally:
        temporary_path.unlink(missing_ok=True)

    return {
        "source_path": str(source_path),
        "output_path": str(destination_path),
        "product_id": metadata.product_id,
        "layer_id": metadata.layer_id,
        "is_quality": metadata.is_quality,
        "variable_kind": (
            "quality"
            if metadata.is_quality
            else (
                "continuous"
                if np.issubdtype(converted.dtype, np.floating)
                else "categorical"
            )
        ),
        "source_metadata_file": metadata.metadata_file,
        "source_units": metadata.units,
        "output_units": metadata.units,
        "scale_factor_applied": effective_scale,
        "add_offset_applied": effective_offset,
        "fill_values_masked": list(metadata.fill_values),
        "raw_valid_range": (
            [metadata.raw_valid_min, metadata.raw_valid_max]
            if use_raw_range
            else None
        ),
        "physical_valid_range": (
            [metadata.valid_min, metadata.valid_max]
            if not use_raw_range
            else None
        ),
        "quality_layers": list(metadata.quality_layers),
        "auxiliary_layers": list(metadata.auxiliary_layers),
        "quality_filter_applied": False,
        "valid_pixel_count": int(valid_mask.sum()),
        "masked_pixel_count": int(valid_mask.size - valid_mask.sum()),
        **raster_info,
    }


def apply_quality_filter(
    values: np.ndarray | np.ma.MaskedArray,
    quality_values: np.ndarray | np.ma.MaskedArray,
    *,
    accepted_values: set[int],
) -> np.ma.MaskedArray:
    """Mask data unless its aligned categorical QA value is explicitly accepted."""

    if not accepted_values:
        raise ValueError("accepted_values must contain at least one QA value.")
    data = np.ma.asarray(values)
    quality = np.ma.asarray(quality_values)
    if data.shape != quality.shape:
        raise ValueError(
            f"Data and QA arrays must align; received {data.shape} and {quality.shape}."
        )
    mask = np.ma.getmaskarray(data) | np.ma.getmaskarray(quality)
    mask |= ~np.isin(np.ma.getdata(quality), list(accepted_values))
    return np.ma.array(np.ma.getdata(data), mask=mask)


def product_quality_mask(
    product_id: str,
    layer_id: str,
    quality_values: np.ndarray | np.ma.MaskedArray,
) -> np.ndarray:
    """Return accepted pixels using documented product-specific QA bit fields."""
    quality = np.ma.asarray(quality_values)
    values = np.asarray(np.ma.getdata(quality), dtype=np.uint64)
    valid = ~np.ma.getmaskarray(quality)
    if product_id == "MOD13Q1.061" and layer_id == "_250m_16_days_VI_Quality":
        # MODLAND must be good; VI usefulness 0 (highest) or 1 (lower).
        valid &= (values & 0b11) == 0
        valid &= ((values >> 2) & 0b1111) <= 1
    elif product_id == "MOD11A2.061" and layer_id in {"QC_Day", "QC_Night"}:
        # Require mandatory/data QA good and both reported errors <= 2 K / 0.02.
        valid &= (values & 0b11) == 0
        valid &= ((values >> 2) & 0b11) == 0
        valid &= ((values >> 4) & 0b11) <= 1
        valid &= ((values >> 6) & 0b11) <= 1
    elif product_id == "MOD16A2GF.061" and layer_id == "ET_QC_500m":
        # Accept the main algorithm, with or without saturation; gap-filled
        # meteorological inputs are part of this product's design.
        valid &= (values & 0b1) == 0
        valid &= ((values >> 5) & 0b111) <= 1
    elif product_id == "MCD12Q1.061" and layer_id == "QC":
        # Keep classified/corrected labels; reject unclassified and no-data.
        valid &= np.isin(values, [0, 2, 4, 5, 6, 7, 8, 9])
    else:
        raise ValueError(f"No documented QA acceptance rule for {product_id}/{layer_id}.")
    return valid


_QUALITY_POLICIES = {
    ("MOD13Q1.061", "_250m_16_days_VI_Quality"): (
        "MODLAND QA == good and VI usefulness <= 1 "
        "(MOD13Q1 Collection 6.1 User Guide, Table 5)."
    ),
    ("MOD11A2.061", "QC_Day"): (
        "Mandatory and data-quality flags == good; emissivity and LST error "
        "flags <= 1 (MOD11 User Guide, Table 13)."
    ),
    ("MOD11A2.061", "QC_Night"): (
        "Mandatory and data-quality flags == good; emissivity and LST error "
        "flags <= 1 (MOD11 User Guide, Table 13)."
    ),
    ("MOD16A2GF.061", "ET_QC_500m"): (
        "MODLAND QA == main algorithm and SCF_QC <= 1; cloud-state bits are "
        "retained because MOD16A2GF is gap-filled (MOD16 User Guide, section 6.2.1)."
    ),
    ("MCD12Q1.061", "QC"): (
        "Retain classified or corrected labels (QA 0, 2, 4-9); reject "
        "unclassified/no-data labels 1, 3, 10 (MCD12 Collection 6.1 User Guide, Table 11)."
    ),
}


def apply_task_quality_filters(
    task_directory: str | Path,
    records: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Write task-scoped QA-filtered rasters when a matching companion exists."""
    task_root = Path(task_directory).resolve()
    by_observation: dict[tuple[str, str, str, str], dict[str, Any]] = {}
    for record in records:
        if record.get("variable_kind") != "quality":
            continue
        source_path = _task_record_path(task_root, record.get("output_path"))
        match = re.search(r"_(\d{8}T\d{6})_(aid\d+)", source_path.name)
        if match:
            by_observation[
                (
                    str(record.get("product_id")),
                    str(record.get("layer_id")),
                    match[1],
                    match[2],
                )
            ] = record

    filtered_records: list[dict[str, Any]] = []
    for record in records:
        if record.get("variable_kind") == "quality":
            filtered_records.append(record)
            continue
        if (
            record.get("quality_filter_applied") is True
            and record.get("quality_status") == "filtered"
        ):
            filtered_records.append(record)
            continue
        quality_layers = record.get("quality_layers", [])
        if not quality_layers:
            filtered_records.append(record)
            continue
        source_path = _task_record_path(task_root, record.get("output_path"))
        match = re.search(r"_(\d{8}T\d{6})_(aid\d+)", source_path.name)
        if not match:
            missing = dict(record)
            missing["quality_filter_applied"] = False
            missing["quality_status"] = "observation_date_unavailable"
            filtered_records.append(missing)
            continue

        companions = [
            by_observation.get(
                (str(record.get("product_id")), layer_id, match[1], match[2])
            )
            for layer_id in quality_layers
        ]
        if any(companion is None for companion in companions):
            missing = dict(record)
            missing["quality_filter_applied"] = False
            missing["quality_status"] = "missing_quality_companion"
            missing["quality_missing_layers"] = [
                layer_id
                for layer_id, companion in zip(quality_layers, companions)
                if companion is None
            ]
            filtered_records.append(missing)
            continue

        policy_keys = [
            (str(record.get("product_id")), str(companion["layer_id"]))
            for companion in companions
            if companion is not None
        ]
        if any(key not in _QUALITY_POLICIES for key in policy_keys):
            unsupported = dict(record)
            unsupported["quality_filter_applied"] = False
            unsupported["quality_status"] = "unsupported_quality_policy"
            filtered_records.append(unsupported)
            continue

        output_path = source_path
        output_path_rel = output_path.relative_to(task_root)
        for companion, (_, qa_layer_id) in zip(companions, policy_keys):
            if companion is None:
                raise ValueError(
                    "A required quality companion disappeared during filtering."
                )
            quality_path = _task_record_path(task_root, companion.get("output_path"))
            destination_path = (
                task_root
                / "processed"
                / "rasters"
                / "quality_filtered"
                / f"{output_path.stem}.qa-filtered.tif"
            )
            accepted_count, rejected_count = _write_quality_filtered_raster(
                output_path,
                quality_path,
                destination_path,
                product_id=str(record["product_id"]),
                quality_layer=qa_layer_id,
                policy=_QUALITY_POLICIES[(str(record["product_id"]), qa_layer_id)],
            )
            output_path = destination_path
            output_path_rel = destination_path.relative_to(task_root)

        filtered = dict(record)
        filtered.update(
            {
                "unfiltered_output_path": record["output_path"],
                "output_path": output_path_rel.as_posix(),
                "quality_filter_applied": True,
                "quality_status": "filtered",
                "quality_policy": [
                    _QUALITY_POLICIES[key] for key in policy_keys
                ],
                "quality_accepted_pixel_count": accepted_count,
                "quality_rejected_pixel_count": rejected_count,
            }
        )
        filtered_records.append(filtered)
    return filtered_records


def _temporary_raster_path(destination_path: Path) -> Path:
    for _ in range(10):
        temporary_path = destination_path.with_name(
            f".{destination_path.name}.{secrets.token_hex(8)}.tmp"
        )
        try:
            descriptor = os.open(
                temporary_path,
                os.O_CREAT | os.O_EXCL | os.O_WRONLY,
                0o666,
            )
        except FileExistsError:
            continue
        os.close(descriptor)
        return temporary_path
    raise FileExistsError(
        f"Could not allocate a unique temporary raster beside {destination_path}."
    )


def _task_record_path(task_root: Path, path_value: Any) -> Path:
    if not isinstance(path_value, str) or not path_value:
        raise ValueError("A task raster record is missing its output path.")
    path = Path(path_value)
    if not path.is_absolute():
        path = task_root / path
    resolved = path.resolve()
    if not resolved.is_relative_to(task_root):
        raise ValueError(f"Task raster path is outside its task directory: {path_value}")
    if not resolved.is_file():
        raise FileNotFoundError(f"Task raster is missing: {resolved}")
    return resolved


def _write_quality_filtered_raster(
    source_path: Path,
    quality_path: Path,
    destination_path: Path,
    *,
    product_id: str,
    quality_layer: str,
    policy: str,
) -> tuple[int, int]:
    with rasterio.open(source_path) as source, rasterio.open(quality_path) as quality:
        if source.count != 1 or quality.count != 1:
            raise ValueError("QA filtering requires single-band science and QA rasters.")
        if (
            source.shape != quality.shape
            or source.transform != quality.transform
            or source.crs != quality.crs
        ):
            raise ValueError(
                f"Science and QA rasters do not share an identical grid: "
                f"{source_path.name} and {quality_path.name}."
            )
        values = source.read(1, masked=True)
        qa_values = quality.read(1, masked=True)
        accepted = product_quality_mask(product_id, quality_layer, qa_values)
        data_mask = np.ma.getmaskarray(values)
        quality_mask = np.ma.getmaskarray(qa_values)
        combined_mask = data_mask | quality_mask | ~accepted
        filtered = np.ma.array(np.ma.getdata(values), mask=combined_mask)
        accepted_count = int((~combined_mask).sum())
        rejected_count = int((~data_mask & (quality_mask | ~accepted)).sum())
        profile = source.profile.copy()
        profile.pop("blockxsize", None)
        profile.pop("blockysize", None)
        profile.update(compress="deflate")
        destination_path.parent.mkdir(parents=True, exist_ok=True)
        temporary_path = _temporary_raster_path(destination_path)
        try:
            with rasterio.Env(GDAL_TIFF_INTERNAL_MASK=True):
                with rasterio.open(temporary_path, "w", **profile) as output:
                    fill_value = source.nodata
                    if fill_value is None:
                        fill_value = float("nan") if np.issubdtype(
                            filtered.dtype, np.floating
                        ) else 0
                    output.write(filtered.filled(fill_value), 1)
                    output.write_mask((~combined_mask).astype(np.uint8) * 255)
                    output.set_band_description(1, source.descriptions[0] or "")
                    output.scales = source.scales
                    output.offsets = source.offsets
                    if source.units[0]:
                        output.set_band_unit(1, source.units[0])
                    output.update_tags(**source.tags())
                    output.update_tags(
                        quality_filter_applied="true",
                        quality_status="filtered",
                        quality_layer_id=quality_layer,
                        quality_policy=policy,
                        quality_accepted_pixel_count=str(accepted_count),
                        quality_rejected_pixel_count=str(rejected_count),
                    )
            temporary_path.replace(destination_path)
        finally:
            temporary_path.unlink(missing_ok=True)
    return accepted_count, rejected_count


def standardize_collection(
    *,
    task_name: str,
    task_id: str,
    tasks_directory: str | Path = "data/tasks",
    metadata_directory: str | Path = "data/metadata",
    overwrite: bool = False,
) -> dict[str, Any]:
    """Standardize one registry-selected task; global raw-tree discovery is disabled."""
    registry = TaskRegistry(tasks_directory)
    task = registry.resolve(task_name)
    if task.get("task_id") != task_id:
        raise ValueError("Selected NASA task ID does not match the task registry.")
    return standardize_task(
        task["task_directory"],
        task_name=task_name,
        task_id=task_id,
        metadata_directory=metadata_directory,
        overwrite=overwrite,
    )


def standardize_task(
    task_directory: str | Path,
    *,
    task_name: str,
    task_id: str,
    metadata_directory: str | Path | None = None,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Standardize only validated downloads owned by one registered NASA task."""
    task_root = Path(task_directory).resolve()
    registry = TaskRegistry(task_root.parent)
    task, task_manifest_path, task_manifest = registry.load_task_manifest(task_name)
    if task["task_directory"] != task_root or task.get("task_id") != task_id:
        raise ValueError("Selected task identity does not match the requested standardization.")
    if task_manifest.get("download_status") != "completed":
        raise ValueError("Task downloads must be complete before standardization.")

    metadata_dir = (
        Path(metadata_directory).resolve()
        if metadata_directory is not None
        else task_root.parents[1] / "metadata"
    )
    metadata_index = _load_metadata_index(metadata_dir)
    downloads = task_manifest.get("downloads")
    if not isinstance(downloads, list) or not downloads:
        raise ValueError(f"Task {task_name} has no download records to standardize.")

    manifest_path = task_root / "processed" / "standardized_manifest.json"
    previous_external_records: list[dict[str, Any]] = []
    if manifest_path.is_file():
        _, previous_manifest, _ = registry.validate_raster_manifest(
            task_name, manifest_path
        )
        previous_records = previous_manifest.get(
            "rasters", previous_manifest.get("records", [])
        )
        if isinstance(previous_records, list):
            previous_external_records = [
                dict(record)
                for record in previous_records
                if isinstance(record, dict)
                and record.get("external_source") in {"NASA/POWER", "ISRIC/SoilGrids"}
            ]

    records: list[dict[str, Any]] = []
    seen_sources: set[Path] = set()
    for download in downloads:
        if not isinstance(download, dict) or download.get("status") != "validated":
            raise ValueError(f"Task {task_name} has an unvalidated download record.")
        raw_value = download.get("local_path")
        if not isinstance(raw_value, str):
            raise ValueError(f"Task {task_name} has a download record without a local path.")
        source_path = Path(raw_value)
        if not source_path.is_absolute():
            source_path = task_root / source_path
        source_path = source_path.resolve()
        if not source_path.is_relative_to((task_root / "raw").resolve()):
            raise ValueError(f"Task download is outside its raw data directory: {raw_value}")
        if source_path in seen_sources:
            raise ValueError(f"Duplicate task download record: {source_path}")
        seen_sources.add(source_path)
        if not source_path.is_file():
            raise FileNotFoundError(f"Task download is missing: {source_path}")
        if source_path.suffix.casefold() not in {".tif", ".tiff"}:
            continue

        try:
            product_id, product_metadata, metadata_file = _find_product(
                source_path, metadata_index
            )
            layer_id = _find_layer(source_path, product_metadata)
            metadata = LayerMetadata.from_appeears(
                product_id,
                layer_id,
                product_metadata["layers"][layer_id],
                metadata_file,
            )
        except ValueError as error:
            quality_metadata = _find_quality_companion(
                source_path,
                task_manifest,
                metadata_index,
            )
            if quality_metadata is None:
                raise error
            product_id, layer_id, metadata = quality_metadata
        relative_source = source_path.relative_to(task_root)
        destination_path = task_root / "processed" / "rasters" / relative_source
        record = standardize_raster(
            source_path,
            destination_path,
            metadata,
            overwrite=overwrite,
        )
        record.update(
            {
                "source_path": relative_source.as_posix(),
                "output_path": destination_path.relative_to(task_root).as_posix(),
                "product_id": product_id,
                "layer_id": layer_id,
            }
        )
        if metadata.description == "Quality companion without product metadata":
            record["metadata_status"] = (
                "companion_product_metadata_unavailable; raw values preserved"
            )
        records.append(record)

    if not records:
        raise ValueError(f"Task {task_name} contains no supported GeoTIFF downloads.")
    records = apply_task_quality_filters(task_root, records)
    dem_records = [
        record
        for record in records
        if record.get("variable_kind") != "quality"
        and record.get("is_quality") is not True
        and (
            "dem" in str(record.get("layer_id", "")).casefold()
            or "elevation" in str(record.get("layer_id", "")).casefold()
        )
    ]
    derived_layer_statuses: list[dict[str, Any]] = []
    for dem_record in dem_records:
        dem_path = task_root / str(dem_record["output_path"])
        slope_path = (
            task_root
            / "processed"
            / "rasters"
            / "derived"
            / f"slope-{dem_path.stem}.tif"
        )
        slope_record = derive_slope_raster(
            dem_path,
            slope_path,
            source_product_id=str(dem_record["product_id"]),
            source_layer_id=str(dem_record["layer_id"]),
            overwrite=overwrite,
        )
        if slope_record is not None:
            slope_record["source_path"] = dem_record["source_path"]
            records.append(slope_record)
        else:
            derived_layer_statuses.append(
                {
                    "product_id": "_DERIVED",
                    "layer_id": "slope",
                    "source_product_id": dem_record["product_id"],
                    "source_layer_id": dem_record["layer_id"],
                    "status": "insufficient_dem_data",
                    "detail": (
                        "A valid slope surface requires at least a 3×3 neighborhood "
                        "of valid DEM pixels."
                    ),
                }
            )
    if not dem_records:
        derived_layer_statuses.append(
            {
                "product_id": "_DERIVED",
                "layer_id": "slope",
                "status": "source_dem_not_selected",
                "detail": "Select an elevation DEM product to derive slope.",
            }
        )

    current_external_keys = {
        (
            record.get("product_id"),
            record.get("layer_id"),
            record.get("temporal_start"),
            record.get("temporal_end"),
        )
        for record in records
        if record.get("external_source")
    }
    for record in previous_external_records:
        key = (
            record.get("product_id"),
            record.get("layer_id"),
            record.get("temporal_start"),
            record.get("temporal_end"),
        )
        if key not in current_external_keys:
            records.append(record)
            current_external_keys.add(key)

    output_directory = task_root / "processed" / "rasters"
    created_at = datetime.now(timezone.utc).isoformat()
    result = {
        "task_name": task_name,
        "task_id": task_id,
        "created_at": created_at,
        "generated_at": created_at,
        "source_directory": str(task_root / "raw"),
        "output_directory": str(output_directory),
        "raster_count": len(records),
        "rasters": records,
        "records": records,
        "derived_layer_statuses": derived_layer_statuses,
        "quality_policy": (
            "QA layers are retained and aligned but not applied to science layers "
            "until a product-specific acceptance rule is explicitly configured."
        ),
    }
    write_manifest(manifest_path, result)
    registry.validate_raster_manifest(task_name, manifest_path)
    task_manifest["preprocessing_status"] = "completed"
    task_manifest["standardized_manifest_path"] = str(manifest_path)
    write_manifest(task_manifest_path, task_manifest)
    registry.update(task_name, preprocessing_status="completed")
    return result


def standardize_external_rasters(
    registry: TaskRegistry,
    identifier: str,
) -> list[dict[str, Any]]:
    """Standardize newly acquired task-owned SoilGrids rasters."""
    task, task_manifest_path, task_manifest = registry.load_task_manifest(identifier)
    if task_manifest.get("preprocessing_status") != "completed":
        raise ValueError("Task preprocessing must be complete before external standardization.")
    manifest_path = task["task_directory"] / "processed" / "standardized_manifest.json"
    _, raster_manifest, existing_records = registry.validate_raster_manifest(
        task["task_name"], manifest_path
    )
    downloads = task_manifest.get("external_downloads", [])
    if not isinstance(downloads, list):
        raise ValueError("Task external download records are invalid.")
    latest_downloads: dict[tuple[str, str], dict[str, Any]] = {}
    for record in downloads:
        if (
            isinstance(record, dict)
            and record.get("external_source") == "ISRIC/SoilGrids"
            and record.get("status") == "validated"
            and isinstance(record.get("product_id"), str)
            and isinstance(record.get("layer_id"), str)
        ):
            latest_downloads[(record["product_id"], record["layer_id"])] = record
    replacement_versions = {
        key: str(record["georeferencing_version"])
        for key, record in latest_downloads.items()
        if isinstance(record.get("georeferencing_version"), str)
    }
    replacement_keys = set(replacement_versions)
    existing_keys = {
        (record.get("product_id"), record.get("layer_id"))
        for record in existing_records
        if not (
            record.get("external_source") == "ISRIC/SoilGrids"
            and (record.get("product_id"), record.get("layer_id")) in replacement_keys
            and record.get("georeferencing_version")
            != replacement_versions[
                (record.get("product_id"), record.get("layer_id"))
            ]
        )
    }

    new_records: list[dict[str, Any]] = []
    for download in latest_downloads.values():
        product_id = download.get("product_id")
        layer_id = download.get("layer_id")
        source_value = download.get("local_path")
        if not isinstance(product_id, str) or not isinstance(layer_id, str):
            raise ValueError("SoilGrids download is missing its product or coverage ID.")
        if (product_id, layer_id) in existing_keys:
            continue
        if not isinstance(source_value, str):
            raise ValueError(f"SoilGrids download is not validated: {layer_id}")
        source_path = (task["task_directory"] / source_value).resolve()
        if not source_path.is_relative_to((task["task_directory"] / "raw").resolve()):
            raise ValueError(f"SoilGrids source raster is outside the task raw directory.")
        if not source_path.is_file():
            raise FileNotFoundError(f"SoilGrids source raster is missing: {source_path}")
        raw_range = download.get("raw_valid_range")
        valid_range = download.get("valid_range")
        scale_factor = _number(download.get("scale_factor"))
        if (
            not isinstance(raw_range, list)
            or len(raw_range) != 2
            or not isinstance(valid_range, list)
            or len(valid_range) != 2
            or scale_factor is None
        ):
            raise ValueError(f"SoilGrids standardization metadata is invalid: {layer_id}")
        metadata_path = download.get("source_metadata_file")
        if not isinstance(metadata_path, str):
            raise ValueError(f"SoilGrids provenance file is missing: {layer_id}")
        resolved_metadata_path = (task["task_directory"] / metadata_path).resolve()
        if (
            not resolved_metadata_path.is_relative_to(task["task_directory"])
            or not resolved_metadata_path.is_file()
        ):
            raise ValueError(f"SoilGrids provenance file is missing or outside the task.")
        variable = str(download.get("variable", layer_id))
        units = download.get("output_units")
        if not isinstance(units, str) or not units:
            raise ValueError(f"SoilGrids output units are missing: {layer_id}")
        metadata = LayerMetadata(
            product_id=product_id,
            layer_id=layer_id,
            scale_factor=scale_factor,
            add_offset=0.0,
            fill_values=(-32768.0,),
            raw_valid_min=_number(raw_range[0]),
            raw_valid_max=_number(raw_range[1]),
            valid_min=_number(valid_range[0]),
            valid_max=_number(valid_range[1]),
            units=units,
            is_quality=False,
            quality_layers=(),
            description=variable,
            metadata_file=str(resolved_metadata_path),
        )
        destination_path = (
            task["task_directory"]
            / "processed"
            / "rasters"
            / "soilgrids"
            / (
                f"{layer_id}_"
                f"{str(download.get('georeferencing_version', 'legacy')).replace(':', '-')}.tif"
            )
        )
        result = standardize_raster(source_path, destination_path, metadata)
        with rasterio.open(destination_path, "r+") as output:
            output.update_tags(
                external_source="ISRIC/SoilGrids",
                data_scope="task_scoped",
                task_name=task["task_name"],
                task_id=task["task_id"],
                source_version=str(download.get("source_version", "")),
                depth_interval=str(download.get("depth_interval", "")),
                quantile=str(download.get("quantile", "")),
                source_units=str(download.get("source_units", "")),
                native_resolution_m=str(download.get("spatial_resolution_m", "")),
                georeferencing_version=str(
                    download.get("georeferencing_version", "")
                ),
            )
        result.update(
            {
                key: value
                for key, value in download.items()
                if key
                not in {
                    "local_path",
                    "filename",
                    "status",
                    "width",
                    "height",
                    "crs",
                    "resolution",
                    "valid_pixel_count",
                    "masked_pixel_count",
                    "raw_valid_range",
                    "valid_range",
                    "scale_factor",
                }
            }
        )
        result.update(
            {
                "source_path": source_path.relative_to(task["task_directory"]).as_posix(),
                "output_path": destination_path.relative_to(task["task_directory"]).as_posix(),
                "product_id": product_id,
                "layer_id": layer_id,
                "external_source": "ISRIC/SoilGrids",
                "data_scope": "task_scoped",
                "task_name": task["task_name"],
                "task_id": task["task_id"],
                "source_units": str(download.get("source_units", "")),
                "output_units": units,
                "native_resolution_m": download.get("spatial_resolution_m", 250),
                "depth_interval": download.get("depth_interval"),
                "temporal_resolution": "static prediction",
            }
        )
        new_records.append(result)
        existing_keys.add((product_id, layer_id))

    if not new_records:
        return []
    superseded_records = [
        record
        for record in existing_records
        if record.get("external_source") == "ISRIC/SoilGrids"
        and (record.get("product_id"), record.get("layer_id")) in replacement_keys
        and record.get("georeferencing_version")
        != replacement_versions[(record.get("product_id"), record.get("layer_id"))]
    ]
    all_records = [
        record
        for record in existing_records
        if record not in superseded_records
    ] + new_records
    if superseded_records:
        history = raster_manifest.setdefault("superseded_rasters", [])
        history.extend(
            {
                **record,
                "status": "superseded_georeferencing",
                "superseded_at": datetime.now(timezone.utc).isoformat(),
            }
            for record in superseded_records
        )
    replacements = {
        (record.get("product_id"), record.get("layer_id"))
        for record in new_records
    }
    statuses = raster_manifest.get("derived_layer_statuses", [])
    if isinstance(statuses, list):
        raster_manifest["derived_layer_statuses"] = [
            status
            for status in statuses
            if not (
                isinstance(status, dict)
                and (status.get("product_id"), status.get("layer_id")) in replacements
            )
        ]
    raster_manifest["rasters"] = all_records
    raster_manifest["records"] = all_records
    raster_manifest["raster_count"] = len(all_records)
    raster_manifest["updated_at"] = datetime.now(timezone.utc).isoformat()
    raster_manifest.setdefault("external_integrations", []).append(
        {
            "source": "ISRIC/SoilGrids",
            "task_name": task["task_name"],
            "task_id": task["task_id"],
            "variables": [record["layer_id"] for record in new_records],
        }
    )
    write_manifest(manifest_path, raster_manifest)
    registry.validate_raster_manifest(task["task_name"], manifest_path)
    task_manifest.setdefault("standardized_external_data", []).append(
        {
            "source": "ISRIC/SoilGrids",
            "manifest_path": str(manifest_path),
            "variables": [record["layer_id"] for record in new_records],
            "standardized_at": datetime.now(timezone.utc).isoformat(),
        }
    )
    write_manifest(task_manifest_path, task_manifest)
    return new_records


def derive_slope_raster(
    dem_path: str | Path,
    destination_path: str | Path,
    *,
    source_product_id: str,
    source_layer_id: str,
    overwrite: bool = False,
) -> dict[str, Any] | None:
    """Derive slope in degrees from a DEM, preserving its task-owned grid."""
    dem_path = Path(dem_path)
    destination_path = Path(destination_path)
    if destination_path.exists() and not overwrite:
        raise FileExistsError(f"Derived slope raster already exists: {destination_path}.")

    with rasterio.open(dem_path) as source:
        if source.crs is None:
            raise ValueError(f"DEM has no CRS: {dem_path}")
        if source.width < 3 or source.height < 3:
            return None
        if not np.isclose(source.transform.b, 0) or not np.isclose(source.transform.d, 0):
            raise ValueError(f"Rotated DEM grids are not supported for slope: {dem_path}")
        elevation = source.read(1, masked=True).astype(np.float64)
        values = np.asarray(elevation.filled(np.nan), dtype=np.float64)
        valid = ~np.ma.getmaskarray(elevation) & np.isfinite(values)
        if not valid.any():
            return None

        if source.crs.is_projected:
            projected_crs = CRS.from_user_input(source.crs)
            x_factor = projected_crs.axis_info[0].unit_conversion_factor
            y_factor = projected_crs.axis_info[1].unit_conversion_factor
            if not np.isfinite(x_factor) or not np.isfinite(y_factor):
                raise ValueError(f"DEM CRS uses unsupported linear units: {source.crs}")
            dx = abs(source.transform.a) * x_factor
            dy = abs(source.transform.e) * y_factor
            dz_dx = np.gradient(values, dx, axis=1)
            dz_dy = np.gradient(values, dy, axis=0)
        elif source.crs.is_geographic:
            geod: Geod = CRS.from_user_input(source.crs).get_geod()
            center_col = source.width // 2
            row_distances = []
            for row in range(source.height):
                left = source.transform * (center_col, row + 0.5)
                right = source.transform * (center_col + 1, row + 0.5)
                row_distances.append(abs(geod.inv(*left, *right)[2]))
            column_distances = []
            for row in range(source.height - 1):
                upper = source.transform * (center_col + 0.5, row + 0.5)
                lower = source.transform * (center_col + 0.5, row + 1.5)
                column_distances.append(abs(geod.inv(*upper, *lower)[2]))
            dy_coordinates = np.concatenate(([0.0], np.cumsum(column_distances)))
            dz_dy = np.gradient(values, dy_coordinates, axis=0)
            dz_dx = np.empty_like(values)
            for row, spacing in enumerate(row_distances):
                dz_dx[row] = np.gradient(values[row], spacing)
        else:
            raise ValueError(f"DEM CRS must be projected or geographic: {source.crs}")

        slope = np.degrees(np.arctan(np.hypot(dz_dx, dz_dy))).astype(np.float32)
        # A slope estimate adjacent to a hole can be dominated by its fill value.
        valid_neighborhood = valid.copy()
        valid_neighborhood[1:, :] &= valid[:-1, :]
        valid_neighborhood[:-1, :] &= valid[1:, :]
        valid_neighborhood[:, 1:] &= valid[:, :-1]
        valid_neighborhood[:, :-1] &= valid[:, 1:]
        valid_neighborhood[~np.isfinite(slope)] = False
        slope[~valid_neighborhood] = np.nan
        profile = source.profile.copy()
        profile.pop("blockxsize", None)
        profile.pop("blockysize", None)
        profile.update(
            driver="GTiff",
            dtype="float32",
            nodata=float("nan"),
            compress="deflate",
        )
        destination_path.parent.mkdir(parents=True, exist_ok=True)
        temporary_path = _temporary_raster_path(destination_path)
        try:
            with rasterio.Env(GDAL_TIFF_INTERNAL_MASK=True):
                with rasterio.open(temporary_path, "w", **profile) as output:
                    output.write(slope, 1)
                    output.write_mask(valid_neighborhood.astype(np.uint8) * 255)
                    output.set_band_description(1, "slope")
                    output.set_band_unit(1, "degree")
                    output.update_tags(
                        source_product_id=source_product_id,
                        source_layer_id=source_layer_id,
                        variable_kind="continuous",
                        derived_from="digital_elevation_model",
                        units="degree",
                    )
            temporary_path.replace(destination_path)
        finally:
            temporary_path.unlink(missing_ok=True)
        transform = source.transform
        crs = source.crs.to_string()
        shape = [source.height, source.width]

    return {
        "source_path": str(dem_path),
        "output_path": str(destination_path),
        "product_id": "_DERIVED",
        "layer_id": "slope",
        "source_product_id": source_product_id,
        "source_layer_id": source_layer_id,
        "is_quality": False,
        "variable_kind": "continuous",
        "source_metadata_file": None,
        "source_units": "m",
        "output_units": "degree",
        "scale_factor_applied": None,
        "add_offset_applied": 0.0,
        "quality_layers": [],
        "quality_filter_applied": False,
        "derived": True,
        "valid_pixel_count": int(valid_neighborhood.sum()),
        "masked_pixel_count": int(valid_neighborhood.size - valid_neighborhood.sum()),
        "crs": crs,
        "transform": list(transform),
        "resolution": [transform.a, transform.e],
        "shape": shape,
    }


def _load_metadata_index(
    metadata_dir: Path,
) -> dict[str, tuple[dict[str, Any], Path]]:
    index = {}
    for path in sorted(metadata_dir.glob("*.json")):
        if path.name in {"availability_report.json", "dataset_inventory.json"}:
            continue
        with path.open(encoding="utf-8") as file:
            content = json.load(file)
        for product in content.get("available_products", []):
            catalog = product.get("catalog", {})
            product_id = catalog.get("ProductAndVersion")
            if isinstance(product_id, str) and isinstance(product.get("layers"), dict):
                index[product_id] = (product, path)
    return index


def _find_product(
    path: Path,
    metadata_index: dict[str, tuple[dict[str, Any], Path]],
) -> tuple[str, dict[str, Any], Path]:
    for product_id in sorted(metadata_index, key=len, reverse=True):
        if product_id in path.name or product_id.replace(".", "-") in path.name:
            product, metadata_path = metadata_index[product_id]
            return product_id, product, metadata_path
    raise ValueError(f"No saved AppEEARS product metadata matches raster {path}.")


def _find_layer(path: Path, product: dict[str, Any]) -> str:
    layers = product["layers"]
    matches = [layer_id for layer_id in layers if layer_id in path.name]
    if not matches:
        raise ValueError(
            f"No exact layer identifier in saved AppEEARS metadata matches raster {path}."
        )
    return max(matches, key=len)


def _find_quality_companion(
    path: Path,
    task_manifest: dict[str, Any],
    metadata_index: dict[str, tuple[dict[str, Any], Path]],
) -> tuple[str, str, LayerMetadata] | None:
    product_id = task_manifest.get("product_id")
    requested_layers = task_manifest.get("layers", [])
    if (
        not isinstance(product_id, str)
        or not isinstance(requested_layers, list)
        or not all(isinstance(layer, str) for layer in requested_layers)
    ):
        return None
    metadata_entry = metadata_index.get(product_id)
    if metadata_entry is None:
        return None
    product_metadata, metadata_file = metadata_entry
    layers = product_metadata.get("layers")
    if not isinstance(layers, dict):
        return None

    for requested_layer in requested_layers:
        science_metadata = layers.get(requested_layer)
        if not isinstance(science_metadata, dict):
            continue
        quality_layers = science_metadata.get("QualityLayers", [])
        if isinstance(quality_layers, str):
            try:
                quality_layers = ast.literal_eval(quality_layers) if quality_layers else []
            except (SyntaxError, ValueError):
                continue
        quality_product = science_metadata.get(
            "QualityProductAndVersion", product_id
        )
        if (
            not isinstance(quality_layers, list)
            or not isinstance(quality_product, str)
            or not quality_product
        ):
            continue
        for quality_layer in quality_layers:
            if (
                isinstance(quality_layer, str)
                and quality_layer in path.name
                and quality_product in path.name
            ):
                return (
                    quality_product,
                    quality_layer,
                    LayerMetadata(
                        product_id=quality_product,
                        layer_id=quality_layer,
                        scale_factor=None,
                        add_offset=0.0,
                        fill_values=(),
                        raw_valid_min=None,
                        raw_valid_max=None,
                        valid_min=None,
                        valid_max=None,
                        units=None,
                        is_quality=True,
                        quality_layers=(),
                        description="Quality companion without product metadata",
                        metadata_file=str(metadata_file),
                    ),
                )
    return None


def _validate_raster_scale_tags(
    source: rasterio.io.DatasetReader,
    scale_factor: float,
    add_offset: float,
) -> None:
    tags = {**source.tags(), **source.tags(1)}
    tagged_scale = _number(tags.get("scale_factor", tags.get("ScaleFactor")))
    tagged_offset = _number(tags.get("add_offset", tags.get("AddOffset")))
    if tagged_scale is not None and not math.isclose(
        tagged_scale, scale_factor, rel_tol=1e-6, abs_tol=1e-9
    ):
        raise ValueError(
            f"Raster scale tag {tagged_scale} does not match product metadata "
            f"scale {scale_factor} for {source.name}."
        )
    if tagged_offset is not None and not math.isclose(
        tagged_offset, add_offset, rel_tol=1e-6, abs_tol=1e-9
    ):
        raise ValueError(
            f"Raster offset tag {tagged_offset} does not match product metadata "
            f"offset {add_offset} for {source.name}."
        )


def _without_scale_tags(tags: dict[str, str]) -> dict[str, str]:
    excluded = {"scale_factor", "add_offset", "units"}
    return {key: value for key, value in tags.items() if key.casefold() not in excluded}


def _number(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        number = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"Expected numeric AppEEARS metadata; received {value!r}.") from error
    if not math.isfinite(number):
        raise ValueError(f"AppEEARS metadata value must be finite; received {value!r}.")
    return number


def _transform_description(
    scale_factor: float | None,
    add_offset: float,
) -> str:
    if scale_factor is None:
        return "identity; product metadata did not provide a scale factor"
    return f"physical = raw * {scale_factor} + {add_offset}"


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Standardize rasters belonging to one registered AppEEARS task."
    )
    task_group = parser.add_mutually_exclusive_group(required=True)
    task_group.add_argument("--task-name")
    task_group.add_argument("--task-id")
    parser.add_argument("--tasks-directory", default="data/tasks")
    parser.add_argument("--metadata-directory", default="data/metadata")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    registry = TaskRegistry(args.tasks_directory)
    task = registry.resolve(args.task_name or args.task_id)
    task_id = task.get("task_id")
    if not isinstance(task_id, str):
        raise ValueError("Selected task has no NASA task ID.")
    result = standardize_task(
        task["task_directory"],
        task_name=task["task_name"],
        task_id=task_id,
        metadata_directory=args.metadata_directory,
        overwrite=args.overwrite,
    )
    print(f"Task: {result['task_name']} ({result['task_id']})")
    print(f"Standardized {result['raster_count']} raster files to {result['output_directory']}")
    print(f"Manifest: {task['task_directory'] / 'processed/standardized_manifest.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
