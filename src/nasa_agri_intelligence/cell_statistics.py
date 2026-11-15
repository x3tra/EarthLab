from __future__ import annotations

import json
import logging
import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import rasterio
from pyproj import Transformer
from shapely import contains_xy
from shapely import transform as transform_geometry
from shapely.geometry import box, mapping, shape
from shapely.ops import unary_union

from .collection.manifest import write_manifest
from .collection.task_registry import TaskRegistry
from .collection.task_manager import (
    AOISettings,
    CollectionSettings,
    normalize_aoi,
)
from .spatial_harmonization import (
    GridSpec,
    build_grid,
    native_pixel_resolution_in_crs,
)

logger = logging.getLogger(__name__)

_DATE_IN_NAME = re.compile(r"_(\d{8})T\d{6}_")
_SAFE_PATH_COMPONENT = re.compile(r"[^A-Za-z0-9._-]+")
_COVERAGE_DEFINITION = (
    "valid native pixel centers divided by native pixel centers inside the AOI "
    "and cell; this is sample coverage, not exact area coverage"
)


def run_cell_statistics(
    settings: CollectionSettings,
    *,
    task_directory: str | Path,
    task_name: str,
    task_id: str,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Create AOI-clipped cells and per-raster statistics from native pixels."""

    task_root = Path(task_directory).resolve()
    registry = TaskRegistry(task_root.parent)
    task, _, task_manifest = registry.load_task_manifest(task_name)
    if task["task_directory"] != task_root or task.get("task_id") != task_id:
        raise ValueError("Selected task identity does not match the cell-statistics request.")
    aoi_record = task_manifest.get("aoi")
    if not isinstance(aoi_record, dict):
        raise ValueError("Selected task does not contain AOI metadata.")
    spatial_record = task_manifest.get("spatial_analysis", {})
    settings = settings.model_copy(
        update={
            "aoi": AOISettings.model_validate(aoi_record),
            "spatial_analysis": settings.spatial_analysis.model_copy(
                update={
                    "analysis_crs": spatial_record.get(
                        "analysis_crs", settings.spatial_analysis.analysis_crs
                    ),
                    "cell_size_m": spatial_record.get(
                        "requested_cell_size_m",
                        settings.spatial_analysis.cell_size_m,
                    ),
                }
            ),
        }
    )
    manifest_path = task_root / "processed" / "standardized_manifest.json"
    _, standardized, records = registry.validate_raster_manifest(task_name, manifest_path)
    if task_id != standardized.get("task_id"):
        raise ValueError("Standardized manifest NASA task ID does not match the selected task.")
    output_directory = task_root / "processed" / "cell_statistics"
    output_directory.mkdir(parents=True, exist_ok=True)

    summary_path = output_directory / "cell_statistics_manifest.json"
    result: dict[str, Any] = {
        "generated_at": datetime.now().astimezone().isoformat(),
        "task_name": task_name,
        "task_id": task_id,
        "source_directory": str(task_root / "raw"),
        "source_standardization_manifest": str(manifest_path.resolve()),
        "aoi_name": settings.aoi.name,
        "analysis_crs": settings.spatial_analysis.analysis_crs,
        "requested_cell_size_m": settings.spatial_analysis.cell_size_m,
        "max_grid_cells": settings.spatial_analysis.max_grid_cells,
        "spatial_support_policy": (
            "Statistics use original valid raster samples assigned by native pixel "
            "center. No source pixels are resampled or upsampled."
        ),
        "files": [],
    }

    for record in records:
        if not isinstance(record, dict) or not isinstance(record.get("output_path"), str):
            raise ValueError(f"Invalid standardized raster record in {manifest_path}")
        source_path = Path(record["output_path"])
        if not source_path.is_absolute():
            source_path = task_root / source_path
        output_path = _output_path(
            output_directory,
            record,
            source_path,
        )
        if output_path.exists() and not overwrite:
            raise FileExistsError(
                f"Cell statistics already exist: {output_path}. "
                "Use --overwrite to replace them."
            )
        item = process_raster_cells(
            source_path,
            record,
            settings=settings,
            output_path=output_path,
        )
        result["files"].append(item)
        write_manifest(summary_path, result)
        logger.info(
            "Wrote %d cell summaries for %s/%s to %s",
            item["cell_count"],
            item["source_product_id"],
            item["source_layer_id"],
            output_path,
        )

    result["raster_count"] = len(result["files"])
    write_manifest(summary_path, result)
    result["manifest_path"] = str(summary_path)
    _, task_manifest_path, task_manifest = registry.load_task_manifest(task_name)
    task_manifest["cell_statistics_manifest_path"] = str(summary_path)
    write_manifest(task_manifest_path, task_manifest)
    return result


def process_raster_cells(
    raster_path: str | Path,
    record: dict[str, Any],
    *,
    settings: CollectionSettings,
    output_path: str | Path,
) -> dict[str, Any]:
    """Calculate cell summaries for one standardized raster without resampling it."""

    source_path = Path(raster_path)
    destination_path = Path(output_path)
    aoi = normalize_aoi(settings.aoi.geojson)
    target_crs = settings.spatial_analysis.analysis_crs
    requested_cell_size = settings.spatial_analysis.cell_size_m
    projected_aoi = _project_aoi(aoi, target_crs)

    with rasterio.open(source_path) as source:
        if source.count != 1:
            raise ValueError(f"Expected a single-band raster: {source_path}")
        if source.crs is None:
            raise ValueError(f"Raster has no CRS: {source_path}")
        native_resolution_m = native_pixel_resolution_in_crs(source, target_crs)
        if not math.isfinite(native_resolution_m) or native_resolution_m <= 0:
            raise ValueError(f"Could not determine native pixel resolution: {source_path}")
        effective_cell_size = _effective_cell_size(
            requested_cell_size,
            native_resolution_m,
        )
        grid = build_grid(
            aoi,
            target_crs=target_crs,
            resolution=effective_cell_size,
        )
        cells, valid_cell_mask = _build_cells(
            grid,
            projected_aoi,
            max_grid_cells=settings.spatial_analysis.max_grid_cells,
        )
        product_id = str(record.get("product_id") or source.tags().get("source_product_id", ""))
        layer_id = str(record.get("layer_id") or source.tags().get("source_layer_id", ""))
        variable_kind = str(
            record.get("variable_kind")
            or source.tags().get("variable_kind")
            or ("quality" if source.tags().get("is_quality") == "true" else "")
        )
        if variable_kind not in {"continuous", "categorical", "quality"}:
            raise ValueError(
                f"Variable kind is not recorded for {product_id}/{layer_id}; "
                "rerun metadata-driven standardization."
            )
        quality_layers = record.get("quality_layers")
        if not isinstance(quality_layers, list):
            try:
                quality_layers = json.loads(source.tags().get("quality_layers", "[]"))
            except json.JSONDecodeError as error:
                raise ValueError(f"Invalid QA metadata in {source_path}") from error
        if not isinstance(quality_layers, list) or not all(
            isinstance(layer, str) for layer in quality_layers
        ):
            raise ValueError(f"Invalid QA metadata in {source_path}")
        quality_filter_applied = (
            record.get("quality_filter_applied") is True
            or source.tags().get("quality_filter_applied") == "true"
        )
        cell_samples, candidate_counts = _collect_native_samples(
            source,
            grid,
            projected_aoi,
            valid_cell_mask,
            variable_kind=variable_kind,
            target_crs=target_crs,
        )
        source_crs = source.crs.to_string()
        source_resolution = [abs(float(value)) for value in source.res]

    observation_date = _observation_date(source_path)
    output_features = []
    to_geojson = Transformer.from_crs(target_crs, "EPSG:4326", always_xy=True)
    for cell in cells:
        values = _cell_values(cell_samples.get(cell.index, []))
        candidate_count = candidate_counts.get(cell.index, 0)
        valid_count = int(values.size)
        coverage = (
            valid_count / candidate_count
            if candidate_count > 0
            else None
        )
        properties: dict[str, Any] = {
            "cell_id": cell.cell_id,
            "grid_row": cell.row,
            "grid_column": cell.column,
            "requested_cell_size_m": requested_cell_size,
            "effective_cell_size_m": effective_cell_size,
            "cell_area_m2": float(effective_cell_size**2),
            "aoi_overlap_area_m2": float(cell.geometry.area),
            "native_pixel_center_count": candidate_count,
            "valid_pixel_count": valid_count,
            "valid_data_coverage": coverage,
            "coverage_definition": _COVERAGE_DEFINITION,
        }
        if variable_kind == "continuous" and values.size:
            properties.update(
                mean=float(np.mean(values)),
                median=float(np.median(values)),
                minimum=float(np.min(values)),
                maximum=float(np.max(values)),
                category_counts=None,
            )
        elif variable_kind == "continuous":
            properties.update(
                mean=None,
                median=None,
                minimum=None,
                maximum=None,
                category_counts=None,
            )
        else:
            properties.update(
                mean=None,
                median=None,
                minimum=None,
                maximum=None,
                category_counts=_category_counts(values),
            )
        geometry_wgs84 = transform_geometry(
            cell.geometry,
            to_geojson.transform,
            interleaved=False,
        )
        output_features.append(
            {
                "type": "Feature",
                "id": cell.cell_id,
                "geometry": mapping(geometry_wgs84),
                "properties": properties,
            }
        )

    relative_source = _relative_path(source_path, _project_root_for_raster(source_path))
    file_metadata = {
        "source_path": relative_source,
        "source_product_id": product_id,
        "source_layer_id": layer_id,
        "observation_date": observation_date,
        "variable_kind": variable_kind,
        "source_crs": source_crs,
        "source_resolution_native_crs": source_resolution,
        "native_pixel_resolution_m": native_resolution_m,
        "analysis_crs": target_crs,
        "requested_cell_size_m": requested_cell_size,
        "effective_cell_size_m": effective_cell_size,
        "grid_dimensions": [grid.height, grid.width],
        "cell_count": len(output_features),
        "quality_layer_ids": quality_layers,
        "quality_filter_applied": quality_filter_applied,
        "quality_status": (
            "unfiltered; QA retained separately"
            if quality_layers and not quality_filter_applied
            else "filtered"
            if quality_layers
            else "no QA dependency recorded"
        ),
        "statistics": (
            ["mean", "median", "minimum", "maximum", "valid_data_coverage"]
            if variable_kind == "continuous"
            else ["category_counts", "valid_data_coverage"]
        ),
        "coverage_definition": _COVERAGE_DEFINITION,
        "native_data_policy": (
            "Native pixel centers are assigned to cells. Values are not resampled; "
            "the effective cell size is never finer than the source pixel size."
        ),
    }
    geojson = {
        "type": "FeatureCollection",
        "name": f"{product_id}_{layer_id}_{observation_date}",
        "metadata": file_metadata,
        "features": output_features,
    }
    write_manifest(destination_path, geojson)
    return {
        **file_metadata,
        "output_path": str(destination_path),
    }


def _project_aoi(aoi: dict[str, Any], target_crs: str) -> Any:
    geometry = unary_union([shape(feature["geometry"]) for feature in aoi["features"]])
    transformer = Transformer.from_crs("EPSG:4326", target_crs, always_xy=True)
    projected = transform_geometry(
        geometry,
        transformer.transform,
        interleaved=False,
    )
    if projected.is_empty or not projected.is_valid:
        raise ValueError("AOI is invalid after projection to the analysis CRS.")
    return projected


def _effective_cell_size(requested_cell_size: float, native_resolution: float) -> float:
    multiples = max(
        1,
        math.ceil(native_resolution / requested_cell_size - 1e-10),
    )
    return float(multiples * requested_cell_size)


def _build_cells(
    grid: GridSpec,
    projected_aoi: Any,
    *,
    max_grid_cells: int,
) -> tuple[list[GridCell], np.ndarray]:
    total_cells = grid.width * grid.height
    if total_cells > max_grid_cells:
        raise ValueError(
            f"Grid has {total_cells:,} candidate cells, exceeding the configured "
            f"limit of {max_grid_cells:,}. Increase cell_size_m or reduce the AOI."
        )
    cells = []
    valid_cell_mask = np.zeros(total_cells, dtype=bool)
    for row in range(grid.height):
        top = grid.transform.f - row * grid.resolution
        bottom = top - grid.resolution
        for column in range(grid.width):
            left = grid.transform.c + column * grid.resolution
            right = left + grid.resolution
            cell_geometry = box(left, bottom, right, top).intersection(projected_aoi)
            if cell_geometry.is_empty or cell_geometry.area <= 0:
                continue
            index = row * grid.width + column
            valid_cell_mask[index] = True
            cells.append(
                GridCell(
                    index=index,
                    row=row,
                    column=column,
                    cell_id=f"r{row:06d}_c{column:06d}",
                    geometry=cell_geometry,
                )
            )
    if not cells:
        raise ValueError("No grid cells intersect the selected AOI.")
    return cells, valid_cell_mask


def _collect_native_samples(
    source: rasterio.io.DatasetReader,
    grid: GridSpec,
    projected_aoi: Any,
    valid_cell_mask: np.ndarray,
    *,
    variable_kind: str,
    target_crs: str,
) -> tuple[dict[int, list[np.ndarray]], dict[int, int]]:
    transformer = Transformer.from_crs(source.crs, target_crs, always_xy=True)
    sample_chunks: dict[int, list[np.ndarray]] = defaultdict(list)
    candidate_counts: dict[int, int] = defaultdict(int)

    for _, window in source.block_windows(1):
        data = source.read(1, window=window, masked=True)
        values = np.asarray(data.data)
        valid = ~np.ma.getmaskarray(data)
        if variable_kind == "continuous":
            valid &= np.isfinite(values)
        row_indexes, column_indexes = np.indices(
            (int(window.height), int(window.width)),
            dtype=np.float64,
        )
        x_source, y_source = source.window_transform(window) @ (
            column_indexes + 0.5,
            row_indexes + 0.5,
        )
        x_target, y_target = transformer.transform(x_source, y_source)
        x_target = np.asarray(x_target)
        y_target = np.asarray(y_target)
        coordinate_valid = np.isfinite(x_target) & np.isfinite(y_target)
        inside_aoi = np.zeros(x_target.shape, dtype=bool)
        inside_aoi[coordinate_valid] = contains_xy(
            projected_aoi,
            x_target[coordinate_valid],
            y_target[coordinate_valid],
        )
        safe_x = np.where(coordinate_valid, x_target, grid.transform.c)
        safe_y = np.where(coordinate_valid, y_target, grid.transform.f)
        grid_columns = np.floor((safe_x - grid.transform.c) / grid.resolution).astype(
            np.int64
        )
        grid_rows = np.floor((grid.transform.f - safe_y) / grid.resolution).astype(
            np.int64
        )
        in_grid = (
            inside_aoi
            & coordinate_valid
            & (grid_columns >= 0)
            & (grid_columns < grid.width)
            & (grid_rows >= 0)
            & (grid_rows < grid.height)
        )
        flat_indices = np.flatnonzero(in_grid)
        if not flat_indices.size:
            continue
        cell_indexes = (
            grid_rows.ravel()[flat_indices] * grid.width
            + grid_columns.ravel()[flat_indices]
        )
        intersects_aoi_cell = valid_cell_mask[cell_indexes]
        flat_indices = flat_indices[intersects_aoi_cell]
        cell_indexes = cell_indexes[intersects_aoi_cell]
        if not flat_indices.size:
            continue

        unique_cells, counts = np.unique(cell_indexes, return_counts=True)
        for cell_index, count in zip(unique_cells, counts):
            candidate_counts[int(cell_index)] += int(count)

        valid_in_aoi = valid.ravel()[flat_indices]
        if not valid_in_aoi.any():
            continue
        valid_cells = cell_indexes[valid_in_aoi]
        valid_values = values.ravel()[flat_indices[valid_in_aoi]]
        order = np.argsort(valid_cells, kind="stable")
        valid_cells = valid_cells[order]
        valid_values = valid_values[order]
        unique_cells, starts, counts = np.unique(
            valid_cells,
            return_index=True,
            return_counts=True,
        )
        for cell_index, start, count in zip(unique_cells, starts, counts):
            sample_chunks[int(cell_index)].append(
                np.asarray(valid_values[start : start + count])
            )
    return sample_chunks, candidate_counts


def _cell_values(chunks: list[np.ndarray]) -> np.ndarray:
    if not chunks:
        return np.array([], dtype=np.float64)
    return np.concatenate(chunks)


def _category_counts(values: np.ndarray) -> dict[str, int]:
    if not values.size:
        return {}
    if np.issubdtype(values.dtype, np.integer):
        labels = (str(int(value)) for value in values)
    else:
        labels = (format(float(value), ".12g") for value in values)
    return dict(sorted(Counter(labels).items()))


def _output_path(
    output_directory: Path,
    record: dict[str, Any],
    source_path: Path,
) -> Path:
    product_id = _safe_component(str(record.get("product_id", "unknown-product")))
    layer_id = _safe_component(str(record.get("layer_id", "unknown-layer")))
    return output_directory / product_id / layer_id / f"{source_path.stem}.geojson"


def _safe_component(value: str) -> str:
    cleaned = _SAFE_PATH_COMPONENT.sub("_", value).strip("._")
    if not cleaned:
        raise ValueError("Product and layer IDs must produce a safe output path.")
    return cleaned


def _observation_date(path: Path) -> str | None:
    match = _DATE_IN_NAME.search(path.name)
    if match is None:
        return None
    try:
        return datetime.strptime(match.group(1), "%Y%m%d").date().isoformat()
    except ValueError:
        return None


def _project_root(manifest_path: Path) -> Path:
    resolved = manifest_path.resolve()
    if resolved.parent.name == "processed" and resolved.parent.parent.name == "data":
        return resolved.parent.parent.parent
    return Path.cwd()


def _project_root_for_raster(raster_path: Path) -> Path:
    for parent in raster_path.resolve().parents:
        if parent.name == "data":
            return parent.parent
    return Path.cwd()


def _relative_path(path: Path, root: Path) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return path.resolve().as_posix()


@dataclass(frozen=True)
class GridCell:
    index: int
    row: int
    column: int
    cell_id: str
    geometry: Any


def main() -> int:
    import argparse

    from .collection.task_manager import CollectionSettings

    parser = argparse.ArgumentParser(
        description="Summarize rasters belonging to one registered AppEEARS task."
    )
    parser.add_argument("--settings", default="config/settings.yaml")
    task_group = parser.add_mutually_exclusive_group(required=True)
    task_group.add_argument("--task-name")
    task_group.add_argument("--task-id")
    parser.add_argument("--tasks-directory", default="data/tasks")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    settings = CollectionSettings.load(args.settings)
    registry = TaskRegistry(args.tasks_directory)
    task = registry.resolve(args.task_name or args.task_id)
    if not isinstance(task.get("task_id"), str):
        raise ValueError("Selected task has no NASA task ID.")
    result = run_cell_statistics(
        settings,
        task_directory=task["task_directory"],
        task_name=task["task_name"],
        task_id=task["task_id"],
        overwrite=args.overwrite,
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
