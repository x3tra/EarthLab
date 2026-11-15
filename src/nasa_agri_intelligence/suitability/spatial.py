from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import numpy as np
import rasterio
from pyproj import Transformer
from rasterio.transform import from_origin
from shapely.geometry import shape

from ..collection.manifest import write_manifest
from ..collection.task_manager import AOISettings, CollectionSettings, normalize_aoi
from ..collection.task_registry import TaskRegistry
from ..crop_requirements import DEFAULT_CROP_REQUIREMENTS_PATH, load_crop_requirements
from ..spatial_harmonization import build_grid
from .engine import evaluate_cell

SPATIAL_RASTER_VARIABLES = frozenset(
    {
        "ndvi",
        "evi",
        "temperature_day",
        "temperature_night",
        "air_temperature",
        "precipitation",
        "et",
        "pet",
        "elevation",
        "slope",
        "sand",
        "silt",
        "clay",
        "soil_ph",
        "soil_organic_carbon",
    }
)


def _layer_key(layer_id: str) -> str:
    lowered = layer_id.strip().lower()
    if "ndvi" in lowered:
        return "ndvi"
    if "evi" in lowered:
        return "evi"
    if "lst_day" in lowered or "day_1km" in lowered:
        return "temperature_day"
    if "lst_night" in lowered or "night_1km" in lowered:
        return "temperature_night"
    if "t2m" in lowered or "air_temperature" in lowered:
        return "air_temperature"
    if "prectot" in lowered or "precipitation" in lowered:
        return "precipitation"
    if "et_" in lowered:
        return "et"
    if "pet_" in lowered:
        return "pet"
    if "qc" in lowered or "quality" in lowered:
        return "quality"
    if "dem" in lowered:
        return "elevation"
    if "slope" in lowered:
        return "slope"
    if "sand" in lowered:
        return "sand"
    if "silt" in lowered:
        return "silt"
    if "clay" in lowered:
        return "clay"
    if "ph" in lowered:
        return "soil_ph"
    if "organic" in lowered:
        return "soil_organic_carbon"
    cleaned = lowered.strip("_").replace("-", "_").replace(" ", "_")
    return cleaned or "value"


def _project_aoi(geojson: dict[str, Any], target_crs: str) -> Any:
    transformed = []
    transformer = Transformer.from_crs("EPSG:4326", target_crs, always_xy=True)
    for feature in normalize_aoi(geojson)["features"]:
        geometry = shape(feature["geometry"])
        transformed.append(
            {
                "type": "Feature",
                "properties": feature.get("properties", {}),
                "geometry": geometry.__geo_interface__,
            }
        )
    return {"type": "FeatureCollection", "features": transformed}


def _sample_raster_value(
    raster_path: str | Path,
    x_coord: float,
    y_coord: float,
    *,
    target_crs: str,
) -> float | None:
    with rasterio.open(raster_path) as source:
        if source.crs is None:
            return None
        transformer = Transformer.from_crs(target_crs, source.crs, always_xy=True)
        source_x, source_y = transformer.transform(x_coord, y_coord)
        row, col = source.index(source_x, source_y)
        if row < 0 or row >= source.height or col < 0 or col >= source.width:
            return None
        value = source.read(1, window=rasterio.windows.Window(col, row, 1, 1), masked=True)
        if value.size == 0:
            return None
        sampled = value[0, 0]
        if np.ma.is_masked(sampled) or not np.isfinite(float(sampled)):
            return None
        return float(sampled)


def _crop_id_to_index(crop_ids: list[str]) -> dict[str, int]:
    return {crop_id: index + 1 for index, crop_id in enumerate(crop_ids)}


def run_spatial_suitability(
    settings: CollectionSettings,
    *,
    task_directory: str | Path,
    task_name: str,
    task_id: str,
    analysis_name: str | None = None,
    crop_ids: list[str] | None = None,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Create one isolated analysis run from a single registered NASA task."""
    requirements = load_crop_requirements(DEFAULT_CROP_REQUIREMENTS_PATH)
    selected_crops = crop_ids or [crop["id"] for crop in requirements["crops"]]
    if len(set(selected_crops)) != len(selected_crops):
        raise ValueError("Selected crop IDs must not contain duplicates.")
    crops_by_id = {crop["id"]: crop for crop in requirements["crops"]}
    unknown_crops = sorted(set(selected_crops) - crops_by_id.keys())
    if unknown_crops:
        raise ValueError(f"Unknown crop IDs: {', '.join(unknown_crops)}.")
    unsupported_crops = sorted(
        crop_id
        for crop_id in selected_crops
        if not isinstance(crops_by_id[crop_id].get("environmental_suitability"), dict)
    )
    if unsupported_crops:
        raise ValueError(
            "Spatial suitability analysis is unavailable for the selected crops: "
            f"{', '.join(unsupported_crops)} have no verified environmental "
            "suitability criteria. The current crop references contain irrigation "
            "guidance, not spatial crop-suitability criteria; no scores or "
            "recommendations were generated."
        )
    task_root = Path(task_directory).resolve()
    registry = TaskRegistry(task_root.parent)
    task, _, task_manifest = registry.load_task_manifest(task_name)
    if task["task_directory"] != task_root or task.get("task_id") != task_id:
        raise ValueError("Selected task identity does not match the suitability request.")
    manifest_path = task_root / "processed" / "standardized_manifest.json"
    _, input_manifest, records = registry.validate_raster_manifest(task_name, manifest_path)
    if input_manifest.get("task_id") != task_id:
        raise ValueError("Standardized manifest NASA task ID does not match the selected task.")
    if task_manifest.get("preprocessing_status") != "completed":
        raise ValueError("Task preprocessing must be complete before suitability analysis.")
    aoi_record = task_manifest.get("aoi")
    if not isinstance(aoi_record, dict) or not isinstance(aoi_record.get("geojson"), dict):
        raise ValueError("Selected task does not contain a valid AOI.")
    aoi_settings = AOISettings.model_validate(aoi_record)
    spatial_config = task_manifest.get("spatial_analysis", {})
    analysis_crs = str(spatial_config.get("analysis_crs") or settings.spatial_analysis.analysis_crs)
    resolution = float(
        spatial_config.get("requested_cell_size_m")
        or settings.spatial_analysis.cell_size_m
    )
    settings = settings.model_copy(
        update={
            "aoi": aoi_settings,
            "spatial_analysis": settings.spatial_analysis.model_copy(
                update={"analysis_crs": analysis_crs, "cell_size_m": resolution}
            ),
        }
    )
    aoi = normalize_aoi(aoi_settings.geojson)
    runs_directory = task_root / "suitability" / "runs"
    runs_directory.mkdir(parents=True, exist_ok=True)
    if analysis_name is None:
        existing_numbers = [
            int(match.group(1))
            for path in runs_directory.iterdir()
            if (match := re.fullmatch(r"ANALYSIS_(\d{3,})", path.name))
        ]
        analysis_name = f"ANALYSIS_{max(existing_numbers, default=0) + 1:03d}"
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,119}", analysis_name):
        raise ValueError("Analysis names must be safe path components.")
    output_dir = runs_directory / analysis_name
    if output_dir.exists() and not overwrite:
        raise FileExistsError(
            f"Suitability run already exists: {analysis_name}. Choose another name "
            "or explicitly pass overwrite=True."
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    grid = build_grid(
        aoi,
        target_crs=analysis_crs,
        resolution=resolution,
    )
    if grid.width * grid.height > settings.spatial_analysis.max_grid_cells:
        raise ValueError(
            "Configured AOI and resolution exceed the maximum allowed grid size; "
            "increase cell size or reduce the AOI."
        )

    transform = grid.transform
    best_crop_array = np.full((grid.height, grid.width), np.nan, dtype=np.float32)
    confidence_array = np.full((grid.height, grid.width), np.nan, dtype=np.float32)
    crop_score_arrays: dict[str, np.ndarray] = {
        crop_id: np.full((grid.height, grid.width), np.nan, dtype=np.float32)
        for crop_id in selected_crops
    }
    score_manifest: list[dict[str, Any]] = []
    cell_count = 0
    missing_variables: set[str] = set()
    project_root = task_root.parents[2]
    crop_index = _crop_id_to_index(selected_crops)

    for row in range(grid.height):
        for col in range(grid.width):
            center_x = transform.c + (col + 0.5) * transform.a
            center_y = transform.f + (row + 0.5) * transform.e
            environment: dict[str, float | None] = {}
            for record in records:
                path = Path(record["output_path"])
                value = _sample_raster_value(
                    path,
                    center_x,
                    center_y,
                    target_crs=analysis_crs,
                )
                if value is None:
                    continue
                key = _layer_key(str(record.get("layer_id") or "value"))
                environment[key] = value

            evaluation = evaluate_cell(
                environment,
                crop_ids=selected_crops,
                requirements_path=project_root / DEFAULT_CROP_REQUIREMENTS_PATH,
            )
            best = evaluation["best_crop"]
            if best is None:
                best_crop_id = None
                cell_confidence = None
            else:
                best_crop_id = best["crop_id"]
                cell_confidence = float(best.get("confidence", 0.0))
            best_crop_array[row, col] = (
                crop_index[best_crop_id]
                if best_crop_id is not None and best_crop_id in crop_index
                else np.nan
            )
            confidence_array[row, col] = (
                cell_confidence if cell_confidence is not None else np.nan
            )
            for crop_result in evaluation["results"]:
                crop_id = crop_result["crop_id"]
                if crop_id in crop_score_arrays:
                    if crop_result.get("criteria_evaluated", 0) > 0:
                        crop_score_arrays[crop_id][row, col] = float(crop_result["score"])
            if best_crop_id is not None:
                cell_count += 1
            cell_missing = sorted(
                {parameter for result in evaluation["results"] for parameter in result.get("missing_parameters", [])}
            )
            missing_variables.update(cell_missing)
            score_manifest.append(
                {
                    "row": row,
                    "column": col,
                    "center_x": float(center_x),
                    "center_y": float(center_y),
                    "best_crop": best_crop_id,
                    "confidence": cell_confidence,
                    "scores": {
                        item["crop_id"]: (
                            float(item["score"])
                            if item.get("criteria_evaluated", 0) > 0
                            else None
                        )
                        for item in evaluation["results"]
                    },
                    "missing_parameters": cell_missing,
                    "limiting_factors": {
                        item["crop_id"]: [
                            {
                                "parameter": observation["parameter"],
                                "score": observation["score"],
                                "source": observation.get("source"),
                            }
                            for observation in item.get("observations", [])
                            if observation["score"] < 60
                        ]
                        for item in evaluation["results"]
                    },
                }
            )

    best_crop_path = output_dir / "best_crop.tif"
    confidence_path = output_dir / "confidence.tif"
    crop_dir = output_dir / "crops"
    crop_dir.mkdir(parents=True, exist_ok=True)
    _write_float_raster(best_crop_path, best_crop_array, transform, grid.crs, nodata=np.nan)
    _write_float_raster(confidence_path, confidence_array, transform, grid.crs, nodata=np.nan)
    for crop_id, array in crop_score_arrays.items():
        _write_float_raster(crop_dir / f"{crop_id}.tif", array, transform, grid.crs, nodata=np.nan)

    manifest_result = {
        "task_name": task_name,
        "task_id": task_id,
        "analysis_name": analysis_name,
        "generated_at": __import__("datetime").datetime.now(__import__("datetime").timezone.utc).isoformat(),
        "source_directory": str(task_root / "raw"),
        "input_manifest": str(manifest_path),
        "input_raster_count": len(records),
        "configuration": {
            "analysis_crs": analysis_crs,
            "requested_cell_size_m": resolution,
            "crop_ids": selected_crops,
            "max_grid_cells": settings.spatial_analysis.max_grid_cells,
        },
        "aoi_name": settings.aoi.name,
        "analysis_crs": grid.crs,
        "requested_cell_size_m": settings.spatial_analysis.cell_size_m,
        "effective_grid_width": grid.width,
        "effective_grid_height": grid.height,
        "output_directory": str(output_dir),
        "best_crop_raster": str(best_crop_path),
        "confidence_raster": str(confidence_path),
        "score_rasters": {crop_id: str(crop_dir / f"{crop_id}.tif") for crop_id in selected_crops},
        "crop_code_map": crop_index,
        "crops_evaluated": selected_crops,
        "cells_with_data": cell_count,
        "missing_variables": sorted(missing_variables),
        "data_limitations": [
            "The workflow uses the standardized rasters currently available in the project manifest.",
            "When required environmental variables are missing, no crop receives a definitive recommendation.",
            "Confidence is computed from the share of criteria actually observed, not from invented values.",
        ],
        "cell_results": score_manifest,
    }
    manifest_path_out = output_dir / "suitability_manifest.json"
    write_manifest(manifest_path_out, manifest_result)
    task_record, task_manifest_path, current_task_manifest = registry.load_task_manifest(task_name)
    runs = current_task_manifest.get("analysis_runs", [])
    if not isinstance(runs, list):
        raise ValueError("Task analysis run history is invalid.")
    runs.append(
        {
            "analysis_name": analysis_name,
            "manifest_path": str(manifest_path_out),
            "input_manifest": str(manifest_path),
            "created_at": manifest_result["generated_at"],
        }
    )
    current_task_manifest["analysis_runs"] = runs
    write_manifest(task_manifest_path, current_task_manifest)
    registry.update(task_name, analysis_run_count=len(runs))
    return manifest_result


def _write_float_raster(path: Path, array: np.ndarray, transform: Any, crs: str, *, nodata: float) -> None:
    profile = {
        "driver": "GTiff",
        "height": array.shape[0],
        "width": array.shape[1],
        "count": 1,
        "dtype": "float32",
        "crs": crs,
        "transform": transform,
        "nodata": nodata,
        "compress": "deflate",
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(path, "w", **profile) as dataset:
        dataset.write(array.astype(np.float32), 1)


def list_suitability_results(base_directory: str | Path) -> list[dict[str, Any]]:
    directory = Path(base_directory)
    manifest_path = directory / "suitability_manifest.json"
    if not manifest_path.is_file():
        return []
    return [json.loads(manifest_path.read_text(encoding="utf-8"))]


def suitability_layers(base_directory: str | Path) -> list[dict[str, Any]]:
    results = list_suitability_results(base_directory)
    if not results:
        return []
    manifest = results[0]
    layers = [
        {
            "id": "suitability-best-crop",
            "name": "Best crop",
            "variable": "best_crop",
            "variable_kind": "categorical",
            "path": manifest["best_crop_raster"],
            "render_range": [0.0, 1.0],
        },
        {
            "id": "suitability-confidence",
            "name": "Confidence",
            "variable": "confidence",
            "variable_kind": "continuous",
            "path": manifest["confidence_raster"],
            "render_range": [0.0, 1.0],
        },
    ]
    for crop_id, candidate in manifest.get("score_rasters", {}).items():
        layers.append(
            {
                "id": f"suitability-{crop_id}",
                "name": f"{crop_id} suitability",
                "variable": crop_id,
                "variable_kind": "continuous",
                "path": candidate,
                "render_range": [0.0, 100.0],
            }
        )
    return layers
