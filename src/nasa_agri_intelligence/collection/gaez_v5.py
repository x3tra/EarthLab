from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import math
from contextlib import ExitStack
from pathlib import Path
from typing import Any, Callable, Literal
from urllib.parse import quote

import numpy as np
import rasterio
from pyproj import CRS, Transformer
from rasterio.features import geometry_mask
from rasterio.windows import Window, from_bounds
from shapely import transform as transform_geometry
from shapely.geometry import box, mapping, shape
from shapely.ops import unary_union

from ..spatial_harmonization import validate_aoi_crs_area
from .manifest import load_manifest, write_manifest
from .task_manager import normalize_aoi
from .task_registry import TaskRegistry

GAEZ_BUCKET = "fao-gismgr-gaez-v5-data"
GAEZ_ANALYSIS_VERSION = 3
GAEZ_PRODUCTS = {
    "SXX": "RES05-SXX30AS",
    "SIX": "RES05-SIX30AS",
}
WATER_REGIMES = {
    "rainfed": {"high": "HRLM", "low": "LRLM"},
    "irrigated": {"high": "HILM", "low": "LILM"},
}
_CLASS_COLORS = {
    "1": "#007000",
    "2": "#15ae11",
    "3": "#ace15e",
    "4": "#fed801",
    "5": "#ce9d53",
    "6": "#cd6c01",
    "7": "#808080",
    "8": "#cfcfcf",
    "9": "#aad3df",
}
def load_gaez_catalog(path: str | Path) -> dict[str, Any]:
    """Load the small, source-verified GAEZ crop and product configuration."""
    catalog_path = Path(path)
    with catalog_path.open(encoding="utf-8") as file:
        catalog = json.load(file)
    if not isinstance(catalog, dict) or catalog.get("schema_version") != 2:
        raise ValueError(f"Unsupported GAEZ v5 catalog: {catalog_path}")
    crops = catalog.get("crops")
    if not isinstance(crops, list) or not crops:
        raise ValueError(f"GAEZ v5 catalog has no configured crops: {catalog_path}")
    supported_scenarios = {
        scenario
        for regimes in WATER_REGIMES.values()
        for scenario in regimes.values()
    }
    seen_codes: set[str] = set()
    for crop in crops:
        if (
            not isinstance(crop, dict)
            or not isinstance(crop.get("code"), str)
            or not crop["code"].isalnum()
            or not isinstance(crop.get("name"), str)
            or not crop["name"]
            or not isinstance(crop.get("category"), str)
            or not crop["category"]
            or not isinstance(crop.get("project_crop_id"), str)
            or crop.get("availability_status") not in {"available", "partially_available"}
            or not isinstance(crop.get("available_scenario_codes"), list)
            or crop["code"] in seen_codes
        ):
            raise ValueError(f"GAEZ v5 catalog has an invalid crop: {catalog_path}")
        scenario_codes = crop["available_scenario_codes"]
        if (
            not scenario_codes
            or not all(isinstance(scenario, str) for scenario in scenario_codes)
            or len(scenario_codes) != len(set(scenario_codes))
            or not set(scenario_codes) <= supported_scenarios
            or (
                crop["availability_status"] == "available"
                and set(scenario_codes) != supported_scenarios
            )
            or (
                crop["availability_status"] == "partially_available"
                and set(scenario_codes) == supported_scenarios
            )
        ):
            raise ValueError(f"GAEZ v5 crop availability is inconsistent: {catalog_path}")
        seen_codes.add(crop["code"])
    unavailable_crops = catalog.get("unavailable_crops")
    if not isinstance(unavailable_crops, list):
        raise ValueError(f"GAEZ v5 catalog has no unavailable crop records: {catalog_path}")
    seen_unavailable_ids: set[str] = set()
    for crop in unavailable_crops:
        if (
            not isinstance(crop, dict)
            or not isinstance(crop.get("id"), str)
            or not crop["id"]
            or not isinstance(crop.get("name"), str)
            or not crop["name"]
            or not isinstance(crop.get("category"), str)
            or crop.get("availability_status")
            not in {"unavailable", "not_yet_integrated"}
            or not isinstance(crop.get("availability_reason"), str)
            or crop["id"] in seen_unavailable_ids
        ):
            raise ValueError(f"GAEZ v5 unavailable crop record is invalid: {catalog_path}")
        seen_unavailable_ids.add(crop["id"])
    continuous_product = catalog.get("continuous_product")
    class_product = catalog.get("class_product")
    inventory_snapshot = catalog.get("inventory_snapshot")
    if (
        not isinstance(continuous_product, dict)
        or not isinstance(class_product, dict)
        or not isinstance(inventory_snapshot, dict)
        or continuous_product.get("mapset") != GAEZ_PRODUCTS["SXX"]
        or class_product.get("mapset") != GAEZ_PRODUCTS["SIX"]
        or catalog.get("period_code") != "HP0120"
        or catalog.get("crs") != "EPSG:4326"
        or catalog.get("index_nodata") != -9
        or catalog.get("class_nodata") != 0
        or catalog.get("index_valid_range") != [0, 10000]
        or catalog.get("class_valid_range") != [1, 9]
        or catalog.get("index_scale_factor") != 1.0
        or catalog.get("class_scale_factor") != 1.0
        or inventory_snapshot.get("period_code") != catalog.get("period_code")
        or inventory_snapshot.get("climate_code") != catalog.get("climate_code")
        or inventory_snapshot.get("scenario_code") != catalog.get("scenario_code")
        or inventory_snapshot.get("continuous_mapset") != GAEZ_PRODUCTS["SXX"]
        or inventory_snapshot.get("class_mapset") != GAEZ_PRODUCTS["SIX"]
    ):
        raise ValueError(f"GAEZ v5 source metadata is inconsistent: {catalog_path}")
    return catalog


def gaez_catalog_response(catalog: dict[str, Any]) -> dict[str, Any]:
    """Return the public crop and source metadata used by the UI."""
    return {
        "provider": catalog["provider"],
        "period": catalog["period"],
        "spatial_resolution_degrees": catalog["spatial_resolution_degrees"],
        "crs": catalog["crs"],
        "continuous_product": catalog["continuous_product"],
        "class_product": catalog["class_product"],
        "index_metadata": {
            "units": catalog["index_units"],
            "valid_range": catalog["index_valid_range"],
            "scale_factor": catalog["index_scale_factor"],
            "no_data": catalog["index_nodata"],
        },
        "class_metadata": {
            "units": "official suitability class",
            "valid_range": catalog["class_valid_range"],
            "scale_factor": catalog["class_scale_factor"],
            "no_data": catalog["class_nodata"],
            "legend": catalog["class_legend"],
        },
        "minimum_coverage_for_ranking": catalog["minimum_coverage_for_ranking"],
        "crops": catalog["crops"],
        "unavailable_crops": catalog["unavailable_crops"],
        "inventory_snapshot": catalog["inventory_snapshot"],
        "source": catalog["source"],
        "attribution": catalog["attribution"],
        "water_regimes": [
            {"id": "rainfed", "name": "Rain-fed"},
            {"id": "irrigated", "name": "Irrigated"},
        ],
        "management_levels": [
            {"id": "high", "name": "High input"},
            {"id": "low", "name": "Low input"},
        ],
    }


def make_gaez_run_id(
    crop_codes: list[str],
    water_regime: str,
    management_level: str,
) -> str:
    identity = json.dumps(
        {
            "analysis_version": GAEZ_ANALYSIS_VERSION,
            "crop_codes": sorted(crop_codes),
            "water_regime": water_regime,
            "management_level": management_level,
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    return "GAEZ_" + hashlib.sha256(identity.encode("utf-8")).hexdigest()[:16]


def run_gaez_v5_suitability(
    registry: TaskRegistry,
    task_identifier: str,
    *,
    run_id: str,
    crop_codes: list[str],
    water_regime: Literal["rainfed", "irrigated", "both"],
    management_level: Literal["low", "high"],
    catalog_path: str | Path,
    on_operation: Callable[[dict[str, Any]], None] | None = None,
) -> dict[str, Any]:
    catalog = load_gaez_catalog(catalog_path)
    task, _, task_manifest = registry.load_task_manifest(task_identifier)
    if task_manifest.get("task_type") != "aoi_analysis":
        raise ValueError("GAEZ v5 suitability requires a registered AOI task.")
    if not isinstance(task.get("task_id"), str):
        raise ValueError("Selected AOI task has no task ID.")
    aoi_record = task_manifest.get("aoi")
    if not isinstance(aoi_record, dict) or not isinstance(aoi_record.get("geojson"), dict):
        raise ValueError("Selected AOI task does not contain a valid AOI.")
    normalized_aoi = normalize_aoi(aoi_record["geojson"])
    validate_aoi_crs_area(normalized_aoi, "EPSG:6933")
    geometry = unary_union(
        [shape(feature["geometry"]) for feature in normalized_aoi["features"]]
    )
    if (
        geometry.is_empty
        or not geometry.is_valid
        or geometry.geom_type not in {"Polygon", "MultiPolygon"}
    ):
        raise ValueError("Selected task AOI must be a valid polygon or multipolygon.")

    crops_by_code = {item["code"]: item for item in catalog["crops"]}
    if not crop_codes or len(crop_codes) != len(set(crop_codes)):
        raise ValueError("Select one or more distinct GAEZ crop codes.")
    unknown = sorted(set(crop_codes) - crops_by_code.keys())
    if unknown:
        raise ValueError(f"Unsupported GAEZ v5 crop codes: {', '.join(unknown)}.")
    if water_regime not in {"rainfed", "irrigated", "both"}:
        raise ValueError("Water regime must be rainfed, irrigated, or both.")
    if management_level not in {"low", "high"}:
        raise ValueError("Management level must be low or high.")
    if not run_id.startswith("GAEZ_") or len(run_id) != 21:
        raise ValueError("Invalid GAEZ analysis run identifier.")

    regimes = ["rainfed", "irrigated"] if water_regime == "both" else [water_regime]
    source_transformer = Transformer.from_crs(
        CRS.from_epsg(4326), CRS.from_epsg(6933), always_xy=True
    )
    projected_aoi = transform_geometry(
        geometry,
        source_transformer.transform,
        interleaved=False,
    )
    total_aoi_area = projected_aoi.area
    if not math.isfinite(total_aoi_area) or total_aoi_area <= 0:
        raise ValueError("Selected AOI has no measurable area.")

    task_root = task["task_directory"]
    output_directory = task_root / "raw" / "external" / "gaez_v5" / run_id
    output_directory.mkdir(parents=True, exist_ok=True)
    coverage_threshold = float(catalog["minimum_coverage_for_ranking"])
    if not 0 < coverage_threshold <= 1:
        raise ValueError("GAEZ minimum ranking coverage must be between 0 and 1.")

    scenario_results: list[dict[str, Any]] = []
    failures: list[dict[str, str]] = []
    raster_records: list[dict[str, Any]] = []
    operations: list[dict[str, Any]] = []
    for regime in regimes:
        crop_results: list[dict[str, Any]] = []
        for crop_code in crop_codes:
            crop = crops_by_code[crop_code]
            scenario_code = WATER_REGIMES[regime][management_level]
            operation = {
                "crop_code": crop_code,
                "crop_name": crop["name"],
                "water_regime": regime,
                "management_level": management_level,
                "status": "running",
                "stage": "checking_product_inventory",
            }
            operations.append(operation)
            if scenario_code not in crop["available_scenario_codes"]:
                operation["status"] = "failed"
                operation["failure_stage"] = "checking_product_inventory"
                operation["failure_type"] = "source_product_missing"
                operation["error"] = (
                    f"GAEZ v5 does not publish {crop['name']} for {regime} "
                    f"{management_level}-input management in {catalog['period']}."
                )
                failures.append(
                    {
                        "crop_code": crop_code,
                        "crop_name": crop["name"],
                        "water_regime": regime,
                        "management_level": management_level,
                        "product": "SXX/SIX",
                        "failure_stage": operation["failure_stage"],
                        "failure_type": operation["failure_type"],
                        "error": operation["error"],
                    }
                )
                if on_operation is not None:
                    on_operation(
                        {
                            "run_id": run_id,
                            "operations": [dict(item) for item in operations],
                            "status": "running",
                        }
                    )
                continue
            operation["stage"] = "opening_index_source"
            if on_operation is not None:
                on_operation(
                    {
                        "run_id": run_id,
                        "analysis_version": GAEZ_ANALYSIS_VERSION,
                        "operations": [dict(item) for item in operations],
                        "status": "running",
                    }
                )
            stage = "opening_index_source"
            class_data_error: str | None = None
            class_failure_stage = "opening_class_source"
            try:
                index_url = _source_url("SXX", crop_code, scenario_code, catalog)
                class_url = _source_url("SIX", crop_code, scenario_code, catalog)
                with ExitStack() as stack:
                    index_source = stack.enter_context(_open_remote_raster(index_url))
                    _validate_source_raster(index_source, "SXX", catalog)

                    class_source = None
                    stage = "opening_class_source"
                    try:
                        class_source = stack.enter_context(_open_remote_raster(class_url))
                        stage = "validating_class_source"
                        class_failure_stage = stage
                        _validate_source_pair(index_source, class_source, catalog)
                    except (
                        OSError,
                        ValueError,
                        rasterio.errors.RasterioError,
                        rasterio.errors.RasterioIOError,
                    ) as error:
                        class_source = None
                        class_data_error = f"{type(error).__name__}: {error}"

                    stage = "reading_aoi_window"
                    window = _aoi_window(index_source, geometry)
                    if window.width <= 0 or window.height <= 0:
                        raise ValueError("AOI does not overlap the GAEZ source raster.")
                    stage = "reading_index_window"
                    index_values = index_source.read(1, window=window)
                    class_values = None
                    if class_source is not None:
                        try:
                            class_values = class_source.read(1, window=window)
                        except (
                            OSError,
                            rasterio.errors.RasterioError,
                            rasterio.errors.RasterioIOError,
                        ) as error:
                            class_failure_stage = "reading_class_window"
                            class_data_error = f"{type(error).__name__}: {error}"
                            class_source = None
                    transform = index_source.window_transform(window)
                    inside_aoi = geometry_mask(
                        [mapping(geometry)],
                        out_shape=index_values.shape,
                        transform=transform,
                        invert=True,
                        all_touched=True,
                    )
                    area_weights = _cell_area_weights(
                        geometry,
                        index_values.shape,
                        transform,
                        source_transformer,
                    )
                    aoi_cells = inside_aoi & (area_weights > 0)
                    stage = "validating_source_values"
                    valid_classes = (
                        aoi_cells & (class_values != catalog["class_nodata"])
                        if class_values is not None
                        else np.zeros(index_values.shape, dtype=bool)
                    )
                    if class_values is not None and np.any(
                        (class_values[valid_classes] < catalog["class_valid_range"][0])
                        | (class_values[valid_classes] > catalog["class_valid_range"][1])
                    ):
                        class_failure_stage = "validating_class_values"
                        invalid_classes = np.unique(
                            class_values[
                                valid_classes
                                & (
                                    (class_values < catalog["class_valid_range"][0])
                                    | (class_values > catalog["class_valid_range"][1])
                                )
                            ]
                        )
                        class_data_error = (
                            "ValueError: GAEZ SIX source contains unknown official "
                            f"class values: {invalid_classes[:10].tolist()}."
                        )
                        class_values = None
                        valid_classes = np.zeros(index_values.shape, dtype=bool)
                        class_source = None
                    water_cells = (
                        valid_classes & (class_values == 9)
                        if class_values is not None
                        else np.zeros(index_values.shape, dtype=bool)
                    )
                    out_of_range = aoi_cells & (
                        (index_values < catalog["index_valid_range"][0])
                        | (index_values > catalog["index_valid_range"][1])
                    ) & (index_values != catalog["index_nodata"])
                    unexplained_out_of_range = out_of_range & ~(
                        (index_values == -1) & water_cells
                    )
                    if np.any(unexplained_out_of_range):
                        values = np.unique(index_values[unexplained_out_of_range])
                        raise ValueError(
                            "GAEZ SXX source contains values outside 0-10000 and "
                            "the documented no-data value; SIX water class 9 only "
                            f"explains -1. Unexpected values: {values[:10].tolist()}."
                        )
                    valid_index = (
                        aoi_cells
                        & (index_values != catalog["index_nodata"])
                        & (index_values >= catalog["index_valid_range"][0])
                        & (index_values <= catalog["index_valid_range"][1])
                        & ~water_cells
                    )
                    valid_index_area = float(area_weights[valid_index].sum())
                    water_area = float(area_weights[water_cells].sum())
                    non_water_aoi_area = max(0.0, total_aoi_area - water_area)
                    coverage = (
                        min(1.0, valid_index_area / non_water_aoi_area)
                        if non_water_aoi_area > 0
                        else 0.0
                    )
                    coverage_of_aoi = min(1.0, valid_index_area / total_aoi_area)
                    weighted_mean = (
                        float(
                            np.average(
                                index_values[valid_index].astype(np.float64),
                                weights=area_weights[valid_index],
                            )
                        )
                        if valid_index.any()
                        else None
                    )
                    class_distribution = (
                        _class_distribution(
                            class_values,
                            valid_classes,
                            area_weights,
                            catalog["class_legend"],
                            total_aoi_area=total_aoi_area,
                        )
                        if class_values is not None
                        else []
                    )
                    index_output = output_directory / f"SXX_{crop_code}_{regime}_{management_level}.tif"
                    class_output = output_directory / f"SIX_{crop_code}_{regime}_{management_level}.tif"
                    stage = "writing_task_rasters"
                    _write_aoi_raster(
                        index_source,
                        index_values,
                        aoi_cells,
                        window,
                        index_output,
                        tags=_raster_tags(
                            task,
                            crop,
                            catalog,
                            product="SXX",
                            scenario_code=scenario_code,
                            regime=regime,
                            management=management_level,
                            nodata=catalog["index_nodata"],
                        ),
                    )
                    if class_source is not None and class_values is not None:
                        _write_aoi_raster(
                            class_source,
                            class_values,
                            aoi_cells,
                            window,
                            class_output,
                            tags=_raster_tags(
                                task,
                                crop,
                                catalog,
                                product="SIX",
                                scenario_code=scenario_code,
                                regime=regime,
                                management=management_level,
                                nodata=catalog["class_nodata"],
                            ),
                        )
                eligible = coverage >= coverage_threshold and weighted_mean is not None
                if class_data_error is not None:
                    failures.append(
                        {
                            "crop_code": crop_code,
                            "crop_name": crop["name"],
                            "water_regime": regime,
                            "management_level": management_level,
                            "product": "SIX",
                            "failure_stage": class_failure_stage,
                            "failure_type": _classify_failure(
                                class_failure_stage, class_data_error
                            ),
                            "error": class_data_error,
                        }
                    )
                crop_result = {
                    "crop_code": crop_code,
                    "crop_id": crop["project_crop_id"],
                    "crop_name": crop["name"],
                    "water_regime": regime,
                    "management_level": management_level,
                    "management_code": scenario_code,
                    "mean_suitability_index": weighted_mean,
                    "suitability_index_units": catalog["index_units"],
                    "valid_data_coverage": coverage,
                    "valid_data_coverage_of_aoi": coverage_of_aoi,
                    "valid_data_coverage_basis": (
                        "AOI area excluding cells classified as Water (SIX class 9)"
                    ),
                    "aoi_area_km2": total_aoi_area / 1_000_000,
                    "water_area_km2": water_area / 1_000_000,
                    "water_area_share_of_aoi": water_area / total_aoi_area,
                    "intersecting_cell_count": int(aoi_cells.sum()),
                    "valid_index_cell_count": int(valid_index.sum()),
                    "water_cell_count": int(water_cells.sum()),
                    "index_nodata_cell_count": int(
                        (aoi_cells & (index_values == catalog["index_nodata"])).sum()
                    ),
                    "class_data_available": (
                        bool(valid_classes.any())
                        and float(area_weights[valid_classes].sum()) > 0
                    ),
                    "class_data_status": (
                        "available"
                        if class_data_error is None
                        and valid_classes.any()
                        and float(area_weights[valid_classes].sum()) > 0
                        else "unavailable"
                    ),
                    "class_data_error": class_data_error,
                    "valid_class_cell_count": int(valid_classes.sum()),
                    "class_data_coverage": min(
                        1.0,
                        float(area_weights[valid_classes].sum()) / total_aoi_area,
                    ),
                    "class_data_coverage_of_aoi": min(
                        1.0,
                        float(area_weights[valid_classes].sum()) / total_aoi_area,
                    ),
                    "classified_area_km2": (
                        float(area_weights[valid_classes].sum()) / 1_000_000
                    ),
                    "ranking_eligible": eligible,
                    "ranking_exclusion_reason": (
                        None
                        if eligible
                        else (
                            "No valid suitability-index cells overlap the AOI."
                            if weighted_mean is None
                            else (
                                f"Coverage {coverage:.1%} is below the configured "
                                f"{coverage_threshold:.0%} ranking requirement."
                            )
                        )
                    ),
                    "class_distribution": class_distribution,
                    "continuous_map_layer": f"SXX_{crop_code}_{regime}_{management_level}",
                    "class_map_layer": (
                        f"SIX_{crop_code}_{regime}_{management_level}"
                        if class_data_error is None
                        else None
                    ),
                    "continuous_raster": str(index_output.relative_to(task_root)),
                    "class_raster": (
                        str(class_output.relative_to(task_root))
                        if class_data_error is None
                        else None
                    ),
                    "source_metadata": _source_metadata(
                        crop,
                        catalog,
                        scenario_code,
                        regime,
                        management_level,
                    ),
                }
                crop_results.append(crop_result)
                raster_records.extend(
                    _raster_records(
                        task,
                        crop_result,
                        index_output,
                        class_output if class_data_error is None else None,
                    )
                )
                operation["status"] = "completed"
                operation["stage"] = "completed"
                if class_data_error is not None:
                    operation["class_data_status"] = "unavailable"
                    operation["class_data_error"] = class_data_error
            except (
                OSError,
                ValueError,
                rasterio.errors.RasterioError,
                rasterio.errors.RasterioIOError,
            ) as error:
                operation["status"] = "failed"
                operation["failure_stage"] = stage
                operation["failure_type"] = _classify_failure(stage, error)
                operation["error"] = f"{type(error).__name__}: {error}"
                failures.append(
                    {
                        "crop_code": crop_code,
                        "water_regime": regime,
                        "management_level": management_level,
                        "failure_stage": stage,
                        "failure_type": operation["failure_type"],
                        "error": operation["error"],
                    }
                )
            if on_operation is not None:
                on_operation(
                    {
                        "run_id": run_id,
                        "operations": [dict(item) for item in operations],
                        "status": "running",
                    }
                )
        ranked = sorted(
            (item for item in crop_results if item["ranking_eligible"]),
            key=lambda item: (-item["mean_suitability_index"], item["crop_name"]),
        )
        previous_score: float | None = None
        previous_rank = 0
        for position, item in enumerate(ranked, start=1):
            score = item["mean_suitability_index"]
            if score != previous_score:
                previous_rank = position
                previous_score = score
            item["rank"] = previous_rank
        scenario_results.append(
            {
                "water_regime": regime,
                "management_level": management_level,
                "management_code": WATER_REGIMES[regime][management_level],
                "product_identifiers": {
                    "continuous_index": catalog["continuous_product"]["mapset"],
                    "official_classes": catalog["class_product"]["mapset"],
                },
                "ranked_crops": ranked,
                "unranked_crops": [
                    item for item in crop_results if not item["ranking_eligible"]
                ],
            }
        )

    success_count = sum(item["status"] == "completed" for item in operations)
    status = (
        "completed"
        if success_count == len(operations) and not failures
        else "partial"
        if success_count
        else "failed"
    )
    result = {
        "task_name": task["task_name"],
        "task_id": task["task_id"],
        "run_id": run_id,
        "analysis_version": GAEZ_ANALYSIS_VERSION,
        "status": status,
        "source": catalog["provider"],
        "period": catalog["period"],
        "period_code": catalog["period_code"],
        "climate_code": catalog["climate_code"],
        "scenario_code": catalog["scenario_code"],
        "crs": catalog["crs"],
        "spatial_resolution_degrees": catalog["spatial_resolution_degrees"],
        "minimum_coverage_for_ranking": coverage_threshold,
        "attribution": catalog["attribution"],
        "legend_url": catalog["legend_url"],
        "continuous_product": catalog["continuous_product"],
        "class_product": catalog["class_product"],
        "index_metadata": {
            "units": catalog["index_units"],
            "valid_range": catalog["index_valid_range"],
            "scale_factor": catalog["index_scale_factor"],
            "no_data": catalog["index_nodata"],
        },
        "class_metadata": {
            "units": "official suitability class",
            "valid_range": catalog["class_valid_range"],
            "scale_factor": catalog["class_scale_factor"],
            "no_data": catalog["class_nodata"],
            "legend": catalog["class_legend"],
        },
        "water_regime_request": water_regime,
        "management_level": management_level,
        "crop_codes": crop_codes,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "scenarios": scenario_results,
        "failures": failures,
        "limitations": [
            "Suitability is a long-term modeled agro-ecological index, not a crop-yield guarantee.",
            "Rain-fed and irrigated rankings are separate; irrigated suitability does not establish irrigation infrastructure or water availability.",
            "GAEZ grid cells are approximately 1 km and do not support field-level accuracy claims.",
            "The categorical SIX product classifies all land and is independent of the continuous SXX raster.",
            "SXX values of -1 are treated as water only when the paired official SIX product identifies the same cell as class 9; water area is excluded from the SXX data-coverage denominator and remains visible in class distribution.",
            "A value of 100% data coverage describes valid SXX coverage of non-water AOI area, not crop suitability.",
        ],
    }
    manifest_path = task_root / "suitability" / "gaez" / run_id / "gaez_manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    write_manifest(manifest_path, result)
    _merge_raster_records(registry, task, raster_records)
    _, task_manifest_path, current = registry.load_task_manifest(task_identifier)
    runs = current.setdefault("gaez_runs", [])
    if not isinstance(runs, list):
        raise ValueError("AOI task has invalid GAEZ run history.")
    existing_run = next(
        (item for item in runs if isinstance(item, dict) and item.get("run_id") == run_id),
        None,
    )
    run_reference = {
        "run_id": run_id,
        "status": status,
        "manifest_path": str(manifest_path),
        "created_at": result["generated_at"],
    }
    if existing_run is None:
        runs.append(run_reference)
    else:
        existing_run.update(run_reference)
    current["updated_at"] = result["generated_at"]
    write_manifest(task_manifest_path, current)
    return result


def load_gaez_run(
    registry: TaskRegistry,
    task_identifier: str,
    run_id: str | None = None,
) -> dict[str, Any]:
    task, _, task_manifest = registry.load_task_manifest(task_identifier)
    runs = task_manifest.get("gaez_runs", [])
    if not isinstance(runs, list):
        raise ValueError("AOI task has invalid GAEZ run history.")
    run = next(
        (
            item
            for item in reversed(runs)
            if isinstance(item, dict)
            and (run_id is None or item.get("run_id") == run_id)
        ),
        None,
    )
    if run is None:
        provider = task_manifest.get("acquisition", {}).get("providers", {}).get("gaez_v5", {})
        if provider.get("status") in {"queued", "running", "failed"} and (
            run_id is None or provider.get("run_id") == run_id
        ):
            return {
                "task_name": task["task_name"],
                "task_id": task["task_id"],
                "status": provider["status"],
                "run_id": provider.get("run_id"),
                "operations": provider.get("operations", []),
                "error": provider.get("error"),
            }
        return {
            "task_name": task["task_name"],
            "task_id": task["task_id"],
            "status": "not_started",
            "scenarios": [],
        }
    run_id_value = run.get("run_id")
    if not isinstance(run_id_value, str) or not run_id_value.startswith("GAEZ_"):
        raise ValueError("AOI task has an invalid GAEZ run reference.")
    run_directory = (task["task_directory"] / "suitability" / "gaez" / run_id_value).resolve()
    manifest_path = Path(str(run.get("manifest_path", ""))).resolve()
    if not manifest_path.is_relative_to(run_directory) or not manifest_path.is_file():
        raise FileNotFoundError("GAEZ run manifest is missing from its task directory.")
    result = load_manifest(manifest_path)
    if (
        result.get("task_name") != task["task_name"]
        or result.get("task_id") != task["task_id"]
        or result.get("run_id") != run_id_value
    ):
        raise ValueError("GAEZ run manifest ownership does not match the AOI task.")
    result["analysis_outdated"] = (
        result.get("analysis_version") != GAEZ_ANALYSIS_VERSION
    )
    return result


def _source_url(
    product_key: str,
    crop_code: str,
    management_code: str,
    catalog: dict[str, Any],
) -> str:
    mapset = GAEZ_PRODUCTS[product_key]
    filename = (
        f"GAEZ-V5.{mapset}.{catalog['period_code']}.{catalog['climate_code']}."
        f"{catalog['scenario_code']}.{crop_code}.{management_code}.tif"
    )
    object_name = f"DATA/GAEZ-V5/MAPSET/{mapset}/{filename}"
    return (
        "https://storage.googleapis.com/download/storage/v1/b/"
        f"{GAEZ_BUCKET}/o/{quote(object_name, safe='')}?alt=media"
    )


def _open_remote_raster(url: str) -> rasterio.DatasetReader:
    return rasterio.open(
        f"/vsicurl/{url}",
        sharing=False,
        **{"NUM_THREADS": "1"},
    )


def _validate_source_pair(
    index_source: rasterio.DatasetReader,
    class_source: rasterio.DatasetReader,
    catalog: dict[str, Any],
) -> None:
    _validate_source_raster(index_source, "SXX", catalog)
    _validate_source_raster(class_source, "SIX", catalog)
    if (
        index_source.width != class_source.width
        or index_source.height != class_source.height
        or index_source.transform != class_source.transform
    ):
        raise ValueError("GAEZ SXX and SIX products do not share the same source grid.")


def _validate_source_raster(
    source: rasterio.DatasetReader,
    product: str,
    catalog: dict[str, Any],
) -> None:
    if source.count != 1 or source.crs is None or CRS.from_user_input(source.crs) != CRS.from_epsg(4326):
        raise ValueError("GAEZ source raster must be a single-band EPSG:4326 raster.")
    expected_resolution = float(catalog["spatial_resolution_degrees"])
    if (
        not math.isclose(abs(source.transform.a), expected_resolution, rel_tol=0, abs_tol=1e-10)
        or not math.isclose(abs(source.transform.e), expected_resolution, rel_tol=0, abs_tol=1e-10)
    ):
        raise ValueError("GAEZ source raster resolution does not match the verified 1 km product.")
    expected_nodata = catalog["index_nodata"] if product == "SXX" else catalog["class_nodata"]
    if source.nodata != expected_nodata:
        raise ValueError(
            f"GAEZ {product} no-data value does not match official product metadata."
        )
    expected_scale = (
        catalog["index_scale_factor"] if product == "SXX" else catalog["class_scale_factor"]
    )
    if source.scales != (expected_scale,) or source.offsets != (0.0,):
        raise ValueError(
            f"GAEZ {product} scale or offset differs from verified source metadata."
        )


def _classify_failure(stage: str, error: BaseException | str) -> str:
    message = str(error).casefold()
    if stage == "reading_aoi_window" and "does not overlap" in message:
        return "no_aoi_coverage"
    if stage in {"validating_source_values", "validating_class_values"}:
        return "invalid_source_values"
    if stage == "reading_index_window":
        return "source_download_error"
    if stage in {"opening_index_source", "opening_class_source"}:
        if any(marker in message for marker in ("404", "not found", "nosuchkey")):
            return "source_product_missing"
        return "source_download_error"
    if stage == "reading_class_window":
        return "source_download_error"
    if stage == "validating_class_source":
        return "source_metadata_error"
    return "processing_error"


def _aoi_window(source: rasterio.DatasetReader, geometry: Any) -> Window:
    west, south, east, north = geometry.bounds
    fractional = from_bounds(west, south, east, north, source.transform)
    col_start = math.floor(fractional.col_off)
    row_start = math.floor(fractional.row_off)
    col_stop = math.ceil(fractional.col_off + fractional.width)
    row_stop = math.ceil(fractional.row_off + fractional.height)
    candidate = Window(
        col_start,
        row_start,
        col_stop - col_start,
        row_stop - row_start,
    )
    return candidate.intersection(Window(0, 0, source.width, source.height))


def _cell_area_weights(
    geometry: Any,
    shape_: tuple[int, int],
    transform: Any,
    transformer: Transformer,
) -> np.ndarray:
    weights = np.zeros(shape_, dtype=np.float64)
    for row in range(shape_[0]):
        top = transform.f + row * transform.e
        bottom = top + transform.e
        for column in range(shape_[1]):
            left = transform.c + column * transform.a
            right = left + transform.a
            intersection = geometry.intersection(
                box(min(left, right), min(bottom, top), max(left, right), max(bottom, top))
            )
            if intersection.is_empty:
                continue
            projected = transform_geometry(
                intersection,
                transformer.transform,
                interleaved=False,
            )
            weights[row, column] = projected.area
    return weights


def _class_distribution(
    values: np.ndarray,
    valid: np.ndarray,
    weights: np.ndarray,
    legend: dict[str, str],
    *,
    total_aoi_area: float,
) -> list[dict[str, Any]]:
    classified_area = float(weights[valid].sum())
    distribution = []
    for code, label in legend.items():
        class_area = float(weights[valid & (values == int(code))].sum())
        distribution.append(
            {
                "class_code": int(code),
                "label": label,
                "area_km2": class_area / 1_000_000,
                "share_of_aoi": class_area / total_aoi_area,
                "share_of_classified_area": (
                    class_area / classified_area if classified_area else None
                ),
            }
        )
    return distribution


def _raster_tags(
    task: dict[str, Any],
    crop: dict[str, Any],
    catalog: dict[str, Any],
    *,
    product: str,
    scenario_code: str,
    regime: str,
    management: str,
    nodata: int,
) -> dict[str, str]:
    return {
        "source_product_id": "FAO_GAEZ_V5",
        "source_layer_id": f"{product}_{crop['code']}_{regime}_{management}",
        "external_source": "FAO/GAEZ-V5",
        "task_name": str(task["task_name"]),
        "task_id": str(task["task_id"]),
        "crop_code": str(crop["code"]),
        "crop_name": str(crop["name"]),
        "gaez_mapset": GAEZ_PRODUCTS[product],
        "period": catalog["period"],
        "period_code": catalog["period_code"],
        "climate_code": catalog["climate_code"],
        "scenario_code": catalog["scenario_code"],
        "water_regime": regime,
        "management_level": management,
        "management_code": scenario_code,
        "crs": catalog["crs"],
        "source_resolution_degrees": str(catalog["spatial_resolution_degrees"]),
        "units": catalog["index_units"] if product == "SXX" else "official suitability class",
        "source_nodata": str(nodata),
        "attribution": catalog["attribution"],
        "legend_url": catalog["legend_url"],
        "downloaded_at": datetime.now(timezone.utc).isoformat(),
    }


def _write_aoi_raster(
    source: rasterio.DatasetReader,
    values: np.ndarray,
    inside_aoi: np.ndarray,
    window: Window,
    output_path: Path,
    *,
    tags: dict[str, str],
) -> None:
    if output_path.exists():
        with rasterio.open(output_path) as existing:
            existing_tags = existing.tags()
            if (
                existing.count != 1
                or existing.crs != source.crs
                or existing_tags.get("task_id") != tags["task_id"]
                or existing_tags.get("source_layer_id") != tags["source_layer_id"]
            ):
                raise ValueError(f"Existing GAEZ output does not match this task: {output_path}")
        return
    output_path.parent.mkdir(parents=True, exist_ok=True)
    profile = source.profile.copy()
    profile.pop("blockxsize", None)
    profile.pop("blockysize", None)
    profile.update(
        height=values.shape[0],
        width=values.shape[1],
        transform=source.window_transform(window),
        compress="deflate",
    )
    output_values = values.copy()
    output_values[~inside_aoi] = source.nodata
    with rasterio.open(output_path, "w", **profile) as output:
        output.write(output_values, 1)
        output.update_tags(**tags)
        output.set_band_description(1, tags["source_layer_id"])
        output.scales = source.scales
        output.offsets = source.offsets


def _source_metadata(
    crop: dict[str, Any],
    catalog: dict[str, Any],
    scenario_code: str,
    regime: str,
    management: str,
) -> dict[str, Any]:
    return {
        "provider": catalog["provider"],
        "continuous_product": catalog["continuous_product"],
        "class_product": catalog["class_product"],
        "crop_code": crop["code"],
        "crop_name": crop["name"],
        "period": catalog["period"],
        "period_code": catalog["period_code"],
        "climate_code": catalog["climate_code"],
        "scenario_code": catalog["scenario_code"],
        "water_regime": regime,
        "management_level": management,
        "management_code": scenario_code,
        "crs": catalog["crs"],
        "source_resolution_degrees": catalog["spatial_resolution_degrees"],
        "index_units": catalog["index_units"],
        "index_valid_range": catalog["index_valid_range"],
        "index_scale_factor": catalog["index_scale_factor"],
        "class_valid_range": catalog["class_valid_range"],
        "class_scale_factor": catalog["class_scale_factor"],
        "index_nodata": catalog["index_nodata"],
        "class_nodata": catalog["class_nodata"],
        "class_legend": catalog["class_legend"],
        "legend_url": catalog["legend_url"],
        "attribution": catalog["attribution"],
    }


def _raster_records(
    task: dict[str, Any],
    crop_result: dict[str, Any],
    index_path: Path,
    class_path: Path | None,
) -> list[dict[str, Any]]:
    common = {
        "external_source": "FAO/GAEZ-V5",
        "data_scope": "task_scoped",
        "task_name": task["task_name"],
        "task_id": task["task_id"],
        "crop_code": crop_result["crop_code"],
        "crop_name": crop_result["crop_name"],
        "period": crop_result["source_metadata"]["period"],
        "water_regime": crop_result["water_regime"],
        "management_level": crop_result["management_level"],
    }
    records = [{
            **common,
            "product_id": "FAO_GAEZ_V5",
            "layer_id": crop_result["continuous_map_layer"],
            "output_path": str(index_path),
            "source_path": str(index_path),
            "source_product_id": GAEZ_PRODUCTS["SXX"],
            "variable_kind": "continuous",
            "output_units": "GAEZ suitability index (0-10000)",
            "physical_valid_range": [0, 10000],
            "render_range": [0, 10000],
        }]
    if class_path is not None:
        records.append({
            **common,
            "product_id": "FAO_GAEZ_V5",
            "layer_id": crop_result["class_map_layer"],
            "output_path": str(class_path),
            "source_path": str(class_path),
            "source_product_id": GAEZ_PRODUCTS["SIX"],
            "variable_kind": "categorical",
            "output_units": "official suitability class",
            "physical_valid_range": [1, 9],
            "render_range": [1, 9],
            "category_labels": {
                item["class_code"]: item["label"]
                for item in crop_result["class_distribution"]
            },
            "category_colors": _CLASS_COLORS,
        })
    return records


def _merge_raster_records(
    registry: TaskRegistry,
    task: dict[str, Any],
    records: list[dict[str, Any]],
) -> None:
    manifest_path = task["task_directory"] / "processed" / "standardized_manifest.json"
    _, raster_manifest, existing = registry.validate_raster_manifest(
        str(task["task_name"]),
        manifest_path,
    )
    by_layer = {
        (item["product_id"], item["layer_id"]): item
        for item in existing
    }
    for record in records:
        key = (record["product_id"], record["layer_id"])
        previous = by_layer.get(key)
        if previous is not None:
            if Path(previous["output_path"]).resolve() == Path(record["output_path"]).resolve():
                continue
            if (
                previous.get("external_source") != "FAO/GAEZ-V5"
                or previous.get("task_id") != task.get("task_id")
            ):
                raise ValueError(f"Existing GAEZ layer points to a different output: {key[1]}")
        by_layer[key] = record
    raster_manifest["records"] = list(by_layer.values())
    raster_manifest["rasters"] = list(by_layer.values())
    raster_manifest["raster_count"] = len(by_layer)
    raster_manifest["external_integrations"] = sorted(
        {
            item.get("external_source")
            for item in by_layer.values()
            if isinstance(item.get("external_source"), str)
        }
    )
    raster_manifest["updated_at"] = datetime.now(timezone.utc).isoformat()
    write_manifest(manifest_path, raster_manifest)
    registry.validate_raster_manifest(str(task["task_name"]), manifest_path)
