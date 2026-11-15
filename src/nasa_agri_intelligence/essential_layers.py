from __future__ import annotations

import ast
from pathlib import Path
from typing import Any

from .collection.manifest import load_manifest
from .collection.task_registry import TaskRegistry
from .suitability.spatial import SPATIAL_RASTER_VARIABLES

_CATALOG_DEFINITIONS: tuple[dict[str, Any], ...] = (
    {
        "id": "ndvi",
        "name": "NDVI",
        "category": "Vegetation",
        "description": "Vegetation greenness for seasonal crop monitoring; not a direct measure of crop suitability.",
        "source": "MODIS MOD13Q1",
        "variables": [
            {
                "id": "ndvi",
                "name": "NDVI",
                "product_id": "MOD13Q1.061",
                "layer_id": "_250m_16_days_NDVI",
            }
        ],
    },
    {
        "id": "lst",
        "name": "Land Surface Temperature",
        "category": "Temperature",
        "description": "Satellite-observed surface skin temperature; it is not air temperature.",
        "source": "MODIS MOD11A2",
        "variables": [
            {
                "id": "lst_day",
                "name": "Daytime LST",
                "product_id": "MOD11A2.061",
                "layer_id": "LST_Day_1km",
            },
            {
                "id": "lst_night",
                "name": "Nighttime LST",
                "product_id": "MOD11A2.061",
                "layer_id": "LST_Night_1km",
            },
        ],
    },
    {
        "id": "evapotranspiration",
        "name": "Evapotranspiration",
        "category": "Water and energy",
        "description": "Actual evapotranspiration; it is not a direct measurement of irrigation supply or water availability.",
        "source": "MODIS MOD16A2GF",
        "variables": [
            {
                "id": "et",
                "name": "Actual ET",
                "product_id": "MOD16A2GF.061",
                "layer_id": "ET_500m",
            },
        ],
    },
    {
        "id": "potential_evapotranspiration",
        "name": "Potential Evapotranspiration",
        "category": "Water and energy",
        "description": "Potential evapotranspiration estimates atmospheric demand; it is not a direct measure of irrigation supply.",
        "source": "MODIS MOD16A2GF",
        "variables": [
            {
                "id": "pet",
                "name": "Potential ET",
                "product_id": "MOD16A2GF.061",
                "layer_id": "PET_500m",
            },
        ],
    },
    {
        "id": "soil_properties",
        "name": "Soil Properties",
        "category": "Soil",
        "description": "SoilGrids 250 m predicted soil properties at the selected depth; not field observations.",
        "source": "ISRIC SoilGrids 2.0",
        "variables": [
            {
                "id": "soil_ph",
                "name": "pH",
                "product_id": "ISRIC_SOILGRIDS",
                "layer_id": "phh2o_0-5cm_Q0.5",
                "units": "pH",
                "spatial_resolution": "250 m",
                "temporal_resolution": "Static",
                "external_source": "ISRIC/SoilGrids",
                "depth_interval": "0-5cm",
            },
            {
                "id": "soil_sand",
                "name": "Sand fraction",
                "product_id": "ISRIC_SOILGRIDS",
                "layer_id": "sand_0-5cm_Q0.5",
                "units": "%",
                "spatial_resolution": "250 m",
                "temporal_resolution": "Static",
                "external_source": "ISRIC/SoilGrids",
                "depth_interval": "0-5cm",
            },
            {
                "id": "soil_silt",
                "name": "Silt fraction",
                "product_id": "ISRIC_SOILGRIDS",
                "layer_id": "silt_0-5cm_Q0.5",
                "units": "%",
                "spatial_resolution": "250 m",
                "temporal_resolution": "Static",
                "external_source": "ISRIC/SoilGrids",
                "depth_interval": "0-5cm",
            },
            {
                "id": "soil_clay",
                "name": "Clay fraction",
                "product_id": "ISRIC_SOILGRIDS",
                "layer_id": "clay_0-5cm_Q0.5",
                "units": "%",
                "spatial_resolution": "250 m",
                "temporal_resolution": "Static",
                "external_source": "ISRIC/SoilGrids",
                "depth_interval": "0-5cm",
            },
            {
                "id": "soil_organic_carbon",
                "name": "Organic carbon",
                "product_id": "ISRIC_SOILGRIDS",
                "layer_id": "soc_0-5cm_Q0.5",
                "units": "g/kg",
                "spatial_resolution": "250 m",
                "temporal_resolution": "Static",
                "external_source": "ISRIC/SoilGrids",
                "depth_interval": "0-5cm",
            },
        ],
    },
    {
        "id": "elevation",
        "name": "Elevation",
        "category": "Terrain",
        "description": "Static terrain elevation from the digital elevation model.",
        "source": "ASTER GDEM",
        "variables": [
            {
                "id": "elevation",
                "name": "Elevation",
                "product_id": "ASTGTM_NC.003",
                "layer_id": "ASTER_GDEM_DEM",
            }
        ],
    },
    {
        "id": "slope",
        "name": "Slope",
        "category": "Terrain",
        "description": "Terrain steepness derived in degrees from the elevation raster retained by this task.",
        "source": "Derived from ASTER GDEM",
        "variables": [
            {
                "id": "slope",
                "name": "Slope",
                "product_id": "_DERIVED",
                "layer_id": "slope",
                "source_product_id": "ASTGTM_NC.003",
                "source_layer_id": "ASTER_GDEM_DEM",
                "derived": True,
            }
        ],
    },
    {
        "id": "land_cover",
        "name": "Land Cover",
        "category": "Land use",
        "description": "Annual categorical land-cover classes; class codes are not continuous measurements.",
        "source": "MODIS MCD12Q1",
        "variables": [
            {
                "id": "land_cover",
                "name": "Land cover (IGBP)",
                "product_id": "MCD12Q1.061",
                "layer_id": "LC_Type1",
            }
        ],
    },
    {
        "id": "air_temperature",
        "name": "Air Temperature",
        "category": "Temperature",
        "description": "Daily 2 m air temperature, averaged over the task dates; distinct from satellite land surface temperature.",
        "source": "NASA POWER (MERRA-2)",
        "variables": [
            {
                "id": "air_temperature",
                "name": "2 m air temperature",
                "product_id": "NASA_POWER_DAILY",
                "layer_id": "T2M",
                "units": "°C",
                "spatial_resolution": "0.5° × 0.625°",
                "temporal_resolution": "Daily",
                "external_source": "NASA/POWER",
            },
            {
                "id": "air_temperature_min",
                "name": "Daily minimum 2 m air temperature",
                "product_id": "NASA_POWER_DAILY",
                "layer_id": "T2M_MIN",
                "units": "°C",
                "spatial_resolution": "0.5° × 0.625°",
                "temporal_resolution": "Daily",
                "external_source": "NASA/POWER",
            },
            {
                "id": "air_temperature_max",
                "name": "Daily maximum 2 m air temperature",
                "product_id": "NASA_POWER_DAILY",
                "layer_id": "T2M_MAX",
                "units": "°C",
                "spatial_resolution": "0.5° × 0.625°",
                "temporal_resolution": "Daily",
                "external_source": "NASA/POWER",
            },
        ],
    },
    {
        "id": "precipitation",
        "name": "Precipitation",
        "category": "Water and energy",
        "description": "Corrected daily precipitation accumulated over the task dates; not a measure of local irrigation supply.",
        "source": "NASA POWER (MERRA-2)",
        "variables": [
            {
                "id": "precipitation",
                "name": "Corrected precipitation",
                "product_id": "NASA_POWER_DAILY",
                "layer_id": "PRECTOTCORR",
                "units": "mm",
                "spatial_resolution": "0.5° × 0.625°",
                "temporal_resolution": "Daily",
                "external_source": "NASA/POWER",
            }
        ],
    },
)


def build_essential_layer_catalog(
    metadata_directory: str | Path,
    *,
    tasks_directory: str | Path,
    crop_requirements_path: str | Path,
    task_name: str | None = None,
) -> dict[str, Any]:
    """Build the curated UI catalog and distinguish catalog metadata from task data."""

    products = _load_verified_product_metadata(Path(metadata_directory))
    task_rasters: dict[tuple[str, str], dict[str, Any]] = {}
    task_rasters_by_variable: dict[tuple[str, str], dict[str, Any]] = {}
    task_derived_statuses: dict[tuple[str, str], dict[str, Any]] = {}
    task_status = "no_task_selected"
    if task_name:
        registry = TaskRegistry(tasks_directory)
        task, _, _ = registry.load_task_manifest(task_name)
        task_status = "not_standardized"
        manifest_path = task["task_directory"] / "processed" / "standardized_manifest.json"
        if manifest_path.is_file():
            _, standardized_manifest, records = registry.validate_raster_manifest(
                task_name, manifest_path
            )
            task_rasters = {
                (str(record["product_id"]), str(record["layer_id"])): record
                for record in records
            }
            task_rasters_by_variable = {
                (str(record["product_id"]), str(record["variable"])): record
                for record in records
                if isinstance(record.get("variable"), str)
            }
            statuses = standardized_manifest.get("derived_layer_statuses", [])
            if isinstance(statuses, list):
                task_derived_statuses = {
                    (str(record.get("product_id")), str(record.get("layer_id"))): record
                    for record in statuses
                    if isinstance(record, dict)
                }
            task_status = "processed"

    catalog = []
    for definition in _CATALOG_DEFINITIONS:
        variables = []
        for variable in definition["variables"]:
            product_id = variable.get("product_id")
            layer_id = variable.get("layer_id")
            product = products.get(
                variable.get("source_product_id") if variable.get("derived") else product_id
            )
            layer = product["layers"].get(layer_id) if product and layer_id else None
            task_record = task_rasters.get((product_id, layer_id))
            if variable.get("external_source") == "ISRIC/SoilGrids":
                task_record = task_rasters_by_variable.get(
                    (str(product_id), str(variable.get("id")))
                ) or task_record
                if task_record and isinstance(task_record.get("layer_id"), str):
                    layer_id = task_record["layer_id"]
            derived_status = task_derived_statuses.get((product_id, layer_id))
            requestable = bool(product and layer)
            on_task = bool(
                task_status == "processed"
                and task_record
                and (
                    requestable
                    or variable.get("derived") is True
                    or isinstance(variable.get("external_source"), str)
                )
            )
            availability_status = (
                "processed_on_task"
                if on_task
                else "requestable_not_downloaded"
                if requestable
                else str(derived_status.get("status"))
                if derived_status and isinstance(derived_status.get("status"), str)
                else "available_on_demand"
                if isinstance(variable.get("external_source"), str)
                else "derived_when_source_is_processed"
                if variable.get("derived") is True and product is not None
                else "metadata_not_verified"
                if product_id and layer_id
                else "not_integrated"
            )
            raw_companion_layers = (
                _metadata_string_list(layer.get("QualityLayers")) if layer else []
            )
            auxiliary_layers = (
                raw_companion_layers
                if product_id == "ASTGTM_NC.003" and layer_id == "ASTER_GDEM_DEM"
                else []
            )
            quality_layers = (
                []
                if auxiliary_layers
                else raw_companion_layers
            )
            quality_filter_applied = (
                task_record.get("quality_filter_applied")
                if task_record
                else None
            )
            if not quality_layers:
                quality_status = "not_required"
            elif not task_record:
                quality_status = "not_assessed"
            else:
                quality_status = task_record.get("quality_status")
                if not isinstance(quality_status, str):
                    quality_status = (
                        "filtered"
                        if quality_filter_applied is True
                        else "unfiltered"
                    )
            variables.append(
                {
                    **variable,
                    "product_id": product_id,
                    "layer_id": layer_id,
                    "depth_interval": (
                        task_record.get("depth_interval", variable.get("depth_interval"))
                        if task_record
                        else variable.get("depth_interval")
                    ),
                    "units": variable.get("units") or (
                        layer.get("Units")
                        if layer
                        else None
                    ),
                    "spatial_resolution": (
                        variable.get("spatial_resolution")
                        or (product["catalog"].get("Resolution") if product else None)
                    ),
                    "temporal_resolution": (
                        variable.get("temporal_resolution")
                        or (product["catalog"].get("TemporalGranularity") if product else None)
                    ),
                    "temporal_start": (
                        task_record.get("temporal_start")
                        if task_record and task_record.get("temporal_start")
                        else product["catalog"].get("TemporalExtentStart") if product else None
                    ),
                    "temporal_end": (
                        task_record.get("temporal_end")
                        if task_record and task_record.get("temporal_end")
                        else product["catalog"].get("TemporalExtentEnd") if product else None
                    ),
                    "scale_factor": layer.get("ScaleFactor") if layer else None,
                    "add_offset": layer.get("AddOffset") if layer else None,
                    "fill_values": _metadata_string_list(layer.get("FillValueAll") or layer.get("FillValue")) if layer else [],
                    "raw_valid_range": [
                        layer.get("OrigValidMin"),
                        layer.get("OrigValidMax"),
                    ] if layer else None,
                    "valid_range": [
                        layer.get("ValidMin"),
                        layer.get("ValidMax"),
                    ] if layer else None,
                    "quality_layers": quality_layers,
                    "auxiliary_layers": auxiliary_layers,
                    "qa_required": bool(quality_layers),
                    "quality_filter_applied": quality_filter_applied,
                    "quality_status": quality_status,
                    "quality_missing_layers": (
                        task_record.get("quality_missing_layers", [])
                        if task_record
                        else []
                    ),
                    "quality_policy": (
                        task_record.get("quality_policy", [])
                        if task_record
                        else []
                    ),
                    "requestable_from_saved_metadata": requestable,
                    "processed_on_selected_task": on_task,
                    "availability_status": availability_status,
                    "availability_detail": (
                        derived_status.get("detail") if derived_status else None
                    ),
                    "data_scope": (
                        "selected_task"
                        if product_id and (requestable or variable.get("derived") or variable.get("external_source"))
                        else "not_integrated"
                    ),
                }
            )
        processed = any(item["processed_on_selected_task"] for item in variables)
        requestable = any(item["requestable_from_saved_metadata"] for item in variables)
        on_demand = any(item["availability_status"] == "available_on_demand" for item in variables)
        invalid_coverage = any(
            item["availability_status"] == "invalid_coverage"
            for item in variables
        )
        catalog.append(
            {
                "id": definition["id"],
                "name": definition["name"],
                "category": definition["category"],
                "description": definition["description"],
                "source": definition["source"],
                "required_for_suitability": False,
                "derived": any(item.get("derived") is True for item in variables),
                "variables": variables,
                "requestable_from_saved_metadata": requestable,
                "processed_on_selected_task": processed,
                "availability_status": (
                    "processed_on_task"
                    if processed
                    else "invalid_coverage"
                    if invalid_coverage
                    else "requestable_not_downloaded"
                    if requestable
                    else "available_on_demand"
                    if on_demand
                    else "derived_when_source_is_processed"
                    if any(item["availability_status"] == "derived_when_source_is_processed" for item in variables)
                    else "not_integrated"
                ),
            }
        )

    crop_requirements = load_manifest(crop_requirements_path)
    scoring_criteria = sorted(
        {
            guidance["parameter"]
            for crop in crop_requirements.get("crops", [])
            for guidance in crop.get("guidance", [])
            if isinstance(guidance, dict)
            and isinstance(guidance.get("parameter"), str)
        }
    )
    matched_criteria = sorted(set(scoring_criteria) & SPATIAL_RASTER_VARIABLES)
    return {
        "layers": catalog,
        "logical_layer_count": len(catalog),
        "task_name": task_name,
        "task_data_status": task_status,
        "suitability_inputs": {
            "map_visibility_changes_analysis_inputs": False,
            "crop_reference_scoring_criteria": scoring_criteria,
            "spatial_raster_variables_sampled": sorted(SPATIAL_RASTER_VARIABLES),
            "matching_raster_scoring_criteria": matched_criteria,
            "required_raster_variables": [],
            "status_kind": "unsupported",
            "status": (
                "No crop-reference scoring criteria currently match sampled raster variables; "
                "spatial suitability scores are not scientifically supported by the current criteria."
                if not matched_criteria
                else "Analysis inputs come from the selected task, independently of map visibility."
            ),
        },
        "availability_note": (
            "Requestable means the saved AppEEARS metadata marks the product/layer available. "
            "Processed means it is listed in the selected task's validated standardized manifest; "
            "neither status guarantees complete valid-pixel coverage throughout the AOI. "
            "NASA POWER daily climate and ISRIC SoilGrids 2.0 soil properties are fetched "
            "only on explicit request and recorded in the selected task."
        ),
    }


def user_map_raster_keys() -> set[tuple[str, str]]:
    """Return product/layer identifiers allowed as ordinary user-facing map layers."""

    return {
        (str(variable["product_id"]), str(variable["layer_id"]))
        for definition in _CATALOG_DEFINITIONS
        for variable in definition["variables"]
        if variable.get("product_id") and variable.get("layer_id")
    }


def _metadata_string_list(value: Any) -> list[str]:
    if isinstance(value, str):
        try:
            value = ast.literal_eval(value)
        except (SyntaxError, ValueError):
            return [value] if value else []
    if value is None:
        return []
    values = value if isinstance(value, list) else [value]
    return [str(item) for item in values if isinstance(item, (str, int, float))]


def _load_verified_product_metadata(metadata_directory: Path) -> dict[str, dict[str, Any]]:
    products: dict[str, dict[str, Any]] = {}
    metadata_files = {
        "MOD13Q1.061": "MOD13Q1.json",
        "MOD11A2.061": "MOD11A2.json",
        "MOD16A2GF.061": "MOD16A2GF.json",
        "ASTGTM_NC.003": "ASTER.json",
        "MCD12Q1.061": "MCD12Q1.json",
    }
    for product_id, filename in metadata_files.items():
        path = metadata_directory / filename
        if not path.is_file():
            continue
        payload = load_manifest(path)
        for entry in payload.get("available_products", []):
            if not isinstance(entry, dict):
                continue
            catalog = entry.get("catalog")
            layers = entry.get("layers")
            if (
                isinstance(catalog, dict)
                and catalog.get("ProductAndVersion") == product_id
                and catalog.get("Available") is True
                and isinstance(layers, dict)
            ):
                products[product_id] = {
                    "catalog": catalog,
                    "layers": {
                        key: value
                        for key, value in layers.items()
                        if isinstance(key, str)
                        and isinstance(value, dict)
                        and value.get("Available") is True
                        and value.get("IsQA") is not True
                    },
                }
    return products
