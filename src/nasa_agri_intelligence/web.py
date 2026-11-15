from __future__ import annotations

import hashlib
import io
import math
import ast
import re
import shutil
import uuid
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Literal

import numpy as np
import rasterio
import requests
from fastapi import BackgroundTasks, FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from PIL import Image
from pydantic import BaseModel, ConfigDict, Field, model_validator
from pyproj import CRS, Transformer
from pyproj.exceptions import CRSError
from rasterio.enums import Resampling
from rasterio.transform import from_bounds
from rasterio.vrt import WarpedVRT
from shapely import transform as transform_geometry
from shapely.geometry import shape
from shapely.ops import unary_union

from .collection.appeears_client import AppEEARSClient
from .collection.downloader import DownloadManager
from .collection.manifest import load_manifest, write_manifest
from .collection.power_climate import collect_power_climate
from .collection.gaez_v5 import (
    gaez_catalog_response,
    load_gaez_catalog,
    load_gaez_run,
    make_gaez_run_id,
    run_gaez_v5_suitability,
)
from .collection.soilgrids import collect_soilgrids
from .collection.product_discovery import DatasetConfig, DatasetSpec, ProductDiscovery
from .collection.task_manager import (
    AOISettings,
    CollectionSettings,
    DuplicateTaskError,
    SpatialAnalysisSettings,
    TaskManager,
    normalize_aoi,
)
from .collection.task_monitor import TaskMonitor, normalize_task_status
from .collection.task_registry import TaskRegistry
from .crop_requirements import load_crop_requirements
from .essential_layers import build_essential_layer_catalog, user_map_raster_keys
from .spatial_harmonization import build_grid
from .standardization import standardize_task
from .suitability.engine import evaluate_cell
from .suitability.spatial import run_spatial_suitability, suitability_layers


class AOIStatisticsRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str = Field(default="Untitled area", min_length=1, max_length=120)
    geojson: dict[str, Any]
    analysis_crs: str = "EPSG:6933"
    resolution_m: float = Field(default=250, gt=0, le=100_000)


class ProductLayerSelection(BaseModel):
    model_config = ConfigDict(extra="forbid")

    product_id: str = Field(min_length=1, max_length=100)
    layer_ids: list[str] = Field(min_length=1)


class TaskSubmissionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    aoi_name: str = Field(min_length=1, max_length=120)
    task_name: str | None = Field(default=None, min_length=1, max_length=120)
    aoi_geojson: dict[str, Any]
    start_date: date
    end_date: date
    analysis_crs: str = "EPSG:6933"
    resolution_m: float = Field(default=250, gt=0, le=100_000)
    products: list[ProductLayerSelection] = Field(min_length=1)
    confirm_real_submission: bool = False

    @model_validator(mode="after")
    def validate_request(self) -> TaskSubmissionRequest:
        if self.end_date < self.start_date:
            raise ValueError("end_date must be on or after start_date.")
        if not self.aoi_name.strip():
            raise ValueError("aoi_name cannot be empty.")
        if self.task_name is not None and not self.task_name.strip():
            raise ValueError("task_name cannot be empty.")
        product_ids = [product.product_id for product in self.products]
        if len(product_ids) != len(set(product_ids)):
            raise ValueError("Each product may only appear once per request.")
        for product in self.products:
            if len(product.layer_ids) != len(set(product.layer_ids)):
                raise ValueError(f"Duplicate layer selected for {product.product_id}.")
        return self


class AOITaskCreateRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    aoi_name: str = Field(min_length=1, max_length=120)
    task_name: str | None = Field(default=None, min_length=1, max_length=120)
    aoi_geojson: dict[str, Any]
    start_date: date
    end_date: date
    analysis_crs: str = "EPSG:6933"
    resolution_m: float = Field(default=250, gt=0, le=100_000)
    soil_properties: bool = False
    soil_depth: str = "0-5cm"
    nasa_power_temperature: bool = False
    nasa_power_precipitation: bool = False
    gaez_v5: bool = False
    products: list[ProductLayerSelection] = Field(default_factory=list)
    confirm_acquisition: bool = False

    @model_validator(mode="after")
    def validate_request(self) -> AOITaskCreateRequest:
        if self.end_date < self.start_date:
            raise ValueError("end_date must be on or after start_date.")
        if not self.aoi_name.strip():
            raise ValueError("aoi_name cannot be empty.")
        if self.task_name is not None and not self.task_name.strip():
            raise ValueError("task_name cannot be empty.")
        if not (
            self.soil_properties
            or self.nasa_power_temperature
            or self.nasa_power_precipitation
            or self.gaez_v5
            or self.products
        ):
            raise ValueError("Select at least one environmental dataset to acquire.")
        if not self.confirm_acquisition:
            raise ValueError("Confirm the selected real data acquisition request.")
        if self.soil_depth not in {
            "0-5cm", "5-15cm", "15-30cm", "30-60cm", "60-100cm", "100-200cm"
        }:
            raise ValueError("Unsupported SoilGrids depth interval.")
        product_ids = [product.product_id for product in self.products]
        if len(product_ids) != len(set(product_ids)):
            raise ValueError("Each AppEEARS product may only appear once per request.")
        for product in self.products:
            if len(product.layer_ids) != len(set(product.layer_ids)):
                raise ValueError(f"Duplicate layer selected for {product.product_id}.")
        return self


class SuitabilityAnalysisRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    task_name: str = Field(min_length=1, max_length=120)
    analysis_name: str | None = Field(default=None, min_length=1, max_length=120)
    crop_ids: list[str] | None = None
    overwrite: bool = False


class GaezSuitabilityRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    crop_codes: list[str] | None = Field(default=None, min_length=1, max_length=77)
    water_regime: Literal["rainfed", "irrigated", "both"] = "both"
    management_level: Literal["low", "high"] = "high"


def create_app(
    *,
    settings_path: str | Path = "config/settings.yaml",
    datasets_path: str | Path = "config/datasets.yaml",
    project_root: str | Path | None = None,
) -> FastAPI:
    app = FastAPI(
        title="NASA Agricultural Intelligence API",
        version="0.1.0",
        description="Web API for the existing NASA/AppEEARS agricultural data pipeline.",
    )
    root = Path(project_root or Path.cwd()).resolve()
    app.state.project_root = root
    loaded_settings = CollectionSettings.load(_resolve_config(root, settings_path))
    settings_data = loaded_settings.model_dump()
    for key in ("directory", "manifests_directory", "metadata_directory"):
        settings_data["output"][key] = _resolve_config(root, settings_data["output"][key])
    settings_data["spatial_analysis"]["output_directory"] = _resolve_config(
        root, settings_data["spatial_analysis"]["output_directory"]
    )
    app.state.settings = CollectionSettings.model_validate(settings_data)
    app.state.dataset_config = ProductDiscovery.load_config(
        _resolve_config(root, datasets_path)
    )
    app.state.tasks_directory = root / "data" / "tasks"
    TaskRegistry(app.state.tasks_directory)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=[
            "http://localhost:5173",
            "http://127.0.0.1:5173",
            "http://localhost:5174",
            "http://127.0.0.1:5174",
            "http://localhost:5175",
            "http://127.0.0.1:5175",
        ],
        allow_methods=["GET", "POST"],
        allow_headers=["Content-Type"],
    )

    @app.get("/api/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/api/settings")
    def get_settings() -> dict[str, Any]:
        settings: CollectionSettings = app.state.settings
        return {
            "analysis_crs": settings.spatial_analysis.analysis_crs,
            "resolution_m": settings.spatial_analysis.cell_size_m,
            "max_grid_cells": settings.spatial_analysis.max_grid_cells,
            "start_date": settings.start_date.isoformat(),
            "end_date": settings.end_date.isoformat(),
            "default_aoi": settings.aoi.model_dump(mode="json"),
        }

    @app.get("/api/datasets")
    def get_datasets() -> dict[str, Any]:
        return {"products": _available_products(app)}

    @app.get("/api/essential-layers")
    def get_essential_layers(
        task_name: str | None = Query(default=None, min_length=1),
    ) -> dict[str, Any]:
        settings: CollectionSettings = app.state.settings
        try:
            return build_essential_layer_catalog(
                settings.output.metadata_directory,
                tasks_directory=_task_registry(app).root,
                crop_requirements_path=(
                    app.state.project_root / "data/reference/crop_requirements.json"
                ),
                task_name=task_name,
            )
        except FileNotFoundError as error:
            raise HTTPException(status_code=404, detail=str(error)) from error
        except ValueError as error:
            raise HTTPException(status_code=422, detail=str(error)) from error

    @app.get("/api/datasets/{product_id}")
    def get_dataset(product_id: str) -> dict[str, Any]:
        product = next(
            (
                item
                for item in _available_products(app)
                if item["product_id"] == product_id
            ),
            None,
        )
        if product is None:
            raise HTTPException(status_code=404, detail="Available product not found.")
        return product

    @app.post("/api/aoi/statistics")
    def get_aoi_statistics(request: AOIStatisticsRequest) -> dict[str, Any]:
        try:
            normalized = normalize_aoi(request.geojson)
            analysis_crs = _validate_analysis_crs(request.analysis_crs)
            geometry = unary_union(
                [shape(feature["geometry"]) for feature in normalized["features"]]
            )
            transformer = Transformer.from_crs(
                "EPSG:4326", analysis_crs, always_xy=True
            )
            projected = transform_geometry(
                geometry,
                transformer.transform,
                interleaved=False,
            )
            grid = build_grid(
                normalized,
                target_crs=analysis_crs.to_string(),
                resolution=request.resolution_m,
            )
        except (ValueError, CRSError) as error:
            raise HTTPException(status_code=422, detail=str(error)) from error

        geographic_bounds = geometry.bounds
        candidate_cells = grid.width * grid.height
        settings: CollectionSettings = app.state.settings
        return {
            "name": request.name.strip(),
            "geometry_type": geometry.geom_type,
            "bounds": {
                "west": geographic_bounds[0],
                "south": geographic_bounds[1],
                "east": geographic_bounds[2],
                "north": geographic_bounds[3],
            },
            "area_km2": projected.area / 1_000_000,
            "estimated_grid_cells": candidate_cells,
            "resolution_m": request.resolution_m,
            "analysis_crs": analysis_crs.to_string(),
            "grid_limit_exceeded": candidate_cells
            > settings.spatial_analysis.max_grid_cells,
        }

    @app.get("/api/crops")
    def get_crops() -> dict[str, Any]:
        requirements = load_crop_requirements(
            app.state.project_root / "data/reference/crop_requirements.json"
        )
        suitability_available = _has_verified_suitability_criteria(app)
        return {
            "suitability_available": suitability_available,
            "suitability_status": (
                "Verified environmental crop criteria are available."
                if suitability_available
                else (
                    "Crop suitability is disabled: the available crop references contain "
                    "irrigation guidance, not verified environmental suitability thresholds."
                )
            ),
            "model": "transparent multi-criteria scoring",
            "crops": [
                {"id": crop["id"], "name": crop["name"]}
                for crop in requirements["crops"]
            ],
        }

    @app.get("/api/suitability")
    def get_suitability_summary(task_name: str = Query(min_length=1)) -> dict[str, Any]:
        suitability_available = _has_verified_suitability_criteria(app)
        summary = {
            "available": suitability_available,
            "model": "transparent multi-criteria scoring",
            "status": "Verified environmental suitability criteria are available."
            if suitability_available
            else (
                "No verified environmental suitability criteria are configured; "
                "no crop recommendation can be generated."
            ),
            "inputs": [
                "temperature",
                "water",
                "soil",
                "terrain",
                "land_cover",
            ],
        }
        registry = _task_registry(app)
        try:
            task, _, task_manifest = registry.load_task_manifest(task_name)
        except (FileNotFoundError, ValueError) as error:
            raise HTTPException(status_code=404, detail=str(error)) from error
        summary.update(
            {
                "task_name": task["task_name"],
                "task_id": task.get("task_id"),
                "analysis_runs": task_manifest.get("analysis_runs", []),
                "result_count": len(task_manifest.get("analysis_runs", [])),
            }
        )
        return summary

    @app.post("/api/suitability")
    @app.post("/api/suitability/evaluate")
    def evaluate_crop_suitability(
        request: dict[str, Any],
    ) -> dict[str, Any]:
        if not _has_verified_suitability_criteria(app):
            raise HTTPException(
                status_code=422,
                detail=(
                    "Crop suitability scoring is unavailable because no verified "
                    "environmental suitability criteria are configured."
                ),
            )
        profile = request.get("environment", {}) if isinstance(request, dict) else {}
        if not isinstance(profile, dict):
            raise HTTPException(status_code=422, detail="environment must be an object.")

        selected = request.get("crops") if isinstance(request, dict) else None
        if selected is None:
            selected = None
        elif isinstance(selected, str):
            selected = [selected]
        elif not isinstance(selected, list):
            raise HTTPException(status_code=422, detail="crops must be a list or single crop id.")

        result = evaluate_cell(
            {str(key): (float(value) if value is not None else None) for key, value in profile.items()},
            crop_ids=selected,
            requirements_path=app.state.project_root / "data/reference/crop_requirements.json",
        )
        return {
            "best_crop": result["best_crop"],
            "results": result["results"],
            "legend": result["legend"],
            "crop_count": result["crop_count"],
        }

    @app.get("/api/tasks")
    def get_tasks(include_remote: bool = Query(default=False)) -> dict[str, Any]:
        tasks, warnings = _local_tasks(app)
        if include_remote:
            try:
                manager = _task_manager(app, AppEEARSClient())
                tasks = _merge_remote_tasks(
                    tasks,
                    manager.remote_tasks(),
                    hidden_task_ids=_linked_provider_task_ids(app),
                )
            except (OSError, ValueError, requests.RequestException) as error:
                raise HTTPException(status_code=502, detail=str(error)) from error
        return {"tasks": tasks, "warnings": warnings}

    @app.get("/api/tasks/{task_id}")
    def get_task(task_id: str) -> dict[str, Any]:
        _validate_task_id(task_id)
        try:
            _, _, manifest = _task_registry(app).load_task_manifest(task_id)
        except (FileNotFoundError, ValueError) as error:
            raise HTTPException(status_code=404, detail=str(error)) from error
        return manifest

    @app.get("/api/gaez-v5/catalog")
    def get_gaez_catalog() -> dict[str, Any]:
        try:
            catalog = load_gaez_catalog(
                app.state.project_root / "data/reference/gaez_v5_crops.json"
            )
        except (OSError, ValueError) as error:
            raise HTTPException(status_code=500, detail=str(error)) from error
        return gaez_catalog_response(catalog)

    @app.post("/api/aoi-tasks/{task_id}/gaez-suitability", status_code=202)
    def start_gaez_suitability(
        task_id: str,
        request: GaezSuitabilityRequest,
        background_tasks: BackgroundTasks,
    ) -> dict[str, Any]:
        _validate_task_id(task_id)
        registry = _task_registry(app)
        try:
            task, manifest_path, manifest = registry.load_task_manifest(task_id)
            catalog = load_gaez_catalog(
                app.state.project_root / "data/reference/gaez_v5_crops.json"
            )
            crop_codes = request.crop_codes or [
                crop["code"] for crop in catalog["crops"]
            ]
            if len(crop_codes) != len(set(crop_codes)):
                raise HTTPException(
                    status_code=422,
                    detail="GAEZ crop codes must not contain duplicates.",
                )
            supported_codes = {crop["code"] for crop in catalog["crops"]}
            unknown_codes = sorted(set(crop_codes) - supported_codes)
            if unknown_codes:
                raise HTTPException(
                    status_code=422,
                    detail=f"Unsupported GAEZ crop codes: {', '.join(unknown_codes)}.",
                )
            run_id = make_gaez_run_id(
                crop_codes,
                request.water_regime,
                request.management_level,
            )
            if task.get("task_type") != "aoi_analysis" or manifest.get("task_type") != "aoi_analysis":
                raise HTTPException(
                    status_code=400,
                    detail="GAEZ v5 suitability is available for unified AOI tasks.",
                )
            provider_states = manifest.setdefault("acquisition", {}).setdefault(
                "providers", {}
            )
            gaez_state = provider_states.setdefault("gaez_v5", {"status": "not_selected"})
            if gaez_state.get("status") in {"queued", "running"}:
                raise HTTPException(
                    status_code=409,
                    detail="A GAEZ v5 operation is already in progress for this AOI task.",
                )
            existing = next(
                (
                    item
                    for item in reversed(manifest.get("gaez_runs", []))
                    if isinstance(item, dict) and item.get("run_id") == run_id
                ),
                None,
            )
            if existing is not None and existing.get("status") == "completed":
                return {"run_id": run_id, "status": existing["status"], "cached": True}
            gaez_state.update(
                {
                    "status": "queued",
                    "run_id": run_id,
                    "stage": "preparing_aoi",
                    "operations": [],
                    "error": None,
                }
            )
            manifest["acquisition"]["status"] = "running"
            manifest["acquisition"]["stage"] = "acquiring_suitability"
            manifest["task_status"] = "processing"
            manifest["updated_at"] = datetime.now(timezone.utc).isoformat()
            write_manifest(manifest_path, manifest)
        except HTTPException:
            raise
        except FileNotFoundError as error:
            raise HTTPException(status_code=404, detail=str(error)) from error
        except (OSError, ValueError) as error:
            raise HTTPException(status_code=422, detail=str(error)) from error

        background_tasks.add_task(
            _run_gaez_v5_provider,
            app,
            task_id,
            run_id,
            crop_codes,
            request.water_regime,
            request.management_level,
        )
        return {"run_id": run_id, "status": "queued", "cached": False}

    @app.get("/api/aoi-tasks/{task_id}/gaez-suitability")
    def get_gaez_suitability(
        task_id: str,
        run_id: str | None = Query(default=None, pattern=r"GAEZ_[a-f0-9]{16}"),
    ) -> dict[str, Any]:
        _validate_task_id(task_id)
        try:
            return load_gaez_run(_task_registry(app), task_id, run_id)
        except FileNotFoundError as error:
            raise HTTPException(status_code=404, detail=str(error)) from error
        except ValueError as error:
            raise HTTPException(status_code=422, detail=str(error)) from error

    @app.get("/api/task-names/{task_name}")
    def get_task_by_name(task_name: str) -> dict[str, Any]:
        try:
            _, _, manifest = _task_registry(app).load_task_manifest(task_name)
        except (FileNotFoundError, ValueError) as error:
            raise HTTPException(status_code=404, detail=str(error)) from error
        return manifest

    @app.post("/api/aoi-tasks", status_code=202)
    def create_aoi_task(
        request: AOITaskCreateRequest,
        background_tasks: BackgroundTasks,
    ) -> dict[str, Any]:
        try:
            normalized_aoi = normalize_aoi(request.aoi_geojson)
            analysis_crs = _validate_analysis_crs(request.analysis_crs)
            grid = build_grid(
                normalized_aoi,
                target_crs=analysis_crs.to_string(),
                resolution=request.resolution_m,
            )
            if grid.width * grid.height > app.state.settings.spatial_analysis.max_grid_cells:
                raise ValueError(
                    "Estimated grid exceeds the configured cell limit; "
                    "increase the resolution or reduce the AOI."
                )
            registry = _task_registry(app)
            task_name = request.task_name.strip() if request.task_name else ""
            if not task_name:
                task_name = (
                    f"aoi-{datetime.now(timezone.utc):%Y%m%dT%H%M%S}-"
                    f"{uuid.uuid4().hex[:8]}"
                )
            task_name = registry.validate_name(task_name)
            task_id = str(uuid.uuid4())
            created_at = datetime.now(timezone.utc).isoformat()
            provider_states = {
                "soilgrids": {"status": "queued" if request.soil_properties else "not_selected"},
                "nasa_power": {
                    "status": "queued"
                    if request.nasa_power_temperature or request.nasa_power_precipitation
                    else "not_selected"
                },
                "appeears": {
                    "status": "queued" if request.products else "not_selected",
                    "task_ids": [],
                    "failures": [],
                },
                "gaez_v5": {
                    "status": "queued" if request.gaez_v5 else "not_selected"
                },
            }
            task_record = {
                "task_id": task_id,
                "task_type": "aoi_analysis",
                "created_at": created_at,
                "updated_at": created_at,
                "task_status": "queued",
                "download_status": "pending",
                "preprocessing_status": "completed",
                "analysis_runs": [],
                "task_directory": str(registry.root / task_name),
            }
            task_root = registry.reserve(task_name, task_record)
            task_manifest = {
                "task_type": "aoi_analysis",
                "task_name": task_name,
                "task_id": task_id,
                "product_id": None,
                "dataset": "AOI environmental acquisition",
                "output_directory": str(task_root / "raw"),
                "source_directory": str(task_root / "raw"),
                "layers": [],
                "aoi": {
                    "name": request.aoi_name.strip(),
                    "crs": "EPSG:4326",
                    "geojson": normalized_aoi,
                },
                "start_date": request.start_date.isoformat(),
                "end_date": request.end_date.isoformat(),
                "spatial_analysis": {
                    "analysis_crs": analysis_crs.to_string(),
                    "requested_cell_size_m": request.resolution_m,
                },
                "selected_datasets": {
                    "soil_properties": request.soil_properties,
                    "soil_depth": request.soil_depth if request.soil_properties else None,
                    "nasa_power_temperature": request.nasa_power_temperature,
                    "nasa_power_precipitation": request.nasa_power_precipitation,
                    "gaez_v5": request.gaez_v5,
                    "appeears": [
                        selection.model_dump() for selection in request.products
                    ],
                },
                "submitted_at": created_at,
                "created_at": created_at,
                "updated_at": created_at,
                "task_status": "queued",
                "submission_status": "local_task_created",
                "download_status": "pending",
                "preprocessing_status": "completed",
                "analysis_runs": [],
                "downloads": [],
                "external_downloads": [],
                "acquisition": {
                    "status": "queued",
                    "stage": "preparing_aoi",
                    "providers": provider_states,
                },
                "appeears_tasks": [],
                "error": None,
            }
            write_manifest(task_root / "task_manifest.json", task_manifest)
            write_manifest(
                task_root / "task_metadata.json",
                {
                    "task_type": "aoi_analysis",
                    "task_name": task_name,
                    "task_id": task_id,
                    "created_at": created_at,
                    "aoi": task_manifest["aoi"],
                    "selected_datasets": task_manifest["selected_datasets"],
                    "start_date": task_manifest["start_date"],
                    "end_date": task_manifest["end_date"],
                    "source_directory": str(task_root / "raw"),
                    "local_task_directory": str(task_root),
                },
            )
            raster_manifest_path = task_root / "processed" / "standardized_manifest.json"
            write_manifest(
                raster_manifest_path,
                {
                    "task_type": "aoi_analysis",
                    "task_name": task_name,
                    "task_id": task_id,
                    "created_at": created_at,
                    "source_directory": str(task_root / "raw"),
                    "output_directory": str(task_root / "processed" / "rasters"),
                    "raster_count": 0,
                    "rasters": [],
                    "records": [],
                    "external_integrations": [],
                },
            )
        except FileExistsError as error:
            raise HTTPException(status_code=409, detail=str(error)) from error
        except (OSError, ValueError, CRSError) as error:
            raise HTTPException(status_code=422, detail=str(error)) from error

        background_tasks.add_task(
            _run_aoi_task_acquisition,
            app,
            task_name,
            request,
            normalized_aoi,
            analysis_crs.to_string(),
        )
        return {
            "task_name": task_name,
            "task_id": task_id,
            "aoi": task_manifest["aoi"],
            "task_status": "queued",
            "acquisition_status": "queued",
            "selected_datasets": task_manifest["selected_datasets"],
        }

    @app.post("/api/aoi-tasks/{task_id}/retry", status_code=202)
    def retry_aoi_task(
        task_id: str,
        background_tasks: BackgroundTasks,
    ) -> dict[str, Any]:
        _validate_task_id(task_id)
        registry = _task_registry(app)
        try:
            parent, manifest_path, manifest = registry.load_task_manifest(task_id)
            if manifest.get("task_type") != "aoi_analysis":
                raise HTTPException(
                    status_code=400,
                    detail="Provider retries are available for AOI analysis tasks.",
                )
            acquisition = manifest.get("acquisition", {})
            provider_states = acquisition.get("providers", {})
            if any(
                isinstance(state, dict) and state.get("status") in {"queued", "running"}
                for state in provider_states.values()
            ):
                raise HTTPException(
                    status_code=409,
                    detail="A provider operation is already in progress.",
                )
            providers = {
                name
                for name, state in provider_states.items()
                if isinstance(state, dict) and state.get("status") in {"failed", "partial"}
            }
            if not providers:
                raise HTTPException(
                    status_code=409,
                    detail="The AOI task has no failed provider operations to retry.",
                )

            selected = manifest.get("selected_datasets")
            if not isinstance(selected, dict):
                raise ValueError("AOI task has invalid selected-dataset metadata.")
            request = AOITaskCreateRequest.model_validate(
                {
                    "aoi_name": manifest["aoi"]["name"],
                    "task_name": manifest["task_name"],
                    "aoi_geojson": manifest["aoi"]["geojson"],
                    "start_date": manifest["start_date"],
                    "end_date": manifest["end_date"],
                    "analysis_crs": manifest["spatial_analysis"]["analysis_crs"],
                    "resolution_m": manifest["spatial_analysis"]["requested_cell_size_m"],
                    "soil_properties": selected.get("soil_properties", False),
                    "soil_depth": selected.get("soil_depth") or "0-5cm",
                    "nasa_power_temperature": selected.get("nasa_power_temperature", False),
                    "nasa_power_precipitation": selected.get("nasa_power_precipitation", False),
                    "gaez_v5": selected.get("gaez_v5", False),
                    "products": selected.get("appeears", []),
                    "confirm_acquisition": True,
                }
            )
            normalized_aoi = normalize_aoi(request.aoi_geojson)
            analysis_crs = _validate_analysis_crs(request.analysis_crs)
            for provider in providers:
                provider_states[provider]["status"] = "queued"
                provider_states[provider]["error"] = None
            acquisition["status"] = "queued"
            acquisition["stage"] = "retrying_failed_providers"
            manifest["task_status"] = "queued"
            manifest["updated_at"] = datetime.now(timezone.utc).isoformat()
            write_manifest(manifest_path, manifest)
            registry.update(
                str(parent["task_name"]),
                task_status="queued",
                acquisition_status="queued",
            )
        except HTTPException:
            raise
        except (FileNotFoundError, KeyError, TypeError, ValueError, CRSError) as error:
            raise HTTPException(status_code=422, detail=str(error)) from error

        background_tasks.add_task(
            _run_aoi_task_acquisition,
            app,
            str(parent["task_name"]),
            request,
            normalized_aoi,
            analysis_crs.to_string(),
            providers=providers,
        )
        return {
            "task_name": manifest["task_name"],
            "task_id": manifest["task_id"],
            "acquisition_status": "queued",
            "retried_providers": sorted(providers),
        }

    @app.post("/api/tasks")
    def submit_tasks(request: TaskSubmissionRequest) -> dict[str, Any]:
        if not request.confirm_real_submission:
            raise HTTPException(
                status_code=400,
                detail="Explicit confirmation is required before creating NASA tasks.",
            )
        try:
            normalized_aoi = normalize_aoi(request.aoi_geojson)
            analysis_crs = _validate_analysis_crs(request.analysis_crs)
            settings: CollectionSettings = app.state.settings
            grid = build_grid(
                normalized_aoi,
                target_crs=analysis_crs.to_string(),
                resolution=request.resolution_m,
            )
            if grid.width * grid.height > settings.spatial_analysis.max_grid_cells:
                raise ValueError(
                    "Estimated grid exceeds the configured cell limit; "
                    "increase the resolution or reduce the AOI."
                )
        except (ValueError, CRSError) as error:
            raise HTTPException(status_code=422, detail=str(error)) from error

        try:
            client = AppEEARSClient()
            managers = [
                _manager_for_submission(
                    app,
                    client,
                    request,
                    normalized_aoi,
                    analysis_crs.to_string(),
                    selection,
                )
                for selection in request.products
            ]
            task_names = [
                _submission_task_name(request.task_name, selection.product_id, len(request.products))
                for selection in request.products
            ]
            for task_name, manager in zip(task_names, managers, strict=True):
                if task_name is not None and (
                    (manager.tasks_directory / task_name).exists()
                    or any(
                        item.get("task_name") == task_name
                        for item in manager.registry.records()
                    )
                ):
                    raise DuplicateTaskError(f"Task name already exists: {task_name}")
            for selection, manager, task_name in zip(
                request.products, managers, task_names, strict=True
            ):
                task_request = manager.build_request(selection.product_id, task_name=task_name)
                duplicates = manager.find_duplicates(task_request)
                if duplicates:
                    raise DuplicateTaskError(
                        f"Equivalent request already exists for "
                        f"{selection.product_id}: {', '.join(duplicates)}."
                    )

            submitted: list[dict[str, Any]] = []
            failures: list[dict[str, str]] = []
            for selection, manager, task_name in zip(
                request.products, managers, task_names, strict=True
            ):
                try:
                    submitted.append(manager.submit(selection.product_id, task_name=task_name))
                except FileExistsError as error:
                    raise DuplicateTaskError(str(error)) from error
                except (DuplicateTaskError, OSError, ValueError, requests.RequestException) as error:
                    failures.append(
                        {
                            "product_id": selection.product_id,
                            "error": f"{type(error).__name__}: {error}",
                        }
                    )
                    break
        except DuplicateTaskError as error:
            raise HTTPException(status_code=409, detail=str(error)) from error
        except (OSError, ValueError, requests.RequestException) as error:
            raise HTTPException(status_code=422, detail=str(error)) from error
        return {
            "submitted": submitted,
            "failures": failures,
            "partial": bool(submitted and failures),
        }

    @app.post("/api/tasks/{task_id}/monitor")
    def monitor_task(task_id: str) -> dict[str, Any]:
        _validate_task_id(task_id)
        try:
            manager = _task_manager(app, AppEEARSClient())
            try:
                manager.get_manifest(task_id)
            except FileNotFoundError:
                return TaskMonitor(manager).check_remote_once(task_id)
            return TaskMonitor(manager).check_once(task_id)
        except FileNotFoundError as error:
            raise HTTPException(status_code=404, detail=str(error)) from error
        except (OSError, ValueError, requests.RequestException) as error:
            raise HTTPException(status_code=502, detail=str(error)) from error

    @app.post("/api/tasks/{task_id}/download")
    def download_task(task_id: str) -> dict[str, Any]:
        _validate_task_id(task_id)
        try:
            manager = _task_manager(app, AppEEARSClient())
            return DownloadManager(manager).download_task(task_id)
        except FileNotFoundError as error:
            raise HTTPException(status_code=404, detail=str(error)) from error
        except (OSError, ValueError, requests.RequestException) as error:
            raise HTTPException(status_code=502, detail=str(error)) from error

    @app.post("/api/aoi-tasks/{task_id}/download")
    def download_aoi_bundle(task_id: str) -> dict[str, Any]:
        _validate_task_id(task_id)
        registry = _task_registry(app)
        try:
            parent, _, manifest = registry.load_task_manifest(task_id)
            if manifest.get("task_type") != "aoi_analysis":
                raise HTTPException(
                    status_code=400,
                    detail="Download-all is available for AOI analysis tasks.",
                )
            references = manifest.get("appeears_tasks", [])
            if not isinstance(references, list):
                raise ValueError("AOI task has an invalid AppEEARS child-task list.")

            child_tasks: list[dict[str, Any]] = []
            seen_ids: set[str] = set()
            for reference in references:
                if not isinstance(reference, dict) or not isinstance(reference.get("task_id"), str):
                    raise ValueError("AOI task contains an invalid AppEEARS child-task reference.")
                child_id = reference["task_id"]
                if child_id in seen_ids:
                    continue
                seen_ids.add(child_id)
                child, _, child_manifest = registry.load_task_manifest(child_id)
                if child_manifest.get("parent_aoi_task") != {
                    "task_name": parent["task_name"],
                    "task_id": manifest["task_id"],
                }:
                    raise ValueError(
                        f"AppEEARS child task {child_id} is not linked to this AOI task."
                    )
                child_tasks.append(child)

            download_errors: list[dict[str, str]] = []
            integrated_rasters = 0
            manager = _task_manager(app, AppEEARSClient())
            for child in child_tasks:
                child_id = child.get("task_id")
                if not isinstance(child_id, str):
                    raise ValueError("Registered AppEEARS child task has no task ID.")
                try:
                    DownloadManager(manager).download_task(child_id)
                    child_raster_manifest = (
                        child["task_directory"]
                        / "processed"
                        / "standardized_manifest.json"
                    )
                    if child_raster_manifest.is_file():
                        _, _, child_rasters = registry.validate_raster_manifest(
                            str(child["task_name"]),
                            child_raster_manifest,
                        )
                    else:
                        try:
                            standardize_task(
                                child["task_directory"],
                                task_name=str(child["task_name"]),
                                task_id=child_id,
                                metadata_directory=app.state.settings.output.metadata_directory,
                            )
                        except FileExistsError:
                            standardize_task(
                                child["task_directory"],
                                task_name=str(child["task_name"]),
                                task_id=child_id,
                                metadata_directory=app.state.settings.output.metadata_directory,
                                overwrite=True,
                            )
                        _, _, child_rasters = registry.validate_raster_manifest(
                            str(child["task_name"]),
                            child_raster_manifest,
                        )
                    integrated_rasters += _integrate_child_rasters(
                        parent=parent,
                        child=child,
                        parent_manifest=manifest,
                        child_rasters=child_rasters,
                        registry=registry,
                    )
                except (
                    FileNotFoundError,
                    OSError,
                    ValueError,
                    requests.RequestException,
                    rasterio.errors.RasterioError,
                ) as error:
                    download_errors.append({
                        "task_name": str(child.get("task_name", child_id)),
                        "error": f"{type(error).__name__}: {error}",
                    })

            _, _, parent_rasters = registry.validate_raster_manifest(
                str(parent["task_name"]),
                parent["task_directory"]
                / "processed"
                / "standardized_manifest.json",
            )
            operation_states: list[dict[str, Any]] = []
            pending_child_names: set[str] = set()
            completed_child_ids: set[str] = set()
            for child in child_tasks:
                child_id = str(child["task_id"])
                _, _, child_manifest = registry.load_task_manifest(child_id)
                child_status = child_manifest.get("task_status", "unknown")
                operation_states.append(
                    {
                        "task_id": child_id,
                        "task_name": child.get("task_name"),
                        "product_id": child_manifest.get("product_id"),
                        "status": child_status,
                        "download_status": child_manifest.get("download_status"),
                        "error": child_manifest.get("error"),
                    }
                )
                if child_status not in {"completed", "failed", "expired"}:
                    pending_child_names.add(str(child.get("task_name", child_id)))
                elif (
                    child_status == "completed"
                    and child_manifest.get("download_status") == "completed"
                ):
                    completed_child_ids.add(child_id)

            pending_children = bool(pending_child_names)
            reported_download_errors = [
                failure
                for failure in download_errors
                if failure["task_name"] not in pending_child_names
            ]
            if references:
                acquisition = manifest.setdefault(
                    "acquisition",
                    {"status": "running", "stage": "awaiting_provider_completion", "providers": {}},
                )
                provider_states = acquisition.setdefault("providers", {})
                appeears_state = provider_states.setdefault(
                    "appeears",
                    {"status": "submitted", "task_ids": [], "failures": []},
                )
                appeears_state["operations"] = operation_states
                if pending_children:
                    appeears_state["status"] = "submitted"
                    acquisition["status"] = "running"
                    acquisition["stage"] = "awaiting_provider_completion"
                    manifest["task_status"] = "processing"
                elif reported_download_errors:
                    appeears_state["status"] = (
                        "partial" if completed_child_ids else "failed"
                    )
                    appeears_state["error"] = "; ".join(
                        failure["error"] for failure in reported_download_errors
                    )
                else:
                    appeears_state["status"] = "completed"
                    appeears_state["error"] = None

                if not pending_children:
                    provider_statuses = [
                        state["status"]
                        for state in provider_states.values()
                        if state["status"] != "not_selected"
                    ]
                    acquisition["status"] = (
                        "completed"
                        if all(status == "completed" for status in provider_statuses)
                        else "partial"
                        if any(
                            status in {"completed", "partial", "submitted"}
                            for status in provider_statuses
                        )
                        else "failed"
                    )
                    acquisition["stage"] = (
                        "completed"
                        if acquisition["status"] == "completed"
                        else "completed_with_warnings"
                        if acquisition["status"] == "partial"
                        else "failed"
                    )
                    manifest["task_status"] = acquisition["status"]

            manifest["analysis_data"] = {
                "status": (
                    "pending"
                    if pending_children
                    else "partial"
                    if reported_download_errors
                    else "ready"
                ),
                "linked_task_count": len(child_tasks),
                "integrated_raster_count": len(parent_rasters),
                "last_updated": datetime.now(timezone.utc).isoformat(),
                "failures": reported_download_errors,
            }
            manifest["updated_at"] = datetime.now(timezone.utc).isoformat()
            write_manifest(
                registry.root / str(parent["task_name"]) / "task_manifest.json",
                manifest,
            )
            acquisition_status = manifest.get("acquisition", {}).get(
                "status", manifest.get("task_status", "unknown")
            )
            registry.update(
                str(parent["task_name"]),
                task_status=manifest["task_status"],
                acquisition_status=acquisition_status,
            )
            result_status = (
                "pending"
                if pending_children
                else "partial"
                if reported_download_errors
                else "ready"
            )
            return {
                "task_name": parent["task_name"],
                "status": result_status,
                "linked_task_count": len(child_tasks),
                "integrated_raster_count": len(parent_rasters),
                "newly_integrated_raster_count": integrated_rasters,
                "failures": reported_download_errors,
            }
        except HTTPException:
            raise
        except FileNotFoundError as error:
            raise HTTPException(status_code=404, detail=str(error)) from error
        except (OSError, ValueError, requests.RequestException) as error:
            raise HTTPException(status_code=422, detail=str(error)) from error

    @app.post("/api/tasks/{task_id}/standardize")
    def standardize_task_endpoint(task_id: str) -> dict[str, Any]:
        _validate_task_id(task_id)
        try:
            task = _task_registry(app).resolve(task_id)
            if not isinstance(task.get("task_id"), str):
                raise ValueError("Selected task has no NASA task ID.")
            result = standardize_task(
                task["task_directory"],
                task_name=task["task_name"],
                task_id=task["task_id"],
                metadata_directory=app.state.settings.output.metadata_directory,
            )
        except FileNotFoundError as error:
            raise HTTPException(status_code=404, detail=str(error)) from error
        except (OSError, ValueError, rasterio.errors.RasterioError) as error:
            raise HTTPException(status_code=422, detail=str(error)) from error
        return result

    @app.post("/api/tasks/{task_id}/climate")
    def collect_task_climate(task_id: str) -> dict[str, Any]:
        _validate_task_id(task_id)
        try:
            return collect_power_climate(_task_registry(app), task_id)
        except FileNotFoundError as error:
            raise HTTPException(status_code=404, detail=str(error)) from error
        except requests.RequestException as error:
            raise HTTPException(status_code=502, detail=f"NASA POWER request failed: {error}") from error
        except (OSError, ValueError, rasterio.errors.RasterioError) as error:
            raise HTTPException(status_code=422, detail=str(error)) from error

    @app.post("/api/tasks/{task_id}/soil")
    def collect_task_soil(task_id: str, depth: str = Query(default="0-5cm")) -> dict[str, Any]:
        _validate_task_id(task_id)
        try:
            return collect_soilgrids(_task_registry(app), task_id, depth=depth)
        except FileNotFoundError as error:
            raise HTTPException(status_code=404, detail=str(error)) from error
        except requests.RequestException as error:
            raise HTTPException(status_code=502, detail=f"ISRIC SoilGrids request failed: {error}") from error
        except (OSError, ValueError, rasterio.errors.RasterioError) as error:
            raise HTTPException(status_code=422, detail=str(error)) from error

    @app.get("/api/layers")
    def get_layers(
        task_name: str = Query(min_length=1),
        analysis_name: str | None = Query(default=None),
    ) -> dict[str, Any]:
        map_keys = user_map_raster_keys()
        layers = [
            layer
            for layer in _standardized_layers(app, task_name, analysis_name)
            if layer["product_id"] in {"crop-suitability", "FAO_GAEZ_V5"}
            or (layer["product_id"], layer["variable"]) in map_keys
            or (
                layer["product_id"] == "NASA_POWER_DAILY"
                and re.fullmatch(
                    r"(?:T2M|T2M_MIN|T2M_MAX|PRECTOTCORR)_"
                    r"(?:\d{6}_MONTHLY|\d{4}_(?:DJF|MAM|JJA|SON))",
                    str(layer["variable"]),
                )
                is not None
            )
            or (
                layer["product_id"] == "ISRIC_SOILGRIDS"
                and re.fullmatch(
                    r"(?:phh2o|sand|silt|clay|soc)_"
                    r"(?:0-5cm|5-15cm|15-30cm|30-60cm|60-100cm|100-200cm)_Q0\.5",
                    str(layer["variable"]),
                )
                is not None
            )
        ]
        return {"task_name": task_name, "layers": layers}

    @app.post("/api/suitability/analyze")
    def analyze_suitability(request: SuitabilityAnalysisRequest) -> dict[str, Any]:
        settings: CollectionSettings = app.state.settings
        try:
            task = _task_registry(app).resolve(request.task_name)
            if not isinstance(task.get("task_id"), str):
                raise ValueError("Selected task has no NASA task ID.")
            result = run_spatial_suitability(
                settings,
                task_directory=task["task_directory"],
                task_name=task["task_name"],
                task_id=task["task_id"],
                analysis_name=request.analysis_name,
                crop_ids=request.crop_ids,
                overwrite=request.overwrite,
            )
        except (FileNotFoundError, ValueError, OSError) as error:
            raise HTTPException(status_code=422, detail=str(error)) from error
        return result

    @app.get("/api/suitability/results")
    def suitability_results(
        task_name: str = Query(min_length=1),
        analysis_name: str | None = Query(default=None),
    ) -> dict[str, Any]:
        layers = _suitability_layers_for_task(app, task_name, analysis_name)
        if not layers:
            return {"available": False, "task_name": task_name, "layers": []}
        return {
            "available": True,
            "task_name": task_name,
            "layers": layers,
        }

    @app.get("/api/suitability/cell")
    def inspect_suitability_cell(
        task_name: str = Query(min_length=1),
        lat: float = Query(ge=-90, le=90),
        lon: float = Query(ge=-180, le=180),
        analysis_name: str | None = Query(default=None),
    ) -> dict[str, Any]:
        task, manifest_path, manifest = _suitability_manifest_for_task(
            app, task_name, analysis_name
        )
        best_path = Path(manifest["best_crop_raster"]).resolve()
        run_directory = manifest_path.parent.resolve()
        if not best_path.is_relative_to(run_directory) or not best_path.is_file():
            raise HTTPException(status_code=422, detail="Best-crop raster is outside its task run.")
        with rasterio.open(best_path) as source:
            transformer = Transformer.from_crs("EPSG:4326", source.crs, always_xy=True)
            x_coord, y_coord = transformer.transform(lon, lat)
            row, column = source.index(x_coord, y_coord)
        cell = next(
            (
                item
                for item in manifest.get("cell_results", [])
                if item.get("row") == row and item.get("column") == column
            ),
            None,
        )
        if cell is None:
            return {"task_name": task_name, "no_data": True, "row": row, "column": column}
        return {
            "task_name": task_name,
            "task_id": task.get("task_id"),
            "analysis_name": manifest.get("analysis_name"),
            "no_data": cell.get("best_crop") is None,
            **cell,
        }

    @app.get("/api/layers/{layer_id}")
    def get_layer(layer_id: str, task_name: str = Query(min_length=1)) -> dict[str, Any]:
        layer = _find_raster_layer(app, task_name, layer_id)
        return {key: value for key, value in layer.items() if key != "path"}

    @app.get("/api/layers/{layer_id}/tiles/{z}/{x}/{y}.png")
    def get_layer_tile(
        layer_id: str,
        z: int,
        x: int,
        y: int,
        task_name: str = Query(min_length=1),
    ) -> Response:
        if z < 0 or z > 22 or x < 0 or y < 0 or x >= 2**z or y >= 2**z:
            raise HTTPException(status_code=422, detail="Invalid XYZ tile coordinates.")
        layer = _find_raster_layer(app, task_name, layer_id)
        bounds = _web_mercator_tile_bounds(z, x, y)
        try:
            with rasterio.open(layer["path"]) as source:
                tile_transform = from_bounds(*bounds, 256, 256)
                method = (
                    Resampling.nearest
                    if layer["variable_kind"] in {"quality", "categorical"}
                    else Resampling.bilinear
                )
                with WarpedVRT(
                    source,
                    crs="EPSG:3857",
                    transform=tile_transform,
                    width=256,
                    height=256,
                    resampling=method,
                ) as warped:
                    values = warped.read(1, masked=True)
                    valid = ~np.ma.getmaskarray(values)
                    rgba = _colorize(
                        np.asarray(values.filled(0), dtype=np.float64),
                        valid,
                        layer,
                    )
            buffer = io.BytesIO()
            Image.fromarray(rgba, mode="RGBA").save(buffer, format="PNG")
        except (OSError, ValueError, rasterio.errors.RasterioError) as error:
            raise HTTPException(status_code=500, detail=str(error)) from error
        return Response(
            content=buffer.getvalue(),
            media_type="image/png",
            headers={"Cache-Control": "private, max-age=300"},
        )

    @app.get("/api/layers/{layer_id}/value")
    def get_layer_value(
        layer_id: str,
        task_name: str = Query(min_length=1),
        lat: float = Query(ge=-90, le=90),
        lon: float = Query(ge=-180, le=180),
    ) -> dict[str, Any]:
        layer = _find_raster_layer(app, task_name, layer_id)
        try:
            with rasterio.open(layer["path"]) as source:
                transformer = Transformer.from_crs(
                    "EPSG:4326", source.crs, always_xy=True
                )
                x_coord, y_coord = transformer.transform(lon, lat)
                row, column = source.index(x_coord, y_coord)
                if not (0 <= row < source.height and 0 <= column < source.width):
                    value = None
                else:
                    sample = next(source.sample([(x_coord, y_coord)], masked=True))
                    value = (
                        float(sample[0])
                        if not np.ma.is_masked(sample[0])
                        and math.isfinite(float(sample[0]))
                        else None
                    )
        except (OSError, ValueError, rasterio.errors.RasterioError) as error:
            raise HTTPException(status_code=500, detail=str(error)) from error
        return {
            "layer_id": layer_id,
            "product_id": layer["product_id"],
            "variable": layer["variable"],
            "observation_date": layer["observation_date"],
            "lat": lat,
            "lon": lon,
            "value": value,
            "units": layer["units"],
            "no_data": value is None,
        }

    return app


def _has_verified_suitability_criteria(app: FastAPI) -> bool:
    requirements = load_crop_requirements(
        app.state.project_root / "data/reference/crop_requirements.json"
    )
    return any(
        isinstance(crop.get("environmental_suitability"), dict)
        and bool(crop["environmental_suitability"])
        for crop in requirements["crops"]
    )


def _resolve_config(root: Path, path: str | Path) -> Path:
    candidate = Path(path)
    return candidate.resolve() if candidate.is_absolute() else (root / candidate).resolve()


def _validate_analysis_crs(value: str) -> CRS:
    try:
        crs = CRS.from_user_input(value)
    except CRSError as error:
        raise ValueError(f"Invalid analysis CRS: {value}") from error
    if not crs.is_projected:
        raise ValueError("Analysis CRS must be projected and use meter units.")
    if not crs.axis_info or crs.axis_info[0].unit_name.casefold() not in {
        "metre",
        "meter",
    }:
        raise ValueError("Analysis CRS must use meter units.")
    return crs


def _available_products(app: FastAPI) -> list[dict[str, Any]]:
    settings: CollectionSettings = app.state.settings
    products: dict[str, dict[str, Any]] = {}
    for spec in app.state.dataset_config.datasets:
        metadata_path = settings.output.metadata_directory / spec.metadata_file
        if not metadata_path.is_file():
            continue
        metadata = load_manifest(metadata_path)
        for entry in metadata.get("available_products", []):
            if not isinstance(entry, dict):
                continue
            catalog = entry.get("catalog")
            layers = entry.get("layers")
            if (
                not isinstance(catalog, dict)
                or catalog.get("Available") is not True
                or not isinstance(catalog.get("ProductAndVersion"), str)
                or not isinstance(layers, dict)
            ):
                continue
            product_id = catalog["ProductAndVersion"]
            available_layers = []
            for layer_id, layer_metadata in layers.items():
                if (
                    not isinstance(layer_id, str)
                    or not isinstance(layer_metadata, dict)
                    or layer_metadata.get("Available") is not True
                ):
                    continue
                available_layers.append(
                    {
                        "layer_id": layer_id,
                        "name": _layer_name(layer_id, layer_metadata),
                        "category": _layer_category(layer_id, layer_metadata),
                        "description": layer_metadata.get("Description"),
                        "units": layer_metadata.get("Units") or None,
                        "spatial_resolution": catalog.get("Resolution"),
                        "temporal_resolution": catalog.get("TemporalGranularity"),
                        "available": True,
                        "is_quality": layer_metadata.get("IsQA") is True,
                        "quality_layers": _quality_layers(
                            layer_metadata, product_id=product_id, layer_id=layer_id
                        ),
                        "auxiliary_layers": _auxiliary_layers(
                            layer_metadata, product_id=product_id, layer_id=layer_id
                        ),
                        "valid_min": _metadata_number(layer_metadata.get("ValidMin")),
                        "valid_max": _metadata_number(layer_metadata.get("ValidMax")),
                    }
                )
            if not available_layers:
                continue
            product = products.get(product_id)
            if product is None:
                products[product_id] = {
                    "product_id": product_id,
                    "name": catalog.get("Product") or spec.name,
                    "description": catalog.get("Description"),
                    "spatial_resolution": catalog.get("Resolution"),
                    "temporal_resolution": catalog.get("TemporalGranularity"),
                    "start_date": catalog.get("TemporalExtentStart"),
                    "end_date": catalog.get("TemporalExtentEnd"),
                    "available": True,
                    "metadata_file": spec.metadata_file,
                    "layers": available_layers,
                }
                continue
            known_layers = {layer["layer_id"] for layer in product["layers"]}
            product["layers"].extend(
                layer for layer in available_layers if layer["layer_id"] not in known_layers
            )
    return sorted(products.values(), key=lambda item: (item["name"], item["product_id"]))


def _layer_name(layer_id: str, metadata: dict[str, Any]) -> str:
    description = metadata.get("Description")
    return description.strip() if isinstance(description, str) and description.strip() else layer_id


def _layer_category(layer_id: str, metadata: dict[str, Any]) -> str:
    text = f"{layer_id} {metadata.get('Description', '')}".casefold()
    if metadata.get("IsQA") is True or any(
        term in text for term in ("quality", "qa", " qc", "fmask")
    ):
        return "Quality"
    categories = (
        ("Vegetation", ("ndvi", "evi", "vegetation", "leaf area", "fpar")),
        (
            "Water",
            (
                "soil moisture",
                "moisture",
                "evapotranspiration",
                "water",
                "precipitation",
                "wetness",
                "pet",
            ),
        ),
        ("Temperature", ("temperature", "lst")),
        ("Terrain", ("elevation", "dem", "slope", "aspect")),
        ("Land Cover", ("land cover", "lc_type")),
    )
    for category, terms in categories:
        if any(term in text for term in terms):
            return category
    return "Other"


def _quality_layers(
    metadata: dict[str, Any],
    *,
    product_id: str | None = None,
    layer_id: str | None = None,
) -> list[str]:
    if product_id == "ASTGTM_NC.003" and layer_id == "ASTER_GDEM_DEM":
        return []
    value = metadata.get("QualityLayers", [])
    if isinstance(value, str):
        try:
            value = ast.literal_eval(value)
        except (SyntaxError, ValueError):
            value = []
    return value if isinstance(value, list) and all(isinstance(item, str) for item in value) else []


def _auxiliary_layers(
    metadata: dict[str, Any],
    *,
    product_id: str | None = None,
    layer_id: str | None = None,
) -> list[str]:
    if product_id != "ASTGTM_NC.003" or layer_id != "ASTER_GDEM_DEM":
        return []
    return _quality_layers(metadata)


def _metadata_number(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) else None


def _task_manager(app: FastAPI, client: AppEEARSClient) -> TaskManager:
    return TaskManager(
        client,
        dataset_config=app.state.dataset_config,
        settings=app.state.settings,
    )


def _manager_for_submission(
    app: FastAPI,
    client: AppEEARSClient,
    request: TaskSubmissionRequest,
    normalized_aoi: dict[str, Any],
    analysis_crs: str,
    selection: ProductLayerSelection,
) -> TaskManager:
    settings: CollectionSettings = app.state.settings
    available = next(
        (
            item
            for item in _available_products(app)
            if item["product_id"] == selection.product_id
        ),
        None,
    )
    if available is None:
        raise ValueError(f"Product is not confirmed available: {selection.product_id}.")
    available_layer_ids = {item["layer_id"] for item in available["layers"]}
    unknown_layers = sorted(set(selection.layer_ids) - available_layer_ids)
    if unknown_layers:
        raise ValueError(
            f"Unavailable layer(s) for {selection.product_id}: {', '.join(unknown_layers)}."
        )
    collection_start_date = request.start_date
    collection_end_date = request.end_date
    temporal_resolution = available.get("temporal_resolution")
    if isinstance(temporal_resolution, str) and temporal_resolution.casefold() == "static":
        if isinstance(available.get("start_date"), str):
            collection_start_date = date.fromisoformat(available["start_date"][:10])
        if isinstance(available.get("end_date"), str):
            collection_end_date = date.fromisoformat(available["end_date"][:10])

    extent_start = available.get("start_date")
    is_static = (
        isinstance(temporal_resolution, str)
        and temporal_resolution.casefold() == "static"
    )
    if not is_static and isinstance(extent_start, str):
        first_date = date.fromisoformat(extent_start[:10])
        if collection_start_date < first_date:
            raise ValueError(
                f"{selection.product_id} is not available before {first_date.isoformat()}."
            )
    extent_end = available.get("end_date")
    if (
        not is_static
        and isinstance(extent_end, str)
        and extent_end.casefold() != "present"
    ):
        last_date = date.fromisoformat(extent_end[:10])
        if collection_end_date > last_date:
            raise ValueError(
                f"{selection.product_id} is not available after {last_date.isoformat()}."
            )

    dataset_matches = [
        spec
        for spec in app.state.dataset_config.datasets
        if spec.product_id == selection.product_id
    ]
    if not dataset_matches:
        dataset_matches = [
            spec
            for spec in app.state.dataset_config.datasets
            if spec.metadata_file == available["metadata_file"]
        ]
    if not dataset_matches:
        raise ValueError(
            f"No configured output location exists for {selection.product_id}."
        )
    dataset = dataset_matches[0].model_copy(
        update={
            "product_id": selection.product_id,
            "requested_layers": selection.layer_ids,
            "enabled": True,
        }
    )
    dataset_config = DatasetConfig(datasets=[dataset])
    updated_settings = settings.model_dump()
    updated_settings.update(
        {
            "aoi": AOISettings(
                name=request.aoi_name.strip(),
                crs="EPSG:4326",
                geojson=normalized_aoi,
            ),
            "start_date": collection_start_date,
            "end_date": collection_end_date,
            "spatial_analysis": SpatialAnalysisSettings.model_validate(
                {
                    **settings.spatial_analysis.model_dump(),
                    "analysis_crs": analysis_crs,
                    "cell_size_m": request.resolution_m,
                }
            ),
        }
    )
    configured_settings = CollectionSettings.model_validate(updated_settings)
    return TaskManager(
        client,
        dataset_config=dataset_config,
        settings=configured_settings,
    )


def _local_tasks(app: FastAPI) -> tuple[list[dict[str, Any]], list[str]]:
    tasks = []
    warnings = []
    registry = _task_registry(app)
    for record in registry.records():
        if isinstance(record.get("parent_aoi_task"), dict):
            continue
        task_name = str(record.get("task_name", ""))
        task_directory = registry.root / registry.validate_name(task_name)
        if not task_directory.is_dir():
            warnings.append(
                f"Task {task_name!r} is in the registry, but its task directory is missing."
            )
            continue
        manifest_path = task_directory / "task_manifest.json"
        if not manifest_path.is_file():
            warnings.append(
                f"Task {task_name!r} has a directory but no task manifest."
            )
            continue
        manifest = load_manifest(manifest_path)
        if isinstance(manifest.get("parent_aoi_task"), dict):
            continue
        task = registry.resolve(task_name)
        tasks.append(_task_summary(manifest, task))
    return (
        sorted(tasks, key=lambda item: item.get("submitted_at") or "", reverse=True),
        warnings,
    )


def _linked_provider_task_ids(app: FastAPI) -> set[str]:
    registry = _task_registry(app)
    task_ids: set[str] = set()
    for record in registry.records():
        task_id = record.get("task_id")
        linked = isinstance(record.get("parent_aoi_task"), dict)
        if not linked:
            task_name = record.get("task_name")
            if isinstance(task_name, str):
                manifest_path = registry.root / task_name / "task_manifest.json"
                if manifest_path.is_file():
                    linked = isinstance(
                        load_manifest(manifest_path).get("parent_aoi_task"), dict
                    )
        if linked and isinstance(task_id, str):
            task_ids.add(task_id)
    return task_ids


def _run_aoi_task_acquisition(
    app: FastAPI,
    task_name: str,
    request: AOITaskCreateRequest,
    normalized_aoi: dict[str, Any],
    analysis_crs: str,
    *,
    providers: set[str] | None = None,
) -> None:
    registry = _task_registry(app)
    _, _, task_manifest = registry.load_task_manifest(task_name)
    acquisition = task_manifest["acquisition"]
    acquisition["status"] = "running"
    acquisition["stage"] = "preparing_aoi"
    task_manifest["task_status"] = "processing"
    task_manifest["updated_at"] = datetime.now(timezone.utc).isoformat()
    write_manifest(registry.root / task_name / "task_manifest.json", task_manifest)
    registry.update(task_name, task_status="processing", acquisition_status="running")

    def update_stage(stage: str) -> None:
        _, manifest_path, current = registry.load_task_manifest(task_name)
        current["acquisition"]["stage"] = stage
        current["updated_at"] = datetime.now(timezone.utc).isoformat()
        write_manifest(manifest_path, current)

    def update_provider(provider: str, **updates: Any) -> None:
        _, manifest_path, current = registry.load_task_manifest(task_name)
        provider_state = current["acquisition"]["providers"][provider]
        provider_state.update(updates)
        current["updated_at"] = datetime.now(timezone.utc).isoformat()
        write_manifest(manifest_path, current)

    def should_run(provider: str, requested: bool) -> bool:
        return requested and (providers is None or provider in providers)

    failures: list[str] = []
    successes: list[str] = []
    if should_run("soilgrids", request.soil_properties):
        update_stage("acquiring_soil")
        update_provider("soilgrids", status="running", error=None)
        try:
            result = collect_soilgrids(
                registry,
                str(task_manifest["task_id"]),
                depth=request.soil_depth,
            )
            update_provider(
                "soilgrids",
                status="completed",
                variables=result["new_variables"] or result["already_integrated"],
            )
            successes.append("soilgrids")
        except (
            OSError,
            ValueError,
            RuntimeError,
            requests.RequestException,
            rasterio.errors.RasterioError,
        ) as error:
            update_provider(
                "soilgrids",
                status="failed",
                error=f"{type(error).__name__}: {error}",
            )
            failures.append("soilgrids")

    power_variables = (
        ("T2M", "T2M_MIN", "T2M_MAX")
        if request.nasa_power_temperature
        else ()
    ) + (("PRECTOTCORR",) if request.nasa_power_precipitation else ())
    if should_run("nasa_power", bool(power_variables)):
        update_stage("acquiring_climate")
        update_provider("nasa_power", status="running", error=None)
        try:
            result = collect_power_climate(
                registry,
                str(task_manifest["task_id"]),
                variables=power_variables,
            )
            update_provider(
                "nasa_power",
                status="completed",
                variables=result["new_variables"] or result["already_integrated"],
            )
            successes.append("nasa_power")
        except (
            OSError,
            ValueError,
            RuntimeError,
            requests.RequestException,
            rasterio.errors.RasterioError,
        ) as error:
            update_provider(
                "nasa_power",
                status="failed",
                error=f"{type(error).__name__}: {error}",
            )
            failures.append("nasa_power")

    appeears_successes = 0
    appeears_failures: list[dict[str, str]] = []
    if should_run("appeears", bool(request.products)):
        update_stage("acquiring_satellite")
        update_provider("appeears", status="running", error=None)
        submitted_product_ids: set[str] = set()
        if providers is not None:
            _, _, current = registry.load_task_manifest(task_name)
            for reference in current.get("appeears_tasks", []):
                if not isinstance(reference, dict) or not isinstance(reference.get("task_id"), str):
                    continue
                try:
                    _, _, child_manifest = registry.load_task_manifest(reference["task_id"])
                except (FileNotFoundError, ValueError):
                    continue
                if child_manifest.get("task_status") not in {"failed", "expired"}:
                    product_id = reference.get("product_id")
                    if isinstance(product_id, str):
                        submitted_product_ids.add(product_id)
        for selection in request.products:
            if selection.product_id in submitted_product_ids:
                continue
            try:
                task_request = TaskSubmissionRequest(
                    aoi_name=request.aoi_name,
                    task_name=None,
                    aoi_geojson=normalized_aoi,
                    start_date=request.start_date,
                    end_date=request.end_date,
                    analysis_crs=analysis_crs,
                    resolution_m=request.resolution_m,
                    products=[selection],
                    confirm_real_submission=True,
                )
                manager = _manager_for_submission(
                    app,
                    AppEEARSClient(),
                    task_request,
                    normalized_aoi,
                    analysis_crs,
                    selection,
                )
                submitted = manager.submit(
                    selection.product_id,
                    allow_duplicate=providers is not None,
                    parent_aoi_task={
                        "task_name": task_name,
                        "task_id": task_manifest["task_id"],
                    },
                )
                appeears_successes += 1
                _, child_manifest_path, child_manifest = registry.load_task_manifest(
                    submitted["task_id"]
                )
                child_manifest["updated_at"] = datetime.now(timezone.utc).isoformat()
                write_manifest(child_manifest_path, child_manifest)
                _, manifest_path, current = registry.load_task_manifest(task_name)
                current.setdefault("appeears_tasks", []).append(submitted)
                appeears_state = current["acquisition"]["providers"]["appeears"]
                appeears_state.setdefault("task_ids", []).append(submitted["task_id"])
                appeears_state.setdefault("operations", []).append(
                    {
                        "task_id": submitted["task_id"],
                        "task_name": submitted["task_name"],
                        "product_id": submitted["product_id"],
                        "status": submitted["task_status"],
                    }
                )
                current["updated_at"] = datetime.now(timezone.utc).isoformat()
                write_manifest(manifest_path, current)
            except (
                DuplicateTaskError,
                FileExistsError,
                OSError,
                ValueError,
                requests.RequestException,
            ) as error:
                failure = {
                    "product_id": selection.product_id,
                    "error": f"{type(error).__name__}: {error}",
                }
                appeears_failures.append(failure)
                _, manifest_path, current = registry.load_task_manifest(task_name)
                appeears_state = current["acquisition"]["providers"]["appeears"]
                appeears_state.setdefault("failures", []).append(failure)
                current["updated_at"] = datetime.now(timezone.utc).isoformat()
                write_manifest(manifest_path, current)
        _, _, current = registry.load_task_manifest(task_name)
        existing_task_ids = current["acquisition"]["providers"]["appeears"].get(
            "task_ids", []
        )
        appeears_status = (
            "submitted"
            if existing_task_ids and not appeears_failures
            else "partial"
            if existing_task_ids
            else "failed"
        )
        update_provider(
            "appeears",
            status=appeears_status,
            failures=appeears_failures,
            error=(
                f"{len(appeears_failures)} AppEEARS product request(s) failed."
                if appeears_failures
                else None
            ),
        )
        if appeears_successes:
            successes.append("appeears")
        if appeears_failures:
            failures.append("appeears")

    if should_run("gaez_v5", request.gaez_v5):
        update_stage("acquiring_suitability")
        update_provider("gaez_v5", status="queued", error=None)
        try:
            catalog = load_gaez_catalog(
                app.state.project_root / "data/reference/gaez_v5_crops.json"
            )
            crop_codes = [crop["code"] for crop in catalog["crops"]]
            run_id = make_gaez_run_id(crop_codes, "both", "high")
            _run_gaez_v5_provider(
                app,
                str(task_manifest["task_id"]),
                run_id,
                crop_codes,
                "both",
                "high",
            )
            _, _, current = registry.load_task_manifest(task_name)
            provider_status = current["acquisition"]["providers"]["gaez_v5"]["status"]
            if provider_status in {"completed", "partial"}:
                successes.append("gaez_v5")
            else:
                failures.append("gaez_v5")
        except (OSError, ValueError, RuntimeError) as error:
            update_provider(
                "gaez_v5",
                status="failed",
                error=f"{type(error).__name__}: {error}",
            )
            failures.append("gaez_v5")

    _, manifest_path, final_manifest = registry.load_task_manifest(task_name)
    selected_provider_statuses = [
        state["status"]
        for state in final_manifest["acquisition"]["providers"].values()
        if state["status"] != "not_selected"
    ]
    pending = any(
        state["status"] in {"queued", "running", "submitted"}
        or (
            state["status"] == "partial"
            and bool(state.get("task_ids"))
        )
        for state in final_manifest["acquisition"]["providers"].values()
    )
    has_success = any(
        status in {"completed", "submitted", "partial"}
        for status in selected_provider_statuses
    )
    overall_status = (
        "running"
        if pending
        else "completed"
        if all(status == "completed" for status in selected_provider_statuses)
        else "partial"
        if has_success
        else "failed"
    )
    final_manifest["acquisition"]["status"] = overall_status
    final_manifest["acquisition"]["stage"] = (
        "awaiting_provider_completion"
        if pending
        else "completed"
        if overall_status == "completed"
        else "completed_with_warnings"
        if overall_status == "partial"
        else "failed"
    )
    final_manifest["task_status"] = (
        "processing" if overall_status == "running" else overall_status
    )
    final_manifest["download_status"] = (
        "pending"
        if pending
        else "completed"
        if successes and overall_status == "completed"
        else "partial"
        if successes
        else "failed"
    )
    final_manifest["updated_at"] = datetime.now(timezone.utc).isoformat()
    if failures:
        final_manifest["error"] = (
            "One or more selected providers failed: " + ", ".join(failures)
        )
    elif overall_status == "completed":
        final_manifest["error"] = None
    write_manifest(manifest_path, final_manifest)
    registry.update(
        task_name,
        task_status=final_manifest["task_status"],
        download_status=final_manifest["download_status"],
        acquisition_status=overall_status,
    )


def _run_gaez_v5_provider(
    app: FastAPI,
    task_id: str,
    run_id: str,
    crop_codes: list[str],
    water_regime: Literal["rainfed", "irrigated", "both"],
    management_level: Literal["low", "high"],
) -> None:
    registry = _task_registry(app)

    def update_provider(**updates: Any) -> None:
        _, manifest_path, manifest = registry.load_task_manifest(task_id)
        provider = manifest["acquisition"]["providers"]["gaez_v5"]
        provider.update(updates)
        manifest["acquisition"]["stage"] = updates.get(
            "stage", "acquiring_suitability"
        )
        manifest["updated_at"] = datetime.now(timezone.utc).isoformat()
        write_manifest(manifest_path, manifest)

    update_provider(status="running", stage="acquiring_suitability", error=None)

    def update_operation(state: dict[str, Any]) -> None:
        update_provider(
            status=state["status"],
            run_id=state["run_id"],
            operations=state["operations"],
            stage="acquiring_suitability",
        )

    try:
        result = run_gaez_v5_suitability(
            registry,
            task_id,
            run_id=run_id,
            crop_codes=crop_codes,
            water_regime=water_regime,
            management_level=management_level,
            catalog_path=app.state.project_root / "data/reference/gaez_v5_crops.json",
            on_operation=update_operation,
        )
        final_status = result["status"]
        update_provider(
            status=final_status,
            run_id=run_id,
            operations=[
                operation
                for scenario in result["scenarios"]
                for operation in [
                    {
                        "crop_code": item["crop_code"],
                        "crop_name": item["crop_name"],
                        "water_regime": scenario["water_regime"],
                        "management_level": scenario["management_level"],
                        "status": "completed",
                    }
                    for item in scenario["ranked_crops"] + scenario["unranked_crops"]
                ]
            ]
            + result["failures"],
            error=(
                f"{len(result['failures'])} crop/scenario operation(s) failed."
                if result["failures"]
                else None
            ),
            stage=(
                "completed"
                if final_status == "completed"
                else "completed_with_warnings"
                if final_status == "partial"
                else "failed"
            ),
        )
    except Exception as error:
        _, manifest_path, manifest = registry.load_task_manifest(task_id)
        provider = manifest["acquisition"]["providers"]["gaez_v5"]
        provider.update(
            {
                "status": "failed",
                "run_id": run_id,
                "error": f"{type(error).__name__}: {error}",
                "stage": "failed",
            }
        )
        manifest["updated_at"] = datetime.now(timezone.utc).isoformat()
        write_manifest(manifest_path, manifest)
    _, manifest_path, manifest = registry.load_task_manifest(task_id)
    providers = manifest.get("acquisition", {}).get("providers", {})
    statuses = [
        state.get("status")
        for state in providers.values()
        if isinstance(state, dict) and state.get("status") != "not_selected"
    ]
    pending = any(status in {"queued", "running", "submitted"} for status in statuses)
    successful = any(status in {"completed", "partial", "submitted"} for status in statuses)
    overall_status = (
        "running"
        if pending
        else "completed"
        if statuses and all(status == "completed" for status in statuses)
        else "partial"
        if successful
        else "failed"
    )
    manifest["acquisition"]["status"] = overall_status
    manifest["acquisition"]["stage"] = (
        "awaiting_provider_completion"
        if pending
        else "completed"
        if overall_status == "completed"
        else "completed_with_warnings"
        if overall_status == "partial"
        else "failed"
    )
    manifest["task_status"] = "processing" if pending else overall_status
    manifest["updated_at"] = datetime.now(timezone.utc).isoformat()
    write_manifest(manifest_path, manifest)
    registry.update(
        str(manifest["task_name"]),
        task_status=manifest["task_status"],
        acquisition_status=overall_status,
    )


def _integrate_child_rasters(
    *,
    parent: dict[str, Any],
    child: dict[str, Any],
    parent_manifest: dict[str, Any],
    child_rasters: list[dict[str, Any]],
    registry: TaskRegistry,
) -> int:
    parent_task_name = str(parent["task_name"])
    parent_root = Path(parent["task_directory"]).resolve()
    child_task_name = str(child["task_name"])
    child_task_id = str(child["task_id"])
    child_root = Path(child["task_directory"]).resolve()
    parent_raster_manifest_path = (
        parent_root / "processed" / "standardized_manifest.json"
    )
    _, parent_raster_manifest, existing_records = registry.validate_raster_manifest(
        parent_task_name,
        parent_raster_manifest_path,
    )
    existing_sources = {
        (record.get("source_task_id"), record.get("source_output_path"))
        for record in existing_records
        if isinstance(record.get("source_task_id"), str)
        and isinstance(record.get("source_output_path"), str)
    }
    imported_records: list[dict[str, Any]] = []
    imported_root = parent_root / "processed" / "rasters" / "appeears" / child_task_name

    for record in child_rasters:
        output_value = record.get("output_path")
        if not isinstance(output_value, str) or not output_value:
            raise ValueError(f"Standardized raster in {child_task_name} has no output path.")
        source_raster = Path(output_value)
        if not source_raster.is_absolute():
            source_raster = child_root / source_raster
        source_raster = source_raster.resolve()
        if not source_raster.is_relative_to(child_root) or not source_raster.is_file():
            raise ValueError(
                f"Standardized raster escapes or is missing from {child_task_name}: "
                f"{output_value}"
            )
        source_output_path = source_raster.relative_to(child_root).as_posix()
        source_identity = (child_task_id, source_output_path)
        if source_identity in existing_sources:
            continue
        with rasterio.open(source_raster) as raster:
            if raster.count != 1 or raster.crs is None:
                raise ValueError(f"Standardized raster is invalid: {source_raster}")

        destination = imported_root / source_raster.name
        if destination.exists():
            destination = imported_root / (
                f"{record['product_id']}__{record['layer_id']}__{source_raster.name}"
            )
        if destination.exists():
            raise FileExistsError(f"Refusing to overwrite imported raster: {destination}")
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_raster, destination)
        imported = dict(record)
        imported.pop("source_path", None)
        imported.update(
            {
                "output_path": destination.relative_to(parent_root).as_posix(),
                "source_task_name": child_task_name,
                "source_task_id": child_task_id,
                "source_output_path": source_output_path,
                "source_raw_path": record.get("source_path"),
                "task_name": parent_task_name,
                "task_id": parent_manifest["task_id"],
                "data_scope": "task_scoped",
            }
        )
        imported_records.append(imported)
        existing_sources.add(source_identity)

    all_records = [*existing_records, *imported_records]
    parent_raster_manifest["rasters"] = all_records
    parent_raster_manifest["records"] = all_records
    parent_raster_manifest["raster_count"] = len(all_records)
    parent_raster_manifest["updated_at"] = datetime.now(timezone.utc).isoformat()
    integrations = parent_raster_manifest.setdefault("appeears_integrations", [])
    if isinstance(integrations, list):
        integrations[:] = [
            item
            for item in integrations
            if not isinstance(item, dict) or item.get("source_task_id") != child_task_id
        ]
        integrations.append(
            {
                "source_task_name": child_task_name,
                "source_task_id": child_task_id,
                "raster_count": sum(
                    record.get("source_task_id") == child_task_id
                    for record in all_records
                ),
                "updated_at": parent_raster_manifest["updated_at"],
            }
        )
    write_manifest(parent_raster_manifest_path, parent_raster_manifest)
    registry.validate_raster_manifest(parent_task_name, parent_raster_manifest_path)
    return len(imported_records)


def _merge_remote_tasks(
    local_tasks: list[dict[str, Any]],
    remote_tasks: list[dict[str, Any]],
    *,
    hidden_task_ids: set[str] | None = None,
) -> list[dict[str, Any]]:
    hidden_task_ids = hidden_task_ids or set()
    by_id = {task["task_id"]: task for task in local_tasks}
    for task in remote_tasks:
        task_id = task.get("task_id")
        if not isinstance(task_id, str) or task_id in hidden_task_ids:
            continue
        if task_id in by_id:
            by_id[task_id]["task_status"] = normalize_task_status(
                str(task.get("status", by_id[task_id]["task_status"]))
            )
            by_id[task_id]["remote_status"] = task.get("status")
            continue
        params = task.get("params")
        dates = params.get("dates", [{}]) if isinstance(params, dict) else [{}]
        layers = params.get("layers", []) if isinstance(params, dict) else []
        by_id[task_id] = {
            "task_id": task_id,
            "product_id": next(
                (item.get("product") for item in layers if isinstance(item, dict)),
                None,
            ),
            "aoi_name": None,
            "start_date": dates[0].get("startDate") if dates and isinstance(dates[0], dict) else None,
            "end_date": dates[0].get("endDate") if dates and isinstance(dates[0], dict) else None,
            "submitted_at": task.get("created"),
            "task_status": normalize_task_status(str(task.get("status", "unknown"))),
            "download_status": "unknown",
            "downloaded_file_count": 0,
            "error": None,
            "remote_only": True,
        }
    return sorted(
        by_id.values(),
        key=lambda item: item.get("submitted_at") or "",
        reverse=True,
    )


def _task_summary(manifest: dict[str, Any], record: dict[str, Any]) -> dict[str, Any]:
    return {
        "task_name": manifest.get("task_name"),
        "task_id": manifest.get("task_id"),
        "task_type": manifest.get("task_type"),
        "product_id": manifest.get("product_id"),
        "aoi_name": manifest.get("aoi", {}).get("name"),
        "start_date": manifest.get("start_date"),
        "end_date": manifest.get("end_date"),
        "submitted_at": manifest.get("submitted_at"),
        "task_status": manifest.get("task_status"),
        "download_status": manifest.get("download_status"),
        "downloaded_file_count": sum(
            item.get("status") == "validated"
            for item in manifest.get("downloads", [])
            if isinstance(item, dict)
        ),
        "error": manifest.get("error"),
        "preprocessing_status": manifest.get("preprocessing_status", "pending"),
        "analysis_run_count": len(manifest.get("analysis_runs", [])),
        "acquisition": manifest.get("acquisition"),
        "parent_aoi_task": manifest.get("parent_aoi_task"),
        "task_directory": record.get("task_directory"),
    }


def _validate_task_id(task_id: str) -> None:
    if not task_id or any(character not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-" for character in task_id):
        raise HTTPException(status_code=422, detail="Invalid task ID.")


def _task_registry(app: FastAPI) -> TaskRegistry:
    return TaskRegistry(app.state.tasks_directory)


def _submission_task_name(
    requested_name: str | None,
    product_id: str,
    product_count: int,
) -> str | None:
    if requested_name is None:
        return None
    if product_count == 1:
        return TaskRegistry.validate_name(requested_name.strip())
    suffix = re.sub(r"[^A-Za-z0-9._-]+", "-", product_id).strip("-.")
    return TaskRegistry.validate_name(f"{requested_name.strip()}-{suffix}")


def _standardized_layers(
    app: FastAPI,
    task_name: str,
    analysis_name: str | None = None,
) -> list[dict[str, Any]]:
    return sorted(
        (
            {key: value for key, value in layer.items() if key != "path"}
            for layer in _standardized_layers_with_paths(app, task_name, analysis_name)
        ),
        key=lambda item: (
            item["product_id"],
            item["variable"],
            item["observation_date"] or "",
        ),
    )


def _find_raster_layer(
    app: FastAPI,
    task_name: str,
    layer_id: str,
) -> dict[str, Any]:
    for layer in _standardized_layers_with_paths(app, task_name):
        if layer["id"] == layer_id:
            return layer
    raise HTTPException(status_code=404, detail="Processed raster layer not found.")


def _standardized_layers_with_paths(
    app: FastAPI,
    task_name: str,
    analysis_name: str | None = None,
) -> list[dict[str, Any]]:
    layers = []
    try:
        task, _, _ = _task_registry(app).load_task_manifest(task_name)
        task_root = task["task_directory"]
        manifest_path = task_root / "processed" / "standardized_manifest.json"
        _, _, records = _task_registry(app).validate_raster_manifest(task_name, manifest_path)
    except FileNotFoundError as error:
        raise HTTPException(status_code=404, detail=str(error)) from error
    except ValueError as error:
        raise HTTPException(status_code=422, detail=str(error)) from error

    for record in records:
        product_id = record.get("product_id")
        layer_id = record.get("layer_id")
        path = Path(record["output_path"])
        is_gaez_layer = record.get("external_source") == "FAO/GAEZ-V5"
        observation_date = _observation_date(path.name)
        identity = f"{task_name}:{product_id}:{layer_id}:{observation_date}:{path}"
        layer_key = hashlib.sha256(identity.encode("utf-8")).hexdigest()[:20]
        layers.append(
            {
                "id": layer_key,
                "task_name": task_name,
                "product_id": product_id,
                "variable": layer_id,
                "name": _display_layer_name(layer_id),
                "observation_date": observation_date,
                "temporal_start": record.get("temporal_start"),
                "temporal_end": record.get("temporal_end"),
                "units": record.get("output_units"),
                "variable_kind": record.get("variable_kind", "continuous"),
                "quality_filter_applied": record.get("quality_filter_applied") is True,
                "quality_layers": record.get("quality_layers", []),
                "render_range": (
                    record["render_range"]
                    if isinstance(record.get("render_range"), list)
                    else _render_range(record)
                ),
                "category_colors": record.get("category_colors")
                if is_gaez_layer
                else None,
                "category_labels": record.get("category_labels")
                if is_gaez_layer
                else None,
                "path": path,
            }
        )

    for layer in _suitability_layers_for_task(app, task_name, analysis_name):
        layers.append(layer)
    return layers


def _suitability_layers_for_task(
    app: FastAPI,
    task_name: str,
    analysis_name: str | None,
) -> list[dict[str, Any]]:
    try:
        task, _, task_manifest = _task_registry(app).load_task_manifest(task_name)
    except (FileNotFoundError, ValueError) as error:
        raise HTTPException(status_code=404, detail=str(error)) from error
    runs = task_manifest.get("analysis_runs", [])
    if not isinstance(runs, list):
        raise HTTPException(status_code=422, detail="Task analysis run history is invalid.")
    run = next(
        (
            item
            for item in reversed(runs)
            if isinstance(item, dict)
            and (analysis_name is None or item.get("analysis_name") == analysis_name)
        ),
        None,
    )
    if run is None:
        return []
    try:
        _, manifest_path, manifest = _suitability_manifest_for_task(
            app, task_name, str(run["analysis_name"])
        )
        run_directory = manifest_path.parent.resolve()
        layers = suitability_layers(run_directory)
        color_map = manifest.get("crop_code_map", {})
        colors = _crop_category_colors(color_map)
        output = []
        for layer in layers:
            path = Path(layer["path"]).resolve()
            if not path.is_relative_to(run_directory) or not path.is_file():
                raise ValueError(f"Suitability raster escapes or is missing from its run: {path}")
            with rasterio.open(path) as source:
                if source.count != 1 or source.crs is None:
                    raise ValueError(f"Suitability raster is unreadable: {path}")
            is_gaez_layer = record.get("external_source") == "FAO/GAEZ-V5"
            output.append(
                {
                    "id": hashlib.sha256(
                        f"{task_name}:{run['analysis_name']}:{layer['id']}".encode("utf-8")
                    ).hexdigest()[:20],
                    "task_name": task_name,
                    "analysis_name": run["analysis_name"],
                    "product_id": "crop-suitability",
                    "variable": layer["variable"],
                    "name": layer["name"],
                    "observation_date": None,
                    "units": "index" if layer["variable"] == "best_crop" else (
                        "confidence" if layer["variable"] == "confidence" else "score"
                    ),
                    "variable_kind": layer["variable_kind"],
                    "quality_filter_applied": False,
                    "quality_layers": [],
                    "render_range": layer.get("render_range"),
                    "category_colors": colors if layer["variable"] == "best_crop" else None,
                    "category_labels": {
                        str(code): crop_id
                        for crop_id, code in manifest.get("crop_code_map", {}).items()
                    }
                    if layer["variable"] == "best_crop"
                    else None,
                    "path": path,
                }
            )
        return output
    except (FileNotFoundError, OSError, ValueError, rasterio.errors.RasterioError) as error:
        raise HTTPException(status_code=422, detail=str(error)) from error


def _suitability_manifest_for_task(
    app: FastAPI,
    task_name: str,
    analysis_name: str | None,
) -> tuple[dict[str, Any], Path, dict[str, Any]]:
    try:
        task, _, task_manifest = _task_registry(app).load_task_manifest(task_name)
    except (FileNotFoundError, ValueError) as error:
        raise HTTPException(status_code=404, detail=str(error)) from error
    runs = task_manifest.get("analysis_runs", [])
    run = next(
        (
            item
            for item in reversed(runs)
            if isinstance(item, dict)
            and (analysis_name is None or item.get("analysis_name") == analysis_name)
        ),
        None,
    )
    if run is None:
        raise HTTPException(status_code=404, detail="No suitability run exists for this task.")
    run_directory = (
        task["task_directory"] / "suitability" / "runs" / str(run["analysis_name"])
    ).resolve()
    manifest_path = Path(str(run.get("manifest_path", ""))).resolve()
    if not manifest_path.is_relative_to(run_directory) or not manifest_path.is_file():
        raise HTTPException(
            status_code=422,
            detail="Suitability manifest is missing or outside its task run directory.",
        )
    manifest = load_manifest(manifest_path)
    if (
        manifest.get("task_name") != task_name
        or manifest.get("task_id") != task.get("task_id")
        or manifest.get("analysis_name") != run.get("analysis_name")
    ):
        raise HTTPException(status_code=422, detail="Suitability manifest ownership is invalid.")
    return task, manifest_path, manifest


def _crop_category_colors(crop_code_map: Any) -> dict[str, str]:
    if not isinstance(crop_code_map, dict):
        return {}
    palette = [
        "#e76f51", "#2a9d8f", "#e9c46a", "#457b9d", "#8d5a97",
        "#f4a261", "#6a994e", "#d45087", "#577590", "#90be6d",
        "#f94144", "#43aa8b", "#f8961e", "#277da1", "#9b5de5",
        "#00b4d8", "#bc6c25", "#6d6875", "#a7c957", "#ff6d00",
    ]
    return {
        str(code): palette[(int(code) - 1) % len(palette)]
        for code in crop_code_map.values()
        if isinstance(code, int) and code > 0
    }


def _display_layer_name(layer_id: str) -> str:
    if layer_id == "T2M":
        return "2 m air temperature"
    if layer_id == "PRECTOTCORR":
        return "Corrected precipitation"
    return layer_id.removeprefix("_").replace("_", " ")


def _observation_date(filename: str) -> str | None:
    match = re.search(r"_(\d{8})T\d{6}_", filename)
    if match is None:
        return None
    return date(int(match.group(1)[:4]), int(match.group(1)[4:6]), int(match.group(1)[6:8])).isoformat()


def _render_range(record: dict[str, Any]) -> list[float] | None:
    provided_range = record.get("render_range")
    if (
        isinstance(provided_range, list)
        and len(provided_range) == 2
        and all(isinstance(value, (int, float)) for value in provided_range)
        and provided_range[1] > provided_range[0]
    ):
        return [float(provided_range[0]), float(provided_range[1])]
    layer = str(record.get("layer_id", "")).casefold()
    if "ndvi" in layer or "evi" in layer:
        return [-0.2, 1.0]
    units = str(record.get("output_units") or "").casefold()
    if "lst" in layer or "temperature" in layer:
        return [250.0, 330.0] if units in {"k", "kelvin"} else [-20.0, 60.0]
    if "moisture" in layer or "wetness" in layer:
        return [0.0, 0.6]
    if "dem" in layer or "elevation" in layer:
        return [-500.0, 9000.0]
    minimum = _metadata_number(record.get("physical_valid_range", [None, None])[0]) if record.get("physical_valid_range") else None
    maximum = _metadata_number(record.get("physical_valid_range", [None, None])[1]) if record.get("physical_valid_range") else None
    return [minimum, maximum] if minimum is not None and maximum is not None and maximum > minimum else None


def _web_mercator_tile_bounds(z: int, x: int, y: int) -> tuple[float, float, float, float]:
    origin_shift = 20_037_508.342789244
    tile_size = 2 * origin_shift / (2**z)
    min_x = -origin_shift + x * tile_size
    max_x = min_x + tile_size
    max_y = origin_shift - y * tile_size
    min_y = max_y - tile_size
    return min_x, min_y, max_x, max_y


def _colorize(values: np.ndarray, valid: np.ndarray, layer: dict[str, Any]) -> np.ndarray:
    result = np.zeros((256, 256, 4), dtype=np.uint8)
    if not valid.any():
        return result
    variable = layer["variable"].casefold()
    if layer["variable_kind"] in {"quality", "categorical"}:
        category_colors = layer.get("category_colors")
        if isinstance(category_colors, dict):
            for category, color in category_colors.items():
                try:
                    rgb = tuple(int(color[index : index + 2], 16) for index in (1, 3, 5))
                    mask = valid & (np.rint(values).astype(np.int64) == int(category))
                except (TypeError, ValueError):
                    continue
                result[mask, :3] = rgb
                result[mask, 3] = 220
            return result
        colors = np.array(
            [
                [49, 54, 149],
                [69, 117, 180],
                [116, 173, 209],
                [171, 217, 233],
                [224, 243, 248],
                [254, 224, 144],
                [253, 174, 97],
                [244, 109, 67],
            ],
            dtype=np.uint8,
        )
        indices = np.rint(values).astype(np.int64) % len(colors)
        result[valid, :3] = colors[indices[valid]]
        result[valid, 3] = 220
        return result

    render_range = layer.get("render_range")
    if not render_range:
        return result
    minimum, maximum = render_range
    normalized = np.clip((values[valid] - minimum) / (maximum - minimum), 0, 1)
    if "ndvi" in variable or "evi" in variable:
        stops = np.array(
            [
                [140, 81, 10],
                [246, 232, 195],
                [166, 219, 160],
                [1, 102, 94],
            ],
            dtype=np.float64,
        )
    elif "lst" in variable or "temperature" in variable:
        stops = np.array(
            [[49, 54, 149], [69, 117, 180], [255, 255, 191], [244, 109, 67], [165, 0, 38]],
            dtype=np.float64,
        )
    elif "moisture" in variable or "wetness" in variable:
        stops = np.array([[255, 247, 188], [127, 205, 187], [44, 127, 184], [37, 52, 148]], dtype=np.float64)
    elif "dem" in variable or "elevation" in variable:
        stops = np.array([[38, 97, 63], [145, 153, 89], [190, 151, 105], [245, 239, 218], [255, 255, 255]], dtype=np.float64)
    else:
        stops = np.array([[247, 252, 240], [186, 228, 188], [49, 163, 84], [0, 68, 27]], dtype=np.float64)
    positions = np.linspace(0, 1, len(stops))
    colors = np.stack(
        [np.interp(normalized, positions, stops[:, channel]) for channel in range(3)],
        axis=1,
    ).astype(np.uint8)
    result[valid, :3] = colors
    result[valid, 3] = 220
    return result


app = create_app()
