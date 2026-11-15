from __future__ import annotations

import hashlib
import json
import logging
import math
import re
import uuid
from copy import deepcopy
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Literal

import requests
import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator
from pyproj import CRS
from pyproj.exceptions import CRSError
from shapely.errors import ShapelyError
from shapely.geometry import shape

from .appeears_client import AppEEARSClient
from .manifest import load_manifest, write_manifest
from .product_discovery import DatasetConfig, DatasetSpec, ProductDiscovery
from .task_registry import TaskRegistry

logger = logging.getLogger(__name__)


class AOISettings(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str = Field(min_length=1)
    crs: str = "EPSG:4326"
    geojson: dict[str, Any]


class OutputSettings(BaseModel):
    model_config = ConfigDict(extra="forbid")

    format: Literal["geotiff", "netcdf4"] = "geotiff"
    projection: str = "native"
    directory: Path = Path("data/raw")
    manifests_directory: Path = Path("data/manifests")
    metadata_directory: Path = Path("data/metadata")
    filename_date: Literal["calendar"] | None = "calendar"


class MonitoringSettings(BaseModel):
    model_config = ConfigDict(extra="forbid")

    poll_interval_seconds: int = Field(default=30, ge=5)
    request_timeout_seconds: int = Field(default=60, gt=0)
    max_retries: int = Field(default=4, ge=0)
    max_backoff_seconds: int = Field(default=120, ge=1)
    max_monitor_minutes: int = Field(default=1440, gt=0)


class SpatialAnalysisSettings(BaseModel):
    model_config = ConfigDict(extra="forbid")

    analysis_crs: str = "EPSG:6933"
    cell_size_m: float = Field(default=250.0, gt=0)
    max_grid_cells: int = Field(default=500_000, ge=1)
    output_directory: Path = Path("data/processed/cell_statistics")

    @model_validator(mode="after")
    def validate_analysis_crs(self) -> SpatialAnalysisSettings:
        try:
            crs = CRS.from_user_input(self.analysis_crs)
        except CRSError as error:
            raise ValueError(f"Invalid spatial analysis CRS: {self.analysis_crs}") from error
        if not crs.is_projected:
            raise ValueError("Spatial analysis CRS must be projected with meter units.")
        if not crs.axis_info or crs.axis_info[0].unit_name.casefold() not in {
            "metre",
            "meter",
        }:
            raise ValueError("Spatial analysis CRS must use meter units.")
        return self


class CollectionSettings(BaseModel):
    model_config = ConfigDict(extra="forbid")

    task_name_prefix: str = "nasa-agri"
    aoi: AOISettings
    start_date: date
    end_date: date
    output: OutputSettings = Field(default_factory=OutputSettings)
    monitoring: MonitoringSettings = Field(default_factory=MonitoringSettings)
    spatial_analysis: SpatialAnalysisSettings = Field(
        default_factory=SpatialAnalysisSettings
    )
    additional_options: dict[str, Any] | None = None

    @model_validator(mode="after")
    def validate_date_range(self) -> CollectionSettings:
        if self.end_date < self.start_date:
            raise ValueError("end_date must be on or after start_date.")
        if self.aoi.crs.upper() != "EPSG:4326":
            raise ValueError("AppEEARS area GeoJSON coordinates must use EPSG:4326.")
        return self

    @classmethod
    def load(cls, path: str | Path) -> CollectionSettings:
        with Path(path).open(encoding="utf-8") as file:
            data = yaml.safe_load(file)
        return cls.model_validate(data)


class DuplicateTaskError(ValueError):
    """Raised when an equivalent task already exists locally or remotely."""


def normalize_aoi(geojson: dict[str, Any]) -> dict[str, Any]:
    """Validate WGS84 polygon GeoJSON and preserve its submitted coordinates."""

    if not isinstance(geojson, dict):
        raise ValueError("AOI must be a GeoJSON Geometry, Feature, or FeatureCollection.")
    _validate_geojson_crs(geojson.get("crs"))
    geojson_type = geojson.get("type")
    if geojson_type == "FeatureCollection":
        features = geojson.get("features")
        if not isinstance(features, list) or not features:
            raise ValueError("AOI FeatureCollection must contain at least one feature.")
        normalized_features = [deepcopy(feature) for feature in features]
    elif geojson_type == "Feature":
        normalized_features = [deepcopy(geojson)]
    elif isinstance(geojson_type, str):
        normalized_features = [
            {"type": "Feature", "properties": {}, "geometry": deepcopy(geojson)}
        ]
    else:
        raise ValueError("AOI must be a GeoJSON Geometry, Feature, or FeatureCollection.")

    longitude_bounds: list[tuple[float, float]] = []
    for feature in normalized_features:
        if not isinstance(feature, dict) or feature.get("type") != "Feature":
            raise ValueError("Every AOI FeatureCollection item must be a GeoJSON Feature.")
        geometry_data = feature.get("geometry")
        if not isinstance(geometry_data, dict):
            raise ValueError("Every AOI feature must contain a GeoJSON geometry.")
        properties = feature.get("properties", {})
        if properties is not None and not isinstance(properties, dict):
            raise ValueError("GeoJSON Feature properties must be an object or null.")
        _validate_polygon_coordinates(geometry_data)
        try:
            geometry = shape(geometry_data)
        except (ShapelyError, TypeError, ValueError, KeyError) as error:
            raise ValueError("AOI geometry is malformed GeoJSON.") from error
        if geometry.geom_type not in {"Polygon", "MultiPolygon"}:
            raise ValueError("AppEEARS area requests require polygon or multipolygon AOIs.")
        if geometry.is_empty or not geometry.is_valid or geometry.area <= 0:
            raise ValueError("AOI geometry must be non-empty, valid, and have non-zero area.")
        min_lon, min_lat, max_lon, max_lat = geometry.bounds
        longitude_bounds.append((min_lon, max_lon))
        if (
            not all(math.isfinite(value) for value in (min_lon, min_lat, max_lon, max_lat))
            or min_lon < -180
            or max_lon > 180
            or min_lat < -90
            or max_lat > 90
        ):
            raise ValueError(
                "AOI coordinates are outside valid EPSG:4326 longitude/latitude bounds."
            )

    if max(east for _, east in longitude_bounds) - min(
        west for west, _ in longitude_bounds
    ) > 180:
        raise ValueError("AOIs spanning the antimeridian are not supported.")

    return {"type": "FeatureCollection", "features": normalized_features}


def _validate_geojson_crs(value: Any) -> None:
    if value is None:
        return
    if isinstance(value, str):
        crs_name = value
    elif (
        isinstance(value, dict)
        and value.get("type") == "name"
        and isinstance(value.get("properties"), dict)
        and isinstance(value["properties"].get("name"), str)
    ):
        crs_name = value["properties"]["name"]
    else:
        raise ValueError("AOI GeoJSON CRS must identify EPSG:4326.")
    try:
        crs = CRS.from_user_input(crs_name)
    except CRSError as error:
        raise ValueError(f"Invalid AOI GeoJSON CRS: {crs_name}.") from error
    if crs != CRS.from_epsg(4326):
        raise ValueError("AOI GeoJSON coordinates must use EPSG:4326.")


def _validate_polygon_coordinates(geometry: dict[str, Any]) -> None:
    geometry_type = geometry.get("type")
    coordinates = geometry.get("coordinates")
    if geometry_type == "Polygon":
        polygons = [coordinates]
    elif geometry_type == "MultiPolygon":
        polygons = coordinates
    else:
        raise ValueError("AppEEARS area requests require polygon or multipolygon AOIs.")
    if not isinstance(polygons, list) or not polygons:
        raise ValueError("AOI polygon coordinates must contain at least one linear ring.")

    for polygon in polygons:
        if not isinstance(polygon, list) or not polygon:
            raise ValueError("AOI polygon coordinates must contain at least one linear ring.")
        for ring in polygon:
            if not isinstance(ring, list) or len(ring) < 4:
                raise ValueError("AOI polygon linear rings must contain at least four positions.")
            positions: list[list[int | float]] = []
            for position in ring:
                if (
                    not isinstance(position, (list, tuple))
                    or len(position) < 2
                    or any(
                        isinstance(ordinate, bool)
                        or not isinstance(ordinate, (int, float))
                        or not _is_finite_number(ordinate)
                        for ordinate in position
                    )
                ):
                    raise ValueError("AOI positions must contain finite numeric coordinates.")
                longitude, latitude = position[:2]
                if not -180 <= longitude <= 180 or not -90 <= latitude <= 90:
                    raise ValueError(
                        "AOI coordinates are outside valid EPSG:4326 longitude/latitude bounds."
                    )
                positions.append(list(position))
            if positions[0] != positions[-1]:
                raise ValueError("AOI polygon linear rings must be closed.")
            if any(
                abs(first[0] - second[0]) > 180
                for first, second in zip(positions, positions[1:])
            ):
                raise ValueError(
                    "AOIs crossing the antimeridian are not supported."
                )


def _is_finite_number(value: int | float) -> bool:
    try:
        return math.isfinite(float(value))
    except OverflowError:
        return False


def request_signature(request: dict[str, Any]) -> str:
    """Stable fingerprint for the data-affecting fields in an AppEEARS request."""

    params = request["params"]
    dates = params["dates"][0]
    normalized = {
        "task_type": request["task_type"],
        "product_layers": sorted(
            (layer["product"], layer["layer"])
            for layer in params["layers"]
        ),
        "aoi": normalize_aoi(params["geo"]),
        "start_date": _canonical_date(dates["startDate"]),
        "end_date": _canonical_date(dates["endDate"]),
        "output_format": params["output"]["format"]["type"],
        "projection": params["output"].get("projection"),
        "additional_options": params.get("additionalOptions"),
    }
    encoded = json.dumps(normalized, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _canonical_date(value: str) -> str:
    for date_format in ("%m-%d-%Y", "%Y-%m-%d"):
        try:
            return datetime.strptime(value, date_format).date().isoformat()
        except ValueError:
            continue
    raise ValueError(f"Unrecognized AppEEARS date format: {value!r}")


class TaskManager:
    """Validate, submit, and locally persist AppEEARS area tasks."""

    def __init__(
        self,
        client: AppEEARSClient | None = None,
        *,
        dataset_config: DatasetConfig | None = None,
        settings: CollectionSettings | None = None,
        dataset_config_path: str | Path = "config/datasets.yaml",
        settings_path: str | Path = "config/settings.yaml",
    ) -> None:
        self.client = client or AppEEARSClient()
        self.dataset_config = dataset_config or ProductDiscovery.load_config(dataset_config_path)
        self.settings = settings or CollectionSettings.load(settings_path)
        self.tasks_directory = (
            self.settings.output.manifests_directory.parent / "tasks"
        ).resolve()
        self.registry = TaskRegistry(self.tasks_directory)

    def dataset_for_product(self, product_id: str) -> DatasetSpec:
        matches = [
            dataset
            for dataset in self.dataset_config.datasets
            if dataset.product_id == product_id
        ]
        if len(matches) != 1:
            raise ValueError(
                f"Expected exactly one dataset configuration for {product_id}; found {len(matches)}."
            )
        dataset = matches[0]
        if not dataset.enabled:
            raise ValueError(
                f"Dataset {dataset.name} is disabled in the dataset configuration."
            )
        return dataset

    def build_request(
        self,
        product_id: str,
        *,
        task_name: str | None = None,
        dataset: DatasetSpec | None = None,
    ) -> dict[str, Any]:
        dataset = dataset or self.dataset_for_product(product_id)
        if dataset.product_id != product_id:
            raise ValueError(
                f"Configured dataset {dataset.name} does not select {product_id}."
            )
        if not dataset.requested_layers:
            raise ValueError(f"No requested layers are configured for {product_id}.")

        product_metadata = self._metadata_for_product(dataset, product_id)
        invalid_layers = [
            layer
            for layer in dataset.requested_layers
            if layer not in product_metadata
            or product_metadata[layer].get("Available") is not True
        ]
        if invalid_layers:
            raise ValueError(
                f"Layers not available for {product_id} in saved AppEEARS metadata: "
                f"{', '.join(invalid_layers)}. Refresh product discovery before submission."
            )

        geo = normalize_aoi(self.settings.aoi.geojson)
        name = self.registry.validate_name(
            self._new_task_name(product_id) if task_name is None else task_name.strip()
        )

        output_format: dict[str, Any] = {"type": self.settings.output.format}
        if self.settings.output.format == "geotiff" and self.settings.output.filename_date:
            output_format["filename_date"] = self.settings.output.filename_date
        params: dict[str, Any] = {
            "geo": geo,
            "dates": [
                {
                    "startDate": self.settings.start_date.strftime("%m-%d-%Y"),
                    "endDate": self.settings.end_date.strftime("%m-%d-%Y"),
                    "recurring": False,
                }
            ],
            "layers": [
                {"product": product_id, "layer": layer}
                for layer in dataset.requested_layers
            ],
            "output": {
                "format": output_format,
                "projection": self.settings.output.projection,
            },
        }
        if self.settings.additional_options is not None:
            params["additionalOptions"] = self.settings.additional_options
        return {"task_name": name, "task_type": "area", "params": params}

    def submit(
        self,
        product_id: str,
        *,
        task_name: str | None = None,
        allow_duplicate: bool = False,
        parent_aoi_task: dict[str, str] | None = None,
    ) -> dict[str, Any]:
        if parent_aoi_task is not None:
            parent_name = parent_aoi_task.get("task_name")
            parent_id = parent_aoi_task.get("task_id")
            if (
                not isinstance(parent_name, str)
                or not isinstance(parent_id, str)
                or not re.fullmatch(r"[A-Za-z0-9-]+", parent_id)
            ):
                raise ValueError("Parent AOI task reference is invalid.")
            parent_aoi_task = {
                "task_name": self.registry.validate_name(parent_name),
                "task_id": parent_id,
            }
        dataset = self.dataset_for_product(product_id)
        name = self.registry.validate_name(
            self._new_task_name(product_id) if task_name is None else task_name.strip()
        )
        if (self.tasks_directory / name).exists() or any(
            item.get("task_name") == name for item in self.registry.records()
        ):
            raise FileExistsError(f"Task name already exists: {name}")
        request = self.build_request(product_id, task_name=name, dataset=dataset)
        duplicates = self.find_duplicates(request)
        if duplicates and not allow_duplicate:
            ids = ", ".join(duplicates)
            raise DuplicateTaskError(
                f"Equivalent AppEEARS task already exists ({ids}). "
                "Use --allow-duplicate to explicitly submit another task."
            )

        local_id = str(uuid.uuid4())
        now = self._now()
        task_directory = self.registry.reserve(
            name,
            {
                "task_id": None,
                "created_at": now,
                "updated_at": now,
                "task_status": "submission_started",
                "download_status": "pending",
                "preprocessing_status": "pending",
                "analysis_runs": [],
                "task_directory": str(self.tasks_directory / name),
                **(
                    {"parent_aoi_task": parent_aoi_task}
                    if parent_aoi_task is not None
                    else {}
                ),
            },
        )
        for directory in (
            task_directory / "raw" / dataset.output_group / (dataset.output_name or dataset.name),
            task_directory / "processed" / "rasters",
            task_directory / "suitability" / "runs",
        ):
            directory.mkdir(parents=True, exist_ok=True)
        intent_path = task_directory / "task_manifest.json"
        manifest = self._new_manifest(local_id, name, product_id, dataset, request, task_directory)
        if parent_aoi_task is not None:
            manifest["parent_aoi_task"] = parent_aoi_task
        write_manifest(
            task_directory / "task_metadata.json",
            {
                "task_name": name,
                "task_id": None,
                "created_at": now,
                "aoi": manifest["aoi"],
                "requested_products": [product_id],
                "requested_layers": dataset.requested_layers,
                "start_date": manifest["start_date"],
                "end_date": manifest["end_date"],
                "source_directory": str(task_directory / "raw"),
                "local_task_directory": str(task_directory),
                **(
                    {"parent_aoi_task": parent_aoi_task}
                    if parent_aoi_task is not None
                    else {}
                ),
            },
        )
        write_manifest(intent_path, manifest)
        try:
            response_data, response = self.client.post_json("task", request)
        except (requests.RequestException, ValueError) as error:
            response = error.response if isinstance(error, requests.HTTPError) else None
            known_rejection = response is not None and 400 <= response.status_code < 500
            manifest["submission_status"] = "failed" if known_rejection else "submission_unknown"
            manifest["task_status"] = "failed" if known_rejection else "submission_unknown"
            manifest["error"] = f"{type(error).__name__}: {error}"
            manifest["updated_at"] = self._now()
            write_manifest(intent_path, manifest)
            self.registry.update(
                name,
                task_status=manifest["task_status"],
                submission_status=manifest["submission_status"],
            )
            raise

        task_id = response_data.get("task_id") if isinstance(response_data, dict) else None
        if not task_id:
            location = response.headers.get("Location", "")
            match = re.search(r"/(?:status|task)/([A-Za-z0-9-]+)(?:$|[/?])", location)
            task_id = match.group(1) if match else None
        if not isinstance(task_id, str) or not re.fullmatch(r"[A-Za-z0-9-]+", task_id):
            manifest["submission_status"] = "submission_unknown"
            manifest["task_status"] = "submission_unknown"
            manifest["error"] = "Successful response did not contain a valid AppEEARS task ID."
            manifest["api_response"] = response_data
            manifest["updated_at"] = self._now()
            write_manifest(intent_path, manifest)
            self.registry.update(
                name,
                task_status="submission_unknown",
                submission_status="submission_unknown",
            )
            raise ValueError(manifest["error"])

        submitted_at = self._now()
        status = response_data.get("status", "pending")
        manifest.update(
            {
                "task_id": task_id,
                "task_name": name,
                "submission_status": "submitted",
                "task_status": normalize_task_status(str(status)),
                "submitted_at": submitted_at,
                "updated_at": submitted_at,
                "api_response": response_data,
                "error": None,
            }
        )
        if manifest["task_status"] == "completed":
            manifest["completed_at"] = submitted_at
        write_manifest(intent_path, manifest)
        write_manifest(
            task_directory / "task_metadata.json",
            {
                "task_name": name,
                "task_id": task_id,
                "created_at": manifest["created_at"],
                "submitted_at": submitted_at,
                "aoi": manifest["aoi"],
                "requested_products": [product_id],
                "requested_layers": dataset.requested_layers,
                "start_date": manifest["start_date"],
                "end_date": manifest["end_date"],
                "source_directory": str(task_directory / "raw"),
                "local_task_directory": str(task_directory),
            },
        )
        self.registry.update(
            name,
            task_id=task_id,
            task_status=manifest["task_status"],
            submission_status="submitted",
        )
        logger.info("Submitted AppEEARS task %s for %s.", task_id, product_id)
        return manifest

    def find_duplicates(self, request: dict[str, Any]) -> list[str]:
        signature = request_signature(request)
        duplicates: list[str] = []
        for record in self.registry.records():
            name = self.registry.validate_name(str(record.get("task_name", "")))
            path = self.tasks_directory / name / "task_manifest.json"
            if not path.is_file():
                continue
            manifest = load_manifest(path)
            if (
                manifest.get("request_signature") == signature
                and manifest.get("submission_status") != "failed"
            ):
                duplicates.append(str(manifest.get("task_id") or path.stem))

        for task in self.remote_tasks():
            if self._is_remote_duplicate(task, request, signature):
                task_id = task.get("task_id")
                if isinstance(task_id, str) and task_id not in duplicates:
                    duplicates.append(task_id)
        return duplicates

    def _is_remote_duplicate(
        self,
        task: dict[str, Any],
        request: dict[str, Any],
        signature: str,
    ) -> bool:
        if task.get("task_type") != request["task_type"]:
            return False
        params = task.get("params")
        if not isinstance(params, dict):
            return False

        try:
            if sorted((item["product"], item["layer"]) for item in params["layers"]) != sorted(
                (item["product"], item["layer"])
                for item in request["params"]["layers"]
            ):
                return False
            remote_dates = params["dates"][0]
            request_dates = request["params"]["dates"][0]
            if (
                _canonical_date(remote_dates["startDate"])
                != _canonical_date(request_dates["startDate"])
                or _canonical_date(remote_dates["endDate"])
                != _canonical_date(request_dates["endDate"])
            ):
                return False
            if (
                params["output"]["format"]["type"]
                != request["params"]["output"]["format"]["type"]
                or params["output"].get("projection")
                != request["params"]["output"].get("projection")
            ):
                return False
        except (IndexError, KeyError, TypeError, ValueError):
            return False

        if "geo" not in params:
            task_id = task.get("task_id")
            if not isinstance(task_id, str):
                return False
            details = self.client.get_json(f"task/{task_id}")
            if not isinstance(details, dict) or not isinstance(details.get("params"), dict):
                return False
            params = details["params"]
        remote_request = {
            "task_type": task["task_type"],
            "params": params,
        }
        try:
            return request_signature(remote_request) == signature
        except (IndexError, KeyError, TypeError, ValueError):
            return False

    def remote_tasks(self, *, page_size: int = 1000) -> list[dict[str, Any]]:
        if page_size < 1:
            raise ValueError("page_size must be greater than zero.")
        tasks: list[dict[str, Any]] = []
        offset = 0
        while True:
            page = self.client.get_json(
                "task",
                params={"limit": page_size, "offset": offset},
            )
            if not isinstance(page, list) or any(not isinstance(item, dict) for item in page):
                raise ValueError("AppEEARS task list endpoint returned an unexpected response.")
            tasks.extend(page)
            if len(page) < page_size:
                return tasks
            offset += len(page)

    def local_tasks(self) -> list[dict[str, Any]]:
        tasks = []
        for record in self.registry.records():
            name = self.registry.validate_name(str(record.get("task_name", "")))
            path = self.tasks_directory / name / "task_manifest.json"
            if path.is_file():
                tasks.append(load_manifest(path))
        return tasks

    def get_manifest(self, task_id: str) -> tuple[Path, dict[str, Any]]:
        self._validate_task_id(task_id)
        task = self.registry.resolve(task_id)
        if not task.get("task_id"):
            raise FileNotFoundError(f"NASA task ID has not been assigned to {task['task_name']}.")
        path = task["task_directory"] / "task_manifest.json"
        manifest = load_manifest(path)
        if manifest.get("task_id") != task_id or manifest.get("task_name") != task["task_name"]:
            raise ValueError("Task manifest identity does not match the local registry.")
        return path, manifest

    def save_manifest(self, path: Path, manifest: dict[str, Any]) -> None:
        task_name = manifest.get("task_name")
        if not isinstance(task_name, str):
            raise ValueError("Task manifest must include its task name.")
        task = self.registry.resolve(task_name)
        if not path.resolve().is_relative_to(task["task_directory"]):
            raise ValueError("Task manifest destination is outside its task directory.")
        manifest["updated_at"] = self._now()
        write_manifest(path, manifest)
        self.registry.update(
            task_name,
            task_id=manifest.get("task_id"),
            task_status=manifest.get("task_status"),
            submission_status=manifest.get("submission_status"),
            download_status=manifest.get("download_status"),
            preprocessing_status=manifest.get("preprocessing_status", "pending"),
        )

    def _metadata_for_product(
        self,
        dataset: DatasetSpec,
        product_id: str,
    ) -> dict[str, dict[str, Any]]:
        metadata_path = self.settings.output.metadata_directory / dataset.metadata_file
        if not metadata_path.exists():
            raise FileNotFoundError(
                f"Saved metadata is missing: {metadata_path}. Run product discovery first."
            )
        metadata = load_manifest(metadata_path)
        for product in metadata.get("available_products", []):
            if (
                isinstance(product, dict)
                and product.get("catalog", {}).get("ProductAndVersion") == product_id
                and product.get("catalog", {}).get("Available") is True
                and isinstance(product.get("layers"), dict)
            ):
                return product["layers"]
        raise ValueError(
            f"{product_id} is not recorded as available in {metadata_path}. "
            "Run product discovery and verify the configured version."
        )

    def _new_manifest(
        self,
        local_id: str,
        task_name: str,
        product_id: str,
        dataset: DatasetSpec,
        request: dict[str, Any],
        task_directory: Path,
    ) -> dict[str, Any]:
        return {
            "local_submission_id": local_id,
            "task_name": task_name,
            "task_id": None,
            "created_at": self._now(),
            "source_directory": str(task_directory / "raw"),
            "local_task_directory": str(task_directory),
            "product_id": product_id,
            "dataset": dataset.name,
            "output_directory": str(
                task_directory
                / "raw"
                / dataset.output_group
                / (dataset.output_name or dataset.name)
            ),
            "layers": dataset.requested_layers,
            "aoi": {
                "name": self.settings.aoi.name,
                "crs": self.settings.aoi.crs,
                "geojson": request["params"]["geo"],
            },
            "start_date": self.settings.start_date.isoformat(),
            "end_date": self.settings.end_date.isoformat(),
            "spatial_analysis": {
                "analysis_crs": self.settings.spatial_analysis.analysis_crs,
                "requested_cell_size_m": self.settings.spatial_analysis.cell_size_m,
            },
            "submitted_at": None,
            "completed_at": None,
            "task_status": "submission_started",
            "submission_status": "submitting",
            "download_status": "pending",
            "preprocessing_status": "pending",
            "analysis_runs": [],
            "downloads": [],
            "task_request": request,
            "request_signature": request_signature(request),
            "created_at": self._now(),
            "updated_at": self._now(),
            "error": None,
        }

    def _new_task_name(self, product_id: str) -> str:
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        clean_product = re.sub(r"[^A-Za-z0-9-]+", "-", product_id)
        return f"{self.settings.task_name_prefix}-{clean_product}-{timestamp}-{uuid.uuid4().hex[:8]}"

    @staticmethod
    def _validate_task_id(task_id: str) -> None:
        if not re.fullmatch(r"[A-Za-z0-9-]+", task_id):
            raise ValueError("Invalid AppEEARS task ID.")

    @staticmethod
    def _now() -> str:
        return datetime.now(timezone.utc).isoformat()


def normalize_task_status(api_status: str) -> str:
    status = api_status.casefold()
    statuses = {
        "pending": "pending",
        "processing": "processing",
        "done": "completed",
        "completed": "completed",
        "error": "failed",
        "failed": "failed",
        "expired": "expired",
    }
    if status not in statuses:
        return "unknown"
    return statuses[status]
