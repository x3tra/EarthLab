from __future__ import annotations

import ast
import fcntl
import json
import re
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import rasterio

from .manifest import load_manifest, write_manifest

_SAFE_TASK_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9 ._-]{0,119}\Z")


class TaskRegistry:
    """Persistent mapping between human-readable names, NASA IDs, and task roots."""

    def __init__(self, tasks_directory: str | Path) -> None:
        self.root = Path(tasks_directory).resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.path = self.root / "task_registry.json"

    @staticmethod
    def validate_name(task_name: str) -> str:
        if (
            not _SAFE_TASK_NAME.fullmatch(task_name)
            or task_name in {".", ".."}
            or task_name.endswith((" ", "."))
        ):
            raise ValueError(
                "Task names must be 1-120 characters, start with a letter or digit, "
                "and contain only letters, digits, spaces, dots, underscores, or hyphens."
            )
        return task_name

    def reserve(self, task_name: str, record: dict[str, Any]) -> Path:
        task_name = self.validate_name(task_name)
        task_directory = (self.root / task_name).resolve()
        if not task_directory.is_relative_to(self.root):
            raise ValueError("Task name resolves outside the task directory.")
        with self._locked():
            records = self.records()
            if task_directory.exists() or any(
                item.get("task_name") == task_name for item in records
            ):
                raise FileExistsError(f"Task name already exists: {task_name}")
            task_directory.mkdir(parents=True, exist_ok=False)
            (task_directory / "raw").mkdir()
            (task_directory / "processed" / "rasters").mkdir(parents=True)
            (task_directory / "suitability" / "runs").mkdir(parents=True)
            self._write_records([*records, {**record, "task_name": task_name}])
        return task_directory

    def records(self) -> list[dict[str, Any]]:
        if not self.path.is_file():
            return []
        payload = load_manifest(self.path)
        records = payload.get("tasks")
        if not isinstance(records, list) or any(not isinstance(item, dict) for item in records):
            raise ValueError(f"Invalid task registry: {self.path}")
        names = [item.get("task_name") for item in records]
        ids = [item.get("task_id") for item in records if item.get("task_id") is not None]
        if (
            any(
                not isinstance(name, str)
                or not _SAFE_TASK_NAME.fullmatch(name)
                or name.endswith((" ", "."))
                for name in names
            )
            or len(names) != len(set(names))
            or len(ids) != len(set(ids))
        ):
            raise ValueError(f"Task registry contains duplicate or invalid task identities: {self.path}")
        return records

    def update(self, task_name: str, **updates: Any) -> dict[str, Any]:
        task_name = self.validate_name(task_name)
        with self._locked():
            records = self.records()
            for item in records:
                if item.get("task_name") == task_name:
                    item.update(updates)
                    item["updated_at"] = _now()
                    self._write_records(records)
                    return item
        raise FileNotFoundError(f"Task is not registered: {task_name}")

    def _is_requested_quality_layer(
        self,
        product_id: Any,
        layer_id: Any,
        requested_layers: list[Any],
        requested_product_id: Any,
    ) -> bool:
        if (
            not isinstance(product_id, str)
            or not isinstance(layer_id, str)
            or not isinstance(requested_product_id, str)
            or not requested_layers
            or not all(isinstance(layer, str) for layer in requested_layers)
        ):
            return False

        metadata_directory = self.root.parent / "metadata"
        for metadata_path in sorted(metadata_directory.glob("*.json")):
            if metadata_path.name in {"availability_report.json", "dataset_inventory.json"}:
                continue
            try:
                with metadata_path.open(encoding="utf-8") as file:
                    metadata = json.load(file)
            except (OSError, json.JSONDecodeError):
                continue
            products = metadata.get("available_products", [])
            if not isinstance(products, list):
                continue
            for product in products:
                if not isinstance(product, dict):
                    continue
                catalog = product.get("catalog")
                layers = product.get("layers")
                if (
                    not isinstance(catalog, dict)
                    or catalog.get("ProductAndVersion") != requested_product_id
                    or not isinstance(layers, dict)
                ):
                    continue
                for requested_layer in requested_layers:
                    requested_metadata = layers.get(requested_layer)
                    if not isinstance(requested_metadata, dict):
                        continue
                    quality_layers = requested_metadata.get("QualityLayers", [])
                    if isinstance(quality_layers, str):
                        try:
                            quality_layers = ast.literal_eval(quality_layers) if quality_layers else []
                        except (SyntaxError, ValueError):
                            continue
                    quality_product = requested_metadata.get(
                        "QualityProductAndVersion", requested_product_id
                    )
                    if (
                        isinstance(quality_layers, list)
                        and layer_id in quality_layers
                        and quality_product == product_id
                    ):
                        return True
        return False

    def resolve(self, identifier: str) -> dict[str, Any]:
        matches = [
            item
            for item in self.records()
            if item.get("task_name") == identifier or item.get("task_id") == identifier
        ]
        if len(matches) != 1:
            raise FileNotFoundError(f"No unique registered task matches {identifier!r}.")
        record = matches[0]
        task_name = self.validate_name(str(record.get("task_name", "")))
        task_directory = (self.root / task_name).resolve()
        if not task_directory.is_relative_to(self.root) or not task_directory.is_dir():
            raise FileNotFoundError(f"Task directory is missing for {task_name}.")
        return {**record, "task_directory": task_directory}

    def load_task_manifest(self, identifier: str) -> tuple[dict[str, Any], Path, dict[str, Any]]:
        task = self.resolve(identifier)
        manifest_path = task["task_directory"] / "task_manifest.json"
        if not manifest_path.is_file():
            raise FileNotFoundError(f"Task manifest is missing: {manifest_path}")
        manifest = load_manifest(manifest_path)
        if manifest.get("task_name") != task["task_name"]:
            raise ValueError("Task manifest name does not match the selected registry task.")
        if manifest.get("task_id") != task.get("task_id"):
            raise ValueError("Task manifest NASA ID does not match the task registry.")
        metadata_path = task["task_directory"] / "task_metadata.json"
        if metadata_path.is_file():
            metadata = load_manifest(metadata_path)
            if (
                metadata.get("task_name") != task["task_name"]
                or metadata.get("task_id") != task.get("task_id")
            ):
                raise ValueError("Task metadata identity does not match the task registry.")
        return task, manifest_path, manifest

    def validate_raster_manifest(
        self,
        identifier: str,
        manifest_path: str | Path,
    ) -> tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]]]:
        task, _, task_manifest = self.load_task_manifest(identifier)
        task_directory = task["task_directory"]
        path = Path(manifest_path).resolve()
        if not path.is_relative_to(task_directory):
            raise ValueError("Raster manifest is outside the selected task directory.")
        if not path.is_file():
            raise FileNotFoundError(f"Task raster manifest is missing: {path}")
        manifest = load_manifest(path)
        if manifest.get("task_name") != task["task_name"]:
            raise ValueError("Raster manifest task name does not match the selected task.")
        task_id = task.get("task_id")
        if not task_id or manifest.get("task_id") != task_id:
            raise ValueError("Raster manifest NASA task ID does not match the registry.")
        if not isinstance(manifest.get("created_at"), str) or not manifest["created_at"]:
            raise ValueError(f"Task raster manifest is missing its creation timestamp: {path}")
        source_directory = manifest.get("source_directory")
        if not isinstance(source_directory, str) or not source_directory:
            raise ValueError(f"Task raster manifest is missing its source directory: {path}")
        source_root = _resolve_task_path(task_directory, source_directory)
        if source_root != (task_directory / "raw").resolve():
            raise ValueError("Raster manifest source directory is not this task's raw directory.")
        if not source_root.is_dir():
            raise FileNotFoundError(f"Task raw data directory is missing: {source_root}")
        expected_product = task_manifest.get("product_id")
        expected_layers = task_manifest.get("layers", [])
        if not isinstance(expected_layers, list):
            raise ValueError("Selected task requested-layer metadata is invalid.")
        rasters = manifest.get("rasters", manifest.get("records"))
        if not isinstance(rasters, list) or (
            not rasters and task_manifest.get("task_type") != "aoi_analysis"
        ):
            raise ValueError(f"Task raster manifest contains no raster entries: {path}")

        seen_paths: set[Path] = set()
        checked: list[dict[str, Any]] = []
        for item in rasters:
            if not isinstance(item, dict):
                raise ValueError(f"Invalid raster entry in task manifest: {path}")
            product_id = item.get("product_id")
            layer_id = item.get("layer_id")
            raster_value = item.get("output_path")
            if not all(isinstance(value, str) and value for value in (product_id, layer_id, raster_value)):
                raise ValueError(f"Raster entry lacks product, layer, or output path: {path}")
            is_derived = item.get("derived") is True
            external_source = item.get("external_source")
            is_external = external_source in {
                "NASA/POWER",
                "ISRIC/SoilGrids",
                "FAO/GAEZ-V5",
            }
            source_product = item.get("source_product_id") if is_derived else product_id
            source_layer = item.get("source_layer_id") if is_derived else layer_id
            if is_derived and (
                product_id != "_DERIVED"
                or layer_id != "slope"
                or not isinstance(source_product, str)
                or not isinstance(source_layer, str)
                or not any(
                    term in source_layer.casefold()
                    for term in ("dem", "elevation")
                )
            ):
                raise ValueError(f"Derived raster has invalid source identity: {path}")
            if is_external:
                if external_source == "NASA/POWER":
                    valid_identity = (
                        product_id == "NASA_POWER_DAILY"
                        and (
                            layer_id in {"T2M", "T2M_MIN", "T2M_MAX", "PRECTOTCORR"}
                            or re.fullmatch(
                                r"(?:T2M|T2M_MIN|T2M_MAX|PRECTOTCORR)_"
                                r"(?:\d{6}_MONTHLY|\d{4}_(?:DJF|MAM|JJA|SON))",
                                str(layer_id),
                            )
                            is not None
                        )
                    )
                elif external_source == "ISRIC/SoilGrids":
                    valid_identity = (
                        product_id == "ISRIC_SOILGRIDS"
                        and re.fullmatch(
                            r"(?:phh2o|sand|silt|clay|soc)_"
                            r"(?:0-5cm|5-15cm|15-30cm|30-60cm|60-100cm|100-200cm)_Q0\.5",
                            str(layer_id),
                        )
                        is not None
                    )
                else:
                    valid_identity = (
                        product_id == "FAO_GAEZ_V5"
                        and isinstance(layer_id, str)
                        and re.fullmatch(
                            r"(?:SXX|SIX)_[A-Z0-9]+_(?:rainfed|irrigated)_(?:low|high)",
                            layer_id,
                        )
                        is not None
                        and item.get("source_product_id")
                        in {"RES05-SXX30AS", "RES05-SIX30AS"}
                    )
                if (
                    not valid_identity
                    or item.get("data_scope") != "task_scoped"
                    or item.get("task_name") != task["task_name"]
                    or item.get("task_id") != task_id
                ):
                    raise ValueError(
                        f"External raster is not owned by the selected task: {path}"
                    )
            is_requested_quality = (
                source_layer not in expected_layers
                and self._is_requested_quality_layer(
                    source_product,
                    source_layer,
                    expected_layers,
                    expected_product,
                )
            )
            if (
                expected_product
                and source_product != expected_product
                and not is_external
                and not is_requested_quality
            ):
                raise ValueError(
                    f"Raster source product {source_product} does not belong to selected task "
                    f"{task['task_name']} ({expected_product})."
                )
            if (
                expected_layers
                and source_layer not in expected_layers
                and not is_external
                and not is_requested_quality
            ):
                raise ValueError(
                    f"Raster source layer {source_layer} was not requested by task "
                    f"{task['task_name']}."
                )
            raster_path = _resolve_task_path(task_directory, raster_value)
            if raster_path in seen_paths:
                raise ValueError(f"Duplicate raster entry in task manifest: {raster_path}")
            seen_paths.add(raster_path)
            source_value = item.get("source_path")
            if source_value is not None:
                if not isinstance(source_value, str):
                    raise ValueError(f"Invalid source path in raster entry: {path}")
                source_path = _resolve_task_path(task_directory, source_value)
                if not source_path.is_relative_to((task_directory / "raw").resolve()):
                    raise ValueError(f"Source raster is outside task raw data: {source_path}")
                if not source_path.is_file():
                    raise FileNotFoundError(f"Task source raster is missing: {source_path}")
            if not raster_path.is_file():
                raise FileNotFoundError(f"Task raster is missing: {raster_path}")
            try:
                with rasterio.open(raster_path) as source:
                    if source.count != 1 or source.crs is None:
                        raise ValueError(f"Task raster is unreadable or has no CRS: {raster_path}")
                    tags = source.tags()
                    source_product = tags.get("source_product_id")
                    source_layer = tags.get("source_layer_id")
                    expected_tag_product = item.get("source_product_id") if is_derived else product_id
                    expected_tag_layer = item.get("source_layer_id") if is_derived else layer_id
                    if (
                        source_product != expected_tag_product
                        or source_layer != expected_tag_layer
                    ):
                        raise ValueError(
                            f"Raster identity does not match manifest entry "
                            f"{expected_tag_product}/{expected_tag_layer}: {raster_path}"
                        )
                    if is_external:
                        invalid_external_tags = (
                            tags.get("external_source") != external_source
                            or tags.get("task_name") != task["task_name"]
                            or tags.get("task_id") != task_id
                        )
                        if external_source == "NASA/POWER":
                            invalid_external_tags = invalid_external_tags or (
                                tags.get("temporal_start") != item.get("temporal_start")
                                or tags.get("temporal_end") != item.get("temporal_end")
                            )
                        if invalid_external_tags:
                            raise ValueError(
                                f"External raster ownership tags are invalid: {raster_path}"
                            )
            except rasterio.errors.RasterioError as error:
                raise ValueError(f"Task raster cannot be read: {raster_path}") from error
            item_copy = dict(item)
            item_copy["output_path"] = str(raster_path)
            checked.append(item_copy)
        if task_manifest.get("task_name") != task["task_name"]:
            raise ValueError("Task metadata is inconsistent with the task registry.")
        return task, manifest, checked

    def _write_records(self, records: list[dict[str, Any]]) -> None:
        write_manifest(
            self.path,
            {"updated_at": _now(), "tasks": records},
        )

    @contextmanager
    def _locked(self):
        lock_path = self.root / ".task_registry.lock"
        with lock_path.open("a", encoding="utf-8") as lock_file:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)


def _resolve_task_path(task_directory: Path, value: str) -> Path:
    candidate = Path(value)
    if not candidate.is_absolute():
        candidate = task_directory / candidate
    resolved = candidate.resolve()
    if not resolved.is_relative_to(task_directory.resolve()):
        raise ValueError(f"Raster path escapes the selected task directory: {value}")
    return resolved


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()
