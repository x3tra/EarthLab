from __future__ import annotations

import errno
import hashlib
import json
import logging
import re
import shutil
import time
import xml.etree.ElementTree as ET
import zipfile
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any
from urllib.parse import quote

import rasterio
import requests
from pydantic import BaseModel, ConfigDict, Field

from .manifest import write_manifest
from .task_manager import TaskManager
from .task_monitor import TaskMonitor

logger = logging.getLogger(__name__)


class BundleFile(BaseModel):
    model_config = ConfigDict(extra="allow")

    file_id: str = Field(min_length=1)
    file_name: str = Field(min_length=1)
    file_size: int | None = Field(default=None, ge=0)
    sha256: str | None = None
    file_type: str | None = None


class DownloadManager:
    """Discover and stream AppEEARS bundle files into dataset folders."""

    def __init__(self, manager: TaskManager) -> None:
        self.manager = manager

    def download_task(
        self,
        task_id: str,
        *,
        overwrite: bool = False,
    ) -> dict[str, Any]:
        path, manifest = self.manager.get_manifest(task_id)
        task = self.manager.registry.resolve(manifest["task_name"])
        task_directory = task["task_directory"]
        if manifest.get("task_status") != "completed":
            manifest = TaskMonitor(self.manager).check_once(task_id)
            path, manifest = self.manager.get_manifest(task_id)
        if manifest.get("task_status") != "completed":
            raise ValueError(
                f"Task {task_id} is {manifest.get('task_status')!r}; "
                "downloads are available only after AppEEARS reports completion."
            )

        bundle = self.manager.client.get_json(f"bundle/{task_id}")
        if not isinstance(bundle, dict) or not isinstance(bundle.get("files"), list):
            raise ValueError(f"AppEEARS returned an invalid bundle listing for {task_id}.")
        listing_path = path.with_name(f"{task_id}.outputs.json")
        write_manifest(
            listing_path,
            {
                **bundle,
                "task_name": manifest["task_name"],
                "task_id": task_id,
                "created_at": self.manager._now(),
                "source_directory": manifest["output_directory"],
            },
        )
        manifest["output_listing_path"] = str(listing_path)
        manifest["bundle"] = {
            key: value for key, value in bundle.items() if key != "files"
        }
        manifest["downloads"] = manifest.get("downloads", [])
        manifest["download_status"] = "downloading"
        manifest["error"] = None
        self.manager.save_manifest(path, manifest)

        files = [BundleFile.model_validate(item) for item in bundle["files"]]
        if not files:
            manifest["download_status"] = "failed"
            manifest["error"] = (
                f"Completed AppEEARS task {task_id} contains no output files."
            )
            self.manager.save_manifest(path, manifest)
            raise ValueError(manifest["error"])
        output_format = (
            manifest.get("task_request", {})
            .get("params", {})
            .get("output", {})
            .get("format", {})
            .get("type")
        )
        expected_suffixes = {
            "geotiff": {".tif", ".tiff"},
            "netcdf4": {".nc", ".nc4"},
        }.get(output_format)
        if expected_suffixes and not any(
            Path(item.file_name).suffix.casefold() in expected_suffixes for item in files
        ):
            manifest["download_status"] = "failed"
            manifest["error"] = (
                f"Completed AppEEARS task {task_id} has no {output_format} data output."
            )
            self.manager.save_manifest(path, manifest)
            raise ValueError(manifest["error"])

        destination_dir = Path(manifest["output_directory"]).resolve()
        if not destination_dir.is_relative_to(task_directory / "raw"):
            raise ValueError("Task download destination is outside its raw data directory.")
        destination_dir.mkdir(parents=True, exist_ok=True)
        existing_by_id = {
            item.get("file_id"): item
            for item in manifest["downloads"]
            if item.get("file_id")
        }
        for bundle_file in files:
            destination = self._destination_path(destination_dir, bundle_file.file_name)
            record = existing_by_id.get(bundle_file.file_id)
            if record is None:
                record = {
                    "file_id": bundle_file.file_id,
                    "filename": bundle_file.file_name,
                    "file_type": bundle_file.file_type,
                    "expected_size_bytes": bundle_file.file_size,
                    "checksum": bundle_file.sha256,
                    "status": "pending",
                    "local_path": str(destination_dir / bundle_file.file_name),
                }
                manifest["downloads"].append(record)
                existing_by_id[bundle_file.file_id] = record

            if destination.exists() and not overwrite:
                try:
                    metadata = self._validate_file(
                        destination,
                        expected_size=bundle_file.file_size,
                        checksum=bundle_file.sha256,
                        original_name=bundle_file.file_name,
                        content_type=record.get("content_type"),
                    )
                except (OSError, ValueError) as error:
                    logger.warning(
                        "Existing download %s is invalid; downloading it again: %s",
                        bundle_file.file_name,
                        error,
                    )
                else:
                    self._mark_validated(record, destination, metadata)
                    self.manager.save_manifest(path, manifest)
                    continue

            record["status"] = "downloading"
            record["error"] = None
            self.manager.save_manifest(path, manifest)
            try:
                metadata = self._download_with_retries(
                    task_id,
                    bundle_file,
                    destination,
                )
            except (OSError, requests.RequestException, ValueError) as error:
                record["status"] = "failed"
                record["error"] = f"{type(error).__name__}: {error}"
                manifest["download_status"] = "failed"
                self.manager.save_manifest(path, manifest)
                if isinstance(error, OSError) and error.errno == errno.ENOSPC:
                    raise OSError(
                        errno.ENOSPC,
                        f"Insufficient disk space while downloading {bundle_file.file_name}.",
                    ) from error
                raise

            self._mark_validated(record, destination, metadata)
            self.manager.save_manifest(path, manifest)

        manifest["download_status"] = (
            "completed"
            if all(item.get("status") == "validated" for item in manifest["downloads"])
            else "failed"
        )
        self.manager.save_manifest(path, manifest)
        return manifest

    def validate_task_downloads(self, task_id: str) -> list[dict[str, Any]]:
        path, manifest = self.manager.get_manifest(task_id)
        task = self.manager.registry.resolve(manifest["task_name"])
        results: list[dict[str, Any]] = []
        for record in manifest.get("downloads", []):
            file_path = Path(record["local_path"]).resolve()
            if not file_path.is_relative_to(task["task_directory"] / "raw"):
                raise ValueError(
                    f"Task download path escapes its raw data directory: {file_path}"
                )
            if not file_path.is_file():
                record["status"] = "missing"
                results.append(
                    {"filename": record["filename"], "valid": False, "reason": "File is missing."}
                )
                continue
            try:
                details = self._validate_file(
                    file_path,
                    expected_size=record.get("expected_size_bytes"),
                    checksum=record.get("checksum"),
                    original_name=record["filename"],
                    content_type=record.get("content_type"),
                )
            except (OSError, ValueError) as error:
                record["status"] = "invalid"
                record["error"] = f"{type(error).__name__}: {error}"
                results.append(
                    {"filename": record["filename"], "valid": False, "reason": str(error)}
                )
            else:
                self._mark_validated(record, file_path, details)
                results.append({"filename": record["filename"], "valid": True, **details})
        manifest["download_status"] = (
            "completed"
            if results and all(result["valid"] for result in results)
            else "failed"
        )
        self.manager.save_manifest(path, manifest)
        return results

    def _download_with_retries(
        self,
        task_id: str,
        bundle_file: BundleFile,
        destination: Path,
    ) -> dict[str, Any]:
        monitoring = self.manager.settings.monitoring
        last_error: BaseException | None = None
        for attempt in range(monitoring.max_retries + 1):
            try:
                return self._download_once(
                    task_id,
                    bundle_file,
                    destination,
                )
            except requests.HTTPError as error:
                response = error.response
                retryable = response is None or (
                    response.status_code == 429 or response.status_code >= 500
                )
                retry_after = response.headers.get("Retry-After") if response else None
                last_error = error
            except (requests.Timeout, requests.ConnectionError, OSError) as error:
                retryable = True
                retry_after = None
                last_error = error
                if isinstance(error, OSError) and error.errno == errno.ENOSPC:
                    raise
            except ValueError as error:
                error_message = str(error).casefold()
                retryable = any(
                    marker in error_message
                    for marker in ("checksum", "size", "empty")
                )
                retry_after = None
                last_error = error

            if not retryable or attempt >= monitoring.max_retries:
                raise last_error
            if any(
                marker in str(last_error).casefold()
                for marker in ("checksum", "size mismatch", "empty")
            ):
                destination.with_name(f"{destination.name}.part").unlink(missing_ok=True)
            delay = self._retry_delay(attempt, retry_after, monitoring.max_backoff_seconds)
            logger.warning(
                "Download of %s failed temporarily; retrying in %s seconds (%d/%d).",
                bundle_file.file_name,
                delay,
                attempt + 1,
                monitoring.max_retries,
            )
            time.sleep(delay)

        raise RuntimeError("Unreachable download retry state.")

    def _download_once(
        self,
        task_id: str,
        bundle_file: BundleFile,
        destination: Path,
    ) -> dict[str, Any]:
        destination.parent.mkdir(parents=True, exist_ok=True)
        part_path = destination.with_name(f"{destination.name}.part")
        partial_size = part_path.stat().st_size if part_path.exists() else 0
        if bundle_file.file_size is not None and partial_size > bundle_file.file_size:
            part_path.unlink()
            partial_size = 0
        self._check_disk_space(destination.parent, bundle_file.file_size, partial_size)

        endpoint = (
            f"bundle/{quote(task_id, safe='-')}/{quote(bundle_file.file_id, safe='-')}"
        )
        headers = {"Range": f"bytes={partial_size}-"} if partial_size else None
        try:
            response = self.manager.client.get_stream(endpoint, headers=headers)
        except requests.HTTPError as error:
            if (
                partial_size == 0
                or error.response is None
                or error.response.status_code != 416
            ):
                raise
            part_path.unlink(missing_ok=True)
            self._check_disk_space(destination.parent, bundle_file.file_size, 0)
            response = self.manager.client.get_stream(endpoint)
        with response:
            range_info = self._range_info(response.headers.get("Content-Range"))
            if response.status_code == 206:
                if partial_size == 0 or range_info is None or range_info[0] != partial_size:
                    raise ValueError(
                        f"AppEEARS returned an unexpected partial range for {bundle_file.file_name}."
                    )
                if (
                    bundle_file.file_size is not None
                    and range_info[2] is not None
                    and range_info[2] != bundle_file.file_size
                ):
                    raise ValueError(
                        f"AppEEARS returned an unexpected total size for {bundle_file.file_name}."
                    )
                mode = "ab"
            elif response.status_code == 200:
                mode = "wb"
            else:
                response.raise_for_status()
                raise ValueError(
                    f"Unexpected HTTP {response.status_code} while downloading {bundle_file.file_name}."
                )

            content_type = response.headers.get("Content-Type", "").split(";", 1)[0].casefold()
            expected_binary = Path(bundle_file.file_name).suffix.casefold() in {
                ".tif", ".tiff", ".nc", ".nc4", ".zip"
            }
            if expected_binary and content_type in {"text/html", "application/xhtml+xml"}:
                raise ValueError(
                    f"AppEEARS returned {content_type} instead of {bundle_file.file_name}."
                )
            self._validate_content_type(bundle_file.file_name, content_type)
            try:
                with part_path.open(mode) as file:
                    for chunk in response.iter_content(chunk_size=1024 * 1024):
                        if chunk:
                            file.write(chunk)
            except OSError as error:
                if error.errno == errno.ENOSPC:
                    raise OSError(
                        errno.ENOSPC,
                        f"Insufficient disk space while writing {part_path}.",
                    ) from error
                raise

        details = self._validate_file(
            part_path,
            expected_size=bundle_file.file_size,
            checksum=bundle_file.sha256,
            original_name=bundle_file.file_name,
        )
        details["content_type"] = content_type
        part_path.replace(destination)
        return details

    @staticmethod
    def _check_disk_space(directory: Path, expected_size: int | None, partial_size: int) -> None:
        if expected_size is None:
            return
        free_bytes = shutil.disk_usage(directory).free
        remaining_bytes = max(0, expected_size - partial_size)
        if free_bytes < remaining_bytes:
            raise OSError(
                errno.ENOSPC,
                f"Need {remaining_bytes} bytes to complete download; only {free_bytes} bytes are free.",
            )

    @staticmethod
    def _safe_filename(filename: str) -> PurePosixPath:
        path = PurePosixPath(filename)
        if (
            not filename
            or "\\" in filename
            or "\x00" in filename
            or path.is_absolute()
            or PureWindowsPath(filename).drive
            or any(part in {"", ".", ".."} for part in filename.split("/"))
        ):
            raise ValueError(f"Unsafe AppEEARS bundle filename: {filename!r}")
        return path

    @classmethod
    def _destination_path(cls, root: Path, filename: str) -> Path:
        relative_path = cls._safe_filename(filename)
        resolved_root = root.resolve()
        destination = root.joinpath(*relative_path.parts)
        if not destination.resolve().is_relative_to(resolved_root):
            raise ValueError(f"Unsafe AppEEARS bundle filename: {filename!r}")
        destination.parent.mkdir(parents=True, exist_ok=True)
        if not destination.resolve().is_relative_to(resolved_root):
            raise ValueError(f"Unsafe AppEEARS bundle filename: {filename!r}")
        return destination

    @staticmethod
    def _range_info(content_range: str | None) -> tuple[int, int, int | None] | None:
        if content_range is None:
            return None
        match = re.fullmatch(r"bytes (\d+)-(\d+)/(\d+|\*)", content_range.strip())
        if not match:
            return None
        start, end = int(match.group(1)), int(match.group(2))
        total = None if match.group(3) == "*" else int(match.group(3))
        if end < start or (total is not None and end >= total):
            return None
        return start, end, total

    @staticmethod
    def _retry_delay(attempt: int, retry_after: str | None, maximum: int) -> int:
        if retry_after is not None:
            try:
                return min(maximum, max(1, int(float(retry_after))))
            except ValueError:
                pass
        return min(maximum, 2**attempt)

    @classmethod
    def _validate_file(
        cls,
        path: Path,
        *,
        expected_size: int | None,
        checksum: str | None,
        original_name: str,
        content_type: str | None = None,
    ) -> dict[str, Any]:
        if not path.is_file():
            raise ValueError(f"Downloaded file is missing: {path}")
        size = path.stat().st_size
        if size == 0:
            raise ValueError(f"Downloaded file is empty: {original_name}")
        if expected_size is not None and size != expected_size:
            raise ValueError(
                f"Downloaded size mismatch for {original_name}: "
                f"expected {expected_size}, received {size}."
            )
        digest = cls._sha256(path) if checksum else None
        if checksum and digest.casefold() != checksum.casefold():
            raise ValueError(f"SHA-256 checksum mismatch for {original_name}.")

        suffix = Path(original_name).suffix.casefold()
        details: dict[str, Any] = {"size_bytes": size, "sha256": digest}
        if content_type:
            cls._validate_content_type(original_name, content_type.casefold())
            details["content_type"] = content_type
        if suffix in {".tif", ".tiff"}:
            try:
                with rasterio.open(path) as dataset:
                    if dataset.width <= 0 or dataset.height <= 0 or dataset.count <= 0:
                        raise ValueError(
                            f"Invalid GeoTIFF dimensions or band count: {original_name}"
                        )
                    details["raster"] = {
                        "width": dataset.width,
                        "height": dataset.height,
                        "band_count": dataset.count,
                        "crs": dataset.crs.to_string() if dataset.crs else None,
                        "transform": list(dataset.transform),
                        "resolution": list(dataset.res),
                        "nodata": dataset.nodata,
                        "dtypes": list(dataset.dtypes),
                    }
            except rasterio.errors.RasterioError as error:
                raise ValueError(f"Invalid GeoTIFF file {original_name}: {error}") from error
        elif suffix == ".json":
            with path.open(encoding="utf-8") as file:
                json.load(file)
            details["format_validated"] = "json"
        elif suffix == ".xml":
            try:
                ET.parse(path)
            except ET.ParseError as error:
                details["format_warning"] = (
                    f"AppEEARS metadata XML is not well-formed: {error}"
                )
            else:
                details["format_validated"] = "xml"
        elif suffix == ".zip":
            try:
                with zipfile.ZipFile(path) as archive:
                    bad_member = archive.testzip()
            except zipfile.BadZipFile as error:
                raise ValueError(f"Invalid ZIP file {original_name}: {error}") from error
            if bad_member:
                raise ValueError(f"Corrupt ZIP member {bad_member!r} in {original_name}.")
            details["format_validated"] = "zip"
        elif suffix in {".nc", ".nc4"}:
            with path.open("rb") as file:
                signature = file.read(8)
            if not (
                signature.startswith(b"CDF\x01")
                or signature.startswith(b"CDF\x02")
                or signature.startswith(b"CDF\x05")
                or signature == b"\x89HDF\r\n\x1a\n"
            ):
                raise ValueError(f"Invalid NetCDF signature in {original_name}.")
            details["format_validated"] = "netcdf"
        else:
            details["format_validated"] = "non_empty"
        return details

    @staticmethod
    def _validate_content_type(filename: str, content_type: str) -> None:
        if not content_type or content_type in {
            "application/octet-stream",
            "binary/octet-stream",
        }:
            return
        suffix = Path(filename).suffix.casefold()
        accepted = {
            ".tif": {"image/tiff", "image/geotiff", "application/geotiff"},
            ".tiff": {"image/tiff", "image/geotiff", "application/geotiff"},
            ".nc": {"application/x-netcdf", "application/netcdf"},
            ".nc4": {"application/x-netcdf", "application/netcdf"},
            ".json": {"application/json", "text/json", "application/geo+json"},
            ".xml": {"application/xml", "text/xml"},
            ".zip": {"application/zip", "application/x-zip-compressed"},
            ".csv": {"text/csv", "application/csv", "text/plain"},
            ".txt": {"text/plain", "text/csv"},
        }.get(suffix)
        if accepted is not None and content_type not in accepted:
            raise ValueError(
                f"Unexpected content type {content_type!r} for {filename!r}."
            )

    @staticmethod
    def _sha256(path: Path) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as file:
            for chunk in iter(lambda: file.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()

    @staticmethod
    def _mark_validated(
        record: dict[str, Any],
        path: Path,
        details: dict[str, Any],
    ) -> None:
        warning = details.get("format_warning")
        record.update(
            {
                "status": "validated",
                "size_bytes": details["size_bytes"],
                "local_path": str(path),
                "validation": details,
                "content_type": details.get("content_type"),
                "error": None,
            }
        )
        if warning:
            record["validation_warning"] = warning
            logger.warning("Downloaded file %s has a validation warning: %s", path, warning)
        else:
            record.pop("validation_warning", None)
