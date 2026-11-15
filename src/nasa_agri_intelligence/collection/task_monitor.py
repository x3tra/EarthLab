from __future__ import annotations

import logging
import time
from datetime import datetime, timezone
from typing import Any

import requests

from .task_manager import TaskManager, normalize_task_status

logger = logging.getLogger(__name__)

TERMINAL_STATUSES = {"completed", "failed", "expired"}


class TaskMonitor:
    """Poll AppEEARS task status and persist status transitions."""

    def __init__(self, manager: TaskManager) -> None:
        self.manager = manager

    def check_once(self, task_id: str) -> dict[str, Any]:
        path, manifest = self.manager.get_manifest(task_id)
        status_data = self._fetch_status(task_id)
        api_status = status_data.get("status")
        if not isinstance(api_status, str):
            raise ValueError(f"AppEEARS returned no task status for {task_id}.")

        normalized_status = normalize_task_status(api_status)
        previous_status = manifest.get("task_status")
        manifest.update(
            {
                "task_status": normalized_status,
                "api_status": api_status,
                "last_status_check": self._now(),
                "status_details": status_data,
            }
        )
        if normalized_status == "completed" and not manifest.get("completed_at"):
            manifest["completed_at"] = (
                status_data.get("completed") or status_data.get("updated") or self._now()
            )
        if normalized_status in {"failed", "expired"}:
            summary = status_data.get("summary")
            summary_error = (
                summary.get("description")
                if isinstance(summary, dict)
                else None
            )
            manifest["error"] = (
                status_data.get("error")
                or summary_error
                or f"AppEEARS task entered terminal status {api_status!r}."
            )
        self.manager.save_manifest(path, manifest)

        if normalized_status != previous_status:
            logger.info(
                "AppEEARS task %s status changed: %s -> %s",
                task_id,
                previous_status,
                normalized_status,
            )
        progress = status_data.get("progress")
        if isinstance(progress, dict) and "summary" in progress:
            logger.info("AppEEARS task %s progress: %s%%", task_id, progress["summary"])
        return manifest

    def check_remote_once(self, task_id: str) -> dict[str, Any]:
        """Read a remote task's current status when no local manifest exists."""

        status_data = self._fetch_status(task_id)
        api_status = status_data.get("status")
        if not isinstance(api_status, str):
            raise ValueError(f"AppEEARS returned no task status for {task_id}.")
        return {
            "task_id": task_id,
            "task_status": normalize_task_status(api_status),
            "api_status": api_status,
            "status_details": status_data,
            "remote_only": True,
        }

    def monitor(
        self,
        task_id: str,
        *,
        poll_interval_seconds: int | None = None,
        max_wait_minutes: int | None = None,
    ) -> dict[str, Any]:
        monitoring = self.manager.settings.monitoring
        interval = (
            monitoring.poll_interval_seconds
            if poll_interval_seconds is None
            else poll_interval_seconds
        )
        max_wait = (
            monitoring.max_monitor_minutes
            if max_wait_minutes is None
            else max_wait_minutes
        )
        if interval < 1 or max_wait < 1:
            raise ValueError("Polling interval and monitor timeout must be positive.")
        deadline = time.monotonic() + max_wait * 60

        while True:
            manifest = self._check_with_retries(task_id)
            if manifest.get("task_status") in TERMINAL_STATUSES:
                return manifest
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                logger.info("Stopped monitoring %s after the configured timeout.", task_id)
                return manifest
            time.sleep(min(interval, remaining))

    def monitor_all(self) -> list[dict[str, Any]]:
        unfinished = [
            manifest
            for manifest in self.manager.local_tasks()
            if manifest.get("task_id")
            and manifest.get("task_status") not in TERMINAL_STATUSES
            and manifest.get("submission_status") == "submitted"
        ]
        return [self.monitor(manifest["task_id"]) for manifest in unfinished]

    def _check_with_retries(self, task_id: str) -> dict[str, Any]:
        monitoring = self.manager.settings.monitoring
        for attempt in range(monitoring.max_retries + 1):
            last_error: requests.RequestException | None = None
            try:
                return self.check_once(task_id)
            except (requests.Timeout, requests.ConnectionError) as error:
                last_error = error
                retry_after = None
            except requests.HTTPError as error:
                response = error.response
                if response is None or (
                    response.status_code != 429 and response.status_code < 500
                ):
                    raise
                last_error = error
                retry_after = response.headers.get("Retry-After")
            if attempt >= monitoring.max_retries:
                if last_error is None:
                    raise RuntimeError("Retry loop ended without a request error.")
                raise last_error
            try:
                server_delay = float(retry_after) if retry_after is not None else 0
            except ValueError:
                server_delay = 0
            backoff = min(
                monitoring.max_backoff_seconds,
                2**attempt,
                max(0, server_delay),
            )
            if server_delay <= 0:
                backoff = min(monitoring.max_backoff_seconds, 2**attempt)
            logger.warning(
                "Temporary AppEEARS status error for %s; retrying in %s seconds (%d/%d).",
                task_id,
                backoff,
                attempt + 1,
                monitoring.max_retries,
            )
            time.sleep(backoff)
        raise RuntimeError("Unreachable retry state.")

    def _fetch_status(self, task_id: str) -> dict[str, Any]:
        self.manager._validate_task_id(task_id)
        payload = self.manager.client.get_json(f"status/{task_id}")
        if isinstance(payload, list):
            candidates = [
                item for item in payload
                if isinstance(item, dict) and item.get("task_id") == task_id
            ]
            if not candidates:
                raise ValueError(f"AppEEARS returned no status record for task {task_id}.")
            status_data = candidates[0]
        elif isinstance(payload, dict):
            status_data = payload
        else:
            raise ValueError(f"AppEEARS returned an invalid status response for {task_id}.")

        if isinstance(status_data.get("status"), str):
            return status_data
        if isinstance(status_data.get("progress"), dict):
            return {**status_data, "status": "processing"}

        task_details = self.manager.client.get_json(f"task/{task_id}")
        if not isinstance(task_details, dict) or not isinstance(task_details.get("status"), str):
            raise ValueError(f"AppEEARS returned no recognized task state for {task_id}.")
        return task_details

    @staticmethod
    def _now() -> str:
        return datetime.now(timezone.utc).isoformat()
