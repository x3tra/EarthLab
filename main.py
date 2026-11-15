from __future__ import annotations

import argparse
import json
import logging
import sys
from typing import Any

import requests

from nasa_agri_intelligence.collection.downloader import DownloadManager
from nasa_agri_intelligence.collection.task_manager import (
    CollectionSettings,
    TaskManager,
)
from nasa_agri_intelligence.collection.task_monitor import TaskMonitor
from nasa_agri_intelligence.collection.task_registry import TaskRegistry
from nasa_agri_intelligence.cell_statistics import run_cell_statistics
from nasa_agri_intelligence.suitability.spatial import run_spatial_suitability


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="NASA AppEEARS agricultural data collection.")
    parser.add_argument("--datasets-config", default="config/datasets.yaml")
    parser.add_argument("--settings", default="config/settings.yaml")
    parser.add_argument("--log-level", default="INFO")
    commands = parser.add_subparsers(dest="command", required=True)

    commands.add_parser("validate", help="Validate enabled product and layer configurations.")

    task_commands = commands.add_parser("tasks", help="Manage AppEEARS extraction tasks.")
    task_subcommands = task_commands.add_subparsers(dest="task_command", required=True)
    list_tasks = task_subcommands.add_parser("list", help="List locally registered tasks.")
    list_tasks.add_argument("--remote", action="store_true", help="List NASA tasks from AppEEARS.")
    inspect_task = task_subcommands.add_parser("inspect", help="Inspect a registered task by name or NASA ID.")
    inspect_task.add_argument("identifier")

    submit = task_subcommands.add_parser("submit", help="Submit one configured area extraction.")
    submit.add_argument("--product", required=True, help="Configured AppEEARS product/version ID.")
    submit.add_argument("--task-name", help="Optional task name; generated if omitted.")
    submit.add_argument(
        "--allow-duplicate",
        action="store_true",
        help="Explicitly allow submission when an equivalent task already exists.",
    )

    status = task_subcommands.add_parser("status", help="Refresh and display one task's status.")
    status.add_argument("task_id")

    monitor = task_subcommands.add_parser("monitor", help="Poll one task until terminal or timeout.")
    monitor.add_argument("task_id")
    monitor.add_argument("--poll-interval", type=int)
    monitor.add_argument("--max-wait-minutes", type=int)

    task_subcommands.add_parser("monitor-all", help="Resume monitoring unfinished local tasks.")

    download = commands.add_parser("download", help="Download a completed task bundle.")
    download.add_argument("task_id")
    download.add_argument("--overwrite", action="store_true")

    validate_downloads = commands.add_parser(
        "validate-downloads",
        help="Revalidate downloaded files recorded in a task manifest.",
    )
    validate_downloads.add_argument("task_id")

    spatial = commands.add_parser(
        "spatial",
        help="Create configured AOI grid cells and per-raster statistics.",
    )
    spatial_commands = spatial.add_subparsers(dest="spatial_command", required=True)
    stats = spatial_commands.add_parser(
        "stats",
        help="Summarize standardized native rasters by configurable grid cell.",
    )
    stats_task = stats.add_mutually_exclusive_group(required=True)
    stats_task.add_argument("--task-name")
    stats_task.add_argument("--task-id")
    stats.add_argument("--overwrite", action="store_true")

    suitability = commands.add_parser(
        "suitability",
        help="Evaluate crop suitability for the configured AOI using existing standardized rasters.",
    )
    suitability_commands = suitability.add_subparsers(dest="suitability_command", required=True)
    analyze = suitability_commands.add_parser(
        "analyze",
        help="Generate crop suitability rasters and metadata for the configured AOI.",
    )
    analyze_task = analyze.add_mutually_exclusive_group(required=True)
    analyze_task.add_argument("--task-name")
    analyze_task.add_argument("--task-id")
    analyze.add_argument("--analysis-name")
    analyze.add_argument("--crop", action="append", dest="crop_ids")
    analyze.add_argument("--overwrite", action="store_true")
    return parser


def _print_json(value: Any) -> None:
    print(json.dumps(value, indent=2, ensure_ascii=False, default=str))


def run(args: argparse.Namespace) -> int:
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    if args.command == "spatial":
        settings = CollectionSettings.load(args.settings)
        task = TaskRegistry(settings.output.manifests_directory.parent / "tasks").resolve(
            args.task_name or args.task_id
        )
        if not isinstance(task.get("task_id"), str):
            raise ValueError("Selected task has no NASA task ID.")
        result = run_cell_statistics(
            settings,
            task_directory=task["task_directory"],
            task_name=task["task_name"],
            task_id=task["task_id"],
            overwrite=args.overwrite,
        )
        _print_json(
            {
                "manifest_path": result["manifest_path"],
                "task_name": result["task_name"],
                "task_id": result["task_id"],
                "aoi_name": result["aoi_name"],
                "analysis_crs": result["analysis_crs"],
                "requested_cell_size_m": result["requested_cell_size_m"],
                "raster_count": result["raster_count"],
                "cell_summary_count": sum(
                    item["cell_count"] for item in result["files"]
                ),
            }
        )
        return 0

    if args.command == "suitability":
        settings = CollectionSettings.load(args.settings)
        task = TaskRegistry(settings.output.manifests_directory.parent / "tasks").resolve(
            args.task_name or args.task_id
        )
        if not isinstance(task.get("task_id"), str):
            raise ValueError("Selected task has no NASA task ID.")
        result = run_spatial_suitability(
            settings,
            task_directory=task["task_directory"],
            task_name=task["task_name"],
            task_id=task["task_id"],
            analysis_name=args.analysis_name,
            crop_ids=args.crop_ids,
            overwrite=args.overwrite,
        )
        _print_json({
            "task_name": result["task_name"],
            "task_id": result["task_id"],
            "input_manifest": result["input_manifest"],
            "input_raster_count": result["input_raster_count"],
            "missing_required_variables": result["missing_variables"],
            "analysis_run_id": result["analysis_name"],
            "manifest_path": result["output_directory"] + "/suitability_manifest.json",
            "output_directory": result["output_directory"],
            "aoi_name": result["aoi_name"],
            "analysis_crs": result["analysis_crs"],
            "requested_cell_size_m": result["requested_cell_size_m"],
            "cells_with_data": result["cells_with_data"],
            "crops_evaluated": len(result["crops_evaluated"]),
        })
        return 0

    manager = TaskManager(
        dataset_config_path=args.datasets_config,
        settings_path=args.settings,
    )

    if args.command == "validate":
        results = []
        for dataset in manager.dataset_config.datasets:
            if not dataset.enabled:
                continue
            if dataset.product_id is None:
                raise ValueError(f"Enabled dataset {dataset.name} has no product_id.")
            request = manager.build_request(dataset.product_id, dataset=dataset)
            results.append(
                {
                    "dataset": dataset.name,
                    "product_id": dataset.product_id,
                    "layers": [item["layer"] for item in request["params"]["layers"]],
                    "status": "valid",
                }
            )
        _print_json(results)
        return 0

    if args.command == "tasks":
        task_manager = TaskMonitor(manager)
        if args.task_command == "list":
            _print_json(
                manager.remote_tasks()
                if args.remote
                else manager.registry.records()
            )
        elif args.task_command == "inspect":
            task, path, manifest = manager.registry.load_task_manifest(args.identifier)
            _print_json({"registry": task, "manifest_path": path, "task": manifest})
        elif args.task_command == "submit":
            manifest = manager.submit(
                args.product,
                task_name=args.task_name,
                allow_duplicate=args.allow_duplicate,
            )
            _print_json(manifest)
        elif args.task_command == "status":
            _print_json(task_manager.check_once(args.task_id))
        elif args.task_command == "monitor":
            _print_json(
                task_manager.monitor(
                    args.task_id,
                    poll_interval_seconds=args.poll_interval,
                    max_wait_minutes=args.max_wait_minutes,
                )
            )
        elif args.task_command == "monitor-all":
            _print_json(task_manager.monitor_all())
        return 0

    if args.command == "download":
        _print_json(DownloadManager(manager).download_task(args.task_id, overwrite=args.overwrite))
        return 0

    if args.command == "validate-downloads":
        results = DownloadManager(manager).validate_task_downloads(args.task_id)
        _print_json(results)
        return 0 if all(item["valid"] for item in results) else 1

    raise ValueError(f"Unsupported command: {args.command}")


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    try:
        return run(args)
    except (OSError, ValueError, requests.RequestException) as error:
        logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
        logging.getLogger(__name__).error("%s", error)
        return 1


if __name__ == "__main__":
    sys.exit(main())
