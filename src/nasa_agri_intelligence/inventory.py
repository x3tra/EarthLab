from __future__ import annotations

import csv
import hashlib
import json
import logging
import re
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import rasterio

logger = logging.getLogger(__name__)

_RASTER_SUFFIXES = {".tif", ".tiff"}
_LAYER_DATE = re.compile(r"__(.+?)_(\d{8})T\d{6}_")


def build_inventory(data_dir: str | Path = "data") -> dict[str, Any]:
    """Inspect local AppEEARS metadata, manifests, and downloaded files."""

    data_path = Path(data_dir).resolve()
    project_root = data_path.parent
    metadata_dir = data_path / "metadata"
    raw_dir = data_path / "raw"
    manifest_dir = data_path / "manifests"

    metadata_products, availability = _load_product_metadata(metadata_dir)
    manifests = _load_task_manifests(manifest_dir, project_root)
    records_by_path = {
        record["resolved_path"]: {
            **record,
            "product_id": manifest.get("product_id"),
            "task_id": manifest.get("task_id"),
        }
        for manifest in manifests
        for record in manifest["downloads"]
    }

    files: list[dict[str, Any]] = []
    discovered_paths: set[str] = set()
    if raw_dir.exists():
        for path in sorted(item for item in raw_dir.rglob("*") if item.is_file()):
            resolved_path = str(path.resolve())
            discovered_paths.add(resolved_path)
            record = records_by_path.get(resolved_path)
            files.append(
                _inspect_file(
                    path,
                    project_root=project_root,
                    manifest_record=record,
                    metadata_products=metadata_products,
                )
            )
    else:
        logger.warning("Raw data directory does not exist: %s", raw_dir)

    for resolved_path, record in records_by_path.items():
        if resolved_path not in discovered_paths:
            files.append(
                _missing_manifest_file(
                    Path(resolved_path),
                    record,
                    project_root=project_root,
                    metadata_products=metadata_products,
                )
            )

    grouped_files: dict[str, list[dict[str, Any]]] = {}
    for item in files:
        key = item.get("product_id") or "unidentified"
        grouped_files.setdefault(key, []).append(item)

    datasets = []
    for product_id, product_files in sorted(grouped_files.items()):
        products = metadata_products.get(product_id, {})
        datasets.append(
            {
                "product_id": product_id if product_id != "unidentified" else None,
                "product_name": products.get("product_name"),
                "metadata_file": products.get("metadata_file"),
                "availability": availability.get(product_id),
                "layers": _layers_for_files(product_files, products),
                "files": product_files,
            }
        )

    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "data_directory": str(data_path),
        "task_manifests": [
            {
                "task_id": manifest.get("task_id"),
                "product_id": manifest.get("product_id"),
                "requested_layers": manifest.get("layers", []),
                "start_date": manifest.get("start_date"),
                "end_date": manifest.get("end_date"),
                "task_status": manifest.get("task_status"),
                "download_status": manifest.get("download_status"),
                "download_count": len(manifest["downloads"]),
            }
            for manifest in manifests
        ],
        "file_count": len(files),
        "valid_file_count": sum(item["validation"]["valid"] for item in files),
        "invalid_file_count": sum(not item["validation"]["valid"] for item in files),
        "missing_file_count": sum(
            item["validation"].get("exists") is False for item in files
        ),
        "datasets": datasets,
        "unassociated_files": grouped_files.get("unidentified", []),
        "catalog_metadata": _catalog_summary(metadata_dir),
    }


def write_inventory(
    inventory: dict[str, Any],
    *,
    json_path: str | Path = "data/metadata/dataset_inventory.json",
    report_path: str | Path = "reports/data_inventory.md",
) -> tuple[Path, Path]:
    """Persist machine-readable inventory and its human-readable summary."""

    json_destination = Path(json_path)
    report_destination = Path(report_path)
    json_destination.parent.mkdir(parents=True, exist_ok=True)
    report_destination.parent.mkdir(parents=True, exist_ok=True)
    json_destination.write_text(
        json.dumps(inventory, indent=2, ensure_ascii=False, default=str) + "\n",
        encoding="utf-8",
    )
    report_destination.write_text(_format_report(inventory), encoding="utf-8")
    return json_destination, report_destination


def _load_product_metadata(
    metadata_dir: Path,
) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]]]:
    products: dict[str, dict[str, Any]] = {}
    availability: dict[str, dict[str, Any]] = {}
    if not metadata_dir.exists():
        return products, availability

    report_path = metadata_dir / "availability_report.json"
    if report_path.exists():
        report = _read_json(report_path)
        if isinstance(report, dict):
            for dataset in report.get("datasets", []):
                if not isinstance(dataset, dict):
                    continue
                for product_id in dataset.get("available_product_ids", []):
                    if isinstance(product_id, str):
                        availability[product_id] = {
                            "dataset": dataset.get("dataset"),
                            "status": dataset.get("status"),
                        }

    for metadata_path in sorted(metadata_dir.glob("*.json")):
        if metadata_path.name == "availability_report.json":
            continue
        content = _read_json(metadata_path)
        if not isinstance(content, dict):
            continue
        for entry in content.get("available_products", []):
            if not isinstance(entry, dict):
                continue
            catalog = entry.get("catalog", {})
            product_id = catalog.get("ProductAndVersion")
            if isinstance(product_id, str):
                products[product_id] = {
                    "metadata_file": str(metadata_path),
                    "product_name": catalog.get("Product"),
                    "metadata": entry,
                }
    return products, availability


def _catalog_summary(metadata_dir: Path) -> dict[str, Any]:
    report_path = metadata_dir / "availability_report.json"
    if not report_path.exists():
        return {"available": False, "reason": "No availability_report.json was found."}
    report = _read_json(report_path)
    if not isinstance(report, dict):
        return {"available": False, "reason": "Availability report is not a JSON object."}
    return {
        "available": True,
        "generated_at": report.get("generated_at"),
        "catalog_product_count": report.get("catalog_product_count"),
        "datasets": [
            {
                "dataset": entry.get("dataset"),
                "status": entry.get("status"),
                "available_product_ids": entry.get("available_product_ids", []),
                "reason": entry.get("reason"),
            }
            for entry in report.get("datasets", [])
            if isinstance(entry, dict)
        ],
    }


def _load_task_manifests(manifest_dir: Path, project_root: Path) -> list[dict[str, Any]]:
    manifests = []
    if not manifest_dir.exists():
        return manifests
    for path in sorted(manifest_dir.glob("*.json")):
        if path.name.endswith(".outputs.json"):
            continue
        content = _read_json(path)
        if not isinstance(content, dict) or not isinstance(content.get("downloads"), list):
            logger.warning("Skipping non-task manifest: %s", path)
            continue
        resolved_records = []
        for record in content["downloads"]:
            if not isinstance(record, dict) or not isinstance(record.get("local_path"), str):
                continue
            local_path = Path(record["local_path"])
            if not local_path.is_absolute():
                local_path = project_root / local_path
            resolved_records.append(
                {**record, "resolved_path": str(local_path.resolve())}
            )
        manifests.append({**content, "downloads": resolved_records})
    return manifests


def _inspect_file(
    path: Path,
    *,
    project_root: Path,
    manifest_record: dict[str, Any] | None,
    metadata_products: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    relative_path = _relative_path(path, project_root)
    product_id = manifest_record.get("product_id") if manifest_record else None
    if product_id is None:
        product_id = _product_id_from_path(path, metadata_products)
    product = metadata_products.get(product_id, {}) if product_id else {}
    layer_id = _layer_id_from_path(path, product)
    item: dict[str, Any] = {
        "path": relative_path,
        "file_name": path.name,
        "format": path.suffix.casefold().lstrip(".") or "unknown",
        "size_bytes": path.stat().st_size,
        "product_id": product_id,
        "layer_id": layer_id,
        "acquisition_date": _date_from_path(path),
        "validation": _validate_manifest_file(path, manifest_record),
    }
    if path.suffix.casefold() in _RASTER_SUFFIXES:
        try:
            item["raster"] = _inspect_raster(path)
            item["validation"]["format_valid"] = True
        except (OSError, rasterio.errors.RasterioError, ValueError) as error:
            item["validation"].update(
                {
                    "format_valid": False,
                    "valid": False,
                    "errors": [*item["validation"]["errors"], str(error)],
                }
            )
    else:
        try:
            item["content"] = _inspect_non_raster(path)
        except (OSError, UnicodeDecodeError, csv.Error, json.JSONDecodeError) as error:
            item["content"] = {
                "parse_valid": False,
                "kind": item["format"],
                "error": str(error),
            }
        format_valid = item["content"].get("parse_valid")
        item["validation"]["format_valid"] = format_valid
        if format_valid is False:
            item["validation"]["errors"].append(
                item["content"].get("error", "File format validation failed.")
            )
        item["validation"]["valid"] = (
            item["validation"]["integrity_valid"] and format_valid is True
        )
        if format_valid is None:
            item["validation"]["errors"].append(
                "No format validator is available for this file type."
            )

    if layer_id and product:
        item["product_layer_metadata"] = product.get("metadata", {}).get(
            "layers", {}
        ).get(layer_id)
    if manifest_record:
        item["task_id"] = manifest_record.get("task_id")
    return item


def _missing_manifest_file(
    path: Path,
    record: dict[str, Any],
    *,
    project_root: Path,
    metadata_products: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    product_id = record.get("product_id")
    product = metadata_products.get(product_id, {}) if product_id else {}
    filename = record.get("filename", path.name)
    display_path = Path(filename)
    layer_id = _layer_id_from_path(display_path, product)
    if layer_id is None:
        layer_id = _layer_id_from_path(path, product)
    return {
        "path": _relative_path(path, project_root),
        "file_name": path.name,
        "format": path.suffix.casefold().lstrip(".") or "unknown",
        "size_bytes": None,
        "product_id": product_id,
        "layer_id": layer_id,
        "acquisition_date": _date_from_path(display_path) or _date_from_path(path),
        "task_id": record.get("task_id"),
        "validation": {
            "valid": False,
            "integrity_valid": False,
            "format_valid": None,
            "exists": False,
            "manifest_status": record.get("status"),
            "errors": ["File listed in the task manifest is missing from disk."],
        },
    }


def _inspect_raster(path: Path) -> dict[str, Any]:
    with rasterio.open(path) as source:
        mask = source.read_masks()
        valid_pixels = int((mask > 0).sum())
        total_pixels = source.width * source.height * source.count
        return {
            "driver": source.driver,
            "width": source.width,
            "height": source.height,
            "band_count": source.count,
            "dtypes": list(source.dtypes),
            "crs": source.crs.to_string() if source.crs else None,
            "crs_label": _crs_label(source.crs),
            "crs_wkt": source.crs.to_wkt() if source.crs else None,
            "bounds": list(source.bounds),
            "transform": list(source.transform),
            "resolution": list(source.res),
            "nodata": source.nodata,
            "scales": list(source.scales),
            "offsets": list(source.offsets),
            "units": list(source.units),
            "band_descriptions": list(source.descriptions),
            "tags": source.tags(),
            "valid_pixel_count": valid_pixels,
            "masked_pixel_count": total_pixels - valid_pixels,
        }


def _inspect_non_raster(path: Path) -> dict[str, Any]:
    suffix = path.suffix.casefold()
    if suffix == ".json":
        _read_json(path)
        return {"parse_valid": True, "kind": "json"}
    if suffix == ".csv":
        with path.open(encoding="utf-8-sig", newline="") as stream:
            rows = csv.reader(stream)
            header = next(rows, [])
            row_count = sum(1 for _ in rows)
        return {
            "parse_valid": bool(header),
            "kind": "csv",
            "columns": header,
            "data_row_count": row_count,
        }
    if suffix == ".xml":
        try:
            ET.parse(path)
        except ET.ParseError as error:
            return {"parse_valid": False, "kind": "xml", "error": str(error)}
        return {"parse_valid": True, "kind": "xml"}
    if suffix in {".txt", ".md"}:
        try:
            with path.open(encoding="utf-8") as stream:
                first_line = stream.readline().rstrip()
        except UnicodeDecodeError as error:
            return {"parse_valid": False, "kind": "text", "error": str(error)}
        return {"parse_valid": True, "kind": "text", "first_line": first_line[:250]}
    return {"parse_valid": None, "kind": "uninspected"}


def _validate_manifest_file(
    path: Path,
    record: dict[str, Any] | None,
) -> dict[str, Any]:
    errors = []
    if not path.is_file():
        errors.append("File is missing.")
        return {"valid": False, "integrity_valid": False, "errors": errors}
    result: dict[str, Any] = {
        "valid": False,
        "integrity_valid": False,
        "exists": True,
        "manifest_status": record.get("status") if record else None,
        "errors": errors,
    }
    if record:
        expected_size = record.get("expected_size_bytes")
        if expected_size is not None and path.stat().st_size != expected_size:
            errors.append(
                f"File size mismatch: expected {expected_size}, found {path.stat().st_size}."
            )
        expected_digest = record.get("checksum")
        if expected_digest:
            digest = _sha256(path)
            result["sha256"] = digest
            if digest.casefold() != str(expected_digest).casefold():
                errors.append("SHA-256 checksum does not match the task manifest.")
        result["integrity_valid"] = (
            not errors and record.get("status") == "validated"
        )
        result["valid"] = result["integrity_valid"]
        if record.get("status") != "validated":
            errors.append(
                f"Task manifest marks this file {record.get('status')!r}, not 'validated'."
            )
            result["valid"] = False
    else:
        result["sha256"] = _sha256(path)
        result["valid"] = False
        errors.append("No task manifest record was found for this file.")
    return result


def _layers_for_files(
    files: list[dict[str, Any]],
    product: dict[str, Any],
) -> list[dict[str, Any]]:
    layer_ids = sorted({item["layer_id"] for item in files if item.get("layer_id")})
    layer_metadata = product.get("metadata", {}).get("layers", {})
    return [
        {
            "layer_id": layer_id,
            "metadata": layer_metadata.get(layer_id),
            "file_count": sum(item.get("layer_id") == layer_id for item in files),
            "acquisition_dates": sorted(
                {
                    item["acquisition_date"]
                    for item in files
                    if item.get("layer_id") == layer_id and item.get("acquisition_date")
                }
            ),
        }
        for layer_id in layer_ids
    ]


def _product_id_from_path(
    path: Path,
    metadata_products: dict[str, dict[str, Any]],
) -> str | None:
    for product_id in sorted(metadata_products, key=len, reverse=True):
        if product_id in path.name or product_id.replace(".", "-") in path.name:
            return product_id
    return None


def _layer_id_from_path(path: Path, product: dict[str, Any]) -> str | None:
    layers = product.get("metadata", {}).get("layers", {})
    for layer_id in sorted(layers, key=len, reverse=True):
        if layer_id in path.name:
            return layer_id
    match = _LAYER_DATE.search(path.name)
    if match:
        layer_id = match.group(1)
        return f"_{layer_id}" if layer_id[0].isdigit() else layer_id
    return None


def _date_from_path(path: Path) -> str | None:
    match = _LAYER_DATE.search(path.name)
    if not match:
        return None
    try:
        return datetime.strptime(match.group(2), "%Y%m%d").date().isoformat()
    except ValueError:
        return None


def _relative_path(path: Path, root: Path) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return path.resolve().as_posix()


def _read_json(path: Path) -> Any:
    with path.open(encoding="utf-8") as stream:
        return json.load(stream)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _crs_label(crs: rasterio.crs.CRS | None) -> str | None:
    if crs is None:
        return None
    epsg = crs.to_epsg()
    if epsg is not None:
        return f"EPSG:{epsg}"
    return crs.to_proj4()


def _format_report(inventory: dict[str, Any]) -> str:
    lines = [
        "# Local Satellite Data Inventory",
        "",
        f"- Generated: {inventory['generated_at']}",
        f"- Data directory: `{inventory['data_directory']}`",
        f"- Files inspected: {inventory['file_count']}",
        f"- Files passing integrity and supported format checks: {inventory['valid_file_count']}",
        f"- Files needing attention or without a manifest: {inventory['invalid_file_count']}",
        f"- Manifest-listed files missing from disk: {inventory['missing_file_count']}",
        "",
        "## Collected products",
        "",
    ]
    datasets = [
        dataset
        for dataset in inventory["datasets"]
        if dataset.get("product_id") is not None
    ]
    if not datasets:
        lines.append("No files associated with a known product were found.")
    for dataset in datasets:
        lines.extend(
            [
                f"### {dataset['product_id']}",
                "",
                f"- Product: {dataset.get('product_name') or 'not recorded'}",
                f"- Files: {len(dataset['files'])}",
            ]
        )
        if dataset.get("availability"):
            lines.append(
                f"- Catalog status: {dataset['availability'].get('status', 'not recorded')}"
            )
        for layer in dataset["layers"]:
            metadata = layer.get("metadata") or {}
            lines.append(
                f"- Layer `{layer['layer_id']}`: {layer['file_count']} file(s); "
                f"units `{metadata.get('Units') or 'not documented'}`, "
                f"scale {metadata.get('ScaleFactor') or 'not documented'}, "
                f"offset {metadata.get('AddOffset') if metadata.get('AddOffset') is not None else 'not documented'}, "
                f"valid range [{metadata.get('ValidMin', 'unknown')}, "
                f"{metadata.get('ValidMax', 'unknown')}], "
                f"NoData {metadata.get('FillValue', 'unknown')}; "
                f"dates {', '.join(layer['acquisition_dates']) or 'unknown'}."
            )
        for item in dataset["files"]:
            raster = item.get("raster")
            if raster:
                lines.append(
                    f"  - `{item['path']}`: {raster['width']}×{raster['height']}, "
                    f"{raster['band_count']} band(s), {raster['dtypes']}, "
                    f"resolution {raster['resolution']}, CRS "
                    f"`{raster['crs_label'] or 'unknown'}`, "
                    f"valid/masked pixels {raster['valid_pixel_count']}/"
                    f"{raster['masked_pixel_count']}; "
                    f"validation {'valid' if item['validation']['valid'] else 'CHECK'}."
                )
            else:
                content = item.get("content", {})
                description = f"{item['format']} file, {item['size_bytes']} bytes"
                if content.get("kind") == "csv":
                    description += (
                        f", {content['data_row_count']} data row(s), "
                        f"{len(content['columns'])} column(s)"
                    )
                elif content.get("kind") == "text" and content.get("first_line"):
                    description += f", starts: {content['first_line']!r}"
                description += (
                    "; format "
                    + ("valid" if item["validation"].get("format_valid") else "CHECK")
                )
                if item["validation"]["errors"]:
                    description += "; issue(s): " + "; ".join(
                        item["validation"]["errors"]
                    )
                lines.append(f"  - `{item['path']}`: {description}.")
        lines.append("")

    lines.extend(["## Collection task records", ""])
    if inventory["task_manifests"]:
        for task in inventory["task_manifests"]:
            lines.append(
                f"- `{task['task_id']}` — {task['product_id']}, "
                f"{task['task_status']}/{task['download_status']}, "
                f"{task['download_count']} bundle files, "
                f"requested {task['start_date']} to {task['end_date']}."
            )
    else:
        lines.append("No collection task manifests were found.")

    catalog = inventory["catalog_metadata"]
    lines.extend(["", "## Catalog context", ""])
    if catalog.get("available"):
        lines.append(
            f"Saved AppEEARS catalog snapshot: {catalog.get('catalog_product_count')} "
            f"catalog products; generated {catalog.get('generated_at')}."
        )
        for entry in catalog["datasets"]:
            products_text = ", ".join(entry["available_product_ids"]) or "none"
            lines.append(
                f"- {entry['dataset']}: {entry['status']}; available product IDs: "
                f"{products_text}."
            )
    else:
        lines.append(catalog["reason"])

    if inventory["unassociated_files"]:
        lines.extend(["", "## Unassociated files", ""])
        lines.extend(f"- `{item['path']}`" for item in inventory["unassociated_files"])
    lines.extend(
        [
            "",
            "## Interpretation limits",
            "",
            "- Raster validity and manifest integrity do not establish that the sample "
            "is representative of a larger region.",
            "- Resampling a single pixel does not create additional spatial information.",
            "- Scale factors and units shown here come from saved AppEEARS product metadata; "
            "quality-mask decoding and scientific suitability analysis are separate steps.",
        ]
    )
    raster_files = [
        item for dataset in inventory["datasets"] for item in dataset["files"]
        if item.get("raster")
    ]
    if raster_files and all(
        item["raster"]["width"] * item["raster"]["height"] == 1
        for item in raster_files
    ):
        lines.append(
            f"- All {len(raster_files)} currently available raster files contain one "
            "pixel each; this test collection cannot support spatial suitability mapping."
        )
    observation_dates = sorted(
        {
            item["acquisition_date"]
            for item in raster_files
            if item.get("acquisition_date")
        }
    )
    if len(observation_dates) < 3:
        lines.append(
            f"- Only {len(observation_dates)} raster acquisition date(s) are available; "
            "this is insufficient for seasonal or historical trend analysis."
        )
    lines.append("")
    return "\n".join(lines)


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Inventory local AppEEARS data and metadata.")
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--json", default="data/metadata/dataset_inventory.json")
    parser.add_argument("--report", default="reports/data_inventory.md")
    args = parser.parse_args()

    inventory = build_inventory(args.data_dir)
    json_path, report_path = write_inventory(
        inventory,
        json_path=args.json,
        report_path=args.report,
    )
    print(f"Inventory: {json_path}")
    print(f"Report: {report_path}")
    print(
        f"Inspected {inventory['file_count']} files: "
        f"{inventory['valid_file_count']} valid, "
        f"{inventory['invalid_file_count']} requiring attention."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
