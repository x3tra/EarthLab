from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any


def load_manifest(path: str | Path) -> dict[str, Any]:
    manifest_path = Path(path)
    with manifest_path.open(encoding="utf-8") as file:
        manifest = json.load(file)
    if not isinstance(manifest, dict):
        raise ValueError(f"Manifest must contain a JSON object: {manifest_path}")
    return manifest


def write_manifest(path: str | Path, manifest: dict[str, Any]) -> Path:
    """Write a manifest atomically in the same directory as its destination."""

    manifest_path = Path(path)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = manifest_path.with_name(f".{manifest_path.name}.tmp")
    try:
        with temporary_path.open("w", encoding="utf-8") as file:
            json.dump(manifest, file, indent=2, ensure_ascii=False)
            file.write("\n")
            file.flush()
            os.fsync(file.fileno())
        temporary_path.replace(manifest_path)
    except OSError:
        temporary_path.unlink(missing_ok=True)
        raise
    return manifest_path


def manifest_paths(directory: str | Path) -> list[Path]:
    manifest_dir = Path(directory)
    if not manifest_dir.exists():
        return []
    return sorted(
        path
        for path in manifest_dir.glob("*.json")
        if path.name != "availability_report.json"
    )
