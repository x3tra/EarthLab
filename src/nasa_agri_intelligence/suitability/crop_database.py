from __future__ import annotations

from pathlib import Path
from typing import Any

from ..crop_requirements import DEFAULT_CROP_REQUIREMENTS_PATH, get_crop_requirements, load_crop_requirements


def list_available_crops(path: str | Path = DEFAULT_CROP_REQUIREMENTS_PATH) -> list[dict[str, Any]]:
    """Return all crop entries in the curated reference dataset."""
    requirements = load_crop_requirements(path)
    return list(requirements["crops"])


def get_crop_requirements_for_analysis(
    crop_id_or_name: str,
    path: str | Path = DEFAULT_CROP_REQUIREMENTS_PATH,
) -> dict[str, Any]:
    """Resolve a crop by ID or name using the validated reference file."""
    return get_crop_requirements(crop_id_or_name, load_crop_requirements(path))
