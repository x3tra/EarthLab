from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any


DEFAULT_CROP_REQUIREMENTS_PATH = Path("data/reference/crop_requirements.json")


def load_crop_requirements(
    path: str | Path = DEFAULT_CROP_REQUIREMENTS_PATH,
) -> dict[str, Any]:
    """Load and validate the curated, source-linked crop guidance dataset."""

    requirements_path = Path(path)
    with requirements_path.open(encoding="utf-8") as file:
        requirements = json.load(file)
    if not isinstance(requirements, dict):
        raise ValueError(f"Crop requirements must be a JSON object: {requirements_path}")
    if requirements.get("schema_version") != 1:
        raise ValueError(f"Unsupported crop requirements schema: {requirements_path}")
    sources = requirements.get("sources")
    crops = requirements.get("crops")
    parameter_units = requirements.get("parameter_units")
    if (
        not isinstance(sources, dict)
        or not isinstance(crops, list)
        or not isinstance(parameter_units, dict)
    ):
        raise ValueError(f"Crop requirements lack sources or crop records: {requirements_path}")

    crop_ids: set[str] = set()
    for crop in crops:
        if not isinstance(crop, dict):
            raise ValueError(f"Invalid crop record in {requirements_path}")
        crop_id = crop.get("id")
        if not isinstance(crop_id, str) or crop_id in crop_ids:
            raise ValueError(f"Missing or duplicate crop ID in {requirements_path}")
        crop_ids.add(crop_id)
        if not isinstance(crop.get("name"), str):
            raise ValueError(f"Crop {crop_id} has no name in {requirements_path}")
        if crop.get("environmental_suitability") is not None:
            raise ValueError(
                f"Unverified environmental suitability must remain null for {crop_id}"
            )
        guidance = crop.get("guidance")
        if not isinstance(guidance, list):
            raise ValueError(f"Crop {crop_id} has invalid guidance in {requirements_path}")
        for item in guidance:
            if not isinstance(item, dict):
                raise ValueError(f"Crop {crop_id} contains an invalid guidance item")
            if not isinstance(item.get("parameter"), str):
                raise ValueError(f"Crop {crop_id} has an unnamed guidance parameter")
            if item["parameter"] not in parameter_units:
                raise ValueError(
                    f"Crop {crop_id} uses an unknown parameter {item['parameter']!r}"
                )
            source_id = item.get("source")
            if not isinstance(source_id, str) or source_id not in sources:
                raise ValueError(
                    f"Crop {crop_id} references unknown source {source_id!r}"
                )
            _validate_values(item.get("values"), crop_id, requirements_path)
    return requirements


def get_crop_requirements(
    crop_id_or_name: str,
    requirements: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Get one crop record by its stable ID or display name."""

    dataset = requirements if requirements is not None else load_crop_requirements()
    query = crop_id_or_name.casefold()
    for crop in dataset["crops"]:
        if crop["id"].casefold() == query or crop["name"].casefold() == query:
            return crop
    raise KeyError(f"No crop requirements record found for {crop_id_or_name!r}")


def _validate_values(values: Any, crop_id: str, path: Path) -> None:
    candidates = values if isinstance(values, list) else [values]
    if not candidates or any(
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        for value in candidates
    ):
        raise ValueError(f"Crop {crop_id} has invalid numeric guidance in {path}")
