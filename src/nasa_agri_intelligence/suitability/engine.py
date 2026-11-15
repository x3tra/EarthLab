from __future__ import annotations

from pathlib import Path
from typing import Any

from ..crop_requirements import DEFAULT_CROP_REQUIREMENTS_PATH, load_crop_requirements
from .crop_database import get_crop_requirements_for_analysis
from .scoring import evaluate_crop


def select_best_crop(results: list[dict[str, Any]]) -> dict[str, Any] | None:
    """Pick the highest-scoring crop from a set of evaluated results."""
    if not results:
        return None
    qualified = [
        item for item in results if item.get("criteria_evaluated", 0) > 0
    ]
    if not qualified:
        return None
    return max(qualified, key=lambda item: item.get("score", 0.0))


def evaluate_cell(
    environmental_profile: dict[str, float | int | None],
    *,
    crop_ids: list[str] | None = None,
    requirements_path: str | Path = DEFAULT_CROP_REQUIREMENTS_PATH,
) -> dict[str, Any]:
    """Evaluate all selected crops for a single cell / profile and return the best match."""
    requirements = load_crop_requirements(requirements_path)
    if crop_ids is None:
        crop_ids = [crop["id"] for crop in requirements["crops"]]

    evaluated = [
        evaluate_crop(crop_id, environmental_profile, requirements_path=str(requirements_path))
        for crop_id in crop_ids
    ]
    best = select_best_crop(evaluated)
    return {
        "best_crop": best,
        "results": evaluated,
        "crop_count": len(evaluated),
        "legend": [
            {
                "crop_id": item["crop_id"],
                "crop_name": item["crop_name"],
                "score": item["score"],
                "criteria_evaluated": item["criteria_evaluated"],
            }
            for item in evaluated
        ],
    }
