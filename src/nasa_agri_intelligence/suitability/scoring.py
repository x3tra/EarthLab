from __future__ import annotations

from typing import Any

from ..crop_requirements import DEFAULT_CROP_REQUIREMENTS_PATH, load_crop_requirements
from .criteria import criterion_score
from .crop_database import get_crop_requirements_for_analysis


def normalize_weights(weights: dict[str, float] | None) -> dict[str, float]:
    """Normalize a criterion-weight map so that the total equals 1.0."""
    if not weights:
        return {}
    total = sum(float(value) for value in weights.values())
    if total <= 0:
        raise ValueError("Weight total must be greater than zero.")
    return {key: float(value) / total for key, value in weights.items()}


def evaluate_crop(
    crop_id_or_name: str,
    environmental_profile: dict[str, float | int | None],
    *,
    requirements_path: str = str(DEFAULT_CROP_REQUIREMENTS_PATH),
) -> dict[str, Any]:
    """Return a transparent crop suitability summary for a single cell/profile."""
    requirements = load_crop_requirements(requirements_path)
    crop = get_crop_requirements_for_analysis(crop_id_or_name, requirements_path)

    observed: list[dict[str, Any]] = []
    missing: list[str] = []
    total = 0.0
    weighted = 0.0
    for item in crop.get("guidance", []):
        parameter = item.get("parameter")
        if not isinstance(parameter, str):
            continue
        current_value = environmental_profile.get(parameter)
        if current_value is None:
            missing.append(parameter)
            continue
        score = criterion_score(current_value, item)
        if score is None:
            missing.append(parameter)
            continue
        observed.append({
            "parameter": parameter,
            "score": float(score),
            "source": item.get("source"),
            "qualification": item.get("qualification"),
        })
        total += 1.0
        weighted += float(score)

    if total:
        average_score = weighted / total
    else:
        average_score = 0.0

    return {
        "crop_id": crop["id"],
        "crop_name": crop["name"],
        "score": round(float(min(max(average_score, 0.0), 100.0)), 2),
        "criteria_evaluated": len(observed),
        "criteria_missing": len(missing),
        "missing_parameters": missing,
        "observations": observed,
        "suitable": len(observed) > 0 and average_score >= 60.0,
        "confidence": 0.0 if not observed else round(min(1.0, len(observed) / max(1, len(observed) + len(missing))), 4),
    }
