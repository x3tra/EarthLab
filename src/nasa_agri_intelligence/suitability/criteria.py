from __future__ import annotations

from typing import Any


def score_numeric_range(
    value: float | int | None,
    *,
    lower_bound: float | int | None = None,
    upper_bound: float | int | None = None,
    preferred_lower: float | int | None = None,
    preferred_upper: float | int | None = None,
) -> float | None:
    """Return a 0-100 suitability score for a numeric variable.

    The function is intentionally conservative: it rewards values inside the
    preferred range, fades gracefully toward the acceptable bounds, and returns
    None for missing data.
    """
    if value is None:
        return None
    value = float(value)
    lower_bound = float(lower_bound) if lower_bound is not None else None
    upper_bound = float(upper_bound) if upper_bound is not None else None
    preferred_lower = float(preferred_lower) if preferred_lower is not None else None
    preferred_upper = float(preferred_upper) if preferred_upper is not None else None

    if preferred_lower is not None and preferred_upper is not None:
        lower_limit = preferred_lower
        upper_limit = preferred_upper
        if lower_bound is None:
            lower_bound = lower_limit
        if upper_bound is None:
            upper_bound = upper_limit
    elif lower_bound is not None and upper_bound is not None:
        lower_limit = lower_bound
        upper_limit = upper_bound
    else:
        return 100.0

    if lower_bound is not None and upper_bound is not None and lower_bound > upper_bound:
        lower_bound, upper_bound = upper_bound, lower_bound

    if preferred_lower is not None and preferred_upper is not None:
        if lower_limit <= value <= upper_limit:
            return 100.0
        if value < lower_limit:
            if lower_bound is None or value >= lower_bound:
                return 100.0 * (value - lower_bound) / (lower_limit - lower_bound) if lower_limit > lower_bound else 100.0
            return 0.0
        if value > upper_limit:
            if upper_bound is None or value <= upper_bound:
                return 100.0 * (upper_bound - value) / (upper_bound - upper_limit) if upper_bound > upper_limit else 100.0
            return 0.0
        return 50.0

    if value < lower_bound:
        if lower_bound == lower_limit:
            return 0.0
        return max(0.0, 100.0 * (value - lower_bound) / (lower_limit - lower_bound))
    if value > upper_bound:
        if upper_bound == upper_limit:
            return 0.0
        return max(0.0, 100.0 * (upper_bound - value) / (upper_bound - upper_limit))
    return 100.0


def criterion_score(value: float | int | None, guidance: dict[str, Any]) -> float | None:
    """Score a single crop guidance item against a measured value."""
    if value is None:
        return None
    values = guidance.get("values")
    if values is None:
        return 100.0

    if isinstance(values, list):
        if len(values) == 0:
            return 100.0
        if len(values) == 1:
            choices = (float(values[0]), float(values[0]))
        else:
            choices = (float(min(values)), float(max(values)))
    elif isinstance(values, (int, float)):
        choices = (float(values), float(values))
    else:
        return 100.0

    preferred_lower, preferred_upper = choices
    lower_bound = guidance.get("minimum")
    upper_bound = guidance.get("maximum")
    if lower_bound is None and upper_bound is None:
        return score_numeric_range(
            value,
            preferred_lower=preferred_lower,
            preferred_upper=preferred_upper,
        )
    return score_numeric_range(
        value,
        lower_bound=lower_bound,
        upper_bound=upper_bound,
        preferred_lower=preferred_lower,
        preferred_upper=preferred_upper,
    )
