from __future__ import annotations


def calculate_confidence(criteria_evaluated: int, criteria_missing: int) -> float:
    """Provide a simple confidence estimate from the ratio of observed versus missing criteria."""
    total = max(1, criteria_evaluated + criteria_missing)
    return round(min(1.0, criteria_evaluated / total), 4)
