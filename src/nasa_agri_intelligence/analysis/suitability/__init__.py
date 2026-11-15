"""Compatibility exports for the crop suitability workflow."""

from nasa_agri_intelligence.suitability import (
    calculate_confidence,
    criterion_score,
    evaluate_cell,
    evaluate_crop,
    get_crop_requirements_for_analysis,
    list_available_crops,
    normalize_weights,
    run_spatial_suitability,
    score_numeric_range,
    select_best_crop,
    suitability_layers,
)

__all__ = [
    "calculate_confidence",
    "criterion_score",
    "score_numeric_range",
    "get_crop_requirements_for_analysis",
    "list_available_crops",
    "evaluate_crop",
    "normalize_weights",
    "evaluate_cell",
    "select_best_crop",
    "run_spatial_suitability",
    "suitability_layers",
]
