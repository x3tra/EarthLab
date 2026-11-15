from __future__ import annotations

from nasa_agri_intelligence.suitability.engine import evaluate_cell
from nasa_agri_intelligence.suitability.scoring import evaluate_crop


def test_evaluate_crop_accepts_reference_guidance() -> None:
    result = evaluate_crop(
        "cotton",
        {
            "season_length_days": 200,
            "kc_mid": 1.15,
            "kc_end": 0.6,
            "effective_root_depth_m": 1.2,
            "depletion_fraction_p": 0.65,
        },
    )

    assert result["crop_id"] == "cotton"
    assert result["score"] > 90
    assert result["criteria_missing"] == 0
    assert result["confidence"] > 0.7


def test_evaluate_cell_selects_best_crop_and_reports_missing_data() -> None:
    result = evaluate_cell(
        {
            "season_length_days": 200,
            "kc_mid": 1.15,
            "kc_end": 0.6,
            "effective_root_depth_m": 1.2,
            "depletion_fraction_p": 0.65,
        },
        crop_ids=["cotton", "wheat"],
    )

    assert result["crop_count"] == 2
    assert result["best_crop"] is not None
    assert result["best_crop"]["crop_id"] in {"cotton", "wheat"}
    assert result["best_crop"]["score"] >= 0
