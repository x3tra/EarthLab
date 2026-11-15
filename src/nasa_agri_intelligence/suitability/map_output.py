from __future__ import annotations

from typing import Any


def build_crop_map(results: list[dict[str, Any]]) -> dict[str, Any]:
    """Create a simple categorical crop map summary for a collection of evaluated cells."""
    palette = {
        "cotton": "#2e7d32",
        "wheat": "#f2c94c",
        "rice": "#1e88e5",
        "maize": "#fb8c00",
        "barley": "#8d6e63",
        "sunflower": "#8e24aa",
        "soybean": "#00acc1",
    }
    best_crop = max(results, key=lambda item: item.get("score", 0.0), default={"crop_id": "no_data"})
    return {
        "best_crop_id": best_crop.get("crop_id", "no_data"),
        "best_crop_name": best_crop.get("crop_name", "No data"),
        "palette": palette,
        "legend": [
            {"crop_id": crop_id, "color": color}
            for crop_id, color in palette.items()
        ],
    }
