from __future__ import annotations

import json
from pathlib import Path

import pytest

from nasa_agri_intelligence.crop_requirements import (
    get_crop_requirements,
    load_crop_requirements,
)


ROOT = Path(__file__).resolve().parents[1]
CROP_FILE = ROOT / "data/reference/crop_requirements.json"


def test_crop_database_contains_22_source_linked_records_and_no_invented_limits() -> None:
    requirements = load_crop_requirements(CROP_FILE)

    assert len(requirements["crops"]) == 22
    assert len({crop["id"] for crop in requirements["crops"]}) == 22
    assert all(crop["environmental_suitability"] is None for crop in requirements["crops"])
    assert get_crop_requirements("Cotton", requirements)["id"] == "cotton"
    assert get_crop_requirements("pomegranate", requirements)["guidance"] == []
    assert "not universal" in requirements["interpretation"][0]
    for crop in requirements["crops"]:
        for item in crop["guidance"]:
            assert item["source"] in requirements["sources"]


def test_crop_database_rejects_unknown_sources(tmp_path: Path) -> None:
    requirements = json.loads(CROP_FILE.read_text(encoding="utf-8"))
    requirements["crops"][0]["guidance"][0]["source"] = "unverified-source"
    path = tmp_path / "invalid.json"
    path.write_text(json.dumps(requirements), encoding="utf-8")

    with pytest.raises(ValueError, match="unknown source"):
        load_crop_requirements(path)


def test_crop_lookup_raises_for_unknown_name() -> None:
    with pytest.raises(KeyError, match="No crop requirements record"):
        get_crop_requirements("unknown crop")
