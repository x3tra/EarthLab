from pathlib import Path

from nasa_agri_intelligence.collection.layer_validation import (
    LayerRule,
    validate_layers,
)
from nasa_agri_intelligence.collection.product_discovery import (
    DatasetSpec,
    ProductDiscovery,
)


def test_layer_validation_returns_actual_api_layer_identifiers() -> None:
    metadata = {
        "_250m_16_days_NDVI": {
            "Description": "16 day NDVI average",
            "Units": "NDVI",
        },
        "_250m_16_days_VI_Quality": {
            "Description": "Vegetation Index quality",
            "IsQA": True,
        },
    }

    checks = validate_layers(
        metadata,
        [
            LayerRule(label="NDVI", match_any=["NDVI"]),
            LayerRule(label="DetailedQA", match_any=["DetailedQA", "detailed quality"]),
        ],
    )

    assert checks[0]["status"] == "available"
    assert checks[0]["matched_layers"][0]["layer_id"] == "_250m_16_days_NDVI"
    assert checks[1]["status"] == "unavailable"


def test_exact_layer_match_does_not_match_a_longer_identifier() -> None:
    metadata = {
        "ET_500m": {"Description": "Total Evapotranspiration"},
        "PET_500m": {"Description": "Total Potential Evapotranspiration"},
    }

    checks = validate_layers(
        metadata,
        [LayerRule(label="ET_500m", match_any=["ET_500m"])],
    )

    assert [layer["layer_id"] for layer in checks[0]["matched_layers"]] == ["ET_500m"]


def test_product_config_loads_from_yaml() -> None:
    config_path = Path(__file__).parents[1] / "config" / "datasets.yaml"

    config = ProductDiscovery.load_config(config_path)

    assert len(config.datasets) == 12
    hls = next(dataset for dataset in config.datasets if dataset.name == "HLSL30")
    assert hls.requested_product_ids == ["HLSL30.020"]


def test_product_family_match_includes_catalog_description() -> None:
    spec = DatasetSpec(
        name="GPM",
        metadata_file="GPM.json",
        product_contains=["IMERG"],
    )

    assert ProductDiscovery._matches_family(
        {"ProductAndVersion": "3IMERGDF.07", "Description": "GPM IMERG Final Run"},
        spec,
    )


def test_hls_vegetation_index_product_is_not_a_reflectance_version_match() -> None:
    spec = DatasetSpec(
        name="HLSL30",
        metadata_file="HLSL30.json",
        product_prefixes=["HLSL30."],
    )

    assert not ProductDiscovery._matches_family(
        {"ProductAndVersion": "HLSL30_VI.020", "Product": "HLSL30_VI"},
        spec,
    )
