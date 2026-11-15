import numpy as np

from nasa_agri_intelligence.indices import mndwi, ndmi, ndvi, savi
from nasa_agri_intelligence.models import DATASET_CATALOG


def test_ndvi_formula_matches_expected() -> None:
    red = np.array([0.2, 0.3])
    nir = np.array([0.8, 0.6])

    result = ndvi(red, nir)

    expected = np.array([(0.8 - 0.2) / (0.8 + 0.2), (0.6 - 0.3) / (0.6 + 0.3)])
    np.testing.assert_allclose(result, expected)


def test_ndmi_and_savi_are_calculated() -> None:
    nir = np.array([0.7, 0.8])
    swir = np.array([0.2, 0.1])
    red = np.array([0.2, 0.1])

    ndmi_result = ndmi(nir, swir)
    savi_result = savi(red, nir)

    np.testing.assert_allclose(ndmi_result, np.array([(0.7 - 0.2) / (0.7 + 0.2), (0.8 - 0.1) / (0.8 + 0.1)]))
    np.testing.assert_allclose(
        savi_result,
        np.array([
            ((0.7 - 0.2) / (0.7 + 0.2 + 0.5)) * (1 + 0.5),
            ((0.8 - 0.1) / (0.8 + 0.1 + 0.5)) * (1 + 0.5),
        ]),
    )


def test_mndwi_uses_green_and_swir() -> None:
    green = np.array([0.6, 0.7])
    swir = np.array([0.2, 0.15])

    result = mndwi(green, swir)
    expected = np.array([(0.6 - 0.2) / (0.6 + 0.2), (0.7 - 0.15) / (0.7 + 0.15)])
    np.testing.assert_allclose(result, expected)


def test_dataset_catalog_contains_required_datasets() -> None:
    required = [
        "hls",
        "modis_ndvi",
        "modis_lai_fpar",
        "modis_lst",
        "modis_et",
        "smap",
        "ecostress",
        "aster_gdem",
        "gpm",
    ]

    for key in required:
        assert key in DATASET_CATALOG
