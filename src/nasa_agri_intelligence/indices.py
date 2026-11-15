from __future__ import annotations

import numpy as np


def _as_float_array(data: np.ndarray) -> np.ndarray:
    return np.asarray(data, dtype=np.float64)


def _valid_mask(values: np.ndarray, nodata: float | None = None) -> np.ndarray:
    mask = np.isfinite(values)
    if nodata is not None:
        mask &= values != nodata
    return mask


def safe_ratio(numerator: np.ndarray, denominator: np.ndarray, *, nodata: float | None = None) -> np.ndarray:
    numerator = _as_float_array(numerator)
    denominator = _as_float_array(denominator)
    result = np.full(np.broadcast_shapes(numerator.shape, denominator.shape), np.nan, dtype=np.float64)
    valid = _valid_mask(numerator, nodata) & _valid_mask(denominator, nodata) & (np.abs(denominator) > 0)
    result[valid] = numerator[valid] / denominator[valid]
    return result


def ndvi(red: np.ndarray, nir: np.ndarray, *, nodata: float | None = None) -> np.ndarray:
    """Compute NDVI = (NIR - RED) / (NIR + RED)."""

    red_arr = _as_float_array(red)
    nir_arr = _as_float_array(nir)
    denominator = nir_arr + red_arr
    result = np.full(np.broadcast_shapes(red_arr.shape, nir_arr.shape), np.nan, dtype=np.float64)
    valid = _valid_mask(red_arr, nodata) & _valid_mask(nir_arr, nodata) & (np.abs(denominator) > 0)
    result[valid] = (nir_arr[valid] - red_arr[valid]) / denominator[valid]
    return result


def ndmi(nir: np.ndarray, swir: np.ndarray, *, nodata: float | None = None) -> np.ndarray:
    """Compute NDMI = (NIR - SWIR) / (NIR + SWIR)."""

    nir_arr = _as_float_array(nir)
    swir_arr = _as_float_array(swir)
    denominator = nir_arr + swir_arr
    result = np.full(np.broadcast_shapes(nir_arr.shape, swir_arr.shape), np.nan, dtype=np.float64)
    valid = _valid_mask(nir_arr, nodata) & _valid_mask(swir_arr, nodata) & (np.abs(denominator) > 0)
    result[valid] = (nir_arr[valid] - swir_arr[valid]) / denominator[valid]
    return result


def savi(red: np.ndarray, nir: np.ndarray, *, L: float = 0.5, nodata: float | None = None) -> np.ndarray:
    """Compute SAVI = ((NIR - RED) / (NIR + RED + L)) * (1 + L)."""

    red_arr = _as_float_array(red)
    nir_arr = _as_float_array(nir)
    denominator = nir_arr + red_arr + L
    result = np.full(np.broadcast_shapes(red_arr.shape, nir_arr.shape), np.nan, dtype=np.float64)
    valid = _valid_mask(red_arr, nodata) & _valid_mask(nir_arr, nodata) & (np.abs(denominator) > 0)
    result[valid] = ((nir_arr[valid] - red_arr[valid]) / denominator[valid]) * (1.0 + L)
    return result


def mndwi(green: np.ndarray, swir: np.ndarray, *, nodata: float | None = None) -> np.ndarray:
    """Compute MNDWI = (GREEN - SWIR) / (GREEN + SWIR)."""

    green_arr = _as_float_array(green)
    swir_arr = _as_float_array(swir)
    denominator = green_arr + swir_arr
    result = np.full(np.broadcast_shapes(green_arr.shape, swir_arr.shape), np.nan, dtype=np.float64)
    valid = _valid_mask(green_arr, nodata) & _valid_mask(swir_arr, nodata) & (np.abs(denominator) > 0)
    result[valid] = (green_arr[valid] - swir_arr[valid]) / denominator[valid]
    return result


def evi(red: np.ndarray, nir: np.ndarray, blue: np.ndarray, *, G: float = 2.5, C1: float = 6.0, C2: float = 7.5, L: float = 1.0, nodata: float | None = None) -> np.ndarray:
    """Compute EVI with the standard coefficients from MODIS vegetation products."""

    red_arr = _as_float_array(red)
    nir_arr = _as_float_array(nir)
    blue_arr = _as_float_array(blue)
    denominator = (nir_arr + (C1 * red_arr) - (C2 * blue_arr) + L)
    result = np.full(np.broadcast_shapes(red_arr.shape, nir_arr.shape, blue_arr.shape), np.nan, dtype=np.float64)
    valid = _valid_mask(red_arr, nodata) & _valid_mask(nir_arr, nodata) & _valid_mask(blue_arr, nodata) & (np.abs(denominator) > 0)
    result[valid] = G * (nir_arr[valid] - red_arr[valid]) / denominator[valid]
    return result
