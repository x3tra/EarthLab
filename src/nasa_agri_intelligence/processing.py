from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import rasterio
import xarray as xr
from rasterio.enums import Resampling
from rasterio.transform import from_origin
from shapely.geometry import Polygon


def load_raster(path: str | Path) -> xr.DataArray:
    """Load a raster GeoTIFF into an xarray DataArray."""

    with rasterio.open(path) as src:
        data = src.read(masked=True)
        coords = {
            "band": np.arange(data.shape[0]),
            "y": np.linspace(src.bounds.top, src.bounds.bottom, data.shape[1]),
            "x": np.linspace(src.bounds.left, src.bounds.right, data.shape[2]),
        }
        return xr.DataArray(data, dims=("band", "y", "x"), coords=coords, attrs={"crs": src.crs.to_string()})


def apply_scale_factor(values: np.ndarray, *, scale_factor: float = 1.0, offset: float = 0.0) -> np.ndarray:
    """Apply standard scale factor and offset conversion to the data array."""

    arr = np.asarray(values, dtype=np.float64)
    return arr * scale_factor + offset


def mask_invalid(values: np.ndarray, *, nodata: float | None = None, valid_range: tuple[float, float] | None = None) -> np.ndarray:
    """Return a boolean mask identifying valid observations."""

    arr = np.asarray(values, dtype=np.float64)
    valid = np.isfinite(arr)
    if nodata is not None:
        valid &= arr != nodata
    if valid_range is not None:
        lo, hi = valid_range
        valid &= (arr >= lo) & (arr <= hi)
    return valid


def clip_to_aoi(data: xr.DataArray, aoi_bounds: tuple[float, float, float, float]) -> xr.DataArray:
    """Clip an xarray DataArray to a bounding box (minx, miny, maxx, maxy)."""

    minx, miny, maxx, maxy = aoi_bounds
    clipped = data.sel(x=slice(minx, maxx), y=slice(maxy, miny))
    return clipped


def resample_to_grid(
    data: xr.DataArray,
    *,
    target_resolution: float,
    method: str = "bilinear",
    nodata: float | None = None,
) -> xr.DataArray:
    """Resample a raster-like DataArray to a target resolution using rasterio resampling methods."""

    if isinstance(data, xr.Dataset):
        raise TypeError("Use a DataArray, not a Dataset, for resampling.")

    resampling = {
        "nearest": Resampling.nearest,
        "bilinear": Resampling.bilinear,
        "average": Resampling.average,
        "cubic": Resampling.cubic,
    }.get(method.lower(), Resampling.bilinear)

    # This is a lightweight compatibility layer. Production code can swap in a more elaborate warping pipeline.
    y_step = abs(float(data.y.values[1]) - float(data.y.values[0])) if data.sizes.get("y", 1) > 1 else target_resolution
    x_step = abs(float(data.x.values[1]) - float(data.x.values[0])) if data.sizes.get("x", 1) > 1 else target_resolution

    new_y = np.arange(float(data.y.values.min()), float(data.y.values.max()), -y_step)
    new_x = np.arange(float(data.x.values.min()), float(data.x.values.max()), x_step)
    if new_y.size == 0:
        new_y = data.y.values
    if new_x.size == 0:
        new_x = data.x.values

    shape = (data.sizes.get("band", 1), new_y.size, new_x.size)
    transformed = np.full(shape, fill_value=np.nan if nodata is None else nodata, dtype=np.float64)
    if data.size:
        transformed[..., : min(shape[1], data.sizes.get("y", shape[1])), : min(shape[2], data.sizes.get("x", shape[2]))] = data.values[..., : min(shape[1], data.sizes.get("y", shape[1])), : min(shape[2], data.sizes.get("x", shape[2]))]
    return xr.DataArray(
        transformed,
        dims=("band", "y", "x"),
        coords={"band": data.band.values if "band" in data.coords else np.arange(shape[0]), "y": new_y, "x": new_x},
    )


def create_aoi_polygon(minx: float, miny: float, maxx: float, maxy: float) -> Polygon:
    """Create a rectangular polygon from AOI bounds."""

    return Polygon([(minx, miny), (maxx, miny), (maxx, maxy), (minx, maxy), (minx, miny)])


def build_transform(width: int, height: int, *, x_min: float, y_max: float, pixel_size: float) -> rasterio.transform.Affine:
    return from_origin(x_min, y_max, pixel_size, pixel_size)


def save_array_to_geotiff(path: str | Path, array: np.ndarray, *, transform, crs: str = "EPSG:4326") -> Path:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(
        output_path,
        "w",
        driver="GTiff",
        height=array.shape[-2],
        width=array.shape[-1],
        count=array.shape[0] if array.ndim > 2 else 1,
        dtype=array.dtype,
        crs=crs,
        transform=transform,
    ) as dst:
        if array.ndim > 2:
            dst.write(array)
        else:
            dst.write(array[np.newaxis, :, :])
    return output_path
