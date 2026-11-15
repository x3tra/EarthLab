from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np
import rasterio
from pyproj import CRS, Transformer
from rasterio.enums import Resampling
from rasterio.features import geometry_mask
from rasterio.transform import Affine, from_origin
from rasterio.warp import calculate_default_transform, reproject, transform_geom
from rasterio.windows import Window
from shapely.geometry import Polygon, shape

from .collection.task_manager import normalize_aoi
from .collection.task_registry import TaskRegistry

RasterKind = Literal["continuous", "categorical", "quality"]


@dataclass(frozen=True)
class GridSpec:
    crs: str
    transform: Affine
    width: int
    height: int
    resolution: float

    def __post_init__(self) -> None:
        if self.width <= 0 or self.height <= 0:
            raise ValueError("Grid width and height must be positive.")
        if not math.isfinite(self.resolution) or self.resolution <= 0:
            raise ValueError("Grid resolution must be a positive finite number.")


def build_grid(
    aoi_geojson: dict[str, Any],
    *,
    target_crs: str,
    resolution: float,
) -> GridSpec:
    """Create an AOI-bounded, north-up grid snapped to the requested pixel size."""

    if not math.isfinite(resolution) or resolution <= 0:
        raise ValueError("Grid resolution must be a positive finite number.")
    normalized_aoi = normalize_aoi(aoi_geojson)
    validate_aoi_crs_area(normalized_aoi, target_crs)
    transformed_geometries = [
        transform_geom("EPSG:4326", target_crs, feature["geometry"])
        for feature in normalized_aoi["features"]
    ]
    bounds = [shape(geometry).bounds for geometry in transformed_geometries]
    min_x = min(bound[0] for bound in bounds)
    min_y = min(bound[1] for bound in bounds)
    max_x = max(bound[2] for bound in bounds)
    max_y = max(bound[3] for bound in bounds)

    snap_tolerance = 1e-10
    left = math.floor(min_x / resolution + snap_tolerance) * resolution
    bottom = math.floor(min_y / resolution + snap_tolerance) * resolution
    right = math.ceil(max_x / resolution - snap_tolerance) * resolution
    top = math.ceil(max_y / resolution - snap_tolerance) * resolution
    width = max(1, math.ceil((right - left) / resolution))
    height = max(1, math.ceil((top - bottom) / resolution))
    return GridSpec(
        crs=target_crs,
        transform=from_origin(left, top, resolution, resolution),
        width=width,
        height=height,
        resolution=resolution,
    )


def validate_aoi_crs_area(
    normalized_aoi: dict[str, Any],
    target_crs: str,
) -> None:
    """Reject geometries outside a projected CRS's documented area of use."""

    crs = CRS.from_user_input(target_crs)
    area = crs.area_of_use
    if area is None:
        return
    for feature in normalized_aoi["features"]:
        west, south, east, north = shape(feature["geometry"]).bounds
        if (
            west < area.west
            or east > area.east
            or south < area.south
            or north > area.north
        ):
            raise ValueError(
                f"AOI falls outside the documented area of use for {crs.to_string()}."
            )


def coarsest_native_resolution(
    raster_paths: list[str | Path],
    *,
    task_directory: str | Path,
    task_name: str,
    task_id: str,
    target_crs: str,
) -> float:
    """Return the largest selected-task source pixel size in target_crs."""

    if not raster_paths:
        raise ValueError("At least one raster is required to select a grid resolution.")
    _validate_task_rasters(task_directory, task_name, task_id, raster_paths)
    resolutions = []
    for path in raster_paths:
        with rasterio.open(path) as source:
            if source.crs is None:
                raise ValueError(f"Raster has no CRS: {path}")
            resolution = native_pixel_resolution_in_crs(source, target_crs)
            if not math.isfinite(resolution) or resolution <= 0:
                raise ValueError(f"Could not determine transformed resolution for {path}.")
            resolutions.append(resolution)
    return max(resolutions)


def native_pixel_resolution_in_crs(
    source: rasterio.io.DatasetReader,
    target_crs: str,
) -> float:
    """Estimate the largest area-equivalent source pixel size in target CRS."""

    if source.crs is None:
        raise ValueError("Raster has no CRS.")
    target = CRS.from_user_input(target_crs)
    if not target.is_projected:
        target_transform, _, _ = calculate_default_transform(
            source.crs,
            target_crs,
            source.width,
            source.height,
            *source.bounds,
        )
        return max(abs(target_transform.a), abs(target_transform.e))

    transformer = Transformer.from_crs(source.crs, target, always_xy=True)
    rows = sorted({0, source.height // 2, source.height - 1})
    columns = sorted({0, source.width // 2, source.width - 1})
    resolutions = []
    for row in rows:
        for column in columns:
            pixel_transform = source.window_transform(Window(column, row, 1, 1))
            corners = [
                pixel_transform @ (0, 0),
                pixel_transform @ (1, 0),
                pixel_transform @ (1, 1),
                pixel_transform @ (0, 1),
            ]
            x_coordinates, y_coordinates = transformer.transform(
                [point[0] for point in corners],
                [point[1] for point in corners],
            )
            area = Polygon(zip(x_coordinates, y_coordinates, strict=True)).area
            resolution = math.sqrt(area)
            if not math.isfinite(resolution) or resolution <= 0:
                raise ValueError("Could not determine source pixel area in target CRS.")
            resolutions.append(resolution)
    return max(resolutions)


def harmonize_raster(
    source_path: str | Path,
    destination_path: str | Path,
    *,
    task_directory: str | Path,
    task_name: str,
    task_id: str,
    grid: GridSpec,
    aoi_geojson: dict[str, Any],
    kind: RasterKind,
    continuous_method: Literal["auto", "bilinear", "average", "nearest"] = "auto",
    allow_upsampling: bool = False,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Reproject one selected-task raster without changing its semantic type."""

    task_root = Path(task_directory).resolve()
    source_path = Path(source_path).resolve()
    destination_path = Path(destination_path).resolve()
    _validate_task_rasters(task_root, task_name, task_id, [source_path])
    if not destination_path.is_relative_to(task_root):
        raise ValueError("Harmonized output must stay inside the selected task directory.")
    if destination_path.exists() and not overwrite:
        raise FileExistsError(
            f"Harmonized output already exists: {destination_path}. "
            "Use overwrite=True to replace it."
        )

    with rasterio.open(source_path) as source:
        if source.count != 1:
            raise ValueError(f"Expected a single-band raster; found {source.count} in {source_path}.")
        if source.crs is None:
            raise ValueError(f"Raster has no CRS: {source_path}")
        native_resolution = native_pixel_resolution_in_crs(source, grid.crs)
        if (
            grid.resolution < native_resolution * (1 - 1e-6)
            and not allow_upsampling
        ):
            raise ValueError(
                f"Target resolution {grid.resolution:g} is finer than the source "
                f"resolution {native_resolution:g}; refusing to create false detail."
            )
        resampling, resampling_name = _select_resampling(
            kind,
            continuous_method,
            target_is_coarser=grid.resolution > native_resolution * (1 + 1e-6),
        )
        source_mask = source.read_masks(1) > 0
        if source.nodata is None and not source_mask.all():
            raise ValueError(
                f"Raster has masked cells but no NoData value for safe resampling: {source_path}"
            )
        if kind == "continuous":
            source_values = source.read(1).astype(np.float32)
            source_values[~source_mask] = np.nan
            source_nodata: float | int | None = float("nan")
            destination_nodata: float | int | None = float("nan")
            destination = np.full(
                (grid.height, grid.width),
                np.nan,
                dtype=np.float32,
            )
        else:
            source_values = source.read(1)
            source_nodata = source.nodata
            destination_nodata = source.nodata
            destination = np.zeros((grid.height, grid.width), dtype=source.dtypes[0])

        reproject(
            source=source_values,
            destination=destination,
            src_transform=source.transform,
            src_crs=source.crs,
            src_nodata=source_nodata,
            dst_transform=grid.transform,
            dst_crs=grid.crs,
            dst_nodata=destination_nodata,
            resampling=resampling,
            init_dest_nodata=True,
        )
        valid_fraction = np.zeros((grid.height, grid.width), dtype=np.float32)
        mask_resampling = (
            Resampling.average
            if kind == "continuous" and resampling == Resampling.average
            else Resampling.nearest
        )
        reproject(
            source=source_mask.astype(np.float32),
            destination=valid_fraction,
            src_transform=source.transform,
            src_crs=source.crs,
            dst_transform=grid.transform,
            dst_crs=grid.crs,
            dst_nodata=0,
            resampling=mask_resampling,
            init_dest_nodata=True,
        )
        transformed_geometries = [
            transform_geom("EPSG:4326", grid.crs, feature["geometry"])
            for feature in normalize_aoi(aoi_geojson)["features"]
        ]
        inside_aoi = geometry_mask(
            transformed_geometries,
            out_shape=(grid.height, grid.width),
            transform=grid.transform,
            invert=True,
            all_touched=False,
        )
        valid_mask = (valid_fraction > 0) & inside_aoi
        if kind == "continuous":
            valid_mask &= np.isfinite(destination)
            destination[~valid_mask] = np.nan
        elif destination_nodata is not None:
            destination[~valid_mask] = destination_nodata

        destination_path.parent.mkdir(parents=True, exist_ok=True)
        temporary_path = destination_path.with_name(f".{destination_path.name}.tmp")
        profile = source.profile.copy()
        profile.pop("blockxsize", None)
        profile.pop("blockysize", None)
        profile.update(
            driver="GTiff",
            width=grid.width,
            height=grid.height,
            transform=grid.transform,
            crs=grid.crs,
            dtype=destination.dtype,
            nodata=destination_nodata,
            compress="deflate",
        )
        raster_info = {
            "source_crs": source.crs.to_string(),
            "source_resolution": [abs(source.res[0]), abs(source.res[1])],
            "source_shape": [source.height, source.width],
        }
        try:
            with rasterio.Env(GDAL_TIFF_INTERNAL_MASK=True):
                with rasterio.open(temporary_path, "w", **profile) as output:
                    output.write(destination, 1)
                    output.write_mask(valid_mask.astype(np.uint8) * 255)
                    output.set_band_description(1, source.descriptions[0])
                    output.scales = source.scales
                    output.offsets = source.offsets
                    output.update_tags(**source.tags())
                    output.update_tags(
                        harmonization_source=str(source_path),
                        harmonization_method=resampling_name,
                        target_resolution=str(grid.resolution),
                        grid_policy="coarsest included native resolution; no upsampling",
                        task_name=task_name,
                        task_id=task_id,
                    )
            temporary_path.replace(destination_path)
        finally:
            temporary_path.unlink(missing_ok=True)

    return {
        "source_path": str(source_path),
        "output_path": str(destination_path),
        "task_name": task_name,
        "task_id": task_id,
        "kind": kind,
        "resampling": resampling_name,
        "target_crs": grid.crs,
        "target_resolution": grid.resolution,
        "target_shape": [grid.height, grid.width],
        "valid_pixel_count": int(valid_mask.sum()),
        "masked_pixel_count": int(valid_mask.size - valid_mask.sum()),
        **raster_info,
    }


def _validate_task_rasters(
    task_directory: str | Path,
    task_name: str,
    task_id: str,
    raster_paths: list[str | Path],
) -> list[Path]:
    task_root = Path(task_directory).resolve()
    registry = TaskRegistry(task_root.parent)
    task, _, records = registry.validate_raster_manifest(
        task_name,
        task_root / "processed" / "standardized_manifest.json",
    )
    if task["task_directory"] != task_root or task.get("task_id") != task_id:
        raise ValueError("Selected harmonization task identity does not match the registry.")
    owned_paths = {Path(item["output_path"]).resolve() for item in records}
    selected_paths = [Path(path).resolve() for path in raster_paths]
    if any(path not in owned_paths for path in selected_paths):
        raise ValueError("Harmonization input is not listed in the selected task manifest.")
    return selected_paths


def _select_resampling(
    kind: RasterKind,
    continuous_method: str,
    *,
    target_is_coarser: bool,
) -> tuple[Resampling, str]:
    if kind in {"categorical", "quality"}:
        return Resampling.nearest, "nearest"
    if continuous_method == "auto":
        if target_is_coarser:
            return Resampling.average, "average"
        return Resampling.bilinear, "bilinear"
    methods = {
        "nearest": Resampling.nearest,
        "bilinear": Resampling.bilinear,
        "average": Resampling.average,
    }
    try:
        return methods[continuous_method], continuous_method
    except KeyError as error:
        raise ValueError(f"Unsupported continuous resampling method: {continuous_method}") from error


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Spatial harmonization helpers for use with explicit AOI and grid settings."
    )
    parser.add_argument(
        "--describe-policy",
        action="store_true",
        help="Print the configured design policy and exit; processing requires explicit grid inputs.",
    )
    args = parser.parse_args()
    if args.describe_policy:
        print(
            "Preserve native-resolution products. For combined analysis, use the "
            "coarsest included native resolution; continuous data may be averaged "
            "when downsampling and categorical/QA data use nearest-neighbor. "
            "Upsampling is rejected by default."
        )
        return 0
    parser.error(
        "No collection-wide spatial CLI is configured yet; call build_grid() and "
        "harmonize_raster() with an explicit target CRS and dataset selection."
    )
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
