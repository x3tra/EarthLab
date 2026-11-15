from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin

from nasa_agri_intelligence.standardization import (
    LayerMetadata,
    apply_task_quality_filters,
    apply_quality_filter,
    derive_slope_raster,
    product_quality_mask,
    standardize_raster,
    standardize_task,
)
from nasa_agri_intelligence.collection.manifest import write_manifest
from nasa_agri_intelligence.collection.task_registry import TaskRegistry


PRODUCT_ID = "MOD13Q1.061"
NDVI_LAYER = "_250m_16_days_NDVI"


def _write_raster(
    path: Path,
    values: np.ndarray,
    *,
    dtype: str,
    nodata: int,
    scale: float = 1.0,
    offset: float = 0.0,
) -> None:
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        width=values.shape[1],
        height=values.shape[0],
        count=1,
        dtype=dtype,
        crs="EPSG:4326",
        transform=from_origin(69, 42, 0.01, 0.01),
        nodata=nodata,
    ) as dataset:
        dataset.write(values, 1)
        dataset.scales = (scale,)
        dataset.offsets = (offset,)
        dataset.update_tags(scale_factor=str(scale), add_offset=str(offset))


def _ndvi_metadata() -> LayerMetadata:
    return LayerMetadata(
        product_id=PRODUCT_ID,
        layer_id=NDVI_LAYER,
        scale_factor=0.0001,
        add_offset=0.0,
        fill_values=(-3000,),
        raw_valid_min=-2000,
        raw_valid_max=10000,
        valid_min=-0.2,
        valid_max=1.0,
        units="NDVI",
        is_quality=False,
        quality_layers=("_250m_16_days_VI_Quality",),
        description="16 day NDVI average",
        metadata_file="MOD13Q1.json",
    )


def test_standardize_raster_scales_data_and_preserves_missing_mask(
    tmp_path: Path,
) -> None:
    source_path = tmp_path / "source.tif"
    output_path = tmp_path / "processed.tif"
    _write_raster(
        source_path,
        np.array([[4265, -3000]], dtype=np.int16),
        dtype="int16",
        nodata=-3000,
        scale=0.0001,
    )

    record = standardize_raster(source_path, output_path, _ndvi_metadata())

    with rasterio.open(output_path) as output:
        values = output.read(1, masked=True)
        assert output.dtypes == ("float32",)
        assert output.nodata is not None and np.isnan(output.nodata)
        assert values[0, 0] == pytest.approx(0.4265)
        assert np.ma.getmaskarray(values).tolist() == [[False, True]]
        assert output.scales == (1.0,)
        assert output.crs.to_epsg() == 4326
        assert output.transform == from_origin(69, 42, 0.01, 0.01)
        assert output.tags()["source_layer_id"] == NDVI_LAYER

    assert record["scale_factor_applied"] == 0.0001
    assert record["valid_pixel_count"] == 1
    assert record["masked_pixel_count"] == 1


def test_concurrent_standardization_uses_independent_temporary_files(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source_path = tmp_path / "source.tif"
    output_path = tmp_path / "processed.tif"
    _write_raster(
        source_path,
        np.array([[4265, 1200], [3000, -3000]], dtype=np.int16),
        dtype="int16",
        nodata=-3000,
        scale=0.0001,
    )
    original_open = rasterio.open
    write_paths: list[Path] = []

    def track_write_path(path: str | Path, *args: object, **kwargs: object):
        if args and args[0] == "w":
            write_paths.append(Path(path))
        return original_open(path, *args, **kwargs)

    monkeypatch.setattr(rasterio, "open", track_write_path)
    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = [
            executor.submit(
                standardize_raster,
                source_path,
                output_path,
                _ndvi_metadata(),
                overwrite=True,
            )
            for _ in range(2)
        ]
        for future in futures:
            future.result()

    assert len(write_paths) == 2
    assert len(set(write_paths)) == 2
    with original_open(output_path) as output:
        assert output.read(1, masked=True)[0, 0] == pytest.approx(0.4265)


def test_quality_layer_remains_categorical_without_scaling(
    tmp_path: Path,
) -> None:
    source_path = tmp_path / "quality.tif"
    output_path = tmp_path / "quality-standardized.tif"
    _write_raster(
        source_path,
        np.array([[2116, 65535]], dtype=np.uint16),
        dtype="uint16",
        nodata=65535,
    )
    metadata = LayerMetadata(
        product_id=PRODUCT_ID,
        layer_id="_250m_16_days_VI_Quality",
        scale_factor=None,
        add_offset=0,
        fill_values=(65535,),
        raw_valid_min=0,
        raw_valid_max=65534,
        valid_min=0,
        valid_max=65534,
        units=None,
        is_quality=True,
        quality_layers=(),
        description="VI quality bit field",
        metadata_file="MOD13Q1.json",
    )

    record = standardize_raster(source_path, output_path, metadata)

    with rasterio.open(output_path) as output:
        values = output.read(1, masked=True)
        assert output.dtypes == ("uint16",)
        assert values[0, 0] == 2116
        assert np.ma.getmaskarray(values).tolist() == [[False, True]]
    assert record["scale_factor_applied"] is None
    assert record["quality_filter_applied"] is False


def test_aster_scene_count_is_auxiliary_not_a_quality_filter(tmp_path: Path) -> None:
    metadata = LayerMetadata.from_appeears(
        "ASTGTM_NC.003",
        "ASTER_GDEM_DEM",
        {
            "QualityLayers": "['ASTER_GDEM_NUM']",
            "FillValueAll": [-9999],
            "OrigValidMin": -1000,
            "OrigValidMax": 10000,
            "ValidMin": -1000,
            "ValidMax": 10000,
            "Units": "m",
            "IsQA": False,
        },
        tmp_path / "ASTER.json",
    )

    assert metadata.quality_layers == ()
    assert metadata.auxiliary_layers == ("ASTER_GDEM_NUM",)


def test_apply_quality_filter_requires_aligned_explicit_accepted_values() -> None:
    values = np.ma.array([[0.4, 0.5]], mask=[[False, True]])
    quality = np.array([[2116, 2116]])

    filtered = apply_quality_filter(values, quality, accepted_values={2116})

    assert filtered[0, 0] == pytest.approx(0.4)
    assert np.ma.getmaskarray(filtered).tolist() == [[False, True]]
    with pytest.raises(ValueError, match="must align"):
        apply_quality_filter(values, quality[:, :1], accepted_values={2116})
    with pytest.raises(ValueError, match="at least one"):
        apply_quality_filter(values, quality, accepted_values=set())


@pytest.mark.parametrize(
    ("product_id", "layer_id", "quality", "expected"),
    [
        ("MOD13Q1.061", "_250m_16_days_VI_Quality", [2116, 2185], [True, False]),
        ("MOD11A2.061", "QC_Day", [0, 4, 16, 64], [True, False, True, True]),
        ("MOD16A2GF.061", "ET_QC_500m", [0, 16, 32, 157], [True, True, True, False]),
        ("MCD12Q1.061", "QC", [0, 2, 1, 10], [True, True, False, False]),
    ],
)
def test_product_quality_masks_follow_documented_flag_fields(
    product_id: str,
    layer_id: str,
    quality: list[int],
    expected: list[bool],
) -> None:
    assert product_quality_mask(product_id, layer_id, np.array(quality)).tolist() == expected


def test_task_quality_filter_matches_same_observation_and_preserves_original(
    tmp_path: Path,
) -> None:
    science_path = tmp_path / "processed/rasters/science_MOD13Q1.061_NDVI_20250626T000000_aid0001.tif"
    quality_path = tmp_path / "processed/rasters/quality_MOD13Q1.061_VI_Quality_20250626T000000_aid0001.tif"
    science_path.parent.mkdir(parents=True)
    _write_raster(
        science_path,
        np.array([[0.5, 0.6]], dtype=np.float32),
        dtype="float32",
        nodata=-9999,
    )
    _write_raster(
        quality_path,
        np.array([[2116, 2185]], dtype=np.uint16),
        dtype="uint16",
        nodata=65535,
    )
    records = [
        {
            "product_id": "MOD13Q1.061",
            "layer_id": "_250m_16_days_NDVI",
            "variable_kind": "continuous",
            "quality_layers": ["_250m_16_days_VI_Quality"],
            "quality_filter_applied": False,
            "output_path": science_path.relative_to(tmp_path).as_posix(),
        },
        {
            "product_id": "MOD13Q1.061",
            "layer_id": "_250m_16_days_VI_Quality",
            "variable_kind": "quality",
            "quality_layers": [],
            "output_path": quality_path.relative_to(tmp_path).as_posix(),
        },
    ]

    filtered_records = apply_task_quality_filters(tmp_path, records)

    filtered_record = filtered_records[0]
    assert filtered_record["quality_filter_applied"] is True
    assert filtered_record["quality_status"] == "filtered"
    assert filtered_record["unfiltered_output_path"] == records[0]["output_path"]
    with rasterio.open(tmp_path / filtered_record["output_path"]) as filtered:
        assert filtered.read(1, masked=True).tolist() == [[0.5, None]]
        assert filtered.tags()["quality_filter_applied"] == "true"
    with rasterio.open(science_path) as original:
        assert original.read(1).tolist()[0] == pytest.approx([0.5, 0.6])
    assert apply_task_quality_filters(tmp_path, filtered_records)[0]["output_path"] == (
        filtered_record["output_path"]
    )


def test_task_quality_filter_reports_missing_companion_without_masking(
    tmp_path: Path,
) -> None:
    science_path = tmp_path / "science_MOD11A2.061_LST_Day_1km_20250626T000000_aid0001.tif"
    _write_raster(
        science_path,
        np.array([[300.0]], dtype=np.float32),
        dtype="float32",
        nodata=-9999,
    )
    records = [
        {
            "product_id": "MOD11A2.061",
            "layer_id": "LST_Day_1km",
            "variable_kind": "continuous",
            "quality_layers": ["QC_Day"],
            "quality_filter_applied": False,
            "output_path": science_path.relative_to(tmp_path).as_posix(),
        },
    ]

    result = apply_task_quality_filters(tmp_path, records)[0]

    assert result["quality_filter_applied"] is False
    assert result["quality_status"] == "missing_quality_companion"
    assert result["quality_missing_layers"] == ["QC_Day"]
    assert result["output_path"] == records[0]["output_path"]


def test_task_quality_filter_rejects_misaligned_companion(tmp_path: Path) -> None:
    science_path = tmp_path / "science_MOD13Q1.061_NDVI_20250626T000000_aid0001.tif"
    quality_path = tmp_path / "quality_MOD13Q1.061_VI_Quality_20250626T000000_aid0001.tif"
    _write_raster(
        science_path,
        np.array([[0.5]], dtype=np.float32),
        dtype="float32",
        nodata=-9999,
    )
    _write_raster(
        quality_path,
        np.array([[2116]], dtype=np.uint16),
        dtype="uint16",
        nodata=65535,
    )
    with rasterio.open(quality_path, "r+") as dataset:
        dataset.transform = from_origin(69.1, 42, 0.01, 0.01)
    records = [
        {
            "product_id": "MOD13Q1.061",
            "layer_id": "_250m_16_days_NDVI",
            "variable_kind": "continuous",
            "quality_layers": ["_250m_16_days_VI_Quality"],
            "output_path": science_path.relative_to(tmp_path).as_posix(),
        },
        {
            "product_id": "MOD13Q1.061",
            "layer_id": "_250m_16_days_VI_Quality",
            "variable_kind": "quality",
            "output_path": quality_path.relative_to(tmp_path).as_posix(),
        },
    ]

    with pytest.raises(ValueError, match="do not share an identical grid"):
        apply_task_quality_filters(tmp_path, records)


def test_standardize_rejects_conflicting_scale_tags(tmp_path: Path) -> None:
    source_path = tmp_path / "source.tif"
    _write_raster(
        source_path,
        np.array([[4265]], dtype=np.int16),
        dtype="int16",
        nodata=-3000,
        scale=0.0002,
    )

    with pytest.raises(ValueError, match="does not match product metadata"):
        standardize_raster(source_path, tmp_path / "processed.tif", _ndvi_metadata())


def test_derive_slope_uses_projected_dem_units_and_preserves_task_grid(
    tmp_path: Path,
) -> None:
    dem_path = tmp_path / "dem.tif"
    slope_path = tmp_path / "slope.tif"
    transform = from_origin(500_000, 4_600_000, 30, 30)
    rows, columns = np.indices((3, 3))
    elevation = (3 * columns * 30 - 4 * rows * 30).astype(np.float32)
    with rasterio.open(
        dem_path,
        "w",
        driver="GTiff",
        width=3,
        height=3,
        count=1,
        dtype="float32",
        crs="EPSG:32642",
        transform=transform,
        nodata=np.nan,
    ) as dataset:
        dataset.write(elevation, 1)

    record = derive_slope_raster(
        dem_path,
        slope_path,
        source_product_id="ASTGTM_NC.003",
        source_layer_id="ASTER_GDEM_DEM",
    )

    assert record is not None
    assert record["product_id"] == "_DERIVED"
    assert record["layer_id"] == "slope"
    assert record["derived"] is True
    with rasterio.open(slope_path) as dataset:
        assert dataset.crs.to_epsg() == 32642
        assert dataset.transform == transform
        assert dataset.units == ("degree",)
        assert dataset.read(1)[1, 1] == pytest.approx(np.degrees(np.arctan(5)))


def test_task_standardization_keeps_external_quality_companion_and_derives_only_dem_slope(
    tmp_path: Path,
) -> None:
    task_name = "aster-task"
    task_id = "aster-task-id"
    product_id = "ASTGTM_NC.003"
    quality_product_id = "ASTGTM_NUMNC.003"
    dem_layer = "ASTER_GDEM_DEM"
    quality_layer = "ASTER_GDEM_NUM"
    registry = TaskRegistry(tmp_path / "data/tasks")
    task_root = registry.reserve(
        task_name,
        {"task_id": task_id, "task_directory": str(tmp_path / "data/tasks" / task_name)},
    )
    raw_directory = task_root / "raw/ASTER/GDEM"
    raw_directory.mkdir(parents=True)
    dem_source = raw_directory / f"{product_id}_{dem_layer}.tif"
    quality_source = raw_directory / f"{quality_product_id}_{quality_layer}.tif"
    _write_raster(
        dem_source,
        np.array([[0, 30, 60], [40, 70, 100], [80, 110, 140]], dtype=np.int16),
        dtype="int16",
        nodata=-9999,
    )
    _write_raster(
        quality_source,
        np.zeros((3, 3), dtype=np.uint8),
        dtype="uint8",
        nodata=255,
    )
    with rasterio.open(dem_source, "r+") as dataset:
        dataset.update_tags(
            source_product_id=product_id,
            source_layer_id=dem_layer,
        )
    with rasterio.open(quality_source, "r+") as dataset:
        dataset.update_tags(
            source_product_id=quality_product_id,
            source_layer_id=quality_layer,
        )

    metadata_directory = tmp_path / "data/metadata"
    metadata_directory.mkdir(parents=True)
    write_manifest(
        metadata_directory / "ASTER.json",
        {
            "available_products": [
                {
                    "catalog": {"ProductAndVersion": product_id},
                    "layers": {
                        dem_layer: {
                            "FillValue": -9999,
                            "IsQA": False,
                            "Units": "Meters",
                            "QualityLayers": f"['{quality_layer}']",
                            "QualityProductAndVersion": quality_product_id,
                        }
                    },
                }
            ]
        },
    )
    task_manifest = {
        "task_name": task_name,
        "task_id": task_id,
        "product_id": product_id,
        "layers": [dem_layer],
        "download_status": "completed",
        "downloads": [
            {"local_path": str(dem_source), "status": "validated"},
            {"local_path": str(quality_source), "status": "validated"},
        ],
    }
    write_manifest(task_root / "task_manifest.json", task_manifest)

    result = standardize_task(
        task_root,
        task_name=task_name,
        task_id=task_id,
        metadata_directory=metadata_directory,
    )

    quality_records = [
        record for record in result["rasters"] if record["variable_kind"] == "quality"
    ]
    slope_records = [
        record for record in result["rasters"] if record["layer_id"] == "slope"
    ]
    assert len(quality_records) == 1
    assert quality_records[0]["product_id"] == quality_product_id
    assert quality_records[0]["metadata_status"].startswith(
        "companion_product_metadata_unavailable"
    )
    assert len(slope_records) == 1
    assert slope_records[0]["source_path"] == dem_source.relative_to(task_root).as_posix()
