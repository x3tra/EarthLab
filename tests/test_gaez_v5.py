from __future__ import annotations

from datetime import datetime, timezone
import re
from pathlib import Path

from fastapi.testclient import TestClient
import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin
from shapely.geometry import box

from nasa_agri_intelligence.collection import gaez_v5
from nasa_agri_intelligence.collection.manifest import write_manifest
from nasa_agri_intelligence.collection.task_registry import TaskRegistry
from nasa_agri_intelligence.web import create_app

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CATALOG_PATH = PROJECT_ROOT / "data/reference/gaez_v5_crops.json"
RESOLUTION = 1 / 120
AOI = {
    "type": "FeatureCollection",
    "features": [
        {
            "type": "Feature",
            "properties": {},
            "geometry": {
                "type": "Polygon",
                "coordinates": [[
                    [67.0, 40 - 2 * RESOLUTION],
                    [67.0 + 2 * RESOLUTION, 40 - 2 * RESOLUTION],
                    [67.0 + 2 * RESOLUTION, 40],
                    [67.0, 40],
                    [67.0, 40 - 2 * RESOLUTION],
                ]],
            },
        }
    ],
}


def _create_task(
    tasks_directory: Path,
    name: str,
    task_id: str,
    *,
    aoi: dict[str, object] = AOI,
) -> tuple[TaskRegistry, Path]:
    registry = TaskRegistry(tasks_directory)
    task_root = registry.reserve(
        name,
        {
            "task_id": task_id,
            "task_type": "aoi_analysis",
            "task_status": "completed",
            "task_directory": str(tasks_directory / name),
        },
    )
    now = datetime.now(timezone.utc).isoformat()
    write_manifest(
        task_root / "task_manifest.json",
        {
            "task_name": name,
            "task_id": task_id,
            "task_type": "aoi_analysis",
            "aoi": {"name": name, "crs": "EPSG:4326", "geojson": aoi},
            "start_date": "2025-01-01",
            "end_date": "2025-01-02",
            "acquisition": {
                "status": "completed",
                "stage": "completed",
                "providers": {"gaez_v5": {"status": "not_selected"}},
            },
            "analysis_runs": [],
            "gaez_runs": [],
            "updated_at": now,
        },
    )
    write_manifest(
        task_root / "task_metadata.json",
        {"task_name": name, "task_id": task_id},
    )
    write_manifest(
        task_root / "processed/standardized_manifest.json",
        {
            "task_name": name,
            "task_id": task_id,
            "created_at": now,
            "source_directory": "raw",
            "rasters": [],
            "records": [],
        },
    )
    return registry, task_root


def _write_source(
    path: Path,
    values: np.ndarray,
    *,
    nodata: int,
    origin: tuple[float, float] = (67.0, 40.0),
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=values.shape[0],
        width=values.shape[1],
        count=1,
        dtype="int16",
        crs="EPSG:4326",
        transform=from_origin(*origin, RESOLUTION, RESOLUTION),
        nodata=nodata,
    ) as output:
        output.write(values.astype(np.int16), 1)


def _source_opener(
    root: Path,
    index_values: dict[tuple[str, str], np.ndarray] | None = None,
    class_values: dict[tuple[str, str], np.ndarray] | None = None,
    *,
    origin: tuple[float, float] = (67.0, 40.0),
):
    arrays = index_values or {}
    classes = class_values or {}

    def open_source(url: str):
        match = re.search(        r"\.HIST\.([A-Z0-9]+)\.([A-Z0-9]+)\.tif", url)
        assert match is not None, url
        crop_code, management_code = match.groups()
        if "RES05-SXX30AS" in url:
            path = root / f"SXX-{crop_code}-{management_code}.tif"
            values = arrays.get(
                (crop_code, management_code),
                np.full((2, 2), 5000, dtype=np.int16),
            )
            nodata = -9
        else:
            path = root / f"SIX-{crop_code}-{management_code}.tif"
            values = classes.get(
                (crop_code, management_code),
                np.array([[2, 3], [4, 5]], dtype=np.int16),
            )
            nodata = 0
        if not path.exists():
            _write_source(path, values, nodata=nodata, origin=origin)
        return rasterio.open(path)

    return open_source


def test_gaez_catalog_uses_verified_products_crops_and_scenarios() -> None:
    catalog = gaez_v5.load_gaez_catalog(CATALOG_PATH)

    assert catalog["continuous_product"]["mapset"] == "RES05-SXX30AS"
    assert catalog["class_product"]["mapset"] == "RES05-SIX30AS"
    assert catalog["period_code"] == "HP0120"
    assert catalog["crs"] == "EPSG:4326"
    assert catalog["index_nodata"] == -9
    assert catalog["class_nodata"] == 0
    assert catalog["index_valid_range"] == [0, 10000]
    assert catalog["index_scale_factor"] == 1.0
    assert catalog["class_valid_range"] == [1, 9]
    assert catalog["class_scale_factor"] == 1.0
    assert len(catalog["crops"]) == 77
    assert len(catalog["unavailable_crops"]) == 12
    assert {crop["code"] for crop in catalog["crops"]} >= {
        "COT", "WHE", "MZE", "BRL", "SFL", "RCD", "RCW", "CSV", "BAN"
    }
    assert gaez_v5.WATER_REGIMES == {
        "rainfed": {"high": "HRLM", "low": "LRLM"},
        "irrigated": {"high": "HILM", "low": "LILM"},
    }
    wheat = next(crop for crop in catalog["crops"] if crop["code"] == "WHE")
    assert wheat["project_crop_id"] == "wheat"
    assert wheat["availability_status"] == "available"
    assert wheat["available_scenario_codes"] == ["HRLM", "HILM", "LRLM", "LILM"]
    avocado = next(crop for crop in catalog["crops"] if crop["code"] == "AVOC")
    assert avocado["availability_status"] == "partially_available"
    assert avocado["available_scenario_codes"] == ["HRLM", "HILM"]
    assert all(
        crop["availability_status"] == "unavailable"
        for crop in catalog["unavailable_crops"]
    )


def test_small_aoi_window_includes_overlapping_source_pixel(tmp_path: Path) -> None:
    source_path = tmp_path / "source.tif"
    _write_source(source_path, np.full((2, 2), 5000, dtype=np.int16), nodata=-9)

    with rasterio.open(source_path) as source:
        window = gaez_v5._aoi_window(source, box(67.001, 39.998, 67.002, 39.999))

    assert (window.col_off, window.row_off, window.width, window.height) == (0, 0, 1, 1)


@pytest.mark.parametrize(
    ("west", "north"),
    [(-46.0, -23.0), (37.0, 1.0), (2.0, 48.0), (151.0, -33.0)],
)
def test_gaez_processes_aois_across_continents(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    west: float,
    north: float,
) -> None:
    ring = [
        [west, north - 2 * RESOLUTION],
        [west + 2 * RESOLUTION, north - 2 * RESOLUTION],
        [west + 2 * RESOLUTION, north],
        [west, north],
        [west, north - 2 * RESOLUTION],
    ]
    aoi = {
        "type": "FeatureCollection",
        "features": [{"type": "Feature", "properties": {}, "geometry": {
            "type": "Polygon", "coordinates": [ring],
        }}],
    }
    registry, _ = _create_task(
        tmp_path / "tasks", "global-aoi", "task-global", aoi=aoi
    )
    monkeypatch.setattr(
        gaez_v5,
        "_open_remote_raster",
        _source_opener(tmp_path / "sources", origin=(west, north)),
    )

    result = gaez_v5.run_gaez_v5_suitability(
        registry,
        "task-global",
        run_id=gaez_v5.make_gaez_run_id(["WHE"], "rainfed", "high"),
        crop_codes=["WHE"],
        water_regime="rainfed",
        management_level="high",
        catalog_path=CATALOG_PATH,
    )

    crop = result["scenarios"][0]["ranked_crops"][0]
    assert result["status"] == "completed"
    assert crop["valid_data_coverage_of_aoi"] == pytest.approx(1.0)
    assert crop["ranking_eligible"] is True


def test_gaez_partial_raster_boundary_is_reported_as_missing_coverage(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    west = -RESOLUTION
    east = 3 * RESOLUTION
    north = 0.0
    south = -2 * RESOLUTION
    ring = [[west, south], [east, south], [east, north], [west, north], [west, south]]
    aoi = {
        "type": "FeatureCollection",
        "features": [{"type": "Feature", "properties": {}, "geometry": {
            "type": "Polygon", "coordinates": [ring],
        }}],
    }
    registry, _ = _create_task(
        tmp_path / "tasks", "boundary-aoi", "task-boundary", aoi=aoi
    )
    monkeypatch.setattr(
        gaez_v5,
        "_open_remote_raster",
        _source_opener(tmp_path / "sources", origin=(0.0, 0.0)),
    )

    result = gaez_v5.run_gaez_v5_suitability(
        registry,
        "task-boundary",
        run_id=gaez_v5.make_gaez_run_id(["WHE"], "rainfed", "high"),
        crop_codes=["WHE"],
        water_regime="rainfed",
        management_level="high",
        catalog_path=CATALOG_PATH,
    )

    crop = result["scenarios"][0]["unranked_crops"][0]
    assert crop["ranking_eligible"] is False
    assert crop["valid_data_coverage_of_aoi"] < 0.95
    assert crop["ranking_exclusion_reason"]


def test_gaez_missing_scenario_product_is_not_opened_or_ranked(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    registry, _ = _create_task(tmp_path / "tasks", "partial-crop", "task-partial")

    def unexpected_source_open(url: str):
        raise AssertionError(f"Unavailable scenario should not open a source: {url}")

    monkeypatch.setattr(gaez_v5, "_open_remote_raster", unexpected_source_open)

    result = gaez_v5.run_gaez_v5_suitability(
        registry,
        "task-partial",
        run_id=gaez_v5.make_gaez_run_id(["AVOC"], "both", "low"),
        crop_codes=["AVOC"],
        water_regime="both",
        management_level="low",
        catalog_path=CATALOG_PATH,
    )

    assert result["status"] == "failed"
    assert result["scenarios"][0]["ranked_crops"] == []
    assert len(result["failures"]) == 2
    assert all(item["failure_type"] == "source_product_missing" for item in result["failures"])


def test_equal_gaez_scores_share_a_competition_rank(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    registry, _ = _create_task(tmp_path / "tasks", "ties", "task-ties")
    monkeypatch.setattr(gaez_v5, "_open_remote_raster", _source_opener(tmp_path / "sources"))

    result = gaez_v5.run_gaez_v5_suitability(
        registry,
        "task-ties",
        run_id=gaez_v5.make_gaez_run_id(["MZE", "WHE"], "rainfed", "high"),
        crop_codes=["MZE", "WHE"],
        water_regime="rainfed",
        management_level="high",
        catalog_path=CATALOG_PATH,
    )

    rankings = result["scenarios"][0]["ranked_crops"]
    assert [item["crop_name"] for item in rankings] == ["Maize", "Wheat"]
    assert [item["rank"] for item in rankings] == [1, 1]


def test_partial_boundary_cell_weights_sum_to_aoi_area() -> None:
    geometry = box(67.002, 39.986, 67.011, 39.996)
    transform = from_origin(67.0, 40.0, RESOLUTION, RESOLUTION)
    transformer = gaez_v5.Transformer.from_crs(
        "EPSG:4326", "EPSG:6933", always_xy=True
    )
    weights = gaez_v5._cell_area_weights(
        geometry,
        (2, 2),
        transform,
        transformer,
    )
    projected = gaez_v5.transform_geometry(
        geometry,
        transformer.transform,
        interleaved=False,
    )

    assert int((weights > 0).sum()) == 4
    assert weights.sum() == pytest.approx(projected.area, rel=1e-10)


def test_gaez_catalog_rejects_unverified_product_configuration(tmp_path: Path) -> None:
    catalog = gaez_v5.load_gaez_catalog(CATALOG_PATH)
    catalog["continuous_product"]["mapset"] = "RES05-SIX30AS"
    path = tmp_path / "invalid.json"
    path.write_text(__import__("json").dumps(catalog), encoding="utf-8")

    with pytest.raises(ValueError, match="source metadata is inconsistent"):
        gaez_v5.load_gaez_catalog(path)


def test_gaez_source_urls_keep_continuous_and_class_products_distinct() -> None:
    catalog = gaez_v5.load_gaez_catalog(CATALOG_PATH)

    index_url = gaez_v5._source_url("SXX", "WHE", "HRLM", catalog)
    class_url = gaez_v5._source_url("SIX", "WHE", "HRLM", catalog)

    assert "RES05-SXX30AS" in index_url
    assert "RES05-SIX30AS" in class_url
    assert "HP0120.AGERA5.HIST.WHE.HRLM.tif" in index_url
    assert index_url != class_url


def test_gaez_area_weighting_nodata_and_class_distribution(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    registry, task_root = _create_task(tmp_path / "tasks", "weighting-task", "task-weighting")
    values = np.array([[1000, 3000], [5000, -9]], dtype=np.int16)
    monkeypatch.setattr(gaez_v5, "_open_remote_raster", _source_opener(tmp_path / "sources", {("WHE", "HRLM"): values}))

    result = gaez_v5.run_gaez_v5_suitability(
        registry,
        "task-weighting",
        run_id=gaez_v5.make_gaez_run_id(["WHE"], "rainfed", "high"),
        crop_codes=["WHE"],
        water_regime="rainfed",
        management_level="high",
        catalog_path=CATALOG_PATH,
    )

    crop = result["scenarios"][0]["unranked_crops"][0]
    assert result["status"] == "completed"
    assert crop["valid_data_coverage"] == pytest.approx(0.75, abs=0.01)
    assert crop["valid_data_coverage_of_aoi"] == pytest.approx(0.75, abs=0.01)
    assert crop["intersecting_cell_count"] == 4
    assert crop["valid_index_cell_count"] == 3
    assert crop["ranking_eligible"] is False
    assert crop["mean_suitability_index"] == pytest.approx(3000, abs=20)
    assert crop["ranking_exclusion_reason"]
    assert sum(item["area_km2"] for item in crop["class_distribution"]) > 0
    assert sum(
        item["share_of_classified_area"] or 0
        for item in crop["class_distribution"]
    ) == pytest.approx(1.0)
    assert sum(item["share_of_aoi"] for item in crop["class_distribution"]) == pytest.approx(1.0)
    assert crop["class_data_available"] is True
    assert crop["valid_class_cell_count"] == 4
    with rasterio.open(task_root / crop["continuous_raster"]) as raster:
        assert raster.crs.to_string() == "EPSG:4326"
        assert raster.nodata == -9
        assert raster.res == pytest.approx((RESOLUTION, RESOLUTION))
        assert raster.width <= 2 and raster.height <= 2


def test_gaez_rankings_keep_water_regimes_and_management_separate(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    registry, task_root = _create_task(tmp_path / "tasks", "scenario-task", "task-scenario")
    values = {
        ("WHE", "HRLM"): np.full((2, 2), 7000),
        ("COT", "HRLM"): np.full((2, 2), 3000),
        ("WHE", "HILM"): np.full((2, 2), 2000),
        ("COT", "HILM"): np.full((2, 2), 8000),
        ("WHE", "LRLM"): np.full((2, 2), 1000),
        ("COT", "LRLM"): np.full((2, 2), 9000),
        ("WHE", "LILM"): np.full((2, 2), 6000),
        ("COT", "LILM"): np.full((2, 2), 4000),
    }
    monkeypatch.setattr(gaez_v5, "_open_remote_raster", _source_opener(tmp_path / "sources", values))
    high = gaez_v5.run_gaez_v5_suitability(
        registry,
        "task-scenario",
        run_id=gaez_v5.make_gaez_run_id(["WHE", "COT"], "both", "high"),
        crop_codes=["WHE", "COT"],
        water_regime="both",
        management_level="high",
        catalog_path=CATALOG_PATH,
    )
    low = gaez_v5.run_gaez_v5_suitability(
        registry,
        "task-scenario",
        run_id=gaez_v5.make_gaez_run_id(["WHE", "COT"], "both", "low"),
        crop_codes=["WHE", "COT"],
        water_regime="both",
        management_level="low",
        catalog_path=CATALOG_PATH,
    )

    assert high["scenarios"][0]["ranked_crops"][0]["crop_code"] == "WHE"
    assert high["scenarios"][1]["ranked_crops"][0]["crop_code"] == "COT"
    assert [scenario["water_regime"] for scenario in high["scenarios"]] == [
        "rainfed",
        "irrigated",
    ]
    assert {scenario["management_level"] for scenario in low["scenarios"]} == {"low"}
    assert all(
        scenario["product_identifiers"] == {
            "continuous_index": "RES05-SXX30AS",
            "official_classes": "RES05-SIX30AS",
        }
        for scenario in high["scenarios"]
    )
    assert {
        scenario["water_regime"]: scenario["management_code"]
        for scenario in high["scenarios"]
    } == {"rainfed": "HRLM", "irrigated": "HILM"}
    assert len(registry.validate_raster_manifest(
        "scenario-task",
        task_root / "processed/standardized_manifest.json",
    )[2]) == 16


def test_new_gaez_run_updates_latest_layer_paths_and_retains_prior_run(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    registry, task_root = _create_task(tmp_path / "tasks", "rerun-task", "task-rerun")
    monkeypatch.setattr(gaez_v5, "_open_remote_raster", _source_opener(tmp_path / "sources"))
    crop_codes = ["WHE"]
    first = gaez_v5.run_gaez_v5_suitability(
        registry,
        "task-rerun",
        run_id=gaez_v5.make_gaez_run_id(crop_codes, "rainfed", "high"),
        crop_codes=crop_codes,
        water_regime="rainfed",
        management_level="high",
        catalog_path=CATALOG_PATH,
    )
    second_run_id = "GAEZ_0123456789abcdef"
    second = gaez_v5.run_gaez_v5_suitability(
        registry,
        "task-rerun",
        run_id=second_run_id,
        crop_codes=crop_codes,
        water_regime="rainfed",
        management_level="high",
        catalog_path=CATALOG_PATH,
    )

    layer_record = next(
        record
        for record in registry.validate_raster_manifest(
            "task-rerun",
            task_root / "processed/standardized_manifest.json",
        )[2]
        if record["layer_id"] == "SXX_WHE_rainfed_high"
    )
    old_raster = task_root / first["scenarios"][0]["ranked_crops"][0]["continuous_raster"]
    new_raster = task_root / second["scenarios"][0]["ranked_crops"][0]["continuous_raster"]
    assert old_raster.is_file() and new_raster.is_file()
    assert old_raster != new_raster
    assert Path(layer_record["output_path"]).resolve() == new_raster.resolve()
    assert len(registry.load_task_manifest("task-rerun")[2]["gaez_runs"]) == 2


def test_gaez_source_failure_is_reported_without_creating_false_rankings(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    registry, _ = _create_task(tmp_path / "tasks", "failed-task", "task-failed")

    def unavailable(_: str):
        raise OSError("GAEZ test source unavailable")

    monkeypatch.setattr(gaez_v5, "_open_remote_raster", unavailable)
    result = gaez_v5.run_gaez_v5_suitability(
        registry,
        "task-failed",
        run_id=gaez_v5.make_gaez_run_id(["WHE"], "rainfed", "high"),
        crop_codes=["WHE"],
        water_regime="rainfed",
        management_level="high",
        catalog_path=CATALOG_PATH,
    )

    assert result["status"] == "failed"
    assert result["scenarios"][0]["ranked_crops"] == []
    assert result["failures"][0]["crop_code"] == "WHE"
    assert "GAEZ test source unavailable" in result["failures"][0]["error"]
    assert result["failures"][0]["failure_type"] == "source_download_error"


def test_official_water_class_excludes_water_sentinel_from_index_mean(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    registry, _ = _create_task(tmp_path / "tasks", "water-task", "task-water")
    index_values = np.array([[1000, -1], [3000, -9]], dtype=np.int16)
    class_values = np.array([[3, 9], [4, 0]], dtype=np.int16)
    monkeypatch.setattr(
        gaez_v5,
        "_open_remote_raster",
        _source_opener(
            tmp_path / "sources",
            {("WHE", "HRLM"): index_values},
            {("WHE", "HRLM"): class_values},
        ),
    )

    result = gaez_v5.run_gaez_v5_suitability(
        registry,
        "task-water",
        run_id=gaez_v5.make_gaez_run_id(["WHE"], "rainfed", "high"),
        crop_codes=["WHE"],
        water_regime="rainfed",
        management_level="high",
        catalog_path=CATALOG_PATH,
    )

    crop = result["scenarios"][0]["unranked_crops"][0]
    water_class = next(item for item in crop["class_distribution"] if item["class_code"] == 9)
    assert result["status"] == "completed"
    assert crop["mean_suitability_index"] == pytest.approx(2000, abs=1)
    assert crop["intersecting_cell_count"] == 4
    assert crop["valid_index_cell_count"] == 2
    assert crop["water_cell_count"] == 1
    assert crop["index_nodata_cell_count"] == 1
    assert crop["valid_data_coverage"] == pytest.approx(2 / 3, abs=0.01)
    assert crop["valid_data_coverage_of_aoi"] == pytest.approx(0.5, abs=0.01)
    assert water_class["share_of_aoi"] == pytest.approx(0.25, abs=0.01)


def test_missing_six_keeps_index_ranking_and_reports_classes_unavailable(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    registry, _ = _create_task(tmp_path / "tasks", "missing-six", "task-missing-six")
    source_opener = _source_opener(tmp_path / "sources")

    def open_index_only(url: str):
        if "RES05-SIX30AS" in url:
            raise OSError("HTTP 404: class product not found")
        return source_opener(url)

    monkeypatch.setattr(gaez_v5, "_open_remote_raster", open_index_only)
    result = gaez_v5.run_gaez_v5_suitability(
        registry,
        "task-missing-six",
        run_id=gaez_v5.make_gaez_run_id(["WHE"], "rainfed", "high"),
        crop_codes=["WHE"],
        water_regime="rainfed",
        management_level="high",
        catalog_path=CATALOG_PATH,
    )

    crop = result["scenarios"][0]["ranked_crops"][0]
    assert result["status"] == "partial"
    assert crop["class_data_available"] is False
    assert crop["class_data_status"] == "unavailable"
    assert crop["class_distribution"] == []
    assert crop["class_raster"] is None
    assert result["failures"][0]["failure_type"] == "source_product_missing"
    assert result["failures"][0]["product"] == "SIX"


def test_unpaired_negative_index_value_is_not_silently_treated_as_nodata(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    registry, _ = _create_task(tmp_path / "tasks", "bad-index", "task-bad-index")
    values = np.array([[1000, -1], [3000, 4000]], dtype=np.int16)
    monkeypatch.setattr(
        gaez_v5,
        "_open_remote_raster",
        _source_opener(tmp_path / "sources", {("WHE", "HRLM"): values}),
    )

    result = gaez_v5.run_gaez_v5_suitability(
        registry,
        "task-bad-index",
        run_id=gaez_v5.make_gaez_run_id(["WHE"], "rainfed", "high"),
        crop_codes=["WHE"],
        water_regime="rainfed",
        management_level="high",
        catalog_path=CATALOG_PATH,
    )

    assert result["status"] == "failed"
    assert result["scenarios"][0]["ranked_crops"] == []
    assert result["failures"][0]["failure_type"] == "invalid_source_values"
    assert "Unexpected values: [-1]" in result["failures"][0]["error"]


def test_gaez_runs_are_isolated_between_aoi_tasks(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    registry, first_root = _create_task(tmp_path / "tasks", "first-aoi", "task-first")
    _create_task(tmp_path / "tasks", "second-aoi", "task-second")
    monkeypatch.setattr(gaez_v5, "_open_remote_raster", _source_opener(tmp_path / "sources"))
    run_id = gaez_v5.make_gaez_run_id(["WHE"], "rainfed", "high")

    first = gaez_v5.run_gaez_v5_suitability(
        registry,
        "task-first",
        run_id=run_id,
        crop_codes=["WHE"],
        water_regime="rainfed",
        management_level="high",
        catalog_path=CATALOG_PATH,
    )
    gaez_v5.run_gaez_v5_suitability(
        registry,
        "task-second",
        run_id=run_id,
        crop_codes=["WHE"],
        water_regime="rainfed",
        management_level="high",
        catalog_path=CATALOG_PATH,
    )

    stored = gaez_v5.load_gaez_run(registry, "task-first", run_id)
    assert stored["task_id"] == "task-first"
    assert stored["scenarios"][0]["ranked_crops"][0]["mean_suitability_index"] == (
        first["scenarios"][0]["ranked_crops"][0]["mean_suitability_index"]
    )
    first_raster = Path(first_root / stored["scenarios"][0]["ranked_crops"][0]["continuous_raster"])
    second_raster = Path(
        registry.resolve("second-aoi")["task_directory"]
        / stored["scenarios"][0]["ranked_crops"][0]["continuous_raster"]
    )
    assert first_raster.is_file() and second_raster.is_file()
    assert first_raster != second_raster


def test_legacy_gaez_run_is_marked_outdated_without_mutating_its_manifest(
    tmp_path: Path,
) -> None:
    registry, task_root = _create_task(tmp_path / "tasks", "legacy-result", "task-legacy")
    run_id = "GAEZ_0123456789abcdef"
    manifest_path = task_root / "suitability/gaez" / run_id / "gaez_manifest.json"
    legacy = {
        "task_name": "legacy-result",
        "task_id": "task-legacy",
        "run_id": run_id,
        "status": "partial",
        "scenarios": [],
    }
    write_manifest(manifest_path, legacy)
    _, task_manifest_path, task_manifest = registry.load_task_manifest("legacy-result")
    task_manifest["gaez_runs"] = [
        {
            "run_id": run_id,
            "status": "partial",
            "manifest_path": str(manifest_path),
        }
    ]
    write_manifest(task_manifest_path, task_manifest)

    loaded = gaez_v5.load_gaez_run(registry, "legacy-result")

    assert loaded["analysis_outdated"] is True
    persisted = gaez_v5.load_manifest(manifest_path)
    assert "analysis_outdated" not in persisted
    assert "analysis_version" not in persisted


def test_gaez_api_returns_task_scoped_results_and_map_layers(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    app = create_app(project_root=PROJECT_ROOT)
    app.state.tasks_directory = tmp_path / "tasks"
    registry, _ = _create_task(app.state.tasks_directory, "api-aoi", "task-api")
    monkeypatch.setattr(gaez_v5, "_open_remote_raster", _source_opener(tmp_path / "sources"))

    with TestClient(app) as client:
        catalog_response = client.get("/api/gaez-v5/catalog")
        start_response = client.post(
            "/api/aoi-tasks/task-api/gaez-suitability",
            json={
                "crop_codes": ["WHE"],
                "water_regime": "both",
                "management_level": "high",
            },
        )
        run_id = start_response.json()["run_id"]
        result_response = client.get(
            f"/api/aoi-tasks/task-api/gaez-suitability?run_id={run_id}"
        )
        layers_response = client.get("/api/layers", params={"task_name": "api-aoi"})

    assert catalog_response.status_code == 200
    assert catalog_response.json()["continuous_product"]["mapset"] == "RES05-SXX30AS"
    assert len(catalog_response.json()["crops"]) == 77
    assert len(catalog_response.json()["unavailable_crops"]) == 12
    catalog_wheat = next(
        crop for crop in catalog_response.json()["crops"] if crop["code"] == "WHE"
    )
    assert catalog_wheat["availability_status"] == "available"
    assert catalog_wheat["category"] == "Cereals"
    assert catalog_response.json()["index_metadata"] == {
        "units": "GAEZ suitability index (0-10000)",
        "valid_range": [0, 10000],
        "scale_factor": 1.0,
        "no_data": -9,
    }
    assert catalog_response.json()["class_metadata"]["no_data"] == 0
    assert start_response.status_code == 202
    result = result_response.json()
    assert result["task_id"] == "task-api"
    assert result["status"] == "completed"
    assert result["analysis_version"] == gaez_v5.GAEZ_ANALYSIS_VERSION
    assert result["analysis_outdated"] is False
    assert len(result["scenarios"]) == 2
    rainfed = next(item for item in result["scenarios"] if item["water_regime"] == "rainfed")
    crop = rainfed["ranked_crops"][0]
    assert crop["intersecting_cell_count"] == 4
    assert crop["valid_index_cell_count"] == 4
    assert crop["class_data_available"] is True
    assert result["continuous_product"]["mapset"] == "RES05-SXX30AS"
    assert result["class_product"]["mapset"] == "RES05-SIX30AS"
    assert len(layers_response.json()["layers"]) == 4
    assert all(
        layer["product_id"] == "FAO_GAEZ_V5"
        for layer in layers_response.json()["layers"]
    )
    task_manifest = registry.load_task_manifest("api-aoi")[2]
    assert task_manifest["acquisition"]["providers"]["gaez_v5"]["status"] == "completed"
    assert len(task_manifest["gaez_runs"]) == 1


def test_aoi_creation_runs_gaez_inside_one_task_with_submitted_geometry(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    app = create_app(project_root=PROJECT_ROOT)
    app.state.tasks_directory = tmp_path / "tasks"
    monkeypatch.setattr(gaez_v5, "_open_remote_raster", _source_opener(tmp_path / "sources"))
    request = {
        "aoi_name": "Uzbekistan test area",
        "task_name": "uzbekistan-gaez",
        "aoi_geojson": AOI,
        "start_date": "2024-01-01",
        "end_date": "2024-01-02",
        "analysis_crs": "EPSG:32642",
        "resolution_m": 250,
        "gaez_v5": True,
        "confirm_acquisition": True,
    }

    with TestClient(app) as client:
        created = client.post("/api/aoi-tasks", json=request)
        task_list = client.get("/api/tasks").json()["tasks"]
        task_response = client.get(f"/api/tasks/{created.json()['task_id']}")

    assert created.status_code == 202
    assert len(task_list) == 1
    assert task_list[0]["task_name"] == "uzbekistan-gaez"
    manifest = task_response.json()
    assert manifest["aoi"]["geojson"] == AOI
    assert manifest["selected_datasets"]["gaez_v5"] is True
    assert manifest["acquisition"]["providers"]["gaez_v5"]["status"] == "completed"
    assert manifest["acquisition"]["status"] == "completed"
    assert len(manifest["gaez_runs"]) == 1
    run_id = manifest["gaez_runs"][0]["run_id"]
    with TestClient(app) as client:
        result = client.get(
            f"/api/aoi-tasks/{created.json()['task_id']}/gaez-suitability",
            params={"run_id": run_id},
        ).json()
    assert len(result["scenarios"]) == 2
