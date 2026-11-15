from __future__ import annotations

import hashlib
import json
from datetime import date
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import rasterio
import requests
from rasterio.transform import from_origin

from nasa_agri_intelligence.collection.downloader import DownloadManager
from nasa_agri_intelligence.collection.manifest import load_manifest
from nasa_agri_intelligence.collection.product_discovery import DatasetConfig, DatasetSpec
from nasa_agri_intelligence.collection.task_manager import (
    AOISettings,
    CollectionSettings,
    DuplicateTaskError,
    MonitoringSettings,
    OutputSettings,
    TaskManager,
    normalize_aoi,
)
from nasa_agri_intelligence.collection.task_monitor import TaskMonitor

TASK_ID = "11111111-2222-4333-8444-555555555555"
PRODUCT_ID = "MOD13Q1.061"
LAYER_ID = "_250m_16_days_NDVI"


class FakeResponse:
    def __init__(
        self,
        *,
        status_code: int = 200,
        headers: dict[str, str] | None = None,
        chunks: list[bytes] | None = None,
        fail_after_first_chunk: bool = False,
    ) -> None:
        self.status_code = status_code
        self.headers = headers or {}
        self.chunks = chunks or []
        self.fail_after_first_chunk = fail_after_first_chunk
        self.closed = False

    def __enter__(self) -> FakeResponse:
        return self

    def __exit__(self, *_: object) -> None:
        self.closed = True

    def iter_content(self, chunk_size: int) -> Any:
        del chunk_size
        for index, chunk in enumerate(self.chunks):
            yield chunk
            if self.fail_after_first_chunk and index == 0:
                raise requests.ConnectionError("interrupted stream")

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            response = requests.Response()
            response.status_code = self.status_code
            response.headers.update(self.headers)
            raise requests.HTTPError(f"HTTP {self.status_code}", response=response)


class FakeClient:
    base_url = "https://appeears.earthdatacloud.nasa.gov/api"

    def __init__(self) -> None:
        self.submitted_request: dict[str, Any] | None = None
        self.submit_error: BaseException | None = None
        self.remote_tasks: list[dict[str, Any]] = []
        self.status_responses: list[Any] = []
        self.bundle: dict[str, Any] = {"files": []}
        self.stream_responses: list[FakeResponse] = []
        self.stream_headers: list[dict[str, str] | None] = []
        self.get_stream_count = 0

    def get_json(
        self,
        endpoint: str,
        *,
        params: dict[str, Any] | None = None,
    ) -> Any:
        del params
        if endpoint == "task":
            return self.remote_tasks
        if endpoint.startswith("status/"):
            if not self.status_responses:
                return {"task_id": TASK_ID, "status": "done"}
            response = self.status_responses.pop(0)
            if isinstance(response, BaseException):
                raise response
            return response
        if endpoint == f"task/{TASK_ID}":
            return {"task_id": TASK_ID, "status": "done"}
        if endpoint == f"bundle/{TASK_ID}":
            return self.bundle
        raise AssertionError(f"Unexpected endpoint {endpoint}")

    def post_json(self, endpoint: str, payload: dict[str, Any]) -> tuple[Any, Any]:
        assert endpoint == "task"
        self.submitted_request = payload
        if self.submit_error:
            raise self.submit_error
        return {"task_id": TASK_ID, "status": "pending"}, type(
            "Response",
            (),
            {"headers": {}},
        )()

    def get_stream(
        self,
        endpoint: str,
        *,
        headers: dict[str, str] | None = None,
    ) -> FakeResponse:
        assert endpoint.startswith(f"bundle/{TASK_ID}/")
        self.get_stream_count += 1
        self.stream_headers.append(headers)
        if not self.stream_responses:
            raise AssertionError("No streaming response was queued.")
        return self.stream_responses.pop(0)


def _geojson() -> dict[str, Any]:
    return {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "properties": {},
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [[
                        [69.2797, 41.3111],
                        [69.2807, 41.3111],
                        [69.2807, 41.3121],
                        [69.2797, 41.3121],
                        [69.2797, 41.3111],
                    ]],
                },
            }
        ],
    }


def _make_manager(tmp_path: Path, client: FakeClient) -> TaskManager:
    metadata_dir = tmp_path / "metadata"
    metadata_dir.mkdir(exist_ok=True)
    (metadata_dir / "MOD13Q1.json").write_text(
        json.dumps(
            {
                "available_products": [
                    {
                        "catalog": {
                            "ProductAndVersion": PRODUCT_ID,
                            "Available": True,
                        },
                        "layers": {
                            LAYER_ID: {"Available": True, "Description": "NDVI"},
                        },
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    dataset = DatasetSpec(
        name="MOD13Q1",
        metadata_file="MOD13Q1.json",
        enabled=True,
        product_id=PRODUCT_ID,
        requested_layers=[LAYER_ID],
        output_group="MODIS",
    )
    settings = CollectionSettings(
        task_name_prefix="test",
        aoi=AOISettings(name="test-aoi", geojson=_geojson()),
        start_date=date(2025, 7, 1),
        end_date=date(2025, 7, 16),
        output=OutputSettings(
            directory=tmp_path / "raw",
            manifests_directory=tmp_path / "manifests",
            metadata_directory=metadata_dir,
        ),
        monitoring=MonitoringSettings(
            poll_interval_seconds=5,
            request_timeout_seconds=10,
            max_retries=2,
            max_backoff_seconds=5,
            max_monitor_minutes=1,
        ),
    )
    return TaskManager(
        client,  # type: ignore[arg-type]
        dataset_config=DatasetConfig(datasets=[dataset]),
        settings=settings,
    )


def _submit(manager: TaskManager) -> dict[str, Any]:
    return manager.submit(PRODUCT_ID, task_name="phase-test")


def _geotiff_bytes(path: Path) -> bytes:
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=2,
        width=3,
        count=1,
        dtype="uint8",
        crs="EPSG:4326",
        transform=from_origin(69.0, 42.0, 0.01, 0.01),
        nodata=0,
    ) as dataset:
        dataset.write(np.ones((1, 2, 3), dtype=np.uint8))
    return path.read_bytes()


def test_request_builder_uses_validated_metadata_and_official_area_schema(
    tmp_path: Path,
) -> None:
    client = FakeClient()
    manager = _make_manager(tmp_path, client)

    request = manager.build_request(PRODUCT_ID, task_name="test")

    assert request["task_type"] == "area"
    assert request["task_name"] == "test"
    assert request["params"]["layers"] == [{"product": PRODUCT_ID, "layer": LAYER_ID}]
    assert request["params"]["dates"] == [
        {"startDate": "07-01-2025", "endDate": "07-16-2025", "recurring": False}
    ]
    assert request["params"]["geo"]["type"] == "FeatureCollection"
    assert request["params"]["output"] == {
        "format": {"type": "geotiff", "filename_date": "calendar"},
        "projection": "native",
    }


def test_request_builder_rejects_unavailable_layer(tmp_path: Path) -> None:
    client = FakeClient()
    manager = _make_manager(tmp_path, client)
    manager.dataset_config.datasets[0].requested_layers = ["unverified-layer"]

    with pytest.raises(ValueError, match="not available"):
        manager.build_request(PRODUCT_ID)


def test_aoi_validation_rejects_invalid_and_non_polygon_geometry() -> None:
    with pytest.raises(ValueError, match="polygon"):
        normalize_aoi({"type": "Point", "coordinates": [69.2, 41.3]})

    invalid_polygon = {
        "type": "Polygon",
        "coordinates": [[
            [0, 0],
            [1, 1],
            [1, 0],
            [0, 1],
            [0, 0],
        ]],
    }
    with pytest.raises(ValueError, match="valid"):
        normalize_aoi(invalid_polygon)


def test_aoi_normalization_preserves_submitted_geometry_and_coordinate_order() -> None:
    selected_geometry = _geojson()["features"][0]["geometry"]
    submitted = {
        "type": "Feature",
        "properties": {"selection": "drawn"},
        "geometry": selected_geometry,
    }

    normalized = normalize_aoi(submitted)
    restored = json.loads(json.dumps(normalized))

    assert restored["type"] == "FeatureCollection"
    assert restored["features"][0]["properties"] == {"selection": "drawn"}
    assert restored["features"][0]["geometry"] == selected_geometry
    assert restored["features"][0]["geometry"]["coordinates"][0][0] == [
        69.2797,
        41.3111,
    ]


def test_rectangle_aoi_serialization_preserves_both_drawn_corners() -> None:
    rectangle = {
        "type": "Polygon",
        "coordinates": [[
            [64.4200, 39.6500],
            [64.4210, 39.6500],
            [64.4210, 39.6510],
            [64.4200, 39.6510],
            [64.4200, 39.6500],
        ]],
    }

    normalized = normalize_aoi(rectangle)
    restored = json.loads(json.dumps(normalized))

    assert restored["features"][0]["geometry"] == rectangle
    assert restored["features"][0]["geometry"]["coordinates"][0][0] == [
        64.42,
        39.65,
    ]


@pytest.mark.parametrize(
    ("geometry", "message"),
    [
        (
            {
                "type": "Polygon",
                "coordinates": [[
                    [69, 41],
                    [70, 41],
                    [70, 42],
                    [69, 42],
                    [69, float("nan")],
                ]],
            },
            "finite numeric",
        ),
        (
            {
                "type": "Polygon",
                "coordinates": [[
                    [181, 41],
                    [182, 41],
                    [182, 42],
                    [181, 42],
                    [181, 41],
                ]],
            },
            "longitude/latitude bounds",
        ),
        (
            {
                "type": "Polygon",
                "coordinates": [[
                    [69, 41],
                    [70, 41],
                    [70, 42],
                    [69, 42],
                ]],
            },
            "must be closed",
        ),
    ],
)
def test_aoi_validation_rejects_invalid_coordinates_and_rings(
    geometry: dict[str, Any],
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        normalize_aoi(geometry)


def test_aoi_validation_rejects_non_wgs84_crs_and_accepts_explicit_wgs84() -> None:
    geometry = _geojson()

    with pytest.raises(ValueError, match="must use EPSG:4326"):
        normalize_aoi({**geometry, "crs": "EPSG:3857"})

    normalized = normalize_aoi(
        {
            **geometry,
            "crs": {
                "type": "name",
                "properties": {"name": "EPSG:4326"},
            },
        }
    )
    assert normalized["features"][0]["geometry"] == geometry["features"][0]["geometry"]


def test_submit_persists_task_id_and_prevents_duplicates(tmp_path: Path) -> None:
    client = FakeClient()
    manager = _make_manager(tmp_path, client)

    manifest = _submit(manager)

    assert manifest["task_id"] == TASK_ID
    assert manifest["submission_status"] == "submitted"
    assert manifest["task_status"] == "pending"
    assert manifest["task_request"] == client.submitted_request
    assert manifest["submitted_at"]
    stored = load_manifest(tmp_path / "tasks" / "phase-test" / "task_manifest.json")
    assert stored["product_id"] == PRODUCT_ID
    registry = manager.registry.records()
    assert registry[0]["task_name"] == "phase-test"
    assert registry[0]["task_id"] == TASK_ID
    with pytest.raises(DuplicateTaskError, match="allow-duplicate"):
        manager.submit(PRODUCT_ID, task_name="different-name")
    assert len(manager.registry.records()) == 1


def test_duplicate_check_compares_remote_product_layers_and_area(tmp_path: Path) -> None:
    client = FakeClient()
    manager = _make_manager(tmp_path, client)
    request = manager.build_request(PRODUCT_ID, task_name="first")
    client.remote_tasks = [
        {
            "task_id": "remote-task-id",
            "task_name": "same request",
            "task_type": request["task_type"],
            "params": request["params"],
        }
    ]

    assert manager.find_duplicates(request) == ["remote-task-id"]


def test_duplicate_check_skips_remote_point_and_nonmatching_area_tasks(
    tmp_path: Path,
) -> None:
    client = FakeClient()
    manager = _make_manager(tmp_path, client)
    request = manager.build_request(PRODUCT_ID, task_name="new request")
    client.remote_tasks = [
        {
            "task_id": "remote-point",
            "task_type": "point",
            "params": {"coordinates": [69.28, 41.31]},
        },
        {
            "task_id": "remote-area",
            "task_type": "area",
            "params": {
                "dates": [{"startDate": "08-01-2026", "endDate": "08-16-2026"}],
                "layers": request["params"]["layers"],
                "output": request["params"]["output"],
            },
        },
    ]

    assert manager.find_duplicates(request) == []


def test_submission_timeout_is_persisted_as_unknown_without_retry(tmp_path: Path) -> None:
    client = FakeClient()
    client.submit_error = requests.Timeout("timeout")
    manager = _make_manager(tmp_path, client)

    with pytest.raises(requests.Timeout):
        manager.submit(PRODUCT_ID, task_name="uncertain")

    manifest = load_manifest(tmp_path / "tasks" / "uncertain" / "task_manifest.json")
    assert manifest["submission_status"] == "submission_unknown"
    assert client.submitted_request is not None


def test_rejected_submission_is_persisted_as_failed(tmp_path: Path) -> None:
    client = FakeClient()
    response = requests.Response()
    response.status_code = 400
    client.submit_error = requests.HTTPError("invalid request", response=response)
    manager = _make_manager(tmp_path, client)

    with pytest.raises(requests.HTTPError):
        manager.submit(PRODUCT_ID, task_name="rejected")

    manifest = load_manifest(tmp_path / "tasks" / "rejected" / "task_manifest.json")
    assert manifest["submission_status"] == "failed"
    assert manifest["task_status"] == "failed"


def test_monitoring_persists_processing_then_completion(tmp_path: Path) -> None:
    client = FakeClient()
    manager = _make_manager(tmp_path, client)
    _submit(manager)
    client.status_responses = [
        [
            {
                "task_id": TASK_ID,
                "progress": {"summary": 35},
                "status_type": "task",
            }
        ],
        {"task_id": TASK_ID, "status": "done", "updated": "2025-07-17T00:00:00"},
    ]
    monitor = TaskMonitor(manager)

    processing = monitor.check_once(TASK_ID)
    completed = monitor.check_once(TASK_ID)

    assert processing["task_status"] == "processing"
    assert completed["task_status"] == "completed"
    assert completed["completed_at"] == "2025-07-17T00:00:00"
    assert completed["last_status_check"]


def test_monitor_retries_temporary_network_errors_and_resumes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = FakeClient()
    manager = _make_manager(tmp_path, client)
    _submit(manager)
    client.status_responses = [
        requests.Timeout("temporary"),
        {"task_id": TASK_ID, "status": "done"},
    ]
    sleeps: list[float] = []
    monkeypatch.setattr("nasa_agri_intelligence.collection.task_monitor.time.sleep", sleeps.append)

    resumed_manager = _make_manager(tmp_path, client)
    result = TaskMonitor(resumed_manager).monitor(TASK_ID, max_wait_minutes=1)

    assert result["task_status"] == "completed"
    assert sleeps == [1]


def test_failed_task_status_is_terminal_and_recorded(tmp_path: Path) -> None:
    client = FakeClient()
    manager = _make_manager(tmp_path, client)
    _submit(manager)
    client.status_responses = [
        {"task_id": TASK_ID, "status": "error", "error": "No granules found."}
    ]

    result = TaskMonitor(manager).check_once(TASK_ID)

    assert result["task_status"] == "failed"
    assert result["error"] == "No granules found."


def test_downloader_streams_and_validates_geotiff_then_skips_valid_file(
    tmp_path: Path,
) -> None:
    client = FakeClient()
    manager = _make_manager(tmp_path, client)
    submitted = _submit(manager)
    submitted["task_status"] = "completed"
    path, _ = manager.get_manifest(TASK_ID)
    manager.save_manifest(path, submitted)

    data = _geotiff_bytes(tmp_path / "source.tif")
    name = "MOD13Q1.061_ndvi.tif"
    checksum = hashlib.sha256(data).hexdigest()
    client.bundle = {
        "task_id": TASK_ID,
        "files": [
            {
                "file_id": "file-1",
                "file_name": name,
                "file_size": len(data),
                "sha256": checksum,
                "file_type": "tif",
            }
        ],
    }
    client.stream_responses = [
        FakeResponse(
            headers={"Content-Type": "image/tiff"},
            chunks=[data],
        )
    ]

    downloader = DownloadManager(manager)
    manifest = downloader.download_task(TASK_ID)
    result = manifest["downloads"][0]

    assert manifest["download_status"] == "completed"
    assert result["status"] == "validated"
    assert result["validation"]["raster"]["width"] == 3
    assert Path(result["local_path"]).read_bytes() == data
    assert client.get_stream_count == 1

    repeated = downloader.download_task(TASK_ID)
    assert repeated["download_status"] == "completed"
    assert client.get_stream_count == 1


def test_downloader_resumes_partial_file_only_after_range_confirmation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = FakeClient()
    manager = _make_manager(tmp_path, client)
    manifest = _submit(manager)
    manifest["task_status"] = "completed"
    path, _ = manager.get_manifest(TASK_ID)
    manager.save_manifest(path, manifest)

    data = _geotiff_bytes(tmp_path / "source.tif")
    checksum = hashlib.sha256(data).hexdigest()
    filename = "granule.tif"
    destination = Path(manifest["output_directory"]) / filename
    destination.parent.mkdir(parents=True, exist_ok=True)
    other_task = manager.registry.reserve(
        "other-task",
        {
            "task_id": "other-task-id",
            "task_directory": str(tmp_path / "tasks" / "other-task"),
        },
    )
    other_destination = other_task / "raw/MODIS/MOD13Q1" / filename
    other_destination.parent.mkdir(parents=True, exist_ok=True)
    other_destination.write_bytes(b"other-task-data")
    partial_size = 5
    destination.with_name(f"{filename}.part").write_bytes(data[:partial_size])
    client.bundle = {
        "task_id": TASK_ID,
        "files": [
            {
                "file_id": "file-2",
                "file_name": filename,
                "file_size": len(data),
                "sha256": checksum,
                "file_type": "tif",
            }
        ],
    }
    client.stream_responses = [
        FakeResponse(
            status_code=206,
            headers={
                "Content-Range": f"bytes {partial_size}-{len(data) - 1}/{len(data)}",
                "Content-Type": "image/tiff",
            },
            chunks=[data[partial_size:]],
        )
    ]
    monkeypatch.setattr("nasa_agri_intelligence.collection.downloader.time.sleep", lambda _: None)

    result = DownloadManager(manager).download_task(TASK_ID)

    assert client.stream_headers == [{"Range": f"bytes={partial_size}-"}]
    assert Path(result["downloads"][0]["local_path"]).read_bytes() == data
    assert other_destination.read_bytes() == b"other-task-data"
    assert result["downloads"][0]["status"] == "validated"


def test_downloader_retries_interrupted_transfer_using_http_range(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = FakeClient()
    manager = _make_manager(tmp_path, client)
    manifest = _submit(manager)
    manifest["task_status"] = "completed"
    path, _ = manager.get_manifest(TASK_ID)
    manager.save_manifest(path, manifest)

    data = _geotiff_bytes(tmp_path / "source.tif")
    checksum = hashlib.sha256(data).hexdigest()
    filename = "results.tif"
    client.bundle = {
        "task_id": TASK_ID,
        "files": [
            {
                "file_id": "file-3",
                "file_name": filename,
                "file_size": len(data),
                "sha256": checksum,
                "file_type": "tif",
            }
        ],
    }
    client.stream_responses = [
        FakeResponse(
            headers={"Content-Type": "image/tiff"},
            chunks=[data[:7]],
            fail_after_first_chunk=True,
        ),
        FakeResponse(
            status_code=206,
            headers={
                "Content-Range": f"bytes 7-{len(data) - 1}/{len(data)}",
                "Content-Type": "image/tiff",
            },
            chunks=[data[7:]],
        ),
    ]
    monkeypatch.setattr("nasa_agri_intelligence.collection.downloader.time.sleep", lambda _: None)

    result = DownloadManager(manager).download_task(TASK_ID)

    assert client.stream_headers == [None, {"Range": "bytes=7-"}]
    assert Path(result["downloads"][0]["local_path"]).read_bytes() == data
    assert result["downloads"][0]["status"] == "validated"


def test_downloader_retries_empty_response_and_saves_valid_file(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = FakeClient()
    manager = _make_manager(tmp_path, client)
    manifest = _submit(manager)
    manifest["task_status"] = "completed"
    path, _ = manager.get_manifest(TASK_ID)
    manager.save_manifest(path, manifest)

    data = _geotiff_bytes(tmp_path / "source.tif")
    filename = "statistics.tif"
    client.bundle = {
        "task_id": TASK_ID,
        "files": [
            {
                "file_id": "file-empty-retry",
                "file_name": filename,
                "file_size": len(data),
                "sha256": hashlib.sha256(data).hexdigest(),
                "file_type": "tif",
            }
        ],
    }
    client.stream_responses = [
        FakeResponse(headers={"Content-Type": "image/tiff"}),
        FakeResponse(headers={"Content-Type": "image/tiff"}, chunks=[data]),
    ]
    monkeypatch.setattr("nasa_agri_intelligence.collection.downloader.time.sleep", lambda _: None)

    result = DownloadManager(manager).download_task(TASK_ID)

    assert client.get_stream_count == 2
    assert result["download_status"] == "completed"
    assert Path(result["downloads"][0]["local_path"]).read_bytes() == data
    assert result["downloads"][0]["status"] == "validated"


def test_downloader_replaces_invalid_existing_file(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = FakeClient()
    manager = _make_manager(tmp_path, client)
    manifest = _submit(manager)
    manifest["task_status"] = "completed"
    path, _ = manager.get_manifest(TASK_ID)
    manager.save_manifest(path, manifest)

    data = _geotiff_bytes(tmp_path / "source.tif")
    filename = "statistics.tif"
    destination = Path(manifest["output_directory"]) / filename
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(b"")
    client.bundle = {
        "task_id": TASK_ID,
        "files": [
            {
                "file_id": "file-invalid-local",
                "file_name": filename,
                "file_size": len(data),
                "sha256": hashlib.sha256(data).hexdigest(),
                "file_type": "tif",
            }
        ],
    }
    client.stream_responses = [
        FakeResponse(headers={"Content-Type": "image/tiff"}, chunks=[data]),
    ]
    monkeypatch.setattr("nasa_agri_intelligence.collection.downloader.time.sleep", lambda _: None)

    result = DownloadManager(manager).download_task(TASK_ID)

    assert client.get_stream_count == 1
    assert result["download_status"] == "completed"
    assert destination.read_bytes() == data
    assert result["downloads"][0]["status"] == "validated"


def test_downloader_rejects_completed_bundle_without_requested_data_output(
    tmp_path: Path,
) -> None:
    client = FakeClient()
    manager = _make_manager(tmp_path, client)
    manifest = _submit(manager)
    manifest["task_status"] = "completed"
    path, _ = manager.get_manifest(TASK_ID)
    manager.save_manifest(path, manifest)
    client.bundle = {
        "task_id": TASK_ID,
        "files": [
            {
                "file_id": "metadata-1",
                "file_name": "request.json",
                "file_size": 2,
                "sha256": hashlib.sha256(b"{}").hexdigest(),
                "file_type": "json",
            }
        ],
    }

    with pytest.raises(ValueError, match="no geotiff data output"):
        DownloadManager(manager).download_task(TASK_ID)

    persisted = load_manifest(path)
    assert persisted["download_status"] == "failed"
    assert Path(persisted["output_listing_path"]).exists()


def test_geotiff_validation_rejects_non_raster_content(tmp_path: Path) -> None:
    path = tmp_path / "corrupt.tif"
    path.write_bytes(b"not-a-raster")

    with pytest.raises(ValueError, match="Invalid GeoTIFF"):
        DownloadManager._validate_file(
            path,
            expected_size=None,
            checksum=None,
            original_name=path.name,
        )


def test_bundle_paths_allow_subdirectories_but_reject_traversal(tmp_path: Path) -> None:
    root = tmp_path / "downloads"

    destination = DownloadManager._destination_path(
        root,
        "MOD13Q1.061_2025167_to_2025197/ndvi.tif",
    )

    assert destination == root / "MOD13Q1.061_2025167_to_2025197" / "ndvi.tif"
    assert destination.parent.is_dir()
    with pytest.raises(ValueError, match="Unsafe AppEEARS bundle filename"):
        DownloadManager._destination_path(root, "../outside.tif")


def test_malformed_appeears_xml_is_preserved_with_validation_warning(
    tmp_path: Path,
) -> None:
    path = tmp_path / "metadata.xml"
    content = b"<metadata><title>NDVI & EVI</title></metadata>"
    path.write_bytes(content)

    details = DownloadManager._validate_file(
        path,
        expected_size=len(content),
        checksum=hashlib.sha256(content).hexdigest(),
        original_name=path.name,
    )

    assert details["sha256"] == hashlib.sha256(content).hexdigest()
    assert "not well-formed" in details["format_warning"]
