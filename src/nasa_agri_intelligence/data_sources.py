from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import requests

from .collection.appeears_client import AppEEARSClient
from .config import EarthdataSettings
from .models import DATASET_CATALOG


@dataclass(frozen=True)
class DataRequest:
    dataset_key: str
    product_name: str
    layers: list[str]
    start_date: str
    end_date: str
    aoi_name: str
    crs: str


class NASADataClient:
    """Thin wrapper around NASA Earthdata/AppEEARS request patterns."""

    def __init__(self, settings: EarthdataSettings | None = None) -> None:
        self.settings = settings or EarthdataSettings()
        self.base_url = self.settings.nasa_earthdata_base_url
        self.api_client = AppEEARSClient(self.settings)
        self.session = self.api_client.session

    def catalog(self) -> dict[str, Any]:
        """Return the project's local dataset definitions."""

        return DATASET_CATALOG

    def product_catalog(self) -> list[dict[str, Any]]:
        """Return the live AppEEARS product catalog."""

        return self.api_client.product_catalog()

    def product_metadata(self, product_id: str) -> dict[str, dict[str, Any]]:
        """Return live layer metadata for an AppEEARS product/version."""

        return self.api_client.product_metadata(product_id)

    def check_availability(
        self,
        dataset_key: str,
        product_name: str | None = None,
        layers: list[str] | None = None,
    ) -> dict[str, Any]:
        dataset = DATASET_CATALOG.get(dataset_key)
        if dataset is None:
            raise ValueError(f"Unsupported dataset key: {dataset_key}")

        requested_layers = list(layers or dataset.variables.keys())
        missing = [layer for layer in requested_layers if layer not in dataset.variables]
        return {
            "dataset": dataset_key,
            "product": product_name or dataset.product_name,
            "requested_layers": requested_layers,
            "available": not missing,
            "missing_layers": missing,
            "provider": dataset.provider,
        }

    def build_request(
        self,
        *,
        dataset_key: str,
        layers: list[str],
        start_date: str,
        end_date: str,
        aoi_name: str,
        crs: str,
    ) -> DataRequest:
        dataset = DATASET_CATALOG[dataset_key]
        availability = self.check_availability(dataset_key, layers=layers)
        if not availability["available"]:
            missing = ", ".join(availability["missing_layers"])
            raise ValueError(f"Requested dataset layers are not available: {missing}")

        return DataRequest(
            dataset_key=dataset_key,
            product_name=dataset.product_name,
            layers=layers,
            start_date=start_date,
            end_date=end_date,
            aoi_name=aoi_name,
            crs=crs,
        )

    def download(self, url: str, destination: str | Path) -> Path:
        destination_path = Path(destination)
        destination_path.parent.mkdir(parents=True, exist_ok=True)
        target_host = urlparse(url).hostname
        appeears_host = urlparse(self.base_url).hostname
        if target_host == appeears_host:
            response = self.session.get(url, timeout=120)
        else:
            response = requests.get(url, timeout=120)
        response.raise_for_status()
        destination_path.write_bytes(response.content)
        return destination_path

    def validate_download(self, file_path: str | Path) -> bool:
        path = Path(file_path)
        return path.exists() and path.stat().st_size > 0
