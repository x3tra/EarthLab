from __future__ import annotations

import json
from typing import Any
from urllib.parse import quote

import requests

from ..config import EarthdataSettings
from .authentication import EarthdataAuthenticator


class AppEEARSClient:
    """Small client for the AppEEARS catalog and product metadata endpoints."""

    def __init__(
        self,
        settings: EarthdataSettings | None = None,
        *,
        session: requests.Session | None = None,
    ) -> None:
        self.settings = settings or EarthdataSettings()
        self.base_url = f"{self.settings.nasa_earthdata_base_url.rstrip('/')}/api"
        self.authenticator = EarthdataAuthenticator(self.settings, session)
        self.session = self.authenticator.session

    def get_json(self, endpoint: str, *, params: dict[str, Any] | None = None) -> Any:
        response = self._get_response(endpoint, params=params, timeout=60)
        return response.json()

    def post_json(self, endpoint: str, payload: dict[str, Any]) -> tuple[Any, requests.Response]:
        """POST one request without retrying an ambiguous submission."""

        url = f"{self.base_url}/{endpoint.lstrip('/')}"
        response = self.session.post(url, json=payload, timeout=60)
        if response.status_code == 403 and self.authenticator.can_refresh:
            self.authenticator.authenticate()
            response = self.session.post(url, json=payload, timeout=60)
        self._raise_for_status(response)
        try:
            return response.json(), response
        except requests.exceptions.JSONDecodeError as error:
            raise ValueError(f"AppEEARS returned invalid JSON from {endpoint}.") from error

    def get_stream(
        self,
        endpoint: str,
        *,
        headers: dict[str, str] | None = None,
    ) -> requests.Response:
        """Return a streamed response, following AppEEARS bundle redirects."""

        return self._get_response(
            endpoint,
            timeout=120,
            stream=True,
            headers=headers,
        )

    def _get_response(
        self,
        endpoint: str,
        *,
        params: dict[str, Any] | None = None,
        timeout: int,
        stream: bool = False,
        headers: dict[str, str] | None = None,
    ) -> requests.Response:
        url = f"{self.base_url}/{endpoint.lstrip('/')}"
        response = self.session.get(
            url,
            params=params,
            timeout=timeout,
            stream=stream,
            allow_redirects=True,
            headers=headers,
        )
        if response.status_code == 403 and self.authenticator.can_refresh:
            self.authenticator.authenticate()
            response = self.session.get(
                url,
                params=params,
                timeout=timeout,
                stream=stream,
                allow_redirects=True,
                headers=headers,
            )
        self._raise_for_status(response)
        return response

    @staticmethod
    def _raise_for_status(response: requests.Response) -> None:
        if response.ok:
            return
        message = ""
        try:
            payload = response.json()
        except (requests.exceptions.JSONDecodeError, json.JSONDecodeError):
            payload = None
        if isinstance(payload, dict):
            message = str(payload.get("message") or payload.get("error") or "")
        elif response.text:
            message = response.text[:500]
        detail = f": {message}" if message else ""
        raise requests.HTTPError(
            f"AppEEARS request failed with HTTP {response.status_code}{detail}",
            response=response,
        )

    def product_catalog(self, *, page_size: int = 1000) -> list[dict[str, Any]]:
        """Return the complete live AppEEARS product catalog using API pagination."""

        if page_size < 1:
            raise ValueError("page_size must be greater than zero.")

        products: list[dict[str, Any]] = []
        offset = 0
        while True:
            page = self.get_json(
                "product",
                params={"limit": page_size, "offset": offset},
            )
            if not isinstance(page, list) or any(not isinstance(item, dict) for item in page):
                raise ValueError("AppEEARS returned an unexpected product catalog format.")
            products.extend(page)
            if len(page) < page_size:
                return products
            offset += len(page)

    def product_metadata(self, product_id: str) -> dict[str, dict[str, Any]]:
        """Return all layer metadata for a catalog product/version."""

        payload = self.get_json(f"product/{quote(product_id, safe='.')}")
        if not isinstance(payload, dict) or any(
            not isinstance(metadata, dict) for metadata in payload.values()
        ):
            raise ValueError(
                f"AppEEARS returned unexpected layer metadata for {product_id}."
            )
        return payload
