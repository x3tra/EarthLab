from __future__ import annotations

from typing import Any

import requests

from ..config import EarthdataSettings


class EarthdataAuthenticator:
    """Create and refresh AppEEARS Bearer-token authentication."""

    def __init__(
        self,
        settings: EarthdataSettings,
        session: requests.Session | None = None,
    ) -> None:
        self.settings = settings
        self.session = session or requests.Session()
        self.session.headers.update({"User-Agent": "nasa-agri-intelligence/0.1.0"})

        token = (
            settings.nasa_app_ears_token.get_secret_value()
            if settings.nasa_app_ears_token
            else ""
        )
        if token.strip():
            self.set_token(token)
        elif self.can_refresh:
            self.authenticate()
        else:
            raise ValueError(
                "Set NASA_APP_EARS_TOKEN or both EARTHDATA_USERNAME and "
                "EARTHDATA_PASSWORD in .env before using AppEEARS."
            )

    @property
    def can_refresh(self) -> bool:
        password = self.settings.earthdata_password
        return bool(
            self.settings.earthdata_username.strip()
            and password
            and password.get_secret_value()
        )

    def set_token(self, token: str) -> None:
        if not token.strip():
            raise ValueError("The AppEEARS token cannot be empty.")
        self.session.headers["Authorization"] = f"Bearer {token}"

    def authenticate(self) -> None:
        username = self.settings.earthdata_username
        password = self.settings.earthdata_password
        if not username.strip() or password is None or not password.get_secret_value():
            raise ValueError(
                "The AppEEARS token cannot be refreshed without "
                "EARTHDATA_USERNAME and EARTHDATA_PASSWORD."
            )

        response = self.session.post(
            f"{self.settings.nasa_earthdata_base_url.rstrip('/')}/api/login",
            auth=(
                username,
                password.get_secret_value(),
            ),
            timeout=30,
        )
        response.raise_for_status()
        payload: Any = response.json()
        token = payload.get("token") if isinstance(payload, dict) else None
        if not isinstance(token, str) or not token:
            raise ValueError("AppEEARS login succeeded but did not return a token.")
        self.set_token(token)
