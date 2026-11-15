from __future__ import annotations

from typing import Literal

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class EarthdataSettings(BaseSettings):
    """Environment-backed settings for NASA data access."""

    earthdata_username: str = Field(default="", description="NASA Earthdata username")
    earthdata_password: SecretStr | None = Field(
        default=None,
        description="NASA Earthdata password or token for authenticated requests",
    )
    nasa_app_ears_token: SecretStr | None = Field(
        default=None,
        description="Optional AppEEARS token for the NASA API",
    )
    nasa_earthdata_base_url: str = Field(
        default="https://appeears.earthdatacloud.nasa.gov",
        description="NASA Earthdata/AppEEARS endpoint",
    )
    workspace_dir: str = Field(default="./data", description="Root directory for downloaded datasets")
    aoi_name: str = Field(default="uzbekistan_sample", description="Area-of-interest label")
    aoi_crs: str = Field(default="EPSG:4326", description="CRS for AOI geometry")
    common_resolution_m: float = Field(default=250.0, gt=0, description="Target common raster resolution in meters")
    common_crs: str = Field(default="EPSG:4326", description="Output CRS for aligned products")
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = "INFO"

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )


class ProcessingSettings:
    """Simple processing settings container for raster and AOI workflows."""

    def __init__(
        self,
        *,
        common_crs: str = "EPSG:4326",
        common_resolution_m: float = 250.0,
        aoi_crs: str = "EPSG:4326",
        nodata_policy: str = "preserve",
    ) -> None:
        self.common_crs = common_crs
        self.common_resolution_m = common_resolution_m
        self.aoi_crs = aoi_crs
        self.nodata_policy = nodata_policy
