from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping


@dataclass(frozen=True)
class VariableDefinition:
    name: str
    description: str
    units: str
    valid_range: tuple[float, float] | None = None
    no_data_value: float | None = None
    scale_factor: float = 1.0
    offset: float = 0.0
    temporal_resolution: str = "variable"
    spatial_resolution: str = "variable"
    quality_flag: bool = False
    source: str = "NASA"


@dataclass(frozen=True)
class DatasetDefinition:
    key: str
    title: str
    provider: str
    product_name: str
    spatial_resolution: str
    temporal_resolution: str
    crs: str
    variables: Mapping[str, VariableDefinition] = field(default_factory=dict)


DATASET_CATALOG: dict[str, DatasetDefinition] = {
    "hls": DatasetDefinition(
        key="hls",
        title="HLS Harmonized Landsat and Sentinel",
        provider="NASA LPDAAC / Earthdata",
        product_name="HLS",
        spatial_resolution="10-30 m",
        temporal_resolution="daily to 8-day composite",
        crs="EPSG:4326",
        variables={
            "B03": VariableDefinition("B03", "Green band", "reflectance", valid_range=(0, 1), scale_factor=0.0001, temporal_resolution="daily", spatial_resolution="10-30 m"),
            "B04": VariableDefinition("B04", "Red band", "reflectance", valid_range=(0, 1), scale_factor=0.0001, temporal_resolution="daily", spatial_resolution="10-30 m"),
            "B05": VariableDefinition("B05", "Red-edge / SWIR band", "reflectance", valid_range=(0, 1), scale_factor=0.0001, temporal_resolution="daily", spatial_resolution="10-30 m"),
            "B06": VariableDefinition("B06", "Shortwave infrared band", "reflectance", valid_range=(0, 1), scale_factor=0.0001, temporal_resolution="daily", spatial_resolution="10-30 m"),
            "B07": VariableDefinition("B07", "Shortwave infrared band", "reflectance", valid_range=(0, 1), scale_factor=0.0001, temporal_resolution="daily", spatial_resolution="10-30 m"),
            "Fmask": VariableDefinition("Fmask", "Quality mask", "class", valid_range=(0, 255), quality_flag=True, temporal_resolution="daily", spatial_resolution="10-30 m"),
            "NDVI": VariableDefinition("NDVI", "Normalized Difference Vegetation Index", "index", valid_range=(-1, 1), temporal_resolution="daily", spatial_resolution="10-30 m"),
            "NDMI": VariableDefinition("NDMI", "Normalized Difference Moisture Index", "index", valid_range=(-1, 1), temporal_resolution="daily", spatial_resolution="10-30 m"),
            "SAVI": VariableDefinition("SAVI", "Soil Adjusted Vegetation Index", "index", valid_range=(-1, 1), temporal_resolution="daily", spatial_resolution="10-30 m"),
            "MNDWI": VariableDefinition("MNDWI", "Modified Normalized Difference Water Index", "index", valid_range=(-1, 1), temporal_resolution="daily", spatial_resolution="10-30 m"),
        },
    ),
    "modis_ndvi": DatasetDefinition(
        key="modis_ndvi",
        title="MODIS Vegetation Indices",
        provider="NASA LPDAAC",
        product_name="MOD13A1/ MOD13Q1",
        spatial_resolution="250 m - 1 km",
        temporal_resolution="16-day",
        crs="EPSG:4326",
        variables={
            "NDVI": VariableDefinition("NDVI", "Normalized Difference Vegetation Index", "index", valid_range=(-1, 1), temporal_resolution="16-day", spatial_resolution="250 m"),
            "EVI": VariableDefinition("EVI", "Enhanced Vegetation Index", "index", valid_range=(-1, 1), temporal_resolution="16-day", spatial_resolution="250 m"),
            "DetailedQA": VariableDefinition("DetailedQA", "Detailed pixel quality mask", "bitmask", quality_flag=True, temporal_resolution="16-day", spatial_resolution="250 m"),
            "SummaryQA": VariableDefinition("SummaryQA", "Summary quality mask", "bitmask", quality_flag=True, temporal_resolution="16-day", spatial_resolution="250 m"),
        },
    ),
    "modis_lai_fpar": DatasetDefinition(
        key="modis_lai_fpar",
        title="MODIS LAI/FPAR",
        provider="NASA LPDAAC",
        product_name="MOD15A2H",
        spatial_resolution="500 m",
        temporal_resolution="8-day",
        crs="EPSG:4326",
        variables={
            "Lai": VariableDefinition("Lai", "Leaf area index", "m2/m2", valid_range=(0, 10), temporal_resolution="8-day", spatial_resolution="500 m"),
            "Fpar": VariableDefinition("Fpar", "Fraction of photosynthetically active radiation", "fraction", valid_range=(0, 1), temporal_resolution="8-day", spatial_resolution="500 m"),
            "FparLai_QC": VariableDefinition("FparLai_QC", "Quality control bitmask", "bitmask", quality_flag=True, temporal_resolution="8-day", spatial_resolution="500 m"),
        },
    ),
    "modis_lst": DatasetDefinition(
        key="modis_lst",
        title="MODIS Land Surface Temperature",
        provider="NASA LPDAAC",
        product_name="MOD11A1",
        spatial_resolution="1 km",
        temporal_resolution="daily",
        crs="EPSG:4326",
        variables={
            "LST_Day_1km": VariableDefinition("LST_Day_1km", "Daytime land surface temperature", "degC", valid_range=(-273.15, 80), scale_factor=0.02, offset=0.0, temporal_resolution="daily", spatial_resolution="1 km"),
            "LST_Night_1km": VariableDefinition("LST_Night_1km", "Nighttime land surface temperature", "degC", valid_range=(-273.15, 80), scale_factor=0.02, offset=0.0, temporal_resolution="daily", spatial_resolution="1 km"),
            "QC_Day": VariableDefinition("QC_Day", "Daytime quality control mask", "bitmask", quality_flag=True, temporal_resolution="daily", spatial_resolution="1 km"),
            "QC_Night": VariableDefinition("QC_Night", "Nighttime quality control mask", "bitmask", quality_flag=True, temporal_resolution="daily", spatial_resolution="1 km"),
        },
    ),
    "modis_et": DatasetDefinition(
        key="modis_et",
        title="MODIS Evapotranspiration",
        provider="NASA LPDAAC",
        product_name="MOD16A2",
        spatial_resolution="500 m",
        temporal_resolution="8-day",
        crs="EPSG:4326",
        variables={
            "ET_500M": VariableDefinition("ET_500M", "Evapotranspiration", "mm", valid_range=(0, 1000), scale_factor=0.1, temporal_resolution="8-day", spatial_resolution="500 m"),
            "PET_500M": VariableDefinition("PET_500M", "Potential evapotranspiration", "mm", valid_range=(0, 1000), scale_factor=0.1, temporal_resolution="8-day", spatial_resolution="500 m"),
            "ET_QC_500M": VariableDefinition("ET_QC_500M", "Evapotranspiration quality mask", "bitmask", quality_flag=True, temporal_resolution="8-day", spatial_resolution="500 m"),
        },
    ),
    "smap": DatasetDefinition(
        key="smap",
        title="SMAP Soil Moisture",
        provider="NASA NSIDC",
        product_name="SMAP L3",
        spatial_resolution="9 km",
        temporal_resolution="daily",
        crs="EPSG:4326",
        variables={
            "sm_surface": VariableDefinition("sm_surface", "Surface soil moisture", "m3/m3", valid_range=(0, 1), scale_factor=0.00004, temporal_resolution="daily", spatial_resolution="9 km"),
            "sm_rootzone": VariableDefinition("sm_rootzone", "Root-zone soil moisture", "m3/m3", valid_range=(0, 1), scale_factor=0.00004, temporal_resolution="daily", spatial_resolution="9 km"),
            "sm_surface_wetness": VariableDefinition("sm_surface_wetness", "Surface wetness index", "percent", valid_range=(0, 100), temporal_resolution="daily", spatial_resolution="9 km"),
            "sm_rootzone_wetness": VariableDefinition("sm_rootzone_wetness", "Root-zone wetness index", "percent", valid_range=(0, 100), temporal_resolution="daily", spatial_resolution="9 km"),
        },
    ),
    "ecostress": DatasetDefinition(
        key="ecostress",
        title="ECOSTRESS thermal and water stress",
        provider="NASA JPL",
        product_name="ECOSTRESS",
        spatial_resolution="70 m",
        temporal_resolution="daily",
        crs="EPSG:4326",
        variables={
            "LST": VariableDefinition("LST", "Land surface temperature", "degC", valid_range=(-273.15, 80), temporal_resolution="daily", spatial_resolution="70 m"),
            "ETinst": VariableDefinition("ETinst", "Instantaneous evapotranspiration", "mm/hr", valid_range=(0, 50), temporal_resolution="daily", spatial_resolution="70 m"),
            "WUE": VariableDefinition("WUE", "Water use efficiency", "g C / kg H2O", valid_range=(0, 100), temporal_resolution="daily", spatial_resolution="70 m"),
            "ESI": VariableDefinition("ESI", "Evaporative stress index", "index", valid_range=(0, 1), temporal_resolution="daily", spatial_resolution="70 m"),
            "QC": VariableDefinition("QC", "Quality control mask", "bitmask", quality_flag=True, temporal_resolution="daily", spatial_resolution="70 m"),
        },
    ),
    "aster_gdem": DatasetDefinition(
        key="aster_gdem",
        title="ASTER Global Digital Elevation Model",
        provider="NASA LPDAAC",
        product_name="ASTER GDEM",
        spatial_resolution="30 m",
        temporal_resolution="static",
        crs="EPSG:4326",
        variables={
            "Elevation": VariableDefinition("Elevation", "Elevation above sea level", "m", valid_range=(-500, 9000), temporal_resolution="static", spatial_resolution="30 m"),
            "Slope": VariableDefinition("Slope", "Terrain slope", "deg", valid_range=(0, 90), temporal_resolution="static", spatial_resolution="30 m"),
            "Aspect": VariableDefinition("Aspect", "Terrain aspect direction", "deg", valid_range=(0, 360), temporal_resolution="static", spatial_resolution="30 m"),
        },
    ),
    "gpm": DatasetDefinition(
        key="gpm",
        title="GPM precipitation",
        provider="NASA GES DISC",
        product_name="GPM IMERG",
        spatial_resolution="10 km",
        temporal_resolution="half-hourly, daily, monthly",
        crs="EPSG:4326",
        variables={
            "DailyPrecipitation": VariableDefinition("DailyPrecipitation", "Daily precipitation accumulation", "mm/day", valid_range=(0, 500), temporal_resolution="daily", spatial_resolution="10 km"),
            "PrecipitationRate": VariableDefinition("PrecipitationRate", "Instantaneous precipitation rate", "mm/hr", valid_range=(0, 200), temporal_resolution="half-hourly", spatial_resolution="10 km"),
            "MonthlyAccumulation": VariableDefinition("MonthlyAccumulation", "Monthly precipitation total", "mm/month", valid_range=(0, 6000), temporal_resolution="monthly", spatial_resolution="10 km"),
        },
    ),
}


def get_dataset_catalog() -> dict[str, DatasetDefinition]:
    """Return the registry of supported NASA datasets."""

    return DATASET_CATALOG


def ensure_required_dataset_keys() -> list[str]:
    """Return the required dataset keys for the project specification."""

    return [
        "hls",
        "modis_ndvi",
        "modis_lai_fpar",
        "modis_lst",
        "modis_et",
        "smap",
        "ecostress",
        "aster_gdem",
        "gpm",
    ]
