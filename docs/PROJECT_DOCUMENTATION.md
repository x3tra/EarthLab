# Field Atlas Project Documentation

## Project status

Field Atlas is a local web interface and Python pipeline for collecting,
validating, and exploring NASA AppEEARS raster data and FAO/IIASA GAEZ v5
historical suitability data for agricultural assessment using user-selected
areas. It provides task-scoped acquisition and processing, a curated map
catalog, raster tile rendering, grid statistics, and crop-suitability screening.

GAEZ v5 screening uses its own historical suitability index and official
class products. It is not an agronomically validated local recommendation,
yield forecast, or field-level prediction. The separate environmental-variable
scoring workflow remains guarded because crop-reference data does not provide
locally validated thresholds compatible with its raster inputs. Local task
data may be cleared; do not infer real-data verification from files that were
previously used as temporary samples.

## Repository map

| Area | Main implementation |
|---|---|
| CLI | `main.py` |
| Configuration | `config/settings.yaml`, `config/datasets.yaml`, `.env` |
| AppEEARS authentication and requests | `src/nasa_agri_intelligence/collection/appeears_client.py`, `authentication.py` |
| Product discovery and validation | `src/nasa_agri_intelligence/collection/product_discovery.py` |
| Task creation and request configuration | `src/nasa_agri_intelligence/collection/task_manager.py` |
| Task registry, ownership, path safety | `src/nasa_agri_intelligence/collection/task_registry.py` |
| Monitoring and resumable download | `src/nasa_agri_intelligence/collection/task_monitor.py`, `downloader.py` |
| GAEZ v5 suitability provider | `src/nasa_agri_intelligence/collection/gaez_v5.py`, `data/reference/gaez_v5_crops.json` |
| Raster standardization | `src/nasa_agri_intelligence/standardization.py` |
| Grid and raster harmonization | `src/nasa_agri_intelligence/spatial_harmonization.py` |
| Temporal inventory and aggregation | `src/nasa_agri_intelligence/temporal.py` |
| Native-pixel cell summaries | `src/nasa_agri_intelligence/cell_statistics.py` |
| Curated map catalog | `src/nasa_agri_intelligence/essential_layers.py` |
| Spatial crop workflow and outputs | `src/nasa_agri_intelligence/suitability/spatial.py` |
| Explainable scoring engine | `src/nasa_agri_intelligence/suitability/engine.py` |
| Crop reference data | `data/reference/crop_requirements.json` |
| Web API | `src/nasa_agri_intelligence/web.py` |
| Browser application | `frontend/src/App.tsx`, `MapView.tsx`, `style.css` |
| Tests | `tests/` |

The browser navigation currently includes Map, Data, Tasks, and Crop
Suitability. It does not have a separate Dashboard page.

## Curated map catalog

The API constructs the curated catalog from definitions in
`essential_layers.py` and saved, verified AppEEARS metadata. The catalog has
ten logical entries; it does not expose every discovered product, QA band, or
intermediate raster as an ordinary layer.

| Logical layer | Variables shown within the group | Source and saved metadata |
|---|---|---|
| NDVI | NDVI | MODIS MOD13Q1.061, `_250m_16_days_NDVI`; 250 m, 16-day cadence, scale factor 0.0001 |
| Land Surface Temperature | Daytime and nighttime LST | MODIS MOD11A2.061, `LST_Day_1km` and `LST_Night_1km`; 1 km, 8-day cadence, scale factor 0.02, Kelvin |
| Evapotranspiration | Actual ET | MODIS MOD16A2GF.061, `ET_500m`; 500 m, 8-day cadence, scale factor 0.1 |
| Potential Evapotranspiration | Potential ET (PET) | MODIS MOD16A2GF.061, `PET_500m`; 500 m, 8-day cadence, scale factor 0.1 |
| Soil Properties | pH, sand, silt, clay, organic carbon | ISRIC SoilGrids 2.0 WCS, median prediction; 250 m; static; retrieved for the selected task AOI |
| Elevation | Elevation | ASTER GDEM `ASTGTM_NC.003`, `ASTER_GDEM_DEM`; 30 m, static |
| Slope | Slope | Derived in degrees from the selected task's ASTER GDEM during standardization; needs valid 3×3 DEM neighborhoods |
| Land Cover | IGBP land-cover class | MODIS MCD12Q1.061, `LC_Type1`; 500 m, yearly |
| Air Temperature | Daily 2 m mean, minimum, and maximum | NASA POWER daily regional data (`T2M`, `T2M_MIN`, `T2M_MAX`; MERRA-2); 0.5° × 0.625° grid; collected on demand for task dates |
| Precipitation | Corrected daily precipitation | NASA POWER daily regional data (`PRECTOTCORR`, MERRA-2); 0.5° × 0.625° grid; daily values are accumulated into task-period, monthly, and complete-season totals |

Resolution and cadence above are from saved product metadata and are not a
promise of coverage for any particular task or AOI. A group can report:

- **Requestable, not downloaded**: the saved product metadata lists the
  product/layer as available.
- **Processed on selected task**: a task-owned standardized manifest lists
  the raster.
- **Available on demand**: a verified external source is fetched only after
  an explicit request and is written to the selected task.
- **Derived from task data**: the layer is generated from a validated raster
  in the selected task.
- **Unavailable**: no integrated source or usable task data is available.

Even processed status does not guarantee valid data at every AOI location.
Users should inspect the selected task's coverage and quality before
interpretation. Scale factors, offsets, fill values, validity ranges, units,
temporal extent, and QA dependencies are displayed when the corresponding
saved product metadata is present. QA bands are kept separate from ordinary
map choices. Documented per-product rules filter matching, grid-aligned
MOD13Q1, MOD16A2GF, and MCD12Q1 science rasters during task standardization.
Missing or unsupported QA is recorded explicitly; MOD11A2 data in the sample
task remain unfiltered because QC_Day/QC_Night were not acquired.
The ASTER GDEM scene-count band is recorded as auxiliary coverage information,
not decoded or treated as a QA mask.

The current map catalog entries are for visualization, not a declaration of
crop-model inputs. Hiding a visualization layer does not change processing
inputs. NASA POWER temperature and precipitation are retrieved from its daily
regional API only when requested for a preprocessed task; their coarse grid
does not add local field-scale detail. Temperature summaries average valid
daily values, while precipitation summaries sum valid daily amounts. Monthly
and seasonal products retain their dates and aggregation metadata, and are
created only when the task covers the full period.

Soil variables are retrieved on demand from the official ISRIC SoilGrids 2.0
WCS, clipped to the selected task AOI, and written to task-owned storage before
standardization. The provider offers six static depth intervals
(`0-5cm`, `5-15cm`, `15-30cm`, `30-60cm`, `60-100cm`, and `100-200cm`);
the endpoint defaults to `0-5cm`. pH source values are scaled from pH × 10,
texture values are converted from g/kg to percent, and soil organic carbon is
reported in g/kg. Provenance records include depth, quantile, source, units,
and retrieval metadata. Live acquisition and standardization were verified
for an Uzbekistan AOI around Tashkent (69.27–69.29°E, 41.30–41.32°N).
This verifies the sampled AOI, not every location or soil depth in Uzbekistan.

NASA POWER live data acquisition was also verified over the same AOI for
2024-03-01 through 2024-05-31, including `T2M`, `T2M_MIN`, `T2M_MAX`, and
`PRECTOTCORR`; generated task-period, monthly, and complete MAM rasters were
standardized and registered in the temporary task manifest. NASA POWER
precipitation is not irrigation availability. The climate and soil APIs are:

```bash
curl -X POST 'http://127.0.0.1:8000/api/tasks/TASK_ID/climate'
curl -X POST 'http://127.0.0.1:8000/api/tasks/TASK_ID/soil?depth=0-5cm'
```

`TASK_ID` must identify a preprocessed task with its AOI and date interval.
The soil endpoint accepts any of the six listed depth intervals. Existing
validated task-owned downloads are reused.

The spatial suitability workflow currently rejects analysis: crop references
do not provide locally validated suitability thresholds for the available
environmental rasters. Consequently, there is no scientifically defensible
set of required scoring variables yet. Map visibility is independent of
analysis inputs, and hiding a layer does not remove a required input from a
future validated model.

## Task isolation and storage

All new AppEEARS tasks are represented under `data/tasks/`:

```text
data/tasks/
├── task_registry.json
└── TASK_NAME/
    ├── task_metadata.json
    ├── task_manifest.json
    ├── raw/
    ├── processed/
    │   ├── rasters/
    │   ├── standardized_manifest.json
    │   ├── cell_statistics/
    │   └── temporal/
    └── suitability/
        └── runs/
            └── ANALYSIS_NAME/
```

The registry maps a unique human-readable task name to its actual NASA task
ID and local directory. Requests can generate a name or provide one; duplicate
names are rejected instead of reused or overwritten. A NASA ID is persisted
after a successful submission. One NASA collection task corresponds to a
product selection in the current submission flow.

Task manifests record the AOI and CRS, requested products/layers, date range,
NASA ID, download/preprocessing status, local paths, and analysis history.
Before processing, the registry checks the manifest identity, selected task
ID, source directory, expected product/layers, raster existence/readability,
and resolved paths. Raster paths must remain within the selected task's
directory. Standardization and analysis read only files referenced by the
selected task's validated manifests; there is no legacy global-manifest
fallback in these workflows.

Legacy files in `data/raw/` or `data/processed/` are not moved or assigned to
a task automatically. The inventory utility can report on legacy files, but
its output is not an input source for task processing.

Analysis is distinct from data collection. Repeated analyses create
`suitability/runs/ANALYSIS_NNN/` directories by default. A caller can provide
an analysis name. Existing outputs are not overwritten unless overwrite is
explicitly selected.

## Raster processing and spatial support

1. **Discover and select products.** Product and layer metadata is queried
   from AppEEARS and stored locally. `config/datasets.yaml` identifies
   configured products and exact layer IDs.
2. **Submit and monitor.** An AOI is validated as polygonal EPSG:4326
   GeoJSON. The task request stores dates and the requested output settings.
3. **Download and validate.** Bundle data is written beneath the owning
   task's `raw/` directory. Downloads use partial files and resume only if the
   server confirms a byte-range request; checksums, sizes, and supported file
   formats are validated.
4. **Standardize.** AppEEARS scale/offset and fill/valid-range metadata are
   applied while preserving native CRS, transform, resolution, and masks.
   QA rasters remain categorical and unscaled. Documented product-specific
   rules write task-scoped filtered derivatives and retain the original
   standardized data. Missing, unsupported, or misaligned companions are
   never treated as accepted data.
5. **Harmonize (optional).** `spatial_harmonization.py` builds an AOI-bounded
   projected grid and reprojects rasters using semantic resampling: nearest
   neighbor for categorical/quality data, and averaging or bilinear methods
   for continuous data. Upsampling is rejected unless explicitly allowed.
   The original raster remains untouched.
6. **Summarize (optional).** Cell statistics assign native pixel centers to
   AOI grid cells without resampling. Coverage is a native-sample coverage
   statistic, not exact area coverage.
7. **Inspect or map.** Raster tiles are generated by the API on demand; the
   full GeoTIFF is not loaded into the browser.

The configured cell size is an output sampling grid, not a statement of
information content. A finer grid does not add detail beyond the native
source pixels. Different products can have different resolutions, dates, and
valid-data footprints.

## Suitability scoring status

### GAEZ v5 historical screening

The GAEZ provider is task-scoped: its outputs, provider operations, and run
metadata belong to the selected AOI task. The task-creation form offers GAEZ
as an acquisition option and requests the verified catalog for both water
regimes at high-input management without creating another application-level
task. The Crop Suitability page can also start a run for an existing task
with a chosen crop subset, water regime, and management level.

The crop catalog contains 77 crop codes verified in the public GAEZ v5
inventory for period `HP0120` (2001–2020), climate `AGERA5`, and scenario
`HIST`, using the continuous `RES05-SXX30AS` and official class
`RES05-SIX30AS` products. Sixty-four crops have all four high/low-input and
rain-fed/irrigated scenarios in the checked inventory; 13 have only the
high-input rain-fed and irrigated scenarios. The catalog records each crop's
official code, category, scenario codes, and availability status. Twelve
proposed crops (apple, pear, peach, apricot, cherry, lentil, cucumber, pepper,
eggplant, grapes, pomegranate, and date palm) are listed as unavailable
because no matching crop product was found in that inventory snapshot. These
are source-inventory findings, not claims that GAEZ never models those crops.
Only catalogued product/scenario combinations are ranked; absent products are
reported as unavailable, never assigned a fabricated score.

The verified products use a 30 arc-second source grid in EPSG:4326. The
continuous suitability index is preserved on its 0–10,000 range with scale
factor 1 and `-9` as no-data; class values use codes 1–9, scale factor 1, and
`0` as no-data. These values and class captions are sourced from the
[official SXX metadata](https://data.apps.fao.org/catalog/dataset/3b81fb8a-af97-43c8-b3f0-9d0393557878)
and [official SIX metadata](https://data.apps.fao.org/catalog/dataset/94c469f7-3352-457e-8025-9bae98982109).
The class product is not a substitute for the continuous `SXX` index.

For each raster, the provider uses task AOI overlap areas to compute
area-weighted statistics, excludes no-data, reports valid-area coverage, and
preserves the original cropped source resolution. Rankings are separate for
rain-fed and irrigated scenarios and each management level. The configured
minimum valid-data coverage is an operational completeness criterion, not a
scientific crop threshold. Coverage is reported both as valid SXX area over
the whole AOI and as valid SXX area over non-water AOI area. An observed
SXX value of `-1` is excluded as water only when the matching SIX cell is
official class 9; unpaired out-of-range values are processing validation
failures, not silently recoded as no-data. The class distribution is computed
independently from SIX and reports class area and share of the whole AOI, as
well as share of classified area. If SIX is unavailable, the SXX ranking is
retained and class data is explicitly marked unavailable.

The browser crop catalog is searchable and grouped by category. It shows
validated availability by selected management/water regime and links to the
GAEZ source. Exact AOI coverage is only known after the requested source
windows are processed, so the selector labels it as pending; results report
the actual coverage, official class distribution, and exclusion reasons.
Rankings are separate by water regime and management level, and the historical
period and product identifiers are shown with each result. Equal indices
share a competition rank. A 95% AOI-coverage requirement is an operational
completeness rule, not a crop suitability threshold.

Irrigated suitability does not assert local water availability. The UI
displays the exact water regime, management code/level, historical period,
product mapsets, cell counts, coverage basis, class areas, attribution, and
map layers; NASA observations and legacy environmental scoring remain
separate and are not combined into an unsupported composite suitability
score. GAEZ is a long-term modeled indicator, not a success probability or
yield guarantee.

GAEZ v5 is a global product, but crop-specific source coverage is checked for
each AOI at run time. A global product label does not guarantee valid pixels
in a selected area. The processing code excludes NoData and water, reports
AOI coverage, and does not rank crops below the coverage threshold or for
missing scenario products. Implementation and synthetic rasters do not
establish that live GAEZ objects were successfully retrieved for a particular
user AOI; verify each actual run before interpreting it.

AOIs are submitted as valid EPSG:4326 polygon GeoJSON with longitude and
latitude bounds checked by the API. The default analysis grid CRS is the
global equal-area EPSG:6933 rather than a regional UTM zone; callers can still
choose another projected CRS in metres. The map opens at a global view and
does not impose a country-specific AOI. Coverage remains provider- and
product-dependent: NASA AppEEARS products, SoilGrids, NASA POWER, and GAEZ
must be checked independently for the requested AOI, and missing
supplementary data are reported rather than treated as unsuitable conditions.
Antimeridian-spanning rings are rejected rather than interpreted as polygons
across most of the world. Polar AOIs beyond the supported EPSG:6933 area of
use are rejected rather than transformed with clipped coordinates. These
documented edge restrictions mean the application should not be described as
supporting every geometry on the globe.

`suitability/engine.py` evaluates crop criteria and returns scores,
observations, missing parameters, limiting factors, and a best crop where
supported. `suitability/spatial.py` resolves one registered task, reads only
its `processed/standardized_manifest.json`, samples raster values at AOI-grid
cell centers, and would write:

- a best-crop raster;
- a confidence raster;
- a continuous score raster per selected crop;
- a manifest with per-cell crop scores, missing parameters, and limiting
  factors.

However, the current crop reference data primarily contains irrigation
guidance (for example crop coefficients, season length, root depth, and
depletion fraction), not locally validated environmental suitability
thresholds for the raster variables currently sampled. The workflow checks
this mismatch and refuses the unsupported crop analyses. No meaningful real
AOI suitability rasters have been produced or displayed. Existing single-cell
tests validate mechanics only, not agronomic validity.

Do not use the prototype `/api/suitability/evaluate` endpoint with invented or
manually entered environmental values as evidence of spatial analysis. A
scientifically supported spatial recommendation requires verified
crop-specific thresholds, suitable task-owned input rasters, quality rules,
and validation against local agronomic evidence.

## Running the project

### Install

The project requires Python 3.12 or newer:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev,web]"
npm --prefix frontend install
```

Copy `.env.example` to `.env`. Configure either `NASA_APP_EARS_TOKEN` or both
`EARTHDATA_USERNAME` and `EARTHDATA_PASSWORD` for authenticated NASA
operations. Do not expose credentials to the browser or commit them.

### Start API and frontend

In separate terminals, from the project root:

```bash
.venv/bin/python -m uvicorn nasa_agri_intelligence.web:app --host 127.0.0.1 --port 8000
npm --prefix frontend run dev
```

Open the Vite URL (normally `http://127.0.0.1:5173`). Keep the API bound to
localhost; this development API does not provide user authentication. Set
`VITE_API_BASE_URL` when the API uses a different port. If the browser reports
that `/api/essential-layers` is missing, it is likely connected to an older
API process; restart the backend from this project directory.

### Collect data and process a task

Validate configured products before submitting:

```bash
.venv/bin/python main.py validate
```

Submit an explicitly configured product. This contacts NASA and creates a
real remote task:

```bash
.venv/bin/python main.py tasks submit --product MOD13Q1.061 --task-name MY_TASK
```

Omit `--task-name` to generate a unique name. Inspect tasks and check status:

```bash
.venv/bin/python main.py tasks list
.venv/bin/python main.py tasks inspect MY_TASK
.venv/bin/python main.py tasks status NASA_TASK_ID
.venv/bin/python main.py tasks monitor NASA_TASK_ID
```

After AppEEARS reports completion:

```bash
.venv/bin/python main.py download NASA_TASK_ID
.venv/bin/python main.py validate-downloads NASA_TASK_ID
.venv/bin/python -m nasa_agri_intelligence.standardization --task-name MY_TASK
.venv/bin/python main.py spatial stats --task-name MY_TASK
.venv/bin/python -m nasa_agri_intelligence.temporal --task-name MY_TASK
```

The browser's Tasks/Data workflow offers task creation, status checks,
downloads, and standardization. Avoid creating a real AppEEARS request solely
for tests; standard tests use temporary directories and mocks.

The spatial suitability CLI is exposed, but currently rejects crop analyses
without supported environmental criteria:

```bash
.venv/bin/python main.py suitability analyze --task-name MY_TASK
```

Do not interpret the command's existence as evidence that the project can
currently produce a valid crop recommendation.

## Web API reference

The API is implemented in `web.py`; task-specific endpoints require an
explicit task name or NASA task ID as shown.

| Endpoint | Purpose |
|---|---|
| `GET /api/health` | Local service health |
| `GET /api/settings` | Analysis defaults and configured AOI |
| `GET /api/datasets` | Configured/available product summaries |
| `GET /api/essential-layers?task_name=...` | Curated ten-entry catalog and task-specific availability |
| `GET /api/datasets/{product_id}` | Product metadata |
| `POST /api/aoi/statistics` | AOI geometry/grid estimate |
| `GET /api/crops` | Crop reference names |
| `GET /api/tasks?include_remote=false` | Local tasks; optionally merge remote task list |
| `GET /api/tasks/{NASA_task_id}` | Registered task manifest |
| `GET /api/task-names/{task_name}` | Task manifest by registered name |
| `GET /api/gaez-v5/catalog` | Verified GAEZ v5 product, crop, scenario, and legend metadata |
| `POST /api/aoi-tasks` with `gaez_v5: true` | Create one AOI task and run the default GAEZ provider operation within it |
| `POST /api/aoi-tasks/{task_id}/gaez-suitability` | Start a task-scoped GAEZ crop/scenario run |
| `GET /api/aoi-tasks/{task_id}/gaez-suitability?run_id=...` | Retrieve that task's GAEZ results |
| `POST /api/tasks` | Submit a task; request requires explicit real-submission confirmation |
| `POST /api/tasks/{NASA_task_id}/monitor` | Refresh AppEEARS status |
| `POST /api/tasks/{NASA_task_id}/download` | Download completed bundle |
| `POST /api/tasks/{NASA_task_id}/standardize` | Standardize that task's validated downloads |
| `GET /api/layers?task_name=...` | Task-owned curated raster layer metadata |
| `GET /api/layers/{layer_id}/tiles/{z}/{x}/{y}.png?task_name=...` | On-demand raster tile |
| `GET /api/layers/{layer_id}/value?task_name=...&lat=...&lon=...` | Sample one task-owned raster |
| `POST /api/tasks/{NASA_task_id}/climate` | Fetch NASA POWER daily air temperature and precipitation for the selected preprocessed task |
| `POST /api/suitability/analyze` | Run guarded task-scoped suitability analysis |
| `GET /api/suitability/results?task_name=...` | Results for the task and optional run |
| `GET /api/suitability/cell?task_name=...&lat=...&lon=...` | Inspect a cell in a task-owned run |

Task ownership is validated by the backend, not just by frontend filtering.
Task switching reloads catalog/layers using the selected task. The tile
endpoint uses the corresponding processed raster and selects nearest-neighbor
sampling for categorical/quality data and bilinear sampling for continuous
data.

## Testing and verification

Run the backend tests:

```bash
.venv/bin/pytest -q
```

Build and type-check the frontend:

```bash
npm --prefix frontend run build
```

The suite covers mocked task collection, task ownership, path-escape
rejection, manifest validation, resumable downloads, raster standardization,
grid statistics, scoring behavior, GAEZ area-weighted calculations, one-task
AOI orchestration, and web/API selection. GAEZ unit tests use synthetic
rasters for controlled algorithm checks; those are not real analysis outputs.

## Known gaps and scientific cautions

- GAEZ suitability is not yet locally validated against field trials, yield,
  or current crop production outcomes; its historical period is 2001–2020.
- A GAEZ map pixel is a 30-arc-second source cell, not a field-scale estimate.
- Real GAEZ provider retrieval and processing must be verified for the target
  AOI; synthetic tests alone are insufficient evidence.
- NASA GPM IMERG Final Daily V07 is a verified global alternative precipitation
  product (0.1°, daily, data available from 1998), but its March 1, 2024
  granule download was not completed: the official GES DISC URL redirected to
  Earthdata Login and returned HTTP 401. It is not integrated; NASA POWER is
  the currently working precipitation source.
- ERA5-Land was investigated as a higher-resolution hourly alternative
  (~9 km native grid, available from 1950), but is not integrated. CDS access
  requires its authenticated API and credentials, which were not configured
  for this verification. NASA POWER remains the available, coarser air
  temperature source.
- The legacy spatial scoring route still rejects crop analyses when
  scientifically validated environmental criteria are unavailable.
- Crop-reference guidance is not equivalent to crop-environment response
  functions and is not locally validated for arbitrary AOIs.
- ET and PET are not irrigation water supply. LST is not air temperature.
- A common output grid, dense display, or interpolated tile is not new
  scientific information and must not be presented as such.
- The local development web API has no authentication; keep it on localhost.
