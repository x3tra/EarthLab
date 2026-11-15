# Local Satellite Data Inventory

- Generated: 2026-11-10T14:52:32.657452+00:00
- Data directory: `/home/max-s/programming/space/data`
- Files inspected: 11
- Files passing integrity and supported format checks: 10
- Files needing attention or without a manifest: 1
- Manifest-listed files missing from disk: 0

## Collected products

### MOD13Q1.061

- Product: MOD13Q1
- Files: 11
- Catalog status: available
- Layer `_250m_16_days_NDVI`: 2 file(s); units `NDVI`, scale 0.0001, offset 0.0, valid range [-0.200000003, 1.0], NoData -3000; dates 2025-06-26, 2025-07-12.
- Layer `_250m_16_days_VI_Quality`: 2 file(s); units `not documented`, scale not documented, offset , valid range [0.0, 65534.0], NoData 65535; dates 2025-06-26, 2025-07-12.
  - `data/raw/MODIS/MOD13Q1/MOD13Q1-061-250m-16-days-VI-Quality-Statistics-QA.csv`: csv file, 189 bytes, 2 data row(s), 4 column(s); format valid.
  - `data/raw/MODIS/MOD13Q1/MOD13Q1-061-250m-16-days-VI-Quality-lookup.csv`: csv file, 256 bytes, 1 data row(s), 10 column(s); format valid.
  - `data/raw/MODIS/MOD13Q1/MOD13Q1-061-Statistics.csv`: csv file, 511 bytes, 2 data row(s), 16 column(s); format valid.
  - `data/raw/MODIS/MOD13Q1/MOD13Q1.061_2025167_to_2025197/MOD13Q1.061__250m_16_days_NDVI_20250626T000000_aid0001.tif`: 1×1, 1 band(s), ['int16'], resolution [231.6563582625, 231.6563582625], CRS `+proj=sinu +lon_0=0 +x_0=0 +y_0=0 +R=6371007.181 +units=m +no_defs=True`, valid/masked pixels 1/0; validation valid.
  - `data/raw/MODIS/MOD13Q1/MOD13Q1.061_2025167_to_2025197/MOD13Q1.061__250m_16_days_NDVI_20250712T000000_aid0001.tif`: 1×1, 1 band(s), ['int16'], resolution [231.6563582625, 231.6563582625], CRS `+proj=sinu +lon_0=0 +x_0=0 +y_0=0 +R=6371007.181 +units=m +no_defs=True`, valid/masked pixels 1/0; validation valid.
  - `data/raw/MODIS/MOD13Q1/MOD13Q1.061_2025167_to_2025197/MOD13Q1.061__250m_16_days_VI_Quality_20250626T000000_aid0001.tif`: 1×1, 1 band(s), ['uint16'], resolution [231.6563582625, 231.6563582625], CRS `+proj=sinu +lon_0=0 +x_0=0 +y_0=0 +R=6371007.181 +units=m +no_defs=True`, valid/masked pixels 1/0; validation valid.
  - `data/raw/MODIS/MOD13Q1/MOD13Q1.061_2025167_to_2025197/MOD13Q1.061__250m_16_days_VI_Quality_20250712T000000_aid0001.tif`: 1×1, 1 band(s), ['uint16'], resolution [231.6563582625, 231.6563582625], CRS `+proj=sinu +lon_0=0 +x_0=0 +y_0=0 +R=6371007.181 +units=m +no_defs=True`, valid/masked pixels 1/0; validation valid.
  - `data/raw/MODIS/MOD13Q1/README.md`: md file, 39672 bytes, starts: '# AppEEARS Area Sample Extraction Readme'; format valid.
  - `data/raw/MODIS/MOD13Q1/Uzbekistan-MOD13Q1-NDVI-test-MOD13Q1-061-metadata.xml`: xml file, 22038 bytes; format CHECK; issue(s): not well-formed (invalid token): line 197, column 59.
  - `data/raw/MODIS/MOD13Q1/Uzbekistan-MOD13Q1-NDVI-test-granule-list.txt`: txt file, 324 bytes, starts: 'https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/MOD13Q1.061/MOD13Q1.A2025177.h23v04.061.2025195141345/MOD13Q1.A2025177.h23v04.061.2025195141345.hdf'; format valid.
  - `data/raw/MODIS/MOD13Q1/Uzbekistan-MOD13Q1-NDVI-test-request.json`: json file, 1066 bytes; format valid.

## Collection task records

- `4d12ee1d-df48-4bbb-9d38-1b8038dff491` — MOD13Q1.061, completed/completed, 11 bundle files, requested 2025-07-01 to 2025-07-16.

## Catalog context

Saved AppEEARS catalog snapshot: 193 catalog products; generated 2026-11-09T10:15:21.878890+00:00.
- HLSL30: available; available product IDs: HLSL30.020.
- HLSS30: available; available product IDs: HLSS30.020.
- MOD13Q1: available; available product IDs: MOD13Q1.061.
- MCD15A3H: available; available product IDs: MCD15A3H.061.
- MOD11A2: available; available product IDs: MOD11A2.061.
- MOD16A2GF: available; available product IDs: MOD16A2GF.061.
- SMAP: available; available product IDs: SPL3SMP_E.006, SPL3SMP.009.
- ECOSTRESS: available; available product IDs: ECO_L2T_LSTE.002, ECO_L3T_ET_ALEXI.002.
- ECOSTRESS_ET_ALEXI: available; available product IDs: ECO_L3T_ET_ALEXI.002.
- ASTER: available; available product IDs: ASTGTM_NC.003.
- GPM: unavailable; available product IDs: none.
- MCD12Q1: available; available product IDs: MCD12Q1.061.

## Interpretation limits

- Raster validity and manifest integrity do not establish that the sample is representative of a larger region.
- Resampling a single pixel does not create additional spatial information.
- Scale factors and units shown here come from saved AppEEARS product metadata; quality-mask decoding and scientific suitability analysis are separate steps.
- All 4 currently available raster files contain one pixel each; this test collection cannot support spatial suitability mapping.
- Only 2 raster acquisition date(s) are available; this is insufficient for seasonal or historical trend analysis.
