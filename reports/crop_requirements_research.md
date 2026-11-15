# Crop Requirements Research

## Result

The curated dataset in [`crop_requirements.json`](../data/reference/crop_requirements.json)
contains general FAO-56 crop water-management guidance for the 22 requested
crops. Every numeric guidance value links to the table it came from and
includes a qualification where the source gives a crop group, regional
example, or management-specific value.

The fields below remain unknown for every crop and are stored as `null`:

- minimum, optimum, and maximum temperature
- precipitation range
- soil pH, texture, and drainage
- crop-specific soil salinity tolerance (`ECe`)

No values are inferred from root depth or water-use coefficients. Pomegranate
has no crop-specific value verified in the reviewed sources. The database
therefore does not support a complete suitability score.

## Interpretation limits

- FAO-56 Table 11 gives crop-stage duration examples by location and planting
  period. These are not universal growth durations or Uzbekistan planting
  recommendations.
- FAO-56 Table 12 gives `Kc` coefficients for non-stressed, well-managed crops
  in the stated subhumid-climate context. `ETc` is estimated using local
  reference evapotranspiration (`ETc = Kc × ETo`); `Kc` is not a seasonal
  irrigation volume.
- FAO-56 Table 22 gives maximum effective rooting depths and depletion
  fractions. The listed `p` values are for approximately 5 mm/day `ETc`;
  FAO-56 explains that `p` varies with atmospheric demand and weather.
- FAO-29 discusses salinity and irrigation-water assessment in general. No
  crop-specific salinity threshold was verified for this dataset, so none is
  encoded.

## Sources

- [FAO Irrigation and Drainage Paper 56, Chapter 6, Table 11 and Table 12](https://www.fao.org/4/x0490e/x0490e0b.htm)
- [FAO Irrigation and Drainage Paper 56, Chapter 8, Table 22](https://www.fao.org/4/x0490e/x0490e0e.htm)
- [FAO Irrigation and Drainage Paper 29 Rev. 1, *Water Quality for Agriculture*](https://www.fao.org/4/t0234e/t0234e00.htm)
- [FAO EcoCrop database](https://ecocrop.fao.org/) is a recommended next source for crop environmental descriptors. Its crop records were not successfully retrieved during this research; no EcoCrop values are included.

The current data should be treated as sourced general guidance only. A
Uzbekistan suitability model still needs locally applicable temperature,
soil, salinity, crop calendar, and variety information before threshold-based
scoring can be justified.
