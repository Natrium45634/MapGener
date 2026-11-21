# Baseline audit for TZ 4.5 (Gate 0)

Carried out on the real code of this branch (HexWorld 4.2 + the 4.5 fixes), not on the
TZ's assumptions. The TZ assumed version 3.0 and an unknown state of 4.0; in fact **v4 is
complete**: climate of four seasons with a mass-conservative moisture budget, seasonal hex
water balance, D6 routing with priority flood, lakes, rivers, erosion and sediment, coasts.

## Checks of TZ 4.5, section 1

| Check | Result |
|---|---|
| one current surface `elevation_m`, `sea_level_m` separate | yes: after v4 `elevation_m` is the post-erosion surface; `sea_level_m` is stored with the data |
| land mask, oceans / lakes | `land_mask` (v4 final surface), `lake_id` / `lake_level_m` |
| basins | `basin_id` + `basins[]` with terminal type (sea, window_edge, endorheic) |
| `rock_type`, `rock_resistance` | v3 epoch layers (6 rock types) |
| `sediment_thickness_m` | v3 + v4 deposition |
| monthly / annual T, P, PET, AET, snow, runoff | annual layers + four seasons from moisture transport (v4.2); **no months before 4.5** |
| hex area and neighbourhood | `HexLayout.hex_area_km2`, exactly six neighbours (`raster.neighbor_index`) |
| v1-v4 data not overwritten | v4.5 writes its own files (`environment_*.npz`, `environment.json`) and has its own stage key |

## Inputs read by v4.5

| field | stage | producer | dtype | units | shape | consumer in v4.5 | verified |
|---|---|---|---|---|---|---|---|
| `elevation_m` | v4 | landscape_evolution (post-erosion) | float32 | m | (240, 360) | months (lapse in v4 climate), soils, HAND, aspect | read in a generated Terra7 project; finite on land |
| `land_mask` | v4 | network assembly | bool | bool | (240, 360) | every mask of v4.5 | read in a generated Terra7 project; finite on land |
| `lake_id` | v4 | hydrology | int32 | id | (240, 360) | lakes are water, not soil | read in a generated Terra7 project; finite on land |
| `lake_level_m` | v4 | hydrology | float32 | m | (240, 360) | HAND reference of lakes | read in a generated Terra7 project; finite on land |
| `latitude_deg` | v4 | climate | float32 | deg | (240, 360) | FAO-56 radiation, day length | read in a generated Terra7 project; finite on land |
| `temperature_mean_c` | v4 | climate | float32 | degC | (240, 360) | annual mean kept by the months; weathering | read in a generated Terra7 project; finite on land |
| `humidity_index` | v4 | climate | float32 | P/PET | (240, 360) | weathering, parent material (dry ground) | read in a generated Terra7 project; finite on land |
| `precipitation_mm_yr` | v4 | climate | float32 | mm/yr | (240, 360) | check: months add up to it | read in a generated Terra7 project; finite on land |
| `runoff_mm_yr` | v4 | climate (seasonal bucket) | float32 | mm/yr | (240, 360) | comparison with the v4.5 monthly runoff | read in a generated Terra7 project; finite on land |
| `slope_m_per_km` | v4 | network assembly | float32 | m/km | (240, 360) | soil depth, TWI, colluvium, floodplain | read in a generated Terra7 project; finite on land |
| `flow_receiver` | v4 | hydrology (D6 + priority flood) | int64 | index | (240, 360) | HAND, steps to a river | read in a generated Terra7 project; finite on land |
| `flow_accumulation_area_km2` | v4 | hydrology | float32 | km2 | (240, 360) | TWI (physical area, not hex count) | read in a generated Terra7 project; finite on land |
| `river_mask` | v4 | river_network | bool | bool | (240, 360) | HAND reference, alluvium, floodplain | read in a generated Terra7 project; finite on land |
| `discharge_m3_s` | v4 | hydrology | float32 | m3/s | (240, 360) | floodplain width (hydraulic geometry) | read in a generated Terra7 project; finite on land |
| `endorheic_mask` | v4 | hydrology | bool | bool | (240, 360) | salinity potential, lake deposits | read in a generated Terra7 project; finite on land |
| `basin_id` | v4 | hydrology | int32 | id | (240, 360) | approximate_regional flag (basins at the window edge) | read in a generated Terra7 project; finite on land |
| `external_inflow_m3_s` | v4 | climate entry | float32 | m3/s | (240, 360) | approximate_regional flag | read in a generated Terra7 project; finite on land |
| `sediment_thickness_m` | v4 | landscape_evolution | float32 | m | (240, 360) | loose material (aeolian, loess, lake deposits) | read in a generated Terra7 project; finite on land |
| `v4_deposition_m` | v4 | landscape_evolution | float32 | m | (240, 360) | fresh alluvium | read in a generated Terra7 project; finite on land |
| `lgm_ice_mask` | v4 | coasts | bool | bool | (240, 360) | glacial till, glacial history flag | read in a generated Terra7 project; finite on land |
| `glacial_potential` | v4 | climate | float32 | 0..1 | (240, 360) | no soil under perennial ice | read in a generated Terra7 project; finite on land |
| `coast_type` | v4 | coasts | uint8 | enum | (240, 360) | (not used yet; for coastal sites later) | read in a generated Terra7 project; finite on land |
| `rock_type` | v3 | sample_epoch | uint8 | enum | (240, 360) | bedrock parent material | read in a generated Terra7 project; finite on land |
| `volcanic_age_ma` | v3 | sample_epoch | float32 | Ma | (240, 360) | young volcanic material | read in a generated Terra7 project; finite on land |
| `ClimateLayers.seasons` (t, p, pet) | v4 | `stages.climate.hex_climate` | float32 | degC, mm per season | (4, rows, cols) | the 12 months | new in 4.5: exposed (not stored); recomputed from the world climate when absent |

## Consequences for the design

* The 12 months are **derived from seasonal physics** (`derived_from_seasonal_physics`), not a
  synthetic split of annual values: v4 already has four seasons of moisture transport.
* PET: v4 has only a temperature-index PET (no radiation balance, humidity or wind at 2 m),
  so Penman-Monteith and Hargreaves (needs Tmax/Tmin) are not claimed; the v4 seasonal PET
  is distributed by month temperature and FAO-56 day length.
* Rivers: the v4 network stays the river state; the v4.5 monthly runoff is compared with the
  v4 annual runoff (`summary.runoff_vs_v4`), not silently swapped in.
* No sub-hex coastline exists: `land_fraction` is 1 on land (documented, not faked).
