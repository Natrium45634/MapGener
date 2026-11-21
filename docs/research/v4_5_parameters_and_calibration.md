# Parameters and calibration of v4.5

The starting values below are engineering assumptions, not constants of nature (TZ 4.5, 11). All of them live in `config.environment` unless marked otherwise.

## Sensitivity (Terra7, 360×240, 75 km)

Only the v4.5 stage is re-run on the same v4 state.

| Parameter | Value | Mean AWC, mm | AET, mm/yr | Runoff, mm/yr | Waterlogged ≥ 2 months | Spin-up converged |
|---|---|---|---|---|---|---|
| `reference_root_depth_m` | 0.5 | 52 | 335 | 355 | 14.9 % | 98 % |
| | **1.0** | **96** | **315** | **373** | **11.2 %** | **98 %** |
| | 2.0 | 154 | 284 | 403 | 7.6 % | 98 % |
| `max_soil_depth_m` | 1.5 | 77 | 326 | 363 | 12.0 % | 98 % |
| | 5.0 | 103 | 313 | 376 | 11.1 % | 98 % |
| `depletion_fraction` | 0.3 | 96 | 303 | 385 | 11.3 % | 97 % |
| | 0.7 | 96 | 327 | 362 | 11.1 % | 99 % |
| `rain_days_per_month` | 3 | 96 | 303 | 390 | 8.8 % | 98 % |
| | 12 | 96 | 316 | 370 | 11.4 % | 98 % |
| `spinup_max_years` | 10 | 96 | 315 | 373 | 11.2 % | **11 %** |
| | 20 | 96 | 315 | 373 | 11.2 % | 98 % |

Precipitation over land is 751 mm/yr in every row. The water residual stays below 1.2e-10 mm.

**Reading the table.**

- A deeper root zone gives *less* AET here. That is the FAO-56 stress law: K_s depends on the relative depletion of the root zone. The same water spread through a deeper zone leaves the cover more stressed whenever the bucket does not fill, which is the common case on this dry-ish world (P/PET of the land about 0.6). The behaviour is intended and documented, not tuned away.
- Ten years of spin-up are too few: groundwater, with a 1-year time scale, needs about 15–20 years to become periodic. The default of 40 is safe; hexes that never become periodic are flagged.
- Hexes where snow keeps accumulating every year are flagged `persistent_snow` and are excluded from the convergence requirement. They make up the remaining 2 %.

## Constants of the modules

| Constant | Value | Where | Basis |
|---|---|---|---|
| rain / snow split | normal CDF around +1 °C, sd 4 °C | `WaterParams` | daily mean spread within a month; typical 0–2 °C thresholds |
| degree-day factor | 3 mm/(°C·day) | `degree_day_mm` | typical 2–6 for snow (HEC-HMS docs) |
| infiltration | ½ Ks × 24 h × 6 rain days | `rain_days_per_month` | monthly integral of daily intake; tunable |
| percolation share | 0.03 + 0.2 log10(1 + Ks mm/h), within [0.02, 0.6] per month | `pipeline_v45` | sand drains fast, clay slowly |
| baseflow | 1 − exp(−days / 365 / τ), τ = 1 yr | `baseflow_timescale_yr` | slow store, as in v4 |
| deep loss | `climate.deep_infiltration` (5 %) of recharge | v4 parameter | an explicit external flux |
| bare rock storage | 5 mm | `WaterParams.rock_storage_mm` | wet surface and fissures (AWC = 0 is a special case) |
| dry month | P < 0.5 PET | `dry_month_ratio` | profile value, not a climate constant |
| GDD base | 5 °C | `gdd_base_c` | common threshold |
| permafrost potential | (F − 0.4) / 0.25, F = frost number | code | F = 0.5 marks the discontinuous / sporadic transition |
| TWI centre of wetness | anomaly 3.9 (10 hexes upstream, 0.2 % slope) | code | a valley floor of the map's scale |
| floodplain width | 20 × channel width W = 5 Q^0.5 | code | floodplain to channel width ratios of ~10–30 |
| texture of parent materials | table in `environment.TEXTURE` | code | typical USDA classes of those soils |

## Checks against references

| Check | Result |
|---|---|
| FAO-56 example 8 (R_a) | 32.19 vs 32.2 MJ m⁻² d⁻¹ |
| Saxton & Rawls table 3 (5 classes) | PWP / FC within ±0.015 m³/m³ |
| USDA AWC example | 120 mm exactly |
| TZ 4.5 §6.2 control month | S 10, GW 4, AET 80, runoff 56, total 150 mm |
| months → annual (Terra7) | precipitation: relative error 5e-16; temperature: 1e-14 °C |
| v4.5 monthly runoff vs v4 annual runoff (Terra7 land) | 378 vs 382 mm/yr (−0.5 %) |
| land water budget (Terra7) | P 751 = AET 310 + runoff 378 + deep loss 14 + snow accumulation 48 mm/yr; residual 1e-10 mm |
