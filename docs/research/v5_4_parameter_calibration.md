# TZ 5.4 — parameters and calibration ([H] = calibrated, configurable in code)

| Parameter | Value | Meaning / reason | Checked by |
|---|---|---|---|
| `circulation.TAU_LAND_D`, `TAU_COAST_D`, `TAU_SEA_D` | 30, 50, 60 d | one-box lag: land warmest ~1 month after the solstice, sea ~2 | T03, T11 |
| Hadley edge | 30° × sqrt(ΔT/50 K) × (Ω_E a_E)/(Ω a) | Held–Hou scaling calibrated to the Earth | T14 |
| zonal SLP belts | trough −6, subtropical high +11, subpolar low −18, polar high +6 hPa | Earth-like amplitudes; scaled by `climate.wind_strength` | T18 |
| `LAND_SUMMER_HIGH_RELIEF` | 0.8 | over warm land in its summer the subtropical high gives way to the heat low (summer highs are oceanic cells) | Terra7 land 30–45°: July rain 30 → 65 mm |
| `LAND_HEAT_TRIGGER` | 0.3 | summer heating of the land lowers the convective RH threshold | same |
| `THERMAL_HPA_PER_K` | 1.0 hPa/K | heat lows / winter highs from the regional T anomaly (Siberia ≈ +20 hPa) | sensitivity below |
| `THICKNESS_HPA_PER_K` | 0.6 | hypsometric thickness at 850 hPa (ρ R ln(1013/850)) | T15 |
| friction `r` | sea 2.4e-5, land 4.5e-5 (×(1+h/1.5 km)), 850 hPa 1e-5, equatorial +5e-5 | cross-isobar angle ~15–25°, trades 5–8 m/s, doldrums | T13, T18, T19 |
| `H_V` | 2200 m | vapour scale height: W(27 °C) ≈ 57 kg/m², W(0 °C) ≈ 11 | — |
| `C_E` | 1.2e-3 | bulk transfer coefficient (neutral, 10 m) | T24 |
| `U_GUST`, `U_STORM` | 4 m/s, 5 m/s × baroclinicity | mean speed of a variable wind > speed of the monthly mean vector | evaporation by latitude |
| `RH_BL_A`, `RH_BL_B` | 0.5, 0.45 | surface RH = a + b × column RH (moist boundary layer under a dry column) | — |
| `TAU_BG_D`, `RH_POWER` | 6 d, 2 | large-scale rain RH² / τ | deserts below |
| `TAU_CONV_D`, ascent floor | 7 d, 0.15 | convection, suppressed but not zero under mean subsidence | T25 |
| `TAU_FRONT_D` | 5 d | frontal rain with baroclinicity ≥ 8 K/1000 km | — |
| `EPS_ORO`, cap | 0.5, 1/(6 h) | condensed share of the forced ascent | T26 |
| `LATENT_HPA_PER_MM_DAY` | 0.5 hPa per mm/day | trough under convective rain (one feedback pass) | latent test |
| `SOIL_MEMORY` | 0.3 | dry-season land ET partly from the annual supply | — |
| `storms.LAMBDA0` | 2e-19 m⁻² s⁻¹ | ~tens of TC / yr for an Earth-like tropical ocean | N09 (formula) |
| TC intensity | a = 1/36 h, shear 1/48 h per 10 m/s, land 1/18 h, PI = 35 + 12 (SST − 26) | weakening at landfall, shear | T43, T47 |
| `weather.ETC_RATE` | 2.3e-18 m⁻² s⁻¹ per unit activity | ~2500 ETC / yr on an Earth-sized planet | — |
| `WINDTHROW_MAX`, Holland R_max | 0.6, 40 km | stand share thrown at ≥ 55 m/s; land wind 0.75 of gradient wind | windthrow tests |
| Markov persistence r | 0.35 | p_WW − p_DW | T61 |
| Gamma k | 0.75 | wet-day amounts | T62 |
| daily correlation | 600 km, AR(1) 0.6 | regional weather states | T63 |
| interannual | AR(1) ρ = 0.6 (ocean mode), persistence ρ = 0.4 | | T65, T69 |

## Measured calibration: aridity (4 worlds, window 160×110 hexes × 60 km)

`P_land` mm/yr, desert = share of land with P/PET < 0.2 (Earth ≈ 20 %).

| World | v4 (old) | 5.4 `full` |
|---|---|---|
| Terra7 | 972 mm, 23 % | 748 mm, 35 % |
| Kq9 | — | 662 mm, 26 % |
| M4p | 1835 mm, 16 % | 880 mm, 45 % |
| Zz31 | 1049 mm, 0 % | 852 mm, 6 % |

The world-raster (whole simulation area) distributions of the two models are close (e.g.
M4p desert 28 % vs 32 %); the difference is concentrated in subtropical windows, where the
old model's land-ITCZ shift kept the land very wet (M4p window: mean 1835 mm). Varying C_E
(1.2–1.4e-3), the ascent floor (0.15–0.35), the thermal-low strength (1.0–1.6 hPa/K) and
the grid resolution changed the window desert share by ≤ 3 points: the drier subtropics are
a structural property of the pressure-driven circulation, not one parameter. Because worlds
had previously been calibrated against over-dry continents (CHANGELOG, «пересохшие
материки»), the default is `seasonality_only` (v4 annual totals, 5.4 monthly course);
`full` is the physically complete option. This is an open calibration item.

## Rain regimes: Mediterranean test

The winter-wet regime follows Köppen `Cs` (driest summer month < 40 mm and < 1/3 of the
wettest winter month, 25–48°). Before the summer-high relief and the land trigger the
Terra7 window classified 55 % of its land as Mediterranean (summer continents were dry
under a zonal subtropical high); after: 33 % of the 30–40° band, 9 % of 20–30°, 14 % of
40–50°.
