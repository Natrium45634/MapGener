# TZ 5.4 — baseline audit (G0)

Audit of the repository at `227c300` (HexWorld 1.2.0, generator 5.5.0) before any 5.4
code. Status labels: `verified_existing` (read in code and covered by a test),
`implemented_now` (added by 5.4), `approximation`, `not_available`.

| Input / module | File, real API | Data and units | Test | State before 5.4 | Fallback / decision |
|---|---|---|---|---|---|
| SST by month | `earth/ocean.py::run_ocean -> OceanResult.sst_monthly` (12, ny, nx) on the world raster | °C, connected ocean only | `test_v53.py` T09–T16 | verified_existing, CALIBRATED_APPROXIMATION (amplitude ~1/3 of the Earth's, midlatitudes ~3 K cold) | the atmosphere reads it; the coupled iteration re-runs it with the new winds (one SST) |
| Ocean currents, upwelling, sea ice | `OceanResult.current_u/v` (4 seasons), `upwelling_m_day`, `sea_ice_monthly` | m/s, m/day, 0..1 | T09–T24 | verified_existing | upwelling weakens the TC heat factor; ice switches off evaporation |
| Land / sea / ocean connectivity | `earth/ocean.py::ocean_mask`, hex `ocean_mask`, `lake_id` | bool | v4.6 package A tests | verified_existing | the atmosphere's sea = connected ocean; lakes are not a moisture or TC ocean (T45) |
| 12-month T / P / PET / AET | `pipeline_v45.py::run_env`; `env.monthly_temperature` (first harmonic), `env.monthly_precipitation` (Fourier curve inside the 4 v4 seasons) | °C, mm/month | `test_environment.py` | verified_existing; **months were a smooth profile through 4 seasons** (`derived_from_seasonal_physics`) | replaced by the monthly normals (`atmosphere_5_4_monthly_normals`) |
| Surface wind | `stages/climate.py::seasonal_winds` | m/s, 4 seasons; belts = `lat - shift`, monsoon term down the seasonal T gradient | `test_v4.py` | verified_existing, **prescribed belts** (not from pressure) | 5.4 derives winds from a pressure field with a frictional balance |
| Winds aloft 850/500/250 | `earth/winds.py::thermal_wind` (4 seasons), `wind_at_height` | m/s, m | T25–T32 | verified_existing; equatorial proxy below 12° | kept for v5.3 layers; 5.4 adds 12-month levels and `windAt` |
| Surface pressure | — | — | — | **not_available** | implemented_now (`slp_pa`, `pressure_sfc_pa`, 12 months) |
| Column humidity, IVT | — (v4 moisture `q` is a relative unit of the seasonal loop) | — | — | **not_available** in physical units | implemented_now (`W` kg/m², `Q` kg/(m s)) |
| Orographic redistribution | v4: `RATES["orographic"]` with the 850 hPa wind; hex: `hex_climate.orographic` | relative | `test_v4.py` | verified_existing | 5.4 world grid: `k_oro = eps max(0,u·∇h)/H_v`; hex correction of v4/v4.7 stays (no second orographic rain at the hex scale) |
| Water balance | v4 seasonal loop budget (`WorldClimate.balance`), v4.5 monthly bucket | mm/yr | `test_v4.py` | verified_existing | 5.4: monthly kg ledger per month (`budget_months`) |
| Rivers / snow | v4 routing (`flow_receiver`, `flow_accumulation_area_km2`, `discharge_m3_s`), v4.5 snowpack | m³/s, mm SWE | v4 tests | verified_existing | weather-year flood day is routed down `flow_receiver` |
| Event API v5.1 | `volcanism/*` (events, `MorphologyDelta`) | — | `test_v51.py` | verified_existing | tephra now reads `windAt` (T73) |
| TerrainDelta | `volcanism/terrain.py` | m | T73–T79 (5.3) | verified_existing | not used by weather (no relief change from one year) |
| Biome layers | `pipeline_v50.py` (potential classes + actual ecosystem state) | — | `test_v50.py`, `test_v53_ecology.py` | verified_existing | weather year never rewrites potential biomes (T72) |
| Calendar, planet | `earth/planet.py::Planet`, `MONTH_FRACTION`; `environment.MONTH_DAYS` | days | T69–T72 (5.3) | verified_existing | `circulation.month_days`: Earth calendar or equal months on other planets (T70) |
| Synthetic jet v5.1 | `volcanism/tephra.py::upper_wind_ms` | m/s | `test_v51.py` | verified_existing, labelled synthetic | never returned as "physically computed"; used only without v5.3/v5.4 |
| Monsoon profile v4.5 | `MONSOON` term of `seasonal_winds`, Fourier months | — | — | approximation | not presented as 5.4 output |

Not changed by 5.4 (TZ 0.4): `.seed` and the world RNG streams, flat-top odd-q geometry,
the km world coordinates, `elevation_m`, `sea_level_m`, ocean by connectivity, stored
projects (a project without an `atmosphere` section loads with `enabled=False`), the
PySide6 view registry (one new view, `atmosphere_seasons`).
