# Evidence matrix for TZ 4.5 (section 13.4)

Status levels (TZ 4.5, 13.4.4):
- **described**: stated in a publication or in the project documentation;
- **checked in the repository**: the source file was read at the pinned commit;
- **checked in HexWorld**: a HexWorld test reproduces the number or behaviour.

How the repositories were read: shallow clones into a scratch directory outside the repository, read as text and **never built, imported or run**. No code or data was copied. Accessed 21.11.2026.

## Priority A

| Source | Commit / version | License | What was read | Claim | Status | Decision |
|---|---|---|---|---|---|---|
| FAO-56, ch. 3 (eq. 21-25, example 8) | 1998 publication | FAO terms | equations of R_a, δ, ω_s, N | R_a(20° S, J = 246) = 32.2 MJ m⁻² d⁻¹ | **checked in HexWorld** (`test_fao56_extraterrestrial_radiation_and_polar_cases`: 32.19) | adopted: `environment.extraterrestrial_radiation`; polar day and night by clipping |
| FAO-56, ch. 8 (root-zone balance, eq. 84; table 22) | 1998 | FAO terms | K_s, TAW/RAW, p | K_s = (TAW − D_r)/((1 − p) TAW); p adjusted by ET | described; same form in SWB (below) | adopted: stress-limited AET in `month_step`, with the p adjustment |
| [pyet-org/pyet](https://github.com/pyet-org/pyet) | `a97c97cf4852` (2026-05-26) | MIT | `pyet/radiation.py` (`hargreaves(tmean, tmax, tmin, lat, ...)`), `pyet/combination.py` (`pm_fao56`), `pyet/temperature.py` | Hargreaves needs T_max and T_min; Penman-Monteith needs R_n, wind and humidity | **checked in the repository** (function signatures) | **declined as a runtime dependency**; both methods declined for want of inputs (HexWorld has no diurnal range, no radiation balance, no 2 m wind) |
| [Ouranosinc/xclim](https://github.com/Ouranosinc/xclim) | `8250905da783` (2026-11-19) | Apache-2.0 | listing of `src/xclim/indices/` | GDD and frost days are defined on **daily** series | described (docs) + listing checked | adopted as a rule: only `*_proxy` from monthly means; `frost_days_exact` not calculated |
| USDA NRCS AWC note | 2022 PDF | public | AWC definition | AWC = depth × (FC − PWP) × (1 − coarse) | **checked in HexWorld** (`test_available_water_control_and_bare_rock`: 120 mm) | adopted |
| Saxton & Rawls 2006 (SSSAJ 70:1569) | publication | journal | equations 1-3, 5, 16 and table 3 | PWP / FC of 12 texture classes | **checked in HexWorld** (`test_saxton_rawls_reproduces_the_paper_table`: 5 classes within ±0.015) | adopted, with a texture-table fallback when the result is outside physical limits |
| [DOI-USGS/swb2](https://github.com/DOI-USGS/swb2) | `203736e2f22b` (2026-08-27) | public domain (LICENSE.md) | `src/actual_et__fao56.F90`, `src/mass_balance__snow.F90` | depletion fraction adjusted: `p = p_table + 0.04 (5 − ET0 mm/d)`, clipped to [0.1, 0.8]; snow storage += snowfall, melt limited by storage | **checked in the repository** | **adopted** (p adjustment) — a formula, no code copied |
| [DOI-USGS/pywatershed](https://github.com/DOI-USGS/pywatershed) | `ba3974b75f8f` (2026-11-09) | CC0-1.0 | listing: `autotest/test_prms_soilzone.py`, `..._snow.py`, `..._groundwater.py` | soil zone + snow + groundwater as separate processes with budget tests | listing checked; process code not read | concept only: our month step keeps the same separation and budget test |
| [ImperialCollegeLondon/pyrealm](https://github.com/ImperialCollegeLondon/pyrealm) | `4344b040a189` (2026-10-20) | MIT | listing: `pyrealm/splash/{solar,evap,splash}.py` | SPLASH bucket (T, P, sunshine) | listing checked | not adopted: needs sunshine fraction; our bucket is the TZ's |
| ISRIC SoilGrids | docs | CC-BY-4.0 (data) | field list | sand / silt / clay, coarse fragments, SOC, depth layers | described | structure of the fields only; no data downloaded |
| TOPMODEL TWI (Beven & Kirkby 1979; Sørensen et al. 2006) | publications | journal | definition | TWI = ln(a / tan β), a = area / contour length | **checked in HexWorld** (physical areas, hex edge as contour) | adopted, with a grid-free anomaly against one hex on a 1 % slope |
| HAND (Rennó et al. 2008; Nobre et al. 2011) | publications | journal | definition | height above the nearest drainage along the flow path | **checked in HexWorld** (`test_hand_follows_the_flow_path_not_the_straight_line`) | adopted |
| Frost number (Nelson & Outcalt 1987) | publication | journal | F = √FDD / (√FDD + √TDD) | permafrost likely above ~0.5 | described | adopted as `permafrost_potential` (potential, not fact) |
| HEC-HMS temperature-index snow | docs | public | degree-day melt | melt = k_dd · max(T − T_melt, 0) · days | described | adopted with expected positive degree days of a month (normal daily spread) |

## Priority B and C

None of these were opened for 4.5:
- WorldEngine, terrain-generator, cartographer, Landlab, pysheds, Fastscape, Badlands, SoilMachine, SimpleHydrology, GeoTime, plate-tectonics;
- WorldClim, ERA5, Köppen–Geiger, CoastalME, ShorelineEVOL.

The ones already read for 4.2 are in `source_matrix.md`. Their status is **not checked**, and nothing was taken from them.
