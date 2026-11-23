# TZ 5.4.1 — acceptance report

**Overall status: PARTIAL** (not `FULL_ACCEPTED`): the P0 physics of G1–G7 are implemented
and tested; open items are listed at the end. Windows GUI: NOT VERIFIED (no Windows, no
real display; the dialogs are exercised offscreen only — status `UNVERIFIED_WINDOWS`).

Commits: `5692ca6` (a/b: boundaries, SST), `826b025` (c: water / ice state),
`72d36b1` (d: marine habitats), this commit (e: T20, fixtures, legacy, reports).
Environment: Python 3.13.18, NumPy 2.5.4, SciPy 1.18.1, PySide6 6.12.0, Linux x86_64,
4 vCPU / 16 GB. Commands: `pytest tests -n 3` (quick + integration),
`pytest -m slow tests/test_v541_resolution.py` (T20, ~8 min). Results: FULL_SUITE_RESULT.
Observational validation (ERA5 / OISST / WOA / GLORYS): **NOT_RUN** — the model is not
called observationally calibrated.

## Defects of the report (D01–D09) and extensions (E01–E02)

| ID | Status | Evidence |
|---|---|---|
| D01 boundary profiles in GUI / API / JSON | implemented, tested | `boundary.py`, settings dialog «Границы региона» editor, `boundary_schema.md`, `test_v541_boundary.py` (T12–T17, N02, S10, offscreen editor) |
| D02 cold SST | root cause found and fixed | `defect_sst_before.md`, `defect_sst_root_cause.md`, `ocean_model_adr.md`, `test_v541_ocean.py`, 5.3 ocean tests |
| D03 weather year → soil | implemented, tested (monthly mode) | `hydrostate.py`, `weather_state_contract.md`, `test_v541_hydro.py` |
| D04 weather year → glaciers | implemented, tested (mass; geometry pending) | same |
| D05 v5_4::T20 on real windows | passed (A resolution, B nested windows) | `test_v541_resolution.py` (slow): SLOW_RESULT |
| D06 fixtures 5.3 | 24 fixtures linked to existing tests; statuses PASS / PARTIAL / OPEN | `tests/fixtures/v5_3/S01–S24.json`, `test_v541_fixtures.py` |
| D07 golden / legacy | v1/v2 genuine read + upgrade copy + current golden; v3–v5.4 BLOCKED_MISSING_FIXTURE | `legacy_fixture_inventory.md`, `test_v541_legacy.py`, `tests/golden/` |
| D08 moisture / storms | conservative column budget (5.4 tests), honest proxies; boundary import booked | `test_v54_*.py`, `test_v541_boundary.py` |
| D09 ash winds | regression kept | `test_v54_integration.py::test_t73_wind_at_for_ash` |
| E01 currents | Stommel transport kept; slab Ekman; closed deep reservoir; open-side exchange | `ocean_model_adr.md` |
| E02 biomes | registry audit, marine habitat axes (potential), drought-stress state | `biome_registry_audit.csv`, `test_v541_marine.py` |

## v5_4_1::T01–T84 (test matrix)

| IDs | Status | Evidence / note |
|---|---|---|
| T01 audit manifest | PASS | `baseline.md`, `audit.csv` |
| T02 geometry (A, L, d, normals) | PASS | `test_v541_fixtures.py::test_n01_t02_hex_geometry_and_parity`, `test_v54_atmosphere.py::test_hex_mesh_has_six_faces_and_closed_divergence` |
| T03 viewport | PASS | `test_v541_resolution.py::test_t20_b_nested_windows_and_viewport` (world arrays identical) |
| T04 realm | PASS | v4.6 / v5.3 closed-depression tests (fixture S23 of 5.3) |
| T05 one SST / one normals snapshot | PASS | `test_v54_integration.py::test_t10_one_sst_one_ocean` |
| T06, T07 legacy read / upgrade copy | PASS for v1/v2; v3–v5.4 BLOCKED_MISSING_FIXTURE | `test_v541_legacy.py` |
| T08 legacy regeneration | BLOCKED (old solvers not kept as runners) | inventory |
| T09 new deterministic + golden | PASS | `test_v541_legacy.py::test_t09_…` |
| T10 seed namespaces | PASS | `test_v54_atmosphere.py::test_t07_t08_seed_namespaces`, T49 |
| T11 cancel / rollback | PARTIAL: stages build a new data object and commit it at once; no explicit cancel test in 5.4.1 | — |
| T12–T17 boundaries | PASS | `test_v541_boundary.py` |
| T18 parent nested | NOT AVAILABLE (`nested_region` not implemented; validation says so) | `boundary_schema.md` |
| T19 BC event track | PASS (incoming TC with ids) | `test_v54_integration.py::test_s24_…` |
| T20–T23 wet graph, strait, antisymmetry, projection | PASS for the raster ocean (5.3 T09–T13); the edge-incidence projection of TZ §5.5 is not used (exactly divergence-free streamfunction instead) | `test_v53.py` |
| T24, T25 equator / Coriolis | PASS | N03 |
| T26, T27 gyre / island-channel | PASS (gyre); periodic channel NOT tested (no periodic topology on the flat map) | `test_v53.py::test_subtropical_gyre_has_a_western_boundary_current` |
| T28 upwelling | PASS | `test_v53.py::test_t16_…` |
| T29–T34 CFL, tracer, heat, flux units, vertical exchange, mixed layer | PASS | `test_v53.py` T14/T15, `test_v541_ocean.py` N04–N06, T30, T33 |
| T35 SST root cause | PASS | root-cause doc + `test_t35_ebm_is_planet_physics_not_a_table` |
| T36 ocean evaporation / latent heat | PARTIAL: bulk evaporation of the moisture model uses the SST; the ocean heat budget uses the linearised exchange (latent heat not separated) | — |
| T37–T40 ice freeze / melt / fraction / albedo | PASS | N09, N10, 5.3 T17–T22 |
| T41 ice drift, T42 salt | drift PASS (5.3 T20); salt NOT IMPLEMENTED (O2) | — |
| T43 spin-up | PASS | `test_t43_spin_up_forgets_the_start` |
| T44 coast coupling | PARTIAL: SST anomaly → land temperature via the coupler; no dedicated offshore-wind negative test | — |
| T45–T47 moisture, dry column, storm rain | PASS | 5.4 tests |
| T48 storm statistics | PASS (proxies labelled) | 5.4 tests |
| T49 PET | PASS | 5.3 FAO-56 tests, N11 in 5.3 |
| T50–T66 weather → water / ice | PASS (monthly mode) | `test_v541_hydro.py` |
| T67–T75 biomes / marine | PASS (T70 rare land classes: predicates tested in 5.3; T73 kelp/seagrass 5.3 tests) | `test_v541_marine.py`, `test_v53_ecology.py` |
| T76, T77 T20 resolution / nested | PASS | slow tests |
| T78 physical scale | PARTIAL: CFL / transport in physical units are tested; no dedicated fixed-shape km sweep | — |
| T79 remap | PASS (conservative block remap of rain, 5.4) | `normals.upsample(conservative=True)` |
| T80 1000×800 benchmark | measured | `performance_report.md` |
| T81 GUI layers | PASS offscreen (new map modes, boundary editor) | — |
| T82 old 5.3 fixtures | PASS (linked; statuses honest) | `test_v541_fixtures.py` |
| T83 old 5.4 regression | PASS | `test_v54_*.py` |
| T84 save / replay | PASS (golden save / reload; atmosphere save / load in 5.4) | — |

## N01–N20

N01 (`test_v541_fixtures.py`), N02, N03, N04, N05, N06, N09, N10, N12 (5.4
budget), N13–N20: PASS in `test_v541_*`; N07 (entrainment with variable h) NOT
IMPLEMENTED (constant h); N08 (latent heat of E): PARTIAL (see T36); N11 (FAO-56): PASS in
`test_v53.py::test_t30_fao56_example_18`.

## Open items (not claimed)

- Legacy read of v4+ data produced by an older algorithm: the loader does not attach it
  (policy since v4) — conflicts with §11.1; v3–v5.4 genuine fixtures missing.
- `nested_region` (parent boundary data), salinity / two-layer ocean (O2), periodic
  channel topology, variable mixed layer with entrainment.
- 5.3 recovery backlog (fixture statuses PARTIAL / OPEN): isostasy and sea level not
  applied to the relief, dunes / loess as layers only, fog potential.
- Weather year: monthly aggregated mode, one-way atmosphere, closed-lake integration only
  as a formula.
- `full` precipitation coupling stays optional (`seasonality_only` default, 5.4 decision).
