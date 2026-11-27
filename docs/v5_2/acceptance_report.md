# TZ 5.2 — acceptance report (soil fertility, geochemistry, mineral resources)

**Overall status: PARTIAL** (not `FULL_ACCEPTED`). The P0 core of results A and B is
implemented and tested: layered soil stocks with closing C / N / P / K / salt ledgers,
history-conditioned deposit systems with parametric 3-D bodies, grade, tonnage, depth and
stable ids, no random ore per hex, no global voxel grid. Open items are listed at the end.
Windows GUI on a real display: **NOT VERIFIED** (offscreen only, status `UNVERIFIED_WINDOWS`).

Environment: Linux x86_64, 4 vCPU / 16 GB, Python 3.13.18, NumPy 2.5.4, SciPy 1.18.1,
PySide6 6.12.0. Tests: `tests/test_v52_soil.py` (27), `tests/test_v52_deposits.py` (25, one
slow), `tests/test_v52_integration.py` (16). Results: full suite at `e80c4d7`: **763 passed,
0 failed** (31 min); the slow N20 test failed on its own window (outside the simulation
bounds), fixed in the next commit; a duplicate province id found by the 500×300 benchmark is
fixed as well (see `performance_report.md`).

## Gates

| Gate | Status | Evidence |
|---|---|---|
| G0 baseline audit | done | `docs/v5_2_baseline_audit.md` (field → producer → units → time → dims → quality → save → consumer; missing inputs and decisions) |
| G1 schemas, versions, units, RNG | done | `resources/__init__.py` (`hexworld-resources-5.2.0`), `ResourceData`, `ResourceParams`, per-object streams `resources/ore/<model>/<key>/<version>`, old configs load with the stage off (I01) |
| G2 static fertility | done (P0) | `resources/soil.py::compute_fertility_static`; S01–S05, S13–S16, S19–S22, S24, scenes |
| G3 monthly / annual budgets, bounded spin-up | done (P0): monthly explicit step with bounded fluxes, ledger per element; static state = analytic equilibrium + soil-age approach (implicit Euler, converging, S08) | `advance_soil_month`, `pipeline_v52._monthly_check`; S06–S12, S17, S18, N07–N09 |
| G4 provinces, prospectivity, missing-field semantics | done | `deposits.tract_suitability`, `context_from_project` (`unknown_fields` skip dependent types and are reported) |
| G5 sparse bodies, grade–tonnage, ids, age / depth | done | `resources/models.py`, `deposits.build_system`; N11–N16, R01–R03, R05–R09, R12, R15–R19, R23–R26 |
| G6 secondary processes | done (P0): river placers with a ledger, laterites, ion clays, beach sands, closed-basin salts and Li brines, construction materials | `resources/secondary.py`; R04, R10, R11, R13, R14 |
| G7 integration, invalidation | done | stage after v5.0 in both pipeline paths, stage key over all upstream stage keys + params (I13 reuse), v4.5 / v3 fields unchanged (S21) |
| G8 GUI, sections, save / load, exports | done offscreen | views `soil_fertility`, `mineral_resources`; hex panel sections; `query_geological_cross_section`; save / load (I02); GeoJSON / JSON / CSV / NPZ; CLI `hexworld resources` |
| G9 tests, scenes, benchmarks, docs | PARTIAL | see below; benchmarks in `docs/v5_2/performance_report.md` |

## Test matrix (TZ sec. 24) — 66 tests + N01–N20

| IDs | Status | Test / note |
|---|---|---|
| N01–N10 | PASS | `test_v52_soil.py` (N10 = S04) |
| N11–N16, N18, N19 | PASS | `test_v52_deposits.py` |
| N17 | PASS (relation) | scenes: equal CEC, different water supply (`test_s20`, `idx_water`) |
| N20 | PARTIAL (slow) | `test_n20_r20_same_world_two_resolutions`: systems present at both hex sizes keep ids and world coordinates exactly, but only 38 % of the volcano-anchored systems are present in both (the qualifying conditions use hex-sampled v5.1 environment and nearby lithology); tract systems may move as well |
| S01–S24 | PASS | `test_v52_soil.py` (S21 in `test_v52_integration.py::test_s21_*` and I14) |
| R01–R08, R10–R19, R21–R26 | PASS | `test_v52_deposits.py` |
| R09 oil / gas | PASS for the permissive rule; the petroleum system itself is **P1, simplified** (no kinetics, no migration graph) | `test_r08_r09_*` |
| R20 | PARTIAL: common systems identical, 38 % overlap of anchored systems between two resolutions | N20 test |
| I01–I16 | PASS (I11 on synthetic fragments; I14 offscreen only) | `test_v52_integration.py` |

## Scenes (TZ sec. 25)

Scenes 1–8 (carbonate valley, wet old tropics, loess steppe, sand desert, closed basin,
floodplain, peat bog, young / older lava) are synthetic columns with asserted relations
(`test_scene_patterns`, S19–S24). Scenes 11–16 and 19 are covered by the setting tests
(arc porphyry, sea-floor sulfides / VMS, old craton BIF / kimberlite, ultramafic laterite,
closed-basin brine, river placers, abyssal nodules). Scenes 9, 10, 17, 18, 20 are covered
only indirectly (acid ash pH, NoData under ice, coal under inherited basins, oil P1,
construction materials) — no dedicated end-to-end scene test.

## Open items (not claimed)

- `ecological_dynamics` (P1) is not implemented: the monthly step exists and is checked
  for one year from the static state, but no multi-year C / N / P / K dynamics with fire
  and succession runs in the pipeline.
- Soil chemistry is per hex (no v4.7 terrain-unit profiles); pH is a buffer-curve proxy;
  EC is a proxy from salt mass and pore water.
- The lithology beyond the six v3 rock classes is *inferred* (noise in the material frame
  plus setting); paleoclimate of old epochs is not simulated (old coal / evaporites are
  `inherited_stratigraphy`).
- Resolution independence (R20 / N20) is partial: which volcano systems host deposits and
  where tract anchors fall depend on hex-sampled inputs; measured overlap of anchored
  systems between 75 km and 50 km hexes of one world: 38 %.
- Oil and gas: simplified permissive traps only (P1). Geothermal (R35): reported by v5.1,
  no resource objects.
- Grade–tonnage medians are rounded orders of magnitude (`synthetic_not_calibrated`); no
  USGS data file was read in this session (see `docs/research/v5_2_sources.md`).
- No exploration / survey model (discovery states other than `undiscovered` and visible
  outcrops), no manual override editor.
