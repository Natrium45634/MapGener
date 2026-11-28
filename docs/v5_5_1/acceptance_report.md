# TZ 5.5.1 — acceptance report (mineral resources, deposit geology, sparse subsurface)

Status of the release: **PARTIAL**. P0 and P1 are implemented as reduced models with tests;
several obligations are open and listed below — nothing missing is claimed. Every numeric
distribution is **provisional**: the TZ package ships no calibrated grade–tonnage data
(`grade_tonnage_distribution: null`), and none was downloaded (T81: no runtime network).
Passing the synthetic fixtures proves the physics and the bookkeeping, not the natural
frequency of deposits (T83 open).

Windows GUI: **NOT VERIFIED** (T82). The Qt code is exercised offscreen on Linux only.

## Gates

| gate | content | status | evidence |
|---|---|---|---|
| G0 | audit of 5.2, baseline, hashes, concordance | done | `docs/v5_5_1/audit.md` |
| G1 | catalogue 194 / 92 bundled + validated, typed units, ledger, hard gates with *missing* state | done | `minerals/catalog.py`, `units.py`, `ledger.py`, `features.py`, `gates.py`; `tests/test_v551_catalog.py` (305 tests) |
| G2 | 20 morphologies on 4 primitives, true thickness, exact intersections, volumes, hex split, grade field, faults, erosion | done | `minerals/bodies.py`, `morphology.py`, `query.py`; `tests/test_v551_bodies.py` |
| G3 | industrial / construction / gem templates, quality, varieties, size bins | partial | regional quarry / sand / clay / aquifer units, gem carriers and varieties; sand grain-size quality and clay mineralogy not modelled (T53) |
| G4 | placers and reworking with the ledger; geologic events | partial | `minerals/placers.py` source-to-sink; supergene enrichment (T47), folds (T25) and paleochannel scenarios not in the world plan |
| G5 | fluids, coal, petroleum, aquifers, brines, geothermal, hydrates, marine (P1) | partial | `minerals/fluids.py`; unconventional accumulations gated only (reported `not_implemented`); nodule thickness not yet from growth age in the plan (formula + fixture only) |
| G6 | GUI tabs, depth filter, section with bodies, hex panel, export | done | tabs «Недра и месторождения», «Геология недр», «Гидрогеология и топливо»; section dialog draws bodies with vertical exaggeration; GeoJSON / CSV / JSON export; `tests/test_v551_gui_export.py` |
| G7 | cross-layer validation, determinism, real-size benchmarks, old saves | partial | determinism, old saves, soil independence tested; benchmarks below; crop test (T79) and resolution-scaling test (T16) open |
| G8 | docs, report, calibration | partial | this report, `user_guide.md`, `research_manifest.json`; calibration not done (no data) |

## Test matrix

PASS = automated test; PARTIAL = implemented, the test covers only a part or the model is a
reduced proxy; MISSING = not implemented; BLOCKED = depends on an absent layer.

| ID | status | where / note |
|---|---|---|
| T01 | PASS | audit baseline; full suite at `847754b`: 803 passed, 0 failed |
| T02 | PASS | `test_t02_t03_save_load_round_trip_and_old_saves` |
| T03 | PASS | old configs load with `minerals.enabled = False`; 5.2 IDs untouched |
| T04 | PASS | `test_schema_rejects_invalid_records` |
| T05 | PASS | gates missing → unresolved; `test_plan_on_features_with_missing_inputs…` |
| T06 | PASS | 92 templates × positive / each gate absent / each gate missing |
| T07 | PASS | 194 commodities: model, unit, basis |
| T08 | PASS | REE group = sum of elements, TREO separate |
| T09 | PASS | varieties are views of the carrier; gem lots are subsets |
| T10 | PASS | byproducts only with their host and ratio |
| T11 | PASS | technology profile leaves the plan hash unchanged |
| T12 | PASS | inherited ages older than the simulated history |
| T13 | PARTIAL | VMS gate allows present-day land with marine facies; no forced scenario test |
| T14 | PARTIAL | the plan reads no weather year (by construction), no explicit test |
| T15 | PARTIAL | ocean = hydrology's connected ocean mask, not z < 0; no explicit test |
| T16 | PARTIAL | λ from physical area, stable physical province keys; two-resolution test open |
| T17 | MISSING | cluster statistics |
| T18–T24 | PASS | `test_v551_bodies.py` (N01, N04, N05) |
| T25 | MISSING | fold operator |
| T26, T27 | PASS | faults move older bodies only; erosion clip + volume |
| T28 | PARTIAL | water depth reported apart from burial; no sea-body test |
| T29, T30 | PASS | stacked bodies; sub-hex vein found in every crossed hex |
| T31–T37 | PASS | density once, grade units, weighted grade, mass-conditioned shapes, anisotropic deterministic field, host clipping |
| T38, T39 | PASS | no negative stock, idempotent replay |
| T40 | PARTIAL | alternative uses listed as non-additive; no test |
| T41–T43 | PASS | parent loses the transported mass; no source → no placer; downstream only |
| T44 | PARTIAL | `placers.paleochannel` (archived course), not placed in worlds |
| T45 | PARTIAL | coastal placers only from routed mass; wave sorting is a constant proxy |
| T46 | PASS | Ni laterite (ultramafic) and bauxite (Al-silicate) gates differ |
| T47 | MISSING | supergene enrichment |
| T48 | PASS | barren kimberlites, no random diamonds |
| T49, T50 | PARTIAL | ruby needs a Cr host nearby, emerald Be + Cr/V; emerald excluded from placers; no scenario test |
| T51 | PASS | size bins ≤ carrier, gem subset |
| T52 | PARTIAL | usable fraction < 1 per quarry unit; joints / block size not modelled |
| T53 | MISSING | sand grain size / roundness / purity |
| T54 | PARTIAL | clay, kaolin, bentonite separate gates; mineral fractions not modelled |
| T55 | PARTIAL | precipitation formula + N16 / N22; basin evaporation not wired |
| T56–T60, T62 | PASS | coal rank; petroleum seal / timing / capacity / B; aquifer Sy ≤ φ; J vs W |
| T61 | PARTIAL | brine dissolved stock separate; recharge of solutes not modelled |
| T63–T65 | PARTIAL | formulas + fixtures; plan bodies of nodules / crusts not from growth age yet; SMS tied to ridge volcanoes |
| T66 | BLOCKED | no 5.5 (biological resources) stage in this repository |
| T67 | PASS | the plan does not touch soil fertility |
| T68 | PARTIAL | the stage key follows v5.2 and its parameters; invalidation by affected IDs not implemented |
| T69 | PARTIAL | aquifer recharge from the soil water balance; no test |
| T70–T75 | PASS | hex split exact, depth bins, section, export with CRS, depth filter hides only |
| T76 | PARTIAL | surface-only (discovery) filter and `discovery_state`; no independent estimate |
| T77, T78 | PASS | seed case; deterministic (single-threaded) |
| T79 | MISSING | crop / region consistency test |
| T80 | PARTIAL | stage timings below (saved worlds, single seed per size) |
| T81 | PASS | catalogues bundled; no network at runtime |
| T82 | NOT VERIFIED | Windows GUI |
| T83 | MISSING | statistical calibration (no datasets) |
| T84 | PASS | full suite on the 5.5.1 code: 1150 passed, 2 failed — both tests asserted the pre-5.5.1 behaviour (the commodity tab now reads the 5.5.1 plan; the default world code now carries `minerals.enabled`); updated, the 4 affected modules re-run: 49 passed. Golden snapshots unchanged |

## Scenarios

S02 (barren pipes), S10 (river without a source), S13 (basin without a seal), S19 (stacked
bodies) are covered by tests. The other scenarios run through the same gates in natural
worlds but have no forced synthetic fixture yet: **PARTIAL**.

## Benchmarks (stage v551 only, on saved worlds, this container)

| world | hexes | hex km | systems | lots | v551, s | peak RSS +, MB |
|---|---|---|---|---|---|---|
| p500 | 500×300 | 30 | 644 | 649 | 2.9 | +101 |
| user world | 250×200 | 100 | 1805 | 1885 | 12.3 | +38 |
| lKsDscB4 | 1000×800 | 25 | 1842 | 1893 | 12.1 | +570 |

The count follows the physical area, not the number of hexes; the time follows the number of
systems (vein networks dominate). The proposed budget (≤ 120 s, ≤ 1 GB extra) holds at these
sizes.

## Calibration

`research_manifest.json` lists the sources of the TZ research appendix as **not used**: no
dataset was downloaded or fitted. All priors (`minerals/priors.py`) are engineering
placeholders; 35 of 92 reuse the 5.2 medians through the concordance.
