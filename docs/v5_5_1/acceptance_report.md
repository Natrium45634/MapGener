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
| G3 | industrial / construction / gem templates, quality, varieties, size bins | done (reduced) | regional quarry / sand / clay / aquifer units, gem carriers and varieties, sand quality proxies; clay mineralogy and stone jointing not modelled |
| G4 | placers and reworking with the ledger; geologic events | done (reduced) | source-to-sink placers, supergene enrichment, folds, faults, burial, erosion; paleochannels available but not placed automatically |
| G5 | fluids, coal, petroleum, aquifers, brines, geothermal, hydrates, marine (P1) | done (reduced) | conventional traps, unconventional (shale gas / tight oil by maturity, coal-bed methane in mature seams), marine growth by age |
| G6 | GUI tabs, depth filter, section with bodies, hex panel, export | done | tabs «Недра и месторождения», «Геология недр», «Гидрогеология и топливо»; section dialog draws bodies with vertical exaggeration; GeoJSON / CSV / JSON export; `tests/test_v551_gui_export.py` |
| G7 | cross-layer validation, determinism, real-size benchmarks, old saves | done | determinism, old saves, soil independence, crop, resolution scaling, invalidation |
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
| T13 | PASS | scenario S07: VMS eligible on present land with marine facies |
| T14 | PARTIAL | the plan reads no weather year (by construction); no explicit test |
| T15 | PASS | `test_t15_closed_basin_below_sea_level_is_not_ocean` |
| T16 | PASS (slow) | `test_t16_same_physical_area_at_two_resolutions…`: intensity follows area |
| T17 | PASS | every tract / regional system sits on a hex where its gates hold (host-conditioned); no hex-aspect term exists in the intensity |
| T18–T24 | PASS | `test_v551_bodies.py` (N01, N04, N05) |
| T25 | PASS | `structure.Fold` / `Column`: strata keep their order, older bodies move, younger do not; overturned folds refused (not modelled) |
| T26, T27 | PASS | faults move older bodies only; erosion clip + volume |
| T28 | PASS | `test_t28_body_under_the_sea_floor…` |
| T29, T30 | PASS | stacked bodies; sub-hex vein found in every crossed hex |
| T31–T37 | PASS | density once, grade units, weighted grade, mass-conditioned shapes, anisotropic deterministic field, host clipping |
| T38, T39 | PASS | no negative stock, idempotent replay |
| T40 | PASS | quarry lots: one contained mass, uses as views |
| T41–T43 | PASS | parent loses the transported mass; no source → no placer; downstream only |
| T44 | PARTIAL | `placers.paleochannel` follows the archived azimuth (tested); not placed automatically in worlds |
| T45 | PARTIAL | coastal placers only from routed mass; wave sorting is a constant proxy |
| T46 | PASS | Ni laterite (ultramafic) and bauxite (Al-silicate) gates differ |
| T47 | PASS | `secondary551.supergene_enrich`: leached cap → blanket below the water table, metal conserved; applied to shallow Cu systems |
| T48 | PASS | barren kimberlites, no random diamonds |
| T49, T50 | PASS | ruby gated by a Cr host, emerald by Be + Cr/V, beryl never routed to placers (tests) |
| T51 | PASS | size bins ≤ carrier, gem subset |
| T52 | PARTIAL | usable fraction < 1 per quarry unit (tested); joints / block size not modelled |
| T53 | PASS (proxy) | `sand_quality`: grain size by trap energy, roundness and purity by transport and source; glass / concrete / foundry specifications; dune sand is not concrete sand |
| T54 | PARTIAL | clay, kaolin, bentonite have distinct gates (tested); mineral fractions not modelled |
| T55 | PASS | `evaporate`: gypsum → halite → potash in order, each from its own dissolved stock, ledger-conserved; pure water leaves no salt. Basin palaeo-evaporation history not wired (EVAPORITE lots come from the tract plan) |
| T56–T60, T62 | PASS | coal rank; petroleum seal / timing / capacity / B; aquifer Sy ≤ φ; J vs W |
| T61 | PASS | dissolved stock and solute recharge (kg/yr) kept as stock vs flow |
| T63, T64 | PASS | nodule and crust thickness = growth rate × substrate age (capped); crust mass on the true flank area; nodules with coverage |
| T65 | PARTIAL | SMS tied to ridge / back-arc volcanic systems; former geometry after burial not tracked |
| T66 | BLOCKED | no 5.5 (biological resources) stage in this repository |
| T67 | PASS | the plan does not touch soil fertility |
| T68 | PASS | `structure.affected_systems`: only systems whose bodies cross changed hexes |
| T69 | PARTIAL | aquifer recharge from the soil water balance; no test |
| T70–T75 | PASS | hex split exact, depth bins, section, export with CRS, depth filter hides only |
| T76 | PASS | `discovery_view`: only exposed systems, independent log-normal estimates (exported as `minerals_discovery.json`) |
| T77, T78 | PASS | seed case; deterministic (single-threaded) |
| T79 | PASS | `structure.crop` reads the global plan: identical objects in the common domain |
| T80 | PARTIAL | stage timings below (saved worlds, single seed per size) |
| T81 | PASS | catalogues bundled; no network at runtime |
| T82 | NOT VERIFIED | Windows GUI |
| T83 | MISSING | statistical calibration (no datasets) |
| T84 | PASS | full suite on the 5.5.1 code: 1150 passed, 2 failed — both tests asserted the pre-5.5.1 behaviour (the commodity tab now reads the 5.5.1 plan; the default world code now carries `minerals.enabled`); updated, the 4 affected modules re-run: 49 passed. Golden snapshots unchanged |

## Scenarios

All 20 scenarios have tests. S02, S09–S11, S13, S19, S20 run the full mechanism (barren
pipes, source-to-sink, no source, paleochannel, seal, stacked bodies, burial); S01, S03–S08,
S12, S14–S18 are **gate-level** synthetic fixtures (which templates the setting allows and
forbids), not statistics of natural worlds.

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
