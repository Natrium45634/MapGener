# TZ 5.5.1 — G0 audit of the 5.2 mineral resources

Date 28.11.2026. Commit at the start: `847754b` (HexWorld 1.9.0, generator 5.12.0,
resources algorithm `hexworld-resources-5.2.2`, project schema `hexworld-project-4`).
Python 3.13.18, NumPy 2.5.4, SciPy 1.18.1. No 5.5 (biological resources) stage exists in
this repository: the 5.5 interfaces below are **MISSING** and only reserved.

Statuses: IMPLEMENTED (code + test), APPROX (reduced model, tested), PARTIAL (part of the
obligation), MISSING, UNVERIFIED (code exists, no test). A claim without code and test is
never IMPLEMENTED.

## Baseline (T01)

Two real generations, full pipeline (`v50` + `v52`), world mode:

| seed | size, hex km | time, s | systems | bodies | systems / hex |
|---|---|---|---|---|---|
| Ab12 | 120×96, 60 | 209 | 51 | 105 | 0.0044 |
| lKsDscB4 | 160×128, 120 | 206 | 299 | 729 | 0.0146 |

Semantic hashes (sha256 of the stage `content_hashes`, first 16 hex):

| stage | Ab12 | lKsDscB4 |
|---|---|---|
| geo | 5d8aaab35c08244f | b0e27a4ce873d93e |
| hist | 1df13f2c6e3dc25a | 6cbd4849abe4addc |
| terrain | 4c2e0aa3330efb72 | ae41f669add7f6fe |
| hydro | 966c1e10c7201c21 | c25272116e8cf443 |
| env (soils, months) | e4b31cf23f9452db | 47af3821d048df0d |
| earth | 4f084e41e0ac6cd1 | 6722238bd24ff653 |
| atmosphere | a73ee9e5f116684d | 42b8783b80916610 |
| biomes | cdddc2b653546df5 | d83d1fb26c402140 |
| volcanism | f9cb0891f03f4e34 | 8915889d926118b4 |
| resources (5.2) | da610ed93cbbb8da | b96f347939423351 |

Raw record: model and shape counts per world are kept with the audit script output
(`baseline.json`, not committed). The full test suite at `847754b` is recorded in the G0
section of `docs/v5_5_1/acceptance_report.md`.

## What 5.2 actually has (§§12–19 of TZ 5.2)

| requirement | actual_module | evidence | status | gap | action | test_id |
|---|---|---|---|---|---|---|
| deposit model catalogue | `resources/models.py` `MODELS` | 35 models R01–R35, ~30 commodities | PARTIAL | 5.5.1 needs 92 models, 194 commodity records | new data-defined catalogue, concordance below | T06, T07 |
| permissive tracts + count + grade-tonnage (USGS 3-part) | `deposits.tract_suitability`, `plan_deposit_systems` | boolean mask × sigmoid score, Poisson per connected province, λ from physical area | APPROX | no `unresolved` state for missing inputs; score mixes gates and soft factors | per-model hard gates with True/False/None, S = eligible·∏s^w | T05, T06 |
| count independent of hex size | `plan_deposit_systems` (λ·∫S dA) | `test_v52_deposits` R20/N20 (overlap 38 %, PARTIAL) | PARTIAL | anchors drawn per hex of a province | anchors from physical tiles | T16 |
| grade–tonnage sampler | `models.sample_grade_tonnage` | joint log-normal, Cholesky | APPROX | |z|≤3 truncation and grade caps are silent | log rejection / normalisation, provisional status | T34 |
| geometry ↔ mass consistency | `axes_for_volume` from ore t / density | mass-conditioned | APPROX | veins: vertical envelope, no dip, no true vs apparent thickness | new body geometry with true thickness, dip, finite ends | T18–T23 |
| morphologies | 12 shapes (`ELLIPSOID/SLAB/VEIN_SHAPES`) | vein, lens, stockwork, sheet, stratabound, regolith cap, placer ribbon, pipe, crust, reservoir | PARTIAL | missing vein_network graph, disseminated, replacement, breccia, pod, paleoplacer, quarry_unit, nodules (separate), seam, cavity_fill, laterite profile zones | 21 morphologies | T18–T30 |
| depth, burial, exposure | `build_system`, `_exhumation`, `eroded_above_m` | depth_top, burial, outcrop area | APPROX | depth below current solid surface; water depth of sea bodies not separate | separate `water_depth_m` / `burial_below_seabed_m` | T28 |
| erosion of bodies, eroded mass | `catalog["eroded"]` | eroded commodity mass per system | APPROX | parent mass is reduced, but no ledger object | `ResourceLedger` with idempotent transfers | T38, T39, T41 |
| river placers from eroded parents | `secondary.river_placers` | released / deposited / exported ledger | PARTIAL | only the 40 richest hexes become bodies; the rest of the deposited mass is booked nowhere (the ledger says deposited, the catalogue holds less) | every deposited mass in a lot or an explicit sub-economic store | T41–T43 |
| coastal heavy sands, laterites, salts | `secondary.coastal_tracts`, `weathering_tracts`, `closed_basin_salts` | tracts and systems | APPROX | no source mass for coastal sands; laterite Ni vs bauxite by lithology only | source-to-sink and protolith gates | T45, T46, T55 |
| hex aggregation | `deposits.aggregate` | body mass split equally over the hexes its footprint touches | APPROX | not area-weighted; footprint radius, not geometry | clipped integration per hex | T70 |
| depth windows / cross-section | `api.query_deposits_at_world_xy`, `query_geological_cross_section` | vertical intervals of bodies | APPROX | vein intervals from an envelope plate | exact ray / plane intersections | T29, T72 |
| discovery / ground truth | `surface_occurrence`, `discovery_state`, player vs truth view | GUI overlay | IMPLEMENTED | — | keep | T76 |
| units | mass fractions inside; `ppm_to_fraction`, `g_per_t_to_fraction`, `brine_mass_kg` | helpers + tests | PARTIAL | no typed unit registry, no ct, g/m³, kg/m² | typed unit registry | T32 |
| byproducts with host chemistry | model grades (Ag in Cu systems, etc.) | fixed per model | PARTIAL | no primary_host / trace ratio / budget | byproduct records with host | T10 |
| gems | `R34` generic `gem`, `R22` kimberlite | one aggregate | MISSING (varieties) | no carrier / variety / gem fraction / size bins | gem lots | T09, T48–T51 |
| construction / industrial | `secondary.construction_materials` (class per hex) | dense class layer | PARTIAL | class per hex, no quarry units, no quality spec | quarry units from lithology volumes | T52–T54 |
| petroleum | `R25` trap with basin fill sigmoid | systems | APPROX | no source / maturity / seal / spill logic | petroleum system with timing and capacity | T57–T59 |
| aquifers, geothermal, brines | `R27` Li brine, `R35` geothermal | systems | PARTIAL / MISSING (aquifer) | no aquifer model, J vs W not separated | P1 models | T60–T62 |
| marine nodules, crusts, SMS | `R29–R31` | systems | APPROX | growth rate / age, true area | P1 models | T63–T65 |
| exports | `api.export_resources` | JSON/CSV | PARTIAL | no GeoJSON footprints, no CRS metadata | GeoJSON + manifest | T73 |
| GUI | «Полезные ископаемые (v5.2)», «Ресурсы по видам» | tabs | PARTIAL | no depth filter, no subsurface geology tab, section only of 5.2 bodies | new tabs + depth filter | T74, T75 |

Checks asked for explicitly:

* **Random assignment by biome** — none: the deposit code does not read biomes (grep of
  `resources/deposits.py`, `secondary.py`). Lithology is inferred from rock type, basins,
  volcanism and the orogen (`infer_lithology`, provenance «inferred»).
* **Dense objects on every hex** — none: 0.004–0.015 systems per hex; the dense layers are
  9 family aggregates and indices.
* **Wrong units** — internal mass fractions; tests cover ppm and g/t only.
* **Double counting** — a body shared by n hexes is split 1/n (no double count, but not
  area weighted). The placer ledger is not closed (see above).

## Concordance 5.2 → 5.5.1 (before any migration)

| 5.2 | name | 5.5.1 model(s) |
|---|---|---|
| R01 | porphyry Cu–Mo–Au | PORPHYRY |
| R02 | epithermal Au–Ag | EPITHERMAL |
| R03 | skarn Cu/Fe/W/Zn | SKARN |
| R04 | orogenic Au | OROGENIC_AU |
| R05 | VMS | VMS |
| R06 | SEDEX | SEDEX |
| R07 | sediment-hosted Cu | SED_CU |
| R08 | magmatic Ni–Cu–PGE | NI_SULFIDE, PGE_REEF |
| R09 | chromite | CHROM_POD, CHROM_LAYER |
| R10 | Li pegmatite | LCT_PEG |
| R11 | carbonatite REE–Nb | CARBONATITE |
| R12 | greisen Sn–W | GREISEN |
| R13 | sandstone U | U_ROLL |
| R14 | unconformity U | U_UNCONF |
| R15 | BIF | BIF |
| R16 | sedimentary Fe–Mn | FE_OOLITE, MN_SED |
| R17 | bauxite | BAUXITE (and BAUXITE_KARST, new) |
| R18 | Ni–Co laterite | NI_LATERITE |
| R19 | REE ion clay | REE_CLAY |
| R20 | fluvial placer | PLACER_RIVER (PALEOPLACER new) |
| R21 | heavy mineral sands | PLACER_COAST |
| R22 | kimberlite | KIMBERLITE |
| R23 | graphite | GRAPHITE |
| R24 | coal | COAL (PEAT, OIL_SHALE new) |
| R25 | oil and gas | PETROLEUM (UNCONVENTIONAL, HYDRATE new) |
| R26 | evaporite salts | EVAPORITE |
| R27 | Li brine | BRINE |
| R28 | phosphorite | PHOSPHORITE, MARINE_PHOS |
| R29 | SMS | SMS |
| R30 | Fe–Mn nodules | MARINE_NODULE |
| R31 | Co crusts | MARINE_CRUST |
| R32 | volcanic sulfur | SULFUR |
| R33 | construction materials | QUARRY_IGNEOUS, QUARRY_SED, QUARRY_META, SAND_GRAVEL, CLAY |
| R34 | gemstones (aggregate) | GEM_PEG, GEM_META, GEM_MAFIC, CORUNDUM_META, CORUNDUM_MAG, EMERALD_META, EMERALD_SED, SILICA_GEODE, OPAL, … |
| R35 | geothermal | GEOTHERMAL |

New in 5.5.1 without a 5.2 predecessor: FE_TI_V, BOG_IRON, CARLIN, IOCG, MVT, POLY_VEIN,
NYF_PEG, ALKALINE, U_VEIN, SB_HG, JADEITE, NEPHRITE, CU_OXID, LAPIS, CHAROITE, AMBER,
FOSSIL_ORG, VOLCANIC_GLASS, KAOLIN, BENTONITE, ZEOLITE, DIATOMITE, TALC, REFRACTORY,
WOLLAST, MICA_FELDSPAR, BARITE_FLUOR, VERMICULITE, APATITE, BORATE_SODA, PIGMENTS,
AQUIFER, CELESTINE, MAGNESITE, FELDSPAR_GEM.

## Decision (no new project, no replacement of working code)

* 5.2 stays as it is: soils, the 5.2 catalogue and its tests are unchanged; old saves keep
  their 5.2 plan (T02).
* 5.5.1 is a new stage `v551` after `v52` with its own algorithm version, data object and
  save section. It reuses the 5.2 geological context (`GeoContext`: lithology inferred,
  basins, volcanic systems, orogens, ages) through an **adapter** with provenance
  `inferred`; inherited pre-40 Ma complexes are labelled `inherited_*`.
* Catalogues: the 194 / 92 records of the package are bundled as data and validated; the
  numeric distributions stay `provisional` (the package ships them as null), so every
  generated object carries `calibration_status = provisional`.
