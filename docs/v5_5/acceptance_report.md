# TZ 5.5 — acceptance report

**Overall status: PARTIAL** (not `FULL_ACCEPTED`). P0 is implemented and tested except the
items marked below; P1 is partly implemented (dynamics, ledger, coupler, migration and
demography operators) and partly open. Windows GUI: **NOT VERIFIED** (Linux, Qt offscreen
only). External reference validation (B82): **BLOCKED** (no datasets).

Build: HexWorld 1.11.0, generator 5.14.0, algorithm `hexworld-biosphere-5.5.0`, registry
`hexworld-bio-registry-1`. Commands: `pytest tests/test_v55_numeric.py
tests/test_v55_scenarios.py tests/test_v55_integration.py`; benchmark
`python tools/bench_v55.py OUT`.

Statuses: PASS (implemented + test), PARTIAL (part of the requirement, named), MISSING,
BLOCKED (dependency absent), NOT VERIFIED.

## Gates

| gate | status | evidence |
|---|---|---|
| G0 audit, baseline | PASS | `audit.md`, `input_contracts.csv`, hashes |
| G1 schema, registry, habitats, niches | PASS | `registry.py`, `inputs.py`, `niche.py`; B08–B18 |
| G2 productivity, plant stocks | PASS (uncalibrated) | `production.py`; N04–N06, B19–B25 |
| G3 graphs, history, colonisation | PARTIAL | `graph.py`; B27–B31, B33–B35 PASS; B32 synthetic only (no stored past land masks) |
| G4 food allocation, fauna, fresh water, sea | PASS (uncalibrated) | `foodweb.py`, `production.py`; B40–B57 |
| G5 metrics, resources, aggregation, UI | PASS | `metrics.py`, `resources.py`, `export.py`, `views55.py`, hex panel; B58–B65, B78, B79 |
| G6 P1 dynamics, migration, disturbance, feedback | PARTIAL | `dynamics.py`; B26, B37, B38, B45–B47, B68, B71, B77, B84 PASS; B39, B70, B75, B76 MISSING; C09–C16 proposals only |
| G7 validation, real grids, legacy, performance | PARTIAL | B02–B04, B80, B81 see below; B82 BLOCKED; Windows NOT VERIFIED |

## Test matrix B01–B84

| ID | status | evidence / reason |
|---|---|---|
| B01 | PASS | `audit.md` lists every open 5.4.1 item; none counted as pass |
| B02 | PASS | `test_v55_integration::test_b02_b03_old_save_and_roundtrip` |
| B03 | PASS | `test_b03_bio_generate_makes_a_copy` (copy, refusal to overwrite, original bytes unchanged) |
| B04 | PASS | `test_b04_b07_physics_hashes_unchanged`, `test_life_disabled_has_no_biota` |
| B05 | PASS | `test_b05_b06_seeds_and_repeatability` |
| B06 | PASS | same (repeat → identical arrays and ids; single-threaded NumPy order) |
| B07 | PASS | `test_stage_keys_ignore_display_and_follow_seed` |
| B08 | PASS | `test_b08_registry` |
| B09 | PASS | `test_b09_missing_nutrients_give_unknown_not_zero` |
| B10 | PASS | `test_b10_b05_taxa_are_not_groups_and_ids_are_stable` |
| B11, B12 | PASS | `test_b11_b12_fractions_close_and_dry_depression_stays_land` |
| B13 | PASS | `test_b13_mixed_coastal_hex_counts_only_its_habitat` |
| B14 | PASS | `test_b14_mountain_microsite_keeps_a_cold_template` |
| B15 | PASS | `test_n02_b15_trapezoid` |
| B16, B18 | PASS | `test_b16_b17_b18_seasons` |
| B17 | PASS | `test_b17_mirrored_hemispheres_shift_the_season` |
| B19, B20 | PASS | `test_b19_b20_water_and_nutrients_limit_npp` |
| B21 | PASS | `test_b21_pet_zero_no_division` |
| B22 | PASS | `test_b22_light_absorbed_not_more_than_incident` |
| B23, B24 | PASS | `test_n04_b23…`, `test_n06_b24…` |
| B25 | PASS | `test_b25_layer_covers_within_available_area` |
| B26 | PASS | `test_b71_b84_dynamics…` (dry years lower stocks, wood keeps > 60 %) |
| B27, B28 | PASS | `test_b27_s09…`, `test_b28…` |
| B29, B30, B31 | PASS | `test_b29…`, `test_b30…`, `test_b31…` |
| B32 | PARTIAL | `test_b32_land_bridge_opens_exchange_synthetic`; whole-world BLOCKED (no past land masks) |
| B33, B34 | PASS | `test_b33_b34_s08…` |
| B35 | PASS | `test_n07_n08_b35_occupancy` |
| B36 | PARTIAL | lineage keys are world-km stable (`test_b36…`); a cropped regional regeneration is not implemented |
| B37, B38 | PASS | `test_n09_b37…`, `test_b38…` (operators; seasonal movement on the map is a presence layer) |
| B39 | MISSING | stop-over feasibility of migration routes not modelled |
| B40–B44 | PASS | `test_n10_b40…`, `test_b41_b42…`, `test_b43_b48…`, `test_n11_b44…` |
| B45, B46 | PASS | `test_b45_b46…` |
| B47 | PASS | `test_n12_b47…`; dynamics move litter once (ledger) |
| B48 | PASS | `test_b43_b48…` |
| B49 | PASS | `test_b49_s12…` |
| B50 | PARTIAL | the `anadromous` gate and the river–ocean graph exist; no dedicated test of every stage gate |
| B51, B52 | PASS | `test_b51…`, `test_n13_n14_b51_b52…` |
| B53, B54 | PASS | `test_b53_b54_s15…` |
| B55 | PASS | `test_b55_s16…` |
| B56 | PASS | `test_b56_b57…` |
| B57 | PARTIAL | coral gates tested; kelp / seagrass substrate cases only via the shared gates |
| B58–B60 | PASS | `test_n15…`, `test_n16…`, `test_n17…` |
| B61 | PASS | `test_b61_range_touching_the_window_edge_is_unknown` |
| B62, B63 | PASS | `test_n18…`, `test_b63…` |
| B64, B65 | PASS | `test_b64_b65…` (numeric + integration inventory) |
| B66 | PARTIAL | MSY only for logistic stocks (N19); harvest depletion / recovery not simulated |
| B67 | PARTIAL | precursor score separate; an accessibility field is not computed |
| B68 | PASS | `test_n20_b68…` |
| B69 | PARTIAL | young / remote land gets no instant forest (occupancy, wood regrowth τ 40 yr); no lava-flow event test |
| B70 | MISSING | seed banks of secondary succession |
| B71 | PASS | `test_b71_b84…` |
| B72 | PARTIAL | owner interception operator tested; not wired into the 4.x hydrology |
| B73 | PARTIAL | litter → SOC is a recorded proposal (one-way) |
| B74 | PARTIAL | owner erosion-cover operator tested; not applied to the erosion stage |
| B75 | MISSING | river → sea subsidy |
| B76 | MISSING | geometry revision remap |
| B77 | PASS | `test_b77_coupler_rolls_back_every_owner` |
| B78 | PASS (Linux offscreen) / NOT VERIFIED (Windows) | hex report from stored state, `test_b78…`; no solver on hover |
| B79 | PASS | `test_b79_region_report_union` |
| B80 | see performance | `performance_report.md` |
| B81 | see performance | `performance_report.md` |
| B82 | BLOCKED | no MOD17 / GBIF / OBIS / TRY data |
| B83 | see regression | below |
| B84 | PASS | `test_b71_b84…` (replay identical, yearly residual < 1e-9) |

## Numerical references N01–N20

All PASS in `tests/test_v55_numeric.py` (tolerance 1e-8 exact, 1e-5 rounded).

## Scenarios S01–S20

| ID | status | evidence |
|---|---|---|
| S01 | PARTIAL | tropical forest of `Bio55`: NPP 956 gC/m²/yr, 13.7 kgC/m² (`scientific_validation.md`) |
| S02 | PASS | B20 |
| S03 | PARTIAL | dry-season water limit and seasonal activity; no grazer pulse test |
| S04 | PASS | `test_s04_desert…` |
| S05 | PARTIAL | winter survival and dormancy (B16); no dedicated scenario |
| S06 | PARTIAL | not a dedicated gradient test |
| S07 | PARTIAL | microsites (B14) |
| S08 | PASS | B34 |
| S09 | PASS | B27 |
| S10 | PARTIAL | observed on `Bio55` islands (richness falls with distance), no paired fixture |
| S11 | PARTIAL | B32 synthetic; world history BLOCKED |
| S12 | PASS | B49 |
| S13 | MISSING | floodplain hydroperiod |
| S14 | PARTIAL | substrate shares from coast types; no dedicated test |
| S15 | PASS | B53 |
| S16 | PARTIAL | export and vents (B55); arctic bloom not tested |
| S17 | PARTIAL | droughts and fire (B26, B71); floods not |
| S18 | PARTIAL | lava age and remote land; no ice-retreat test |
| S19 | PARTIAL | stable keys; no nested regional generation |
| S20 | PARTIAL | seeds and no-life case PASS; other planet parameters not tested |

## Coupling contracts C01–C22

| ID | status | note |
|---|---|---|
| C01 | PARTIAL | present-only topology; no epochs of land |
| C02–C04 | PASS | fractions, depth, monthly climate, soil pools |
| C05 | PARTIAL | flows and barriers; no drying event |
| C06 | PASS | SST, salinity, currents, ice, PAR, upwelling (nutrients as proxy) |
| C07 | PARTIAL | present-day footprints, no yearly volcanic event ids |
| C08 | PARTIAL | glacier fractions; retreat age not used |
| C09, C11, C12, C15 | PARTIAL | typed proposals, owner operators, one-way by default |
| C10, C16, C20 | MISSING | not implemented |
| C13 | PARTIAL | detritus decomposition inside the biosphere ledger |
| C14 | PARTIAL | consumption partitions NPP flux; plant stocks not reduced by grazing in the baseline |
| C17 | PASS | export removed from the surface, Martin attenuation |
| C18 | PARTIAL | migration operator; no movement fluxes on the world |
| C19 | PASS | fire emissions to a prescribed atmosphere receiver, ledger entries with event ids |
| C21 | PASS | read-only inventory |
| C22 | PARTIAL | resources, water, hazards and precursors readable; no dedicated schema |

## 5.5.1 item closed here

T66 (organic / biogenic mass counted once): **PASS** — `test_551_t66_biogenic_mass_counted_once`
(running 5.5 leaves the 5.5.1 plan hash unchanged; the 5.5 inventory has no geological
commodity).
