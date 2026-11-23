# TZ 5.3 — acceptance report (living document)

**Status:** 5.3a «Климат и океан» (G0–G3), 5.3b «Криосфера» (G4–G5), 5.3c «Геоморфология»
(G6–G7), 5.3d «Экосистемы» (G8). G9/G10 partial (see below). Full «5.3 done» is NOT
claimed: several P1 items are reduced-order and listed as such.
G6–G10 are listed with their real state below. Nothing here claims more than the tests
show.

## Gates

| Gate | Content | State | Evidence |
|---|---|---|---|
| G0 | audit, baseline, compatibility | done | `v5_3_baseline_audit.md`, T01 |
| G1 | planet, winds aloft, FAO-56 | done | T25–T32, T69–T72, numeric controls |
| G2 | ocean flow, SST, upwelling | done, CALIBRATED_APPROXIMATION | T09–T16, western boundary current test |
| G3 | sea ice | done | T17–T24 |
| G4 | ice-sheet seed, SIA spin-up, convergence | done (P0); NOT_CONVERGED reported honestly when 50 kyr is not enough | T33–T40, Vialov steady state within 10 % |
| G5 | shelves, erosion, isostasy, sea level | P0 shelves (flux-constrained); isostasy and SLE as diagnostic layers, **not applied to the relief** | T41–T48 |
| G6 | aeolian and coastal morphodynamics | done (P1, hex scale): sand from soils / beaches, Bagnold threshold flux, FV transport 2000 yr, dune types, dust → loess, one-line longshore budget, rock-law cliffs; layers only (dunes not added to the relief) | T49–T55 |
| G7 | volcanic TerrainDelta and drainage rebuild | done (P0): per-event deltas, atomic apply, whole-window rebuild of routing / lakes / ocean; v5.0 reads new dam lakes; v4 layers kept (snapshot) | T73–T75, T77–T79, T08 |
| G8 | dynamic vegetation, fire, rare biomes | done (P1, reduced order): potential vs actual state, Miami NPP, logistic growth with seed sources and soil development, deterministic fires, volcanic resets, succession stages; rare classes páramo, oceanic heath, thorn scrub, coastal dunes, kelp, seagrass, hydrothermal field, floating shelf, multiyear sea ice; absence reasons per class | T57–T68 |
| G9 | integration, determinism, old saves, sizes | partial: determinism, save/load, old-save keys | T01, T05, T07 |
| G10 | release | sub-releases only | CHANGELOG |

## Test identifiers → tests

| ID | Test |
|---|---|
| T01 | `test_v53.py::test_t01_legacy_config_keeps_the_old_climate`, `test_config_validation_and_legacy_default` |
| T05 | `test_v53.py::test_t05_deterministic_and_saved` |
| T07 | `test_v53.py::test_t07_views_render_without_physics` |
| T09–T11 | `test_t09_t11_closed_basin_streamfunction_divergence_free`, `test_t10_face_transport_shared_by_both_cells` |
| T12 | `test_t12_disconnected_basins_do_not_share_heat` |
| T13 | `test_t13_strait_carries_flux_only_through_sea_cells` |
| T14 | `test_t14_advection_conserves_heat_without_forcing` |
| T15 | `test_t15_sst_bounded_and_heat_budget_closed` |
| T16 | `test_t16_upwelling_on_an_eastern_boundary_with_equatorward_wind` |
| T17–T18 | `test_t17_t18_polar_ocean_freezes_in_winter_and_retreats_in_summer` |
| T19 | `test_t19_ice_fields_valid` |
| T20 | `test_t20_ice_drift_conserves_volume` |
| T21 | `test_t21_albedo_feedback_changes_absorbed_sunlight` |
| T22 | `test_t22_warm_ocean_has_no_sea_ice` |
| T23, T68 | `test_t23_t68_lakes_are_inland_water_with_lake_ice` |
| T24 | `test_t24_types_and_realms_distinguished`, `test_v53_cryo.py::test_t24_ice_kinds_are_separate_covers` |
| T25 | `test_t25_equator_has_finite_winds` |
| T26 | `test_t26_upper_wind_differs_from_surface_with_a_gradient` (+ southern hemisphere) |
| T27 | `test_t27_levels_and_below_ground` |
| T28, T76 | `test_t28_ash_uses_the_climate_winds_aloft` |
| T29 | the moisture budget closes (1e-15) with the 850-hPa lifting: `test_v4` budget tests + `test_stage_runs_and_validates` |
| T30 | `test_t30_fao56_example_18` |
| T31 | `test_t31_pet_rises_with_radiation_vpd_and_wind`, `test_t31_dry_windy_valley_beats_cloudy_calm_place` |
| T32 | `test_t32_pet_non_negative_and_polar_night` (AET ≤ supply by the v4.5 water balance tests) |
| T33 | `test_v53_cryo.py::test_t33_vialov_profile_shape` |
| T34 | `test_t34_no_seed_without_accumulation` |
| T35 | `test_t35_flow_conserves_volume_in_a_closed_case` |
| T36 | `test_t36_cold_start_short_run_is_not_called_equilibrium` |
| T37 | `test_t37_convergence_needs_mass_and_shape_over_windows` |
| T38 | `test_t38_dry_polar_land_stays_bare_with_negative_balance` |
| T39 | v5.0 `test_v50` mass audit of snow → firn → ice (the sheet is its initial state) |
| T40 | `test_t40_strong_ablation_flag_is_not_the_negative_smb_zone` |
| T41 | `test_t41_flotation_draft_and_freeboard` |
| T42 | `test_t42_cold_ocean_alone_makes_no_shelf` |
| T43–T44 | `test_t43_t44_shelf_flux_is_conserved_and_calving_booked` |
| T45 | `test_t45_basal_melt_uses_ocean_heat` |
| T46 | `test_t46_no_erosion_on_a_frozen_bed` |
| T47 | `test_t47_isostasy_responds_with_delay` |
| T48 | `test_t48_sea_level_from_grounded_ice_not_from_sea_ice` |
| T69–T72 | `test_t69_*`, `test_t70_*`, `test_t71_*`, `test_t72_*` |
| T80 | `test_t80_coupler_reports_its_status` |
| T73 | `test_v53_volcano_terrain.py::test_t73_*` |
| T74 | `test_t74_lava_dam_creates_an_inland_lake` |
| T75 | `test_t75_lava_entering_the_sea_makes_new_land` |
| T77 | `test_t77_subglacial_uses_the_ice_sheet_when_present` |
| T78 | `test_t73_t78_stage_applies_and_rebuilds`, dam test (receivers change) |
| T79 | `test_t79_same_event_cannot_deposit_twice`, `test_regeneration_does_not_deposit_twice` |
| T08 | `test_t08_invalid_delta_rolls_everything_back` |
| T49 | `test_v53_morpho.py::test_t49_no_transport_below_the_threshold` |
| T50 | `test_t50_dry_fetch_conserves_sand_and_moves_it_downwind` |
| T51 | `test_t51_vegetation_and_moisture_hold_the_sand` |
| T52 | `test_t52_dust_is_a_separate_fraction_and_closes` |
| T53 | `test_t53_longshore_sand_is_conserved_along_the_coast` (beach → dune transfer booked in the sand ledger `beach_loss_kg`) |
| T54 | `test_t54_cliff_follows_the_rock_law_not_the_sand_law` |
| T55 | `test_t55_coast_does_not_depend_on_the_window_edge` |
| T56 | `test_t56_relative_sea_level_moves_the_shore` (the stage passes RSLR = 0: no sea-level change is applied in the present state) |
| T57 | `test_v53_ecology.py::test_t57_potential_and_actual_are_separate` |
| T58 | `test_t58_valid_nonnegative_state` |
| T59 | `test_t59_no_fire_without_fuel` |
| T60 | `test_t60_fire_reduces_biomass_and_books_emissions` |
| T61 | `test_t61_succession_after_lava_takes_years`, `test_regrowth_needs_a_seed_source` |
| T62 | `test_t62_paramo_needs_cold_without_winter` |
| T63 | `test_t63_heath_needs_poor_soil_and_oceanic_climate` (gate data) |
| T64 | `test_t64_thorn_scrub_needs_a_long_dry_season` |
| T65 | coastal dunes only on `dune_type == 5` hexes (beach source) — `test_v53_morpho.py` + v5.0 fringe |
| T66 | `test_t66_kelp_and_seagrass_differ_by_substrate_light_depth` |
| T67 | `test_t67_hydrothermal_field_needs_a_source` |
| T06 | `test_t06_rng_independent_of_hex_order` (ecology); ocean/ice use no random numbers |
| T02–T04 | T01 covers old keys; bit-stability of a pre-5.3 save is by construction (no `earth` section → Earth system off), not by a stored golden hash — open as a test |

## Control world: polar continent (centre 70°, seed Pol1, 240×160 hexes of 40 km)

These are the values of the TZ appendix B (point 4), after 5.3b.

| Quantity | Before (v5.0, 100-yr spin-up) | After (5.3b, Vialov seed + SIA) |
|---|---|---|
| `H_max` | ~2.1 km | 5.5 km |
| `H_mean` (ice > 10 m) | 0.7–1.3 km | 2.7 km |
| ice area | ~32 % of land at 55–91° | 15.0 million km² (all land poleward of ~65°) |
| volume | — | 41.0 million km³, ≈ 123 m sea-level equivalent |
| convergence | not checked | NOT_CONVERGED after 50 kyr: drift 8·10⁻⁴ per 100 yr (criterion 10⁻⁴), still growing; reported, not hidden |
| mass ledger | per unit | relative residual 1.6·10⁻¹⁶ |
| negative-SMB share of the ice area | "71 % ablation zone at 72–80°" (counted over all hexes) | 2.5 million of 15.0 million km² of ice (17 %), at the margins |
| shelves | none | 84 hexes; grounding flux 6 500 km³/yr, basal melt 41 km³/yr, calving 6 460 km³/yr |
| bed deflection | none | up to −1.3 km (diagnostic, the relief is not changed) |

## Not done (honest list)

- **Not applied to the base relief.** Dunes, glacial erosion, isostatic deflection and
  sea-level change are layers. Only the volcanic deposits rebuild the drainage
  (snapshot).
- **Coastline.** The one-line model works at hex scale, not on sub-hex shoreline segments.
  The RSLR term exists, but the stage applies RSLR = 0.
- **Ice-sheet grid.** No coarse-grid SIA for 1000×800, and no SSA or buttressing for
  shelves.
- **Ecosystem feedbacks.** Fire does not feed back on the climate. There is no
  ecosystem↔climate iteration (P2), and soil carbon is not exchanged with v4.5 soils.
- **Control worlds.** The 24 control worlds of §22 have no JSON fixtures yet; the
  synthetic scenes in the tests cover 1–5, 8, 10–14 and 19–21 qualitatively.
- **Windows GUI on a physical screen:** NOT VERIFIED.

## Known limits (stated, not hidden)

- **Ocean amplitudes.** East–west SST contrasts are about 1/3 of the Earth's, and the zonal
  mean is calibrated.
- **Coastal temperature.** The SST anomaly reaches land through an upwind proxy, not a
  boundary-layer model.
- **Ice-sheet relief.** The sheets do not modify `elevation_m` or the v4 rivers. Bed
  deflection, the surface with ice and the sea-level equivalent are layers for now
  (G8/G9 integration).
- **Shelf model.** P0 flux-constrained: no buttressing, no SSA.
- **Glacial erosion** is a sliding proxy (SYNTHETIC_PROXY).
- **Cost.** A polar 240×160 world spends ~70 s in the ice spin-up. Coarse-grid SIA for
  1000×800 is not implemented.
- **Windows GUI on a physical screen:** NOT VERIFIED.
