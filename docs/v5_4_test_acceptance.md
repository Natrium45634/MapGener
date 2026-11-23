# TZ 5.4 — test acceptance (living document)

**Status:** 5.4a (seasonal circulation, moisture transport, ITCZ / monsoons, climate
normals, diagnostic layers) — implemented and tested. 5.4b (storm climatology TC / ETC /
fronts / AR, upper-wind integration with v5.1 ash) — implemented as calibrated proxies.
5.4c (weather year, interannual mode, droughts, event → hydrology delta) — implemented in
reduced order, opt-in (`atmosphere.weather_year`). Nothing here is a full weather system,
a GCM or a forecast.

Windows GUI: NOT VERIFIED (no Windows machine and no real display were used; the
`atmosphere_seasons` view is exercised offscreen by tests only — that is not a display test).

Run: `pytest tests/test_v54_atmosphere.py tests/test_v54_integration.py`
(unit file ~2 s; integration file generates one 100×100 world, ~2 min).
U = `tests/test_v54_atmosphere.py`, I = `tests/test_v54_integration.py`.

## Gates

| Gate | State | Evidence |
|---|---|---|
| G0 audit | done | `v5_4_baseline_audit.md` |
| G1 planet / inertia | done | U T03, T04, T11, T14 |
| G2 FV operators, levels | done | U T13, T15–T23, hex mesh test |
| G3 moisture, P, ITCZ | done | U T21–T30, T38 |
| G4 monsoon diagnostics | done | U T31–T40, N08, N12, N18 |
| G5 storm climatology | done, CALIBRATED_PROXY | I T41–T60 |
| G6 weather year | done, reduced order | I T61–T70 |
| G7 downstream integration | partial: v4.5 months, tephra windAt, flood routing delta; no dynamic soil / ice / biome commit | I T71–T75, v45 test |
| G8 GUI / save / load | done (offscreen) | I T76, T79; T80 = NOT VERIFIED |
| G9 24 scenarios, benchmark | fixtures + tests; benchmark in `v5_4_benchmarks_and_coverage.md` | `tests/fixtures/v5_4/S01–S24.json` |
| G10 migration / audit | old configs load with the atmosphere off | I T09 |

## T01–T80

| ID | Test | State |
|---|---|---|
| T01 | U `test_t01_missing_v53_input_is_reported_not_hidden` | pass |
| T02 | viewport is display only: the atmosphere runs on the world raster in world km; storm points outside the window — I `test_n17_t59…` | pass (indirect) |
| T03 | U `test_t03_hemispheres_mirror_by_half_a_year` | pass |
| T04 | U `test_t04_zero_obliquity_has_no_season` | pass |
| T05 | U `test_t05_regional_window_is_marked_approximate` | pass |
| T06 | U `test_t06_t17_levels_below_ground_and_extrapolation` | pass |
| T07, T08 | U `test_t07_t08_seed_namespaces`; I `test_t49_n13…` (normals do not read the event seed) | pass |
| T09 | I `test_t09_old_config_loads_without_atmosphere` | pass |
| T10 | I `test_t10_one_sst_one_ocean` | pass |
| T11 | U `test_t11_land_and_sea_have_different_inertia` | pass |
| T12 | U `test_t12_t01_aquaplanet_has_no_continental_monsoon` | pass |
| T13 | U `test_t13_n16_equator_is_finite` | pass |
| T14 | U `test_t14_rotation_changes_the_belts_not_fixed_30_60` | pass |
| T15 | U `test_t15_850_is_not_the_surface_wind` | pass |
| T16 | U `test_t16_jet_aloft_with_quality` | pass |
| T17 | U `test_t06_t17…` | pass |
| T18 | U `test_t18_pressure_and_speeds_are_physical` | pass |
| T19 | U `test_t19_mountains_slow_the_surface_wind` | pass |
| T20 | U `test_t20_resolution_keeps_the_large_scale_winds` (two raster resolutions; 360×240 vs 500×300 hex windows not run) | pass (reduced) |
| T21 | U `test_t21_empty_dry_column_gives_no_rain` | pass |
| T22 | U `test_t22_uniform_transport_creates_no_water` | pass |
| T23 | U `test_t23_n10_internal_face_is_antisymmetric`, `test_hex_mesh_has_six_faces…` | pass |
| T24 | U `test_t24_more_evaporation_more_vapour_and_rain` | pass |
| T25 | U `test_t25_convergence_strengthens_rain` | pass |
| T26 | U `test_t26_orography_moves_water_windward` | pass |
| T27 | U `test_t27_dry_cold_wind_gives_little_rain` | pass |
| T28 | U `test_t28_open_boundary_import_is_booked` | pass |
| T29 | U `test_t29_n15_halving_the_step_keeps_the_budget` | pass |
| T30 | U `test_t30_annual_sums_and_day_weighted_temperature` | pass |
| T31, T33 | U `test_t31_t33_tropical_continents_rain_in_their_summer` | pass |
| T32 | U `test_t32_hot_interior_without_moisture_stays_dry` | pass |
| T34–T36 | U `test_t34_t35_t36_rain_regimes` | pass |
| T37 | U `test_t37_warmer_sea_moves_the_rain` | pass |
| T38 | U `test_t38_itcz_segments_may_be_several` | pass |
| T39 | U `test_t39_false_start_is_not_an_onset` | pass |
| T40 | U `test_n18_t40_reversal_without_moisture_is_no_monsoon` | pass |
| T41–T45 | I `test_t41_t42_t43_t44_t45_genesis_gates` | pass |
| T46 | I `test_t46_track_follows_the_steering_wind_in_metres` | pass |
| T47 | I `test_t47_landfall_weakens` | pass |
| T48 | I `test_t48_mountains_take_more_storm_rain` | pass |
| T49 | I `test_t49_n13_event_seeds_change_tracks_not_the_climatology` | pass |
| T50, T60 | I `test_t50_t60_n14_year_ledger_and_shared_vapour` | pass |
| T51, T52 | I `test_t51_t52_frontogenesis_needs_a_gradient` | pass |
| T53 | I `test_t53_eady_is_symmetric_between_hemispheres` | pass |
| T54–T56 | I `test_t54_t55_t56_fronts_are_continuous_and_cold_ones_snow` | pass |
| T57, T58 | I `test_t57_t58_atmospheric_river_detection` | pass |
| T59 | I `test_n17_t59_tracks_are_not_clipped_by_the_viewport` (an incoming *boundary* storm catalogue is not implemented: `incoming_storms` = not_available) | pass (partial) |
| T61 | I `test_n07_t61_markov_wet_share` | pass |
| T62 | I `test_t62_gamma_amounts` | pass |
| T63 | I `test_t63_neighbours_share_the_weather` | pass |
| T64, T71, T72, T74, T75, T77 | I `test_t64_t71_t72_t74_t75_t77_weather_year_stage` | pass |
| T65, T68, T69 | I `test_t65_t69_years_differ_and_no_ocean_mode_without_a_basin` | pass |
| T66, T67 | I `test_t66_t67_droughts_relative_to_the_own_norm` | pass |
| T70 | I `test_t70_calendars` | pass |
| T73 | I `test_t73_wind_at_for_ash` | pass |
| T76 | I `test_t76_save_load_keeps_the_snapshot` | pass |
| T78 | benchmark script, not a pytest case (time / RSS in `v5_4_benchmarks_and_coverage.md`) | measured, not gated |
| T79 | I `test_t79_views_do_not_change_the_snapshot` | pass |
| T80 | I `test_t80_windows_gui_not_verified_is_stated` — Windows GUI: NOT VERIFIED | stated |

## Numeric controls N01–N18

| ID | Test |
|---|---|
| N01–N06 | U `test_n01_n06_units` |
| N07 | I `test_n07_t61_markov_wet_share` |
| N08 | U `test_n08_wettest_three_months_cyclic` |
| N09 | I `test_n09_poisson_rate` |
| N10 | U `test_t23_n10_internal_face_is_antisymmetric` |
| N11 (+ negative control) | U `test_n11_budget_validator_and_negative_control` |
| N12 | U `test_n12_wet_season_share` |
| N13 | I `test_t49_n13…` |
| N14 | I `test_t50_t60_n14…` |
| N15 | U `test_t29_n15…` |
| N16 | U `test_t13_n16…` |
| N17 | I `test_n17_t59…` |
| N18 | U `test_n18_t40…` |

## Not done / limits (honest list)

- `full` precipitation coupling is not the default: in subtropical windows it gives
  noticeably more desert than the v4 calibration (see calibration doc); default
  `seasonality_only` keeps the v4 annual totals.
- No `specified_boundary` user input form (API argument only); no incoming-storm catalogue.
- The circulation does not respond to latent heating; mid-level humidity is the column RH.
- ETC / AR / front climatologies are proxies; TC intensity is an upper-bound ODE.
- Weather-year effects are delta layers (actual rain, flood day routed through the rivers,
  drought, onset); no dynamic soil / snow / glacier / vegetation commit from one year.
- Ocean mode of the interannual variability only in `planetary_background` worlds with an
  equatorial upwelling tongue (thermodynamic response, circulation fixed).
- T20 at the 360×240 / 500×300 hex sizes and T78 at 1000×800 are measured, not tests.
