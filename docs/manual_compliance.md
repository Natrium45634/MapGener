# Engineering manual v1: compliance matrix (HexWorld 4.2)

Every item of the manual's sections 12 and 13 and of its main rules, with its status and
the test that checks it.

Statuses:
- **already**: worked before 4.2; a test exists or was added.
- **improved**: changed in 4.2.
- **declined**: with the reason given.
- **planned**: not done in 4.2.

Tests are in `tests/`; `test_manual.py` is named after the manual's IDs.

## §13: automatic acceptance

| ID | Check | Status | Test |
|---|---|---|---|
| G01 | odd-q ↔ axial ↔ cube, 6 neighbours | already | `test_hexgrid.py::test_offset_axial_roundtrip`, `test_six_distinct_neighbors_at_distance_one` |
| G02 | seed `A` ≠ `a`; a repeat gives the same result | already | `test_generation.py::test_seed_case_changes_the_world`, `test_same_seed_same_hashes`, `test_v4_is_deterministic` |
| G03 | resize / viewport: geology stable in world coordinates | already | `test_generation.py::test_two_windows_of_one_world_agree`, `test_v3.py::test_two_windows_show_the_same_evolved_world` |
| G04 | converging / diverging plates: sign and boundary type | already | `test_boundaries.py::test_head_on_motion_is_convergent`, `test_moving_apart_is_divergent` |
| G05 | 1 mm/yr × 1 Ma = 1 km drift = 1000 m uplift | already | `test_v3.py::test_units_*`, `test_manual.py::test_12_plate_drift_and_uplift_units` |
| G06 | `sea_level` does not change elevation or crust | already | `test_v2.py::test_sea_level_change_floods_monotonically_and_touches_nothing_else`, `test_v3.py::test_sea_level_changes_only_the_masks_of_the_selected_surface` |
| G07 | peaks above passes; connected ridges; no seams | already | `test_v3.py::test_mountain_plan_is_a_connected_structure`, `test_reference_sampler_is_continuous_and_seamless`, `test_mountains.py` |
| C01 | hemispheres and opposite seasons | **improved** (4 seasons) | `test_manual.py::test_C01_hemispheres_have_opposite_seasons_and_the_rain_belt_follows_the_sun` |
| C02 | windward wetter than leeward | already | `test_v4.py::test_rain_shadow_windward_wet_leeward_dry` |
| C03 | E − P ± border flux = Δ storage | **improved** (conservative transport) | `test_manual.py::test_C03_world_moisture_budget_closes`, `test_C03_upwind_transport_is_conservative_and_matches_its_matrix` |
| C04 | +1 km height → −Γ | already | `test_v4.py::test_temperature_falls_with_height_at_the_lapse_rate`, `test_manual.py::test_12_C04_*` (per season too) |
| H01 | 600 mm/yr on 1 km² ≈ 0.019 m³/s | already | `test_manual.py::test_12_H01_*` |
| H02 | receiver is a real D6 neighbour | already | `test_manual.py::test_H02_H03_*` (exception: the second exit of a lake is linked across the lake to its one outlet) |
| H03 | no cycles; every path ends at the sea, a lake or a closed basin | already | `test_manual.py::test_H02_H03_*`, `test_v4.py::_assert_flow_graph_is_sound` |
| H04 | flat plain: finite route, stable tie-break | already | `test_manual.py::test_H04_flat_plain_drains_with_a_finite_stable_route` |
| H05 | priority-flood never raises the real surface; lake not above the sill | already | `test_manual.py::test_H05_*`, `test_v4.py::test_wet_depression_fills_and_spills_through_its_sill` |
| H06 | Q_down = Σ upstream + local − losses | already | `test_manual.py::test_12_H06_confluence_3_plus_5_plus_local_2_is_10` |
| H07 | stores, rivers, rain and evaporation close per seasonal step | **improved** (seasonal stores) | `test_manual.py::test_H07_*` (3 tests) |
| E01 | no water → no fluvial erosion | already (threshold **improved**) | `test_manual.py::test_12_E01_*` |
| E02 | hillslope diffusion conserves volume on a closed grid | already | `test_manual.py::test_E02_*` |
| E03 | different rock → different erosion | already | `test_v4.py::test_erosion_cuts_valleys_and_depends_on_rock` |
| E04 | production + import − export − sink = Δ sediment | already | `test_v4.py::test_v4_project_layers_are_consistent` (`closure_error_m3`) |
| E05 | no negative thickness or NaN; error falls with dt | already | `test_manual.py::test_E05_*` |
| I01 | archipelago origin follows volcanism, crust and history | planned | islands are the next stage (manual §9). The coast data are kept |
| I02 | no `fjord` without glacial history | already | `test_coasts.py::test_generated_world_coasts_follow_their_causes` (fjord ⇒ LGM ice) |
| P01 | older projects open and are not changed | already | `test_v2.py::test_v1_project_opens_and_is_not_changed`, `test_v4.py::test_v2_project_goes_to_v4_and_keeps_its_baseline`, `test_stale_v4_data_is_never_loaded_silently` |
| P02 | tabs switch without regeneration | already | `test_v2.py::test_views_are_pure_presentations`, `test_v3.py::test_gui_v3_tabs` |
| P03 | 3 sizes: time, RSS, versions | improved | `docs/REPORT_4.2.md` (360×240, 500×300, 1000×800) |

## §12: control calculations (all in `test_manual.py`)

| Case | Expected | Test |
|---|---|---|
| regular hex, side 10 km | A ≈ 259.808 km², centres 17.321 km | `test_12_regular_hex_with_a_10_km_side` |
| plate 35 mm/yr × 20 Ma | 700 km | `test_12_plate_drift_and_uplift_units` |
| uplift 0.5 mm/yr × 1 Ma | +500 m | same |
| Γ = 6.5 °C/km, +1500 m | −9.75 °C (annual and every season) | `test_12_C04_*` |
| 600 mm/yr × 1 km² | 0.0190 m³/s | `test_12_H01_*` |
| 10 m drop over 2000 m | S = 0.005 | `test_12_slope_10_m_over_2000_m` |
| K = 0.002, Q/Q0 = 4, m = 0.5, S = 0.1 | 0.0004 m/yr, 0.4 m per 1000 yr | `test_12_E01_*` |
| H = +24 m, sea level +25 m | flooded, −1 m | `test_12_sea_mask_24_m_under_a_25_m_sea` |
| confluence 3 + 5 + local 2 | 10 m³/s | `test_12_H06_*` |
| 40 of 100 cells above the sea | land fraction 0.4 | `test_12_land_fraction_40_of_100` |

## Main rules of sections 5–8

| Rule | Status | Where |
|---|---|---|
| §5.1 seasons: T_monthly / seasonal, snow season | **improved**: 4 seasons; DJF/JJA layers and views | `climate.season_temperature` |
| §5.2 mass-conservative transport, CFL sub-steps, budget | **improved** | `climate.upwind_operator`; `WorldClimate.balance` |
| §5.2 orographic rain shares the available water | already: `P ≤ q` each step | `world_climate` |
| §5.3 `AET ≤ PET`, `AET ≤ available water`, degree-day melt | **improved** | `climate.season_balance` |
| §5.3 do not call the PET "Penman–Monteith" | already | `potential_et` docstring: "Thornthwaite-like" |
| §6.1 bucket model with a balance check | **improved** | `season_balance`, residual < 1e-9 mm |
| §6.2 D6 + priority-flood on a copy, stable flats | already | `hydrology.route` |
| §6.2/4 a cut viewport is not surrounded by outlets | **improved** | `pipeline_v4.edge_outlets` (`hydrology.edge_outlets`) |
| §6.3 channels by Q and area; width W = a Q^b | **improved**: width, depth, velocity | `hydrology.hydraulic_geometry` |
| §6.3 meanders | planned [Н] | — |
| §7.1 E = max(0, K (Q/Q0)^m S^n − E_c), stable solver | **improved**: E_c added | `landscape.implicit_incision` |
| §7.2 conservative hillslope flux, sub-steps | already | `landscape._laplacian`, `evolve` |
| §7.3 capacity → deposit the excess, closed budget | already | `landscape.evolve` |
| §7.4 glaciers, fjord carving | planned [Н] | `glacial_potential`, `lgm_ice_mask` kept |
| §8 climate ↔ relief within an epoch | already: climate recomputed every erosion step | `erosion.update_climate` |
| §11 1000×800: dense NumPy arrays, O(N) passes | checked | `docs/REPORT_4.2.md` |
