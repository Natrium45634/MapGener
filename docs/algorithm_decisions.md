# Algorithm decisions (HexWorld 4.2)

Format of manual §14: **formula · input units · source · why this variant · main limits ·
golden test · time / memory**. Only the algorithms changed or checked against the manual
in 4.2 are listed. Everything older is described in `docs/research_v4.md` and
`docs/RESEARCH.md`.

Status words: **implemented** (code + test), **checked** (already matched the manual,
test added), **prototype**, **planned**, **declined**.

---

## 1. Seasons and latitude — implemented (`stages/climate.py`)

- **Formula.**
  - `T_s = T_annual + 0.9·A·phase_s·sign(lat)`, with phase = −1, 0, +1, 0 for DJF, MAM, JJA, SON.
  - `A = seasonality·(1+0.32|lat|)·(0.25+0.75·continentality)` is half the annual range.
  - 0.9 is the mean of a sinusoid over a 3-month window.
- **Units.** °C, degrees of latitude (north = map up, −y).
- **Source.** Manual §5.1. The amplitude was calibrated in v4.0 (`research_v4.md`).
- **Why.** Four seasons are the manual's minimum. They are enough for snow, spring floods, monsoons and Mediterranean (winter) rain. Twelve months would cost 3× the world transport time.
- **Limits.**
  - No lag of the seasons behind the sun: MAM = SON.
  - Daily temperature spread is a fixed normal with σ = 4 °C.
- **Test.** `test_C01_hemispheres_have_opposite_seasons_and_the_rain_belt_follows_the_sun`.
- **Cost.** Negligible.

## 2. Seasonal circulation and monsoon — implemented

- **Formula.**
  - Belts are those of `lat_eff = lat − 23.44°·phase_s·(0.25 + 0.45·continentality_broad)·(1 − (lat/90)²)`.
  - Monsoon wind = `0.45 m/s per °C/1000 km` × the gradient of the seasonal temperature anomaly smoothed over ~900 km, fading poleward of ~38°.
  - Mountains slow the wind: `exp(−h/4000 m)`.
- **Units.** m/s, °C/1000 km.
- **Source.** Manual §5.2. ITCZ excursions of ~5° over oceans and up to ~15–20° over large continents (standard climatology).
- **Why.** The belts must move with the sun, and monsoon reversal must come from land–sea heating, not be painted.
- **Limits.**
  - No Coriolis turning of the monsoon flow.
  - No pressure field.
  - The taper keeps the polar belts on the map.
- **Test.** `test_C01_*`: the rain belt is north of the equator in JJA and south in DJF.
- **Cost.** 4 wind fields of the world raster.

## 3. Mass-conservative moisture transport — implemented

- **Formula.**
  - Finite-volume first-order upwind: `M_i += Σ F_j→i − Σ F_i→j + E_i − P_i`, with face flux `F = max(0, u·n)·q_upwind·Δt/Δx`.
  - Courant sub-steps keep every cell's outgoing Courant sum ≤ 0.9, so no cell gives away more water than it has.
  - The open border is a zero-gradient ghost cell; its inflow and outflow are counted.
  - The operator is assembled once per season as a sparse matrix: `q ← A q`, `in = b_in·q`, `out = b_out·q`.
  - Four quasi-steady seasons of 150 physics steps each after a 150-step spin-up. The season's precipitation is averaged over its last 70 steps.
- **Units.** Relative column water `q`. Converted to mm by one scale fixed so the annual ocean mean equals the profile target (1100 mm/yr Earth-like).
- **Source.** Manual §5.2. Upwind finite volumes (textbook).
- **Why.**
  - The v4.0 semi-Lagrangian step was not conservative.
  - Flux form conserves mass by construction, and convergence zones (ITCZ) gain water physically.
  - The sparse matrix makes the step 3.6× faster than array slicing.
- **Limits.**
  - Numerical diffusion of first-order upwind.
  - `q` is a conditional saturation proxy, not kg/m² of a real atmosphere (manual: "do not present it as physical humidity").
- **Test.**
  - `test_C03_world_moisture_budget_closes`: ΔM = E − P + in − out, relative residual < 1e-12. Measured: 1e-16…1e-15.
  - `test_C03_upwind_transport_is_conservative_and_matches_its_matrix`.
- **Cost.** 512×450 world raster: 5.6 s (v4.0: 3.5 s).

## 4. Snow, soil, groundwater — implemented (`season_balance`)

- **Formula.**
  - Snow share `Φ((1 °C − T_s)/4 °C)`.
  - Melt = `min(SWE, 3 mm/(°C·day) × PDD_s)`, with expected positive degree days `PDD = days·(σ φ(T/σ) + T Φ(T/σ))`.
  - Soil bucket of 150 mm: `AET = Fu(soil + water, PET_s)`.
  - Surplus over the capacity: deep loss share `d`, recharge 35 %, the rest quick flow.
  - Baseflow `1 − exp(−0.25/τ)` of groundwater per season, τ = 1 yr.
  - Five yearly cycles from empty stores; the last one is reported.
  - A snowpack still growing after the spin-up is a glacier in equilibrium: its yearly gain leaves as ice discharge in the runoff.
- **Units.** mm per season; annual layers in mm/yr.
- **Source.**
  - Manual §5.3 and §6.1.
  - Degree days: Calov & Greve 2005.
  - Fu 1981 (Budyko curve) within a season.
- **Why.**
  - Seasonal snow storage gives spring floods and the snowmelt share.
  - Starting empty makes every store rise monotonically, so over the reported year no store loses water. Hence `AET ≤ P` in every hex, which the validator checks.
- **Limits.**
  - PET is a heuristic curve of temperature (not Penman–Monteith; manual §5.3).
  - No glacier dynamics, no frozen ground.
  - The world-scale inflow routing still uses annual Budyko runoff.
- **Test.** `P = AET + R + D + ΔS` per hex, residual < 1e-9 mm. Measured ~1e-11.
  - `test_H07_seasonal_stores_close_the_balance_in_every_hex`.
  - `test_H07_snowmelt_gives_a_spring_or_summer_flood_and_glaciers_discharge`.
  - `test_H07_hex_climate_annual_layers_are_sums_of_the_seasons`.
- **Cost.** 0.04 s on 86 400 hexes; called once per erosion step.

## 5. Window-edge outlets — implemented (`pipeline_v4.edge_outlets`)

- **Rule.**
  - A border hex drains out of the window only where the drainage of the whole model area (coarse 8-neighbour priority-flood routing of the world raster) crosses the window edge outward.
  - Hexes within 0.75 of a world cell of each crossing qualify.
  - Where the window edge is the edge of the model area, that whole side stays an outlet.
  - Option `hydrology.edge_outlets = "world"` (default) or `"all"` (v4.0 behaviour).
- **Source.** Manual §6.2 item 4: "do not make every side of a cut viewport an outlet".
- **Why.** Before, every border hex was a sink, so basins cut by the window leaked water out of the side.
- **Limits.**
  - The crossing is known only to one world cell (60–90 km).
  - A window with no real exit can hold an edge lake that the full world would not have.
  - True halo routing on hexes outside the window is **planned**.
- **Test.**
  - World mode gives identical results with both settings (`edge_outlet_hexes` in the summary).
  - Regional comparison in `REPORT_4.2.md`.
- **Cost.** One coarse routing, shared with the external inflow.

## 6. Stream power with threshold — implemented (`landscape.implicit_incision`)

- **Formula.**
  - `E = max(0, K A_eff^m S − E_c)`, solved implicitly from the outlets up.
  - `z_new = (z + f z_r + E_c Δt)/(1+f)` with `f = K A^m Δt / L`, applied only where `f (z − z_r) > E_c Δt`.
  - `E_c` is `erosion.incision_threshold_m_ma` (m/Ma; default 0, so the results of 4.1 are kept).
- **Source.** Manual §7.1. Landlab `StreamPowerEroder` (`threshold_sp`); Braun & Willett 2013.
- **Why.** The implicit solver is stable at any step and never cuts below the receiver.
- **Limits.**
  - n = 1 only.
  - Long steps are stable but not exact for transients (the Landlab warning, confirmed by our convergence test).
- **Test.**
  - `test_12_E01_stream_power_control_and_zero_water`: 0.0004 m/yr, 0.4 m per 1000 yr, Q = 0 gives no incision.
  - `test_E05_implicit_incision_is_stable_and_converges_with_smaller_steps`.
- **Cost.** Unchanged (one pass in topological order).

## 7. Hillslope diffusion — checked

- **Formula.**
  - Symmetric hex Laplacian `(2/3L²) Σ (z_j − z)`, i.e. the antisymmetric edge flux of manual §7.2.
  - Explicit sub-steps under `κΔt/L² ≤ 0.2`.
- **Test.** `test_E02_hillslope_diffusion_conserves_volume_on_a_closed_grid`: relative 1e-12.
- **Limits.** Land-to-sea transfer is booked as `hillslope_to_sea_m3`.

## 8. Hydraulic geometry — implemented (`hydrology.hydraulic_geometry`)

- **Formula.**
  - `W = 5 Q^0.5` m, `D = 0.3 Q^0.4` m, `V = Q/(W D)`.
  - At Q = 1000 m³/s: 158 m, 4.8 m, 1.3 m/s.
- **Source.** Leopold & Maddock 1953; manual §6.3 ("`b ≈ 0.3–0.6` as a starting range").
- **Use.**
  - Segment fields `width_m`, `depth_m`, `velocity_m_s`.
  - Layers `river_width_m`, `river_depth_m`.
  - Channel transmission losses use the same width.
- **Test.** `test_hydraulic_geometry_grows_with_discharge`, including continuity `Q = W D V`.

## 9. Declined or deferred

| Item | Decision | Reason |
|---|---|---|
| 12 months instead of 4 seasons | declined for now | 3× world transport time; 4 seasons give every effect the manual lists |
| D8 / raster tools (pysheds, RichDEM epsilon fill) | declined | wrong topology on hexes; our flood order already resolves flats |
| MFD routing | declined | D6 SFD with Rho8 randomisation (4.1) gives dendritic networks; MFD blurs rivers at 30–80 km hexes |
| Particle hydrology (SimpleHydrology) | declined | no mass budget per basin; no license file |
| Re-evaporation of land rain as the main inland moisture source | tried, declined | made wet coasts wetter instead of watering interiors (see `REPORT_4.2.md`) |
| Glacier dynamics, fjord carving | planned | manual §7.4 [Н]; `glacial_potential` and the LGM mask are kept |
| Halo routing on hexes outside the window | planned | edge outlets from the world drainage are the interim rule |
