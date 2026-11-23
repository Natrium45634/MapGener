# TZ 5.3 — architecture and decisions (ADR)

## ADR-1. The ocean lives on the world climate raster, inside the climate stage

**Context.** The v4 climate runs on the world raster: 60–170 km cells over the whole model
area, larger than the viewport. Its moisture transport needs the SST of all water upwind,
not only of the window. The TZ asks for 6-neighbour hex faces.

**Decision.**

- Ocean circulation, heat, sea ice and winds aloft are solved on the same raster, by
  `hexworld.earth.coupler.couple` called from `stages.climate.world_climate` when
  `config.earth.enabled`.
- The raster is a 4-face grid. Its FV discretisation keeps the TZ invariants: face fluxes,
  zero normal flow at coasts, exact divergence, conservative upwind heat.
- A new stage, `v53_earth` (after v4.5), samples the state to the hexes and saves it.

**Consequences.**

- The SST feeds the evaporation, the coastal temperature and the moisture transport *in
  the same run*. There is no second climate pass.
- A sub-hex strait narrower than a raster cell is closed for the ocean. This is
  documented, not hidden.
- The cost: 1–8 s for 40 k–160 k ocean cells (one LU factorisation of the Stommel operator
  and four of the heat operator per coupler iteration).

## ADR-2. Circulation: Stommel streamfunction (option A of §4.1)

**Model.**

- `r ∇²ψ + β ∂ψ/∂x = curl τ / ρ`, with ψ on the cell corners and ψ = 0 on coasts and
  borders.
- Face transports are corner differences, so the divergence is exactly zero.
- `r = max(β, β₄₅) · max(2 dx, 150 km)` puts the western boundary layer on the grid and
  stops polar gyres from blowing up.
- The Ekman transport `(τ_y, −τ_x) / (ρ f)` is added in the mixed layer and tapered to
  zero within 5° of the equator. Its divergence is balanced by booked vertical exchange
  (upwelling at `T_sub`, downwelling at `T`).

**Not claimed:** thermohaline circulation, eddies, equatorial current dynamics,
time-dependent flow.

## ADR-3. SST: mixed-layer heat with restoring + calibration of the zonal mean

**Model.**

- Backward Euler, half-month steps, upwind advection by the gyre transport of a 400 m
  layer (of 800 m wind-driven flow) plus Ekman.
- Eddy diffusivity 2·10⁴ m²/s.
- Surface flux `γ (T_eq − T)` with γ = 10 W m⁻² K⁻¹ (large-scale effective restoring), plus
  the seasonal insolation anomaly and the albedo of the ice.

**Calibration.** After half of the spin-up, `T_eq` of every latitude row is shifted so that
the *zonal mean* SST equals the Earth climatology. The model supplies the deviations from
it: warm western boundaries, cold upwelling eastern boundaries, ice.

**Result** (small and large test worlds):

- the signs are right in both hemispheres: in the subtropics the west is warmer than the
  east, and at 40–55° the western side is colder;
- the amplitudes are about 1/3 of the Earth's.

**Quality:** `CALIBRATED_APPROXIMATION`.

## ADR-4. Sea ice: zero-layer energy budget, conservative drift

**Thermodynamics.**

- The water cannot cool below −1.8 °C; the energy deficit freezes ice.
- A warming flux melts ice first.
- Ice insulates cooling by `1/(1 + h/0.5 m)`.
- Concentration `1 − e^(−h/0.5 m)`; albedo 0.06 for open water and 0.5–0.65 for ice.
- Latent heat is booked in the ledger.

**Drift.** With the surface current plus 2 % of the wind, capped at 0.5 m/s, by an
explicit upwind sparse operator that is conservative and positive.

**Types** come from the yearly history: seasonal, multiyear, fast ice along the coast.
Lake ice is separate (`realm = inland water`, `cover = lake_ice`).

## ADR-5. Winds aloft: thermal wind from the climate's own season temperature

**Model.** `U(p) = U_surface − (R/f) ∂T/∂y · 0.8 · ln(p_s/p)`, where ∂T/∂y is the
1000-km-smoothed gradient. Within 12° of the equator, f is held at its 12° value and the
shear is tapered: a proxy, never 1/f. Levels are 850, 500 and 250 hPa, with heights from
the hypsometric equation and a below-ground mask on the hexes.

**Users of the winds aloft.**

- **Climate:** air is lifted over mountains by the 850-hPa wind (no water is created; the
  moisture budget still closes to 1e-15).
- **v5.1 tephra:** the wind at each grain bin's release height, from the same arrays.

## ADR-6. PET: FAO-56 with FAO's own fallbacks

- The equation is the standard FAO-56 Penman–Monteith (Example 18 reproduces 3.9 mm/day).
- Missing humidity, radiation and Tmax/Tmin are estimated by FAO-56 procedures:
  - `ea = e⁰(Tmin − K0)`;
  - Hargreaves radiation `k_Rs √DTR · Ra`;
  - a DTR estimated from the month's wetness.
- The method is recorded as `FAO56_PM_estimated_inputs`.
- The legacy and Hargreaves methods remain selectable.
- The ET0 "day" is 86 400 s for any planet; Ra is a rotation-mean flux.
- The v4 world-climate recycling keeps its temperature-index PET. It is internal to the
  moisture transport, and replacing it would need a recalibration of the precipitation.

## ADR-7. Planet parameters

- Obliquity, day length, year length, solar constant and radius are in
  `config.earth`. The orbit is circular.
- They change:
  - the insolation;
  - the seasonal belts (declination);
  - the seasonal temperature range (ratio of the insolation ranges);
  - the annual zonal temperature (`0.7 ΔS / 3.5 W m⁻² K⁻¹`, capped at ±25 K, a calibrated
    approximation);
  - Coriolis f and β, for the ocean and the winds aloft.
- The tectonic history is not affected (T69–T72).
- The v4.7 topoclimate keeps the Earth's solar geometry. This is a documented limitation.

## ADR-8. Backward compatibility

- **Old saves.** `Config.from_dict` without an `earth` section gives `enabled = False`.
  The v4 key then has no Earth part, so old projects keep their keys and their climate.
- **New worlds.** Enabled by default.
- **When the stored state does not match.** The loader refuses it with a warning
  ("not generated"); a missing state is never read as zeros.

## Status of the TZ gates

| Gate | State |
|---|---|
| G1 planet, winds, PET | done (5.3a) |
| G2 ocean | done (5.3a, approximate amplitudes) |
| G3 sea ice | done (5.3a) |
| G4–G10 | not started in 5.3a; listed in `v5_3_acceptance_report.md` |
