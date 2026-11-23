# v5.1 architecture

```
v3 history ─┐  v4.6 ridges / rifts ─┐  v4 winds, drainage, sea, ice ─┐  v4.7 snow
            ▼                       ▼                                ▼
      provinces.py ──► systems.py (magma.py: profile, viscosity, Reservoir)
                              │
                       scheduler.py  (deterministic events, styles.py: style + mass partition)
                              │
         ┌──────── explicit events (largest / most recent / on land) ────────┐
         │  lava.py      local D6 hex lattice, cooling-limited, volume exact  │
         │  tephra.py    FV advection-diffusion-settling on world-hex faces  │
         │  hazards.py   PDC energy cone (hazard) + weighted routing (deposit)│
         │  lahars.py    water AND sediment, routed on v4 receivers          │
         │  environment.py  subglacial melt / outbursts, hydrothermal,       │
         │               emergence, degassing, climate pulse, icy world      │
         └───────────────────────────────────────────────────────────────────┘
         aggregated events: conserved totals, analytic fall field (fast mode)
                              │
                 pipeline_v51.py ─► VolcanismData (layers, systems, events, summary)
                              │
                 pipeline_v50.py: volcanic_fringe (barren share) + habitat modifiers
```

## Pipeline position

- **Stage key.** `v51_volcanism` runs after v4.7 and before v5.0, through
  `generate(through="v51")` or the default `through="v50"`.
- **Inputs of the key.** The algorithm version, the seed, the volcanism parameters and the
  upstream keys of v3, v4.6, v4, v4.5 and v4.7.
- **v5.0 dependency.** The v5.0 key includes the v5.1 key. A change in volcanism therefore
  recomputes the biomes, and reusing the project reuses both stages.

## Three clocks

| Clock | Unit | Where |
|---|---|---|
| Geological | Ma | Provinces, edifices, v3 ages |
| Event | years before present, within `event_window_years` | Eruptions, products |
| Ecological | years | Vegetation recovery `recovery_years(T, P)`: wet tropics ~150 yr, cold or dry climates > 1500 yr |

## Determinism

- **Event times.** Event `k` of a volcano uses the thresholds `u_k = H(seed, volcano, k)`.
  No generator is advanced by the time step.
- **Event IDs.** `E-` + `sha256(seed, province, volcano, k, algorithm)`.
- **Style draw.** One draw per event, taken from the event key, and only after the
  conditions are fixed.

## Explicit vs aggregated events

- **Number of explicit events.** Each explicit event costs about the same, whatever the
  world size. Their number is `min(max_explicit_events, 24 + hexes/1000)`.
- **Ranking.** Events are ranked by size + recency + inside-window + VEI − deep-sea
  penalty. Deep-submarine effusions take at most a quarter of the slots.
- **Aggregated events.** All other events are kept per volcano as conserved totals: DRE,
  lava, tephra, gas, SO₂ and the age span.
- **Aggregated lava.** It covers `V / 6 m` of area in the vent hex (at most 60 %). Its
  freshness is `Σ V·max(0, 1 − age/t_rec) / ΣV`.
- **Aggregated tephra.** It gets a normalised analytic downwind field (`fast_analytic`).

## Ecology coupling (v5.0)

- **Barren share.** `volcanic_barren_fraction` = 1 − Π(1 − fᵢ·rᵢ), where `fᵢ` is the hex
  share of a lava, PDC or lahar deposit and `rᵢ = max(0, 1 − age/t_rec)`. Thick fresh ash
  is added: (load − 300 kg/m²)/3000 × e^(−age/50). The share is capped at 0.95 and is 0 at
  sea.
- **How v5.0 uses it.** v5.0 takes that share out of the dry units of the hex and gives it
  to one `barren_volcanic_recent` unit, like the tidal fringe. A 4 km flow is a few per cent
  of a 25 km hex.
- **Habitat modifiers** (append-only bits):
  - `geothermal_habitat`;
  - `volcanic_degassing_stress`;
  - `hydrothermal_vent_field` (sea floor);
  - `volcanic_ash_young`.

## Limitations (honest status)

| Item | Status |
|---|---|
| Lava rheology | APPROXIMATE. Cellular D6 with residual thickness and a cooling length instead of a Bingham momentum solver |
| Upper winds | APPROXIMATE / synthetic zonal jet (`synthetic_upper_wind` flag). The generator has no upper-air circulation |
| Tephra release | APPROXIMATE. Fixed release heights per grain bin, no Suzuki profile, no plume dynamics |
| PDC | The energy-cone hazard is standard. The routing of the deposit is a weighted distribution, not a depth-averaged flow |
| Lahars | Receiver routing with bulking. No hydraulics, no dilution downstream |
| Terrain feedback | NOT_IMPLEMENTED for rerouting. Deposits, new land and dams are layers; v4 rivers and lakes are not recomputed |
| Subglacial ice thickness | APPROXIMATE proxy (`glacial_potential × 400 m`). v5.0 ice comes later in the pipeline |
| Climate pulses | A record only. The present climate is not changed by them |
| Icy world | APPROXIMATE. Cryomagma swap (no sulfur, ice products); no separate cryo-physics |
| Mud volcanoes | Fields and flags only. No mud-flow products |
| CO₂ pooling | A hazard flag for calderas or maars in closed lows. No gas dispersion |
| Windows GUI on a physical screen | NOT VERIFIED (headless rendering tested) |
