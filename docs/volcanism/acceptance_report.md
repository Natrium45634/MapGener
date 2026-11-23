# TZ 5.1 — acceptance report

## Gate status

| Gate | Content | Status |
|---|---|---|
| G0 | Audit, configuration, schema, versions | DONE (`baseline_audit.md`, `source_audit.md`) |
| G1 | Provinces, systems, magma, reservoir budget | DONE |
| G2 | Deterministic events, styles, mass partition, explicit / aggregated | DONE |
| G3 | Lava: local D6 lattice, mass conservation, cooling, hex fractions | DONE / APPROXIMATE rheology |
| G4 | Tephra FV on hex faces with grain bins; PDC hazard vs deposit; lahars (water + sediment) | DONE / APPROXIMATE (synthetic upper wind, weighted PDC routing) |
| G5 | Submarine (pressure, hydrothermal, emergence), subglacial (energy-bound melt, outbursts), gases, climate pulse, icy world, mud vents | DONE / APPROXIMATE; mud volcanoes are flags only |
| G6 | Ecology forcing in v5.0 (barren share, recovery, modifiers) | DONE |
| G7 | Pipeline stage, keys, reuse, save / load, CLI | DONE |
| G8 | Map tab, hex report, settings, tests ≥ 68, scenario worlds ≥ 30, docs | DONE (93 tests, 36 worlds) |

## Acceptance checks

**Mass budgets close.** Over 36 worlds the relative residuals are:

| Budget | Residual |
|---|---|
| Reservoir | ≤ 4.3e-15 |
| Lava | ≤ 1.0e-15 |
| Tephra | ≤ 1.5e-16 |
| PDC | ≤ 1.1e-16 |
| Lahars | ≤ 7.7e-17 |

**Determinism.**

- The same seed and config give identical layers and event IDs (tested).
- A rerun reuses the cached `v51_volcanism` and `v50_biomes` stages.

**Physical constraints.**

- No eruption without melt.
- Lahars need both water and sediment.
- Melt is capped by energy and by the ice that is there.
- A deep-submarine vent has no eruption column.
- Stratospheric SO₂ appears only above the tropopause.
- There is no cryovolcanism on a silicate world.

**Fractions.**

- A lava flow, a PDC, a lahar or the barren ground is a share of a hex.
- No barren land and no hazard index is set at sea.

## Not done or simplified

These items are reported honestly; see also `architecture.md` › Limitations.

- **Terrain feedback (NOT_IMPLEMENTED).** Deposits, new land and dam flags are layers only.
  The v4 rivers and lakes are not rerouted.
- **Upper-air winds (APPROXIMATE).** They are a synthetic zonal profile, because the
  generator has no upper-air circulation.
- **Lava (APPROXIMATE).** It is a cellular model with residual thickness and a cooling
  length, not a Bingham momentum solver.
- **Subglacial ice (APPROXIMATE).** It comes from a proxy (v4 glacial potential), because
  the v5.0 ice sheet is computed later in the pipeline.
- **Climate pulses.** They are recorded only and do not change the present climate.
- **Windows GUI on a physical screen: NOT VERIFIED.** The views render headless in the
  tests, and the Windows installer CI runs an offscreen self-test.
