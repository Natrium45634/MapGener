# v5.1 volcanism — baseline audit (G0)

This audit records what existed before v5.1 and where the new stage plugs in.

## What existed (v3 / v4.6 / v5.0)

| Source | What it gives | Limits found |
|---|---|---|
| v3 `hist.volcanoes` | arc and hotspot edifices: position, age, summit, radius, `active`, `submarine`, kind (`arc` / `hotspot`) | No magma, no eruptions, no products. "active" is geological, not "erupting now" |
| v3 `hist.hotspots` | hotspot positions, strength, age-sorted tracks | — |
| v3 `volcanic_potential`, `volcano_id`, `volcanic_age_ma` | raster fields for the v3 «Вулканизм» tab | A potential is not a probability of eruption |
| v4.6 `ridge_piece_id`, `spreading_axis_distance_km`, half rate | spreading axes on the sea floor | Ridges had no volcanism of their own |
| v4.6 `plans.basins` (`rift_graben`) | rift axes, opening rate | Rifts had no volcanic fields |
| v4 hydro | winds (surface only), receivers, ocean mask, glacial potential, sea level | No upper-air winds |
| v4.5 parent material `young_volcanic` | whole-hex young volcanic soil | Whole hex, no age within the recent window |
| v5.0 `barren_volcanic_recent` | biome for fresh volcanic ground (from v4.5 parent material) | Never a fraction of a hex |

## Gaps closed by v5.1

1. Volcanic provinces as sources (arcs, oceanic and continental hotspots, ridges, rifts,
   mud-vent basins) with a supply rate.
2. Systems with separate axes: source, edifice, environment, magmatic status, activity phase,
   lava-lake state.
3. A reservoir mass budget. No eruption can happen without melt.
4. A deterministic event history in a recent window, with stable IDs.
5. Products:
   - lava on a local lattice;
   - tephra by face fluxes;
   - pyroclastic currents (hazard kept apart from deposit);
   - lahars, which need both water and sediment;
   - subglacial melt and outbursts;
   - hydrothermal fields;
   - emergence of new islands;
   - a climate pulse record.
6. Ecological forcing for v5.0, as fractions of a hex and as habitat modifiers.

## Inputs v5.1 does not modify

No v1–v4.7 layer is changed. The deposit thickness, new land and dam flags are separate
layers (`volcanic_deposit_delta_m`, `volcanic_new_land_km2`, `volcanic_dam_flag`). Rivers
are not rerouted. See `architecture.md` › Limitations.
