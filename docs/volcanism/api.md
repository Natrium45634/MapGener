# API and data format

## Configuration — `Config.volcanism: VolcanismParams`

| Field | Default | Meaning |
|---|---|---|
| `enabled` | True | Run the v5.1 stage |
| `world_profile` | `earthlike_silicate` | Or `icy_world` (cryovolcanism). Cryovolcanism never happens on a silicate world |
| `physics_quality` | `balanced` | `fast` / `balanced` / `high`: lava pulses, tephra hours and window |
| `event_window_years` | 10 000 | Recent window, 100 … 1e6 |
| `max_explicit_events` | 160 | Upper bound. The effective number is `min(max, 24 + hexes/1000)` |
| `eruption_rate_scale` | 1.0 | × intensity |
| `hotspot_activity` | 1.0 | × hotspot strength |
| `rift_volcanism`, `ridge_volcanism` | True | Sources on / off |
| `lava_lakes` | True | Rare lava lakes in open basaltic conduits |
| `tephra_bins` | 4 | 2..4 grain classes |
| `climate_aerosol` | True | Record of the stratospheric SO₂ pulses |
| `subglacial`, `hydrothermal`, `mud_volcanoes` | True | Sub-models on / off |
| `lava_cell_m` | 400 | Cell of the local lava lattice (it is doubled for very large flows) |

All of these fields are in the world settings dialog, group «Вулканизм».

## Project data — `project.volcanism: VolcanismData | None`

`None` means "not generated", never "no volcanoes".

- **`layers`** — (H, W) arrays:
  - `volcano_count`, `volcano_status` (0 none, 1 extinct, 2 uncertain, 3 dormant,
    4 active);
  - `lava_fraction`, `lava_volume_m3`, `lava_youngest_years_bp`;
  - `tephra_load_kg_m2`, `tephra_thickness_m`;
  - `pdc_hazard_fraction`, `pdc_deposit_fraction`, `lahar_fraction`;
  - `volcanic_hazard_index` (land only), `volcanic_barren_fraction`,
    `volcanic_ash_stress`, `degassing_stress_fraction`;
  - `co2_pooling_flag`, `mud_volcano_flag`, `geothermal_heat_mw`, `hydrothermal_kind`;
  - `vegetation_recovery_years`, `volcanic_deposit_delta_m`, `volcanic_new_land_km2`,
    `volcanic_dam_flag`.
- **`systems`** — dicts: id, province, source, edifice, environment, status, phase,
  position, vents, magma (with viscosity and class), reservoir budget, last eruption, lava
  lake, hex.
- **`events`** — the explicit events: id, sequence, `years_bp`, duration, vent, style
  phases, conditions, products and per-product `results`.
- **`aggregates`** — per volcano totals of the other events.
- **`summary`** — counts, VEI and style distributions, budgets, quality labels and
  `extras` (hydrothermal fields, subglacial events, climate pulses, new islands, mud
  vents).

## Functions

- `hexworld.pipeline_v51.run_volcanism(config, project)` returns a dict and does not
  attach anything.
- `run_v51(...)` is the stage wrapper with stage keys and caching.
- `hexworld.stages.validate.validate_volcanism(project)` checks the closed budgets and the
  value ranges.

## Files on disk

| File | Content |
|---|---|
| `volcanism_fields.npz` | The layers |
| `volcanism.json` | Provinces, systems, events, aggregates, summary, stage keys, units |

- The manifest section is `v51`, with `volc_hash` and the content hashes.
- A stored state from another algorithm version or another upstream state is NOT loaded.
  A warning says so.

## CLI and GUI

- **CLI.**

  ```bash
  hexworld volcanism PROJECT --events N
  ```

  `hexworld generate --through v51` stops before the biomes.
- **Map.** The tab «Извержения (v5.1)» has 10 modes and a systems layer coloured by
  activity phase.
- **Hex info.** The section «Вулканизм (последние тысячелетия)».
