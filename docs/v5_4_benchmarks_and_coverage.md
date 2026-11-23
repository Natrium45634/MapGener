# TZ 5.4 — benchmarks and coverage

Machine: the cloud container of this work (4 vCPU, 16 GB RAM, Linux, Python 3.13, NumPy /
SciPy reference CPU code, no Numba). Seed `Terra7`, default settings; "on" = atmosphere
with storms and a weather year, "off" = `atmosphere.enabled = false`. Times include the
whole generation (v1–v5.3 stages too). Peak RSS = `ru_maxrss` of the process.

| Window (hexes × km) | Atmosphere | Total, s | Peak RSS, MB | v4 climate, s | of which 5.4 normals + coupled ocean, s | v5.4 hex stage, s |
|---|---|---|---|---|---|---|
| 360×240 × 40 | off | 168 | 2048 | 48 | — | — |
| 360×240 × 40 | on | 221 | 2448 | 76 | 25 | 15 |
| 500×300 × 30 | off | 250 | 2012 | 45 | — | — |
| 500×300 × 30 | on | 294 | 2278 | 76 | 23 | 15 |
| 1000×800 × 25 | on | 1113 | 5811 | 90 | 25 | 23 |

The atmosphere grid does not grow with the hex window: it is the world raster coarsened
to ~26k cells (factor 3, ~272 km cells for these worlds), so the 5.4 cost is roughly
constant (~25 s of normals and coupled ocean, ~15 s for the hex stage with a weather
year; ~5 s without one). The hex stage samples the grid; its memory is a few 2-D float32
layers per hex (no monthly hex arrays are stored by 5.4). At 1000×800 the vapour budget
still closes to 4e-17 (worst month); the 5.8 GB peak is the whole pipeline (v4.5 monthly
fields, v4.7 units), not the atmosphere. The weather year keeps at most
150 explicit events of a type per month (`MAX_EVENTS_MONTH`); the rest of the climatology
stays in the normals.

The numbers above were measured with the code of commits `6479804`–`ce3dedf` (the
1000×800 run had no "off" twin: the rest of the pipeline dominates it); later changes
(boundary booking, summer heat lows, event-first weather year) do not change the cost
order (one extra right-hand-side term, a few array operations).

## Coverage

- Unit tests (`tests/test_v54_atmosphere.py`, 39 cases, ~2 s): finite volumes on 2-cell,
  raster and hex meshes; circulation; moisture budget; ITCZ; monsoon diagnostics; latent
  feedback; numeric controls.
- Integration tests (`tests/test_v54_integration.py`, 34 cases, ~2 min): the stage on a
  generated 100×100 world, save / load, views, windAt for ash, storms, weather year,
  snow, windthrow, export, boundary storms.
- Scenario fixtures `tests/fixtures/v5_4/S01–S24.json` each name the executable test.
- Full suite (final run, 3 workers, ~25 min): 646 passed, 1 failed — the settings schema
  missed `atmosphere.latent_feedback`; added, `test_world_settings.py` 19/19 passed after
  the fix.

Not measured: a real display, Windows (Windows GUI: NOT VERIFIED), GPU / Numba kernels
(not used).
