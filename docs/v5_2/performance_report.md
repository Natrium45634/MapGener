# TZ 5.2 — performance (I15)

One full `generate` per size (seed `Terra7`, all stages on), Linux x86_64, 4 vCPU / 16 GB,
Python 3.13.18, NumPy 2.5.4, SciPy 1.18.1. Script: `scratchpad/s52/bench52.py W H KM`.
«v5.2» is the stage alone; «total» is the whole pipeline (v1 … v5.2). Query = one
`query_deposits_at_world_xy` (ground truth) averaged over up to 200 bodies.

| Window × km | Hexes | Total, s | v5.2, s | of which soils, s | deposits, s | Systems / bodies | Soil columns | Query, ms | v5.2 layers, MB | Peak RSS, MB |
|---|---|---|---|---|---|---|---|---|---|---|
| 100×100 × 40 | 10 000 | 207 | 0.43 | 0.38 | 0.03 | 7 / 26 | 1 546 | 0.06 | 2.9 | 3 661 |
| 360×240 × 40 | 86 400 | 217 | 3.95 | 3.43 | 0.25 | 127 / 293 | 18 693 | 0.13 | 25.2 | 3 299 |
| 500×300 × 30 | 150 000 | 272 | 6.94 | 6.23 | 0.38 | 150 / 282 | 33 184 | 0.12 | 43.8 | 3 070 |
| 1000×800 × 25 | 800 000 | 923 | 45.1 | 40.3 | 2.35 | 537 / 1 376 | 250 578 | 1.29 | 233.6 | 6 782 |

- The stage costs ~5 % of the whole generation; the soils (two passes of the static
  chemistry over the soil columns, batched 5 × 5 solves) dominate it. The deposits are a
  few seconds even at 1000×800 (no per-hex objects, no voxels).
- The layers grow linearly (float32; 3 soil layers and 9 commodity families stored as
  stacks). Peak memory is that of the whole pipeline; the catalogue is a few MB of JSON.
- The 500×300 run of this table reported `validation_ok = false`: two tract provinces of
  the same type whose centroids rounded to one 100 km cell got the same province id, so
  their systems shared a deposit id. Fixed (`deposits._unique_pid`, test
  `test_province_ids_stay_unique_when_centroids_share_a_cell`); a rerun of that world
  validates.
