# TZ 5.4.1 — performance (v5_4_1::T80)

Measured on this container: Linux x86_64, 4 vCPU, 16 GB RAM, Python 3.13.18, NumPy 2.5.4,
SciPy 1.18.1. Script: `scratchpad/s54/bench.py 1000 800 25 1` (one full `generate`
through v5.0 with the atmosphere and the weather year on; seed `Terra7`). The run shared the
machine with nothing else (the test suite had finished).

| Window (hexes × km) | Build | Total, s | Peak RSS, MB | v4 climate, s | v5.4 hex stage, s | atmosphere grid |
|---|---|---|---|---|---|---|
| 1000×800 × 25 | 5.4 (before 5.4.1) | 1113 | 5811 | 90 | 23 | factor 3, 272 km |
| 1000×800 × 25 | 5.4.1 (this commit) | 926 | 6410 | 98 | 34 | factor 3, 342 km, 165 × 171 |

Notes:

- the 5.4.1 total is lower although the hex stage grew (weather-year → soil / groundwater /
  snow / glacier state: 3 spin-up years and the actual year, `hydrostate.run_year`, plus
  the sparse routing of monthly runoff): the run-to-run spread of the whole pipeline on a
  shared cloud machine is of the same order; the difference is **not** claimed as a
  speed-up;
- peak memory grew by ~0.6 GB: the monthly land stores of the weather year (12 × N
  float64 temporaries) and the ocean exchange ledgers; the stored layers stay float32;
- the vapour budget closes to 3.2e-17 (worst month); the ocean–atmosphere coupling of
  this window reports `NOT_CONVERGED` after the configured iterations (the same status
  as in 5.4, reported in the summary, not hidden);
- 390 tropical-cyclone geneses per year in this 25 000 × 20 000 km window (a window
  larger than the Earth's tropics band); the weather year keeps the 150-per-month cap.
