# v5.1 benchmarks

Timings come from a Linux container with 4 cores, numpy and one process, unless a note
says otherwise.

## Reference world: 500 × 300 hexes of 25 km (150 000 hexes)

The stage ran on its own, `balanced` quality, 160 explicit events. The world has 634
systems and 40 644 events.

| Step | Before tuning | After |
|---|---|---|
| Provinces + systems | 0.1 s | 0.1 s |
| Event schedule (40 644 events) | 0.7 s | 0.7 s |
| Lava (explicit events) | 45 s | 19 s |
| Tephra (FV transport) | 83 s | 21 s |
| PDC / lahars / aggregates | 2.3 s | 1.5 s |
| **v51 total** | **131 s** | **44 s** |

**Tuning steps**

- **Neighbour table of the lava lattice.** The neighbours are now looked up by vectorised
  `searchsorted`; before, a Python dict was built per cell.
- **Lava pulses.** 60 × 20 pulse/relaxation steps instead of 120 × 25.
- **Tephra scatter.** `np.bincount` instead of `np.add.at`.
- **Tephra CFL.** The bound now uses the real summed outflow, not 6 × the maximum face
  speed. This gives 3–5× fewer steps.
- **Tephra window.** 600 km and 24 h for `balanced`.

## Small worlds: 120 × 90 hexes of 50 km (10 800 hexes)

These are the 36 scenarios, run as 3 processes in parallel.

- **Whole generation (v1 … v5.0):** 25–71 s.
- **v5.1 alone:** 2–20 s, and 40 s once (icy world under load).
- **Explicit events:** limited to `24 + hexes/1000` = 34. With a flat 160, v5.1 took 37 s
  of a 47 s generation.

## Quality levels

| `physics_quality` | Lava pulses × relax | Tephra hours / window | v51 on V1 (s) |
|---|---|---|---|
| `fast` | 30 × 12 | 8 h / 350 km | 3.5 |
| `balanced` | 60 × 20 | 24 h / 600 km | 9.5 |
| `high` | 150 × 30 | 72 h / 1500 km | not timed |

## Test suite

- **`tests/test_v51.py`:** 93 tests in 129 s. Most of that time goes to the shared
  `hist_project` fixture and two regenerations.
- **Full suite** (`pytest -n 4`, collected before `test_v51.py` existed): 381 passed in
  21 min.
