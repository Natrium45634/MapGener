# TZ 5.5 — performance and resolution (B80, B81)

Machine: Linux x86-64 container, 4 CPU, 15 GiB RAM, Python 3.13.18, NumPy 2.5.4,
SciPy 1.18.1, single-threaded NumPy order. Seed `Bench55`, the same physical world width
(36 000 km) for every grid, `tools/bench_v55.py`. Each run is a fresh process that loads
the saved world and runs the biosphere stage alone (first run cold, the others warm).
Peak traced = `tracemalloc` peak of the stage; added RSS = max RSS after − before the stage.

| grid | hexes | wall time, s (cold / warm / warm) | peak traced, MiB | added RSS, MiB | ledger |
|---|---|---|---|---|---|
| 360×240 | 86 400 | 13.4 / 13.7 / 13.3 | 338 | 311 | closes |
| 500×300 | 150 000 | 24.9 / 26.0 / 27.5 | 595 | 580 | closes |
| 1000×800 | 800 000 | 139.5 / 131.9 / 127.0 | 2 936 | ~3 070 | closes |

Per stage at 1000×800 (warm, s): inputs 2.0, production 70.8, graphs 9.2, suitability 36.0,
food web 4.7, metrics 4.2.

Targets of §16 [К]: ≤120 s at 500×300 — **met** (≈26 s); ≤300 s at 1000×800 — **met**
(≈130 s); added peak ≤2 GiB at 800 k hexes — **NOT met** (≈2.9 GiB). Before the memory
work of this release the stage was killed by the OOM guard at 1000×800 (8.7 GB RSS); the
group arrays are now kept only on the hexes of their realm, the food allocation keeps no
per-pool copies and the per-template weights are accumulated on the fly. The remaining
peak is the land production (23 groups × 3 microsite bins × 12 months on 440 k land
hexes) and the template tables (uint8, 387 templates); a further reduction needs a tiled
production loop (not done).

## Resolution comparison (B80) — PARTIAL

| grid | land, M km² | inland water, M km² | land NPP, PgC/yr | plant C, PgC | median land richness | bioregions |
|---|---|---|---|---|---|---|
| 360×240 | 202.1 | 0.64 | 24.2 | 304 | 39 | 145 |
| 500×300 | 181.2 | 0.58 | 54.0 | 763 | 48 | 532 |
| 1000×800 | 183.8 | 2.73 | 69.4 | 1003 | 56 | 487 |

These are **not** a discretization test: the world generator itself produces different
geography and climate at the three grids (land area differs by 10 %, the climate of the
coarse grid is colder and drier), so the biosphere receives different physical inputs. A
same-input remap test (one set of physical fields resampled to the three grids) was not
built; B80 stays PARTIAL. The cost scale of the dispersal graph is checked separately on
a physical route (B31, PASS).
