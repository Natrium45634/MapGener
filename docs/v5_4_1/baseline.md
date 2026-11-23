# TZ 5.4.1 — G0 baseline

- Repository state before 5.4.1: branch `claude/laughing-goldberg-gu0z9w`, commit
  `e5d7cfe` (HexWorld 1.3.0, generator 5.6.0), clean tree.
- Environment: Python 3.13.18, NumPy 2.5.4, SciPy 1.18.1, PySide6 6.12.0,
  Linux x86_64 (cloud container), 4 vCPU, 16 GB RAM. Tests run with `pytest -n 2/3`
  (xdist); exact-hash comparisons are single-process.
- Test state at `e5d7cfe`: 646 passed, 1 failed (settings-schema coverage, fixed in
  `e5d7cfe` itself, 19/19 after the fix); slow / GUI offscreen tests included; no real
  display, no Windows (Windows GUI: NOT VERIFIED).
- Benchmarks at that state (Terra7, default config): 360×240 × 40 km 221 s / 2.4 GB,
  500×300 × 30 km 294 s / 2.3 GB, 1000×800 × 25 km 1113 s / 5.8 GB
  (`docs/v5_4_benchmarks_and_coverage.md`).
- SST baseline (before any 5.4.1 change): `defect_sst_before.md` (tables by latitude band
  and hemisphere, ablations A0/A3/A3b/A5/A8, SST hash), script `sst_diag.py`.
- Real producers (`audit.csv`): one SST producer (`earth/ocean.py::run_ocean`, called by
  the coupler inside the v4 world climate and once more by the 5.4 coupled iteration —
  the second result replaces the first, never a second field), climate normals
  (`atmosphere/normals.py::build`), PET (`earth/pet.py` FAO-56 in v4.5), weather year
  (`atmosphere/weather.py::generate`), sea ice (`run_ocean`), land ice
  (`earth/icesheet.py::spinup` in v5.3), biomes (`pipeline_v50.py`), saves (`io.py`).

The original report gave no seed for the cold-SST observation; the diagnostic world is
`Terra7` 160×110 × 60 km (see `defect_sst_before.md`). Heavier arrays are not committed;
hashes and tables are.
