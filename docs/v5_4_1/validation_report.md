# Validation report (5.4.1)

Levels (TZ §15): (1) analytic / unit tests and conservation — done (`tests/test_v541_*.py`,
N01–N20 except N07, N08 partial); (2) synthetic physical geography with negative controls
— done for the boundary, ocean, hydrology and marine fixtures; (3) Earth-like
observational benchmark — **NOT_RUN** (no ERA5 / OISST / WOA23 / GLORYS12 data in this
environment; the model is not called observationally calibrated).

Measured on generated worlds:

- SST by latitude band before / after (`defect_sst_before.md`, `defect_sst_root_cause.md`,
  Terra7 160×110 × 60 km): after the fix the zonal means follow the planet's energy
  balance (calibrated to an approximate Earth profile, annual RMS 0.74 K for the
  background), no hidden zonal calibration, heat ledger ~1e-16.
- v5_4::T20 on real windows (`test_v541_resolution.py`, one physical world, BOUNDS fixed):
  360×240 × 50 km vs 500×300 × 36 km — worst over months 1/7 and levels sfc/850/250:
  NRMSE 0.015, pattern correlation 0.9999, median angle 0.2°, zonal sign agreement 100 %,
  rain total difference < 10 %; nested windows of the same world: atmosphere arrays on the
  world grid identical; hex-field overlap NRMSE 0.006, median angle 0°.
- Aridity of the 5.4 `full` coupling on 4 seeds (`docs/research/v5_4_parameter_calibration.md`).

Not done: an ensemble of 10–20 independent seeds with confidence intervals for the SST /
monsoon / storm statistics (one diagnostic world for the SST, 4 for aridity, 1 for T20);
listed as an open validation item.
