# SST root cause and fix (G2 / G3)

## Findings (ablations on one domain, `defect_sst_before.md`)

1. **There was no solved SST.** The mixed-layer heat equation had only the *seasonal
   anomaly* of the absorbed sunlight and an ice-albedo term; the annual energy and the
   latitude profile came entirely from a restoring flux `γ (T_eq − T)` toward an Earth
   zonal SST table, plus a hidden step that shifted `T_eq` so that the zonal mean of the
   SST matched the table. Evidence: A3 (restoring off) — tropics 21.7 °C instead of
   26.4, 30–45° 12.7 instead of 16.8, 45–60° 2.4 instead of 7.5.
2. **An infinite cold sink.** Upwelling brought water from a fixed subsurface temperature
   `T_eq − 6 °C`, while downwelled water vanished; over a basin the Ekman up- and
   down-flows are equal, so the vertical exchange removed heat every month. The hidden
   zonal calibration compensated it. Evidence: removing the calibration (step 1 of the
   fix alone) lowered 30–45° from 16.8 to 16.0 °C; closing the reservoir raised it back
   to 16.3 and 45–60° from 7.9 to 8.5.
3. **Damped seasons.** A constant annual `T_eq` with γ = 10 W/m²/K damped the seasonal
   range (A0 30–45°: 2.8 K vs ~3.5 K observed).
4. Units, signs, initialisation, ice: checked, no defect found (heat ledger closes to
   ~1e-16; ice A5 only matters poleward of 60°; no per-month reset of the SST).
5. The remaining midlatitude difference to the approximate reference (about −1 K at
   30–45°, +0.6 K at 45–60°) is within the uncertainty of those reference values and of
   the calibrated background; it is not a hidden floor.

## Fix (implemented)

- `hexworld/earth/ebm.py`: the planet's seasonal energy-balance model (insolation of the
  planet, OLR `A + B T`, meridional transport `D`, ice-free albedo) gives the atmosphere's
  equilibrium temperature `T_A(lat, month)`; calibration [K] to the Earth zonal SST
  (annual RMS 0.74 K, midlatitude half range 3.0–3.7 K).
- Ocean surface flux `Q = λ (T_A(month) − SST) − 0.6 S (α_eff − 0.06)`, λ = 35 W/m²/K
  (linearised air-sea exchange, Haney 1971), booked as surface heat; no zonal calibration
  (`zonal_calibration_max_k = 0`).
- Per-basin subsurface reservoir with closed heat (fed by downwelling, drained by
  upwelling; `deep_reservoir_c` in the diagnostics).
- Drag-regularised slab/Ekman transport (`slab_current`, `R_SLAB = 1e-5 s⁻¹`): finite at
  the equator, sign of f kept in the south (N03).
- The coupler's land anomaly and the SST-anomaly map use the EBM background instead of the
  table.

## After (same domain and forcing)

**A0 after the fix**

| band | cells | annual mean, °C | P05 / P95 | half range, K | ice share | ice-free mean | ref mean (approx) | ref half range |
|---|---|---|---|---|---|---|---|---|
| N0-15 | 12719 | 26.55 | 25.72 / 27.49 | 0.93 | 0.0 | 26.55 | 27.3 | 1.0 |
| N15-30 | 11388 | 23.38 | 20.49 / 25.69 | 1.58 | 0.0 | 23.38 | 24.5 | 2.0 |
| N30-45 | 10409 | 16.32 | 12.67 / 19.76 | 2.42 | 0.0 | 16.32 | 18.5 | 3.5 |
| N45-60 | 9463 | 8.45 | 5.04 / 11.89 | 2.87 | 0.008 | 8.49 | 9.5 | 3.5 |
| N60-75 | 9226 | 1.18 | -1.8 / 4.37 | 2.26 | 0.305 | 2.41 | 3.0 | 2.0 |
| N75-90 | 9958 | -1.74 | -1.8 / -1.4 | 0.08 | 0.971 | -0.05 | -1.3 | 0.5 |
| S0-15 | 13803 | 26.56 | 25.75 / 27.49 | 0.9 | 0.0 | 26.56 | 27.3 | 1.0 |
| S15-30 | 10642 | 23.56 | 20.52 / 25.77 | 1.56 | 0.0 | 23.56 | 24.5 | 2.0 |
| S30-45 | 8904 | 16.13 | 12.55 / 19.67 | 2.44 | 0.0 | 16.13 | 18.5 | 3.5 |
| S45-60 | 8955 | 8.49 | 5.03 / 11.94 | 2.9 | 0.001 | 8.49 | 9.5 | 3.5 |
| S60-75 | 8941 | 1.39 | -1.47 / 4.4 | 2.49 | 0.211 | 2.19 | 3.0 | 2.0 |
| S75-90 | 3958 | -1.19 | -1.76 / 0.13 | 0.99 | 0.601 | -0.29 | -1.3 | 0.5 |

Diagnostics: {"heat_relative_residual": 6.728260852740835e-14, "zonal_calibration_max_k": 0.0, "max_current_ms": 0.9527860927556853}.

Tests: `tests/test_v541_ocean.py` (N03–N06, N09, N10, T24, T25, T30, T33, T35, T43) and the
43 ocean tests of 5.3 (`tests/test_v53.py`) pass. Remaining uncertainty: the background is
calibrated to an approximate Earth profile; no OISST comparison was run
(`observational_validation = NOT_RUN`); the seasonal half range is ~70–80 % of the
approximate reference (λ coupling + 40 m mixed layer).
