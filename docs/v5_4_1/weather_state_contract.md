# Weather year → water and ice state (G5)

`atmosphere.weather_year = true` runs, after the climate normals:

1. **Spin-up on the normals** (3 years of the monthly bucket from field capacity) — the
   start state; one more normal year from it is the reference for the anomalies.
2. **Weather years 1 … `weather_year_index`** (at most the last 5) in sequence from that
   state: each generated with its own seed namespace
   (`atmosphere/5.4/weather/<year_seed>/<year>/…`), the state carries over (no January
   reset). Monthly actual rain on the hexes = hex normal × (actual / normal) of the
   atmosphere grid; temperature = normal + the year's anomaly field (1 K, ~1500 km, warmer
   where the warm season is dry); PET × (1 + 0.04 ΔT).
3. **Stores per land hex** (`hexworld/atmosphere/hydrostate.py`): root-zone soil (stores at
   wilting point / field capacity / saturation from the v4.5 soil layers), seasonal snow,
   groundwater, firn and glacier ice (from the v5.3 sheet thickness). Fluxes: Gaussian
   snow share, expected PDD, snow-first degree-day melt, infiltration limited by capacity
   and frozen soil, FAO-56 water stress, drainage → groundwater (internal), baseflow,
   runoff. Ledger per hex `d(S+SWE+G+firn+ice) = P − AET − runoff`, residual reported
   (`water_state.ledger_max_abs_residual_mm`).
4. **Rivers**: monthly runoff volumes routed down the existing `flow_receiver` DAG
   (sparse triangular solve; no second drainage network); `weather_discharge_ratio` =
   annual mean discharge actual / normal. The flood day of the year is routed as a one-day
   volume (`weather_flood_peak_m3_s`).
5. **Glaciers**: SMB = accumulation − ablation on glacier hexes; surviving snow becomes
   firn, firn becomes ice with a 5-year time scale; ice mass changes, the geometry does not
   (`geometry_pending`: the v5.3 flow solver acts on its own slower step).

Outputs (hex layers): `weather_soil_moisture_end_mm`, `…_anomaly_mm`, `weather_aet_mm`,
`…_anomaly_mm`, `weather_runoff_mm`, `weather_runoff_anomaly_pct`, `weather_recharge_mm`,
`weather_snow_max_swe_mm`, `weather_water_stress_change`, `weather_glacier_smb_mm_we`,
`weather_glacier_runoff_mm`, `weather_discharge_ratio`, and the carry-over state
`state_soil_mm`, `state_swe_mm`, `state_groundwater_mm`, `state_firn_mm_we`,
`state_glacier_ice_mm_we`. The normals and the potential biomes are never written.

Limits: monthly aggregated mode (no flood peaks inside a month); closed lakes are
integrated only in the formula / fixture (`lake_balance`), not yet per lake in the stage;
the atmosphere is prescribed for the year (one-way: AET does not feed back into the
year's rain); no salinity of lakes.
