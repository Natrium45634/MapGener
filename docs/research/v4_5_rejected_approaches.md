# Approaches considered and rejected for v4.5

| Approach | Why it was rejected | What was done instead |
|---|---|---|
| `annual_disaggregation` (synthetic profiles: winter-wet, summer-wet, bimodal) | v4 already computes four seasons of moisture transport; a synthetic profile would compete with the physics | months from the seasonal physics (`derived_from_seasonal_physics`) |
| P_m = P_ann / 12, T_m = T_ann | explicitly forbidden by the TZ; it gives false frost, GDD and dryness | harmonic temperature, season-conserving precipitation |
| FAO-56 Penman-Monteith | needs R_n, G, u₂ and vapour pressures that the model does not compute; invented values would be worse than an honest simple model (TZ 4.3) | v4 temperature-index PET per season, spread by FAO-56 day length; labelled a heuristic |
| Hargreaves | needs T_max − T_min (diurnal range), which the model has no source for | as above |
| pyet / xclim / pyrealm as runtime dependencies | pandas / xarray stacks for a handful of formulas; xclim's indices are daily | the formulas re-derived and tested (FAO-56, frost number, monthly proxies) |
| integrating dD/dt of the soil over millions of years | the TZ warns against long integration with unreliable constants | a static depth potential f(slope, weathering, material), flagged approximate |
| TWI from the number of hexes | not dimensionally correct, and depends on resolution | physical upstream area / hex edge length, plus an anomaly against a reference hex |
| HAND to the nearest river in a straight line | a ridge next to a river in another basin would become a floodplain | HAND along the D6 flow path (pointer jumping) |
| floodplain = every river hex | at 75 km a hex is not a floodplain; it holds a strip of floodplain | the share of the hex a floodplain of the local discharge can cover |
| waterlogging = soil ≥ 95 % of AWC at month end | percolation always pulls month-end storage below that, so nothing was ever waterlogged | months with saturation excess while the supply exceeds PET |
| glacier discharge as runoff (as in v4) | v4.5 has no ice flow; inventing it would be a hidden water source for the rivers of the months | `persistent_snow_accumulation_mm_yr` and a quality flag; the snow stays stored |
| recomputing river discharge from the v4.5 runoff | v4 rivers are the river state, and a change of method must be a new snapshot (TZ 6.4) | `river_q_source` = v4; the difference is reported (−0.5 % on Terra7) |
| vegetation feedback on AET | there are no biomes yet; it would be circular (TZ 2.7) | a reference low cover |
| exact frost days from monthly means | not knowable from 12 means | `frost_days_exact` not calculated; FDD / TDD and the frost number are proxies |
| sub-hex coastline (`land_fraction` < 1) | the model has no sub-hex coastline yet | land hexes count as wholly land; documented |
