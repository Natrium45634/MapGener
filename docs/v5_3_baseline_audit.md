# TZ 5.3 — baseline audit (G0)

**Revision audited:** branch `claude/laughing-goldberg-gu0z9w` at 14bba42 (HexWorld 1.1.0,
generator 5.4.0).

**Environment:**

- Linux container with 4 cores.
- Python 3.13, NumPy 2.x, SciPy 1.x, PySide6 (offscreen).

**Test state before 5.3:** 381 tests passed in 21 min (`pytest -n 4`), plus
`test_v51.py` (93 tests).

**CI before 5.3:** green, on Linux and on the Windows installer build.

## Producers of the physical fields (before → after 5.3a)

| Field | Producer | Time scale | Units | Quality before | Saved | GUI | Test |
|---|---|---|---|---|---|---|---|
| `elevation_m` (surface after erosion) | v4 hydro | geological epoch | m | model | hydro_fields | Рельеф | test_v4 |
| `surface_elevation_m` (v4.6 sea floor + land) | v4.6 terrain | epoch | m | model | terrain_fields | Дно океана | test_v46 |
| `sea_level_m` | v3 history | epoch | m | model | hydrology.json | — | test_v3 |
| `ocean_mask` (hex, connected) | v4.6 package A | epoch | bool | connectivity | hydro_fields | Вода | test_v46 |
| world-raster `sea` (climate) | v4 climate | — | bool | **height ≤ 0, not connectivity** → 5.3: connected | not saved | — | test_v53 |
| `lake_surface_level_m` | v4 network | — | m | model | hydro_fields | Вода | test_v4 |
| `sea_surface_temperature_c` | v4 climate → coasts | annual | °C | **latitude only** → 5.3: ocean model | hydro_fields (coast) | Берега | test_v53 |
| `sst_monthly_c` | — → v5.3 `earth` | 12 months | °C | — → CALIBRATED_APPROXIMATION | earth_fields | Океан | test_v53 |
| currents, upwelling | — → v5.3 | 4 seasons | m/s, m/day | — → CALIBRATED_APPROXIMATION | earth_fields | Океан | test_v53 |
| sea ice | v5.0 `sea_ice_potential_months` (T < −2) → v5.3 | 12 months | fraction, m | LEGACY_POTENTIAL_ONLY → energy budget | earth_fields | Морской лёд | test_v53 |
| `T_monthly`, `P_monthly` | v4.5 months | 12 months | °C, mm | model | environment_monthly | Климат | test_environment |
| `RH_monthly` | v4.7 topoclimate (P/PET) | 12 months | fraction | proxy | topoclimate | Топоклимат | test_v47 |
| surface wind | v4 climate | 4 seasons, annual | m/s | belts + monsoon | hydro_fields | Климат | test_v4 |
| winds aloft | — → v5.3 | 4 seasons × 850/500/250 hPa | m/s, m | — → thermal wind | earth_fields | Атмосфера | test_v53 |
| PET | v4 `potential_et(T)` → v4.5 monthly | 12 months | mm | **temperature index** → FAO-56 | environment_monthly | Испаряемость | test_v53 |
| AET, soil water | v4.5 water balance | 12 months | mm | model | environment_monthly | Почвы и вода | test_environment |
| land ice, firn, snow | v5.0 cryosphere | 100-yr spin-up | m, mm SWE | spin-up too short for sheets (G4) | biomes_fields | Снег и ледники | test_v50 |
| soil profile | v4.5 soils | static | texture, m | static | environment_fields | Почвы | test_environment |
| PFT, biome fractions | v5.0 | equilibrium | fractions | potential vegetation | biomes_* | Биомы | test_v50 |
| volcanic events | v5.1 | 10 kyr window | kg, m³ | deposits as layers only (G7) | volcanism_* | Извержения | test_v51 |

## Findings that drive the 5.3 work

1. **SST is latitude-only.** The climate used `SST = max(T_zone, −1.8)`. Coastal warmth
   was added by a heuristic (`MARITIME_WARMING`).
2. **Sea ice was only a heuristic.** It was a smooth function of the zonal temperature that
   made coasts continental, and the v5.0 potential `T < −2 °C`.
3. **The climate treated every low cell as sea.** On the world raster, closed depressions
   below sea level counted as sea and evaporated like ocean.
4. **Single wind level.** The orographic lift and the v5.1 ash used the surface wind and a
   synthetic zonal jet.
5. **PET depended on temperature only.** v4.5 distributed the v4 Thornthwaite-like
   seasonal totals with day length.
6. **Ice sheets never reached equilibrium.** v5.0 spins up for 100 years from a paleo
   seed. This is item ICE-LAND, gate G4 (open).
7. **Volcanic deposits do not change the relief.** v5.1 deposits are layers only. This is
   item VOL-GEO, gate G7 (open).
8. **Earlier fixes stay as regression tests, not re-implemented:**
   - humidity fixed in 1.0.1;
   - no prairie in tundra climates;
   - polar desert at −10 °C;
   - strong-ablation threshold of −300 mm;
   - a lake hex has no land biome.

## Backward compatibility

- **Old saves.** A project saved before 5.3 has no `earth` section in `config.json`. It
  loads with `earth.enabled = False`, its v4 keys do not change, and its climate stays
  bit-identical (tests T01/T02).
- **New worlds.** They have the Earth system on.
