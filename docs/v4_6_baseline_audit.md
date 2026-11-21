# v4.6 baseline audit (TZ 4.6, 0.1)

Audited on the current branch (generator 4.5.0 → 4.6.0) before the changes; the test suite
of 4.5 passed 193 + 4 slow on Linux and in CI on Windows 3.12 / 3.13. No `CLAUDE.md` exists
in the repository.

## How the surface was made before 4.6

`elevation_m` of the v3 epoch was one aggregate:

```
v2 macro elevation at the rock's t = 0 position (base_component)
+ v3 history: uplift - erosion + sediment + volcanic material
+ v3 instantaneous trench / outer rise (trench_relief)
+ mountain relief patch (sampler) + volcanic cones (hex-footprint means)
```

On the sea floor the v2 part was a flat `ocean_floor_m` (-4430 m) reached over 230 km of
continentality, ±340 m noise, a 1500 m Gaussian ridge swell along divergent boundaries and
a -2800 m Gaussian trench. Only the crust opened *during* the 30 Ma history (gaps) followed
an age-depth curve. Hence what the sea floor showed:

| Symptom on the map | Cause found |
|---|---|
| parallel "terraces" around ridges | two depth models side by side: age-depth in the gaps, a Gaussian swell on the older crust |
| one dark tone over most of the ocean | old crust at the flat `ocean_floor_m`, independent of age |
| black dashes in odd places | the v2 trench travels with the plate (a fossil) and v3 adds a current one |
| the same shelf / slope ring around every coast | one margin profile in continentality for every margin |
| no fracture zones, no abyssal fabric | the crust age was the isotropic distance to the nearest ridge / 35 |
| `land = elevation > sea level` | every low area, also closed basins far inland, became sea |

## Fields

| field | meaning | units | source stage | consumers | mutable? | tested? |
|---|---|---|---|---|---|---|
| `macro_elevation_m` (v2) | baseline macro relief | m | v2 | v3 history (material frame) | no | test_v2 |
| `elevation_m` (v3) | epoch surface | m | v3 | v4.6, views | no (read only from 4.6) | test_v3 |
| `base_component_m` (v3, new) | v2 part of the v3 surface | m | v3 | v4.6 composition | no | test_v46 (integration) |
| `trench_relief_m` (v3, new) | v3 instantaneous trench | m | v3 | v4.6 (removed on the floor) | no | idem |
| `material_x/y_km` (v3, new) | t = 0 position of the rock (plate frame) | km | v3 | v4.6 age, fabric, sags | no | idem |
| `continentality_km` (v3, new) | signed distance to the crust edge at t = 0 | km | v3 | v4.6 margins | no | idem |
| `trench_distance_km`, `trench_side`, `trench_rate_mm_yr` (v3, new) | subduction geometry of the epoch | km, enum, mm/yr | v3 | v4.6 trenches, margins | no | idem |
| `spreading_axis_distance_km` (v3, new) | distance to the spreading cells | km | v3 | v4.6 axial morphology | no | idem |
| `crust_age_ma` (v3) | age of the crust | Ma | v3 (now from ridge pieces) | v3 subduction priority, v4.6 | no | test_v46 |
| `continental_fraction` (v3) | continental crust share | 0..1 | v3 | v4.6, v4 shelf | no | test_v3 |
| `sediment_thickness_m` (v3) | history sediment | m | v3 | v4.6 (+ basin fill, pelagic) | no | test_v3 |
| `mountain_system_id`, `mountain_class`, plan | mountain structure | id, enum, JSON | v3 | v4.6 massifs, foredeep load | no | test_mountains |
| `orogenic_relief_m` (v3 world raster) | mountain load of the epoch | m | v3 | v4.6 flexure | no | test_v46 |
| `surface_elevation_m` (v4.6) | composed solid surface | m | v4.6 | v4 (input), v4.5 via v4 | yes (stage output) | test_v46 |
| `ocean_mask` … `water_depth_m` (v4, new) | water classification | bool, m | v4 + v4.6 | views, v4.5, coasts | recomputed with the sea level | test_v46 |
| `flow_receiver`, rivers, lakes (v4) | drainage | – | v4 | v4.6 massif valley graphs, basin water | no | test_v4 |
| `lake_level_m` (v4) | lake surface from the water balance | m | v4 | v4.6 water layers | no | test_v4 |

## Consumers that had to change

| consumer | used | now |
|---|---|---|
| v3 `derive_land` | `elevation > sea` | ocean = flood from world-ocean sources (`surface_water`) |
| v4 `evolve` (each step) | `z <= sea` | the same flood each step |
| v4 `hex_climate` | `height <= 0` is sea | the ocean mask; dry land below the sea keeps its negative height (warmer) |
| v4 hydrology | sea = `z <= sea` | sea = ocean mask; closed low ground is land with lakes / playas |
| coasts | `z > sea` | `~ocean_mask` |
| validators | `land == z > sea` | no sea hex above the sea; masks exclusive and complete |
| views | depth tint where `z < sea` | ocean tint only on the ocean; depressions dark green; lakes lake-blue |

## What was not available

* No real DEM / GEBCO cut is shipped with the repository and none was downloaded (TZ 10.3
  asks not to load 7 GB for a test). Statistics are compared with published ranges, not
  with real regional cuts.
* No `CLAUDE.md`.
* Windows GUI on a physical screen: NOT VERIFIED (only CI on Windows and a real Qt window on
  Xvfb under Linux).
