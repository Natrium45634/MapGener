# Sources of magma (provinces)

`hexworld.volcanism.provinces.build_provinces(project, params)` builds the provinces.
Every province has a geological cause. None comes from a decorative random field.

## Arc provinces

- **Members.** v3 arc volcanoes grouped by (overriding plate, subducting plate) and cut
  into chains of at most 400 km.
- **Strength.** Taken from the v3 `trench_rate_mm_yr`.
- **Crust.** `crust_type` decides between the `arc_continental` and `arc_oceanic` magma
  profiles.
- **Flux.** `FLUX_REF["subduction_arc"] · strength · sqrt(n)`. The reference is 1000 kg/s.

## Oceanic and continental hotspots

- **Members.** The hotspot and its age-sorted track.
- **Strength.** v3 strength × `hotspot_activity`.
- **Flux.** About 6000 kg/s for an oceanic hotspot (Hawaii ~1e4 kg/s) and 1500 kg/s for a
  continental one.
- **Supereruptions.** Only continental (rhyolitic) hotspot calderas are eligible.

## Mid-ocean ridges

- **Location.** v4.6 ridge pieces, on hexes within 0.75 spacing of the axis and in the
  ocean.
- **Strength.** Half spreading rate / 25 mm/yr.
- **Systems.** One fissure field per ~120 km of axis, with at most 8 per piece and 4
  fissure vents each.
- **Eruptions.** Sheet / pillow flows of 1e6–5e8 m³ (dike-fed).

## Continental rifts

- **Location.** v4.6 `rift_graben` basins, grouped by rift.
- **Field.** A monogenetic field of 6–60 scoria cones along the axis, with alkali basalt.

## Mud vents

- **Location.** Foreland and intracratonic basins where subsidence × fill > 2500 m
  (overpressured sediment).
- **What they are.** Non-magmatic: no lava, no SO₂. They are recorded as flags and
  fields.

## Status of a system

| Status | Meaning |
|---|---|
| `active_system` | Recent geological activity (v3 `active`, ridges, rifts) |
| `dormant_system` | Was active within the last ~0.1 Ma. Its reactivation probability in the window is ≤ 1 |
| `uncertain` | Evidence is weak. Rate = 2 % of active |
| `extinct_system` | No supply and no eruptions. The edifice is kept for the map |

On the reference 500 × 300 world, 50 provinces are found:

- 34 ridge;
- 10 arc;
- 4 rift;
- 1 oceanic hotspot;
- 1 continental hotspot.

They hold 634 systems, 472 of them in the window.
