# TZ 5.5 — registry schema

Files (package data, checksummed in the stage provenance):

* `biosphere/data/functional_groups_tz55.json` — the 94 groups of the TZ package, byte-identical
  (sha256 `9ddf116c…e34c`): `id, name_ru, realm, trophic_role, example_analogue_ru,
  mobility_mode, habitat_hint, parameter_status`.
* `biosphere/data/traits_v1.json` — built by `tools/build_bio_traits.py` (deterministic).

## Traits (`traits[group_id]`)

| field | unit | groups | meaning |
|---|---|---|---|
| kind | — | all | plant / alga / symbiosis / microbe / fungus / animal |
| level | — | all | trophic level; 1 = producer; a consumer eats only pools fed by lower levels |
| gates | — | all | hard gates (`land_ground, root_zone, canopy, woody_cover, herb_cover, flowers, wet_ground, tidal_coast, inland_water, river, any_fresh, any_water, marine, shelf, deep, deep_shelf, lit_bottom, hard_bottom, soft_bottom, sea_ice, vent, coast, nest_coast, haulout, anadromous, host_plants, hosts, soil, large_land, shallow_water`) |
| niche.{tc, tw, mi, gdd, ph, sal} | °C, °C, P/PET, °C·d, pH, dS/m (land) or psu (water) | all | trapezoids [a, b, c, d] |
| t_act | °C | all | monthly activity (f_T of plants, survival of animals) |
| w_act | AET/PET | plants | monthly water response |
| lue, cue, k, lai_max, sla | gC/MJ APAR, 1, 1, m²/m², m²/kgC | plants | light use, carbon use efficiency, extinction, leaf area |
| alloc, turn | shares, 1/yr | plants | NPP allocation and turnover of leaf / wood / root / repro |
| kn, kp | kgN/m²/yr, kgP/m² | plants | half-saturation of the nutrient limit |
| poll, edible, layer | — | plants | pollination mode, edible share of leaves, canopy layer |
| diet, provides | shares, pool | consumers | food pools and the pool its production feeds |
| mass_kg, d0, assim, npe, pb, phi | kg wet, gC/m²/yr, 1, 1, 1/yr, 1 | consumers | body mass, reference demand, efficiencies, P/B, max share of a pool |
| dormant, migratory, breed_months, needs_water | — | consumers | seasonal strategy |
| mob, L_km, e0 | class, km, 1/yr | all | dispersal class, kernel scale, extinction coefficient |
| split, split2 | (var, n) | all | how the group is divided into templates |

Validation (`registry.validate`, B08): unique ids, a tree without cycles (template → group →
kingdom/level), every diet edge resolved to a pool whose providers are strictly lower,
allocation and diet shares sum to 1, every trapezoid ordered, efficiencies in range.

Result: 94 groups, 387 templates (148 flora, 231 fauna, 8 microbial/fungal), 0 errors.
