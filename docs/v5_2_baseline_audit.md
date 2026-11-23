# TZ 5.2 — baseline audit (G0)

Date: 23.11.2026. Code: branch `claude/laughing-goldberg-gu0z9w` at `47f29dd` (HexWorld 1.4.0,
generator 5.7.0). This audit reads the code; it does not claim that a TZ equals a working module.

Baseline tests at the audit commit: the full suite (`pytest tests -n 3`) and the slow T20 tests
were running at audit time; their totals are recorded in `docs/v5_4_1/acceptance_report.md`.
Earlier recorded run of the same tree (5.4.1d): all quick tests passed on Linux. Windows GUI:
NOT VERIFIED (no Windows display here).

## Pipeline that actually exists

`v1 plates/crust → v2 continents/macro relief → v3 history (30 Ma default) → v4.6 terrain
(basins, sea floor) → v4 climate / hydrology / erosion (with the v5.3 earth coupler and the
v5.4 atmosphere inside) → v4.5 environment (soils, monthly water) → v5.3 earth → v5.4
atmosphere → v4.7 topoclimate → v5.1 volcanism → v5.0 biomes`. `generate(config,
through="v50")` runs all of them. TZ 5.2 adds a stage **v52 resources** after v5.0 (it reads
the v5.0 cover; it does not change any earlier layer).

## field → producer → units → time → dims → quality → save/load → consumers

| Field | Producer | Units | Time / epoch | Dims | Quality | Saved | Used by 5.2 for |
|---|---|---|---|---|---|---|---|
| `plate_id`, `crust_type`, `boundary_type` | v1 | id / enum | present | (H, W) | derived | `layers.npz` | tectonic setting |
| `crust_age_ma` | v3 (material-frame baseline + epoch) | Ma | selected epoch | (H, W) | derived; cratons 550–2450 Ma are *inherited* (baseline noise), not simulated | `history_layers.npz` | old crust (BIF, kimberlite, unconformity U), parent age |
| `rock_type` | v3 | enum `ROCK_TYPES` (oceanic basalt, craton crystalline, platform sedimentary, orogenic metamorphic, volcanic, unconsolidated sediment) | epoch | (H, W) | 6 classes only; no granite vs gneiss, no carbonate, no ultramafic | yes | lithology adapter (see below) |
| `continental_fraction`, `new_crust` | v3 | 0..1 / bool | epoch | (H, W) | derived | yes | oceanic vs continental crust |
| `orogen_age_ma`, `mountain_system_id`, `mountain_class` | v3 | Ma / id / enum | epoch | (H, W) | derived | yes | collision / arc orogens (orogenic Au, pegmatites, Sn–W, ophiolites) |
| `uplift_total_m`, `erosion_total_m` | v3 | m over the history | 0–30 Ma | (H, W) | derived | yes | exhumation of ore bodies (erosion, burial) |
| `sediment_thickness_m` | v3 → v4.6 → v4 | m | present | (H, W) | derived | yes | basin fill, cover thickness, aquifer |
| `volcanic_age_ma`, `volcano_id`, `volcanic_potential` | v3 | Ma / id / 0..1 | epoch | (H, W) | diagnostic (potential is *not* a probability) | yes | young volcanic surfaces |
| `trench_distance_km`, `trench_side`, `spreading_axis_distance_km` | v3 | km / sign | epoch | (H, W) | derived | yes | arc vs back-arc vs ridge setting |
| `basin_id`, `basin_subsidence_m`, `basin_fill_m`, plans (`basin_type`, `center_km`, axis, subsidence) | v4.6 | id / m | present (fill assumed) | (H, W) + list | fill `[А]` (assumed from denudation) | `terrain_*` | sedimentary basins (coal, SEDEX, sediment Cu, sandstone U, evaporites, oil/gas P1) |
| `bedrock_elevation_m`, `marine_sediment_thickness_m`, `oceanic_crust_age_ma`, `seamount_type`, `marine_province`, `ridge_piece_id` | v4.6 | m / Ma / enum | present | (H, W) | derived | yes | sea-floor resources (nodules, crusts, SMS) |
| `elevation_m`, `ocean_mask`, `lake_id`, `endorheic_mask`, `flow_receiver`, `discharge_m3_s`, `v4_erosion_m`, `v4_deposition_m`, `sediment_flux_m3_yr`, `river_mask`, `river_mouth` | v4 | m, m³/s, m³/yr | present (erosion over `erosion.duration_ma`) | (H, W) | derived; `land_below_sea_level != ocean` kept | `hydro_fields.npz` | placers, nutrient transport, evaporite basins |
| `parent_material_class`, `sand/silt/clay_frac`, `coarse_fragment_frac`, `soil_depth_m`, `soil_organic_potential`, FC/PWP/porosity, `awc_mm`, `ks_mm_h`, `drainage_class`, `soil_salinity_proxy` (0/1), `waterlogging_months`, `floodplain_potential`, `coastal_marine_influence` | v4.5 | as specified in `ENV_LAYER_SPECS` | static potential | (H, W) | approximate (no soil age, no chemistry) | `env_fields.npz` | B0 baseline profile |
| 12-month T, P, PET, AET, snow SWE, soil storage, runoff, recharge | v4.5 | °C, mm/month | climatological year | (12, H, W) | approximate | yes | decomposition modifiers, leaching, salt |
| `upwelling_m_day`, SST, sea ice | v5.3 | m/day, °C | monthly normals | (12, H, W) | derived | yes | phosphorites, marine productivity |
| volcanic provinces (`source_type`: subduction_arc, oceanic/continental hotspot, mid_ocean_ridge, divergent_rift, sedimentary_fluid_vent), systems (magma SiO₂, `formed_ma`, environment), `tephra_thickness_m`, `tephra_load_kg_m2`, `lava_youngest_years_bp`, `hydrothermal_kind`, `geothermal_heat_mw`, `volcanic_ash_stress` | v5.1 | Ma, years BP, kg/m², MW | event window + geological ages | lists + (H, W) | derived; magma profile per province | `volcanism_*` | magmatic-hydrothermal ore systems, ash chemistry of soils, geothermal (P1) |
| `vegetation_fraction`, `productivity_index`, `pft_cover`, `dominant_biome`, `succession_stage` | v5.0 | 0..1 index, enum | equilibrium + disturbance | (H, W) | productivity is an **index**, not NPP | `biome_*` | litter input proxy (provenance `proxy_from_v5_index`) |
| `lgm_ice_mask`, `permanent_ice_mask`, `glacier_fraction` | v4 / v4.5 / v5.0 | bool / 0..1 | LGM / present | (H, W) | derived | yes | soil age after glacial scouring, NoData under ice |

## What is missing (and how 5.2 handles it)

| Missing input | Effect | 5.2 decision |
|---|---|---|
| Soil age, soil chemistry (C/N/P/K, pH, CEC, salts) | no fertility model existed | new B0–B4 model; soil age from surface residence `D / E`, LGM, lava/ash age; quality `approximate` |
| Real lithology beyond 6 rock classes (granite, carbonate, ultramafic, red beds, black shale) | ore models need host / source rocks | **lithology adapter** `resources.geocontext`: inferred sub-classes from the 6 classes + setting (arc, collision suture, rift, craton age, basin type) + smooth world-fixed noise; provenance `inferred_lithology`, never stored over `rock_type` |
| Paleoclimate of past epochs | coal, ancient evaporites, laterites of old epochs | old basins get `inherited_stratigraphy` with an explicit paleo-setting draw; only *late* (Quaternary) processes (laterites, brines, placers) use the present climate |
| World-frame rasters of crust age / craton / rock | strict resolution independence of tract fields | systems anchored to discrete world objects (volcano systems, provinces, basin plans, mountain systems) are resolution-independent; tract fields come from a fixed world lattice (25 km) sampled from the hexes — counts and anchors can differ between resolutions by sampling error (declared, measured in the R20 test) |
| v4.7 sub-hex terrain units for soils | no sub-hex soil profiles | soils per hex (aggregate), flagged; TU aggregation (I11) tested on synthetic fragments only |
| Stratigraphy with dated horizons | sedimentary deposits need horizons | parametric horizons inside each basin (depth from basin fill, age from basin plan) — sparse, per basin |

## Must stay unchanged

Existing geology and snapshots, seeds `[A-Za-z0-9]+` (case-sensitive), separate RNG streams
(`resources/...` namespace), one world and sea level, `land_below_sea_level != ocean`,
`soil_depth != sediment_thickness`, biome ≠ soil, ash ≠ fertility, geological source ≠ deposit.
5.2 writes only new layers and catalogues; tests assert that `rock_type`, `elevation_m`,
`soil_depth_m` and the biome layers are byte-identical with the stage on and off.
