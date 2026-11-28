# TZ 5.5 — G0 audit: what the world already gives to a biosphere

Date 28.11.2026. Commit at the start: `5bbfe0c` (HexWorld 1.10.1, generator 5.13.1). Python
3.13.18, NumPy 2.5.4, SciPy 1.18.1, Linux x86-64, 4 CPU, 15 GiB RAM. Before 5.5 no
biosphere / biogeography stage exists: there are biomes (v5.0), an actual-vegetation year
model (5.3 §13), marine habitats (5.4.1 G6) and soil fertility (5.2). 5.5 adds a stage on
top of them and rewrites none of them.

Statuses: IMPLEMENTED (code + test), APPROX (reduced model, tested), PARTIAL, MISSING,
BLOCKED (a dependency is absent), UNVERIFIED (no test / no real Windows display).

## B01 — inherited obligations of 5.4.1

`docs/v5_4_1/acceptance_report.md` ends **PARTIAL**, not FULL_ACCEPTED. Open items that
matter for 5.5:

| 5.4.1 item | status there | effect on 5.5 |
|---|---|---|
| T06/T07 legacy v3–v5.4 genuine fixtures | BLOCKED_MISSING_FIXTURE | B02/B03 can use only the v1/v2 genuine files and current golden saves |
| T08 legacy regeneration | BLOCKED | same |
| T11 cancel / rollback | PARTIAL | 5.5 adds its own transaction test (B77) for the bio stage |
| T36 latent heat of ocean evaporation | PARTIAL | marine NPP uses SST and light only; no heat feedback |
| T44 coast coupling | PARTIAL | none |
| T78 fixed-shape km sweep | PARTIAL | B31/B80 add a physical-scale test for the biological graph |
| Windows GUI | NOT VERIFIED | B78 stays NOT VERIFIED for Windows |

None of these is silently counted as pass in 5.5.

## Baseline (world `Bio55`, 120×80, 60 km, full pipeline incl. 5.5.1)

Generation 106 s. Semantic hashes (sha256 of the stage `content_hashes`, first 16 hex) — the
5.5 stage must leave every one of them unchanged (B04, B07):

| stage | hash |
|---|---|
| geo | acf8245c0b01d35f |
| hist | c3bbca22b9e488f4 |
| terrain | 9f188aa609b422e6 |
| hydro | 174b797f72c97cd5 |
| env (soils, months) | 7ef77d861b81e816 |
| topo (4.7) | d022f320f9b94516 |
| earth (5.3/5.4.1 ocean, ice) | 3df53e6b27d841bb |
| atmosphere (5.4) | fc9eb99772f5d703 |
| biomes (5.0) | 75a105f0257702dc |
| volcanism (5.1) | 7b01679845b66aa2 |
| resources (5.2) | f060b5ccc2cd2fcb |

The test `tests/test_v55_integration.py::test_physics_hashes_unchanged` repeats this check
with and without the biosphere on a fresh world.

## Source map: existing layers that 5.5 reads (input contracts)

The full table with units is `input_contracts.csv`. Summary:

| contract | existing source | quality |
|---|---|---|
| C02 geometry → habitat | `hydro.land_mask`, `ocean_mask`, `inland_water_mask`, `water_depth_m`, `lake_depth_m`, `below_sea_level_land_mask`; `topo.topo_ocean_fraction`, `topo_lake_fraction`; `biomes.glacier_fraction`, `cover_*`; `earth.sea_ice_concentration_monthly` | real, per hex; sub-hex fractions from 4.7 terrain units |
| C03 climate → plants/fauna | `env.monthly.temperature_monthly_c`, `precip`, `pet`, `aet`, `soil_storage`; `topo.monthly.topo_temperature_c`, `topo_radiation_mj_m2_day` (land surface shortwave); `env.gdd5_proxy_c_day`, `coldest/warmest_month_c`; hypsometric quantiles `topo.hyps_q05…q95_m` | real monthly normals; surface shortwave only over land — the ocean gets Angström–Prescott from planet TOA (APPROX) |
| C04 soil → flora | `env.awc_mm`, `soil_depth_m`, `resources.soil_ph`, `soil_nitrate/ammonium_N`, `soil_N_net_mineralisation`, `soil_P_labile`, `soil_salinity_ec_ds_m` | real 5.2 pools (kg/m²) |
| C05 hydro → aquatic | `hydro.discharge_m3_s`, `river_mask`, `river_width/depth_m`, `flow_receiver`, `basin_id`, `lake_id`, `endorheic_mask` | real |
| C06 ocean → marine | `earth.sst_monthly_c`, `sea_surface_salinity_psu`, `current_u/v_ms` (4 seasons), `upwelling_m_day`, `coastal_upwelling`, sea ice; `biomes.marine_kd_per_m`, `marine_euphotic_depth_m`, `marine_nutrient_supply_proxy`, `marine_benthic_zone`, `marine_coral_reef_potential` | real; nutrients only as a proxy (no nutrient pools) |
| C07 volcano → biology | `volcanism.lava_fraction`, `tephra_load_kg_m2`, `volcanic_barren_fraction`, `lava_youngest_years_bp`, `hydrothermal_kind`, `geothermal_heat_mw` | real present-day footprints, no event replay ids per year |
| C08 cryo → ecology | `biomes.glacier_fraction`, `ice_form`, `earth.land_ice_thickness_m`, `hydro.lgm_ice_mask` (retreat since the last maximum) | real |
| C01 geology → bioregion | `geo.continental_block_id` (ancient blocks), `hist.crust_age_ma`, `volcanic_age_ma`, `hist.events` (159 events), checkpoints every 5 Ma **without land masks** | PARTIAL: lineage pools from blocks and island ages; past land bridges are not reconstructable → `history_quality = present_only` |
| weather / actual year (P1) | `atmosphere` weather year (off by default, `weather_year=false`), drought masks of the 5.4 generator; 5.4.1 G5 actual soil water | available only when the user enables the weather year |
| existing vegetation | `biomes.pft_*` (21 PFT covers), `lai_proxy`, `potential/actual_biomass_kg_c_m2` (Miami NPP), `stand_age_yr`, `succession_stage` | used as *comparison* only; 5.5 NPP is a separate LUE model, not a rename of the Miami proxy |

## What does not exist (must be built in 5.5)

| item | status before 5.5 |
|---|---|
| functional-group / taxon registry, stable taxon ids | MISSING |
| habitat area fractions summing to 1 with sea-ice overlay | PARTIAL (biomes covers exist per hex, no realm fractions) |
| niche response curves per group, seasonal survival, breeding window | MISSING |
| LUE GPP/NPP with fAPAR, CUE, nutrient and water limits | MISSING (only Miami NPP) |
| dispersal graphs by mobility, colonisation, occupancy | MISSING |
| bioregions / lineage pools distinct from biomes | MISSING |
| joint food allocation, consumer capacities, food web | MISSING |
| marine / freshwater production and consumers | PARTIAL (habitat potentials only) |
| richness, α/β/γ, Hill, endemism, index, percentile | MISSING |
| renewable resource inventory | MISSING (5.2 crop suitability only) |
| biosphere dynamics, ledger, typed feedback | PARTIAL (5.3 actual vegetation year model, no ledger) |

## Blockers and decisions

* **Past land connections** (B32, S11): the history keeps no land masks per epoch. The
  bridge test is therefore run on a synthetic graph fixture; on generated worlds history
  quality is reported as `present_only` (B33). Status of whole-world B32: BLOCKED.
* **External observations** (B82: MOD17, GBIF, OBIS, TRY): no data in the package, no
  network import approved → BLOCKED; no fake comparison.
* **Nutrient pools in the ocean**: only `marine_nutrient_supply_proxy` exists; marine NPP
  uses it as a dimensionless limitation (APPROX, labelled proxy).
* **Owners for feedback** (C09–C16): the hydrology and soil owners are static normals
  stages; 5.5 implements typed `EcoFeedback` proposals and an owner-side interception /
  litter / erosion-protection operator run in the coupler on copies, with rollback. Re-running
  the 4.x hydrology from a changed canopy is not done (PARTIAL, documented).

G0 approval: P0 can proceed — every P0 input above exists in the current world, except the
ones explicitly marked BLOCKED.
