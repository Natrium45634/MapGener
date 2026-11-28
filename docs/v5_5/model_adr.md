# TZ 5.5 — model decisions (ADR)

Package `src/hexworld/biosphere/`, stage `v55_biosphere` (after v5.2 resources and the
5.5.1 plan). Algorithm `hexworld-biosphere-5.5.0`, registry `hexworld-bio-registry-1`.

| # | Decision | Why | Rejected |
|---|---|---|---|
| A1 | Functional groups → taxon templates (thermal × moisture variants) → procedural taxa (template × lineage pool), ids = sha256 of `bio_registry_v1 / seed / template / lineage key` | the TZ asks for analogues, not species; stable ids independent of traversal order; case-sensitive seed | random species list per biome |
| A2 | Habitat area fractions land_ground + inland_water + marine_water + grounded_ice = 1, sea ice an overlay | densities integrate only the right habitat once (B11–B13) | whole-hex densities |
| A3 | Niche = trapezoids; suitability S = E · weighted geometric mean (TZ §6); structural cover uses the most limiting score (Liebig) | the soft mean is too lenient for cover: one dry factor must cap trees | S as a probability |
| A4 | LUE production: APAR by canopy → shrub → ground layers (Beer–Lambert), GPP = ε·APAR·f_T·f_W·f_nutr, NPP = CUE·GPP; LAI iterated from leaf carbon; a water-use ceiling GPP ≤ 3.5 gC/kg·AET; rainfall caps of woody (Sankaran) and herb cover | MOD17 / LPJ-like, cheap, explainable; transmitted + absorbed ≤ incident (B22) | Miami NPP of 5.3 (kept for comparison only) |
| A5 | Microsites: 1 / 3 / 5 elevation bins of the 4.7 hypsometry per hex (lapse 6.5 K/km) | a cold template survives in the high part of a warm hex (B14) | mean temperature only |
| A6 | Dispersal: one sparse directed graph per mobility class on the native grid, costs in km, current drift on the sea and downstream drift on rivers, multi-source Dijkstra (`min_only`) from lineage pools | O(E log V), no all-pairs; physical km make grids comparable (B31) | diffusion to random neighbours |
| A7 | Lineage pools: large mostly continental landmasses (≥ 2·10⁵ km²), large drainage basins (≥ 4·10⁴ km²), ocean basin × thermal belt | the world keeps no past land masks: present-only history, no invented bridges (B33) | lineages from biomes |
| A8 | Occupancy: closed form of dp/dt = c(1−p) − e·p with a fat-tailed kernel c = κ0[(1−ρ)e^{−d/L} + ρ e^{−d/(ΛL)}], e from area, body mass and suitability | exact, bounded, islands depauperate with distance (S09, S10) | thresholds on distance |
| A9 | Joint food allocation per pool, λ = min(1, A/Σreq); one demand per group split over its diet; pools of animal production open only after their providers; Damuth space cap for land vertebrates | no double accounting of shared food (B40–B44) | an independent carrying capacity per group |
| A10 | Marine: phytoplankton from surface PAR, ice transmission once, nutrient proxy; lit-bottom producers by the photic share and substrate; export (pe 0.2) with the Martin curve to the bottom; vents an explicit heat subsidy | light and nutrients both needed (B53, B54); the bottom eats only what sinks (B55) | SST-only placement |
| A11 | Richness = unique present templates (× lineage for the world count); functional richness separate; α/β/γ, Hill, index with fixed versioned bounds, area-weighted world percentile; species–area prior separate | B10, B58–B63 | sum of guild counts as «species» |
| A12 | Bioregions = combination of walker / seed / flier lineages × the drainage province | barrier structure, not climate: two same-climate continents differ (B34) | clusters of biomes |
| A13 | Resources: stocks vs increments, a shared `pool` key for alternative uses (timber ≡ fuelwood), harvest cap = safe share × surplus ≥ 0, MSY only for logistic stocks, wild-precursor score separate from accessibility | C21, B64–B67 | additive totals of alternative uses |
| A14 | Dynamics (P1): yearly rain factors (AR(1) in time, 900 km in space, namespace `bio_events_v1`), exact pool relaxation dX/dt = in − X/τ, fire 1/FRI (doubled in dry years) with event ids applied once, detritus first-order with Q10, typed ledger and residual | B26, B68–B71, B84 | rewriting the biome map each year |
| A15 | Feedback: typed `EcoFeedback` proposals applied by owner rules through a transactional `Coupler` (validate → stage → commit, rollback on any failure); default one-way: recorded, not applied | C09–C16, B77; no recursive regeneration | writing into other layers |

## What is approximate (APPROX) or reduced

* ocean surface shortwave: Angström–Prescott with n/N = 0.5 (no ocean cloud field);
* marine and freshwater nutrients: dimensionless proxies (no nutrient pools in the world);
* the occupancy uses the class graph (not a per-template habitat path);
* bioregion lineages are present-only; procedural splitting of lineages is off;
* the food web is annual (seasonal states are reported for migratory groups only);
* in DYNAMIC mode the yearly forcing is a regional rain factor, not the 5.4 weather year.
