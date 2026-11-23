# TZ 5.2 — research report: sources, formulas, decisions

Accessed: 23.11.2026. **No external page or repository was opened in this session** (the
work used no network fetch): every row below is `not inspected` for the URL / tag / licence
columns. The formulas used are standard published relations known to the author; no code
was copied from any project, so no licence applies to the implementation. Coefficients
marked [A] are HexWorld calibrations, not Earth constants.

| Source | accessed | checked tag / commit | licence | scientific assumption | usable formula | data units | constraints | decision | implementation location | verified with test |
|---|---|---|---|---|---|---|---|---|---|---|
| S01 USDA NRCS soil health | not inspected | — | — | fertility = physical + chemical + biological indicators | indicator matrix (no single SOC number) | — | — | adapt (concept) | `resources/soil.py` factor vector, `FACTORS` | S19, S20, S24 |
| S02 FAO GAEZ | not inspected | — | — | crop-specific edaphic suitability with limiting factors | `S = min^a (prod s^w)^(1-a)` (TZ 11) | 0..1 | not a yield | adapt | `soil.suitability`, `PROFILES` | S24, I03 |
| S03 FAO HWSD | not inspected | — | — | soil qualities as separate classes | pH / salinity / rooting / oxygen / retention separate | — | — | adapt (concept) | `soil.factor_scores` | S19, S20 |
| S04/S05 ISRIC SoilGrids | not inspected | — | — | profile intervals, units | functional layers 0–0.3, 0.3–1.0 m, rest | kg/m², cmol(+)/kg | not a P/K source | adapt (layering) | `soil.layer_thickness` | S04 |
| S06/S07 Rosetta | not inspected | — | — | pedotransfer of hydraulics | — | — | v4.5 already uses Saxton–Rawls 2006 | reject (no duplicate) | `stages/environment.saxton_rawls` | v4.5 tests |
| S08–S10 RothC 26.3 | not inspected | — | — | DPM / RPM / BIO / HUM / IOM pools, monthly modifiers | `a(T)=47.91/(1+exp(106.06/(T+18.27)))`; b from topsoil deficit; `x=1.67(1.85+1.60 e^{-0.0786 clay%})`; BIO 46 % / HUM 54 %; k = 10, 0.3, 0.66, 0.02 /yr; IOM `0.049 SOC^1.139` (Falloon 1998) | kg C/m², 1/yr | topsoil model: the profile follows a depth distribution [A] | adapt | `soil.rothc_*`, `rothc_equilibrium`, `advance_soil_month` | S06, S07, N03 |
| S11 DayCent | not inspected | — | — | minimal N pools, nitrification, denitrification under saturation | NH4 / NO3 steady state; `k_nit`, `k_denit` [A] | kg N/m² | no N2O split | adapt (concept) | `soil.compute_fertility_static`, `advance_soil_month` | S09–S12, N07, N09 |
| S12 EPIC | not inspected | — | — | erosion removes soil mass and its nutrients | eroded = E ρ c, routed with retention | kg/yr | — | adapt | `pipeline_v52` transport, `secondary.retention_route` | S23, I12 |
| S13 NRCS agronomy | not inspected | — | — | total ≠ available P; pH / CEC control availability | P pools; Al-toxicity at pH < 5 and BS < 0.3 | — | proxy | adapt | `soil.factor_scores` | S16 |
| Lieth 1975 (Miami model) | not inspected | — | — | climate-limited NPP | `min(3000/(1+e^{1.315-0.119T}), 3000(1-e^{-0.000664P}))` g DM/m²/yr | g DM | litter input proxy | adapt | `soil.miami_npp_kg_c` | scenes |
| Adams 1973 | not inspected | — | — | bulk density of mineral + organic mixtures | `100/(OM/244 + (100-OM)/BD_min)` | kg/m³ | — | adapt | `soil.bulk_density_adams` | S02 |
| Maas & Hoffman 1977 | not inspected | — | — | relative yield vs salinity | `1 - b (EC - a)` | dS/m | thresholds per profile [A] | adapt | `soil.salinity_score` | S19 |
| Walker & Syers 1976 | not inspected | — | — | P transformations with soil age | primary → labile / sorbed / organic → occluded | kg P/m² | rates [A] | adapt | `soil.integrate_linear` P system | S13, S14, scenes |
| USDA Handbook 60 (1954) | not inspected | — | — | EC ≈ TDS / 640 | TDS mg/L / 640 | dS/m | proxy; solid salt above saturation not counted | adapt | `compute_fertility_static` | N08, S17 |
| M01–M18 USGS deposit models, grade-tonnage | not inspected | — | — | permissive tract → count → grade-tonnage (3-part method) | tract masks, `N ~ Poisson(λ ∫ S dA)`, joint log-normal | t, mass fraction | medians rounded, `synthetic_not_calibrated` | adapt | `resources/models.py`, `deposits.py` | R01–R26, N19, R26 |
| M07 USGS Li brines | not inspected | — | — | closed basin + source + aquifer + time | brine = pore water × concentration | mg/L, m³ | retention 0.1 [A] | adapt | `secondary.closed_basin_salts` | R10, N16 |
| M08 USGS gold placers | not inspected | — | — | eroded lode → river concentration | released = η × eroded; deposit share from slope drop | t | mass-limited | adapt | `secondary.river_placers` | R13, R14 |
| M12/M13 USGS / BOEM marine | not inspected | — | — | nodules on slow abyssal plains, crusts on seamount rock | areal density × area | kg/m² | — | adapt | tract R30, R31 | R16, R17 |
| M19 USGS aggregates | not inspected | — | — | widespread construction materials, quality differs | dense class + quality | — | — | adapt | `secondary.construction_materials` | R33 layer (I14) |
| G01–G05 GemPy, LoopStructural, GSTools, PyKrige, SimPEG | not inspected | — | — | implicit geometry, geostatistics | — | — | no runtime dependency | optional (not used) | — | — |
| G06/G07 GLiM, Macrostrat | not inspected | — | — | lithology ranges | — | — | reference only | optional (not used) | lithology adapter is procedural | — |
| G09–G11 WorldEngine, Landlab, Badlands | not inspected | — | — | staged pipelines, sediment transport | — | — | — | reject (own pipeline exists) | — | — |
