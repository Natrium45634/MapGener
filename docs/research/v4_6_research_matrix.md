# v4.6 research matrix (TZ 4.6, 10.4)

No third-party code was copied, linked or run for v4.6. Every algorithm below was written
for HexWorld from the published formula or from a standard numerical method; the "license"
column therefore only says whether code was taken (none). Web pages were **not** opened in
this session; "verified" states what was checked and how.

| source | component / file | algorithm | verified | unit assumptions | license | action |
|---|---|---|---|---|---|---|
| Parsons & Sclater 1977, JGR 82, doi 10.1029/JB082i005p00803 | `geomorph/ocean_age.py` `parsons_sclater_depth` | d = 2500 + 350 √t; d = 6400 − 3200 e^(−t/62.8); smoothstep blend 60–80 Ma | control numbers of TZ 4.3 (0 / 25 / 100 Ma → 2500 / 4250 / 5749 m) in `test_v46`; monotone; slope continuous through the blend | m below datum, Ma | formula only | reuse as reference, parameters in config |
| ridge segmentation / fracture zones (USGS "Understanding plate motions"; standard marine geology) | `ocean_age.build_ridge_pieces`, `flowline_age` | orthogonal ridge pieces along the mean relative-velocity direction; age = distance along spreading / half-rate; piece switch = fracture zone | synthetic ridge: symmetric ages, slower ridge older; offset ridges: age jump = offset / half-rate, along the spreading direction; active transform between the axes | km, mm/yr = km/Ma | own code | approximation, flagged `age_estimation_mode = approx` |
| thin elastic plate (Turcotte & Schubert, *Geodynamics*; TZ 3.4) | `geomorph/basins.py` `flexure` | D∇⁴w + Δρ g w = q, D = E Te³ / 12(1−ν²), spectral solution on a zero-padded raster | zero load → zero; linear in the load; foredeep then forebulge; α = (4D/Δρg)^¼ ≈ 80 km at Te = 25 km | Pa, m, N·m | own code | reuse; only outside the load (the root is in the surface) |
| pull-apart kinematics (Dead Sea basin literature listed in TZ 10.2) | `basins.plan_boundary_basins` | releasing step: v_n > 0 with \|v_t\| > 2 v_n; rhomb basin with two border faults and sub-basins | releasing → basin, restraining → none, opening → rift (`test_v46`) | mm/yr, km, m | own code | heuristic [А] |
| half-graben rifts (TZ 3.5) | `basins._evaluate` rift branch | segmented half-grabens, alternating border fault, en-echelon offsets, relay ramps, shoulders | elongated, one steep and one gentle wall, positive shoulders (`test_v46`) | km, m | own code | heuristic [А] |
| Fritsch & Carlson 1980 (monotone PCHIP) | `geomorph/margins.py` `pchip3` | monotone cubic Hermite through break / foot / rise end, zero end slopes | monotone, no gradient kink (C1), widths in km independent of the hex size (`test_v46`) | km, m | own code | reuse |
| submarine canyons and turbidity currents (USGS references of TZ 10.3) | `geomorph/canyons.py` | sources at river mouths with v4 sediment delivery; strictly downhill path; sub-hex canyon (depth, width), fan cone; closed volume budget | no supply → no fan; downhill only; budget closure 0 (`test_v46`) | m³/yr, m, km | own code | heuristic [А], `approximate` |
| priority-flood / spill (Barnes et al. 2014, standard) | existing `stages/hydrology.route` | water level of a depression = its lowest way out; the physical surface untouched | `test_priority_flood_keeps_the_physical_surface` | m | own code (v4) | reuse |
| topographic prominence (standard definition) | `geomorph/massifs.py` `prominence` | union-find from the top down, key col at the merge | two-peak test | m | own code | reuse |
| Harris et al. 2014 / Yu et al. 2025 (submarine landform classes) | `geomorph/classify.py` | provinces from geology + morphology + neighbourhood (not depth thresholds) | shares on Aurelia4 / Terra7 reported in the report | – | none taken | names aligned with TZ 4.1 |
| GEBCO 2026, ETOPO 2022 | – | real regional statistics | **not done**: no download (TZ asks not to load 7 GB for a test) | – | – | open item |
| EarthByte/simple_paleobathymetry (GPL-2.0 per TZ) | – | – | not opened, not used | – | GPL-2.0 (per TZ) — not copied | declined |
| Landlab SPACE / stream power | – | – | not used: v4 already has a stream-power + sediment solver on the hex graph | – | – | declined (reuse v4) |
