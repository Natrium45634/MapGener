# TZ 5.5 — scientific validation (separate from the unit results)

Validation tiers of §17 and what was actually done. No external code or dataset was run or
downloaded; the research sources R01–R15 were used as published descriptions only.

| tier | what | status |
|---|---|---|
| 1 analytical | N01–N20 equations, units, masks, budgets (`tests/test_v55_numeric.py`) | PASS (31 tests) |
| 2 synthetic causality | manufactured worlds with positive and negative controls (`tests/test_v55_scenarios.py`, `biosphere/synthetic.py`) | PASS (20 tests) |
| 3 whole world | repeatability, hashes of the other layers, save/load, dynamics ledger (`tests/test_v55_integration.py`), performance and resolution (`performance_report.md`) | PASS / see report |
| 4 Earth reference (MOD17, GBIF, OBIS, TRY) | needs licensed offline datasets and approval | **BLOCKED** — no data; nothing is claimed |

## Plausibility of the productivity (not a calibration)

Median NPP and plant carbon by the biome family of v5.0 on the world `Bio55` (120×80,
60 km, full pipeline), land hexes ≥ 80 % ground, per m² of ground, after the LUE scaling of
`parameter_provenance.csv`:

| family (v5.0) | hexes | NPP gC/m²/yr | plant C kgC/m² | typical literature range NPP / C |
|---|---|---|---|---|
| tropical forest | 9 | 956 | 13.7 | 800–1200 / 15–20 |
| subtropical forest | 26 | 975 | 13.9 | 600–1000 / 10–18 |
| temperate forest | 1023 | 603 | 10.4 | 500–800 / 8–15 |
| boreal forest | 25 | 538 | 10.1 | 200–450 / 4–10 |
| woodland | 132 | 534 | 9.4 | 300–600 / 3–8 |
| shrubland | 117 | 614 | 8.7 | 200–500 / 2–6 |
| savanna | 107 | 610 | 8.0 | 400–800 / 2–6 |
| grassland | 528 | 496 | 6.2 | 200–500 / 0.5–2 |
| wetland | 87 | 529 | 8.1 | 400–1000 / 2–10 |

Reading: the forests sit inside the usual ranges; **boreal, shrubland and grassland NPP and
the plant carbon of open vegetation are high** (the woody share of open landscapes is too
large in this world, fire is only in the P1 dynamics). This is a known bias of the
uncalibrated priors, reported here rather than tuned per map. A calibration needs MOD17-like
reference grids (tier 4, BLOCKED).

## Qualitative patterns checked on generated worlds

* remote oceanic islands are poorer than the mainland and get poorer with distance; walking
  mammals are absent without a land route; an old continental fragment keeps its own
  lineages (endemics) — observed on `Bio55` (islands Talenland … Elenor, fragment Ilorvar);
* bioregions follow lineages and drainage provinces, not climate belts;
* freshwater richness concentrates along rivers and lakes; marine production is higher on
  shelves and in upwelling than in the open warm sea.

## Limits (honest)

The analogues are functional groups and their variants, not species; richness numbers are
counts of these analogues, not of real species; the species–area prior is reported
separately and never added. History is `present_only`. Every parameter is provisional.
