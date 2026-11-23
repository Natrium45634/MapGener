# TZ 5.4 — external models and sources (R01–R28)

**Verification level.** No external repository was cloned, executed or copied for 5.4.
Rows below record what HexWorld took as an *idea or equation* (re-derived, own units) and
from which public description. "Commit / licence checked" is **not claimed**: the sandbox
of this work had no audit of those repositories; licences are listed as published by the
projects where widely known, and must be re-checked before any code reuse. No code or
calibration data of these projects is in HexWorld.

| № | Source | Idea used in HexWorld | Where | Not taken |
|---|---|---|---|---|
| R01 Isca, R27 | idealised moist GCM hierarchy | separate the monthly forcing from the event time step; test each component alone | `normals`, `weather` | the GCM itself |
| R02/R03/R28 PlaSim, ExoPlaSim | other rotation / obliquity / day length | Hadley width scales with rotation; flag `global_hadley_regime_unvalidated` outside Earth-like range | `circulation.hadley_edge_deg` | the dynamical core |
| R04 climlab | component-wise testing | each operator has its own unit test | tests | code |
| R05 MPAS | conservative fluxes on Voronoi/hex faces (C-grid idea) | antisymmetric face fluxes on a 4- or 6-face mesh | `fv.py` | the dynamical core |
| R06 ERA5 | Earth levels and shear | sanity ranges of jets (30–60 m/s at 250 hPa) | tests T16 | runtime data |
| R07/R08/R25 IMERG, GPCP | monthly climatology; monsoon diagnostic (summer–winter > 2 mm/day, summer share ≥ 55 %) | suitability uses a rain range and a summer share as comparison, not as a generator | `monsoon.suitability` | data |
| R09/R10/R11 monsoon & ITCZ reviews | monsoon = seasonal migration of convergence; ITCZ toward the warmer hemisphere; not zonally uniform | thermal equator per column; ITCZ from `-div Q`; regional heat lows | `circulation`, `normals.itcz` | fixed parallels |
| R12 WMO TC guide | genesis needs warm deep ocean, moist mid-levels, low shear, vorticity, instability | product of six factors | `storms.genesis_factors` | a single SST threshold |
| R13/R24/R26 IBTrACS | basin differences, ~85 TC / yr on Earth | λ0 calibration order of magnitude | `storms.LAMBDA0` | track data |
| R14/R15 STORM, IH-STORM | statistical synthetic tracks; ENSO dependence only with a basin mode | Poisson genesis + steered tracks; interannual only with an ocean mode | `storms`, `weather` | code, data (STORM is unmaintained) |
| R16 tropical_cyclone_risk | steering by a deep-layer mean wind, intensity ODE | 850/500/250 weights 0.4/0.35/0.25, `dV/dt` with PI, shear, land | `storms.run_track` | code |
| R17 Holland 1980, PaHM | radial pressure profile, gradient wind | `holland_pressure`, `holland_wind` (finite at f = 0) | `storms` | calibrated B on land |
| R18/R19 TempestExtremes, CyTRACK | object detection with stable ids | event ids `TYPE-yearseed-year-month-n`; AR as connected IVT objects | `weather`, `storms.detect_ar` | code |
| R20 MetPy | vorticity, divergence, frontogenesis definitions | same definitions on the hex/raster mesh | `fv.Mesh.divergence`, `storms.frontogenesis_proxy` | square-grid code on hexes |
| R21 GWGEN | Markov wet/dry + Gamma amounts | `weather.markov_params`, Gamma k = 0.75 | `weather` | code |
| R22 EGGS-WG | spatially correlated weather generator (GPL-3.0) | idea of spatial correlation only | `weather.correlated_uniforms` | **no code** (licence) |
| R23 NOAA CPC ENSO | regional ENSO impacts on Earth | *rejected* as a template for procedural planets | — | teleconnection maps |
