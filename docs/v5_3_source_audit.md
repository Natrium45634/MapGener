# TZ 5.3 — source audit

The TZ (§24) requires opening every listed source before relying on it.

**What was consulted.** This session had no confirmed web access to the listed URLs, so
nothing was fetched from them. Every row below is marked `reference_only`, from published
literature known to the author. That is not a verified reading of the current repository
or page.

**What was copied.** No code was copied from any project, so there is no license mixing.
The HexWorld implementations are original, reduced-order models that follow the cited
equations.

| ID | Source | What it is used for | Decision | Gate | Status of verification |
|---|---|---|---|---|---|
| O01–O03 | NOAA Ocean Service (currents, coastal upwelling) | qualitative mechanisms: wind-driven gyres, Ekman offshore transport → upwelling, cool foggy coasts | reference_only | G2 | literature knowledge; pages not opened in this session |
| O04 | GFDL MOM6 | C-grid finite-volume face fluxes, conservative heat transport | adapt_idea (face transports from corner streamfunction) | G2 | not opened; no code used |
| O05 | Oceananigans wind-driven gyres | benchmark idea: western intensification depends on β | reference_only (our test: western boundary current) | G2 | not opened |
| — | Stommel (1948), *Trans. AGU* 29 | `r ∇²ψ + β ψ_x = curl τ / ρ` | equation used | G2 | textbook equation |
| I01–I04 | CICE / Icepack | separate thermodynamics, drift and concentration; snow and albedo | adapt_idea (zero-layer energy budget, conservative drift) | G3 | not opened; no code used |
| I05–I06 | NSIDC ice shelves | sea ice ≠ floating shelf ≠ grounded ice | reference_only (realm and cover axes) | G3, G5 | literature knowledge |
| A01 | ERA5 pressure levels | 850 / 500 / 250 hPa, below-ground masks | adapt_idea | G1 | not opened |
| A02–A03 | FAO-56 Penman–Monteith; chapter 4 | ET0 equation; missing-data procedures (eqs. 48, 50, 52) | equation used; Example 18 is the test | G1 | equations and example from FAO-56 (Allen et al. 1998) |
| A04 | pyet | cross-check of PET implementations | reference_only | G1 | not opened; no dependency |
| G01–G07 | PISM, SICOPOLIS, OGGM, EISMINT | SIA, Vialov profile, spin-up practice | planned for G4 | G4 | open |
| E01–E05 | USGS aeolian, ReSCAL, ShorelineS, CoastalME | Bagnold flux, one-line model | planned for G6 | G6 | open |
| B01–B06 | CLM, LPJ-GUESS, páramo, FAO zoning | PFT dynamics, fire, rare classes | planned for G8 | G8 | open |
| — | Mastin et al. 2009; Malin & Sheridan 1982; Iverson et al. 1998; Clague & Mathews 1973 | v5.1 volcanism formulas | equations used | v5.1 | literature knowledge |

**Risks.** Coefficients taken from memory may deviate from the current versions of these
sources. Every such coefficient is marked `[H]` or `CALIBRATED_APPROXIMATION` in code and
docs, and the solver's behaviour is tested by numerical invariants (budgets, signs,
benchmarks), not by claims of agreement with these models.
