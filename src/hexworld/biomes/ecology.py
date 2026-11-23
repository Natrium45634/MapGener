"""Actual ecosystems: growth, fire, volcanic disturbance, succession (TZ 5.3 sec. 13, P1).

The v5.0 classes stay the **potential** natural vegetation; this module keeps a separate
**actual** state per hex, run year by year (``dynamic_vegetation_annual``; a deterministic
reduced model, not a DGVM)::

    NPP     = 0.45 min(3 (1 - e^(-0.000664 P)), 3 / (1 + e^(1.315 - 0.119 T)))   kg C/m^2/yr
              (Miami model, Lieth 1975, dry matter -> carbon)
    dC/dt   = NPP (1 - C / C_max) - m C          (C_max from the potential family)
    fire    p = 1 - exp(-lambda),  lambda = 0.08 fuel dryness^2 warmth
    burn    C <- C (1 - s),  emitted s C (tracked),  stand age reset on the burnt share
    volcano a share f of the hex at the event year is reset to bare ground and bare rock:
            growth x soil_dev, d(soil_dev)/dt = (1 - soil_dev) / 100 yr (primary succession)

Random draws come from ``splitmix64(seed, hex, year)``: the same in any order of hexes
(no global generator consumed in sequence). Establishment needs soil, and needs seeds:
a bare hex regrows only from its own surviving share or from vegetated neighbours.
No wildfire without fuel (ice, bare stone, hyper-arid desert: C ~ 0).
"""

from __future__ import annotations

import numpy as np

#: carrying capacity of living biomass by potential family, kg C/m^2 [H]
C_MAX = {
    "forest_tropical": 18.0, "forest_temperate": 13.0, "forest_boreal": 8.0,
    "woodland": 6.0, "shrubland": 2.5, "grassland": 1.0, "tundra": 0.6, "mountain": 0.8,
    "wetland": 4.0, "desert": 0.15, "barren": 0.0, "ice": 0.0, "polar": 0.05,
    "special": 0.5, "freshwater": 0.0, "marine": 0.0,
}  # fmt: skip
MORTALITY = 0.008  # 1/yr
#: time scale of soil formation on fresh volcanic ground (primary succession), years [H]
SOIL_TAU_YR = 100.0
STAGES = ("bare", "pioneer", "grass_shrub", "young_forest", "mature")


def _splitmix64(x):
    x = (x + np.uint64(0x9E3779B97F4A7C15)) & np.uint64(0xFFFFFFFFFFFFFFFF)
    z = x
    z = ((z ^ (z >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)) & np.uint64(0xFFFFFFFFFFFFFFFF)
    z = ((z ^ (z >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)) & np.uint64(0xFFFFFFFFFFFFFFFF)
    return z ^ (z >> np.uint64(31))


def uniform(seed_int: int, hex_ids: np.ndarray, year: int) -> np.ndarray:
    """Uniform [0, 1) per hex for one year, independent of the order of the hexes."""
    with np.errstate(over="ignore"):
        k = (np.asarray(hex_ids, dtype=np.uint64) * np.uint64(1_000_003)
             + np.uint64(year) * np.uint64(7_919) + np.uint64(seed_int % (2**63)))  # fmt: skip
        return (_splitmix64(k) >> np.uint64(11)).astype(np.float64) / float(2**53)


def miami_npp(t_c, p_mm):
    t = np.asarray(t_c, dtype=np.float64)
    p = np.maximum(np.asarray(p_mm, dtype=np.float64), 0.0)
    dm = np.minimum(3.0 * (1.0 - np.exp(-0.000664 * p)), 3.0 / (1.0 + np.exp(1.315 - 0.119 * t)))
    return 0.45 * np.maximum(dm, 0.0)


def run(family_names, t_ann, p_ann, dry_months, soil, nbr, seed_int, years=300,
        disturbances=None) -> dict:  # fmt: skip
    """Year-by-year actual state; ``disturbances``: list of (year_index, hex_idx, fraction)."""
    n = len(family_names)
    cmax = np.array([C_MAX.get(f, 1.0) for f in family_names], dtype=np.float64)
    cmax = np.where(np.asarray(soil, bool), cmax, np.minimum(cmax, 0.05))
    npp = miami_npp(t_ann, p_ann)
    c = 0.5 * cmax  # start half-grown (the record before the window is unknown)
    age = np.full(n, 50.0)
    soil_dev = np.ones(n)  # 0 on fresh lava / ash, 1 on a mature soil
    emitted = np.zeros(n)
    burnt_last = np.zeros(n)
    burn_count = np.zeros(n)
    dist = {}
    for y, idx, f in disturbances or []:
        dist.setdefault(int(y), []).append((np.asarray(idx, dtype=np.int64),
                                            np.asarray(f, dtype=np.float64)))  # fmt: skip
    dryness = np.clip(np.asarray(dry_months, dtype=np.float64) / 8.0, 0.0, 1.0)
    warmth = np.clip((np.asarray(t_ann, dtype=np.float64) - 0.0) / 20.0, 0.0, 1.0)
    hex_ids = np.arange(n)
    for year in range(years):
        # volcanic resets of a share of the hex (lava, PDC, thick ash, lahar)
        for idx, f in dist.get(year, []):
            f = np.clip(f, 0.0, 1.0)
            c[idx] = c[idx] * (1.0 - f)
            age[idx] = age[idx] * (1.0 - f)
            soil_dev[idx] = soil_dev[idx] * (1.0 - f)  # primary succession starts on rock
        # growth with a seed source: own living share or vegetated neighbours
        seeds = c > 0.02 * np.maximum(cmax, 1e-9)
        nb_seed = np.zeros(n, dtype=bool)
        for k in range(nbr.shape[1]):
            j = nbr[:, k]
            nb_seed |= (j >= 0) & seeds[np.maximum(j, 0)]
        can = (seeds | nb_seed) & (cmax > 0)
        growth = np.where(can, npp * soil_dev * (1.0 - c / np.maximum(cmax, 1e-9)), 0.0)
        soil_dev = soil_dev + (1.0 - soil_dev) / SOIL_TAU_YR  # weathering, humus
        c = np.clip(c + growth - MORTALITY * c, 0.0, cmax)  # logistic: never above C_max
        age = age + 1.0
        # fire: fuel (grass, litter, shrubs ~ the first 2 kg C), a dry season, warmth
        fuel = np.minimum(c, 2.0)
        lam = 0.08 * fuel * dryness**2 * warmth  # wet forests seldom burn
        p = 1.0 - np.exp(-lam)
        u = uniform(seed_int, hex_ids, year)
        burn = (u < p) & (fuel > 0.05)
        sev = 0.35
        loss = np.where(burn, sev * c, 0.0)
        c = c - loss
        emitted += loss
        age = np.where(burn, age * (1.0 - sev), age)
        burn_count += burn
        if year >= years - 10:
            burnt_last += burn * sev
    frac = np.where(cmax > 0, c / np.maximum(cmax, 1e-9), 0.0)
    forest_pot = cmax >= 6.0
    stage = np.select(
        [c < 0.03, frac < 0.15, (frac < 0.45) | ~forest_pot, frac < 0.7],
        [0, 1, 2, 3], default=4,
    ).astype(np.uint8)  # fmt: skip
    fri = np.where(burn_count > 0, years / np.maximum(burn_count, 1), np.inf)
    return {
        "actual_biomass_kg_c_m2": c, "potential_biomass_kg_c_m2": cmax,
        "stand_age_yr": age, "succession_stage": stage,
        "burn_fraction_last_decade": burnt_last / 10.0,
        "fire_return_interval_yr": fri, "carbon_emitted_kg_c_m2_yr": emitted / years,
        "fuel_load_kg_c_m2": np.minimum(c, 2.0),
        "summary": {
            "mode": "dynamic_vegetation_annual", "years": years,
            "living_carbon_mean_kg_m2": float(c.mean()) if n else 0.0,
            "actual_over_potential": float(c.sum() / max(cmax.sum(), 1e-9)),
            "burnt_hex_years": int(burn_count.sum()),
            "stage_counts_vegetable_land": {STAGES[i]: int(((stage == i) & (cmax > 0)).sum())
                                            for i in range(5)},  # fmt: skip
            "quality": "reduced_order_CALIBRATED_APPROXIMATION (not a DGVM)",
        },
    }  # fmt: skip
