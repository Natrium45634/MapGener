# ruff: noqa: E501  (trait / class tables read better one row per line)
"""Plant functional types: traits, suitability, productivity proxy, cover (TZ 5.0, 5-6).

A light equilibrium PFT model in the spirit of BIOME4 (Kaplan et al. 2003) - architecture
only, no code - with smooth tolerance curves instead of hard cut-offs [А]. Every number in
``TRAITS`` is a **starting value of the profile ``pft_earthlike_v1`` [П]**, to be calibrated,
not a natural constant.

Composition of the suitability (documented choice, no double count of drought):

    S_p = H_p * s_cold * s_chill * s_heat * s_water * s_soil * s_flood * s_salt * s_snow

* thermal: coldest-month mean (``tc``), warmth (GDD above the PFT base) and the v4.7 thermal
  tree score for trees (the tree limit of 4.7 is an input, not a second rule);
* water: **one** water stress - the reference AET/PET of the unit (4.5 kernel on the 4.7
  unit), annual and its dry-season length. The aridity index P/PET is diagnostic only.
* soil, flooding, salinity, snow: independent constraints.

Productivity proxy: ``GPP_m ~ eps PAR_incident,m (1 - exp(-k LAI)) s_T,m s_w,m`` with
``PAR = 0.48 Rs`` from the 4.7 radiation; normalised by an ideal year (dimensionless index,
not gC/m^2). Cover: vegetation fraction ``V`` from the best productivity, the rest stays bare
(no softmax forcing a forest into a desert); PFT weights ``exp(score/tau)`` among the
suitable PFTs only.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .registry import PFT_REGISTRY_VERSION


@dataclass(frozen=True)
class Trait:
    id: str
    form: str  # tree / shrub / grass / forb / moss / succulent / halophyte
    leaf: str
    tc_min: float  # coldest-month mean below which it dies, degC
    tc_max: float  # coldest-month mean above which it lacks chilling / is out-competed
    gdd_base: float  # degC
    gdd_min: float  # degC day above the base
    tw_min: float  # warmest-month minimum, degC
    alpha_min: float  # reference AET/PET needed
    dry_max: float  # tolerated dry months (AET/PET < 0.4 in the warm season)
    flood_tol: float  # tolerated waterlogged months
    flood_need: float  # waterlogged months needed (wetland specialists)
    salt_tol: float  # salinity proxy 0..1 tolerated
    soil_min_m: float
    snow_tol: float  # tolerated snow-covered months
    lai: float
    eps: float  # light use efficiency, relative


T = Trait
TRAITS: tuple[Trait, ...] = (
    T(
        "TREE_TROP_EVERGREEN",
        "tree",
        "evergreen",
        15.5,
        99,
        5,
        5500,
        22,
        0.80,
        2.0,
        1.0,
        0,
        0.10,
        0.6,
        0,
        6.0,
        1.0,
    ),
    T(
        "TREE_TROP_SEASONAL",
        "tree",
        "raingreen",
        15.5,
        99,
        5,
        5000,
        22,
        0.55,
        6.0,
        1.0,
        0,
        0.10,
        0.5,
        0,
        4.5,
        0.95,
    ),
    T(
        "TREE_TROP_DRY",
        "tree",
        "raingreen",
        12.0,
        99,
        5,
        4500,
        22,
        0.32,
        9.0,
        1.0,
        0,
        0.15,
        0.4,
        0,
        2.5,
        0.85,
    ),
    T(
        "TREE_TEMP_DECID",
        "tree",
        "summergreen",
        -18.0,
        15.5,
        5,
        1100,
        14,
        0.55,
        3.5,
        1.5,
        0,
        0.10,
        0.5,
        6,
        5.0,
        0.9,
    ),
    T(
        "TREE_TEMP_EVERGREEN",
        "tree",
        "evergreen",
        3.0,
        18.5,
        5,
        2200,
        16,
        0.60,
        3.0,
        1.0,
        0,
        0.10,
        0.5,
        2,
        5.0,
        0.9,
    ),
    T(
        "TREE_TEMP_CONIFER",
        "tree",
        "evergreen",
        -22.0,
        12.0,
        5,
        800,
        12,
        0.45,
        4.0,
        1.0,
        0,
        0.10,
        0.4,
        7,
        5.0,
        0.85,
    ),
    T(
        "TREE_BOREAL_EVERGREEN",
        "tree",
        "evergreen",
        -35.0,
        -1.0,
        5,
        380,
        10,
        0.45,
        4.0,
        1.5,
        0,
        0.10,
        0.3,
        9,
        4.0,
        0.8,
    ),
    T(
        "TREE_BOREAL_DECID",
        "tree",
        "summergreen",
        -60.0,
        3.0,
        5,
        300,
        10,
        0.32,
        5.0,
        2.0,
        0,
        0.10,
        0.3,
        9,
        3.0,
        0.8,
    ),
    T(
        "TREE_MONTANE_CLOUD",
        "tree",
        "evergreen",
        4.0,
        20.0,
        5,
        1800,
        10,
        0.75,
        2.0,
        1.0,
        0,
        0.10,
        0.4,
        1,
        5.0,
        0.9,
    ),
    T(
        "TREE_FLOOD_TOLERANT",
        "tree",
        "mixed",
        -30.0,
        99,
        5,
        600,
        12,
        0.40,
        6.0,
        9.0,
        3.0,
        0.20,
        0.3,
        7,
        4.0,
        0.85,
    ),
    T(
        "TREE_MANGROVE",
        "tree",
        "evergreen",
        16.0,
        99,
        10,
        4000,
        24,
        0.0,
        12.0,
        12.0,
        12.0,
        0.95,
        0.2,
        0,
        3.0,
        0.8,
    ),
    T(
        "SHRUB_MESIC",
        "shrub",
        "mixed",
        -25.0,
        99,
        5,
        500,
        10,
        0.35,
        5.0,
        2.0,
        0,
        0.15,
        0.2,
        7,
        2.5,
        0.8,
    ),
    T(
        "SHRUB_XERIC",
        "shrub",
        "raingreen",
        -20.0,
        99,
        5,
        900,
        18,
        0.07,
        12.0,
        0.5,
        0,
        0.35,
        0.1,
        4,
        1.0,
        0.7,
    ),
    T(
        "SHRUB_SCLEROPHYLL",
        "shrub",
        "evergreen",
        -1.0,
        15.0,
        5,
        1600,
        18,
        0.25,
        6.0,
        0.5,
        0,
        0.20,
        0.2,
        1,
        2.0,
        0.75,
    ),
    T(
        "SHRUB_DWARF_COLD",
        "shrub",
        "mixed",
        -55.0,
        99,
        0,
        180,
        4,
        0.20,
        6.0,
        4.0,
        0,
        0.10,
        0.05,
        11,
        1.0,
        0.7,
    ),
    T(
        "GRASS_C3",
        "grass",
        "summergreen",
        -45.0,
        99,
        5,
        250,
        8,
        0.15,
        7.0,
        3.0,
        0,
        0.25,
        0.1,
        8,
        2.0,
        0.8,
    ),
    T(
        "GRASS_C4",
        "grass",
        "raingreen",
        5.0,
        99,
        10,
        1800,
        22,
        0.12,
        9.0,
        3.0,
        0,
        0.25,
        0.1,
        2,
        2.5,
        0.9,
    ),
    T(
        "SEDGE_RUSH",
        "grass",
        "summergreen",
        -50.0,
        99,
        0,
        120,
        5,
        0.30,
        4.0,
        12.0,
        2.0,
        0.30,
        0.0,
        10,
        2.0,
        0.75,
    ),
    T(
        "FORB_ALPINE",
        "forb",
        "summergreen",
        -40.0,
        99,
        0,
        120,
        4,
        0.20,
        5.0,
        2.0,
        0,
        0.10,
        0.05,
        10,
        1.0,
        0.7,
    ),
    T(
        "MOSS_LICHEN",
        "moss",
        "evergreen",
        -80.0,
        99,
        0,
        15,
        0,
        0.10,
        9.0,
        6.0,
        0,
        0.10,
        0.0,
        11.5,
        0.6,
        0.5,
    ),
    T(
        "SUCCULENT_XEROPHYTE",
        "succulent",
        "evergreen",
        0.0,
        99,
        10,
        1500,
        22,
        0.02,
        12.0,
        0.5,
        0,
        0.40,
        0.05,
        0,
        0.4,
        0.6,
    ),
    T(
        "HALOPHYTE",
        "halophyte",
        "mixed",
        -30.0,
        99,
        5,
        600,
        15,
        0.05,
        10.0,
        4.0,
        0,
        0.95,
        0.05,
        6,
        0.8,
        0.6,
    ),
)
PFT_IDS = tuple(t.id for t in TRAITS)
PFT_INDEX = {t.id: i for i, t in enumerate(TRAITS)}
FORMS = tuple(dict.fromkeys(t.form for t in TRAITS))
IS_TREE = np.array([t.form == "tree" for t in TRAITS])
IS_SHRUB = np.array([t.form == "shrub" for t in TRAITS])
IS_HERB = np.array([t.form in ("grass", "forb") for t in TRAITS])
IS_MOSS = np.array([t.form == "moss" for t in TRAITS])
VERSION = PFT_REGISTRY_VERSION


def _sig(x):
    """Smooth ramp of a margin in units of its width: 0 at <= -1, 1 at >= +1 [А].

    Unlike a logistic curve it is exactly 1 well inside the tolerance, so independent
    constraints that are all satisfied do not multiply into a spurious penalty.
    """
    t = np.clip((np.asarray(x, dtype=np.float64) + 1.0) / 2.0, 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


@dataclass
class Features:
    """Bioclimatic features of a block of units (all (U,) unless stated)."""

    tc: np.ndarray  # coldest month mean, degC
    tw: np.ndarray  # warmest month mean
    t_m: np.ndarray  # (12, U)
    gdd0: np.ndarray
    gdd5: np.ndarray
    gdd10: np.ndarray
    alpha: np.ndarray  # annual reference AET / PET (0..1)
    alpha_m: np.ndarray  # (12, U) monthly AET / PET
    dry_months: np.ndarray  # warm-season months with AET/PET < 0.4
    summer_dry: np.ndarray  # dry months in the warm half of the year
    ai: np.ndarray  # P / PET (NaN where PET ~ 0)
    p_year: np.ndarray
    pet_year: np.ndarray
    par_m: np.ndarray  # (12, U) MJ/m^2/month incident PAR
    soil_depth: np.ndarray
    waterlog: np.ndarray  # months
    salinity: np.ndarray  # 0..1
    snow_months: np.ndarray
    tree_score: np.ndarray  # v4.7 thermal tree score (1 = warm enough)
    cloud: np.ndarray  # v4.7 orographic cloud potential


def suitability(f: Features) -> np.ndarray:
    """(P, U) suitability 0..1 of every PFT on every unit."""
    P = len(TRAITS)
    out = np.empty((P, f.tc.size))
    for k, t in enumerate(TRAITS):
        s = _sig((f.tc - t.tc_min) / 1.2)
        if t.tc_max < 90:
            s = s * _sig((t.tc_max - f.tc) / 1.2)
        gdd = f.gdd0 if t.gdd_base <= 0 else (f.gdd5 if t.gdd_base <= 5 else f.gdd10)
        s = s * _sig((gdd - t.gdd_min) / (0.15 * t.gdd_min + 40.0))
        s = s * _sig((f.tw - t.tw_min) / 1.5)
        if t.alpha_min > 0:
            s = s * _sig((f.alpha - t.alpha_min) / 0.05)
        s = s * _sig((t.dry_max + 0.5 - f.dry_months) / 0.8)
        if t.soil_min_m > 0:
            s = s * _sig((f.soil_depth - t.soil_min_m) / (0.25 * t.soil_min_m + 0.03))
        s = s * _sig((t.flood_tol + 0.3 - f.waterlog) / 0.6)
        if t.flood_need > 0:
            s = s * _sig((f.waterlog - t.flood_need) / 0.6)
        s = s * _sig((t.salt_tol - f.salinity) / 0.08)
        s = s * _sig((t.snow_tol + 0.5 - f.snow_months) / 0.7)
        if t.form == "tree":
            s = s * np.clip(f.tree_score, 0.0, 1.0)
        if t.id == "TREE_MONTANE_CLOUD":
            s = s * _sig((f.cloud - 0.12) / 0.05)
        if t.id == "SHRUB_SCLEROPHYLL":
            s = s * _sig((f.summer_dry - 1.5) / 0.5)  # needs a dry summer
        if t.id == "TREE_MANGROVE":
            s = s * 0.0  # needs a tide model (v6): never claimed here
        out[k] = s
    return out


#: Temperature where photosynthesis starts, degC, where it differs from the growing-degree base:
#: boreal and temperate conifers and larch fix carbon from ~0-2 degC (Jarvis & Linder 2000;
#: Suni et al. 2003), so in short cool summers they are not out-competed by dwarf shrubs.
PHOTO_BASE_C = {
    "TREE_BOREAL_EVERGREEN": 0.0,
    "TREE_BOREAL_DECID": 1.0,
    "TREE_TEMP_CONIFER": 1.0,
    "TREE_TEMP_DECID": 3.0,
}


def productivity(f: Features, suit: np.ndarray) -> np.ndarray:
    """(P, U) dimensionless productivity index of every PFT (0..~1)."""
    out = np.empty_like(suit)
    ref = 0.48 * 20.0 * 30.4 * 12.0 * (1.0 - np.exp(-0.5 * 6.0))  # an ideal tropical year
    for k, t in enumerate(TRAITS):
        s_t = np.clip((f.t_m - PHOTO_BASE_C.get(t.id, t.gdd_base)) / 15.0, 0.0, 1.0)
        # drought-adapted types reach their full rate at a lower moisture (water-use
        # efficiency): saturation at 2.5 alpha_min + 0.15, within [0.2, 0.8] [А]
        w_sat = float(np.clip(2.5 * t.alpha_min + 0.15, 0.2, 0.8))
        s_w = np.clip(f.alpha_m / w_sat, 0.0, 1.0)
        gpp = (f.par_m * (1.0 - np.exp(-0.5 * t.lai)) * s_t * s_w).sum(axis=0) * t.eps
        out[k] = suit[k] * gpp / ref
    return out


def allocate_cover(suit, prod, open_frac, tau=0.08, v_ref=0.25):
    """Vegetation fraction V and PFT cover fractions (P, U); the rest stays bare.

    ``open_frac`` = share of the unit physically open to plants (not ice, rock, salt crust).
    ``V = open * clip(best productivity / v_ref, 0, 1)``: a desert keeps ``V ~ 0`` - no
    softmax fills it with forest. Weights ``exp(score / tau)`` among PFTs with suitability
    > 0.05 only.
    """
    score = prod
    best = score.max(axis=0)
    v = np.clip(open_frac, 0.0, 1.0) * np.clip(best / v_ref, 0.0, 1.0)
    ok = suit > 0.05
    z = np.where(ok, (score - best[None, :]) / tau, -np.inf)
    e = np.exp(z)
    tot = e.sum(axis=0)
    w = np.where(tot > 0, e / np.where(tot > 0, tot, 1.0), 0.0)
    cover = w * v[None, :]
    return v, cover
