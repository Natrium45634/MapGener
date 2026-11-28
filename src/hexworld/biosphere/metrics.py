"""Biodiversity metrics of TZ 5.5 §11 (G5). Raw values keep their units and denominators;
an index is a separate, versioned display transform.

* taxon richness = number of unique leaf taxa present (never the number of groups, B10)
* functional richness = number of groups present
* expected richness = Σ p (occupancy-weighted), reported separately
* α = mean local richness, γ = |union|, β = γ / α (Whittaker), Jaccard turnover (N15)
* Shannon H = −Σ p ln p, Hill numbers q0, q1 = e^H, q2 = 1/Σp², Pielou J = H / ln S (N16)
* species–area prior S = c (A_eff / A_ref)^z — a statistical prior, never added to the
  realized count (N17, B60)
* index = clip((raw − lo) / (hi − lo)) · 100 with fixed, versioned bounds (N18, B62)
* percentile = area-weighted CDF over the whole parent world, not the viewport (B63)
"""

from __future__ import annotations

import numpy as np

INDEX_VERSION = "provisional_v1"
#: fixed raw bounds of the 0–100 indices (provisional, not calibrated)
INDEX_BOUNDS = {
    "flora_richness": (0.0, 80.0),
    "fauna_richness": (0.0, 140.0),
    "total_richness": (0.0, 200.0),
    "npp_gc_m2_yr": (0.0, 1500.0),
    "biomass_kgc_m2": (0.0, 25.0),
    "weighted_endemism": (0.0, 2.0),
}


def alpha_beta_gamma(sets: list) -> dict:
    """Whittaker partition over equal-area units; repeated taxa counted once in γ (B58)."""
    if not sets:
        return {"alpha": 0.0, "gamma": 0, "beta": float("nan"), "turnover": float("nan")}
    sets = [set(s) for s in sets]
    gamma = len(set().union(*sets))
    alpha = float(np.mean([len(s) for s in sets]))
    beta = gamma / alpha if alpha > 0 else float("nan")
    turn = float("nan")
    if len(sets) == 2:
        u = sets[0] | sets[1]
        turn = 1.0 - len(sets[0] & sets[1]) / len(u) if u else float("nan")
    return {"alpha": alpha, "gamma": gamma, "beta": beta, "turnover": turn}


def jaccard_turnover(a: set, b: set) -> float:
    u = set(a) | set(b)
    return 1.0 - len(set(a) & set(b)) / len(u) if u else float("nan")


def hill(abundances) -> dict:
    """Shannon, Hill q0/q1/q2 and Pielou of one abundance vector (zero entries ignored)."""
    a = np.asarray([x for x in np.ravel(abundances) if x > 0], dtype=np.float64)
    s = a.size
    if s == 0:
        return {"S": 0, "H": 0.0, "q0": 0, "q1": 0.0, "q2": 0.0, "J": float("nan")}
    p = a / a.sum()
    h = float(-(p * np.log(p)).sum())
    return {"S": int(s), "H": h, "q0": int(s), "q1": float(np.exp(h)),
            "q2": float(1.0 / (p**2).sum()), "J": h / np.log(s) if s > 1 else float("nan")}


def shannon_rows(ab):
    """Shannon H of every column of an abundance matrix (taxa × hexes)."""
    a = np.asarray(ab, dtype=np.float64)
    tot = a.sum(axis=0)
    with np.errstate(invalid="ignore", divide="ignore"):
        p = np.where(tot > 0, a / np.where(tot > 0, tot, 1.0), 0.0)
        h = -np.sum(np.where(p > 0, p * np.log(np.where(p > 0, p, 1.0)), 0.0), axis=0)
    return h


def species_area_prior(c: float, a_eff_km2, a_ref_km2: float, z: float, f: float = 1.0,
                       cap: float | None = None):
    s = c * f * (np.asarray(a_eff_km2, dtype=np.float64) / a_ref_km2) ** z
    return np.minimum(s, cap) if cap is not None else s


def index(raw, lo: float, hi: float):
    """Linear 0–100 with clipping; NaN (missing component) stays NaN (B62)."""
    r = np.asarray(raw, dtype=np.float64)
    return np.clip((r - lo) / (hi - lo), 0.0, 1.0) * 100.0


def percentile_area_weighted(values, area, valid=None):
    """Area-weighted CDF rank of every value in the whole world (0–100), NaN where invalid."""
    v = np.asarray(values, dtype=np.float64).ravel()
    w = np.broadcast_to(np.asarray(area, dtype=np.float64), v.shape).ravel()
    ok = np.isfinite(v) if valid is None else (np.asarray(valid).ravel() & np.isfinite(v))
    out = np.full(v.shape, np.nan)
    if not ok.any():
        return out
    order = np.argsort(v[ok], kind="stable")
    vs = v[ok][order]
    cw = np.cumsum(w[ok][order])
    tot = cw[-1]
    # mid-rank of ties: the share of the area strictly below plus half the tie
    lo = np.searchsorted(vs, v[ok], side="left")
    hi = np.searchsorted(vs, v[ok], side="right")
    below = np.where(lo > 0, cw[np.maximum(lo - 1, 0)], 0.0)
    tie = cw[hi - 1] - below
    out[ok] = 100.0 * (below + 0.5 * tie) / tot
    return out
