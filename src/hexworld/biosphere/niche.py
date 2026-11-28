"""Niche responses of TZ 5.5 §6: trapezoids, the soft combination, seasonal survival and the
breeding window. Suitability is a response score in [0, 1], never a probability.

    trapezoid(x; a, b, c, d) = 0 below a and above d, 1 on [b, c], linear on the flanks
    S = E · exp( Σ w_i ln max(s_i, ε) / Σ w_i )            (E = product of hard gates)
    s_ann = exp( − Σ_m μ_m Δt_m ),  μ_m = μ_max (1 − s_m)  (dormancy lowers μ in bad months)

A missing input (NaN) gives NaN, not 0 and not 1: «unknown» stays unknown (B09).
"""

from __future__ import annotations

import numpy as np

EPS = 1e-6
#: monthly mortality hazard of an active organism in a fully unsuitable month, 1/yr [К]
MU_MAX = 12.0
#: the same for a dormant (hibernating / aestivating / resting) organism [К]
MU_DORMANT = 0.6


def trapezoid(x, a: float, b: float, c: float, d: float):
    """Piecewise-linear response; degenerate flanks (a == b or c == d) are steps that keep
    the plateau closed: x == b counts as inside (B15)."""
    x = np.asarray(x, dtype=np.float64)
    out = np.zeros_like(x)
    with np.errstate(invalid="ignore", divide="ignore"):
        up = (x - a) / (b - a) if b > a else np.where(x >= b, 1.0, 0.0)
        dn = (d - x) / (d - c) if d > c else np.where(x <= c, 1.0, 0.0)
        out = np.clip(np.minimum(up, dn), 0.0, 1.0)
    out = np.where((x >= b) & (x <= c), 1.0, out)
    out = np.where((x < a) | (x > d), 0.0, out)
    return np.where(np.isnan(x), np.nan, out)


def combine(scores: list, weights: list | None = None, gate=None):
    """Weighted geometric mean of the component scores times the hard gate (N03).
    A NaN component makes the result NaN (not guessed)."""
    if not scores:
        raise ValueError("no components")
    w = np.ones(len(scores)) if weights is None else np.asarray(weights, dtype=np.float64)
    acc = 0.0
    for wi, s in zip(w, scores, strict=True):
        acc = acc + wi * np.log(np.maximum(np.asarray(s, dtype=np.float64), EPS))
    out = np.exp(acc / w.sum())
    # a component that is exactly 0 excludes (strict hard exclusion, B15)
    for s in scores:
        out = np.where(np.asarray(s) <= 0.0, 0.0, out)
    if gate is not None:
        out = out * np.asarray(gate, dtype=np.float64)
    return out


def survival(s_month, days, dormant: bool = False, dormant_mask=None):
    """Annual survival from 12 monthly suitabilities (axis 0), weighted by the month lengths
    (B16): months of different duration count by their days."""
    s = np.asarray(s_month, dtype=np.float64)
    dt = np.asarray(days, dtype=np.float64).reshape((-1,) + (1,) * (s.ndim - 1)) / 365.25
    mu = MU_MAX * (1.0 - np.nan_to_num(s, nan=0.0))
    if dormant:
        dm = (s < 0.5) if dormant_mask is None else dormant_mask
        mu = np.where(dm, np.minimum(mu, MU_DORMANT), mu)
    return np.exp(-np.sum(mu * dt, axis=0))


def longest_run(active):
    """Longest run of consecutive active months on a circular year (Dec–Jan wraps, B18).
    ``active`` is boolean (12, ...)."""
    a = np.asarray(active, dtype=bool)
    twice = np.concatenate([a, a], axis=0)
    run = np.zeros(a.shape[1:], dtype=np.int16)
    best = np.zeros(a.shape[1:], dtype=np.int16)
    for m in range(twice.shape[0]):
        run = np.where(twice[m], run + 1, 0)
        best = np.maximum(best, run)
    return np.minimum(best, 12)


def breeding_ok(s_month, months_needed: int, threshold: float = 0.5):
    """A breeding window exists when the suitable months form one continuous run long enough
    (a fragmented set of the same total length does not count)."""
    return longest_run(np.asarray(s_month) >= threshold) >= int(months_needed)


def mirror_months(arr):
    """Shift a monthly series by six months (the other hemisphere's calendar, B17)."""
    return np.roll(np.asarray(arr), 6, axis=0)
