"""Intraplate seamount field and island arcs of the sea floor (4.6 realism update).

Seamounts [Ф/А]
---------------
Most of the ~10^5 seamounts higher than 1 km of the real ocean floor are not on hotspot
chains: they are scattered volcanoes built near the ridges or later on the plate (Wessel et
al. 2010, Kim & Wessel 2011: about one >1 km seamount per 10-15 thousand km^2, heights with
a heavy tail). Here they are a Poisson field in the **plate frame** (material coordinates),
so they ride with the plate and every window / resolution sees the same edifices:

* cells of ``CELL_KM`` hold 0-4 seamounts (Poisson, mean 0.45; a seeded integer hash of
  the cell, no RNG state);
* height ``h = 800 u^-0.42`` m, capped at 5.5 km; basal radius ``5-9 h`` (flanks 6-11 deg);
* each formed ``e`` Ma ago (exponential, mean 25 Ma, never before its crust); it subsided
  with the crust since then: ``top_now = top_then - (PS(age) - PS(age - e))``;
* a top that reached the sea surface when built was planed at the wave base (-40 m) and
  sank as a **guyot**; a young one still above the sea is a **volcanic island**.

Island arcs [Ф/А]
-----------------
Where oceanic crust subducts under oceanic crust, the overriding plate carries a volcanic
arc platform 120-200 km from the trench axis (Mariana, Tonga, Aleutian, Lesser Antilles):
thick arc crust standing 1-2 km below the sea, with the arc cones of v3 on top. The platform
is a broad ridge (``ARC_PLATFORM``) whose top varies along strike; with the cones it makes
chains of islands - archipelagos, not one continuous strip of land.
"""

from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree

from ..rng import Stream
from .ocean_age import parsons_sclater_depth

CELL_KM = 110.0
MEAN_PER_CELL = 0.45
WAVE_BASE_M = -40.0
#: distance of the arc axis from the trench, its half-width and the depth of its crest, km / m
ARC_DISTANCE_KM = 165.0
ARC_SIGMA_KM = 48.0
ARC_CREST_M = -1000.0

_M1 = np.uint64(0x9E3779B97F4A7C15)
_M2 = np.uint64(0xBF58476D1CE4E5B9)
_M3 = np.uint64(0x94D049BB133111EB)


def _mix(x):
    with np.errstate(over="ignore"):
        x = x + _M1
        x = (x ^ (x >> np.uint64(30))) * _M2
        x = (x ^ (x >> np.uint64(27))) * _M3
        return x ^ (x >> np.uint64(31))


def _uniform(base: np.uint64, i, j, k, field: int):
    """Uniform [0, 1) per (cell i, cell j, slot k, field): pure function of the seed."""
    with np.errstate(over="ignore"):
        h = _mix(base ^ np.uint64(field * 0x632BE59BD9B4E019 & 0xFFFFFFFFFFFFFFFF))
        h = _mix(
            h ^ (np.asarray(i, dtype=np.int64).astype(np.uint64) * np.uint64(0x8CB92BA72F3D8DD7))
        )
        h = _mix(
            h ^ (np.asarray(j, dtype=np.int64).astype(np.uint64) * np.uint64(0xD6E8FEB86659FD93))
        )
        h = _mix(h ^ np.asarray(k, dtype=np.int64).astype(np.uint64))
    return (h >> np.uint64(11)).astype(np.float64) / float(2**53)


def seamount_catalogue(seed: str, x0, y0, x1, y1, mean_per_cell: float = MEAN_PER_CELL):
    """Seamounts (plate-frame km) of every cell touching the rectangle, as arrays."""
    base = np.uint64(int(Stream(seed, "geomorph/seamount_field").integer(2**62)))
    i0, i1 = int(np.floor(x0 / CELL_KM)) - 1, int(np.floor(x1 / CELL_KM)) + 1
    j0, j1 = int(np.floor(y0 / CELL_KM)) - 1, int(np.floor(y1 / CELL_KM)) + 1
    ii, jj = np.meshgrid(np.arange(i0, i1 + 1), np.arange(j0, j1 + 1), indexing="ij")
    ii, jj = ii.ravel(), jj.ravel()
    # Poisson count by inversion of the cumulative distribution (at most 4 per cell)
    u = _uniform(base, ii, jj, 0, 1)
    lam = mean_per_cell
    p = np.exp(-lam)
    cdf = p
    count = np.zeros(ii.size, dtype=np.int64)
    for k in range(1, 5):
        count += u > cdf
        p *= lam / k
        cdf += p
    rows = []
    for k in range(4):
        sel = count > k
        if not sel.any():
            continue
        ci, cj = ii[sel], jj[sel]
        fx = _uniform(base, ci, cj, k, 2)
        fy = _uniform(base, ci, cj, k, 3)
        fh = _uniform(base, ci, cj, k, 4)
        fr = _uniform(base, ci, cj, k, 5)
        fe = _uniform(base, ci, cj, k, 6)
        h = np.minimum(800.0 * np.maximum(fh, 1e-4) ** -0.42, 5500.0)
        rows.append(
            np.column_stack(
                [
                    (ci + fx) * CELL_KM,
                    (cj + fy) * CELL_KM,
                    h,
                    h / 1000.0 * (5.0 + 4.0 * fr),
                    -25.0 * np.log(np.maximum(1.0 - fe, 1e-9)),  # eruption age, Ma
                ]
            )
        )
    if not rows:
        return np.zeros((0, 5))
    return np.concatenate(rows, axis=0)


def seamount_field(seed, mx, my, age_ma, marine, spacing_km, ridge_depth_m, blend):
    """Relief (m) the seamount field adds to marine hexes, and its classes.

    Returns ``relief``, ``summit_now`` (m vs the datum, NaN off seamounts), ``kind``
    (0 none, 1 seamount, 2 guyot, 3 island), and a summary. Heights are the mean over the
    hex footprint (``r_eff = sqrt(r^2 + (0.6 s)^2)``, volume kept); the summit is reported
    separately like the v3 volcanoes.
    """
    n = mx.size
    relief = np.zeros(n)
    summit = np.full(n, np.nan)
    kind = np.zeros(n, dtype=np.uint8)
    m = np.nonzero(marine)[0]
    summary = {"field_seamounts": 0, "field_guyots": 0, "field_islands": 0}
    if m.size == 0:
        return relief, summit, kind, summary
    pad = 2.5 * float(np.sqrt((5.5 * 9.0) ** 2 + (0.6 * spacing_km) ** 2)) + 10.0
    cat = seamount_catalogue(
        seed, mx[m].min() - pad, my[m].min() - pad, mx[m].max() + pad, my[m].max() + pad
    )
    if cat.shape[0] == 0:
        return relief, summit, kind, summary
    sx, sy, h, r, erupt = cat.T
    r_eff = np.sqrt(r * r + (0.6 * spacing_km) ** 2)
    h_eff = h * (r / r_eff) ** 2
    tree = cKDTree(np.column_stack([sx, sy]))
    # a fixed reach (the largest possible edifice), not the largest of this catalogue: the
    # catalogue depends on the window, the reach must not (window-independent relief)
    r_max = 5.5 * 9.0
    reach = 2.5 * float(np.sqrt(r_max * r_max + (0.6 * spacing_km) ** 2))
    k = min(6, sx.size)
    d, idx = tree.query(np.column_stack([mx[m], my[m]]), k=k, distance_upper_bound=reach)
    if k == 1:
        d, idx = d[:, None], idx[:, None]
    ok = np.isfinite(d)
    idx_c = np.where(ok, idx, 0)
    age_here = np.nan_to_num(age_ma[m], nan=100.0)
    # every seamount sits on crust it erupted through: subsidence since the eruption
    e = np.minimum(erupt[idx_c], age_here[:, None])
    a_then = np.clip(age_here[:, None] - e, 0.0, None)
    ps_now = parsons_sclater_depth(age_here, ridge_depth_m, blend_ma=blend)[:, None]
    ps_then = parsons_sclater_depth(a_then, ridge_depth_m, blend_ma=blend)
    sink = ps_now - ps_then
    top_then = -ps_then + h[idx_c]
    planed = top_then > WAVE_BASE_M
    # a planed top lost the part above the wave base: lower the cone by that much
    cut = np.where(planed & (e > 1.5), top_then - WAVE_BASE_M, 0.0)
    shape = np.exp(-0.5 * (d / r_eff[idx_c]) ** 2)
    shape = np.where(ok, shape, 0.0)
    contrib = (h_eff[idx_c] - cut * (r[idx_c] / r_eff[idx_c]) ** 2) * shape
    relief[m] = np.clip(contrib, 0.0, None).sum(axis=1)
    near = np.argmax(np.where(ok, shape, -1.0), axis=1)
    rows = np.arange(m.size)
    on = shape[rows, near] > 0.6
    top_now = top_then[rows, near] - cut[rows, near] - sink[rows, near]
    summit[m[on]] = top_now[on]
    guy = on & planed[rows, near] & (e[rows, near] > 1.5) & (top_now < -150.0)
    isl = on & (top_now > 0.0)
    kind[m[on]] = 1
    kind[m[guy]] = 2
    kind[m[isl]] = 3
    summary = {
        "field_seamounts": int(np.unique(idx[ok & (shape > 0.6)]).size),
        "field_guyots": int(guy.sum()),
        "field_islands": int(isl.sum()),
        "catalogue_per_10k_km2": round(
            float(
                cat.shape[0]
                * 1e4
                / max((mx[m].max() - mx[m].min() + 120) * (my[m].max() - my[m].min() + 120), 1.0)
            ),
            3,
        ),
    }
    return relief, summit, kind, summary


def arc_platform(z_floor, tdist, side, rate, marine, oceanic_over, along_var):
    """Raise the overriding oceanic plate to the arc crest along the volcanic arc axis.

    ``along_var`` in [-1, 1] (world noise along the arc) moves the crest between -1.9 and
    -0.1 km: gaps and high segments, so the cones on top make island
    chains. Returns the added relief (>= 0).
    """
    active = np.clip(np.asarray(rate, dtype=np.float64) / 25.0, 0.0, 1.0)
    here = marine & (side == -1) & oceanic_over & np.isfinite(tdist)
    crest = ARC_CREST_M + 900.0 * np.asarray(along_var, dtype=np.float64)
    shape = np.exp(
        -0.5 * ((np.asarray(tdist, dtype=np.float64) - ARC_DISTANCE_KM) / ARC_SIGMA_KM) ** 2
    )
    weight = np.where(here, shape * active, 0.0)
    lift = np.clip(crest - z_floor, 0.0, None) * weight
    return lift, weight


def shelf_highs(mx, my, z, shelf, seed):
    """Island highs on shallow continental shelves (drowned hills, horsts, moraines) [А].

    Archipelagos cluster: a regional modulator (1500 km) makes some coasts island-rich
    (Aegean, Dalmatia, the Canadian archipelago) and leaves others plain; the islands
    themselves are 30-180 km hills of the drowned relief. Plate-frame world noise: the same
    in every window. Returns the added relief (>= 0, m).
    """
    from ..noise import FractalNoise

    st = Stream(seed, "geomorph/shelf_highs")
    region = FractalNoise(st.child("region"), [1500.0, 700.0], [1.0, 0.4])
    hills = FractalNoise(st.child("hills"), [180.0, 70.0, 30.0], [1.0, 0.6, 0.35])
    out = np.zeros(np.shape(z))
    idx = np.nonzero(shelf)[0]
    if idx.size == 0:
        return out
    mod = np.clip(region(mx[idx], my[idx]) * 1.3 + 0.15, 0.0, 1.0)
    h = np.clip((hills(mx[idx], my[idx]) - 0.25) / 0.45, 0.0, 1.0) ** 1.5
    out[idx] = 420.0 * mod * h
    return out
