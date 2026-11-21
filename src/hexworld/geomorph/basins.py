"""Continental basins and lowlands from tectonics (TZ 4.6, section 3, package C).

Four families, each a :class:`BasinPlan`-like record with its own geometry and cause, not a
generator of round pits:

* ``rift_graben`` - active continental divergence at the epoch: segmented half-grabens
  along the boundary, the border fault switching sides between segments, overlapping
  segment ends (relay ramps), raised shoulders on the fault side;
* ``strike_slip_pull_apart`` - only at *releasing* steps of a continental strike-slip
  boundary (the boundary opens a little: ``v_n > 0`` while ``|v_t| > 2 v_n``); a restraining
  step (``v_n < 0``) makes none. Narrow, rhomb-shaped, deep, two border faults, two
  sub-basins split by a saddle - the Dead Sea type;
* ``foreland_foredeep`` - flexure of an elastic plate under the mountain load of the epoch:
  ``D lap^2 w + (rho_m - rho_fill) g w = q``, ``D = E Te^3 / 12 (1 - nu^2)``, solved
  spectrally on the world raster [Ф]. Applied only outside the load (inside the mountains
  the root is already in the surface) and only on continental crust: a foredeep along the
  mountain front and a low forebulge beyond it. No mountains, no foredeep;
* ``intracratonic_sag`` - slow, broad subsidence of the platform interior (Williston,
  Michigan, Paris, the Caspian lowland): wide, gentle, filled by sediment to the brim (deep
  bedrock under a plain); one sag may be under-filled and lie below the sea level.

The **bedrock** subsidence and the **sediment fill** are separate: the surface sinks by
``subsidence - fill``. A deep basin can have an almost flat surface. The fill is assumed to
come from long-term denudation around the basin (before the modelled history); its volume
is reported, not taken from the v3/v4 erosion budgets [А].
"""

from __future__ import annotations

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree

from ..rng import Stream

BASIN_TYPES = (
    "none",
    "intracratonic_sag",
    "foreland_foredeep",
    "rift_graben",
    "strike_slip_pull_apart",
)
SAG, FOREDEEP, RIFT, PULL_APART = 1, 2, 3, 4

G = 9.81
RHO_MANTLE = 3300.0
RHO_CRUST = 2700.0
RHO_FILL = 2400.0
YOUNG = 7.0e10
POISSON = 0.25


def flexural_rigidity(te_km: float) -> float:
    te = te_km * 1000.0
    return YOUNG * te**3 / (12.0 * (1.0 - POISSON**2))


def flexure(load_height_m: np.ndarray, cell_km: float, te_km: float, pad: int | None = None):
    """Deflection (m, positive down) of a thin elastic plate under topographic load [Ф].

    ``load_height_m`` is rock above the surroundings; ``q = rho_crust g h``. Solved with the
    FFT on a zero-padded raster (no wrap-around between opposite edges).
    """
    h = np.clip(np.asarray(load_height_m, dtype=np.float64), 0.0, None)
    d = flexural_rigidity(te_km)
    drho = RHO_MANTLE - RHO_FILL
    alpha_km = (4.0 * d / (drho * G)) ** 0.25 / 1000.0
    pad = int(np.ceil(4.0 * alpha_km / cell_km)) if pad is None else pad
    ny, nx = h.shape
    big = np.zeros((ny + 2 * pad, nx + 2 * pad))
    big[pad : pad + ny, pad : pad + nx] = h * RHO_CRUST * G
    ky = np.fft.fftfreq(big.shape[0], d=cell_km * 1000.0) * 2.0 * np.pi
    kx = np.fft.fftfreq(big.shape[1], d=cell_km * 1000.0) * 2.0 * np.pi
    k2 = ky[:, None] ** 2 + kx[None, :] ** 2
    w = np.real(np.fft.ifft2(np.fft.fft2(big) / (d * k2 * k2 + drho * G)))
    return w[pad : pad + ny, pad : pad + nx], alpha_km


def foreland_field(world: dict, params, layout, in_load_hex):
    """Foredeep subsidence (m, + down; negative = forebulge) at the hexes, and alpha (km)."""
    from ..stages.history import HistoryGrid

    g = world["grid"]
    grid = HistoryGrid(g["x0_km"], g["y0_km"], g["cell_km"], g["nx"], g["ny"])
    relief = np.asarray(world.get("orogenic_relief_m", np.zeros(grid.shape)), dtype=np.float64)
    w, alpha = flexure(relief, g["cell_km"], params.elastic_thickness_km)
    hx, hy = layout.centers()
    w_hex = grid.bilinear(w, hx.ravel(), hy.ravel()).reshape(hx.shape)
    load_hex = grid.bilinear((relief > 150.0).astype(np.float64), hx.ravel(), hy.ravel())
    outside = (load_hex.reshape(hx.shape) < 0.5) & ~in_load_hex
    return np.where(outside, w_hex, 0.0), alpha


def _edges(layout, plate, cont_ok):
    """Hex edges between two plates on continental crust: (i, j, midpoint, normal)."""
    from ..stages.raster import neighbor_index

    hx, hy = (a.ravel() for a in layout.centers())
    p = plate.ravel()
    ok = cont_ok.ravel()
    src, dst = [], []
    for d in range(3):  # half of the six directions: every edge once
        nb, valid = neighbor_index(layout, d)
        nb = nb.ravel()
        hit = valid.ravel() & (p[nb] != p) & ok & ok[nb]
        idx = np.nonzero(hit)[0]
        src.append(idx)
        dst.append(nb[idx])
    i = np.concatenate(src)
    j = np.concatenate(dst)
    mx, my = 0.5 * (hx[i] + hx[j]), 0.5 * (hy[i] + hy[j])
    nx, ny = hx[j] - hx[i], hy[j] - hy[i]
    norm = np.maximum(np.hypot(nx, ny), 1e-9)
    return i, j, mx, my, nx / norm, ny / norm


def _clusters(x, y, link_km):
    n = x.size
    if n == 0:
        return 0, np.zeros(0, dtype=np.int64)
    pairs = cKDTree(np.column_stack([x, y])).query_pairs(link_km, output_type="ndarray")
    graph = coo_matrix(
        (np.ones(len(pairs)), (pairs[:, 0], pairs[:, 1])) if len(pairs) else ([], ([], [])),
        shape=(n, n),
    )
    return connected_components(graph, directed=False)


def _axis(x, y):
    """Centre, unit axis direction and half-length of a cloud of points (PCA)."""
    cx, cy = float(np.mean(x)), float(np.mean(y))
    if x.size < 2:
        return cx, cy, 1.0, 0.0, 0.0
    cov = np.cov(np.vstack([x - cx, y - cy]))
    vals, vecs = np.linalg.eigh(cov)
    ax, ay = vecs[:, -1]
    t = (x - cx) * ax + (y - cy) * ay
    return cx, cy, float(ax), float(ay), float(0.5 * (t.max() - t.min()))


def plan_boundary_basins(layout, plate, cont_ok, motion, seed: str, spacing_km: float):
    """Rift grabens and pull-apart basins on the continental boundaries of a hex grid."""
    i, j, mx, my, nx, ny = _edges(layout, plate, cont_ok)
    if i.size == 0:
        return []
    pi, pj = plate.ravel()[i], plate.ravel()[j]
    return _plan_from_edges(mx, my, nx, ny, pi, pj, motion, seed, spacing_km)


def plan_boundary_basins_world(world: dict, motion, seed: str, continental_above_m=-1200.0):
    """The same on the world raster of the epoch (window independent).

    Continental crust is taken where the epoch surface lies above ``continental_above_m``
    (shelves at most a few hundred metres deep, oceanic floor below -2500 m).
    """
    g = world["grid"]
    cell = float(g["cell_km"])
    plate = np.asarray(world["plate_id"])
    z = np.asarray(world["elevation_m"])
    ny_, nx_ = plate.shape
    xs = g["x0_km"] + cell * np.arange(nx_)
    ys = g["y0_km"] + cell * np.arange(ny_)
    cont = z > continental_above_m
    parts = []
    for dr, dc in ((0, 1), (1, 0)):
        a = (slice(0, ny_ - dr), slice(0, nx_ - dc))
        b = (slice(dr, ny_), slice(dc, nx_))
        hit = (plate[a] != plate[b]) & cont[a] & cont[b]
        r, c = np.nonzero(hit)
        parts.append(
            (
                xs[c] + 0.5 * dc * cell,
                ys[r] + 0.5 * dr * cell,
                np.full(r.size, float(dc)),
                np.full(r.size, float(dr)),
                plate[a][r, c],
                plate[b][r, c],
            )
        )
    mx, my, nx, ny, pi, pj = (np.concatenate([p[k] for p in parts]) for k in range(6))
    if mx.size == 0:
        return []
    return _plan_from_edges(
        mx, my, nx, ny, pi.astype(np.int64), pj.astype(np.int64), motion, seed, cell
    )


def _plan_from_edges(mx, my, nx, ny, pi, pj, motion, seed: str, spacing_km: float):
    # hex edges point in six directions only: take the boundary trend over ~200 km of the
    # same plate pair (PCA of the edge midpoints), oriented from plate i towards plate j
    lo, hi = np.minimum(pi, pj), np.maximum(pi, pj)
    tree = cKDTree(np.column_stack([mx, my]))
    groups = tree.query_ball_point(np.column_stack([mx, my]), r=max(200.0, 2.5 * spacing_km))
    snx, sny = nx.copy(), ny.copy()
    for e, nbh in enumerate(groups):
        nbh = np.asarray(nbh)
        nbh = nbh[(lo[nbh] == lo[e]) & (hi[nbh] == hi[e])]
        if nbh.size < 3:
            continue
        _cx, _cy, ax, ay, _h = _axis(mx[nbh], my[nbh])
        cand_x, cand_y = -ay, ax
        sign = 1.0 if cand_x * nx[e] + cand_y * ny[e] >= 0.0 else -1.0
        snx[e], sny[e] = sign * cand_x, sign * cand_y
    nx, ny = snx, sny
    vax, vay = motion.velocity_at(pi, mx, my)
    vbx, vby = motion.velocity_at(pj, mx, my)
    rx, ry = vbx - vax, vby - vay
    vn = rx * nx + ry * ny  # > 0: the plates move apart across the edge
    vt = -rx * ny + ry * nx
    rift = vn > np.maximum(3.0, 0.7 * np.abs(vt))
    releasing = (vn > 1.0) & (np.abs(vt) > 2.0 * vn) & (np.abs(vt) > 8.0)
    stream = Stream(seed, "geomorph_v4_6/basins")
    plans = []
    for kind, sel, link in ((RIFT, rift, 1.6), (PULL_APART, releasing, 1.6)):
        idx = np.nonzero(sel)[0]
        count, lab = _clusters(mx[idx], my[idx], link * spacing_km)
        for c in range(count):
            mem = idx[lab == c]
            if mem.size < (2 if kind == RIFT else 2):
                continue
            # the opening must hold over the whole step, not on one edge
            if kind == PULL_APART and float(np.mean(vn[mem])) < 1.0:
                continue
            cx, cy, ax, ay, half = _axis(mx[mem], my[mem])
            rs = stream.child(f"{BASIN_TYPES[kind]}/{int(round(cx))}/{int(round(cy))}")
            u = rs.random(8)
            open_rate = float(np.mean(vn[mem]))
            slip = float(np.mean(np.abs(vt[mem])))
            if kind == RIFT:
                plans.append(
                    {
                        "basin_type": BASIN_TYPES[RIFT],
                        "center_km": [cx, cy],
                        "axis": [ax, ay],
                        "half_length_km": half + 0.5 * spacing_km,
                        "segment_length_km": 70.0 + 90.0 * u[0],
                        "half_width_km": 18.0 + 17.0 * u[1],
                        "subsidence_m": float(np.clip(1200.0 + 60.0 * open_rate, 1200.0, 4000.0))
                        * (0.8 + 0.4 * u[2]),
                        "fill_fraction": 0.45 + 0.3 * u[3],
                        "flank_uplift_fraction": 0.18 + 0.12 * u[4],
                        # border-fault side of each segment (+1 / -1), drawn from the seed
                        "polarity": [1 if v < 0.5 else -1 for v in rs.child("polarity").random(64)],
                        "opening_mm_yr": round(open_rate, 3),
                        "slip_mm_yr": round(slip, 3),
                        "cause": "continental divergence at the epoch (normal faults)",
                    }
                )
            else:
                ratio = open_rate / max(slip, 1e-6)
                plans.append(
                    {
                        "basin_type": BASIN_TYPES[PULL_APART],
                        "center_km": [cx, cy],
                        "axis": [ax, ay],
                        "half_length_km": float(np.clip(half + 40.0, 35.0, 130.0)),
                        "half_width_km": 10.0 + 10.0 * u[1],
                        "subsidence_m": float(np.clip(2500.0 + 9000.0 * ratio, 2500.0, 7000.0))
                        * (0.85 + 0.3 * u[2]),
                        "fill_fraction": 0.55 + 0.2 * u[3],
                        "sub_basins": 2 if u[4] < 0.6 else 1,
                        "opening_mm_yr": round(open_rate, 3),
                        "slip_mm_yr": round(slip, 3),
                        "cause": "releasing step of a continental strike-slip boundary",
                    }
                )
    return plans


def plan_sags(continentality, x, y, boundary_distance_km, seed: str, density: float, cell_km):
    """Intracratonic sag basins in the platform interiors (material frame, t = 0)."""
    stream = Stream(seed, "geomorph_v4_6/sags")
    ok = (continentality > 550.0) & (boundary_distance_km > 500.0)
    area = float(ok.sum()) * cell_km**2
    if area <= 0 or density <= 0:
        return []
    expected = density * area / 1.0e7
    count = int(np.floor(expected + stream.random()))
    cand = np.column_stack([x[ok], y[ok]])
    order = stream.permutation(cand.shape[0])
    plans = []
    for k in order:
        if len(plans) >= count:
            break
        px, py = cand[k]
        if any(np.hypot(px - q["center_km"][0], py - q["center_km"][1]) < 1300.0 for q in plans):
            continue
        rs = stream.child(f"sag/{len(plans)}")
        u = rs.random(6)
        caspian = len(plans) == 0 and u[5] < 0.6  # the first sag may be under-filled
        w = rs.child("outline").random(4)
        subsidence = 1200.0 + 2800.0 * u[2]
        # a sag is normally filled to the brim (Williston, Michigan, Paris: a deep sediment
        # column under a plain, no closed bowl); only the under-filled one is a depression
        lowering = (180.0 + 260.0 * u[3]) if caspian else 0.0
        plans.append(
            {
                "basin_type": BASIN_TYPES[SAG],
                "center_km": [float(px), float(py)],
                "radius_km": 350.0 + 500.0 * u[0],
                "aspect": 1.0 + 0.9 * u[1],
                "angle_rad": float(np.pi * u[4]),
                "lobes": [
                    float(0.12 + 0.18 * w[0]),
                    float(2 * np.pi * w[1]),
                    float(0.08 + 0.14 * w[2]),
                    float(2 * np.pi * w[3]),
                ],
                "subsidence_m": subsidence,
                "fill_fraction": 1.0 - lowering / subsidence,
                "under_filled": bool(caspian),
                "cause": "slow thermal / structural subsidence of the platform interior",
                "frame": "material (moves with the plate)",
            }
        )
    return plans


def _smooth_box(x, half, edge):
    """1 inside |x| < half, smooth fall to 0 over ``edge`` (km)."""
    t = np.clip((half + edge - np.abs(x)) / (2.0 * edge), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def evaluate(plans, hx, hy, mx, my, radius_km: float = 0.0):
    """Bedrock subsidence S (m, + down), fill (m), flank uplift (m) and basin index.

    With ``radius_km`` (the hex radius) the narrow fault-bounded basins are averaged over
    the hex footprint (7 points), so a 20 km pull-apart is a 20 km basin at any hex size.
    """
    if radius_km > 0.0:
        ring = [(0.0, 0.0)] + [
            (0.55 * radius_km * np.cos(a), 0.55 * radius_km * np.sin(a))
            for a in np.arange(6) * np.pi / 3.0
        ]
        acc = None
        for ox, oy in ring:
            part = _evaluate(plans, hx + ox, hy + oy, mx + ox, my + oy)
            acc = (
                [a.astype(np.float64) for a in part[:3]]
                if acc is None
                else [a + b for a, b in zip(acc, part[:3], strict=True)]
            )
        sub, fill, flank = (a / len(ring) for a in acc)
        which = _evaluate(plans, hx, hy, mx, my)[3]
        which = np.where(sub > 30.0, which, -1)
        return sub, fill, flank, which
    return _evaluate(plans, hx, hy, mx, my)


def _reach_km(p) -> float:
    if p["basin_type"] == BASIN_TYPES[SAG]:
        return 2.6 * p["radius_km"] * max(p["aspect"], 1.0) * 1.5
    return p["half_length_km"] + 4.0 * p["half_width_km"] + 120.0


def _evaluate(plans, hx, hy, mx, my):
    n = hx.size
    best = np.zeros(n)
    fill = np.zeros(n)
    flank = np.zeros(n)
    which = np.full(n, -1, dtype=np.int32)
    for b, p in enumerate(plans):
        kind = p["basin_type"]
        cx, cy = p["center_km"]
        reach = _reach_km(p)
        px, py = (mx, my) if kind == BASIN_TYPES[SAG] else (hx, hy)
        sel = np.nonzero((np.abs(px - cx) < reach) & (np.abs(py - cy) < reach))[0]
        if sel.size == 0:
            continue
        if kind == BASIN_TYPES[SAG]:
            dx, dy = mx[sel] - cx, my[sel] - cy
            c, s = np.cos(p["angle_rad"]), np.sin(p["angle_rad"])
            a = (dx * c + dy * s) / (p["radius_km"] * p["aspect"])
            r = (-dx * s + dy * c) / p["radius_km"]
            # irregular outline: the radius is warped by a smooth function of the direction
            ang = np.arctan2(r, a)
            lobes = p.get("lobes", [0.0, 0.0, 0.0, 0.0])
            warp = (
                1.0 + lobes[0] * np.cos(2 * ang + lobes[1]) + lobes[2] * np.cos(3 * ang + lobes[3])
            )
            rr = np.sqrt(a * a + r * r) / np.maximum(warp, 0.4)
            sub = p["subsidence_m"] * np.exp(-(rr**2.6))
            up = np.zeros(sel.size)
        else:
            ax, ay = p["axis"]
            dx, dy = hx[sel] - cx, hy[sel] - cy
            t = dx * ax + dy * ay  # along the boundary
            d = -dx * ay + dy * ax  # across
            w = p["half_width_km"]
            if kind == BASIN_TYPES[RIFT]:
                seg = p["segment_length_km"]
                k = np.floor((t + p["half_length_km"]) / seg)
                local = t + p["half_length_km"] - (k + 0.5) * seg
                flips = np.asarray(p["polarity"], dtype=np.float64)
                pol = flips[np.clip(k.astype(np.int64), 0, flips.size - 1)]
                offset = 0.35 * w * pol  # en-echelon: segments step sideways
                dd = d - offset
                fault_side = dd * pol > 0.0
                cross = np.where(
                    fault_side,
                    np.exp(-((np.abs(dd) / w) ** 4)),  # steep border fault
                    np.exp(-((np.abs(dd) / (1.8 * w)) ** 1.5)),  # gentle hinge
                )
                along = _smooth_box(local, 0.5 * seg, 18.0) * _smooth_box(
                    t, p["half_length_km"], 30.0
                )
                # relay ramps: neighbouring segments overlap where the boxes fade
                sub = p["subsidence_m"] * cross * np.maximum(along, 0.0)
                shoulder = p["flank_uplift_fraction"] * p["subsidence_m"]
                up = shoulder * np.exp(-(((np.abs(dd) - (w + 22.0)) / 22.0) ** 2))
                up *= np.where(fault_side, 1.0, 0.35) * _smooth_box(t, p["half_length_km"], 40.0)
            else:  # pull-apart: rhomb between two border faults, sub-basins along the axis
                half = p["half_length_km"]
                cross = np.exp(-((np.abs(d) / w) ** 4))
                along = np.exp(-((np.abs(t) / half) ** 4))
                if p.get("sub_basins", 1) == 2:
                    along *= 1.0 - 0.45 * np.exp(-((t / (0.18 * half)) ** 2))  # central saddle
                sub = p["subsidence_m"] * cross * along
                up = 0.08 * p["subsidence_m"] * np.exp(-(((np.abs(d) - (w + 15.0)) / 15.0) ** 2))
                up *= along
        take = sub > best[sel]
        idx = sel[take]
        best[idx] = sub[take]
        fill[idx] = sub[take] * p["fill_fraction"]
        which[idx[sub[take] > 30.0]] = b
        flank[sel] = np.maximum(flank[sel], up)
    return best, fill, flank, which
