"""Age of the oceanic crust from spreading ridges, and the depth of the sea floor from age.

Ridge segments and fracture zones
---------------------------------
A spreading axis lies at right angles to the spreading direction; where a plate boundary
runs obliquely, the real axis breaks into short orthogonal segments joined by transform
faults (a staircase), and every transform leaves a fracture zone - a line parallel to the
spreading direction - on both plates. Here each divergent run of the world skeleton is cut
into **ridge pieces**:

* the spreading direction ``s`` is the mean direction of the relative plate velocity;
* the run is projected on the ridge-parallel axis ``t`` (perpendicular to ``s``) and cut
  into pieces 40-400 km long (seeded, log-normal); the axis of a piece sits at the median
  ``s``-coordinate of its boundary samples, plus a small seeded offset - so an oblique
  boundary becomes a staircase, an orthogonal one still has the second-order offsets of
  real ridges;
* every piece builds crust on the two plates it separates, at its own half-rate.

Crust at a point of plate ``P`` was made by the piece of a ridge of ``P`` that lies
upstream along the spreading direction: the piece whose ``t``-range contains the point
(a lateral overshoot costs 4 km per km), with the smallest distance along ``s``. Its age is
that distance over the half-rate. Where the chosen piece changes, the age jumps by the
offset over the half-rate - a fracture zone, parallel to the spreading direction, with the
step in depth the age-depth relation gives it. [A] engineering adaptation, flagged
``age_estimation_mode = approx`` (TZ 4.6, 4.2.5).

Depth from age [Ф]
------------------
Parsons & Sclater (1977): ``d = 2500 + 350 sqrt(t)`` for young crust and
``d = 6400 - 3200 exp(-t / 62.8)`` for old crust (m, t in Ma), joined with a smoothstep
between 60 and 80 Ma (TZ 4.6, 4.3). The constants are the reference of an Earth-like world,
not universal values; they are parameters.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree

from ..rng import Stream
from ..stages.boundaries import TYPE_DIVERGENT

AGE_MODE = "approx"
MAX_AGE_MA = 180.0
LATERAL_COST = 4.0


def _smoothstep(x):
    x = np.clip(x, 0.0, 1.0)
    return x * x * (3.0 - 2.0 * x)


def parsons_sclater_depth(
    age_ma,
    ridge_depth_m: float = 2500.0,
    young_k: float = 350.0,
    deep_m: float = 6400.0,
    deep_k: float = 3200.0,
    tau_ma: float = 62.8,
    blend_ma: tuple[float, float] = (60.0, 80.0),
) -> np.ndarray:
    """Depth below the datum (m, positive down) of crust of ``age_ma``, PS77 hybrid."""
    t = np.clip(np.asarray(age_ma, dtype=np.float64), 0.0, None)
    young = ridge_depth_m + young_k * np.sqrt(t)
    old = deep_m - deep_k * np.exp(-t / tau_ma)
    lo, hi = blend_ma
    s = _smoothstep((t - lo) / max(hi - lo, 1e-9))
    return young * (1.0 - s) + old * s


@dataclass
class RidgePieces:
    """Orthogonal ridge segments; every piece is listed once per flank plate."""

    plate: np.ndarray  # (P,) int32 plate whose crust the row builds
    ax: np.ndarray  # a point on the axis, km
    ay: np.ndarray
    sx: np.ndarray  # unit spreading direction pointing from the axis into ``plate``
    sy: np.ndarray
    t0: np.ndarray  # range of the piece along t = (-sy, sx) measured from (ax, ay)
    t1: np.ndarray
    half_rate: np.ndarray  # km/Ma (= mm/yr)
    piece: np.ndarray  # id of the physical piece (shared by its two rows)
    run: np.ndarray  # id of the boundary run the piece belongs to

    @property
    def count(self) -> int:
        return int(np.unique(self.piece).size) if self.piece.size else 0

    def to_dict(self) -> dict:
        return {k: np.asarray(getattr(self, k)).tolist() for k in self.__dataclass_fields__}


def build_ridge_pieces(skeleton, seed: str, mean_length_km: float = 200.0) -> RidgePieces:
    """Cut the divergent boundaries of ``skeleton`` into orthogonal ridge pieces."""
    sk = skeleton
    div = np.nonzero(sk.kind == TYPE_DIVERGENT)[0]
    empty = np.zeros(0)
    if div.size == 0:
        z = empty.astype(np.int32)
        return RidgePieces(z, empty, empty, empty, empty, empty, empty, empty, z, z)
    bx, by = sk.bx[div], sk.by[div]
    rx = sk.vn[div] * sk.nx[div] - sk.vt[div] * sk.ny[div]
    ry = sk.vn[div] * sk.ny[div] + sk.vt[div] * sk.nx[div]
    pa, pb = sk.plate_a[div], sk.plate_b[div]
    lo, hi = np.minimum(pa, pb), np.maximum(pa, pb)

    # runs: samples of one plate pair joined within 1.6 cells
    pairs = cKDTree(np.column_stack([bx, by])).query_pairs(1.6 * sk.cell_km, output_type="ndarray")
    if pairs.size:
        same = (lo[pairs[:, 0]] == lo[pairs[:, 1]]) & (hi[pairs[:, 0]] == hi[pairs[:, 1]])
        pairs = pairs[same]
    n = div.size
    graph = coo_matrix(
        (np.ones(len(pairs)), (pairs[:, 0], pairs[:, 1])) if len(pairs) else ([], ([], [])),
        shape=(n, n),
    )
    n_runs, run_of = connected_components(graph, directed=False)

    stream = Stream(seed, "geomorph_v4_6/ridge_pieces")
    rows: dict[str, list] = {k: [] for k in RidgePieces.__dataclass_fields__}
    piece_id = 0
    for r in range(n_runs):
        idx = np.nonzero(run_of == r)[0]
        if idx.size < 2:
            continue
        # spreading direction: relative velocity of the higher-id plate w.r.t. the lower
        sign = np.where(pb[idx] == hi[idx], 1.0, -1.0)
        mx, my = (sign * rx[idx]).sum(), (sign * ry[idx]).sum()
        norm = np.hypot(mx, my)
        if norm < 1e-9:
            continue
        sx, sy = mx / norm, my / norm  # crust of plate ``hi`` moves along +s
        tx, ty = -sy, sx
        u = bx[idx] * tx + by[idx] * ty
        v = bx[idx] * sx + by[idx] * sy
        speed = np.hypot(rx[idx], ry[idx])
        order = np.argsort(u)
        u, v, speed = u[order], v[order], speed[order]
        rs = stream.child(f"run/{int(lo[idx[0]])}-{int(hi[idx[0]])}/{r}")
        start = u[0] - 0.5 * sk.cell_km
        stop = u[-1] + 0.5 * sk.cell_km
        cuts = [start]
        draws = rs.random(256)
        k = 0
        while cuts[-1] < stop and k < 255:
            length = float(
                np.clip(mean_length_km * np.exp(0.6 * (draws[k] * 2.0 - 1.0) * 1.7), 40, 400)
            )
            cuts.append(cuts[-1] + length)
            k += 1
        cuts[-1] = max(cuts[-1], stop)
        offsets = rs.normal(len(cuts))
        for j in range(len(cuts) - 1):
            inside = (u >= cuts[j]) & (u < cuts[j + 1])
            if not inside.any():
                continue
            axis_v = float(np.median(v[inside])) + 12.0 * float(np.clip(offsets[j], -2.5, 2.5))
            half = float(np.clip(0.5 * np.mean(speed[inside]), 5.0, 90.0))
            # axis point at the middle of the piece
            umid = 0.5 * (cuts[j] + cuts[j + 1])
            px, py = umid * tx + axis_v * sx, umid * ty + axis_v * sy
            half_len = 0.5 * (cuts[j + 1] - cuts[j])
            for plate, sgn in ((int(hi[idx[0]]), 1.0), (int(lo[idx[0]]), -1.0)):
                rows["plate"].append(plate)
                rows["ax"].append(px)
                rows["ay"].append(py)
                rows["sx"].append(sgn * sx)
                rows["sy"].append(sgn * sy)
                # t = (-s_y, s_x) flips with s: keep the range symmetric about the middle
                rows["t0"].append(-half_len)
                rows["t1"].append(half_len)
                rows["half_rate"].append(half)
                rows["piece"].append(piece_id)
                rows["run"].append(r)
            piece_id += 1
    return RidgePieces(
        plate=np.asarray(rows["plate"], dtype=np.int32),
        ax=np.asarray(rows["ax"], dtype=np.float64),
        ay=np.asarray(rows["ay"], dtype=np.float64),
        sx=np.asarray(rows["sx"], dtype=np.float64),
        sy=np.asarray(rows["sy"], dtype=np.float64),
        t0=np.asarray(rows["t0"], dtype=np.float64),
        t1=np.asarray(rows["t1"], dtype=np.float64),
        half_rate=np.asarray(rows["half_rate"], dtype=np.float64),
        piece=np.asarray(rows["piece"], dtype=np.int32),
        run=np.asarray(rows["run"], dtype=np.int32),
    )


@dataclass
class FlowlineAge:
    age_ma: np.ndarray  # Ma at the reference time of the pieces
    piece: np.ndarray  # int32 physical piece id, -1 = no ridge on the plate
    along_km: np.ndarray  # distance from the axis along the spreading direction
    tx: np.ndarray  # unit isochron direction (ridge-parallel) at the point
    ty: np.ndarray
    half_rate: np.ndarray  # km/Ma of the piece (NaN without a piece)


def flowline_age(
    pieces: RidgePieces,
    x,
    y,
    plate,
    fallback_age_ma: float = 110.0,
    chunk: int = 40_000,
) -> FlowlineAge:
    """Crust age of points ``(x, y)`` on ``plate`` from the ridge pieces of that plate."""
    x = np.asarray(x, dtype=np.float64).ravel()
    y = np.asarray(y, dtype=np.float64).ravel()
    plate = np.asarray(plate).ravel().astype(np.int64)
    n = x.size
    age = np.full(n, np.nan)
    piece = np.full(n, -1, dtype=np.int32)
    along = np.full(n, np.nan)
    tx = np.zeros(n)
    ty = np.zeros(n)
    rate = np.full(n, np.nan)
    orphan = np.zeros(0, dtype=np.int64)
    if pieces.plate.size:
        groups = [
            (np.nonzero(pieces.plate == p)[0], np.nonzero(plate == p)[0]) for p in np.unique(plate)
        ]
        # plates without a ridge of their own: crust carried across another boundary; its
        # age grows smoothly with the distance to the nearest ridge axis (no piece frame,
        # which would draw straight artificial edges), plus ``fallback_age_ma`` / 4
        orphan = np.concatenate(
            [pts for rows, pts in groups if rows.size == 0] or [np.zeros(0, dtype=np.int64)]
        )
        if orphan.size:
            d, k = cKDTree(np.column_stack([pieces.ax, pieces.ay])).query(
                np.column_stack([x[orphan], y[orphan]])
            )
            age[orphan] = d / np.mean(pieces.half_rate)
            along[orphan] = d
            tx[orphan] = -pieces.sy[k]
            ty[orphan] = pieces.sx[k]
            rate[orphan] = np.mean(pieces.half_rate)
        for rows, pts in groups:
            if rows.size == 0 or pts.size == 0:
                continue
            ax, ay = pieces.ax[rows], pieces.ay[rows]
            sx, sy = pieces.sx[rows], pieces.sy[rows]
            t0, t1 = pieces.t0[rows], pieces.t1[rows]
            hr = pieces.half_rate[rows]
            for c0 in range(0, pts.size, chunk):
                sel = pts[c0 : c0 + chunk]
                dx = x[sel, None] - ax[None, :]
                dy = y[sel, None] - ay[None, :]
                a = dx * sx[None, :] + dy * sy[None, :]
                t = -dx * sy[None, :] + dy * sx[None, :]
                over = np.maximum(t0[None, :] - t, 0.0) + np.maximum(t - t1[None, :], 0.0)
                # crust lies downstream (a >= 0); a point behind the axis is penalised too.
                # Compare in time, not distance: where a plate is built by two ridges the
                # ages meet continuously (an age maximum, not a step)
                metric = (np.abs(a) + LATERAL_COST * over + 2.0 * np.maximum(-a, 0.0)) / hr[None, :]
                best = np.argmin(metric, axis=1)
                r = rows[best]
                k_ = np.arange(sel.size)
                ab = np.abs(a[k_, best])
                along[sel] = ab
                age_piece = ab / pieces.half_rate[r]
                # far beyond the lateral end of every piece the crust was not made by them
                # (their ridges are gone); a linear tie between remote pieces would draw a
                # straight artificial edge there, so blend into a smooth distance-age
                ov = over[k_, best]
                w = np.clip((ov - 100.0) / 300.0, 0.0, 1.0)
                w = w * w * (3.0 - 2.0 * w)
                if (w > 0).any():
                    dist = np.sqrt(dx * dx + dy * dy).min(axis=1)
                    age_iso = dist / np.mean(hr)
                    age_piece = (1.0 - w) * age_piece + w * np.maximum(age_piece, age_iso)
                age[sel] = age_piece
                piece[sel] = pieces.piece[r]
                tx[sel] = -pieces.sy[r]
                ty[sel] = pieces.sx[r]
                rate[sel] = pieces.half_rate[r]
    if orphan.size:
        age[orphan] += 0.25 * fallback_age_ma
        piece[orphan] = -1
    missing = np.isnan(age)
    age[missing] = fallback_age_ma
    return FlowlineAge(
        age_ma=np.clip(age, 0.0, MAX_AGE_MA),
        piece=piece,
        along_km=along,
        tx=tx,
        ty=ty,
        half_rate=rate,
    )


def baseline_age_scale(skeleton, pieces: RidgePieces, continents, target_median_ma: float) -> float:
    """Calibration factor (<= 1) of the baseline crust ages [А].

    The baseline is a world without a past: its crust ages are distances from today's ridges
    over today's half-rates. Model oceans are often wider than the plate speeds can have
    made in 180 Ma, which would leave most of the floor at the saturated old depth. The
    ages are therefore scaled so that the median oceanic age of the whole model area does
    not exceed ``target_median_ma`` (Earth: about 60-65 Ma). Order and fracture-zone steps
    are kept; computed on the world raster, so it does not depend on the window.
    """
    if target_median_ma <= 0 or pieces.plate.size == 0:
        return 1.0
    c = continents.continentality(skeleton.x, skeleton.y)
    oceanic = continents.fraction_of(c) < 0.5
    if not oceanic.any():
        return 1.0
    age = flowline_age(pieces, skeleton.x[oceanic], skeleton.y[oceanic], skeleton.plate[oceanic])
    med = float(np.median(age.age_ma))
    return min(1.0, target_median_ma / med) if med > 0 else 1.0
