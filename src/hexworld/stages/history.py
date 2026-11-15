"""v3 stage ``geo_history``: a forward, deterministic history of the tectonic world.

Time runs from the baseline snapshot (``t = 0``, the v1/v2 world) to
``t = duration_ma``. Nothing of the baseline is edited: the history produces new
state on top of it.

Units: horizontal km, time Ma, plate speed mm/yr (**1 mm/yr = 1 km/Ma**),
heights and thicknesses m, vertical rates m/Ma (1 mm/yr = 1000 m/Ma).

Numerical scheme
----------------
* The world is a fixed raster in the mantle frame (the v2 skeleton grid).
* Plates are rigid: a plate's footprint at time ``t`` is its baseline footprint
  moved by the exact rigid transform of its velocity field. Geometry therefore
  never drifts numerically and a collision cannot be "jumped over": the overlap
  of two footprints is evaluated exactly at every step.
* Where footprints **overlap**, plates converge: continental over continental is
  a collision, anything over oceanic is subduction (the more buoyant material
  overrides; between two oceanic plates the younger crust overrides).
* Where the moved footprints leave a **gap**, plates diverge: new crust forms.
  Its age is the time since the cell was first uncovered; narrow gaps between
  continental margins are rifts, not oceans.
* Everything that belongs to rock (uplift, erosion, sediment, orogen ages) is
  accumulated in the **material frame** of the plates - on the baseline raster -
  so mountain belts ride with their plate instead of smearing over the mantle
  frame. Rates are computed in the mantle frame each step and sampled at the
  current position of every material cell.
* Steps are sub-divided so that no plate moves more than one cell per sub-step.

Limits (honest): plates do not deform, split or merge; their velocities are
constant; a collision does not slow them down. Topology changes are future work.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
from scipy import ndimage
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import spsolve_triangular

from ..config import HistoryParams

MM_YR_TO_KM_MA = 1.0  # 1 mm/yr == 1 km/Ma
MM_YR_TO_M_MA = 1000.0  # 1 mm/yr == 1000 m/Ma

INTENSITY = {"calm": 0.6, "earthlike": 1.0, "active": 1.5}
KIND_NONE, KIND_COLLISION, KIND_ARC, KIND_INHERITED, KIND_RIFT = 0, 1, 2, 3, 4
KIND_NAMES = ("none", "collision", "arc", "inherited", "rift")
RIFT_HALF_WIDTH_KM = 130.0
_H_CAP_M = 5200.0


def _gauss(d, width):
    return np.exp(-0.5 * (d / width) ** 2)


@dataclass
class HistoryGrid:
    x0: float
    y0: float
    cell: float
    nx: int
    ny: int

    @property
    def shape(self):
        return (self.ny, self.nx)

    def centers(self):
        xs = self.x0 + np.arange(self.nx) * self.cell
        ys = self.y0 + np.arange(self.ny) * self.cell
        return np.meshgrid(xs, ys)

    def index(self, x, y):
        """Fractional (row, col) of world point(s)."""
        return (np.asarray(y) - self.y0) / self.cell, (np.asarray(x) - self.x0) / self.cell

    def nearest(self, x, y):
        r, c = self.index(x, y)
        ri = np.rint(r).astype(np.int64)
        ci = np.rint(c).astype(np.int64)
        ok = (ri >= 0) & (ri < self.ny) & (ci >= 0) & (ci < self.nx)
        return np.clip(ri, 0, self.ny - 1), np.clip(ci, 0, self.nx - 1), ok

    def bilinear(self, arr, x, y):
        r, c = self.index(x, y)
        coords = np.vstack([np.ravel(r), np.ravel(c)])
        out = ndimage.map_coordinates(arr, coords, order=1, mode="nearest")
        return out.reshape(np.shape(x))

    def to_dict(self):
        return {
            "x0_km": self.x0,
            "y0_km": self.y0,
            "cell_km": self.cell,
            "nx": self.nx,
            "ny": self.ny,
        }

    @classmethod
    def from_dict(cls, d):
        return cls(d["x0_km"], d["y0_km"], d["cell_km"], int(d["nx"]), int(d["ny"]))


class Kinematics:
    """Exact rigid motion of every plate for a constant velocity field.

    ``v(p) = v0 + w * J (p - pole)`` (``J`` turns +x towards +y). For ``w != 0``
    this is a rotation about the fixed point where ``v = 0``; otherwise a
    translation. ``forward(pid, x, y, t)`` moves baseline positions to time ``t``.
    """

    def __init__(self, velocity, omega_deg, pole):
        self.v = np.asarray(velocity, dtype=np.float64) * MM_YR_TO_KM_MA  # km/Ma
        self.w = np.deg2rad(np.asarray(omega_deg, dtype=np.float64))  # rad/Ma
        self.pole = np.asarray(pole, dtype=np.float64)
        rot = np.abs(self.w) > 1e-9
        safe = np.where(rot, self.w, 1.0)
        self.rot = rot
        self.cx = np.where(rot, self.pole[:, 0] - self.v[:, 1] / safe, 0.0)
        self.cy = np.where(rot, self.pole[:, 1] + self.v[:, 0] / safe, 0.0)

    @property
    def count(self):
        return self.v.shape[0]

    def forward(self, pid, x, y, t):
        pid = np.asarray(pid)
        x = np.asarray(x, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        a = self.w[pid] * t
        ca, sa = np.cos(a), np.sin(a)
        dx = x - self.cx[pid]
        dy = y - self.cy[pid]
        rx = self.cx[pid] + ca * dx - sa * dy
        ry = self.cy[pid] + sa * dx + ca * dy
        rot = self.rot[pid]
        return (
            np.where(rot, rx, x + self.v[pid, 0] * t),
            np.where(rot, ry, y + self.v[pid, 1] * t),
        )

    def inverse(self, pid, x, y, t):
        return self.forward(pid, x, y, -t)

    def velocity_at(self, pid, x, y):
        """Velocity (km/Ma == mm/yr) of plate(s) ``pid`` at mantle-frame point(s)."""
        pid = np.asarray(pid)
        rx = np.asarray(x) - self.pole[pid, 0]
        ry = np.asarray(y) - self.pole[pid, 1]
        return self.v[pid, 0] - self.w[pid] * ry, self.v[pid, 1] + self.w[pid] * rx

    def max_speed(self, grid: HistoryGrid, plate0) -> float:
        gx, gy = grid.centers()
        vx, vy = self.velocity_at(plate0, gx, gy)
        return float(np.hypot(vx, vy).max())


@dataclass
class Baseline:
    """Immutable inputs of the history: the v1/v2 world on the history raster."""

    grid: HistoryGrid
    plate: np.ndarray  # (ny, nx) int32, baseline plate of every cell
    continental: np.ndarray  # (ny, nx) float, continental fraction 0..1
    elevation: np.ndarray  # (ny, nx) float, baseline macro elevation, m
    inherited_relief: np.ndarray  # (ny, nx) float >= 0, orogenic relief already present, m
    crust_age: np.ndarray  # (ny, nx) float, Ma at t = 0
    craton: np.ndarray  # (ny, nx) float 0..1, share of old crystalline basement
    #: Age (Ma before t = 0) of the inherited relief; None = HistoryParams default.
    inherited_age: np.ndarray | None = None
    #: Elevation of ancient belts that the v2 macro relief does not contain, m (>= 0).
    extra_elevation: np.ndarray | None = None

    def __post_init__(self):
        if self.extra_elevation is None:
            self.extra_elevation = np.zeros(self.grid.shape)


@dataclass
class Snapshot:
    """Mantle-frame state of the world at one time (a ``GeologicalSnapshot``)."""

    time_ma: float
    owner: np.ndarray  # int32, plate owning the cell (gaps are filled with the nearest plate)
    covered: np.ndarray  # bool, False where the cell is new crust in a gap
    material: np.ndarray  # int64, flat baseline index of the rock now at the cell (-1 in gaps)
    under: np.ndarray  # int32, plate beneath (-1 if none)
    overlap_kind: np.ndarray  # uint8: 0 none, 1 collision, 2 subduction
    rift: np.ndarray  # bool, gap that is still a continental rift
    new_crust_age: np.ndarray  # float32 Ma, NaN outside gaps
    trench_distance: np.ndarray  # float32 km to the nearest trench cell (inf if none)
    trench_over: np.ndarray  # int32 overriding plate of that trench
    trench_under: np.ndarray  # int32
    trench_rate: np.ndarray  # float32 mm/yr
    collision_distance: np.ndarray  # float32 km
    collision_rate: np.ndarray  # float32 mm/yr
    gap_distance: np.ndarray  # float32 km from covered cells to the nearest gap
    gap_half_width: np.ndarray  # float32 km, local half-width of the gap
    ridge: np.ndarray  # bool, spreading axis cells
    uplift_rate: np.ndarray  # float32 m/Ma, tectonic vertical rate in the mantle frame
    arc_zone: np.ndarray  # float32 0..1 volcanic arc intensity
    pairs: dict = field(default_factory=dict)


@dataclass
class MaterialState:
    """Accumulated history of the rock, on the baseline raster (material frame)."""

    uplift: np.ndarray  # m, net tectonic vertical displacement
    orogenic: np.ndarray  # m, mountain-building part of the uplift (>= 0)
    erosion: np.ndarray  # m removed (>= 0)
    sediment: np.ndarray  # m deposited (>= 0)
    orogen_first: np.ndarray  # Ma, time of the first mountain-building (NaN = never)
    orogen_last: np.ndarray  # Ma, time of the last mountain-building
    orogen_kind: np.ndarray  # uint8 KIND_*
    last_tectonic: np.ndarray  # Ma, time of the last significant tectonic change
    volcanic: np.ndarray  # m, accumulated volcanic arc material
    gap_open: np.ndarray  # Ma, mantle-frame raster: when the cell was first uncovered (NaN)
    gap_sediment: np.ndarray  # m, mantle-frame raster: sediment on new crust

    FIELDS = (
        "uplift",
        "orogenic",
        "erosion",
        "sediment",
        "orogen_first",
        "orogen_last",
        "orogen_kind",
        "last_tectonic",
        "volcanic",
        "gap_open",
        "gap_sediment",
    )

    def copy(self) -> MaterialState:
        return MaterialState(**{k: getattr(self, k).copy() for k in self.FIELDS})

    def arrays(self) -> dict:
        return {k: getattr(self, k) for k in self.FIELDS}


def initial_state(base: Baseline, params: HistoryParams) -> MaterialState:
    shape = base.grid.shape
    nan = np.full(shape, np.nan, dtype=np.float32)
    inherited = base.inherited_relief > 300.0
    kind = np.where(inherited, KIND_INHERITED, KIND_NONE).astype(np.uint8)
    age = params.inherited_orogen_age_ma if base.inherited_age is None else base.inherited_age
    old = np.where(inherited, -np.asarray(age, dtype=np.float64), np.nan).astype(np.float32)
    zero = np.zeros(shape, dtype=np.float32)
    return MaterialState(
        uplift=zero.copy(),
        orogenic=zero.copy(),
        erosion=zero.copy(),
        sediment=zero.copy(),
        orogen_first=old.copy(),
        orogen_last=old.copy(),
        orogen_kind=kind,
        last_tectonic=old.copy(),
        volcanic=zero.copy(),
        gap_open=nan.copy(),
        gap_sediment=zero.copy(),
    )


class HistorySimulator:
    def __init__(
        self, base: Baseline, kin: Kinematics, params: HistoryParams, seed: str, focus=None
    ):
        """``focus`` = (xmin, ymin, xmax, ymax): where events and centres are recorded.

        The physics runs on the whole raster (plates continue outside the window);
        only the bookkeeping of events is limited to the area of interest.
        """
        self.base = base
        self.kin = kin
        self.params = params
        self.seed = seed
        self.grid = base.grid
        self.gx, self.gy = base.grid.centers()
        if focus is None:
            self.focus = np.ones(base.grid.shape, dtype=bool)
        else:
            self.focus = (
                (self.gx >= focus[0])
                & (self.gx <= focus[2])
                & (self.gy >= focus[1])
                & (self.gy <= focus[3])
            )
        self.plate_ids = np.unique(base.plate)
        self._boxes = {}
        for j in self.plate_ids:
            rows, cols = np.nonzero(base.plate == j)
            self._boxes[int(j)] = (rows.min(), rows.max(), cols.min(), cols.max())
        self.intensity = INTENSITY[params.intensity]
        # buoyancy priority of the rock: continental overrides oceanic; among oceanic
        # crust the younger overrides the older (which is denser and subducts)
        self.priority0 = base.continental - 0.25 * (1.0 - base.continental) * np.clip(
            base.crust_age / 180.0, 0.0, 1.0
        )

    # -- geometry at time t -------------------------------------------------
    def coverage(self, t: float):
        """Which plates cover every mantle-frame cell at time ``t`` (best and second)."""
        g = self.grid
        shape = g.shape
        best = np.full(shape, -1, dtype=np.int32)
        second = np.full(shape, -1, dtype=np.int32)
        best_p = np.full(shape, -np.inf)
        second_p = np.full(shape, -np.inf)
        best_q = np.full(shape, -1, dtype=np.int64)
        second_q = np.full(shape, -1, dtype=np.int64)
        plate0 = self.base.plate
        for j in self.plate_ids:
            j = int(j)
            r0, r1, c0, c1 = self._boxes[j]
            cx = g.x0 + np.array([c0, c1, c0, c1]) * g.cell
            cy = g.y0 + np.array([r0, r0, r1, r1]) * g.cell
            fx, fy = self.kin.forward(np.full(4, j), cx, cy, t)
            a0 = max(0, int(math.floor((fy.min() - g.y0) / g.cell)) - 2)
            a1 = min(g.ny - 1, int(math.ceil((fy.max() - g.y0) / g.cell)) + 2)
            b0 = max(0, int(math.floor((fx.min() - g.x0) / g.cell)) - 2)
            b1 = min(g.nx - 1, int(math.ceil((fx.max() - g.x0) / g.cell)) + 2)
            # A plate that reaches the edge of the raster continues beyond it: the
            # raster is a window, so no artificial ocean may open along its rim.
            if r0 == 0:
                a0 = 0
            if r1 == g.ny - 1:
                a1 = g.ny - 1
            if c0 == 0:
                b0 = 0
            if c1 == g.nx - 1:
                b1 = g.nx - 1
            if a1 < a0 or b1 < b0:
                continue
            sl = (slice(a0, a1 + 1), slice(b0, b1 + 1))
            px, py = self.gx[sl], self.gy[sl]
            qx, qy = self.kin.inverse(np.full(px.shape, j), px, py, t)
            ri, ci, _inside = g.nearest(qx, qy)  # clamped: edge cells extend outwards
            cov = plate0[ri, ci] == j
            if not cov.any():
                continue
            flat = ri * g.nx + ci
            prio = self.priority0[ri, ci] - 1e-7 * j
            bp, sp = best_p[sl], second_p[sl]
            better = cov & (prio > bp)
            mid = cov & ~better & (prio > sp)
            # demote the previous best where the new plate is better
            second[sl] = np.where(better, best[sl], np.where(mid, j, second[sl]))
            second_q[sl] = np.where(better, best_q[sl], np.where(mid, flat, second_q[sl]))
            second_p[sl] = np.where(better, bp, np.where(mid, prio, sp))
            best[sl] = np.where(better, j, best[sl])
            best_q[sl] = np.where(better, flat, best_q[sl])
            best_p[sl] = np.where(better, prio, bp)
        return best, best_q, second, second_q

    def snapshot(self, t: float, state: MaterialState) -> Snapshot:
        g = self.grid
        base = self.base
        owner, material, under, under_q = self.coverage(t)
        covered = owner >= 0
        cont_flat = base.continental.ravel()
        cont_over = np.where(covered, cont_flat[np.maximum(material, 0)], 0.0)
        cont_under = np.where(under >= 0, cont_flat[np.maximum(under_q, 0)], 0.0)
        overlap = under >= 0
        kind = np.zeros(g.shape, dtype=np.uint8)
        kind[overlap & (cont_over > 0.5) & (cont_under > 0.5)] = 1
        kind[overlap & (kind == 0)] = 2

        # relative speed of the two plates where they overlap, mm/yr
        rate = np.zeros(g.shape, dtype=np.float32)
        if overlap.any():
            ox, oy = self.kin.velocity_at(owner[overlap], self.gx[overlap], self.gy[overlap])
            ux, uy = self.kin.velocity_at(under[overlap], self.gx[overlap], self.gy[overlap])
            rate[overlap] = np.hypot(ox - ux, oy - uy)

        gap = ~covered
        owner_filled = owner.copy()
        gap_dist_in = np.zeros(g.shape, dtype=np.float32)
        new_age = np.full(g.shape, np.nan, dtype=np.float32)
        rift = np.zeros(g.shape, dtype=bool)
        ridge = np.zeros(g.shape, dtype=bool)
        half_width = np.zeros(g.shape, dtype=np.float32)
        if gap.any() and covered.any():
            d_in, inds = ndimage.distance_transform_edt(gap, return_indices=True)
            owner_filled[gap] = owner[inds[0][gap], inds[1][gap]]
            gap_dist_in = (d_in * g.cell).astype(np.float32)
            size = max(3, int(round(2.0 * RIFT_HALF_WIDTH_KM / g.cell)) | 1)
            half_width = ndimage.maximum_filter(gap_dist_in, size=size + 2)
            margin_cont = cont_over[inds[0], inds[1]]
            rift = gap & (half_width < RIFT_HALF_WIDTH_KM) & (margin_cont > 0.5)
            for shift, axis in ((1, 0), (1, 1)):
                other = np.roll(owner_filled, -shift, axis=axis)
                both = gap & np.roll(gap, -shift, axis=axis) & (other != owner_filled)
                if axis == 0:
                    both[-1, :] = False
                else:
                    both[:, -1] = False
                ridge |= both
            # Age of new crust: zero on the spreading axis, growing to the time since
            # the opening began at the old plate margin (symmetric accretion).
            gap_labels, n_gaps = ndimage.label(gap)
            opened = np.where(np.isnan(state.gap_open), t, state.gap_open)
            began = ndimage.minimum(opened, gap_labels, np.arange(1, n_gaps + 1))
            span = np.clip(t - np.asarray(began)[np.maximum(gap_labels, 1) - 1], 0.0, None)
            if ridge.any():
                d_axis = ndimage.distance_transform_edt(~ridge) * g.cell
                share = d_axis / np.maximum(d_axis + gap_dist_in, 1e-6)
            else:
                share = np.ones(g.shape)
            new_age[gap] = (span * share)[gap]
        gap_dist_out = (
            (ndimage.distance_transform_edt(covered) * g.cell).astype(np.float32)
            if gap.any()
            else np.full(g.shape, np.inf, dtype=np.float32)
        )

        # trench cells: subduction overlap cells that touch the exposed lower plate
        sub = kind == 2
        trench = np.zeros(g.shape, dtype=bool)
        for shift, axis in ((1, 0), (-1, 0), (1, 1), (-1, 1)):
            nb_owner = np.roll(owner, shift, axis=axis)
            nb_under = np.roll(under, shift, axis=axis)
            trench |= sub & (nb_owner == under) & (nb_under != owner)
        trench[0, :] = trench[-1, :] = trench[:, 0] = trench[:, -1] = False
        inf = np.full(g.shape, np.inf, dtype=np.float32)
        t_dist, t_over, t_under = (
            inf.copy(),
            np.full(g.shape, -1, np.int32),
            np.full(g.shape, -1, np.int32),
        )
        t_rate = np.zeros(g.shape, dtype=np.float32)
        t_arc = np.full(g.shape, 200.0, dtype=np.float32)
        if trench.any():
            d, inds = ndimage.distance_transform_edt(~trench, return_indices=True)
            t_dist = (d * g.cell).astype(np.float32)
            t_over = owner[inds[0], inds[1]]
            t_under = under[inds[0], inds[1]]
            t_rate = rate[inds[0], inds[1]]
            # Where the arc stands: above the point where the slab reaches the depth of
            # magma generation (~105 km). Old, dense crust sinks steeply (arc close to the
            # trench); young crust and fast convergence give a flat slab (arc far inland).
            slab_age = np.clip(base.crust_age.ravel()[np.maximum(under_q, 0)] + t, 0.0, 170.0)
            dip = np.clip(26.0 + 0.22 * slab_age - 0.10 * rate, 18.0, 62.0)
            arc_km = np.clip(105.0 / np.tan(np.deg2rad(dip)), 95.0, 320.0)
            t_arc = arc_km[inds[0], inds[1]].astype(np.float32)
        col = kind == 1
        c_dist, c_rate = inf.copy(), np.zeros(g.shape, dtype=np.float32)
        if col.any():
            d, inds = ndimage.distance_transform_edt(~col, return_indices=True)
            c_dist = (d * g.cell).astype(np.float32)
            c_rate = rate[inds[0], inds[1]]

        snap = Snapshot(
            time_ma=float(t),
            owner=owner_filled,
            covered=covered,
            material=np.where(covered, material, -1),
            under=under,
            overlap_kind=kind,
            rift=rift,
            new_crust_age=new_age,
            trench_distance=t_dist,
            trench_over=t_over,
            trench_under=t_under,
            trench_rate=t_rate,
            collision_distance=c_dist,
            collision_rate=c_rate,
            gap_distance=gap_dist_out,
            gap_half_width=half_width,
            ridge=ridge,
            uplift_rate=np.zeros(g.shape, dtype=np.float32),
            arc_zone=np.zeros(g.shape, dtype=np.float32),
        )
        snap.continental = cont_over
        snap.trench_arc_km = t_arc
        self._rates(snap, cont_over)
        snap.pairs = self._pairs(snap, rate)
        return snap

    def _rates(self, snap: Snapshot, cont_over) -> None:
        """Tectonic vertical rates (m/Ma) in the mantle frame for one snapshot."""
        p = self.params
        k = self.intensity
        up = p.uplift_scale
        down = p.subsidence_scale
        s_col = np.clip(snap.collision_rate / 50.0, 0.0, 2.0)
        collision = 260.0 * k * up * s_col * _gauss(snap.collision_distance, 240.0)
        collision = np.where(snap.covered & (cont_over > 0.3), collision, 0.0)

        s_sub = np.clip(snap.trench_rate / 50.0, 0.0, 2.0)
        over_side = snap.covered & (snap.owner == snap.trench_over)
        d = snap.trench_distance
        arc_km = snap.trench_arc_km
        # the volcanic line is narrow; the uplifted belt around it is broad
        arc_profile = _gauss(d - arc_km, 40.0)
        belt = _gauss(d - arc_km - 30.0, 115.0)
        arc = (60.0 + 80.0 * cont_over) * k * up * s_sub * belt
        back_arc = -28.0 * k * down * s_sub * _gauss(d - arc_km - 430.0, 210.0)
        subduction = np.where(over_side, arc + back_arc, 0.0)
        snap.arc_zone = np.where(over_side, np.clip(s_sub, 0, 1) * arc_profile, 0.0).astype(
            np.float32
        )

        gd = snap.gap_distance
        young = np.isfinite(gd)
        shoulders = np.where(
            snap.covered & young & (cont_over > 0.5),
            k * (34.0 * up * _gauss(gd - 150.0, 80.0) - 70.0 * down * _gauss(gd, 70.0)),
            0.0,
        )
        snap.collision_part = collision.astype(np.float32)
        snap.arc_part = np.where(over_side, arc, 0.0).astype(np.float32)
        snap.uplift_rate = (collision + subduction + shoulders).astype(np.float32)

    def _pairs(self, snap: Snapshot, rate) -> dict:
        """Per plate-pair interaction measures for the event log."""
        n = self.kin.count
        out: dict = {}
        for code, name in ((1, "collision"), (2, "subduction")):
            m = (snap.overlap_kind == code) & self.focus
            if not m.any():
                continue
            a = snap.owner[m].astype(np.int64)
            b = snap.under[m].astype(np.int64)
            keys = a * n + b
            for key in np.unique(keys):
                sel = keys == key
                out[(name, int(key // n), int(key % n))] = {
                    "cells": int(sel.sum()),
                    "x": float(self.gx[m][sel].mean()),
                    "y": float(self.gy[m][sel].mean()),
                    "rate": float(rate[m][sel].mean()),
                }
        if (snap.ridge & self.focus).any():
            m = snap.ridge & self.focus
            a = snap.owner[m].astype(np.int64)
            for shift, axis in ((1, 0), (1, 1)):
                other = np.roll(snap.owner, -shift, axis=axis)[m].astype(np.int64)
                diff = other != a
                lo = np.minimum(a[diff], other[diff])
                hi = np.maximum(a[diff], other[diff])
                keys = lo * n + hi
                for key in np.unique(keys):
                    sel = keys == key
                    item = out.setdefault(
                        ("rifting", int(key // n), int(key % n)),
                        {"cells": 0, "x": 0.0, "y": 0.0, "rate": 0.0, "width": 0.0},
                    )
                    cells = int(sel.sum())
                    total = item["cells"] + cells
                    item["x"] = (
                        item["x"] * item["cells"] + float(self.gx[m][diff][sel].sum())
                    ) / total
                    item["y"] = (
                        item["y"] * item["cells"] + float(self.gy[m][diff][sel].sum())
                    ) / total
                    item["width"] = max(
                        item["width"], 2.0 * float(snap.gap_half_width[m][diff][sel].max())
                    )
                    item["cells"] = total
        return out

    # -- composite surface --------------------------------------------------
    def new_crust_elevation(self, snap: Snapshot, state: MaterialState) -> np.ndarray:
        """Elevation of cells in gaps: rift floor or ocean floor that deepens with age."""
        age = np.nan_to_num(snap.new_crust_age, nan=0.0)
        ocean = -2600.0 - 350.0 * np.sqrt(np.clip(age, 0.0, 120.0))
        stretch = np.clip(snap.gap_half_width / RIFT_HALF_WIDTH_KM, 0.0, 1.0)
        rift_floor = -150.0 - 1500.0 * stretch
        floor = np.where(snap.rift, rift_floor, ocean)
        return floor + state.gap_sediment

    def trench_relief(self, snap: Snapshot) -> np.ndarray:
        """Instantaneous trench and outer rise on the subducting side (mantle frame), m."""
        s = np.clip(snap.trench_rate / 50.0, 0.0, 2.0) ** 0.5
        width = max(85.0, 1.1 * self.grid.cell)
        d = snap.trench_distance
        under_side = snap.covered & (snap.owner == snap.trench_under)
        near = snap.overlap_kind == 2
        relief = np.where(
            under_side, -2600.0 * s * _gauss(d, width) + 220.0 * s * _gauss(d - 280.0, 120.0), 0.0
        )
        relief = np.where(near & (d < width), -1200.0 * s * _gauss(d, width), relief)
        return relief * self.intensity**0.5

    def surface(self, snap: Snapshot, state: MaterialState) -> np.ndarray:
        """Macro elevation of the whole raster at the snapshot time, m."""
        mat = np.maximum(snap.material, 0)
        base = self.base
        rock = (
            base.elevation.ravel()[mat]
            + base.extra_elevation.ravel()[mat]
            + state.uplift.ravel()[mat]
            - state.erosion.ravel()[mat]
            + state.sediment.ravel()[mat]
            + state.volcanic.ravel()[mat]
        )
        out = np.where(snap.covered, rock, self.new_crust_elevation(snap, state))
        return out + self.trench_relief(snap)

    # -- erosion ------------------------------------------------------------
    def resistance(self, snap: Snapshot, state: MaterialState) -> np.ndarray:
        """Erosion resistance 0..1 of the rock now at every mantle-frame cell."""
        mat = np.maximum(snap.material, 0)
        b = self.base
        cont = b.continental.ravel()[mat]
        craton = b.craton.ravel()[mat]
        orog = (b.inherited_relief.ravel()[mat] + state.orogenic.ravel()[mat]) > 400.0
        res = np.where(cont > 0.5, 0.42 + 0.42 * craton, 0.62)
        res = np.where(orog, np.maximum(res, 0.7), res)
        res = np.where(state.volcanic.ravel()[mat] > 150.0, 0.6, res)
        res = np.where(state.sediment.ravel()[mat] > 150.0, 0.22, res)
        return np.where(snap.covered, res, 0.3)

    def erode(self, elevation, resistance, dt: float):
        """Hillslope + runoff erosion of land for ``dt`` Ma.

        Returns ``(removed, deposited, exported_volume_m_cells)``: both arrays are
        thicknesses in m (>= 0). Land is everything above the 0 m model datum, so
        the history does not depend on the sea-level scenario.
        """
        amount = self.params.erosion_intensity
        zeros = np.zeros_like(elevation)
        if amount <= 0.0 or dt <= 0.0:
            return zeros, zeros, 0.0
        g = self.grid
        h = elevation
        land = h > 0.0
        if not land.any():
            return zeros, zeros, 0.0
        ny, nx = h.shape
        n = h.size
        # steepest-descent receiver among the 8 neighbours
        pad = np.pad(h, 1, mode="edge")
        best_drop = np.zeros_like(h)
        recv = np.arange(n).reshape(h.shape)
        index = np.arange(n).reshape(h.shape)
        for dr, dc in ((-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)):
            nb = pad[1 + dr : 1 + dr + ny, 1 + dc : 1 + dc + nx]
            dist = g.cell * math.hypot(dr, dc)
            slope = (h - nb) / dist
            rr = np.clip(np.arange(ny)[:, None] + dr, 0, ny - 1)
            cc = np.clip(np.arange(nx)[None, :] + dc, 0, nx - 1)
            better = slope > best_drop
            best_drop = np.where(better, slope, best_drop)
            recv = np.where(better, rr * nx + cc, recv)
        slope = best_drop  # m per km, >= 0
        # drainage area by a triangular solve in order of decreasing height:
        # every donor precedes its receiver, so (I - M) is lower triangular
        order = np.argsort(-h.ravel(), kind="stable")
        rank = np.empty(n, dtype=np.int64)
        rank[order] = np.arange(n)
        donors = index.ravel()
        has = recv.ravel() != donors
        rows = rank[recv.ravel()[has]]
        cols = rank[donors[has]]
        lower = csr_matrix(
            (
                np.concatenate([np.ones(n), -np.ones(rows.size)]),
                (np.concatenate([np.arange(n), rows]), np.concatenate([np.arange(n), cols])),
            ),
            shape=(n, n),
        )
        runoff = np.where(land, 1.0, 0.0).ravel()[order]
        area = np.empty(n)
        area[order] = spsolve_triangular(lower, runoff, lower=True)
        area = area.reshape(h.shape)

        soft = 1.0 / np.clip(resistance, 0.12, 1.0)
        fluvial = 0.55 * amount * soft * np.sqrt(area) * slope  # m/Ma
        lap = ndimage.laplace(h, mode="nearest") / g.cell**2  # m/km^2
        hillslope = np.clip(-1100.0 * amount * soft * lap, 0.0, None)  # removes convexities
        removed = np.where(land, (fluvial + hillslope) * dt, 0.0)
        # never cut below the receiver, never below the datum, never more than there is
        drop = slope * g.cell
        removed = np.minimum(removed, 0.45 * drop + 0.02 * np.maximum(h, 0.0))
        removed = np.clip(np.minimum(removed, np.maximum(h, 0.0)), 0.0, None)

        # sediment flux through every cell (same triangular system), then deposition
        # where the flow loses its capacity: gentle land, and the sea next to the coast
        flux = np.empty(n)
        flux[order] = spsolve_triangular(lower, removed.ravel()[order], lower=True)
        flux = flux.reshape(h.shape)
        gentle = 1.0 / (1.0 + (slope / 1.5) ** 2)
        coast = ndimage.binary_dilation(land, iterations=3) & ~land
        weight = flux * (np.where(land, 0.25 * gentle, 0.0) + np.where(coast, 1.0, 0.0))
        total = float(removed.sum())
        wsum = float(weight.sum())
        kept = 0.85  # the rest leaves as dissolved / far-travelled load (reported)
        deposited = weight * (kept * total / wsum) if wsum > 0 else zeros
        deposited = np.minimum(deposited, 0.6 * removed.max() + 400.0 * dt)
        exported = total - float(deposited.sum())
        return removed, deposited, exported

    # -- one step -----------------------------------------------------------
    def advance(self, state: MaterialState, t0: float, dt: float, log: list) -> Snapshot:
        """Advance the material state from ``t0`` to ``t0 + dt``; returns the snapshot at t1."""
        t1 = t0 + dt
        g = self.grid
        snap = self.snapshot(t1, state)
        # gaps: remember when each cell was first uncovered (mantle frame)
        gap = ~snap.covered
        state.gap_open[gap & np.isnan(state.gap_open)] = t0 + 0.5 * dt
        state.gap_open[~gap] = np.nan
        state.gap_sediment[~gap] = 0.0

        surface = self.surface(snap, state)
        # isostatic ceiling: uplift fades as the tectonic relief approaches its limit
        mat = np.maximum(snap.material, 0)
        relief_now = (
            self.base.inherited_relief.ravel()[mat]
            + state.orogenic.ravel()[mat]
            - state.erosion.ravel()[mat]
        )
        headroom = np.clip(1.0 - np.maximum(relief_now, 0.0) / _H_CAP_M, 0.0, 1.0)
        rate = np.where(snap.uplift_rate > 0, snap.uplift_rate * headroom, snap.uplift_rate)
        building = (snap.collision_part + snap.arc_part) * headroom
        removed, deposited, exported = self.erode(surface, self.resistance(snap, state), dt)

        # sample the mantle-frame rates at the current position of every material cell
        px, py = self.kin.forward(self.base.plate, self.gx, self.gy, t1)
        ri, ci, ok = g.nearest(px, py)
        alive = ok & (snap.owner[ri, ci] == self.base.plate) & snap.covered[ri, ci]

        def at_material(field_):
            return np.where(alive, g.bilinear(field_, px, py), 0.0).astype(np.float32)

        d_up = at_material(rate) * dt
        d_build = at_material(building) * dt
        state.uplift += d_up
        state.orogenic += np.clip(d_build, 0.0, None)
        state.erosion += at_material(removed)
        state.sediment += at_material(deposited)
        state.volcanic += at_material(snap.arc_part * 0.35) * dt
        state.gap_sediment += np.where(gap, deposited, 0.0).astype(np.float32)
        strong = d_build > 20.0 * dt
        kind_here = np.where(
            at_material(snap.collision_part) >= at_material(snap.arc_part), KIND_COLLISION, KIND_ARC
        ).astype(np.uint8)
        state.orogen_kind = np.where(strong, kind_here, state.orogen_kind)
        state.orogen_first = np.where(
            strong & np.isnan(state.orogen_first), t1, state.orogen_first
        ).astype(np.float32)
        state.orogen_last = np.where(strong, t1, state.orogen_last).astype(np.float32)
        active = np.abs(d_up) > 5.0 * dt
        state.last_tectonic = np.where(active, t1, state.last_tectonic).astype(np.float32)
        snap.uplift_rate = rate.astype(np.float32)
        log.append(
            {
                "time_ma": round(t1, 6),
                "eroded_m_cells": float(removed.sum()),
                "deposited_m_cells": float(deposited.sum()),
                "exported_m_cells": float(exported),
            }
        )
        return snap


def run_history(sim: HistorySimulator, progress=None, observer=None):
    """Run the whole history. Returns ``(checkpoints, events, balance_log, step_info)``.

    ``checkpoints`` is a list of ``(time_ma, MaterialState)`` including the initial
    and the final state.
    """
    p = sim.params
    duration = p.duration_ma if p.enabled else 0.0
    state = initial_state(sim.base, p)
    checkpoints = [(0.0, state.copy())]
    events: list[dict] = []
    balance: list[dict] = []
    info = {"steps": 0, "substeps": 0, "max_speed_mm_yr": 0.0}
    if duration <= 0.0:
        return checkpoints, events, balance, info
    vmax = sim.kin.max_speed(sim.grid, sim.base.plate)
    n_steps = max(1, int(math.ceil(duration / p.time_step_ma - 1e-9)))
    dt = duration / n_steps
    sub = max(1, int(math.ceil(vmax * dt / sim.grid.cell)))
    info.update(steps=n_steps, substeps=sub, max_speed_mm_yr=vmax, dt_ma=dt, substep_ma=dt / sub)
    active: dict = {}
    counters: dict = {}
    t = 0.0
    for step in range(n_steps):
        if progress is not None:
            progress(step / n_steps)
        for _ in range(sub):
            snap = sim.advance(state, t, dt / sub, balance)
            t += dt / sub
            if observer is not None:
                observer(snap, t)
        _log_events(snap, active, counters, events, t)
        if (step + 1) % p.checkpoint_stride == 0 and step + 1 < n_steps:
            checkpoints.append((round(t, 6), state.copy()))
    for _key, item in sorted(active.items()):
        item["active_at_end"] = True
    checkpoints.append((round(t, 6), state.copy()))
    events.sort(key=lambda e: (e["time_ma"], e["id"]))
    return checkpoints, events, balance, info


def _log_events(snap: Snapshot, active: dict, counters: dict, events: list, t: float) -> None:
    """Start / end events when a plate pair starts or stops interacting."""
    minimum = 6
    seen = set()
    for key, m in sorted(snap.pairs.items()):
        kind, a, b = key
        if m["cells"] < minimum:
            continue
        seen.add(key)
        if key not in active:
            n = counters.get(key, 0)
            counters[key] = n + 1
            event = {
                "id": f"{kind}:{a}-{b}:{n}",
                "type": f"{kind}_start",
                "time_ma": round(t, 6),
                "x_km": round(m["x"], 1),
                "y_km": round(m["y"], 1),
                "plates": [a, b],
                "roles": {
                    "collision": ["upper", "lower"],
                    "subduction": ["overriding", "subducting"],
                    "rifting": ["side", "side"],
                }[kind],
                "intensity_mm_yr": round(m["rate"], 3),
                "cells": m["cells"],
                "end_time_ma": None,
            }
            active[key] = event
            events.append(event)
        else:
            ev = active[key]
            ev["cells"] = max(ev["cells"], m["cells"])
            ev["x_km"], ev["y_km"] = round(m["x"], 1), round(m["y"], 1)
        if kind == "rifting" and m.get("width", 0.0) > 2.2 * RIFT_HALF_WIDTH_KM:
            ocean_key = ("ocean", a, b)
            if ocean_key not in counters:
                counters[ocean_key] = 1
                events.append(
                    {
                        "id": f"ocean_opening:{a}-{b}:0",
                        "type": "ocean_opening",
                        "time_ma": round(t, 6),
                        "x_km": round(m["x"], 1),
                        "y_km": round(m["y"], 1),
                        "plates": [a, b],
                        "roles": ["side", "side"],
                        "intensity_mm_yr": 0.0,
                        "cells": m["cells"],
                        "end_time_ma": None,
                    }
                )
    # an interaction ends only when it is completely gone (not when it merely thins)
    for key in [k for k in active if k not in snap.pairs]:
        active.pop(key)["end_time_ma"] = round(t, 6)
