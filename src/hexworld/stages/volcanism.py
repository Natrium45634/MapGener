"""v3: volcanic centres - intraplate hotspots and subduction arcs.

A hotspot is a point fixed in the mantle frame. The plate above it moves, so the
volcanoes it builds are carried away and line up in an age-progressive chain:
each volcano stores the plate it sits on and its position in that plate's
material frame, so its place at any time is exact. Arc volcanoes are spawned on
the overriding plate at arc distance from an active trench.

``volcanic_potential`` (0..1) is a diagnostic of where magmatism is plausible
(active arcs, spreading axes, hotspots). It is not an eruption probability.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from ..rng import Stream

HOTSPOT, ARC = "hotspot", "arc"
#: A centre counts as active while it erupted within this time, Ma.
_EXTINCT_AFTER_MA = 1.5


@dataclass
class Hotspot:
    id: int
    x: float  # mantle frame, km
    y: float
    strength: float
    interval_ma: float
    phase_ma: float


@dataclass
class Volcano:
    id: int
    kind: str
    source: int  # hotspot id, or -1 for arcs
    plate: int
    qx: float  # material-frame position (baseline coordinates of its plate), km
    qy: float
    birth_ma: float
    last_active_ma: float
    height_m: float  # edifice height when fully grown
    radius_km: float
    partner: int = -1  # subducting plate for arcs

    def to_dict(self) -> dict:
        return dict(vars(self))  # full precision: the record must reproduce exactly


@dataclass
class VolcanoSet:
    hotspots: list[Hotspot] = field(default_factory=list)
    volcanoes: list[Volcano] = field(default_factory=list)

    def positions(self, kin, t: float):
        """Mantle-frame positions of all volcanoes born by ``t``: (x, y, index)."""
        idx = [i for i, v in enumerate(self.volcanoes) if v.birth_ma <= t + 1e-9]
        if not idx:
            return np.zeros(0), np.zeros(0), np.zeros(0, dtype=np.int64)
        plate = np.array([self.volcanoes[i].plate for i in idx])
        qx = np.array([self.volcanoes[i].qx for i in idx])
        qy = np.array([self.volcanoes[i].qy for i in idx])
        x, y = kin.forward(plate, qx, qy, t)
        return x, y, np.array(idx, dtype=np.int64)

    def heights(self, idx, t: float, erosion_intensity: float) -> np.ndarray:
        """Edifice height at time ``t``: grows while active, wears down once extinct."""
        out = np.zeros(len(idx))
        for k, i in enumerate(idx):
            v = self.volcanoes[int(i)]
            active_span = max(0.0, min(t, v.last_active_ma) - v.birth_ma)
            grown = 1.0 - math.exp(-(active_span + 0.6) / 1.2)
            idle = max(0.0, t - v.last_active_ma - _EXTINCT_AFTER_MA)
            rate = 0.012 if v.kind == HOTSPOT else 0.03  # share lost per Ma once extinct
            out[k] = v.height_m * grown * math.exp(-rate * erosion_intensity * idle)
        return out

    def is_active(self, i: int, t: float) -> bool:
        return t - self.volcanoes[i].last_active_ma <= _EXTINCT_AFTER_MA


def place_hotspots(seed: str, bounds, sim_area_km2: float, density: float) -> list[Hotspot]:
    """Hotspots fixed in the mantle frame, about one per 3.5e7 km^2 of the simulation area."""
    stream = Stream(seed, "geohistory/v3/hotspots")
    count = int(round(density * sim_area_km2 / 3.5e7))
    if density > 0:
        count = max(1, count)
    xmin, ymin, xmax, ymax = bounds
    spots: list[Hotspot] = []
    min_dist = 0.5 * math.sqrt((xmax - xmin) * (ymax - ymin) / max(1, count))
    tries = 0
    while len(spots) < count and tries < 4000:
        tries += 1
        u = stream.random(5)
        x = xmin + u[0] * (xmax - xmin)
        y = ymin + u[1] * (ymax - ymin)
        if any(math.hypot(x - s.x, y - s.y) < min_dist for s in spots):
            min_dist *= 0.995
            continue
        spots.append(
            Hotspot(
                len(spots),
                float(x),
                float(y),
                0.6 + 0.9 * float(u[2]),
                1.6 + 1.6 * float(u[3]),
                float(u[4]) * 1.5,
            )
        )
    return spots


class VolcanoBuilder:
    """Collects volcanoes while the history runs."""

    def __init__(self, seed: str, kin, hotspots: list[Hotspot], focus=None):
        self.focus = focus
        self.seed = seed
        self.kin = kin
        self.set = VolcanoSet(hotspots=hotspots)
        self._next_pulse = {h.id: h.phase_ma for h in hotspots}
        self._arc_clock = 0.0

    def _add(self, kind, source, plate, x, y, t, height, radius, partner=-1):
        qx, qy = self.kin.inverse(np.array([plate]), np.array([x]), np.array([y]), t)
        self.set.volcanoes.append(
            Volcano(
                id=len(self.set.volcanoes),
                kind=kind,
                source=source,
                plate=int(plate),
                qx=float(qx[0]),
                qy=float(qy[0]),
                birth_ma=float(t),
                last_active_ma=float(t),
                height_m=float(height),
                radius_km=float(radius),
                partner=int(partner),
            )
        )

    def step(self, snap, grid, cont_at_cell, t: float, intensity: float) -> None:
        # -- hotspots: one volcano per pulse on whatever plate is above ---------
        for h in self.set.hotspots:
            while self._next_pulse[h.id] <= t + 1e-9:
                when = self._next_pulse[h.id]
                self._next_pulse[h.id] = when + h.interval_ma
                ri, ci, ok = grid.nearest(h.x, h.y)
                if not ok:
                    continue
                plate = int(snap.owner[ri, ci])
                # keep the newest volcano of this hotspot alive instead of stacking
                serial = len(self.set.volcanoes)
                stream = Stream(self.seed, f"geohistory/v3/hotspot/{h.id}/{serial}")
                u = stream.random(3)
                continental = float(cont_at_cell[ri, ci]) > 0.5
                height = (700.0 + 900.0 * u[0]) if continental else (3300.0 + 2400.0 * u[0])
                height *= h.strength**0.5 * intensity**0.3
                radius = (38.0 + 22.0 * u[1]) if continental else (55.0 + 35.0 * u[1])
                self._add(HOTSPOT, h.id, plate, h.x, h.y, when, height, radius)
        # every volcano still above its hotspot stays active
        for v in self.set.volcanoes:
            if v.kind == HOTSPOT and self.set.is_active(v.id, t):
                h = self.set.hotspots[v.source]
                x, y = self.kin.forward(np.array([v.plate]), np.array([v.qx]), np.array([v.qy]), t)
                if math.hypot(float(x[0]) - h.x, float(y[0]) - h.y) < 0.8 * v.radius_km:
                    v.last_active_ma = t

        # -- arcs: centres stand on the narrow arc line of an active subduction zone ----
        if t + 1e-9 < self._arc_clock:
            return
        self._arc_clock = t + 1.0
        zone = snap.arc_zone > 0.45
        if self.focus is not None:
            zone &= self.focus
        if not zone.any():
            return
        x_now, y_now, idx_now = self.set.positions(self.kin, t)
        arcs = [k for k, i in enumerate(idx_now) if self.set.volcanoes[int(i)].kind == ARC]
        ax, ay = list(x_now[arcs]), list(y_now[arcs])
        arc_ids = [int(idx_now[k]) for k in arcs]
        gx, gy = grid.centers()
        # move every candidate cell onto the line itself (along the trench normal)
        gy_d, gx_d = np.gradient(snap.trench_distance.astype(np.float64), grid.cell)
        norm = np.maximum(np.hypot(gx_d, gy_d), 1e-6)
        off = snap.trench_distance - snap.trench_arc_km
        rows, cols = np.nonzero(zone & np.isfinite(off))
        stream = Stream(self.seed, f"geohistory/v3/arc/{t:.4f}")
        order = stream.permutation(rows.size)
        spacing = 68.0  # arc volcanoes stand tens of km apart
        budget = 600
        for k in order:
            if budget <= 0:
                break
            r, c = int(rows[k]), int(cols[k])
            jitter = stream.random(4)
            shift = float(off[r, c]) - (jitter[0] - 0.5) * 16.0
            x = float(gx[r, c]) - shift * float(gx_d[r, c] / norm[r, c])
            y = float(gy[r, c]) - shift * float(gy_d[r, c] / norm[r, c])
            if ax:
                near = np.hypot(np.array(ax) - x, np.array(ay) - y)
                j = int(np.argmin(near))
                if near[j] < spacing:
                    if arc_ids[j] >= 0:
                        self.set.volcanoes[arc_ids[j]].last_active_ma = t
                    continue
            continental = float(cont_at_cell[r, c]) > 0.5
            height = (900.0 + 1300.0 * jitter[2]) if continental else (1800.0 + 2200.0 * jitter[2])
            radius = 14.0 + 12.0 * jitter[3]
            self._add(
                ARC,
                -1,
                int(snap.owner[r, c]),
                x,
                y,
                t,
                height * intensity**0.3,
                radius,
                partner=int(snap.trench_under[r, c]),
            )
            ax.append(x)
            ay.append(y)
            arc_ids.append(len(self.set.volcanoes) - 1)
            budget -= 1


def cone_field(x, y, vx, vy, heights, radii):
    """Sum of volcanic cones at world points: ``(relief_m, nearest_index, distance_km)``."""
    from scipy.spatial import cKDTree

    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    relief = np.zeros(x.shape)
    nearest = np.full(x.shape, -1, dtype=np.int64)
    dist = np.full(x.shape, np.inf)
    if len(vx) == 0:
        return relief, nearest, dist
    tree = cKDTree(np.column_stack([vx, vy]))
    reach = float(np.max(radii)) * 3.2
    k = min(4, len(vx))
    d, idx = tree.query(np.column_stack([x.ravel(), y.ravel()]), k=k, distance_upper_bound=reach)
    if k == 1:
        d, idx = d[:, None], idx[:, None]
    ok = np.isfinite(d)
    idx = np.where(ok, idx, 0)
    h = np.asarray(heights)[idx]
    r = np.asarray(radii)[idx]
    shape = np.where(ok, h * np.exp(-((d / r) ** 1.6)), 0.0)
    # overlapping edifices merge (max) rather than pile up
    relief = shape.max(axis=1).reshape(x.shape)
    first = np.where(ok[:, 0], idx[:, 0], -1)
    nearest = first.reshape(x.shape)
    dist = np.where(ok[:, 0], d[:, 0], np.inf).reshape(x.shape)
    return relief, nearest, dist
