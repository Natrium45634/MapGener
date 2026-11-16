"""Structural mountain plan: systems, ridge axes, knots, peaks, passes, valleys.

Mountains are built as a *structure* first. Input is the orogenic relief the
geological history left at a snapshot (how much mountain-building happened
where, what kind, how long ago, how much erosion followed). From it:

1. connected areas of orogenic relief become **mountain systems** with stable ids;
2. the **main ridge** of a system is the least-cost path between its two far ends,
   where cost falls with relief - so the axis follows the crest of the belt and
   bends with it instead of copying a plate boundary;
3. parts of the system far from the main ridge get **branch ridges** that join it
   at a **knot**; **spurs** leave ridges sideways into the flanks;
4. along every ridge **peaks** alternate with **passes**: a pass is by construction
   the saddle between two named peaks of the same ridge;
5. **valleys** start between neighbouring spurs and run down the flank, so they
   never cross a peak.

Morphology (sharpness, relief contrast, valley width) follows from the system's
age and from how much of it the modelled erosion has removed.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
from scipy.sparse import coo_matrix

RANK_MAIN, RANK_BRANCH, RANK_SPUR = 0, 1, 2
RANK_NAMES = ("main", "branch", "spur")
KIND_NAMES = ("none", "collision", "arc", "inherited", "rift")
_NEIGHBOURS = ((-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1))


@dataclass
class Ridge:
    id: int
    system: int
    rank: int
    parent: int  # ridge it joins, -1 for a main ridge
    points: np.ndarray  # (n, 2) world km
    crest: np.ndarray  # (n,) relative crest height 0..1
    half_width_km: float
    #: exponent of the cross profile; None = the system's sharpness
    sharpness: float | None = None
    #: share of the half-width that is a level top (0 = ridge, 0.6 = plateau)
    plateau: float = 0.0
    #: -1..1: the right side (normal direction) is wider (+) or narrower (-) than the left
    asymmetry: float = 0.0
    #: what the range is in its system: main, frontal, back, secondary, plateau, base,
    #: core, fold, block, branch, spur
    role: str = "main"


@dataclass
class Valley:
    id: int
    system: int
    ridge: int
    points: np.ndarray  # (n, 2), from the head near the ridge down the flank
    depth: np.ndarray  # (n,) relative depth 0..1
    half_width_km: float


@dataclass
class MountainSystem:
    id: int
    kind: str  # collision / arc / inherited
    morphology: str  # young_sharp / mature / old_rounded / plateau / volcanic_arc
    age_ma: float  # since the last mountain-building
    maturity: float  # 0 fresh .. 1 worn down
    sharpness: float  # exponent of the cross profile (higher = sharper crest)
    relief_gain: float  # share of the orogenic relief turned into ridge / valley contrast
    area_km2: float
    centroid_km: tuple
    bounds_km: tuple
    max_relief_m: float
    length_km: float = 0.0
    touches_raster_edge: bool = False
    #: architecture of the system (see :data:`hexworld.mountains.builder.STYLES`)
    style: str = "alpine"
    #: number of separate ranges (not counting spurs)
    ranges: int = 1
    #: median width of the belt across its axis, km
    width_km: float = 0.0


@dataclass
class MountainPlan:
    time_ma: float
    systems: list[MountainSystem] = field(default_factory=list)
    ridges: list[Ridge] = field(default_factory=list)
    valleys: list[Valley] = field(default_factory=list)
    peaks: list[dict] = field(default_factory=list)
    passes: list[dict] = field(default_factory=list)
    knots: list[dict] = field(default_factory=list)
    #: raster of system ids on the history grid (-1 = none)
    system_raster: np.ndarray | None = None

    def to_dict(self) -> dict:
        def r3(a):
            return np.round(np.asarray(a, dtype=np.float64), 3).tolist()

        return {
            "time_ma": self.time_ma,
            "systems": [
                vars(s) | {"centroid_km": list(s.centroid_km), "bounds_km": list(s.bounds_km)}
                for s in self.systems
            ],
            "ridges": [
                {
                    "id": r.id,
                    "system": r.system,
                    "rank": RANK_NAMES[r.rank],
                    "parent": r.parent,
                    "half_width_km": r.half_width_km,
                    "sharpness": r.sharpness,
                    "plateau": r.plateau,
                    "asymmetry": r.asymmetry,
                    "role": r.role,
                    "points_km": r3(r.points),
                    "crest": r3(r.crest),
                }
                for r in self.ridges
            ],
            "valleys": [
                {
                    "id": v.id,
                    "system": v.system,
                    "ridge": v.ridge,
                    "half_width_km": v.half_width_km,
                    "points_km": r3(v.points),
                    "depth": r3(v.depth),
                }
                for v in self.valleys
            ],
            "peaks": self.peaks,
            "passes": self.passes,
            "knots": self.knots,
        }

    @classmethod
    def from_dict(cls, d: dict) -> MountainPlan:
        plan = cls(time_ma=float(d["time_ma"]))
        known = set(MountainSystem.__dataclass_fields__)
        for s in d["systems"]:
            s = {k: v for k, v in s.items() if k in known}  # drop per-window statistics
            s["centroid_km"] = tuple(s["centroid_km"])
            s["bounds_km"] = tuple(s["bounds_km"])
            plan.systems.append(MountainSystem(**s))
        for r in d["ridges"]:
            plan.ridges.append(
                Ridge(
                    r["id"],
                    r["system"],
                    RANK_NAMES.index(r["rank"]),
                    r["parent"],
                    np.asarray(r["points_km"], dtype=np.float64),
                    np.asarray(r["crest"], dtype=np.float64),
                    r["half_width_km"],
                    r.get("sharpness"),
                    float(r.get("plateau", 0.0)),
                    float(r.get("asymmetry", 0.0)),
                    r.get("role", "main" if r["rank"] == "main" else r["rank"]),
                )
            )
        for v in d["valleys"]:
            plan.valleys.append(
                Valley(
                    v["id"],
                    v["system"],
                    v["ridge"],
                    np.asarray(v["points_km"], dtype=np.float64),
                    np.asarray(v["depth"], dtype=np.float64),
                    v["half_width_km"],
                )
            )
        plan.peaks, plan.passes, plan.knots = d["peaks"], d["passes"], d["knots"]
        return plan


# -- geometry helpers -----------------------------------------------------------
def chaikin(points: np.ndarray, rounds: int = 2) -> np.ndarray:
    for _ in range(rounds):
        if len(points) < 3:
            break
        a, b = points[:-1], points[1:]
        mid = np.empty((2 * len(a), 2))
        mid[0::2] = 0.75 * a + 0.25 * b
        mid[1::2] = 0.25 * a + 0.75 * b
        points = np.vstack([points[:1], mid, points[-1:]])
    return points


def resample(points: np.ndarray, step: float) -> np.ndarray:
    seg = np.hypot(*np.diff(points, axis=0).T)
    s = np.concatenate([[0.0], np.cumsum(seg)])
    if s[-1] < 1e-6:
        return points[:1]
    n = max(2, int(round(s[-1] / step)) + 1)
    t = np.linspace(0.0, s[-1], n)
    return np.column_stack([np.interp(t, s, points[:, 0]), np.interp(t, s, points[:, 1])])


def arclength(points: np.ndarray) -> np.ndarray:
    if len(points) < 2:
        return np.zeros(len(points))
    return np.concatenate([[0.0], np.cumsum(np.hypot(*np.diff(points, axis=0).T))])


def _cosine_profile(s, nodes_s, nodes_v):
    """Smooth (cosine) interpolation through nodes - rounded peaks and saddles."""
    idx = np.clip(np.searchsorted(nodes_s, s, side="right") - 1, 0, len(nodes_s) - 2)
    s0, s1 = nodes_s[idx], nodes_s[idx + 1]
    t = np.clip((s - s0) / np.maximum(s1 - s0, 1e-9), 0.0, 1.0)
    w = 0.5 - 0.5 * np.cos(np.pi * t)
    return nodes_v[idx] * (1 - w) + nodes_v[idx + 1] * w


def _system_graph(rows, cols, relief, cell):
    """Sparse 8-neighbour graph of a system; edge cost falls with relief."""
    n = rows.size
    lookup = {(int(r), int(c)): i for i, (r, c) in enumerate(zip(rows, cols, strict=True))}
    src, dst, cost, hop = [], [], [], []
    for i in range(n):
        r, c = int(rows[i]), int(cols[i])
        for dr, dc in _NEIGHBOURS:
            j = lookup.get((r + dr, c + dc))
            if j is None:
                continue
            dist = cell * math.hypot(dr, dc)
            mean = 0.5 * (relief[i] + relief[j])
            src.append(i)
            dst.append(j)
            cost.append(dist / (mean / 1000.0 + 0.15) ** 2)
            hop.append(dist)
    shape = (n, n)
    return (
        coo_matrix((cost, (src, dst)), shape=shape).tocsr(),
        coo_matrix((hop, (src, dst)), shape=shape).tocsr(),
    )


def _path(pred, start, source_set):
    out = [start]
    while out[-1] not in source_set:
        nxt = pred[out[-1]]
        if nxt < 0:
            break
        out.append(int(nxt))
    return out
