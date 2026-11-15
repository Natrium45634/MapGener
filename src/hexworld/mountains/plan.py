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
from scipy import ndimage
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import dijkstra

from ..rng import Stream

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


def build_plan(
    grid,
    relief,
    kind,
    age_ma,
    erosion_share,
    elevation,
    params,
    seed: str,
    time_ma: float,
    focus=None,
) -> MountainPlan:
    """Build the plan for one snapshot from mantle-frame rasters on the history grid.

    ``relief`` - orogenic relief left by the history, m; ``kind`` - KIND_* code;
    ``age_ma`` - time since the last mountain-building; ``erosion_share`` - share
    of the built relief already eroded (0..1); ``elevation`` - macro elevation, m.
    """
    plan = MountainPlan(time_ma=float(time_ma))
    cell = grid.cell
    mask = (relief > params.system_threshold_m) & (elevation > -1500.0)
    labels, count = ndimage.label(mask, structure=np.ones((3, 3)))
    plan.system_raster = np.full(grid.shape, -1, dtype=np.int32)
    if count == 0:
        return plan
    gx, gy = grid.centers()
    sizes = np.bincount(labels.ravel(), minlength=count + 1)
    cx = ndimage.mean(gx, labels, np.arange(1, count + 1))
    cy = ndimage.mean(gy, labels, np.arange(1, count + 1))
    keep = [k for k in range(1, count + 1) if sizes[k] >= 4]
    if focus is not None:  # only systems that reach the area of interest get a structure
        present = np.bincount(labels[focus], minlength=count + 1)
        keep = [k for k in keep if present[k] > 0]
    keep.sort(key=lambda k: (-int(sizes[k]), float(cx[k - 1]), float(cy[k - 1])))
    inside_km = ndimage.distance_transform_edt(mask) * cell
    step = max(8.0, min(14.0, 0.25 * cell))

    for sid, label in enumerate(keep):
        rows, cols = np.nonzero(labels == label)
        rel = relief[rows, cols].astype(np.float64)
        plan.system_raster[rows, cols] = sid
        kinds = np.bincount(kind[rows, cols], minlength=5)
        kinds[0] = 0
        kind_name = KIND_NAMES[int(np.argmax(kinds))] if kinds.any() else "inherited"
        known = age_ma[rows, cols]
        known = known[np.isfinite(known)]
        age = float(np.median(known)) if known.size else 0.0
        worn = float(np.nanmean(erosion_share[rows, cols]))
        maturity = float(np.clip(0.6 * worn + 0.4 * min(age / 150.0, 1.0), 0.0, 1.0))
        area = float(rows.size * cell * cell)
        width = 2.0 * float(inside_km[rows, cols].max())
        if kind_name == "arc" and maturity < 0.5:
            morphology = "volcanic_arc"
        elif width > 520.0 and float(rel.mean()) > 2200.0:
            morphology = "plateau"
        elif maturity < 0.33:
            morphology = "young_sharp"
        elif maturity < 0.66:
            morphology = "mature"
        else:
            morphology = "old_rounded"
        system = MountainSystem(
            id=sid,
            kind=kind_name,
            morphology=morphology,
            age_ma=round(age, 3),
            maturity=round(maturity, 4),
            sharpness=round(1.55 - 0.85 * maturity, 4),
            relief_gain=round(
                (0.95 - 0.5 * maturity) * (0.75 if morphology == "plateau" else 1.0), 4
            ),
            area_km2=area,
            centroid_km=(float(gx[rows, cols].mean()), float(gy[rows, cols].mean())),
            bounds_km=(
                float(gx[rows, cols].min()),
                float(gy[rows, cols].min()),
                float(gx[rows, cols].max()),
                float(gy[rows, cols].max()),
            ),
            max_relief_m=float(rel.max()),
            touches_raster_edge=bool(
                rows.min() == 0
                or cols.min() == 0
                or rows.max() == grid.ny - 1
                or cols.max() == grid.nx - 1
            ),
        )
        plan.systems.append(system)
        stream = Stream(seed, f"mountains/v3/system/{sid}/{rows.size}")
        cost_graph, hop_graph = _system_graph(rows, cols, rel, cell)
        px = gx[rows, cols]
        py = gy[rows, cols]

        # main ridge: far end -> far end, following the crest
        start = int(np.argmax(rel))
        far_a = int(
            np.argmax(np.where(np.isfinite(d := dijkstra(hop_graph, indices=start)), d, -1.0))
        )
        d_a = dijkstra(hop_graph, indices=far_a)
        far_b = int(np.argmax(np.where(np.isfinite(d_a), d_a, -1.0)))
        _dist, pred = dijkstra(cost_graph, indices=far_a, return_predecessors=True)
        main = _path(pred, far_b, {far_a})
        ridge_cells: list[tuple[list[int], int, int]] = [(main, RANK_MAIN, -1)]
        on_ridge = set(main)

        # branches: parts of the system far from every ridge get their own range
        max_branches = int(min(8, rows.size // 25))
        reach = 2.6 * params.range_half_width_km
        for _ in range(max_branches):
            sources = np.array(sorted(on_ridge))
            hop_d = dijkstra(hop_graph, indices=sources, min_only=True)
            hop_d = np.where(np.isfinite(hop_d), hop_d, -1.0)
            far = int(np.argmax(hop_d))
            if hop_d[far] < reach:
                break
            _d, pred_b, _src = dijkstra(
                cost_graph, indices=sources, min_only=True, return_predecessors=True
            )
            branch = _path(pred_b, far, on_ridge)
            if len(branch) < 3:
                break
            ridge_cells.append((branch[::-1], RANK_BRANCH, -2))
            on_ridge.update(branch)

        first_ridge = len(plan.ridges)
        cell_to_ridge: dict[int, int] = {}
        for cells, rank, _parent in ridge_cells:
            pts = np.column_stack([px[cells], py[cells]])
            jit = (stream.random(2 * len(pts)).reshape(-1, 2) - 0.5) * 0.5 * cell
            jit[0] = jit[-1] = 0.0
            pts = resample(chaikin(pts + jit, 2), step)
            rid = len(plan.ridges)
            parent = -1
            if rank == RANK_BRANCH:
                parent = cell_to_ridge.get(cells[0], first_ridge)
                plan.knots.append(
                    {
                        "id": len(plan.knots),
                        "system": sid,
                        "x_km": float(pts[0, 0]),
                        "y_km": float(pts[0, 1]),
                        "ridges": [parent, rid],
                    }
                )
            for c in cells:
                cell_to_ridge.setdefault(c, rid)
            hw = params.range_half_width_km * (1.0 if rank == RANK_MAIN else 0.85)
            plan.ridges.append(Ridge(rid, sid, rank, parent, pts, np.ones(len(pts)), hw))
        system.length_km = float(arclength(plan.ridges[first_ridge].points)[-1])

        # peaks and passes along every main / branch ridge
        rel_max = max(float(rel.max()), 1.0)
        for ridge in plan.ridges[first_ridge:]:
            s = arclength(ridge.points)
            total = s[-1]
            local = np.clip(
                grid.bilinear(relief, ridge.points[:, 0], ridge.points[:, 1]) / rel_max, 0.15, 1.0
            )
            spacing = params.peak_spacing_km * (1.0 + 0.5 * system.maturity)
            n_peaks = max(1, int(round(total / spacing)))
            u = stream.random(3 * n_peaks + 3)
            peak_s = np.sort((np.arange(n_peaks) + 0.25 + 0.5 * u[:n_peaks]) * total / n_peaks)
            peak_v = 0.78 + 0.22 * u[n_peaks : 2 * n_peaks]
            nodes_s, nodes_v = [0.0], [0.42]
            peak_ids = []
            for k in range(n_peaks):
                if k > 0:
                    saddle = 0.5 * (peak_s[k - 1] + peak_s[k])
                    depth = 0.52 + 0.2 * u[2 * n_peaks + k] + 0.12 * system.maturity
                    nodes_s.append(saddle)
                    nodes_v.append(min(peak_v[k - 1], peak_v[k]) * min(depth, 0.9))
                nodes_s.append(peak_s[k])
                nodes_v.append(peak_v[k])
            nodes_s.append(total + 1e-6)
            nodes_v.append(0.42)
            ridge.crest = _cosine_profile(s, np.array(nodes_s), np.array(nodes_v)) * (
                0.55 + 0.45 * local
            )
            for k in range(n_peaks):
                i = int(np.argmin(np.abs(s - peak_s[k])))
                peak_ids.append(len(plan.peaks))
                plan.peaks.append(
                    {
                        "id": len(plan.peaks),
                        "system": sid,
                        "ridge": ridge.id,
                        "x_km": float(ridge.points[i, 0]),
                        "y_km": float(ridge.points[i, 1]),
                        "crest": float(ridge.crest[i]),
                    }
                )
                if k > 0:
                    j = int(np.argmin(np.abs(s - 0.5 * (peak_s[k - 1] + peak_s[k]))))
                    plan.passes.append(
                        {
                            "id": len(plan.passes),
                            "system": sid,
                            "ridge": ridge.id,
                            "x_km": float(ridge.points[j, 0]),
                            "y_km": float(ridge.points[j, 1]),
                            "crest": float(ridge.crest[j]),
                            "between_peaks": [peak_ids[k - 1], peak_ids[k]],
                        }
                    )

        # spurs and the valleys between them
        for ridge in list(plan.ridges[first_ridge:]):
            s = arclength(ridge.points)
            total = s[-1]
            gap = 150.0 + 60.0 * system.maturity
            n_spurs = int(total // gap)
            if n_spurs < 1:
                continue
            u = stream.random(4 * n_spurs)
            roots = {1: [], -1: []}
            for k in range(n_spurs):
                side = 1 if (k + ridge.id) % 2 == 0 else -1
                at = (k + 0.3 + 0.4 * u[4 * k]) * total / n_spurs
                i = int(np.clip(np.searchsorted(s, at), 1, len(s) - 2))
                tangent = ridge.points[i + 1] - ridge.points[i - 1]
                tangent /= max(np.hypot(*tangent), 1e-9)
                normal = np.array([-tangent[1], tangent[0]]) * side
                ang = (u[4 * k + 1] - 0.5) * math.radians(50.0)
                ca, sa = math.cos(ang), math.sin(ang)
                direction = np.array(
                    [ca * normal[0] - sa * normal[1], sa * normal[0] + ca * normal[1]]
                )
                room = float(grid.bilinear(inside_km, ridge.points[i, 0], ridge.points[i, 1]))
                length = float(np.clip(0.8 * room, 0.0, 280.0)) * (0.7 + 0.5 * u[4 * k + 2])
                if length < 55.0:
                    continue
                bend = np.array([-direction[1], direction[0]]) * (u[4 * k + 3] - 0.5) * 0.35
                raw = np.array(
                    [
                        ridge.points[i],
                        ridge.points[i] + length * 0.55 * (direction + bend),
                        ridge.points[i] + length * direction,
                    ]
                )
                pts = resample(chaikin(raw, 2), step)
                frac = arclength(pts) / max(arclength(pts)[-1], 1e-9)
                crest = ridge.crest[i] * (0.72 - 0.5 * frac) * (1.0 + 0.12 * np.sin(6.3 * frac))
                rid = len(plan.ridges)
                plan.ridges.append(
                    Ridge(rid, sid, RANK_SPUR, ridge.id, pts, crest, params.spur_half_width_km)
                )
                plan.knots.append(
                    {
                        "id": len(plan.knots),
                        "system": sid,
                        "x_km": float(pts[0, 0]),
                        "y_km": float(pts[0, 1]),
                        "ridges": [ridge.id, rid],
                    }
                )
                roots[side].append((at, i, direction, length))
            for _side, items in roots.items():
                for (a0, i0, d0, l0), (a1, i1, d1, l1) in zip(items[:-1], items[1:], strict=True):
                    if a1 - a0 > 2.6 * gap:
                        continue
                    mid = (i0 + i1) // 2
                    direction = d0 + d1
                    direction /= max(np.hypot(*direction), 1e-9)
                    head = ridge.points[mid] + direction * 0.45 * ridge.half_width_km
                    length = 1.15 * 0.5 * (l0 + l1)
                    pts = resample(np.array([head, head + direction * length]), step)
                    frac = arclength(pts) / max(arclength(pts)[-1], 1e-9)
                    depth = np.clip(frac / 0.25, 0.0, 1.0) * (1.0 - 0.75 * frac**2)
                    plan.valleys.append(
                        Valley(
                            len(plan.valleys),
                            sid,
                            ridge.id,
                            pts,
                            depth,
                            14.0 + 16.0 * system.maturity,
                        )
                    )
    return plan
