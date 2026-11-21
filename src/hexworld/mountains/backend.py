"""Mountain height backends behind one adapter boundary.

HexWorld asks a backend for a :class:`HeightPatch` - relief to be *added* to the
macro surface inside mountain systems - and never lets a backend replace the
surface. The request carries everything a backend may use (seed, snapshot id,
bounds, the structural plan, uplift field, age, constraints, composition
policy), and nothing of PySide6.

Two backends:

* ``reference`` - a small CPU implementation in NumPy that turns the structural
  plan into heights. **It is a reference fallback, not PMGE**: no erosion-carved
  valleys, no rock-dependent shapes beyond a few classes, no local re-synthesis.
* ``pmge`` - the external C++ library through its C ABI. Only the loader and the
  version negotiation live here; if the library is absent the backend reports
  itself unavailable instead of pretending.

Switching the backend changes how relief is synthesised, not the HexWorld data
contract (plan, layers, ids stay the same).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

import numpy as np
from scipy.spatial import cKDTree

from ..noise import FractalNoise
from ..rng import Stream
from .plan import MountainPlan

ADAPTER_ABI_VERSION = 1
POLICIES = ("add", "masked", "constrained_fit")
CLASS_NONE, CLASS_CREST, CLASS_SLOPE, CLASS_PASS, CLASS_VALLEY, CLASS_FOOTHILL, CLASS_VOLCANO = (
    range(7)
)
CLASS_NAMES = ("none", "crest", "slope", "pass", "valley", "foothill", "volcano")


class BackendUnavailable(RuntimeError):
    pass


@dataclass
class MountainRequest:
    """What HexWorld hands to a mountain backend."""

    world_seed: str
    snapshot_id: str
    world_bounds_km: tuple
    plan: MountainPlan
    #: ``uplift(x, y)`` -> orogenic relief in m at world points (the target field).
    uplift: object
    #: ``gain(x, y)`` -> share of that relief to turn into ridge / valley contrast.
    gain: object
    detail: float = 1.0
    max_elevation_m: float = 8800.0
    policy: str = "masked"
    constraints: dict = field(default_factory=dict)


@dataclass
class HeightPatch:
    """Relief to add to the macro surface (m) plus structural diagnostics."""

    relief_m: np.ndarray
    structure: np.ndarray  # 0..1: 1 on a crest at a peak, 0 away from ridges
    valley: np.ndarray  # 0..1 valley strength
    system_id: np.ndarray  # int32, -1 outside mountain systems
    ridge_distance_km: np.ndarray


class MountainBackend(Protocol):
    name: str
    version: str

    def capabilities(self) -> dict: ...

    def sampler(self, request: MountainRequest) -> HeightSampler: ...


class HeightSampler(Protocol):
    def sample_points(self, x, y) -> HeightPatch: ...


class _Group:
    """Points of ridges (or valleys) of similar width in one KD-tree."""

    def __init__(self, arrays: dict):
        self.__dict__.update(arrays)
        self.tree = cKDTree(np.column_stack([self.px, self.py]))
        self.reach = float((self.hw * (1.0 + np.abs(self.asym))).max())


def _point_normals(points: np.ndarray) -> np.ndarray:
    d = np.gradient(points, axis=0) if len(points) > 1 else np.array([[1.0, 0.0]])
    t = d / np.maximum(np.hypot(d[:, 0], d[:, 1]), 1e-9)[:, None]
    return np.column_stack([-t[:, 1], t[:, 0]])


def _groups(items, width_of) -> list[_Group]:
    """Split by half-width (narrow, wide, very wide) and keep spurs apart, so neither a
    narrow crest nor a swarm of dense spurs hides a range, a plateau or a basin from the
    nearest-neighbour search (a hidden range would show as a cliff)."""
    buckets: dict[tuple, list] = {}
    for it in items:
        hw = width_of(it)
        spur = getattr(it, "role", "") == "spur"
        # narrow width classes keep the search bound tight (see _ridge_values)
        cls = int(np.searchsorted([40.0, 60.0, 90.0, 140.0, 220.0], hw))
        buckets.setdefault((spur, cls), []).append(it)
    return [buckets[k] for k in sorted(buckets)]


def ridge_groups(ridges, sharp_of) -> list[_Group]:
    """Search groups of ridge vertices (see :func:`_groups`) with what the profile needs:
    crest, half-width, sharpness, plateau, asymmetry, normals and the next vertex."""
    out = []
    for group in _groups(list(ridges), lambda r: r.half_width_km):
        names = ("px", "py", "crest", "hw", "sharp", "sysid", "plat", "asym", "nx", "ny", "nxt",
                 "prv")  # fmt: skip
        parts: dict[str, list] = {k: [] for k in names}
        offset = 0
        for r in group:
            n = len(r.points)
            normals = _point_normals(r.points)
            parts["px"].append(r.points[:, 0])
            parts["py"].append(r.points[:, 1])
            parts["crest"].append(r.crest)
            parts["hw"].append(np.full(n, r.half_width_km))
            parts["sharp"].append(np.full(n, sharp_of(r)))
            parts["sysid"].append(np.full(n, r.system))
            parts["plat"].append(np.full(n, min(r.plateau, 0.9)))
            parts["asym"].append(np.full(n, r.asymmetry))
            parts["nx"].append(normals[:, 0])
            parts["ny"].append(normals[:, 1])
            # the next / previous vertex of the same ridge (the ends point to themselves)
            nxt = np.arange(offset + 1, offset + n + 1)
            nxt[-1] = offset + n - 1
            prv = np.arange(offset - 1, offset + n - 1)
            prv[0] = offset
            parts["nxt"].append(nxt)
            parts["prv"].append(prv)
            offset += n
        arrays = {k: np.concatenate(v) for k, v in parts.items()}
        arrays["sysid"] = arrays["sysid"].astype(np.int32)
        arrays["nxt"] = arrays["nxt"].astype(np.int64)
        arrays["prv"] = arrays["prv"].astype(np.int64)
        grp = _Group(arrays)
        # bounds for the exact search (see ReferenceSampler._ridge_values)
        grp.crest_max = float(grp.crest.max())
        grp.sharp_min = float(grp.sharp.min())
        grp.plat_max = float(grp.plat.max())
        grp.seg_max = float(np.hypot(grp.px[grp.nxt] - grp.px, grp.py[grp.nxt] - grp.py).max())
        out.append(grp)
    return out


class ReferenceSampler:
    """Analytic sampler: relief at any world point, so windows and chunks never seam.

    Cross profile of a ridge at distance ``d`` on side ``side`` of its axis::

        hw_side = hw * (1 + asymmetry * side)          # asymmetric flanks
        u = d / hw_side;  u' = max(0, (u - plateau) / (1 - plateau))
        structure = crest * (1 - u') ** sharpness      # level top up to ``plateau``
    """

    def __init__(self, request: MountainRequest):
        self.request = request
        plan = request.plan
        sharp_by_system = {s.id: s.sharpness for s in plan.systems}
        self.empty = not plan.ridges
        if self.empty:
            self.groups, self.vgroups = [], []
        else:
            self.groups = ridge_groups(
                plan.ridges,
                lambda r: (
                    r.sharpness if r.sharpness is not None else sharp_by_system.get(r.system, 1.2)
                ),
            )
            self.vgroups = []
            for group in _groups(plan.valleys, lambda v: v.half_width_km):
                arrays = {
                    "px": np.concatenate([v.points[:, 0] for v in group]),
                    "py": np.concatenate([v.points[:, 1] for v in group]),
                    "vd": np.concatenate([v.depth for v in group]),
                    "hw": np.concatenate([np.full(len(v.points), v.half_width_km) for v in group]),
                }
                arrays["asym"] = np.zeros(arrays["px"].size)
                self.vgroups.append(_Group(arrays))
        pts = np.array([[p["x_km"], p["y_km"]] for p in plan.passes]) if plan.passes else None
        self.pass_tree = cKDTree(pts) if pts is not None else None
        stream = Stream(request.world_seed, "mountains/v3/detail")
        self.detail = FractalNoise(stream, [46.0, 23.0, 11.5], [1.0, 0.55, 0.3])

    @staticmethod
    def _segment_values(g: _Group, pts: np.ndarray, idx, ok, a_idx, b_idx):
        """Profile value of the segments a -> b (crest interpolated along them)."""
        ax, ay = g.px[a_idx], g.py[a_idx]
        bx, by = g.px[b_idx] - ax, g.py[b_idx] - ay
        seg2 = bx * bx + by * by
        rx, ry = pts[:, 0:1] - ax, pts[:, 1:2] - ay
        t = np.clip((rx * bx + ry * by) / np.maximum(seg2, 1e-12), 0.0, 1.0)
        t = np.where(seg2 > 0, t, 0.0)
        d = np.hypot(rx - t * bx, ry - t * by)
        crest = g.crest[a_idx] * (1.0 - t) + g.crest[b_idx] * t
        # which flank (-1 .. 1), blended over a quarter of the half-width: a hard sign
        # flips beyond the ends of a ridge and switched an asymmetric flank's width (ledge)
        side = np.clip((rx * g.nx[idx] + ry * g.ny[idx]) / (0.25 * g.hw[idx]), -1.0, 1.0)
        hw = g.hw[idx] * (1.0 + g.asym[idx] * side)
        u = np.where(ok, d / hw, 2.0)
        plat = g.plat[idx]
        u2 = np.clip((u - plat) / (1.0 - plat), 0.0, None)
        prof = np.clip(1.0 - u2, 0.0, 1.0) ** g.sharp[idx]
        return np.where(ok, crest * prof, 0.0)

    @staticmethod
    def _ridge_values(g: _Group, pts: np.ndarray, exact: bool = True):
        """Highest ridge profile at each point, exactly.

        The k nearest vertices are tried first; every other vertex lies beyond the k-th
        distance, so their segments are at least ``d_k - seg_max / 2`` away and cannot
        give more than ``crest_max * profile(that distance)``. Points whose best value does
        not beat that bound are searched again with 4 x more neighbours. (A fixed k let a
        crowd of nearby vertices hide the ridge that is highest there: a ledge.)
        ``exact=False``: one pass of the k nearest (a guide field, e.g. for tracing spurs).
        """
        if not exact:  # a guide field: nearest vertices only, one pass
            k = min(8, g.px.size)
            d, idx = g.tree.query(pts, k=k, distance_upper_bound=g.reach)
            if k == 1:
                d, idx = d[:, None], idx[:, None]
            ok = np.isfinite(d)
            idx = np.where(ok, idx, 0)
            side = np.clip(
                ((pts[:, 0:1] - g.px[idx]) * g.nx[idx] + (pts[:, 1:2] - g.py[idx]) * g.ny[idx])
                / (0.25 * g.hw[idx]),
                -1.0,
                1.0,
            )
            u = np.where(ok, d / (g.hw[idx] * (1.0 + g.asym[idx] * side)), 2.0)
            u2 = np.clip((u - g.plat[idx]) / (1.0 - g.plat[idx]), 0.0, None)
            val = np.where(ok, g.crest[idx] * np.clip(1.0 - u2, 0.0, 1.0) ** g.sharp[idx], 0.0)
            pick = np.argmax(val, axis=1)
            rows = np.arange(len(pts))
            return val[rows, pick], g.sysid[idx[rows, pick]], np.where(ok[:, 0], d[:, 0], np.inf)
        n = len(pts)
        best_val = np.zeros(n)
        best_sid = np.full(n, -1, dtype=np.int32)
        dist = np.full(n, np.inf)
        todo = np.arange(n)
        k = min(8, g.px.size)
        while todo.size:
            sub = pts[todo]
            d, idx = g.tree.query(sub, k=k, distance_upper_bound=g.reach)
            if k == 1:
                d, idx = d[:, None], idx[:, None]
            ok = np.isfinite(d)
            idx = np.where(ok, idx, 0)
            v_next = ReferenceSampler._segment_values(g, sub, idx, ok, idx, g.nxt[idx])
            v_prev = ReferenceSampler._segment_values(g, sub, idx, ok, g.prv[idx], idx)
            val = np.maximum(v_next, v_prev)
            pick = np.argmax(val, axis=1)
            rows = np.arange(len(todo))
            top = val[rows, pick]
            # what any vertex beyond the k-th could still give
            dk = np.where(ok[:, -1], d[:, -1], np.inf)
            reach_d = np.maximum(dk - 0.5 * g.seg_max, 0.0)
            u = reach_d / max(g.reach, 1e-9)
            u2 = np.clip((u - g.plat_max) / max(1.0 - g.plat_max, 1e-9), 0.0, None)
            bound = g.crest_max * np.clip(1.0 - u2, 0.0, 1.0) ** g.sharp_min
            done = (top >= bound) | ~ok[:, -1] | (k >= g.px.size) | (not exact)
            keep = todo[done]
            best_val[keep] = top[done]
            best_sid[keep] = g.sysid[idx[rows, pick]][done]
            dist[keep] = np.where(ok[done, 0], d[done, 0], np.inf)
            todo = todo[~done]
            k = min(4 * k, g.px.size)
        return best_val, best_sid, dist

    def sample_points(self, x, y) -> HeightPatch:
        x = np.asarray(x, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        shape = x.shape
        n = x.size
        structure = np.zeros(n)
        valley = np.zeros(n)
        system = np.full(n, -1, dtype=np.int32)
        dist = np.full(n, np.inf)
        if self.empty or n == 0:
            zero = np.zeros(shape)
            return HeightPatch(zero, zero, zero, system.reshape(shape), dist.reshape(shape))
        pts = np.column_stack([x.ravel(), y.ravel()])
        for g in self.groups:
            val, sid, d = self._ridge_values(g, pts)
            better = val > structure
            structure = np.where(better, val, structure)
            system = np.where(better, sid, system)
            dist = np.minimum(dist, d)
        system = np.where(structure > 0, system, -1).astype(np.int32)
        for g in self.vgroups:
            kv = min(12, g.px.size)  # as for ridges: dense valleys must not hide a basin
            dv, iv = g.tree.query(pts, k=kv, distance_upper_bound=g.reach)
            if kv == 1:
                dv, iv = dv[:, None], iv[:, None]
            okv = np.isfinite(dv)
            iv = np.where(okv, iv, 0)
            u = np.where(okv, dv / g.hw[iv], 2.0)
            vprof = np.clip(1.0 - u, 0.0, 1.0)
            vprof = vprof * vprof * (3.0 - 2.0 * vprof)  # smooth edges for wide basins
            valley = np.maximum(valley, np.where(okv, g.vd[iv] * vprof, 0.0).max(axis=1))
        # a valley lowers the flanks but never cuts a crest
        # the extra deepening of valley floors fades in smoothly below the flanks (a hard
        # switch at one structure value left a ledge of 0.22 x valley x amplitude)
        floor = np.clip((0.45 - structure) / 0.2, 0.0, 1.0)
        floor = floor * floor * (3.0 - 2.0 * floor)
        pattern = (
            structure * (1.0 - 0.6 * valley * (1.0 - structure) ** 0.5) - 0.22 * valley * floor
        )
        uplift = np.asarray(self.request.uplift(x.ravel(), y.ravel()), dtype=np.float64)
        gain = np.asarray(self.request.gain(x.ravel(), y.ravel()), dtype=np.float64)
        amp = self.request.detail * gain * np.maximum(uplift, 0.0)
        envelope = np.clip(structure * 2.0, 0.0, 1.0)
        rough = 0.09 * envelope * (1.0 - np.abs(self.detail(x.ravel(), y.ravel())).clip(0, 1))
        relief = amp * (pattern + rough - 0.33)
        return HeightPatch(
            relief.reshape(shape),
            structure.reshape(shape),
            valley.reshape(shape),
            system.reshape(shape),
            dist.reshape(shape),
        )

    def pass_distance(self, x, y) -> np.ndarray:
        if self.pass_tree is None:
            return np.full(np.shape(x), np.inf)
        d, _ = self.pass_tree.query(np.column_stack([np.ravel(x), np.ravel(y)]))
        return d.reshape(np.shape(x))


class ReferenceBackend:
    name = "reference"
    version = "reference-1"

    def capabilities(self) -> dict:
        return {
            "is_pmge": False,
            "note": "CPU reference fallback built into HexWorld; PMGE is not integrated",
            "sample_points": True,
            "batch": True,
            "local_patch": True,
            "hex_statistics": True,
            "policies": list(POLICIES),
            "missing_vs_pmge": [
                "erosion-carved valley networks",
                "rock-dependent landform catalogue",
                "local re-synthesis of an edited area",
                "GPU / multithreaded evaluation",
                "seamless LOD pyramid",
            ],
        }

    def sampler(self, request: MountainRequest) -> ReferenceSampler:
        if request.policy not in POLICIES:
            raise ValueError(f"unknown composition policy {request.policy!r}")
        return ReferenceSampler(request)


class PmgeSampler:
    """Heights from the PMGE library for HexWorld's structural plan."""

    def __init__(self, request: MountainRequest, library):
        from .pmge_binding import ExternalPlan

        self.request = request
        plan = request.plan
        sharp = {s.id: s.sharpness for s in plan.systems}
        ridges = [
            {
                "points": r.points,
                "crest": r.crest,
                "half_width_km": r.half_width_km,
                "rank": r.rank,
                "system": r.system,
                "sharpness": r.sharpness if r.sharpness is not None else sharp.get(r.system, 1.2),
                "plateau": r.plateau,
                "asymmetry": r.asymmetry,
            }
            for r in plan.ridges
        ]
        valleys = [
            {"points": v.points, "depth": v.depth, "half_width_km": v.half_width_km}
            for v in plan.valleys
        ]
        grid = request.constraints["grid"]
        seed = int.from_bytes(request.world_seed.encode("ascii")[:8].ljust(8, b"\0"), "little")
        self._plan = (
            ExternalPlan(
                library,
                seed,
                request.detail,
                ridges,
                valleys,
                grid,
                request.constraints["amplitude"],
                request.constraints["gain"],
                mean_offset=0.33,
            )
            if ridges
            else None
        )
        passes = [[p["x_km"], p["y_km"]] for p in plan.passes]
        self.pass_tree = cKDTree(np.array(passes)) if passes else None

    def sample_points(self, x, y) -> HeightPatch:
        x = np.asarray(x, dtype=np.float64)
        shape = x.shape
        if self._plan is None or x.size == 0:
            zero = np.zeros(shape)
            return HeightPatch(
                zero, zero, zero, np.full(shape, -1, dtype=np.int32), np.full(shape, np.inf)
            )
        s = self._plan.sample(x, y)
        dist = np.where(s["ridge_distance"] >= 0, s["ridge_distance"] / 1000.0, np.inf)
        return HeightPatch(
            s["relief"].reshape(shape),
            s["structure"].reshape(shape),
            s["valley"].reshape(shape),
            s["system"].reshape(shape),
            dist.reshape(shape),
        )

    def pass_distance(self, x, y) -> np.ndarray:
        if self.pass_tree is None:
            return np.full(np.shape(x), np.inf)
        d, _ = self.pass_tree.query(np.column_stack([np.ravel(x), np.ravel(y)]))
        return d.reshape(np.shape(x))


class PmgeBackend:
    """The external PMGE library (C++20) through its C ABI."""

    name = "pmge"

    def __init__(self, library: str | None = None):
        from .pmge_binding import ABI_VERSION, Library, find_library

        path = library or find_library()
        if not path:
            raise BackendUnavailable(
                "PMGE library not found: build it (cmake -S pmge -B pmge/build && cmake "
                "--build pmge/build) or set HEXWORLD_PMGE_LIBRARY to its path."
            )
        try:
            self.library = Library(path)
        except (OSError, AttributeError) as exc:
            raise BackendUnavailable(f"PMGE library could not be loaded: {exc}") from exc
        if self.library.abi != ABI_VERSION:
            raise BackendUnavailable(
                f"PMGE C ABI version {self.library.abi} is not supported (adapter speaks "
                f"{ABI_VERSION})"
            )
        self.version = self.library.version

    def capabilities(self) -> dict:
        return {
            "is_pmge": True,
            "abi": self.library.abi,
            "library": self.library.version,
            "path": self.library.path,
            "note": "PMGE C++ library: heights for HexWorld's structural plan",
        }

    def sampler(self, request: MountainRequest) -> PmgeSampler:
        if request.policy not in POLICIES:
            raise ValueError(f"unknown composition policy {request.policy!r}")
        for key in ("grid", "amplitude", "gain"):
            if key not in request.constraints:
                raise ValueError(f"the PMGE backend needs constraints[{key!r}]")
        return PmgeSampler(request, self.library)


def get_backend(name: str):
    if name == "reference":
        return ReferenceBackend()
    if name == "pmge":
        return PmgeBackend()
    if name == "auto":
        try:
            return PmgeBackend()
        except BackendUnavailable:
            return ReferenceBackend()
    raise ValueError(f"unknown mountain backend {name!r}")


def resolve_backend_name(name: str) -> str:
    """The backend that ``name`` stands for in this environment (for stage keys)."""
    return get_backend(name).name if name == "auto" else name


def limit_elevation(elevation, cap: float):
    """Point-wise soft ceiling: heights approach ``cap`` and never exceed it.

    Unlike rescaling the whole map, this cannot change heights far below the cap.
    """
    knee = 0.82 * cap
    over = np.maximum(elevation - knee, 0.0)
    return np.where(elevation > knee, knee + (cap - knee) * np.tanh(over / (cap - knee)), elevation)
