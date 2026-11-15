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
from .plan import RANK_SPUR, MountainPlan

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


class ReferenceSampler:
    """Analytic sampler: relief at any world point, so windows and chunks never seam."""

    def __init__(self, request: MountainRequest):
        self.request = request
        plan = request.plan
        xs, ys, crest, hw, sharp, sysid, spur = [], [], [], [], [], [], []
        sharp_by_system = {s.id: s.sharpness for s in plan.systems}
        for r in plan.ridges:
            n = len(r.points)
            xs.append(r.points[:, 0])
            ys.append(r.points[:, 1])
            crest.append(r.crest)
            hw.append(np.full(n, r.half_width_km))
            sharp.append(np.full(n, sharp_by_system.get(r.system, 1.2)))
            sysid.append(np.full(n, r.system))
            spur.append(np.full(n, r.rank == RANK_SPUR))
        self.empty = not xs
        if self.empty:
            return
        self.px, self.py = np.concatenate(xs), np.concatenate(ys)
        self.crest, self.hw = np.concatenate(crest), np.concatenate(hw)
        self.sharp, self.sysid = np.concatenate(sharp), np.concatenate(sysid).astype(np.int32)
        self.tree = cKDTree(np.column_stack([self.px, self.py]))
        self.reach = float(self.hw.max())
        vx, vy, vd, vw = [], [], [], []
        for v in plan.valleys:
            vx.append(v.points[:, 0])
            vy.append(v.points[:, 1])
            vd.append(v.depth)
            vw.append(np.full(len(v.points), v.half_width_km))
        self.vtree = None
        if vx:
            self.vx, self.vy = np.concatenate(vx), np.concatenate(vy)
            self.vd, self.vw = np.concatenate(vd), np.concatenate(vw)
            self.vtree = cKDTree(np.column_stack([self.vx, self.vy]))
        pts = np.array([[p["x_km"], p["y_km"]] for p in plan.passes]) if plan.passes else None
        self.pass_tree = cKDTree(pts) if pts is not None else None
        stream = Stream(request.world_seed, "mountains/v3/detail")
        self.detail = FractalNoise(stream, [46.0, 23.0, 11.5], [1.0, 0.55, 0.3])

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
        k = min(8, self.px.size)
        d, idx = self.tree.query(pts, k=k, distance_upper_bound=self.reach)
        if k == 1:
            d, idx = d[:, None], idx[:, None]
        ok = np.isfinite(d)
        idx = np.where(ok, idx, 0)
        u = np.where(ok, d / self.hw[idx], 2.0)
        prof = np.clip(1.0 - u, 0.0, 1.0) ** self.sharp[idx]
        val = np.where(ok, self.crest[idx] * prof, 0.0)
        best = np.argmax(val, axis=1)
        rows = np.arange(n)
        structure = val[rows, best]
        system = np.where(structure > 0, self.sysid[idx[rows, best]], -1).astype(np.int32)
        dist = np.where(ok[:, 0], d[:, 0], np.inf)
        if self.vtree is not None:
            kv = min(4, self.vx.size)
            dv, iv = self.vtree.query(pts, k=kv, distance_upper_bound=float(self.vw.max()))
            if kv == 1:
                dv, iv = dv[:, None], iv[:, None]
            okv = np.isfinite(dv)
            iv = np.where(okv, iv, 0)
            vprof = np.clip(1.0 - np.where(okv, dv / self.vw[iv], 2.0), 0.0, 1.0)
            valley = np.where(okv, self.vd[iv] * vprof, 0.0).max(axis=1)
        # a valley lowers the flanks but never cuts a crest
        pattern = structure * (1.0 - 0.6 * valley * (1.0 - structure) ** 0.5) - 0.22 * valley * (
            structure < 0.35
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
                "sharpness": sharp.get(r.system, 1.2),
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
