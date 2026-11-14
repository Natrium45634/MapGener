"""Stage ``plates``: lithospheric plate regions as a resolution-independent model.

Approach (chosen over grid region-growing, see docs/RESEARCH.md):

* plates are cells of an **additively weighted Voronoi diagram** (Apollonius
  diagram) of sites placed in world km: ``plate(p) = argmin_i |W(p) - s_i| - w_i``;
* ``W`` is a smooth multi-octave **domain warp**, which bends the otherwise
  hyperbolic borders into irregular, naturally winding ones;
* large / medium sites are spread by dart throwing with random weights, small
  plates are inserted *on existing boundaries*, where real microplates live.

Every Apollonius cell is star-shaped about its site, hence connected, and a
one-to-one warp preserves connectivity - so plates are connected by
construction, with no destructive repair pass. The model is a pure function of
world coordinates: any hex grid, window or resolution samples the same geology,
and nothing special happens at the edge of the visible map.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

import numpy as np

from ..config import Config
from ..geometry import PROFILE_CLASS_SHARES, Bounds, WorldGeometry
from ..noise import DomainWarp
from ..rng import Stream

SIZE_CLASSES = ("large", "medium", "small")
_CHUNK = 32768


@dataclass
class PlateModel:
    sites: np.ndarray  # (M, 2) km, in warped space
    weights: np.ndarray  # (M,) km
    size_class: np.ndarray  # (M,) int8 index into SIZE_CLASSES
    halo: np.ndarray  # (M,) bool: site outside the simulation area
    warp: DomainWarp | None = None

    @property
    def count(self) -> int:
        return int(self.sites.shape[0])

    @classmethod
    def synthetic(cls, sites, weights=None) -> PlateModel:
        """Unwarped model with explicit sites - for tests and experiments."""
        sites = np.asarray(sites, dtype=np.float64).reshape(-1, 2)
        m = sites.shape[0]
        weights = np.zeros(m) if weights is None else np.asarray(weights, dtype=np.float64)
        return cls(sites, weights, np.zeros(m, dtype=np.int8), np.zeros(m, dtype=bool), None)

    def _warp(self, x, y):
        x = np.asarray(x, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        return (x, y) if self.warp is None else self.warp(x, y)

    def plate_at(self, x, y) -> np.ndarray:
        """Plate ID at world point(s), km. Works for any coordinates."""
        x = np.asarray(x, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        shape = np.broadcast(x, y).shape
        xf = np.broadcast_to(x, shape).ravel()
        yf = np.broadcast_to(y, shape).ravel()
        out = np.empty(xf.size, dtype=np.int32)
        sx, sy, w = self.sites[:, 0], self.sites[:, 1], self.weights
        for start in range(0, xf.size, _CHUNK):
            sl = slice(start, start + _CHUNK)
            wx, wy = self._warp(xf[sl], yf[sl])
            d = np.hypot(wx[:, None] - sx[None, :], wy[:, None] - sy[None, :]) - w[None, :]
            out[sl] = np.argmin(d, axis=1)
        return out.reshape(shape)

    def _pair_difference(self, x, y, a, b):
        wx, wy = self._warp(x, y)
        fa = np.hypot(wx - self.sites[a, 0], wy - self.sites[a, 1]) - self.weights[a]
        fb = np.hypot(wx - self.sites[b, 0], wy - self.sites[b, 1]) - self.weights[b]
        return fa - fb

    def boundary_normal(self, x, y, a, b, step_km: float = 1.0):
        """Unit normal of the A|B boundary near ``(x, y)``, pointing from A into B.

        It is the gradient of ``f_A - f_B`` (negative inside A, positive inside
        B), so it follows the smooth geological boundary rather than the
        zig-zag of hex edges. Returns NaN where the gradient vanishes.
        """
        x = np.asarray(x, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        a = np.asarray(a)
        b = np.asarray(b)
        h = step_km
        gx = self._pair_difference(x + h, y, a, b) - self._pair_difference(x - h, y, a, b)
        gy = self._pair_difference(x, y + h, a, b) - self._pair_difference(x, y - h, a, b)
        norm = np.hypot(gx, gy)
        with np.errstate(invalid="ignore", divide="ignore"):
            nx = np.where(norm > 1e-9, gx / norm, np.nan)
            ny = np.where(norm > 1e-9, gy / norm, np.nan)
        return nx, ny

    def model_hash(self) -> str:
        h = hashlib.sha256()
        for arr in (self.sites, self.weights, self.size_class, self.halo):
            h.update(np.ascontiguousarray(arr).tobytes())
        h.update(repr(self.warp.to_dict() if self.warp else None).encode())
        return h.hexdigest()

    def to_dict(self) -> dict:
        return {
            "kind": "additively-weighted-voronoi+domain-warp",
            "formula": "plate(p) = argmin_i |W(p) - site_i| - weight_i",
            "warp": self.warp.to_dict() if self.warp else None,
            "warp_stream": "plates/warp",
            "model_hash": self.model_hash(),
        }


def _min_distance(points: list[tuple[float, float]], x: float, y: float) -> float:
    if not points:
        return float("inf")
    arr = np.asarray(points)
    return float(np.min(np.hypot(arr[:, 0] - x, arr[:, 1] - y)))


def _throw_darts(stream, count, region: Bounds, exclude: Bounds | None, existing, min_dist):
    """Place ``count`` points with a (gracefully shrinking) minimum distance."""
    placed: list[tuple[float, float]] = []
    everything = list(existing)
    dist = min_dist
    while len(placed) < count:
        ok = False
        cand = stream.random(2 * 300).reshape(300, 2)
        for u, v in cand:
            x = region.xmin + u * region.width
            y = region.ymin + v * region.height
            if exclude is not None and exclude.contains(x, y):
                continue
            if _min_distance(everything, x, y) >= dist:
                placed.append((x, y))
                everything.append((x, y))
                ok = True
                break
        if not ok:
            dist *= 0.9
    return placed


def _insert_small(stream, count, region, exclude, sites, weights, small_sites, spacing):
    """Insert small plates near existing boundaries (in place)."""
    for _ in range(count):
        cand = stream.random(2 * 96).reshape(96, 2)
        r_target = stream.uniform(0.12, 0.3) * spacing
        cx = region.xmin + cand[:, 0] * region.width
        cy = region.ymin + cand[:, 1] * region.height
        s = np.asarray(sites)
        w = np.asarray(weights)
        dist = np.hypot(cx[:, None] - s[None, :, 0], cy[:, None] - s[None, :, 1])
        f = dist - w[None, :]
        part = np.partition(f, 1, axis=1)
        gap = part[:, 1] - part[:, 0]
        new_w = r_target - part[:, 0]
        # the new site must not swallow an existing site's own position
        survives = np.all(dist - new_w[:, None] > -w[None, :] + 0.05 * spacing, axis=1)
        valid = survives
        if exclude is not None:
            valid &= ~exclude.contains(cx, cy)
        if small_sites:
            ss = np.asarray(small_sites)
            dmin = np.min(np.hypot(cx[:, None] - ss[None, :, 0], cy[:, None] - ss[None, :, 1]), 1)
            valid &= dmin > 0.45 * spacing
        if not np.any(valid):
            continue
        k = int(np.argmin(np.where(valid, gap, np.inf)))
        sites.append((float(cx[k]), float(cy[k])))
        weights.append(float(new_w[k]))
        small_sites.append((float(cx[k]), float(cy[k])))


def build_plate_model(config: Config, geometry: WorldGeometry) -> PlateModel:
    stream = Stream(config.seed, "plates/sites")
    sim = geometry.simulation_bounds
    model = geometry.model_bounds
    n_large, n_medium, n_small = geometry.class_counts
    n_primary = n_large + n_medium
    spacing = geometry.plate_spacing_km
    ring_ratio = (model.area - sim.area) / sim.area
    shares = PROFILE_CLASS_SHARES[geometry.profile]
    large_share = shares[0] / (shares[0] + shares[1])

    min_dist = 0.62 * spacing
    sim_primary = _throw_darts(stream, n_primary, sim, None, [], min_dist)
    n_ring = int(round(n_primary * ring_ratio))
    ring_primary = _throw_darts(stream, n_ring, model, sim, sim_primary, min_dist)

    def primary_weight(is_large: bool) -> float:
        u = stream.random()
        return (0.22 + 0.23 * u) * spacing if is_large else 0.18 * u * spacing

    sim_classes = [0] * n_large + [1] * n_medium
    ring_classes = [0 if stream.random() < large_share else 1 for _ in range(n_ring)]
    sim_w = [primary_weight(c == 0) for c in sim_classes]
    ring_w = [primary_weight(c == 0) for c in ring_classes]

    sites = list(sim_primary) + list(ring_primary)
    weights = sim_w + ring_w
    base = len(sites)
    small_all: list[tuple[float, float]] = []
    _insert_small(stream, n_small, sim, None, sites, weights, small_all, spacing)
    sim_small = len(sites) - base
    _insert_small(
        stream, int(round(n_small * ring_ratio)), model, sim, sites, weights, small_all, spacing
    )
    ring_small = len(sites) - base - sim_small

    # Stable IDs: plates of the simulation area first, halo plates after them.
    order = (
        list(range(0, n_primary))
        + list(range(base, base + sim_small))
        + list(range(n_primary, base))
        + list(range(base + sim_small, base + sim_small + ring_small))
    )
    classes = sim_classes + ring_classes + [2] * (sim_small + ring_small)
    n_sim = n_primary + sim_small
    halo = np.arange(len(order)) >= n_sim

    warp = DomainWarp(
        Stream(config.seed, "plates/warp"),
        wavelength_km=config.plates.warp_wavelength_factor * spacing,
        octaves=config.plates.warp_octaves,
        strength=config.plates.warp_strength,
    )
    return PlateModel(
        sites=np.asarray(sites, dtype=np.float64)[order],
        weights=np.asarray(weights, dtype=np.float64)[order],
        size_class=np.asarray(classes, dtype=np.int8)[order],
        halo=halo,
        warp=warp,
    )
