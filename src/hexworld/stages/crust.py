"""Stage ``crust``: continental / oceanic crust field, independent of plates.

The field is a function of world coordinates: a union of warped elliptical
"nuclei" (cratonic cores grouped into a few masses) plus mild long-wave
irregularity. ``continentality_km`` is a signed pseudo-distance (positive on
continental crust), ``crust_type`` is its sign. One plate may carry both types.

This is *not* a land/sea mask: the shoreline appears only after elevation and
sea level exist (v1.2+).
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass, field

import numpy as np

from ..config import Config
from ..geometry import WorldGeometry
from ..noise import DomainWarp, FractalNoise
from ..rng import Stream
from .plates import _throw_darts

CRUST_OCEANIC = 0
CRUST_CONTINENTAL = 1
CRUST_NAMES = ("oceanic", "continental")
_CHUNK = 32768


@dataclass
class CrustModel:
    centers: np.ndarray  # (K, 2) km
    radii: np.ndarray  # (K,) km
    stretch: np.ndarray  # (K,) axis ratio >= 1
    angle: np.ndarray  # (K,) radians
    warp: DomainWarp | None
    rough: FractalNoise | None
    #: Calibrated so the simulation area holds the requested continental share.
    offset_km: float = 0.0
    #: How the masses were laid out (layout, degree of assembly, size law ...).
    layout_info: dict = field(default_factory=dict)

    def _base(self, x, y) -> np.ndarray:
        x = np.asarray(x, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        shape = np.broadcast(x, y).shape
        xf = np.broadcast_to(x, shape).ravel()
        yf = np.broadcast_to(y, shape).ravel()
        out = np.empty(xf.size)
        ca, sa = np.cos(self.angle), np.sin(self.angle)
        a = np.sqrt(self.stretch)
        for start in range(0, xf.size, _CHUNK):
            sl = slice(start, start + _CHUNK)
            px, py = xf[sl], yf[sl]
            wx, wy = (px, py) if self.warp is None else self.warp(px, py)
            val = self._nearest_nucleus(wx, wy, ca, sa, a)
            if self.rough is not None:
                val = val + self.rough(px, py)
            out[sl] = val
        return out.reshape(shape)

    def _nearest_nucleus(self, wx, wy, ca, sa, a) -> np.ndarray:
        """``max_i(r_i - |E_i(p - c_i)|)`` evaluated in compact blocks of points.

        Inside a block only the nuclei that can still win are evaluated: with the
        elliptical metric ``|d| / a_i <= |E_i d| <= |d| * a_i``, a nucleus whose best case
        over the block's bounding box is below another nucleus's worst case can never be the
        maximum. The surviving nuclei are evaluated with the full formula, so the result is
        exactly that of evaluating all nuclei.
        """
        n = wx.size
        out = np.empty(n)
        cx, cy, r = self.centers[:, 0], self.centers[:, 1], self.radii
        # compact blocks: sort the points along a coarse spatial grid
        span = max(float(np.ptp(wx)), float(np.ptp(wy)), 1e-9)
        cell = span / 16.0
        gx = np.floor((wx - wx.min()) / cell).astype(np.int64)
        gy = np.floor((wy - wy.min()) / cell).astype(np.int64)
        order = np.argsort(gy * 64 + gx, kind="stable")
        block = 2048
        for b0 in range(0, n, block):
            idx = order[b0 : b0 + block]
            bx, by = wx[idx], wy[idx]
            x0, x1, y0, y1 = bx.min(), bx.max(), by.min(), by.max()
            # distance from each centre to the block box: nearest and farthest corner
            ddx = np.maximum(np.maximum(x0 - cx, cx - x1), 0.0)
            ddy = np.maximum(np.maximum(y0 - cy, cy - y1), 0.0)
            dmin = np.hypot(ddx, ddy)
            fdx = np.maximum(np.abs(cx - x0), np.abs(cx - x1))
            fdy = np.maximum(np.abs(cy - y0), np.abs(cy - y1))
            dmax = np.hypot(fdx, fdy)
            best_case = r - dmin / a
            worst_case = r - dmax * a
            keep = best_case >= worst_case.max() - 1e-9
            k = np.nonzero(keep)[0]
            dx = bx[:, None] - cx[None, k]
            dy = by[:, None] - cy[None, k]
            u = (ca[k] * dx + sa[k] * dy) / a[k]
            v = (-sa[k] * dx + ca[k] * dy) * a[k]
            out[idx] = np.max(r[None, k] - np.hypot(u, v), axis=1)
        return out

    def continentality(self, x, y) -> np.ndarray:
        """Signed pseudo-distance in km: > 0 continental, < 0 oceanic."""
        return self._base(x, y) + self.offset_km

    def crust_at(self, x, y) -> np.ndarray:
        return (self.continentality(x, y) > 0.0).astype(np.uint8)

    def model_hash(self) -> str:
        h = hashlib.sha256()
        for arr in (self.centers, self.radii, self.stretch, self.angle):
            h.update(np.ascontiguousarray(arr).tobytes())
        h.update(repr(float(self.offset_km)).encode())
        return h.hexdigest()

    def to_dict(self) -> dict:
        return {
            "kind": "warped-elliptical-nuclei",
            "formula": "continentality(p) = max_i(r_i - |E_i(W(p) - c_i)|) + rough(p) + offset",
            "nuclei": int(self.centers.shape[0]),
            "offset_km": float(self.offset_km),
            "layout": self.layout_info,
            "warp": self.warp.to_dict() if self.warp else None,
            "model_hash": self.model_hash(),
        }


def _uniform_nuclei(config: Config, geometry: WorldGeometry):
    """Layout of generator versions <= 3.0: masses spread evenly, similar sizes."""
    params = config.crust
    stream = Stream(config.seed, "crust/nuclei")
    region = geometry.model_bounds
    n_masses = max(
        1, int(round(params.continental_fraction * region.area / params.mean_continent_area_km2))
    )
    mass_spacing = math.sqrt(region.area / n_masses)
    mass_centers = _throw_darts(stream, n_masses, region, None, [], 0.55 * mass_spacing)

    centers, radii, stretch, angle = [], [], [], []
    sigma = 0.65
    for cx, cy in mass_centers:
        area = params.mean_continent_area_km2 * math.exp(sigma * stream.normal() - 0.5 * sigma**2)
        radius = math.sqrt(area / math.pi)
        lobes = 2 + stream.integer(4)
        trend = stream.uniform(0.0, math.pi)  # shared elongation trend of the mass
        for k in range(lobes):
            if k == 0:
                ox = oy = 0.0
                r = radius * stream.uniform(0.6, 0.85)
            else:
                along = stream.normal() * 0.75 * radius
                across = stream.normal() * 0.35 * radius
                ox = along * math.cos(trend) - across * math.sin(trend)
                oy = along * math.sin(trend) + across * math.cos(trend)
                r = radius * stream.uniform(0.35, 0.7)
            centers.append((cx + ox, cy + oy))
            radii.append(r)
            stretch.append(stream.uniform(1.0, 2.0))
            angle.append(trend + stream.uniform(-0.6, 0.6))
    return centers, radii, stretch, angle, {"layout": "uniform", "masses": n_masses}


#: Typical spacing of former supercontinents ("hemisphere scale"), km.
_DOMAIN_KM = (15000.0, 23000.0)


def _clustered_nuclei(config: Config, geometry: WorldGeometry):
    """Continents as the remains of a few supercontinents.

    Real continents are not spread evenly: they are fragments of former
    supercontinents, so they come in groups separated by wide ocean basins, and
    their sizes are very unequal. Three things are drawn from the seed:

    * the **degree of assembly** (0 = fragments drifted far apart, 1 = still one
      assembled mass) - a bell-shaped draw, so both extremes are possible but rare;
    * the **size law**: areas fall off with rank as ``rank ** -alpha``;
    * where the former supercontinents were (their spacing is hemisphere-scale).
    """
    params = config.crust
    stream = Stream(config.seed, "crust/clusters")
    region = geometry.model_bounds
    u = stream.random(6)
    if params.assembly == "auto":
        assembly = float(np.clip(0.5 + 0.5 * (u[0] + u[1] + u[2] - 1.5) / 1.5 * 1.25, 0.0, 1.0))
    else:
        assembly = float(params.assembly)
    alpha = 0.8 + 0.7 * u[3]
    domain = _DOMAIN_KM[0] + (_DOMAIN_KM[1] - _DOMAIN_KM[0]) * u[4]

    total_area = params.continental_fraction * region.area
    n_masses = max(1, int(round(total_area / params.mean_continent_area_km2)))
    n_parents = max(1, int(round(region.area / domain**2)))
    parents = _throw_darts(stream, n_parents, region, None, [], 0.7 * domain)
    # unequal supercontinents: some carry much more crust than others
    parent_weight = np.array([0.35 + stream.random() ** 2 * 1.65 for _ in parents])
    rank = np.arange(1, n_masses + 1, dtype=np.float64)
    share = rank**-alpha
    areas = total_area * share / share.sum()
    # deal the masses out by size so every former supercontinent gets large ones first
    owner = []
    load = np.zeros(len(parents))
    for k in range(n_masses):
        if stream.random() < 0.1 and k > 2:
            owner.append(-1)  # a stray microcontinent, far from any group
            continue
        j = int(np.argmin(load / parent_weight))
        owner.append(j)
        load[j] += areas[k]

    centers, radii, stretch, angle = [], [], [], []
    placed: list[tuple[float, float, float]] = []
    for k in range(n_masses):
        radius = math.sqrt(areas[k] / math.pi)
        j = owner[k]
        best = None
        for attempt in range(60):
            v = stream.random(2)
            if j < 0:
                x = region.xmin + v[0] * region.width
                y = region.ymin + v[1] * region.height
            else:
                packed = math.sqrt(load[j] / math.pi)  # radius if the group were one mass
                reach = 0.95 * packed + (1.0 - assembly) * 0.46 * domain
                r = reach * math.sqrt(v[0])
                x = parents[j][0] + r * math.cos(2.0 * math.pi * v[1])
                y = parents[j][1] + r * math.sin(2.0 * math.pi * v[1])
            # fragments may touch (assembled) but must not sit on top of each other
            need = (0.55 + 0.75 * (1.0 - assembly)) * (1.0 - attempt / 90.0)
            ok = all(math.hypot(x - px, y - py) >= need * (radius + pr) for px, py, pr in placed)
            best = (x, y)
            if ok:
                break
        cx, cy = best
        placed.append((cx, cy, radius))
        lobes = 2 + stream.integer(3) + int(min(4, radius / 1100.0))
        trend = stream.uniform(0.0, math.pi)
        for lobe in range(lobes):
            if lobe == 0:
                ox = oy = 0.0
                r = radius * stream.uniform(0.6, 0.85)
            else:
                along = stream.normal() * 0.8 * radius
                across = stream.normal() * 0.38 * radius
                ox = along * math.cos(trend) - across * math.sin(trend)
                oy = along * math.sin(trend) + across * math.cos(trend)
                r = radius * stream.uniform(0.3, 0.68)
            centers.append((cx + ox, cy + oy))
            radii.append(r)
            stretch.append(stream.uniform(1.0, 2.2))
            angle.append(trend + stream.uniform(-0.6, 0.6))
    info = {
        "layout": "clustered",
        "assembly": round(assembly, 4),
        "size_exponent": round(alpha, 4),
        "domain_km": round(domain, 1),
        "former_supercontinents": len(parents),
        "masses": n_masses,
        "largest_mass_share": round(float(areas[0] / total_area), 4),
    }
    return centers, radii, stretch, angle, info


def build_crust_model(config: Config, geometry: WorldGeometry, ref_x, ref_y) -> CrustModel:
    """Build the crust model; ``ref_x/ref_y`` are reference points inside the simulation area."""
    params = config.crust
    if params.layout == "uniform":
        centers, radii, stretch, angle, info = _uniform_nuclei(config, geometry)
    else:
        centers, radii, stretch, angle, info = _clustered_nuclei(config, geometry)

    warp = DomainWarp(
        Stream(config.seed, "crust/warp"),
        wavelength_km=params.warp_wavelength_km,
        octaves=params.warp_octaves,
        strength=params.warp_strength,
    )
    mean_radius = math.sqrt(params.mean_continent_area_km2 / math.pi)
    rough = FractalNoise(
        Stream(config.seed, "crust/rough"),
        wavelengths_km=[3200.0, 1600.0, 800.0],
        amplitudes=[params.roughness * mean_radius * g for g in (1.0, 0.5, 0.25)],
    )
    model = CrustModel(
        centers=np.asarray(centers, dtype=np.float64),
        radii=np.asarray(radii, dtype=np.float64),
        stretch=np.asarray(stretch, dtype=np.float64),
        angle=np.asarray(angle, dtype=np.float64),
        warp=warp,
        rough=rough,
    )
    model.layout_info = info
    base = model._base(ref_x, ref_y)
    model.offset_km = -float(np.quantile(base, 1.0 - params.continental_fraction))
    return model
