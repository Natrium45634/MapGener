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
from dataclasses import dataclass

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
            dx = wx[:, None] - self.centers[None, :, 0]
            dy = wy[:, None] - self.centers[None, :, 1]
            u = (ca * dx + sa * dy) / a
            v = (-sa * dx + ca * dy) * a
            val = np.max(self.radii[None, :] - np.hypot(u, v), axis=1)
            if self.rough is not None:
                val = val + self.rough(px, py)
            out[sl] = val
        return out.reshape(shape)

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
            "warp": self.warp.to_dict() if self.warp else None,
            "model_hash": self.model_hash(),
        }


def build_crust_model(config: Config, geometry: WorldGeometry, ref_x, ref_y) -> CrustModel:
    """Build the crust model; ``ref_x/ref_y`` are reference points inside the simulation area."""
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
    base = model._base(ref_x, ref_y)
    model.offset_km = -float(np.quantile(base, 1.0 - params.continental_fraction))
    return model
