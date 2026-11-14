"""World geometry: physical bounds, the hex layout and plate-count resolution.

Four things are deliberately separate:

* ``simulation_bounds_km`` - the physical area the geology is generated for;
* ``viewport_bounds_km``   - the visible rectangle (a window into that area);
* ``width / height``       - how many hexes discretise the viewport;
* ``hex_spacing_km``       - the physical size of one hex.

The pixel size of a hex is a render-only parameter and never appears here.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from . import hexgrid
from .config import Config

#: Mean plate area (km^2) per profile, counted over all size classes.
PROFILE_MEAN_AREA_KM2 = {"few": 6.0e7, "earthlike": 2.6e7, "fragmented": 1.0e7}
#: Share of (large, medium, small) plates per profile.
PROFILE_CLASS_SHARES = {
    "few": (0.50, 0.35, 0.15),
    "earthlike": (0.30, 0.35, 0.35),
    "fragmented": (0.18, 0.37, 0.45),
}
_QUANTUM_KM = 1000.0


@dataclass(frozen=True)
class Bounds:
    xmin: float
    ymin: float
    xmax: float
    ymax: float

    @property
    def width(self) -> float:
        return self.xmax - self.xmin

    @property
    def height(self) -> float:
        return self.ymax - self.ymin

    @property
    def area(self) -> float:
        return self.width * self.height

    def expand(self, margin: float) -> Bounds:
        return Bounds(
            self.xmin - margin, self.ymin - margin, self.xmax + margin, self.ymax + margin
        )

    def contains(self, x, y):
        x = np.asarray(x)
        y = np.asarray(y)
        return (x >= self.xmin) & (x <= self.xmax) & (y >= self.ymin) & (y <= self.ymax)

    def contains_bounds(self, other: Bounds, tol: float = 1e-6) -> bool:
        return (
            other.xmin >= self.xmin - tol
            and other.ymin >= self.ymin - tol
            and other.xmax <= self.xmax + tol
            and other.ymax <= self.ymax + tol
        )

    def as_list(self) -> list[float]:
        return [self.xmin, self.ymin, self.xmax, self.ymax]


@dataclass(frozen=True)
class HexLayout:
    """Flat-top odd-q hex grid placed in world coordinates (km)."""

    width: int
    height: int
    spacing_km: float
    x0: float  # world x of the left edge of the viewport rectangle
    y0: float  # world y of the top edge of the viewport rectangle

    @property
    def radius(self) -> float:
        return self.spacing_km / hexgrid.SQRT3

    @property
    def hex_area_km2(self) -> float:
        return 1.5 * hexgrid.SQRT3 * self.radius**2

    @property
    def bounds(self) -> Bounds:
        r = self.radius
        return Bounds(
            self.x0,
            self.y0,
            self.x0 + r * (1.5 * (self.width - 1) + 2.0),
            self.y0 + self.spacing_km * (self.height + 0.5),
        )

    def center(self, col, row):
        col = np.asarray(col)
        row = np.asarray(row)
        r = self.radius
        x = self.x0 + r + 1.5 * r * col
        y = self.y0 + 0.5 * self.spacing_km + self.spacing_km * (row + 0.5 * (col & 1))
        return x, y

    def centers(self):
        """World coordinates of all hex centres as two ``(height, width)`` arrays."""
        cols = np.arange(self.width, dtype=np.int64)[None, :]
        rows = np.arange(self.height, dtype=np.int64)[:, None]
        x, y = self.center(cols, rows)
        return np.broadcast_to(x, (self.height, self.width)).copy(), np.asarray(y)

    def world_to_offset(self, x, y, with_edge: bool = False):
        """Hex ``(col, row)`` containing world point(s); may be outside the grid."""
        r = self.radius
        fq, fr = hexgrid.plane_to_fractional_axial(
            np.asarray(x, dtype=np.float64) - (self.x0 + r),
            np.asarray(y, dtype=np.float64) - (self.y0 + 0.5 * self.spacing_km),
            r,
        )
        q, rr, edge = hexgrid.cube_round(fq, fr)
        col, row = hexgrid.axial_to_offset(q, rr)
        return (col, row, edge) if with_edge else (col, row)

    def inside(self, col, row):
        col = np.asarray(col)
        row = np.asarray(row)
        return (col >= 0) & (col < self.width) & (row >= 0) & (row < self.height)

    def padded(self, pad: int) -> HexLayout:
        """Same grid extended by ``pad`` hexes on every side (``pad`` must be even)."""
        if pad % 2:
            raise ValueError("pad must be even to preserve odd-q parity")
        return HexLayout(
            self.width + 2 * pad,
            self.height + 2 * pad,
            self.spacing_km,
            self.x0 - 1.5 * self.radius * pad,
            self.y0 - self.spacing_km * pad,
        )


@dataclass(frozen=True)
class WorldGeometry:
    mode: str
    simulation_bounds: Bounds
    viewport_bounds: Bounds
    layout: HexLayout
    plate_count: int
    profile: str
    #: Mean spacing of primary (large + medium) plate sites, km.
    plate_spacing_km: float
    #: Width of the ring around the simulation area that also receives geology.
    halo_km: float
    class_counts: tuple[int, int, int]
    warnings: tuple[str, ...] = ()

    @property
    def model_bounds(self) -> Bounds:
        return self.simulation_bounds.expand(self.halo_km)

    def to_dict(self) -> dict:
        return {
            "mode": self.mode,
            "simulation_bounds_km": self.simulation_bounds.as_list(),
            "viewport_bounds_km": self.viewport_bounds.as_list(),
            "model_bounds_km": self.model_bounds.as_list(),
            "width_hex": self.layout.width,
            "height_hex": self.layout.height,
            "hex_spacing_km": self.layout.spacing_km,
            "hex_radius_km": self.layout.radius,
            "hex_area_km2": self.layout.hex_area_km2,
            "plate_count": self.plate_count,
            "plate_class_counts": {
                "large": self.class_counts[0],
                "medium": self.class_counts[1],
                "small": self.class_counts[2],
            },
            "plate_spacing_km": self.plate_spacing_km,
            "halo_km": self.halo_km,
        }


def _class_counts(n: int, profile: str) -> tuple[int, int, int]:
    large_share, _, small_share = PROFILE_CLASS_SHARES[profile]
    small = int(round(n * small_share))
    small = max(0, min(small, n - min(n, 2)))
    primary = n - small
    large = max(1, min(primary, int(round(n * large_share))))
    return large, primary - large, small


def resolve_geometry(config: Config) -> WorldGeometry:
    layout = HexLayout(
        config.width,
        config.height,
        float(config.hex_spacing_km),
        float(config.viewport_origin_km[0]),
        float(config.viewport_origin_km[1]),
    )
    viewport = layout.bounds
    warnings: list[str] = []
    mean_area = PROFILE_MEAN_AREA_KM2[config.plates.profile]

    if config.simulation_bounds_km is not None:
        sim = Bounds(*[float(v) for v in config.simulation_bounds_km])
        if not sim.contains_bounds(viewport):
            raise ValueError(
                "viewport_bounds_km must lie inside simulation_bounds_km "
                f"(viewport {viewport.as_list()}, simulation {sim.as_list()})"
            )
    elif config.mode == "world":
        sim = viewport
    else:
        # Regional: geology is generated well beyond the window. Bounds are snapped
        # to a coarse quantum so small changes of resolution keep the same world.
        buffer = max(0.35 * max(viewport.width, viewport.height), 1.1 * math.sqrt(mean_area))
        cx = round(0.5 * (viewport.xmin + viewport.xmax) / (_QUANTUM_KM / 2)) * (_QUANTUM_KM / 2)
        cy = round(0.5 * (viewport.ymin + viewport.ymax) / (_QUANTUM_KM / 2)) * (_QUANTUM_KM / 2)
        half_w = math.ceil((0.5 * viewport.width + buffer) / _QUANTUM_KM + 0.25) * _QUANTUM_KM
        half_h = math.ceil((0.5 * viewport.height + buffer) / _QUANTUM_KM + 0.25) * _QUANTUM_KM
        sim = Bounds(cx - half_w, cy - half_h, cx + half_w, cy + half_h)

    if config.plates.count == "auto":
        count = int(min(400, max(3, round(sim.area / mean_area))))
    else:
        count = int(config.plates.count)
        per_plate = sim.area / count
        if per_plate < 2.0e5:
            warnings.append(
                f"plates.count={count}: only {per_plate:,.0f} km^2 per plate - far smaller "
                "than real microplates; boundaries will be unrealistically dense."
            )
        elif per_plate > 2.5e8:
            warnings.append(
                f"plates.count={count}: {per_plate:,.0f} km^2 per plate exceeds any real "
                "plate; the window may show no boundaries at all."
            )

    large, medium, small = _class_counts(count, config.plates.profile)
    spacing = math.sqrt(sim.area / max(1, large + medium))
    halo = 1.6 * spacing

    if max(viewport.width, viewport.height) < 0.35 * spacing:
        warnings.append(
            "The viewport is much smaller than a typical plate "
            f"(~{spacing:,.0f} km): it will probably show one or two plates only."
        )
    if layout.spacing_km > 0.12 * spacing:
        warnings.append(
            f"hex_spacing_km={layout.spacing_km:g} is coarse relative to the plate "
            f"spacing (~{spacing:,.0f} km); small plates may be poorly resolved."
        )
    return WorldGeometry(
        mode=config.mode,
        simulation_bounds=sim,
        viewport_bounds=viewport,
        layout=layout,
        plate_count=count,
        profile=config.plates.profile,
        plate_spacing_km=spacing,
        halo_km=halo,
        class_counts=(large, medium, small),
        warnings=tuple(warnings),
    )
