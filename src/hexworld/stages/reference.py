"""World-level reference raster.

A square-cell raster over the model area (simulation area + halo), independent
of the hex viewport. Plate metadata (area, centroid, crust shares, neighbours)
is measured here, so it describes the *world*, not whichever window is shown.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..geometry import WorldGeometry


@dataclass
class ReferenceRaster:
    x: np.ndarray  # (ny, nx) cell-centre world x, km
    y: np.ndarray
    cell_km: float
    in_sim: np.ndarray  # (ny, nx) bool
    plate: np.ndarray | None = None  # (ny, nx) int32
    crust: np.ndarray | None = None  # (ny, nx) uint8

    @property
    def cell_area_km2(self) -> float:
        return self.cell_km**2


def build_reference_grid(geometry: WorldGeometry, cells: int) -> ReferenceRaster:
    b = geometry.model_bounds
    cell = max(b.width, b.height) / cells
    nx = max(2, int(np.ceil(b.width / cell)))
    ny = max(2, int(np.ceil(b.height / cell)))
    xs = b.xmin + (np.arange(nx) + 0.5) * cell
    ys = b.ymin + (np.arange(ny) + 0.5) * cell
    x, y = np.meshgrid(xs, ys)
    return ReferenceRaster(x=x, y=y, cell_km=cell, in_sim=geometry.simulation_bounds.contains(x, y))


def reference_adjacency(ref: ReferenceRaster, count: int) -> list[list[int]]:
    """Plates sharing a border inside the simulation area (4-neighbourhood)."""
    p = ref.plate
    pairs = set()
    for a, b, m in (
        (p[:, :-1], p[:, 1:], ref.in_sim[:, :-1] | ref.in_sim[:, 1:]),
        (p[:-1, :], p[1:, :], ref.in_sim[:-1, :] | ref.in_sim[1:, :]),
    ):
        sel = (a != b) & m
        lo = np.minimum(a[sel], b[sel]).astype(np.int64)
        hi = np.maximum(a[sel], b[sel]).astype(np.int64)
        for code in np.unique(lo * count + hi):
            pairs.add((int(code // count), int(code % count)))
    out: list[set[int]] = [set() for _ in range(count)]
    for a, b in pairs:
        out[a].add(b)
        out[b].add(a)
    return [sorted(s) for s in out]
