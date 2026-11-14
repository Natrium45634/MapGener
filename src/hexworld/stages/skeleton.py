"""v2 stage ``world_tectonics``: plate boundaries traced in world coordinates.

v1.0 stores its boundary graph on the hex grid of one window. Continents and
relief must not depend on a window or a resolution, so v2 traces the same
plate boundaries on a fixed world-level raster over the model area and snaps
every sample onto the continuous boundary of the plate model. The result is a
set of world points with normal, relative velocity and regime - the input of
every tectonic term of v2.

Nothing of v1.0 is modified: the plate model and plate motion are only read.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.spatial import cKDTree

from ..config import BoundaryParams
from ..geometry import WorldGeometry
from .boundaries import classify
from .motion import PlateMotion
from .plates import PlateModel
from .reference import build_reference_grid

_CHUNK = 100_000


@dataclass
class WorldSkeleton:
    x: np.ndarray  # (ny, nx) cell centres, km
    y: np.ndarray
    cell_km: float
    in_sim: np.ndarray
    plate: np.ndarray  # (ny, nx) int32
    # boundary samples (N,)
    bx: np.ndarray
    by: np.ndarray
    nx: np.ndarray  # unit normal from plate_a into plate_b
    ny: np.ndarray
    plate_a: np.ndarray
    plate_b: np.ndarray
    vn: np.ndarray  # mm/yr, < 0 convergent
    vt: np.ndarray
    adv_a: np.ndarray  # plate_a's own speed towards the boundary, mm/yr
    adv_b: np.ndarray
    kind: np.ndarray  # uint8, boundaries.TYPE_*
    intensity: np.ndarray
    tree: cKDTree | None = None

    @property
    def x0(self) -> float:
        return float(self.x[0, 0])

    @property
    def y0(self) -> float:
        return float(self.y[0, 0])

    def grid_dict(self) -> dict:
        return {
            "x0_km": self.x0,
            "y0_km": self.y0,
            "cell_km": self.cell_km,
            "nx": int(self.x.shape[1]),
            "ny": int(self.x.shape[0]),
            "note": "cell (j, i) is centred at (x0 + i*cell, y0 + j*cell)",
        }


def build_skeleton(
    geometry: WorldGeometry,
    plate_model: PlateModel,
    motion: PlateMotion,
    params: BoundaryParams,
    cells: int,
) -> WorldSkeleton:
    ref = build_reference_grid(geometry, cells)
    plate = plate_model.plate_at(ref.x, ref.y)
    xs, ys, pa, pb = [], [], [], []
    for a, b, xa, ya, xb, yb in (
        (plate[:, :-1], plate[:, 1:], ref.x[:, :-1], ref.y[:, :-1], ref.x[:, 1:], ref.y[:, 1:]),
        (plate[:-1, :], plate[1:, :], ref.x[:-1, :], ref.y[:-1, :], ref.x[1:, :], ref.y[1:, :]),
    ):
        m = a != b
        xs.append(0.5 * (xa[m] + xb[m]))
        ys.append(0.5 * (ya[m] + yb[m]))
        pa.append(np.minimum(a[m], b[m]))
        pb.append(np.maximum(a[m], b[m]))
    bx = np.concatenate(xs)
    by = np.concatenate(ys)
    plate_a = np.concatenate(pa).astype(np.int32)
    plate_b = np.concatenate(pb).astype(np.int32)

    # snap the samples onto the continuous boundary f_a - f_b = 0 (two Newton steps)
    h = 1.0
    for _ in range(2):
        g = plate_model._pair_difference(bx, by, plate_a, plate_b)
        gx = (
            plate_model._pair_difference(bx + h, by, plate_a, plate_b)
            - plate_model._pair_difference(bx - h, by, plate_a, plate_b)
        ) / (2 * h)
        gy = (
            plate_model._pair_difference(bx, by + h, plate_a, plate_b)
            - plate_model._pair_difference(bx, by - h, plate_a, plate_b)
        ) / (2 * h)
        g2 = np.maximum(gx * gx + gy * gy, 1e-6)
        step = np.clip(g / g2, -ref.cell_km, ref.cell_km)
        scale = np.minimum(1.0, ref.cell_km / np.maximum(np.abs(step) * np.sqrt(g2), 1e-9))
        bx = bx - step * gx * scale
        by = by - step * gy * scale
    nx, ny = plate_model.boundary_normal(bx, by, plate_a, plate_b)
    ok = np.isfinite(nx) & np.isfinite(ny)
    bx, by, nx, ny, plate_a, plate_b = (v[ok] for v in (bx, by, nx, ny, plate_a, plate_b))

    vax, vay = motion.velocity_at(plate_a, bx, by)
    vbx, vby = motion.velocity_at(plate_b, bx, by)
    rx, ry = vbx - vax, vby - vay
    vn = rx * nx + ry * ny
    vt = -rx * ny + ry * nx
    kind, intensity = classify(vn, vt, params)
    return WorldSkeleton(
        x=ref.x,
        y=ref.y,
        cell_km=ref.cell_km,
        in_sim=ref.in_sim,
        plate=plate,
        bx=bx,
        by=by,
        nx=nx,
        ny=ny,
        plate_a=plate_a,
        plate_b=plate_b,
        vn=vn,
        vt=vt,
        adv_a=vax * nx + vay * ny,
        adv_b=-(vbx * nx + vby * ny),
        kind=kind,
        intensity=intensity.astype(np.float64),
        tree=cKDTree(np.column_stack([bx, by])) if bx.size else None,
    )


def accumulate(skeleton: WorldSkeleton, x, y, contribution, max_km: float, outputs: int, k=6):
    """Blend per-boundary-sample contributions at world points.

    ``contribution(idx, d, side_b)`` receives sample indices ``(n, k)``, their
    distances in km and whether the point lies on the plate_b side, and returns
    a list of ``outputs`` arrays ``(n, k)``. Nearby samples are averaged with
    weights that depend only on world geometry, so the result is a function of
    the world position alone.
    """
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    shape = x.shape
    xf, yf = x.ravel(), y.ravel()
    out = [np.zeros(xf.size) for _ in range(outputs)]
    if skeleton.tree is None:
        return [o.reshape(shape) for o in out]
    k = min(k, skeleton.bx.size)
    blend = skeleton.cell_km
    for start in range(0, xf.size, _CHUNK):
        sl = slice(start, start + _CHUNK)
        px, py = xf[sl], yf[sl]
        d, idx = skeleton.tree.query(np.column_stack([px, py]), k=k, distance_upper_bound=max_km)
        if k == 1:
            d, idx = d[:, None], idx[:, None]
        found = np.isfinite(d)
        if not found.any():
            continue
        idx = np.where(found, idx, 0)
        dd = np.where(found, d, max_km)
        weight = np.where(found, np.exp(-0.5 * ((dd - dd[:, :1]) / blend) ** 2), 0.0)
        wsum = weight.sum(axis=1)
        wsum[wsum == 0] = 1.0
        side_b = (px[:, None] - skeleton.bx[idx]) * skeleton.nx[idx] + (
            py[:, None] - skeleton.by[idx]
        ) * skeleton.ny[idx] >= 0
        for o, value in zip(out, contribution(idx, dd, side_b), strict=True):
            o[sl] = (value * weight).sum(axis=1) / wsum
    return [o.reshape(shape) for o in out]
