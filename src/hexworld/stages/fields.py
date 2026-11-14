"""Stage ``tectonic_fields``: diagnostic hex fields derived from the boundary graph.

These are **signals for future modules** (history, elevation, volcanoes ...),
not a geophysical simulation and not terrain. All potentials are in [0, 1].

* ``boundary_distance_km`` - distance from the hex centre to the nearest plate boundary;
* ``uplift``     - collision belts (continent-continent, both sides, wide) and the
  overriding side of subduction zones;
* ``rifting``    - divergent boundaries;
* ``subduction`` - convergent boundaries where at least one side is oceanic;
* ``volcanism``  - arc on the *overriding* plate, offset from the trench, plus
  divergent-boundary volcanism. Continental collision and pure transform faults
  produce none: convergence alone is not a volcano source;
* ``seismicity`` - every active boundary, wider on the overriding side.
"""

from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree

from ..config import FieldParams
from ..geometry import HexLayout
from .boundaries import (
    SUBDUCTING_A,
    SUBDUCTING_B,
    TYPE_CONVERGENT,
    TYPE_DIVERGENT,
    TYPE_NONE,
    TYPE_TRANSFORM,
    BoundaryEdges,
)

FIELD_NAMES = ("uplift", "rifting", "subduction", "volcanism", "seismicity")
_K = 8


def _gauss(d, width):
    return np.exp(-0.5 * (d / width) ** 2)


def hex_boundary_type(edges: BoundaryEdges, shape) -> np.ndarray:
    """Per hex: type of its most intense boundary edge (0 if the hex has none)."""
    height, width = shape
    out = np.zeros(height * width, dtype=np.uint8)
    idx = np.concatenate(
        [
            edges.row_a.astype(np.int64) * width + edges.col_a,
            edges.row_b.astype(np.int64) * width + edges.col_b,
        ]
    )
    intensity = np.concatenate([edges.intensity, edges.intensity])
    kind = np.concatenate([edges.boundary_type, edges.boundary_type])
    # ascending intensity: with repeated indices the last (strongest) write wins
    order = np.argsort(intensity, kind="stable")
    out[idx[order]] = kind[order]
    return out.reshape(shape)


def compute_fields(
    edges: BoundaryEdges, layout: HexLayout, params: FieldParams
) -> dict[str, np.ndarray]:
    shape = (layout.height, layout.width)
    n = layout.height * layout.width
    result = {name: np.zeros(n, dtype=np.float32) for name in FIELD_NAMES}
    distance = np.full(n, np.float32(params.max_influence_km), dtype=np.float32)
    if len(edges) == 0:
        out = {k: v.reshape(shape) for k, v in result.items()}
        out["boundary_distance_km"] = distance.reshape(shape)
        return out

    tree = cKDTree(np.column_stack([edges.mid_x, edges.mid_y]))
    hx, hy = layout.centers()
    hx = hx.ravel()
    hy = hy.ravel()
    k = min(_K, len(edges))
    blend = layout.spacing_km

    kind = edges.boundary_type
    inten = edges.intensity.astype(np.float64)
    conv = np.where(kind == TYPE_CONVERGENT, inten, 0.0)
    div = np.where(kind == TYPE_DIVERGENT, inten, 0.0)
    shear = np.where(kind == TYPE_TRANSFORM, inten, 0.0)
    has_sub = edges.subducting != 0
    collision = conv * ~has_sub  # continent-continent convergence
    subd = conv * has_sub
    nxe = edges.normal_x.astype(np.float64)
    nye = edges.normal_y.astype(np.float64)

    chunk = 200_000
    for start in range(0, n, chunk):
        sl = slice(start, min(n, start + chunk))
        px, py = hx[sl], hy[sl]
        d, idx = tree.query(
            np.column_stack([px, py]), k=k, distance_upper_bound=params.max_influence_km
        )
        if k == 1:
            d = d[:, None]
            idx = idx[:, None]
        found = np.isfinite(d)
        idx = np.where(found, idx, 0)
        dd = np.where(found, d, params.max_influence_km)
        dmin = dd[:, 0]
        distance[sl] = dmin.astype(np.float32)
        weight = np.where(found, np.exp(-0.5 * ((dd - dmin[:, None]) / blend) ** 2), 0.0)
        wsum = weight.sum(axis=1)
        wsum[wsum == 0] = 1.0

        # which side of the edge the hex is on: +1 => B side, -1 => A side
        side_b = (px[:, None] - edges.mid_x[idx]) * nxe[idx] + (
            py[:, None] - edges.mid_y[idx]
        ) * nye[idx] >= 0
        sub = edges.subducting[idx]
        on_subducting = ((sub == SUBDUCTING_A) & ~side_b) | ((sub == SUBDUCTING_B) & side_b)
        on_overriding = (sub != 0) & ~on_subducting

        c_col = collision[idx]
        c_sub = subd[idx]
        c_div = div[idx]
        c_shear = shear[idx]

        uplift = (
            c_col * _gauss(dd, params.collision_width_km)
            + c_sub * on_overriding * 0.75 * _gauss(dd, params.subduction_width_km)
            + c_sub * on_subducting * 0.12 * _gauss(dd, params.trench_width_km)
            + c_shear * 0.15 * _gauss(dd, params.transform_width_km)
        )
        rifting = c_div * _gauss(dd, params.rift_width_km)
        subduction = c_sub * (
            on_overriding * _gauss(dd, params.subduction_width_km)
            + on_subducting * _gauss(dd, params.trench_width_km)
        )
        volcanism = c_sub * on_overriding * _gauss(
            dd - params.arc_offset_km, params.arc_width_km
        ) + 0.6 * c_div * _gauss(dd, params.ridge_volcanism_width_km)
        seismicity = np.maximum.reduce(
            [
                c_col * _gauss(dd, params.collision_width_km),
                c_sub
                * np.where(
                    on_overriding,
                    _gauss(dd, params.subduction_width_km),
                    _gauss(dd, params.trench_width_km),
                ),
                0.6 * c_div * _gauss(dd, params.transform_width_km),
                0.8 * c_shear * _gauss(dd, params.transform_width_km),
            ]
        )
        for name, value in (
            ("uplift", uplift),
            ("rifting", rifting),
            ("subduction", subduction),
            ("volcanism", volcanism),
            ("seismicity", seismicity),
        ):
            blended = (value * weight).sum(axis=1) / wsum
            result[name][sl] = np.clip(blended, 0.0, 1.0).astype(np.float32)

    out = {name: arr.reshape(shape) for name, arr in result.items()}
    out["boundary_distance_km"] = distance.reshape(shape)
    return out


__all__ = ["FIELD_NAMES", "compute_fields", "hex_boundary_type", "TYPE_NONE"]
