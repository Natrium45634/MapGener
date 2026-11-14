"""Stage ``boundary_edges``: graph of hex edges shared by different plates.

Sign conventions (world axes: +x east, +y south; all speeds in mm/yr)
---------------------------------------------------------------------
For every boundary edge the two hexes are ordered so that ``plate_a < plate_b``.

* ``normal`` ``n``  - unit vector across the boundary, pointing **from A into B**;
* ``tangent`` ``t`` - ``(-n_y, n_x)``: ``n`` turned 90 degrees clockwise as seen on a
  map with north up (east -> south);
* ``v_rel = v_B - v_A`` evaluated at the edge midpoint;
* ``vn = v_rel . n``:  ``vn < 0`` plates approach (**convergent**),
  ``vn > 0`` plates separate (**divergent**);
* ``vt = v_rel . t``:  ``vt > 0`` right-lateral (dextral), ``vt < 0`` left-lateral.

Only hexes sharing an *edge* are neighbours; touching at a vertex does not count.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, fields

import numpy as np

from .. import hexgrid
from ..config import BoundaryParams
from ..geometry import HexLayout
from .motion import PlateMotion
from .plates import PlateModel
from .raster import neighbor_index

TYPE_NONE = 0
TYPE_CONVERGENT = 1
TYPE_DIVERGENT = 2
TYPE_TRANSFORM = 3
TYPE_QUIET = 4
TYPE_NAMES = ("none", "convergent", "divergent", "transform", "quiet")

SUBDUCTING_NONE = 0
SUBDUCTING_A = 1
SUBDUCTING_B = 2


@dataclass
class BoundaryEdges:
    """Structure-of-arrays graph; one entry per shared hex edge (see module docstring)."""

    col_a: np.ndarray  # int32, hex on the plate_a side
    row_a: np.ndarray
    col_b: np.ndarray  # int32, hex on the plate_b side
    row_b: np.ndarray
    direction: np.ndarray  # uint8, hex direction index from A to B (0..5)
    plate_a: np.ndarray  # int32, plate_a < plate_b
    plate_b: np.ndarray
    mid_x: np.ndarray  # float64 km, edge midpoint
    mid_y: np.ndarray
    normal_x: np.ndarray  # float32, unit, A -> B
    normal_y: np.ndarray
    vrel_x: np.ndarray  # float32 mm/yr, v_B - v_A
    vrel_y: np.ndarray
    vn: np.ndarray  # float32 mm/yr, <0 convergent, >0 divergent
    vt: np.ndarray  # float32 mm/yr, >0 dextral, <0 sinistral
    boundary_type: np.ndarray  # uint8, TYPE_*
    intensity: np.ndarray  # float32 in [0, 1]
    crust_a: np.ndarray  # uint8, 0 oceanic / 1 continental
    crust_b: np.ndarray
    subducting: np.ndarray  # uint8, SUBDUCTING_* polarity hint
    length_km: float = 0.0  # length of every edge (= hex circumradius)

    def __len__(self) -> int:
        return int(self.plate_a.size)

    def arrays(self) -> dict[str, np.ndarray]:
        return {f.name: getattr(self, f.name) for f in fields(self) if f.name != "length_km"}

    def select(self, mask) -> BoundaryEdges:
        data = {k: v[mask] for k, v in self.arrays().items()}
        return BoundaryEdges(**data, length_km=self.length_km)

    def tangent(self):
        return -self.normal_y, self.normal_x

    def segment_endpoints(self):
        """World endpoints ``(x1, y1, x2, y2)`` of every edge, for drawing."""
        tx, ty = self.tangent_of_hex_edge()
        h = 0.5 * self.length_km
        return (self.mid_x - h * tx, self.mid_y - h * ty, self.mid_x + h * tx, self.mid_y + h * ty)

    def tangent_of_hex_edge(self):
        units = hexgrid.direction_unit_vectors()[self.direction]
        return -units[:, 1], units[:, 0]


def classify(vn, vt, params: BoundaryParams):
    """Classify relative motion; returns ``(type uint8, intensity float32)``."""
    vn = np.asarray(vn, dtype=np.float64)
    vt = np.asarray(vt, dtype=np.float64)
    speed = np.hypot(vn, vt)
    ratio = np.divide(np.abs(vn), speed, out=np.zeros_like(speed), where=speed > 0)
    normal_dominated = ratio >= math.sin(math.radians(params.obliquity_deg))
    kind = np.full(speed.shape, TYPE_TRANSFORM, dtype=np.uint8)
    kind[normal_dominated & (vn < 0)] = TYPE_CONVERGENT
    kind[normal_dominated & (vn > 0)] = TYPE_DIVERGENT
    quiet = speed < params.quiet_threshold_mm_yr
    kind[quiet] = TYPE_QUIET
    dominant = np.where(kind == TYPE_TRANSFORM, np.abs(vt), np.abs(vn))
    dominant = np.where(quiet, speed, dominant)
    intensity = np.clip(dominant / params.reference_speed_mm_yr, 0.0, 1.0)
    return kind, intensity.astype(np.float32)


def extract_boundaries(
    plate_id: np.ndarray,
    crust_type: np.ndarray,
    layout: HexLayout,
    motion: PlateMotion,
    params: BoundaryParams,
    model: PlateModel | None = None,
) -> BoundaryEdges:
    height, width = plate_id.shape
    flat = plate_id.ravel()
    cols = np.broadcast_to(np.arange(width, dtype=np.int64)[None, :], plate_id.shape).ravel()
    rows = np.broadcast_to(np.arange(height, dtype=np.int64)[:, None], plate_id.shape).ravel()

    src_l, dst_l, dir_l = [], [], []
    for d in hexgrid.HALF_DIRECTIONS:
        nb, valid = neighbor_index(layout, d)
        nb = nb.ravel()
        diff = valid.ravel() & (flat[nb] != flat)
        src = np.nonzero(diff)[0]
        src_l.append(src)
        dst_l.append(nb[src])
        dir_l.append(np.full(src.size, d, dtype=np.uint8))
    src = np.concatenate(src_l)
    dst = np.concatenate(dst_l)
    direction = np.concatenate(dir_l)

    swap = flat[src] > flat[dst]
    a = np.where(swap, dst, src)
    b = np.where(swap, src, dst)
    direction = np.where(swap, (direction + 3) % 6, direction).astype(np.uint8)
    order = np.lexsort((b, a))  # canonical, deterministic edge order
    a, b, direction = a[order], b[order], direction[order]

    plate_a = flat[a].astype(np.int32)
    plate_b = flat[b].astype(np.int32)
    ax, ay = layout.center(cols[a], rows[a])
    bx, by = layout.center(cols[b], rows[b])
    mid_x = 0.5 * (ax + bx)
    mid_y = 0.5 * (ay + by)
    hex_n = hexgrid.direction_unit_vectors()[direction]
    nx, ny = hex_n[:, 0].copy(), hex_n[:, 1].copy()
    if model is not None and a.size:
        gx, gy = model.boundary_normal(mid_x, mid_y, plate_a, plate_b)
        with np.errstate(invalid="ignore"):
            good = np.isfinite(gx) & np.isfinite(gy) & (gx * nx + gy * ny > 0.0)
        nx = np.where(good, gx, nx)
        ny = np.where(good, gy, ny)

    vax, vay = motion.velocity_at(plate_a, mid_x, mid_y)
    vbx, vby = motion.velocity_at(plate_b, mid_x, mid_y)
    rx = vbx - vax
    ry = vby - vay
    vn = rx * nx + ry * ny
    vt = rx * (-ny) + ry * nx
    kind, intensity = classify(vn, vt, params)

    crust_flat = crust_type.ravel()
    crust_a = crust_flat[a].astype(np.uint8)
    crust_b = crust_flat[b].astype(np.uint8)
    # Polarity hint for convergent edges that involve oceanic crust: oceanic goes
    # under continental; ocean-ocean - the plate advancing faster on the boundary.
    adv_a = vax * nx + vay * ny  # A's own speed towards B
    adv_b = -(vbx * nx + vby * ny)  # B's own speed towards A
    subducting = np.full(a.size, SUBDUCTING_NONE, dtype=np.uint8)
    conv = kind == TYPE_CONVERGENT
    subducting[conv & (crust_a == 0) & (crust_b == 1)] = SUBDUCTING_A
    subducting[conv & (crust_a == 1) & (crust_b == 0)] = SUBDUCTING_B
    both = conv & (crust_a == 0) & (crust_b == 0)
    subducting[both & (adv_a >= adv_b)] = SUBDUCTING_A
    subducting[both & (adv_a < adv_b)] = SUBDUCTING_B

    return BoundaryEdges(
        col_a=cols[a].astype(np.int32),
        row_a=rows[a].astype(np.int32),
        col_b=cols[b].astype(np.int32),
        row_b=rows[b].astype(np.int32),
        direction=direction,
        plate_a=plate_a,
        plate_b=plate_b,
        mid_x=mid_x,
        mid_y=mid_y,
        normal_x=nx.astype(np.float32),
        normal_y=ny.astype(np.float32),
        vrel_x=rx.astype(np.float32),
        vrel_y=ry.astype(np.float32),
        vn=vn.astype(np.float32),
        vt=vt.astype(np.float32),
        boundary_type=kind,
        intensity=intensity,
        crust_a=crust_a,
        crust_b=crust_b,
        subducting=subducting,
        length_km=layout.radius,
    )


def summarize_pairs(edges: BoundaryEdges) -> list[dict]:
    """Per plate-pair statistics of the boundary graph."""
    if len(edges) == 0:
        return []
    out = []
    key = edges.plate_a.astype(np.int64) * (int(edges.plate_b.max()) + 1) + edges.plate_b
    for code in np.unique(key):
        m = key == code
        kinds = edges.boundary_type[m]
        n = int(m.sum())
        out.append(
            {
                "plate_a": int(edges.plate_a[m][0]),
                "plate_b": int(edges.plate_b[m][0]),
                "edges": n,
                "length_km": round(n * edges.length_km, 3),
                "type_share": {
                    TYPE_NAMES[t]: round(float(np.mean(kinds == t)), 4) for t in range(1, 5)
                },
                "mean_vn_mm_yr": round(float(edges.vn[m].mean()), 4),
                "mean_vt_mm_yr": round(float(edges.vt[m].mean()), 4),
                "mean_intensity": round(float(edges.intensity[m].mean()), 4),
            }
        )
    return out
