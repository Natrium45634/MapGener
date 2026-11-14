"""Hexagon math: flat-top hexes, odd-q offset storage, axial/cube computation.

Conventions (after Red Blob Games, "Hexagonal Grids"):

* flat-top hexagons; circumradius ``R``; neighbour centre spacing ``sqrt(3) * R``;
* storage is a rectangular matrix ``[row, col]`` in **odd-q** offset layout
  (odd columns are shoved half a hex towards +y);
* axial ``(q, r)``: ``q = col``, ``r = row - (col - (col & 1)) // 2``;
  cube ``(x, y, z) = (q, -q - r, r)``;
* plane axes: ``+x`` east, ``+y`` south (same direction as growing rows).
"""

from __future__ import annotations

import numpy as np

SQRT3 = 1.7320508075688772

#: Axial neighbour offsets ``(dq, dr)``. Direction ``d`` and ``(d + 3) % 6`` are opposite.
AXIAL_DIRECTIONS: tuple[tuple[int, int], ...] = (
    (1, 0),
    (1, -1),
    (0, -1),
    (-1, 0),
    (-1, 1),
    (0, 1),
)
#: Three directions that enumerate every undirected hex edge exactly once.
HALF_DIRECTIONS: tuple[int, ...] = (0, 1, 5)


def offset_to_axial(col, row):
    col = np.asarray(col)
    row = np.asarray(row)
    return col, row - (col - (col & 1)) // 2


def axial_to_offset(q, r):
    q = np.asarray(q)
    r = np.asarray(r)
    return q, r + (q - (q & 1)) // 2


def axial_to_cube(q, r):
    q = np.asarray(q)
    r = np.asarray(r)
    return q, -q - r, r


def axial_distance(q1, r1, q2, r2):
    dq = np.asarray(q1) - np.asarray(q2)
    dr = np.asarray(r1) - np.asarray(r2)
    return (np.abs(dq) + np.abs(dr) + np.abs(dq + dr)) // 2


def offset_distance(col1, row1, col2, row2):
    return axial_distance(*offset_to_axial(col1, row1), *offset_to_axial(col2, row2))


def offset_neighbor(col, row, direction: int):
    """Offset coordinates of the neighbour in ``direction`` (0..5)."""
    dq, dr = AXIAL_DIRECTIONS[direction]
    q, r = offset_to_axial(col, row)
    return axial_to_offset(q + dq, r + dr)


def axial_to_plane(q, r, radius: float):
    """Centre of hex ``(q, r)`` relative to the centre of hex ``(0, 0)``."""
    q = np.asarray(q, dtype=np.float64)
    r = np.asarray(r, dtype=np.float64)
    return radius * 1.5 * q, radius * SQRT3 * (r + 0.5 * q)


def plane_to_fractional_axial(x, y, radius: float):
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    q = (2.0 / 3.0) * x / radius
    r = (-x / 3.0 + (SQRT3 / 3.0) * y) / radius
    return q, r


def cube_round(fq, fr):
    """Round fractional axial coordinates to the containing hex.

    Returns ``(q, r, edge)`` where ``edge`` in [0, 1] is 0 at the hex centre and
    1 on its outline (useful for drawing hex borders).
    """
    fx = fq
    fz = fr
    fy = -fx - fz
    rx = np.rint(fx)
    ry = np.rint(fy)
    rz = np.rint(fz)
    dx = np.abs(rx - fx)
    dy = np.abs(ry - fy)
    dz = np.abs(rz - fz)
    fix_x = (dx > dy) & (dx > dz)
    fix_z = ~fix_x & (dz > dy)
    rx = np.where(fix_x, -ry - rz, rx)
    rz = np.where(fix_z, -rx - ry, rz)
    ry = -rx - rz
    ex = fx - rx
    ey = fy - ry
    ez = fz - rz
    edge = np.maximum(np.maximum(np.abs(ex - ey), np.abs(ey - ez)), np.abs(ez - ex))
    return rx.astype(np.int64), rz.astype(np.int64), edge


def direction_unit_vectors() -> np.ndarray:
    """Unit vectors (6, 2) from a hex centre towards each neighbour."""
    out = np.empty((6, 2))
    for d, (dq, dr) in enumerate(AXIAL_DIRECTIONS):
        x, y = axial_to_plane(dq, dr, 1.0)
        n = np.hypot(x, y)
        out[d] = (x / n, y / n)
    return out
