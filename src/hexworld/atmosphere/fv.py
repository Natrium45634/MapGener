"""Finite-volume flux operators on an edge mesh (TZ 5.4 sec. 3.1-3.2).

A mesh is cells with areas ``A_i`` [m^2] and faces with lengths ``L_e`` [m] and unit normals
(east, north). Internal face ``e`` joins cells ``i -> j`` (normal from ``i`` to ``j``);
a boundary face belongs to one cell and points out of the domain. The same code serves
the square world raster (4 faces per cell) and the hex window (6 faces per cell)::

    d(A_i X_i)/dt = A_i S_i - sum_e L_e F_(i->e),     F_(i->j) = -F_(j->i)

with the first-order upwind flux ``F = u_n X_upwind`` (positive, monotone). The operator
is a sparse matrix ``D`` with ``dX/dt = -D X + b`` where ``b`` holds the imports through
the boundary (``inflow``: a specified column value, or ``zero_gradient``: the boundary
cell's own value). Internal faces cancel exactly in the domain sum (sum_i A_i D_ik = only
boundary terms), so the budget closes to round-off: storage change = sources - sinks +
import - export.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy import sparse

EAST, NORTH = 0, 1


@dataclass
class Mesh:
    area: np.ndarray  # (n,) m^2
    ei: np.ndarray  # (m,) internal faces: from cell
    ej: np.ndarray  # (m,) to cell
    length: np.ndarray  # (m,) m
    normal: np.ndarray  # (m, 2) unit (east, north) from i to j
    dist: np.ndarray  # (m,) centre distance, m
    bi: np.ndarray  # (k,) boundary faces: cell
    blength: np.ndarray  # (k,) m
    bnormal: np.ndarray  # (k, 2) outward unit normal (east, north)
    kind: str = "raster"

    @property
    def n(self) -> int:
        return int(self.area.size)

    def face_velocity(self, u_east, v_north):
        """Normal velocity on the internal and boundary faces (cell vectors averaged)."""
        u = np.asarray(u_east, dtype=np.float64).ravel()
        v = np.asarray(v_north, dtype=np.float64).ravel()
        un = 0.5 * (
            (u[self.ei] + u[self.ej]) * self.normal[:, 0]
            + (v[self.ei] + v[self.ej]) * self.normal[:, 1]
        )
        ub = u[self.bi] * self.bnormal[:, 0] + v[self.bi] * self.bnormal[:, 1]
        return un, ub

    def cfl(self, u_east, v_north, dt_s: float) -> float:
        un, ub = self.face_velocity(u_east, v_north)
        out = np.zeros(self.n)
        np.add.at(out, self.ei, np.maximum(un, 0) * self.length)
        np.add.at(out, self.ej, np.maximum(-un, 0) * self.length)
        np.add.at(out, self.bi, np.maximum(ub, 0) * self.blength)
        return float((out / self.area).max() * dt_s) if self.n else 0.0

    def divergence(self, fx_east, fy_north):
        """Discrete divergence of a cell vector field (flux per unit length), per m."""
        un, ub = self.face_velocity(fx_east, fy_north)
        out = np.zeros(self.n)
        np.add.at(out, self.ei, un * self.length)
        np.add.at(out, self.ej, -un * self.length)
        np.add.at(out, self.bi, ub * self.blength)
        return out / self.area


def raster_mesh(ny: int, nx: int, cell_m: float) -> Mesh:
    """Square cells, rows growing southward (row j+1 is south of row j)."""
    idx = np.arange(ny * nx).reshape(ny, nx)
    ex_i, ex_j = idx[:, :-1].ravel(), idx[:, 1:].ravel()  # west -> east
    ey_i, ey_j = idx[:-1].ravel(), idx[1:].ravel()  # north -> south
    ei = np.concatenate([ex_i, ey_i])
    ej = np.concatenate([ex_j, ey_j])
    normal = np.zeros((ei.size, 2))
    normal[: ex_i.size, EAST] = 1.0
    normal[ex_i.size :, NORTH] = -1.0
    b_cells, b_norm = [], []
    for cells, nrm in (
        (idx[:, -1], (1.0, 0.0)),
        (idx[:, 0], (-1.0, 0.0)),
        (idx[0], (0.0, 1.0)),
        (idx[-1], (0.0, -1.0)),
    ):
        b_cells.append(cells.ravel())
        b_norm.append(np.tile(nrm, (cells.size, 1)))
    bi = np.concatenate(b_cells)
    return Mesh(
        area=np.full(ny * nx, cell_m * cell_m),
        ei=ei,
        ej=ej,
        length=np.full(ei.size, cell_m),
        normal=normal,
        dist=np.full(ei.size, cell_m),
        bi=bi,
        blength=np.full(bi.size, cell_m),
        bnormal=np.concatenate(b_norm),
    )


def hex_mesh(layout, nbr: np.ndarray) -> Mesh:
    """The six-neighbour hex window (``nbr`` (n, 6), -1 off the window); map y is south."""
    s_m = layout.spacing_km * 1000.0
    side = s_m / math.sqrt(3.0)
    area = np.full(nbr.shape[0], 0.5 * math.sqrt(3.0) * s_m * s_m)
    x, y = layout.centers()
    x = x.ravel() * 1000.0
    y = y.ravel() * 1000.0
    ii, kk = np.nonzero(nbr >= 0)
    jj = nbr[ii, kk]
    keep = ii < jj
    ei, ej = ii[keep], jj[keep]
    dx, dy = x[ej] - x[ei], -(y[ej] - y[ei])  # north = -map y
    d = np.hypot(dx, dy)
    normal = np.stack([dx / d, dy / d], axis=1)
    # missing neighbours: boundary faces with the outward direction of that hex side
    bi_c, bk = np.nonzero(nbr < 0)
    # direction of side k: mean of the directions of the existing faces with that index
    dirs = np.zeros((6, 2))
    for k in range(6):
        sel = nbr[:, k] >= 0
        if sel.any():
            c, j = np.nonzero(sel)[0], nbr[sel, k]
            v = np.stack([x[j] - x[c], -(y[j] - y[c])], axis=1)
            v /= np.linalg.norm(v, axis=1, keepdims=True)
            m = v.mean(axis=0)
            dirs[k] = m / max(np.linalg.norm(m), 1e-12)
    return Mesh(
        area=area,
        ei=ei,
        ej=ej,
        length=np.full(ei.size, side),
        normal=normal,
        dist=d,
        bi=bi_c,
        blength=np.full(bi_c.size, side),
        bnormal=dirs[bk],
        kind="hex",
    )


@dataclass
class Transport:
    """Upwind operator ``dX/dt = -D X + b_in``; ``export`` / ``import`` weights per cell."""

    D: sparse.csr_matrix
    export_rate: np.ndarray  # (n,) 1/s: outflow through the boundary per unit X
    import_rate: np.ndarray  # (n,) 1/s: inflow through the boundary (times X_bc)


def transport(mesh: Mesh, u_east, v_north) -> Transport:
    un, ub = mesh.face_velocity(u_east, v_north)
    a = mesh.area
    pos = np.maximum(un, 0.0) * mesh.length
    neg = np.maximum(-un, 0.0) * mesh.length
    i, j = mesh.ei, mesh.ej
    rows = np.concatenate([i, j, j, i])
    cols = np.concatenate([i, i, j, j])
    vals = np.concatenate([pos / a[i], -pos / a[j], neg / a[j], -neg / a[i]])
    out_b = np.zeros(mesh.n)
    in_b = np.zeros(mesh.n)
    np.add.at(out_b, mesh.bi, np.maximum(ub, 0.0) * mesh.blength)
    np.add.at(in_b, mesh.bi, np.maximum(-ub, 0.0) * mesh.blength)
    export_rate, import_rate = out_b / a, in_b / a
    rows = np.concatenate([rows, np.arange(mesh.n)])
    cols = np.concatenate([cols, np.arange(mesh.n)])
    vals = np.concatenate([vals, export_rate])
    D = sparse.csr_matrix((vals, (rows, cols)), shape=(mesh.n, mesh.n))
    return Transport(D, export_rate, import_rate)


def advect_explicit(
    mesh: Mesh, x, u_east, v_north, seconds: float, cfl_target: float = 0.45, x_bc=None
) -> tuple[np.ndarray, dict]:
    """Explicit upwind transport over ``seconds`` with CFL-limited substeps (double buffer).

    ``x_bc`` (n,) column value of the inflow through the boundary; ``None`` = zero
    gradient. Returns the new field and a ledger in units of X * m^2.
    """
    tr = transport(mesh, u_east, v_north)
    c = mesh.cfl(u_east, v_north, 1.0)
    nsub = max(1, int(math.ceil(c * seconds / cfl_target))) if c > 0 else 1
    dt = seconds / nsub
    x = np.asarray(x, dtype=np.float64).copy()
    a = mesh.area
    led = {
        "initial": float((a * x).sum()),
        "import": 0.0,
        "export": 0.0,
        "substeps": nsub,
        "cfl": c * dt,
    }
    for _ in range(nsub):
        xb = x if x_bc is None else np.asarray(x_bc, dtype=np.float64)
        imp = tr.import_rate * xb
        led["import"] += float((a * imp).sum() * dt)
        led["export"] += float((a * tr.export_rate * x).sum() * dt)
        x = x + dt * (imp - tr.D @ x)  # one old state for all faces
        if (x < -1e-12 * max(1.0, float(np.abs(x).max()))).any():
            raise FloatingPointError("negative column after an upwind step (CFL)")
        x = np.maximum(x, 0.0)
    led["final"] = float((a * x).sum())
    led["residual"] = led["final"] - led["initial"] - led["import"] + led["export"]
    return x, led


def mm_to_m3(mm, area_km2):
    """Water depth over an area as a volume: 1 mm on 1 km^2 = 1000 m^3 (N01, N02)."""
    return np.asarray(mm, dtype=np.float64) * 1e-3 * np.asarray(area_km2, dtype=np.float64) * 1e6


def mm_per_day_to_m3_s(mm_day, area_km2):
    """Runoff depth rate over a catchment as a discharge (N03)."""
    return mm_to_m3(mm_day, area_km2) / 86400.0


def budget_residual(p_mm, e_mm, d_storage_mm, net_import_mm):
    """``dW = E - P + NetImport``: returns the residual ``dW - (E - P + NetImport)``."""
    return d_storage_mm - (e_mm - p_mm + net_import_mm)


class BudgetError(ValueError):
    pass


def validate_budget(p_mm, e_mm, d_storage_mm, net_import_mm, tol_mm: float = 1e-6):
    """Reject (never silently correct) a water budget that does not close (N11)."""
    r = np.asarray(budget_residual(p_mm, e_mm, d_storage_mm, net_import_mm), dtype=float)
    if np.any(np.abs(r) > tol_mm):
        worst = float(np.max(np.abs(r)))
        raise BudgetError(f"water budget does not close: residual {worst:.3g} mm")
    return r
