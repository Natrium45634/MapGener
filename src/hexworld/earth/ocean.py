"""Wind-driven ocean circulation, mixed-layer heat, upwelling and sea ice (world raster).

**Grid.** The ocean lives on the world climate raster: cells of ``cell_km``, 4 faces each,
in a north-up frame inside this module (rows of the raster grow southward; arrays are
flipped on entry). Only *connected* sea is ocean: the sea components that touch the border
of the model area or hold at least 5 % of the sea; a closed depression below sea level is
not ocean (it stays land for the climate, the v4 hydrology decides about lakes).

**Circulation** (Stommel 1948, ``[ADAPT]``) for the transport streamfunction ``psi`` (m^3/s)
on the cell corners::

    r lap(psi) + beta d(psi)/dx = curl(tau) / rho_w,     psi = 0 on coasts and borders
    tau = rho_a Cd |U| U                                   (season wind of the climate)

``r = beta * dx`` puts the western boundary layer on the grid (numerical, documented).
Face transports are corner differences of ``psi``, so the discrete divergence of every
cell is exactly zero and no water crosses a coast. Ekman transport
``M = (tau_y, -tau_x) / (rho_w f)`` (|f| >= f(5 deg), tapered to zero within 5 deg of the
equator: the equatorial regime is only approximate) is added in the mixed layer; its
divergence is balanced by vertical exchange (upwelling of cooler subsurface water,
downwelling of mixed-layer water), which is booked in the heat ledger.

**Heat** of the mixed layer (``h`` m, ``E = rho c h T``), backward Euler, two steps per
month, per-season operators::

    dE/dt = - div(rho c F_ml T) + div(rho c K h grad T) + W_up rho c (T_sub - T) + Q
    Q = lambda (T_A(lat, month) - T) - 0.6 S(m) (alpha_eff - 0.06)

``T_A`` is the atmosphere's equilibrium temperature over an ice-free ocean from the
planet's seasonal energy-balance model (:mod:`hexworld.earth.ebm`: insolation, OLR,
meridional transport), ``lambda`` = 35 W m^-2 K^-1 the linearised air-sea exchange
(sensible + latent + long-wave; Haney 1971). The second term is the local ice-albedo
loss of absorbed sunlight. TZ 5.4.1: the former restoring to an Earth SST table with a
hidden zonal-mean calibration is gone (``docs/v5_4_1/defect_sst_root_cause.md``). The
gyre moves heat in a layer of ``HEAT_LAYER_M`` of the ``WIND_LAYER_M`` deep flow.

**Sea ice** (zero-layer thermodynamics, ``[ADAPT]`` from CICE/Icepack ideas): water cannot
cool below ``T_f = -1.8 degC``; the deficit freezes ``dV = E_def / (rho_i L_f)``; a warming
flux melts ice first. Concentration ``c = 1 - exp(-h_i / 0.5 m)``; albedo
``(1 - c) 0.06 + c a_ice`` (0.65 cold, 0.5 melting); cooling through ice is insulated by
``1 / (1 + h_i / 0.5 m)``. Ice drifts with the surface current + 2 % of the wind,
conservatively (no ice crosses a coast).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
from scipy import ndimage
from scipy.sparse import coo_matrix, csc_matrix, identity
from scipy.sparse.linalg import splu

RHO_W = 1025.0
RHO_A = 1.2
C_W = 4000.0
RHO_ICE = 917.0
L_F = 334_000.0
T_FREEZE = -1.8
#: [K] linearised air-sea heat exchange with the atmosphere's equilibrium temperature
#: (Haney 1971: 30-50 W m^-2 K^-1)
GAMMA = 35.0
WIND_LAYER_M = 800.0
HEAT_LAYER_M = 400.0
#: Free-drift speed limit of sea ice (internal ice stress), m/s.
ICE_DRIFT_MAX = 0.5
H_ICE_SCALE = 0.5
ALPHA_WATER = 0.06
SECONDS_PER_MONTH = 365.25 * 86400.0 / 12.0
#: Earth zonal SST climatology (restoring target), degC at |lat| = 0, 10, ..., 90.
_SST_LAT = np.arange(0.0, 91.0, 10.0)
_SST_T = np.array([27.5, 27.0, 25.0, 21.5, 15.5, 9.0, 3.5, 0.0, -1.5, -1.8])


def sst_target(lat_deg):
    return np.interp(np.abs(np.asarray(lat_deg, dtype=np.float64)), _SST_LAT, _SST_T)


@dataclass
class OceanResult:
    ocean: np.ndarray  # (ny, nx) bool connected ocean (raster orientation)
    basin_id: np.ndarray  # int32, -1 off the ocean
    sst_monthly: np.ndarray  # (12, ny, nx) degC
    sea_ice_monthly: np.ndarray  # (12, ny, nx) concentration
    ice_thickness_monthly: np.ndarray  # (12, ny, nx) m, mean over the ice-covered part
    current_u: np.ndarray  # (4, ny, nx) m/s east, mixed-layer
    current_v: np.ndarray  # (4, ny, nx) m/s north
    psi: np.ndarray  # (4, ny+1, nx+1) Sv on the corners
    upwelling_m_day: np.ndarray  # (4, ny, nx)
    coastal_upwelling: np.ndarray  # (ny, nx) bool: cells of strong coastal upwelling
    heat_content_j_m2: np.ndarray  # (12, ny, nx)
    salinity_psu: np.ndarray  # (ny, nx) qualitative
    ice_type: np.ndarray  # (ny, nx) 0 open, 1 seasonal, 2 multiyear, 3 fast
    diagnostics: dict = field(default_factory=dict)


def ocean_mask(sea: np.ndarray, min_share: float = 0.05):
    """Connected ocean (touching the border or large) and the basin labels."""
    lab, n = ndimage.label(sea)
    if n == 0:
        return np.zeros_like(sea, dtype=bool), np.full(sea.shape, -1, np.int32)
    sizes = np.bincount(lab.ravel(), minlength=n + 1)
    border = np.unique(np.concatenate([lab[0], lab[-1], lab[:, 0], lab[:, -1]]))
    total = sizes[1:].sum()
    keep = np.zeros(n + 1, dtype=bool)
    keep[border] = True
    keep[sizes >= min_share * total] = True
    keep[0] = False
    ocean = keep[lab]
    ids = np.full(n + 1, -1, dtype=np.int32)
    ids[np.nonzero(keep)[0]] = np.arange(int(keep.sum()), dtype=np.int32)
    return ocean, ids[lab]


class Circulation:
    """Stommel operator on the corner nodes of the ocean cells, factorised once."""

    def __init__(self, ocean_n: np.ndarray, lat_n: np.ndarray, dx: float, planet):
        ny, nx = ocean_n.shape
        self.shape = (ny, nx)
        self.dx = dx
        o = ocean_n
        # a node is free when all four cells around it are ocean
        free = np.zeros((ny + 1, nx + 1), dtype=bool)
        free[1:-1, 1:-1] = o[:-1, :-1] & o[:-1, 1:] & o[1:, :-1] & o[1:, 1:]
        self.free = free
        idx = -np.ones(free.shape, dtype=np.int64)
        idx[free] = np.arange(int(free.sum()))
        self.idx = idx
        lat_nodes = np.interp(np.arange(ny + 1) - 0.5, np.arange(ny), lat_n[:, 0])
        self.lat_nodes = np.repeat(lat_nodes[:, None], nx + 1, axis=1)
        beta = planet.beta(self.lat_nodes)
        self.beta = beta
        # bottom friction: Stommel layer delta = r / beta of at least two cells / 150 km,
        # with beta floored at its 45-degree value so polar gyres do not blow up
        beta_ref = np.maximum(beta, planet.beta(45.0))
        self.r = beta_ref * max(2.0 * dx, 1.5e5)
        jj, ii = np.nonzero(free)
        k = idx[jj, ii]
        rows, cols, vals = [k], [k], [-4.0 * self.r[jj, ii] / dx**2]
        for dj, di, bcoef in ((0, 1, 1.0), (0, -1, -1.0), (1, 0, 0.0), (-1, 0, 0.0)):
            nb = idx[jj + dj, ii + di]
            ok = nb >= 0
            coef = self.r[jj, ii] / dx**2 + bcoef * beta[jj, ii] / (2.0 * dx)
            rows.append(k[ok])
            cols.append(nb[ok])
            vals.append(coef[ok])
        n = int(free.sum())
        self.n = n
        if n:
            a = coo_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
                           shape=(n, n)).tocsc()  # fmt: skip
            self.lu = splu(a)
        else:
            self.lu = None

    def solve(self, tau_x: np.ndarray, tau_y: np.ndarray) -> np.ndarray:
        """psi (m^3/s) on the corners from the cell wind stress (north-up frame)."""
        ny, nx = self.shape
        psi = np.zeros((ny + 1, nx + 1))
        if self.lu is None:
            return psi
        dx = self.dx
        cy = np.zeros((ny + 1, nx + 1))
        # curl at the interior corners from the four cells around each
        tx, ty = tau_x, tau_y
        dty_dx = ((ty[:-1, 1:] + ty[1:, 1:]) - (ty[:-1, :-1] + ty[1:, :-1])) / (2.0 * dx)
        dtx_dy = ((tx[1:, :-1] + tx[1:, 1:]) - (tx[:-1, :-1] + tx[:-1, 1:])) / (2.0 * dx)
        cy[1:-1, 1:-1] = dty_dx - dtx_dy
        rhs = cy[self.free] / RHO_W
        psi[self.free] = self.lu.solve(rhs)
        return psi

    @staticmethod
    def face_transport(psi: np.ndarray):
        """East-face and north-face transports (m^3/s) of every cell; div = 0 exactly."""
        fx = psi[:-1, 1:] - psi[1:, 1:]  # east face of cell (j, i)
        fy = psi[1:, 1:] - psi[1:, :-1]  # north face of cell (j, i)
        return fx, fy


#: [K] linear drag of the wind-driven surface layer, 1/s (slab response, TZ 5.4.1 sec. 5.3)
R_SLAB = 1.0e-5


def slab_current(tau_x, tau_y, f, r=R_SLAB, h=50.0, rho=RHO_W):
    """Wind-driven slab current (m/s, east / north) with linear drag (TZ 5.4.1 sec. 5.3).

    ``r u - f v = tau_x / (rho h)``, ``f u + r v = tau_y / (rho h)``: to the right of the
    stress for f > 0, to the left for f < 0, and down the stress (drag-limited) at f = 0.
    """
    den = rho * h * (r * r + np.asarray(f, dtype=np.float64) ** 2)
    u = (r * tau_x + f * tau_y) / den
    v = (r * tau_y - f * tau_x) / den
    return u, v


def ekman_transport(tau_x, tau_y, lat, planet, r=R_SLAB):
    """Wind-driven surface transport (m^2/s) east and north: ``h x`` the slab current.

    The drag regularises the equator (no ``1/f``, the sign of f is kept in the south);
    off the equator it equals the Ekman transport ``(tau_y, -tau_x) / (rho f)`` within
    ``(r/f)^2``.
    """
    u, v = slab_current(tau_x, tau_y, planet.coriolis(lat), r=r, h=1.0)
    return u, v


def mixed_layer_warming_k(q_w_m2, h_m, seconds, rho=RHO_W, cp=3990.0):
    """Temperature change of a mixed layer under a net flux (N05)."""
    return q_w_m2 * seconds / (rho * cp * h_m)


def exchange(t_a, t_b, v_a, v_b, q_m3):
    """Equal, opposite volume exchange ``q_m3`` between two well-mixed boxes (N04 / N06)."""
    ta = (t_a * (v_a - q_m3) + t_b * q_m3) / v_a
    tb = (t_b * (v_b - q_m3) + t_a * q_m3) / v_b
    return ta, tb


def freeze_volume_m(energy_j_m2, rho_ice=RHO_ICE, lf=L_F):
    """Ice thickness grown by a heat deficit at the freezing point (N09, isothermal ice)."""
    return np.maximum(-np.asarray(energy_j_m2, dtype=np.float64), 0.0) / (rho_ice * lf)


def effective_albedo(c, alpha_water=ALPHA_WATER, alpha_ice=0.6):
    """Area-weighted surface albedo of a partly ice-covered cell (N10)."""
    return (1.0 - c) * alpha_water + c * alpha_ice


def _faces_from_cells(mx, my, ocean, dx):
    """Face transports (m^3/s) of a cell-centred transport per width, zero on land faces."""
    ex = 0.5 * (mx[:, :-1] + mx[:, 1:]) * dx
    ex = np.where(ocean[:, :-1] & ocean[:, 1:], ex, 0.0)
    ny_ = 0.5 * (my[:-1] + my[1:]) * dx
    ny_ = np.where(ocean[:-1] & ocean[1:], ny_, 0.0)
    fx = np.zeros(mx.shape)
    fy = np.zeros(my.shape)
    fx[:, :-1] = ex
    fy[:-1] = ny_
    return fx, fy


def _divergence(fx, fy):
    d = fx.copy()
    d[:, 1:] -= fx[:, :-1]
    d += fy
    d[1:] -= fy[:-1]
    return d


def heat_operator(fx, fy, ocean, idx, vol, k_diff, dx, h):
    """Sparse matrix of advection (upwind) + diffusion, units 1/s on T (cells of idx)."""
    rows, cols, vals = [], [], []
    ny, nx = ocean.shape
    diag = np.zeros(int(ocean.sum()))

    def link(a_mask, ia, ib, flux):
        # flux > 0 from a to b; upwind: out of a at T_a, into b at T_a
        a, b = ia[a_mask], ib[a_mask]
        f = flux[a_mask]
        pos = f > 0
        # T_a leaves a, enters b
        np.add.at(diag, a[pos], -f[pos] / vol[a[pos]])
        rows.append(b[pos])
        cols.append(a[pos])
        vals.append(f[pos] / vol[b[pos]])
        neg = ~pos
        np.add.at(diag, b[neg], f[neg] / vol[b[neg]])
        rows.append(a[neg])
        cols.append(b[neg])
        vals.append(-f[neg] / vol[a[neg]])
        # diffusion K h L / d (T_b - T_a)
        g = k_diff * h * dx / dx
        np.add.at(diag, a, -g / vol[a])
        np.add.at(diag, b, -g / vol[b])
        rows.extend([a, b])
        cols.extend([b, a])
        vals.extend([np.full(a.size, g) / vol[a], np.full(b.size, g) / vol[b]])

    both_x = ocean[:, :-1] & ocean[:, 1:]
    link(both_x, idx[:, :-1], idx[:, 1:], fx[:, :-1])
    both_y = ocean[:-1] & ocean[1:]
    link(both_y, idx[:-1], idx[1:], fy[:-1])
    n = diag.size
    rows.append(np.arange(n))
    cols.append(np.arange(n))
    vals.append(diag)
    return csc_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
                      shape=(n, n))  # fmt: skip


def ice_transport_operator(u, v, ocean, idx, dx, dt):
    """Explicit upwind drift as (sparse operator, substeps); conservative, positive.

    ``u, v`` (m/s, north-up) are capped at ``ICE_DRIFT_MAX``; faces to land carry nothing.
    """
    sp = np.hypot(u, v)
    k = np.minimum(1.0, ICE_DRIFT_MAX / np.maximum(sp, 1e-9))
    u, v = u * k, v * k
    ue = 0.5 * (u[:, :-1] + u[:, 1:])
    vn = 0.5 * (v[:-1] + v[1:])
    okx = ocean[:, :-1] & ocean[:, 1:]
    oky = ocean[:-1] & ocean[1:]
    cfl = ICE_DRIFT_MAX * dt / dx * 2.0
    nsub = max(1, int(math.ceil(cfl / 0.45)))
    c = dt / nsub / dx
    n = int(ocean.sum())
    rows, cols, vals = [np.arange(n)], [np.arange(n)], [np.ones(n)]
    for ok, vel, ia, ib in ((okx, ue, idx[:, :-1], idx[:, 1:]), (oky, vn, idx[:-1], idx[1:])):
        a, b, w = ia[ok], ib[ok], vel[ok] * c
        pos = w > 0
        src = np.where(pos, a, b)
        dst = np.where(pos, b, a)
        ww = np.abs(w)
        rows += [src, dst]
        cols += [src, src]
        vals += [-ww, ww]
    m = csc_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
                   shape=(n, n))  # fmt: skip
    return m, nsub


def ice_concentration(hi):
    return 1.0 - np.exp(-np.maximum(hi, 0.0) / H_ICE_SCALE)


def run_ocean(planet, lat_raster, sea_raster, winds, cell_km, params, t_air_seasons=None,
              years: int = 4, prev: OceanResult | None = None,
              exchange=None) -> OceanResult:  # fmt: skip
    """Ocean of the four seasons of winds ``winds = [(u, v), ...]`` (raster frame, v south)."""
    flip = lambda a: a[..., ::-1, :]  # noqa: E731  raster (rows south) -> north-up
    ocean_r, basin_r = ocean_mask(sea_raster)
    ocean = flip(ocean_r)
    lat = flip(lat_raster)
    ny, nx = ocean.shape
    dx = cell_km * 1000.0
    h = float(params.mixed_layer_m)
    cd = float(params.wind_drag_cd)
    circ = Circulation(ocean, lat, dx, planet)
    n = int(ocean.sum())
    idx = -np.ones(ocean.shape, dtype=np.int64)
    idx[ocean] = np.arange(n)
    area = dx * dx
    vol = np.full(n, area * h)
    from .ebm import for_planet

    ebm = for_planet(planet)
    t_eq_m = np.stack([ebm.at(lat, m) for m in range(12)])  # (12, ny, nx) seasonal
    t_eq = t_eq_m.mean(axis=0)
    s_m = planet.monthly_insolation(lat)  # (12, ny, nx) W/m^2
    s_ann = s_m.mean(axis=0)
    psis, fxs, fys, us, vs, ups = [], [], [], [], [], []
    w_up, w_down = [], []
    for u_r, v_r in winds:
        u = flip(u_r)
        v = -flip(v_r)  # +y south in the raster -> north
        sp = np.hypot(u, v)
        tx = RHO_A * cd * sp * u
        ty = RHO_A * cd * sp * v
        psi = circ.solve(tx, ty)
        fgx, fgy = Circulation.face_transport(psi)
        fgx = np.where(ocean, fgx, 0.0) * (HEAT_LAYER_M / WIND_LAYER_M)
        fgy = np.where(ocean, fgy, 0.0) * (HEAT_LAYER_M / WIND_LAYER_M)
        mx, my = ekman_transport(tx, ty, lat, planet)
        mx, my = np.where(ocean, mx, 0.0), np.where(ocean, my, 0.0)
        fex, fey = _faces_from_cells(mx, my, ocean, dx)
        div_e = np.where(ocean, _divergence(fex, fey), 0.0)  # m^3/s out of the cell
        fx, fy = fgx + fex, fgy + fey
        psis.append(psi)
        fxs.append(fx)
        fys.append(fy)
        # mixed-layer current (m/s) at the cell centres
        uc = np.zeros(ocean.shape)
        vc = np.zeros(ocean.shape)
        uc[:, 1:] += 0.5 * fx[:, :-1]
        uc += 0.5 * fx
        vc[1:] += 0.5 * fy[:-1]
        vc += 0.5 * fy
        us.append(np.where(ocean, uc / (dx * h), 0.0))
        vs.append(np.where(ocean, vc / (dx * h), 0.0))
        ups.append(np.where(ocean, div_e / area * 86400.0, 0.0))  # m/day (+ up)
        w_up.append(np.maximum(div_e, 0.0)[ocean])
        w_down.append(np.maximum(-div_e, 0.0)[ocean])
    # open ocean sides (TZ 5.4.1 G1): two-way exchange with an external sea, no net volume;
    # per cell of the side an exchange g (m^3/s) at the external temperature T_ext
    g_cell = np.zeros((12, n))
    t_ext = np.zeros((12, n))
    for side, q_m, t_m in exchange or []:
        mask = np.zeros(ocean.shape, dtype=bool)
        if side == "west":
            mask[:, 0] = True
        elif side == "east":
            mask[:, -1] = True
        elif side == "north":
            mask[-1] = True  # north-up frame inside this module
        elif side == "south":
            mask[0] = True
        cells = idx[mask & ocean]
        if cells.size == 0:
            continue
        for m_ in range(12):
            g_cell[m_, cells] += float(q_m[m_]) / cells.size
            t_ext[m_, cells] = float(t_m[m_])
    season_mid = (0, 3, 6, 9)
    ops = []
    dt = SECONDS_PER_MONTH / 2.0
    k_diff = float(params.eddy_diffusivity_m2_s)
    for s in range(4):
        a = heat_operator(fxs[s], fys[s], ocean, idx, vol, k_diff, dx, h)
        # vertical exchange: the Ekman outflow of a cell is replaced by upwelled water at
        # T_sub (right-hand side); an Ekman inflow leaves downward at the cell's T (here)
        a = a - csc_matrix((w_down[s] / vol, (np.arange(n), np.arange(n))), shape=(n, n))
        diag = np.arange(n)
        restore = csc_matrix((np.full(n, GAMMA / (RHO_W * C_W * h)), (diag, diag)), shape=(n, n))
        exch = csc_matrix((g_cell[season_mid[s]] / vol, (diag, diag)), shape=(n, n))
        m = identity(n, format="csc") - dt * (a - restore - exch)
        ops.append((splu(m.tocsc()), a))
    drift_ops = []
    for s_ in range(4):
        u_w, v_w = winds[s_]
        drift_ops.append(ice_transport_operator(us[s_] + 0.02 * flip(u_w),
                                                vs[s_] - 0.02 * flip(v_w),
                                                ocean, idx, dx, SECONDS_PER_MONTH))  # fmt: skip
    # initial state: restart from the previous coupler result if there is one
    if prev is not None and prev.sst_monthly.shape[1:] == ocean.shape[::1]:
        t = flip(prev.sst_monthly[-1])[ocean].astype(np.float64)
        hi = flip(prev.ice_thickness_monthly[-1] * prev.sea_ice_monthly[-1])[ocean]
    else:
        t = t_eq[ocean].copy()
        hi = np.zeros(n)
    t = np.maximum(t, T_FREEZE)
    season_of = (0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3, 0)
    sst = np.zeros((12, n))
    ice = np.zeros((12, n))
    heat = np.zeros((12, n))
    ledger = {"surface_j": 0.0, "vertical_j": 0.0, "latent_ice_j": 0.0, "storage_start_j": 0.0,
              "boundary_j": 0.0}  # fmt: skip
    rc = RHO_W * C_W * h
    hi_min = np.full(n, np.inf)
    hi_max = np.zeros(n)
    sm_o = s_m[:, ocean]
    sa_o = s_ann[ocean]

    teq_mo = t_eq_m[:, ocean]
    t_year = np.zeros(n)
    # subsurface reservoir of each basin (closed heat): downwelled mixed-layer water enters
    # it, upwelled water leaves it at the reservoir temperature; T_deep is the flux-weighted
    # mean of the water that went down in the previous year (no infinite cold sink)
    from scipy import ndimage as _nd

    lab, nb = _nd.label(ocean)
    basin_o = lab[ocean] - 1
    t_deep_b = np.full(max(nb, 1), float(np.mean(t)) - 4.0 if n else 0.0)
    if prev is not None and getattr(prev, "diagnostics", {}).get("deep_reservoir_c") is not None:
        old = np.asarray(prev.diagnostics["deep_reservoir_c"], dtype=np.float64)
        if old.size == t_deep_b.size:
            t_deep_b = old
    for year in range(years):
        t_year = np.zeros(n)
        down_heat = np.zeros(max(nb, 1))
        down_vol = np.zeros(max(nb, 1))
        if year == years - 1:
            ledger["storage_start_j"] = float(
                (rc * t).sum() * area - RHO_ICE * L_F * (hi * area).sum()
            )
        for m in range(12):
            s = season_of[m]
            lu, a = ops[s]
            for _half in range(2):
                c = ice_concentration(hi)
                hot = (t > T_FREEZE + 0.5) | (s_m[m][ocean] > sa_o)
                a_ice = np.where(hot, 0.5, 0.65)
                alpha = (1 - c) * ALPHA_WATER + c * a_ice
                # sunlight lost to the ice albedo (the ice-free cycle is in T_A)
                q_sw = -0.6 * sm_o[m] * (alpha - ALPHA_WATER)
                teq_now = teq_mo[m]
                # air-sea exchange: implicit part in the matrix + the explicit part
                q_rest_explicit = GAMMA * teq_now
                q = q_sw
                # insulation by ice of the cooling part of the flux
                ins = np.where(q + GAMMA * (teq_now - t) < 0, 1.0 / (1.0 + hi / H_ICE_SCALE), 1.0)
                q_eff = (1 - c) * q + c * q * ins
                t_sub = t_deep_b[basin_o]
                g_s = g_cell[season_mid[s]]
                rhs = (t + dt * (q_eff + q_rest_explicit) / rc + dt * w_up[s] * t_sub / vol
                       + dt * g_s * t_ext[m] / vol)  # fmt: skip
                t_new = lu.solve(rhs)
                # surface heat actually exchanged (for the ledger)
                q_tot = q_eff + GAMMA * (teq_now - t_new)
                if year == years - 1:
                    ledger["surface_j"] += float((q_tot * area).sum() * dt)
                    ledger["boundary_j"] += float(
                        (rc * dt * g_s * (t_ext[m] - t_new) / vol * area).sum()
                    )
                    vert = (w_up[s] * t_sub - w_down[s] * t_new) / vol
                    ledger["vertical_j"] += float((rc * dt * vert * area).sum())
                np.add.at(down_heat, basin_o, w_down[s] * t_new * dt)
                np.add.at(down_vol, basin_o, w_down[s] * dt)
                # freezing and melting
                deficit = np.maximum(T_FREEZE - t_new, 0.0) * rc  # J/m^2 below freezing
                grow = deficit / (RHO_ICE * L_F)
                t_new = np.maximum(t_new, T_FREEZE)
                excess = np.where(hi > 0, (t_new - T_FREEZE) * rc, 0.0)
                melt = np.minimum(hi, excess / (RHO_ICE * L_F))
                t_new = t_new - melt * RHO_ICE * L_F / rc
                hi = hi + grow - melt
                if year == years - 1:
                    ledger["latent_ice_j"] += float(((grow - melt) * RHO_ICE * L_F * area).sum())
                t = t_new
            # drift: surface current + 2 % of the wind (operator built once per season)
            if params.sea_ice and hi.any():
                m_op, nsub = drift_ops[s]
                for _ in range(nsub):
                    hi = m_op @ hi
                hi = np.maximum(hi, 0.0)
            if not params.sea_ice:
                hi = np.zeros(n)
            sst[m] = t
            t_year += t / 12.0
            ice[m] = hi
            heat[m] = rc * (t + 273.15)
            if m == 11:
                ok = down_vol > 0
                t_deep_b = np.where(ok, down_heat / np.maximum(down_vol, 1e-30), t_deep_b)
            if year == years - 1:
                hi_min = np.minimum(hi_min, hi)
                hi_max = np.maximum(hi_max, hi)
    storage_end = float((rc * t).sum() * area - RHO_ICE * L_F * (hi * area).sum())
    resid = (storage_end - ledger["storage_start_j"]) - (
        ledger["surface_j"] + ledger["vertical_j"] + ledger["boundary_j"]
    )

    # back to the raster frame
    def grid(vals, fill=np.nan):
        g = np.full((vals.shape[0],) + ocean.shape, fill, dtype=np.float64)
        g[:, ocean] = vals
        return flip(g)

    conc = ice_concentration(ice)
    thick = np.where(conc > 1e-3, ice / np.maximum(conc, 1e-3), 0.0)
    itype = np.zeros(ocean.shape, dtype=np.uint8)
    seasonal = (hi_max > 0.02) & (hi_min <= 0.02)
    multi = hi_min > 0.02
    itype[ocean] = np.where(multi, 2, np.where(seasonal, 1, 0))
    coast = ocean & ~ndimage.binary_erosion(ocean, border_value=1)
    up_ann = np.mean(ups, axis=0)
    itype[coast & (itype > 0)] = 3  # land-fast ice along the coast
    # qualitative salinity (not modelled): saltier under the subtropical highs, fresher
    # where sea ice forms and melts every year
    seasonal_g = np.zeros(ocean.shape)
    seasonal_g[ocean] = (hi_max > 0.02).astype(float)
    sal = np.where(ocean, 34.5 + 1.2 * np.exp(-(((np.abs(lat) - 25.0) / 12.0) ** 2))
                   - 1.5 * seasonal_g, np.nan)  # fmt: skip
    coastal_up = coast & (up_ann > 0.3)
    diag = {
        "ocean_cells": n,
        "basins": int(basin_r.max() + 1) if n else 0,
        "streamfunction_max_sv": float(max(np.abs(p).max() for p in psis) / 1e6) if n else 0.0,
        "max_current_ms": float(max(np.hypot(u, v).max() for u, v in zip(us, vs, strict=True))),
        "heat_residual_j": resid,
        "heat_relative_residual": abs(resid)
        / max(
            abs(ledger["surface_j"])
            + abs(ledger["vertical_j"])
            + abs(ledger["storage_start_j"]) * 1e-6,
            1.0,
        ),  # fmt: skip
        "ice_volume_km3_max": float((hi_max * area).sum() / 1e9),
        "divergence_max": 0.0,
        "flow_quality": "approximate_stommel_streamfunction",
        "zonal_calibration_max_k": 0.0,  # removed in 5.4.1 (no hidden zonal forcing)
        "sst_forcing": "planetary_seasonal_ebm_air_sea_exchange",
        "boundary_heat_j": ledger["boundary_j"],
        "boundary_sides": [e[0] for e in exchange or []],
        "deep_reservoir_c": [round(float(x), 4) for x in t_deep_b],
        "deep_reservoir": "per-basin closed heat: fed by downwelling, drained by upwelling",
        "air_sea_lambda_w_m2_k": GAMMA,
        "ebm_converged": bool(ebm.converged),
    }
    return OceanResult(
        ocean=ocean_r,
        basin_id=basin_r,
        sst_monthly=grid(sst).astype(np.float32),
        sea_ice_monthly=np.nan_to_num(grid(conc, 0.0)).astype(np.float32),
        ice_thickness_monthly=np.nan_to_num(grid(thick, 0.0)).astype(np.float32),
        current_u=np.stack([flip(u) for u in us]).astype(np.float32),
        current_v=np.stack([-flip(v) for v in vs]).astype(np.float32),
        psi=np.stack([flip(p) / 1e6 for p in psis]).astype(np.float32),
        upwelling_m_day=np.stack([flip(u) for u in ups]).astype(np.float32),
        coastal_upwelling=flip(coastal_up),
        heat_content_j_m2=grid(heat).astype(np.float32),
        salinity_psu=flip(sal).astype(np.float32),
        ice_type=flip(itype),
        diagnostics=diag,
    )


def _planet_shift(planet, lat):
    from .planet import annual_temperature_shift

    return annual_temperature_shift(planet, lat)
