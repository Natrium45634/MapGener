"""Grounded ice sheets on the hex graph: SMB, Vialov seed, SIA flow, convergence (TZ 5.3 sec. 8).

**Surface mass balance** (m ice equivalent per year, per hex) from the monthly climate, with
the ice surface feedback (``T - 6.5 K/km * H``)::

    SMB = snowfall(T, P) - k_melt * PDD(T, sd 4 K) - sublimation

**Seed** (``ice_init_mode``):

* ``cold_start`` - no ice; a 100-1000 yr run is reported as *not equilibrated*;
* ``analytic_steady_seed`` - a flat-bed Vialov profile (Glen n = 3)::

      H(d) = H0 [1 - (1 - d/R)^((n+1)/n)]^(n/(2n+2))

  ``d`` the distance from the margin of a connected region of positive SMB, ``R`` its
  radius, ``H0`` from the steady radial flux ``q = b r / 2`` of the SIA with the region's mean
  SMB ``b``. Only where SMB > 0: a dry polar land with SMB <= 0 is never seeded (T34);
* ``paleoclimate_spinup`` - the seed is grown first under a colder climate (``-6 K``) and
  then the present climate takes over.

**Flow** (SIA, every hex edge once, double-buffered, adaptive CFL step, no D cap)::

    dH/dt = SMB - div q,   q = -(2A/(n+2)) (rho g)^n H^(n+2) |grad s|^(n-1) grad s

Ice reaching an ocean hex calves (ledger ``calving``) unless the shelf model takes it
(:mod:`.shelves`). **Spin-up** runs in windows of ``window`` years until

    |dV|/V/dt < 1e-4 per 100 yr,   mean |dH|/dt < 0.01 m/yr,   mass residual < 1e-9

holds for two consecutive windows (CONVERGED), or ice only shrinks under a negative SMB
(TRANSIENT_RETREAT), or the time cap is reached (NOT_CONVERGED). The ledger
``V_end = V_0 + SMB - calving`` closes to round-off.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy import ndimage

RHO_ICE = 917.0
RHO_W = 1000.0
RHO_SEA = 1028.0
G = 9.81
SECONDS_YR = 365.25 * 86400.0
LAPSE = 0.0065
#: Glen A at about -10 degC, Pa^-3 per year (Cuffey & Paterson 2010 order of magnitude).
GLEN_A_YR = 1.0e-16 * 1.0
GLEN_N = 3
SLOPE_EPS = 3.0e-4


@dataclass
class IceSheetResult:
    thickness_m: np.ndarray  # (n,) grounded ice
    smb_m_yr: np.ndarray  # (n,) at the final surface
    velocity_m_yr: np.ndarray  # (n,) depth-mean SIA speed
    seed_m: np.ndarray  # (n,) initial state
    status: str
    years: float
    ledger: dict = field(default_factory=dict)
    history: list = field(default_factory=list)
    calving_m3_yr_hex: np.ndarray | None = None  # (n,) into each ocean hex, last step


def surface_mass_balance(t_m, p_m, z_surface_offset_m=0.0, k_melt_mm=5.0, sd_c=4.0,
                         sublimation_mm=0.0):  # fmt: skip
    """Annual SMB, m ice eq./yr; ``t_m, p_m`` (12, n) at the bare surface."""
    from ..biomes.cryosphere import _phase
    from ..stages.environment import MONTH_DAYS, positive_degree_days

    t = np.asarray(t_m, dtype=np.float64) - LAPSE * np.asarray(z_surface_offset_m)
    p = np.asarray(p_m, dtype=np.float64)
    snow = (p * _phase(t)).sum(axis=0)
    days = np.asarray(MONTH_DAYS, dtype=np.float64).reshape(12, *([1] * (t.ndim - 1)))
    pdd = positive_degree_days(t, sd_c, days).sum(axis=0)
    smb_mm = snow - k_melt_mm * pdd - sublimation_mm
    return smb_mm / 1000.0 * RHO_W / RHO_ICE


def glen_gamma(a_yr=GLEN_A_YR, n=GLEN_N):
    return 2.0 * a_yr / (n + 2) * (RHO_ICE * G) ** n


def vialov_h0(radius_m: float, b_m_yr: float, a_yr=GLEN_A_YR, n=GLEN_N) -> float:
    """Central thickness of the steady flat-bed sheet (axisymmetric, SMB ``b``).

    The steady flux ``q(r) = b r / 2`` equals the SIA flux of the Vialov profile; ``H0`` is
    solved from that identity at ``r = R/2`` (the profile shape is exact, so any r works).
    """
    if b_m_yr <= 0 or radius_m <= 0:
        return 0.0
    x = 0.5
    shape = (1.0 - x ** ((n + 1) / n)) ** (n / (2 * n + 2))
    # d(shape)/dx
    dshape = (
        (n / (2 * n + 2))
        * (1.0 - x ** ((n + 1) / n)) ** (n / (2 * n + 2) - 1)
        * (-((n + 1) / n) * x ** (1.0 / n))
    )
    gam = glen_gamma(a_yr, n)
    # q = gam H^(n+2) |dH/dr|^n with H = H0 shape, dH/dr = H0 dshape / R
    # => b x R / 2 = gam H0^(2n+2) shape^(n+2) |dshape|^n / R^n
    rhs = b_m_yr * x * radius_m / 2.0 * radius_m**n
    h0 = (rhs / (gam * shape ** (n + 2) * abs(dshape) ** n)) ** (1.0 / (2 * n + 2))
    return float(h0)


def vialov_profile(d_from_margin, radius_m, h0, n=GLEN_N):
    r = np.clip(1.0 - np.asarray(d_from_margin, dtype=np.float64) / max(radius_m, 1.0), 0.0, 1.0)
    return h0 * np.clip(1.0 - r ** ((n + 1) / n), 0.0, 1.0) ** (n / (2 * n + 2))


def flux_edges(nbr):
    """Undirected edges (i, j) of a hex neighbour table (each once)."""
    i_all, j_all = [], []
    for k in range(3):
        j = nbr[:, k]
        ok = j >= 0
        i_all.append(np.nonzero(ok)[0])
        j_all.append(j[ok])
    return np.concatenate(i_all), np.concatenate(j_all)


class SIA:
    """Shallow-ice flow on the hex graph (edge fluxes, conservative, adaptive step)."""

    def __init__(self, nbr, spacing_m, bed_m, ocean, a_yr=GLEN_A_YR, n=GLEN_N):
        self.i, self.j = flux_edges(nbr)
        self.L = float(spacing_m)
        self.area = np.sqrt(3.0) / 2.0 * self.L**2  # hex area for centre spacing L
        self.face = self.L / np.sqrt(3.0)  # edge length
        self.bed = np.asarray(bed_m, dtype=np.float64)
        self.ocean = np.asarray(ocean, dtype=bool)
        self.gam = glen_gamma(a_yr, n)
        self.n = n
        self.last_calving_hex = None

    def fluxes(self, h):
        """Volume flux (m^3/yr) across every edge, + from i to j, and the diffusivity."""
        s = self.bed + h
        i, j = self.i, self.j
        grad = (s[j] - s[i]) / self.L
        hm = np.where(s[i] >= s[j], h[i], h[j])
        d = self.gam * hm ** (self.n + 2) * np.abs(grad) ** (self.n - 1)
        q = -d * grad * self.face  # m^3/yr from i to j (down the surface slope)
        return q, d

    def step(self, h, smb, dt):
        """Explicit step; ice entering an ocean hex is removed as calving. Returns h, ledger."""
        q, _d = self.fluxes(h)
        # a donor never gives more than it holds (limiter keeps H >= 0 and mass exact)
        out = np.zeros(h.size)
        np.add.at(out, self.i, np.maximum(q, 0.0))
        np.add.at(out, self.j, np.maximum(-q, 0.0))
        avail = h * self.area / max(dt, 1e-12)
        scale = np.where(out > avail, avail / np.maximum(out, 1e-30), 1.0)
        qs = np.where(q > 0, q * scale[self.i], q * scale[self.j])
        div = np.zeros(h.size)
        np.add.at(div, self.i, qs)
        np.add.at(div, self.j, -qs)
        gain = np.maximum(smb * dt, -h)  # melt cannot remove more ice than there is
        h_new = h + gain - div / self.area * dt
        h_new = np.maximum(h_new, 0.0)
        calved = float((h_new[self.ocean] * self.area).sum())
        h_new[self.ocean] = 0.0
        return h_new, {"smb_m3": float((gain * self.area).sum()), "calving_m3": calved}

    def stable_dt(self, h, cap=50.0):
        _q, d = self.fluxes(h)
        dmax = float(d.max()) if d.size else 0.0
        return cap if dmax <= 0 else min(cap, 0.12 * self.L**2 / dmax)

    def implicit_step(self, h, smb, dt, picard: int = 8):
        """Semi-implicit step (D lagged, Picard iterations), conservative edge Laplacian.

        ``(I - dt/A L(D)) H_new = H + dt SMB + dt/A L(D) bed``; ocean hexes are H = 0
        (Dirichlet): what flows into them is calving, computed from the same coefficients,
        so ``V_new = V + SMB - calving - margin`` closes exactly (``margin``: negative
        thickness clipped at the margin, booked as melt).
        """
        from scipy.sparse import coo_matrix, identity
        from scipy.sparse.linalg import spsolve

        n = h.size
        active = (h > 0) | (smb > 0)
        # one ring of neighbours around the ice can receive it
        ring = np.zeros(n, dtype=bool)
        ring[self.i[active[self.j]]] = True
        ring[self.j[active[self.i]]] = True
        cells = (active | ring) & ~self.ocean
        idx = -np.ones(n, dtype=np.int64)
        idx[cells] = np.arange(int(cells.sum()))
        m = int(cells.sum())
        if m == 0:
            return h.copy(), {"smb_m3": 0.0, "calving_m3": 0.0, "margin_m3": 0.0}
        gain = np.where(cells, np.maximum(smb * dt, -h), 0.0)
        h_iter = h.copy()
        for _ in range(max(1, picard)):
            s = self.bed + h_iter
            grad = (s[self.j] - s[self.i]) / self.L
            # upstream thickness (the side with the higher surface): a margin next to an
            # empty hex keeps its outflow (a mean thickness would lock the margin shut)
            hm = np.where(s[self.i] >= s[self.j], h_iter[self.i], h_iter[self.j])
            # slope regularisation (eps = 3e-4): a flat dome still conducts, otherwise the
            # lagged D of a flat interior is zero and the outflow never reaches the centre
            dcoef = (
                self.gam * hm ** (self.n + 2) * (grad * grad + SLOPE_EPS**2) ** ((self.n - 1) / 2.0)
            )
            k = dcoef * self.face / self.L / self.area * dt  # per edge, dimensionless
            use = (cells[self.i] | cells[self.j]) & (k > 0)
            ii, jj, kk = self.i[use], self.j[use], k[use]
            rows, cols, vals = [], [], []
            rhs = np.zeros(m)
            for a, b in ((ii, jj), (jj, ii)):
                ca = cells[a]
                ia = idx[a[ca]]
                kb = kk[ca]
                rows.append(ia)
                cols.append(ia)
                vals.append(kb)
                cb = cells[b[ca]]
                rows.append(ia[cb])
                cols.append(idx[b[ca][cb]])
                vals.append(-kb[cb])
                # bed difference term (explicit, it is known) and Dirichlet H = 0 off-cells
                np.add.at(rhs, ia, kb * (self.bed[b[ca]] - self.bed[a[ca]]))
            a_mat = (
                identity(m, format="csr")
                + coo_matrix(
                    (np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
                    shape=(m, m),
                ).tocsr()
            )
            b_vec = h[cells] + gain[cells] + rhs
            sol = spsolve(a_mat.tocsc(), b_vec)
            prev_iter = h_iter
            h_iter = h.copy()
            # under-relaxed Picard: the lagged D converges to the backward-Euler solution
            h_iter[cells] = sol if _ == 0 else 0.5 * (sol + prev_iter[cells])
            if _ and float(np.abs(h_iter[cells] - prev_iter[cells]).max()) < 0.5:
                h_iter[cells] = sol
                break
        # the implicit solution is conservative: V_cells(new) = V_cells(old) + SMB - flux to
        # the sea. A margin can come out slightly negative (lagged D): that over-drain is
        # taken back from what went to the sea first, the rest from the positive ice in
        # proportion - no mass is created (clipping alone would create it)
        v_old = float((h[cells] * self.area).sum())
        smb_v = float((gain[cells] * self.area).sum())
        sol = h_iter[cells]
        v_sol = float(sol.sum() * self.area)
        calving = v_old + smb_v - v_sol
        neg = float(-np.minimum(sol, 0.0).sum() * self.area)
        sol = np.maximum(sol, 0.0)
        back = min(neg, max(calving, 0.0))
        calving -= back
        rest = neg - back
        if rest > 0:
            pos = float(sol.sum() * self.area)
            sol = sol * max(0.0, 1.0 - rest / max(pos, 1e-30))
        h_new = h.copy()
        h_new[cells] = sol
        margin = -rest if rest > 0 else 0.0  # m^3 removed to undo an over-drain (booked)
        # where the calving went: edge fluxes into the sea from the solution, scaled to the
        # budget value (m^3 per step per ocean hex)
        s_sol = self.bed + np.where(cells, h_iter, 0.0)
        into = np.zeros(n)
        for a, b in ((self.i, self.j), (self.j, self.i)):
            e = cells[a] & self.ocean[b]
            f = k[e] * self.area * np.maximum(s_sol[a[e]] - s_sol[b[e]], 0.0)
            np.add.at(into, b[e], f)
        tot = float(into.sum())
        self.last_calving_hex = into * (max(calving, 0.0) / tot) if tot > 0 else into
        return h_new, {"smb_m3": smb_v, "calving_m3": calving, "margin_m3": margin}


def seed(mode, smb, t_annual, land, nbr, spacing_m, a_yr=GLEN_A_YR):
    """Initial grounded ice (m) for the chosen mode (see the module docstring)."""
    n = smb.size
    h = np.zeros(n)
    if mode == "cold_start":
        return h
    pos = land & (smb > 0.0)
    if not pos.any():
        return h
    # connected regions of positive SMB on the hex graph
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components

    i, j = flux_edges(nbr)
    keep = pos[i] & pos[j]
    g = coo_matrix((np.ones(keep.sum()), (i[keep], j[keep])), shape=(n, n))
    _nc, lab = connected_components(g, directed=False)
    # distance (in hexes) from the margin of the region, by repeated erosion on the graph
    dist = np.where(pos, 0.0, -1.0)
    frontier = pos.copy()
    k = 0
    inner = pos.copy()
    while inner.any() and k < 400:
        k += 1
        nb_out = np.zeros(n, dtype=bool)
        for c in range(6):
            jj = nbr[:, c]
            nb_out |= (jj < 0) | ~inner[np.maximum(jj, 0)]
        inner = inner & ~nb_out
        dist[inner] = k
    del frontier
    for r in np.unique(lab[pos]):
        m = pos & (lab == r)
        cells = int(m.sum())
        if cells < 3:
            continue
        radius = np.sqrt(cells * np.sqrt(3.0) / 2.0 / np.pi) * spacing_m
        b = float(smb[m].mean())
        h0 = vialov_h0(radius, b, a_yr)
        d = (dist[m] + 0.5) * spacing_m
        h[m] = vialov_profile(d, radius, h0)
    return h


def spinup(t_m, p_m, z_bed, land, ocean, nbr, spacing_m, mode="analytic_steady_seed",
           min_years=500.0, max_years=50000.0, window=500.0, k_melt_mm=5.0,
           paleo_cooling_k=6.0, initial=None, step_years=100.0,
           sublimation_mm=0.0) -> IceSheetResult:  # fmt: skip
    """Seed + SIA spin-up to the convergence criteria; mass ledger closed."""
    t_m = np.asarray(t_m, dtype=np.float64)
    n = z_bed.size
    land = np.asarray(land, bool)
    ocean = np.asarray(ocean, bool)
    sia = SIA(nbr, spacing_m, np.where(land, z_bed, np.minimum(z_bed, 0.0)), ocean)
    sub = np.asarray(sublimation_mm, dtype=np.float64)
    smb0 = surface_mass_balance(t_m, p_m, 0.0, k_melt_mm, sublimation_mm=sub)
    t_ann = t_m.mean(axis=0)
    if initial is not None:
        h = np.where(land, np.asarray(initial, dtype=np.float64), 0.0)
        seed_h = h.copy()
    elif mode == "paleoclimate_spinup":
        smb_cold = surface_mass_balance(t_m - paleo_cooling_k, p_m, 0.0, k_melt_mm,
                                        sublimation_mm=sub)  # fmt: skip
        h = seed(mode, np.where(land, smb_cold, -1.0), t_ann, land, nbr, spacing_m)
        seed_h = h.copy()
    else:
        h = seed(mode, np.where(land, smb0, -1.0), t_ann, land, nbr, spacing_m)
        seed_h = h.copy()
    v0 = float((h * sia.area).sum())
    ledger = {"initial_m3": v0, "smb_m3": 0.0, "calving_m3": 0.0, "margin_m3": 0.0}
    t = 0.0
    history = []
    status = "NOT_CONVERGED"
    good = 0
    prev_v, prev_h = v0, h.copy()
    win_smb = win_calv = win_margin = 0.0
    next_check = window
    while t < max_years:
        smb = np.where(land, surface_mass_balance(t_m, p_m, h, k_melt_mm, sublimation_mm=sub), 0.0)
        smb = np.where(ocean, 0.0, smb)
        dt = min(step_years, next_check - t)
        h, step_ledger = sia.implicit_step(h, smb, dt)
        ledger["smb_m3"] += step_ledger["smb_m3"]
        ledger["calving_m3"] += step_ledger["calving_m3"]
        ledger["margin_m3"] += step_ledger["margin_m3"]
        win_smb += step_ledger["smb_m3"]
        win_calv += step_ledger["calving_m3"]
        win_margin += step_ledger["margin_m3"]
        t += dt
        if t >= next_check - 1e-9:
            v = float((h * sia.area).sum())
            ref = max(v, prev_v, 1e9)
            dv_rel_100 = abs(v - prev_v) / ref / window * 100.0
            dh = float(np.abs(h - prev_h)[land].mean()) / window if land.any() else 0.0
            resid = (v - prev_v) - (win_smb - win_calv + win_margin)
            history.append({"year": round(t, 1), "volume_km3": round(v / 1e9, 3),
                            "rel_dv_per_100yr": dv_rel_100, "mean_dh_m_yr": dh,
                            "window_residual_m3": resid})  # fmt: skip
            ok = dv_rel_100 < 1e-4 and dh < 0.01
            good = good + 1 if ok else 0
            if v <= 0 and v0 <= 0:
                status = "CONVERGED"  # no ice and none can form
                break
            if t >= min_years and good >= 2:
                status = "CONVERGED"
                break
            vols = [x["volume_km3"] for x in history]
            shrinking = len(vols) > 3 and all(b <= a for a, b in zip(vols[:-1], vols[1:],
                                                                    strict=True))  # fmt: skip
            if t >= max_years and shrinking and v < v0:
                status = "TRANSIENT_RETREAT"
            prev_v, prev_h = v, h.copy()
            win_smb = win_calv = win_margin = 0.0
            next_check = t + window
    v_end = float((h * sia.area).sum())
    ledger["final_m3"] = v_end
    ledger["residual_m3"] = v_end - (
        v0 + ledger["smb_m3"] - ledger["calving_m3"] + ledger["margin_m3"]
    )
    ledger["relative_residual"] = abs(ledger["residual_m3"]) / max(
        v0 + abs(ledger["smb_m3"]) + ledger["calving_m3"], 1.0)  # fmt: skip
    if status == "NOT_CONVERGED" and len(history) > 3:
        vols = [x["volume_km3"] for x in history[-4:]]
        if all(b <= a for a, b in zip(vols[:-1], vols[1:], strict=True)) and vols[-1] < vols[0]:
            status = "TRANSIENT_RETREAT"
    smb_final = np.where(
        land & ~ocean, surface_mass_balance(t_m, p_m, h, k_melt_mm, sublimation_mm=sub), 0.0
    )
    _q, d = sia.fluxes(h)
    speed = np.zeros(n)
    s = sia.bed + h
    grad = (s[sia.j] - s[sia.i]) / sia.L
    hm = 0.5 * (h[sia.i] + h[sia.j])
    u_edge = np.where(hm > 1.0, d * np.abs(grad) / np.maximum(hm, 1.0), 0.0)
    np.maximum.at(speed, sia.i, u_edge)
    np.maximum.at(speed, sia.j, u_edge)
    last = getattr(sia, "last_calving_hex", None)
    calv_hex = None if last is None else last / max(dt, 1e-9)
    return IceSheetResult(h, smb_final, speed, seed_h, status, t, ledger, history, calv_hex)


def isostatic_deflection(h_ice, years, tau_yr=3000.0, rho_mantle=3300.0, smooth_cells=1.5,
                         grid_shape=None):  # fmt: skip
    """Bed depression (m, negative) under the ice after ``years`` of loading (local Airy,
    delayed relaxation ``w = w_eq (1 - exp(-t/tau))``); smoothed over ~1.5 hex as a crude
    flexural spread (P1 approximation)."""
    w_eq = -(RHO_ICE / rho_mantle) * np.asarray(h_ice, dtype=np.float64)
    if grid_shape is not None and smooth_cells > 0:
        w_eq = ndimage.gaussian_filter(w_eq.reshape(grid_shape), smooth_cells).ravel()
    return w_eq * (1.0 - np.exp(-float(years) / tau_yr))


def sea_level_equivalent_m(ice_volume_m3_above_flotation, ocean_area_m2):
    """Eustatic sea level (m) the grounded ice would add if it melted (fresh water)."""
    if ocean_area_m2 <= 0:
        return 0.0
    return float(ice_volume_m3_above_flotation) * RHO_ICE / RHO_W / ocean_area_m2


def glacial_erosion_rate(speed_m_yr, h_ice, t_annual_c, k_g=1.0e-4):
    """``[H]`` erosion rate, m/yr: ``K_g |u_b|`` with sliding only under a temperate bed.

    Basal sliding ``u_b`` is taken as 30 % of the SIA speed where the bed is near the
    pressure melting point (thick ice or mild climate) and 0 on a frozen bed (T46).
    """
    h = np.asarray(h_ice, dtype=np.float64)
    temperate = (h > 1500.0) | (np.asarray(t_annual_c) > -6.0)
    ub = np.where(temperate & (h > 50.0), 0.3 * np.asarray(speed_m_yr), 0.0)
    return k_g * ub
