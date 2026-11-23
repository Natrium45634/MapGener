"""Column water vapour ``W`` [kg/m^2], its transport ``Q`` [kg/(m s)] and monthly rain.

Per month (TZ 5.4 sec. 6), implicit in time with the month as the step (the vapour
residence time, ~1 week, is shorter than a month: a quasi-steady normal, sec. 3.2)::

    (W - W_prev) / dt + div(u_q W) = E - P          (finite volume, upwind, exact budget)
    E_sea  = rho C_E |U| (q_s(SST) - (a + b W / W_sat) q_s(T_a)) (1 - ice)  bulk, linear in W
    E_land = Budyko share of the recent rain (moisture recycling), <= the bulk potential
    P      = (k_bg + k_conv + k_oro + k_front) W                     each >= 0, each booked

    k_bg    = RH^2 / tau_bg + saturation excess                       large-scale rain
    k_conv  = warm(T) trig(RH) ascent(-div u_sfc) / tau_conv          tropical convection
    k_oro   = eps max(0, u_q . grad h) / H_v  x moist(RH)              forced ascent
    k_front = baroclinicity(|grad T| / 8 K per 1000 km) |f|/f45 moist(RH) / tau_front

with ``RH = W / W_sat(T)``, ``W_sat = e_s(T) / (R_v T) H_v`` and the vapour-weighted wind
``u_q = 0.6 u_sfc + 0.4 u_850``. The rates depend on ``W`` through ``RH``: a few Picard
iterations per month (lagged rates, one sparse LU each); two years, the first a spin-up.
Every term is integrated in kg over the domain: storage change = E - P + import - export
to the solver precision (``relative_residual`` reported, not hidden). Orography only moves
water (it shortens the vapour's path); it never adds water.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy import ndimage, sparse
from scipy.sparse import linalg as sla

from . import fv

R_V = 461.5
H_V = 2200.0  # m, scale height of water vapour
RHO_AIR = 1.2
C_E = 1.2e-3
U_GUST = 4.0
U_STORM = 5.0
RH_BL_A = 0.5
RH_BL_B = 0.45
DAY = 86400.0
#: [H] calibrated time scales (Earth-like defaults, documented in v5_4_parameter_calibration)
TAU_BG_D = 6.0
RH_POWER = 2.0
ASCENT_FLOOR = 0.15
SOIL_MEMORY = 0.3
LAND_HEAT_TRIGGER = 0.3
TAU_CONV_D = 7.0
TAU_FRONT_D = 5.0
TAU_SAT_D = 0.5
EPS_ORO = 0.5
ORO_CAP_PER_S = 1.0 / (6 * 3600.0)
COMPONENTS = ("convective", "orographic", "frontal", "storm", "background")


def e_sat_pa(t_c):
    t = np.asarray(t_c, dtype=np.float64)
    return 611.2 * np.exp(17.67 * t / (t + 243.5))


def w_sat(t_c):
    """Saturated column vapour (kg/m^2) of an exponential vapour profile at the surface T."""
    t = np.asarray(t_c, dtype=np.float64)
    return e_sat_pa(t) / (R_V * (t + 273.15)) * H_V


def q_sat_kg_kg(t_c, p_pa=101325.0):
    e = e_sat_pa(t_c)
    return 0.622 * e / np.maximum(p_pa - 0.378 * e, 1.0)


def _fu(pet_over_p, w=2.6):
    x = np.maximum(pet_over_p, 0.0)
    return 1.0 + x - (1.0 + x**w) ** (1.0 / w)


@dataclass
class MoistureResult:
    w: np.ndarray  # (12, n) kg/m^2 column vapour at the end of each month
    ivt_e: np.ndarray  # (12, n) kg/(m s) east
    ivt_n: np.ndarray  # (12, n) kg/(m s) north
    conv: np.ndarray  # (12, n) kg/(m^2 s) moisture convergence -div(Q)
    evap_mm: np.ndarray  # (12, n) mm/month
    precip_mm: np.ndarray  # (12, n) mm/month total
    components_mm: np.ndarray  # (5, 12, n) mm/month by COMPONENTS
    rh: np.ndarray  # (12, n) column relative humidity
    ledger: list = field(default_factory=list)  # per month, kg
    diagnostics: dict = field(default_factory=dict)


def solve(
    mesh: fv.Mesh,
    circ_u,
    circ_v,
    t_air,
    sst,
    ice,
    sea_frac,
    land_h,
    lat,
    planet,
    days,
    cell_m,
    pet_mm=None,
    picard=3,
    years=2,
    x_bc=None,
    w_init=None,
) -> MoistureResult:
    """Monthly normals of the vapour budget on ``mesh``.

    ``circ_u/v`` (12, 2, n): surface and 850 hPa wind (east / north); ``t_air`` (12, n)
    surface air temperature; ``sst`` / ``ice`` (12, n) on the sea share (``sea_frac``);
    ``pet_mm`` (12, n) land PET for the recycling; ``x_bc`` (12, n) inflow column at the
    open boundary (None = zero gradient, ``boundary_forcing = synthetic``).
    """
    n = mesh.n
    a = mesh.area
    sea_frac = np.asarray(sea_frac, dtype=np.float64)
    land_frac = 1.0 - sea_frac
    shape2 = land_h.shape
    gx, gy = np.gradient(np.asarray(land_h, dtype=np.float64).reshape(shape2), cell_m)
    gh_e, gh_n = gy.ravel(), -gx.ravel()  # rows grow southward
    f = planet.coriolis(lat).ravel()
    f45 = abs(float(planet.coriolis(45.0)))
    sig = max(1.0, 800e3 / cell_m)
    t_mean = np.asarray(t_air, dtype=np.float64).mean(axis=0)
    w_state = 0.6 * w_sat(t_air[0]) if w_init is None else np.full(n, float(w_init))
    p_recent = np.zeros(n)
    p_year = np.zeros(n)
    res = None
    ledger = []
    picard_log = []
    out = {k: np.zeros((12, n)) for k in ("w", "ive", "ivn", "conv", "e", "p", "rh")}
    comp = np.zeros((5, 12, n))
    for year in range(years):
        final = year == years - 1
        for m in range(12):
            dt = float(days[m]) * DAY
            us, vs = circ_u[m, 0], circ_v[m, 0]
            uq = 0.6 * us + 0.4 * circ_u[m, 1]
            vq = 0.6 * vs + 0.4 * circ_v[m, 1]
            tr = fv.transport(mesh, uq, vq)
            div_s = mesh.divergence(us, vs)
            ws = w_sat(t_air[m])
            warm = np.clip((t_air[m] - 14.0) / 12.0, 0.0, 1.0)
            heat = land_frac * np.clip((t_air[m] - t_mean) / 8.0, 0.0, 1.0)
            # transient disturbances keep some convection under the mean subsidence [A]
            ascent = np.clip(1.0 + (-div_s * 1000.0) / 2.0e-3, ASCENT_FLOOR, 3.0)
            w_up = np.maximum(uq * gh_e + vq * gh_n, 0.0)
            tg = ndimage.gaussian_filter(t_air[m].reshape(shape2), sig, mode="nearest")
            g1, g2 = np.gradient(tg, cell_m)
            baro = np.clip(np.hypot(g1, g2).ravel() * 1e6 / 8.0, 0.0, 2.0)
            frontal = baro * np.clip(np.abs(f) / f45, 0.0, 1.0)
            # transient eddies: the storm tracks add gusts to the mean wind (bulk formula)
            speed = np.hypot(us, vs) + U_GUST + U_STORM * frontal
            # ocean: E = e0 - e1 W ; zero over the ice-covered share
            open_sea = sea_frac * (1.0 - np.clip(ice[m], 0.0, 1.0))
            # boundary-layer humidity RH_bl = a + b RH_col (the subsiding column is dry
            # aloft while the surface layer stays moist); q_a = RH_bl q_s(T_air)
            qa_sat = q_sat_kg_kg(np.minimum(t_air[m], sst[m]))
            e0 = RHO_AIR * C_E * speed * (q_sat_kg_kg(sst[m]) - RH_BL_A * qa_sat) * open_sea
            e0 = np.maximum(e0, 0.0)
            e1 = RHO_AIR * C_E * speed * RH_BL_B * qa_sat / np.maximum(ws, 1e-3) * open_sea
            if pet_mm is not None:
                pet_rate = np.maximum(pet_mm[m], 0.0) / dt  # kg/m^2/s
            else:
                pet_rate = np.full(n, 3.0 / DAY)
            w_prev = w_state.copy()
            w_it = w_state.copy()
            for _it in range(picard if final else max(1, picard - 1)):
                rh = w_it / np.maximum(ws, 1e-3)
                moist = np.clip((rh - 0.3) / 0.4, 0.0, 1.0)
                k_bg = np.minimum(rh, 1.5) ** RH_POWER / (TAU_BG_D * DAY)
                k_bg += np.maximum(rh - 1.0, 0.0) / np.maximum(rh, 1e-9) / (TAU_SAT_D * DAY)
                # summer heating of the land lowers the convective threshold (afternoon
                # convection of continental summers) [A]
                trig = np.clip((rh - 0.45 + LAND_HEAT_TRIGGER * heat) / 0.35, 0.0, 1.0)
                k_conv = warm * trig * ascent / (TAU_CONV_D * DAY)
                k_oro = np.minimum(EPS_ORO * w_up / H_V * moist, ORO_CAP_PER_S)
                k_front = frontal * moist / (TAU_FRONT_D * DAY)
                k_tot = k_bg + k_conv + k_oro + k_front
                # land recycling: the Budyko share of the rain supply, capped by the PET;
                # soil water buffers the seasons: the supply blends this month's rain with
                # the running annual mean (dry-season evapotranspiration from storage)
                p_rate = (1.0 - SOIL_MEMORY) * p_recent + SOIL_MEMORY * p_year
                e_land = land_frac * np.minimum(
                    p_rate * _fu(pet_rate / np.maximum(p_rate, 1e-12)), pet_rate
                )
                # inflow through the open boundary: a specified column, or (zero gradient)
                # the boundary cell's own column of the previous iterate; on the right-hand
                # side, so the matrix stays an M-matrix and W >= 0 without clipping
                xb_used = x_bc[m] if x_bc is not None else w_it
                rhs = w_prev / dt + e0 + e_land + tr.import_rate * xb_used
                mat = (tr.D + sparse.diags(1.0 / dt + e1 + k_tot)).tocsc()
                w_new = sla.splu(mat).solve(rhs)
                clipped = np.minimum(w_new, 0.0)
                w_new = w_new - clipped
                change = float(np.abs(w_new - w_it).max() / max(float(w_new.max()), 1e-9))
                w_it = w_new
                p_recent = 0.5 * p_recent + 0.5 * (k_tot * w_it)
                if final:
                    picard_log.append(change)
            p_year = p_year + (p_recent - p_year) / 12.0
            w_state = w_it
            if not final:
                continue
            # bookkeeping with the final rates (kg over the domain in the month)
            evap = (e0 - e1 * w_state + e_land) * dt  # kg/m^2
            prec_c = np.stack([k_conv, k_oro, k_front, np.zeros(n), k_bg]) * w_state * dt
            prec = prec_c.sum(axis=0)
            imp = tr.import_rate * xb_used * dt
            exp_ = tr.export_rate * w_state * dt
            d_store = w_state - w_prev
            clip_kg = float((a * clipped).sum())  # negative W set to 0 (booked, ~never)
            resid = float((a * (d_store - evap + prec - imp + exp_ + clipped)).sum())
            led = {
                "month": m + 1,
                "evaporation_kg": float((a * evap).sum()),
                "precipitation_kg": float((a * prec).sum()),
                "import_kg": float((a * imp).sum()),
                "export_kg": float((a * exp_).sum()),
                "storage_change_kg": float((a * d_store).sum()),
                "residual_kg": resid,
                "cfl_month_step": mesh.cfl(uq, vq, dt),
                "negative_clipped_kg": -clip_kg,
            }
            led["relative_residual"] = abs(resid) / max(led["precipitation_kg"], 1e-30)
            ledger.append(led)
            out["w"][m] = w_state
            out["ive"][m] = uq * w_state
            out["ivn"][m] = vq * w_state
            out["conv"][m] = -mesh.divergence(uq * w_state, vq * w_state)
            out["e"][m] = evap
            out["p"][m] = prec
            out["rh"][m] = w_state / np.maximum(ws, 1e-3)
            comp[:, m] = prec_c
    res = MoistureResult(
        out["w"], out["ive"], out["ivn"], out["conv"], out["e"], out["p"], comp, out["rh"], ledger
    )
    tot_p = sum(x["precipitation_kg"] for x in ledger)
    res.diagnostics = {
        "method": "conservative_finite_volume_upwind_implicit_monthly",
        "picard_last_change_max": float(max(picard_log[-12:])) if picard_log else 0.0,
        "annual_relative_residual": abs(sum(x["residual_kg"] for x in ledger)) / max(tot_p, 1e-30),
        "boundary_forcing": "synthetic_zero_gradient" if x_bc is None else "specified",
    }
    return res
