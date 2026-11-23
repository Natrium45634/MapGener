"""Twelve monthly circulation normals: temperature, pressure and winds on several levels.

Procedure (TZ 5.4 sec. 4-5, [A] = HexWorld approximation, every constant configurable)::

    insolation S(lat, month) of the planet           -> anomaly S' (first two harmonics)
    one-box surface response C dT/dt = S' - lambda T -> lag phi_k = atan(k omega tau),
        tau = 30 d (interior land) .. 50 d (coast); the sea is the v5.3 SST itself
    T_m = T_ann + amp r_m      (amp = the climate's seasonal half range, r: unit response)
    thermal equator phi_te(x, m): latitude of the warmest smoothed air in each column
    Hadley edge phi_H = 30 deg sqrt(dT_ep / 50 K) (Omega_E a_E) / (Omega a)  (Held-Hou scaling)
    SLP = zonal belts around phi_te (trough, subtropical highs at phi_H, subpolar lows,
          polar highs) + thermal lows / highs  -1 hPa/K x (regional T anomaly, 1000 km)
    winds from the frictional balance  r u - f v = -p_x / rho,  r v + f u = -p_y / rho
        u = (-r p_x - f p_y) / (rho (r^2 + f^2)),  v = (f p_x - r p_y) / (rho (r^2 + f^2))
    (r > 0 everywhere: no division by f; at the equator the wind blows down the gradient)
    850 hPa: the same balance on the 850 hPa height, which carries the hypsometric
        thickness of the warm / cold columns (+0.6 hPa/K): heat lows weaken aloft
    500 / 250 hPa: thermal wind of the monthly temperature upward from 850 hPa
        (equatorial taper below 12 deg, as v5.3)

Coordinates: rows grow southward, ``v`` is returned north-positive (``v_north``).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
from scipy import ndimage

from ..earth.planet import EARTH, MONTH_FRACTION, Planet
from ..earth.winds import EQUATORIAL_LAT, R_AIR, level_height_m

G = 9.81
RHO_AIR = 1.2
P0_PA = 101325.0
LEVELS_PA = (85000.0, 50000.0, 25000.0)
#: levels of the generalised profile (TZ sec. 14); 925 / 700 interpolate, 100 / 50 extrapolate
PROFILE_LEVELS_PA = (92500.0, 85000.0, 70000.0, 50000.0, 25000.0, 10000.0, 5000.0)
MONTH_DAYS = np.array([31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31], dtype=np.float64)
TAU_LAND_D = 30.0
TAU_COAST_D = 50.0
TAU_SEA_D = 60.0  # only without a v5.3 ocean
THERMAL_HPA_PER_K = 1.0
THICKNESS_HPA_PER_K = 0.6
R_SURFACE_SEA = 2.4e-5
R_SURFACE_LAND = 4.5e-5
R_850 = 1.0e-5
R_EQUATORIAL = 5.0e-5
SURFACE_CAP = 25.0
UPPER_CAP = 90.0


def month_days(planet: Planet) -> np.ndarray:
    """Length of the 12 months in days: the Earth calendar, or equal months elsewhere."""
    if abs(planet.year_length_days - 365.25) < 0.5:
        return MONTH_DAYS.copy()
    return np.full(12, planet.year_length_days / 12.0)


def hadley_edge_deg(planet: Planet, dt_equator_pole_k: float) -> tuple[float, str]:
    """Held-Hou scaling of the Hadley-cell edge, calibrated to 30 deg for the Earth."""
    scale = (EARTH.omega * EARTH.radius_km) / (planet.omega * planet.radius_km)
    phi = 30.0 * math.sqrt(max(dt_equator_pole_k, 5.0) / 50.0) * scale
    quality = "held_hou_scaled"
    if phi > 60.0:
        quality = "global_hadley_regime_unvalidated"
    return float(np.clip(phi, 8.0, 70.0)), quality


def unit_response(planet: Planet, lat, tau_days):
    """(12, ...) unit seasonal response of a one-box surface to the insolation cycle.

    First two harmonics of the monthly insolation anomaly, each delayed and damped by
    ``1 / (1 + i k omega tau)``; scaled to a unit first-harmonic-equivalent half range and
    to zero day-weighted mean. Zero obliquity gives zero (no season without a sun cycle).
    """
    lat = np.asarray(lat, dtype=np.float64)
    s = planet.monthly_insolation(lat)  # (12, ...)
    s = s - s.mean(axis=0)
    th = 2.0 * np.pi * np.arange(12) / 12.0
    tau = np.asarray(tau_days, dtype=np.float64)
    omega = 2.0 * np.pi / planet.year_length_days
    out = np.zeros_like(s)
    for k in (1, 2):
        c = (2.0 / 12.0) * np.tensordot(np.exp(-1j * k * th), s, axes=(0, 0))
        h = 1.0 / (1.0 + 1j * k * omega * tau)
        for m in range(12):
            out[m] += np.real(c * h * np.exp(1j * k * th[m]))
    half = 0.5 * (out.max(axis=0) - out.min(axis=0))
    e = EARTH.monthly_insolation(lat)
    ref = 0.5 * (e.max(axis=0) - e.min(axis=0))
    # normalise by the response the Earth's cycle would give, so the planet's amplitude
    # ratio stays in ``amp`` (seasonal_amplitude_ratio) and is not normalised away
    floor = np.maximum(0.25 * np.maximum(ref, 1.0), 5.0)
    r = out / np.maximum(half, floor)
    days = month_days(planet).reshape((12,) + (1,) * lat.ndim)
    return r - (days * r).sum(axis=0) / days.sum()


def monthly_temperature(planet, lat, t_annual, amp, cont, sea, sst_monthly=None):
    """(12, ny, nx) sea-level air temperature; over the ocean the v5.3 SST (one SST only)."""
    tau = TAU_LAND_D * cont + TAU_COAST_D * (1.0 - cont)
    r = unit_response(planet, lat, np.where(sea, TAU_SEA_D, tau))
    t = t_annual[None] + amp[None] * r
    if sst_monthly is not None:
        t = np.where(sea[None], sst_monthly, t)
    return t


def thermal_equator(lat, t_monthly, cell_km, obliquity_deg):
    """(12, nx) latitude of the warmest smoothed air per column, limited to the tropics.

    Columns without tropical rows keep the prior ``0.3 x declination`` (``idealized``).
    """
    sig = max(1.0, 800.0 / cell_km)
    lat_col = lat[:, 0]
    band = np.abs(lat_col) <= abs(obliquity_deg) + 8.0
    out = np.zeros((12, lat.shape[1]))
    decl = np.degrees(EARTH.declination(MONTH_FRACTION)) * (obliquity_deg / 23.44)
    have = band.sum() >= 3
    for m in range(12):
        prior = 0.3 * decl[m]
        if not have:
            out[m] = prior
            continue
        t = ndimage.gaussian_filter(t_monthly[m], sig, mode="nearest")
        rows = np.nonzero(band)[0]
        w = np.exp((t[rows] - t[rows].max(axis=0)) / 1.5)  # soft argmax
        phi = (w * lat_col[rows, None]).sum(axis=0) / w.sum(axis=0)
        phi = ndimage.gaussian_filter1d(phi, max(1.0, 1500.0 / cell_km), mode="nearest")
        out[m] = np.clip(phi, -abs(obliquity_deg) - 5.0, abs(obliquity_deg) + 5.0)
    return out


def zonal_slp_hpa(lat, phi_te, phi_h):
    """Zonal belts (hPa anomaly) around the thermal equator; the poles stay fixed."""
    s = lat - phi_te[None, :] * (1.0 - (lat / 90.0) ** 2)
    a = np.abs(s)
    phi_sp = phi_h + 0.5 * (90.0 - phi_h)
    w = 0.4 * phi_h
    return (
        -6.0 * np.exp(-((s / (0.27 * phi_h)) ** 2))
        + 11.0 * np.exp(-(((a - phi_h) / w) ** 2))
        - 18.0 * np.exp(-(((a - phi_sp) / w) ** 2))
        + 6.0 * np.exp(-(((a - 90.0) / w) ** 2))
    )


def frictional_wind(px, py, f, r, cap):
    """Wind of the friction-pressure-Coriolis balance (east, north); finite at f = 0."""
    den = RHO_AIR * (r * r + f * f)
    u = (-r * px - f * py) / den
    v = (f * px - r * py) / den
    sp = np.hypot(u, v)
    k = np.minimum(1.0, cap / np.maximum(sp, 1e-12))
    return u * k, v * k


def _grad_north(field_, cell_m):
    g_row, g_col = np.gradient(field_, cell_m)
    return g_col, -g_row  # d/dx east, d/dy north (rows grow southward)


@dataclass
class Circulation:
    months_days: np.ndarray
    t_sl: np.ndarray  # (12, ny, nx) sea-level air temperature degC
    t_air: np.ndarray  # (12, ny, nx) at the surface (lapse rate)
    slp_pa: np.ndarray  # (12, ny, nx)
    p_sfc_pa: np.ndarray  # (12, ny, nx)
    levels_pa: tuple  # surface + LEVELS_PA
    u: np.ndarray  # (12, 1 + L, ny, nx) m/s east; level 0 = surface (10 m)
    v: np.ndarray  # (12, 1 + L, ny, nx) m/s north
    z: np.ndarray  # (12, L, ny, nx) m geopotential height of the levels
    thermal_equator_deg: np.ndarray  # (12, nx)
    hadley_edge_deg: float
    quality: dict = field(default_factory=dict)


def build(
    planet,
    lat,
    sea,
    land_h,
    t_annual,
    amp,
    cont,
    cell_km,
    sst_monthly=None,
    dt_equator_pole_k=50.0,
    lapse_c_per_km=6.5,
    wind_strength=1.0,
) -> Circulation:
    cell_m = cell_km * 1000.0
    t_sl = monthly_temperature(planet, lat, t_annual, amp, cont, sea, sst_monthly)
    t_air = t_sl - lapse_c_per_km * land_h[None] / 1000.0
    phi_te = thermal_equator(lat, t_sl, cell_km, planet.obliquity_deg)
    phi_h, q_hadley = hadley_edge_deg(planet, dt_equator_pole_k)
    sig = max(1.0, 1000.0 / cell_km)
    f = planet.coriolis(lat)
    f_eq = abs(float(planet.coriolis(EQUATORIAL_LAT)))
    f_eff = np.where(np.abs(lat) < EQUATORIAL_LAT, np.where(lat >= 0, f_eq, -f_eq), f)
    taper = np.clip(np.abs(lat) / EQUATORIAL_LAT, 0.0, 1.0) ** 2
    r_s = np.where(sea, R_SURFACE_SEA, R_SURFACE_LAND * (1.0 + land_h / 1500.0))
    # doldrums: the weak-rotation belt is damped (convective momentum mixing) [A]
    r_s = r_s + R_EQUATORIAL * np.exp(-((lat / 8.0) ** 2))
    r_8 = R_850 + R_EQUATORIAL * np.exp(-((lat / 12.0) ** 2))
    blocked = np.exp(-land_h / 4000.0)
    ny, nx = lat.shape
    nl = len(LEVELS_PA)
    u = np.zeros((12, 1 + nl, ny, nx), dtype=np.float32)
    v = np.zeros_like(u)
    z = np.zeros((12, nl, ny, nx), dtype=np.float32)
    slp = np.zeros((12, ny, nx))
    p_sfc = np.zeros((12, ny, nx))
    for m in range(12):
        ts = ndimage.gaussian_filter(t_sl[m], sig, mode="nearest")
        t_star = ts - ts.mean(axis=1, keepdims=True)  # regional thermal anomaly in the row
        p_hpa = zonal_slp_hpa(lat, phi_te[m], phi_h) - THERMAL_HPA_PER_K * t_star
        p_hpa = wind_strength * p_hpa
        p_hpa = ndimage.gaussian_filter(p_hpa, max(1.0, 300.0 / cell_km), mode="nearest")
        slp[m] = 101100.0 + 100.0 * p_hpa
        t_k = t_air[m] + 273.15
        p_sfc[m] = slp[m] * np.exp(-G * land_h / (R_AIR * np.maximum(t_k + 0.00325 * land_h, 150)))
        px, py = _grad_north(slp[m], cell_m)
        us, vs = frictional_wind(px, py, f, r_s, SURFACE_CAP)
        u[m, 0] = us * blocked
        v[m, 0] = vs * blocked
        # 850 hPa: pressure-equivalent height = SLP + hypsometric thickness of the column
        p8 = slp[m] + 100.0 * THICKNESS_HPA_PER_K * ts
        px8, py8 = _grad_north(p8, cell_m)
        u8, v8 = frictional_wind(px8, py8, f, r_8, 60.0)
        u[m, 1], v[m, 1] = u8, v8
        # thermal wind upward from 850 hPa
        gx, gy = _grad_north(ts, cell_m)
        su = 0.8 * R_AIR / f_eff * gy * taper  # du/dln(p)
        sv = -0.8 * R_AIR / f_eff * gx * taper
        ln8 = math.log(P0_PA / LEVELS_PA[0])
        for k, p in enumerate(LEVELS_PA):
            z[m, k] = level_height_m(t_sl[m], p / 100.0)
            if k == 0:
                continue
            d = math.log(P0_PA / p) - ln8
            uu = u8 - su * d  # du/dln(p) * (ln p - ln p_850)
            vv = v8 - sv * d
            sp = np.hypot(uu, vv)
            cap = np.minimum(1.0, UPPER_CAP / np.maximum(sp, 1e-9))
            u[m, 1 + k], v[m, 1 + k] = uu * cap, vv * cap
    quality = {
        "pressure": "zonal_belts_from_thermal_structure_plus_thermal_lows",
        "surface_wind": "frictional_balance_equator_safe",
        "wind_850": "frictional_balance_on_850_height",
        "wind_500_250": "thermal_wind_equatorial_taper",
        "hadley": q_hadley,
        "thermal_equator": "warmest_column"
        if (np.abs(lat[:, 0]) <= abs(planet.obliquity_deg) + 8.0).sum() >= 3
        else "idealized_itcz",
    }
    return Circulation(
        month_days(planet),
        t_sl,
        t_air,
        slp,
        p_sfc,
        (None,) + LEVELS_PA,
        u,
        v,
        z,
        phi_te,
        phi_h,
        quality,
    )


#: quality codes of :func:`wind_at`
Q_MODELED, Q_INTERPOLATED, Q_EXTRAPOLATED, Q_BELOW_GROUND = 0, 1, 2, 3
QUALITY_NAMES = (
    "modeled_level",
    "log_p_interpolated",
    "extrapolated_above_250hPa",
    "below_ground_surface_wind",
)
#: above 250 hPa the wind weakens into the stratosphere (no modelled level): [A]
STRAT_FACTOR = {10000.0: 0.55, 5000.0: 0.4}


def wind_at(circ: Circulation, month: int, pressure_pa=None, altitude_m=None, cell=None):
    """Wind (u east, v north) and a quality code at a pressure level or altitude.

    Interpolates in ln(p) between the modelled levels (surface, 850, 500, 250 hPa);
    above 250 hPa the profile is extrapolated (quality ``extrapolated``), a level below
    the local surface returns the surface wind with ``below_ground``. ``altitude_m`` (above
    sea level) is converted with the month's level heights. ``cell`` selects flat indices.
    """
    m = int(month) % 12
    ps = circ.p_sfc_pa[m].ravel()
    us = circ.u[m].reshape(circ.u.shape[1], -1).astype(np.float64)
    vs = circ.v[m].reshape(circ.v.shape[1], -1).astype(np.float64)
    zl = circ.z[m].reshape(circ.z.shape[1], -1).astype(np.float64)
    if cell is not None:
        idx = np.asarray(cell).ravel()
        ps, us, vs, zl = ps[idx], us[:, idx], vs[:, idx], zl[:, idx]
    n = ps.size
    lev = np.array(LEVELS_PA)
    if altitude_m is not None:
        h = np.broadcast_to(np.asarray(altitude_m, dtype=np.float64).ravel(), (n,))
        # ln p linear in height between the level heights (surface: height 0 at p_sfc)
        lnp = np.empty(n)
        zz = np.vstack([np.zeros(n), zl])
        pp = np.log(np.vstack([ps, np.repeat(lev[:, None], n, axis=1)]))
        lnp[:] = pp[-1] + (h - zz[-1]) * (pp[-1] - pp[-2]) / np.maximum(zz[-1] - zz[-2], 1.0)
        for k in range(zz.shape[0] - 1):
            inside = (h >= zz[k]) & (h <= zz[k + 1])
            w = (h - zz[k]) / np.maximum(zz[k + 1] - zz[k], 1.0)
            lnp = np.where(inside, pp[k] + w * (pp[k + 1] - pp[k]), lnp)
        lnp = np.where(h < 0.0, pp[0], lnp)
        p = np.exp(lnp)
    else:
        p = np.broadcast_to(np.asarray(pressure_pa, dtype=np.float64).ravel(), (n,)).copy()
    u = np.empty(n)
    v = np.empty(n)
    q = np.full(n, Q_INTERPOLATED, dtype=np.int8)
    lnp = np.log(np.maximum(p, 1.0))
    col_p = np.vstack([ps, np.repeat(lev[:, None], n, axis=1)])
    lnc = np.log(col_p)
    u[:] = us[0]
    v[:] = vs[0]
    for k in range(col_p.shape[0] - 1):
        lo, hi = lnc[k], lnc[k + 1]  # lo > hi (pressure falls upward)
        inside = (lnp <= lo) & (lnp >= hi)
        w = (lo - lnp) / np.maximum(lo - hi, 1e-9)
        u = np.where(inside, us[k] + w * (us[k + 1] - us[k]), u)
        v = np.where(inside, vs[k] + w * (vs[k + 1] - vs[k]), v)
    for pl in lev:
        q = np.where(np.isclose(p, pl), Q_MODELED, q)
    above = p < lev[-1]
    if above.any():
        # 250 hPa -> 100 hPa -> 50 hPa: weakening factors interpolated in ln(p)
        fac = np.interp(
            np.log(p[above])[::-1],
            np.log([5000.0, 10000.0, 25000.0]),
            [STRAT_FACTOR[5000.0], STRAT_FACTOR[10000.0], 1.0],
        )[::-1]
        u[above] = us[-1][above] * fac
        v[above] = vs[-1][above] * fac
        q[above] = Q_EXTRAPOLATED
    below = p > ps
    u = np.where(below, us[0], u)
    v = np.where(below, vs[0], v)
    q = np.where(below, Q_BELOW_GROUND, q)
    return u, v, q


def level_available(circ: Circulation, month: int, pressure_pa: float) -> np.ndarray:
    """False where the pressure level lies under the ground (T06)."""
    return circ.p_sfc_pa[int(month) % 12] >= pressure_pa
