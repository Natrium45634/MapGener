"""Reference evapotranspiration: FAO-56 Penman-Monteith with the FAO fallbacks.

``fao56_daily`` is the standard grass-reference equation (FAO-56 eq. 6)::

    ET0 = [0.408 D (Rn - G) + g 900/(T+273) u2 (es - ea)] / [D + g (1 + 0.34 u2)]   mm/day

with ``D`` (kPa/K), ``g = 0.665e-3 P`` (kPa/K, P from the elevation), ``Rn, G`` in
MJ m^-2 day^-1, ``u2`` m/s, ``es, ea`` kPa. "day" is 86 400 s whatever the planet's
rotation; ``Ra`` comes from :mod:`.planet` as a rotation-mean flux in that unit, so the
constant 900 keeps its meaning.

The monthly climate gives T, P and the wind, but no measured humidity, sunshine or
T_max / T_min. Missing inputs are estimated **with the procedures FAO-56 itself gives for
missing data** (chapter 3), and the method is recorded (``FAO56_estimated_inputs``):

* T_max / T_min from an estimated diurnal range ``DTR = 15 - 8 h`` (h: wetness of the month
  ``P / (P + 50 mm)``; 7..15 K, smaller at the coast);
* ``ea = e0(T_min - K0)`` (eq. 48), ``K0 = 0`` in humid months, up to 3 K when dry;
* ``Rs = k_Rs sqrt(DTR) Ra`` (Hargreaves radiation, eq. 50), ``k_Rs`` 0.16 inland /
  0.19 at the coast, capped by ``Rso = (0.75 + 2e-5 z) Ra``;
* ``Rn = (1 - a) Rs - Rnl`` (eq. 39), a = 0.23 (0.6 on snow); ``G = 0.14 (T_m - T_m-1)``;
* ``u2 = 0.748 u10`` (eq. 47), at least 0.5 m/s.

``hargreaves_month`` (eq. 52) is the temperature-radiation fallback.
"""

from __future__ import annotations

import numpy as np

from .planet import MONTH_FRACTION

SIGMA_MJ = 4.903e-9  # MJ K^-4 m^-2 day^-1
MONTH_DAYS = np.array([31, 28.25, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31], dtype=np.float64)


def sat_vapour_kpa(t_c):
    t = np.asarray(t_c, dtype=np.float64)
    return 0.6108 * np.exp(17.27 * t / (t + 237.3))


def slope_kpa_k(t_c):
    t = np.asarray(t_c, dtype=np.float64)
    return 4098.0 * sat_vapour_kpa(t) / (t + 237.3) ** 2


def pressure_kpa(z_m):
    return 101.3 * ((293.0 - 0.0065 * np.asarray(z_m, dtype=np.float64)) / 293.0) ** 5.26


def fao56_daily(t_mean, t_max, t_min, ea_kpa, u2, rn_mj, g_mj, z_m):
    """FAO-56 reference ET0, mm per 86 400 s (eq. 6); negative values clipped to 0."""
    es = 0.5 * (sat_vapour_kpa(t_max) + sat_vapour_kpa(t_min))
    d = slope_kpa_k(t_mean)
    gam = 0.665e-3 * pressure_kpa(z_m)
    u2 = np.asarray(u2, dtype=np.float64)
    num = 0.408 * d * (rn_mj - g_mj) + gam * 900.0 / (np.asarray(t_mean) + 273.0) * u2 * np.maximum(
        es - ea_kpa, 0.0
    )
    return np.maximum(num / (d + gam * (1.0 + 0.34 * u2)), 0.0)


def net_longwave(t_max, t_min, ea_kpa, rs, rso):
    tk4 = 0.5 * ((np.asarray(t_max) + 273.16) ** 4 + (np.asarray(t_min) + 273.16) ** 4)
    ratio = np.clip(np.where(rso > 0, rs / np.maximum(rso, 1e-9), 0.5), 0.3, 1.0)
    return SIGMA_MJ * tk4 * (0.34 - 0.14 * np.sqrt(np.maximum(ea_kpa, 0.0))) * (1.35 * ratio - 0.35)


def monthly_inputs(t_m, p_m, u10, lat, z, planet, coastal=None):
    """Estimated daily-mean inputs of every month (dict of (12, N) arrays) + method."""
    t_m = np.asarray(t_m, dtype=np.float64)
    p_m = np.maximum(np.asarray(p_m, dtype=np.float64), 0.0)
    lat = np.broadcast_to(np.asarray(lat, dtype=np.float64), t_m.shape[1:])
    z = np.broadcast_to(np.asarray(z, dtype=np.float64), t_m.shape[1:])
    coast = (np.zeros(t_m.shape[1:], bool) if coastal is None
             else np.broadcast_to(np.asarray(coastal, bool), t_m.shape[1:]))  # fmt: skip
    wet = p_m / (p_m + 50.0)
    dtr = np.clip(15.0 - 8.0 * wet, 7.0, 15.0) * np.where(coast, 0.75, 1.0)
    t_max, t_min = t_m + 0.5 * dtr, t_m - 0.5 * dtr
    k0 = np.where(coast, 0.0, 3.0 * np.clip(1.0 - p_m / 60.0, 0.0, 1.0))
    ea = sat_vapour_kpa(t_min - k0)
    ra = np.stack([planet.ra_mj(lat, MONTH_FRACTION[m]) for m in range(12)])
    k_rs = np.where(coast, 0.19, 0.16)
    rso = (0.75 + 2e-5 * z) * ra
    rs = np.minimum(k_rs * np.sqrt(dtr) * ra, rso)
    albedo = np.where(t_m < -2.0, 0.6, 0.23)
    rn = (1.0 - albedo) * rs - net_longwave(t_max, t_min, ea, rs, rso)
    g = 0.14 * (t_m - np.roll(t_m, 1, axis=0))
    u2 = np.maximum(0.748 * np.broadcast_to(np.asarray(u10, dtype=np.float64), t_m.shape), 0.5)
    return {"t_mean": t_m, "t_max": t_max, "t_min": t_min, "ea": ea, "u2": u2, "rn": rn,
            "g": g, "ra": ra, "rs": rs, "dtr": dtr, "vpd": np.maximum(
                0.5 * (sat_vapour_kpa(t_max) + sat_vapour_kpa(t_min)) - ea, 0.0)}  # fmt: skip


def fao56_monthly(t_m, p_m, u10, lat, z, planet, coastal=None):
    """(12, N) monthly ET0 in mm and the estimated inputs."""
    x = monthly_inputs(t_m, p_m, u10, lat, z, planet, coastal)
    zz = np.broadcast_to(np.asarray(z, dtype=np.float64), np.shape(t_m)[1:])
    day = fao56_daily(x["t_mean"], x["t_max"], x["t_min"], x["ea"], x["u2"], x["rn"], x["g"], zz)
    days = MONTH_DAYS.reshape((12,) + (1,) * (np.ndim(t_m) - 1))
    return day * days, x


def hargreaves_monthly(t_m, p_m, lat, planet):
    """FAO-56 eq. 52 (temperature + radiation), mm per month."""
    x = monthly_inputs(t_m, p_m, 2.0, lat, 0.0, planet)
    day = 0.0023 * (x["t_mean"] + 17.8) * np.sqrt(x["dtr"]) * x["ra"] * 0.408
    days = MONTH_DAYS.reshape((12,) + (1,) * (np.ndim(t_m) - 1))
    return np.maximum(day, 0.0) * days, x
