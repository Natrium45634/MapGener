"""Winds aloft (850 / 500 / 250 hPa) from the thermal wind of the season's temperature.

    du_g/dln(p) = +(R/f) dT/dy        dv_g/dln(p) = -(R/f) dT/dx          (north-up axes)
    U(p) = U_surface + shear * ln(p_s / p)

``T`` is the large-scale (1000 km smoothed) sea-level temperature of the season, the
mean tropospheric gradient being taken as 0.8 of it. ``f`` comes from the planet's
rotation; within 12 deg of the equator ``f`` is held at its 12-deg value and the shear is
tapered (an equatorial proxy, flagged, never ``1/f``). The surface wind is the v4 season
wind, so the profile starts from the climate's own boundary condition. Speeds are capped
at 90 m/s. Geopotential height of a level ``z = H ln(p_s/p) / (1 + 0.5 G H ln(p_s/p)/T)``
with ``H = R T / g`` and the lapse rate G = 6.5 K/km; a level whose height is below the
local surface is masked (``below_ground``).

The tephra transport of v5.1 and the orographic lifting of the moisture model read these
same arrays (``wind_at_height``), not a separate synthetic jet.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import ndimage

R_AIR = 287.05
G = 9.81
P_SURFACE_HPA = 1013.25
LEVELS_HPA = (850.0, 500.0, 250.0)
LAPSE = 0.0065
EQUATORIAL_LAT = 12.0
MAX_SPEED = 90.0


@dataclass
class WindsAloft:
    levels_hpa: tuple
    u: np.ndarray  # (4, L, ny, nx) m/s east
    v: np.ndarray  # (4, L, ny, nx) m/s, raster convention (+ south)
    z: np.ndarray  # (4, L, ny, nx) m geopotential height
    jet_intensity: np.ndarray  # (4, ny, nx) speed at the top level
    shear: np.ndarray  # (4, ny, nx) m/s per unit ln(p)
    quality: str = "thermal_wind_equatorial_proxy"


def level_height_m(t_surface_c, p_hpa):
    t = np.asarray(t_surface_c, dtype=np.float64) + 273.15
    ln = np.log(P_SURFACE_HPA / p_hpa)
    h = R_AIR * t / G
    return h * ln / (1.0 + 0.5 * LAPSE * h * ln / t)


def thermal_wind(t_seasons, u_seasons, v_seasons, lat, cell_km, planet,
                 levels=LEVELS_HPA) -> WindsAloft:  # fmt: skip
    """Winds aloft of the four seasons on the world raster (rows grow southward)."""
    dx = cell_km * 1000.0
    sigma = max(1.0, 1000.0 / cell_km)
    f = planet.coriolis(lat)
    f_eq = np.abs(planet.coriolis(EQUATORIAL_LAT))
    f_eff = np.where(np.abs(lat) < EQUATORIAL_LAT, np.where(lat >= 0, f_eq, -f_eq), f)
    taper = np.clip(np.abs(lat) / EQUATORIAL_LAT, 0.0, 1.0) ** 2
    nl = len(levels)
    shape = (4, nl) + lat.shape
    u_out = np.zeros(shape, dtype=np.float32)
    v_out = np.zeros(shape, dtype=np.float32)
    z_out = np.zeros(shape, dtype=np.float32)
    jets = np.zeros((4,) + lat.shape, dtype=np.float32)
    shears = np.zeros((4,) + lat.shape, dtype=np.float32)
    for s in range(4):
        t = ndimage.gaussian_filter(np.asarray(t_seasons[s], dtype=np.float64), sigma,
                                    mode="nearest")  # fmt: skip
        g_row, g_col = np.gradient(t, dx)
        dtdy_north = -g_row  # rows grow southward
        dtdx = g_col
        su = 0.8 * R_AIR / f_eff * dtdy_north * taper  # du/dln(p)
        sv = -0.8 * R_AIR / f_eff * dtdx * taper  # dv/dln(p), north-positive
        shears[s] = np.hypot(su, sv)
        for k, p in enumerate(levels):
            ln = np.log(P_SURFACE_HPA / p)
            uu = u_seasons[s] - su * ln  # du/dln(p) * (ln p - ln p_s)
            vv_north = -v_seasons[s] - sv * ln
            sp = np.hypot(uu, vv_north)
            k_cap = np.minimum(1.0, MAX_SPEED / np.maximum(sp, 1e-9))
            u_out[s, k] = uu * k_cap
            v_out[s, k] = -vv_north * k_cap
            z_out[s, k] = level_height_m(t_seasons[s], p)
        jets[s] = np.hypot(u_out[s, -1], v_out[s, -1])
    return WindsAloft(tuple(levels), u_out, v_out, z_out, jets, shears)


def wind_at_height(u_levels, v_levels, z_levels, u_surface, v_surface, height_m):
    """Wind at ``height_m`` (m above sea level) interpolated in the column.

    ``*_levels`` have the level axis first; below the lowest level the surface wind is
    blended in; above the top level the top wind is kept (``extrapolated``).
    """
    zs = np.concatenate([np.zeros((1,) + np.shape(z_levels)[1:]), np.asarray(z_levels)])
    us = np.concatenate([np.asarray(u_surface)[None], np.asarray(u_levels)])
    vs = np.concatenate([np.asarray(v_surface)[None], np.asarray(v_levels)])
    h = np.asarray(height_m, dtype=np.float64)
    out_u = np.empty(np.broadcast(h, zs[0]).shape)
    out_v = np.empty_like(out_u)
    hb = np.broadcast_to(h, out_u.shape)
    out_u[...] = us[-1]
    out_v[...] = vs[-1]
    for k in range(zs.shape[0] - 1, 0, -1):
        lo, hi = zs[k - 1], zs[k]
        inside = (hb >= lo) & (hb <= hi)
        w = np.clip((hb - lo) / np.maximum(hi - lo, 1.0), 0.0, 1.0)
        out_u = np.where(inside, us[k - 1] + w * (us[k] - us[k - 1]), out_u)
        out_v = np.where(inside, vs[k - 1] + w * (vs[k] - vs[k - 1]), out_v)
    below = hb < zs[1]
    w = np.clip(hb / np.maximum(zs[1], 1.0), 0.0, 1.0)
    out_u = np.where(below, us[0] + w * (us[1] - us[0]), out_u)
    out_v = np.where(below, vs[0] + w * (vs[1] - vs[0]), out_v)
    return out_u, out_v
