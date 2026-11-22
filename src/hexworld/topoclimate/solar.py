"""Sun geometry, horizons, sky view and shaded radiation on slopes (TZ 4.7, 2.3, 4).

Conventions (TZ 4.7, 0.5): local ENU - x east, y **north**, z up. The world of HexWorld has
+y to the *south*, so ``north = -y_world``; every function here works in ENU and the
callers convert once. Azimuths are clockwise from north, radians. Energy: W/m^2 for flux
densities, MJ/m^2/day for daily sums (seconds, not hours, in the integration).
"""

from __future__ import annotations

import numpy as np

GSC_MJ_MIN = 0.0820  # FAO-56 solar constant, MJ m-2 min-1
I_SC_W = GSC_MJ_MIN * 1e6 / 60.0  # ~1366.7 W/m^2
MONTH_DAYS = np.array([31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31])
_STARTS = np.concatenate([[0], np.cumsum(MONTH_DAYS)[:-1]])
#: three representative days per month (start, middle, end) with equal weights
REPRESENTATIVE_DOY = np.stack(
    [_STARTS + 4, _STARTS + MONTH_DAYS // 2 + 1, _STARTS + MONTH_DAYS - 3], 1
)


def declination(doy, tilt_rad=0.409):
    """FAO-56 eq. 24: delta = 0.409 sin(2 pi J / 365 - 1.39) (tilt replaceable)."""
    return tilt_rad * np.sin(2.0 * np.pi * np.asarray(doy) / 365.0 - 1.39)


def inverse_distance(doy):
    return 1.0 + 0.033 * np.cos(2.0 * np.pi * np.asarray(doy) / 365.0)


def sunset_hour_angle(lat_rad, decl_rad):
    """H0 = acos(-tan phi tan delta), clipped: pi in polar day, 0 in polar night."""
    with np.errstate(invalid="ignore", over="ignore"):
        x = -np.tan(np.asarray(lat_rad)) * np.tan(np.asarray(decl_rad))
    return np.arccos(np.clip(np.nan_to_num(x, nan=0.0, posinf=1.0, neginf=-1.0), -1.0, 1.0))


def day_length_h(lat_rad, decl_rad):
    return 24.0 * sunset_hour_angle(lat_rad, decl_rad) / np.pi


def sun_vector(lat_rad, decl_rad, hour_angle_rad):
    """Unit vector to the sun in ENU (TZ 4.7, 4.1)."""
    phi = np.asarray(lat_rad, dtype=np.float64)
    d = np.asarray(decl_rad, dtype=np.float64)
    h = np.asarray(hour_angle_rad, dtype=np.float64)
    sx = -np.cos(d) * np.sin(h)
    sy = np.cos(phi) * np.sin(d) - np.sin(phi) * np.cos(d) * np.cos(h)
    sz = np.sin(phi) * np.sin(d) + np.cos(phi) * np.cos(d) * np.cos(h)
    return sx, sy, sz


def ra_analytic(lat_rad, doy):
    """FAO-56 eq. 21, MJ/m^2/day; 0 in polar night."""
    phi = np.asarray(lat_rad, dtype=np.float64)
    d = declination(doy)
    h0 = sunset_hour_angle(phi, d)
    return (
        (24.0 * 60.0 / np.pi)
        * GSC_MJ_MIN
        * inverse_distance(doy)
        * (h0 * np.sin(phi) * np.sin(d) + np.cos(phi) * np.cos(d) * np.sin(h0))
    )


def surface_normal(p_east, q_north):
    """Unit normal (ENU) of a surface with gradient (dz/dE, dz/dN)."""
    p = np.asarray(p_east, dtype=np.float64)
    q = np.asarray(q_north, dtype=np.float64)
    norm = np.sqrt(1.0 + p * p + q * q)
    return -p / norm, -q / norm, 1.0 / norm


def slope_aspect(p_east, q_north, flat_tol=1e-4):
    """Slope (rad) and downslope aspect (rad, clockwise from north); aspect NaN when flat."""
    p = np.asarray(p_east, dtype=np.float64)
    q = np.asarray(q_north, dtype=np.float64)
    slope = np.arctan(np.hypot(p, q))
    aspect = np.mod(np.arctan2(-p, -q), 2.0 * np.pi)
    return slope, np.where(np.hypot(p, q) > flat_tol, aspect, np.nan)


def horizon_angles(dist_m, dz_m, azimuth_rad, n_dir: int, signed: bool = False):
    """Horizon per direction bin from obstacles at (distance, height difference, azimuth).

    Inputs (..., m): one row per observer, ``m`` obstacles. Returns (..., n_dir) angles
    in rad: the maximum elevation angle per azimuth sector; 0 where nothing rises
    (``signed=False``), or the true signed maximum (negative = observer above everything)
    with ``signed=True`` - the Winstral ``Sx`` needs the sign.
    """
    ang = np.arctan2(np.asarray(dz_m, dtype=np.float64), np.maximum(np.asarray(dist_m), 1e-6))
    sector = np.floor(np.mod(np.asarray(azimuth_rad), 2.0 * np.pi) / (2.0 * np.pi) * n_dir)
    sector = np.clip(sector.astype(np.int64), 0, n_dir - 1)
    out = np.full(ang.shape[:-1] + (n_dir,), -np.pi / 2.0)
    valid = np.isfinite(ang)
    for k in range(n_dir):
        hit = valid & (sector == k)
        out[..., k] = np.max(np.where(hit, ang, -np.pi / 2.0), axis=-1)
    if signed:
        return np.where(out <= -np.pi / 2.0 + 1e-9, np.nan, out)
    return np.maximum(out, 0.0)


def sky_view(horizon_rad, slope_rad=None):
    """(visible sky fraction, radiative SVF of a horizontal plane, approx. SVF of a slope).

    ``V = mean(1 - sin H+)``, ``SVF_h = mean(cos^2 H+)``; for a slope the budget form
    ``SVF_h (1 + cos beta) / 2`` - an approximation, flagged as such (TZ 4.7, 2.3).
    """
    h = np.clip(np.asarray(horizon_rad, dtype=np.float64), 0.0, np.pi / 2.0)
    vis = np.mean(1.0 - np.sin(h), axis=-1)
    svf_h = np.mean(np.cos(h) ** 2, axis=-1)
    if slope_rad is None:
        return vis, svf_h, svf_h
    return vis, svf_h, svf_h * (1.0 + np.cos(slope_rad)) / 2.0


def monthly_radiation(
    lat_rad,
    normal,
    horizon,
    svf_slope,
    slope_rad,
    sunshine_frac,
    n_hours: int = 12,
    albedo: float = 0.2,
):
    """Shortwave radiation of every unit and month, MJ/m^2/day (surface and horizontal).

    ``normal`` = (nx, ny, nz) ENU arrays (U,); ``horizon`` (U, K) rad; ``sunshine_frac``
    (12, U) in 0..1. Angstrom-Prescott gives the horizontal global radiation from the TOA
    radiation and the sunshine fraction (``Rs = (0.25 + 0.5 n/N) Ra``, FAO-56 eq. 35 [S]);
    the diffuse share grows with cloudiness [H]. On the slope: direct ``B_n max(0, n.s)``
    where the sun is above the horizon of its azimuth, diffuse ``D_h SVF``, ground
    reflection ``albedo GHI (1 - cos beta) / 2``. Three days per month, ``n_hours``
    midpoint samples of the daylight; polar night gives zeros, never a division by zero.
    """
    lat = np.asarray(lat_rad, dtype=np.float64)
    nx, ny, nz = (np.asarray(a, dtype=np.float64) for a in normal)
    hor = np.asarray(horizon, dtype=np.float64)
    k_dir = hor.shape[-1]
    u = lat.size
    r_slope = np.zeros((12, u))
    r_ref = np.zeros((12, u))
    sun_h = np.zeros((12, u))
    refl = albedo * (1.0 - np.cos(slope_rad)) / 2.0
    for m in range(12):
        s = np.clip(np.asarray(sunshine_frac[m], dtype=np.float64), 0.0, 1.0)
        kd = np.clip(1.0 - 0.75 * s, 0.15, 1.0)
        trans = 0.25 + 0.5 * s
        for doy in REPRESENTATIVE_DOY[m]:
            d = declination(doy)
            dr = inverse_distance(doy)
            h0 = sunset_hour_angle(lat, d)
            dt = (2.0 * h0 / n_hours) * 86400.0 / (2.0 * np.pi)  # seconds per sample
            for k in range(n_hours):
                h = -h0 + (k + 0.5) * 2.0 * h0 / n_hours
                sx, sy, sz = sun_vector(lat, d, h)
                up = sz > 0.0
                toa_h = I_SC_W * dr * np.maximum(sz, 0.0)
                ghi = trans * toa_h
                b_h = ghi * (1.0 - kd)
                d_h = ghi * kd
                b_n = np.where(up, b_h / np.maximum(sz, 0.05), 0.0)
                az = np.mod(np.arctan2(sx, sy), 2.0 * np.pi)
                alt = np.arcsin(np.clip(sz, -1.0, 1.0))
                sec = np.clip((az / (2.0 * np.pi) * k_dir).astype(np.int64), 0, k_dir - 1)
                lit = up & (alt > np.take_along_axis(hor, sec[:, None], axis=1)[:, 0])
                cos_i = np.maximum(nx * sx + ny * sy + nz * sz, 0.0)
                flux = np.where(lit, b_n * cos_i, 0.0) + d_h * svf_slope + refl * ghi
                r_slope[m] += flux * dt * 1e-6 / 3.0
                r_ref[m] += ghi * dt * 1e-6 / 3.0
                sun_h[m] += np.where(lit & (cos_i > 0), dt / 3600.0, 0.0) / 3.0
    return r_slope, r_ref, sun_h


def toa_flat_numeric(lat_rad, doy, n_hours: int = 48) -> np.ndarray:
    """Numerical TOA radiation of a flat unobstructed plane, MJ/m^2/day (test oracle)."""
    lat = np.asarray(lat_rad, dtype=np.float64)
    d = declination(doy)
    dr = inverse_distance(doy)
    h0 = sunset_hour_angle(lat, d)
    total = np.zeros(lat.shape)
    dt = (2.0 * h0 / n_hours) * 86400.0 / (2.0 * np.pi)
    for k in range(n_hours):
        h = -h0 + (k + 0.5) * 2.0 * h0 / n_hours
        _sx, _sy, sz = sun_vector(lat, d, h)
        total += I_SC_W * dr * np.maximum(sz, 0.0) * dt * 1e-6
    return total
