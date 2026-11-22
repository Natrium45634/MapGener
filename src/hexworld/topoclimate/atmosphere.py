"""Vertical temperature profile, pressure, humidity, dew point, LCL (TZ 4.7, 3, 6.4).

Every function states its units. ``[S]`` published formula, ``[H]`` HexWorld adaptation.
"""

from __future__ import annotations

import numpy as np

G0 = 9.80665  # m/s^2
R_D = 287.05  # J/(kg K)
EPS = 0.622


def lapse_temperature(t_ref_c, z_ref_m, z_m, lapse_k_km=6.5):
    """Air temperature at ``z`` from ``t_ref`` measured at ``z_ref`` (constant lapse) [H].

    The reference height is where the coarse temperature *was computed* - never 0 by
    default - so the correction is applied once. Negative heights (dry land below the
    datum) are valid: ``z = -300, z_ref = 0`` is 1.95 degC warmer at 6.5 K/km.
    """
    return (
        np.asarray(t_ref_c, dtype=np.float64)
        - lapse_k_km
        * (np.asarray(z_m, dtype=np.float64) - np.asarray(z_ref_m, dtype=np.float64))
        / 1000.0
    )


def preserve_mean(t_ref_c, anomaly_c, weights):
    """``preserve_coarse_mean``: T_i = T_ref + dT_i - sum(w dT) (weights sum to 1)."""
    a = np.asarray(anomaly_c, dtype=np.float64)
    w = np.asarray(weights, dtype=np.float64)
    return np.asarray(t_ref_c, dtype=np.float64) + a - float(np.sum(w * a) / np.sum(w))


def pressure_exponential_kpa(p_ref_kpa, z_ref_m, z_m, tv_mean_k):
    """Hypsometric equation with a mean virtual temperature [S]."""
    dz = np.asarray(z_m, dtype=np.float64) - np.asarray(z_ref_m, dtype=np.float64)
    return np.asarray(p_ref_kpa, dtype=np.float64) * np.exp(
        -G0 * dz / (R_D * np.asarray(tv_mean_k))
    )


def pressure_fao_kpa(z_m):
    """FAO-56 eq. 7, Earth-like troposphere: 101.3 ((293 - 0.0065 z) / 293)^5.26 [S]."""
    z = np.asarray(z_m, dtype=np.float64)
    return 101.3 * ((293.0 - 0.0065 * z) / 293.0) ** 5.26


def es_kpa(t_c):
    """Saturation vapour pressure over liquid water, FAO-56 eq. 11 [S]."""
    t = np.asarray(t_c, dtype=np.float64)
    return 0.6108 * np.exp(17.27 * t / (t + 237.3))


def delta_kpa_k(t_c):
    t = np.asarray(t_c, dtype=np.float64)
    return 4098.0 * es_kpa(t) / (t + 237.3) ** 2


def specific_humidity(e_kpa, p_kpa):
    e = np.asarray(e_kpa, dtype=np.float64)
    return EPS * e / (np.asarray(p_kpa, dtype=np.float64) - (1.0 - EPS) * e)


def vapour_pressure(q, p_kpa):
    q = np.asarray(q, dtype=np.float64)
    return np.asarray(p_kpa, dtype=np.float64) * q / (EPS + (1.0 - EPS) * q)


def local_humidity(t_ref_c, rh_ref, p_ref_kpa, t_c, p_kpa):
    """Carry the specific humidity of the reference air to the local level [H fallback].

    Returns ``(rh, vpd_kpa, condensation)``: RH is capped at 1; where the parcel would be
    supersaturated ``condensation`` is True (``potential_condensation``, not extra rain).
    """
    e_ref = np.clip(rh_ref, 0.0, 1.0) * es_kpa(t_ref_c)
    q = specific_humidity(e_ref, p_ref_kpa)
    e = vapour_pressure(q, p_kpa)
    es = es_kpa(t_c)
    rh_raw = e / es
    rh = np.clip(rh_raw, 0.0, 1.0)
    return rh, np.maximum(es - np.minimum(e, es), 0.0), rh_raw > 1.0


def dew_point_c(t_c, rh):
    """Magnus over liquid water (a = 17.625, b = 243.04 degC) [S]; NaN for RH <= 0."""
    a, b = 17.625, 243.04
    rh = np.asarray(rh, dtype=np.float64)
    with np.errstate(divide="ignore", invalid="ignore"):
        g = np.log(np.where(rh > 0.0, rh, np.nan)) + a * np.asarray(t_c) / (b + np.asarray(t_c))
        return b * g / (a - g)


def lcl_height_agl_m(t_c, td_c):
    """Rule of thumb ``125 (T - Td)`` m above the parcel's origin [H]; 0 at saturation."""
    return np.maximum(125.0 * (np.asarray(t_c) - np.asarray(td_c)), 0.0)


def mass_elevation_effect(
    z_large_mean_m, dist_to_low_km, amplitude_c, z_low_ref=500.0, z_scale=2000.0, l_mass_km=150.0
):
    """Interior warmth of large high massifs (a game parameter, off by default) [H].

    ``M = clip((H_large - z_low) / Zscale, 0, 1) * (1 - exp(-dist / Lmass))``; ``dT = A M``.
    """
    mass_z = np.clip((np.asarray(z_large_mean_m) - z_low_ref) / z_scale, 0.0, 1.0)
    interior = 1.0 - np.exp(-np.asarray(dist_to_low_km, dtype=np.float64) / l_mass_km)
    return float(amplitude_c) * mass_z * interior
