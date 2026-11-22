"""Local factors that move heat, precipitation and snow *within* a cell (TZ 4.7, 5-7).

Everything here conserves the coarse amount it redistributes: precipitation and snow are
normalised inside their group (a hex) with area weights; a factor never creates water.
"""

from __future__ import annotations

import numpy as np


def smoothstep(lo, hi, x):
    t = np.clip((np.asarray(x, dtype=np.float64) - lo) / (hi - lo), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


# -- precipitation (TZ 6.2, P0 mode ``redistribute_coarse``) ----------------------------------
def normalise_factors(factor, weight, group, n_groups, bounds=None, iterations=60):
    """Area-weighted mean of the factor is 1 in every group; optional hard caps.

    Without ``bounds``: ``f_i / sum_g(w f)``. With ``bounds = (lo, hi)``, ``lo <= 1 <= hi``:
    the positive lambda of ``sum w clip(lambda f, lo, hi) = 1`` by bisection, so every final
    factor stays inside the bounds and the mass is still conserved.
    """
    f = np.asarray(factor, dtype=np.float64)
    w = np.asarray(weight, dtype=np.float64)
    g = np.asarray(group, dtype=np.int64)
    wsum = np.bincount(g, weights=w, minlength=n_groups)
    if bounds is None:
        mean = np.bincount(g, weights=w * f, minlength=n_groups) / np.maximum(wsum, 1e-300)
        safe = mean > 1e-12
        return np.where(safe[g], f / np.where(safe, mean, 1.0)[g], 1.0)
    lo, hi = bounds
    if not lo <= 1.0 <= hi:
        raise ValueError("factor bounds must contain 1")
    lam_lo = np.full(n_groups, 0.0)
    lam_hi = np.full(n_groups, 1.0)
    # grow the upper bracket until the clipped mean reaches 1 (or the cap binds everywhere)
    for _ in range(80):
        mean = np.bincount(g, weights=w * np.clip(lam_hi[g] * f, lo, hi), minlength=n_groups)
        mean /= np.maximum(wsum, 1e-300)
        low = mean < 1.0
        if not low.any():
            break
        lam_hi = np.where(low, lam_hi * 2.0, lam_hi)
    for _ in range(iterations):
        mid = 0.5 * (lam_lo + lam_hi)
        mean = np.bincount(g, weights=w * np.clip(mid[g] * f, lo, hi), minlength=n_groups)
        mean /= np.maximum(wsum, 1e-300)
        too_low = mean < 1.0
        lam_lo = np.where(too_low, mid, lam_lo)
        lam_hi = np.where(too_low, lam_hi, mid)
    return np.clip(0.5 * (lam_lo + lam_hi)[g] * f, lo, hi)


def orographic_factor(
    uplift_ms, valley_shadow, w_scale=0.5, k_up=0.8, k_lee=0.5, k_v=0.8, fmin=0.15, fmax=5.0
):
    """Raw factor ``exp(k_up uplift - k_lee lee - k_v valley)`` from the *detail* relief [H].

    ``uplift_ms`` = U . grad(h_detail): positive on windward slopes, negative in the lee.
    """
    w = np.asarray(uplift_ms, dtype=np.float64)
    up = np.clip(np.maximum(w, 0.0) / w_scale, 0.0, 3.0)
    lee = np.clip(np.maximum(-w, 0.0) / w_scale, 0.0, 2.0)
    log_f = k_up * up - k_lee * lee - k_v * np.asarray(valley_shadow, dtype=np.float64)
    return np.exp(np.clip(log_f, np.log(fmin), np.log(fmax)))


# -- snow (TZ 7.3, ``statistical_storage_partition``) -----------------------------------------
def redistribute_snow(snowfall_mm, area, factor, eta, group, n_groups):
    """Move a mobile share ``eta`` of the new snow to sheltered units of the same group.

    ``P'_i = P_i - mobile_i + F_i sum_j A_j mobile_j / sum_j A_j F_j``: the area-integrated
    snow (mm x area) is unchanged in every group; nothing appears from the air.
    """
    p = np.asarray(snowfall_mm, dtype=np.float64)
    a = np.asarray(area, dtype=np.float64)
    f = np.asarray(factor, dtype=np.float64)
    g = np.asarray(group, dtype=np.int64)
    mobile = np.asarray(eta, dtype=np.float64) * p
    pool = np.bincount(g, weights=a * mobile, minlength=n_groups)
    denom = np.bincount(g, weights=a * f, minlength=n_groups)
    share = np.where(denom[g] > 0.0, f * pool[g] / np.where(denom[g] > 0, denom[g], 1.0), mobile)
    return p - mobile + share


def shelter_factor(sx_rad, k_s=1.2, fmin=0.4, fmax=2.5):
    """``F = exp(k_s Sx)`` bounded: sheltered units (Sx > 0) collect drifting snow [H]."""
    sx = np.nan_to_num(np.asarray(sx_rad, dtype=np.float64))
    return np.clip(np.exp(k_s * sx), fmin, fmax)


# -- cold-air pools (TZ 5.1) ---------------------------------------------------------------
def cap_static(relief_m, elevation_rank, slope_deg, saddle_minus_z_m, mean_horizon_rad):
    """Static terrain potential for cold-air pooling, 0..1 [H, after Lundquist et al. 2008]."""
    relief_gate = smoothstep(50.0, 250.0, relief_m)
    low_rank = 1.0 - smoothstep(0.15, 0.45, elevation_rank)
    flat_floor = 1.0 - smoothstep(3.0, 15.0, slope_deg)
    enclosure = smoothstep(40.0, 200.0, saddle_minus_z_m)
    shield = 1.0 - np.exp(-np.maximum(0.0, np.asarray(mean_horizon_rad)) / 0.25)
    return np.clip(relief_gate * low_rank * flat_floor * (0.5 + 0.5 * shield) * enclosure, 0, 1)


def cap_anomaly(cap, wind_ms, cloud_fraction, night_hours, amplitude_c=6.0, event_share=0.25):
    """(event amplitude, monthly mean anomaly) of a cold pool, degC (both <= 0) [H].

    A full clear, calm night gives ``-A cap``; the monthly mean gets only the night share
    of the nights that qualify (``event_share``) - never the whole night on the 24 h mean.
    """
    wind_gate = 1.0 - smoothstep(1.0, 6.0, wind_ms)
    clear_gate = 1.0 - smoothstep(0.2, 0.9, cloud_fraction)
    event = amplitude_c * np.asarray(cap) * wind_gate * clear_gate
    return -event, -event * (np.asarray(night_hours) / 24.0) * event_share


def solar_temperature_anomaly(r_slope, r_ref, wind_ms, k_t=2.0, r_scale=5.0, u0=3.0):
    """Daytime warming of sunlit slopes / cooling of shaded ones; half enters the mean [H]."""
    day = k_t * np.tanh((np.asarray(r_slope) - np.asarray(r_ref)) / r_scale)
    return 0.5 * day / (1.0 + np.asarray(wind_ms) / u0)


def wind_speed_local(u_ref, sx_rad, sx_norm=0.35, k_e=0.3, k_s=1.5, fmin=0.3, fmax=1.8):
    """Exposure / shelter factor on the reference wind speed (rough, TZ 6.1) [H]."""
    sx = np.nan_to_num(np.asarray(sx_rad, dtype=np.float64))
    shelter = np.clip(np.maximum(sx, 0.0) / sx_norm, 0.0, 1.0)
    exposure = np.clip(np.maximum(-sx, 0.0) / sx_norm, 0.0, 1.0)
    return np.asarray(u_ref) * np.clip((1.0 + k_e * exposure) / (1.0 + k_s * shelter), fmin, fmax)


def cloud_potential(rh_upstream, uplift_ms, z_m, cloud_base_m, thickness_m=900.0, exposure=0.0):
    """Orographic cloud-belt potential 0..1; needs moisture AND lift (TZ 6.4) [H]."""
    f_moist = smoothstep(0.65, 0.95, rh_upstream)
    f_lift = smoothstep(0.0, 0.6, np.maximum(np.asarray(uplift_ms), 0.0))
    center = np.asarray(cloud_base_m) + 0.5 * thickness_m
    f_layer = np.exp(-0.5 * ((np.asarray(z_m) - center) / (0.5 * thickness_m)) ** 2)
    return np.clip(f_moist * (0.3 + 0.7 * f_lift) * f_layer * (0.7 + 0.3 * exposure), 0.0, 1.0)


def fog_potential(rh_local, cap, wind_ms):
    """Radiation / valley fog potential 0..1: saturation near the ground, calm, pooling."""
    return np.clip(
        smoothstep(0.85, 1.0, rh_local)
        * (0.4 + 0.6 * np.asarray(cap))
        * (1.0 - smoothstep(2.0, 6.0, wind_ms)),
        0.0,
        1.0,
    )


# -- reference bucket of TZ 7.4 (control calculation T27) -----------------------------------
def tz_bucket_step(
    w, gw, swe, rain, melt, awc, infiltration_limit, aet, recharge_fraction, release_fraction
):
    """The TZ 7.4 baseline bucket, one month (mm). Returns the new state and the fluxes.

    The generator itself runs the v4.5 kernel on every terrain unit (one soil model);
    this function is the documented reference used to check the order of the phases.
    """
    liquid = rain + melt
    infil = min(liquid, infiltration_limit, max(0.0, awc - w))
    surface = liquid - infil
    w += infil
    aet = min(aet, w)
    w -= aet
    recharge = min(w, recharge_fraction * w)
    w -= recharge
    gw += recharge
    baseflow = min(gw, release_fraction * gw)
    gw -= baseflow
    return {
        "W": w,
        "GW": gw,
        "SWE": swe,
        "infiltration": infil,
        "surface_runoff": surface,
        "aet": aet,
        "recharge": recharge,
        "baseflow": baseflow,
        "runoff": surface + baseflow,
    }


# -- snow phase and degree-day melt (TZ 7.2; reference forms, the v4.5 kernel is used) ------
def phase_split(p_mm, t_c, snow_t_c=-1.0, rain_t_c=2.0, snow_fraction=None):
    """(snow, rain) of a precipitation sum: smooth air-temperature phase or a given fraction.

    ``snow + rain == P`` exactly; P = 0 gives zeros.
    """
    p = np.asarray(p_mm, dtype=np.float64)
    if snow_fraction is None:
        snow_fraction = 1.0 - smoothstep(snow_t_c, rain_t_c, t_c)
    snow = p * np.clip(np.asarray(snow_fraction, dtype=np.float64), 0.0, 1.0)
    return snow, p - snow


def degree_day_melt(swe_mm, k_dd_mm, pdd_c_day):
    """(potential melt, actual melt, SWE left): actual melt never exceeds the snow."""
    potential = np.asarray(k_dd_mm, dtype=np.float64) * np.maximum(pdd_c_day, 0.0)
    actual = np.minimum(np.asarray(swe_mm, dtype=np.float64), potential)
    return potential, actual, np.asarray(swe_mm, dtype=np.float64) - actual
