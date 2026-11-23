"""Monsoon / rainy-season diagnostics (TZ 5.4 sec. 7). They describe the monthly normals
computed by the moisture transport; they never generate rain.

Indices (per cell, months on the model calendar)::

    rainfall_seasonality      = std(P_m) / (mean(P_m) + eps)        (P as mm/day rates)
    wettest_3month_fraction   = max_cyclic(P_m + P_m+1 + P_m+2) / P_ann
    precip_range_mm_day       = max(P_m / d_m) - min(P_m / d_m)
    summer_fraction           = P(local MJJAS) / P_ann  (NH; NDJFM in the south)
    wind_reversal_score R_u   = 1 - |u_wet + u_dry| / (|u_wet| + |u_dry| + eps)
    suitability               = f_range f_frac (f_contrast + f_onshore) / 2 f_lat (0.8 + 0.2 R_u)

A monsoon needs a seasonal rain range, a summer share, a seasonal land-sea contrast or
onshore vapour flux; ``R_u`` alone never makes one (T40, N18). Mediterranean winter rain
is its own regime (T36). The thresholds are configurable calibration, not laws.
"""

from __future__ import annotations

import numpy as np
from scipy import ndimage

EPS = 1e-9
#: rain regimes (uint8 layer ``rain_regime``)
REGIMES = (
    "arid",
    "perennially_wet",
    "tropical_monsoon",
    "equatorial_bimodal",
    "winter_wet_mediterranean",
    "summer_wet_non_monsoon",
    "uniform_or_other",
)
NH_SUMMER = (4, 5, 6, 7, 8)  # May..September
SH_SUMMER = (10, 11, 0, 1, 2)  # November..March


def _days(days, ndim):
    return np.asarray(days, dtype=np.float64).reshape((12,) + (1,) * ndim)


def seasonality(p12, days):
    r = np.asarray(p12, dtype=np.float64) / _days(days, np.ndim(p12) - 1)
    return r.std(axis=0) / (r.mean(axis=0) + EPS)


def wettest_3month_fraction(p12):
    p = np.asarray(p12, dtype=np.float64)
    s3 = p + np.roll(p, -1, axis=0) + np.roll(p, -2, axis=0)
    return s3.max(axis=0) / np.maximum(p.sum(axis=0), EPS)


def wettest_3month_start(p12):
    p = np.asarray(p12, dtype=np.float64)
    s3 = p + np.roll(p, -1, axis=0) + np.roll(p, -2, axis=0)
    return s3.argmax(axis=0)


def precip_range_mm_day(p12, days):
    r = np.asarray(p12, dtype=np.float64) / _days(days, np.ndim(p12) - 1)
    return r.max(axis=0) - r.min(axis=0)


def summer_fraction(p12, lat):
    p = np.asarray(p12, dtype=np.float64)
    nh = p[list(NH_SUMMER)].sum(axis=0)
    sh = p[list(SH_SUMMER)].sum(axis=0)
    return np.where(np.asarray(lat) >= 0.0, nh, sh) / np.maximum(p.sum(axis=0), EPS)


def reversal_score(u_wet, v_wet, u_dry, v_dry):
    s = np.hypot(u_wet + u_dry, v_wet + v_dry)
    return 1.0 - s / (np.hypot(u_wet, v_wet) + np.hypot(u_dry, v_dry) + 1e-6)


def wet_dry_months(p12, pet12):
    """Humid months (P >= PET) and dry months (P < PET / 2) with a valid PET."""
    p = np.asarray(p12, dtype=np.float64)
    pet = np.asarray(pet12, dtype=np.float64)
    valid = np.isfinite(pet)
    wet = ((p >= pet) & valid).sum(axis=0)
    dry = ((p < 0.5 * pet) & valid & (pet > 0)).sum(axis=0)
    return wet.astype(np.uint8), dry.astype(np.uint8)


def _runs(mask12):
    """Cyclic runs of True in a 12-vector: list of (start, length)."""
    m = list(bool(x) for x in mask12)
    if all(m):
        return [(0, 12)]
    if not any(m):
        return []
    start = m.index(False)  # rotate so a run never wraps
    rot = m[start + 1 :] + m[: start + 1]
    runs, k = [], 0
    while k < 12:
        if rot[k]:
            j = k
            while j < 12 and rot[j]:
                j += 1
            runs.append(((k + start + 1) % 12, j - k))
            k = j
        else:
            k += 1
    return runs


def onset_withdrawal(p12, days, min_range_mm_day=1.0, min_seasonality=0.3):
    """Climatological onset / withdrawal months (1..12, 0 = not defined) and wet seasons.

    A wet month has a rate above ``mean + 0.25 (max - min)``; the longest cyclic run of wet
    months is the main rainy season, a second separate run the second one. No season is
    defined when the annual cycle is too weak (perennially wet or arid, T35).
    """
    p = np.asarray(p12, dtype=np.float64).reshape(12, -1)
    d = np.asarray(days, dtype=np.float64).reshape(12, 1)
    r = p / d
    rng = r.max(axis=0) - r.min(axis=0)
    sea = r.std(axis=0) / (r.mean(axis=0) + EPS)
    thr = r.mean(axis=0) + 0.25 * rng
    n = p.shape[1]
    onset = np.zeros(n, np.uint8)
    withdrawal = np.zeros(n, np.uint8)
    onset2 = np.zeros(n, np.uint8)
    seasons = np.zeros(n, np.uint8)
    ok = (rng >= min_range_mm_day) & (sea >= min_seasonality)
    wet = r >= thr[None]
    for i in np.nonzero(ok)[0]:
        runs = _runs(wet[:, i])
        if not runs or runs == [(0, 12)]:
            continue
        runs.sort(key=lambda t: (-t[1], t[0]))
        s0, l0 = runs[0]
        onset[i] = s0 + 1
        withdrawal[i] = (s0 + l0 - 1) % 12 + 1
        seasons[i] = len(runs)
        if len(runs) > 1:
            onset2[i] = runs[1][0] + 1
    shape = np.shape(p12)[1:]
    return (
        onset.reshape(shape),
        withdrawal.reshape(shape),
        onset2.reshape(shape),
        seasons.reshape(shape),
    )


def peak_count(p12, prominence=0.2):
    """Cyclic local maxima of the 3-month-smoothed rain above ``prominence`` x range."""
    p = np.asarray(p12, dtype=np.float64)
    s = (np.roll(p, 1, axis=0) + p + np.roll(p, -1, axis=0)) / 3.0
    rng = s.max(axis=0) - s.min(axis=0)
    is_max = (s > np.roll(s, 1, axis=0)) & (s >= np.roll(s, -1, axis=0))
    # prominence: the peak stands above the lowest month between it and the next peak
    count = np.zeros(p.shape[1:], dtype=np.uint8)
    low = s.min(axis=0)
    count += (
        (is_max & (s - low[None] >= prominence * np.maximum(rng, EPS)[None])).sum(axis=0)
    ).astype(np.uint8)
    return np.where(rng > EPS, count, 0).astype(np.uint8)


def suitability(range_mm_day, summer_frac, contrast_k, onshore_change, reversal, lat):
    f_range = np.clip((range_mm_day - 1.0) / 3.0, 0.0, 1.0)
    f_frac = np.clip((summer_frac - 0.5) / 0.25, 0.0, 1.0)
    f_con = np.clip(contrast_k / 6.0, 0.0, 1.0)
    f_on = np.clip(onshore_change / 100.0, 0.0, 1.0)
    f_lat = np.clip((40.0 - np.abs(lat)) / 10.0, 0.0, 1.0)
    return f_range * f_frac * 0.5 * (f_con + f_on) * f_lat * (0.8 + 0.2 * np.clip(reversal, 0, 1))


def regime(p_ann, dry_months, seas, suit, peaks, summer_frac, lat, p12=None):
    """Rain regime codes (``REGIMES``).

    Mediterranean (winter-wet) follows the Koppen ``Cs`` test when the months are given:
    the driest summer month below 40 mm and below a third of the wettest winter month,
    at 25-48 deg; otherwise a winter share >= 0.6 with two dry months.
    """
    p_ann = np.asarray(p_ann, dtype=np.float64)
    out = np.full(p_ann.shape, 6, dtype=np.uint8)
    winter_frac = 1.0 - summer_frac
    out = np.where(summer_frac >= 0.6, 5, out)
    belt = (np.abs(lat) >= 25.0) & (np.abs(lat) <= 48.0)
    if p12 is not None:
        p = np.asarray(p12, dtype=np.float64)
        north = np.asarray(lat) >= 0.0
        summer = np.where(north, p[[5, 6, 7]].min(axis=0), p[[11, 0, 1]].min(axis=0))
        winter = np.where(north, p[[10, 11, 0, 1, 2]].max(axis=0), p[[4, 5, 6, 7, 8]].max(axis=0))
        med = belt & (summer < 40.0) & (summer < winter / 3.0) & (winter_frac >= 0.5)
    else:
        med = belt & (winter_frac >= 0.6) & (dry_months >= 2)
    out = np.where(med, 4, out)
    out = np.where((peaks >= 2) & (np.abs(lat) < 20.0) & (suit < 0.4), 3, out)
    out = np.where(suit >= 0.4, 2, out)
    out = np.where((dry_months == 0) & (seas < 0.4) & (p_ann >= 1200.0), 1, out)
    out = np.where(p_ann < 200.0, 0, out)
    return out.astype(np.uint8)


# -- grid diagnostics (wind- and SST-based; on the atmosphere grid) ----------------------
def land_sea_contrast(t_air, sst, sea_frac, u, v, cell_km, reach_km=1000.0):
    """(12, ny, nx) regional land T minus the SST upwind (TZ eq. dT_LS), K."""
    sig = max(1.0, 500.0 / cell_km)
    land_w = 1.0 - sea_frac
    sea_w = sea_frac
    ny, nx = sea_frac.shape
    jj, ii = np.mgrid[0:ny, 0:nx].astype(np.float64)
    out = np.zeros_like(t_air)
    for m in range(12):
        tl = ndimage.gaussian_filter(t_air[m] * land_w, sig, mode="nearest") / np.maximum(
            ndimage.gaussian_filter(land_w, sig, mode="nearest"), 1e-6
        )
        ws = ndimage.gaussian_filter(sea_w, 2 * sig, mode="nearest")
        ss = ndimage.gaussian_filter(sst[m] * sea_w, 2 * sig, mode="nearest") / np.maximum(ws, 1e-6)
        sp = np.maximum(np.hypot(u[m], v[m]), 0.5)
        k = reach_km / cell_km / sp
        up = ndimage.map_coordinates(ss, [jj + v[m] * k, ii - u[m] * k], order=1, mode="nearest")
        has_sea = (
            ndimage.map_coordinates(ws, [jj + v[m] * k, ii - u[m] * k], order=1, mode="nearest")
            > 0.02
        )
        out[m] = np.where(has_sea, tl - up, 0.0)
    return out


def landward_normal(sea_frac, cell_km):
    """Unit vector (east, north) pointing from the sea into the land (distance gradient)."""
    sea = sea_frac > 0.5
    if not sea.any() or sea.all():
        z = np.zeros(sea.shape)
        return z, z
    d = ndimage.distance_transform_edt(~sea) * cell_km
    d = ndimage.gaussian_filter(d, 1.0)
    g_row, g_col = np.gradient(d)
    ne, nn = g_col, -g_row
    s = np.maximum(np.hypot(ne, nn), 1e-9)
    return ne / s, nn / s


def source_distance_km(sea_frac, u, v, cell_km, max_km=6000.0):
    """Distance upwind along the (annual, vapour-weighted) wind to the nearest sea, km."""
    ny, nx = sea_frac.shape
    sea = sea_frac > 0.5
    jj, ii = np.mgrid[0:ny, 0:nx].astype(np.float64)
    sp = np.maximum(np.hypot(u, v), 1e-6)
    ue, vn = u / sp, v / sp
    dist = np.where(sea, 0.0, np.nan)
    step = 0.5
    y, x = jj.copy(), ii.copy()
    travelled = 0.0
    while travelled < max_km and np.isnan(dist).any():
        de = ndimage.map_coordinates(ue, [y, x], order=1, mode="nearest")
        dn = ndimage.map_coordinates(vn, [y, x], order=1, mode="nearest")
        x = x - de * step
        y = y + dn * step  # rows grow southward; going upwind against v_north
        travelled += step * cell_km
        inside = (x >= 0) & (x <= nx - 1) & (y >= 0) & (y <= ny - 1)
        hit = ndimage.map_coordinates(sea.astype(float), [y, x], order=0, mode="nearest") > 0.5
        new = np.isnan(dist) & (hit | ~inside)
        dist = np.where(new & hit, travelled, dist)
        dist = np.where(new & ~hit, max_km, dist)  # left the domain over land
    return np.nan_to_num(dist, nan=max_km)


def detect_onset_daily(p_daily, k=5, threshold_mm=None, persist_days=10, wet_share=0.5):
    """Onset day of a rainy season in a daily series (index, or -1): false starts rejected.

    The ``k``-day running sum must exceed ``threshold_mm`` (default: k x the series mean
    + 1 mm/day), and in the following ``persist_days`` at least ``wet_share`` of the days
    must be wet (> 1 mm): one storm followed by a dry spell does not start the season.
    """
    p = np.asarray(p_daily, dtype=np.float64)
    if threshold_mm is None:
        threshold_mm = k * (p.mean() + 1.0)
    run = np.convolve(p, np.ones(k), mode="valid")
    for d in np.nonzero(run >= threshold_mm)[0]:
        after = p[d + k : d + k + persist_days]
        if after.size >= persist_days and (after > 1.0).mean() >= wet_share:
            return int(d)
    return -1


def break_days(p_daily, onset, withdrawal, dry_mm=1.0, min_len=5):
    """Days inside the rainy season belonging to dry spells of ``min_len`` days or more."""
    if onset < 0 or withdrawal <= onset:
        return 0
    seg = np.asarray(p_daily[onset:withdrawal]) < dry_mm
    total, run = 0, 0
    for x in list(seg) + [False]:
        if x:
            run += 1
        else:
            if run >= min_len:
                total += run
            run = 0
    return total
