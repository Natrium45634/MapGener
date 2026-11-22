"""Growing season, thermal tree limit and thermal belts (TZ 4.7, 8, 9).

[S] Paulsen & Koerner 2014 (TREELIM): days with mean T > 0.9 degC; the tree limit needs a
season of >= 94 such days with a mean of >= 6.4 degC. These are empirical Earth reference
values, kept as parameters. [H] the smoothing widths and the belt table are HexWorld's.
The thermal limit is a *potential*: it never claims that a forest exists.
"""

from __future__ import annotations

import numpy as np

from .solar import MONTH_DAYS

DAYS = 365
_MID = np.cumsum(MONTH_DAYS) - MONTH_DAYS / 2.0  # day of the month centres
_MONTH_OF_DAY = np.repeat(np.arange(12), MONTH_DAYS)

BELTS = (
    "unknown",
    "thermal_nival",
    "thermal_subnival",
    "thermal_upper_alpine",
    "thermal_lower_alpine",
    "thermal_upper_montane",
    "thermal_lower_montane",
    "thermal_frost_free",
)
BELT_TITLES = {
    "unknown": "нет данных",
    "thermal_nival": "нивальный (тепла почти нет)",
    "thermal_subnival": "субнивальный (короткое тёплое окно)",
    "thermal_upper_alpine": "верхний альпийский (холодно для деревьев)",
    "thermal_lower_alpine": "нижний альпийский (выше термической границы деревьев)",
    "thermal_upper_montane": "верхний горный (у термической границы деревьев)",
    "thermal_lower_montane": "нижний горный (тепло для деревьев)",
    "thermal_frost_free": "без заметных холодов (по оценке)",
}
#: colours for maps, cold to warm
BELT_RGB = np.array(
    [
        [150, 150, 150],
        [245, 248, 255],
        [200, 215, 235],
        [175, 190, 160],
        [205, 200, 130],
        [120, 165, 95],
        [70, 135, 70],
        [200, 140, 70],
    ],
    dtype=np.uint8,
)


def _daily_operator(iterations: int = 4) -> np.ndarray:
    """The (365, 12) linear operator of :func:`daily_from_monthly` (built once)."""
    eye = np.eye(12)
    days = np.arange(DAYS) + 0.5
    x = np.concatenate([_MID - DAYS, _MID, _MID + DAYS])
    daily = np.stack([np.interp(days, x, np.concatenate([e, e, e])) for e in eye.T], axis=1)
    kernel = np.array([1.0, 2.0, 3.0, 2.0, 1.0]) / 9.0

    def month_means(a):
        return np.stack([a[_MONTH_OF_DAY == m].mean(axis=0) for m in range(12)])

    for _ in range(iterations):
        daily = daily + (eye - month_means(daily))[_MONTH_OF_DAY]
        padded = np.concatenate([daily[-2:], daily, daily[:2]], axis=0)
        daily = sum(kernel[k] * padded[k : k + DAYS] for k in range(5))
    return daily + (eye - month_means(daily))[_MONTH_OF_DAY]


_DAILY_OP = _daily_operator()


def daily_from_monthly(t_monthly):
    """Periodic daily curve (365, n) whose monthly means equal the input (12, n) [H].

    Periodic linear interpolation between month centres, four rounds of mean correction
    and light periodic smoothing, then an exact final correction: every monthly mean is
    kept (to float rounding). The steps are linear, so they are one fixed matrix.
    """
    t = np.asarray(t_monthly, dtype=np.float64)
    return (_DAILY_OP @ t.reshape(12, -1)).reshape((DAYS,) + t.shape[1:])


def longest_cyclic_run(mask):
    """Longest run of True along axis 0 of (365, n), wrapping over the year end."""
    m = np.asarray(mask, dtype=bool)
    n_days = m.shape[0]
    all_true = m.all(axis=0)
    doubled = np.concatenate([m, m], axis=0).astype(np.int32)
    run = np.zeros(doubled.shape[1:], dtype=np.int32)
    best = np.zeros_like(run)
    for d in range(doubled.shape[0]):
        run = (run + 1) * doubled[d]
        best = np.maximum(best, run)
    return np.where(all_true, n_days, np.minimum(best, n_days))


def season_metrics(daily_t, t_start=0.9, w_extra=None):
    """GSL_T (days), GST_T (degC, NaN when GSL = 0), longest continuous run, degree days."""
    d = np.asarray(daily_t, dtype=np.float64)
    warm = d > t_start
    gsl = warm.sum(axis=0)
    with np.errstate(invalid="ignore", divide="ignore"):
        gst = np.where(gsl > 0, (d * warm).sum(axis=0) / np.maximum(gsl, 1), np.nan)
    out = {
        "gsl_days": gsl.astype(np.float64),
        "gst_c": gst,
        "longest_season_days": longest_cyclic_run(warm).astype(np.float64),
        "gdd0": np.maximum(d, 0.0).sum(axis=0),
        "gdd5": np.maximum(d - 5.0, 0.0).sum(axis=0),
        "gdd10": np.maximum(d - 10.0, 0.0).sum(axis=0),
        "fdd": np.maximum(-d, 0.0).sum(axis=0),
        "tdd": np.maximum(d, 0.0).sum(axis=0),
        "min_daily_c": d.min(axis=0),
    }
    if w_extra is not None:
        w = np.clip(np.asarray(w_extra, dtype=np.float64), 0.0, 1.0) * warm
        site = w.sum(axis=0)
        with np.errstate(invalid="ignore", divide="ignore"):
            out["gsl_site_days"] = site
            out["gst_site_c"] = np.where(
                site > 0, (d * w).sum(axis=0) / np.maximum(site, 1e-9), np.nan
            )
    return out


def thermal_margin(gsl, gst, gsl_min=94.0, gst_min=6.4, w_days=25.0, w_c=1.5):
    """``min((GSL - 94) / 25, (GST - 6.4) / 1.5)``; -inf-like (-40) where there is no season."""
    gsl = np.asarray(gsl, dtype=np.float64)
    gst = np.asarray(gst, dtype=np.float64)
    m = np.minimum((gsl - gsl_min) / w_days, (np.nan_to_num(gst, nan=-100.0) - gst_min) / w_c)
    return np.where(gsl > 0, m, -40.0)


def thermal_score(margin):
    return 1.0 / (1.0 + np.exp(-np.clip(np.asarray(margin, dtype=np.float64), -40.0, 40.0)))


def thermal_belt(gsl, gst, margin, frost_free, valid=None):
    """Belt index into :data:`BELTS` in the TZ order of checks (a thermal class, not a biome)."""
    gsl = np.asarray(gsl, dtype=np.float64)
    gst = np.nan_to_num(np.asarray(gst, dtype=np.float64), nan=-100.0)
    hard = (gsl >= 94.0) & (gst >= 6.4)
    out = np.full(gsl.shape, BELTS.index("thermal_lower_alpine"), dtype=np.uint8)
    out[~hard & ((gsl < 65.0) | (gst < 4.0))] = BELTS.index("thermal_upper_alpine")
    out[hard] = BELTS.index("thermal_lower_montane")
    out[hard & (np.asarray(margin) < 1.0)] = BELTS.index("thermal_upper_montane")
    out[hard & np.asarray(frost_free, dtype=bool)] = BELTS.index("thermal_frost_free")
    out[gsl < 35.0] = BELTS.index("thermal_subnival")
    out[gsl < 10.0] = BELTS.index("thermal_nival")
    if valid is not None:
        out[~np.asarray(valid, dtype=bool)] = 0
    return out


def crossings(z, value, level=0.0):
    """All heights where ``value(z)`` crosses ``level`` along a profile (sorted by z).

    Linear interpolation between neighbouring samples; several crossings (inversions,
    patchy snow) are all returned - never forced into one horizontal line.
    """
    order = np.argsort(np.asarray(z, dtype=np.float64), kind="stable")
    zz = np.asarray(z, dtype=np.float64)[order]
    vv = np.asarray(value, dtype=np.float64)[order] - level
    out = []
    for a, b, va, vb in zip(zz[:-1], zz[1:], vv[:-1], vv[1:], strict=True):
        if va == 0.0:
            out.append(float(a))
        elif va * vb < 0.0:
            out.append(float(a + (b - a) * va / (va - vb)))
    if vv.size and vv[-1] == 0.0:
        out.append(float(zz[-1]))
    return out


def zero_degree_level(t_ref_c, z_ref_m, lapse_k_km=6.5):
    """Height of 0 degC in a constant-lapse profile: ``z_ref + 1000 T_ref / Gamma``."""
    return np.asarray(z_ref_m, dtype=np.float64) + 1000.0 * np.asarray(t_ref_c) / lapse_k_km
