"""``weather_year``: one synthetic year drawn from the climate normals (TZ 5.4 sec. 10).

A sample, not a forecast and not the norm: ``P_actual`` may differ from ``P_normal``
(``unconditioned``); ``conditioned`` nudges each month toward the norm with the factor
reported. The normals and the geology never depend on the weather seed (T08, N13).

Daily generator on the atmosphere grid (regional weather states, never independent
hexes, T63)::

    wet/dry: two-state Markov chain, p_DW = pi (1 - r), p_WW = p_DW + r,  r = 0.35
             (stationary wet share pi = p_DW / (1 - p_WW + p_DW), N07)
    pi_m    = 0.9 (1 - exp(-P_bg,m / 120 mm))     wet-day share of the month
    amount  ~ Gamma(k = 0.75, theta = mean_wet / k) on wet days
    the uniforms come from a daily Gaussian field, AR(1) in time (rho 0.6) and smoothed
    over ~600 km in space -> neighbouring cells share the same weather

Events (sparse, metric geometry in world km, own ids, one RNG stream each):
tropical cyclones (Poisson of the month's Lambda, steered tracks), extratropical cyclones
with a cold and a warm front, atmospheric rivers (IVT features of a transient day),
droughts (3-month deficit against the cell's own norm; no drought in a hyperarid cell,
T67). Event rain is not added on top of the norm: the generator draws the background
rain (norm minus the expected storm and frontal shares) and the events add what they
realise, so the month keeps a closed ledger (T50, T60).

Interannual mode: with a planetary domain and a tropical ocean basin with equatorial
upwelling, ``X_y = rho X_(y-1) + sqrt(1 - rho^2) eps_y`` scales a rain response computed
by re-solving the vapour budget with +1 K on that upwelling tongue (thermodynamic
response, circulation fixed). Otherwise no ocean mode (``not_available``), only a
regional persistence proxy of wet / dry years (T69).
"""

from __future__ import annotations

import math

import numpy as np
from scipy import ndimage
from scipy.special import ndtr
from scipy.stats import gamma as gamma_dist

from . import storms
from .monsoon import detect_onset_daily

PERSISTENCE = 0.35
GAMMA_K = 0.75
DAILY_RHO = 0.6
INTERANNUAL_RHO = 0.6
PERSIST_RHO_YEARS = 0.4
FRONT_LENGTH_KM = (1500.0, 800.0)  # cold, warm
#: ETC genesis per m^2 per s per unit activity [H]: ~2500 cyclones / yr on an Earth-sized
#: planet with ~5e13 m^2 of storm track at activity ~0.7
ETC_RATE = 2.3e-18
#: safety cap of explicit events of one type in one month (the rest stays climatology)
MAX_EVENTS_MONTH = 150


def stationary_wet_fraction(p_ww, p_dw):
    return p_dw / (1.0 - p_ww + p_dw)


def markov_params(pi, r=PERSISTENCE):
    pi = np.clip(pi, 0.0, 0.95)
    r = np.minimum(r, 1.0 - pi)  # keep p_WW <= 1
    p_dw = pi * (1.0 - r)
    return p_dw + r, p_dw


def wet_share(p_month_mm):
    return 0.9 * (1.0 - np.exp(-np.maximum(p_month_mm, 0.0) / 120.0))


def correlated_uniforms(rng, shape, days, cell_km, corr_km=600.0, rho=DAILY_RHO):
    """(days, *shape) uniforms with spatial (Gaussian) and temporal (AR1) correlation."""
    sig = max(0.5, corr_km / cell_km / 2.0)
    out = np.empty((days,) + shape)
    z = np.zeros(shape)
    for d in range(days):
        e = rng.s.normal(int(np.prod(shape))).reshape(shape)
        e = ndimage.gaussian_filter(e, sig, mode="wrap")
        e /= max(float(e.std()), 1e-9)
        z = rho * z + math.sqrt(1.0 - rho * rho) * e if d else e
        out[d] = ndtr(z)
    return out


def daily_series(p_bg_mm, days_in_month, u_wet, u_amt):
    """Daily rain for one month on cells: Markov wet/dry, Gamma amounts."""
    nd = len(u_wet)
    pi = wet_share(p_bg_mm)
    p_ww, p_dw = markov_params(pi)
    mean_wet = p_bg_mm / np.maximum(pi * days_in_month, 1e-9)
    theta = mean_wet / GAMMA_K
    rain = np.zeros((nd,) + np.shape(p_bg_mm))
    wet = u_wet[0] < pi
    for d in range(nd):
        if d:
            wet = np.where(wet, u_wet[d] < p_ww, u_wet[d] < p_dw)
        amt = gamma_dist.ppf(
            np.clip(u_amt[d], 1e-9, 1 - 1e-9), GAMMA_K, scale=np.maximum(theta, 1e-12)
        )
        rain[d] = np.where(wet & (p_bg_mm > 0), amt, 0.0)
    return rain


def _poly_km(points):
    return [[round(float(x), 3), round(float(y), 3)] for x, y in points]


def tc_events(G, grid, planet, months_lambda, gen, rng_factory, year_tag):
    out = []
    lat = G["lat_deg"]
    area = (grid["cell_km"] * 1000.0) ** 2
    for m in range(12):
        rng = rng_factory(f"tc/{m + 1}")
        n = min(rng.poisson(float(months_lambda[m])), MAX_EVENTS_MONTH)
        if n == 0:
            continue
        p = (gen[m] * area).ravel()
        if p.sum() <= 0:
            continue
        picks = rng.choice(p, n)
        for k, pick in enumerate(picks):
            j, i = divmod(int(pick), lat.shape[1])
            x = grid["x0_km"] + (i + rng.uniform(-0.5, 0.5)) * grid["cell_km"]
            y = grid["y0_km"] + (j + rng.uniform(-0.5, 0.5)) * grid["cell_km"]
            tf = storms.TrackField(grid, G, m)
            eid = f"TC-{year_tag}-{m + 1:02d}-{k + 1:02d}"
            tr = storms.run_track(tf, x, y, rng, planet, event_id=eid)
            out.append(
                {
                    "event_id": eid,
                    "event_type": "TC",
                    "month": m + 1,
                    "start_day": int(rng.uniform(0, 28)),
                    "track": tr,
                    "classification_quality": "environment_weighted_stochastic_track",
                }
            )
    return out


def etc_events(G, grid, rng_factory, year_tag, days):
    """Extratropical cyclones with a trailing cold front and a warm front (polylines)."""
    out = []
    cell = grid["cell_km"]
    act = G["etc_activity"]
    area = (cell * 1000.0) ** 2
    for m in range(12):
        rng = rng_factory(f"etc/{m + 1}")
        lam = float((act[m] * area * ETC_RATE * days[m] * 86400.0).sum())
        n = min(rng.poisson(lam), MAX_EVENTS_MONTH)
        if n == 0:
            continue
        picks = rng.choice(act[m].ravel() + 1e-12, n)
        u5 = G["wind_u_ms"][m, 2]
        v5 = G["wind_v_ms"][m, 2]
        for k, pick in enumerate(picks):
            j, i = divmod(int(pick), act.shape[2])
            x = grid["x0_km"] + i * cell
            y = grid["y0_km"] + j * cell
            tf = storms.TrackField(grid, G, m)
            lat0 = tf.at(G["lat_deg"], x, y)
            hemi = 1.0 if lat0 >= 0 else -1.0
            pts = []
            life_h = rng.uniform(72.0, 120.0)
            depth = 0.0
            for t in np.arange(0.0, life_h + 1e-9, 6.0):
                if not tf.inside(x, y):
                    pts.append({"t_h": float(t), "x_km": x, "y_km": y, "status": "left_domain"})
                    break
                sig = tf.at(G["etc_eady_growth_per_day"][m], x, y)
                depth += (sig * 0.25 - 0.08) * 6.0 * (1.0 if t < life_h * 0.6 else -1.5)
                depth = max(depth, 0.0)
                pts.append(
                    {
                        "t_h": float(t),
                        "x_km": x,
                        "y_km": y,
                        "pressure_deficit_hpa": round(4.0 * depth, 2),
                    }
                )
                ue = 0.7 * tf.at(u5, x, y)
                vn = 0.7 * tf.at(v5, x, y) + 1.5 * hemi  # cyclones drift poleward
                x += ue * 6 * 3.6
                y -= vn * 6 * 3.6
            cx, cy = pts[len(pts) // 2]["x_km"], pts[len(pts) // 2]["y_km"]
            t_air = tf.at(G["t_air_c"][m], cx, cy)
            # fronts at mid-life: cold front trails equatorward-westward, warm front eastward
            eq = 1.0 if hemi > 0 else -1.0  # map y grows southward = toward the equator (NH)
            cold = [
                (cx + s * -0.35 * FRONT_LENGTH_KM[0], cy + s * eq * 0.95 * FRONT_LENGTH_KM[0])
                for s in np.linspace(0, 1, 8)
            ]
            warm = [
                (cx + s * FRONT_LENGTH_KM[1], cy + s * eq * 0.25 * FRONT_LENGTH_KM[1])
                for s in np.linspace(0, 1, 6)
            ]
            t850 = t_air - 9.75  # 1.5 km at 6.5 K/km
            snow = float(np.clip((1.0 - t850) / 4.0 - 0.5, 0.0, 1.0)) if t_air < 3.0 else 0.0
            eid = f"ETC-{year_tag}-{m + 1:02d}-{k + 1:02d}"
            out.append(
                {
                    "event_id": eid,
                    "event_type": "ETC",
                    "month": m + 1,
                    "start_day": int(rng.uniform(0, 28)),
                    "track": {"event_id": eid, "points": pts},
                    "fronts": [
                        {
                            "front_type": "cold",
                            "geometry_world_km": _poly_km(cold),
                            "width_m": 150e3,
                            "delta_t_k": 6.0,
                            "wind_shift_deg": 90.0,
                            "snow_fraction": snow,
                        },
                        {
                            "front_type": "warm",
                            "geometry_world_km": _poly_km(warm),
                            "width_m": 300e3,
                            "delta_t_k": 4.0,
                            "wind_shift_deg": 45.0,
                            "snow_fraction": snow,
                        },
                    ],
                    "classification_quality": "baroclinicity_front_proxy",
                }
            )
    return out


def _footprint(shape, grid, x_km, y_km, radius_km):
    ny, nx = shape
    c = grid["cell_km"]
    fy = (y_km - grid["y0_km"]) / c
    fx = (x_km - grid["x0_km"]) / c
    r = int(math.ceil(2 * radius_km / c))
    j0, i0 = int(round(fy)), int(round(fx))
    jl, jh, il, ih = max(0, j0 - r), min(ny, j0 + r + 1), max(0, i0 - r), min(nx, i0 + r + 1)
    if jl >= jh or il >= ih:
        return None
    jj, ii = np.mgrid[jl:jh, il:ih]
    w = np.exp(-((np.hypot((jj - fy) * c, (ii - fx) * c) / radius_km) ** 2))
    return (slice(jl, jh), slice(il, ih)), w


def event_rain(ev, G, grid, shape, pool=None):
    """Rain field (mm) of one event, limited by the vapour it can draw (ledger).

    ``pool`` (ny, nx) is the month's vapour still available to events (mm); the event
    takes its rain out of it, so overlapping events never spend the same water (T60).
    """
    rain = np.zeros(shape)
    w_col = G["column_vapor_kg_m2"][ev["month"] - 1]
    land_h = G["land_height_m"]
    supply = 0.0
    want = 0.0
    if ev["event_type"] == "TC":
        for p in ev["track"]["points"]:
            v = p.get("v_ms", 0.0)
            if v < 12.0:
                continue
            fp = _footprint(shape, grid, p["x_km"], p["y_km"], 200.0)
            if fp is None:
                continue
            sl, w = fp
            rate = 1.2 * (0.5 + v / 30.0)  # mm/h near the core [A]
            oro = 1.0 + np.clip(land_h[sl] / 1500.0, 0.0, 1.5)  # windward hills wring it out
            r6 = rate * 6.0 * w * oro
            want += float(r6.sum())
            # vapour drawn: the column over the footprint, renewed by convergence each step
            supply += float((w * w_col[sl]).sum()) * 1.5
            rain[sl] += r6
    elif ev["event_type"] == "ETC":
        for fr in ev["fronts"]:
            for x, y in fr["geometry_world_km"]:
                fp = _footprint(shape, grid, x, y, fr["width_m"] / 1000.0)
                if fp is None:
                    continue
                sl, w = fp
                r = 6.0 * w * np.clip(G["column_rh"][ev["month"] - 1][sl] - 0.3, 0.0, 1.0) / 0.7
                want += float(r.sum())
                supply += float((w * w_col[sl]).sum()) * 0.8
                rain[sl] += r
    elif ev["event_type"] == "AR":
        for x, y in ev["axis_world_km"]:
            fp = _footprint(shape, grid, x, y, ev["width_km"] / 2.0)
            if fp is None:
                continue
            sl, w = fp
            oro = np.clip(land_h[sl] / 1000.0, 0.0, 2.0) * (G["sea_fraction"][sl] < 0.5)
            r = 15.0 * w * oro  # AR rain falls where it is lifted (T58)
            want += float(r.sum())
            supply += float((w * w_col[sl]).sum())
            rain[sl] += r
    scale = min(1.0, supply / want) if want > 0 else 1.0
    rain *= scale
    if pool is not None:
        taken = np.minimum(rain, pool)
        pool -= taken
        scale = (
            scale * (float(taken.sum()) / max(float(rain.sum()), 1e-12)) if rain.sum() else scale
        )
        rain = taken
    ev["budget_ledger"] = {
        "requested_mm_cells": round(want, 3),
        "vapour_available_mm_cells": round(supply, 3),
        "rain_mm_cells": round(float(rain.sum()), 3),
        "limited_by_vapour": scale < 1.0,
        "residual_mm": 0.0,
    }
    return rain


def ar_events(G, grid, rng_factory, year_tag):
    out = []
    shape = G["lat_deg"].shape
    for m in range(12):
        rng = rng_factory(f"ar/{m + 1}")
        e = G["ivt_east_kg_m_s"][m].astype(np.float64)
        n = G["ivt_north_kg_m_s"][m].astype(np.float64)
        anom = rng.s.normal(int(np.prod(shape))).reshape(shape)
        anom = ndimage.gaussian_filter(anom, (0.6, 2.5), mode="nearest")
        anom /= max(float(anom.std()), 1e-9)
        amp = 0.8 * np.hypot(e, n) + 80.0 * G["etc_activity"][m]
        mag = np.hypot(e, n) + np.maximum(anom, 0.0) * amp
        k = mag / np.maximum(np.hypot(e, n), 1e-9)
        for q, f in enumerate(storms.detect_ar(e * k, n * k, grid["cell_km"])):
            (ax0, ay0), (ax1, ay1) = f["axis_km"]
            pts = [
                (grid["x0_km"] + ax0 + s * (ax1 - ax0), grid["y0_km"] + ay0 + s * (ay1 - ay0))
                for s in np.linspace(0, 1, 6)
            ]
            eid = f"AR-{year_tag}-{m + 1:02d}-{q + 1:02d}"
            out.append(
                {
                    "event_id": eid,
                    "event_type": "AR",
                    "month": m + 1,
                    "start_day": int(rng.uniform(0, 28)),
                    "axis_world_km": _poly_km(pts),
                    "length_km": f["length_km"],
                    "width_km": f["width_km"],
                    "mean_ivt_kg_m_s": f["mean_ivt_kg_m_s"],
                    "classification_quality": "ivt_connected_feature",
                }
            )
    return out


def drought_mask(p_actual12, p_norm12, min_norm_mm=10.0, deficit=0.4):
    """Cells with a 3-month rain deficit > ``deficit`` of their own norm (T66, T67)."""
    a3 = sum(np.roll(p_actual12, -k, axis=0) for k in range(3))
    n3 = sum(np.roll(p_norm12, -k, axis=0) for k in range(3))
    rel = 1.0 - a3 / np.maximum(n3, 1e-9)
    valid = n3 >= 3 * min_norm_mm  # a hyperarid cell has no drought of its own
    return ((rel > deficit) & valid).any(axis=0), np.where(valid, rel.max(axis=0), 0.0)


def persistent_drought(year_masks):
    """Cells in drought in every one of at least two consecutive years."""
    if len(year_masks) < 2:
        return np.zeros_like(year_masks[0], dtype=bool)
    out = np.zeros_like(year_masks[0], dtype=bool)
    for a, b in zip(year_masks[:-1], year_masks[1:], strict=True):
        out |= a & b
    return out


def interannual_sequence(rng_factory, years, rho=INTERANNUAL_RHO):
    x, out = 0.0, []
    for y in range(1, years + 1):
        e = rng_factory(f"interannual/{y}").standard_normal()
        x = rho * x + math.sqrt(1.0 - rho * rho) * e if y > 1 else e
        out.append(x)
    return out


def persistence_field(rng_factory, shape, years, cell_km, rho=PERSIST_RHO_YEARS):
    z = np.zeros(shape)
    for y in range(1, years + 1):
        e = rng_factory(f"persistence/{y}").s.normal(int(np.prod(shape))).reshape(shape)
        e = ndimage.gaussian_filter(e, max(0.5, 900.0 / cell_km), mode="nearest")
        e /= max(float(e.std()), 1e-9)
        z = rho * z + math.sqrt(1 - rho * rho) * e if y > 1 else e
    return z


def generate(
    G,
    grid,
    planet,
    days,
    seed: str,
    year_seed: str,
    year: int,
    storm_lambda,
    mode="unconditioned",
    ocean_mode=None,
    with_daily_cells=None,
) -> dict:
    """One ``WeatherYearSnapshot`` (dict) on the atmosphere grid."""
    from .storms import EventRng

    tag = f"{year_seed}-{year}"

    def rng_factory(name):
        return EventRng(seed, f"atmosphere/5.4/weather/{year_seed}/{year}/{name}")

    shape = G["lat_deg"].shape
    cell = grid["cell_km"]
    comps = G["precip_components_mm"].astype(np.float64)
    p_norm = G["precipitation_mm"].astype(np.float64)
    # background = the norm without the storm / frontal shares the events will realise
    p_bg = np.maximum(p_norm - comps[3] - 0.5 * comps[2], 0.0)
    # interannual: an ocean mode where one exists, else a regional persistence proxy
    xs = interannual_sequence(rng_factory, year)
    if ocean_mode is not None:
        factor = np.maximum(1.0 + xs[-1] * ocean_mode["response"], 0.2)
        inter = {
            "mode": "ocean_basin_mode",
            "x": xs[-1],
            "rho": INTERANNUAL_RHO,
            "pattern": ocean_mode["pattern_note"],
        }
    else:
        z = persistence_field(rng_factory, shape, year, cell)
        factor = np.exp(0.15 * z)[None] * np.ones((12,) + shape)
        inter = {
            "mode": "regional_persistence_proxy",
            "ocean_mode": "not_available",
            "rho_years": PERSIST_RHO_YEARS,
        }
    p_bg = p_bg * factor
    u_all = correlated_uniforms(rng_factory("daily/wet"), shape, int(round(sum(days))), cell)
    a_all = correlated_uniforms(
        rng_factory("daily/amount"), shape, int(round(sum(days))), cell, corr_km=300.0, rho=0.3
    )
    start = np.concatenate([[0], np.cumsum(np.round(days).astype(int))])
    daily = np.zeros((int(start[-1]),) + shape)
    for m in range(12):
        d0, d1 = int(start[m]), int(start[m + 1])
        daily[d0:d1] = daily_series(p_bg[m], d1 - d0, u_all[d0:d1], a_all[d0:d1])
    gen_m = np.stack([daily[int(start[m]) : int(start[m + 1])].sum(axis=0) for m in range(12)])
    if mode == "conditioned":
        nudge = np.where(gen_m > 0, p_bg / np.maximum(gen_m, 1e-9), 1.0)
        nudge = np.clip(nudge, 0.5, 2.0)
        gen_m = gen_m * nudge
        for m in range(12):
            daily[int(start[m]) : int(start[m + 1])] *= nudge[m]
    events = []
    gen_idx = G.get("tc_genesis_index")
    if gen_idx is not None and storm_lambda is not None:
        events += tc_events(G, grid, planet, storm_lambda, gen_idx, rng_factory, tag)
    if "etc_activity" in G:
        events += etc_events(G, grid, rng_factory, tag, days)
        events += ar_events(G, grid, rng_factory, tag)
    ev_m = np.zeros((12,) + shape)
    # snow share of the month's rain from the surface air temperature [A]; ETC fronts use
    # their own profile-based share: snow is a part of the same water, never added (T74)
    snow_share = np.clip((1.0 - G["t_air_c"].astype(np.float64)) / 4.0, 0.0, 1.0)
    ev_snow = np.zeros((12,) + shape)
    # the vapour events can spend in a month: the column renewed every ~8 days
    pool = G["column_vapor_kg_m2"].astype(np.float64) * (np.asarray(days).reshape(12, 1, 1) / 8.0)
    events.sort(key=lambda e: e["event_id"])  # deterministic, independent of generation order
    for ev in events:
        r = event_rain(ev, G, grid, shape, pool[ev["month"] - 1])
        m = ev["month"] - 1
        ev_m[m] += r
        if ev["event_type"] == "ETC":
            ev_snow[m] += r * ev["fronts"][0]["snow_fraction"]
        else:
            ev_snow[m] += r * snow_share[m]
        d = int(start[m]) + min(int(ev.get("start_day", 0)), int(days[m]) - 1)
        daily[d] += r  # the event day carries its rain (dynamic hydrology input)
    p_act = gen_m + ev_m
    # the flood day: the day with the largest rain excess over the land (one real day)
    land = G["sea_fraction"] < 0.5
    excess_day = (np.maximum(daily - 30.0, 0.0) * land[None]).sum(axis=(1, 2))
    flood_day = int(np.argmax(excess_day))
    dmask, ddef = drought_mask(p_act, p_norm)
    # onset of the rainy season in the daily series (persistent rain, not one storm)
    onset = np.full(shape, -1, dtype=np.int16)
    wet3 = (p_norm + np.roll(p_norm, -1, 0) + np.roll(p_norm, -2, 0)).argmax(axis=0)
    seasonal = (p_norm.max(axis=0) / np.maximum(p_norm.mean(axis=0), 1e-9)) > 2.0
    for j, i in zip(*np.nonzero(seasonal), strict=True):
        d_start = max(int(start[(wet3[j, i] - 2) % 12]), 0)
        series = np.roll(daily[:, j, i], -d_start)
        o = detect_onset_daily(series)
        onset[j, i] = -1 if o < 0 else (o + d_start) % daily.shape[0]
    snapshot = {
        "climate_id": None,
        "year_seed": year_seed,
        "year": year,
        "mode": mode,
        "events": events,
        "aggregate_monthly_actuals_mm": p_act.astype(np.float32),
        "generator_monthly_mm": gen_m.astype(np.float32),
        "event_monthly_mm": ev_m.astype(np.float32),
        "snowfall_monthly_mm": (gen_m * snow_share + ev_snow).astype(np.float32),
        "annual_anomaly_pct": (
            100.0 * (p_act.sum(0) / np.maximum(p_norm.sum(0), 1e-9) - 1.0)
        ).astype(np.float32),
        "max_daily_mm": daily.max(axis=0).astype(np.float32),
        "flood_day": flood_day,
        "flood_day_mm": daily[flood_day].astype(np.float32),
        "drought": dmask,
        "drought_deficit": ddef.astype(np.float32),
        "onset_day": onset,
        "interannual": inter,
        "ledger": {
            "generator_mm_cells": float(gen_m.sum()),
            "event_mm_cells": float(ev_m.sum()),
            "norm_mm_cells": float(p_norm.sum()),
            "background_target_mm_cells": float(p_bg.sum()),
            "note": "events realise the storm / frontal shares removed from the background",
        },
        "quality": "synthetic_weather_year_sample (not a forecast)",
    }
    if with_daily_cells is not None:
        snapshot["daily_sample"] = daily[:, with_daily_cells[0], with_daily_cells[1]]
    return snapshot
