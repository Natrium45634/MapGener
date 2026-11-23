"""HexWorld v5.4: the atmosphere on the hexes (stage ``v54_atmosphere``, after v5.3).

The monthly normals (circulation, moisture transport, ITCZ) are solved with the climate on
the coarse atmosphere grid inside the v4 climate stage, so the rivers, soils and biomes
already use their rain. This stage keeps the normals (``grid_fields``), derives the
monsoon / rainy-season diagnostics on the hexes from the v4.5 monthly rain and PET, adds
the storm climatology (v5.4b) and, when asked, one synthetic weather year (v5.4c).
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import asdict

import numpy as np

from .atmosphere import monsoon as mon
from .atmosphere import settings as atmosphere_settings
from .atmosphere.circulation import LEVELS_PA, Circulation, month_days, wind_at
from .atmosphere.normals import ALGORITHM_VERSION
from .project import AtmosphereData

COMPONENT_LAYERS = (
    "precip_share_convective",
    "precip_share_orographic",
    "precip_share_frontal",
    "precip_share_storm",
    "precip_share_background",
)


def _key(*parts) -> str:
    return hashlib.sha256(json.dumps(parts, sort_keys=True, default=str).encode()).hexdigest()


def atmosphere_stage_keys(config, project) -> dict:
    earth = getattr(project, "earth", None)
    return {
        "v54_atmosphere": _key(
            "v54",
            ALGORITHM_VERSION,
            asdict(config.atmosphere),
            earth.stage_keys if earth is not None else {},
        )
    }


def world_with_normals(config, project):
    entry = project.cache.get("v4_climate")
    wc = entry.get("world") if isinstance(entry, dict) else None
    if wc is not None and getattr(wc, "atmosphere", None) is not None:
        return wc
    from .stages.climate import resolve_latitude, world_climate

    hist = project.hist
    latitude = resolve_latitude(config.climate, project.geometry.simulation_bounds)
    wc = world_climate(
        config.climate,
        hist.world["grid"],
        hist.world["elevation_m"],
        hist.sea_level_m,
        latitude,
        earth=config.earth,
        atmosphere=atmosphere_settings(config),
    )
    if isinstance(entry, dict):
        entry["world"] = wc
    return wc


def grid_fields_of(normals) -> dict:
    c, m = normals.circ, normals.moist
    sh = normals.shape
    f32 = np.float32
    r12 = lambda a: np.asarray(a).reshape((12,) + sh).astype(f32)  # noqa: E731
    return {
        "lat_deg": normals.lat.astype(f32),
        "sea_fraction": normals.sea_frac.astype(f32),
        "land_height_m": normals.land_h.astype(f32),
        "t_sea_level_c": c.t_sl.astype(f32),
        "t_air_c": c.t_air.astype(f32),
        "slp_pa": c.slp_pa.astype(f32),
        "pressure_sfc_pa": c.p_sfc_pa.astype(f32),
        "wind_u_ms": c.u.astype(f32),
        "wind_v_ms": c.v.astype(f32),
        "level_height_m": c.z.astype(f32),
        "thermal_equator_deg": c.thermal_equator_deg.astype(f32),
        "column_vapor_kg_m2": r12(m.w),
        "ivt_east_kg_m_s": r12(m.ivt_e),
        "ivt_north_kg_m_s": r12(m.ivt_n),
        "moisture_convergence_kg_m2_s": r12(m.conv),
        "evaporation_mm": r12(m.evap_mm),
        "precipitation_mm": r12(m.precip_mm),
        "precip_components_mm": m.components_mm.reshape((5, 12) + sh).astype(f32),
        "column_rh": r12(m.rh),
        "itcz_score": normals.itcz_score.astype(f32),
    }


def circulation_of(atm: AtmosphereData) -> Circulation:
    G = atm.grid_fields
    return Circulation(
        np.asarray(atm.grid["month_days"], dtype=np.float64),
        G["t_sea_level_c"],
        G["t_air_c"],
        G["slp_pa"],
        G["pressure_sfc_pa"],
        (None,) + LEVELS_PA,
        G["wind_u_ms"],
        G["wind_v_ms"],
        G["level_height_m"],
        G["thermal_equator_deg"],
        float(atm.grid["hadley_edge_deg"]),
    )


def sample(atm_or_grid, field_, x_km, y_km):
    """Bilinear sample of an atmosphere-grid field (..., ny, nx) at world coordinates."""
    from scipy import ndimage

    g = atm_or_grid.grid if hasattr(atm_or_grid, "grid") else atm_or_grid
    fx = (np.asarray(x_km, dtype=np.float64) - g["x0_km"]) / g["cell_km"]
    fy = (np.asarray(y_km, dtype=np.float64) - g["y0_km"]) / g["cell_km"]
    coords = np.vstack([np.ravel(fy), np.ravel(fx)])
    f = np.asarray(field_, dtype=np.float64)
    if f.ndim == 2:
        return ndimage.map_coordinates(f, coords, order=1, mode="nearest").reshape(np.shape(x_km))
    flat = f.reshape((-1,) + f.shape[-2:])
    out = np.stack([ndimage.map_coordinates(a, coords, order=1, mode="nearest") for a in flat])
    return out.reshape(f.shape[:-2] + np.shape(x_km))


def wind_at_world(atm: AtmosphereData, x_km, y_km, month, pressure_pa=None, altitude_m=None):
    """``windAt``: wind (east, north, m/s) and quality codes at world points.

    Bilinear over the four surrounding atmosphere cells of the per-column profiles
    (``circulation.wind_at``); a display or tephra call never recomputes the normals.
    """
    circ = circulation_of(atm)
    g = atm.grid
    ny, nx = np.asarray(atm.grid_fields["lat_deg"]).shape
    fx = np.clip((np.ravel(x_km) - g["x0_km"]) / g["cell_km"], 0, nx - 1)
    fy = np.clip((np.ravel(y_km) - g["y0_km"]) / g["cell_km"], 0, ny - 1)
    i0 = np.minimum(np.floor(fx).astype(int), nx - 2) if nx > 1 else np.zeros(fx.size, int)
    j0 = np.minimum(np.floor(fy).astype(int), ny - 2) if ny > 1 else np.zeros(fy.size, int)
    wx, wy = fx - i0, fy - j0
    u = np.zeros(fx.size)
    v = np.zeros(fx.size)
    q = np.zeros(fx.size, dtype=np.int8)
    for dj, di, w in (
        (0, 0, (1 - wx) * (1 - wy)),
        (0, 1, wx * (1 - wy)),
        (1, 0, (1 - wx) * wy),
        (1, 1, wx * wy),
    ):
        jj = np.minimum(j0 + dj, ny - 1)
        ii = np.minimum(i0 + di, nx - 1)
        cu, cv, cq = wind_at(
            circ, month, pressure_pa=pressure_pa, altitude_m=altitude_m, cell=jj * nx + ii
        )
        u += w * cu
        v += w * cv
        q = np.maximum(q, cq)
    return u.reshape(np.shape(x_km)), v.reshape(np.shape(x_km)), q.reshape(np.shape(x_km))


def grid_diagnostics(G: dict, cell_km: float, days) -> dict:
    """Wind / SST based monsoon ingredients on the atmosphere grid."""
    p = G["precipitation_mm"].astype(np.float64)
    sea_frac = G["sea_fraction"].astype(np.float64)
    u = G["wind_u_ms"][:, 0].astype(np.float64)
    v = G["wind_v_ms"][:, 0].astype(np.float64)
    uq = 0.6 * u + 0.4 * G["wind_u_ms"][:, 1]
    vq = 0.6 * v + 0.4 * G["wind_v_ms"][:, 1]
    wet0 = mon.wettest_3month_start(p)
    s3 = p + np.roll(p, -1, axis=0) + np.roll(p, -2, axis=0)
    dry0 = s3.argmin(axis=0)

    def mean3(arr, start):
        idx = (start[None] + np.arange(3).reshape(3, 1, 1)) % 12
        return np.take_along_axis(arr, idx, axis=0).mean(axis=0)

    rev = mon.reversal_score(mean3(u, wet0), mean3(v, wet0), mean3(u, dry0), mean3(v, dry0))
    ne, nn = mon.landward_normal(sea_frac, cell_km)
    onshore = np.maximum(G["ivt_east_kg_m_s"] * ne + G["ivt_north_kg_m_s"] * nn, 0.0)
    onshore = np.where(sea_frac[None] > 0.5, 0.0, onshore)
    sst = G["t_sea_level_c"].astype(np.float64)
    dtls = mon.land_sea_contrast(G["t_air_c"].astype(np.float64), sst, sea_frac, uq, vq, cell_km)
    w_p = p / np.maximum(p.sum(axis=0), 1e-9)
    u_ann = (w_p * uq).sum(axis=0)
    v_ann = (w_p * vq).sum(axis=0)
    return {
        "wind_reversal_score": rev,
        "onshore_change": mean3(onshore, wet0) - mean3(onshore, dry0),
        "onshore_seasonality": (onshore.max(axis=0) - onshore.min(axis=0))
        / (onshore.max(axis=0) + 1e-6),
        "contrast_change": mean3(dtls, wet0) - mean3(dtls, dry0),
        "contrast_wet": mean3(dtls, wet0),
        "itcz_overlap_months": (G["itcz_score"] > 0.5).sum(axis=0).astype(np.float64),
        "source_distance_km": mon.source_distance_km(sea_frac, u_ann, v_ann, cell_km),
        "onshore": onshore,
        "dtls": dtls,
    }


def run_atmosphere(config, project) -> dict:
    t0 = time.perf_counter()
    wc = world_with_normals(config, project)
    normals = getattr(wc, "atmosphere", None)
    if normals is None:
        return {"blocked": wc.balance.get("atmosphere", {"reason": "no normals"})}
    lay = project.layout
    shape = (lay.height, lay.width)
    G = grid_fields_of(normals)
    days = month_days_of(config)
    cell = float(normals.grid["cell_km"])
    gd = grid_diagnostics(G, cell, days)
    G["land_sea_contrast_k"] = gd["dtls"].astype(np.float32)
    G["onshore_ivt_kg_m_s"] = gd["onshore"].astype(np.float32)
    grid = dict(
        normals.grid,
        shape=list(normals.shape),
        factor=normals.factor,
        month_days=[float(d) for d in days],
        hadley_edge_deg=float(normals.circ.hadley_edge_deg),
    )
    storm_summary = None
    if getattr(config.atmosphere, "storms", True):
        storm_summary = _storm_climatology(config, wc, normals, G, grid, days)
    hx, hy = lay.centers()
    M = project.env.monthly
    p12 = np.nan_to_num(M["precip_monthly_mm"].astype(np.float64))
    pet12 = M["pet_monthly_mm"].astype(np.float64)
    lat = project.hydro.layers["latitude_deg"].astype(np.float64)
    p_ann = p12.sum(axis=0)
    seas = mon.seasonality(p12, days)
    w3 = mon.wettest_3month_fraction(p12)
    rng = mon.precip_range_mm_day(p12, days)
    sfrac = mon.summer_fraction(p12, lat)
    wet_m, dry_m = mon.wet_dry_months(p12, pet12)
    onset, withdrawal, onset2, n_seasons = mon.onset_withdrawal(p12, days)
    peaks = mon.peak_count(p12)
    gsample = {k: sample(grid, v, hx, hy) for k, v in gd.items() if k not in ("onshore", "dtls")}
    suit = mon.suitability(
        rng,
        sfrac,
        gsample["contrast_change"],
        gsample["onshore_change"],
        gsample["wind_reversal_score"],
        lat,
    )
    water = ~project.hydro.layers["land_mask"].astype(bool)
    suit = np.where(water, 0.0, suit)
    reg = mon.regime(p_ann, dry_m, seas, suit, peaks, sfrac, lat)
    comps = G["precip_components_mm"].astype(np.float64).sum(axis=1)  # (5, ny, nx)
    tot = np.maximum(comps.sum(axis=0), 1e-9)
    f32 = np.float32
    layers = {
        "rainfall_seasonality": seas.astype(f32),
        "wettest_3month_fraction": w3.astype(f32),
        "monsoon_precip_range_mm_day": rng.astype(f32),
        "monsoon_rainfall_fraction": sfrac.astype(f32),
        "wet_season_months": wet_m,
        "dry_season_months": dry_m,
        "monsoon_onset_month": onset,
        "monsoon_withdrawal_month": withdrawal,
        "second_rainy_season_onset_month": onset2,
        "rainy_season_count": n_seasons,
        "rain_peak_count": peaks,
        "monsoon_suitability": suit.astype(f32),
        "rain_regime": reg,
        "wind_reversal_score": gsample["wind_reversal_score"].astype(f32),
        "onshore_flux_seasonality": gsample["onshore_seasonality"].astype(f32),
        "land_sea_contrast_wet_k": gsample["contrast_wet"].astype(f32),
        "itcz_overlap_months": np.rint(gsample["itcz_overlap_months"]).astype(np.uint8),
        "moisture_source_distance_km": gsample["source_distance_km"].astype(f32),
    }
    for k, name in enumerate(COMPONENT_LAYERS):
        layers[name] = sample(grid, comps[k] / tot, hx, hy).astype(f32)
    if storm_summary is not None:
        for name in STORM_HEX_LAYERS:
            arr = G[name]
            a2 = (
                arr.max(axis=0)
                if name in ("tc_potential_intensity_ms",)
                else (arr.mean(axis=0) if arr.ndim == 3 else arr)
            )
            layers[name] = sample(grid, a2, hx, hy).astype(f32)
        ocean = project.hydro.layers["ocean_mask"].astype(bool)
        from .stages.raster import neighbor_index

        coast = np.zeros(shape, dtype=bool)
        for d in range(6):
            nb, valid = neighbor_index(lay, d)
            coast |= np.where(valid, ocean.ravel()[nb.ravel()].reshape(shape), False)
        coast &= ~ocean
        v = layers["tc_max_wind_ms"].astype(np.float64)
        layers["tc_surge_potential_index"] = np.where(
            coast, layers["tc_passage_per_yr"] * (v / 33.0) ** 2, 0.0
        ).astype(f32)
    for k, v in layers.items():
        layers[k] = np.asarray(v).reshape(shape)
    events: list = []
    wy_summary = None
    if getattr(config.atmosphere, "weather_year", False):
        wy_summary, events = _weather_year(
            config, wc, normals, G, grid, days, storm_summary, project, layers, shape
        )
    land = ~water
    summary = {
        "mode": "climate_normals",
        "algorithm_version": ALGORITHM_VERSION,
        "source_snapshot_hash": normals.source_hash,
        "seed_namespace": normals.seed_namespace,
        "quality": normals.quality,
        "grid": {k: grid[k] for k in ("factor", "cell_km", "shape")},
        "hadley_edge_deg": grid["hadley_edge_deg"],
        "itcz_segments": normals.itcz_segments,
        "world_climate": {k: v for k, v in wc.balance.get("atmosphere", {}).items()},
        "budget_months": normals.diagnostics["budget"]["months"],
        "regime_shares_land": {
            name: float(((reg == i) & land).sum() / max(land.sum(), 1))
            for i, name in enumerate(mon.REGIMES)
        },
        "monsoon_land_share": float(((suit >= 0.4) & land).sum() / max(land.sum(), 1)),
        "calendar": {"month_days": grid["month_days"]},
        "units": {
            "precipitation": "mm/month",
            "wind": "m/s (east, north)",
            "pressure": "Pa",
            "ivt": "kg/(m s)",
            "column_vapor": "kg/m^2",
        },
        "levels_pa": [None, *LEVELS_PA],
        "event_mode": "weather_year" if wy_summary else "not_generated",
        "storms": storm_summary,
        "weather_year": wy_summary,
    }
    return {
        "layers": layers,
        "grid_fields": G,
        "grid": grid,
        "summary": summary,
        "events": events,
        "timings": {"v54_atmosphere": round(time.perf_counter() - t0, 3)},
    }


STORM_HEX_LAYERS = (
    "tc_genesis_index",
    "tc_potential_intensity_ms",
    "tc_passage_per_yr",
    "tc_max_wind_ms",
    "etc_activity",
    "front_frequency_index",
    "ar_day_frequency",
)


def _storm_climatology(config, wc, normals, G, grid, days):
    """v5.4b: risk fields on the grid; TC rain moved from the convective component."""
    from .atmosphere import storms
    from .atmosphere.normals import block_mean
    from .earth.planet import Planet

    planet = Planet.from_params(config.earth)
    upw = None
    oc = getattr(getattr(wc, "earth", None), "ocean", None)
    if oc is not None and getattr(oc, "upwelling_m_day", None) is not None:
        season_of = (0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3, 0)
        u4 = np.stack(
            [block_mean(np.nan_to_num(oc.upwelling_m_day[s]), normals.factor) for s in range(4)]
        )
        upw = np.stack([np.maximum(u4[season_of[m]], 0.0) for m in range(12)])
    rng = storms.EventRng(config.seed, "atmosphere/5.4/tc_climatology")
    st = storms.climatology(G, grid, planet, days, rng, config.atmosphere.quality, upw)
    summary = st.pop("summary")
    G.update(st)
    # expected TC rain per month: passages x the month's genesis share x rain per passage
    lam = np.asarray(summary["tc_genesis_per_month"], dtype=np.float64)
    share = lam / max(lam.sum(), 1e-12)
    rain = storms.tc_rain_per_passage_mm(st["tc_max_wind_ms"]) * st["tc_passage_per_yr"]
    comps = G["precip_components_mm"].astype(np.float64)
    tc_m = np.minimum(share[:, None, None] * rain[None], 0.6 * comps[0])
    comps[3] = tc_m  # the storm share is taken out of the convective rain: no double count
    comps[0] = comps[0] - tc_m
    G["precip_components_mm"] = comps.astype(np.float32)
    summary["tc_rain_attribution"] = {
        "method": "passage_frequency_x_rain_per_passage, moved from convective",
        "tc_share_of_tropical_ocean_rain": float(tc_m.sum() / max(comps.sum(), 1e-9)),
    }
    return summary


def ocean_mode_response(normals, upwelling):
    """Rain response to +1 K on the equatorial upwelling tongue (circulation fixed), or None."""
    from .atmosphere import moisture

    if upwelling is None:
        return None
    lat = normals.lat
    sea = normals.sea_frac > 0.5
    pat = np.where(sea & (np.abs(lat) < 12.0), np.maximum(upwelling.mean(axis=0), 0.0), 0.0)
    from scipy import ndimage

    pat = ndimage.gaussian_filter(pat, 2.0)
    if pat.max() <= 1e-6:
        return None
    pat = pat / pat.max()
    inp = normals.inputs
    n = lat.size
    sst = inp["sst"] + pat.ravel()[None] * 1.0
    m1 = moisture.solve(
        inp["mesh"],
        inp["cu"],
        inp["cv"],
        inp["t_air"],
        sst,
        inp["ice"],
        normals.sea_frac.ravel(),
        normals.land_h,
        lat,
        inp["planet"],
        inp["days"],
        normals.grid["cell_km"] * 1000.0,
        pet_mm=inp["pet"],
        picard=inp["picard"],
    )
    p0 = normals.moist.precip_mm.reshape(12, n)
    resp = (m1.precip_mm - p0) / np.maximum(p0, 1.0)
    return {
        "response": np.clip(resp, -0.8, 0.8).reshape((12,) + lat.shape),
        "pattern_note": "+1 K on the equatorial upwelling tongue of the v5.3 ocean; "
        "thermodynamic response with the circulation fixed",
    }


def _weather_year(config, wc, normals, G, grid, days, storm_summary, project, layers, shape):
    """v5.4c: one synthetic year; its effects are a delta committed as separate layers."""
    from .atmosphere import weather
    from .atmosphere.normals import block_mean
    from .earth.planet import Planet

    atm = config.atmosphere
    planet = Planet.from_params(config.earth)
    year_seed = atm.weather_year_seed or config.seed
    lam = storm_summary["tc_genesis_per_month"] if storm_summary else None
    ocean_mode = None
    domain = normals.quality.get("domain")
    oc = getattr(getattr(wc, "earth", None), "ocean", None)
    if domain == "planetary_background" and oc is not None and normals.inputs:
        season_of = (0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3, 0)
        u4 = np.stack(
            [block_mean(np.nan_to_num(oc.upwelling_m_day[s]), normals.factor) for s in range(4)]
        )
        ocean_mode = ocean_mode_response(normals, np.stack([u4[season_of[m]] for m in range(12)]))
    wy = weather.generate(
        G,
        grid,
        planet,
        days,
        config.seed,
        year_seed,
        int(atm.weather_year_index),
        lam,
        atm.weather_year_mode,
        ocean_mode,
        incoming_tc_per_year=float(getattr(atm, "incoming_tc_per_year", 0.0)),
    )
    wy["climate_id"] = normals.source_hash
    G["wy_monthly_actual_mm"] = wy["aggregate_monthly_actuals_mm"]
    G["wy_annual_anomaly_pct"] = wy["annual_anomaly_pct"]
    G["wy_max_daily_mm"] = wy["max_daily_mm"]
    G["wy_drought"] = wy["drought"].astype(np.uint8)
    G["wy_onset_day"] = wy["onset_day"]
    G["wy_flood_day_mm"] = wy["flood_day_mm"]
    delta = apply_weather_effects(project, grid, wy, G["precipitation_mm"], shape)
    layers.update(delta)  # committed together with the stage (atomic: one data object)
    events = [_plain_event(e) for e in wy["events"]]
    counts = {}
    for e in events:
        counts[e["event_type"]] = counts.get(e["event_type"], 0) + 1
    summary = {
        "year_seed": year_seed,
        "year": int(atm.weather_year_index),
        "mode": atm.weather_year_mode,
        "climate_id": wy["climate_id"],
        "event_counts": counts,
        "interannual": wy["interannual"],
        "ledger": wy["ledger"],
        "quality": wy["quality"],
        "flood_day_of_year": wy["flood_day"] + 1,
        "effects": "delta layers weather_*: the climate normals and the potential "
        "biomes are not changed by one year (T72)",
    }
    return summary, events


def _plain_event(e):
    out = {}
    for k, v in e.items():
        if isinstance(v, dict):
            out[k] = _plain_event(v)
        elif isinstance(v, list):
            out[k] = [_plain_event(x) if isinstance(x, dict) else x for x in v]
        elif isinstance(v, (np.floating, np.integer, np.bool_)):
            out[k] = v.item()
        else:
            out[k] = v
    return out


def apply_weather_effects(project, grid, wy, p_norm_grid, shape) -> dict:
    """``applyWeatherEffects``: the year's delta for the hexes (nothing is modified here).

    Annual actual rain = hex norm x (actual / norm) of the atmosphere grid; the rain of the
    flood day (the day with the largest excess over the land) is routed down the existing
    river network (``flow_receiver``) as a one-day volume (rain above 30 mm of
    infiltration); storm surge stays a potential only.
    """
    lay = project.layout
    hx, hy = lay.centers()
    L = project.hydro.layers
    land = L["land_mask"].astype(bool)
    p_norm_hex = np.nan_to_num(project.env.monthly["precip_monthly_mm"].astype(np.float64))
    act = wy["aggregate_monthly_actuals_mm"].astype(np.float64).sum(axis=0)
    ratio = act / np.maximum(np.asarray(p_norm_grid, dtype=np.float64).sum(axis=0), 1e-9)
    p_act_hex = p_norm_hex.sum(axis=0) * sample(grid, ratio, hx, hy)
    max_daily = sample(grid, wy["max_daily_mm"], hx, hy)
    flood_rain = sample(grid, wy["flood_day_mm"], hx, hy)
    excess = np.where(land, np.maximum(flood_rain - 30.0, 0.0), 0.0)
    area_m2 = lay.hex_area_km2 * 1e6
    flood_q = route_volume(L, excess.ravel() * 1e-3 * area_m2).reshape(shape) / 86400.0
    q_mean = np.maximum(L["discharge_m3_s"].astype(np.float64), 1e-3)
    storm_wind = storm_max_wind_hex(project, wy["events"])
    damage = np.where(land, WINDTHROW_MAX * np.clip((storm_wind - 25.0) / 30.0, 0.0, 1.0), 0.0)
    snow = sample(grid, wy["snowfall_monthly_mm"].astype(np.float64).sum(axis=0), hx, hy)
    snow_share = np.clip(snow / np.maximum(sample(grid, act, hx, hy), 1e-9), 0.0, 1.0)
    f32 = np.float32
    return {
        "weather_snowfall_annual_mm": np.where(land, p_act_hex * snow_share, np.nan).astype(f32),
        "weather_storm_max_wind_ms": storm_wind.astype(f32),
        "weather_canopy_damage_fraction": damage.astype(f32),
        "weather_precip_actual_annual_mm": np.where(land, p_act_hex, np.nan).astype(f32),
        "weather_annual_anomaly_pct": sample(grid, wy["annual_anomaly_pct"], hx, hy).astype(f32),
        "weather_max_daily_rain_mm": max_daily.astype(f32),
        "weather_drought": (sample(grid, wy["drought"].astype(np.float64), hx, hy) > 0.5) & land,
        "weather_onset_day": np.rint(
            sample(grid, wy["onset_day"].astype(np.float64), hx, hy)
        ).astype(np.int16),
        "weather_flood_peak_m3_s": np.where(land, flood_q, 0.0).astype(f32),
        "weather_flood_peak_ratio": np.where(land, flood_q / q_mean, 0.0).astype(f32),
    }


#: [H] largest share of a stand thrown by the strongest storm winds (55 m/s and more)
WINDTHROW_MAX = 0.6
HOLLAND_RMAX_KM = 40.0


def storm_max_wind_hex(project, events) -> np.ndarray:
    """Max storm wind of the year per hex: Holland gradient wind around each TC position.

    The eye scale (tens of km) is resolved on the hexes, not on the ~300 km atmosphere
    grid; over land the surface wind is reduced to 0.75 of the gradient wind [A].
    """
    from scipy.spatial import cKDTree

    from .atmosphere.storms import dp_for_vmax, holland_wind
    from .earth.planet import EARTH

    lay = project.layout
    hx, hy = lay.centers()
    shape = hx.shape
    out = np.zeros(hx.size)
    pts = [
        p
        for e in events
        if e["event_type"] == "TC"
        for p in e["track"]["points"]
        if p.get("v_ms", 0.0) >= 17.0
    ]
    if not pts:
        return out.reshape(shape)
    tree = cKDTree(np.column_stack([hx.ravel(), hy.ravel()]))
    lat = project.hydro.layers["latitude_deg"].ravel()
    land = project.hydro.layers["land_mask"].ravel().astype(bool)
    for p in pts:
        idx = tree.query_ball_point([p["x_km"], p["y_km"]], 300.0)
        if not idx:
            continue
        idx = np.asarray(idx)
        r = np.hypot(hx.ravel()[idx] - p["x_km"], hy.ravel()[idx] - p["y_km"]) * 1000.0
        dp = dp_for_vmax(p["v_ms"])
        f = float(EARTH.coriolis(np.mean(lat[idx])))
        v = holland_wind(r, 101000.0 - dp, 101000.0, HOLLAND_RMAX_KM * 1000.0, f)
        v = np.minimum(v, p["v_ms"]) * np.where(land[idx], 0.75, 1.0)
        out[idx] = np.maximum(out[idx], v)
    return out.reshape(shape)


def route_volume(L, volume_m3):
    """Accumulate a volume down ``flow_receiver`` (upstream first, by drainage area)."""
    rc = L["flow_receiver"].ravel().astype(np.int64)
    order = np.argsort(L["flow_accumulation_area_km2"].ravel(), kind="stable")
    acc = np.asarray(volume_m3, dtype=np.float64).copy()
    for i in order:
        j = rc[i]
        if 0 <= j < acc.size and j != i:
            acc[j] += acc[i]
    return acc


def month_days_of(config):
    from .earth.planet import Planet

    return month_days(Planet.from_params(config.earth))


def run_v54(config, geometry, project, old_keys: dict, old_cache: dict, report):
    keys = atmosphere_stage_keys(config, project)
    if old_keys.get("v54_atmosphere") == keys["v54_atmosphere"] and "v54_atmosphere" in old_cache:
        data = old_cache["v54_atmosphere"]
        return data, {"v54_atmosphere": data}, ["v54_atmosphere"]
    report("v54_atmosphere", 0.9975)
    res = run_atmosphere(config, project)
    if "blocked" in res:
        data = AtmosphereData(
            layers={},
            grid_fields={},
            grid={},
            summary={"status": "blocked", "reason": res["blocked"]},
            provenance={"algorithm_version": ALGORITHM_VERSION},
            stage_keys=keys,
            warnings=["v5.4 atmosphere blocked: needs the v5.3 Earth system"],
        )
        return data, {"v54_atmosphere": data}, []
    data = AtmosphereData(
        layers=res["layers"],
        grid_fields=res["grid_fields"],
        grid=res["grid"],
        summary=res["summary"],
        provenance={"algorithm_version": ALGORITHM_VERSION, "params": asdict(config.atmosphere)},
        stage_keys=keys,
        events=res["events"],
        warnings=[],
        timings=res["timings"],
    )
    coup = res["summary"]["world_climate"].get("ocean_coupling", {})
    if coup.get("status") == "NOT_CONVERGED":
        data.warnings.append(
            "v5.4: atmosphere-ocean coupled iteration NOT_CONVERGED "
            "(iteration limit reached; see the report)"
        )
    return data, {"v54_atmosphere": data}, []
