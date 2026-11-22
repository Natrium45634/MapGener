"""v4.7 stage: topoclimate on terrain units (TZ 4.7, section 10).

    v4.6 surface + water connectivity + v3 mountain plan
    + v4 / v4.5 coarse climate, soils and water of the *same* surface revision
      -> samples of the sub-hex surface, water inside the hex, terrain units (TU)
      -> morphometrics, horizons, sky view, upwind slope Sx
      -> per TU and month: temperature (lapse from the reference height, cold-air pool,
         sun on the slope), radiation, humidity, wind, precipitation (redistributed and
         conserved per hex), snow drift, PET, soil water
      -> the v4.5 water kernel on every dry TU (spin-up, snow, soil, groundwater)
      -> growing season (daily curve), thermal tree limit, thermal belts, snow cover
      -> hex aggregates (area shares, hypsometry, boundaries) and massif summaries

Nothing upstream is changed. Every coarse value carries its reference height, so no
correction is applied twice; the precipitation of every hex is the v4.5 precipitation.

Terrain units are defined by world coordinates (sample points, their heights and the
landform), not by the hex: the hex only groups them. A future hex-free surface can feed
the same unit pipeline with its own sample points.
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import asdict

import numpy as np

from .project import TopoData
from .stages import environment as env
from .stages.climate import potential_et
from .topoclimate import atmosphere as atm
from .topoclimate import redistribution as rd
from .topoclimate import solar
from .topoclimate import thermal as th
from .topoclimate.units import (
    LANDFORMS,
    CoarseSurface,
    SubHexSurface,
    build_samples,
    gradient,
    midrank,
    stratify,
    water_type,
)
from .version import TOPOCLIMATE_ALGORITHM_VERSION

#: per-unit monthly fields kept (float32, shape (12, U))
UNIT_MONTHLY = {
    "temperature_c": ("degC", "air temperature of the unit"),
    "precip_mm": ("mm", "precipitation after redistribution and snow drift"),
    "pet_mm": ("mm", "reference PET of the unit (v4.5 index, radiation-scaled)"),
    "aet_mm": ("mm", "actual ET of the reference cover"),
    "swe_mm": ("mm", "snow water equivalent at the month end"),
    "runoff_mm": ("mm", "local runoff (not river discharge)"),
    "radiation_mj_m2_day": ("MJ/m^2/day", "shortwave on the slope surface"),
    "rh_frac": ("0..1", "relative humidity (synthetic profile, see provenance)"),
}
#: per-hex monthly aggregates for the maps, (12, rows, cols) float32
HEX_MONTHLY = {
    "topo_temperature_c": ("degC", "area-weighted air temperature of the dry units"),
    "topo_delta_temperature_c": ("degC", "dry-unit mean minus the coarse v4.5 temperature"),
    "topo_radiation_mj_m2_day": ("MJ/m^2/day", "area-weighted shortwave on the slopes"),
    "topo_swe_mm": ("mm", "area-weighted snow water equivalent, month end"),
    "topo_snow_cover_frac": ("0..1", "dry-land share with SWE >= the cover threshold"),
    "topo_zero_degree_level_m": ("m", "height of 0 degC in the reference profile"),
    "topo_snowline_m": ("m", "seasonal snowline bracket inside the hex (SWE criterion), NaN"),
}
#: hex layers (rows, cols)
TOPO_LAYER_SPECS = {
    "topo_dry_fraction": ("0..1", "dry land share of the hex area (f_dry)"),
    "topo_lake_fraction": ("0..1", "lake share of the hex area (f_lake)"),
    "topo_ocean_fraction": ("0..1", "sea share of the hex area (f_ocean)"),
    "topo_unit_count": ("count", "terrain units of the hex"),
    "hyps_z_mean_m": ("m", "mean height of the dry samples"),
    "hyps_z_std_m": ("m", "standard deviation of the dry sample heights"),
    "hyps_q05_m": ("m", "5% quantile of the dry hypsometry"),
    "hyps_q25_m": ("m", "25% quantile"),
    "hyps_q50_m": ("m", "median"),
    "hyps_q75_m": ("m", "75% quantile"),
    "hyps_q95_m": ("m", "95% quantile"),
    "hyps_relief_m": ("m", "range of the dry samples"),
    "dominant_thermal_belt": ("enum", "index into topoclimate.json -> belts"),
    "thermal_transition_fraction": ("0..1", "dry share with |thermal margin| < 0.5"),
    "thermal_margin_mean": ("-", "area mean of the thermal tree margin (clipped to +-10)"),
    "thermal_tree_score_mean": ("0..1", "area mean of the thermal tree score"),
    "site_tree_score_mean": ("0..1", "thermal score x snow, soil, wind, stability factors"),
    "gsl_days_mean": ("days", "growing season length, T > 0.9 degC (area mean)"),
    "gst_c_mean": ("degC", "mean temperature of the growing season (area mean)"),
    "gdd5_mean": ("degC day", "growing degree days above 5 degC"),
    "treeline_proxy_m": ("m", "height of the thermal tree limit inside the hex, NaN if none"),
    "treeline_state": ("enum", "0 n/a, 1 crossing, 2 all units too cold, 3 all warm enough"),
    "perennial_snow_fraction": ("0..1", "dry share with snow in every month"),
    "warm_season_snow_cover_frac": ("0..1", "snow cover in the locally warmest month"),
    "snowline_patchy": ("bool", "snow-free units above snow-covered ones in some month"),
    "permafrost_potential_mean": ("0..1", "frost-number proxy with snow insulation [H]"),
    "snow_surplus_mm_yr": ("mm/yr", "snowfall minus melt (perennial snow accumulation)"),
    "cap_potential_max": ("0..1", "largest cold-air-pool potential of the hex"),
    "cap_temperature_jan_c": ("degC", "January mean CAP anomaly (area mean)"),
    "svf_mean": ("0..1", "radiative sky view factor of the slopes (area mean)"),
    "wind_exposure_mean": ("rad", "exposure: max(-Sx, 0), area mean"),
    "cloud_potential_max": ("0..1", "largest orographic cloud potential of the year"),
    "fog_potential_mean": ("0..1", "mean fog potential of the year"),
    "precip_factor_max": ("-", "largest redistribution factor among the dry units"),
    "topo_local_runoff_mm_yr": ("mm/yr", "area mean local runoff of the dry units"),
}


def _key(*parts) -> str:
    return hashlib.sha256(json.dumps(parts, sort_keys=True, default=str).encode()).hexdigest()


def topo_stage_keys(config, upstream: dict) -> dict:
    return {
        "v47_topoclimate": _key(
            "v47", TOPOCLIMATE_ALGORITHM_VERSION, upstream, asdict(config.topoclimate)
        )
    }


def mountain_detail(project):
    """``detail(x, y) -> (relief, structure, valley)`` of the v3 mountain plan, or None.

    The same deterministic sampler v3 used for the hex centres, asked at sub-hex points.
    """
    from scipy import ndimage

    from .mountains import MountainRequest, get_backend
    from .mountains.plan import MountainPlan
    from .stages.history import HistoryGrid

    hist = project.hist
    cfg = project.config
    plan_d = hist.plan or {}
    w = hist.world or {}
    if not plan_d.get("ridges") or "orogenic_relief_m" not in w or cfg.mountains.detail <= 0:
        return None
    grid = HistoryGrid.from_dict(w["grid"])
    relief = np.asarray(w["orogenic_relief_m"], dtype=np.float64)
    plan = MountainPlan.from_dict(plan_d)
    sys_raster = np.asarray(w.get("mountain_system_id", np.full(grid.shape, -1)))
    gain_raster = np.full(grid.shape, 0.6)
    for s in plan.systems:
        gain_raster[sys_raster == s.id] = s.relief_gain
    gain_raster = ndimage.gaussian_filter(gain_raster, 1.0)
    request = MountainRequest(
        world_seed=cfg.seed,
        snapshot_id=f"t{hist.epoch_time_ma:g}",
        world_bounds_km=tuple(project.geometry.layout.bounds.as_list()),
        plan=plan,
        uplift=lambda x, y: grid.bilinear(relief, x, y),
        gain=lambda x, y: grid.bilinear(gain_raster, x, y),
        detail=cfg.mountains.detail,
        max_elevation_m=cfg.mountains.max_elevation_m,
        policy="masked",
        constraints={
            "grid": grid.to_dict(),
            "amplitude": relief.astype(np.float32),
            "gain": gain_raster.astype(np.float32),
        },
    )
    sampler = get_backend(cfg.mountains.backend).sampler(request)

    def detail(x, y):
        x = np.asarray(x, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        rel = np.zeros(x.size)
        st = np.zeros(x.size)
        va = np.zeros(x.size)
        step = 400_000
        for c0 in range(0, x.size, step):
            res = sampler.sample_points(x[c0 : c0 + step], y[c0 : c0 + step])
            rel[c0 : c0 + step] = res.relief_m
            st[c0 : c0 + step] = res.structure
            va[c0 : c0 + step] = res.valley
        return rel, st, va

    return detail


def _hex_inputs(project):
    """Coarse hex inputs of the same surface revision, flattened (n,) / (12, n)."""
    L = project.hydro.layers
    E = project.env.layers
    M = project.env.monthly
    n = L["elevation_m"].size

    def f(a):
        return np.asarray(a).reshape(-1).astype(np.float64)

    terrain = project.terrain.layers if project.terrain is not None else {}
    return {
        "z": f(L["elevation_m"]),
        "ocean": np.asarray(L["ocean_mask"]).ravel().astype(bool),
        "lake": np.asarray(L["inland_water_mask"]).ravel().astype(bool),
        "lake_level": f(L["lake_surface_level_m"]),
        "lat": f(L["latitude_deg"]),
        "wind_u": f(L["wind_u_ms"]),
        "wind_v": f(L["wind_v_ms"]),
        "t_m": np.asarray(M["temperature_monthly_c"], dtype=np.float64).reshape(12, n),
        "p_m": np.nan_to_num(np.asarray(M["precip_monthly_mm"], dtype=np.float64).reshape(12, n)),
        "pet_m": np.nan_to_num(np.asarray(M["pet_monthly_mm"], dtype=np.float64).reshape(12, n)),
        "awc": np.nan_to_num(f(E["awc_mm"])),
        "soil_depth": np.nan_to_num(f(E["soil_depth_m"])),
        "icap": np.nan_to_num(f(E["infiltration_capacity_mm_day"]), nan=5.0),
        "ks": np.nan_to_num(f(E["ks_mm_h"]), nan=0.5),
        "marine": np.nan_to_num(f(E["coastal_marine_influence"])),
        "massif": np.asarray(project.hist.layers["mountain_system_id"]).ravel(),
        "spill": f(terrain["spill_level_m"]) if "spill_level_m" in terrain else None,
    }


def _units(sample_hex, code, weights, feats, n_hex):
    """Group samples into units: ids, weight, representative sample, hex offsets.

    The representative is the sample nearest to the unit mean in (height, slope, aspect):
    a real point of the surface, never an average that may lie in the air.
    """
    key = sample_hex.astype(np.int64) * 256 + code
    uniq, inverse = np.unique(key, return_inverse=True)
    u = uniq.size
    wsum = np.bincount(inverse, weights=weights, minlength=u)
    means = [np.bincount(inverse, weights=weights * f, minlength=u) / wsum for f in feats]
    dist = sum((f - m[inverse]) ** 2 for f, m in zip(feats, means, strict=True))
    order = np.lexsort((np.arange(dist.size), dist, inverse))
    first = np.ones(order.size, dtype=bool)
    first[1:] = inverse[order][1:] != inverse[order][:-1]
    rep = order[first]
    unit_hex = (uniq // 256).astype(np.int64)
    offsets = np.searchsorted(unit_hex, np.arange(n_hex + 1))
    return inverse, wsum, rep, unit_hex, (uniq % 256).astype(np.int64), offsets


def _near_horizon(ux, uy, uz, uhex, sx, sy, sz, s_hex, n_dir):
    """Signed horizon (U, n_dir) from the samples of the unit's own hex (near field)."""
    u = ux.size
    if u == 0:
        return np.full((0, n_dir), np.nan)
    starts = np.searchsorted(s_hex, uhex, side="left")  # samples are sorted by hex
    ends = np.searchsorted(s_hex, uhex, side="right")
    m = int((ends - starts).max())
    idx = starts[:, None] + np.arange(m)[None, :]
    ok = idx < ends[:, None]
    idx = np.minimum(idx, s_hex.size - 1)
    dxe = sx[idx] - ux[:, None]
    dyn = -(sy[idx] - uy[:, None])  # world +y is south
    dist = np.hypot(dxe, dyn) * 1000.0
    ok &= dist > 1.0
    dz = np.where(ok, sz[idx] - uz[:, None], np.nan)
    return solar.horizon_angles(dist, dz, np.arctan2(dxe, dyn), n_dir, signed=True)


def _far_horizon(layout, far_z, ux, uy, uz, n_dir, max_km):
    """Signed horizon (U, n_dir) by rays over the far field (hex 95% heights) [H].

    ``approx_multires_horizon``: geometric steps from 0.6 spacing to ``max_km``.
    """
    sp = layout.spacing_km
    steps = []
    d = 0.6 * sp
    while d <= max_km + 1e-9:
        steps.append(d)
        d = max(d * 1.45, d + 0.5 * sp)
    out = np.full((ux.size, n_dir), np.nan)
    for k in range(n_dir):
        az = (k + 0.5) * 2.0 * np.pi / n_dir
        se, sn = np.sin(az), np.cos(az)
        best = np.full(ux.size, -np.inf)
        for d in steps:
            col, row = layout.world_to_offset(ux + d * se, uy - d * sn)
            inside = layout.inside(col, row)
            h = np.where(inside, row * layout.width + col, 0)
            ang = np.arctan2(far_z[h] - uz, d * 1000.0)
            best = np.where(inside, np.maximum(best, ang), best)
        out[:, k] = np.where(np.isfinite(best), best, np.nan)
    return out


def _chunks(offsets, target):
    """Unit slices aligned to hex boundaries, about ``target`` units each."""
    total = int(offsets[-1])
    if total == 0:
        return []
    cuts = np.unique(offsets[np.searchsorted(offsets, np.arange(target, total, target))])
    bounds = [0] + [int(c) for c in cuts if 0 < c < total] + [total]
    return [slice(a, b) for a, b in zip(bounds[:-1], bounds[1:], strict=True) if b > a]


def _hex_max(values, unit_hex, n_hex, mask):
    out = np.full(n_hex, -np.inf)
    v = np.asarray(values, dtype=np.float64)
    np.maximum.at(out, unit_hex[mask], v[mask])
    return np.where(np.isfinite(out), out, np.nan)


_INT_LAYERS = {
    "topo_unit_count": np.int16,
    "dominant_thermal_belt": np.uint8,
    "treeline_state": np.uint8,
    "snowline_patchy": np.bool_,
}


def _empty_result(config, shape, H, sea, lapse):
    """Result of a window without land: sea fractions 1, every land field NoData."""
    n = shape[0] * shape[1]
    layers = {}
    for name in TOPO_LAYER_SPECS:
        dt = _INT_LAYERS.get(name)
        layers[name] = np.zeros(shape, dtype=dt) if dt else np.full(shape, np.nan, dtype=np.float32)
    layers["topo_dry_fraction"][:] = 0.0
    layers["topo_lake_fraction"][:] = 0.0
    layers["topo_ocean_fraction"][:] = 1.0
    for name in th.BELTS[1:]:
        layers[f"belt_share_{name.replace('thermal_', '')}"] = np.zeros(shape, dtype=np.float32)
    monthly = {k: np.full((12,) + shape, np.nan, dtype=np.float32) for k in HEX_MONTHLY}
    units = {"hex_offsets": np.zeros(n + 1, dtype=np.int64)}
    units["unit_hex"] = np.zeros(0, dtype=np.int32)
    for k in UNIT_MONTHLY:
        units[k] = np.zeros((12, 0), dtype=np.float32)
    for k in ("unit_weight_hex", "unit_weight_land", "z_surface_m"):
        units[k] = np.zeros(0, dtype=np.float32)
    units["unit_surface_type"] = np.zeros(0, dtype=np.uint8)
    units["unit_landform"] = np.zeros(0, dtype=np.uint8)
    units["thermal_belt"] = np.zeros(0, dtype=np.uint8)
    summary = {
        "units": 0, "dry_units": 0, "samples": 0, "sampled_hexes": 0, "detail_hexes": 0,
        "units_per_land_hex": 0.0, "area_closure_max_error": 0.0,
        "precipitation_conservation_max_rel": 0.0, "water_budget_max_residual_mm": 0.0,
        "spinup_years": 0, "spinup_converged_share": 1.0, "persistent_snow_units": 0,
        "belt_land_shares": {b: 0.0 for b in th.BELTS}, "treeline_hexes": 0,
        "treeline_crossings": 0, "regional_treelines_m": [],
        "mean_abs_delta_temperature_c": 0.0, "max_precip_factor": 1.0,
        "cap_units_over_0_5": 0, "condensation_unit_months": 0, "no_land": True,
    }  # fmt: skip
    provenance = {
        "note": "no land in the window: topoclimate not calculated",
        "algorithm_version": TOPOCLIMATE_ALGORITHM_VERSION,
    }
    return {
        "layers": layers,
        "monthly": monthly,
        "units": units,
        "summary": summary,
        "massifs": [],
        "provenance": provenance,
        "timings": {"v47_total": 0.0},
    }


def run_topoclimate(config, geometry, project, report=None):  # noqa: C901 - one linear stage
    """Compute v4.7 for a project that holds v4.6, v4 and v4.5 of the same surface."""
    tp = config.topoclimate
    layout = geometry.layout
    shape = (layout.height, layout.width)
    n_hex = shape[0] * shape[1]
    timings: dict = {}
    rep = report or (lambda f: None)
    t0 = time.perf_counter()
    H = _hex_inputs(project)
    sea = float(project.hydro.sea_level_m)
    lapse = float(config.climate.lapse_rate_c_per_km)

    # -- Gate A: samples of the sub-hex surface ---------------------------------------------
    coarse = CoarseSurface(layout, H["z"])
    land = ~H["ocean"]
    near_land = land.copy()
    for k in range(6):
        nb = coarse.nbr[:, k]
        near_land |= (nb >= 0) & land[np.maximum(nb, 0)]
    hexes = np.nonzero(near_land)[0]
    if hexes.size == 0:  # an all-sea window: nothing to downscale (NoData, not defaults)
        return _empty_result(config, shape, H, sea, lapse)
    detail = mountain_detail(project) if (H["massif"][hexes] >= 0).any() else None
    mountain = hexes[H["massif"][hexes] >= 0] if detail else np.zeros(0, dtype=np.int64)
    surface = SubHexSurface(coarse, detail, mountain, config.mountains.max_elevation_m)
    x, y, s_hex, s_tri, w = build_samples(layout, hexes, mountain, tp.quality)
    z, struct, valley = surface(x, y, want_structure=True)
    h_km = float(np.clip(0.1 * layout.spacing_km, 0.05, 5.0))
    p_e, q_n = gradient(surface, x, y, h_km)
    slope, aspect = solar.slope_aspect(p_e, q_n)
    slope_deg = np.degrees(slope)
    wt = water_type(z, s_hex, s_tri, coarse.nbr, H["ocean"], H["lake"], H["lake_level"], sea)
    timings["v47_samples"] = round(time.perf_counter() - t0, 3)
    rep(0.2)

    # -- Gate B: terrain units, hypsometry ----------------------------------------------------
    t0 = time.perf_counter()
    code, landform, _band, relief = stratify(
        s_hex, wt, z, slope_deg, aspect, struct, valley, n_hex, tp.max_units_per_hex
    )
    steep = np.minimum(slope_deg / 10.0, 1.0)
    feats = [
        z / 200.0,
        slope_deg / 10.0,
        np.nan_to_num(np.sin(aspect)) * steep,
        np.nan_to_num(np.cos(aspect)) * steep,
    ]
    inv, u_w, rep_s, u_hex, u_code, offsets = _units(s_hex, code, w, feats, n_hex)
    n_units = u_hex.size
    u_type = np.where(u_code == 0, 2, np.where(u_code == 1, 1, 0)).astype(np.uint8)
    dry_u = u_type == 0
    u_x, u_y, u_z = x[rep_s], y[rep_s], z[rep_s]
    u_slope, u_aspect = slope[rep_s], aspect[rep_s]
    u_p, u_q = p_e[rep_s], q_n[rep_s]
    u_landform = np.where(dry_u, landform[rep_s], 0).astype(np.uint8)
    u_valley = np.bincount(inv, weights=w * valley, minlength=n_units) / u_w
    u_struct = np.bincount(inv, weights=w * struct, minlength=n_units) / u_w
    dry = wt == 0
    rank_s = midrank(np.where(dry, z, np.inf), s_hex, n_hex)
    u_rank = np.bincount(inv, weights=w * rank_s, minlength=n_units) / u_w
    dry_w = np.bincount(s_hex, weights=w * dry, minlength=n_hex)
    lake_w = np.bincount(s_hex, weights=w * (wt == 1), minlength=n_hex)
    ocean_w = np.bincount(s_hex, weights=w * (wt == 2), minlength=n_hex)
    unsampled = np.ones(n_hex, dtype=bool)
    unsampled[hexes] = False
    ocean_w[unsampled] = 1.0  # only open-sea hexes are left unsampled
    safe_dw = np.maximum(dry_w, 1e-12)
    zmean = np.bincount(s_hex, weights=w * dry * z, minlength=n_hex) / safe_dw
    zsq = np.bincount(s_hex, weights=w * dry * z * z, minlength=n_hex) / safe_dw
    zstd = np.where(dry_w > 0, np.sqrt(np.maximum(zsq - zmean**2, 0.0)), np.nan)
    zmean = np.where(dry_w > 0, zmean, np.nan)
    # weighted quantiles of the dry samples of every hex (hypsometric CDF)
    order = np.lexsort((z, s_hex))
    zs, hs, ds, ws_ = z[order], s_hex[order], dry[order], w[order]
    cw = np.cumsum(np.where(ds, ws_, 0.0))
    first_pos = np.searchsorted(hs, np.arange(n_hex), side="left")
    base = np.concatenate([[0.0], cw])[first_pos]
    cdf = (cw - base[hs]) / safe_dw[hs]
    quant = {}
    for q in (0.05, 0.25, 0.5, 0.75, 0.95):
        idx = np.nonzero(ds & (cdf >= q - 1e-9))[0]
        out = np.full(n_hex, np.nan)
        if idx.size:
            hh = hs[idx]
            firsts = idx[np.r_[True, hh[1:] != hh[:-1]]]
            out[hs[firsts]] = zs[firsts]
        quant[q] = out
    del order, zs, hs, ds, ws_, cw, cdf
    timings["v47_units"] = round(time.perf_counter() - t0, 3)

    # -- Gate B: horizons, sky view, upwind slope -----------------------------------------------
    t0 = time.perf_counter()
    kdir = tp.horizon_directions
    far_z = np.where(np.isfinite(quant[0.95]), quant[0.95], np.maximum(H["z"], sea))
    hor_signed = np.full((n_units, kdir), np.nan)
    for sl in _chunks(offsets, 60_000):
        near = _near_horizon(u_x[sl], u_y[sl], u_z[sl], u_hex[sl], x, y, z, s_hex, kdir)
        far = _far_horizon(
            layout, far_z, u_x[sl], u_y[sl], u_z[sl], kdir, tp.horizon_max_distance_km
        )
        hor_signed[sl] = np.fmax(near, far)
    hor = np.maximum(np.nan_to_num(hor_signed, nan=0.0), 0.0)
    vis, svf_h, svf_slope = solar.sky_view(hor, u_slope)
    ue, un = H["wind_u"][u_hex], -H["wind_v"][u_hex]  # ENU: north = -v_world
    speed_ref = np.hypot(ue, un)
    from_az = np.mod(np.arctan2(-ue, -un), 2.0 * np.pi)
    centers = (np.arange(kdir) + 0.5) * 2.0 * np.pi / kdir
    diff = np.abs(np.angle(np.exp(1j * (centers[None, :] - from_az[:, None]))))
    sector = diff <= np.deg2rad(30.0) + np.pi / kdir
    sec_vals = np.where(sector, hor_signed, np.nan)
    n_ok = np.isfinite(sec_vals).sum(axis=1)
    sx_up = np.where(n_ok > 0, np.nansum(sec_vals, axis=1) / np.maximum(n_ok, 1), 0.0)
    sx_up = np.where(speed_ref > 0.2, sx_up, 0.0)
    del diff, sector, sec_vals
    timings["v47_horizon"] = round(time.perf_counter() - t0, 3)
    rep(0.4)

    # -- Gates C-F per chunk of hexes: climate, precipitation, water, season ---------------------
    t0 = time.perf_counter()
    z_ref_hex = np.where(H["ocean"], sea, H["z"])
    spill = H["spill"] if H["spill"] is not None else H["z"]
    closed_hex = np.maximum(spill - H["z"], 0.0)
    lake_level = np.where(np.isfinite(H["lake_level"]), H["lake_level"], sea)
    ep = config.environment
    wp = env.WaterParams(
        k_dd_mm=ep.degree_day_mm,
        depletion_p=ep.depletion_fraction,
        rain_days=ep.rain_days_per_month,
        baseflow_timescale_yr=ep.baseflow_timescale_yr,
        deep_loss_share=config.climate.deep_infiltration,
        max_years=ep.spinup_max_years,
        tolerance_mm=ep.water_balance_tolerance_mm,
    )
    f32 = np.float32
    U = {k: np.zeros((12, n_units), dtype=f32) for k in UNIT_MONTHLY}
    scalars = (
        "z_ref_m", "cap_static", "cap_jan_c", "wind_local_ms", "precip_factor", "awc_mm",
        "soil_depth_m", "gsl_t_days", "gst_t_c", "gsl_site_days", "longest_season_days",
        "gdd0", "gdd5", "gdd10", "fdd", "min_daily_c", "thermal_margin", "thermal_tree_score",
        "site_tree_score", "f_snowfree", "f_soil", "f_wind", "f_stability",
        "snow_surplus_mm_yr", "cloud_potential_max", "fog_potential_mean",
        "frost_number", "permafrost_potential",
    )  # fmt: skip
    S = {k: np.zeros(n_units, dtype=np.float64) for k in scalars}
    belt = np.zeros(n_units, dtype=np.uint8)
    perennial = np.zeros(n_units, dtype=bool)
    converged = np.ones(n_units, dtype=bool)
    persistent = np.zeros(n_units, dtype=bool)
    condensation_months = 0
    worst = 0.0
    years = 0
    month_of_day = np.repeat(np.arange(12), solar.MONTH_DAYS)
    decl_mid = solar.declination(solar.REPRESENTATIVE_DOY[:, 1])
    chunks = _chunks(offsets, 40_000)
    for ci, sl in enumerate(chunks):
        hx_ = u_hex[sl]
        h0 = int(hx_[0])
        g = hx_ - h0
        ng = int(g.max()) + 1
        typ = u_type[sl]
        d_u = typ == 0
        w_ = u_w[sl]
        # temperature: lapse from the reference height of the coarse value, then anomalies
        z_ref = z_ref_hex[hx_]
        z_air = np.where(d_u, u_z[sl], np.where(typ == 1, lake_level[hx_], sea))
        t_hex = H["t_m"][:, hx_]
        t_base = atm.lapse_temperature(t_hex, z_ref[None, :], z_air[None, :], lapse)
        lat = np.deg2rad(H["lat"][hx_])
        p_hex = H["p_m"][:, hx_]
        pet_hex = H["pet_m"][:, hx_]
        cloud = np.clip(0.15 + 0.65 * p_hex / (p_hex + pet_hex + 5.0), 0.1, 0.85)
        normal = solar.surface_normal(u_p[sl], u_q[sl])
        r_slope, r_ref, _sun_h = solar.monthly_radiation(
            lat, normal, hor[sl], svf_slope[sl], u_slope[sl], 1.0 - cloud,
            tp.sunlight_hours_per_day_samples,
        )  # fmt: skip
        saddle = closed_hex[hx_] + np.maximum(np.nan_to_num(quant[0.75][hx_]) - u_z[sl], 0.0)
        cap = rd.cap_static(
            relief[hx_], u_rank[sl], np.degrees(u_slope[sl]), saddle, hor[sl].mean(axis=1)
        )
        cap = np.where(d_u, cap, 0.0)
        u_loc = rd.wind_speed_local(speed_ref[sl], sx_up[sl])
        night = 24.0 - solar.day_length_h(lat[None, :], decl_mid[:, None])
        _cap_event, cap_mean = rd.cap_anomaly(
            cap[None, :], u_loc[None, :], cloud, night, tp.cap_night_max_c, tp.cap_event_frequency
        )
        sun_dt = rd.solar_temperature_anomaly(
            r_slope, r_ref, u_loc[None, :], k_t=tp.solar_day_anomaly_max_c
        )
        anomaly = cap_mean + np.where(d_u[None, :], sun_dt, 0.0)
        if tp.temperature_mode == "preserve_coarse_mean":
            wd = w_ * d_u
            den = np.bincount(g, weights=wd, minlength=ng)
            for m in range(12):
                num = np.bincount(g, weights=wd * anomaly[m], minlength=ng)
                anomaly[m] -= np.where(den > 0, num / np.maximum(den, 1e-12), 0.0)[g]
        t_loc = t_base + anomaly
        # humidity: synthetic reference RH carried as specific humidity to the local level
        rh_ref = np.clip(tp.rh_inland + (tp.rh_marine - tp.rh_inland) * H["marine"][hx_], 0.05, 1.0)
        p_ref = atm.pressure_fao_kpa(z_ref)
        p_loc = atm.pressure_fao_kpa(z_air)
        rh, _vpd, condense = atm.local_humidity(
            t_hex, rh_ref[None, :], p_ref[None, :], t_loc, p_loc[None, :]
        )
        condensation_months += int((condense & d_u[None, :]).sum())
        td_ref = atm.dew_point_c(t_hex, rh_ref[None, :])
        cloud_base = z_ref[None, :] + atm.lcl_height_agl_m(t_hex, td_ref)
        # precipitation: the detail gradient (the hex P already holds the hex-mean slope)
        wd = w_ * d_u
        den = np.maximum(np.bincount(g, weights=wd, minlength=ng), 1e-12)
        mp = np.bincount(g, weights=wd * u_p[sl], minlength=ng) / den
        mq = np.bincount(g, weights=wd * u_q[sl], minlength=ng) / den
        uplift = ue[sl] * (u_p[sl] - mp[g]) + un[sl] * (u_q[sl] - mq[g])
        shelter = np.clip(np.maximum(sx_up[sl], 0.0) / 0.35, 0.0, 1.0)
        raw = rd.orographic_factor(uplift, u_valley[sl] * shelter)
        factor = np.ones(raw.size)
        di = np.nonzero(d_u)[0]
        if di.size:
            factor[di] = rd.normalise_factors(
                raw[di], w_[di], g[di], ng, tp.precipitation_factor_bounds
            )
        precip = p_hex * factor[None, :]
        snow_share = 1.0 - rd.smoothstep(-1.0, 2.0, t_loc)
        drift = rd.shelter_factor(sx_up[sl])
        if di.size and tp.snow_mobile_share > 0:
            for m in range(12):
                snow = precip[m, di] * snow_share[m, di]
                moved = rd.redistribute_snow(
                    snow, w_[di], drift[di], tp.snow_mobile_share, g[di], ng
                )
                precip[m, di] += moved - snow
        cloud_pot = rd.cloud_potential(rh_ref[None, :], uplift[None, :], z_air[None, :], cloud_base)
        fog_pot = rd.fog_potential(rh, cap[None, :], u_loc[None, :])
        # PET and soil of the unit, then the v4.5 water kernel
        pe_hex = potential_et(t_hex)
        pe_loc = potential_et(t_loc)
        ratio_t = np.where(pe_hex > 1e-6, pe_loc / np.maximum(pe_hex, 1e-6), 1.0)
        ratio_r = np.clip(r_slope / np.maximum(r_ref, 1e-6), 0.5, 1.6) ** 0.5
        ratio_r = np.where(r_ref > 1e-6, ratio_r, 1.0)
        pet = pet_hex * ratio_t * ratio_r
        soil_f = np.clip(np.exp(-1.5 * np.tan(u_slope[sl])) * (1.0 + 0.5 * u_valley[sl]), 0.05, 2.0)
        rock = (u_landform[sl] == LANDFORMS.index("steep_rock")) | (u_slope[sl] > np.deg2rad(40.0))
        awc = np.where(rock, 0.0, H["awc"][hx_] * soil_f)
        depth = np.where(rock, 0.0, H["soil_depth"][hx_] * soil_f)
        drainage = np.clip(0.03 + 0.2 * np.log10(1.0 + H["ks"][hx_]), 0.02, 0.6)
        aet = np.zeros_like(precip)
        swe = np.zeros_like(precip)
        runoff = np.zeros_like(precip)
        surplus = np.zeros(raw.size)
        if di.size:
            wb = env.water_balance(
                precip[:, di], t_loc[:, di], pet[:, di], awc[di], H["icap"][hx_][di],
                drainage[di], wp,
            )  # fmt: skip
            aet[:, di] = wb.monthly["aet"]
            swe[:, di] = wb.monthly["snowpack"]
            runoff[:, di] = wb.monthly["runoff"]
            converged[sl.start + di] = wb.converged | wb.persistent_snow
            persistent[sl.start + di] = wb.persistent_snow
            surplus[di] = np.maximum(wb.monthly["snowfall"].sum(0) - wb.monthly["melt"].sum(0), 0.0)
            worst = max(worst, float(wb.max_residual_mm))
            years = max(years, int(wb.years))
            del wb
        # growing season on a mean-preserving daily curve; site factors
        daily = th.daily_from_monthly(t_loc)
        snowfree = (swe < tp.cover_swe_threshold_mm).astype(np.float64)
        moist = rd.smoothstep(0.2, 0.6, np.clip(aet / np.maximum(pet, 1e-6), 0.0, 1.0))
        moist = np.where(pet > 1e-6, moist, 1.0)
        met = th.season_metrics(daily, tp.tree_start_temp_c, snowfree[month_of_day])
        del daily
        gsl = met["gsl_days"]
        margin = th.thermal_margin(
            gsl, met["gst_c"], tp.tree_gsl_days, tp.tree_gst_c, tp.smoothing_days, tp.smoothing_c
        )
        score = th.thermal_score(margin)
        f_snow = np.where(gsl > 0, np.clip(met["gsl_site_days"] / np.maximum(gsl, 1e-9), 0, 1), 0)
        f_soil = rd.smoothstep(0.05, 0.5, depth) * (0.5 + 0.5 * moist.mean(axis=0))
        f_wind = 1.0 - 0.4 * np.clip(np.maximum(-sx_up[sl], 0.0) / 0.35, 0.0, 1.0)
        f_stab = 1.0 - rd.smoothstep(30.0, 45.0, np.degrees(u_slope[sl]))
        frost_free = met["min_daily_c"] > 5.0  # frost_free_proxy (no daily Tmin): flagged
        # permafrost potential [H]: Nelson-Outcalt frost number with snow insulation of the
        # freezing season (more snow = warmer ground in winter, never the opposite)
        cold = t_loc < 0.0
        winter_cover = np.where(
            cold.any(axis=0), (cold & (swe >= tp.cover_swe_threshold_mm)).sum(0)
            / np.maximum(cold.sum(0), 1), 0.0,
        )  # fmt: skip
        fdd_eff = met["fdd"] * (1.0 - 0.6 * winter_cover)
        frost_n = np.sqrt(fdd_eff) / np.maximum(np.sqrt(fdd_eff) + np.sqrt(met["tdd"]), 1e-9)
        frost_n = np.where(d_u, frost_n, 0.0)
        belt[sl] = th.thermal_belt(gsl, met["gst_c"], margin, frost_free, valid=d_u)
        perennial[sl] = d_u & np.all(swe >= tp.cover_swe_threshold_mm, axis=0)
        monthly_out = {
            "temperature_c": t_loc, "precip_mm": precip, "pet_mm": pet, "aet_mm": aet,
            "swe_mm": swe, "runoff_mm": runoff, "radiation_mj_m2_day": r_slope, "rh_frac": rh,
        }  # fmt: skip
        for name, arr in monthly_out.items():
            U[name][:, sl] = arr
        vals = {
            "z_ref_m": z_ref, "cap_static": cap, "cap_jan_c": cap_mean[0],
            "wind_local_ms": u_loc, "precip_factor": factor, "awc_mm": awc,
            "soil_depth_m": depth, "gsl_t_days": gsl, "gst_t_c": met["gst_c"],
            "gsl_site_days": met["gsl_site_days"],
            "longest_season_days": met["longest_season_days"], "gdd0": met["gdd0"],
            "gdd5": met["gdd5"], "gdd10": met["gdd10"], "fdd": met["fdd"],
            "min_daily_c": met["min_daily_c"], "thermal_margin": margin,
            "thermal_tree_score": score,
            "site_tree_score": score * f_snow * f_soil * f_wind * f_stab,
            "f_snowfree": f_snow, "f_soil": f_soil, "f_wind": f_wind, "f_stability": f_stab,
            "snow_surplus_mm_yr": surplus, "cloud_potential_max": cloud_pot.max(axis=0),
            "fog_potential_mean": fog_pot.mean(axis=0), "frost_number": frost_n,
            "permafrost_potential": rd.smoothstep(0.45, 0.6, frost_n),
        }  # fmt: skip
        for k_, v_ in vals.items():
            S[k_][sl] = v_
        rep(0.4 + 0.5 * (ci + 1) / len(chunks))
    for k_ in ("gsl_t_days", "gst_t_c", "gsl_site_days", "thermal_margin", "thermal_tree_score",
               "site_tree_score"):  # fmt: skip
        S[k_][~dry_u] = np.nan
    timings["v47_climate_water_season"] = round(time.perf_counter() - t0, 3)

    # -- Gate F: hex aggregates ------------------------------------------------------------------
    t0 = time.perf_counter()
    wdry = u_w * dry_u
    hex_w = np.bincount(u_hex, weights=wdry, minlength=n_hex)
    has = hex_w > 0

    def hmean(v, mask=None):
        ww = wdry if mask is None else wdry * mask
        den = np.bincount(u_hex, weights=ww, minlength=n_hex)
        vv = np.nan_to_num(np.asarray(v, dtype=np.float64))
        num = np.bincount(u_hex, weights=ww * vv, minlength=n_hex)
        return np.where(den > 0, num / np.maximum(den, 1e-12), np.nan)

    def grid(a, dtype=np.float32):
        return np.ascontiguousarray(np.asarray(a).reshape(shape).astype(dtype))

    belt_frac = np.zeros((len(th.BELTS), n_hex))
    for b in range(len(th.BELTS)):
        belt_frac[b] = np.bincount(u_hex, weights=wdry * (belt == b), minlength=n_hex)
    belt_share = np.where(has, belt_frac / np.maximum(hex_w, 1e-12), 0.0)
    dominant = np.where(has, np.argmax(belt_frac, axis=0), 0).astype(np.uint8)
    margin = S["thermal_margin"]
    tree_z = np.full(n_hex, np.nan)
    tree_state = np.zeros(n_hex, dtype=np.uint8)
    du = np.nonzero(dry_u)[0]
    crossings_total = 0
    if du.size:
        hh, mg, zz = u_hex[du], margin[du], u_z[du]
        warm = np.bincount(hh, weights=(mg >= 0), minlength=n_hex)
        cnt = np.bincount(hh, minlength=n_hex)
        tree_state[(cnt > 0) & (warm == cnt)] = 3
        tree_state[(cnt > 0) & (warm == 0)] = 2
        mixed = np.nonzero((cnt > 0) & (warm > 0) & (warm < cnt))[0]
        order_u = np.lexsort((zz, hh))
        hs_, zs_, ms_ = hh[order_u], zz[order_u], mg[order_u]
        st = np.searchsorted(hs_, mixed, side="left")
        en = np.searchsorted(hs_, mixed, side="right")
        for h_, a_, b_ in zip(mixed.tolist(), st.tolist(), en.tolist(), strict=True):
            cr = th.crossings(zs_[a_:b_], ms_[a_:b_])
            if cr:
                crossings_total += len(cr)
                tree_z[h_] = float(np.median(cr))
                tree_state[h_] = 1
    # seasonal snowline: bracket between the highest snow-free and the lowest snow-covered
    # unit of a hex (month-end SWE >= threshold); NaN without a transition, never from T < 0
    snowline = np.full((12, n_hex), np.nan)
    patchy = np.zeros(n_hex, dtype=bool)
    if du.size:
        for m in range(12):
            cov = U["swe_mm"][m, du] >= tp.cover_swe_threshold_mm
            lo_cov = np.full(n_hex, np.inf)
            hi_free = np.full(n_hex, -np.inf)
            np.minimum.at(lo_cov, u_hex[du][cov], u_z[du][cov])
            np.maximum.at(hi_free, u_hex[du][~cov], u_z[du][~cov])
            both = np.isfinite(lo_cov) & np.isfinite(hi_free)
            snowline[m, both] = 0.5 * (lo_cov[both] + hi_free[both])
            patchy |= both & (hi_free > lo_cov)
    t_coarse = H["t_m"]
    warm_month = np.argmax(t_coarse, axis=0)  # the local warm season, not a fixed month
    cover_m = (U["swe_mm"] >= tp.cover_swe_threshold_mm).astype(np.float64)
    late_cover = cover_m[warm_month[u_hex], np.arange(n_units)]
    zero_level = th.zero_degree_level(t_coarse, z_ref_hex[None, :], lapse)
    zero_level = np.where(near_land[None, :], zero_level, np.nan)
    temp_mean = [hmean(U["temperature_c"][m]) for m in range(12)]
    monthly = {
        "topo_temperature_c": np.stack([grid(t) for t in temp_mean]),
        "topo_delta_temperature_c": np.stack([grid(temp_mean[m] - t_coarse[m]) for m in range(12)]),
        "topo_radiation_mj_m2_day": np.stack(
            [grid(hmean(U["radiation_mj_m2_day"][m])) for m in range(12)]
        ),
        "topo_swe_mm": np.stack([grid(hmean(U["swe_mm"][m])) for m in range(12)]),
        "topo_snow_cover_frac": np.stack(
            [grid(hmean(U["swe_mm"][m] >= tp.cover_swe_threshold_mm)) for m in range(12)]
        ),
        "topo_zero_degree_level_m": np.stack([grid(zero_level[m]) for m in range(12)]),
        "topo_snowline_m": np.stack([grid(snowline[m]) for m in range(12)]),
    }
    runoff_year = U["runoff_mm"].sum(axis=0, dtype=np.float64)
    layers = {
        "topo_dry_fraction": grid(dry_w),
        "topo_lake_fraction": grid(lake_w),
        "topo_ocean_fraction": grid(ocean_w),
        "topo_unit_count": grid(np.diff(offsets), np.int16),
        "hyps_z_mean_m": grid(zmean),
        "hyps_z_std_m": grid(zstd),
        "hyps_q05_m": grid(quant[0.05]),
        "hyps_q25_m": grid(quant[0.25]),
        "hyps_q50_m": grid(quant[0.5]),
        "hyps_q75_m": grid(quant[0.75]),
        "hyps_q95_m": grid(quant[0.95]),
        "hyps_relief_m": grid(np.where(has, relief, np.nan)),
        "dominant_thermal_belt": grid(dominant, np.uint8),
        "thermal_transition_fraction": grid(hmean(np.abs(np.nan_to_num(margin)) < 0.5)),
        "thermal_margin_mean": grid(hmean(np.clip(np.nan_to_num(margin), -10.0, 10.0))),
        "thermal_tree_score_mean": grid(hmean(S["thermal_tree_score"])),
        "site_tree_score_mean": grid(hmean(S["site_tree_score"])),
        "gsl_days_mean": grid(hmean(S["gsl_t_days"])),
        "gst_c_mean": grid(hmean(S["gst_t_c"], np.nan_to_num(S["gsl_t_days"]) > 0)),
        "gdd5_mean": grid(hmean(S["gdd5"])),
        "treeline_proxy_m": grid(tree_z),
        "treeline_state": grid(tree_state, np.uint8),
        "perennial_snow_fraction": grid(hmean(perennial)),
        "warm_season_snow_cover_frac": grid(hmean(late_cover)),
        "snowline_patchy": grid(patchy, np.bool_),
        "permafrost_potential_mean": grid(hmean(S["permafrost_potential"])),
        "snow_surplus_mm_yr": grid(hmean(S["snow_surplus_mm_yr"])),
        "cap_potential_max": grid(_hex_max(S["cap_static"], u_hex, n_hex, dry_u)),
        "cap_temperature_jan_c": grid(hmean(S["cap_jan_c"])),
        "svf_mean": grid(hmean(svf_slope)),
        "wind_exposure_mean": grid(hmean(np.maximum(-sx_up, 0.0))),
        "cloud_potential_max": grid(_hex_max(S["cloud_potential_max"], u_hex, n_hex, dry_u)),
        "fog_potential_mean": grid(hmean(S["fog_potential_mean"])),
        "precip_factor_max": grid(_hex_max(S["precip_factor"], u_hex, n_hex, dry_u)),
        "topo_local_runoff_mm_yr": grid(hmean(runoff_year)),
    }
    for b, name in enumerate(th.BELTS):
        if b:
            layers[f"belt_share_{name.replace('thermal_', '')}"] = grid(belt_share[b])

    massifs = []
    for mid in np.unique(H["massif"][H["massif"] >= 0]).tolist():
        inside = (H["massif"] == mid) & has
        if not inside.any():
            continue
        sel = inside & (tree_state == 1)
        vals = tree_z[sel]
        area = float((hex_w * inside).sum())

        def pct(q, v=vals):
            return round(float(np.percentile(v, q)), 1) if v.size else None

        massifs.append(
            {
                "massif_id": int(mid),
                "hexes": int(inside.sum()),
                "treeline_hexes": int(sel.sum()),
                "regional_treeline_m": pct(50),
                "treeline_q10_m": pct(10),
                "treeline_q90_m": pct(90),
                "coverage": round(float(sel.sum() / inside.sum()), 3),
                "warm_hexes": int((inside & (tree_state == 3)).sum()),
                "cold_hexes": int((inside & (tree_state == 2)).sum()),
                "highest_m": round(float(np.nanmax(quant[0.95][inside])), 1),
                "belt_shares": {
                    th.BELTS[b]: round(float((belt_frac[b] * inside).sum() / max(area, 1e-12)), 4)
                    for b in range(1, len(th.BELTS))
                },
            }
        )
    units = {
        "hex_offsets": offsets.astype(np.int64),
        "unit_hex": u_hex.astype(np.int32),
        "unit_weight_hex": u_w.astype(f32),
        "unit_weight_land": np.where(dry_u, u_w / np.maximum(dry_w[u_hex], 1e-12), 0.0).astype(f32),
        "unit_surface_type": u_type,
        "unit_landform": u_landform,
        "unit_code": u_code.astype(np.uint8),
        "x_km": u_x.astype(np.float64),
        "y_km": u_y.astype(np.float64),
        "z_surface_m": u_z.astype(f32),
        "slope_deg": np.degrees(u_slope).astype(f32),
        "aspect_deg": np.degrees(u_aspect).astype(f32),
        "elevation_rank": u_rank.astype(f32),
        "valley_index": u_valley.astype(f32),
        "ridge_index": u_struct.astype(f32),
        "svf_slope": svf_slope.astype(f32),
        "svf_horizontal": svf_h.astype(f32),
        "visible_sky_fraction": vis.astype(f32),
        "upwind_slope_rad": sx_up.astype(f32),
        "horizon_rad": np.nan_to_num(hor_signed, nan=-np.pi / 2).astype(np.float16),
        "thermal_belt": belt,
        "perennial_snow": perennial,
        "spinup_converged": converged,
        "persistent_snow": persistent,
    }
    units.update({k: v.astype(f32) for k, v in S.items()})
    units.update(U)
    timings["v47_aggregate"] = round(time.perf_counter() - t0, 3)

    # -- budgets, summary, provenance ------------------------------------------------------------
    p_year = U["precip_mm"].sum(axis=0, dtype=np.float64)
    p_units = np.bincount(u_hex, weights=wdry * p_year, minlength=n_hex)
    p_hex = H["p_m"].sum(0)
    rel = np.abs(p_units[has] / hex_w[has] - p_hex[has]) / np.maximum(p_hex[has], 1.0)
    area_err = np.abs(dry_w + lake_w + ocean_w - 1.0)
    summary = {
        "units": int(n_units),
        "dry_units": int(dry_u.sum()),
        "samples": int(x.size),
        "sampled_hexes": int(hexes.size),
        "detail_hexes": int(mountain.size),
        "units_per_land_hex": round(float(dry_u.sum() / max(int(has.sum()), 1)), 2),
        "area_closure_max_error": float(area_err.max()) if area_err.size else 0.0,
        "precipitation_conservation_max_rel": float(rel.max()) if rel.size else 0.0,
        "water_budget_max_residual_mm": worst,
        "spinup_years": years,
        "spinup_converged_share": round(float(converged[dry_u].mean()), 4) if du.size else 1.0,
        "persistent_snow_units": int(persistent[dry_u].sum()),
        "belt_land_shares": {
            th.BELTS[b]: round(float(belt_frac[b].sum() / max(hex_w.sum(), 1e-12)), 4)
            for b in range(len(th.BELTS))
        },
        "treeline_hexes": int((tree_state == 1).sum()),
        "treeline_crossings": crossings_total,
        "regional_treelines_m": [
            m["regional_treeline_m"] for m in massifs if m["regional_treeline_m"] is not None
        ],
        "mean_abs_delta_temperature_c": round(
            float(np.nanmean(np.abs(monthly["topo_delta_temperature_c"]))), 3
        )
        if has.any()
        else 0.0,
        "max_precip_factor": round(float(np.nanmax(S["precip_factor"][dry_u])), 3)
        if du.size
        else 1.0,
        "cap_units_over_0_5": int((S["cap_static"] > 0.5).sum()),
        "condensation_unit_months": int(condensation_months),
    }
    provenance = {
        "temperature_reference_kind": "coarse_surface",
        "climate_reference_elevation": "v4 hex elevation (sea hexes: sea level)",
        "temperature_downscale_mode": tp.temperature_mode,
        "lapse_rate_k_km": lapse,
        "humidity_source": "synthetic_profile (rh_inland .. rh_marine by marine influence)",
        "wind_source": "v4 annual mean wind (no monthly winds); exposure from upwind Sx",
        "cloud_fraction": "cloud_fraction_proxy = f(P, PET) of the month [H]",
        "radiation": "Angstrom-Prescott from TOA; diffuse share from cloudiness; direct "
        "beam with horizon shading on the slope; diffuse x SVF (approx); ground reflection",
        "precipitation_mode": "redistribute_coarse (per hex, dry units, conserved)",
        "snow_redistribution": "statistical_storage_partition inside the hex",
        "pet": "v4.5 PET x temperature-index ratio x sqrt(radiation ratio) [H]",
        "water_kernel": "v4.5 water_balance on every dry unit (one soil model)",
        "phase_quality": "monthly_proxy",
        "horizon": f"approx_multires_horizon: own-hex samples + rays over hex 95% heights "
        f"to {tp.horizon_max_distance_km:g} km, {tp.horizon_directions} directions",
        "subgrid_quality": "sampled with the v3 mountain plan in massifs, coarse_only elsewhere",
        "units": "deterministic stratification (landform x elevation band x aspect), "
        "representative = the sample nearest to the unit mean",
        "thermal_daily": "synthetic mean-preserving daily curve from 12 monthly means",
        "belts": "thermal classes, not biomes",
        "mass_elevation_effect_c": tp.mass_elevation_effect_max_c,
        "frost_free_proxy": "min(daily mean T) > 5 degC (no daily Tmin): flagged proxy",
        "snowline": "bracket between snow-free and snow-covered units, month_end_proxy",
        "permafrost": "potential only: Nelson-Outcalt frost number, snow insulation [H]",
        "algorithm_version": TOPOCLIMATE_ALGORITHM_VERSION,
    }
    timings["v47_total"] = round(sum(timings.values()), 3)
    return {
        "layers": layers,
        "monthly": monthly,
        "units": units,
        "summary": summary,
        "massifs": massifs,
        "provenance": provenance,
        "timings": timings,
    }


def upstream_keys(project) -> dict:
    """Stage keys of every input of v4.7 (surface, climate, soils, mountain plan)."""
    return {
        "env": project.env.stage_keys,
        "terrain": project.terrain.stage_keys if project.terrain is not None else {},
        "hydro": project.hydro.stage_keys,
        "hist": project.hist.stage_keys,
        "mountains": project.config.mountains.__dict__ if project.config else {},
    }


def run_v47(config, geometry, project, old_keys: dict, old_cache: dict, report):
    """Stage wrapper with caching: returns ``(TopoData, cache, reused)``."""
    keys = topo_stage_keys(config, upstream_keys(project))
    if (
        old_keys.get("v47_topoclimate") == keys["v47_topoclimate"]
        and "v47_topoclimate" in old_cache
    ):
        data = old_cache["v47_topoclimate"]
        return data, {"v47_topoclimate": data}, ["v47_topoclimate"]
    report("v47_topoclimate", 0.9975)
    from .progress import sub_report

    res = run_topoclimate(
        config, geometry, project, report=lambda f: report(sub_report("v47_topoclimate", f), 0.9975)
    )
    warnings = []
    s = res["summary"]
    if s["precipitation_conservation_max_rel"] > 1e-6:
        warnings.append(
            "v4.7: precipitation redistribution is not conserved in some hexes "
            f"(max rel. error {s['precipitation_conservation_max_rel']:.2e})"
        )
    if s["spinup_converged_share"] < 0.99:
        warnings.append(
            f"v4.7: the water balance spin-up converged on {s['spinup_converged_share']:.1%} "
            "of the dry terrain units only"
        )
    data = TopoData(
        layers=res["layers"],
        monthly=res["monthly"],
        units=res["units"],
        summary=s,
        massifs=res["massifs"],
        provenance=res["provenance"],
        stage_keys=keys,
        warnings=warnings,
        timings=res["timings"],
    )
    return data, {"v47_topoclimate": data}, []
