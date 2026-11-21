"""v4.5 stage ``environment``: the ecological foundation on top of the final v4 state.

Inputs (never modified): the v4 surface after erosion, its climate of four seasons, rivers,
lakes and basins; rock, sediment and volcanism of the v3 epoch; the LGM ice estimate of
the coasts. Outputs: :class:`~hexworld.project.EnvData` - 12-month fields, soils, the monthly
water balance, wetness on the flow graph, bioclimatic proxies, site flags, provenance.

River discharge is **not** recomputed here: the v4 network stays the state of the rivers
(``summary.river_q_source``) and the difference between the v4 annual runoff and the
v4.5 monthly runoff is reported, not hidden.
"""

from __future__ import annotations

import hashlib
import json
import math
import time
from dataclasses import asdict

import numpy as np
from scipy import ndimage

from .noise import FractalNoise
from .project import EnvData
from .rng import Stream
from .stages import environment as env
from .stages import raster as raster_stage
from .version import ENV_ALGORITHM_VERSION

#: bits of ``environment_feature_flags``
FLAGS = {
    "river": 1,
    "lake_shore": 2,
    "floodplain_potential": 4,
    "waterlogged": 8,
    "coastal_marine": 16,
    "rocky": 32,
    "steep": 64,
    "young_volcanic": 128,
    "glacial_history": 256,
    "permafrost_potential": 512,
    "permanent_ice": 1024,
    "saline_potential": 2048,
    "endorheic": 4096,
}
#: bits of ``quality_flags``
QUALITY = {
    "approximate_regional": 1,  # the basin leaves the window: upstream context incomplete
    "bare_rock": 2,  # no soil: a few mm of surface storage stand in for the AWC
    "persistent_snow": 4,  # snow accumulates every year: no periodic state, no ice flow
    "spinup_not_converged": 8,
    "pedotransfer_fallback": 16,  # Saxton-Rawls out of range: texture-table FC / PWP
}
MONTHLY_FIELDS = {
    "temperature_monthly_c": ("degC", "monthly mean air temperature"),
    "precip_monthly_mm": ("mm/month", "precipitation (rain + snow water equivalent)"),
    "pet_monthly_mm": ("mm/month", "reference potential evapotranspiration"),
    "aet_monthly_mm": ("mm/month", "actual evapotranspiration of the reference cover"),
    "snowpack_monthly_swe_mm": ("mm", "snow water equivalent at the end of the month"),
    "soil_storage_monthly_mm": ("mm", "plant-available water at the end of the month"),
    "runoff_monthly_mm": ("mm/month", "local runoff (quick flow + baseflow), not river Q"),
    "groundwater_recharge_mm": ("mm/month", "recharge of the model groundwater store"),
}


#: units and meaning of every v4.5 hex field (documentation, manifest, DATA_FORMAT)
ENV_LAYER_SPECS: dict[str, tuple[str, str]] = {
    "soil_present": ("bool", "soil on this land hex (not bare rock, ice or water)"),
    "parent_material_class": ("enum", "parent material, stages.environment.PARENT_MATERIALS"),
    "sand_frac": ("0..1", "sand share of the fine earth; NaN where no soil"),
    "silt_frac": ("0..1", "silt share of the fine earth; NaN where no soil"),
    "clay_frac": ("0..1", "clay share of the fine earth; NaN where no soil"),
    "coarse_fragment_frac": ("0..1", "stones by volume"),
    "soil_depth_m": ("m", "soil thickness (static potential, approximate)"),
    "soil_organic_potential": ("0..1", "organic matter potential (estimate)"),
    "field_capacity_vol": ("m3/m3", "FC at -33 kPa, Saxton & Rawls 2006"),
    "wilting_point_vol": ("m3/m3", "PWP at -1500 kPa, Saxton & Rawls 2006"),
    "porosity_vol": ("m3/m3", "saturated water content"),
    "awc_mm": ("mm", "available water in the reference root zone (0 = no soil)"),
    "ks_mm_h": ("mm/h", "saturated hydraulic conductivity"),
    "infiltration_capacity_mm_day": ("mm/day", "daily intake limit"),
    "terrain_rocky_fraction": ("0..1", "bare rock share of the surface"),
    "soil_salinity_proxy": ("0/1", "potential salinity (dry closed basins, low coasts)"),
    "drainage_class": ("enum", "0 excessively .. 5 very poorly, 255 water"),
    "twi": ("ln m", "topographic wetness index ln(a / tan b), grid-dependent"),
    "twi_anomaly": ("ln", "TWI minus that of one hex on a 1 % slope"),
    "hand_m": ("m", "height above the first drainage cell along the flow path"),
    "wetness_potential": ("0..1", "potential waterlogging (proxy)"),
    "floodplain_potential": ("0..1", "share of the hex a floodplain can cover (potential)"),
    "waterlogging_months": ("months", "months with the soil saturated (proxy)"),
    "slope_deg": ("deg", "mean hex-scale slope"),
    "aspect_deg": ("deg", "downslope direction, 0 = north, clockwise; NaN on flats"),
    "coastal_marine_influence": ("0..1", "exp(-distance to the sea / 100 km)"),
    "lake_shore_influence": ("0..1", "exp(-distance to a lake / 30 km)"),
    "solar_radiation_toa_mj_day": ("MJ/m2/day", "annual mean extraterrestrial radiation"),
    "day_length_range_h": ("h", "longest minus shortest day"),
    "local_runoff_annual_mm": ("mm/yr", "sum of the monthly local runoff"),
    "runoff_difference_vs_v4_mm": ("mm/yr", "v4.5 monthly runoff minus v4 runoff"),
    "groundwater_recharge_annual_mm": ("mm/yr", "recharge of the model groundwater store"),
    "groundwater_storage_proxy_mm": ("mm", "mean model groundwater store (not a water table)"),
    "persistent_snow_accumulation_mm_yr": ("mm/yr", "snow left every year (no ice flow)"),
    "permanent_ice_mask": ("bool", "perennial snow accumulation on ground < 60 m/km"),
    "environment_feature_flags": ("bitset", "site flags, pipeline_v45.FLAGS"),
    "quality_flags": ("bitset", "quality, pipeline_v45.QUALITY"),
    "gdd5_proxy_c_day": ("degC day", "growing degree days > 5 degC from monthly means"),
    "freezing_degree_days": ("degC day", "sum of negative monthly means x days"),
    "thawing_degree_days": ("degC day", "sum of positive monthly means x days"),
    "frost_number": ("0..1", "air frost number (Nelson & Outcalt 1987)"),
    "permafrost_potential": ("0..1", "permafrost potential from the frost number"),
    "warmest_month_c": ("degC", "warmest monthly mean"),
    "coldest_month_c": ("degC", "coldest monthly mean"),
    "annual_temperature_range_c": ("degC", "warmest minus coldest month"),
    "cold_month_count": ("months", "months below 0 degC"),
    "dry_month_count": ("months", "months with P < 0.5 PET"),
    "growing_month_count": ("months", "months >= 5 degC with AET >= 0.5 PET"),
    "precip_seasonality_cv": ("%", "coefficient of variation of monthly P"),
    "aridity_p_over_pet": ("P/PET", "annual aridity index from 12 months"),
    "aet_over_pet": ("AET/PET", "annual water supply of the reference cover"),
    "snow_cover_months": ("months", "months ending with > 10 mm of snow"),
    "annual_max_swe_mm": ("mm", "largest end-of-month snowpack"),
}


def _key(*parts) -> str:
    h = hashlib.sha256()
    for p in parts:
        h.update(json.dumps(p, sort_keys=True, default=str).encode())
    return h.hexdigest()[:16]


def env_stage_keys(config, hydro_keys: dict) -> dict:
    return {
        "v45_environment": _key(
            "v45_environment",
            ENV_ALGORITHM_VERSION,
            hydro_keys.get("v4_network"),
            asdict(config.environment),
            config.climate.deep_infiltration,
        )
    }


def seasonal_inputs(config, project) -> dict:
    """The four seasons of the final v4 climate (from the v4 cache, else recomputed)."""
    net = project.cache.get("v4_network")
    clim = net.get("climate") if isinstance(net, dict) else None
    seasons = getattr(clim, "seasons", None) if clim is not None else None
    if seasons:
        return seasons
    from .stages.climate import hex_climate, resolve_latitude, world_climate

    hist, hydro = project.hist, project.hydro
    entry = project.cache.get("v4_climate")
    wc = entry.get("world") if isinstance(entry, dict) else None
    if wc is None:
        latitude = resolve_latitude(config.climate, project.geometry.simulation_bounds)
        wc = world_climate(
            config.climate, hist.world["grid"], hist.world["elevation_m"], hist.sea_level_m,
            latitude,
        )  # fmt: skip
    z = hydro.layers["elevation_m"].astype(np.float64)
    lakes = hydro.layers["lake_id"] >= 0
    return hex_climate(config.climate, wc, project.layout, z, hydro.sea_level_m, lakes).seasons


def _neighbour_any(mask, layout):
    out = mask.copy()
    for d in range(6):
        nb, valid = raster_stage.neighbor_index(layout, d)
        out |= np.where(valid, mask.ravel()[nb.ravel()].reshape(mask.shape), False)
    return out


def _distance_km(mask, layout):
    if not mask.any():
        return np.full(mask.shape, 1e6)
    return ndimage.distance_transform_edt(~mask) * layout.spacing_km


def run_env(config, geometry, project, old_keys: dict, old_cache: dict, report):
    hydro, hist = project.hydro, project.hist
    keys = env_stage_keys(config, hydro.stage_keys)
    if (
        old_keys.get("v45_environment") == keys["v45_environment"]
        and "v45_environment" in old_cache
    ):
        data = old_cache["v45_environment"]
        return data, {"v45_environment": data}, ["v45_environment"]
    timings: dict[str, float] = {}
    t0 = time.perf_counter()
    report("v45_months", 0.9965)
    layout = geometry.layout
    shape = (layout.height, layout.width)
    ep = config.environment
    L, H = hydro.layers, hist.layers
    sea_level = float(hydro.sea_level_m)
    z = L["elevation_m"].astype(np.float64)
    land = L["land_mask"].astype(bool)
    lake = L["lake_id"] >= 0
    lat = L["latitude_deg"].astype(np.float64)

    # -- Gate 1: twelve months --------------------------------------------------------------
    seasons = seasonal_inputs(config, project)
    t_m = env.monthly_temperature(seasons["t"], L["temperature_mean_c"])
    p_m = env.monthly_precipitation(seasons["p"])
    pet_m = env.monthly_pet(seasons["pet"], t_m, lat)
    doy = env.MONTH_MID_DOY.reshape(12, 1, 1)
    ra_m, day_len = env.extraterrestrial_radiation(lat[None], doy)
    days = env.MONTH_DAYS.reshape(12, 1, 1)
    t_weighted = (t_m * days).sum(axis=0) / env.YEAR_DAYS
    p_year = seasons["p"].astype(np.float64).sum(axis=0)
    p_check = float((np.abs(p_m.sum(axis=0) - p_year) / np.maximum(p_year, 1.0)).max())
    t_check = float(np.abs(t_weighted - L["temperature_mean_c"]).max())
    timings["v45_months"] = round(time.perf_counter() - t0, 4)

    # -- Gate 2: soils ------------------------------------------------------------------------
    t0 = time.perf_counter()
    report("v45_soils", 0.9972)
    hx, hy = layout.centers()
    noise = np.stack(
        [
            FractalNoise(Stream(config.seed, f"environment/4.5/texture/{k}"), [900.0, 300.0],
                         [1.0, 0.5])(hx, hy) / 1.5
            for k in range(3)
        ]
    )  # fmt: skip
    coast = land & _neighbour_any(~land & ~lake, layout)
    lake_shore = land & _neighbour_any(lake, layout)
    slope = L["slope_m_per_km"].astype(np.float64)
    humidity = L["humidity_index"].astype(np.float64)
    inp = env.SoilInputs(
        land=land & ~lake,
        rock_type=H["rock_type"],
        sediment_m=L["sediment_thickness_m"].astype(np.float64),
        deposition_m=L["v4_deposition_m"].astype(np.float64),
        slope_m_km=slope,
        elevation_above_sea_m=np.maximum(z - sea_level, 0.0),
        t_annual=L["temperature_mean_c"].astype(np.float64),
        humidity=humidity,
        river=L["river_mask"].astype(bool),
        discharge_m3_s=L["discharge_m3_s"].astype(np.float64),
        lake_shore=lake_shore | lake,
        coastal=coast,
        lgm_ice=L["lgm_ice_mask"].astype(bool),
        volcanic_age_ma=H["volcanic_age_ma"].astype(np.float64),
        endorheic=L["endorheic_mask"].astype(bool),
        glacial_now=L["glacial_potential"].astype(np.float64),
        texture_noise=noise,
    )
    pm = env.parent_material(inp)
    soil = env.soil_properties(inp, pm, ep)
    rejected = soil.pop("_pedotransfer_rejected")
    if rejected.any():  # texture-table fallback (loam-like values) where the PTF fails
        soil["field_capacity_vol"][rejected] = 0.27
        soil["wilting_point_vol"][rejected] = 0.13
        awc = env.available_water(
            soil["soil_depth_m"][rejected], 0.27, 0.13,
            soil["coarse_fragment_frac"][rejected], ep.reference_root_depth_m,
        )  # fmt: skip
        soil["awc_mm"][rejected] = awc.astype(np.float32)
    timings["v45_soils"] = round(time.perf_counter() - t0, 4)

    # -- Gate 3: monthly water -----------------------------------------------------------------
    t0 = time.perf_counter()
    report("v45_water", 0.998)
    wet_land = (land & ~lake).ravel()
    present = soil["soil_present"].ravel()
    awc = np.nan_to_num(soil["awc_mm"].ravel().astype(np.float64))
    ks = np.nan_to_num(soil["ks_mm_h"].ravel().astype(np.float64), nan=0.5)
    icap = np.where(present, np.nan_to_num(soil["infiltration_capacity_mm_day"].ravel()), 5.0)
    drainage = np.where(present, np.clip(0.03 + 0.2 * np.log10(1.0 + ks), 0.02, 0.6), 0.6)
    wp = env.WaterParams(
        k_dd_mm=ep.degree_day_mm,
        depletion_p=ep.depletion_fraction,
        rain_days=ep.rain_days_per_month,
        baseflow_timescale_yr=ep.baseflow_timescale_yr,
        deep_loss_share=config.climate.deep_infiltration,
        max_years=ep.spinup_max_years,
        tolerance_mm=ep.water_balance_tolerance_mm,
    )
    idx = np.flatnonzero(wet_land)
    flat = lambda a: a.reshape(12, -1)[:, idx]  # noqa: E731
    wb = env.water_balance(
        flat(p_m), flat(t_m), flat(pet_m), awc[idx], icap[idx], drainage[idx], wp
    )

    def to_grid(values, fill=np.nan):
        out = np.full(z.size, fill, dtype=np.float64)
        out[idx] = values
        return out.reshape(shape)

    def to_grid12(values):
        out = np.full((12, z.size), np.nan, dtype=np.float64)
        out[:, idx] = values
        return out.reshape((12,) + shape)

    aet_m = to_grid12(wb.monthly["aet"])
    swe_m = to_grid12(wb.monthly["snowpack"])
    s_m = to_grid12(wb.monthly["soil_storage"])
    runoff_m = to_grid12(wb.monthly["runoff"])
    recharge_m = to_grid12(wb.monthly["recharge"])
    liquid_m = to_grid12(wb.monthly["rain"] + wb.monthly["melt"])
    runoff_year = to_grid(wb.annual["runoff"])
    v4_runoff = L["runoff_mm_yr"].astype(np.float64)
    timings["v45_water"] = round(time.perf_counter() - t0, 4)

    # -- Gate 4: wetness on the flow graph and indices -------------------------------------------
    t0 = time.perf_counter()
    report("v45_wetness", 0.9988)
    river = L["river_mask"].astype(bool)
    twi_value, twi_anom = env.twi(L["flow_accumulation_area_km2"], slope, layout.spacing_km)
    reference = np.where(
        river, z, np.where(lake, L["lake_level_m"].astype(np.float64),
                           np.where(~land, sea_level, np.nan))
    )  # fmt: skip
    rc = L["flow_receiver"]
    hand_m = env.hand(z, rc, reference).reshape(shape)
    steps_to_river = env.path_steps(rc, river.ravel(), limit=8).reshape(shape)
    # a saturated month: the soil bucket overflowed while the water supply exceeded PET
    saturated_m = (to_grid12(wb.monthly["saturation_excess"]) > 1.0) & (liquid_m > pet_m)
    surplus_months = saturated_m.sum(axis=0).astype(np.float64)
    drain_grid = drainage.reshape(shape)
    # centred on a valley-floor hex: 10 hexes upstream on a 0.2 % slope (TWI anomaly 3.9)
    zw = (
        0.9 * (twi_anom.reshape(shape) - 3.9)
        - 0.35 * hand_m / 10.0
        + 3.0 * surplus_months / 12.0
        - 4.0 * (drain_grid - 0.2)
    )
    wetness = np.where(land & ~lake, env.sigmoid(zw), np.nan)
    q_ref = L["discharge_m3_s"].astype(np.float64)
    # share of the hex a floodplain can cover: ~20 channel widths (W = 5 Q^0.5) across a hex
    # of ``spacing`` km; neighbours of the river only where they lie low above it
    from .stages.hydrology import hydraulic_geometry

    width_m, _depth, _v = hydraulic_geometry(q_ref)
    strip = np.clip(20.0 * width_m / (layout.spacing_km * 1000.0), 0.0, 1.0)
    path_q = np.where(river, q_ref, 0.0)
    near_q = np.where(steps_to_river == 1, q_ref, 0.0)  # the river it drains into is larger
    near_w, _d2, _v2 = hydraulic_geometry(near_q)
    near_strip = np.clip(10.0 * near_w / (layout.spacing_km * 1000.0), 0.0, 1.0)
    flood = np.where(
        steps_to_river == 0,
        strip * (path_q > 0),
        np.where(steps_to_river == 1, near_strip * env.sigmoid((5.0 - hand_m) / 2.0), 0.0),
    )
    flood = flood * (slope < 10.0)
    flood = np.where(land & ~lake, flood, np.nan)
    waterlog = np.where(
        land & ~lake,
        (saturated_m & ((np.nan_to_num(wetness) > 0.55) | (drain_grid < 0.12))[None]).sum(axis=0),
        0,
    ).astype(np.uint8)
    ks_grid = np.nan_to_num(soil["ks_mm_h"], nan=0.5)
    drainage_class = np.select(
        [waterlog >= 6, waterlog >= 3, (waterlog >= 1) | (np.nan_to_num(wetness) > 0.6),
         ks_grid < 5.0, ks_grid < 50.0],
        [5, 4, 3, 2, 1],
        default=0,
    ).astype(np.uint8)  # fmt: skip
    drainage_class = np.where(land & ~lake, drainage_class, 255).astype(np.uint8)
    idx_bio = env.bioclimatic(
        t_m, p_m, pet_m, np.nan_to_num(aet_m), np.nan_to_num(swe_m), ep.dry_month_ratio,
        ep.gdd_base_c,
    )  # fmt: skip
    persistent = to_grid(wb.persistent_snow.astype(np.float64), 0.0) > 0.5
    permanent_ice = persistent & (slope < 60.0)
    from .views import hex_gradient

    gx, gy = hex_gradient(z, layout)
    aspect = (np.degrees(np.arctan2(-gx, gy)) + 360.0) % 360.0
    aspect = np.where(np.hypot(gx, gy) > 1e-3, aspect, np.nan)
    sea_mask = ~land & ~lake
    coastal_marine = np.exp(-_distance_km(sea_mask, layout) / 100.0)
    lake_influence = np.exp(-_distance_km(lake, layout) / 30.0)
    young_volc = np.nan_to_num(H["volcanic_age_ma"].astype(np.float64), nan=999.0) < 2.0
    flags = np.zeros(shape, dtype=np.uint16)
    for name, mask in (
        ("river", river),
        ("lake_shore", lake_shore),
        ("floodplain_potential", np.nan_to_num(flood) > 0.5),
        ("waterlogged", waterlog >= 2),
        ("coastal_marine", coast),
        ("rocky", np.nan_to_num(soil["terrain_rocky_fraction"]) > 0.5),
        ("steep", slope > 25.0),
        ("young_volcanic", young_volc),
        ("glacial_history", L["lgm_ice_mask"].astype(bool)),
        ("permafrost_potential", idx_bio["permafrost_potential"] > 0.5),
        ("permanent_ice", permanent_ice),
        ("saline_potential", soil["soil_salinity_proxy"] > 0),
        ("endorheic", L["endorheic_mask"].astype(bool)),
    ):
        flags |= np.where(mask & land, FLAGS[name], 0).astype(np.uint16)
    basin_edge = np.zeros(shape, dtype=bool)
    for b in hydro.basins:
        if b["terminal"] == "window_edge":
            basin_edge |= L["basin_id"] == b["id"]
    inflow_reach = L["external_inflow_m3_s"] > 0
    quality = np.zeros(shape, dtype=np.uint16)
    for name, mask in (
        ("approximate_regional", basin_edge | inflow_reach),
        ("bare_rock", land & ~lake & ~soil["soil_present"]),
        ("persistent_snow", persistent),
        ("spinup_not_converged", to_grid((~wb.converged).astype(np.float64), 0.0) > 0.5),
        ("pedotransfer_fallback", rejected),
    ):
        quality |= np.where(mask, QUALITY[name], 0).astype(np.uint16)
    timings["v45_wetness"] = round(time.perf_counter() - t0, 4)

    def land_only(a, dtype=np.float32):
        a = np.asarray(a)
        if np.issubdtype(np.dtype(dtype), np.floating):
            return np.where(land & ~lake, a, np.nan).astype(dtype)
        return np.where(land & ~lake, a, 0).astype(dtype)

    layers = {
        **soil,
        "drainage_class": drainage_class,
        "twi": land_only(twi_value.reshape(shape)),
        "twi_anomaly": land_only(twi_anom.reshape(shape)),
        "hand_m": land_only(hand_m),
        "wetness_potential": wetness.astype(np.float32),
        "floodplain_potential": flood.astype(np.float32),
        "waterlogging_months": waterlog,
        "slope_deg": land_only(np.degrees(np.arctan(slope / 1000.0))),
        "aspect_deg": land_only(aspect),
        "coastal_marine_influence": land_only(coastal_marine),
        "lake_shore_influence": land_only(lake_influence),
        "solar_radiation_toa_mj_day": ra_m.mean(axis=0).astype(np.float32),
        "day_length_range_h": (day_len.max(axis=0) - day_len.min(axis=0)).astype(np.float32),
        "local_runoff_annual_mm": land_only(runoff_year),
        "runoff_difference_vs_v4_mm": land_only(runoff_year - v4_runoff),
        "groundwater_recharge_annual_mm": land_only(to_grid(wb.annual["recharge"])),
        "groundwater_storage_proxy_mm": land_only(to_grid(wb.monthly["groundwater"].mean(axis=0))),
        "persistent_snow_accumulation_mm_yr": land_only(
            to_grid(wb.annual["snow_accumulation"], 0.0)
        ),
        "permanent_ice_mask": permanent_ice & land,
        "environment_feature_flags": flags,
        "quality_flags": quality,
    }
    for name, values in idx_bio.items():
        if name in ("cold_month_count", "dry_month_count", "growing_month_count",
                    "snow_cover_months"):  # fmt: skip
            layers[name] = land_only(values, np.uint8)
        elif name in ("aridity_p_over_pet",):
            layers[name] = land_only(values)
        else:
            layers[name] = land_only(values)
    monthly = {
        "temperature_monthly_c": t_m.astype(np.float32),
        "precip_monthly_mm": p_m.astype(np.float32),
        "pet_monthly_mm": pet_m.astype(np.float32),
        "aet_monthly_mm": aet_m.astype(np.float32),
        "snowpack_monthly_swe_mm": swe_m.astype(np.float32),
        "soil_storage_monthly_mm": s_m.astype(np.float32),
        "runoff_monthly_mm": runoff_m.astype(np.float32),
        "groundwater_recharge_mm": recharge_m.astype(np.float32),
    }

    # -- summary, provenance --------------------------------------------------------------------
    lw = land & ~lake
    n_land = max(int(lw.sum()), 1)

    def share(mask):
        return round(float((mask & lw).sum()) / n_land, 4)

    rel_diff = float(
        np.abs(np.nansum(runoff_year[lw]) - float(v4_runoff[lw].sum()))
        / max(float(v4_runoff[lw].sum()), 1e-9)
    )
    summary = {
        "calendar": {
            "year_days": int(env.YEAR_DAYS),
            "month_lengths": env.MONTH_DAYS.astype(int).tolist(),
        },
        "month_check": {
            "max_rel_precip_sum_error": p_check,
            "max_abs_annual_temperature_error_c": t_check,
        },
        "water_balance": {
            **wb.budget,
            "spinup_years": wb.years,
            "converged_share": round(float(wb.converged.mean()) if wb.converged.size else 1.0, 4),
            "persistent_snow_share": share(persistent),
            "max_abs_residual_mm": wb.max_residual_mm,
            "tolerance_mm": ep.water_balance_tolerance_mm,
        },
        "runoff_vs_v4": {
            "v4_land_mean_mm_yr": float(v4_runoff[lw].mean()) if lw.any() else None,
            "v45_land_mean_mm_yr": float(np.nanmean(runoff_year[lw])) if lw.any() else None,
            "relative_difference": round(rel_diff, 4),
            "reasons": "monthly instead of seasonal steps, soil-specific AWC and infiltration "
            "limits, snow kept as accumulation instead of glacier discharge",
        },
        "river_q_source": "v4 snapshot: rivers follow the v4 annual runoff, not recomputed",
        "parent_material_shares": {
            name: share(layers["parent_material_class"] == k)
            for k, name in enumerate(env.PARENT_MATERIALS)
            if k > 0
        },
        "drainage_class_shares": {
            name: share(drainage_class == k) for k, name in enumerate(env.DRAINAGE_CLASSES)
        },
        "soil_present_share": share(soil["soil_present"]),
        "mean_awc_mm": float(np.nanmean(soil["awc_mm"][lw])) if lw.any() else None,
        "floodplain_area_share": round(float(np.nansum(flood[lw])) / n_land, 4),
        "waterlogged_share": share(waterlog >= 2),
        "permafrost_potential_share": share(idx_bio["permafrost_potential"] > 0.5),
        "permanent_ice_share": share(permanent_ice),
        "mean_dry_months": float(idx_bio["dry_month_count"][lw].mean()) if lw.any() else None,
        "mean_gdd5": float(idx_bio["gdd5_proxy_c_day"][lw].mean()) if lw.any() else None,
        "pedotransfer_fallback_hexes": int(rejected.sum()),
        "approximate_regional_share": share((quality & 1) > 0),
        "land_fraction_note": "hexes are wholly land or sea (no sub-hex coastline yet): "
        "land_fraction is 1 on land",
    }
    methods = {
        "months": {
            "method": "derived_from_seasonal_physics",
            "detail": "temperature: first harmonic of the four v4 season means, day-weighted to "
            "the v4 annual mean; precipitation: the v4 seasonal moisture-transport totals, "
            "spread inside each season by a periodic curve through the season rates",
            "confidence": "medium",
        },
        "pet": {
            "method": "v4_temperature_index_monthly",
            "detail": "the v4 seasonal PET (a temperature-index heuristic, not Penman-Monteith) "
            "spread over months by temperature and FAO-56 day length (Thornthwaite)",
            "confidence": "low-medium",
        },
        "radiation": {"method": "FAO-56 eq. 21-25", "confidence": "high (geometry)"},
        "soil_texture": {
            "method": "parent-material table + weathering + smooth world noise",
            "confidence": "approximate",
        },
        "soil_hydraulics": {"method": "saxton_rawls_2006", "fallback": "texture_table"},
        "soil_depth": {
            "method": "static potential f(slope, weathering, material)",
            "confidence": "approximate",
        },  # fmt: skip
        "water_balance": {"method": "monthly bucket, TZ 4.5 section 6.2, periodic spin-up"},
        "twi": {"method": "ln(a / tan b), a = area / hex edge; grid-dependent"},
        "hand": {"method": "height above the first drainage cell along the D6 flow path"},
        "frost": {"method": "monthly proxies; frost_days_exact = not_calculated"},
        "permafrost": {"method": "air frost number (Nelson & Outcalt 1987), potential only"},
        "reference_surface_profile": "reference low cover, no vegetation feedback",
    }
    warnings = []
    if wb.max_residual_mm > max(ep.water_balance_tolerance_mm, 1e-6):
        warnings.append(f"The monthly water balance does not close: {wb.max_residual_mm:.2e} mm")
    data = EnvData(
        layers=layers,
        monthly=monthly,
        summary=summary,
        methods=methods,
        stage_keys=keys,
        warnings=warnings,
        timings=timings,
    )
    return data, {"v45_environment": data}, []


__all__ = [
    "FLAGS",
    "MONTHLY_FIELDS",
    "QUALITY",
    "env_stage_keys",
    "run_env",
    "seasonal_inputs",
    "math",
]
