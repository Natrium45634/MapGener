"""HexWorld v5.3a: the Earth system on the hexes (stage ``v53_earth``, after v4.5).

The ocean, the sea ice and the winds aloft are computed with the climate on the world
raster (``hexworld.earth.coupler``, inside the v4 climate stage, so the moisture budget,
the hex climate and the rivers already feel them). This stage samples their state to the
hexes, adds the lake ice and the realm / cover axes, and keeps the coupler report, so that
a saved project carries them (the world raster itself is not saved).

Realm and cover are separate axes (TZ 5.3 sec. 2): realm 1 land, 2 inland water,
3 ocean; ``ice_cover`` 0 none, 1 seasonal sea ice, 2 multiyear sea ice, 3 fast ice,
4 lake ice. A frozen lake is ``inland_water`` + ``lake_ice``, never sea ice.
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import asdict

import numpy as np
from scipy import ndimage

from .project import EarthData
from .version import EARTH_ALGORITHM_VERSION

QUALITY = {
    "sst_monthly_c": "CALIBRATED_APPROXIMATION",
    "sea_ice_concentration_monthly": "CALIBRATED_APPROXIMATION",
    "current_u_ms": "CALIBRATED_APPROXIMATION",
    "winds_aloft_u_ms": "CALIBRATED_APPROXIMATION",
    "pet": "CALIBRATED_APPROXIMATION",
    "salinity": "SYNTHETIC_PROXY",
    "lake_ice_months": "SYNTHETIC_PROXY",
}


def _key(*parts) -> str:
    return hashlib.sha256(json.dumps(parts, sort_keys=True, default=str).encode()).hexdigest()


def earth_stage_keys(config, project) -> dict:
    return {"v53_earth": _key("v53", EARTH_ALGORITHM_VERSION, asdict(config.earth),
                              project.env.stage_keys if project.env else {})}  # fmt: skip


def _world_climate(config, project):
    entry = project.cache.get("v4_climate")
    wc = entry.get("world") if isinstance(entry, dict) else None
    if wc is not None and getattr(wc, "earth", None) is not None:
        return wc
    from .stages.climate import resolve_latitude, world_climate

    hist = project.hist
    latitude = resolve_latitude(config.climate, project.geometry.simulation_bounds)
    return world_climate(config.climate, hist.world["grid"], hist.world["elevation_m"],
                         hist.sea_level_m, latitude, earth=config.earth)  # fmt: skip


def _filled(field_, valid):
    """Copy of a raster field with every invalid cell taking its nearest valid value."""
    if valid.all() or not valid.any():
        return field_
    _d, (jj, ii) = ndimage.distance_transform_edt(~valid, return_indices=True)
    return field_[..., jj, ii]


STEP_YEARS = {"fast": 500.0, "balanced": 250.0, "high": 100.0}


def _ice_sheets(config, project, out, ocean_hex, shape):
    """Grounded sheets (spin-up to convergence), shelves, isostasy, sea-level equivalent."""
    from .earth.icesheet import (
        glacial_erosion_rate,
        isostatic_deflection,
        sea_level_equivalent_m,
        spinup,
    )
    from .earth.shelves import build_shelves
    from .pipeline_v50 import _hex_neighbours

    ep = config.earth
    lay = project.layout
    L = project.hydro.layers
    n = lay.width * lay.height
    nbr = _hex_neighbours(lay)
    ocean = ocean_hex.ravel()
    land = L["land_mask"].ravel().astype(bool) & (L["lake_id"].ravel() < 0) & ~ocean
    bed = (L["elevation_m"].astype(np.float64) - project.hydro.sea_level_m).ravel()
    M = project.env.monthly
    t_m = M["temperature_monthly_c"].reshape(12, n).astype(np.float64)
    p_m = np.nan_to_num(M["precip_monthly_mm"].reshape(12, n).astype(np.float64))
    pet = np.nan_to_num(M["pet_monthly_mm"].reshape(12, n).astype(np.float64))
    # sublimation: a share of the evaporative demand of the freezing months (as v5.0)
    sub = 0.12 * np.where(t_m < 0.0, pet, 0.0).sum(axis=0)
    spacing = lay.spacing_km * 1000.0
    res = spinup(t_m, p_m, bed, land, ocean, nbr, spacing, mode=ep.ice_init_mode,
                 min_years=ep.ice_min_spinup_years, max_years=ep.ice_max_spinup_years,
                 window=ep.ice_window_years, step_years=STEP_YEARS.get(ep.quality, 250.0),
                 sublimation_mm=sub)  # fmt: skip
    h = res.thickness_m
    area = lay.hex_area_km2 * 1e6
    shelf = np.zeros(n)
    gl = np.zeros(n, dtype=bool)
    shelf_ledger = {}
    if ep.ice_shelves and h.any():
        depth = np.nan_to_num(L["water_depth_m"].ravel().astype(np.float64))
        sst = out.get("sst_monthly_c")
        if sst is not None:
            sst_ann = np.nan_to_num(sst.astype(np.float64), nan=0.0).mean(axis=0).ravel()
        else:
            sst_ann = np.full(n, -1.8)
        sh = build_shelves(h, bed, depth, ocean, sst_ann, nbr, spacing,
                           inflow_hex=res.calving_m3_yr_hex)  # fmt: skip
        shelf, gl, shelf_ledger = sh.thickness_m, sh.grounding_line, sh.ledger
    w = np.zeros(n)
    if ep.isostasy and h.any():
        years = res.years if ep.ice_init_mode == "cold_start" else 1e6  # an old sheet: w_eq
        w = isostatic_deflection(h, years, grid_shape=shape)
    t_ann = t_m.mean(axis=0)
    erosion = glacial_erosion_rate(res.velocity_m_yr, h, t_ann) * 1000.0  # mm/yr
    # the ice above flotation is what would raise the sea (bed below sea level displaces)
    above = np.maximum(h + np.minimum(bed, 0.0) * 1028.0 / 917.0, 0.0)
    # the planet's ocean (share of sea in this map x the sphere): a regional window does not
    # hold the whole ocean, so its own sea area would exaggerate the equivalent
    sea_share = float(ocean.mean()) if n else 0.7
    radius_m = config.earth.planet_radius_km * 1000.0
    ocean_area = max(sea_share, 0.05) * 4.0 * np.pi * radius_m**2
    g = lambda a, dt=np.float32: np.asarray(a).reshape(shape).astype(dt)  # noqa: E731
    out["land_ice_thickness_m"] = g(h)
    out["land_ice_smb_m_yr"] = g(np.where(land, res.smb_m_yr, np.nan))
    out["ice_velocity_m_yr"] = g(res.velocity_m_yr)
    out["ice_seed_m"] = g(res.seed_m)
    out["shelf_thickness_m"] = g(shelf)
    out["grounding_line"] = g(gl, np.bool_)
    out["bedrock_deflection_m"] = g(w)
    out["land_surface_with_ice_m"] = g(np.where(land, bed + w + h, np.nan))
    out["glacial_erosion_mm_yr"] = g(erosion)
    ice = h > 10.0
    return {
        "status": res.status,
        "init_mode": ep.ice_init_mode,
        "years": round(res.years, 1),
        "ledger": {k: float(v) for k, v in res.ledger.items()},
        "convergence_history": res.history[-6:],
        "ice_hexes": int(ice.sum()),
        "area_km2": round(float(ice.sum() * area / 1e6), 1),
        "volume_km3": round(float((h * area).sum() / 1e9), 1),
        "h_max_m": round(float(h.max()), 1) if n else 0.0,
        "h_mean_m": round(float(h[ice].mean()), 1) if ice.any() else 0.0,
        "smb_positive_area_km2": round(float(((res.smb_m_yr > 0) & land).sum() * area / 1e6), 1),
        "smb_negative_ice_area_km2": round(float(((res.smb_m_yr < 0) & ice).sum() * area / 1e6),
                                           1),  # fmt: skip
        "sea_level_equivalent_m": round(sea_level_equivalent_m((above * area).sum(),
                                                               ocean_area), 3),  # fmt: skip
        "shelves": {k: float(v) for k, v in shelf_ledger.items()},
        "shelf_hexes": int((shelf > 0).sum()),
        "max_bed_deflection_m": round(float(w.min()), 1) if n else 0.0,
        "applied_to_relief": False,
        "quality": {"flow": "SIA_semi_implicit_CALIBRATED_APPROXIMATION",
                    "shelves": "flux_constrained_P0_no_buttressing",
                    "isostasy": "delayed_local_airy_smoothed_P1",
                    "erosion": "sliding_proxy_SYNTHETIC_PROXY"},
    }  # fmt: skip


def _morphodynamics(project, out, ocean_hex, lake_hex, shape):
    """Coastal sand budget first (beach stores), then wind-blown sand and dust."""
    from .earth.morpho import aeolian, coastal, mobile_sand
    from .pipeline_v50 import _hex_neighbours
    from .stages.coasts import COAST_TYPES
    from .stages.environment import PM

    lay = project.layout
    L, E = project.hydro.layers, project.env.layers
    n = lay.width * lay.height
    nbr = _hex_neighbours(lay)
    spacing = lay.spacing_km * 1000.0
    ocean = ocean_hex.ravel()
    lake = lake_hex.ravel()
    pm = E["parent_material_class"].ravel()
    resist = np.select([pm == PM["crystalline_bedrock"], pm == PM["orogenic_bedrock"],
                        np.isin(pm, [PM["young_volcanic"], PM["weathered_volcanic"]]),
                        pm == PM["sedimentary_bedrock"]], [1.0, 0.8, 0.6, 0.4], 0.3)  # fmt: skip
    ct = L["coast_type"].ravel() if "coast_type" in L else np.zeros(n, dtype=np.uint8)
    co = coastal(nbr, spacing, COAST_TYPES, ct, L["wave_exposure"].ravel(), ocean,
                 L["wind_u_ms"].ravel().astype(np.float64),
                 L["wind_v_ms"].ravel().astype(np.float64),
                 L["sediment_delivery_m3_yr"].ravel().astype(np.float64), resist)  # fmt: skip
    inv = mobile_sand(pm, E["sand_frac"].ravel(), E["soil_depth_m"].ravel(), co["beach_sand_m"])
    t_m = project.env.monthly["temperature_monthly_c"].reshape(12, n)
    frozen = np.stack([(t_m[list(mm)] < 0.0).mean(axis=0) for mm in
                       ((11, 0, 1), (2, 3, 4), (5, 6, 7), (8, 9, 10))])  # fmt: skip
    wet = np.clip(np.nan_to_num(E["aridity_p_over_pet"].ravel(), nan=1.0), 0.0, 1.0)
    veg = np.clip(np.nan_to_num(E["aet_over_pet"].ravel(), nan=0.0) * 1.2, 0.0, 1.0)
    ice = out.get("land_ice_thickness_m")
    if ice is not None:  # sand under an ice sheet does not blow
        frozen = np.where(ice.ravel()[None] > 10.0, 1.0, frozen)
    us = out["wind_surface_u_seasonal_ms"].reshape(4, n).astype(np.float64)
    vs = out["wind_surface_v_seasonal_ms"].reshape(4, n).astype(np.float64)
    ae = aeolian(nbr, spacing, us, vs, inv, wet, veg, frozen, ocean, lake,
                 silt_frac=E["silt_frac"].ravel(), beach_source=co["beach_sand_m"])  # fmt: skip
    g = lambda a, dt=np.float32: np.asarray(a).reshape(shape).astype(dt)  # noqa: E731
    out["sand_thickness_m"] = g(ae["sand_thickness_m"])
    out["sand_accumulation_m"] = g(ae["sand_accumulation_m"])
    out["sand_flux_kg_m_yr"] = g(ae["sand_flux_kg_m_yr"])
    out["dune_type"] = g(ae["dune_type"], np.uint8)
    out["loess_mm_kyr"] = g(ae["loess_mm_kyr"])
    out["shoreline_change_m_yr"] = g(co["shoreline_change_m_yr"])
    out["longshore_transport_m3_yr"] = g(co["longshore_m3_yr"])
    names = ("нет", "барханы", "поперечные гряды", "продольные гряды", "звёздчатые",
             "прибрежные дюны")  # fmt: skip
    counts = np.bincount(ae["dune_type"], minlength=6)
    return {
        "sand_ledger": ae["ledger"],
        "dust_ledger": ae.get("dust_ledger"),
        "coast_ledger": co["ledger"],
        "dune_hexes": {names[i]: int(counts[i]) for i in range(1, 6)},
        "loess_hexes_over_10mm_kyr": int((ae["loess_mm_kyr"] > 10.0).sum()),
        "eroding_coast_hexes": int((np.nan_to_num(co["shoreline_change_m_yr"]) < -0.05).sum()),
        "accreting_coast_hexes": int((np.nan_to_num(co["shoreline_change_m_yr"]) > 0.05).sum()),
        "quality": {"aeolian": "bagnold_threshold_fv_CALIBRATED_APPROXIMATION",
                    "coast": "one_line_hex_scale_P1_APPROXIMATE",
                    "relief": "layers_only (dunes are not added to elevation_m)"},
    }  # fmt: skip


def run_earth(config, project) -> dict:
    from .stages.climate import _bilinear

    t0 = time.perf_counter()
    wc = _world_climate(config, project)
    es = wc.earth
    lay = project.layout
    shape = (lay.height, lay.width)
    g = wc.grid
    hx, hy = lay.centers()
    L = project.hydro.layers
    ocean_hex = L["ocean_mask"].astype(bool)
    lake_hex = (L["lake_id"] >= 0) & ~ocean_hex
    oc = es.ocean
    out = {}
    if oc is not None:
        sea_r = oc.ocean

        def sample(a, fill=np.nan, dtype=np.float32):
            a = np.asarray(a, dtype=np.float64)
            valid = sea_r if a.ndim == 2 else np.broadcast_to(sea_r, a.shape)
            filled = _filled(np.where(valid, np.nan_to_num(a), 0.0), sea_r)
            if filled.ndim == 2:
                v = _bilinear(filled, g, hx, hy)
            else:
                v = np.stack([_bilinear(filled[k], g, hx, hy) for k in range(filled.shape[0])])
            return np.where(ocean_hex, v, fill).astype(dtype)

        out["sst_monthly_c"] = sample(oc.sst_monthly)
        out["sea_ice_concentration_monthly"] = np.clip(
            sample(oc.sea_ice_monthly, 0.0, np.float16), 0, 1)  # fmt: skip
        out["sea_ice_thickness_monthly_m"] = sample(oc.ice_thickness_monthly, 0.0, np.float16)
        out["current_u_ms"] = sample(oc.current_u, 0.0, np.float16)
        out["current_v_ms"] = sample(oc.current_v, 0.0, np.float16)
        out["upwelling_m_day"] = sample(oc.upwelling_m_day, 0.0, np.float16)
        out["sea_surface_salinity_psu"] = sample(oc.salinity_psu)
        # categorical fields: nearest world cell
        col = np.clip(((hx - g["x0_km"]) / g["cell_km"]).astype(int), 0, g["nx"] - 1)
        row = np.clip(((hy - g["y0_km"]) / g["cell_km"]).astype(int), 0, g["ny"] - 1)
        itype = _filled(oc.ice_type.astype(np.int64), sea_r)[row, col]
        basin = _filled(oc.basin_id.astype(np.int64), sea_r)[row, col]
        out["sea_ice_type"] = np.where(ocean_hex, itype, 0).astype(np.uint8)
        out["ocean_basin_id"] = np.where(ocean_hex, basin, -1).astype(np.int32)
        cu = _filled(oc.coastal_upwelling.astype(np.int64), sea_r)[row, col] > 0
        out["coastal_upwelling"] = (cu & ocean_hex).astype(np.bool_)
    al = es.aloft
    if al is not None:
        nlev = len(al.levels_hpa)

        def column(a):
            return np.stack(
                [np.stack([_bilinear(a[s, k], g, hx, hy) for k in range(nlev)]) for s in range(4)]
            )

        su, sv, sz = column(al.u), column(al.v), column(al.z)
        surface = np.maximum(L["elevation_m"].astype(np.float64) - project.hydro.sea_level_m, 0.0)
        out["winds_aloft_u_ms"] = su.astype(np.float16)
        out["winds_aloft_v_ms"] = sv.astype(np.float16)
        out["winds_aloft_height_m"] = sz.astype(np.float16)
        out["winds_aloft_below_ground"] = (sz < surface[None, None]).astype(np.bool_)
        out["jet_intensity_ms"] = np.stack([_bilinear(al.jet_intensity[s], g, hx, hy)
                                            for s in range(4)]).astype(np.float16)  # fmt: skip
    # seasonal surface winds (the climate's own) for the sediment transport and the map
    out["wind_surface_u_seasonal_ms"] = np.stack(
        [_bilinear(wc.seasonal_wind_u[s], g, hx, hy) for s in range(4)]
    ).astype(np.float16)
    out["wind_surface_v_seasonal_ms"] = np.stack(
        [_bilinear(wc.seasonal_wind_v[s], g, hx, hy) for s in range(4)]
    ).astype(np.float16)
    ice_summary = None
    if config.earth.ice_sheets:
        ice_summary = _ice_sheets(config, project, out, ocean_hex, shape)
    morpho_summary = None
    if getattr(config.earth, "morphodynamics", False):
        morpho_summary = _morphodynamics(project, out, ocean_hex, lake_hex, shape)
    # lake ice: a lake freezes in the months with a mean air temperature below -1 degC
    t_m = project.env.monthly["temperature_monthly_c"]
    lake_ice = ((t_m < -1.0).sum(axis=0) * lake_hex).astype(np.uint8)
    if "land_ice_thickness_m" in out:  # a lake under the sheet is subglacial, not ice-covered
        lake_ice = np.where(out["land_ice_thickness_m"] > 10.0, 0, lake_ice).astype(np.uint8)
    out["lake_ice_months"] = lake_ice
    realm = np.where(ocean_hex, 3, np.where(lake_hex, 2, 1)).astype(np.uint8)
    out["realm"] = realm
    cover = np.zeros(shape, dtype=np.uint8)
    if "sea_ice_type" in out:
        st = out["sea_ice_type"]
        cover = np.where(st == 1, 1, np.where(st == 2, 2, np.where(st == 3, 3, 0)))
    cover = np.where(lake_ice > 0, 4, cover).astype(np.uint8)
    if "land_ice_thickness_m" in out:
        cover = np.where(out["land_ice_thickness_m"] > 10.0, 5, cover)
        cover = np.where(out["shelf_thickness_m"] > 0.0, 6, cover).astype(np.uint8)
    out["ice_cover"] = cover
    pet = project.env.monthly["pet_monthly_mm"].astype(np.float64).sum(axis=0)
    out["et0_annual_mm"] = np.where(realm == 1, pet, np.nan).astype(np.float32)
    summary = {
        "algorithm_version": EARTH_ALGORITHM_VERSION,
        "coupler": es.report,
        "pet_method": project.env.summary.get("pet_method"),
        "quality": QUALITY,
        "levels_hpa": list(al.levels_hpa) if al is not None else [],
        "ocean_hexes": int(ocean_hex.sum()),
        "sea_ice_hexes_max": int((np.asarray(out.get("sea_ice_concentration_monthly", 0)).max(
            axis=0) > 0.15).sum()) if "sea_ice_concentration_monthly" in out else 0,  # fmt: skip
        "lake_ice_hexes": int((lake_ice > 0).sum()),
        "ice": ice_summary,
        "morphodynamics": morpho_summary,
        "conventions": {"current_v_ms / winds_aloft_v_ms": "+ south (map y), like wind_v_ms",
                        "realm": "1 land, 2 inland water, 3 ocean",
                        "ice_cover": "0 none, 1 seasonal sea ice, 2 multiyear, 3 fast, 4 lake, "
                                     "5 grounded ice, 6 floating ice shelf"},
    }  # fmt: skip
    return {"layers": out, "summary": summary,
            "timings": {"v53_earth": round(time.perf_counter() - t0, 3)}}  # fmt: skip


def run_v53(config, geometry, project, old_keys: dict, old_cache: dict, report):
    keys = earth_stage_keys(config, project)
    if old_keys.get("v53_earth") == keys["v53_earth"] and "v53_earth" in old_cache:
        data = old_cache["v53_earth"]
        return data, {"v53_earth": data}, ["v53_earth"]
    report("v53_earth", 0.997)
    res = run_earth(config, project)
    data = EarthData(layers=res["layers"], summary=res["summary"],
                     provenance={"algorithm_version": EARTH_ALGORITHM_VERSION,
                                 "params": asdict(config.earth)},
                     stage_keys=keys, warnings=[], timings=res["timings"])  # fmt: skip
    if res["summary"]["coupler"].get("status") == "NOT_CONVERGED":
        data.warnings.append("v5.3: ocean-ice-climate coupler NOT_CONVERGED (see the report)")
    return data, {"v53_earth": data}, []
