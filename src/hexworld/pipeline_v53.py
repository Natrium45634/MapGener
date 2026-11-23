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
    # lake ice: a lake freezes in the months with a mean air temperature below -1 degC
    t_m = project.env.monthly["temperature_monthly_c"]
    lake_ice = ((t_m < -1.0).sum(axis=0) * lake_hex).astype(np.uint8)
    out["lake_ice_months"] = lake_ice
    realm = np.where(ocean_hex, 3, np.where(lake_hex, 2, 1)).astype(np.uint8)
    out["realm"] = realm
    cover = np.zeros(shape, dtype=np.uint8)
    if "sea_ice_type" in out:
        st = out["sea_ice_type"]
        cover = np.where(st == 1, 1, np.where(st == 2, 2, np.where(st == 3, 3, 0)))
    cover = np.where(lake_ice > 0, 4, cover).astype(np.uint8)
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
        "conventions": {"current_v_ms / winds_aloft_v_ms": "+ south (map y), like wind_v_ms",
                        "realm": "1 land, 2 inland water, 3 ocean",
                        "ice_cover": "0 none, 1 seasonal sea ice, 2 multiyear, 3 fast, 4 lake"},
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
