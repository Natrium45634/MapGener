"""Manufactured BioInputs for the acceptance fixtures of TZ 5.5 (B11–B57, S01–S20): a hex grid
with explicit physical fields, no generator behind it, so the expected outcome is known.

Every field has its physical unit; the defaults describe a mild, wet, fertile land and a
temperate ocean. A test overrides only what it controls (positive / negative controls)."""

from __future__ import annotations

import numpy as np
from scipy import ndimage

from ..geometry import HexLayout
from .inputs import BioInputs, _derived, neighbours

DAYS = np.array([31, 28.25, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31], dtype=np.float64)


def make(land, *, spacing_km=50.0, t_mean=14.0, t_amp=8.0, sst=None, precip_mm=1000.0,
         pet_mm=800.0, aet_ratio=0.9, par_mj_m2_day=12.0, n_min=0.04, p_lab=0.0015, ph=6.0,
         depth_m=3000.0, block=None, basin=None, river=None, lake=None, flow_receiver=None,
         nutr=0.2, sea_ice=None, kd=0.06, hemisphere=1, **over) -> BioInputs:  # fmt: skip
    """``land``: bool (H, W). Scalars broadcast; arrays (H, W) or (12, H, W) are taken as is."""
    land = np.asarray(land, dtype=bool)
    h, w = land.shape
    n = h * w
    lay = HexLayout(w, h, spacing_km, 0.0, 0.0)
    cols = np.arange(w)[None, :].repeat(h, 0)
    rows = np.arange(h)[:, None].repeat(w, 1)
    x, y = lay.center(cols, rows)

    def full(v):
        a = np.asarray(v, dtype=np.float64)
        return np.broadcast_to(a, (h, w)).ravel().copy() if a.ndim < 3 else a.reshape(12, n)

    def monthly(mean, amp):
        m = np.arange(12)
        phase = np.cos(2 * np.pi * (m - (6 if hemisphere > 0 else 0)) / 12.0)
        base = full(mean)
        return base[None, :] + np.asarray(amp, dtype=np.float64) * phase[:, None]

    ocean = ~land
    if lake is not None:
        ocean = ocean & ~np.asarray(lake, bool)
    f: dict = {}
    f["land"] = land.ravel()
    f["ocean"] = ocean.ravel()
    f["lake"] = (np.asarray(lake, bool) if lake is not None else np.zeros((h, w), bool)).ravel()
    f["below_sea_land"] = np.zeros(n, bool)
    f["river"] = (np.asarray(river, bool) if river is not None else np.zeros((h, w), bool)).ravel()
    f["elevation_m"] = np.where(f["land"], 200.0, -depth_m)
    f["depth_m"] = np.where(f["ocean"], full(depth_m), 0.0)
    f["lake_depth_m"] = np.where(f["lake"], 5.0, np.nan)
    f["discharge_m3_s"] = np.where(f["river"], 100.0, 0.0)
    f["river_width_m"] = np.where(f["river"], 200.0, 0.0)
    f["river_depth_m"] = np.where(f["river"], 3.0, 0.0)
    f["flow_receiver"] = (np.asarray(flow_receiver).ravel() if flow_receiver is not None
                          else np.full(n, -1)).astype(np.int64)  # fmt: skip
    lm, _k = ndimage.label(land)
    f["landmass_id"] = np.where(land, lm - 1, -1).ravel().astype(np.int64)
    f["basin_id"] = (np.asarray(basin).ravel() if basin is not None else
                     f["landmass_id"].copy()).astype(np.int64)  # fmt: skip
    f["lake_id"] = np.where(f["lake"], 0, -1).astype(np.int64)
    f["endorheic"] = np.zeros(n, bool)
    f["river_mouth"] = np.zeros(n, bool)
    f["tidal_range_m"] = np.full(n, 1.0)
    f["coast_type"] = np.zeros(n, np.int64)
    f["lgm_ice"] = np.zeros(n, bool)
    f["block_id"] = (np.asarray(block).ravel() if block is not None else
                     np.where(land, 0, -1).ravel()).astype(np.int64)  # fmt: skip
    f["topo_ocean_fraction"] = np.full(n, np.nan)
    f["topo_lake_fraction"] = np.full(n, np.nan)
    f["glacier_fraction"] = np.zeros(n)
    for k in ("cover_rock", "cover_salt", "cover_sand", "cover_snow_perennial", "cover_firn"):
        f[k] = np.zeros(n)
    f["t_air_m"] = monthly(t_mean, t_amp)
    f["sst_m"] = np.where(f["ocean"][None, :], monthly(t_mean if sst is None else sst,
                                                       0.5 * t_amp), np.nan)  # fmt: skip
    f["t_fresh_m"] = np.maximum(f["t_air_m"], 0.0)
    p = full(precip_mm) / 12.0
    pet = full(pet_mm) / 12.0
    f["precip_m"] = np.repeat(p[None, :], 12, 0)
    f["pet_m"] = np.repeat(pet[None, :], 12, 0)
    f["aet_m"] = f["pet_m"] * np.asarray(aet_ratio, dtype=np.float64)
    f["awc_mm"] = np.full(n, 150.0)
    f["soil_depth_m"] = np.where(f["land"], 1.0, 0.0)
    with np.errstate(divide="ignore", invalid="ignore"):
        f["mi"] = np.where(pet > 0, p / pet, np.nan)
    f["gdd5"] = np.sum(np.maximum(f["t_air_m"] - 5.0, 0.0) * DAYS[:, None], axis=0)
    f["wetness"] = np.full(n, 0.2)
    f["waterlogging_months"] = np.zeros(n)
    f["soil_present"] = f["land"].copy()
    f["ph"] = np.where(f["land"], full(ph), np.nan)
    f["ec_ds_m"] = np.where(f["land"], 0.5, np.nan)
    f["n_min_kg_m2_yr"] = np.where(f["land"], full(n_min), np.nan)
    f["p_lab_kg_m2"] = np.where(f["land"], full(p_lab), np.nan)
    f["soc_kg_m2"] = np.where(f["land"], 8.0, np.nan)
    par = full(par_mj_m2_day) if np.ndim(par_mj_m2_day) < 3 else None
    f["par_m"] = (np.repeat(par[None, :], 12, 0) if par is not None else
                  np.asarray(par_mj_m2_day, float).reshape(12, n)) * 0.48 / 0.48 * DAYS[:, None]
    f["microsites"] = [(np.zeros(n), 1.0)]
    f["sea_ice_m"] = (np.asarray(sea_ice, float).reshape(12, n) if sea_ice is not None
                      else np.zeros((12, n)))  # fmt: skip
    f["sss_psu"] = np.where(f["ocean"], 35.0, np.nan)
    f["upwelling_m_day"] = np.zeros(n)
    f["current_u_ms"] = np.zeros(n)
    f["current_v_ms"] = np.zeros(n)
    f["marine_kd_per_m"] = np.where(f["ocean"], full(kd), np.nan)
    f["marine_euphotic_depth_m"] = np.log(100.0) / f["marine_kd_per_m"]
    f["marine_bottom_light_fraction"] = np.where(f["ocean"], np.exp(-full(kd) * f["depth_m"]),
                                                 np.nan)  # fmt: skip
    f["marine_nutrient_supply_proxy"] = np.where(f["ocean"], full(nutr), np.nan)
    f["marine_coral_reef_potential"] = np.where(f["ocean"], 0.0, np.nan)
    f["marine_benthic_zone"] = np.where(f["ocean"], np.where(f["depth_m"] <= 200, 1, 3), 0)
    f["ocean_basin_id"] = np.where(f["ocean"], 0, -1)
    f["lava_fraction"] = np.zeros(n)
    f["volcanic_barren_fraction"] = np.zeros(n)
    f["lava_youngest_years_bp"] = np.full(n, np.inf)
    f["hydrothermal_kind"] = np.zeros(n, np.int64)
    f["geothermal_heat_mw"] = np.zeros(n)
    f["crust_age_ma"] = np.where(f["block_id"] >= 0, 1000.0, 20.0)
    f["volcanic_age_ma"] = np.full(n, np.nan)
    f["spreading_axis_km"] = np.full(n, np.inf)
    for k, v in over.items():
        f[k] = v if not isinstance(v, np.ndarray) else (v.reshape(12, n) if v.ndim == 3 else v.ravel())
    inp = BioInputs(n, (h, w), float(lay.hex_area_km2), float(spacing_km), np.full(n, 45.0),
                    x.ravel().astype(float), y.ravel().astype(float), neighbours(lay), DAYS, f, {})
    _derived(inp)
    return inp


def landmasses(inp) -> list:
    ids = sorted(set(inp.f["landmass_id"][inp.f["landmass_id"] >= 0].tolist()))
    return [{"id": i, "name": f"L{i}"} for i in ids]
