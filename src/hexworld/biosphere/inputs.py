"""BioInputs: the physical fields of the existing world that the biosphere reads (input
contracts C01–C08, ``docs/v5_5/input_contracts.csv``). Flat arrays over the N hexes, in the
units named in the field names. A missing input stays NaN (B09); nothing here invents a
value for a field the world does not have.

The biosphere never writes back into these arrays: every stage of the world keeps its hash.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

#: PAR share of the global shortwave (McCree) [S]
PAR_SHARE = 0.48
#: sunshine fraction assumed over the ocean where no surface radiation exists (APPROX)
OCEAN_SUNSHINE = 0.5
#: lapse rate for the sub-hex elevation bins, K/km
LAPSE_K_KM = 6.5
#: elevation quantiles of the hex used as microsites and their area weights (B14)
MICROSITES = {"standard": (("hyps_q05_m", 0.2), ("hyps_q50_m", 0.6), ("hyps_q95_m", 0.2)),
              "detailed": (("hyps_q05_m", 0.1), ("hyps_q25_m", 0.2), ("hyps_q50_m", 0.4),
                           ("hyps_q75_m", 0.2), ("hyps_q95_m", 0.1)),
              "basic": (("hyps_q50_m", 1.0),)}  # fmt: skip


def neighbours(layout) -> np.ndarray:
    """(N, 6) flat neighbour indices, -1 outside the grid."""
    from ..stages.raster import neighbor_index

    cols = []
    for d in range(6):
        nb, valid = neighbor_index(layout, d)
        cols.append(np.where(valid.ravel(), nb.ravel(), -1))
    return np.stack(cols, axis=1).astype(np.int64)


@dataclass
class BioInputs:
    n: int
    shape: tuple
    area_km2: float
    spacing_km: float
    lat: np.ndarray
    x_km: np.ndarray
    y_km: np.ndarray
    nbr: np.ndarray
    days: np.ndarray  # (12,) month lengths, days
    f: dict = field(default_factory=dict)  # name -> flat array (or (12, N))
    quality: dict = field(default_factory=dict)

    def __getitem__(self, k):
        return self.f[k]


def _flat(a):
    a = np.asarray(a)
    return a.reshape(a.shape[0], -1) if a.ndim == 3 else a.ravel()


def _get(obj, name, kind="layers"):
    d = getattr(obj, kind, None) if obj is not None else None
    if not isinstance(d, dict) or name not in d:
        return None
    return np.asarray(d[name])


def from_project(project, detail: str = "standard") -> BioInputs:
    from ..earth.planet import MONTH_FRACTION, Planet
    from ..pipeline_v54 import month_days_of

    lay = project.layout
    n = lay.width * lay.height
    H, E, B = project.hydro.layers, project.env.layers, project.biomes.layers
    EM = project.env.monthly
    T = project.topo
    ea = project.earth.layers if project.earth is not None else {}
    R = project.resources.layers if project.resources is not None else {}
    V = project.volcanism.layers if project.volcanism is not None else {}
    q: dict = {}

    def h(k, src=H, fill=None):
        a = src.get(k) if isinstance(src, dict) else None
        if a is None:
            q[k] = "missing"
            return None if fill is None else np.full(n, fill)
        return _flat(np.asarray(a, dtype=np.float64) if np.asarray(a).dtype != bool
                     else np.asarray(a))  # fmt: skip

    cols = np.arange(lay.width)[None, :].repeat(lay.height, 0)
    rows = np.arange(lay.height)[:, None].repeat(lay.width, 1)
    xc, yc = lay.center(cols, rows)
    f: dict = {}
    land = h("land_mask").astype(bool)
    ocean = h("ocean_mask").astype(bool)
    lake = h("inland_water_mask").astype(bool) & ~ocean
    f["land"], f["ocean"], f["lake"] = land, ocean, lake
    f["below_sea_land"] = h("below_sea_level_land_mask").astype(bool) & ~ocean
    f["river"] = h("river_mask").astype(bool)
    f["elevation_m"] = h("elevation_m")
    f["depth_m"] = np.where(ocean, np.nan_to_num(h("water_depth_m")), 0.0)
    f["lake_depth_m"] = np.where(lake, h("lake_depth_m"), np.nan)
    f["discharge_m3_s"] = np.nan_to_num(h("discharge_m3_s"))
    f["river_width_m"] = np.nan_to_num(h("river_width_m"))
    f["river_depth_m"] = np.nan_to_num(h("river_depth_m"))
    f["flow_receiver"] = _flat(H["flow_receiver"]).astype(np.int64)
    f["basin_id"] = _flat(H["basin_id"]).astype(np.int64)
    f["lake_id"] = _flat(H["lake_id"]).astype(np.int64)
    f["landmass_id"] = _flat(H["landmass_id"]).astype(np.int64)
    f["endorheic"] = h("endorheic_mask").astype(bool)
    f["river_mouth"] = h("river_mouth").astype(bool)
    f["tidal_range_m"] = np.nan_to_num(h("tidal_range_m"))
    f["coast_type"] = _flat(H["coast_type"]).astype(np.int64)
    f["lgm_ice"] = h("lgm_ice_mask").astype(bool)
    f["block_id"] = _flat(project.geo.layers["continental_block_id"]).astype(np.int64)
    # sub-hex fractions of open water (4.7 terrain units) where they exist
    to = T.layers.get("topo_ocean_fraction") if T is not None else None
    tl = T.layers.get("topo_lake_fraction") if T is not None else None
    f["topo_ocean_fraction"] = _flat(to) if to is not None else np.full(n, np.nan)
    f["topo_lake_fraction"] = _flat(tl) if tl is not None else np.full(n, np.nan)
    f["glacier_fraction"] = np.nan_to_num(_flat(B["glacier_fraction"]))
    for k in ("cover_rock", "cover_salt", "cover_sand", "cover_snow_perennial", "cover_firn"):
        f[k] = np.nan_to_num(_flat(B[k])) if k in B else np.zeros(n)
    # climate (monthly)
    t_env = _flat(EM["temperature_monthly_c"]).astype(np.float64)
    t_topo = _flat(T.monthly["topo_temperature_c"]).astype(np.float64) if T is not None else None
    t_land = np.where(np.isfinite(t_topo), t_topo, t_env) if t_topo is not None else t_env
    sst = _flat(ea["sst_monthly_c"]).astype(np.float64) if "sst_monthly_c" in ea else None
    if sst is None:
        q["sst_monthly_c"] = "missing"
        sst = np.where(ocean[None, :], t_env, np.nan)
    f["t_air_m"] = t_land.astype(np.float32)
    f["sst_m"] = np.where(ocean[None, :], sst, np.nan).astype(np.float32)
    # water temperature of lakes and rivers: air temperature above freezing (APPROX)
    f["t_fresh_m"] = np.maximum(f["t_air_m"], 0.0)
    del t_env, t_topo, t_land, sst
    for k, src in (("precip_m", "precip_monthly_mm"), ("pet_m", "pet_monthly_mm"),
                   ("aet_m", "aet_monthly_mm")):  # fmt: skip
        f[k] = _flat(EM[src]).astype(np.float32)
    f["awc_mm"] = h("awc_mm", E)
    f["soil_depth_m"] = h("soil_depth_m", E)
    f["mi"] = h("aridity_p_over_pet", E)
    f["gdd5"] = h("gdd5_proxy_c_day", E)
    f["wetness"] = np.nan_to_num(h("wetness_potential", E))
    f["waterlogging_months"] = np.nan_to_num(h("waterlogging_months", E))
    f["soil_present"] = h("soil_present", E).astype(bool)
    # soil chemistry of 5.2 (kg/m²); NaN where the soil model has no value
    for k, src in (("ph", "soil_ph"), ("ec_ds_m", "soil_salinity_ec_ds_m"),
                   ("n_min_kg_m2_yr", "soil_N_net_mineralisation_kg_m2_yr"),
                   ("p_lab_kg_m2", "soil_P_labile_kg_m2"),
                   ("soc_kg_m2", "soil_soc_kg_m2")):  # fmt: skip
        a = R.get(src)
        f[k] = _flat(a).astype(np.float64) if a is not None else np.full(n, np.nan)
        if a is None:
            q[src] = "missing"
    # radiation: land shortwave of 4.7; the ocean gets Angström–Prescott from the TOA
    planet = Planet.from_params(project.config.earth)
    days = np.asarray(month_days_of(project.config), dtype=np.float64)
    lat = _flat(H["latitude_deg"]).astype(np.float64)
    ra = np.stack([planet.ra_mj(lat, MONTH_FRACTION[m]) for m in range(12)])  # MJ/m²/day
    rs_ocean = (0.25 + 0.5 * OCEAN_SUNSHINE) * ra
    rs_land = _flat(T.monthly["topo_radiation_mj_m2_day"]).astype(np.float64) if T else None
    rs = np.where(np.isfinite(rs_land), rs_land, rs_ocean) if rs_land is not None else rs_ocean
    f["par_m"] = (PAR_SHARE * rs * days[:, None]).astype(np.float32)  # MJ/m²/month
    del ra, rs_ocean, rs_land, rs
    # microsite elevation offsets (K) from the hex hypsometry
    zmean = np.nan_to_num(h("hyps_z_mean_m", T.layers) if T is not None else np.zeros(n))
    bins = []
    for name, w in MICROSITES[detail]:
        z = T.layers.get(name) if T is not None else None
        dz = (_flat(z).astype(np.float64) - zmean) if z is not None else np.zeros(n)
        bins.append((np.nan_to_num(-LAPSE_K_KM * dz / 1000.0), w))
    f["microsites"] = bins

    # ocean
    def e(k, fill=0.0):
        a = ea.get(k)
        if a is None:
            q[k] = "missing"
            return np.full(n, fill)
        return _flat(np.asarray(a, dtype=np.float64))

    ice = ea.get("sea_ice_concentration_monthly")
    f["sea_ice_m"] = np.nan_to_num(_flat(ice).astype(np.float32)) if ice is not None else \
        np.zeros((12, n), np.float32)  # fmt: skip
    f["sss_psu"] = np.where(ocean, e("sea_surface_salinity_psu", 35.0), np.nan)
    up = ea.get("upwelling_m_day")
    f["upwelling_m_day"] = (np.maximum(_flat(up).astype(np.float64), 0).mean(axis=0)
                            if up is not None else np.zeros(n))  # fmt: skip
    cu, cv = ea.get("current_u_ms"), ea.get("current_v_ms")
    f["current_u_ms"] = _flat(cu).mean(axis=0) if cu is not None else np.zeros(n)
    f["current_v_ms"] = _flat(cv).mean(axis=0) if cv is not None else np.zeros(n)
    for k in (
        "marine_kd_per_m",
        "marine_euphotic_depth_m",
        "marine_bottom_light_fraction",
        "marine_nutrient_supply_proxy",
        "marine_coral_reef_potential",
    ):
        f[k] = _flat(B[k]).astype(np.float64) if k in B else np.full(n, np.nan)
    f["marine_benthic_zone"] = _flat(B["marine_benthic_zone"]).astype(np.int64)
    f["ocean_basin_id"] = (_flat(ea["ocean_basin_id"]).astype(np.int64)
                           if "ocean_basin_id" in ea else np.where(ocean, 0, -1))  # fmt: skip

    # volcanism, history
    def v(k, fill=0.0):
        a = V.get(k)
        return np.nan_to_num(_flat(np.asarray(a, dtype=np.float64)), nan=fill) if a is not None \
            else np.full(n, fill)  # fmt: skip

    f["lava_fraction"] = v("lava_fraction")
    f["volcanic_barren_fraction"] = v("volcanic_barren_fraction")
    f["lava_youngest_years_bp"] = v("lava_youngest_years_bp", np.inf)
    f["hydrothermal_kind"] = v("hydrothermal_kind").astype(np.int64)
    f["geothermal_heat_mw"] = v("geothermal_heat_mw")
    hl = project.hist.layers if project.hist is not None else {}
    f["crust_age_ma"] = _flat(hl["crust_age_ma"]).astype(np.float64) if "crust_age_ma" in hl \
        else np.full(n, np.nan)  # fmt: skip
    f["volcanic_age_ma"] = _flat(hl["volcanic_age_ma"]).astype(np.float64) \
        if "volcanic_age_ma" in hl else np.full(n, np.nan)  # fmt: skip
    f["spreading_axis_km"] = _flat(hl["spreading_axis_distance_km"]).astype(np.float64) \
        if "spreading_axis_distance_km" in hl else np.full(n, np.inf)  # fmt: skip
    inp = BioInputs(n, (lay.height, lay.width), float(lay.hex_area_km2), float(lay.spacing_km),
                    lat, xc.ravel().astype(np.float64), yc.ravel().astype(np.float64),
                    neighbours(lay), days, f, q)  # fmt: skip
    _derived(inp)
    return inp


def _derived(inp: BioInputs) -> None:
    f = inp.f
    t = f["t_air_m"]
    f["tc"] = np.where(
        f["ocean"],
        np.nanmin(f["sst_m"], axis=0, initial=np.inf, where=np.isfinite(f["sst_m"])),
        t.min(axis=0),
    )
    f["tw"] = np.where(
        f["ocean"],
        np.nanmax(f["sst_m"], axis=0, initial=-np.inf, where=np.isfinite(f["sst_m"])),
        t.max(axis=0),
    )
    f["tc"] = np.where(np.isfinite(f["tc"]), f["tc"], np.nan)
    f["tw"] = np.where(np.isfinite(f["tw"]), f["tw"], np.nan)
    nb = inp.nbr
    ok = nb >= 0
    ocean_n = np.where(ok, f["ocean"][np.maximum(nb, 0)], False)
    land_n = np.where(ok, f["land"][np.maximum(nb, 0)], False)
    f["coast_land"] = f["land"] & ocean_n.any(axis=1)
    f["coast_sea"] = f["ocean"] & land_n.any(axis=1)
    f["sea_ice_months"] = (f["sea_ice_m"] >= 0.15).sum(axis=0)


def habitat_fractions(inp: BioInputs) -> dict:
    """Area fractions of every hex: land_ground + inland_water + marine_water + grounded_ice
    = 1 exactly; sea ice is an overlay on the marine part, not extra area (B11). A dry
    depression below sea level stays terrestrial (B12)."""
    f = inp.f
    land, ocean = f["land"], f["ocean"]
    to = f["topo_ocean_fraction"]
    marine = np.where(ocean, 1.0, 0.0)
    coast = land & np.isfinite(to) & f["coast_land"]
    marine = np.where(coast, np.clip(to, 0.0, 0.5), marine)  # a coastal land hex: sea share
    marine = np.where(ocean & np.isfinite(to) & f["coast_sea"], np.clip(to, 0.5, 1.0), marine)
    marine = np.where(f["below_sea_land"] & ~f["coast_land"], 0.0, marine)
    # inland water: lakes, plus the channel area of a river (width x one hex length)
    channel = np.where(f["river"], f["river_width_m"] / 1000.0 * inp.spacing_km / inp.area_km2,
                       0.0)  # fmt: skip
    tl = np.nan_to_num(f["topo_lake_fraction"])
    inland = np.where(f["lake"], 1.0 - marine, np.clip(tl + channel, 0.0, 0.5))
    inland = np.minimum(inland, 1.0 - marine)
    rest = 1.0 - marine - inland
    ice = np.where(land | f["lake"], np.clip(f["glacier_fraction"], 0.0, 1.0) * rest, 0.0)
    ground = np.maximum(rest - ice, 0.0)
    tot = ground + inland + marine + ice
    ground = ground / tot
    inland, marine, ice = inland / tot, marine / tot, ice / tot
    sea_ice = f["sea_ice_m"] * marine[None, :]  # overlay area share per month
    return {"land_ground": ground, "inland_water": inland, "marine_water": marine,
            "grounded_ice": ice, "sea_ice_overlay_m": sea_ice}  # fmt: skip
