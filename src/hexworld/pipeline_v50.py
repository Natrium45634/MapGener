"""v5.0 stage: biomes, natural cover, ecosystems, cryosphere (TZ 5.0).

    v4.6 surface + water connectivity, v4 climate / rivers / lakes, v4.5 soils and water,
    v4.7 terrain units with their local 12-month climate (same surface revision)
      -> EnvironmentSnapshot adapter (TU, or one coarse unit per hex: ``coarse_hex_fallback``)
      -> snow / firn / glacier ice reservoirs with a spin-up, a seeded paleo-ice scenario
         where documented, and a mass-conserving ice flow on the hex graph
      -> bioclimatic features (one water stress: reference AET/PET), aridity (diagnostic)
      -> PFT suitability, productivity proxy, vegetation fraction (bare stays bare)
      -> causal biome classes + limiting factors + habitat modifiers per unit
      -> whole-hex biome fractions (CSR), cover fractions, dominant class, climate zone

A new derived layer: climate, water, soils and relief are read, never changed. The potential
natural vegetation is classified (``potential_only``); there is no vegetation feedback.
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import asdict

import numpy as np

from .biomes import classify as bclass
from .biomes import pft
from .biomes.cryosphere import CryoParams, paleo_seed, run_cryosphere
from .biomes.registry import (
    BIOMES,
    COVER_CLASSES,
    INDEX,
    PFT_REGISTRY_VERSION,
    REGISTRY_VERSION,
)
from .project import BiomeData
from .stages.environment import MONTH_DAYS, PM
from .version import BIOMES_ALGORITHM_VERSION

N_B = len(BIOMES)
#: Köppen-Geiger main classes (diagnostic climate zone, not a biome)
KOPPEN = ("none", "Af", "Am", "Aw", "BWh", "BWk", "BSh", "BSk", "Cs", "Cw", "Cf", "Ds", "Dw",
          "Df", "ET", "EF")  # fmt: skip


def _key(*parts) -> str:
    return hashlib.sha256(json.dumps(parts, sort_keys=True, default=str).encode()).hexdigest()


def biome_stage_keys(config, upstream: dict) -> dict:
    return {"v50_biomes": _key("v50", BIOMES_ALGORITHM_VERSION, REGISTRY_VERSION,
                               PFT_REGISTRY_VERSION, upstream, asdict(config.biomes))}  # fmt: skip


def upstream_keys(project) -> dict:
    topo = getattr(project, "topo", None)
    return {
        "env": project.env.stage_keys,
        "topo": topo.stage_keys if topo is not None else {},
        "hydro": project.hydro.stage_keys,
    }


# -- adapter -------------------------------------------------------------------------------------
def _units_from_project(project):
    """Terrain units of v4.7, or one coarse unit per hex (flagged ``coarse_hex_fallback``)."""
    layout = project.layout
    n = layout.width * layout.height
    L, E, M = project.hydro.layers, project.env.layers, project.env.monthly
    topo = getattr(project, "topo", None)
    if topo is not None and topo.units["unit_hex"].size:
        U = topo.units
        d = {
            "hex": U["unit_hex"].astype(np.int64),
            "w": U["unit_weight_hex"].astype(np.float64),
            "type": U["unit_surface_type"].astype(np.uint8),
            "t_m": U["temperature_c"].astype(np.float64),
            "p_m": U["precip_mm"].astype(np.float64),
            "pet_m": U["pet_mm"].astype(np.float64),
            "aet_m": U["aet_mm"].astype(np.float64),
            "rad_m": U["radiation_mj_m2_day"].astype(np.float64),
            "z": U["z_surface_m"].astype(np.float64),
            "slope": U["slope_deg"].astype(np.float64),
            "aspect": U["aspect_deg"].astype(np.float64),
            "rank": U["elevation_rank"].astype(np.float64),
            "valley": U["valley_index"].astype(np.float64),
            "soil_depth": U["soil_depth_m"].astype(np.float64),
            "tree_score": np.nan_to_num(U["thermal_tree_score"].astype(np.float64)),
            "margin": np.nan_to_num(U["thermal_margin"].astype(np.float64), nan=-40.0),
            "gdd0": U["gdd0"].astype(np.float64),
            "gdd5": U["gdd5"].astype(np.float64),
            "gdd10": U["gdd10"].astype(np.float64),
            "cloud": U["cloud_potential_max"].astype(np.float64),
            "fog": U["fog_potential_mean"].astype(np.float64),
            "cap": U["cap_static"].astype(np.float64),
            "permafrost": U["permafrost_potential"].astype(np.float64),
            "quality": "terrain_units_v4_7",
        }
        # sea hexes without units: one whole-hex sea unit each (open ocean, never a forest)
        missing = np.nonzero(np.diff(topo.units["hex_offsets"]) == 0)[0]
        if missing.size:
            d = _append_sea(d, missing, project)
        order = np.argsort(d["hex"], kind="stable")
        return {k: (v[..., order] if isinstance(v, np.ndarray) else v) for k, v in d.items()}
    # coarse fallback: one unit per hex from the v4.5 hex fields
    ocean = L["ocean_mask"].ravel()
    lake = L["lake_id"].ravel() >= 0
    t_m = M["temperature_monthly_c"].reshape(12, n).astype(np.float64)
    p_m = np.nan_to_num(M["precip_monthly_mm"].reshape(12, n).astype(np.float64))
    pet_m = np.nan_to_num(M["pet_monthly_mm"].reshape(12, n).astype(np.float64))
    aet_m = np.nan_to_num(M["aet_monthly_mm"].reshape(12, n).astype(np.float64))
    lat = np.deg2rad(L["latitude_deg"].ravel().astype(np.float64))
    from .topoclimate.solar import REPRESENTATIVE_DOY, ra_analytic

    rad = np.stack([0.5 * ra_analytic(lat, REPRESENTATIVE_DOY[m, 1]) for m in range(12)])
    days = np.asarray(MONTH_DAYS, dtype=np.float64)[:, None]
    gdd = lambda b: (np.maximum(t_m - b, 0.0) * days).sum(axis=0)  # noqa: E731
    gst_ok = gdd(0.9) > 0
    return {
        "hex": np.arange(n),
        "w": np.ones(n),
        "type": np.where(ocean, 2, np.where(lake, 1, 0)).astype(np.uint8),
        "t_m": t_m, "p_m": p_m, "pet_m": pet_m, "aet_m": aet_m, "rad_m": rad,
        "z": L["elevation_m"].ravel().astype(np.float64),
        "slope": np.nan_to_num(E["slope_deg"].ravel().astype(np.float64)),
        "aspect": np.nan_to_num(E["aspect_deg"].ravel().astype(np.float64), nan=-1.0),
        "rank": np.full(n, 0.5), "valley": np.zeros(n),
        "soil_depth": np.nan_to_num(E["soil_depth_m"].ravel().astype(np.float64)),
        "tree_score": np.where(gst_ok, 1.0, 0.0),
        "margin": np.where(gst_ok, 1.0, -40.0),
        "gdd0": gdd(0.0), "gdd5": gdd(5.0), "gdd10": gdd(10.0),
        "cloud": np.zeros(n), "fog": np.zeros(n), "cap": np.zeros(n),
        "permafrost": np.nan_to_num(E["permafrost_potential"].ravel().astype(np.float64)),
        "quality": "coarse_hex_fallback",
    }  # fmt: skip


def _append_sea(d, hexes, project):
    k = hexes.size
    M = project.env.monthly
    n = project.layout.width * project.layout.height
    t = M["temperature_monthly_c"].reshape(12, n)[:, hexes].astype(np.float64)
    add = {
        "hex": hexes.astype(np.int64), "w": np.ones(k), "type": np.full(k, 2, dtype=np.uint8),
        "t_m": t, "p_m": np.zeros((12, k)), "pet_m": np.zeros((12, k)),
        "aet_m": np.zeros((12, k)), "rad_m": np.zeros((12, k)),
        "z": project.hydro.layers["elevation_m"].ravel()[hexes].astype(np.float64),
    }  # fmt: skip
    out = {}
    for key, v in d.items():
        if not isinstance(v, np.ndarray):
            out[key] = v
            continue
        if key in add:
            out[key] = np.concatenate([v, add[key]], axis=-1)
        else:
            fill = np.zeros(v.shape[:-1] + (k,), dtype=v.dtype)
            out[key] = np.concatenate([v, fill], axis=-1)
    return out


def _koppen(t_m, p_m, lat_deg):
    """Köppen-Geiger main class code (index into ``KOPPEN``) of 12 monthly values."""
    t_ann = t_m.mean(axis=0)
    p_ann = p_m.sum(axis=0)
    tw, tc = t_m.max(axis=0), t_m.min(axis=0)
    north = lat_deg >= 0
    summer = np.where(north[None, :], np.isin(np.arange(12), [3, 4, 5, 6, 7, 8])[:, None],
                      ~np.isin(np.arange(12), [3, 4, 5, 6, 7, 8])[:, None])  # fmt: skip
    p_sum = np.where(summer, p_m, 0).sum(axis=0)
    share = np.where(p_ann > 0, p_sum / np.maximum(p_ann, 1e-9), 0.5)
    pth = 20.0 * t_ann + np.where(share >= 0.7, 280.0, np.where(share < 0.3, 0.0, 140.0))
    out = np.zeros(t_ann.size, dtype=np.uint8)
    dry_s = np.where(summer, p_m, np.inf).min(axis=0)
    wet_w = np.where(~summer, p_m, 0).max(axis=0)
    dry_w = np.where(~summer, p_m, np.inf).min(axis=0)
    wet_s = np.where(summer, p_m, 0).max(axis=0)
    p_min = p_m.min(axis=0)
    ef = tw < 0.0
    et = (tw >= 0.0) & (tw < 10.0)
    b = (~(ef | et)) & (p_ann < 2.0 * pth)
    a = (~(ef | et | b)) & (tc >= 18.0)
    cfull = ~(ef | et | b | a)
    c_ = cfull & (tc > -3.0)
    d_ = cfull & ~c_
    out[a & (p_min >= 60)] = KOPPEN.index("Af")
    out[a & (p_min < 60) & (p_min >= 100 - p_ann / 25.0)] = KOPPEN.index("Am")
    out[a & (out == 0)] = KOPPEN.index("Aw")
    desert = b & (p_ann < pth)
    hot = t_ann >= 18.0
    out[desert & hot] = KOPPEN.index("BWh")
    out[desert & ~hot] = KOPPEN.index("BWk")
    out[b & ~desert & hot] = KOPPEN.index("BSh")
    out[b & ~desert & ~hot] = KOPPEN.index("BSk")
    s_dry = (dry_s < 40) & (dry_s < wet_w / 3.0)
    w_dry = dry_w < wet_s / 10.0
    for grp, (s_name, w_name, f_name) in ((c_, ("Cs", "Cw", "Cf")), (d_, ("Ds", "Dw", "Df"))):
        out[grp & s_dry] = KOPPEN.index(s_name)
        out[grp & ~s_dry & w_dry] = KOPPEN.index(w_name)
        out[grp & ~s_dry & ~w_dry] = KOPPEN.index(f_name)
    out[et] = KOPPEN.index("ET")
    out[ef] = KOPPEN.index("EF")
    return out


def _ice_forms(ice_hex_m, dry_hex, nbr, spacing_km, mountain_hex):
    """Connected ice bodies of the hex graph: 0 none, 1 mountain glacier, 2 ice cap, 3 sheet."""
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components

    n = ice_hex_m.size
    ice = (ice_hex_m >= 10.0) & dry_hex
    src, dst = [], []
    for k in range(3):
        j = nbr[:, k]
        ok = ice & (j >= 0)
        ok[ok] &= ice[j[ok]]
        i = np.nonzero(ok)[0]
        src.append(i)
        dst.append(j[i])
    s, t = np.concatenate(src), np.concatenate(dst)
    g = coo_matrix((np.ones(s.size), (s, t)), shape=(n, n))
    n_comp, lab = connected_components(g, directed=False)
    lab = np.where(ice, lab, -1)
    hex_area = spacing_km**2 * np.sqrt(3.0) / 2.0
    form = np.zeros(n, dtype=np.uint8)
    ice_id = np.full(n, -1, dtype=np.int32)
    bodies = []
    if ice.any():
        ids, inv, counts = np.unique(lab[ice], return_inverse=True, return_counts=True)
        area = counts * hex_area
        mnt = np.bincount(inv, weights=mountain_hex[ice].astype(float)) / counts
        f = np.where(area >= 150_000.0, 3, np.where((area >= 3_000.0) | (mnt < 0.5), 2, 1))
        form[ice] = f[inv]
        ice_id[ice] = inv.astype(np.int32)
        for b, (a, ff) in enumerate(zip(area, f, strict=True)):
            bodies.append({"ice_id": b, "area_km2": round(float(a), 1),
                           "form": ("", "glacier_mountain", "glacier_ice_cap",
                                    "glacier_ice_sheet")[int(ff)]})  # fmt: skip
    return form, ice_id, bodies


def run_biomes(config, geometry, project, report=None):  # noqa: C901 - one linear stage
    bp = config.biomes
    layout = geometry.layout
    shape = (layout.height, layout.width)
    n_hex = shape[0] * shape[1]
    rep = report or (lambda f: None)
    timings: dict = {}
    t0 = time.perf_counter()
    L, E, H = project.hydro.layers, project.env.layers, project.hist.layers
    d = _units_from_project(project)
    U = d["hex"].size
    uh = d["hex"]
    typ = d["type"]
    dry = typ == 0
    flat = lambda a: np.asarray(a).ravel()  # noqa: E731
    lat = flat(L["latitude_deg"]).astype(np.float64)
    nbr = _hex_neighbours(layout)
    timings["v50_inputs"] = round(time.perf_counter() - t0, 3)

    # -- cryosphere -------------------------------------------------------------------------------
    t0 = time.perf_counter()
    cp = CryoParams(spinup_years=bp.cryo_spinup_years, paleo_ice=bp.paleo_ice_initialization,
                    flow=bp.glacier_flow)  # fmt: skip
    cold_u = dry & (d["t_m"].min(axis=0) < 2.0)
    t_hex = project.env.monthly["temperature_monthly_c"].reshape(12, n_hex).astype(np.float64)
    p_hex = np.nan_to_num(project.env.monthly["precip_monthly_mm"].reshape(12, n_hex))
    snow_hex = (p_hex * (1.0 - np.clip((t_hex + 1.0) / 3.0, 0, 1))).sum(axis=0)
    land_hex = flat(L["land_mask"]) & (flat(L["lake_id"]) < 0)
    seed = paleo_seed(t_hex.mean(axis=0), snow_hex, land_hex, cp)
    cidx = np.nonzero(cold_u)[0]
    cryo_unit = {k: np.zeros(U) for k in ("snow_max_mm", "firn_swe_mm", "firn_age_yr",
                                          "ice_swe_mm", "ice_thickness_m", "annual_balance_mm",
                                          "annual_melt_mm", "annual_snowfall_mm",
                                          "ice_flux_net_mm_yr")}  # fmt: skip
    cryo_unit["snow_months"] = np.zeros(U, dtype=np.int8)
    cryo_unit["firn_density_kg_m3"] = np.full(U, np.nan)
    cryo_summary = {"spinup_years": cp.spinup_years, "mass_residual_mm_max": 0.0,
                    "ice_hexes": 0, "max_ice_thickness_m": 0.0, "seeded_paleo_ice_hexes": 0,
                    "flow_d_capped_edges": 0}  # fmt: skip
    ice_hex = np.zeros(n_hex)
    bal_hex = np.zeros(n_hex)
    if cidx.size:
        res = run_cryosphere(
            uh[cidx], d["w"][cidx], np.ones(cidx.size, dtype=bool), d["t_m"][:, cidx],
            d["p_m"][:, cidx], d["pet_m"][:, cidx], flat(L["elevation_m"]).astype(np.float64),
            nbr, layout.spacing_km, n_hex, cp, seed_ice_hex=seed if cp.paleo_ice else None,
            progress=lambda f: rep(0.02 + 0.85 * f),
        )  # fmt: skip
        for k, v in res.unit.items():
            if k in cryo_unit:
                cryo_unit[k][cidx] = v
        cryo_summary.update(res.summary)
        ice_hex = res.hex["ice_thickness_m"]
        bal_hex = res.hex["annual_balance_mm"]
    dry_hex_w = np.bincount(uh, weights=d["w"] * dry, minlength=n_hex)
    mountain_hex = flat(H["mountain_system_id"]) >= 0
    form_hex, ice_id_hex, bodies = _ice_forms(ice_hex, dry_hex_w > 0, nbr, layout.spacing_km,
                                              mountain_hex)  # fmt: skip
    timings["v50_cryosphere"] = round(time.perf_counter() - t0, 3)
    rep(0.88)

    # -- features ----------------------------------------------------------------------------------
    t0 = time.perf_counter()
    t_m, p_m, pet_m, aet_m = d["t_m"], d["p_m"], d["pet_m"], d["aet_m"]
    p_year, pet_year, aet_year = p_m.sum(0), pet_m.sum(0), aet_m.sum(0)
    alpha_m = np.where(pet_m > 1.0, np.clip(aet_m / np.maximum(pet_m, 1e-9), 0, 1), 1.0)
    alpha = np.where(pet_year > 10.0, np.clip(aet_year / np.maximum(pet_year, 1e-9), 0, 1), 1.0)
    warm = t_m > 5.0
    dry_months = ((alpha_m < 0.4) & warm).sum(axis=0).astype(np.float64)
    rank_t = np.argsort(np.argsort(-t_m, axis=0), axis=0)
    summer_dry = ((alpha_m < 0.4) & (rank_t < 6) & warm).sum(axis=0).astype(np.float64)
    ai = np.where(pet_year > bp.annual_pet_epsilon_mm, p_year / np.maximum(pet_year, 1e-9), np.nan)
    days = np.asarray(MONTH_DAYS, dtype=np.float64)[:, None]
    par_m = 0.48 * d["rad_m"] * days

    def hexv(name, default=0.0, src=None):
        a = (src or E).get(name)
        if a is None:
            return np.full(U, default)
        return np.nan_to_num(flat(a).astype(np.float64), nan=default)[uh]

    salinity = hexv("soil_salinity_proxy")
    waterlog = hexv("waterlogging_months")
    rocky = hexv("terrain_rocky_fraction")
    organic = hexv("soil_organic_potential")
    flood = hexv("floodplain_potential")
    marine_inf = hexv("coastal_marine_influence")
    parent = flat(E["parent_material_class"]).astype(np.int64)[uh]
    drain = flat(E["drainage_class"]).astype(np.int64)[uh]
    river = (flat(L["river_mask"]).astype(bool))[uh]
    endor = (flat(L["endorheic_mask"]).astype(bool))[uh]
    coastal = marine_inf > 0.5
    vage = np.nan_to_num(flat(H["volcanic_age_ma"]).astype(np.float64), nan=1e9)[uh]
    mountain = mountain_hex[uh] | (d["z"] > 2500.0)
    snow_months = np.maximum(cryo_unit["snow_months"].astype(np.float64), 0)
    # unit rock: the hex rock share sharpened by the unit slope; no soil = rock
    rock_u = np.clip(rocky + np.clip((d["slope"] - 25.0) / 25.0, 0, 1), 0, 1)
    rock_u = np.where(d["soil_depth"] <= 0.02, np.maximum(rock_u, 0.85), rock_u)
    ice_c = _smooth(cryo_unit["ice_swe_mm"], 150.0, 1500.0)
    firn_dense = np.nan_to_num(cryo_unit["firn_density_kg_m3"]) >= 550.0
    firn_c = (1 - ice_c) * _smooth(cryo_unit["firn_swe_mm"], 30.0, 400.0)
    snow_c = np.where(firn_dense, 0.0, firn_c)
    firn_c = np.where(firn_dense, firn_c, 0.0)
    hyper = np.nan_to_num(ai, nan=9.0) < 0.05
    salt_crust = np.where((salinity > 0.6) & hyper, 0.6, 0.0)
    sand = np.where((parent == PM["aeolian_sand"]) & (np.nan_to_num(ai, nan=9) < 0.2), 0.5, 0.0)
    open_frac = np.clip(1 - ice_c - firn_c - snow_c, 0, 1) * (1 - rock_u) * (1 - salt_crust)
    feats = pft.Features(
        tc=t_m.min(0), tw=t_m.max(0), t_m=t_m, gdd0=d["gdd0"], gdd5=d["gdd5"], gdd10=d["gdd10"],
        alpha=alpha, alpha_m=alpha_m, dry_months=dry_months, summer_dry=summer_dry, ai=ai,
        p_year=p_year, pet_year=pet_year, par_m=par_m, soil_depth=d["soil_depth"],
        waterlog=waterlog, salinity=salinity, snow_months=snow_months,
        tree_score=d["tree_score"], cloud=d["cloud"],
    )  # fmt: skip
    timings["v50_features"] = round(time.perf_counter() - t0, 3)

    # -- PFT, cover, classes (chunks) ----------------------------------------------------------
    t0 = time.perf_counter()
    P = len(pft.TRAITS)
    cover = np.zeros((P, U), dtype=np.float32)
    veg = np.zeros(U)
    prod_best = np.zeros(U)
    lai = np.zeros(U)
    biome = np.full(U, INDEX["unresolved"], dtype=np.uint16)
    limiting = np.zeros(U, dtype=np.int64)
    modifiers = np.zeros(U, dtype=np.int64)
    canopy = np.zeros(U)
    lai_p = np.array([t.lai for t in pft.TRAITS])
    di = np.nonzero(dry)[0]
    step = 60_000
    in_ice_body = form_hex[uh] > 0
    shadow = np.where(lat[uh] >= 0, np.abs(((d["aspect"] + 180) % 360) - 180) < 45,
                      np.abs(d["aspect"] - 180) < 45) & (d["slope"] > 15)  # fmt: skip
    # ombrotrophic proxy (poor drainage, rain surplus) and soft-sediment badlands proxy
    bog_rain = ((drain >= 4) & (alpha > 0.9) & (p_year > pet_year)).astype(float)
    badlands = ((parent == PM["sedimentary_bedrock"]) & (d["slope"] > 15)
                & (np.nan_to_num(ai, nan=9.0) < 0.3)).astype(float)  # fmt: skip
    for c0 in range(0, di.size, step):
        sel = di[c0 : c0 + step]
        f = pft.Features(**{k: (v[..., sel] if isinstance(v, np.ndarray) else v)
                            for k, v in feats.__dict__.items()})  # fmt: skip
        suit = pft.suitability(f)
        prod = pft.productivity(f, suit)
        v, cov = pft.allocate_cover(suit, prod, open_frac[sel], bp.pft_tau, bp.cover_v_ref)
        cover[:, sel] = cov
        veg[sel] = v
        prod_best[sel] = prod.max(axis=0)
        lai[sel] = (cov * lai_p[:, None]).sum(axis=0)
        cls = bclass.classify_units({
            "tc": f.tc, "tw": f.tw, "gdd0": f.gdd0, "gdd5": f.gdd5, "alpha": f.alpha,
            "dry_months": f.dry_months, "summer_dry": f.summer_dry, "veg_frac": v,
            "pft_cover": cov, "mountain": mountain[sel], "z": d["z"][sel], "ai": f.ai,
            "salinity": f.salinity, "waterlog": f.waterlog, "slope_deg": d["slope"][sel],
            "parent": parent[sel], "rocky": rock_u[sel], "thermal_margin": d["margin"][sel],
            "ice_cover": ice_c[sel], "firn_cover": firn_c[sel], "snow_cover": snow_c[sel],
            "ice_form": form_hex[uh[sel]], "ice_balance": cryo_unit["annual_balance_mm"][sel],
            "in_ice_body": in_ice_body[sel], "pm_young_volcanic": PM["young_volcanic"],
            "pm_aeolian": PM["aeolian_sand"], "volcanic_age": vage[sel], "rank": d["rank"][sel],
            "organic": organic[sel], "drainage_poor_rain": bog_rain[sel],
            "floodplain": flood[sel], "river": river[sel], "valley": d["valley"][sel],
            "p_year": f.p_year, "endorheic": endor[sel], "fog": d["fog"][sel],
            "coastal": coastal[sel], "badlands": badlands[sel],
            "gypsum": np.zeros(sel.size), "erosion": rock_u[sel],
            "permafrost": d["permafrost"][sel], "cloud": d["cloud"][sel],
            "soil_depth": f.soil_depth, "snow_months": f.snow_months, "cap": d["cap"][sel],
            "marine": marine_inf[sel], "shadow": shadow[sel],
        })  # fmt: skip
        biome[sel] = cls["biome"]
        limiting[sel] = cls["limiting"]
        modifiers[sel] = cls["modifiers"]
        canopy[sel] = cls["canopy_fraction"]
    timings["v50_pft_classes"] = round(time.perf_counter() - t0, 3)
    rep(0.96)

    # -- water units: lakes and the sea ----------------------------------------------------------
    t0 = time.perf_counter()
    depth = np.nan_to_num(flat(L["water_depth_m"]).astype(np.float64))
    trench_hex = flat(project.terrain.layers["trench_axis_mask"]) if (
        project.terrain is not None and "trench_axis_mask" in project.terrain.layers) else np.zeros(
        n_hex, dtype=bool)  # fmt: skip
    t_cold_hex = t_hex.min(axis=0)
    ai_hex = np.nan_to_num(flat(E["aridity_p_over_pet"]).astype(np.float64), nan=1.0)
    river_hex = flat(L["river_mask"]).astype(bool)
    river_mouth = np.zeros(n_hex, dtype=bool)
    ocean_hex = flat(L["ocean_mask"]).astype(bool)
    for k in range(6):
        j = nbr[:, k]
        ok = j >= 0
        river_mouth[ok] |= ocean_hex[ok] & river_hex[j[ok]]
    marine_cls = marine_classes(depth, trench_hex, t_cold_hex, river_mouth)
    sea_ice_months = (t_hex < -2.0).sum(axis=0).astype(np.uint8)
    endor_hex = flat(L["endorheic_mask"]).astype(bool)
    lake_cls = lake_classes(endor_hex, ai_hex)
    biome[typ == 2] = marine_cls[uh[typ == 2]]
    biome[typ == 1] = lake_cls[uh[typ == 1]]
    limiting[typ != 0] |= 1 << 14  # deep_water for land plants

    # -- cover fractions per unit (non-overlapping) ----------------------------------------------
    cov_u = np.zeros((len(COVER_CLASSES), U))
    ci = {name: i for i, name in enumerate(COVER_CLASSES)}
    open_ = np.clip(1 - ice_c - firn_c - snow_c, 0, 1)
    cov_u[ci["glacier_ice"]] = ice_c
    cov_u[ci["firn"]] = firn_c
    cov_u[ci["snow_perennial"]] = snow_c
    cov_u[ci["vegetation"]] = veg
    rest = np.clip(open_ - veg, 0, 1)
    rock_share = np.clip(rock_u, 0, 1)
    cov_u[ci["rock"]] = rest * rock_share
    rest2 = rest * (1 - rock_share)
    cov_u[ci["salt"]] = rest2 * np.clip(salt_crust / 0.6, 0, 1) * 0.6
    rest3 = rest2 - cov_u[ci["salt"]]
    cov_u[ci["sand"]] = rest3 * sand
    cov_u[ci["bare_soil"]] = rest3 - cov_u[ci["sand"]]
    water_u = typ != 0
    cov_u[:, water_u] = 0.0
    cov_u[ci["open_water"], water_u] = 1.0
    # sea-ice potential stays an overlay of the sea (never land, never cover)

    # -- hex aggregation: whole-hex biome fractions (CSR) -----------------------------------------
    w = d["w"]
    fractions, fr, hex_of, dominant = aggregate_fractions(uh, w, biome, n_hex)
    offsets = fractions["offsets"]
    _f2, _fr2, _h2, dominant_land = aggregate_fractions(uh[dry], w[dry], biome[dry], n_hex)

    def hmean(v, mask=None):
        ww = w if mask is None else w * mask
        den = np.bincount(uh, weights=ww, minlength=n_hex)
        num = np.bincount(uh, weights=ww * np.nan_to_num(v), minlength=n_hex)
        return np.where(den > 0, num / np.maximum(den, 1e-12), np.nan)

    def grid(a, dtype=np.float32):
        return np.ascontiguousarray(np.asarray(a).reshape(shape).astype(dtype))

    fams = list(dict.fromkeys(b.family for b in BIOMES))
    family_of = np.array([fams.index(b.family) for b in BIOMES], dtype=np.uint8)
    lim_hex = np.zeros(n_hex, dtype=np.int64)
    np.bitwise_or.at(lim_hex, uh[dry], limiting[dry])
    mod_hex = np.zeros(n_hex, dtype=np.int64)
    np.bitwise_or.at(mod_hex, uh[dry], modifiers[dry])
    lim_share = np.stack([np.nan_to_num(hmean(((limiting >> b) & 1).astype(float), dry))
                          for b in range(15)])  # fmt: skip
    main_lim = np.where(lim_share.max(0) > 0, lim_share.argmax(0) + 1, 0).astype(np.uint8)
    koppen = _koppen(t_hex, p_hex, lat)
    koppen = np.where(land_hex | (dry_hex_w > 0), koppen, 0).astype(np.uint8)
    realm = np.where(dry_hex_w >= 0.5, 1, np.where(hmean(typ == 1) >= 0.5, 2,
                     np.where(hmean(typ == 2) >= 0.5, 3, 4))).astype(np.uint8)  # fmt: skip
    layers = {
        "dominant_biome": grid(dominant, np.uint16),
        "dominant_land_biome": grid(dominant_land, np.uint16),
        "biome_family": grid(family_of[dominant], np.uint8),
        "realm": grid(realm, np.uint8),
        "climate_zone_koppen": grid(koppen, np.uint8),
        "vegetation_fraction": grid(hmean(veg, dry)),
        "tree_canopy_fraction": grid(hmean(canopy, dry)),
        "lai_proxy": grid(hmean(lai, dry)),
        "productivity_index": grid(hmean(prod_best, dry)),
        "bare_fraction": grid(hmean(cov_u[ci["bare_soil"]] + cov_u[ci["rock"]]
                                    + cov_u[ci["sand"]] + cov_u[ci["salt"]], dry)),
        "aridity_index": grid(hmean(np.nan_to_num(ai, nan=np.nan), dry)),
        "aridity_valid": grid(hmean(np.isfinite(ai).astype(float), dry) > 0.5, np.bool_),
        "moisture_index": grid(hmean(alpha, dry)),
        "dry_season_months": grid(hmean(dry_months, dry)),
        "growing_months": grid(hmean(((t_m > 5) & (alpha_m >= 0.4)).sum(0), dry)),
        "main_limiting_factor": grid(main_lim, np.uint8),
        "limiting_flags": grid(lim_hex, np.int64),
        "habitat_modifiers": grid(mod_hex, np.int64),
        "ice_thickness_m": grid(ice_hex),
        "ice_form": grid(form_hex, np.uint8),
        "ice_id": grid(ice_id_hex, np.int32),
        "ice_mass_balance_mm": grid(np.where(dry_hex_w > 0, bal_hex, np.nan)),
        "firn_swe_mm": grid(hmean(cryo_unit["firn_swe_mm"], dry)),
        "perennial_snow_fraction": grid(hmean(snow_c + firn_c, dry)),
        "glacier_fraction": grid(hmean(ice_c, dry)),
        "snow_months": grid(hmean(cryo_unit["snow_months"].astype(float), dry)),
        "ice_initial_condition": grid(seed > 0, np.bool_),
        "sea_ice_potential_months": grid(np.where(ocean_hex, sea_ice_months, 0), np.uint8),
        "glacier_state": grid(np.where(ice_hex >= 10, np.where(seed > 0, 2, 1),
                                       np.where(bal_hex > 0, 3, 0)), np.uint8),
    }  # fmt: skip
    for k, name in enumerate(COVER_CLASSES):
        layers[f"cover_{name}"] = grid(hmean(cov_u[k]))
    pft_hex = np.stack([hmean(cover[k], dry) for k in range(P)])
    for k, t in enumerate(pft.TRAITS):
        layers[f"pft_{t.id.lower()}"] = grid(pft_hex[k])
    units = {
        "unit_hex": uh.astype(np.int32),
        "unit_weight_hex": w.astype(np.float32),
        "unit_surface_type": typ,
        "biome": biome,
        "limiting_flags": limiting,
        "habitat_modifiers": modifiers,
        "vegetation_fraction": veg.astype(np.float32),
        "canopy_fraction": canopy.astype(np.float32),
        "productivity_index": prod_best.astype(np.float32),
        "pft_cover": cover,
        "cover_fractions": cov_u.astype(np.float32),
        "aridity_index": np.nan_to_num(ai, nan=-1.0).astype(np.float32),
        "moisture_index": alpha.astype(np.float32),
        "dry_season_months": dry_months.astype(np.float32),
        "z_m": d["z"].astype(np.float32),
    }
    for k, v in cryo_unit.items():
        units[f"cryo_{k}"] = np.asarray(v).astype(np.float32)
    timings["v50_aggregate"] = round(time.perf_counter() - t0, 3)
    # budgets
    hex_tot = np.bincount(uh, weights=w, minlength=n_hex)
    cov_sum = cov_u.sum(axis=0)
    fr_sum = np.bincount(hex_of, weights=fr, minlength=n_hex)
    has = np.diff(offsets) > 0
    shares = np.bincount(biome[dry].astype(np.int64), weights=w[dry], minlength=N_B)
    tot_dry = max(shares.sum(), 1e-12)
    top = np.argsort(-shares)[:25]
    summary = {
        "units": int(U),
        "dry_units": int(dry.sum()),
        "input_quality": d["quality"],
        "classes_present_land": int((shares > 0).sum()),
        "land_shares_top": {BIOMES[i].id: round(float(shares[i] / tot_dry), 4)
                            for i in top if shares[i] > 0},
        "fraction_sum_max_error": (
            float(np.abs(fr_sum[has] - hex_tot[has]).max()) if has.any() else 0.0
        ),
        "cover_sum_max_error": float(np.abs(cov_sum - 1.0).max()) if U else 0.0,
        "cryosphere": cryo_summary,
        "ice_bodies": bodies[:200],
        "ice_body_count": len(bodies),
        "vegetation_free_land_share": round(float((w * dry * (veg < 0.01)).sum() / max(
            (w * dry).sum(), 1e-12)), 4),
    }  # fmt: skip
    provenance = {
        "classification_version": BIOMES_ALGORITHM_VERSION,
        "biome_registry": REGISTRY_VERSION,
        "pft_registry": PFT_REGISTRY_VERSION,
        "profile": bp.profile,
        "mode": "equilibrium potential natural vegetation (potential_only)",
        "input_units": d["quality"],
        "water_stress": "one stress: reference AET/PET of the 4.5 kernel on the 4.7 unit; "
        "P/PET is diagnostic (UNEP classes), invalid where PET ~ 0",
        "productivity": "dimensionless index (PAR x light capture x T x water), not gC/m^2",
        "cryosphere": f"3 reservoirs, {cp.spinup_years} yr spin-up, regional_lite SIA flow "
        f"(D capped), paleo seed {'on' if cp.paleo_ice else 'off'}",
        "mangroves": "not claimed: no tide model (v6)",
        "coral_reef": "potential only: warm shallow sea, no SST / nutrients",
        "sea_ice": "potential months from air T < -2 degC over the sea, not modelled ice",
        "aquatic": "physical habitat only (depth classes), no marine ecology",
        "co2_ppm": bp.co2_ppm,
        "co2_effect": "metadata only (suitability proxy has no physiology)",
        "dunes_gypsum_badlands": "substrate proxies only (no aeolian / gypsum module)",
    }
    return {"layers": layers, "units": units, "fractions": fractions, "summary": summary,
            "provenance": provenance, "timings": timings}  # fmt: skip


def aggregate_fractions(unit_hex, weights, biome, n_hex):
    """Whole-hex biome fractions as CSR + the dominant class (ties: the lower code).

    ``fraction`` of a (hex, class) is the sum of the plan-area weights of its units; the
    fractions of a hex sum to the covered share of its area (1 for a fully sampled hex).
    """
    key = np.asarray(unit_hex, dtype=np.int64) * N_B + np.asarray(biome, dtype=np.int64)
    uk, inv = np.unique(key, return_inverse=True)
    fr = np.bincount(inv, weights=np.asarray(weights, dtype=np.float64))
    hex_of = (uk // N_B).astype(np.int64)
    offsets = np.searchsorted(hex_of, np.arange(n_hex + 1))
    codes = (uk % N_B).astype(np.uint16)
    fractions = {"offsets": offsets.astype(np.int64), "biome": codes,
                 "fraction": fr.astype(np.float32)}  # fmt: skip
    dominant = np.zeros(n_hex, dtype=np.uint16)
    if uk.size:
        order = np.lexsort((codes, -fr, hex_of))
        first = np.r_[True, hex_of[order][1:] != hex_of[order][:-1]]
        dominant[hex_of[order][first]] = codes[order][first]
    return fractions, fr, hex_of, dominant


def marine_classes(depth_m, trench, t_cold_c, river_mouth):
    """Physical marine habitat by depth (+ potential flags); never an invented ecology."""
    depth = np.asarray(depth_m, dtype=np.float64)
    out = np.full(depth.shape, INDEX["marine_pelagic_open"], dtype=np.uint16)
    out[depth > 3000] = INDEX["marine_abyssal_benthic"]
    out[(depth > 200) & (depth <= 3000)] = INDEX["marine_slope_benthic"]
    out[(depth > 50) & (depth <= 200)] = INDEX["marine_shelf_deep"]
    out[depth <= 50] = INDEX["marine_shelf_shallow"]
    out[(depth > 6000) & trench] = INDEX["marine_hadal_trench"]
    coral = (depth <= 50) & (t_cold_c >= 18.0) & ~river_mouth
    out[coral] = INDEX["marine_coral_reef_potential"]
    out[river_mouth] = INDEX["marine_estuary_transition"]
    out[(t_cold_c < -10.0) & (depth < 3000)] = INDEX["marine_polar_ice_margin"]
    return out


def lake_classes(endorheic, ai):
    """Lake habitat: fresh (drained), saline / seasonal (closed and dry), else unknown."""
    out = np.full(np.shape(endorheic), INDEX["water_lake_fresh_open"], dtype=np.uint16)
    out[endorheic & (ai < 0.3)] = INDEX["water_lake_saline"]
    out[endorheic & (ai < 0.08)] = INDEX["water_lake_seasonal"]
    out[endorheic & (ai >= 0.3)] = INDEX["lake_unknown_salinity"]
    return out


def _smooth(x, lo, hi):
    t = np.clip((np.asarray(x, dtype=np.float64) - lo) / (hi - lo), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def _hex_neighbours(layout):
    from .stages.raster import neighbor_index

    n = layout.width * layout.height
    nbr = np.full((n, 6), -1, dtype=np.int64)
    for k in range(6):
        idx, valid = neighbor_index(layout, k)
        nbr[:, k] = np.where(valid.ravel(), idx.ravel(), -1)
    return nbr


def run_v50(config, geometry, project, old_keys: dict, old_cache: dict, report):
    keys = biome_stage_keys(config, upstream_keys(project))
    if old_keys.get("v50_biomes") == keys["v50_biomes"] and "v50_biomes" in old_cache:
        data = old_cache["v50_biomes"]
        return data, {"v50_biomes": data}, ["v50_biomes"]
    report("v50_biomes", 0.998)
    from .progress import sub_report

    res = run_biomes(
        config, geometry, project, report=lambda f: report(sub_report("v50_biomes", f), 0.998)
    )
    warnings = []
    s = res["summary"]
    if s["input_quality"] == "coarse_hex_fallback":
        warnings.append("v5.0: no v4.7 terrain units - biomes from one coarse unit per hex "
                        "(coarse_hex_fallback)")  # fmt: skip
    data = BiomeData(layers=res["layers"], units=res["units"], fractions=res["fractions"],
                     summary=s, provenance=res["provenance"], stage_keys=keys,
                     warnings=warnings, timings=res["timings"])  # fmt: skip
    return data, {"v50_biomes": data}, []
