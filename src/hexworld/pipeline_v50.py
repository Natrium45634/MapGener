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
    MOD_BIT,
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
    volc = getattr(project, "volcanism", None)
    return {
        "env": project.env.stage_keys,
        "topo": topo.stage_keys if topo is not None else {},
        "hydro": project.hydro.stage_keys,
        "volcanism": volc.stage_keys if volc is not None else {},
        **({"earth": project.earth.stage_keys} if getattr(project, "earth", None) else {}),
        **(
            {"atmosphere": project.atmosphere.stage_keys}
            if getattr(project, "atmosphere", None) is not None
            else {}
        ),
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
    volc0 = getattr(project, "volcanism", None)
    if volc0 is not None and "volcanic_new_lake" in volc0.layers:
        # v5.3 G7: a lake dammed by lava / lahar / PDC deposits is water now (the rebuilt
        # post-volcanic drainage, not an overlay)
        newlake = volc0.layers["volcanic_new_lake"].ravel()
        typ = np.where((typ == 0) & newlake[d["hex"]], 1, typ).astype(typ.dtype)
        d["type"] = typ
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
    earth = getattr(project, "earth", None)
    sheet = earth.layers.get("land_ice_thickness_m") if earth is not None else None
    if sheet is not None:
        # v5.3 grounded ice sheets spun up to (or towards) equilibrium are the initial state
        seed = flat(sheet).astype(np.float64)
        cold_u = cold_u | (dry & (seed[uh] > 0))
        seed_source = "v53_ice_sheet"
    else:
        seed = paleo_seed(t_hex.mean(axis=0), snow_hex, land_hex, cp)
        seed_source = "v50_paleo_seed" if cp.paleo_ice else "none"
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
    cryo_summary["seed_source"] = seed_source
    ice_hex = np.zeros(n_hex)
    bal_hex = np.zeros(n_hex)
    if cidx.size:
        res = run_cryosphere(
            uh[cidx], d["w"][cidx], np.ones(cidx.size, dtype=bool), d["t_m"][:, cidx],
            d["p_m"][:, cidx], d["pet_m"][:, cidx], flat(L["elevation_m"]).astype(np.float64),
            nbr, layout.spacing_km, n_hex, cp,
            seed_ice_hex=seed if (cp.paleo_ice or sheet is not None) else None,
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
    coast_u = (
        flat(L["coast_type"]).astype(np.int64)[uh]
        if "coast_type" in L
        else np.zeros(U, dtype=np.int64)
    )
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
    badlands = ((parent == PM["sedimentary_bedrock"]) & (d["slope"] > 8)
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
            "marine": marine_inf[sel], "shadow": shadow[sel], "coast": coast_u[sel],
            "drain": drain[sel],
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
    earth_l = earth.layers if earth is not None else None
    if earth_l is not None and "sst_monthly_c" in earth_l:
        from .stages.coasts import COAST_TYPES

        sst_m = np.nan_to_num(earth_l["sst_monthly_c"].reshape(12, n_hex).astype(np.float64),
                              nan=-1.8)  # fmt: skip
        ct = flat(L["coast_type"]).astype(np.int64) if "coast_type" in L else np.zeros(n_hex, int)
        rocky_codes = [COAST_TYPES.index(k) for k in ("cliff", "fault_scarp", "fjord", "skerry",
                                                      "dalmatian", "ria", "volcanic")]  # fmt: skip
        rocky_land = np.isin(ct, rocky_codes)
        rocky_coast = np.zeros(n_hex, dtype=bool)
        for k in range(6):
            j = nbr[:, k]
            rocky_coast |= (j >= 0) & rocky_land[np.maximum(j, 0)]
        expo = np.zeros(n_hex)
        we = np.nan_to_num(flat(L["wave_exposure"]).astype(np.float64))
        for k in range(6):  # the exposure of the neighbouring coast
            j = nbr[:, k]
            expo = np.maximum(expo, np.where(j >= 0, we[np.maximum(j, 0)], 0.0))
        volc = getattr(project, "volcanism", None)
        hk = flat(volc.layers["hydrothermal_kind"]) if volc is not None else np.zeros(n_hex)
        ice_m = earth_l["sea_ice_concentration_monthly"].reshape(12, n_hex).astype(np.float64)
        marine_cls = marine_rare_classes(
            marine_cls, depth, sst_m.mean(axis=0), sst_m.max(axis=0), expo, rocky_coast,
            flat(earth_l["coastal_upwelling"]).astype(bool), hk,
            flat(earth_l.get("shelf_thickness_m", np.zeros(n_hex))).astype(np.float64),
            flat(earth_l["sea_ice_type"]) == 2, ice_m.max(axis=0),
        )  # fmt: skip
    marine_layers = {}
    if earth_l is not None and "sst_monthly_c" in earth_l:
        from .biomes import marine as mar

        land_nb = ~ocean_hex
        coast_hex = np.zeros(n_hex, dtype=bool)
        warm_kind = np.zeros(n_hex, dtype=bool)
        from .stages.coasts import COAST_TYPES

        soft_codes = [
            COAST_TYPES.index(k)
            for k in (
                "mangrove",
                "tidal_flat",
                "delta",
                "estuary",
                "barrier_lagoon",
                "sabkha",
                "liman",
            )
        ]
        soft_land = np.isin(ct, soft_codes)
        for k in range(6):
            j = nbr[:, k]
            ok = j >= 0
            coast_hex |= ocean_hex & ok & land_nb[np.maximum(j, 0)]
            warm_kind |= ok & soft_land[np.maximum(j, 0)]
        up = earth_l["upwelling_m_day"].reshape(4, n_hex).astype(np.float64).mean(axis=0)
        marine_layers = mar.habitats(
            depth, ocean_hex, sst_m, ice_m, up, river_mouth, trench_hex, coast_hex, warm_kind, nbr
        )
        marine_layers["marine_ecoregion_id"] = mar.ecoregions_fast(
            ocean_hex, sst_m.mean(axis=0), ice_m.max(axis=0), depth, shape
        ).ravel()
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
    # tidal fringes (mangroves, salt marshes, coastal sabkhas): a belt a few km wide along the
    # coast of the hex, taken out of its dry units - a 40 km hex is not all mangrove
    f_uh, f_w, f_b, fringe_info = tidal_fringe(
        uh, w, biome, dry, d["slope"], flat(L["coast_type"]) if "coast_type" in L else None,
        t_hex.min(axis=0), t_hex.max(axis=0), ai_hex, layout.spacing_km, n_hex,
    )  # fmt: skip
    f_dry0 = np.concatenate([dry, np.ones(f_uh.size - uh.size, dtype=bool)])
    volc = getattr(project, "volcanism", None)
    volc_info = {"hexes": 0}
    if volc is not None:
        f_uh, f_w, f_b, volc_info = volcanic_fringe(
            f_uh, f_w, f_b, f_dry0, flat(volc.layers["volcanic_barren_fraction"]), n_hex
        )
    dune_info = {"hexes": 0}
    if earth is not None and "dune_type" in earth.layers:
        f_dry1 = np.concatenate([dry, np.ones(f_uh.size - uh.size, dtype=bool)])
        dune = flat(earth.layers["dune_type"]) == 5
        sand = np.nan_to_num(flat(earth.layers["sand_thickness_m"]).astype(np.float64))
        share = np.where(dune, np.clip(0.08 + 0.04 * sand, 0.0, 0.25), 0.0)
        f_uh, f_w, f_b, dune_info = volcanic_fringe(f_uh, f_w, f_b, f_dry1, share, n_hex,
                                                    biome_id="coastal_dune_vegetation")  # fmt: skip
    fractions, fr, hex_of, dominant = aggregate_fractions(f_uh, f_w, f_b, n_hex)
    offsets = fractions["offsets"]
    f_dry = np.concatenate([dry, np.ones(f_uh.size - uh.size, dtype=bool)])
    _f2, _fr2, _h2, dominant_land = aggregate_fractions(f_uh[f_dry], f_w[f_dry], f_b[f_dry], n_hex)

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
    if volc is not None:  # v5.1 habitat modifiers
        V = volc.layers
        heat = flat(V["geothermal_heat_mw"]) > 0
        kind = flat(V["hydrothermal_kind"])
        mod_hex |= np.where(heat & land_hex, MOD_BIT["geothermal_habitat"], 0)
        mod_hex |= np.where(flat(V["degassing_stress_fraction"]) > 0,
                            MOD_BIT["volcanic_degassing_stress"], 0)  # fmt: skip
        mod_hex |= np.where(flat(V["volcanic_ash_stress"]) > 0.2,
                            MOD_BIT["volcanic_ash_young"], 0)  # fmt: skip
        mod_hex |= np.where(~land_hex & (kind >= 1) & (kind <= 3),
                            MOD_BIT["hydrothermal_vent_field"], 0)  # fmt: skip
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
    for k, v in marine_layers.items():  # TZ 5.4.1 marine habitat axes (potential)
        dt = v.dtype if v.dtype in (np.uint8, np.int32) else np.float32
        layers[k] = grid(v, dt)
    # -- v5.3 actual ecosystems: growth, fire, volcanic resets, succession -------------------
    eco_summary = {"mode": "static_potential_only"}
    if (
        getattr(config.biomes, "ecosystem_mode", "static_potential_only")
        == "dynamic_vegetation_annual"
    ):
        from .biomes import ecology as eco

        t0 = time.perf_counter()
        fam_of = [BIOMES[int(k)].family for k in dominant]
        years = int(config.biomes.ecosystem_years)
        dist = []
        volc_e = getattr(project, "volcanism", None)
        if volc_e is not None:
            V = volc_e.layers
            f = np.clip(flat(V["lava_fraction"]) + flat(V["pdc_deposit_fraction"]), 0.0, 1.0)
            age = np.nan_to_num(flat(V["lava_youngest_years_bp"]).astype(np.float64), nan=1e9)
            hit = (f > 0) & (age < years)
            for yr in np.unique((years - age[hit]).astype(int)):
                sel = hit & ((years - age).astype(int) == yr)
                dist.append((int(yr), np.nonzero(sel)[0], f[sel]))
        atm = getattr(project, "atmosphere", None)
        storm_damage = 0
        if atm is not None and "weather_canopy_damage_fraction" in atm.layers:
            # v5.4c: the weather year's windthrow is the last simulated year (keeps the soil)
            dmg = flat(atm.layers["weather_canopy_damage_fraction"]).astype(np.float64)
            sel = dmg > 0
            if sel.any():
                dist.append((years - 1, np.nonzero(sel)[0], dmg[sel], False))
                storm_damage = int(sel.sum())
        soil = flat(E["soil_present"]).astype(bool) if "soil_present" in E else np.ones(n_hex, bool)
        seed_int = int(hashlib.sha256(str(config.seed).encode()).hexdigest()[:15], 16)
        er = eco.run(fam_of, t_hex.mean(axis=0), p_hex.sum(axis=0),
                     np.nan_to_num(layers["dry_season_months"].ravel()), soil, nbr, seed_int,
                     years=years, disturbances=dist)  # fmt: skip
        for k in (
            "actual_biomass_kg_c_m2",
            "potential_biomass_kg_c_m2",
            "stand_age_yr",
            "burn_fraction_last_decade",
            "carbon_emitted_kg_c_m2_yr",
            "fuel_load_kg_c_m2",
        ):
            layers[k] = grid(er[k])
        layers["succession_stage"] = grid(er["succession_stage"], np.uint8)
        layers["fire_return_interval_yr"] = grid(np.minimum(er["fire_return_interval_yr"], 1e6))
        eco_summary = er["summary"]
        eco_summary["weather_storm_damage_hexes"] = storm_damage
        if atm is not None and "weather_water_stress_change" in atm.layers:
            # T68: one weather year updates the drought stress of the actual state
            # (D = D e^(-1/tau) + stress (1 - e^(-1/tau)), tau = 2 yr); no class change
            ds = np.nan_to_num(flat(atm.layers["weather_water_stress_change"]).astype(float))
            layers["actual_drought_stress"] = grid(
                np.clip(ds, 0.0, 1.0) * (1.0 - np.exp(-1.0 / 2.0))
            )
        timings["v50_ecosystems"] = round(time.perf_counter() - t0, 3)
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
        "volcanic_fringe": volc_info,
        "coastal_dune_fringe": dune_info,
        "ecosystems": eco_summary,
        "class_audit": _class_audit(shares, tot_dry, t_hex, ai_hex, E, L, land_hex, project),
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


#: Width of the tidal fringe, km: mangrove coast, other sheltered tropical coasts, tidal-flat
#: (wadden) coast, other sheltered temperate coasts, sabkha.
FRINGE_KM = {"mangrove": 4.0, "soft_tropical": 1.5, "tidal_flat": 2.0, "soft_temperate": 0.6,
             "sabkha": 3.0}  # fmt: skip
_SOFT_COASTS = ("delta", "estuary", "liman", "tidal_flat", "barrier_lagoon")


def tidal_fringe(uh, w, biome, dry, slope, coast, tc_hex, tw_hex, ai_hex, spacing_km, n_hex):
    """Extra units of mangrove / salt marsh / sabkha on coast hexes (and the rest shrunk).

    The coast type of the hex (v4.1: tides, waves, sediment, sea temperature) picks the
    fringe; its share of the hex is ``belt width x coast length / hex area`` (coast length
    ~ one hex spacing), at most half of the hex's dry land, and only where the hex has
    low, gentle ground (a unit with slope < 3 deg). Returns the extended unit arrays.
    """
    if coast is None:
        return uh, w, biome, {"hexes": 0}
    from .stages.coasts import COAST_TYPES

    code = {name: i for i, name in enumerate(COAST_TYPES)}
    coast = np.asarray(coast).astype(np.int64)
    soft = np.isin(coast, [code[k] for k in _SOFT_COASTS])
    tropical = (tc_hex >= 18.0) & (tw_hex >= 24.0)
    kind = np.full(n_hex, -1)
    belt = np.zeros(n_hex)
    mang = (coast == code["mangrove"]) | (soft & tropical)
    kind[mang] = 0
    belt[mang] = np.where(coast[mang] == code["mangrove"], FRINGE_KM["mangrove"],
                          FRINGE_KM["soft_tropical"])  # fmt: skip
    marsh = ~mang & ((coast == code["tidal_flat"]) | soft) & (tw_hex > 6.0) & (ai_hex > 0.05)
    kind[marsh] = 1
    belt[marsh] = np.where(coast[marsh] == code["tidal_flat"], FRINGE_KM["tidal_flat"],
                           FRINGE_KM["soft_temperate"])  # fmt: skip
    sab = (coast == code["sabkha"]) & (ai_hex < 0.2) & (tw_hex >= 22.0)
    kind[sab] = 2
    belt[sab] = FRINGE_KM["sabkha"]
    gentle = np.zeros(n_hex, dtype=bool)
    np.logical_or.at(gentle, uh[dry], slope[dry] < 3.0)
    dry_w = np.bincount(uh[dry], weights=w[dry], minlength=n_hex)
    hex_area = 0.5 * np.sqrt(3.0) * spacing_km**2
    share = np.minimum(belt * spacing_km / hex_area, 0.5) * dry_w
    take = (kind >= 0) & gentle & (dry_w > 0) & (share > 0)
    hx = np.nonzero(take)[0]
    if hx.size == 0:
        return uh, w, biome, {"hexes": 0}
    # shrink the dry units of those hexes, then add the fringe units
    scale = np.ones(n_hex)
    scale[hx] = 1.0 - share[hx] / dry_w[hx]
    w2 = np.where(dry, w * scale[uh], w)
    new_uh, new_w, new_b = [], [], []
    for k, (a, b_) in enumerate((("forest_tropical_mangrove", "wetland_mangrove_intertidal"),
                                 ("wetland_saltmarsh_tidal", None),
                                 ("desert_hot_salt_flat", None))):  # fmt: skip
        sel = hx[kind[hx] == k]
        if sel.size == 0:
            continue
        if b_ is None:
            new_uh.append(sel)
            new_w.append(share[sel])
            new_b.append(np.full(sel.size, INDEX[a]))
        else:  # mangrove forest behind its intertidal fringe (2 : 1)
            new_uh += [sel, sel]
            new_w += [share[sel] * 2 / 3, share[sel] / 3]
            new_b += [np.full(sel.size, INDEX[a]), np.full(sel.size, INDEX[b_])]
    info = {
        "hexes": int(hx.size),
        "mangrove": int((kind[hx] == 0).sum()),
        "salt_marsh": int((kind[hx] == 1).sum()),
        "sabkha": int((kind[hx] == 2).sum()),
    }
    return (
        np.concatenate([uh, *new_uh]),
        np.concatenate([w2, *new_w]),
        np.concatenate([biome, *[b.astype(biome.dtype) for b in new_b]]),
        info,
    )


def _class_audit(shares, tot_dry, t_hex, ai_hex, E, L, land_hex, project):
    from .biomes.audit import class_audit, envelopes_from

    lw = land_hex
    wl = np.asarray(E["waterlogging_months"]).ravel() if "waterlogging_months" in E else 0
    env = envelopes_from(
        t_hex.min(axis=0)[lw], t_hex.max(axis=0)[lw], ai_hex[lw],
        np.asarray(wl)[lw] if np.ndim(wl) else np.zeros(int(lw.sum())),
        np.asarray(L["elevation_m"]).ravel()[lw] - project.hydro.sea_level_m,
        (np.asarray(L["coast_type"]).ravel() > 0) if "coast_type" in L else np.zeros(1, bool),
        getattr(project, "volcanism", None) is not None,
    )  # fmt: skip
    return class_audit(shares / max(tot_dry, 1e-12), env)


def volcanic_fringe(uh, w, biome, dry, barren_hex, n_hex, biome_id="barren_volcanic_recent"):
    """v5.1 fresh lava / PDC / lahar / thick-ash ground as a share of the hex's dry land.

    The share (``volcanic_barren_fraction`` of v5.1, already reduced by the vegetation
    recovery of its age and climate) is taken out of the dry units of the hex and given
    to one ``barren_volcanic_recent`` unit: a 4 km flow is a few per cent of a 25 km hex,
    not the hex.
    """
    barren = np.clip(np.nan_to_num(np.asarray(barren_hex, dtype=np.float64)), 0.0, 0.95)
    dry_w = np.bincount(uh[dry], weights=w[dry], minlength=n_hex)
    share = barren * dry_w
    hx = np.nonzero((share > 1e-4) & (dry_w > 0))[0]
    if hx.size == 0:
        return uh, w, biome, {"hexes": 0}
    scale = np.ones(n_hex)
    scale[hx] = 1.0 - share[hx] / dry_w[hx]
    w2 = np.where(dry, w * scale[uh], w)
    return (
        np.concatenate([uh, hx]),
        np.concatenate([w2, share[hx]]),
        np.concatenate([biome, np.full(hx.size, INDEX[biome_id], biome.dtype)]),
        {"hexes": int(hx.size), "area_share_sum": float(share[hx].sum())},
    )


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


def marine_rare_classes(out, depth, sst_ann, sst_max, exposure, rocky_coast, upwelling,
                        hydro_kind, shelf, multiyear, ice_max):  # fmt: skip
    """v5.3 marine classes with their own gates (TZ 5.3 sec. 15); arrays per hex.

    * kelp: shallow (<= 40 m, light), hard ground (a rocky coast next to it), cool water
      (annual SST 4-18, warmest month < 21) with nutrients (exposure or upwelling);
    * seagrass: very shallow (<= 20 m), soft sheltered ground (exposure < 0.4), not rocky,
      warmest month > 10 degC, no long ice cover;
    * hydrothermal field: an active sea-floor vent field of v5.1 (source needed, not depth);
    * floating ice shelf and multiyear sea ice: physical cover states of v5.3.
    """
    o = out.copy()
    d = np.asarray(depth, dtype=np.float64)
    ice_free = ice_max < 0.5
    kelp = (d <= 40.0) & rocky_coast & (sst_ann >= 4.0) & (sst_ann <= 18.0) & (sst_max < 21.0) & (
        (exposure > 0.4) | upwelling) & ice_free  # fmt: skip
    grass = (d <= 20.0) & ~rocky_coast & (exposure < 0.4) & (sst_max > 10.0) & ice_free & ~kelp
    o[kelp] = INDEX["marine_kelp_forest"]
    o[grass] = INDEX["marine_seagrass_meadow"]
    o[(hydro_kind >= 1) & (hydro_kind <= 3)] = INDEX["marine_hydrothermal_field"]
    o[multiyear] = INDEX["marine_sea_ice_multiyear"]
    o[shelf > 0] = INDEX["marine_ice_shelf_floating"]
    return o


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
