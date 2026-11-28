"""Primary production of TZ 5.5 §8 (G2): light-use efficiency on land, in fresh water and in
the sea, with explicit light, temperature, water and nutrient limits, canopy layers and
plant carbon stocks.

    APAR   = PAR · (1 − exp(−k LAI))            (per layer; the layer below gets the rest)
    GPP    = ε · APAR · f_T · f_W · f_nutr      gC/m²
    NPP    = CUE · GPP                          (autotrophic respiration once, B23)
    B*_p   = NPP · a_p / λ_p                    (pool p ∈ leaf, wood, root, repro; N06)
    f_nutr = min(f_N, f_P),  f_X = [X/(X+K)] / [X_ref/(X_ref+K)] ≤ 1

All producer NPP is gC per m² of the habitat patch (land ground / inland water / sea), not
of the hex. Every number is a provisional prior (``traits_v1.json``), not a calibration.
"""

from __future__ import annotations

import math

import numpy as np

from .niche import combine, trapezoid
from .registry import TERRESTRIAL_LAYERS, load

#: reference supply at which the nutrient factor is 1 (typical fertile soil of 5.2) [К]
N_REF_KG_M2_YR = 0.04
P_REF_KG_M2 = 0.0015
#: maximum cover of a layer and the shading of the layers below (fractions) [К]
CMAX = {"canopy": 0.95, "shrub": 0.7, "ground": 0.95, "epiphyte": 0.3}
#: ecosystem water-use efficiency of GPP, gC per kg of actual ET [S: 2–4 for most biomes]
WUE_GC_PER_KG = 3.5
#: share of the hex's coastal ground that is a tidal fringe (mangrove gate) [К]
TIDAL_FRINGE = 0.08
#: lake ice is opaque below this monthly air temperature, °C [К]
LAKE_ICE_T = -2.0
#: light transmission of sea ice (applied once, B52) and the share absorbed by ice algae
SEA_ICE_TRANSMISSION = 0.1
ICE_ALGAE_ABSORB = 0.05
#: share of the exported carbon of the phytoplankton (pe-ratio) and the Martin exponent
EXPORT_RATIO = 0.2
MARTIN_B = 0.858
#: explicit chemosynthetic subsidy of a vent field per MW of hydrothermal heat, gC/yr [К]
CHEMO_GC_PER_MW_YR = 5.0e4
AXIS_VENT_MW = 10.0
#: edible share carved off the root NPP for the root symbionts
ROOT_EXUDATE = 0.1
#: nectar share of the reproductive NPP of animal-pollinated plants
NECTAR_SHARE = 0.1
#: dry mass / carbon, wood density for the timber volume
C_PER_DRY = 0.47
WOOD_DENSITY_T_M3 = 0.5


CRYPTOGAMS = ("moss", "lichen")


def _share(s):
    """Shares of a layer among its groups ∝ S⁴ (the best-suited dominate)."""
    w4 = s**4
    tot = w4.sum(axis=0)
    return np.where(tot > 0, w4 / np.maximum(tot, 1e-30), 0.0)


def _growing_degree_days(t_m, days):
    return np.sum(np.maximum(t_m - 5.0, 0.0) * days[:, None], axis=0)


def nutrient_factor(n_supply, p_supply, kn, kp):
    """f_nutr = min(f_N, f_P), normalised to 1 at the reference supply (B20)."""
    fn = (n_supply / (n_supply + kn)) / (N_REF_KG_M2_YR / (N_REF_KG_M2_YR + kn))
    fp = (p_supply / (p_supply + kp)) / (P_REF_KG_M2 / (P_REF_KG_M2 + kp))
    return np.clip(np.minimum(fn, fp), 0.0, 1.0)


def lue_npp(par, fapar, lue, f_t=1.0, f_w=1.0, f_n=1.0, cue=0.5):
    """The LUE equation for one interval (N04); returns (GPP, NPP) in gC/m²."""
    gpp = lue * par * fapar * f_t * f_w * f_n
    return gpp, cue * gpp


def fapar(lai, k=0.5):
    return 1.0 - np.exp(-k * np.asarray(lai, dtype=np.float64))  # N05


def equilibrium_stock(npp, turnover):
    """B* = NPP / λ for a simple uncapped turnover model (N06)."""
    return np.asarray(npp, dtype=np.float64) / turnover


def niche_score(niche: dict, env: dict, weights: dict | None = None, liebig: bool = False):
    """Soft combination of the trapezoid scores of the variables present in ``env``; with
    ``liebig`` the most limiting score instead (used for structural cover, not for S)."""
    sc, w = [], []
    for var, pts in niche.items():
        if var not in env or env[var] is None:
            continue
        sc.append(trapezoid(env[var], *pts))
        w.append((weights or {}).get(var, 1.0))
    if not sc:
        return np.ones_like(next(iter(env.values())), dtype=np.float64)
    if liebig:
        return np.min(np.stack(sc), axis=0)
    return combine(sc, w)


# -- land -------------------------------------------------------------------------------------
def land_production(inp, frac, *, n_fix_supply=None, pollinators=None, iterations: int = 2):
    """Producers of the land ground on every land hex, averaged over the microsite bins.

    Returns a dict of per-group arrays over the land subset ``idx`` plus pool totals."""
    reg = load()
    f = inp.f
    ground = frac["land_ground"]
    idx = np.nonzero(ground > 0)[0]
    n = idx.size
    days = inp.days
    groups = [g for g in reg.groups.values() if g.is_producer and g.traits["layer"]
              in TERRESTRIAL_LAYERS]  # fmt: skip
    G = len(groups)
    par = f["par_m"][:, idx]
    pet, aet = f["pet_m"][:, idx], f["aet_m"][:, idx]
    with np.errstate(invalid="ignore", divide="ignore"):
        wr = np.where(pet > 0, aet / pet, np.nan)  # PET = 0: undefined, not 0 (B21)
    mi, ph, ec = f["mi"][idx], f["ph"][idx], f["ec_ds_m"][idx]
    p_ann = f["precip_m"][:, idx].sum(axis=0)
    n_sup = np.nan_to_num(f["n_min_kg_m2_yr"][idx], nan=np.nan)
    if n_fix_supply is not None:
        n_sup = n_sup + n_fix_supply[idx]
    p_sup = f["p_lab_kg_m2"][idx]
    soil_ok = f["soil_present"][idx] | (np.nan_to_num(f["soil_depth_m"][idx]) >= 0.1)
    wet = (f["wetness"][idx] >= 0.5) | (f["waterlogging_months"][idx] >= 1)
    tidal = f["coast_land"][idx] & (f["tidal_range_m"][idx] >= 0.3)
    substrate = np.clip(f["cover_rock"][idx] + f["cover_salt"][idx] + f["cover_sand"][idx]
                        + f["cover_snow_perennial"][idx], 0.0, 1.0)  # fmt: skip
    young_lava = f["lava_fraction"][idx] * (f["lava_youngest_years_bp"][idx] < 300.0)
    veg_share = np.clip(1.0 - substrate - young_lava, 0.0, 1.0)  # rooted plants
    out = {
        "idx": idx, "groups": [g.id for g in groups],
        "npp": np.zeros((G, n)), "gpp": np.zeros((G, n)), "apar": np.zeros((G, n)),
        "cover": np.zeros((G, n)), "lai": np.zeros((G, n)), "suit": np.zeros((G, n)),
        "npp_month": np.zeros((12, n)), "limit": np.zeros((4, n)),
    }  # fmt: skip
    t0 = f["t_air_m"][:, idx]
    for off, wbin in f["microsites"]:
        t = t0 + off[idx][None, :]
        env = {"tc": t.min(axis=0), "tw": t.max(axis=0), "mi": mi,
               "gdd": _growing_degree_days(t, days), "ph": ph, "sal": ec}  # fmt: skip
        S = np.zeros((G, n))
        Smin = np.zeros((G, n))
        for gi, g in enumerate(groups):
            gates = np.ones(n)
            gw = g.traits["gates"]
            if "root_zone" in gw:
                gates = gates * soil_ok
            if "wet_ground" in gw:
                gates = gates * wet
            if "tidal_coast" in gw:
                gates = gates * tidal
            S[gi] = np.nan_to_num(niche_score(g.traits["niche"], env)) * gates
            Smin[gi] = np.nan_to_num(niche_score(g.traits["niche"], env, liebig=True)) * gates
        # covers by layer
        cover = np.zeros((G, n))
        layer_cover = {}
        for layer in ("canopy", "shrub", "ground", "epiphyte"):
            sel = [i for i, g in enumerate(groups) if g.traits["layer"] == layer]
            if not sel:
                continue
            avail = veg_share.copy()
            if layer == "canopy":  # rainfall cap of woody cover (Sankaran et al. 2005) [S]
                avail = avail * np.clip(0.0014 * (p_ann - 100.0), 0.0, 1.0)
            if layer == "shrub":
                avail = avail * np.clip(0.002 * (p_ann - 50.0), 0.0, 1.0)
                avail = avail * (1.0 - 0.6 * layer_cover.get("canopy", 0.0))
            elif layer == "ground":
                avail = np.clip(veg_share * (1.0 - 0.5 * layer_cover.get("canopy", 0.0)
                                             - 0.3 * layer_cover.get("shrub", 0.0)), 0.0, 1.0)
                avail = avail * np.clip(p_ann / 250.0, 0.05, 1.0)  # sparse cover in deserts
            elif layer == "epiphyte":
                avail = layer_cover.get("canopy", np.zeros(n))
            s = Smin[sel]  # structure follows the most limiting factor (Liebig)
            if layer == "canopy":
                s = s * np.where(np.array([groups[i].id == "tree_mangrove" for i in sel])[:, None],
                                 TIDAL_FRINGE, 1.0)  # fmt: skip
            union = 1.0 - np.prod(1.0 - s * CMAX[layer], axis=0)
            if layer == "ground":
                # vascular plants first; mosses and lichens take the gaps and bare rock
                crypt = np.array([groups[i].id in CRYPTOGAMS for i in sel])
                vs = [i for i, c in zip(sel, crypt, strict=True) if not c]
                cs = [i for i, c in zip(sel, crypt, strict=True) if c]
                cover[vs] = _share(Smin[vs]) * (1.0 - np.prod(1.0 - Smin[vs] * CMAX[layer],
                                                               axis=0)) * avail  # fmt: skip
                gap = np.clip(avail - cover[vs].sum(axis=0), 0.0, 1.0) + substrate * 0.5
                cover[cs] = _share(Smin[cs]) * (1.0 - np.prod(1.0 - Smin[cs] * CMAX[layer],
                                                               axis=0)) * np.clip(gap, 0, 1)
            else:
                cover[sel] = _share(s) * union * avail
            layer_cover[layer] = cover[sel].sum(axis=0)
        # canopy-dependent groups need a canopy
        for i, g in enumerate(groups):
            if "canopy" in g.traits["gates"]:
                cover[i] = np.minimum(cover[i], layer_cover.get("canopy", 0.0))
        lai_cap = np.array([g.traits["lai_max"] for g in groups])[:, None] * cover
        lai = lai_cap * S
        fn = np.stack([nutrient_factor(n_sup, p_sup, g.traits["kn"], g.traits["kp"])
                       for g in groups])  # fmt: skip
        t_act = [g.traits["t_act"] for g in groups]
        w_act = [g.traits["w_act"] for g in groups]

        def ft_of(m):
            return np.stack([trapezoid(t[m], *ta) for ta in t_act])

        def fw_of(m):
            frozen = t[m] <= 0
            return np.stack([np.where(np.isnan(wr[m]), np.where(frozen, 1.0, np.nan),
                                      trapezoid(wr[m], *wa)) for wa in w_act])

        lue = np.array([g.traits["lue"] for g in groups])[:, None]
        cue = np.array([g.traits["cue"] for g in groups])[:, None]
        kx = np.array([g.traits["k"] for g in groups])[:, None]
        deciduous = np.array([g.traits["turn"]["leaf"] >= 1.0 for g in groups])[:, None]
        for _it in range(iterations + 1):
            gpp = np.zeros((G, n))
            ap = np.zeros((G, n))
            gm = np.zeros((12, n))
            ft_acc = np.zeros((G, n))
            fw_acc = np.zeros((G, n))
            for m in range(12):
                ft_m, fw_m = ft_of(m), fw_of(m)
                ft_acc += ft_m / 12.0
                fw_acc += np.nan_to_num(fw_m) / 12.0
                phen = np.where(deciduous, ft_m > 0, 1.0)
                lai_m = lai * phen
                light = par[m].copy()
                g_m = np.zeros((G, n))
                for layer in ("canopy", "epiphyte", "shrub", "ground"):
                    sel = np.array([g.traits["layer"] == layer for g in groups])
                    if not sel.any():
                        continue
                    L = (lai_m * sel[:, None]).sum(axis=0)
                    k_l = float(np.mean(kx[sel]))
                    absorbed = light * (1.0 - np.exp(-k_l * L))
                    share = np.where(L > 0, lai_m / np.maximum(L, 1e-30), 0.0) * sel[:, None]
                    a_g = absorbed[None, :] * share
                    ap += a_g
                    g_m += lue * a_g * ft_m * np.nan_to_num(fw_m, nan=0.0) * fn
                    light = light - absorbed  # transmitted to the layer below (B22)
                # water-use ceiling: GPP of the month <= WUE x actual ET (no carbon without
                # the water to fix it; desert plants cannot outgrow their rain)
                tot = g_m.sum(axis=0)
                cap = WUE_GC_PER_KG * np.nan_to_num(aet[m], nan=0.0)
                g_m = g_m * np.where(tot > cap, cap / np.maximum(tot, 1e-30), 1.0)
                gpp += g_m
                gm[m] = (cue * g_m).sum(axis=0)
            npp = cue * gpp
            if _it == iterations:
                break
            a_leaf = np.array([g.traits["alloc"]["leaf"] for g in groups])[:, None]
            t_leaf = np.array([g.traits["turn"]["leaf"] for g in groups])[:, None]
            sla = np.array([g.traits["sla"] for g in groups])[:, None]
            lai_eq = npp / 1000.0 * a_leaf / t_leaf * sla
            lai = np.minimum(lai_cap, np.maximum(lai_eq, 0.0))
        out["npp"] += wbin * npp
        out["gpp"] += wbin * gpp
        out["apar"] += wbin * ap
        out["cover"] += wbin * cover
        out["lai"] += wbin * lai
        out["suit"] += wbin * S
        out["npp_month"] += wbin * gm
        # main limitation of the hex (light / temperature / water / nutrients), for explain
        ft_mean = ft_acc.max(axis=0)
        fw_mean = fw_acc.max(axis=0)
        out["limit"] += wbin * np.stack([np.clip(par.sum(axis=0) / 3000.0, 0, 1), ft_mean,
                                         np.nan_to_num(fw_mean), fn.max(axis=0)])  # fmt: skip
    # unknown nutrient supply: NPP unknown, not guessed (B09)
    unknown = ~np.isfinite(n_sup) | ~np.isfinite(p_sup)
    out["npp"][:, unknown] = np.nan
    out["unknown_nutrients"] = unknown
    _stocks(out, groups)
    return out


def _stocks(out, groups) -> None:
    """Pool NPP and equilibrium stocks (kgC per m² of ground) of every group."""
    npp = np.nan_to_num(out["npp"])
    G, n = npp.shape
    pools = {}
    for p in ("leaf", "wood", "root", "repro"):
        a = np.array([g.traits["alloc"][p] for g in groups])[:, None]
        lam = np.array([max(g.traits["turn"][p], 1e-9) for g in groups])[:, None]
        pools[f"npp_{p}"] = npp * a
        pools[f"stock_{p}"] = np.where(a > 0, npp * a / lam / 1000.0, 0.0)  # kgC/m²
    out.update(pools)
    out["biomass_kgc_m2"] = sum(pools[f"stock_{p}"] for p in ("leaf", "wood", "root", "repro"))


def land_pools(prod, groups_ids, pollination=None) -> dict:
    """Split the producer NPP (gC/m² of ground) into the food pools and the inedible litter
    (no carbon counted twice: the pools partition the NPP exactly, B43)."""
    reg = load()
    n = prod["npp_leaf"].shape[1]
    pools = {k: np.zeros(n) for k in ("leaf_woody", "leaf_herb", "fruit_seed", "nectar",
                                      "root_exudate", "litter")}  # fmt: skip
    for gi, gid in enumerate(groups_ids):
        g = reg.groups[gid]
        tr = g.traits
        edible = tr["edible"]
        leaf = prod["npp_leaf"][gi]
        pool = "leaf_woody" if tr["layer"] in ("canopy", "shrub", "epiphyte") else "leaf_herb"
        pools[pool] += leaf * edible
        pools["litter"] += leaf * (1.0 - edible) + prod["npp_wood"][gi]
        root = prod["npp_root"][gi]
        pools["root_exudate"] += root * ROOT_EXUDATE
        pools["litter"] += root * (1.0 - ROOT_EXUDATE)
        rep = prod["npp_repro"][gi]
        if tr["poll"] == "animal":
            # without pollinators an animal-pollinated plant sets less seed (B48)
            pf = 1.0 if pollination is None else (0.3 + 0.7 * pollination)
            nect = rep * NECTAR_SHARE
            seed = (rep - nect) * pf
            pools["nectar"] += nect
            pools["fruit_seed"] += seed
            pools["litter"] += rep - nect - seed
        else:
            pools["fruit_seed"] += rep * 0.8
            pools["litter"] += rep * 0.2
    return pools


# -- fresh water --------------------------------------------------------------------------------
def fresh_production(inp, frac):
    """Producers of lakes and rivers per m² of inland water: phytoplankton, periphyton and
    macrophytes. Lake ice (air < −2 °C) is opaque; the nutrient factor is a labelled proxy
    from the catchment soil N mineralisation (no dissolved nutrient pools exist)."""
    reg = load()
    f = inp.f
    idx = np.nonzero(frac["inland_water"] > 0)[0]
    n = idx.size
    par = f["par_m"][:, idx]
    t = f["t_fresh_m"][:, idx]
    open_m = f["t_air_m"][:, idx] > LAKE_ICE_T
    nut = np.clip(np.nan_to_num(f["n_min_kg_m2_yr"][idx], nan=0.01) / N_REF_KG_M2_YR, 0.05, 1.0)
    river = f["river"][idx] & ~f["lake"][idx]
    depth = np.where(f["lake"][idx], np.nan_to_num(f["lake_depth_m"][idx], nan=5.0),
                     np.maximum(f["river_depth_m"][idx], 0.5))  # fmt: skip
    shallow = np.where(river, 0.3, np.exp(-depth / 4.0))
    env = {"tw": t.max(axis=0), "tc": t.min(axis=0)}
    out = {"idx": idx, "groups": [], "npp": [], "suit": [], "cover": []}
    for g in reg.groups.values():
        if not g.is_producer or g.traits["layer"] not in ("fw_plankton", "fw_benthic", "fw_macro"):
            continue
        tr = g.traits
        S = np.nan_to_num(niche_score({k: v for k, v in tr["niche"].items()
                                       if k in ("tw", "tc")}, env))  # fmt: skip
        if "river" in tr["gates"]:
            S = S * river
        if "shallow_water" in tr["gates"]:
            S = S * (shallow > 0.05)
        if tr["layer"] == "fw_plankton":
            area, light_f = np.ones(n), 0.5 * nut  # share of light the plankton absorbs
        elif tr["layer"] == "fw_benthic":
            area, light_f = river.astype(float), np.exp(-1.0 * depth) * 0.6
        else:
            area, light_f = shallow, np.full(n, 0.6)
        npp = np.zeros(n)
        for m in range(12):
            ft = trapezoid(t[m], *tr["t_act"])
            npp += tr["cue"] * tr["lue"] * par[m] * open_m[m] * light_f * ft * np.maximum(nut, 0.2)
        cover = area * S
        out["groups"].append(g.id)
        out["npp"].append(npp * cover)
        out["suit"].append(S)
        out["cover"].append(cover)
    for k in ("npp", "suit", "cover"):
        out[k] = np.array(out[k])
    return out


# -- sea ----------------------------------------------------------------------------------------
def _substrate(inp, idx):
    """Hard / soft bottom shares from the coast type of the shore (rocky vs sandy coasts)
    and sediment on the open shelf."""
    from ..stages.coasts import COAST_TYPES, ROCKY, SANDY

    f = inp.f
    ct = f["coast_type"]
    nb = inp.nbr[idx]
    rocky_codes = np.array([COAST_TYPES.index(c) for c in ROCKY])
    sandy_codes = np.array([COAST_TYPES.index(c) for c in SANDY + ("tidal_flat", "delta",
                                                                   "estuary", "mangrove")])
    near = np.where(nb >= 0, ct[np.maximum(nb, 0)], 0)
    rocky = np.isin(near, rocky_codes).any(axis=1) | np.isin(ct[idx], rocky_codes)
    sandy = np.isin(near, sandy_codes).any(axis=1) | np.isin(ct[idx], sandy_codes)
    volcanic = f["lava_fraction"][idx] > 0
    hard = np.where(rocky | volcanic, 0.7, np.where(sandy, 0.2, 0.3))
    reef = np.nan_to_num(f["marine_coral_reef_potential"][idx]) > 0
    hard = np.where(reef, np.maximum(hard, 0.6), hard)
    return hard, 1.0 - hard


def marine_production(inp, frac):
    """Phytoplankton, ice algae, lit-bottom producers and vent chemosynthesis per m² of
    sea, the export flux and what reaches the bottom (Martin curve)."""
    reg = load()
    f = inp.f
    idx = np.nonzero(frac["marine_water"] > 0)[0]
    n = idx.size
    par = f["par_m"][:, idx]
    sst = np.nan_to_num(f["sst_m"][:, idx], nan=np.nan)
    sst = np.where(np.isfinite(sst), sst, np.maximum(f["t_air_m"][:, idx], -1.8))
    ice = np.clip(f["sea_ice_m"][:, idx], 0.0, 1.0)
    nutr = np.nan_to_num(f["marine_nutrient_supply_proxy"][idx], nan=0.1)
    depth = np.maximum(np.nan_to_num(f["depth_m"][idx]), 1.0)
    kd = np.nan_to_num(f["marine_kd_per_m"][idx], nan=0.1)
    z_eu = math.log(100.0) / kd  # N13
    sal = np.nan_to_num(f["sss_psu"][idx], nan=35.0)
    lit_share = np.clip(1.0 - depth / z_eu, 0.0, 1.0)
    hard, soft = _substrate(inp, idx)
    env = {"tw": sst.max(axis=0), "tc": sst.min(axis=0), "sal": sal}
    vent_mw = np.where(f["hydrothermal_kind"][idx] > 0, np.maximum(f["geothermal_heat_mw"][idx],
                       1.0), 0.0)  # fmt: skip
    axis = (np.nan_to_num(f["spreading_axis_km"][idx], nan=np.inf) < 0.5 * inp.spacing_km) & \
        (depth > 1500.0)  # fmt: skip
    vent_mw = vent_mw + np.where(axis, AXIS_VENT_MW, 0.0)
    out = {"idx": idx, "groups": [], "npp": [], "suit": [], "cover": [], "z_eu": z_eu,
           "lit_share": lit_share, "hard": hard, "soft": soft}  # fmt: skip
    phyto_total = np.zeros(n)
    for g in reg.groups.values():
        tr = g.traits
        layer = tr.get("layer")
        if not g.is_producer or layer not in ("m_plankton", "m_ice", "m_benthic", "chemo"):
            continue
        S = np.nan_to_num(niche_score(tr["niche"], env))
        npp = np.zeros(n)
        cover = np.ones(n)
        if layer == "m_plankton":
            for m in range(12):
                par_w = par[m] * ((1.0 - ice[m]) + ice[m] * SEA_ICE_TRANSMISSION)
                f_abs = nutr / (nutr + 0.3)
                npp += tr["cue"] * tr["lue"] * par_w * f_abs * trapezoid(sst[m], *tr["t_act"])
            phyto_total = npp * S
        elif layer == "m_ice":
            S = S * (f["sea_ice_months"][idx] >= 1)
            for m in range(12):
                npp += (tr["cue"] * tr["lue"] * par[m] * ice[m] * (1.0 - SEA_ICE_TRANSMISSION)
                        * ICE_ALGAE_ABSORB * trapezoid(sst[m], *tr["t_act"]))  # fmt: skip
            cover = (ice.max(axis=0) > 0).astype(float)
        elif layer == "m_benthic":
            sub = hard if "hard_bottom" in tr["gates"] else (soft if "soft_bottom" in tr["gates"]
                                                              else np.ones(n))  # fmt: skip
            zmax = np.minimum(depth, z_eu)
            mean_light = (1.0 - np.exp(-kd * zmax)) / np.maximum(kd * zmax, 1e-9)
            for m in range(12):
                par_b = par[m] * ((1.0 - ice[m]) + ice[m] * SEA_ICE_TRANSMISSION) * mean_light
                npp += tr["cue"] * tr["lue"] * par_b * trapezoid(sst[m], *tr["t_act"]) * 0.5
            cover = lit_share * sub
        else:  # chemo: an explicit subsidy from the heat of the vents, not sunlight (B55)
            S = S * (vent_mw > 0)
            area_m2 = inp.area_km2 * 1e6 * frac["marine_water"][idx]
            npp = CHEMO_GC_PER_MW_YR * vent_mw / np.maximum(area_m2, 1.0)
        out["groups"].append(g.id)
        out["npp"].append(npp * S * cover)
        out["suit"].append(S)
        out["cover"].append(cover * (S > 0))
    for k in ("npp", "suit", "cover"):
        out[k] = np.array(out[k])
    gi = {g: i for i, g in enumerate(out["groups"])}
    export0 = EXPORT_RATIO * (out["npp"][gi["phytoplankton_marine"]] + out["npp"][gi["ice_algae"]])
    zref = np.maximum(z_eu, 100.0)
    att = np.where(depth > zref, (depth / zref) ** (-MARTIN_B), 1.0)
    out["export_gc_m2"] = export0
    out["bottom_flux_gc_m2"] = export0 * att
    out["remineralised_gc_m2"] = export0 - out["bottom_flux_gc_m2"]
    out["depth"] = depth
    return out


def marine_pools(mp) -> dict:
    """Food pools of the sea (gC/m² of sea): the exported share has left the surface and is
    not food there (B55); the bottom gets only what reaches it."""
    gi = {g: i for i, g in enumerate(mp["groups"])}
    npp = mp["npp"]
    phyto = npp[gi["phytoplankton_marine"]]
    ice = npp[gi["ice_algae"]]
    stay_p = phyto * (1.0 - EXPORT_RATIO)
    stay_i = ice * (1.0 - EXPORT_RATIO)
    macro = sum(npp[gi[k]] for k in ("kelp", "algae_macro_other", "seagrass"))
    pools = {
        "m_phyto": 0.7 * stay_p,
        "m_ice_algae": 0.7 * stay_i,
        "m_detritus": 0.3 * (stay_p + stay_i) + 0.4 * macro,
        "m_macro": 0.6 * (npp[gi["kelp"]] + npp[gi["algae_macro_other"]]),
        "m_seagrass": 0.6 * npp[gi["seagrass"]],
        "m_benthic_flux": mp["bottom_flux_gc_m2"],
        "m_chemo": npp[gi["chemo_microbes"]],
    }
    pools["_sink_remineralised"] = mp["remineralised_gc_m2"]
    return pools


def fresh_pools(fp) -> dict:
    gi = {g: i for i, g in enumerate(fp["groups"])}
    npp = fp["npp"]
    phyto = npp[gi["phytoplankton_fresh"]]
    peri = npp[gi["periphyton"]]
    macro = sum(npp[gi[k]] for k in ("aquatic_submerged", "aquatic_floating", "aquatic_emergent"))
    return {"fw_phyto": 0.8 * phyto, "fw_periphyton": 0.8 * peri, "fw_macro": 0.5 * macro,
            "fw_detritus": 0.2 * (phyto + peri) + 0.5 * macro}  # fmt: skip
