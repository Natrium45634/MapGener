"""The biosphere solver of TZ 5.5 (G1–G5): potential suitability → accessibility →
occupancy → food-limited realized state, on the physical fields of the existing world.

The modes of §2 are explicit:

* ``POTENTIAL`` — what the physics allows (producers at full potential, every template
  where suitable), no history, no dispersal;
* ``REALIZED_BASELINE`` — the same physics after dispersal from the lineage pools, closed
  -form occupancy and joint food allocation (the default);
* ``DYNAMIC`` — the baseline advanced year by year by :mod:`.dynamics` (P1).

Nothing here changes a field of another stage.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np

from . import foodweb as fw
from . import graph as gr
from . import metrics as mt
from . import production as pr
from .inputs import BioInputs, habitat_fractions
from .niche import breeding_ok, combine, survival, trapezoid
from .registry import TERRESTRIAL_LAYERS, load, taxon_id

ALGORITHM_VERSION = "hexworld-biosphere-5.5.0"
P_MIN = 0.5
S_MIN = 0.2
MODES = ("POTENTIAL", "REALIZED_BASELINE", "DYNAMIC")


@dataclass
class RealmTable:
    """Templates of one home realm on its hexes: occupancy and suitability as uint8 (×255)."""

    realm: str
    idx: np.ndarray  # hexes of the realm (flat indices)
    templates: list  # template ids
    occ: np.ndarray  # (T, n) uint8
    suit: np.ndarray  # (T, n) uint8
    present: np.ndarray  # (T, n) bool

    def row(self, tid):
        return self.templates.index(tid)


@dataclass
class BioState:
    mode: str
    seed: str
    n: int
    shape: tuple
    area_km2: float
    frac: dict
    groups: dict  # group id -> {"realm", "idx", arrays...}
    realms: dict  # realm -> RealmTable
    lineage: dict  # class -> {"pool": int16 (N), "d_km": float32 (N)}
    lineage_names: dict  # family -> list of names
    lineage_keys: dict
    layers: dict  # per-hex maps (float32 / int)
    ledger: dict
    summary: dict
    quality: dict
    timings: dict = field(default_factory=dict)


def _home(g) -> str:
    gates = g.traits["gates"]
    if g.is_producer:
        layer = g.traits["layer"]
        if layer in TERRESTRIAL_LAYERS:
            return "land"
        return "fresh" if layer.startswith("fw_") else "marine"
    first = gates[0] if gates else "land_ground"
    if first in ("marine",):
        return "marine"
    if first in ("inland_water", "any_fresh", "anadromous", "river"):
        return "fresh"
    if first == "any_water":
        return "water"
    return "land"


def family_of(cls: str) -> str:
    if cls in gr.MARINE_CLASSES:
        return "marine"
    if cls in ("river", "river_ocean", "water_drift", "water_seed", "river_flight"):
        return "fresh"
    if cls in gr.COSMO:
        return "cosmo"
    return "land"


def _gates(inp: BioInputs, frac, struct) -> dict:
    f = inp.f
    depth = np.nan_to_num(f["depth_m"])
    any_fresh = (frac["inland_water"] > 0) | f["river"]
    lm = f["landmass_id"]
    lm_area = np.zeros(inp.n)
    if (lm >= 0).any():
        cnt = np.bincount(lm[lm >= 0])
        lm_area = np.where(lm >= 0, cnt[np.maximum(lm, 0)] * inp.area_km2, 0.0)
    mouth_basin = np.zeros(inp.n, dtype=bool)
    exo = ~f["endorheic"]
    return {
        "land_ground": frac["land_ground"] > 0,
        "root_zone": f["soil_present"],
        "canopy": struct["canopy"] > 0.2,
        "woody_cover": struct["woody"] > 0.15,
        "herb_cover": struct["herb"] > 0.1,
        "flowers": struct["nectar"] > 0,
        "host_plants": struct["plant_cover"] > 0.05,
        "hosts": frac["land_ground"] > 0,
        "soil": f["soil_present"],
        "wet_ground": (f["wetness"] >= 0.5) | (f["waterlogging_months"] >= 1),
        "large_land": lm_area >= 5.0e4,
        "inland_water": frac["inland_water"] > 0,
        "river": f["river"],
        "any_fresh": any_fresh,
        "any_water": (frac["inland_water"] > 0) | (frac["marine_water"] > 0),
        "anadromous": (any_fresh & exo) | (mouth_basin | f["coast_sea"]),
        "marine": frac["marine_water"] > 0,
        "shelf": (frac["marine_water"] > 0) & (depth <= 200.0),
        "deep": (frac["marine_water"] > 0) & (depth > 2000.0),
        "deep_shelf": (frac["marine_water"] > 0) & (depth > 150.0) & (depth <= 2000.0),
        "lit_bottom": struct["lit_share"] > 0.05,
        "hard_bottom": struct["hard"] > 0.2,
        "soft_bottom": struct["soft"] > 0.2,
        "sea_ice": f["sea_ice_months"] >= 1,
        "vent": struct["vent"],
        "coast": f["coast_land"] | f["coast_sea"],
        "nest_coast": (frac["marine_water"] > 0) & (f["coast_sea"] | (depth <= 200.0)),
        "haulout": f["coast_sea"],
        "tidal_coast": f["coast_land"] & (f["tidal_range_m"] >= 0.3),
    }


def _fauna_suit(tpl, g, env, gates_all, idx, days):
    tr = g.traits
    sc = []
    for var in ("tc", "tw", "mi", "sal"):
        if var in tpl.niche and env.get(var) is not None:
            sc.append(trapezoid(env[var], *tpl.niche[var]))
    t_m = env["t_m"]
    s_m = np.stack([trapezoid(t_m[m], *tr["t_act"]) for m in range(12)])
    if tr["migratory"]:
        surv = np.ones(idx.size)  # leaves in the bad months (seasonal range, P1 movement)
    else:
        surv = survival(s_m, days, dormant=tr["dormant"])
    sc.append(surv)
    S = np.nan_to_num(combine(sc))
    S = S * breeding_ok(s_m, tr["breed_months"])
    for gname in tr["gates"]:
        S = S * gates_all[gname][idx]
    return S


def run(project, mode: str = "REALIZED_BASELINE", detail: str = "standard", seed: str | None = None,
        capture_mult=gr.CAPTURE_MULT, progress=None) -> BioState:  # fmt: skip
    from .inputs import from_project

    if mode not in MODES:
        raise ValueError(f"mode must be one of {MODES}")
    rep = progress or (lambda f: None)
    T = {}
    t0 = time.perf_counter()
    inp = from_project(project, detail)
    frac = habitat_fractions(inp)
    T["inputs"] = time.perf_counter() - t0
    state = solve(inp, frac, mode=mode, seed=seed if seed else project.config.seed,
                  capture_mult=capture_mult, landmasses=project.hydro.landmasses,
                  progress=rep, timings=T)  # fmt: skip
    return state


def solve(inp: BioInputs, frac, mode="REALIZED_BASELINE", seed="", capture_mult=gr.CAPTURE_MULT,
          landmasses=(), progress=None, timings=None) -> BioState:  # fmt: skip
    reg = load()
    rep = progress or (lambda f: None)
    T = timings if timings is not None else {}
    f = inp.f
    n = inp.n
    area_m2 = inp.area_km2 * 1e6
    t = time.perf_counter()
    # n-fixing symbionts add nitrogen where they are suited (before the plants)
    nf = reg.groups["n_fixers"].traits
    nfix = nf["n_fix_kg_m2_yr"] * np.nan_to_num(trapezoid(f["tw"], *nf["niche"]["tw"])) * \
        (frac["land_ground"] > 0)  # fmt: skip
    lp = pr.land_production(inp, frac, n_fix_supply=nfix)
    fp = pr.fresh_production(inp, frac)
    mp = pr.marine_production(inp, frac)
    T["production"] = time.perf_counter() - t
    rep(0.35)
    # structure seen by the animals (shares of the land ground)
    li = lp["idx"]
    gl = {g: i for i, g in enumerate(lp["groups"])}
    layer_of = {g: reg.groups[g].traits["layer"] for g in lp["groups"]}

    def layer_sum(layers, arr):
        out = np.zeros(n)
        rows = [gl[g] for g in lp["groups"] if layer_of[g] in layers]
        if rows:
            out[li] = arr[rows].sum(axis=0)
        return out

    struct = {"canopy": layer_sum(("canopy",), lp["cover"]),
              "woody": layer_sum(("canopy", "shrub"), lp["cover"]),
              "herb": layer_sum(("ground",), lp["cover"]),
              "plant_cover": layer_sum(TERRESTRIAL_LAYERS, lp["cover"]),
              "nectar": np.zeros(n), "lit_share": np.zeros(n), "hard": np.zeros(n),
              "soft": np.zeros(n), "vent": np.zeros(n, dtype=bool)}  # fmt: skip
    mi_ = mp["idx"]
    struct["lit_share"][mi_] = mp["lit_share"]
    struct["hard"][mi_] = mp["hard"]
    struct["soft"][mi_] = mp["soft"]
    gm = {g: i for i, g in enumerate(mp["groups"])}
    struct["vent"][mi_] = mp["suit"][gm["chemo_microbes"]] > 0
    poll_rows = [gl[g] for g in lp["groups"] if reg.groups[g].traits["poll"] == "animal"]
    struct["nectar"][li] = (np.nan_to_num(lp["npp_repro"][poll_rows]).sum(axis=0) > 1.0)
    gates = _gates(inp, frac, struct)
    # -- graphs and lineage pools ------------------------------------------------------------
    t = time.perf_counter()
    land_l = gr.land_pools(inp, frac, landmasses)
    fresh_l = gr.fresh_pools(inp, frac, land_l)
    sea_l = gr.marine_pools(inp, frac)
    cosmo_l = gr.Lineages(["cosmo"], ["космополитная линия"],
                          np.where(np.arange(n) == 0, 0, -1).astype(np.int64))  # fmt: skip
    fam_pools = {"land": land_l, "fresh": fresh_l, "marine": sea_l, "cosmo": cosmo_l}
    classes = sorted({g.traits["mob"] for g in reg.groups.values()})
    lineage = {}
    for cls in classes:
        fam = family_of(cls)
        if fam == "cosmo" or mode == "POTENTIAL":
            pool = np.zeros(n, dtype=np.int64) if fam == "cosmo" else \
                _nearest_pool_euclid(inp, fam_pools[fam])  # fmt: skip
            lineage[cls] = {"pool": pool.astype(np.int16), "d_km": np.zeros(n, np.float32)}
            continue
        g_cls = gr.build_graph(cls, inp, frac, canopy=struct["canopy"], capture_mult=capture_mult)
        d, lin = gr.nearest_lineage(g_cls, fam_pools[fam])
        lineage[cls] = {"pool": lin.astype(np.int16), "d_km": np.where(np.isfinite(d), d,
                        np.inf).astype(np.float32)}  # fmt: skip
    age, age_unknown = gr.substrate_age_yr(inp, frac, land_l)
    T["graphs"] = time.perf_counter() - t
    rep(0.55)
    # -- template suitability and occupancy per home realm -------------------------------------
    t = time.perf_counter()
    lm = f["landmass_id"]
    cnt = np.bincount(lm[lm >= 0]) if (lm >= 0).any() else np.zeros(1)
    lm_area = np.where(lm >= 0, cnt[np.maximum(lm, 0)] * inp.area_km2, inp.area_km2)
    bs = f["basin_id"]
    bcnt = np.bincount(bs[bs >= 0]) if (bs >= 0).any() else np.zeros(1)
    basin_area = np.where(bs >= 0, bcnt[np.maximum(bs, 0)] * inp.area_km2, inp.area_km2)
    realm_idx = {"land": np.nonzero(frac["land_ground"] > 0)[0],
                 "fresh": np.nonzero((frac["inland_water"] > 0))[0],
                 "marine": np.nonzero(frac["marine_water"] > 0)[0]}  # fmt: skip
    realm_idx["water"] = np.union1d(realm_idx["fresh"], realm_idx["marine"])
    env_r = {
        "land": {"tc": f["tc"], "tw": f["tw"], "mi": f["mi"], "sal": f["ec_ds_m"],
                 "t_m": f["t_air_m"]},
        "fresh": {"tc": f["t_fresh_m"].min(axis=0), "tw": f["t_fresh_m"].max(axis=0),
                  "sal": np.where(f["endorheic"], 2.0, 0.2), "t_m": f["t_fresh_m"]},
        "marine": {"tc": f["tc"], "tw": f["tw"], "sal": f["sss_psu"],
                   "t_m": np.where(np.isfinite(f["sst_m"]), f["sst_m"], f["t_air_m"])},
    }  # fmt: skip
    env_r["water"] = {k: np.where(frac["marine_water"] > frac["inland_water"], env_r["marine"][k],
                                  env_r["fresh"][k]) for k in ("tc", "tw", "sal", "t_m")}
    occ_area = {"land": lm_area, "fresh": basin_area, "marine": np.full(n, 1e7),
                "water": np.full(n, 1e6)}  # fmt: skip
    # producer cover realized by occupancy; per group the potential cover
    prod_cover = {}
    for g_id, i in gl.items():
        c = np.zeros(n)
        c[li] = lp["cover"][i]
        prod_cover[g_id] = c
    for src, tab in ((fp, None), (mp, None)):
        for i, g_id in enumerate(src["groups"]):
            c = np.zeros(n)
            c[src["idx"]] = src["cover"][i]
            prod_cover[g_id] = c
    realms: dict = {}
    tmpl_occ: dict = {}
    tmpl_suit: dict = {}
    by_realm: dict = {}
    for tpl in reg.templates.values():
        by_realm.setdefault(_home(reg.groups[tpl.group_id]), []).append(tpl)
    # microsite environment for land producers (the best microsite keeps the template, B14)
    bins = f["microsites"]
    for realm, tpls in by_realm.items():
        idx = realm_idx[realm]
        env = {k: (v[..., idx] if v is not None else None) for k, v in env_r[realm].items()}
        T_n = len(tpls)
        occ_a = np.zeros((T_n, idx.size), dtype=np.uint8)
        suit_a = np.zeros((T_n, idx.size), dtype=np.uint8)
        for k, tpl in enumerate(tpls):
            g = reg.groups[tpl.group_id]
            if g.is_producer:
                gate = prod_cover[g.id][idx] > 1e-4
                if realm == "land":
                    best = np.zeros(idx.size)
                    for off, _w in bins:
                        tb = f["t_air_m"][:, idx] + off[idx][None, :]
                        e2 = {"tc": tb.min(axis=0), "tw": tb.max(axis=0), "mi": env["mi"],
                              "gdd": pr._growing_degree_days(tb, inp.days),
                              "ph": f["ph"][idx], "sal": f["ec_ds_m"][idx]}  # fmt: skip
                        best = np.maximum(best, np.nan_to_num(pr.niche_score(tpl.niche, e2)))
                    S = best * gate
                else:
                    S = np.nan_to_num(pr.niche_score({k2: v for k2, v in tpl.niche.items()
                                                      if k2 in ("tc", "tw", "sal")}, env)) * gate
            else:
                S = _fauna_suit(tpl, g, env, gates, idx, inp.days)
            tr = g.traits
            if mode == "POTENTIAL":
                occ = (S > 0).astype(np.float64)
            else:
                lin = lineage[tr["mob"]]
                d = lin["d_km"][idx].astype(np.float64)
                c = gr.colonisation_rate(np.where(np.isfinite(d), d, 1e12), tr["L_km"])
                mass = tr.get("mass_kg", 1e-3)
                e = gr.extinction_rate(tr["e0"], occ_area[realm][idx], mass, S)
                occ = gr.occupancy(c, e, age[idx]) * (S > 0)
            occ_a[k] = np.round(occ * 255).astype(np.uint8)
            suit_a[k] = np.round(np.clip(S, 0, 1) * 255).astype(np.uint8)
            tmpl_occ[tpl.id] = occ
            tmpl_suit[tpl.id] = S
        realms[realm] = RealmTable(realm, idx, [t_.id for t_ in tpls], occ_a, suit_a,
                                   np.zeros((T_n, idx.size), dtype=bool))  # fmt: skip
    T["suitability"] = time.perf_counter() - t
    rep(0.75)
    # -- group weights (occupancy-weighted presence) -----------------------------------------
    t = time.perf_counter()
    weights = {}
    for g in reg.groups.values():
        realm = _home(g)
        idx = realm_idx[realm]
        q = np.ones(idx.size)
        for tpl in reg.templates_of(g.id):
            q = q * (1.0 - tmpl_occ[tpl.id] * tmpl_suit[tpl.id])
        w = np.zeros(n)
        w[idx] = 1.0 - q
        weights[g.id] = w
    # realized producers: potential cover x occupancy weight of the group (a remote young
    # island has no trees until their lineage arrives, S09)
    real = (lambda g_id: np.ones(n)) if mode == "POTENTIAL" else \
        (lambda g_id: np.clip(weights[g_id] / np.maximum(_wmax(reg, g_id, tmpl_suit, realm_idx, n),
                                                         1e-9), 0, 1))  # fmt: skip
    groups_out: dict = {}
    land_scale = np.zeros((len(lp["groups"]), li.size))
    for g_id, i in gl.items():
        land_scale[i] = real(g_id)[li]
    npp_land = np.nan_to_num(lp["npp"]) * land_scale
    prod_r = {k: lp[k] * land_scale for k in ("npp_leaf", "npp_wood", "npp_root", "npp_repro",
                                              "stock_leaf", "stock_wood", "stock_root",
                                              "stock_repro")}  # fmt: skip
    prod_r["npp"] = npp_land
    poll = weights["insect_pollinator"][li]
    lpools = pr.land_pools(prod_r, lp["groups"], pollination=poll)
    fscale = np.stack([real(g)[fp["idx"]] for g in fp["groups"]]) if fp["groups"] else 0
    mscale = np.stack([real(g)[mp["idx"]] for g in mp["groups"]])
    fp_r = dict(fp, npp=fp["npp"] * fscale)
    mp_r = dict(mp, npp=mp["npp"] * mscale)
    fpools = pr.fresh_pools(fp_r)
    mpools = pr.marine_pools(mp_r)
    # pools in kgC/yr per hex
    A = {}
    g_area = frac["land_ground"] * area_m2
    for k, v in lpools.items():
        a = np.zeros(n)
        a[li] = v * g_area[li] / 1000.0
        A[k] = a
    for src, pools, ar in ((fp, fpools, frac["inland_water"]), (mp, mpools, frac["marine_water"])):
        for k, v in pools.items():
            if k.startswith("_"):
                continue
            a = np.zeros(n)
            a[src["idx"]] = v * ar[src["idx"]] * area_m2 / 1000.0
            A[k] = A.get(k, 0) + a
    home_area = {}
    for g in reg.groups.values():
        realm = _home(g)
        ar = {"land": frac["land_ground"], "fresh": frac["inland_water"],
              "marine": frac["marine_water"],
              "water": frac["inland_water"] + frac["marine_water"]}[realm]  # fmt: skip
        home_area[g.id] = ar * area_m2
    water_ok = (frac["inland_water"] > 0) | f["river"] | (np.nan_to_num(f["mi"]) >= 0.15)
    web = fw.run_foodweb(A, {g: weights[g] for g in weights if not reg.groups[g].is_producer},
                         home_area, n, gates_water=water_ok)  # fmt: skip
    T["foodweb"] = time.perf_counter() - t
    rep(0.85)
    # -- realized presence of templates, group outputs ---------------------------------------
    t = time.perf_counter()
    realized_group = {}
    for g in reg.groups.values():
        if g.is_producer:
            c = prod_cover[g.id] * real(g.id)
            realized_group[g.id] = c > 1e-3
            groups_out[g.id] = {"realm": _home(g), "cover": c.astype(np.float32),
                                "potential_cover": prod_cover[g.id].astype(np.float32)}
        else:
            r = web["groups"][g.id]
            b = r.get("biomass_kgc", np.zeros(n))
            realized_group[g.id] = b > 1e-6 * np.maximum(home_area[g.id] / 1e6, 1e-9)
            groups_out[g.id] = {"realm": _home(g), "biomass_kgc": b.astype(np.float32),
                                "individuals": r.get("individuals", np.zeros(n)).astype(np.float32),
                                "production_kgc_yr": r.get("production", np.zeros(n)).astype(np.float32),
                                "limit": r.get("limit", np.zeros(n, np.uint8))}  # fmt: skip
        groups_out[g.id]["weight"] = weights[g.id].astype(np.float32)
    # NPP per group (gC/m² of the hex) for the producers
    for i, g_id in enumerate(lp["groups"]):
        a = np.zeros(n, np.float32)
        a[li] = npp_land[i] * frac["land_ground"][li]
        groups_out[g_id]["npp_gc_m2_hex"] = a
    for src, ar in ((fp_r, frac["inland_water"]), (mp_r, frac["marine_water"])):
        for i, g_id in enumerate(src["groups"]):
            a = np.zeros(n, np.float32)
            a[src["idx"]] = src["npp"][i] * ar[src["idx"]]
            groups_out[g_id]["npp_gc_m2_hex"] = a
    for realm, tab in realms.items():
        for k, tid in enumerate(tab.templates):
            g_id = reg.templates[tid].group_id
            tab.present[k] = ((tab.occ[k] >= int(P_MIN * 255)) & (tab.suit[k] >= int(S_MIN * 255))
                              & realized_group[g_id][tab.idx])  # fmt: skip
    layers, summary = _maps(inp, frac, reg, realms, groups_out, lineage, fam_pools, lp, mp, fp,
                            npp_land, prod_r, li, seed, age_unknown)  # fmt: skip
    T["metrics"] = time.perf_counter() - t
    rep(1.0)
    ledger = {"foodweb": web["ledger"], "npp_total_kgc_yr": float(
        sum(np.nansum(groups_out[g]["npp_gc_m2_hex"]) for g in groups_out
            if "npp_gc_m2_hex" in groups_out[g]) * area_m2 / 1000.0)}  # fmt: skip
    quality = {"history": "present_only", "inputs": dict(inp.quality),
               "age_unknown_hexes": int(age_unknown.sum()),
               "nutrients_unknown_hexes": int(np.asarray(lp["unknown_nutrients"]).sum()),
               "ocean_radiation": "angstrom_prescott (n/N = 0.5), APPROX",
               "marine_nutrients": "proxy (marine_nutrient_supply_proxy)",
               "parameters": "provisional, not calibrated"}  # fmt: skip
    summary.update({"mode": mode, "algorithm_version": ALGORITHM_VERSION,
                    "registry": reg.version, "templates": len(reg.templates),
                    "groups": len(reg.groups), "ledger_ok": bool(web["ledger"]["ok"])})
    return BioState(mode, seed, n, inp.shape, inp.area_km2, {k: v.astype(np.float32) for k, v in
                    frac.items()}, groups_out, realms, lineage,
                    {k: v.names for k, v in fam_pools.items()},
                    {k: v.keys for k, v in fam_pools.items()}, layers, ledger, summary, quality,
                    T)  # fmt: skip


def _wmax(reg, g_id, tmpl_suit, realm_idx, n):
    """The weight the group would have with full occupancy (for the realized share)."""
    g = reg.groups[g_id]
    idx = realm_idx[_home(g)]
    q = np.ones(idx.size)
    for tpl in reg.templates_of(g_id):
        q = q * (1.0 - tmpl_suit[tpl.id])
    out = np.zeros(n)
    out[idx] = 1.0 - q
    return out


def _nearest_pool_euclid(inp, pools):
    """POTENTIAL mode: the lineage label of the nearest pool cell by straight distance (no
    graph, no barrier) — only a label for the potential map."""
    from scipy.spatial import cKDTree

    src = np.nonzero(pools.node_pool >= 0)[0]
    if src.size == 0:
        return np.full(inp.n, -1)
    tree = cKDTree(np.column_stack([inp.x_km[src], inp.y_km[src]]))
    _d, j = tree.query(np.column_stack([inp.x_km, inp.y_km]))
    return pools.node_pool[src[j]]


def _maps(inp, frac, reg, realms, groups_out, lineage, fam_pools, lp, mp, fp, npp_land, prod_r,
          li, seed, age_unknown):  # fmt: skip
    n = inp.n
    area = inp.area_km2
    L = {}
    flora_r = np.zeros(n, np.int32)
    fauna_r = np.zeros(n, np.int32)
    func_r = np.zeros(n, np.int32)
    exp_r = np.zeros(n)
    groups_present: dict = {}
    # range sizes of every taxon (template x lineage) over the whole domain -> endemism
    w_end = np.zeros(n)
    endemic_cnt = np.zeros(n, np.int32)
    ranges = {}
    edge = _edge_mask(inp)
    taxa_total = 0
    for realm, tab in realms.items():
        for k, tid in enumerate(tab.templates):
            tpl = reg.templates[tid]
            g = reg.groups[tpl.group_id]
            pres = tab.present[k]
            if not pres.any():
                continue
            idx = tab.idx[pres]
            if g.is_flora:
                flora_r[idx] += 1
            elif g.is_fauna:
                fauna_r[idx] += 1
            groups_present.setdefault(g.id, np.zeros(n, bool))[idx] = True
            exp_r[tab.idx] += (tab.occ[k] / 255.0) * (tab.suit[k] >= int(S_MIN * 255))
            lin = lineage[g.traits["mob"]]["pool"][idx].astype(np.int64)
            lin = np.where(lin < 0, 0, lin)
            u, inv, cnts = np.unique(lin, return_inverse=True, return_counts=True)
            rng_area = cnts * area
            touches = np.zeros(u.size, bool)
            np.logical_or.at(touches, inv, edge[idx])
            for j, lj in enumerate(u.tolist()):
                fam = family_of(g.traits["mob"])
                key = fam_pools[fam].keys[lj] if fam_pools[fam].keys else "none"
                ranges[taxon_id(seed, tid, key)] = (float(rng_area[j]), bool(touches[j]), tid, key)
            taxa_total += u.size
            w_end[idx] += 1.0 / np.maximum(rng_area[inv] / area, 1.0)
            endemic_cnt[idx] += ((rng_area[inv] <= ENDEMIC_RANGE_KM2) & ~touches[inv]).astype(np.int32)
    for _gid, m in groups_present.items():
        func_r += m
    L["flora_richness"] = flora_r
    L["fauna_richness"] = fauna_r
    L["total_richness"] = flora_r + fauna_r
    L["functional_richness"] = func_r
    L["expected_richness"] = exp_r.astype(np.float32)
    L["weighted_endemism"] = w_end.astype(np.float32)
    L["endemic_taxa"] = endemic_cnt
    # productivity and stocks per m² of the hex
    npp = np.zeros(n)
    for g in groups_out.values():
        if "npp_gc_m2_hex" in g:
            npp += np.nan_to_num(g["npp_gc_m2_hex"])
    L["npp_gc_m2_yr"] = npp.astype(np.float32)
    bm = np.zeros(n)
    bm[li] = (prod_r["stock_leaf"] + prod_r["stock_wood"] + prod_r["stock_root"]
              + prod_r["stock_repro"]).sum(axis=0) * frac["land_ground"][li]  # fmt: skip
    L["plant_biomass_kgc_m2"] = bm.astype(np.float32)
    wood = np.zeros(n)
    wood[li] = prod_r["stock_wood"].sum(axis=0) * frac["land_ground"][li]
    L["wood_kgc_m2"] = wood.astype(np.float32)
    wood_inc = np.zeros(n)
    wood_inc[li] = prod_r["npp_wood"].sum(axis=0) * frac["land_ground"][li] / 1000.0
    L["wood_increment_kgc_m2_yr"] = wood_inc.astype(np.float32)
    an = np.zeros(n)
    for g_id, g in groups_out.items():
        if "biomass_kgc" in g:
            an += g["biomass_kgc"]
    L["animal_biomass_kgc_km2"] = (an / area).astype(np.float32)
    # Shannon of the flora (by cover) and of the fauna (by biomass) per hex
    covers = np.stack([np.nan_to_num(g["cover"]) for gid, g in groups_out.items()
                       if "cover" in g and reg.groups[gid].traits["layer"] in TERRESTRIAL_LAYERS])
    L["flora_shannon_groups"] = mt.shannon_rows(covers).astype(np.float32)
    fb = np.stack([g["biomass_kgc"] for gid, g in groups_out.items()
                   if "biomass_kgc" in g and reg.groups[gid].is_fauna])  # fmt: skip
    L["fauna_shannon_groups"] = mt.shannon_rows(fb).astype(np.float32)
    for k in ("flora_richness", "fauna_richness", "total_richness", "npp_gc_m2_yr",
              "weighted_endemism"):  # fmt: skip
        lo, hi = mt.INDEX_BOUNDS[k if k != "plant_biomass_kgc_m2" else "biomass_kgc_m2"]
        L[f"{k}_index"] = mt.index(L[k], lo, hi).astype(np.float32)
    life = (frac["land_ground"] + frac["inland_water"] + frac["marine_water"]) > 0
    L["total_richness_percentile"] = mt.percentile_area_weighted(
        L["total_richness"].astype(float), area, life).astype(np.float32)
    # biogeographic regions: the combination of lineages of the indicator classes
    sig_cls = ("ground", "seed", "flight", "river", "ocean")
    sig = np.zeros(n, dtype=np.int64)
    for c in sig_cls:
        sig = sig * 64 + (lineage[c]["pool"].astype(np.int64) + 1)
    land = frac["land_ground"] > 0
    sig = np.where(land, sig, -(lineage["ocean"]["pool"].astype(np.int64) + 2))
    u, inv = np.unique(sig, return_inverse=True)
    L["bioregion"] = inv.astype(np.int32)
    L["land_lineage"] = lineage["ground"]["pool"].astype(np.int16)
    L["marine_lineage"] = lineage["ocean"]["pool"].astype(np.int16)
    d_eff = np.where(land, lineage["ground"]["d_km"], lineage["ocean"]["d_km"])
    L["isolation_km"] = np.where(np.isfinite(d_eff), d_eff, np.nan).astype(np.float32)
    L["age_unknown"] = age_unknown
    # fresh / marine life
    fr = np.zeros(n)
    mr = np.zeros(n)
    for realm, tab in realms.items():
        cnt = tab.present.sum(axis=0)
        if realm == "fresh":
            fr[tab.idx] += cnt
        elif realm == "marine":
            mr[tab.idx] += cnt
    L["freshwater_richness"] = fr.astype(np.int32)
    L["marine_richness"] = mr.astype(np.int32)
    summary = {
        "taxa_world": len(ranges),
        "taxa_endemic": int(sum(1 for a, tch, _t, _k in ranges.values()
                                if a <= ENDEMIC_RANGE_KM2 and not tch)),
        "taxa_range_unknown": int(sum(1 for _a, tch, _t, _k in ranges.values() if tch)),
        "lineage_pools": {k: len(v.keys) for k, v in fam_pools.items()},
        "bioregions": int(u.size),
        "npp_mean_land_gc_m2_yr": float(np.nanmean(npp[land])) if land.any() else 0.0,
        "flora_richness_max": int(flora_r.max()) if n else 0,
        "fauna_richness_max": int(fauna_r.max()) if n else 0,
        "index_version": mt.INDEX_VERSION,
        "endemic_range_km2": ENDEMIC_RANGE_KM2,
    }
    L["_ranges"] = ranges
    return L, summary


ENDEMIC_RANGE_KM2 = 1.0e5


def _edge_mask(inp):
    """Hexes on the edge of a regional window: a range that touches it is not known in full
    (B61); a planetary domain wraps and has no such edge."""
    h, w = inp.shape
    m = np.zeros((h, w), bool)
    m[0, :] = m[-1, :] = True
    m[:, 0] = m[:, -1] = True
    return m.ravel()
