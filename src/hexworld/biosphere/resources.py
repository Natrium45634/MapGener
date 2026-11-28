"""Renewable natural resources of TZ 5.5 §12 (contract C21): stocks and annual increments
derived read-only from the biosphere state. Potential, not extraction: no human activity
exists in 5.5.

* a stock (t, m³) and its annual increment (t/yr) are separate quantities (B64);
* alternative uses of one physical pool share a ``pool`` key and are not additive (B65):
  timber and fuelwood are two views of the same wood increment;
* ``harvest_cap`` = safe share × surplus, never negative;
* MSY = r K / 4 only for a stock declared logistic (N19, B66);
* wild precursors (domestication potential) are a score separate from the accessibility
  (B67).
"""

from __future__ import annotations

import numpy as np

from .foodweb import C_PER_WET
from .production import C_PER_DRY, WOOD_DENSITY_T_M3

#: safe harvest share of the surplus (provisional) [К]
SAFE_SHARE = {"wood": 0.5, "forage": 0.5, "game": 0.2, "fish_fresh": 0.3, "fish_marine": 0.3,
              "fruit": 0.3}  # fmt: skip
GAME_GROUPS = ("grazer_large", "browser_large", "grazer_small", "omnivore_land", "megaherbivore",
               "bird_waterfowl")  # fmt: skip
FISH_FRESH = ("fish_fresh_grazer", "fish_fresh_omnivore", "fish_fresh_predator", "crustacean_fresh")
FISH_MARINE = ("nekton_forage", "nekton_medium", "fish_dem_grazer", "fish_dem_predator",
               "crustacean_marine", "fish_anadromous", "benthic_filter")  # fmt: skip
#: wild precursors: group -> (title, domestication traits score 0..1) [К]
PRECURSORS = {
    "grass_c3": ("злаки C3 — предки зерновых", 0.9),
    "grass_c4": ("злаки C4 — предки проса и сорго", 0.8),
    "forb": ("бобовые и корнеплодные травы", 0.5),
    "tree_palm": ("плодовые пальмы", 0.4),
    "grazer_large": ("стадные копытные — предки скота", 0.9),
    "grazer_small": ("мелкие травоядные", 0.4),
    "bird_waterfowl": ("водоплавающие птицы", 0.6),
    "bird_granivore": ("куриные и зерноядные птицы", 0.5),
    "browser_large": ("олени и лоси", 0.3),
    "insect_pollinator": ("общественные опылители (мёд)", 0.5),
}


def logistic_msy(r, K):
    """Maximum sustainable yield of a logistic stock: r K / 4 (valid only under that model)."""
    return np.asarray(r, dtype=np.float64) * np.asarray(K, dtype=np.float64) / 4.0


def inventory(state, inp_area_km2: float) -> dict:
    """Per-hex inventory (arrays) and world totals with units and pool keys."""
    L = state.layers
    G = state.groups
    area_m2 = inp_area_km2 * 1e6
    n = state.n
    out: dict = {}
    wood_c = L["wood_kgc_m2"].astype(np.float64) * area_m2  # kgC per hex
    inc_c = L["wood_increment_kgc_m2_yr"].astype(np.float64) * area_m2
    dry = 1.0 / C_PER_DRY
    out["timber_stock_m3"] = wood_c * dry / 1000.0 / WOOD_DENSITY_T_M3
    out["timber_increment_m3_yr"] = inc_c * dry / 1000.0 / WOOD_DENSITY_T_M3
    out["fuelwood_increment_t_yr"] = inc_c * dry / 1000.0  # the same wood: not additive
    # forage: edible herb production not eaten by the wild grazers (surplus)
    forage = np.zeros(n)
    for g in ("grass_c3", "grass_c4", "forb", "sedge"):
        forage += np.nan_to_num(G[g]["npp_gc_m2_hex"].astype(np.float64)) * 0.45 * 0.7
    from .registry import load

    reg = load()
    wild = np.zeros(n)  # what the wild herbivores already eat of the herb pool, gC/m² of hex
    for g, v in G.items():
        share = reg.groups[g].traits.get("diet", {}).get("leaf_herb", 0.0)
        if share > 0 and "intake_kgc_yr" in v:
            wild += v["intake_kgc_yr"].astype(np.float64) * share * 1000.0 / area_m2
    out["forage_surplus_t_dm_yr"] = np.maximum(forage - wild, 0.0) * area_m2 / 1000.0 * dry / 1000.0
    game = sum(G[g]["production_kgc_yr"].astype(np.float64) for g in GAME_GROUPS)
    out["game_production_t_wet_yr"] = game / C_PER_WET / 1000.0
    ff = sum(G[g]["production_kgc_yr"].astype(np.float64) for g in FISH_FRESH)
    fm = sum(G[g]["production_kgc_yr"].astype(np.float64) for g in FISH_MARINE)
    out["fish_fresh_production_t_wet_yr"] = ff / C_PER_WET / 1000.0
    out["fish_marine_production_t_wet_yr"] = fm / C_PER_WET / 1000.0
    fruit = np.zeros(n)
    for g, v in G.items():
        if "npp_gc_m2_hex" in v and g.startswith(("tree_", "shrub_")):
            fruit += np.nan_to_num(v["npp_gc_m2_hex"].astype(np.float64)) * 0.05
    out["wild_fruit_t_dm_yr"] = fruit * area_m2 / 1000.0 * dry / 1000.0
    caps = {
        "timber_increment_m3_yr": SAFE_SHARE["wood"],
        "fuelwood_increment_t_yr": SAFE_SHARE["wood"],
        "forage_surplus_t_dm_yr": SAFE_SHARE["forage"],
        "game_production_t_wet_yr": SAFE_SHARE["game"],
        "fish_fresh_production_t_wet_yr": SAFE_SHARE["fish_fresh"],
        "fish_marine_production_t_wet_yr": SAFE_SHARE["fish_marine"],
        "wild_fruit_t_dm_yr": SAFE_SHARE["fruit"],
    }
    for k, s in caps.items():
        out[k.replace("_yr", "_cap_yr").replace("increment", "harvest").replace(
            "production", "harvest").replace("surplus", "harvest")] = np.maximum(out[k] * s, 0.0)
    # wild precursors: trait score × local presence (accessibility is a separate field)
    prec = np.zeros(n)
    for g, (_t, score) in PRECURSORS.items():
        w = G[g].get("weight")
        if w is not None:
            prec = np.maximum(prec, score * np.asarray(w, dtype=np.float64))
    out["wild_precursor_score"] = prec
    totals = {k: float(np.nansum(v)) for k, v in out.items() if not k.endswith("score")}
    meta = {
        "units": {k: k.rsplit("_", 2)[-2] + "/" + k.rsplit("_", 1)[-1] if k.endswith("_yr")
                  else k.rsplit("_", 1)[-1] for k in out},  # fmt: skip
        "pools": {"timber_increment_m3_yr": "wood", "fuelwood_increment_t_yr": "wood",
                  "timber_stock_m3": "wood_stock", "forage_surplus_t_dm_yr": "herb",
                  "game_production_t_wet_yr": "game", "fish_fresh_production_t_wet_yr": "fish_f",
                  "fish_marine_production_t_wet_yr": "fish_m", "wild_fruit_t_dm_yr": "fruit"},
        "not_additive": [["timber_increment_m3_yr", "fuelwood_increment_t_yr"]],
        "status": "potential, provisional parameters; no harvest is simulated",
    }  # fmt: skip
    return {"layers": out, "totals": totals, "meta": meta}
