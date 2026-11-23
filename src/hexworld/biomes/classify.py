"""Causal classification of dry terrain units into the biome registry (TZ 5.0, 7, 8, 20.3).

Order (first match wins; every rule needs its physical cause):

1. ice / firn / perennial snow dominate the unit - the surface classes (nunatak for an ice-free
   rock unit inside an ice body);
2. recent lava / ash without soil - ``barren_volcanic_recent``;
3. wetlands: months of waterlogging (4.5) in a low, flat unit - saline / swamp forest / peat /
   floodplain / marsh by salinity, trees and cold; ``wetland_unclassified`` when the type of
   water supply is not known;
4. riparian strips: a river hex in a dry climate gives its valley unit a gallery class;
5. very low vegetation fraction - deserts and barrens **by cause and substrate** (hot sand /
   gravel / rock / salt / fog coast, cold, polar, high-mountain, cliff, badlands);
6. above the thermal tree limit (4.7) - tundra / alpine classes;
7. tree canopy >= 0.6 forest, 0.1-0.6 woodland, else grass / shrub - the type from the
   dominant PFT of each form and the seasonality.

The potential natural vegetation is classified, never a season: a deciduous forest in winter
stays a forest, the snow is an overlay.
"""

from __future__ import annotations

import numpy as np

from ..stages.environment import PM as _PM
from . import pft
from .registry import INDEX, LF_BIT, MOD_BIT

#: parent materials of poor, acid soils (oceanic heath)
PM_POOR = {k: _PM[k] for k in ("aeolian_sand", "crystalline_bedrock", "glacial_till")}

I = INDEX  # noqa: E741


def classify_units(c: dict) -> dict:
    """Classify a block of dry units. ``c`` holds (U,) arrays (see the pipeline)."""
    u = c["tc"].size
    out = np.full(u, I["unresolved"], dtype=np.uint16)
    lim = np.zeros(u, dtype=np.int64)
    mods = np.zeros(u, dtype=np.int64)
    done = np.zeros(u, dtype=bool)

    def put(mask, biome):
        nonlocal done
        m = mask & ~done
        out[m] = I[biome]
        done |= m
        return m

    tc, tw, gdd0, gdd5 = c["tc"], c["tw"], c["gdd0"], c["gdd5"]
    alpha, dry_m, sdry = c["alpha"], c["dry_months"], c["summer_dry"]
    v, cover = c["veg_frac"], c["pft_cover"]
    tree = cover[pft.IS_TREE].sum(axis=0)
    shrub = cover[pft.IS_SHRUB].sum(axis=0)
    herb = cover[pft.IS_HERB].sum(axis=0)
    canopy = np.where(v > 0, tree / np.maximum(v, 1e-9), 0.0)
    shrub_sh = np.where(v > 0, shrub / np.maximum(v, 1e-9), 0.0)
    herb_sh = np.where(v > 0, herb / np.maximum(v, 1e-9), 0.0)
    tree_cover_abs = tree
    dom_tree = np.argmax(np.where(pft.IS_TREE[:, None], cover, -1.0), axis=0)
    # flood-tolerant trees make a floodplain forest only on floodplains and along rivers;
    # elsewhere the zonal tree type of the unit names its forest
    fl = pft.PFT_INDEX["TREE_FLOOD_TOLERANT"]
    # (a river in a 40 km hex does not make the hex a floodplain)
    on_flood = (c["floodplain"] > 0.3) | (
        c["river"] & (c["waterlog"] >= 4.0) & (c["slope_deg"] < 0.5) & (c["valley"] > 0.2)
    )
    others = np.where(pft.IS_TREE[:, None], cover, -1.0)
    others[fl] = -1.0
    second = np.argmax(others, axis=0)
    swap = (dom_tree == fl) & ~on_flood & (others.max(axis=0) > 0.0)
    dom_tree = np.where(swap, second, dom_tree)
    mountain = c["mountain"]
    z = c["z"]
    ai = c["ai"]
    hyper = np.nan_to_num(ai, nan=9.0) < 0.05
    arid = np.nan_to_num(ai, nan=9.0) < 0.2
    salt = c["salinity"]
    wlog = c["waterlog"]
    slope = c["slope_deg"]
    pm = c["parent"]  # parent material code (4.5)
    rocky = c["rocky"]
    cold_polar = gdd0 < 350.0
    margin = c["thermal_margin"]

    # --- 1. snow, firn, ice ----------------------------------------------------------------
    ice_c, firn_c, snow_c = c["ice_cover"], c["firn_cover"], c["snow_cover"]
    lim |= np.where(ice_c + firn_c + snow_c > 0.5, LF_BIT["ice"], 0)
    ice_form = c["ice_form"]  # 0 none, 1 mountain glacier, 2 ice cap, 3 ice sheet
    ablation = c["ice_balance"] < 0
    is_ice = ice_c > 0.5
    # the ablation zone is the margin that loses more than a few decimetres a year, not every
    # part of an ice body whose balance dips below zero
    put(is_ice & ablation & (c["ice_balance"] < -300.0) & (ice_form > 0), "glacier_ablation_zone")
    put(is_ice & (ice_form == 3), "glacier_ice_sheet")
    put(is_ice & (ice_form == 2), "glacier_ice_cap")
    put(is_ice, "glacier_mountain")
    put(firn_c > 0.5, "firn_field")
    put(snow_c > 0.5, "snow_perennial_field")
    nunatak = (c["in_ice_body"]) & (ice_c + firn_c + snow_c < 0.5) & ((slope > 20) | (rocky > 0.5))
    put(nunatak, "barren_nunatak")

    # --- 2. recent volcanic surfaces --------------------------------------------------------
    young_volc = (pm == c["pm_young_volcanic"]) & (c["volcanic_age"] < 1.0) & (v < 0.5)
    mods |= np.where(pm == c["pm_young_volcanic"], MOD_BIT["volcanic_ash_young"], 0)
    lim |= np.where(young_volc, LF_BIT["young_volcanic"], 0)
    put(young_volc, "barren_volcanic_recent")

    # --- 3. wetlands -------------------------------------------------------------------------
    # standing water half the year, or 4+ months on very poorly drained ground: a hex that is
    # merely damp for a season is not a swamp
    low_flat = (slope < 4.0) & (c["rank"] < 0.6)
    drain = c.get("drain")
    very_poor = (drain >= 5) if drain is not None else np.ones(u, dtype=bool)
    wet = ((wlog >= 6.0) | ((wlog >= 4.0) & very_poor)) & low_flat & ~hyper
    mods |= np.where(wlog >= 2.0, MOD_BIT["seasonal_flood"], 0)
    lim |= np.where(wlog >= 4.0, LF_BIT["flooding"], 0)
    put(wet & (salt > 0.45), "wetland_inland_saline")
    # salt marshes of closed basins: seasonally wet, saline lowest ground
    saline_basin = c["endorheic"] & (wlog >= 2.0) & (salt > 0.3) & (slope < 2.0) & ~hyper
    put(saline_basin & (c["rank"] <= 0.5), "wetland_inland_saline")
    trees_wet = wet & (tree_cover_abs > 0.3)
    put(trees_wet & (tc > 15.0), "forest_tropical_swamp")
    put(trees_wet & (gdd5 < 1300), "boreal_peat_swamp_forest")
    put(trees_wet, "forest_wetland_temperate")
    peat = wet & (c["organic"] > 0.6) & (gdd5 < 2200)
    mods |= np.where(peat, MOD_BIT["peat_accumulation"], 0)
    put(peat & (c["drainage_poor_rain"] > 0.5), "wetland_bog")
    # minerotrophic peat: fed by rivers or groundwater rather than rain
    put(peat & (c["river"] | (c["floodplain"] > 0.2)), "wetland_fen")
    put(peat, "wetland_peatland_mosaic")
    put(wet & ((c["floodplain"] > 0.5) | c["river"]), "wetland_floodplain_seasonal")
    put(wet & (wlog >= 6.0), "wetland_fresh_marsh")
    put(wet, "wetland_unclassified")

    # --- 4. riparian galleries in dry climates -------------------------------------------------
    rip = c["river"] & (c["valley"] > 0.3) & (alpha < 0.5) & (gdd5 > 900) & (c["rank"] < 0.5)
    mods |= np.where(rip, MOD_BIT["river_oasis"], 0)
    put(rip & (tc > 15.0) & (tree_cover_abs > 0.1), "forest_tropical_floodplain")
    put(rip & (tree_cover_abs > 0.15), "forest_floodplain_temperate")
    put(rip, "wetland_riparian_gallery")

    # --- 5. deserts and barrens by cause ------------------------------------------------------
    barren = v < 0.10
    hot = tw >= 22.0
    lim |= np.where(barren & hyper & hot, LF_BIT["hyperarid_heat"], 0)
    lim |= np.where(barren & arid, LF_BIT["drought"], 0)
    lim |= np.where(barren & cold_polar & ~arid, LF_BIT["cold"], 0)
    lim |= np.where(barren & (rocky > 0.5), LF_BIT["rock"], 0)
    lim |= np.where(salt > 0.5, LF_BIT["salinity"], 0)
    mods |= np.where(salt > 0.45, MOD_BIT["saline_soil"], 0)
    put(barren & (slope > 38.0), "barren_rock_cliff")
    # (polar mountains too: a summit with a -10 degC summer is rock, not alpine tundra)
    polar = barren & (tw < 10.0) & cold_polar & (~mountain | (tw < 5.0))
    lim |= np.where(polar & (c["p_year"] < 250), LF_BIT["cold_polar_dry"], 0)
    put(polar & (v < 0.01), "barren_polar_rock")
    put(polar, "tundra_polar_desert_sparse")
    high_cold = barren & mountain & (tc < 0.0) & (z > 2500)
    put(high_cold & arid, "desert_high_mountain_cold")
    put(high_cold & (rocky > 0.4), "scree_alpine_sparse")
    put(high_cold, "desert_high_mountain_cold")
    hot_dry = barren & hot & arid
    endor = c["endorheic"]
    # salt crusts and flats only on the lowest, flattest ground of a closed basin (the playa
    # floor), not on every saline soil of an arid region
    pan = endor & (c["rank"] < 0.25) & (slope < 2.0)
    put(hot_dry & pan & (salt > 0.7) & (c["rank"] < 0.12), "desert_hot_salt_flat")
    # takyr / playa: the flat clay floor of a dry lake that fills now and then
    put(barren & arid & pan & ((wlog >= 0.5) | (c["rank"] < 0.18)), "desert_playa_dry_lake")
    put(hot_dry & pan & (salt > 0.55), "desert_saline_crust")
    put(hot_dry & (c["fog"] > 0.2) & c["coastal"], "desert_hot_fog_coast")
    put(hot_dry & (pm == c["pm_aeolian"]), "desert_hot_sand_erg")
    put(hot_dry & (c["badlands"] > 0.5), "desert_badlands_barren")
    # gypsum and alkali flats: hyper-arid closed basins, moderately saline (no gypsum model)
    # (the gentle rims of hyper-arid closed basins, around their salt pans)
    gyps = (c["gypsum"] > 0.5) | (endor & hyper & ~pan & (slope < 5.0))
    put(hot_dry & gyps, "barren_gypsum_alkali")
    put(hot_dry & ((rocky > 0.5) | (slope > 12.0)), "desert_hot_rock_hamada")
    put(hot_dry, "desert_hot_gravel_reg")
    put(barren & (tc < -2.0) & arid, "desert_temperate_cold")
    put(barren & arid & (v >= 0.02), "desert_semi_arid_sparse")
    put(barren & arid, "desert_sparse_undifferentiated")
    put(barren & (rocky > 0.5), "barren_rock_cliff")
    put(barren & (slope > 25.0) & (c["erosion"] > 0.6), "barren_erosional_surface")

    # --- 6. above the thermal tree limit: tundra and alpine -----------------------------------
    above = (margin < -0.3) & (tree_cover_abs < 0.15)
    mods |= np.where(c["permafrost"] > 0.6, MOD_BIT["permafrost_polygon_proxy"], 0)
    lim |= np.where(above, LF_BIT["short_season"], 0)
    alp = above & mountain
    # paramo: cold *without winter* (small annual range) and wet, above the tropical tree
    # limit - a snowy continental steppe is not paramo, the dry puna stays alpine steppe
    no_winter = (tw - tc) < 6.0
    put(alp & no_winter & (tc > -6.0) & (alpha > 0.6) & (rocky < 0.6), "tropical_montane_paramo")
    put(alp & (rocky > 0.45) & (v < 0.3), "scree_alpine_sparse")
    put(alp & (alpha < 0.3), "steppe_alpine")
    put(alp & (shrub_sh > 0.4), "heath_alpine")
    put(alp & (margin > -1.0) & (herb_sh > 0.4), "meadow_subalpine")
    put(alp & (herb_sh > 0.3), "grassland_alpine")
    put(alp, "tundra_alpine")
    put(above & (wlog >= 2.0), "tundra_sedge_graminoid")
    put(above & (shrub_sh > 0.35) & (gdd0 > 300), "tundra_shrub")
    put(above & (gdd0 < 250), "tundra_moss_lichen")
    put(above, "tundra_sedge_graminoid")

    # --- 7. forests, woodlands, grass and shrub --------------------------------------------
    ids = pft.PFT_INDEX
    d = dom_tree
    # boreal and larch forests are naturally open (crown cover 30-60 % counts as taiga)
    boreal_tree = np.isin(d, [ids["TREE_BOREAL_EVERGREEN"], ids["TREE_BOREAL_DECID"]])
    forest = ((canopy >= 0.6) | (boreal_tree & (canopy >= 0.4))) & (v >= 0.4)
    woodland = (canopy >= 0.1) & ~forest
    cover_of = {k: cover[ids[k]] for k in ids}
    con = cover_of["TREE_TEMP_CONIFER"] + cover_of["TREE_BOREAL_EVERGREEN"]
    broad = cover_of["TREE_TEMP_DECID"] + cover_of["TREE_TEMP_EVERGREEN"]
    mixed = (np.minimum(con, broad) > 0.6 * np.maximum(con, broad)) & (con + broad > 0)
    cloud = (d == ids["TREE_MONTANE_CLOUD"]) | (mountain & (c["cloud"] > 0.15) & (tc > 8))
    mods |= np.where(c["cloud"] > 0.35, MOD_BIT["montane_cloud_belt"], 0)
    put(forest & cloud & (tc > 8.0), "forest_tropical_cloud_montane")
    put(forest & (d == ids["TREE_TROP_EVERGREEN"]), "forest_tropical_everwet")
    put(forest & (d == ids["TREE_TROP_SEASONAL"]) & (dry_m <= 4.5), "forest_tropical_monsoon")
    put(forest & (d == ids["TREE_TROP_SEASONAL"]), "forest_tropical_seasonal_dry")
    put(forest & (d == ids["TREE_TROP_DRY"]), "forest_tropical_seasonal_dry")
    sub_ever = forest & (d == ids["TREE_TEMP_EVERGREEN"])
    put(sub_ever & (alpha > 0.95) & (c["p_year"] > 1800) & (tc < 10), "forest_temperate_rainforest")
    put(sub_ever & (sdry >= 2.0), "forest_mediterranean_sclerophyll")
    # a dry winter (monsoon subtropics) rather than a dry summer
    put(sub_ever & (tc > 7.0) & (dry_m >= 2.0), "forest_subtropical_seasonal")
    put(sub_ever & (tc > 7.0), "forest_subtropical_evergreen")
    put(sub_ever, "forest_temperate_evergreen_broadleaf")
    temp = forest & np.isin(d, [ids["TREE_TEMP_DECID"], ids["TREE_TEMP_CONIFER"]])
    put(temp & mountain & ((z > 1200.0) | (margin < 1.5)), "forest_montane_temperate")
    put(temp & (alpha > 0.95) & (c["p_year"] > 2000), "forest_temperate_rainforest")
    put(temp & mixed, "forest_temperate_mixed")
    put(forest & (d == ids["TREE_TEMP_DECID"]), "forest_temperate_deciduous")
    put(forest & (d == ids["TREE_TEMP_CONIFER"]), "forest_temperate_conifer")
    bor = forest & np.isin(d, [ids["TREE_BOREAL_EVERGREEN"], ids["TREE_BOREAL_DECID"]])
    b_ev, b_de = cover_of["TREE_BOREAL_EVERGREEN"], cover_of["TREE_BOREAL_DECID"]
    bmix = np.minimum(b_ev, b_de) > 0.65 * np.maximum(b_ev, b_de)
    put(bor & bmix & (d != ids["TREE_BOREAL_DECID"]), "forest_boreal_mixed")
    put(bor & (d == ids["TREE_BOREAL_DECID"]), "forest_boreal_deciduous_larch")
    put(bor & (alpha > 0.8), "forest_boreal_dark_conifer")
    put(bor, "forest_boreal_light_conifer")
    flood_forest = forest & (d == ids["TREE_FLOOD_TOLERANT"])
    put(flood_forest & on_flood & (tc > 15.0), "forest_tropical_floodplain")
    put(flood_forest & on_flood, "forest_floodplain_temperate")
    # flood-tolerant trees off the floodplain (no other tree type there): a swamp forest only
    # on real wetland ground, otherwise the zonal forest of the climate - soil that is wet
    # for a few months is not a swamp
    put(flood_forest & wet & (tc > 15.0), "forest_tropical_swamp")
    put(flood_forest & wet, "forest_wetland_temperate")
    put(flood_forest & (tc > 15.0) & (dry_m <= 2.0), "forest_tropical_everwet")
    put(flood_forest & (tc > 15.0), "forest_tropical_monsoon")
    put(flood_forest & (gdd5 < 1300), "forest_boreal_dark_conifer")
    put(forest, "forest_temperate_mixed")

    trop = tc > 15.0
    boreal_like = gdd5 < 1100
    near_limit = (margin > -0.3) & (margin < 0.6)
    put(woodland & boreal_like & near_limit & (tc < -5), "forest_tundra_mosaic")
    put(woodland & boreal_like, "woodland_boreal_sparse")
    # thorny, sparse dry-tropical woodland with a long dry season is caatinga-like scrub
    thorn_w = trop & (dry_m >= 7.0) & (alpha > 0.12) & (alpha < 0.4) & (canopy < 0.35)
    put(woodland & thorn_w, "tropical_thorn_scrub")
    put(woodland & trop & (canopy > 0.35), "woodland_dry_tropical")
    put(woodland & trop, "savanna_tree_grass")
    put(woodland & (sdry >= 2.0) & (tc > 0), "forest_mediterranean_sclerophyll")
    put(woodland & (tc > 13.0), "woodland_tropical_dry")
    put(woodland & (herb_sh > 0.35) & (tc < 0), "forest_steppe_mosaic")
    put(woodland, "woodland_temperate_open")

    open_ = ~done
    # open ground of a tundra climate (summers too cool and short) is tundra, not prairie
    tundra_clim = open_ & ((tw < 10.0) | (gdd5 < 350.0)) & ~trop
    put(tundra_clim & (wlog >= 2.0), "tundra_sedge_graminoid")
    put(tundra_clim & (shrub_sh > 0.35) & (gdd0 > 300), "tundra_shrub")
    put(tundra_clim & (gdd0 < 250), "tundra_moss_lichen")
    put(tundra_clim, "tundra_sedge_graminoid")
    # caatinga-like thorn scrub: hot, a long dry season, water stress, a shrubby cover
    thorn = trop & (dry_m >= 6.0) & (alpha > 0.12) & (alpha < 0.45) & (shrub_sh > 0.3)
    put(open_ & thorn, "tropical_thorn_scrub")
    put(open_ & trop & (shrub_sh > 0.35), "savanna_shrub_grass")
    put(open_ & trop, "savanna_open_grass")
    put(open_ & (shrub_sh > 0.45) & (sdry >= 2.0) & (tc > 0), "shrubland_mediterranean")
    # oceanic heath: a mild, wet oceanic climate on poor, acid ground (sand, till, granite);
    # a fertile sheltered basin (alluvium, loess) never qualifies
    oceanic = ((tw - tc) < 13.0) & (tc > -3.0) & (tw < 20.0) & (alpha > 0.8)
    poor = np.isin(pm, [PM_POOR[k] for k in PM_POOR])
    put(open_ & oceanic & poor & (shrub_sh > 0.25), "oceanic_heathland")
    put(open_ & (shrub_sh > 0.45) & (tc > 7.0), "scrub_subtropical")
    put(open_ & (shrub_sh > 0.45), "shrubland_temperate_dry")
    put(open_ & mountain & (margin < 1.0) & (margin > -0.3), "meadow_subalpine")
    put(open_ & mountain & (alpha > 0.4), "grassland_montane")
    put(open_ & (alpha >= 0.5), "grassland_temperate_prairie")
    put(open_ & (alpha >= 0.25), "grassland_steppe")
    put(open_, "grassland_dry_steppe")
    lim |= np.where((dry_m >= 4) & ~barren, LF_BIT["seasonal_drought"], 0)
    lim |= np.where(c["soil_depth"] < 0.2, LF_BIT["soil_shallow"], 0)
    lim |= np.where(c["snow_months"] >= 7, LF_BIT["snow"], 0)
    lim |= np.where((gdd5 < 600) & ~barren, LF_BIT["cold"], 0)
    mods |= np.where(c["cap"] > 0.5, MOD_BIT["cold_air_pool"], 0)
    mods |= np.where((c["fog"] > 0.3) & arid, MOD_BIT["fog_oasis"], 0)
    mods |= np.where(c["coastal"] & (c["marine"] > 0.7), MOD_BIT["coastal_salt_spray"], 0)
    mods |= np.where((slope > 30) & (rocky > 0.4), MOD_BIT["scree"], 0)
    mods |= np.where(c["shadow"], MOD_BIT["north_slope_solar_shadow"], 0)
    fire = (dry_m >= 3) & (v > 0.3) & (gdd5 > 1500)
    mods |= np.where(fire, MOD_BIT["fire_prone"], 0)
    return {
        "biome": out,
        "limiting": lim,
        "modifiers": mods,
        "canopy_fraction": canopy * v,
        "shrub_fraction": shrub,
        "herb_fraction": herb,
    }
