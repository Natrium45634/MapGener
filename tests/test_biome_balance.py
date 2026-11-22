"""Biome balance: rare classes can appear, tidal fringes are a fraction of a hex, taiga wins
in boreal climates, fog and cloud reach the classifier."""

from __future__ import annotations

import numpy as np
import pytest

from hexworld.biomes import classify as bc
from hexworld.biomes import pft
from hexworld.biomes.registry import BIOMES, INDEX
from hexworld.pipeline_v50 import tidal_fringe
from hexworld.stages.coasts import COAST_TYPES


def _units(n=1, **over):
    """Classifier input of ``n`` identical units (a temperate, moist lowland by default)."""
    P = len(pft.TRAITS)
    base = {
        "tc": -2.0, "tw": 18.0, "gdd0": 2500.0, "gdd5": 1600.0, "alpha": 0.9, "dry_months": 0.0,
        "summer_dry": 0.0, "veg_frac": 0.9, "pft_cover": np.zeros((P, 1)), "mountain": False,
        "z": 200.0, "ai": 1.2, "salinity": 0.0, "waterlog": 0.0, "slope_deg": 1.0, "parent": 0,
        "rocky": 0.0, "thermal_margin": 2.0, "ice_cover": 0.0, "firn_cover": 0.0,
        "snow_cover": 0.0, "ice_form": 0, "ice_balance": 0.0, "in_ice_body": False,
        "pm_young_volcanic": 99, "pm_aeolian": 98, "volcanic_age": 1e9, "rank": 0.5,
        "organic": 0.0, "drainage_poor_rain": 0.0, "floodplain": 0.0, "river": False,
        "valley": 0.0, "p_year": 900.0, "endorheic": False, "fog": 0.0, "coastal": False,
        "badlands": 0.0, "gypsum": 0.0, "erosion": 0.0, "permafrost": 0.0, "cloud": 0.0,
        "soil_depth": 1.0, "snow_months": 0.0, "cap": 0.0, "marine": 0.0, "shadow": False,
        "coast": 0, "drain": 1,
    }  # fmt: skip
    base.update(over)
    out = {}
    for k, v in base.items():
        if k == "pft_cover":
            out[k] = np.repeat(np.asarray(v, dtype=np.float64), n, axis=1)
        elif k.startswith("pm_"):
            out[k] = v
        else:
            out[k] = np.full(n, v)
    return out


def _cover(**shares):
    cov = np.zeros((len(pft.TRAITS), 1))
    for name, x in shares.items():
        cov[pft.PFT_INDEX[name], 0] = x
    return cov


def _name(c):
    return BIOMES[int(bc.classify_units(c)["biome"][0])].id


def test_taiga_is_a_forest_at_open_canopy():
    c = _units(tc=-18.0, tw=16.0, gdd5=900.0, alpha=0.9,
               pft_cover=_cover(TREE_BOREAL_EVERGREEN=0.45, SHRUB_DWARF_COLD=0.45))  # fmt: skip
    assert _name(c) == "forest_boreal_dark_conifer"
    c["alpha"] = np.full(1, 0.6)
    assert _name(c) == "forest_boreal_light_conifer"


def test_mixed_forest_needs_a_real_mixture():
    pure = _units(pft_cover=_cover(TREE_TEMP_DECID=0.7, TREE_TEMP_CONIFER=0.2))
    assert _name(pure) == "forest_temperate_deciduous"
    mixed = _units(pft_cover=_cover(TREE_TEMP_DECID=0.45, TREE_TEMP_CONIFER=0.45))
    assert _name(mixed) == "forest_temperate_mixed"


def test_flood_tolerant_trees_need_a_floodplain():
    cov = _cover(TREE_FLOOD_TOLERANT=0.6, TREE_TEMP_DECID=0.3)
    assert _name(_units(pft_cover=cov)) == "forest_temperate_deciduous"
    assert _name(_units(pft_cover=cov, floodplain=0.6)) == "forest_floodplain_temperate"


def test_wetlands_need_long_waterlogging_or_very_poor_drainage():
    c = _units(waterlog=4.5, drain=3, pft_cover=_cover(GRASS_C3=0.9))
    assert BIOMES[INDEX[_name(c)]].family != "wetland"
    assert BIOMES[INDEX[_name(_units(waterlog=4.5, drain=5))]].family == "wetland"
    assert BIOMES[INDEX[_name(_units(waterlog=7.0, drain=3))]].family == "wetland"


@pytest.mark.parametrize(
    "over, expect",
    [
        (dict(mountain=True, tc=14.0, tw=20.0, gdd5=4000.0, cloud=0.2, z=2200.0,
              pft_cover=_cover(TREE_TROP_EVERGREEN=0.8)), "forest_tropical_cloud_montane"),
        (dict(tc=14.0, tw=27.0, gdd5=6000.0, ai=0.02, alpha=0.05, veg_frac=0.02, fog=0.4,
              coastal=True), "desert_hot_fog_coast"),
        (dict(tc=12.0, tw=30.0, gdd5=6000.0, ai=0.1, alpha=0.1, veg_frac=0.02, endorheic=True,
              rank=0.15, slope_deg=0.5, salinity=0.6), "desert_playa_dry_lake"),
        (dict(tc=12.0, tw=30.0, gdd5=6000.0, ai=0.15, alpha=0.1, veg_frac=0.02, badlands=1.0,
              slope_deg=10.0), "desert_badlands_barren"),
        (dict(tc=12.0, tw=30.0, gdd5=6000.0, ai=0.02, alpha=0.05, veg_frac=0.02,
              endorheic=True, rank=0.5, slope_deg=2.0, salinity=1.0), "barren_gypsum_alkali"),
        (dict(parent=99, volcanic_age=0.8, veg_frac=0.2), "barren_volcanic_recent"),
        (dict(waterlog=7.0, organic=0.8, gdd5=1200.0, river=True), "wetland_fen"),
        (dict(waterlog=7.0, river=True, pft_cover=_cover(SEDGE_RUSH=0.9)),
         "wetland_floodplain_seasonal"),
        (dict(tc=9.0, tw=26.0, gdd5=4000.0, dry_months=3.0, summer_dry=0.0,
              pft_cover=_cover(TREE_TEMP_EVERGREEN=0.8)), "forest_subtropical_seasonal"),
    ],
)  # fmt: skip
def test_rare_classes_can_be_reached(over, expect):
    assert _name(_units(**over)) == expect


def test_tidal_fringe_is_a_fraction_of_the_hex():
    code = {n: i for i, n in enumerate(COAST_TYPES)}
    n_hex = 3
    uh = np.array([0, 1, 2, 2])
    w = np.array([1.0, 1.0, 0.6, 0.4])
    biome = np.full(4, INDEX["forest_tropical_everwet"], dtype=np.uint16)
    dry = np.ones(4, dtype=bool)
    slope = np.array([1.0, 1.0, 1.0, 20.0])
    coast = np.array([code["mangrove"], code["tidal_flat"], code["none"]])
    tc = np.array([22.0, 2.0, 22.0])
    tw = np.array([28.0, 17.0, 28.0])
    ai = np.array([1.5, 1.0, 1.5])
    u2, w2, b2, info = tidal_fringe(uh, w, biome, dry, slope, coast, tc, tw, ai, 40.0, n_hex)
    assert info["mangrove"] == 1 and info["salt_marsh"] == 1
    for h in range(n_hex):  # the hex totals are unchanged
        assert w2[u2 == h].sum() == pytest.approx(w[uh == h].sum())
    mang = (u2 == 0) & np.isin(b2, [INDEX["forest_tropical_mangrove"],
                                     INDEX["wetland_mangrove_intertidal"]])  # fmt: skip
    share = w2[mang].sum()
    assert 0.05 < share < 0.2  # a 4 km belt on a 40 km hex, not the whole hex
    marsh = (u2 == 1) & (b2 == INDEX["wetland_saltmarsh_tidal"])
    assert 0.0 < w2[marsh].sum() < 0.1


def test_cold_trees_photosynthesise_in_a_short_summer():
    """In a short cool summer the boreal conifer out-produces dwarf shrubs (was the reverse)."""
    t_m = np.array([-25, -22, -14, -4, 5, 12, 15, 12, 5, -4, -15, -22], float)[:, None]
    f = pft.Features(
        tc=np.array([-25.0]), tw=np.array([15.0]), t_m=t_m, gdd0=np.array([1600.0]),
        gdd5=np.array([900.0]), gdd10=np.array([350.0]), alpha=np.array([0.9]),
        alpha_m=np.full((12, 1), 0.9), dry_months=np.array([0.0]), summer_dry=np.array([0.0]),
        ai=np.array([1.2]), p_year=np.array([500.0]), pet_year=np.array([400.0]),
        par_m=np.full((12, 1), 150.0), soil_depth=np.array([1.0]), waterlog=np.array([0.0]),
        salinity=np.array([0.0]), snow_months=np.array([6.0]), tree_score=np.array([1.0]),
        cloud=np.array([0.0]),
    )  # fmt: skip
    suit = pft.suitability(f)
    prod = pft.productivity(f, suit)
    idx = pft.PFT_INDEX
    assert prod[idx["TREE_BOREAL_EVERGREEN"], 0] > prod[idx["SHRUB_DWARF_COLD"], 0]


def test_advection_fog_on_a_hot_coast():
    from hexworld.topoclimate.redistribution import advection_fog_potential

    coast = advection_fog_potential(np.array([0.8]), np.array([0.9]), np.array([22.0]),
                                    np.array([3.0]))  # fmt: skip
    inland = advection_fog_potential(np.array([0.3]), np.array([0.0]), np.array([22.0]),
                                     np.array([3.0]))  # fmt: skip
    assert coast[0] > 0.5 and inland[0] == 0.0


def test_generated_world_has_varied_biomes(hist_project):
    """A whole world: many classes, no single class covers most of the land."""
    bio = hist_project.biomes
    F = bio.fractions
    area = np.bincount(F["biome"].astype(int), weights=F["fraction"], minlength=len(BIOMES))
    land_ids = [i for i, b in enumerate(BIOMES) if b.family not in ("marine", "freshwater")]
    land = area[land_ids]
    assert (land > 0).sum() >= 12
    assert land.max() / land.sum() < 0.4
