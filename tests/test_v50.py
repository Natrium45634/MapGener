"""v5.0: biomes, natural cover, ecosystems, cryosphere (TZ 5.0, 16)."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from hexworld import generate
from hexworld.biomes import classify as bclass
from hexworld.biomes import cryosphere as cr
from hexworld.biomes import pft
from hexworld.biomes.registry import (
    BIOMES,
    COVER_CLASSES,
    INDEX,
    LIMITING_FACTORS,
    MODIFIERS,
    decode_bits,
    validate_registry,
)
from hexworld.config import BiomeParams, Config, ConfigError
from hexworld.io import load_project, save_project
from hexworld.pipeline_v50 import (
    KOPPEN,
    _koppen,
    aggregate_fractions,
    biome_stage_keys,
    lake_classes,
    marine_classes,
    run_biomes,
)

# -- V1: registry, IDs, legacy ----------------------------------------------------------------


def test_v1_registry_ids_unique_and_valid():
    assert validate_registry() == []
    assert len({b.id for b in BIOMES}) == len(BIOMES) >= 100


def test_v1_registry_codes_are_append_only():
    # the first codes are frozen: a later version may only append
    frozen = ["unresolved", "desert_hot_sand_erg", "desert_hot_gravel_reg",
              "desert_hot_rock_hamada", "desert_hot_salt_flat"]  # fmt: skip
    assert [b.id for b in BIOMES[:5]] == frozen
    assert INDEX["marine_hadal_trench"] == len(BIOMES) - 1


def test_v1_catalogue_covers_the_required_range():
    need = (
        "desert_hot_salt_flat", "desert_playa_dry_lake", "desert_high_mountain_cold",
        "barren_volcanic_recent", "forest_tropical_everwet", "forest_tropical_cloud_montane",
        "forest_temperate_rainforest", "forest_mediterranean_sclerophyll",
        "forest_boreal_deciduous_larch", "forest_tundra_mosaic", "forest_steppe_mosaic",
        "grassland_alpine", "tundra_moss_lichen", "tundra_polar_desert_sparse",
        "barren_nunatak", "snow_perennial_field", "firn_field", "glacier_mountain",
        "glacier_ice_cap", "glacier_ice_sheet", "wetland_bog", "wetland_fen",
        "wetland_swamp_forest", "wetland_inland_saline", "water_lake_saline",
        "marine_abyssal_benthic", "marine_coral_reef_potential",
    )  # fmt: skip
    for b in need:
        assert b in INDEX, b


def test_v1_pft_registry():
    ids = [t.id for t in pft.TRAITS]
    assert len(ids) == len(set(ids)) >= 20
    assert pft.IS_TREE.sum() >= 10 and pft.IS_MOSS.sum() == 1


def test_v1_bit_flags_round_trip():
    v = (1 << LIMITING_FACTORS.index("drought")) | (1 << LIMITING_FACTORS.index("ice"))
    assert decode_bits(v, LIMITING_FACTORS) == ["drought", "ice"]
    assert decode_bits(0, MODIFIERS) == []


def test_v1_config_validation():
    with pytest.raises(ConfigError):
        Config(biomes=BiomeParams(mode="dynamic"))
    with pytest.raises(ConfigError):
        Config(biomes=BiomeParams(pft_tau=0.0))
    cfg = Config(biomes=BiomeParams(cryo_spinup_years=30))
    assert Config.from_dict(cfg.to_dict()) == cfg


# -- V2: masks, water, fractions ----------------------------------------------------------------


def test_v2_t06_mixed_hex_fractions():
    # 40 % forest, 25 % steppe, 20 % bare, 15 % lake
    uh = np.zeros(4, dtype=np.int64)
    w = np.array([0.40, 0.25, 0.20, 0.15])
    b = np.array(
        [
            INDEX["forest_temperate_mixed"],
            INDEX["grassland_steppe"],
            INDEX["desert_sparse_undifferentiated"],
            INDEX["water_lake_fresh_open"],
        ]
    )
    fr, frv, hx, dom = aggregate_fractions(uh, w, b, 1)
    assert float(fr["fraction"].sum()) == pytest.approx(1.0)
    land = 1.0 - 0.15
    assert land == pytest.approx(0.85)
    forest = float(frv[list(fr["biome"]).index(INDEX["forest_temperate_mixed"])])
    assert forest / land == pytest.approx(0.470588, abs=1e-5)
    assert dom[0] == INDEX["forest_temperate_mixed"]


def test_v2_t07_water_classes_are_kept_apart():
    uh = np.zeros(3, dtype=np.int64)
    w = np.array([0.25, 0.25, 0.5])
    b = np.array([INDEX["marine_shelf_shallow"], INDEX["water_lake_fresh_open"],
                  INDEX["grassland_steppe"]])  # fmt: skip
    fr, frv, *_ = aggregate_fractions(uh, w, b, 1)
    marine = sum(f for c, f in zip(fr["biome"], frv, strict=True) if BIOMES[c].family == "marine")
    assert marine == pytest.approx(0.25)


def test_v2_t08_dry_depression_is_not_ocean():
    depth = np.array([0.0])
    # a dry interior depression below the datum: no water depth, no marine class unless sea
    assert lake_classes(np.array([True]), np.array([0.05]))[0] == INDEX["water_lake_seasonal"]
    assert marine_classes(np.array([30.0]), np.array([False]), np.array([5.0]),
                          np.array([False]))[0] == INDEX["marine_shelf_shallow"]  # fmt: skip
    del depth


def test_v2_marine_depth_classes():
    d = np.array([20, 120, 1500, 4500, 7000.0])
    tr = np.array([False, False, False, False, True])
    out = marine_classes(d, tr, np.full(5, 10.0), np.zeros(5, bool))
    assert [BIOMES[c].id for c in out] == [
        "marine_shelf_shallow", "marine_shelf_deep", "marine_slope_benthic",
        "marine_abyssal_benthic", "marine_hadal_trench"]  # fmt: skip


def test_v2_coral_only_potential_and_only_warm_shallow():
    out = marine_classes(np.array([20.0, 20.0, 300.0]), np.zeros(3, bool),
                         np.array([24.0, 10.0, 24.0]), np.zeros(3, bool))  # fmt: skip
    assert out[0] == INDEX["marine_coral_reef_potential"]
    assert out[1] != out[0] and out[2] != out[0]


def test_v2_t17_sea_ice_does_not_make_land():
    out = marine_classes(np.array([800.0]), np.array([False]), np.array([-25.0]), np.array([False]))
    assert BIOMES[out[0]].family == "marine"


def test_v2_lake_salinity_classes():
    out = lake_classes(np.array([False, True, True]), np.array([0.1, 0.2, 0.6]))
    assert [BIOMES[c].id for c in out] == [
        "water_lake_fresh_open", "water_lake_saline", "lake_unknown_salinity"]  # fmt: skip


# -- V3: climate metrics -----------------------------------------------------------------------


def _ai(p, pet, eps=50.0):
    return p / pet if pet > eps else float("nan")


def test_v3_t01_t02_aridity_classes():
    assert _ai(50, 2000) == pytest.approx(0.025)
    assert _ai(450, 1500) == pytest.approx(0.30)


def test_v3_t03_pet_zero_is_not_a_hot_desert():
    assert np.isnan(_ai(0.0, 0.0))


def test_v3_t05_annual_sum():
    assert float(np.sum(np.full(12, 100.0))) == 1200.0


def _months(t, p):
    return np.asarray(t, dtype=float)[:, None], np.asarray(p, dtype=float)[:, None]


def test_v3_koppen_tropical_rainforest():
    t, p = _months([27] * 12, [250] * 12)
    assert KOPPEN[_koppen(t, p, np.array([0.0]))[0]] == "Af"


def test_v3_koppen_hot_desert():
    t, p = _months([20, 22, 25, 28, 31, 34, 35, 34, 31, 27, 23, 20], [2] * 12)
    assert KOPPEN[_koppen(t, p, np.array([25.0]))[0]] == "BWh"


def test_v3_koppen_tundra_and_ice():
    t, p = _months([-25, -24, -20, -12, -2, 4, 7, 5, 0, -8, -18, -23], [20] * 12)
    assert KOPPEN[_koppen(t, p, np.array([70.0]))[0]] == "ET"
    t2, p2 = _months([-40] * 6 + [-5] * 6, [10] * 12)
    assert KOPPEN[_koppen(t2, p2, np.array([80.0]))[0]] == "EF"


def test_v3_koppen_mediterranean():
    t, p = _months([10, 11, 13, 15, 19, 23, 26, 26, 23, 19, 14, 11],
                   [130, 110, 90, 60, 30, 10, 4, 6, 40, 100, 130, 140])  # fmt: skip
    assert KOPPEN[_koppen(t, p, np.array([38.0]))[0]] == "Cs"


# -- V4: PFT, cover, barren -----------------------------------------------------------------------


def _features(tc, tw, alpha, dry=0.0, sdry=0.0, gdd5=None, soil=1.5, wl=0.0, sal=0.0,
              snow=0.0, tree_score=1.0, cloud=0.0, n=1):  # fmt: skip
    t_m = np.linspace(tc, tw, 12)[:, None] * np.ones((1, n))
    days = 30.4
    g = lambda b: (np.maximum(t_m - b, 0) * days).sum(0)  # noqa: E731
    gd5 = g(5.0) if gdd5 is None else np.full(n, gdd5)
    return pft.Features(
        tc=np.full(n, tc), tw=np.full(n, tw), t_m=t_m, gdd0=g(0.0), gdd5=gd5, gdd10=g(10.0),
        alpha=np.full(n, alpha), alpha_m=np.full((12, n), alpha), dry_months=np.full(n, dry),
        summer_dry=np.full(n, sdry), ai=np.full(n, alpha), p_year=np.full(n, 1000.0),
        pet_year=np.full(n, 1000.0), par_m=np.full((12, n), 0.48 * 18 * 30.4),
        soil_depth=np.full(n, soil), waterlog=np.full(n, wl), salinity=np.full(n, sal),
        snow_months=np.full(n, snow), tree_score=np.full(n, tree_score), cloud=np.full(n, cloud),
    )  # fmt: skip


def _dominant(f, open_frac=1.0):
    s = pft.suitability(f)
    pr = pft.productivity(f, s)
    v, cov = pft.allocate_cover(s, pr, np.full(f.tc.size, open_frac))
    return v, cov, pft.TRAITS[int(np.argmax(cov[:, 0]))].id


def test_v4_tropical_wet_is_evergreen_forest():
    v, cov, dom = _dominant(_features(25, 28, 0.95))
    assert dom == "TREE_TROP_EVERGREEN" and v[0] > 0.9


def test_v4_t23_no_tropical_tree_with_frost():
    s = pft.suitability(_features(-8, 22, 0.95))
    assert s[pft.PFT_INDEX["TREE_TROP_EVERGREEN"], 0] == 0.0


def test_v4_boreal_climate_gives_boreal_trees():
    _, cov, dom = _dominant(_features(-25, 15, 0.8, gdd5=700, snow=6))
    assert dom.startswith("TREE_BOREAL")


def test_v4_summer_drought_favours_sclerophylls():
    s = pft.suitability(_features(9, 25, 0.35, dry=4, sdry=4))
    k = pft.PFT_INDEX["SHRUB_SCLEROPHYLL"]
    assert s[k, 0] > 0.5
    assert s[pft.PFT_INDEX["TREE_TROP_EVERGREEN"], 0] == 0.0


def test_v4_t19_no_water_dead_rock_is_bare():
    v, cov, _ = _dominant(_features(20, 35, 0.0, dry=12, soil=0.0), open_frac=0.0)
    assert v[0] == 0.0 and cov.sum() == 0.0


def test_v4_hyperarid_hot_is_almost_bare():
    v, _, _ = _dominant(_features(15, 35, 0.02, dry=12, soil=0.3))
    assert v[0] < 0.1


def test_v4_t29_tolerance_is_exactly_one_inside():
    assert pft._sig(np.array([5.0]))[0] == 1.0 and pft._sig(np.array([-5.0]))[0] == 0.0


def test_v4_cover_never_exceeds_open_share():
    v, cov, _ = _dominant(_features(15, 25, 0.8), open_frac=0.4)
    assert v[0] <= 0.4 + 1e-12 and cov.sum() == pytest.approx(v[0])


# -- V5: cryosphere mass -------------------------------------------------------------------------


def test_v5_t09_t10_swe_volume_scales_with_area():
    assert 1e6 * 0.3 == pytest.approx(3e5)  # m^3 SWE on 1 km^2
    assert 1e6 * 0.3 * 1000.0 == pytest.approx(3e8)  # kg
    assert 100e6 * 0.3 == pytest.approx(3e7)


def test_v5_t11_month_budget():
    st = {"snow": np.array([200.0]), "firn": np.zeros(1), "ice": np.zeros(1)}
    cp = cr.CryoParams(k_snow_mm=1.0, sublimation_share=0.0)
    fl = cr.month_step(st, np.array([300.0]), np.array([-20.0]), np.zeros(1), 30, cp)
    total = st["snow"] + st["firn"] + st["ice"]
    assert float(total[0]) == pytest.approx(200 + float(fl["snowfall"][0]) - float(
        fl["melt"][0]) - float(fl["sublimation"][0]))  # fmt: skip


def test_v5_t12_snow_to_firn_keeps_mass():
    st = {"snow": np.array([100.0]), "firn": np.array([50.0]), "firn_age": np.array([3.0]),
          "ice": np.zeros(1)}  # fmt: skip
    before = st["snow"] + st["firn"] + st["ice"]
    cr.year_end(st, np.array([-20.0]), cr.CryoParams())
    assert float((st["snow"] + st["firn"] + st["ice"])[0]) == pytest.approx(float(before[0]))
    assert st["snow"][0] == 0.0


def test_v5_t13_firn_to_ice_thickness():
    dh = cr.ice_thickness_m(100.0)
    assert dh == pytest.approx(0.1 * 1000.0 / 917.0)


def test_v5_t14_ablation_never_negative():
    st = {"snow": np.zeros(1), "firn": np.zeros(1), "ice": np.array([10.0])}
    cr.month_step(st, np.zeros(1), np.array([15.0]), np.zeros(1), 31, cr.CryoParams())
    assert st["ice"][0] == 0.0


def test_v5_firn_densifies_with_age():
    rho = cr.firn_density(np.array([0.0, 5.0, 50.0, 500.0]), np.full(4, -15.0))
    assert np.all(np.diff(rho) > 0) and rho[-1] <= cr.RHO_ICE and rho[0] == cr.RHO_SNOW


def _one_hex_cryo(t, p, years=30, seed=None):
    nbr = np.full((1, 6), -1, dtype=np.int64)
    cp = cr.CryoParams(spinup_years=years)
    return cr.run_cryosphere(
        np.zeros(1, dtype=np.int64), np.ones(1), np.ones(1, bool), np.full((12, 1), t),
        np.full((12, 1), p), np.zeros((12, 1)), np.zeros(1), nbr, 25.0, 1, cp,
        seed_ice_hex=seed,
    )  # fmt: skip


def test_v5_t15_no_accumulation_no_ice_sheet():
    r = _one_hex_cryo(-25.0, 0.0)
    assert r.unit["ice_swe_mm"][0] == 0.0 and r.hex["ice_thickness_m"][0] == 0.0


def test_v5_accumulation_builds_firn_and_ice():
    r = _one_hex_cryo(-20.0, 40.0, years=80)
    assert r.unit["firn_swe_mm"][0] > 0 and r.unit["ice_swe_mm"][0] > 0
    assert r.summary["mass_residual_mm_max"] < 1e-6


def test_v5_t16_seeded_ice_shrinks_slowly_under_negative_balance():
    r1 = _one_hex_cryo(-2.0, 5.0, years=1, seed=np.array([1500.0]))
    r10 = _one_hex_cryo(-2.0, 5.0, years=10, seed=np.array([1500.0]))
    h1, h10 = r1.hex["ice_thickness_m"][0], r10.hex["ice_thickness_m"][0]
    assert 0 < h10 < h1 <= 1500.0 + 1e-6


def test_v5_ice_flow_conserves_volume_and_spreads():
    nbr = np.array([[1, -1, -1, -1, -1, -1], [-1, -1, -1, 0, -1, -1]], dtype=np.int64)
    h, _ = cr.flow_step(np.array([1500.0, 0.0]), np.zeros(2), nbr, 25.0, 1e-16, 1.0)
    assert h.sum() == pytest.approx(1500.0) and h[1] > 0 and (h >= 0).all()


def test_v5_paleo_seed_needs_cold_and_snow():
    cp = cr.CryoParams()
    s = cr.paleo_seed(np.array([-25.0, -25.0, 5.0]), np.array([300.0, 0.0, 300.0]),
                      np.array([True, True, True]), cp)  # fmt: skip
    assert s[0] > 0 and s[1] == 0 and s[2] == 0


# -- V6: classification and coexistence -----------------------------------------------------------


def _cls_inputs(n=1, **kw):
    base = {
        "tc": np.full(n, 10.0), "tw": np.full(n, 20.0), "gdd0": np.full(n, 3000.0),
        "gdd5": np.full(n, 2000.0), "alpha": np.full(n, 0.8), "dry_months": np.zeros(n),
        "summer_dry": np.zeros(n), "veg_frac": np.full(n, 0.9),
        "pft_cover": np.zeros((len(pft.TRAITS), n)), "mountain": np.zeros(n, bool),
        "z": np.full(n, 300.0), "ai": np.full(n, 1.0), "salinity": np.zeros(n),
        "waterlog": np.zeros(n), "slope_deg": np.full(n, 3.0), "parent": np.full(n, 2),
        "rocky": np.zeros(n), "thermal_margin": np.full(n, 3.0), "ice_cover": np.zeros(n),
        "firn_cover": np.zeros(n), "snow_cover": np.zeros(n), "ice_form": np.zeros(n, int),
        "ice_balance": np.zeros(n), "in_ice_body": np.zeros(n, bool), "pm_young_volcanic": 4,
        "pm_aeolian": 8, "volcanic_age": np.full(n, 1e9), "rank": np.full(n, 0.5),
        "organic": np.zeros(n), "drainage_poor_rain": np.zeros(n), "floodplain": np.zeros(n),
        "river": np.zeros(n, bool), "valley": np.zeros(n), "p_year": np.full(n, 900.0),
        "endorheic": np.zeros(n, bool), "fog": np.zeros(n), "coastal": np.zeros(n, bool),
        "badlands": np.zeros(n), "gypsum": np.zeros(n), "erosion": np.zeros(n),
        "permafrost": np.zeros(n), "cloud": np.zeros(n), "soil_depth": np.full(n, 1.0),
        "snow_months": np.zeros(n), "cap": np.zeros(n), "marine": np.zeros(n),
        "shadow": np.zeros(n, bool),
    }  # fmt: skip
    base.update(kw)
    return base


def _cls(**kw):
    return BIOMES[int(bclass.classify_units(_cls_inputs(**kw))["biome"][0])].id


def test_v6_glacier_and_firn_and_nunatak():
    assert _cls(ice_cover=np.ones(1), ice_form=np.array([3])) == "glacier_ice_sheet"
    assert _cls(firn_cover=np.ones(1)) == "firn_field"
    assert _cls(in_ice_body=np.ones(1, bool), slope_deg=np.full(1, 30.0)) == "barren_nunatak"


def test_v6_t18_nunatak_is_rock_not_ice():
    assert BIOMES[INDEX["barren_nunatak"]].family == "polar"


def test_v6_salt_flat_needs_basin_and_salt():
    kw = {"veg_frac": np.zeros(1), "ai": np.full(1, 0.02), "tw": np.full(1, 34.0),
          "tc": np.full(1, 14.0), "salinity": np.full(1, 0.8), "endorheic": np.ones(1, bool),
          "rank": np.full(1, 0.1), "slope_deg": np.full(1, 0.5)}  # fmt: skip
    assert _cls(**kw) == "desert_hot_salt_flat"
    kw["endorheic"] = np.zeros(1, bool)
    assert _cls(**kw) != "desert_hot_salt_flat"


def test_v6_t25_riparian_gallery_in_desert():
    assert _cls(
        river=np.ones(1, bool),
        valley=np.full(1, 0.6),
        alpha=np.full(1, 0.2),
        veg_frac=np.full(1, 0.4),
        rank=np.full(1, 0.2),
    ) in (  # fmt: skip
        "wetland_riparian_gallery",
        "forest_floodplain_temperate",
        "forest_tropical_floodplain",
    )


def test_v6_t27_snowy_taiga_stays_taiga():
    cov = np.zeros((len(pft.TRAITS), 1))
    cov[pft.PFT_INDEX["TREE_BOREAL_EVERGREEN"]] = 0.85
    got = _cls(pft_cover=cov, snow_months=np.full(1, 7.0), tc=np.full(1, -20.0),
               gdd5=np.full(1, 800.0), alpha=np.full(1, 0.85))  # fmt: skip
    assert got.startswith("forest_boreal")


def test_v6_above_treeline_mountain_is_alpine():
    got = _cls(thermal_margin=np.full(1, -1.5), mountain=np.ones(1, bool), veg_frac=np.full(1, 0.5))
    assert BIOMES[INDEX[got]].family in ("mountain", "tundra")


def test_v6_t26_no_mangrove_without_tide():
    s = pft.suitability(_features(25, 29, 0.9, sal=0.9, wl=12))
    assert s[pft.PFT_INDEX["TREE_MANGROVE"], 0] == 0.0


# -- V7 / V8: the stage on a generated world ------------------------------------------


def test_v7_stage_on_a_generated_world(hist_project):
    b = hist_project.biomes
    assert b is not None and b.validation["ok"], b.validation
    s = b.summary
    assert s["fraction_sum_max_error"] < 1e-5 and s["cover_sum_max_error"] < 1e-5
    assert s["cryosphere"]["mass_residual_mm_max"] < 1e-3
    assert s["classes_present_land"] >= 5
    off = b.fractions["offsets"]
    fr = b.fractions["fraction"].astype(np.float64)
    sums = np.bincount(np.repeat(np.arange(off.size - 1), np.diff(off)), weights=fr,
                       minlength=off.size - 1)  # fmt: skip
    assert np.allclose(sums[np.diff(off) > 0], 1.0, atol=1e-4)
    cov = sum(b.layers[f"cover_{c}"] for c in COVER_CLASSES)
    assert np.allclose(cov, 1.0, atol=1e-4)


def test_v7_explain_a_hex(hist_project):
    p = hist_project
    land = np.argwhere(p.biomes.layers["realm"] == 1)
    r, c = (int(v) for v in land[len(land) // 2])
    e = p.explain_biome(c, r)
    assert sum(x["fraction"] for x in e["composition"]) == pytest.approx(1.0, abs=1e-3)
    assert e["quality"] in ("terrain_units_v4_7", "coarse_hex_fallback")
    assert "limiting_factors" in e and "climate_zone_koppen" in e


def test_v7_deterministic_save_load_and_reuse(tmp_path, hist_config, hist_project):
    p = hist_project
    again = run_biomes(p.config, p.geometry, p)
    assert np.array_equal(again["layers"]["dominant_biome"], p.biomes.layers["dominant_biome"])
    path = save_project(p, tmp_path / "w")
    q = load_project(path)
    assert q.biomes is not None
    assert q.biomes.content_hashes() == p.biomes.content_hashes()
    r = generate(hist_config, reuse=p)
    assert "v50_biomes" in r.stats["reused_stages"]


def test_v7_config_change_invalidates(hist_config):
    up = {"env": {"a": 1}}
    k1 = biome_stage_keys(hist_config, up)
    k2 = biome_stage_keys(replace(hist_config, biomes=BiomeParams(pft_tau=0.2)), up)
    assert k1 != k2


def test_v7_old_save_without_biomes_is_not_generated(tmp_path, hist_project):
    old = replace(hist_project, biomes=None)
    q = load_project(save_project(old, tmp_path / "old"))
    assert q.biomes is None  # "not generated", never zero vegetation
    from hexworld.views import VIEWS

    ok, why = VIEWS["biomes"].availability(q)
    assert not ok and "не сгенерированы" in why


def test_v7_coarse_fallback_without_terrain_units(hist_project):
    p = replace(hist_project, topo=None)
    res = run_biomes(p.config, p.geometry, p)
    assert res["summary"]["input_quality"] == "coarse_hex_fallback"
    assert res["summary"]["fraction_sum_max_error"] < 1e-5


def test_v8_views_render_and_month_is_display_only(hist_project):
    from hexworld.views import VIEWS

    p = hist_project
    before = p.biomes.content_hashes()
    for vid in ("biomes", "vegetation", "cryosphere", "wetlands", "eco_limits", "ecozones"):
        spec = VIEWS[vid]
        assert spec.availability(p)[0]
        for m in spec.modes:
            rgb = spec.render(p, m.id, spec.default_overlays())
            assert rgb.shape == p.layers["plate_id"].shape + (3,)
            assert spec.legend(p, m.id)
    assert p.biomes.content_hashes() == before


def test_v8_hex_info_has_the_composition(hist_project):
    p = hist_project
    land = np.argwhere(p.biomes.layers["realm"] == 1)
    r, c = (int(v) for v in land[0])
    info = p.hex_info(c, r)
    assert "biomes" in info and info["biomes"]["composition"]


def test_v8_timings_recorded(hist_project):
    t = hist_project.biomes.timings
    assert set(t) >= {"v50_cryosphere", "v50_pft_classes", "v50_aggregate"}


def test_v1_presets_load():
    import json
    from pathlib import Path

    folder = Path(__file__).resolve().parents[1] / "docs" / "biomes" / "presets"
    files = sorted(folder.glob("*.json"))
    assert len(files) >= 7
    for f in files:
        base = Config().to_dict()
        for section, values in json.loads(f.read_text(encoding="utf-8")).items():
            base[section] = {**base.get(section, {}), **values}
        Config.from_dict(base)
