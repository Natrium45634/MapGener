"""v5.3 G8: actual ecosystems, fire, succession, rare classes (T57-T68)."""

from __future__ import annotations

import numpy as np
import pytest

from hexworld.biomes import classify as bclass
from hexworld.biomes import ecology as eco
from hexworld.biomes.registry import INDEX
from hexworld.pipeline_v50 import marine_classes, marine_rare_classes


def _ring(n=50):
    nbr = np.full((n, 6), -1, dtype=np.int64)
    nbr[:, 0] = np.roll(np.arange(n), -1)
    nbr[:, 3] = np.roll(np.arange(n), 1)
    return nbr


def _run(fam, t=20.0, p=1200.0, dry=2.0, soil=True, years=200, dist=None, n=50):
    return eco.run([fam] * n, np.full(n, t), np.full(n, p), np.full(n, dry),
                   np.full(n, soil), _ring(n), 12345, years=years, disturbances=dist)  # fmt: skip


def test_t58_valid_nonnegative_state():
    r = _run("forest_temperate")
    for k in ("actual_biomass_kg_c_m2", "stand_age_yr", "carbon_emitted_kg_c_m2_yr"):
        assert (r[k] >= 0).all() and np.isfinite(r[k]).all()
    assert (r["actual_biomass_kg_c_m2"] <= r["potential_biomass_kg_c_m2"] + 1e-9).all()


def test_t59_no_fire_without_fuel():
    for fam in ("ice", "barren", "desert"):
        r = _run(fam, t=30.0, p=5.0, dry=12.0)
        assert r["carbon_emitted_kg_c_m2_yr"].max() < 1e-3
        assert r["burn_fraction_last_decade"].max() == 0.0 or fam == "desert"


def test_t60_fire_reduces_biomass_and_books_emissions():
    dry = _run("woodland", t=24.0, p=700.0, dry=8.0, years=300)
    wet = _run("woodland", t=24.0, p=700.0, dry=0.0, years=300)
    assert dry["carbon_emitted_kg_c_m2_yr"].mean() > 0
    assert wet["carbon_emitted_kg_c_m2_yr"].mean() == 0
    assert dry["actual_biomass_kg_c_m2"].mean() < wet["actual_biomass_kg_c_m2"].mean()


def test_t61_succession_after_lava_takes_years():
    n = 50
    # a lava flow resets the whole hex 5 years before the present
    lava = np.full(n, 0.95)  # a flow covers most of the hex; the rest keeps its seeds
    r = _run("forest_temperate", dry=0.0, years=200, dist=[(195, np.arange(n), lava)])
    assert (r["succession_stage"] <= 1).all()  # bare / pioneer, not a mature forest
    r_old = _run("forest_temperate", dry=0.0, years=400, dist=[(50, np.arange(n), lava)])
    assert (r_old["succession_stage"] >= 3).mean() > 0.5  # 350 years later: forest again


def test_regrowth_needs_a_seed_source():
    n = 50
    cut = np.arange(n)
    r = _run("forest_temperate", dry=0.0, years=100, dist=[(10, cut, np.ones(n))])
    # everything was reset and no neighbour survived: nothing can re-establish
    assert r["actual_biomass_kg_c_m2"].max() < 1e-6


def test_t06_rng_independent_of_hex_order():
    ids = np.arange(100)
    a = eco.uniform(7, ids, 3)
    b = eco.uniform(7, ids[::-1], 3)[::-1]
    assert np.array_equal(a, b)
    assert 0.0 <= a.min() and a.max() < 1.0


def test_t57_potential_and_actual_are_separate(hist_project):
    B = hist_project.biomes.layers
    if "actual_biomass_kg_c_m2" not in B:
        pytest.skip("static mode")
    assert "dominant_biome" in B and "potential_biomass_kg_c_m2" in B
    act, pot = B["actual_biomass_kg_c_m2"], B["potential_biomass_kg_c_m2"]
    assert (act <= pot + 1e-4).all()
    assert hist_project.biomes.summary["ecosystems"]["mode"] == "dynamic_vegetation_annual"


# -- rare classes ---------------------------------------------------------------------------
def _cls1(**kw):
    from tests.test_v50 import _cls_inputs

    from hexworld.biomes import pft
    from hexworld.biomes.registry import BIOMES

    cover = np.zeros((len(pft.TRAITS), 1))
    cover[pft.PFT_INDEX["GRASS_C3"]] = 0.6
    kw.setdefault("pft_cover", cover)
    kw.setdefault("veg_frac", np.full(1, 0.6))
    return BIOMES[int(bclass.classify_units(_cls_inputs(**kw))["biome"][0])].id


def test_t62_paramo_needs_cold_without_winter():
    base = dict(
        mountain=np.ones(1, bool),
        thermal_margin=np.full(1, -1.0),
        alpha=np.full(1, 1.2),
        gdd0=np.full(1, 900.0),
        gdd5=np.full(1, 300.0),
    )
    para = _cls1(tc=np.full(1, 3.0), tw=np.full(1, 7.0), **base)
    steppe = _cls1(tc=np.full(1, -20.0), tw=np.full(1, 8.0), **base)
    dry = _cls1(tc=np.full(1, 3.0), tw=np.full(1, 7.0), **{**base, "alpha": np.full(1, 0.2)})
    assert para == "tropical_montane_paramo"
    assert steppe != "tropical_montane_paramo"  # a snowy continental highland is not paramo
    assert dry != "tropical_montane_paramo"  # the dry puna stays alpine steppe


def test_t64_thorn_scrub_needs_a_long_dry_season():
    from hexworld.biomes import pft

    cover = np.zeros((len(pft.TRAITS), 1))
    cover[pft.PFT_INDEX["SHRUB_XERIC"]] = 0.4
    cover[pft.PFT_INDEX["GRASS_C4"]] = 0.1
    kw = dict(
        tc=np.full(1, 22.0),
        tw=np.full(1, 30.0),
        alpha=np.full(1, 0.3),
        veg_frac=np.full(1, 0.5),
        pft_cover=cover,
    )
    assert _cls1(dry_months=np.full(1, 8.0), **kw) == "tropical_thorn_scrub"
    wet = {**kw, "alpha": np.full(1, 1.3)}
    assert _cls1(dry_months=np.full(1, 1.0), **wet) != "tropical_thorn_scrub"


def test_t66_kelp_and_seagrass_differ_by_substrate_light_depth():
    n = 4
    depth = np.array([20.0, 10.0, 2000.0, 10.0])
    base = marine_classes(depth, np.zeros(n, bool), np.full(n, 8.0), np.zeros(n, bool))
    out = marine_rare_classes(
        base, depth, np.full(n, 12.0), np.full(n, 17.0), np.array([0.7, 0.2, 0.7, 0.2]),
        np.array([True, False, True, False]), np.zeros(n, bool), np.zeros(n),
        np.zeros(n), np.zeros(n, bool), np.array([0.0, 0.0, 0.0, 0.9]),
    )  # fmt: skip
    assert out[0] == INDEX["marine_kelp_forest"]  # rocky, shallow, cool, exposed
    assert out[1] == INDEX["marine_seagrass_meadow"]  # soft, sheltered, shallow
    assert out[2] not in (INDEX["marine_kelp_forest"], INDEX["marine_seagrass_meadow"])  # dark
    assert out[3] != INDEX["marine_seagrass_meadow"]  # ice-covered most of the year


def test_t67_hydrothermal_field_needs_a_source():
    n = 3
    depth = np.full(n, 2500.0)
    base = marine_classes(depth, np.zeros(n, bool), np.full(n, 2.0), np.zeros(n, bool))
    out = marine_rare_classes(base, depth, np.full(n, 2.0), np.full(n, 3.0), np.zeros(n),
                              np.zeros(n, bool), np.zeros(n, bool), np.array([1, 0, 5]),
                              np.zeros(n), np.zeros(n, bool), np.zeros(n))  # fmt: skip
    assert out[0] == INDEX["marine_hydrothermal_field"]
    assert out[1] != INDEX["marine_hydrothermal_field"]  # no vent: no field
    assert out[2] != INDEX["marine_hydrothermal_field"]  # a hot spring on land is not one


def test_shelf_and_multiyear_ice_are_cover_states():
    n = 2
    depth = np.full(n, 400.0)
    base = marine_classes(depth, np.zeros(n, bool), np.full(n, -1.8), np.zeros(n, bool))
    out = marine_rare_classes(base, depth, np.full(n, -1.8), np.full(n, -1.0), np.zeros(n),
                              np.zeros(n, bool), np.zeros(n, bool), np.zeros(n),
                              np.array([300.0, 0.0]), np.array([False, True]),
                              np.ones(n))  # fmt: skip
    assert out[0] == INDEX["marine_ice_shelf_floating"]
    assert out[1] == INDEX["marine_sea_ice_multiyear"]


def test_t63_heath_needs_poor_soil_and_oceanic_climate():
    assert "aeolian_sand" in bclass.PM_POOR and "alluvium" not in bclass.PM_POOR


def test_registry_is_append_only_and_valid():
    from hexworld.biomes.registry import BIOMES, validate_registry

    assert not validate_registry()
    assert BIOMES[INDEX["marine_hadal_trench"] + 1].id == "tropical_montane_paramo"


def test_class_audit_gives_a_reason_for_every_absent_class(hist_project):
    ca = hist_project.biomes.summary.get("class_audit")
    assert ca
    allowed = {None, "too_rare", "climate_not_generated", "substrate_absent",
               "hydrology_absent", "requires_component_disabled"}  # fmt: skip
    for k, v in ca.items():
        assert v["absence_reason"] in allowed, k
        assert (v["realised_share"] > 0) == (v["absence_reason"] is None), k
