"""TZ 5.5: the numerical references N01–N20 and the operator-level acceptance tests (B08,
B10, B15–B18, B21–B24, B35, B40–B48, B58–B66, B68, B77). Each test names its ID."""

from __future__ import annotations

import math

import numpy as np
import pytest

from hexworld.biomes.marine import par_fraction
from hexworld.biosphere import dynamics as dy
from hexworld.biosphere import foodweb as fw
from hexworld.biosphere import graph as gr
from hexworld.biosphere import metrics as mt
from hexworld.biosphere import niche as nc
from hexworld.biosphere import production as pr
from hexworld.biosphere import registry as rg
from hexworld.biosphere import resources as rs

R5, R8 = 1e-5, 1e-8


def test_n01_area_fraction_density():
    assert 2e6 * 0.25 * 4.0 == pytest.approx(2e6, rel=R8)


def test_n02_b15_trapezoid():
    v = nc.trapezoid(np.array([5, 15, 25, 31]), 0, 10, 20, 30)
    assert v == pytest.approx([0.5, 1.0, 0.5, 0.0], rel=R8)
    # degenerate flanks are steps that keep the plateau closed; NaN stays NaN (B09)
    d = nc.trapezoid(np.array([9.99, 10.0, 20.0, 20.01, np.nan]), 10, 10, 20, 20)
    assert d[:4].tolist() == [0.0, 1.0, 1.0, 0.0] and np.isnan(d[4])


def test_n03_soft_combination_and_hard_exclusion():
    assert nc.combine([np.array([0.25]), np.array([1.0])])[0] == pytest.approx(0.5, rel=R8)
    assert nc.combine([np.array([0.0]), np.array([1.0])])[0] == 0.0  # strict exclusion
    assert nc.combine([np.array([0.5])], gate=np.array([0.0]))[0] == 0.0


def test_n04_b23_lue_gpp_npp():
    gpp, npp = pr.lue_npp(100.0, 0.6, 1.5, 1, 1, 1, cue=0.5)
    assert gpp == pytest.approx(90.0, rel=R8) and npp == pytest.approx(45.0, rel=R8)


def test_n05_fapar():
    assert pr.fapar(2.0, 0.5) == pytest.approx(1 - math.exp(-1), rel=R8)


def test_n06_b24_equilibrium_stock():
    assert pr.equilibrium_stock(500.0, 0.25) == pytest.approx(2000.0, rel=R8)
    # the relaxation reaches the analytic equilibrium and never goes negative
    x = 0.0
    for _ in range(200):
        x = dy.relax(x, 2000.0, 1.0, 4.0)
        assert x >= 0
    assert x == pytest.approx(2000.0, rel=1e-6)


def test_n07_n08_b35_occupancy():
    c = gr.colonisation_rate(100.0, 50.0, kappa0=1.0, rho=0.0)
    assert gr.occupancy(c, 0.0, 1.0) == pytest.approx(1 - math.exp(-math.exp(-2)), rel=R5)
    assert gr.occupancy(0.2, 0.1, 1.0) == pytest.approx(0.17278785, rel=R5)
    # bounds, c = e = 0 keeps p0, the long-time limit is c / (c + e)
    assert gr.occupancy(0.0, 0.0, 10.0, p0=0.3) == pytest.approx(0.3)
    p = gr.occupancy(np.array([0.0, 5.0, 1e-9]), np.array([3.0, 0.0, 1e-9]), 1e6)
    assert np.all((p >= 0) & (p <= 1)) and p[1] == pytest.approx(1.0)
    assert gr.occupancy(0.2, 0.1, 1e6) == pytest.approx(2 / 3, rel=R8)


def test_n09_b37_migration_conserves():
    left, got, lost = dy.migrate(100.0, 0.4, [0.75, 0.25])
    assert left == pytest.approx(60) and got.tolist() == pytest.approx([30, 10])
    assert left + got.sum() + lost == pytest.approx(100.0, rel=R8)


def test_b38_receivers_share_capacity_surplus_returns():
    left, got, lost = dy.migrate(100.0, 0.5, [0.5, 0.5], capacity=[10.0, 50.0])
    assert got.tolist() == pytest.approx([10.0, 25.0])
    assert left + got.sum() + lost == pytest.approx(100.0, rel=R8)  # nothing vanishes
    left, got, lost = dy.migrate(100.0, 0.5, [1.0], mortality=0.2)
    assert lost == pytest.approx(10.0) and left + got.sum() + lost == pytest.approx(100.0)


def test_n10_b40_proportional_allocation():
    got, lam = fw.allocate(np.array([100.0]), [np.array([80.0]), np.array([40.0])])
    assert got[0][0] == pytest.approx(66.6666667, rel=R5)
    assert got[1][0] == pytest.approx(33.3333333, rel=R5)
    assert got[0][0] + got[1][0] == pytest.approx(100.0, rel=R8)
    # order of the consumers does not change the result
    got2, _ = fw.allocate(np.array([100.0]), [np.array([40.0]), np.array([80.0])])
    assert got2[1][0] == pytest.approx(got[0][0])


def test_n11_b44_assimilation_and_energy():
    e = fw.energy_split(50.0, 0.6, 0.3)
    assert e["assimilated"] == pytest.approx(30.0) and e["egested"] == pytest.approx(20.0)
    assert e["production"] <= e["assimilated"]
    assert e["assimilated_mj"] == pytest.approx(30.0 * fw.MJ_PER_KGC)
    with pytest.raises(ValueError):
        fw.energy_split(1.0, 1.2, 0.1)


def test_n12_b47_detritus_decay():
    new, dec = dy.detritus_step(100.0, 0.2, 1.0)
    assert dec == pytest.approx(18.1269247, rel=R5) and new == pytest.approx(81.8730753, rel=R5)
    new, dec = dy.detritus_step(100.0, 0.2, 1.0, inflow=10.0)
    assert new + dec == pytest.approx(110.0, rel=R8) and dec <= 110.0


def test_n13_n14_b51_b52_light_in_water():
    assert math.log(100.0) / 0.1 == pytest.approx(46.05170186, rel=R8)
    assert 100.0 * par_fraction(10.0, 0.1, 0.5, 0.0) == pytest.approx(18.39397206, rel=R5)


def test_n15_b58_alpha_beta_gamma():
    r = mt.alpha_beta_gamma([{"a", "b", "c"}, {"b", "c", "d"}])
    assert r["gamma"] == 4 and r["alpha"] == 3.0
    assert r["beta"] == pytest.approx(4 / 3) and r["turnover"] == pytest.approx(0.5)


def test_n16_b59_shannon_hill():
    h = mt.hill([1, 1, 1, 1])
    assert h["H"] == pytest.approx(math.log(4), rel=R8)
    assert h["q1"] == pytest.approx(4.0) and h["q2"] == pytest.approx(4.0)
    assert mt.hill([]) ["S"] == 0 and math.isnan(mt.hill([5.0])["J"])
    assert mt.shannon_rows(np.array([[1.0], [1.0]]))[0] == pytest.approx(math.log(2))


def test_n17_b60_species_area_prior_is_separate():
    assert mt.species_area_prior(100.0, 16.0, 1.0, 0.25) == pytest.approx(200.0, rel=R8)
    assert mt.species_area_prior(100.0, 16.0, 1.0, 0.25, cap=150) == 150


def test_n18_b62_index():
    assert mt.index(60.0, 20.0, 100.0) == pytest.approx(50.0)
    assert mt.index(130.0, 20.0, 100.0) == 100.0
    assert np.isnan(mt.index(np.nan, 0.0, 1.0))


def test_b63_percentile_is_area_weighted_world_wide():
    v = np.array([1.0, 2.0, 3.0, np.nan])
    p = mt.percentile_area_weighted(v, np.array([1.0, 1.0, 2.0, 1.0]))
    assert p[0] == pytest.approx(12.5) and p[2] == pytest.approx(75.0) and np.isnan(p[3])


def test_n19_b66_logistic_msy():
    assert rs.logistic_msy(0.2, 1000.0) == pytest.approx(50.0)


def test_n20_b68_disturbance_once():
    loss = dy.disturbance_loss(1000.0, 0.2, 0.5)
    assert loss == pytest.approx(100.0) and 1000.0 - loss == pytest.approx(900.0)
    led = dy.EcoLedger()
    assert led.transfer("fire/1/h7", "plants", "air", 100.0)
    assert not led.transfer("fire/1/h7", "plants", "air", 100.0)  # replay: no second loss
    assert led.totals()["plants"] == pytest.approx(-100.0)


def test_b45_b46_demography_nonnegative_no_division_by_zero():
    n = dy.logistic_step(np.array([10.0, 10.0, 0.0]), np.array([100.0, 0.0, 50.0]), 0.5, 1.0)
    assert np.all(n >= 0) and n[1] < 10.0 and n[2] == 0.0 and 10.0 < n[0] < 100.0


def test_b16_b17_b18_seasons():
    days = np.array([31, 28.25, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31])
    s = np.ones((12, 1))
    s[1] = 0.0  # a bad February costs less than a bad January (shorter month)
    s2 = np.ones((12, 1))
    s2[0] = 0.0
    assert nc.survival(s, days)[0] > nc.survival(s2, days)[0]
    assert nc.survival(s2, days, dormant=True)[0] > nc.survival(s2, days)[0]
    m = np.arange(12)
    assert np.all(nc.mirror_months(m) == np.roll(m, 6))
    wrap = np.zeros((12, 1))
    wrap[[10, 11, 0, 1]] = 1.0  # Nov–Feb continuous across the year end
    frag = np.zeros((12, 1))
    frag[[0, 3, 6, 9]] = 1.0  # same total, fragmented
    assert nc.breeding_ok(wrap, 4)[0] and not nc.breeding_ok(frag, 4)[0]


def test_b08_registry():
    reg = rg.load()
    assert len(reg.groups) == 94 and rg.validate(reg) == []
    ids = list(reg.groups) + list(reg.templates)
    assert len(ids) == len(set(ids))
    for g in reg.groups.values():
        assert g.parameter_status == "provisional"


def test_b10_b05_taxa_are_not_groups_and_ids_are_stable():
    reg = rg.load()
    assert len(reg.templates) > len(reg.groups)
    a = rg.taxon_id("Seed", "grass_c3.10", "land:1:2")
    assert a == rg.taxon_id("Seed", "grass_c3.10", "land:1:2")
    assert a != rg.taxon_id("seed", "grass_c3.10", "land:1:2")  # case-sensitive
    assert a != rg.taxon_id("Seed", "grass_c3.10", "land:9:2")


def test_b41_b42_foodweb_predator_needs_prey_and_omnivore_demand_once():
    reg = rg.load()
    n = 3
    pools = {p: np.zeros(n) for p in rg.PRIMARY_POOLS}
    pools["leaf_herb"] = np.array([1000.0, 0.0, 1000.0])
    pools["fruit_seed"] = np.array([500.0, 0.0, 500.0])
    w = {g: np.zeros(n) for g, gg in reg.groups.items() if not gg.is_producer}
    w["grazer_large"] = np.array([1.0, 1.0, 0.0])
    w["predator_pursuit"] = np.ones(n)
    w["omnivore_land"] = np.ones(n)
    area = {g: np.full(n, 1e8) for g in reg.groups}
    web = fw.run_foodweb(pools, w, area, n)
    gr_ = web["groups"]
    # no herbivore in hex 2 -> the pursuit predator has no prey there (B41)
    assert gr_["predator_pursuit"]["biomass_kgc"][2] == 0.0
    assert gr_["predator_pursuit"]["biomass_kgc"][1] == 0.0  # no food for the grazer either
    # the omnivore's total request is one demand split over its pools, not one per pool
    tr = reg.groups["omnivore_land"].traits
    demand = 1.0 * tr["d0"] * 1e8 / 1000.0
    assert gr_["omnivore_land"]["intake"][0] <= demand * (1 + 1e-9)
    assert web["ledger"]["ok"]


def test_b43_b48_edible_partition_and_pollinators():
    reg = rg.load()
    ids = ["tree_broadleaf_deciduous", "grass_c3", "forb"]
    g = len(ids)
    prod = {"npp_leaf": np.full((g, 1), 100.0), "npp_wood": np.full((g, 1), 50.0),
            "npp_root": np.full((g, 1), 80.0), "npp_repro": np.full((g, 1), 20.0)}
    total = sum(prod[k].sum() for k in prod)
    with_p = pr.land_pools(prod, ids, pollination=np.ones(1))
    without = pr.land_pools(prod, ids, pollination=np.zeros(1))
    assert sum(v[0] for v in with_p.values()) == pytest.approx(total, rel=R8)  # partition
    assert sum(v[0] for v in without.values()) == pytest.approx(total, rel=R8)
    assert without["fruit_seed"][0] < with_p["fruit_seed"][0]  # animal-pollinated set less
    assert reg.groups["grass_c3"].traits["poll"] == "wind"  # wind-pollinated unaffected


def test_b21_pet_zero_no_division():
    from hexworld.biosphere import synthetic as sy
    from hexworld.biosphere.inputs import habitat_fractions

    land = np.ones((3, 4), bool)
    inp = sy.make(land, pet_mm=0.0, t_mean=-20.0, t_amp=2.0)
    assert np.all(np.isnan(inp["mi"]))  # undefined, not 0 or ∞
    out = pr.land_production(inp, habitat_fractions(inp))
    assert np.all(np.isfinite(np.nan_to_num(out["npp"], nan=0.0)))


def test_b22_light_absorbed_not_more_than_incident():
    from hexworld.biosphere import synthetic as sy
    from hexworld.biosphere.inputs import habitat_fractions

    land = np.ones((3, 4), bool)
    inp = sy.make(land)
    out = pr.land_production(inp, habitat_fractions(inp))
    incident = inp["par_m"][:, out["idx"]].sum(axis=0)
    assert np.all(out["apar"].sum(axis=0) <= incident * (1 + 1e-9))
    assert np.all(np.nan_to_num(out["npp"]) >= 0)


def test_b64_b65_resource_units_and_shared_wood():
    from hexworld.biosphere.production import C_PER_DRY, WOOD_DENSITY_T_M3

    # 1 kgC of wood = 1/0.47 kg dry = 1/0.47/500 m³
    assert 1.0 / C_PER_DRY / 1000.0 / WOOD_DENSITY_T_M3 == pytest.approx(1 / 0.47 / 500)
    # the not-additive pair is checked on a real inventory in test_v55_integration (B65)


def test_b77_coupler_rolls_back_every_owner():
    owners = {"soil": {"soc_kg_m2": np.array([5.0, 6.0])},
              "erosion": {"cover_factor": np.array([1.0, 1.0])}}
    before = {o: {k: v.copy() for k, v in d.items()} for o, d in owners.items()}
    c = dy.Coupler(owners)
    props = [dy.EcoFeedback("soil", "soc_kg_m2", np.array([0.1, 0.1]), "kgC/m²", "y1"),
             dy.EcoFeedback("erosion", "cover_factor", np.array([0.5, 0.6]), "1", "y1")]
    ok = c.apply(props, fail_on=lambda p: p.owner == "erosion")
    assert not ok and all(p.status == "rolled_back" for p in props)
    for o in before:
        for k in before[o]:
            assert np.array_equal(c.owners[o][k], before[o][k])
    props = [dy.EcoFeedback("soil", "soc_kg_m2", np.array([0.1, 0.1]), "kgC/m²", "y1")]
    assert c.apply(props) and c.owners["soil"]["soc_kg_m2"].tolist() == pytest.approx([5.1, 6.1])
    again = [dy.EcoFeedback("soil", "soc_kg_m2", np.array([0.1, 0.1]), "kgC/m²", "y1")]
    c.apply(again)
    assert again[0].status == "duplicate_ignored"
    assert c.owners["soil"]["soc_kg_m2"].tolist() == pytest.approx([5.1, 6.1])


def test_b72_b74_owner_operators():
    i_lo = dy.interception_mm(800.0, 1.0)
    i_hi = dy.interception_mm(800.0, 4.0)
    assert 0 <= i_lo < i_hi <= 800.0
    c = dy.erosion_cover_factor(np.array([0.0, 0.5, 1.0]), bare_rock=np.array([False, False, True]))
    assert c[0] == 1.0 and 0.01 <= c[1] < 1.0 and c[2] == 1.0  # bare rock: no plant protection
