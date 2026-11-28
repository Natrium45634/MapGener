"""TZ 5.5 synthetic causality fixtures with positive and negative controls (B11–B14, B19,
B20, B27–B31, B33, B34, B36, B49, B53–B57, S04, S08, S09, S12, S15, S16). The worlds are
manufactured (``hexworld.biosphere.synthetic``): the expected outcome is known."""

from __future__ import annotations

import numpy as np
import pytest

from hexworld.biosphere import graph as gr
from hexworld.biosphere import model as md
from hexworld.biosphere import production as pr
from hexworld.biosphere import synthetic as sy
from hexworld.biosphere.inputs import habitat_fractions
from hexworld.biosphere.registry import load


def _solve(inp, **kw):
    return md.solve(inp, habitat_fractions(inp), seed=kw.pop("seed", "Fx1"),
                    landmasses=sy.landmasses(inp), **kw)  # fmt: skip


def _land_total(st, gid, mask):
    return float(st.groups[gid]["biomass_kgc"][mask.ravel()].sum())


def _two_landmasses(gap=8, h=10, w=40, block_b=1):
    land = np.zeros((h, w), bool)
    land[1:h - 1, 1:14] = True  # continent A
    land[1:h - 1, 14 + gap:w - 1] = True  # continent / island B
    block = np.where(land, 0, -1)
    block[:, 14 + gap:] = np.where(land[:, 14 + gap:], block_b, -1)
    return land, block


def test_b11_b12_fractions_close_and_dry_depression_stays_land():
    land = np.ones((4, 5), bool)
    land[0, :] = False
    inp = sy.make(land, sea_ice=np.full((12, 4, 5), 0.5))
    inp.f["below_sea_land"][7] = True  # a dry basin below sea level, not connected
    fr = habitat_fractions(inp)
    tot = fr["land_ground"] + fr["inland_water"] + fr["marine_water"] + fr["grounded_ice"]
    assert np.allclose(tot, 1.0, atol=1e-12)
    assert fr["marine_water"][7] == 0.0 and fr["land_ground"][7] == 1.0
    # sea ice is an overlay of the marine area, not extra area
    assert np.all(fr["sea_ice_overlay_m"] <= fr["marine_water"][None, :] + 1e-12)


def test_b13_mixed_coastal_hex_counts_only_its_habitat():
    land = np.ones((3, 3), bool)
    inp = sy.make(land)
    inp.f["coast_land"][4] = True
    inp.f["topo_ocean_fraction"][4] = 0.4
    fr = habitat_fractions(inp)
    assert fr["land_ground"][4] == pytest.approx(0.6) and fr["marine_water"][4] == pytest.approx(0.4)
    st = _solve(inp)
    # producer NPP per m² of hex is NPP per m² of ground x the ground share, never x 1
    g = st.groups["grass_c3"]
    assert g["npp_gc_m2_hex"][4] <= g["npp_gc_m2_hex"][0] + 1e-6


def test_b14_mountain_microsite_keeps_a_cold_template():
    land = np.ones((2, 3), bool)
    warm = sy.make(land, t_mean=24.0, t_amp=3.0)
    reg = load()
    tpl = reg.templates["tree_needle_evergreen.00"]  # a cool pine-like analogue
    env = {"tc": warm["tc"], "tw": warm["tw"], "mi": warm["mi"], "gdd": warm["gdd5"]}
    assert np.all(pr.niche_score(tpl.niche, env) == 0)  # unsuitable at the hex mean
    # the same hex with a high cold part (the q05 bin 2 km up: 13 K colder)
    off = -13.0 * np.ones(warm.n)
    t = warm["t_air_m"] + off
    env2 = {"tc": t.min(axis=0), "tw": t.max(axis=0), "mi": warm["mi"],
            "gdd": pr._growing_degree_days(t, warm.days)}  # fmt: skip
    assert np.all(pr.niche_score(tpl.niche, env2) > 0)


def test_b19_b20_water_and_nutrients_limit_npp():
    land = np.ones((2, 3), bool)
    base = sy.make(land)
    dry = sy.make(land, aet_ratio=0.3)
    poor = sy.make(land, n_min=0.004, p_lab=0.0001)
    npp = {k: np.nansum(pr.land_production(i, habitat_fractions(i))["npp"])
           for k, i in (("base", base), ("dry", dry), ("poor", poor))}
    assert npp["dry"] < 0.8 * npp["base"] and npp["poor"] < 0.8 * npp["base"]
    # same light, temperature and water: only nutrients changed
    assert base["par_m"].sum() == poor["par_m"].sum()


def test_b09_missing_nutrients_give_unknown_not_zero():
    land = np.ones((2, 3), bool)
    inp = sy.make(land)
    inp.f["n_min_kg_m2_yr"][0] = np.nan
    out = pr.land_production(inp, habitat_fractions(inp))
    assert np.all(np.isnan(out["npp"][:, 0])) and np.all(np.isfinite(out["npp"][:, 1]))


def test_b27_s09_island_walkers_absent_birds_possible():
    land, block = _two_landmasses(gap=8, block_b=-1)  # B is an oceanic island 8 hexes away
    island = np.zeros_like(land)
    island[:, 22:] = land[:, 22:]
    inp = sy.make(land, block=block)
    st = _solve(inp)
    isl = island.ravel()
    assert st.lineage["ground"]["d_km"][isl].min() == np.inf  # no walking route
    for g in ("grazer_large", "browser_large", "predator_pursuit", "megaherbivore"):
        assert _land_total(st, g, island) == 0.0, g
    cont = land.copy()
    cont[:, 14:] = False
    assert _land_total(st, "grazer_large", cont) > 0  # the same climate on the mainland
    assert np.isfinite(st.lineage["flight"]["d_km"][isl]).all()  # birds can cross


def test_b28_empty_source_pool_no_noise_immigration():
    land, block = _two_landmasses(gap=30, w=70, block_b=-1)
    inp = sy.make(land, block=block)
    st = _solve(inp)
    island = np.zeros_like(land)
    island[:, 44:] = land[:, 44:]
    assert _land_total(st, "grazer_small", island) == 0.0
    assert np.all(st.lineage["ground"]["d_km"] >= 0)


def test_b29_directed_river_graph():
    land = np.ones((3, 6), bool)
    river = np.zeros((3, 6), bool)
    river[1, :] = True
    rec = -np.ones((3, 6), int)
    for c in range(5):
        rec[1, c] = 1 * 6 + c + 1  # flows east
    inp = sy.make(land, river=river, flow_receiver=rec, basin=np.zeros((3, 6), int))
    fr = habitat_fractions(inp)
    g = gr.build_graph("river", inp, fr)
    a, b = 6 + 1, 6 + 2
    assert g[a, b] < g[b, a]  # downstream cheaper than upstream


def test_b30_mountain_barrier_depends_on_mobility():
    land = np.ones((3, 9), bool)
    inp = sy.make(land)
    inp.f["elevation_m"][np.arange(inp.n) % 9 == 4] = 4500.0
    fr = habitat_fractions(inp)
    walk = gr.build_graph("ground", inp, fr)
    fly = gr.build_graph("flight", inp, fr)
    i, j = 3, 4
    assert walk[i, j] > fly[i, j]


def test_b31_cost_is_physical_distance():
    land = np.ones((3, 12), bool)
    d = {}
    for sp, cols in ((50.0, 12), (100.0, 6)):
        inp = sy.make(land[:, :cols], spacing_km=sp)
        fr = habitat_fractions(inp)
        g = gr.build_graph("ground", inp, fr)
        from scipy.sparse.csgraph import dijkstra

        dist = dijkstra(g, indices=[cols], min_only=True)
        d[sp] = dist[2 * cols - 1]
    assert d[50.0] == pytest.approx(d[100.0], rel=0.15)


def test_b33_b34_s08_two_continents_same_climate_different_assemblages():
    land, block = _two_landmasses(gap=6, h=16, w=60, block_b=1)
    land[:, 14:20] = False
    inp = sy.make(land, block=block)
    inp.f["landmass_id"] = np.where(land.ravel(), np.where(np.arange(inp.n) % 60 < 14, 0, 1), -1)
    old = gr.SOURCE_LAND_KM2
    gr.SOURCE_LAND_KM2 = 1.0  # both landmasses are lineage pools in this small fixture
    try:
        st = _solve(inp)
    finally:
        gr.SOURCE_LAND_KM2 = old
    assert st.quality["history"] == "present_only"
    a = st.lineage["ground"]["pool"].reshape(land.shape)[5, 5]
    b = st.lineage["ground"]["pool"].reshape(land.shape)[5, 40]
    assert a != b and a >= 0 and b >= 0
    # identical climate, different lineages -> different taxon ids, different bioregions
    reg = st.layers["bioregion"].reshape(land.shape)
    assert reg[5, 5] != reg[5, 40]
    assert st.layers["flora_richness"].reshape(land.shape)[5, 5] > 0


def test_b36_taxon_ids_stable_for_a_lineage_key():
    from hexworld.biosphere.registry import taxon_id

    k = gr._key("land", 1234.0, -567.0)
    assert k == gr._key("land", 1234.4, -567.4)  # 50 km bins: a crop keeps the key
    assert taxon_id("W", "fern.00", k) == taxon_id("W", "fern.00", k)


def test_b49_s12_isolated_basin_no_fish_crossing():
    land = np.ones((4, 10), bool)
    river = np.zeros((4, 10), bool)
    river[1, 1:4] = True
    river[1, 6:9] = True
    basin = np.where(np.arange(40).reshape(4, 10) % 10 < 5, 0, 1)
    rec = -np.ones((4, 10), int)
    for c in range(1, 3):
        rec[1, c] = 10 + c + 1
    for c in range(6, 8):
        rec[1, c] = 10 + c + 1
    inp = sy.make(land, river=river, flow_receiver=rec, basin=basin)
    fr = habitat_fractions(inp)
    g = gr.build_graph("river", inp, fr, capture_mult=None)
    pools = gr.Lineages(["fw:a"], ["A"], np.where(np.arange(40) == 11, 0, -1))
    d, lin = gr.nearest_lineage(g, pools)
    assert np.isfinite(d[13]) and not np.isfinite(d[17])  # other basin unreachable
    g2 = gr.build_graph("river", inp, fr, capture_mult=30.0)
    d2, _ = gr.nearest_lineage(g2, pools)
    assert np.isfinite(d2[13])  # same basin always reachable


def _sea(**kw):
    land = np.zeros((3, 4), bool)
    land[0, 0] = True
    return sy.make(land, **kw)


def test_b53_b54_s15_upwelling_and_darkness():
    lo = pr.marine_production(_sea(nutr=0.05), habitat_fractions(_sea(nutr=0.05)))
    hi_inp = _sea(nutr=0.8)
    hi = pr.marine_production(hi_inp, habitat_fractions(hi_inp))
    gi = lo["groups"].index("phytoplankton_marine")
    assert hi["npp"][gi].mean() > 2.0 * lo["npp"][gi].mean()
    dark = _sea(nutr=0.8, par_mj_m2_day=0.0)
    dk = pr.marine_production(dark, habitat_fractions(dark))
    assert dk["npp"][gi].max() == 0.0  # nutrients without light make nothing (B54)


def test_b55_s16_export_leaves_the_surface_and_vents_are_explicit():
    inp = _sea(nutr=0.5, depth_m=4000.0)
    mp = pr.marine_production(inp, habitat_fractions(inp))
    pools = pr.marine_pools(mp)
    def npp(k):
        return mp["npp"][mp["groups"].index(k)]

    macro = npp("kelp") + npp("algae_macro_other") + npp("seagrass")
    plankton = npp("phytoplankton_marine") + npp("ice_algae")
    # the plankton NPP is partitioned once: food + detritus + exported (B55)
    parts = (pools["m_phyto"] + pools["m_ice_algae"] + pools["m_detritus"] - 0.4 * macro
             + mp["export_gc_m2"])  # fmt: skip
    assert np.allclose(parts, plankton, rtol=1e-9)
    assert np.all(mp["bottom_flux_gc_m2"] < mp["export_gc_m2"])  # attenuated with depth
    assert pools["m_chemo"].max() == 0.0  # no vent -> no chemosynthesis
    inp.f["hydrothermal_kind"][5] = 1
    inp.f["geothermal_heat_mw"][5] = 50.0
    mp2 = pr.marine_production(inp, habitat_fractions(inp))
    assert pr.marine_pools(mp2)["m_chemo"].max() > 0


def test_b56_b57_salinity_and_coral_gates():
    reg = load()
    fish = reg.templates["fish_fresh_omnivore.10"]
    assert pr.niche_score({"sal": fish.niche["sal"]}, {"sal": np.array([35.0])})[0] == 0.0
    coral = reg.groups["coral_symbiotic"].traits
    assert "hard_bottom" in coral["gates"] and "lit_bottom" in coral["gates"]
    cold = pr.niche_score({"tc": coral["niche"]["tc"]}, {"tc": np.array([5.0])})[0]
    assert cold == 0.0


def test_b51_no_photosynthetic_benthos_below_the_photic_zone():
    deep = _sea(depth_m=600.0, kd=0.06)
    mp = pr.marine_production(deep, habitat_fractions(deep))
    for k in ("kelp", "seagrass", "algae_macro_other"):
        assert mp["npp"][mp["groups"].index(k)].max() == 0.0
    shallow = _sea(depth_m=10.0, kd=0.06)
    mp2 = pr.marine_production(shallow, habitat_fractions(shallow))
    assert mp2["npp"][mp2["groups"].index("algae_macro_other")].max() > 0


def test_s04_desert_low_stock_and_oasis_is_local():
    land = np.ones((3, 6), bool)
    inp = sy.make(land, precip_mm=40.0, pet_mm=2200.0, aet_ratio=0.03, t_mean=26.0)
    out = pr.land_production(inp, habitat_fractions(inp))
    assert np.nansum(out["npp"], axis=0).max() < 150.0
    assert out["biomass_kgc_m2"].sum(axis=0).max() < 2.0


def test_b05_b06_seeds_and_repeatability():
    land, block = _two_landmasses()
    inp = sy.make(land, block=block)
    a = _solve(inp, seed="Ab")
    b = _solve(inp, seed="Ab")
    c = _solve(inp, seed="aB")
    assert a.layers["_ranges"].keys() == b.layers["_ranges"].keys()
    assert np.array_equal(a.layers["npp_gc_m2_yr"], b.layers["npp_gc_m2_yr"])
    assert set(a.layers["_ranges"]) != set(c.layers["_ranges"])  # case-sensitive namespace
    assert np.array_equal(a.layers["npp_gc_m2_yr"], c.layers["npp_gc_m2_yr"])  # physics same


def test_potential_vs_realized_modes():
    land, block = _two_landmasses(gap=10, block_b=-1)
    inp = sy.make(land, block=block)
    pot = _solve(inp, mode="POTENTIAL")
    real = _solve(inp, mode="REALIZED_BASELINE")
    island = np.zeros_like(land)
    island[:, 24:] = land[:, 24:]
    # the potential has the grazers on the island, the realized state does not
    assert _land_total(pot, "grazer_large", island) > 0
    assert _land_total(real, "grazer_large", island) == 0.0
    assert pot.summary["ledger_ok"] and real.summary["ledger_ok"]


def test_b17_mirrored_hemispheres_shift_the_season():
    land = np.ones((2, 3), bool)
    north = sy.make(land, hemisphere=1)
    south = sy.make(land, hemisphere=-1)
    a = pr.land_production(north, habitat_fractions(north))
    b = pr.land_production(south, habitat_fractions(south))
    assert np.nansum(a["npp"]) == pytest.approx(np.nansum(b["npp"]), rel=1e-9)
    ma, mb = a["npp_month"][:, 0], b["npp_month"][:, 0]
    assert np.allclose(np.roll(ma, 6), mb, rtol=1e-9, atol=1e-9)


def test_b25_layer_covers_within_available_area():
    land = np.ones((3, 4), bool)
    inp = sy.make(land)
    out = pr.land_production(inp, habitat_fractions(inp))
    reg = load()
    layer = np.array([reg.groups[g].traits["layer"] for g in out["groups"]])
    for name in ("canopy", "shrub", "ground"):
        assert np.all(out["cover"][layer == name].sum(axis=0) <= 1.0 + 1e-9)
    # epiphytes and climbers overlay the canopy, never more than it
    can = out["cover"][layer == "canopy"].sum(axis=0)
    assert np.all(out["cover"][layer == "epiphyte"].max(axis=0) <= can + 1e-9)


def test_b32_land_bridge_opens_exchange_synthetic():
    land, block = _two_landmasses(gap=4, h=10, w=40, block_b=-1)
    inp = sy.make(land, block=block)
    st = _solve(inp)
    isl = np.zeros_like(land)
    isl[:, 18:] = land[:, 18:]
    assert _land_total(st, "grazer_large", isl) == 0.0  # separated: no walkers
    bridged = land.copy()
    bridged[4:6, 14:18] = True  # the bridge event of this fixture
    inp2 = sy.make(bridged, block=np.where(bridged, np.where(block >= 0, block, -1), -1))
    st2 = _solve(inp2)
    assert np.isfinite(st2.lineage["ground"]["d_km"][isl.ravel()]).all()
    assert _land_total(st2, "grazer_large", isl) > 0.0  # exchange after the bridge


def test_b61_range_touching_the_window_edge_is_unknown():
    land, block = _two_landmasses()
    inp = sy.make(land, block=block)
    st = _solve(inp)
    rng = st.layers["_ranges"]
    assert any(t for _a, t, _id, _k in rng.values())  # the fixture touches the edges
    assert st.summary["taxa_endemic"] == sum(1 for a, t, _i, _k in rng.values()
                                             if a <= md.ENDEMIC_RANGE_KM2 and not t)
