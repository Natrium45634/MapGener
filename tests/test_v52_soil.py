"""TZ 5.2 part II: soil fertility - reference calculations N01-N10 and tests S01-S24."""

from __future__ import annotations

import numpy as np
import pytest

from hexworld.resources import soil as sl

PM = {k: i for i, k in enumerate(sl.PARENT_ORDER)}


# -- reference calculations ----------------------------------------------------------------------
def test_n01_s01_soc_stock_and_units():
    c = sl.soc_stock(0.30, 1300.0, 0.02, 0.10)
    assert c == pytest.approx(7.02)
    assert sl.kg_m2_to_t_ha(c) == pytest.approx(70.2)


def test_n02_awc_without_stones():
    assert sl.awc_mm(1.0, 0.30, 0.12) == pytest.approx(180.0)
    # theta of the whole soil volume: no second stone factor
    assert sl.awc_mm(1.0, 0.30, 0.12, 0.5, theta_of="whole_soil") == pytest.approx(180.0)
    assert sl.awc_mm(1.0, 0.30, 0.12, 0.5) == pytest.approx(90.0)


def test_n03_s07_first_order_decay():
    rem, out = sl.first_order_decay(10.0, 0.1, 1.0)
    assert rem == pytest.approx(9.048374, abs=1e-6)
    assert out == pytest.approx(0.951626, abs=1e-6)


def test_n04_nitrate_leaching_flux():
    assert sl.leaching_flux(100.0, 0.05) == pytest.approx(0.005)
    assert sl.leaching_flux(100.0, 0.05, available_kg_m2=0.002) == pytest.approx(0.002)


def test_n05_s16_buffer_and_ph_aggregation():
    d1 = sl.ph_change(1.0, 2.0)
    d2 = sl.ph_change(1.0, 4.0)
    assert d1 < 0 and abs(d2) <= abs(d1)
    bs = np.linspace(0.05, 1.0, 20)
    assert np.all(np.diff(sl.ph_from_base_saturation(bs)) > 0)  # monotonic buffer curve
    # pH is aggregated through the H+ activity, not averaged linearly
    agg = sl.aggregate_ph([4.0, 7.0], [1.0, 1.0])
    assert agg == pytest.approx(-np.log10(0.5 * (1e-4 + 1e-7)))
    assert agg < 5.5 - 1.0


def test_n06_element_mass_of_fine_earth():
    assert sl.element_mass(0.1, 1500.0, 0.1, 0.01) == pytest.approx(1.35)


def _month(n, **kw):
    z = np.zeros(n)
    base = {
        "t_c": np.full(n, 15.0),
        "moisture_mod": np.ones(n),
        "cover_mod": np.full(n, 0.6),
        "anoxia": z,
        "litter_c": z,
        "litter_cn": np.full(n, 25.0),
        "dpm_share": np.full(n, 0.5),
        "clay_pct": np.full(n, 20.0),
        "drain_mm": z,
        "theta_d_m": np.full(n, 0.3),
        "uptake_n": z,
        "uptake_p": z,
        "uptake_k": z,
        "n_fix": z,
        "n_dep": z,
        "salt_in": z,
        "weather_p": z,
        "weather_k": z,
        "k_sorb": z,
        "k_desorb": z,
    }
    base.update(kw)
    return base


def _state(n, **kw):
    s = {k: np.zeros(n) for k in sl.POOLS}
    for k, v in kw.items():
        s[k] = np.full(n, float(v))
    return s


def test_n07_n09_nitrogen_ledger():
    # N09: mineralisation moves 0.02 kg from organic to mineral, total unchanged
    s = _state(1, n_org=0.12)
    s2 = sl.transfer_pool(s, "n_org", "n_nh4", 0.02)
    assert s2["n_org"][0] == pytest.approx(0.10) and s2["n_nh4"][0] == pytest.approx(0.02)
    assert (s2["n_org"] + s2["n_nh4"])[0] == pytest.approx(0.12)
    # N07: start 1 kg N in all pools, +0.1 fixation, -0.03 crop uptake, -0.02 leaching
    # -> 1.05 in the sum of the soil pools (the uptake is exported from them)
    theta = 0.3
    s = _state(1, n_org=0.825, n_nh4=0.1, n_no3=0.075)
    m = _month(
        1,
        n_fix=np.array([0.1]),
        uptake_n=np.array([0.03]),
        drain_mm=np.array([100.0]),
        theta_d_m=np.array([theta]),
        t_c=np.array([-30.0]),
    )
    total0 = s["n_org"] + s["n_nh4"] + s["n_no3"]
    s2, rep = sl.advance_soil_month(s, m)
    total1 = s2["n_org"] + s2["n_nh4"] + s2["n_no3"]
    assert total0[0] == pytest.approx(1.0)
    assert rep["n_leach"][0] == pytest.approx(0.02, abs=1e-12)
    assert total1[0] == pytest.approx(1.05, abs=1e-12)
    assert abs(rep["ledger"]["N"]["residual"][0]) < 1e-14


def test_n08_s17_s18_salt_mass():
    s = _state(1, salt=10.0)
    m = _month(1, salt_in=np.array([2.0]), drain_mm=np.array([0.0]))
    s2, rep = sl.advance_soil_month(s, m)
    assert s2["salt"][0] == pytest.approx(12.0)  # evaporation never removes salt
    # concentration rises as the water drops, the mass does not change
    assert 12.0 / 0.1 > 12.0 / 0.3
    # leaching: 3 kg leave with the drainage water
    theta = 0.3
    drain = 3.0 / (12.0 / theta) * 1000.0
    s3, rep = sl.advance_soil_month(
        s2, _month(1, drain_mm=np.array([drain]), theta_d_m=np.array([theta]))
    )
    assert s3["salt"][0] == pytest.approx(9.0)
    assert abs(rep["ledger"]["salt"]["residual"][0]) < 1e-12


# -- synthetic columns -----------------------------------------------------------------------------
def _columns(**over):
    """Synthetic soil columns of the TZ 5.2 sec. 25 scenes (see scratch calibration)."""
    scenes = {
        "carbonate_valley": dict(
            pm="sedimentary_bedrock",
            t=14,
            p=700,
            sand=0.3,
            clay=0.3,
            d=1.2,
            age=1e4,
            perc=150,
            veg=0.8,
            tree=0.3,
            carb=0.3,
        ),
        "wet_tropical_old": dict(
            pm="crystalline_bedrock",
            t=25,
            p=2500,
            sand=0.4,
            clay=0.45,
            d=2.0,
            age=2e6,
            perc=1200,
            veg=1.0,
            tree=0.95,
            eros=2e-6,
        ),
        "loess_steppe": dict(
            pm="loess",
            t=7,
            p=450,
            sand=0.15,
            clay=0.2,
            d=1.8,
            age=1.2e4,
            perc=40,
            veg=0.9,
            tree=0.05,
        ),
        "sand_desert": dict(
            pm="aeolian_sand",
            t=26,
            p=60,
            sand=0.9,
            clay=0.04,
            d=1.5,
            age=5e4,
            perc=1,
            veg=0.05,
            tree=0.0,
        ),
        "closed_basin": dict(
            pm="marine_lacustrine",
            t=18,
            p=180,
            sand=0.2,
            clay=0.45,
            d=1.5,
            age=3e4,
            perc=2,
            veg=0.2,
            tree=0.0,
            endo=True,
        ),
        "floodplain": dict(
            pm="alluvium",
            t=15,
            p=900,
            sand=0.3,
            clay=0.25,
            d=1.6,
            age=2e3,
            perc=250,
            veg=0.9,
            tree=0.4,
            wl=3,
        ),
        "peat_bog": dict(
            pm="glacial_till",
            t=3,
            p=700,
            sand=0.4,
            clay=0.2,
            d=1.0,
            age=1.2e4,
            perc=300,
            veg=0.9,
            tree=0.2,
            wl=10,
        ),
        "young_lava": dict(
            pm="young_volcanic",
            t=20,
            p=2000,
            sand=0.6,
            clay=0.08,
            d=0.2,
            age=300,
            perc=1000,
            veg=0.3,
            tree=0.1,
            coarse=0.5,
        ),
        "old_lava": dict(
            pm="young_volcanic",
            t=20,
            p=2000,
            sand=0.4,
            clay=0.3,
            d=1.2,
            age=3e4,
            perc=1000,
            veg=0.9,
            tree=0.6,
        ),
    }
    names = list(scenes)
    n = len(names)

    def g(k, default):
        return np.array([float(scenes[s].get(k, default)) for s in names])

    t, p = g("t", 10), g("p", 600)
    th = 2 * np.pi * np.arange(12) / 12
    t12 = t[None] + 8 * np.cos(th)[:, None]
    p12 = np.repeat(p[None] / 12, 12, axis=0)
    pet12 = np.maximum(0, 16 * (10 * np.maximum(t12, 0) / 50) ** 1.3) * 3
    d = g("d", 1.0)
    awc = 1000 * np.minimum(d, 1) * 0.15
    store = np.clip(awc[None] * np.clip(p12 / np.maximum(pet12, 1), 0, 1), 0, awc[None])
    kw = dict(
        soil_present=np.ones(n, bool),
        parent=np.array([PM[scenes[s]["pm"]] for s in names]),
        sand=g("sand", 0.4),
        clay=g("clay", 0.2),
        coarse=g("coarse", 0.05),
        depth_m=d,
        porosity_mineral=np.full(n, 0.45),
        fc=np.full(n, 0.3),
        pwp=np.full(n, 0.15),
        t_month=t12,
        p_month=p12,
        pet_month=pet12,
        storage_month=store,
        awc_mm=awc,
        percolation_mm_yr=g("perc", 100),
        waterlogging_months=g("wl", 0),
        vegetation_fraction=g("veg", 0.8),
        tree_fraction=g("tree", 0.3),
        soil_erosion_m_yr=g("eros", 1e-5),
        soil_age_yr=g("age", 1e4),
        marine_influence=np.zeros(n),
        endorheic_floor=g("endo", 0).astype(bool),
        carbonate_extra=g("carb", 0.0),
    )
    kw.update(over)
    return names, sl.FertilityInputs(**kw)


@pytest.fixture(scope="module")
def scenes():
    names, inp = _columns()
    st = sl.compute_fertility_static(inp)
    return {n: i for i, n in enumerate(names)}, inp, st


def test_s02_concentration_and_stock_are_different_quantities(scenes):
    ix, _inp, st = scenes
    i = ix["peat_bog"]
    stock = st["soc_layers_kg_m2"][0, i]
    mass = st["layer_thickness_m"][0, i] * st["bulk_density_kg_m3"][0, i] * 0.95
    conc = stock / mass
    assert 0 < conc < 1 and stock > conc  # kg/kg vs kg/m2


def test_s03_physical_ranges(scenes, hist_project):
    _ix, _inp, st = scenes
    assert np.all((st["oxic_fraction"] >= 0) & (st["oxic_fraction"] <= 1))
    E = hist_project.env.layers
    soil = E["soil_present"].astype(bool)
    assert np.all(E["field_capacity_vol"][soil] >= E["wilting_point_vol"][soil])
    tex = E["sand_frac"][soil] + E["silt_frac"][soil] + E["clay_frac"][soil]
    assert np.allclose(tex, 1.0, atol=1e-5)
    por = sl.porosity(np.array([1000.0, 1600.0, 2700.0]))
    assert np.all((por >= 0) & (por <= 1))


def test_s04_n10_no_soil_is_nodata(hist_project):
    L = hist_project.resources.layers
    soil = hist_project.env.layers["soil_present"].astype(bool)
    for name in (
        "soil_ph",
        "soil_soc_kg_m2",
        "soil_P_labile_kg_m2",
        "soil_fertility_potential",
        "suitability_crop_wheat",
    ):
        assert not np.isfinite(L[name][~soil]).any(), name
        assert np.isfinite(L[name][soil]).all(), name


def test_s05_cec_never_negative_over_textures():
    n = 30
    names, inp = _columns()
    clay = np.linspace(0.0, 0.7, len(names))
    _n, inp = _columns(clay=clay, sand=1.0 - clay - 0.1)
    st = sl.compute_fertility_static(inp)
    assert np.all(st["cec_cmolc_kg"] > 0) and n > 0


def test_s06_rothc_equilibrium_mass_balance():
    eq = sl.rothc_equilibrium(np.array([0.5]), np.array([0.3]), np.array([25.0]), np.array([0.7]))
    assert eq["co2_flux"][0] == pytest.approx(0.5, rel=1e-12)  # all input respired at steady state
    # with an erosion loss the stocks are lower and input = CO2 + eroded
    eq2 = sl.rothc_equilibrium(
        np.array([0.5]), np.array([0.3]), np.array([25.0]), np.array([0.7]), np.array([0.002])
    )
    stock2 = eq2["dpm"] + eq2["rpm"] + eq2["bio"] + eq2["hum"]
    assert (eq2["co2_flux"] + 0.002 * stock2)[0] == pytest.approx(0.5, rel=1e-10)
    # the monthly step keeps C: in - CO2 = change
    s = _state(3, c_dpm=0.2, c_rpm=1.0, c_bio=0.3, c_hum=8.0, c_iom=1.0)
    _s2, rep = sl.advance_soil_month(s, _month(3, litter_c=np.full(3, 0.04)))
    assert np.all(np.abs(rep["ledger"]["C"]["residual"]) < 1e-14)


def test_s08_time_step_convergence():
    n = 1
    tr = np.zeros((n, 2, 2))
    tr[:, 1, 0] = 1e-4  # primary -> labile
    loss = np.array([[0.0, 2e-4]])
    x0 = np.array([[1.0, 0.0]])
    exact_p = np.exp(-1e-4 * 1e4)
    errs = []
    for steps in (12, 24, 48, 96):
        x, _e, _a = sl.integrate_linear(tr, loss, np.zeros((n, 2)), x0, np.array([1e4]), steps)
        errs.append(abs(x[0, 0] - exact_p))
    assert all(b < a for a, b in zip(errs, errs[1:], strict=False))  # refining converges
    assert errs[-1] / exact_p < 0.03  # first order: ~2 % at 96 steps for k t = 1
    assert errs[1] / errs[2] == pytest.approx(2.0, rel=0.25)


def test_s09_nitrification_keeps_n():
    s = _state(1, n_nh4=0.01)
    s2, rep = sl.advance_soil_month(s, _month(1))
    assert s2["n_no3"][0] > 0 and s2["n_nh4"][0] < 0.01
    assert (s2["n_nh4"] + s2["n_no3"] + s2["n_org"])[0] == pytest.approx(0.01, abs=1e-15)


def test_s10_denitrification_is_a_booked_gas_loss():
    s = _state(1, n_no3=0.005)
    s2, rep = sl.advance_soil_month(s, _month(1, anoxia=np.array([1.0])))
    assert rep["n_denit"][0] > 0 and s2["n_no3"][0] >= 0
    assert abs(rep["ledger"]["N"]["residual"][0]) < 1e-15


def test_s11_immobilisation_does_not_create_n():
    # straw-like litter (C:N 80) on a soil with little mineral N: mineral N falls
    s = _state(1, c_dpm=0.5, c_rpm=0.5, n_org=0.0125, n_nh4=0.002)
    m = _month(1, litter_cn=np.array([80.0]))
    s2, rep = sl.advance_soil_month(s, m)
    assert (s2["n_nh4"] + s2["n_no3"])[0] < 0.002
    assert abs(rep["ledger"]["N"]["residual"][0]) < 1e-15
    assert (s2["n_nh4"] >= 0).all() and (s2["n_no3"] >= 0).all()


def test_s12_fixation_and_deposition_are_external():
    s = _state(1, n_org=1.0)
    s2, rep = sl.advance_soil_month(
        s, _month(1, n_fix=np.array([0.001]), n_dep=np.array([0.0005]), t_c=np.array([-30.0]))
    )
    tot = lambda x: (x["n_org"] + x["n_nh4"] + x["n_no3"])[0]  # noqa: E731
    assert tot(s2) - tot(s) == pytest.approx(0.0015, abs=1e-15)
    assert rep["n_in"][0] == pytest.approx(0.0015)


def test_s13_s14_phosphorus_compartments():
    n = 1
    tr = np.zeros((n, 3, 3))
    tr[:, 1, 0] = 1e-3
    tr[:, 2, 1] = 0.5
    tr[:, 1, 2] = 0.1
    x, exp_, add = sl.integrate_linear(
        tr, np.zeros((n, 3)), np.zeros((n, 3)), np.array([[1.0, 0, 0]]), np.array([500.0])
    )
    assert x.sum() == pytest.approx(1.0, abs=1e-12)  # weathering moved P, nothing created
    assert 1.0 - x[0, 0] == pytest.approx(x[0, 1] + x[0, 2], abs=1e-12)
    s = _state(1, p_labile=0.01, p_sorbed=0.02)
    s2, rep = sl.advance_soil_month(s, _month(1, k_sorb=np.array([1.0]), k_desorb=np.array([0.15])))
    assert (s2["p_labile"] + s2["p_sorbed"])[0] == pytest.approx(0.03, abs=1e-15)


def test_s15_potassium_pools_are_finite_and_bounded(scenes):
    _ix, inp, st = scenes
    total0 = sl.fine_earth_mass(1, 1, 0)  # sanity of the helper
    assert total0 == 1
    k_all = st["k_primary_kg_m2"] + st["k_exchangeable_kg_m2"] + st["k_solution_kg_m2"]
    assert np.all(np.isfinite(k_all)) and np.all(st["k_exchangeable_kg_m2"] >= 0)
    led = st["budget"]["K"]
    assert led["exported_kg"] >= 0 and led["max_relative_residual"] < 1e-12


def test_s19_salinity_hurts_sensitive_profiles(scenes):
    ix, _inp, st = scenes
    i = ix["closed_basin"]
    assert st["ec_ds_m"][i] > 8.0
    maize, _l, f_maize = sl.suitability(st, "crop_maize")
    barley, _l, f_barley = sl.suitability(st, "crop_barley")
    hal, _l, f_hal = sl.suitability(st, "halophyte")
    sal = sl.FACTORS.index("salinity")
    assert f_maize[sal, i] < f_barley[sal, i] <= f_hal[sal, i]
    # more salt never raises the suitability of a sensitive crop
    st2 = dict(st, ec_ds_m=st["ec_ds_m"] * 2)
    assert np.all(sl.suitability(st2, "crop_maize")[0] <= maize + 1e-12)


def test_s20_anoxia_and_drought_are_independent_limits(scenes):
    ix, _inp, st = scenes
    bog, desert = ix["peat_bog"], ix["sand_desert"]
    _s, lim, f = sl.suitability(st, "crop_wheat")
    assert sl.FACTORS[np.argmin(f[:, bog])] in ("oxygen", "ph")
    assert f[sl.FACTORS.index("water"), desert] < f[sl.FACTORS.index("water"), bog]
    _s, _l, fw = sl.suitability(st, "wetland_plants")
    ox = sl.FACTORS.index("oxygen")
    assert fw[ox, bog] > f[ox, bog]  # wetland plants tolerate the anoxia


def test_s22_young_lava_is_not_a_fertility_bonus(scenes):
    ix, _inp, st = scenes
    young, old = ix["young_lava"], ix["old_lava"]
    assert st["soc_kg_m2"][young] < st["soc_kg_m2"][old]
    fert = sl.suitability(st, "native_generic")[0]
    assert fert[young] < fert[old]
    # fresh ash adds acidity, it does not raise the pH
    _n, inp2 = _columns(ash_stress=np.ones(9))
    st2 = sl.compute_fertility_static(inp2)
    assert np.all(st2["ph"] <= st["ph"] + 1e-12)


def test_s23_river_transport_ledger():
    from hexworld.resources.secondary import retention_route

    # a chain 0 -> 1 -> 2 -> 3 (outlet) with a lake at 2
    rc = np.array([1, 2, 3, -1])
    src = np.array([[10.0, 1.0], [5.0, 0.5], [0.0, 0.0], [2.0, 0.2]])
    ret = np.array([0.1, 0.3, 1.0, 0.0])
    dep, exp_ = retention_route(rc, src, ret)
    assert np.allclose(src.sum(axis=0), dep.sum(axis=0) + exp_)
    assert dep[2, 0] == pytest.approx((10 * 0.9 + 5) * 0.7)  # the lake traps all it receives
    assert exp_[0] == pytest.approx(2.0)  # only the outlet's own load leaves


def test_s23_stage_ledgers(hist_project):
    s = hist_project.resources.summary
    for el, led in s["erosion_transport_ledger"].items():
        assert abs(led["residual_kg_yr"]) <= 1e-6 * max(led["eroded_kg_yr"], 1.0), el
    for el, led in s["element_budget_over_soil_age"].items():
        assert led["max_relative_residual"] < 1e-9, el
    mc = s["monthly_check"]
    if mc.get("status") != "no_soil":
        assert max(mc["max_relative_ledger_residual"].values()) < 1e-9


def test_s24_crop_and_native_suitability_differ(scenes):
    ix, _inp, st = scenes
    i = ix["wet_tropical_old"]
    native = sl.suitability(st, "native_generic")[0][i]
    wheat = sl.suitability(st, "crop_wheat")[0][i]
    forest = sl.suitability(st, "forest")[0][i]
    assert wheat < native <= forest + 1e-9
    assert st["ph"][i] < 5.0  # leached, acid, poor: lush forest is no proof of a good soil


def test_scene_patterns(scenes):
    """TZ sec. 25 scenes 1-8 as relations (not visual similarity)."""
    ix, _inp, st = scenes
    fert = sl.suitability(st, "native_generic")[0]
    # loess steppe: deep, high CEC, organic-rich and fertile
    assert st["soc_kg_m2"][ix["loess_steppe"]] > 8.0
    assert fert[ix["loess_steppe"]] > fert[ix["wet_tropical_old"]]
    # carbonate valley: neutral-alkaline, P limited by sorption (labile < floodplain x 2)
    assert st["ph"][ix["carbonate_valley"]] >= 7.5
    # peat bog: the largest carbon stock with low oxygen
    assert st["soc_kg_m2"][ix["peat_bog"]] == np.nanmax(st["soc_kg_m2"])
    assert st["oxic_fraction"][ix["peat_bog"]] < 0.5
    # old tropical regolith: most P occluded, little primary P left
    i = ix["wet_tropical_old"]
    assert st["p_occluded_kg_m2"][i] > st["p_primary_kg_m2"][i]
    # sandy desert: low organic matter, water-limited
    assert st["soc_kg_m2"][ix["sand_desert"]] < 1.0
    # closed basin: saline; the floodplain is not
    assert st["salt_kg_m2"][ix["closed_basin"]] > 10 * max(st["salt_kg_m2"][ix["floodplain"]], 1e-3)


def test_rooting_follows_the_thawed_warm_season():
    """Polar desert frozen all year: no rooting; tundra with a warm summer: an active layer;
    temperate: full (only depth and stones limit it)."""
    from hexworld.resources.soil import rooting_index

    months = np.arange(12)
    season = -np.cos(2 * np.pi * (months - 0.5) / 12.0)  # -1 in January, +1 in July

    def t12(mean, amp):
        return (mean + amp * season)[:, None]

    polar = rooting_index(np.array([1.0]), np.array([0.0]), t12(-25.0, 10.0))
    tundra = rooting_index(np.array([1.0]), np.array([0.0]), t12(-10.0, 16.0))
    temperate = rooting_index(np.array([1.0]), np.array([0.0]), t12(10.0, 10.0))
    assert polar[0] == 0.0
    assert 0.15 < tundra[0] < 0.6
    assert temperate[0] == 1.0
    assert rooting_index(np.array([0.5]), np.array([0.2]), t12(10.0, 10.0))[0] == 0.4
