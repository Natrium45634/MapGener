"""TZ 5.4.1 G5: the weather year changes soil, groundwater, snow, glaciers and rivers
(v5_4_1::T50-T66, N13-N18)."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from hexworld.atmosphere import fv
from hexworld.atmosphere import hydrostate as hs

DAYS = np.array([31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31], float)


def _soil(n=1, z=0.5, fc=0.30, wp=0.12, sat=0.45, icap=50.0):
    awc = (fc - wp) * z * 1000.0
    return hs.soil_from_layers(np.full(n, fc), np.full(n, wp), np.full(n, sat), np.full(n, awc),
                               np.full(n, icap))  # fmt: skip


def test_n13_soil_stores():
    s = _soil()
    assert s.taw[0] == pytest.approx(90.0)
    assert (s.s_fc[0], s.s_wp[0], s.s_sat[0]) == (pytest.approx(150.0), pytest.approx(60.0),
                                                  pytest.approx(225.0))  # fmt: skip
    assert 100.0 + 30.0 - 10.0 - 0.0 == 120.0


def test_n14_groundwater_recession():
    out = hs.baseflow(100.0, 1.0, 1.0)
    assert out == pytest.approx(63.2120559, rel=1e-9)
    assert 100.0 - out == pytest.approx(36.7879441, rel=1e-9)


def test_n15_runoff_to_discharge():
    assert fv.mm_to_m3(10.0, 1.0) == pytest.approx(10_000.0)
    assert fv.mm_per_day_to_m3_s(10.0, 1.0) == pytest.approx(0.11574074, rel=1e-7)


def test_n16_t57_closed_lake():
    assert hs.lake_balance(1000.0, 0.01, 0.005, 100.0, 10.0) == pytest.approx(1010.5)


def test_n17_t60_pdd():
    assert hs.pdd_daily([-2, 3, -1, 1]) == pytest.approx(4.0)
    assert max(np.mean([-2, 3, -1, 1]), 0.0) * 4 == pytest.approx(1.0)  # the monthly proxy
    # a month with a negative mean still thaws on its warm days (expected PDD > 0)
    assert hs.expected_pdd(-2.0, 30.0) > 5.0
    assert hs.expected_pdd(-2.0, 30.0, sigma=0.0) == 0.0


def test_n18_t61_snow_melts_before_ice():
    snow, ice, left = hs.degree_day_melt(6.0, 1000.0, 4.0, 3.0, 6.0)
    assert snow == pytest.approx(6.0) and ice == pytest.approx(12.0)
    assert snow + ice == pytest.approx(18.0) and left == pytest.approx(0.0)


def _climate(n=4, t=10.0, p=80.0, pet=60.0):
    return (np.full((12, n), p), np.full((12, n), t), np.full((12, n), pet))


def test_t52_t53_t55_t65_budgets_close_and_no_water_from_nothing():
    soil = _soil(4)
    st = hs.initial_state(soil, np.zeros(4))
    p, t, pet = _climate()
    new, flux, led = hs.run_year(st, p, t, pet, DAYS, soil, np.zeros(4, bool))
    assert np.abs(led["residual_mm"]).max() < 1e-9
    assert (new.soil >= 0).all() and (new.soil <= soil.s_sat + 1e-9).all()
    dry, fdry, ldry = hs.run_year(replace(new), np.zeros((12, 4)), t, pet, DAYS, soil,
                                  np.zeros(4, bool))  # fmt: skip
    assert np.abs(ldry["residual_mm"]).max() < 1e-9
    assert (fdry["soil"][-1] <= fdry["soil"][0] + 1e-9).all()  # drought dries the soil
    assert (fdry["aet"][-1] < fdry["aet"][0]).all()
    empty = hs.State(soil.s_wp.copy(), np.zeros(4), np.zeros(4), np.zeros(4), np.zeros(4))
    _n, f0, _l = hs.run_year(empty, np.zeros((12, 4)), t, pet, DAYS, soil, np.zeros(4, bool))
    assert f0["aet"].max() == 0.0  # no rain, no available water: no transpiration


def test_t54_frozen_soil_sends_more_water_to_the_rivers():
    soil = _soil(1, icap=2.0)
    st = hs.initial_state(soil, np.zeros(1))
    p = np.full((12, 1), 60.0)
    pet = np.zeros((12, 1))
    _a, warm, _ = hs.run_year(st, p, np.full((12, 1), 5.0), pet, DAYS, soil, np.zeros(1, bool))
    # cold, but rain (the snow threshold moved for the fixture): frozen infiltration only
    old = hs.T_SNOW
    try:
        hs.T_SNOW = -50.0
        _b, frozen, _ = hs.run_year(st, p, np.full((12, 1), -5.0), pet, DAYS, soil,
                                    np.zeros(1, bool))  # fmt: skip
    finally:
        hs.T_SNOW = old
    assert frozen["surface_runoff"].sum() > warm["surface_runoff"].sum()


def test_t59_snow_partition():
    soil = _soil(3)
    st = hs.initial_state(soil, np.zeros(3))
    p = np.full((12, 3), 50.0)
    t = np.tile(np.array([-10.0, 1.0, 10.0]), (12, 1))
    _s, f, _l = hs.run_year(st, p, t, np.zeros((12, 3)), DAYS, soil, np.zeros(3, bool))
    assert np.allclose(f["rain"] + f["snowfall"], p)
    assert f["snowfall"][:, 0].sum() > f["snowfall"][:, 2].sum()


def test_t51_t63_carry_over_and_firn_memory():
    soil = _soil(2)
    ice = np.array([0.0, 5e5])
    glacier = np.array([False, True])
    st = hs.initial_state(soil, ice)
    p, t, pet = _climate(2, t=-8.0, p=60.0, pet=5.0)
    y1, _f1, l1 = hs.run_year(st, p, t, pet, DAYS, soil, glacier)
    y2, _f2, l2 = hs.run_year(y1, p, t, pet, DAYS, soil, glacier)
    assert y1.firn[1] > 0 and y1.ice[1] - ice[1] < y1.firn[1]  # snow -> firn, not instant ice
    assert y2.ice[1] > y1.ice[1]  # firn slowly becomes ice
    assert y1.swe[0] > 0 and y2.swe[0] > y1.swe[0]  # the second year starts where the first ended
    assert abs(l2["residual_mm"]).max() < 1e-6


def test_t62_warm_year_lowers_the_glacier_balance():
    soil = _soil(1)
    glacier = np.array([True])
    st = hs.initial_state(soil, np.array([5e5]))
    p, t, pet = _climate(1, t=-3.0, p=70.0, pet=10.0)
    _c, fc, lc = hs.run_year(st, p, t, pet, DAYS, soil, glacier)
    _w, fw, lw = hs.run_year(st, p, t + 4.0, pet, DAYS, soil, glacier)
    assert lw["smb_mm_we"][0] < lc["smb_mm_we"][0]
    assert fw["ice_melt"].sum() > fc["ice_melt"].sum()  # more melt runoff in the warm year


def test_t56_routing_counts_each_upstream_once():
    rc = np.array([2, 2, 3, -1])  # 0 -> 2, 1 -> 2, 2 -> 3, 3 outlet
    lu = hs.routing_operator(rc)
    q = hs.route(lu, np.array([1.0, 2.0, 3.0, 4.0]))
    assert np.allclose(q, [1.0, 2.0, 6.0, 10.0])
    q2 = hs.route(lu, np.array([[1.0, 2.0, 3.0, 4.0], [0, 0, 0, 1.0]]))
    assert np.allclose(q2[1], [0, 0, 0, 1.0])


def test_t58_t64_t66_weather_year_state_in_the_stage(hist_project):
    from hexworld.pipeline_v54 import run_atmosphere

    cfg = hist_project.config
    normals_before = dict(hist_project.atmosphere.grid_fields)
    wet = replace(cfg, atmosphere=replace(cfg.atmosphere, weather_year=True, weather_year_seed="A"))
    r1 = run_atmosphere(wet, hist_project)
    r1b = run_atmosphere(wet, hist_project)
    L = r1["layers"]
    for k in ("weather_soil_moisture_end_mm", "weather_runoff_anomaly_pct", "weather_aet_mm",
              "weather_discharge_ratio", "state_groundwater_mm"):  # fmt: skip
        assert k in L
    ws = r1["summary"]["weather_year"]["water_state"]
    assert ws["ledger_max_abs_residual_mm"] < 1e-6
    assert "geometry_pending" in ws["geometry"]  # mass changes, geometry not faked
    # T66: the same year again gives the same state (no double water / melt)
    assert np.array_equal(L["state_soil_mm"], r1b["layers"]["state_soil_mm"], equal_nan=True)
    # the normals are untouched by the weather
    for k in ("precipitation_mm", "slp_pa"):
        assert np.array_equal(hist_project.atmosphere.grid_fields[k], normals_before[k])
    two = replace(cfg, atmosphere=replace(cfg.atmosphere, weather_year=True,
                                          weather_year_seed="A", weather_year_index=2))  # fmt: skip
    r2 = run_atmosphere(two, hist_project)
    assert r2["summary"]["weather_year"]["water_state"]["years_in_sequence"] == 2
    assert not np.array_equal(r2["layers"]["state_soil_mm"], L["state_soil_mm"], equal_nan=True)
