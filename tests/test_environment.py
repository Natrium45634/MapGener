"""v4.5 ecological foundation: the control cases of TZ 4.5, section 12."""

from __future__ import annotations

import math
from dataclasses import replace

import numpy as np
import pytest

from hexworld import views
from hexworld.api import export_hex_months_csv, hex_months, stage_depth
from hexworld.io import load_project, save_project
from hexworld.stages import environment as env

DAYS = env.MONTH_DAYS


# -- months -----------------------------------------------------------------------------------
def test_monthly_precipitation_keeps_the_annual_and_season_sums():
    p_s = np.array([[100.0, 0.0], [300.0, 0.0], [600.0, 1200.0], [200.0, 0.0]])  # 1200 each
    p_m = env.monthly_precipitation(p_s)
    assert p_m.shape == (12, 2) and (p_m >= 0).all()
    assert np.allclose(p_m.sum(axis=0), 1200.0, rtol=1e-12)
    for s in range(4):
        assert np.allclose(p_m[env.SEASON_OF_MONTH == s].sum(axis=0), p_s[s], rtol=1e-12)
    assert p_m[6, 0] > p_m[0, 0]  # the wet season stays wet
    assert np.allclose(p_m[[11, 0, 1, 3, 4, 8, 9, 10], 1], 0.0)  # a dry season stays dry


def test_monthly_temperature_keeps_the_annual_mean_and_the_hemisphere():
    t_s = np.array([[5.0, 25.0], [15.0, 15.0], [25.0, 5.0], [15.0, 15.0]])  # north, south
    t_m = env.monthly_temperature(t_s, np.array([15.0, 15.0]))
    assert np.allclose((t_m * DAYS[:, None]).sum(axis=0) / DAYS.sum(), 15.0, atol=1e-12)
    assert np.argmax(t_m[:, 0]) == 6 and np.argmin(t_m[:, 0]) == 0  # July / January
    assert np.argmax(t_m[:, 1]) == 0 and np.argmin(t_m[:, 1]) == 6  # the opposite phase
    # the season means are reproduced by the 3-month means
    jja = (t_m[5:8, 0] * DAYS[5:8]).sum() / DAYS[5:8].sum()
    assert jja == pytest.approx(25.0, abs=0.3)


def test_fao56_extraterrestrial_radiation_and_polar_cases():
    ra, n = env.extraterrestrial_radiation(-20.0, 246)  # FAO-56 example 8: 3 September
    assert ra == pytest.approx(32.2, abs=0.1)
    lat = np.array([90.0, -90.0, 89.99, -89.99, 0.0])
    ra_j, n_j = env.extraterrestrial_radiation(lat, np.full(5, 172))  # northern summer
    assert np.isfinite(ra_j).all() and (ra_j >= 0).all() and np.isfinite(n_j).all()
    assert ra_j[1] == pytest.approx(0.0, abs=1e-9) and n_j[1] == pytest.approx(0.0)  # night
    assert n_j[0] == pytest.approx(24.0) and ra_j[0] > ra_j[4]  # polar day beats the equator
    assert n == pytest.approx(11.7, abs=0.1)


def test_monthly_pet_keeps_the_v4_season_sums():
    pet_s = np.array([[20.0], [150.0], [400.0], [160.0]])
    t_m = env.monthly_temperature(np.array([[0.0], [12.0], [24.0], [12.0]]))
    pet_m = env.monthly_pet(pet_s, t_m, np.array([45.0]))
    assert np.allclose(pet_m.sum(axis=0), pet_s.sum(axis=0))
    assert (pet_m >= 0).all() and np.argmax(pet_m[:, 0]) in (5, 6, 7)


# -- soils --------------------------------------------------------------------------------------
def test_saxton_rawls_reproduces_the_paper_table():
    # (sand, clay) -> (PWP, FC) of Saxton & Rawls 2006, table 3, OM 2.5 %
    table = {
        "sand": ((0.88, 0.05), (0.05, 0.10)),
        "loam": ((0.40, 0.20), (0.14, 0.28)),
        "silt loam": ((0.20, 0.15), (0.11, 0.31)),
        "clay loam": ((0.32, 0.34), (0.22, 0.36)),
        "clay": ((0.25, 0.50), (0.30, 0.42)),
    }
    for name, ((s, c), (wp_ref, fc_ref)) in table.items():
        wp, fc, sat, ks = env.saxton_rawls(s, c, 2.5)
        assert wp == pytest.approx(wp_ref, abs=0.015), name
        assert fc == pytest.approx(fc_ref, abs=0.015), name
        assert 0.0 <= wp < fc <= sat and ks > 0.0, name
    assert env.saxton_rawls(0.88, 0.05, 2.5)[3] > 30 * env.saxton_rawls(0.25, 0.5, 2.5)[3]


def test_available_water_control_and_bare_rock():
    assert env.available_water(1.0, 0.28, 0.13, 0.20) == pytest.approx(120.0)
    assert env.available_water(0.0, 0.28, 0.13, 0.20) == 0.0
    assert env.available_water(2.5, 0.28, 0.13, 0.0, root_depth_m=1.0) == pytest.approx(150.0)


# -- water ----------------------------------------------------------------------------------------
def _month(s=50.0, gw=0.0, swe=0.0, p=100.0, t=20.0, pet=80.0, awc=100.0, icap_month=50.0,
           drainage=0.5, k_base=0.6, deep=0.0):  # fmt: skip
    w = env.WaterParams(
        rain_days=6.0, deep_loss_share=deep,
        baseflow_timescale_yr=-30.0 / env.YEAR_DAYS / math.log(1.0 - k_base),
    )  # fmt: skip
    a = lambda v: np.array([float(v)])  # noqa: E731
    state = {"S": a(s), "GW": a(gw), "SWE": a(swe)}
    return env.month_step(state, a(p), a(t), a(pet), a(awc), a(icap_month / 6.0), drainage, 30.0, w)


def test_tz_section_6_2_control_month():
    new, f = _month()
    assert f["surface_runoff"][0] == pytest.approx(50.0)
    assert f["aet"][0] == pytest.approx(80.0)
    assert new["S"][0] == pytest.approx(10.0) and new["GW"][0] == pytest.approx(4.0)
    assert f["runoff"][0] == pytest.approx(56.0, abs=1e-3)
    total_in = 50.0 + 0.0 + 0.0 + 100.0
    total_out = new["S"][0] + new["GW"][0] + new["SWE"][0] + f["aet"][0] + f["runoff"][0]
    assert total_out == pytest.approx(total_in, abs=1e-9)


def test_no_water_makes_no_water_and_snow_is_stored_then_melted():
    new, f = _month(s=0.0, p=0.0, pet=80.0)
    assert all(abs(v[0]) < 1e-12 for v in f.values()) and new["S"][0] == 0.0
    winter, fw = _month(s=0.0, p=60.0, t=-15.0, pet=0.0)
    assert fw["snowfall"][0] == pytest.approx(60.0, abs=0.01)  # a smooth rain/snow split
    assert winter["SWE"][0] == pytest.approx(60.0, abs=0.05) and fw["runoff"][0] < 0.05
    spring, fs = _month(s=0.0, swe=60.0, p=0.0, t=8.0, pet=20.0)
    assert 60.0 - spring["SWE"][0] == pytest.approx(fs["melt"][0] - fs["snowfall"][0])


def test_periodic_year_closes_the_budget_and_the_seasons_matter():
    t = np.full((12, 2), 24.0)
    pet = np.full((12, 2), 110.0)
    even = np.full(12, 100.0)
    wet_dry = np.where(np.arange(12) < 6, 200.0, 0.0)  # the same 1200 mm in 6 months
    p = np.stack([even, wet_dry], axis=1)
    w = env.WaterParams()
    res = env.water_balance(p, t, pet, np.full(2, 150.0), np.full(2, 40.0), np.full(2, 0.2), w)
    assert res.max_residual_mm < 1e-6 and res.converged.all()
    s = res.monthly["soil_storage"]
    assert s[:, 0].std() < 5.0  # wet tropics: steady stores
    assert s[11, 1] < 0.3 * s[5, 1]  # seasonal tropics: the dry half empties the soil
    stress_even = 1.0 - res.monthly["aet"][:, 0] / 110.0
    stress_seasonal = 1.0 - res.monthly["aet"][:, 1] / 110.0
    assert stress_seasonal.max() > stress_even.max() + 0.3


def test_soil_texture_changes_the_water_not_the_climate():
    t = np.full((12, 2), 15.0)
    p = np.tile(np.array([60, 60, 60, 30, 10, 5, 5, 10, 30, 60, 60, 60.0])[:, None], (1, 2))
    pet = np.tile(np.array([10, 20, 40, 70, 100, 130, 140, 120, 80, 40, 20, 10.0])[:, None], (1, 2))
    res = env.water_balance(
        p, t, pet, np.array([40.0, 200.0]), np.array([200.0, 20.0]), np.array([0.4, 0.05]),
        env.WaterParams(),
    )  # fmt: skip
    assert res.annual["aet"][1] > res.annual["aet"][0] + 30.0  # deep loam keeps summer water
    assert res.max_residual_mm < 1e-6


def test_hand_follows_the_flow_path_not_the_straight_line():
    # 0 <- 1 <- 2 (river at 0) | 3 -> 4 (sea) : hex 3 is next to the river but drains away
    z = np.array([10.0, 30.0, 60.0, 15.0, -5.0])
    rc = np.array([0, 0, 1, 4, 4])
    ref = np.array([10.0, np.nan, np.nan, np.nan, 0.0])
    h = env.hand(z, rc, ref)
    assert h.tolist() == [0.0, 20.0, 50.0, 15.0, 0.0]


# -- the whole stage ------------------------------------------------------------------------------
def test_environment_stage_on_a_generated_world(hist_project):
    p = hist_project
    assert stage_depth(p) == "v45" and p.env is not None
    assert p.env.validation["ok"], p.env.validation
    L, M = p.env.layers, p.env.monthly
    land = p.hydro.layers["land_mask"] & (p.hydro.layers["lake_id"] < 0)
    soil = L["soil_present"]
    assert soil.any() and not soil[~land].any()
    tex = L["sand_frac"][soil] + L["silt_frac"][soil] + L["clay_frac"][soil]
    assert np.allclose(tex, 1.0, atol=1e-5)
    assert np.isnan(L["sand_frac"][~land]).all()  # no soil on the sea floor
    assert np.isnan(M["aet_monthly_mm"][:, ~land]).all()
    for name, arr in M.items():
        assert np.isfinite(arr[:, land]).all(), name
    assert (L["wilting_point_vol"][soil] < L["field_capacity_vol"][soil]).all()
    # the months reproduce the v4 annual fields
    t_ann = (M["temperature_monthly_c"] * DAYS[:, None, None]).sum(axis=0) / DAYS.sum()
    assert np.allclose(t_ann, p.hydro.layers["temperature_mean_c"], atol=1e-3)
    p_v4 = p.hydro.layers["precipitation_mm_yr"]
    assert np.allclose(M["precip_monthly_mm"].sum(axis=0), p_v4, rtol=1e-4, atol=1e-2)
    wb = p.env.summary["water_balance"]
    assert wb["max_abs_residual_mm"] < 1e-6
    assert p.env.summary["river_q_source"].startswith("v4 snapshot")
    # no biome is named anywhere
    text = repr(p.env.summary) + repr(p.env.methods) + repr(list(L))
    for word in ("biome", "forest", "savanna", "desert", "tundra"):
        assert word not in text.lower()


def test_environment_views_and_month_choice_are_display_only(hist_project):
    p = hist_project
    before = {k: v.copy() for k, v in p.env.monthly.items()}
    for vid in ("seasonality", "soils", "moisture", "ecology"):
        spec = views.VIEWS[vid]
        assert spec.availability(p)[0]
        for mode in spec.modes:
            for month in (0, 6):
                views.view_rgb(p, vid, mode.id, set(spec.default_overlays()) | {f"month:{month}"})
            assert spec.legend(p, mode.id)
    a = views.view_rgb(p, "seasonality", "temperature", {"month:0"})
    b = views.view_rgb(p, "seasonality", "temperature", {"month:6"})
    assert not np.array_equal(a, b)
    assert all(np.array_equal(before[k], p.env.monthly[k], equal_nan=True) for k in before)


def test_environment_is_deterministic_saved_and_loaded(tmp_path, hist_config, hist_project):
    from hexworld.pipeline import generate

    again = generate(hist_config, reuse=hist_project)  # reuses v1-v4, recomputes nothing
    assert again.env.content_hashes() == hist_project.env.content_hashes()
    fresh = replace(hist_project, env=None, cache={})
    redo = generate(hist_config, reuse=fresh)  # v4.5 recomputed from the same v4 state
    assert redo.env.content_hashes() == hist_project.env.content_hashes()
    path = save_project(hist_project, tmp_path / "w")
    loaded = load_project(path)
    assert loaded.env is not None and loaded.env.validation["ok"]
    assert loaded.env.content_hashes() == hist_project.env.content_hashes()
    rows = hex_months(loaded, 10, 10)
    assert len(rows) == 12 and rows[0]["month"] == 1
    csv_path = export_hex_months_csv(loaded, 10, 10, tmp_path / "hex.csv")
    assert (tmp_path / "hex.csv").read_text(encoding="utf-8").count("\n") == 13
    assert csv_path.endswith("hex.csv")


def test_a_v4_project_without_v45_opens_and_says_not_calculated(tmp_path, hist_project):
    v4_only = replace(hist_project, env=None)
    path = save_project(v4_only, tmp_path / "v4")
    loaded = load_project(path)
    assert loaded.env is None and loaded.hydro is not None
    ok, reason = views.VIEWS["soils"].availability(loaded)
    assert not ok and "не рассчитаны" in reason
    with pytest.raises(LookupError):
        hex_months(loaded, 1, 1)
