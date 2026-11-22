"""v4.7: altitudinal zonation, topoclimate, sub-hex conditions (TZ 4.7, tests T01-T42)."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from hexworld import generate
from hexworld.config import Config, ConfigError, TopoclimateParams
from hexworld.io import load_project, save_project
from hexworld.pipeline_v47 import _chunks, _units, run_topoclimate, topo_stage_keys
from hexworld.topoclimate import atmosphere as atm
from hexworld.topoclimate import redistribution as rd
from hexworld.topoclimate import solar
from hexworld.topoclimate import thermal as th
from hexworld.topoclimate.units import midrank, stratify

# -- T01-T03: areas, hypsometry, water ------------------------------------------------------


def _belt_shares(weights, wtype, belts, n_belts):
    dry = wtype == 0
    hex_w = weights[dry].sum()
    whole = np.array([weights[dry & (belts == b)].sum() for b in range(n_belts)])
    return whole, whole / hex_w


def test_t01_mixed_hex_area_closure_and_belt_shares():
    n = 10
    wtype = np.array([0] * 6 + [1] * 2 + [2] * 2, dtype=np.uint8)
    z = np.array([100, 200, 300, 400, 500, 600, -5, -5, -10, -10], dtype=np.float64)
    hexes = np.zeros(n, dtype=np.int64)
    w = np.full(n, 1.0 / n)
    zero = np.zeros(n)
    code, *_ = stratify(hexes, wtype, z, zero, np.full(n, np.nan), zero, zero, 1)
    inv, uw, rep, uhex, ucode, off = _units(hexes, code, w, [z / 200.0], 1)
    assert uw.sum() == pytest.approx(1.0)
    utype = np.where(ucode == 0, 2, np.where(ucode == 1, 1, 0))
    assert uw[utype == 0].sum() == pytest.approx(0.6)
    assert uw[utype == 1].sum() == pytest.approx(0.2) and uw[utype == 2].sum() == pytest.approx(0.2)
    belts = np.where(utype == 0, 5, 0)
    whole, land = _belt_shares(uw, utype, belts, 8)
    assert whole.sum() == pytest.approx(0.6) and land.sum() == pytest.approx(1.0)


def test_t02_two_units_give_two_belts_not_one_mean():
    z = np.array([500.0, 2500.0])
    w = np.array([0.5, 0.5])
    assert float((w * z).sum()) == 1500.0
    rank = midrank(z, np.zeros(2, dtype=np.int64), 1)
    assert rank.tolist() == [0.25, 0.75]  # CDF(1000 m) = 0.5: half the area is below
    t_ref = np.full(12, 9.0)
    belts = []
    for zz in (500.0, 2500.0, 1500.0):
        t = atm.lapse_temperature(t_ref, 1500.0, zz)
        met = th.season_metrics(th.daily_from_monthly(t[:, None]))
        margin = th.thermal_margin(met["gsl_days"], met["gst_c"])
        belts.append(
            int(th.thermal_belt(met["gsl_days"], met["gst_c"], margin, met["min_daily_c"] > 5)[0])
        )
    assert belts[0] != belts[1]  # the hex holds both belts; the mean height gives a third answer


def test_t03_dry_depression_below_datum_and_lake_air_at_its_surface():
    t_dry = atm.lapse_temperature(20.0, 0.0, -500.0)
    assert t_dry == pytest.approx(23.25)  # z < 0 is valid and warmer
    t_lake = atm.lapse_temperature(20.0, 0.0, -430.0)  # lake air at the surface, not the bed
    assert t_lake == pytest.approx(22.795)
    assert atm.lapse_temperature(20.0, 0.0, 0.0) == 20.0  # ocean air at sea level


# -- T04-T07: profile, pressure, humidity --------------------------------------------------------


def test_t04_lapse_from_the_reference_height_once():
    assert atm.lapse_temperature(16.0, 1000.0, 2500.0) == pytest.approx(6.25)


def test_t05_preserve_mean():
    out = atm.preserve_mean(10.0, np.array([-4.0, 4.0]), np.array([0.5, 0.5]))
    assert out.tolist() == [6.0, 14.0] and out.mean() == 10.0


def test_t06_hypsometric_pressure():
    assert atm.pressure_exponential_kpa(101.3, 0.0, 1000.0, 288.0) == pytest.approx(90.0, abs=0.2)


def test_t07_saturation_and_vpd():
    es = atm.es_kpa(20.0)
    assert es == pytest.approx(2.338, abs=2e-3)
    assert es * (1 - 0.5) == pytest.approx(1.169, abs=2e-3)


# -- T08-T15: geometry, horizon, sun ----------------------------------------------------------


def test_t08_flat_plain():
    slope, aspect = solar.slope_aspect(np.array([0.0]), np.array([0.0]))
    assert slope[0] == 0.0 and np.isnan(aspect[0])
    nx, ny, nz = solar.surface_normal(0.0, 0.0)
    assert (float(nx), float(ny), float(nz)) == (0.0, 0.0, 1.0)
    vis, svf_h, svf_s = solar.sky_view(np.zeros((1, 16)), np.zeros(1))
    assert vis[0] == 1.0 and svf_h[0] == 1.0 and svf_s[0] == 1.0
    assert rd.cap_static(0.0, 0.5, 0.0, 0.0, 0.0) == 0.0


def test_t09_inclined_plane_gradient_and_normal():
    from hexworld.topoclimate.units import gradient

    def plane(x, y):  # z = 0.1 east + 0.2 north (metres); the world y points south
        return 0.1 * np.asarray(x) * 1000.0 + 0.2 * (-np.asarray(y)) * 1000.0

    p, q = gradient(plane, np.array([3.0]), np.array([7.0]), 0.5)
    assert p[0] == pytest.approx(0.1) and q[0] == pytest.approx(0.2)
    slope, _ = solar.slope_aspect(p, q)
    assert np.degrees(slope[0]) == pytest.approx(12.604, abs=1e-3)
    n = np.array(solar.surface_normal(0.1, 0.2), dtype=float)
    for tangent in ([1.0, 0.0, 0.1], [0.0, 1.0, 0.2]):  # east and north along the plane
        assert np.dot(n, tangent) == pytest.approx(0.0, abs=1e-12)
    assert n[2] > 0 and np.linalg.norm(n) == pytest.approx(1.0)


def test_t10_visible_sky_vs_radiative_svf():
    vis, svf_h, _ = solar.sky_view(np.full((1, 32), np.deg2rad(30.0)))
    assert vis[0] == pytest.approx(0.5) and svf_h[0] == pytest.approx(0.75)


def test_t11_wall_shades_low_sun_only():
    h = solar.horizon_angles(np.array([[1000.0]]), np.array([[1000.0]]), np.array([[0.1]]), 8)
    assert np.degrees(h[0, 0]) == pytest.approx(45.0)
    assert np.deg2rad(30.0) < h[0, 0] < np.deg2rad(60.0)  # 30 deg shaded, 60 deg not


def test_t12_equinox_noon_slopes_mirror_between_hemispheres():
    for lat, sunny in ((45.0, "south"), (-45.0, "north")):
        sx, sy, sz = solar.sun_vector(np.deg2rad(lat), 0.0, 0.0)
        assert float(sz) == pytest.approx(0.7071, abs=1e-4)
        q = np.tan(np.deg2rad(30.0))
        south = np.array(solar.surface_normal(0.0, q), dtype=float)  # rises to the north
        north = np.array(solar.surface_normal(0.0, -q), dtype=float)
        s = np.array([sx, sy, sz], dtype=float)
        a, b = float(south @ s), float(north @ s)
        if sunny == "north":
            a, b = b, a
        assert a == pytest.approx(0.966, abs=1e-3) and b == pytest.approx(0.259, abs=1e-3)


def test_t13_equator_equinox_day_and_ra():
    doy = 80
    d = solar.declination(doy)
    assert float(solar.day_length_h(0.0, d)) == pytest.approx(12.0, abs=0.05)
    ra = float(solar.ra_analytic(0.0, doy))
    assert float(solar.toa_flat_numeric(0.0, doy, 96)) == pytest.approx(ra, rel=2e-3)


def test_t14_polar_night_and_day_are_finite():
    lat = np.deg2rad(np.array([80.0, -80.0]))
    d = solar.declination(355)
    assert solar.day_length_h(lat, d).tolist() == pytest.approx([0.0, 24.0])
    assert float(solar.ra_analytic(lat[0], 355)) == pytest.approx(0.0, abs=1e-9)
    hor = np.zeros((2, 16))
    sun = np.full((12, 2), 0.5)
    r, ref, hours = solar.monthly_radiation(
        lat, solar.surface_normal(np.zeros(2), np.zeros(2)), hor, np.ones(2), np.zeros(2), sun
    )
    assert np.isfinite(r).all() and np.isfinite(ref).all() and r[11, 0] == 0.0


def test_t15_energy_units():
    assert 480.0 * 3600.0 * 1e-6 == pytest.approx(1.728)
    assert solar.I_SC_W * 60.0 / 1e6 == pytest.approx(solar.GSC_MJ_MIN)


# -- T16-T18: cold-air pools, mass effect ----------------------------------------------------


def test_t16_calm_clear_nights_pool_cold_air():
    calm, _ = rd.cap_anomaly(0.8, 0.5, 0.1, 14.0)
    windy, _ = rd.cap_anomaly(0.8, 10.0, 0.1, 14.0)
    assert calm < windy <= 0.0


def test_t17_cap_needs_a_basin_not_just_low_height():
    flat = rd.cap_static(0.0, 0.05, 1.0, 0.0, 0.0)
    basin = rd.cap_static(400.0, 0.05, 1.0, 300.0, 0.2)
    assert flat == 0.0 and basin > 0.3


def test_t18_mass_elevation_effect_is_off_by_default():
    assert TopoclimateParams().mass_elevation_effect_max_c == 0.0
    assert atm.mass_elevation_effect(4000.0, 300.0, 0.0) == 0.0
    plateau = atm.mass_elevation_effect(4000.0, 300.0, 2.0)
    peak = atm.mass_elevation_effect(800.0, 10.0, 2.0)
    assert plateau > peak


# -- T19-T23: precipitation, clouds -------------------------------------------------------------


def test_t19_conservative_redistribution():
    f = rd.normalise_factors(np.array([2.0, 0.5]), np.array([0.25, 0.75]), np.zeros(2, int), 1)
    p = 100.0 * f
    assert p.tolist() == pytest.approx([228.571, 57.143], abs=1e-3)
    assert float((p * [0.25, 0.75]).sum()) == pytest.approx(100.0)
    assert (0.0 * f).tolist() == [0.0, 0.0]


def test_t20_windward_and_lee_follow_the_wind():
    p = np.array([0.2, -0.2])  # west slope rises to the east, east slope falls
    w = np.array([0.5, 0.5])
    g = np.zeros(2, dtype=int)
    for u, wet in ((10.0, 0), (-10.0, 1)):
        raw = rd.orographic_factor(u * p * 0.1, np.zeros(2))
        f = rd.normalise_factors(raw, w, g, 1, (0.15, 5.0))
        assert f[wet] > 1.0 > f[1 - wet]
        assert float((w * f).sum()) == pytest.approx(1.0, abs=1e-9)


def test_t21_capped_factors_keep_the_mass():
    raw = np.array([100.0, 1.0, 1.0, 1.0])
    w = np.full(4, 0.25)
    f = rd.normalise_factors(raw, w, np.zeros(4, int), 1, (0.25, 3.0))
    assert (f >= 0.25 - 1e-12).all() and (f <= 3.0 + 1e-12).all()
    assert float((w * f).sum()) == pytest.approx(1.0, abs=1e-9)
    with pytest.raises(ValueError):
        rd.normalise_factors(raw, w, np.zeros(4, int), 1, (1.5, 3.0))


def test_t22_lcl():
    assert float(atm.lcl_height_agl_m(20.0, 15.0)) == pytest.approx(625.0)
    assert 400.0 + float(atm.lcl_height_agl_m(20.0, 15.0)) == pytest.approx(1025.0)
    assert float(atm.lcl_height_agl_m(20.0, 20.0)) == 0.0
    td = atm.dew_point_c(20.0, 1.0)
    assert float(td) == pytest.approx(20.0, abs=1e-9)


def test_t23_dry_air_makes_no_clouds_or_fog():
    assert float(rd.cloud_potential(0.3, 2.0, 1500.0, 1500.0)) < 0.01
    assert float(rd.fog_potential(0.5, 1.0, 0.5)) == 0.0


# -- T24-T29: snow and water -------------------------------------------------------------------


def test_t24_phase_split_preserves_p():
    snow, rain = rd.phase_split(100.0, 0.0, snow_fraction=0.6)
    assert (float(snow), float(rain)) == pytest.approx((60.0, 40.0))
    s, r = rd.phase_split(np.array([0.0, 50.0]), np.array([-5.0, 0.5]))
    assert np.allclose(s + r, [0.0, 50.0])


def test_t25_melt_is_limited_by_the_snow():
    pot, act, left = rd.degree_day_melt(50.0, 4.0, 20.0)
    assert (float(pot), float(act), float(left)) == (80.0, 50.0, 0.0)


def test_t26_snow_redistribution_keeps_the_volume():
    out = rd.redistribute_snow(
        np.array([10.0, 10.0]), np.array([1.0, 3.0]), np.array([1.0, 3.0]), 0.5,
        np.zeros(2, int), 1,
    )  # fmt: skip
    assert out.tolist() == pytest.approx([7.0, 11.0])
    assert float((out * [1.0, 3.0]).sum()) == pytest.approx(40.0)


def test_t27_reference_bucket():
    r = rd.tz_bucket_step(20.0, 5.0, 0.0, 50.0, 0.0, 40.0, 100.0, 10.0, 0.1, 0.2)
    assert r["infiltration"] == 20.0 and r["surface_runoff"] == 30.0
    assert r["W"] == pytest.approx(27.0) and r["recharge"] == pytest.approx(3.0)
    assert r["GW"] == pytest.approx(6.4) and r["baseflow"] == pytest.approx(1.6)
    assert r["runoff"] == pytest.approx(31.6)
    assert 20 + 5 + 50 == pytest.approx(r["W"] + r["GW"] + r["aet"] + r["runoff"])


def _bucket(p, t):
    from hexworld.stages import environment as env

    n = 1
    wp = env.WaterParams()
    return env.water_balance(
        np.full((12, n), p), np.full((12, n), t), np.zeros((12, n)),
        np.full(n, 100.0), np.full(n, 20.0), np.full(n, 0.2), wp,
    )  # fmt: skip


def test_t28_growing_snow_is_flagged_not_hidden():
    wb = _bucket(100.0, -20.0)
    assert wb.persistent_snow.all() and wb.years <= env_max_years()
    assert wb.max_residual_mm < 1e-6


def test_t29_cold_and_dry_is_not_snowy():
    assert float(_bucket(0.0, -20.0).monthly["snowpack"].max()) == 0.0
    assert float(_bucket(80.0, -20.0).monthly["snowpack"].max()) > 0.0


# -- T30-T35: growing season, belts, crossings -------------------------------------------------


def _daily(warm_days, warm_t, cold_t=-5.0):
    d = np.full((365, 1), cold_t)
    d[100 : 100 + warm_days] = warm_t
    return d


def test_t30_season_of_100_days_at_7c():
    m = th.season_metrics(_daily(100, 7.0))
    assert m["gsl_days"][0] == 100 and m["gst_c"][0] == pytest.approx(7.0)
    margin = th.thermal_margin(m["gsl_days"], m["gst_c"])
    assert margin[0] > 0
    belt = th.thermal_belt(m["gsl_days"], m["gst_c"], margin, np.array([False]))
    assert th.BELTS[int(belt[0])] in ("thermal_upper_montane", "thermal_lower_montane")


def test_t31_short_or_cool_or_no_season():
    short = th.season_metrics(_daily(93, 10.0))
    cool = th.season_metrics(_daily(150, 5.0))
    none = th.season_metrics(np.full((365, 1), -10.0))
    for m in (short, cool):
        assert th.thermal_margin(m["gsl_days"], m["gst_c"])[0] < 0
    assert th.thermal_margin(short["gsl_days"], short["gst_c"])[0] == pytest.approx(-1 / 25)
    assert none["gsl_days"][0] == 0 and np.isnan(none["gst_c"][0])
    assert th.thermal_margin(none["gsl_days"], none["gst_c"])[0] == -40.0


def test_t32_southern_summer_is_one_season_across_the_year_end():
    t = np.full((12, 1), -10.0)
    t[[11, 0, 1]] = 10.0
    m = th.season_metrics(th.daily_from_monthly(t))
    assert m["longest_season_days"][0] == m["gsl_days"][0]
    assert 60 <= m["gsl_days"][0] <= 120


def test_t33_constant_7c_is_frost_free_only_by_proxy():
    m = th.season_metrics(np.full((365, 1), 7.0))
    assert m["gsl_days"][0] == 365
    margin = th.thermal_margin(m["gsl_days"], m["gst_c"])
    belt = th.thermal_belt(m["gsl_days"], m["gst_c"], margin, m["min_daily_c"] > 5.0)
    assert th.BELTS[int(belt[0])] == "thermal_frost_free"  # a thermal class, not a forest


def test_t34_inversion_gives_several_crossings():
    z = np.array([0.0, 500.0, 1000.0, 1500.0, 2000.0])
    margin = np.array([-1.0, 1.0, -0.5, 0.5, -1.0])
    assert len(th.crossings(z, margin)) == 4


def test_t35_zero_degree_level():
    assert float(th.zero_degree_level(13.0, 1000.0, 6.5)) == pytest.approx(3000.0)


def test_daily_curve_keeps_monthly_means():
    t = np.linspace(-10, 20, 12)[:, None]
    d = th.daily_from_monthly(t)
    month = np.repeat(np.arange(12), solar.MONTH_DAYS)
    means = np.array([d[month == m].mean() for m in range(12)])
    assert np.allclose(means, t[:, 0], atol=1e-9)


# -- T36-T42: the stage on a generated world -------------------------------------------------


def test_stage_on_a_generated_world(hist_project):
    p = hist_project
    t = p.topo
    assert t is not None and t.validation["ok"], t.validation
    s = t.summary
    assert s["area_closure_max_error"] < 1e-9
    assert s["precipitation_conservation_max_rel"] < 1e-6
    assert s["water_budget_max_residual_mm"] < 1e-6
    U = t.units
    dry = U["unit_surface_type"] == 0
    assert dry.sum() > 0
    land = t.layers["topo_dry_fraction"] > 0
    share = sum(v for k, v in t.layers.items() if k.startswith("belt_share_"))
    assert np.allclose(share[land], 1.0, atol=1e-4)
    # higher units of a hex are colder (same coarse T, one lapse rate, small anomalies)
    hot = U["temperature_c"][6, dry]
    assert np.corrcoef(U["z_surface_m"][dry], hot)[0, 1] < -0.5


def test_t36_chunking_does_not_change_the_result(hist_project, monkeypatch):
    p = hist_project
    a = run_topoclimate(p.config, p.geometry, p)
    import hexworld.pipeline_v47 as mod

    monkeypatch.setattr(mod, "_chunks", lambda off, target: _chunks(off, max(1, target // 7)))
    b = run_topoclimate(p.config, p.geometry, p)
    for key in ("temperature_c", "precip_mm", "swe_mm", "gsl_t_days", "thermal_belt"):
        assert np.allclose(a["units"][key], b["units"][key], equal_nan=True, atol=1e-5), key
    assert np.array_equal(a["units"]["unit_hex"], b["units"]["unit_hex"])


def test_t38_upstream_change_invalidates_v47(hist_config):
    up = {"env": {"v45_environment": "a"}}
    k1 = topo_stage_keys(hist_config, up)
    k2 = topo_stage_keys(hist_config, {"env": {"v45_environment": "b"}})
    k3 = topo_stage_keys(replace(hist_config, topoclimate=TopoclimateParams(tree_gst_c=7.0)), up)
    assert k1 != k2 and k1 != k3


def test_t39_save_load_and_reuse(tmp_path, hist_config, hist_project):
    path = save_project(hist_project, tmp_path / "w")
    loaded = load_project(path)
    assert loaded.topo is not None
    assert loaded.topo.content_hashes() == hist_project.topo.content_hashes()
    assert loaded.topo.validation["ok"]
    again = generate(hist_config, reuse=hist_project)
    assert "v47_topoclimate" in again.stats["reused_stages"]
    # a mixed hex keeps its units and classes after the round trip
    off = loaded.topo.units["hex_offsets"]
    h = int(np.argmax(np.diff(off)))
    col, row = h % loaded.layout.width, h // loaded.layout.width
    assert loaded.terrain_units(col, row) == hist_project.terrain_units(col, row)


def test_t40_threshold_changes_only_the_thermal_class(hist_project):
    p = hist_project
    cfg = replace(p.config, topoclimate=TopoclimateParams(tree_gst_c=8.0))
    b = run_topoclimate(cfg, p.geometry, p)
    a = p.topo.units
    assert np.array_equal(a["temperature_c"], b["units"]["temperature_c"])
    assert np.array_equal(a["swe_mm"], b["units"]["swe_mm"])
    assert not np.array_equal(
        np.nan_to_num(a["thermal_margin"]), np.nan_to_num(b["units"]["thermal_margin"])
    )
    from hexworld.views import VIEWS

    before = p.topo.content_hashes()
    for month in (0, 6):
        VIEWS["topoclimate"].render(p, "temperature", frozenset({f"month:{month}"}))
        VIEWS["belts"].render(p, "dominant", frozenset())
    assert p.topo.content_hashes() == before  # palette / month: display only


def test_t41_invalid_input_is_refused_not_guessed():
    with pytest.raises(ConfigError):
        Config(topoclimate=TopoclimateParams(quality="ultra"))
    with pytest.raises(ConfigError):
        Config(topoclimate=TopoclimateParams(precipitation_factor_bounds=(1.5, 3.0)))
    assert np.isnan(atm.dew_point_c(10.0, 0.0))
    m = th.season_metrics(np.full((365, 1), -30.0))
    assert th.thermal_margin(m["gsl_days"], m["gst_c"])[0] == -40.0  # no division by zero


def test_t41_all_sea_window_is_nodata(hist_project):
    from hexworld.pipeline_v47 import _empty_result

    p = hist_project
    shape = (p.layout.height, p.layout.width)
    r = _empty_result(p.config, shape, {}, 0.0, 6.5)
    assert (r["layers"]["topo_ocean_fraction"] == 1).all()
    assert np.isnan(r["layers"]["gsl_days_mean"]).all()
    assert r["units"]["unit_hex"].size == 0


def test_t42_sampling_convergence_is_reported(hist_project):
    p = hist_project
    lo = run_topoclimate(
        replace(p.config, topoclimate=TopoclimateParams(quality="preview")), p.geometry, p
    )
    hi = p.topo
    both = (lo["layers"]["topo_dry_fraction"] > 0) & (hi.layers["topo_dry_fraction"] > 0)
    d_frac = np.abs(lo["layers"]["topo_dry_fraction"] - hi.layers["topo_dry_fraction"])
    assert float(d_frac.mean()) < 0.03
    t_lo = lo["monthly"]["topo_temperature_c"][6][both]
    t_hi = hi.monthly["topo_temperature_c"][6][both]
    assert float(np.abs(t_lo - t_hi).mean()) < 0.5
    assert lo["summary"]["samples"] < hi.summary["samples"]


def test_section_has_climate_columns(hist_project, tmp_path):
    from hexworld.api import cross_section, export_cross_section_csv, query_environment

    p = hist_project
    b = p.layout.bounds
    rows = cross_section(p, (b.xmin + 100, b.ymin + 100), (b.xmax - 100, b.ymax - 100), month=6)
    land = [r for r in rows if r["material_type"] == "land"]
    assert land and all(r["quality"].startswith("tu_") for r in land)
    assert all(r["t_month_c"] is not None and r["belt"] for r in land)
    path = export_cross_section_csv(p, (b.xmin, b.ymin), (b.xmax, b.ymax), tmp_path / "s.csv")
    head = open(path, encoding="utf-8").readline()
    assert "gsl_days" in head and "thermal_margin" in head and "belt" in head
    r = land[len(land) // 2]
    q = query_environment(p, r["x_km"], r["y_km"])
    assert q["quality"].startswith("tu_")


def env_max_years():
    from hexworld.stages import environment as env

    return env.WaterParams().max_years
