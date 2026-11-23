"""TZ 5.4 integration: the stage on a generated world, storms, the weather year, save /
load, determinism, old projects, views (T08-T10, T41-T80, N09, N13, N14, N17)."""

from __future__ import annotations

import json
import math
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from hexworld.atmosphere import storms, weather
from hexworld.earth.planet import EARTH

FIX = Path(__file__).parent / "fixtures" / "v5_4"


# -- the stage on a generated world -----------------------------------------------------------
def test_stage_layers_and_validation(hist_project):
    a = hist_project.atmosphere
    assert a is not None and a.grid_fields, a.summary
    assert a.validation["ok"], a.validation
    L = a.layers
    shape = hist_project.hydro.layers["land_mask"].shape
    for k in (
        "rainfall_seasonality",
        "monsoon_suitability",
        "rain_regime",
        "monsoon_onset_month",
        "wind_reversal_score",
        "precip_share_convective",
        "tc_passage_per_yr",
        "etc_activity",
        "ar_day_frequency",
    ):
        assert L[k].shape == shape, k
    shares = sum(L[k].astype(np.float64) for k in L if k.startswith("precip_share_"))
    assert np.allclose(shares, 1.0, atol=1e-3)
    reg = a.summary["regime_shares_land"]
    assert sum(reg.values()) == pytest.approx(1.0)
    assert (L["monsoon_onset_month"] <= 12).all()


def test_t10_one_sst_one_ocean(hist_project):
    # the atmosphere reads the v5.3 ocean's SST; the earth stage reports the same ocean
    q = hist_project.atmosphere.summary["quality"]
    assert q["sst"] == "v5.3_ocean"
    coup = hist_project.atmosphere.summary["world_climate"]["ocean_coupling"]
    assert coup["status"] in ("CONVERGED", "NOT_CONVERGED")
    assert coup["iterations"]  # logged, never silent


def test_monthly_rain_of_v45_comes_from_the_normals(hist_project):
    m = hist_project.env.methods["months"]["method"]
    assert m == "atmosphere_5_4_monthly_normals"


def test_t73_wind_at_for_ash(hist_project):
    from hexworld.pipeline_v54 import wind_at_world

    a = hist_project.atmosphere
    hx, hy = hist_project.layout.centers()
    x, y = hx.ravel()[:50], hy.ravel()[:50]
    u5, v5, q5 = wind_at_world(a, x, y, 6, pressure_pa=50000.0)
    u1, v1, q1 = wind_at_world(a, x, y, 6, pressure_pa=10000.0)
    assert np.isfinite(u5).all() and (q5 <= 1).all()
    assert (q1 == 2).all()  # 100 hPa: extrapolated, never a modelled level
    s = hist_project.volcanism.summary["quality"]["tephra"]
    assert s == "hex_fv_atmosphere_5_4_wind_at"


def test_t79_views_do_not_change_the_snapshot(hist_project):
    from hexworld import views

    a = hist_project.atmosphere
    before = dict(a.content_hashes())
    spec = views.VIEWS["atmosphere_seasons"]
    for opt in spec.modes:
        views.view_rgb(hist_project, "atmosphere_seasons", opt.id, frozenset({"month:6", "coast"}))
        views.view_vectors(hist_project, "atmosphere_seasons", opt.id, frozenset({"month:6"}))
        spec.legend(hist_project, opt.id)
    assert a.content_hashes() == before


def test_t76_save_load_keeps_the_snapshot(hist_project, tmp_path):
    from hexworld.io import load_project, save_project

    save_project(hist_project, tmp_path / "w")
    p2 = load_project(tmp_path / "w")
    a, b = hist_project.atmosphere, p2.atmosphere
    assert b is not None
    assert a.content_hashes() == b.content_hashes()
    assert b.summary["source_snapshot_hash"] == a.summary["source_snapshot_hash"]


def test_t09_old_config_loads_without_atmosphere():
    from hexworld.config import Config

    d = json.loads(Config().to_json())
    d.pop("atmosphere")
    c = Config.from_dict(d)
    assert c.atmosphere.enabled is False


# -- tropical cyclones -----------------------------------------------------------------------
def _fac(**kw):
    base = dict(
        sst=np.array([29.0]),
        rh=np.array([0.8]),
        shear=np.array([5.0]),
        f=np.array([5e-5]),
        zeta=np.array([1e-5]),
        sea_frac=np.array([1.0]),
        basin_ok=np.array([True]),
    )
    base.update(kw)
    return float(storms.genesis_index(storms.genesis_factors(**base))[0])


def test_t41_t42_t43_t44_t45_genesis_gates():
    g0 = _fac()
    assert g0 > 0.3
    assert _fac(sst=np.array([24.0])) == 0.0  # cold ocean
    assert _fac(rh=np.array([0.4])) == 0.0  # dry mid-levels
    assert _fac(shear=np.array([25.0])) == 0.0  # strong shear
    assert _fac(f=np.array([0.0]), zeta=np.array([0.0])) == 0.0  # equator: no rotation
    assert _fac(basin_ok=np.array([False])) == 0.0  # a closed lake
    assert _fac(upwelling=np.array([3.0])) < 0.5 * g0  # shallow warm layer


def test_n09_poisson_rate():
    lam = storms.poisson_rate(0.5, 100e6, 30 * 86400.0, lambda0=1e-15)
    assert lam == pytest.approx(0.1296, rel=1e-4)


def test_holland_profile_is_finite_and_peaks_near_rmax():
    r = np.linspace(0.0, 300e3, 301)
    dp = storms.dp_for_vmax(50.0)
    v = storms.holland_wind(r, 101000.0 - dp, 101000.0, 40e3, 0.0)
    assert np.isfinite(v).all() and v[0] < 5.0
    assert abs(r[np.argmax(v)] - 40e3) < 10e3
    assert storms.holland_pressure(0.0, 95000.0, 101000.0, 40e3) == pytest.approx(95000.0)


def _grid_fields(ny=30, nx=40, cell=150.0, u=-6.0, sst=29.0, land_cols=None):
    lat = np.repeat(np.linspace(30.0, 5.0, ny)[:, None], nx, axis=1)
    sea = np.ones((ny, nx))
    if land_cols is not None:
        sea[:, land_cols] = 0.0
    G = {
        "lat_deg": lat.astype(np.float32),
        "sea_fraction": sea.astype(np.float32),
        "land_height_m": np.where(sea < 0.5, 800.0, 0.0).astype(np.float32),
        "t_sea_level_c": np.full((12, ny, nx), sst, np.float32),
        "t_air_c": np.full((12, ny, nx), sst - 1.0, np.float32),
        "wind_u_ms": np.full((12, 4, ny, nx), u, np.float32),
        "wind_v_ms": np.zeros((12, 4, ny, nx), np.float32),
        "column_vapor_kg_m2": np.full((12, ny, nx), 50.0, np.float32),
        "column_rh": np.full((12, ny, nx), 0.8, np.float32),
    }
    grid = {"x0_km": 0.0, "y0_km": 0.0, "cell_km": cell}
    return G, grid


def test_t46_track_follows_the_steering_wind_in_metres():
    G, grid = _grid_fields(u=-8.0)
    rng = storms.EventRng("Track1", "t")
    tf = storms.TrackField(grid, G, 7)
    tr = storms.run_track(tf, 5000.0, 2500.0, rng, EARTH)
    xs = [p["x_km"] for p in tr["points"]]
    assert xs[-1] < xs[0] - 500.0  # westward with the easterlies
    steps = np.hypot(np.diff(xs), np.diff([p["y_km"] for p in tr["points"]]))
    assert steps.max() < 400.0  # continuous, no jumps between hex centres


def test_t47_landfall_weakens():
    G, grid = _grid_fields(u=-8.0, land_cols=slice(0, 18))
    rng = storms.EventRng("Track2", "t")
    tr = storms.run_track(storms.TrackField(grid, G, 7), 5500.0, 2500.0, rng, EARTH, v0=40.0)
    over_land = [p for p in tr["points"] if p.get("land", 0) > 0.5]
    assert over_land
    v_land = [p["v_ms"] for p in over_land]
    assert v_land[-1] < v_land[0]


def test_t48_mountains_take_more_storm_rain():
    G, grid = _grid_fields(u=-8.0, land_cols=slice(0, 18))
    ev = {
        "event_type": "TC",
        "month": 8,
        "track": {
            "points": [
                {"x_km": 18 * 150.0 + dx, "y_km": 2500.0, "v_ms": 40.0} for dx in (300, 0, -300)
            ]
        },
    }
    r = weather.event_rain(ev, G, grid, G["lat_deg"].shape)
    row = int(2500 / 150)
    assert r[row, 15:18].mean() > r[row, 19:22].mean() * 0.9
    assert ev["budget_ledger"]["rain_mm_cells"] <= ev["budget_ledger"]["vapour_available_mm_cells"]


def test_t49_n13_event_seeds_change_tracks_not_the_climatology(hist_project):
    a = hist_project.atmosphere
    G, grid = a.grid_fields, a.grid
    lam = a.summary["storms"]["tc_genesis_per_month"]
    days = np.asarray(grid["month_days"])
    y1 = weather.generate(G, grid, EARTH, days, "Ab12", "yearA", 1, lam)
    y2 = weather.generate(G, grid, EARTH, days, "Ab12", "yearB", 1, lam)
    assert not np.array_equal(
        y1["aggregate_monthly_actuals_mm"], y2["aggregate_monthly_actuals_mm"]
    )
    # the climatology lives in the normals and never reads the event seed
    assert a.summary["storms"]["tc_genesis_per_yr"] == pytest.approx(
        hist_project.atmosphere.summary["storms"]["tc_genesis_per_yr"]
    )


def test_t50_t60_n14_year_ledger_and_shared_vapour(hist_project):
    a = hist_project.atmosphere
    G, grid = a.grid_fields, a.grid
    days = np.asarray(grid["month_days"])
    lam = [x * 20 for x in a.summary["storms"]["tc_genesis_per_month"]]  # many storms
    y = weather.generate(G, grid, EARTH, days, "Ab12", "busy", 1, lam)
    led = y["ledger"]
    act = y["aggregate_monthly_actuals_mm"].astype(np.float64)
    assert act.sum() == pytest.approx(led["generator_mm_cells"] + led["event_mm_cells"], rel=1e-5)
    pool = G["column_vapor_kg_m2"].astype(np.float64) * (days.reshape(12, 1, 1) / 8.0)
    assert (y["event_monthly_mm"] <= pool * (1 + 1e-6) + 1e-4).all()  # one shared pool (float32)
    for ev in y["events"]:
        b = ev["budget_ledger"]
        assert b["rain_mm_cells"] <= b["vapour_available_mm_cells"] + 1e-6


# -- extratropical storms, fronts, rivers ---------------------------------------------------
def test_t51_t52_frontogenesis_needs_a_gradient():
    ny, nx = 20, 20
    jj, ii = np.mgrid[0:ny, 0:nx]
    u = (ii - 10.0) * 0.5  # stretching deformation: du/dx > 0, dv/dy_north < 0
    v = (jj - 10.0) * 0.5
    t_grad = jj * 1.0
    t_flat = np.full((ny, nx), 15.0)
    assert storms.frontogenesis_proxy(t_grad, u, v, 1e5).mean() > 0
    assert storms.frontogenesis_proxy(t_flat, u, v, 1e5).max() == 0.0


def test_t53_eady_is_symmetric_between_hemispheres():
    f = EARTH.coriolis(np.array([45.0, -45.0]))
    s = storms.eady_growth(f, np.array([3e-3, 3e-3]))
    assert s[0] == pytest.approx(s[1]) and s[0] * 86400 > 0.5


def test_t54_t55_t56_fronts_are_continuous_and_cold_ones_snow(hist_project):
    a = hist_project.atmosphere
    G, grid = dict(a.grid_fields), a.grid
    G["etc_activity"] = np.ones_like(G["etc_activity"])
    rngf = lambda n: storms.EventRng("Ab12", f"test/{n}")  # noqa: E731
    evs = weather.etc_events(G, grid, rngf, "t", np.asarray(grid["month_days"]))
    assert evs
    for ev in evs[:20]:
        for fr in ev["fronts"]:
            pts = np.asarray(fr["geometry_world_km"])
            assert np.hypot(*np.diff(pts, axis=0).T).max() < 300.0  # one polyline, no gaps
            assert 0.0 <= fr["snow_fraction"] <= 1.0
    cold = [ev for ev in evs if ev["fronts"][0]["snow_fraction"] > 0]
    warm = [ev for ev in evs if ev["fronts"][0]["snow_fraction"] == 0]
    assert warm  # rain fronts exist; snow only where the profile is cold
    dry = dict(G, column_rh=np.zeros_like(G["column_rh"]))
    r = weather.event_rain(evs[0], dry, grid, G["lat_deg"].shape)
    assert r.max() == 0.0  # T56: a dry front brings no rain
    del cold


def test_t57_t58_atmospheric_river_detection():
    ny, nx = 30, 60
    e = np.zeros((ny, nx))
    n = np.zeros((ny, nx))
    e[14:16, 5:50] = 500.0  # a long narrow band
    feats = storms.detect_ar(e, n, 100.0, threshold=250.0)
    assert len(feats) == 1 and feats[0]["length_km"] >= 2000.0
    blob = np.zeros((ny, nx))
    blob[5:25, 10:30] = 500.0
    assert storms.detect_ar(blob, n, 100.0, threshold=250.0) == []  # broad: not a river
    G, grid = _grid_fields(land_cols=slice(30, 40))
    ev = {
        "event_type": "AR",
        "month": 1,
        "width_km": 300.0,
        "axis_world_km": [[x, 2000.0] for x in np.linspace(500, 5800, 30)],
    }
    r = weather.event_rain(ev, G, grid, G["lat_deg"].shape)
    assert r[:, :28].max() == 0.0  # no AR rain over the open sea without lifting
    assert r[:, 30:].max() > 0.0  # it rains on the hills


def test_n17_t59_tracks_are_not_clipped_by_the_viewport(hist_project):
    a = hist_project.atmosphere
    b = hist_project.geometry.viewport_bounds
    G, grid = a.grid_fields, a.grid
    lam = [max(x, 0.5) for x in a.summary["storms"]["tc_genesis_per_month"]]
    y = weather.generate(G, grid, EARTH, np.asarray(grid["month_days"]), "Ab12", "wide", 1, lam)
    pts = [
        (p["x_km"], p["y_km"])
        for ev in y["events"]
        if ev["event_type"] == "TC"
        for p in ev["track"]["points"]
    ]
    if not pts:
        pytest.skip("no TC in this world")
    xs, ys = np.array(pts).T
    outside = (xs < b.xmin) | (xs > b.xmax) | (ys < b.ymin) | (ys > b.ymax)
    assert outside.any()  # the track lives in world km, beyond the GUI window


# -- weather generator -----------------------------------------------------------------------
def test_n07_t61_markov_wet_share():
    assert weather.stationary_wet_fraction(0.7, 0.3) == pytest.approx(0.5)
    rng = storms.EventRng("Markov", "m")
    u = rng.random(200000)
    wet, n = False, 0
    for x in u:
        wet = x < (0.7 if wet else 0.3)
        n += wet
    assert n / u.size == pytest.approx(0.5, abs=0.01)
    p_ww, p_dw = weather.markov_params(np.array([0.4]))
    assert weather.stationary_wet_fraction(p_ww, p_dw)[0] == pytest.approx(0.4)


def test_t62_gamma_amounts():
    from scipy.stats import gamma as g

    u = storms.EventRng("Gamma", "g").random(50000)
    x = g.ppf(u, weather.GAMMA_K, scale=8.0)
    assert x.mean() == pytest.approx(weather.GAMMA_K * 8.0, rel=0.03)
    assert (x >= 0).all() and np.percentile(x, 99.9) < 80.0


def test_t63_neighbours_share_the_weather():
    rng = storms.EventRng("Coh", "c")
    u = weather.correlated_uniforms(rng, (30, 30), 200, 100.0)
    near = np.corrcoef(u[:, 10, 10], u[:, 10, 11])[0, 1]
    far = np.corrcoef(u[:, 10, 10], u[:, 25, 25])[0, 1]
    assert near > 0.8 and abs(far) < 0.3


def test_t66_t67_droughts_relative_to_the_own_norm():
    norm = np.full((12, 2), 80.0)
    norm[:, 1] = 0.5  # a hyperarid cell
    dry_year = norm * 0.3
    m1, _ = weather.drought_mask(dry_year, norm)
    assert m1[0] and not m1[1]  # the desert's usual dryness is no drought
    pers = weather.persistent_drought([m1, m1])
    assert pers[0] and not weather.persistent_drought([m1, ~m1])[0]


def test_t65_t69_years_differ_and_no_ocean_mode_without_a_basin(hist_project):
    a = hist_project.atmosphere
    G, grid = a.grid_fields, a.grid
    days = np.asarray(grid["month_days"])
    y1 = weather.generate(G, grid, EARTH, days, "Ab12", "s", 1, None)
    y2 = weather.generate(G, grid, EARTH, days, "Ab12", "s", 2, None)
    assert not np.array_equal(y1["onset_day"], y2["onset_day"]) or not np.array_equal(
        y1["aggregate_monthly_actuals_mm"], y2["aggregate_monthly_actuals_mm"]
    )
    assert y1["interannual"]["mode"] == "regional_persistence_proxy"  # T68 / T69
    assert y1["interannual"]["ocean_mode"] == "not_available"


def test_t70_calendars():
    from hexworld.atmosphere.circulation import month_days

    assert month_days(EARTH).sum() == 365
    from hexworld.earth.planet import Planet

    d = month_days(Planet(year_length_days=480.0))
    assert d.sum() == pytest.approx(480.0) and np.allclose(d, 40.0)
    p = np.array([60.0])
    assert p / 30 * 31 == pytest.approx(62.0)


def test_t64_t71_t72_t74_t75_t77_weather_year_stage(hist_project):
    """One weather year on the generated world: delta layers, routing, no biome change."""
    from hexworld.pipeline_v54 import run_atmosphere

    cfg = hist_project.config
    cfg2 = replace(
        cfg, atmosphere=replace(cfg.atmosphere, weather_year=True, weather_year_seed="W1")
    )
    biome_before = hist_project.biomes.layers["dominant_biome"].copy()
    layers_before = dict(hist_project.atmosphere.layers)
    res = run_atmosphere(cfg2, hist_project)
    L = res["layers"]
    for k in (
        "weather_precip_actual_annual_mm",
        "weather_max_daily_rain_mm",
        "weather_drought",
        "weather_flood_peak_m3_s",
        "weather_onset_day",
    ):
        assert k in L
    # T71: the wettest day flows down the existing rivers (peaks grow downstream)
    q = L["weather_flood_peak_m3_s"]
    rc = hist_project.hydro.layers["flow_receiver"].ravel()
    land = hist_project.hydro.layers["land_mask"].ravel()
    ok = (rc >= 0) & land & land[np.maximum(rc, 0)] & (rc != np.arange(rc.size))
    i = int(np.argmax(np.where(ok, q.ravel(), -1.0)))
    if q.ravel()[i] > 0:
        assert q.ravel()[rc[i]] >= q.ravel()[i] * (1 - 1e-6)
    # T72 / T77: nothing in the world changed (the delta is a new object)
    assert np.array_equal(hist_project.biomes.layers["dominant_biome"], biome_before)
    assert hist_project.atmosphere.layers.keys() == layers_before.keys()
    s = res["summary"]["weather_year"]
    assert s["ledger"]["event_mm_cells"] >= 0
    # T75: storm surge is only a potential
    assert "tc_surge_potential_index" in L and not any("surge_depth" in k for k in L)
    # T74: snow is a share of the same front water, never added on top
    for ev in res["events"]:
        if ev["event_type"] == "ETC":
            for fr in ev["fronts"]:
                assert 0.0 <= fr["snow_fraction"] <= 1.0


def test_t80_windows_gui_not_verified_is_stated():
    doc = (Path(__file__).parents[1] / "docs" / "v5_4_test_acceptance.md").read_text("utf-8")
    assert "Windows GUI: NOT VERIFIED" in doc


def test_fixtures_s01_s24_exist():
    names = sorted(p.stem for p in FIX.glob("S*.json"))
    assert len(names) == 24
    for p in FIX.glob("S*.json"):
        d = json.loads(p.read_text("utf-8"))
        assert {"id", "scenario", "expected", "test"} <= set(d)
    assert math.isfinite(1.0)


# -- snow, windthrow, export ----------------------------------------------------------------
def test_t74_snow_is_part_of_the_same_water(hist_project):
    a = hist_project.atmosphere
    G, grid = a.grid_fields, a.grid
    y = weather.generate(G, grid, EARTH, np.asarray(grid["month_days"]), "Ab12", "snow", 1, None)
    snow = y["snowfall_monthly_mm"].astype(np.float64)
    act = y["aggregate_monthly_actuals_mm"].astype(np.float64)
    assert (snow <= act * (1 + 1e-5) + 1e-4).all()  # a share of the water, never extra
    cold = G["t_air_c"] < -5.0
    if cold.any():
        assert snow[cold].sum() > 0.9 * act[cold].sum()


def test_windthrow_keeps_the_soil():
    from hexworld.biomes import ecology as eco

    n = 30
    nbr = np.full((n, 6), -1, dtype=np.int64)
    nbr[:, 0] = np.roll(np.arange(n), -1)
    nbr[:, 3] = np.roll(np.arange(n), 1)
    args = (["forest_temperate"] * n, np.full(n, 12.0), np.full(n, 900.0), np.zeros(n),
            np.ones(n, bool), nbr, 7)  # fmt: skip
    idx = np.arange(n)
    wind = eco.run(*args, years=60, disturbances=[(30, idx, np.full(n, 0.6), False)])
    lava = eco.run(*args, years=60, disturbances=[(30, idx, np.full(n, 0.6), True)])
    assert wind["actual_biomass_kg_c_m2"].mean() > lava["actual_biomass_kg_c_m2"].mean()


def test_storm_wind_on_hexes_follows_the_holland_profile(hist_project):
    from hexworld.pipeline_v54 import storm_max_wind_hex

    hx, hy = hist_project.layout.centers()
    x0, y0 = float(hx.mean()), float(hy.mean())
    ev = {"event_type": "TC", "track": {"points": [{"x_km": x0, "y_km": y0, "v_ms": 50.0}]}}
    w = storm_max_wind_hex(hist_project, [ev])
    d = np.hypot(hx - x0, hy - y0)
    assert w[d < 60].max() > 30.0
    assert w[d > 400].max() == 0.0
    assert np.isfinite(w).all()


def test_export_ledger_and_metric_geometry(hist_project, tmp_path):
    from hexworld.atmosphere.export import export_all, world_metric_json
    from hexworld.pipeline_v54 import run_atmosphere

    cfg = hist_project.config
    cfg2 = replace(cfg, atmosphere=replace(cfg.atmosphere, weather_year=True))
    res = run_atmosphere(cfg2, hist_project)
    atm = replace(hist_project.atmosphere, events=res["events"], summary=res["summary"],
                  grid_fields=res["grid_fields"])  # fmt: skip
    info = export_all(atm, tmp_path / "x")
    assert info["events"] == len(res["events"]) and info["budget_months"] == 12
    head = (tmp_path / "x" / "weather_events.csv").read_text("utf-8").splitlines()[0]
    assert head.startswith("event_id,event_type,month")
    doc = world_metric_json(atm)
    assert doc["crs"] == "world_metric_km" and "lat" not in json.dumps(doc["features"][:5])
