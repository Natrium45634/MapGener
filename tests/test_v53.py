"""v5.3a Earth system: planet, winds aloft, FAO-56, ocean, sea ice (TZ 5.3; ids T.. of sec. 21).

Synthetic geographies are built on small rasters so every test isolates one mechanism;
an end-to-end world (``hist_project``) checks the stage, save/load and the views.
"""

from __future__ import annotations

import math
from dataclasses import replace

import numpy as np
import pytest

from hexworld.config import Config, ConfigError, EarthParams
from hexworld.earth import ocean as oc
from hexworld.earth.pet import fao56_daily, fao56_monthly, hargreaves_monthly, slope_kpa_k
from hexworld.earth.planet import EARTH, MONTH_FRACTION, Planet, seasonal_amplitude_ratio
from hexworld.earth.winds import thermal_wind, wind_at_height


def _lat(ny, nx, south=10.0, north=60.0):
    rows = np.linspace(north, south, ny)  # raster rows grow southward
    return np.repeat(rows[:, None], nx, axis=1)


def _westerlies_trades(lat):
    """Idealised zonal wind: trades (east) below 30 deg, westerlies above (raster v=0)."""
    u = -6.0 * np.cos(np.radians(lat) * 3.0)
    return u, np.zeros_like(u)


# -- numeric controls (sec. 20) -------------------------------------------------------------
def test_n_coriolis_and_double_day():
    assert EARTH.coriolis(45.0) == pytest.approx(1.03e-4, rel=0.01)
    assert Planet(day_length_h=48.0).coriolis(45.0) == pytest.approx(EARTH.coriolis(45.0) / 2)


def test_n_wind_stress():
    tau = oc.RHO_A * 0.0015 * 10.0 * 10.0
    assert tau == pytest.approx(0.18)


def test_n_mixed_layer_heat():
    assert 1025 * 4000 * 50 * 1.0 == pytest.approx(205e6)
    dt = 100.0 * 30 * 86400 / (1025 * 4000 * 50)
    assert dt == pytest.approx(1.264, abs=1e-3)


def test_n_sea_ice_melt_energy_and_albedo():
    assert 917 * 334000 * 1.0 == pytest.approx(306.3e6, rel=1e-3)
    c = 0.75
    assert (1 - c) * 0.06 + c * 0.60 == pytest.approx(0.465)


# -- T69-T72 planet ----------------------------------------------------------------------------
def test_t69_zero_tilt_no_seasonal_declination():
    p = Planet(obliquity_deg=0.0)
    assert np.allclose(p.declination(MONTH_FRACTION), 0.0)
    ra = [p.ra_mj(50.0, f) for f in MONTH_FRACTION]
    assert max(ra) - min(ra) < 1e-9


def test_t70_hemispheres_in_antiphase():
    s_n = [EARTH.insolation(45.0, f) for f in MONTH_FRACTION]
    s_s = [EARTH.insolation(-45.0, f) for f in MONTH_FRACTION]
    assert int(np.argmax(s_n)) in (5, 6) and int(np.argmax(s_s)) in (11, 0)


def test_t71_polar_day_and_night():
    p = Planet(obliquity_deg=60.0)
    assert p.insolation(70.0, 0.75) == 0.0  # northern winter solstice: polar night
    assert p.daylight_hours(70.0, 0.25) == pytest.approx(24.0)  # polar day


def test_t72_double_day_halves_omega():
    assert Planet(day_length_h=48.0).omega == pytest.approx(EARTH.omega / 2.0)


def test_planet_ra_matches_fao_within_eccentricity():
    from hexworld.stages.environment import extraterrestrial_radiation

    for lat in (0.0, 45.0, 70.0):
        a = EARTH.ra_mj(lat, MONTH_FRACTION[6])
        b = float(extraterrestrial_radiation(lat, 196)[0])
        assert a == pytest.approx(b, rel=0.05)


def test_tilt_scales_seasonality():
    r0 = seasonal_amplitude_ratio(Planet(obliquity_deg=0.0), np.array([45.0]))
    r60 = seasonal_amplitude_ratio(Planet(obliquity_deg=60.0), np.array([45.0]))
    assert r0[0] < 0.1 and r60[0] > 1.5
    assert seasonal_amplitude_ratio(EARTH, np.array([45.0]))[0] == pytest.approx(1.0)


def test_config_validation_and_legacy_default():
    with pytest.raises(ConfigError):
        Config(earth=EarthParams(obliquity_deg=120.0))
    with pytest.raises(ConfigError):
        Config(earth=EarthParams(day_length_h=0.0))
    d = Config().to_dict()
    d.pop("earth")
    assert Config.from_dict(d).earth.enabled is False  # a pre-5.3 save stays as it was
    assert Config.from_dict(Config().to_dict()).earth.enabled is True


# -- T25-T29 winds aloft ------------------------------------------------------------------------
def _thermal(lat, t_offset=0.0, planet=EARTH):
    t = 28.0 - 0.55 * np.abs(lat) + t_offset
    u, v = _westerlies_trades(lat)
    return thermal_wind([t] * 4, [u] * 4, [v] * 4, lat, 100.0, planet)


def test_t25_equator_has_finite_winds():
    lat = _lat(41, 10, -20.0, 20.0)
    w = _thermal(lat)
    assert np.isfinite(w.u).all() and np.abs(w.u).max() < 100.0


def test_t26_upper_wind_differs_from_surface_with_a_gradient():
    lat = _lat(40, 10, 10.0, 70.0)
    w = _thermal(lat)
    mid = np.abs(lat - 45.0) < 3.0
    u_sfc = _westerlies_trades(lat)[0]
    assert (w.u[0, 2][mid] - u_sfc[mid]).mean() > 10.0  # westerly jet aloft
    assert (w.u[0, 1][mid] < w.u[0, 2][mid]).all()


def test_t26_southern_hemisphere_also_westerly_aloft():
    lat = _lat(40, 10, -70.0, -10.0)
    w = _thermal(lat)
    mid = np.abs(lat + 45.0) < 3.0
    assert (w.u[0, 2][mid] - _westerlies_trades(lat)[0][mid]).mean() > 10.0


def test_t27_levels_and_below_ground():
    lat = _lat(20, 10, 30.0, 50.0)
    w = _thermal(lat)
    z = w.z[0, :, 5, 5]
    assert 1000 < z[0] < 2000 and 5000 < z[1] < 6500 and 9500 < z[2] < 11500
    assert (z < 4000.0).tolist() == [True, False, False]  # 850 hPa under a 4 km summit


def test_wind_at_height_interpolates_the_column():
    lat = _lat(20, 10, 30.0, 50.0)
    w = _thermal(lat)
    us, vs = _westerlies_trades(lat)
    u_lo, _ = wind_at_height(w.u[0], w.v[0], w.z[0], us, vs, 0.0)
    u_hi, _ = wind_at_height(w.u[0], w.v[0], w.z[0], us, vs, 20000.0)
    assert np.allclose(u_lo, us)
    assert np.allclose(u_hi, w.u[0, 2])


# -- T30-T32 evapotranspiration -----------------------------------------------------------------
def test_t30_fao56_example_18():
    """FAO-56 Example 18 (Brussels, 6 July): ET0 = 3.9 mm/day."""
    et0 = fao56_daily(16.9, 21.5, 12.3, 1.409, 2.078, 13.28, 0.0, 100.0)
    assert et0 == pytest.approx(3.9, abs=0.05)
    assert slope_kpa_k(16.9) == pytest.approx(0.122, abs=0.001)


def test_t31_pet_rises_with_radiation_vpd_and_wind():
    base = fao56_daily(20.0, 26.0, 14.0, 1.2, 2.0, 12.0, 0.0, 0.0)
    assert fao56_daily(20.0, 26.0, 14.0, 1.2, 2.0, 16.0, 0.0, 0.0) > base  # radiation
    assert fao56_daily(20.0, 26.0, 14.0, 0.8, 2.0, 12.0, 0.0, 0.0) > base  # drier air
    assert fao56_daily(20.0, 26.0, 14.0, 1.2, 4.0, 12.0, 0.0, 0.0) > base  # wind


def test_t31_dry_windy_valley_beats_cloudy_calm_place():
    t = np.full((12, 2), 22.0)
    p = np.array([[5.0, 220.0]] * 12)
    e, _ = fao56_monthly(t, p, np.array([6.0, 1.0]), [25.0, 25.0], [0.0, 0.0], EARTH)
    assert e[:, 0].sum() > 1.3 * e[:, 1].sum()


def test_t32_pet_non_negative_and_polar_night():
    e, x = fao56_monthly(np.full((12, 1), -25.0), np.full((12, 1), 5.0), [3.0], [85.0], [0.0],
                         EARTH)  # fmt: skip
    assert (e >= 0).all() and np.isfinite(e).all()
    assert x["ra"][0, 0] == 0.0  # January at 85 N: no sun
    h, _ = hargreaves_monthly(np.full((12, 1), 10.0), np.full((12, 1), 50.0), [45.0], EARTH)
    assert (h >= 0).all()


# -- T09-T16 ocean ------------------------------------------------------------------------------
def _basin(ny=40, nx=50, south=10.0, north=60.0, land_cols=3):
    lat = _lat(ny, nx, south, north)
    sea = np.ones((ny, nx), dtype=bool)
    sea[:, :land_cols] = False
    sea[:, -land_cols:] = False
    sea[:2] = False
    sea[-2:] = False
    return lat, sea


def _run(lat, sea, winds=None, years=3, params=None, cell_km=150.0):
    u, v = _westerlies_trades(lat) if winds is None else winds
    return oc.run_ocean(EARTH, lat, sea, [(u, v)] * 4, cell_km, params or EarthParams(),
                        years=years)  # fmt: skip


def test_t09_t11_closed_basin_streamfunction_divergence_free():
    lat, sea = _basin()
    ocean, _ = oc.ocean_mask(sea)
    flip = ocean[::-1]
    circ = oc.Circulation(flip, lat[::-1], 150e3, EARTH)
    u, v = _westerlies_trades(lat)
    tx = oc.RHO_A * 1.3e-3 * np.abs(u[::-1]) * u[::-1]
    psi = circ.solve(tx, np.zeros_like(tx))
    fx, fy = oc.Circulation.face_transport(psi)
    div = oc._divergence(fx, fy)
    assert np.abs(div[flip]).max() < 1e-6 * np.abs(psi).max()
    # no flux through any face that touches land (T09)
    land_e = ~(flip[:, :-1] & flip[:, 1:])
    assert np.abs(fx[:, :-1][land_e]).max() == 0.0
    land_n = ~(flip[:-1] & flip[1:])
    assert np.abs(fy[:-1][land_n]).max() == 0.0


def test_t10_face_transport_shared_by_both_cells():
    psi = np.random.default_rng(1).normal(size=(6, 7))
    fx, fy = oc.Circulation.face_transport(psi)
    # one value per face: what leaves cell (j, i) eastward enters (j, i+1) from the west
    div = oc._divergence(fx, fy)
    assert abs(div.sum() - (fx[:, -1].sum() + fy[-1].sum())) < 1e-9


def test_subtropical_gyre_has_a_western_boundary_current():
    lat, sea = _basin(south=10.0, north=50.0)
    r = _run(lat, sea)
    band = (np.abs(lat - 30.0) < 4.0) & r.ocean
    v_north = -r.current_v[0]  # raster v is + south
    cols = np.nonzero(band.any(axis=0))[0]
    west, east = cols[:4], cols[-4:]
    assert v_north[band][:, None].size
    assert v_north[:, west][band[:, west]].mean() > 0.0  # poleward on the western side
    assert v_north[:, west][band[:, west]].mean() > abs(v_north[:, east][band[:, east]].mean())


def test_t12_disconnected_basins_do_not_share_heat():
    lat, sea = _basin()
    sea[:, 24:27] = False  # a continent splits the ocean in two
    sea[:, 10:12] = sea[:, 10:12]  # keep both halves large
    ocean, basin = oc.ocean_mask(sea)
    assert len(set(basin[ocean].tolist())) == 2
    r = _run(lat, sea)
    # the operator has no link between the basins: a heat tracer cannot cross
    flip = r.ocean[::-1]
    idx = -np.ones(flip.shape, dtype=np.int64)
    idx[flip] = np.arange(int(flip.sum()))
    fx = np.ones(flip.shape) * 1e6
    fy = np.ones(flip.shape) * 1e6
    a = oc.heat_operator(fx, fy, flip, idx, np.full(int(flip.sum()), 1e12), 1e4, 150e3, 50.0)
    b = basin[::-1][flip]
    rows, cols = a.nonzero()
    assert (b[rows] == b[cols]).all()


def test_t13_strait_carries_flux_only_through_sea_cells():
    lat, sea = _basin()
    sea[:, 24:27] = False
    sea[20:22, 24:27] = True  # a two-cell strait
    r = _run(lat, sea)
    land = ~r.ocean
    assert np.abs(r.current_u[:, land]).max() == 0.0
    assert np.abs(r.current_v[:, land]).max() == 0.0


def test_t14_advection_conserves_heat_without_forcing():
    lat, sea = _basin()
    flip = oc.ocean_mask(sea)[0][::-1]
    n = int(flip.sum())
    idx = -np.ones(flip.shape, dtype=np.int64)
    idx[flip] = np.arange(n)
    rng = np.random.default_rng(3)
    psi = rng.normal(size=(flip.shape[0] + 1, flip.shape[1] + 1)) * 1e6
    free = np.zeros_like(psi, dtype=bool)
    free[1:-1, 1:-1] = flip[:-1, :-1] & flip[:-1, 1:] & flip[1:, :-1] & flip[1:, 1:]
    psi[~free] = 0.0
    fx, fy = oc.Circulation.face_transport(psi)
    vol = np.full(n, 150e3**2 * 50.0)
    a = oc.heat_operator(fx, fy, flip, idx, vol, 2e4, 150e3, 50.0)
    t = rng.normal(size=n) * 5 + 15
    t2 = t.copy()
    for _ in range(50):
        t2 = t2 + 3600.0 * (a @ t2)
    assert (vol * t2).sum() == pytest.approx((vol * t).sum(), rel=1e-12)


def test_t15_sst_bounded_and_heat_budget_closed():
    lat, sea = _basin(south=-60.0, north=60.0)
    r = _run(lat, sea, years=3)
    sst = r.sst_monthly[:, r.ocean]
    assert np.isfinite(sst).all() and sst.min() >= oc.T_FREEZE - 1e-6 and sst.max() < 35
    assert r.diagnostics["heat_relative_residual"] < 1e-9


def test_t16_upwelling_on_an_eastern_boundary_with_equatorward_wind():
    ny, nx = 30, 30
    lat = _lat(ny, nx, 20.0, 40.0)
    sea = np.ones((ny, nx), dtype=bool)
    sea[:, -4:] = False  # land to the east: an eastern boundary (California-like)
    u = np.zeros((ny, nx))
    v = np.full((ny, nx), 7.0)  # raster + south: equatorward along the coast
    r = oc.run_ocean(EARTH, lat, sea, [(u, v)] * 4, 100.0, EarthParams(), years=2)
    coast_col = nx - 5
    assert (r.upwelling_m_day[0][5:-5, coast_col] > 0).all()
    # the reverse wind gives downwelling there
    r2 = oc.run_ocean(EARTH, lat, sea, [(u, -v)] * 4, 100.0, EarthParams(), years=2)
    assert (r2.upwelling_m_day[0][5:-5, coast_col] < 0).all()


# -- T17-T24 sea ice ---------------------------------------------------------------------------
def _polar(years=3, **kw):
    lat, sea = _basin(south=50.0, north=88.0)
    return lat, _run(lat, sea, years=years, params=EarthParams(**kw))


def test_t17_t18_polar_ocean_freezes_in_winter_and_retreats_in_summer():
    lat, r = _polar()
    north = (lat > 75) & r.ocean
    conc = r.sea_ice_monthly[:, north].mean(axis=1)
    assert conc[1] > 0.5  # February
    assert conc[8] < conc[1]  # September retreat
    assert (r.ice_thickness_monthly >= 0).all()


def test_t19_ice_fields_valid():
    _lat_, r = _polar()
    assert np.isfinite(r.sea_ice_monthly).all()
    assert r.sea_ice_monthly.min() >= 0 and r.sea_ice_monthly.max() <= 1.0


def test_t20_ice_drift_conserves_volume():
    lat, sea = _basin(south=60.0, north=85.0)
    flip = oc.ocean_mask(sea)[0][::-1]
    n = int(flip.sum())
    idx = -np.ones(flip.shape, dtype=np.int64)
    idx[flip] = np.arange(n)
    u = np.full(flip.shape, 0.3)
    v = np.full(flip.shape, -0.2)
    m, nsub = oc.ice_transport_operator(u, v, flip, idx, 150e3, oc.SECONDS_PER_MONTH)
    h = np.random.default_rng(2).uniform(0, 2, n)
    h2 = h.copy()
    for _ in range(nsub):
        h2 = m @ h2
    assert h2.sum() == pytest.approx(h.sum(), rel=1e-12)
    assert h2.min() >= 0.0


def test_t21_albedo_feedback_changes_absorbed_sunlight():
    _lat1, with_ice = _polar(sea_ice=True)
    _lat2, no_ice = _polar(sea_ice=False)
    m = (_lat1 > 75) & with_ice.ocean
    # without ice the summer water absorbs more sunlight: it is warmer in August
    assert no_ice.sst_monthly[7][m].mean() > with_ice.sst_monthly[7][m].mean()


def test_t22_warm_ocean_has_no_sea_ice():
    lat, sea = _basin(south=0.0, north=40.0)
    r = _run(lat, sea)
    assert r.sea_ice_monthly.max() == 0.0


def test_t24_types_and_realms_distinguished():
    lat, r = _polar()
    assert set(np.unique(r.ice_type[r.ocean]).tolist()) <= {0, 1, 2, 3}
    assert (r.ice_type[~r.ocean] == 0).all()


def test_t80_coupler_reports_its_status():
    from hexworld.config import ClimateParams
    from hexworld.earth.coupler import couple
    from hexworld.stages.climate import zonal_temperature

    lat, sea = _basin(south=10.0, north=70.0)
    phi = np.abs(lat)
    cont = np.where(sea, 0.0, 0.5)
    t_zone = zonal_temperature(lat)
    st = couple(ClimateParams(), EarthParams(coupler_max_iter=1), EARTH, lat, phi, sea, cont,
                t_zone, np.zeros_like(lat), 150.0, np.zeros_like(lat))  # fmt: skip
    assert st.report["status"] == "NOT_CONVERGED"  # one iteration cannot show convergence
    st2 = couple(ClimateParams(), EarthParams(coupler_max_iter=12), EARTH, lat, phi, sea, cont,
                 t_zone, np.zeros_like(lat), 150.0, np.zeros_like(lat))  # fmt: skip
    assert st2.report["status"] in ("CONVERGED", "NOT_CONVERGED")
    assert len(st2.report["iterations"]) >= 2


# -- the stage on a generated world --------------------------------------------------------------
@pytest.fixture(scope="module")
def eproject(hist_project):
    assert hist_project.earth is not None
    return hist_project


def test_stage_runs_and_validates(eproject):
    e = eproject.earth
    assert e.validation["ok"], e.validation["errors"]
    assert e.summary["pet_method"].startswith("FAO56")


def test_t23_t68_lakes_are_inland_water_with_lake_ice(eproject):
    L = eproject.earth.layers
    lake = (eproject.hydro.layers["lake_id"] >= 0) & ~eproject.hydro.layers["ocean_mask"]
    assert (L["realm"][lake] == 2).all()
    assert (L["sea_ice_concentration_monthly"][:, lake] == 0).all()
    assert (L["ice_cover"][lake & (L["lake_ice_months"] > 0)] == 4).all()


def test_t28_ash_uses_the_climate_winds_aloft(eproject):
    v = eproject.volcanism
    if v is None or not v.events:
        pytest.skip("no explicit eruption in this world")
    flags = {f for e in v.events for f in e.get("results", {}).get("tephra", {}).get(
        "quality_flags", [])}  # fmt: skip
    if flags:
        assert any(f.startswith("thermal_wind_aloft_v53") for f in flags)
        assert "synthetic_upper_wind" not in flags


def test_t05_deterministic_and_saved(tmp_path, eproject):
    from hexworld.io import load_project, save_project
    from hexworld.pipeline_v53 import run_earth

    again = run_earth(eproject.config, eproject)
    for k, a in eproject.earth.layers.items():
        b = again["layers"][k]
        a64 = np.nan_to_num(np.asarray(a, dtype=np.float64))
        assert np.array_equal(a64, np.nan_to_num(np.asarray(b, dtype=np.float64))), k
    q = load_project(save_project(eproject, tmp_path / "w"))
    assert q.earth is not None
    assert q.earth.content_hashes() == eproject.earth.content_hashes()


def test_t01_legacy_config_keeps_the_old_climate(hist_config):
    from hexworld.pipeline_v4 import v4_stage_keys

    legacy = replace(hist_config, earth=EarthParams(enabled=False))
    up = {"v3_raster": "x"}
    k_old = v4_stage_keys(legacy, up, 0.0)["v4_climate"]
    k_new = v4_stage_keys(hist_config, up, 0.0)["v4_climate"]
    assert k_old != k_new
    legacy2 = replace(hist_config, earth=EarthParams(enabled=False, obliquity_deg=10.0))
    # when off, Earth parameters do not enter the key (old projects keep their keys)
    assert v4_stage_keys(legacy2, up, 0.0)["v4_climate"] == k_old


def test_t07_views_render_without_physics(eproject, monkeypatch):
    import hexworld.earth.ocean as ocean_mod
    from hexworld.views import VIEWS

    def boom(*a, **k):
        raise AssertionError("a view must not run the ocean model")

    monkeypatch.setattr(ocean_mod, "run_ocean", boom)
    for vid in ("ocean_physics", "sea_ice", "atmosphere", "evapotranspiration"):
        v = VIEWS[vid]
        assert v.availability(eproject)[0]
        for m in v.modes:
            ov = frozenset(v.default_overlays() | {"month:6"})
            assert v.render(eproject, m.id, ov).shape[:2] == eproject.layer("elevation_m").shape
            assert v.legend(eproject, m.id)


def test_sst_not_latitude_only(eproject):
    L = eproject.earth.layers
    ocean = eproject.hydro.layers["ocean_mask"].astype(bool)
    lat = eproject.hydro.layers["latitude_deg"]
    sst = L["sst_monthly_c"].mean(axis=0)
    spread = []
    for a in np.arange(-60, 60, 5.0):
        m = ocean & (lat >= a) & (lat < a + 5)
        if m.sum() > 30:
            spread.append(float(np.nanstd(sst[m])))
    assert spread and max(spread) > 0.2


def test_math_sanity():
    assert math.isclose(EARTH.omega, 7.27e-5, rel_tol=0.01)
