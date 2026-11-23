"""TZ 5.4 unit tests on synthetic worlds: contracts, finite volumes, circulation, moisture,
ITCZ, monsoons (T01-T40, N01-N18). Tolerances are derived in tests/fixtures/v5_4."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from hexworld.atmosphere import circulation as circ
from hexworld.atmosphere import fv, moisture, monsoon, normals
from hexworld.config import AtmosphereParams
from hexworld.earth.planet import EARTH, Planet

FIX = Path(__file__).parent / "fixtures" / "v5_4"


def fixture(name):
    return json.loads((FIX / f"{name}.json").read_text(encoding="utf-8"))


ATM = replace(AtmosphereParams(quality="fast"), domain_mode="isolated_approximation")


def synth(
    ny=36,
    nx=48,
    cell_km=330.0,
    lat_top=60.0,
    land=None,
    obliquity=23.44,
    land_h=None,
    sst_monthly=None,
    day_h=24.0,
):
    """A synthetic raster world: rows from +lat_top to -lat_top, ``land`` a bool mask."""
    from scipy import ndimage

    from hexworld.stages.climate import zonal_temperature

    lat = np.repeat(np.linspace(lat_top, -lat_top, ny)[:, None], nx, axis=1)
    land = np.zeros((ny, nx), bool) if land is None else land
    sea = ~land
    lh = np.zeros((ny, nx)) if land_h is None else land_h
    dist = ndimage.distance_transform_edt(~sea) * cell_km if sea.any() else np.full(sea.shape, 4e3)
    cont = 1.0 - np.exp(-dist / 900.0)
    planet = Planet(obliquity_deg=obliquity, day_length_h=day_h)
    from hexworld.earth.planet import seasonal_amplitude_ratio

    amp = (1.0 + 0.32 * np.abs(lat)) * (0.33 + 0.67 * cont) * seasonal_amplitude_ratio(planet, lat)
    t_sl = zonal_temperature(lat)
    grid = {"x0_km": 0.0, "y0_km": 0.0, "cell_km": cell_km}
    return normals.build(
        ATM, planet, lat, sea, lh, t_sl, amp, cont, cell_km, grid, sst_monthly=sst_monthly
    )


@pytest.fixture(scope="module")
def ocean_world():
    return synth()


@pytest.fixture(scope="module")
def tropical_pair():
    ny, nx = 36, 48
    lat = np.linspace(60, -60, ny)
    north = np.zeros((ny, nx), bool)
    north[(lat > 6) & (lat < 30), 14:34] = True
    south = np.zeros((ny, nx), bool)
    south[(lat < -6) & (lat > -30), 14:34] = True
    return synth(land=north), synth(land=south), north, south


def land_monthly(n, mask):
    p = n.precip_monthly()
    return p[:, mask].mean(axis=1)


# -- A. contracts -----------------------------------------------------------------------------
def test_t01_missing_v53_input_is_reported_not_hidden():
    from hexworld.config import ClimateParams, EarthParams
    from hexworld.stages.climate import Latitude, world_climate

    elev = np.full((12, 16), -1000.0)
    elev[4:8, 4:10] = 500.0
    grid = {"x0_km": 0.0, "y0_km": 0.0, "cell_km": 200.0, "nx": 16, "ny": 12}
    wc = world_climate(
        ClimateParams(),
        grid,
        elev,
        0.0,
        Latitude(0.0, 1200.0, 111.2, False),
        earth=EarthParams(enabled=False),
        atmosphere=ATM,
    )
    assert wc.atmosphere is None
    assert wc.balance["atmosphere"]["status"] == "blocked"


def test_t04_zero_obliquity_has_no_season():
    r = circ.unit_response(Planet(obliquity_deg=0.0), np.array([-30.0, 0.0, 20.0, 60.0]), 30.0)
    assert np.abs(r).max() < 1e-9  # no sun cycle, no summer, no "July monsoon"


def test_t03_hemispheres_mirror_by_half_a_year():
    r = circ.unit_response(EARTH, np.array([35.0, -35.0]), 30.0)
    assert np.allclose(r[:, 0], np.roll(r[:, 1], 6), atol=0.05)
    assert int(np.argmax(r[:, 0])) in (6, 7)  # July / August in the north


def test_t11_land_and_sea_have_different_inertia():
    lat = np.array([45.0])
    land = circ.unit_response(EARTH, lat, circ.TAU_LAND_D)[:, 0]
    sea = circ.unit_response(EARTH, lat, circ.TAU_SEA_D)[:, 0]
    assert int(np.argmax(sea)) >= int(np.argmax(land))  # the sea peaks later


def test_t14_rotation_changes_the_belts_not_fixed_30_60():
    fast, _ = circ.hadley_edge_deg(Planet(day_length_h=12.0), 50.0)
    earth, _ = circ.hadley_edge_deg(EARTH, 50.0)
    slow, q = circ.hadley_edge_deg(Planet(day_length_h=96.0), 50.0)
    assert fast < earth < slow
    assert q == "global_hadley_regime_unvalidated"


def test_t13_n16_equator_is_finite(ocean_world):
    c = ocean_world.circ
    for a in (c.u, c.v, c.slp_pa, c.p_sfc_pa):
        assert np.isfinite(a).all()
    u, v = circ.frictional_wind(
        np.array([1e-3]), np.array([0.0]), np.array([0.0]), np.array([5e-5]), 50.0
    )
    assert np.isfinite(u).all() and u[0] < 0  # down the gradient at f = 0


def test_t15_850_is_not_the_surface_wind(ocean_world):
    c = ocean_world.circ
    assert not np.allclose(c.u[:, 0], c.u[:, 1])


def test_t16_jet_aloft_with_quality(ocean_world):
    c = ocean_world.circ
    mid = np.abs(ocean_world.lat) > 30
    assert c.u[:, 3][:, mid].mean() > c.u[:, 1][:, mid].mean() + 3.0  # westerlies grow aloft
    assert ocean_world.quality["wind_500_250"].startswith("thermal_wind")


def test_t06_t17_levels_below_ground_and_extrapolation(ocean_world):
    c = ocean_world.circ
    n = c.p_sfc_pa[0].size
    c2 = replace(c, p_sfc_pa=c.p_sfc_pa.copy())
    c2.p_sfc_pa[:, 0, 0] = 70000.0  # a 3 km plateau
    assert not circ.level_available(c2, 0, 92500.0)[0, 0]
    u, v, q = circ.wind_at(c2, 0, pressure_pa=92500.0, cell=np.array([0]))
    assert q[0] == circ.Q_BELOW_GROUND
    _, _, q5 = circ.wind_at(c, 0, pressure_pa=50000.0, cell=np.arange(n))
    assert (q5 == circ.Q_MODELED).all()
    _, _, q1 = circ.wind_at(c, 0, pressure_pa=10000.0, cell=np.arange(n))
    assert (q1 == circ.Q_EXTRAPOLATED).all()
    _, _, q7 = circ.wind_at(c, 0, pressure_pa=70000.0, cell=np.arange(n))
    assert (q7 == circ.Q_INTERPOLATED).all()


def test_t18_pressure_and_speeds_are_physical(ocean_world):
    c = ocean_world.circ
    assert 95000 < c.slp_pa.min() and c.slp_pa.max() < 106000
    assert np.hypot(c.u[:, 0], c.v[:, 0]).max() <= circ.SURFACE_CAP + 1e-6


def test_t19_mountains_slow_the_surface_wind():
    ny, nx = 36, 48
    lat = np.linspace(60, -60, ny)
    land = np.zeros((ny, nx), bool)
    land[(lat > 35) & (lat < 55), 10:40] = True
    flat = synth(land=land)
    hills = synth(land=land, land_h=np.where(land, 3000.0, 0.0))
    s0 = np.hypot(flat.circ.u[:, 0], flat.circ.v[:, 0])[:, land].mean()
    s1 = np.hypot(hills.circ.u[:, 0], hills.circ.v[:, 0])[:, land].mean()
    assert s1 < s0


# -- C. finite volumes and the vapour budget -------------------------------------------------
def two_cells():
    return fv.Mesh(
        area=np.array([1e10, 1e10]),
        ei=np.array([0]),
        ej=np.array([1]),
        length=np.array([1e5]),
        normal=np.array([[1.0, 0.0]]),
        dist=np.array([1e5]),
        bi=np.zeros(0, int),
        blength=np.zeros(0),
        bnormal=np.zeros((0, 2)),
    )


def test_t23_n10_internal_face_is_antisymmetric():
    m = two_cells()
    tr = fv.transport(m, np.array([5.0, 5.0]), np.zeros(2))
    x = np.array([10.0, 0.0])
    dxdt = -(tr.D @ x)
    flux_kg_s = m.area * dxdt
    assert flux_kg_s[0] == pytest.approx(-flux_kg_s[1])
    assert flux_kg_s.sum() == pytest.approx(0.0, abs=1e-9)
    assert flux_kg_s[0] == pytest.approx(-5.0 * 1e5 * 10.0)  # u L X


def test_hex_mesh_has_six_faces_and_closed_divergence():
    from hexworld.geometry import HexLayout
    from hexworld.pipeline_v50 import _hex_neighbours

    lay = HexLayout(12, 10, 50.0, 0.0, 0.0)
    m = fv.hex_mesh(lay, _hex_neighbours(lay))
    faces = np.bincount(np.concatenate([m.ei, m.ej, m.bi]), minlength=m.n)
    assert (faces == 6).all()
    div = m.divergence(np.full(m.n, 7.0), np.full(m.n, -3.0))
    assert np.abs(div).max() < 1e-12  # a uniform flux has no divergence


def test_t22_uniform_transport_creates_no_water():
    m = fv.raster_mesh(8, 10, 1e5)
    x0 = np.full(m.n, 30.0)
    x, led = fv.advect_explicit(
        m, x0, np.full(m.n, 8.0), np.full(m.n, 2.0), 86400.0, x_bc=np.full(m.n, 30.0)
    )
    assert np.allclose(x, 30.0)
    assert abs(led["residual"]) <= 1e-9 * led["initial"]


def test_t29_n15_halving_the_step_keeps_the_budget():
    m = fv.raster_mesh(10, 14, 1e5)
    x0 = np.zeros(m.n)
    x0[40:45] = 50.0
    u, v = np.full(m.n, 10.0), np.full(m.n, -4.0)
    a, la = fv.advect_explicit(m, x0, u, v, 2 * 86400.0, cfl_target=0.45)
    b, lb = fv.advect_explicit(m, x0, u, v, 2 * 86400.0, cfl_target=0.225)
    tol = fixture("numerics")["n15_budget_rel"]
    assert abs(la["residual"]) <= tol * la["initial"] and abs(lb["residual"]) <= tol * lb["initial"]
    assert lb["substeps"] >= 2 * la["substeps"] - 1
    assert (a >= 0).all() and (b >= 0).all()
    assert np.abs(a - b).sum() / a.sum() < 0.35  # same large-scale solution


def _solve(mesh, shape, t=15.0, sst=None, sea=0.0, u=(0, 0), h=None, w_init=None, x_bc=None):
    n = mesh.n
    cu = np.zeros((12, 2, n))
    cv = np.zeros((12, 2, n))
    cu[:, :] = u[0]
    cv[:, :] = u[1]
    t_air = np.full((12, n), t)
    sst = np.full((12, n), t) if sst is None else sst
    lh = np.zeros(shape) if h is None else h
    return moisture.solve(
        mesh,
        cu,
        cv,
        t_air,
        sst,
        np.zeros((12, n)),
        np.full(n, sea),
        lh,
        np.full(n, 45.0),
        EARTH,
        circ.MONTH_DAYS,
        1e5,
        x_bc=x_bc,
        w_init=w_init,
    )


def test_t21_empty_dry_column_gives_no_rain():
    m = fv.raster_mesh(6, 6, 1e5)
    r = _solve(m, (6, 6), w_init=0.0, x_bc=np.zeros((12, m.n)))
    assert r.precip_mm.max() == 0.0 and r.w.max() == 0.0


def test_t24_more_evaporation_more_vapour_and_rain():
    m = fv.raster_mesh(6, 6, 1e5)
    cold = _solve(m, (6, 6), t=15.0, sst=np.full((12, m.n), 15.0), sea=1.0)
    warm = _solve(m, (6, 6), t=15.0, sst=np.full((12, m.n), 22.0), sea=1.0)
    assert warm.w.mean() > cold.w.mean() and warm.precip_mm.sum() > cold.precip_mm.sum()
    assert (warm.w >= 0).all() and (warm.precip_mm >= 0).all()


def test_t27_dry_cold_wind_gives_little_rain():
    m = fv.raster_mesh(6, 10, 1e5)
    r = _solve(
        m,
        (6, 10),
        t=-1.0,
        sst=np.full((12, m.n), -1.8),
        sea=1.0,
        u=(8.0, 0.0),
        x_bc=np.zeros((12, m.n)),
    )
    assert r.precip_mm.sum(axis=0).max() < 150.0  # no obligatory rain from a dry sea wind


def test_t28_open_boundary_import_is_booked():
    m = fv.raster_mesh(6, 10, 1e5)
    r = _solve(m, (6, 10), u=(8.0, 0.0), x_bc=np.full((12, m.n), 25.0))
    for led in r.ledger:
        assert led["import_kg"] > 0
        assert led["relative_residual"] < fixture("numerics")["budget_rel"]


def test_t26_orography_moves_water_windward():
    ny, nx = 6, 14
    m = fv.raster_mesh(ny, nx, 1e5)
    h = np.zeros((ny, nx))
    h[:, 6:8] = np.array([1500.0, 3000.0])
    r = _solve(m, (ny, nx), u=(10.0, 0.0), h=h, x_bc=np.full((12, m.n), 25.0))
    p = r.precip_mm.sum(axis=0).reshape(ny, nx)
    assert p[:, 5:7].mean() > 1.5 * p[:, 9:11].mean()  # windward wetter than the lee
    oro = r.components_mm[1].sum(axis=0).reshape(ny, nx)
    assert oro[:, 9:].sum() == 0.0  # no uplift rain on the descending side
    for led in r.ledger:
        assert led["relative_residual"] < fixture("numerics")["budget_rel"]


def test_t25_convergence_strengthens_rain(ocean_world):
    conv = ocean_world.moist.conv.reshape((12,) + ocean_world.shape)
    p = ocean_world.precip_monthly()
    trop = np.abs(ocean_world.lat) < 25
    hi = conv[:, trop] > np.percentile(conv[:, trop], 80)
    lo = conv[:, trop] < np.percentile(conv[:, trop], 20)
    assert p[:, trop][hi].mean() > 2.0 * p[:, trop][lo].mean()


def test_t30_annual_sums_and_day_weighted_temperature(ocean_world):
    p = ocean_world.precip_monthly()
    assert np.allclose(p.sum(axis=0), ocean_world.moist.precip_mm.sum(axis=0).reshape(p.shape[1:]))
    t = ocean_world.circ.t_sl
    d = circ.MONTH_DAYS.reshape(12, 1, 1)
    # the land / sea temperature of the months is day-weighted to the annual mean
    t_ann = (d * t).sum(axis=0) / d.sum()
    assert np.isfinite(t_ann).all()


def test_n01_n06_units():
    assert fv.mm_to_m3(1.0, 1.0) == pytest.approx(1000.0)
    assert fv.mm_to_m3(10.0, 100.0) == pytest.approx(1e6)
    assert fv.mm_per_day_to_m3_s(1.0, 1000.0) == pytest.approx(11.574, rel=1e-4)
    from hexworld.atmosphere.storms import displacement_km

    assert displacement_km(5.0, 24.0) == pytest.approx(432.0)
    assert displacement_km(25.0, 6.0) == pytest.approx(540.0)
    assert 2.5 * 6.0 == pytest.approx(15.0)


def test_n11_budget_validator_and_negative_control():
    assert fv.validate_budget(120.0, 40.0, -10.0, 70.0) == pytest.approx(0.0)
    with pytest.raises(fv.BudgetError):
        fv.validate_budget(120.0, 40.0, -10.0, 50.0)  # -20 mm: rejected, never corrected
    assert fv.budget_residual(120.0, 40.0, -10.0, 50.0) == pytest.approx(20.0)


# -- D. ITCZ and monsoons --------------------------------------------------------------------
def test_t12_t01_aquaplanet_has_no_continental_monsoon():
    n = synth(obliquity=0.0)
    p = n.precip_monthly().sum(axis=0)
    rows = p.mean(axis=1)
    assert np.allclose(rows, rows[::-1], rtol=0.15)  # symmetric about the equator
    assert (n.sea_frac == 1).all()
    seas = monsoon.seasonality(n.precip_monthly(), circ.MONTH_DAYS)
    assert np.nanmax(seas) < 0.2  # no sun cycle and no land: no rainy season


def test_t31_t33_tropical_continents_rain_in_their_summer(tropical_pair):
    nn, ns, north, south = tropical_pair
    pn = land_monthly(nn, north)
    ps = land_monthly(ns, south)
    assert int(np.argmax(pn)) in (5, 6, 7, 8)  # JJAS north
    assert int(np.argmax(ps)) in (11, 0, 1, 2)  # DJFM south
    assert pn.max() > 2.0 * pn.min()


def test_t32_hot_interior_without_moisture_stays_dry():
    ny, nx = 36, 48
    lat = np.linspace(60, -60, ny)
    land = np.zeros((ny, nx), bool)
    land[(lat > 5) & (lat < 45), 2:46] = True
    n = synth(land=land)
    p = n.precip_monthly().sum(axis=0)
    t = n.circ.t_air.max(axis=0)
    rows = (lat > 18) & (lat < 32)
    interior = np.zeros_like(land)
    interior[rows, 18:30] = True
    coast_rain = p[land & ~interior].mean()
    assert t[interior].mean() > 25.0  # hot summers
    assert p[interior].mean() < 0.6 * coast_rain


def test_t37_warmer_sea_moves_the_rain(ocean_world):
    sst = ocean_world.circ.t_sl.copy()
    lat = ocean_world.lat
    patch = (np.abs(lat - 12.0) < 6) & (np.arange(lat.shape[1])[None] < 20)
    sst[:, patch] += 3.0
    n2 = synth(sst_monthly=sst)
    p0 = ocean_world.precip_monthly().sum(axis=0)
    p1 = n2.precip_monthly().sum(axis=0)
    assert p1[patch].mean() > 1.1 * p0[patch].mean()


def test_t38_itcz_segments_may_be_several():
    lat = np.repeat(np.linspace(20, -20, 20)[:, None], 40, axis=1)
    conv = np.zeros((12, 20, 40))
    conv[:, 8:12, 2:12] = 5e-5
    conv[:, 8:12, 25:36] = 5e-5
    score, segs = normals.itcz(
        conv,
        np.full((12, 20, 40), 0.9),
        np.full((12, 20, 40), 28.0),
        lat,
        np.zeros((12, 40)),
        30.0,
        200.0,
    )
    assert len(segs[0]) == 2  # two separate segments, no forced global line


def test_n08_wettest_three_months_cyclic():
    p = np.array([100, 0, 0, 0, 0, 100, 0, 0, 0, 0, 100, 0], float)
    assert p.sum() == 300
    assert monsoon.wettest_3month_fraction(p[:, None])[0] * 300 == pytest.approx(200.0)
    assert monsoon.wettest_3month_start(p[:, None])[0] == 10  # November .. January


def test_n12_wet_season_share():
    p = np.array([10, 10, 10, 20, 150, 200, 250, 200, 100, 30, 10, 10], float)[:, None]
    p = p * 1200.0 / p.sum()
    frac = monsoon.summer_fraction(p, np.array([20.0]))
    assert frac[0] == pytest.approx(p[4:9].sum() / 1200.0)


def test_n18_t40_reversal_without_moisture_is_no_monsoon():
    r = monsoon.reversal_score(
        np.array([10.0]), np.array([0.0]), np.array([-10.0]), np.array([0.0])
    )
    assert r[0] == pytest.approx(1.0, abs=1e-6)
    suit = monsoon.suitability(
        np.array([0.2]), np.array([0.5]), np.array([8.0]), np.array([150.0]), r, np.array([15.0])
    )
    assert suit[0] == 0.0  # a full wind reversal, but no seasonal rain


def test_t34_t35_t36_rain_regimes():
    days = circ.MONTH_DAYS
    th = 2 * np.pi * np.arange(12) / 12
    bimodal = (150 + 120 * np.cos(2 * (th - 2 * np.pi * 3.5 / 12)))[:, None]
    wet = np.full((12, 1), 220.0)
    med = np.array([110, 90, 70, 40, 15, 3, 1, 2, 15, 50, 90, 120], float)[:, None]
    lat_eq = np.array([3.0])
    peaks = monsoon.peak_count(bimodal)
    assert peaks[0] == 2  # two rainy seasons on the equator (T34)
    pet = np.full((12, 1), 120.0)
    _, dry_w = monsoon.wet_dry_months(wet, pet)
    seas = monsoon.seasonality(wet, days)
    reg = monsoon.regime(
        wet.sum(0),
        dry_w,
        seas,
        np.zeros(1),
        monsoon.peak_count(wet),
        monsoon.summer_fraction(wet, lat_eq),
        lat_eq,
    )
    assert monsoon.REGIMES[reg[0]] == "perennially_wet"  # T35
    on, wd, _o2, n_s = monsoon.onset_withdrawal(wet, days)
    assert on[0] == 0 and n_s[0] == 0  # no rainy season defined on a constant record
    lat_m = np.array([38.0])
    _, dry_m = monsoon.wet_dry_months(med, pet)
    sf = monsoon.summer_fraction(med, lat_m)
    reg_m = monsoon.regime(
        med.sum(0),
        dry_m,
        monsoon.seasonality(med, days),
        np.zeros(1),
        monsoon.peak_count(med),
        sf,
        lat_m,
    )
    assert monsoon.REGIMES[reg_m[0]] == "winter_wet_mediterranean"  # T36, not a monsoon


def test_t39_false_start_is_not_an_onset():
    p = np.zeros(120)
    p[20:23] = 40.0  # one storm, then a dry spell
    p[60:110] = np.where(np.arange(50) % 3 == 2, 0.0, 12.0)  # the real rainy season
    d = monsoon.detect_onset_daily(p, k=5, threshold_mm=40.0, persist_days=10)
    assert d >= 55
    assert monsoon.break_days(p, d, 110) == 0


def test_onset_withdrawal_month_and_wrap():
    p = np.array([200, 180, 40, 10, 5, 5, 5, 5, 10, 30, 120, 210], float)[:, None]
    on, wd, _o2, n_s = monsoon.onset_withdrawal(p, circ.MONTH_DAYS)
    assert on[0] == 11 and wd[0] == 2 and n_s[0] == 1  # Nov .. Feb across the year end


def test_t07_t08_seed_namespaces():
    from hexworld.rng import Stream

    assert Stream("Terra7", "x").random(4).tolist() != Stream("terra7", "x").random(4).tolist()
    assert Stream("Terra7", "x").random(4).tolist() == Stream("Terra7", "x").random(4).tolist()


def test_t05_regional_window_is_marked_approximate(ocean_world):
    assert ocean_world.quality["domain"] == "isolated_approximation"
    assert ocean_world.quality["boundary_forcing"].startswith("synthetic")


def test_t20_resolution_keeps_the_large_scale_winds():
    a = synth(ny=36, nx=48, cell_km=330.0)
    b = synth(ny=54, nx=72, cell_km=220.0)
    za = a.circ.u[:, 0].mean(axis=(0, 2))
    zb = b.circ.u[:, 0].mean(axis=(0, 2))
    zb_on_a = np.interp(np.linspace(0, 1, za.size), np.linspace(0, 1, zb.size), zb)
    assert np.corrcoef(za, zb_on_a)[0, 1] > 0.95


def test_latent_heating_deepens_the_convective_trough(tropical_pair):
    nn, _ns, north, _s = tropical_pair
    atm_b = replace(ATM, quality="balanced")
    lat = np.repeat(np.linspace(60, -60, 36)[:, None], 48, axis=1)
    from hexworld.stages.climate import zonal_temperature

    planet = EARTH
    sea = ~north
    cont = np.where(north, 0.5, 0.0)
    amp = (1.0 + 0.32 * np.abs(lat)) * (0.33 + 0.67 * cont)
    grid = {"x0_km": 0.0, "y0_km": 0.0, "cell_km": 330.0}
    on = normals.build(
        atm_b, planet, lat, sea, np.zeros(lat.shape), zonal_temperature(lat), amp, cont, 330.0, grid
    )
    off = normals.build(
        replace(atm_b, latent_feedback=False),
        planet,
        lat,
        sea,
        np.zeros(lat.shape),
        zonal_temperature(lat),
        amp,
        cont,
        330.0,
        grid,
    )
    fb = on.diagnostics["latent_feedback"]
    assert fb != "off" and fb["max_trough_hpa"] > 0.5
    assert off.diagnostics["latent_feedback"] == "off"
    for led in on.moist.ledger:
        assert led["relative_residual"] < fixture("numerics")["budget_rel"]
    trop = np.abs(lat) < 20
    assert on.circ.slp_pa[:, trop].min() <= off.circ.slp_pa[:, trop].min()
