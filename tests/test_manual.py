"""Acceptance tests named after the engineering manual (sections 12 and 13).

Each test carries the manual's ID (G.., C.., H.., E..) or the row of the control table in
section 12. IDs already covered elsewhere are listed in docs/manual_compliance.md.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from hexworld.config import ClimateParams, SeaParams
from hexworld.geometry import HexLayout
from hexworld.pipeline_v4 import _slope
from hexworld.stages.climate import (
    SEASON_PHASE,
    SECONDS_PER_YEAR,
    Latitude,
    _faces,
    hex_climate,
    season_balance,
    season_temperature,
    upwind_operator,
    upwind_step,
    world_climate,
)
from hexworld.stages.hydrology import Graph, hex_graph, hydraulic_geometry, hydrology
from hexworld.stages.landscape import _laplacian, implicit_incision, stream_power_rate
from hexworld.stages.sea import derive_land

GRID = {"x0_km": 0.0, "y0_km": 0.0, "cell_km": 60.0}


# -- section 12: control calculations ----------------------------------------------------
def test_12_regular_hex_with_a_10_km_side():
    layout = HexLayout(4, 4, 10.0 * math.sqrt(3.0), 0.0, 0.0)
    assert layout.radius == pytest.approx(10.0)
    assert layout.hex_area_km2 == pytest.approx(259.808, abs=1e-3)
    assert layout.spacing_km == pytest.approx(17.321, abs=1e-3)


def test_12_plate_drift_and_uplift_units():
    from hexworld.stages.history import MM_YR_TO_KM_MA, MM_YR_TO_M_MA

    assert 35.0 * 20.0 * MM_YR_TO_KM_MA == pytest.approx(700.0)  # km
    assert 0.5 * 1.0 * MM_YR_TO_M_MA == pytest.approx(500.0)  # m before erosion


def _flat_world(height_m=0.0, lat=45.0, ny=20, nx=40):
    z = np.full((ny, nx), -3000.0)
    z[:, 10:30] = 200.0
    lat_ = Latitude(lat, 0.5 * ny * GRID["cell_km"], 111.2, False)
    return world_climate(ClimateParams(), GRID, z, 0.0, lat_)


def test_12_C04_orographic_temperature_minus_9_75_for_1500_m():
    wc = _flat_world()
    layout = HexLayout(10, 6, 60.0, 900.0, 300.0)
    low = hex_climate(ClimateParams(), wc, layout, np.full((6, 10), 1.0), 0.0)
    high = hex_climate(ClimateParams(), wc, layout, np.full((6, 10), 1501.0), 0.0)
    for key in ("temperature_mean_c", "temperature_djf_c", "temperature_jja_c"):
        assert np.allclose(low[key] - high[key], 9.75, atol=1e-3), key


def _chain(heights, area_m2=1e6, runoff_mm=None, sea=None):
    """A line of cells i -> i+1 ... with the last one the sea, as a hydrology graph."""
    n = len(heights)
    nbr = np.full((n, 6), -1, dtype=np.int64)
    for i in range(n - 1):
        nbr[i, 0] = i + 1
        nbr[i + 1, 1] = i
    dist = np.full((n, 6), 1000.0)
    graph = Graph(nbr, dist, np.zeros(n, dtype=bool))
    z = np.asarray(heights, dtype=np.float64)
    sea = (z <= 0.0) if sea is None else sea
    runoff = np.zeros(n) if runoff_mm is None else np.asarray(runoff_mm, dtype=np.float64)
    return hydrology(z, sea, graph, area_m2, runoff, np.zeros(n), np.zeros(n))


def test_12_H01_local_runoff_600_mm_on_1_km2_is_0_019_m3_s():
    hyd = _chain([10.0, -5.0], area_m2=1e6, runoff_mm=[600.0, 0.0])
    assert hyd["discharge_m3_s"][0] == pytest.approx(600_000.0 / SECONDS_PER_YEAR)
    assert hyd["discharge_m3_s"][0] == pytest.approx(0.0190, abs=1e-4)


def test_12_slope_10_m_over_2000_m():
    layout = HexLayout(2, 1, 2.0, 0.0, 0.0)  # two hexes 2 km apart
    s = _slope(np.array([[110.0, 100.0]]), layout)  # m per km
    assert s[0] / 1000.0 == pytest.approx(0.005)


def test_12_E01_stream_power_control_and_zero_water():
    rate = stream_power_rate(0.002, 4.0, 0.5, 0.1, n=1.0, threshold=0.0)
    assert rate == pytest.approx(0.0004)
    assert rate * 1000.0 == pytest.approx(0.4)  # m in 1000 years before limits
    assert stream_power_rate(0.002, 0.0, 0.5, 0.1, threshold=0.0) == 0.0
    assert stream_power_rate(0.002, 0.0, 0.5, 0.1, threshold=1e-4) == 0.0
    assert stream_power_rate(0.002, 4.0, 0.5, 0.1, threshold=1e-3) == 0.0  # below E_c
    # the implicit solver: no water -> no incision; threshold blocks weak flows
    z = np.array([0.0, 100.0])
    rc = np.array([0, 0])
    order = np.array([0, 1])
    skip = np.array([True, False])
    assert implicit_incision(z, rc, order, np.array([0.0, 0.0]), 0.0, skip)[1] == 100.0
    assert implicit_incision(z, rc, order, np.array([0.0, 1e-3]), 0.5, skip)[1] == 100.0
    cut = implicit_incision(z, rc, order, np.array([0.0, 1e-3]), 0.05, skip)[1]
    assert cut == pytest.approx((100.0 + 0.05) / 1.001) and cut < 100.0


def test_12_sea_mask_24_m_under_a_25_m_sea():
    layout = HexLayout(2, 1, 10.0, 0.0, 0.0)
    elevation = np.array([[24.0, 30.0]])
    land, *_ = derive_land(elevation, np.ones((1, 2)), 25.0, layout, SeaParams())
    assert not land[0, 0] and land[0, 1]
    assert elevation[0, 0] - 25.0 == pytest.approx(-1.0)


def test_12_H06_confluence_3_plus_5_plus_local_2_is_10():
    # cells 0 and 1 are two rivers joining in cell 2, which drains to the sea (cell 3)
    nbr = np.full((4, 6), -1, dtype=np.int64)
    nbr[0, 0], nbr[1, 0], nbr[2, :3] = 2, 2, [0, 1, 3]
    nbr[3, 0] = 2
    graph = Graph(nbr, np.full((4, 6), 1000.0), np.zeros(4, dtype=bool))
    area = 1e6
    to_mm = SECONDS_PER_YEAR / area * 1000.0  # m^3/s on 1 km^2 -> mm/yr
    runoff = np.array([3.0, 5.0, 2.0, 0.0]) * to_mm
    z = np.array([50.0, 60.0, 10.0, -100.0])
    hyd = hydrology(z, z <= 0, graph, area, runoff, np.zeros(4), np.zeros(4))
    assert list(hyd["receiver"][:3]) == [2, 2, 3]
    assert hyd["discharge_m3_s"][2] == pytest.approx(10.0)
    # with explicit losses the confluence closes as upstream + local - loss
    q = hyd["discharge_m3_s"]
    assert q[2] == pytest.approx(q[0] + q[1] + 2.0 - hyd["channel_loss_m3_s"][2])


def test_12_land_fraction_40_of_100():
    layout = HexLayout(10, 10, 10.0, 0.0, 0.0)
    elevation = np.full((10, 10), -100.0)
    elevation.ravel()[:40] = 50.0
    land, *_ = derive_land(elevation, np.ones((10, 10)), 0.0, layout, SeaParams())
    assert land.mean() == pytest.approx(0.4)


# -- C01: hemispheres and seasons --------------------------------------------------------
def test_C01_hemispheres_have_opposite_seasons_and_the_rain_belt_follows_the_sun():
    ny, nx = 120, 40  # 7200 km tall, centred on the equator: about 32 S .. 32 N
    z = np.full((ny, nx), -4000.0)
    lat = Latitude(0.0, 0.5 * ny * GRID["cell_km"], 111.2, False)
    wc = world_climate(ClimateParams(), GRID, z, 0.0, lat)
    phi = wc.lat[:, 0]
    assert phi[0] > 0 > phi[-1]  # north is map up (-y)
    north = (np.abs(phi - 25.0)).argmin()
    south = (np.abs(phi + 25.0)).argmin()
    t = wc.t_sea_level
    amp = wc.seasonal_amplitude
    djf = season_temperature(t, amp, wc.lat, 0)
    jja = season_temperature(t, amp, wc.lat, 2)
    assert jja[north].mean() > djf[north].mean()  # northern summer in June-August
    assert djf[south].mean() > jja[south].mean()  # southern summer in December-February
    assert SEASON_PHASE[1] == SEASON_PHASE[3] == 0.0
    # the tropical rain belt lies north of the equator in JJA and south of it in DJF
    p_djf = wc.seasonal_precipitation[0].mean(axis=1)
    p_jja = wc.seasonal_precipitation[2].mean(axis=1)
    assert phi[np.argmax(p_jja)] > 0.0 > phi[np.argmax(p_djf)]
    # the annual mean is the mean of the seasons
    assert np.allclose(wc.precipitation, wc.seasonal_precipitation.mean(axis=0))


# -- C03: moisture budget ----------------------------------------------------------------
def test_C03_world_moisture_budget_closes():
    z = np.full((40, 80), -3000.0)
    z[10:30, 30:60] = 800.0
    z[15:25, 45:48] = 3000.0
    wc = world_climate(ClimateParams(), GRID, z, 0.0, Latitude(30.0, 1200.0, 111.2, False))
    b = wc.balance
    assert b["relative_residual"] < 1e-12
    lhs = b["storage_change_mm_yr"]
    rhs = (
        b["evaporation_mm_yr"]
        - b["precipitation_mm_yr"]
        + b["border_inflow_mm_yr"]
        - b["border_outflow_mm_yr"]
    )
    assert lhs == pytest.approx(rhs, abs=1e-9 * b["precipitation_mm_yr"])
    assert (wc.seasonal_precipitation >= 0).all() and (wc.moisture >= 0).all()


def test_C03_upwind_transport_is_conservative_and_matches_its_matrix():
    rng = np.random.default_rng(3)
    ny, nx = 17, 23
    u = rng.normal(0.0, 4.0, (ny, nx))
    v = rng.normal(0.0, 4.0, (ny, nx))
    uf, vf = _faces(u, v)
    scale = 0.2 / max(np.abs(uf).max(), np.abs(vf).max())
    cxf, cyf = uf * scale, vf * scale
    q = rng.random((ny, nx))
    q_ref, f_in, f_out = upwind_step(q, cxf, cyf)
    op, b_in, b_out = upwind_operator(cxf, cyf)
    q_mat = (op @ q.ravel()).reshape(ny, nx)
    assert np.allclose(q_ref, q_mat, atol=1e-14)
    assert q_ref.sum() - q.sum() == pytest.approx(f_in - f_out, abs=1e-12)
    assert b_in @ q.ravel() == pytest.approx(f_in) and b_out @ q.ravel() == pytest.approx(f_out)
    assert (q_ref >= 0).all()  # positive under the Courant limit
    # a closed box (no wind at the border faces) keeps every drop
    cxf[:, [0, -1]] = 0.0
    cyf[[0, -1], :] = 0.0
    q_box, f_in, f_out = upwind_step(q, cxf, cyf)
    assert f_in == f_out == 0.0 and q_box.sum() == pytest.approx(q.sum(), abs=1e-12)


# -- H07: seasonal water balance ---------------------------------------------------------
def test_H07_seasonal_stores_close_the_balance_in_every_hex():
    rng = np.random.default_rng(7)
    n = 500
    t_mean = rng.uniform(-25.0, 28.0, n)
    amp = rng.uniform(1.0, 25.0, n)
    lat = np.full(n, 45.0)
    t_s = np.stack([season_temperature(t_mean, amp, lat, s) for s in range(4)])
    p_s = rng.gamma(2.0, 80.0, (4, n))
    pet_s = rng.uniform(0.0, 400.0, (4, n))
    b = season_balance(ClimateParams(), t_s, p_s, pet_s)
    assert np.abs(b["residual"]).max() < 1e-9
    p = p_s.sum(axis=0)
    assert (b["aet"] <= pet_s.sum(axis=0) + 1e-9).all()
    assert (b["aet"] <= p + 1e-9).all()  # stores start empty: no store loses water
    for key in ("aet", "runoff", "deep", "snow", "baseflow", "melt", "ice_discharge"):
        assert (b[key] >= -1e-12).all(), key
    assert (b["storage_change"] >= -1e-9).all()


def test_H07_snowmelt_gives_a_spring_or_summer_flood_and_glaciers_discharge():
    t_s = np.array([[-15.0, -40.0], [2.0, -30.0], [12.0, -12.0], [1.0, -28.0]])  # DJF..SON
    p_s = np.full((4, 2), 100.0)
    pet_s = np.array([[0.0, 0.0], [40.0, 0.0], [200.0, 0.0], [30.0, 0.0]])
    b = season_balance(ClimateParams(), t_s, p_s, pet_s)
    peak = np.argmax(b["seasonal_runoff"], axis=0)
    assert peak[0] in (1, 2)  # melt in spring / early summer, none in winter
    assert b["seasonal_runoff"][0, 0] < b["seasonal_runoff"][peak[0], 0]
    assert b["snow"][0] > 0 and b["melt"][0] > 0
    # the polar hex never melts: its yearly snow leaves as ice discharge, not storage
    assert b["melt"][1] < 1.0
    assert b["ice_discharge"][1] == pytest.approx(b["snow"][1] - b["melt"][1], rel=1e-6)
    assert np.abs(b["residual"]).max() < 1e-9


def test_H07_hex_climate_annual_layers_are_sums_of_the_seasons():
    wc = _flat_world()
    layout = HexLayout(10, 6, 60.0, 900.0, 300.0)
    h = hex_climate(ClimateParams(), wc, layout, np.full((6, 10), 300.0), 0.0)
    assert h.balance["max_abs_residual_mm_yr"] < 1e-9
    p = h["precipitation_mm_yr"]
    aet = h["actual_evapotranspiration_mm_yr"]
    assert (aet <= p + 1e-3).all()
    assert (h["runoff_mm_yr"] >= 0).all()
    assert set(np.unique(h["runoff_peak_season"])) <= {0, 1, 2, 3, 255}


# -- E02 / E05: hillslope and incision -----------------------------------------------------
def test_E02_hillslope_diffusion_conserves_volume_on_a_closed_grid():
    layout = HexLayout(23, 17, 10.0, 0.0, 0.0)
    graph = hex_graph(layout)
    rng = np.random.default_rng(2)
    z = rng.normal(500.0, 200.0, layout.width * layout.height)
    total = z.sum()
    for _ in range(50):
        z = z + 0.1 * (10_000.0**2) * _laplacian(z, graph) / 3.0
    assert z.sum() == pytest.approx(total, rel=1e-12)
    assert z.std() < 200.0  # smoothed


def test_E05_implicit_incision_is_stable_and_converges_with_smaller_steps():
    # one cell above a fixed outlet: dz/dt = -K' (z - z_r), exact z = z_r + h0 exp(-K' t)
    k_rate, t_end, h0 = 2e-5, 1e5, 1000.0  # 1/yr, yr, m
    exact = h0 * math.exp(-k_rate * t_end)
    errors = []
    for steps in (1, 4, 16, 64):
        dt = t_end / steps
        z = np.array([0.0, h0])
        for _ in range(steps):
            z = implicit_incision(
                z,
                np.array([0, 0]),
                np.array([0, 1]),
                np.array([0.0, k_rate * dt]),
                0.0,
                np.array([True, False]),
            )
            assert np.isfinite(z).all() and z[1] >= z[0]  # never below its receiver
        errors.append(abs(z[1] - exact))
    assert errors == sorted(errors, reverse=True) and errors[-1] < 0.05 * errors[0]


def test_hydraulic_geometry_grows_with_discharge():
    w, d, v = hydraulic_geometry(np.array([0.0, 10.0, 1000.0, 100000.0]))
    assert w[0] == d[0] == v[0] == 0.0
    assert (np.diff(w) > 0).all() and (np.diff(d) > 0).all()
    assert w[2] == pytest.approx(158.1, abs=0.1) and d[2] == pytest.approx(4.75, abs=0.01)
    # continuity: Q = W * D * V
    assert np.allclose((w * d * v)[1:], [10.0, 1000.0, 100000.0])


# -- H02 / H03 / H04 / H05: routing on the hex graph -----------------------------------------
def _hex_hyd(z, layout, sea=None, choice=None):
    graph = hex_graph(layout)
    z = np.asarray(z, dtype=np.float64).ravel()
    sea = z <= 0.0 if sea is None else sea
    n = z.size
    runoff = np.where(sea, 0.0, 300.0)
    return graph, hydrology(
        z,
        sea,
        graph,
        layout.hex_area_km2 * 1e6,
        runoff,
        np.full(n, 300.0),
        np.full(n, 800.0),
        choice=choice,
    )


def test_H02_H03_receivers_are_real_hex_neighbours_and_the_graph_is_acyclic():
    layout = HexLayout(31, 23, 10.0, 0.0, 0.0)
    rng = np.random.default_rng(11)
    z = rng.normal(300.0, 150.0, (23, 31))
    z[:, :3] = -50.0
    graph, hyd = _hex_hyd(z, layout)
    rc = hyd["receiver"]
    moved = rc != np.arange(rc.size)
    lake = hyd["lake_id"]
    for c in np.nonzero(moved)[0]:
        # D6, never D8; the one exception is a lake: a second exit of a lake is linked across
        # the water surface to the lake's single outlet (both cells belong to the lake)
        assert rc[c] in graph.nbr[c] or (lake[c] >= 0 and lake[c] == lake[rc[c]])
    rank = hyd["routing"].rank
    assert (rank[rc[moved]] < rank[moved]).all()  # acyclic
    # every land cell reaches the sea, the window edge or a closed basin
    for c in range(rc.size):
        seen = 0
        while rc[c] != c:
            c = rc[c]
            seen += 1
            assert seen <= rc.size


def test_H04_flat_plain_drains_with_a_finite_stable_route():
    layout = HexLayout(25, 15, 10.0, 0.0, 0.0)
    z = np.full((15, 25), 50.0)
    z[:, 0] = -20.0  # the sea on the west side
    sea = (z <= 0).ravel()
    border = hex_graph(layout).border
    edge = np.zeros(z.size, dtype=bool)  # only the sea is an outlet
    graph = hex_graph(layout)
    args = (z.ravel(), sea, graph, layout.hex_area_km2 * 1e6, np.where(sea, 0.0, 300.0))
    rest = (np.full(z.size, 300.0), np.zeros(z.size))
    a = hydrology(*args, *rest, edge_outlet=edge)
    b = hydrology(*args, *rest, edge_outlet=edge)
    assert (a["receiver"] == b["receiver"]).all()  # stable tie-break
    assert border.any() and not a["lakes"]
    land = ~sea
    supplied = (0.3 * layout.hex_area_km2 * 1e6 / SECONDS_PER_YEAR) * land.sum()
    assert a["discharge_m3_s"][a["routing"].outlet].sum() == pytest.approx(supplied, rel=1e-9)


def test_H05_priority_flood_never_raises_the_real_surface():
    layout = HexLayout(21, 15, 10.0, 0.0, 0.0)
    yy, xx = np.mgrid[0:15, 0:21]
    z = 200.0 + 8.0 * np.hypot(xx - 12, yy - 7)  # a bowl
    z[:, 0] = -30.0
    z[7, 1:5] = [40.0, 120.0, 230.0, 240.0]  # the sill toward the sea
    before = z.copy()
    _graph, hyd = _hex_hyd(z, layout)
    assert np.array_equal(z, before)  # the input heights are not modified
    assert (hyd["routing"].routing_m >= z.ravel() - 1e-12).all()  # fill only on the copy
    assert hyd["lakes"], "the bowl holds a lake"
    lake = hyd["lakes"][0]
    sill = float(hyd["routing"].routing_m[hyd["lake_id"] == lake["id"]].max())
    assert lake["level_m"] <= sill + 1e-6  # never above the spill level


def test_6_4_routing_numbers_follow_the_world_position_not_the_window():
    from hexworld.stages.hydrology import routing_choice

    spacing = 40.0
    col_w = 1.5 * spacing / math.sqrt(3.0)
    a = HexLayout(60, 40, spacing, 1000.0, 500.0)
    b = HexLayout(60, 40, spacing, 1000.0 + 12 * col_w, 500.0 + 7 * spacing)
    ca = routing_choice("Shift1", 0, "stochastic", a).reshape(40, 60)
    cb = routing_choice("Shift1", 0, "stochastic", b).reshape(40, 60)
    assert np.array_equal(ca[7:, 12:], cb[:33, :48])  # the same hexes draw the same numbers
    assert 0.0 <= ca.min() and ca.max() < 1.0 and abs(ca.mean() - 0.5) < 0.03
    other = routing_choice("Shift2", 0, "stochastic", a).reshape(40, 60)
    assert not np.array_equal(ca, other)  # and they depend on the seed


@pytest.mark.slow
def test_6_4_two_windows_of_one_world_route_their_common_area_the_same_way():
    from dataclasses import replace

    from hexworld.config import Config
    from hexworld.pipeline import generate

    spacing, dc, dr, w, h = 40.0, 20, 10, 160, 110
    col_w = 1.5 * spacing / math.sqrt(3.0)
    projects = []
    for x, y in ((7000.0, 1500.0), (7000.0 + dc * col_w, 1500.0 + dr * spacing)):
        cfg = Config(
            seed="Terra7",
            width=w,
            height=h,
            hex_spacing_km=spacing,
            mode="regional",
            viewport_origin_km=(x, y),
            simulation_bounds_km=(2000.0, -2000.0, 14000.0, 8000.0),
        )
        cfg = replace(cfg, erosion=replace(cfg.erosion, enabled=False))
        projects.append(generate(cfg, through="v4"))
    a, b = (p.hydro.layers for p in projects)
    m = 3
    sa = (slice(dr + m, h - m), slice(dc + m, w - m))
    sb = (slice(m, h - dr - m), slice(m, w - dc - m))
    assert np.array_equal(a["elevation_m"][sa], b["elevation_m"][sb])
    land = a["land_mask"][sa]
    assert land.sum() > 1000
    assert (a["flow_direction"][sa][land] == b["flow_direction"][sb][land]).mean() > 0.99
