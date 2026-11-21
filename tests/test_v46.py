"""TZ 4.6 section 9: massifs, basins, water connectivity and the sea floor."""

from types import SimpleNamespace

import numpy as np
import pytest

from hexworld.config import Config, TerrainParams
from hexworld.geometry import HexLayout
from hexworld.geomorph import basins, margins
from hexworld.geomorph.bathymetry import footprint, seamount_types
from hexworld.geomorph.canyons import submarine_sediment
from hexworld.geomorph.massifs import prominence
from hexworld.geomorph.ocean_age import (
    build_ridge_pieces,
    flowline_age,
    parsons_sclater_depth,
)
from hexworld.geomorph.sections import cross_section
from hexworld.geomorph.surface_water import flood, land_fractions, water_layers, world_ocean
from hexworld.stages.boundaries import TYPE_DIVERGENT, TYPE_TRANSFORM
from hexworld.stages.hydrology import grid_graph, hex_graph, route


def _layout(w=40, h=30, spacing=20.0):
    return HexLayout(width=w, height=h, spacing_km=spacing, x0=0.0, y0=0.0)


# -- 9.3 sea floor ------------------------------------------------------------------------------
def test_age_depth_reference_values_and_smooth_transition():
    d = parsons_sclater_depth(np.array([0.0, 25.0, 100.0]))
    assert d[0] == pytest.approx(2500.0)
    assert d[1] == pytest.approx(4250.0)
    assert d[2] == pytest.approx(6400.0 - 3200.0 * np.exp(-100.0 / 62.8), abs=1.0)  # 5749 m
    t = np.linspace(0.0, 200.0, 4001)
    dd = parsons_sclater_depth(t)
    assert np.all(np.diff(dd) > 0.0)  # deeper with age everywhere
    slope = np.diff(dd) / np.diff(t)
    # no step at the 60-80 Ma blend: across 50-90 Ma the slope changes smoothly
    band = (t[1:-1] > 50.0) & (t[1:-1] < 90.0)
    assert np.max(np.abs(np.diff(slope))[band]) < 0.2  # a step would be hundreds of m/Ma
    assert abs(float(parsons_sclater_depth(60.0)) - (2500.0 + 350.0 * np.sqrt(60.0))) < 1e-6


def _skeleton(points, normals, vn, vt, pa, pb, kind, cell=20.0):
    pts = np.asarray(points, dtype=float)
    nrm = np.asarray(normals, dtype=float)
    return SimpleNamespace(
        bx=pts[:, 0],
        by=pts[:, 1],
        nx=nrm[:, 0],
        ny=nrm[:, 1],
        vn=np.asarray(vn, dtype=float),
        vt=np.asarray(vt, dtype=float),
        plate_a=np.asarray(pa),
        plate_b=np.asarray(pb),
        kind=np.asarray(kind, dtype=np.uint8),
        cell_km=cell,
    )


def _straight_ridge(rate=50.0, x=1000.0, length=800.0):
    ys = np.arange(0.0, length, 20.0)
    n = ys.size
    return _skeleton(
        np.column_stack([np.full(n, x), ys]),
        np.tile([1.0, 0.0], (n, 1)),
        np.full(n, rate),
        np.zeros(n),
        np.zeros(n, dtype=int),
        np.ones(n, dtype=int),
        np.full(n, TYPE_DIVERGENT),
    )


def test_ridge_symmetry_and_rate_asymmetry():
    sk = _straight_ridge(rate=50.0)
    pieces = build_ridge_pieces(sk, "Sym1", mean_length_km=2000.0)
    y = np.full(6, 400.0)
    x = np.array([700.0, 800.0, 900.0, 1100.0, 1200.0, 1300.0])
    plate = np.array([0, 0, 0, 1, 1, 1])
    fa = flowline_age(pieces, x, y, plate)
    axis = np.array([pieces.ax[pieces.piece == k][0] for k in fa.piece])
    expect = np.abs(x - axis) / 25.0  # half-rate 25 mm/yr = 25 km/Ma
    assert np.allclose(fa.age_ma, expect, atol=1e-6)
    assert fa.age_ma[0] > fa.age_ma[2] and fa.age_ma[5] > fa.age_ma[3]  # older away from the axis
    slow = flowline_age(build_ridge_pieces(_straight_ridge(rate=20.0), "Sym1", 2000.0), x, y, plate)
    assert np.all(slow.age_ma > fa.age_ma)  # slower spreading: older at the same distance


def test_transform_offset_makes_a_fracture_zone_parallel_to_spreading():
    # two ridge segments offset by 200 km along the spreading direction, joined by a transform
    seg1 = _straight_ridge(x=1000.0, length=400.0)
    seg2 = _straight_ridge(x=1200.0, length=400.0)
    seg2.by = seg2.by + 460.0
    tr_y = np.arange(400.0, 460.0, 20.0)
    sk = _skeleton(
        np.concatenate(
            [
                np.column_stack([seg1.bx, seg1.by]),
                np.column_stack([np.full(tr_y.size, 1100.0), tr_y]),
                np.column_stack([seg2.bx, seg2.by]),
            ]
        ),
        np.concatenate(
            [
                np.tile([1.0, 0.0], (seg1.bx.size, 1)),
                np.tile([0.0, 1.0], (tr_y.size, 1)),
                np.tile([1.0, 0.0], (seg2.bx.size, 1)),
            ]
        ),
        np.concatenate(
            [np.full(seg1.bx.size, 50.0), np.zeros(tr_y.size), np.full(seg2.bx.size, 50.0)]
        ),
        np.concatenate([np.zeros(seg1.bx.size), np.full(tr_y.size, 50.0), np.zeros(seg2.bx.size)]),
        np.zeros(seg1.bx.size * 2 + tr_y.size, dtype=int),
        np.ones(seg1.bx.size * 2 + tr_y.size, dtype=int),
        np.concatenate(
            [
                np.full(seg1.bx.size, TYPE_DIVERGENT),
                np.full(tr_y.size, TYPE_TRANSFORM),
                np.full(seg2.bx.size, TYPE_DIVERGENT),
            ]
        ),
    )
    pieces = build_ridge_pieces(sk, "Fz1", mean_length_km=2000.0)
    assert pieces.count == 2
    # far from both axes on plate 1 (x > axes): ages on the two sides of y = 430 differ by
    # the offset / half-rate, and the step runs along x (the spreading direction)
    for x in (1600.0, 2200.0):
        a = flowline_age(pieces, [x, x], [380.0, 480.0], [1, 1])
        assert a.piece[0] != a.piece[1]
        assert abs((a.age_ma[0] - a.age_ma[1]) - (pieces.ax.max() - pieces.ax.min()) / 25.0) < 0.5
    # between the axes the crust of plate 0 and plate 1 meet at the active transform
    mid = flowline_age(pieces, [1100.0, 1100.0], [300.0, 600.0], [0, 1])
    assert np.all(mid.age_ma < 10.0)


def test_trench_footprint_conserves_cross_section_and_resolution():
    d = np.linspace(-400.0, 400.0, 8001)
    for spacing in (25.0, 75.0):
        f = footprint(d, 22.0, spacing)
        area = np.trapezoid(f, d) if hasattr(np, "trapezoid") else np.trapz(f, d)
        assert area == pytest.approx(22.0 * np.sqrt(2.0 * np.pi), rel=1e-3)
    assert footprint(0.0, 22.0, 75.0) < footprint(0.0, 22.0, 25.0) < 1.0


def test_passive_margin_has_a_rise_active_does_not_and_profile_is_c1():
    u = np.linspace(0.0, 800.0, 8001)
    basement = np.full(u.size, -5500.0)
    ones, zeros = np.ones(u.size), np.zeros(u.size)
    passive, ws_p, wr_p = margins.margin_profile(u, -200.0, basement, (ones, zeros, zeros), 1.0)
    active, ws_a, wr_a = margins.margin_profile(u, -200.0, basement, (zeros, ones, zeros), 1.0)
    assert wr_p[0] > 200.0 and wr_a[0] < 40.0
    # the passive slope foot is far shallower than the abyss (a rise follows); the active
    # slope falls straight to the basement
    foot_p = passive[np.searchsorted(u, ws_p[0])]
    foot_a = active[np.searchsorted(u, ws_a[0])]
    assert foot_p > -3500.0 and foot_a == pytest.approx(-5500.0, abs=5.0)
    for prof in (passive, active):
        assert np.all(np.diff(prof) <= 1e-9)  # monotone seaward
        g = np.diff(prof) / np.diff(u)
        assert np.max(np.abs(np.diff(g))) < 2.0  # C1: no kink in the gradient
    # widths are kilometres: the same margin on another hex size is the same curve
    coarse = margins.margin_profile(
        u[::10], -200.0, basement[::10], (ones[::10], zeros[::10], zeros[::10]), 1.0
    )[0]
    assert np.allclose(coarse, passive[::10])


def test_guyot_only_where_the_top_reached_the_surface():
    on = np.ones(4, dtype=bool)
    crust_age = np.array([60.0, 60.0, 60.0, 60.0])
    vol_age = np.array([40.0, 40.0, 1.0, 40.0])
    summit_now = np.array([-900.0, -3500.0, -800.0, 50.0])
    codes = seamount_types(on, crust_age, vol_age, summit_now, 2500.0, (60.0, 80.0))
    # 1st: sank ~1150 m since birth (20 -> 60 Ma crust), its top was at the surface -> guyot
    # 2nd: never reached the surface -> pointed seamount; 3rd: too young; 4th: an island
    assert list(codes) == [2, 1, 1, 3]


def test_canyons_run_downhill_need_supply_and_close_the_budget():
    layout = _layout(30, 20, 20.0)
    hx, hy = layout.centers()
    z = np.where(hx < 100.0, 50.0, -120.0 - 12.0 * (hx - 100.0))  # coast, then a slope seaward
    ocean = z < 0.0
    mouth = np.zeros(z.shape, dtype=bool)
    mouth[10, 5] = True
    nbr = hex_graph(layout).nbr
    none = submarine_sediment(z, ocean, mouth, np.zeros(z.shape), nbr, layout, 1e6)
    assert not none["rows"] and float(none["fan_m"].sum()) == 0.0  # no supply: no fan
    supply = np.zeros(z.shape)
    supply[10, 5] = 5e7
    res = submarine_sediment(z, ocean, mouth, supply, nbr, layout, 1e6)
    row = res["rows"][0]
    path = row["path_hexes"]
    zf = z.ravel()
    assert all(zf[a] > zf[b] for a, b in zip(path, path[1:], strict=False))  # strictly downhill
    assert row["is_canyon"]
    assert abs(res["budget"]["closure_error_m3"]) < 1e-3 * res["budget"]["fan_deposited_m3"]


# -- 9.2 water and basins -----------------------------------------------------------------------
def _bowl(n=31, floor=-500.0, sill=100.0, channel=False):
    """A sea at x < 5 and a closed basin with a +100 m rim; optionally a low channel."""
    yy, xx = np.mgrid[0:n, 0:n]
    z = np.full((n, n), 300.0)
    z[:, :5] = -2000.0  # world ocean
    r = np.hypot(xx - 20, yy - 15)
    z[r < 7] = floor
    z[(r >= 7) & (r < 9)] = sill
    if channel:
        z[15, 5:14] = -50.0
    return z


def test_closed_basin_below_sea_level_is_not_ocean():
    z = _bowl()
    graph = grid_graph(*z.shape, 1000.0)
    sources = np.zeros(z.shape, dtype=bool)
    sources[:, 0] = True
    ocean = flood(z.ravel() <= 0.0, sources.ravel(), graph.nbr).reshape(z.shape)
    assert ocean[:, :5].all()
    assert not ocean[15, 20]  # the -500 m bowl behind the +100 m sill
    # a lake with its own level of -430 m: 70 m of water over the floor
    lake_level = np.where(
        np.hypot(*np.mgrid[0:31, 0:31][::-1] - np.array([20, 15])[:, None, None]) < 7,
        -430.0,
        np.nan,
    )
    w = water_layers(z, 0.0, ocean, lake_level)
    assert w["inland_water_mask"][15, 20] and not w["ocean_mask"][15, 20]
    assert w["water_depth_m"][15, 20] == pytest.approx(70.0)
    assert w["water_surface_level_m"][15, 20] == pytest.approx(-430.0)
    parts = w["ocean_mask"].astype(int) + w["inland_water_mask"] + w["dry_land_mask"]
    assert np.all(parts == 1)
    # dried out: dry land below the sea level
    dry = water_layers(z, 0.0, ocean, None)
    assert dry["dry_land_mask"][15, 20] and dry["below_sea_level_land_mask"][15, 20]
    fr = land_fractions(z, 0.0, ocean, dry["inland_water_mask"])
    assert fr["non_ocean_fraction"] > fr["above_datum_fraction"]
    # a real low passage to the sea floods the basin at 0 m
    zc = _bowl(channel=True)
    oc = flood(zc.ravel() <= 0.0, sources.ravel(), graph.nbr).reshape(zc.shape)
    assert oc[15, 20]


def test_two_basins_only_the_connected_one_gets_the_sea():
    z = _bowl(channel=True)
    yy, xx = np.mgrid[0:31, 0:31]
    z[np.hypot(xx - 20, yy - 4) < 3] = -300.0  # a second, closed basin
    z[(np.hypot(xx - 20, yy - 4) >= 3) & (np.hypot(xx - 20, yy - 4) < 4)] = 80.0
    graph = grid_graph(*z.shape, 1000.0)
    sources = np.zeros(z.shape, dtype=bool)
    sources[:, 0] = True
    ocean = flood(z.ravel() <= 0.0, sources.ravel(), graph.nbr).reshape(z.shape)
    assert ocean[15, 20] and not ocean[4, 20]


def test_world_ocean_needs_size_and_a_cut_bay_stays_ocean():
    world = np.full((60, 80), 500.0)
    world[:, :30] = -4000.0  # big ocean (30 x 60 cells of 100 km = 18 million km^2)
    world[20:24, 30:60] = -50.0  # a long bay reaching inland
    world[45:50, 60:66] = -100.0  # a small closed inland sea
    ocean = world_ocean(world, 0.0, 100.0, min_area_km2=1.5e6)
    assert ocean[22, 55] and not ocean[47, 62]
    # a window that shows only the inner end of the bay: no open water in view, but its
    # hexes are world ocean - the cut does not turn the bay into a lake
    window = world[18:26, 50:60]
    src = ocean[18:26, 50:60] & (window <= 0.0)
    g = grid_graph(*window.shape, 100.0)
    wet = flood(window.ravel() <= 0.0, src.ravel(), g.nbr).reshape(window.shape)
    assert wet[2:6, :].all()


def test_priority_flood_keeps_the_physical_surface():
    z = _bowl()
    before = z.copy()
    graph = grid_graph(*z.shape, 1000.0)
    outlet = np.zeros(z.size, dtype=bool)
    outlet[:: z.shape[1]] = True
    r = route(z.ravel(), outlet, graph)
    assert np.array_equal(z, before)  # routing never fills the real depression
    # the water level of the depression is its lowest way out: over the 300 m plateau
    assert r.routing_m.reshape(z.shape)[15, 20] == pytest.approx(300.0)
    assert z[15, 20] == -500.0


def _two_plate_layout(vb, n=36):
    layout = _layout(n, n, 20.0)
    hx, hy = layout.centers()
    plate = (hx > hx.mean()).astype(np.int32)
    motion = SimpleNamespace(
        velocity_at=lambda pid, x, y: (
            np.where(np.asarray(pid) == 1, vb[0], 0.0) * np.ones_like(np.asarray(x, dtype=float)),
            np.where(np.asarray(pid) == 1, vb[1], 0.0) * np.ones_like(np.asarray(x, dtype=float)),
        )
    )
    return layout, plate, motion


def test_pull_apart_only_at_releasing_steps():
    # plate 1 east of plate 0 slides north (strike-slip) and opens slightly: releasing
    layout, plate, motion = _two_plate_layout((3.0, -40.0))
    cont = np.ones(plate.shape, dtype=bool)
    plans = basins.plan_boundary_basins(layout, plate, cont, motion, "Pa1", 20.0)
    kinds = {p["basin_type"] for p in plans}
    assert "strike_slip_pull_apart" in kinds
    # the same slip with slight convergence: restraining, no pull-apart
    layout, plate, motion = _two_plate_layout((-3.0, -40.0))
    plans = basins.plan_boundary_basins(layout, plate, cont, motion, "Pa1", 20.0)
    assert not any(p["basin_type"] == "strike_slip_pull_apart" for p in plans)
    # pure opening: a rift
    layout, plate, motion = _two_plate_layout((25.0, 0.0))
    plans = basins.plan_boundary_basins(layout, plate, cont, motion, "Pa1", 20.0)
    assert any(p["basin_type"] == "rift_graben" for p in plans)


def test_rift_is_elongated_with_fault_walls_and_shoulders():
    plan = {
        "basin_type": "rift_graben",
        "center_km": [0.0, 0.0],
        "axis": [0.0, 1.0],
        "half_length_km": 300.0,
        "segment_length_km": 120.0,
        "half_width_km": 25.0,
        "subsidence_m": 3000.0,
        "fill_fraction": 0.5,
        "flank_uplift_fraction": 0.25,
        "polarity": [1, -1] * 32,
    }
    x = np.linspace(-150.0, 150.0, 301)
    sub, fill, flank, _w = basins.evaluate([plan], x, np.zeros(x.size), x, np.zeros(x.size))
    assert sub.max() > 2000.0 and flank.max() > 300.0
    assert np.all(fill <= sub)
    # one wall steep (the border fault), the other gentle (the hinge)
    left, right = sub[:150], sub[151:][::-1]
    assert np.abs(np.argmax(left > 0.5 * sub.max()) - np.argmax(right > 0.5 * sub.max())) > 3
    # elongated: along the axis it stays deep for hundreds of km
    y = np.linspace(-280.0, 280.0, 57)
    along, *_ = basins.evaluate([plan], np.zeros(y.size), y, np.zeros(y.size), y)
    assert (along > 0.3 * sub.max()).mean() > 0.6


def test_sag_is_broad_gentle_and_bedrock_differs_from_surface():
    plan = basins.plan_sags(
        np.full((40, 40), 2000.0),
        *np.meshgrid(np.arange(40) * 100.0, np.arange(40) * 100.0),
        np.full((40, 40), 5000.0),
        "Sag1",
        density=20.0,
        cell_km=100.0,
    )[0]
    cx, cy = plan["center_km"]
    r = np.linspace(0.0, 1500.0, 151)
    sub, fill, _f, _w = basins.evaluate(
        [plan], cx + r, np.full(r.size, cy), cx + r, np.full(r.size, cy)
    )
    surface_drop = sub - fill
    if plan["under_filled"]:
        assert 150.0 < surface_drop[0] < 0.5 * sub[0]  # the Caspian type: a real depression
    else:
        assert np.all(surface_drop == 0.0)  # filled to the brim: bedrock deep, surface flat
    assert sub[0] > 1000.0 and not np.array_equal(sub, surface_drop)  # bedrock != surface
    width = r[np.argmax(sub < 0.5 * sub[0])]
    assert width > 200.0  # broad
    assert np.max(np.abs(np.diff(surface_drop)) / 10.0) < 5.0  # gentle: < 5 m per km


def test_foredeep_scales_with_load_and_vanishes_without_it():
    h = np.zeros((80, 80))
    h[30:50, 35:45] = 3000.0
    w, alpha = basins.flexure(h, 20.0, 25.0)
    zero, _ = basins.flexure(np.zeros_like(h), 20.0, 25.0)
    assert np.all(zero == 0.0)
    assert 40.0 < alpha < 150.0
    profile = w[40, 45:]  # outward from the load edge
    assert profile[1] > 0.0  # foredeep next to the mountains
    assert profile.min() < 0.0  # forebulge further out
    assert np.argmin(profile) > 2
    w2, _ = basins.flexure(2.0 * h, 20.0, 25.0)
    assert np.allclose(w2, 2.0 * w)  # linear in the load


# -- 9.1 massifs -----------------------------------------------------------------------------------
def test_prominence_of_two_peaks():
    x = np.linspace(0.0, 10.0, 101)
    z = 1000.0 * np.exp(-((x - 3.0) ** 2)) + 700.0 * np.exp(-((x - 7.0) ** 2))
    z = np.tile(z, (3, 1))
    g = grid_graph(*z.shape, 1000.0)
    prom = prominence(z, np.ones(z.shape, dtype=bool), g.nbr).reshape(z.shape)
    col = z[1, 30:71].min()
    k_low = 70
    # the summit is one of the equal hexes of that column
    assert prom[:, k_low].max() == pytest.approx(z[1, k_low] - col, abs=1.0)
    assert prom.max() == pytest.approx(z.max() - z.min(), abs=1.0)


# -- integration on a generated world ----------------------------------------------------
def test_terrain_on_a_generated_world(hist_project):
    p = hist_project
    t = p.terrain
    assert t is not None and t.validation["ok"], t.validation
    assert p.hydro.validation["ok"], p.hydro.validation
    L = p.hydro.layers
    parts = L["ocean_mask"].astype(int) + L["inland_water_mask"] + L["dry_land_mask"]
    assert np.all(parts == 1)
    assert np.all(L["water_depth_m"] >= 0.0)
    assert float(t.layers["sediment_thickness_m"].min()) >= -1e-3
    marine = t.layers["marine_floor_mask"]
    age = t.layers["oceanic_crust_age_ma"][marine]
    depth = t.layers["basement_depth_m"][marine]
    # younger floor is shallower (the age-depth law holds on the generated map)
    young, old = depth[age < np.percentile(age, 20)], depth[age > np.percentile(age, 80)]
    assert np.median(young) + 500.0 < np.median(old)
    # every trench axis lies on a subducting oceanic plate or right at the boundary
    ax = t.layers["trench_axis_mask"]
    assert np.all(p.hist.layers["trench_side"][ax] != 0)
    assert set(t.plans) >= {"basins", "massifs"}
    rows = cross_section(
        p, (p.layout.x0 + 100.0, p.layout.y0 + 100.0), (p.layout.x0 + 3000.0, p.layout.y0 + 2500.0)
    )
    assert rows and all(r["height_m"] >= r["bedrock_m"] for r in rows)


def test_sea_level_change_only_reclassifies_water(hist_project):
    from hexworld import generate

    p = hist_project
    cfg = p.config.with_(
        sea=type(p.config.sea)(sea_mode="fixed_sea_level_m", fixed_sea_level_m=150.0)
    )
    q = generate(cfg, reuse=p)
    assert "v46_terrain" in q.stats["reused_stages"]  # the geology of v4.6 is not recomputed
    assert np.array_equal(
        q.terrain.layers["surface_elevation_m"], p.terrain.layers["surface_elevation_m"]
    )
    assert np.array_equal(q.hist.layers["crust_age_ma"], p.hist.layers["crust_age_ma"])
    assert q.hydro.layers["ocean_mask"].sum() >= p.hydro.layers["ocean_mask"].sum()


def test_terrain_save_load_roundtrip(hist_project, tmp_path):
    from hexworld.io import load_project, save_project

    save_project(hist_project, tmp_path / "w")
    q = load_project(tmp_path / "w")
    assert q.terrain is not None and q.hydro is not None
    for k, v in hist_project.terrain.layers.items():
        assert np.array_equal(v, q.terrain.layers[k], equal_nan=True), k
    assert q.terrain.plans["basins"] == hist_project.terrain.plans["basins"]
    assert np.array_equal(q.hydro.layers["elevation_m"], hist_project.hydro.layers["elevation_m"])


def test_terrain_config_validation():
    with pytest.raises(ValueError):
        Config(terrain=TerrainParams(basins=("volcanic_caldera",)))
    with pytest.raises(ValueError):
        Config(terrain=TerrainParams(age_depth_transition_ma=(80.0, 60.0)))
