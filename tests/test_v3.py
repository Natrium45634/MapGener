"""v3.0: geological history, mountains, volcanism, limited erosion (TZ 3.0, section 8)."""

import json
import os
import shutil
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from hexworld import (
    Config,
    derive_landmasses,
    generate,
    generate_history,
    get_layer,
    history_events,
    load_project,
    sample_points,
    save_project,
    select_epoch,
    views,
)
from hexworld.cli import main
from hexworld.config import HistoryParams, MountainParams
from hexworld.mountains import (
    BackendUnavailable,
    MountainRequest,
    PmgeBackend,
    ReferenceBackend,
    build_plan,
    get_backend,
    limit_elevation,
)
from hexworld.stages.history import (
    MM_YR_TO_KM_MA,
    MM_YR_TO_M_MA,
    Baseline,
    HistoryGrid,
    HistorySimulator,
    Kinematics,
    run_history,
)
from hexworld.stages.volcanism import Hotspot, VolcanoBuilder

V2_FIXTURE = Path(__file__).parent / "data" / "v2_project"
GRID = HistoryGrid(0.0, 0.0, 50.0, 120, 60)


def two_plates(v_left, v_right, cont_left, cont_right, duration=20.0, hotspots=(), **history):
    """A synthetic world: plate 0 west of x = 3000 km, plate 1 east of it."""
    gx, gy = GRID.centers()
    plate = (gx > 3000.0).astype(np.int32)
    cont = np.where(plate == 0, cont_left, cont_right).astype(float)
    base = Baseline(
        grid=GRID,
        plate=plate,
        continental=cont,
        elevation=np.where(cont > 0.5, 300.0, -4000.0),
        inherited_relief=np.zeros(GRID.shape),
        crust_age=np.where(cont > 0.5, 1000.0, 80.0),
        craton=np.full(GRID.shape, 0.5),
    )
    kin = Kinematics([v_left, v_right], [0.0, 0.0], [[0.0, 0.0], [0.0, 0.0]])
    params = HistoryParams(duration_ma=duration, time_step_ma=1.0, checkpoint_stride=5, **history)
    sim = HistorySimulator(base, kin, params, "Synthetic")
    builder = VolcanoBuilder("Synthetic", kin, list(hotspots))
    checkpoints, events, balance, info = run_history(
        sim, observer=lambda snap, t: builder.step(snap, GRID, snap.continental, t, 1.0)
    )
    t_end, state = checkpoints[-1]
    snap = sim.snapshot(t_end, state)
    return {
        "sim": sim,
        "state": state,
        "snap": snap,
        "surface": sim.surface(snap, state),
        "events": events,
        "checkpoints": checkpoints,
        "volcanoes": builder.set,
        "balance": balance,
        "info": info,
        "kin": kin,
    }


# -- units -----------------------------------------------------------------------
def test_units_one_mm_per_year_is_one_km_per_million_years():
    assert MM_YR_TO_KM_MA == 1.0 and MM_YR_TO_M_MA == 1000.0
    kin = Kinematics([[40.0, 0.0]], [0.0], [[0.0, 0.0]])
    x, y = kin.forward(np.array([0]), np.array([100.0]), np.array([50.0]), 10.0)
    assert x[0] == pytest.approx(500.0) and y[0] == pytest.approx(50.0)  # 40 mm/yr * 10 Ma
    back = kin.inverse(np.array([0]), x, y, 10.0)
    assert back[0][0] == pytest.approx(100.0)
    # a rotating plate: exact rigid motion keeps distances
    spin = Kinematics([[10.0, -5.0]], [0.2], [[300.0, 200.0]])
    px, py = spin.forward(np.zeros(2, int), np.array([0.0, 400.0]), np.array([0.0, 0.0]), 25.0)
    assert np.hypot(px[1] - px[0], py[1] - py[0]) == pytest.approx(400.0)
    vx, vy = spin.velocity_at(np.array([0]), np.array([300.0]), np.array([200.0]))
    assert (vx[0], vy[0]) == pytest.approx((10.0, -5.0))


# -- 2. controlled physical cases ----------------------------------------------------
def test_continents_converging_build_an_uplift_zone_without_arc_volcanoes():
    r = two_plates((30, 0), (-30, 0), 1, 1)
    snap, state = r["snap"], r["state"]
    assert (snap.overlap_kind == 1).sum() > 100 and not (snap.overlap_kind == 2).any()
    assert state.orogenic.max() > 2000.0
    row = r["surface"][30]
    assert row.max() > 2000.0 and row.max() > row[:20].mean() + 1500.0
    assert [e["type"] for e in r["events"]] == ["collision_start"]
    assert not r["volcanoes"].volcanoes  # collision alone is not a volcano source
    assert snap.arc_zone.max() == 0.0


def test_ocean_under_continent_gives_trench_and_arc_on_the_right_side():
    r = two_plates((30, 0), (-30, 0), 0, 1)  # oceanic plate 0 goes under continental plate 1
    snap = r["snap"]
    assert (snap.overlap_kind == 2).any() and not (snap.overlap_kind == 1).any()
    over = snap.owner[snap.overlap_kind == 2]
    under = snap.under[snap.overlap_kind == 2]
    assert set(over.tolist()) == {1} and set(under.tolist()) == {0}  # continent overrides
    row = r["surface"][30]
    trench = int(np.argmin(row))
    assert row[trench] < -4300.0  # deeper than the ordinary ocean floor
    assert snap.owner[30, trench - 2] == 0  # the trench borders the subducting plate
    arc = snap.arc_zone[30]
    arc_at = int(np.argmax(arc))
    assert arc.max() > 0.3 and snap.owner[30, arc_at] == 1 and arc_at > trench
    assert 100.0 < (arc_at - trench) * GRID.cell < 400.0  # arc offset from the trench
    arcs = [v for v in r["volcanoes"].volcanoes if v.kind == "arc"]
    assert arcs and {v.plate for v in arcs} == {1} and {v.partner for v in arcs} == {0}
    assert [e["type"] for e in r["events"]] == ["subduction_start"]
    assert r["events"][0]["plates"] == [1, 0]


def test_diverging_plates_open_an_elongated_rift_then_an_ocean():
    r = two_plates((-30, 0), (30, 0), 1, 1)
    gap = ~r["snap"].covered
    rows, cols = np.nonzero(gap)
    assert rows.max() - rows.min() + 1 == GRID.ny  # runs along the whole boundary
    assert (cols.max() - cols.min() + 1) * GRID.cell == pytest.approx(1200.0, abs=110.0)
    age = r["snap"].new_crust_age[30]
    centre = 60
    assert age[centre] < 3.0 and age[cols.min() + 1] > 15.0  # young at the axis, old at margins
    assert r["surface"][30, centre] < -2000.0
    assert not r["volcanoes"].volcanoes
    assert [e["type"] for e in r["events"]] == ["rifting_start", "ocean_opening"]
    # early on it is still a continental rift, not an ocean
    early = two_plates((-30, 0), (30, 0), 1, 1, duration=3.0)
    assert early["snap"].rift.any() and early["surface"][30].min() > -2500.0
    assert [e["type"] for e in early["events"]] == ["rifting_start"]


def test_transform_motion_makes_no_volcanic_chain_and_no_orogen():
    r = two_plates((0, 30), (0, -30), 1, 1)
    assert not (r["snap"].under >= 0).any() and r["snap"].covered.all()
    assert not r["volcanoes"].volcanoes and r["snap"].arc_zone.max() == 0.0
    assert r["state"].orogenic.max() == 0.0 and r["events"] == []


def test_hotspot_under_a_moving_plate_builds_an_age_progressive_chain():
    spot = Hotspot(0, 1500.0, 1500.0, 1.0, 2.0, 0.5)
    r = two_plates((40, 0), (40, 0), 0, 0, duration=20.0, hotspots=[spot])
    chain = [v for v in r["volcanoes"].volcanoes if v.kind == "hotspot"]
    assert len(chain) >= 8
    x, y, idx = r["volcanoes"].positions(r["kin"], 20.0)
    ages = np.array([20.0 - r["volcanoes"].volcanoes[int(i)].birth_ma for i in idx])
    # the plate moved east at 40 mm/yr: a volcano of age a sits 40 * a km east of the hotspot
    assert np.allclose(x - spot.x, 40.0 * ages, atol=1e-6) and np.allclose(y, spot.y)
    order = np.argsort(x)
    assert np.all(np.diff(ages[order]) > 0)  # age grows monotonically along the chain
    heights = r["volcanoes"].heights(idx, 20.0, 1.0)
    assert heights[np.argmax(ages)] < heights[np.argsort(ages)[2]]  # old edifices are worn
    assert r["volcanoes"].is_active(int(idx[np.argmin(ages)]), 20.0)
    assert not r["volcanoes"].is_active(int(idx[np.argmax(ages)]), 20.0)


# -- 1. it is a history ------------------------------------------------------------------
def test_processes_off_leave_the_baseline_untouched():
    r = two_plates((0, 0), (0, 0), 1, 0)
    assert r["state"].uplift.max() == 0.0 and r["state"].orogenic.max() == 0.0
    assert r["snap"].covered.all() and r["events"] == []
    still = two_plates((30, 0), (-30, 0), 1, 1, uplift_scale=0.0, erosion_intensity=0.0)
    assert still["state"].uplift.max() == 0.0 and still["state"].erosion.max() == 0.0


def test_snapshots_of_one_run_differ(hist_project):
    hist = hist_project.hist
    assert len(hist.epochs) >= 4 and hist.epochs[0] == 0.0 and hist.epochs[-1] == 16.0
    middle = select_epoch(hist_project, hist.epochs[len(hist.epochs) // 2])
    assert middle.hist.epoch_time_ma not in (0.0, 16.0)
    assert "geo_history" in middle.stats["reused_stages"]  # no re-simulation
    a, b = middle.hist.layers, hist.layers
    assert not np.array_equal(a["plate_id_epoch"], b["plate_id_epoch"])  # geometry moved on
    assert np.abs(a["elevation_m"] - b["elevation_m"]).max() > 500.0
    assert b["new_crust"].sum() > a["new_crust"].sum() > 0
    assert b["uplift_total_m"].max() > a["uplift_total_m"].max()
    first = select_epoch(hist_project, "initial")
    assert first.hist.epoch_time_ma == 0.0
    assert not first.hist.layers["new_crust"].any()
    assert np.array_equal(first.hist.layers["plate_id_epoch"], hist_project.layer("plate_id"))
    assert first.hist.layers["uplift_total_m"].max() == 0.0
    assert [e for e in first.hist.events if not e.get("derived")] == []
    # events carry time, place, participants and intensity
    raw = [e for e in hist.events if not e.get("derived")]
    assert raw and {"collision_start", "subduction_start", "rifting_start"} & {
        e["type"] for e in raw
    }
    for e in raw:
        assert 0.0 <= e["time_ma"] <= 16.0 and len(e["plates"]) == 2 and "x_km" in e


def test_history_disabled_gives_stable_layers(hist_config):
    flat = generate(
        replace(
            hist_config, history=HistoryParams(enabled=False), mountains=MountainParams(detail=0.0)
        )
    )
    hist = flat.hist
    assert hist.epochs == [0.0] and hist.events == [] and not hist.volcanoes
    assert np.allclose(hist.layers["elevation_m"], flat.geo.layers["macro_elevation_m"], atol=0.01)
    assert float(np.abs(hist.layers["delta_elevation_m"]).max()) < 0.01
    assert np.array_equal(hist.layers["land_mask"], flat.geo.layers["land_mask"])
    assert not hist.layers["new_crust"].any()


# -- 3. age of mountains ---------------------------------------------------------------------
def test_long_weathering_wears_mountains_down_and_reuplift_renews_them():
    fresh = two_plates((30, 0), (-30, 0), 1, 1, erosion_intensity=0.0)
    worn = two_plates((30, 0), (-30, 0), 1, 1, erosion_intensity=3.0)
    assert worn["surface"].max() < fresh["surface"].max() - 400.0
    assert worn["state"].erosion.max() > 1000.0 and fresh["state"].erosion.max() == 0.0
    assert worn["state"].sediment.max() > 50.0 and worn["state"].sediment.min() >= 0.0
    total = sum(b["eroded_m_cells"] for b in worn["balance"])
    kept = sum(b["deposited_m_cells"] for b in worn["balance"])
    assert 0.0 < kept <= total  # diagnosable balance: nothing is created

    def plan_of(run, extra_age=0.0):
        state, snap = run["state"], run["snap"]
        mat = np.maximum(snap.material, 0)
        built = state.orogenic.ravel()[mat]
        share = np.clip(state.erosion.ravel()[mat] / (built + 1.0), 0.0, 0.9)
        relief = np.where(snap.covered, built * (1 - share), 0.0)
        age = np.where(snap.covered, 20.0 - state.orogen_last.ravel()[mat], np.nan) + extra_age
        kind = state.orogen_kind.ravel()[mat]
        return build_plan(
            GRID, relief, kind, age, share, run["surface"], MountainParams(), "Synthetic", 20.0
        )

    young, old = plan_of(fresh), plan_of(worn, extra_age=160.0)
    assert young.systems and old.systems
    y, o = young.systems[0], old.systems[0]
    assert y.maturity < o.maturity and y.sharpness > o.sharpness and y.relief_gain > o.relief_gain
    assert y.morphology in ("young_sharp", "plateau") and o.morphology in (
        "old_rounded",
        "mature",
        "plateau",
    )
    assert y.kind == "collision"
    # renewed mountain-building resets the age of an old belt
    renewed = two_plates((30, 0), (-30, 0), 1, 1)
    assert np.nanmax(renewed["state"].orogen_last) == pytest.approx(20.0)


# -- 4. structure of mountains ------------------------------------------------------------------
def test_mountain_plan_is_a_connected_structure(hist_project):
    plan = hist_project.hist.plan
    assert plan["systems"] and plan["ridges"] and plan["peaks"] and plan["passes"]
    peaks = {p["id"]: p for p in plan["peaks"]}
    ridges = {r["id"]: r for r in plan["ridges"]}
    for r in plan["ridges"]:
        pts = np.array(r["points_km"])
        steps = np.hypot(*np.diff(pts, axis=0).T)
        assert len(pts) >= 2 and steps.max() < 40.0  # an unbroken polyline
        if r["rank"] != "main":
            parent = np.array(ridges[r["parent"]]["points_km"])
            d = np.hypot(parent[:, 0] - pts[0, 0], parent[:, 1] - pts[0, 1]).min()
            assert d < 80.0  # a branch or spur starts on its parent ridge
    for item in plan["passes"]:
        a, b = (peaks[i] for i in item["between_peaks"])
        assert a["ridge"] == b["ridge"] == item["ridge"]  # a pass joins two peaks of one ridge
        assert item["crest"] < min(a["crest"], b["crest"])  # and is a saddle between them
    assert len({s["id"] for s in plan["systems"]}) == len(plan["systems"])
    assert any(r["rank"] == "spur" for r in plan["ridges"]) and plan["valleys"]


def test_reference_sampler_is_continuous_and_seamless(hist_project):
    hist = hist_project.hist
    from hexworld.mountains.plan import MountainPlan

    plan = MountainPlan.from_dict(hist.plan)
    system = max(hist.plan["systems"], key=lambda s: s["area_km2"])
    cx, cy = system["centroid_km"]
    request = MountainRequest(
        world_seed="Ab12",
        snapshot_id="t",
        world_bounds_km=(0, 0, 1, 1),
        plan=plan,
        uplift=lambda x, y: np.full(np.shape(x), 2000.0),
        gain=lambda x, y: np.full(np.shape(x), 0.8),
    )
    sampler = ReferenceBackend().sampler(request)
    main = next(
        r for r in hist.plan["ridges"] if r["system"] == system["id"] and r["rank"] == "main"
    )
    mid = np.array(main["points_km"])[len(main["points_km"]) // 2]
    cx, cy = float(mid[0]), float(mid[1])
    line = np.linspace(-300.0, 300.0, 2401)  # 0.25 km steps across the main ridge
    east = sampler.sample_points(cx + line, np.full(line.shape, cy)).relief_m
    south = sampler.sample_points(np.full(line.shape, cx), cy + line).relief_m
    assert max(np.ptp(east), np.ptp(south)) > 300.0
    for profile in (east, south):
        assert np.abs(np.diff(profile)).max() < 60.0  # no cliffs: smooth flanks and foothills
    # two adjacent "chunks" evaluated separately agree exactly on their shared edge
    edge_y = cy + np.linspace(-200.0, 200.0, 300)
    left = sampler.sample_points(np.full(edge_y.shape, cx), edge_y).relief_m
    grid_x, grid_y = np.meshgrid(cx + np.array([0.0, 5.0, 10.0]), edge_y)
    right = sampler.sample_points(grid_x, grid_y).relief_m[:, 0]
    assert np.array_equal(left, right)
    # valleys never cut a crest: the structure stays high on the ridge line itself
    main = next(
        r for r in hist.plan["ridges"] if r["system"] == system["id"] and r["rank"] == "main"
    )
    pts = np.array(main["points_km"])
    on_ridge = sampler.sample_points(pts[:, 0], pts[:, 1])
    # (near a junction another ridge may stand higher, so the crest is a lower bound)
    assert np.all(on_ridge.structure >= np.array(main["crest"]) - 1e-3)


def test_hex_values_agree_with_the_continuous_surface(hist_project):
    L = hist_project.hist.layers
    assert L["local_relief_m"].min() >= 0.0 and L["slope_m_per_km"].min() >= 0.0
    crest = L["mountain_class"] == 1
    assert crest.any() and L["mountain_system_id"][crest].min() >= 0
    assert np.median(L["local_relief_m"][L["mountain_system_id"] >= 0]) > np.median(
        L["local_relief_m"][L["mountain_system_id"] < 0]
    )
    cap = hist_project.config.mountains.max_elevation_m
    assert L["elevation_m"].max() <= cap
    # the ceiling is point-wise, not a rescale: low ground is untouched
    h = np.array([-4000.0, 0.0, 3000.0, 7000.0, 9000.0, 30000.0])
    capped = limit_elevation(h, 8800.0)
    assert np.array_equal(capped[:4], h[:4]) and np.all(capped <= 8800.0)
    assert capped[4] < capped[5] <= 8800.0


# -- 5. determinism --
def test_history_is_deterministic(hist_config, hist_project):
    again = generate(hist_config)
    assert again.hist.hist_hash() == hist_project.hist.hist_hash()
    assert again.hist.events == hist_project.hist.events
    assert again.hist.plan == hist_project.hist.plan
    assert again.hist.volcanoes == hist_project.hist.volcanoes
    other = generate(replace(hist_config, seed="ab12"))
    assert other.hist.hist_hash() != hist_project.hist.hist_hash()
    before = hist_project.hist.hist_hash()
    for spec in views.VIEWS.values():  # any drawing order, any tab: data stay the same
        if spec.status == "ready":
            for mode in reversed(spec.modes):
                views.view_rgb(hist_project, spec.id, mode.id)
                views.view_vectors(hist_project, spec.id, mode.id)
    assert hist_project.hist.hist_hash() == before


def test_dependencies_between_stages(hist_config, hist_project):
    base_v1 = hist_project.content_hashes()
    base_v2 = hist_project.geo.geo_hash()
    # mountain detail does not re-run the history
    sharper = generate(
        replace(hist_config, mountains=MountainParams(detail=1.6)), reuse=hist_project
    )
    assert "geo_history" in sharper.stats["reused_stages"]
    assert "v3_raster" not in sharper.stats["reused_stages"]
    assert sharper.hist.events == hist_project.hist.events
    assert not np.array_equal(
        sharper.hist.layers["elevation_m"], hist_project.hist.layers["elevation_m"]
    )
    # another history invalidates mountains and elevation but never the baseline
    longer = generate(
        replace(hist_config, history=replace(hist_config.history, erosion_intensity=2.0)),
        reuse=hist_project,
    )
    assert "geo_history" not in longer.stats["reused_stages"]
    for p in (sharper, longer):
        assert p.content_hashes() == base_v1 and p.geo.geo_hash() == base_v2
        assert p.geo.layers["macro_elevation_m"] is hist_project.geo.layers["macro_elevation_m"]
        assert p.layer("base_macro_elevation_m") is p.geo.layers["macro_elevation_m"]


# -- 6. backward compatibility --
def test_v2_project_gets_a_history_and_keeps_its_baseline(tmp_path):
    work = tmp_path / "proj"
    shutil.copytree(V2_FIXTURE, work)
    original = {p.name: p.read_bytes() for p in V2_FIXTURE.iterdir()}
    manifest_v2 = json.loads((V2_FIXTURE / "manifest.json").read_text(encoding="utf-8"))
    assert manifest_v2["schema_version"] == "hexworld-project-2"
    v2 = load_project(work)
    assert v2.geo is not None and v2.hist is None
    assert not views.VIEWS["relief"].availability(v2)[0]
    with pytest.raises(KeyError):
        v2.layer("uplift_total_m")
    v3 = generate_history(
        v2, replace(v2.config, history=HistoryParams(duration_ma=12.0, time_step_ma=2.0))
    )
    assert v3.hist is not None and v3.hist.validation["ok"], v3.hist.validation
    assert v3.content_hashes() == manifest_v2["content_hashes"]
    assert v3.geo.content_hashes() == manifest_v2["v2"]["content_hashes"]
    save_project(v3, work)
    for name in (
        "plates.json",
        "fields.npz",
        "boundaries.npz",
        "boundaries.json",
        "continental_fields.npz",
        "continents.json",
    ):
        assert (work / name).read_bytes() == original[name], name  # baseline files untouched
    assert (work / "manifest.v2.json").read_bytes() == original["manifest.json"]
    back = load_project(work)
    assert back.hist.hist_hash() == v3.hist.hist_hash()
    assert back.hist.events == v3.hist.events and back.hist.plan == v3.hist.plan
    assert back.hist.volcanoes == v3.hist.volcanoes and back.hist.epochs == v3.hist.epochs
    assert back.geo.geo_hash() == v3.geo.geo_hash()
    # an epoch can be opened after loading without simulating again, and matches a fresh one
    t_mid = back.hist.epochs[1]
    assert np.array_equal(
        select_epoch(back, t_mid).hist.layers["elevation_m"],
        select_epoch(v3, t_mid).hist.layers["elevation_m"],
    )


def test_stale_history_is_never_loaded_silently(tmp_path, hist_project):
    path = save_project(hist_project, tmp_path / "p")
    doc = json.loads((path / "history.json").read_text(encoding="utf-8"))
    doc["algorithm_version"] = "geohistory-0"
    (path / "history.json").write_text(json.dumps(doc), encoding="utf-8")
    stale = load_project(path, verify=False)
    assert stale.hist is None and stale.geo is not None  # baseline intact, history refused
    assert any("NOT loaded" in w for w in stale.warnings)
    with pytest.raises(Exception, match="manifest"):
        load_project(path)  # and with verification the tampering is detected


# -- 7. window edge, other resolution --
def test_two_windows_show_the_same_evolved_world(hist_project):
    g = hist_project.geometry
    pinned = replace(
        hist_project.config,
        simulation_bounds_km=tuple(g.simulation_bounds.as_list()),
        plates=replace(hist_project.config.plates, count=g.plate_count),
    )
    base = generate(pinned)
    assert base.hist.hist_hash() == hist_project.hist.hist_hash()
    fine = generate(
        replace(
            pinned, width=180, height=160, hex_spacing_km=20.0, viewport_origin_km=(1500.0, 2000.0)
        )
    )
    assert fine.hist.events == base.hist.events  # one world, one history
    x, y = fine.layout.centers()
    col, row = base.layout.world_to_offset(x, y)
    diff = np.abs(base.hist.layers["elevation_m"][row, col] - fine.hist.layers["elevation_m"])
    assert np.median(diff) < 80.0
    same_land = base.hist.layers["land_mask"][row, col] == fine.hist.layers["land_mask"]
    assert same_land.mean() > 0.95
    same_plate = base.hist.layers["plate_id_epoch"][row, col] == fine.hist.layers["plate_id_epoch"]
    assert same_plate.mean() > 0.95
    # ridges are world polylines: they are not cut at a window edge
    b = fine.geometry.viewport_bounds
    wb = base.geometry.viewport_bounds
    all_pts = np.vstack([np.array(r["points_km"]) for r in base.hist.plan["ridges"]])
    assert not wb.contains(all_pts[:, 0], all_pts[:, 1]).all()
    assert [r["points_km"] for r in fine.hist.plan["ridges"]] == [
        r["points_km"] for r in base.hist.plan["ridges"]
    ]
    layout, elev = get_layer(base, "macro_elevation_m", b.as_list(), hex_spacing_km=20.0)
    assert elev.shape == (layout.height, layout.width)


# -- 8. sea --
def test_sea_level_changes_only_the_masks_of_the_selected_surface(hist_project):
    previous = None
    for level in (-300.0, 0.0, 250.0, 800.0):
        p = derive_landmasses(hist_project, sea_level_m=level)
        h = p.hist
        assert h.sea_level_m == level == p.geo.sea_level_m and h.validation["ok"]
        assert {"geo_history", "mountain_plan", "v3_raster"} <= set(p.stats["reused_stages"])
        assert h.layers["elevation_m"] is hist_project.hist.layers["elevation_m"]
        assert (
            h.events == hist_project.hist.events
            and h.plan["ridges"] == hist_project.hist.plan["ridges"]
        )
        assert [v["id"] for v in h.volcanoes] == [v["id"] for v in hist_project.hist.volcanoes]
        land = h.layers["land_mask"]
        assert np.array_equal(land, h.layers["elevation_m"] > np.float32(level))
        if previous is not None:
            assert not np.any(land & ~previous)
        previous = land
    assert hist_project.layer("land_mask") is hist_project.hist.layers["land_mask"]
    assert hist_project.baseline_layer("land_mask") is hist_project.geo.layers["land_mask"]


# -- 11. PMGE ------------------------------------------------------------------------
def test_backend_choice_is_explicit_and_honest(hist_config, hist_project, monkeypatch):
    backend = hist_project.hist.summary["mountain_backend"]
    assert backend["name"] in ("reference", "pmge")
    if not backend["is_pmge"]:
        assert any("PMGE library was not found" in w for w in hist_project.hist.warnings)
    ref = generate(
        replace(hist_config, mountains=MountainParams(backend="reference")), reuse=hist_project
    )
    assert ref.hist.summary["mountain_backend"]["is_pmge"] is False
    assert ReferenceBackend().capabilities()["missing_vs_pmge"]
    # a missing library is reported, never faked
    monkeypatch.setenv("HEXWORLD_PMGE_LIBRARY", "/nonexistent/libpmge.so")
    monkeypatch.setattr(
        "hexworld.mountains.pmge_binding._candidates", lambda: ["/nonexistent/libpmge.so"]
    )
    with pytest.raises(BackendUnavailable):
        PmgeBackend()
    with pytest.raises(BackendUnavailable):
        generate(replace(hist_config, mountains=MountainParams(backend="pmge")), reuse=hist_project)
    assert get_backend("auto").name == "reference"  # auto falls back, and says so
    assert get_backend("reference").name == "reference"


def test_pmge_backend_through_the_c_abi(hist_config, hist_project):
    from hexworld.mountains.pmge_binding import find_library

    if not find_library():
        pytest.skip("PMGE library not built (cmake -S pmge -B pmge/build)")
    ref = generate(
        replace(hist_config, mountains=MountainParams(backend="reference")), reuse=hist_project
    )
    pm = generate(
        replace(hist_config, mountains=MountainParams(backend="pmge")), reuse=hist_project
    )
    assert pm.hist.validation["ok"], pm.hist.validation
    info = pm.hist.summary["mountain_backend"]
    assert info["is_pmge"] and info["abi"] == 1 and "pmge" in info["library"]
    assert "geo_history" in pm.stats["reused_stages"]  # the backend does not touch history
    assert pm.hist.stage_keys["mountain_plan"] != ref.hist.stage_keys["mountain_plan"]
    a, b = ref.hist.layers["elevation_m"], pm.hist.layers["elevation_m"]
    mountains = ref.hist.layers["mountain_system_id"] >= 0
    assert np.abs(a - b)[~mountains].max() < 1.0  # only mountain relief comes from PMGE
    assert np.median(np.abs(a - b)[mountains]) < 50.0  # same structure, same heights
    # the structural plan stays HexWorld's; only the sampled summit heights differ
    assert pm.hist.plan["ridges"] == ref.hist.plan["ridges"]
    assert [(p["x_km"], p["y_km"]) for p in pm.hist.plan["peaks"]] == [
        (p["x_km"], p["y_km"]) for p in ref.hist.plan["peaks"]
    ]


# -- views, API, CLI, GUI --
def test_v3_views_show_computed_data(hist_project):
    for view_id in ("history", "relief", "volcanism"):
        spec = views.VIEWS[view_id]
        assert spec.availability(hist_project)[0]
        for mode in spec.modes:
            rgb = views.view_rgb(hist_project, view_id, mode.id)
            assert rgb.shape == (100, 100, 3) and len(np.unique(rgb.reshape(-1, 3), axis=0)) > 3
            assert spec.legend(hist_project, mode.id)
    assert views.view_vectors(hist_project, "relief", "shaded", {"ridges", "peaks", "passes"})
    assert views.view_vectors(hist_project, "volcanism", "centres", {"volcanoes", "hotspots"})
    # heights: the source is explicit and switchable between v3 and the v2 baseline
    now = views.view_rgb(hist_project, "heights", "hypsometric", frozenset())
    then = views.view_rgb(hist_project, "heights", "baseline", frozenset())
    assert not np.array_equal(now, then)
    assert "v3" in views.VIEWS["heights"].legend(hist_project, "hypsometric")[0].label
    assert "v2" in views.VIEWS["heights"].legend(hist_project, "baseline")[0].label
    compare = views.view_rgb(hist_project, "continents", "compare", frozenset())
    assert len(np.unique(compare.reshape(-1, 3), axis=0)) >= 3
    planned = {s.id for s in views.VIEWS.values() if s.status == "planned"}
    assert {
        "temperature",
        "precipitation",
        "hydrology",
        "biomes",
        "geosymbols",
        "seismicity",
    } <= planned
    assert all(views.VIEWS[i].render is None for i in planned)


def test_history_api_and_inspector(hist_project):
    hist = hist_project.hist
    early = history_events(hist_project, time_range=(0.0, 8.0))
    assert all(e["time_ma"] <= 8.0 for e in early) and len(early) < len(hist.events)
    kinds = history_events(hist_project, kinds=["volcanic_episode"])
    assert all(e["type"] == "volcanic_episode" for e in kinds)
    x, y = hist_project.layout.center(40, 60)
    got = sample_points(hist_project, [x, -1e7], [y, 0.0], fields=("elevation_m", "rock_type"))
    assert got["inside"].tolist() == [True, False]
    assert got["elevation_m"][0] == hist.layers["elevation_m"][60, 40]
    assert np.isnan(got["elevation_m"][1]) and got["rock_type"][1] == -1
    info = hist_project.hex_info(40, 60)
    assert info["epoch_time_ma"] == 16.0 and "baseline_land_mask" in info
    assert info["depth_m"] == max(0.0, info["sea_level_m"] - info["elevation_m"])
    assert info["rock"] in hist.summary["rock_types"]
    for v in hist.volcanoes:
        assert v["kind"] in ("arc", "hotspot") and v["age_ma"] >= 0.0
    assert any(v["submarine"] for v in hist.volcanoes if v["in_window"]) or not any(
        v["in_window"] for v in hist.volcanoes
    )


def test_cli_v3_flow(tmp_path, capsys):
    out = tmp_path / "p"
    assert (
        main(
            [
                "generate",
                "--seed",
                "Cli7",
                "--width",
                "40",
                "--height",
                "30",
                "--hex-km",
                "150",
                "-o",
                str(out),
                "-q",
                "--no-history",
            ]
        )
        == 0
    )
    assert not (out / "history.json").exists()
    v2_bytes = (out / "continental_fields.npz").read_bytes()
    assert main(["history", str(out), "--duration-ma", "10", "--time-step-ma", "2"]) == 0
    assert (out / "continental_fields.npz").read_bytes() == v2_bytes
    assert (out / "events.json").exists() and (out / "mountain_plan.json").exists()
    assert main(["epoch", str(out), "initial"]) == 0
    assert load_project(out).hist.epoch_time_ma == 0.0
    assert main(["epoch", str(out), "final"]) == 0
    assert main(["sea", str(out), "--level", "150"]) == 0
    loaded = load_project(out)
    assert loaded.hist.sea_level_m == 150.0 and loaded.hist.epoch_time_ma == 10.0
    assert main(["render", str(out), "--view", "all"]) == 0
    for name in (
        "view_history_delta.png",
        "view_relief_shaded.png",
        "view_volcanism_centres.png",
        "view_heights_baseline.png",
    ):
        assert (out / name).exists(), name
    capsys.readouterr()
    assert main(["events", str(out)]) == 0
    assert main(["info", str(out)]) == 0
    text = capsys.readouterr().out
    assert "backend: pmge" in text or "PMGE library not found" in text
    assert "history" in text
    assert main(["export", str(out), "-o", str(tmp_path / "e")]) == 0
    assert (tmp_path / "e" / "epoch_elevation_m.npy").exists()
    assert (tmp_path / "e" / "mountain_systems.csv").exists()


def test_gui_v3_tabs(hist_project):
    pytest.importorskip("PySide6")
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    try:
        from PySide6.QtWidgets import QApplication

        from hexworld.gui.app import MainWindow
    except ImportError as exc:
        pytest.skip(f"Qt unavailable: {exc}")
    app = QApplication.instance() or QApplication([])
    window = MainWindow()
    window.resize(1200, 760)
    window.show()
    ids = list(views.VIEWS)
    window._show_project(load_project(V2_FIXTURE), "v2", sync_form=True)
    window.tabs.setCurrentIndex(ids.index("relief"))
    app.processEvents()
    assert "геоистория ещё не сгенерирована" in window.legend.text()
    assert not window.epoch_btn.isEnabled()

    window._show_project(hist_project, "v3", sync_form=True)
    before = hist_project.hist.hist_hash()
    assert window.epoch_box.count() == len(hist_project.hist.epochs)
    assert window.epoch_box.currentData() == hist_project.hist.epoch_time_ma
    window.view.select_hex((40, 60))
    window._show_info()
    state = window.view.view_state()
    for name in ("plates", "continents", "heights", "history", "relief", "volcanism"):
        window.tabs.setCurrentIndex(ids.index(name))
        for m in range(window.mode_box.count()):
            window.mode_box.setCurrentIndex(m)
            app.processEvents()
            assert not window.view.grab().isNull()
        assert window.view.view_state() == state and window.view.selected_hex == (40, 60)
        assert window.legend.text()
    text = window.info.toPlainText()
    for key in ("снимок t = 16", "итоговая высота", "порода", "горная система", "вулкан"):
        assert key in text
    window._on_hover("строка 1\nстрока 2")
    assert window.hover.text().startswith("строка 1")
    window.ui_scale.setValue(130)
    app.processEvents()
    assert window.build_config().history.duration_ma == 16.0
    assert hist_project.hist.hist_hash() == before
    window.close()


@pytest.mark.slow
def test_v3_on_the_main_preset():
    from hexworld import PRESETS

    project = generate(Config(seed="Ab12", **PRESETS["large"]))
    assert project.hist.validation["ok"], project.hist.validation
    assert project.hist.layers["elevation_m"].shape == (800, 1000)


# -- 3.1: realism fixes ----------------------------------------------------------------
def test_continents_are_grouped_with_unequal_sizes_and_random_assembly():
    from hexworld.config import ContinentParams, CrustParams

    fast = ContinentParams(skeleton_cells=224)
    world = Config(
        seed="Ab12", width=100, height=75, hex_spacing_km=280.0, mode="world", continents=fast
    )
    assemblies, largest = [], []
    for seed in ("Ab12", "ab12", "Zx9", "Kq3Lm", "Q", "N0va"):
        p = generate(replace(world, seed=seed), through="v2")
        info = p.crust_model.layout_info
        assert info["layout"] == "clustered" and 0.0 <= info["assembly"] <= 1.0
        assemblies.append(info["assembly"])
        largest.append(p.geo.summary["world"]["largest_landmass_share_of_land"])
        assert p.geo.validation["ok"]
    # the degree of assembly is a random element: neither always scattered nor always one
    assert max(assemblies) - min(assemblies) > 0.3
    assert min(largest) < 0.6 < max(largest)

    def run(assembly, layout="clustered"):
        crust = CrustParams(layout=layout, assembly=assembly)
        return generate(replace(world, crust=crust), through="v2")

    together, apart, even = run(1.0), run(0.0), run("auto", layout="uniform")
    share = "largest_landmass_share_of_land"
    assert together.geo.summary["world"][share] > apart.geo.summary["world"][share]

    def unevenness(project):  # how unequal the land shares of the four map quarters are
        land = project.geo.layers["land_mask"]
        h, w = land.shape
        q = [
            land[: h // 2, : w // 2].mean(),
            land[: h // 2, w // 2 :].mean(),
            land[h // 2 :, : w // 2].mean(),
            land[h // 2 :, w // 2 :].mean(),
        ]
        return float(np.std(q))

    assert unevenness(together) > unevenness(even)  # grouped, not spread like noise
    for p in (together, apart, even):  # the world share of continental crust is kept
        assert abs(p.geo.summary["continental_crust_fraction_simulation"] - 0.36) < 0.03


def test_arc_volcanoes_stand_on_a_narrow_line_set_by_the_slab():
    young = two_plates((30, 0), (-30, 0), 0, 1)  # 80 Ma old ocean floor goes down
    snap = young["snap"]
    arcs = [v for v in young["volcanoes"].volcanoes if v.kind == "arc"]
    x, y, idx = young["volcanoes"].positions(young["kin"], 20.0)
    active = np.array([young["volcanoes"].is_active(int(i), 20.0) for i in idx])
    assert arcs and active.any()
    off = GRID.bilinear(
        (snap.trench_distance - snap.trench_arc_km).astype(float), x[active], y[active]
    )
    assert np.abs(off).max() < 70.0  # one line, not a band several rows wide
    assert np.ptp(x[active]) < 140.0  # the boundary runs north-south: a thin strip
    ys = np.sort(y[active])
    assert 40.0 < np.median(np.diff(ys)) < 140.0  # tens of km between neighbours
    # a steeper (older) slab puts the arc closer to the trench
    arc_km = snap.trench_arc_km[np.isfinite(snap.trench_distance) & (snap.arc_zone > 0.5)]
    assert 95.0 <= arc_km.min() and arc_km.max() <= 320.0
    assert all(r.radius_km < 30.0 for r in arcs)


def test_volcano_marks_are_filtered_and_drawn_to_scale(hist_project):
    default = views.view_vectors(hist_project, "volcanism", "centres")
    everything = views.view_vectors(
        hist_project, "volcanism", "centres", {"volcanoes", "extinct", "submarine", "hotspots"}
    )

    def count(layers):
        return sum(len(v.x) for v in layers if v.kind == "points" and v.label != "hotspots")

    volcanoes = hist_project.hist.volcanoes
    assert count(default) == sum(1 for v in volcanoes if v["active"] and not v["submarine"])
    assert count(everything) == len(volcanoes) >= count(default)
    assert all(v.size_km > 0 for v in default if v.label.endswith("active"))
    legend = " ".join(i.label for i in views.VIEWS["volcanism"].legend(hist_project, "centres"))
    assert "один знак = один вулканический центр" in legend and "1,5 млн лет" in legend
