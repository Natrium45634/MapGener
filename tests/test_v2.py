"""v2.0: continental blocks, macro relief, sea level, landmasses, views, compatibility."""

import json
import os
import shutil
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from hexworld import (
    GenerationCancelled,
    derive_landmasses,
    generate,
    generate_continents,
    generate_macro_relief,
    generate_v2,
    get_layer,
    load_project,
    save_project,
    views,
)
from hexworld.cli import main
from hexworld.config import ContinentParams, ReliefParams, SeaParams
from hexworld.project import array_hash
from hexworld.stages.sea import label_mask
from hexworld.stages.validate import validate_geo

V1_FIXTURE = Path(__file__).parent / "data" / "v1_project"
FAST = ContinentParams(skeleton_cells=224)


# -- 1. compatibility with v1.0 projects ---------------------------------------
def test_v1_project_opens_and_is_not_changed(tmp_path):
    work = tmp_path / "proj"
    shutil.copytree(V1_FIXTURE, work)
    original = {p.name: p.read_bytes() for p in V1_FIXTURE.iterdir()}
    manifest_v1 = json.loads((V1_FIXTURE / "manifest.json").read_text(encoding="utf-8"))
    assert manifest_v1["schema_version"] == "hexworld-project-1"

    v1 = load_project(work)
    assert v1.geo is None and v1.validation["ok"]
    assert v1.content_hashes() == manifest_v1["content_hashes"]
    assert views.VIEWS["plates"].availability(v1)[0]
    ok, reason = views.VIEWS["continents"].availability(v1)
    assert not ok and "ещё не сгенерированы" in reason
    with pytest.raises(LookupError):
        views.view_rgb(v1, "heights")  # never a placeholder picture
    with pytest.raises(KeyError):
        v1.layer("macro_elevation_m")  # never zeros in place of missing data

    v2 = generate_v2(v1, replace(v1.config, continents=FAST))
    assert v2.geo is not None and v2.geo.validation["ok"], v2.geo.validation
    assert v2.world_id == v1.world_id and v2.config.seed == v1.config.seed
    assert v2.content_hashes() == manifest_v1["content_hashes"]  # v1 fields untouched
    assert v2.layers["plate_id"] is v1.layers["plate_id"]  # not even recomputed

    save_project(v2, work)
    for name in ("plates.json", "fields.npz", "boundaries.npz", "boundaries.json"):
        assert (work / name).read_bytes() == original[name], name  # source data not rewritten
    assert (work / "config.v1.json").read_bytes() == original["config.json"]
    assert (work / "manifest.v1.json").read_bytes() == original["manifest.json"]
    assert (work / "continental_fields.npz").exists() and (work / "continents.json").exists()

    again = load_project(work)
    assert again.content_hashes() == manifest_v1["content_hashes"]
    assert again.geo.geo_hash() == v2.geo.geo_hash()
    assert again.geo.sea_level_m == v2.geo.sea_level_m
    assert again.config == v2.config
    assert again.geo.landmasses == v2.geo.landmasses and again.geo.blocks == v2.geo.blocks
    # adding v2 keeps the project a v2 project: later stages are not computed silently
    assert again.hist is None and again.hydro is None
    for spec in views.VIEWS.values():
        expected = spec.status == "ready" and spec.stage in ("v1", "v2")
        assert spec.availability(again)[0] == expected, spec.id


# -- 3. repeatability ------------------------------------------------------------
def test_v2_is_deterministic_and_seed_sensitive(small_config, small_project):
    again = generate(small_config)
    assert again.geo.content_hashes() == small_project.geo.content_hashes()
    assert again.geo.sea_level_m == small_project.geo.sea_level_m
    assert again.geo.blocks == small_project.geo.blocks
    other = generate(replace(small_config, seed="ab12"))
    assert other.geo.geo_hash() != small_project.geo.geo_hash()


def test_v2_parameters_never_change_v1_outputs(small_config, small_project):
    v1_hashes = small_project.content_hashes()
    changed = generate(
        replace(
            small_config,
            continents=replace(FAST, continent_profile="fragmented", coast_irregularity=1.1),
            relief=ReliefParams(relief_strength=1.8),
            sea=SeaParams(target_land_fraction=0.4),
        ),
        reuse=small_project,
    )
    assert changed.content_hashes() == v1_hashes
    assert changed.world_id == small_project.world_id
    assert changed.plates == small_project.plates
    assert changed.layers["plate_id"] is small_project.layers["plate_id"]
    assert changed.geo.geo_hash() != small_project.geo.geo_hash()
    # and the incremental result equals a from-scratch run with the same config
    assert generate(changed.config).geo.content_hashes() == changed.geo.content_hashes()


# -- 4. physical consistency -----------------------------------------------------
def test_land_comes_only_from_elevation_and_sea_level(small_project):
    geo = small_project.geo
    elev = geo.layers["macro_elevation_m"]
    land = geo.layers["land_mask"]
    assert elev.dtype == np.float32 and land.dtype == np.bool_
    assert np.array_equal(land, elev > np.float32(geo.sea_level_m))
    ids = geo.layers["landmass_id"]
    assert np.array_equal(ids >= 0, land)  # every land hex in exactly one landmass, sea in none
    relabelled, sizes = label_mask(land, small_project.layout)
    assert np.array_equal(relabelled, ids)  # ids are the edge-connected components
    assert len(geo.landmasses) == sizes.size
    assert sum(m["hexes"] for m in geo.landmasses) == int(land.sum())
    info = small_project.hex_info(50, 50)
    assert info["depth_m"] == max(0.0, info["sea_level_m"] - info["macro_elevation_m"])
    assert info["height_above_sea_m"] * info["depth_m"] == 0.0


def test_blocks_landmasses_and_plates_are_different_things(small_project):
    geo = small_project.geo
    frac = geo.layers["continental_fraction"]
    block = geo.layers["continental_block_id"]
    land = geo.layers["land_mask"]
    assert 0.0 <= frac.min() and frac.max() <= 1.0
    assert np.array_equal(block >= 0, frac >= 0.5)
    assert np.array_equal(label_mask(block >= 0, small_project.layout)[0], block)
    # continental crust under water exists (shelf): crust edge is not the coastline
    assert np.any((block >= 0) & ~land)
    shelf = geo.layers["shelf_mask"]
    assert shelf.any() and not np.any(shelf & land) and not np.any(shelf & (frac < 0.5))
    # a block may span several plates, a plate may carry several crust types
    assert max(b["plate_count"] for b in geo.blocks) >= 2
    plate_id = small_project.layer("plate_id")
    mixed = [p for p in np.unique(plate_id) if 0.02 < (block[plate_id == p] >= 0).mean() < 0.98]
    assert mixed
    # the v1 prior is kept as it was; v2 stores its own refined field
    prior = small_project.layer("crust_type")
    assert not np.array_equal(prior == 1, block >= 0)
    for b in geo.blocks:
        assert abs(b["land_area_km2"] + b["submerged_area_km2"] - b["area_km2"]) < 1e-3
        assert len(b["bounds_km"]) == 4 and len(b["centroid_km"]) == 2
    assert validate_geo(small_project)["ok"]


def test_relief_has_the_expected_macro_structure(small_project):
    geo = small_project.geo
    elev = geo.layers["macro_elevation_m"]
    frac = geo.layers["continental_fraction"]
    assert np.median(elev[frac > 0.95]) > np.median(elev[frac < 0.05]) + 2500.0
    assert elev.min() < -3500.0 and elev.max() < 9000.0
    assert np.abs(geo.layers["tectonic_relief_m"]).max() > 300.0
    flat = generate(
        replace(small_project.config, relief=ReliefParams(relief_strength=0.0)), reuse=small_project
    )
    assert np.abs(flat.geo.layers["tectonic_relief_m"]).max() == 0.0
    # relief strength does not move the crust
    assert np.array_equal(flat.geo.layers["continental_fraction"], frac)


# -- 5. sea level ------------------------------------------------------------------
def test_sea_level_change_floods_monotonically_and_touches_nothing_else(small_project):
    base = small_project
    previous_land = None
    areas = []
    for level in (-400.0, -100.0, 0.0, 150.0, 400.0, 900.0):
        p = derive_landmasses(base, sea_level_m=level)
        geo = p.geo
        assert geo.sea_level_m == level and geo.validation["ok"]
        assert {"world_tectonics", "continental_blocks", "macro_elevation", "v2_raster"} <= set(
            p.stats["reused_stages"]
        )
        for name in (
            "macro_elevation_m",
            "continental_fraction",
            "continental_block_id",
            "tectonic_relief_m",
            "continental_margin_km",
        ):
            assert geo.layers[name] is base.geo.layers[name]  # same arrays, not recomputed
        assert p.content_hashes() == base.content_hashes()
        land = geo.layers["land_mask"]
        if previous_land is not None:
            assert not np.any(land & ~previous_land)  # rising sea never creates land
        previous_land = land
        areas.append(int(land.sum()))
    assert areas == sorted(areas, reverse=True) and areas[0] > areas[-1]


def test_sea_modes_are_exclusive_and_reported(small_config, small_project):
    target = generate(
        replace(small_config, sea=SeaParams(target_land_fraction=0.45)), reuse=small_project
    )
    world = target.geo.summary["world"]
    assert abs(world["land_fraction_simulation"] - 0.45) < 0.01
    assert target.geo.summary["target_land_fraction"] == 0.45
    fixed = generate(
        replace(
            small_config, sea=SeaParams(sea_mode="fixed_sea_level_m", fixed_sea_level_m=-1500.0)
        ),
        reuse=small_project,
    )
    assert fixed.geo.sea_level_m == -1500.0
    assert fixed.geo.summary["target_land_fraction"] is None
    # the land share is measured, not forced: at -1500 m even part of the slope is dry
    assert fixed.geo.summary["world"]["land_fraction_simulation"] > 0.36
    # more land than there is continental crust needs an implausible level: reported
    assert any("unusual sea level" in w for w in target.geo.warnings)
    with pytest.raises(ValueError):
        derive_landmasses(small_project, sea_level_m=0.0, target_land_fraction=0.3)


# -- settings ----------------------------------------------------------------------
@pytest.mark.parametrize("profile", ["balanced", "supercontinent", "fragmented"])
def test_profiles_give_valid_and_different_worlds(profile, small_config, small_project):
    p = generate(
        replace(small_config, continents=replace(FAST, continent_profile=profile)),
        reuse=small_project,
    )
    assert p.geo.validation["ok"], p.geo.validation
    assert p.geo.validation["checks"]["speck_land_share"] < 0.02  # no fringe of single hexes
    share = p.geo.summary["continental_crust_fraction_simulation"]
    assert abs(share - small_config.crust.continental_fraction) < 0.03


def test_profiles_change_the_configuration_of_land(small_config):
    world = replace(small_config, mode="world", hex_spacing_km=220.0)
    stats = {}
    for profile in ("supercontinent", "fragmented"):
        p = generate(replace(world, continents=replace(FAST, continent_profile=profile)))
        stats[profile] = p.geo.summary["world"]
    assert (
        stats["supercontinent"]["largest_landmass_share_of_land"]
        > stats["fragmented"]["largest_landmass_share_of_land"]
    )
    assert (
        stats["fragmented"]["major_landmass_count"]
        >= stats["supercontinent"]["major_landmass_count"]
    )


def test_major_landmass_count_is_a_goal_with_an_honest_report(small_config):
    world = replace(small_config, mode="world", hex_spacing_km=220.0)
    few = generate(replace(world, continents=replace(FAST, major_landmass_count=2)))
    many = generate(replace(world, continents=replace(FAST, major_landmass_count=12)))
    n_few = few.geo.summary["world"]["major_landmass_count"]
    n_many = many.geo.summary["world"]["major_landmass_count"]
    assert n_few < n_many
    for project, wanted, got in ((few, 2, n_few), (many, 12, n_many)):
        assert project.geo.validation["ok"]
        if got != wanted:  # an unreachable goal must be reported, not silently faked
            assert any(str(wanted) in w for w in project.geo.warnings)


def test_coast_irregularity_and_size_variety_act_on_shape(small_config, small_project):
    smooth = generate(
        replace(small_config, continents=replace(FAST, coast_irregularity=0.0)), reuse=small_project
    )
    rough = generate(
        replace(small_config, continents=replace(FAST, coast_irregularity=1.3)), reuse=small_project
    )

    def coast_length(project):
        land = project.geo.layers["land_mask"]
        return int(views.mask_rim(land, project.layout).sum()) / max(1, int(land.sum()))

    assert coast_length(rough) > coast_length(smooth)
    varied = generate(
        replace(small_config, continents=replace(FAST, continent_size_variety=1.0)),
        reuse=small_project,
    )
    assert varied.geo.geo_hash() != small_project.geo.geo_hash()


# -- 8. windows and resolutions ------------------------------------------------------
def test_two_windows_show_the_same_continents_and_heights(small_project):
    g = small_project.geometry
    pinned = replace(
        small_project.config,
        simulation_bounds_km=tuple(g.simulation_bounds.as_list()),
        plates=replace(small_project.config.plates, count=g.plate_count),
    )
    base = generate(pinned)
    assert base.geo.content_hashes() == small_project.geo.content_hashes()
    fine = generate(
        replace(
            pinned, width=180, height=160, hex_spacing_km=20.0, viewport_origin_km=(1500.0, 2000.0)
        )
    )
    assert fine.geo.sea_level_m == base.geo.sea_level_m  # the sea level is a world property
    x, y = fine.layout.centers()
    col, row = base.layout.world_to_offset(x, y)
    for name, tolerance in (("land_mask", 0.97), ("shelf_mask", 0.95)):
        same = base.geo.layers[name][row, col] == fine.geo.layers[name]
        assert same.mean() > tolerance, name
    block_same = (base.geo.layers["continental_block_id"][row, col] >= 0) == (
        fine.geo.layers["continental_block_id"] >= 0
    )
    assert block_same.mean() > 0.97
    # elevation profile along a row agrees within what one coarse hex can hide
    diff = np.abs(
        base.geo.layers["macro_elevation_m"][row, col] - fine.geo.layers["macro_elevation_m"]
    )
    assert np.median(diff) < 60.0 and np.percentile(diff, 90) < 400.0
    # get_layer reads the same world through any window
    layout, elev = get_layer(
        base, "macro_elevation_m", fine.geometry.viewport_bounds.as_list(), hex_spacing_km=20.0
    )
    assert elev.shape == (layout.height, layout.width)
    c, r = fine.layout.world_to_offset(*layout.centers())
    ok = fine.layout.inside(c, r)
    assert np.allclose(elev[ok], fine.geo.layers["macro_elevation_m"][r[ok], c[ok]], atol=0.01)


def test_no_forced_ocean_at_the_window_edge(small_config):
    edge_land = []
    for seed in ("Ab12", "Zx9", "Terra7", "Kq3Lm"):
        p = generate(replace(small_config, seed=seed))
        land = p.geo.layers["land_mask"]
        rim = np.concatenate([land[0], land[-1], land[:, 0], land[:, -1]])
        edge_land.append(rim.mean())
        assert any(m["touches_window_edge"] for m in p.geo.landmasses) == bool(rim.any())
    assert max(edge_land) > 0.05  # continents do run off the window


# -- API, views, CLI ------------------------------------------------------------------
def test_stage_api(small_project):
    continents = generate_continents(small_project)
    relief = generate_macro_relief(small_project, continents)
    cached = small_project.cache["macro_elevation"]["world_elevation_m"]
    assert np.array_equal(relief.world_elevation_m.astype(np.float32), cached)
    x, y = small_project.layout.centers()
    assert np.allclose(
        relief.model.elevation(x[:5, :5], y[:5, :5]),
        small_project.geo.layers["macro_elevation_m"][:5, :5],
        atol=0.01,
    )
    layout, stored = get_layer(small_project, "landmass_id")
    assert stored is small_project.layer("landmass_id") and layout == small_project.layout


def test_cancel_keeps_the_previous_project(small_config, small_project):
    before = small_project.geo.geo_hash()

    def progress(stage, _fraction):
        if stage == "continental_blocks":
            raise GenerationCancelled()

    with pytest.raises(GenerationCancelled):
        generate(
            replace(small_config, continents=replace(FAST, coast_irregularity=0.2)),
            progress=progress,
            reuse=small_project,
        )
    assert small_project.geo.geo_hash() == before


def test_views_are_pure_presentations(small_project):
    before = (small_project.content_hashes(), small_project.geo.content_hashes())
    for spec in views.VIEWS.values():
        if spec.status != "ready":
            assert not spec.availability(small_project)[0]
            assert spec.render is None  # a planned view cannot be drawn at all
            continue
        for mode in spec.modes:
            for overlays in (frozenset(), frozenset(o.id for o in spec.overlays)):
                rgb = views.view_rgb(small_project, spec.id, mode.id, overlays)
                assert rgb.shape == (100, 100, 3) and rgb.dtype == np.uint8
            assert spec.legend(small_project, mode.id)
    assert {"plates", "continents", "heights"} <= set(views.VIEWS)
    blocks = views.view_rgb(small_project, "continents", "blocks", frozenset())
    lands = views.view_rgb(small_project, "continents", "landmasses", frozenset())
    assert not np.array_equal(blocks, lands)  # crust blocks vs land above water
    legend = views.VIEWS["heights"].legend(small_project, "baseline")
    assert "без детализации и эрозии" in legend[0].label
    assert (small_project.content_hashes(), small_project.geo.content_hashes()) == before


def test_cli_v2_flow(tmp_path, capsys):
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
                "--v1-only",
            ]
        )
        == 0
    )
    assert not (out / "continental_fields.npz").exists()
    fields_before = (out / "fields.npz").read_bytes()
    assert (
        main(
            ["continents", str(out), "--continent-profile", "fragmented", "--land-fraction", "0.35"]
        )
        == 0
    )
    assert (out / "fields.npz").read_bytes() == fields_before
    assert main(["sea", str(out), "--level", "200"]) == 0
    assert load_project(out).geo.sea_level_m == 200.0
    assert main(["render", str(out), "--view", "all"]) == 0
    for name in (
        "view_plates_plates.png",
        "view_continents_landmasses.png",
        "view_continents_blocks.png",
        "view_heights_hypsometric.png",
    ):
        assert (out / name).read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"
    # a v2 project stays a v2 project after a sea change: v4 views have no data to draw
    assert load_project(out).hist is None and load_project(out).hydro is None
    assert main(["render", str(out), "--view", "hydrology"]) == 1
    assert main(["export", str(out), "-o", str(tmp_path / "exp")]) == 0
    assert (tmp_path / "exp" / "landmasses.csv").exists()
    assert np.load(tmp_path / "exp" / "macro_elevation_m.npy").shape == (30, 40)
    assert main(["views"]) == 0
    assert "Запланировано" in capsys.readouterr().out
    assert main(["sea", str(out), "--level", "1", "--land-fraction", "0.3"]) == 1


def test_hash_of_stored_v2_layers_matches_manifest(tmp_path, small_project):
    path = save_project(small_project, tmp_path / "p")
    manifest = json.loads((path / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["schema_version"] == "hexworld-project-4"
    with np.load(path / "continental_fields.npz") as data:
        for name in small_project.geo.layers:
            assert array_hash(data[name]) == manifest["v2"]["content_hashes"][f"continental/{name}"]
    loaded = load_project(path)
    # after loading, a sea change still reuses the stored elevation
    risen = derive_landmasses(loaded, sea_level_m=loaded.geo.sea_level_m + 200.0)
    assert risen.geo.layers["macro_elevation_m"] is loaded.geo.layers["macro_elevation_m"]
    assert risen.geo.layers["land_mask"].sum() <= loaded.geo.layers["land_mask"].sum()


def test_gui_tabs(tmp_path, small_project):
    pytest.importorskip("PySide6")
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    try:
        from PySide6.QtWidgets import QApplication

        from hexworld.gui.app import MainWindow
    except ImportError as exc:  # missing system GL libraries on a headless box
        pytest.skip(f"Qt unavailable: {exc}")
    app = QApplication.instance() or QApplication([])
    window = MainWindow()
    window.resize(1100, 700)
    window.show()
    ids = list(views.VIEWS)
    for i, spec in enumerate(views.VIEWS.values()):
        assert window.tabs.isTabEnabled(i) == (spec.status == "ready")

    # a v1 project: plates are shown, the v2 tabs say they are not generated yet
    window._show_project(load_project(V1_FIXTURE), "v1", sync_form=True)
    window.tabs.setCurrentIndex(ids.index("continents"))
    app.processEvents()
    assert window.v2_button.isVisible()
    assert "ещё не сгенерированы" in window.legend.text()
    assert not window.view.grab().isNull()

    window._show_project(small_project, "v2", sync_form=True)
    before = (small_project.content_hashes(), small_project.geo.content_hashes())
    window.view.select_hex((40, 60))
    window._show_info()
    state = window.view.view_state()
    for name in ("plates", "continents", "heights"):
        window.tabs.setCurrentIndex(ids.index(name))
        for m in range(window.mode_box.count()):
            window.mode_box.setCurrentIndex(m)
            app.processEvents()
            assert not window.view.grab().isNull()
        assert window.view.view_state() == state  # same coordinates on every tab
        assert window.view.selected_hex == (40, 60)  # same selection
        assert window.legend.text()
    assert not window.v2_button.isVisible()
    assert "источник" in window.legend.text()  # the height source is always stated
    text = window.info.toPlainText()
    for key in ("plate_id", "continental_block_id", "landmass_id", "уровень моря", "глубина"):
        assert key in text
    assert (small_project.content_hashes(), small_project.geo.content_hashes()) == before
    config = window.build_config()
    assert config.seed == small_project.config.seed
    assert config.sea.sea_mode == small_project.config.sea.sea_mode
    window.close()
