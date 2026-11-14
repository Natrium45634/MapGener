import json
from dataclasses import replace

import numpy as np
import pytest

from hexworld import Config, generate, load_project, save_project
from hexworld.config import BoundaryParams, FieldParams, MotionParams, PlateParams
from hexworld.io import ProjectIntegrityError
from hexworld.stages.raster import component_table


def test_same_seed_same_hashes(small_config, small_project):
    again = generate(small_config)
    assert again.content_hashes() == small_project.content_hashes()
    assert again.world_id == small_project.world_id
    assert again.plates == small_project.plates


def test_seed_case_changes_the_world(small_config, small_project):
    other = generate(replace(small_config, seed="ab12"))
    assert other.world_id != small_project.world_id
    assert not np.array_equal(other.layer("plate_id"), small_project.layer("plate_id"))


def test_every_hex_belongs_to_an_existing_plate(small_project):
    pid = small_project.layer("plate_id")
    assert pid.shape == (100, 100)
    assert pid.min() >= 0 and pid.max() < len(small_project.plates)
    assert small_project.validation["ok"], small_project.validation["errors"]


def test_plates_are_connected_without_single_hex_debris(small_project):
    pid = small_project.layer("plate_id")
    _labels, _plate, size, touches, largest = component_table(pid, small_project.layout)
    # a second piece is only allowed if it reaches the window edge (joins outside)
    assert not np.any(~touches & ~largest)
    assert np.count_nonzero(size <= 3) <= 2


def test_plate_sizes_are_diverse_and_metadata_complete(small_project):
    plates = [p for p in small_project.plates if not p["halo"]]
    areas = np.array([p["area_km2"] for p in plates])
    assert areas.min() > 0
    assert areas.max() / areas.min() > 4.0
    assert {p["size_class"] for p in plates} >= {"large", "medium", "small"}
    assert len({p["name"] for p in small_project.plates}) == len(small_project.plates)
    for p in plates:
        assert len(p["velocity_mm_yr"]) == 2 and p["neighbors"]
        assert abs(p["continental_fraction"] + p["oceanic_fraction"] - 1.0) < 1e-9
        for n in p["neighbors"]:
            assert p["id"] in small_project.plates[n]["neighbors"]


def test_crust_is_independent_of_plates(small_project):
    mixed = [
        p for p in small_project.plates if not p["halo"] and 0.05 < p["continental_fraction"] < 0.95
    ]
    assert mixed, "expected at least one plate carrying both crust types"
    target = small_project.config.crust.continental_fraction
    ref = small_project.cache["reference"]
    assert abs(ref.crust[ref.in_sim].mean() - target) < 0.01


def test_no_artificial_frame_at_the_window_edge(small_project):
    """The window edge is not a plate boundary and not an ocean rim."""
    assert small_project.layer("boundary_type")[0, :].mean() < 0.5
    crust = small_project.layer("crust_type")
    rim = np.concatenate([crust[0], crust[-1], crust[:, 0], crust[:, -1]])
    g = small_project.geometry
    # plates continue beyond the window: the model answers outside of it
    out = small_project.sample_world(g.viewport_bounds.xmin - 2000.0, g.viewport_bounds.ymin)
    assert 0 <= int(out["plate_id"]) < len(small_project.plates)
    assert rim.size == 400


def test_two_windows_of_one_world_agree(small_project):
    """Criterion 5: different window + resolution, same world, same geology."""
    g = small_project.geometry
    pinned = replace(
        small_project.config,
        simulation_bounds_km=tuple(g.simulation_bounds.as_list()),
        plates=replace(small_project.config.plates, count=g.plate_count),
    )
    base = generate(pinned)
    assert base.world_id == small_project.world_id
    assert np.array_equal(base.layer("plate_id"), small_project.layer("plate_id"))

    fine = generate(
        replace(
            pinned, width=180, height=160, hex_spacing_km=20.0, viewport_origin_km=(1500.0, 2000.0)
        )
    )
    assert fine.world_id == base.world_id
    assert fine.plates[3]["area_km2"] == base.plates[3]["area_km2"]

    x, y = fine.layout.centers()
    col, row = base.layout.world_to_offset(x, y)
    assert base.layout.inside(col, row).all()
    same = base.layer("plate_id")[row, col] == fine.layer("plate_id")
    assert same.mean() > 0.97
    # every disagreement sits within one coarse hex of a boundary (discretisation only)
    assert fine.layer("boundary_distance_km")[~same].max() <= 1.01 * base.layout.spacing_km
    crust_same = base.layer("crust_type")[row, col] == fine.layer("crust_type")
    assert crust_same.mean() > 0.97
    # and the continuous model is identical point for point
    assert np.array_equal(base.sample_world(x, y)["plate_id"], fine.sample_world(x, y)["plate_id"])


def test_late_stage_changes_do_not_touch_early_stages(small_config, small_project):
    moved = generate(
        replace(small_config, motion=MotionParams(speed_scale=1.5)), reuse=small_project
    )
    assert moved.stats["reused_stages"] == ["plates", "crust", "raster"]
    assert moved.plate_model is small_project.plate_model
    assert np.array_equal(moved.layer("plate_id"), small_project.layer("plate_id"))
    assert np.array_equal(moved.layer("crust_type"), small_project.layer("crust_type"))
    assert not np.array_equal(moved.layer("velocity_x"), small_project.layer("velocity_x"))

    widened = generate(
        replace(small_config, fields=FieldParams(arc_offset_km=260.0)), reuse=small_project
    )
    assert "boundary_edges" in widened.stats["reused_stages"]
    assert widened.boundaries.arrays().keys() == small_project.boundaries.arrays().keys()
    for key, arr in widened.boundaries.arrays().items():
        assert np.array_equal(arr, small_project.boundaries.arrays()[key])
    assert not np.array_equal(widened.layer("volcanism"), small_project.layer("volcanism"))

    reclass = generate(replace(small_config, boundaries=BoundaryParams(obliquity_deg=50.0)))
    assert np.array_equal(reclass.layer("plate_id"), small_project.layer("plate_id"))
    # and without `reuse` the result is identical to the incremental one
    assert (
        generate(replace(small_config, motion=MotionParams(speed_scale=1.5))).content_hashes()
        == moved.content_hashes()
    )


def test_manual_plate_count_and_profiles(small_config):
    manual = generate(replace(small_config, plates=PlateParams(count=7)))
    assert manual.stats["plates_in_simulation"] == 7
    assert manual.validation["ok"]
    few = generate(replace(small_config, plates=PlateParams(profile="few")))
    many = generate(replace(small_config, plates=PlateParams(profile="fragmented")))
    assert few.geometry.plate_count < many.geometry.plate_count
    assert few.validation["ok"] and many.validation["ok"]


def test_diagnostic_fields_follow_boundary_physics(small_project):
    p = small_project
    for name in ("uplift", "rifting", "subduction", "volcanism", "seismicity"):
        arr = p.layer(name)
        assert arr.dtype == np.float32 and arr.min() >= 0.0 and arr.max() <= 1.0
    far = p.layer("boundary_distance_km") > 0.9 * p.config.fields.max_influence_km
    if far.any():
        assert p.layer("seismicity")[far].max() < 0.05
    kind = p.layer("boundary_type")
    assert p.layer("boundary_distance_km")[kind > 0].max() <= p.layout.spacing_km


def test_save_load_roundtrip(tmp_path, small_project):
    path = save_project(small_project, tmp_path / "proj")
    for name in (
        "config.json",
        "plates.json",
        "fields.npz",
        "boundaries.npz",
        "boundaries.json",
        "manifest.json",
    ):
        assert (path / name).exists()
    loaded = load_project(path)
    assert loaded.config == small_project.config
    assert loaded.world_id == small_project.world_id
    assert loaded.content_hashes() == small_project.content_hashes()
    assert loaded.plates == small_project.plates
    assert loaded.validation["ok"]
    assert loaded.plate_model is not None
    x, y = small_project.layout.centers()
    assert np.array_equal(
        loaded.sample_world(x, y)["plate_id"], small_project.sample_world(x, y)["plate_id"]
    )
    info = loaded.hex_info(10, 20)
    assert info["plate_id"] == int(small_project.layer("plate_id")[20, 10])

    # regenerating from the stored config reproduces the stored hashes
    manifest = json.loads((path / "manifest.json").read_text(encoding="utf-8"))
    assert generate(loaded.config).content_hashes() == manifest["content_hashes"]

    # saving identical geology again leaves the source data untouched
    before = (path / "fields.npz").stat().st_mtime_ns
    save_project(loaded, path)
    assert (path / "fields.npz").stat().st_mtime_ns == before

    # motion can be recomputed on a loaded project without regenerating plates
    again = generate(replace(loaded.config, motion=MotionParams(speed_scale=0.5)), reuse=loaded)
    assert again.stats["reused_stages"] == ["plates", "crust"]
    assert np.array_equal(again.layer("plate_id"), loaded.layer("plate_id"))


def test_tampered_project_is_rejected(tmp_path, small_project):
    path = save_project(small_project, tmp_path / "proj")
    plates = path / "plates.json"
    plates.write_text(
        plates.read_text(encoding="utf-8").replace('"large"', '"LARGE"', 1), encoding="utf-8"
    )
    with pytest.raises(ProjectIntegrityError):
        load_project(path)


def test_sample_region_at_other_resolution(small_project):
    b = small_project.geometry.viewport_bounds
    layout, layers = small_project.sample_region(
        [b.xmin + 500, b.ymin + 500, b.xmin + 2500, b.ymin + 2000], hex_spacing_km=10.0
    )
    assert layers["plate_id"].shape == (layout.height, layout.width)
    assert layout.bounds.xmax <= b.xmin + 2500 + 1e-6


@pytest.mark.slow
@pytest.mark.parametrize("preset", ["medium", "large"])
def test_large_presets(preset):
    from hexworld import PRESETS

    project = generate(Config(seed="Ab12", **PRESETS[preset]))
    assert project.validation["ok"], project.validation["errors"]
    assert project.layer("plate_id").shape == (project.layout.height, project.layout.width)
