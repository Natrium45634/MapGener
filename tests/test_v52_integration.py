"""TZ 5.2 part VI.C: save / load, GUI layers, scale, profiles (I01-I16) on the generated
test world (``hist_project``) and synthetic checks."""

from __future__ import annotations

import json
import os
import threading

import numpy as np
import pytest

from hexworld import Config
from hexworld.resources import api
from hexworld.resources import secondary as sec


def test_i01_old_configs_load_without_the_stage():
    d = json.loads(Config(seed="Old1").to_json())
    d.pop("resources")
    cfg = Config.from_dict(d)
    assert cfg.resources.enabled is False  # an old project is not silently recomputed


def test_i02_i08_save_load_keeps_ids_profiles_and_ledgers(hist_project, tmp_path):
    from hexworld.io import load_project, save_project

    path = save_project(hist_project, tmp_path / "p52")
    doc = json.loads((path / "resources.json").read_text(encoding="utf-8"))
    assert (
        "nodata" in doc and doc["provenance"]["deposits"]["quality"] == "synthetic_not_calibrated"
    )
    back = load_project(path)
    a, b = hist_project.resources, back.resources
    assert b is not None
    assert [s["deposit_id"] for s in a.catalog["systems"]] == [
        s["deposit_id"] for s in b.catalog["systems"]
    ]
    assert [x["body_id"] for x in a.catalog["bodies"]] == [
        x["body_id"] for x in b.catalog["bodies"]
    ]
    for k, v in a.layers.items():
        assert np.array_equal(v, b.layers[k], equal_nan=True), k
    assert a.summary["element_budget_over_soil_age"] == b.summary["element_budget_over_soil_age"]
    assert np.array_equal(a.catalog["hex_offsets"], b.catalog["hex_offsets"])
    assert b.validation["ok"]


def test_i03_layers_are_distinct_quantities(hist_project):
    L = hist_project.resources.layers
    for name in (
        "soil_fertility_potential",
        "suitability_crop_wheat",
        "prospectivity_by_family",
        "resource_in_situ_t_by_family",
    ):
        assert name in L
    assert L["prospectivity_by_family"].shape[0] == L["resource_in_situ_t_by_family"].shape[0] == 9
    soil = hist_project.env.layers["soil_present"].astype(bool)
    f = L["soil_fertility_potential"][soil]
    w = L["suitability_crop_wheat"][soil]
    assert not np.allclose(f, w)  # native potential is not a crop suitability


def test_i04_i05_player_view_hides_unexposed_bodies(hist_project):
    from hexworld import views

    cat = hist_project.resources.catalog
    player = views._vectors_res52(hist_project, "outcrops", frozenset())
    truth = views._vectors_res52(hist_project, "outcrops", frozenset({"ground_truth"}))
    n_vis = sum(1 for s in cat["systems"] if s["surface_occurrence"])
    n_hidden = len(cat["systems"]) - n_vis
    assert sum(v.x.size for v in player) == n_vis
    assert sum(v.x.size for v in truth) == n_vis + n_hidden
    if n_hidden:
        hidden = next(s for s in cat["systems"] if not s["surface_occurrence"])
        b = next(x for x in cat["bodies"] if x["system_id"] == hidden["deposit_id"])
        got = api.query_deposits_at_world_xy(hist_project, b["center_m"][0], b["center_m"][1])
        assert all(h["system_id"] != hidden["deposit_id"] for h in got)
        got = api.query_deposits_at_world_xy(
            hist_project, b["center_m"][0], b["center_m"][1], ground_truth=True
        )
        assert any(h["system_id"] == hidden["deposit_id"] for h in got)
        assert all(h["depth_top_m"] >= -1e-6 for h in got)  # depths in metres below the surface


def test_i06_i07_cross_section_cuts_only_real_bodies(hist_project):
    cat = hist_project.resources.catalog
    if not cat["bodies"]:
        pytest.skip("no bodies in this world")
    from hexworld.resources.models import VEIN_SHAPES

    # thin veins may fall between section samples; take the widest non-vein body
    b = max(
        (x for x in cat["bodies"] if x["shape"] not in VEIN_SHAPES), key=lambda x: x["axes_m"][0]
    )
    cx, cy = b["center_m"][0] / 1000.0, b["center_m"][1] / 1000.0
    sec_ = api.query_geological_cross_section(
        hist_project, (cx - 60.0, cy), (cx + 60.0, cy), depth_limits_m=(0.0, 1e5), step_km=0.25
    )
    hit = [x for x in sec_["bodies"] if x["body_id"] == b["body_id"]]
    assert hit, "the section through the body centre crosses it"
    for _d, top, bot in hit[0]["points"]:
        assert top >= bot
    # a line far away from every body shows no ore
    far = api.query_geological_cross_section(hist_project, (-5000.0, -5000.0), (-4900.0, -5000.0))
    assert far["bodies"] == []


def test_i09_i10_parallel_queries_give_the_same_answer(hist_project):
    cat = hist_project.resources.catalog
    pts = [(b["center_m"][0], b["center_m"][1]) for b in cat["bodies"][:20]]
    ref = [api.query_deposits_at_world_xy(hist_project, x, y, ground_truth=True) for x, y in pts]
    out = [None] * len(pts)

    def work(k):
        out[k] = api.query_deposits_at_world_xy(hist_project, *pts[k], ground_truth=True)

    th = [threading.Thread(target=work, args=(k,)) for k in range(len(pts))]
    for t in th:
        t.start()
    for t in th:
        t.join()
    assert out == ref


def test_i11_area_weighted_aggregation_not_by_fragment_count():
    from hexworld.resources.soil import aggregate_ph

    # one hex made of three fragments with very different areas
    areas = np.array([0.9, 0.05, 0.05])
    soc = np.array([2.0, 20.0, 20.0])
    by_area = float((soc * areas).sum() / areas.sum())
    by_count = float(soc.mean())
    assert by_area == pytest.approx(3.8) and by_count == pytest.approx(14.0)
    assert aggregate_ph([5.0, 8.0, 8.0], areas) < 6.0  # pH via H+ activity, area-weighted


def test_i12_ledger_across_a_chunk_boundary():
    rc = np.array([1, 2, 3, 4, 5, -1])
    src = np.array([3.0, 1.0, 2.0, 0.5, 1.5, 0.2])
    ret = np.array([0.2, 0.1, 0.5, 0.0, 0.3, 0.0])
    dep_all, exp_all = sec.retention_route(rc, src, ret)
    # chunk A = hexes 0-2 (its outlet passes to hex 3 of chunk B)
    rc_a = np.array([1, 2, -1])
    dep_a, exp_a = sec.retention_route(rc_a, src[:3], ret[:3])
    src_b = src[3:].copy()
    src_b[0] += exp_a  # the export of chunk A enters chunk B
    dep_b, exp_b = sec.retention_route(np.array([1, 2, -1]), src_b, ret[3:])
    assert np.allclose(np.concatenate([dep_a, dep_b]), dep_all)
    assert exp_b == pytest.approx(exp_all)


def test_i13_reentry_reuses_and_keeps_the_geology(hist_project, hist_config):
    from hexworld import generate

    again = generate(hist_config, reuse=hist_project)
    assert "v52_resources" in again.stats["reused_stages"]
    assert np.array_equal(again.hist.layers["rock_type"], hist_project.hist.layers["rock_type"])
    assert again.resources is hist_project.resources


def test_i14_gui_views_and_report_offscreen(hist_project):
    from hexworld import render, views
    from hexworld.hex_report import hex_report

    hashes = {k: hash(v.tobytes()) for k, v in hist_project.resources.layers.items()}
    for vid in ("soil_fertility", "mineral_resources"):
        spec = views.VIEWS[vid]
        ok, why = spec.availability(hist_project)
        assert ok, why
        for opt in spec.modes:
            rgb = spec.render(hist_project, opt.id, spec.default_overlays())
            assert rgb.shape[:2] == hist_project.layers["plate_id"].shape
            assert spec.legend(hist_project, opt.id)
    # drawing never changes the data (S21 / R21)
    assert hashes == {k: hash(v.tobytes()) for k, v in hist_project.resources.layers.items()}
    land = np.argwhere(hist_project.env.layers["soil_present"])
    r, c = (int(v) for v in land[len(land) // 2])
    text = hex_report(hist_project, c, r).as_text()
    assert "Плодородие почвы (v5.2)" in text and "Полезные ископаемые (v5.2)" in text
    assert render is not None


def test_i15_timings_and_counts_are_reported(hist_project):
    r = hist_project.resources
    for k in ("soil_pass1", "deposits_primary", "aggregate", "total"):
        assert k in r.timings
    assert r.summary["systems"] == len(r.catalog["systems"])


def test_i16_economics_never_changes_geology(hist_project):
    tot = api.compute_in_situ_resources(hist_project)
    before = json.dumps(tot, sort_keys=True, default=float)
    from hexworld.resources.models import recoverable_t

    for row in tot.values():
        recoverable_t(row["in_situ_t"], 0.8, 0.7)  # an economic estimate is a new number
    assert (
        json.dumps(api.compute_in_situ_resources(hist_project), sort_keys=True, default=float)
        == before
    )
    fields = {f for f in Config.__dataclass_fields__["resources"].default_factory().__dict__}
    assert not any("price" in f or "cost" in f for f in fields)


def test_exports(hist_project, tmp_path):
    files = api.export_resources(hist_project, tmp_path)
    for f in files:
        assert (tmp_path / f).exists()
    gj = json.loads((tmp_path / "deposit_systems.geojson").read_text(encoding="utf-8"))
    assert len(gj["features"]) == len(hist_project.resources.catalog["systems"])


def test_s21_v45_soils_are_not_modified(hist_project):
    # v5.2 adds layers; the v4.5 soil fields, rock type and elevation stay as produced
    from hexworld.pipeline_v52 import run_resources

    env_before = {k: v.copy() for k, v in hist_project.env.layers.items()}
    rock = hist_project.hist.layers["rock_type"].copy()
    run_resources(hist_project.config, hist_project)
    for k, v in env_before.items():
        assert np.array_equal(v, hist_project.env.layers[k], equal_nan=True), k
    assert np.array_equal(rock, hist_project.hist.layers["rock_type"])


def test_cli_resources(hist_project, tmp_path, capsys):
    from hexworld.cli import main
    from hexworld.io import save_project

    path = save_project(hist_project, tmp_path / "cli52")
    assert main(["resources", str(path), "--export", str(tmp_path / "ex")]) == 0
    out = capsys.readouterr().out
    assert "systems" in out and "soil P" in out
    assert os.path.exists(tmp_path / "ex" / "resource_totals.csv")


def test_views_have_no_shadowed_module_names(hist_project):
    """A later stage must not silently redefine an earlier view's palette or renderer
    (this shadowing once gave the runoff-regime map the v5.4 rain-regime legend)."""
    import re
    from collections import Counter
    from pathlib import Path

    import hexworld.views as views

    src = Path(views.__file__).read_text(encoding="utf-8")
    names = re.findall(r"^(_[A-Za-z0-9_]+)\s*=", src, re.M)
    names += re.findall(r"^def (_[A-Za-z0-9_]+)\(", src, re.M)
    dup = [n for n, c in Counter(names).items() if c > 1]
    assert not dup, dup
    spec = views.VIEWS["precipitation"] if "precipitation" in views.VIEWS else None
    if spec is not None and any(m.id == "regime" for m in spec.modes):
        labels = [x.label for x in spec.legend(hist_project, "regime")]
        assert any("декабрь" in t for t in labels)
