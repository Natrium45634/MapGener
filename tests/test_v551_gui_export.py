"""TZ 5.5.1 G6: map tabs (T74, T75), hex panel section, section with bodies (T72), export
(T73) and the GUI menu."""

from __future__ import annotations

import csv
import json
import os

import numpy as np
import pytest

from hexworld import views


@pytest.fixture(scope="module")
def proj(hist_project):
    assert hist_project.minerals is not None
    return hist_project


@pytest.mark.parametrize("vid", ["minerals_551", "geology_551", "fluids_551"])
def test_tabs_render_every_mode(proj, vid):
    spec = views.VIEWS[vid]
    assert spec.availability(proj)[0]
    for o in spec.modes:
        img = views.render_view_image(proj, vid, o.id, None)
        assert img.ndim == 3 and img.dtype == np.uint8
        assert spec.legend(proj, o.id)


def test_t74_t75_depth_filter_hides_without_changing_the_plan(proj):
    spec = views.VIEWS["minerals_551"]
    before = proj.minerals.content_hashes()
    all_v = spec.vectors(proj, "all", frozenset())
    shallow = spec.vectors(proj, "all", frozenset({"d100"}))
    n_all = sum(len(v.x) for v in all_v if v.kind == "points")
    n_sh = sum(len(v.x) for v in shallow if v.kind == "points")
    assert n_sh <= n_all
    assert proj.minerals.content_hashes() == before  # nothing deleted, nothing reseeded


def test_tab_unavailable_without_the_plan(proj):
    import copy

    p2 = copy.copy(proj)
    p2.minerals = None
    ok, why = views.VIEWS["minerals_551"].availability(p2)
    assert not ok and "5.5.1" in why


def test_commodity_tab_reads_the_551_plan(proj):
    spec = views.VIEWS["resources_by_type"]
    legend = spec.legend(proj, "com_Fe")[0].label
    n = sum(1 for lot in proj.minerals.plan.lots if lot.get("contained", {}).get("Fe", 0) > 0)
    systems = {lot["system_id"] for lot in proj.minerals.plan.lots
               if lot.get("contained", {}).get("Fe", 0) > 0}  # fmt: skip
    assert f"месторождений {len(systems)}" in legend and n >= len(systems)


def test_hex_report_lists_bodies_crossing_the_hex(proj):
    from hexworld.hex_report import hex_report
    from hexworld.minerals.index import hex_index

    idx = hex_index(proj.minerals, proj.layout)
    w, h = proj.layout.width, proj.layout.height
    (c, r), _b = next((k, v) for k, v in sorted(idx.items()) if 0 <= k[0] < w and 0 <= k[1] < h)
    rep = hex_report(proj, c, r)
    sec = [s for s in rep.sections if s.key == "minerals"]
    assert sec and sec[0].rows and "provisional" in sec[0].rows[0][1]


def test_t72_section_shows_the_bodies(proj):
    from hexworld.geomorph.sections import cross_section

    s = next(s for s in proj.minerals.plan.systems if s.get("body_ids"))
    b = proj.minerals.plan.bodies[s["body_ids"][0]]
    x, y = b.origin_m[0] / 1000.0, b.origin_m[1] / 1000.0
    rows = cross_section(proj, (x - 20.0, y), (x + 20.0, y))
    hits = [iv for r in rows for iv in r.get("ore", []) if iv["body_id"] == b.body_id]
    col = b.column(b.origin_m[0], b.origin_m[1])
    if col:  # the body is on the line: it must appear with its real interval
        assert hits
        assert any(abs(h["z_top_m"] - col[0][0]) < 200.0 or not h["at_profile_point"]
                   for h in hits)  # fmt: skip


def test_t73_export_typed_units_lot_keys_and_crs(proj, tmp_path):
    from hexworld.minerals.export import export_minerals

    files = export_minerals(proj, tmp_path)
    names = {os.path.basename(f) for f in files}
    assert {"minerals_footprints.geojson", "minerals_lots.csv", "minerals_plan.json",
            "minerals_discovery.json", "minerals_manifest.json"} <= names  # fmt: skip
    gj = json.loads((tmp_path / "minerals_footprints.geojson").read_text("utf-8"))
    assert (
        gj["crs"]["properties"]["name"] == "HEXWORLD:PLANAR_KM"
        and "not WGS84" in gj["hexworld_crs"]
    )
    assert gj["features"] and gj["features"][0]["geometry"]["type"] in ("Polygon", "MultiPolygon")
    rows = list(csv.DictReader(open(tmp_path / "minerals_lots.csv", encoding="utf-8")))
    keys = [r["lot_key"] for r in rows]
    assert len(keys) == len(set(keys)) and all(r["unit"] == "t" for r in rows)
    man = json.loads((tmp_path / "minerals_manifest.json").read_text("utf-8"))
    assert "5.5.1" in man["algorithm_version"] and "not WGS84" in man["crs"]


def test_gui_export_menu_and_section_label(proj, tmp_path, monkeypatch):
    pytest.importorskip("PySide6")
    monkeypatch.setenv("HEXWORLD_DATA_DIR", str(tmp_path / "data"))
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    os.environ["HEXWORLD_NO_START_DIALOG"] = "1"
    try:
        from PySide6.QtWidgets import QApplication
    except ImportError as exc:
        pytest.skip(f"Qt unavailable: {exc}")
    QApplication.instance() or QApplication([])
    from hexworld.gui.app import MainWindow

    w = MainWindow()
    w._show_project(proj, "test")
    titles = [a.text() for m in w.menuBar().findChildren(type(w.menuBar().actions()[0].menu()))
              for a in m.actions()]  # fmt: skip
    assert any("Экспорт недр" in t for t in titles)
    files = w.export_minerals(str(tmp_path / "exp"))
    assert len(files) == 5  # footprints, lots, plan, discovery view, manifest
    w.close()
