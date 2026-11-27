"""Hex information panel (report + Qt panel) and the tiled map widget."""

from __future__ import annotations

import os
import re

import numpy as np
import pytest

from hexworld.hex_report import hex_report


def _land_and_sea(project):
    land = np.argwhere(project.layer("dry_land_mask"))
    sea = np.argwhere(project.layer("ocean_mask"))
    r, c = land[len(land) // 2]
    rs, cs = sea[len(sea) // 2]
    return (int(c), int(r)), (int(cs), int(rs))


def test_report_of_a_land_hex_has_every_theme(hist_project):
    (c, r), _sea = _land_and_sea(hist_project)
    rep = hex_report(hist_project, c, r)
    keys = [s.key for s in rep.sections]
    for key in ("place", "relief", "geology", "climate", "months", "water", "soils", "biomes"):
        assert key in keys, key
    assert rep.title.startswith(f"Гекс ({c}, {r})") and "суша" in rep.title
    assert len(rep.section("months").rows) == 12
    assert rep.climate is not None and len(rep.climate["t"]) == 12
    text = rep.as_text()
    assert not re.search(r"\bnan\b", text, re.IGNORECASE) and "None" not in text
    for s in rep.sections:
        assert s.rows and all(isinstance(v, str) and v for _k, v in s.rows)


def test_report_of_a_sea_hex(hist_project):
    _land, (c, r) = _land_and_sea(hist_project)
    rep = hex_report(hist_project, c, r)
    assert "море" in rep.title and "глубина" in rep.subtitle
    assert "soils" not in [s.key for s in rep.sections]
    sea = rep.section("sea")
    assert sea is not None and any(k == "Провинция дна" for k, _v in sea.rows)


def test_report_of_an_early_stage_project(small_config):
    from hexworld import generate

    rep = hex_report(generate(small_config, through="v2"), 10, 10)
    keys = [s.key for s in rep.sections]
    assert "place" in keys and "geology" in keys and "climate" not in keys


def test_relief_heights_are_above_sea_level(hist_project):
    (c, r), _sea = _land_and_sea(hist_project)
    rep = hex_report(hist_project, c, r)
    info = hist_project.hex_info(c, r)
    rows = dict(rep.section("relief").rows)
    expect = info["elevation_m"] - info["sea_level_m"]
    assert (
        rows["Высота после эрозии"].replace(" ", "").startswith(f"{expect:+,.0f}".replace(",", ""))
    )


@pytest.fixture
def qapp():
    pytest.importorskip("PySide6")
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    try:
        from PySide6.QtWidgets import QApplication
    except ImportError as exc:
        pytest.skip(f"Qt unavailable: {exc}")
    return QApplication.instance() or QApplication([])


def test_tiles_are_reused_when_panning(qapp, hist_project):
    from PySide6.QtGui import QImage

    from hexworld import views
    from hexworld.gui.map_view import TILE, MapView

    mv = MapView()
    mv.resize(800, 500)
    mv.set_project(hist_project)
    spec = views.VIEWS["hydrology"]
    mv.set_view("hydrology", spec.default_mode(), spec.default_overlays())
    img = QImage(800, 500, QImage.Format.Format_RGB32)
    mv.render(img)  # the first frame of a view is drawn at once
    assert mv._all_visible_ready()
    first = mv.tiles_rendered
    mv._cx += 0.25 * TILE * mv.km_per_px  # pan by a quarter tile
    mv.render(img)
    # at most one new column of tiles, never the whole view again
    assert mv.tiles_rendered - first <= len(mv._visible_tiles())
    mv.flush()
    before = mv.tiles_rendered
    mv._cx -= 0.25 * TILE * mv.km_per_px  # back: everything is cached
    mv.render(img)
    assert mv.tiles_rendered == before


def test_zoom_shows_the_old_frame_until_tiles_are_ready(qapp, hist_project):
    from PySide6.QtGui import QImage

    from hexworld.gui.map_view import MapView

    mv = MapView()
    mv.resize(640, 400)
    mv.set_project(hist_project)
    img = QImage(640, 400, QImage.Format.Format_RGB32)
    mv.render(img)
    k = mv.km_per_px
    mv.zoom_by(0.5)
    assert mv.km_per_px == pytest.approx(0.5 * k)
    assert mv._backdrop is not None  # the previous frame stands in
    mv.render(img)
    while mv._pump.isActive():
        mv._render_some()
    assert mv._all_visible_ready() and mv._backdrop is None


def test_tiles_match_the_whole_map_render(qapp, hist_project):
    """Tiles are a cache only: the composed frame equals a direct render of the view."""
    from PySide6.QtGui import QImage

    from hexworld import render, views
    from hexworld.gui.map_view import MapView

    mv = MapView()
    mv.resize(512, 384)
    mv.set_project(hist_project)
    mv.set_view("heights", views.VIEWS["heights"].default_mode(), ())
    mv.show_names = False  # labels are drawn over the tiles
    mv.flush()
    img = QImage(512, 384, QImage.Format.Format_RGB888)
    mv.render(img)
    ptr = img.constBits()
    got = np.frombuffer(ptr, np.uint8).reshape(384, img.bytesPerLine())[:, : 512 * 3]
    got = got.reshape(384, 512, 3)
    x0, y0 = mv._origin()
    want = render.render_view(hist_project.layout, mv._hex_rgb, x0, y0, mv.km_per_px, 512, 384)
    assert (np.abs(got.astype(int) - want.astype(int)) > 2).mean() < 0.002


def test_click_opens_the_hex_panel(qapp, hist_project):
    from hexworld.gui.app import MainWindow

    os.environ["HEXWORLD_NO_START_DIALOG"] = "1"
    w = MainWindow()
    w._show_project(hist_project, "test")
    (c, r), _sea = _land_and_sea(hist_project)
    w.view.select_hex((c, r))
    w._on_hex_clicked(c, r)
    assert not w.hex_dock.isHidden()
    panel = w.hex_panel
    assert panel.title.text().startswith(f"Гекс ({c}, {r})")
    assert len(panel.full_report.sections) >= 8  # everything, incl. technical sections
    assert 0 < panel.tree.topLevelItemCount() < len(panel.full_report.sections)
    assert "Положение" in panel.text()
    assert not panel.card.isHidden()
    panel.search.setText("почв")
    shown = [
        panel.tree.topLevelItem(i).text(0)
        for i in range(panel.tree.topLevelItemCount())
        if not panel.tree.topLevelItem(i).isHidden()
    ]
    assert shown and any("почв" in t.lower() for t in shown)
    panel.search.setText("")
    panel.zoom_requested.emit(c, r)
    assert w.view.km_per_px == pytest.approx(hist_project.layout.spacing_km / 40.0)
    panel.close_requested.emit()
    assert w.hex_dock.isHidden() and w.view.selected_hex is None
    w.close()


def test_report_of_a_saved_and_loaded_project(tmp_path, hist_project):
    """Saved projects keep less (e.g. river polylines, not cell lists): the report still works."""
    from hexworld.io import load_project, save_project

    p = load_project(save_project(hist_project, tmp_path / "w"))
    rivers = np.argwhere(p.layer("river_mask"))
    for r, c in rivers[:: max(1, len(rivers) // 5)]:
        rep = hex_report(p, int(c), int(r))
        rows = dict(rep.section("water").rows)
        assert "Название реки" in rows
    (c, r), _sea = _land_and_sea(p)
    assert hex_report(p, c, r).section("biomes") is not None
