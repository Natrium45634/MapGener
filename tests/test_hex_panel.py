"""Hex descriptions, the side panel's card / section chooser / fonts, and the sea level floor
when the requested land share cannot be reached."""

from __future__ import annotations

import os

import numpy as np
import pytest

from hexworld import Config
from hexworld.hex_description import BIOME_TEXT, describe
from hexworld.hex_report import hex_report


# -- descriptions ------------------------------------------------------------------------------
def test_every_biome_has_a_description():
    from hexworld.biomes.registry import BIOMES

    missing = [b.id for b in BIOMES if not BIOME_TEXT.get(b.id)]
    assert missing == []


def _land(project):
    land = project.layer("land_mask").astype(bool)
    rows, cols = np.nonzero(land)
    k = len(rows) // 2
    return int(cols[k]), int(rows[k])


def test_description_of_a_land_hex(hist_project):
    col, row = _land(hist_project)
    d = describe(hist_project, col, row)
    assert d["biome"] and d["conditions"]
    labels = [lab for lab, _v in d["facts"]]
    assert "Высота" in labels and "Температура" in labels
    for sentence in d["conditions"]:
        # numbers keep their thousands space, sentences keep their commas
        assert "  " not in sentence and sentence.endswith(".")
    rep = hex_report(hist_project, col, row)
    assert rep.description is not None
    text = rep.as_text()
    assert text.index(d["biome"].capitalize()[:6]) < text.index("Рельеф")


def test_description_of_a_sea_hex(hist_project):
    land = hist_project.layer("land_mask").astype(bool)
    rows, cols = np.nonzero(~land)
    d = describe(hist_project, int(cols[0]), int(rows[0]))
    assert any("Глубина" in lab for lab, _v in d["facts"])


# -- panel ------------------------------------------------------------------------------------
@pytest.fixture
def qapp(tmp_path, monkeypatch):
    pytest.importorskip("PySide6")
    monkeypatch.setenv("HEXWORLD_DATA_DIR", str(tmp_path / "data"))
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    try:
        from PySide6.QtWidgets import QApplication
    except ImportError as exc:
        pytest.skip(f"Qt unavailable: {exc}")
    return QApplication.instance() or QApplication([])


def _keys(tree):
    return [tree.topLevelItem(i).text(0) for i in range(tree.topLevelItemCount())]


def test_panel_shows_card_and_only_the_chosen_sections(qapp, hist_project):
    from hexworld.gui import hex_panel as hp

    col, row = _land(hist_project)
    panel = hp.HexInfoPanel()
    panel.show_hex(hist_project, col, row)
    assert not panel.card.isHidden()
    assert "<b" in panel.card.text() and panel.full_report.description["biome"]
    full = panel.full_report.sections
    shown = {s.key for s in panel.report.sections}
    assert shown <= hp.DEFAULT_SHOWN | {s.key for s in full if s.key not in hp.SECTION_TITLES}
    assert len(panel.report.sections) < len(full)
    assert "+" in panel.all_btn.text()
    # the chooser's choice is remembered and applied
    hp.set_shown_sections(["climate"])
    panel._render()
    assert [s.key for s in panel.report.sections] == ["climate"]
    assert hp.shown_sections() == {"climate"}
    # «Все данные…» still holds everything; copying copies everything
    dlg = hp.FullReportDialog(panel.full_report, panel.pt)
    assert dlg.tree.topLevelItemCount() == len(full)
    assert "Рельеф" in panel.text()
    dlg.close()


def test_panel_font_buttons_are_remembered(qapp, hist_project):
    from hexworld.gui import hex_panel as hp

    panel = hp.HexInfoPanel()
    pt = panel.pt
    panel.bigger_btn.click()
    panel.bigger_btn.click()
    assert panel.pt == pt + 2 and panel.tree.font().pointSizeF() == pt + 2
    assert hp.HexInfoPanel().pt == pt + 2
    panel.smaller_btn.click()
    assert hp.panel_font_pt(0) == pt + 1


def test_sections_dialog_quick_sets(qapp):
    from hexworld.gui import hex_panel as hp

    dlg = hp.SectionsDialog()
    for b in dlg.findChildren(type(dlg.findChild(hp.QPushButton))):
        if b.text() == "Всё":
            b.click()
    dlg._ok()
    assert hp.shown_sections() == set(hp.SECTION_TITLES)


def test_window_font_buttons(qapp):
    from PySide6.QtWidgets import QDialog, QLabel, QPushButton, QVBoxLayout

    from hexworld import saved_worlds as sw
    from hexworld.gui.saved_dialogs import font_buttons

    dlg = QDialog()
    lay = QVBoxLayout(dlg)
    label = QLabel("x")
    lay.addWidget(label)
    lay.addLayout(font_buttons(dlg, "test"))
    base = dlg.font().pointSizeF()
    plus = [b for b in dlg.findChildren(QPushButton) if b.text() == "A+"][0]
    plus.click()
    assert label.font().pointSizeF() == base + 1
    assert sw.load_prefs()["font_test"] == base + 1


# -- sea level floor ---------------------------------------------------------------------------
def test_sea_level_floor():
    from hexworld.stages.sea import resolve_sea_level

    z = np.linspace(-6000.0, 3000.0, 1001)
    import dataclasses

    params = dataclasses.replace(Config().sea, target_land_fraction=0.8)
    free = resolve_sea_level(z, params)
    assert free < -3000
    assert resolve_sea_level(z, params, floor_m=-650.0) == -650.0


def test_unreachable_land_share_is_reported():
    from hexworld import world_tools as wt

    values = {
        "sea.sea_mode": "target_land_fraction",
        "sea.target_land_fraction": 0.49,
        "crust.continental_fraction": 0.40,
    }
    assert any("суш" in m.lower() for m in wt.conflicts(values))
    values["sea.target_land_fraction"] = 0.40
    assert not any("суш" in m.lower() for m in wt.conflicts(values))
