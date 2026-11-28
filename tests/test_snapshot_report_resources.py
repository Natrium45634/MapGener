"""One-click layer snapshot, the hex «for a report» text and the «Ресурсы по видам» tab."""

from __future__ import annotations

import os

import numpy as np
import pytest

from hexworld import saved_worlds as sw
from hexworld import views


@pytest.fixture
def qapp(tmp_path, monkeypatch):
    pytest.importorskip("PySide6")
    monkeypatch.setenv("HEXWORLD_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("HEXWORLD_SNAPSHOT_DIR", str(tmp_path / "snaps"))
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    os.environ["HEXWORLD_NO_START_DIALOG"] = "1"
    try:
        from PySide6.QtWidgets import QApplication
    except ImportError as exc:
        pytest.skip(f"Qt unavailable: {exc}")
    return QApplication.instance() or QApplication([])


def _land(project):
    rows, cols = np.nonzero(project.layer("land_mask").astype(bool))
    k = len(rows) // 2
    return int(cols[k]), int(rows[k])


def test_snapshot_one_click_into_the_folder(qapp, hist_project, tmp_path):
    from hexworld.gui.app import MainWindow

    w = MainWindow()
    w._show_project(hist_project, "test")
    path = w.snapshot()
    assert path is not None and path.exists() and path.parent == tmp_path / "snaps"
    assert path.name.startswith(hist_project.config.seed) and path.suffix == ".png"
    assert w.current_view().id in path.name
    w.close()


def test_snapshot_names_are_safe():
    import datetime as dt

    name = sw.snapshot_name("Ab:12", "heights", "hypsometric", dt.datetime(2026, 11, 28, 9, 5, 7))
    assert name == "Ab_12_heights_hypsometric_20261128-090507.png"


def test_hex_report_text_for_a_bug_report(qapp, hist_project):
    from PySide6.QtWidgets import QApplication

    from hexworld.gui.app import MainWindow

    w = MainWindow()
    w._show_project(hist_project, "test")
    c, r = _land(hist_project)
    w.view.select_hex((c, r))
    w._on_hex_clicked(c, r)
    text = w.hex_panel.report_text()
    assert "HW2-" in text and f"гекс ({c}, {r})" in text and "Вкладка:" in text
    assert "км" in text and "Замечание:" in text and len(text.splitlines()) <= 8
    w.hex_panel.report_btn.click()
    assert QApplication.clipboard().text() == text
    w.close()


def test_resources_by_type_tab(hist_project):
    spec = views.VIEWS["resources_by_type"]
    assert spec.availability(hist_project)[0]
    assert {o.id for o in spec.modes} >= {"com_coal", "com_oil", "com_gas", "com_Fe", "com_Au"}
    cat = hist_project.resources.catalog["systems"]
    for el in ("coal", "oil", "Au"):
        img = views.render_view_image(hist_project, "resources_by_type", f"com_{el}", None)
        assert img.ndim == 3 and img.shape[2] == 3
        # the tab reads the 5.5.1 plan when the world has one, else the 5.2 catalogue
        n = len(views._commodity_systems(hist_project, el, False))
        if hist_project.minerals is None:
            assert n == sum(1 for s in cat if s["contained_t"].get(el, 0) > 0)
        legend = spec.legend(hist_project, f"com_{el}")
        assert f"месторождений {n}" in legend[0].label
        marks = spec.vectors(hist_project, f"com_{el}", frozenset())
        assert sum(len(v.x) for v in marks) == n
    # the background is a black-and-white relief: grey everywhere without overlays
    img = spec.render(hist_project, "com_diamond", frozenset())
    no_dia = not views._commodity_systems(hist_project, "diamond", False)
    if no_dia:
        assert np.all(img[..., 0] == img[..., 1]) and np.all(img[..., 1] == img[..., 2])


def test_every_commodity_of_the_catalogue_has_a_mode():
    from hexworld.resources.models import MODELS

    elements = {el for m in MODELS.values() for el in m.grades}
    assert elements <= {k for k, *_r in views.COMMODITIES}
