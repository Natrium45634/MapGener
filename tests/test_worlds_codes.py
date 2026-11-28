"""World codes HW2 (every setting, compact, version-stable), saved worlds (one folder per
world), the visible map tabs, and the bounded uplift of the v4 landscape evolution."""

from __future__ import annotations

import hashlib
import json

import numpy as np
import pytest

from hexworld import Config
from hexworld import saved_worlds as sw
from hexworld import world_tools as wt


# -- world codes ----------------------------------------------------------------------------------
def _custom():
    d = json.loads(Config(seed="lKsDscB4", width=1000, height=800, hex_spacing_km=25.0,
                          mode="world").to_json())  # fmt: skip
    d["erosion"]["duration_ma"] = 20.0
    d["history"]["uplift_scale"] = 1.8
    d["mountains"]["max_elevation_m"] = 5000.0
    d["atmosphere"]["weather_year"] = True
    d["terrain"]["basins"] = ["rift_graben"]
    return Config.from_dict(d)


def test_code_restores_every_setting_and_is_short():
    cfg = _custom()
    code = wt.world_code(cfg, "v4")
    assert code.startswith("HW2-") and len(code) < 160
    back, through = wt.parse_world_code(code)
    assert back == cfg and through == "v4"
    assert back.to_dict() == cfg.to_dict()
    assert len(wt.world_code(Config(seed="Ab12"))) < 64


def test_code_does_not_depend_on_this_builds_defaults(monkeypatch):
    """The code stores changes against the frozen dictionary, not the running defaults."""
    cfg = _custom()
    code = wt.world_code(cfg)
    frozen = wt._frozen_defaults(1)
    assert frozen["erosion"]["duration_ma"] == 2.0  # the frozen text, whatever Config() says
    # decoding builds the configuration from the frozen text + the stored changes
    back, _t = wt.parse_world_code(code)
    assert back.erosion.duration_ma == 20.0 and back.seed == "lKsDscB4"


def test_frozen_dictionary_never_changes():
    from hexworld._world_code_dict import DICT_1

    assert hashlib.sha256(DICT_1.encode()).hexdigest()[:16] == "b551c8c58d9e401f"


def test_damaged_or_foreign_codes_are_rejected():
    code = wt.world_code(_custom())
    bad = code[:-6] + ("A" if code[-6] != "A" else "B") + code[-5:]
    with pytest.raises(ValueError):
        wt.parse_world_code(bad)
    with pytest.raises(ValueError):
        wt.parse_world_code("XX1-abc")
    assert wt.is_world_code(" " + code) and not wt.is_world_code("Terra7")


def test_old_hw1_codes_still_open():
    import base64
    import zlib

    data = {"s": "Old1", "d": {"width": 60}}
    raw = json.dumps(data, separators=(",", ":")).encode()
    old = "HW1-" + base64.urlsafe_b64encode(zlib.compress(raw, 9)).decode().rstrip("=")
    cfg, through = wt.parse_world_code(old)
    assert cfg.seed == "Old1" and cfg.width == 60 and through == "v50"


def test_configs_saved_before_the_uplift_decay_keep_their_behaviour():
    d = json.loads(Config().to_json())
    d["erosion"].pop("uplift_decay_ma")
    assert Config.from_dict(d).erosion.uplift_decay_ma == 0.0
    assert Config().erosion.uplift_decay_ma == 3.0


@pytest.fixture
def qapp():
    import os

    pytest.importorskip("PySide6")
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    try:
        from PySide6.QtWidgets import QApplication
    except ImportError as exc:
        pytest.skip(f"Qt unavailable: {exc}")
    return QApplication.instance() or QApplication([])


# -- saved worlds ----------------------------------------------------------------------------
@pytest.fixture()
def worlds(tmp_path, monkeypatch):
    monkeypatch.setenv("HEXWORLD_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("HEXWORLD_WORLDS_DIR", str(tmp_path / "worlds"))
    return tmp_path / "worlds"


def test_save_code_only_world_and_reopen(worlds):
    cfg = _custom()
    folder = sw.save_code(cfg, "Мой мир: тест/1", through="v4", notes="горы")
    assert folder.parent == worlds and "/" not in folder.name and ":" not in folder.name
    assert (folder / "code.txt").read_text(encoding="utf-8").strip().startswith("HW2-")
    lst = sw.list_worlds()
    assert len(lst) == 1 and lst[0]["title"] == "Мой мир: тест/1" and not lst[0]["has_data"]
    project, back, through = sw.open_world(folder)
    assert project is None and back == cfg and through == "v4"
    assert sw.recent_worlds() == [folder]
    again = sw.save_code(cfg, "Мой мир: тест/1")
    assert again != folder and again.name.endswith("(2)")


def test_save_world_with_data_rename_delete(worlds, hist_project):
    folder = sw.save_world(hist_project, "Ab12 мир", notes="заметка")
    meta = sw.read_meta(folder)
    assert meta["has_data"] and (folder / "project" / "manifest.json").exists()
    assert (folder / "preview.png").exists() and meta["notes"] == "заметка"
    project, cfg, _t = sw.open_world(folder)
    assert project is not None and cfg == hist_project.config
    assert np.array_equal(project.layer("elevation_m"), hist_project.layer("elevation_m"))
    new = sw.rename_world(folder, "Новое имя")
    assert new.name == "Новое имя" and sw.read_meta(new)["title"] == "Новое имя"
    sw.delete_world(new)
    assert sw.list_worlds() == []
    with pytest.raises(ValueError):
        sw.delete_world(worlds)  # never deletes a folder that is not a saved world


def test_safe_names():
    assert sw.safe_name('a<b>c:"d"|e?*') == "a_b_c_d_e_"
    assert sw.safe_name("CON") == "_CON" and sw.safe_name("   ") == "Мир"
    assert sw.safe_name("Мир Аурелия") == "Мир Аурелия"


def test_visible_views_preference(worlds):
    ids = ["plates", "heights", "biomes", "soils"]
    assert sw.visible_views(ids) == ids  # nothing chosen: all
    sw.set_visible_views(["heights", "biomes"], ids)
    assert sw.visible_views(ids) == ["heights", "biomes"]
    sw.set_visible_views([], ids)
    assert sw.visible_views(ids) == ids  # never an empty tab bar


def test_main_window_shows_only_chosen_tabs(worlds, qapp):
    import os

    from hexworld import views
    from hexworld.gui.app import MainWindow

    os.environ["HEXWORLD_NO_START_DIALOG"] = "1"
    sw.set_visible_views(["heights", "biomes"], list(views.VIEWS))
    w = MainWindow()
    titles = [w.tabs.tabText(i) for i in range(w.tabs.count())]
    assert titles == [views.VIEWS["heights"].title, views.VIEWS["biomes"].title]
    assert w.current_view().id in ("heights", "biomes")
    sw.set_visible_views(list(views.VIEWS), list(views.VIEWS))
    w._rebuild_tabs()
    assert w.tabs.count() == len(views.VIEWS)


def test_settings_dialog_keeps_settings_without_fields(qapp):
    from hexworld.gui.world_settings import WorldSettingsDialog

    d = json.loads(Config(seed="Keep1").to_json())
    d["reference_cells"] = 256  # no field in the dialog
    cfg = Config.from_dict(d)
    dlg = WorldSettingsDialog(cfg, "v50")
    dlg.seed.setText(wt.world_code(cfg))  # a pasted code fills seed + settings
    assert dlg.seed.text() == "Keep1"
    code_cfg, _t = wt.parse_world_code(dlg.world_code())
    assert code_cfg.reference_cells == 256


# -- bounded uplift (the flat-plateau defect) ----------------------------------------------
def test_uplift_decays_and_skips_the_deep_sea_floor():
    from hexworld.stages.landscape import _uplift_reach, _uplift_step

    T, tau, n = 20e6, 3e6, 8
    total = sum(_uplift_step(k * T / n, T / n, tau) for k in range(n))
    assert total == pytest.approx(tau * (1 - np.exp(-T / tau)))
    assert total < 0.2 * T  # a pulse, not 20 Ma at the epoch's rate
    assert _uplift_step(0.0, 5.0, 0.0) == 5.0  # constant rate kept for old projects
    reach = _uplift_reach(np.array([-4000.0, -2500.0, -100.0, 300.0]), 0.0)
    assert reach[0] == 0.0 and reach[1] == 0.0 and reach[2] == 1.0 and reach[3] == 1.0


@pytest.mark.slow
def test_long_erosion_strong_uplift_gives_no_plateau_continents():
    from hexworld import generate

    d = json.loads(Config(seed="lKsDscB4", width=125, height=100, hex_spacing_km=200.0,
                          mode="world").to_json())  # fmt: skip
    d["erosion"]["duration_ma"] = 20.0
    d["history"]["uplift_scale"] = 1.8
    p = generate(Config.from_dict(d), through="v4")
    z = p.hydro.layers["elevation_m"].astype(float) - p.current_sea_level_m
    land = p.hydro.layers["land_mask"].astype(bool)
    assert np.percentile(z[land], 50) < 1000.0  # lowlands stay lowlands
    assert (z[land] > 0.85 * p.config.mountains.max_elevation_m).mean() < 0.01


def test_cli_worlds(worlds, hist_project, tmp_path, capsys):
    from hexworld.cli import main
    from hexworld.io import save_project

    path = save_project(hist_project, tmp_path / "p")
    assert main(["worlds", "--save", str(path), "--title", "CLI мир", "--code-only"]) == 0
    assert main(["worlds"]) == 0
    out = capsys.readouterr().out
    assert "CLI мир" in out and "HW2-" in out and "[code]" in out


def test_left_panel_hides_and_returns_in_one_click(worlds, qapp):
    import os

    from hexworld.gui.app import MainWindow

    os.environ["HEXWORLD_NO_START_DIALOG"] = "1"
    w = MainWindow()
    assert not w.side_panel.isHidden() and w.side_toggle.text() == "◀"
    w.side_toggle.click()
    assert w.side_panel.isHidden() and w.side_toggle.text() == "▶"
    assert not w.side_action.isChecked()
    assert sw.load_prefs()["side_panel_hidden"] is True
    again = MainWindow()  # remembered
    assert again.side_panel.isHidden()
    again.side_toggle.click()
    assert not again.side_panel.isHidden() and again.side_action.isChecked()
    w.close()
    again.close()
