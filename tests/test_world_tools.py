"""World code, sketch, conflicts, partial regeneration plan, library, names, game export,
globe, climate resolution independence, exact crust speed-up, and the new windows."""

from __future__ import annotations

import os
import struct
import zlib
from dataclasses import replace

import numpy as np
import pytest

from hexworld import settings_schema as ss
from hexworld import world_tools as wt
from hexworld.config import Config


# -- world code --------------------------------------------------------------------------------
def test_world_code_round_trip():
    c = Config.from_dict(
        {**Config().to_dict(), "seed": "Code7", "width": 240, "height": 160,
         "climate": {**Config().to_dict()["climate"], "temperature_offset_c": -4.5}}
    )  # fmt: skip
    code = wt.world_code(c, "v47")
    assert code.startswith("HW1-") and len(code) < 300
    back, through = wt.parse_world_code(code)
    assert back == c and through == "v47"
    assert wt.parse_world_code(wt.world_code(Config()))[0] == Config()
    # whitespace from a chat or an e-mail does not matter
    spaced = code[:20] + "\n " + code[20:]
    assert wt.parse_world_code(spaced)[0] == c


@pytest.mark.parametrize("bad", ["", "HW2-abc", "HW1-!!!", "HW1-" + "A" * 40])
def test_bad_world_codes_are_rejected(bad):
    with pytest.raises(ValueError):
        wt.parse_world_code(bad)


# -- sketch, conflicts, plan -------------------------------------------------------------------
def test_preview_config_keeps_the_window():
    c = Config(width=360, height=240, hex_spacing_km=25.0, mode="world", seed="Pv")
    p = wt.preview_config(c)
    assert max(p.width, p.height) <= 96 and p.seed == c.seed
    span = lambda cfg: cfg.hex_spacing_km * (cfg.width - 1)  # noqa: E731
    assert span(p) == pytest.approx(span(c), rel=1e-6)
    assert not p.erosion.enabled and not p.terrain.enabled
    small = Config(width=60, height=40)
    assert wt.preview_config(small).width == 60


def test_conflicts_name_contradictions():
    v = ss.values_from_config(Config())
    assert wt.conflicts(v) == []
    v2 = dict(v, **{"plates.count": 3, "continents.major_landmass_count": 6})
    assert any("материков" in m for m in wt.conflicts(v2))
    v3 = dict(v, **{"climate.temperature_offset_c": -12.0, "climate.precipitation_scale": 2.0})
    msgs = wt.conflicts(v3)
    assert any("холоднее" in m for m in msgs) and any("ледниковые" in m for m in msgs)
    v4 = dict(v, **{"climate.enabled": False})
    assert any("Биомы" in m for m in wt.conflicts(v4))


def test_regeneration_plan_follows_the_changed_section():
    a = Config(seed="Re1")
    climate = replace(a, climate=replace(a.climate, precipitation_scale=1.3))
    plan = wt.regeneration_plan(a, climate)
    assert plan["first"] == "v4"
    assert "Геологическая история" in plan["reused"]
    assert plan["computed"][0] == "Климат, реки, эрозия"
    assert wt.regeneration_plan(a, replace(a, seed="Re2"))["first"] == "v1"
    assert wt.regeneration_plan(a, a)["first"] is None
    biomes = replace(a, biomes=replace(a.biomes, pft_tau=0.3))
    assert wt.regeneration_plan(a, biomes)["first"] == "v50"


def test_regeneration_plan_matches_the_pipeline(hist_config, hist_project):
    """What the plan says can be reused is what generate() actually reuses."""
    from hexworld import generate

    new = replace(hist_config, climate=replace(hist_config.climate, precipitation_scale=1.2))
    plan = wt.regeneration_plan(hist_config, new)
    assert plan["first"] == "v4"
    p = generate(new, reuse=hist_project)
    reused = set(p.stats["reused_stages"])
    assert {"plates", "crust", "geo_history"} <= reused
    assert "v4_climate" not in reused


def test_affected_views():
    from hexworld.views import VIEWS

    climate = wt.affected_views("climate.temperature_offset_c")
    assert VIEWS["temperature"].title in climate and VIEWS["biomes"].title in climate
    assert VIEWS["plates"].title not in climate
    assert VIEWS["plates"].title in wt.affected_views("plates.count")


def test_summary_and_thumbnail(hist_project):
    s = wt.world_summary(hist_project)
    assert 0 < s["land_share"] < 1 and s["landmasses"] >= 1 and s["top_biomes"]
    rows = wt.summary_rows(s)
    assert ("Доля суши", f"{s['land_share']:.0%}") in rows
    img = wt.thumbnail(hist_project, "biomes", 200)
    assert img.dtype == np.uint8 and img.shape[2] == 3 and max(img.shape[:2]) <= 200


# -- library -------------------------------------------------------------------------------------
def test_library_history_favourites(tmp_path, hist_project, monkeypatch):
    from hexworld import world_library as wl

    lib = wl.WorldLibrary(tmp_path)
    e = lib.add(hist_project)
    assert lib.thumb_path(e) is not None and e["code"].startswith("HW1-")
    lib.set_favourite(e["id"])
    lib.rename(e["id"], "Мой первый мир")
    again = wl.WorldLibrary(tmp_path)  # survives a restart
    assert again.get(e["id"])["favourite"] and again.get(e["id"])["title"] == "Мой первый мир"
    monkeypatch.setattr(wl, "MAX_HISTORY", 2)
    for _ in range(3):
        lib.entries.insert(0, dict(e, id=os.urandom(6).hex(), favourite=False, thumb=None))
    lib._prune()
    assert len([x for x in lib.entries if not x["favourite"]]) == 2
    assert lib.get(e["id"]) is not None  # favourites are never dropped
    lib.remove(e["id"])
    assert lib.get(e["id"]) is None


# -- names ------------------------------------------------------------------------------------
def test_names_are_stable_and_cyrillic(hist_project):
    from hexworld import names

    assert names.ru("Pelomund") == "Пеломунд" and names.ru("Quenanmar") == "Квенанмар"
    assert names.make_names("S1", "river", 5) == names.make_names("S1", "river", 5)
    assert names.make_names("S1", "river", 5) != names.make_names("S2", "river", 5)
    labels = names.place_labels(hist_project)
    kinds = {lab.kind for lab in labels}
    assert {"landmass", "mountains"} <= kinds
    for lab in labels:
        assert all(not ("a" <= ch.lower() <= "z") for ch in lab.title_ru), lab.title_ru
    b = hist_project.geometry.viewport_bounds
    for lab in labels:
        assert b.xmin - 1 <= lab.x_km <= b.xmax + 1 and b.ymin - 1 <= lab.y_km <= b.ymax + 1


# -- game export -------------------------------------------------------------------------------
def _read_png16(path):
    data = path.read_bytes()
    assert data[:8] == b"\x89PNG\r\n\x1a\n"
    w, h, depth, ctype = struct.unpack(">IIBB", data[16:26])
    assert depth == 16 and ctype == 0
    pos, idat = 8, b""
    while pos < len(data):
        n = struct.unpack(">I", data[pos : pos + 4])[0]
        tag = data[pos + 4 : pos + 8]
        if tag == b"IDAT":
            idat += data[pos + 8 : pos + 8 + n]
        pos += 12 + n
    raw = np.frombuffer(zlib.decompress(idat), np.uint8).reshape(h, 1 + 2 * w)
    return raw[:, 1:].copy().view(">u2").reshape(h, w)


def test_game_export(tmp_path, hist_project):
    import json

    from hexworld.export_game import export_game

    files = export_game(hist_project, tmp_path / "game", px_per_hex=3)
    names = {f.name for f in files}
    for n in ("heightmap.png", "heightmap.r16", "biomes.png", "biome_ids.png", "water.png",
              "rivers.png", "hexes.csv", "hexes.json", "world.json", "ИМПОРТ.txt"):  # fmt: skip
        assert n in names, n
    meta = json.loads((tmp_path / "game" / "world.json").read_text("utf-8"))
    hm = _read_png16(tmp_path / "game" / "heightmap.png")
    assert list(hm.shape[::-1]) == meta["image_size_px"]
    assert hm.min() == 0 and hm.max() == 65535
    raw = np.frombuffer((tmp_path / "game" / "heightmap.r16").read_bytes(), "<u2")
    assert np.array_equal(raw.reshape(hm.shape), hm)
    # metres back from the 16-bit value
    top = meta["height_min_m"] + hm.max() / 65535 * meta["height_range_m"]
    assert top == pytest.approx(meta["height_max_m"])
    lines = (tmp_path / "game" / "hexes.csv").read_text("utf-8").splitlines()
    lay = hist_project.layout
    assert len(lines) == 1 + lay.width * lay.height
    assert lines[0].startswith("col,row,q,r,x_km,y_km,height_m,water")
    assert meta["names"] and meta["biomes"]


# -- globe ---------------------------------------------------------------------------------------
def test_globe(hist_project):
    from hexworld import views
    from hexworld.globe import SPACE, render_globe, window_extent

    spec = views.VIEWS["heights"]
    rgb = views.view_rgb(hist_project, "heights", spec.default_mode(), ())
    img = render_globe(hist_project, rgb, 200, lon0=0.0, lat0=20.0)
    assert img.shape == (200, 200, 3)
    assert tuple(img[0, 0]) == SPACE  # corner: space
    assert len(np.unique(img.reshape(-1, 3), axis=0)) > 50  # the map is on the sphere
    ext = window_extent(hist_project)
    assert 0 < ext["surface_share"] <= 1 and ext["lat_north"] > ext["lat_south"]


# -- climate -------------------------------------------------------------------------------------
def test_fu_share():
    from hexworld.stages.climate import fu_share

    assert fu_share(0.0) == 0.0
    assert fu_share(1.0) == pytest.approx(2 - 2 ** (1 / 2.6))
    assert fu_share(50.0) > 0.98 and np.all(np.diff(fu_share(np.linspace(0, 10, 50))) > 0)


def test_climate_does_not_depend_on_the_world_raster_resolution():
    """Moisture decays per km, not per cell: the same world at two skeleton resolutions."""
    from hexworld import generate

    base = Config(seed="Res1", width=60, height=40, hex_spacing_km=75.0, mode="world")
    land_p = []
    for cells in (160, 320):
        c = wt.preview_config(base)
        c = replace(c, continents=replace(c.continents, skeleton_cells=cells))
        p = generate(c, through="v4")
        land_p.append(p.hydro.summary["land_mean_precipitation_mm_yr"])
    assert land_p[1] == pytest.approx(land_p[0], rel=0.25), land_p


def test_crust_speedup_is_exact(small_project):
    """The pruned nucleus search returns exactly the brute-force maximum."""
    from hexworld.stages import crust as C

    model = small_project.cache["crust"]
    model = model[0] if isinstance(model, tuple) else model
    b = small_project.geometry.simulation_bounds
    rng = np.random.default_rng(3)
    x = rng.uniform(b.xmin, b.xmax, 20000)
    y = rng.uniform(b.ymin, b.ymax, 20000)
    ca, sa, a = np.cos(model.angle), np.sin(model.angle), np.sqrt(model.stretch)
    fast = model._nearest_nucleus(x, y, ca, sa, a)
    dx = x[:, None] - model.centers[None, :, 0]
    dy = y[:, None] - model.centers[None, :, 1]
    u = (ca * dx + sa * dy) / a
    v = (-sa * dx + ca * dy) * a
    brute = np.max(model.radii[None, :] - np.hypot(u, v), axis=1)
    assert np.array_equal(fast, brute)
    assert isinstance(C.CrustModel, type)


# -- windows -------------------------------------------------------------------------------------
@pytest.fixture
def qapp():
    pytest.importorskip("PySide6")
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    try:
        from PySide6.QtWidgets import QApplication
    except ImportError as exc:
        pytest.skip(f"Qt unavailable: {exc}")
    return QApplication.instance() or QApplication([])


def test_library_and_compare_windows(qapp, tmp_path, hist_project):
    from hexworld.gui.worlds_dialogs import CompareDialog, LibraryDialog
    from hexworld.world_library import WorldLibrary

    lib = WorldLibrary(tmp_path)
    a = lib.add(hist_project)
    lib.add(hist_project, title="второй")
    dlg = LibraryDialog(lib)
    assert dlg.list.count() == 2
    dlg.filter.setCurrentIndex(1)
    assert dlg.list.count() == 0
    lib.set_favourite(a["id"])
    dlg.fill()
    assert dlg.list.count() == 1
    dlg.list.item(0).setSelected(True)
    dlg._finish("generate")
    assert dlg.action == "generate" and dlg.entry["id"] == a["id"]
    cmp_ = CompareDialog([("A", a["summary"], None), ("B", a["summary"], None)])
    assert cmp_.table.rowCount() >= 10


def test_batch_window_configs(qapp):
    from hexworld.gui.worlds_dialogs import BatchDialog

    values = ss.values_from_config(Config(width=60, height=40))
    dlg = BatchDialog(values, "Base", {"climate.temperature_offset_c"}, None)
    cfgs = dlg.make_configs(5)
    assert len({c.seed for c in cfgs}) == 5 and all(c.width == 60 for c in cfgs)
    dlg.mode.setCurrentIndex(1)
    for c in dlg.make_configs(4):
        assert c.climate.temperature_offset_c == values["climate.temperature_offset_c"]


def test_settings_window_code_and_warnings(qapp):
    from hexworld.gui.world_settings import WorldSettingsDialog

    base = Config(seed="Win1")
    d = WorldSettingsDialog(base, current=base, preview=False)
    d.rows["climate.temperature_offset_c"].set(-12.0)
    d._update_footer()
    assert "холоднее" in d.footer.text() and "пересчёт начнётся" in d.footer.text()
    code = d.world_code()
    d2 = WorldSettingsDialog(Config(), preview=False)
    d2.apply_code(code)
    assert d2.seed.text() == "Win1" and d2.values()["climate.temperature_offset_c"] == -12.0


def test_main_window_menu_globe_and_names(qapp, hist_project):
    from PySide6.QtGui import QImage

    from hexworld.gui.app import MainWindow
    from hexworld.gui.worlds_dialogs import GlobeDialog

    os.environ["HEXWORLD_NO_START_DIALOG"] = "1"
    w = MainWindow()
    titles = [a.text() for a in w.menuBar().actions()]
    assert titles[:3] == ["&Мир", "&Вид", "&Справка"]
    w._show_project(hist_project, "test")
    w.view.resize(900, 600)
    w.view.fit()
    w.view.flush()
    img = QImage(900, 600, QImage.Format.Format_RGB32)
    w.view.render(img)
    assert w.view.labels_drawn  # names are on the map
    g = GlobeDialog(hist_project)
    g.lon0 = 30.0
    g.redraw()
    assert not g.image.pixmap().isNull()
    w.close()
