import os

import numpy as np
import pytest

from hexworld import load_project, render
from hexworld.cli import main


def test_cli_generate_info_render(tmp_path, capsys):
    out = tmp_path / "p"
    code = main(
        [
            "generate",
            "--seed",
            "Cli7",
            "--width",
            "40",
            "--height",
            "30",
            "--hex-km",
            "120",
            "-o",
            str(out),
            "--png",
            "plates",
            "-q",
        ]
    )
    assert code == 0
    assert (out / "manifest.json").exists() and (out / "plates.png").exists()
    assert main(["info", str(out), "--plates"]) == 0
    assert "world_id" in capsys.readouterr().out
    assert main(["validate", str(out)]) == 0
    assert main(["render", str(out), "--layer", "boundaries", "-o", str(tmp_path / "b.png")]) == 0
    assert (tmp_path / "b.png").read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"
    assert main(["hex", str(out), "3", "4"]) == 0
    assert main(["generate", "--seed", "bad seed", "-q"]) == 1


def test_display_scale_never_changes_data(small_project):
    before = small_project.content_hashes()
    small = render.render_project(small_project, "plates", px_per_hex=3)
    big = render.render_project(small_project, "plates", px_per_hex=9)
    assert big.shape[0] > small.shape[0]
    for layer in render.LAYERS:
        img = render.render_project(small_project, layer, px_per_hex=2)
        assert img.dtype == np.uint8 and img.shape[2] == 3
    assert small_project.content_hashes() == before


def test_render_maps_pixels_to_the_right_hex(small_project):
    layout = small_project.layout
    rgb = render.layer_rgb(small_project, "plates")
    x, y = layout.center(17, 23)
    img = render.render_view(layout, rgb, x - 5.0, y - 5.0, 10.0, 1, 1, outline=False)
    assert tuple(img[0, 0]) == tuple(rgb[23, 17])


def test_gui_smoke(tmp_path, small_project):
    pytest.importorskip("PySide6")
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    try:
        from PySide6.QtWidgets import QApplication

        from hexworld.gui.app import MainWindow
    except ImportError as exc:  # missing system GL libraries on a headless box
        pytest.skip(f"Qt unavailable: {exc}")
    from hexworld import save_project

    app = QApplication.instance() or QApplication([])
    window = MainWindow()
    window.resize(900, 600)
    window.show()
    window._show_project(
        load_project(save_project(small_project, tmp_path / "g")), "test", sync_form=True
    )
    assert window.seed.text() == small_project.config.seed
    before = small_project.content_hashes()
    for index in range(window.layer.count()):
        window.layer.setCurrentIndex(index)
        app.processEvents()
        assert not window.view.grab().isNull()
    window.view._km_per_px *= 0.3  # zoom is display state only
    window.view._invalidate()
    window._on_plate_selected(int(small_project.layer("plate_id")[50, 50]))
    app.processEvents()
    assert "Плита" in window.info.toPlainText()
    assert window.project.content_hashes() == before
    assert window.build_config().seed == small_project.config.seed
    window.close()
