"""Entry point of the installed HexWorld application (Windows, PyInstaller).

``HexWorld.exe``                 opens the program (the «Новый мир» window first);
``HexWorld.exe PROJECT_DIR``     opens a saved world;
``HexWorld.exe --self-test OUT`` generates a small world without a window, exports it and
                                 writes ``OUT`` (JSON); exit code 0 = everything works.
                                 Used by the installer build to check the packaged program.
"""

from __future__ import annotations

import json
import os
import sys
import time
import traceback


def self_test(report_path: str) -> int:
    result: dict = {"ok": False}
    t0 = time.time()
    try:
        from hexworld import Config, generate
        from hexworld.export_game import export_game
        from hexworld.mountains.pmge_binding import find_library
        from hexworld.version import GENERATOR_VERSION, PRODUCT_VERSION

        result["product_version"] = PRODUCT_VERSION
        result["generator_version"] = GENERATOR_VERSION
        result["pmge"] = find_library()
        cfg = Config.from_dict(
            {
                **Config().to_dict(),
                "seed": "SelfTest1",
                "width": 48,
                "height": 32,
                "hex_spacing_km": 75.0,
                "mode": "world",
            }
        )
        project = generate(cfg)
        result["land_share"] = float(project.hydro.layers["land_mask"].mean())
        result["biomes"] = project.biomes is not None
        out = os.path.join(os.path.dirname(os.path.abspath(report_path)), "selftest_export")
        result["export_files"] = len(export_game(project, out, px_per_hex=4))
        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
        from PySide6.QtWidgets import QApplication

        from hexworld.gui.app import MainWindow

        app = QApplication.instance() or QApplication([])
        os.environ["HEXWORLD_NO_START_DIALOG"] = "1"
        window = MainWindow()
        window._show_project(project, "self-test")
        window.show_hex_panel(10, 10)
        app.processEvents()
        result["gui"] = True
        window.close()
        result["ok"] = bool(result["biomes"]) and result["export_files"] >= 8
    except Exception:  # noqa: BLE001 - the report says what failed
        result["error"] = traceback.format_exc()
    result["seconds"] = round(time.time() - t0, 1)
    with open(report_path, "w", encoding="utf-8") as fh:
        json.dump(result, fh, ensure_ascii=False, indent=1)
    return 0 if result["ok"] else 1


def main() -> int:
    args = sys.argv[1:]
    if args[:1] == ["--self-test"]:
        return self_test(args[1] if len(args) > 1 else "hexworld_selftest.json")
    from hexworld.gui.app import run

    return run(args[0] if args else None)


if __name__ == "__main__":
    raise SystemExit(main())
