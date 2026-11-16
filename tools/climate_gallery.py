"""Diagnostic images of v4 (climate, rivers, erosion) and the Earth comparison for a few seeds.

    python tools/climate_gallery.py --out docs/diagnostics/v4 --report docs/realism_v4.md

Each seed: a 360x240 world of 75 km hexes (the documentation setting), history 30 Ma.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from hexworld import Config, generate, render, views
from hexworld.realism import earth_comparison, report_markdown

SHOTS = (
    ("temperature", "mean", ("coast", "isotherms")),
    ("temperature", "djf", ("coast", "isotherms")),
    ("temperature", "jja", ("coast", "isotherms")),
    ("precipitation", "precipitation", ("coast", "wind")),
    ("precipitation", "djf", ("coast",)),
    ("precipitation", "jja", ("coast",)),
    ("precipitation", "regime", ("coast",)),
    ("precipitation", "aridity", ("coast",)),
    ("hydrology", "rivers", ("rivers", "lakes", "coast")),
    ("hydrology", "basins", ("rivers", "coast")),
    ("erosion", "delta", ("coast",)),
)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--seeds", nargs="*", default=["Terra7", "Mundo42", "Kq3Lm"])
    parser.add_argument("--out", default="docs/diagnostics/v4")
    parser.add_argument("--report", default="docs/realism_v4.md")
    parser.add_argument("--px-per-hex", type=float, default=3.0)
    args = parser.parse_args(argv)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    sections = [
        "# Сравнение сгенерированных миров с Землёй (v4)",
        "",
        "Создано `tools/climate_gallery.py`. Мир 360×240 гексов по 75 км (режим world),",
        "история 30 млн лет, климат `earthlike`, широта центра +30°. Эталоны Земли — глобальные",
        "средние; окно генерированного мира — регион, поэтому умеренные отличия ожидаемы.",
        "Диапазоны «правдоподобно» заданы в `hexworld/realism.py` и намеренно широкие.",
        "",
    ]
    stats = {}
    for seed in args.seeds:
        cfg = Config.from_dict(
            {
                "seed": seed,
                "width": 360,
                "height": 240,
                "hex_spacing_km": 75.0,
                "mode": "world",
                "history": {"duration_ma": 30.0},
            }
        )
        project = generate(cfg)
        for view, mode, overlays in SHOTS:
            img = views.render_view_image(project, view, mode, overlays, px_per_hex=args.px_per_hex)
            render.write_png(out / f"{seed}_{view}_{mode}.png", img)
        metrics = earth_comparison(project)
        sections += [f"## {seed}", "", report_markdown(project, metrics), ""]
        stats[seed] = {
            "summary": project.hydro.summary,
            "budget_m3": project.hydro.budget,
            "metrics": [m.to_dict() for m in metrics],
            "total_s": project.timings.get("total"),
        }
        print(seed, "ok" if project.hydro.validation["ok"] else "VALIDATION FAILED")
    (out / "climate_gallery_stats.json").write_text(
        json.dumps(stats, ensure_ascii=False, indent=1, default=float), encoding="utf-8"
    )
    Path(args.report).write_text("\n".join(sections), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
