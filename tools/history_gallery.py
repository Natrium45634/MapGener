"""Diagnostic images of the v3 geological history (TZ 3.0, section 8).

For every seed: the same window at the initial and the final epoch, the change
map, the volcanic centres, and three close-ups chosen from the data itself:

* the youngest collision mountain system  ("young collisional mountains"),
* the oldest, most worn system            ("ancient eroded system"),
* the strongest rift next to an arc or a hotspot chain.

    python tools/history_gallery.py --out docs/diagnostics/v3
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from hexworld import Config, generate
from hexworld.config import HistoryParams
from hexworld.render import write_png
from hexworld.views import render_view_image

SEEDS = ["Ab12", "Terra7", "Kq3Lm"]
PX = 6.0


def crop(img, project, x_km, y_km, half_km=1500.0):
    b = project.geometry.viewport_bounds
    k = project.layout.spacing_km / PX
    cx, cy = (x_km - b.xmin) / k, (y_km - b.ymin) / k
    r = half_km / k
    x0, x1 = int(max(0, cx - r)), int(min(img.shape[1], cx + r))
    y0, y1 = int(max(0, cy - r)), int(min(img.shape[0], cy + r))
    part = img[y0:y1, x0:x1]
    return np.repeat(np.repeat(part, 3, axis=0), 3, axis=1)  # enlarge for viewing


def row(images, gap=8):
    h = max(i.shape[0] for i in images)
    w = sum(i.shape[1] for i in images) + gap * (len(images) + 1)
    out = np.full((h + 2 * gap, w, 3), 255, dtype=np.uint8)
    x = gap
    for img in images:
        out[gap : gap + img.shape[0], x : x + img.shape[1]] = img
        x += img.shape[1] + gap
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="docs/diagnostics/v3")
    parser.add_argument("--seeds", nargs="*", default=SEEDS)
    parser.add_argument("--duration", type=float, default=40.0)
    args = parser.parse_args()
    out = Path(args.out)
    stats = []
    for seed in args.seeds:
        config = Config(
            seed=seed,
            width=300,
            height=220,
            hex_spacing_km=90.0,
            mode="world",
            history=HistoryParams(duration_ma=args.duration),
        )
        project = generate(config)
        hist = project.hist
        views = {
            "initial": render_view_image(project, "heights", "baseline", px_per_hex=PX),
            "final": render_view_image(project, "relief", "shaded", {"coast"}, px_per_hex=PX),
            "delta": render_view_image(project, "history", "delta", px_per_hex=PX),
            "volcanism": render_view_image(project, "volcanism", "centres", px_per_hex=PX),
            "structure": render_view_image(
                project,
                "relief",
                "shaded",
                {"coast", "ridges", "peaks", "passes", "valleys"},
                px_per_hex=PX,
            ),
            "classes": render_view_image(project, "relief", "classes", {"coast"}, px_per_hex=PX),
            "age": render_view_image(project, "relief", "orogen_age", {"coast"}, px_per_hex=PX),
        }
        small = [
            v[::2, ::2]
            for v in (views["initial"], views["final"], views["delta"], views["volcanism"])
        ]
        write_png(out / f"{seed}_initial_final_delta_volcanism.png", row(small))

        b = project.geometry.viewport_bounds
        inside = [
            s
            for s in hist.plan["systems"]
            if b.contains(*s["centroid_km"]) and s["visible_hexes"] > 30
        ]
        morph: dict = {}
        for s_ in hist.plan["systems"]:
            morph[s_["morphology"]] = morph.get(s_["morphology"], 0) + 1
        record = {
            "seed": seed,
            "duration_ma": hist.duration_ma,
            "morphologies": morph,
            "summary": {
                k: hist.summary[k]
                for k in (
                    "mountain_systems",
                    "ridges",
                    "peaks",
                    "passes",
                    "volcanoes",
                    "active_volcanoes",
                    "hotspots",
                    "events",
                    "max_elevation_m",
                    "changed_hex_share",
                    "new_crust_share_viewport",
                )
            },
        }
        young = [s for s in inside if s["kind"] == "collision"]
        if young:
            s = min(young, key=lambda s: (s["maturity"], -s["area_km2"]))
            x, y = s["centroid_km"]
            write_png(
                out / f"{seed}_young_collision_mountains.png",
                row([crop(views[k], project, x, y) for k in ("structure", "classes", "age")]),
            )
            record["young"] = {
                k: s[k]
                for k in (
                    "id",
                    "kind",
                    "morphology",
                    "age_ma",
                    "maturity",
                    "max_elevation_m",
                    "length_km",
                )
            }
        if inside:
            s = max(inside, key=lambda s: (s["maturity"], s["area_km2"]))
            x, y = s["centroid_km"]
            write_png(
                out / f"{seed}_ancient_eroded_system.png",
                row([crop(views[k], project, x, y) for k in ("structure", "classes", "age")]),
            )
            record["ancient"] = {
                k: s[k]
                for k in (
                    "id",
                    "kind",
                    "morphology",
                    "age_ma",
                    "maturity",
                    "max_elevation_m",
                    "length_km",
                )
            }
        rifts = [
            e
            for e in hist.events
            if e["type"] in ("ocean_opening", "rifting_start") and b.contains(e["x_km"], e["y_km"])
        ]
        centres = (
            np.array([[v["x_km"], v["y_km"]] for v in hist.volcanoes])
            if hist.volcanoes
            else np.zeros((0, 2))
        )
        if rifts and len(centres):

            def nearby(e, centres=centres):
                d = np.hypot(centres[:, 0] - e["x_km"], centres[:, 1] - e["y_km"])
                return int((d < 1500.0).sum())

            e = max(rifts, key=lambda e: (nearby(e), e["cells"]))
            write_png(
                out / f"{seed}_rift_with_arc_or_hotspot.png",
                row(
                    [
                        crop(views[k], project, e["x_km"], e["y_km"])
                        for k in ("final", "delta", "volcanism")
                    ]
                ),
            )
            record["rift"] = {
                "event": e["id"],
                "time_ma": e["time_ma"],
                "volcanic_centres_within_1500_km": nearby(e),
            }
        stats.append(record)
        print(json.dumps(record, ensure_ascii=False))
    out.mkdir(parents=True, exist_ok=True)
    (out / "history_gallery_stats.json").write_text(
        json.dumps(stats, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
