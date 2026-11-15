"""Diagnostic contact sheets for acceptance check 6 (shape quality).

Generates worlds for several seeds and the three continent profiles and writes
one PNG sheet per profile and view, plus a JSON table of measured statistics.

    python tools/quality_gallery.py --out docs/diagnostics
"""

from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path

import numpy as np

from hexworld import Config, generate
from hexworld.config import CONTINENT_PROFILES, ContinentParams
from hexworld.render import write_png
from hexworld.views import render_view_image

SEEDS = ["Ab12", "ab12", "Zx9", "Terra7", "Kq3Lm", "N0va", "Pangea7", "Q", "x7Y2z", "Mundo42"]


def sheet(images: list[np.ndarray], columns: int, gap: int = 6) -> np.ndarray:
    h = max(i.shape[0] for i in images)
    w = max(i.shape[1] for i in images)
    rows = -(-len(images) // columns)
    out = np.full((rows * (h + gap) + gap, columns * (w + gap) + gap, 3), 255, dtype=np.uint8)
    for k, img in enumerate(images):
        r, c = divmod(k, columns)
        y, x = gap + r * (h + gap), gap + c * (w + gap)
        out[y : y + img.shape[0], x : x + img.shape[1]] = img
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="docs/diagnostics")
    parser.add_argument("--width", type=int, default=200)
    parser.add_argument("--height", type=int, default=150)
    parser.add_argument("--hex-km", type=float, default=140.0)
    parser.add_argument("--seeds", nargs="*", default=SEEDS)
    parser.add_argument("--profiles", nargs="*", default=list(CONTINENT_PROFILES))
    args = parser.parse_args()
    out = Path(args.out)
    table = []
    for profile in args.profiles:
        tiles: dict[str, list[np.ndarray]] = {"landmasses": [], "blocks": [], "heights": []}
        for seed in args.seeds:
            config = Config(
                seed=seed,
                width=args.width,
                height=args.height,
                hex_spacing_km=args.hex_km,
                mode="world",
            )
            config = replace(config, continents=ContinentParams(continent_profile=profile))
            project = generate(config, through="v2")
            geo = project.geo
            tiles["landmasses"].append(
                render_view_image(project, "continents", "landmasses", px_per_hex=2)
            )
            tiles["blocks"].append(
                render_view_image(
                    project, "continents", "blocks", {"coast", "plate_boundaries"}, px_per_hex=2
                )
            )
            tiles["heights"].append(
                render_view_image(project, "heights", "hypsometric", px_per_hex=2)
            )
            world = geo.summary["world"]
            table.append(
                {
                    "profile": profile,
                    "seed": seed,
                    "sea_level_m": round(geo.sea_level_m, 1),
                    "land_fraction": round(world["land_fraction_simulation"], 4),
                    "major_landmasses": world["major_landmass_count"],
                    "largest_share_of_land": round(world["largest_landmass_share_of_land"], 3),
                    "landmasses_in_window": len(geo.landmasses),
                    "speck_land_share": geo.validation["checks"].get("speck_land_share"),
                    "valid": bool(geo.validation["ok"] and project.validation["ok"]),
                    "operations": [
                        o["op"]
                        for o in geo.summary["structure"]["operations"]
                        if o["op"] != "summary"
                    ],
                    "warnings": geo.warnings,
                }
            )
            print(table[-1])
        for name, images in tiles.items():
            write_png(out / f"{profile}_{name}.png", sheet(images, 5))
    out.mkdir(parents=True, exist_ok=True)
    (out / "gallery_stats.json").write_text(
        json.dumps(table, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
