"""PMGE gallery: every mountain architecture, and one range growing old.

    python tools/pmge_styles.py docs/pmge_showcase

Writes ``pmge_styles.png`` (nine styles, same footprint and seed) and ``pmge_ageing.png``
(one alpine range at 0, 20, 80 and 300 Ma), each tile with its plan drawn on top.
Needs the PMGE library (cmake -S pmge -B pmge/build && cmake --build pmge/build).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

from hexworld.mountains.pmge_binding import (
    STYLE_NAMES,
    Library,
    StandalonePlan,
    find_library,
    new_request,
)
from hexworld.render import _draw_line, write_png
from hexworld.views import hypsometric

W, H = 160_000.0, 80_000.0  # footprint, m
NX, NY = 400, 200
CELL = W / NX
TITLES = {
    "volcanic_arc": "volcanic arc",
    "alpine": "alpine",
    "himalayan": "himalayan + plateau",
    "andean": "andean + altiplano",
    "fold_belt": "fold belt",
    "old_massif": "old massif",
    "broad_upland": "broad upland",
    "mixed": "mixed: broad base, sharp core",
    "block": "tilted blocks",
}


def tile(lib, style: str, age: float = 0.0, seed: int = 11):
    req = new_request(
        lib,
        seed=seed,
        footprint_type=1,
        box_min_x=0.0,
        box_min_y=0.0,
        box_max_x=W,
        box_max_y=H,
        base_m=200.0,
        min_m=0.0,
        max_m=4800.0,
        style=style,
        age_ma=age,
        ridge_half_width_m=3500.0,
        spur_spacing_m=6500.0,
        spur_length_m=9000.0,
        peak_spacing_m=8000.0,
    )
    plan = StandalonePlan(lib, req)
    height = plan.heightmap(0.5 * CELL, 0.5 * CELL, CELL, NX, NY).astype(np.float64)
    gy, gx = np.gradient(height, CELL)
    light = (-gx * 0.7 - gy * 0.7 + 1.0) / np.sqrt(gx * gx + gy * gy + 1.0) / 1.6
    img = (hypsometric(height) * (0.35 + 0.85 * np.clip(light, 0, 1)[..., None])).clip(0, 255)
    img = img.astype(np.uint8)
    doc = plan.json()
    colors = {"main": (35, 20, 10), "branch": (70, 40, 20), "spur": (120, 85, 55)}
    for r in doc["ridges"]:
        if r["role"] in ("plateau", "base"):
            continue
        pts = np.array(r["points_m"]) / CELL - 0.5
        for a, b in zip(pts[:-1], pts[1:], strict=True):
            _draw_line(
                img, a[0], a[1], b[0], b[1], colors[r["rank"]], 2 if r["rank"] == "main" else 1
            )
    return img, doc, float(height.max())


def grid(tiles, cols):
    rows = (len(tiles) + cols - 1) // cols
    out = np.full((rows * (NY + 6) + 6, cols * (NX + 6) + 6, 3), 245, dtype=np.uint8)
    for k, img in enumerate(tiles):
        r, c = divmod(k, cols)
        out[6 + r * (NY + 6) : 6 + r * (NY + 6) + NY, 6 + c * (NX + 6) : 6 + c * (NX + 6) + NX] = (
            img
        )
    return out


def main(argv=None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    out = Path(argv[0] if argv else "docs/pmge_showcase")
    out.mkdir(parents=True, exist_ok=True)
    path = find_library()
    if not path:
        print("PMGE library not found: cmake -S pmge -B pmge/build", file=sys.stderr)
        return 1
    lib = Library(path)
    tiles = []
    for style in STYLE_NAMES[1:]:
        img, doc, top = tile(lib, style)
        ranges = sum(1 for r in doc["ridges"] if r["rank"] == "main")
        spurs = sum(1 for r in doc["ridges"] if r["rank"] == "spur")
        print(
            f"{style:13s} {TITLES[style]:32s} ranges {ranges:2d} spurs {spurs:3d} "
            f"peaks {len(doc['peaks']):3d} max {top:6.0f} m"
        )
        tiles.append(img)
    write_png(out / "pmge_styles.png", grid(tiles, 3))
    ages = []
    for age in (0.0, 20.0, 80.0, 300.0):
        img, doc, top = tile(lib, "alpine", age)
        print(
            f"alpine at {age:5.0f} Ma: max {top:6.0f} m, maturity {doc['maturity_eff']:.2f}, "
            f"spurs {sum(1 for r in doc['ridges'] if r['rank'] == 'spur')}"
        )
        ages.append(img)
    write_png(out / "pmge_ageing.png", grid(ages, 2))
    print(out / "pmge_styles.png", out / "pmge_ageing.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
