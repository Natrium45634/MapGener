"""Shaded renders of PMGE sample terrains, optionally with the plan drawn on top.

    python tools/pmge_showcase.py OUT.png NAME[:extent_m[:overlay]] ...

NAME is the path stem of a pmge_cli output (NAME.pgm + NAME.json).
"""

import json
import sys

import numpy as np

from hexworld.render import _draw_line, write_png
from hexworld.views import hypsometric


def load(stem):
    raw = open(stem + ".pgm", "rb").read()
    head = raw.split(b"\n", 3)
    w, h = map(int, head[1].split())
    return np.frombuffer(head[3], dtype=">u2").reshape(h, w).astype(float) / 65535 * 2500.0


def shaded(height, cell):
    gy, gx = np.gradient(height, cell)
    light = (-gx * 0.6 - gy * 0.6 + 1.0) / np.sqrt(gx * gx + gy * gy + 1.0) / 1.6
    shade = np.clip(light, 0.0, 1.0)
    return (hypsometric(height) * (0.35 + 0.85 * shade[..., None])).clip(0, 255).astype(np.uint8)


def overlay(img, stem, cell):
    plan = json.load(open(stem + ".json"))
    colors = {"main": (40, 20, 10), "branch": (70, 40, 20), "spur": (110, 80, 50)}
    for r in plan["ridges"]:
        pts = np.array(r["points_m"]) / cell - 0.5
        for a, b in zip(pts[:-1], pts[1:], strict=True):
            _draw_line(img, a[0], a[1], b[0], b[1], colors[r["rank"]], 2 if r["rank"] == "main" else 1)
    for v in plan["valleys"]:
        pts = np.array(v["points_m"]) / cell - 0.5
        for a, b in zip(pts[:-1], pts[1:], strict=True):
            _draw_line(img, a[0], a[1], b[0], b[1], (60, 140, 230), 1)
    for group, color, r in (("passes", (250, 200, 40), 4), ("peaks", (255, 255, 255), 5)):
        for p in plan[group]:
            cx, cy = int(p["x_m"] / cell), int(p["y_m"] / cell)
            yy, xx = np.ogrid[: img.shape[0], : img.shape[1]]
            d2 = (yy - cy) ** 2 + (xx - cx) ** 2
            img[d2 <= r * r] = (20, 20, 20)
            img[d2 <= (r - 1.5) ** 2] = color


def main():
    out = sys.argv[1]
    tiles = []
    for spec in sys.argv[2:]:
        parts = spec.split(":")
        stem, extent = parts[0], float(parts[1]) if len(parts) > 1 else 60000.0
        h = load(stem)
        cell = extent / h.shape[0]
        img = shaded(h, cell)
        if len(parts) > 2:
            overlay(img, stem, cell)
        tiles.append(img)
    cols = 3
    gap = 8
    th, tw = tiles[0].shape[:2]
    rows = -(-len(tiles) // cols)
    sheet = np.full((rows * (th + gap) + gap, cols * (tw + gap) + gap, 3), 255, np.uint8)
    for k, t in enumerate(tiles):
        r, c = divmod(k, cols)
        sheet[gap + r * (th + gap) : gap + r * (th + gap) + th,
              gap + c * (tw + gap) : gap + c * (tw + gap) + tw] = t
    write_png(out, sheet)


if __name__ == "__main__":
    main()
