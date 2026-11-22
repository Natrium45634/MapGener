"""Export a world for game engines and other programs.

Writes into one folder:

* ``heightmap.png`` — 16-bit greyscale height map (Unreal, Godot, Unity via PNG plugins),
  ``heightmap.r16`` — the same as RAW 16-bit little-endian (Unity «Import Raw»);
  0 = ``height_min_m``, 65535 = ``height_max_m`` (in ``world.json``);
* ``biomes.png`` (colours), ``biome_ids.png`` (16-bit IDs, legend in ``world.json``),
  ``water.png`` (sea / lake / land mask), ``rivers.png`` (river mask, brighter = bigger),
  ``temperature.png`` and ``precipitation.png`` (8-bit, ranges in ``world.json``);
* ``hexes.csv`` and ``hexes.json`` — one record per hex for hex-based games
  (offset and axial coordinates, height, water, biome, climate, river, landmass);
* ``world.json`` — scale, ranges, legends, place names with their pixel positions;
* ``ИМПОРТ.txt`` — how to bring it into Unity, Unreal Engine, Godot.

The raster maps are sampled at ``px_per_hex`` pixels per hex; heights are interpolated
smoothly between hex centres (terrain engines dislike hex steps), the class maps keep the
hex they fall in.
"""

from __future__ import annotations

import csv
import json
import struct
import zlib
from pathlib import Path

import numpy as np
from scipy import ndimage

from . import render


def write_png16(path: str | Path, gray: np.ndarray) -> Path:
    """16-bit greyscale PNG without dependencies."""
    path = Path(path)
    h, w = gray.shape
    data = np.ascontiguousarray(gray, dtype=">u2")
    raw = np.empty((h, 1 + 2 * w), dtype=np.uint8)
    raw[:, 0] = 0
    raw[:, 1:] = data.view(np.uint8).reshape(h, 2 * w)

    def chunk(tag: bytes, payload: bytes) -> bytes:
        return (
            struct.pack(">I", len(payload))
            + tag
            + payload
            + struct.pack(">I", zlib.crc32(tag + payload) & 0xFFFFFFFF)
        )

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 16, 0, 0, 0, 0))
        + chunk(b"IDAT", zlib.compress(raw.tobytes(), 6))
        + chunk(b"IEND", b"")
    )
    return path


def _pixel_hexes(project, px_per_hex: float):
    """Hex (col, row) of every output pixel, the pixel grid and its scale."""
    lay = project.layout
    b = project.geometry.viewport_bounds
    k = lay.spacing_km / float(px_per_hex)
    w, h = max(2, int(b.width / k)), max(2, int(b.height / k))
    xs = b.xmin + (np.arange(w) + 0.5) * k
    ys = b.ymin + (np.arange(h) + 0.5) * k
    col, row = lay.world_to_offset(xs[None, :], ys[:, None])
    col = np.clip(col, 0, lay.width - 1)
    row = np.clip(row, 0, lay.height - 1)
    return col, row, k, (w, h), b


def _elevation(project) -> tuple[np.ndarray, float]:
    if project.hydro is not None:
        return project.hydro.layers["elevation_m"], project.hydro.sea_level_m
    if project.hist is not None:
        return project.hist.layers["elevation_m"], project.hist.sea_level_m
    return project.geo.layers["macro_elevation_m"], project.geo.sea_level_m


def export_game(project, out_dir: str | Path, px_per_hex: float = 8.0, through: str = "v50"):
    """Write the export folder; returns the list of files written."""
    from .names import place_labels, ru
    from .world_tools import world_code

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    lay = project.layout
    col, row, k, (w, h), b = _pixel_hexes(project, px_per_hex)
    z, sea = _elevation(project)
    rel = np.asarray(z, dtype=np.float64) - sea

    # -- heights: hex values, smoothed over about half a hex ------------------------------------
    hm = rel[row, col]
    hm = ndimage.gaussian_filter(hm, sigma=max(0.5, 0.45 * px_per_hex))
    lo, hi = float(hm.min()), float(hm.max())
    scale = 65535.0 / max(hi - lo, 1.0)
    h16 = np.clip(np.rint((hm - lo) * scale), 0, 65535).astype(np.uint16)
    written.append(write_png16(out / "heightmap.png", h16))
    (out / "heightmap.r16").write_bytes(h16.astype("<u2").tobytes())
    written.append(out / "heightmap.r16")

    # -- class maps ------------------------------------------------------------------------------
    L = project.hydro.layers if project.hydro is not None else {}
    ocean = L.get("ocean_mask")
    if ocean is None:
        ocean = rel <= 0
    lake = L.get("lake_id")
    water = np.where(ocean, 0, 255).astype(np.uint8)
    if lake is not None:
        water = np.where(lake >= 0, 128, water)
    wpx = water[row, col]
    written.append(render.write_png(out / "water.png", np.repeat(wpx[..., None], 3, axis=2)))
    if "discharge_m3_s" in L:
        river = L["river_mask"].astype(bool)
        q = np.log10(np.maximum(L["discharge_m3_s"], 1.0))
        val = np.where(
            river, 60 + 195 * q / max(float(q[river].max()) if river.any() else 1, 1e-6), 0
        )
        rpx = np.clip(val, 0, 255).astype(np.uint8)[row, col]
        written.append(render.write_png(out / "rivers.png", np.repeat(rpx[..., None], 3, axis=2)))
    ranges = {}
    for name, key, lo_v, hi_v in (
        ("temperature", "temperature_mean_c", -40.0, 40.0),
        ("precipitation", "precipitation_mm_yr", 0.0, 4000.0),
    ):
        if key in L:
            v = np.clip((L[key] - lo_v) / (hi_v - lo_v), 0, 1)
            g = np.rint(255 * v).astype(np.uint8)[row, col]
            written.append(
                render.write_png(out / f"{name}.png", np.repeat(g[..., None], 3, axis=2))
            )
            ranges[name] = {
                "value_at_0": lo_v,
                "value_at_255": hi_v,
                "unit": key.rsplit("_", 1)[-1],
            }

    bio = getattr(project, "biomes", None)
    legend = []
    if bio is not None:
        from .biomes.registry import BIOMES

        dom = bio.layers["dominant_biome"].astype(np.uint16)
        colors = np.array([b_.rgb for b_ in BIOMES], dtype=np.uint8)
        written.append(render.write_png(out / "biomes.png", colors[dom[row, col]]))
        written.append(write_png16(out / "biome_ids.png", dom[row, col]))
        used = np.unique(dom)
        legend = [
            {"id": int(i), "key": BIOMES[int(i)].id, "ru": BIOMES[int(i)].ru,
             "en": BIOMES[int(i)].en, "rgb": list(BIOMES[int(i)].rgb)}
            for i in used
        ]  # fmt: skip

    # -- hex table -------------------------------------------------------------------------------
    cx, cy = lay.centers()
    cols = np.broadcast_to(np.arange(lay.width)[None, :], (lay.height, lay.width))
    rows = np.broadcast_to(np.arange(lay.height)[:, None], (lay.height, lay.width))
    q = cols
    r = rows - (cols - (cols & 1)) // 2  # odd-q offset -> axial
    fields = {
        "col": cols, "row": rows, "q": q, "r": r,
        "x_km": np.round(cx.astype(np.float64), 1), "y_km": np.round(cy.astype(np.float64), 1),
        "height_m": np.round(rel.astype(np.float64), 1),
        "water": np.where(ocean, "sea", np.where(lake >= 0, "lake", "land"))
        if lake is not None else np.where(ocean, "sea", "land"),
    }  # fmt: skip
    for name, key, digits in (("temperature_c", "temperature_mean_c", 1),
                              ("precipitation_mm", "precipitation_mm_yr", 0),
                              ("river_order", "strahler_order", None),
                              ("landmass", "landmass_id", None)):  # fmt: skip
        if key in L:
            fields[name] = L[key] if digits is None else np.round(L[key].astype(np.float64), digits)
    if bio is not None:
        from .biomes.registry import BIOMES

        ids = bio.layers["dominant_biome"]
        fields["biome_id"] = ids
        fields["biome"] = np.array([BIOMES[i].id for i in range(len(BIOMES))])[ids]
    keys = list(fields)
    flat = {k_: np.asarray(v).ravel() for k_, v in fields.items()}
    n = lay.width * lay.height
    with (out / "hexes.csv").open("w", newline="", encoding="utf-8") as fh:
        wri = csv.writer(fh)
        wri.writerow(keys)
        for i in range(n):
            wri.writerow([_plain(flat[k_][i]) for k_ in keys])
    written.append(out / "hexes.csv")
    (out / "hexes.json").write_text(
        json.dumps({"columns": keys, "width": lay.width, "height": lay.height,
                    "data": {k_: [_plain(v) for v in flat[k_]] for k_ in keys}},
                   ensure_ascii=False, separators=(",", ":")),
        encoding="utf-8",
    )  # fmt: skip
    written.append(out / "hexes.json")

    # -- metadata --------------------------------------------------------------------------------
    labels = []
    for lab in place_labels(project):
        labels.append({
            "kind": lab.kind, "name": lab.name, "name_ru": ru(lab.name), "title_ru": lab.title_ru,
            "x_km": round(lab.x_km, 1), "y_km": round(lab.y_km, 1),
            "x_px": round((lab.x_km - b.xmin) / k, 1), "y_px": round((lab.y_km - b.ymin) / k, 1),
        })  # fmt: skip
    meta = {
        "format": "hexworld-game-export-1",
        "seed": project.config.seed,
        "world_code": world_code(project.config, through),
        "image_size_px": [w, h],
        "km_per_pixel": k,
        "meters_per_pixel": k * 1000.0,
        "px_per_hex": px_per_hex,
        "hex_grid": {"width": lay.width, "height": lay.height, "spacing_km": lay.spacing_km,
                     "layout": "flat-top, odd-q offset (col, row); axial (q, r)"},
        "height_min_m": lo,
        "height_max_m": hi,
        "height_range_m": hi - lo,
        "sea_level_in_heightmap": float((0.0 - lo) * scale),
        "water_png": {"0": "sea", "128": "lake", "255": "land"},
        "ranges": ranges,
        "biomes": legend,
        "names": labels,
        "north": "up (-y)",
    }  # fmt: skip
    (out / "world.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1), "utf-8")
    written.append(out / "world.json")
    (out / "ИМПОРТ.txt").write_text(
        _import_text(w=w, h=h, lo=lo, hi=hi, rng=hi - lo, mpp=k * 1000.0), encoding="utf-8"
    )
    written.append(out / "ИМПОРТ.txt")
    return written


def _plain(v):
    if isinstance(v, np.generic):
        return v.item()
    return v


_IMPORT = """Экспорт мира HexWorld для игровых движков
==========================================

Размер карт: {w} × {h} пикселей, {mpp:.0f} м на пиксель.
Высоты: 0 = {lo:.0f} м, 65535 = {hi:.0f} м (перепад {rng:.0f} м); уровень моря — в world.json.

Unity (Terrain)
  1. Создайте Terrain, вкладка Terrain Settings → Import Raw…
  2. Файл heightmap.r16, Depth 16 bit, Byte Order Windows, ширина {w}, высота {h}.
     Unity просит квадрат 2^n+1 — при необходимости увеличьте холст в редакторе картинок
     или укажите Resolution, ближайшее к размеру.
  3. Terrain Height = {rng:.0f} (м), Terrain Width / Length = {w} × {mpp:.0f} и {h} × {mpp:.0f} м.
  4. biomes.png можно использовать как карту раскраски (splat) или подсказку для текстур.

Unreal Engine (Landscape)
  1. Landscape Mode → Import from File → heightmap.png (16-bit PNG).
  2. Масштаб Z = {rng:.0f} м × 100 / 512 ≈ {z} (Unreal: 512 = 256 см по умолчанию).
  3. Масштаб X/Y = {mpp:.0f} м × 100 = {xy} см на пиксель.

Godot 4
  1. Image.load("heightmap.png") → HeightMapShape3D или плагин Terrain3D (Import → Height).
  2. Высота = пиксель / 65535 × {rng:.0f} + {lo:.0f} м.

Гексагональные игры
  hexes.csv / hexes.json — по записи на гекс: столбец/строка (odd-q), осевые q/r, высота над
  морем, вода, биом, температура, осадки, порядок реки, материк. Подходит для Tiled, Godot
  TileMap (hex), стратегий на гексах.

Названия мест (океаны, материки, горы, реки, озёра, пустыни, ледники) — в world.json → names,
с координатами в км и в пикселях карт.
"""


def _import_text(**kw) -> str:
    kw["z"] = f"{kw['rng'] * 100.0 / 512.0:.1f}"
    kw["xy"] = f"{kw['mpp'] * 100.0:.0f}"
    return _IMPORT.format(**kw)
