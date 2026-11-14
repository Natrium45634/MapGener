"""Raster rendering shared by the CLI (PNG export) and the GUI.

The map is drawn as one image by mapping every *pixel* to its hex with
vectorised NumPy - no per-hex objects - so 1000x800 maps stay cheap. Nothing
here changes project data: rendering parameters are display-only.
"""

from __future__ import annotations

import struct
import zlib
from pathlib import Path

import numpy as np

from .geometry import HexLayout
from .project import Project
from .stages.fields import FIELD_NAMES

LAYERS = ("plates", "crust", "boundaries", "motion", "boundary_distance", *FIELD_NAMES)
LAYER_TITLES = {
    "plates": "Плиты",
    "crust": "Кора",
    "boundaries": "Границы",
    "motion": "Движение",
    "boundary_distance": "Близость к границе",
    "uplift": "Потенциал поднятия",
    "rifting": "Потенциал рифтинга",
    "subduction": "Потенциал субдукции",
    "volcanism": "Потенциал вулканизма",
    "seismicity": "Потенциал сейсмичности",
}
BOUNDARY_COLORS = np.array(
    [[0, 0, 0], [214, 48, 49], [9, 132, 227], [0, 168, 107], [150, 150, 150]], dtype=np.uint8
)
BACKGROUND = (24, 26, 30)
_RAMP = np.array(
    [
        [13, 16, 33],
        [46, 30, 94],
        [112, 31, 129],
        [182, 54, 121],
        [241, 96, 93],
        [254, 159, 109],
        [252, 228, 170],
    ],
    dtype=np.float64,
)


def _hex_to_rgb(color: str) -> tuple[int, int, int]:
    return int(color[1:3], 16), int(color[3:5], 16), int(color[5:7], 16)


def plate_color_table(project: Project) -> np.ndarray:
    return np.array([_hex_to_rgb(p["color"]) for p in project.plates], dtype=np.uint8)


def ramp(values: np.ndarray) -> np.ndarray:
    """Map values in [0, 1] to RGB through a perceptual dark-to-light ramp."""
    v = np.clip(values, 0.0, 1.0) * (len(_RAMP) - 1)
    lo = np.minimum(v.astype(np.int64), len(_RAMP) - 2)
    t = (v - lo)[..., None]
    return (_RAMP[lo] * (1 - t) + _RAMP[lo + 1] * t).astype(np.uint8)


def layer_rgb(project: Project, layer: str) -> np.ndarray:
    """Colour of every hex for ``layer`` as ``(height, width, 3) uint8``."""
    plate_id = project.layer("plate_id")
    plate_rgb = plate_color_table(project)[plate_id]
    crust = project.layer("crust_type")
    if layer == "plates":
        return plate_rgb
    if layer == "crust":
        base = np.where(crust[..., None] == 1, np.array([196, 164, 110]), np.array([52, 92, 150]))
        # faint plate tint so that mixed-crust plates stay readable
        return (0.82 * base + 0.18 * plate_rgb).astype(np.uint8)
    if layer == "boundaries":
        gray = plate_rgb.astype(np.float64).mean(axis=-1, keepdims=True)
        base = 0.35 * plate_rgb + 0.25 * gray + np.where(crust[..., None] == 1, 70.0, 40.0)
        kind = project.layer("boundary_type")
        out = np.clip(base, 0, 255).astype(np.uint8)
        mask = kind > 0
        out[mask] = BOUNDARY_COLORS[kind[mask]]
        return out
    if layer == "motion":
        speed = np.hypot(project.layer("velocity_x"), project.layer("velocity_y"))
        return ramp(0.12 + 0.88 * speed / max(1e-6, float(speed.max())))
    if layer == "boundary_distance":
        d = project.layer("boundary_distance_km")
        return ramp(1.0 - d / max(1e-6, float(project.config.fields.max_influence_km)))
    if layer in FIELD_NAMES:
        return ramp(project.layer(layer))
    raise KeyError(f"Unknown render layer {layer!r}; available: {LAYERS}")


def render_view(
    layout: HexLayout,
    hex_rgb: np.ndarray,
    x0: float,
    y0: float,
    km_per_px: float,
    out_w: int,
    out_h: int,
    outline: bool | None = None,
) -> np.ndarray:
    """Render the world rectangle starting at ``(x0, y0)`` to an RGB image."""
    xs = x0 + (np.arange(out_w, dtype=np.float64) + 0.5) * km_per_px
    ys = y0 + (np.arange(out_h, dtype=np.float64) + 0.5) * km_per_px
    col, row, edge = layout.world_to_offset(xs[None, :], ys[:, None], with_edge=True)
    inside = layout.inside(col, row)
    img = np.empty((out_h, out_w, 3), dtype=np.uint8)
    img[:] = BACKGROUND
    img[inside] = hex_rgb[row[inside], col[inside]]
    px_per_hex = layout.spacing_km / km_per_px
    if outline is None:
        outline = px_per_hex >= 9.0
    if outline:
        line = inside & (edge > 1.0 - min(0.3, 1.6 / px_per_hex))
        img[line] = (img[line] * 0.72).astype(np.uint8)
    return img


def _draw_line(img, x1, y1, x2, y2, color, thickness=1):
    n = int(max(abs(x2 - x1), abs(y2 - y1))) + 1
    xs = np.rint(np.linspace(x1, x2, n)).astype(np.int64)
    ys = np.rint(np.linspace(y1, y2, n)).astype(np.int64)
    h, w = img.shape[:2]
    for dx in range(thickness):
        for dy in range(thickness):
            ok = (xs + dx >= 0) & (xs + dx < w) & (ys + dy >= 0) & (ys + dy < h)
            img[ys[ok] + dy, xs[ok] + dx] = color


def motion_arrows(project: Project, step_km: float):
    """Arrow anchors on a regular world grid: ``(x, y, vx, vy)`` arrays (mm/yr)."""
    b = project.geometry.viewport_bounds
    xs = np.arange(b.xmin + 0.5 * step_km, b.xmax, step_km)
    ys = np.arange(b.ymin + 0.5 * step_km, b.ymax, step_km)
    gx, gy = np.meshgrid(xs, ys)
    col, row = project.layout.world_to_offset(gx, gy)
    ok = project.layout.inside(col, row)
    gx, gy, col, row = gx[ok], gy[ok], col[ok], row[ok]
    return gx, gy, project.layer("velocity_x")[row, col], project.layer("velocity_y")[row, col]


def render_project(
    project: Project,
    layer: str = "plates",
    px_per_hex: float | None = None,
    max_size: int = 2400,
    arrows: bool | None = None,
) -> np.ndarray:
    """Render the whole viewport. ``px_per_hex`` is display-only."""
    b = project.geometry.viewport_bounds
    spacing = project.layout.spacing_km
    if px_per_hex is None:
        px_per_hex = min(12.0, max_size / max(b.width / spacing, b.height / spacing))
    km_per_px = spacing / px_per_hex
    out_w = max(1, int(round(b.width / km_per_px)))
    out_h = max(1, int(round(b.height / km_per_px)))
    img = render_view(
        project.layout, layer_rgb(project, layer), b.xmin, b.ymin, km_per_px, out_w, out_h
    )
    if arrows is None:
        arrows = layer == "motion"
    if arrows:
        draw_motion_arrows(img, project, km_per_px)
    return img


def draw_motion_arrows(img: np.ndarray, project: Project, km_per_px: float) -> None:
    """Draw plate-motion arrows on a rendered image of the whole window (in place)."""
    b = project.geometry.viewport_bounds
    step_px = 46.0
    gx, gy, vx, vy = motion_arrows(project, step_px * km_per_px)
    vmax = max(1e-6, float(np.hypot(vx, vy).max())) if gx.size else 1.0
    for x, y, u, v in zip(gx, gy, vx, vy, strict=True):
        px = (x - b.xmin) / km_per_px
        py = (y - b.ymin) / km_per_px
        length = 0.8 * step_px * float(np.hypot(u, v)) / vmax
        if length < 2.0:
            continue
        ang = float(np.arctan2(v, u))
        ex, ey = px + length * np.cos(ang), py + length * np.sin(ang)
        _draw_line(img, px, py, ex, ey, (255, 255, 255), 2)
        for da in (2.6, -2.6):
            _draw_line(
                img,
                ex,
                ey,
                ex + 7 * np.cos(ang + da),
                ey + 7 * np.sin(ang + da),
                (255, 255, 255),
                2,
            )


def write_png(path: str | Path, rgb: np.ndarray) -> Path:
    """Minimal dependency-free PNG writer (8-bit RGB)."""
    path = Path(path)
    h, w = rgb.shape[:2]
    raw = np.empty((h, 1 + 3 * w), dtype=np.uint8)
    raw[:, 0] = 0
    raw[:, 1:] = np.ascontiguousarray(rgb[..., :3], dtype=np.uint8).reshape(h, 3 * w)

    def chunk(tag: bytes, data: bytes) -> bytes:
        return (
            struct.pack(">I", len(data))
            + tag
            + data
            + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)
        )

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0))
        + chunk(b"IDAT", zlib.compress(raw.tobytes(), 6))
        + chunk(b"IEND", b"")
    )
    return path
