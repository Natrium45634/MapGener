"""Registry of map views (tabs) shared by the GUI and the CLI.

A view is a *presentation* of data that already exists: switching views never
runs a generator and never changes the project. Each generation stage adds its
view(s) here with ``register_view``; the map widget itself knows nothing about
individual views, so future tabs (rivers, climate, biomes, geo-symbols) need no
changes to it. Views whose data stage does not exist yet are registered with
``status="planned"`` and are shown disabled - never drawn as if they were ready.
"""

from __future__ import annotations

import colorsys
from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np

from . import hexgrid, render
from .project import Project
from .stages.fields import FIELD_NAMES
from .stages.raster import neighbor_index


@dataclass(frozen=True)
class Option:
    id: str
    title: str
    default: bool = False


@dataclass(frozen=True)
class LegendItem:
    label: str
    color: tuple[int, int, int] | None = None


@dataclass(frozen=True)
class ViewSpec:
    id: str
    title: str
    #: "ready" - implemented; "planned" - reserved for a future stage, shown disabled.
    status: str = "ready"
    #: Which stage provides the data: "v1", "v2" or None for planned views.
    stage: str | None = None
    #: Layers the view reads (documentation + availability check).
    layers: tuple[str, ...] = ()
    modes: tuple[Option, ...] = ()
    overlays: tuple[Option, ...] = ()
    note: str = ""
    render: Callable[[Project, str, frozenset], np.ndarray] | None = field(default=None, repr=False)
    legend: Callable[[Project, str], list[LegendItem]] | None = field(default=None, repr=False)

    def availability(self, project: Project | None) -> tuple[bool, str]:
        """Whether the view can be drawn for ``project``, and why not if it cannot."""
        if self.status != "ready":
            return False, "Запланировано"
        if project is None:
            return False, "Нет проекта"
        if self.stage == "v2" and project.geo is None:
            return False, f"«{self.title}» ещё не сгенерированы для этого проекта"
        return True, ""

    def default_mode(self) -> str:
        return self.modes[0].id if self.modes else ""

    def default_overlays(self) -> frozenset:
        return frozenset(o.id for o in self.overlays if o.default)


VIEWS: dict[str, ViewSpec] = {}


def register_view(spec: ViewSpec) -> ViewSpec:
    VIEWS[spec.id] = spec
    return spec


# -- shared drawing helpers ---------------------------------------------------
def id_colors(count: int, offset: float = 0.0, sat=(0.45, 0.7), val=(0.62, 0.9)) -> np.ndarray:
    """Deterministic, well separated colours for component ids."""
    out = np.zeros((max(count, 1), 3), dtype=np.uint8)
    for i in range(count):
        hue = (offset + i * 0.6180339887498949) % 1.0
        s = sat[0] + (sat[1] - sat[0]) * ((i * 7) % 5) / 4.0
        v = val[0] + (val[1] - val[0]) * ((i * 3) % 4) / 3.0
        out[i] = [int(c * 255) for c in colorsys.hsv_to_rgb(hue, s, v)]
    return out


def mask_rim(mask: np.ndarray, layout) -> np.ndarray:
    """Hexes of ``mask`` that share an edge with a hex outside it."""
    flat = mask.ravel()
    rim = np.zeros(flat.size, dtype=bool)
    for d in range(6):
        nb, valid = neighbor_index(layout, d)
        rim |= flat & valid.ravel() & ~flat[nb.ravel()]
    return rim.reshape(mask.shape)


def id_rim(ids: np.ndarray, layout) -> np.ndarray:
    """Hexes whose id differs from a neighbour's (outlines of components)."""
    flat = ids.ravel()
    rim = np.zeros(flat.size, dtype=bool)
    for d in hexgrid.HALF_DIRECTIONS:
        nb, valid = neighbor_index(layout, d)
        nbf = nb.ravel()
        diff = valid.ravel() & (flat[nbf] != flat)
        rim |= diff
    return rim.reshape(ids.shape)


def _shade(rgb: np.ndarray, mask: np.ndarray, factor: float, toward=(0, 0, 0)) -> None:
    rgb[mask] = (rgb[mask] * factor + np.array(toward) * (1.0 - factor)).astype(np.uint8)


def _plate_boundary_overlay(project: Project, rgb: np.ndarray) -> None:
    kind = project.layer("boundary_type")
    m = kind > 0
    rgb[m] = render.BOUNDARY_COLORS[kind[m]]


_HYPSO_STOPS = np.array(
    [-7000, -4500, -3000, -1500, -200, -1, 0, 150, 500, 1000, 2000, 3500, 5500], dtype=np.float64
)
_HYPSO_RGB = np.array(
    [
        [5, 12, 40],
        [10, 32, 82],
        [20, 62, 124],
        [44, 104, 160],
        [112, 170, 205],
        [176, 214, 232],
        [58, 132, 72],
        [112, 164, 86],
        [192, 190, 116],
        [186, 148, 92],
        [140, 104, 74],
        [176, 166, 160],
        [250, 250, 250],
    ],
    dtype=np.float64,
)


def hypsometric(relative_m: np.ndarray) -> np.ndarray:
    """Hypsometric tint for heights relative to sea level (metres)."""
    out = np.empty(relative_m.shape + (3,), dtype=np.uint8)
    for c in range(3):
        out[..., c] = np.interp(relative_m, _HYPSO_STOPS, _HYPSO_RGB[:, c]).astype(np.uint8)
    return out


def contour_rim(relative_m: np.ndarray, layout, land_step=500.0, sea_step=1000.0) -> np.ndarray:
    band = np.where(
        relative_m >= 0, np.floor(relative_m / land_step) + 100, np.floor(relative_m / sea_step)
    ).astype(np.int32)
    return id_rim(band, layout)


# -- Plates (v1.0) --------------------------------------------------------------
_PLATE_MODES = (
    Option("plates", "Плиты"),
    Option("crust", "Кора (prior v1.0)"),
    Option("boundaries", "Классы границ"),
    Option("motion", "Скорость движения"),
    Option("boundary_distance", "Близость к границе"),
    *(Option(name, render.LAYER_TITLES[name]) for name in FIELD_NAMES),
)


def _render_plates(project: Project, mode: str, overlays: frozenset) -> np.ndarray:
    rgb = render.layer_rgb(project, mode).copy()
    if "boundary_hexes" in overlays and mode != "boundaries":
        _plate_boundary_overlay(project, rgb)
    return rgb


def _legend_plates(project: Project, mode: str) -> list[LegendItem]:
    boundary = [
        LegendItem("сближение", tuple(render.BOUNDARY_COLORS[1])),
        LegendItem("расхождение", tuple(render.BOUNDARY_COLORS[2])),
        LegendItem("сдвиг", tuple(render.BOUNDARY_COLORS[3])),
        LegendItem("спокойная", tuple(render.BOUNDARY_COLORS[4])),
    ]
    if mode == "plates":
        return [LegendItem("цвет = плита (ID и имя - по щелчку)"), *boundary]
    if mode == "crust":
        return [
            LegendItem("континентальная кора (prior)", (196, 164, 110)),
            LegendItem("океаническая кора (prior)", (52, 92, 150)),
            LegendItem("не суша и не море: исходное поле v1.0"),
        ]
    if mode == "boundaries":
        return boundary
    low, high = tuple(render.ramp(np.array(0.0))), tuple(render.ramp(np.array(1.0)))
    if mode == "motion":
        vmax = float(np.hypot(project.layer("velocity_x"), project.layer("velocity_y")).max())
        return [LegendItem("0 мм/год", low), LegendItem(f"{vmax:.0f} мм/год", high)]
    if mode == "boundary_distance":
        far = project.config.fields.max_influence_km
        return [LegendItem(f"≥ {far:.0f} км от границы", low), LegendItem("на границе", high)]
    return [
        LegendItem("0 - нет сигнала", low),
        LegendItem("1 - максимум", high),
        LegendItem("диагностический потенциал, не рельеф"),
    ]


register_view(
    ViewSpec(
        id="plates",
        title="Плиты",
        stage="v1",
        layers=(
            "plate_id",
            "crust_type",
            "boundary_type",
            "velocity_x",
            "velocity_y",
            "boundary_distance_km",
            *FIELD_NAMES,
        ),
        modes=_PLATE_MODES,
        overlays=(
            Option("boundaries", "Линии границ", True),
            Option("motion", "Векторы движения", True),
            Option("boundary_hexes", "Гексы границ по классам"),
        ),
        note="Тектоническая основа v1.0",
        render=_render_plates,
        legend=_legend_plates,
    )
)

# -- Continents (v2.0) ----------------------------------------------------------
_DEEP = np.array([18, 44, 92], dtype=np.uint8)
_SHELF = np.array([122, 176, 208], dtype=np.uint8)
_SUBMERGED = np.array([62, 110, 160], dtype=np.uint8)


def _render_continents(project: Project, mode: str, overlays: frozenset) -> np.ndarray:
    geo = project.geo
    layout = project.layout
    land = geo.layers["land_mask"]
    block = geo.layers["continental_block_id"]
    shelf = geo.layers["shelf_mask"]
    shape = land.shape
    rgb = np.empty(shape + (3,), dtype=np.uint8)
    rgb[:] = _DEEP
    if mode == "blocks":
        colors = id_colors(len(geo.blocks), 0.13)
        inside = block >= 0
        rgb[inside] = colors[block[inside]]
        # the submerged part of a block keeps its hue but is pulled towards sea blue
        under = inside & ~land
        rgb[under] = (rgb[under] * 0.5 + _SUBMERGED * 0.5).astype(np.uint8)
        off_block_land = land & ~inside
        rgb[off_block_land] = (150, 150, 140)
    elif mode == "landmasses":
        rgb[(block >= 0) & ~land] = _SUBMERGED
        colors = id_colors(len(geo.landmasses), 0.41, sat=(0.4, 0.62), val=(0.66, 0.9))
        ids = geo.layers["landmass_id"]
        rgb[land] = colors[ids[land]]
    elif mode == "fraction":
        rgb = render.ramp(geo.layers["continental_fraction"]).copy()
    else:
        raise KeyError(f"Unknown mode {mode!r} of view 'continents'")
    if "shelf" in overlays and mode != "fraction":
        rgb[shelf] = (rgb[shelf] * 0.35 + _SHELF * 0.65).astype(np.uint8)
    if "block_outline" in overlays:
        _shade(rgb, mask_rim(block >= 0, layout), 0.45, (255, 235, 150))
    if "coast" in overlays:
        _shade(rgb, mask_rim(land, layout), 0.35)
    if "plate_boundaries" in overlays:
        _plate_boundary_overlay(project, rgb)
    return rgb


def _legend_continents(project: Project, mode: str) -> list[LegendItem]:
    geo = project.geo
    common = [
        LegendItem("шельф (затопленная окраина)", tuple(_SHELF)),
        LegendItem("океаническая кора", tuple(_DEEP)),
        LegendItem(f"уровень моря {geo.sea_level_m:+.0f} м"),
    ]
    if mode == "blocks":
        return [
            LegendItem(f"цвет = континентальный блок ({len(geo.blocks)} в окне)"),
            LegendItem("яркий - над водой, приглушённый - под водой"),
            LegendItem("суша вне континентальной коры", (150, 150, 140)),
            *common,
        ]
    if mode == "landmasses":
        majors = geo.summary["major_landmasses_in_window"]
        return [
            LegendItem(f"цвет = надводный массив ({len(geo.landmasses)} в окне, крупных {majors})"),
            LegendItem("континентальная кора под водой", tuple(_SUBMERGED)),
            *common,
        ]
    return [
        LegendItem("0 - океаническая кора", tuple(render.ramp(np.array(0.0)))),
        LegendItem("1 - континентальная кора", tuple(render.ramp(np.array(1.0)))),
    ]


register_view(
    ViewSpec(
        id="continents",
        title="Материки",
        stage="v2",
        layers=(
            "continental_fraction",
            "continental_block_id",
            "land_mask",
            "landmass_id",
            "shelf_mask",
        ),
        modes=(
            Option("landmasses", "Надводная суша (материки)"),
            Option("blocks", "Континентальные блоки (кора)"),
            Option("fraction", "Доля континентальной коры"),
        ),
        overlays=(
            Option("coast", "Береговая линия", True),
            Option("shelf", "Шельф", True),
            Option("block_outline", "Край континентальной коры"),
            Option("plate_boundaries", "Границы плит"),
        ),
        note="Блок коры ≠ надводный материк ≠ плита",
        render=_render_continents,
        legend=_legend_continents,
    )
)


# -- Heights (v2.0) -------------------------------------------------------------
def _render_heights(project: Project, mode: str, overlays: frozenset) -> np.ndarray:
    geo = project.geo
    layout = project.layout
    elev = geo.layers["macro_elevation_m"].astype(np.float64)
    rel = elev - geo.sea_level_m
    if mode == "hypsometric":
        rgb = hypsometric(rel)
    elif mode == "gray":
        lo, hi = float(elev.min()), float(elev.max())
        g = ((elev - lo) / max(1e-6, hi - lo) * 255.0).astype(np.uint8)
        rgb = np.repeat(g[..., None], 3, axis=2)
    elif mode == "tectonic":
        t = geo.layers["tectonic_relief_m"].astype(np.float64)
        scale = max(1.0, float(np.abs(t).max()))
        v = np.clip(t / scale, -1.0, 1.0)
        base = np.full(v.shape + (3,), 236.0)
        up = np.array([150.0, 30.0, 30.0])
        down = np.array([25.0, 60.0, 150.0])
        a = np.abs(v)[..., None] ** 0.7
        rgb = np.where(v[..., None] >= 0, base * (1 - a) + up * a, base * (1 - a) + down * a)
        rgb = rgb.astype(np.uint8)
    else:
        raise KeyError(f"Unknown mode {mode!r} of view 'heights'")
    if "contours" in overlays:
        _shade(rgb, contour_rim(rel, layout), 0.72)
    if "coast" in overlays:
        _shade(rgb, mask_rim(geo.layers["land_mask"], layout), 0.4)
    if "plate_boundaries" in overlays:
        _plate_boundary_overlay(project, rgb)
    return rgb


def _legend_heights(project: Project, mode: str) -> list[LegendItem]:
    geo = project.geo
    sea = geo.sea_level_m
    head = [LegendItem(f"уровень моря {sea:+.0f} м (0 м - нуль модели)")]
    if mode == "hypsometric":
        steps = [
            (-4500, "глубина 4500 м"),
            (-1500, "глубина 1500 м"),
            (-200, "глубина 200 м"),
            (1, "берег, 0 м"),
            (500, "высота 500 м"),
            (1000, "высота 1000 м"),
            (2000, "высота 2000 м"),
            (3500, "высота 3500 м и выше"),
        ]
        return head + [
            LegendItem(label, tuple(int(c) for c in hypsometric(np.array([float(v)]))[0]))
            for v, label in steps
        ]
    if mode == "gray":
        lo, hi = geo.summary["min_elevation_m"], geo.summary["max_elevation_m"]
        return head + [
            LegendItem(f"{lo:+.0f} м", (0, 0, 0)),
            LegendItem(f"{hi:+.0f} м", (255, 255, 255)),
        ]
    scale = float(np.abs(geo.layers["tectonic_relief_m"]).max())
    return head + [
        LegendItem(f"тектоническое поднятие до +{scale:.0f} м", (150, 30, 30)),
        LegendItem("без тектонического вклада", (236, 236, 236)),
        LegendItem(f"впадина до −{scale:.0f} м", (25, 60, 150)),
    ]


register_view(
    ViewSpec(
        id="heights",
        title="Высоты",
        stage="v2",
        layers=("macro_elevation_m", "tectonic_relief_m", "land_mask"),
        modes=(
            Option("hypsometric", "Высоты и глубины (цвет)"),
            Option("gray", "Абсолютная высота (серая шкала)"),
            Option("tectonic", "Тектонический вклад"),
        ),
        overlays=(
            Option("coast", "Береговая линия", True),
            Option("contours", "Изолинии (500 м суша / 1000 м море)"),
            Option("plate_boundaries", "Границы плит"),
        ),
        note="Макрорельеф v2.0 (без детализации и эрозии)",
        render=_render_heights,
        legend=_legend_heights,
    )
)

# -- reserved for future stages: registered, never drawn as ready ---------------
for _id, _title in (
    ("rivers", "Реки"),
    ("climate", "Климат"),
    ("biomes", "Биомы"),
    ("geosymbols", "Геосимволы"),
):
    register_view(ViewSpec(id=_id, title=_title, status="planned", note="Запланировано"))


def view_rgb(
    project: Project, view_id: str, mode: str | None = None, overlays: frozenset | set | None = None
) -> np.ndarray:
    """Colour of every hex for a view. Raises if the view has no data for ``project``."""
    spec = VIEWS[view_id]
    ok, reason = spec.availability(project)
    if not ok:
        raise LookupError(reason)
    mode = mode or spec.default_mode()
    if mode not in {m.id for m in spec.modes}:
        raise KeyError(f"View {view_id!r} has no mode {mode!r}")
    chosen = spec.default_overlays() if overlays is None else frozenset(overlays)
    return spec.render(project, mode, chosen)


def render_view_image(
    project: Project,
    view_id: str,
    mode: str | None = None,
    overlays=None,
    px_per_hex: float | None = None,
    max_size: int = 2400,
) -> np.ndarray:
    """Render the whole window for a view as an RGB image (display-only parameters)."""
    spec = VIEWS[view_id]
    mode = mode or spec.default_mode()
    chosen = spec.default_overlays() if overlays is None else frozenset(overlays)
    rgb = view_rgb(project, view_id, mode, chosen)
    b = project.geometry.viewport_bounds
    spacing = project.layout.spacing_km
    if px_per_hex is None:
        px_per_hex = min(12.0, max_size / max(b.width / spacing, b.height / spacing))
    km_per_px = spacing / px_per_hex
    out_w = max(1, int(round(b.width / km_per_px)))
    out_h = max(1, int(round(b.height / km_per_px)))
    img = render.render_view(
        project.layout,
        rgb,
        b.xmin,
        b.ymin,
        km_per_px,
        out_w,
        out_h,
        outline=False if px_per_hex < 9 else None,
    )
    if view_id == "plates" and "motion" in chosen:
        render.draw_motion_arrows(img, project, km_per_px)
    return img
