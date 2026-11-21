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


@dataclass
class VectorLayer:
    """Lines or points in world km drawn on top of the hex colours (GUI and PNG alike)."""

    kind: str  # "lines" | "points"
    x: object  # points: array; lines: list of arrays (one per polyline)
    y: object
    color: tuple[int, int, int]
    size: float = 2.0  # line width or point radius, px
    filled: bool = True
    label: str = ""
    #: Real radius of the object, km. When set, the mark is drawn to scale and never
    #: smaller than ``size`` pixels - so a symbol does not exaggerate what it stands for.
    size_km: float = 0.0


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
    #: Optional vector overlays: ``vectors(project, mode, overlays) -> list[VectorLayer]``.
    vectors: Callable | None = field(default=None, repr=False)
    #: The view shows one month at a time: the GUI offers a month selector and passes the
    #: choice as the overlay token ``"month:<0..11>"`` (a display choice, never a recompute).
    months: bool = False

    def availability(self, project: Project | None) -> tuple[bool, str]:
        """Whether the view can be drawn for ``project``, and why not if it cannot."""
        if self.status != "ready":
            return False, "Запланировано"
        if project is None:
            return False, "Нет проекта"
        if self.stage in ("v2", "v3", "v46", "v4", "v45") and project.geo is None:
            return False, f"«{self.title}» ещё не сгенерированы для этого проекта"
        if self.stage == "v4" and project.hydro is None:
            return False, f"«{self.title}»: климат и гидрология (v4) ещё не рассчитаны"
        if self.stage in ("v3", "v4") and project.hist is None:
            return False, f"«{self.title}»: геоистория ещё не сгенерирована для этого проекта"
        if self.stage == "v46" and (
            project.hist is None
            or getattr(project, "terrain", None) is None
            or project.hydro is None
        ):
            return False, f"«{self.title}»: v4.6 (массивы, бассейны, дно) ещё не рассчитан"
        if self.stage == "v45" and project.env is None:
            return False, f"«{self.title}»: экологические параметры (v4.5) не рассчитаны"
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


#: dry land below the sea level (an atlas convention: dark green depressions)
DEPRESSION_RGB = np.array([[58, 132, 72], [26, 82, 44]], dtype=np.float64)
LAKE_RGB = (96, 176, 214)


def hypsometric(relative_m: np.ndarray, ocean=None, lake=None) -> np.ndarray:
    """Hypsometric tint for heights relative to sea level (metres).

    Land: a continuous ramp. Sea: the stepped depth tints of a physical atlas (shelf to
    200 m, then 1000, 2000, ... m; deeper than 6 km - trenches - almost black), so shelves,
    slopes, abyssal plains and trenches read at a glance. With the v4.6 water masks the sea
    is the world ocean only: land below the sea level is a dark-green depression, a lake is
    lake-blue - never ocean blue just because it is low.
    """
    from .stages.ocean import depth_tint

    rel = np.asarray(relative_m, dtype=np.float64)
    out = np.empty(rel.shape + (3,), dtype=np.uint8)
    for c in range(3):
        out[..., c] = np.interp(rel, _HYPSO_STOPS, _HYPSO_RGB[:, c]).astype(np.uint8)
    sea = rel < 0.0 if ocean is None else np.asarray(ocean, dtype=bool)
    if sea.any():
        out[sea] = depth_tint(np.maximum(-rel[sea], 0.0))
    if ocean is not None:
        dep = ~sea & (rel < 0.0)
        if dep.any():
            t = np.clip(-rel[dep] / 500.0, 0.0, 1.0)[:, None]
            out[dep] = (DEPRESSION_RGB[0] * (1 - t) + DEPRESSION_RGB[1] * t).astype(np.uint8)
    if lake is not None:
        out[np.asarray(lake, dtype=bool)] = LAKE_RGB
    return out


def water_masks(project: Project):
    """(ocean, lake) of the final surface (v4.6), or (None, None) for older projects."""
    if project.hydro is not None and "ocean_mask" in project.hydro.layers:
        L = project.hydro.layers
        return L["ocean_mask"], L["inland_water_mask"]
    return None, None


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


def _land_of(project: Project, baseline: bool):
    """(land, landmass_id, count, shelf) of the final surface or of the v2 baseline."""
    if project.hydro is not None and not baseline:
        h = project.hydro
        return (
            h.layers["land_mask"],
            h.layers["landmass_id"],
            len(h.landmasses),
            h.layers["shelf_mask"],
        )
    if project.hist is not None and not baseline:
        h = project.hist
        return (
            h.layers["land_mask"],
            h.layers["landmass_id"],
            len(h.landmasses),
            h.layers["shelf_mask"],
        )
    g = project.geo
    return g.layers["land_mask"], g.layers["landmass_id"], len(g.landmasses), g.layers["shelf_mask"]


def _epoch_note(project: Project, baseline: bool) -> str:
    if project.hist is None or baseline:
        return "исходная эпоха v2 (t = 0)"
    return f"эпоха v3: t = {project.hist.epoch_time_ma:g} млн лет"


def _render_continents(project: Project, mode: str, overlays: frozenset) -> np.ndarray:
    geo = project.geo
    layout = project.layout
    baseline = mode in ("baseline_landmasses", "blocks", "fraction")
    land, land_ids, land_count, shelf = _land_of(project, baseline)
    block = geo.layers["continental_block_id"]
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
    elif mode in ("landmasses", "baseline_landmasses"):
        if baseline or project.hist is None:
            continental = block >= 0
        else:
            continental = project.hist.layers["continental_fraction"] >= 0.5
        rgb[continental & ~land] = _SUBMERGED
        colors = id_colors(land_count, 0.41, sat=(0.4, 0.62), val=(0.66, 0.9))
        rgb[land] = colors[land_ids[land]]
    elif mode == "compare":
        if project.hist is None:
            raise LookupError("Сравнение эпох доступно после генерации геоистории (v3.0)")
        then = geo.layers["land_mask"]
        now = project.hist.layers["land_mask"]
        rgb[then & now] = (150, 160, 120)
        rgb[now & ~then] = (60, 170, 80)
        rgb[then & ~now] = (60, 120, 200)
        land = now
        shelf = project.hist.layers["shelf_mask"]
    elif mode == "fraction":
        rgb = render.ramp(geo.layers["continental_fraction"]).copy()
    else:
        raise KeyError(f"Unknown mode {mode!r} of view 'continents'")
    if "shelf" in overlays and mode not in ("fraction", "compare"):
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
    baseline = mode in ("baseline_landmasses", "blocks", "fraction")
    sea = project.current_sea_level_m
    common = [
        LegendItem("шельф (затопленная окраина)", tuple(_SHELF)),
        LegendItem("океаническая кора", tuple(_DEEP)),
        LegendItem(f"уровень моря {sea:+.0f} м"),
        LegendItem(f"источник: {_epoch_note(project, baseline)}"),
    ]
    if mode == "blocks":
        return [
            LegendItem(f"цвет = континентальный блок ({len(geo.blocks)} в окне)"),
            LegendItem("яркий - над водой, приглушённый - под водой"),
            LegendItem("суша вне континентальной коры", (150, 150, 140)),
            *common,
        ]
    if mode in ("landmasses", "baseline_landmasses"):
        _land, _ids, count, _shelf = _land_of(project, baseline)
        return [
            LegendItem(f"цвет = надводный массив ({count} в окне)"),
            LegendItem("континентальная кора под водой", tuple(_SUBMERGED)),
            *common,
        ]
    if mode == "compare":
        return [
            LegendItem("суша в обе эпохи", (150, 160, 120)),
            LegendItem(f"новая суша к t = {project.hist.epoch_time_ma:g} млн лет", (60, 170, 80)),
            LegendItem("суша исходной эпохи, теперь под водой", (60, 120, 200)),
            LegendItem(f"уровень моря {sea:+.0f} м (один для обеих эпох)"),
        ]
    return [
        LegendItem("0 - океаническая кора", tuple(render.ramp(np.array(0.0)))),
        LegendItem("1 - континентальная кора", tuple(render.ramp(np.array(1.0)))),
        LegendItem("источник: исходная эпоха v2 (t = 0)"),
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
            Option("landmasses", "Надводная суша текущей эпохи"),
            Option("compare", "Сравнение: исходная эпоха ↔ текущая"),
            Option("baseline_landmasses", "Надводная суша исходной эпохи (v2)"),
            Option("blocks", "Континентальные блоки (кора, v2)"),
            Option("fraction", "Доля континентальной коры (v2)"),
        ),
        overlays=(
            Option("coast", "Береговая линия", True),
            Option("shelf", "Шельф", True),
            Option("block_outline", "Край континентальной коры"),
            Option("plate_boundaries", "Границы плит (исходные)"),
        ),
        note="Блок коры ≠ надводный материк ≠ плита",
        render=_render_continents,
        legend=_legend_continents,
    )
)


# -- Heights (v2.0) -------------------------------------------------------------
def _height_source(project: Project, mode: str):
    """(elevation, land, label) for the heights view: v4 final, v3 epoch or the v2 baseline."""
    if project.hydro is not None and mode != "baseline":
        h = project.hydro
        label = (
            f"итоговая высота v4 после эрозии (поверхность t = {h.epoch_time_ma:g} "
            "млн лет + климатическая эрозия)"
        )
        return h.layers["elevation_m"], h.layers["land_mask"], label
    if project.hist is not None and mode != "baseline":
        h = project.hist
        label = f"итоговая высота v3, t = {h.epoch_time_ma:g} млн лет"
        return h.layers["elevation_m"], h.layers["land_mask"], label
    g = project.geo
    label = "исходная макровысота v2 (t = 0): макрорельеф без детализации и эрозии"
    return g.layers["macro_elevation_m"], g.layers["land_mask"], label


WATER_CLASSES = (
    ("мировой океан", (40, 90, 170)),
    ("озеро / внутреннее море (свой уровень)", LAKE_RGB),
    ("суша ниже уровня моря (сухая)", (40, 110, 55)),
    ("суша выше уровня моря", (215, 205, 175)),
    ("ниже уровня моря, но не океан: расчёт v3 без связности", (150, 150, 150)),
)


def _water_rgb(project: Project, rel) -> np.ndarray:
    ocean, lake = water_masks(project)
    rgb = np.empty(np.shape(rel) + (3,), dtype=np.uint8)
    if ocean is None:
        rgb[:] = WATER_CLASSES[3][1]
        rgb[rel <= 0] = WATER_CLASSES[4][1]
        return rgb
    rgb[:] = WATER_CLASSES[3][1]
    rgb[~ocean & (rel < 0)] = WATER_CLASSES[2][1]
    rgb[lake] = WATER_CLASSES[1][1]
    deep = np.clip(-rel / 6000.0, 0.0, 1.0)[..., None]
    sea = (np.array(WATER_CLASSES[0][1]) * (1.15 - 0.6 * deep)).clip(0, 255).astype(np.uint8)
    rgb[ocean] = sea[ocean]
    return rgb


def hex_gradient(values: np.ndarray, layout):
    """Least-squares gradient (per km) of a hex layer from its six neighbours."""
    flat = values.ravel().astype(np.float64)
    units = hexgrid.direction_unit_vectors()
    gx = np.zeros(flat.size)
    gy = np.zeros(flat.size)
    for d in range(6):
        nb, valid = neighbor_index(layout, d)
        dz = np.where(valid.ravel(), flat[nb.ravel()] - flat, 0.0)
        gx += dz * units[d, 0]
        gy += dz * units[d, 1]
    scale = 1.0 / (3.0 * layout.spacing_km)
    return (gx * scale).reshape(values.shape), (gy * scale).reshape(values.shape)


def hillshade(elevation: np.ndarray, layout, exaggeration: float = 14.0) -> np.ndarray:
    """Shading 0..1 with light from the north-west."""
    gx, gy = hex_gradient(elevation, layout)
    nx, ny, nz = -gx * exaggeration / 1000.0, -gy * exaggeration / 1000.0, 1.0
    norm = np.sqrt(nx * nx + ny * ny + nz * nz)
    lx, ly, lz = -0.5, -0.5, 0.707
    return np.clip((nx * lx + ny * ly + nz * lz) / norm, 0.0, 1.0)


def _render_heights(project: Project, mode: str, overlays: frozenset) -> np.ndarray:
    geo = project.geo
    layout = project.layout
    source, land, _label = _height_source(project, mode)
    elev = source.astype(np.float64)
    sea = project.current_sea_level_m
    rel = elev - sea
    if mode == "hypsometric":
        ocean, lake = water_masks(project)
        rgb = hypsometric(rel, ocean, lake)
    elif mode == "baseline":
        rgb = hypsometric(rel)
    elif mode == "water":
        rgb = _water_rgb(project, rel)
    elif mode == "gray":
        lo, hi = float(elev.min()), float(elev.max())
        g = ((elev - lo) / max(1e-6, hi - lo) * 255.0).astype(np.uint8)
        rgb = np.repeat(g[..., None], 3, axis=2)
    elif mode == "tectonic":
        t = geo.layers["tectonic_relief_m"].astype(np.float64)
        rgb = diverging(t, max(1.0, float(np.abs(t).max())))
    else:
        raise KeyError(f"Unknown mode {mode!r} of view 'heights'")
    if "contours" in overlays:
        _shade(rgb, contour_rim(rel, layout), 0.72)
    if "coast" in overlays:
        _shade(rgb, mask_rim(land, layout), 0.4)
    if "plate_boundaries" in overlays:
        _plate_boundary_overlay(project, rgb)
    return rgb


def diverging(values: np.ndarray, scale: float) -> np.ndarray:
    """Red (positive) - light grey (zero) - blue (negative)."""
    v = np.clip(values / scale, -1.0, 1.0)
    base = np.full(v.shape + (3,), 236.0)
    up = np.array([150.0, 30.0, 30.0])
    down = np.array([25.0, 60.0, 150.0])
    a = np.abs(v)[..., None] ** 0.7
    return np.where(v[..., None] >= 0, base * (1 - a) + up * a, base * (1 - a) + down * a).astype(
        np.uint8
    )


def _legend_heights(project: Project, mode: str) -> list[LegendItem]:
    geo = project.geo
    sea = project.current_sea_level_m
    source, _land, label = _height_source(project, mode)
    head = [
        LegendItem(f"источник: {label}"),
        LegendItem(f"уровень моря {sea:+.0f} м (0 м - нуль модели)"),
    ]
    if mode == "water":
        ocean, _lake = water_masks(project)
        items = [LegendItem(t, tuple(int(c) for c in rgb)) for t, rgb in WATER_CLASSES[:4]]
        if ocean is not None:
            L = project.hydro.layers
            items += [
                LegendItem(
                    f"озёр: {int(L['inland_water_mask'].sum())} гексов; суши ниже моря: "
                    f"{int(L['below_sea_level_land_mask'].sum())} гексов"
                ),
                LegendItem("океан — только связное с мировым океаном (не знак высоты)"),
            ]
        else:
            items.append(
                LegendItem("проект без v4.6: связность не рассчитана", WATER_CLASSES[4][1])
            )
        return head + items
    if mode in ("hypsometric", "baseline"):
        steps = [
            (-7500, "глубже 7000 м (дно желобов)"),
            (-6500, "6000–7000 м: глубоководные желоба"),
            (-5500, "5000–6000 м"),
            (-4500, "4000–5000 м: абиссаль"),
            (-3500, "3000–4000 м"),
            (-2500, "2000–3000 м"),
            (-1500, "1000–2000 м"),
            (-500, "200–1000 м: материковый склон"),
            (-100, "0–200 м: шельф"),
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
        lo, hi = float(source.min()), float(source.max())
        return head + [
            LegendItem(f"{lo:+.0f} м", (0, 0, 0)),
            LegendItem(f"{hi:+.0f} м", (255, 255, 255)),
        ]
    scale = float(np.abs(geo.layers["tectonic_relief_m"]).max())
    return [
        LegendItem("источник: тектонический вклад исходной макровысоты v2"),
        LegendItem(f"поднятие до +{scale:.0f} м", (150, 30, 30)),
        LegendItem("без тектонического вклада", (236, 236, 236)),
        LegendItem(f"впадина до −{scale:.0f} м", (25, 60, 150)),
    ]


register_view(
    ViewSpec(
        id="heights",
        title="Высоты",
        stage="v2",
        layers=("elevation_m", "macro_elevation_m", "tectonic_relief_m", "land_mask"),
        modes=(
            Option("hypsometric", "Текущая высота (итог v3 или v2)"),
            Option("water", "Вода: океан, озёра, суша ниже уровня моря"),
            Option("baseline", "Исходная макровысота v2"),
            Option("gray", "Абсолютная высота (серая шкала)"),
            Option("tectonic", "Тектонический вклад v2"),
        ),
        overlays=(
            Option("coast", "Береговая линия", True),
            Option("contours", "Изолинии (500 м суша / 1000 м море)"),
            Option("plate_boundaries", "Границы плит (исходные)"),
        ),
        note="Источник высоты указан в легенде",
        render=_render_heights,
        legend=_legend_heights,
    )
)

# -- Geo-history (v3.0) ---------------------------------------------------------
_EVENT_COLORS = {
    "collision_start": (214, 48, 49),
    "subduction_start": (160, 60, 200),
    "rifting_start": (9, 132, 227),
    "ocean_opening": (0, 180, 200),
    "orogeny": (120, 80, 40),
    "volcanic_episode": (240, 140, 20),
}
_EVENT_TITLES = {
    "collision_start": "начало коллизии",
    "subduction_start": "начало субдукции",
    "rifting_start": "начало рифтинга",
    "ocean_opening": "раскрытие океана",
    "orogeny": "орогенез",
    "volcanic_episode": "вулканический эпизод (горячая точка)",
}
_BOUNDARY_EPOCH = np.array(
    [[0, 0, 0], [214, 48, 49], [160, 60, 200], [9, 132, 227], [0, 168, 107]], dtype=np.uint8
)
_AGE_NONE = np.array([70, 74, 82], dtype=np.uint8)


def _age_ramp(age: np.ndarray, oldest: float) -> np.ndarray:
    """Young = bright, old = dark; NaN (never) = neutral grey."""
    rgb = render.ramp(1.0 - np.clip(np.nan_to_num(age, nan=oldest) / oldest, 0.0, 1.0)).copy()
    rgb[np.isnan(age)] = _AGE_NONE
    return rgb


def _render_history(project: Project, mode: str, overlays: frozenset) -> np.ndarray:
    h = project.hist
    layout = project.layout
    L = h.layers
    if mode == "delta":
        rgb = diverging(L["delta_elevation_m"].astype(np.float64), 3000.0)
    elif mode == "plates":
        rgb = render.plate_color_table(project)[L["plate_id_epoch"]].copy()
        fresh = L["new_crust"]
        rgb[fresh] = (rgb[fresh] * 0.45 + np.array([30, 60, 110]) * 0.55).astype(np.uint8)
    elif mode == "uplift_rate":
        rgb = diverging(L["uplift_rate_mm_yr"].astype(np.float64), 0.25)
    elif mode == "last_event":
        rgb = _age_ramp(L["last_tectonic_ma"], max(1.0, h.duration_ma))
    elif mode == "crust_age":
        age = L["crust_age_ma"]
        rgb = render.ramp(1.0 - np.clip(np.log1p(age) / np.log1p(2500.0), 0.0, 1.0)).copy()
    else:
        raise KeyError(f"Unknown mode {mode!r} of view 'history'")
    if "boundaries" in overlays:
        cls = L["boundary_class"]
        m = cls > 0
        rgb[m] = _BOUNDARY_EPOCH[cls[m]]
    if "coast" in overlays:
        _shade(rgb, mask_rim(L["land_mask"], layout), 0.4)
    if "baseline_coast" in overlays:
        _shade(rgb, mask_rim(project.geo.layers["land_mask"], layout), 0.35, (255, 255, 255))
    return rgb


def _legend_history(project: Project, mode: str) -> list[LegendItem]:
    h = project.hist
    head = [
        LegendItem(
            f"снимок: t = {h.epoch_time_ma:g} из {h.duration_ma:g} млн лет "
            f"(t = 0 - исходная эпоха v1/v2)"
        )
    ]
    if mode == "delta":
        body = [
            LegendItem("выше исходной на 3000 м и более", (150, 30, 30)),
            LegendItem("без изменений", (236, 236, 236)),
            LegendItem("ниже исходной на 3000 м и более", (25, 60, 150)),
        ]
    elif mode == "plates":
        body = [
            LegendItem("цвет = плита, владеющая гексом в этот момент"),
            LegendItem("затемнено - кора, созданная за время истории"),
        ]
    elif mode == "uplift_rate":
        body = [
            LegendItem("поднятие ≥ 0,25 мм/год", (150, 30, 30)),
            LegendItem("покой", (236, 236, 236)),
            LegendItem("опускание ≥ 0,25 мм/год", (25, 60, 150)),
        ]
    elif mode == "last_event":
        body = [
            LegendItem("изменения прямо сейчас", tuple(render.ramp(np.array(1.0)))),
            LegendItem(
                f"{h.duration_ma:g} млн лет назад и раньше", tuple(render.ramp(np.array(0.0)))
            ),
            LegendItem("тектонических изменений не было", tuple(_AGE_NONE)),
        ]
    else:
        body = [
            LegendItem("молодая кора", tuple(render.ramp(np.array(1.0)))),
            LegendItem(
                "древняя кора (логарифмическая шкала до 2500 млн лет)",
                tuple(render.ramp(np.array(0.0))),
            ),
        ]
    lines = [
        LegendItem("границы эпохи: коллизия", tuple(_BOUNDARY_EPOCH[1])),
        LegendItem("субдукция", tuple(_BOUNDARY_EPOCH[2])),
        LegendItem("расхождение", tuple(_BOUNDARY_EPOCH[3])),
        LegendItem("прочие", tuple(_BOUNDARY_EPOCH[4])),
    ]
    chronology = [LegendItem("— хронология —")]
    for e in h.events[:10]:
        chronology.append(
            LegendItem(
                f"{e['time_ma']:.1f} млн лет: {_EVENT_TITLES.get(e['type'], e['type'])}",
                _EVENT_COLORS.get(e["type"]),
            )
        )
    if len(h.events) > 10:
        chronology.append(LegendItem(f"… всего событий: {len(h.events)}"))
    return head + body + lines + chronology


def _vectors_history(project: Project, mode: str, overlays: frozenset) -> list[VectorLayer]:
    out: list[VectorLayer] = []
    if "events" not in overlays:
        return out
    h = project.hist
    for kind, color in _EVENT_COLORS.items():
        pts = [
            (e["x_km"], e["y_km"])
            for e in h.events
            if e["type"] == kind and e["time_ma"] <= h.epoch_time_ma + 1e-9
        ]
        if pts:
            arr = np.array(pts)
            out.append(VectorLayer("points", arr[:, 0], arr[:, 1], color, 4.5, True, kind))
    return out


register_view(
    ViewSpec(
        id="history",
        title="Геоистория",
        stage="v3",
        layers=(
            "delta_elevation_m",
            "plate_id_epoch",
            "uplift_rate_mm_yr",
            "last_tectonic_ma",
            "crust_age_ma",
            "boundary_class",
        ),
        modes=(
            Option("delta", "Изменение высоты с исходной эпохи"),
            Option("plates", "Плиты в выбранную эпоху"),
            Option("uplift_rate", "Скорость поднятия / опускания"),
            Option("last_event", "Время последнего тектонического события"),
            Option("crust_age", "Возраст коры"),
        ),
        overlays=(
            Option("boundaries", "Границы плит эпохи", True),
            Option("events", "События", True),
            Option("coast", "Берег эпохи", True),
            Option("baseline_coast", "Берег исходной эпохи"),
        ),
        note="Прямая симуляция от t = 0; упрощённая кинематика жёстких плит",
        render=_render_history,
        legend=_legend_history,
        vectors=_vectors_history,
    )
)

# -- Relief (v3.0) --------------------------------------------------------------
_CLASS_COLORS = np.array(
    [
        [58, 64, 72],
        [236, 236, 236],
        [150, 120, 96],
        [240, 190, 60],
        [70, 130, 90],
        [128, 142, 110],
        [214, 80, 60],
    ],
    dtype=np.uint8,
)
_CLASS_TITLES = (
    "вне гор",
    "гребень",
    "склон",
    "перевал",
    "долина",
    "предгорье",
    "вулканическая постройка",
)
_ROCK_COLORS = np.array(
    [
        [40, 70, 120],
        [170, 90, 110],
        [214, 196, 140],
        [130, 110, 150],
        [200, 80, 50],
        [226, 214, 170],
    ],
    dtype=np.uint8,
)
_ROCK_TITLES = (
    "океанический базальт",
    "кристаллический щит",
    "осадочный чехол платформы",
    "складчато-метаморфический пояс",
    "вулканические породы",
    "рыхлые осадки",
)


_STYLE_RGB = {
    "volcanic_arc": (190, 60, 50),
    "alpine": (120, 90, 200),
    "himalayan": (240, 240, 250),
    "andean": (220, 150, 60),
    "fold_belt": (90, 160, 90),
    "old_massif": (160, 130, 90),
    "broad_upland": (200, 190, 120),
    "mixed": (70, 150, 200),
    "block": (180, 90, 150),
    "highland": (150, 110, 60),
    "composite": (230, 90, 120),
}
_STYLE_TITLES = {
    "volcanic_arc": "вулканическая дуга",
    "alpine": "альпийский: острый главный хребет, второстепенные, предгорья",
    "himalayan": "гималайский: предгорья, хребты, высокое плато, тыловой хребет",
    "andean": "андский: кордильеры и альтиплано",
    "fold_belt": "складчатый: много параллельных гряд",
    "old_massif": "древний массив: широкий, низкий, сглаженный",
    "broad_upland": "широкое нагорье, пологие края",
    "mixed": "смешанный: широкое пологое основание и острое ядро",
    "block": "глыбовый: наклонённые блоки, крутые с одной стороны",
    "highland": "горная страна: много хребтов разного простирания, котловины",
    "composite": "горный узел: пучки складчатых хребтов расходятся из узла",
}


def _render_relief(project: Project, mode: str, overlays: frozenset) -> np.ndarray:
    h = project.hist
    layout = project.layout
    L = h.layers
    elev = L["elevation_m"].astype(np.float64)
    if mode == "shaded":
        shade = hillshade(elev, layout)
        rgb = hypsometric(elev - h.sea_level_m, ~L["land_mask"])
        rgb = (rgb * (0.45 + 0.75 * shade[..., None])).clip(0, 255)
        rgb = rgb.astype(np.uint8)
    elif mode == "slope":
        rgb = render.ramp(np.clip(L["slope_m_per_km"] / 60.0, 0.0, 1.0) ** 0.6).copy()
    elif mode == "classes":
        rgb = _CLASS_COLORS[L["mountain_class"]].copy()
        sea = ~L["land_mask"] & (L["mountain_class"] == 0)
        rgb[sea] = (24, 44, 84)
    elif mode == "systems":
        ids = L["mountain_system_id"]
        shade = hillshade(elev, layout)
        rgb = np.where(L["land_mask"][..., None], np.array([92, 98, 92]), np.array([24, 44, 84]))
        rgb = rgb.astype(np.float64)
        colors = id_colors(max(1, len(h.plan["systems"])), 0.27)
        inside = ids >= 0
        rgb[inside] = colors[ids[inside]]
        rgb = (rgb * (0.55 + 0.6 * shade[..., None])).clip(0, 255).astype(np.uint8)
    elif mode == "styles":
        ids = L["mountain_system_id"]
        shade = hillshade(elev, layout)
        rgb = np.where(L["land_mask"][..., None], np.array([150, 150, 140]), np.array([24, 44, 84]))
        rgb = rgb.astype(np.float64)
        table = np.array(
            [_STYLE_RGB.get(s.get("style", ""), (120, 120, 120)) for s in h.plan["systems"]]
            or [(0, 0, 0)],
            dtype=np.float64,
        )
        inside = ids >= 0
        rgb[inside] = table[ids[inside]]
        rgb = (rgb * (0.5 + 0.65 * shade[..., None])).clip(0, 255).astype(np.uint8)
    elif mode == "rock":
        rgb = _ROCK_COLORS[L["rock_type"]].copy()
    elif mode == "resistance":
        rgb = render.ramp(L["rock_resistance"]).copy()
    elif mode == "orogen_age":
        rgb = _age_ramp(L["orogen_age_ma"], 200.0)
    elif mode == "sediment":
        rgb = render.ramp(np.clip(L["sediment_thickness_m"] / 1500.0, 0.0, 1.0) ** 0.5).copy()
    elif mode == "erosion":
        rgb = render.ramp(np.clip(L["erosion_total_m"] / 2500.0, 0.0, 1.0) ** 0.5).copy()
    else:
        raise KeyError(f"Unknown mode {mode!r} of view 'relief'")
    if "coast" in overlays:
        _shade(rgb, mask_rim(L["land_mask"], layout), 0.4)
    return rgb


def _legend_relief(project: Project, mode: str) -> list[LegendItem]:
    h = project.hist
    backend = h.summary["mountain_backend"]
    head = [
        LegendItem(f"снимок: t = {h.epoch_time_ma:g} млн лет; уровень моря {h.sea_level_m:+.0f} м"),
        LegendItem(
            "горы: "
            + (
                "PMGE (" + backend.get("library", "") + ")"
                if backend["is_pmge"]
                else "встроенный модуль (библиотека PMGE не найдена)"
            )
        ),
    ]
    lo, hi = tuple(render.ramp(np.array(0.0))), tuple(render.ramp(np.array(1.0)))
    if mode == "shaded":
        return head + [
            LegendItem("итоговая высота v3 с отмывкой, свет с северо-запада"),
            LegendItem(
                f"высоты {h.summary['min_elevation_m']:+.0f} … "
                f"{h.summary['max_elevation_m']:+.0f} м"
            ),
        ]
    if mode == "slope":
        return head + [LegendItem("0 м/км", lo), LegendItem("60 м/км и круче", hi)]
    if mode == "classes":
        return head + [
            LegendItem(t, tuple(c)) for t, c in zip(_CLASS_TITLES, _CLASS_COLORS, strict=True)
        ]
    if mode == "styles":
        counts: dict[str, int] = {}
        for s in h.plan["systems"]:
            counts[s.get("style", "")] = counts.get(s.get("style", ""), 0) + 1
        return head + [
            LegendItem(f"{_STYLE_TITLES[k]} — {counts.get(k, 0)}", _STYLE_RGB[k])
            for k in sorted(_STYLE_RGB, key=lambda k: -counts.get(k, 0))
        ]
    if mode == "systems":
        items = [LegendItem(f"горных систем в плане: {len(h.plan['systems'])}")]
        names = {
            "young_sharp": "молодые острые",
            "mature": "зрелые",
            "old_rounded": "древние сглаженные",
            "plateau": "плато",
            "volcanic_arc": "вулканическая дуга",
        }
        colors = id_colors(max(1, len(h.plan["systems"])), 0.27)
        for s in h.plan["systems"][:8]:
            items.append(
                LegendItem(
                    f"№{s['id']}: {names.get(s['morphology'], s['morphology'])}, "
                    f"{s['age_ma']:.0f} млн лет, {s['length_km']:,.0f} км",
                    tuple(colors[s["id"]]),
                )
            )
        return head + items
    if mode == "rock":
        return head + [
            LegendItem(t, tuple(c)) for t, c in zip(_ROCK_TITLES, _ROCK_COLORS, strict=True)
        ]
    if mode == "resistance":
        return head + [LegendItem("легко размывается", lo), LegendItem("стойкая порода", hi)]
    if mode == "orogen_age":
        return head + [
            LegendItem("горообразование идёт сейчас", hi),
            LegendItem("200 млн лет и старше", lo),
            LegendItem("горообразования не было", tuple(_AGE_NONE)),
        ]
    if mode == "sediment":
        return head + [LegendItem("0 м", lo), LegendItem("1500 м и больше осадков", hi)]
    return head + [LegendItem("0 м", lo), LegendItem("снято 2500 м и больше", hi)]


def _polylines(items, key="points_km", bounds=None):
    """Polylines of plan items; with ``bounds`` only their parts inside the window."""
    xs, ys = [], []
    for item in items:
        pts = np.asarray(item[key])
        if len(pts) < 2:
            continue
        if bounds is None:
            xs.append(pts[:, 0])
            ys.append(pts[:, 1])
            continue
        inside = bounds.contains(pts[:, 0], pts[:, 1])
        edges = np.diff(np.concatenate([[0], inside.astype(np.int8), [0]]))
        for a, b in zip(np.nonzero(edges == 1)[0], np.nonzero(edges == -1)[0], strict=True):
            if b - a >= 2:
                xs.append(pts[a:b, 0])
                ys.append(pts[a:b, 1])
    return xs, ys


def _vectors_relief(project: Project, mode: str, overlays: frozenset) -> list[VectorLayer]:
    plan = project.hist.plan
    out: list[VectorLayer] = []
    if "valleys" in overlays and plan["valleys"]:
        xs, ys = _polylines(plan["valleys"], bounds=project.geometry.viewport_bounds)
        out.append(VectorLayer("lines", xs, ys, (60, 150, 220), 1.2, True, "valleys"))
    if "ridges" in overlays:
        for rank, color, width in (
            ("spur", (120, 84, 60), 1.0),
            ("branch", (70, 44, 30), 1.6),
            ("main", (30, 18, 12), 2.4),
        ):
            xs, ys = _polylines(
                [r for r in plan["ridges"] if r["rank"] == rank],
                bounds=project.geometry.viewport_bounds,
            )
            if xs:
                out.append(VectorLayer("lines", xs, ys, color, width, True, f"ridges_{rank}"))
    if "passes" in overlays and plan["passes"]:
        arr = np.array([[p["x_km"], p["y_km"]] for p in plan["passes"]])
        out.append(
            VectorLayer("points", arr[:, 0], arr[:, 1], (240, 190, 60), 3.0, False, "passes")
        )
    if "peaks" in overlays and plan["peaks"]:
        arr = np.array([[p["x_km"], p["y_km"]] for p in plan["peaks"]])
        out.append(VectorLayer("points", arr[:, 0], arr[:, 1], (255, 255, 255), 2.6, True, "peaks"))
    return out


register_view(
    ViewSpec(
        id="relief",
        title="Рельеф",
        stage="v3",
        layers=(
            "elevation_m",
            "slope_m_per_km",
            "mountain_class",
            "mountain_system_id",
            "rock_type",
            "rock_resistance",
            "orogen_age_ma",
            "sediment_thickness_m",
            "erosion_total_m",
        ),
        modes=(
            Option("shaded", "Итоговая высота с отмывкой"),
            Option("classes", "Классы: гребни, перевалы, долины, предгорья"),
            Option("systems", "Горные системы"),
            Option("styles", "Архитектура гор"),
            Option("slope", "Уклон"),
            Option("orogen_age", "Возраст горообразования"),
            Option("rock", "Породы"),
            Option("resistance", "Сопротивление эрозии"),
            Option("erosion", "Снято эрозией"),
            Option("sediment", "Толщина осадков"),
        ),
        overlays=(
            Option("coast", "Берег", True),
            Option("ridges", "Оси хребтов", True),
            Option("peaks", "Вершины"),
            Option("passes", "Перевалы"),
            Option("valleys", "Долины"),
        ),
        note="Рельеф эпохи v3: системы гор разной архитектуры; климатическая эрозия — «Эрозия»",
        render=_render_relief,
        legend=_legend_relief,
        vectors=_vectors_relief,
    )
)


# -- Mountain massifs and basins (v4.6) -----------------------------------------------------
def _final_relief_rgb(project: Project, k_shade: float = 0.75) -> np.ndarray:
    L = project.hydro.layers
    z = L["elevation_m"].astype(np.float64)
    ocean, lake = water_masks(project)
    shade = hillshade(z, project.layout)
    rgb = hypsometric(z - project.hydro.sea_level_m, ocean, lake)
    return (rgb * (0.45 + k_shade * shade[..., None])).clip(0, 255).astype(np.uint8)


def _render_massifs(project: Project, mode: str, overlays: frozenset) -> np.ndarray:
    T = project.terrain.layers
    H = project.hist.layers
    L = project.hydro.layers
    layout = project.layout
    inside = H["mountain_system_id"] >= 0
    if mode == "structure":
        rgb = _final_relief_rgb(project)
    elif mode == "dissection":
        rgb = _ramp_rgb(T["dissection_index"], 0.0, 1.0, 0.8)
        rgb[~inside] = (215, 210, 200)
    elif mode == "prominence":
        rgb = _final_relief_rgb(project, 0.5)
        _shade(rgb, ~inside, 0.55, (255, 255, 255))
    elif mode == "slope":
        rgb = _ramp_rgb(L["slope_m_per_km"], 0.0, 40.0, 0.6)
        rgb[~inside] = (215, 210, 200)
    elif mode == "valleys":
        rgb = _final_relief_rgb(project, 0.6)
        _shade(rgb, ~inside, 0.5, (255, 255, 255))
    else:
        raise KeyError(f"Unknown mode {mode!r} of view 'massifs'")
    if "coast" in overlays:
        _shade(rgb, mask_rim(~L["ocean_mask"], layout), 0.4)
    if "outlines" in overlays:
        rgb[id_rim(H["mountain_system_id"], layout) & inside] = (90, 30, 90)
    return rgb


def _vectors_massifs(project: Project, mode: str, overlays: frozenset) -> list[VectorLayer]:
    out = _vectors_relief(project, mode, overlays - {"peaks"})
    massifs = project.terrain.plans.get("massifs", [])
    if "peaks" in overlays and massifs:
        pts = [p for m in massifs for p in m["peaks"] if p["prominence_m"] >= 300.0]
        for lo, size, color in (
            (300.0, 2.4, (255, 255, 255)),
            (1000.0, 4.0, (255, 235, 120)),
            (2500.0, 6.0, (255, 120, 60)),
        ):
            sel = [p for p in pts if p["prominence_m"] >= lo]
            if sel:
                arr = np.array([[p["x_km"], p["y_km"]] for p in sel])
                out.append(
                    VectorLayer(
                        "points", arr[:, 0], arr[:, 1], color, size, True, f"prominence_{int(lo)}"
                    )
                )
    if ("rivers" in overlays or mode == "valleys") and project.hydro is not None:
        ids = project.hist.layers["mountain_system_id"]
        xs, ys = [], []
        for seg in project.hydro.rivers:
            if len(seg["x_km"]) < 2:
                continue
            col, row = project.layout.world_to_offset(seg["x_km"][0], seg["y_km"][0])
            col, row = int(col), int(row)
            if project.layout.inside(col, row) and ids[row, col] >= 0:
                xs.append(np.asarray(seg["x_km"]))
                ys.append(np.asarray(seg["y_km"]))
        if xs:
            out.append(VectorLayer("lines", xs, ys, (40, 110, 220), 1.6, True, "massif_valleys"))
    return out


def _legend_massifs(project: Project, mode: str) -> list[LegendItem]:
    s = project.terrain.summary.get("massifs", {})
    head = [
        LegendItem(
            f"массивов в окне: {s.get('count', 0)}; перевалов ниже своих вершин: "
            f"{s.get('passes_below_peaks', 0)} из {s.get('passes_total', 0)}"
        ),
        LegendItem(f"гексов, где река идёт по гребню: {s.get('rivers_on_crest_hexes', 0)}"),
    ]
    lo, hi = tuple(render.ramp(np.array(0.0))), tuple(render.ramp(np.array(1.0)))
    if mode == "structure":
        return head + [
            LegendItem("главный хребет", (30, 18, 12)),
            LegendItem("ветвь", (70, 44, 30)),
            LegendItem("отрог", (120, 84, 60)),
            LegendItem("перевал", (240, 190, 60)),
            LegendItem("вершины по относительной высоте: ≥ 300 м", (255, 255, 255)),
            LegendItem("≥ 1000 м", (255, 235, 120)),
            LegendItem("≥ 2500 м", (255, 120, 60)),
        ]
    if mode == "dissection":
        return head + [
            LegendItem("0: слабо расчленено", lo),
            LegendItem("1: сильно расчленено", hi),
            LegendItem("эвристика [А]: площадь водосбора × возраст × податливость породы × сток"),
        ]
    if mode == "slope":
        return head + [LegendItem("0 м/км", lo), LegendItem("40 м/км и круче", hi)]
    if mode == "prominence":
        return head + [LegendItem("точки — вершины с относительной высотой ≥ 300 м")]
    return head + [LegendItem("синим — речные долины v4 внутри массивов", (40, 110, 220))]


_BASIN_RGB = {
    "intracratonic_sag": (230, 190, 90),
    "foreland_foredeep": (120, 170, 220),
    "rift_graben": (220, 90, 70),
    "strike_slip_pull_apart": (170, 60, 170),
}
_BASIN_TITLES = {
    "intracratonic_sag": "внутриплатформенная впадина (прогиб платформы)",
    "foreland_foredeep": "предгорный прогиб (изгиб плиты под горами)",
    "rift_graben": "рифтовый грабен (полуграбены, плечи)",
    "strike_slip_pull_apart": "бассейн растяжения на сдвиге (pull-apart)",
}


def _render_basins(project: Project, mode: str, overlays: frozenset) -> np.ndarray:
    T = project.terrain.layers
    L = project.hydro.layers
    layout = project.layout
    bid = T["basin_id"]
    plans = project.terrain.plans.get("basins", [])
    if mode == "types":
        rgb = _final_relief_rgb(project, 0.6).astype(np.float64)
        for i, p in enumerate(plans):
            m = bid == i
            if m.any():
                rgb[m] = 0.45 * rgb[m] + 0.55 * np.array(_BASIN_RGB[p["basin_type"]])
        rgb = rgb.clip(0, 255).astype(np.uint8)
    elif mode == "subsidence":
        rgb = _ramp_rgb(T["basin_subsidence_m"], 0.0, 5000.0, 0.6)
        rgb[T["basin_subsidence_m"] <= 0] = (225, 220, 210)
    elif mode == "fill":
        rgb = _ramp_rgb(T["basin_fill_m"], 0.0, 4000.0, 0.6)
        rgb[T["basin_fill_m"] <= 0] = (225, 220, 210)
    elif mode == "surface":
        rgb = diverging(T["basin_delta_m"].astype(np.float64), 600.0)
    elif mode == "water":
        rgb = _water_rgb(project, L["elevation_m"].astype(np.float64) - project.hydro.sea_level_m)
    elif mode == "closed":
        z = L["elevation_m"].astype(np.float64)
        closed = np.where(L["ocean_mask"], 0.0, T["spill_level_m"] - z)
        rgb = _ramp_rgb(closed, 0.0, 400.0, 0.6)
        rgb[closed <= 0.5] = (225, 220, 210)
        rgb[L["ocean_mask"]] = (60, 90, 150)
    else:
        raise KeyError(f"Unknown mode {mode!r} of view 'basins'")
    if "outlines" in overlays:
        rgb[id_rim(bid, layout) & (bid >= 0)] = (40, 20, 40)
    if "coast" in overlays:
        _shade(rgb, mask_rim(~L["ocean_mask"], layout), 0.4)
    if "lakes" in overlays:
        rgb[L["inland_water_mask"]] = LAKE_RGB
    return rgb


def _vectors_basins(project: Project, mode: str, overlays: frozenset) -> list[VectorLayer]:
    if "rivers" not in overlays:
        return []
    return _river_vectors(project, 300.0)


def _legend_basins(project: Project, mode: str) -> list[LegendItem]:
    s = project.terrain.summary.get("basins", {})
    seen = s.get("in_window", {})
    w = project.terrain.summary.get("water", {})
    head = [
        LegendItem(
            "в окне: "
            + ", ".join(
                f"{_BASIN_TITLES.get(k, k).split(' (')[0]} {v}" for k, v in seen.items() if v
            )
        ),
        LegendItem(
            f"суши ниже уровня моря: {w.get('below_sea_level_dry_hexes', 0)} гексов; "
            f"озёр: {w.get('lake_hexes', 0)} гексов"
        ),
    ]
    lo, hi = tuple(render.ramp(np.array(0.0))), tuple(render.ramp(np.array(1.0)))
    if mode == "types":
        return head + [LegendItem(t, _BASIN_RGB[k]) for k, t in _BASIN_TITLES.items()]
    if mode == "subsidence":
        return head + [
            LegendItem("0 м", lo),
            LegendItem("5000 м опускания фундамента", hi),
            LegendItem("фундамент, не поверхность: поверхность = фундамент + осадки"),
        ]
    if mode == "fill":
        return head + [LegendItem("0 м", lo), LegendItem("4000 м осадков в бассейне", hi)]
    if mode == "surface":
        return head + [
            LegendItem("опускание поверхности", (25, 60, 150)),
            LegendItem("подъём (плечи рифтов, краевой вал)", (150, 30, 30)),
        ]
    if mode == "water":
        return head + [LegendItem(t, tuple(int(c) for c in rgb)) for t, rgb in WATER_CLASSES[:4]]
    return head + [
        LegendItem("0 м: сток есть", lo),
        LegendItem("400 м и больше до порога перелива", hi),
        LegendItem("замкнутые понижения: озеро или сухое дно — по балансу воды"),
    ]


register_view(
    ViewSpec(
        id="massifs",
        title="Горные массивы",
        stage="v46",
        layers=("dissection_index", "prominence_m", "mountain_system_id", "slope_m_per_km"),
        modes=(
            Option("structure", "Хребты, перевалы, вершины"),
            Option("valleys", "Долины (реки v4) внутри массивов"),
            Option("dissection", "Расчленённость"),
            Option("slope", "Уклоны"),
            Option("prominence", "Вершины по относительной высоте"),
        ),
        overlays=(
            Option("coast", "Берег", True),
            Option("ridges", "Оси хребтов", True),
            Option("peaks", "Вершины", True),
            Option("passes", "Перевалы"),
            Option("rivers", "Реки"),
            Option("outlines", "Контуры массивов"),
        ),
        note="Граф хребтов и долин массивов; вершины с относительной высотой; v4.6",
        render=_render_massifs,
        legend=_legend_massifs,
        vectors=_vectors_massifs,
    )
)
register_view(
    ViewSpec(
        id="basins",
        title="Низменности и бассейны",
        stage="v46",
        layers=("basin_id", "basin_subsidence_m", "basin_fill_m", "spill_level_m"),
        modes=(
            Option("types", "Типы бассейнов"),
            Option("water", "Вода и суша ниже уровня моря"),
            Option("closed", "Замкнутые понижения (глубина до порога)"),
            Option("subsidence", "Опускание фундамента"),
            Option("fill", "Осадочное заполнение"),
            Option("surface", "Изменение поверхности"),
        ),
        overlays=(
            Option("coast", "Берег", True),
            Option("outlines", "Контуры бассейнов", True),
            Option("lakes", "Озёра"),
            Option("rivers", "Реки"),
        ),
        note="Рифты, pull-apart, предгорные прогибы, прогибы платформ; v4.6",
        render=_render_basins,
        legend=_legend_basins,
        vectors=_vectors_basins,
    )
)

# -- Volcanism (v3.0) -----------------------------------------------------------
_VOLCANO_COLORS = {"arc": (230, 70, 50), "hotspot": (250, 170, 30)}


def _render_volcanism(project: Project, mode: str, overlays: frozenset) -> np.ndarray:
    h = project.hist
    layout = project.layout
    L = h.layers
    if mode == "potential":
        rgb = render.ramp(L["volcanic_potential"]).copy()
    elif mode == "centres":
        shade = hillshade(L["elevation_m"].astype(np.float64), layout)
        base = np.where(
            L["land_mask"][..., None], np.array([104, 110, 100]), np.array([26, 46, 86])
        ).astype(np.float64)
        rgb = (base * (0.6 + 0.5 * shade[..., None])).clip(0, 255).astype(np.uint8)
        # tint only the edifices that the chosen overlays show
        ids = L["volcano_id"]
        size = max((v["id"] for v in h.volcanoes), default=-1) + 2
        shown = np.zeros(size, dtype=bool)
        for v in h.volcanoes:
            wanted = "volcanoes" in overlays if v["active"] else "extinct" in overlays
            if wanted and ("submarine" in overlays or not v["submarine"]):
                shown[v["id"]] = True
        inside = (ids >= 0) & shown[np.maximum(ids, 0)]
        rgb[inside] = (rgb[inside] * 0.4 + np.array([214, 80, 60]) * 0.6).astype(np.uint8)
    elif mode == "age":
        rgb = _age_ramp(L["volcanic_age_ma"], max(1.0, h.duration_ma))
    else:
        raise KeyError(f"Unknown mode {mode!r} of view 'volcanism'")
    if "coast" in overlays:
        _shade(rgb, mask_rim(L["land_mask"], layout), 0.4)
    return rgb


def _volcano_groups(h):
    v = h.volcanoes
    return {
        "arc_active": [x for x in v if x["kind"] == "arc" and x["active"]],
        "arc_extinct": [x for x in v if x["kind"] == "arc" and not x["active"]],
        "hot_active": [x for x in v if x["kind"] == "hotspot" and x["active"]],
        "hot_extinct": [x for x in v if x["kind"] == "hotspot" and not x["active"]],
    }


def _legend_volcanism(project: Project, mode: str) -> list[LegendItem]:
    h = project.hist
    g = _volcano_groups(h)
    window = [x for x in h.volcanoes if x["in_window"]]
    active = [x for x in window if x["active"]]
    head = [
        LegendItem(f"снимок: t = {h.epoch_time_ma:g} млн лет"),
        LegendItem(
            "один знак = один вулканический центр (постройка или группа конусов), "
            "нарисован в масштабе"
        ),
        LegendItem(f"действующие дуговые: {len(g['arc_active'])}", _VOLCANO_COLORS["arc"]),
        LegendItem(
            f"действующие над горячими точками: {len(g['hot_active'])}", _VOLCANO_COLORS["hotspot"]
        ),
        LegendItem(
            f"потухшие (контур): дуговых {len(g['arc_extinct'])}, горячих точек "
            f"{len(g['hot_extinct'])}"
        ),
        LegendItem(
            f"в окне действующих {len(active)}, из них под водой "
            f"{sum(1 for x in active if x['submarine'])}"
        ),
        LegendItem("«действующий» = извергался в последние 1,5 млн лет"),
        LegendItem(f"горячих точек (неподвижны в мантии): {len(h.hotspots)}", (255, 255, 255)),
    ]
    lo, hi = tuple(render.ramp(np.array(0.0))), tuple(render.ramp(np.array(1.0)))
    if mode == "potential":
        return head + [
            LegendItem("магматизм маловероятен", lo),
            LegendItem("активная дуга, ось спрединга или горячая точка", hi),
            LegendItem("это потенциал, а не вероятность извержения"),
        ]
    if mode == "age":
        return head + [
            LegendItem("постройка возникла только что", hi),
            LegendItem(f"{h.duration_ma:g} млн лет назад", lo),
            LegendItem("вулканических построек нет", tuple(_AGE_NONE)),
        ]
    return head


def _vectors_volcanism(project: Project, mode: str, overlays: frozenset) -> list[VectorLayer]:
    h = project.hist
    out: list[VectorLayer] = []
    submarine_too = "submarine" in overlays
    if "chains" in overlays:
        xs, ys = [], []
        for spot in h.hotspots:
            chain = sorted(
                (v for v in h.volcanoes if v["hotspot_id"] == spot["id"]), key=lambda v: v["age_ma"]
            )
            if len(chain) >= 2:
                xs.append(np.array([spot["x_km"]] + [v["x_km"] for v in chain]))
                ys.append(np.array([spot["y_km"]] + [v["y_km"] for v in chain]))
        if xs:
            out.append(VectorLayer("lines", xs, ys, (255, 220, 120), 1.2, True, "chains"))
    for kind, color in _VOLCANO_COLORS.items():
        for active, wanted in ((False, "extinct" in overlays), (True, "volcanoes" in overlays)):
            if not wanted:
                continue
            chosen = [
                v
                for v in h.volcanoes
                if v["kind"] == kind
                and v["active"] == active
                and (submarine_too or not v["submarine"])
            ]
            if chosen:
                arr = np.array([[v["x_km"], v["y_km"]] for v in chosen])
                radius = float(np.median([v["radius_km"] for v in chosen]))
                out.append(
                    VectorLayer(
                        "points",
                        arr[:, 0],
                        arr[:, 1],
                        color,
                        2.0,
                        active,
                        f"{kind}_{'active' if active else 'extinct'}",
                        size_km=radius,
                    )
                )
    if "hotspots" in overlays and h.hotspots:
        arr = np.array([[s["x_km"], s["y_km"]] for s in h.hotspots])
        out.append(
            VectorLayer("points", arr[:, 0], arr[:, 1], (255, 255, 255), 6.0, False, "hotspots")
        )
    return out


register_view(
    ViewSpec(
        id="volcanism",
        title="Вулканизм",
        stage="v3",
        layers=("volcanic_potential", "volcano_id", "volcanic_age_ma"),
        modes=(
            Option("centres", "Вулканические центры"),
            Option("potential", "Вулканический потенциал"),
            Option("age", "Возраст построек"),
        ),
        overlays=(
            Option("volcanoes", "Действующие вулканы", True),
            Option("extinct", "Потухшие"),
            Option("submarine", "Подводные"),
            Option("hotspots", "Горячие точки", True),
            Option("chains", "Возрастные цепочки", True),
            Option("coast", "Берег", True),
        ),
        note="Дуги субдукции и внутриплитные горячие точки; потенциал ≠ вероятность извержения",
        render=_render_volcanism,
        legend=_legend_volcanism,
        vectors=_vectors_volcanism,
    )
)


# -- reserved for future stages: registered, never drawn as ready ---------------
# -- v4.0: climate, water, erosion ------------------------------------------------------
def _v4_head(project: Project) -> list[LegendItem]:
    h = project.hydro
    lat = h.latitude
    note = " (шкала широт сжата)" if lat.get("compressed") else ""
    return [
        LegendItem(
            f"источник: v4, поверхность после эрозии (эпоха t = {h.epoch_time_ma:g} "
            f"млн лет); центр области — {lat['center_deg']:g}° с. ш.{note}"
        )
    ]


def _ramp_rgb(values, lo, hi, power=1.0):
    t = np.clip((np.asarray(values, dtype=np.float64) - lo) / max(hi - lo, 1e-9), 0.0, 1.0)
    t = np.nan_to_num(t, nan=0.0)  # NoData: the caller paints it in its own colour
    return render.ramp(t**power).copy()


_TEMP_STOPS = np.array([-40, -20, -10, 0, 10, 20, 30], dtype=np.float64)
_TEMP_RGB = np.array(
    [
        [90, 40, 140],
        [60, 90, 200],
        [90, 170, 230],
        [235, 245, 250],
        [250, 220, 120],
        [235, 120, 50],
        [170, 20, 30],
    ],
    dtype=np.float64,
)
_PRECIP_STOPS = np.array([0, 100, 250, 500, 1000, 2000, 4000], dtype=np.float64)
_PRECIP_RGB = np.array(
    [
        [150, 80, 30],
        [215, 160, 80],
        [240, 225, 150],
        [190, 220, 140],
        [90, 180, 120],
        [40, 120, 170],
        [30, 50, 140],
    ],
    dtype=np.float64,
)
_ARIDITY_RGB = np.array(
    [[180, 70, 40], [230, 150, 70], [240, 215, 120], [190, 215, 130], [80, 160, 110]],
    dtype=np.uint8,
)
_ARIDITY_TITLES = (
    "гипераридная (P/PET < 0,05)",
    "аридная (0,05–0,2)",
    "полуаридная (0,2–0,5)",
    "сухая субгумидная (0,5–0,65)",
    "гумидная (> 0,65)",
)


def _stops(values, stops, colors):
    v = np.nan_to_num(np.asarray(values, dtype=np.float64), nan=float(stops[0]))
    out = np.empty(v.shape + (3,), dtype=np.uint8)
    for c in range(3):
        out[..., c] = np.interp(v, stops, colors[:, c]).astype(np.uint8)
    return out


def _sea_tint(project, rgb, land, strength=0.65):
    sea = ~land
    rgb[sea] = (rgb[sea] * (1 - strength) + np.array([24, 44, 84]) * strength).astype(np.uint8)


def _render_temperature(project: Project, mode: str, overlays: frozenset) -> np.ndarray:
    L = project.hydro.layers
    land = L["land_mask"]
    if mode == "mean":
        rgb = _stops(L["temperature_mean_c"], _TEMP_STOPS, _TEMP_RGB)
    elif mode in ("djf", "jja"):
        rgb = _stops(L[f"temperature_{mode}_c"], _TEMP_STOPS, _TEMP_RGB)
    elif mode == "seasonal":
        rgb = _ramp_rgb(2.0 * L["temperature_seasonal_c"], 0.0, 50.0)
    elif mode == "snow":
        rgb = _ramp_rgb(L["snow_fraction"], 0.0, 1.0)
    elif mode == "glacial":
        rgb = _ramp_rgb(L["glacial_potential"], 0.0, 1.0)
    else:
        raise KeyError(f"Unknown mode {mode!r} of view 'temperature'")
    if mode not in ("mean", "djf", "jja"):
        _sea_tint(project, rgb, land)
    if "isotherms" in overlays:
        field = L[f"temperature_{mode}_c"] if mode in ("djf", "jja") else L["temperature_mean_c"]
        band = np.floor(field / 10.0).astype(np.int32)
        _shade(rgb, id_rim(band, project.layout), 0.55)
    if "coast" in overlays:
        _shade(rgb, mask_rim(land, project.layout), 0.35)
    return rgb


def _legend_temperature(project: Project, mode: str) -> list[LegendItem]:
    head = _v4_head(project)
    s = project.hydro.summary
    if mode == "mean":
        return (
            head
            + [
                LegendItem(f"{int(t):+d} °C", tuple(int(c) for c in rgb))
                for t, rgb in zip(_TEMP_STOPS, _TEMP_RGB, strict=True)
            ]
            + [
                LegendItem(
                    f"средняя по суше {_num(s['land_mean_temperature_c'], '+.1f')} °C; градиент "
                    f"{project.config.climate.lapse_rate_c_per_km:g} °C/км"
                )
            ]
        )
    if mode in ("djf", "jja"):
        name = "декабрь–февраль" if mode == "djf" else "июнь–август"
        land = project.hydro.layers["land_mask"]
        field = project.hydro.layers[f"temperature_{mode}_c"]
        mean = float(field[land].mean()) if land.any() else None
        return (
            head
            + [LegendItem(f"средняя температура: {name}")]
            + [
                LegendItem(f"{int(t):+d} °C", tuple(int(c) for c in rgb))
                for t, rgb in zip(_TEMP_STOPS, _TEMP_RGB, strict=True)
            ]
            + [
                LegendItem(f"средняя по суше {_num(mean, '+.1f')} °C"),
                LegendItem("лето северного полушария — июнь–август, южного — декабрь–февраль"),
            ]
        )
    lo, hi = tuple(render.ramp(np.array(0.0))), tuple(render.ramp(np.array(1.0)))
    if mode == "seasonal":
        return head + [
            LegendItem("годовой размах 0 °C", lo),
            LegendItem("годовой размах 50 °C и больше", hi),
            LegendItem("размах растёт с широтой и удалённостью от моря"),
        ]
    if mode == "snow":
        return head + [LegendItem("снега нет", lo), LegendItem("все осадки — снег", hi)]
    return head + [
        LegendItem("льду не удержаться", lo),
        LegendItem("условия для многолетнего льда", hi),
        LegendItem("потенциал по климату; ледники не моделируются"),
    ]


def _num(value, spec: str) -> str:
    """A summary number, or a dash where it does not exist (e.g. no land in the window)."""
    return "—" if value is None else format(value, spec)


def _wind_vectors(project: Project, step_hex: int = 9) -> VectorLayer:
    L = project.hydro.layers
    layout = project.layout
    hx, hy = layout.centers()
    sl = (slice(step_hex // 2, None, step_hex), slice(step_hex // 2, None, step_hex))
    u, v = L["wind_u_ms"][sl], L["wind_v_ms"][sl]
    scale = 0.55 * step_hex * layout.spacing_km / 8.0  # 8 m/s -> half the arrow spacing
    xs, ys = [], []
    for x, y, a, b in zip(hx[sl].ravel(), hy[sl].ravel(), u.ravel(), v.ravel(), strict=True):
        ex, ey = x + a * scale, y + b * scale
        sp = max(np.hypot(a, b), 1e-6)
        dx, dy = a / sp, b / sp
        head = 0.3 * scale * min(sp, 8.0) / 8.0 + 1e-6
        xs.append(np.array([x, ex, ex - head * (dx - 0.5 * dy), ex, ex - head * (dx + 0.5 * dy)]))
        ys.append(np.array([y, ey, ey - head * (dy + 0.5 * dx), ey, ey - head * (dy - 0.5 * dx)]))
    return VectorLayer("lines", xs, ys, (255, 255, 255), 1.2, True, "wind")


_REGIME_RGB = np.array(
    [[70, 110, 200], [90, 170, 90], [210, 80, 60], [220, 160, 50], [150, 150, 150]],
    dtype=np.float64,
)
_REGIME_TITLES = (
    "декабрь–февраль",
    "март–май (весеннее половодье)",
    "июнь–август",
    "сентябрь–ноябрь",
    "стока нет",
)


def _render_precipitation(project: Project, mode: str, overlays: frozenset) -> np.ndarray:
    L = project.hydro.layers
    land = L["land_mask"]
    if mode == "precipitation":
        rgb = _stops(L["precipitation_mm_yr"], _PRECIP_STOPS, _PRECIP_RGB)
    elif mode == "aridity":
        from .stages.climate import aridity_class

        rgb = _ARIDITY_RGB[aridity_class(L["humidity_index"])].copy()
        _sea_tint(project, rgb, land, 0.85)
    elif mode == "runoff":
        rgb = _stops(L["runoff_mm_yr"], _PRECIP_STOPS, _PRECIP_RGB)
        _sea_tint(project, rgb, land, 0.85)
    elif mode in ("djf", "jja"):
        rgb = _stops(L[f"precipitation_{mode}_mm"], _PRECIP_STOPS / 4.0, _PRECIP_RGB)
    elif mode == "regime":
        peak = L["runoff_peak_season"]
        rgb = _REGIME_RGB[np.minimum(peak, 4)].copy()
        melt = np.clip(L["snowmelt_share"], 0.0, 1.0)[..., None]
        rgb = (rgb * (1.0 - 0.55 * melt) + np.array([245, 245, 255]) * 0.55 * melt).astype(np.uint8)
        _sea_tint(project, rgb, land, 0.85)
    elif mode == "pet":
        rgb = _ramp_rgb(L["potential_evapotranspiration_mm_yr"], 0.0, 2000.0)
    else:
        raise KeyError(f"Unknown mode {mode!r} of view 'precipitation'")
    if "coast" in overlays:
        _shade(rgb, mask_rim(land, project.layout), 0.35)
    return rgb


def _legend_precipitation(project: Project, mode: str) -> list[LegendItem]:
    head = _v4_head(project)
    s = project.hydro.summary
    if mode in ("precipitation", "runoff"):
        what = "осадки" if mode == "precipitation" else "сток (осадки − испарение − потери)"
        items = [LegendItem(f"{what}, мм/год")] + [
            LegendItem(f"{int(v)}", tuple(int(c) for c in rgb))
            for v, rgb in zip(_PRECIP_STOPS, _PRECIP_RGB, strict=True)
        ]
        extra = (
            f"средние по суше {_num(s['land_mean_precipitation_mm_yr'], '.0f')} мм/год"
            if mode == "precipitation"
            else f"доля стока от осадков {s['runoff_ratio']:.2f}"
        )
        return (
            head
            + items
            + [LegendItem(extra), LegendItem("стрелки — преобладающий ветер, длина ~ скорость")]
        )
    if mode in ("djf", "jja"):
        name = "декабрь–февраль" if mode == "djf" else "июнь–август"
        return (
            head
            + [LegendItem(f"осадки за {name}, мм за сезон")]
            + [
                LegendItem(f"{int(v / 4)}", tuple(int(c) for c in rgb))
                for v, rgb in zip(_PRECIP_STOPS, _PRECIP_RGB, strict=True)
            ]
            + [
                LegendItem("пояса дождей и муссоны смещаются к летнему полушарию"),
                LegendItem("стрелки — средний за год ветер"),
            ]
        )
    if mode == "regime":
        wb = s.get("water_balance") or {}
        return (
            head
            + [LegendItem("сезон наибольшего стока")]
            + [
                LegendItem(t, tuple(int(c) for c in rgb))
                for t, rgb in zip(_REGIME_TITLES, _REGIME_RGB, strict=True)
            ]
            + [
                LegendItem("светлее — больше талых вод (снег, ледники)"),
                LegendItem(
                    f"баланс суши: P {_num(wb.get('precipitation_mm_yr'), '.0f')} = "
                    f"ET {_num(wb.get('evapotranspiration_mm_yr'), '.0f')} + сток "
                    f"{_num(wb.get('runoff_mm_yr'), '.0f')} + глубокие потери "
                    f"{_num(wb.get('deep_infiltration_mm_yr'), '.0f')} мм/год"
                ),
            ]
        )
    if mode == "aridity":
        sh = s["aridity_shares"]
        values = (sh["hyper_arid"], sh["arid"], sh["semi_arid"], sh["dry_subhumid"], sh["humid"])
        return head + [
            LegendItem(f"{t}: {100 * v:.0f} % суши", tuple(c))
            for t, v, c in zip(_ARIDITY_TITLES, values, _ARIDITY_RGB, strict=True)
        ]
    return head + [
        LegendItem("0 мм/год", tuple(render.ramp(np.array(0.0)))),
        LegendItem("2000 мм/год и больше", tuple(render.ramp(np.array(1.0)))),
        LegendItem("потенциальное испарение (не фактическое)"),
    ]


def _vectors_precipitation(project, mode, overlays):
    return [_wind_vectors(project)] if "wind" in overlays else []


_LAKE_RGB = np.array([70, 140, 215], dtype=np.uint8)
_CLOSED_LAKE_RGB = np.array([150, 190, 170], dtype=np.uint8)


def _render_hydrology(project: Project, mode: str, overlays: frozenset) -> np.ndarray:
    h = project.hydro
    L = h.layers
    layout = project.layout
    land = L["land_mask"]
    if mode == "rivers":
        shade = hillshade(L["elevation_m"].astype(np.float64), layout)
        base = np.where(land[..., None], np.array([200, 196, 178]), np.array([30, 52, 92])).astype(
            np.float64
        )
        rgb = (base * (0.55 + 0.55 * shade[..., None])).clip(0, 255).astype(np.uint8)
        dry = L["no_outflow_mask"] & land
        rgb[dry] = (rgb[dry] * 0.75 + np.array([230, 190, 120]) * 0.25).astype(np.uint8)
    elif mode == "discharge":
        q = L["discharge_m3_s"].astype(np.float64)
        rgb = _ramp_rgb(np.log10(np.maximum(q, 0.01)), -1.0, 4.5)
        _sea_tint(project, rgb, land, 0.9)
    elif mode == "basins":
        ids = L["basin_id"]
        colors = id_colors(max(1, int(ids.max()) + 1), 0.61, sat=(0.3, 0.55), val=(0.7, 0.92))
        rgb = np.empty(ids.shape + (3,), dtype=np.uint8)
        rgb[:] = (30, 52, 92)
        inside = ids >= 0
        rgb[inside] = colors[ids[inside]]
        endo = L["endorheic_mask"]
        rgb[endo] = (rgb[endo] * 0.5 + np.array([200, 120, 60]) * 0.5).astype(np.uint8)
    elif mode == "accumulation":
        rgb = _ramp_rgb(np.log10(np.maximum(L["flow_accumulation_area_km2"], 1.0)), 3.0, 7.0)
        _sea_tint(project, rgb, land, 0.9)
    else:
        raise KeyError(f"Unknown mode {mode!r} of view 'hydrology'")
    if "lakes" in overlays:
        lake = L["lake_id"] >= 0
        closed = lake & L["endorheic_mask"] & (L["outflow_share"] < 0.5)
        rgb[lake] = _LAKE_RGB
        rgb[closed] = _CLOSED_LAKE_RGB
    if "coast" in overlays:
        _shade(rgb, mask_rim(land, layout), 0.35)
    return rgb


def _river_vectors(project: Project, min_discharge: float = 0.0) -> list[VectorLayer]:
    out = []
    tiers = (
        (0.0, 1.0, (120, 175, 230)),
        (300.0, 1.8, (70, 140, 215)),
        (3000.0, 2.8, (35, 100, 200)),
        (20000.0, 4.0, (20, 60, 170)),
    )
    for i, (lo, width, color) in enumerate(tiers):
        hi = tiers[i + 1][0] if i + 1 < len(tiers) else np.inf
        xs, ys = [], []
        for seg in project.hydro.rivers:
            q = seg["discharge_m3_s"]
            if lo <= q < hi and q >= min_discharge and len(seg["x_km"]) >= 2:
                xs.append(np.asarray(seg["x_km"]))
                ys.append(np.asarray(seg["y_km"]))
        if xs:
            out.append(VectorLayer("lines", xs, ys, color, width, True, f"rivers_{int(lo)}"))
    return out


def _vectors_hydrology(project, mode, overlays):
    if "rivers" not in overlays:
        return []
    minimum = 0.0 if "small_rivers" in overlays else 200.0
    return _river_vectors(project, minimum)


def _legend_hydrology(project: Project, mode: str) -> list[LegendItem]:
    s = project.hydro.summary
    thr = project.config.hydrology.river_min_discharge_m3_s
    head = _v4_head(project) + [
        LegendItem(
            f"рек (систем) {s['rivers']}, наибольший порядок Стралера {s['max_strahler']}, "
            f"наибольший расход {s['max_discharge_m3_s']:,.0f} м³/с"
        ),
        LegendItem(
            f"река — расход от {thr:g} м³/с; толщина линии по расходу: "
            "< 300, < 3000, < 20 000, больше",
            (35, 100, 200),
        ),
        LegendItem(
            f"озёра: {s['lakes']} (бессточных {s['closed_lakes']}; ещё сухих котловин "
            f"{s.get('dry_closed_basins', 0)}), "
            f"{100 * s['lake_share_of_land']:.1f} % суши",
            tuple(_LAKE_RGB),
        ),
        LegendItem("бессточное (концевое) озеро", tuple(_CLOSED_LAKE_RGB)),
    ]
    if mode == "rivers":
        return head + [
            LegendItem(
                f"вода не доходит до моря: {100 * s['no_outflow_share_of_land']:.0f} % суши "
                f"(замкнутые бассейны {100 * s['endorheic_share_of_land']:.0f} %)",
                (230, 190, 120),
            )
        ]
    if mode == "discharge":
        return head + [
            LegendItem("0,1 м³/с", tuple(render.ramp(np.array(0.0)))),
            LegendItem("30 000 м³/с и больше", tuple(render.ramp(np.array(1.0)))),
        ]
    if mode == "basins":
        return head + [
            LegendItem("цвет = водосборный бассейн"),
            LegendItem("бассейн замкнутого озера", (200, 120, 60)),
        ]
    return head + [
        LegendItem("1000 км²", tuple(render.ramp(np.array(0.0)))),
        LegendItem("10 млн км²", tuple(render.ramp(np.array(1.0)))),
    ]


def _v4_uplift(project: Project) -> np.ndarray:
    """Tectonic uplift rate acting during the v4 erosion period, mm/yr (0 when switched off)."""
    rate = np.asarray(project.hist.layers["uplift_rate_mm_yr"], dtype=np.float64)
    return rate if project.config.erosion.continue_uplift else np.zeros_like(rate)


def _render_erosion(project: Project, mode: str, overlays: frozenset) -> np.ndarray:
    h = project.hydro
    L = h.layers
    layout = project.layout
    land = L["land_mask"]
    sea = h.sea_level_m
    if mode in ("after", "before"):
        z = L["elevation_m" if mode == "after" else "pre_erosion_elevation_m"].astype(np.float64)
        shade = hillshade(z, layout)
        ocean, lake = water_masks(project) if mode == "after" else (None, None)
        rgb = (hypsometric(z - sea, ocean, lake) * (0.45 + 0.75 * shade[..., None])).clip(0, 255)
        rgb = rgb.astype(np.uint8)
        if mode == "before":
            land = z > sea
    elif mode == "delta":
        rgb = diverging(L["v4_delta_elevation_m"].astype(np.float64), 800.0)
    elif mode == "erosion":
        rgb = _ramp_rgb(L["v4_erosion_m"], 0.0, 1500.0, 0.5)
    elif mode == "deposition":
        rgb = _ramp_rgb(L["v4_deposition_m"], 0.0, 500.0, 0.5)
    elif mode == "sediment":
        rgb = _ramp_rgb(L["sediment_thickness_m"], 0.0, 1500.0, 0.5)
    elif mode == "slope":
        rgb = _ramp_rgb(L["slope_m_per_km"], 0.0, 40.0, 0.6)
    elif mode == "uplift":
        rgb = diverging(_v4_uplift(project), 2.0)
    else:
        raise KeyError(f"Unknown mode {mode!r} of view 'erosion'")
    if "coast" in overlays:
        _shade(rgb, mask_rim(land, layout), 0.4)
    return rgb


def _legend_erosion(project: Project, mode: str) -> list[LegendItem]:
    h = project.hydro
    s = h.summary
    b = h.budget
    head = _v4_head(project) + [
        LegendItem(
            f"эрозия {s['erosion_duration_ma']:g} млн лет за {s['erosion_steps']} шагов; средний "
            f"размыв суши {s['mean_denudation_m_ma']:.0f} м/млн лет "
            f"({s['mean_denudation_m_ma'] / 1000:.3f} мм/год)"
        )
    ]
    if b:
        total = max(b.get("fluvial_eroded_m3", 0.0), 1.0)
        sea = 100 * (b["deposited_sea_m3"] + b["offshore_m3"]) / total
        land = 100 * (b["deposited_land_m3"] + b["deposited_lakes_m3"]) / total
        out = 100 * b["left_window_m3"] / total
        head.append(
            LegendItem(
                f"баланс наносов: в море {sea:.0f} %, на суше и в озёрах {land:.0f} %, "
                f"за край окна {out:.0f} %, невязка {b['closure_error_m3']:.0f} м³"
            )
        )
    lo, hi = tuple(render.ramp(np.array(0.0))), tuple(render.ramp(np.array(1.0)))
    if mode == "after":
        return head + [LegendItem("высота после эрозии (итоговая поверхность)")]
    if mode == "before":
        return head + [LegendItem("высота до эрозии (поверхность эпохи v3)")]
    if mode == "delta":
        return head + [
            LegendItem("выше на 800 м и больше", (150, 30, 30)),
            LegendItem("без изменений", (236, 236, 236)),
            LegendItem("ниже на 800 м и больше", (25, 60, 150)),
            LegendItem("с учётом продолжающегося поднятия"),
        ]
    if mode == "uplift":
        on = project.config.erosion.continue_uplift
        return head + [
            LegendItem("поднятие 2 мм/год и больше", (150, 30, 30)),
            LegendItem("покой", (236, 236, 236)),
            LegendItem("опускание 2 мм/год и больше", (25, 60, 150)),
            LegendItem(
                "скорость тектоники выбранной эпохи; действует во время эрозии"
                if on
                else "поднятие во время эрозии выключено (continue_uplift = false)"
            ),
        ]
    scales = {
        "erosion": "снято до 1500 м",
        "deposition": "отложено до 500 м",
        "sediment": "рыхлые осадки до 1500 м",
        "slope": "уклон до 40 м/км",
    }
    return head + [LegendItem("0", lo), LegendItem(scales[mode], hi)]


def _vectors_erosion(project, mode, overlays):
    return _river_vectors(project, 1000.0) if "rivers" in overlays else []


register_view(
    ViewSpec(
        id="temperature",
        title="Температура",
        stage="v4",
        layers=(
            "temperature_mean_c",
            "temperature_seasonal_c",
            "temperature_djf_c",
            "temperature_jja_c",
            "snow_fraction",
            "glacial_potential",
        ),
        modes=(
            Option("mean", "Среднегодовая температура"),
            Option("djf", "Декабрь–февраль"),
            Option("jja", "Июнь–август"),
            Option("seasonal", "Годовой размах"),
            Option("snow", "Доля снега в осадках"),
            Option("glacial", "Условия для льда"),
        ),
        overlays=(Option("coast", "Берег", True), Option("isotherms", "Изотермы через 10 °C")),
        note="Климат v4.0: параметрическая модель, не модель общей циркуляции",
        render=_render_temperature,
        legend=_legend_temperature,
    )
)
register_view(
    ViewSpec(
        id="precipitation",
        title="Осадки и влажность",
        stage="v4",
        layers=(
            "precipitation_mm_yr",
            "humidity_index",
            "runoff_mm_yr",
            "potential_evapotranspiration_mm_yr",
            "wind_u_ms",
            "wind_v_ms",
            "precipitation_djf_mm",
            "precipitation_jja_mm",
            "runoff_peak_season",
            "snowmelt_share",
        ),
        modes=(
            Option("precipitation", "Осадки"),
            Option("djf", "Осадки декабря–февраля"),
            Option("jja", "Осадки июня–августа"),
            Option("aridity", "Увлажнение (индекс аридности P/PET)"),
            Option("runoff", "Сток"),
            Option("regime", "Режим стока (сезон половодья)"),
            Option("pet", "Потенциальное испарение"),
        ),
        overlays=(Option("coast", "Берег", True), Option("wind", "Ветер", True)),
        note="Перенос влаги ветром: наветренные склоны влажные, за горами — дождевая тень",
        render=_render_precipitation,
        legend=_legend_precipitation,
        vectors=_vectors_precipitation,
    )
)
register_view(
    ViewSpec(
        id="hydrology",
        title="Реки и озёра",
        stage="v4",
        layers=(
            "river_mask",
            "discharge_m3_s",
            "lake_id",
            "basin_id",
            "flow_accumulation_area_km2",
        ),
        modes=(
            Option("rivers", "Реки и озёра"),
            Option("discharge", "Расход"),
            Option("basins", "Водосборные бассейны"),
            Option("accumulation", "Площадь водосбора"),
        ),
        overlays=(
            Option("rivers", "Реки", True),
            Option("small_rivers", "Малые реки"),
            Option("lakes", "Озёра", True),
            Option("coast", "Берег", True),
        ),
        note="Сток по шести соседям гекса; озёра по водному балансу; реки — линии в мировых км",
        render=_render_hydrology,
        legend=_legend_hydrology,
        vectors=_vectors_hydrology,
    )
)
register_view(
    ViewSpec(
        id="erosion",
        title="Эрозия",
        stage="v4",
        layers=(
            "elevation_m",
            "pre_erosion_elevation_m",
            "v4_delta_elevation_m",
            "v4_erosion_m",
            "v4_deposition_m",
            "sediment_thickness_m",
            "slope_m_per_km",
        ),
        modes=(
            Option("after", "Высота после эрозии"),
            Option("before", "Высота до эрозии"),
            Option("delta", "Разность"),
            Option("erosion", "Снято"),
            Option("deposition", "Отложено"),
            Option("sediment", "Толщина осадков"),
            Option("slope", "Уклон"),
            Option("uplift", "Тектоническое поднятие"),
        ),
        overlays=(Option("coast", "Берег", True), Option("rivers", "Крупные реки")),
        note="Речная (stream power) и склоновая эрозия, перенос и отложение наносов",
        render=_render_erosion,
        legend=_legend_erosion,
        vectors=_vectors_erosion,
    )
)

# -- ocean floor ------------------------------------------------------------------------
_AGE_STOPS = np.array([0, 10, 25, 50, 80, 120, 180], dtype=np.float64)
_AGE_RGB = np.array(
    [[200, 30, 30], [235, 120, 40], [240, 200, 60], [110, 190, 90], [60, 160, 190],
     [70, 90, 190], [120, 60, 150]],
    dtype=np.float64,
)  # fmt: skip


def _ocean_sea_level(project: Project) -> float:
    sea = project.current_sea_level_m
    return 0.0 if sea is None else float(sea)


def _sea_mask(project: Project, z, sea_level):
    ocean, _lake = water_masks(project)
    return (sea_level - z > 0.0) if ocean is None else ocean


def _render_ocean(project: Project, mode: str, overlays: frozenset) -> np.ndarray:
    from .geomorph import classify as v46
    from .stages import ocean

    layout = project.layout
    z = project.layer("elevation_m").astype(np.float64)
    sea_level = _ocean_sea_level(project)
    depth = sea_level - z
    sea = _sea_mask(project, z, sea_level)
    T = project.terrain.layers if getattr(project, "terrain", None) is not None else None
    shade = hillshade(z, layout)
    land_rgb = (np.array([168.0, 164.0, 150.0]) * (0.55 + 0.6 * shade[..., None])).clip(0, 255)
    if mode == "bathymetry":
        rgb = ocean.depth_tint(np.maximum(depth, 0.0)).astype(np.float64)
        rgb = rgb * (0.7 + 0.45 * shade[..., None])
    elif mode == "shaded":
        # continuous depth ramp + strong hillshade: the structure of the floor at a glance
        d = np.clip(depth / 7000.0, 0.0, 1.0)
        rgb = np.stack([np.interp(d, _SHADED_STOPS, _SHADED_RGB[:, c]) for c in range(3)], -1)
        rgb = rgb * (0.35 + 0.9 * hillshade(z, layout, 30.0)[..., None])
    elif mode == "provinces":
        if T is not None and "marine_province" in T:
            codes = T["marine_province"]
            rgb = np.array([v46.RGB[name] for name in v46.PROVINCES], dtype=np.float64)[codes]
        else:
            rgb = ocean.RGB[ocean.project_provinces(project)].astype(np.float64)
        rgb = rgb * (0.85 + 0.2 * shade[..., None])
    elif mode == "age":
        src = "oceanic_crust_age_ma" if T is not None else "crust_age_ma"
        age = np.nan_to_num(project.layer(src).astype(np.float64), nan=200.0)
        rgb = np.stack([np.interp(age, _AGE_STOPS, _AGE_RGB[:, c]) for c in range(3)], axis=-1)
        if T is not None:
            rgb[~T["marine_floor_mask"]] = (200.0, 200.0, 200.0)
        else:
            rgb[project.layer("continental_fraction") >= 0.5] = (200.0, 200.0, 200.0)
        if "isochrons" in overlays:
            band = np.floor(age / 10.0).astype(np.int32)
            rgb[id_rim(band, layout) & sea] *= 0.55
    elif mode == "margins" and T is not None:
        colors = np.array([[150, 150, 150], [90, 170, 120], [210, 80, 70], [230, 190, 60]])
        rgb = colors[T["margin_type"]].astype(np.float64)
        rgb[T["margin_type"] == 0] = ocean.depth_tint(np.maximum(depth, 0.0))[T["margin_type"] == 0]
        rgb = rgb * (0.75 + 0.35 * shade[..., None])
    elif mode == "sediment":
        src = "marine_sediment_thickness_m" if T is not None else "sediment_thickness_m"
        rgb = _ramp_rgb(project.layer(src), 0.0, 2000.0, 0.5).astype(np.float64)
    else:
        raise KeyError(f"Unknown mode {mode!r} of view 'ocean'")
    rgb = np.where(sea[..., None], rgb, land_rgb).clip(0, 255).astype(np.uint8)
    if "isobaths" in overlays:
        steps = np.asarray(ocean.DEPTH_STEPS)
        band = np.where(sea, np.searchsorted(steps, depth, side="right"), -1).astype(np.int32)
        rim = id_rim(band, layout) & sea
        _shade(rgb, rim, 0.6)
    if "coast" in overlays:
        _shade(rgb, mask_rim(~sea, layout), 0.35)
    if T is not None and "faults" in overlays:
        # major fracture zones only (>= 4 Ma of age jump): thin dark lineations, as on
        # bathymetric charts; minor offsets stay in the layer, not on the picture
        major = T["fracture_zone_mask"] & (T["fracture_zone_offset_ma"] >= 4.0) & sea
        _shade(rgb, major, 0.55, (20, 20, 30))
        rgb[T["trench_axis_mask"] & sea] = (0, 0, 0)
    if T is not None and "canyons" in overlays and "submarine_canyon_mask" in T:
        rgb[T["submarine_canyon_mask"] & sea] = (255, 255, 255)
        fan = (T["submarine_fan_m"] > 50.0) & sea
        _shade(rgb, fan, 0.55, (215, 180, 110))
    if "plate_boundaries" in overlays:
        kind = project.layer("boundary_class")
        rgb[(kind == 2) & sea] = (250, 250, 250)  # subduction: trench side
        rgb[(kind == 3) & sea] = (255, 210, 60)  # spreading
    return rgb


_SHADED_STOPS = np.array([0.0, 0.03, 0.15, 0.35, 0.55, 0.72, 0.86, 1.0])
_SHADED_RGB = np.array(
    [
        [200, 235, 250],
        [150, 210, 240],
        [100, 170, 225],
        [60, 125, 200],
        [40, 90, 175],
        [30, 60, 140],
        [18, 35, 95],
        [5, 10, 35],
    ],
    dtype=np.float64,
)


def _vectors_ocean(project: Project, mode: str, overlays: frozenset) -> list[VectorLayer]:
    T = getattr(project, "terrain", None)
    if T is None or "canyons" not in overlays:
        return []
    xs, ys = [], []
    for c in T.plans.get("canyons", []):
        if c.get("is_canyon") and len(c["path_km"]) >= 2:
            arr = np.asarray(c["path_km"])
            xs.append(arr[:, 0])
            ys.append(arr[:, 1])
    return [VectorLayer("lines", xs, ys, (255, 255, 255), 1.4, True, "canyons")] if xs else []


def _legend_ocean(project: Project, mode: str) -> list[LegendItem]:
    from .geomorph import classify as v46
    from .stages import ocean

    T = getattr(project, "terrain", None)
    head = [LegendItem(f"уровень моря {_ocean_sea_level(project):+.0f} м; поверхность — итоговая")]
    if mode == "bathymetry":
        names = (
            "0–200 м: шельф, мелководья",
            "200–1000 м: бровка и верх склона",
            "1000–2000 м: материковый склон",
            "2000–3000 м: склон, гребни хребтов",
            "3000–4000 м: подножие, склоны хребтов",
            "4000–5000 м: абиссальные равнины",
            "5000–6000 м: глубокие котловины",
            "6000–7000 м: глубоководные желоба",
            "глубже 7000 м: дно желобов",
        )
        return head + [
            LegendItem(t, tuple(int(c) for c in rgb))
            for t, rgb in zip(names, ocean.DEPTH_RGB, strict=True)
        ]
    if mode == "shaded":
        return head + [
            LegendItem("0 м", tuple(int(c) for c in _SHADED_RGB[0])),
            LegendItem("3500 м", tuple(int(c) for c in _SHADED_RGB[4])),
            LegendItem("7000 м и глубже", tuple(int(c) for c in _SHADED_RGB[-1])),
            LegendItem("непрерывная шкала + отмывка: хребты, разломы, холмы, желоба"),
        ]
    if mode == "provinces" and T is not None and "marine_province" in T.layers:
        shares = T.summary.get("marine_province_shares", {})
        return (
            head
            + [
                LegendItem(
                    f"{v46.TITLES[name]}: {100 * shares.get(name, 0.0):.1f} %", v46.RGB[name]
                )
                for name in v46.PROVINCES
                if name not in ("land", "inland_water") and shares.get(name, 0.0) > 0
            ]
            + [LegendItem("по геологии + форме + соседству; доли — от площади моря")]
        )
    if mode == "provinces":
        codes = ocean.project_provinces(project)
        sea = codes > 0
        total = max(int(sea.sum()), 1)
        items = []
        for name in ocean.PROVINCES[1:]:
            share = float((codes == ocean.CODE[name]).sum()) / total
            items.append(
                LegendItem(
                    f"{ocean.TITLES[name]}: {100 * share:.1f} %",
                    tuple(int(c) for c in ocean.RGB[ocean.CODE[name]]),
                )
            )
        return head + items + [LegendItem("доли — от площади моря в окне")]
    if mode == "age":
        return (
            head
            + [
                LegendItem(f"{int(a)} млн лет", tuple(int(c) for c in rgb))
                for a, rgb in zip(_AGE_STOPS, _AGE_RGB, strict=True)
            ]
            + [
                LegendItem("молодая кора у осей спрединга, древняя — у окраин"),
                LegendItem("ступени возраста поперёк — зоны разломов (трансформы)"),
                LegendItem("серым — континентальная кора", (200, 200, 200)),
            ]
        )
    if mode == "margins":
        return head + [
            LegendItem("пассивная окраина: широкий склон и подножие", (90, 170, 120)),
            LegendItem("активная: узкий крутой склон к желобу, без подножия", (210, 80, 70)),
            LegendItem("трансформная: крутой уступ, короткое подножие", (230, 190, 60)),
        ]
    lo, hi = tuple(render.ramp(np.array(0.0))), tuple(render.ramp(np.array(1.0)))
    return head + [LegendItem("0 м", lo), LegendItem("2000 м осадков и больше", hi)]


register_view(
    ViewSpec(
        id="ocean",
        title="Подводный рельеф",
        stage="v3",
        layers=("elevation_m", "crust_age_ma", "continental_fraction", "boundary_class"),
        modes=(
            Option("bathymetry", "Глубины (ступени атласа)"),
            Option("shaded", "Рельеф дна с отмывкой"),
            Option("provinces", "Формы дна (геоморфология)"),
            Option("age", "Возраст океанической коры"),
            Option("margins", "Типы окраин"),
            Option("sediment", "Толщина осадков"),
        ),
        overlays=(
            Option("coast", "Берег", True),
            Option("isobaths", "Изобаты 200, 1000, 2000 … м", True),
            Option("faults", "Зоны разломов и оси желобов"),
            Option("canyons", "Каньоны и конусы выноса"),
            Option("isochrons", "Изохроны (10 млн лет)"),
            Option("plate_boundaries", "Оси спрединга и зоны субдукции"),
        ),
        note="Шельф, склон, подножие, абиссаль, хребты, разломы, желоба, каньоны, гайоты",
        render=_render_ocean,
        legend=_legend_ocean,
        vectors=_vectors_ocean,
    )
)

# -- coasts (v4) ----------------------------------------------------------------------------
_COAST_RGB = {
    "ice": (235, 245, 255),
    "fjord": (70, 90, 170),
    "skerry": (120, 140, 200),
    "volcanic": (150, 40, 40),
    "delta": (60, 150, 60),
    "estuary": (40, 170, 170),
    "liman": (110, 200, 190),
    "coral_reef": (255, 120, 170),
    "mangrove": (20, 110, 50),
    "sabkha": (240, 220, 150),
    "tidal_flat": (170, 140, 100),
    "fault_scarp": (110, 50, 120),
    "dalmatian": (200, 120, 40),
    "ria": (230, 160, 60),
    "cliff": (90, 80, 70),
    "barrier_lagoon": (250, 200, 80),
    "beach": (250, 235, 170),
}


def _coast_table():
    from .stages.coasts import COAST_TYPES

    table = np.zeros((len(COAST_TYPES), 3), dtype=np.uint8)
    for i, name in enumerate(COAST_TYPES):
        if name in _COAST_RGB:
            table[i] = _COAST_RGB[name]
    return table


def _render_coasts(project: Project, mode: str, overlays: frozenset) -> np.ndarray:
    L = project.hydro.layers
    layout = project.layout
    land = L["land_mask"]
    shade = hillshade(L["elevation_m"].astype(np.float64), layout)
    base = np.where(land[..., None], np.array([168, 172, 160]), np.array([34, 56, 96]))
    rgb = (base * (0.6 + 0.45 * shade[..., None])).clip(0, 255).astype(np.uint8)
    coast = L["coast_type"] > 0
    if mode == "types":
        rgb[coast] = _coast_table()[L["coast_type"][coast]]
    else:
        field, lo, hi = {
            "exposure": ("wave_exposure", 0.0, 0.8),
            "tides": ("tidal_range_m", 0.0, 5.0),
            "shelf": ("shelf_width_km", 0.0, 500.0),
            "sst": ("sea_surface_temperature_c", -2.0, 30.0),
        }[mode]
        values = np.nan_to_num(L[field][coast].astype(np.float64), nan=lo)
        rgb[coast] = _ramp_rgb(values, lo, hi)
    if "lgm" in overlays:
        ice = L["lgm_ice_mask"] & land
        rgb[ice] = (rgb[ice] * 0.6 + np.array([255, 255, 255]) * 0.4).astype(np.uint8)
    return rgb


def _sea_direction(project: Project):
    """Unit vector from each coast hex towards its sea neighbours (world km)."""
    from .stages import raster as raster_stage

    L = project.hydro.layers
    layout = project.layout
    hx, hy = layout.centers()
    sea = ~L["land_mask"] & (L["lake_id"] < 0)
    dx = np.zeros(hx.shape)
    dy = np.zeros(hx.shape)
    for d in range(6):
        nb, valid = raster_stage.neighbor_index(layout, d)
        wet = valid & sea.ravel()[nb.ravel()].reshape(hx.shape)
        dx += np.where(wet, hx.ravel()[nb.ravel()].reshape(hx.shape) - hx, 0.0)
        dy += np.where(wet, hy.ravel()[nb.ravel()].reshape(hx.shape) - hy, 0.0)
    norm = np.maximum(np.hypot(dx, dy), 1e-9)
    return hx, hy, dx / norm, dy / norm


def _vectors_coasts(project: Project, mode: str, overlays: frozenset) -> list[VectorLayer]:
    """Coast symbols: barrier islands, reefs, fjord inlets, delta fans (not to scale)."""
    if "symbols" not in overlays:
        return []
    from .stages.coasts import COAST_TYPES

    L = project.hydro.layers
    s = project.layout.spacing_km
    hx, hy, sx, sy = _sea_direction(project)
    ct = L["coast_type"]
    out = []

    def where(name):
        return np.nonzero(ct == COAST_TYPES.index(name))

    def lines(name, make, color, width):
        xs, ys = [], []
        for r, c in zip(*where(name), strict=True):
            for x, y in make(hx[r, c], hy[r, c], sx[r, c], sy[r, c]):
                xs.append(np.asarray(x))
                ys.append(np.asarray(y))
        if xs:
            out.append(VectorLayer("lines", xs, ys, color, width, True, name))

    def barrier(x, y, ux, uy):  # an island chain parallel to the shore, offshore
        px, py = -uy, ux
        ox, oy = x + ux * 0.62 * s, y + uy * 0.62 * s
        yield [ox - px * 0.45 * s, ox + px * 0.45 * s], [oy - py * 0.45 * s, oy + py * 0.45 * s]

    def reef(x, y, ux, uy):  # a reef line a little further out, broken into dashes
        px, py = -uy, ux
        ox, oy = x + ux * 0.8 * s, y + uy * 0.8 * s
        for a in (-0.45, -0.1, 0.25):
            b = a + 0.2
            yield [ox + px * a * s, ox + px * b * s], [oy + py * a * s, oy + py * b * s]

    def fjord(x, y, ux, uy):  # narrow inlets cutting into the land
        px, py = -uy, ux
        for a in (-0.3, 0.05, 0.35):
            x0, y0 = x + ux * 0.5 * s + px * a * s, y + uy * 0.5 * s + py * a * s
            yield [x0, x0 - ux * 0.75 * s], [y0, y0 - uy * 0.75 * s]

    def delta(x, y, ux, uy):  # distributaries fanning out into the sea
        for a in (-0.5, 0.0, 0.5):
            ca, sa = np.cos(a), np.sin(a)
            vx, vy = ux * ca - uy * sa, ux * sa + uy * ca
            yield [x, x + vx * 0.85 * s], [y, y + vy * 0.85 * s]

    lines("barrier_lagoon", barrier, (250, 215, 110), 2.0)
    lines("coral_reef", reef, (255, 120, 170), 2.0)
    lines("fjord", fjord, (30, 50, 120), 1.5)
    lines("delta", delta, (40, 120, 200), 1.5)
    return out


def _legend_coasts(project: Project, mode: str) -> list[LegendItem]:
    from .stages.coasts import COAST_TITLES

    c = project.hydro.summary.get("coasts", {})
    head = _v4_head(project) + [
        LegendItem(
            f"берег: {c.get('coast_hexes', 0)} гексов (~{c.get('coast_length_km', 0):,.0f} км "
            f"по сетке); скалистых {100 * c.get('rocky_share', 0):.0f} %, песчаных "
            f"{100 * c.get('sandy_share', 0):.0f} %"
        )
    ]
    if mode == "types":
        shares = c.get("shares", {})
        items = sorted(_COAST_RGB, key=lambda k: -shares.get(k, 0.0))
        return head + [
            LegendItem(f"{COAST_TITLES[k]} — {100 * shares.get(k, 0.0):.1f} %", _COAST_RGB[k])
            for k in items
        ]
    lo, hi = tuple(render.ramp(np.array(0.0))), tuple(render.ramp(np.array(1.0)))
    labels = {
        "exposure": ("защищённый берег", "открытый штормовым волнам (оценка)"),
        "tides": ("прилив 0 м", "прилив 5 м и больше (оценка)"),
        "shelf": ("шельф 0 км", "шельф 500 км и шире"),
        "sst": ("вода −2 °C", "вода 30 °C"),
    }[mode]
    return head + [LegendItem(labels[0], lo), LegendItem(labels[1], hi)]


register_view(
    ViewSpec(
        id="coasts",
        title="Берега",
        stage="v4",
        layers=("coast_type", "wave_exposure", "tidal_range_m", "shelf_width_km"),
        modes=(
            Option("types", "Типы берегов"),
            Option("exposure", "Открытость волнам"),
            Option("tides", "Приливы"),
            Option("shelf", "Ширина шельфа"),
            Option("sst", "Температура воды"),
        ),
        overlays=(
            Option("symbols", "Знаки: барьеры, рифы, фьорды, дельты", True),
            Option("lgm", "Оледенение последнего максимума"),
        ),
        note="Тип берега из причин: рельеф, порода, оледенение, волны, приливы, реки, климат",
        render=_render_coasts,
        legend=_legend_coasts,
        vectors=_vectors_coasts,
    )
)

# -- v4.5 ecological foundation --------------------------------------------------------------
_SEA_RGB = (34, 46, 66)
_NODATA_RGB = (150, 150, 150)
_PM_RGB = np.array(
    [[34, 46, 66], [196, 120, 110], [200, 170, 120], [150, 110, 150], [210, 70, 60],
     [150, 80, 60], [90, 170, 200], [150, 190, 220], [235, 210, 120], [225, 200, 160],
     [120, 170, 150], [170, 160, 150]],
    dtype=np.uint8,
)  # fmt: skip
_DRAIN_RGB = np.array(
    [[230, 200, 120], [160, 200, 110], [110, 175, 120], [90, 160, 170], [60, 110, 180],
     [40, 60, 140]],
    dtype=np.uint8,
)  # fmt: skip


def _month_of(overlays) -> int:
    for token in overlays:
        if isinstance(token, str) and token.startswith("month:"):
            return int(token.split(":", 1)[1]) % 12
    return 0


def _env_land(project):
    h = project.hydro.layers
    return h["land_mask"] & (h["lake_id"] < 0)


def _finish_env(project, rgb, valid, overlays):
    """Sea, lakes and missing data have their own colours; never a value of the scale."""
    land = _env_land(project)
    rgb = np.asarray(rgb, dtype=np.uint8).copy()
    rgb[~project.hydro.layers["land_mask"]] = _SEA_RGB
    rgb[project.hydro.layers["lake_id"] >= 0] = (60, 110, 170)
    rgb[land & ~valid] = _NODATA_RGB
    if "coast" in overlays:
        _shade(rgb, mask_rim(project.hydro.layers["land_mask"], project.layout), 0.4)
    if "rivers" in overlays:
        rgb[project.hydro.layers["river_mask"]] = (40, 90, 200)
    return rgb


_MONTHLY_MODES = {
    "temperature": ("temperature_monthly_c", "°C"),
    "precipitation": ("precip_monthly_mm", "мм/мес"),
    "pet": ("pet_monthly_mm", "мм/мес"),
    "snowpack": ("snowpack_monthly_swe_mm", "мм в.э."),
    "soil_water": ("soil_storage_monthly_mm", "мм"),
    "aet": ("aet_monthly_mm", "мм/мес"),
    "recharge": ("groundwater_recharge_mm", "мм/мес"),
    "runoff": ("runoff_monthly_mm", "мм/мес"),
}


def _render_seasonality(project: Project, mode: str, overlays: frozenset) -> np.ndarray:
    e = project.env
    month = _month_of(overlays)
    if mode == "temperature":
        v = e.monthly["temperature_monthly_c"][month]
        rgb = _stops(v, _TEMP_STOPS, _TEMP_RGB)
        return _finish_env(project, rgb, np.isfinite(v), overlays)
    if mode == "precipitation":
        v = e.monthly["precip_monthly_mm"][month]
        rgb = _stops(v, _PRECIP_STOPS / 12.0, _PRECIP_RGB)
    elif mode == "pet":
        v = e.monthly["pet_monthly_mm"][month]
        rgb = _ramp_rgb(v, 0.0, 200.0)
    elif mode == "snowpack":
        v = e.monthly["snowpack_monthly_swe_mm"][month]
        rgb = _ramp_rgb(v, 0.0, 400.0, 0.6)
    elif mode == "wettest":
        p = e.monthly["precip_monthly_mm"]
        v = np.argmax(p, axis=0).astype(np.float64)
        rgb = _MONTH_RGB[v.astype(int)]
    elif mode == "coldest":
        v = np.argmin(e.monthly["temperature_monthly_c"], axis=0).astype(np.float64)
        rgb = _MONTH_RGB[v.astype(int)]
    else:
        raise KeyError(f"Unknown mode {mode!r} of view 'seasonality'")
    return _finish_env(project, rgb, np.isfinite(v), overlays)


_MONTH_RGB = np.array(
    [[60, 90, 200], [80, 120, 220], [90, 170, 120], [120, 190, 90], [170, 200, 80],
     [230, 200, 60], [240, 150, 50], [230, 100, 60], [200, 120, 90], [160, 110, 150],
     [110, 100, 190], [70, 80, 180]],
    dtype=np.uint8,
)  # fmt: skip


def _legend_seasonality(project: Project, mode: str) -> list[LegendItem]:
    from .stages.environment import MONTH_NAMES

    head = [LegendItem("месяц выбирается в списке справа от режима")]
    lo, hi = tuple(render.ramp(np.array(0.0))), tuple(render.ramp(np.array(1.0)))
    src = project.env.methods.get("months", {})
    note = LegendItem(
        f"источник: {src.get('method', '—')} (уверенность {src.get('confidence', '—')})"
    )
    if mode == "temperature":
        return (
            head
            + [
                LegendItem(f"{int(t):+d} °C", tuple(int(c) for c in rgb))
                for t, rgb in zip(_TEMP_STOPS, _TEMP_RGB, strict=True)
            ]
            + [note]
        )
    if mode == "precipitation":
        return (
            head
            + [
                LegendItem(f"{v / 12.0:.0f} мм/мес", tuple(int(c) for c in rgb))
                for v, rgb in zip(_PRECIP_STOPS, _PRECIP_RGB, strict=True)
            ]
            + [note]
        )
    if mode in ("wettest", "coldest"):
        what = "самый влажный месяц" if mode == "wettest" else "самый холодный месяц"
        return [LegendItem(what)] + [
            LegendItem(name, tuple(int(c) for c in _MONTH_RGB[k]))
            for k, name in enumerate(MONTH_NAMES)
        ]
    scale = {"pet": "200 мм/мес", "snowpack": "400 мм водного эквивалента"}[mode]
    extra = (
        [LegendItem("эталонная испаряемость (эвристика по температуре и длине дня)")]
        if mode == "pet"
        else [LegendItem("запас снега на конец месяца")]
    )
    return head + [LegendItem("0", lo), LegendItem(scale, hi)] + extra


def _render_soils(project: Project, mode: str, overlays: frozenset) -> np.ndarray:
    L = project.env.layers
    present = L["soil_present"]
    if mode == "parent":
        rgb = _PM_RGB[np.minimum(L["parent_material_class"], len(_PM_RGB) - 1)]
        valid = _env_land(project)
    elif mode == "texture":
        # sand -> red, silt -> green, clay -> blue (the texture triangle as colour)
        tri = np.stack([L["sand_frac"], L["silt_frac"], L["clay_frac"]], axis=-1)
        rgb = (60.0 + 195.0 * np.nan_to_num(tri)).clip(0, 255)
        valid = present
    elif mode == "depth":
        rgb, valid = _ramp_rgb(L["soil_depth_m"], 0.0, 3.0), _env_land(project)
    elif mode == "coarse":
        rgb, valid = _ramp_rgb(L["coarse_fragment_frac"], 0.0, 0.8), present
    elif mode == "awc":
        rgb, valid = _ramp_rgb(L["awc_mm"], 0.0, 250.0), _env_land(project)
    elif mode == "permeability":
        rgb = _ramp_rgb(np.log10(np.maximum(np.nan_to_num(L["ks_mm_h"]), 0.1)), -1.0, 2.5)
        valid = present
    elif mode == "drainage":
        rgb = _DRAIN_RGB[np.minimum(L["drainage_class"], 5)]
        valid = _env_land(project)
    elif mode == "organic":
        rgb, valid = _ramp_rgb(L["soil_organic_potential"], 0.0, 1.0), present
    elif mode == "salinity":
        rgb = np.where(L["soil_salinity_proxy"][..., None] > 0, (235, 235, 210), (120, 140, 110))
        valid = _env_land(project)
    else:
        raise KeyError(f"Unknown mode {mode!r} of view 'soils'")
    return _finish_env(project, rgb, valid, overlays)


def _legend_soils(project: Project, mode: str) -> list[LegendItem]:
    from .stages.environment import DRAINAGE_TITLES, PARENT_MATERIALS, PARENT_TITLES

    s = project.env.summary
    tail = [
        LegendItem("серым — голая скала или лёд (почвы нет)", _NODATA_RGB),
        LegendItem("почвы — оценка по геологии и климату, не измерение"),
    ]
    if mode == "parent":
        sh = s.get("parent_material_shares", {})
        return [
            LegendItem(
                f"{PARENT_TITLES[n]}: {100 * sh.get(n, 0.0):.0f} %",
                tuple(int(c) for c in _PM_RGB[k]),
            )
            for k, n in enumerate(PARENT_MATERIALS)
            if k > 0
        ] + tail
    if mode == "texture":
        return [
            LegendItem("песок — красный, пыль (ил) — зелёный, глина — синий", None),
            LegendItem("песчаные", (240, 90, 90)),
            LegendItem("пылеватые (лёсс)", (90, 230, 100)),
            LegendItem("глинистые", (90, 100, 240)),
            LegendItem("суглинки — смешанные тона"),
        ] + tail
    if mode == "drainage":
        sh = s.get("drainage_class_shares", {})
        from .stages.environment import DRAINAGE_CLASSES

        return [
            LegendItem(
                f"дренаж {t}: {100 * sh.get(n, 0.0):.0f} %", tuple(int(c) for c in _DRAIN_RGB[k])
            )
            for k, (n, t) in enumerate(zip(DRAINAGE_CLASSES, DRAINAGE_TITLES, strict=True))
        ] + tail
    if mode == "salinity":
        return [
            LegendItem(
                "возможное засоление (сухие бессточные и приморские низины)", (235, 235, 210)
            ),
            LegendItem("нет признаков", (120, 140, 110)),
        ]
    lo, hi = tuple(render.ramp(np.array(0.0))), tuple(render.ramp(np.array(1.0)))
    scale = {
        "depth": ("0 м", "3 м и глубже"),
        "coarse": ("без камней", "80 % обломков"),
        "awc": ("0 мм", f"250 мм (среднее {_num(s.get('mean_awc_mm'), '.0f')} мм)"),
        "permeability": ("0,1 мм/ч (глина)", "300 мм/ч (песок)"),
        "organic": ("низкий потенциал органики", "высокий (оценка)"),
    }[mode]
    return [LegendItem(scale[0], lo), LegendItem(scale[1], hi)] + tail


def _render_moisture(project: Project, mode: str, overlays: frozenset) -> np.ndarray:
    e = project.env
    month = _month_of(overlays)
    L, M = e.layers, e.monthly
    if mode == "soil_water":
        cap = np.maximum(np.nan_to_num(L["awc_mm"]), 5.0)
        v = M["soil_storage_monthly_mm"][month] / cap
        rgb = _stops(np.nan_to_num(v), np.array([0.0, 0.25, 0.5, 0.75, 1.0]), _WET_RGB)
    elif mode == "stress":
        pet = np.maximum(M["pet_monthly_mm"][month], 1e-6)
        v = 1.0 - M["aet_monthly_mm"][month] / pet
        v = np.where(M["pet_monthly_mm"][month] < 5.0, 0.0, v)
        rgb = _stops(np.nan_to_num(v), np.array([0.0, 0.25, 0.5, 0.75, 1.0]), _WET_RGB[::-1])
    elif mode in ("aet", "recharge", "runoff"):
        v = M[_MONTHLY_MODES[mode][0]][month]
        rgb = _stops(np.nan_to_num(v), _PRECIP_STOPS / 12.0, _PRECIP_RGB)
    elif mode == "dry_months":
        v = L["dry_month_count"].astype(np.float64)
        rgb = _stops(v, np.array([0.0, 3.0, 6.0, 9.0, 12.0]), _WET_RGB[::-1])
    elif mode == "wetness":
        v = L["wetness_potential"]
        rgb = _stops(np.nan_to_num(v), np.array([0.0, 0.25, 0.5, 0.75, 1.0]), _WET_RGB)
    elif mode == "twi":
        v = L["twi_anomaly"]
        rgb = _ramp_rgb(np.nan_to_num(v), -2.0, 8.0)
    elif mode == "hand":
        v = L["hand_m"]
        rgb = _ramp_rgb(np.log10(1.0 + np.nan_to_num(v)), 0.0, 3.0)
    else:
        raise KeyError(f"Unknown mode {mode!r} of view 'moisture'")
    return _finish_env(project, rgb, np.isfinite(v) & _env_land(project), overlays)


_WET_RGB = np.array(
    [[170, 110, 60], [220, 190, 120], [200, 220, 170], [110, 180, 200], [40, 90, 170]],
    dtype=np.float64,
)


def _legend_moisture(project: Project, mode: str) -> list[LegendItem]:
    s = project.env.summary
    wb = s.get("water_balance", {})
    head = [
        LegendItem(
            f"баланс суши: P {_num(wb.get('precipitation_mm_yr'), '.0f')} = ET "
            f"{_num(wb.get('aet_mm_yr'), '.0f')} + сток {_num(wb.get('runoff_mm_yr'), '.0f')} + "
            f"глубокие потери {_num(wb.get('deep_loss_mm_yr'), '.0f')} + Δзапасов "
            f"{_num(wb.get('storage_change_mm_yr'), '.0f')} мм/год; невязка "
            f"{_num(wb.get('max_abs_residual_mm'), '.1e')} мм"
        )
    ]
    steps = ("0", "1/4", "1/2", "3/4", "1")
    if mode in ("soil_water", "wetness"):
        what = "запас влаги / AWC" if mode == "soil_water" else "потенциал переувлажнения"
        return (
            head
            + [LegendItem(what)]
            + [
                LegendItem(t, tuple(int(c) for c in rgb))
                for t, rgb in zip(steps, _WET_RGB, strict=True)
            ]
        )
    if mode == "stress":
        return (
            head
            + [LegendItem("водный стресс: 1 − AET/PET (месяц)")]
            + [
                LegendItem(t, tuple(int(c) for c in rgb))
                for t, rgb in zip(steps, _WET_RGB[::-1], strict=True)
            ]
        )
    if mode == "dry_months":
        return (
            head
            + [LegendItem("сухих месяцев (P < 0,5·PET)")]
            + [
                LegendItem(t, tuple(int(c) for c in rgb))
                for t, rgb in zip(("0", "3", "6", "9", "12"), _WET_RGB[::-1], strict=True)
            ]
        )
    lo, hi = tuple(render.ramp(np.array(0.0))), tuple(render.ramp(np.array(1.0)))
    if mode == "twi":
        return head + [
            LegendItem("TWI − TWI(гекс на уклоне 1 %): −2", lo),
            LegendItem("+8", hi),
            LegendItem("склонность к накоплению воды; не уровень грунтовых вод"),
        ]
    if mode == "hand":
        return head + [
            LegendItem("0 м над дреной", lo),
            LegendItem("1000 м и выше", hi),
            LegendItem("высота над ближайшей рекой/водоёмом вниз по стоку"),
        ]
    return (
        head
        + [
            LegendItem(f"{int(v / 12)} мм/мес", tuple(int(c) for c in rgb))
            for v, rgb in zip(_PRECIP_STOPS, _PRECIP_RGB, strict=True)
        ]
        + [LegendItem("местный сток — не расход реки; реки — из v4")]
    )


def _render_ecology(project: Project, mode: str, overlays: frozenset) -> np.ndarray:
    L = project.env.layers
    if mode == "gdd":
        v = L["gdd5_proxy_c_day"]
        rgb = _ramp_rgb(np.nan_to_num(v), 0.0, 6000.0)
    elif mode == "aridity":
        from .stages.climate import aridity_class

        v = L["aridity_p_over_pet"]
        rgb = _ARIDITY_RGB[aridity_class(np.nan_to_num(v))].copy()
    elif mode == "seasonality":
        v = L["precip_seasonality_cv"]
        rgb = _ramp_rgb(np.nan_to_num(v), 0.0, 120.0)
    elif mode == "cold":
        v = L["cold_month_count"].astype(np.float64)
        rgb = _ramp_rgb(v, 0.0, 12.0)
    elif mode == "growing":
        v = L["growing_month_count"].astype(np.float64)
        rgb = _ramp_rgb(v, 0.0, 12.0)
    elif mode == "flood":
        v = np.maximum(np.nan_to_num(L["floodplain_potential"]), L["waterlogging_months"] / 12.0)
        rgb = _stops(v, np.array([0.0, 0.25, 0.5, 0.75, 1.0]), _WET_RGB)
    elif mode == "frost":
        v = L["permafrost_potential"]
        rgb = _ramp_rgb(np.nan_to_num(v), 0.0, 1.0)
        rgb[L["permanent_ice_mask"]] = (245, 250, 255)
    elif mode == "rocky":
        v = L["terrain_rocky_fraction"]
        rgb = _ramp_rgb(np.nan_to_num(v), 0.0, 1.0)
    else:
        raise KeyError(f"Unknown mode {mode!r} of view 'ecology'")
    return _finish_env(project, rgb, np.isfinite(v) & _env_land(project), overlays)


def _legend_ecology(project: Project, mode: str) -> list[LegendItem]:
    lo, hi = tuple(render.ramp(np.array(0.0))), tuple(render.ramp(np.array(1.0)))
    s = project.env.summary
    if mode == "aridity":
        return [
            LegendItem(t, tuple(c)) for t, c in zip(_ARIDITY_TITLES, _ARIDITY_RGB, strict=True)
        ] + [LegendItem("индекс P/PET по 12 месяцам")]
    if mode == "flood":
        return [
            LegendItem("потенциал поймы / доля месяцев с переувлажнением"),
            *(
                LegendItem(t, tuple(int(c) for c in rgb))
                for t, rgb in zip(("0", "1/4", "1/2", "3/4", "1"), _WET_RGB, strict=True)
            ),
            LegendItem(
                f"поймы ~{100 * s.get('floodplain_area_share', 0):.1f} % площади суши; "
                f"переувлажнено ≥2 мес — {100 * s.get('waterlogged_share', 0):.0f} %"
            ),
            LegendItem("потенциал, не карта фактических паводков"),
        ]
    if mode == "frost":
        return [
            LegendItem("потенциал мерзлоты 0 (число мороза < 0,4)", lo),
            LegendItem("1 (число мороза ≥ 0,65)", hi),
            LegendItem("многолетний снег/лёд (накопление без стаивания)", (245, 250, 255)),
            LegendItem("оценка по воздуху (Nelson & Outcalt 1987), не мерзлота-факт"),
        ]
    scale = {
        "gdd": ("0 °C·сут", "6000 °C·сут (сумма T > 5 °C, оценка по месяцам)"),
        "seasonality": ("осадки ровные по году", "сезонность CV 120 %"),
        "cold": ("нет месяцев < 0 °C", "12 холодных месяцев"),
        "growing": ("0 месяцев роста", "12 (T ≥ 5 °C и AET ≥ 0,5·PET)"),
        "rocky": ("рыхлый покров", "голая скала"),
    }[mode]
    return [LegendItem(scale[0], lo), LegendItem(scale[1], hi)]


_ENV_OVERLAYS = (Option("coast", "Берег", True), Option("rivers", "Реки"))
register_view(
    ViewSpec(
        id="seasonality",
        title="Сезонность",
        stage="v45",
        layers=("temperature_monthly_c", "precip_monthly_mm", "pet_monthly_mm"),
        modes=(
            Option("temperature", "Температура месяца"),
            Option("precipitation", "Осадки месяца"),
            Option("pet", "Испаряемость (PET) месяца"),
            Option("snowpack", "Снежный запас на конец месяца"),
            Option("wettest", "Самый влажный месяц"),
            Option("coldest", "Самый холодный месяц"),
        ),
        overlays=_ENV_OVERLAYS,
        note="12 месяцев из сезонной физики v4; выбор месяца не пересчитывает климат",
        render=_render_seasonality,
        legend=_legend_seasonality,
        months=True,
    )
)
register_view(
    ViewSpec(
        id="soils",
        title="Почвы",
        stage="v45",
        layers=("parent_material_class", "sand_frac", "awc_mm", "drainage_class"),
        modes=(
            Option("parent", "Почвообразующая порода"),
            Option("texture", "Механический состав"),
            Option("depth", "Мощность почвы"),
            Option("coarse", "Каменистость"),
            Option("awc", "Доступная влага (AWC)"),
            Option("permeability", "Водопроницаемость"),
            Option("drainage", "Класс дренажа"),
            Option("organic", "Потенциал органики"),
            Option("salinity", "Признаки засоления"),
        ),
        overlays=_ENV_OVERLAYS,
        note="Почвы из геологии, рельефа и климата; не биомы и не плодородие",
        render=_render_soils,
        legend=_legend_soils,
    )
)
register_view(
    ViewSpec(
        id="moisture",
        title="Влага",
        stage="v45",
        layers=("soil_storage_monthly_mm", "aet_monthly_mm", "twi", "hand_m"),
        modes=(
            Option("soil_water", "Запас влаги в почве (месяц)"),
            Option("stress", "Водный стресс (месяц)"),
            Option("aet", "Фактическое испарение (месяц)"),
            Option("recharge", "Подпитка грунтовых вод (месяц)"),
            Option("runoff", "Местный сток (месяц)"),
            Option("dry_months", "Сухие месяцы за год"),
            Option("wetness", "Потенциал переувлажнения"),
            Option("twi", "Индекс влажности рельефа (TWI)"),
            Option("hand", "Высота над дреной (HAND)"),
        ),
        overlays=_ENV_OVERLAYS,
        note="Месячный водный баланс: вода не появляется и не исчезает",
        render=_render_moisture,
        legend=_legend_moisture,
        months=True,
    )
)
register_view(
    ViewSpec(
        id="ecology",
        title="Экологические параметры",
        stage="v45",
        layers=("gdd5_proxy_c_day", "aridity_p_over_pet", "permafrost_potential"),
        modes=(
            Option("gdd", "Сумма температур выше 5 °C"),
            Option("aridity", "Аридность P/PET"),
            Option("seasonality", "Сезонность осадков"),
            Option("cold", "Холодные месяцы"),
            Option("growing", "Месяцы вегетации"),
            Option("flood", "Поймы и переувлажнение (потенциал)"),
            Option("frost", "Мерзлота и многолетний снег (потенциал)"),
            Option("rocky", "Каменистость поверхности"),
        ),
        overlays=_ENV_OVERLAYS,
        note="Условия среды для будущих биомов; биомов здесь нет",
        render=_render_ecology,
        legend=_legend_ecology,
    )
)

for _id, _title in (
    ("biomes", "Биомы"),
    ("geosymbols", "Геосимволы"),
    ("seismicity", "Сейсмичность"),
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


def view_vectors(
    project: Project, view_id: str, mode: str | None = None, overlays=None
) -> list[VectorLayer]:
    spec = VIEWS[view_id]
    if spec.vectors is None or not spec.availability(project)[0]:
        return []
    chosen = spec.default_overlays() if overlays is None else frozenset(overlays)
    return spec.vectors(project, mode or spec.default_mode(), chosen)


def draw_vectors(
    img: np.ndarray, layers: list[VectorLayer], x0: float, y0: float, km_per_px: float
) -> None:
    """Rasterise vector overlays into an RGB image (used for PNG export)."""
    h, w = img.shape[:2]
    for layer in layers:
        if layer.kind == "lines":
            thickness = max(1, int(round(layer.size)))
            for xs, ys in zip(layer.x, layer.y, strict=True):
                px = (np.asarray(xs) - x0) / km_per_px
                py = (np.asarray(ys) - y0) / km_per_px
                for i in range(len(px) - 1):
                    if max(px[i], px[i + 1]) < 0 or min(px[i], px[i + 1]) > w:
                        continue
                    if max(py[i], py[i + 1]) < 0 or min(py[i], py[i + 1]) > h:
                        continue
                    render._draw_line(
                        img, px[i], py[i], px[i + 1], py[i + 1], layer.color, thickness
                    )
        else:
            px = np.rint((np.asarray(layer.x) - x0) / km_per_px).astype(np.int64)
            py = np.rint((np.asarray(layer.y) - y0) / km_per_px).astype(np.int64)
            r = max(1, int(round(max(layer.size, layer.size_km / km_per_px))))
            for cx, cy in zip(px, py, strict=True):
                if not (-r <= cx < w + r and -r <= cy < h + r):
                    continue
                ya, yb = max(0, cy - r), min(h, cy + r + 1)
                xa, xb = max(0, cx - r), min(w, cx + r + 1)
                if ya >= yb or xa >= xb:
                    continue
                yy, xx = np.ogrid[ya:yb, xa:xb]
                d2 = (yy - cy) ** 2 + (xx - cx) ** 2
                disc = d2 <= r * r
                ring = disc & (d2 >= (r - 1.2) ** 2)
                block = img[ya:yb, xa:xb]
                block[disc if layer.filled else ring] = layer.color
                block[ring & layer.filled] = (20, 20, 20)


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
    draw_vectors(img, view_vectors(project, view_id, mode, chosen), b.xmin, b.ymin, km_per_px)
    return img
