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

    def availability(self, project: Project | None) -> tuple[bool, str]:
        """Whether the view can be drawn for ``project``, and why not if it cannot."""
        if self.status != "ready":
            return False, "Запланировано"
        if project is None:
            return False, "Нет проекта"
        if self.stage in ("v2", "v3") and project.geo is None:
            return False, f"«{self.title}» ещё не сгенерированы для этого проекта"
        if self.stage == "v3" and project.hist is None:
            return False, f"«{self.title}»: геоистория ещё не сгенерирована для этого проекта"
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


def _land_of(project: Project, baseline: bool):
    """(land, landmass_id, count, shelf) of the selected v3 epoch or of the v2 baseline."""
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
    """(elevation, land, label) for the heights view: v3 epoch or the v2 baseline."""
    if project.hist is not None and mode != "baseline":
        h = project.hist
        label = f"итоговая высота v3, t = {h.epoch_time_ma:g} млн лет"
        return h.layers["elevation_m"], h.layers["land_mask"], label
    g = project.geo
    label = "исходная макровысота v2 (t = 0): макрорельеф без детализации и эрозии"
    return g.layers["macro_elevation_m"], g.layers["land_mask"], label


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
    if mode in ("hypsometric", "baseline"):
        rgb = hypsometric(rel)
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
    if mode in ("hypsometric", "baseline"):
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


def _render_relief(project: Project, mode: str, overlays: frozenset) -> np.ndarray:
    h = project.hist
    layout = project.layout
    L = h.layers
    elev = L["elevation_m"].astype(np.float64)
    if mode == "shaded":
        shade = hillshade(elev, layout)
        rgb = (hypsometric(elev - h.sea_level_m) * (0.45 + 0.75 * shade[..., None])).clip(0, 255)
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
            "горы: " + ("PMGE" if backend["is_pmge"] else "референсный backend (PMGE не подключён)")
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


def _polylines(items, key="points_km"):
    xs, ys = [], []
    for item in items:
        pts = np.asarray(item[key])
        if len(pts) >= 2:
            xs.append(pts[:, 0])
            ys.append(pts[:, 1])
    return xs, ys


def _vectors_relief(project: Project, mode: str, overlays: frozenset) -> list[VectorLayer]:
    plan = project.hist.plan
    out: list[VectorLayer] = []
    if "valleys" in overlays and plan["valleys"]:
        xs, ys = _polylines(plan["valleys"])
        out.append(VectorLayer("lines", xs, ys, (60, 150, 220), 1.2, True, "valleys"))
    if "ridges" in overlays:
        for rank, color, width in (
            ("spur", (120, 84, 60), 1.0),
            ("branch", (70, 44, 30), 1.6),
            ("main", (30, 18, 12), 2.4),
        ):
            xs, ys = _polylines([r for r in plan["ridges"] if r["rank"] == rank])
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
        note="Рельеф v3.0: структура гор + ограниченная эрозия; рек, ледников и климата нет",
        render=_render_relief,
        legend=_legend_relief,
        vectors=_vectors_relief,
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
        inside = L["volcano_id"] >= 0
        rgb[inside] = (rgb[inside] * 0.4 + np.array([214, 80, 60]) * 0.6).astype(np.uint8)
    elif mode == "age":
        rgb = _age_ramp(L["volcanic_age_ma"], max(1.0, h.duration_ma))
    else:
        raise KeyError(f"Unknown mode {mode!r} of view 'volcanism'")
    if "coast" in overlays:
        _shade(rgb, mask_rim(L["land_mask"], layout), 0.4)
    return rgb


def _legend_volcanism(project: Project, mode: str) -> list[LegendItem]:
    h = project.hist
    v = h.volcanoes
    arcs = [x for x in v if x["kind"] == "arc"]
    spots = [x for x in v if x["kind"] == "hotspot"]
    sub = sum(1 for x in v if x["submarine"])
    head = [
        LegendItem(f"снимок: t = {h.epoch_time_ma:g} млн лет"),
        LegendItem(
            f"дуговые вулканы (субдукция): {len(arcs)}, действующих "
            f"{sum(1 for x in arcs if x['active'])}",
            _VOLCANO_COLORS["arc"],
        ),
        LegendItem(
            f"вулканы горячих точек: {len(spots)}, действующих "
            f"{sum(1 for x in spots if x['active'])}",
            _VOLCANO_COLORS["hotspot"],
        ),
        LegendItem(f"горячих точек (неподвижны в мантии): {len(h.hotspots)}", (255, 255, 255)),
        LegendItem(f"подводных построек в окне: {sub}"),
        LegendItem("закрашенный знак - действующий, контур - потухший"),
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
            out.append(VectorLayer("lines", xs, ys, (255, 220, 120), 1.4, True, "chains"))
    if "volcanoes" in overlays:
        for kind, color in _VOLCANO_COLORS.items():
            for active in (False, True):
                pts = [
                    (v["x_km"], v["y_km"])
                    for v in h.volcanoes
                    if v["kind"] == kind and v["active"] == active
                ]
                if pts:
                    arr = np.array(pts)
                    out.append(
                        VectorLayer(
                            "points",
                            arr[:, 0],
                            arr[:, 1],
                            color,
                            3.4,
                            active,
                            f"{kind}_{'active' if active else 'extinct'}",
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
            Option("volcanoes", "Вулканы", True),
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
for _id, _title in (
    ("temperature", "Температура"),
    ("precipitation", "Осадки и влажность"),
    ("hydrology", "Гидрология и реки"),
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
            r = max(1, int(round(layer.size)))
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
