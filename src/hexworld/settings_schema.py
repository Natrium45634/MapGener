# ruff: noqa: E501  (the parameter table reads better one row per line)
"""World settings: every generator parameter with a label, a slider range and a random range.

The schema is the single description of "what can be set" (no Qt here): the settings
dialog, presets, randomisation and tests all use it.

* ``Param.lo / hi`` - the slider range, always inside what :class:`Config` accepts;
* ``Param.rand`` - the range a random world draws from (narrower: plausible worlds);
  ``None`` = never randomised (technical settings);
* a value is addressed by a ``path``: ``"climate.temperature_offset_c"``, a tuple element
  ``"terrain.age_depth_transition_ma[1]"`` or a set member ``"terrain.basins:rift_graben"``;
* ``"auto"`` parameters (plate count, landmass count, assembly) have ``auto=True``: the value
  ``None`` means "auto".

Randomisation is reproducible: :func:`randomize` takes a seed string; locked paths keep their
values.
"""

from __future__ import annotations

import random
import re
import string
from dataclasses import dataclass, fields, is_dataclass
from typing import Any

from .config import Config

GROUPS = (
    ("world", "Мир и размер"),
    ("tectonics", "Плиты и тектоника"),
    ("continents", "Континенты и острова"),
    ("history", "Горы, вулканы, история"),
    ("ocean", "Дно океана"),
    ("climate", "Климат"),
    ("water", "Реки и эрозия"),
    ("soils", "Почвы и вода"),
    ("topo", "Высотные пояса"),
    ("biomes", "Биомы и лёд"),
    ("technical", "Технические"),
)


@dataclass(frozen=True)
class Param:
    path: str
    label: str
    group: str
    kind: str  # float / int / bool / choice
    lo: float = 0.0
    hi: float = 1.0
    rand: tuple | None = None  # (lo, hi) for numbers; for choice/bool: True = any
    decimals: int = 2
    choices: tuple = ()  # ((value, label), ...)
    auto: bool = False
    log: bool = False  # slider in log scale
    unit: str = ""
    help: str = ""
    technical: bool = False  # not randomised, locked by default


def _p(path, label, group, kind, lo=0.0, hi=1.0, rand=None, dec=2, choices=(), auto=False,
       log=False, unit="", help="", tech=False):  # fmt: skip
    return Param(path, label, group, kind, lo, hi, rand, dec, choices, auto, log, unit, help, tech)


ON = True
PARAMS: tuple[Param, ...] = (
    # -- world and size ---------------------------------------------------------------------------
    _p(
        "width",
        "Ширина карты, гексов",
        "world",
        "int",
        20,
        3000,
        None,
        0,
        tech=True,
        help="Число гексов по горизонтали.",
    ),
    _p(
        "height",
        "Высота карты, гексов",
        "world",
        "int",
        20,
        3000,
        None,
        0,
        tech=True,
        help="Число гексов по вертикали.",
    ),
    _p(
        "hex_spacing_km",
        "Размер гекса",
        "world",
        "float",
        2.0,
        200.0,
        None,
        1,
        unit="км",
        log=True,
        tech=True,
        help="Расстояние между центрами соседних гексов.",
    ),
    _p(
        "mode",
        "Охват",
        "world",
        "choice",
        choices=(("regional", "регион (часть мира)"), ("world", "весь мир на карте")),
        tech=True,
        help="«Весь мир» — модель занимает всю карту; «регион» — окно в большой мир.",
    ),
    _p(
        "viewport_origin_km[0]",
        "Начало окна X",
        "world",
        "float",
        -50000,
        50000,
        None,
        0,
        unit="км",
        tech=True,
        help="Сдвиг окна по миру (только для режима «регион»).",
    ),
    _p(
        "viewport_origin_km[1]",
        "Начало окна Y",
        "world",
        "float",
        -50000,
        50000,
        None,
        0,
        unit="км",
        tech=True,
    ),
    _p(
        "climate.latitude_center_deg",
        "Широта центра карты",
        "world",
        "float",
        -80,
        80,
        (-45, 55),
        1,
        unit="°",
        help="Где на планете лежит карта: 0 — экватор, 60 — субполярные широты.",
    ),
    _p(
        "climate.km_per_degree",
        "Км на градус широты (0 = авто)",
        "world",
        "float",
        0,
        400,
        None,
        0,
        unit="км/°",
        tech=True,
        help="Масштаб широт: 0 — по размеру карты; 111 — как на Земле.",
    ),
    # -- tectonics --------------------------------------------------------------------------------
    _p(
        "plates.count",
        "Число литосферных плит",
        "tectonics",
        "int",
        2,
        80,
        (4, 28),
        0,
        auto=True,
        help="Сколько плит в мире; «авто» — по размеру мира.",
    ),
    _p(
        "plates.profile",
        "Профиль плит",
        "tectonics",
        "choice",
        rand=ON,
        choices=(
            ("few", "немного крупных"),
            ("earthlike", "как на Земле"),
            ("fragmented", "много мелких"),
        ),
        help="Распределение размеров плит.",
    ),
    _p(
        "plates.warp_strength",
        "Извилистость границ плит",
        "tectonics",
        "float",
        0.0,
        1.2,
        (0.4, 1.0),
        help="0 — прямые границы, 1 — сильно изрезанные.",
    ),
    _p(
        "plates.warp_wavelength_factor",
        "Масштаб изгибов границ",
        "tectonics",
        "float",
        0.2,
        3.0,
        (0.6, 1.4),
    ),
    _p(
        "plates.warp_octaves",
        "Детальность изгибов границ",
        "tectonics",
        "int",
        1,
        10,
        None,
        0,
        tech=True,
    ),
    _p(
        "motion.speed_scale",
        "Скорость плит",
        "tectonics",
        "float",
        0.1,
        4.0,
        (0.5, 2.0),
        help="Множитель скорости движения: быстрее — выше горы и шире океаны.",
    ),
    _p("motion.rotation_scale", "Вращение плит", "tectonics", "float", 0.0, 4.0, (0.4, 1.8)),
    _p(
        "boundaries.quiet_threshold_mm_yr",
        "Порог «спокойной» границы",
        "tectonics",
        "float",
        0.0,
        30.0,
        (1.5, 6.0),
        unit="мм/год",
    ),
    _p(
        "boundaries.obliquity_deg",
        "Угол косых границ",
        "tectonics",
        "float",
        0.0,
        89.0,
        (20, 45),
        0,
        unit="°",
    ),
    _p(
        "boundaries.reference_speed_mm_yr",
        "Опорная скорость границ",
        "tectonics",
        "float",
        10.0,
        400.0,
        (70, 140),
        0,
        unit="мм/год",
    ),
    _p(
        "fields.collision_width_km",
        "Ширина зоны столкновения",
        "tectonics",
        "float",
        50,
        1500,
        (250, 500),
        0,
        unit="км",
    ),
    _p(
        "fields.subduction_width_km",
        "Ширина зоны субдукции",
        "tectonics",
        "float",
        50,
        1500,
        (200, 400),
        0,
        unit="км",
    ),
    _p(
        "fields.trench_width_km",
        "Ширина жёлоба (тектоника)",
        "tectonics",
        "float",
        20,
        500,
        (60, 140),
        0,
        unit="км",
    ),
    _p(
        "fields.arc_offset_km",
        "Удаление вулканической дуги",
        "tectonics",
        "float",
        30,
        600,
        (120, 240),
        0,
        unit="км",
    ),
    _p(
        "fields.arc_width_km",
        "Ширина вулканической дуги",
        "tectonics",
        "float",
        20,
        500,
        (60, 140),
        0,
        unit="км",
    ),
    _p(
        "fields.rift_width_km",
        "Ширина рифта",
        "tectonics",
        "float",
        20,
        800,
        (110, 240),
        0,
        unit="км",
    ),
    _p(
        "fields.ridge_volcanism_width_km",
        "Ширина вулканизма хребтов",
        "tectonics",
        "float",
        10,
        400,
        (50, 100),
        0,
        unit="км",
    ),
    _p(
        "fields.transform_width_km",
        "Ширина трансформной зоны",
        "tectonics",
        "float",
        20,
        500,
        (80, 160),
        0,
        unit="км",
    ),
    _p(
        "fields.max_influence_km",
        "Радиус влияния границ",
        "tectonics",
        "float",
        100,
        4000,
        None,
        0,
        unit="км",
        tech=True,
    ),
    _p(
        "fields.max_pad_hexes",
        "Запас за краем окна",
        "tectonics",
        "int",
        0,
        512,
        None,
        0,
        tech=True,
    ),
    # -- continents and islands -------------------------------------------------------------------
    _p(
        "sea.target_land_fraction",
        "Доля суши",
        "continents",
        "float",
        0.02,
        0.95,
        (0.15, 0.5),
        help="Сколько поверхности над морем. Земля — 0,29.",
    ),
    _p(
        "crust.continental_fraction",
        "Доля континентальной коры",
        "continents",
        "float",
        0.05,
        0.95,
        (0.25, 0.5),
        help="Сколько коры континентальной (суша + шельфы).",
    ),
    _p(
        "crust.mean_continent_area_km2",
        "Средний размер континента",
        "continents",
        "float",
        2e5,
        1e8,
        (3e6, 3e7),
        0,
        log=True,
        unit="км²",
        help="Меньше — много мелких материков, больше — несколько огромных.",
    ),
    _p(
        "continents.major_landmass_count",
        "Число крупных материков",
        "continents",
        "int",
        1,
        30,
        (1, 9),
        0,
        auto=True,
        help="Цель по числу больших массивов суши; «авто» — из сида.",
    ),
    _p(
        "continents.continent_size_variety",
        "Разнообразие размеров",
        "continents",
        "float",
        0.0,
        1.0,
        (0.2, 0.9),
        help="0 — материки одного размера, 1 — от гигантов до малышей.",
    ),
    _p(
        "continents.coast_irregularity",
        "Изрезанность берегов",
        "continents",
        "float",
        0.0,
        1.5,
        (0.3, 1.2),
        help="Заливы, полуострова, фьорды.",
    ),
    _p(
        "continents.continent_profile",
        "Расположение материков",
        "continents",
        "choice",
        rand=ON,
        choices=(
            ("balanced", "сбалансированно"),
            ("supercontinent", "суперконтинент"),
            ("fragmented", "раздробленно"),
        ),
    ),
    _p(
        "crust.layout",
        "Группировка коры",
        "continents",
        "choice",
        rand=ON,
        choices=(("clustered", "скоплениями"), ("uniform", "равномерно")),
    ),
    _p(
        "crust.assembly",
        "Собранность в суперконтинент",
        "continents",
        "float",
        0.0,
        1.0,
        (0.0, 1.0),
        auto=True,
        help="0 — материки разбросаны, 1 — собраны в один.",
    ),
    _p(
        "terrain.island_amount",
        "Количество островов",
        "continents",
        "float",
        0.0,
        3.0,
        (0.3, 2.5),
        help="0 — почти без островов, 1 — как на Земле, 3 — мир архипелагов (дуги, шельфовые и вулканические острова).",
    ),
    _p(
        "history.hotspot_density",
        "Горячие точки (вулканические острова)",
        "continents",
        "float",
        0.0,
        5.0,
        (0.3, 2.5),
        help="Больше — больше цепочек вулканов и островов типа Гавайев.",
    ),
    _p("crust.roughness", "Неровность края коры", "continents", "float", 0.0, 1.0, (0.2, 0.7)),
    _p(
        "crust.warp_strength", "Извилистость материков", "continents", "float", 0.0, 1.5, (0.5, 1.1)
    ),
    _p(
        "crust.warp_wavelength_km",
        "Масштаб извилин материков",
        "continents",
        "float",
        500,
        12000,
        (2500, 6000),
        0,
        unit="км",
    ),
    _p("crust.warp_octaves", "Детальность извилин", "continents", "int", 1, 10, None, 0, tech=True),
    _p(
        "continents.major_area_km2",
        "С какой площади массив «крупный»",
        "continents",
        "float",
        1e5,
        2e7,
        None,
        0,
        log=True,
        unit="км²",
        tech=True,
    ),
    _p(
        "continents.transition_km",
        "Ширина перехода суша–океан",
        "continents",
        "float",
        10,
        400,
        (60, 140),
        0,
        unit="км",
    ),
    _p(
        "relief.continental_interior_m",
        "Высота внутренних равнин",
        "continents",
        "float",
        -200,
        1500,
        (100, 600),
        0,
        unit="м",
    ),
    _p(
        "relief.shelf_edge_m",
        "Глубина края шельфа",
        "continents",
        "float",
        -1500,
        -50,
        (-700, -250),
        0,
        unit="м",
    ),
    _p(
        "relief.shelf_scale_km",
        "Ширина шельфа",
        "continents",
        "float",
        10,
        600,
        (80, 260),
        0,
        unit="км",
    ),
    _p(
        "sea.sea_mode",
        "Как задать уровень моря",
        "continents",
        "choice",
        choices=(
            ("target_land_fraction", "по доле суши"),
            ("fixed_sea_level_m", "фиксированная высота"),
        ),
        tech=True,
    ),
    _p(
        "sea.fixed_sea_level_m",
        "Уровень моря (если фиксирован)",
        "continents",
        "float",
        -3000,
        3000,
        None,
        0,
        unit="м",
        tech=True,
    ),
    _p(
        "sea.shelf_max_depth_m",
        "Предел глубины шельфа",
        "continents",
        "float",
        50,
        2000,
        (300, 700),
        0,
        unit="м",
    ),
    # -- history, mountains, volcanoes -----------------------------------------------------------
    _p("history.enabled", "Считать геологическую историю", "history", "bool", tech=True),
    _p(
        "history.duration_ma",
        "Длительность истории",
        "history",
        "float",
        0,
        400,
        (15, 80),
        0,
        unit="млн лет",
        help="Дольше — старше и выше горы, шире океаны; дольше считается.",
    ),
    _p(
        "history.intensity",
        "Активность тектоники",
        "history",
        "choice",
        rand=ON,
        choices=(("calm", "спокойная"), ("earthlike", "как на Земле"), ("active", "бурная")),
    ),
    _p(
        "history.uplift_scale",
        "Горность (поднятие)",
        "history",
        "float",
        0.0,
        5.0,
        (0.5, 2.0),
        help="Главная ручка гор: выше и больше горных систем.",
    ),
    _p(
        "relief.relief_strength",
        "Сила макрорельефа",
        "history",
        "float",
        0.0,
        3.0,
        (0.6, 1.6),
        help="Общий контраст высот материков и океанов.",
    ),
    _p(
        "mountains.complexity",
        "Сложность горных систем",
        "history",
        "float",
        0.0,
        2.0,
        (0.5, 1.6),
        help="Больше — ветвистые хребты, узлы, нагорья.",
    ),
    _p("mountains.detail", "Детальность гор", "history", "float", 0.0, 2.0, (0.6, 1.5)),
    _p(
        "mountains.max_elevation_m",
        "Предел высоты гор",
        "history",
        "float",
        2000,
        12000,
        (6000, 10000),
        0,
        unit="м",
    ),
    _p(
        "mountains.style",
        "Стиль гор",
        "history",
        "choice",
        rand=ON,
        choices=(
            ("auto", "по тектонике (авто)"),
            ("volcanic_arc", "вулканические дуги"),
            ("alpine", "альпийский"),
            ("himalayan", "гималайский"),
            ("andean", "андский"),
            ("fold_belt", "складчатые пояса"),
            ("old_massif", "древние массивы"),
            ("broad_upland", "широкие плато"),
            ("mixed", "смешанный"),
            ("block", "глыбовый"),
            ("highland", "нагорья"),
            ("composite", "горные узлы"),
        ),
    ),
    _p(
        "mountains.peak_spacing_km",
        "Шаг вершин",
        "history",
        "float",
        20,
        300,
        (60, 120),
        0,
        unit="км",
    ),
    _p(
        "mountains.range_half_width_km",
        "Полуширина хребта",
        "history",
        "float",
        10,
        300,
        (40, 90),
        0,
        unit="км",
    ),
    _p(
        "mountains.spur_half_width_km",
        "Полуширина отрогов",
        "history",
        "float",
        5,
        200,
        (22, 50),
        0,
        unit="км",
    ),
    _p(
        "mountains.system_threshold_m",
        "Порог горной системы",
        "history",
        "float",
        50,
        3000,
        (300, 700),
        0,
        unit="м",
    ),
    _p(
        "history.subsidence_scale",
        "Прогибание (бассейны)",
        "history",
        "float",
        0.0,
        5.0,
        (0.5, 1.8),
    ),
    _p(
        "history.erosion_intensity",
        "Разрушение гор за историю",
        "history",
        "float",
        0.0,
        5.0,
        (0.5, 1.8),
        help="Сильнее — горы старше и ниже, больше осадков.",
    ),
    _p(
        "history.inherited_orogen_age_ma",
        "Возраст древних горных поясов",
        "history",
        "float",
        0,
        2000,
        (60, 400),
        0,
        unit="млн лет",
    ),
    _p(
        "history.time_step_ma",
        "Шаг истории",
        "history",
        "float",
        0.05,
        20,
        None,
        2,
        unit="млн лет",
        tech=True,
    ),
    _p(
        "history.checkpoint_stride",
        "Сохранять эпоху каждые N шагов",
        "history",
        "int",
        1,
        100,
        None,
        0,
        tech=True,
    ),
    _p(
        "history.epoch",
        "Эпоха карты",
        "history",
        "choice",
        choices=(("final", "итоговая"), ("initial", "начальная")),
        tech=True,
    ),
    _p(
        "mountains.backend",
        "Генератор гор",
        "history",
        "choice",
        choices=(("auto", "авто"), ("reference", "встроенный"), ("pmge", "PMGE (C++)")),
        tech=True,
    ),
    # -- ocean floor -------------------------------------------------------------------------------
    _p(
        "relief.ocean_floor_m",
        "Глубина океанского ложа (опорная)",
        "ocean",
        "float",
        -7000,
        -1500,
        (-5200, -3800),
        0,
        unit="м",
    ),
    _p(
        "terrain.ridge_depth_m",
        "Глубина срединных хребтов",
        "ocean",
        "float",
        1000,
        4500,
        (2200, 2900),
        0,
        unit="м",
    ),
    _p(
        "terrain.target_median_age_ma",
        "Средний возраст океанской коры",
        "ocean",
        "float",
        0,
        200,
        (35, 110),
        0,
        unit="млн лет",
        help="Моложе — мельче океаны (больше хребтов).",
    ),
    _p(
        "terrain.trench_depth_m",
        "Глубина жёлобов",
        "ocean",
        "float",
        0,
        8000,
        (1500, 4500),
        0,
        unit="м",
    ),
    _p(
        "terrain.abyssal_relief_m",
        "Холмистость ложа",
        "ocean",
        "float",
        0,
        1500,
        (100, 400),
        0,
        unit="м",
    ),
    _p(
        "terrain.fracture_zone_depth_m",
        "Глубина зон разломов",
        "ocean",
        "float",
        0,
        3000,
        (300, 900),
        0,
        unit="м",
    ),
    _p(
        "terrain.ridge_piece_length_km",
        "Длина сегментов хребта",
        "ocean",
        "float",
        20,
        1000,
        (120, 350),
        0,
        unit="км",
    ),
    _p(
        "terrain.pelagic_rate_m_per_ma",
        "Скорость осадконакопления",
        "ocean",
        "float",
        0,
        50,
        (2, 8),
        1,
        unit="м/млн лет",
    ),
    _p(
        "terrain.age_depth_transition_ma[0]",
        "Переход возраст–глубина: начало",
        "ocean",
        "float",
        0,
        150,
        None,
        0,
        unit="млн лет",
        tech=True,
    ),
    _p(
        "terrain.age_depth_transition_ma[1]",
        "Переход возраст–глубина: конец",
        "ocean",
        "float",
        1,
        200,
        None,
        0,
        unit="млн лет",
        tech=True,
    ),
    _p("terrain.transform_fracture_history", "Зоны разломов на дне", "ocean", "bool", rand=None),
    _p("terrain.subduction_trenches", "Глубоководные жёлоба", "ocean", "bool"),
    _p("terrain.submarine_canyons", "Подводные каньоны и конусы", "ocean", "bool"),
    _p("terrain.sediment_cover", "Осадочный чехол дна", "ocean", "bool"),
    _p("terrain.oceanic_plateaus", "Океанические плато", "ocean", "bool", rand=ON),
    _p(
        "terrain.min_ocean_area_km2",
        "Минимальная площадь мирового океана",
        "ocean",
        "float",
        1e4,
        1e8,
        None,
        0,
        log=True,
        unit="км²",
        tech=True,
    ),
    _p("terrain.basins:intracratonic_sag", "Бассейны: прогибы платформ", "ocean", "bool", rand=ON),
    _p("terrain.basins:rift_graben", "Бассейны: рифты", "ocean", "bool", rand=ON),
    _p(
        "terrain.basins:foreland_foredeep", "Бассейны: предгорные прогибы", "ocean", "bool", rand=ON
    ),
    _p("terrain.basins:pull_apart", "Бассейны: впадины растяжения", "ocean", "bool", rand=ON),
    _p(
        "terrain.elastic_thickness_km",
        "Упругая толщина литосферы",
        "ocean",
        "float",
        2,
        150,
        (15, 45),
        0,
        unit="км",
    ),
    _p(
        "terrain.foreland_fill",
        "Заполнение предгорных прогибов",
        "ocean",
        "float",
        0,
        1,
        (0.6, 1.0),
    ),
    _p(
        "terrain.sag_density",
        "Число прогибов платформ",
        "ocean",
        "float",
        0,
        20,
        (0.5, 3.0),
        1,
        unit="на 10 млн км²",
    ),
    _p("terrain.massif_graphs", "Графы горных массивов", "ocean", "bool", tech=True),
    _p("terrain.enabled", "Считать рельеф v4.6", "ocean", "bool", tech=True),
    # -- climate -------------------------------------------------------------------------------------
    _p(
        "climate.temperature_offset_c",
        "Температура мира",
        "climate",
        "float",
        -15,
        15,
        (-8, 6),
        1,
        unit="°C",
        help="Холоднее — тундры, ледники и ледниковые щиты; теплее — тропики до высоких широт.",
    ),
    _p(
        "climate.precipitation_scale",
        "Влажность (осадки)",
        "climate",
        "float",
        0.1,
        4.0,
        (0.6, 1.8),
        help="Меньше — пустыни, больше — влажные леса и болота.",
    ),
    _p(
        "climate.profile",
        "Профиль климата",
        "climate",
        "choice",
        rand=ON,
        choices=(("earthlike", "как на Земле"), ("dry", "сухой"), ("wet", "влажный")),
    ),
    _p(
        "climate.seasonality",
        "Сезонность",
        "climate",
        "float",
        0.1,
        4.0,
        (0.6, 1.6),
        help="Контраст лета и зимы, сила муссонов.",
    ),
    _p("climate.wind_strength", "Сила ветров", "climate", "float", 0.1, 4.0, (0.7, 1.5)),
    _p(
        "climate.moisture_transport_damping",
        "Затухание влаги вглубь суши",
        "climate",
        "float",
        0.1,
        4.0,
        (0.6, 1.4),
        help="Больше — суше внутренние районы материков.",
    ),
    _p(
        "climate.lapse_rate_c_per_km",
        "Охлаждение с высотой",
        "climate",
        "float",
        3.0,
        10.0,
        (5.5, 7.5),
        1,
        unit="°C/км",
    ),
    _p(
        "climate.deep_infiltration",
        "Уход воды в глубокие горизонты",
        "climate",
        "float",
        0.0,
        0.5,
        (0.02, 0.1),
    ),
    _p("climate.enabled", "Считать климат и реки (v4)", "climate", "bool", tech=True),
    # -- water and erosion ------------------------------------------------------------------------------
    _p(
        "hydrology.river_min_discharge_m3_s",
        "Порог реки (расход)",
        "water",
        "float",
        0.1,
        5000,
        (10, 80),
        1,
        log=True,
        unit="м³/с",
        help="Меньше — гуще сеть рек на карте.",
    ),
    _p(
        "hydrology.river_min_upstream_hexes",
        "Порог реки (площадь водосбора)",
        "water",
        "int",
        1,
        200,
        (3, 8),
        0,
        unit="гексов",
    ),
    _p(
        "hydrology.river_min_system_hexes",
        "Минимальная речная система",
        "water",
        "int",
        1,
        500,
        (5, 15),
        0,
        unit="гексов",
    ),
    _p(
        "hydrology.min_lake_depth_m",
        "Минимальная глубина озера",
        "water",
        "float",
        0,
        500,
        (1, 6),
        1,
        unit="м",
    ),
    _p("hydrology.channel_loss", "Потери воды в руслах", "water", "float", 0, 200, (10, 40), 0),
    _p(
        "hydrology.routing",
        "Сток по рельефу",
        "water",
        "choice",
        rand=ON,
        choices=(("stochastic", "с ветвлением"), ("steepest", "по крутизне")),
    ),
    _p(
        "hydrology.edge_outlets",
        "Выходы рек за край",
        "water",
        "choice",
        choices=(("world", "по стоку мира"), ("all", "везде")),
        tech=True,
    ),
    _p("hydrology.external_inflow", "Приток рек из-за края", "water", "bool", tech=True),
    _p("erosion.enabled", "Эрозия рек", "water", "bool", tech=True),
    _p(
        "erosion.duration_ma",
        "Длительность эрозии",
        "water",
        "float",
        0,
        50,
        (1, 6),
        1,
        unit="млн лет",
        help="Дольше — глубже долины и сглаженнее горы.",
    ),
    _p("erosion.steps", "Шагов эрозии", "water", "int", 1, 200, None, 0, tech=True),
    _p(
        "erosion.k_stream",
        "Сила речного размыва",
        "water",
        "float",
        1e-9,
        1e-4,
        (4e-8, 3e-7),
        9,
        log=True,
    ),
    _p("erosion.m_exponent", "Показатель площади (m)", "water", "float", 0.2, 0.8, (0.4, 0.6)),
    _p(
        "erosion.incision_threshold_m_ma",
        "Порог врезания",
        "water",
        "float",
        0,
        1000,
        (0, 50),
        0,
        unit="м/млн лет",
    ),
    _p(
        "erosion.reference_runoff_mm_yr",
        "Опорный сток",
        "water",
        "float",
        10,
        5000,
        None,
        0,
        unit="мм/год",
        tech=True,
    ),
    _p(
        "erosion.diffusivity_m2_yr",
        "Сглаживание склонов",
        "water",
        "float",
        0,
        1e5,
        (300, 2000),
        0,
        log=True,
        unit="м²/год",
    ),
    _p("erosion.k_transport", "Перенос наносов", "water", "float", 0, 10, (0.01, 0.05), 3),
    _p("erosion.continue_uplift", "Поднятие во время эрозии", "water", "bool", tech=True),
    _p("erosion.update_climate", "Обновлять климат при эрозии", "water", "bool", tech=True),
    # -- soils and water ---------------------------------------------------------------------------------
    _p("environment.enabled", "Почвы и сезоны (v4.5)", "soils", "bool", tech=True),
    _p(
        "environment.max_soil_depth_m",
        "Максимальная мощность почвы",
        "soils",
        "float",
        0.2,
        10,
        (1.5, 5.0),
        1,
        unit="м",
    ),
    _p(
        "environment.reference_root_depth_m",
        "Глубина корней",
        "soils",
        "float",
        0.1,
        5,
        (0.6, 1.6),
        1,
        unit="м",
    ),
    _p(
        "environment.depletion_fraction",
        "Доступная растениям влага",
        "soils",
        "float",
        0.05,
        0.95,
        (0.4, 0.6),
    ),
    _p(
        "environment.rain_days_per_month",
        "Дождливых дней в месяце",
        "soils",
        "float",
        0.5,
        31,
        (4, 10),
        1,
    ),
    _p(
        "environment.baseflow_timescale_yr",
        "Время отдачи грунтовых вод",
        "soils",
        "float",
        0.05,
        20,
        (0.5, 3),
        2,
        unit="лет",
    ),
    _p(
        "environment.degree_day_mm",
        "Таяние снега",
        "soils",
        "float",
        0.5,
        10,
        (2.5, 4.5),
        1,
        unit="мм/°C·сут",
    ),
    _p(
        "environment.dry_month_ratio",
        "Порог сухого месяца P/PET",
        "soils",
        "float",
        0.1,
        1.0,
        (0.4, 0.6),
    ),
    _p(
        "environment.gdd_base_c",
        "База сумм температур",
        "soils",
        "float",
        0,
        15,
        None,
        1,
        unit="°C",
        tech=True,
    ),
    _p(
        "environment.spinup_max_years",
        "Разгон водного баланса",
        "soils",
        "int",
        1,
        200,
        None,
        0,
        unit="лет",
        tech=True,
    ),
    _p(
        "environment.water_balance_tolerance_mm",
        "Точность баланса",
        "soils",
        "float",
        1e-8,
        1.0,
        None,
        8,
        log=True,
        unit="мм",
        tech=True,
    ),
    # -- topoclimate -------------------------------------------------------------------------------------
    _p("topoclimate.enabled", "Высотные пояса и топоклимат (v4.7)", "topo", "bool", tech=True),
    _p(
        "topoclimate.quality",
        "Детальность участков рельефа",
        "topo",
        "choice",
        choices=(
            ("preview", "черновая (быстро)"),
            ("balanced", "обычная"),
            ("reference", "точная (медленно)"),
        ),
        tech=True,
    ),
    _p(
        "topoclimate.temperature_mode",
        "Пересчёт температуры",
        "topo",
        "choice",
        choices=(
            ("reference_anchored", "от опорной высоты"),
            ("preserve_coarse_mean", "сохранять среднее гекса"),
        ),
        tech=True,
    ),
    _p(
        "topoclimate.tree_start_temp_c",
        "Порог сезона роста",
        "topo",
        "float",
        -2,
        5,
        (0.5, 1.5),
        1,
        unit="°C",
    ),
    _p(
        "topoclimate.tree_gsl_days",
        "Граница леса: длина сезона",
        "topo",
        "float",
        30,
        200,
        (80, 110),
        0,
        unit="сут",
    ),
    _p(
        "topoclimate.tree_gst_c",
        "Граница леса: средняя T сезона",
        "topo",
        "float",
        3,
        12,
        (5.5, 7.5),
        1,
        unit="°C",
    ),
    _p(
        "topoclimate.smoothing_days",
        "Плавность границы (сутки)",
        "topo",
        "float",
        1,
        100,
        None,
        0,
        tech=True,
    ),
    _p(
        "topoclimate.smoothing_c",
        "Плавность границы (°C)",
        "topo",
        "float",
        0.1,
        10,
        None,
        1,
        tech=True,
    ),
    _p(
        "topoclimate.cap_night_max_c",
        "Холодные карманы: охлаждение",
        "topo",
        "float",
        0,
        15,
        (3, 9),
        1,
        unit="°C",
    ),
    _p(
        "topoclimate.cap_event_frequency",
        "Холодные карманы: частота",
        "topo",
        "float",
        0,
        1,
        (0.15, 0.35),
    ),
    _p(
        "topoclimate.solar_day_anomaly_max_c",
        "Прогрев склонов солнцем",
        "topo",
        "float",
        0,
        8,
        (1, 3),
        1,
        unit="°C",
    ),
    _p(
        "topoclimate.mass_elevation_effect_max_c",
        "Эффект массивности гор",
        "topo",
        "float",
        0,
        3,
        (0, 1.5),
        1,
        unit="°C",
    ),
    _p("topoclimate.rh_marine", "Влажность воздуха у моря", "topo", "float", 0, 1, (0.7, 0.9)),
    _p(
        "topoclimate.rh_inland", "Влажность воздуха внутри суши", "topo", "float", 0, 1, (0.35, 0.6)
    ),
    _p("topoclimate.snow_mobile_share", "Переметание снега", "topo", "float", 0, 1, (0.15, 0.45)),
    _p(
        "topoclimate.cover_swe_threshold_mm",
        "Порог снежного покрова",
        "topo",
        "float",
        0.5,
        50,
        None,
        1,
        unit="мм",
        tech=True,
    ),
    _p(
        "topoclimate.precipitation_factor_bounds[0]",
        "Осадки на склонах: минимум",
        "topo",
        "float",
        0.01,
        1.0,
        None,
        2,
        tech=True,
    ),
    _p(
        "topoclimate.precipitation_factor_bounds[1]",
        "Осадки на склонах: максимум",
        "topo",
        "float",
        1.0,
        10.0,
        None,
        1,
        tech=True,
    ),
    _p(
        "topoclimate.max_units_per_hex",
        "Участков рельефа на гекс",
        "topo",
        "int",
        1,
        24,
        None,
        0,
        tech=True,
    ),
    _p(
        "topoclimate.horizon_directions",
        "Направлений горизонта",
        "topo",
        "int",
        4,
        72,
        None,
        0,
        tech=True,
    ),
    _p(
        "topoclimate.horizon_max_distance_km",
        "Дальность горизонта",
        "topo",
        "float",
        10,
        300,
        None,
        0,
        unit="км",
        tech=True,
    ),
    _p(
        "topoclimate.sunlight_hours_per_day_samples",
        "Точек солнца в сутках",
        "topo",
        "int",
        4,
        48,
        None,
        0,
        tech=True,
    ),
    # -- biomes and ice -------------------------------------------------------------------------------------
    _p("biomes.enabled", "Биомы и лёд (v5.0)", "biomes", "bool", tech=True),
    _p(
        "biomes.paleo_ice_initialization",
        "Древние ледниковые щиты",
        "biomes",
        "bool",
        rand=ON,
        help="Холодные снежные районы начинают с древнего льда (затем по балансу).",
    ),
    _p("biomes.glacier_flow", "Течение льда", "biomes", "bool", tech=True),
    _p(
        "biomes.cryo_spinup_years",
        "Разгон снега и льда",
        "biomes",
        "int",
        0,
        1000,
        None,
        0,
        unit="лет",
        tech=True,
    ),
    _p(
        "biomes.pft_tau",
        "Мягкость конкуренции растений",
        "biomes",
        "float",
        0.01,
        1.0,
        (0.05, 0.15),
        3,
    ),
    _p(
        "biomes.cover_v_ref",
        "Продуктивность сплошного покрова",
        "biomes",
        "float",
        0.05,
        2.0,
        (0.18, 0.35),
        2,
    ),
    _p(
        "biomes.annual_pet_epsilon_mm",
        "Порог расчёта аридности",
        "biomes",
        "float",
        1,
        500,
        None,
        0,
        unit="мм/год",
        tech=True,
    ),
    _p(
        "biomes.profile",
        "Профиль биомов",
        "biomes",
        "choice",
        choices=(("earthlike_potential_natural", "земной, потенциальная растительность"),),
        tech=True,
    ),
    _p(
        "biomes.mode",
        "Режим растительности",
        "biomes",
        "choice",
        choices=(("equilibrium", "равновесный (потенциальный)"),),
        tech=True,
    ),
    _p(
        "biomes.co2_ppm",
        "CO₂ (метаданные)",
        "biomes",
        "float",
        150,
        2000,
        None,
        0,
        unit="ppm",
        tech=True,
    ),
    # -- technical ------------------------------------------------------------------------------------------
    _p(
        "reference_cells",
        "Разрешение мирового растра",
        "technical",
        "int",
        32,
        2048,
        None,
        0,
        tech=True,
    ),
    _p(
        "continents.skeleton_cells",
        "Разрешение каркаса материков",
        "technical",
        "int",
        128,
        2048,
        None,
        0,
        tech=True,
    ),
)
_BY_PATH = {p.path: p for p in PARAMS}

#: size presets: (title, width, height, hex km, mode)
SIZE_PRESETS = (
    ("Крошечный — 60×40, 75 км (секунды)", 60, 40, 75.0, "world"),
    ("Маленький — 100×100, 40 км", 100, 100, 40.0, "regional"),
    ("Средний — 240×160, 40 км (весь мир)", 240, 160, 40.0, "world"),
    ("Большой — 360×240, 25 км (весь мир)", 360, 240, 25.0, "world"),
    ("Очень большой — 500×300, 25 км", 500, 300, 25.0, "world"),
    ("Огромный — 1000×800, 25 км (минуты)", 1000, 800, 25.0, "world"),
)

#: world presets: partial overrides (path -> value), applied on top of the current values
WORLD_PRESETS = {
    "Как Земля": {
        "climate.temperature_offset_c": 0.0,
        "climate.precipitation_scale": 1.0,
        "sea.target_land_fraction": 0.29,
        "terrain.island_amount": 1.0,
        "history.uplift_scale": 1.0,
        "continents.continent_profile": "balanced",
    },
    "Ледниковый период": {
        "climate.temperature_offset_c": -9.0,
        "climate.precipitation_scale": 0.8,
        "sea.target_land_fraction": 0.33,
        "biomes.paleo_ice_initialization": True,
    },
    "Снежок (ледяная планета)": {
        "climate.temperature_offset_c": -15.0,
        "climate.precipitation_scale": 0.6,
        "biomes.paleo_ice_initialization": True,
    },
    "Тепличный мир": {
        "climate.temperature_offset_c": 10.0,
        "climate.precipitation_scale": 1.5,
        "sea.target_land_fraction": 0.25,
    },
    "Пустынный мир": {
        "climate.temperature_offset_c": 5.0,
        "climate.precipitation_scale": 0.35,
        "climate.profile": "dry",
        "climate.moisture_transport_damping": 1.6,
    },
    "Влажные тропики": {
        "climate.latitude_center_deg": 5.0,
        "climate.temperature_offset_c": 2.0,
        "climate.precipitation_scale": 1.8,
        "climate.profile": "wet",
    },
    "Архипелаги": {
        "terrain.island_amount": 3.0,
        "sea.target_land_fraction": 0.15,
        "continents.continent_profile": "fragmented",
        "crust.mean_continent_area_km2": 1.5e6,
        "history.hotspot_density": 3.0,
    },
    "Пангея": {
        "continents.continent_profile": "supercontinent",
        "crust.assembly": 1.0,
        "continents.major_landmass_count": 1,
        "sea.target_land_fraction": 0.35,
        "terrain.island_amount": 0.5,
    },
    "Горный мир": {
        "history.uplift_scale": 2.5,
        "relief.relief_strength": 1.6,
        "mountains.complexity": 1.8,
        "history.intensity": "active",
        "plates.count": 24,
    },
    "Старый спокойный мир": {
        "history.uplift_scale": 0.4,
        "history.intensity": "calm",
        "history.erosion_intensity": 2.5,
        "history.duration_ma": 120.0,
    },
    "Водный мир": {
        "sea.target_land_fraction": 0.06,
        "terrain.island_amount": 2.5,
        "crust.continental_fraction": 0.12,
    },
}

_INDEX = re.compile(r"^(?P<base>[^\[:]+)(\[(?P<i>\d+)\]|:(?P<member>\w+))?$")


def param(path: str) -> Param:
    return _BY_PATH[path]


def _get(data: dict, dotted: str):
    cur: Any = data
    for part in dotted.split("."):
        cur = cur[part]
    return cur


def _set(data: dict, dotted: str, value) -> None:
    parts = dotted.split(".")
    cur = data
    for part in parts[:-1]:
        cur = cur.setdefault(part, {})
    cur[parts[-1]] = value


def values_from_config(config: Config) -> dict:
    """``{path: value}`` of every schema parameter (``None`` = auto)."""
    data = config.to_dict()
    out = {}
    for p in PARAMS:
        m = _INDEX.match(p.path)
        base, idx, member = m["base"], m["i"], m["member"]
        v = _get(data, base)
        if idx is not None:
            v = list(v)[int(idx)]
        elif member is not None:
            v = member in set(v)
        if p.auto and v == "auto":
            v = None
        out[p.path] = v
    return out


def config_from_values(values: dict, base: Config | None = None, seed: str | None = None) -> Config:
    """A :class:`Config` from schema values (missing paths keep ``base``)."""
    data = (base or Config()).to_dict()
    tuples: dict[str, list] = {}
    members: dict[str, set] = {}
    for p in PARAMS:
        if p.path not in values:
            continue
        v = values[p.path]
        m = _INDEX.match(p.path)
        base_path, idx, member = m["base"], m["i"], m["member"]
        if idx is not None:
            tuples.setdefault(base_path, list(_get(data, base_path)))[int(idx)] = v
            continue
        if member is not None:
            members.setdefault(base_path, set(_get(data, base_path)))
            (members[base_path].add if v else members[base_path].discard)(member)
            continue
        if p.auto and v is None:
            v = "auto"
        elif p.kind == "int" and v is not None:
            v = int(round(float(v)))
        elif p.kind == "float" and v is not None:
            v = float(v)
        elif p.kind == "bool":
            v = bool(v)
        _set(data, base_path, v)
    for path, v in tuples.items():
        if path == "terrain.age_depth_transition_ma" and v[0] >= v[1]:
            v = [min(v), max(v) + (1.0 if v[0] == v[1] else 0.0)]
        if path == "topoclimate.precipitation_factor_bounds":
            v = [min(v[0], 1.0), max(v[1], 1.0)]
        _set(data, path, v)
    order = ("intracratonic_sag", "rift_graben", "foreland_foredeep", "pull_apart")
    for path, v in members.items():
        _set(data, path, [b for b in order if b in v])
    if seed is not None:
        data["seed"] = seed
    return Config.from_dict(data)


def random_seed(rng: random.Random | None = None, length: int = 8) -> str:
    """A new world seed ``[A-Za-z0-9]+`` (starts with a letter)."""
    rng = rng or random.SystemRandom()
    first = rng.choice(string.ascii_letters)
    return first + "".join(
        rng.choice(string.ascii_letters + string.digits) for _ in range(length - 1)
    )


def randomize(values: dict, locked: set | frozenset = frozenset(), seed: str | None = None,
              include_technical: bool = False) -> dict:  # fmt: skip
    """New values: every unlocked randomisable parameter drawn from its ``rand`` range.

    ``seed`` makes the draw reproducible (same seed + same locks = same settings). Locked
    and technical parameters keep their values. Auto parameters stay auto one time in three.
    """
    rng = random.Random(seed) if seed is not None else random.SystemRandom()
    out = dict(values)
    for p in PARAMS:
        if p.path in locked or (p.technical and not include_technical):
            continue
        if p.kind in ("float", "int"):
            if p.rand is None:
                continue
            if p.auto and rng.random() < 1.0 / 3.0:
                out[p.path] = None
                continue
            lo, hi = p.rand
            if p.log and lo > 0:
                import math

                v = math.exp(rng.uniform(math.log(lo), math.log(hi)))
            else:
                v = rng.uniform(lo, hi)
            out[p.path] = int(round(v)) if p.kind == "int" else round(v, max(p.decimals, 2))
        elif p.kind == "choice":
            if p.rand:
                out[p.path] = rng.choice([c[0] for c in p.choices])
        elif p.kind == "bool":
            if p.rand:
                out[p.path] = rng.random() < 0.75
    return out


def default_locked() -> set:
    """Technical parameters start locked (size, mode, resolution, quality)."""
    return {p.path for p in PARAMS if p.technical}


def validate_values(values: dict) -> list[str]:
    """Problems of a value set (empty = valid): builds the config and reports errors."""
    try:
        config_from_values(values)
    except Exception as exc:  # noqa: BLE001 - shown to the user
        return [str(exc)]
    return []


def estimate_seconds(width: int, height: int, through: str = "v50") -> float:
    """Rough generation time on the reference machine, s (see :mod:`hexworld.progress`)."""
    from .progress import expected_seconds

    return expected_seconds(through, int(width) * int(height))


def all_config_paths() -> list[str]:
    """Every leaf of :class:`Config` (to check that the schema covers it)."""
    c = Config()
    out = []
    for f in fields(c):
        v = getattr(c, f.name)
        if is_dataclass(v):
            out += [f"{f.name}.{g.name}" for g in fields(v)]
        else:
            out.append(f.name)
    return out
