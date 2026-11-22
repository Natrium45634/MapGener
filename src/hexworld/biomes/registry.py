# ruff: noqa: E501  (trait / class tables read better one row per line)
"""Biome registry, physical cover classes, realms and plant functional types (TZ 5.0, 2, 5, 8).

Every class has a stable string ID and a stable numeric code (its position in the list). The
list is **append-only**: a later version may add classes at the end or mark one deprecated,
it never renumbers. ``[П]`` thresholds live in :mod:`hexworld.biomes.pft` and the config,
not here; this module is names, families and colours only.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

REGISTRY_VERSION = "biome_registry_v1"
PFT_REGISTRY_VERSION = "pft_earthlike_v1"

REALMS = ("unknown", "terrestrial", "inland_water", "marine", "transitional")

#: non-overlapping ground cover (the additive area budget of a unit; sums to 1)
COVER_CLASSES = (
    "vegetation",
    "bare_soil",
    "rock",
    "sand",
    "salt",
    "snow_perennial",
    "firn",
    "glacier_ice",
    "open_water",
)


@dataclass(frozen=True)
class BiomeClass:
    id: str
    family: str
    ru: str
    en: str
    rgb: tuple[int, int, int]


_B = BiomeClass
#: append-only list: the index is the stable numeric code
BIOMES: tuple[BiomeClass, ...] = (
    _B("unresolved", "special", "не определено (нет данных)", "unresolved", (128, 128, 128)),
    # 8.1 hot / temperate barren and sparse
    _B(
        "desert_hot_sand_erg",
        "desert",
        "жаркая песчаная пустыня (эрг)",
        "hot sand desert (erg)",
        (242, 214, 150),
    ),
    _B(
        "desert_hot_gravel_reg",
        "desert",
        "щебнистая пустыня (рег)",
        "hot gravel desert (reg)",
        (222, 196, 150),
    ),
    _B(
        "desert_hot_rock_hamada",
        "desert",
        "каменистая пустыня (хамада)",
        "rock desert (hamada)",
        (196, 168, 132),
    ),
    _B("desert_hot_salt_flat", "desert", "солончаковая пустошь", "salt flat", (245, 240, 230)),
    _B(
        "desert_playa_dry_lake",
        "desert",
        "такыр / сухое озёрное дно",
        "playa / dry lake bed",
        (232, 222, 200),
    ),
    _B("desert_badlands_barren", "desert", "бедленды", "badlands", (205, 150, 110)),
    _B(
        "desert_hot_fog_coast",
        "desert",
        "туманная прибрежная пустыня",
        "coastal fog desert",
        (214, 210, 180),
    ),
    _B("desert_saline_crust", "desert", "сухой солончак", "saline crust", (236, 230, 214)),
    _B("desert_semi_arid_sparse", "desert", "полупустыня", "semi-desert", (226, 206, 150)),
    _B(
        "desert_temperate_cold",
        "desert",
        "умеренная холодная пустыня",
        "cold temperate desert",
        (205, 196, 165),
    ),
    _B(
        "desert_high_mountain_cold",
        "desert",
        "холодная высокогорная пустыня",
        "cold high-mountain desert",
        (190, 180, 165),
    ),
    _B(
        "barren_volcanic_recent",
        "barren",
        "молодые лавовые и пепловые поля",
        "recent lava / ash fields",
        (90, 80, 80),
    ),
    _B("barren_rock_cliff", "barren", "голые скалы и осыпи", "bare rock / scree", (165, 160, 155)),
    _B(
        "barren_erosional_surface",
        "barren",
        "активно размываемая поверхность",
        "erosional surface",
        (190, 160, 135),
    ),
    _B(
        "barren_gypsum_alkali",
        "barren",
        "гипсовые / щелочные пустоши",
        "gypsum / alkali barrens",
        (235, 225, 205),
    ),
    _B(
        "desert_sparse_undifferentiated",
        "desert",
        "пустыня (без уточнения субстрата)",
        "desert (undifferentiated)",
        (230, 205, 160),
    ),
    # 8.2 tropical / subtropical forests
    _B(
        "forest_tropical_everwet",
        "forest_tropical",
        "влажный тропический лес",
        "tropical rainforest",
        (20, 100, 40),
    ),
    _B(
        "forest_tropical_monsoon",
        "forest_tropical",
        "муссонный лес",
        "monsoon forest",
        (50, 125, 45),
    ),
    _B(
        "forest_tropical_seasonal_dry",
        "forest_tropical",
        "сезонно-сухой тропический лес",
        "seasonally dry tropical forest",
        (110, 140, 55),
    ),
    _B(
        "forest_tropical_cloud_montane",
        "forest_tropical",
        "горный облачный лес",
        "tropical montane cloud forest",
        (40, 120, 95),
    ),
    _B(
        "forest_tropical_swamp",
        "forest_tropical",
        "заболоченный тропический лес",
        "tropical swamp forest",
        (40, 95, 75),
    ),
    _B(
        "forest_tropical_floodplain",
        "forest_tropical",
        "тропический пойменный (галерейный) лес",
        "tropical floodplain forest",
        (60, 115, 70),
    ),
    _B(
        "forest_tropical_mangrove",
        "forest_tropical",
        "мангровый лес",
        "mangrove forest",
        (30, 90, 90),
    ),
    _B(
        "forest_subtropical_evergreen",
        "forest_subtropical",
        "субтропический вечнозелёный лес",
        "subtropical evergreen forest",
        (45, 120, 70),
    ),
    _B(
        "forest_subtropical_seasonal",
        "forest_subtropical",
        "субтропический сезонный лес",
        "subtropical seasonal forest",
        (95, 140, 70),
    ),
    _B(
        "woodland_tropical_dry",
        "woodland",
        "тропическое сухое редколесье",
        "dry tropical woodland",
        (150, 160, 75),
    ),
    # 8.3 temperate
    _B(
        "forest_temperate_deciduous",
        "forest_temperate",
        "широколиственный листопадный лес",
        "temperate deciduous forest",
        (85, 150, 60),
    ),
    _B(
        "forest_temperate_mixed",
        "forest_temperate",
        "смешанный лес",
        "temperate mixed forest",
        (70, 130, 70),
    ),
    _B(
        "forest_temperate_evergreen_broadleaf",
        "forest_temperate",
        "вечнозелёный широколиственный лес",
        "temperate evergreen broadleaf forest",
        (60, 130, 85),
    ),
    _B(
        "forest_temperate_conifer",
        "forest_temperate",
        "умеренный хвойный лес",
        "temperate conifer forest",
        (45, 105, 75),
    ),
    _B(
        "forest_temperate_rainforest",
        "forest_temperate",
        "умеренный дождевой лес",
        "temperate rainforest",
        (30, 95, 70),
    ),
    _B(
        "forest_mediterranean_sclerophyll",
        "forest_temperate",
        "средиземноморский жёстколистный лес",
        "Mediterranean sclerophyll forest",
        (125, 140, 70),
    ),
    _B(
        "shrubland_mediterranean",
        "shrubland",
        "маквис / чапараль",
        "Mediterranean shrubland",
        (160, 150, 85),
    ),
    _B(
        "woodland_temperate_open",
        "woodland",
        "умеренное редколесье",
        "open temperate woodland",
        (130, 160, 85),
    ),
    _B(
        "forest_montane_temperate",
        "forest_temperate",
        "горный умеренный лес",
        "temperate montane forest",
        (55, 115, 85),
    ),
    _B(
        "forest_floodplain_temperate",
        "forest_temperate",
        "умеренный пойменный лес",
        "temperate floodplain forest",
        (70, 140, 95),
    ),
    _B(
        "forest_wetland_temperate",
        "forest_temperate",
        "заболоченный лес",
        "temperate swamp forest",
        (65, 110, 90),
    ),
    # 8.4 boreal
    _B(
        "forest_boreal_dark_conifer",
        "forest_boreal",
        "темнохвойная тайга",
        "dark conifer taiga",
        (35, 85, 70),
    ),
    _B(
        "forest_boreal_light_conifer",
        "forest_boreal",
        "светлохвойная тайга",
        "light conifer taiga",
        (75, 115, 80),
    ),
    _B(
        "forest_boreal_deciduous_larch",
        "forest_boreal",
        "лиственничная тайга",
        "larch taiga",
        (105, 125, 80),
    ),
    _B("forest_boreal_mixed", "forest_boreal", "смешанная тайга", "mixed taiga", (60, 105, 80)),
    _B(
        "woodland_boreal_sparse",
        "forest_boreal",
        "северное редколесье",
        "sparse boreal woodland",
        (120, 135, 100),
    ),
    _B("forest_tundra_mosaic", "forest_boreal", "лесотундра", "forest-tundra", (140, 150, 115)),
    _B(
        "boreal_peat_swamp_forest",
        "forest_boreal",
        "заболоченная тайга",
        "boreal peat swamp forest",
        (85, 105, 95),
    ),
    # 8.5 grass / shrub / transition
    _B("savanna_tree_grass", "savanna", "древесная саванна", "tree savanna", (175, 175, 75)),
    _B("savanna_shrub_grass", "savanna", "кустарниковая саванна", "shrub savanna", (195, 180, 95)),
    _B(
        "savanna_open_grass",
        "savanna",
        "открытая травяная саванна",
        "open grass savanna",
        (215, 200, 105),
    ),
    _B(
        "woodland_dry_tropical",
        "savanna",
        "сухое тропическое редколесье (мёомбо)",
        "dry tropical woodland (miombo)",
        (160, 165, 85),
    ),
    _B(
        "grassland_temperate_prairie",
        "grassland",
        "прерия / луговая степь",
        "temperate prairie",
        (185, 200, 110),
    ),
    _B("grassland_steppe", "grassland", "степь", "steppe", (200, 200, 120)),
    _B("grassland_dry_steppe", "grassland", "сухая степь", "dry steppe", (215, 205, 140)),
    _B("forest_steppe_mosaic", "grassland", "лесостепь", "forest-steppe", (150, 175, 100)),
    _B(
        "shrubland_temperate_dry",
        "shrubland",
        "сухие умеренные кустарники",
        "dry temperate shrubland",
        (190, 180, 125),
    ),
    _B(
        "scrub_subtropical",
        "shrubland",
        "субтропический кустарник",
        "subtropical scrub",
        (180, 170, 100),
    ),
    _B("grassland_montane", "mountain", "горные луга", "montane grassland", (165, 190, 115)),
    _B("grassland_alpine", "mountain", "альпийские луга", "alpine meadow", (180, 200, 150)),
    _B("steppe_alpine", "mountain", "высокогорная степь", "alpine steppe", (200, 195, 150)),
    _B("meadow_subalpine", "mountain", "субальпийские луга", "subalpine meadow", (150, 185, 130)),
    _B(
        "heath_alpine", "mountain", "высокогорная пустошь (вереск)", "alpine heath", (170, 150, 140)
    ),
    _B(
        "scree_alpine_sparse",
        "mountain",
        "растительность осыпей",
        "sparse scree vegetation",
        (185, 180, 170),
    ),
    # 8.6 tundra, polar, snow and ice
    _B("tundra_shrub", "tundra", "кустарниковая тундра", "shrub tundra", (150, 160, 130)),
    _B("tundra_sedge_graminoid", "tundra", "осоковая тундра", "sedge tundra", (165, 175, 145)),
    _B(
        "tundra_moss_lichen",
        "tundra",
        "мохово-лишайниковая тундра",
        "moss-lichen tundra",
        (180, 185, 165),
    ),
    _B("tundra_alpine", "tundra", "горная тундра", "alpine tundra", (175, 170, 155)),
    _B("tundra_polar_desert_sparse", "polar", "полярная пустыня", "polar desert", (205, 205, 195)),
    _B(
        "barren_polar_rock",
        "polar",
        "безрастительный полярный камень",
        "barren polar rock",
        (150, 150, 155),
    ),
    _B("barren_nunatak", "polar", "нунатак", "nunatak", (120, 115, 120)),
    _B(
        "snow_perennial_field",
        "ice",
        "многолетнее снежное поле",
        "perennial snowfield",
        (250, 252, 255),
    ),
    _B("firn_field", "ice", "фирновое поле", "firn field", (225, 235, 248)),
    _B("glacier_mountain", "ice", "горный ледник", "mountain glacier", (190, 215, 240)),
    _B("glacier_ice_cap", "ice", "ледниковая шапка", "ice cap", (205, 225, 245)),
    _B("glacier_ice_sheet", "ice", "материковый ледниковый щит", "ice sheet", (215, 232, 250)),
    _B(
        "glacier_ablation_zone",
        "ice",
        "зона абляции ледника",
        "glacier ablation zone",
        (170, 195, 220),
    ),
    # 8.7 wetlands
    _B(
        "wetland_fresh_marsh",
        "wetland",
        "пресноводное травяное болото",
        "freshwater marsh",
        (95, 160, 150),
    ),
    _B("wetland_fen", "wetland", "низинное болото", "fen", (110, 150, 120)),
    _B("wetland_bog", "wetland", "верховое болото", "bog", (130, 140, 110)),
    _B(
        "wetland_peatland_mosaic",
        "wetland",
        "торфяная болотная мозаика",
        "peatland mosaic",
        (120, 145, 115),
    ),
    _B("wetland_swamp_forest", "wetland", "заболоченный лес", "swamp forest", (70, 120, 100)),
    _B(
        "wetland_floodplain_seasonal",
        "wetland",
        "сезонная пойма",
        "seasonal floodplain",
        (120, 175, 150),
    ),
    _B(
        "wetland_riparian_gallery",
        "wetland",
        "приречная галерейная растительность",
        "riparian gallery",
        (80, 150, 110),
    ),
    _B(
        "wetland_inland_saline",
        "wetland",
        "внутреннее солёное болото",
        "inland saline marsh",
        (175, 190, 170),
    ),
    _B(
        "wetland_saltmarsh_tidal",
        "wetland",
        "приморский солёный марш",
        "tidal salt marsh",
        (150, 180, 165),
    ),
    _B(
        "wetland_mangrove_intertidal",
        "wetland",
        "мангровая литораль",
        "intertidal mangrove",
        (40, 110, 100),
    ),
    _B(
        "wetland_unclassified",
        "wetland",
        "болото (тип не определён)",
        "wetland (unclassified)",
        (110, 160, 140),
    ),
    # 8.8 fresh water and marine habitats
    _B("water_river_channel", "freshwater", "русло реки", "river channel", (50, 110, 200)),
    _B(
        "water_river_ephemeral",
        "freshwater",
        "временный водоток",
        "ephemeral stream",
        (140, 170, 200),
    ),
    _B("water_lake_fresh_open", "freshwater", "пресное озеро", "freshwater lake", (60, 130, 205)),
    _B("water_lake_saline", "freshwater", "солёное озеро", "saline lake", (120, 175, 195)),
    _B("water_lake_seasonal", "freshwater", "временное озеро", "seasonal lake", (130, 170, 205)),
    _B(
        "water_shallow_marsh",
        "freshwater",
        "прибрежное мелководье",
        "shallow marsh water",
        (100, 160, 190),
    ),
    _B("marine_shelf_shallow", "marine", "мелководный шельф", "shallow shelf", (110, 185, 215)),
    _B("marine_shelf_deep", "marine", "внешний шельф", "outer shelf", (80, 150, 200)),
    _B(
        "marine_slope_benthic",
        "marine",
        "континентальный склон",
        "continental slope",
        (50, 105, 170),
    ),
    _B("marine_abyssal_benthic", "marine", "абиссаль", "abyssal plain", (25, 55, 115)),
    _B("marine_pelagic_open", "marine", "открытый океан", "open ocean", (30, 70, 140)),
    _B(
        "marine_polar_ice_margin",
        "marine",
        "приполярная кромка льда (потенциал)",
        "polar ice margin (potential)",
        (175, 200, 225),
    ),
    _B(
        "marine_coral_reef_potential",
        "marine",
        "пригодность для кораллов (потенциал)",
        "coral reef (potential)",
        (90, 200, 200),
    ),
    _B("marine_estuary_transition", "marine", "эстуарий", "estuary", (90, 160, 180)),
    _B(
        "lake_unknown_salinity",
        "freshwater",
        "озеро (солёность неизвестна)",
        "lake (salinity unknown)",
        (90, 145, 200),
    ),
    _B(
        "marine_hadal_trench",
        "marine",
        "глубоководный жёлоб (хадаль)",
        "hadal trench",
        (10, 25, 70),
    ),
)
INDEX = {b.id: i for i, b in enumerate(BIOMES)}
FAMILIES = tuple(dict.fromkeys(b.family for b in BIOMES))
BIOME_RGB = np.array([b.rgb for b in BIOMES], dtype=np.uint8)


def code(biome_id: str) -> int:
    return INDEX[biome_id]


def family_codes() -> np.ndarray:
    """Family index of every biome code."""
    return np.array([FAMILIES.index(b.family) for b in BIOMES], dtype=np.uint8)


#: reasons a unit is limited / barren (bit flags, stable order, append-only)
LIMITING_FACTORS = (
    "cold",  # too little warmth: GDD / coldest month
    "short_season",
    "drought",
    "seasonal_drought",
    "soil_shallow",
    "rock",
    "salinity",
    "flooding",
    "snow",
    "ice",
    "young_volcanic",
    "erosion",
    "hyperarid_heat",
    "cold_polar_dry",
    "deep_water",
)
LF_BIT = {name: 1 << i for i, name in enumerate(LIMITING_FACTORS)}

#: local habitat modifiers (bit flags, append-only)
MODIFIERS = (
    "river_oasis",
    "saline_soil",
    "volcanic_ash_young",
    "montane_cloud_belt",
    "scree",
    "cold_air_pool",
    "north_slope_solar_shadow",
    "seasonal_flood",
    "permafrost_polygon_proxy",
    "coastal_salt_spray",
    "fog_oasis",
    "peat_accumulation",
    "fire_prone",
    "glacial_moraine",
)
MOD_BIT = {name: 1 << i for i, name in enumerate(MODIFIERS)}


def decode_bits(value: int, names) -> list[str]:
    return [n for i, n in enumerate(names) if int(value) >> i & 1]


def validate_registry() -> list[str]:
    """Distinct IDs, valid colours, every family known; returns the problems found."""
    errors = []
    ids = [b.id for b in BIOMES]
    if len(set(ids)) != len(ids):
        errors.append("duplicate biome IDs")
    for b in BIOMES:
        if len(b.rgb) != 3 or not all(0 <= c <= 255 for c in b.rgb):
            errors.append(f"bad colour of {b.id}")
        if not b.ru or not b.en:
            errors.append(f"missing name of {b.id}")
    return errors
