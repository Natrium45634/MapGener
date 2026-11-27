"""Plain-language description of a hex: what the biome is, what the conditions are like,
and the few facts that matter most (shown first in the side panel). Qt-free."""

from __future__ import annotations

#: one or two sentences per biome of the registry (``biomes.registry.BIOMES``)
BIOME_TEXT = {
    "unresolved": "Данных для определения биома не хватило.",
    "desert_hot_sand_erg": "Море песчаных дюн в жарком сухом климате; растений почти нет, вода — "
    "только в редких оазисах.",
    "desert_hot_gravel_reg": "Ровная щебнистая равнина жаркой пустыни, отполированная ветром; "
    "редкие кустики по сухим руслам.",
    "desert_hot_rock_hamada": "Каменистое плато жаркой пустыни: голые скалы и россыпи камня, "
    "растительность лишь в трещинах.",
    "desert_hot_salt_flat": "Плоская солончаковая пустошь на месте высохших вод; белая соляная "
    "корка, жизнь почти невозможна.",
    "desert_playa_dry_lake": "Дно временного озера (такыр): плотная растрескавшаяся глина, "
    "после редких дождей ненадолго заполняется водой.",
    "desert_badlands_barren": "Бедленды — изрезанные оврагами голые склоны из мягких пород, "
    "которые быстро размывают редкие ливни.",
    "desert_hot_fog_coast": "Прибрежная пустыня, где почти не бывает дождей, но часты туманы с "
    "холодного моря; растения пьют влагу из тумана.",
    "desert_saline_crust": "Сухой солончак с соляной коркой на поверхности; растут лишь "
    "солеустойчивые кустарники.",
    "desert_semi_arid_sparse": "Полупустыня: редкие кустарники и злаки, дожди скудные и "
    "нерегулярные.",
    "desert_temperate_cold": "Холодная пустыня умеренного пояса: сухо круглый год, жаркое лето и "
    "морозная зима, полынь и солянки.",
    "desert_high_mountain_cold": "Высокогорная холодная пустыня: разрежённый воздух, сильные "
    "морозы и сухость, растительность подушками.",
    "barren_volcanic_recent": "Свежие лавовые поля и пепел недавних извержений; жизнь только "
    "начинает их заселять.",
    "barren_rock_cliff": "Голые скалы, обрывы и осыпи, где почва не удерживается.",
    "barren_erosional_surface": "Поверхность, которую быстро размывает: почва не успевает "
    "сформироваться.",
    "barren_gypsum_alkali": "Гипсовые и щелочные пустоши с ядовитыми для большинства растений "
    "солями.",
    "desert_sparse_undifferentiated": "Пустыня с очень редкой растительностью.",
    "forest_tropical_everwet": "Влажный тропический лес: жарко и дождливо круглый год, "
    "многоярусный полог, огромное разнообразие видов.",
    "forest_tropical_monsoon": "Муссонный лес: тропическое тепло, ливни в сезон муссона и сухие "
    "месяцы, когда часть деревьев сбрасывает листву.",
    "forest_tropical_seasonal_dry": "Сезонно-сухой тропический лес: долгая засуха, деревья "
    "сбрасывают листья до прихода дождей.",
    "forest_tropical_cloud_montane": "Горный облачный лес: склоны в постоянных туманах, деревья "
    "в мхах и эпифитах.",
    "forest_tropical_swamp": "Заболоченный тропический лес на постоянно влажных низинах.",
    "forest_tropical_floodplain": "Галерейный лес вдоль тропических рек, ежегодно заливаемый "
    "паводками.",
    "forest_tropical_mangrove": "Мангровые леса на тропических побережьях: деревья на "
    "ходульных корнях в солоноватой воде приливов.",
    "forest_subtropical_evergreen": "Субтропический вечнозелёный лес: мягкая зима, тёплое "
    "влажное лето.",
    "forest_subtropical_seasonal": "Субтропический сезонный лес с сухим периодом.",
    "woodland_tropical_dry": "Сухое тропическое редколесье: разреженные деревья среди трав, "
    "долгий сухой сезон.",
    "forest_temperate_deciduous": "Широколиственный лес умеренного пояса: дуб, бук, клён; "
    "листопад осенью, плодородные почвы.",
    "forest_temperate_mixed": "Смешанный лес из хвойных и лиственных пород умеренного пояса.",
    "forest_temperate_evergreen_broadleaf": "Вечнозелёный широколиственный лес тёплого "
    "умеренного климата с мягкой зимой.",
    "forest_temperate_conifer": "Хвойный лес умеренного пояса: сосна, ель, пихта.",
    "forest_temperate_rainforest": "Умеренный дождевой лес: очень влажное прохладное побережье, "
    "гигантские хвойные и мхи.",
    "forest_mediterranean_sclerophyll": "Средиземноморский лес из жестколистных деревьев: сухое "
    "жаркое лето и дождливая мягкая зима.",
    "shrubland_mediterranean": "Маквис и чапараль: густые колючие вечнозелёные кустарники "
    "средиземноморского климата, частые пожары.",
    "woodland_temperate_open": "Светлое умеренное редколесье с полянами.",
    "forest_montane_temperate": "Горный лес умеренного пояса на склонах.",
    "forest_floodplain_temperate": "Пойменный лес вдоль рек умеренного пояса, заливаемый "
    "весенними паводками.",
    "forest_wetland_temperate": "Заболоченный лес на сырых низинах.",
    "forest_boreal_dark_conifer": "Темнохвойная тайга: ель, пихта, кедр; долгая снежная зима, "
    "короткое прохладное лето.",
    "forest_boreal_light_conifer": "Светлохвойная тайга: сосновые леса на песках и сухих "
    "почвах холодного пояса.",
    "forest_boreal_deciduous_larch": "Лиственничная тайга: самые суровые зимы, вечная мерзлота; "
    "лиственница сбрасывает хвою на зиму.",
    "forest_boreal_mixed": "Смешанная тайга из хвойных, берёзы и осины.",
    "woodland_boreal_sparse": "Северное редколесье: низкорослые деревья на границе леса.",
    "forest_tundra_mosaic": "Лесотундра: островки низкого леса среди тундры.",
    "boreal_peat_swamp_forest": "Заболоченная тайга на торфяниках.",
    "savanna_tree_grass": "Древесная саванна: высокие травы и отдельные деревья, дождливый и "
    "сухой сезоны, пожары.",
    "savanna_shrub_grass": "Кустарниковая саванна: травы с кустарниками, долгий сухой сезон.",
    "savanna_open_grass": "Открытая травяная саванна почти без деревьев.",
    "woodland_dry_tropical": "Сухое тропическое редколесье (мёомбо): деревья с плоскими кронами "
    "среди трав.",
    "grassland_temperate_prairie": "Прерия, луговая степь: высокие травы, тёплое лето, "
    "холодная зима, очень плодородные чернозёмы.",
    "grassland_steppe": "Степь: травы и злаки без деревьев, сухое лето и морозная зима.",
    "grassland_dry_steppe": "Сухая степь: низкие разреженные злаки, засухи часты.",
    "forest_steppe_mosaic": "Лесостепь: перелески среди степных лугов.",
    "shrubland_temperate_dry": "Сухие кустарники умеренного пояса.",
    "scrub_subtropical": "Субтропический кустарник на сухих склонах.",
    "grassland_montane": "Горные луга на склонах выше леса или между лесами.",
    "grassland_alpine": "Альпийские луга: низкие травы и цветы высокогорья, короткое лето.",
    "steppe_alpine": "Высокогорная степь: сухо и холодно, разреженные злаки.",
    "meadow_subalpine": "Субальпийские луга у верхней границы леса: высокотравье.",
    "heath_alpine": "Высокогорная пустошь с вереском и кустарничками.",
    "scree_alpine_sparse": "Осыпи высокогорья с редкими растениями между камней.",
    "tundra_shrub": "Кустарниковая тундра: карликовые берёзы и ивы, вечная мерзлота.",
    "tundra_sedge_graminoid": "Осоковая тундра: влажные осоковые луга на мерзлоте.",
    "tundra_moss_lichen": "Мохово-лишайниковая тундра: самый холодный край растительности.",
    "tundra_alpine": "Горная тундра выше границы леса.",
    "tundra_polar_desert_sparse": "Полярная пустыня: мороз и сухость, лишь мхи и лишайники.",
    "barren_polar_rock": "Голый полярный камень без растительности.",
    "barren_nunatak": "Нунатак — скалистая вершина, торчащая из ледника.",
    "snow_perennial_field": "Многолетнее снежное поле, не тающее летом.",
    "firn_field": "Фирновое поле: уплотнённый многолетний снег на пути ко льду.",
    "glacier_mountain": "Горный ледник, стекающий по долине.",
    "glacier_ice_cap": "Ледниковая шапка, покрывающая возвышенность.",
    "glacier_ice_sheet": "Материковый ледниковый щит толщиной в сотни и тысячи метров.",
    "glacier_ablation_zone": "Нижняя часть ледника, где лёд тает и испаряется быстрее, чем "
    "нарастает.",
    "wetland_fresh_marsh": "Пресноводное травяное болото: тростник, рогоз, открытая вода.",
    "wetland_fen": "Низинное болото, питаемое грунтовыми водами; богатое растениями.",
    "wetland_bog": "Верховое болото на торфе, питаемое только дождями; кислое и бедное.",
    "wetland_peatland_mosaic": "Мозаика торфяных болот и озерков.",
    "wetland_swamp_forest": "Заболоченный лес, где вода стоит большую часть года.",
    "wetland_floodplain_seasonal": "Сезонная пойма, заливаемая паводками.",
    "wetland_riparian_gallery": "Приречная галерейная растительность вдоль русел.",
    "wetland_inland_saline": "Внутреннее солёное болото в бессточной впадине.",
    "wetland_saltmarsh_tidal": "Приморский солёный марш, заливаемый приливами.",
    "wetland_mangrove_intertidal": "Мангровая литораль тропических берегов.",
    "wetland_unclassified": "Болото, тип которого не определён.",
    "water_river_channel": "Русло крупной реки.",
    "water_river_ephemeral": "Временный водоток, пересыхающий в сухой сезон.",
    "water_lake_fresh_open": "Пресное озеро.",
    "water_lake_saline": "Солёное озеро бессточной котловины.",
    "water_lake_seasonal": "Временное озеро, появляющееся в сезон дождей.",
    "water_shallow_marsh": "Прибрежное мелководье с водной растительностью.",
    "marine_shelf_shallow": "Мелководный шельф: светлое тёплое море до ~50 м, богатое жизнью.",
    "marine_shelf_deep": "Внешний шельф глубиной до ~200 м.",
    "marine_slope_benthic": "Континентальный склон: дно круто уходит в глубину.",
    "marine_abyssal_benthic": "Абиссаль: холодное тёмное дно океана на глубине нескольких км.",
    "marine_pelagic_open": "Открытый океан вдали от берегов.",
    "marine_polar_ice_margin": "Кромка морского льда: богатая жизнью граница льда и воды.",
    "marine_coral_reef_potential": "Тёплое прозрачное мелководье, пригодное для коралловых рифов.",
    "marine_estuary_transition": "Эстуарий: устье реки, где пресная вода смешивается с морской.",
    "lake_unknown_salinity": "Озеро (солёность не определена).",
    "marine_hadal_trench": "Глубоководный жёлоб глубже 6 км — самые глубокие места океана.",
    "tropical_montane_paramo": "Парамо: холодное влажное тропическое высокогорье с "
    "гигантскими розеточными растениями.",
    "oceanic_heathland": "Океанические верещатники: ветреные влажные пустоши с вереском.",
    "tropical_thorn_scrub": "Колючие кустарники (каатинга): жаркий край с долгой засухой.",
    "coastal_dune_vegetation": "Растительность прибрежных дюн, закрепляющая песок.",
    "marine_kelp_forest": "Ламинариевые леса: подводные леса водорослей в холодном "
    "прибрежном море.",
    "marine_seagrass_meadow": "Луга морских трав на мелководье.",
    "marine_hydrothermal_field": "Гидротермальное поле дна: горячие источники и особая жизнь "
    "без солнечного света.",
    "marine_ice_shelf_floating": "Шельфовый ледник: плавучая плита материкового льда.",
    "marine_sea_ice_multiyear": "Многолетний морской лёд, не тающий летом.",
}


def _k(v: float) -> str:
    """A whole number with a space between thousands (1 461)."""
    return f"{v:,.0f}".replace(",", "\u00a0")


def _temp_word(t: float) -> str:
    if t >= 28:
        return "очень жаркое"
    if t >= 22:
        return "жаркое"
    if t >= 16:
        return "тёплое"
    if t >= 10:
        return "прохладное"
    if t >= 3:
        return "холодное"
    return "морозное"


def _winter_word(t: float) -> str:
    if t >= 18:
        return "зимы нет (тепло круглый год)"
    if t >= 8:
        return "мягкая зима"
    if t >= 0:
        return "прохладная зима без сильных морозов"
    if t >= -10:
        return "морозная зима"
    if t >= -25:
        return "суровая зима"
    return "очень суровая зима"


def _wet_word(p: float, ai: float | None) -> str:
    if ai is not None:
        if ai < 0.05:
            return "крайне сухо"
        if ai < 0.2:
            return "засушливо"
        if ai < 0.5:
            return "полузасушливо"
        if ai < 0.75:
            return "умеренно сухо"
        if ai < 1.5:
            return "достаточно влажно"
        return "очень влажно"
    return "сухо" if p < 300 else ("умеренно влажно" if p < 1000 else "влажно")


def describe(project, col: int, row: int, info: dict | None = None) -> dict:
    """``{"biome", "biome_text", "conditions": [sentences], "facts": [(label, value)]}``."""
    info = info if info is not None else project.hex_info(col, row)
    bio = info.get("biomes") or {}
    comp = bio.get("composition") or []
    biome_id = comp[0]["biome"] if comp else None
    biome = comp[0]["title"] if comp else None
    text = BIOME_TEXT.get(biome_id, "") if biome_id else ""
    sea = info.get("sea_level_m", 0.0)
    z = info.get("elevation_m")
    ocean = bool(info.get("ocean_mask"))
    cond: list[str] = []
    facts: list[tuple[str, str]] = []
    if biome:
        share = comp[0]["fraction"]
        facts.append(
            (
                "Биом",
                f"{biome[:1].upper()}{biome[1:]}"
                + (f" ({100 * share:.0f} %)" if share < 0.95 else ""),
            )
        )
        if len(comp) > 1 and comp[1]["fraction"] >= 0.15:
            facts.append(("Также", f"{comp[1]['title']} ({100 * comp[1]['fraction']:.0f} %)"))
    if z is not None:
        rel = z - sea
        facts.append(("Глубина" if ocean else "Высота", f"{abs(rel):,.0f} м"))
    t = info.get("temperature_mean_c")
    tw, tc = info.get("warmest_month_c"), info.get("coldest_month_c")
    p = info.get("precipitation_mm_yr")
    ai = info.get("aridity_p_over_pet", info.get("humidity_index"))
    if t is not None:
        facts.append(
            (
                "Температура",
                f"{t:+.0f} °C в среднем"
                + (f" (от {tc:+.0f} до {tw:+.0f})" if tw is not None and tc is not None else ""),
            )
        )
    if p is not None:
        facts.append(("Осадки", f"{_k(p)} мм в год"))
    if not ocean and tw is not None and tc is not None and p is not None:
        cond.append(
            f"Лето {_temp_word(tw)} (самый тёплый месяц {tw:+.0f} °C), "
            f"{_winter_word(tc)} (самый холодный {tc:+.0f} °C)."
        )
        dry = info.get("dry_month_count")
        wet = _wet_word(p, ai)
        tail = f", сухих месяцев: {dry:.0f}" if dry else ""
        cond.append(f"{wet[:1].upper()}{wet[1:]}: {_k(p)} мм осадков в год{tail}.")
        snow = info.get("snow_cover_months")
        if snow and snow >= 1:
            cond.append(f"Снег лежит около {snow:.0f} мес. в году.")
    if bio.get("climate_zone_koppen"):
        facts.append(("Климат (Кёппен)", str(bio["climate_zone_koppen"])))
    if not ocean and z is not None:
        slope = info.get("slope_deg")
        relief = info.get("hyps_relief_m") or info.get("local_relief_m") or 0
        rel = z - sea
        if rel > 3000:
            form = "высокогорье"
        elif relief > 1500 or (slope or 0) > 12:
            form = "горы"
        elif relief > 500 or (slope or 0) > 5:
            form = "холмы и низкие горы"
        elif rel < 0:
            form = "впадина ниже уровня моря"
        else:
            form = "равнина"
        cond.append(f"Рельеф: {form}, перепад высот в гексе около {_k(relief)} м.")
    if info.get("river_mask"):
        q = info.get("discharge_m3_s")
        cond.append("Через гекс течёт река" + (f" (расход около {_k(q)} м³/с)" if q else "") + ".")
    if (info.get("lake_id") or -1) >= 0:
        cond.append("Здесь есть озеро.")
    res = getattr(project, "resources", None)
    if res is not None and not ocean:
        import numpy as np

        L = res.layers
        fert = float(L["soil_fertility_potential"][row, col])
        if np.isfinite(fert):
            word = (
                "очень плодородная"
                if fert > 0.75
                else "плодородная"
                if fert > 0.55
                else "средняя"
                if fert > 0.35
                else "бедная"
            )
            facts.append(("Почва", f"{word} ({fert:.2f})"))
            from .resources.soil import FACTORS

            lim = int(L["soil_limiting_factor"][row, col])
            lim_ru = {
                "nutrient": "бедность питательными веществами",
                "water": "нехватка воды",
                "ph": "кислотность",
                "oxygen": "переувлажнение",
                "rooting": "тонкий или каменистый слой",
                "salinity": "засоление",
            }
            if lim < len(FACTORS):
                cond.append(f"Почва {word}; главное ограничение — {lim_ru[FACTORS[lim]]}.")
        if bool(L["surface_occurrence_mask"][row, col]):
            sy = [
                res.system(b["system_id"]) for b in res.bodies_at(row * project.layout.width + col)
            ]
            names = sorted({s["title"] for s in sy if s and s["surface_occurrence"]})
            if names:
                facts.append(("Ископаемые на поверхности", ", ".join(names[:3])))
    vol = getattr(project, "volcanism", None)
    if vol is not None and float(vol.layers["volcanic_hazard_index"][row, col]) > 0.3:
        cond.append("Рядом действующий вулкан — опасность извержений.")
    return {
        "biome": biome,
        "biome_id": biome_id,
        "biome_text": text,
        "conditions": cond,
        "facts": facts,
    }
