"""Everything known about one hex, as titled sections of (label, value) rows.

Qt-free: the GUI side panel, the CLI (``hexworld hex``) and the tests all use it.
Rows whose value is unknown are left out; sections without rows are dropped.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .project import Project

MONTHS = ("янв", "фев", "мар", "апр", "май", "июн", "июл", "авг", "сен", "окт", "ноя", "дек")

_MOUNTAIN_CLASS = ("нет", "гребень хребта", "склон", "перевал", "долина", "предгорье", "вулкан")
_BOUNDARY = {
    "none": "нет",
    "convergent": "сходящаяся (столкновение / субдукция)",
    "divergent": "расходящаяся (рифт / срединный хребет)",
    "transform": "трансформная (сдвиг)",
    "quiet": "спокойная",
}
_CRUST = {"oceanic": "океаническая", "continental": "континентальная"}
_MARGIN = {"none": "", "passive": "пассивная", "active": "активная", "transform": "трансформная"}
_SEASONS = ("декабрь–февраль", "март–май", "июнь–август", "сентябрь–ноябрь")
_COVER = {
    "vegetation": "растительность",
    "bare_soil": "голая почва",
    "rock": "скалы",
    "sand": "пески",
    "salt": "соль",
    "snow_perennial": "многолетний снег",
    "firn": "фирн",
    "glacier_ice": "ледниковый лёд",
    "open_water": "открытая вода",
}
_FACTORS = {
    "cold": "холод",
    "short_season": "короткий сезон роста",
    "drought": "засуха",
    "seasonal_drought": "сезонная засуха",
    "soil_shallow": "маломощная почва",
    "rock": "скалы",
    "salinity": "засоление",
    "flooding": "переувлажнение",
    "snow": "снег",
    "ice": "лёд",
    "young_volcanic": "молодые вулканические породы",
    "erosion": "эрозия",
    "hyperarid_heat": "жаркая сверхсухость",
    "cold_polar_dry": "полярная сухость",
    "deep_water": "глубокая вода",
}
_MODIFIERS = {
    "river_oasis": "речной оазис",
    "saline_soil": "засолённые почвы",
    "volcanic_ash_young": "свежий вулканический пепел",
    "montane_cloud_belt": "пояс горных облаков",
    "scree": "осыпи",
    "cold_air_pool": "застой холодного воздуха",
    "north_slope_solar_shadow": "теневые склоны",
    "seasonal_flood": "сезонные разливы",
    "permafrost_polygon_proxy": "мерзлотные полигоны",
    "coastal_salt_spray": "морские брызги",
    "fog_oasis": "туманный оазис",
    "peat_accumulation": "торфонакопление",
    "fire_prone": "пожароопасность",
    "glacial_moraine": "морены",
    "geothermal_habitat": "геотермальные местообитания",
    "volcanic_degassing_stress": "угнетение вулканическими газами",
    "hydrothermal_vent_field": "гидротермальные источники дна",
}
_MAGMA_CLASS = {
    "basaltic": "базальтовая", "basaltic_andesitic": "андезибазальтовая",
    "andesitic": "андезитовая", "dacitic": "дацитовая", "rhyolitic": "риолитовая",
}  # fmt: skip
_VOLC_STATUS = {"active_system": "действующая", "dormant_system": "спящая",
                "uncertain": "неясный статус", "extinct_system": "потухшая"}  # fmt: skip
_VOLC_PHASE = {"erupting": "извергается", "waning": "затухает", "recovering": "после извержения",
               "unrest": "волнение (накопление магмы)", "degassing": "дегазация",
               "quiet": "спокойствие"}  # fmt: skip
_VOLC_EDIFICE = {
    "stratocone": "стратовулкан", "shield": "щитовой вулкан", "caldera_complex": "кальдера",
    "lava_dome_complex": "экструзивные купола", "scoria_cone": "поле шлаковых конусов",
    "fissure_field": "трещинное поле", "submarine_seamount": "подводный вулкан",
    "subglacial_tuya": "подлёдный вулкан (туйя)", "maar_tuff_ring": "маар",
    "mixed": "смешанная постройка",
}  # fmt: skip
_VOLC_STYLE = {
    "hawaiian_effusive": "гавайское излияние", "fissure_effusive": "трещинное излияние",
    "submarine_effusive": "подводное излияние", "strombolian": "стромболианское",
    "vulcanian": "вулканское", "subplinian": "субплинианское", "plinian": "плинианское",
    "ultraplinian": "ультраплинианское", "dome_growth_collapse": "рост и обрушение купола",
    "phreatomagmatic_surtseyan": "фреатомагматическое (суртсейское)", "phreatic": "фреатическое",
}  # fmt: skip
_ICE = {
    "none": "",
    "glacier_mountain": "горный ледник",
    "glacier_ice_cap": "ледниковая шапка",
    "glacier_ice_sheet": "ледниковый щит",
}
_REALM = {
    "terrestrial": "суша",
    "inland_water": "внутренние воды",
    "marine": "море",
    "transitional": "переходная зона",
    "unknown": "—",
}


@dataclass
class Section:
    key: str
    title: str
    rows: list[tuple[str, str]] = field(default_factory=list)

    def add(self, label: str, value) -> None:
        if value is None or value == "":
            return
        self.rows.append((label, str(value)))


@dataclass
class HexReport:
    col: int
    row: int
    title: str
    subtitle: str
    sections: list[Section]
    #: Monthly climate for the climograph: {"t": [12], "p": [12], "pet": [12]} or None.
    climate: dict | None = None
    #: Plain-language card (``hex_description.describe``): biome, its text, conditions,
    #: key facts; ``None`` when not available.
    description: dict | None = None

    def section(self, key: str) -> Section | None:
        return next((s for s in self.sections if s.key == key), None)

    def as_text(self) -> str:
        out = [self.title, self.subtitle, ""]
        d = self.description or {}
        if d.get("biome"):
            out.append(d["biome"][:1].upper() + d["biome"][1:])
        if d.get("biome_text"):
            out.append(d["biome_text"])
        out += list(d.get("conditions", []))
        out += [f"{a}: {b}" for a, b in d.get("facts", [])]
        if d:
            out.append("")
        for s in self.sections:
            out.append(f"— {s.title} —")
            width = max((len(a) for a, _b in s.rows), default=0)
            out += [f"  {a:<{width}}  {b}" for a, b in s.rows]
            out.append("")
        return "\n".join(out).rstrip() + "\n"


def _n(v, digits: int = 0, unit: str = "", sign: bool = False):
    if v is None:
        return None
    try:
        f = float(v)
    except (TypeError, ValueError):
        return str(v)
    if f != f:  # NaN
        return None
    s = f"{f:+,.{digits}f}" if sign else f"{f:,.{digits}f}"
    return s.replace(",", " ") + (f" {unit}" if unit else "")


def _pct(v, digits: int = 0):
    return None if v is None else f"{100 * float(v):.{digits}f} %"


def _yes(v) -> str | None:
    return "да" if v else None


def _surface_kind(info: dict) -> str:
    if info.get("ocean_mask") or (
        "ocean_mask" not in info and info.get("depth_m", 0) > 0 and not info.get("land_mask")
    ):
        return "море"
    if info.get("lake_id", -1) >= 0 or info.get("inland_water_mask"):
        return "озеро"
    if info.get("below_sea_level_land_mask"):
        return "суша ниже уровня моря"
    if "land_mask" in info or "macro_elevation_m" in info:
        return "суша"
    return "—"


def hex_report(project: Project, col: int, row: int) -> HexReport:
    p = project
    info = p.hex_info(col, row)
    kind = _surface_kind(info)
    height = info.get("elevation_m", info.get("macro_elevation_m"))
    sea = info.get("sea_level_m", 0.0)
    bio = info.get("biomes")

    # -- headline ------------------------------------------------------------------------------
    title = f"Гекс ({col}, {row}) · {kind}"
    sub = []
    if height is not None:
        rel = height - sea
        sub.append(("высота " if rel >= 0 else "глубина ") + _n(abs(rel), 0, "м"))
    if bio:
        top = bio["composition"][0] if bio["composition"] else None
        if top:
            sub.append(top["title"])
    if "temperature_mean_c" in info:
        sub.append(f"{info['temperature_mean_c']:+.1f} °C")
    if info.get("precipitation_mm_yr") is not None:
        sub.append(_n(info["precipitation_mm_yr"], 0, "мм/год"))
    sections: list[Section] = []

    # -- location ------------------------------------------------------------------------------
    s = Section("place", "Положение")
    s.add("Гекс (столбец, строка)", f"{col}, {row}")
    s.add("Координаты", f"x {_n(info['x_km'], 0)} км, y {_n(info['y_km'], 0)} км")
    s.add("Площадь гекса", _n(p.layout.hex_area_km2, 0, "км²"))
    if "latitude_deg" in info:
        lat = info["latitude_deg"]
        s.add("Широта", f"{abs(lat):.1f}° {'с. ш.' if lat >= 0 else 'ю. ш.'}")
    s.add("Тип поверхности", kind)
    if height is not None:
        s.add("Высота над морем" if height >= sea else "Глубина", _n(abs(height - sea), 0, "м"))
    s.add("Уровень моря (модели)", _n(sea, 0, "м", sign=True))
    landmass = _landmass(p, info)
    s.add("Материк / остров", landmass)
    river = _names().river_name_at(p, col, row) if info.get("river_mask") else None
    if p.geo is not None:
        block = info.get("continental_block_id", -1)
        if block is not None and block >= 0:
            s.add("Континентальный блок", p.geo.blocks[block]["name"])
    sections.append(s)

    # -- relief --------------------------------------------------------------------------------
    s = Section("relief", "Рельеф (высоты над уровнем моря)")

    def above(key):
        v = info.get(key)
        return None if v is None else _n(v - sea, 0, "м", sign=True)

    if "pre_erosion_elevation_m" in info:
        s.add("Высота после эрозии", above("elevation_m"))
        s.add("Высота до эрозии", above("pre_erosion_elevation_m"))
    elif "elevation_m" in info:
        s.add("Высота (геоистория)", above("elevation_m"))
    if "macro_elevation_m" in info and p.geo is not None:
        s.add(
            "Исходная макровысота (v2)",
            _n(info["macro_elevation_m"] - p.geo.sea_level_m, 0, "м", sign=True),
        )
    s.add("Коренные породы (без рыхлого чехла)", above("bedrock_elevation_m"))
    s.add("Уклон", _n(info.get("slope_deg"), 1, "°") or _n(info.get("slope_m_per_km"), 1, "м/км"))
    s.add(
        "Перепад высот в гексе", _n(info.get("hyps_relief_m") or info.get("local_relief_m"), 0, "м")
    )
    if info.get("hyps_q05_m") is not None:
        s.add(
            "Высоты внутри гекса",
            f"5 % {_n(info['hyps_q05_m'] - sea, 0)} · медиана {_n(info['hyps_q50_m'] - sea, 0)}"
            f" · 95 % {_n(info['hyps_q95_m'] - sea, 0)} м",
        )
    if (info.get("prominence_m") or 0) > 0:
        s.add("Вершина: относительная высота", _n(info["prominence_m"], 0, "м"))
    mc = info.get("mountain_class")
    if mc:
        s.add("Форма", _MOUNTAIN_CLASS[mc] if mc < len(_MOUNTAIN_CLASS) else mc)
    sys_id = info.get("mountain_system_id", -1)
    if sys_id is not None and sys_id >= 0:
        s.add("Горная система", _mountain_system(p, sys_id))
        mname = _names().feature_names(p).get("mountains", {}).get(sys_id)
        if mname:
            s.rows[-1] = ("Горная система", f"горы {_names().ru(mname)} — " + s.rows[-1][1])
        s.add("Возраст горообразования", _n(info.get("orogen_age_ma"), 0, "млн лет"))
    if (info.get("dissection_index") or 0) > 0:
        s.add("Расчленённость", _n(info["dissection_index"], 2))
    if info.get("aspect_deg") is not None and (info.get("slope_deg") or 0) > 1:
        s.add("Экспозиция склона", _aspect(info["aspect_deg"]))
    sections.append(s)

    # -- geology -------------------------------------------------------------------------------
    s = Section("geology", "Тектоника и геология")
    plate = p.plate(info["plate_id"])
    s.add("Плита", f"{plate['name']} (№ {plate['id']}, {plate.get('size_class', '')})")
    vx, vy = plate["velocity_mm_yr"]
    s.add("Движение плиты", f"{(vx * vx + vy * vy) ** 0.5:.1f} мм/год ({_direction(vx, vy)})")
    if info.get("plate_id_epoch") is not None and info["plate_id_epoch"] != info["plate_id"]:
        s.add("Плита в эпоху снимка", p.plate(info["plate_id_epoch"])["name"])
    s.add("Граница плит рядом", _BOUNDARY.get(info.get("boundary"), info.get("boundary")))
    s.add("До границы плит", _n(info.get("boundary_distance_km"), 0, "км"))
    s.add("Кора", _CRUST.get(info.get("crust"), info.get("crust")))
    s.add("Доля континентальной коры", _pct(info.get("continental_fraction")))
    s.add(
        "Возраст коры",
        _n(info.get("crust_age_ma") or info.get("oceanic_crust_age_ma"), 0, "млн лет"),
    )
    if "rock" in info:
        from .views import _ROCK_TITLES

        s.add("Порода", _ROCK_TITLES[info["rock_type"]])
        s.add("Сопротивление эрозии", _n(info.get("rock_resistance"), 2))
    s.add("Скорость поднятия", _n(info.get("uplift_rate_mm_yr"), 3, "мм/год", sign=True))
    s.add("Поднято за историю", _n(info.get("uplift_total_m"), 0, "м", sign=True))
    s.add("Снято эрозией за историю", _n(info.get("erosion_total_m"), 0, "м"))
    s.add("Осадки (мощность)", _n(info.get("sediment_thickness_m"), 0, "м"))
    vid = info.get("volcano_id", -1)
    if vid is not None and vid >= 0:
        s.add("Вулкан", _volcano(p, vid, info))
    s.add("Потенциал вулканизма", _n(info.get("volcanic_potential"), 2))
    s.add("Сейсмичность (потенциал)", _n(info.get("seismicity"), 2))
    if info.get("new_crust"):
        s.add("Новая кора", "образовалась за время истории")
    sections.append(s)

    # -- climate -------------------------------------------------------------------------------
    climate = None
    if "temperature_mean_c" in info:
        s = Section("climate", "Климат")
        s.add("Средняя температура", _n(info["temperature_mean_c"], 1, "°C", sign=True))
        s.add("Самый тёплый месяц", _n(info.get("warmest_month_c"), 1, "°C", sign=True))
        s.add("Самый холодный месяц", _n(info.get("coldest_month_c"), 1, "°C", sign=True))
        s.add("Зима (дек–фев)", _n(info.get("temperature_djf_c"), 1, "°C", sign=True))
        s.add("Лето (июн–авг)", _n(info.get("temperature_jja_c"), 1, "°C", sign=True))
        s.add("Осадки", _n(info.get("precipitation_mm_yr"), 0, "мм/год"))
        s.add("Испаряемость (PET)", _n(info.get("potential_evapotranspiration_mm_yr"), 0, "мм/год"))
        s.add("Испарение (AET)", _n(info.get("actual_evapotranspiration_mm_yr"), 0, "мм/год"))
        ai = info.get("aridity_p_over_pet", info.get("humidity_index"))
        s.add("Увлажнение P/PET", None if ai is None else f"{ai:.2f} — {_aridity_class(ai)}")
        s.add("Сезонность осадков (CV)", _n(info.get("precip_seasonality_cv"), 0, "%"))
        s.add("Доля снега в осадках", _pct(info.get("snow_fraction")))
        s.add("Месяцев со снегом", _n(info.get("snow_cover_months"), 0))
        s.add("Сухих месяцев", _n(info.get("dry_month_count"), 0))
        u, v = info.get("wind_u_ms"), info.get("wind_v_ms")
        if u is not None and v is not None:
            s.add("Ветер", f"{(u * u + v * v) ** 0.5:.1f} м/с, дует {_direction(u, v)}")
        if bio:
            s.add("Климат по Кёппену", bio.get("climate_zone_koppen"))
        s.add("Сумма температур > 5 °C", _n(info.get("gdd5_proxy_c_day"), 0, "°C·сут"))
        s.add("Мерзлота (потенциал)", _n(info.get("permafrost_potential"), 2))
        sections.append(s)
        monthly = info.get("monthly") or {}
        t = monthly.get("temperature_monthly_c")
        pr = monthly.get("precip_monthly_mm")
        if t and pr:
            climate = {"t": t, "p": pr, "pet": monthly.get("pet_monthly_mm")}
            s = Section("months", "По месяцам")
            for i, m in enumerate(MONTHS):
                parts = []
                if t[i] is not None:
                    parts.append(f"{t[i]:+.0f} °C")
                if pr[i] is not None:
                    parts.append(f"осадки {pr[i]:.0f} мм")
                pet = (monthly.get("pet_monthly_mm") or [None] * 12)[i]
                if pet is not None:
                    parts.append(f"PET {pet:.0f}")
                swe = (monthly.get("snowpack_monthly_swe_mm") or [None] * 12)[i]
                if swe:
                    parts.append(f"снег {swe:.0f} мм")
                s.add(m, " · ".join(parts))
            sections.append(s)

    # -- water ---------------------------------------------------------------------------------
    if p.hydro is not None:
        s = Section("water", "Вода: реки, озёра, сток")
        if info.get("river_mask"):
            s.add("Название реки", f"р. {river}" if river else "безымянная (малая) река")
            s.add(
                "Река",
                f"порядок {info.get('strahler_order')}, ширина ~{_n(info.get('river_width_m'), 0)}"
                f" м, глубина ~{_n(info.get('river_depth_m'), 1)} м",
            )
            if info.get("river_mouth"):
                s.add("Устье", "да")
        s.add("Расход воды", _n(info.get("discharge_m3_s"), 1, "м³/с"))
        s.add("Площадь водосбора", _n(info.get("flow_accumulation_area_km2"), 0, "км²"))
        s.add("Сток", _n(info.get("runoff_mm_yr"), 0, "мм/год"))
        peak = info.get("runoff_peak_season")
        if peak is not None and peak < 4:
            s.add("Половодье", _SEASONS[peak])
        s.add("Подземное питание", _pct(info.get("baseflow_index")))
        s.add("Талые воды в стоке", _pct(info.get("snowmelt_share")))
        if info.get("lake_id", -1) >= 0:
            lname = _names().feature_names(p).get("lake", {}).get(info["lake_id"])
            s.add("Озеро", f"оз. {_names().ru(lname)}" if lname else None)
            s.add(
                "Озеро (данные)",
                f"№ {info['lake_id']}, уровень {_n(info.get('lake_level_m'), 0)} м, глубина "
                f"{_n(info.get('lake_depth_m'), 0)} м",
            )
        basin = info.get("basin_id", -1)
        if basin is not None and basin >= 0:
            endo = " (бессточный)" if info.get("endorheic_mask") else ""
            s.add("Речной бассейн", f"№ {basin}{endo}")
            s.add("Доходит до моря", _pct(info.get("outflow_share")))
        s.add("Пойма (потенциал)", _n(info.get("floodplain_potential"), 2))
        s.add("Переувлажнение", _n(info.get("wetness_potential"), 2))
        s.add("Индекс влажности рельефа (TWI)", _n(info.get("twi"), 1))
        s.add("Высота над ближайшей рекой (HAND)", _n(info.get("hand_m"), 0, "м"))
        s.add("Размыв v4", _n(info.get("v4_erosion_m"), 1, "м"))
        s.add("Отложение v4", _n(info.get("v4_deposition_m"), 1, "м"))
        sections.append(s)

    # -- soils ---------------------------------------------------------------------------------
    if p.env is not None and info.get("land_mask") and info.get("lake_id", -1) < 0:
        from .stages.environment import DRAINAGE_TITLES, PARENT_MATERIALS, PARENT_TITLES

        s = Section("soils", "Почвы")
        pm = PARENT_MATERIALS[int(info.get("parent_material_class", 0))]
        s.add("Материнская порода", PARENT_TITLES[pm])
        if info.get("soil_present"):
            s.add(
                "Механический состав",
                f"песок {_pct(info.get('sand_frac'))}, пыль {_pct(info.get('silt_frac'))}, "
                f"глина {_pct(info.get('clay_frac'))}",
            )
            s.add("Мощность почвы", _n(info.get("soil_depth_m"), 1, "м"))
            s.add("Камни", _pct(info.get("coarse_fragment_frac")))
            s.add("Доступная влага (AWC)", _n(info.get("awc_mm"), 0, "мм"))
            dc = int(info.get("drainage_class", 255))
            if dc < len(DRAINAGE_TITLES):
                s.add("Дренаж", DRAINAGE_TITLES[dc])
            s.add("Засоление (оценка)", _n(info.get("soil_salinity_proxy"), 2))
            s.add("Органика (потенциал)", _n(info.get("soil_organic_potential"), 2))
        else:
            s.add("Почва", "нет (скала или лёд)")
        sections.append(s)

    # -- altitudinal belts ---------------------------------------------------------------------
    if "terrain_units" in info and info.get("topo_dry_fraction"):
        from .topoclimate.thermal import BELT_TITLES

        s = Section("belts", "Высотная поясность")
        belt = info.get("dominant_thermal_belt_name", "unknown")
        s.add("Преобладающий пояс", BELT_TITLES.get(belt, belt))
        for key, val in sorted(info.items()):
            if key.startswith("belt_share_") and val and val > 0.005:
                name = "thermal_" + key[len("belt_share_") :]
                s.add(f"  {BELT_TITLES.get(name, name)}", _pct(val))
        s.add("Сезон роста", _n(info.get("gsl_days_mean"), 0, "сут"))
        s.add("Температура сезона роста", _n(info.get("gst_c_mean"), 1, "°C"))
        state = info.get("treeline_state", 0)
        s.add(
            "Граница леса",
            {
                1: f"проходит в гексе, ~{_n(info.get('treeline_proxy_m'), 0)} м",
                2: "вся суша выше границы леса",
                3: "вся суша ниже границы леса",
            }.get(state),
        )
        s.add(
            "Многолетний снег",
            _pct(info.get("perennial_snow_fraction"))
            if info.get("perennial_snow_fraction")
            else None,
        )
        units = [u for u in info["terrain_units"] if u["surface"] == "dry"]
        s.add("Участков рельефа", len(units))
        for u in sorted(units, key=lambda r: r["z_m"] or 0)[:10]:
            s.add(
                f"  {_n(u['z_m'], 0)} м, {u['landform'].replace('_', ' ')}",
                f"{_pct(u['weight_land'])} · {u['t_jan_c']:+.0f}/{u['t_jul_c']:+.0f} °C · "
                f"{_n(u['precip_mm_yr'], 0)} мм",
            )
        sections.append(s)

    # -- biomes --------------------------------------------------------------------------------
    if bio:
        s = Section("biomes", "Биомы и природный покров")
        for c in bio["composition"][:10]:
            s.add(f"  {c['title']}", _pct(c["fraction"], 1))
        s.add("Область", _REALM.get(bio.get("realm"), bio.get("realm")))
        cover = sorted(bio.get("cover", {}).items(), key=lambda kv: -kv[1])
        s.add(
            "Покров поверхности",
            ", ".join(f"{_COVER.get(k, k)} {100 * v:.0f} %" for k, v in cover if v >= 0.01),
        )
        s.add("Растительность", _pct(bio.get("vegetation_fraction")))
        s.add("Сомкнутость крон", _pct(bio.get("tree_canopy_fraction")))
        s.add("Продуктивность (индекс)", _n(bio.get("productivity_index"), 2))
        if bio.get("limiting_factors"):
            s.add(
                "Что ограничивает жизнь",
                ", ".join(_FACTORS.get(f, f) for f in bio["limiting_factors"]),
            )
        if bio.get("habitat_modifiers"):
            s.add(
                "Особенности местообитаний",
                ", ".join(_MODIFIERS.get(m, m) for m in bio["habitat_modifiers"]),
            )
        if bio.get("ice_form") and bio["ice_form"] != "none":
            s.add(
                "Лёд",
                f"{_ICE.get(bio['ice_form'], bio['ice_form'])}, "
                f"{_n(bio.get('ice_thickness_m'), 0)} м",
            )
        if (bio.get("firn_swe_mm") or 0) > 0:
            s.add("Фирн", _n(bio["firn_swe_mm"], 0, "мм в. э."))
        BL = project.biomes.layers if project.biomes is not None else {}
        if "actual_biomass_kg_c_m2" in BL and BL["potential_biomass_kg_c_m2"][row, col] > 0:
            stages = ("голая поверхность", "пионерная стадия", "травы и кустарники",
                      "молодой лес", "зрелое сообщество")  # fmt: skip
            st = stages[int(BL["succession_stage"][row, col])]
            act = float(BL["actual_biomass_kg_c_m2"][row, col])
            pot = float(BL["potential_biomass_kg_c_m2"][row, col])
            s.add("Фактическое состояние", f"{st}; биомасса {act:.1f} из {pot:.1f} кг C/м²")
            fri = float(BL["fire_return_interval_yr"][row, col])
            if fri < 1e5:
                s.add("Пожары", f"примерно раз в {fri:.0f} лет")
        sections.append(s)

    # -- Earth system (v5.3) ------------------------------------------------------------------
    earth = getattr(project, "earth", None)
    if earth is not None:
        _earth_section(project, earth, col, row, sections)
    atm = getattr(project, "atmosphere", None)
    if atm is not None and atm.layers:
        _atmosphere_section(project, atm, col, row, sections)

    # -- volcanism (v5.1) -------------------------------------------------------------------
    volc = getattr(project, "volcanism", None)
    if volc is not None:
        _volcanism_section(project, volc, col, row, sections)

    # -- soil fertility and resources (v5.2) ---------------------------------------------------
    res = getattr(project, "resources", None)
    if res is not None:
        _resources_sections(project, res, col, row, sections)
    if getattr(project, "minerals", None) is not None:
        _minerals_section(project, col, row, sections)
    if getattr(project, "biosphere", None) is not None:
        _biosphere_section(project, col, row, sections)

    # -- coast and sea floor -------------------------------------------------------------------
    s = Section("sea", "Берег и дно моря")
    if info.get("coast_type"):
        from .stages.coasts import COAST_TITLES, COAST_TYPES

        s.add("Тип берега", COAST_TITLES[COAST_TYPES[info["coast_type"]]])
        s.add("Волнение", _n(info.get("wave_exposure"), 2))
        s.add("Прилив", _n(info.get("tidal_range_m"), 1, "м"))
        s.add("Ширина шельфа", _n(info.get("shelf_width_km"), 0, "км"))
        s.add("Температура моря", _n(info.get("sea_surface_temperature_c"), 1, "°C"))
    if info.get("seafloor_province_title"):
        s.add("Провинция дна", info["seafloor_province_title"])
        s.add("Глубина", _n(info.get("water_depth_m") or info.get("depth_m"), 0, "м"))
        if info.get("marine_floor_mask"):
            s.add("Осадки на дне", _n(info.get("marine_sediment_thickness_m"), 0, "м"))
            from .geomorph.margins import MARGIN_NAMES

            mt = info.get("margin_type") or 0
            s.add("Окраина материка", _MARGIN.get(MARGIN_NAMES[mt], "") if mt else None)
            s.add("Зона разломов", _yes(info.get("fracture_zone_mask")))
            s.add("Ось глубоководного жёлоба", _yes(info.get("trench_axis_mask")))
            st = info.get("seamount_type") or 0
            s.add(
                "Подводная гора",
                (None, "подводная гора", "гайот", "вулканический остров")[st] if st < 4 else None,
            )
            s.add("Подводный каньон", _yes(info.get("submarine_canyon_mask")))
    if info.get("shelf_mask"):
        s.add("Шельф", "да")
    sections.append(s)

    # -- basin ---------------------------------------------------------------------------------
    b = info.get("basin")
    if b:
        from .views import _BASIN_TITLES

        s = Section("basin", "Тектонический бассейн")
        s.add("Тип", _BASIN_TITLES.get(b["basin_type"], b["basin_type"]))
        s.add("Опускание фундамента", _n(info.get("basin_subsidence_m"), 0, "м"))
        s.add("Заполнение осадками", _n(info.get("basin_fill_m"), 0, "м"))
        s.add("Причина", b.get("cause"))
        sections.append(s)

    sections = [s for s in sections if s.rows]
    try:
        from .hex_description import describe

        desc = describe(project, col, row, info)
    except Exception:  # noqa: BLE001 - the card is a convenience, never a failure
        desc = None
    return HexReport(col, row, title, " · ".join(sub), sections, climate, desc)


_REALM = {1: "суша", 2: "внутренние воды", 3: "океан"}
_ICE_COVER = {0: None, 1: "сезонный морской лёд", 2: "многолетний морской лёд",
              3: "припай", 4: "озёрный лёд", 5: "ледниковый щит (на суше)",
              6: "шельфовый ледник (плавучий)"}  # fmt: skip


_MONTHS_RU = (
    "январь",
    "февраль",
    "март",
    "апрель",
    "май",
    "июнь",
    "июль",
    "август",
    "сентябрь",
    "октябрь",
    "ноябрь",
    "декабрь",
)


def _atmosphere_section(project, atm, col, row, sections) -> None:
    from .atmosphere.monsoon import REGIMES

    L = atm.layers
    s = Section("atmosphere", "Атмосфера и сезоны (v5.4, нормы)")
    names = {
        "arid": "засушливый",
        "perennially_wet": "влажно круглый год",
        "tropical_monsoon": "тропический муссон",
        "equatorial_bimodal": "два сезона дождей",
        "winter_wet_mediterranean": "дожди зимой (средиземноморский)",
        "summer_wet_non_monsoon": "летние дожди (не муссон)",
        "uniform_or_other": "равномерный / иной",
    }
    land = bool(project.hydro.layers["land_mask"][row, col])
    if land:
        s.add("Режим осадков", names.get(REGIMES[int(L["rain_regime"][row, col])]))
        on = int(L["monsoon_onset_month"][row, col])
        wd = int(L["monsoon_withdrawal_month"][row, col])
        if on:
            s.add("Сезон дождей", f"{_MONTHS_RU[on - 1]} — {_MONTHS_RU[wd - 1]}")
        s.add(
            "Муссонность",
            f"{float(L['monsoon_suitability'][row, col]):.2f} (0…1); "
            f"реверс ветра {float(L['wind_reversal_score'][row, col]):.2f}",
        )
        s.add("Сухих месяцев", str(int(L["dry_season_months"][row, col])))
        s.add(
            "Источник влаги",
            f"~{float(L['moisture_source_distance_km'][row, col]):.0f} км против ветра",
        )
    shares = [
        (k.replace("precip_share_", ""), float(L[k][row, col]))
        for k in L
        if k.startswith("precip_share_")
    ]
    ru = {
        "convective": "конвект.",
        "orographic": "орограф.",
        "frontal": "фронт.",
        "storm": "штормы",
        "background": "фон",
    }
    s.add(
        "Происхождение осадков",
        ", ".join(f"{ru.get(k, k)} {100 * v:.0f} %" for k, v in shares if v >= 0.05),
    )
    if "tc_passage_per_yr" in L:
        f = float(L["tc_passage_per_yr"][row, col])
        if f > 0.01:
            s.add(
                "Тропические циклоны",
                f"{f:.2f} прохождений в год (300 км), "
                f"до {float(L['tc_max_wind_ms'][row, col]):.0f} м/с",
            )
        if "tc_surge_potential_index" in L and float(L["tc_surge_potential_index"][row, col]) > 0:
            s.add(
                "Нагон (потенциал)",
                f"{float(L['tc_surge_potential_index'][row, col]):.2f} — "
                "только индекс, без гидродинамики",
            )
        s.add("Внетропические штормы", f"{float(L['etc_activity'][row, col]):.2f} (0…1)")
    if "weather_annual_anomaly_pct" in L:
        s.add(
            "Погодный год",
            f"осадки {float(L['weather_annual_anomaly_pct'][row, col]):+.0f} % "
            "к норме" + (", засуха" if bool(L["weather_drought"][row, col]) else ""),
        )
    if s.rows:
        sections.append(s)


def _earth_section(project, earth, col, row, sections) -> None:
    E = earth.layers
    s = Section("earth", "Океан, лёд и атмосфера (v5.3)")
    realm = int(E["realm"][row, col])
    s.add("Среда", _REALM.get(realm))
    s.add("Покров льда", _ICE_COVER.get(int(E["ice_cover"][row, col])))
    if realm == 3 and "sst_monthly_c" in E:
        sst = E["sst_monthly_c"][:, row, col].astype(float)
        s.add("Температура моря", f"{sst.min():+.1f} … {sst.max():+.1f} °C (мин./макс. месяца)")
        c = E["sea_ice_concentration_monthly"][:, row, col].astype(float)
        if c.max() > 0.01:
            h = E["sea_ice_thickness_monthly_m"][:, row, col].astype(float)
            s.add(
                "Морской лёд", f"сплочённость до {100 * c.max():.0f} %, толщина до {h.max():.1f} м"
            )
        u = E["current_u_ms"][:, row, col].astype(float).mean()
        v = E["current_v_ms"][:, row, col].astype(float).mean()
        sp = float(np.hypot(u, v))
        if sp > 0.005:
            s.add("Течение", f"{100 * sp:.0f} см/с, {_direction(u, v)}")
        up = float(E["upwelling_m_day"][:, row, col].astype(float).mean())
        if E["coastal_upwelling"][row, col] or up > 0.2:
            s.add("Апвеллинг", f"{up:+.2f} м/сут (подъём холодной воды)")
    if realm == 2 and E["lake_ice_months"][row, col] > 0:
        s.add("Ледостав на озере", f"{int(E['lake_ice_months'][row, col])} мес.")
    if "winds_aloft_u_ms" in E:
        lv = earth.summary.get("levels_hpa", [850, 500, 250])
        parts = []
        for k, p in enumerate(lv):
            u = float(E["winds_aloft_u_ms"][0, k, row, col])
            v = float(E["winds_aloft_v_ms"][0, k, row, col])
            below = bool(E["winds_aloft_below_ground"][0, k, row, col])
            parts.append(f"{int(p)} гПа: " + ("ниже поверхности" if below else
                         f"{np.hypot(u, v):.0f} м/с {_direction(u, v)}"))  # fmt: skip
        s.add("Ветер на высотах (зима)", "; ".join(parts))
    if "land_ice_thickness_m" in E:
        h = float(E["land_ice_thickness_m"][row, col])
        if h > 1.0:
            smb = float(np.nan_to_num(E["land_ice_smb_m_yr"][row, col]))
            s.add("Наземный лёд", f"{h:.0f} м; баланс массы {smb:+.2f} м/год; "
                  f"течение {float(E['ice_velocity_m_yr'][row, col]):.0f} м/год")  # fmt: skip
            w = float(E["bedrock_deflection_m"][row, col])
            if w < -1.0:
                s.add("Прогиб ложа под льдом", f"{w:.0f} м (рельеф v4 не изменён)")
        sh = float(E["shelf_thickness_m"][row, col])
        if sh > 0:
            d = sh * 917.0 / 1028.0
            s.add("Шельфовый ледник", f"{sh:.0f} м (осадка {d:.0f} м, над водой {sh - d:.0f} м)")
        if E["grounding_line"][row, col]:
            s.add("Линия налегания", "лёд здесь сходит в море")
    if realm == 1 and np.isfinite(E["et0_annual_mm"][row, col]):
        s.add("Испаряемость ET0", _n(float(E["et0_annual_mm"][row, col]), 0, "мм/год"))
    if s.rows:
        sections.append(s)


def _volcanism_section(project, volc, col, row, sections) -> None:
    flat = row * project.layout.width + col
    V = volc.layers
    s = Section("volcanism", "Вулканизм (последние тысячелетия)")
    for sy in volc.system_at(flat)[:3]:
        m = sy.get("magma") or {}
        s.add(
            f"  {_VOLC_EDIFICE.get(sy['edifice_type'], sy['edifice_type'])}",
            f"{_VOLC_STATUS.get(sy['magmatic_status'], sy['magmatic_status'])}, "
            f"{_VOLC_PHASE.get(sy['activity_phase'], sy['activity_phase'])}",
        )
        if m:
            cls = _MAGMA_CLASS.get(m.get("magma_class", ""), m.get("magma_class", ""))
            s.add("  Магма", f"{cls}, SiO₂ {m.get('silica_wt_pct', 0):.0f} %, "
                  f"вязкость 10^{m.get('log10_viscosity_pa_s', 0):.1f} Па·с")  # fmt: skip
        if sy.get("last_eruption_years_bp") is not None:
            s.add("  Последнее извержение", f"{sy['last_eruption_years_bp']:.0f} лет назад")
        if sy.get("lava_lake_state") not in (None, "none"):
            s.add("  Лавовое озеро", sy["lava_lake_state"].replace("_", " "))
    evs = [e for e in volc.events if e.get("vent_ids") and any(
        sy["volcano_id"] == e["volcano_id"] for sy in volc.system_at(flat))]  # fmt: skip
    for e in sorted(evs, key=lambda e: e["years_bp"])[:3]:
        style = ", ".join(_VOLC_STYLE.get(p, p) for p in e["style_phases"])
        vol = e["products"]["volume_dre_m3"] / 1e6
        s.add(f"  Извержение {e['years_bp']:.0f} лет назад",
              f"{style}; VEI {e.get('vei', 0)}; {vol:.1f} млн м³")  # fmt: skip
    s.add("Лава (доля гекса)", _pct(float(V["lava_fraction"][row, col]), 1)
          if V["lava_fraction"][row, col] > 0 else None)  # fmt: skip
    load = float(V["tephra_load_kg_m2"][row, col])
    s.add("Пепел за окно", f"{load:.0f} кг/м² (~{load / 10:.0f} см)" if load >= 1 else None)
    s.add("Зона пирокластических потоков", _pct(float(V["pdc_hazard_fraction"][row, col]))
          if V["pdc_hazard_fraction"][row, col] > 0 else None)  # fmt: skip
    s.add("Лахары и паводки", _pct(float(V["lahar_fraction"][row, col]), 1)
          if V["lahar_fraction"][row, col] > 0 else None)  # fmt: skip
    s.add("Вулканическая опасность", _n(float(V["volcanic_hazard_index"][row, col]), 2)
          if V["volcanic_hazard_index"][row, col] > 0 else None)  # fmt: skip
    s.add("Свежая пустошь", _pct(float(V["volcanic_barren_fraction"][row, col]), 1)
          if V["volcanic_barren_fraction"][row, col] > 0 else None)  # fmt: skip
    s.add("Геотермальное тепло", _n(float(V["geothermal_heat_mw"][row, col]), 0, "МВт")
          if V["geothermal_heat_mw"][row, col] > 0 else None)  # fmt: skip
    s.add("Скопления CO₂ в низинах", "возможны" if V["co2_pooling_flag"][row, col] else None)
    s.add("Грязевые вулканы", "есть" if V["mud_volcano_flag"][row, col] else None)
    if s.rows:
        sections.append(s)


def _fmt_mass_t(t: float) -> str:
    if t >= 1e6:
        return f"{t / 1e6:,.1f} млн т"
    if t >= 1e3:
        return f"{t / 1e3:,.1f} тыс. т"
    if t >= 1.0:
        return f"{t:,.1f} т"
    return f"{t * 1000:,.1f} кг"


_ELEMENT_TITLES = {"diamond": "алмазы", "coal": "уголь", "oil": "нефть", "gas": "газ",
                   "NaCl": "каменная соль", "C_graphite": "графит", "gem": "самоцветы",
                   "REE": "РЗЭ", "Li_mg_per_l": "Li, мг/л"}  # fmt: skip


def _biosphere_section(project, col, row, sections) -> None:
    """TZ 5.5: named analogues, why present or absent, productivity, richness, resources —
    read from the stored state (no solver on hover)."""
    from .biosphere.explain import REASON_RU, hex_report

    st = project.biosphere.state
    i = int(row) * project.layout.width + int(col)
    rep = hex_report(st, i)
    s = Section("nature", "Природа (5.5)")

    def names(items, key=None):
        out = []
        for x in items:
            t = x["analogue"]
            if key == "hex_share" and x.get("hex_share", 0) > 0:
                t += f" {x['hex_share']:.0%}"
            out.append(t)
        return ", ".join(out) or "—"

    s.add("Растения", names(rep["flora"], "hex_share"))
    s.add("Животные", names(rep["fauna"]))
    s.add("Беспозвоночные", names(rep.get("invertebrates", [])))
    from .biosphere.explain import _months

    seas = [f"{x['analogue']} — активность в мес. {_months(x['season']['active_months'])}, "
            f"в остальное время {x['season']['otherwise']}" for x in rep["fauna"]
            if x.get("season")]  # fmt: skip
    if seas:
        s.add("Сезоны", "; ".join(seas[:3]))
    s.add("Богатство", f"флора {rep['flora_richness']}, фауна {rep['fauna_richness']}, "
          f"пресные воды {rep['freshwater_richness']}, море {rep['marine_richness']} "
          f"аналогов; групп {rep['functional_richness']}; индекс "
          f"{rep['total_richness_index']:.0f}/100, перцентиль мира "
          f"{rep['total_richness_percentile']:.0f}")  # fmt: skip
    s.add("Продуктивность", f"{rep['npp_gc_m2_yr']:,.0f} гC/м²/год; фитомасса "
          f"{rep['plant_biomass_kgc_m2']:.1f} кгC/м²; животные "
          f"{rep['animal_biomass_kgc_km2']:,.0f} кгC/км²")  # fmt: skip
    if rep.get("npp_limit"):
        s.add("Ограничитель", rep["npp_limit"])
    if rep["endemic_taxa"]:
        s.add("Эндемики", f"{rep['endemic_taxa']} аналогов с ареалом ≤ 100 000 км²")
    for x in rep["absent_examples"][:4]:
        why = REASON_RU[x["status"]] + (f" — {x['limiting']}" if x.get("limiting") else "")
        s.add(f"Нет: {x['analogue']}", why)
    res = project.biosphere.resources()["layers"]
    area = st.area_km2
    parts = []
    if res["timber_increment_m3_yr"][i] > 0:
        parts.append(f"древесина {res['timber_increment_m3_yr'][i] / area:,.0f} м³/км²/год "
                     "(то же, что дрова)")  # fmt: skip
    if res["game_production_t_wet_yr"][i] > 0:
        parts.append(f"дичь {res['game_production_t_wet_yr'][i] / area:.2f} т/км²/год")
    fish = res["fish_fresh_production_t_wet_yr"][i] + res["fish_marine_production_t_wet_yr"][i]
    if fish > 0:
        parts.append(f"рыба {fish / area:.2f} т/км²/год")
    if parts:
        s.add("Возобновляемые ресурсы", "; ".join(parts) + " (потенциал, не добыча)")
    s.add("Качество", f"история {rep['quality']}; параметры предварительные; аналоги — не виды")
    sections.append(s)


def _minerals_section(project, col, row, sections) -> None:
    """TZ 5.5.1: every body that crosses the hex (not only its centre), with typed units."""
    from .minerals.index import hex_index
    from .minerals.units import GRADE_UNITS

    m = project.minerals
    bodies = hex_index(m, project.layout).get((int(col), int(row)), [])
    s = Section("minerals", "Недра и месторождения (5.5.1)")
    if not bodies:
        s.add("Тела под гексом", "нет (в плане 5.5.1)")
        sections.append(s)
        return
    sys_by = {x["system_id"]: x for x in m.plan.systems}
    lot_by = {x["body_id"]: x for x in m.plan.lots if "body_id" in x}
    unit_ru = {"percent": "%", "ppm": "ppm", "g_per_t": "г/т", "ct_per_t": "кар/т",
               "g_per_m3": "г/м³", "fraction": "доля", "kg_per_t": "кг/т"}  # fmt: skip
    for bid in sorted(bodies)[:12]:
        b = m.plan.bodies[bid]
        sy = sys_by.get(b.system_id, {})
        lot = lot_by.get(bid, {})
        head = f"{sy.get('title', sy.get('model_id', ''))} — {b.morphology}"
        parts = [f"кровля {sy.get('depth_top_m', 0):,.0f} м"]
        if hasattr(b, "t0") and b.morphology != "vein_network":
            parts.append(f"истинная мощность {b.t0:,.2f} м")
        if b.morphology not in ("mantle_pipe",):
            parts.append(f"простирание {sy.get('strike_deg', 0):.0f}°, падение "
                         f"{sy.get('dip_deg', 0):.0f}°")  # fmt: skip
        if lot.get("rock_t"):
            parts.append(f"{lot['rock_t']:,.3g} т горной массы")
        grades = []
        for com, tg in list(lot.get("typed_grades", {}).items())[:3]:
            if tg.get("unit") in GRADE_UNITS:
                grades.append(f"{com} {tg['value']:.3g} {unit_ru.get(tg['unit'], tg['unit'])}")
        if grades:
            parts.append(", ".join(grades))
        if lot.get("gem"):
            g = lot["gem"]
            parts.append(f"{g.get('variety') or g['carrier']}: {g['gem_rough_ct']:,.0f} кар "
                         "ювелирного сырца")  # fmt: skip
        parts.append(f"{sy.get('provenance', '')}, {lot.get('calibration_status', 'provisional')}")
        s.add(head, "; ".join(parts))
    if len(bodies) > 12:
        s.add("…", f"ещё {len(bodies) - 12} тел")
    sections.append(s)


def _resources_sections(project, res, col, row, sections) -> None:
    from .resources.deposits import LITHOLOGY, LITHOLOGY_TITLES
    from .resources.secondary import CONSTRUCTION_TITLES
    from .resources.soil import FACTORS

    L = res.layers
    flat = row * project.layout.width + col
    s = Section("fertility", "Плодородие почвы (v5.2)")
    if np.isfinite(L["soil_ph"][row, col]):
        lim_titles = {"nutrient": "питание N/P/K", "water": "вода", "ph": "кислотность",
                      "oxygen": "переувлажнение", "rooting": "корнеобитаемый слой",
                      "salinity": "засоление"}  # fmt: skip
        s.add("Природный потенциал", f"{float(L['soil_fertility_potential'][row, col]):.2f} "
              f"(0..1, не урожайность)")  # fmt: skip
        lim = int(L["soil_limiting_factor"][row, col])
        s.add(
            "Главное ограничение", lim_titles.get(FACTORS[lim], "") if lim < len(FACTORS) else None
        )
        s.add("Органический углерод", f"{float(L['soil_soc_kg_m2'][row, col]):.1f} кг C/м² "
              f"({float(L['soil_soc_kg_m2'][row, col]) * 10:.0f} т C/га)")  # fmt: skip
        n_av = float(L["soil_ammonium_N_kg_m2"][row, col] + L["soil_nitrate_N_kg_m2"][row, col])
        s.add("Азот доступный / органический", f"{n_av * 1e4:.0f} / "
              f"{float(L['soil_organic_N_kg_m2'][row, col]) * 1e4:,.0f} кг N/га")  # fmt: skip
        s.add("Фосфор доступный / окклюдированный",
              f"{float(L['soil_P_labile_kg_m2'][row, col]) * 1e4:.0f} / "
              f"{float(L['soil_P_occluded_kg_m2'][row, col]) * 1e4:.0f} кг P/га")  # fmt: skip
        s.add("Калий обменный", f"{float(L['soil_K_exchange_kg_m2'][row, col]) * 1e4:,.0f} кг K/га")
        s.add("pH / насыщенность основаниями", f"{float(L['soil_ph'][row, col]):.1f} / "
              f"{100 * float(L['soil_base_saturation_frac'][row, col]):.0f} %")  # fmt: skip
        s.add("ЕКО", _n(float(L["soil_cec_cmolc_kg"][row, col]), 1, "смоль(+)/кг"))
        ec = float(L["soil_salinity_ec_ds_m"][row, col])
        s.add("Засоление (EC, proxy)", f"{ec:.1f} dS/m" if ec >= 0.5 else None)
        s.add("Возраст почвы", f"{float(L['soil_age_yr'][row, col]):,.0f} лет")
        wheat = float(L["suitability_crop_wheat"][row, col])
        rice = float(L["suitability_crop_rice_paddy"][row, col])
        s.add("Пригодность: пшеница / рис", f"{wheat:.2f} / {rice:.2f}")
    else:
        s.add("Почва", "нет (скала, лёд или вода): NoData, не «бедная почва»")
    sections.append(s)

    s = Section("resources", "Полезные ископаемые (v5.2)")
    s.add(
        "Литология (выведена)",
        LITHOLOGY_TITLES.get(LITHOLOGY[int(L["geo_lithology_inferred"][row, col])]),
    )
    cls = int(L["construction_material_class"][row, col])
    if cls:
        s.add("Строительные материалы", f"{CONSTRUCTION_TITLES[cls]}, качество "
              f"{float(L['construction_material_quality'][row, col]):.2f}")  # fmt: skip
    bodies = res.bodies_at(flat)
    seen = []
    for b in bodies:
        sy = res.system(b["system_id"])
        if sy is None or sy["deposit_id"] in seen:
            continue
        seen.append(sy["deposit_id"])
        visible = sy["surface_occurrence"]
        tag = "" if visible else " [истина генератора: скрыто]"
        comps = ", ".join(
            f"{_ELEMENT_TITLES.get(el, el)} {_fmt_mass_t(m)}"
            for el, m in sorted(sy["contained_t"].items(), key=lambda kv: -kv[1])[:3]
        )
        s.add(f"  {sy['title']}{tag}", f"{_fmt_mass_t(sy['ore_t'])} руды; {comps}")
        s.add("    глубина кровли", f"{sy['depth_top_m']:,.0f} м; образовано "
              f"{sy['formation_epoch_ma']:,.1f} млн лет назад ({sy['provenance']})")  # fmt: skip
        if len(seen) >= 4:
            break
    if not seen and int(L["deposit_count"][row, col]) == 0:
        s.add("Месторождения", "не сгенерированы под этим гексом")
    sections.append(s)


def _names():
    from . import names

    return names


def _landmass(p: Project, info: dict) -> str | None:
    lid = info.get("landmass_id", -1)
    if lid is None or lid < 0:
        return None
    lands = p.hist.landmasses if p.hist is not None else (p.geo.landmasses if p.geo else [])
    m = next((m for m in lands if m.get("id") == lid), None)
    if m is None and 0 <= lid < len(lands):
        m = lands[lid]
    if m is None:
        return f"№ {lid}"
    area = m.get("area_km2", 0) / 1e6
    size = f"{area:,.2f} млн км²" if area >= 0.01 else f"{area * 1e6:,.0f} км²"
    return f"{_names().ru(m.get('name', str(lid)))} ({size.replace(',', ' ')})"


def _mountain_system(p: Project, sid: int) -> str:
    systems = (p.hist.plan or {}).get("systems", []) if p.hist is not None else []
    s = next((x for x in systems if x.get("id") == sid), None)
    if s is None:
        return f"№ {sid}"
    style = s.get("style") or s.get("kind") or ""
    top = s.get("max_elevation_m")
    tail = f", высшая точка {_n(top, 0, 'м')}" if top is not None else ""
    name = s.get("name")
    head = f"«{name}» " if name else ""
    return f"{head}№ {sid} ({style}{tail})"


def _volcano(p: Project, vid: int, info: dict) -> str:
    vols = p.hist.volcanoes if p.hist is not None else []
    v = next((x for x in vols if x.get("id") == vid), None)
    age = info.get("volcanic_age_ma")
    parts = [f"№ {vid}"]
    if v is not None:
        parts.append(str(v.get("kind", "")))
        parts.append("действующий" if v.get("active") else "потухший")
    if age is not None:
        parts.append(f"возраст {age:.1f} млн лет")
    return ", ".join(x for x in parts if x)


def _direction(vx: float, vy: float) -> str:
    """Compass direction of a vector with +x east, +y south."""
    import math

    if abs(vx) + abs(vy) < 1e-9:
        return "стоит"
    ang = math.degrees(math.atan2(-vy, vx)) % 360  # 0 = east, counter-clockwise
    names = ("на восток", "на северо-восток", "на север", "на северо-запад", "на запад",
             "на юго-запад", "на юг", "на юго-восток")  # fmt: skip
    return names[int((ang + 22.5) // 45) % 8]


def _aspect(deg: float) -> str:
    names = ("северная", "северо-восточная", "восточная", "юго-восточная", "южная",
             "юго-западная", "западная", "северо-западная")  # fmt: skip
    return names[int(((deg % 360) + 22.5) // 45) % 8]


def _aridity_class(ai: float) -> str:
    if ai < 0.05:
        return "сверхсухо"
    if ai < 0.2:
        return "сухо (пустыня)"
    if ai < 0.5:
        return "полусухо (степь, саванна)"
    if ai < 0.65:
        return "умеренно сухо"
    if ai < 1.0:
        return "умеренно влажно"
    return "влажно"
