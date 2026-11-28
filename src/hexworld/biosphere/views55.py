"""Map tab «Природа (5.5)» of TZ 5.5 §15: one tab with sublayers instead of 17 windows.

Display only: switching a sublayer, the index / percentile scale or the legend never
re-runs the model (B07). Missing data is grey, never a value of the scale; the indices
have fixed, versioned bounds (B62) and the percentile is area-weighted over the whole
world, not the viewport (B63).
"""

from __future__ import annotations

import numpy as np

from ..views import LegendItem, Option, ViewSpec, _ramp_rgb, _shade, id_colors, mask_rim, register_view
from . import metrics as mt
from .registry import TERRESTRIAL_LAYERS, load

SEA_RGB = np.array((38, 58, 92), dtype=np.float64)
MISSING_RGB = np.array((150, 150, 150), dtype=np.float64)

MODES = (
    ("flora", "Флора: преобладающая группа растений"),
    ("fauna", "Фауна: преобладающая группа позвоночных"),
    ("flora_richness", "Богатство флоры"),
    ("fauna_richness", "Богатство фауны"),
    ("total_richness", "Общее биоразнообразие"),
    ("biomass", "Биомасса растений"),
    ("npp", "Биопродуктивность (NPP)"),
    ("bioregion", "Биогеография: области по линиям"),
    ("endemism", "Эндемизм (взвешенный)"),
    ("connectivity", "Экологическая связность (изоляция)"),
    ("migration", "Миграция: перелётные и кочующие группы"),
    ("age", "Возраст экосистем (субстрата)"),
    ("disturbance", "Нарушения (пожары, лава)"),
    ("res_timber", "Биоресурсы: древесина (прирост)"),
    ("res_game", "Биоресурсы: дичь (продукция)"),
    ("res_fish", "Биоресурсы: рыба (продукция)"),
    ("freshwater", "Пресноводная жизнь"),
    ("marine", "Морская жизнь"),
    ("domestication", "Потенциал одомашнивания (дикие предки)"),
    ("limit", "Главный ограничитель продуктивности"),
)  # fmt: skip
SCALE_NOTE = {"index": "индекс 0–100 (фиксированные границы)", "percentile": "перцентиль мира"}
_LIMIT_RGB = ((240, 220, 90), (120, 160, 230), (220, 140, 60), (150, 90, 170), (60, 60, 60))


def _bio(project):
    return project.biosphere.state


def _scalar(project, mode):
    st = _bio(project)
    L = st.layers
    area = st.area_km2
    if mode in ("flora_richness", "fauna_richness", "total_richness"):
        return L[mode].astype(float)
    if mode == "biomass":
        return L.get("dyn_plant_biomass_kgc_m2", L["plant_biomass_kgc_m2"]).astype(float)
    if mode == "npp":
        return L["npp_gc_m2_yr"].astype(float)
    if mode == "endemism":
        return L["weighted_endemism"].astype(float)
    if mode == "connectivity":
        return np.log10(1.0 + np.nan_to_num(L["isolation_km"].astype(float), nan=np.inf))
    if mode == "migration":
        return L["migratory_groups"].astype(float)
    if mode == "age":
        a = L.get("dyn_stand_age_yr")
        return np.log10(1.0 + (a if a is not None else L["substrate_age_yr"]).astype(float))
    if mode == "disturbance":
        if "dyn_burned_fraction" in L:
            return L["dyn_burned_fraction"].astype(float)
        # without the P1 run: the burnt share of the last decade (5.3) and fresh volcanic
        # ground (5.1) of the world itself
        b = project.biomes.layers.get("burn_fraction_last_decade") if project.biomes else None
        v = project.volcanism.layers.get("volcanic_barren_fraction") if project.volcanism else None
        out = np.zeros(st.n)
        if b is not None:
            out += np.nan_to_num(np.asarray(b, float).ravel())
        if v is not None:
            out += np.nan_to_num(np.asarray(v, float).ravel())
        return np.clip(out, 0, 1)
    if mode == "freshwater":
        return L["freshwater_richness"].astype(float)
    if mode == "marine":
        return L["marine_richness"].astype(float)
    res = project.biosphere.resources()["layers"]
    if mode == "res_timber":
        return res["timber_increment_m3_yr"] / area
    if mode == "res_game":
        return res["game_production_t_wet_yr"] / area
    if mode == "res_fish":
        return (res["fish_fresh_production_t_wet_yr"] + res["fish_marine_production_t_wet_yr"]) / area
    if mode == "domestication":
        return res["wild_precursor_score"]
    raise KeyError(mode)


_BOUNDS = {"flora_richness": (0, 50), "fauna_richness": (0, 90), "total_richness": (0, 150),
           "biomass": (0, 25), "npp": (0, 1500), "endemism": (0, 2), "connectivity": (0, 4),
           "migration": (0, 10), "age": (0, 6), "disturbance": (0, 1), "freshwater": (0, 40),
           "marine": (0, 40), "res_timber": (0, 2000), "res_game": (0, 2), "res_fish": (0, 20),
           "domestication": (0, 1)}  # fmt: skip


def _dominant(project, kind):
    st = _bio(project)
    reg = load()
    rows, ids = [], []
    for gid, d in st.groups.items():
        g = reg.groups[gid]
        if kind == "flora" and "cover" in d and g.traits.get("layer") in TERRESTRIAL_LAYERS:
            rows.append(d["cover"].astype(float) * st.frac["land_ground"])
            ids.append(gid)
        elif kind == "fauna" and "biomass_kgc" in d and g.is_fauna and \
                g.traits["mass_kg"] >= 0.005:  # fmt: skip
            rows.append(d["biomass_kgc"].astype(float))
            ids.append(gid)
    m = np.stack(rows)
    best = m.argmax(axis=0)
    has = m.max(axis=0) > 0
    return ids, best, has


def _render(project, mode, overlays):
    st = _bio(project)
    h, w = st.shape
    life = (st.frac["land_ground"] + st.frac["inland_water"]) > 0
    land_view = mode not in ("marine", "res_fish", "bioregion", "npp", "total_richness",
                             "fauna_richness", "flora_richness", "connectivity", "limit")
    rgb = np.empty((st.n, 3))
    if mode in ("flora", "fauna"):
        ids, best, has = _dominant(project, mode)
        cols = id_colors(len(ids), 0.07)
        rgb[:] = cols[best]
        rgb[~has] = MISSING_RGB
    elif mode == "bioregion":
        reg_id = st.layers["bioregion"].astype(int)
        rgb[:] = id_colors(int(reg_id.max()) + 1, 0.31)[reg_id]
    elif mode == "limit":
        rgb[:] = np.array(_LIMIT_RGB, dtype=float)[st.layers["npp_limit"].astype(int)]
    else:
        v = _scalar(project, mode)
        if "percentile" in overlays:
            valid = np.isfinite(v) & ((st.frac["land_ground"] > 0) if land_view else
                                      np.ones(st.n, bool))  # fmt: skip
            v = mt.percentile_area_weighted(v, st.area_km2, valid)
            lo, hi = 0.0, 100.0
        else:
            lo, hi = _BOUNDS[mode]
        rgb[:] = _ramp_rgb(v, lo, hi, 0.8 if mode in ("npp", "biomass") else 1.0)
        rgb[~np.isfinite(v)] = MISSING_RGB
    if land_view:
        rgb[~life] = SEA_RGB
    rgb = rgb.reshape(h, w, 3).astype(np.uint8)
    if "coast" in overlays:
        _shade(rgb, mask_rim(project.hydro.layers["land_mask"].astype(bool), project.layout), 0.55)
    return rgb


def _legend(project, mode):
    st = _bio(project)
    s = st.summary
    items = [LegendItem(f"5.5: аналогов в мире {s['taxa_world']}, эндемиков "
                        f"{s['taxa_endemic']}, областей {s['bioregions']}; журнал углерода "
                        f"{'сходится' if s.get('ledger_ok') else 'НЕ сходится'}; режим "
                        f"{st.mode}; параметры предварительные")]  # fmt: skip
    if mode in ("flora", "fauna"):
        reg = load()
        ids, best, has = _dominant(project, mode)
        cols = id_colors(len(ids), 0.07)
        counts = np.bincount(best[has], minlength=len(ids))
        for j in np.argsort(-counts)[:12]:
            if counts[j] == 0:
                break
            g = reg.groups[ids[j]]
            items.append(LegendItem(f"{g.analogue_ru} — {g.name_ru}", tuple(int(c) for c in cols[j])))
        items.append(LegendItem("серый — нет такой жизни (лёд, голый камень, не дошла)", (150, 150, 150)))
    elif mode == "bioregion":
        items.append(LegendItem("цвет — сочетание линий наземных, летающих, семенных, речных и "
                                "морских групп; одинаковый климат на разных материках даёт "
                                "разные области"))  # fmt: skip
        items.append(LegendItem("история сухопутных мостов в мире не хранится: качество "
                                "present_only"))  # fmt: skip
    elif mode == "limit":
        for name, c in zip(st.layers.get("_npp_limit_names") or [], _LIMIT_RGB, strict=False):
            items.append(LegendItem(str(name), c))
    else:
        lo, hi = _BOUNDS[mode]
        units = {"biomass": "кгC/м²", "npp": "гC/м²/год", "res_timber": "м³/км²/год",
                 "res_game": "т/км²/год", "res_fish": "т/км²/год", "connectivity": "lg(1+км)",
                 "age": "lg(лет)"}.get(mode, "")  # fmt: skip
        items.append(LegendItem(f"шкала {lo}–{hi} {units} (светлое — больше); серый — нет "
                                "данных"))  # fmt: skip
        items.append(LegendItem("«Перцентиль мира» — ранг по площади всего мира"))
        if mode.endswith("richness"):
            items.append(LegendItem("богатство = число уникальных аналогов (вариантов групп), не "
                                    "реальных видов"))  # fmt: skip
    return items


register_view(
    ViewSpec(
        id="nature_55",
        title="Природа (5.5)",
        stage="v55",
        layers=("npp_gc_m2_yr", "total_richness"),
        modes=tuple(Option(k, t) for k, t in MODES),
        overlays=(Option("percentile", "Перцентиль мира вместо шкалы"),
                  Option("coast", "Берег", True)),
        note="Растения, животные, биогеография и возобновляемые ресурсы по ТЗ 5.5; аналоги — "
        "функциональные группы, не виды",
        render=_render,
        legend=_legend,
    )
)
