"""Map tabs of TZ 5.5.1 (§15): subsurface geology, mineral resources by group with a depth
filter, hydrogeology and fuels.

Display only: a filter hides bodies, it never deletes them or reseeds (T74, T75). Glyphs
mark presence at every zoom; vein traces are drawn along strike at their real length, not
as the physical width; the hexes a body crosses are tinted (a footprint, not solid ore).
"""

from __future__ import annotations

import numpy as np

from ..views import (
    LegendItem,
    Option,
    VectorLayer,
    ViewSpec,
    _relief_gray,
    _shade,
    id_colors,
    mask_rim,
    register_view,
)
from .catalog import load

GROUPS = (
    ("all", "Все группы сырья", None),
    ("metals", "Чёрные, цветные, редкие металлы", ("ferrous", "nonferrous", "rare", "radioactive")),
    ("precious", "Благородные металлы", ("precious",)),
    ("gems", "Самоцветы и поделочные камни", ("gem", "ornamental")),
    ("industrial", "Промышленное и агрохимическое сырьё", ("industrial", "chemical_agro")),
    ("construction", "Строительное сырьё", ("construction",)),
    ("fuel", "Топливо (уголь, торф, нефть, газ)", ("fuel",)),
    ("water_heat", "Воды и тепло", ("water_heat",)),
)  # fmt: skip
_GROUP = {k: cats for k, _t, cats in GROUPS}
_COLORS = {"metals": (200, 60, 40), "precious": (235, 185, 20), "gems": (40, 170, 200),
           "industrial": (150, 90, 170), "construction": (150, 150, 150),
           "fuel": (30, 30, 30), "water_heat": (40, 110, 220)}  # fmt: skip
DEPTHS = (("d25", 25.0), ("d100", 100.0), ("d500", 500.0), ("d1000", 1000.0), ("d3000", 3000.0))
_MORPH_SIZE = {"mantle_pipe": 5.5, "placer": 3.5, "vein": 3.0, "vein_network": 3.5,
               "stockwork": 5.0, "reservoir": 5.0}  # fmt: skip


def _category_of(system) -> str | None:
    cat = load()
    for com in system.get("principal", []):
        c = cat.commodities.get(com)
        if c is None:
            continue
        for g, cats in _GROUP.items():
            if cats and c.category in cats:
                return g
    return None


def _max_depth(overlays) -> float:
    ds = [d for k, d in DEPTHS if k in overlays]
    return min(ds) if ds else float("inf")


def _visible(project, mode, overlays) -> list:
    m = project.minerals
    out = []
    lim = _max_depth(overlays)
    for s in m.plan.systems:
        g = _category_of(s)
        if mode != "all" and g != mode:
            continue
        if s.get("depth_top_m", 0.0) > lim:
            continue
        if "surface_only" in overlays and not s.get("surface_occurrence"):
            continue
        out.append((s, g))
    return out


def _render_minerals(project, mode, overlays):
    from .index import hex_index

    rgb = _relief_gray(project).astype(np.float64)
    lay = project.layout
    vis = _visible(project, mode, overlays)
    ids = {bid: g for s, g in vis for bid in s.get("body_ids", [])}
    idx = hex_index(project.minerals, lay)
    for (c, r), bodies in idx.items():
        if not (0 <= c < lay.width and 0 <= r < lay.height):
            continue
        gs = [ids[b] for b in bodies if b in ids and ids[b]]
        if gs:
            rgb[r, c] = 0.45 * rgb[r, c] + 0.55 * np.array(_COLORS.get(gs[0], (200, 0, 0)))
    rgb = rgb.astype(np.uint8)
    if "coast" in overlays:
        _shade(rgb, mask_rim(project.hydro.layers["land_mask"].astype(bool), lay), 0.55)
    return rgb


def _vectors_minerals(project, mode, overlays):
    m = project.minerals
    out = []
    vis = _visible(project, mode, overlays)
    by_g: dict = {}
    for s, g in vis:
        by_g.setdefault((g, s["morphology"]), []).append(s)
    for (g, morph), ss in sorted(by_g.items(), key=lambda kv: (str(kv[0][0]), kv[0][1])):
        color = _COLORS.get(g, (200, 0, 0))
        x = np.array([s["x_m"] / 1000.0 for s in ss])
        y = np.array([s["y_m"] / 1000.0 for s in ss])
        out.append(VectorLayer("points", x, y, color, _MORPH_SIZE.get(morph, 3.0),
                               morph not in ("vein", "vein_network"), morph))  # fmt: skip
    # vein traces: the real strike length, a thin line (not the width of the vein)
    xs, ys = [], []
    for s, _g in vis:
        if s["morphology"] not in ("vein", "vein_network", "seam", "tabular"):
            continue
        for bid in s.get("body_ids", []):
            b = m.plan.bodies.get(bid)
            veins = getattr(b, "veins", None) or [b]
            for v in veins:
                if not hasattr(v, "footprint"):
                    continue
                fp = v.footprint() / 1000.0
                xs.append(np.r_[fp[:, 0], fp[0, 0]])
                ys.append(np.r_[fp[:, 1], fp[0, 1]])
    if xs:
        out.append(VectorLayer("lines", xs, ys, (90, 20, 20), 1.2, True, "проекция тел"))
    return out


def _legend_minerals(project, mode):
    s = project.minerals.summary
    items = [LegendItem(f"5.5.1: систем {s['systems']}, лотов {s['lots']}, журнал масс "
                        f"{'сходится' if s['ledger_ok'] else 'НЕ сходится'}; параметры "
                        "предварительные (provisional)")]  # fmt: skip
    for k, t, _c in GROUPS[1:]:
        if mode in ("all", k):
            items.append(LegendItem(t, _COLORS[k]))
    items += [LegendItem("гекс подкрашен — его пересекает тело (проекция, не сплошная руда)"),
              LegendItem("метка: крупнее — трубка / штокверк / залежь флюида, мельче — жила"),
              LegendItem("линии — контур проекции жил и пластов в истинную длину"),
              LegendItem("фильтр глубины по кровле тела ниже твёрдой поверхности; толща воды "
                         "отдельно")]  # fmt: skip
    return items


register_view(
    ViewSpec(
        id="minerals_551",
        title="Недра и месторождения (5.5.1)",
        stage="v551",
        layers=("system_count",),
        modes=tuple(Option(k, t) for k, t, _c in GROUPS),
        overlays=(
            *(Option(k, f"кровля ≤ {int(d)} м") for k, d in DEPTHS),
            Option("surface_only", "Только с выходом на поверхность"),
            Option("coast", "Берег", True),
        ),
        note="Месторождения 5.5.1 на чёрно-белом рельефе; глубинный фильтр скрывает тела, "
        "не удаляя их",
        render=_render_minerals,
        legend=_legend_minerals,
        vectors=_vectors_minerals,
    )
)


# -- subsurface geology -------------------------------------------------------------------------
def _ctx(project):
    cache = getattr(project.minerals, "_ctx", None)
    if cache is None:
        from ..resources.deposits import context_from_project
        from .features import features_from_project

        ctx = context_from_project(project)
        cache = (ctx, features_from_project(project, ctx))
        project.minerals._ctx = cache
    return cache


def _render_geology(project, mode, overlays):
    from ..resources.deposits import LITHOLOGY
    from ..views import _ramp_rgb

    ctx, F = _ctx(project)
    sh = ctx.shape
    if mode == "lithology":
        rgb = id_colors(len(LITHOLOGY), 0.13)[ctx.lith.reshape(sh).astype(int)]
    elif mode == "basin_fill":
        rgb = _ramp_rgb(ctx.basin_fill_m.reshape(sh), 0.0, 6000.0, 0.7)
    elif mode == "crust_age":
        rgb = _ramp_rgb(np.log10(np.maximum(np.nan_to_num(ctx.crust_age_ma), 1.0)).reshape(sh),
                        0.0, 3.5)  # fmt: skip
    else:  # unresolved: hexes where a gate had no input
        from .gates import RULES, evaluate

        cnt = np.zeros(ctx.n)
        for r in RULES.values():
            cnt += evaluate(r, F)[1]
        rgb = _ramp_rgb(cnt.reshape(sh), 0.0, max(1.0, float(cnt.max())))
    rgb = np.asarray(rgb, dtype=np.uint8).copy()
    if "coast" in overlays:
        _shade(rgb, mask_rim(project.hydro.layers["land_mask"].astype(bool), project.layout),
               0.5)  # fmt: skip
    return rgb


def _legend_geology(project, mode):
    from ..resources.deposits import LITHOLOGY, LITHOLOGY_TITLES

    if mode == "lithology":
        cols = id_colors(len(LITHOLOGY), 0.13)
        return [LegendItem("литология выведена из геоистории (provenance: inferred)")] + [
            LegendItem(LITHOLOGY_TITLES[n], tuple(int(c) for c in cols[i]))
            for i, n in enumerate(LITHOLOGY) if i]  # fmt: skip
    if mode == "basin_fill":
        return [LegendItem("мощность осадочного выполнения бассейнов, 0 — 6 км")]
    if mode == "crust_age":
        return [LegendItem("возраст коры, лог-шкала 1 — 3000 млн лет; древнее 40 млн лет — "
                           "унаследованные комплексы")]  # fmt: skip
    return [LegendItem("число типов месторождений с неразрешённым условием (нет данных — не "
                       "«нет месторождения»)")]  # fmt: skip


register_view(
    ViewSpec(
        id="geology_551",
        title="Геология недр (5.5.1)",
        stage="v551",
        layers=(),
        modes=(
            Option("lithology", "Породы (выведенные)"),
            Option("basin_fill", "Мощность осадков бассейнов"),
            Option("crust_age", "Возраст коры"),
            Option("unresolved", "Нет данных для условий"),
        ),  # fmt: skip
        overlays=(Option("coast", "Берег", True),),
        note="Геологическая основа планирования месторождений; правый щелчок — разрез с телами",
        render=_render_geology,
        legend=_legend_geology,
    )
)


# -- hydrogeology and fuels ----------------------------------------------------------------------
_FLUID_MODES = (("petroleum", "Нефть и газ (ловушки)", ("PETROLEUM",), (150, 40, 160)),
                ("coal", "Уголь и торф", ("COAL", "PEAT", "OIL_SHALE"), (25, 25, 25)),
                ("water", "Водоносные горизонты", ("AQUIFER",), (40, 110, 220)),
                ("brine_heat", "Рассолы, геотермия, гидраты", ("BRINE", "GEOTHERMAL", "HYDRATE"),
                 (220, 90, 40)))  # fmt: skip


def _fluid_systems(project, mode):
    models = next(m for k, _t, m, _c in _FLUID_MODES if k == mode)
    return [s for s in project.minerals.plan.systems if s["model_id"] in models]


def _render_fluids(project, mode, overlays):
    rgb = _relief_gray(project)
    if "coast" in overlays:
        _shade(rgb, mask_rim(project.hydro.layers["land_mask"].astype(bool), project.layout),
               0.55)  # fmt: skip
    return rgb


def _vectors_fluids(project, mode, overlays):
    color = next(c for k, _t, _m, c in _FLUID_MODES if k == mode)
    ss = _fluid_systems(project, mode)
    if not ss:
        return []
    x = np.array([s["x_m"] / 1000.0 for s in ss])
    y = np.array([s["y_m"] / 1000.0 for s in ss])
    return [VectorLayer("points", x, y, color, 5.0, True, mode)]


def _legend_fluids(project, mode):
    title, _m, color = next((t, m, c) for k, t, m, c in _FLUID_MODES if k == mode)
    ss = _fluid_systems(project, mode)
    items = [LegendItem(f"{title}: {len(ss)}", color)]
    lots = {lot["system_id"]: lot for lot in project.minerals.plan.lots}
    if mode == "petroleum":
        oil = sum(lots[s["system_id"]].get("oil_standard_m3", 0.0) for s in ss
                  if s["system_id"] in lots)  # fmt: skip
        items.append(LegendItem(f"нефть в стандартных условиях: {oil / 1e6:,.1f} млн м³ "
                                "(15 °C, 101,325 кПа)"))  # fmt: skip
        items.append(LegendItem("только ловушки с покрышкой, сформированные до заряда"))
    if mode == "water":
        items.append(LegendItem("запас воды, отдаваемая вода и питание — разные величины"))
    if mode == "brine_heat":
        items.append(LegendItem("тепло в недрах (Дж) и тепловой поток (Вт) не складываются"))
    if not ss:
        items.append(LegendItem("в этом мире таких систем не образовалось"))
    return items


register_view(
    ViewSpec(
        id="fluids_551",
        title="Гидрогеология и топливо (5.5.1)",
        stage="v551",
        layers=(),
        modes=tuple(Option(k, t) for k, t, _m, _c in _FLUID_MODES),
        overlays=(Option("coast", "Берег", True),),
        note="Нефтегазовые ловушки, уголь, водоносные горизонты, рассолы, геотермия",
        render=_render_fluids,
        legend=_legend_fluids,
        vectors=_vectors_fluids,
    )
)
