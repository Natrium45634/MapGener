"""Geological features of the hard gates (§3): per hex 1 (present), 0 (absent) or NaN
(unresolved: the input is missing). Missing is never false (T05).

Built from the 5.2 geological context (lithology inferred, basins, volcanism, orogens, crust
ages) plus climate and hydrology where present. Neighbourhoods use a physical radius in km
(«near» = within ``NEAR_KM``), never a number of hexes.
"""

from __future__ import annotations

import math

import numpy as np
from scipy import ndimage

from ..resources.deposits import LITH

NEAR_KM = 60.0
#: crust older than this is an inherited ancient complex (craton / shield), Ma
ANCIENT_MA = 900.0

FEATURES_RU = {
    "land": "суша",
    "ocean": "океан (связный с мировым)",
    "ancient_crust": "древняя (унаследованная) кора, > 900 млн лет",
    "ancient_shield": "щит / кратон (граниты-гнейсы, зеленокаменные пояса)",
    "greenstone": "зеленокаменный пояс",
    "preserved_strata": "сохранившаяся стратиграфия (эрозия < 3 км)",
    "sedimentary_basin": "осадочный бассейн (мощность > 300 м)",
    "deep_burial": "погребение > 1,5 км (зрелость органики)",
    "marine_facies": "морские фации (шельф, карбонаты, чёрные сланцы)",
    "carbonate": "карбонатные толщи",
    "clastic_permeable": "проницаемые обломочные толщи",
    "reduced_facies": "восстановительные фации (чёрные сланцы, угленосные)",
    "evaporitic": "эвапоритовые / красноцветные толщи",
    "organic_burial": "захоронённое органическое вещество",
    "ultramafic": "ультрамафиты",
    "ophiolite": "офиолиты",
    "layered_intrusion": "расслоенная мафит-ультрамафитовая интрузия",
    "mafic": "мафитовые породы",
    "mafic_ultramafic": "мафиты или ультрамафиты",
    "felsic_intrusion": "гранитоидная интрузия",
    "al_silicate_rock": "алюмосиликатные породы (граниты, гнейсы, базальты)",
    "igneous_rock": "магматические породы",
    "sedimentary_rock": "осадочные породы",
    "metamorphic": "метаморфические породы / орогены",
    "volcanic_rock": "вулканиты",
    "arc": "магматическая дуга над субдукцией",
    "magmatic_activity": "магматическая система рядом (вулкан ≤ 60 км)",
    "alkaline_magmatism": "внутриплитный / рифтовый (щелочной) магматизм рядом",
    "hydrothermal": "гидротермальная система (магматизм или орогенные флюиды)",
    "shear_zone": "зона сдвига в коллизионном орогене",
    "high_p_low_t": "высокие давления при низкой температуре (зона субдукции)",
    "pegmatite": "пегматитовое поле (гранитоиды в орогене / щите)",
    "unconformity": "древнее несогласие под бассейном",
    "be_source_near": "источник бериллия рядом (гранит, пегматит)",
    "cr_host_near": "Cr/V-содержащий хозяин рядом (ультрамафиты, чёрные сланцы)",
    "carbonate_near_intrusion": "карбонаты у контакта интрузии",
    "weathering_long": "длительное влажное тропическое выветривание",
    "weathering_any": "выветривание (тепло и влага)",
    "karst_surface": "закарстованная карбонатная поверхность",
    "wetland": "заболоченность",
    "river": "река",
    "coast": "берег моря",
    "closed_basin": "бессточный бассейн",
    "arid": "аридный климат",
    "aquatic_basin": "водоём (озеро, шельф, бессточная впадина)",
    "volcanic_ash": "вулканический пепел",
    "porous_host": "пористый коллектор (песчаники, карбонаты)",
    "seal_rock": "покрышка (эвапориты, глинистые сланцы)",
    "heat_source": "источник тепла (магматизм, рифт)",
    "deep_ocean": "абиссаль (> 4 км)",
    "low_sedimentation": "медленное осадконакопление (< 600 м осадков)",
    "old_ocean_floor": "старое дно (> 10 млн лет)",
    "hard_substrate": "твёрдый субстрат подводной горы (0,8–3 км)",
    "submarine_hydrothermal": "подводная гидротермальная система (хребет, задуговый бассейн)",
    "shelf": "шельф",
    "upwelling": "апвеллинг",
    "cold_seafloor": "холодное дно с осадками (стабильность гидратов)",
    "mg_rock": "магнезиальные породы (ультрамафиты, доломиты)",
    "feldspathic_rock": "полевошпатовые породы",
    "s_source": "источник серы (вулканизм, эвапориты)",
    "copper_body": "первичное медное тело (для зоны окисления)",
}


def _near(mask: np.ndarray, shape, spacing_km: float, radius_km: float = NEAR_KM):
    it = max(1, int(math.ceil(radius_km / max(spacing_km, 1e-6) - 0.5)))
    m2 = np.asarray(mask, dtype=bool).reshape(shape)
    return ndimage.binary_dilation(m2, iterations=it).ravel()


def _f(mask, missing=None) -> np.ndarray:
    out = np.asarray(mask, dtype=np.float64)
    if missing is not None:
        out = np.where(missing, np.nan, out)
    return out


def build_features(ctx, climate=None, hydro=None) -> dict:
    """Feature arrays (n,) of 1 / 0 / NaN. ``climate``: dict with ``t_ann_c``, ``p_ann_mm``,
    ``aridity``, ``waterlogging_months`` (or None: climate features unresolved). ``hydro``:
    dict with ``river``, ``coast``, ``endorheic``, ``lake`` (or None)."""
    L = ctx.lith
    sh, km = ctx.shape, ctx.spacing_km
    unknown_lith = L == LITH["unknown"]

    def lith(*names):
        return np.isin(L, [LITH[k] for k in names])

    land, ocean = ctx.land, ctx.ocean
    F = {"land": _f(land), "ocean": _f(ocean)}
    ancient = np.nan_to_num(ctx.crust_age_ma) > ANCIENT_MA
    F["ancient_crust"] = _f(ancient & ~ocean)
    F["ancient_shield"] = _f(lith("granite_gneiss", "greenstone_belt"), unknown_lith)
    F["greenstone"] = _f(lith("greenstone_belt"), unknown_lith)
    F["preserved_strata"] = _f(ctx.erosion_total_m < 3000.0)
    basins_missing = "basins (v4.6)" in ctx.unknown_fields
    basin = (ctx.basin_type > 0) & (ctx.basin_fill_m > 300.0)
    F["sedimentary_basin"] = _f(basin, np.full(ctx.n, basins_missing))
    F["deep_burial"] = _f(basin & (ctx.basin_fill_m > 1500.0), np.full(ctx.n, basins_missing))
    F["carbonate"] = _f(lith("carbonate_platform"), unknown_lith)
    F["marine_facies"] = _f(
        lith("carbonate_platform", "black_shale", "platform_siliciclastic") | ctx.shelf,
        unknown_lith,
    )
    F["clastic_permeable"] = _f(
        lith("platform_siliciclastic", "red_beds_evaporitic", "unconsolidated_sediment"),
        unknown_lith,
    )
    F["reduced_facies"] = _f(lith("black_shale", "coal_measures"), unknown_lith)
    F["evaporitic"] = _f(lith("red_beds_evaporitic"), unknown_lith)
    F["organic_burial"] = _f(
        lith("coal_measures", "black_shale") & (ctx.basin_fill_m > 600.0), unknown_lith
    )
    F["ultramafic"] = _f(lith("ophiolite_ultramafic", "layered_mafic_intrusion"), unknown_lith)
    F["ophiolite"] = _f(lith("ophiolite_ultramafic"), unknown_lith)
    F["layered_intrusion"] = _f(lith("layered_mafic_intrusion"), unknown_lith)
    F["mafic"] = _f(
        lith("basaltic_volcanic", "oceanic_basalt", "layered_mafic_intrusion"), unknown_lith
    )
    F["mafic_ultramafic"] = _f(
        lith("ophiolite_ultramafic", "layered_mafic_intrusion", "basaltic_volcanic"),
        unknown_lith,
    )
    F["felsic_intrusion"] = _f(lith("granitoid_intrusion"), unknown_lith)
    F["al_silicate_rock"] = _f(
        lith("granite_gneiss", "granitoid_intrusion", "basaltic_volcanic", "arc_volcanic",
             "metamorphic_schist"), unknown_lith)  # fmt: skip
    F["igneous_rock"] = _f(
        lith("granitoid_intrusion", "basaltic_volcanic", "arc_volcanic", "granite_gneiss",
             "layered_mafic_intrusion"), unknown_lith)  # fmt: skip
    F["sedimentary_rock"] = _f(
        lith("carbonate_platform", "platform_siliciclastic", "red_beds_evaporitic",
             "black_shale", "coal_measures"), unknown_lith)  # fmt: skip
    meta = lith("metamorphic_schist", "granite_gneiss", "greenstone_belt") | ctx.orogen
    F["metamorphic"] = _f(meta, unknown_lith)
    F["volcanic_rock"] = _f(lith("arc_volcanic", "basaltic_volcanic"), unknown_lith)
    F["arc"] = _f(ctx.arc_side | lith("arc_volcanic"))
    # magmatism from the volcanic systems of 5.1 (missing stage: unresolved, not false)
    volc_missing = "volcano systems (v5.1)" in ctx.unknown_fields
    vmask = np.zeros(ctx.n, bool)
    alk = np.zeros(ctx.n, bool)
    ridge = np.zeros(ctx.n, bool)
    for vs in ctx.volcano_systems:
        h = vs.get("hex_flat", vs.get("hex"))
        if h is None or not 0 <= int(h) < ctx.n:
            continue
        vmask[int(h)] = True
        src = vs.get("source_type", "")
        if src in ("continental_hotspot", "divergent_rift", "intraplate"):
            alk[int(h)] = True
        if src in ("mid_ocean_ridge", "back_arc"):
            ridge[int(h)] = True
    vm = np.full(ctx.n, volc_missing)
    magm = _near(vmask, sh, km)
    F["magmatic_activity"] = _f(magm, vm)
    F["alkaline_magmatism"] = _f(_near(alk, sh, km) & ~ocean, vm)
    F["hydrothermal"] = _f(magm | ctx.orogen, vm & ~ctx.orogen)
    F["shear_zone"] = _f(ctx.orogen & ctx.collision)
    F["high_p_low_t"] = _f(
        ctx.arc_side & lith("ophiolite_ultramafic", "metamorphic_schist"), unknown_lith
    )
    peg = lith("granitoid_intrusion") & (ctx.orogen | ancient)
    F["pegmatite"] = _f(peg, unknown_lith)
    F["unconformity"] = _f(
        basin & _near(ancient & lith("granite_gneiss", "greenstone_belt"), sh, km),
        np.full(ctx.n, basins_missing),
    )
    F["be_source_near"] = _f(_near(lith("granitoid_intrusion") | peg, sh, km), unknown_lith)
    F["cr_host_near"] = _f(
        _near(lith("ophiolite_ultramafic", "layered_mafic_intrusion", "black_shale"), sh, km),
        unknown_lith,
    )
    F["carbonate_near_intrusion"] = _f(
        lith("carbonate_platform") & _near(lith("granitoid_intrusion") | magm, sh, km),
        unknown_lith,
    )
    F["submarine_hydrothermal"] = _f(_near(ridge, sh, km) & ocean, vm)
    F["volcanic_ash"] = _f(_near(vmask & ctx.arc_side, sh, 2 * NEAR_KM), vm)
    F["porous_host"] = _f(
        lith("platform_siliciclastic", "carbonate_platform", "unconsolidated_sediment",
             "red_beds_evaporitic"), unknown_lith)  # fmt: skip
    F["seal_rock"] = _f(
        _near(lith("red_beds_evaporitic", "black_shale"), sh, km) & basin, unknown_lith
    )
    F["heat_source"] = _f(magm | (ctx.basin_type == 3), vm)
    depth = ctx.water_depth_m
    F["deep_ocean"] = _f(ocean & (depth > 4000.0))
    F["low_sedimentation"] = _f(ctx.marine_sed_m < 600.0)
    oca = ctx.ocean_crust_age_ma
    F["old_ocean_floor"] = _f(np.nan_to_num(oca) > 10.0, ocean & ~np.isfinite(oca))
    F["hard_substrate"] = _f(ctx.seamount & ocean & (depth > 800.0) & (depth < 3000.0))
    F["shelf"] = _f(ctx.shelf)
    F["upwelling"] = _f(ctx.upwelling)
    F["cold_seafloor"] = _f(ocean & (depth > 500.0) & (ctx.marine_sed_m > 200.0))
    F["mg_rock"] = _f(
        lith("ophiolite_ultramafic", "layered_mafic_intrusion", "carbonate_platform"),
        unknown_lith,
    )
    F["feldspathic_rock"] = _f(
        lith("granitoid_intrusion", "granite_gneiss", "layered_mafic_intrusion"), unknown_lith
    )
    F["s_source"] = _f(magm | lith("red_beds_evaporitic"), vm & ~lith("red_beds_evaporitic"))
    # climate (5.4.1 / 4.5) and hydrology
    if climate is None:
        for k in ("weathering_long", "weathering_any", "karst_surface", "wetland", "arid"):
            F[k] = np.full(ctx.n, np.nan)
    else:
        t, p = np.asarray(climate["t_ann_c"], float), np.asarray(climate["p_ann_mm"], float)
        ar = np.asarray(climate["aridity"], float)
        wl = np.asarray(climate["waterlogging_months"], float)
        F["weathering_long"] = _f(
            land & (t >= 18.0) & (p >= 1200.0) & (ctx.slope_m_km < 40.0), ~np.isfinite(t)
        )
        F["weathering_any"] = _f(land & (t >= 8.0) & (p >= 500.0), ~np.isfinite(t))
        F["karst_surface"] = _f(land & lith("carbonate_platform") & (p >= 600.0), unknown_lith)
        F["wetland"] = _f(land & (wl >= 3.0), ~np.isfinite(wl))
        F["arid"] = _f(land & (ar < 0.3), ~np.isfinite(ar))
    if hydro is None:
        for k in ("river", "coast", "closed_basin", "aquatic_basin"):
            F[k] = np.full(ctx.n, np.nan)
    else:
        F["river"] = _f(hydro["river"])
        F["coast"] = _f(hydro["coast"])
        F["closed_basin"] = _f(np.asarray(hydro["endorheic"], bool) & land)
        F["aquatic_basin"] = _f(
            np.asarray(hydro["lake"], bool) | ctx.shelf | np.asarray(hydro["endorheic"], bool)
        )
    F["copper_body"] = np.full(ctx.n, np.nan)  # set by the plan from primary Cu bodies
    missing = set(FEATURES_RU) - set(F)
    if missing:
        raise KeyError(f"features not built: {sorted(missing)}")
    return F


def features_from_project(project, ctx) -> dict:
    """Climate and hydrology inputs of :func:`build_features` from a generated project."""
    climate = hydro = None
    n = ctx.n
    if project.env is not None:
        E = project.env.layers
        t_m = project.env.monthly.get("temperature_monthly_c") if project.env.monthly else None
        p_m = project.env.monthly.get("precip_monthly_mm") if project.env.monthly else None
        if t_m is not None and p_m is not None:
            climate = {
                "t_ann_c": np.asarray(t_m, float).reshape(12, n).mean(axis=0),
                "p_ann_mm": np.asarray(p_m, float).reshape(12, n).sum(axis=0),
                "aridity": np.asarray(E["aridity_p_over_pet"], float).ravel(),
                "waterlogging_months": np.asarray(E["waterlogging_months"], float).ravel(),
            }
    Hy = project.hydro.layers
    if "river_mask" in Hy:
        from ..views import mask_rim

        land2 = np.asarray(Hy["land_mask"], bool)
        hydro = {
            "river": np.asarray(Hy["river_mask"], bool).ravel(),
            "coast": mask_rim(land2, project.layout).ravel(),
            "endorheic": np.asarray(Hy.get("endorheic_mask", np.zeros_like(land2)), bool).ravel(),
            "lake": (np.asarray(Hy.get("lake_id", -np.ones(land2.shape)), int) >= 0).ravel(),
        }
    return build_features(ctx, climate, hydro)
