"""BioExplanation of TZ 5.5 §15: what lives in a hex and why — from the stored state only
(no solver runs on hover, B78).

Absence is never one word: ``no_habitat`` (the realm has no area here), ``unsuitable``
(a niche or gate excludes; the limiting variable is named), ``not_reached`` (suitable, but
the lineage pool is too far along the graph for the group's dispersal), ``no_food``
(suitable and reached, but the shared food does not support it), ``unknown`` (an input is
missing), and ``present``.
"""

from __future__ import annotations

import numpy as np

from .model import P_MIN, S_MIN, _home, family_of
from .niche import trapezoid
from .registry import KINGDOM_RU, load

REASON_RU = {
    "present": "есть",
    "no_habitat": "нет местообитания этого типа",
    "unsuitable": "условия не подходят",
    "not_reached": "условия подходят, но группа не дошла (изоляция)",
    "no_food": "условия подходят, но не хватает корма",
    "outcompeted": "условия подходят, но место в ярусе занято более приспособленными",
    "not_persistent": "линия рядом, но популяция неустойчива (мала площадь или пригодность)",
    "unknown": "нет данных",
}
VAR_RU = {"tc": "холодный месяц", "tw": "тёплый месяц", "mi": "увлажнение (P/PET)",
          "gdd": "сумма тепла", "ph": "кислотность почвы", "sal": "солёность",
          "survival": "переживание неблагоприятного сезона", "breeding": "сезон размножения",
          "gate": "структура местообитания"}  # fmt: skip
REALM_AREA = {"land": "land_ground", "fresh": "inland_water", "marine": "marine_water"}


def _limiting(tpl, env_vals: dict):
    worst, val = None, 2.0
    for var, pts in tpl.niche.items():
        x = env_vals.get(var)
        if x is None or not np.isfinite(x):
            continue
        s = float(trapezoid(np.array([x]), *pts)[0])
        if s < val:
            worst, val = var, s
    return worst, val


def group_status(state, gid: str, i: int) -> dict:
    """Status of one functional group in hex ``i`` with the reason and the best template."""
    reg = load()
    g = reg.groups[gid]
    realm = _home(g)
    if realm == "water":
        area = float(state.frac["inland_water"][i] + state.frac["marine_water"][i])
    else:
        area = float(state.frac[REALM_AREA[realm]][i])
    out = {"group": gid, "name": g.name_ru, "analogue": g.analogue_ru, "realm": realm,
           "kind": KINGDOM_RU.get(g.kind, g.kind)}  # fmt: skip
    if area <= 0:
        return out | {"status": "no_habitat"}
    tabs = [state.realms[r] for r in (("fresh", "marine") if realm == "water" else (realm,))
            if r in state.realms]  # fmt: skip
    best = None
    for tab in tabs:
        j = np.searchsorted(tab.idx, i)
        if j >= tab.idx.size or tab.idx[j] != i:
            continue
        for k, tid in enumerate(tab.templates):
            if reg.templates[tid].group_id != gid:
                continue
            cand = (int(tab.present[k, j]), int(tab.suit[k, j]), int(tab.occ[k, j]), tid)
            if best is None or cand > best:
                best = cand
    if best is None:
        return out | {"status": "no_habitat"}
    present, s8, o8, tid = best
    tpl = reg.templates[tid]
    out.update({"template": tid, "template_name": tpl.name_ru,
                "suitability": round(s8 / 255.0, 2), "occupancy": round(o8 / 255.0, 2)})  # fmt: skip
    gd = state.groups[gid]
    if "biomass_kgc" in gd:
        out["biomass_kgc"] = float(gd["biomass_kgc"][i])
        out["individuals"] = float(gd["individuals"][i])
    if "cover" in gd:
        out["cover"] = float(gd["cover"][i])
        out["hex_share"] = out["cover"] * area  # share of the whole hex
    lin = state.lineage[g.traits["mob"]]
    fam = family_of(g.traits["mob"])
    li = int(lin["pool"][i])
    names = state.lineage_names.get(fam, [])
    out["lineage"] = names[li] if 0 <= li < len(names) else "—"
    out["distance_km"] = float(lin["d_km"][i]) if np.isfinite(lin["d_km"][i]) else None
    if present:
        tm = state.layers.get("t_month")
        if g.is_fauna and tm is not None:
            s_m = trapezoid(np.asarray(tm[:, i], dtype=np.float64), *g.traits["t_act"])
            act = [m for m in range(12) if s_m[m] >= 0.5]
            if len(act) < 12:
                tag = ("откочёвка или отлёт" if g.traits["migratory"] else
                       "покой (спячка, диапауза)" if g.traits["dormant"] else "снижена")
                out["season"] = {"active_months": [m + 1 for m in act], "otherwise": tag}
        return out | {"status": "present"}
    if s8 < int(S_MIN * 255):
        env = state.layers.get("_env", {})
        vals = {k: float(v[i]) for k, v in env.items()}
        var, sv = _limiting(tpl, vals)
        out["limiting"] = VAR_RU.get(var, "структура местообитания или сезон") if var and sv < 0.5 \
            else "сезон (переживание зимы или засухи, размножение) или структура местообитания"  # fmt: skip
        return out | {"status": "unsuitable"}
    if o8 < int(P_MIN * 255):
        from .graph import KAPPA0, colonisation_rate

        d = out["distance_km"]
        near = d is not None and colonisation_rate(d, g.traits["L_km"]) >= 0.5 * KAPPA0
        return out | {"status": "not_persistent" if near else "not_reached"}
    return out | {"status": "outcompeted" if g.is_producer else "no_food"}


def hex_report(state, i: int, top: int = 6) -> dict:
    """Region-click report of one hex (flat index ``i``)."""
    reg = load()
    L = state.layers
    rep = {"area_km2": state.area_km2,
           "fractions": {k: round(float(state.frac[k][i]), 3) for k in
                         ("land_ground", "inland_water", "marine_water", "grounded_ice")},
           "npp_gc_m2_yr": float(L["npp_gc_m2_yr"][i]),
           "plant_biomass_kgc_m2": float(L["plant_biomass_kgc_m2"][i]),
           "animal_biomass_kgc_km2": float(L["animal_biomass_kgc_km2"][i]),
           "flora_richness": int(L["flora_richness"][i]), "fauna_richness": int(L["fauna_richness"][i]),
           "freshwater_richness": int(L["freshwater_richness"][i]),
           "marine_richness": int(L["marine_richness"][i]),
           "functional_richness": int(L["functional_richness"][i]),
           "expected_richness": round(float(L["expected_richness"][i]), 1),
           "total_richness_index": round(float(L["total_richness_index"][i]), 1),
           "total_richness_percentile": round(float(L["total_richness_percentile"][i]), 1),
           "endemic_taxa": int(L["endemic_taxa"][i]),
           "bioregion": int(L["bioregion"][i]),
           "npp_limit": str(L["_npp_limit_names"][int(L["npp_limit"][i])])
           if "npp_limit" in L else None,
           "index_reference": f"{L.get('_index_version', 'provisional_v1')}: фиксированные "
                              "границы, не калиброваны",
           "quality": state.quality.get("history", "present_only")}  # fmt: skip
    flora, fauna, absent = [], [], []
    for gid, g in reg.groups.items():
        st = group_status(state, gid, i)
        if st["status"] == "present":
            if g.is_flora:
                flora.append(st)
            elif g.is_fauna:
                fauna.append(st)
        elif st["status"] in ("not_reached", "no_food", "not_persistent") or (
                st["status"] == "unsuitable" and st.get("suitability", 0) > 0):
            absent.append(st)
    flora.sort(key=lambda s: -s.get("hex_share", 0.0))
    fauna.sort(key=lambda s: -s.get("biomass_kgc", 0.0))
    vert = [s for s in fauna if reg.groups[s["group"]].traits["mass_kg"] >= 0.005]
    inv = [s for s in fauna if reg.groups[s["group"]].traits["mass_kg"] < 0.005]
    rep["flora"] = flora[:top]
    rep["fauna"] = vert[:top]
    rep["invertebrates"] = inv[:3]
    rep["flora_count"] = len(flora)
    rep["fauna_count"] = len(fauna)
    rep["absent_examples"] = absent[:top]
    return rep


def _months(ms) -> str:
    """Compact month ranges: [4, 5, 6, 9] -> «4–6, 9»."""
    if not ms:
        return "—"
    out, start, prev = [], ms[0], ms[0]
    for m in ms[1:] + [None]:
        if m is not None and m == prev + 1:
            prev = m
            continue
        out.append(f"{start}–{prev}" if prev > start else f"{start}")
        if m is not None:
            start = prev = m
    return ", ".join(out)


def text_ru(rep: dict) -> str:
    def names(items, key):
        return ", ".join(f"{s['analogue']}" + (f" ({s[key]:.0%})" if key == "hex_share" else "")
                         for s in items) or "—"  # fmt: skip

    seas = [f"{s['analogue']}: активность в мес. {_months(s['season']['active_months'])}, в "
            f"остальное время {s['season']['otherwise']}" for s in rep["fauna"] if s.get("season")]
    lines = [
        f"Флора (доля гекса): {names(rep['flora'], 'hex_share')}",
        f"Фауна: {names(rep['fauna'], 'biomass')}",
        f"Беспозвоночные и мелкие: {names(rep.get('invertebrates', []), 'biomass')}",
        *([f"Сезоны: {'; '.join(seas[:3])}"] if seas else []),
        f"Богатство: флора {rep['flora_richness']}, фауна {rep['fauna_richness']} аналогов "
        f"(групп {rep['functional_richness']}); индекс {rep['total_richness_index']:.0f}/100, "
        f"перцентиль мира {rep['total_richness_percentile']:.0f}",
        f"Продуктивность (NPP) {rep['npp_gc_m2_yr']:.0f} гC/м²/год, фитомасса "
        f"{rep['plant_biomass_kgc_m2']:.1f} кгC/м²",
    ]
    if rep.get("npp_limit"):
        lines.append(f"Главный ограничитель продуктивности: {rep['npp_limit']}")
    for s in rep["absent_examples"][:3]:
        why = REASON_RU[s["status"]]
        if s.get("limiting"):
            why += f" — {s['limiting']}"
        if s["status"] == "not_reached" and s.get("distance_km") is not None:
            why += f", эфф. расстояние {s['distance_km']:.0f} км"
        lines.append(f"Нет: {s['analogue']} — {why}")
    lines.append("Аналоги — функциональные группы и их варианты, не реальные виды; параметры "
                 "предварительные.")
    return "\n".join(lines)
