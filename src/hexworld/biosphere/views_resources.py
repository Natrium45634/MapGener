"""Map tab «Ресурсы мира: природные и минеральные» — one map with one legend for the
renewable resources of TZ 5.5 (wood, pasture and game, fish, fertile soils of 5.2) and the
mineral deposits of TZ 5.5.1, for comparing regions at a glance (e.g. for the geopolitics of
a world). Display only: nothing is recomputed; the snapshot (📷, ⚙) prints it at any of
the three resolutions with this legend under the map.

Each land hex is tinted by its leading renewable resource where that resource exceeds a
fixed threshold (the shade gets stronger with the amount); sea hexes by their fish
production. Deposits are glyphs coloured by the 5.5.1 groups, the same colours as the tab
«Недра и месторождения»."""

from __future__ import annotations

import numpy as np

from ..views import LegendItem, Option, ViewSpec, _relief_gray, _shade, mask_rim, register_view

#: renewable classes: key -> (title, colour, scale per km² that gives the full shade)
RENEWABLE = {
    "timber": ("древесина (прирост)", (34, 120, 50), 800.0),  # m³/km²/yr
    "pasture": ("пастбища и дичь", (205, 170, 60), 1.0),  # t/km²/yr of game + forage index
    "farmland": ("плодородные почвы (земледелие)", (150, 200, 70), 1.0),  # 0..1
    "fish_fresh": ("рыба пресных вод", (70, 170, 210), 5.0),  # t/km²/yr
    "fish_marine": ("морской промысел", (30, 90, 200), 10.0),  # t/km²/yr
}
#: a hex shows a resource when it is in the richer part of the world for it (world rank)
THRESHOLD = 60.0
MODES = (("all", "Все ресурсы"), ("renewable", "Только природные (возобновляемые)"),
         ("minerals", "Только минеральные"))  # fmt: skip


def _scores(project):
    """Per-hex score 0..1 of every renewable class (missing stages: zeros)."""
    n = project.layout.width * project.layout.height
    out = {k: np.zeros(n) for k in RENEWABLE}
    b = getattr(project, "biosphere", None)
    if b is not None:
        res = b.resources()["layers"]
        area = b.state.area_km2
        out["timber"] = res["timber_increment_m3_yr"] / area / RENEWABLE["timber"][2]
        game = res["game_production_t_wet_yr"] / area
        forage = res["forage_surplus_t_dm_yr"] / area / 100.0
        out["pasture"] = (game + forage) / RENEWABLE["pasture"][2]
        out["fish_fresh"] = (
            res["fish_fresh_production_t_wet_yr"] / area / RENEWABLE["fish_fresh"][2]
        )
        out["fish_marine"] = res["fish_marine_production_t_wet_yr"] / area / \
            RENEWABLE["fish_marine"][2]  # fmt: skip
    r = getattr(project, "resources", None)
    if r is not None:
        crops = [np.nan_to_num(np.asarray(r.layers[k], float).ravel()) for k in
                 ("suitability_crop_wheat", "suitability_crop_maize", "suitability_crop_rice_paddy",
                  "suitability_crop_cassava") if k in r.layers]  # fmt: skip
        if crops:
            out["farmland"] = np.max(crops, axis=0)
    out = {k: np.clip(np.nan_to_num(v), 0.0, None) for k, v in out.items()}
    # world rank of every class among the hexes that have it (0–100), area-weighted: the
    # leading resource of a hex is the one where it stands out most in the world
    from .metrics import percentile_area_weighted

    ranks = {}
    for k, v in out.items():
        has = v > 1e-6
        r = np.zeros(n)
        if has.any():
            r = np.nan_to_num(percentile_area_weighted(np.where(has, v, np.nan), 1.0, has))
        ranks[k] = r
    return ranks


def _render(project, mode, overlays):
    rgb = _relief_gray(project).astype(np.float64)
    h, w, _ = rgb.shape
    flat = rgb.reshape(-1, 3)
    if mode in ("all", "renewable"):
        sc = _scores(project)
        keys = list(RENEWABLE)
        m = np.stack([sc[k] for k in keys])
        best = m.argmax(axis=0)
        val = m.max(axis=0)
        show = val >= THRESHOLD
        cols = np.array([RENEWABLE[k][1] for k in keys], dtype=np.float64)
        a = np.clip(0.35 + 0.5 * (val - THRESHOLD) / (100.0 - THRESHOLD), 0.35, 0.85)[:, None]
        flat[show] = (1 - a[show]) * flat[show] + a[show] * cols[best[show]]
    if mode in ("all", "minerals") and getattr(project, "minerals", None) is not None:
        from ..minerals import views551 as v551

        tint = v551._render_minerals(project, "all", frozenset(o for o in overlays
                                                              if o != "coast"))  # fmt: skip
        base = _relief_gray(project).astype(np.float64)
        moved = np.any(np.abs(tint.astype(np.float64) - base) > 1.0, axis=2).ravel()
        flat[moved] = 0.5 * flat[moved] + 0.5 * tint.reshape(-1, 3)[moved]
    rgb = flat.reshape(h, w, 3).astype(np.uint8)
    if "coast" in overlays:
        _shade(rgb, mask_rim(project.hydro.layers["land_mask"].astype(bool), project.layout), 0.55)
    return rgb


def _vectors(project, mode, overlays):
    if mode == "renewable" or getattr(project, "minerals", None) is None:
        return []
    from ..minerals import views551 as v551

    return v551._vectors_minerals(project, "all", overlays)


def _legend(project, mode):
    items = [
        LegendItem(
            "Общая карта ресурсов: природные (заливка гекса) и минеральные (значки "
            "месторождений); всё — потенциал, не добыча; параметры предварительные"
        )
    ]
    if mode in ("all", "renewable"):
        if getattr(project, "biosphere", None) is None:
            items.append(LegendItem("природные ресурсы 5.5 не рассчитаны (включите «Природу»)"))
        for _k, (title, color, _s) in RENEWABLE.items():
            items.append(LegendItem(f"природные: {title}", color))
        items.append(LegendItem("заливка — природный ресурс, которым гекс выделяется в мире (ранг "
                                "≥ 60-го перцентиля по площади); насыщеннее — выше "
                                "ранг"))  # fmt: skip
    if mode in ("all", "minerals"):
        if getattr(project, "minerals", None) is None:
            items.append(LegendItem("минеральные ресурсы 5.5.1 не рассчитаны (включите «Недра»)"))
        else:
            from ..minerals import views551 as v551

            for k, t, _c in v551.GROUPS[1:]:
                items.append(LegendItem(f"минеральные: {t}", v551._COLORS[k]))
            items.append(LegendItem("значок крупнее — трубка / штокверк / залежь флюида, мельче "
                                    "— жила; фильтр глубины — флажки «кровля ≤ …»"))  # fmt: skip
    return items


register_view(
    ViewSpec(
        id="resources_world",
        title="Ресурсы мира (природные и минеральные)",
        stage="v551",
        layers=("system_count",),
        modes=tuple(Option(k, t) for k, t in MODES),
        overlays=(
            Option("d500", "кровля ≤ 500 м"),
            Option("d1000", "кровля ≤ 1000 м"),
            Option("surface_only", "Только с выходом на поверхность"),
            Option("coast", "Берег", True),
        ),  # fmt: skip
        note="Одна карта с общей легендой: природные ресурсы (5.5) заливкой, месторождения "
        "(5.5.1) значками",
        render=_render,
        legend=_legend,
        vectors=_vectors,
    )
)
