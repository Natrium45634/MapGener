"""Tools around a world's settings: share code, quick sketch, conflicts, partial regeneration
plan, summary for comparison and thumbnails. Qt-free (GUI, CLI and tests use it).
"""

from __future__ import annotations

import base64
import json
import zlib
from dataclasses import replace

import numpy as np

from .config import Config

CODE_PREFIX = "HW2-"
LEGACY_CODE_PREFIX = "HW1-"
CODE_PREFIXES = (CODE_PREFIX, LEGACY_CODE_PREFIX)

#: Pipeline levels in run order: (through key, title, config sections that start it).
LEVELS = (
    ("v1", "Плиты и тектоника", {"seed", "width", "height", "hex_spacing_km", "mode",
                                 "viewport_origin_km", "simulation_bounds_km", "reference_cells",
                                 "plates", "crust", "motion", "boundaries", "fields"}),
    ("v2", "Материки и макрорельеф", {"continents", "relief", "sea"}),
    ("v3", "Геологическая история", {"history", "mountains"}),
    ("v46", "Дно океана и бассейны", {"terrain"}),
    ("v4", "Климат, реки, эрозия", {"climate", "hydrology", "erosion"}),
    ("v45", "Почвы и водный баланс", {"environment"}),
    ("v47", "Высотная поясность", {"topoclimate"}),
    ("v50", "Биомы и лёд", {"biomes"}),
)  # fmt: skip
_ORDER = [k for k, _t, _s in LEVELS]

#: Map tabs that each level feeds (and every later level).
LEVEL_VIEWS = {
    "v1": ("plates",),
    "v2": ("continents", "heights"),
    "v3": ("history", "relief", "massifs", "volcanism"),
    "v46": ("ocean", "basins", "coasts"),
    "v4": ("temperature", "precipitation", "hydrology", "erosion"),
    "v45": ("seasonality", "soils", "moisture", "ecology"),
    "v47": ("belts", "topoclimate", "subhex"),
    "v50": ("biomes", "vegetation", "cryosphere", "wetlands", "eco_limits", "ecozones"),
}


# -- world code --------------------------------------------------------------------------------
def _diff(new: dict, base: dict) -> dict:
    out = {}
    for k, v in new.items():
        b = base.get(k)
        if isinstance(v, dict) and isinstance(b, dict):
            d = _diff(v, b)
            if d:
                out[k] = d
        elif v != b:
            out[k] = v
    return out


def _frozen_defaults(dict_id: int) -> dict:
    return json.loads(_dictionary(dict_id).decode("utf-8"))["c"]


def _decoded_defaults(dict_id: int) -> dict:
    """What decoding builds from the frozen text with no changes: settings added after the
    dictionary was frozen get the value ``Config.from_dict`` gives a missing field, so a
    default of a new setting costs nothing in the code."""
    return Config.from_dict(_frozen_defaults(dict_id)).to_dict()


def _merge(base: dict, diff: dict) -> dict:
    for k, v in diff.items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            _merge(base[k], v)
        else:
            base[k] = v
    return base


def _canonical(config: Config, through: str) -> bytes:
    """Changes of the full settings against the *frozen* defaults of dictionary 1 (not the
    defaults of this build): decoding merges them back into the same frozen text, so every
    build gets the identical full configuration."""
    payload = {"d": _diff(config.to_dict(), _decoded_defaults(1))}
    if through != "v50":
        payload["t"] = through
    return json.dumps(payload, separators=(",", ":"), sort_keys=True, ensure_ascii=False).encode(
        "utf-8"
    )


def world_code(config: Config, through: str = "v50") -> str:
    """A short text that recreates the world exactly: seed + **every** setting.

    ``HW2-`` + base64url(dictionary id, CRC-24, raw deflate of the full settings JSON with a
    frozen dictionary). The settings are stored as changes against the frozen default text
    of the dictionary (never against this build's defaults), so the full configuration is
    restored exactly in every later build; ~60-90 characters for a world with a few changes.
    """
    raw = _canonical(config, through)
    comp = zlib.compressobj(9, zlib.DEFLATED, -15, 9, zlib.Z_DEFAULT_STRATEGY, zdict=_dictionary(1))
    body = comp.compress(raw) + comp.flush()
    crc = (zlib.crc32(raw) & 0xFFFFFF).to_bytes(3, "big")
    packed = base64.urlsafe_b64encode(bytes([1]) + crc + body).decode("ascii").rstrip("=")
    return CODE_PREFIX + packed


def _dictionary(dict_id: int) -> bytes:
    from . import _world_code_dict as wcd

    try:
        return getattr(wcd, f"DICT_{dict_id}").encode("utf-8")
    except AttributeError as exc:
        raise ValueError(f"код мира из более новой версии (словарь {dict_id})") from exc


def is_world_code(text: str) -> bool:
    return "".join(str(text).split()).startswith(CODE_PREFIXES)


def parse_world_code(code: str) -> tuple[Config, str]:
    """``(config, through)`` of a world code (``HW2-`` or the older ``HW1-``); raises
    ``ValueError`` on a bad or damaged code."""
    text = "".join(str(code).split())
    if text.startswith(LEGACY_CODE_PREFIX):
        return _parse_v1(text[len(LEGACY_CODE_PREFIX) :])
    if not text.startswith(CODE_PREFIX):
        raise ValueError("код мира должен начинаться с «HW2-» (или «HW1-»)")
    body = text[len(CODE_PREFIX) :]
    try:
        blob = base64.urlsafe_b64decode(body + "=" * (-len(body) % 4))
        dec = zlib.decompressobj(-15, zdict=_dictionary(blob[0]))
        raw = dec.decompress(blob[4:]) + dec.flush()
    except ValueError:
        raise
    except Exception as exc:  # noqa: BLE001
        raise ValueError(f"код мира повреждён: {exc}") from exc
    if (zlib.crc32(raw) & 0xFFFFFF).to_bytes(3, "big") != blob[1:4]:
        raise ValueError("код мира повреждён: контрольная сумма не совпадает")
    data = json.loads(raw.decode("utf-8"))
    full = _merge(_frozen_defaults(blob[0]), data.get("d", {}))
    return Config.from_dict(full), data.get("t", "v50")


def _parse_v1(body: str) -> tuple[Config, str]:
    try:
        raw = zlib.decompress(base64.urlsafe_b64decode(body + "=" * (-len(body) % 4)))
        data = json.loads(raw.decode("utf-8"))
    except Exception as exc:  # noqa: BLE001
        raise ValueError(f"код мира повреждён: {exc}") from exc

    d = _merge(Config().to_dict(), data.get("d", {}))
    d["seed"] = data["s"]
    return Config.from_dict(d), data.get("t", "v50")


# -- quick sketch ------------------------------------------------------------------------------
def preview_config(config: Config, long_side: int = 96) -> Config:
    """The same world at sketch quality: ~4 s through biomes instead of minutes.

    The window keeps its extent (fewer, larger hexes); the world skeleton is coarse, the
    history runs in larger steps, v4.6 terrain and v4 erosion are skipped, the
    topoclimate is at the lowest quality. Continents, climate belts, deserts, ice and biomes
    come out where they will be; details and exact shapes differ.
    """
    w, h = config.width, config.height
    f = max(w, h) / float(long_side)
    if f > 1.0:
        w2, h2 = max(8, round(w / f)), max(8, round(h / f))
        km = config.hex_spacing_km * (w - 1) / max(w2 - 1, 1)
    else:
        w2, h2, km = w, h, config.hex_spacing_km
    hist = config.history
    step = max(hist.time_step_ma, hist.duration_ma / 8.0 if hist.duration_ma > 0 else 1.0)
    stride = max(1, round(hist.duration_ma / step) if hist.duration_ma > 0 else 1)
    topo = config.topoclimate
    return replace(
        config,
        width=int(w2),
        height=int(h2),
        hex_spacing_km=float(km),
        continents=replace(config.continents, skeleton_cells=160),
        history=replace(hist, time_step_ma=float(step), checkpoint_stride=int(stride)),
        terrain=replace(config.terrain, enabled=False),
        erosion=replace(config.erosion, enabled=False),
        topoclimate=replace(topo, quality="preview"),
        biomes=replace(config.biomes, cryo_spinup_years=min(config.biomes.cryo_spinup_years, 40)),
    )


# -- conflicts ---------------------------------------------------------------------------------
def conflicts(values: dict) -> list[str]:
    """Warnings about settings that contradict each other or will surprise (Russian text).

    ``values`` is the schema dict of :mod:`hexworld.settings_schema` (``None`` = auto).
    """
    v = values.get
    out = []
    if v("biomes.enabled") and not v("climate.enabled"):
        out.append("Биомы считаются по климату: без «Считать климат и реки» их не будет.")
    if v("topoclimate.enabled") and not v("environment.enabled"):
        out.append("Высотные пояса нужны почвы и сезоны (v4.5): включите «Почвы и сезоны».")
    if v("biomes.enabled") and not v("environment.enabled"):
        out.append("Биомы используют месячный водный баланс (v4.5): включите «Почвы и сезоны».")
    plates, majors = v("plates.count"), v("continents.major_landmass_count")
    if plates is not None and majors is not None and majors > plates:
        out.append(f"Крупных материков ({majors}) больше, чем плит ({plates}): столько не выйдет.")
    land = v("sea.target_land_fraction")
    if land is not None and majors is not None and land < 0.12 and majors >= 5:
        out.append(f"Суши всего {land:.0%}, а материков {majors}: они будут совсем маленькими.")
    cc = v("crust.continental_fraction")
    if land is not None and cc is not None and v("sea.sea_mode") == "target_land_fraction":
        if land > cc + 0.03:
            out.append(
                f"Доля суши {land:.0%} больше доли континентальной коры {cc:.0%}: столько "
                "суши не будет — море не опускается ниже края шельфов (иначе материки "
                f"встали бы плоскими плато, а сушей стало бы океанское дно). Суши будет "
                f"около {cc:.0%}; поднимите долю континентальной коры."
            )
    t = v("climate.temperature_offset_c")
    p = v("climate.precipitation_scale")
    if t is not None and t <= -9:
        out.append(f"Мир холоднее Земли на {-t:.0f} °C: большая часть суши — тундра и льды.")
        if p is not None and p >= 1.5:
            out.append("Холодно и очень влажно: вырастут огромные ледниковые щиты.")
    if t is not None and t >= 8:
        out.append(f"Мир теплее Земли на {t:.0f} °C: льдов почти не будет, тропики очень жаркие.")
    if p is not None and p <= 0.4:
        out.append("Осадков очень мало: почти вся суша станет пустыней.")
    if not v("history.enabled") and v("history.enabled") is not None:
        out.append(
            "Без геологической истории нет гор, вулканов и истории плит: настройки вкладки "
            "«Горы, вулканы, история» не действуют."
        )
    if v("history.enabled") and (v("history.duration_ma") or 0) < 5:
        out.append("История короче 5 млн лет: горы едва начнут расти.")
    if v("terrain.island_amount") == 0 and (v("history.hotspot_density") or 0) > 1.5:
        out.append("Островов 0, но горячих точек много: вулканические острова всё равно будут.")
    if not v("erosion.enabled") and v("erosion.enabled") is not None:
        out.append("Без эрозии реки не прорежут долины: рельеф останется «сырым».")
    w, h = v("width") or 0, v("height") or 0
    if w * h >= 1_500_000:
        out.append(f"Очень большая карта ({w}×{h}): генерация займёт много минут и памяти.")
    return out


def affected_views(path: str) -> list[str]:
    """Tabs a parameter changes: its level's tabs and every later level's."""
    from .views import VIEWS

    section = path.split(".")[0].split("[")[0].split(":")[0]
    start = next((i for i, (_k, _t, secs) in enumerate(LEVELS) if section in secs), 0)
    out = []
    for key in _ORDER[start:]:
        out += [VIEWS[v].title for v in LEVEL_VIEWS[key] if v in VIEWS]
    return out


# -- partial regeneration ------------------------------------------------------------------------
def regeneration_plan(old: Config | None, new: Config) -> dict:
    """Which levels the pipeline can take from ``old`` and which it must compute.

    The pipeline compares stage keys (inputs and algorithm versions) and reuses any stage
    whose inputs are unchanged; this predicts that from the changed config sections.
    """
    if old is None:
        return {"first": "v1", "reused": [], "computed": [t for _k, t, _s in LEVELS]}
    a, b = old.to_dict(), new.to_dict()
    changed = {k for k in b if a.get(k) != b.get(k)}
    first = len(LEVELS)
    for i, (_key, _title, sections) in enumerate(LEVELS):
        if changed & sections:
            first = i
            break
    return {
        "first": _ORDER[first] if first < len(LEVELS) else None,
        "reused": [t for _k, t, _s in LEVELS[:first]],
        "computed": [t for _k, t, _s in LEVELS[first:]],
    }


# -- summary and thumbnails ---------------------------------------------------------------------
def world_summary(project) -> dict:
    """A handful of numbers that describe a world (for the library and comparison)."""
    p = project
    s: dict = {
        "seed": p.config.seed,
        "size": f"{p.config.width}×{p.config.height}",
        "hex_km": p.config.hex_spacing_km,
        "plates": len(p.visible_plates()),
    }
    land = None
    if p.hydro is not None:
        L = p.hydro.layers
        land = L["land_mask"]
        rel = L["elevation_m"] - p.hydro.sea_level_m
        hs = p.hydro.summary
        s.update(
            land_share=float(land.mean()),
            highest_m=float(rel.max()),
            deepest_m=float(-rel.min()),
            land_t_c=hs.get("land_mean_temperature_c"),
            land_p_mm=hs.get("land_mean_precipitation_mm_yr"),
            dry_share=1.0 - float(hs["aridity_shares"]["humid"]),
            arid_share=float(hs["aridity_shares"]["hyper_arid"] + hs["aridity_shares"]["arid"]),
            rivers=int(hs.get("rivers", 0)),
            lakes=int(hs.get("lakes", 0)),
        )
    elif p.geo is not None:
        land = p.geo.layers["land_mask"]
        s["land_share"] = float(land.mean())
    if p.hist is not None:
        lands = [m for m in p.hist.landmasses if m.get("area_km2", 0) > 0]
        s["landmasses"] = len(lands)
        if lands:
            big = max(lands, key=lambda m: m["area_km2"])
            from .names import ru

            s["largest_landmass"] = (
                f"{ru(big.get('name', ''))} ({big['area_km2'] / 1e6:.1f} млн км²)"
            )
        s["mountain_systems"] = len((p.hist.plan or {}).get("systems", []))
        s["volcanoes"] = len(p.hist.volcanoes)
    bio = getattr(p, "biomes", None)
    if bio is not None:
        from .biomes.registry import BIOMES

        dom = bio.layers["dominant_land_biome"]
        m = land if land is not None and land.any() else np.ones(dom.shape, bool)
        ids, counts = np.unique(dom[m], return_counts=True)
        order = np.argsort(-counts)[:5]
        s["top_biomes"] = [
            (BIOMES[int(ids[i])].ru, float(counts[i] / max(m.sum(), 1))) for i in order
        ]
        ice = bio.layers.get("ice_thickness_m")
        if ice is not None and land is not None and land.any():
            s["ice_share"] = float((np.nan_to_num(ice[land]) > 1.0).mean())
    return s


SUMMARY_ROWS = (
    ("size", "Размер, гексов", "{}"),
    ("hex_km", "Гекс, км", "{:g}"),
    ("plates", "Плит в окне", "{}"),
    ("land_share", "Доля суши", "{:.0%}"),
    ("landmasses", "Массивов суши", "{}"),
    ("largest_landmass", "Крупнейший материк", "{}"),
    ("highest_m", "Высшая точка, м", "{:,.0f}"),
    ("deepest_m", "Наибольшая глубина, м", "{:,.0f}"),
    ("mountain_systems", "Горных систем", "{}"),
    ("volcanoes", "Вулканов", "{}"),
    ("land_t_c", "Температура суши, °C", "{:+.1f}"),
    ("land_p_mm", "Осадки на суше, мм/год", "{:,.0f}"),
    ("dry_share", "Засушливые земли", "{:.0%}"),
    ("arid_share", "Пустыни (P/PET < 0,2)", "{:.0%}"),
    ("ice_share", "Суша подо льдом", "{:.0%}"),
    ("rivers", "Рек", "{}"),
    ("lakes", "Озёр", "{}"),
)


def summary_rows(summary: dict) -> list[tuple[str, str]]:
    """``(label, text)`` rows of a summary, in a fixed order (missing values are "—")."""
    out = []
    for key, label, fmt in SUMMARY_ROWS:
        v = summary.get(key)
        out.append((label, "—" if v is None else fmt.format(v).replace(",", " ")))
    for i, (name, share) in enumerate(summary.get("top_biomes", [])[:5]):
        out.append((f"Биом №{i + 1}", f"{name} ({share:.0%})"))
    return out


def thumbnail(project, view: str = "biomes", max_px: int = 360) -> np.ndarray:
    """An RGB image of the whole window in one of the map views (falls back to heights)."""
    from . import render, views

    if view not in views.VIEWS or not views.VIEWS[view].availability(project)[0]:
        view = "heights" if views.VIEWS["heights"].availability(project)[0] else "plates"
    spec = views.VIEWS[view]
    rgb = views.view_rgb(project, view, spec.default_mode(), spec.default_overlays())
    b = project.geometry.viewport_bounds
    k = max(b.width, b.height) / float(max_px)
    w, h = max(1, int(b.width / k)), max(1, int(b.height / k))
    return render.render_view(project.layout, rgb, b.xmin, b.ymin, k, w, h, outline=False)
