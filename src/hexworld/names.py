"""Place names: continents, mountains, rivers, lakes, seas, oceans, deserts, glaciers.

Names come from the seed (the same world always has the same names) and are built from
syllables in Latin script, like the names of plates and landmasses; :func:`ru` writes them
in Cyrillic for the Russian interface. Positions are world km; nothing is stored in the
project — the names are derived on demand and cached on the project object.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import ndimage

from .rng import Stream

_START = ("Ar", "Bal", "Cor", "Dar", "El", "Fen", "Gor", "Hal", "Is", "Kar", "Lor", "Mar",
          "Nar", "Or", "Pel", "Ran", "Sar", "Tor", "Ul", "Var", "Zel", "Ast", "Bor", "Kel",
          "Mir", "Ost", "Tal", "Vel", "Sev", "Dol")  # fmt: skip
_MID = ("a", "e", "i", "o", "u", "an", "en", "ar", "or", "el", "il", "ur", "a", "e")
_END = {
    "mountains": ("rat", "gar", "dun", "kar", "tor", "grim", "vash", "drak", "zan", "ket"),
    "river": ("na", "va", "ra", "la", "ya", "sa", "da", "ka", "ta", "ma"),
    "lake": ("ny", "ra", "via", "ja", "ka", "el", "in", "ol"),
    "sea": ("is", "ys", "ane", "ria", "ion", "ea", "os", "ara"),
    "ocean": ("ic", "ian", "ar", "ean", "is"),
    "desert": ("hab", "sur", "rak", "zid", "dum", "hal", "kesh", "tar"),
    "ice": ("heim", "vik", "fjel", "gard", "brek", "strand"),
}


@dataclass
class Label:
    kind: str  # ocean, sea, landmass, mountains, desert, ice, lake, river
    name: str  # Latin script
    x_km: float
    y_km: float
    #: Rough size of the feature, km (decides from which zoom the label is shown).
    size_km: float
    ref: int = -1  # id of the feature in its own table

    @property
    def title_ru(self) -> str:
        return f"{_PREFIX_RU[self.kind]}{ru(self.name)}{_SUFFIX_RU[self.kind]}"


_PREFIX_RU = {"ocean": "", "sea": "море ", "landmass": "", "mountains": "горы ",
              "desert": "пустыня ", "ice": "ледник ", "lake": "оз. ", "river": "р. "}  # fmt: skip
_SUFFIX_RU = {"ocean": " океан", "sea": "", "landmass": "", "mountains": "", "desert": "",
              "ice": "", "lake": "", "river": ""}  # fmt: skip

_DIGRAPHS = (("sch", "ш"), ("th", "т"), ("sh", "ш"), ("ch", "ч"), ("ph", "ф"), ("qu", "кв"),
             ("ou", "у"), ("ia", "ия"), ("ja", "я"), ("ju", "ю"), ("ya", "я"), ("yu", "ю"),
             ("ye", "е"), ("ck", "к"), ("ea", "еа"), ("kh", "х"), ("zh", "ж"),
             ("ts", "ц"))  # fmt: skip
_LETTERS = dict(
    zip(
        "abcdefghijklmnopqrstuvwxyz",
        "а б к д е ф г х и дж к л м н о п к р с т у в в кс й з".split(),
        strict=True,
    )
)


def ru(name: str) -> str:
    """Latin fantasy name in Cyrillic: «Pelomund» → «Пеломунд»."""
    out = []
    for word in str(name).split(" "):
        if word.isdigit():
            out.append(word)
            continue
        low = word.lower()
        res, i = "", 0
        while i < len(low):
            for a, b in _DIGRAPHS:
                if low.startswith(a, i):
                    res += b
                    i += len(a)
                    break
            else:
                res += _LETTERS.get(low[i], low[i])
                i += 1
        out.append(res[:1].upper() + res[1:])
    return " ".join(out)


def make_names(seed: str, kind: str, count: int) -> list[str]:
    """``count`` distinct names of one kind (deterministic per seed)."""
    stream = Stream(seed, f"names/{kind}")
    ends = _END.get(kind, _END["sea"])
    out, used = [], set()
    for i in range(count):
        for _attempt in range(6):
            name = _START[stream.integer(len(_START))]
            if stream.integer(3) > 0:
                name += _MID[stream.integer(len(_MID))]
            name += ends[stream.integer(len(ends))]
            if name not in used:
                break
        else:
            name = f"{name} {i + 1}"
        used.add(name)
        out.append(name)
    return out


def place_labels(project) -> list[Label]:
    """Every named feature of the window, cached on the project."""
    cached = getattr(project, "_place_labels", None)
    if cached is not None:
        return cached
    labels = _build(project)
    try:
        project._place_labels = labels
    except AttributeError:
        pass
    return labels


def feature_names(project) -> dict:
    """``{kind: {ref: Latin name}}`` for the hex panel."""
    out: dict = {}
    for lab in place_labels(project):
        out.setdefault(lab.kind, {})[lab.ref] = lab.name
    return out


def _build(project) -> list[Label]:
    p = project
    seed = p.config.seed
    lay = p.layout
    cx, cy = lay.centers()
    labels: list[Label] = []

    # -- land masses (names exist since v2); labels go on the visible part ------------------
    lands = p.hist.landmasses if p.hist is not None else (p.geo.landmasses if p.geo else [])
    land_id = p.layer("landmass_id") if (p.hist is not None or p.geo is not None) else None
    centre = _visible_centres(land_id, cx, cy) if land_id is not None else {}
    for m in lands:
        area = float(m.get("area_km2", 0.0))
        at = centre.get(int(m["id"]))
        if area <= 0 or at is None:
            continue
        labels.append(Label("landmass", m["name"], at[0], at[1], area**0.5, int(m["id"])))

    # -- mountain systems --------------------------------------------------------------------
    if p.hist is not None:
        systems = list((p.hist.plan or {}).get("systems", []))
        names = make_names(seed, "mountains", len(systems))
        centre = _visible_centres(p.hist.layers["mountain_system_id"], cx, cy)
        for s, name in zip(systems, names, strict=True):
            at = centre.get(int(s["id"]))
            if at is None:
                continue
            size = max(float(s.get("length_km", 0.0)), float(s.get("area_km2", 0.0)) ** 0.5)
            labels.append(Label("mountains", name, at[0], at[1], size, int(s["id"])))

    # -- oceans and seas: open water far from land ------------------------------------------------
    sea = None
    if p.hydro is not None and "ocean_mask" in p.hydro.layers:
        sea = p.hydro.layers["ocean_mask"]
    elif p.geo is not None:
        sea = ~p.geo.layers["land_mask"]
    if sea is not None and sea.any() and (~sea).any():
        dist = ndimage.distance_transform_edt(sea) * lay.spacing_km
        labels += _open_water(seed, dist, cx, cy, lay.spacing_km)

    # -- lakes ---------------------------------------------------------------------------------
    if p.hydro is not None:
        lake_id = p.hydro.layers["lake_id"]
        big = [lk for lk in p.hydro.lakes if lk["cells"] >= 3]
        names = make_names(seed, "lake", len(big))
        for lk, name in zip(big, names, strict=True):
            m = lake_id == lk["id"]
            if m.any():
                labels.append(Label("lake", name, float(cx[m].mean()), float(cy[m].mean()),
                                    float(lk["area_km2"]) ** 0.5, int(lk["id"])))  # fmt: skip
        labels += _rivers(seed, p)

    # -- deserts and glaciers (v5.0) --------------------------------------------------------------
    bio = getattr(p, "biomes", None)
    if bio is not None:
        from .biomes.registry import BIOMES

        fam = np.array([b.family for b in BIOMES])
        dom = bio.layers["dominant_land_biome"]
        for kind, family, min_hexes in (("desert", "desert", 40), ("ice", "ice", 25)):
            mask = fam[dom] == family
            labels += _regions(seed, kind, mask, cx, cy, lay, min_hexes)
    return labels


def _visible_centres(ids: np.ndarray, cx, cy) -> dict[int, tuple[float, float]]:
    """Label point of every id visible in the window: its member hex nearest the centroid."""
    ids = np.asarray(ids).ravel()
    ok = ids >= 0
    if not ok.any():
        return {}
    x, y, i = cx.ravel()[ok], cy.ravel()[ok], ids[ok].astype(np.int64)
    n = np.bincount(i)
    mx = np.bincount(i, x) / np.maximum(n, 1)
    my = np.bincount(i, y) / np.maximum(n, 1)
    d = (x - mx[i]) ** 2 + (y - my[i]) ** 2
    order = np.lexsort((d, i))
    first = order[np.r_[True, i[order][1:] != i[order][:-1]]]
    return {int(i[k]): (float(x[k]), float(y[k])) for k in first}


def _open_water(seed, dist, cx, cy, spacing_km) -> list[Label]:
    """Seas and oceans at the far-from-land points (non-maximum suppression)."""
    out = []
    # candidates: local maxima of the distance to land (centres of open water bodies)
    peak = (dist == ndimage.maximum_filter(dist, size=5)) & (dist > 0)
    cand = np.flatnonzero(peak)
    flat = cand[np.argsort(-dist.flat[cand])]
    taken: list[tuple[float, float, float]] = []
    names_ocean = make_names(seed, "ocean", 12)
    names_sea = make_names(seed, "sea", 40)
    for idx in flat[:20000]:
        d = float(dist.flat[idx])
        if d < max(250.0, 4 * spacing_km):
            break
        x, y = float(cx.flat[idx]), float(cy.flat[idx])
        # a new body of water must lie outside the circles of the larger ones already named
        if any((x - a) ** 2 + (y - b) ** 2 < max(1.1 * r, 2.0 * d) ** 2 for a, b, r in taken):
            continue
        taken.append((x, y, d))
        # keep the label inside the window (the far-from-land point may be on its edge)
        b_x0, b_x1 = float(cx.min()), float(cx.max())
        b_y0, b_y1 = float(cy.min()), float(cy.max())
        mx, my = 0.08 * (b_x1 - b_x0), 0.06 * (b_y1 - b_y0)
        x = min(max(x, b_x0 + mx), b_x1 - mx)
        y = min(max(y, b_y0 + my), b_y1 - my)
        ocean = d >= 900.0
        pool = names_ocean if ocean else names_sea
        k = sum(1 for lab in out if (lab.kind == "ocean") == ocean)
        if k >= len(pool):
            continue
        out.append(Label("ocean" if ocean else "sea", pool[k], x, y, 2.0 * d, k))
    return out


def _rivers(seed, p) -> list[Label]:
    systems = [r for r in p.hydro.river_systems if r["segments"] >= 3]
    systems = sorted(systems, key=lambda r: -r["discharge_m3_s"])[:60]
    names = make_names(seed, "river", len(systems))
    out = []
    segs = {s["id"]: s for s in p.hydro.rivers}
    for r, name in zip(systems, names, strict=True):
        stem = [segs[i] for i in r["main_stem"] if i in segs]
        xs = [x for s in stem for x in s["x_km"]]
        ys = [y for s in stem for y in s["y_km"]]
        if not xs:
            continue
        mid = len(xs) // 2
        out.append(Label("river", name, float(xs[mid]), float(ys[mid]),
                         float(r["main_stem_length_km"]), int(r["id"])))  # fmt: skip
    return out


def _regions(seed, kind, mask, cx, cy, lay, min_hexes) -> list[Label]:
    lab, n = ndimage.label(mask)
    if n == 0:
        return []
    sizes = np.bincount(lab.ravel())[1:]
    order = [i + 1 for i in np.argsort(-sizes) if sizes[i] >= min_hexes][:20]
    names = make_names(seed, kind, len(order))
    out = []
    for comp, name in zip(order, names, strict=True):
        m = lab == comp
        area = float(m.sum()) * lay.hex_area_km2
        # the label goes on the member hex nearest the centroid (concave shapes)
        x0, y0 = cx[m].mean(), cy[m].mean()
        d = (cx[m] - x0) ** 2 + (cy[m] - y0) ** 2
        i = int(np.argmin(d))
        out.append(Label(kind, name, float(cx[m][i]), float(cy[m][i]), area**0.5, int(comp)))
    return out


def river_system_of_segment(project) -> dict[int, int]:
    """``{river segment id: river system id}`` (tributaries belong to their system)."""
    cached = getattr(project, "_segment_system", None)
    if cached is not None:
        return cached
    up: dict[int, list[int]] = {}
    for s in project.hydro.rivers:
        if s["next_segment"] >= 0:
            up.setdefault(s["next_segment"], []).append(s["id"])
    out = {}
    for r in project.hydro.river_systems:
        stack = [r["mouth_segment"]]
        while stack:
            sid = stack.pop()
            out[sid] = r["id"]
            stack.extend(up.get(sid, []))
    try:
        project._segment_system = out
    except AttributeError:
        pass
    return out


def river_name_at(project, col: int, row: int) -> str | None:
    """Cyrillic name of the river system flowing through the hex, if it has one."""
    if project.hydro is None:
        return None
    cell_seg = getattr(project, "_cell_segment", None)
    if cell_seg is None:
        cell_seg = {}
        lay = project.layout
        for s in project.hydro.rivers:
            cells = s.get("cells")
            if cells is None:
                # a saved project keeps the river polyline (hex centres), not the cell list
                col_, row_ = lay.world_to_offset(np.asarray(s["x_km"]), np.asarray(s["y_km"]))
                ok = lay.inside(col_, row_)
                cells = np.asarray(row_)[ok] * lay.width + np.asarray(col_)[ok]
            for c in np.asarray(cells).tolist():
                cell_seg.setdefault(int(c), s["id"])
        try:
            project._cell_segment = cell_seg
        except AttributeError:
            pass
    seg = cell_seg.get(row * project.layout.width + col)
    if seg is None:
        return None
    system = river_system_of_segment(project).get(seg)
    name = feature_names(project).get("river", {}).get(system)
    return ru(name) if name else None
