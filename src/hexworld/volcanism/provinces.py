"""Volcanic provinces from the tectonics: where magma comes from and how much.

* **subduction_arc** - the arc volcanoes of v3 grouped by the plate pair (overriding,
  subducting) and by distance; the front lies on the overriding plate.
* **oceanic_hotspot / continental_hotspot** - every v3 hotspot (mantle frame) with the
  age-ordered track its plate carried away (1 mm/yr x 1 Ma = 1 km).
* **mid_ocean_ridge** - spreading-axis segments of v4.6 (``ridge_piece_id``), axis from the
  hexes of the segment, supply from the half-spreading rate.
* **divergent_rift** - continental rift segments of v4.6 (``rift_graben`` basins): axis,
  length, opening rate.
* **sedimentary_fluid_vent** - mud volcanoes of thick, overpressured basins: kept apart,
  not magmatic (no lava, no VEI).

Collisions and transform boundaries get no province unless one of the above applies.
``volcanic_potential`` of v3 is a diagnostic; it is never read as a probability here.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

from .units import stable_hash

#: Long-term magma supply per unit source strength, kg/s [H] (a 1 m/yr-of-melt scale).
FLUX_REF = {
    "subduction_arc": 1.0e3,  # per arc province (tens of volcanoes)
    "oceanic_hotspot": 6.0e3,  # Hawaii ~ 1e4 kg/s
    "continental_hotspot": 1.5e3,
    "mid_ocean_ridge": 4.0e3,  # per ~100-300 km segment
    "divergent_rift": 8.0e2,
    "sedimentary_fluid_vent": 0.0,
}


@dataclass(frozen=True)
class VolcanicProvince:
    province_id: str
    source_type: str
    #: axis / track / front as a polyline in world km
    x_km: tuple[float, ...]
    y_km: tuple[float, ...]
    width_km: float
    epoch_created_ma: float  # Ma before present the province began
    source_strength: float  # dimensionless, ~0..3
    source_flux_kg_s: float
    source_quality: str  # "derived" | "approximate"
    magma_profile_id: str
    plates: tuple[int, ...]
    active: bool
    note: str = ""

    def to_dict(self) -> dict:
        d = asdict(self)
        d["x_km"] = [round(v, 2) for v in self.x_km]
        d["y_km"] = [round(v, 2) for v in self.y_km]
        return d


def _pid(seed, kind, *key) -> str:
    return f"P-{kind[:3]}-{stable_hash(seed, 'province', kind, *key)[:10]}"


def _axis(xs, ys):
    """Principal axis of points: (cx, cy, ux, uy, half_length)."""
    xs, ys = np.asarray(xs, float), np.asarray(ys, float)
    cx, cy = xs.mean(), ys.mean()
    if xs.size < 2:
        return cx, cy, 1.0, 0.0, 0.0
    m = np.cov(np.vstack([xs - cx, ys - cy]))
    w, v = np.linalg.eigh(m)
    ux, uy = v[:, -1]
    t = (xs - cx) * ux + (ys - cy) * uy
    return cx, cy, float(ux), float(uy), float(max(abs(t.min()), abs(t.max())))


def _sample(project, layer, x, y, default=np.nan):
    """Value of a hex layer under a world point (``default`` outside the window)."""
    lay = project.layout
    col, row = lay.world_to_offset(x, y)
    col, row = int(col), int(row)
    if not lay.inside(col, row):
        return default
    return project.layer(layer)[row, col]


def build_provinces(project, params) -> tuple[list[VolcanicProvince], dict]:
    """All provinces of the world (also outside the window, for ash and plates)."""
    seed = project.config.seed
    hist = project.hist
    vols = hist.volcanoes if hist is not None else []
    out: list[VolcanicProvince] = []
    groups: dict = {}

    # -- arcs ----------------------------------------------------------------------------
    arcs = [v for v in vols if v["kind"] == "arc"]
    by_pair: dict = {}
    for v in arcs:
        by_pair.setdefault((v["plate"], v.get("subducting_plate")), []).append(v)
    for (plate, sub), vs in sorted(by_pair.items(), key=lambda kv: str(kv[0])):
        # split a pair into chains: vents closer than 400 km belong together
        xs = np.array([v["x_km"] for v in vs])
        ys = np.array([v["y_km"] for v in vs])
        label = -np.ones(len(vs), dtype=int)
        n = 0
        for i in range(len(vs)):
            if label[i] >= 0:
                continue
            stack = [i]
            label[i] = n
            while stack:
                j = stack.pop()
                d = np.hypot(xs - xs[j], ys - ys[j])
                for k in np.nonzero((d < 400.0) & (label < 0))[0]:
                    label[k] = n
                    stack.append(int(k))
            n += 1
        for g in range(n):
            members = [vs[i] for i in np.nonzero(label == g)[0]]
            mx = [m["x_km"] for m in members]
            my = [m["y_km"] for m in members]
            cx, cy, ux, uy, half = _axis(mx, my)
            t = np.array([(a - cx) * ux + (b - cy) * uy for a, b in zip(mx, my, strict=True)])
            order = np.argsort(t)
            rate = []
            cont = []
            for m in members:
                r = _sample(project, "trench_rate_mm_yr", m["x_km"], m["y_km"])
                if np.isfinite(r) and r > 0:
                    rate.append(float(r))
                c = _sample(project, "crust_type", m["x_km"], m["y_km"], default=-1)
                if c >= 0:
                    cont.append(int(c))
            conv = float(np.median(rate)) if rate else 50.0
            continental = (np.mean(cont) > 0.5) if cont else False
            strength = float(np.clip(conv / 50.0, 0.2, 3.0))
            first = min(m["birth_ma"] + m["age_ma"] for m in members)
            pid = _pid(seed, "arc", plate, sub, round(cx, -1), round(cy, -1))
            out.append(VolcanicProvince(
                province_id=pid, source_type="subduction_arc",
                x_km=tuple(float(mx[i]) for i in order), y_km=tuple(float(my[i]) for i in order),
                width_km=60.0, epoch_created_ma=float(first),
                source_strength=strength,
                source_flux_kg_s=FLUX_REF["subduction_arc"] * strength * len(members) ** 0.5,
                source_quality="derived" if rate else "approximate",
                magma_profile_id="arc_continental" if continental else "arc_oceanic",
                plates=(int(plate), int(sub) if sub is not None else -1),
                active=any(m["active"] for m in members),
                note="front on the overriding plate (v3 arc line)",
            ))  # fmt: skip
            for m in members:
                groups[("v3", m["id"])] = pid

    # -- hotspots ----------------------------------------------------------------------------
    spots = hist.hotspots if hist is not None else []
    for h in spots:
        track = sorted((v for v in vols if v["kind"] == "hotspot" and v["hotspot_id"] == h["id"]),
                       key=lambda v: v["age_ma"])  # fmt: skip
        if not track:
            continue
        young = track[0]
        c = _sample(project, "crust_type", young["x_km"], young["y_km"], default=-1)
        continental = (c == 1) if c >= 0 else (not young.get("submarine", True))
        kind = "continental_hotspot" if continental else "oceanic_hotspot"
        strength = float(h["strength"]) * float(params.hotspot_activity)
        pid = _pid(seed, "hotspot", h["id"])
        out.append(VolcanicProvince(
            province_id=pid, source_type=kind,
            x_km=tuple(float(v["x_km"]) for v in track),
            y_km=tuple(float(v["y_km"]) for v in track),
            width_km=150.0 if continental else 90.0,
            epoch_created_ma=float(max(v["age_ma"] for v in track)),
            source_strength=strength, source_flux_kg_s=FLUX_REF[kind] * strength,
            source_quality="derived",
            magma_profile_id="continental_hotspot" if continental else "ocean_island_basalt",
            plates=tuple(sorted({int(v["plate"]) for v in track})),
            active=strength > 0 and any(v["active"] for v in track),
            note=f"hotspot {h['id']} at ({h['x_km']:.0f}, {h['y_km']:.0f}) km, mantle frame",
        ))  # fmt: skip
        for v in track:
            groups[("v3", v["id"])] = pid

    terrain = getattr(project, "terrain", None)
    lay = project.layout
    cx_all, cy_all = lay.centers()

    # -- mid-ocean ridges ---------------------------------------------------------------------
    if params.ridge_volcanism and terrain is not None and "ridge_piece_id" in terrain.layers:
        piece = terrain.layers["ridge_piece_id"]
        dist = project.layer("spreading_axis_distance_km") if project.hist else None
        rate = terrain.layers.get("spreading_half_rate_mm_yr")
        on_axis = (piece >= 0) & (dist is not None) & (dist <= 0.75 * lay.spacing_km)
        if project.hydro is not None:
            on_axis &= project.hydro.layers["ocean_mask"]
        for pc in np.unique(piece[on_axis]):
            m = on_axis & (piece == pc)
            if m.sum() < 2:
                continue
            xs, ys = cx_all[m], cy_all[m]
            cx, cy, ux, uy, half = _axis(xs, ys)
            half_rate = float(np.nanmedian(rate[m])) if rate is not None else 25.0
            strength = float(np.clip(half_rate / 25.0, 0.1, 3.0))
            pid = _pid(seed, "ridge", int(pc))
            out.append(VolcanicProvince(
                province_id=pid, source_type="mid_ocean_ridge",
                x_km=(cx - ux * half, cx + ux * half), y_km=(cy - uy * half, cy + uy * half),
                width_km=20.0, epoch_created_ma=0.0, source_strength=strength,
                source_flux_kg_s=FLUX_REF["mid_ocean_ridge"] * strength * max(half, 50.0) / 150.0,
                source_quality="derived", magma_profile_id="mor_basalt",
                plates=(), active=True, note=f"ridge segment {int(pc)}",
            ))  # fmt: skip

    # -- continental rifts ----------------------------------------------------------------------
    if params.rift_volcanism and terrain is not None:
        rifts: dict = {}
        for b in terrain.plans.get("basins", []):
            if b.get("basin_type") == "rift_graben" and b.get("hexes", 0) > 0:
                rifts.setdefault(b.get("rift_id", b.get("basin_id")), []).append(b)
        for rid, segs in sorted(rifts.items(), key=lambda kv: str(kv[0])):
            xs, ys = [], []
            opening = []
            for b in segs:
                cx, cy = b["center_km"]
                ax = b.get("axis") or (1.0, 0.0)
                h = float(b.get("segment_length_km") or 2 * b.get("half_length_km", 50.0)) / 2
                xs += [cx - ax[0] * h, cx + ax[0] * h]
                ys += [cy - ax[1] * h, cy + ax[1] * h]
                opening.append(float(b.get("opening_mm_yr") or 5.0))
            cx, cy, ux, uy, half = _axis(xs, ys)
            strength = float(np.clip(np.mean(opening) / 8.0, 0.1, 2.0))
            pid = _pid(seed, "rift", rid)
            out.append(VolcanicProvince(
                province_id=pid, source_type="divergent_rift",
                x_km=(cx - ux * half, cx + ux * half), y_km=(cy - uy * half, cy + uy * half),
                width_km=float(np.mean([b.get("half_width_km", 30.0) for b in segs])) * 2,
                epoch_created_ma=0.0, source_strength=strength,
                source_flux_kg_s=FLUX_REF["divergent_rift"] * strength,
                source_quality="approximate", magma_profile_id="rift_alkali_basalt",
                plates=(), active=True, note=f"rift {rid}, {len(segs)} segments",
            ))  # fmt: skip

    # -- mud volcanoes (non-magmatic) -------------------------------------------------------------
    mud = []
    if params.mud_volcanoes and terrain is not None:
        for b in terrain.plans.get("basins", []):
            if b.get("basin_type") not in ("foreland_foredeep", "intracratonic_sag"):
                continue
            if b.get("hexes", 0) <= 0:
                continue
            fill = float(b.get("subsidence_m", 0.0)) * float(b.get("fill_fraction", 0.0))
            if fill < 2500.0:  # thick, rapidly buried sediment builds overpressure
                continue
            cx, cy = b["center_km"]
            mud.append({
                "vent_id": f"M-{stable_hash(seed, 'mud', b.get('basin_id'))[:10]}",
                "kind": "sedimentary_fluid_vent", "x_km": float(cx), "y_km": float(cy),
                "basin_id": b.get("basin_id"), "sediment_m": round(fill, 1),
                "magmatic": False, "products": ["mud", "brine", "methane"],
            })  # fmt: skip
    return out, {"v3_province": groups, "mud_vents": mud}
