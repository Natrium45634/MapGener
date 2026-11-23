"""Volcanic systems: stable IDs, vents, edifice type, environment, status.

A system belongs to one province and has one reservoir. Its axes are kept apart
(TZ 5.1, section 1): ``edifice_type`` (shape), ``environment`` (from the *present* surface:
connected ocean, depth, ice), ``magmatic_status`` (active / dormant / extinct / uncertain,
from the supply of the source - not from a fixed "no eruption for 10 000 years" rule) and
``activity_phase`` (the short-term state, set by the event scheduler).

IDs are hashes of the world seed, the source and the world-frame position of the vent
(rounded to 1 km), so they do not depend on the window, its resolution or the order of
the objects.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field

import numpy as np

from .magma import DEFAULT_INTRUSIVE_SHARE, INTRUSIVE_SHARE, Magma, Reservoir, magma_for
from .units import stable_hash, stable_uniform

#: Critical (eruptible) melt of a reservoir by edifice, kg [H].
CRITICAL_KG = {
    "shield": 2.0e13,
    "stratocone": 5.0e12,
    "lava_dome_complex": 3.0e12,
    "caldera_complex": 2.0e14,
    "scoria_cone": 2.0e10,
    "fissure_field": 3.0e12,
    "flood_basalt_province": 1.0e15,
    "submarine_seamount": 5.0e12,
    "subglacial_tuya": 3.0e12,
    "maar_tuff_ring": 1.0e10,
    "mixed": 5.0e12,
}


@dataclass
class VolcanoSystem:
    volcano_id: str
    province_id: str
    source_type: str
    edifice_type: str
    environment: str
    magmatic_status: str
    activity_phase: str
    x_km: float
    y_km: float
    vents: list = field(default_factory=list)  # [{vent_id, x_km, y_km, kind}]
    summit_m: float = 0.0  # above the sea level
    radius_km: float = 10.0
    formed_ma: float = 0.0
    last_activity_ma: float = 0.0  # geological: Ma before present
    last_eruption_years_bp: float | None = None
    reactivation_probability: float = 0.0  # within the event window
    status_confidence: float = 0.5
    lava_lake_state: str = "none"
    supereruption_eligible: bool = False
    in_window: bool = False
    hex: tuple[int, int] | None = None
    geometry_quality: str = "hex_scale"
    v3_id: int | None = None
    magma: Magma | None = None
    reservoir: Reservoir | None = None
    depth_m: float = 0.0  # water above the vent (connected ocean), m
    ice_cover: float = 0.0  # 0..1 ice / perennial snow over the vent
    notes: list = field(default_factory=list)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["magma"] = self.magma.to_dict() if self.magma else None
        d["reservoir"] = self.reservoir.to_dict() if self.reservoir else None
        return d


def _vid(seed, kind, x, y, extra=""):
    return f"V-{stable_hash(seed, 'volcano', kind, round(x), round(y), extra)[:10]}"


def _environment(project, x, y, summit_above_sea: float, submarine_hint: bool | None):
    """(environment, depth_m, ice_cover, hex) from the present surface under the vent."""
    lay = project.layout
    col, row = lay.world_to_offset(x, y)
    col, row = int(col), int(row)
    if not lay.inside(col, row):
        if submarine_hint:
            depth = max(0.0, -summit_above_sea)
            env = "deep_submarine" if depth > 500 else "shallow_submarine"
            return env, depth, 0.0, None
        return "land", 0.0, 0.0, None
    hx = (col, row)
    ocean = bool(project.hydro.layers["ocean_mask"][row, col]) if project.hydro else False
    sea = project.hydro.sea_level_m if project.hydro else 0.0
    z = float(project.layer("elevation_m")[row, col]) - sea
    top = max(z, summit_above_sea) if not ocean else summit_above_sea
    ice = 0.0
    if project.hydro is not None:
        ice = float(np.nan_to_num(project.hydro.layers["glacial_potential"][row, col]))
    topo = getattr(project, "topo", None)
    if topo is not None and "perennial_snow_fraction" in topo.layers:
        ice = max(ice, float(np.nan_to_num(topo.layers["perennial_snow_fraction"][row, col])))
    if ocean and top < 0:
        depth = -top
        return ("deep_submarine" if depth > 500.0 else "shallow_submarine"), depth, 0.0, hx
    if ice > 0.5:
        return "subglacial", 0.0, ice, hx
    if ocean:
        return "island_emergent", 0.0, ice, hx
    if top > 4500.0:
        return "ice_free_high_altitude", 0.0, ice, hx
    return "land", 0.0, ice, hx


def _status(active: bool, province_active: bool, idle_ma: float, seed, vid):
    """magmatic status, reactivation probability (window), confidence."""
    if active and province_active:
        return "active_system", 1.0, 0.9
    if province_active and idle_ma < 2.0:
        p = 0.25 * math.exp(-idle_ma / 1.0)
        return "dormant_system", p, 0.6
    if not province_active and idle_ma > 1.0:
        return "extinct_system", 0.0, 0.8
    if idle_ma > 6.0:
        return "extinct_system", 0.0, 0.7
    return "uncertain", 0.02, 0.3


def build_systems(project, provinces, groups, params, margin_km: float = 600.0):
    """Systems of the window and of a margin around it (their ash can reach the window)."""
    seed = project.config.seed
    b = project.geometry.viewport_bounds
    prov = {p.province_id: p for p in provinces}

    def near(x, y):
        return (b.xmin - margin_km <= x <= b.xmax + margin_km
                and b.ymin - margin_km <= y <= b.ymax + margin_km)  # fmt: skip

    def inside(x, y):
        return b.xmin <= x <= b.xmax and b.ymin <= y <= b.ymax

    out: list[VolcanoSystem] = []
    hist = project.hist
    sea = project.hist.sea_level_m if hist is not None else 0.0
    for v in hist.volcanoes if hist is not None else []:
        x, y = float(v["x_km"]), float(v["y_km"])
        if not near(x, y):
            continue
        pid = groups.get(("v3", v["id"]))
        if pid is None:
            continue
        p = prov[pid]
        vid = _vid(seed, v["kind"], x, y, f"v3:{v['birth_ma']:.4f}")
        raw = v.get("summit_elevation_m")
        if raw is None:  # outside the window: the edifice height over a typical base
            base = -3500.0 if v.get("submarine") else 300.0
            raw = sea + base + float(v.get("edifice_height_m", 0.0))
        summit = float(raw) - sea
        idle = float(v["age_ma"]) if not v["active"] else 0.0
        status, react, conf = _status(bool(v["active"]), p.active, idle, seed, vid)
        u = stable_uniform(seed, vid, "edifice")
        if p.source_type == "subduction_arc":
            mg = magma_for(p.magma_profile_id, seed, vid)
            if mg.log10_viscosity_pa_s > 7.0 and u < 0.35:
                edifice = "lava_dome_complex"
            elif v["edifice_height_m"] > 3000 and u > 0.93:
                edifice = "caldera_complex"
            else:
                edifice = "stratocone"
        elif p.source_type == "continental_hotspot":
            mg = magma_for("continental_hotspot", seed, vid)
            edifice = "caldera_complex" if u < 0.5 else "fissure_field"
        else:
            mg = magma_for(p.magma_profile_id, seed, vid)
            edifice = "shield"
        env, depth, ice, hx = _environment(project, x, y, summit, v.get("submarine"))
        if env in ("deep_submarine", "shallow_submarine") and edifice == "shield":
            edifice = "submarine_seamount"
        if env == "subglacial" and edifice in ("shield", "stratocone") and u > 0.7:
            edifice = "subglacial_tuya"
        sys_ = VolcanoSystem(
            volcano_id=vid, province_id=pid, source_type=p.source_type, edifice_type=edifice,
            environment=env, magmatic_status=status, activity_phase="quiet", x_km=x, y_km=y,
            vents=[{"vent_id": vid + "-c", "x_km": x, "y_km": y, "kind": "central"}],
            summit_m=summit, radius_km=float(v["radius_km"]),
            formed_ma=float(v["age_ma"]),
            last_activity_ma=idle, reactivation_probability=react, status_confidence=conf,
            in_window=inside(x, y), hex=hx, v3_id=int(v["id"]), magma=mg, depth_m=depth,
            ice_cover=ice,
        )  # fmt: skip
        sys_.supereruption_eligible = (
            edifice == "caldera_complex" and mg.silica_wt_pct > 65 and p.source_strength > 0.8
        ) or (edifice == "caldera_complex" and p.source_type == "continental_hotspot")
        if sys_.supereruption_eligible:
            sys_.magma = magma_for("rhyolite_caldera", seed, vid)
        _attach_reservoir(sys_, p, provinces_count=_members(groups, pid))
        out.append(sys_)

    # -- ridges: fissure segments along the axis (mostly deep submarine) ----------------------
    for p in provinces:
        if p.source_type != "mid_ocean_ridge":
            continue
        x0, x1 = p.x_km
        y0, y1 = p.y_km
        length = math.hypot(x1 - x0, y1 - y0)
        n = max(1, min(8, int(length // 120.0) + 1))
        for k in range(n):
            f = (k + 0.5) / n
            x, y = x0 + (x1 - x0) * f, y0 + (y1 - y0) * f
            if not near(x, y):
                continue
            vid = _vid(seed, "ridge", x, y, p.province_id)
            env, depth, ice, hx = _environment(project, x, y, -2600.0, True)
            ux, uy = (x1 - x0) / max(length, 1e-6), (y1 - y0) / max(length, 1e-6)
            seg = 0.5 * length / n
            vents = [{"vent_id": f"{vid}-f{i}", "x_km": x + ux * seg * t, "y_km": y + uy * seg * t,
                      "kind": "fissure"} for i, t in enumerate((-0.8, -0.3, 0.3, 0.8))]  # fmt: skip
            s = VolcanoSystem(
                volcano_id=vid, province_id=p.province_id, source_type="mid_ocean_ridge",
                edifice_type="fissure_field", environment=env, magmatic_status="active_system",
                activity_phase="quiet", x_km=x, y_km=y, vents=vents, summit_m=-depth,
                radius_km=seg, in_window=inside(x, y), hex=hx,
                magma=magma_for("mor_basalt", seed, vid), depth_m=depth,
                reactivation_probability=1.0, status_confidence=0.8,
            )  # fmt: skip
            s.notes.append(f"fissure axis ({ux:+.2f}, {uy:+.2f})")
            _attach_reservoir(s, p, provinces_count=n)
            out.append(s)

    # -- rifts: monogenetic fields of scoria cones and fissures along the axis --------------------
    for p in provinces:
        if p.source_type != "divergent_rift":
            continue
        x0, x1 = p.x_km
        y0, y1 = p.y_km
        length = math.hypot(x1 - x0, y1 - y0)
        ux, uy = (x1 - x0) / max(length, 1e-6), (y1 - y0) / max(length, 1e-6)
        n_cones = int(np.clip(length / 25.0 * p.source_strength, 6, 60))
        vents = []
        for k in range(n_cones):
            a = stable_uniform(seed, p.province_id, "cone", k, "a")
            c = stable_uniform(seed, p.province_id, "cone", k, "c")
            t = (a - 0.5) * length
            w = (c - 0.5) * p.width_km * 0.8
            vx = 0.5 * (x0 + x1) + ux * t - uy * w
            vy = 0.5 * (y0 + y1) + uy * t + ux * w
            vents.append({"vent_id": f"{p.province_id}-cone{k}", "x_km": vx, "y_km": vy,
                          "kind": "scoria_cone",
                          "age_years": round(2e5 * stable_uniform(seed, p.province_id, k, "age")
                                             ** 1.5)})  # fmt: skip
        cx, cy = 0.5 * (x0 + x1), 0.5 * (y0 + y1)
        if not near(cx, cy):
            continue
        vid = _vid(seed, "rift", cx, cy, p.province_id)
        env, depth, ice, hx = _environment(project, cx, cy, 300.0, False)
        s = VolcanoSystem(
            volcano_id=vid, province_id=p.province_id, source_type="divergent_rift",
            edifice_type="scoria_cone", environment=env, magmatic_status="active_system",
            activity_phase="quiet", x_km=cx, y_km=cy, vents=vents, summit_m=300.0,
            radius_km=0.5 * length, in_window=inside(cx, cy), hex=hx,
            magma=magma_for("rift_alkali_basalt", seed, vid), reactivation_probability=1.0,
            status_confidence=0.6, geometry_quality="field_of_vents",
        )  # fmt: skip
        s.notes.append(f"{n_cones} cones along the rift axis ({ux:+.2f}, {uy:+.2f})")
        _attach_reservoir(s, p, provinces_count=1)
        out.append(s)
    out.sort(key=lambda s: s.volcano_id)
    return out


def _members(groups, pid) -> int:
    return max(1, sum(1 for k, v in groups.items() if v == pid))


def _attach_reservoir(s: VolcanoSystem, p, provinces_count: int) -> None:
    """Share of the province supply and a reservoir that fits the edifice."""
    share = p.source_flux_kg_s / max(provinces_count, 1)
    supply = share if s.magmatic_status == "active_system" else share * 0.05
    if s.magmatic_status == "extinct_system":
        supply = 0.0
    crit = CRITICAL_KG.get(s.edifice_type, 5e12)
    fill = 0.3 + 0.6 * stable_uniform(s.volcano_id, "fill") if supply > 0 else 0.02
    s.reservoir = Reservoir.steady(
        supply, crit, fill, INTRUSIVE_SHARE.get(s.source_type, DEFAULT_INTRUSIVE_SHARE)
    )
