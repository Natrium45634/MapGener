"""Geological resources of TZ 5.2 part III: settings -> permissive tracts -> deposit systems
-> parametric ore bodies -> exhumation / burial -> in-situ grade and tonnage.

Two kinds of generation, both with per-object random streams:

* **object-anchored** systems grow on discrete world objects of earlier stages: the v5.1
  volcano systems (porphyry, epithermal, skarn, VMS, sea-floor sulfides, carbonatites,
  magmatic Ni-Cu-PGE, sulfur). Their ids and world coordinates do not depend on the hex
  size (TZ N20 / R20);
* **tract** systems: a permissive mask M_d (hard requirements, unknown != false) and a
  suitability S_d (0..1) per hex, connected tracts as provinces, a Poisson count
  N ~ Poisson(lambda_d * integral S dA) and anchors drawn inside the tract with the
  province stream (TZ 14.2). The tract integral is resolution-independent up to the
  sampling of the hex fields; anchors may move between resolutions (declared).

No deposit is placed per hex, no 3-D voxel grid exists: bodies are parametric
(:mod:`hexworld.resources.models`), the dense layers are aggregates.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
from scipy import ndimage
from scipy.spatial import cKDTree

from ..noise import FractalNoise
from ..rng import Stream
from ..volcanism.units import stable_hash
from . import ALGORITHM_VERSION
from .models import (
    ELLIPSOID_SHAPES,
    FAMILIES,
    MODELS,
    axes_for_volume,
    ellipsoid_cap_volume,
    footprint_radius_m,
    poisson_from_uniform,
    sample_grade_tonnage,
    shape_volume,
)

#: inferred lithology of the adapter (never written over ``rock_type``)
LITHOLOGY = (
    "unknown",
    "oceanic_basalt",
    "granite_gneiss",
    "greenstone_belt",
    "platform_siliciclastic",
    "carbonate_platform",
    "metamorphic_schist",
    "granitoid_intrusion",
    "ophiolite_ultramafic",
    "layered_mafic_intrusion",
    "arc_volcanic",
    "basaltic_volcanic",
    "unconsolidated_sediment",
    "red_beds_evaporitic",
    "coal_measures",
    "black_shale",
)
LITH = {k: i for i, k in enumerate(LITHOLOGY)}
LITHOLOGY_TITLES = {
    "unknown": "нет данных",
    "oceanic_basalt": "океанические базальты",
    "granite_gneiss": "граниты и гнейсы щитов",
    "greenstone_belt": "зеленокаменные пояса (древние)",
    "platform_siliciclastic": "терригенный платформенный чехол",
    "carbonate_platform": "карбонатные толщи",
    "metamorphic_schist": "метаморфические сланцы",
    "granitoid_intrusion": "гранитоидные интрузии",
    "ophiolite_ultramafic": "офиолиты / ультрамафиты",
    "layered_mafic_intrusion": "расслоенные мафит-ультрамафитовые массивы",
    "arc_volcanic": "вулканиты островных дуг",
    "basaltic_volcanic": "базальтовые вулканиты",
    "unconsolidated_sediment": "рыхлые осадки",
    "red_beds_evaporitic": "красноцветы и эвапоритовые толщи рифтов",
    "coal_measures": "угленосные толщи",
    "black_shale": "чёрные сланцы бассейнов",
}

#: Poisson intensity of tract systems, systems per km2 of suitability-weighted area [A]
#: (``synthetic_not_calibrated``); object-anchored types use a probability per object.
TRACT_LAMBDA = {
    "R04": 6e-6,
    "R05": 1.5e-6,
    "R06": 4e-6,
    "R07": 3e-6,
    "R09": 2e-5,
    "R10": 8e-6,
    "R12": 6e-6,
    "R13": 4e-6,
    "R14": 1.5e-6,
    "R15": 1e-6,
    "R16": 8e-7,
    "R22": 1.2e-6,
    "R23": 2e-6,
    "R24": 6e-6,
    "R25": 2e-6,
    "R26": 1.5e-6,
    "R28": 2e-6,
    "R30": 2e-7,
    "R31": 2e-6,
    "R34": 3e-6,
    "R08": 2e-6,
}
ANCHOR_PROB = {
    "R01": 0.12,
    "R02": 0.10,
    "R03": 0.20,
    "R05": 0.03,
    "R29": 0.05,
    "R11": 0.06,
    "R08": 0.05,
    "R32": 0.15,
}
#: long-term exhumation outside the simulated history, m/Ma [A]
INHERITED_EXHUMATION_M_MA = {"craton": 6.0, "orogen": 30.0, "other": 12.0}
MAX_INHERITED_EXHUMATION_M = 12000.0


@dataclass
class GeoContext:
    """Flat per-hex inputs (length N = rows x cols) and the discrete world objects."""

    seed: str
    shape: tuple
    hex_area_km2: float
    spacing_km: float
    duration_ma: float
    x_km: np.ndarray
    y_km: np.ndarray
    land: np.ndarray
    ocean: np.ndarray
    surface_m: np.ndarray  # solid surface (sea floor under the sea), datum m
    water_depth_m: np.ndarray
    rock: np.ndarray
    lith: np.ndarray
    crust_age_ma: np.ndarray
    cont_frac: np.ndarray
    orogen: np.ndarray  # bool: v3 mountain system / orogenic rock
    orogen_age_ma: np.ndarray
    collision: np.ndarray  # bool: collisional (not arc) setting
    arc_side: np.ndarray  # bool: overriding plate near a trench
    erosion_total_m: np.ndarray
    sediment_m: np.ndarray
    basin_type: np.ndarray  # 0 none, 1 sag, 2 foredeep, 3 rift, 4 pull-apart
    basin_fill_m: np.ndarray
    slope_m_km: np.ndarray
    shelf: np.ndarray
    upwelling: np.ndarray  # bool coastal upwelling
    seamount: np.ndarray  # bool
    marine_sed_m: np.ndarray
    ocean_crust_age_ma: np.ndarray
    volcano_systems: list = field(default_factory=list)
    unknown_fields: list = field(default_factory=list)

    @property
    def n(self) -> int:
        return self.x_km.size


# -- lithology adapter -----------------------------------------------------------------------------
def infer_lithology(
    seed, rock, crust_age_ma, mx_km, my_km, orogen, collision, arc_side, basin_type, basin_fill_m
):
    """Inferred sub-lithology (``LITHOLOGY`` codes) from the six v3 rock classes, the
    setting and smooth noise fixed in the material frame (moves with the plates, does not
    depend on the hex size). Provenance ``inferred_lithology`` [A]."""
    st = Stream(seed, "resources/lithology")
    n1 = FractalNoise(st.child("a"), [900.0, 300.0], [1.0, 0.45])(mx_km, my_km)
    n2 = FractalNoise(st.child("b"), [260.0, 90.0], [1.0, 0.5])(mx_km, my_km)
    u1 = np.clip(0.5 + 0.5 * n1 / 0.7, 0.0, 1.0)
    u2 = np.clip(0.5 + 0.5 * n2 / 0.7, 0.0, 1.0)
    lith = np.zeros(rock.shape, dtype=np.uint8)
    age = np.nan_to_num(crust_age_ma)
    r0, r1, r2, r3, r4, r5 = (rock == k for k in range(6))
    lith[r0] = LITH["oceanic_basalt"]
    lith[r1] = LITH["granite_gneiss"]
    lith[r1 & (age > 2400) & (u1 > 0.70)] = LITH["greenstone_belt"]
    lith[r1 & (u2 > 0.93)] = LITH["layered_mafic_intrusion"]
    lith[r2] = LITH["platform_siliciclastic"]
    lith[r2 & (u1 > 0.56)] = LITH["carbonate_platform"]
    lith[r3] = LITH["metamorphic_schist"]
    lith[r3 & (u1 > 0.68)] = LITH["carbonate_platform"]
    lith[r3 & (u2 > 0.55) & (u2 < 0.72)] = LITH["granitoid_intrusion"]
    lith[r3 & collision & (u2 > 0.86)] = LITH["ophiolite_ultramafic"]
    lith[r4] = LITH["basaltic_volcanic"]
    lith[r4 & arc_side] = LITH["arc_volcanic"]
    lith[r5] = LITH["unconsolidated_sediment"]
    # basin fills: rifts -> red beds / evaporites, foredeeps and sags -> coal measures or
    # black shales (the paleo-environment draw is the noise, flagged as inherited)
    in_b = (basin_type > 0) & (np.nan_to_num(basin_fill_m) > 300.0) & (rock >= 2) & ~r4
    lith[in_b & (basin_type == 3) & (u1 > 0.35)] = LITH["red_beds_evaporitic"]
    lith[in_b & (basin_type == 3) & (u2 > 0.7)] = LITH["black_shale"]
    sag_fore = in_b & ((basin_type == 1) | (basin_type == 2) | (basin_type == 4))
    lith[sag_fore & (u1 > 0.62)] = LITH["coal_measures"]
    lith[sag_fore & (u1 <= 0.62) & (u2 > 0.6)] = LITH["black_shale"]
    return lith


def context_from_project(project) -> GeoContext:
    """Build the flat inputs from a generated project (v3 .. v5.1)."""
    lay = project.layout
    shape = (lay.height, lay.width)
    hx, hy = lay.centers()
    H = project.hist.layers
    hy_ = project.hydro.layers

    def f(a, dtype=np.float64):
        return np.asarray(a, dtype=dtype).ravel()

    rock = f(H["rock_type"], np.uint8)
    crust_age = f(H["crust_age_ma"])
    orogen = (f(H["mountain_system_id"], np.int64) >= 0) | (rock == 3)
    td = f(H["trench_distance_km"])
    side = f(H["trench_side"], np.int64)
    arc_side = (side == -1) & (td < 700.0)
    collision = orogen & ~((td < 600.0) & (side != 0))
    unknown = []
    if project.terrain is not None:
        T = project.terrain.layers
        plans = project.terrain.plans.get("basins", [])
        bid = f(T["basin_id"], np.int64)
        codes = {
            "intracratonic_sag": 1,
            "foreland_foredeep": 2,
            "rift_graben": 3,
            "strike_slip_pull_apart": 4,
        }
        lut = np.array([codes.get(p.get("basin_type"), 0) for p in plans] + [0], dtype=np.uint8)
        basin_type = np.where(bid >= 0, lut[np.clip(bid, 0, len(plans))], 0).astype(np.uint8)
        basin_fill = f(T.get("basin_fill_m", np.zeros(shape)))
        seamount = f(T.get("seamount_type", np.zeros(shape)), np.int64) > 0
        marine_sed = f(T.get("marine_sediment_thickness_m", np.zeros(shape)))
        oc_age = f(T.get("oceanic_crust_age_ma", np.full(shape, np.nan)))
    else:
        basin_type = np.zeros(rock.size, np.uint8)
        basin_fill = np.zeros(rock.size)
        seamount = np.zeros(rock.size, bool)
        marine_sed = np.zeros(rock.size)
        oc_age = crust_age.copy()
        unknown += ["basins (v4.6)", "sea-floor provinces (v4.6)"]
    mx = f(H["material_x_km"])
    my = f(H["material_y_km"])
    lith = infer_lithology(
        project.config.seed,
        rock,
        crust_age,
        mx,
        my,
        orogen,
        collision,
        arc_side,
        basin_type,
        basin_fill,
    )
    ocean = f(hy_["ocean_mask"], bool)
    surface = f(project.layer("elevation_m"))
    sea = float(project.hydro.sea_level_m)
    up = np.zeros(rock.size, bool)
    if project.earth is not None and "coastal_upwelling" in project.earth.layers:
        up = f(project.earth.layers["coastal_upwelling"], bool)
    elif project.earth is None:
        unknown.append("coastal upwelling (v5.3)")
    systems = list(project.volcanism.systems) if project.volcanism is not None else []
    if project.volcanism is None:
        unknown.append("volcano systems (v5.1)")
    return GeoContext(
        seed=project.config.seed,
        shape=shape,
        hex_area_km2=float(lay.hex_area_km2),
        spacing_km=float(lay.spacing_km),
        duration_ma=float(project.hist.duration_ma),
        x_km=f(hx),
        y_km=f(hy),
        land=~ocean,
        ocean=ocean,
        surface_m=surface,
        water_depth_m=np.where(ocean, np.maximum(sea - surface, 0.0), 0.0),
        rock=rock,
        lith=lith,
        crust_age_ma=crust_age,
        cont_frac=f(H["continental_fraction"]),
        orogen=orogen,
        orogen_age_ma=f(H["orogen_age_ma"]),
        collision=collision,
        arc_side=arc_side,
        erosion_total_m=np.maximum(np.nan_to_num(f(H["erosion_total_m"])), 0.0),
        sediment_m=np.maximum(np.nan_to_num(f(project.layer("sediment_thickness_m"))), 0.0),
        basin_type=basin_type,
        basin_fill_m=np.nan_to_num(basin_fill),
        slope_m_km=np.nan_to_num(f(project.layer("slope_m_per_km"))),
        shelf=f(hy_.get("shelf_mask", np.zeros(shape)), bool),
        upwelling=up,
        seamount=seamount,
        marine_sed_m=np.nan_to_num(marine_sed),
        ocean_crust_age_ma=oc_age,
        volcano_systems=systems,
        unknown_fields=unknown,
    )


# -- permissive tracts (hard requirements x soft suitability) --------------------------------------
def _sig(x):
    return 1.0 / (1.0 + np.exp(-x))


def tract_suitability(ctx: GeoContext) -> dict:
    """``{model_id: (permissive_mask, S, provenance)}`` of the tract types.

    The masks are the hard requirements of TZ 13 / 14.1; S in 0..1 is a soft suitability,
    not a probability of a deposit."""
    L = ctx.lith
    age = np.nan_to_num(ctx.crust_age_ma)
    land_or_shelf = ctx.land | ctx.shelf
    fill = ctx.basin_fill_m
    out = {}

    def put(mid, mask, s, prov):
        out[mid] = (mask, np.where(mask, np.clip(s, 0.0, 1.0), 0.0), prov)

    greenstone = L == LITH["greenstone_belt"]
    schist = L == LITH["metamorphic_schist"]
    granitoid = L == LITH["granitoid_intrusion"]
    put(
        "R04",
        (greenstone | (schist & ctx.orogen)) & land_or_shelf,
        np.where(greenstone, 0.9, 0.5),
        "orogen / greenstone (inherited for > 2.4 Ga)",
    )
    put(
        "R05",
        (greenstone | (schist & ctx.collision)) & land_or_shelf,
        np.where(greenstone, 0.8, 0.3),
        "former sea-floor volcanics in belts (inherited)",
    )
    put("R10", granitoid & ctx.collision & (ctx.cont_frac > 0.5), 0.7, "collisional granites")
    put("R12", granitoid & (ctx.cont_frac > 0.5), 0.6, "evolved granites")
    put(
        "R34",
        (granitoid | schist) & ctx.orogen & (ctx.cont_frac > 0.5),
        0.4,
        "pegmatite / metamorphic gem settings",
    )
    put(
        "R09",
        (L == LITH["ophiolite_ultramafic"]) | (L == LITH["layered_mafic_intrusion"]),
        0.8,
        "ophiolites and layered intrusions",
    )
    put("R08", L == LITH["layered_mafic_intrusion"], 0.6, "layered mafic-ultramafic intrusions")
    put(
        "R15",
        (greenstone | (L == LITH["granite_gneiss"])) & (age > 1800.0),
        np.where(greenstone, 0.9, 0.3),
        "inherited_stratigraphy (> 1.8 Ga crust)",
    )
    put(
        "R22",
        (ctx.rock <= 2) & (ctx.rock >= 1) & (age > 1500.0) & (ctx.cont_frac > 0.6) & ctx.land,
        _sig((age - 2200.0) / 300.0),
        "old cratonic root (inherited)",
    )
    put(
        "R14",
        (L == LITH["platform_siliciclastic"])
        & (age > 1500.0)
        & (ctx.sediment_m > 100.0)
        & (ctx.sediment_m < 3000.0),
        0.6,
        "old basement under sandstone (unconformity)",
    )
    seds = (L == LITH["platform_siliciclastic"]) | (L == LITH["red_beds_evaporitic"])
    put(
        "R13",
        seds & (ctx.sediment_m > 300.0) & (ctx.sediment_m < 4000.0) & ctx.land,
        0.5,
        "permeable sandstones with redox fronts",
    )
    put(
        "R06",
        ((L == LITH["black_shale"]) | (L == LITH["red_beds_evaporitic"])) & (ctx.basin_type == 3),
        0.7,
        "rift basin with brines",
    )
    put(
        "R07",
        (L == LITH["red_beds_evaporitic"]) & (ctx.basin_type > 0),
        0.7,
        "red beds over reduced horizons",
    )
    put(
        "R16",
        ((L == LITH["carbonate_platform"]) | (L == LITH["platform_siliciclastic"])) & (age > 300.0),
        0.25,
        "old shelf sequences (inherited)",
    )
    put("R23", schist & ctx.orogen, 0.35, "metamorphosed carbonaceous sediments")
    put(
        "R24",
        (L == LITH["coal_measures"]) & (fill > 800.0),
        _sig((fill - 1500.0) / 700.0),
        "peat-forming basin + burial (inherited paleo-swamp)",
    )
    put(
        "R25",
        (ctx.basin_type > 0)
        & (fill > 2500.0)
        & (
            (L == LITH["black_shale"])
            | (L == LITH["coal_measures"])
            | (L == LITH["red_beds_evaporitic"])
        ),
        _sig((fill - 3500.0) / 800.0),
        "P1: source, burial, reservoir, seal assumed",
    )
    put(
        "R26",
        (L == LITH["red_beds_evaporitic"]) & (fill > 1000.0),
        0.6,
        "restricted rift basin (inherited evaporitic episodes)",
    )
    put(
        "R28",
        (ctx.shelf & ctx.upwelling) | ((L == LITH["carbonate_platform"]) & (age > 200.0)),
        np.where(ctx.upwelling, 0.8, 0.15),
        "upwelling shelf (modern) / old phosphatic shelf",
    )
    deep = ctx.ocean & (ctx.water_depth_m > 4000.0) & ~ctx.seamount
    put(
        "R30",
        deep & (np.nan_to_num(ctx.ocean_crust_age_ma) > 10.0) & (ctx.marine_sed_m < 600.0),
        _sig((ctx.water_depth_m - 4500.0) / 400.0) * _sig((400.0 - ctx.marine_sed_m) / 100.0),
        "abyssal plain with slow sedimentation",
    )
    put(
        "R31",
        ctx.ocean & ctx.seamount & (ctx.water_depth_m > 800.0) & (ctx.water_depth_m < 3000.0),
        0.8,
        "hard seamount flanks",
    )
    return out


def _provinces(mask2d):
    lab, k = ndimage.label(mask2d, structure=np.ones((3, 3), bool))
    return lab.ravel(), k


# -- system construction ---------------------------------------------------------------------------
def _stream(seed, *key):
    return Stream(seed, "resources/ore/" + "/".join(str(k) for k in key) + "/" + ALGORITHM_VERSION)


def _formation(model, ctx, i, u, kind):
    """(formation_epoch_ma, provenance) of a system at hex ``i``."""
    dur = ctx.duration_ma
    age = float(np.nan_to_num(ctx.crust_age_ma[i]))
    if kind == "inherited_old":
        return max(dur + 1.0, age * (0.6 + 0.35 * u)), "inherited_stratigraphy"
    if kind == "orogen":
        oa = ctx.orogen_age_ma[i]
        if np.isfinite(oa) and oa <= dur:
            return float(max(oa, 0.2)), "simulated_history"
        return float(dur + (age - dur) * 0.3 * u if age > dur else dur * u), "inherited_province"
    if kind == "basin":
        return float(dur * (0.2 + 0.8 * u)), "simulated_history_basin"
    return float(dur * u), "simulated_history"


def _exhumation(ctx, i, epoch_ma):
    """Erosion of the site since the formation epoch (m)."""
    dur = max(ctx.duration_ma, 1e-6)
    e = float(ctx.erosion_total_m[i]) * min(1.0, epoch_ma / dur)
    if epoch_ma > dur:
        key = "orogen" if ctx.orogen[i] else ("craton" if ctx.rock[i] == 1 else "other")
        e += min(INHERITED_EXHUMATION_M_MA[key] * (epoch_ma - dur), MAX_INHERITED_EXHUMATION_M)
    return e


SEDIMENT_HOSTED = ("R06", "R07", "R13", "R24", "R25", "R26", "R28", "R16")
SURFACE_TYPES = ("R30", "R31", "R29")


def build_system(
    ctx: GeoContext,
    model_id: str,
    x_km: float,
    y_km: float,
    i: int,
    key: tuple,
    kind: str,
    extra: dict | None = None,
):
    """One deposit system with its bodies, or a record of a fully eroded one.

    Returns ``(system | None, bodies, eroded_record | None)``.
    """
    model = MODELS[model_id]
    st = _stream(ctx.seed, model_id, *key)
    u = st.random(8)
    z = st.normal(len(model.grades) + 1)
    epoch, prov = _formation(model, ctx, i, u[0], kind)
    sys_id = "D-" + stable_hash(ctx.seed, "deposit", model_id, *key)[:12]
    surface = float(ctx.surface_m[i])
    # grade and tonnage of the whole system (ore_t, grades); special masses below
    if model.tonnage_median_t > 0:
        ore_t, grades = sample_grade_tonnage(model, z)
    else:
        ore_t, grades = 0.0, {}
    if model_id in ("R30", "R31"):
        # areal deposits: dry areal density (kg/m2) x covered area of the tract cell
        dens = (
            15.0 if model_id == "R30" else 40.0
        )  # kg/m2 nodules / crust (0.05 m x 2 g/cm3 x share)
        area_m2 = ctx.hex_area_km2 * 1e6 * (0.3 + 0.4 * u[1])
        ore_t = dens * area_m2 / 1000.0 * (0.5 + u[2])
        grades = {}
        for j, (el, (med, sd)) in enumerate(model.grades.items()):
            grades[el] = float(med * math.exp(sd * float(np.clip(z[j + 1], -3, 3))))
    nb_lo, nb_hi = model.bodies
    nb = nb_lo + int(u[3] * (nb_hi - nb_lo + 1)) if nb_hi > nb_lo else nb_lo
    nb = max(nb, 1)
    w = st.random(nb) + 0.2
    w = w / w.sum()
    d_lo, d_hi = model.formation_depth_m
    d0 = d_lo + (d_hi - d_lo) * u[4]
    if model_id in SEDIMENT_HOSTED:
        fill = max(float(ctx.sediment_m[i]), 50.0)
        d0 = min(max(d0, 0.0), 0.9 * fill) if model_id != "R28" else min(d0, 0.5 * fill)
        exhum = 0.0
        burial = 0.0
    elif model_id in SURFACE_TYPES:
        exhum, burial = 0.0, 0.0
        d0 = 0.0
    elif model_id == "R05" and kind == "object":
        exhum, burial = 0.0, 100.0 + 900.0 * u[5]  # buried by the later volcanic pile
    else:
        if epoch > ctx.duration_ma:
            # older than the simulated history: the unknown earlier exhumation took a random
            # share of the formation depth (inherited); then the simulated erosion applies
            if model_id == "R05":
                d0 = 500.0 + 5500.0 * u[4]  # sea-floor lens under the later volcanic pile
            d0 = d0 * u[5]
            exhum = float(ctx.erosion_total_m[i])
        else:
            exhum = _exhumation(ctx, i, epoch)
            if model_id == "R05":
                d0 = 300.0 + 2000.0 * u[4]
        # basement deposits under a younger sedimentary cover
        burial = float(ctx.sediment_m[i]) if ctx.rock[i] in (2, 5) else 0.0
    strike = 180.0 * u[6]
    bodies, eroded_mass_t, total_mass_t = [], 0.0, 0.0
    eroded_commodity = {el: 0.0 for el in grades}
    cluster_m = {"R04": 6000.0, "R10": 3000.0, "R02": 1500.0, "R09": 4000.0}.get(model_id, 1500.0)
    for j in range(nb):
        bj = Stream(ctx.seed, f"{st.name}/body{j}")
        ub = bj.random(4)
        ore_j = ore_t * w[j]
        n_veins = 1 + int(6 * ub[3]) if model.shape in ("vein", "fault_hosted_vein_network") else 1
        vol = ore_j * 1000.0 / (model.density_kg_m3 * (1.0 - model.f_barren)) if ore_j > 0 else 0.0
        if model_id in ("R30", "R31"):
            a = b = math.sqrt(ctx.hex_area_km2 * 1e6 * w[j] / math.pi)
            c = 0.03
            vol = shape_volume("sheet", a, b, c)
        elif vol > 0:
            a, b, c = axes_for_volume(model.shape, vol, model.aspect, n_veins)
        else:
            continue
        r = cluster_m * math.sqrt(ub[0]) if nb > 1 else 0.0
        ang = 2 * math.pi * ub[1]
        cx = x_km * 1000.0 + r * math.cos(ang)
        cy = y_km * 1000.0 + r * math.sin(ang)
        # top depth below the present (bedrock) surface after exhumation, then the cover
        top_depth_bedrock = d0 - exhum
        c_vert = c
        eroded_frac = 0.0
        eroded_above = None
        if top_depth_bedrock < 0.0:
            h = -top_depth_bedrock
            if h >= 2.0 * c_vert:
                eroded_mass_t += ore_j
                total_mass_t += ore_j
                for el, g in grades.items():
                    eroded_commodity[el] += ore_j * g
                continue
            if model.shape in ELLIPSOID_SHAPES:
                cap = ellipsoid_cap_volume(a, b, c_vert, h)
                eroded_frac = cap / max(shape_volume(model.shape, a, b, c, n_veins), 1e-9)
            else:
                eroded_frac = h / (2.0 * c_vert)
            top_depth_bedrock = 0.0
        bedrock_surface = surface - burial
        z_top_intact = bedrock_surface - (d0 - exhum)
        zc = z_top_intact - c_vert
        if eroded_frac > 0:
            eroded_above = bedrock_surface
        ore_kept = ore_j * (1.0 - eroded_frac)
        eroded_mass_t += ore_j * eroded_frac
        total_mass_t += ore_j
        for el, g in grades.items():
            eroded_commodity[el] += ore_j * eroded_frac * g
        top_now = min(z_top_intact, bedrock_surface)
        burial_depth = surface - top_now
        outcrop = 0.0
        if eroded_frac > 0 and burial <= 0.0:
            if model.shape in ELLIPSOID_SHAPES:
                hh = min(-(d0 - exhum), 2 * c_vert)
                rr = max(1.0 - ((c_vert - hh) / c_vert) ** 2, 0.0)
                outcrop = math.pi * a * b * rr
            else:
                outcrop = (
                    math.pi * a * b
                    if model.shape not in ("vein", "fault_hosted_vein_network")
                    else n_veins * 4 * a * b
                )
        if model_id in ("R30", "R31", "R29") or (model_id == "R28" and kind == "modern"):
            outcrop = math.pi * a * b if model_id != "R29" else math.pi * a * b
        bodies.append(
            {
                "body_id": f"{sys_id}-B{j}",
                "system_id": sys_id,
                "shape": model.shape,
                "center_m": [round(cx, 1), round(cy, 1), round(zc, 2)],
                "axes_m": [a, b, c],
                "strike_deg": round(strike + 20.0 * (ub[2] - 0.5), 2),
                "dip_deg": 90.0 if model.shape in ("vein", "fault_hosted_vein_network") else 0.0,
                "n_veins": n_veins,
                "z_top_m": round(top_now, 2),
                "z_bottom_m": round(zc - c_vert, 2),
                "volume_m3": shape_volume(model.shape, a, b, c, n_veins) * (1.0 - eroded_frac),
                "density_kg_m3": model.density_kg_m3,
                "f_barren": model.f_barren,
                "ore_t": ore_kept,
                "grades": dict(grades),
                "contained_t": {el: ore_kept * g for el, g in grades.items()},
                "formed_ma": round(epoch, 4),
                "erosion_fraction": eroded_frac,
                "eroded_above_m": eroded_above,
                "burial_depth_m": round(burial_depth, 2),
                "outcrop_area_m2": outcrop,
                "geometry_version": 1,
            }
        )
    record = None
    if eroded_mass_t > 0:
        record = {
            "system_id": sys_id,
            "model_id": model_id,
            "x_km": x_km,
            "y_km": y_km,
            "hex": int(i),
            "eroded_ore_t": eroded_mass_t,
            "eroded_commodity_t": eroded_commodity,
            "formed_ma": epoch,
        }
    if not bodies:
        return None, [], record
    ore_kept = sum(b_["ore_t"] for b_ in bodies)
    contained = {el: sum(b_["contained_t"][el] for b_ in bodies) for el in grades}
    system = {
        "deposit_id": sys_id,
        "model_id": model_id,
        "genetic_class": model.name,
        "title": model.title,
        "family": model.family,
        "priority": model.priority,
        "anchor_kind": kind,
        "anchor_key": [str(k) for k in key],
        "x_km": round(x_km, 4),
        "y_km": round(y_km, 4),
        "hex": int(i),
        "formation_epoch_ma": round(epoch, 4),
        "provenance": prov,
        "host_lithology": LITHOLOGY[int(ctx.lith[i])],
        "ore_t": ore_kept,
        "ore_t_formed": total_mass_t,
        "grades": grades,
        "contained_t": contained,
        "eroded_fraction": eroded_mass_t / max(total_mass_t, 1e-12),
        "preservation_state": (
            "partly_eroded"
            if eroded_mass_t > 0
            else ("buried" if bodies[0]["burial_depth_m"] > 0 else "intact")
        ),
        "depth_top_m": min(b_["burial_depth_m"] for b_ in bodies),
        "outcrop_area_m2": sum(b_["outcrop_area_m2"] for b_ in bodies),
        "surface_occurrence": any(b_["outcrop_area_m2"] > 0 for b_ in bodies),
        "discovery_state": "undiscovered",
        "quality": "synthetic_not_calibrated",
        "body_ids": [b_["body_id"] for b_ in bodies],
        **(extra or {}),
    }
    return system, bodies, record


# -- the plan --------------------------------------------------------------------------------------
def _anchor_types(vs: dict, ctx: GeoContext, lith_near_carbonate) -> list:
    """Deposit types an individual volcano system may host (hard conditions)."""
    src = vs.get("source_type", "")
    env = vs.get("environment", "land")
    magma = vs.get("magma") or {}
    sio2 = float(magma.get("silica_wt_pct", 50.0))
    sub = "submarine" in env
    out = []
    if src == "subduction_arc" and not sub and 55.0 <= sio2 <= 72.0:
        out += ["R01", "R02"]
        if lith_near_carbonate:
            out.append("R03")
        if vs.get("magmatic_status") in ("active_system", "dormant_system"):
            out.append("R32")
    if src == "divergent_rift" and not sub and sio2 >= 60.0:
        out.append("R02")
    if sub and src in ("subduction_arc", "mid_ocean_ridge"):
        out += (
            ["R05", "R29"]
            if vs.get("magmatic_status") in ("active_system", "dormant_system")
            else ["R05"]
        )
    if src in ("continental_hotspot", "divergent_rift") and not sub:
        out.append("R08")
    if src in ("continental_hotspot", "divergent_rift") and not sub:
        out.append("R11")
    return out


def plan_deposit_systems(ctx: GeoContext, prospect: dict | None = None, lambda_scale: float = 1.0):
    """The deposit catalogue of the primary (hypogene / sedimentary / marine) systems.

    Returns ``{"systems", "bodies", "provinces", "eroded", "unknown", "requirements"}``.
    """
    prospect = prospect if prospect is not None else tract_suitability(ctx)
    systems, bodies, provinces, eroded = [], [], [], []
    shape = ctx.shape
    area = ctx.hex_area_km2
    # 1) object-anchored systems on the volcano systems (stable world objects)
    tree = cKDTree(np.column_stack([ctx.x_km, ctx.y_km]))
    carb = ctx.lith == LITH["carbonate_platform"]
    for vs in sorted(ctx.volcano_systems, key=lambda v: str(v.get("volcano_id"))):
        x, y = float(vs["x_km"]), float(vs["y_km"])
        d, i = tree.query([x, y])
        if d > 1.5 * ctx.spacing_km:
            continue  # outside the window
        near = tree.query_ball_point([x, y], max(30.0, ctx.spacing_km * 0.75))
        near_carb = bool(np.any(carb[near]))
        for mid in _anchor_types(vs, ctx, near_carb):
            if mid == "R11" and float(np.nan_to_num(ctx.crust_age_ma[i])) < 500.0:
                continue  # carbonatites need old, thick lithosphere
            st = _stream(ctx.seed, mid, "volcano", vs["volcano_id"], "draw")
            if st.random() >= ANCHOR_PROB[mid] * lambda_scale:
                continue
            off = st.random(2)
            ox = x + (off[0] - 0.5) * 16.0
            oy = y + (off[1] - 0.5) * 16.0
            fm = float(vs.get("formed_ma", 0.0) or 0.0)
            s_, b_, rec = build_system(
                ctx,
                mid,
                ox,
                oy,
                int(i),
                ("volcano", vs["volcano_id"]),
                "object",
                {
                    "parent_event_ids": [vs["volcano_id"]],
                    "province_id": vs.get("province_id"),
                    "source_pathway": vs.get("source_type"),
                    "volcano_formed_ma": fm,
                },
            )
            if s_ is not None:
                s_["formation_epoch_ma"] = round(
                    min(max(fm, 0.05), s_["formation_epoch_ma"])
                    if fm
                    else s_["formation_epoch_ma"],
                    4,
                )
                systems.append(s_)
                bodies.extend(b_)
            if rec:
                eroded.append(rec)
    # 2) tract systems: provinces -> Poisson count -> anchors
    for mid in sorted(prospect):
        mask, s, prov = prospect[mid]
        if mid not in TRACT_LAMBDA or not mask.any():
            continue
        lab, k = _provinces(mask.reshape(shape))
        if k == 0:
            continue
        sums = np.bincount(lab, weights=s * area, minlength=k + 1)
        cx = np.bincount(lab, weights=ctx.x_km * s, minlength=k + 1)
        cy = np.bincount(lab, weights=ctx.y_km * s, minlength=k + 1)
        wsum = np.bincount(lab, weights=s, minlength=k + 1)
        order = np.argsort(lab, kind="stable")
        starts = np.searchsorted(lab[order], np.arange(k + 2))
        for p in range(1, k + 1):
            if sums[p] <= 0:
                continue
            pcx, pcy = cx[p] / wsum[p], cy[p] / wsum[p]
            ph = stable_hash(ctx.seed, "province", mid, round(pcx / 100.0), round(pcy / 100.0))
            pid = f"P-{mid}-{ph[:10]}"
            lam = TRACT_LAMBDA[mid] * lambda_scale * sums[p]
            st = _stream(ctx.seed, mid, pid, "count")
            count = poisson_from_uniform(lam, st.random())
            provinces.append(
                {
                    "province_id": pid,
                    "model_id": mid,
                    "x_km": round(pcx, 2),
                    "y_km": round(pcy, 2),
                    "area_km2": float((starts[p + 1] - starts[p]) * area),
                    "suitability_integral_km2": float(sums[p]),
                    "lambda": lam,
                    "count": count,
                    "provenance": prov,
                }
            )
            if count == 0:
                continue
            idx = order[starts[p] : starts[p + 1]]
            cw = np.cumsum(s[idx])
            for kk in range(count):
                sa = _stream(ctx.seed, mid, pid, "anchor", kk)
                ua = sa.random(3)
                j = int(np.searchsorted(cw, ua[0] * cw[-1], side="right"))
                i = int(idx[min(j, idx.size - 1)])
                r = 0.45 * ctx.spacing_km * math.sqrt(ua[1])
                th = 2 * math.pi * ua[2]
                x = float(ctx.x_km[i] + r * math.cos(th))
                y = float(ctx.y_km[i] + r * math.sin(th))
                kind = (
                    "inherited_old"
                    if mid in ("R15", "R22", "R14", "R16")
                    or (mid in ("R04", "R05") and ctx.lith[i] == LITH["greenstone_belt"])
                    else "basin"
                    if mid in SEDIMENT_HOSTED
                    else "modern"
                    if mid in ("R30", "R31")
                    else "orogen"
                )
                s_, b_, rec = build_system(
                    ctx,
                    mid,
                    x,
                    y,
                    i,
                    (pid, kk),
                    kind,
                    {"province_id": pid, "parent_event_ids": [pid]},
                )
                if s_ is not None:
                    systems.append(s_)
                    bodies.extend(b_)
                if rec:
                    eroded.append(rec)
    systems.sort(key=lambda s_: s_["deposit_id"])
    bodies.sort(key=lambda b_: b_["body_id"])
    req = {
        mid: {"permissive_hexes": int(m.sum()), "provenance": p}
        for mid, (m, _s, p) in prospect.items()
    }
    return {
        "systems": systems,
        "bodies": bodies,
        "provinces": provinces,
        "eroded": eroded,
        "unknown": list(ctx.unknown_fields),
        "requirements": req,
    }


# -- aggregation to the hexes ----------------------------------------------------------------------
def hex_index(bodies, x_km, y_km, spacing_km):
    """CSR ``offsets, body_index`` of the bodies whose footprint covers each hex centre (the
    anchor hex always), for a flat hex array."""
    n = x_km.size
    tree = cKDTree(np.column_stack([x_km, y_km]))
    pairs = []
    for bi, b in enumerate(bodies):
        cx, cy = b["center_m"][0] / 1000.0, b["center_m"][1] / 1000.0
        r = footprint_radius_m(b) / 1000.0
        hits = set(tree.query_ball_point([cx, cy], r)) if r > 0.3 * spacing_km else set()
        d, i0 = tree.query([cx, cy])
        if d <= spacing_km:
            hits.add(int(i0))
        for h in hits:
            pairs.append((int(h), bi))
    pairs.sort()
    offs = np.zeros(n + 1, dtype=np.int64)
    idx = np.zeros(len(pairs), dtype=np.int32)
    for k, (h, bi) in enumerate(pairs):
        offs[h + 1] += 1
        idx[k] = bi
    offs = np.cumsum(offs)
    return offs, idx


def aggregate(catalog, ctx: GeoContext, prospect: dict):
    """Dense hex layers of the catalogue (aggregates and indices only)."""
    n = ctx.n
    bodies = catalog["bodies"]
    sys_by_id = {s["deposit_id"]: s for s in catalog["systems"]}
    offs, idx = hex_index(bodies, ctx.x_km, ctx.y_km, ctx.spacing_km)
    fam_ix = {f: k for k, f in enumerate(FAMILIES)}
    in_situ = np.zeros((len(FAMILIES), n), dtype=np.float64)
    count = np.zeros(n, dtype=np.int32)
    shallow = np.full(n, np.nan)
    outcrop = np.zeros(n, dtype=bool)
    counts = np.diff(offs)
    for h in np.nonzero(counts)[0]:
        seen = set()
        for bi in idx[offs[h] : offs[h + 1]]:
            b = bodies[bi]
            s = sys_by_id[b["system_id"]]
            share = 1.0 / max(1, _n_hexes(offs, idx, bi, cache=_n_cache))
            in_situ[fam_ix[s["family"]], h] += (
                sum(b["contained_t"].values()) * share
                if s["family"] not in ("energy",)
                else b["ore_t"] * share
            )
            seen.add(b["system_id"])
            d = b["burial_depth_m"]
            shallow[h] = d if not np.isfinite(shallow[h]) else min(shallow[h], d)
            outcrop[h] |= b["outcrop_area_m2"] > 0
        count[h] = len(seen)
    _n_cache.clear()
    prosp = np.zeros((len(FAMILIES), n), dtype=np.float32)
    for mid, (_m, s, _p) in prospect.items():
        k = fam_ix[MODELS[mid].family]
        prosp[k] = np.maximum(prosp[k], s)
    # anchored types: proximity of the volcano systems that may host them
    if ctx.volcano_systems:
        tree = cKDTree(np.column_stack([ctx.x_km, ctx.y_km]))
        for vs in ctx.volcano_systems:
            src = vs.get("source_type", "")
            fam = {
                "subduction_arc": "base_metals",
                "mid_ocean_ridge": "marine",
                "continental_hotspot": "critical_rare",
                "divergent_rift": "precious_metals",
            }.get(src)
            if fam is None:
                continue
            near = tree.query_ball_point([vs["x_km"], vs["y_km"]], 60.0)
            if near:
                d = np.hypot(ctx.x_km[near] - vs["x_km"], ctx.y_km[near] - vs["y_km"])
                prosp[fam_ix[fam], near] = np.maximum(
                    prosp[fam_ix[fam], near], 0.6 * np.exp(-d / 40.0)
                )
    access = np.where(np.isfinite(shallow), np.exp(-np.nan_to_num(shallow) / 600.0), 0.0)
    access = access * np.where(ctx.ocean, np.exp(-ctx.water_depth_m / 1500.0), 1.0)
    return {
        "in_situ_t_by_family": in_situ.astype(np.float32),
        "prospectivity_by_family": prosp,
        "deposit_count": count,
        "shallowest_body_depth_m": shallow.astype(np.float32),
        "surface_occurrence": outcrop,
        "overburden_m": ctx.sediment_m.astype(np.float32),
        "accessibility_proxy": access.astype(np.float32),
        "hex_offsets": offs,
        "hex_bodies": idx,
    }


_n_cache: dict = {}


def _n_hexes(offs, idx, bi, cache):
    if not cache:
        c = np.bincount(idx, minlength=int(idx.max()) + 1 if idx.size else 1)
        cache["c"] = c
    return int(cache["c"][bi])
