"""v3.0 stages, run strictly after the v1/v2 baseline::

    v1 baseline plates / crust / kinematics  +  v2 baseline continents / macro elevation
      -> geo_history   (time steps, checkpoints, events, volcanoes, erosion)
      -> mountain_plan (structure of the selected epoch)
      -> v3_raster     (hex sampling: elevation and every v3 layer)
      -> land / sea    (one surface, one sea level)

Dependencies are explicit stage keys: mountain parameters do not re-run the
history; the sea level re-runs nothing but the land mask; a changed history
invalidates plan and raster but never the v1/v2 baseline, whose arrays are only
read. Stale caches are never reused: a key mismatch means recomputation.
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import asdict

import numpy as np
from scipy import ndimage
from scipy.spatial import cKDTree

from .config import Config
from .geometry import WorldGeometry
from .mountains import MountainRequest, build_plan, get_backend, limit_elevation
from .mountains.backend import (
    CLASS_CREST,
    CLASS_FOOTHILL,
    CLASS_NONE,
    CLASS_PASS,
    CLASS_SLOPE,
    CLASS_VALLEY,
    CLASS_VOLCANO,
)
from .noise import FractalNoise
from .pipeline_v2 import _component_table, _names
from .project import ROCK_TYPES, HistoryData
from .rng import Stream
from .stages import raster as raster_stage
from .stages.boundaries import TYPE_DIVERGENT
from .stages.continents import build_continents, smoothstep
from .stages.history import (
    Baseline,
    HistoryGrid,
    HistorySimulator,
    Kinematics,
    MaterialState,
    run_history,
)
from .stages.relief import build_relief
from .stages.sea import derive_land
from .stages.skeleton import build_skeleton
from .stages.volcanism import ARC, HOTSPOT, VolcanoBuilder, VolcanoSet, cone_field, place_hotspots
from .version import HISTORY_ALGORITHM_VERSION, MOUNTAINS_ALGORITHM_VERSION

V3_STAGES = ("geo_history", "mountain_plan", "v3_raster", "v3_sea")


def _key(*parts) -> str:
    return hashlib.sha256(json.dumps(parts, sort_keys=True, default=str).encode()).hexdigest()


def resolve_epoch(config: Config, epochs: list[float]) -> float:
    """Time (Ma) of the stored snapshot selected by ``history.epoch``."""
    want = config.history.epoch
    if want == "final":
        return epochs[-1]
    if want == "initial":
        return epochs[0]
    nearest = min(epochs, key=lambda t: abs(t - float(want)))
    return nearest


def v3_stage_keys(config: Config, v2_keys: dict, layout, epoch_time: float | None) -> dict:
    hist = asdict(config.history)
    hist.pop("epoch")
    history = _key("geo_history", HISTORY_ALGORITHM_VERSION, v2_keys["macro_elevation"], hist)
    plan = _key(
        "mountain_plan", MOUNTAINS_ALGORITHM_VERSION, history, epoch_time, asdict(config.mountains)
    )
    raster = _key(
        "v3_raster", plan, layout.width, layout.height, layout.spacing_km, layout.x0, layout.y0
    )
    sea = _key("v3_sea", raster, asdict(config.sea), v2_keys["sea_level_landmasses"])
    return {"geo_history": history, "mountain_plan": plan, "v3_raster": raster, "v3_sea": sea}


# -- baseline -------------------------------------------------------------------
def ensure_v2_models(
    config: Config, geometry: WorldGeometry, plate_model, crust_model, motion, cache: dict
) -> tuple:
    """World skeleton and relief model of the v2 baseline (rebuilt if only arrays were loaded)."""
    entry = cache.get("macro_elevation") or {}
    skeleton = cache.get("world_tectonics")
    model = entry.get("model") or cache.get("macro_elevation_model")
    if model is not None and skeleton is not None:
        return skeleton, model
    if plate_model is None or crust_model is None:
        raise RuntimeError(
            "The v3.0 stage needs the v1/v2 world models, which could not be reproduced for "
            "this project (saved by an incompatible generator version)."
        )
    skeleton = build_skeleton(
        geometry, plate_model, motion, config.boundaries, config.continents.skeleton_cells
    )
    continents, _world_c = build_continents(config, crust_model, skeleton)
    model = build_relief(config, continents, skeleton)
    cache["world_tectonics"] = skeleton
    cache["macro_elevation_model"] = model
    return skeleton, model


def build_baseline(config: Config, skeleton, relief_model) -> Baseline:
    grid = HistoryGrid(
        skeleton.x0, skeleton.y0, skeleton.cell_km, skeleton.x.shape[1], skeleton.x.shape[0]
    )
    x, y = skeleton.x, skeleton.y
    world_c = relief_model.continents.continentality(x, y)
    elevation, tectonic = relief_model.elevation(x, y, continentality=world_c, parts=True)
    fraction = relief_model.continents.fraction_of(world_c)
    inherited = np.where(fraction > 0.3, np.clip(tectonic, 0.0, None), 0.0)
    noise = FractalNoise(Stream(config.seed, "rocks/v3/craton"), [3200.0, 1600.0], [1.0, 0.5])
    craton = np.clip(0.5 + 0.42 * noise(x, y), 0.0, 1.0) * smoothstep((world_c - 150.0) / 700.0)
    div = skeleton.kind == TYPE_DIVERGENT
    if div.any():
        d, _ = cKDTree(np.column_stack([skeleton.bx[div], skeleton.by[div]])).query(
            np.column_stack([x.ravel(), y.ravel()])
        )
        ocean_age = np.clip(d.reshape(x.shape) / 35.0, 0.0, 170.0)
    else:
        ocean_age = np.full(x.shape, 90.0)
    age = fraction * (550.0 + 1900.0 * craton) + (1.0 - fraction) * ocean_age
    # Ancient sutures: where v2 welded two continental blocks, an old collision belt
    # runs across the weld. It is unrelated to today's boundaries, long extinct and worn.
    ages = np.full(x.shape, float(config.history.inherited_orogen_age_ma))
    ancient = np.zeros(x.shape)
    sutures = Stream(config.seed, "rocks/v3/sutures")
    welds = relief_model.continents.bridges if config.history.enabled else []
    for seg in welds:
        x1, y1, x2, y2, half_width = (float(v) for v in seg)
        u = sutures.random(3)
        mx, my = 0.5 * (x1 + x2), 0.5 * (y1 + y2)
        length = max(np.hypot(x2 - x1, y2 - y1), 1e-6)
        px_, py_ = -(y2 - y1) / length, (x2 - x1) / length  # across the weld
        half = 1.25 * half_width
        t_along = np.clip((x - mx) * px_ + (y - my) * py_, -half, half)
        d = np.hypot(x - (mx + t_along * px_), y - (my + t_along * py_))
        belt = (1100.0 + 900.0 * u[0]) * np.exp(-((d / 150.0) ** 2))
        belt = belt * (1.0 - 0.6 * (np.abs(t_along) / half) ** 2)
        stronger = belt > ancient
        ancient = np.where(stronger, belt, ancient)
        ages = np.where(stronger & (belt > 150.0), 190.0 + 240.0 * u[1], ages)
    ancient = np.where(fraction > 0.5, ancient, 0.0)
    older = ancient > inherited
    ages = np.where(older, ages, float(config.history.inherited_orogen_age_ma))
    return Baseline(
        grid=grid,
        plate=skeleton.plate.astype(np.int32),
        continental=fraction,
        elevation=elevation,
        inherited_relief=np.maximum(inherited, ancient),
        crust_age=age,
        craton=craton,
        inherited_age=ages,
        extra_elevation=0.45 * np.where(older, ancient, 0.0),
    )


def _state_from_arrays(arrays: dict) -> MaterialState:
    return MaterialState(**{k: np.array(arrays[k]) for k in MaterialState.FIELDS})


# -- one epoch on the world raster ------------------------------------------------
def epoch_world(sim: HistorySimulator, state: MaterialState, t: float, config: Config):
    """Mantle-frame rasters of one epoch needed by the plan and by hex sampling."""
    snap = sim.snapshot(t, state)
    base = sim.base
    mat = np.maximum(snap.material, 0)
    built = base.inherited_relief.ravel()[mat] + state.orogenic.ravel()[mat]
    share = np.clip(state.erosion.ravel()[mat] / (built + 1.0), 0.0, 0.9)
    relief = np.where(snap.covered, built * (1.0 - share), 0.0)
    last = state.orogen_last.ravel()[mat]
    age = np.where(snap.covered & np.isfinite(last), t - last, np.nan)
    kind = np.where(snap.covered, state.orogen_kind.ravel()[mat], 0).astype(np.uint8)
    surface = sim.surface(snap, state)
    return {
        "snap": snap,
        "relief": relief,
        "share": np.where(snap.covered, share, 0.0),
        "age": age,
        "kind": kind,
        "surface": surface,
    }


def _fill_from(mask, values):
    """Extend ``values`` defined on ``mask`` to every cell (nearest neighbour)."""
    if mask.all() or not mask.any():
        return values
    _d, inds = ndimage.distance_transform_edt(~mask, return_indices=True)
    return values[inds[0], inds[1]]


def sample_epoch(
    config: Config,
    layout,
    sim: HistorySimulator,
    state: MaterialState,
    t: float,
    world: dict,
    plan,
    volcanoes: VolcanoSet,
    relief_model,
    plate_model,
    base_macro: np.ndarray,
    base_plate: np.ndarray,
    report=None,
) -> dict:
    """Every v3 hex layer of one epoch except land / sea (which depend on the sea level)."""
    grid = sim.grid
    kin = sim.kin
    snap = world["snap"]
    hx, hy = layout.centers()
    shape = hx.shape
    px, py = hx.ravel(), hy.ravel()
    n = px.size

    # -- which plate's rock is at the hex centre ---------------------------------
    ri, ci, _ok = grid.nearest(px, py)
    owner = snap.owner[ri, ci].astype(np.int64)
    covered = snap.covered[ri, ci].copy()
    # near plate margins the 60-km raster is too coarse: decide analytically there
    fr, fc = grid.index(px, py)
    r0 = np.clip(np.floor(fr).astype(np.int64), 0, grid.ny - 2)
    c0 = np.clip(np.floor(fc).astype(np.int64), 0, grid.nx - 2)
    sig = snap.owner.astype(np.int64) * 4 + snap.covered * 2 + (snap.under >= 0)
    corners = [(r0, c0), (r0, c0 + 1), (r0 + 1, c0), (r0 + 1, c0 + 1)]
    s00 = sig[r0, c0]
    near = np.zeros(n, dtype=bool)
    for rr, cc in corners[1:]:
        near |= sig[rr, cc] != s00
    idx = np.nonzero(near)[0]
    if idx.size and t > 0.0:
        cand = []
        for rr, cc in corners:
            cand.append(snap.owner[rr[idx], cc[idx]].astype(np.int64))
            cand.append(snap.under[rr[idx], cc[idx]].astype(np.int64))
        best_p = np.full(idx.size, -np.inf)
        best_j = np.full(idx.size, -1, dtype=np.int64)
        for k, c in enumerate(cand):
            fresh = c >= 0
            for earlier in cand[:k]:
                fresh &= c != earlier
            if not fresh.any():
                continue
            sub = np.nonzero(fresh)[0]
            j = c[sub]
            qx, qy = kin.inverse(j, px[idx][sub], py[idx][sub], t)
            cov = plate_model.plate_at(qx, qy) == j
            prio = grid.bilinear(sim.priority0, qx, qy) - 1e-7 * j
            take = cov & (prio > best_p[sub])
            best_p[sub[take]] = prio[take]
            best_j[sub[take]] = j[take]
        covered[idx] = best_j >= 0
        owner[idx] = np.where(best_j >= 0, best_j, owner[idx])
    if t == 0.0:  # the initial snapshot is the baseline itself
        owner = base_plate.ravel().astype(np.int64)
        covered[:] = True
    if report:
        report(0.25)

    qx, qy = kin.inverse(owner, px, py, t)
    cov = np.nonzero(covered)[0]
    gap = np.nonzero(~covered)[0]

    # -- rock that rides with the plate: baseline surface + accumulated history ----
    continentality = np.full(n, -9999.0)
    macro = np.zeros(n)
    if cov.size:
        c = relief_model.continents.continentality(qx[cov], qy[cov])
        continentality[cov] = c
        macro[cov] = relief_model.elevation(qx[cov], qy[cov], continentality=c)
    if report:
        report(0.6)

    def material(field_):
        out = np.zeros(n, dtype=np.float32)
        if cov.size:
            out[cov] = grid.bilinear(field_, qx[cov], qy[cov])
        return out

    uplift = material(state.uplift)
    macro += material(sim.base.extra_elevation)  # ancient belts absent from the v2 relief
    erosion = np.clip(material(state.erosion), 0.0, None)
    sediment = np.clip(material(state.sediment), 0.0, None)
    volcanic = np.clip(material(state.volcanic), 0.0, None)
    macro += uplift - erosion + sediment + volcanic

    gap_mask = ~snap.covered
    if gap.size:
        floor = _fill_from(gap_mask, sim.new_crust_elevation(snap, state))
        floor = ndimage.gaussian_filter(floor, 0.8)  # hide the raster cells of the new floor
        macro[gap] = grid.bilinear(floor, px[gap], py[gap])
        sediment[gap] = np.clip(
            grid.bilinear(_fill_from(gap_mask, state.gap_sediment), px[gap], py[gap]), 0.0, None
        )
    macro += grid.bilinear(sim.trench_relief(snap), px, py)

    # -- mountains: structural relief inside the systems, coast protected ----------
    relief_field = grid.bilinear(world["relief"], px, py)
    gain_raster = np.full(grid.shape, 0.6)
    for system in plan.systems:
        gain_raster[plan.system_raster == system.id] = system.relief_gain
    gain_raster = ndimage.gaussian_filter(gain_raster, 1.0)
    patch = np.zeros(n)
    structure = np.zeros(n)
    valley = np.zeros(n)
    system_id = np.full(n, -1, dtype=np.int32)
    pass_dist = np.full(n, np.inf)
    ridge_dist = np.full(n, np.inf)
    backend = get_backend(config.mountains.backend)
    mnt = np.nonzero((relief_field > 60.0) & covered)[0]
    sampler = None
    if mnt.size and plan.ridges and config.mountains.detail > 0:
        request = MountainRequest(
            world_seed=config.seed,
            snapshot_id=f"t{t:g}",
            world_bounds_km=tuple(layout.bounds.as_list()),
            plan=plan,
            uplift=lambda x, y: grid.bilinear(world["relief"], x, y),
            gain=lambda x, y: grid.bilinear(gain_raster, x, y),
            detail=config.mountains.detail,
            max_elevation_m=config.mountains.max_elevation_m,
            policy="masked",
        )
        sampler = backend.sampler(request)
        result = sampler.sample_points(px[mnt], py[mnt])
        shield = smoothstep((macro[mnt] + 300.0) / 600.0)  # keep coasts and sea floor intact
        patch[mnt] = result.relief_m * shield
        structure[mnt] = result.structure
        valley[mnt] = result.valley
        system_id[mnt] = result.system_id
        ridge_dist[mnt] = result.ridge_distance_km
        pass_dist[mnt] = sampler.pass_distance(px[mnt], py[mnt])
    raster_system = plan.system_raster[ri, ci]
    in_system = (relief_field > config.mountains.system_threshold_m) & covered
    system_id = np.where(system_id >= 0, system_id, np.where(in_system, raster_system, -1))

    # -- volcanoes ------------------------------------------------------------------
    vx, vy, vidx = volcanoes.positions(kin, t)
    vheight = volcanoes.heights(vidx, t, config.history.erosion_intensity)
    vradius = np.array([volcanoes.volcanoes[int(i)].radius_km for i in vidx])
    cones, vnear, vdist = cone_field(px, py, vx, vy, vheight, vradius)
    volcano_id = np.full(n, -1, dtype=np.int32)
    volcanic_age = np.full(n, np.nan, dtype=np.float32)
    if vidx.size:
        hit = (vnear >= 0) & (vdist <= 1.25 * vradius[np.maximum(vnear, 0)])
        volcano_id[hit] = vidx[vnear[hit]]
        births = np.array([volcanoes.volcanoes[int(i)].birth_ma for i in vidx])
        volcanic_age[hit] = t - births[vnear[hit]]

    elevation = limit_elevation(macro + patch + cones, config.mountains.max_elevation_m)

    # -- diagnostics and rock ---------------------------------------------------------
    fraction = np.zeros(n, dtype=np.float32)
    fraction[cov] = relief_model.continents.fraction_of(continentality[cov])
    if gap.size:
        rift = _fill_from(gap_mask, snap.rift.astype(np.float64))
        width = _fill_from(gap_mask, snap.gap_half_width.astype(np.float64))
        is_rift = grid.bilinear(rift, px[gap], py[gap]) > 0.5
        stretch = np.clip(grid.bilinear(width, px[gap], py[gap]) / 130.0, 0.0, 1.0)
        fraction[gap] = np.where(is_rift, 0.45 * (1.0 - stretch), 0.0)
    crust_age = np.zeros(n, dtype=np.float32)
    crust_age[cov] = grid.bilinear(sim.base.crust_age, qx[cov], qy[cov]) + t
    if gap.size:
        new_age = _fill_from(gap_mask, np.nan_to_num(snap.new_crust_age, nan=0.0))
        crust_age[gap] = np.clip(grid.bilinear(new_age, px[gap], py[gap]), 0.0, None)

    def material_nearest(field_, fill=np.nan):
        out = np.full(n, fill, dtype=np.float32)
        if cov.size:
            rj, cj, _inside = grid.nearest(qx[cov], qy[cov])
            out[cov] = field_[rj, cj]
        return out

    last_orogen = material_nearest(state.orogen_last)
    last_tect = material_nearest(state.last_tectonic)
    craton = material(sim.base.craton)
    built = material(sim.base.inherited_relief) + material(state.orogenic)
    rock = np.where(fraction > 0.5, np.where(craton > 0.55, 1, 2), 0).astype(np.uint8)
    rock = np.where((built > 400.0) & covered, 3, rock)
    rock = np.where((volcanic > 150.0) | (volcano_id >= 0), 4, rock)
    rock = np.where(sediment > 150.0, 5, rock).astype(np.uint8)
    resistance = np.choose(rock, [0.62, 0.86, 0.45, 0.72, 0.6, 0.22]).astype(np.float32)
    resistance = np.where(rock == 1, 0.7 + 0.2 * craton, resistance).astype(np.float32)

    potential = np.clip(grid.bilinear(snap.arc_zone, px, py), 0.0, 1.0)
    if gap.size:
        young = np.exp(-np.clip(crust_age[gap], 0.0, None) / 2.5)
        potential[gap] = np.maximum(potential[gap], 0.85 * young)
    for h in volcanoes.hotspots:
        d = np.hypot(px - h.x, py - h.y)
        potential = np.maximum(
            potential, min(1.0, 0.6 + 0.3 * h.strength) * np.exp(-0.5 * (d / 170.0) ** 2)
        )

    # mountain classes: structure first, then what is left of the system
    cls = np.full(n, CLASS_NONE, dtype=np.uint8)
    cls[(system_id >= 0)] = CLASS_FOOTHILL
    cls[structure > 0.12] = CLASS_SLOPE
    cls[(valley > 0.45) & (structure < 0.6)] = CLASS_VALLEY
    # a hex is "crest" when a ridge axis runs through it, whatever the hex size
    cls[(structure > 0.62) | ((ridge_dist < 0.5 * layout.spacing_km) & (structure > 0.15))] = (
        CLASS_CREST
    )
    cls[(pass_dist < 0.55 * layout.spacing_km + 9.0) & (structure > 0.3)] = CLASS_PASS
    cls[volcano_id >= 0] = CLASS_VOLCANO

    # boundaries of the epoch: where the owning plate changes between hexes
    owner_grid = owner.reshape(shape).astype(np.int32)
    boundary = np.zeros(n, dtype=np.uint8)
    flat_owner = owner_grid.ravel()
    edge = np.zeros(n, dtype=bool)
    for d in range(6):
        nb, valid = raster_stage.neighbor_index(layout, d)
        edge |= valid.ravel() & (flat_owner[nb.ravel()] != flat_owner)
    cell_km = grid.cell
    near_col = snap.collision_distance[ri, ci] < 1.2 * cell_km
    near_sub = snap.trench_distance[ri, ci] < 1.5 * cell_km
    boundary[edge] = 4
    boundary[edge & ~covered] = 3
    boundary[edge & near_sub] = 2
    boundary[edge & near_col] = 1
    ridge_near = ndimage.binary_dilation(snap.ridge)[ri, ci] & ~covered
    boundary[ridge_near & (boundary == 0)] = 3

    elev_grid = elevation.reshape(shape).astype(np.float32)
    slope = np.zeros(n, dtype=np.float32)
    flat_e = elev_grid.ravel()
    for d in range(6):
        nb, valid = raster_stage.neighbor_index(layout, d)
        drop = np.where(valid.ravel(), np.abs(flat_e - flat_e[nb.ravel()]), 0.0)
        slope = np.maximum(slope, drop / layout.spacing_km)
    # height range of the continuous surface inside a hex: six extra samples of the
    # mountain relief where there are mountains, the neighbour slope elsewhere
    local = slope * 0.5 * layout.spacing_km
    if sampler is not None and mnt.size:
        lo = patch[mnt].copy()
        hi = patch[mnt].copy()
        radius = 0.6 * layout.radius
        for a in range(6):
            ox = radius * np.cos(a * np.pi / 3.0)
            oy = radius * np.sin(a * np.pi / 3.0)
            extra = sampler.sample_points(px[mnt] + ox, py[mnt] + oy).relief_m
            lo = np.minimum(lo, extra)
            hi = np.maximum(hi, extra)
        local[mnt] = np.maximum(local[mnt], (hi - lo).astype(np.float32))
    if report:
        report(1.0)

    def grid2(a, dtype):
        return np.ascontiguousarray(np.asarray(a).reshape(shape).astype(dtype))

    return {
        "elevation_m": elev_grid,
        "delta_elevation_m": (elev_grid - base_macro).astype(np.float32),
        "plate_id_epoch": owner_grid,
        "new_crust": grid2(~covered, np.bool_),
        "boundary_class": grid2(boundary, np.uint8),
        "continental_fraction": grid2(fraction, np.float32),
        "crust_age_ma": grid2(crust_age, np.float32),
        "rock_type": grid2(rock, np.uint8),
        "rock_resistance": grid2(resistance, np.float32),
        "uplift_rate_mm_yr": grid2(grid.bilinear(snap.uplift_rate, px, py) / 1000.0, np.float32),
        "uplift_total_m": grid2(uplift, np.float32),
        "erosion_total_m": grid2(erosion, np.float32),
        "sediment_thickness_m": grid2(sediment, np.float32),
        "mountain_system_id": grid2(system_id, np.int32),
        "mountain_class": grid2(cls, np.uint8),
        "orogen_age_ma": grid2(t - last_orogen, np.float32),
        "last_tectonic_ma": grid2(t - last_tect, np.float32),
        "volcanic_potential": grid2(potential, np.float32),
        "volcano_id": grid2(volcano_id, np.int32),
        "volcanic_age_ma": grid2(volcanic_age, np.float32),
        "slope_m_per_km": grid2(slope, np.float32),
        "local_relief_m": grid2(local, np.float32),
    }


def _volcano_table(volcanoes: VolcanoSet, kin, t, layout, elevation, sea, erosion) -> list[dict]:
    vx, vy, vidx = volcanoes.positions(kin, t)
    heights = volcanoes.heights(vidx, t, erosion)
    out = []
    for k, i in enumerate(vidx):
        v = volcanoes.volcanoes[int(i)]
        col, row = layout.world_to_offset(vx[k], vy[k])
        col, row = int(col), int(row)
        inside = bool(layout.inside(col, row))
        summit = float(elevation[row, col]) if inside else None
        out.append(
            {
                "id": v.id,
                "kind": v.kind,
                "hotspot_id": v.source if v.kind == HOTSPOT else None,
                "plate": v.plate,
                "subducting_plate": v.partner if v.kind == ARC else None,
                "x_km": round(float(vx[k]), 2),
                "y_km": round(float(vy[k]), 2),
                "birth_ma": round(v.birth_ma, 4),
                "age_ma": round(t - v.birth_ma, 4),
                "active": volcanoes.is_active(int(i), t),
                "edifice_height_m": round(float(heights[k]), 1),
                "radius_km": round(v.radius_km, 2),
                "in_window": inside,
                "summit_elevation_m": summit,
                "submarine": (summit <= sea) if summit is not None else None,
            }
        )
    return out


def run_v3(
    config: Config, geometry: WorldGeometry, project, old_keys: dict, old_cache: dict, report
):
    """Run the v3 stages for ``project`` (which already has v1 and v2 data)."""
    layout = geometry.layout
    geo = project.geo
    timings: dict[str, float] = {}
    reused: list[str] = []
    cache: dict = {}
    hist_key = v3_stage_keys(config, geo.stage_keys, layout, None)["geo_history"]

    def cached(stage, key):
        if old_keys.get(stage) == key and stage in old_cache:
            return old_cache[stage]
        return None

    # -- history ----------------------------------------------------------------
    t0 = time.perf_counter()
    entry = cached("geo_history", hist_key)
    sim = entry.get("sim") if entry else None
    need_sim = entry is None or sim is None

    def make_sim():
        skeleton, relief_model = ensure_v2_models(
            config,
            geometry,
            project.plate_model,
            project.crust_model,
            project.motion,
            project.cache,
        )
        base = build_baseline(config, skeleton, relief_model)
        kin = Kinematics(project.motion.velocity, project.motion.omega, project.motion.pole)
        sb = geometry.simulation_bounds
        pad = 0.08 * max(sb.width, sb.height)
        focus = (sb.xmin - pad, sb.ymin - pad, sb.xmax + pad, sb.ymax + pad)
        return HistorySimulator(base, kin, config.history, config.seed, focus), relief_model

    if entry is None:
        report("geo_history", 0.80)
        sim, relief_model = make_sim()
        sb = geometry.simulation_bounds
        margin = 0.05 * max(sb.width, sb.height)
        hotspots = place_hotspots(
            config.seed,
            (sb.xmin - margin, sb.ymin - margin, sb.xmax + margin, sb.ymax + margin),
            sb.width * sb.height,
            config.history.hotspot_density if config.history.enabled else 0.0,
        )
        builder = VolcanoBuilder(config.seed, sim.kin, hotspots, sim.focus)
        gain = sim.intensity

        def observe(snap, t):
            builder.step(snap, sim.grid, snap.continental, t, gain)

        checkpoints, events, balance, info = run_history(
            sim, progress=lambda f: report("geo_history", 0.80 + 0.08 * f), observer=observe
        )
        entry = {
            "sim": sim,
            "relief_model": relief_model,
            "checkpoints": {t: s for t, s in checkpoints},
            "events": events,
            "balance": balance,
            "info": info,
            "volcanoes": builder.set,
        }
    else:
        reused.append("geo_history")
        if need_sim:
            entry = dict(entry)
            entry["sim"], entry["relief_model"] = None, None
    cache["geo_history"] = entry
    timings["geo_history"] = round(time.perf_counter() - t0, 4)
    epochs = sorted(entry["checkpoints"])
    t_epoch = resolve_epoch(config, epochs)
    keys = v3_stage_keys(config, geo.stage_keys, layout, t_epoch)

    # -- plan + raster of the selected epoch --------------------------------------
    raster_entry = cached("v3_raster", keys["v3_raster"])
    plan_entry = cached("mountain_plan", keys["mountain_plan"])
    if raster_entry is None:
        if entry.get("sim") is None:  # loaded from disk: rebuild the cheap deterministic parts
            entry["sim"], entry["relief_model"] = make_sim()
        sim = entry["sim"]
        state = entry["checkpoints"][t_epoch]
        t0 = time.perf_counter()
        report("mountain_plan", 0.885)
        world = epoch_world(sim, state, t_epoch, config)
        if plan_entry is None:
            plan = build_plan(
                sim.grid,
                world["relief"],
                world["kind"],
                world["age"],
                world["share"],
                world["surface"],
                config.mountains,
                config.seed,
                t_epoch,
                focus=sim.focus,
            )
        else:
            plan = plan_entry
            reused.append("mountain_plan")
        timings["mountain_plan"] = round(time.perf_counter() - t0, 4)
        t0 = time.perf_counter()
        layers = sample_epoch(
            config,
            layout,
            sim,
            state,
            t_epoch,
            world,
            plan,
            entry["volcanoes"],
            entry["relief_model"],
            project.plate_model,
            geo.layers["macro_elevation_m"],
            project.layers["plate_id"],
            report=lambda f: report("v3_raster", 0.89 + 0.07 * f),
        )
        raster_entry = {
            "layers": layers,
            "world": {
                "grid": sim.grid.to_dict(),
                "elevation_m": world["surface"].astype(np.float32),
                "orogenic_relief_m": world["relief"].astype(np.float32),
                "plate_id": world["snap"].owner.astype(np.int32),
                "new_crust": ~world["snap"].covered,
                "mountain_system_id": plan.system_raster,
            },
            "rates": {
                "max_uplift_rate_mm_yr": float(world["snap"].uplift_rate.max()) / 1000.0,
                "overlap_cells": int((world["snap"].under >= 0).sum()),
                "gap_cells": int((~world["snap"].covered).sum()),
            },
        }
        timings["v3_raster"] = round(time.perf_counter() - t0, 4)
        plan_entry = plan
    else:
        reused.extend(["mountain_plan", "v3_raster"])
        plan_entry = plan_entry if plan_entry is not None else old_cache.get("mountain_plan")
    cache["mountain_plan"] = plan_entry
    cache["v3_raster"] = raster_entry
    plan = plan_entry

    # -- land and sea from the one selected surface ---------------------------------
    t0 = time.perf_counter()
    report("v3_sea", 0.965)
    layers = dict(raster_entry["layers"])
    elev = layers["elevation_m"]
    sea = geo.sea_level_m
    land, landmass_id, land_sizes, shelf = derive_land(
        elev, layers["continental_fraction"], sea, layout, config.sea
    )
    layers["land_mask"], layers["landmass_id"], layers["shelf_mask"] = land, landmass_id, shelf
    hx, hy = layout.centers()
    lands = _component_table(landmass_id, land_sizes, layout, layers["plate_id_epoch"], hx, hy)
    names = _names(config.seed, "sea_level/v3/landmass_names", len(lands))
    for m, name in zip(lands, names, strict=True):
        m["name"] = name
        m["is_major"] = bool(m["area_km2"] >= config.continents.major_area_km2)

    systems = []
    sys_ids = layers["mountain_system_id"]
    plan_dict = plan.to_dict()
    for s in plan_dict["systems"]:
        inside = sys_ids == s["id"]
        s = dict(s)
        s["visible_hexes"] = int(inside.sum())
        s["max_elevation_m"] = float(elev[inside].max()) if inside.any() else None
        systems.append(s)
    plan_dict["systems"] = systems
    for group in ("peaks", "passes"):
        for item in plan_dict[group]:
            col, row = layout.world_to_offset(item["x_km"], item["y_km"])
            col, row = int(col), int(row)
            ok = bool(layout.inside(col, row))
            item["elevation_m"] = float(elev[row, col]) if ok else None
    sim = entry.get("sim")
    volcano_set: VolcanoSet = entry["volcanoes"]
    kin = (
        sim.kin
        if sim is not None
        else Kinematics(project.motion.velocity, project.motion.omega, project.motion.pole)
    )
    volcano_table = _volcano_table(
        volcano_set, kin, t_epoch, layout, elev, sea, config.history.erosion_intensity
    )
    info = entry["info"]
    balance = entry["balance"]
    eroded = sum(b["eroded_m_cells"] for b in balance)
    deposited = sum(b["deposited_m_cells"] for b in balance)
    events = [dict(e) for e in entry["events"] if e["time_ma"] <= t_epoch + 1e-9]
    for h in volcano_set.hotspots:
        chain = [v for v in volcano_set.volcanoes if v.source == h.id and v.kind == HOTSPOT]
        chain = [v for v in chain if v.birth_ma <= t_epoch + 1e-9]
        if chain:
            events.append(
                {
                    "id": f"hotspot_volcanism:{h.id}",
                    "type": "volcanic_episode",
                    "time_ma": round(min(v.birth_ma for v in chain), 6),
                    "x_km": round(h.x, 1),
                    "y_km": round(h.y, 1),
                    "plates": sorted({v.plate for v in chain}),
                    "roles": ["above hotspot"],
                    "intensity_mm_yr": 0.0,
                    "cells": len(chain),
                    "end_time_ma": None,
                    "derived": True,
                }
            )
    for s in plan_dict["systems"]:
        if s["kind"] in ("collision", "arc") and s["age_ma"] <= t_epoch:
            events.append(
                {
                    "id": f"orogeny:{s['id']}@{t_epoch:g}",
                    "type": "orogeny",
                    "time_ma": round(max(0.0, t_epoch - s["age_ma"]), 6),
                    "x_km": round(s["centroid_km"][0], 1),
                    "y_km": round(s["centroid_km"][1], 1),
                    "plates": [],
                    "roles": [],
                    "intensity_mm_yr": 0.0,
                    "cells": int(s["area_km2"] / (info.get("cell_km", 1.0) ** 2 or 1.0)),
                    "end_time_ma": None,
                    "mountain_system": s["id"],
                    "derived": True,
                }
            )
    events.sort(key=lambda e: (e["time_ma"], e["id"]))
    delta = layers["delta_elevation_m"]
    summary = {
        "epoch_time_ma": t_epoch,
        "duration_ma": epochs[-1],
        "epochs_ma": epochs,
        "steps": info.get("steps", 0),
        "substeps_per_step": info.get("substeps", 0),
        "time_step_ma": info.get("dt_ma"),
        "max_plate_speed_mm_yr": info.get("max_speed_mm_yr"),
        "sea_level_m": sea,
        "land_fraction_viewport": float(land.mean()),
        "baseline_land_fraction_viewport": float(geo.layers["land_mask"].mean()),
        "min_elevation_m": float(elev.min()),
        "max_elevation_m": float(elev.max()),
        "max_abs_delta_elevation_m": float(np.abs(delta).max()),
        "changed_hex_share": float((np.abs(delta) > 50.0).mean()),
        "new_crust_share_viewport": float(layers["new_crust"].mean()),
        "mountain_systems": len(plan_dict["systems"]),
        "ridges": len(plan_dict["ridges"]),
        "peaks": len(plan_dict["peaks"]),
        "passes": len(plan_dict["passes"]),
        "valleys": len(plan_dict["valleys"]),
        "volcanoes": len(volcano_table),
        "active_volcanoes": int(sum(1 for v in volcano_table if v["active"])),
        "hotspots": len(volcano_set.hotspots),
        "events": len(events),
        "landmasses_in_window": len(lands),
        "world_rates": raster_entry["rates"],
        "erosion_balance": {
            "unit": "m x raster cells (multiply by cell area for volume)",
            "eroded": eroded,
            "deposited": deposited,
            "left_the_system": eroded - deposited,
            "note": "about 15 % of the eroded rock is treated as dissolved / far-travelled "
            "load and is not deposited; the balance is diagnostic, not closed",
        },
        "mountain_backend": {
            "name": get_backend(config.mountains.backend).name,
            **get_backend(config.mountains.backend).capabilities(),
        },
        "rock_types": list(ROCK_TYPES),
    }
    warnings = []
    halo = geometry.halo_km
    travel = (info.get("max_speed_mm_yr") or 0.0) * epochs[-1]
    if travel > 0.6 * halo:
        warnings.append(
            f"Plates travel up to {travel:,.0f} km in {epochs[-1]:g} Ma, close to the "
            f"{halo:,.0f} km halo of the model area: its rim may show edge effects."
        )
    if not summary["mountain_backend"]["is_pmge"]:
        warnings.append("Mountains use the built-in reference backend; PMGE is not integrated.")
    timings["v3_sea"] = round(time.perf_counter() - t0, 4)

    hist = HistoryData(
        layers=layers,
        epoch_time_ma=float(t_epoch),
        epochs=[float(t) for t in epochs],
        duration_ma=float(epochs[-1]),
        sea_level_m=float(sea),
        events=events,
        plan=plan_dict,
        volcanoes=volcano_table,
        hotspots=[
            {
                "id": h.id,
                "x_km": round(h.x, 2),
                "y_km": round(h.y, 2),
                "strength": round(h.strength, 3),
                "interval_ma": round(h.interval_ma, 3),
            }
            for h in volcano_set.hotspots
        ],
        landmasses=lands,
        summary=summary,
        stage_keys=keys,
        world=raster_entry["world"],
        checkpoints={
            float(t): {k: v for k, v in s.arrays().items()} for t, s in entry["checkpoints"].items()
        },
        balance=balance,
        warnings=warnings,
        timings=timings,
    )
    return hist, cache, reused
