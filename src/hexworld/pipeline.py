"""The staged pipeline::

    config/seed -> world_geometry -> plates -> crust -> motion -> boundary_edges
                -> tectonic_fields -> validate

Each stage has an explicit hash key built from its own parameters and the keys
of the stages it depends on. ``generate(config, reuse=old_project)`` recomputes
only the stages whose keys changed, e.g. new motion parameters keep plates and
crust untouched, and a display-only change touches nothing at all.
"""

from __future__ import annotations

import colorsys
import hashlib
import json
import time
from collections.abc import Callable
from dataclasses import asdict

import numpy as np

from .config import Config
from .geometry import WorldGeometry, resolve_geometry
from .project import Project
from .rng import Stream
from .stages import raster as raster_stage
from .stages.boundaries import extract_boundaries, summarize_pairs
from .stages.crust import build_crust_model
from .stages.fields import compute_fields, hex_boundary_type
from .stages.motion import build_motion
from .stages.plates import SIZE_CLASSES, build_plate_model
from .stages.reference import build_reference_grid, reference_adjacency
from .stages.validate import validate_project
from .version import ALGORITHM_VERSION, PRNG_VERSION

STAGES = ("plates", "crust", "motion", "raster", "boundary_edges", "tectonic_fields")
Progress = Callable[[str, float], None]

_ONSETS = [
    "B",
    "D",
    "G",
    "K",
    "L",
    "M",
    "N",
    "R",
    "S",
    "T",
    "V",
    "Z",
    "Br",
    "Dr",
    "Kh",
    "Sh",
    "Th",
    "Tr",
    "Val",
    "Kar",
    "Mor",
    "Sel",
    "Tar",
    "An",
    "El",
    "Or",
    "Ur",
    "Is",
]
_MIDDLES = ["a", "e", "i", "o", "u", "ae", "ai", "ou", "ara", "eri", "oro", "una", "ali", "eno"]
_CODAS = [
    "n",
    "r",
    "s",
    "th",
    "l",
    "m",
    "nd",
    "rn",
    "sk",
    "x",
    "ria",
    "nia",
    "tis",
    "mar",
    "dor",
    "gan",
    "ven",
    "zan",
    "lune",
    "rak",
]


def _key(*parts) -> str:
    return hashlib.sha256(json.dumps(parts, sort_keys=True, default=str).encode()).hexdigest()


def stage_keys(config: Config, geometry: WorldGeometry, pad: int) -> dict[str, str]:
    world = _key(
        ALGORITHM_VERSION,
        PRNG_VERSION,
        config.seed,
        geometry.simulation_bounds.as_list(),
        geometry.plate_count,
        geometry.profile,
        config.reference_cells,
    )
    # `count` / `profile` enter through the *resolved* geometry above, so an "auto"
    # count and the same explicit number describe the same world.
    plate_params = {k: v for k, v in asdict(config.plates).items() if k not in ("count", "profile")}
    plates = _key("plates", world, plate_params)
    crust = _key("crust", world, asdict(config.crust))
    motion = _key("motion", plates, crust, asdict(config.motion))
    lay = geometry.layout
    raster = _key(
        "raster", plates, crust, lay.width, lay.height, lay.spacing_km, lay.x0, lay.y0, pad
    )
    boundaries = _key("boundaries", raster, motion, asdict(config.boundaries))
    fields = _key("fields", boundaries, asdict(config.fields))
    return {
        "plates": plates,
        "crust": crust,
        "motion": motion,
        "raster": raster,
        "boundary_edges": boundaries,
        "tectonic_fields": fields,
    }


def world_id_from_keys(keys: dict[str, str]) -> str:
    return _key("world", keys["plates"], keys["crust"], keys["motion"])[:16]


def _pad_hexes(config: Config) -> int:
    pad = int(np.ceil(config.fields.max_influence_km / config.hex_spacing_km))
    pad = min(pad, config.fields.max_pad_hexes)
    return pad + (pad % 2)


def _plate_names(seed: str, count: int) -> list[str]:
    stream = Stream(seed, "plates/names")
    names: list[str] = []
    used: set[str] = set()
    while len(names) < count:
        name = (
            _ONSETS[stream.integer(len(_ONSETS))]
            + _MIDDLES[stream.integer(len(_MIDDLES))]
            + _CODAS[stream.integer(len(_CODAS))]
        )
        if name in used:
            name = f"{name}-{len(names)}"
        used.add(name)
        names.append(name)
    return names


def _plate_colors(seed: str, count: int) -> list[str]:
    stream = Stream(seed, "plates/colors")
    offset = stream.random()
    out = []
    for i in range(count):
        hue = (offset + i * 0.6180339887498949) % 1.0
        sat = 0.42 + 0.26 * ((i * 7) % 5) / 4.0
        val = 0.72 + 0.2 * ((i * 3) % 4) / 3.0
        r, g, b = colorsys.hsv_to_rgb(hue, sat, val)
        out.append(f"#{int(r * 255):02x}{int(g * 255):02x}{int(b * 255):02x}")
    return out


def _reference_stats(ref, count: int):
    """World-level per-plate area, centroid and crust shares from the reference raster."""
    cell = ref.cell_area_km2
    sim = ref.in_sim.ravel()
    pid = ref.plate.ravel()
    cont = ref.crust.ravel().astype(np.float64)
    x = ref.x.ravel()
    y = ref.y.ravel()
    n_sim = np.bincount(pid[sim], minlength=count).astype(np.float64)
    n_all = np.bincount(pid, minlength=count).astype(np.float64)
    c_sim = np.bincount(pid[sim], weights=cont[sim], minlength=count)
    c_all = np.bincount(pid, weights=cont, minlength=count)
    sx = np.bincount(pid[sim], weights=x[sim], minlength=count)
    sy = np.bincount(pid[sim], weights=y[sim], minlength=count)
    with np.errstate(invalid="ignore", divide="ignore"):
        frac = np.where(n_sim > 0, c_sim / n_sim, np.where(n_all > 0, c_all / n_all, np.nan))
        cx = np.where(n_sim > 0, sx / n_sim, np.nan)
        cy = np.where(n_sim > 0, sy / n_sim, np.nan)
    return n_sim * cell, frac, cx, cy


def generate(
    config: Config,
    progress: Progress | None = None,
    reuse: Project | None = None,
) -> Project:
    """Run the pipeline. ``reuse`` lets unchanged stages be taken from a previous project."""
    report = progress or (lambda stage, fraction: None)
    timings: dict[str, float] = {}
    t_total = time.perf_counter()
    geometry = resolve_geometry(config)
    pad = _pad_hexes(config)
    keys = stage_keys(config, geometry, pad)
    old_keys = reuse.stage_keys if reuse is not None else {}
    old = reuse.cache if reuse is not None else {}
    reused: list[str] = []

    def fresh(stage: str) -> bool:
        if old_keys.get(stage) == keys[stage] and stage in old:
            reused.append(stage)
            return False
        return True

    def timed(stage: str, t0: float):
        timings[stage] = round(time.perf_counter() - t0, 4)

    cache: dict = {}
    layout = geometry.layout
    padded = layout.padded(pad)

    # -- plates (world model + reference raster) ----------------------------
    report("plates", 0.02)
    t0 = time.perf_counter()
    if fresh("plates"):
        plate_model = build_plate_model(config, geometry)
        ref_plate = None
    else:
        plate_model, ref_plate = old["plates"]
    ref = build_reference_grid(geometry, config.reference_cells)
    if ref_plate is None:
        ref_plate = plate_model.plate_at(ref.x, ref.y)
    ref.plate = ref_plate
    cache["plates"] = (plate_model, ref_plate)
    timed("plates", t0)

    # -- crust --------------------------------------------------------------
    report("crust", 0.12)
    t0 = time.perf_counter()
    if fresh("crust"):
        crust_model = build_crust_model(config, geometry, ref.x[ref.in_sim], ref.y[ref.in_sim])
        ref_crust = crust_model.crust_at(ref.x, ref.y)
    else:
        crust_model, ref_crust = old["crust"]
    ref.crust = ref_crust
    cache["crust"] = (crust_model, ref_crust)
    cache["reference"] = ref
    timed("crust", t0)

    count = plate_model.count
    area, cont_frac, cen_x, cen_y = _reference_stats(ref, count)
    site_crust = crust_model.crust_at(plate_model.sites[:, 0], plate_model.sites[:, 1])
    cont_frac = np.where(np.isnan(cont_frac), site_crust.astype(np.float64), cont_frac)

    # -- motion -------------------------------------------------------------
    report("motion", 0.22)
    t0 = time.perf_counter()
    if fresh("motion"):
        motion = build_motion(config, plate_model.sites, 1.0 - cont_frac, area)
    else:
        motion = old["motion"]
    cache["motion"] = motion
    timed("motion", t0)

    # -- raster: sample the world models on the (padded) hex grid ------------
    report("raster", 0.28)
    t0 = time.perf_counter()
    if fresh("raster"):
        px, py = padded.centers()
        plate_pad = plate_model.plate_at(px, py)
        report("raster", 0.5)
        plate_pad, orphans = raster_stage.absorb_orphans(plate_pad, padded)
        cont_pad = crust_model.continentality(px, py).astype(np.float32)
        raster = (plate_pad, cont_pad, orphans)
    else:
        raster = old["raster"]
    plate_pad, cont_pad, orphans = raster
    cache["raster"] = raster
    crust_pad = (cont_pad > 0).astype(np.uint8)
    timed("raster", t0)

    # -- boundary edges -----------------------------------------------------
    report("boundary_edges", 0.68)
    t0 = time.perf_counter()
    if fresh("boundary_edges"):
        edges_pad = extract_boundaries(
            plate_pad, crust_pad, padded, motion, config.boundaries, plate_model
        )
    else:
        edges_pad = old["boundary_edges"]
    cache["boundary_edges"] = edges_pad
    timed("boundary_edges", t0)

    # -- tectonic fields ----------------------------------------------------
    report("tectonic_fields", 0.76)
    t0 = time.perf_counter()
    if fresh("tectonic_fields"):
        fields_pad = compute_fields(edges_pad, padded, config.fields)
    else:
        fields_pad = old["tectonic_fields"]
    cache["tectonic_fields"] = fields_pad
    timed("tectonic_fields", t0)

    # -- crop to the viewport and assemble -----------------------------------
    report("assemble", 0.93)
    t0 = time.perf_counter()
    crop = (slice(pad, pad + layout.height), slice(pad, pad + layout.width))
    plate_id = np.ascontiguousarray(plate_pad[crop]).astype(np.int32)
    inside = (
        (edges_pad.col_a >= pad)
        & (edges_pad.col_a < pad + layout.width)
        & (edges_pad.row_a >= pad)
        & (edges_pad.row_a < pad + layout.height)
        & (edges_pad.col_b >= pad)
        & (edges_pad.col_b < pad + layout.width)
        & (edges_pad.row_b >= pad)
        & (edges_pad.row_b < pad + layout.height)
    )
    edges = edges_pad.select(inside)
    for name in ("col_a", "row_a", "col_b", "row_b"):
        setattr(edges, name, getattr(edges, name) - np.int32(pad))

    hx, hy = layout.centers()
    vx, vy = motion.velocity_at(plate_id, hx, hy)
    layers: dict[str, np.ndarray] = {
        "plate_id": plate_id,
        "crust_type": np.ascontiguousarray(crust_pad[crop]),
        "continentality_km": np.ascontiguousarray(cont_pad[crop]),
        "velocity_x": vx.astype(np.float32),
        "velocity_y": vy.astype(np.float32),
        "boundary_type": hex_boundary_type(edges, plate_id.shape),
        "boundary_distance_km": np.ascontiguousarray(fields_pad["boundary_distance_km"][crop]),
    }
    for name in ("uplift", "rifting", "subduction", "volcanism", "seismicity"):
        layers[name] = np.ascontiguousarray(fields_pad[name][crop])

    # -- plate table --------------------------------------------------------
    names = _plate_names(config.seed, count)
    colors = _plate_colors(config.seed, count)
    adjacency = reference_adjacency(ref, count)
    visible = np.bincount(plate_id.ravel(), minlength=count)
    vis_cont = np.bincount(
        plate_id.ravel(), weights=layers["crust_type"].ravel().astype(np.float64), minlength=count
    )
    plates: list[dict] = []
    for i in range(count):
        vxi, vyi = (float(v) for v in motion.velocity[i])
        plates.append(
            {
                "id": i,
                "name": names[i],
                "color": colors[i],
                "size_class": SIZE_CLASSES[int(plate_model.size_class[i])],
                "halo": bool(plate_model.halo[i]),
                "site_km": [float(v) for v in plate_model.sites[i]],
                "weight_km": float(plate_model.weights[i]),
                "area_km2": float(area[i]),
                "centroid_km": None if np.isnan(cen_x[i]) else [float(cen_x[i]), float(cen_y[i])],
                "continental_fraction": float(cont_frac[i]),
                "oceanic_fraction": float(1.0 - cont_frac[i]),
                "velocity_mm_yr": [vxi, vyi],
                "speed_mm_yr": float(np.hypot(vxi, vyi)),
                "omega_deg_myr": float(motion.omega[i]),
                "rotation_pole_km": [float(v) for v in motion.pole[i]],
                "neighbors": adjacency[i],
                "visible_hexes": int(visible[i]),
                "visible_area_km2": float(visible[i] * layout.hex_area_km2),
                "visible_continental_fraction": (
                    float(vis_cont[i] / visible[i]) if visible[i] else None
                ),
            }
        )

    project = Project(
        config=config,
        geometry=geometry,
        world_id=world_id_from_keys(keys),
        plates=plates,
        layers=layers,
        boundaries=edges,
        boundary_summary=summarize_pairs(edges),
        motion=motion,
        plate_model=plate_model,
        crust_model=crust_model,
        stage_keys=keys,
        warnings=list(geometry.warnings),
        timings=timings,
        stats={
            "pad_hexes": pad,
            "orphan_hexes_reassigned": int(orphans),
            "reused_stages": reused,
            "plates_total": count,
            "plates_in_simulation": int((~plate_model.halo).sum()),
            "plates_visible": int(np.count_nonzero(visible)),
            "continental_fraction_visible": float(layers["crust_type"].mean()),
        },
        cache=cache,
    )
    timed("assemble", t0)

    report("validate", 0.96)
    t0 = time.perf_counter()
    project.validation = validate_project(project)
    project.warnings.extend(project.validation["warnings"])
    if project.stats["plates_visible"] < 2:
        project.warnings.append(
            "Only one plate is visible: there are no boundaries in this window."
        )
    timed("validate", t0)
    timings["total"] = round(time.perf_counter() - t_total, 4)
    report("done", 1.0)
    return project
