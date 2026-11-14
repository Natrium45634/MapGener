"""v2.0 stages, run strictly *after* the v1.0 outputs::

    v1: world -> plates -> prior_crust -> motion -> boundary_edges -> tectonic_fields
    v2:   world_tectonics -> continental_blocks -> macro_elevation -> sea_level_landmasses

v1.0 outputs are only read. Each v2 stage has its own key, so changing the
continent shape does not move plates and changing the sea level recomputes
nothing but the land mask, landmass ids and shelf.
"""

from __future__ import annotations

import time
from dataclasses import asdict

import numpy as np
from scipy import ndimage

from .config import Config
from .geometry import HexLayout, WorldGeometry
from .project import GeoData
from .rng import Stream
from .stages.continents import build_continents
from .stages.relief import build_relief
from .stages.sea import derive_land, label_mask, resolve_sea_level, world_landmasses
from .stages.skeleton import build_skeleton
from .version import CONTINENTS_ALGORITHM_VERSION

V2_STAGES = (
    "world_tectonics",
    "continental_blocks",
    "macro_elevation",
    "v2_raster",
    "sea_level_landmasses",
)

_SYL_A = [
    "Ar",
    "Bel",
    "Cor",
    "Dal",
    "El",
    "Fen",
    "Gal",
    "Har",
    "Il",
    "Jor",
    "Kel",
    "Lor",
    "Mer",
    "Nor",
    "Or",
    "Pel",
    "Quen",
    "Ros",
    "Sar",
    "Tal",
    "Ul",
    "Ver",
    "Wes",
    "Yar",
    "Zan",
]
_SYL_B = ["a", "e", "i", "o", "u", "an", "en", "or", "ar", "ia", "ea", "ou"]
_SYL_C = [
    "dia",
    "land",
    "mar",
    "nor",
    "ria",
    "thia",
    "var",
    "wen",
    "zia",
    "gard",
    "mund",
    "tica",
    "sia",
    "lon",
    "rath",
    "dor",
]


def _key(*parts) -> str:
    import hashlib
    import json

    return hashlib.sha256(json.dumps(parts, sort_keys=True, default=str).encode()).hexdigest()


def v2_stage_keys(config: Config, v1_keys: dict[str, str], layout: HexLayout) -> dict[str, str]:
    cont = asdict(config.continents)
    cells = cont.pop("skeleton_cells")
    world = _key(
        "world_tectonics",
        CONTINENTS_ALGORITHM_VERSION,
        v1_keys["motion"],
        asdict(config.boundaries),
        cells,
    )
    blocks = _key("continental_blocks", world, v1_keys["crust"], cont)
    relief = _key("macro_elevation", blocks, asdict(config.relief))
    raster = _key(
        "v2_raster", relief, layout.width, layout.height, layout.spacing_km, layout.x0, layout.y0
    )
    sea = _key("sea_level_landmasses", raster, asdict(config.sea), config.continents.major_area_km2)
    return {
        "world_tectonics": world,
        "continental_blocks": blocks,
        "macro_elevation": relief,
        "v2_raster": raster,
        "sea_level_landmasses": sea,
    }


def _names(seed: str, namespace: str, count: int) -> list[str]:
    stream = Stream(seed, namespace)
    out: list[str] = []
    used: set[str] = set()
    for i in range(count):
        name = (
            _SYL_A[stream.integer(len(_SYL_A))]
            + _SYL_B[stream.integer(len(_SYL_B))]
            + _SYL_C[stream.integer(len(_SYL_C))]
        )
        if name in used:
            name = f"{name} {i}"
        used.add(name)
        out.append(name)
    return out


def _component_table(ids, sizes, layout: HexLayout, plate_id, hx, hy) -> list[dict]:
    """Per-component metadata: area, centre, bounds, window contact, plate shares."""
    count = int(sizes.size)
    if count == 0:
        return []
    inside = ids >= 0
    flat = ids[inside]
    index = np.arange(count)
    sx = np.bincount(flat, weights=hx[inside], minlength=count)
    sy = np.bincount(flat, weights=hy[inside], minlength=count)
    lab = ids + 1
    r = layout.radius
    half = 0.5 * layout.spacing_km
    xmin = ndimage.minimum(hx, lab, index + 1) - r
    xmax = ndimage.maximum(hx, lab, index + 1) + r
    ymin = ndimage.minimum(hy, lab, index + 1) - half
    ymax = ndimage.maximum(hy, lab, index + 1) + half
    border = np.zeros(ids.shape, dtype=bool)
    border[0, :] = border[-1, :] = border[:, 0] = border[:, -1] = True
    touches = np.zeros(count, dtype=bool)
    touches[ids[border & inside]] = True
    n_plates = int(plate_id.max()) + 1
    codes, counts = np.unique(
        flat.astype(np.int64) * n_plates + plate_id[inside], return_counts=True
    )
    shares: list[dict] = [{} for _ in range(count)]
    for code, n in zip(codes.tolist(), counts.tolist(), strict=True):
        shares[code // n_plates][int(code % n_plates)] = n
    area = layout.hex_area_km2
    out = []
    for i in range(count):
        top = sorted(shares[i].items(), key=lambda kv: (-kv[1], kv[0]))[:6]
        out.append(
            {
                "id": i,
                "hexes": int(sizes[i]),
                "area_km2": float(sizes[i] * area),
                "centroid_km": [float(sx[i] / sizes[i]), float(sy[i] / sizes[i])],
                "bounds_km": [float(xmin[i]), float(ymin[i]), float(xmax[i]), float(ymax[i])],
                "touches_window_edge": bool(touches[i]),
                "plates": {str(pid): round(n / int(sizes[i]), 4) for pid, n in top},
                "plate_count": len(shares[i]),
            }
        )
    return out


def _overlaps(a_ids, b_ids, a_count, b_count):
    """For components of ``a``: {b_id: hexes} (b = -1 means outside any ``b`` component)."""
    out: list[dict] = [{} for _ in range(a_count)]
    inside = a_ids >= 0
    codes, counts = np.unique(
        a_ids[inside].astype(np.int64) * (b_count + 1) + b_ids[inside] + 1, return_counts=True
    )
    for code, n in zip(codes.tolist(), counts.tolist(), strict=True):
        out[code // (b_count + 1)][int(code % (b_count + 1)) - 1] = n
    return out


def derive_geo_tables(config, layout, plate_id, layers, sea_level_m, block_sizes, land_sizes):
    """Metadata of continental blocks and landmasses for the current window."""
    hx, hy = layout.centers()
    block_id = layers["continental_block_id"]
    land_id = layers["landmass_id"]
    elev = layers["macro_elevation_m"]
    blocks = _component_table(block_id, block_sizes, layout, plate_id, hx, hy)
    lands = _component_table(land_id, land_sizes, layout, plate_id, hx, hy)
    area = layout.hex_area_km2
    b_over = _overlaps(block_id, land_id, len(blocks), len(lands))
    l_over = _overlaps(land_id, block_id, len(lands), len(blocks))
    block_names = _names(config.seed, "continents/v2/block_names", len(blocks))
    land_names = _names(config.seed, "sea_level/v2/landmass_names", len(lands))
    for b, over, name in zip(blocks, b_over, block_names, strict=True):
        land_hexes = sum(n for k, n in over.items() if k >= 0)
        b["name"] = name
        b["land_area_km2"] = float(land_hexes * area)
        b["submerged_area_km2"] = float(over.get(-1, 0) * area)
        b["landmass_ids"] = [
            k
            for k, _ in sorted(
                ((k, n) for k, n in over.items() if k >= 0), key=lambda kv: (-kv[1], kv[0])
            )[:12]
        ]
    if lands:
        index = np.arange(len(lands)) + 1
        above = np.maximum(elev - np.float32(sea_level_m), 0).astype(np.float64)
        mean_h = ndimage.mean(above, land_id + 1, index)
        max_h = ndimage.maximum(above, land_id + 1, index)
    for i, (m, over, name) in enumerate(zip(lands, l_over, land_names, strict=True)):
        m["name"] = name
        m["is_major"] = bool(m["area_km2"] >= config.continents.major_area_km2)
        m["mean_height_m"] = float(mean_h[i])
        m["max_height_m"] = float(max_h[i])
        m["on_oceanic_crust_share"] = round(over.get(-1, 0) / m["hexes"], 4)
        m["block_ids"] = [
            k
            for k, _ in sorted(
                ((k, n) for k, n in over.items() if k >= 0), key=lambda kv: (-kv[1], kv[0])
            )[:12]
        ]
    return blocks, lands


def run_v2(
    config: Config,
    geometry: WorldGeometry,
    v1_keys: dict[str, str],
    plate_model,
    crust_model,
    motion,
    plate_id: np.ndarray,
    old_keys: dict[str, str],
    old_cache: dict,
    old_geo: GeoData | None,
    report,
):
    """Run the v2 stages; returns ``(GeoData, cache_entries, reused_stage_names)``."""
    layout = geometry.layout
    keys = v2_stage_keys(config, v1_keys, layout)
    timings: dict[str, float] = {}
    reused: list[str] = []
    cache: dict = {}

    def cached(stage):
        if old_keys.get(stage) == keys[stage] and stage in old_cache:
            return old_cache[stage]
        return None

    relief_entry = cached("macro_elevation")
    raster_entry = cached("v2_raster")
    need_models = (
        raster_entry is None
        or relief_entry is None
        or (raster_entry is None and relief_entry.get("model") is None)
    )
    if raster_entry is None and relief_entry is not None and relief_entry.get("model") is None:
        relief_entry = None  # loaded from disk without models: rebuild deterministically

    if need_models and relief_entry is None:
        if plate_model is None or crust_model is None:
            raise RuntimeError(
                "The v2.0 stage needs the v1.0 world models, which could not be reproduced "
                "for this project (saved by an incompatible generator version)."
            )
        report("world_tectonics", 0.80)
        t0 = time.perf_counter()
        skeleton = cached("world_tectonics")
        if skeleton is None:
            skeleton = build_skeleton(
                geometry, plate_model, motion, config.boundaries, config.continents.skeleton_cells
            )
        else:
            reused.append("world_tectonics")
        timings["world_tectonics"] = round(time.perf_counter() - t0, 4)

        report("continental_blocks", 0.84)
        t0 = time.perf_counter()
        blocks_entry = cached("continental_blocks")
        if blocks_entry is None:
            cont_model, world_c = build_continents(config, crust_model, skeleton)
            blocks_entry = {"model": cont_model, "world_c": world_c}
        else:
            reused.append("continental_blocks")
        cont_model, world_c = blocks_entry["model"], blocks_entry["world_c"]
        timings["continental_blocks"] = round(time.perf_counter() - t0, 4)

        report("macro_elevation", 0.88)
        t0 = time.perf_counter()
        relief_model = build_relief(config, cont_model, skeleton)
        world_elev = relief_model.elevation(skeleton.x, skeleton.y, continentality=world_c)
        relief_entry = {
            "model": relief_model,
            "world_elevation_m": world_elev.astype(np.float32),
            "world_continentality_km": world_c.astype(np.float32),
            "grid": skeleton.grid_dict(),
            "structure": cont_model.describe(),
            "structure_warnings": list(cont_model.warnings),
        }
        cache["world_tectonics"] = skeleton
        cache["continental_blocks"] = blocks_entry
        timings["macro_elevation"] = round(time.perf_counter() - t0, 4)
    else:
        reused.extend(["world_tectonics", "continental_blocks", "macro_elevation"])
        for stage in ("world_tectonics", "continental_blocks"):
            if stage in old_cache and old_keys.get(stage) == keys[stage]:
                cache[stage] = old_cache[stage]
    cache["macro_elevation"] = relief_entry

    report("v2_raster", 0.91)
    t0 = time.perf_counter()
    if raster_entry is None:
        relief_model = relief_entry["model"]
        hx, hy = layout.centers()
        hex_c = relief_model.continents.continentality(hx, hy)
        elev, tect = relief_model.elevation(hx, hy, continentality=hex_c, parts=True)
        fraction = relief_model.continents.fraction_of(hex_c).astype(np.float32)
        block_id, block_sizes = label_mask(fraction >= 0.5, layout)
        raster_entry = {
            "continental_fraction": fraction,
            "continental_margin_km": hex_c.astype(np.float32),
            "continental_block_id": block_id,
            "block_sizes": block_sizes,
            "macro_elevation_m": elev.astype(np.float32),
            "tectonic_relief_m": tect.astype(np.float32),
        }
    else:
        reused.append("v2_raster")
    cache["v2_raster"] = raster_entry
    timings["v2_raster"] = round(time.perf_counter() - t0, 4)

    report("sea_level_landmasses", 0.96)
    t0 = time.perf_counter()
    grid = relief_entry["grid"]
    xs = grid["x0_km"] + np.arange(grid["nx"]) * grid["cell_km"]
    ys = grid["y0_km"] + np.arange(grid["ny"]) * grid["cell_km"]
    gx, gy = np.meshgrid(xs, ys)
    in_sim = geometry.simulation_bounds.contains(gx, gy)
    world_elev = relief_entry["world_elevation_m"]
    sea_level = resolve_sea_level(world_elev[in_sim], config.sea)
    elev = raster_entry["macro_elevation_m"]
    fraction = raster_entry["continental_fraction"]
    land, landmass_id, land_sizes, shelf = derive_land(
        elev, fraction, sea_level, layout, config.sea
    )
    layers = {
        "continental_fraction": fraction,
        "continental_margin_km": raster_entry["continental_margin_km"],
        "continental_block_id": raster_entry["continental_block_id"],
        "macro_elevation_m": elev,
        "tectonic_relief_m": raster_entry["tectonic_relief_m"],
        "land_mask": land,
        "landmass_id": landmass_id,
        "shelf_mask": shelf,
    }
    blocks, lands = derive_geo_tables(
        config, layout, plate_id, layers, sea_level, raster_entry["block_sizes"], land_sizes
    )
    world_stats = world_landmasses(
        world_elev, in_sim, grid["cell_km"], sea_level, config.continents.major_area_km2
    )
    world_c = relief_entry["world_continentality_km"]
    summary = {
        "sea_mode": config.sea.sea_mode,
        "sea_level_m": sea_level,
        "vertical_datum": "0 m = model datum; sea level is a scenario parameter",
        "target_land_fraction": (
            config.sea.target_land_fraction
            if config.sea.sea_mode == "target_land_fraction"
            else None
        ),
        "land_fraction_viewport": float(land.mean()),
        "shelf_fraction_viewport": float(shelf.mean()),
        "continental_crust_fraction_viewport": float((fraction >= 0.5).mean()),
        "continental_crust_fraction_simulation": float((world_c[in_sim] > 0).mean()),
        "blocks_in_window": len(blocks),
        "landmasses_in_window": len(lands),
        "major_landmasses_in_window": int(sum(1 for m in lands if m["is_major"])),
        "min_elevation_m": float(elev.min()),
        "max_elevation_m": float(elev.max()),
        "world": world_stats,
        "structure": relief_entry["structure"],
        "continent_profile": config.continents.continent_profile,
        "requested_major_landmass_count": config.continents.major_landmass_count,
    }
    warnings = list(relief_entry.get("structure_warnings", []))
    requested = config.continents.major_landmass_count
    actual = world_stats["major_landmass_count"]
    if requested != "auto" and actual != requested:
        warnings.append(
            f"Requested {requested} major landmass(es), the simulation area has {actual} at "
            f"sea level {sea_level:+.0f} m (the count is a goal, not a guarantee)."
        )
    if config.sea.sea_mode == "target_land_fraction":
        if abs(sea_level) > 700.0:
            warnings.append(
                f"target_land_fraction={config.sea.target_land_fraction:g} needs an unusual "
                f"sea level of {sea_level:+.0f} m for this crust."
            )
        diff = abs(float(land.mean()) - config.sea.target_land_fraction)
        if diff > 0.12 and geometry.viewport_bounds.area < 0.9 * geometry.simulation_bounds.area:
            warnings.append(
                f"The window shows {100 * float(land.mean()):.0f} % land; the target of "
                f"{100 * config.sea.target_land_fraction:.0f} % applies to the whole "
                "simulation area."
            )
    timings["sea_level_landmasses"] = round(time.perf_counter() - t0, 4)

    geo = GeoData(
        layers=layers,
        sea_level_m=sea_level,
        blocks=blocks,
        landmasses=lands,
        summary=summary,
        stage_keys=keys,
        world={
            "grid": grid,
            "elevation_m": world_elev,
            "continentality_km": world_c,
        },
        warnings=warnings,
        timings=timings,
    )
    return geo, cache, reused
