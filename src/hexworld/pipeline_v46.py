"""v4.6 stages: one solid surface for massifs, basins and the sea floor (TZ 4.6, section 5).

    v3 epoch surface  [read only]
      -> v46_terrain  marine basement (age-depth, ridges, fracture zones, trenches, margins,
                      plateaus, pelagic sediment) + continental basins (rifts, pull-aparts,
                      foredeeps, sags; bedrock subsidence and sediment fill apart)
                      = surface_elevation_m (input of v4) + ocean connectivity
      -> v4           climate, erosion, rivers and lakes on that surface
      -> v46_finish   final water masks, basin hydrology (z_min, spill, lake level), massif
                      graphs (ridges, passes, peaks with prominence, valleys from the rivers),
                      sea-floor classification - derived, no new geology

The terrain stage depends on the v3 raster and the terrain parameters only; the sea level
changes only the water classification (and v4, whose base level it is).
"""

from __future__ import annotations

import copy
import hashlib
import json
import time
from dataclasses import asdict

import numpy as np
from scipy.spatial import cKDTree

from .geomorph import basins as basin_mod
from .geomorph.bathymetry import compose_sea_floor
from .geomorph.ocean_age import baseline_age_scale, build_ridge_pieces
from .geomorph.surface_water import ocean_mask, sources_from_world
from .project import TerrainData
from .stages.hydrology import hex_graph
from .version import TERRAIN_ALGORITHM_VERSION


def _key(*parts) -> str:
    return hashlib.sha256(json.dumps(parts, sort_keys=True, default=str).encode()).hexdigest()


def _plain(obj):
    """JSON-ready copy: numpy scalars and arrays become Python numbers and lists."""
    if isinstance(obj, dict):
        return {k: _plain(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_plain(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.generic):
        return obj.item()
    return obj


def terrain_stage_keys(config, hist_keys: dict) -> dict:
    base = _key(
        "v46_terrain",
        hist_keys.get("v3_raster"),
        asdict(config.terrain),
        TERRAIN_ALGORITHM_VERSION,
    )
    return {"v46_terrain": base}


REQUIRED_V3 = (
    "material_x_km",
    "material_y_km",
    "continentality_km",
    "base_component_m",
    "trench_relief_m",
    "trench_distance_km",
    "trench_side",
    "trench_rate_mm_yr",
    "spreading_axis_distance_km",
)


def compose_terrain(config, geometry, project, report=None) -> dict:
    """Pure computation of the v4.6 surface from the v3 epoch (no project mutation)."""
    from .pipeline_v3 import ensure_v2_models

    hist = project.hist
    layout = geometry.layout
    params = config.terrain
    missing = [k for k in REQUIRED_V3 if k not in hist.layers]
    if missing:
        raise RuntimeError(
            "v4.6 needs v3 layers this project does not have (saved by generator < 4.6): "
            + ", ".join(missing)
            + ". Regenerate the v3 stage."
        )
    timings = {}
    t0 = time.perf_counter()
    skeleton, relief_model = ensure_v2_models(
        config, geometry, project.plate_model, project.crust_model, project.motion, project.cache
    )
    pieces = build_ridge_pieces(skeleton, config.seed, params.ridge_piece_length_km)
    scale = baseline_age_scale(
        skeleton, pieces, relief_model.continents, params.target_median_age_ma
    )
    sea = compose_sea_floor(
        hist.layers,
        layout,
        hist.epoch_time_ma,
        pieces,
        relief_model,
        params,
        config.seed,
        hist.volcanoes,
        age_scale=scale,
    )
    sea["summary"]["baseline_age_scale"] = round(float(scale), 4)
    timings["v46_sea_floor"] = round(time.perf_counter() - t0, 4)
    if report:
        report(0.4)

    # -- continental basins ----------------------------------------------------------------
    t0 = time.perf_counter()
    shape = (layout.height, layout.width)
    hx, hy = (a.ravel() for a in layout.centers())
    cf = hist.layers["continental_fraction"]
    marine = sea["layers"]["marine_floor_mask"]
    continental = (cf >= 0.5) & ~marine
    in_mountains = hist.layers["mountain_system_id"] >= 0
    plans: list[dict] = []
    if "rift_graben" in params.basins or "pull_apart" in params.basins:
        boundary_plans = basin_mod.plan_boundary_basins(
            layout,
            hist.layers["plate_id_epoch"],
            continental,
            project.motion,
            config.seed,
            layout.spacing_km,
        )
        for p in boundary_plans:
            if p["basin_type"] == "rift_graben" and "rift_graben" in params.basins:
                plans.append(p)
            if p["basin_type"] == "strike_slip_pull_apart" and "pull_apart" in params.basins:
                plans.append(p)
    if "intracratonic_sag" in params.basins:
        c_sk = relief_model.continents.continentality(skeleton.x, skeleton.y)
        if skeleton.bx.size:
            bd, _ = cKDTree(np.column_stack([skeleton.bx, skeleton.by])).query(
                np.column_stack([skeleton.x.ravel(), skeleton.y.ravel()])
            )
            bd = bd.reshape(skeleton.x.shape)
        else:
            bd = np.full(skeleton.x.shape, 1e9)
        sags = basin_mod.plan_sags(
            c_sk, skeleton.x, skeleton.y, bd, config.seed, params.sag_density, skeleton.cell_km
        )
        plans.extend(sags)
    mx = hist.layers["material_x_km"].ravel().astype(np.float64)
    my = hist.layers["material_y_km"].ravel().astype(np.float64)
    sub, fill, flank, which = basin_mod.evaluate(plans, hx, hy, mx, my, layout.radius)
    cont_flat = continental.ravel()
    sub = np.where(cont_flat, sub, 0.0)
    fill = np.where(cont_flat, fill, 0.0)
    flank = np.where(cont_flat & ~in_mountains.ravel(), flank, 0.0)
    which = np.where(cont_flat, which, -1)

    fore = np.zeros(sub.shape)
    alpha = None
    if "foreland_foredeep" in params.basins:
        w, alpha = basin_mod.foreland_field(hist.world, params, layout, in_mountains)
        w = np.where(cont_flat, w.ravel(), 0.0)
        fore = w
        deeper = w > sub
        fill = np.where(deeper & (w > 0), w * params.foreland_fill, fill)
        sub = np.where(deeper & (w > 0), w, sub)
        fore_id = len(plans)
        if (w > 30.0).any():
            plans.append(
                {
                    "basin_type": "foreland_foredeep",
                    "elastic_thickness_km": params.elastic_thickness_km,
                    "flexural_parameter_km": round(float(alpha), 1),
                    "fill_fraction": params.foreland_fill,
                    "cause": "flexure of the plate under the mountain load of the epoch",
                    "frame": "epoch (mantle)",
                }
            )
            which = np.where(deeper & (w > 30.0), fore_id, which)
    # forebulge (w < 0): a low rise, half of it eroded / never built
    bulge = np.where(fore < 0.0, -0.5 * fore, 0.0)
    surface_delta_basins = -(sub - fill) + flank + bulge
    bedrock_delta_basins = -sub + flank + bulge
    timings["v46_basins"] = round(time.perf_counter() - t0, 4)
    if report:
        report(0.75)

    # -- one surface -----------------------------------------------------------------------
    z3 = hist.layers["elevation_m"].astype(np.float64)
    z_marine = sea["surface"]
    surface = z_marine + surface_delta_basins.reshape(shape)
    sed3 = np.clip(hist.layers["sediment_thickness_m"].astype(np.float64), 0.0, None)
    marine_sed = sea["layers"]["marine_sediment_thickness_m"].astype(np.float64)
    sediment = np.where(marine, marine_sed, sed3 + fill.reshape(shape))
    bedrock = surface - sediment
    layers = dict(sea["layers"])
    layers.update(
        {
            "surface_elevation_m": surface.astype(np.float32),
            "bedrock_elevation_m": bedrock.astype(np.float32),
            "sediment_thickness_m": sediment.astype(np.float32),
            "basin_subsidence_m": sub.reshape(shape).astype(np.float32),
            "basin_fill_m": fill.reshape(shape).astype(np.float32),
            "basin_id": which.reshape(shape).astype(np.int32),
            "basin_delta_m": surface_delta_basins.reshape(shape).astype(np.float32),
            "bedrock_basin_delta_m": bedrock_delta_basins.reshape(shape).astype(np.float32),
            "foreland_deflection_m": fore.reshape(shape).astype(np.float32),
            "terrain_delta_m": (surface - z3).astype(np.float32),
        }
    )
    plans = [_plain(p) for p in plans]
    for i, p in enumerate(plans):
        p["basin_id"] = i
        p["source_version"] = TERRAIN_ALGORITHM_VERSION
        inside = which == i
        p["hexes"] = int(inside.sum())
        if inside.any():
            p["max_bedrock_subsidence_m"] = round(float(sub[inside].max()), 1)
            p["fill_volume_km3"] = round(
                float(fill[inside].sum() * layout.hex_area_km2 / 1000.0), 1
            )
            p["min_surface_m"] = round(float(surface.ravel()[inside].min()), 1)
    summary = {
        "sea_floor": sea["summary"],
        "basins": {
            "planned": {
                t: sum(1 for p in plans if p["basin_type"] == t) for t in basin_mod.BASIN_TYPES[1:]
            },
            "in_window": {
                t: sum(1 for p in plans if p["basin_type"] == t and p.get("hexes", 0) > 0)
                for t in basin_mod.BASIN_TYPES[1:]
            },
            "fill_volume_km3": round(float(fill.sum() * layout.hex_area_km2 / 1000.0), 1),
            "flexural_parameter_km": None if alpha is None else round(float(alpha), 1),
            "fill_note": "fill assumed from long-term denudation around the basins [А]; "
            "not taken from the v3/v4 erosion budget",
        },
    }
    return {
        "layers": layers,
        "plans": {"basins": plans, "ridge_pieces": pieces.count},
        "summary": _plain(summary),
        "timings": timings,
    }


def run_terrain(config, geometry, project, old_keys: dict, old_cache: dict, report):
    """The v4.6 terrain stage (before v4). Returns ``(TerrainData, cache, reused)``."""
    hist = project.hist
    keys = terrain_stage_keys(config, hist.stage_keys)
    reused: list[str] = []
    cache: dict = {}
    t0 = time.perf_counter()
    report("v46_terrain", 0.966)
    entry = None
    if old_keys.get("v46_terrain") == keys["v46_terrain"] and "v46_terrain" in old_cache:
        entry = old_cache["v46_terrain"]
        reused.append("v46_terrain")
    if entry is None:
        entry = compose_terrain(
            config, geometry, project, report=lambda f: report("v46_terrain", 0.966 + 0.01 * f)
        )
    cache["v46_terrain"] = entry
    layers = dict(entry["layers"])
    # -- water of this surface (depends on the sea level: cheap, always recomputed) ------------
    layout = geometry.layout
    z = layers["surface_elevation_m"].astype(np.float64)
    sea = float(hist.sea_level_m)
    graph = hex_graph(layout)
    sources = sources_from_world(layout, hist.world, z, sea, config.terrain.min_ocean_area_km2)
    ocean = ocean_mask(z, sea, graph.nbr, sources)
    layers["ocean_source_mask"] = sources.astype(np.bool_)
    layers["ocean_mask_pre_v4"] = ocean.astype(np.bool_)
    timings = dict(entry["timings"])
    timings["v46_terrain"] = round(time.perf_counter() - t0, 4)
    terrain = TerrainData(
        layers=layers,
        plans=copy.deepcopy(entry["plans"]),  # finish_terrain adds to them: never share
        summary=dict(entry["summary"]),
        stage_keys=keys,
        timings=timings,
    )
    terrain.summary["below_sea_not_ocean_hexes"] = int(((z <= sea) & ~ocean).sum())
    return terrain, cache, reused


def _segment_cells(layout, seg) -> np.ndarray:
    """Hex indices of a stored river segment (from its x/y polyline)."""
    cols, rows = layout.world_to_offset(np.asarray(seg["x_km"]), np.asarray(seg["y_km"]))
    cols = np.clip(np.asarray(cols, dtype=np.int64), 0, layout.width - 1)
    rows = np.clip(np.asarray(rows, dtype=np.int64), 0, layout.height - 1)
    return rows * layout.width + cols


def finish_terrain(config, project) -> None:
    """v4.6 after v4: canyons / fans, basin water, massif graphs, sea-floor provinces.

    Changes only sea hexes of the final surface (canyon incision, fan deposition); the land
    surface, rivers, lakes and climate of v4 stay as they are.
    """
    from .geomorph.canyons import submarine_sediment
    from .geomorph.classify import classify_sea_floor, shares
    from .geomorph.massifs import dissection_index, massif_plans
    from .geomorph.surface_water import water_layers
    from .stages.hydrology import route

    t0 = time.perf_counter()
    terrain = project.terrain
    hydro = project.hydro
    hist = project.hist
    layout = project.geometry.layout
    L = hydro.layers
    T = terrain.layers
    graph = hex_graph(layout)
    sea_level = float(hydro.sea_level_m)
    z = np.asarray(L["elevation_m"], dtype=np.float64)
    ocean = np.asarray(L["ocean_mask"], dtype=bool)

    # -- 1. submarine canyons and fans from the delivered river sediment -------------------
    if config.terrain.submarine_canyons:
        res = submarine_sediment(
            z,
            ocean,
            L["river_mouth"],
            L["sediment_delivery_m3_yr"],
            graph.nbr,
            layout,
            duration_yr=max(config.erosion.duration_ma, 1e-6) * 1e6,
        )
        dz = np.where(ocean, res["fan_m"] - res["incision_m"], 0.0)
        z_new = np.where(ocean, np.minimum(z + dz, sea_level - 1.0), z)
        z32 = z_new.astype(np.float32)
        L["elevation_m"] = z32
        L["post_erosion_elevation_m"] = z32
        pre = np.asarray(L["pre_erosion_elevation_m"], dtype=np.float64)
        L["v4_delta_elevation_m"] = (z32.astype(np.float64) - pre).astype(np.float32)
        L.update(
            {
                k: (v.astype(np.float32) if v.dtype == np.float64 else v)
                for k, v in water_layers(z_new, sea_level, ocean, L["lake_level_m"]).items()
            }
        )
        T["submarine_canyon_mask"] = (res["canyon_id"] >= 0).astype(np.bool_)
        T["canyon_depth_m"] = res["canyon_depth_m"].astype(np.float32)
        T["canyon_incision_m"] = res["incision_m"].astype(np.float32)
        T["submarine_fan_m"] = res["fan_m"].astype(np.float32)
        # float64: the evolved surface is restored exactly as post-erosion minus this
        T["canyon_fan_delta_m"] = z32.astype(np.float64) - z
        terrain.plans["canyons"] = _plain(res["rows"])
        terrain.summary["submarine_sediment"] = _plain(res["budget"]) | {
            "canyons": sum(1 for r in res["rows"] if r["is_canyon"]),
            "fans": len(res["rows"]),
        }
        z = z_new

    # -- 2. water of every basin: lowest point, spill sill, lake level --------------------
    routing = route(z.ravel(), (ocean.ravel() | graph.border), graph)
    spill = routing.routing_m.reshape(z.shape)
    basin_id = T["basin_id"]
    lake_level = L["lake_level_m"]
    below_dry = L["below_sea_level_land_mask"]
    for p in terrain.plans.get("basins", []):
        inside = basin_id == p["basin_id"]
        dry_part = inside & ~ocean
        if not dry_part.any():
            if inside.any():
                p["water"] = {"ocean_hexes": int((inside & ocean).sum()), "flooded_by_sea": True}
            continue
        k = np.argmin(np.where(dry_part, z, np.inf))
        r, c = np.unravel_index(k, z.shape)
        lv = lake_level[r, c]
        p["water"] = {
            "z_min_m": round(float(z[r, c]), 1),
            "z_spill_m": round(float(spill[r, c]), 1),
            "closed_depth_m": round(float(spill[r, c] - z[r, c]), 1),
            "lake_level_m": round(float(lv), 1) if np.isfinite(lv) else None,
            "lake_hexes": int((inside & L["inland_water_mask"]).sum()),
            "below_sea_dry_hexes": int((inside & below_dry).sum()),
            "ocean_hexes": int((inside & ocean).sum()),
            "endorheic_share": round(float(L["endorheic_mask"][inside].mean()), 3),
        }
    T["spill_level_m"] = spill.astype(np.float32)

    # -- 3. massif graphs, prominence, dissection index -----------------------------------
    if config.terrain.massif_graphs and hist.plan.get("systems"):
        rivers = []
        for seg in hydro.rivers:
            rivers.append(dict(seg, cells=_segment_cells(layout, seg)))
        plans, prom = massif_plans(
            project,
            z,
            L["slope_m_per_km"],
            hist.layers["local_relief_m"],
            L["river_mask"],
            L["flow_receiver"],
            rivers,
            graph.nbr,
        )
        terrain.plans["massifs"] = _plain(plans)
        T["prominence_m"] = prom.reshape(z.shape).astype(np.float32)
        T["dissection_index"] = dissection_index(
            L["flow_accumulation_area_km2"],
            hist.layers["orogen_age_ma"],
            hist.layers["rock_resistance"],
            L["runoff_mm_yr"],
            hist.layers["mountain_system_id"] >= 0,
        ).astype(np.float32)
        q = [m["quality"] for m in plans]
        terrain.summary["massifs"] = {
            "count": len(plans),
            "passes_below_peaks": sum(v["passes_below_peaks"] for v in q),
            "passes_total": sum(v["passes_total"] for v in q),
            "rivers_on_crest_hexes": sum(v["rivers_on_crest_hexes"] for v in q),
            "styles": sorted({str(m.get("style")) for m in plans}),
        }

    # -- 4. sea-floor provinces of the final surface --------------------------------------
    codes = classify_sea_floor(T, hist.layers, L)
    T["marine_province"] = codes
    terrain.summary["marine_province_shares"] = shares(codes)
    terrain.summary["water"] = {
        "ocean_hexes": int(ocean.sum()),
        "lake_hexes": int(L["inland_water_mask"].sum()),
        "below_sea_level_dry_hexes": int(below_dry.sum()),
        "above_datum_fraction": float((z > sea_level).mean()),
        "non_ocean_fraction": float((~ocean).mean()),
        "exposed_fraction": float(L["dry_land_mask"].mean()),
    }
    terrain.timings["v46_finish"] = round(time.perf_counter() - t0, 4)
