"""v4.0 stages after the v3 epoch surface::

    v3 epoch surface + rocks + uplift + sea level
      -> v4_climate      world raster climate (+ external inflow estimate)
      -> v4_evolution    coupled climate / routing / erosion / sediment loop
      -> v4_network      final climate, hydrology, lakes, rivers on the final surface
      -> land / sea      of the final surface

Keys: changing the climate re-runs everything of v4 but nothing older; changing only the
river threshold re-runs only the network; the sea level is the base level of rivers and
erosion, so it re-runs v4 (but never v1-v3).
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import asdict

import numpy as np

from .pipeline_v2 import _component_table, _names
from .project import HydroData
from .stages import raster as raster_stage
from .stages.climate import (
    SECONDS_PER_YEAR,
    aridity_class,
    budyko_aet,
    hex_climate,
    potential_et,
    resolve_latitude,
    world_climate,
)
from .stages.coasts import coast_fields
from .stages.hydrology import (
    accumulate,
    grid_graph,
    hex_graph,
    hydraulic_geometry,
    hydrology,
    river_network,
    river_systems,
    route,
    routing_choice,
)
from .stages.landscape import evolve
from .stages.sea import label_mask
from .version import CLIMATE_ALGORITHM_VERSION, HYDRO_ALGORITHM_VERSION


def _key(*parts) -> str:
    return hashlib.sha256(json.dumps(parts, sort_keys=True, default=str).encode()).hexdigest()


_EVOLVED_FIELDS = (
    "elevation",
    "erosion_total_m",
    "deposition_total_m",
    "sediment_thickness_m",
    "sediment_flux_m3_yr",
    "sediment_delivery_m3_yr",
)


def v4_stage_keys(config, hist_keys: dict, sea_level_m: float) -> dict:
    climate = _key(
        "v4_climate",
        CLIMATE_ALGORITHM_VERSION,
        hist_keys["v3_raster"],
        round(float(sea_level_m), 6),
        asdict(config.climate),
        # the climate entry also carries the inflow and the edge outlets of the window
        config.hydrology.external_inflow,
        config.hydrology.edge_outlets,
    )
    hydro = asdict(config.hydrology)
    threshold = tuple(
        hydro.pop(k)
        for k in (
            "river_min_discharge_m3_s",
            "river_min_upstream_hexes",
            "river_min_system_hexes",
        )
    )
    evolution = _key(
        "v4_evolution", HYDRO_ALGORITHM_VERSION, climate, asdict(config.erosion), hydro
    )
    network = _key("v4_network", evolution, threshold)
    return {"v4_climate": climate, "v4_evolution": evolution, "v4_network": network}


def _world_drainage(config, wc, world_elev, sea_level_m, viewport):
    """Coarse 8-neighbour routing of the whole model area with the world climate's runoff."""
    grid = wc.grid
    ny, nx = world_elev.shape
    cell_m = grid["cell_km"] * 1000.0
    h = np.asarray(world_elev, dtype=np.float64) - sea_level_m
    sea = h <= 0.0
    t = wc.t_sea_level - config.climate.lapse_rate_c_per_km * np.maximum(h, 0.0) / 1000.0
    runoff = np.where(sea, 0.0, wc.precipitation - budyko_aet(wc.precipitation, potential_et(t)))
    runoff *= 1.0 - config.climate.deep_infiltration
    graph = grid_graph(ny, nx, cell_m)
    routing = route(h.ravel(), sea.ravel() | graph.border, graph)
    acc = accumulate(routing, (runoff / 1000.0 * cell_m * cell_m).ravel())
    xs = grid["x0_km"] + np.arange(nx) * grid["cell_km"]
    ys = grid["y0_km"] + np.arange(ny) * grid["cell_km"]
    gx, gy = np.meshgrid(xs, ys)
    gx, gy = gx.ravel(), gy.ravel()
    inside = viewport.contains(gx, gy)
    return routing, acc, sea.ravel(), gx, gy, inside


def external_inflow(config, wc, world_elev, sea_level_m, layout, viewport, drainage=None):
    """Water entering the window from outside, from a coarse 8-neighbour world routing."""
    routing, acc, sea, gx, gy, inside = drainage or _world_drainage(
        config, wc, world_elev, sea_level_m, viewport
    )
    rc = routing.receiver
    entering = ~inside & inside[rc] & (rc != np.arange(rc.size)) & ~sea
    inflow = np.zeros(layout.width * layout.height)
    cells = np.nonzero(entering)[0]
    if cells.size:
        col, row = layout.world_to_offset(gx[rc[cells]], gy[rc[cells]])
        ok = layout.inside(col, row)
        np.add.at(inflow, (row[ok] * layout.width + col[ok]), acc[cells[ok]] / SECONDS_PER_YEAR)
    return inflow, {"entry_points": int(cells.size), "total_m3_s": round(float(inflow.sum()), 3)}


def edge_outlets(wc, layout, viewport, drainage, graph=None, simulation_bounds=None):
    """Border hexes through which water may leave the window (manual 6.2, item 4).

    A cut-out window is not surrounded by sinks: water leaves it only where the drainage of
    the whole model area crosses the window edge outward. Each such crossing of the coarse
    world routing marks the border hexes within ~one world cell of it; every other border
    hex drains into the window like an interior hex (its water flows on to the sea, a lake
    or one of the real exits). Where the window edge is the edge of the model area itself,
    the world ends there and every border hex of that side stays an outlet.
    """
    routing, _acc, sea, gx, gy, inside = drainage
    rc = routing.receiver
    leaving = inside & ~inside[rc] & (rc != np.arange(rc.size)) & ~sea
    n = layout.width * layout.height
    allowed = np.zeros(n, dtype=bool)
    b = layout.bounds
    hx, hy = layout.centers()
    hx, hy = hx.ravel(), hy.ravel()
    border = np.nonzero(graph.border if graph is not None else hex_graph(layout).border)[0]
    if simulation_bounds is not None:
        sb, tol, near = simulation_bounds, float(wc.grid["cell_km"]), layout.spacing_km
        for edge_w, edge_s, coord in (
            (b.xmin, sb.xmin, hx),
            (b.xmax, sb.xmax, hx),
            (b.ymin, sb.ymin, hy),
            (b.ymax, sb.ymax, hy),
        ):
            if abs(edge_w - edge_s) <= tol:
                allowed[border[np.abs(coord[border] - edge_w) <= near]] = True
    cells = np.nonzero(leaving)[0]
    if not cells.size:
        return allowed, {"exits": 0, "edge_outlet_hexes": int(allowed.sum())}
    # where the step from the inside cell to its outside receiver crosses the edge
    x0, y0, x1, y1 = gx[cells], gy[cells], gx[rc[cells]], gy[rc[cells]]
    with np.errstate(divide="ignore", invalid="ignore"):
        tx = np.where(
            x1 > b.xmax,
            (b.xmax - x0) / (x1 - x0),
            np.where(x1 < b.xmin, (b.xmin - x0) / (x1 - x0), 1.0),
        )
        ty = np.where(
            y1 > b.ymax,
            (b.ymax - y0) / (y1 - y0),
            np.where(y1 < b.ymin, (b.ymin - y0) / (y1 - y0), 1.0),
        )
    tc = np.clip(np.nan_to_num(np.minimum(tx, ty), nan=0.0), 0.0, 1.0)
    cx = np.clip(x0 + tc * (x1 - x0), b.xmin, b.xmax)
    cy = np.clip(y0 + tc * (y1 - y0), b.ymin, b.ymax)
    reach = 0.75 * max(float(wc.grid["cell_km"]), layout.spacing_km)
    for x, y in zip(cx.tolist(), cy.tolist(), strict=True):
        near = (hx[border] - x) ** 2 + (hy[border] - y) ** 2 <= reach * reach
        allowed[border[near]] = True
    return allowed, {"exits": int(cells.size), "edge_outlet_hexes": int(allowed.sum())}


def _slope(z, layout):
    flat = z.ravel()
    slope = np.zeros(flat.size)
    for d in range(6):
        nb, valid = raster_stage.neighbor_index(layout, d)
        drop = np.where(valid.ravel(), np.abs(flat - flat[nb.ravel()]), 0.0)
        slope = np.maximum(slope, drop / layout.spacing_km)
    return slope


def run_v4(config, geometry, project, old_keys: dict, old_cache: dict, report):
    hist = project.hist
    layout = geometry.layout
    shape = (layout.height, layout.width)
    sea_level = float(hist.sea_level_m)
    keys = v4_stage_keys(config, hist.stage_keys, sea_level)
    timings: dict[str, float] = {}
    reused: list[str] = []
    cache: dict = {}

    def cached(stage):
        if old_keys.get(stage) == keys[stage] and stage in old_cache:
            reused.append(stage)
            return old_cache[stage]
        return None

    # -- A. world climate + inflow from outside the window ---------------------------------
    t0 = time.perf_counter()
    report("v4_climate", 0.986)
    climate_entry = cached("v4_climate")
    if climate_entry is None:
        latitude = resolve_latitude(config.climate, geometry.simulation_bounds)
        wc = world_climate(
            config.climate, hist.world["grid"], hist.world["elevation_m"], sea_level, latitude
        )
        drainage = _world_drainage(
            config, wc, hist.world["elevation_m"], sea_level, geometry.viewport_bounds
        )
        inflow, inflow_info = (
            external_inflow(
                config,
                wc,
                hist.world["elevation_m"],
                sea_level,
                layout,
                geometry.viewport_bounds,
                drainage,
            )
            if config.hydrology.external_inflow
            else (np.zeros(layout.width * layout.height), {})
        )
        if config.hydrology.edge_outlets == "world":
            edge, edge_info = edge_outlets(
                wc,
                layout,
                geometry.viewport_bounds,
                drainage,
                simulation_bounds=geometry.simulation_bounds,
            )
        else:
            edge, edge_info = None, {"exits": None, "edge_outlet_hexes": "all border hexes"}
        inflow_info = {**inflow_info, **edge_info}
        climate_entry = {
            "world": wc,
            "latitude": latitude,
            "inflow": inflow,
            "inflow_info": inflow_info,
            "edge_outlet": edge,
        }
    cache["v4_climate"] = climate_entry
    wc = climate_entry["world"]
    timings["v4_climate"] = round(time.perf_counter() - t0, 4)

    # -- B-E. coupled evolution ---------------------------------------------------------------
    t0 = time.perf_counter()
    report("v4_evolution", 0.988)
    graph = hex_graph(layout)
    z0 = hist.layers["elevation_m"].astype(np.float64)
    evo = cached("v4_evolution")
    if evo is None:
        evo = evolve(
            config,
            layout,
            graph,
            z0,
            sea_level,
            hist.layers["rock_resistance"],
            hist.layers["sediment_thickness_m"],
            hist.layers["uplift_rate_mm_yr"],
            wc,
            climate_entry["inflow"],
            report=lambda f: report("v4_evolution", 0.988 + 0.008 * f),
            edge_outlet=climate_entry.get("edge_outlet"),
        )
        # the stored layers are float32: round the fresh result the same way, so a project
        # loaded from disk and one generated now feed the identical surface to the network
        for name in _EVOLVED_FIELDS:
            evo[name] = np.asarray(evo[name], dtype=np.float32).astype(np.float64)
    cache["v4_evolution"] = evo
    timings["v4_evolution"] = round(time.perf_counter() - t0, 4)

    # -- F. final climate, hydrology and rivers on the final surface --------------------------
    t0 = time.perf_counter()
    report("v4_network", 0.996)
    z = np.asarray(evo["elevation"], dtype=np.float64).reshape(shape)
    net = cached("v4_network")
    if net is None:
        sea = (z <= sea_level).ravel()
        area = layout.hex_area_km2 * 1e6
        clim = hex_climate(config.climate, wc, layout, z, sea_level)
        hyd = hydrology(
            z.ravel(),
            sea,
            graph,
            area,
            clim["runoff_mm_yr"].ravel(),
            clim["precipitation_mm_yr"].ravel(),
            clim["potential_evapotranspiration_mm_yr"].ravel() * 1.1,
            climate_entry["inflow"],
            config.hydrology.min_lake_depth_m,
            config.hydrology.channel_loss,
            routing_choice(config.seed, z.size, config.hydrology.routing, layout),
            climate_entry.get("edge_outlet"),
        )
        # open water: recompute the climate fields of lake hexes with lakes as water
        lakes_mask = (hyd["lake_id"] >= 0).reshape(shape)
        clim = hex_climate(config.climate, wc, layout, z, sea_level, lakes_mask)
        slope = _slope(z, layout)
        river, strahler, segments, mouths = river_network(
            hyd,
            ~sea,
            layout,
            config.hydrology.river_min_discharge_m3_s,
            slope,
            config.hydrology.river_min_upstream_hexes * layout.hex_area_km2,
            config.hydrology.river_min_system_hexes * layout.hex_area_km2,
        )
        net = {
            "climate": clim,
            "hyd": hyd,
            "slope": slope,
            "river": river,
            "strahler": strahler,
            "segments": segments,
            "mouths": mouths,
            "systems": river_systems(segments),
        }
    cache["v4_network"] = net
    clim, hyd = net["climate"], net["hyd"]
    timings["v4_network"] = round(time.perf_counter() - t0, 4)

    t0 = time.perf_counter()
    rc = hyd["receiver"]
    n = rc.size
    direction = np.full(n, 255, dtype=np.uint8)
    for d in range(6):
        nb, valid = raster_stage.neighbor_index(layout, d)
        hit = valid.ravel() & (nb.ravel() == rc) & (rc != np.arange(n))
        direction[hit] = d
    q = hyd["discharge_m3_s"]
    sea = z <= sea_level
    mouths = net["mouths"]
    coastal = np.zeros(n)
    for seg in net["segments"]:
        if seg["mouth"] == "sea":
            coastal[seg["cells"][-1]] = q[seg["cells"][-1]]
    years = max(config.erosion.duration_ma, 1e-9)

    def g(a, dtype=np.float32):
        return np.ascontiguousarray(np.asarray(a).reshape(shape).astype(dtype))

    land = ~sea
    width_m, depth_m, _velocity = hydraulic_geometry(q)
    landmass_id, sizes = label_mask(land, layout)
    hx, hy = layout.centers()
    lands = _component_table(landmass_id, sizes, layout, hist.layers["plate_id_epoch"], hx, hy)
    for m, name in zip(
        lands, _names(config.seed, "sea_level/v4/landmass_names", len(lands)), strict=True
    ):
        m["name"] = name
        m["is_major"] = bool(m["area_km2"] >= config.continents.major_area_km2)
    shelf = (
        ~land
        & (hist.layers["continental_fraction"] >= 0.5)
        & (sea_level - z <= config.sea.shelf_max_depth_m)
    )
    z32 = g(z)
    layers = {
        "elevation_m": z32,
        "post_erosion_elevation_m": z32,
        "pre_erosion_elevation_m": hist.layers["elevation_m"],
        "v4_delta_elevation_m": g(z - z0),
        **{k: g(v, np.asarray(v).dtype) for k, v in clim.items()},
        "flow_receiver": g(rc, np.int64),
        "flow_direction": g(direction, np.uint8),
        "flow_accumulation_area_km2": g(hyd["area_km2"]),
        "discharge_m3_s": g(q),
        "external_inflow_m3_s": g(climate_entry["inflow"]),
        "basin_id": g(hyd["basin_id"], np.int32),
        "lake_id": g(hyd["lake_id"], np.int32),
        "lake_level_m": g(hyd["lake_level_m"]),
        "lake_depth_m": g(hyd["lake_depth_m"]),
        "endorheic_mask": g(hyd["endorheic_mask"], np.bool_),
        "channel_loss_m3_s": g(hyd["channel_loss_m3_s"]),
        "outflow_share": g(np.where(sea.ravel(), 1.0, hyd["outflow_share"])),
        "no_outflow_mask": g(~sea.ravel() & (hyd["outflow_share"] < 0.1), np.bool_),
        "river_mask": g(net["river"], np.bool_),
        "river_width_m": g(np.where(net["river"], width_m, 0.0)),
        "river_depth_m": g(np.where(net["river"], depth_m, 0.0)),
        "strahler_order": g(net["strahler"], np.uint8),
        "river_mouth": g(mouths, np.bool_),
        "coastal_discharge_m3_s": g(coastal),
        "sediment_delivery_m3_yr": g(evo["sediment_delivery_m3_yr"]),
        "erosion_rate_m_ma": g(evo["erosion_total_m"] / years),
        "v4_erosion_m": g(evo["erosion_total_m"]),
        "v4_deposition_m": g(evo["deposition_total_m"]),
        "sediment_thickness_m": g(evo["sediment_thickness_m"]),
        "sediment_flux_m3_yr": g(evo["sediment_flux_m3_yr"]),
        "slope_m_per_km": g(net["slope"]),
        "land_mask": land.astype(np.bool_),
        "landmass_id": landmass_id,
        "shelf_mask": shelf.astype(np.bool_),
    }
    # coast types from the final surface, climate, rivers, rock and tectonics
    coast_layers, coast_summary = coast_fields(
        layout, layers, hist.layers, hist.plan.get("ridges", []), sea_level, 0
    )
    layers.update(coast_layers)
    # river segments: keep the geometry, drop the cell lists from the public table
    rivers = [
        {k: v for k, v in seg.items() if k != "cells"} | {"hexes": len(seg["cells"])}
        for seg in net["segments"]
    ]
    land_n = max(int(land.sum()), 1)
    p_land = layers["precipitation_mm_yr"][land]
    r_land = layers["runoff_mm_yr"][land]
    arid = aridity_class(layers["humidity_index"][land])
    shares = np.bincount(arid, minlength=5) / max(arid.size, 1)
    summary = {
        "epoch_time_ma": hist.epoch_time_ma,
        "sea_level_m": sea_level,
        "latitude": climate_entry["latitude"].to_dict(),
        "climate_profile": config.climate.profile,
        "land_mean_temperature_c": float(layers["temperature_mean_c"][land].mean())
        if land.any()
        else None,
        "land_mean_precipitation_mm_yr": float(p_land.mean()) if p_land.size else None,
        "runoff_ratio": float(r_land.sum() / max(p_land.sum(), 1e-9)),
        "aridity_shares": {
            "hyper_arid": round(float(shares[0]), 4),
            "arid": round(float(shares[1]), 4),
            "semi_arid": round(float(shares[2]), 4),
            "dry_subhumid": round(float(shares[3]), 4),
            "humid": round(float(shares[4]), 4),
        },
        "lake_share_of_land": float((layers["lake_id"] >= 0).sum() / land_n),
        "endorheic_share_of_land": float(layers["endorheic_mask"].sum() / land_n),
        "no_outflow_share_of_land": float(layers["no_outflow_mask"].sum() / land_n),
        "glaciated_share_of_land": float((layers["glacial_potential"][land] > 0.5).mean())
        if land.any()
        else 0.0,
        "rivers": len(net["systems"]),
        "river_segments": len(rivers),
        "max_strahler": int(net["strahler"].max()) if net["strahler"].size else 0,
        "max_discharge_m3_s": float(q[land.ravel()].max()) if land.any() else 0.0,
        "lakes": len(hyd["lakes"]),
        "closed_lakes": int(sum(1 for lk in hyd["lakes"] if lk["closed"])),
        "dry_closed_basins": len(hyd["dry_basins"]),
        "external_inflow": climate_entry["inflow_info"],
        "mean_denudation_m_ma": float(evo["erosion_total_m"][land.ravel()].mean() / years)
        if land.any()
        else 0.0,
        "land_fraction_viewport": float(land.mean()),
        "v3_land_fraction_viewport": float(hist.layers["land_mask"].mean()),
        "world_climate": wc.balance,
        "water_balance": getattr(clim, "balance", {}),
        "erosion_duration_ma": config.erosion.duration_ma if config.erosion.enabled else 0.0,
        "erosion_steps": len(evo["steps"]),
        "coasts": coast_summary,
        "units": {
            "precipitation": "mm/yr",
            "discharge": "m3/s",
            "elevation": "m",
            "erosion_rate": "m/Ma (1 m/Ma = 0.001 mm/yr)",
        },
    }
    warnings = []
    if climate_entry["latitude"].compressed:
        warnings.append(
            f"The simulation area is taller than the room between the poles at 111 km per "
            f"degree; latitude is compressed to {climate_entry['latitude'].km_per_deg:.0f} km "
            "per degree (set climate.km_per_degree to override)."
        )
    if abs(evo["budget"].get("closure_error_m3", 0.0)) > 1e-6 * max(
        evo["budget"].get("fluvial_eroded_m3", 0.0), 1.0
    ):
        warnings.append("The sediment budget does not close within 1e-6.")
    timings["v4_assemble"] = round(time.perf_counter() - t0, 4)
    data = HydroData(
        layers=layers,
        sea_level_m=sea_level,
        epoch_time_ma=hist.epoch_time_ma,
        latitude=climate_entry["latitude"].to_dict(),
        rivers=rivers,
        river_systems=net["systems"],
        lakes=hyd["lakes"],
        basins=hyd["basins"],
        landmasses=lands,
        budget=evo["budget"],
        summary=summary,
        stage_keys=keys,
        steps=evo["steps"],
        warnings=warnings,
        timings=timings,
    )
    return data, cache, reused
