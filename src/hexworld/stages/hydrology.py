"""v4 stage ``hydrology``: flow routing, lakes, endorheic basins and the river network.

Routing runs on a **graph**: the six edge neighbours of every hex (D6, all at the hex
spacing) - no square-raster D8 is applied to hexes. The coarse world raster (8 neighbours)
is used only to estimate water that enters the window from outside.

Depressions: priority-flood (Barnes et al. 2014 style) builds a *routing* surface
``routing_m`` = water level a cell would drain at. The physical elevation is **never**
raised. Every cell gets one receiver; all receivers point to cells processed earlier, so
the graph is acyclic by construction. Cells outside depressions use steepest descent;
cells in depressions drain along the flood path to the spill point.

Each depression is then a candidate lake with a water balance:

    inflow + (P - E_open_water) * lake_area   ->  overflow to the spill, or
                                              ->  a closed (endorheic) lake that shrinks
                                                  until evaporation equals inflow.

Units: discharge m^3/s (annual mean), areas km^2, water mm/yr, lengths km.
"""

from __future__ import annotations

import heapq
from dataclasses import dataclass, field

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

from ..rng import Stream
from .climate import SECONDS_PER_YEAR
from .raster import neighbor_index

OUTLET_SEA, OUTLET_EDGE, OUTLET_ENDORHEIC = 1, 2, 3
MOUTH_TYPES = ("none", "sea", "window_edge", "lake", "endorheic")


@dataclass
class Graph:
    """Neighbour lists: ``nbr[i, k]`` is a neighbour of i or -1; ``dist`` in metres."""

    nbr: np.ndarray
    dist: np.ndarray
    border: np.ndarray  # bool: cells on the edge of the computational domain


def hex_graph(layout) -> Graph:
    n = layout.width * layout.height
    nbr = np.full((n, 6), -1, dtype=np.int64)
    for d in range(6):
        idx, valid = neighbor_index(layout, d)
        nbr[:, d] = np.where(valid.ravel(), idx.ravel(), -1)
    border = (nbr < 0).any(axis=1)
    return Graph(nbr, np.full((n, 6), layout.spacing_km * 1000.0), border)


def grid_graph(ny: int, nx: int, cell_m: float) -> Graph:
    rows, cols = np.mgrid[0:ny, 0:nx]
    rows, cols = rows.ravel(), cols.ravel()
    nbr = np.full((ny * nx, 8), -1, dtype=np.int64)
    dist = np.zeros((ny * nx, 8))
    k = 0
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            if dr == 0 and dc == 0:
                continue
            r, c = rows + dr, cols + dc
            ok = (r >= 0) & (r < ny) & (c >= 0) & (c < nx)
            nbr[:, k] = np.where(ok, r * nx + c, -1)
            dist[:, k] = cell_m * np.hypot(dr, dc)
            k += 1
    border = (rows == 0) | (rows == ny - 1) | (cols == 0) | (cols == nx - 1)
    return Graph(nbr, dist, border)


@dataclass
class Routing:
    receiver: np.ndarray  # int64, receiver of every cell (itself for outlets)
    rank: np.ndarray  # int64, processing order: every receiver has a lower rank
    order: np.ndarray  # cells sorted by rank (downstream first)
    routing_m: np.ndarray  # priority-flood water level, >= elevation
    outlet: np.ndarray  # bool


def route(z: np.ndarray, outlet: np.ndarray, graph: Graph, choice=None) -> Routing:
    """Priority-flood routing; outside depressions each cell drains to a lower neighbour.

    ``choice=None``: steepest descent (D6 on hexes, D8 on the grid). With ``choice`` (one
    uniform number in [0, 1) per cell) the receiver is drawn among the *lower* neighbours
    with probability proportional to the drop (Rho8, Fairfield & Leymarie 1991). On a
    smooth inclined plane steepest descent gives straight parallel lines one cell apart;
    the random choice turns them into converging, dendritic networks (Scheidegger's model)
    while a cell in a carved valley still has one clearly lower neighbour. Either way the
    receiver is strictly lower on the filled surface, so the graph stays acyclic.
    """
    z = np.asarray(z, dtype=np.float64)
    n = z.size
    nbr = graph.nbr
    receiver = np.arange(n, dtype=np.int64)
    filled = z.copy()
    rank = np.full(n, -1, dtype=np.int64)
    queued = outlet.copy()
    # outlets that touch the land are the seeds; other outlets just come first
    touching = outlet & ((nbr >= 0) & ~outlet[np.maximum(nbr, 0)]).any(axis=1)
    quiet = np.nonzero(outlet & ~touching)[0]
    rank[quiet] = np.arange(quiet.size)
    counter = quiet.size
    seeds = np.nonzero(touching)[0]
    seeds = seeds[np.lexsort((seeds, z[seeds]))]
    heap = [(float(z[s]), i, int(s)) for i, s in enumerate(seeds)]
    heapq.heapify(heap)
    seq = len(heap)
    zl = z.tolist()
    nb = nbr.tolist()
    fl = filled.tolist()
    q = queued.tolist()
    rk = rank.tolist()
    rc = receiver.tolist()
    pop, push = heapq.heappop, heapq.heappush
    while heap:
        level, _s, c = pop(heap)
        rk[c] = counter
        counter += 1
        for m in nb[c]:
            if m < 0 or q[m]:
                continue
            q[m] = True
            lv = zl[m] if zl[m] > level else level
            fl[m] = lv
            rc[m] = c
            push(heap, (lv, seq, m))
            seq += 1
    rank = np.asarray(rk, dtype=np.int64)
    receiver = np.asarray(rc, dtype=np.int64)
    filled = np.asarray(fl)
    # unreachable cells (isolated components without an outlet) become their own outlets
    lost = rank < 0
    if lost.any():
        idx = np.nonzero(lost)[0]
        rank[idx] = counter + np.arange(idx.size)
        receiver[idx] = idx
        outlet = outlet | lost
    # steepest descent wherever a neighbour drains at a strictly lower level
    valid = nbr >= 0
    nf = np.where(valid, filled[np.maximum(nbr, 0)], np.inf)
    drop = (filled[:, None] - nf) / graph.dist.clip(1e-9)
    drop = np.where(valid & (nf < filled[:, None]), drop, -np.inf)
    if choice is None:
        best = np.argmax(drop, axis=1)
    else:
        weight = np.where(np.isfinite(drop), np.maximum(drop, 0.0), 0.0)
        cum = np.cumsum(weight, axis=1)
        pick = np.asarray(choice, dtype=np.float64).ravel()[:, None] * cum[:, -1:]
        best = np.minimum((cum <= pick).sum(axis=1), nbr.shape[1] - 1)
        # guard the boundary case pick == total (never above a zero-weight slot)
        best = np.where(weight[np.arange(n), best] > 0.0, best, np.argmax(drop, axis=1))
    has = np.isfinite(drop[np.arange(n), best]) & ~outlet
    receiver[has] = nbr[np.nonzero(has)[0], best[has]]
    order = np.argsort(rank, kind="stable")
    return Routing(receiver, rank, order, filled, outlet)


def accumulate(routing: Routing, source: np.ndarray) -> np.ndarray:
    """Sum ``source`` down the receiver graph (upstream first)."""
    acc = np.asarray(source, dtype=np.float64).copy()
    rc = routing.receiver
    for c in routing.order[::-1].tolist():
        r = rc[c]
        if r != c:
            acc[r] += acc[c]
    return acc


@dataclass
class Lakes:
    lake_id: np.ndarray  # int32 per cell, -1 = none
    level_m: np.ndarray  # float32 per cell, NaN = none
    depth_m: np.ndarray
    endorheic_lake: np.ndarray  # bool per cell (closed-lake water surface)
    table: list = field(default_factory=list)


def hydrology(
    z,
    sea,
    graph: Graph,
    cell_area_m2: float,
    runoff_mm_yr,
    precip_mm_yr,
    open_water_evap_mm_yr,
    external_inflow_m3_s=None,
    min_lake_depth_m=2.0,
    channel_loss=0.0,
    choice=None,
):
    """Routing, discharge with lake balances, basins. Returns a dict of arrays + tables.

    ``choice``: per-cell uniform numbers for the stochastic receiver choice of
    :func:`route` (``None`` = steepest descent).

    ``channel_loss`` (dimensionless, 0 = off) adds transmission losses of channels where
    evaporation exceeds rain: ``loss = channel_loss * (E - P) * width * length`` with the
    hydraulic-geometry width ``5 * sqrt(Q)`` m - open-water evaporation plus infiltration
    into a dry bed. Small desert streams dry up; large rivers survive (Nile-like).
    """
    n = z.size
    outlet = sea | (graph.border & ~sea)
    routing = route(z, outlet, graph, choice)
    rc = routing.receiver
    depth_flood = routing.routing_m - z
    in_dep = (depth_flood > 0.01) & ~routing.outlet

    # depressions = connected cells under the flood level
    idx = np.nonzero(in_dep)[0]
    comp = np.full(n, -1, dtype=np.int64)
    comps: list[np.ndarray] = []
    if idx.size:
        nb = graph.nbr[idx]
        ok = nb >= 0
        both = ok & in_dep[np.maximum(nb, 0)]
        src = np.repeat(idx, 6 if nb.shape[1] == 6 else nb.shape[1])[both.ravel()]
        dst = nb[both]
        local = np.full(n, -1, dtype=np.int64)
        local[idx] = np.arange(idx.size)
        g = coo_matrix((np.ones(src.size), (local[src], local[dst])), shape=(idx.size, idx.size))
        n_comp, labels = connected_components(g, directed=False)
        # stable numbering: by the lowest rank in the component (downstream first)
        first = np.full(n_comp, np.iinfo(np.int64).max)
        np.minimum.at(first, labels, routing.rank[idx])
        renum = np.empty(n_comp, dtype=np.int64)
        renum[np.argsort(first, kind="stable")] = np.arange(n_comp)
        comp[idx] = renum[labels]
        order_c = np.argsort(comp[idx], kind="stable")
        bounds = np.searchsorted(comp[idx][order_c], np.arange(n_comp + 1))
        comps = [idx[order_c][bounds[i] : bounds[i + 1]] for i in range(n_comp)]
    is_lake_comp = np.zeros(len(comps), dtype=bool)
    for i, c in enumerate(comps):
        if depth_flood[c].max() >= min_lake_depth_m:
            is_lake_comp[i] = True
    lake_cells = np.zeros(n, dtype=bool)
    for i, c in enumerate(comps):
        if is_lake_comp[i]:
            lake_cells[c] = True

    net = (np.asarray(precip_mm_yr) - np.asarray(open_water_evap_mm_yr)) / 1000.0 * cell_area_m2
    source = np.where(sea, 0.0, np.asarray(runoff_mm_yr) / 1000.0 * cell_area_m2)  # m^3/yr
    source = np.where(lake_cells, 0.0, source)
    if external_inflow_m3_s is not None:
        source = source + np.asarray(external_inflow_m3_s) * SECONDS_PER_YEAR

    # exits of each lake component: cells whose receiver lies outside the component
    exits: dict[int, list[int]] = {}
    for i, c in enumerate(comps):
        if not is_lake_comp[i]:
            continue
        out = c[comp[rc[c]] != i]
        exits[i] = sorted(out.tolist(), key=lambda e: -routing.rank[e])  # upstream first
    remaining = {i: len(v) for i, v in exits.items()}
    inflow = {i: 0.0 for i in exits}
    outflow = {}
    acc = source.copy()
    comp_l = comp.tolist()
    rc_l = rc.tolist()
    deficit = np.maximum(np.asarray(open_water_evap_mm_yr) - np.asarray(precip_mm_yr), 0.0) / 1000.0
    length = float(graph.dist[0, 0])
    lossy = (channel_loss > 0) & ~sea & ~lake_cells & (deficit > 0)
    loss_coef = np.where(lossy, channel_loss * deficit * length * 5.0, 0.0).tolist()
    lost = np.zeros(n)
    passed = np.ones(n)
    for cc in routing.order[::-1].tolist():
        if loss_coef[cc] > 0.0 and acc[cc] > 0.0:
            loss = min(acc[cc], loss_coef[cc] * (acc[cc] / SECONDS_PER_YEAR) ** 0.5)
            passed[cc] = 1.0 - loss / acc[cc]
            lost[cc] = loss
            acc[cc] -= loss
        r = rc_l[cc]
        k = comp_l[cc]
        if k >= 0 and k in exits and comp_l[r] != k:
            inflow[k] += acc[cc]
            remaining[k] -= 1
            acc[cc] = 0.0
            if remaining[k] == 0:
                total_net = float(net[comps[k]].sum())
                out = max(0.0, inflow[k] + total_net)
                outflow[k] = out
                if r != cc:
                    acc[r] += out
            continue
        if r != cc:
            acc[r] += acc[cc]
    # every exit of a lake drains through the main (last processed) one
    for cells in exits.values():
        if len(cells) > 1:
            main = cells[-1]
            for e in cells[:-1]:
                rc[e] = main if routing.rank[main] < routing.rank[e] else rc[e]

    lake_id = np.full(n, -1, dtype=np.int32)
    level = np.full(n, np.nan, dtype=np.float32)
    depth = np.full(n, np.nan, dtype=np.float32)
    closed_water = np.zeros(n, dtype=bool)
    endorheic_terminal = np.zeros(n, dtype=bool)
    table = []
    dry_basins: list[dict] = []
    lid = 0
    for k, cells in enumerate(comps):
        if not is_lake_comp[k]:
            continue
        flood_level = float(routing.routing_m[cells].max())
        closed = outflow.get(k, 0.0) <= 0.0 and (inflow[k] + float(net[cells].sum())) < 0.0
        water = cells
        lvl = flood_level
        if closed:
            # the lake shrinks until evaporation balances the inflow
            order_z = cells[np.argsort(z[cells], kind="stable")]
            budget = inflow[k] + np.cumsum(net[order_z])
            fit = int(np.searchsorted(-budget, 0.0))  # first index where budget < 0
            fit = max(fit, 0)
            water = order_z[:fit]
            lvl = float(z[water].max()) if fit else float(z[order_z[0]])
            endorheic_terminal[cells] = True
        if water.size:
            lake_id[water] = lid
            level[water] = lvl
            depth[water] = np.maximum(lvl - z[water], 0.0)
            if closed:
                closed_water[water] = True
        area_km2 = water.size * cell_area_m2 / 1e6
        if not water.size:
            # the closed basin's inflow evaporates before a lake can form: a dry pan
            # (playa), not a lake; it keeps no lake id but stays a terminal basin
            dry_basins.append(
                {
                    "basin_cells": int(cells.size),
                    "floor_m": round(lvl, 2),
                    "inflow_m3_s": round(inflow[k] / SECONDS_PER_YEAR, 4),
                    "cell": int(cells[np.argmin(z[cells])]),
                }
            )
            continue
        table.append(
            {
                "id": lid,
                "cells": int(water.size),
                "area_km2": round(area_km2, 3),
                "level_m": round(lvl, 2),
                "max_depth_m": round(float(depth[water].max()) if water.size else 0.0, 2),
                "basin_cells": int(cells.size),
                "closed": bool(closed),
                "inflow_m3_s": round(inflow[k] / SECONDS_PER_YEAR, 4),
                "outflow_m3_s": round(outflow.get(k, 0.0) / SECONDS_PER_YEAR, 4),
                "spill_cell": int(rc[exits[k][-1]]) if exits.get(k) else -1,
            }
        )
        lid += 1

    discharge = acc / SECONDS_PER_YEAR
    area_acc = accumulate(
        Routing(rc, routing.rank, routing.order, routing.routing_m, routing.outlet),
        np.where(sea, 0.0, cell_area_m2 / 1e6),
    )

    # basins: every land cell belongs to one terminal (sea, window edge or closed lake)
    basin_term = np.full(n, -1, dtype=np.int64)
    term_kind = {}
    for cc in routing.order.tolist():
        if routing.outlet[cc]:
            basin_term[cc] = cc
            term_kind[cc] = OUTLET_SEA if sea[cc] else OUTLET_EDGE
        elif endorheic_terminal[cc]:
            key = -(comp[cc] + 2)
            basin_term[cc] = key
            term_kind[key] = OUTLET_ENDORHEIC
        else:
            basin_term[cc] = basin_term[rc[cc]]
    land = ~sea
    keys, counts = np.unique(basin_term[land], return_counts=True)
    order_b = np.lexsort((keys, -counts))
    basin_id = np.full(n, -1, dtype=np.int32)
    remap = {int(keys[j]): b for b, j in enumerate(order_b)}
    bl = basin_term[land]
    basin_id[land] = np.array([remap[int(v)] for v in bl], dtype=np.int32) if bl.size else bl
    basins = [
        {
            "id": b,
            "cells": int(counts[j]),
            "area_km2": round(float(counts[j]) * cell_area_m2 / 1e6, 1),
            "terminal": ("sea", "window_edge", "endorheic")[term_kind[int(keys[j])] - 1],
        }
        for b, j in enumerate(order_b)
    ]
    endorheic = np.zeros(n, dtype=bool)
    for b in basins:
        if b["terminal"] == "endorheic":
            endorheic |= basin_id == b["id"]
    # share of a cell's water that reaches the sea or the window edge
    reach = np.zeros(n)
    for cc in routing.order.tolist():
        r = rc[cc]
        if routing.outlet[cc]:
            reach[cc] = 1.0
        elif endorheic_terminal[cc] or r == cc:
            reach[cc] = 0.0
        else:
            reach[cc] = passed[cc] * reach[r]
    return {
        "routing": routing,
        "receiver": rc,
        "discharge_m3_s": discharge,
        "area_km2": area_acc,
        "lake_id": lake_id,
        "lake_level_m": level,
        "lake_depth_m": depth,
        "closed_lake": closed_water,
        "basin_id": basin_id,
        "endorheic_mask": endorheic,
        "lakes": table,
        "dry_basins": dry_basins,
        "basins": basins,
        "depression": comp,
        "flood_depth_m": depth_flood,
        "channel_loss_m3_s": lost / SECONDS_PER_YEAR,
        "outflow_share": reach,
    }


def routing_choice(seed: str, n: int, mode: str):
    """Per-cell numbers for :func:`route`: fixed by the seed, the same in every step."""
    if mode == "steepest":
        return None
    return Stream(seed, "v4/routing").random(n)


def river_network(
    hyd: dict,
    land,
    layout,
    threshold_m3_s: float,
    slope_m_per_km,
    min_area_km2: float = 0.0,
    min_system_area_km2: float = 0.0,
):
    """River cells, Strahler order and segments as world polylines.

    A cell is a river where discharge and upstream area reach the thresholds. Then whole
    systems are generalised away when the basin at their last river cell (mouth, lake or
    window edge) is smaller than ``min_system_area_km2``.
    """
    q = hyd["discharge_m3_s"]
    rc = hyd["receiver"]
    lake = hyd["lake_id"] >= 0
    area = hyd["area_km2"]
    river = land & ~lake & (q >= threshold_m3_s) & (area >= min_area_km2 - 1e-6)
    n = q.size
    rank = hyd["routing"].rank
    if min_system_area_km2 > 0.0:
        terminal = np.arange(n)
        cells = np.nonzero(river)[0]
        for c in cells[np.argsort(rank[cells], kind="stable")].tolist():  # downstream first
            r = int(rc[c])
            if r != c and river[r]:
                terminal[c] = terminal[r]
        river &= area[terminal] >= min_system_area_km2 - 1e-6
    donors: list[list[int]] = [[] for _ in range(n)]
    for c in np.nonzero(river)[0].tolist():
        r = int(rc[c])
        if r != c and river[r]:
            donors[r].append(c)
    strahler = np.zeros(n, dtype=np.uint8)
    for c in sorted(np.nonzero(river)[0].tolist(), key=lambda x: -rank[x]):  # upstream first
        ds = donors[c]
        if not ds:
            strahler[c] = 1
        else:
            orders = sorted((int(strahler[d]) for d in ds), reverse=True)
            strahler[c] = orders[0] + 1 if len(orders) > 1 and orders[1] == orders[0] else orders[0]
    hx, hy = layout.centers()
    hx, hy = hx.ravel(), hy.ravel()
    starts = [c for c in np.nonzero(river)[0].tolist() if len(donors[c]) != 1]
    starts.sort(key=lambda c: (rank[c], c))
    seg_of = np.full(n, -1, dtype=np.int64)
    segments = []
    lake_id = hyd["lake_id"]
    for s in starts:
        cells = [s]
        c = s
        while True:
            r = int(rc[c])
            if r == c or not river[r] or len(donors[r]) != 1:
                break
            cells.append(r)
            c = r
        end = int(rc[cells[-1]])
        sid = len(segments)
        seg_of[cells] = sid
        if end == cells[-1]:
            mouth = "window_edge"
        elif not land[end]:
            mouth = "sea"
        elif lake_id[end] >= 0:
            mouth = "endorheic" if hyd["closed_lake"][end] else "lake"
        elif river[end]:
            mouth = "none"  # flows on into the next segment
        else:
            mouth = "endorheic" if hyd["endorheic_mask"][end] else "lake"
        xs = [float(hx[k]) for k in cells]
        ys = [float(hy[k]) for k in cells]
        if end != cells[-1]:  # connect to the junction / mouth
            xs.append(float(0.5 * (hx[cells[-1]] + hx[end])) if mouth != "none" else float(hx[end]))
            ys.append(float(0.5 * (hy[cells[-1]] + hy[end])) if mouth != "none" else float(hy[end]))
        else:  # leaves the window: carry the line on to the nearest edge of the window
            b = layout.bounds
            x, y = xs[-1], ys[-1]
            side = int(np.argmin((x - b.xmin, b.xmax - x, y - b.ymin, b.ymax - y)))
            xs.append(float(b.xmin if side == 0 else b.xmax if side == 1 else x))
            ys.append(float(b.ymin if side == 2 else b.ymax if side == 3 else y))
        sl = np.asarray(slope_m_per_km)[cells]
        segments.append(
            {
                "id": sid,
                "cells": cells,
                "x_km": xs,
                "y_km": ys,
                "strahler": int(strahler[cells[-1]]),
                "length_km": round(len(cells) * layout.spacing_km, 2),
                "discharge_m3_s": round(float(q[cells[-1]]), 3),
                "drainage_area_km2": round(float(hyd["area_km2"][cells[-1]]), 1),
                "mean_slope_m_per_km": round(float(sl.mean()), 4),
                "mouth": mouth,
                "next_cell": end,
                "next_segment": -1,
                "basin_id": int(hyd["basin_id"][cells[-1]]),
            }
        )
    for seg in segments:
        if seg["mouth"] == "none":
            seg["next_segment"] = int(seg_of[seg["next_cell"]])
    mouths = np.zeros(n, dtype=bool)
    for seg in segments:
        if seg["mouth"] in ("sea", "lake", "endorheic", "window_edge"):
            mouths[seg["cells"][-1]] = True
    return river, strahler, segments, mouths


def river_systems(segments: list[dict]) -> list[dict]:
    """Rivers from mouth to the farthest source along the larger tributary (main stem)."""
    up: dict[int, list[int]] = {}
    for s in segments:
        if s["next_segment"] >= 0:
            up.setdefault(s["next_segment"], []).append(s["id"])
    systems = []
    for s in segments:
        if s["next_segment"] >= 0:
            continue
        stem, total, area = [s["id"]], s["length_km"], s["drainage_area_km2"]
        cur = s["id"]
        count = 1
        stack = list(up.get(cur, []))
        while stack:
            count += 1
            stack.extend(up.get(stack.pop(), []))
        while up.get(cur):
            nxt = max(up[cur], key=lambda i: (segments[i]["discharge_m3_s"], -i))
            stem.append(nxt)
            total += segments[nxt]["length_km"]
            cur = nxt
        systems.append(
            {
                "mouth_segment": s["id"],
                "mouth": s["mouth"],
                "discharge_m3_s": s["discharge_m3_s"],
                "drainage_area_km2": area,
                "main_stem_length_km": round(total, 1),
                "strahler": s["strahler"],
                "segments": count,
                "main_stem": stem,
            }
        )
    systems.sort(key=lambda r: (-r["discharge_m3_s"], r["mouth_segment"]))
    for i, r in enumerate(systems):
        r["id"] = i
    return systems
