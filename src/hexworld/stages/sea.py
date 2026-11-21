"""v2 stage ``sea_level_landmasses``: sea level, land mask, landmasses, shelf.

Everything here is derived from ``macro_elevation_m`` and one number, the sea
level. Changing only the sea level never touches plates, crust or elevation.

Two mutually exclusive modes:

* ``target_land_fraction`` - the level is the quantile of the *world*
  elevation over the simulation area that leaves the requested share above
  water. A regional window may show a different share; both are reported;
* ``fixed_sea_level_m`` - the level is given, the land share is measured.

RNG namespace ``sea_level/v2`` is reserved; the stage is currently fully
deterministic and draws nothing from it.
"""

from __future__ import annotations

import numpy as np
from scipy import ndimage

from ..config import SeaParams
from ..geometry import HexLayout
from .raster import plate_components


def resolve_sea_level(world_elevation_sim: np.ndarray, params: SeaParams) -> float:
    """Sea level (m, model datum) for ``params``; input is world elevation over the sim area."""
    if params.sea_mode == "fixed_sea_level_m":
        return float(params.fixed_sea_level_m)
    level = np.quantile(world_elevation_sim.astype(np.float64), 1.0 - params.target_land_fraction)
    return float(np.round(level, 3))


def label_mask(mask: np.ndarray, layout: HexLayout) -> tuple[np.ndarray, np.ndarray]:
    """Edge-connected components of ``mask`` on the hex grid.

    Returns ``(ids, sizes)``: ``ids`` is int32 with -1 outside the mask; ids are
    ordered by decreasing size (ties: first hex in row-major order).
    """
    ids = np.full(mask.shape, -1, dtype=np.int32)
    if not mask.any():
        return ids, np.zeros(0, dtype=np.int64)
    _count, labels = plate_components(mask.astype(np.int8), layout)
    flat = labels.ravel()[mask.ravel()]
    uniq, first, counts = np.unique(flat, return_index=True, return_counts=True)
    order = np.lexsort((first, -counts))
    remap = np.empty(order.size, dtype=np.int32)
    remap[order] = np.arange(order.size, dtype=np.int32)
    ids.ravel()[mask.ravel()] = remap[np.searchsorted(uniq, flat)]
    return ids, counts[order]


def derive_land(
    elevation_m, continental_fraction, sea_level_m, layout, params: SeaParams, ocean=None
):
    """Land mask, landmass ids and shelf mask from elevation + sea level.

    ``ocean`` (v4.6): the world-ocean mask; land is everything else, including dry ground
    below the sea level. ``None`` keeps the v2 rule ``elevation > sea level``.
    """
    if ocean is None:
        land = elevation_m > np.float32(sea_level_m)
    else:
        land = ~np.asarray(ocean, dtype=bool)
    landmass_id, sizes = label_mask(land, layout)
    depth = np.float32(sea_level_m) - elevation_m
    shelf = ~land & (continental_fraction >= 0.5) & (depth <= params.shelf_max_depth_m)
    return land, landmass_id, sizes, shelf


def world_landmasses(world_elevation, in_sim, cell_km, sea_level_m, major_area_km2):
    """World-level landmass statistics on the skeleton raster (simulation area)."""
    land = world_elevation > sea_level_m
    labels, count = ndimage.label(land)
    area = np.bincount(labels[in_sim], minlength=count + 1).astype(np.float64) * cell_km**2
    area[0] = 0.0
    order = np.argsort(-area)[: min(count, 12)]
    sim_cells = max(1, int(in_sim.sum()))
    return {
        "land_fraction_simulation": float(land[in_sim].mean()),
        "landmass_count_simulation": int(np.count_nonzero(area > 0)),
        "major_landmass_count": int(np.count_nonzero(area >= major_area_km2)),
        "largest_landmass_share_of_land": float(
            area.max() / max(1.0, land[in_sim].sum() * cell_km**2)
        ),
        "largest_landmasses_km2": [round(float(area[i]), 0) for i in order if area[i] > 0],
        "simulation_area_km2": float(sim_cells * cell_km**2),
    }
