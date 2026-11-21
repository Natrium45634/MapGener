"""Which low ground the world ocean actually floods (TZ 4.6, section 1, package A).

``elevation < sea_level`` does **not** mean ocean. Sea water reaches only the below-datum
ground that is connected to the world ocean; a closed depression at -500 m behind a
+100 m sill is dry land or holds a lake with its own level (set by the v4 water balance).

* **World ocean, world raster.** On the world raster of the epoch (the whole model area,
  not the window) the connected areas below the sea level are found; those of at least
  ``min_ocean_area_km2`` - and always the largest - are the world ocean. Inland seas below
  that size (the Caspian is 0.37 million km^2) are lakes. The edge of the window or of the
  model area is *not* taken as an ocean inlet.
* **Sources on the hex grid.** A hex is an ocean source when it lies below the sea level
  and its world cell is world ocean. A bay cut by the window keeps its sources, so it is
  never turned into a lake by the cut.
* **Flood.** Below-sea hexes connected to a source through below-sea hexes (six
  neighbours) are the ocean. Everything else is land: dry ground, possibly below the sea
  level, or a lake decided later by the hydrology.

Changing the sea level re-runs only this classification and what depends on it; plates,
crust and history are never touched.
"""

from __future__ import annotations

import numpy as np
from scipy import ndimage
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

DEFAULT_MIN_OCEAN_AREA_KM2 = 1.5e6


def world_ocean(world_elevation, sea_level_m: float, cell_km: float, min_area_km2=None):
    """World-ocean mask on the world raster: big below-sea components (8-connected)."""
    min_area_km2 = DEFAULT_MIN_OCEAN_AREA_KM2 if min_area_km2 is None else min_area_km2
    below = np.asarray(world_elevation) <= sea_level_m
    labels, count = ndimage.label(below, structure=np.ones((3, 3), dtype=bool))
    if count == 0:
        return np.zeros(below.shape, dtype=bool)
    area = np.bincount(labels.ravel(), minlength=count + 1).astype(np.float64) * cell_km**2
    area[0] = 0.0
    keep = area >= min_area_km2
    keep[int(np.argmax(area))] = True
    keep[0] = False
    return keep[labels]


def hex_sources(layout, grid, ocean_world, z, sea_level_m: float) -> np.ndarray:
    """Hexes below the sea level whose world cell belongs to the world ocean."""
    hx, hy = layout.centers()
    ri, ci, ok = grid.nearest(hx.ravel(), hy.ravel())
    src = ocean_world[ri, ci] & ok
    return src.reshape(np.shape(z)) & (np.asarray(z) <= sea_level_m)


def world_ocean_hexes(layout, world: dict, sea_level_m: float, min_area_km2=None):
    """Hexes whose world cell belongs to the world ocean (``hist.world`` raster)."""
    from ..stages.history import HistoryGrid

    g = world["grid"]
    grid = HistoryGrid(g["x0_km"], g["y0_km"], g["cell_km"], g["nx"], g["ny"])
    ocean_w = world_ocean(world["elevation_m"], sea_level_m, g["cell_km"], min_area_km2)
    hx, hy = layout.centers()
    ri, ci, ok = grid.nearest(hx.ravel(), hy.ravel())
    return (ocean_w[ri, ci] & ok).reshape(hx.shape)


def sources_from_world(layout, world: dict, z, sea_level_m: float, min_area_km2=None):
    """Hex ocean sources: world-ocean hexes that lie below the sea level on ``z``."""
    return world_ocean_hexes(layout, world, sea_level_m, min_area_km2) & (
        np.asarray(z) <= sea_level_m
    )


def flood(below: np.ndarray, sources: np.ndarray, nbr: np.ndarray) -> np.ndarray:
    """Cells of ``below`` connected to a source through ``below`` cells (flat arrays)."""
    below = np.asarray(below, dtype=bool).ravel()
    sources = np.asarray(sources, dtype=bool).ravel() & below
    n = below.size
    if not sources.any():
        return np.zeros(n, dtype=bool)
    src_all, dst_all = [], []
    for k in range(nbr.shape[1]):
        nb = nbr[:, k]
        ok = (nb >= 0) & below
        ok[ok] &= below[nb[ok]]
        idx = np.nonzero(ok)[0]
        src_all.append(idx)
        dst_all.append(nb[idx])
    src = np.concatenate(src_all)
    dst = np.concatenate(dst_all)
    graph = coo_matrix((np.ones(src.size, dtype=np.int8), (src, dst)), shape=(n, n))
    _count, labels = connected_components(graph, directed=False)
    wet = np.zeros(labels.max() + 1, dtype=bool)
    wet[labels[sources]] = True
    return wet[labels] & below


def ocean_mask(z, sea_level_m: float, nbr: np.ndarray, sources: np.ndarray) -> np.ndarray:
    """World-ocean mask of surface ``z`` (any shape; ``nbr`` indexes its flat cells)."""
    z = np.asarray(z)
    return flood(z.ravel() <= sea_level_m, sources, nbr).reshape(z.shape)


def water_layers(z, sea_level_m: float, ocean, lake_level=None) -> dict:
    """Mutually exclusive water / land masks and water depth of the final surface."""
    z = np.asarray(z, dtype=np.float64)
    ocean = np.asarray(ocean, dtype=bool)
    if lake_level is None:
        lake_level = np.full(z.shape, np.nan)
    lake_level = np.asarray(lake_level, dtype=np.float64)
    lake = ~ocean & np.isfinite(lake_level) & (lake_level > z)
    dry = ~ocean & ~lake
    surface = np.where(ocean, sea_level_m, np.where(lake, lake_level, np.nan))
    depth = np.where(ocean | lake, np.maximum(surface - z, 0.0), 0.0)
    return {
        "ocean_mask": ocean,
        "inland_water_mask": lake,
        "dry_land_mask": dry,
        "below_sea_level_land_mask": dry & (z < sea_level_m),
        "above_sea_level_mask": z > sea_level_m,
        "lake_surface_level_m": np.where(lake, lake_level, np.nan).astype(np.float32),
        "water_surface_level_m": surface.astype(np.float32),
        "water_depth_m": depth.astype(np.float32),
    }


def land_fractions(z, sea_level_m: float, ocean, lake) -> dict:
    """The three land-share metrics, never mixed (TZ 4.6, 1.2.6)."""
    z = np.asarray(z)
    n = max(1, z.size)
    return {
        "above_datum_fraction": float((z > sea_level_m).sum() / n),
        "non_ocean_fraction": float((~ocean).sum() / n),
        "exposed_fraction": float((~ocean & ~lake).sum() / n),
    }
