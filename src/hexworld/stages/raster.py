"""Sampling of the continuous world models onto a hex grid, plus hex topology helpers."""

from __future__ import annotations

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

from .. import hexgrid
from ..geometry import HexLayout


def neighbor_index(layout: HexLayout, direction: int):
    """For every hex: flat index of its neighbour in ``direction`` and a validity mask."""
    cols = np.arange(layout.width, dtype=np.int64)[None, :]
    rows = np.arange(layout.height, dtype=np.int64)[:, None]
    ncol, nrow = hexgrid.offset_neighbor(cols, rows, direction)
    ncol = np.broadcast_to(ncol, (layout.height, layout.width))
    nrow = np.broadcast_to(nrow, (layout.height, layout.width))
    valid = layout.inside(ncol, nrow)
    flat = np.where(valid, nrow * layout.width + ncol, 0)
    return flat, valid


def plate_components(plate_id: np.ndarray, layout: HexLayout):
    """Label edge-connected components of equal plate ID. Returns ``(n, labels)``."""
    flat = plate_id.ravel()
    n = flat.size
    src_all, dst_all = [], []
    for d in hexgrid.HALF_DIRECTIONS:
        nb, valid = neighbor_index(layout, d)
        nb = nb.ravel()
        same = valid.ravel() & (flat[nb] == flat)
        src = np.nonzero(same)[0]
        src_all.append(src)
        dst_all.append(nb[src])
    src = np.concatenate(src_all)
    dst = np.concatenate(dst_all)
    graph = coo_matrix((np.ones(src.size, dtype=np.int8), (src, dst)), shape=(n, n))
    count, labels = connected_components(graph, directed=False)
    return count, labels.reshape(plate_id.shape)


def component_table(plate_id: np.ndarray, layout: HexLayout):
    """Per-component arrays: plate, size, touches-grid-border, is-largest-of-plate."""
    count, labels = plate_components(plate_id, layout)
    flat_labels = labels.ravel()
    size = np.bincount(flat_labels, minlength=count)
    plate = np.zeros(count, dtype=np.int64)
    plate[flat_labels] = plate_id.ravel()
    border = np.zeros(plate_id.shape, dtype=bool)
    border[0, :] = border[-1, :] = border[:, 0] = border[:, -1] = True
    touches = np.zeros(count, dtype=bool)
    touches[flat_labels[border.ravel()]] = True
    largest = np.zeros(count, dtype=bool)
    order = np.lexsort((-size, plate))
    first = np.ones(count, dtype=bool)
    first[1:] = plate[order][1:] != plate[order][:-1]
    largest[order[first]] = True
    return labels, plate, size, touches, largest


def absorb_orphans(plate_id: np.ndarray, layout: HexLayout) -> tuple[np.ndarray, int]:
    """Reassign sub-hex slivers that sampling detached from their plate.

    Only components that are neither the largest piece of their plate nor in
    contact with the grid border are touched (a piece on the border may be
    legitimately connected outside the window). In practice these are a few
    isolated hexes at sharp boundary bends; natural shapes are left intact.
    Returns the fixed array and the number of reassigned hexes.
    """
    out = plate_id.copy()
    labels, _plate, _size, touches, largest = component_table(out, layout)
    orphans = np.nonzero(~touches & ~largest)[0]
    if orphans.size == 0:
        return out, 0
    flat = out.ravel()
    flat_labels = labels.ravel()
    neighbors = [neighbor_index(layout, d) for d in range(6)]
    changed = 0
    for comp in orphans:
        cells = np.nonzero(flat_labels == comp)[0]
        own = flat[cells[0]]
        around = []
        for nb, valid in neighbors:
            nbf = nb.ravel()[cells]
            ok = valid.ravel()[cells]
            vals = flat[nbf[ok]]
            around.append(vals[vals != own])
        around = np.concatenate(around)
        if around.size:
            flat[cells] = np.bincount(around).argmax()
            changed += cells.size
    return out, changed
