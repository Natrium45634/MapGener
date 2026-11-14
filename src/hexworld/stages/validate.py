"""Stage ``validate``: structural checks of a generated or loaded project."""

from __future__ import annotations

import numpy as np
from scipy import ndimage

from .. import hexgrid
from .raster import component_table, neighbor_index

TINY_FRAGMENT_HEXES = 3


def validate_project(project) -> dict:
    errors: list[str] = []
    warnings: list[str] = []
    checks: dict = {}
    layout = project.layout
    plate_id = project.layer("plate_id")
    n_plates = len(project.plates)

    # 1. every hex belongs to exactly one existing plate
    if plate_id.shape != (layout.height, layout.width):
        errors.append("plate_id shape does not match the hex grid")
    in_range = bool(plate_id.min() >= 0 and plate_id.max() < n_plates)
    checks["all_hexes_assigned"] = in_range
    if not in_range:
        errors.append("plate_id refers to a non-existent plate")

    # 2. connectivity within the window
    _labels, comp_plate, size, touches, largest = component_table(plate_id, layout)
    detached = ~touches & ~largest
    checks["plates_in_view"] = int(np.unique(plate_id).size)
    checks["components"] = int(size.size)
    checks["detached_components"] = int(detached.sum())
    checks["edge_split_components"] = int((touches & ~largest).sum())
    if detached.any():
        errors.append(
            f"{int(detached.sum())} plate fragment(s) are disconnected inside the window "
            f"(plates {sorted(set(comp_plate[detached].tolist()))[:8]})"
        )
    tiny = (size <= TINY_FRAGMENT_HEXES) & ~largest
    checks["tiny_fragments"] = int(tiny.sum())
    if tiny.sum() > max(3, 0.0005 * plate_id.size):
        errors.append(f"{int(tiny.sum())} tiny plate fragments - mass single-hex artefacts")
    elif tiny.any():
        warnings.append(
            f"{int(tiny.sum())} tiny fragment(s) at the window edge (plates continuing outside)"
        )

    # 3. boundary graph == plate adjacency
    edges = project.boundaries
    expected = 0
    flat = plate_id.ravel()
    for d in hexgrid.HALF_DIRECTIONS:
        nb, valid = neighbor_index(layout, d)
        expected += int(np.count_nonzero(valid.ravel() & (flat[nb.ravel()] != flat)))
    checks["boundary_edges"] = len(edges)
    if expected != len(edges):
        errors.append(f"boundary graph has {len(edges)} edges, adjacency implies {expected}")
    if len(edges):
        if (
            not layout.inside(edges.col_a, edges.row_a).all()
            or not layout.inside(edges.col_b, edges.row_b).all()
        ):
            errors.append("boundary edge refers to a hex outside the grid")
        else:
            pa = plate_id[edges.row_a, edges.col_a]
            pb = plate_id[edges.row_b, edges.col_b]
            if not (np.array_equal(pa, edges.plate_a) and np.array_equal(pb, edges.plate_b)):
                errors.append("boundary edge plates do not match plate_id")
            if not np.all(edges.plate_a < edges.plate_b):
                errors.append("boundary edges must satisfy plate_a < plate_b")
            dist = hexgrid.offset_distance(edges.col_a, edges.row_a, edges.col_b, edges.row_b)
            if not np.all(dist == 1):
                errors.append("boundary edge joins hexes that do not share an edge")
            norm = np.hypot(edges.normal_x, edges.normal_y)
            if not np.allclose(norm, 1.0, atol=1e-4):
                errors.append("boundary normals are not unit vectors")
            speed2 = edges.vrel_x.astype(np.float64) ** 2 + edges.vrel_y.astype(np.float64) ** 2
            comp2 = edges.vn.astype(np.float64) ** 2 + edges.vt.astype(np.float64) ** 2
            if not np.allclose(speed2, comp2, rtol=1e-3, atol=1e-2):
                errors.append("vn/vt do not decompose the relative velocity")

    # 4. numeric sanity of layers
    for name, arr in project.layers.items():
        if np.issubdtype(arr.dtype, np.floating) and not np.isfinite(arr).all():
            errors.append(f"layer {name} contains non-finite values")
    for name in ("uplift", "rifting", "subduction", "volcanism", "seismicity"):
        arr = project.layer(name)
        if arr.min() < 0.0 or arr.max() > 1.0:
            errors.append(f"layer {name} is outside [0, 1]")

    # 5. world-level connectivity on the reference raster (model area)
    ref = project.cache.get("reference")
    if ref is not None and ref.plate is not None:
        split = []
        edge = np.zeros(ref.plate.shape, dtype=bool)
        edge[0, :] = edge[-1, :] = edge[:, 0] = edge[:, -1] = True
        for pid in np.unique(ref.plate[ref.in_sim]):
            labels, count = ndimage.label(ref.plate == pid)
            if count < 2:
                continue
            sizes = np.bincount(labels.ravel(), minlength=count + 1)[1:]
            on_edge = np.zeros(count + 1, dtype=bool)
            on_edge[labels[edge]] = True
            # pieces cut off by the raster rectangle may join outside it: ignore them
            inner = ~on_edge[1:] & (sizes > 2)
            inner[np.argmax(sizes)] = False
            if inner.any():
                split.append(int(pid))
        checks["world_split_plates"] = len(split)
        if split:
            warnings.append(
                f"plates {split[:8]} are disconnected on the world reference raster "
                "(lower plates.warp_strength to avoid this)"
            )

    return {"ok": not errors, "errors": errors, "warnings": warnings, "checks": checks}
