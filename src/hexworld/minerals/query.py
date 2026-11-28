"""Queries of the sparse plan (§14): columns, sections, hexes, regions, depth bins.

A column returns true intersections along the vertical ray (several stacked bodies at one
point, T29); a hex returns the clipped integral, not the result at its centre (T30); hex and
depth-bin aggregates sum to the unique lot totals (T70, T71). Read-only: nothing reseeds.
"""

from __future__ import annotations

import numpy as np


def query_column_bodies(bodies, x_m: float, y_m: float, surface_m: float = 0.0,
                        depth_range=None, water_depth_m: float = 0.0) -> list[dict]:  # fmt: skip
    """Intervals of every body on the vertical line, top first. Depths are below the solid
    surface (sea floor under the sea); the water column is reported separately (T28)."""
    out = []
    for b in bodies:
        for z_top, z_bot in b.column(x_m, y_m):
            top = max(surface_m - z_top, 0.0)  # an exposed body starts at 0, never < 0
            bot = surface_m - z_bot
            if depth_range is not None and (bot < depth_range[0] or top > depth_range[1]):
                continue
            out.append({"body_id": b.body_id, "system_id": b.system_id,
                        "morphology": b.morphology, "z_top_m": z_top, "z_bottom_m": z_bot,
                        "depth_top_m": top, "depth_bottom_m": bot,
                        "drilled_interval_m": z_top - z_bot,
                        "water_depth_m": water_depth_m})  # fmt: skip
    return sorted(out, key=lambda h: h["depth_top_m"])


def section_bodies(bodies, xs_m, ys_m, surface_fn=None) -> list[dict]:
    """Intersections of the bodies along a polyline sampled at (xs, ys): per body the list of
    (index, z_top, z_bottom) — geometry, not decoration (T72)."""
    out = []
    for b in bodies:
        rows = []
        for i, (x, y) in enumerate(zip(xs_m, ys_m, strict=True)):
            for zt, zb in b.column(float(x), float(y)):
                rows.append((i, zt, zb))
        if rows:
            out.append({"body_id": b.body_id, "morphology": b.morphology, "intervals": rows})
    return out


def aggregate_to_hex(lots, layout, n: int = 24) -> dict:
    """Mass per hex of each lot ``{lot_id: {(col, row): t}}``; parts sum to the lot (T70)."""
    out = {}
    for lot in lots:
        b = lot["body"]
        split = b.hex_split(layout, n)
        vol = sum(split.values())
        out[lot["lot_id"]] = {k: lot["rock_t"] * v / vol for k, v in split.items()} if vol else {}
    return out


def depth_bins(body, surface_fn, edges_m, n: int = 24) -> np.ndarray:
    """Volume of a body per depth bin (``edges_m`` ascending, below the solid surface); the
    bins are half-open so adjacent bins never count a boundary twice (T71)."""
    p, w = body.quadrature(n)
    d = np.asarray(surface_fn(p[:, 0], p[:, 1])) - p[:, 2]
    idx = np.searchsorted(np.asarray(edges_m, dtype=np.float64), d, side="right") - 1
    out = np.zeros(len(edges_m) - 1)
    ok = (idx >= 0) & (idx < len(out))
    np.add.at(out, idx[ok], w[ok])
    return out
