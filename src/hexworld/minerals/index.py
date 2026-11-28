"""Spatial helpers of the 5.5.1 plan: body reach, the hex -> bodies index (from the clipped
quadrature, so a thin vein is found in every hex it crosses, T30), section rows."""

from __future__ import annotations

import math

import numpy as np

from .bodies import Ellipsoid, Pipe, Plate, VeinNetwork


def body_reach_m(b) -> float:
    """Horizontal radius around the origin that contains the whole body."""
    if isinstance(b, VeinNetwork):
        return max((body_reach_m(v) for v in b.veins), default=0.0)
    if isinstance(b, Plate):
        return math.hypot(0.5 * b.L, b.W) + b.t0
    if isinstance(b, Ellipsoid):
        return max(b.abc)
    if isinstance(b, Pipe):
        return max(b.rt, b.rb)
    return 0.0


def hex_index(data, layout) -> dict:
    """``{(col, row): [body_id, ...]}`` cached on the MineralData object."""
    cache = getattr(data, "_hex_index", None)
    if cache is not None:
        return cache
    out: dict = {}
    for bid, b in sorted(data.plan.bodies.items()):
        try:
            parts = b.hex_split(layout, n=6)
        except Exception:  # noqa: BLE001 - a degenerate body is skipped, not fatal
            continue
        for key in parts:
            out.setdefault((int(key[0]), int(key[1])), []).append(bid)
    data._hex_index = out
    return out


def section_ore(data, xs_km, ys_km, surface_m) -> list[list[dict]]:
    """Per profile point, the intervals of the bodies under it (world z, m)."""
    xs = np.asarray(xs_km, dtype=np.float64) * 1000.0
    ys = np.asarray(ys_km, dtype=np.float64) * 1000.0
    rows: list[list[dict]] = [[] for _ in xs]
    if xs.size == 0:
        return rows
    a = np.array([xs[0], ys[0]])
    b = np.array([xs[-1], ys[-1]])
    ab = b - a
    lab = float(ab @ ab) or 1.0
    sys_by = {s["system_id"]: s for s in data.plan.systems}
    for bid, body in sorted(data.plan.bodies.items()):
        o = np.asarray(body.origin_m[:2], dtype=np.float64)
        t = float(np.clip((o - a) @ ab / lab, 0.0, 1.0))
        dist = float(np.hypot(*(a + t * ab - o)))
        reach = body_reach_m(body)
        if dist > reach:
            continue
        s = sys_by.get(body.system_id, {})

        def add(i, x, y, exact, body=body, bid=bid, s=s):
            for zt, zb in body.column(float(x), float(y)):
                rows[i].append({"body_id": bid, "morphology": body.morphology,
                                "model_id": s.get("model_id", ""), "z_top_m": zt,
                                "z_bottom_m": zb, "at_profile_point": exact,
                                "principal": list(s.get("principal", []))})  # fmt: skip

        near = np.nonzero(np.hypot(xs - o[0], ys - o[1]) <= reach)[0]
        for i in near:
            add(int(i), xs[i], ys[i], True)
        if not near.size or not any(r["body_id"] == bid for i in near for r in rows[i]):
            # a body smaller than the sampling step: cut it where the line passes closest,
            # shown at the nearest profile point (it exists even between the samples)
            p = a + t * ab
            i = int(np.argmin(np.hypot(xs - p[0], ys - p[1])))
            add(i, p[0], p[1], False)
    return rows
