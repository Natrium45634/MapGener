"""Structural events and plan housekeeping of TZ 5.5.1: folding of strata and older bodies
(T25), local invalidation by affected IDs (T68), region crops that read the global plan
(T79), and a stratigraphic column model whose surfaces keep their order.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np


@dataclass(frozen=True)
class Fold:
    """Cylindrical fold: vertical displacement ``w = A sin(2 pi u / lambda)`` with ``u`` the
    distance across the fold axis (strike from north, clockwise). Upright folds never
    overturn, so the order of the surfaces is kept (overturning needs ``overturned=True``
    and is not modelled)."""

    event_index: int
    origin_m: tuple
    axis_strike_deg: float
    wavelength_m: float
    amplitude_m: float
    overturned: bool = False

    def w(self, x_m, y_m):
        th = math.radians(self.axis_strike_deg)
        nx, ny = math.cos(th), math.sin(th)  # across the axis (x east, y south)
        u = (np.asarray(x_m) - self.origin_m[0]) * nx + (np.asarray(y_m) - self.origin_m[1]) * ny
        return self.amplitude_m * np.sin(2.0 * math.pi * u / self.wavelength_m)

    def apply(self, body):
        """Older bodies move with the strata (translated by w at their origin, the body is
        small against the wavelength); younger bodies are not touched."""
        if body.formation_event >= self.event_index:
            return body
        x, y, _z = body.origin_m
        return body.translated((0.0, 0.0, float(self.w(x, y))))


@dataclass
class Column:
    """Stratigraphic surfaces (top first) as functions z(x, y); a fold adds the same w to
    all of them, so tops stay above bottoms (T25)."""

    surfaces: list = field(default_factory=list)  # [(unit_id, fn)]
    events: list = field(default_factory=list)

    def fold(self, f: Fold) -> None:
        if f.overturned:
            raise NotImplementedError("overturned folds are not modelled (TZ 5.5.1 P2)")
        self.events.append(f)

    def z(self, unit_index: int, x, y):
        val = self.surfaces[unit_index][1](x, y)
        for f in self.events:
            val = val + f.w(x, y)
        return val

    def ordered(self, x, y) -> bool:
        zs = [self.z(k, x, y) for k in range(len(self.surfaces))]
        return all(np.all(a >= b) for a, b in zip(zs[:-1], zs[1:], strict=True))


def affected_systems(plan, layout, changed_hexes) -> list[str]:
    """Systems whose bodies cross any changed hex: only these need a recomputation when the
    local geology changes (T68); the rest of the plan stays valid."""
    from .index import body_reach_m

    changed = {(int(c), int(r)) for c, r in changed_hexes}
    if not changed:
        return []
    cx = np.array([layout.center(c, r)[0] for c, r in changed]) * 1000.0
    cy = np.array([layout.center(c, r)[1] for c, r in changed]) * 1000.0
    out = []
    for _bid, b in sorted(plan.bodies.items()):
        d = np.hypot(cx - b.origin_m[0], cy - b.origin_m[1]).min()
        if d > body_reach_m(b) + layout.spacing_km * 1000.0:
            continue
        keys = set(b.hex_split(layout, n=6))
        if keys & changed:
            out.append(b.system_id)
    return sorted(set(out))


def crop(plan, bbox_km) -> dict:
    """A region window reads the global plan: the same systems, bodies and lots inside the
    box, no new draw (T79). ``bbox_km = (xmin, ymin, xmax, ymax)``."""
    x0, y0, x1, y1 = bbox_km
    sys_in = [s for s in plan.systems
              if x0 <= s["x_m"] / 1000.0 <= x1 and y0 <= s["y_m"] / 1000.0 <= y1]  # fmt: skip
    ids = {s["system_id"] for s in sys_in}
    return {"systems": sys_in,
            "bodies": {k: b for k, b in plan.bodies.items() if b.system_id in ids},
            "lots": [lot for lot in plan.lots if lot["system_id"] in ids],
            "source": "global plan (standalone region generation not used)"}  # fmt: skip


def bury(system: dict, lots: list, thickness_m: float, event: str) -> None:
    """New cover (lava, tephra, sediment) over a system: deeper and no longer exposed; its
    metal does not change (S20)."""
    system["depth_top_m"] = float(system["depth_top_m"]) + thickness_m
    system["exposed"] = system["surface_occurrence"] = system["depth_top_m"] <= 0
    system.setdefault("events", []).append({"event": event, "burial_m": thickness_m})
    for lot in lots:
        lot.setdefault("events", []).append({"event": event, "burial_m": thickness_m})
