"""Continental margins: shelf break -> slope -> rise -> abyss, by margin type (TZ 4.6, 4.7).

The shelf is the submerged continental crust (``continentality >= 0``): it is never made by
dilating the land mask. Seaward of the crust edge the margin profile is a smooth monotone
curve in the signed distance ``u = -continentality`` (km):

    u = 0            shelf break (the v2 crust-edge height)
    u = W_slope      foot of the continental slope (``slope_foot_depth``)
    u = W_s + W_rise outer edge of the rise = the age-depth basement of the hex

drawn with monotone cubic Hermite (Fritsch-Carlson PCHIP) through those nodes, with zero
slope at both ends, so the height and its first derivative are continuous (C1).

Margin type is a property of the geology, blended smoothly so that types meet without a
seam:

* **active** - near a trench on the overriding side: narrow, steep slope straight down to
  the trench; no rise (the sediment goes into the trench);
* **transform** - near a transform plate boundary: a steep escarpment, a short rise;
* **passive** - everything else: a wide slope and a broad sediment rise.

Widths are physical kilometres (never hex counts) and vary slowly along the margin with a
world-position noise of the rock (the material frame), so a shelf stays the same when the
hex resolution changes. Numbers are Earth-like starting values [А], not norms.
"""

from __future__ import annotations

import numpy as np

PASSIVE, ACTIVE, TRANSFORM = 1, 2, 3
MARGIN_NAMES = ("none", "passive", "active", "transform")

#: (slope width km, slope-foot depth m below datum or None = basement, rise width km)
PROFILES = {
    PASSIVE: (70.0, 3000.0, 300.0),
    ACTIVE: (40.0, None, 25.0),
    TRANSFORM: (25.0, 3800.0, 80.0),
}


def smoothstep(x):
    x = np.clip(x, 0.0, 1.0)
    return x * x * (3.0 - 2.0 * x)


def margin_weights(trench_distance_km, trench_side, transform_distance_km):
    """Weights (passive, active, transform) in 0..1, summing to 1."""
    td = np.asarray(trench_distance_km, dtype=np.float64)
    near_trench = np.asarray(trench_side) != 0
    w_act = np.where(near_trench, 1.0 - smoothstep((td - 200.0) / 250.0), 0.0)
    w_tr = (1.0 - smoothstep((np.asarray(transform_distance_km) - 120.0) / 200.0)) * (1.0 - w_act)
    w_pas = np.clip(1.0 - w_act - w_tr, 0.0, 1.0)
    return w_pas, w_act, w_tr


def _hermite(u, u0, u1, y0, y1, m0, m1):
    h = np.maximum(u1 - u0, 1e-9)
    t = np.clip((u - u0) / h, 0.0, 1.0)
    t2, t3 = t * t, t * t * t
    return (
        (2 * t3 - 3 * t2 + 1) * y0
        + (t3 - 2 * t2 + t) * h * m0
        + (-2 * t3 + 3 * t2) * y1
        + (t3 - t2) * h * m1
    )


def pchip3(u, x1, x2, y0, y1, y2):
    """Monotone cubic through (0, y0), (x1, y1), (x2, y2), zero end slopes; flat outside."""
    u = np.asarray(u, dtype=np.float64)
    h0 = np.maximum(x1, 1e-9)
    h1 = np.maximum(x2 - x1, 1e-9)
    d0 = (y1 - y0) / h0
    d1 = (y2 - y1) / h1
    same = d0 * d1 > 0.0
    w1 = 2.0 * h1 + h0
    w2 = h1 + 2.0 * h0
    with np.errstate(divide="ignore", invalid="ignore"):
        m1 = np.where(same, (w1 + w2) / (w1 / d0 + w2 / d1), 0.0)
    m1 = np.nan_to_num(m1)
    first = _hermite(u, 0.0, x1, y0, y1, 0.0, m1)
    second = _hermite(u, x1, x2, y1, y2, m1, 0.0)
    return np.where(u <= 0.0, y0, np.where(u <= x1, first, np.where(u <= x2, second, y2)))


def margin_profile(u_km, z_break, z_basement, weights, variation):
    """Surface of the margin at distance ``u`` seaward of the crust edge (m, datum).

    ``weights`` = (passive, active, transform); ``variation`` (>0, ~0.6..1.5) scales the
    widths along the margin. Beyond the rise the basement ``z_basement`` is returned.
    """
    w_pas, w_act, w_tr = weights
    ws = sum(w * PROFILES[k][0] for k, w in ((PASSIVE, w_pas), (ACTIVE, w_act), (TRANSFORM, w_tr)))
    wr = sum(w * PROFILES[k][2] for k, w in ((PASSIVE, w_pas), (ACTIVE, w_act), (TRANSFORM, w_tr)))
    ws = ws * variation
    wr = wr * variation
    depth_b = -np.asarray(z_basement, dtype=np.float64)
    foot = (
        w_pas * np.minimum(PROFILES[PASSIVE][1], depth_b)
        + w_act * depth_b
        + w_tr * np.minimum(PROFILES[TRANSFORM][1], depth_b)
    )
    z_foot = np.minimum(-foot, z_break - 1.0)
    z_end = np.minimum(z_basement, z_foot)
    return pchip3(u_km, ws, ws + wr, z_break, z_foot, z_end), ws, wr


def classify_margin(weights) -> np.ndarray:
    w = np.stack(weights)
    return (np.argmax(w, axis=0) + 1).astype(np.uint8)
