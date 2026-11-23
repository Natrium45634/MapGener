"""Tephra fall: conservative advection-diffusion-settling on the faces of the world hexes.

State: airborne mass ``m[b, i]`` (kg) of grain bin ``b`` over hex ``i`` of a window
around the vent. One step ``dt``::

    upwind flux over face k:  F_ik = m_i * max(0, u_b . n_k) * dt * L / A
    diffusion over face k:    D_ik = K * dt * L / (s A) * (m_i - m_j)^+
    settling to the ground:   S_i  = m_i * (1 - exp(-dt / t_fall,b))

``L`` is the face length, ``A`` the hex area, ``s`` the centre spacing. Every exchange is
a transfer between two hexes (``np.add.at``) and ``dt`` obeys a CFL bound, so mass is
conserved and stays non-negative; what crosses the window edge is ``exported_kg``; what is
still aloft at the end of the integration is ``airborne_kg`` (very fine ash, far field).
``source = deposited + exported + airborne`` to round-off.

Winds: bin ``b`` is released at a fraction of the column height and moves with a blend of
the v4 surface wind and a *synthetic* zonal upper wind (flag ``synthetic_upper_wind``): the
generator has no upper-air circulation.
"""

from __future__ import annotations

import math

import numpy as np

from .. import hexgrid
from ..stages.raster import neighbor_index
from .units import stable_uniform

#: (name, diameter mm, settling speed m/s, release height as a share of the column) [H].
BINS = (
    ("lapilli", 8.0, 8.0, 0.35),
    ("coarse_ash", 1.0, 2.0, 0.55),
    ("fine_ash", 0.1, 0.35, 0.8),  # aggregated fine ash
    ("very_fine_ash", 0.02, 0.03, 0.9),
)
#: Mass shares of the bins by style [H].
SHARES = {
    "strombolian": (0.6, 0.3, 0.08, 0.02),
    "vulcanian": (0.3, 0.4, 0.2, 0.1),
    "dome_growth_collapse": (0.25, 0.4, 0.25, 0.1),
    "subplinian": (0.25, 0.4, 0.25, 0.1),
    "plinian": (0.15, 0.35, 0.3, 0.2),
    "ultraplinian": (0.1, 0.3, 0.35, 0.25),
    "phreatomagmatic_surtseyan": (0.1, 0.3, 0.4, 0.2),
    "phreatic": (0.2, 0.3, 0.3, 0.2),
    "hawaiian_effusive": (0.7, 0.25, 0.05, 0.0),
    "fissure_effusive": (0.7, 0.25, 0.05, 0.0),
}
DIFFUSIVITY_M2_S = 3.0e3
QUALITY_HOURS = {"fast": 8.0, "balanced": 24.0, "high": 72.0}
QUALITY_WINDOW_KM = {"fast": 350.0, "balanced": 600.0, "high": 1500.0}


def upper_wind_ms(lat_deg: np.ndarray) -> np.ndarray:
    """``[H]`` zonal wind at ~10 km: westerly jet near 40 deg, weak tropical easterlies."""
    a = np.abs(lat_deg)
    return 25.0 * np.exp(-(((a - 40.0) / 15.0) ** 2)) - 6.0 * np.exp(-((lat_deg / 15.0) ** 2))


def bin_shares(phases, n_bins: int) -> np.ndarray:
    s = np.zeros(len(BINS))
    for p in phases:
        s += np.asarray(SHARES.get(p, SHARES["vulcanian"]))
    s /= max(len(phases), 1)
    if n_bins < len(BINS):  # merge the finest bins into the last kept one
        s = np.concatenate([s[: n_bins - 1], [s[n_bins - 1 :].sum()]])
    return s / s.sum()


class Window:
    """A rectangle of world hexes around the vent (even first column keeps odd-q parity)."""

    def __init__(self, layout, col: int, row: int, half_cols: int, half_rows: int):
        c0 = max(0, col - half_cols)
        c0 -= c0 % 2
        c1 = min(layout.width, col + half_cols + 1)
        r0 = max(0, row - half_rows)
        r1 = min(layout.height, row + half_rows + 1)
        self.c0, self.c1, self.r0, self.r1 = c0, c1, r0, r1
        cols = np.arange(c0, c1)
        rows = np.arange(r0, r1)
        cc, rr = np.meshgrid(cols, rows)
        self.flat = (rr * layout.width + cc).ravel()
        local = np.full(layout.width * layout.height, -1, dtype=np.int64)
        local[self.flat] = np.arange(self.flat.size)
        self.nb = np.full((self.flat.size, 6), -1, dtype=np.int64)
        for k in range(6):
            idx, valid = neighbor_index(layout, k)
            g = np.where(valid.ravel(), idx.ravel(), -1)[self.flat]
            self.nb[:, k] = np.where(g >= 0, local[np.maximum(g, 0)], -1)
        self.size = self.flat.size


def transport(project, event, n_bins: int, quality: str) -> dict | None:
    """Fall deposit of one explicit explosive event (kg per world hex and bin)."""
    p = event.products
    mass = float(p.get("tephra_mass_kg", 0.0))
    column = float(p.get("plume_height_m", 0.0))
    if mass <= 0 or column <= 0 or project.hydro is None:
        return None
    lay = project.layout
    H = project.hydro.layers
    col, row = lay.world_to_offset(event.vent_x_km, event.vent_y_km)
    col, row = int(col), int(row)
    if not lay.inside(col, row):
        return None
    s_m = lay.spacing_km * 1000.0
    reach_km = QUALITY_WINDOW_KM.get(quality, 900.0)
    half_c = int(math.ceil(reach_km / (1.5 * lay.radius)))
    half_r = int(math.ceil(reach_km / lay.spacing_km))
    win = Window(lay, col, row, half_c, half_r)
    u_s = np.nan_to_num(H["wind_u_ms"].ravel()[win.flat].astype(np.float64))
    v_s = np.nan_to_num(H["wind_v_ms"].ravel()[win.flat].astype(np.float64))
    lat = H["latitude_deg"].ravel()[win.flat].astype(np.float64)
    u_up = upper_wind_ms(lat)
    shares = bin_shares(event.style_phases, n_bins)
    nb_ = len(shares)
    names = [b[0] for b in BINS[: nb_ - 1]] + [BINS[nb_ - 1][0]]
    vs = np.array([BINS[i][2] for i in range(nb_)])
    rel = np.array([BINS[i][3] for i in range(nb_)]) * column
    t_fall = rel / vs
    earth = getattr(project, "earth", None)
    wind_flag = "synthetic_upper_wind"
    if earth is not None and "winds_aloft_u_ms" in earth.layers:
        # the climate's own winds aloft (v5.3 thermal wind) at each bin's release height,
        # for the season of the event (deterministic from its id)
        from ..earth.winds import wind_at_height

        season = int(stable_uniform(event.event_id, "season") * 4) % 4
        E = earth.layers
        ua = E["winds_aloft_u_ms"][season].reshape(E["winds_aloft_u_ms"].shape[1], -1)
        va = E["winds_aloft_v_ms"][season].reshape(ua.shape)
        za = E["winds_aloft_height_m"][season].reshape(ua.shape)
        ua, va, za = (np.asarray(a[:, win.flat], dtype=np.float64) for a in (ua, va, za))
        u = np.empty((nb_, win.size))
        v = np.empty((nb_, win.size))
        for b in range(nb_):
            u[b], v[b] = wind_at_height(ua, va, za, u_s, v_s, rel[b])
        wind_flag = "thermal_wind_aloft_v53"
        if column > float(np.nanmax(za)):
            wind_flag += "+extrapolated_above_top_level"
    else:
        w = np.clip(rel / 10000.0, 0.0, 1.0)  # weight of the upper wind
        u = (1 - w)[:, None] * u_s[None, :] + w[:, None] * (u_s + u_up)[None, :]
        v = (1 - w)[:, None] * v_s[None, :] + w[:, None] * v_s[None, :]
    dirs = hexgrid.direction_unit_vectors()  # (6, 2) towards each neighbour
    un = u[:, :, None] * dirs[None, None, :, 0] + v[:, :, None] * dirs[None, None, :, 1]
    un = np.maximum(un, 0.0)  # outward normal speed per face (b, i, k)
    face = lay.radius * 1000.0  # face length of a hexagon = circumradius
    area = lay.hex_area_km2 * 1e6
    diff_c = DIFFUSIVITY_M2_S * face / (s_m * area)
    # CFL: the summed outflow share of any hex per step stays below 0.8
    out_rate = float((un.sum(axis=2)).max()) * face / area + 6 * diff_c
    dt = 0.8 / max(out_rate, 1e-12)
    t_end = QUALITY_HOURS.get(quality, 36.0) * 3600.0
    steps = int(min(4000, math.ceil(t_end / dt)))
    dt = t_end / max(steps, 1)
    m = np.zeros((nb_, win.size))
    vent_local = int(np.nonzero(win.flat == row * lay.width + col)[0][0])
    m[:, vent_local] = mass * shares
    dep = np.zeros_like(m)
    exported = 0.0
    nb = win.nb
    valid = nb >= 0
    nbc = np.where(valid, nb, 0)
    adv = un * (dt * face / area)  # share of the hex mass over each face per step
    settle = 1.0 - np.exp(-dt / t_fall)
    for _ in range(steps):
        tot_air = m.sum()
        if tot_air < mass * 1e-6:
            break
        out_adv = m[:, :, None] * adv  # (b, i, k)
        mj = np.where(valid[None], m[:, nbc], 0.0)
        out_dif = np.where(valid[None], diff_c * dt * np.maximum(m[:, :, None] - mj, 0.0), 0.0)
        out = out_adv + out_dif
        s = out.sum(axis=2)
        scale = np.where(s > 0.9 * m, 0.9 * m / np.maximum(s, 1e-300), 1.0)
        out *= scale[:, :, None]
        m -= out.sum(axis=2)
        exported += float(np.where(~valid[None], out, 0.0).sum())
        dst = nbc[valid]
        for b in range(nb_):
            m[b] += np.bincount(dst, weights=out[b][valid], minlength=win.size)
        d = m * settle[:, None]
        dep += d
        m -= d
    airborne = float(m.sum())
    deposited = float(dep.sum())
    load = dep.sum(axis=0) / area  # kg/m^2
    keep = load > 1e-3  # > 1 g/m^2
    return {
        "hex_index": win.flat[keep].astype(np.int64),
        "load_kg_m2": load[keep],
        "bin_kg": {names[b]: float(dep[b].sum()) for b in range(nb_)},
        "source_kg": mass,
        "deposited_kg": deposited,
        "exported_kg": exported,
        "airborne_kg": airborne,
        "budget_residual_kg": mass - deposited - exported - airborne,
        "steps": steps,
        "dt_s": dt,
        "window_hexes": win.size,
        "area_over_1mm_km2": float((load > 1.0).sum() * lay.hex_area_km2),
        "quality_flags": [wind_flag, "approximated_column_release"],
    }


def fast_deposit(project, x_km, y_km, mass_kg, column_m) -> dict | None:
    """Analytic fall field of an aggregated history (no transport solve).

    An elongated exponential thinning downwind of the vent, normalised to ``mass_kg`` on
    the hexes it touches (conservative by construction). Used for the many small,
    non-explicit eruptions and labelled ``fast_analytic``.
    """
    if mass_kg <= 0 or project.hydro is None:
        return None
    lay = project.layout
    H = project.hydro.layers
    col, row = lay.world_to_offset(x_km, y_km)
    col, row = int(col), int(row)
    if not lay.inside(col, row):
        return None
    win = Window(lay, col, row, 8, 8)
    cx, cy = lay.center(win.flat % lay.width, win.flat // lay.width)
    i0 = row * lay.width + col
    u = float(np.nan_to_num(H["wind_u_ms"].ravel()[i0])) + float(
        upper_wind_ms(np.asarray(H["latitude_deg"].ravel()[i0], dtype=np.float64))
    )
    v = float(np.nan_to_num(H["wind_v_ms"].ravel()[i0]))
    sp = math.hypot(u, v) or 1.0
    ex, ey = u / sp, v / sp
    dx, dy = cx - x_km, cy - y_km
    along = dx * ex + dy * ey
    cross = -dx * ey + dy * ex
    L = max(10.0, 3.0 * column_m / 1000.0)  # km
    wgt = np.exp(-np.maximum(along, 0.0) / L - np.maximum(-along, 0.0) / (0.2 * L))
    wgt *= np.exp(-((cross / (0.35 * L)) ** 2))
    wgt[win.flat == i0] += 1.0
    wgt /= wgt.sum()
    area = lay.hex_area_km2 * 1e6
    load = mass_kg * wgt / area
    keep = load > 1e-3
    return {"hex_index": win.flat[keep].astype(np.int64), "load_kg_m2": load[keep],
            "source_kg": float(mass_kg), "deposited_kg": float((load * area).sum()),
            "quality_flags": ["fast_analytic"]}  # fmt: skip
