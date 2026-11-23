"""Lava flows on a local D6 hex lattice around the vent (mass-conserving, APPROXIMATE).

A 25 km world hex cannot hold a lava flow, so each explicit effusive event gets its own
flat-top hex lattice of ``lava_cell_m`` cells around the vent. The ground of the lattice is
the world elevation interpolated between hex centres plus the edifice cone, plus a small
deterministic roughness that lets flows channelise.

Scheme ``[H]`` (cellular finite volume, MAGFLOW / SCIARA family)::

    mobile_i   = max(0, h_i - h_res(d_i))                  # residual thickness ~ yield
    out_i      = min(mobile_i, 0.5 * max_k dH_ik)            # never more than levels heads
    flux_ik    = out_i * dH_ik^+ / sum_k dH_ik^+             # to lower neighbours only
    h_res(d)   = h_res0 * exp(d / L_cool)                    # cooling-limited length
    L_cool(km) = 1.6 * Q^0.5 (Q: effusion rate, m^3/s)       # Etna 5 km, Laki ~50 km [H]

Every exchange is a transfer between two cells (``np.add.at``), so the volume is conserved
to round-off; lava pushed out of the lattice edge is counted as ``escaped_m3``. Results go
to the world hexes as covered *fractions* and volumes: a small flow never repaints a hex.
"""

from __future__ import annotations

import math

import numpy as np

from .. import hexgrid
from .units import stable_uniform

#: Residual (yield) thickness by magma class, m [H] (pahoehoe ~1 m, aa ~3-5 m, andesite ~15).
H_RES = {
    "basaltic": 1.5,
    "basaltic_andesitic": 4.0,
    "andesitic": 12.0,
    "dacitic": 30.0,
    "rhyolitic": 60.0,
}
#: Cooling length factor by setting (water and ice quench the flow front).
COOL_FACTOR = {
    "deep_submarine": 0.25,
    "shallow_submarine": 0.35,
    "subglacial": 0.3,
}
SEA_QUENCH = 0.4  # length factor once a subaerial flow enters the sea


def cooling_length_km(effusion_m3_s: float, environment: str) -> float:
    q = max(float(effusion_m3_s), 0.01)
    return 1.6 * math.sqrt(q) * COOL_FACTOR.get(environment, 1.0)


class Lattice:
    """Flat-top axial hex lattice: centres, neighbour table, edge cells."""

    def __init__(self, x_km: float, y_km: float, n_rings: int, cell_m: float):
        n = int(max(2, n_rings))
        qs, rs = [], []
        for q in range(-n, n + 1):
            for r in range(max(-n, -q - n), min(n, -q + n) + 1):
                qs.append(q)
                rs.append(r)
        self.q = np.asarray(qs, dtype=np.int64)
        self.r = np.asarray(rs, dtype=np.int64)
        self.n_rings = n
        self.cell_m = float(cell_m)
        radius_km = self.cell_m / hexgrid.SQRT3 / 1000.0  # circumradius of a cell
        px, py = hexgrid.axial_to_plane(self.q, self.r, radius_km)
        self.x = x_km + px
        self.y = y_km + py
        self.area_m2 = 1.5 * hexgrid.SQRT3 * (radius_km * 1000.0) ** 2
        span = 2 * n + 3
        key = (self.q + n + 1) * span + (self.r + n + 1)
        order = np.argsort(key)
        skey = key[order]
        dirs = ((1, 0), (1, -1), (0, -1), (-1, 0), (-1, 1), (0, 1))
        self.nb = np.full((self.q.size, 6), -1, dtype=np.int64)
        for k, (dq, dr) in enumerate(dirs):
            kk = (self.q + dq + n + 1) * span + (self.r + dr + n + 1)
            pos = np.clip(np.searchsorted(skey, kk), 0, skey.size - 1)
            hit = skey[pos] == kk
            self.nb[:, k] = np.where(hit, order[pos], -1)
        self.edge = (self.nb < 0).any(axis=1)
        self.dist_km = np.hypot(px, py)
        self.vent = int(np.nonzero((self.q == 0) & (self.r == 0))[0][0])

    @property
    def size(self) -> int:
        return int(self.q.size)


def ground_on_lattice(project, lat: Lattice, system, seed: str) -> np.ndarray:
    """World elevation (inverse-distance between hex centres) + edifice cone + roughness."""
    lay = project.layout
    z_hex = project.layer("elevation_m").astype(np.float64)
    col, row = lay.world_to_offset(lat.x, lat.y)
    col = np.clip(np.asarray(col), 0, lay.width - 1)
    row = np.clip(np.asarray(row), 0, lay.height - 1)
    # the hex and its six neighbours, weighted by inverse squared distance
    acc = np.zeros(lat.size)
    wsum = np.zeros(lat.size)
    cand = [(col, row)] + [hexgrid.offset_neighbor(col, row, k) for k in range(6)]
    for c, r in cand:
        c = np.clip(np.asarray(c), 0, lay.width - 1)
        r = np.clip(np.asarray(r), 0, lay.height - 1)
        cx, cy = lay.center(c, r)
        d2 = (lat.x - cx) ** 2 + (lat.y - cy) ** 2 + 1e-6
        w = 1.0 / d2
        acc += w * z_hex[r, c]
        wsum += w
    z = acc / wsum
    sea = project.hydro.sea_level_m if project.hydro is not None else 0.0
    z_vent = float(z[lat.vent])
    cone_h = max(0.0, float(system.summit_m) + sea - z_vent)
    rad = max(float(system.radius_km), 1.0)
    z += cone_h * np.clip(1.0 - lat.dist_km / rad, 0.0, 1.0) ** 1.5
    # roughness: a few deterministic waves, amplitude ~ 0.6 % of the cone + 4 m
    amp = 4.0 + 0.006 * cone_h
    for i in range(4):
        a = stable_uniform(seed, system.volcano_id, "rough", i, "a") * math.pi * 2
        k = 0.5 + 3.0 * stable_uniform(seed, system.volcano_id, "rough", i, "k")  # 1/km
        ph = stable_uniform(seed, system.volcano_id, "rough", i, "p") * math.pi * 2
        z += amp / 2.0 * np.sin(k * (lat.x * math.cos(a) + lat.y * math.sin(a)) + ph)
    return z - sea  # relative to sea level


def simulate(
    z: np.ndarray,
    lat: Lattice,
    volume_m3: float,
    h_res0: float,
    cool_km: float,
    pulses: int = 60,
    relax: int = 20,
) -> dict:
    """Emplace ``volume_m3`` at the vent of ``lat``; returns thickness and the budget."""
    h = np.zeros(lat.size)
    nb = lat.nb
    valid = nb >= 0
    nbc = np.where(valid, nb, 0)
    hres = h_res0 * np.exp(np.minimum(lat.dist_km / max(cool_km, 0.05), 30.0))
    hres_sea = hres / SEA_QUENCH  # quench below sea level
    escaped = 0.0
    injected = 0.0
    pulse = volume_m3 / lat.area_m2 / pulses  # thickness per pulse
    for _ in range(pulses):
        h[lat.vent] += pulse
        injected += pulse
        for _ in range(relax):
            surf = z + h
            under = surf < 0.0
            res = np.where(under, hres_sea, hres)
            mobile = np.maximum(h - res, 0.0)
            if not mobile.any():
                break
            dh = surf[:, None] - np.where(valid, surf[nbc], np.inf)
            dh = np.where(valid, np.maximum(dh, 0.0), 0.0)
            # cells on the lattice edge drain outward like a lower neighbour
            dh_edge = np.where(lat.edge, mobile, 0.0)
            tot = dh.sum(axis=1) + dh_edge
            out = np.minimum(mobile, 0.5 * np.maximum(dh.max(axis=1), dh_edge))
            out = np.where(tot > 1e-12, out, 0.0)
            if out.max() < 1e-4:
                break
            share = np.where(tot[:, None] > 0, dh / np.maximum(tot[:, None], 1e-30), 0.0)
            flux = out[:, None] * share
            h -= out
            np.add.at(h, nbc[valid], flux[valid])
            lost = out * np.where(tot > 0, dh_edge / np.maximum(tot, 1e-30), 0.0)
            escaped += float(lost.sum())
    placed = float(h.sum())
    residual = injected - placed - escaped
    return {
        "thickness_m": h,
        "injected_m3": injected * lat.area_m2,
        "placed_m3": placed * lat.area_m2,
        "escaped_m3": escaped * lat.area_m2,
        "budget_residual_m3": residual * lat.area_m2,
    }


def rings_for(volume_m3: float, h_res0: float, cell_m: float, cool_km: float) -> int:
    """Lattice size: enough cells for the flow area, but no wider than the cooling reach."""
    area = volume_m3 / max(h_res0 * 1.5, 0.5)
    cells = area / (1.5 * hexgrid.SQRT3 * (cell_m / hexgrid.SQRT3) ** 2)
    by_area = 2.0 * math.sqrt(max(cells, 1.0) / 3.0)
    by_reach = 4.0 * cool_km * 1000.0 / cell_m
    return int(np.clip(min(max(by_area, 8.0), max(by_reach, 8.0)) + 4, 8, 260))


QUALITY_PULSES = {"fast": (30, 12), "balanced": (60, 20), "high": (150, 30)}


def run_flow(project, system, event, seed: str, cell_m: float,
             quality: str = "balanced") -> dict | None:  # fmt: skip
    """Simulate the lava of one explicit event and aggregate it to the world hexes."""
    vol = float(event.products.get("lava_volume_m3", 0.0))
    if vol < 1e5 or project.hydro is None:
        return None
    cls = system.magma.magma_class
    h0 = H_RES.get(cls, 4.0)
    q = vol / max(event.products.get("duration_s", 1.0), 1.0)
    env = event.environment
    cool = cooling_length_km(q, env)
    # coarser cells for big flows keep the lattice small (flood basalts, ridge sheets)
    cell = float(cell_m)
    n = rings_for(vol, h0, cell, cool)
    while n >= 260 and cell < 3200:
        cell *= 2.0
        n = rings_for(vol, h0, cell, cool)
    lat = Lattice(event.vent_x_km, event.vent_y_km, n, cell)
    z = ground_on_lattice(project, lat, system, seed)
    pulses, relax = QUALITY_PULSES.get(quality, QUALITY_PULSES["balanced"])
    sim = simulate(z, lat, vol, h0, cool, pulses, relax)
    h = sim["thickness_m"]
    covered = h > 0.05
    lay = project.layout
    col, row = lay.world_to_offset(lat.x, lat.y)
    col, row = np.asarray(col), np.asarray(row)
    inside = lay.inside(col, row) & covered
    hex_area_m2 = lay.hex_area_km2 * 1e6
    flat = row[inside] * lay.width + col[inside]
    uniq, inv = np.unique(flat, return_inverse=True)
    frac = np.bincount(inv, minlength=uniq.size) * lat.area_m2 / hex_area_m2
    vol_hex = np.bincount(inv, weights=h[inside] * lat.area_m2, minlength=uniq.size)
    new_land = covered & (z < 0) & (z + h >= 0)
    nl_flat = row[inside & new_land] * lay.width + col[inside & new_land]
    nl_u, nl_c = np.unique(nl_flat, return_counts=True)
    length = float(lat.dist_km[covered].max()) if covered.any() else 0.0
    return {
        "hex_index": uniq.astype(np.int64),
        "hex_fraction": np.minimum(frac, 1.0),
        "hex_volume_m3": vol_hex,
        "new_land_hex": nl_u.astype(np.int64),
        "new_land_km2": nl_c * lat.area_m2 / 1e6,
        "area_km2": float(covered.sum() * lat.area_m2 / 1e6),
        "length_km": length,
        "max_thickness_m": float(h.max()),
        "cell_m": cell,
        "lattice_cells": lat.size,
        "effusion_m3_s": q,
        "cooling_length_km": cool,
        "injected_m3": sim["injected_m3"],
        "placed_m3": sim["placed_m3"],
        "escaped_m3": sim["escaped_m3"],
        "budget_residual_m3": sim["budget_residual_m3"],
        "rheology": {"magma_class": cls, "residual_thickness_m": h0},
        "quality": "approximated_cellular_d6",
    }
