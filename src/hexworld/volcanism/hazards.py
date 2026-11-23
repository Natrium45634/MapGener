"""Pyroclastic density currents: a hazard footprint (energy cone) kept apart from the flows.

*Hazard* ``[H]`` (Malin & Sheridan 1982 energy line)::

    reachable(x)  <=>  z_vent + H_collapse - mu * d(x) > z(x)
    mu = clip(0.35 * V_pdc^-0.06, 0.08, 0.35)      (bigger flows run farther, Hayashi & Self)

The hazard is "could be reached by a current of this event", not "was buried".

*Actual* currents: a share of the tephra (``PDC_SHARE`` by style) is routed over the
hazard footprint, preferring low ground and the vent's surroundings::

    w(x) = exp(-d / L) * (1 + max(0, z_vent - z) / 500)   on reachable cells

and distributed so that the deposited volume equals the routed volume exactly.
"""

from __future__ import annotations

import math

import numpy as np

from .lava import Lattice, ground_on_lattice

PDC_SHARE = {
    "vulcanian": 0.08,
    "dome_growth_collapse": 0.6,
    "subplinian": 0.15,
    "plinian": 0.25,
    "ultraplinian": 0.5,
}
RHO_PDC_DEPOSIT = 1400.0  # kg/m^3, ignimbrite / block-and-ash bulk


def pdc_volume_m3(event) -> float:
    share = max((PDC_SHARE.get(p, 0.0) for p in event.style_phases), default=0.0)
    return float(event.products.get("tephra_mass_kg", 0.0)) * share / RHO_PDC_DEPOSIT


def collapse_height_m(event) -> float:
    if "dome_growth_collapse" in event.style_phases:
        return 600.0
    column = float(event.products.get("plume_height_m", 0.0))
    return min(3000.0, 0.15 * column + 300.0)


def run_pdc(project, system, event, seed: str) -> dict | None:
    vol = pdc_volume_m3(event)
    if vol <= 1e4 or not event.products.get("pdc"):
        return None
    mu = float(np.clip(0.35 * vol ** (-0.06), 0.08, 0.35))
    hc = collapse_height_m(event)
    reach_km = hc / mu / 1000.0 + 2.0
    cell = 500.0 if reach_km < 40 else 1000.0
    lat = Lattice(event.vent_x_km, event.vent_y_km, int(reach_km * 1000.0 / cell) + 2, cell)
    z = ground_on_lattice(project, lat, system, seed)
    z_vent = float(z[lat.vent])
    d_m = lat.dist_km * 1000.0
    energy = z_vent + hc - mu * d_m
    reach = energy > np.maximum(z, -50.0)  # the sea stops currents within ~50 m of depth
    L = max(1.0, 0.35 * reach_km)
    w = np.where(reach, np.exp(-lat.dist_km / L) * (1.0 + np.maximum(z_vent - z, 0) / 500.0), 0)
    w[lat.vent] = max(w[lat.vent], 1e-12)
    w = w / w.sum()
    thick = vol * w / lat.area_m2
    lay = project.layout
    col, row = lay.world_to_offset(lat.x, lat.y)
    col, row = np.asarray(col), np.asarray(row)
    inside = lay.inside(col, row)
    hex_area = lay.hex_area_km2 * 1e6

    def agg(mask, weights=None):
        f = row[mask] * lay.width + col[mask]
        u, inv = np.unique(f, return_inverse=True)
        wts = np.ones(inv.size) if weights is None else weights[mask]
        return u.astype(np.int64), np.bincount(inv, weights=wts, minlength=u.size)

    hz_i, hz_n = agg(inside & reach)
    deposit = thick > 0.05
    dp_i, dp_v = agg(inside & deposit, thick * lat.area_m2)
    _, dp_n = agg(inside & deposit)
    return {
        "hazard_hex": hz_i,
        "hazard_fraction": np.minimum(hz_n * lat.area_m2 / hex_area, 1.0),
        "deposit_hex": dp_i,
        "deposit_fraction": np.minimum(dp_n * lat.area_m2 / hex_area, 1.0),
        "deposit_volume_m3": dp_v,
        "volume_m3": vol,
        "placed_m3": float((thick * lat.area_m2).sum()),
        "heim_mu": mu,
        "collapse_height_m": hc,
        "hazard_area_km2": float(reach.sum() * lat.area_m2 / 1e6),
        "runout_km": float(lat.dist_km[deposit].max()) if deposit.any() else 0.0,
        "quality": "energy_cone_hazard+weighted_routing",
    }


def energy_cone_reach_km(volume_m3: float, collapse_m: float) -> float:
    """Flat-ground runout of the energy line (for tests and the hex report)."""
    mu = min(0.35, max(0.08, 0.35 * max(volume_m3, 1.0) ** (-0.06)))
    return collapse_m / mu / 1000.0 if math.isfinite(mu) else 0.0
