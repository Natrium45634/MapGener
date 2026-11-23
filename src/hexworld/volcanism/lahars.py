"""Lahars and glacial outburst floods along the v4 drainage (water AND sediment required).

A lahar is a water-sediment slurry: no water source, or no loose sediment, means no
lahar. Volume ``[H]``::

    W = melt of snow/ice by the event (energy-limited) + one ~30 mm storm on the edifice
    S = loose fresh tephra on the edifice (+ PDC deposit)
    V = min(W / (1 - c), S / c),   c = 0.55 (bulking to a debris flow)

Routing: down ``flow_receiver`` of the v4 hydrology; on each hex the flow leaves the
deposit ``A_dep * h_dep`` where it spreads (``A_inund = 200 V^(2/3)``, Iverson et al.
1998), so the routed volume is conserved: ``V = sum deposits + volume reaching the sea``.
"""

from __future__ import annotations

import numpy as np

from .units import L_FUSION_ICE, RHO_WATER

CONCENTRATION = 0.55
C_P_MAGMA = 1200.0  # J/(kg K)
L_CRYST = 4.0e5  # J/kg latent heat of crystallisation
HEAT_TO_ICE = 0.15  # share of the heat of the erupted mass that reaches snow/ice [H]


def melt_water_kg(magma_kg: float, t_c: float, ice_available_kg: float,
                  efficiency: float = HEAT_TO_ICE) -> float:  # fmt: skip
    """Energy-limited melt: ``eff * M (c_p T + L_c) / L_f``, capped by the ice there."""
    energy = efficiency * magma_kg * (C_P_MAGMA * max(t_c, 0.0) + L_CRYST)
    return float(min(max(ice_available_kg, 0.0), energy / L_FUSION_ICE))


def lahar_volume_m3(water_m3: float, sediment_m3: float, c: float = CONCENTRATION) -> float:
    if water_m3 <= 0 or sediment_m3 <= 0:
        return 0.0
    return float(min(water_m3 / (1.0 - c), sediment_m3 / c))


def route(project, start_flat: int, volume_m3: float, max_hexes: int = 80) -> dict:
    """Walk the receivers from ``start_flat`` and leave the lahar deposit on the way."""
    H = project.hydro.layers
    recv = H["flow_receiver"].ravel()
    ocean = H["ocean_mask"].ravel()
    area_hex = project.layout.hex_area_km2 * 1e6
    inund = 200.0 * volume_m3 ** (2.0 / 3.0)  # m^2 for the whole flow
    left = float(volume_m3)
    path, vols, fracs = [], [], []
    i = int(start_flat)
    seen = set()
    # spread the inundated area along the path: half in the first hex, then thinning
    for step in range(max_hexes):
        if i < 0 or i in seen or left <= volume_m3 * 1e-6:
            break
        seen.add(i)
        if ocean[i]:
            break
        share = 0.5 if step == 0 else 0.35
        dep = left * share
        a = min(area_hex, inund * share)
        path.append(i)
        vols.append(dep)
        fracs.append(a / area_hex)
        left -= dep
        nxt = int(recv[i])
        if nxt == i:
            break
        i = nxt
    return {
        "hex_index": np.asarray(path, dtype=np.int64),
        "deposit_volume_m3": np.asarray(vols),
        "hex_fraction": np.asarray(fracs),
        "volume_m3": float(volume_m3),
        "to_sea_or_lake_m3": left,
        "budget_residual_m3": float(volume_m3) - float(np.sum(vols)) - left,
        "runout_hexes": len(path),
    }


def event_lahar(project, system, event, ice_share: float, precip_mm: float,
                pdc_m3: float = 0.0) -> dict | None:  # fmt: skip
    """Lahar of one explicit event (``None`` without water or without sediment)."""
    if project.hydro is None or event.environment in ("deep_submarine", "shallow_submarine"):
        return None
    p = event.products
    rad_m = max(float(system.radius_km), 2.0) * 1000.0
    edifice_area = np.pi * rad_m**2
    # loose sediment: the tephra that falls on the edifice (~25 % of it) + PDC deposit
    tephra_bulk = float(p.get("tephra_bulk_volume_m3", 0.0))
    sediment = 0.25 * tephra_bulk + pdc_m3
    # water: snow and ice on the edifice (30 m w.e. under ice cover), rain of a wet season
    ice_kg = ice_share * edifice_area * 30.0 * RHO_WATER
    melt = melt_water_kg(float(p.get("magma_mass_kg", 0.0)), system.magma.temperature_c, ice_kg)
    # rain: one intense storm (~30 mm) on the bare edifice in a wet climate, less when dry
    rain_m3 = 0.03 * min(1.0, max(precip_mm, 0.0) / 800.0) * edifice_area
    water = melt / RHO_WATER + rain_m3
    vol = lahar_volume_m3(water, sediment)
    if vol < 1e5:
        return None
    lay = project.layout
    col, row = lay.world_to_offset(event.vent_x_km, event.vent_y_km)
    if not lay.inside(col, row):
        return None
    out = route(project, int(row) * lay.width + int(col), vol)
    out.update({"water_m3": water, "melt_water_m3": melt / RHO_WATER, "sediment_m3": sediment,
                "quality": "receiver_routing_bulked"})  # fmt: skip
    return out
