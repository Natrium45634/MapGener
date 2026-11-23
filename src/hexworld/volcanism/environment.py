"""Gases, the sea floor and the ice: degassing, hydrothermal fields, emergence, subglacial
melt, the climate pulse and the icy-world profile.

* **Degassing stress** ``[H]``: an open, persistently degassing vent (``degassing`` phase or a
  lava lake) damages vegetation downwind within ``L = 8 km * (1 + lake)``; the share of
  the vent hex affected is ``exp(-d/L)`` integrated over the hex (fraction, never 1).
* **CO2 pooling** ``[H]``: dormant/active systems with a crater or caldera in a closed
  low (v4 ``no_outflow_mask``) are flagged - a hazard flag, not a deposit.
* **Hydrothermal fields**: active ridge and submarine systems and active land systems
  with groundwater (P > 300 mm) get a vent field. Ridge fields deeper than 1500 m are
  ``black_smoker``; shallower or off-axis ones ``diffuse_low_temperature``; on land
  ``hot_spring_fumarole``. Their heat output is ``H = 0.05 * supply * (c_p dT + L_c)``.
* **Emergence**: a submarine system whose summit + its window growth reaches the sea
  surface becomes a new island (Surtsey type); growth = erupted volume / cone area.
* **Subglacial melt**: energy-limited, ``M_melt = eff * M (c_p T + L_c) / L_f`` with
  ``eff = 0.6`` under ice (direct contact), capped by the ice over the vent; melt that
  cannot drain stays as a subglacial lake, otherwise it leaves as a jokulhlaup routed like
  a lahar without sediment bulking.
* **Climate pulse**: stratospheric SO2 only; peak global cooling
  ``dT = -0.08 K * (SO2 / Tg)^0.75`` (Pinatubo ~17 Tg -> ~-0.7 K), e-fold 1 yr. It is a
  *record* of recent pulses; the present climate is not changed by them.
* **Icy world** (``world_profile = "icy_world"``): melt is water-ammonia brine at ~-90 degC to
  0 degC; no SO2, no silicate lava; cryolava is emplaced as ice. Never active on an
  earth-like silicate world.
"""

from __future__ import annotations

import math

import numpy as np

from .lahars import route
from .units import L_FUSION_ICE, RHO_ICE, RHO_WATER

C_P = 1200.0
L_CRYST = 4.0e5
SUBGLACIAL_EFF = 0.6


def degassing_stress(system) -> float:
    """Length scale (km) of vegetation damage downwind of an open vent; 0 if none."""
    if system.environment not in ("land", "island_emergent", "ice_free_high_altitude"):
        return 0.0
    lake = system.lava_lake_state in ("persistent_open_vent", "crusted", "transient")
    if system.activity_phase != "degassing" and not lake:
        return 0.0
    return 8.0 * (2.0 if lake else 1.0)


def stress_fraction_of_hex(length_km: float, hex_radius_km: float) -> float:
    """Share of a hex of circumradius R within the exp(-d/L) plume (radially averaged)."""
    if length_km <= 0:
        return 0.0
    r = hex_radius_km
    # mean of exp(-d/L) over a disc of radius r: 2 L^2 / r^2 * (1 - (1 + r/L) e^{-r/L})
    x = r / length_km
    return float(min(1.0, 2.0 / x**2 * (1.0 - (1.0 + x) * math.exp(-x))) * 0.5)


def hydrothermal_field(system) -> dict | None:
    if system.magmatic_status != "active_system" or system.reservoir is None:
        return None
    supply = system.reservoir.input_kg_s
    heat_mw = 0.05 * supply * (C_P * 1000.0 + L_CRYST) / 1e6
    if system.environment == "deep_submarine":
        kind = (
            "black_smoker"
            if system.source_type == "mid_ocean_ridge" and system.depth_m > 1500
            else "diffuse_low_temperature"
        )
    elif system.environment == "shallow_submarine":
        kind = "shallow_hydrothermal"
    elif system.environment == "subglacial":
        kind = "subglacial_geothermal"
    else:
        kind = "hot_spring_fumarole"
    return {
        "volcano_id": system.volcano_id, "kind": kind, "heat_mw": round(heat_mw, 3),
        "x_km": system.x_km, "y_km": system.y_km, "depth_m": round(system.depth_m, 1),
    }  # fmt: skip


def emergence(system, erupted_m3: float) -> dict | None:
    """Island built by a submarine system within the window (Surtsey type), if any."""
    if system.environment not in ("shallow_submarine", "deep_submarine"):
        return None
    r = max(float(system.radius_km), 1.0) * 1000.0
    growth = 3.0 * erupted_m3 / (math.pi * r**2)  # cone of radius r: V = pi r^2 h / 3
    if system.depth_m - growth > 0:
        return None
    return {"volcano_id": system.volcano_id, "growth_m": round(growth, 1),
            "depth_before_m": round(system.depth_m, 1),
            "island_km2": round(math.pi * (0.3 * r / 1000.0) ** 2, 2)}  # fmt: skip


def subglacial_melt(project, system, event, ice_thickness_m: float) -> dict | None:
    """Energy-limited melt under ice and its outburst (jokulhlaup) or lake."""
    if event.environment != "subglacial" or ice_thickness_m <= 0:
        return None
    m = float(event.products.get("magma_mass_kg", 0.0))
    energy = SUBGLACIAL_EFF * m * (C_P * max(system.magma.temperature_c, 0.0) + L_CRYST)
    r = max(float(system.radius_km), 2.0) * 1000.0
    ice_kg = math.pi * r**2 * ice_thickness_m * RHO_ICE
    melt_kg = min(ice_kg, energy / L_FUSION_ICE)
    vol = melt_kg / RHO_WATER
    out = {"energy_j": energy, "melt_kg": melt_kg, "melt_m3": vol, "ice_available_kg": ice_kg,
           "melt_capped_by_ice": melt_kg >= ice_kg, "cauldron_depth_m": min(
               ice_thickness_m, melt_kg / RHO_ICE / (math.pi * (0.5 * r) ** 2))}  # fmt: skip
    lay = project.layout
    col, row = lay.world_to_offset(event.vent_x_km, event.vent_y_km)
    H = project.hydro.layers if project.hydro is not None else None
    if H is None or not lay.inside(col, row):
        out["outburst"] = None
        return out
    flat = int(row) * lay.width + int(col)
    drains = int(H["flow_receiver"].ravel()[flat]) != flat
    if drains and vol > 1e5:
        r_ = route(project, flat, vol)
        # Clague & Mathews (1973): Q_max = 75 (V / 10^6 m^3)^0.67
        r_["peak_discharge_m3_s"] = round(75.0 * (vol / 1e6) ** 0.67, 1)
        out["outburst"] = r_
    else:
        out["outburst"] = None
        out["subglacial_lake_m3"] = vol
    return out


def climate_pulse(event) -> dict | None:
    so2 = float(event.products.get("so2_stratospheric_kg", 0.0))
    if so2 <= 1e8:  # < 0.1 Tg: no measurable forcing
        return None
    tg = so2 / 1e9
    return {"event_id": event.event_id, "years_bp": event.years_bp, "so2_tg": round(tg, 3),
            "peak_cooling_k": round(-0.08 * tg**0.75, 3), "efold_years": 1.0,
            "applied_to_present_climate": False}  # fmt: skip


def icy_world_overrides(system) -> None:
    """Swap the silicate magma for a water-ammonia cryomagma (icy-world profile only)."""
    m = system.magma
    m.profile = "cryomagma_water_ammonia"
    m.silica_wt_pct = 0.0
    m.temperature_c = -60.0
    m.sulfur_wt_pct = 0.0
    m.co2_wt_pct = 0.5
    m.volatile_h2o_wt_pct = 0.0
    m.density_kg_m3 = 1000.0
    system.notes.append("icy_world: cryovolcanic system (brine, ice products)")


def ice_cover_proxy(project) -> np.ndarray | None:
    """Ice thickness proxy (m) per hex from the v4 glacial potential (before v5.0 ice)."""
    if project.hydro is None:
        return None
    g = np.nan_to_num(project.hydro.layers["glacial_potential"].astype(np.float64))
    return (np.clip(g, 0.0, 1.0) * 400.0).ravel()
