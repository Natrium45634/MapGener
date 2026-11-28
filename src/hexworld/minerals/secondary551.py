"""Secondary processes and qualities of TZ 5.5.1: supergene enrichment (T47), sand quality
and applications (T53), sequential evaporite precipitation (T55), and the discovery view
with an independent estimate (T76). All mass moves are ledger transfers inside the domain:
nothing is created.
"""

from __future__ import annotations

import math

from .plan import stable_normal, stable_u

# -- supergene enrichment ---------------------------------------------------------------------
#: soluble metals that move from the leached cap to the blanket below the water table
SUPERGENE_METALS = ("Cu", "Ag", "Zn", "Ni")


def supergene_enrich(plan, lot: dict, water_table_m: float, leach_fraction: float,
                     body_depth_top_m: float, body_height_m: float) -> dict:  # fmt: skip
    """Split the lot into leached cap / enrichment blanket / primary zones: the metal of the
    oxidised part above the water table moves down into a blanket of the same thickness
    below it. Contained metal is unchanged; zone masses move by ledger transfers."""
    if body_height_m <= 0:
        return {}
    above = min(max(water_table_m - body_depth_top_m, 0.0), body_height_m) / body_height_m
    if above <= 0:
        return {}
    blanket = min(above, 1.0 - above)
    primary = 1.0 - above - blanket
    zones = {}
    lid = lot["lot_id"]
    for com in SUPERGENE_METALS:
        m = float(lot["contained"].get(com, 0.0))
        if m <= 0:
            continue
        cap0 = m * above
        moved = cap0 * leach_fraction
        zones[com] = {"leached_cap_t": cap0 - moved, "enriched_blanket_t": m * blanket + moved,
                      "primary_t": m * primary}  # fmt: skip
        led = plan.ledger
        if (lid, com) in led.balances and moved > 0:
            led.transfer(f"leach/{lid}/{com}", lid, f"{lid}:blanket", com, moved)
            led.transfer(f"fix/{lid}/{com}", f"{lid}:blanket", lid, com, moved)
    rock = float(lot.get("rock_t", 0.0))
    lot["supergene"] = {
        "water_table_m": water_table_m, "leach_fraction": leach_fraction,
        "zone_rock_share": {"leached_cap": above, "enriched_blanket": blanket, "primary": primary},
        "zones": zones,
        "blanket_grade_factor": {
            com: (z["enriched_blanket_t"] / (rock * blanket)) / (lot["contained"][com] / rock)
            for com, z in zones.items() if blanket > 0 and rock > 0
        },
    }  # fmt: skip
    return zones


# -- sand quality -------------------------------------------------------------------------------
#: source lithology -> (SiO2 %, Fe2O3 %) of the sand it sheds [К]
SAND_SOURCE = {"platform_siliciclastic": (92.0, 0.3), "granite_gneiss": (75.0, 1.5),
               "granitoid_intrusion": (72.0, 1.8), "metamorphic_schist": (70.0, 3.0),
               "arc_volcanic": (60.0, 5.0), "basaltic_volcanic": (50.0, 9.0),
               "carbonate_platform": (20.0, 0.5),
               "unconsolidated_sediment": (85.0, 1.0)}  # fmt: skip
APPLICATIONS = {
    "glass_sand": {"sio2_min": 95.0, "fe2o3_max": 0.1, "d50_mm": (0.1, 0.6)},
    "concrete_sand": {"d50_mm": (0.25, 2.0), "fines_max": 0.05, "roundness_max": 0.8},
    "foundry_sand": {"sio2_min": 90.0, "d50_mm": (0.15, 0.5), "roundness_min": 0.5},
}


def sand_quality(source_lith: str, transport_km: float, energy: str, seed_key: str) -> dict:
    """Grain size by the energy of the trap (river, beach, dune), roundness and purity by the
    transport distance and the source; applications by explicit specifications."""
    sio2, fe = SAND_SOURCE.get(source_lith, (75.0, 2.0))
    # transport destroys feldspar and lithics: purity rises, iron falls (bounded)
    gain = 1.0 - math.exp(-transport_km / 300.0)
    sio2 = min(99.5, sio2 + (98.0 - sio2) * 0.6 * gain)
    fe = max(0.02, fe * (1.0 - 0.8 * gain))
    roundness = min(0.95, 0.2 + 0.6 * gain + (0.25 if energy == "dune" else 0.0))
    d50 = {"river": 0.8, "beach": 0.35, "dune": 0.18}.get(energy, 0.5)
    d50 *= math.exp(0.3 * stable_normal(seed_key, "d50"))
    fines = {"river": 0.06, "beach": 0.02, "dune": 0.01}.get(energy, 0.05)
    q = {"sio2_pct": sio2, "fe2o3_pct": fe, "d50_mm": d50, "roundness": roundness,
         "fines": fines, "energy": energy, "proxy": True}  # fmt: skip
    q["applications"] = [k for k, spec in APPLICATIONS.items() if _meets(q, spec)]
    return q


def _meets(q, spec) -> bool:
    if "sio2_min" in spec and q["sio2_pct"] < spec["sio2_min"]:
        return False
    if "fe2o3_max" in spec and q["fe2o3_pct"] > spec["fe2o3_max"]:
        return False
    lo, hi = spec.get("d50_mm", (0.0, 1e9))
    if not lo <= q["d50_mm"] <= hi:
        return False
    if "fines_max" in spec and q["fines"] > spec["fines_max"]:
        return False
    if "roundness_max" in spec and q["roundness"] > spec["roundness_max"]:
        return False
    return not ("roundness_min" in spec and q["roundness"] < spec["roundness_min"])


# -- evaporites ---------------------------------------------------------------------------------
#: precipitation order and saturation (kg/m³ of brine) [К], each needs its own ions
EVAPORITE_SEQUENCE = (("gypsum", 2.5), ("salt", 360.0), ("potash", 350.0))


def evaporate(plan, basin_id: str, dissolved_kg: dict, volume_m3: float,
              evaporated_fraction: float) -> dict:  # fmt: skip
    """Sequential precipitation as the brine shrinks: a salt precipitates only above its
    saturation and only from its own dissolved stock (no Cl -> no halite); every kilogram is
    a ledger transfer from the dissolved account to the solid bed (T55)."""
    led = plan.ledger
    remain_v = volume_m3 * (1.0 - evaporated_fraction)
    out = {}
    for salt, c_sat in EVAPORITE_SEQUENCE:
        m = float(dissolved_kg.get(salt, 0.0))
        acc = f"dissolved:{basin_id}"
        if (acc, salt) not in led.balances and m > 0:
            led.open(f"inflow/{basin_id}/{salt}", acc, salt, m, "imported")
        precip = max(0.0, m - c_sat * remain_v)
        if precip > 0:
            led.transfer(f"precip/{basin_id}/{salt}", acc, f"bed:{basin_id}", salt, precip)
        out[salt] = {"precipitated_kg": precip, "dissolved_left_kg": m - precip}
    return out


# -- discovery ------------------------------------------------------------------------------------
def discovery_view(plan, seed: str, sigma_log: float = 0.6) -> list[dict]:
    """What a surveyor sees: systems with a surface occurrence only, with an independent
    estimate (log-normal error, deterministic per system) — never the ground truth."""
    out = []
    lots = {}
    for lot in plan.lots:
        lots.setdefault(lot["system_id"], []).append(lot)
    for s in plan.systems:
        if not s.get("surface_occurrence"):
            continue
        truth = {}
        for lot in lots.get(s["system_id"], []):
            for com, m in lot.get("contained", {}).items():
                truth[com] = truth.get(com, 0.0) + float(m)
        est = {com: m * math.exp(sigma_log * stable_normal(seed, s["system_id"], com, "est")
                                 - 0.5 * sigma_log**2)
               for com, m in truth.items()}  # fmt: skip
        out.append({"system_id": s["system_id"], "model_id": s["model_id"],
                    "estimate_t": est, "confidence": "reconnaissance",
                    "found_by": "outcrop" if stable_u(seed, s["system_id"], "f") < 0.7
                    else "float / stream sediment"})  # fmt: skip
    return out
