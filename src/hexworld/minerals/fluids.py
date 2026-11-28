"""Fuels, fluids, water and heat (§§11–12, P1).

* Petroleum system: source rock (TOC, thickness) -> reduced thermal maturity
  ``I = Σ k_ref exp[-Ea/R (1/T - 1/T_ref)] dt`` (K, s; a parameterised proxy, **not**
  Easy%Ro) -> generation bounded by the organic inventory -> traps that need a seal and must
  exist before the charge -> pore capacity, spill to the next trap, standard volumes by B.
* Coal rank from the maximum burial temperature (fresh peat is never anthracite).
* Aquifers: stored water, drainable water (Sy <= phi) and renewable recharge kept apart.
* Brines: dissolved mass = V × C, a separate basis from solid evaporite salt.
* Geothermal: heat in place (J) and the recharge heat flow (W), never mixed.
* Gas hydrates: cold sea floor with sediment, stability-zone thickness proxy.
"""

from __future__ import annotations

import math

import numpy as np

from .bodies import make_body
from .gates import RULES, evaluate
from .plan import poisson, stable_id, stable_u
from .units import (
    check_saturations,
    dissolved_mass_kg,
    drainable_water_m3,
    heat_in_place_j,
    pore_volume_m3,
    standard_volume_m3,
)

R_GAS = 8.314  # J/mol/K
SEC_PER_MA = 3.15576e13
#: reduced maturity kinetics [К]: k_ref (1/s) at T_ref (K), activation energy (J/mol)
KINETICS = {"k_ref": 1e-14, "T_ref": 373.15, "Ea": 210e3}
GEOTHERMAL_GRADIENT_C_KM = 30.0
SURFACE_T_C = 15.0
#: coal rank by maximum temperature (°C) -> (commodity, calorific MJ/kg, moisture, ash) [К]
COAL_RANKS = ((40.0, "peat", 8.0, 0.80, 0.05), (70.0, "lignite", 15.0, 0.35, 0.12),
              (100.0, "lignite", 20.0, 0.20, 0.12), (150.0, "bituminous_coal", 29.0, 0.06, 0.10),
              (1e9, "anthracite", 31.0, 0.03, 0.08))  # fmt: skip


def maturity_index(burial_m: float, duration_ma: float, steps: int = 200) -> float:
    """I for linear burial to ``burial_m`` over ``duration_ma`` (temperature from the
    geothermal gradient), trapezoid in time. Transformation ratio ``TR = 1 - exp(-I)``."""
    if duration_ma <= 0 or burial_m <= 0:
        return 0.0
    k_ref, t_ref, ea = KINETICS["k_ref"], KINETICS["T_ref"], KINETICS["Ea"]
    t = np.linspace(0.0, duration_ma, steps)
    depth = burial_m * t / duration_ma
    temp_k = SURFACE_T_C + GEOTHERMAL_GRADIENT_C_KM * depth / 1000.0 + 273.15
    rate = k_ref * np.exp(-ea / R_GAS * (1.0 / temp_k - 1.0 / t_ref))
    return float(np.trapezoid(rate, t * SEC_PER_MA))


def coal_rank(max_temp_c: float) -> dict:
    for lim, com, mj, moist, ash in COAL_RANKS:
        if max_temp_c < lim:
            return {"commodity": com, "calorific_MJ_kg": mj, "moisture": moist, "ash": ash,
                    "max_temperature_c": max_temp_c}  # fmt: skip
    raise AssertionError


def apply_coal_ranks(plan, ctx) -> None:
    """Re-label COAL lots by rank from the burial temperature (quality, same mass)."""
    for lot in plan.lots:
        if lot["model_id"] != "COAL":
            continue
        s = plan.system(lot["system_id"])
        burial = float(ctx.basin_fill_m[s["hex"]]) + s["depth_top_m"]
        t_max = SURFACE_T_C + GEOTHERMAL_GRADIENT_C_KM * burial / 1000.0
        q = coal_rank(t_max)
        old = next(iter(lot["contained"]))
        if q["commodity"] != old:
            lot["contained"] = {q["commodity"]: lot["contained"][old]}
            lot["grades"] = {q["commodity"]: lot["grades"][old]}
            # the ledger basis keeps the generic coal mass; the rank is a quality view
        lot["quality"].update(q)
        s["principal"] = [q["commodity"]]


# -- petroleum ---------------------------------------------------------------------------------
def petroleum_systems(plan, pc, lambda_scale: float = 1.0) -> list:
    """Conventional accumulations: source -> maturity -> migration -> trap (seal, timing,
    capacity, spill). Returns the accumulations (also stored as reservoir lots)."""
    ctx, F = pc.ctx, pc.F
    rule = RULES["PETROLEUM"]
    elig, _unres, S = evaluate(rule, F)
    out = []
    if not elig.any():
        plan.report["petroleum"] = {"basins": 0, "reason": "no basin with source, seal, reservoir"}
        return out
    from scipy import ndimage

    lab, k = ndimage.label(elig.reshape(ctx.shape), structure=np.ones((3, 3), bool))
    lab = lab.ravel()
    led = plan.ledger
    for p in range(1, k + 1):
        idx = np.nonzero(lab == p)[0]
        area_km2 = idx.size * pc.area_km2
        cx, cy = float(ctx.x_km[idx].mean()), float(ctx.y_km[idx].mean())
        pid = stable_id("P", ctx.seed, "PETROLEUM", round(cx / 100), round(cy / 100), p)
        burial = float(np.median(ctx.basin_fill_m[idx]))
        src_frac = np.nanmean(np.nan_to_num(F["reduced_facies"][idx]))
        toc = 0.02 + 0.03 * stable_u(ctx.seed, pid, "toc")  # mass fraction [К]
        src_thick = 50.0 + 150.0 * stable_u(ctx.seed, pid, "src_thick")
        organic_t = area_km2 * 1e6 * max(src_frac, 0.1) * src_thick * 2400.0 / 1000.0 * toc
        dur = min(pc.history_ma, 40.0) + 60.0 * stable_u(ctx.seed, pid, "basin_age")
        tr = 1.0 - math.exp(-maturity_index(burial, dur))
        hi_yield = 0.3  # kg hydrocarbons per kg organic carbon [К]
        generated_t = organic_t * hi_yield * tr
        led.open(f"gen/{pid}", f"generated:{pid}", "hydrocarbons", generated_t, "initial")
        n_traps = max(1, poisson(rule.lam_ref * lambda_scale * float(S[idx].sum()) * pc.area_km2,
                                 stable_u(ctx.seed, pid, "traps")))  # fmt: skip
        charge_ma = dur * (0.3 + 0.4 * stable_u(ctx.seed, pid, "charge"))  # before present
        remaining = generated_t * 0.1  # migration efficiency to traps [К]
        led.transfer(f"migrate/{pid}", f"generated:{pid}", f"migrating:{pid}", "hydrocarbons",
                     remaining)  # fmt: skip
        order = idx[np.argsort(-S[idx], kind="stable")][:n_traps]
        for slot, i in enumerate(order):
            sid = stable_id("D", ctx.seed, "PETROLEUM", pid, slot)
            trap_age = dur * stable_u(ctx.seed, sid, "trap_age")
            area_m2 = (2.0 + 18.0 * stable_u(ctx.seed, sid, "area")) * 1e6 * 10
            gross = area_m2 * (20.0 + 130.0 * stable_u(ctx.seed, sid, "gross"))
            ntg = 0.4 + 0.4 * stable_u(ctx.seed, sid, "ntg")
            phi = 0.08 + 0.20 * stable_u(ctx.seed, sid, "phi")
            sw = 0.25 + 0.2 * stable_u(ctx.seed, sid, "sw")
            vp = pore_volume_m3(gross, ntg, phi)
            gas_share = min(0.9, max(0.05, tr - 0.3))  # mature -> gassier
            so, sg = (1 - sw) * (1 - gas_share), (1 - sw) * gas_share
            check_saturations(sw, so, sg)
            cap_t = vp * (so * 0.80 + sg * 0.15)  # reservoir densities t/m³ (oil, gas)
            rec = {"system_id": sid, "trap_age_ma": trap_age, "charge_ma": charge_ma}
            if trap_age < charge_ma:  # formed after the charge passed: empty (T57)
                plan.rejected.append({**rec, "reason": "trap younger than the charge"})
                continue
            have = led.balance(f"migrating:{pid}", "hydrocarbons")
            fill_t = min(have, cap_t)
            if fill_t <= 0:
                continue
            led.transfer(f"fill/{sid}", f"migrating:{pid}", f"{sid}-B0-L0", "hydrocarbons", fill_t)
            frac = fill_t / cap_t
            v_oil_res, v_gas_res = vp * so * frac, vp * sg * frac
            bo, bg = 1.2, 0.004  # formation volume factors at standard 15 °C, 101.325 kPa [К]
            depth = max(burial * 0.7, 500.0)
            x, y = float(ctx.x_km[i] * 1000), float(ctx.y_km[i] * 1000)
            side = math.sqrt(area_m2)
            body = make_body("reservoir", f"{sid}-B0", sid, (x, y, ctx.surface_m[i] - depth),
                             params={"length_m": side, "down_dip_m": side,
                                     "true_thickness_m": gross / area_m2,
                                     "net_to_gross": ntg, "porosity": phi})  # fmt: skip
            plan.bodies[body.body_id] = body
            lot = {"lot_id": f"{sid}-B0-L0", "system_id": sid, "body_id": body.body_id,
                   "model_id": "PETROLEUM", "volume_m3": gross, "pore_volume_m3": vp,
                   "net_to_gross": ntg, "porosity": phi, "saturations": [sw, so, sg],
                   "oil_reservoir_m3": v_oil_res, "gas_reservoir_m3": v_gas_res,
                   "oil_standard_m3": standard_volume_m3(v_oil_res, bo),
                   "gas_standard_m3": standard_volume_m3(v_gas_res, bg),
                   "B_o": bo, "B_g": bg, "standard_conditions": "15 °C, 101.325 kPa",
                   "contained": {"oil": fill_t * (1 - gas_share),
                                 "natural_gas": fill_t * gas_share},
                   "grades": {}, "typed_grades": {}, "rock_t": 0.0,
                   "fill_fraction": frac, "capacity_t": cap_t, "gem": {}, "eroded_m3": 0.0,
                   "unit_basis": "fluid_phase_mass", "balance_owner_id": f"{sid}-B0-L0",
                   "quality": {"maturity_TR": tr, "API_proxy": 35.0 - 10 * (1 - tr)},
                   "calibration_status": "provisional"}  # fmt: skip
            plan.lots.append(lot)
            plan.systems.append({"system_id": sid, "province_id": pid, "model_id": "PETROLEUM",
                                 "title": "нефть и газ (ловушка)", "event_id": pid,
                                 "age_ma": round(charge_ma, 2), "provenance": "basin_history",
                                 "formation_depth_m": depth, "depth_top_m": depth,
                                 "exposed": False, "hex": int(i), "x_m": x, "y_m": y,
                                 "morphology": "reservoir", "strike_deg": 0.0, "dip_deg": 0.0,
                                 "strike_source": "trap", "host": "porous reservoir",
                                 "principal": ["oil", "natural_gas"],
                                 "body_ids": [body.body_id], "volume_m3": gross, "rock_t": 0.0,
                                 "calibration_status": "provisional",
                                 "discovery_state": "undiscovered",
                                 "surface_occurrence": False})  # fmt: skip
            out.append(lot)
        left = led.balance(f"migrating:{pid}", "hydrocarbons")
        if left > 0:  # spilled past the last trap or leaked through the seal
            led.remove(f"leak/{pid}", f"migrating:{pid}", "hydrocarbons", left, "lost")
    plan.report["petroleum"] = {"accumulations": len(out)}
    return out


# -- water, brines, heat, hydrates ---------------------------------------------------------------
def aquifer_units(plan, pc, recharge_mm: np.ndarray | None) -> list:
    """Regional aquifers in porous hosts: stored water, drainable water for a 10 m drawdown
    and renewable recharge (m³/yr) kept as separate quantities (T60)."""
    ctx = pc.ctx
    out = []
    for lot in [lt for lt in plan.lots if lt["model_id"] == "AQUIFER"]:
        out.append(lot)
    for s in [s for s in plan.systems if s["model_id"] == "AQUIFER"]:
        lot = next(lt for lt in plan.lots if lt["system_id"] == s["system_id"])
        body = plan.bodies[s["body_ids"][0]]
        area = body.L * body.W
        b = 30.0 + 120.0 * stable_u(ctx.seed, s["system_id"], "b")
        phi = 0.1 + 0.2 * stable_u(ctx.seed, s["system_id"], "phi")
        sy = phi * (0.4 + 0.4 * stable_u(ctx.seed, s["system_id"], "sy"))
        rech = float(np.nan_to_num(recharge_mm[s["hex"]])) if recharge_mm is not None else 0.0
        stored = area * b * phi
        lot.update({"saturated_thickness_m": b, "porosity": phi, "specific_yield": sy,
                    "water_stored_m3": stored,
                    "water_drainable_10m_m3": drainable_water_m3(area, sy, 10.0, phi),
                    "recharge_m3_per_year": area * rech / 1000.0,
                    "fossil_share": max(0.0, 1.0 - area * rech / 1000.0 * 1e4 / stored)
                    if stored > 0 else None, "unit_basis": "aquifer_water"})  # fmt: skip
    return out


def _system(plan, ctx, sid, model_id, title, i, principal, depth_m, host):
    plan.systems.append({"system_id": sid, "province_id": sid, "model_id": model_id,
                         "title": title, "event_id": sid, "age_ma": None,
                         "provenance": "inferred", "formation_depth_m": depth_m,
                         "depth_top_m": depth_m, "exposed": depth_m <= 0, "hex": int(i),
                         "x_m": float(ctx.x_km[i] * 1000), "y_m": float(ctx.y_km[i] * 1000),
                         "morphology": "reservoir", "strike_deg": 0.0, "dip_deg": 0.0,
                         "strike_source": "n/a", "host": host, "principal": principal,
                         "body_ids": [], "volume_m3": None, "rock_t": 0.0,
                         "calibration_status": "provisional", "discovery_state": "undiscovered",
                         "surface_occurrence": depth_m <= 0})  # fmt: skip


def brines(plan, pc) -> list:
    ctx, F = pc.ctx, pc.F
    elig, _u, S = evaluate(RULES["BRINE"], F)
    out = []
    for i in np.nonzero(elig)[0][:: max(1, int(elig.sum() // 6) or 1)][:6]:
        sid = stable_id("D", ctx.seed, "BRINE", int(i))
        area = (50.0 + 450.0 * stable_u(ctx.seed, sid, "a")) * 1e6
        thick = 20.0 + 80.0 * stable_u(ctx.seed, sid, "t")
        phi = 0.15 + 0.2 * stable_u(ctx.seed, sid, "p")
        vol = area * thick * phi
        conc = {
            "Li": 150 + 600 * stable_u(ctx.seed, sid, "li"),
            "K": 5000.0,
            "Mg": 8000.0,
            "B": 400.0,
        }  # mg/L [К]
        contained = {f"{k}_brine" if k != "B" else "B_brine": dissolved_mass_kg(vol, c) / 1000.0
                     for k, c in conc.items()}  # fmt: skip
        contained = {k if k != "Mg_brine" else "Mg_brine": v for k, v in contained.items()}
        for com, m in contained.items():
            plan.ledger.open(f"brine/{sid}/{com}", f"{sid}-L0", com, m, "initial")
        _system(plan, ctx, sid, "BRINE", "рассол (бессточный бассейн)", i, ["Li_brine"], 0.0,
                "closed basin sediments")  # fmt: skip
        lot = {"lot_id": f"{sid}-L0", "system_id": sid, "model_id": "BRINE",
               "brine_volume_m3": vol, "concentration_mg_L": conc, "contained": contained,
               "unit_basis": "dissolved", "note": "dissolved stock; not added to solid salt",
               "calibration_status": "provisional"}  # fmt: skip
        plan.lots.append(lot)
        out.append(lot)
    return out


def geothermal(plan, pc) -> list:
    ctx, F = pc.ctx, pc.F
    elig, _u, S = evaluate(RULES["GEOTHERMAL"], F)
    out = []
    for vs in ctx.volcano_systems[:200]:
        i = vs.get("hex_flat", vs.get("hex"))
        if i is None or not 0 <= int(i) < ctx.n or not elig[int(i)]:
            continue
        sid = stable_id("D", ctx.seed, "GEOTHERMAL", vs.get("volcano_id", i))
        if stable_u(ctx.seed, sid, "p") > 0.3:
            continue
        vol = (2.0 + 8.0 * stable_u(ctx.seed, sid, "v")) * 1e9  # m³
        t_res = 150.0 + 150.0 * stable_u(ctx.seed, sid, "T")
        q = heat_in_place_j(vol, 0.1, 2650.0, 850.0, 1000.0, 4180.0, t_res - SURFACE_T_C)
        flow_w = 0.15 * (vol ** (2.0 / 3.0)) * 50.0  # heat recharge, W [К]
        _system(plan, ctx, sid, "GEOTHERMAL", "геотермальная система", int(i),
                ["geothermal_heat"], 1000.0, "volcanic heat source")  # fmt: skip
        lot = {"lot_id": f"{sid}-L0", "system_id": sid, "model_id": "GEOTHERMAL",
               "reservoir_volume_m3": vol, "temperature_c": t_res,
               "heat_in_place_J": q, "recharge_W": flow_w, "unit_basis": "heat",
               "note": "J stock and W flow are different quantities",
               "contained": {}, "calibration_status": "provisional"}  # fmt: skip
        plan.lots.append(lot)
        out.append(lot)
    return out


def hydrates(plan, pc) -> list:
    ctx, F = pc.ctx, pc.F
    elig, _u, S = evaluate(RULES["HYDRATE"], F)
    out = []
    idx = np.nonzero(elig)[0]
    for i in idx[:: max(1, idx.size // 5)][:5] if idx.size else []:
        sid = stable_id("D", ctx.seed, "HYDRATE", int(i))
        depth = float(ctx.water_depth_m[i])
        hsz = min(600.0, max(0.0, (depth - 400.0) * 0.25))  # stability-zone thickness [К]
        if hsz <= 0:
            continue
        area = 1e9
        gas_t = area * hsz * 0.4 * 0.03 * 0.9 * 0.16  # phi, saturation, CH4 density [К]
        plan.ledger.open(f"hyd/{sid}", f"{sid}-L0", "gas_hydrate", gas_t, "initial")
        _system(plan, ctx, sid, "HYDRATE", "газовые гидраты", i, ["gas_hydrate"], 50.0,
                "sea-floor sediment")  # fmt: skip
        lot = {"lot_id": f"{sid}-L0", "system_id": sid, "model_id": "HYDRATE",
               "stability_zone_m": hsz, "water_depth_m": depth, "contained": {"gas_hydrate": gas_t},
               "unit_basis": "fluid_phase_mass", "calibration_status": "provisional"}  # fmt: skip
        plan.lots.append(lot)
        out.append(lot)
    return out
