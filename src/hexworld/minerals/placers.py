"""Source-to-sink placers (§9, R05): parent ore -> weathering / erosion -> transport along the
directed drainage graph -> hydraulic trapping -> placer horizon (or dispersal, coast, export).

Only mass that left a parent body (the ledger's ``mobile:*`` accounts) can travel; without a
source there is no placer (S10, T42). Every step is a ledger transfer: what is trapped in a
reach becomes a placer lot (above a minimum) or a dispersed sub-economic store, what reaches
the sea is a coastal placer (where waves sort it) or exported; nothing is created and the
5.2 gap (deposited mass booked nowhere) is closed.
"""

from __future__ import annotations

import math

import numpy as np

from .bodies import make_body
from .plan import PLACER_COMMODITIES, stable_id

#: density g/cm³ and survivability of the travelling grains [К]: trap efficiency grows with
#: the density contrast to quartz (2.65), abrasion loss per 100 km for soft minerals
GRAIN = {"Au": (19.3, 0.0), "Pt": (21.0, 0.0), "Pd": (12.0, 0.0), "Sn": (6.9, 0.02),
         "W": (7.3, 0.05), "diamond": (3.52, 0.0), "corundum": (4.0, 0.01),
         "garnet": (4.0, 0.08), "Ti": (4.7, 0.02), "Zr": (4.6, 0.01), "Ce": (5.1, 0.03),
         "La": (5.1, 0.03), "Nd": (5.1, 0.03)}  # fmt: skip
#: minimum contained mass (t) of a placer lot [К]; less stays a dispersed store
MIN_LOT_T = {"Au": 0.05, "Pt": 0.05, "Pd": 0.05, "diamond": 2e-4, "corundum": 0.5}


def trap_efficiency(density: float, slope_m_km: float, lake: bool) -> float:
    """η in [0, 1): denser grains settle more, a flattening reach traps more (deceleration),
    a lake traps everything that reaches it."""
    if lake:
        return 1.0
    contrast = max(density - 2.65, 0.0) / (19.3 - 2.65)
    decel = 1.0 / (1.0 + slope_m_km / 2.0)
    return min(0.6, 0.02 + 0.35 * contrast * decel + 0.05 * decel)


def route_placers(plan, ctx, receiver: np.ndarray, slope_m_km: np.ndarray, river: np.ndarray,
                  lake: np.ndarray, coast: np.ndarray, layout) -> dict:  # fmt: skip
    """Route every mobile account of a placer commodity; adds placer systems / lots to the plan
    and returns a per-commodity balance report."""
    led = plan.ledger
    n = ctx.n
    order = np.argsort(-ctx.surface_m, kind="stable")  # receivers are lower: upstream first
    report = {}
    mobiles = sorted(
        (o, b) for (o, b), v in led.balances.items()
        if o.startswith("mobile:") and b in PLACER_COMMODITIES and v > 0
    )  # fmt: skip
    if not mobiles:
        return report
    sys_hex = {s["system_id"]: s["hex"] for s in plan.systems}
    for rec in plan.eroded:
        sys_hex[rec["system_id"]] = rec["hex"]
    lot_sys = {lot["lot_id"]: lot["system_id"] for lot in plan.lots}
    for com in sorted({b for _o, b in mobiles}):
        dens, abr = GRAIN.get(com, (4.0, 0.02))
        src = np.zeros(n)
        parents: dict = {}
        transit = f"transit:{com}"
        for owner, b in mobiles:
            if b != com:
                continue
            key = owner.split(":", 1)[1]
            sid = lot_sys.get(key, key)
            h = sys_hex.get(sid)
            if h is None:
                continue
            m = led.balance(owner, com)
            led.transfer(f"route/{owner}/{com}", owner, transit, com, m)
            src[h] += m
            parents.setdefault(int(h), set()).add(sid)
        flux = src.copy()
        dep = np.zeros(n)
        coastal = np.zeros(n)
        lost = 0.0
        exported = 0.0
        mix = {int(h): set(v) for h, v in parents.items()}
        for i in order:
            f = flux[i]
            if f <= 0:
                continue
            if ctx.ocean[i]:
                continue  # handled where the river enters the sea
            eta = trap_efficiency(dens, float(slope_m_km[i]), bool(lake[i]))
            if not river[i] and not lake[i]:
                eta *= 0.3  # slopes and colluvium: little sorting
            d = eta * f
            dep[i] += d
            rest = f - d
            ab = rest * min(abr * ctx.spacing_km / 100.0, 1.0)
            lost += ab
            rest -= ab
            r = int(receiver[i])
            if r < 0 or r >= n or ctx.ocean[r]:
                # river mouth: wave sorting keeps part on the coast, the rest goes offshore
                if coast[i] or (0 <= r < n and ctx.ocean[r]):
                    c = 0.4 * rest
                    coastal[i] += c
                    exported += rest - c
                else:
                    exported += rest  # leaves the map across its edge
                continue
            flux[r] += rest
            if i in mix:
                mix.setdefault(r, set()).update(mix[i])
        _book(plan, ctx, com, dep, coastal, mix, layout, "PLACER_RIVER", "alluvial")
        _book(plan, ctx, com, coastal, None, mix, layout, "PLACER_COAST", "coastal")
        if lost > 0:
            led.remove(f"abrasion/{com}", transit, com, min(lost, led.balance(transit, com)))
        left = led.balance(transit, com)
        if left > 0:
            led.remove(f"export/{com}", transit, com, left, "exported")
        report[com] = {"released_t": float(src.sum()), "deposited_t": float(dep.sum()),
                       "coastal_t": float(coastal.sum()), "abraded_t": float(lost),
                       "exported_t": float(left)}  # fmt: skip
    plan.report["placers"] = report
    plan.report["ledger"] = led.validate()
    return report


def _book(plan, ctx, com, dep, also, mix, layout, model_id, kind):
    """Deposits above the minimum become placer lots (with source mix), the rest a dispersed
    store; both are ledger transfers out of the transit account."""
    led = plan.ledger
    transit = f"transit:{com}"
    min_t = MIN_LOT_T.get(com, 5.0)
    dispersed = 0.0
    for i in np.nonzero(dep > 0)[0]:
        m = float(dep[i])
        if m < min_t:
            dispersed += m
            continue
        sid = stable_id("D", ctx.seed, model_id, com, int(i))
        lot_id = f"{sid}-B0-L0"
        x, y = float(ctx.x_km[i] * 1000.0), float(ctx.y_km[i] * 1000.0)
        length = 0.8 * ctx.spacing_km * 1000.0
        width = 300.0 if kind == "alluvial" else 400.0
        pay = 1.5
        body = make_body("placer", f"{sid}-B0", sid, (x, y, float(ctx.surface_m[i])),
                         strike_deg=0.0, dip_deg=0.0,
                         params={"length_m": length, "down_dip_m": width,
                                 "true_thickness_m": pay})  # fmt: skip
        vol = body.analytic_volume()
        rho = 1900.0
        rock_t = vol * rho / 1000.0
        m = min(m, led.balance(transit, com))
        led.transfer(f"trap/{model_id}/{com}/{int(i)}", transit, lot_id, com, m)
        plan.bodies[body.body_id] = body
        grade_g_m3 = m * 1e6 / vol
        plan.lots.append({"lot_id": lot_id, "system_id": sid, "body_id": body.body_id,
                          "model_id": model_id, "volume_m3": vol, "dry_bulk_density_kg_m3": rho,
                          "rock_t": rock_t, "grades": {com: m / rock_t},
                          "typed_grades": {com: {"value": grade_g_m3, "unit": "g_per_m3"}},
                          "contained": {com: m}, "net_pay_thickness_m": pay,
                          "source_mix": sorted(mix.get(int(i), [])), "gem": {},
                          "eroded_m3": 0.0, "unit_basis": "dry_sediment",
                          "balance_owner_id": lot_id, "quality": {"placer_kind": kind},
                          "calibration_status": "provisional"})  # fmt: skip
        plan.systems.append({"system_id": sid, "province_id": f"drainage:{com}",
                             "model_id": model_id, "title": f"россыпь ({kind})",
                             "event_id": "erosion_v4", "age_ma": 0.01,
                             "provenance": "routed_from_eroded_bodies",
                             "formation_depth_m": 0.0, "depth_top_m": 0.0, "exposed": True,
                             "hex": int(i), "x_m": x, "y_m": y, "morphology": "placer",
                             "strike_deg": 0.0, "dip_deg": 0.0, "strike_source": "drainage",
                             "host": "alluvium" if kind == "alluvial" else "beach sand",
                             "principal": [com], "body_ids": [body.body_id], "volume_m3": vol,
                             "rock_t": rock_t, "source_mix": sorted(mix.get(int(i), [])),
                             "calibration_status": "provisional",
                             "discovery_state": "undiscovered",
                             "surface_occurrence": True})  # fmt: skip
    if dispersed > 0:
        d = min(dispersed, led.balance(transit, com))
        led.transfer(f"disperse/{model_id}/{com}", transit, f"dispersed:{model_id}", com, d)


def placer_sanity(plan) -> dict:
    """Every placer lot names its parents; with no parent body there is no placer."""
    placer = ("PLACER_RIVER", "PLACER_COAST")
    bad = [lot["lot_id"] for lot in plan.lots
           if lot["model_id"] in placer and not lot.get("source_mix")]  # fmt: skip
    return {"without_source": bad, "ok": not bad}


def paleochannel(seed: str, origin_m, azimuth_deg: float, length_m: float, width_m: float,
                 pay_m: float, burial_m: float, surface_m: float):  # fmt: skip
    """A buried placer of an archived channel (T44): its course comes from the old event, not
    from the present D6 drainage."""
    sid = stable_id("D", seed, "PALEOPLACER", *[round(v) for v in origin_m], round(azimuth_deg))
    return make_body("paleoplacer", f"{sid}-B0", sid,
                     (origin_m[0], origin_m[1], surface_m - burial_m),
                     strike_deg=azimuth_deg, dip_deg=0.0,
                     params={"length_m": length_m, "down_dip_m": width_m,
                             "true_thickness_m": pay_m})  # fmt: skip


def grade_g_per_m3(contained_t: float, volume_m3: float) -> float:
    return contained_t * 1e6 / volume_m3 if volume_m3 > 0 else math.nan
