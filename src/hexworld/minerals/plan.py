"""The world plan of TZ 5.5.1 (§§3–8, 10): provinces -> counts -> systems -> bodies -> lots.

USGS three-part logic: permissive tracts from the hard gates, a Poisson count per stable
province (one draw, the key is the province's physical centroid, not its hex count), and
type-specific grade–tonnage priors (provisional). Sizes are mass-conditioned (the geometry
is solved from the sampled mass and shape ratios), orientation follows the structural grain
of the relief where it exists, present depth = formation depth − exhumation; a part lifted
above the surface is cut off and its mass goes to the ledger as eroded (the placers draw on
it). Every lot has a balance owner and comes from an explicit province source reservoir.
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass, field

import numpy as np
from scipy import ndimage

from . import ALGORITHM_VERSION
from .bodies import PRIMITIVE, make_body, solve_grade_field
from .catalog import load
from .gates import RULES, evaluate
from .ledger import ResourceLedger
from .morphology import geometry_for_mass
from .priors import FLUID_MODELS, PRIORS
from .units import GRADE_UNITS, gem_rough_ct, to_internal

#: byproducts carried by a host mineral: byproduct -> (host commodity, mass ratio to host)
#: [К]; only where the host is present in the lot (T10)
BYPRODUCTS = {
    "Ga": ("Al", 5e-5 / 0.24),  # in gibbsite / boehmite of bauxite
    "Re": ("Mo", 3e-4),  # in molybdenite
    "In": ("Zn", 2e-4),  # in sphalerite
    "Ge": ("Zn", 1e-4),
    "Se": ("Cu", 5e-4),
    "Te": ("Cu", 1e-4),
    "Hf": ("Zr", 0.02),  # in zircon
}
#: resistant, dense commodities released to placers when a body erodes
PLACER_COMMODITIES = ("Au", "Pt", "Pd", "Sn", "diamond", "corundum", "Ti", "Zr", "garnet",
                      "W", "Ce", "La", "Nd")  # fmt: skip
#: provisional anchor probability per eligible volcanic system [К]
OBJECT_P = {"PORPHYRY": 0.12, "EPITHERMAL": 0.12, "VOLCANIC_GLASS": 0.1, "GEOTHERMAL": 0.25,
            "SMS": 0.15}  # fmt: skip


def stable_u(*key) -> float:
    """A uniform in [0, 1) from a canonical length-prefixed key (BLAKE2b), no global RNG."""
    h = hashlib.blake2b(digest_size=8)
    for k in key:
        b = str(k).encode("utf-8")
        h.update(len(b).to_bytes(4, "little"))
        h.update(b)
    return int.from_bytes(h.digest(), "little") / 2**64


def stable_normal(*key) -> float:
    u1 = max(stable_u(*key, "n1"), 1e-300)
    u2 = stable_u(*key, "n2")
    return math.sqrt(-2.0 * math.log(u1)) * math.cos(2 * math.pi * u2)


def stable_id(prefix: str, *key) -> str:
    h = hashlib.blake2b(digest_size=6)
    h.update("|".join(str(k) for k in key).encode("utf-8"))
    return f"{prefix}-{h.hexdigest()}"


def poisson(lam: float, u: float) -> int:
    """Inverse-CDF Poisson from one uniform (normal approximation for large λ)."""
    if lam <= 0:
        return 0
    if lam > 60:
        z = math.sqrt(2) * _erfinv(2 * u - 1)
        return max(0, int(round(lam + math.sqrt(lam) * z)))
    k, p = 0, math.exp(-lam)
    c = p
    while u > c and k < 1000:
        k += 1
        p *= lam / k
        c += p
    return k


def _erfinv(y: float) -> float:
    a = 0.147
    ln = math.log(max(1 - y * y, 1e-300))
    t = 2 / (math.pi * a) + ln / 2
    return math.copysign(math.sqrt(math.sqrt(t * t - ln / a) - t), y)


@dataclass
class MineralPlan:
    seed: str
    algorithm_version: str = ALGORITHM_VERSION
    provinces: list = field(default_factory=list)
    systems: list = field(default_factory=list)
    bodies: dict = field(default_factory=dict)  # body_id -> Body
    lots: list = field(default_factory=list)
    estimates: list = field(default_factory=list)
    ledger: ResourceLedger = field(default_factory=ResourceLedger)
    rejected: list = field(default_factory=list)  # candidates refused, with the reason
    eroded: list = field(default_factory=list)  # wholly eroded systems (mass -> mobile)
    unresolved: dict = field(default_factory=dict)  # model -> unresolved hex count
    report: dict = field(default_factory=dict)

    def system(self, system_id: str):
        return next((s for s in self.systems if s["system_id"] == system_id), None)

    def lots_of(self, system_id: str) -> list:
        return [lot for lot in self.lots if lot["system_id"] == system_id]


# -- structural grain ---------------------------------------------------------------------------
def structural_strike(ctx) -> tuple[np.ndarray, np.ndarray]:
    """Strike (deg from north, clockwise) along the contours of the smoothed relief and a
    flag where the grain is too weak to tell (then the strike is unconstrained)."""
    z = np.where(ctx.ocean, ctx.surface_m, ctx.surface_m).reshape(ctx.shape)
    zs = ndimage.gaussian_filter(z.astype(np.float64), 2.0)
    gy, gx = np.gradient(zs, ctx.spacing_km)  # rows = y (south), cols = x (east)
    mag = np.hypot(gx, gy).ravel()
    # contour direction is perpendicular to the gradient; azimuth from north (-y), clockwise
    cx, cy = -gy.ravel(), gx.ravel()
    strike = (np.degrees(np.arctan2(cx, -cy)) + 360.0) % 180.0
    weak = mag < np.percentile(mag, 25)
    return strike, weak


# -- plan ----------------------------------------------------------------------------------------
class PlanContext:
    """Everything the planner reads, flat arrays (n,)."""

    def __init__(self, ctx, features, layout, history_ma: float):
        self.ctx = ctx
        self.F = features
        self.layout = layout
        self.history_ma = history_ma
        self.strike, self.weak = structural_strike(ctx)
        self.area_km2 = ctx.hex_area_km2

    def surface_at(self, i: int) -> float:
        return float(self.ctx.surface_m[i])


def _components(mask: np.ndarray, shape) -> tuple[np.ndarray, int]:
    lab, k = ndimage.label(mask.reshape(shape), structure=np.ones((3, 3), bool))
    return lab.ravel(), k


def build_plan(pc: PlanContext, models=None, lambda_scale: float = 1.0) -> MineralPlan:
    ctx, F = pc.ctx, pc.F
    cat = load()
    plan = MineralPlan(seed=ctx.seed)
    mids = sorted(models or RULES)
    for mid in mids:
        if (mid in FLUID_MODELS and mid != "AQUIFER") or mid in (
            "PLACER_RIVER",
            "PLACER_COAST",
            "CU_OXID",
        ):
            continue  # fluids (fluids.py), placers (placers.py) and secondary Cu (below)
        _plan_model(plan, pc, cat, mid, lambda_scale)
    # secondary copper minerals over exhumed primary Cu bodies (T47 / CU_OXID)
    cu = np.zeros(ctx.n, bool)
    for s in plan.systems:
        if "Cu" in s["principal"] and s["depth_top_m"] < 300.0 and ctx.land[s["hex"]]:
            cu[s["hex"]] = True
    F["copper_body"] = np.where(np.isfinite(F["weathering_any"]), cu.astype(float), np.nan)
    if "CU_OXID" in mids:
        _plan_model(plan, pc, cat, "CU_OXID", lambda_scale)
    plan.report["not_implemented"] = {
        "UNCONVENTIONAL": "shale gas / tight oil / CBM templates gated, no accumulation model"
    }
    plan.report["counts"] = _counts(plan)
    plan.report["ledger"] = plan.ledger.validate()
    return plan


def _counts(plan) -> dict:
    out: dict = {}
    for s in plan.systems:
        out[s["model_id"]] = out.get(s["model_id"], 0) + 1
    return dict(sorted(out.items()))


def _plan_model(plan, pc, cat, mid, lambda_scale):
    ctx, F = pc.ctx, pc.F
    rule = RULES[mid]
    eligible, unresolved, S = evaluate(rule, F)
    plan.unresolved[mid] = int(unresolved.sum())
    if not eligible.any():
        return
    if rule.anchor == "object":
        _plan_objects(plan, pc, cat, mid, S)
        return
    lab, k = _components(eligible, ctx.shape)
    if k == 0:
        return
    area = pc.area_km2
    sums = np.bincount(lab, weights=S * area, minlength=k + 1)
    cx = np.bincount(lab, weights=ctx.x_km * S, minlength=k + 1)
    cy = np.bincount(lab, weights=ctx.y_km * S, minlength=k + 1)
    ws = np.bincount(lab, weights=S, minlength=k + 1)
    order = np.argsort(lab, kind="stable")
    starts = np.searchsorted(lab[order], np.arange(k + 2))
    regional = rule.anchor == "regional"
    if regional:
        # one unit per real lithology region, the largest first (not one per hex)
        ranked = sorted(range(1, k + 1), key=lambda p: -sums[p])[:12]
    else:
        ranked = range(1, k + 1)
    used: set = set()
    for p in ranked:
        if sums[p] <= 0:
            continue
        pcx, pcy = cx[p] / ws[p], cy[p] / ws[p]
        key = (round(pcx / 100.0), round(pcy / 100.0))
        pid = stable_id("P", ctx.seed, mid, *key)
        k_dup = 2
        while pid in used:  # two provinces of one type in one 100 km cell: -2, -3 ...
            pid = stable_id("P", ctx.seed, mid, *key, k_dup)
            k_dup += 1
        used.add(pid)
        lam = rule.lam_ref * lambda_scale * sums[p]
        count = 1 if regional else poisson(lam, stable_u(ctx.seed, mid, pid, "count"))
        idx = order[starts[p] : starts[p + 1]]
        prov = {"province_id": pid, "model_id": mid, "x_km": round(float(pcx), 2),
                "y_km": round(float(pcy), 2), "area_km2": float(idx.size * area),
                "suitability_integral_km2": float(sums[p]), "lambda": lam, "count": count,
                "age_mode": rule.age, "calibration_status": "provisional"}  # fmt: skip
        plan.provinces.append(prov)
        if count == 0:
            continue
        _open_source(plan, mid, pid, max(lam, 1.0))
        cw = np.cumsum(S[idx])
        for slot in range(count):
            u = stable_u(ctx.seed, mid, pid, "anchor", slot)
            j = int(np.searchsorted(cw, u * cw[-1], side="right"))
            i = int(idx[min(j, idx.size - 1)])
            if regional:
                _make_regional(plan, pc, cat, mid, pid, i, idx)
            else:
                _make_system(plan, pc, cat, mid, pid, i, slot)


def _plan_objects(plan, pc, cat, mid, S):
    ctx = pc.ctx
    for vs in ctx.volcano_systems:
        i = vs.get("hex_flat", vs.get("hex"))
        if i is None or not 0 <= int(i) < ctx.n or S[int(i)] <= 0:
            continue
        i = int(i)
        key = vs.get("volcano_id", str(i))
        if stable_u(ctx.seed, mid, key, "object") >= OBJECT_P.get(mid, 0.2) * S[i]:
            continue
        pid = stable_id("P", ctx.seed, mid, key)
        plan.provinces.append({"province_id": pid, "model_id": mid, "anchor": key,
                               "x_km": float(ctx.x_km[i]), "y_km": float(ctx.y_km[i]),
                               "count": 1, "calibration_status": "provisional"})  # fmt: skip
        _open_source(plan, mid, pid, 1.0)
        age = float(vs.get("formed_ma", 1.0) or 1.0)
        _make_system(plan, pc, cat, mid, pid, i, 0, event_id=key, age_ma=age)


def _open_source(plan, mid, pid, lam):
    """Province source reservoir per principal commodity: a finite accessible inventory;
    every lot is a transfer out of it (no mass from nowhere, a source is not spent twice)."""
    pr = PRIORS[mid]
    for com, (val, unit, _sd) in pr.g.items():
        if GRADE_UNITS[unit][0] != "mass":
            continue
        budget = lam * 8.0 * pr.t * to_internal(val, unit) * math.exp(2.0 * pr.sd)
        plan.ledger.open(f"src/{pid}/{com}", f"source:{pid}", com, budget, "initial")


def _depth_and_age(pc, mid, i, slot, pid, age_ma=None):
    ctx = pc.ctx
    pr = PRIORS[mid]
    rule = RULES[mid]
    u = stable_u(ctx.seed, mid, pid, slot, "depth")
    form_depth = pr.depth[0] + u * (pr.depth[1] - pr.depth[0])
    if age_ma is None:
        ua = stable_u(ctx.seed, mid, pid, slot, "age")
        if rule.age == "inherited":
            age_ma = float(max(np.nan_to_num(ctx.crust_age_ma[i]), pc.history_ma)) * (
                0.5 + 0.5 * ua
            )
            provenance = "inherited_province"
        elif rule.age == "any" and ctx.crust_age_ma[i] > pc.history_ma:
            age_ma = float(ctx.crust_age_ma[i]) * (0.3 + 0.7 * ua)
            provenance = "inherited_province"
        else:
            age_ma = pc.history_ma * ua
            provenance = "simulated_history"
    else:
        provenance = "simulated_history"
    # exhumation since formation: the simulated erosion (a lower bound for old bodies)
    exhum = float(ctx.erosion_total_m[i]) * min(1.0, age_ma / max(pc.history_ma, 1e-9))
    if provenance == "inherited_province":
        exhum = float(ctx.erosion_total_m[i]) + form_depth * 0.6 * ua_safe(ctx, mid, pid, slot)
    return form_depth, form_depth - exhum, age_ma, provenance


def ua_safe(ctx, mid, pid, slot) -> float:
    return stable_u(ctx.seed, mid, pid, slot, "exhum")


def _make_system(plan, pc, cat, mid, pid, i, slot, event_id=None, age_ma=None):
    ctx = pc.ctx
    pr = PRIORS[mid]
    tmpl = cat.models[mid]
    morph = pr.morph if pr.morph in tmpl.morphologies else tmpl.morphologies[0]
    sid = stable_id("D", ctx.seed, mid, pid, slot)
    z_raw = stable_normal(ctx.seed, sid, "tonnage")
    zt = max(-3.0, min(3.0, z_raw))  # truncation is logged in the system record
    rock_t = pr.t * math.exp(pr.sd * zt)
    trunc = abs(z_raw) > 3.0
    form_depth, depth_top, age_ma, prov = _depth_and_age(pc, mid, i, slot, pid, age_ma)
    # position inside the hex (metres), orientation from the structural grain
    r = 0.4 * ctx.spacing_km * 1000.0 * math.sqrt(stable_u(ctx.seed, sid, "r"))
    th = 2 * math.pi * stable_u(ctx.seed, sid, "th")
    x = float(ctx.x_km[i] * 1000.0 + r * math.cos(th))
    y = float(ctx.y_km[i] * 1000.0 + r * math.sin(th))
    if pc.weak[i]:
        strike, strike_src = 180.0 * stable_u(ctx.seed, sid, "strike"), "unconstrained"
    else:
        strike = float(pc.strike[i]) + 10.0 * stable_normal(ctx.seed, sid, "strike")
        strike_src = "structural_grain"
    dip = _dip_for(morph, stable_u(ctx.seed, sid, "dip"))
    surface = pc.surface_at(i)
    geom = geometry_for_mass(morph, rock_t, pr.rho)
    body = _place_body(morph, f"{sid}-B0", sid, x, y, surface - depth_top, strike, dip, geom,
                       ctx, i)  # fmt: skip
    body.formation_age_ma = age_ma
    eroded_m3 = 0.0
    if depth_top < 0 or mid in ("NI_LATERITE", "BAUXITE", "REE_CLAY", "KAOLIN"):
        eroded_m3 = body.erode_above(lambda xx, yy: np.full(np.shape(xx), surface), n=10)
    plan.bodies[body.body_id] = body
    lots = _make_lots(plan, pc, cat, mid, pid, sid, body, rock_t, pr, eroded_m3, i)
    if not lots:
        plan.bodies.pop(body.body_id)
        return
    vol = sum(lot["volume_m3"] for lot in lots)
    principal = [c for c in tmpl.principal if any(c in lot["contained"] for lot in lots)]
    plan.systems.append({
        "system_id": sid, "province_id": pid, "model_id": mid, "title": tmpl.label_ru,
        "event_id": event_id or pid, "age_ma": round(age_ma, 3), "provenance": prov,
        "formation_depth_m": round(form_depth, 1), "depth_top_m": round(max(depth_top, 0.0), 1),
        "exposed": bool(depth_top <= 0), "hex": int(i), "x_m": x, "y_m": y,
        "morphology": morph, "strike_deg": round(strike % 360.0, 2), "dip_deg": round(dip, 2),
        "strike_source": strike_src, "host": _host_of(ctx, i), "principal": principal,
        "body_ids": [body.body_id], "volume_m3": vol,
        "rock_t": sum(lot["rock_t"] for lot in lots),
        "calibration_status": "provisional", "sampler": {"z": round(zt, 4), "truncated": trunc},
        "discovery_state": "undiscovered", "surface_occurrence": bool(depth_top <= 0),
    })  # fmt: skip


def _dip_for(morph: str, u: float) -> float:
    if morph in ("vein", "vein_network", "breccia"):
        return 45.0 + 45.0 * u
    if morph in ("tabular", "stratabound", "seam", "reservoir", "paleoplacer"):
        return 15.0 * u
    if morph in ("laterite", "placer", "quarry_unit", "nodules", "crust"):
        return 0.0
    return 60.0 * u


def _place_body(morph, bid, sid, x, y, z_top, strike, dip, geom, ctx, i):
    kind = PRIMITIVE[morph]
    if kind == "pipe":
        return make_body(morph, bid, sid, (x, y, z_top), params=geom)
    if kind == "ellipsoid":
        c = geom["semi_axes_m"][2]
        return make_body(morph, bid, sid, (x, y, z_top - c), strike_deg=strike, dip_deg=dip,
                         params=geom)  # fmt: skip
    if kind == "network":
        # two or three crossing veins sharing the mass of the network
        net = make_body(morph, bid, sid, (x, y, z_top))
        k = 2 + int(stable_u(ctx.seed, sid, "n_veins") * 2)
        veins = []
        for j in range(k):
            gj = dict(geom)
            gj["length_m"] = geom["length_m"] / math.sqrt(k)
            vj = make_body("vein", f"{bid}-v{j}", sid, (x, y, z_top),
                           strike_deg=strike + 50.0 * j, dip_deg=dip, params=gj)  # fmt: skip
            veins.append(vj)
        net.veins = veins
        return net
    # plates: the origin is the upper edge centre
    taper = "ends" if morph in ("vein", "lens") else "none"
    return make_body(morph, bid, sid, (x, y, z_top), strike_deg=strike, dip_deg=dip,
                     params={**geom, "taper": taper})  # fmt: skip


def _host_of(ctx, i) -> str:
    from ..resources.deposits import LITHOLOGY

    return LITHOLOGY[int(ctx.lith[i])]


def _make_lots(plan, pc, cat, mid, pid, sid, body, rock_t_sampled, pr, eroded_m3, i):
    ctx = pc.ctx
    vol_info = body.volume(tol=0.01, n0=8, n_max=32)
    vol = vol_info["volume_m3"]
    if vol <= 0:
        # wholly eroded: the formed mass went to the mobile store (placers draw on it); the
        # system is recorded as eroded, its mass is not lost from the ledger
        rock_f = rock_t_sampled
        moved = {}
        for com, (val, unit, _sd) in pr.g.items():
            if GRADE_UNITS[unit][0] != "mass" or (f"source:{pid}", com) not in plan.ledger.balances:
                continue
            m = min(rock_f * to_internal(val, unit), plan.ledger.balance(f"source:{pid}", com))
            plan.ledger.transfer(f"erode/{sid}/{com}", f"source:{pid}", f"mobile:{sid}", com, m)
            moved[com] = m
        plan.eroded.append({"system_id": sid, "model_id": mid, "hex": int(i),
                            "eroded_contained": moved, "state": "fully_eroded"})  # fmt: skip
        return []
    rock_t = vol * pr.rho / 1000.0
    lot_id = f"{body.body_id}-L0"
    grades, contained, typed = {}, {}, {}
    gem_info = {}
    bearing = True
    if pr.gem.get("bearing_share") is not None:
        bearing = stable_u(ctx.seed, sid, "bearing") < pr.gem["bearing_share"]
    principal = cat.models[mid].principal
    field_params = None
    for k, (com, (val, unit, sd)) in enumerate(pr.g.items()):
        if com == pr.gem.get("carrier") and not bearing:
            continue  # a barren pipe: a valid kimberlite without diamonds (T48)
        z = stable_normal(ctx.seed, sid, "grade", com)
        if k and sd > 0:  # correlated with the first commodity of the system
            z = 0.6 * stable_normal(ctx.seed, sid, "grade", next(iter(pr.g))) + 0.8 * z
        g = to_internal(val, unit) * math.exp(sd * max(-3.0, min(3.0, z)) - 0.5 * sd * sd)
        g = min(g, 0.95)
        if k == 0 and sd > 0 and com in principal:
            field_params = solve_grade_field(body, g, min(sd, 1.2), pr.rho, n=8,
                                             corr_m=pr.corr)  # fmt: skip
            g = field_params["achieved"]
        grades[com] = g
        contained[com] = rock_t * g  # t
        typed[com] = {"value": g / GRADE_UNITS[unit][1], "unit": unit}
    for byp, (host, ratio) in BYPRODUCTS.items():
        if host in contained and byp not in contained and byp in cat.commodities:
            contained[byp] = contained[host] * ratio
            grades[byp] = contained[byp] / rock_t
            typed[byp] = {"value": grades[byp] / 1e-6, "unit": "ppm", "primary_host": host,
                          "trace_ratio": ratio, "recovery": "not modelled"}  # fmt: skip
    if any(not math.isfinite(v) for v in contained.values()) or not math.isfinite(rock_t):
        plan.rejected.append({"system_id": sid, "reason": "non-finite geometry or grade"})
        return []
    if not contained and mid not in ("KIMBERLITE",):
        plan.rejected.append({"system_id": sid, "reason": "no component"})
        return []
    # gems: carrier mass, gem rough carats, size bins; varieties are subsets (T09, T51)
    carrier = pr.gem.get("carrier")
    if carrier and carrier in contained:
        kg = contained[carrier] * 1000.0
        q = 0.3 + 0.7 * stable_u(ctx.seed, sid, "gem_quality")
        frac = pr.gem.get("gem_fraction", 0.0) * q
        bins = {"<0.5 ct": 0.62, "0.5–2 ct": 0.28, "2–10 ct": 0.09, ">10 ct": 0.01}
        gem_kg = kg * frac
        variety = None
        for var, gate in pr.gem.get("varieties", {}).items():
            if gate is None or np.nan_to_num(pc.F[gate][i]) == 1.0:
                variety = var
                break
        gem_info = {"carrier": carrier, "carrier_kg": kg, "gem_fraction": frac,
                    "gem_rough_ct": gem_rough_ct(kg, frac), "variety": variety,
                    "size_bins_kg": {b: gem_kg * f for b, f in bins.items()},
                    "categories": {"industrial": kg * (1 - frac) * 0.7,
                                   "ornamental": kg * (1 - frac) * 0.3, "gem": gem_kg},
                    "placer_transport": pr.gem.get("placer", True)}  # fmt: skip
    # ledger: each component comes out of the province source (budget-limited)
    scale = 1.0
    for com in list(contained):
        src = (f"source:{pid}", com)
        if src in plan.ledger.balances:
            have = plan.ledger.balance(*src)
            if contained[com] > have:
                scale = min(scale, have / contained[com]) if contained[com] > 0 else scale
    if scale < 1.0:
        plan.rejected.append({"system_id": sid, "reason": "province source exhausted",
                              "scaled_to": scale})  # fmt: skip
        for com in contained:
            contained[com] *= scale
    # the body as formed = the preserved lot + the part cut by erosion (same grade)
    frac_e = eroded_m3 / (vol + eroded_m3) if eroded_m3 > 0 else 0.0
    eroded = {com: m * frac_e / (1.0 - frac_e) for com, m in contained.items()}
    for com, m in contained.items():
        src = f"source:{pid}"
        formed = m + eroded[com]
        if (src, com) in plan.ledger.balances:
            have = plan.ledger.balance(src, com)
            if formed > have:  # the eroded share also comes out of the same budget
                formed = have
                eroded[com] = max(formed - m, 0.0)
            plan.ledger.transfer(f"form/{lot_id}/{com}", src, lot_id, com, formed)
        else:  # byproducts ride on their host: generated from the host's explicit source
            plan.ledger.open(f"form/{lot_id}/{com}", lot_id, com, formed, "generated")
        if eroded[com] > 0:  # parent loses exactly what goes to the mobile store (T41)
            plan.ledger.transfer(f"erode/{lot_id}/{com}", lot_id, f"mobile:{lot_id}", com,
                                 eroded[com])  # fmt: skip
    lot = {"lot_id": lot_id, "system_id": sid, "body_id": body.body_id, "model_id": mid,
           "volume_m3": vol, "volume_rel_change": vol_info["rel_change"],
           "dry_bulk_density_kg_m3": pr.rho, "rock_t": rock_t, "rock_t_sampled": rock_t_sampled,
           "grades": grades, "typed_grades": typed, "contained": contained,
           "grade_field": field_params, "gem": gem_info, "eroded_m3": eroded_m3,
           "eroded_contained": eroded, "bearing": bearing, "unit_basis": "dry_rock",
           "balance_owner_id": lot_id,
           "quality": dict(pr.quality), "calibration_status": "provisional"}  # fmt: skip
    plan.lots.append(lot)
    for com, m in contained.items():
        plan.estimates.append({"lot_id": lot_id, "commodity_id": com, "basis": "in_situ",
                               "cutoff": None, "quantity": m, "unit": "t",
                               "label": "mineralized material (no cutoff; not a reserve)",
                               "confidence": "provisional",
                               "not_additive_group": cat.commodities[com].basis
                               if com in cat.commodities else com})  # fmt: skip
    return [lot]


def _make_regional(plan, pc, cat, mid, pid, i, idx):
    """A quarry / sand / clay / peat unit of a real lithology region: a usable slab whose
    footprint follows the region size (capped), quality by lithology; alternative uses of one
    rock are views, never additive (T40, T52)."""
    ctx = pc.ctx
    pr = PRIORS[mid]
    sid = stable_id("D", ctx.seed, mid, pid)
    region_km2 = idx.size * pc.area_km2
    side = min(math.sqrt(region_km2) * 1000.0, 20000.0)
    thick = {"PEAT": 3.0, "SAND_GRAVEL": 8.0, "CLAY": 15.0}.get(mid, 60.0)
    usable = 0.05 + 0.25 * stable_u(ctx.seed, sid, "usable")
    surface = pc.surface_at(i)
    x, y = float(ctx.x_km[i] * 1000.0), float(ctx.y_km[i] * 1000.0)
    body = make_body("quarry_unit" if mid.startswith("QUARRY") else pr.morph or "tabular",
                     f"{sid}-B0", sid, (x, y, surface), strike_deg=float(pc.strike[i]),
                     dip_deg=0.0, params={"length_m": side, "down_dip_m": side * 0.7,
                                          "true_thickness_m": thick,
                                          "usable_fraction": usable})  # fmt: skip
    plan.bodies[body.body_id] = body
    vol = body.analytic_volume()
    rock_t = vol * pr.rho / 1000.0
    lot_id = f"{body.body_id}-L0"
    lith = _host_of(ctx, i)
    com = next(iter(pr.g), "freshwater" if mid == "AQUIFER" else mid.lower())
    if mid != "AQUIFER":  # water is accounted in m³ by fluids.aquifer_units, not as rock
        plan.ledger.open(f"form/{lot_id}/{com}", lot_id, com, rock_t, "initial")
    plan.lots.append({"lot_id": lot_id, "system_id": sid, "body_id": body.body_id,
                      "model_id": mid, "volume_m3": vol, "dry_bulk_density_kg_m3": pr.rho,
                      "rock_t": rock_t, "grades": {com: 1.0}, "typed_grades": {},
                      "contained": {} if mid == "AQUIFER" else {com: rock_t},
                      "usable_fraction": usable,
                      "usable_volume_m3": vol * usable, "uses": _uses(mid, lith),
                      "uses_note": "alternative uses of one rock volume, not additive",
                      "gem": {}, "eroded_m3": 0.0, "unit_basis": "lithology_lot",
                      "balance_owner_id": lot_id, "quality": {"lithology": lith},
                      "calibration_status": "provisional"})  # fmt: skip
    plan.systems.append({"system_id": sid, "province_id": pid, "model_id": mid,
                         "title": cat.models[mid].label_ru, "event_id": pid, "age_ma": None,
                         "provenance": "lithology_region", "formation_depth_m": 0.0,
                         "depth_top_m": 0.0, "exposed": True, "hex": int(i), "x_m": x,
                         "y_m": y, "morphology": body.morphology, "strike_deg": body.strike_deg,
                         "dip_deg": 0.0, "strike_source": "region", "host": lith,
                         "principal": [com], "body_ids": [body.body_id], "volume_m3": vol,
                         "rock_t": rock_t, "region_km2": region_km2,
                         "calibration_status": "provisional", "discovery_state": "known",
                         "surface_occurrence": True})  # fmt: skip


def _uses(mid: str, lith: str) -> list:
    if mid == "QUARRY_IGNEOUS":
        return ["dimension stone (block)", "crushed stone (aggregate)"]
    if mid == "QUARRY_SED":
        return (["cement raw material", "lime", "crushed stone"]
                if "carbonate" in lith else ["building stone", "crushed stone"])  # fmt: skip
    if mid == "QUARRY_META":
        return ["dimension stone", "crushed stone"]
    if mid == "SAND_GRAVEL":
        return ["construction sand", "gravel aggregate"]
    if mid == "CLAY":
        return ["brick clay", "ceramic clay"]
    return ["fuel" if mid == "PEAT" else "industrial"]
