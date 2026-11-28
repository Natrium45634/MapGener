"""Stage ``v551``: the 5.5.1 mineral plan of a generated world, its persistence and hashes.

Runs after ``v52`` (it reuses the 5.2 geological context through the adapter, provenance
``inferred``). Read-only views never reseed; a regenerated world with the same inputs gives
the same plan (stable keys, no global RNG).
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import asdict, dataclass, field, fields

import numpy as np

from . import ALGORITHM_VERSION, SCHEMA_VERSION
from .bodies import Body, Ellipsoid, Pipe, Plate, VeinNetwork
from .ledger import ResourceLedger
from .plan import MineralPlan, PlanContext, build_plan

_CLASSES = {"Plate": Plate, "Ellipsoid": Ellipsoid, "Pipe": Pipe, "VeinNetwork": VeinNetwork}


@dataclass
class MineralData:
    plan: MineralPlan
    layers: dict  # small 2D summaries (non-additive footprint count, top commodity)
    summary: dict
    stage_keys: dict
    provenance: dict
    timings: dict = field(default_factory=dict)
    warnings: list = field(default_factory=list)

    def content_hashes(self) -> dict:
        parts = []
        for lot in sorted(self.plan.lots, key=lambda x: x["lot_id"]):
            c = sorted((k, round(float(v), 6)) for k, v in lot.get("contained", {}).items())
            parts.append(f"{lot['lot_id']}:{c}")
        ids = "|".join(parts)
        return {"minerals/lots": hashlib.sha256(ids.encode()).hexdigest(),
                "minerals/systems": hashlib.sha256("|".join(sorted(
                    s["system_id"] for s in self.plan.systems)).encode()).hexdigest()}  # fmt: skip


def stage_keys(config, project) -> dict:
    up = project.resources.stage_keys.get("v52_resources", "") if project.resources else ""
    doc = {"v": ALGORITHM_VERSION, "up": up, "p": asdict(config.minerals)}
    h = hashlib.sha256(json.dumps(doc, sort_keys=True).encode())
    return {"v551_minerals": h.hexdigest()[:24]}


def run_minerals(config, project, report=lambda f: None) -> MineralData:
    from ..resources.deposits import context_from_project
    from . import fluids, placers
    from .features import features_from_project

    t0 = time.perf_counter()
    ctx = context_from_project(project)
    F = features_from_project(project, ctx)
    pc = PlanContext(ctx, F, project.layout, float(project.hist.duration_ma))
    report(0.2)
    plan = build_plan(pc, lambda_scale=config.minerals.lambda_scale)
    # deeper than the modelled window: kept out of the plan, recorded
    keep = []
    for s in plan.systems:
        if s["depth_top_m"] > config.minerals.max_depth_m:
            plan.rejected.append({"system_id": s["system_id"], "reason": "below max_depth_m"})
        else:
            keep.append(s)
    plan.systems = keep
    t1 = time.perf_counter()
    report(0.55)
    H = project.hydro.layers
    n = ctx.n
    receiver = np.asarray(H.get("flow_receiver", -np.ones(n)), dtype=np.int64).ravel()
    from ..views import mask_rim

    land2 = np.asarray(H["land_mask"], bool)
    placers.route_placers(
        plan, ctx, receiver,
        np.nan_to_num(np.asarray(H["slope_m_per_km"], float).ravel()),
        np.asarray(H["river_mask"], bool).ravel(),
        (np.asarray(H["lake_id"], int) >= 0).ravel(),
        mask_rim(land2, project.layout).ravel(), project.layout,
    )  # fmt: skip
    fluids.apply_coal_ranks(plan, ctx)
    t2 = time.perf_counter()
    report(0.75)
    if config.minerals.include_p1:
        fluids.petroleum_systems(plan, pc, config.minerals.lambda_scale)
        rech = None
        if project.env is not None and "groundwater_recharge_annual_mm" in project.env.layers:
            rech = np.asarray(project.env.layers["groundwater_recharge_annual_mm"], float).ravel()
        fluids.aquifer_units(plan, pc, rech)
        fluids.brines(plan, pc)
        fluids.geothermal(plan, pc)
        fluids.hydrates(plan, pc)
        fluids.unconventional(plan, pc)
    plan.report["ledger"] = plan.ledger.validate()
    t3 = time.perf_counter()
    report(0.95)
    layers = _layers(plan, ctx)
    summary = summarize(plan)
    data = MineralData(
        plan=plan, layers=layers, summary=summary, stage_keys=stage_keys(config, project),
        provenance={"algorithm_version": ALGORITHM_VERSION, "schema_version": SCHEMA_VERSION,
                    "geology": "5.2 GeoContext adapter (lithology inferred)",
                    "calibration_status": "provisional",
                    "unknown_inputs": list(ctx.unknown_fields)},
        timings={"plan": round(t1 - t0, 3), "placers": round(t2 - t1, 3),
                 "fluids": round(t3 - t2, 3), "total": round(time.perf_counter() - t0, 3)},
    )  # fmt: skip
    if not plan.report["ledger"]["ok"]:
        data.warnings.append("5.5.1: the resource ledger does not close (see the report)")
    return data


def _layers(plan, ctx) -> dict:
    """Footprint count per hex (non-additive display) and the main commodity of each hex."""
    count = np.zeros(ctx.n, dtype=np.int32)
    for s in plan.systems:
        count[s["hex"]] += 1
    return {"system_count": count.reshape(ctx.shape)}


def summarize(plan) -> dict:
    by_model: dict = {}
    for s in plan.systems:
        by_model[s["model_id"]] = by_model.get(s["model_id"], 0) + 1
    totals: dict = {}
    for lot in plan.lots:
        for com, m in lot.get("contained", {}).items():
            totals[com] = totals.get(com, 0.0) + float(m)
    return {"systems": len(plan.systems), "lots": len(plan.lots), "by_model": by_model,
            "contained_t_by_commodity": dict(sorted(totals.items())),
            "provinces": len(plan.provinces), "rejected": len(plan.rejected),
            "eroded_systems": len(plan.eroded), "ledger_ok": plan.report["ledger"]["ok"],
            "unresolved_hexes": {k: v for k, v in plan.unresolved.items() if v},
            "calibration_status": "provisional"}  # fmt: skip


# -- persistence ---------------------------------------------------------------------------------
def body_to_dict(b: Body) -> dict:
    d = {f.name: getattr(b, f.name) for f in fields(b) if f.name != "veins"}
    d["class"] = type(b).__name__
    if isinstance(b, VeinNetwork):
        d["veins"] = [body_to_dict(v) for v in b.veins]
    return _jsonable(d)


def body_from_dict(d: dict) -> Body:
    d = dict(d)
    cls = _CLASSES[d.pop("class")]
    veins = d.pop("veins", None)
    d["origin_m"] = tuple(d["origin_m"])
    b = cls(**d)
    if veins is not None:
        b.veins = [body_from_dict(v) for v in veins]
    return b


def _jsonable(x):
    if isinstance(x, dict):
        return {str(k): _jsonable(v) for k, v in x.items()}
    if isinstance(x, list | tuple):
        return [_jsonable(v) for v in x]
    if isinstance(x, np.generic):
        return x.item()
    if isinstance(x, float) and not np.isfinite(x):
        return None
    return x


def to_json(data: MineralData) -> dict:
    p = data.plan
    return _jsonable({
        "schema_version": SCHEMA_VERSION, "algorithm_version": ALGORITHM_VERSION,
        "seed": p.seed, "provinces": p.provinces, "systems": p.systems,
        "bodies": [body_to_dict(b) for _k, b in sorted(p.bodies.items())],
        "lots": p.lots, "estimates": p.estimates, "rejected": p.rejected, "eroded": p.eroded,
        "unresolved": p.unresolved, "report": p.report,
        "ledger": {"balances": [[o, b, v] for (o, b), v in sorted(p.ledger.balances.items())],
                   "totals": {b: dict(t) for b, t in sorted(p.ledger.totals.items())},
                   "ops": sorted(p.ledger.ops)},
        "summary": data.summary, "stage_keys": data.stage_keys, "provenance": data.provenance,
        "timings": data.timings, "warnings": data.warnings,
    })  # fmt: skip


def from_json(doc: dict, layers: dict) -> MineralData:
    led = ResourceLedger()
    for o, b, v in doc["ledger"]["balances"]:
        led.balances[(o, b)] = float(v)
    for b, t in doc["ledger"]["totals"].items():
        led._tot(b).update({k: float(v) for k, v in t.items()})
    for op in doc["ledger"]["ops"]:
        led.ops[op] = {"restored": True}
    plan = MineralPlan(seed=doc["seed"], algorithm_version=doc["algorithm_version"],
                       provinces=doc["provinces"], systems=doc["systems"],
                       bodies={d["body_id"]: body_from_dict(d) for d in doc["bodies"]},
                       lots=doc["lots"], estimates=doc["estimates"], ledger=led,
                       rejected=doc["rejected"], eroded=doc["eroded"],
                       unresolved=doc["unresolved"], report=doc["report"])  # fmt: skip
    return MineralData(plan=plan, layers=layers, summary=doc["summary"],
                       stage_keys=doc["stage_keys"], provenance=doc["provenance"],
                       timings=doc.get("timings", {}),
                       warnings=doc.get("warnings", []))  # fmt: skip
