"""Export of the 5.5.1 plan (§15, T73): GeoJSON footprints in the world's planar frame (never
silently lat/lon), lots and estimates as CSV with units and unique lot keys, systems and the
ledger as JSON, and a manifest (versions, seed, model references, units)."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

from . import ALGORITHM_VERSION, SCHEMA_VERSION
from .bodies import Ellipsoid, Pipe, Plate, VeinNetwork

CRS_NOTE = ("planar world frame of HexWorld: x east, y south, kilometres (not WGS84; GeoJSON "
            "readers must treat the coordinates as a local engineering CRS)")  # fmt: skip


def _footprint(b) -> list:
    """Horizontal projection polygon of a body, world km."""
    if isinstance(b, VeinNetwork):
        return [_footprint(v) for v in b.veins]
    if isinstance(b, Plate):
        fp = b.footprint() / 1000.0
    elif isinstance(b, Ellipsoid):
        a, bb, _c = b.abc
        th = np.linspace(0, 2 * np.pi, 33)
        loc = np.column_stack([a * np.cos(th), bb * np.sin(th), np.zeros_like(th)])
        fp = b.to_world(loc)[:, :2] / 1000.0
    elif isinstance(b, Pipe):
        th = np.linspace(0, 2 * np.pi, 33)
        r = max(b.rt, b.rb)
        fp = np.column_stack([b.origin_m[0] + r * np.cos(th), b.origin_m[1] + r * np.sin(th)])
        fp = fp / 1000.0
    else:
        return []
    ring = [[round(float(x), 4), round(float(y), 4)] for x, y in fp]
    if ring and ring[0] != ring[-1]:
        ring.append(ring[0])
    return ring


def export_minerals(project, out_dir) -> list[str]:
    m = project.minerals
    if m is None:
        raise ValueError("no 5.5.1 plan in this project")
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    plan = m.plan
    sys_by = {s["system_id"]: s for s in plan.systems}
    feats = []
    for bid, b in sorted(plan.bodies.items()):
        fp = _footprint(b)
        if not fp:
            continue
        s = sys_by.get(b.system_id, {})
        if isinstance(b, VeinNetwork):
            geom = {"type": "MultiPolygon", "coordinates": [[ring] for ring in fp]}
        else:
            geom = {"type": "Polygon", "coordinates": [fp]}
        feats.append({"type": "Feature", "geometry": geom,
                      "properties": {"body_id": bid, "system_id": b.system_id,
                                     "model_id": s.get("model_id"), "morphology": b.morphology,
                                     "depth_top_m": s.get("depth_top_m"),
                                     "strike_deg": b.strike_deg, "dip_deg": b.dip_deg,
                                     "principal": s.get("principal", []),
                                     "geometry_revision": b.geometry_revision,
                                     "note": "horizontal projection, not solid ore"}})  # fmt: skip
    gj = {"type": "FeatureCollection", "name": "hexworld_minerals_footprints",
          "crs": {"type": "name", "properties": {"name": "HEXWORLD:PLANAR_KM"}},
          "hexworld_crs": CRS_NOTE, "features": feats}  # fmt: skip
    files = []
    p = out / "minerals_footprints.geojson"
    p.write_text(json.dumps(gj, ensure_ascii=False), encoding="utf-8")
    files.append(str(p))
    p = out / "minerals_lots.csv"
    with open(p, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["lot_key", "lot_id", "system_id", "model_id", "morphology", "commodity",
                    "quantity", "unit", "basis", "not_additive_group", "cutoff", "depth_top_m",
                    "x_km", "y_km", "calibration_status"])  # fmt: skip
        from .catalog import load

        cat = load()
        for lot in sorted(plan.lots, key=lambda x: x["lot_id"]):
            s = sys_by.get(lot["system_id"], {})
            for com, q in sorted(lot.get("contained", {}).items()):
                c = cat.commodities.get(com)
                unit = "t"
                w.writerow([f"{lot['lot_id']}|{com}", lot["lot_id"], lot["system_id"],
                            lot["model_id"], s.get("morphology", ""), com, f"{float(q):.6g}",
                            unit, lot.get("unit_basis", ""), c.basis if c else com, "none",
                            s.get("depth_top_m", ""),
                            round(s.get("x_m", np.nan) / 1000.0, 3) if s else "",
                            round(s.get("y_m", np.nan) / 1000.0, 3) if s else "",
                            lot.get("calibration_status", "provisional")])  # fmt: skip
    files.append(str(p))
    from .stage import to_json

    p = out / "minerals_plan.json"
    p.write_text(json.dumps(to_json(m), ensure_ascii=False), encoding="utf-8")
    files.append(str(p))
    from .secondary551 import discovery_view

    p = out / "minerals_discovery.json"  # what a surveyor sees: estimates, not the truth
    p.write_text(json.dumps(discovery_view(plan, plan.seed), ensure_ascii=False), encoding="utf-8")
    files.append(str(p))
    p = out / "minerals_manifest.json"
    p.write_text(json.dumps({
        "schema_version": SCHEMA_VERSION, "algorithm_version": ALGORITHM_VERSION,
        "seed": plan.seed, "crs": CRS_NOTE,
        "units": {"quantity": "t of the commodity (contained); fluids: phase mass t, "
                              "standard m³ in the plan JSON; heat J and W separately",
                  "depth": "m below the solid surface; water column separate"},
        "model_refs": "minerals/data/deposit_models.json (92 templates, provisional)",
        "files": [Path(f).name for f in files],
        "not_additive": "rows of one not_additive_group are views of one physical inventory",
    }, ensure_ascii=False, indent=1), encoding="utf-8")  # fmt: skip
    files.append(str(p))
    return files
