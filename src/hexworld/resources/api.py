"""Queries and exports of the v5.2 catalogue (TZ 5.2 sec. 18.2, 20).

None of these functions recomputes the climate, the geology or a voxel grid: they read
the sparse catalogue and evaluate the parametric bodies where asked.
"""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path

import numpy as np

from .models import FAMILIES, footprint_radius_m, query_bodies_at, section_intervals


def _surface_m(project, x_km, y_km):
    lay = project.layout
    col, row = lay.world_to_offset(x_km, y_km)
    col, row = int(col), int(row)
    if not lay.inside(col, row):
        return 0.0
    return float(project.layer("elevation_m")[row, col])


def query_deposits_at_world_xy(
    project, x_m, y_m, depth_min_m=None, depth_max_m=None, ground_truth=False
):
    """Bodies under the world point (m), with depths below the present surface.

    ``ground_truth=False`` (player view) returns only systems that crop out."""
    r = project.resources
    surf = _surface_m(project, x_m / 1000.0, y_m / 1000.0)
    hits = query_bodies_at(r.catalog["bodies"], x_m, y_m, depth_min_m, depth_max_m, surf)
    out = []
    for h in hits:
        s = r.system(h["system_id"])
        if s is None or (not ground_truth and not s["surface_occurrence"]):
            continue
        out.append(
            {
                **h,
                "model_id": s["model_id"],
                "title": s["title"],
                "visible": bool(s["surface_occurrence"]),
            }
        )
    return out


def query_geological_cross_section(
    project, start_km, end_km, depth_limits_m=(0.0, 5000.0), step_km=None, ground_truth=True
):
    """A vertical section along a line: surface, sediment base and the intervals of the
    bodies actually cut by the section plane (I06); a potential zone is never drawn as
    ore (I07). Returns ``{"rows": [...], "bodies": [{body_id, points: [(d_km, top, bot)]}]}``.
    """
    from ..geomorph.sections import cross_section

    rows = cross_section(project, start_km, end_km, step_km)
    xs = np.array([r_["x_km"] for r_ in rows]) * 1000.0
    ys = np.array([r_["y_km"] for r_ in rows]) * 1000.0
    dist = np.array([r_["distance_km"] for r_ in rows])
    res = project.resources
    x0, y0 = (v * 1000.0 for v in start_km)
    x1, y1 = (v * 1000.0 for v in end_km)
    seg = math.hypot(x1 - x0, y1 - y0)
    out = []
    for b in res.catalog["bodies"]:
        s = res.system(b["system_id"])
        if not ground_truth and not s["surface_occurrence"]:
            continue
        cx, cy = b["center_m"][:2]
        # distance of the body centre from the section line
        if seg > 0:
            t = max(0.0, min(1.0, ((cx - x0) * (x1 - x0) + (cy - y0) * (y1 - y0)) / seg**2))
            d = math.hypot(cx - (x0 + t * (x1 - x0)), cy - (y0 + t * (y1 - y0)))
        else:
            d = math.hypot(cx - x0, cy - y0)
        if d > footprint_radius_m(b):
            continue
        iv = section_intervals(b, xs, ys)
        if not iv:
            continue
        pts = []
        at_surface = b.get("burial_depth_m", 1.0) <= 0.0
        for i, top, bot in iv:
            surf = rows[i]["height_m"]
            if at_surface:
                # a surface deposit (cap, placer, crust) follows the interpolated section
                # surface, which differs from the hex-centre height it was built on
                top, bot = surf - (b["z_top_m"] - top), surf - (b["z_top_m"] - bot)
            else:
                top = min(top, surf)
                if top <= bot:
                    continue
            dt, db = surf - top, surf - bot
            if db < depth_limits_m[0] or dt > depth_limits_m[1]:
                continue
            pts.append((float(dist[i]), float(top), float(bot)))
        if pts:
            out.append(
                {
                    "body_id": b["body_id"],
                    "system_id": b["system_id"],
                    "model_id": s["model_id"],
                    "title": s["title"],
                    "points": pts,
                }
            )
    return {
        "rows": rows,
        "bodies": out,
        "datum": f"model datum (m), sea level at {project.hydro.sea_level_m:g} m",
    }


def compute_in_situ_resources(project, commodity=None, bbox_km=None) -> dict:
    """Totals (t) of the contained commodities in situ; optionally one commodity and a
    world bbox ``(xmin, ymin, xmax, ymax)`` km. Economic recovery is never applied."""
    tot: dict = {}
    for s in project.resources.catalog["systems"]:
        if bbox_km is not None:
            x0, y0, x1, y1 = bbox_km
            if not (x0 <= s["x_km"] <= x1 and y0 <= s["y_km"] <= y1):
                continue
        for el, m in s["contained_t"].items():
            if commodity is not None and el != commodity:
                continue
            row = tot.setdefault(el, {"in_situ_t": 0.0, "systems": 0, "hidden_t": 0.0})
            row["in_situ_t"] += float(m)
            row["systems"] += 1
            if not s["surface_occurrence"]:
                row["hidden_t"] += float(m)
    return tot


def export_resources(project, out_dir) -> list[str]:
    """GeoJSON of the systems, JSON of the bodies, CSV totals, NPZ of the dense fields."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    r = project.resources
    feats = []
    for s in r.catalog["systems"]:
        props = {k: v for k, v in s.items() if k not in ("x_km", "y_km")}
        feats.append(
            {
                "type": "Feature",
                "id": s["deposit_id"],
                "geometry": {"type": "Point", "coordinates": [s["x_km"], s["y_km"]]},
                "properties": props,
            }
        )
    (out / "deposit_systems.geojson").write_text(
        json.dumps(
            {
                "type": "FeatureCollection",
                "crs_note": "world km of the HexWorld plane",
                "features": feats,
            },
            default=float,
            ensure_ascii=False,
            indent=1,
        ),
        encoding="utf-8",
    )
    (out / "ore_bodies.json").write_text(
        json.dumps(
            {"units": "m (datum), t, mass fraction", "bodies": r.catalog["bodies"]},
            default=float,
            ensure_ascii=False,
            indent=1,
        ),
        encoding="utf-8",
    )
    with open(out / "resource_totals.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["commodity", "in_situ_t", "systems", "hidden_t"])
        for el, row in sorted(compute_in_situ_resources(project).items()):
            w.writerow([el, f"{row['in_situ_t']:.6g}", row["systems"], f"{row['hidden_t']:.6g}"])
    np.savez_compressed(out / "resource_fields.npz", families=np.array(FAMILIES), **r.layers)
    return [
        "deposit_systems.geojson",
        "ore_bodies.json",
        "resource_totals.csv",
        "resource_fields.npz",
    ]
