"""Export of the v5.4 atmosphere (TZ 5.4 sec. 15): CSV event ledger, the event geometry as
``world_metric_json`` (world km, x east / y south; the flat map has no CRS, so no fake
lat/lon and no GeoJSON), the monthly vapour budget as CSV and the grid fields as ``.npz``.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

EVENT_COLUMNS = (
    "event_id",
    "event_type",
    "month",
    "start_day",
    "duration_h",
    "max_wind_ms",
    "final_status",
    "length_km",
    "rain_mm_cells",
    "vapour_available_mm_cells",
    "limited_by_vapour",
    "snow_fraction",
    "classification_quality",
)


def event_rows(events) -> list[dict]:
    rows = []
    for e in events:
        tr = e.get("track") or {}
        pts = tr.get("points") or []
        led = e.get("budget_ledger") or {}
        fronts = e.get("fronts") or []
        rows.append(
            {
                "event_id": e["event_id"],
                "event_type": e["event_type"],
                "month": e.get("month"),
                "start_day": e.get("start_day"),
                "duration_h": pts[-1]["t_h"] if pts else None,
                "max_wind_ms": round(float(tr["max_wind_ms"]), 2) if "max_wind_ms" in tr else None,
                "final_status": tr.get("final_status"),
                "length_km": e.get("length_km"),
                "rain_mm_cells": led.get("rain_mm_cells"),
                "vapour_available_mm_cells": led.get("vapour_available_mm_cells"),
                "limited_by_vapour": led.get("limited_by_vapour"),
                "snow_fraction": fronts[0]["snow_fraction"] if fronts else None,
                "classification_quality": e.get("classification_quality"),
            }
        )
    return rows


def write_events_csv(atm, path) -> int:
    rows = event_rows(atm.events)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=EVENT_COLUMNS)
        w.writeheader()
        for r in rows:
            w.writerow(r)
    return len(rows)


def world_metric_json(atm) -> dict:
    """Event lines and ITCZ segments in world km (``crs: world_metric_km``)."""
    feats = []
    for e in atm.events:
        pts = (e.get("track") or {}).get("points") or []
        if len(pts) >= 2:
            feats.append(
                {
                    "id": e["event_id"],
                    "kind": f"{e['event_type']}_track",
                    "line_km": [[p["x_km"], p["y_km"]] for p in pts],
                    "props": {
                        "month": e.get("month"),
                        "max_wind_ms": (e.get("track") or {}).get("max_wind_ms"),
                    },
                }
            )
        for k, fr in enumerate(e.get("fronts") or []):
            feats.append(
                {
                    "id": f"{e['event_id']}-front{k + 1}",
                    "kind": f"{fr['front_type']}_front",
                    "line_km": fr["geometry_world_km"],
                    "props": {
                        "width_m": fr["width_m"],
                        "snow_fraction": fr["snow_fraction"],
                        "delta_t_k": fr["delta_t_k"],
                    },
                }
            )
        if e["event_type"] == "AR":
            feats.append(
                {
                    "id": e["event_id"],
                    "kind": "atmospheric_river",
                    "line_km": e["axis_world_km"],
                    "props": {"width_km": e["width_km"], "mean_ivt_kg_m_s": e["mean_ivt_kg_m_s"]},
                }
            )
    for m, segs in enumerate(atm.summary.get("itcz_segments") or []):
        for k, sgm in enumerate(segs):
            feats.append(
                {
                    "id": f"ITCZ-{m + 1:02d}-{k + 1}",
                    "kind": "itcz_segment",
                    "props": {"month": m + 1, **sgm},
                }
            )
    return {
        "crs": "world_metric_km",
        "axes": "x east, y south (map), km",
        "note": "flat world without a geographic CRS: no lat/lon is invented",
        "algorithm_version": atm.provenance.get("algorithm_version"),
        "features": feats,
    }


def budget_rows(atm) -> list[dict]:
    out = []
    for led in atm.summary.get("budget_months", []):
        out.append(
            {
                k: led[k]
                for k in (
                    "month",
                    "evaporation_kg",
                    "precipitation_kg",
                    "import_kg",
                    "export_kg",
                    "storage_change_kg",
                    "residual_kg",
                    "relative_residual",
                )
            }
        )
    return out


def export_all(atm, out_dir) -> dict:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    n = write_events_csv(atm, out / "weather_events.csv")
    (out / "atmosphere_features.world_metric.json").write_text(
        json.dumps(world_metric_json(atm), ensure_ascii=False, indent=1, default=float),
        encoding="utf-8",
    )
    rows = budget_rows(atm)
    with open(out / "vapour_budget.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]) if rows else ["month"])
        w.writeheader()
        for r in rows:
            w.writerow(r)
    np.savez_compressed(out / "atmosphere_grid.npz", **atm.grid_fields)
    (out / "atmosphere_config.json").write_text(
        json.dumps(
            {
                "schema_version": "5.4.0",
                "stage": "atmospheric_circulation_5_4",
                "grid": atm.grid,
                "params": atm.provenance.get("params"),
                "units": atm.summary.get("units"),
                "quality": atm.summary.get("quality"),
                "source_snapshot_hash": atm.summary.get("source_snapshot_hash"),
            },
            ensure_ascii=False,
            indent=1,
            default=str,
        ),
        encoding="utf-8",
    )
    return {"events": n, "budget_months": len(rows), "dir": str(out)}
