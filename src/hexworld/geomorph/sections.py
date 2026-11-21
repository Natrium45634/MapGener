"""Vertical cross-sections along an arbitrary line (TZ 4.6, section 7).

One metre scale for land and sea floor. Heights are interpolated between the three nearest
hex centres (inverse distance), so the profile is not a staircase of hex values; the
categorical columns (water, province, basin) come from the hex the point falls in. The
hex values are means over 25-75 km: a 2 km gorge or a 15 km axial valley is *not* in the
profile at that size - the section shows what the map resolves.
"""

from __future__ import annotations

import csv

import numpy as np
from scipy.spatial import cKDTree

COLUMNS = (
    "distance_km",
    "x_km",
    "y_km",
    "height_m",
    "bedrock_m",
    "water_surface_m",
    "material_type",
    "province",
)


def cross_section(project, start, end, step_km: float | None = None) -> list[dict]:
    """Rows of the profile from ``start`` (x, y km) to ``end``."""
    layout = project.layout
    step = float(step_km) if step_km else layout.spacing_km / 3.0
    x0, y0 = (float(v) for v in start)
    x1, y1 = (float(v) for v in end)
    length = float(np.hypot(x1 - x0, y1 - y0))
    n = max(2, int(np.ceil(length / max(step, 1e-6))) + 1)
    t = np.linspace(0.0, 1.0, n)
    px, py = x0 + t * (x1 - x0), y0 + t * (y1 - y0)
    hx, hy = layout.centers()
    tree = cKDTree(np.column_stack([hx.ravel(), hy.ravel()]))
    dist, idx = tree.query(np.column_stack([px, py]), k=3)
    w = 1.0 / np.maximum(dist, 1e-3) ** 2
    w /= w.sum(axis=1, keepdims=True)

    def interp(arr):
        flat = np.asarray(arr, dtype=np.float64).ravel()
        return (flat[idx] * w).sum(axis=1)

    z = project.layer("elevation_m")
    height = interp(z)
    terrain = getattr(project, "terrain", None)
    if terrain is not None and project.hydro is not None:
        bed = height - interp(project.hydro.layers["sediment_thickness_m"])
    else:
        bed = height - interp(project.layer("sediment_thickness_m"))
    sea = float(project.current_sea_level_m or 0.0)
    near = idx[:, 0]
    if project.hydro is not None and "ocean_mask" in project.hydro.layers:
        L = project.hydro.layers
        ocean = L["ocean_mask"].ravel()[near]
        lake = L["inland_water_mask"].ravel()[near]
        lake_level = L["lake_surface_level_m"].ravel()[near]
    else:
        ocean = z.ravel()[near] <= sea
        lake = np.zeros(n, dtype=bool)
        lake_level = np.full(n, np.nan)
    province = np.full(n, "", dtype=object)
    if terrain is not None and "marine_province" in terrain.layers:
        from .classify import PROVINCES

        codes = terrain.layers["marine_province"].ravel()[near]
        province = np.array([PROVINCES[c] for c in codes], dtype=object)
    if terrain is not None:
        bid = terrain.layers["basin_id"].ravel()[near]
        plans = terrain.plans.get("basins", [])
        for k in range(n):
            if province[k] in ("", "land") and 0 <= bid[k] < len(plans):
                province[k] = plans[bid[k]]["basin_type"]
    rows = []
    for k in range(n):
        if ocean[k]:
            material, water = "ocean", sea
        elif lake[k]:
            material, water = "lake", float(lake_level[k])
        elif height[k] < sea:
            material, water = "land_below_sea_level", float("nan")
        else:
            material, water = "land", float("nan")
        rows.append(
            {
                "distance_km": round(float(t[k] * length), 3),
                "x_km": round(float(px[k]), 3),
                "y_km": round(float(py[k]), 3),
                "height_m": round(float(height[k]), 2),
                "bedrock_m": round(float(min(bed[k], height[k])), 2),
                "water_surface_m": None if np.isnan(water) else round(water, 2),
                "material_type": material,
                "province": str(province[k]) or "land",
            }
        )
    return rows


def write_csv(rows: list[dict], path) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNS)
        writer.writeheader()
        for r in rows:
            writer.writerow({k: ("" if r[k] is None else r[k]) for k in COLUMNS})
