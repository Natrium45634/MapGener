"""Massif plans: ridge / divide graph, passes, peaks with prominence, valley graph (TZ 4.6, 2).

The structure of the mountains is already planned in v3 (``mountains.plan``: systems, main
/ branch / spur ridges joined at knots, peaks and passes along the crests, valleys between
spurs) and the v4 rivers have cut the real drainage into it. A :func:`massif_plans` record
puts both together per mountain system and checks them against each other:

* ``ridge_graph`` - nodes (ends, junctions where a ridge joins its parent) and edges (the
  ridge polylines with stable ids and rank);
* ``valley_graph`` - the v4 river segments inside the massif, directed downstream, with
  their outlet (sea, lake, closed basin, window edge) - the real valleys, not a second,
  independent network;
* ``peaks`` - hex summits with **topographic prominence** (height above the key col,
  computed on the whole surface by a union-find sweep from the top down) and the planned
  sub-hex peaks;
* ``passes`` - the planned saddles, checked to lie below both peaks they connect;
* quality: mean slope, local relief, valley density, rivers on crest hexes (a ridge must
  not run along a thalweg), the dissection index.

Dissection index [А] (a heuristic for the preliminary morphology, not a physical law):
``I = clamp(k1 log(1 + A / A0) (1 - exp(-age / tau)) (erodibility / R0) (runoff / 500 mm))``.
"""

from __future__ import annotations

import numpy as np


def prominence(z, land, nbr):
    """Topographic prominence (m) of every hex summit of ``z`` on ``land`` (0 elsewhere).

    Union-find from the highest hex down; when two islands of higher ground meet at a hex,
    the lower summit's prominence is its height minus the height of that hex (the key col).
    A summit that never meets higher ground keeps ``height - min(z on land)``.
    """
    z = np.asarray(z, dtype=np.float64).ravel()
    land = np.asarray(land, dtype=bool).ravel()
    n = z.size
    parent = np.full(n, -1, dtype=np.int64)
    peak = np.full(n, -1, dtype=np.int64)
    prom = np.zeros(n)
    order = np.nonzero(land)[0]
    order = order[np.argsort(-z[order], kind="stable")]
    par = parent.tolist()
    pk = peak.tolist()
    zl = z.tolist()
    nb_list = nbr.tolist()
    pr = prom.tolist()

    def find(a):
        root = a
        while par[root] != root:
            root = par[root]
        while par[a] != root:
            par[a], a = root, par[a]
        return root

    for h in order.tolist():
        par[h] = h
        pk[h] = h
        for q in nb_list[h]:
            if q < 0 or par[q] < 0:
                continue
            ra, rb = find(h), find(q)
            if ra == rb:
                continue
            pa, pb = pk[ra], pk[rb]
            hi, lo = (pa, pb) if zl[pa] >= zl[pb] else (pb, pa)
            pr[lo] = zl[lo] - zl[h]
            par[rb] = ra
            pk[ra] = hi
    if order.size:
        base = float(z[order[-1]])
        roots = {find(h) for h in order.tolist()}
        for r in roots:
            top = pk[r]
            pr[top] = zl[top] - base
    return np.asarray(pr)


def dissection_index(area_km2, orogen_age_ma, rock_resistance, runoff_mm_yr, in_mountains):
    a = np.log1p(np.asarray(area_km2, dtype=np.float64) / 2000.0)
    age = np.nan_to_num(np.asarray(orogen_age_ma, dtype=np.float64), nan=200.0)
    erod = 1.0 - np.asarray(rock_resistance, dtype=np.float64)
    clim = np.clip(np.asarray(runoff_mm_yr, dtype=np.float64) / 500.0, 0.0, 2.0)
    i = 0.45 * a * (1.0 - np.exp(-np.clip(age, 0.0, None) / 15.0)) * (erod / 0.35) * clim
    return np.where(in_mountains, np.clip(i, 0.0, 1.0), 0.0)


def massif_plans(project, z, slope, local_relief, river_mask, receiver, rivers, nbr):
    """One record per mountain system visible in the window (JSON-able)."""
    hist = project.hist
    layout = project.geometry.layout
    sys_id = hist.layers["mountain_system_id"].ravel()
    cls = hist.layers["mountain_class"].ravel()
    land = ~project.hydro.layers["ocean_mask"].ravel()
    prom = prominence(z, land, nbr)
    zf = np.asarray(z, dtype=np.float64).ravel()
    plan = hist.plan
    out = []
    area_hex = layout.hex_area_km2
    for s in plan.get("systems", []):
        sid = s["id"]
        inside = sys_id == sid
        if not inside.any():
            continue
        ridges = [r for r in plan["ridges"] if r["system"] == sid]
        nodes, edges = [], []
        for r in ridges:
            pts = r["points_km"]
            a = {"id": f"r{r['id']}:start", "x_km": pts[0][0], "y_km": pts[0][1], "kind": "end"}
            b = {"id": f"r{r['id']}:end", "x_km": pts[-1][0], "y_km": pts[-1][1], "kind": "end"}
            if r["parent"] >= 0:
                a["kind"] = "junction"
                a["joins_ridge"] = r["parent"]
            nodes += [a, b]
            length = (
                float(np.sum(np.hypot(*np.diff(np.asarray(pts), axis=0).T)))
                if len(pts) > 1
                else 0.0
            )
            edges.append(
                {
                    "ridge_id": r["id"],
                    "rank": r["rank"],
                    "role": r.get("role"),
                    "parent": r["parent"],
                    "from": a["id"],
                    "to": b["id"],
                    "length_km": round(length, 1),
                }
            )
        # summits of the massif: hexes higher than all six neighbours
        idx = np.nonzero(inside & land)[0]
        nb = nbr[idx]
        higher = np.where(nb >= 0, zf[np.maximum(nb, 0)], -np.inf)
        is_top = (zf[idx][:, None] >= higher).all(axis=1)
        tops = idx[is_top]
        tops = tops[np.argsort(-prom[tops])][:40]
        hx, hy = (a.ravel() for a in layout.centers())
        peaks = [
            {
                "hex": int(k),
                "x_km": round(float(hx[k]), 2),
                "y_km": round(float(hy[k]), 2),
                "elevation_m": round(float(zf[k]), 1),
                "prominence_m": round(float(prom[k]), 1),
                "type": "summit" if prom[k] >= 300.0 else "subsidiary",
            }
            for k in tops
        ]
        planned = {p["id"]: p for p in plan["peaks"] if p["system"] == sid}
        passes = []
        for ps in (p for p in plan["passes"] if p["system"] == sid):
            ids = ps.get("between_peaks", [])
            pe = ps.get("elevation_m")
            # the structural check: on the planned crest the saddle lies below both peaks
            # (hex heights at 25-75 km do not resolve a pass and its peaks separately)
            pc = ps.get("crest")
            peak_c = [planned[i].get("crest") for i in ids if i in planned]
            ok = pc is None or not peak_c or all(c is None or pc < c + 1e-9 for c in peak_c)
            passes.append(
                {
                    "id": ps["id"],
                    "ridge": ps["ridge"],
                    "x_km": ps["x_km"],
                    "y_km": ps["y_km"],
                    "elevation_m": pe,
                    "between_peaks": ids,
                    "below_neighbours": bool(ok),
                }
            )
        # valley graph: v4 river segments with a cell in the massif
        segs = []
        for seg_id, seg in enumerate(rivers):
            cells = seg.get("cells")
            if cells is None:
                continue
            if not inside[np.asarray(cells, dtype=np.int64)].any():
                continue
            segs.append(
                {
                    "segment": seg_id,
                    "next_segment": seg.get("next_segment", -1),
                    "strahler": seg.get("strahler"),
                    "mouth": seg.get("mouth"),
                    "basin_id": seg.get("basin_id"),
                    "discharge_m3_s": seg.get("discharge_m3_s"),
                    "leaves_massif": bool(not inside[int(cells[-1])]),
                }
            )
        crest = cls == 1
        river_on_crest = int((np.asarray(river_mask).ravel() & crest & inside).sum())
        area = float(inside.sum() * area_hex)
        out.append(
            {
                "massif_id": sid,
                "orogen_kind": s.get("kind"),
                "morphology": s.get("morphology"),
                "style": s.get("style"),
                "uplift_age_ma": s.get("age_ma"),
                "maturity": s.get("maturity"),
                "area_km2": round(area, 1),
                "centroid_km": s.get("centroid_km"),
                "ridge_graph": {"nodes": nodes, "edges": edges},
                "valley_graph": {
                    "segments": segs,
                    "outlets": sorted({str(v["mouth"]) for v in segs}),
                },
                "peaks": peaks,
                "passes": passes,
                "quality": {
                    "mean_slope_m_per_km": round(
                        float(np.mean(np.asarray(slope).ravel()[inside])), 2
                    ),
                    "p90_local_relief_m": round(
                        float(np.percentile(np.asarray(local_relief).ravel()[inside], 90)), 1
                    ),
                    "valley_density_per_1000km2": round(
                        1000.0
                        * float(np.asarray(river_mask).ravel()[inside].sum())
                        / max(area / area_hex, 1.0)
                        / area_hex,
                        4,
                    ),
                    "rivers_on_crest_hexes": river_on_crest,
                    "passes_below_peaks": sum(1 for p in passes if p["below_neighbours"]),
                    "passes_total": len(passes),
                    "summits_prominence_ge_300m": sum(
                        1 for p in peaks if p["prominence_m"] >= 300.0
                    ),
                },
                "generator_version": "terrain-1",
            }
        )
    return out, prom
