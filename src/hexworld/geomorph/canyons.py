"""Submarine canyons and fans fed by real river sediment (TZ 4.6, 4.8) [А, approximate].

Canyons are cut by turbidity currents and slope failures, not by rivers: a land river is
never continued across the coast with river coefficients. Here:

* **sources** - river mouths whose v4 sediment delivery exceeds ``min_supply_m3_yr``
  (big deltas feed the slope); without supply there is no canyon and no fan;
* **path** - from the sea hex in front of the mouth down the steepest descent of the sea
  floor (six neighbours), strictly downhill, until the slope flattens below
  ``1.5 m/km`` (the base of the slope / the rise) or a local low is reached; a path never
  climbs, so a canyon cannot run from the deep sea up to the shelf;
* **canyon** - only where the path drops more than 500 m (it crosses the shelf break and
  the slope); depth ``D = 150 + 250 log10(1 + Qs / 1e5 m^3/yr)`` (100..1200 m) and width
  3-15 km are sub-hex: the hex gets the canyon's mean incision ``D W / spacing``;
* **fan** - at the end of the path the eroded canyon volume plus the share of the delivered
  sediment that bypasses the shelf (``bypass``, over the erosion duration) is spread as a
  cone ``h = h0 (1 - r / R)^2`` on the floor at and below the end depth. The deposited
  volume equals the supplied volume (closed budget, reported).
"""

from __future__ import annotations

import numpy as np


def submarine_sediment(
    z,
    ocean,
    river_mouth,
    delivery_m3_yr,
    nbr,
    layout,
    duration_yr: float,
    min_supply_m3_yr: float = 1.5e6,
    bypass: float = 0.25,
):
    shape = np.shape(z)
    zf = np.asarray(z, dtype=np.float64).ravel().copy()
    ocean = np.asarray(ocean, dtype=bool).ravel()
    mouth = np.asarray(river_mouth, dtype=bool).ravel()
    supply = np.asarray(delivery_m3_yr, dtype=np.float64).ravel()
    n = zf.size
    hx, hy = (a.ravel() for a in layout.centers())
    spacing = layout.spacing_km
    area_m2 = layout.hex_area_km2 * 1e6
    incision = np.zeros(n)
    fan = np.zeros(n)
    canyon = np.full(n, -1, dtype=np.int32)
    depth_sub = np.zeros(n)
    rows = []
    sources = np.nonzero(mouth & (supply >= min_supply_m3_yr))[0]
    sources = sources[np.argsort(-supply[sources], kind="stable")]
    for s in sources:
        # the sea hex in front of the mouth: the deepest ocean neighbour
        nb = nbr[s]
        nb = nb[(nb >= 0)]
        nb = nb[ocean[nb]]
        if nb.size == 0:
            continue
        cur = int(nb[np.argmin(zf[nb])])
        path = [cur]
        for _ in range(400):
            cand = nbr[cur]
            cand = cand[(cand >= 0)]
            cand = cand[ocean[cand]]
            if cand.size == 0:
                break
            nxt = int(cand[np.argmin(zf[cand])])
            drop = zf[cur] - zf[nxt]
            if drop <= 0.0:
                break  # a local low: stop, never climb
            path.append(nxt)
            cur = nxt
            if drop / spacing < 1.5 and len(path) > 2:
                break
        total_drop = zf[path[0]] - zf[path[-1]]
        qs = supply[s]
        fan_supply = bypass * qs * duration_yr
        eroded = 0.0
        cid = len(rows)
        if total_drop > 500.0 and len(path) >= 2:
            d_c = float(np.clip(150.0 + 250.0 * np.log10(1.0 + qs / 1e5), 100.0, 1200.0))
            w_c = float(np.clip(3.0 + 3.0 * np.log10(1.0 + qs / 1e6), 3.0, 15.0))
            for k in path[:-1]:
                mean = d_c * w_c / spacing
                extra = max(0.0, mean - incision[k])  # canyons sharing a hex do not add up
                incision[k] += extra
                depth_sub[k] = max(depth_sub[k], d_c)
                canyon[k] = cid if canyon[k] < 0 else canyon[k]
                eroded += extra * area_m2
        volume = fan_supply + eroded
        end = path[-1]
        if volume <= 0.0:
            continue
        # cone on the floor at and below the end depth
        radius = float(np.clip(60.0 + 8.0 * (volume / 1e9) ** (1.0 / 3.0), 60.0, 700.0))
        r = np.hypot(hx - hx[end], hy - hy[end])
        target = ocean & (r < radius) & (zf <= zf[end] + 200.0)
        if not target.any():
            target = np.zeros(n, dtype=bool)
            target[end] = True
        shape_w = np.where(target, (1.0 - np.clip(r / radius, 0.0, 1.0)) ** 2, 0.0)
        if shape_w.sum() <= 0.0:
            shape_w[end] = 1.0
        thick = volume * shape_w / (shape_w.sum() * area_m2)
        fan += thick
        rows.append(
            {
                "id": cid,
                "source_hex": int(s),
                "supply_m3_yr": round(float(qs), 1),
                "path_hexes": [int(k) for k in path],
                "path_km": [[round(float(hx[k]), 2), round(float(hy[k]), 2)] for k in path],
                "drop_m": round(float(total_drop), 1),
                "is_canyon": bool(total_drop > 500.0),
                "canyon_depth_m": round(float(depth_sub[path[0]]), 1)
                if total_drop > 500.0
                else 0.0,
                "eroded_m3": round(eroded, 1),
                "fan_volume_m3": round(volume, 1),
                "fan_radius_km": round(radius, 1),
                "provenance": "turbidity-current proxy from v4 river sediment delivery [А]",
            }
        )
    budget = {
        "canyon_eroded_m3": float((incision * area_m2).sum()),
        "shelf_bypass_m3": float(sum(r["fan_volume_m3"] - r["eroded_m3"] for r in rows)),
        "fan_deposited_m3": float((fan * area_m2).sum()),
    }
    budget["closure_error_m3"] = (
        budget["canyon_eroded_m3"] + budget["shelf_bypass_m3"] - budget["fan_deposited_m3"]
    )
    return {
        "incision_m": incision.reshape(shape),
        "canyon_depth_m": depth_sub.reshape(shape),
        "canyon_id": canyon.reshape(shape),
        "fan_m": fan.reshape(shape),
        "rows": rows,
        "budget": budget,
    }
