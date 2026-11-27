"""Secondary (supergene / surface) resources of TZ 5.2 sec. 17 and the transport of soil
elements down the rivers (sec. 10, test S23).

* routing with retention: on the drainage DAG the flux leaving hex i is
  ``F_i = s_i + sum_up (1 - r_j) F_j``; the deposit of hex i is ``r_i F_i`` and what leaves
  the window or reaches the sea is exported, so ``sum s = sum deposited + exported``
  exactly (a sparse triangular solve, no iteration);
* river placers: the heavy, resistant part of the *eroded* volume of primary bodies is
  released (``eta_release``) and routed; the deposit share rises where the stream loses
  slope (transport capacity) and in lakes; never more than the released mass (R13, R14);
* laterites (bauxite, Ni-Co) and ion-adsorption REE clays need the protolith *and* long
  intense weathering (the soil model's weathering degree and soil age), on gentle,
  drained ground (R04 of the soil-ore interface, TZ 17.2);
* closed-basin evaporites and Li brines: solutes delivered by the inflow over the basin
  age; salt precipitates only above saturation, Li is a brine mass (pore water x
  concentration, N16), never the mass of the basin rock (R10, R11);
* construction materials: a dense material class and quality per hex (R33).
"""

from __future__ import annotations

import math

import numpy as np
from scipy import sparse
from scipy.sparse import linalg as sla

from ..volcanism.units import stable_hash
from .deposits import LITH, LITHOLOGY, build_system
from .models import MODELS

ETA_RELEASE = 0.3  # share of an eroded heavy commodity freed as grains [A]
CLOSED_BASIN_AGE_YR = 1.0e6  # closed-basin history (assumption, flagged)
LI_BASE_MG_L = 0.02  # Li of ordinary inflow
LI_SOURCE_MG_L = 1.0  # inflow from felsic volcanics / geothermal springs in the catchment
LI_RETENTION = 0.1  # share of delivered Li kept in the brine (the rest in clays / halite)
AQUIFER_M = 30.0  # brine aquifer thickness of the salt nucleus
AQUIFER_POROSITY = 0.15
HALITE_SAT_KG_M3 = 360.0
LI_CAP_MG_L = 4000.0


def retention_route(receiver, source, retention):
    """Route ``source`` (n,) or (n, k) down ``receiver`` with retention shares ``r``.

    Returns ``(deposited, exported)``: deposited (n[, k]) and exported (k,) or a scalar.
    """
    rc = np.asarray(receiver, dtype=np.int64).ravel()
    n = rc.size
    r = np.clip(np.asarray(retention, dtype=np.float64).ravel(), 0.0, 1.0)
    i = np.nonzero((rc >= 0) & (rc < n) & (rc != np.arange(n)))[0]
    m = sparse.csr_matrix(((1.0 - r[i]), (rc[i], i)), shape=(n, n))
    lu = sla.splu((sparse.identity(n, format="csc") - m).tocsc())
    s = np.asarray(source, dtype=np.float64)
    flux = lu.solve(s) if s.ndim == 1 else lu.solve(np.ascontiguousarray(s))
    dep = (r[:, None] if s.ndim == 2 else r) * flux
    outlet = np.ones(n, dtype=bool)
    outlet[i] = False  # hexes with a receiver pass the rest on
    passed = ((1.0 - r)[:, None] if s.ndim == 2 else (1.0 - r)) * flux
    exported = passed[outlet].sum(axis=0)
    return dep, exported


def placer_retention(receiver, slope_m_km, lake, ocean):
    """Deposit share per hex: 3 % on graded reaches, up to 60 % where the slope drops
    (loss of transport capacity), all of it in lakes, none at sea (exported)."""
    rc = np.asarray(receiver, dtype=np.int64).ravel()
    n = rc.size
    s = np.maximum(np.nan_to_num(np.asarray(slope_m_km, dtype=np.float64).ravel()), 0.0)
    up_slope = np.zeros(n)
    i = np.nonzero((rc >= 0) & (rc < n))[0]
    np.maximum.at(up_slope, rc[i], s[i])
    drop = np.clip((up_slope - s) / np.maximum(up_slope, 1e-3), 0.0, 1.0)
    r = 0.03 + 0.57 * drop * (s < 8.0)
    r = np.where(np.asarray(lake).ravel(), 1.0, r)
    r = np.where(np.asarray(ocean).ravel(), 0.0, r)
    return np.clip(r, 0.0, 1.0)


def river_placers(
    ctx, catalog, receiver, slope_m_km, lake, ocean, river, spacing_km, min_mass_t=None
):
    """Placer systems from the released heavy commodities of eroded primary bodies.

    Returns ``(systems, bodies, ledger)``; the ledger has per element the released,
    deposited, exported masses (t) and ``residual``.
    """
    min_mass_t = min_mass_t or {"Au": 0.05, "Pt": 0.05, "Sn": 50.0, "diamond": 1e-4}
    n = ctx.n
    sources = {}
    for rec in catalog["eroded"]:
        model = MODELS[rec["model_id"]]
        for el in model.placer_elements:
            m = rec["eroded_commodity_t"].get(el, 0.0) * ETA_RELEASE
            if m > 0:
                sources.setdefault(el, []).append((rec["hex"], m, rec["system_id"]))
    retention = placer_retention(receiver, slope_m_km, lake, ocean)
    systems, bodies, ledger = [], [], {}
    for el, src in sorted(sources.items()):
        s = np.zeros(n)
        for h, m, _sid in src:
            s[h] += m
        dep, exported = retention_route(receiver, s, retention)
        released = float(s.sum())
        ledger[el] = {
            "released_t": released,
            "deposited_t": float(dep.sum()),
            "exported_t": float(exported),
            "residual_t": released - float(dep.sum()) - float(exported),
        }
        cand = np.nonzero(
            (dep >= min_mass_t.get(el, 1.0))
            & np.asarray(river).ravel()
            & ~np.asarray(ocean).ravel()
        )[0]
        cand = cand[np.argsort(-dep[cand], kind="stable")][:40]
        parents = sorted({sid for _h, _m, sid in src})
        for h in sorted(cand.tolist()):
            gravel_t = spacing_km * 1000.0 * 200.0 * 2.0 * 2000.0 / 1000.0  # 200 m x 2 m ribbon
            key = (el, int(round(ctx.x_km[h])), int(round(ctx.y_km[h])))
            sid = "D-" + stable_hash(ctx.seed, "placer", *key)[:12]
            grade = float(dep[h]) / gravel_t
            a = 0.5 * spacing_km * 1000.0
            body = {
                "body_id": f"{sid}-B0",
                "system_id": sid,
                "shape": "placer_ribbon",
                "center_m": [
                    float(ctx.x_km[h] * 1000.0),
                    float(ctx.y_km[h] * 1000.0),
                    float(ctx.surface_m[h] - 1.0),
                ],
                "axes_m": [a, 100.0, 1.0],
                "strike_deg": 0.0,
                "dip_deg": 0.0,
                "n_veins": 1,
                "z_top_m": float(ctx.surface_m[h]),
                "z_bottom_m": float(ctx.surface_m[h] - 2.0),
                "volume_m3": 2 * a * 200.0 * 2.0,
                "density_kg_m3": 2000.0,
                "f_barren": 0.0,
                "ore_t": gravel_t,
                "grades": {el: grade},
                "contained_t": {el: float(dep[h])},
                "formed_ma": 0.01,
                "erosion_fraction": 0.0,
                "eroded_above_m": None,
                "burial_depth_m": 0.0,
                "outcrop_area_m2": 2 * a * 200.0,
                "geometry_version": 1,
            }
            systems.append(
                {
                    "deposit_id": sid,
                    "model_id": "R20",
                    "genetic_class": "fluvial_placer",
                    "title": MODELS["R20"].title,
                    "family": "precious_metals"
                    if el in ("Au", "Pt")
                    else ("gems" if el == "diamond" else "critical_rare"),
                    "priority": "P0",
                    "anchor_kind": "river_graph",
                    "anchor_key": [str(k) for k in key],
                    "x_km": float(ctx.x_km[h]),
                    "y_km": float(ctx.y_km[h]),
                    "hex": int(h),
                    "formation_epoch_ma": 0.01,
                    "provenance": "routed_from_eroded_bodies",
                    "host_lithology": "alluvium",
                    "ore_t": gravel_t,
                    "ore_t_formed": gravel_t,
                    "grades": {el: grade},
                    "contained_t": {el: float(dep[h])},
                    "eroded_fraction": 0.0,
                    "preservation_state": "intact",
                    "depth_top_m": 0.0,
                    "outcrop_area_m2": body["outcrop_area_m2"],
                    "surface_occurrence": True,
                    "discovery_state": "undiscovered",
                    "quality": "synthetic_not_calibrated",
                    "body_ids": [body["body_id"]],
                    "parent_event_ids": parents[:20],
                    "source_pathway": f"erosion of {len(parents)} primary system(s) -> rivers",
                }
            )
            bodies.append(body)
    return systems, bodies, ledger


def weathering_tracts(ctx, soil_weathering, soil_age_yr, t_ann_c, p_ann_mm, floodplain):
    """Permissive masks and suitability of the supergene types (needs the soil model)."""
    w = np.nan_to_num(soil_weathering)
    age = np.nan_to_num(soil_age_yr)
    hot_wet = (t_ann_c > 20.0) & (p_ann_mm > 1200.0)
    stable = (ctx.slope_m_km < 12.0) & ~np.asarray(floodplain, dtype=bool) & ctx.land
    alumino = np.isin(
        ctx.lith,
        [
            LITH["granite_gneiss"],
            LITH["basaltic_volcanic"],
            LITH["platform_siliciclastic"],
            LITH["metamorphic_schist"],
            LITH["arc_volcanic"],
            LITH["black_shale"],
        ],
    )
    ultra = np.isin(ctx.lith, [LITH["ophiolite_ultramafic"], LITH["layered_mafic_intrusion"]])
    granite = np.isin(ctx.lith, [LITH["granite_gneiss"], LITH["granitoid_intrusion"]])
    s_w = np.clip((w - 0.8) / 0.2, 0.0, 1.0) * np.clip(
        np.log10(np.maximum(age, 1.0)) - 5.0, 0.0, 1.0
    )
    out = {
        "R17": (
            alumino & hot_wet & stable & (w > 0.85) & (age > 2e5),
            s_w,
            "intense long weathering of aluminosilicates",
        ),
        "R18": (
            ultra & hot_wet & stable & (w > 0.7) & (age > 1e5),
            np.maximum(s_w, 0.5),
            "ultramafic protolith + humid weathering",
        ),
        "R19": (
            granite & (t_ann_c > 15.0) & (p_ann_mm > 1000.0) & stable & (w > 0.6),
            np.clip((w - 0.6) / 0.4, 0, 1),
            "REE granite + weathering (ion clays)",
        ),
    }
    return {k: (m, np.where(m, s, 0.0), p) for k, (m, s, p) in out.items()}


def coastal_tracts(ctx, coast, wave_exposure):
    """Heavy-mineral beach sands: exposed sandy coasts fed by crystalline hinterland."""
    near_cryst = np.isin(
        ctx.lith,
        [
            LITH["granite_gneiss"],
            LITH["granitoid_intrusion"],
            LITH["metamorphic_schist"],
            LITH["unconsolidated_sediment"],
        ],
    )
    m = np.asarray(coast, bool) & ctx.land & (ctx.slope_m_km < 4.0) & near_cryst
    s = np.clip(np.nan_to_num(wave_exposure), 0.0, 1.0)
    return {"R21": (m & (s > 0.3), np.where(m, s, 0.0), "wave-sorted coastal sands")}


TRACT_LAMBDA_SECONDARY = {"R17": 2.5e-5, "R18": 3e-5, "R19": 1e-5, "R21": 1.5e-5}


def closed_basin_salts(
    ctx, lake_id, endorheic, discharge_m3_s, basin_id, aridity, lake_depth_m, li_source
):
    """Evaporite and Li-brine systems of closed basins (present climate, assumed age).

    Returns ``(systems, bodies, rows)``; rows report per basin the solute mass budget:
    delivered = dissolved + precipitated (salt) and Li in the brine.
    """
    lake_id = np.asarray(lake_id).ravel()
    endo = np.asarray(endorheic, bool).ravel()
    q = np.nan_to_num(np.asarray(discharge_m3_s, float).ravel())
    bid = np.asarray(basin_id).ravel()
    ar = np.nan_to_num(np.asarray(aridity, float).ravel(), nan=1.0)
    depth = np.nan_to_num(np.asarray(lake_depth_m, float).ravel())
    systems, bodies, rows = [], [], []
    area_m2 = ctx.hex_area_km2 * 1e6
    for lk in np.unique(lake_id[(lake_id >= 0) & endo]):
        hexes = np.nonzero(lake_id == lk)[0]
        # inflow: the largest discharge entering the lake (m3/yr)
        inflow = float(q[hexes].max()) * 365.25 * 86400.0
        a_mean = float(ar[hexes].mean())
        if inflow <= 0 or a_mean > 0.5:
            continue  # only arid closed basins concentrate solutes
        basins = set(bid[hexes].tolist())
        catch = np.isin(bid, list(basins))
        c_salt = 0.15 + 0.5 * float(
            np.mean(ctx.lith[catch] == LITH["red_beds_evaporitic"])
        )  # kg/m3
        salt_kg = inflow * c_salt * CLOSED_BASIN_AGE_YR
        water_m3 = max(float(depth[hexes].sum()) * area_m2, 1.0)
        dissolved = min(salt_kg, HALITE_SAT_KG_M3 * water_m3)
        solid = salt_kg - dissolved
        has_src = bool(np.any(np.asarray(li_source).ravel()[catch]))
        li_conc_in = LI_BASE_MG_L + (LI_SOURCE_MG_L if has_src else 0.0)  # mg/L
        li_kg = inflow * li_conc_in * 1e-3 * CLOSED_BASIN_AGE_YR * LI_RETENTION
        sed = float(np.clip(ctx.sediment_m[hexes].mean(), 0.0, AQUIFER_M))
        pore_m3 = hexes.size * area_m2 * sed * AQUIFER_POROSITY + water_m3
        li_mg_l = min(li_kg / pore_m3 * 1e3, LI_CAP_MG_L)
        li_brine_kg = pore_m3 * li_mg_l * 1e-3
        h0 = int(hexes[0])
        rows.append(
            {
                "lake_id": int(lk),
                "hexes": int(hexes.size),
                "inflow_m3_yr": inflow,
                "salt_delivered_kg": salt_kg,
                "salt_dissolved_kg": dissolved,
                "salt_precipitated_kg": solid,
                "li_delivered_kg": li_kg,
                "li_brine_kg": li_brine_kg,
                "li_mg_l": li_mg_l,
                "pore_water_m3": pore_m3,
                "li_source_in_catchment": has_src,
                "li_retention": LI_RETENTION,
                "age_yr": CLOSED_BASIN_AGE_YR,
                "quality": "assumed_basin_age",
            }
        )
        x, y = float(ctx.x_km[hexes].mean()), float(ctx.y_km[hexes].mean())
        if solid > 0:
            sid = "D-" + stable_hash(ctx.seed, "evaporite", int(round(x)), int(round(y)))[:12]
            area = hexes.size * area_m2
            thick = solid / 2170.0 / area
            a = math.sqrt(area / math.pi)
            body = _bed_body(
                sid, x, y, float(ctx.surface_m[h0]), a, thick, solid / 1000.0, {"NaCl": 0.9}, 2170.0
            )
            systems.append(
                _secondary_system(
                    sid, "R26", x, y, h0, body, "closed_basin_today", {"NaCl": 0.9}, ctx
                )
            )
            bodies.append(body)
        if li_mg_l >= 100.0:
            sid = "D-" + stable_hash(ctx.seed, "li-brine", int(round(x)), int(round(y)))[:12]
            a = math.sqrt(hexes.size * area_m2 / math.pi)
            body = _bed_body(
                sid,
                x,
                y,
                float(ctx.surface_m[h0]),
                a,
                max(sed, 1.0) / 2.0,
                pore_m3 * 1.2 / 1000.0,
                {},
                1200.0,
            )  # brine mass (t of water)
            body["contained_t"] = {"Li": li_brine_kg / 1000.0}
            body["grades"] = {"Li_mg_per_l": li_mg_l}
            s_ = _secondary_system(sid, "R27", x, y, h0, body, "closed_basin_brine", {}, ctx)
            s_["contained_t"] = {"Li": li_brine_kg / 1000.0}
            s_["brine"] = {"pore_water_m3": pore_m3, "li_mg_l": li_mg_l}
            systems.append(s_)
            bodies.append(body)
    return systems, bodies, rows


def _bed_body(sid, x, y, z_surface, a, thick, ore_t, grades, rho):
    return {
        "body_id": f"{sid}-B0",
        "system_id": sid,
        "shape": "evaporite_bed",
        "center_m": [x * 1000.0, y * 1000.0, z_surface - thick / 2.0],
        "axes_m": [a, a, thick / 2.0],
        "strike_deg": 0.0,
        "dip_deg": 0.0,
        "n_veins": 1,
        "z_top_m": z_surface,
        "z_bottom_m": z_surface - thick,
        "volume_m3": math.pi * a * a * thick,
        "density_kg_m3": rho,
        "f_barren": 0.0,
        "ore_t": ore_t,
        "grades": dict(grades),
        "contained_t": {el: ore_t * g for el, g in grades.items()},
        "formed_ma": 0.05,
        "erosion_fraction": 0.0,
        "eroded_above_m": None,
        "burial_depth_m": 0.0,
        "outcrop_area_m2": math.pi * a * a,
        "geometry_version": 1,
    }


def _secondary_system(sid, mid, x, y, h, body, prov, grades, ctx):
    m = MODELS[mid]
    return {
        "deposit_id": sid,
        "model_id": mid,
        "genetic_class": m.name,
        "title": m.title,
        "family": m.family,
        "priority": m.priority,
        "anchor_kind": "closed_basin",
        "anchor_key": [str(int(round(x))), str(int(round(y)))],
        "x_km": x,
        "y_km": y,
        "hex": int(h),
        "formation_epoch_ma": 0.05,
        "provenance": prov,
        "host_lithology": LITHOLOGY[int(ctx.lith[h])],
        "ore_t": body["ore_t"],
        "ore_t_formed": body["ore_t"],
        "grades": dict(grades),
        "contained_t": dict(body["contained_t"]),
        "eroded_fraction": 0.0,
        "preservation_state": "intact",
        "depth_top_m": 0.0,
        "outcrop_area_m2": body["outcrop_area_m2"],
        "surface_occurrence": True,
        "discovery_state": "undiscovered",
        "quality": "synthetic_not_calibrated",
        "body_ids": [body["body_id"]],
        "parent_event_ids": [],
    }


CONSTRUCTION = (
    "none",
    "sand_gravel",
    "crushed_granite",
    "crushed_basalt",
    "limestone",
    "clay",
    "gypsum",
    "dimension_stone",
)
CONSTRUCTION_TITLES = (
    "нет",
    "песок и гравий",
    "щебень (граниты)",
    "щебень (базальты)",
    "известняк / доломит",
    "глины",
    "гипс",
    "облицовочный камень",
)


def construction_materials(ctx, parent_material, soil_depth_m, weathering):
    """Dense class (``CONSTRUCTION``) and quality 0..1 per land hex (R33)."""
    from ..stages.environment import PM

    pm = np.asarray(parent_material).ravel()
    cls = np.zeros(ctx.n, dtype=np.uint8)
    L = ctx.lith
    cls[np.isin(L, [LITH["granite_gneiss"], LITH["granitoid_intrusion"]])] = 2
    cls[np.isin(L, [LITH["basaltic_volcanic"], LITH["arc_volcanic"]])] = 3
    cls[L == LITH["carbonate_platform"]] = 4
    cls[np.isin(L, [LITH["black_shale"]])] = 5
    cls[L == LITH["red_beds_evaporitic"]] = 6
    cls[L == LITH["metamorphic_schist"]] = 7
    loose = np.isin(pm, [PM["alluvium"], PM["glacial_till"], PM["aeolian_sand"], PM["colluvium"]])
    cls[loose] = 1
    cls[pm == PM["marine_lacustrine"]] = 5
    cls[~ctx.land] = 0
    rock_cls = (cls >= 2) & (cls != 5)
    cover = np.where(
        rock_cls,
        np.exp(
            -np.maximum(np.nan_to_num(soil_depth_m), 0.0) / 5.0 - ctx.sediment_m * rock_cls / 30.0
        ),
        1.0,
    )
    fresh = 1.0 - 0.6 * np.clip(np.nan_to_num(weathering), 0.0, 1.0) * rock_cls
    access = np.clip(1.0 - ctx.slope_m_km / 80.0, 0.2, 1.0)
    q = np.where(cls > 0, np.clip(cover * fresh * access, 0.0, 1.0), 0.0)
    return cls, q.astype(np.float32)


def tract_systems(ctx, prospect, lam, lambda_scale=1.0):
    """Poisson systems of extra tracts (same process as the primary tract types)."""
    from .deposits import _provinces, _stream, _unique_pid, poisson_from_uniform

    systems, bodies, provinces = [], [], []
    area = ctx.hex_area_km2
    used_pids: set = set()
    for mid in sorted(prospect):
        mask, s, prov = prospect[mid]
        if not mask.any():
            continue
        lab, k = _provinces(mask.reshape(ctx.shape))
        sums = np.bincount(lab, weights=s * area, minlength=k + 1)
        order = np.argsort(lab, kind="stable")
        starts = np.searchsorted(lab[order], np.arange(k + 2))
        for p in range(1, k + 1):
            idx = order[starts[p] : starts[p + 1]]
            if sums[p] <= 0 or idx.size == 0:
                continue
            w = s[idx]
            pcx = float((ctx.x_km[idx] * w).sum() / w.sum())
            pcy = float((ctx.y_km[idx] * w).sum() / w.sum())
            ph = stable_hash(ctx.seed, "province", mid, round(pcx / 100.0), round(pcy / 100.0))
            pid = _unique_pid(f"P-{mid}-{ph[:10]}", used_pids)
            L_ = lam[mid] * lambda_scale * sums[p]
            count = poisson_from_uniform(L_, _stream(ctx.seed, mid, pid, "count").random())
            provinces.append(
                {
                    "province_id": pid,
                    "model_id": mid,
                    "x_km": round(pcx, 2),
                    "y_km": round(pcy, 2),
                    "area_km2": float(idx.size * area),
                    "suitability_integral_km2": float(sums[p]),
                    "lambda": L_,
                    "count": count,
                    "provenance": prov,
                }
            )
            cw = np.cumsum(w)
            for kk in range(count):
                ua = _stream(ctx.seed, mid, pid, "anchor", kk).random(3)
                j = int(np.searchsorted(cw, ua[0] * cw[-1], side="right"))
                i = int(idx[min(j, idx.size - 1)])
                r = 0.45 * ctx.spacing_km * math.sqrt(ua[1])
                x = float(ctx.x_km[i] + r * math.cos(2 * math.pi * ua[2]))
                y = float(ctx.y_km[i] + r * math.sin(2 * math.pi * ua[2]))
                s_, b_, _rec = build_system(
                    ctx,
                    mid,
                    x,
                    y,
                    i,
                    (pid, kk),
                    "modern",
                    {"province_id": pid, "parent_event_ids": [pid]},
                )
                if s_ is not None:
                    # supergene caps sit at the surface: the top is the present surface
                    for b in b_:
                        top = float(ctx.surface_m[i])
                        c = b["axes_m"][2]
                        b["center_m"][2] = top - c
                        b["z_top_m"], b["z_bottom_m"] = top, top - 2 * c
                        b["burial_depth_m"] = 0.0
                        b["outcrop_area_m2"] = math.pi * b["axes_m"][0] * b["axes_m"][1]
                    s_["depth_top_m"] = 0.0
                    s_["surface_occurrence"] = True
                    s_["outcrop_area_m2"] = sum(b["outcrop_area_m2"] for b in b_)
                    s_["provenance"] = "supergene_present_climate"
                    systems.append(s_)
                    bodies.extend(b_)
    return systems, bodies, provinces
