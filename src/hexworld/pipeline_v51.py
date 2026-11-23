"""HexWorld v5.1: volcanism of the recent window (stage ``v51``, between v4.7 and v5.0).

Inputs (never modified): v3 volcanoes, hotspots and arcs; v4.6 ridges and rifts; v4 winds,
drainage, sea level, glacial potential; v4.5 parent material; v4.7 perennial snow.

Steps::

    provinces -> systems (+ magma, reservoir) -> scheduled eruptions (deterministic ids)
      -> explicit events: lava (local D6 lattice), tephra (hex FV), PDC (energy cone +
         routing), lahars (water + sediment), subglacial melt / outbursts, emergence
      -> aggregated events: conserved totals, analytic fall field
      -> hex layers + ecology forcing (consumed by v5.0) + climate pulse record

The terrain adapter writes deposit thickness, new land and dam flags as *layers*; the v4
elevation, rivers and lakes are not rewritten (documented limitation).
"""

from __future__ import annotations

import hashlib
import json
import math
import time
from collections import Counter
from dataclasses import asdict

import numpy as np

from .project import VolcanismData
from .version import VOLCANISM_ALGORITHM_VERSION
from .volcanism import environment as venv
from .volcanism.hazards import run_pdc
from .volcanism.lahars import event_lahar
from .volcanism.lava import run_flow
from .volcanism.provinces import build_provinces
from .volcanism.scheduler import aggregate, schedule, select_explicit
from .volcanism.systems import build_systems
from .volcanism.tephra import fast_deposit, transport
from .volcanism.terrain import BULK_DENSITY, apply_deltas, delta_from_volumes, rebuild

HYDROTHERMAL_KINDS = (
    "none",
    "black_smoker",
    "diffuse_low_temperature",
    "shallow_hydrothermal",
    "subglacial_geothermal",
    "hot_spring_fumarole",
)
#: Mean thickness of aggregated (non-explicit) lava, m, used to turn volume into area [H].
AGG_LAVA_THICKNESS_M = 6.0


def _key(*parts) -> str:
    return hashlib.sha256(json.dumps(parts, sort_keys=True, default=str).encode()).hexdigest()


def upstream_keys(project) -> dict:
    topo = getattr(project, "topo", None)
    terrain = getattr(project, "terrain", None)
    return {
        "hist": project.hist.stage_keys if project.hist is not None else {},
        "terrain": terrain.stage_keys if terrain is not None else {},
        "hydro": project.hydro.stage_keys if project.hydro is not None else {},
        "env": project.env.stage_keys if project.env is not None else {},
        "topo": topo.stage_keys if topo is not None else {},
        **({"earth": project.earth.stage_keys} if getattr(project, "earth", None) else {}),
        **(
            {"atmosphere": project.atmosphere.stage_keys}
            if getattr(project, "atmosphere", None) is not None
            else {}
        ),  # fmt: skip
    }


def volcanism_stage_keys(config, upstream: dict) -> dict:
    return {
        "v51_volcanism": _key(
            "v51", VOLCANISM_ALGORITHM_VERSION, config.seed, upstream, asdict(config.volcanism)
        )
    }


def explicit_limit(max_events: int, n_hex: int) -> int:
    """At most ``max_events``; ~24 + 1 per 1000 hexes (a 100x100 world gets 34)."""
    return int(min(max(int(max_events), 0), 24 + n_hex // 1000))


def recovery_years(temp_c: np.ndarray, precip_mm: np.ndarray) -> np.ndarray:
    """``[H]`` years for plants to close over fresh lava / PDC deposits.

    Wet tropics ~150 yr, temperate ~400 yr, cold or dry climates > 1500 yr
    (primary succession on Hawaiian, Icelandic and Andean flows, order of magnitude).
    """
    t = np.nan_to_num(np.asarray(temp_c, dtype=np.float64), nan=0.0)
    p = np.nan_to_num(np.asarray(precip_mm, dtype=np.float64), nan=0.0)
    cold = np.clip((22.0 - t) / 30.0, 0.0, 1.0)
    dry = np.clip((1200.0 - p) / 1000.0, 0.0, 1.0)
    return 150.0 * (1.0 + 4.0 * cold) * (1.0 + 2.5 * dry)


def run_volcanism(config, project, report=None) -> dict:
    rep = report or (lambda f: None)
    prm = config.volcanism
    seed = str(config.seed)
    lay = project.layout
    n = lay.width * lay.height
    shape = (lay.height, lay.width)
    timings = {}
    warnings = []
    t0 = time.perf_counter()
    provinces, extra = build_provinces(project, prm)
    systems = build_systems(project, provinces, extra["v3_province"], prm)
    icy = prm.world_profile == "icy_world"
    if icy:
        for s in systems:
            venv.icy_world_overrides(s)
    timings["v51_systems"] = round(time.perf_counter() - t0, 3)
    rep(0.1)

    # -- eruptions --------------------------------------------------------------------------
    t0 = time.perf_counter()
    H = project.hydro.layers
    lat_hex = H["latitude_deg"].ravel()
    events = []
    by_id = {s.volcano_id: s for s in systems}
    for s in systems:
        lat = float(lat_hex[s.hex[1] * lay.width + s.hex[0]]) if s.hex else 30.0
        events += schedule(s, seed, prm, lat)
    if icy:
        for e in events:  # cryomagma: no sulfur, no silicate column climate effect
            e.products["so2_mass_kg"] = 0.0
            e.products["so2_stratospheric_kg"] = 0.0
            e.quality_flags = (*e.quality_flags, "cryovolcanic")
    in_window = {s.volcano_id for s in systems if s.in_window}
    # explicit products cost per event, not per hex: a small world gets fewer of them
    limit = explicit_limit(prm.max_explicit_events, n)
    select_explicit(events, limit, in_window)
    explicit = [e for e in events if e.explicit and e.volcano_id in in_window]
    agg = aggregate(events)
    timings["v51_schedule"] = round(time.perf_counter() - t0, 3)
    rep(0.2)

    # -- per-hex accumulators ------------------------------------------------------------------
    t_mean = np.nan_to_num(H["temperature_mean_c"].ravel().astype(np.float64))
    p_ann = np.nan_to_num(H["precipitation_mm_yr"].ravel().astype(np.float64))
    t_rec = recovery_years(t_mean, p_ann)
    hex_area_m2 = lay.hex_area_km2 * 1e6
    lava_frac = np.zeros(n)
    lava_vol = np.zeros(n)
    lava_age = np.full(n, np.inf)
    barren_keep = np.ones(n)  # prod(1 - f_barren): union of independent barren shares
    tephra_load = np.zeros(n)
    tephra_recent = np.zeros(n)  # load weighted by exp(-age / 50 yr)
    pdc_hazard = np.zeros(n)
    pdc_dep = np.zeros(n)
    pdc_vol = np.zeros(n)
    lahar_frac = np.zeros(n)
    lahar_vol = np.zeros(n)
    new_land = np.zeros(n)
    budgets = Counter()
    ice_proxy = venv.ice_cover_proxy(project)
    earth = getattr(project, "earth", None)
    if earth is not None and "land_ice_thickness_m" in earth.layers:
        # the v5.3 ice sheet: subglacial melt from the actual ice mass, not a proxy
        ice_proxy = earth.layers["land_ice_thickness_m"].astype(np.float64).ravel()
        ice_quality = "v53_ice_sheet"
    else:
        ice_quality = "proxy_glacial_potential_approximate"
    deltas = []
    glac = np.nan_to_num(H["glacial_potential"].ravel().astype(np.float64))

    def barren_add(idx, frac, age):
        r = np.clip(1.0 - age / t_rec[idx], 0.0, 1.0)
        np.multiply.at(barren_keep, idx, 1.0 - np.clip(frac * r, 0.0, 0.95))

    t_lava = t_pdc = t_teph = t_lah = 0.0
    results = {}
    subglacial, pulses, emergences = [], [], []
    for i, e in enumerate(explicit):
        s = by_id[e.volcano_id]
        res: dict = {}
        tt = time.perf_counter()
        fl = run_flow(project, s, e, seed, prm.lava_cell_m, prm.physics_quality)
        t_lava += time.perf_counter() - tt
        if fl is not None:
            hx = fl["hex_index"]
            np.add.at(lava_frac, hx, fl["hex_fraction"])
            deltas.append(delta_from_volumes(e.event_id, "lava", hx, fl["hex_volume_m3"],
                                             hex_area_m2, e.years_bp))  # fmt: skip
            np.add.at(lava_vol, hx, fl["hex_volume_m3"])
            np.minimum.at(lava_age, hx, e.years_bp)
            barren_add(hx, fl["hex_fraction"], e.years_bp)
            if fl["new_land_hex"].size:
                np.add.at(new_land, fl["new_land_hex"], fl["new_land_km2"])
            budgets["lava_injected_m3"] += fl["injected_m3"]
            budgets["lava_residual_m3"] += abs(fl["budget_residual_m3"])
            budgets["lava_escaped_m3"] += fl["escaped_m3"]
            res["lava"] = {
                k: fl[k]
                for k in (
                    "area_km2",
                    "length_km",
                    "max_thickness_m",
                    "cooling_length_km",
                    "effusion_m3_s",
                    "cell_m",
                    "escaped_m3",
                    "quality",
                )
            }
        tt = time.perf_counter()
        pdc = run_pdc(project, s, e, seed)
        t_pdc += time.perf_counter() - tt
        if pdc is not None:
            np.maximum.at(pdc_hazard, pdc["hazard_hex"], pdc["hazard_fraction"])
            np.add.at(pdc_dep, pdc["deposit_hex"], pdc["deposit_fraction"])
            np.add.at(pdc_vol, pdc["deposit_hex"], pdc["deposit_volume_m3"])
            deltas.append(delta_from_volumes(e.event_id, "pdc", pdc["deposit_hex"],
                                             pdc["deposit_volume_m3"], hex_area_m2,
                                             e.years_bp))  # fmt: skip
            barren_add(pdc["deposit_hex"], pdc["deposit_fraction"], e.years_bp * 2.0)
            budgets["pdc_m3"] += pdc["volume_m3"]
            budgets["pdc_residual_m3"] += abs(pdc["volume_m3"] - pdc["placed_m3"])
            res["pdc"] = {
                k: pdc[k]
                for k in (
                    "volume_m3",
                    "heim_mu",
                    "collapse_height_m",
                    "hazard_area_km2",
                    "runout_km",
                    "quality",
                )
            }
        tt = time.perf_counter()
        tp = transport(project, e, prm.tephra_bins, prm.physics_quality)
        t_teph += time.perf_counter() - tt
        if tp is not None:
            np.add.at(tephra_load, tp["hex_index"], tp["load_kg_m2"])
            deltas.append(delta_from_volumes(
                e.event_id, "tephra", tp["hex_index"],
                tp["load_kg_m2"] * hex_area_m2 / BULK_DENSITY["tephra"], hex_area_m2,
                e.years_bp))  # fmt: skip
            np.add.at(tephra_recent, tp["hex_index"], tp["load_kg_m2"] * math.exp(-e.years_bp / 50))
            for k in ("source_kg", "deposited_kg", "exported_kg", "airborne_kg"):
                budgets[f"tephra_{k}"] += tp[k]
            budgets["tephra_residual_kg"] += abs(tp["budget_residual_kg"])
            res["tephra"] = {
                k: tp[k]
                for k in (
                    "deposited_kg",
                    "exported_kg",
                    "airborne_kg",
                    "area_over_1mm_km2",
                    "bin_kg",
                    "steps",
                    "quality_flags",
                )
            }
        tt = time.perf_counter()
        if s.hex is not None:
            fh = s.hex[1] * lay.width + s.hex[0]
            lh = event_lahar(
                project, s, e, float(glac[fh]), float(p_ann[fh]), pdc["volume_m3"] if pdc else 0.0
            )
            if lh is not None:
                np.add.at(lahar_frac, lh["hex_index"], lh["hex_fraction"])
                np.add.at(lahar_vol, lh["hex_index"], lh["deposit_volume_m3"])
                barren_add(lh["hex_index"], lh["hex_fraction"] * 0.5, e.years_bp * 4.0)
                budgets["lahar_m3"] += lh["volume_m3"]
                budgets["lahar_residual_m3"] += abs(lh["budget_residual_m3"])
                deltas.append(delta_from_volumes(e.event_id, "lahar_sediment", lh["hex_index"],
                                                 lh["deposit_volume_m3"], hex_area_m2,
                                                 e.years_bp))  # fmt: skip
                res["lahar"] = {
                    k: lh[k]
                    for k in (
                        "volume_m3",
                        "water_m3",
                        "sediment_m3",
                        "runout_hexes",
                        "to_sea_or_lake_m3",
                    )
                }
            if prm.subglacial and ice_proxy is not None:
                sg = venv.subglacial_melt(project, s, e, float(ice_proxy[fh]))
                if sg is not None:
                    ob = sg.pop("outburst", None)
                    if ob is not None:
                        np.add.at(lahar_frac, ob["hex_index"], ob["hex_fraction"] * 0.3)
                        sg["outburst_peak_m3_s"] = ob["peak_discharge_m3_s"]
                        sg["outburst_runout_hexes"] = ob["runout_hexes"]
                    sg["event_id"] = e.event_id
                    subglacial.append(sg)
                    res["subglacial"] = {k: v for k, v in sg.items() if k != "event_id"}
        t_lah += time.perf_counter() - tt
        e.results = res
        results[e.event_id] = res
        rep(0.2 + 0.65 * (i + 1) / max(len(explicit), 1))
    timings.update(
        {
            "v51_lava": round(t_lava, 3),
            "v51_pdc": round(t_pdc, 3),
            "v51_tephra": round(t_teph, 3),
            "v51_lahars": round(t_lah, 3),
        }
    )

    if prm.climate_aerosol:  # every event with stratospheric SO2, explicit or not
        for e in events:
            cp = venv.climate_pulse(e)
            if cp is not None:
                pulses.append(cp)
    # freshness of the aggregated lava: sum of V * max(0, 1 - age / t_rec) per volcano
    fresh = Counter()
    for e in events:
        s = by_id[e.volcano_id]
        if e.explicit or s.hex is None:
            continue
        fh = s.hex[1] * lay.width + s.hex[0]
        fresh[e.volcano_id] += e.products["lava_volume_m3"] * max(0.0, 1.0 - e.years_bp / t_rec[fh])
    # -- aggregated history: conserved totals, analytic fields ------------------------------------
    t0 = time.perf_counter()
    for vid, a in agg.items():
        s = by_id[vid]
        if s.hex is None:
            continue
        fh = s.hex[1] * lay.width + s.hex[0]
        area = a["lava_m3"] / AGG_LAVA_THICKNESS_M
        f = min(0.6, area / hex_area_m2)
        lava_frac[fh] += f
        lava_vol[fh] += a["lava_m3"]
        lava_age[fh] = min(lava_age[fh], a["newest_years_bp"])
        deltas.append(delta_from_volumes(f"AGG-{vid}", "lava", [fh], [a["lava_m3"]],
                                         hex_area_m2, a["newest_years_bp"],
                                         method="aggregated_events_vent_hex"))  # fmt: skip
        if s.environment not in ("deep_submarine", "shallow_submarine") and a["lava_m3"] > 0:
            f_fresh = f * fresh[vid] / a["lava_m3"]
            np.multiply.at(barren_keep, [fh], 1.0 - min(f_fresh, 0.95))
        if a["tephra_kg"] > 0:
            fd = fast_deposit(project, s.x_km, s.y_km, a["tephra_kg"], 4000.0)
            if fd is not None:
                np.add.at(tephra_load, fd["hex_index"], fd["load_kg_m2"])
                deltas.append(delta_from_volumes(
                    f"AGG-{vid}", "tephra", fd["hex_index"],
                    fd["load_kg_m2"] * hex_area_m2 / BULK_DENSITY["tephra"], hex_area_m2,
                    a["newest_years_bp"], method="aggregated_fast_analytic"))  # fmt: skip
                budgets["tephra_fast_kg"] += fd["deposited_kg"]
    timings["v51_aggregates"] = round(time.perf_counter() - t0, 3)

    # -- systems: hydrothermal heat, degassing stress, emergence -----------------------------------
    hydro_kind = np.zeros(n, dtype=np.uint8)
    heat_mw = np.zeros(n)
    degas = np.zeros(n)
    co2_flag = np.zeros(n, dtype=bool)
    vent_count = np.zeros(n, dtype=np.int32)
    status_code = np.zeros(n, dtype=np.uint8)
    STATUS = ("none", "extinct_system", "uncertain", "dormant_system", "active_system")
    no_out = H["no_outflow_mask"].ravel() if "no_outflow_mask" in H else np.zeros(n, bool)
    erupted = Counter()
    for e in events:
        erupted[e.volcano_id] += e.products["volume_dre_m3"]
    fields = []
    for s in systems:
        if s.hex is None:
            continue
        fh = s.hex[1] * lay.width + s.hex[0]
        vent_count[fh] += 1
        status_code[fh] = max(status_code[fh], STATUS.index(s.magmatic_status))
        if prm.hydrothermal:
            hf = venv.hydrothermal_field(s)
            if hf is not None:
                fields.append(hf)
                heat_mw[fh] += hf["heat_mw"]
                hydro_kind[fh] = max(hydro_kind[fh], HYDROTHERMAL_KINDS.index(hf["kind"]))
        L = venv.degassing_stress(s)
        if L > 0:
            degas[fh] = max(degas[fh], venv.stress_fraction_of_hex(L, lay.radius))
        if (
            s.edifice_type in ("caldera_complex", "maar_tuff_ring")
            and no_out[fh]
            and (s.magmatic_status in ("active_system", "dormant_system"))
        ):
            co2_flag[fh] = True
        em = venv.emergence(s, erupted[s.volcano_id])
        if em is not None:
            emergences.append(em)
    mud = extra.get("mud_vents", []) if prm.mud_volcanoes else []
    mud_flag = np.zeros(n, dtype=bool)
    for m in mud:
        c, r = lay.world_to_offset(m["x_km"], m["y_km"])
        if lay.inside(c, r):
            mud_flag[int(r) * lay.width + int(c)] = True

    # -- hex layers --------------------------------------------------------------------------------
    lava_frac = np.minimum(lava_frac, 1.0)
    pdc_dep = np.minimum(pdc_dep, 1.0)
    lahar_frac = np.minimum(lahar_frac, 1.0)
    barren = np.clip(1.0 - barren_keep, 0.0, 0.95)
    # thick fresh ash (> ~0.3 m and < ~50 yr) also kills the vegetation of part of the hex
    ash_kill = np.clip((tephra_recent - 300.0) / 3000.0, 0.0, 0.8)
    barren = 1.0 - (1.0 - barren) * (1.0 - ash_kill)
    ocean = H["ocean_mask"].ravel().astype(bool)
    barren[ocean] = 0.0
    ash_stress = np.clip(np.log10(np.maximum(tephra_recent, 1e-9) / 10.0) / 2.0, 0.0, 1.0)
    deposit_m = (lava_vol + pdc_vol + lahar_vol) / hex_area_m2 + tephra_load / 1000.0
    river = H["river_mask"].ravel().astype(bool)
    rdepth = np.nan_to_num(H["river_depth_m"].ravel().astype(np.float64))
    dam = (
        river
        & ((lava_frac > 0.01) | (lahar_frac > 0.01) | (pdc_dep > 0.01))
        & (
            (lava_vol + pdc_vol + lahar_vol)
            / np.maximum(hex_area_m2 * np.maximum(lava_frac + pdc_dep + lahar_frac, 1e-6), 1.0)
            > np.maximum(rdepth, 1.0)
        )
    )
    hazard = np.clip(
        np.maximum.reduce(
            [
                pdc_hazard,
                0.6 * lahar_frac / 0.3,
                0.5 * np.clip(tephra_load / 100.0, 0, 1),
                0.8 * lava_frac / 0.2,
            ]
        ),
        0,
        1,
    )
    hazard[ocean] = 0.0  # a hazard index for land use: the sea floor is not scored

    # -- G7: the deposits change the surface; drainage, lakes, ocean rebuilt (snapshot) -------
    t0 = time.perf_counter()
    terrain_layers: dict = {}
    terrain_summary: dict = {"committed": False}
    terrain_quality = "not_applied"
    if getattr(prm, "apply_terrain", True) and deltas:
        z_pre = H["elevation_m"].astype(np.float64).ravel()
        z_post, applied = apply_deltas(z_pre, deltas, hex_area_m2)
        if applied["committed"]:
            reb = rebuild(project, z_post)
            terrain_layers = {
                "post_volcanic_elevation_m": z_post.reshape(shape).astype(np.float32),
                "volcanic_dz_m": (z_post - z_pre).reshape(shape).astype(np.float32),
                "post_volcanic_flow_receiver": reb["receiver"].reshape(shape).astype(np.int64),
                "post_volcanic_lake_id": reb["lake_id"].reshape(shape).astype(np.int32),
                "post_volcanic_discharge_m3_s": reb["discharge_m3_s"].reshape(shape).astype(
                    np.float32),
                "post_volcanic_ocean_mask": reb["ocean"].reshape(shape),
                "volcanic_new_lake": reb["new_lake"].reshape(shape),
                "volcanic_drainage_changed": reb["drainage_changed"].reshape(shape),
                "volcanic_new_land": reb["new_land"].reshape(shape),
            }  # fmt: skip
            terrain_summary = {**{k: v for k, v in applied.items() if k != "applied_keys"},
                               "applied_events": len(applied["applied"]),
                               **reb["summary"]}  # fmt: skip
            terrain_summary.pop("applied", None)
            terrain_quality = "applied_snapshot_full_drainage_rebuild"
        else:
            terrain_summary = applied
            warnings.append(f"v5.1 terrain delta rejected: {applied.get('error')}")
    timings["v51_terrain"] = round(time.perf_counter() - t0, 3)

    def g(a, dtype=np.float32):
        return np.ascontiguousarray(np.asarray(a).reshape(shape).astype(dtype))

    layers = {
        "volcano_count": g(vent_count, np.int32),
        "volcano_status": g(status_code, np.uint8),
        "lava_fraction": g(lava_frac),
        "lava_volume_m3": g(lava_vol),
        "lava_youngest_years_bp": g(np.where(np.isfinite(lava_age), lava_age, np.nan)),
        "tephra_load_kg_m2": g(tephra_load),
        "tephra_thickness_m": g(tephra_load / 1000.0),
        "pdc_hazard_fraction": g(pdc_hazard),
        "pdc_deposit_fraction": g(pdc_dep),
        "lahar_fraction": g(lahar_frac),
        "volcanic_hazard_index": g(hazard),
        "volcanic_barren_fraction": g(barren),
        "volcanic_ash_stress": g(ash_stress),
        "degassing_stress_fraction": g(degas),
        "co2_pooling_flag": g(co2_flag, np.bool_),
        "mud_volcano_flag": g(mud_flag, np.bool_),
        "geothermal_heat_mw": g(heat_mw),
        "hydrothermal_kind": g(hydro_kind, np.uint8),
        "vegetation_recovery_years": g(t_rec),
        "volcanic_deposit_delta_m": g(deposit_m),
        "volcanic_new_land_km2": g(new_land),
        "volcanic_dam_flag": g(dam, np.bool_),
        **terrain_layers,
    }
    # -- summary and budgets -----------------------------------------------------------------------
    styles = Counter(e.style_phases[0] for e in events)
    vei = Counter(int(e.vei or 0) for e in events)
    summary = {
        "algorithm_version": VOLCANISM_ALGORITHM_VERSION,
        "world_profile": prm.world_profile,
        "event_window_years": prm.event_window_years,
        "provinces": len(provinces),
        "provinces_by_source": dict(Counter(p.source_type for p in provinces)),
        "systems": len(systems),
        "systems_in_window": len(in_window),
        "systems_by_status": dict(Counter(s.magmatic_status for s in systems)),
        "systems_by_edifice": dict(Counter(s.edifice_type for s in systems)),
        "systems_by_environment": dict(Counter(s.environment for s in systems)),
        "activity_phase": dict(Counter(s.activity_phase for s in systems if s.in_window)),
        "lava_lakes": dict(
            Counter(s.lava_lake_state for s in systems if s.lava_lake_state != "none")
        ),
        "events": len(events),
        "explicit_events": len(explicit),
        "explicit_limit": limit,
        "events_by_style": dict(styles),
        "events_by_vei": {str(k): v for k, v in sorted(vei.items())},
        "erupted_dre_km3": round(sum(e.products["volume_dre_m3"] for e in events) / 1e9, 4),
        "budgets": {k: float(v) for k, v in budgets.items()},
        "reservoir_max_relative_residual": max(
            (
                abs(s.reservoir.budget_residual_kg())
                / max(s.reservoir.supplied_kg + s.reservoir.initial_kg, 1.0)
                for s in systems
                if s.reservoir
            ),
            default=0.0,
        ),
        "hydrothermal_fields": dict(Counter(f["kind"] for f in fields)),
        "subglacial_events": len(subglacial),
        "subglacial_ice_source": ice_quality,
        "terrain": terrain_summary,
        "emergent_islands": len(emergences),
        "climate_pulses": len(pulses),
        "mud_volcano_fields": len(mud),
        "quality": {
            "lava": "approximated_cellular_d6",
            "tephra": (
                "hex_fv_atmosphere_5_4_wind_at"
                if getattr(project, "atmosphere", None) is not None
                and project.atmosphere.grid_fields
                else "hex_fv_thermal_wind_aloft"
                if getattr(project, "earth", None) is not None
                else "hex_fv_synthetic_upper_wind"
            ),
            "pdc": "energy_cone_hazard+weighted_routing",
            "lahar": "receiver_routing_bulked",
            "terrain_feedback": terrain_quality,
        },
    }
    if summary["reservoir_max_relative_residual"] > 1e-9:
        warnings.append("v5.1: reservoir mass budget residual above 1e-9")
    if not systems:
        warnings.append(
            "v5.1: no volcanic sources in this world (no arcs, hotspots, ridges, "
            "rifts in the window margin) - an honest empty result"
        )
    for s in systems:
        s.hex_flat = s.hex[1] * lay.width + s.hex[0] if s.hex is not None else None
    sys_dicts = []
    for s in systems:
        d = s.to_dict()
        d["hex_flat"] = s.hex_flat
        d["erupted_dre_m3"] = float(erupted[s.volcano_id])
        sys_dicts.append(d)
    rep(1.0)
    return {
        "layers": layers,
        "systems": sys_dicts,
        "events": [e.to_dict() for e in explicit],
        "provinces": [asdict(p) for p in provinces],
        "aggregates": agg,
        "summary": summary,
        "extras": {
            "hydrothermal_fields": fields,
            "subglacial": subglacial,
            "climate_pulses": pulses,
            "emergent_islands": emergences,
            "mud_vents": mud,
        },
        "warnings": warnings,
        "timings": timings,
    }


def run_v51(config, geometry, project, old_keys: dict, old_cache: dict, report):
    keys = volcanism_stage_keys(config, upstream_keys(project))
    if old_keys.get("v51_volcanism") == keys["v51_volcanism"] and "v51_volcanism" in old_cache:
        data = old_cache["v51_volcanism"]
        return data, {"v51_volcanism": data}, ["v51_volcanism"]
    report("v51_volcanism", 0.9)
    from .progress import sub_report

    t0 = time.perf_counter()
    res = run_volcanism(
        config, project, report=lambda f: report(sub_report("v51_volcanism", f), 0.9)
    )
    res["timings"]["v51_total"] = round(time.perf_counter() - t0, 3)
    summary = dict(res["summary"], extras=res["extras"])
    data = VolcanismData(
        layers=res["layers"],
        systems=res["systems"],
        events=res["events"],
        provinces=res["provinces"],
        aggregates=res["aggregates"],
        summary=summary,
        provenance={
            "algorithm_version": VOLCANISM_ALGORITHM_VERSION,
            "params": asdict(config.volcanism),
            "upstream": upstream_keys(project),
        },
        stage_keys=keys,
        warnings=res["warnings"],
        timings=res["timings"],
    )
    return data, {"v51_volcanism": data}, []
