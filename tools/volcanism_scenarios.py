"""Scenario worlds for v5.1 volcanism: seeds x settings, budgets and plausibility checks.

Usage::

    python tools/volcanism_scenarios.py OUT.jsonl [first last]

Each line of ``OUT.jsonl`` is one world: its scenario, the v5.1 summary, the validation
checks and a few plausibility numbers (share of explosive eruptions, VEI distribution,
barren area, timings). ``docs/volcanism/validation.md`` is written from these lines.
"""

from __future__ import annotations

import json
import sys
import time
from dataclasses import replace

from hexworld import Config, generate
from hexworld.config import ContinentParams, HistoryParams, VolcanismParams

BASE = Config(
    seed="V0",
    width=120,
    height=90,
    hex_spacing_km=50.0,
    continents=ContinentParams(skeleton_cells=224),
    history=HistoryParams(duration_ma=16.0, time_step_ma=2.0, checkpoint_stride=2),
)
#: (name, VolcanismParams overrides, Config overrides)
SETTINGS = [
    ("default", {}, {}),
    ("frequent", {"eruption_rate_scale": 3.0}, {}),
    ("rare", {"eruption_rate_scale": 0.3}, {}),
    ("no_hotspots", {"hotspot_activity": 0.0}, {}),
    ("strong_hotspots", {"hotspot_activity": 2.5}, {}),
    ("no_rifts_ridges", {"rift_volcanism": False, "ridge_volcanism": False}, {}),
    ("short_window", {"event_window_years": 1000.0}, {}),
    ("long_window", {"event_window_years": 50000.0}, {}),
    ("fast", {"physics_quality": "fast"}, {}),
    ("icy_world", {"world_profile": "icy_world"}, {}),
    ("cold_world", {}, {"climate_offset": -8.0}),
    ("warm_world", {}, {"climate_offset": 5.0}),
]
SEEDS = ["V1", "V2", "V3"]


def scenarios():
    out = []
    for seed in SEEDS:
        for name, vp, cp in SETTINGS:
            out.append((f"{seed}-{name}", seed, vp, cp))
    return out


def run_one(name, seed, vp, cp):
    cfg = replace(BASE, seed=seed, volcanism=VolcanismParams(**vp))
    for k, v in cp.items():
        if k == "climate_offset":
            cfg = replace(cfg, climate=replace(cfg.climate, temperature_offset_c=v))
        else:
            cfg = replace(cfg, **{k: v})
    t0 = time.perf_counter()
    p = generate(cfg)
    dt = time.perf_counter() - t0
    v = p.volcanism
    if v is None:
        return {"scenario": name, "volcanism": None, "seconds": round(dt, 1)}
    s = dict(v.summary)
    s.pop("extras", None)
    styles = s.get("events_by_style", {})
    n = max(sum(styles.values()), 1)
    expl = sum(c for k, c in styles.items() if k not in (
        "hawaiian_effusive", "fissure_effusive", "submarine_effusive"))  # fmt: skip
    barren = v.layers["volcanic_barren_fraction"]
    return {
        "scenario": name,
        "seconds_total": round(dt, 1),
        "seconds_v51": v.timings.get("v51_total"),
        "valid": v.validation["ok"],
        "errors": v.validation["errors"],
        "checks": v.validation["checks"],
        "explosive_share": round(expl / n, 3),
        "barren_area_km2": round(float(barren.sum() * p.layout.hex_area_km2), 1),
        "biome_fringe_hexes": (p.biomes.summary.get("volcanic_fringe") or {}).get("hexes")
        if p.biomes is not None
        else None,
        "summary": s,
    }


def main(argv):
    out = argv[1]
    sc = scenarios()
    a = int(argv[2]) if len(argv) > 2 else 0
    b = int(argv[3]) if len(argv) > 3 else len(sc)
    with open(out, "a", encoding="utf-8") as f:
        for name, seed, vp, cp in sc[a:b]:
            try:
                row = run_one(name, seed, vp, cp)
            except Exception as exc:  # a failing scenario is a result, not a crash
                row = {"scenario": name, "exception": repr(exc)}
            f.write(json.dumps(row, default=str) + "\n")
            f.flush()
            print(name, row.get("valid"), row.get("seconds_total"), flush=True)


if __name__ == "__main__":
    main(sys.argv)
