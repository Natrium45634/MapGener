"""Benchmark and resolution comparison of the TZ 5.5 biosphere (B80, B81).

    python tools/bench_v55.py OUT_DIR [--sizes 360x240,500x300,1000x800] [--runs 3]

For every size: generate the world once without the biosphere (time recorded), save it, then
run the biosphere stage ``runs`` times, each in a fresh process (cold: first run; warm: the
others), recording wall times per sub-stage and the peak RSS added by the stage. The
resolution comparison uses the same seed and the same world width in km, and compares the
area-weighted quantities on the common footprint. Results go to ``OUT_DIR/bench_v55.json``.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import resource
import subprocess
import sys
import time
from pathlib import Path

WORLD_KM = 36000.0  # the same physical width for every grid


def _gen(w, h, out):
    from hexworld import Config, generate
    from hexworld.config import BiosphereParams
    from hexworld.io import save_project

    cfg = Config(seed="Bench55", width=w, height=h, hex_spacing_km=WORLD_KM / w,
                 biosphere=BiosphereParams(enabled=False))  # fmt: skip
    t = time.perf_counter()
    p = generate(cfg)
    gen_s = time.perf_counter() - t
    save_project(p, out)
    return gen_s


def _stage(path):
    """One run in this process: load, measure the stage alone."""
    from dataclasses import replace

    from hexworld.biosphere.stage import run_biosphere
    from hexworld.config import BiosphereParams
    from hexworld.io import load_project

    p = load_project(path)
    cfg = replace(p.config, biosphere=BiosphereParams())
    rss0 = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    import tracemalloc

    tracemalloc.start()
    t = time.perf_counter()
    b = run_biosphere(cfg, p)
    wall = time.perf_counter() - t
    _cur, peak_py = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    rss1 = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    st = b.state
    land = st.frac["land_ground"] > 0
    import numpy as np

    area = st.area_km2
    out = {"wall_s": wall, "timings": b.timings, "peak_traced_mib": peak_py / 2**20,
           "maxrss_before_mib": rss0 / 1024, "maxrss_after_mib": rss1 / 1024,
           "hexes": st.n, "ledger_ok": st.summary["ledger_ok"],
           "npp_land_total_tc_yr": float(np.nansum(st.layers["npp_gc_m2_yr"][land]) * area),
           "land_area_km2": float(st.frac["land_ground"].sum() * area),
           "inland_water_km2": float(st.frac["inland_water"].sum() * area),
           "marine_area_km2": float(st.frac["marine_water"].sum() * area),
           "richness_median_land": float(np.median(st.layers["total_richness"][land])),
           "plant_c_total_pgc": float(np.nansum(st.layers["plant_biomass_kgc_m2"]) * area * 1e6
                                      / 1e12),
           "taxa_world": st.summary["taxa_world"], "bioregions": st.summary["bioregions"]}
    print(json.dumps(out))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--sizes", default="360x240,500x300,1000x800")
    ap.add_argument("--runs", type=int, default=3)
    ap.add_argument("--stage-only", help=argparse.SUPPRESS)
    a = ap.parse_args()
    if a.stage_only:
        _stage(a.stage_only)
        return
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    res = {"machine": {"cpu": platform.processor() or platform.machine(),
                       "cpus": os.cpu_count(), "os": platform.platform(),
                       "python": platform.python_version()},
           "world_km": WORLD_KM, "sizes": {}}  # fmt: skip
    try:
        import numpy
        import scipy

        res["machine"].update({"numpy": numpy.__version__, "scipy": scipy.__version__})
    except ImportError:
        pass
    for s in a.sizes.split(","):
        w, h = (int(x) for x in s.split("x"))
        path = out / f"world_{s}"
        if not (path / "manifest.json").exists():
            gen_s = _gen(w, h, path)
        else:
            gen_s = None
        runs = []
        for k in range(a.runs):
            r = subprocess.run([sys.executable, __file__, str(out), "--stage-only", str(path)],
                               capture_output=True, text=True, check=True)  # fmt: skip
            runs.append(json.loads(r.stdout.strip().splitlines()[-1]))
            runs[-1]["kind"] = "cold" if k == 0 else "warm"
        res["sizes"][s] = {"generation_without_bio_s": gen_s, "runs": runs}
        (out / "bench_v55.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
        print(s, [round(r["wall_s"], 1) for r in runs], flush=True)


if __name__ == "__main__":
    main()
