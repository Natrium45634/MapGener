"""SST baseline + ablations on a fixed domain (TZ 5.4.1 sec. 4). Usage: sst_diag.py SEED W H KM OUT.json"""
import sys, json, hashlib, platform
import numpy as np, scipy
from hexworld import Config, generate
from hexworld.config import AtmosphereParams
from hexworld.stages.climate import world_climate, resolve_latitude
from hexworld.earth import ocean as oc_mod
from hexworld.earth.planet import Planet
seed, W, H, KM, out = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), float(sys.argv[4]), sys.argv[5]
cfg = Config(seed=seed, width=W, height=H, hex_spacing_km=KM)
p = generate(cfg.with_(atmosphere=AtmosphereParams(enabled=False)), through="v3")
hist = p.hist
lat = resolve_latitude(cfg.climate, p.geometry.simulation_bounds)
wc = world_climate(cfg.climate, hist.world["grid"], hist.world["elevation_m"], hist.sea_level_m, lat, earth=cfg.earth)
es = wc.earth
winds = [(u, v) for _, u, v in es.winds]
planet = Planet.from_params(cfg.earth)
LAT = wc.lat
# Earth reference (approximate zonal means of the observed SST climatology; literature
# values to the nearest ~1 K, NOT a dataset extraction): annual mean and half range
REF = {(0, 15): (27.3, 1.0), (15, 30): (24.5, 2.0), (30, 45): (18.5, 3.5), (45, 60): (9.5, 3.5), (60, 75): (3.0, 2.0), (75, 90): (-1.3, 0.5)}
def table(res):
    sst = res.sst_monthly.astype(float); ice = res.sea_ice_monthly.astype(float); ocean = res.ocean
    rows = []
    for hemi, sgn in (("N", 1), ("S", -1)):
        for (lo, hi) in REF:
            b = ocean & (sgn * LAT >= lo) & (sgn * LAT < hi)
            if b.sum() < 3: continue
            s = sst[:, b]; free = ice[:, b] < 0.15
            ann = np.nanmean(s, axis=0); amp = 0.5 * (np.nanmax(s, axis=0) - np.nanmin(s, axis=0))
            rows.append({"band": f"{hemi}{lo}-{hi}", "cells": int(b.sum()), "mean": round(float(np.nanmean(ann)), 2),
                         "p05": round(float(np.nanpercentile(ann, 5)), 2), "p95": round(float(np.nanpercentile(ann, 95)), 2),
                         "half_range": round(float(np.nanmean(amp)), 2), "ice_share": round(float((ice[:, b] > 0.15).mean()), 3),
                         "ice_free_mean": round(float(np.nanmean(np.where(free, s, np.nan))), 2) if free.any() else None,
                         "ref_mean": REF[(lo, hi)][0], "ref_half_range": REF[(lo, hi)][1]})
    return rows
def run(**patch):
    saved = {}
    for k, v in patch.items():
        saved[k] = getattr(oc_mod, k); setattr(oc_mod, k, v)
    try:
        r = oc_mod.run_ocean(planet, LAT, wc.sea, winds, float(wc.grid["cell_km"]), cfg.earth, years=4)
    finally:
        for k, v in saved.items(): setattr(oc_mod, k, v)
    return r
res = {"env": {"python": platform.python_version(), "numpy": np.__version__, "scipy": scipy.__version__, "platform": platform.platform()},
       "seed": seed, "window": [W, H, KM], "raster": list(wc.sea.shape), "cell_km": float(wc.grid["cell_km"])}
A0 = run(); res["A0_baseline"] = table(A0); res["A0_diag"] = {k: A0.diagnostics[k] for k in ("heat_relative_residual", "zonal_calibration_max_k", "max_current_ms")}
res["A0_sst_hash"] = hashlib.sha256(np.nan_to_num(A0.sst_monthly).tobytes()).hexdigest()[:16]
res["A3_no_restoring"] = table(run(GAMMA=0.0))
res["A5_ice_off"] = table(oc_mod.run_ocean(planet, LAT, wc.sea, winds, float(wc.grid["cell_km"]), cfg.earth.__class__(**{**cfg.earth.__dict__, "sea_ice": False}), years=4))
calm = [(0 * u, 0 * v) for u, v in winds]
res["A3b_no_wind_transport"] = table(oc_mod.run_ocean(planet, LAT, wc.sea, calm, float(wc.grid["cell_km"]), cfg.earth, years=4))
res["A8_years_8"] = table(oc_mod.run_ocean(planet, LAT, wc.sea, winds, float(wc.grid["cell_km"]), cfg.earth, years=8))
json.dump(res, open(out, "w"), indent=1)
for k in ("A0_baseline", "A3_no_restoring", "A3b_no_wind_transport", "A5_ice_off"):
    print(k, [(r["band"], r["mean"], r["half_range"], r["ref_mean"], r["ref_half_range"]) for r in res[k]])
