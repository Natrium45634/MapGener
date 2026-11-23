"""Coupling of the v5.4 normals into the world climate (inside the v4 climate stage).

Order (TZ 5.4 sec. 11.2)::

    v5.3 coupler (ocean, ice, land T)  ->  normals (circulation, moisture, ITCZ)
      -> optional coupled iteration: the ocean re-run with the normals' monthly surface
         winds (restart from its state), the normals rebuilt with the new SST
      -> stop on max_rel_change of SST / wind / P or the iteration limit (logged)
      -> publish: monthly rain and temperature on the raster for the hex stages

With ``precipitation_coupling = full`` the seasonal rain of the world climate becomes the
season sums of the monthly normals (rain goes back to the raster conservatively); the v4
seasonal transport is then a spin-up diagnostic only. The SST stays the v5.3 ocean's: the
re-run replaces it (one SST, T10).
"""

from __future__ import annotations

import time

import numpy as np

from . import normals as nrm

SEASON_MONTHS = ((11, 0, 1), (2, 3, 4), (5, 6, 7), (8, 9, 10))


def resolve_domain(atm, world_mode: str | None) -> str:
    mode = getattr(atm, "domain_mode", "auto")
    if mode == "auto":
        return "planetary_background" if world_mode == "world" else "isolated_approximation"
    return mode


def _build(wc, params, atm, planet, earth_state, land_h, cont, cell_km, dt_ep):
    oc = earth_state.ocean
    sst_m = oc.sst_monthly if oc is not None else None
    ice_m = oc.sea_ice_monthly if oc is not None else None
    return nrm.build(
        atm,
        planet,
        wc.lat,
        wc.sea,
        land_h,
        wc.t_sea_level,
        wc.seasonal_amplitude,
        cont,
        cell_km,
        wc.grid,
        sst_monthly=sst_m,
        ice_monthly=ice_m,
        dt_equator_pole_k=dt_ep,
        lapse_c_per_km=params.lapse_rate_c_per_km,
        wind_strength=params.wind_strength,
    )


def attach_normals(wc, params, atm, planet, earth, land_h, cont, cell_km, t_zone):
    from ..earth.planet import annual_temperature_shift
    from ..stages.climate import zonal_temperature

    t0 = time.perf_counter()
    eq = np.array([0.0, 90.0])
    dt_ep = float(np.ptp(zonal_temperature(eq) + annual_temperature_shift(planet, eq)))
    es = wc.earth
    normals = _build(wc, params, atm, planet, es, land_h, cont, cell_km, dt_ep)
    coupling = []
    status = "NOT_COUPLED"
    iters = int(getattr(atm, "coupled_iterations", 1))
    if iters and es.ocean is not None and getattr(earth, "ocean", True):
        from ..earth.ocean import run_ocean

        status = "NOT_CONVERGED"
        for k in range(iters):
            u_r = normals.to_raster(normals.circ.u[:, 0])
            v_r = normals.to_raster(normals.circ.v[:, 0])
            winds4 = [
                (u_r[list(ms)].mean(axis=0), -v_r[list(ms)].mean(axis=0)) for ms in SEASON_MONTHS
            ]  # raster convention: v + south
            oc_new = run_ocean(
                planet, wc.lat, wc.sea, winds4, cell_km, earth, years=2, prev=es.ocean
            )
            sea = wc.sea
            d_sst = float(
                np.sqrt(np.nanmean((oc_new.sst_monthly - es.ocean.sst_monthly)[:, sea] ** 2))
            )
            es.ocean = oc_new
            sst_m = np.nan_to_num(oc_new.sst_monthly, nan=-1.8)
            es.sst_season = np.where(
                sea[None], np.stack([sst_m[list(m)].mean(axis=0) for m in SEASON_MONTHS]), 0.0
            )
            es.ice_season = np.where(
                sea[None],
                np.stack([oc_new.sea_ice_monthly[list(m)].mean(axis=0) for m in SEASON_MONTHS]),
                0.0,
            )
            wc.t_sea_level = np.where(sea, sst_m.mean(axis=0), wc.t_sea_level)
            es.t_sl = wc.t_sea_level
            old = normals
            normals = _build(wc, params, atm, planet, es, land_h, cont, cell_km, dt_ep)
            p_old, p_new = old.moist.precip_mm.sum(axis=0), normals.moist.precip_mm.sum(axis=0)
            d_p = float(np.abs(p_new - p_old).mean() / max(float(p_old.mean()), 1e-9))
            sp_old = np.hypot(old.circ.u[:, 0], old.circ.v[:, 0])
            sp_new = np.hypot(normals.circ.u[:, 0], normals.circ.v[:, 0])
            d_w = float(np.abs(sp_new - sp_old).mean() / max(float(sp_old.mean()), 1e-9))
            coupling.append(
                {
                    "iteration": k + 1,
                    "rms_d_sst_k": round(d_sst, 4),
                    "mean_rel_change_p": round(d_p, 4),
                    "mean_rel_change_wind": round(d_w, 4),
                }
            )
            if d_sst < 0.1 and d_p < 0.02 and d_w < 0.02:
                status = "CONVERGED"
                break
    p_m = normals.to_raster(normals.precip_monthly(), conservative=True)  # mm/month
    wc.monthly_precipitation = p_m
    wc.monthly_temperature = normals.to_raster(normals.circ.t_sl)
    coupling_mode = getattr(atm, "precipitation_coupling", "full")
    if coupling_mode == "full":
        wc.seasonal_precipitation = np.stack(
            [4.0 * p_m[list(ms)].sum(axis=0) for ms in SEASON_MONTHS]
        )
        wc.precipitation = p_m.sum(axis=0)
    wc.atmosphere = normals
    sea = wc.sea
    pa = p_m.sum(axis=0)
    wc.balance["atmosphere"] = {
        "status": "ok",
        "precipitation_coupling": coupling_mode,
        "ocean_coupling": {"status": status, "iterations": coupling},
        "land_mean_mm_yr": float(pa[~sea].mean()) if (~sea).any() else None,
        "ocean_mean_mm_yr": float(pa[sea].mean()) if sea.any() else None,
        "seconds": round(time.perf_counter() - t0, 2),
        **{k: v for k, v in normals.diagnostics.items() if k != "budget"},
        "budget_relative_residual": normals.diagnostics["budget"]["annual_relative_residual"],
    }
    return normals
