"""Explicit ocean <-> sea ice <-> climate coupler (no hidden pipeline recursion).

Iteration ``k`` (at most ``earth.coupler_max_iter``)::

    land temperature, seasonal range, winds   <- previous ice cover and SST anomaly
    ocean currents, SST, sea ice              <- these winds   (restart from k-1)
    ice_w = (1-a) ice_w + a ice_new,  anom = (1-a) anom + a anom_new     (a = relaxation)

Converged when the RMS change of the annual SST over the ocean < 0.05 K and of the winter
ice concentration < 0.01; otherwise ``NOT_CONVERGED`` is reported (never silent).
Categories and IDs are never relaxed, only temperature and concentration.

Land feels the ocean through the SST *anomaly* of the water upwind (sampled 300 km
upstream of the annual wind, fading inland with the continentality): warm currents warm
their coasts, upwelling coasts are cooler. The base maritime warming of v4 stays, scaled
by the open-water share of the coast.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy import ndimage

from .ocean import run_ocean, sst_target
from .winds import thermal_wind

UPWIND_KM = 300.0
ANOMALY_LAND_WEIGHT = 0.7
SEASON_MONTHS = ((11, 0, 1), (2, 3, 4), (5, 6, 7), (8, 9, 10))


@dataclass
class EarthState:
    t_sl: np.ndarray
    seasonal: np.ndarray
    winds: list
    sst_season: np.ndarray  # (4, ny, nx)
    ice_season: np.ndarray  # (4, ny, nx) concentration
    ocean: object
    aloft: object | None
    sea: np.ndarray
    report: dict = field(default_factory=dict)


def _fill_from_sea(values, sea, sigma):
    """Normalised Gaussian spread of a sea field onto the land (0 far from any sea)."""
    w = ndimage.gaussian_filter(sea.astype(np.float64), sigma, mode="nearest")
    v = ndimage.gaussian_filter(np.where(sea, values, 0.0), sigma, mode="nearest")
    return np.where(w > 1e-3, v / np.maximum(w, 1e-3), 0.0)


def _upwind(field_, u, v, cell_km):
    sp = np.hypot(u, v)
    k = UPWIND_KM / cell_km / np.maximum(sp, 1e-6)
    ny, nx = field_.shape
    jj, ii = np.mgrid[0:ny, 0:nx].astype(np.float64)
    # v is + south (raster rows): the upwind point is at -U
    return ndimage.map_coordinates(field_, [jj - v * k, ii - u * k], order=1, mode="nearest")


def couple(params, earth, planet, lat, phi, sea, cont, t_zone, land_h, cell_km, sea_ice0):
    from ..stages.climate import (
        COAST_SEASONALITY,
        MARITIME_WARMING,
        season_temperature,
        seasonal_winds,
    )
    from .planet import seasonal_amplitude_ratio

    ratio = seasonal_amplitude_ratio(planet, lat)
    sigma = max(1.0, 250.0 / cell_km)
    ice_w = sea_ice0.copy()  # first guess: the v4 heuristic
    anom = np.zeros_like(t_zone)
    prev = None
    a = float(earth.coupler_relaxation)
    history = []
    status = "NOT_CONVERGED"
    years_first = {"fast": 3, "balanced": 4, "high": 6}.get(earth.quality, 4)
    for k in range(int(earth.coupler_max_iter)):
        t_sl = (
            t_zone
            - cont * np.clip(phi - 35.0, 0.0, None) * 0.12
            + cont * 0.8 * np.exp(-((phi / 20.0) ** 2))
            + (1.0 - cont) * (1.0 - ice_w) * MARITIME_WARMING * np.clip(phi - 40.0, 0.0, 30.0)
            + (1.0 - cont) * ANOMALY_LAND_WEIGHT * anom
        )
        cont_season = np.maximum(cont, ice_w)
        seasonal = (
            params.seasonality
            * (1.0 + 0.32 * phi)
            * (COAST_SEASONALITY + (1.0 - COAST_SEASONALITY) * cont_season)
            * ratio
        )
        winds = seasonal_winds(params, lat, cont, seasonal, land_h, cell_km,
                               declination_deg=abs(planet.obliquity_deg))  # fmt: skip
        if not earth.ocean:
            oc = None
            break
        oc = run_ocean(planet, lat, sea, [(u, v) for _, u, v in winds], cell_km, earth,
                       years=years_first if prev is None else 2, prev=prev)  # fmt: skip
        sst_ann = oc.sst_monthly.mean(axis=0)
        ice_new_sea = oc.sea_ice_monthly.max(axis=0)
        anom_sea = np.where(sea, np.nan_to_num(sst_ann) - sst_target(lat), 0.0)
        anom_fill = _fill_from_sea(anom_sea, sea, sigma)
        u_ann = np.mean([w[1] for w in winds], axis=0)
        v_ann = np.mean([w[2] for w in winds], axis=0)
        anom_new = np.where(sea, anom_sea, _upwind(anom_fill, u_ann, v_ann, cell_km))
        ice_land = _fill_from_sea(np.where(sea, ice_new_sea, 0.0), sea, sigma)
        ice_new = np.where(sea, ice_new_sea, np.maximum(ice_land, 0.0))
        d_sst = float(np.sqrt(np.mean((anom_new - anom)[sea] ** 2))) if sea.any() else 0.0
        d_ice = float(np.sqrt(np.mean((ice_new - ice_w)[sea] ** 2))) if sea.any() else 0.0
        history.append(
            {
                "iteration": k + 1,
                "rms_d_sst_k": round(d_sst, 4),
                "rms_d_ice": round(d_ice, 5),
                "heat_relative_residual": oc.diagnostics["heat_relative_residual"],
            }
        )
        anom = (1.0 - a) * anom + a * anom_new if k else anom_new
        ice_w = (1.0 - a) * ice_w + a * ice_new if k else ice_new
        prev = oc
        if k and d_sst < 0.05 and d_ice < 0.01:
            status = "CONVERGED"
            break
    if oc is None:
        status = "OCEAN_OFF"
        sst_season = np.repeat(np.maximum(t_zone, -1.8)[None], 4, axis=0)
        ice_season = np.zeros((4,) + t_zone.shape)
    else:
        sst_m = np.nan_to_num(oc.sst_monthly, nan=-1.8)
        sst_season = np.stack([sst_m[list(m)].mean(axis=0) for m in SEASON_MONTHS])
        sst_season = np.where(sea[None], sst_season, 0.0)
        ice_season = np.stack([oc.sea_ice_monthly[list(m)].mean(axis=0) for m in SEASON_MONTHS])
        ice_season = np.where(sea[None], ice_season, 0.0)
        if sea.any():
            t_sl = np.where(sea, np.nan_to_num(oc.sst_monthly.mean(axis=0)), t_sl)
    aloft = None
    if earth.upper_winds:
        t_seasons = [
            np.where(sea, sst_season[s], season_temperature(t_sl, seasonal, lat, s))
            for s in range(4)
        ]
        aloft = thermal_wind(t_seasons, [w[1] for w in winds], [w[2] for w in winds], lat,
                             cell_km, planet)  # fmt: skip
    report = {
        "status": status,
        "iterations": history,
        "relaxation": a,
        "ocean": oc.diagnostics if oc is not None else None,
        "planet": {
            "obliquity_deg": planet.obliquity_deg,
            "day_length_h": planet.day_length_h,
            "year_length_days": planet.year_length_days,
            "solar_constant_wm2": planet.solar_constant_wm2,
        },  # fmt: skip
    }
    return EarthState(t_sl, seasonal, winds, sst_season, ice_season, oc, aloft, sea, report)
