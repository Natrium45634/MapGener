"""v4 stage ``climate``: annual-mean climate from latitude, relief, ocean and moisture transport.

A parameterised model, **not** a general circulation model. Two scales:

* **world raster** (the v3 history grid over the whole model area, ~60 km cells): latitude
  zones, prevailing winds, sea-surface temperature, continentality, and a steady-state
  moisture-transport solution - so the climate of the window feels oceans, mountains and
  deserts that lie outside it;
* **hex grid** of the window: the world fields are sampled, then corrected with the hex's
  own elevation (lapse rate) and windward / leeward slope (orographic enhancement).

Moisture model (per world cell, iterated to a steady state)::

    q <- advect(q, wind)                        semi-Lagrangian, one cell per step
    q += E_ocean = r_o * max(0, q_sat(SST) - q)  oceans (and lakes) are sources
    q += E_land  = r_l * q_sat(T) * wetness       recycling over land
    P  = q * (k_zone(lat) + k_conv(T) + k_oro * max(0, U . grad h))
         + max(0, q - q_sat(T_air(h)))           condensation when lifted air saturates
    q -= P

``q_sat`` follows Clausius-Clapeyron (7 %/K). Rain shadows, wet windward slopes and the
drying of continental interiors come out of the transport; nothing is painted. The final
precipitation is scaled by one constant so the mean over oceans matches the profile target
(Earth: ~1100 mm/yr over the ocean). Evapotranspiration uses the Budyko curve (Fu's form),
so actual evaporation never exceeds precipitation nor potential evaporation.

Units: temperature degC, wind m/s, water mm/yr, elevation m above sea level.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy import ndimage

#: Approximate zonal annual-mean sea-level air temperature of the Earth, degC
#: (northern hemisphere, smoothed; e.g. Peixoto & Oort 1992 style climatology - values from
#: memory, accurate to about +-3 degC).
_ZONAL_LAT = np.array([0, 10, 20, 30, 40, 50, 60, 70, 80, 90], dtype=np.float64)
_ZONAL_T = np.array([26.5, 26.0, 24.5, 19.5, 13.5, 6.0, -1.0, -11.0, -19.0, -24.0])
KM_PER_DEG = 111.2
SECONDS_PER_YEAR = 365.25 * 86400.0
PROFILE_TARGET_OCEAN_MM = {"earthlike": 1100.0, "dry": 760.0, "wet": 1550.0}
PROFILE_T_OFFSET = {"earthlike": 0.0, "dry": 1.5, "wet": -0.5}
#: Rates of the moisture model (per transport step). Calibrated against Earth references in
#: tools/realism_report.py: land precipitation, aridity shares, runoff ratio.
#: Strength of the dry subtropical high-pressure belt (0.55 = rain-out almost off at 24 deg).
SUBSIDENCE = 0.53
RATES = {
    "zone": 0.032,  # large-scale rain-out
    "convective": 0.010,  # warm-air convection
    "orographic": 0.028,  # per 1 cm/s of forced ascent
    "orographic_cap": 4.0,  # cm/s
    "condensation": 0.15,  # share of the supersaturation that rains out per step
    "ocean": 0.22,  # relaxation of q to saturation over the sea
    "recycling": 0.0015,  # land evapotranspiration source
    "hex_orographic": 0.6,  # hex-scale windward / leeward correction
}


@dataclass
class Latitude:
    """Conditional latitudes of the flat world: lat = lat_center - (y - y_center) / km_per_deg."""

    center_deg: float
    y_center_km: float
    km_per_deg: float
    compressed: bool

    def at(self, y_km):
        lat = self.center_deg - (np.asarray(y_km, dtype=np.float64) - self.y_center_km) / (
            self.km_per_deg
        )
        return np.clip(lat, -90.0, 90.0)

    def to_dict(self) -> dict:
        return {
            "center_deg": self.center_deg,
            "y_center_km": self.y_center_km,
            "km_per_degree": self.km_per_deg,
            "compressed": self.compressed,
            "north": "-y (map up)",
        }


def resolve_latitude(params, sim_bounds) -> Latitude:
    """Real scale (111.2 km per degree) unless the area would cross a pole."""
    height = sim_bounds.height
    yc = 0.5 * (sim_bounds.ymin + sim_bounds.ymax)
    center = float(params.latitude_center_deg)
    if params.km_per_degree and params.km_per_degree > 0:
        return Latitude(center, yc, float(params.km_per_degree), False)
    span = height / KM_PER_DEG
    half_room = 90.0 - abs(center) - 5.0
    if span / 2.0 <= half_room:
        return Latitude(center, yc, KM_PER_DEG, False)
    # the area is taller than the room to the pole: compress the latitude scale
    km_per_deg = height / (2.0 * max(half_room, 10.0))
    return Latitude(center, yc, km_per_deg, True)


def zonal_temperature(lat):
    return np.interp(np.abs(lat), _ZONAL_LAT, _ZONAL_T)


def prevailing_wind(lat, strength: float):
    """Zonal wind belts: trades (easterlies) 0-30, westerlies 30-60, polar easterlies 60-90.

    Map axes: +x east, +y south. Returns (u, v) in m/s.
    """
    phi = np.abs(lat)
    band = np.sin(np.pi * phi / 30.0)  # +: easterly belts, -: westerly belt
    u = -6.5 * strength * band
    equatorward = 0.35 * 6.5 * strength * band  # trades and polar winds blow to the equator
    v = np.where(lat >= 0, equatorward, -equatorward)  # northern hemisphere: equator is +y
    calm = 1.0 - 0.55 * np.exp(-((phi / 5.0) ** 2))  # doldrums
    return u * calm, v * calm


def circulation_factor(lat):
    """Large-scale ascent: ITCZ wet, subtropical highs dry, mid-latitude storm tracks wet."""
    phi = np.abs(lat)
    return (
        0.55
        + 1.25 * np.exp(-((phi / 9.0) ** 2))
        - SUBSIDENCE * np.exp(-(((phi - 24.0) / 8.0) ** 2))
        + 0.55 * np.exp(-(((phi - 50.0) / 11.0) ** 2))
        - 0.25 * np.exp(-(((phi - 85.0) / 8.0) ** 2))
    )


def q_sat(t_c):
    """Saturation moisture relative to 15 degC (Clausius-Clapeyron, ~7 % per K)."""
    return np.exp(0.068 * (np.asarray(t_c) - 15.0))


def budyko_aet(p, pet, w: float = 2.6):
    """Actual evapotranspiration by Fu's form of the Budyko curve (mm/yr)."""
    p = np.maximum(np.asarray(p, dtype=np.float64), 0.0)
    pet = np.maximum(np.asarray(pet, dtype=np.float64), 0.0)
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.where(p > 0, pet / np.maximum(p, 1e-9), 0.0)
        aet = p * (1.0 + ratio - np.power(1.0 + np.power(ratio, w), 1.0 / w))
    return np.clip(np.where(p > 0, aet, 0.0), 0.0, np.minimum(p, pet))


def potential_et(t_c):
    """Annual potential evapotranspiration, mm/yr (a Thornthwaite-like curve of mean T).

    About 1750 mm at 27 degC, 1280 mm at 20 degC, 700 mm at 10 degC, 250 mm at 0 degC,
    0 below -10 degC (typical annual Penman-Monteith reference values, approximate).
    """
    return 1750.0 * np.clip((np.asarray(t_c) + 10.0) / 37.0, 0.0, 1.25) ** 1.5


@dataclass
class WorldClimate:
    grid: dict
    lat: np.ndarray
    sea: np.ndarray  # bool
    t_sea_level: np.ndarray  # degC, without elevation
    continentality: np.ndarray  # 0 coast .. 1 deep interior
    wind_u: np.ndarray
    wind_v: np.ndarray
    precipitation: np.ndarray  # mm/yr
    moisture: np.ndarray  # relative q
    scale_mm_per_unit: float
    seasonal_amplitude: np.ndarray  # degC, half of the summer-winter range
    iterations: int
    balance: dict


def world_climate(
    params, grid: dict, elevation_m: np.ndarray, sea_level_m: float, latitude: Latitude
) -> WorldClimate:
    ny, nx = elevation_m.shape
    cell_km = float(grid["cell_km"])
    xs = grid["x0_km"] + np.arange(nx) * cell_km
    ys = grid["y0_km"] + np.arange(ny) * cell_km
    gx, gy = np.meshgrid(xs, ys)
    lat = latitude.at(gy)
    height = np.asarray(elevation_m, dtype=np.float64) - sea_level_m
    sea = height <= 0.0
    land_h = np.maximum(height, 0.0)
    profile = params.profile

    dist_km = (
        ndimage.distance_transform_edt(~sea) * cell_km if sea.any() else np.full(sea.shape, 4000.0)
    )
    cont = 1.0 - np.exp(-dist_km / 900.0)
    phi = np.abs(lat)
    t_zone = (
        zonal_temperature(lat) + PROFILE_T_OFFSET.get(profile, 0.0) + params.temperature_offset_c
    )
    # continental interiors: colder annual means at high latitude, a little warmer in the tropics
    t_sl = (
        t_zone
        - cont * np.clip(phi - 35.0, 0.0, None) * 0.12
        + cont * 0.8 * np.exp(-((phi / 20.0) ** 2))
    )
    sst = np.maximum(t_zone, -1.8)
    t_sl = np.where(sea, sst, t_sl)
    seasonal = params.seasonality * (1.0 + 0.32 * phi) * (0.25 + 0.75 * cont)

    u, v = prevailing_wind(lat, params.wind_strength)
    # mountains steer and slow the wind a little; smooth so the field varies gently
    blocked = np.exp(-land_h / 4000.0)
    u = ndimage.gaussian_filter(u * blocked, 1.5)
    v = ndimage.gaussian_filter(v * blocked, 1.5)

    # -- steady-state moisture transport ------------------------------------------------
    speed = np.hypot(u, v)
    smax = max(float(speed.max()), 1e-6)
    dt_s = cell_km * 1000.0 / smax  # one cell per step for the fastest wind
    cx = u * dt_s / (cell_km * 1000.0)  # displacement in cells per step
    cy = v * dt_s / (cell_km * 1000.0)
    gyh, gxh = np.gradient(land_h, cell_km * 1000.0)
    w_up = u * gxh + v * gyh  # m/s of forced ascent (+ windward, - leeward)
    t_air = t_sl - params.lapse_rate_c_per_km * land_h / 1000.0
    qs_air = q_sat(t_air)
    qs_sea = q_sat(sst)
    k_zone = RATES["zone"] * circulation_factor(lat) * params.moisture_transport_damping
    # deep convection needs warm air and large-scale ascent (suppressed under the highs)
    k_conv = (
        RATES["convective"]
        * np.clip((t_air - 20.0) / 8.0, 0.0, 1.0)
        * np.clip(circulation_factor(lat) / 0.6, 0.0, 1.5)
    )
    k_oro = RATES["orographic"] * np.clip(w_up / 0.01, 0.0, RATES["orographic_cap"])
    k_total = np.clip(k_zone + k_conv + k_oro, 0.0, 0.5)
    r_ocean = RATES["ocean"]
    rows, cols = np.mgrid[0:ny, 0:nx].astype(np.float64)
    back = np.vstack([(rows - cy).ravel(), (cols - cx).ravel()])
    q = np.where(sea, qs_sea, 0.3 * qs_air)
    wet = np.full(q.shape, 0.5)
    precip = np.zeros_like(q)
    n_iter = int(min(400, max(60, 3 * max(nx, ny) // 2)))
    for _ in range(n_iter):
        q = ndimage.map_coordinates(q, back, order=1, mode="nearest").reshape(q.shape)
        q = q + np.where(sea, r_ocean * np.maximum(qs_sea - q, 0.0), 0.0)
        q = q + np.where(sea, 0.0, RATES["recycling"] * qs_air * wet)
        excess = np.maximum(q - qs_air, 0.0)
        precip = q * k_total + RATES["condensation"] * excess
        precip = np.minimum(precip, q)
        q = q - precip
        wet = np.clip(precip / np.maximum(0.03 * qs_air, 1e-6), 0.05, 1.0)
    ocean_mean = float(precip[sea].mean()) if sea.any() else float(precip.mean())
    target = PROFILE_TARGET_OCEAN_MM.get(profile, 1100.0) * params.precipitation_scale
    scale = target / max(ocean_mean, 1e-12)
    precip_mm = precip * scale
    return WorldClimate(
        grid=grid,
        lat=lat,
        sea=sea,
        t_sea_level=t_sl,
        continentality=cont,
        wind_u=u,
        wind_v=v,
        precipitation=precip_mm,
        moisture=q,
        scale_mm_per_unit=scale,
        seasonal_amplitude=seasonal,
        iterations=n_iter,
        balance={
            "ocean_mean_mm_yr": float(precip_mm[sea].mean()) if sea.any() else None,
            "land_mean_mm_yr": float(precip_mm[~sea].mean()) if (~sea).any() else None,
            "steps": n_iter,
            "step_seconds": dt_s,
        },
    )


def _bilinear(field_, grid, x, y):
    fx = (np.asarray(x) - grid["x0_km"]) / grid["cell_km"]
    fy = (np.asarray(y) - grid["y0_km"]) / grid["cell_km"]
    coords = np.vstack([np.ravel(fy), np.ravel(fx)])
    return ndimage.map_coordinates(field_, coords, order=1, mode="nearest").reshape(np.shape(x))


def hex_climate(
    params,
    world: WorldClimate,
    layout,
    elevation_m: np.ndarray,
    sea_level_m: float,
    lake_mask: np.ndarray | None = None,
) -> dict:
    """Climate layers on the hex grid of the window for the given surface."""
    from ..views import hex_gradient  # local import: views imports project, not stages

    hx, hy = layout.centers()
    g = world.grid
    height = np.asarray(elevation_m, dtype=np.float64) - sea_level_m
    sea = height <= 0.0
    land_h = np.maximum(height, 0.0)
    lat = _bilinear(world.lat, g, hx, hy)
    t_sl = _bilinear(world.t_sea_level, g, hx, hy)
    u = _bilinear(world.wind_u, g, hx, hy)
    v = _bilinear(world.wind_v, g, hx, hy)
    amp = _bilinear(world.seasonal_amplitude, g, hx, hy)
    p_world = np.maximum(_bilinear(world.precipitation, g, hx, hy), 0.0)

    t = t_sl - params.lapse_rate_c_per_km * land_h / 1000.0
    # orographic correction at hex scale: the world raster cannot see slopes finer than it
    gxh, gyh = hex_gradient(land_h, layout)
    w_hex = (u * gxh + v * gyh) / 1000.0  # m/s (gradient is per km)
    # only the slope scale the world raster misses (a band of ~1-4 hexes), not hex noise
    anomaly = ndimage.gaussian_filter(w_hex, 1.0) - ndimage.gaussian_filter(w_hex, 3.0)
    p = p_world * np.clip(1.0 + RATES["hex_orographic"] * np.tanh(anomaly / 0.004), 0.4, 1.6)
    p = np.where(sea, p_world, p)

    pet = potential_et(t)
    open_water = sea if lake_mask is None else (sea | lake_mask)
    pet = np.where(open_water, 1.1 * pet, pet)  # open water evaporates a little above PET
    aet = np.where(open_water, np.minimum(pet, p + 1e9), budyko_aet(p, pet))
    aet = np.where(open_water, pet, aet)
    # snow: share of the year below 0 degC under a sinusoidal annual cycle
    with np.errstate(invalid="ignore"):
        cold = np.where(
            amp > 1e-6,
            np.arccos(np.clip(t / np.maximum(amp, 1e-6), -1.0, 1.0)) / np.pi,
            (t < 0).astype(float),
        )
    snow_fraction = np.clip(cold, 0.0, 1.0)
    snow_storage = p * snow_fraction * 0.5 * snow_fraction  # peak winter storage, mm water
    summer = t + amp
    glacial = np.clip((1.5 - summer) / 6.0, 0.0, 1.0) * np.clip(p * snow_fraction / 350.0, 0.0, 1.0)
    glacial = np.where(sea, 0.0, glacial)
    infiltration_loss = np.where(sea, 0.0, params.deep_infiltration * np.maximum(p - aet, 0.0))
    runoff = np.where(sea, 0.0, np.maximum(p - aet - infiltration_loss, 0.0))
    humidity = np.where(pet > 1.0, p / np.maximum(pet, 1.0), 5.0)
    return {
        "latitude_deg": lat.astype(np.float32),
        "temperature_mean_c": t.astype(np.float32),
        "temperature_seasonal_c": amp.astype(np.float32),
        "wind_u_ms": u.astype(np.float32),
        "wind_v_ms": v.astype(np.float32),
        "precipitation_mm_yr": p.astype(np.float32),
        "potential_evapotranspiration_mm_yr": pet.astype(np.float32),
        "actual_evapotranspiration_mm_yr": np.where(sea, pet, aet).astype(np.float32),
        "humidity_index": np.clip(humidity, 0.0, 5.0).astype(np.float32),
        "snow_fraction": np.where(sea, 0.0, snow_fraction).astype(np.float32),
        "snow_storage_mm": np.where(sea, 0.0, snow_storage).astype(np.float32),
        "glacial_potential": glacial.astype(np.float32),
        "infiltration_loss_mm_yr": infiltration_loss.astype(np.float32),
        "runoff_mm_yr": runoff.astype(np.float32),
    }


def aridity_class(humidity):
    """UNEP aridity classes of P/PET.

    0 hyper-arid, 1 arid, 2 semi-arid, 3 dry sub-humid, 4 humid.
    """
    h = np.asarray(humidity)
    return np.digitize(h, [0.05, 0.2, 0.5, 0.65]).astype(np.uint8)


__all__ = [
    "Latitude",
    "WorldClimate",
    "aridity_class",
    "budyko_aet",
    "hex_climate",
    "potential_et",
    "resolve_latitude",
    "world_climate",
    "q_sat",
    "SECONDS_PER_YEAR",
    "math",
]
