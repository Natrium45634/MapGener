"""v4 stage ``climate``: four seasons of temperature, wind, a conservative moisture budget
and a hex water balance.

A parameterised model, **not** a general circulation model. Two scales:

* **world raster** (the v3 history grid over the whole model area, ~60-90 km cells): latitude
  zones, sea-surface temperature, continentality, the winds of each season and a
  mass-conservative moisture transport - so the climate of the window feels oceans,
  mountains and deserts that lie outside it;
* **hex grid** of the window: the world fields are sampled, then corrected with the hex's
  own elevation (lapse rate) and windward / leeward slope (orographic enhancement); each
  land hex runs a seasonal snow / soil / groundwater balance.

Seasons (DJF, MAM, JJA, SON; northern summer is JJA, southern is DJF)::

    T_s   = T_annual + 0.9 * A * phase_s * sign(lat)       A: half the annual range
    belts = those of lat - 23.4 deg * phase_s * (0.25 + 0.45 * continentality)
    wind  = belts(lat_eff) + monsoon: down the gradient of the seasonal T anomaly

Moisture model (per world cell and physics step; four quasi-steady seasons after a spin-up)::

    q <- q - div(q U) dt                         finite-volume upwind, conservative
    q += E_ocean = r_o * max(0, q_sat(SST) - q)  oceans are sources
    q += E_land  = P_recent * AET/P(PET/P_recent)  recycling over land (Budyko-Fu)
    P  = q * (k_zone(lat_eff) + k_conv(T) + k_oro * max(0, U . grad h))
         + c * max(0, q - q_sat(T_air(h)))       condensation when lifted air saturates
    q -= P

Land returns to the air the share of the recent rain that evapotranspires (Fu's curve of
the local dryness, as in the hex water balance); the rest is runoff. On Earth ~40 % of the
rain on land comes from land evaporation (Trenberth et al. 2007, van der Ent et al. 2010);
without this recycling continental interiors dry out far too much.

Every step keeps ``dM = E - P + inflow - outflow`` (M = sum of q, the fluxes through the
open border of the model area counted separately); the closure is reported in
``WorldClimate.balance`` (relative residual ~1e-15). ``q_sat`` follows Clausius-Clapeyron
(7 %/K). Rain shadows, wet windward slopes, monsoons and the drying of continental interiors
come out of the transport; nothing is painted. The precipitation is scaled by one constant
so the annual mean over the oceans matches the profile target (Earth: ~1100 mm/yr).

Hex water balance per season: snow below ~+1 degC (normal daily spread), degree-day melt,
a soil bucket with Fu's curve for the evapotranspiration, quick flow, groundwater recharge
and baseflow; ``P = AET + R + D + dS`` closes per hex (see :func:`season_balance`).

Units: temperature degC, wind m/s, water mm/yr (seasonal layers: mm per season),
elevation m above sea level.
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
    # land evapotranspiration source: the share of the Budyko AET of the recent rain that
    # goes back to the air (1 = all of it; lakes, dew and irrigation are not modelled)
    "recycling": 0.7,
    "hex_orographic": 0.6,  # hex-scale windward / leeward correction
}
#: Memory of the recent rain that feeds the land evapotranspiration (share per step).
RECENT_RAIN = 0.08
#: Annual mean warming of open-water coasts per degree of latitude poleward of 40 deg, degC.
MARITIME_WARMING = 0.3
#: Annual temperature range of a coast relative to a deep interior (Bergen vs Yakutsk).
COAST_SEASONALITY = 0.33


def _smooth01(x):
    t = np.clip(x, 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


#: World-raster cell size the per-step rates are calibrated at, km.
CELL_REF_KM = 86.0


def fu_share(pet_over_p, w: float = 2.6):
    """AET / P of Fu's form of the Budyko curve for the dryness ``PET / P`` (0..1)."""
    x = np.maximum(np.asarray(pet_over_p, dtype=np.float64), 0.0)
    return np.clip(1.0 + x - np.power(1.0 + np.power(x, w), 1.0 / w), 0.0, 1.0)


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


SEASONS = ("DJF", "MAM", "JJA", "SON")
#: Phase of the season in the annual cycle, -cos: DJF -1 (northern winter) .. JJA +1.
SEASON_PHASE = np.array([-1.0, 0.0, 1.0, 0.0])
#: Mean of a sinusoid over a 3-month window around its extreme (sin 45 deg / 45 deg).
SEASON_MEAN_FACTOR = 0.9003
#: Solar declination at the middle of the season, deg.
DECLINATION_DEG = 23.44
#: Physics steps per season (each ~3.5 h of wind travel for the fastest wind; the moisture
#: residence time is a few days, so a season reaches its own quasi-steady state well inside
#: this) and the trailing steps averaged into the season's precipitation.
SEASON_STEPS = 150
SEASON_AVERAGE = 70
#: Seasonal shift of the belts as a share of the declination: over the ocean, and the extra
#: over a broad continent.
SHIFT_OCEAN = 0.25
SHIFT_LAND = 0.45
#: Monsoon wind per 1 degC of seasonal land-sea contrast per 1000 km, m/s.
MONSOON = 0.45


def season_temperature(t_annual, amplitude, lat, season: int):
    """Mean air temperature of a 3-month season: northern summer is JJA, southern is DJF."""
    hemi = np.where(np.asarray(lat) >= 0.0, 1.0, -1.0)
    return t_annual + SEASON_MEAN_FACTOR * amplitude * SEASON_PHASE[season] * hemi


def _faces(u, v):
    """Velocities on cell faces: x-faces (ny, nx+1), y-faces (ny+1, nx); open borders."""
    uf = np.empty((u.shape[0], u.shape[1] + 1))
    uf[:, 1:-1] = 0.5 * (u[:, 1:] + u[:, :-1])
    uf[:, 0], uf[:, -1] = u[:, 0], u[:, -1]
    vf = np.empty((v.shape[0] + 1, v.shape[1]))
    vf[1:-1] = 0.5 * (v[1:] + v[:-1])
    vf[0], vf[-1] = v[0], v[-1]
    return uf, vf


def upwind_step(q, cxf, cyf):
    """One conservative first-order upwind step of ``dq/dt + div(q U) = 0`` (reference form).

    ``cxf``/``cyf`` are face Courant numbers (cells per step) from :func:`_faces`. Water moves
    from cell to cell through faces, so the total changes only by what crosses the outer
    border: returns ``(q_new, inflow, outflow)`` with the border fluxes in q*cells. The open
    border uses zero-gradient ghost cells. Positivity holds while the outgoing Courant sum
    of every cell stays <= 1. :func:`upwind_operator` is the same scheme as a sparse matrix.
    """
    qx = np.concatenate([q[:, :1], q, q[:, -1:]], axis=1)
    fx = np.where(cxf > 0, cxf * qx[:, :-1], cxf * qx[:, 1:])
    qy = np.concatenate([q[:1], q, q[-1:]], axis=0)
    fy = np.where(cyf > 0, cyf * qy[:-1], cyf * qy[1:])
    q_new = q - (fx[:, 1:] - fx[:, :-1]) - (fy[1:] - fy[:-1])
    inflow = (
        np.maximum(fx[:, 0], 0).sum()
        + np.maximum(-fx[:, -1], 0).sum()
        + np.maximum(fy[0], 0).sum()
        + np.maximum(-fy[-1], 0).sum()
    )
    outflow = (
        np.maximum(-fx[:, 0], 0).sum()
        + np.maximum(fx[:, -1], 0).sum()
        + np.maximum(-fy[0], 0).sum()
        + np.maximum(fy[-1], 0).sum()
    )
    return q_new, float(inflow), float(outflow)


def upwind_operator(cxf, cyf):
    """The upwind step as ``q_new = A @ q``; border fluxes as ``b_in @ q`` and ``b_out @ q``.

    The wind is fixed within a season, so the step is linear in q: assembling it once makes
    each step one sparse product. ``sum(A @ q) - sum(q) == (b_in - b_out) @ q`` exactly
    up to rounding - the scheme is conservative.
    """
    from scipy import sparse

    ny, nx = cyf.shape[0] - 1, cxf.shape[1] - 1
    idx = np.arange(ny * nx).reshape(ny, nx)
    rows, cols, vals = [], [], []

    def add(r, c, v):
        rows.append(r.ravel())
        cols.append(c.ravel())
        vals.append(v.ravel())

    xp, xn = np.maximum(cxf, 0.0), np.minimum(cxf, 0.0)
    yp, yn = np.maximum(cyf, 0.0), np.minimum(cyf, 0.0)
    j = np.arange(nx)
    left = idx[:, np.maximum(j - 1, 0)]  # ghost cells are the cell itself
    right = idx[:, np.minimum(j + 1, nx - 1)]
    i = np.arange(ny)
    up = idx[np.maximum(i - 1, 0)]
    down = idx[np.minimum(i + 1, ny - 1)]
    add(idx, idx, np.ones((ny, nx)))
    # right face j+1: out with xp from this cell, in with xn from the right neighbour
    add(idx, idx, -xp[:, 1:])
    add(idx, right, -xn[:, 1:])
    # left face j: in with xp from the left neighbour, out with xn from this cell
    add(idx, left, xp[:, :-1])
    add(idx, idx, xn[:, :-1])
    add(idx, idx, -yp[1:])
    add(idx, down, -yn[1:])
    add(idx, up, yp[:-1])
    add(idx, idx, yn[:-1])
    n = ny * nx
    a = sparse.csr_matrix(
        (np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))), shape=(n, n)
    )
    b_in = np.zeros((ny, nx))
    b_out = np.zeros((ny, nx))
    b_in[:, 0] += xp[:, 0]
    b_out[:, 0] -= xn[:, 0]
    b_in[:, -1] -= xn[:, -1]
    b_out[:, -1] += xp[:, -1]
    b_in[0] += yp[0]
    b_out[0] -= yn[0]
    b_in[-1] -= yn[-1]
    b_out[-1] += yp[-1]
    return a, b_in.ravel(), b_out.ravel()


@dataclass
class WorldClimate:
    grid: dict
    lat: np.ndarray
    sea: np.ndarray  # bool
    t_sea_level: np.ndarray  # degC, annual mean, without elevation
    continentality: np.ndarray  # 0 coast .. 1 deep interior
    wind_u: np.ndarray  # annual mean of the seasonal winds
    wind_v: np.ndarray
    precipitation: np.ndarray  # mm/yr, annual mean of the seasons
    moisture: np.ndarray  # relative q at the end of the cycle
    scale_mm_per_unit: float
    seasonal_amplitude: np.ndarray  # degC, half of the summer-winter range
    iterations: int
    balance: dict
    #: (4, ny, nx) per season DJF, MAM, JJA, SON: precipitation rate (mm/yr) and wind (m/s)
    seasonal_precipitation: np.ndarray | None = None
    seasonal_wind_u: np.ndarray | None = None
    seasonal_wind_v: np.ndarray | None = None
    #: v5.3 Earth system state (ocean, sea ice, winds aloft, coupler report) or None
    earth: object | None = None
    #: v5.4 atmosphere normals (``hexworld.atmosphere.normals.AtmosphereNormals``) or None
    atmosphere: object | None = None
    #: (12, ny, nx) monthly rain (mm/month) and sea-level air temperature of v5.4, or None
    monthly_precipitation: np.ndarray | None = None
    monthly_temperature: np.ndarray | None = None


def seasonal_winds(params, lat, cont, seasonal, land_h, cell_km, declination_deg=None):
    """Winds of the four seasons: belts follow the sun, land-sea contrast drives monsoons.

    * the wind and rain belts are those of ``lat - shift``: the thermal equator moves toward
      the summer hemisphere by ~0.25 of the solar declination over the ocean and up to ~0.7
      over large continents (ITCZ at ~6 deg over the ocean, up to ~16-20 deg over Asia);
    * a monsoon component blows down the large-scale gradient of the seasonal temperature
      anomaly: onshore toward the warm continent in summer, offshore in winter; it fades
      poleward of ~40 deg where the westerlies dominate;
    * mountains slow the wind (as in the annual model).
    """
    phi = np.abs(lat)
    hemi = np.where(lat >= 0.0, 1.0, -1.0)
    cont_broad = ndimage.gaussian_filter(cont, max(1.0, 520.0 / cell_km))
    low_scale = max(1.0, 900.0 / cell_km)
    reach = 1.0 / (1.0 + (phi / 38.0) ** 6)
    blocked = np.exp(-land_h / 4000.0)
    taper = 1.0 - (phi / 90.0) ** 2
    out = []
    for s in range(4):
        # the shift fades toward the poles, so the polar belts stay on the map
        decl = DECLINATION_DEG if declination_deg is None else declination_deg
        shift = decl * SEASON_PHASE[s] * (SHIFT_OCEAN + SHIFT_LAND * cont_broad) * taper
        lat_eff = np.clip(lat - shift, -90.0, 90.0)
        u, v = prevailing_wind(lat_eff, params.wind_strength)
        anomaly = SEASON_MEAN_FACTOR * seasonal * SEASON_PHASE[s] * hemi
        smooth = ndimage.gaussian_filter(anomaly, low_scale)
        gy, gx = np.gradient(smooth, cell_km / 1000.0)  # degC per 1000 km, +y south
        u = u + MONSOON * params.wind_strength * reach * gx
        v = v + MONSOON * params.wind_strength * reach * gy
        u = ndimage.gaussian_filter(u * blocked, 1.5)
        v = ndimage.gaussian_filter(v * blocked, 1.5)
        out.append((lat_eff, u, v))
    return out


def world_climate(
    params,
    grid: dict,
    elevation_m: np.ndarray,
    sea_level_m: float,
    latitude: Latitude,
    earth=None,
    atmosphere=None,
) -> WorldClimate:
    """Four seasons of temperature, wind and a mass-conservative moisture budget.

    ``earth`` (``EarthParams`` with ``enabled``) switches on the v5.3 Earth system: the
    planet's insolation, a wind-driven ocean with heat transport, sea ice and winds aloft,
    coupled to this model by :func:`hexworld.earth.coupler.couple` (see its docstring).
    Without it the climate is bit-identical to v4 (latitude SST, heuristic sea ice).

    ``atmosphere`` (``AtmosphereParams``, v5.4, needs ``earth``) adds the 12 monthly
    circulation and moisture normals (:mod:`hexworld.atmosphere`); with
    ``precipitation_coupling = full`` their rain replaces the seasonal rain of this model.
    """
    ny, nx = elevation_m.shape
    cell_km = float(grid["cell_km"])
    cell_m = cell_km * 1000.0
    ys = grid["y0_km"] + np.arange(ny) * cell_km
    lat = np.repeat(latitude.at(ys)[:, None], nx, axis=1)
    height = np.asarray(elevation_m, dtype=np.float64) - sea_level_m
    sea = height <= 0.0
    use_earth = earth is not None and getattr(earth, "enabled", False)
    if use_earth:
        from ..earth.ocean import ocean_mask
        from ..earth.planet import Planet

        planet = Planet.from_params(earth)
        # only sea connected to the world ocean is ocean; a closed depression below sea
        # level is land for the climate (the hydrology decides whether it holds a lake)
        sea, _basins = ocean_mask(sea)
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
    if use_earth:
        from ..earth.planet import annual_temperature_shift

        t_zone = t_zone + annual_temperature_shift(planet, lat)
    # seas that freeze in winter (zonal mean below ~ -3 degC) make their coasts continental:
    # no winter warmth from the water, a large annual range (Arctic coasts)
    sea_ice = _smooth01((2.0 - t_zone) / 10.0)
    # continental interiors: colder annual means at high latitude, a little warmer in the
    # tropics; open-water coasts poleward of 40 deg are warmer than the zonal mean (which is
    # dominated by the interiors): the oceans carry heat there
    t_sl = (
        t_zone
        - cont * np.clip(phi - 35.0, 0.0, None) * 0.12
        + cont * 0.8 * np.exp(-((phi / 20.0) ** 2))
        + (1.0 - cont) * (1.0 - sea_ice) * MARITIME_WARMING * np.clip(phi - 40.0, 0.0, 30.0)
    )
    sst = np.maximum(t_zone, -1.8)
    t_sl = np.where(sea, sst, t_sl)
    cont_season = np.maximum(cont, sea_ice)
    seasonal = (
        params.seasonality
        * (1.0 + 0.32 * phi)
        * (COAST_SEASONALITY + (1.0 - COAST_SEASONALITY) * cont_season)
    )
    earth_state = None
    if use_earth:
        from ..earth.coupler import couple

        earth_state = couple(
            params, earth, planet, lat, phi, sea, cont, t_zone, land_h, cell_km, sea_ice
        )
        t_sl, seasonal, winds = earth_state.t_sl, earth_state.seasonal, earth_state.winds
    else:
        winds = seasonal_winds(params, lat, cont, seasonal, land_h, cell_km)

    # -- conservative moisture transport, season by season -------------------------------
    smax = max(max(float(np.hypot(u, v).max()) for _, u, v in winds), 1e-6)
    dt_s = cell_m / smax  # one physics step: the fastest wind crosses one cell
    gyh, gxh = np.gradient(land_h, cell_m)
    # the rates are per physics step, and a step moves the fastest air one cell: scaling
    # them with the cell size makes moisture decay per kilometre, not per cell, so the same
    # world has the same climate whatever the resolution of the world raster
    f_cell = cell_km / CELL_REF_KM

    def per_step(k):
        return 1.0 - np.power(1.0 - np.clip(k, 0.0, 0.999), f_cell)

    r_ocean = per_step(RATES["ocean"])
    condensation = per_step(RATES["condensation"])
    recent = per_step(RECENT_RAIN)
    q = np.where(sea, q_sat(sst), 0.3 * q_sat(t_sl)).ravel()
    sea_f = sea.astype(np.float64).ravel()
    sea_idx = np.nonzero(sea.ravel())[0]
    land_idx = np.nonzero(~sea.ravel())[0]
    p_recent = np.zeros(q.shape)  # recent rain per cell, moisture units per step
    target = PROFILE_TARGET_OCEAN_MM.get(profile, 1100.0) * params.precipitation_scale
    ocean_p = None
    # mm/yr per moisture unit, estimated on the fly from the ocean rain (the final scale is
    # fitted the same way after the run); None until the spin-up has settled
    scale_est = None if sea.any() else target / 0.03
    seasonal_p = np.zeros((4, ny, nx))
    budget = {"evaporation": 0.0, "precipitation": 0.0, "inflow": 0.0, "outflow": 0.0}
    m_start = 0.0
    measured = 0
    substeps_max = 1
    for run, s in enumerate((3, 0, 1, 2, 3)):  # SON spins the model up, then a full year
        lat_eff, u, v = winds[s]
        t_s = season_temperature(t_sl, seasonal, lat, s)
        t_s = np.where(sea, np.maximum(t_s, -1.8), t_s)
        if earth_state is not None:
            t_s = np.where(sea, earth_state.sst_season[s], t_s)
        t_air = t_s - params.lapse_rate_c_per_km * land_h / 1000.0
        qs_air = q_sat(t_air)
        qs_sea = q_sat(t_s)
        circ = circulation_factor(lat_eff)
        k_zone = RATES["zone"] * circ * params.moisture_transport_damping
        # deep convection needs warm air and large-scale ascent (suppressed under the highs)
        k_conv = (
            RATES["convective"]
            * np.clip((t_air - 20.0) / 8.0, 0.0, 1.0)
            * np.clip(circ / 0.6, 0.0, 1.5)
        )
        if earth_state is not None and earth_state.aloft is not None:
            # air is lifted over the barrier by the wind of the lower troposphere (850 hPa)
            u_o, v_o = earth_state.aloft.u[s, 0], earth_state.aloft.v[s, 0]
        else:
            u_o, v_o = u, v
        w_up = u_o * gxh + v_o * gyh  # m/s of forced ascent (+ windward, - leeward)
        k_oro = RATES["orographic"] * np.clip(w_up / 0.01, 0.0, RATES["orographic_cap"])
        k_total = per_step(np.clip(k_zone + k_conv + k_oro, 0.0, 0.5))
        uf, vf = _faces(u, v)
        cxf = uf * dt_s / cell_m
        cyf = vf * dt_s / cell_m
        out_c = (
            np.maximum(cxf[:, 1:], 0)
            + np.maximum(-cxf[:, :-1], 0)
            + np.maximum(cyf[1:], 0)
            + np.maximum(-cyf[:-1], 0)
        )
        nsub = max(1, int(math.ceil(float(out_c.max()) / 0.9)))
        substeps_max = max(substeps_max, nsub)
        op, b_in, b_out = upwind_operator(cxf / nsub, cyf / nsub)
        qs_air, qs_sea, k_total = qs_air.ravel(), qs_sea.ravel(), k_total.ravel()
        e_sat = sea_f * r_ocean
        if earth_state is not None:  # sea ice shuts the evaporation off
            e_sat = e_sat * (1.0 - earth_state.ice_season[s].ravel())
        # land evapotranspiration returns the Budyko share of the rain that fell (moisture
        # recycling): AET/P from Fu's curve of the local PET/P, the rest is runoff
        pet_land = potential_et(t_air).ravel()[land_idx]
        if run == 1:
            m_start = float(q.sum())
        acc = np.zeros_like(q)
        for step in range(SEASON_STEPS):
            for _ in range(nsub):
                if run:
                    budget["inflow"] += float(b_in @ q)
                    budget["outflow"] += float(b_out @ q)
                q = op @ q
            evap = e_sat * np.maximum(qs_sea - q, 0.0)
            if scale_est is not None and land_idx.size:
                pl = p_recent[land_idx]
                evap[land_idx] += (
                    RATES["recycling"] * pl * fu_share(pet_land / np.maximum(pl * scale_est, 1e-3))
                )
            q += evap
            excess = np.maximum(q - qs_air, 0.0)
            precip = q * k_total
            precip += condensation * excess
            np.minimum(precip, q, out=precip)
            q -= precip
            p_recent += recent * (precip - p_recent)
            if sea.any():
                ocean_now = float(precip[sea_idx].mean())
                ocean_p = ocean_now if ocean_p is None else ocean_p + 0.02 * (ocean_now - ocean_p)
                if run or step >= 40:
                    scale_est = target / max(ocean_p, 1e-12)
            if run:
                budget["evaporation"] += float(evap.sum())
                budget["precipitation"] += float(precip.sum())
                measured += 1
            if step >= SEASON_STEPS - SEASON_AVERAGE:
                acc += precip
        if run:
            seasonal_p[s] = (acc / SEASON_AVERAGE).reshape(ny, nx)
    m_end = float(q.sum())
    annual = seasonal_p.mean(axis=0)
    ocean_mean = float(annual[sea].mean()) if sea.any() else float(annual.mean())
    scale = target / max(ocean_mean, 1e-12)
    precip_mm = annual * scale
    residual = (m_end - m_start) - (
        budget["evaporation"] - budget["precipitation"] + budget["inflow"] - budget["outflow"]
    )
    per_cell_year = scale / max(measured * q.size, 1)  # moisture units -> mm/yr world mean
    wc = WorldClimate(
        grid=grid,
        lat=lat,
        sea=sea,
        t_sea_level=t_sl,
        continentality=cont,
        wind_u=np.mean([w[1] for w in winds], axis=0),
        wind_v=np.mean([w[2] for w in winds], axis=0),
        precipitation=precip_mm,
        moisture=q.reshape(ny, nx),
        scale_mm_per_unit=scale,
        seasonal_amplitude=seasonal,
        iterations=measured,
        balance={
            "ocean_mean_mm_yr": float(precip_mm[sea].mean()) if sea.any() else None,
            "land_mean_mm_yr": float(precip_mm[~sea].mean()) if (~sea).any() else None,
            "steps": measured,
            "step_seconds": dt_s,
            "advection_substeps": substeps_max,
            "seasons": list(SEASONS),
            "seasonal_land_mean_mm_yr": [
                float(seasonal_p[s][~sea].mean() * scale) if (~sea).any() else None
                for s in range(4)
            ],
            # world means over the year, mm/yr: E - P + in - out = dM, exactly
            "evaporation_mm_yr": budget["evaporation"] * per_cell_year,
            "precipitation_mm_yr": budget["precipitation"] * per_cell_year,
            "border_inflow_mm_yr": budget["inflow"] * per_cell_year,
            "border_outflow_mm_yr": budget["outflow"] * per_cell_year,
            "storage_change_mm_yr": (m_end - m_start) * per_cell_year,
            "residual_mm_yr": residual * per_cell_year,
            "relative_residual": abs(residual) / max(budget["precipitation"], 1e-30),
        },
        seasonal_precipitation=seasonal_p * scale,
        seasonal_wind_u=np.stack([w[1] for w in winds]),
        seasonal_wind_v=np.stack([w[2] for w in winds]),
        earth=earth_state,
    )
    use_atm = atmosphere is not None and getattr(atmosphere, "enabled", False)
    if use_atm and use_earth:
        from ..atmosphere.stage import attach_normals

        attach_normals(wc, params, atmosphere, planet, earth, land_h, cont, cell_km, t_zone)
    elif use_atm:
        wc.balance["atmosphere"] = {"status": "blocked", "reason": "needs earth.enabled (v5.3)"}
    return wc


def _bilinear(field_, grid, x, y):
    fx = (np.asarray(x) - grid["x0_km"]) / grid["cell_km"]
    fy = (np.asarray(y) - grid["y0_km"]) / grid["cell_km"]
    coords = np.vstack([np.ravel(fy), np.ravel(fx)])
    return ndimage.map_coordinates(field_, coords, order=1, mode="nearest").reshape(np.shape(x))


#: Hex water balance, per season. Snow: share of P below a +1 degC threshold under normal
#: daily temperatures (sd 4 degC); melt by degree days. Soil: a bucket; the surplus splits
#: into quick flow and groundwater recharge; groundwater drains as baseflow.
DAILY_T_SD_C = 4.0
SNOW_THRESHOLD_C = 1.0
DEGREE_DAY_MM = 3.0  # mm of melt per positive degree-day
SOIL_CAPACITY_MM = 150.0
RECHARGE_SHARE = 0.35
BASEFLOW_TIMESCALE_YR = 1.0
DAYS_PER_SEASON = 365.25 / 4.0
#: Fu's shape parameter within a season: the soil bucket already carries water between the
#: seasons, so the curve is steeper than the annual Budyko curve (w = 2.6).
SEASON_FU_W = 2.6
BALANCE_CYCLES = 5  # years of spin-up of the stores; the last one is reported


def positive_degree_days(t_mean, sd=DAILY_T_SD_C, days=DAYS_PER_SEASON):
    """Expected sum of max(T, 0) over ``days`` with normally distributed daily T."""
    from scipy.special import ndtr

    z = np.asarray(t_mean) / sd
    return days * (sd * np.exp(-0.5 * z * z) / math.sqrt(2.0 * math.pi) + t_mean * ndtr(z))


def season_balance(params, t_s, p_s, pet_s, cycles: int = BALANCE_CYCLES):
    """Snowpack, soil bucket and groundwater through the seasons (mm per season).

    ``t_s``, ``p_s``, ``pet_s``: (4, n) arrays, DJF..SON. Every store is tracked, so the
    reported year closes ``P = AET + R + D + dS`` per hex to rounding. A snowpack that grows
    every year is a glacier in equilibrium: its yearly gain leaves as ice discharge, counted
    in the runoff (and in ``ice_discharge``), not as storage.
    """
    from scipy.special import ndtr

    n = p_s.shape[1]
    swe = np.zeros(n)
    capacity = SOIL_CAPACITY_MM
    # every store starts empty: the stores then rise monotonically toward their periodic
    # cycle (each season's update is non-decreasing in the store), so over the reported
    # year no store loses water and AET <= P holds in every hex
    soil = np.zeros(n)
    ground = np.zeros(n)
    k_base = 1.0 - math.exp(-0.25 / BASEFLOW_TIMESCALE_YR)
    snow_share = ndtr((SNOW_THRESHOLD_C - t_s) / DAILY_T_SD_C)
    melt_pot = DEGREE_DAY_MM * positive_degree_days(t_s)
    deep = params.deep_infiltration
    out = {}
    for cycle in range(cycles):
        last = cycle == cycles - 1
        if last:
            start = (swe.copy(), soil.copy(), ground.copy())
            acc = {k: np.zeros(n) for k in ("aet", "runoff", "deep", "snow", "base", "melt")}
            seasonal_runoff = np.zeros((4, n))
            swe_seasons = np.zeros((4, n))
        for s in range(4):
            snow = p_s[s] * snow_share[s]
            swe += snow
            melt = np.minimum(swe, melt_pot[s])
            swe -= melt
            water = p_s[s] - snow + melt
            supply = soil + water
            aet = budyko_aet(supply, pet_s[s], SEASON_FU_W)
            soil = supply - aet
            surplus = np.maximum(soil - capacity, 0.0)
            soil -= surplus
            lost = deep * surplus
            recharge = RECHARGE_SHARE * (surplus - lost)
            quick = surplus - lost - recharge
            ground += recharge
            base = k_base * ground
            ground -= base
            if last:
                acc["aet"] += aet
                acc["runoff"] += quick + base
                acc["deep"] += lost
                acc["snow"] += snow
                acc["base"] += base
                acc["melt"] += melt
                seasonal_runoff[s] = quick + base
                swe_seasons[s] = swe
    d_swe = swe - start[0]
    ice = np.maximum(d_swe, 0.0)  # perennial accumulation leaves as glacier flow
    # ...in the melt season: the warmest season of the hex
    warm = np.argmax(t_s, axis=0)
    seasonal_runoff[warm, np.arange(n)] += ice
    storage = d_swe - ice + (soil - start[1]) + (ground - start[2])
    p_year = p_s.sum(axis=0)
    out.update(
        aet=acc["aet"],
        runoff=acc["runoff"] + ice,
        ice_discharge=ice,
        deep=acc["deep"],
        snow=acc["snow"],
        baseflow=acc["base"],
        melt=acc["melt"],
        storage_change=storage,
        residual=p_year - acc["aet"] - acc["runoff"] - ice - acc["deep"] - storage,
        seasonal_runoff=seasonal_runoff,
        snow_peak=swe_seasons.max(axis=0) - np.minimum(swe_seasons.min(axis=0), swe),
        soil_end=soil,
    )
    return out


class ClimateLayers(dict):
    """Hex climate layers (name -> array) with the water-balance diagnostics in ``balance``."""

    balance: dict = {}
    #: the four seasons behind the annual layers (DJF, MAM, JJA, SON), (4, rows, cols):
    #: "t" degC, "p" and "pet" mm per season; not stored as layers
    seasons: dict = {}


def hex_climate(
    params,
    world: WorldClimate,
    layout,
    elevation_m: np.ndarray,
    sea_level_m: float,
    lake_mask: np.ndarray | None = None,
    ocean: np.ndarray | None = None,
) -> dict:
    """Climate layers on the hex grid of the window for the given surface.

    Four seasons (DJF, MAM, JJA, SON; northern summer is JJA) of temperature, precipitation
    and PET; the annual layers are their sums or means. Land hexes run a snow / soil /
    groundwater balance through the seasons (:func:`season_balance`). The returned dict
    holds only layers; its ``balance`` attribute holds the land water-balance diagnostics.
    """
    from ..views import hex_gradient  # local import: views imports project, not stages

    hx, hy = layout.centers()
    g = world.grid
    shape = np.shape(hx)
    height = np.asarray(elevation_m, dtype=np.float64) - sea_level_m
    # the world ocean (v4.6: connected to it, not just below the datum); land below the
    # sea level keeps its negative height, so a deep dry depression is warmer (lapse rate)
    if ocean is None:
        sea = height <= 0.0
        land_h = np.maximum(height, 0.0)
    else:
        sea = np.asarray(ocean, dtype=bool).reshape(height.shape)
        land_h = np.where(sea, 0.0, height)
    lat = _bilinear(world.lat, g, hx, hy)
    t_sl = _bilinear(world.t_sea_level, g, hx, hy)
    amp = _bilinear(world.seasonal_amplitude, g, hx, hy)
    t = t_sl - params.lapse_rate_c_per_km * land_h / 1000.0
    gxh, gyh = hex_gradient(land_h, layout)
    n_season = 4 if world.seasonal_precipitation is not None else 0

    def orographic(p_world, u, v):
        # hex-scale windward / leeward correction: only the slope scale the world raster
        # misses (a band of ~1-4 hexes), not hex noise
        w_hex = (u * gxh + v * gyh) / 1000.0  # m/s (gradient is per km)
        anomaly = ndimage.gaussian_filter(w_hex, 1.0) - ndimage.gaussian_filter(w_hex, 3.0)
        p = p_world * np.clip(1.0 + RATES["hex_orographic"] * np.tanh(anomaly / 0.004), 0.4, 1.6)
        return np.where(sea, p_world, p)

    if n_season:
        t_s = np.stack([season_temperature(t, amp, lat, s) for s in range(4)])
        p_s = np.stack(
            [
                orographic(
                    np.maximum(_bilinear(world.seasonal_precipitation[s], g, hx, hy), 0.0),
                    _bilinear(world.seasonal_wind_u[s], g, hx, hy),
                    _bilinear(world.seasonal_wind_v[s], g, hx, hy),
                )
                / 4.0
                for s in range(4)
            ]
        )  # mm per season
        pet_s = potential_et(t_s) / 4.0
    else:  # a world solution without seasons (older caches): one season of a full year
        p1 = orographic(
            np.maximum(_bilinear(world.precipitation, g, hx, hy), 0.0),
            _bilinear(world.wind_u, g, hx, hy),
            _bilinear(world.wind_v, g, hx, hy),
        )
        t_s = np.stack([t] * 4)
        p_s = np.stack([p1 / 4.0] * 4)
        pet_s = np.stack([potential_et(t) / 4.0] * 4)
    u = _bilinear(world.wind_u, g, hx, hy)
    v = _bilinear(world.wind_v, g, hx, hy)
    p = p_s.sum(axis=0)
    pet = pet_s.sum(axis=0)
    open_water = sea if lake_mask is None else (sea | lake_mask)
    pet = np.where(open_water, 1.1 * pet, pet)  # open water evaporates a little above PET

    land = ~open_water
    flat = land.ravel()
    bal = season_balance(
        params,
        t_s.reshape(4, -1)[:, flat],
        p_s.reshape(4, -1)[:, flat],
        pet_s.reshape(4, -1)[:, flat],
    )

    def put(values, fill=0.0):
        out = np.full(t.size, fill, dtype=np.float64)
        out[flat] = values
        return out.reshape(shape)

    aet = np.where(land, put(bal["aet"]), pet)
    lake_surplus = np.maximum(p - pet, 0.0)
    infiltration_loss = np.where(
        sea, 0.0, np.where(land, put(bal["deep"]), params.deep_infiltration * lake_surplus)
    )
    runoff = np.where(
        sea,
        0.0,
        np.where(land, put(bal["runoff"]), (1.0 - params.deep_infiltration) * lake_surplus),
    )
    snow_fraction = np.where(land, put(bal["snow"]) / np.maximum(p, 1e-9), 0.0)
    snow_storage = np.where(land, put(bal["snow_peak"]), 0.0)
    summer = np.maximum(t_s[0], t_s[2]) if n_season else t + amp
    glacial = np.clip((1.5 - summer) / 6.0, 0.0, 1.0) * np.clip(p * snow_fraction / 350.0, 0.0, 1.0)
    glacial = np.where(sea, 0.0, glacial)
    humidity = np.where(pet > 1.0, p / np.maximum(pet, 1.0), 5.0)
    r_season = np.zeros((4, t.size))
    r_season[:, flat] = bal["seasonal_runoff"]
    r_season = r_season.reshape((4,) + shape)
    peak = np.argmax(r_season, axis=0).astype(np.uint8)
    peak = np.where(runoff > 1.0, peak, 255).astype(np.uint8)
    base_index = np.where(land, put(bal["baseflow"]) / np.maximum(put(bal["runoff"]), 1e-9), 0.0)
    melt_share = np.where(land, put(bal["melt"] + bal["ice_discharge"]), 0.0) / np.maximum(
        runoff + infiltration_loss + aet, 1e-9
    )
    lp = p[land]
    resid = bal["residual"]
    layers = ClimateLayers(
        {
            "latitude_deg": lat.astype(np.float32),
            "temperature_mean_c": t.astype(np.float32),
            "temperature_seasonal_c": amp.astype(np.float32),
            "temperature_djf_c": t_s[0].astype(np.float32),
            "temperature_jja_c": t_s[2].astype(np.float32),
            "wind_u_ms": u.astype(np.float32),
            "wind_v_ms": v.astype(np.float32),
            "precipitation_mm_yr": p.astype(np.float32),
            "precipitation_djf_mm": p_s[0].astype(np.float32),
            "precipitation_jja_mm": p_s[2].astype(np.float32),
            "potential_evapotranspiration_mm_yr": pet.astype(np.float32),
            "actual_evapotranspiration_mm_yr": np.where(sea, pet, aet).astype(np.float32),
            "humidity_index": np.clip(humidity, 0.0, 5.0).astype(np.float32),
            "snow_fraction": np.clip(snow_fraction, 0.0, 1.0).astype(np.float32),
            "snow_storage_mm": snow_storage.astype(np.float32),
            "snowmelt_share": np.clip(np.where(land, melt_share, 0.0), 0.0, 1.0).astype(np.float32),
            "glacial_potential": glacial.astype(np.float32),
            "infiltration_loss_mm_yr": infiltration_loss.astype(np.float32),
            "runoff_mm_yr": runoff.astype(np.float32),
            "baseflow_index": np.clip(base_index, 0.0, 1.0).astype(np.float32),
            "runoff_peak_season": peak,
        }
    )
    layers.seasons = {
        "t": t_s.astype(np.float32),
        "p": p_s.astype(np.float32),
        "pet": pet_s.astype(np.float32),
    }
    layers.balance = {
        "land_hexes": int(flat.sum()),
        "precipitation_mm_yr": float(lp.mean()) if lp.size else None,
        "evapotranspiration_mm_yr": float(bal["aet"].mean()) if lp.size else None,
        "runoff_mm_yr": float(bal["runoff"].mean()) if lp.size else None,
        "ice_discharge_mm_yr": float(bal["ice_discharge"].mean()) if lp.size else None,
        "deep_infiltration_mm_yr": float(bal["deep"].mean()) if lp.size else None,
        "storage_change_mm_yr": float(bal["storage_change"].mean()) if lp.size else None,
        "max_abs_residual_mm_yr": float(np.abs(resid).max()) if lp.size else 0.0,
    }
    return layers


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
