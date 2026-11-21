"""v4.5 ecological foundation: 12 months of climate, soils, monthly water, wetness, indices.

Everything here is *derived* from the final v4 state (surface after erosion, climate of the
four seasons, rivers and lakes, rock and sediment of the v3 epoch). Nothing is painted, no
biome is named: these are the environmental conditions a later biome stage will read.

Sources and status of each method (details: docs/research/v4_5_evidence_matrix.md)
---------------------------------------------------------------------------------
* months from the seasonal physics of v4 (``derived_from_seasonal_physics``): temperature
  from the first harmonic of the season means, day-weighted to keep the annual mean;
  precipitation and PET of every season distributed over its three months (seasonal and
  annual sums are kept exactly);
* extraterrestrial radiation and day length: FAO-56 eq. 21-25 [published];
* PET: the v4 temperature-index curve per season (a heuristic, *not* Penman-Monteith),
  spread over the months by temperature and day length as in Thornthwaite's method;
* soils: parent material from geology and surface history; texture tables + smooth
  variation; FC/PWP from Saxton & Rawls (2006) [published, empirical]; AWC as in USDA
  NRCS; saturated conductivity from the same paper;
* water: the monthly bucket of the TZ 4.5 section 6.2 (degree-day snow, infiltration
  limit, stress-limited AET, percolation, groundwater, baseflow), repeated until the year
  is periodic;
* TWI (Beven & Kirkby 1979) and HAND (Renno et al. 2008) on the hex flow graph;
* monthly *proxies* of GDD, frost and dryness; the frost number of Nelson & Outcalt (1987)
  for the permafrost *potential*.

Units: degC, mm (per month unless named ``_yr``), m, MJ m^-2 day^-1, months 0..11 = Jan..Dec.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
from scipy.special import ndtr

MONTH_DAYS = np.array([31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31], dtype=np.float64)
YEAR_DAYS = float(MONTH_DAYS.sum())  # 365, a fixed model calendar (no leap years)
#: day of the year at the middle of each month (1 = 1 January)
MONTH_MID_DOY = np.cumsum(MONTH_DAYS) - MONTH_DAYS / 2.0 + 0.5
MONTH_NAMES = ("янв", "фев", "мар", "апр", "май", "июн", "июл", "авг", "сен", "окт", "ноя", "дек")
#: season of each month: 0 DJF, 1 MAM, 2 JJA, 3 SON (December belongs to DJF)
SEASON_OF_MONTH = np.array([0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3, 0])
#: mean of cos(2 pi (m - m0) / 12) over the three months centred on m0
_SEASON_HARMONIC = (1.0 + 2.0 * math.cos(math.pi / 6.0)) / 3.0

G_SC = 0.0820  # solar constant, MJ m^-2 min^-1 (FAO-56)


# -- solar geometry -------------------------------------------------------------------------
def extraterrestrial_radiation(lat_deg, doy):
    """FAO-56 extraterrestrial radiation R_a (MJ m^-2 day^-1) and day length N (hours).

    Polar day and night are handled by clipping ``-tan(phi) tan(delta)`` to [-1, 1]
    (sunset hour angle pi or 0), so the poles give finite values and R_a >= 0.
    """
    phi = np.radians(np.asarray(lat_deg, dtype=np.float64))
    j = np.asarray(doy, dtype=np.float64)
    dr = 1.0 + 0.033 * np.cos(2.0 * np.pi * j / 365.0)
    delta = 0.409 * np.sin(2.0 * np.pi * j / 365.0 - 1.39)
    cos_phi = np.where(np.abs(np.cos(phi)) < 1e-9, 1e-9, np.cos(phi))
    x = -np.sin(phi) * np.sin(delta) / (cos_phi * np.cos(delta))
    ws = np.arccos(np.clip(x, -1.0, 1.0))
    ra = (
        (24.0 * 60.0 / np.pi)
        * G_SC
        * dr
        * (ws * np.sin(phi) * np.sin(delta) + np.cos(phi) * np.cos(delta) * np.sin(ws))
    )
    return np.maximum(ra, 0.0), 24.0 * ws / np.pi


# -- months from the seasons ------------------------------------------------------------------
def monthly_temperature(t_seasons, t_annual=None):
    """12 monthly means from the four season means (DJF, MAM, JJA, SON), (4, ...) -> (12, ...).

    First harmonic through the season means: ``T_m = T + A (cos(2 pi (m - 6) / 12) - c)``
    with ``A = (T_JJA - T_DJF) / (2 * 0.911)`` (negative in the south: the warm month is
    January) and ``c`` the day-weighted mean of the cosine, so the day-weighted annual mean
    is exactly ``T`` (the v4 annual mean when given).
    """
    t_s = np.asarray(t_seasons, dtype=np.float64)
    t_ann = t_s.mean(axis=0) if t_annual is None else np.asarray(t_annual, dtype=np.float64)
    amp = (t_s[2] - t_s[0]) / (2.0 * _SEASON_HARMONIC)
    cosm = np.cos(2.0 * np.pi * (np.arange(12) - 6.0) / 12.0)
    c = float((MONTH_DAYS * cosm).sum() / YEAR_DAYS)
    shape = (12,) + (1,) * amp.ndim
    return t_ann[None] + amp[None] * (cosm - c).reshape(shape)


def distribute_seasons(season_totals, weights):
    """Split each season total (4, ...) over its three months by ``weights`` (12, ...).

    Within a season the months share its total in proportion to their (non-negative)
    weights; a season whose weights are all zero is split by days. Season and annual sums
    are conserved exactly.
    """
    totals = np.asarray(season_totals, dtype=np.float64)
    w = np.maximum(np.asarray(weights, dtype=np.float64), 0.0)
    out = np.zeros_like(w)
    for s in range(4):
        months = np.flatnonzero(SEASON_OF_MONTH == s)
        ws = w[months]
        total_w = ws.sum(axis=0)
        days = MONTH_DAYS[months].reshape((3,) + (1,) * (w.ndim - 1))
        share = np.where(total_w > 0, ws / np.where(total_w > 0, total_w, 1.0), days / days.sum())
        out[months] = share * totals[s][None]
    return out


def monthly_precipitation(p_seasons):
    """Monthly precipitation (mm) from the season totals of the moisture transport.

    The season rates are joined by the smoothest periodic curve through them (Fourier
    terms 0, 1 and 2), so a wet season fades into the dry one instead of stepping; each
    season keeps exactly its total. Provenance: ``derived_from_seasonal_physics``.
    """
    p_s = np.asarray(p_seasons, dtype=np.float64)
    days_s = np.array([90.0, 92.0, 92.0, 91.0])
    rate = p_s / days_s.reshape((4,) + (1,) * (p_s.ndim - 1))  # mm/day, centres m = 0,3,6,9
    a0 = rate.mean(axis=0)
    a1 = 0.5 * (rate[0] - rate[2])
    b1 = 0.5 * (rate[1] - rate[3])
    a2 = 0.25 * (rate[0] - rate[1] + rate[2] - rate[3])
    m = np.arange(12, dtype=np.float64).reshape((12,) + (1,) * (p_s.ndim - 1))
    th = 2.0 * np.pi * m / 12.0
    curve = a0 + a1 * np.cos(th) + b1 * np.sin(th) + a2 * np.cos(2.0 * th)
    curve = np.maximum(curve, 0.05 * a0[None])
    days = MONTH_DAYS.reshape((12,) + (1,) * (p_s.ndim - 1))
    return distribute_seasons(p_s, curve * days)


def monthly_pet(pet_seasons, t_monthly, lat_deg):
    """Monthly PET (mm): each v4 season's PET spread by temperature and day length.

    Thornthwaite's monthly correction (day length / 12 h, days / 30) applied to the v4
    temperature-index curve; the season sums of v4 are kept exactly.
    """
    from .climate import potential_et

    t_m = np.asarray(t_monthly, dtype=np.float64)
    lat = np.broadcast_to(np.asarray(lat_deg, dtype=np.float64), t_m.shape[1:])
    _ra, day_len = extraterrestrial_radiation(
        lat[None], MONTH_MID_DOY.reshape((12,) + (1,) * lat.ndim)
    )
    days = MONTH_DAYS.reshape((12,) + (1,) * lat.ndim)
    weight = potential_et(t_m) * (day_len / 12.0) * (days / 30.0)
    return distribute_seasons(pet_seasons, weight)


# -- soils ----------------------------------------------------------------------------------
PARENT_MATERIALS = (
    "none",
    "crystalline_bedrock",
    "sedimentary_bedrock",
    "orogenic_bedrock",
    "young_volcanic",
    "weathered_volcanic",
    "alluvium",
    "glacial_till",
    "aeolian_sand",
    "loess",
    "marine_lacustrine",
    "colluvium",
)
PM = {name: i for i, name in enumerate(PARENT_MATERIALS)}
#: base texture (sand, silt, clay of the fine earth) and coarse fragments of each material;
#: typical values of the corresponding soils (USDA texture classes; tunable)
TEXTURE = {
    "none": (0.33, 0.34, 0.33, 0.0),
    "crystalline_bedrock": (0.58, 0.24, 0.18, 0.25),
    "sedimentary_bedrock": (0.38, 0.36, 0.26, 0.15),
    "orogenic_bedrock": (0.46, 0.33, 0.21, 0.35),
    "young_volcanic": (0.62, 0.30, 0.08, 0.35),
    "weathered_volcanic": (0.22, 0.30, 0.48, 0.10),
    "alluvium": (0.30, 0.46, 0.24, 0.03),
    "glacial_till": (0.42, 0.34, 0.24, 0.25),
    "aeolian_sand": (0.90, 0.06, 0.04, 0.0),
    "loess": (0.14, 0.70, 0.16, 0.0),
    "marine_lacustrine": (0.16, 0.40, 0.44, 0.0),
    "colluvium": (0.50, 0.30, 0.20, 0.45),
}
PARENT_TITLES = {
    "none": "нет почвы",
    "crystalline_bedrock": "кристаллические породы щитов",
    "sedimentary_bedrock": "осадочные породы платформ",
    "orogenic_bedrock": "складчатые и метаморфические породы",
    "young_volcanic": "молодые вулканические породы и пеплы",
    "weathered_volcanic": "выветрелые базальты",
    "alluvium": "речной аллювий",
    "glacial_till": "ледниковая морена",
    "aeolian_sand": "эоловые пески",
    "loess": "лёсс",
    "marine_lacustrine": "морские и озёрные отложения",
    "colluvium": "склоновый обломочный чехол",
}
DRAINAGE_CLASSES = (
    "excessively",
    "well",
    "moderately_well",
    "somewhat_poorly",
    "poorly",
    "very_poorly",
)
DRAINAGE_TITLES = (
    "избыточный",
    "хороший",
    "умеренно хороший",
    "несколько затруднённый",
    "плохой",
    "очень плохой",
)


def saxton_rawls(sand, clay, om_percent):
    """Saxton & Rawls (2006) moisture at -1500 and -33 kPa, saturation and Ks.

    ``sand``, ``clay`` - fractions 0..1 of the fine earth; ``om_percent`` - organic matter,
    % by weight (the paper's convention). Returns ``(wp, fc, sat, ks_mm_h)``, volumetric.
    Check: loam (0.40 / 0.20, 2.5 %) gives WP 0.137, FC 0.280.
    """
    s = np.asarray(sand, dtype=np.float64)
    c = np.asarray(clay, dtype=np.float64)
    om = np.asarray(om_percent, dtype=np.float64)
    t1500 = -0.024 * s + 0.487 * c + 0.006 * om + 0.005 * s * om - 0.013 * c * om
    t1500 = t1500 + 0.068 * s * c + 0.031
    wp = t1500 + (0.14 * t1500 - 0.02)
    t33 = -0.251 * s + 0.195 * c + 0.011 * om + 0.006 * s * om - 0.027 * c * om
    t33 = t33 + 0.452 * s * c + 0.299
    fc = t33 + (1.283 * t33 * t33 - 0.374 * t33 - 0.015)
    ts33 = 0.278 * s + 0.034 * c + 0.022 * om - 0.018 * s * om - 0.027 * c * om
    ts33 = ts33 - 0.584 * s * c + 0.078
    s33 = ts33 + (0.636 * ts33 - 0.107)
    sat = fc + s33 - 0.097 * s + 0.043
    with np.errstate(divide="ignore", invalid="ignore"):
        b = (math.log(1500.0) - math.log(33.0)) / (np.log(fc) - np.log(wp))
        lam = 1.0 / b
        ks = 1930.0 * np.maximum(sat - fc, 0.0) ** (3.0 - lam)
    return wp, fc, sat, ks


def available_water(depth_m, fc, wp, coarse, root_depth_m=1.0):
    """AWC (mm) = 1000 * min(depth, root depth) * max(FC - PWP, 0) * (1 - coarse)."""
    d_root = np.minimum(np.asarray(depth_m, dtype=np.float64), root_depth_m)
    return (
        1000.0
        * d_root
        * np.maximum(np.asarray(fc) - np.asarray(wp), 0.0)
        * (1.0 - np.asarray(coarse))
    )


def _simplex(sand, silt, clay):
    total = np.maximum(sand + silt + clay, 1e-12)
    return sand / total, silt / total, clay / total


@dataclass
class SoilInputs:
    land: np.ndarray  # bool
    rock_type: np.ndarray  # ROCK_TYPES code of the v3 epoch
    sediment_m: np.ndarray
    deposition_m: np.ndarray  # deposited during v4 (fresh alluvium / colluvium)
    slope_m_km: np.ndarray
    elevation_above_sea_m: np.ndarray
    t_annual: np.ndarray
    humidity: np.ndarray  # P / PET
    river: np.ndarray  # bool
    discharge_m3_s: np.ndarray
    lake_shore: np.ndarray  # bool: a lake hex or its neighbour
    coastal: np.ndarray  # bool: land at the sea
    lgm_ice: np.ndarray  # bool
    volcanic_age_ma: np.ndarray  # NaN = none
    endorheic: np.ndarray  # bool
    glacial_now: np.ndarray  # 0..1 glacial potential
    texture_noise: np.ndarray  # (3, ...) smooth noise -1..1, fixed in world coordinates


def parent_material(inp: SoilInputs) -> np.ndarray:
    """Dominant parent material of each land hex (``PARENT_MATERIALS`` codes).

    Bedrock from the rock type; transported materials override it where the surface
    history says so: young volcanic rock (< 2 Ma), fresh river deposits on gentle ground,
    glacial till under the ice of the last glacial maximum, wind-blown sand in dry
    lowlands with loose sediment, loess at the dry margins of the old ice, lake and sea
    deposits in low coastal plains and closed basins, debris on steep slopes.
    """
    rock = np.asarray(inp.rock_type)
    pm = np.full(rock.shape, PM["sedimentary_bedrock"], dtype=np.uint8)
    pm[rock == 1] = PM["crystalline_bedrock"]
    pm[rock == 2] = PM["sedimentary_bedrock"]
    pm[rock == 3] = PM["orogenic_bedrock"]
    pm[rock == 4] = PM["weathered_volcanic"]
    pm[(rock == 5) | (rock == 0)] = PM["sedimentary_bedrock"]
    age = np.nan_to_num(inp.volcanic_age_ma, nan=999.0)
    gentle = inp.slope_m_km < 6.0
    loose = inp.sediment_m > 150.0
    dry = inp.humidity < 0.2
    rules = [
        ("weathered_volcanic", (rock == 4) & (age >= 2.0)),
        ("young_volcanic", age < 2.0),
        ("marine_lacustrine", (inp.coastal | inp.endorheic | inp.lake_shore)
         & (inp.elevation_above_sea_m < 120.0) & loose & gentle),
        ("loess", ~inp.lgm_ice & (inp.humidity < 0.75) & (inp.humidity >= 0.2) & gentle
         & (inp.t_annual < 14.0) & loose),
        ("aeolian_sand", dry & loose & gentle & (inp.elevation_above_sea_m < 1500.0)),
        ("glacial_till", inp.lgm_ice & (inp.slope_m_km < 15.0)),
        # fresh deposits thick enough to be the soil's material, or a large lowland river
        (
            "alluvium",
            ((inp.deposition_m > 15.0) | (inp.river & (inp.discharge_m3_s > 100.0)))
            & (inp.slope_m_km < 3.0),
        ),
        ("colluvium", inp.slope_m_km > 25.0),
    ]  # fmt: skip
    for name, mask in rules:  # later rules win
        pm[mask] = PM[name]
    pm[~inp.land] = PM["none"]
    return pm


def soil_properties(inp: SoilInputs, pm: np.ndarray, params) -> dict:
    """Texture, depth, coarse fragments, organic potential, FC/PWP/AWC, Ks, salinity."""
    land = inp.land
    base = np.array([TEXTURE[name] for name in PARENT_MATERIALS])  # (n_pm, 4)
    sand, silt, clay, coarse = (base[pm, k] for k in range(4))
    # weathering intensity: warm, humid and stable ground turns rock to clay
    warm = np.clip((inp.t_annual + 5.0) / 30.0, 0.0, 1.0)
    wet = np.clip(inp.humidity / 1.5, 0.0, 1.0)
    weathering = warm * wet
    bedrock = np.isin(pm, [PM["crystalline_bedrock"], PM["sedimentary_bedrock"],
                           PM["orogenic_bedrock"], PM["weathered_volcanic"]])  # fmt: skip
    clay = clay + 0.18 * weathering * bedrock
    sand = sand - 0.12 * weathering * bedrock
    # smooth local variation, fixed in the world (not one random value per hex)
    n = inp.texture_noise
    sand = np.clip(sand + 0.06 * n[0], 0.02, 0.96)
    silt = np.clip(silt + 0.05 * n[1], 0.02, 0.9)
    clay = np.clip(clay + 0.04 * n[2], 0.02, 0.7)
    sand, silt, clay = _simplex(sand, silt, clay)
    slope = inp.slope_m_km
    coarse = np.clip(coarse + 0.012 * slope, 0.0, 0.85)

    # depth: a static potential of the ground (approximate; not integrated over time)
    f_slope = 1.0 / (1.0 + (slope / 12.0) ** 2)
    rate = np.clip(0.25 + 0.75 * weathering, 0.15, 1.0)  # cold or dry ground forms slowly
    young = np.nan_to_num(inp.volcanic_age_ma, nan=999.0) < 2.0
    depth = params.max_soil_depth_m * f_slope * rate
    deep_materials = np.isin(pm, [PM["alluvium"], PM["loess"], PM["marine_lacustrine"],
                                  PM["aeolian_sand"], PM["glacial_till"]])  # fmt: skip
    depth = np.where(
        deep_materials, np.maximum(depth, 0.9 * params.max_soil_depth_m * f_slope), depth
    )
    depth = np.where(young, 0.25 * depth, depth)
    depth = np.where(inp.glacial_now > 0.5, 0.2 * depth, depth)  # under perennial ice
    depth = np.clip(depth, 0.0, params.max_soil_depth_m)
    rocky = np.clip(1.0 - depth / 0.35, 0.0, 1.0) * np.clip(0.4 + slope / 30.0, 0.0, 1.0)
    rocky = np.where(land, np.maximum(rocky, 0.6 * (inp.glacial_now > 0.5)), 0.0)
    present = land & (depth >= 0.05) & (inp.glacial_now <= 0.5)

    # organic potential: productivity (warm and moist) against decomposition, and wetness
    organic = np.clip(wet * np.clip((inp.t_annual + 8.0) / 25.0, 0.0, 1.0), 0.0, 1.0)
    organic = np.clip(0.6 * organic + 0.4 * wet * (inp.t_annual < 4.0), 0.0, 1.0)
    om = 0.6 + 4.0 * organic  # % by weight for the pedotransfer (Saxton & Rawls: <= 8 %)
    wp, fc, sat, ks = saxton_rawls(sand, clay, om)
    bad = ~np.isfinite(wp) | ~np.isfinite(fc) | (wp >= fc) | (fc > sat) | (wp < 0.0)
    awc = available_water(depth, fc, wp, coarse, params.reference_root_depth_m)
    ks = np.where(np.isfinite(ks), ks, 0.0)
    infil = ks * 24.0 * 0.5  # mm/day: half the saturated rate as a daily mean intake
    saline = (
        land
        & (inp.humidity < 0.2)
        & (inp.endorheic | (inp.coastal & (inp.elevation_above_sea_m < 30.0)))
    )
    nan = np.nan

    def m(a):
        return np.where(present, a, nan).astype(np.float32)

    return {
        "soil_present": present,
        "parent_material_class": np.where(land, pm, PM["none"]).astype(np.uint8),
        "sand_frac": m(sand),
        "silt_frac": m(silt),
        "clay_frac": m(clay),
        "coarse_fragment_frac": m(coarse),
        "soil_depth_m": np.where(land, depth, nan).astype(np.float32),
        "soil_organic_potential": m(organic),
        "field_capacity_vol": m(fc),
        "wilting_point_vol": m(wp),
        "porosity_vol": m(sat),
        "awc_mm": np.where(land, np.where(present, awc, 0.0), nan).astype(np.float32),
        "ks_mm_h": m(ks),
        "infiltration_capacity_mm_day": m(infil),
        "terrain_rocky_fraction": np.where(land, rocky, nan).astype(np.float32),
        "soil_salinity_proxy": (saline & present).astype(np.uint8),
        "_pedotransfer_rejected": bad & present,
    }


# -- the monthly water balance ----------------------------------------------------------------
@dataclass
class WaterParams:
    t_rain_snow_c: float = 1.0  # centre of the rain/snow split
    daily_t_sd_c: float = 4.0  # spread of daily means within a month
    t_melt_c: float = 0.0
    k_dd_mm: float = 3.0  # mm of melt per positive degree-day
    depletion_p: float = 0.5  # share of AWC used before stress (FAO-56 "p")
    rain_days: float = 6.0  # rain days per month that use the daily infiltration capacity
    baseflow_timescale_yr: float = 1.0
    deep_loss_share: float = 0.05  # recharge leaving to deep aquifers (an explicit flux)
    rock_storage_mm: float = 5.0  # wet surface / fissures of bare rock (AWC = 0)
    max_years: int = 40
    tolerance_mm: float = 1e-4


@dataclass
class WaterResult:
    monthly: dict  # name -> (12, n) float64
    annual: dict  # name -> (n,)
    years: int
    converged: np.ndarray  # bool (n,)
    persistent_snow: np.ndarray  # bool (n,)
    max_residual_mm: float
    budget: dict = field(default_factory=dict)


def positive_degree_days(t_mean, sd, days):
    """Expected sum of max(T - 0, 0) over ``days`` with normal daily means (sd ``sd``)."""
    t = np.asarray(t_mean, dtype=np.float64)
    z = t / sd
    return days * (sd * np.exp(-0.5 * z * z) / math.sqrt(2.0 * math.pi) + t * ndtr(z))


def month_step(state, p, t, pet, awc, icap_day, drainage, days, w: WaterParams):
    """One month of the bucket model for all hexes. ``state`` = dict(S, GW, SWE).

    Returns the new state and the fluxes. Conservation per hex:
    ``S + GW + SWE + P == S' + GW' + SWE' + AET + runoff + deep_loss``.
    """
    s0, gw0, swe0 = state["S"], state["GW"], state["SWE"]
    snow = p * ndtr((w.t_rain_snow_c - t) / w.daily_t_sd_c)
    rain = p - snow
    swe = swe0 + snow
    melt = np.minimum(swe, w.k_dd_mm * positive_degree_days(t - w.t_melt_c, w.daily_t_sd_c, days))
    swe = swe - melt
    liquid = rain + melt
    infil = np.minimum(liquid, icap_day * w.rain_days * (days / 30.0))
    surface = liquid - infil
    cap = np.maximum(awc, w.rock_storage_mm)  # bare rock: a few mm of wet surface
    available = s0 + infil
    # FAO-56: the depletion fraction falls on days of high demand,
    # p = p_table + 0.04 (5 - ET mm/day), within [0.1, 0.8] (as in USGS SWB)
    p_adj = np.clip(w.depletion_p + 0.04 * (5.0 - pet / days), 0.1, 0.8)
    s_crit = (1.0 - p_adj) * cap
    stress = np.clip((available - s_crit) / np.maximum(cap - s_crit, 1e-9), 0.0, 1.0)
    aet = np.minimum(available, pet * stress)
    available = available - aet
    excess = np.maximum(available - cap, 0.0)
    available = available - excess
    slow = np.minimum(available, drainage * available)
    available = available - slow
    recharge = excess + slow
    deep = w.deep_loss_share * recharge
    gw = gw0 + recharge - deep
    k_base = 1.0 - math.exp(-days / YEAR_DAYS / w.baseflow_timescale_yr)
    base = k_base * gw
    gw = gw - base
    runoff = surface + base
    new = {"S": available, "GW": gw, "SWE": swe}
    flux = {
        "rain": rain,
        "snowfall": snow,
        "melt": melt,
        "aet": aet,
        "surface_runoff": surface,
        "recharge": recharge,
        "baseflow": base,
        "runoff": runoff,
        "deep_loss": deep,
        "saturation_excess": excess,  # part of the recharge: the bucket overflowed
    }
    return new, flux


def water_balance(p_m, t_m, pet_m, awc, icap_day, drainage, w: WaterParams) -> WaterResult:
    """12-month bucket model repeated until the year is periodic (spin-up).

    All arrays (12, n) / (n,), mm. Snow that accumulates every year without melting
    cannot reach a periodic state: those hexes are flagged ``persistent_snow`` and their
    yearly gain is reported as accumulation, *not* as runoff (no glacier flow is
    modelled). Budget: per hex and month, ``S + GW + SWE + P`` before equals ``S + GW + SWE
    + AET + runoff + deep_loss`` after, to rounding (float64).
    """
    n = awc.shape[0]
    state = {"S": np.minimum(awc, 0.5 * awc), "GW": np.zeros(n), "SWE": np.zeros(n)}
    start = {k: v.copy() for k, v in state.items()}
    keys = (
        "rain",
        "snowfall",
        "melt",
        "aet",
        "surface_runoff",
        "recharge",
        "baseflow",
        "runoff",
        "deep_loss",
        "saturation_excess",
        "soil_storage",
        "groundwater",
        "snowpack",
    )
    monthly = {k: np.zeros((12, n)) for k in keys}
    converged = np.zeros(n, dtype=bool)
    persistent = np.zeros(n, dtype=bool)
    active = np.arange(n)
    worst = 0.0
    years = 0
    for year in range(1, w.max_years + 1):
        if active.size == 0:
            break
        years = year
        # only hexes whose year is not yet periodic run again: a finished hex keeps its
        # last (periodic) year, so the result is the same as running everything
        sub = {k: v[active] for k, v in state.items()}
        first = {k: v.copy() for k, v in sub.items()}
        args = (awc[active], icap_day[active], drainage[active])
        for m in range(12):
            before = sub["S"] + sub["GW"] + sub["SWE"] + p_m[m, active]
            sub, flux = month_step(
                sub, p_m[m, active], t_m[m, active], pet_m[m, active], *args, MONTH_DAYS[m], w
            )
            after = sub["S"] + sub["GW"] + sub["SWE"] + flux["aet"] + flux["runoff"]
            after = after + flux["deep_loss"]
            worst = max(worst, float(np.abs(before - after).max()))
            for k, v in flux.items():
                monthly[k][m, active] = v
            monthly["soil_storage"][m, active] = sub["S"]
            monthly["groundwater"][m, active] = sub["GW"]
            monthly["snowpack"][m, active] = sub["SWE"]
        for k in state:
            state[k][active] = sub[k]
            start[k][active] = first[k]
        change = np.maximum(np.abs(sub["S"] - first["S"]), np.abs(sub["GW"] - first["GW"]))
        snow_gain = sub["SWE"] - first["SWE"]
        periodic = (change < w.tolerance_mm) & (np.abs(snow_gain) < w.tolerance_mm)
        # snow still growing when everything else has settled: accumulation, not spin-up
        growing = (change < w.tolerance_mm) & (snow_gain >= w.tolerance_mm) & (year >= 3)
        converged[active[periodic]] = True
        persistent[active[growing & ~periodic]] = True
        active = active[~(periodic | growing)]
    annual = {k: v.sum(axis=0) for k, v in monthly.items() if k not in
              ("soil_storage", "groundwater", "snowpack", "saturation_excess")}  # fmt: skip
    gain = state["SWE"] - start["SWE"]
    annual["snow_accumulation"] = np.where(persistent, gain, 0.0)
    # perennial snow has no meaningful absolute amount (it is spin-up length x yearly
    # gain): report the snow of the year itself, counted from the start of that year
    monthly["snowpack"][:, persistent] -= start["SWE"][persistent]
    annual["storage_change"] = state["S"] - start["S"] + state["GW"] - start["GW"] + gain
    p_year = p_m.sum(axis=0)
    resid = (
        p_year - annual["aet"] - annual["runoff"] - annual["deep_loss"] - annual["storage_change"]
    )
    return WaterResult(
        monthly=monthly,
        annual=annual,
        years=years,
        converged=converged | persistent,
        persistent_snow=persistent,
        # (hexes neither periodic nor accumulating snow after max_years keep converged=False)
        max_residual_mm=max(worst, float(np.abs(resid).max()) if n else 0.0),
        budget={
            "precipitation_mm_yr": float(p_year.mean()) if n else None,
            "aet_mm_yr": float(annual["aet"].mean()) if n else None,
            "runoff_mm_yr": float(annual["runoff"].mean()) if n else None,
            "deep_loss_mm_yr": float(annual["deep_loss"].mean()) if n else None,
            "storage_change_mm_yr": float(annual["storage_change"].mean()) if n else None,
            "snow_accumulation_mm_yr": float(annual["snow_accumulation"].mean()) if n else None,
        },
    )


# -- wetness on the flow graph -----------------------------------------------------------------
def twi(area_km2, slope_m_km, spacing_km):
    """Topographic wetness index ln(a / tan beta) and its resolution-free anomaly.

    ``a`` = upstream area / contour length (the hex edge, ``spacing / sqrt 3``), m; slope
    floored at 1e-4. The value depends on the grid: ``twi_anomaly = TWI - TWI_ref``, where
    ``TWI_ref`` is a single hex on a 1 % slope, compares grids of different spacing.
    """
    edge_m = spacing_km / math.sqrt(3.0) * 1000.0
    a = np.maximum(np.asarray(area_km2, dtype=np.float64), 0.0) * 1e6 / edge_m
    tan_b = np.maximum(np.asarray(slope_m_km, dtype=np.float64) / 1000.0, 1e-4)
    hex_area_m2 = 1.5 * math.sqrt(3.0) * (spacing_km / math.sqrt(3.0)) ** 2 * 1e6
    value = np.log(np.maximum(a, 1.0) / tan_b)
    ref = math.log(hex_area_m2 / edge_m / 0.01)
    return value, value - ref


def hand(elevation_m, receiver, reference_m):
    """Height above the nearest drainage *along the flow path* (Renno et al. 2008), m.

    ``reference_m``: the water level at drainage cells (a river hex: its elevation; a
    lake: its level; the sea: sea level), NaN elsewhere. Each hex is measured against the
    first drainage cell downstream, so a ridge next to a river in another basin never sees
    that river. Pointer jumping: after round r every hex within 2^r steps of its drainage
    knows it, and the hex 2^r ahead of a farther one is still upstream of that drainage.
    """
    z = np.asarray(elevation_m, dtype=np.float64).ravel()
    rc = np.asarray(receiver, dtype=np.int64).ravel()
    ref = np.asarray(reference_m, dtype=np.float64).ravel().copy()
    found = ~np.isnan(ref)
    nxt = rc.copy()
    for _ in range(64):
        if found.all():
            break
        take = ~found & found[nxt]
        ref = np.where(take, ref[nxt], ref)
        found = found | take
        ahead = nxt[nxt]
        if np.array_equal(ahead, nxt):
            break
        nxt = ahead
    ref = np.where(found, ref, z)  # a closed path without water: its own floor
    return np.maximum(z - ref, 0.0)


def path_steps(receiver, is_drainage, limit=8):
    """Hexes along the flow path to the first drainage cell (capped at ``limit``)."""
    rc = np.asarray(receiver, dtype=np.int64).ravel()
    dist = np.where(np.asarray(is_drainage).ravel(), 0, limit).astype(np.int64)
    for _ in range(limit):
        dist = np.minimum(dist, dist[rc] + 1)
        dist = np.where(np.asarray(is_drainage).ravel(), 0, dist)
    return dist


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(np.asarray(x, dtype=np.float64), -60.0, 60.0)))


# -- indices -------------------------------------------------------------------------------------
def bioclimatic(t_m, p_m, pet_m, aet_m, swe_m, dry_ratio=0.5, gdd_base=5.0):
    """Monthly-proxy indices (from 12 means, not daily data)."""
    d = MONTH_DAYS.reshape((12,) + (1,) * (t_m.ndim - 1))
    gdd = (d * np.maximum(t_m - gdd_base, 0.0)).sum(axis=0)
    fdd = (d * np.maximum(-t_m, 0.0)).sum(axis=0)
    tdd = (d * np.maximum(t_m, 0.0)).sum(axis=0)
    frost_number = np.sqrt(fdd) / np.maximum(np.sqrt(fdd) + np.sqrt(tdd), 1e-9)
    p_year = p_m.sum(axis=0)
    pet_year = pet_m.sum(axis=0)
    mean_p = np.maximum(p_m.mean(axis=0), 1e-9)
    return {
        "gdd5_proxy_c_day": gdd,
        "freezing_degree_days": fdd,
        "thawing_degree_days": tdd,
        "frost_number": frost_number,
        "permafrost_potential": np.clip((frost_number - 0.4) / 0.25, 0.0, 1.0),
        "warmest_month_c": t_m.max(axis=0),
        "coldest_month_c": t_m.min(axis=0),
        "annual_temperature_range_c": t_m.max(axis=0) - t_m.min(axis=0),
        "cold_month_count": (t_m < 0.0).sum(axis=0),
        "dry_month_count": (p_m < dry_ratio * pet_m).sum(axis=0),
        "growing_month_count": ((t_m >= gdd_base) & (aet_m >= 0.5 * pet_m)).sum(axis=0),
        "precip_seasonality_cv": 100.0 * p_m.std(axis=0) / mean_p,
        "aridity_p_over_pet": p_year / np.maximum(pet_year, 1.0),
        "aet_over_pet": aet_m.sum(axis=0) / np.maximum(pet_year, 1.0),
        "snow_cover_months": (swe_m > 10.0).sum(axis=0),
        "annual_max_swe_mm": swe_m.max(axis=0),
    }


__all__ = [
    "DRAINAGE_CLASSES",
    "MONTH_DAYS",
    "PARENT_MATERIALS",
    "SoilInputs",
    "WaterParams",
    "available_water",
    "bioclimatic",
    "distribute_seasons",
    "extraterrestrial_radiation",
    "hand",
    "month_step",
    "monthly_pet",
    "monthly_precipitation",
    "monthly_temperature",
    "parent_material",
    "saxton_rawls",
    "soil_properties",
    "twi",
    "water_balance",
]
