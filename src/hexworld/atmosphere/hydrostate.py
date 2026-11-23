"""Actual water and ice state of a weather year (TZ 5.4.1 G5): soil, groundwater, snow,
firn and glacier ice, river discharge and closed lakes, with carry-over between years.

Monthly aggregated mode (documented limit: no flood peaks inside a month; the flood day of
the weather year is reported separately). Per land hex and month::

    snow share   f_s = Phi((T_snow - T) / sigma)          sigma = 4 K daily spread [K]
    PDD          = days * E[max(X - T_melt, 0)],  X ~ N(T, sigma)   (sigma phi(mu/sigma) +
                   mu Phi(mu/sigma); exact for a Gaussian, not a monthly-mean proxy, N17)
    snow melt    = min(SWE, DDF_snow PDD); the PDD left melts firn then ice (DDF_ice, N18)
    liquid in    = P (1 - f_s) + snow melt
    infiltration = min(liquid, capacity x days x (0.3 if frozen soil), S_sat - S)
    AET          = min(PET K_s, S - S_wp), K_s = 1 above (1 - p) TAW, linear below (FAO-56)
    drainage     = (S - S_fc)+ (1 - exp(-dt / tau_d))           -> groundwater (internal)
    baseflow     = G (1 - exp(-dt / tau_g))                      -> runoff
    runoff       = surface excess + overflow + baseflow + glacier melt

Ledger per hex: d(S + SWE + G + firn + ice_we) = P - AET - runoff, residual reported.
Glacier: snow surviving the year on ice becomes firn; firn turns into ice with a time scale
(firnification is an internal transfer); SMB = accumulation - ablation; the geometry of the
ice is not changed (``geometry_pending``: the v5.3 flow solver acts on a slower step).
Routing: monthly runoff volumes accumulate down ``flow_receiver`` (the existing D6 graph,
sparse triangular solve), closed lakes integrate (P - E) A + inflow.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy import sparse
from scipy.sparse import linalg as sla
from scipy.special import ndtr

SIGMA_T = 4.0
T_SNOW = 1.0
T_MELT = 0.0
DDF_SNOW = 3.0  # mm w.e. per degC day [K]
DDF_ICE = 6.0  # mm w.e. per degC day [K]
TAU_DRAIN_MONTHS = 0.5
TAU_GW_MONTHS = 12.0
FAO_P = 0.5
FROZEN_FACTOR = 0.3
FIRN_TAU_YEARS = 5.0
RHO_ICE = 917.0


def expected_pdd(t_mean, days, sigma=SIGMA_T, t_melt=T_MELT):
    """Positive degree days of a month with Gaussian daily temperatures (exact E[max])."""
    mu = np.asarray(t_mean, dtype=np.float64) - t_melt
    if sigma <= 0:
        return np.maximum(mu, 0.0) * days
    z = mu / sigma
    phi = np.exp(-0.5 * z * z) / math.sqrt(2.0 * math.pi)
    return (sigma * phi + mu * ndtr(z)) * days


def pdd_daily(t_days, t_melt=T_MELT):
    """PDD of a daily series (N17)."""
    return float(np.sum(np.maximum(np.asarray(t_days, dtype=np.float64) - t_melt, 0.0)))


def degree_day_melt(swe, ice_we, pdd, ddf_snow=DDF_SNOW, ddf_ice=DDF_ICE):
    """Snow first, then ice with the PDD left (N18): returns (snow melt, ice melt, PDD left)."""
    snow_melt = np.minimum(swe, ddf_snow * pdd)
    pdd_left = pdd - snow_melt / ddf_snow
    ice_melt = np.minimum(ice_we, ddf_ice * np.maximum(pdd_left, 0.0))
    return snow_melt, ice_melt, pdd_left - ice_melt / ddf_ice


def baseflow(g_mm, dt, tau):
    """Linear-reservoir outflow over ``dt`` (same unit as ``tau``): G (1 - exp(-dt/tau)) (N14)."""
    return np.asarray(g_mm, dtype=np.float64) * (1.0 - math.exp(-dt / tau))


@dataclass
class SoilParams:
    s_wp: np.ndarray  # mm in the root zone at wilting point
    s_fc: np.ndarray
    s_sat: np.ndarray
    infil_mm_day: np.ndarray

    @property
    def taw(self):
        return self.s_fc - self.s_wp


def soil_from_layers(fc, wp, por, awc_mm, icap):
    """Root-zone stores (mm) from the v4.5 soil layers; no soil -> zero capacity."""
    fc = np.nan_to_num(np.asarray(fc, dtype=np.float64), nan=0.27)
    wp = np.nan_to_num(np.asarray(wp, dtype=np.float64), nan=0.13)
    por = np.nan_to_num(np.asarray(por, dtype=np.float64), nan=0.45)
    awc = np.maximum(np.nan_to_num(np.asarray(awc_mm, dtype=np.float64)), 0.0)
    z_root = np.where(fc - wp > 1e-3, awc / np.maximum((fc - wp) * 1000.0, 1e-9), 0.0)  # m
    icap = np.maximum(np.nan_to_num(np.asarray(icap, dtype=np.float64), nan=5.0), 0.0)
    return SoilParams(
        1000.0 * wp * z_root, 1000.0 * fc * z_root, 1000.0 * np.maximum(por, fc) * z_root, icap
    )


@dataclass
class State:
    soil: np.ndarray  # mm root-zone water (total, including below wilting point)
    swe: np.ndarray  # mm seasonal snow
    gw: np.ndarray  # mm groundwater
    firn: np.ndarray  # mm w.e. on glaciers
    ice: np.ndarray  # mm w.e. glacier ice

    def total(self):
        return self.soil + self.swe + self.gw + self.firn + self.ice

    def copy(self):
        return State(*(a.copy() for a in (self.soil, self.swe, self.gw, self.firn, self.ice)))


def initial_state(soil: SoilParams, ice_we):
    return State(
        soil.s_fc.copy(),
        np.zeros_like(soil.s_fc),
        np.zeros_like(soil.s_fc),
        np.zeros_like(soil.s_fc),
        np.asarray(ice_we, dtype=np.float64).copy(),
    )


def run_year(state: State, p_m, t_m, pet_m, days, soil: SoilParams, glacier) -> tuple:
    """One year, 12 monthly steps; returns (new state, monthly fluxes dict, ledger)."""
    s = state.copy()
    n = s.soil.size
    out = {
        k: np.zeros((12, n))
        for k in (
            "aet",
            "runoff",
            "recharge",
            "baseflow",
            "snow_melt",
            "ice_melt",
            "snowfall",
            "rain",
            "soil",
            "swe",
            "surface_runoff",
        )
    }
    start_total = s.total().copy()
    p_tot = np.zeros(n)
    acc_on_glacier = np.zeros(n)
    abl_on_glacier = np.zeros(n)
    for m in range(12):
        d = float(days[m])
        p = np.maximum(np.asarray(p_m[m], dtype=np.float64), 0.0)
        t = np.asarray(t_m[m], dtype=np.float64)
        pet = np.maximum(np.asarray(pet_m[m], dtype=np.float64), 0.0)
        f_snow = ndtr((T_SNOW - t) / SIGMA_T)
        snow = p * f_snow
        rain = p - snow
        s.swe += snow
        pdd = expected_pdd(t, d)
        on_ice = glacier & ((s.ice + s.firn) > 0)
        snow_melt = np.minimum(s.swe, DDF_SNOW * pdd)
        s.swe -= snow_melt
        pdd_left = np.maximum(pdd - snow_melt / DDF_SNOW, 0.0)
        firn_melt = np.where(on_ice, np.minimum(s.firn, DDF_ICE * pdd_left), 0.0)
        s.firn -= firn_melt
        pdd_left = np.maximum(pdd_left - firn_melt / DDF_ICE, 0.0)
        ice_melt = np.where(on_ice, np.minimum(s.ice, DDF_ICE * pdd_left), 0.0)
        s.ice -= ice_melt
        glacier_melt = firn_melt + ice_melt
        acc_on_glacier += np.where(glacier, snow, 0.0)
        abl_on_glacier += np.where(glacier, snow_melt + glacier_melt, 0.0)
        liquid = rain + snow_melt
        frozen = t < -2.0
        cap = soil.infil_mm_day * d * np.where(frozen, FROZEN_FACTOR, 1.0)
        room = np.maximum(soil.s_sat - s.soil, 0.0)
        infil = np.minimum(np.minimum(liquid, cap), room)
        surface = liquid - infil
        s.soil += infil
        avail = np.maximum(s.soil - soil.s_wp, 0.0)
        raw = (1.0 - FAO_P) * soil.taw
        ks = np.where(avail >= raw, 1.0, avail / np.maximum(raw, 1e-9))
        ks = np.where(soil.taw > 0, ks, 0.0)
        aet = np.minimum(pet * ks, avail)
        s.soil -= aet
        drain = np.maximum(s.soil - soil.s_fc, 0.0) * (1.0 - math.exp(-1.0 / TAU_DRAIN_MONTHS))
        s.soil -= drain
        overflow = np.maximum(s.soil - soil.s_sat, 0.0)
        s.soil -= overflow
        s.gw += drain
        base = baseflow(s.gw, 1.0, TAU_GW_MONTHS)
        s.gw -= base
        runoff = surface + overflow + base + glacier_melt
        p_tot += p
        for k, v in (
            ("aet", aet),
            ("runoff", runoff),
            ("recharge", drain),
            ("baseflow", base),
            ("snow_melt", snow_melt),
            ("ice_melt", glacier_melt),
            ("snowfall", snow),
            ("rain", rain),
            ("soil", s.soil),
            ("swe", s.swe),
            ("surface_runoff", surface + overflow),
        ):
            out[k][m] = v
    # end of the year: snow surviving on a glacier becomes firn, firn slowly becomes ice
    to_firn = np.where(glacier, s.swe, 0.0)
    s.swe -= to_firn
    s.firn += to_firn
    to_ice = s.firn * (1.0 - math.exp(-1.0 / FIRN_TAU_YEARS))
    s.firn -= to_ice
    s.ice += to_ice
    resid = (s.total() - start_total) - (p_tot - out["aet"].sum(axis=0) - out["runoff"].sum(axis=0))
    ledger = {
        "precip_mm": p_tot,
        "aet_mm": out["aet"].sum(axis=0),
        "runoff_mm": out["runoff"].sum(axis=0),
        "storage_change_mm": s.total() - start_total,
        "residual_mm": resid,
        "smb_mm_we": np.where(glacier, acc_on_glacier - abl_on_glacier, np.nan),
    }
    return s, out, ledger


def routing_operator(receiver):
    """``I - R`` of the drainage DAG (``R[j, i] = 1`` when ``j`` receives from ``i``)."""
    rc = np.asarray(receiver, dtype=np.int64).ravel()
    n = rc.size
    i = np.nonzero((rc >= 0) & (rc < n) & (rc != np.arange(n)))[0]
    r = sparse.csr_matrix((np.ones(i.size), (rc[i], i)), shape=(n, n))
    return sla.splu((sparse.identity(n, format="csc") - r).tocsc())


def route(lu, volumes):
    """Accumulated volumes (each column a period) down the DAG: solves (I - R) x = v."""
    v = np.asarray(volumes, dtype=np.float64)
    if v.ndim == 2:
        return lu.solve(np.ascontiguousarray(v.T)).T
    return lu.solve(v)


def lake_balance(v0, p_m, e_m, area_m2, inflow_m3):
    """Closed lake: V' = V + P A - E A + inflow (constant-area fixture, N16)."""
    return v0 + (p_m - e_m) * area_m2 + inflow_m3
