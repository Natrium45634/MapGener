"""Snow - firn - glacier ice: three mass reservoirs with memory, and a light ice flow (TZ 5.0, 10).

Reservoirs per dry terrain unit, all in **mm water equivalent on the plan area of the unit**:

* ``snow`` - this year's snow (seasonal pack);
* ``firn`` - snow that survived at least one melt season, with a mass-weighted age (years) and
  a density from the age (``rho = rho_ice - (rho_ice - rho_snow) exp(-age / tau(T))`` [А]);
  firn older than the pore close-off age becomes ice (see :func:`year_end`);
* ``ice`` - glacier ice; thickness ``H = ice / 1000 * rho_w / rho_ice`` (m).

Each month (the year starts after the locally warmest month, so a melt season is never cut
in two): snowfall by a smooth air-temperature phase, sublimation (a share of PET in freezing
months), degree-day melt taken from snow, then firn, then ice (``k_ice > k_snow``); melt and
rain leave as runoff. At the year end the surviving snow becomes firn; firn whose density
reached the close-off density turns into ice. **Every transfer moves mass, none creates it**:
``M_end = M_start + snowfall - melt - sublimation + flux_in - flux_out`` per unit and per
hex (``mass_audit``).

Ice flow [А, regional_lite]: the ice of a hex (area-weighted over its dry units) flows on the
hex graph by the shallow-ice diffusion ``dH/dt = b/rho_i - div q``, ``q = -D grad s``,
``D = Gamma H^5 |grad s|^2`` (Glen n = 3, ``Gamma = 2A(rho g)^3 / 5``), explicit steps with
a capped ``D`` (stability; flagged), double-buffered (order free). Mass leaving one hex enters
its neighbour exactly. A short spin-up cannot grow a kilometre-thick ice sheet; a cold, snowy
region may start from a documented **seeded paleoclimate ice** (``ice_initial_condition``),
which then follows the present balance (it shrinks where the balance is negative).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..stages.environment import MONTH_DAYS, positive_degree_days

RHO_W = 1000.0
RHO_ICE = 917.0
RHO_SNOW = 350.0
RHO_CLOSE = 830.0
SECONDS_YR = 365.25 * 86400.0


@dataclass(frozen=True)
class CryoParams:
    spinup_years: int = 100
    snow_t_c: float = -1.0
    rain_t_c: float = 2.0
    k_snow_mm: float = 3.5  # mm w.e. per degC day
    k_ice_mm: float = 7.0
    daily_sd_c: float = 4.0
    sublimation_share: float = 0.12  # of PET in freezing months
    firn_tau_yr: float = 15.0  # densification time scale at 0 degC [П]
    paleo_ice: bool = True
    paleo_t_c: float = -9.0  # mean annual T below which a seeded paleo ice may exist [П]
    paleo_h_m: float = 1800.0
    glen_a_pa3_yr: float = 1.0e-16  # Glen A (Pa^-3 a^-1), about -10 degC ice
    flow: bool = True
    flow_every_years: int = 1


def _phase(t):
    x = np.clip((t - (-1.0)) / 3.0, 0.0, 1.0)
    return 1.0 - x * x * (3.0 - 2.0 * x)  # snow share


def ice_thickness_m(ice_swe_mm):
    return np.asarray(ice_swe_mm, dtype=np.float64) / 1000.0 * RHO_W / RHO_ICE


def firn_density(age_yr, t_mean_c, tau0=15.0):
    tau = np.clip(tau0 * np.exp(-0.05 * np.asarray(t_mean_c, dtype=np.float64)), 3.0, 200.0)
    return RHO_ICE - (RHO_ICE - RHO_SNOW) * np.exp(-np.asarray(age_yr, dtype=np.float64) / tau)


def month_step(state, p, t, pet, days, cp: CryoParams):
    """One month for every unit; returns the fluxes (mm w.e.). Mass is conserved exactly."""
    snow, firn, ice = state["snow"], state["firn"], state["ice"]
    snowfall = p * _phase(t)
    rain = p - snowfall
    snow = snow + snowfall
    sub = np.where(t < 0.0, cp.sublimation_share * pet, 0.0)
    sub = np.minimum(sub, snow)
    snow = snow - sub
    pdd = positive_degree_days(t, cp.daily_sd_c, days)
    pot = cp.k_snow_mm * pdd
    m_snow = np.minimum(snow, pot)
    snow = snow - m_snow
    left = np.where(pot > 0, 1.0 - m_snow / np.maximum(pot, 1e-12), 0.0)  # unused energy share
    m_firn = np.minimum(firn, left * pot)
    firn = firn - m_firn
    left2 = np.where(pot > 0, left - m_firn / np.maximum(pot, 1e-12), 0.0)
    m_ice = np.minimum(ice, np.clip(left2, 0, None) * pdd * cp.k_ice_mm)
    ice = ice - m_ice
    state["snow"], state["firn"], state["ice"] = snow, firn, ice
    melt = m_snow + m_firn + m_ice
    return {"snowfall": snowfall, "rain": rain, "sublimation": sub, "melt": melt,
            "ice_melt": m_ice, "runoff": rain + melt}  # fmt: skip


def close_off_age(t_mean_c, tau0=15.0):
    """Firn age (years) at which the density reaches the pore close-off ``RHO_CLOSE``."""
    tau = np.clip(tau0 * np.exp(-0.05 * np.asarray(t_mean_c, dtype=np.float64)), 3.0, 200.0)
    return tau * np.log((RHO_ICE - RHO_SNOW) / (RHO_ICE - RHO_CLOSE))


def year_end(state, t_mean, cp: CryoParams):
    """Surviving snow -> firn (age 0), firn ages, its oldest part becomes ice.

    The firn column is one reservoir with a mass-weighted mean age; under steady
    accumulation its mass is spread evenly over ages ``0 .. 2 * mean`` [А]. The share older
    than the close-off age (density ``RHO_CLOSE``) turns into glacier ice each year, and the
    remaining firn keeps the mean age of the younger part. Mass moves, none is created.
    """
    snow, firn, age = state["snow"], state["firn"], state["firn_age"]
    total = firn + snow
    age = np.where(total > 0, firn * (age + 1.0) / np.maximum(total, 1e-12), 0.0)
    firn = total
    t_c = close_off_age(t_mean, cp.firn_tau_yr)
    span = 2.0 * age
    share = np.where(span > t_c, 1.0 - t_c / np.maximum(span, 1e-12), 0.0)
    to_ice = firn * np.clip(share, 0.0, 1.0)
    firn = firn - to_ice
    age = np.where(to_ice > 0, np.minimum(age, 0.5 * t_c), age)
    state["ice"] = state["ice"] + to_ice
    state["snow"] = np.zeros_like(snow)
    state["firn"], state["firn_age"] = firn, age
    return to_ice


def flow_step(h_ice_m, z_bed_m, nbr, spacing_km, glen_a, dt_yr):
    """Shallow-ice diffusion of hex ice thickness on the hex graph; returns the new H (m).

    Fluxes are computed for every edge once (antisymmetric), so what leaves a hex enters its
    neighbour: the total volume is conserved exactly (no ice leaves the graph). ``D`` is
    capped for an explicit stable step; the cap is reported by the caller.
    """
    L = spacing_km * 1000.0
    gamma = 2.0 * glen_a * (RHO_ICE * 9.81) ** 3 / 5.0
    h = np.asarray(h_ice_m, dtype=np.float64)
    s = z_bed_m + h
    n = h.size
    dh = np.zeros(n)
    d_cap = 0.08 * L * L / max(dt_yr, 1e-9)
    capped = 0
    for k in range(3):  # each undirected edge once: directions 0..2 of the six
        j = nbr[:, k]
        ok = (j >= 0) & ((h > 0) | (h[np.maximum(j, 0)] > 0))
        i = np.nonzero(ok)[0]
        jj = j[i]
        grad = (s[jj] - s[i]) / L
        hm = 0.5 * (h[i] + h[jj])
        d = gamma * hm**5 * grad * grad
        capped += int((d > d_cap).sum())
        d = np.minimum(d, d_cap)
        # flux from the higher surface to the lower one (volume per unit area of a hex)
        q = d * grad / L * dt_yr * (2.0 / 3.0)
        # never take more ice than the donor has (a margin hex can empty, not go negative)
        donor_h = np.where(q > 0, h[jj], h[i])
        q = np.sign(q) * np.minimum(np.abs(q), 0.25 * donor_h)
        np.add.at(dh, i, q)
        np.add.at(dh, jj, -q)
    return np.maximum(h + dh, 0.0), capped


@dataclass
class CryoResult:
    unit: dict  # per unit arrays
    hex: dict  # per hex arrays
    summary: dict


def run_cryosphere(units_hex, weights, dry, t_m, p_m, pet_m, z_hex, nbr, spacing_km,
                   n_hex, cp: CryoParams, seed_ice_hex=None):  # fmt: skip
    """Spin the snow / firn / ice reservoirs up on the dry units of every hex.

    ``t_m, p_m, pet_m``: (12, U). ``seed_ice_hex`` (n_hex,) optional initial ice thickness (m)
    of the paleo scenario. Returns per-unit and per-hex results with a mass audit.
    """
    U = units_hex.size
    w = np.where(dry, weights, 0.0)
    hex_w = np.bincount(units_hex, weights=w, minlength=n_hex)
    state = {"snow": np.zeros(U), "firn": np.zeros(U), "firn_age": np.zeros(U), "ice": np.zeros(U)}
    seeded_hex = np.zeros(n_hex, dtype=bool)
    if seed_ice_hex is not None:
        seed_mm = np.asarray(seed_ice_hex, dtype=np.float64)[units_hex] * RHO_ICE / RHO_W * 1000.0
        state["ice"] = np.where(dry, seed_mm, 0.0)
        seeded_hex = (np.asarray(seed_ice_hex) > 0) & (hex_w > 0)
    mass0 = _total(state)
    t_mean = t_m.mean(axis=0)
    start = (np.argmax(t_m, axis=0) + 1) % 12  # the year starts after the warmest month
    order = (start[None, :] + np.arange(12)[:, None]) % 12  # (12, U) month index
    cols = np.arange(U)
    acc = {
        k: np.zeros(U) for k in ("snowfall", "rain", "sublimation", "melt", "ice_melt", "runoff")
    }
    flux_net = np.zeros(U)
    capped_total = 0
    last_year = {}
    snow_monthly_last = np.zeros((12, U))
    for year in range(cp.spinup_years):
        year_acc = {k: np.zeros(U) for k in acc}
        for step in range(12):
            mi = order[step]
            fl = month_step(state, p_m[mi, cols], t_m[mi, cols], pet_m[mi, cols],
                            MONTH_DAYS[mi], cp)  # fmt: skip
            for k in year_acc:
                year_acc[k] += fl[k]
            if year == cp.spinup_years - 1:
                snow_monthly_last[mi, cols] = state["snow"] + state["firn"] + state["ice"]
        to_ice = year_end(state, t_mean, cp)
        del to_ice
        if cp.flow and (year % cp.flow_every_years == 0):
            # hex mean ice (m) -> flow -> back to the units in proportion (mass exact)
            ice_mm_hex = np.bincount(units_hex, weights=w * state["ice"], minlength=n_hex)
            h_hex = np.where(hex_w > 0, ice_thickness_m(ice_mm_hex / np.maximum(hex_w, 1e-12)), 0.0)
            if (h_hex > 0).any():
                h_new, capped = flow_step(h_hex, z_hex, nbr, spacing_km, cp.glen_a_pa3_yr,
                                          float(cp.flow_every_years))  # fmt: skip
                # ice cannot flow into a hex without dry land: keep it where it was
                h_new = np.where(hex_w > 0, h_new, 0.0)
                lost = (h_hex - h_new)[hex_w <= 0].sum() if (hex_w <= 0).any() else 0.0
                del lost
                capped_total += capped
                d_mm = (h_new - h_hex) * RHO_ICE / RHO_W * 1000.0  # per unit area of the hex
                old_hex_mm = np.where(hex_w > 0, ice_mm_hex / np.maximum(hex_w, 1e-12), 0.0)
                gain = np.where(dry, d_mm[units_hex], 0.0)
                # removal in proportion to each unit's ice; addition spread evenly over dry units
                ratio = np.where(old_hex_mm > 0, d_mm / np.maximum(old_hex_mm, 1e-12), 0.0)
                take = np.where(d_mm[units_hex] < 0, state["ice"] * ratio[units_hex], 0.0)
                add = np.where(d_mm[units_hex] > 0, gain, 0.0)
                delta = np.where(dry, take + add, 0.0)
                state["ice"] = np.maximum(state["ice"] + delta, 0.0)
                flux_net += delta
        for k in acc:
            acc[k] += year_acc[k]
        last_year = year_acc
    mass1 = _total(state)
    years = max(cp.spinup_years, 1)
    resid = mass1 - (mass0 + acc["snowfall"] - acc["melt"] - acc["sublimation"] + flux_net)
    ice_m = ice_thickness_m(state["ice"])
    rho = firn_density(state["firn_age"], t_mean, cp.firn_tau_yr)
    balance = last_year.get("snowfall", np.zeros(U)) - last_year.get("melt", np.zeros(U)) - \
        last_year.get("sublimation", np.zeros(U))  # fmt: skip

    def hex_mean(v):
        tot = np.bincount(units_hex, weights=w * v, minlength=n_hex)
        return np.where(hex_w > 0, tot / np.maximum(hex_w, 1e-12), 0.0)

    unit = {
        "snow_max_mm": snow_monthly_last.max(axis=0),
        "snow_months": (snow_monthly_last >= 5.0).sum(axis=0).astype(np.int8),
        "firn_swe_mm": state["firn"],
        "firn_age_yr": state["firn_age"],
        "firn_density_kg_m3": np.where(state["firn"] > 0, rho, np.nan),
        "ice_swe_mm": state["ice"],
        "ice_thickness_m": ice_m,
        "annual_balance_mm": balance,
        "annual_melt_mm": last_year.get("melt", np.zeros(U)),
        "annual_sublimation_mm": last_year.get("sublimation", np.zeros(U)),
        "annual_snowfall_mm": last_year.get("snowfall", np.zeros(U)),
        "ice_flux_net_mm_yr": flux_net / years,
    }
    hx = {
        "ice_thickness_m": hex_mean(ice_m),
        "firn_swe_mm": hex_mean(state["firn"]),
        "annual_balance_mm": hex_mean(balance),
        "seeded_paleo_ice": seeded_hex,
    }
    total_snowfall = float((w * acc["snowfall"]).sum())
    summary = {
        "spinup_years": int(cp.spinup_years),
        "mass_residual_mm_max": float(np.abs(resid).max()) if U else 0.0,
        "flow_d_capped_edges": int(capped_total),
        "ice_hexes": int((hx["ice_thickness_m"] > 10.0).sum()),
        "max_ice_thickness_m": float(hx["ice_thickness_m"].max()) if n_hex else 0.0,
        "seeded_paleo_ice_hexes": int(seeded_hex.sum()),
        "total_snowfall_area_mm": total_snowfall,
    }
    return CryoResult(unit=unit, hex=hx, summary=summary)


def _total(state):
    return state["snow"] + state["firn"] + state["ice"]


def paleo_seed(t_annual_hex, snowfall_hex_mm, land, cp: CryoParams):
    """Seeded paleoclimate ice thickness (m) of cold, snowy land [А, documented scenario].

    Only where the present mean annual T is below ``paleo_t_c`` **and** snow falls
    (> 50 mm/yr): latitude alone or a dry cold desert does not seed ice.
    """
    if not cp.paleo_ice:
        return np.zeros_like(t_annual_hex)
    cold = np.clip((cp.paleo_t_c - t_annual_hex) / 10.0, 0.0, 1.0)
    wet = np.clip((snowfall_hex_mm - 50.0) / 250.0, 0.0, 1.0)
    return np.where(land, cp.paleo_h_m * np.sqrt(cold) * wet, 0.0)
