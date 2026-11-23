"""Eruption styles from the magma and the setting, and the partition of the erupted mass.

``[H]`` explosive tendency::

    E = sigmoid(c0 + c1 log10(eta) + c2 retained_gas + c3 ascent + c4 water - c5 permeability)

The weights are configuration, not constants of nature. A random choice of the style is
made only *after* the conditions (one deterministic draw per event, from its id).

Mass basis (one basis, no double counting): the erupted magma mass ``M`` (DRE x density)
holds its dissolved gas; ``M = M_lava + M_tephra + M_gas``. Bulk tephra volume
(``M_tephra / rho_bulk``) and DRE volume are kept apart.
"""

from __future__ import annotations

import math

from .units import (
    RHO_TEPHRA_BULK,
    SO2_PER_S,
    plume_height_m,
    stable_uniform,
    tropopause_m,
    vei_from_bulk,
)

COEF = {"c0": -4.4, "c1": 0.62, "c2": 0.55, "c3": 0.35, "c4": 1.4, "c5": 1.6}
#: Mass eruption rate of the explosive phase by style, kg/s (order of magnitude) [H].
MER = {
    "strombolian": 3e3,
    "vulcanian": 1e5,
    "subplinian": 3e6,
    "plinian": 3e7,
    "ultraplinian": 3e8,
    "phreatomagmatic_surtseyan": 1e5,
    "phreatic": 1e4,
    "dome_growth_collapse": 3e4,
}
#: Share of the non-gas mass that leaves as tephra, by style.
TEPHRA_SHARE = {
    "hawaiian_effusive": 0.03,
    "fissure_effusive": 0.02,
    "submarine_effusive": 0.0,
    "strombolian": 0.35,
    "vulcanian": 0.7,
    "dome_growth_collapse": 0.45,
    "subplinian": 0.85,
    "plinian": 0.95,
    "ultraplinian": 0.98,
    "phreatomagmatic_surtseyan": 0.8,
    "phreatic": 1.0,
}
PDC_STYLES = ("subplinian", "plinian", "ultraplinian", "dome_growth_collapse", "vulcanian")


def explosive_tendency(log_eta, h2o_wt, ascent, water, permeability) -> float:
    retained = h2o_wt * min(1.0, max(0.0, (log_eta - 1.0) / 6.0))
    z = (
        COEF["c0"]
        + COEF["c1"] * log_eta
        + COEF["c2"] * retained
        + COEF["c3"] * ascent
        + COEF["c4"] * water
        - COEF["c5"] * permeability
    )
    return 1.0 / (1.0 + math.exp(-z))


def choose_style(system, dre_m3: float, event_key: str) -> tuple[list[str], dict]:
    """Style phases of one eruption and the conditions that chose them."""
    m = system.magma
    log_eta = m.log10_viscosity_pa_s
    env = system.environment
    open_conduit = system.edifice_type in ("shield", "fissure_field") or log_eta < 3.0
    water = 0.0
    if env == "shallow_submarine":
        water = 1.0 if system.depth_m < 200 else 0.5
    elif env == "subglacial":
        water = 0.8
    elif env == "island_emergent":
        water = 0.3
    ascent = min(2.0, max(0.0, math.log10(max(dre_m3, 1.0)) - 6.0))
    e = explosive_tendency(log_eta, m.volatile_h2o_wt_pct, ascent, water,
                           1.0 if open_conduit else 0.0)  # fmt: skip
    u = stable_uniform(event_key, "style")
    cond = {"explosive_tendency": round(e, 3), "log10_viscosity": round(log_eta, 2),
            "water_interaction": water, "open_conduit": open_conduit}  # fmt: skip
    if env == "deep_submarine":
        # the water column suppresses fragmentation: pillow lavas, no air column
        return ["submarine_effusive"], cond
    if system.edifice_type == "fissure_field" and system.source_type in (
        "mid_ocean_ridge", "divergent_rift", "continental_hotspot"
    ):  # fmt: skip
        phases = ["fissure_effusive"]
        if water > 0.4:
            phases.insert(0, "phreatomagmatic_surtseyan")
        return phases, cond
    if water >= 0.8 and log_eta < 6.0:
        return ["phreatomagmatic_surtseyan", "strombolian"], cond
    if system.edifice_type == "scoria_cone":
        return (["strombolian", "hawaiian_effusive"] if u < 0.7 else ["strombolian"]), cond
    if e < 0.12:
        return ["hawaiian_effusive"], cond
    if e < 0.4:
        return (["strombolian", "hawaiian_effusive"] if u < 0.6 else ["strombolian"]), cond
    if log_eta > 7.5 and e < 0.75:
        return (
            ["dome_growth_collapse"] if u < 0.6 else ["vulcanian", "dome_growth_collapse"]
        ), cond
    if e < 0.6:
        return ["vulcanian"], cond
    big = dre_m3 >= 1e9
    if e < 0.8 or not big:
        return (["subplinian", "vulcanian"] if u < 0.5 else ["subplinian"]), cond
    if dre_m3 >= 1e11 and system.supereruption_eligible:
        return ["ultraplinian", "plinian"], cond
    return ["plinian", "dome_growth_collapse"] if u < 0.4 else ["plinian"], cond


def partition(system, dre_m3: float, phases: list[str], lat_deg: float) -> dict:
    """Masses, volumes, durations, column height and gases of one eruption."""
    m = system.magma
    mass = dre_m3 * m.density_kg_m3
    degas = 0.95 if any(p not in ("submarine_effusive",) for p in phases) else 0.6
    volatiles = (m.volatile_h2o_wt_pct + m.co2_wt_pct + m.sulfur_wt_pct) / 100.0
    gas = mass * volatiles * degas
    rest = mass - gas
    share = sum(TEPHRA_SHARE.get(p, 0.3) for p in phases) / len(phases)
    tephra = rest * share
    lava = rest - tephra
    so2 = mass * m.sulfur_wt_pct / 100.0 * degas * SO2_PER_S
    explosive = [p for p in phases if p in MER]
    mer = max((MER[p] for p in explosive), default=0.0) * (dre_m3 / 1e8) ** 0.25 if explosive else 0
    t_explosive = tephra / mer if mer > 0 else 0.0
    effusion = 30.0 * (max(lava, 1.0) / m.density_kg_m3 / 1e7) ** 0.5  # m^3/s
    t_effusive = (lava / m.density_kg_m3) / effusion if lava > 0 else 0.0
    dre_rate = (mer / m.density_kg_m3) if mer > 0 else 0.0
    column = plume_height_m(dre_rate)
    if system.environment == "deep_submarine":
        column = 0.0
    bulk = tephra / RHO_TEPHRA_BULK
    strat = column > tropopause_m(lat_deg)
    return {
        "magma_mass_kg": mass,
        "lava_mass_kg": lava,
        "tephra_mass_kg": tephra,
        "gas_mass_kg": gas,
        "so2_mass_kg": so2,
        "so2_stratospheric_kg": so2 if strat else 0.0,
        "volume_dre_m3": dre_m3,
        "lava_volume_m3": lava / m.density_kg_m3,
        "tephra_bulk_volume_m3": bulk,
        "tephra_dre_m3": tephra / m.density_kg_m3,
        "duration_s": t_explosive + t_effusive,
        "mass_eruption_rate_kg_s": mer,
        "plume_height_m": column,
        "vei": vei_from_bulk(bulk) if explosive else 0,
        "pdc": any(p in PDC_STYLES for p in phases) and column > 0,
    }
