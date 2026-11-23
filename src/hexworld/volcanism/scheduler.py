"""Eruptions of the recent window (event time), deterministic and reservoir-limited.

Intensity ``[H]``::

    lambda = lambda_0(edifice) * f_supply * f_status * eruption_rate_scale * f_p(P)

``f_p = clip(P, 0.05, 3)^2`` couples the rate to the reservoir fill ``P = M / M_crit``
(piecewise constant between events, evaluated after each event): an over-filled reservoir
erupts more often, a drained one rests. Without the coupling a strong supply piles melt
up without bound.

Times come from a fixed sequence of thresholds per event *index* k
(``u_k = uniform(seed, volcano, k)``, interval ``-ln(u_k) / lambda``) - never from a
generator that is advanced by the time step - so splitting the window into parts gives
the same events with the same ids. The volume of event k is a Pareto draw capped by the
edifice and by the melt available in the reservoir: no lava without melt.

Events above a size / recency threshold are *explicit* (full products: lava flow, ash,
gas, PDC, lahar); the rest are aggregated per volcano with conserved totals.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from . import ALGORITHM_VERSION
from .styles import choose_style, partition
from .units import SECONDS_PER_YEAR, stable_hash, stable_uniform

#: Eruptions per year of an active system by edifice [H] (order of magnitude).
BASE_RATE = {
    "shield": 1 / 25.0,
    "submarine_seamount": 1 / 250.0,
    "stratocone": 1 / 40.0,
    "lava_dome_complex": 1 / 80.0,
    "caldera_complex": 1 / 1500.0,
    "scoria_cone": 1 / 500.0,
    "fissure_field": 1 / 150.0,
    "subglacial_tuya": 1 / 400.0,
    "maar_tuff_ring": 1 / 3000.0,
    "mixed": 1 / 300.0,
}
#: (smallest, largest) DRE volume of an eruption by edifice, m^3, and the Pareto exponent.
VOLUME = {
    "shield": (1e6, 4e9, 0.75),
    "submarine_seamount": (1e5, 1e9, 0.8),
    "stratocone": (1e5, 2e10, 0.8),
    "lava_dome_complex": (1e5, 8e9, 0.8),
    "caldera_complex": (1e6, 5e10, 0.7),
    "scoria_cone": (1e5, 1e8, 0.9),
    "fissure_field": (1e6, 2e10, 0.75),
    "subglacial_tuya": (1e5, 3e9, 0.8),
    "maar_tuff_ring": (1e5, 3e7, 1.0),
    "mixed": (1e5, 1e10, 0.8),
}
RIDGE_VOLUME = (1e6, 5e8, 0.8)
#: Share of the reservoir melt one eruption can take.
ERUPTIBLE_SHARE = 0.5
#: Reservoir fill above which a resting system counts as in unrest.
UNREST_P = 1.5
#: Fill above which an eruption also taps the melt in excess of the critical mass.
RELIEF_P = 1.3


@dataclass
class EruptionEvent:
    event_id: str
    volcano_id: str
    province_id: str
    sequence: int
    years_bp: float  # start, years before present
    duration_s: float
    vent_ids: tuple
    vent_x_km: float
    vent_y_km: float
    style_phases: tuple
    conditions: dict
    environment: str
    products: dict
    explicit: bool = False
    quality_flags: tuple = ()
    results: dict = field(default_factory=dict)  # filled by the product solvers

    @property
    def vei(self):
        return self.products.get("vei")

    def to_dict(self) -> dict:
        return {
            "event_id": self.event_id, "volcano_id": self.volcano_id,
            "province_id": self.province_id, "sequence": self.sequence,
            "years_bp": round(self.years_bp, 3), "duration_s": round(self.duration_s, 1),
            "vent_ids": list(self.vent_ids), "vent_x_km": round(self.vent_x_km, 3),
            "vent_y_km": round(self.vent_y_km, 3), "style_phases": list(self.style_phases),
            "conditions": self.conditions, "environment": self.environment,
            "products": {k: (float(v) if isinstance(v, float) else v)
                         for k, v in self.products.items()},
            "vei": self.vei, "explicit": self.explicit,
            "quality_flags": list(self.quality_flags), "results": self.results,
        }  # fmt: skip


def intensity(system, params) -> float:
    """Eruptions per year of the system in the recent window."""
    base = BASE_RATE.get(system.edifice_type, 1 / 300.0)
    res = system.reservoir
    supply = res.input_kg_s if res else 0.0
    f_supply = min(3.0, max(0.2, (supply / 500.0) ** 0.3)) if supply > 0 else 0.0
    st = system.magmatic_status
    if st == "active_system":
        f_status = 1.0
    elif st == "dormant_system":
        # probability of >= 1 event in the window = reactivation probability
        p = min(0.999, max(0.0, system.reactivation_probability))
        return -math.log(1.0 - p) / max(params.event_window_years, 1.0) * params.eruption_rate_scale
    elif st == "uncertain":
        f_status = 0.02
    else:
        return 0.0
    return base * f_supply * f_status * params.eruption_rate_scale


def pressure_factor(p: float) -> float:
    """Rate multiplier from the reservoir fill (1 at the critical melt mass)."""
    return min(3.0, max(0.05, p)) ** 2


def schedule(system, seed: str, params, lat_deg: float = 30.0) -> list[EruptionEvent]:
    """Every eruption of ``system`` in the window, oldest first (reservoir updated)."""
    lam = intensity(system, params)
    res = system.reservoir
    window = float(params.event_window_years)
    out: list[EruptionEvent] = []
    if lam <= 0 or res is None:
        if res is not None:
            res.advance(window)
        return out
    vmin, vmax, b = VOLUME.get(system.edifice_type, (1e5, 1e10, 0.8))
    if system.source_type == "mid_ocean_ridge":  # ridge dike-fed sheet flows: 1e6-5e8 m^3
        vmin, vmax, b = RIDGE_VOLUME
    if system.supereruption_eligible:
        vmax = max(vmax, 1e12)
    rho = system.magma.density_kg_m3
    t = -window
    k = 0
    while k < 20000:
        u = stable_uniform(seed, system.volcano_id, k, "t")
        rate = lam * pressure_factor(res.pressure_proxy)
        dt = -math.log(u) / rate
        if t + dt > 0:
            break
        res.advance(dt)
        t += dt
        uv = stable_uniform(seed, system.volcano_id, k, "v")
        dre = min(vmin * uv ** (-1.0 / b), vmax)
        avail = res.melt_kg * ERUPTIBLE_SHARE / rho
        if avail < 0.1 * vmin:  # not enough melt: repose instead of an eruption
            k += 1
            continue
        if res.melt_kg > RELIEF_P * res.critical_kg:  # overpressure relief taps the excess
            excess = res.melt_kg - res.critical_kg
            dre = max(dre, min(vmax, 0.5 * excess / rho))
        dre = min(dre, avail)
        res.withdraw(dre * rho)
        key = stable_hash(seed, system.province_id, system.volcano_id, k, ALGORITHM_VERSION)
        phases, cond = choose_style(system, dre, key)
        prod = partition(system, dre, phases, lat_deg)
        vent = _vent(system, seed, k)
        out.append(EruptionEvent(
            event_id=f"E-{key[:12]}", volcano_id=system.volcano_id,
            province_id=system.province_id, sequence=k, years_bp=-t,
            duration_s=prod["duration_s"], vent_ids=(vent["vent_id"],),
            vent_x_km=vent["x_km"], vent_y_km=vent["y_km"], style_phases=tuple(phases),
            conditions=cond, environment=system.environment, products=prod,
            quality_flags=("approximated_source",),
        ))  # fmt: skip
        k += 1
    res.advance(-t)  # to the present
    _set_phase(system, out, seed, bool(getattr(params, "lava_lakes", True)))
    return out


def _vent(system, seed, k) -> dict:
    """Vent of event k: central, a fissure segment, or a (new) cone of a field."""
    vents = system.vents
    if len(vents) <= 1:
        return vents[0]
    i = int(stable_uniform(seed, system.volcano_id, k, "vent") * len(vents)) % len(vents)
    return vents[i]


SUBAERIAL = ("land", "island_emergent", "ice_free_high_altitude")


def _set_phase(system, events, seed, lakes: bool = True) -> None:
    """Activity phase now and the lava-lake state, from the latest events."""
    if not events:
        system.activity_phase = (
            "degassing" if system.magmatic_status == "active_system"
            and system.environment in SUBAERIAL
            and system.magma.log10_viscosity_pa_s < 3.5 else "quiet"
        )  # fmt: skip
        if system.reservoir and system.reservoir.pressure_proxy > UNREST_P:
            system.activity_phase = "unrest"
        return
    last = events[-1]
    system.last_eruption_years_bp = last.years_bp
    since_end_s = last.years_bp * SECONDS_PER_YEAR - last.duration_s
    if since_end_s < 0:
        system.activity_phase = "erupting"
    elif since_end_s < SECONDS_PER_YEAR:
        system.activity_phase = "waning"
    elif since_end_s < 10 * SECONDS_PER_YEAR:
        system.activity_phase = "recovering"
    elif system.reservoir and system.reservoir.pressure_proxy > UNREST_P:
        system.activity_phase = "unrest"
    elif (
        system.magma.log10_viscosity_pa_s < 3.5
        and system.magmatic_status == "active_system"
        and system.environment in SUBAERIAL
    ):
        system.activity_phase = "degassing"
    else:
        system.activity_phase = "quiet"
    # lava lakes: rare, only an open basaltic conduit with frequent supply
    open_vent = system.edifice_type in ("shield", "stratocone") and (
        system.magma.log10_viscosity_pa_s < 3.0
    )
    frequent = len(events) >= 20 and last.years_bp < 50
    if (
        lakes
        and open_vent
        and frequent
        and system.environment in ("land", "island_emergent", "ice_free_high_altitude")
    ):
        u = stable_uniform(seed, system.volcano_id, "lava_lake")
        if u < 0.25:
            system.lava_lake_state = "persistent_open_vent"
        elif u < 0.4:
            system.lava_lake_state = "crusted" if last.years_bp > 1 else "transient"
        elif u < 0.5:
            system.lava_lake_state = "drained"


def select_explicit(events: list[EruptionEvent], limit: int, in_window: set) -> None:
    """Mark the events that get full products.

    Score = size + recency + setting: the biggest, the most recent and the ones on land or
    in shallow water first. Deep-submarine effusions (most of the events by number) take
    at most a quarter of the explicit slots, so the hazards of the land are not crowded out.
    """

    def score(e):
        size = math.log10(max(e.products["volume_dre_m3"], 1.0))
        recency = 2.0 * math.exp(-e.years_bp / 500.0)
        near = 1.0 if e.volcano_id in in_window else 0.0
        explosive = 0.5 * (e.vei or 0)
        setting = -1.5 if e.environment == "deep_submarine" else 0.0
        return size + recency + near + explosive + setting

    limit = max(0, int(limit))
    ranked = sorted((e for e in events if e.products["volume_dre_m3"] >= 1e6),
                    key=lambda e: (-score(e), e.event_id))  # fmt: skip
    deep_cap = limit // 4
    keep, deep = [], 0
    for e in ranked:
        if len(keep) >= limit:
            break
        if e.environment == "deep_submarine":
            if deep >= deep_cap:
                continue
            deep += 1
        keep.append(e)
    ids = {e.event_id for e in keep}
    for e in events:
        e.explicit = e.event_id in ids


def aggregate(events: list[EruptionEvent]) -> dict:
    """Per-volcano totals of the non-explicit events (conserved masses, age span)."""
    out: dict = {}
    for e in events:
        if e.explicit:
            continue
        a = out.setdefault(e.volcano_id, {
            "events": 0, "dre_m3": 0.0, "lava_m3": 0.0, "tephra_kg": 0.0, "gas_kg": 0.0,
            "so2_kg": 0.0, "oldest_years_bp": 0.0, "newest_years_bp": float("inf"),
        })  # fmt: skip
        p = e.products
        a["events"] += 1
        a["dre_m3"] += p["volume_dre_m3"]
        a["lava_m3"] += p["lava_volume_m3"]
        a["tephra_kg"] += p["tephra_mass_kg"]
        a["gas_kg"] += p["gas_mass_kg"]
        a["so2_kg"] += p["so2_mass_kg"]
        a["oldest_years_bp"] = max(a["oldest_years_bp"], e.years_bp)
        a["newest_years_bp"] = min(a["newest_years_bp"], e.years_bp)
    return out
