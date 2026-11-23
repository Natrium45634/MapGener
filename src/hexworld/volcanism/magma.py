"""Magma profiles, an approximate viscosity model and the reservoir mass budget.

Viscosity ``[H, APPROXIMATE]``::

    log10(eta / Pa s) = 2.0 + 0.25 (SiO2 - 50) - 0.006 (T - 1150 degC) + 1.5 X - 0.25 H2O

calibrated to typical ranges (basalt 10^1-10^3, andesite 10^5-10^6, hydrous rhyolite
10^8-10^10 Pa s); the coefficients are configuration, not geophysical constants. The
Giordano et al. (2008) VFT model needs full oxide compositions and is not used here.

Reservoir budget (per system, SI, float64)::

    M(t + dt) = max(0, M + Mdot_in dt - M_erupted - M_intruded - M_solidified)

Lava and tephra are produced only out of ``M`` (no eruption without melt).
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass

from .units import SECONDS_PER_YEAR, stable_uniform

#: id -> (SiO2 wt%, T degC, H2O wt%, CO2 wt%, S wt%, crystal fraction, DRE density kg/m^3)
PROFILES = {
    "mor_basalt": (49.0, 1200.0, 0.2, 0.15, 0.10, 0.05, 2750.0),
    "ocean_island_basalt": (47.5, 1170.0, 0.5, 0.40, 0.15, 0.05, 2750.0),
    "rift_alkali_basalt": (46.5, 1150.0, 1.0, 0.80, 0.15, 0.10, 2700.0),
    "arc_oceanic": (54.0, 1080.0, 3.0, 0.20, 0.20, 0.20, 2650.0),
    "arc_continental": (61.0, 980.0, 4.0, 0.20, 0.15, 0.30, 2550.0),
    "continental_hotspot": (50.0, 1150.0, 1.0, 0.50, 0.15, 0.10, 2700.0),
    "rhyolite_caldera": (73.0, 820.0, 5.0, 0.10, 0.05, 0.20, 2400.0),
}


def log10_viscosity(sio2: float, t_c: float, crystals: float, h2o: float) -> float:
    """``[H]`` log10 of the melt viscosity, Pa s (see the module docstring)."""
    x = min(max(crystals, 0.0), 0.6)
    return (
        2.0
        + 0.25 * (sio2 - 50.0)
        - 0.006 * (t_c - 1150.0)
        + 1.5 * x
        - 0.25 * min(max(h2o, 0.0), 6.0)
    )


@dataclass
class Magma:
    profile: str
    silica_wt_pct: float
    temperature_c: float
    volatile_h2o_wt_pct: float
    co2_wt_pct: float
    sulfur_wt_pct: float
    crystal_fraction: float
    density_kg_m3: float

    @property
    def log10_viscosity_pa_s(self) -> float:
        return log10_viscosity(
            self.silica_wt_pct, self.temperature_c, self.crystal_fraction,
            self.volatile_h2o_wt_pct,
        )  # fmt: skip

    @property
    def magma_class(self) -> str:
        s = self.silica_wt_pct
        if s < 52:
            return "basaltic"
        if s < 57:
            return "basaltic_andesitic"
        if s < 63:
            return "andesitic"
        if s < 69:
            return "dacitic"
        return "rhyolitic"

    def to_dict(self) -> dict:
        d = asdict(self)
        d["log10_viscosity_pa_s"] = round(self.log10_viscosity_pa_s, 3)
        d["magma_class"] = self.magma_class
        return d


def magma_for(profile: str, *key) -> Magma:
    """The profile with a small deterministic spread per system (keyed by ``key``)."""
    sio2, t, h2o, co2, s, x, rho = PROFILES[profile]
    u = [stable_uniform("magma", profile, i, *key) for i in range(4)]
    return Magma(
        profile=profile,
        silica_wt_pct=round(sio2 + (u[0] - 0.5) * 4.0, 2),
        temperature_c=round(t + (u[1] - 0.5) * 60.0, 1),
        volatile_h2o_wt_pct=round(max(0.05, h2o * (0.75 + 0.5 * u[2])), 3),
        co2_wt_pct=co2,
        sulfur_wt_pct=s,
        crystal_fraction=round(min(0.55, max(0.0, x + (u[3] - 0.5) * 0.1)), 3),
        density_kg_m3=rho,
    )


#: Time scale of solidification of a crustal reservoir without supply, years [H].
SOLIDIFY_YEARS = 5_000.0
#: Share of the supply that freezes as intrusions (dikes, sills, plutons) instead of reaching
#: the eruptible reservoir, by source [H]. Intrusive:extrusive ratios of ~5-10:1 at arcs and
#: ridges, ~1-2:1 at ocean-island hotspots (White et al. 2006 order of magnitude).
INTRUSIVE_SHARE = {
    "mid_ocean_ridge": 0.9,
    "subduction_arc": 0.8,
    "oceanic_hotspot": 0.55,
    "continental_hotspot": 0.9,
    "divergent_rift": 0.85,
}
DEFAULT_INTRUSIVE_SHARE = 0.8


@dataclass
class Reservoir:
    """Eruptible melt of one system; every term in kg, rates in kg/s."""

    melt_kg: float
    input_kg_s: float
    critical_kg: float  # melt above which unrest and eruptions are likely
    erupted_kg: float = 0.0
    intruded_kg: float = 0.0
    solidified_kg: float = 0.0
    supplied_kg: float = 0.0
    initial_kg: float = 0.0
    years_since_recharge: float = 0.0
    intrusive_share: float = DEFAULT_INTRUSIVE_SHARE

    @classmethod
    def steady(
        cls, input_kg_s: float, critical_kg: float, fill: float = 0.6,
        intrusive_share: float = DEFAULT_INTRUSIVE_SHARE,
    ) -> Reservoir:  # fmt: skip
        m0 = max(0.0, fill) * critical_kg
        return cls(
            melt_kg=m0, input_kg_s=input_kg_s, critical_kg=critical_kg, initial_kg=m0,
            intrusive_share=min(1.0, max(0.0, intrusive_share)),
        )  # fmt: skip

    def advance(self, years: float) -> None:
        """Supply, intrusion and solidification over ``years`` (exact for constant input)."""
        if years <= 0:
            return
        supply = self.input_kg_s * SECONDS_PER_YEAR * years
        to_res = supply * (1.0 - self.intrusive_share)
        self.supplied_kg += supply
        self.intruded_kg += supply - to_res
        # dM/dt = s - M / tau  ->  M(t) = M_eq + (M0 - M_eq) e^{-t/tau}
        tau = SOLIDIFY_YEARS
        rate = to_res / years
        m_eq = rate * tau
        m_new = m_eq + (self.melt_kg - m_eq) * math.exp(-years / tau)
        # what froze = what came in - what is left over the change
        self.solidified_kg += max(0.0, self.melt_kg + to_res - m_new)
        self.melt_kg = max(0.0, m_new)
        self.years_since_recharge = (
            0.0 if self.input_kg_s > 0 else self.years_since_recharge + years
        )

    def withdraw(self, mass_kg: float) -> float:
        """Take up to ``mass_kg`` for an eruption; returns what was actually available."""
        take = max(0.0, min(float(mass_kg), self.melt_kg))
        self.melt_kg -= take
        self.erupted_kg += take
        return take

    @property
    def pressure_proxy(self) -> float:
        """0..~2: fill of the reservoir relative to its critical melt mass."""
        return self.melt_kg / max(self.critical_kg, 1.0)

    def budget_residual_kg(self) -> float:
        """``initial + supplied - erupted - intruded - solidified - melt`` (should be ~0)."""
        return (
            self.initial_kg
            + self.supplied_kg
            - self.erupted_kg
            - self.intruded_kg
            - self.solidified_kg
            - self.melt_kg
        )

    def to_dict(self) -> dict:
        d = asdict(self)
        d["pressure_proxy"] = round(self.pressure_proxy, 4)
        return d
