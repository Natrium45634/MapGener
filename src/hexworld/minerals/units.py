"""Typed units and the physical bookkeeping formulas of TZ 5.5.1 (§§5–6, 11–12).

Internally SI: metres, kg, t (1000 kg), m³, J, W, Ma; grades are **mass fractions**. Every
other grade unit is converted here, once, by a registry (no conversion by string switch in
the models). [Ф] formulas carry their equation in the docstring; nothing here is calibrated.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

CARAT_KG = 2e-4  # 1 ct = 0.2 g

#: grade units -> (kind, factor to the internal basis)
#: kind "mass": factor to a mass fraction; "volume": factor to kg per m³ of material;
#: "areal": factor to kg per m² of true area
GRADE_UNITS = {
    "fraction": ("mass", 1.0),
    "percent": ("mass", 1e-2),
    "ppm": ("mass", 1e-6),
    "ppb": ("mass", 1e-9),
    "g_per_t": ("mass", 1e-6),  # 1 g/t = 1 ppm by mass
    "kg_per_t": ("mass", 1e-3),
    "ct_per_t": ("mass", CARAT_KG / 1000.0),
    "kg_per_m3": ("volume", 1.0),
    "g_per_m3": ("volume", 1e-3),
    "mg_per_L": ("volume", 1e-3),  # 1 mg/L = 1 g/m³ = 1e-3 kg/m³
    "ct_per_m3": ("volume", CARAT_KG),
    "kg_per_m2": ("areal", 1.0),
}
#: quantity units (amounts, not grades)
AMOUNT_UNITS = {"t", "kg", "ct", "m3", "m3_standard", "J", "W", "m3_per_year"}


class UnitError(ValueError):
    pass


def grade_kind(unit: str) -> str:
    try:
        return GRADE_UNITS[unit][0]
    except KeyError as exc:
        raise UnitError(f"unknown grade unit {unit!r}") from exc


def to_internal(value, unit: str):
    """A grade in ``unit`` -> mass fraction, kg/m³ or kg/m² (by the unit's kind)."""
    v = np.asarray(value, dtype=np.float64)
    if not np.all(np.isfinite(v)) or np.any(v < 0):
        raise UnitError(f"grade must be finite and >= 0, got {value!r} {unit}")
    out = v * GRADE_UNITS[unit][1] if unit in GRADE_UNITS else None
    if out is None:
        raise UnitError(f"unknown grade unit {unit!r}")
    if grade_kind(unit) == "mass" and np.any(out > 1.0):
        raise UnitError(f"mass fraction above 1: {value!r} {unit}")
    return float(out) if np.ndim(out) == 0 else out


def from_internal(value, unit: str):
    return np.asarray(value, dtype=np.float64) / GRADE_UNITS[unit][1]


@dataclass(frozen=True)
class Grade:
    """A typed grade: value + unit + basis (``dry_rock``, ``material``, ``area``)."""

    value: float
    unit: str
    basis: str = "dry_rock"

    def __post_init__(self):
        to_internal(self.value, self.unit)  # validates

    @property
    def internal(self) -> float:
        return to_internal(self.value, self.unit)


# -- density and mass [Ф] ----------------------------------------------------------------------
def dry_bulk_density(solid_density_kg_m3: float, porosity: float) -> float:
    """``rho_bulk_dry = (1 - phi) rho_solid`` — applied once; never to a bulk density."""
    if not 0.0 <= porosity < 1.0:
        raise UnitError("porosity must be within [0, 1)")
    return (1.0 - porosity) * solid_density_kg_m3


def rock_mass_t(volume_m3, dry_bulk_density_kg_m3):
    """``T = ∫ rho_bulk dV / 1000`` for a uniform body."""
    return np.asarray(volume_m3, dtype=np.float64) * dry_bulk_density_kg_m3 / 1000.0


def contained_t(rock_t, grade_value, unit: str):
    """Contained mass (t) of a mass-fraction grade over a rock mass (t)."""
    if grade_kind(unit) != "mass":
        raise UnitError(f"{unit} is not a mass grade; use contained_from_volume_grade")
    return np.asarray(rock_t, dtype=np.float64) * to_internal(grade_value, unit)


def contained_from_volume_grade_kg(volume_m3, grade_value, unit: str):
    """``M = ∫ grade_kg_m3 dV`` (placers in g/m³, brines in mg/L)."""
    if grade_kind(unit) != "volume":
        raise UnitError(f"{unit} is not a volumetric grade")
    return np.asarray(volume_m3, dtype=np.float64) * to_internal(grade_value, unit)


def weighted_grade(rock_t, grades):
    """``Σ T_i g_i / Σ T_i`` (never the plain mean of hexes or samples)."""
    t = np.asarray(rock_t, dtype=np.float64)
    g = np.asarray(grades, dtype=np.float64)
    tot = t.sum()
    if tot <= 0:
        raise UnitError("weighted grade of no rock")
    return float((t * g).sum() / tot)


# -- groups (REE, PGE): views of the elements, never additive to them -------------------------
REE_ELEMENTS = (
    "La",
    "Ce",
    "Pr",
    "Nd",
    "Sm",
    "Eu",
    "Gd",
    "Tb",
    "Dy",
    "Ho",
    "Er",
    "Tm",
    "Yb",
    "Lu",
    "Y",
)  # fmt: skip  (Pm absent in natural ore; Sc separate)
PGE_ELEMENTS = ("Pt", "Pd", "Rh", "Ru", "Ir", "Os")
#: element -> oxide mass factor (RE2O3 except Ce O2, Pr6O11, Tb4O7), explicit for TREO
OXIDE_FACTORS = {"La": 1.1728, "Ce": 1.2284, "Pr": 1.2082, "Nd": 1.1664, "Sm": 1.1596,
                 "Eu": 1.1579, "Gd": 1.1526, "Tb": 1.1762, "Dy": 1.1477, "Ho": 1.1455,
                 "Er": 1.1435, "Tm": 1.1421, "Yb": 1.1387, "Lu": 1.1371, "Y": 1.2699}  # fmt: skip


def group_total(element_masses: dict, group=REE_ELEMENTS) -> float:
    """Contained group (REE, PGE) = sum of its physical elements; a derived view."""
    unknown = [k for k in element_masses if k not in group]
    if unknown:
        raise UnitError(f"not in the group: {unknown}")
    return float(sum(element_masses.values()))


def treo(element_masses: dict) -> float:
    """Total rare-earth oxides (derived, explicit oxide factors) — not added to elements."""
    return float(sum(m * OXIDE_FACTORS[e] for e, m in element_masses.items()))


# -- gems ---------------------------------------------------------------------------------------
def gem_rough_ct(carrier_kg: float, gem_fraction: float) -> float:
    """``ct_rough_gem = carrier_kg * gem_fraction * 5000`` (cut yield is separate)."""
    if not 0.0 <= gem_fraction <= 1.0:
        raise UnitError("gem fraction must be within [0, 1]")
    return carrier_kg * gem_fraction / CARAT_KG


def size_bins_ok(carrier_kg: float, bins_kg: dict) -> bool:
    """Size bins never hold more than the raw carrier mass."""
    return all(v >= 0 for v in bins_kg.values()) and sum(bins_kg.values()) <= carrier_kg * (
        1 + 1e-12
    )


# -- marine -------------------------------------------------------------------------------------
def nodule_mass_t(area_m2, coverage, areal_density_kg_m2, density_is_area_average=False):
    """``T = A_true * coverage * areal_density / 1000``; an area-average density does not take
    the coverage a second time."""
    cov = 1.0 if density_is_area_average else coverage
    return area_m2 * cov * areal_density_kg_m2 / 1000.0


def crust_thickness_m(growth_mm_per_ma: float, age_ma: float) -> float:
    """Slow accretion: thickness = rate × age (mm/Ma × Ma -> m)."""
    return growth_mm_per_ma * age_ma / 1000.0


# -- fluids ---------------------------------------------------------------------------------------
def dissolved_mass_kg(volume_m3, concentration_mg_l):
    """``M = V * C`` with 1 mg/L = 1e-3 kg/m³."""
    return np.asarray(volume_m3, dtype=np.float64) * concentration_mg_l * 1e-3


def salt_precipitated_kg(dissolved_kg: float, solution_m3: float, saturation_kg_m3: float):
    """``M_precip = max(0, M_dissolved - C_sat V)``: pure water leaves no salt."""
    return max(0.0, dissolved_kg - saturation_kg_m3 * solution_m3)


def pore_volume_m3(gross_m3, net_to_gross, porosity):
    """``V_pore = ∫ NTG phi dV``."""
    return gross_m3 * net_to_gross * porosity


def check_saturations(sw: float, so: float, sg: float) -> None:
    """``S_w + S_o + S_g = 1``, all >= 0."""
    if min(sw, so, sg) < 0 or abs(sw + so + sg - 1.0) > 1e-9:
        raise UnitError(f"saturations must be >= 0 and sum to 1: {sw}, {so}, {sg}")


def standard_volume_m3(reservoir_m3: float, formation_factor: float) -> float:
    """``V_std = V_res / B`` (B_o or B_g > 0 at declared standard P, T)."""
    if formation_factor <= 0:
        raise UnitError("formation volume factor must be > 0")
    return reservoir_m3 / formation_factor


def drainable_water_m3(area_m2: float, specific_yield: float, drawdown_m: float, porosity=None):
    """Unconfined: ``A * Sy * Δh`` (not the whole pore volume); ``Sy <= phi``."""
    if porosity is not None and specific_yield > porosity:
        raise UnitError("specific yield cannot exceed the porosity")
    return area_m2 * specific_yield * drawdown_m


def heat_in_place_j(volume_m3, porosity, rho_rock, c_rock, rho_water, c_water, delta_t_k):
    """``Q = V [(1-phi) rho_r c_r + phi rho_w c_w] (T - T_ref)`` in J (a stock, not W)."""
    return (
        volume_m3
        * ((1 - porosity) * rho_rock * c_rock + porosity * rho_water * c_water)
        * (delta_t_k)
    )


# -- geometry helpers (used by the bodies) ------------------------------------------------------
def ellipsoid_volume(a: float, b: float, c: float) -> float:
    return 4.0 / 3.0 * math.pi * a * b * c


def vertical_intercept(true_thickness_m: float, dip_deg: float) -> float:
    """Vertical intercept of a plane layer of true thickness t dipping δ: ``t / cos δ``."""
    c = math.cos(math.radians(dip_deg))
    if c <= 1e-12:
        return math.inf  # a vertical layer: the finite geometry decides (see bodies)
    return true_thickness_m / c


def horizontal_intercept(true_thickness_m: float, dip_deg: float) -> float:
    """Horizontal intercept along the dip direction: ``t / sin δ``."""
    s = math.sin(math.radians(dip_deg))
    if s <= 1e-12:
        return math.inf
    return true_thickness_m / s


def apparent_thickness(true_thickness_m: float, normal, direction) -> float:
    """``t_app = t / |n·d|`` for a ray crossing both faces (unit vectors)."""
    n = np.asarray(normal, dtype=np.float64)
    d = np.asarray(direction, dtype=np.float64)
    c = abs(float(n @ d)) / (np.linalg.norm(n) * np.linalg.norm(d))
    if c <= 1e-12:
        return math.inf
    return true_thickness_m / c
