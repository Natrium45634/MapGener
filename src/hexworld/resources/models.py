"""Deposit models (TZ 5.2 sec. 13), geometry of parametric bodies (sec. 15) and
grade-tonnage sampling (sec. 16).

A model is a *permissive* description: required geological settings, the body shape, the
formation depth below the surface of its time, density, and a joint log-normal
grade-tonnage distribution. The medians are rounded orders of magnitude of the USGS
grade-tonnage compilations for the type (Cox & Singer 1986, Singer 1995, later updates);
they are labelled ``synthetic_not_calibrated``: a plausible generator, not an estimate of
Earth's deposits.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

#: commodity families of the dense map layers
FAMILIES = (
    "base_metals",  # Cu Pb Zn Ni Co
    "precious_metals",  # Au Ag PGE
    "ferrous_alloy",  # Fe Mn Cr W Mo
    "light_battery",  # Al Li Ti
    "critical_rare",  # REE Nb Ta Sn U
    "energy",  # coal, oil, gas
    "industrial",  # P K NaCl S graphite gypsum
    "gems",  # diamond, gemstones
    "marine",  # nodules, crusts, sea-floor sulfides
)
FAMILY_TITLES = {
    "base_metals": "цветные металлы (Cu, Pb, Zn, Ni, Co)",
    "precious_metals": "благородные металлы (Au, Ag, Pt)",
    "ferrous_alloy": "чёрные и легирующие (Fe, Mn, Cr, W, Mo)",
    "light_battery": "лёгкие и батарейные (Al, Li, Ti)",
    "critical_rare": "редкие и критические (РЗЭ, Nb, Ta, Sn, U)",
    "energy": "энергетическое сырьё (уголь, нефть, газ)",
    "industrial": "агро- и химическое сырьё (P, K, соли, S, графит)",
    "gems": "алмазы и самоцветы",
    "marine": "морские ресурсы (конкреции, корки, сульфиды)",
}


@dataclass(frozen=True)
class DepositModel:
    model_id: str  # R01..R35 of the TZ catalogue
    name: str  # machine name
    title: str  # Russian
    family: str
    shape: str  # body type of sec. 15.1
    #: commodities: element -> (median mass fraction, log sd)
    grades: dict
    tonnage_median_t: float
    tonnage_log_sd: float
    #: correlation of ln tonnage with ln grade (all grades), and among grades
    rho_tonnage_grade: float = -0.2
    rho_grades: float = 0.5
    density_kg_m3: float = 2700.0
    #: semi-axis ratios (a : b : c), a the longest horizontal axis
    aspect: tuple = (1.0, 0.6, 0.4)
    #: depth of the body top below the surface of its time (m), range
    formation_depth_m: tuple = (0.0, 500.0)
    #: grade caps (physical maxima of the mineral), default 0.6
    max_grade: dict = field(default_factory=dict)
    #: number of bodies of one system (range)
    bodies: tuple = (1, 1)
    #: barren share inside the body envelope
    f_barren: float = 0.0
    #: element (heavy, resistant) released to placers when the body erodes
    placer_elements: tuple = ()
    priority: str = "P0"
    note: str = ""


def _m(*args, **kw):
    return DepositModel(*args, **kw)


GT = 1e-6  # 1 g/t as a mass fraction
MODELS = {
    m.model_id: m
    for m in (
        _m(
            "R01",
            "porphyry_cu_mo_au",
            "порфировое Cu–Mo–Au",
            "base_metals",
            "stockwork_zone",
            {
                "Cu": (0.0044, 0.45),
                "Mo": (0.00015, 0.9),
                "Au": (0.2 * GT, 0.9),
                "Ag": (1.5 * GT, 0.9),
            },
            1.4e8,
            1.5,
            -0.25,
            0.4,
            2650.0,
            (1.0, 0.8, 0.9),
            (800.0, 3000.0),
            {"Cu": 0.03},
            (1, 2),
            0.15,
            ("Au",),
        ),
        _m(
            "R02",
            "epithermal_au_ag",
            "эпитермальное Au–Ag",
            "precious_metals",
            "vein",
            {"Au": (7.5 * GT, 1.0), "Ag": (110 * GT, 1.2)},
            7.7e5,
            1.6,
            -0.3,
            0.6,
            2600.0,
            (1.0, 0.02, 0.35),
            (0.0, 600.0),
            {"Au": 200 * GT, "Ag": 5000 * GT},
            (2, 6),
            0.0,
            ("Au",),
        ),
        _m(
            "R03",
            "skarn_cu_fe_w",
            "скарновое (Cu, Fe, W, Zn)",
            "base_metals",
            "lens",
            {"Cu": (0.017, 0.6), "Fe": (0.40, 0.25), "W": (0.004, 0.9), "Zn": (0.02, 0.9)},
            5.6e5,
            1.9,
            -0.2,
            0.3,
            3300.0,
            (1.0, 0.5, 0.3),
            (1000.0, 4000.0),
            {"Fe": 0.68},
            (1, 3),
        ),
        _m(
            "R04",
            "orogenic_gold",
            "орогенное (жильное) золото",
            "precious_metals",
            "fault_hosted_vein_network",
            {"Au": (6.0 * GT, 0.9)},
            1.1e6,
            2.0,
            -0.2,
            0.0,
            2750.0,
            (1.0, 0.01, 0.4),
            (3000.0, 12000.0),
            {"Au": 100 * GT},
            (3, 10),
            0.0,
            ("Au",),
        ),
        _m(
            "R05",
            "vms_cu_zn",
            "колчеданное (VMS) Cu–Zn–Pb",
            "base_metals",
            "lens",
            {
                "Cu": (0.013, 0.6),
                "Zn": (0.020, 0.8),
                "Pb": (0.004, 1.0),
                "Ag": (35 * GT, 1.0),
                "Au": (0.5 * GT, 1.0),
            },
            1.5e6,
            1.8,
            -0.2,
            0.5,
            4000.0,
            (1.0, 0.6, 0.15),
            (0.0, 0.0),
            {"Zn": 0.3},
            (1, 3),
        ),
        _m(
            "R06",
            "sedex_zn_pb",
            "стратиформное SEDEX Zn–Pb",
            "base_metals",
            "stratabound_layer",
            {"Zn": (0.056, 0.5), "Pb": (0.028, 0.7), "Ag": (30 * GT, 1.0)},
            1.5e7,
            1.6,
            -0.1,
            0.6,
            3400.0,
            (1.0, 0.6, 0.006),
            (0.0, 400.0),
            {},
            (1, 2),
        ),
        _m(
            "R07",
            "sediment_hosted_cu",
            "осадочная (медистые песчаники/сланцы) Cu",
            "base_metals",
            "stratabound_layer",
            {"Cu": (0.021, 0.5), "Co": (0.001, 1.0), "Ag": (20 * GT, 1.0)},
            2.2e7,
            1.7,
            -0.1,
            0.3,
            2700.0,
            (1.0, 0.7, 0.004),
            (200.0, 2000.0),
            {},
            (1, 1),
        ),
        _m(
            "R08",
            "magmatic_ni_cu_pge",
            "магматическое Ni–Cu–ЭПГ",
            "base_metals",
            "lens",
            {
                "Ni": (0.008, 0.6),
                "Cu": (0.005, 0.7),
                "Co": (0.0003, 0.7),
                "Pt": (0.5 * GT, 1.0),
                "Pd": (0.7 * GT, 1.0),
            },
            2.1e6,
            2.0,
            -0.1,
            0.6,
            3600.0,
            (1.0, 0.4, 0.2),
            (1000.0, 5000.0),
            {},
            (1, 3),
            0.0,
            ("Pt",),
        ),
        _m(
            "R09",
            "podiform_chromite",
            "хромитовое (офиолиты, расслоенные массивы)",
            "ferrous_alloy",
            "lens",
            {"Cr": (0.29, 0.25)},
            2.0e4,
            2.2,
            -0.05,
            0.0,
            4300.0,
            (1.0, 0.5, 0.3),
            (1000.0, 8000.0),
            {"Cr": 0.46},
            (2, 8),
        ),
        _m(
            "R10",
            "li_pegmatite",
            "пегматиты Li–Cs–Ta",
            "light_battery",
            "vein",
            {"Li": (0.0065, 0.4), "Ta": (0.00012, 0.9), "Sn": (0.0005, 1.0)},
            5.0e6,
            1.7,
            -0.1,
            0.3,
            2700.0,
            (1.0, 0.04, 0.5),
            (3000.0, 10000.0),
            {},
            (3, 12),
        ),
        _m(
            "R11",
            "carbonatite_ree_nb",
            "карбонатиты РЗЭ–Nb",
            "critical_rare",
            "pipe",
            {"REE": (0.012, 0.8), "Nb": (0.006, 0.8), "P": (0.02, 0.6)},
            6.0e7,
            1.6,
            -0.1,
            0.4,
            3000.0,
            (1.0, 0.8, 1.6),
            (500.0, 4000.0),
            {"REE": 0.12},
            (1, 1),
        ),
        _m(
            "R12",
            "greisen_sn_w",
            "грейзены и жилы Sn–W",
            "critical_rare",
            "stockwork_zone",
            {"Sn": (0.003, 0.7), "W": (0.002, 0.8), "Mo": (0.0005, 1.0)},
            7.0e6,
            1.7,
            -0.2,
            0.4,
            2700.0,
            (1.0, 0.6, 0.6),
            (1000.0, 4000.0),
            {},
            (1, 3),
            0.1,
            ("Sn",),
        ),
        _m(
            "R13",
            "sandstone_uranium",
            "урановое (роллы в песчаниках)",
            "critical_rare",
            "stratabound_layer",
            {"U": (0.0017, 0.7)},
            3.0e5,
            1.8,
            -0.2,
            0.0,
            2300.0,
            (1.0, 0.15, 0.01),
            (100.0, 1200.0),
            {},
            (2, 5),
        ),
        _m(
            "R14",
            "unconformity_uranium",
            "урановое (несогласие)",
            "critical_rare",
            "lens",
            {"U": (0.015, 1.0)},
            4.0e5,
            1.8,
            -0.3,
            0.0,
            2600.0,
            (1.0, 0.3, 0.1),
            (100.0, 1500.0),
            {"U": 0.2},
            (1, 2),
        ),
        _m(
            "R15",
            "banded_iron_formation",
            "полосчатые железистые формации (BIF)",
            "ferrous_alloy",
            "sheet",
            {"Fe": (0.52, 0.12), "Mn": (0.002, 0.8)},
            1.7e8,
            1.4,
            0.0,
            0.0,
            3400.0,
            (1.0, 0.2, 0.004),
            (0.0, 3000.0),
            {"Fe": 0.68},
            (1, 3),
        ),
        _m(
            "R16",
            "sedimentary_fe_mn",
            "осадочные Fe–Mn (шельфовые)",
            "ferrous_alloy",
            "stratabound_layer",
            {"Mn": (0.30, 0.3), "Fe": (0.08, 0.6)},
            1.0e7,
            1.6,
            0.0,
            0.0,
            3200.0,
            (1.0, 0.5, 0.002),
            (0.0, 1500.0),
            {"Mn": 0.6},
            (1, 2),
        ),
        _m(
            "R17",
            "bauxite",
            "бокситы (латеритные)",
            "light_battery",
            "regolith_cap",
            {"Al": (0.24, 0.12)},
            3.0e7,
            1.6,
            0.0,
            0.0,
            2000.0,
            (1.0, 0.7, 0.0015),
            (0.0, 0.0),
            {"Al": 0.33},
            (1, 3),
        ),
        _m(
            "R18",
            "ni_co_laterite",
            "Ni–Co латериты",
            "base_metals",
            "regolith_cap",
            {"Ni": (0.014, 0.25), "Co": (0.0007, 0.6)},
            3.0e7,
            1.5,
            0.0,
            0.5,
            1700.0,
            (1.0, 0.7, 0.002),
            (0.0, 0.0),
            {"Ni": 0.04},
            (1, 2),
        ),
        _m(
            "R19",
            "ree_ion_clay",
            "ионно-адсорбционные РЗЭ-глины",
            "critical_rare",
            "regolith_cap",
            {"REE": (0.0008, 0.5)},
            5.0e6,
            1.4,
            0.0,
            0.0,
            1800.0,
            (1.0, 0.8, 0.002),
            (0.0, 0.0),
            {},
            (1, 3),
        ),
        _m(
            "R20",
            "fluvial_placer",
            "речные / террасовые россыпи",
            "precious_metals",
            "placer_ribbon",
            {"Au": (0.5 * GT, 1.0)},
            1.0e6,
            1.5,
            0.0,
            0.0,
            2000.0,
            (1.0, 0.004, 0.0005),
            (0.0, 0.0),
            {},
            (1, 1),
        ),
        _m(
            "R21",
            "heavy_mineral_sands",
            "береговые тяжёлые пески (Ti, Zr)",
            "light_battery",
            "placer_ribbon",
            {"Ti": (0.03, 0.5), "Zr": (0.005, 0.6), "REE": (0.0002, 0.8)},
            5.0e7,
            1.4,
            0.0,
            0.5,
            1900.0,
            (1.0, 0.02, 0.0002),
            (0.0, 0.0),
            {},
            (1, 1),
        ),
        _m(
            "R22",
            "kimberlite_diamond",
            "кимберлитовые трубки (алмазы)",
            "gems",
            "pipe",
            {"diamond": (0.1e-6, 1.0)},
            2.6e7,
            1.8,
            -0.2,
            0.0,
            2800.0,
            (1.0, 0.8, 4.0),
            (0.0, 0.0),
            {"diamond": 5e-6},
            (1, 1),
            0.0,
            ("diamond",),
            note="grade in mass fraction: 0.5 ct/t = 1e-7",
        ),
        _m(
            "R23",
            "graphite",
            "графит (метаморфизованные углеродистые толщи)",
            "industrial",
            "lens",
            {"C_graphite": (0.10, 0.5)},
            5.0e6,
            1.5,
            0.0,
            0.0,
            2600.0,
            (1.0, 0.3, 0.05),
            (2000.0, 8000.0),
            {"C_graphite": 0.4},
            (1, 3),
        ),
        _m(
            "R24",
            "coal",
            "уголь / бурый уголь",
            "energy",
            "stratabound_layer",
            {"coal": (0.85, 0.05)},
            5.0e8,
            1.6,
            0.0,
            0.0,
            1350.0,
            (1.0, 0.6, 0.00004),
            (200.0, 3000.0),
            {"coal": 1.0},
            (2, 6),
        ),
        _m(
            "R25",
            "oil_gas",
            "нефть и газ (ловушка)",
            "energy",
            "reservoir_trap",
            {"oil": (0.012, 0.6), "gas": (0.004, 0.8)},
            3.0e9,
            1.6,
            0.0,
            0.0,
            2400.0,
            (1.0, 0.5, 0.012),
            (1500.0, 4500.0),
            {},
            (1, 1),
            priority="P1",
            note="mass fractions of the reservoir rock (porosity x saturation x density)",
        ),
        _m(
            "R26",
            "evaporite_salts",
            "соленосные и калийные эвапориты",
            "industrial",
            "evaporite_bed",
            {"NaCl": (0.85, 0.08), "K": (0.002, 1.2)},
            3.0e9,
            1.5,
            0.0,
            0.0,
            2170.0,
            (1.0, 0.6, 0.003),
            (100.0, 3000.0),
            {"NaCl": 0.98, "K": 0.2},
            (1, 2),
        ),
        _m(
            "R27",
            "lithium_brine",
            "литиевые рассолы (бессточный бассейн)",
            "light_battery",
            "evaporite_bed",
            {"Li": (0.0, 0.0)},
            0.0,
            0.0,
            0.0,
            0.0,
            1200.0,
            (1.0, 0.7, 0.002),
            (0.0, 300.0),
            {},
            (1, 1),
            note="mass computed from brine volume x concentration",
        ),
        _m(
            "R28",
            "phosphorite",
            "фосфориты (морские)",
            "industrial",
            "stratabound_layer",
            {"P": (0.11, 0.2)},
            1.0e8,
            1.6,
            0.0,
            0.0,
            2600.0,
            (1.0, 0.6, 0.0005),
            (0.0, 1500.0),
            {"P": 0.17},
            (1, 2),
        ),
        _m(
            "R29",
            "seafloor_massive_sulfide",
            "морские массивные сульфиды (SMS)",
            "marine",
            "hydrothermal_seafloor_lens",
            {"Cu": (0.03, 0.8), "Zn": (0.08, 0.9), "Au": (2.0 * GT, 1.0), "Ag": (60 * GT, 1.0)},
            1.0e5,
            1.6,
            -0.1,
            0.5,
            3800.0,
            (1.0, 0.7, 0.15),
            (0.0, 0.0),
            {"Zn": 0.4},
            (1, 4),
        ),
        _m(
            "R30",
            "fe_mn_nodules",
            "железомарганцевые конкреции",
            "marine",
            "seamount_surface_crust",
            {"Mn": (0.25, 0.15), "Ni": (0.013, 0.25), "Cu": (0.011, 0.3), "Co": (0.002, 0.4)},
            0.0,
            0.0,
            0.0,
            0.5,
            2000.0,
            (1.0, 0.7, 0.0),
            (0.0, 0.0),
            {},
            (1, 1),
            note="areal density kg/m2 x area; not a deep mine",
        ),
        _m(
            "R31",
            "cobalt_crust",
            "кобальтоносные корки подводных гор",
            "marine",
            "seamount_surface_crust",
            {"Co": (0.006, 0.3), "Mn": (0.22, 0.15), "REE": (0.002, 0.4)},
            0.0,
            0.0,
            0.0,
            0.5,
            1900.0,
            (1.0, 0.7, 0.0),
            (0.0, 0.0),
            {},
            (1, 1),
        ),
        _m(
            "R32",
            "volcanic_sulfur",
            "сера и фумарольные минералы",
            "industrial",
            "lens",
            {"S": (0.30, 0.4)},
            1.0e6,
            1.6,
            0.0,
            0.0,
            2000.0,
            (1.0, 0.6, 0.2),
            (0.0, 200.0),
            {"S": 0.9},
            (1, 2),
        ),
        _m(
            "R33",
            "construction_materials",
            "строительные материалы",
            "industrial",
            "sheet",
            {},
            0.0,
            0.0,
            0.0,
            0.0,
            2600.0,
            note="dense per-hex material class and quality",
        ),
        _m(
            "R34",
            "gemstones",
            "самоцветы (пегматиты, метаморфиты)",
            "gems",
            "vein",
            {"gem": (2e-6, 1.2)},
            1.0e5,
            1.8,
            0.0,
            0.0,
            2700.0,
            (1.0, 0.03, 0.4),
            (2000.0, 8000.0),
            {},
            (1, 4),
        ),
        _m(
            "R35",
            "geothermal",
            "геотермальные ресурсы",
            "energy",
            "ellipsoidal_intrusion",
            {},
            0.0,
            0.0,
            0.0,
            0.0,
            2600.0,
            priority="P1",
            note="heat, not ore mass: geothermal_heat_mw of v5.1 is reported, no tonnage",
        ),
    )
}


# -- exact geometry (N11-N13) ----------------------------------------------------------------------
def ellipsoid_volume(a, b, c):
    return 4.0 / 3.0 * math.pi * a * b * c


def ore_mass_t(volume_m3, density_kg_m3, f_barren=0.0):
    return volume_m3 * density_kg_m3 * (1.0 - f_barren) / 1000.0


def contained_t(ore_t, grade):
    """Contained element in t (grade a mass fraction of the element, not of an oxide)."""
    if not 0.0 <= grade <= 1.0:
        raise ValueError("grade must be a mass fraction 0..1")
    return ore_t * grade


def ppm_to_fraction(ppm):
    return ppm * 1e-6


def g_per_t_to_fraction(g_t):
    return g_t * 1e-6


def brine_mass_kg(pore_water_m3, conc_mg_l):
    """Element in a brine: pore water volume x concentration (N16: 1e6 m3 x 100 mg/L
    = 1e5 kg), never the mass of the whole basin rock."""
    return pore_water_m3 * conc_mg_l * 1e-3


def recoverable_t(contained, mining_recovery, processing_recovery):
    """Recoverable metal (N18); the geological in-situ stock stays unchanged."""
    return contained * mining_recovery * processing_recovery


def ellipsoid_cap_volume(a, b, c, h):
    """Volume of the top cap of height ``h`` (0..2c) cut from an ellipsoid with vertical
    semi-axis ``c``: ``pi a b h^2 (3c - h) / (3 c^2)``."""
    h = min(max(h, 0.0), 2.0 * c)
    return math.pi * a * b * h * h * (3.0 * c - h) / (3.0 * c * c)


SLAB_SHAPES = (
    "sheet",
    "stratabound_layer",
    "regolith_cap",
    "evaporite_bed",
    "seamount_surface_crust",
    "placer_ribbon",
)
ELLIPSOID_SHAPES = (
    "ellipsoidal_intrusion",
    "lens",
    "stockwork_zone",
    "hydrothermal_seafloor_lens",
    "reservoir_trap",
)
VEIN_SHAPES = ("vein", "fault_hosted_vein_network")


def shape_volume(shape, a, b, c, n_veins=1):
    """Envelope volume of a body (m3). Slabs: elliptic plate ``pi a b (2c)``; pipes:
    elliptic cylinder of depth ``2c``; veins: ``n`` plates ``2a x 2c x 2b``."""
    if shape in ELLIPSOID_SHAPES:
        return ellipsoid_volume(a, b, c)
    if shape in SLAB_SHAPES:
        return math.pi * a * b * 2.0 * c
    if shape == "pipe":
        return math.pi * a * b * 2.0 * c
    if shape in VEIN_SHAPES:
        return n_veins * (2.0 * a) * (2.0 * b) * (2.0 * c)
    raise ValueError(shape)


def axes_for_volume(shape, volume_m3, aspect, n_veins=1):
    """Semi-axes (a, b, c) of the given volume and aspect ratio (a : b : c)."""
    ra, rb, rc = aspect
    unit = shape_volume(shape, ra, rb, rc, n_veins)
    s = (volume_m3 / unit) ** (1.0 / 3.0)
    return ra * s, rb * s, rc * s


# -- grade-tonnage ---------------------------------------------------------------------------------
def sample_grade_tonnage(model: DepositModel, z: np.ndarray, size_scale: float = 1.0):
    """Joint log-normal (ln M, ln g_1..g_k) with the model correlations, from standard
    normals ``z`` (k + 1,) of a per-object stream; |z| clipped at 3 (truncation).

    Returns ``(ore_t, {element: grade})``. Correlation: Sigma = D R D with
    R[0, i] = rho_tg, R[i, j] = rho_gg (Cholesky); grades capped by ``max_grade``.
    """
    els = list(model.grades)
    k = len(els)
    r = np.full((k + 1, k + 1), model.rho_grades)
    r[0, :] = r[:, 0] = model.rho_tonnage_grade
    np.fill_diagonal(r, 1.0)
    L = np.linalg.cholesky(r + 1e-9 * np.eye(k + 1))
    zz = np.clip(np.asarray(z[: k + 1], dtype=np.float64), -3.0, 3.0)
    w = L @ zz
    ore = model.tonnage_median_t * size_scale * math.exp(model.tonnage_log_sd * w[0])
    grades = {}
    for i, el in enumerate(els):
        med, sd = model.grades[el]
        g = med * math.exp(sd * w[i + 1]) if med > 0 else 0.0
        grades[el] = float(min(g, model.max_grade.get(el, 0.6), 1.0))
    return float(ore), grades


# -- Poisson counts from uniforms (no hidden RNG) --------------------------------------------------
def poisson_from_uniform(lam: float, u: float) -> int:
    """Inverse CDF of Poisson(lam) at ``u`` in (0, 1); lam = 0 gives 0 (N19)."""
    if lam <= 0.0:
        return 0
    if lam > 500.0:  # normal approximation for big tracts
        from statistics import NormalDist

        return max(
            0,
            int(round(lam + math.sqrt(lam) * NormalDist().inv_cdf(min(max(u, 1e-12), 1 - 1e-12)))),
        )
    k, p = 0, math.exp(-lam)
    cdf = p
    while u > cdf and k < 10000:
        k += 1
        p *= lam / k
        cdf += p
    return k


# -- bodies: membership, depth intervals, cross-sections -------------------------------------------
def _local(body, x_m, y_m):
    cx, cy = body["center_m"][0], body["center_m"][1]
    th = math.radians(body.get("strike_deg", 0.0))
    dx, dy = np.asarray(x_m) - cx, np.asarray(y_m) - cy
    u = dx * math.cos(th) + dy * math.sin(th)
    v = -dx * math.sin(th) + dy * math.cos(th)
    return u, v


def body_interval(body, x_m, y_m):
    """Vertical interval (z_top, z_bottom) of the body at world points (m, datum); NaN
    where the vertical line misses it. Ellipsoids are exact; slabs, pipes and vein sets
    use their envelope plates."""
    a, b, c = body["axes_m"]
    zc = body["center_m"][2]
    u, v = _local(body, x_m, y_m)
    shape = body["shape"]
    r2 = (u / a) ** 2 + (v / b) ** 2
    if shape in ELLIPSOID_SHAPES:
        h = c * np.sqrt(np.clip(1.0 - r2, 0.0, None))
        top, bot = zc + h, zc - h
    else:
        top, bot = np.full_like(r2, zc + c), np.full_like(r2, zc - c)
        if shape in VEIN_SHAPES:
            r2 = np.maximum((u / a) ** 2, 0.0) + 0.0 * v
            hit = np.abs(v) <= b * max(body.get("n_veins", 1), 1) * 6.0
            r2 = np.where(hit, r2, 2.0)
    inside = r2 <= 1.0
    # the eroded part above the present surface is gone
    cut = body.get("eroded_above_m")
    if cut is not None:
        top = np.minimum(top, cut)
        inside &= top > bot
    return np.where(inside, top, np.nan), np.where(inside, bot, np.nan)


def footprint_radius_m(body):
    a, b, _c = body["axes_m"]
    if body["shape"] in VEIN_SHAPES:
        return math.hypot(a, b * max(body.get("n_veins", 1), 1) * 6.0)
    return max(a, b)


def query_bodies_at(catalog_bodies, x_m, y_m, depth_min_m=None, depth_max_m=None, surface_m=0.0):
    """Bodies whose vertical line at (x, y) crosses the depth window (depths below the
    given surface height). No voxels, no climate recomputation."""
    out = []
    for body in catalog_bodies:
        cx, cy = body["center_m"][:2]
        if math.hypot(x_m - cx, y_m - cy) > footprint_radius_m(body):
            continue
        top, bot = body_interval(body, np.array([x_m]), np.array([y_m]))
        if not np.isfinite(top[0]):
            continue
        d_top, d_bot = surface_m - top[0], surface_m - bot[0]
        if depth_max_m is not None and d_top > depth_max_m:
            continue
        if depth_min_m is not None and d_bot < depth_min_m:
            continue
        out.append(
            {
                "body_id": body["body_id"],
                "system_id": body["system_id"],
                "depth_top_m": float(d_top),
                "depth_bottom_m": float(d_bot),
            }
        )
    return out


def section_intervals(body, xs_m, ys_m):
    """Intervals of one body along a section polyline: list of (index, z_top, z_bottom)."""
    top, bot = body_interval(body, xs_m, ys_m)
    ok = np.isfinite(top)
    return [(int(i), float(top[i]), float(bot[i])) for i in np.nonzero(ok)[0]]
