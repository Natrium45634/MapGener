"""Sea-floor provinces from geology + relative morphology + neighbourhood (TZ 4.6, 4.1).

A province is never a depth threshold alone (``depth < 200 => shelf`` is forbidden): the
shelf is submerged *continental crust*, the slope and rise follow the crest-edge distance
and the margin type, the ridge follows the crust age, trenches the subduction geometry,
seamounts and guyots the volcanic history. One primary province per hex plus separate
overlay masks (a slope can contain a canyon, an abyssal plain a fracture zone).
"""

from __future__ import annotations

import numpy as np

PROVINCES = (
    "land",
    "continental_shelf",
    "submerged_continental_plateau",
    "continental_slope",
    "continental_rise",
    "abyssal_plain",
    "abyssal_hills",
    "mid_ocean_ridge",
    "axial_rift_valley",
    "transform_fault",
    "fracture_zone",
    "subduction_trench",
    "outer_rise",
    "forearc",
    "marginal_sea",
    "seamount",
    "guyot",
    "volcanic_plateau",
    "submarine_canyon",
    "submarine_fan",
    "inland_water",
)
CODE = {name: i for i, name in enumerate(PROVINCES)}
TITLES = {
    "land": "суша",
    "continental_shelf": "шельф (материковая кора под водой)",
    "submerged_continental_plateau": "подводное материковое плато",
    "continental_slope": "материковый склон",
    "continental_rise": "материковое подножие",
    "abyssal_plain": "абиссальная равнина (осадки)",
    "abyssal_hills": "абиссальные холмы",
    "mid_ocean_ridge": "срединно-океанический хребет",
    "axial_rift_valley": "осевая рифтовая долина",
    "transform_fault": "трансформный разлом (активный)",
    "fracture_zone": "зона разломов (след)",
    "subduction_trench": "глубоководный желоб",
    "outer_rise": "краевой вал перед желобом",
    "forearc": "преддуговый склон",
    "marginal_sea": "окраинное (задуговое) море",
    "seamount": "подводная гора",
    "guyot": "гайот",
    "volcanic_plateau": "океаническое вулканическое плато",
    "submarine_canyon": "подводный каньон",
    "submarine_fan": "конус выноса",
    "inland_water": "озеро / внутреннее море",
}
RGB = {
    "land": (205, 196, 172),
    "continental_shelf": (170, 220, 240),
    "submerged_continental_plateau": (120, 185, 215),
    "continental_slope": (95, 150, 200),
    "continental_rise": (120, 140, 190),
    "abyssal_plain": (45, 70, 140),
    "abyssal_hills": (70, 90, 160),
    "mid_ocean_ridge": (235, 140, 70),
    "axial_rift_valley": (190, 60, 40),
    "transform_fault": (240, 220, 60),
    "fracture_zone": (200, 200, 120),
    "subduction_trench": (10, 10, 20),
    "outer_rise": (120, 110, 170),
    "forearc": (150, 110, 140),
    "marginal_sea": (90, 160, 160),
    "seamount": (200, 80, 160),
    "guyot": (240, 120, 200),
    "volcanic_plateau": (150, 90, 60),
    "submarine_canyon": (255, 255, 255),
    "submarine_fan": (210, 180, 120),
    "inland_water": (90, 200, 230),
}


def classify_sea_floor(T: dict, hist: dict, hydro: dict) -> np.ndarray:
    """Primary province code (uint8) of every hex of the final surface."""
    ocean = np.asarray(hydro["ocean_mask"], dtype=bool)
    lake = np.asarray(hydro["inland_water_mask"], dtype=bool)
    shape = ocean.shape
    out = np.full(shape, CODE["land"], dtype=np.uint8)
    out[lake] = CODE["inland_water"]
    marine = np.asarray(T["marine_floor_mask"], dtype=bool)
    sea_level = (
        float(np.nanmax(np.where(ocean, hydro["water_surface_level_m"], np.nan)))
        if ocean.any()
        else 0.0
    )
    z = np.asarray(hydro["elevation_m"], dtype=np.float64)
    depth = sea_level - z
    cont = ocean & ~marine
    out[cont] = CODE["continental_shelf"]
    out[cont & (depth > 1000.0)] = CODE["submerged_continental_plateau"]
    u = -np.asarray(T["crust_edge_distance_km"], dtype=np.float64)
    margin = np.asarray(T["margin_type"])
    age = np.nan_to_num(np.asarray(T["oceanic_crust_age_ma"], dtype=np.float64), nan=999.0)
    sed = np.asarray(T["marine_sediment_thickness_m"], dtype=np.float64)
    rough = np.asarray(T["marine_local_relief_m"], dtype=np.float64)
    sea = ocean & marine
    # deep floor first, then the more specific structures on top
    out[sea] = CODE["abyssal_hills"]
    # plains: where sediment buries the fabric - thick pelagic cover on old crust, or the
    # turbidite apron within a few hundred km beyond the rise of a passive margin
    ws_ = np.asarray(T["slope_width_km"], dtype=np.float64)
    wr_ = np.asarray(T["rise_width_km"], dtype=np.float64)
    apron = (margin == 1) | ((u >= 0.0) & (u < ws_ + wr_ + 400.0) & (wr_ > 100.0))
    out[sea & ((sed > 600.0) | apron | (rough < 10.0))] = CODE["abyssal_plain"]
    out[sea & (age < 15.0)] = CODE["mid_ocean_ridge"]
    slow = (
        np.nan_to_num(np.asarray(T["spreading_half_rate_mm_yr"], dtype=np.float64), nan=99.0) < 20.0
    )
    out[sea & (age < 2.5) & slow] = CODE["axial_rift_valley"]
    side = np.asarray(hist["trench_side"])
    td = np.asarray(hist["trench_distance_km"], dtype=np.float64)
    # back-arc / marginal sea: behind the arc, between the trench and a continent
    out[sea & (side == -1) & (td > 250.0) & (td < 900.0) & (u < 600.0)] = CODE["marginal_sea"]
    ws = np.asarray(T["slope_width_km"], dtype=np.float64)
    wr = np.asarray(T["rise_width_km"], dtype=np.float64)
    out[sea & (margin == 1) & (u >= ws) & (u < ws + wr)] = CODE["continental_rise"]
    out[sea & (margin > 0) & (u >= 0.0) & (u < ws)] = CODE["continental_slope"]
    out[sea & (side == -1) & (td <= 250.0)] = CODE["forearc"]
    # every fracture zone is in the overlay mask; a hex is *classed* as one only where a
    # major offset (>= 4 Ma of age jump, > 100 km at medium rates) makes it dominate the hex
    major = np.asarray(T.get("fracture_zone_offset_ma", np.zeros(shape))) >= 4.0
    out[sea & np.asarray(T["fracture_zone_mask"], dtype=bool) & major] = CODE["fracture_zone"]
    bclass = np.asarray(hist["boundary_class"])
    out[sea & (bclass == 4) & (age < 20.0)] = CODE["transform_fault"]
    out[sea & (np.asarray(T["oceanic_plateau_m"]) > 500.0)] = CODE["volcanic_plateau"]
    out[sea & np.asarray(T["outer_rise_mask"], dtype=bool)] = CODE["outer_rise"]
    out[ocean & np.asarray(T["trench_axis_mask"], dtype=bool)] = CODE["subduction_trench"]
    smt = np.asarray(T["seamount_type"])
    out[ocean & (smt == 1)] = CODE["seamount"]
    out[ocean & (smt == 2)] = CODE["guyot"]
    if "submarine_fan_m" in T:
        out[ocean & (np.asarray(T["submarine_fan_m"]) > 50.0)] = CODE["submarine_fan"]
    if "submarine_canyon_mask" in T:
        out[ocean & np.asarray(T["submarine_canyon_mask"], dtype=bool)] = CODE["submarine_canyon"]
    return out


def shares(codes: np.ndarray) -> dict:
    sea = codes != CODE["land"]
    sea &= codes != CODE["inland_water"]
    n = max(int(sea.sum()), 1)
    return {
        name: round(float((codes == i).sum()) / n, 4)
        for i, name in enumerate(PROVINCES)
        if name not in ("land", "inland_water") and (codes == i).any()
    }
