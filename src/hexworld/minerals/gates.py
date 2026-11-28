"""Hard gates and soft factors of the 92 templates (§3).

``eligible`` = every hard gate present; a gate that is absent (0) forbids the type, a gate
whose input is missing (NaN) makes the hex **unresolved** (never false, never true).
``S = eligible * prod(s_k ** w_k)`` over the soft factors (a dimensionless suitability, not
a probability). Intensities ``lam_ref`` (systems per km² of S-weighted area) are
[К] provisional frequency classes, not calibrated: the TZ package ships no distributions.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .catalog import load
from .features import FEATURES_RU

#: provisional frequency classes, systems per km² of suitability-weighted area [К]
COMMON, MODERATE, RARE, VERY_RARE, EXCEPTIONAL = 2e-5, 5e-6, 1e-6, 2e-7, 4e-8


@dataclass(frozen=True)
class Rule:
    gates: tuple  # hard: every feature must be 1
    soft: tuple = ()  # ((feature, weight), ...): s = 1 if present else floor
    lam_ref: float = RARE
    anchor: str = "tract"  # tract | object | secondary | regional
    age: str = "simulated"  # simulated | inherited | any
    floor: float = 0.35  # soft factor value where the feature is absent


R = Rule
RULES = {
    # -- iron, manganese, chromium, nickel, PGE ----------------------------------------------
    "BIF": R(("ancient_crust", "marine_facies", "preserved_strata"), (("ancient_shield", 1.0),),
             MODERATE, age="inherited"),
    "FE_OOLITE": R(("sedimentary_basin", "marine_facies"), (("weathering_any", 0.5),), RARE),
    "BOG_IRON": R(("land", "wetland", "weathering_any"), (), MODERATE),
    "FE_TI_V": R(("layered_intrusion",), (), RARE, age="any"),
    "MN_SED": R(("sedimentary_basin", "marine_facies"), (("reduced_facies", 1.0),), RARE),
    "CHROM_LAYER": R(("layered_intrusion",), (), RARE, age="any"),
    "CHROM_POD": R(("ophiolite",), (), MODERATE),
    "NI_SULFIDE": R(("mafic_ultramafic",), (("ancient_crust", 0.5),), RARE, age="any"),
    "PGE_REEF": R(("layered_intrusion",), (("ancient_crust", 1.0),), VERY_RARE, age="any"),
    "NI_LATERITE": R(("ultramafic", "weathering_long", "land"), (), MODERATE, "secondary"),
    "BAUXITE": R(("al_silicate_rock", "weathering_long", "land"), (), MODERATE, "secondary"),
    "BAUXITE_KARST": R(("karst_surface", "weathering_any"), (), RARE, "secondary"),
    # -- base, precious and rare metals -----------------------------------------------------
    "PORPHYRY": R(("arc", "magmatic_activity"), (("felsic_intrusion", 0.5),), MODERATE, "object"),
    "EPITHERMAL": R(("magmatic_activity", "volcanic_rock", "hydrothermal"), (), MODERATE, "object"),
    "OROGENIC_AU": R(("shear_zone", "metamorphic"), (("ancient_crust", 0.3),), MODERATE),
    "CARLIN": R(("carbonate", "hydrothermal"), (("shear_zone", 0.5),), RARE),
    "IOCG": R(("hydrothermal", "land"), (("ancient_crust", 1.0),), VERY_RARE, age="any"),
    "SKARN": R(("carbonate_near_intrusion",), (), MODERATE),
    "VMS": R(("mafic", "marine_facies"), (("greenstone", 0.7),), MODERATE, age="any"),
    "SEDEX": R(("sedimentary_basin", "reduced_facies"), (("evaporitic", 0.5),), RARE),
    "MVT": R(("carbonate", "sedimentary_basin"), (), RARE),
    "SED_CU": R(("sedimentary_basin", "clastic_permeable"), (("reduced_facies", 0.7),), RARE),
    "POLY_VEIN": R(("hydrothermal",), (("felsic_intrusion", 0.5),), MODERATE),
    "GREISEN": R(("felsic_intrusion",), (("metamorphic", 0.3),), RARE),
    "LCT_PEG": R(("pegmatite",), (), RARE, age="any"),
    "NYF_PEG": R(("pegmatite",), (("alkaline_magmatism", 0.7),), VERY_RARE, age="any"),
    "CARBONATITE": R(("alkaline_magmatism", "land"), (("ancient_crust", 0.7),), VERY_RARE),
    "ALKALINE": R(("alkaline_magmatism",), (), RARE),
    "REE_CLAY": R(("felsic_intrusion", "weathering_long"), (), RARE, "secondary"),
    "U_ROLL": R(("sedimentary_basin", "clastic_permeable"), (("felsic_intrusion", 0.3),), RARE),
    "U_UNCONF": R(("unconformity",), (), VERY_RARE, age="inherited"),
    "U_VEIN": R(("felsic_intrusion", "hydrothermal"), (), RARE),
    "SB_HG": R(("hydrothermal",), (("magmatic_activity", 0.5),), RARE),
    # -- placers (secondary: need a parent source; the routing decides, see placers.py) ----
    "PLACER_RIVER": R(("river", "land"), (), MODERATE, "secondary"),
    "PLACER_COAST": R(("coast",), (), MODERATE, "secondary"),
    "PALEOPLACER": R(("sedimentary_basin", "ancient_crust"), (), VERY_RARE, "secondary",
                     "inherited"),
    # -- gems --------------------------------------------------------------------------------
    "KIMBERLITE": R(("ancient_shield", "land"), (), RARE, age="inherited"),
    "CORUNDUM_META": R(("metamorphic",), (("carbonate", 0.7),), VERY_RARE),
    "CORUNDUM_MAG": R(("alkaline_magmatism",), (), VERY_RARE),
    "EMERALD_META": R(("be_source_near", "cr_host_near", "metamorphic"), (), EXCEPTIONAL),
    "EMERALD_SED": R(("reduced_facies", "shear_zone"), (("evaporitic", 0.5),), EXCEPTIONAL),
    "GEM_PEG": R(("pegmatite",), (), VERY_RARE, age="any"),
    "GEM_META": R(("metamorphic",), (), VERY_RARE),
    "GEM_MAFIC": R(("mafic",), (("alkaline_magmatism", 0.5),), VERY_RARE),
    "JADEITE": R(("high_p_low_t",), (), EXCEPTIONAL),
    "NEPHRITE": R(("ultramafic", "metamorphic"), (), VERY_RARE),
    "SILICA_GEODE": R(("volcanic_rock",), (), RARE),
    "OPAL": R(("weathering_any", "sedimentary_rock"), (("arid", 0.7),), VERY_RARE),
    "CU_OXID": R(("copper_body",), (("arid", 0.5),), MODERATE, "secondary"),
    "LAPIS": R(("carbonate", "metamorphic"), (), EXCEPTIONAL),
    "CHAROITE": R(("alkaline_magmatism", "metamorphic"), (), EXCEPTIONAL),
    "AMBER": R(("sedimentary_basin", "organic_burial"), (("coast", 0.3),), VERY_RARE),
    "FOSSIL_ORG": R(("sedimentary_basin", "organic_burial"), (), RARE),
    "VOLCANIC_GLASS": R(("volcanic_rock", "magmatic_activity"), (), MODERATE, "object"),
    # -- construction and industrial (regional units of real lithology volumes) ------------
    "QUARRY_IGNEOUS": R(("igneous_rock", "land"), (), COMMON, "regional"),
    "QUARRY_SED": R(("sedimentary_rock", "land"), (), COMMON, "regional"),
    "QUARRY_META": R(("metamorphic", "land"), (), COMMON, "regional"),
    "SAND_GRAVEL": R(("land", "river"), (("coast", 0.5),), COMMON, "regional"),
    "CLAY": R(("land", "sedimentary_rock"), (("aquatic_basin", 0.5),), COMMON, "regional"),
    "KAOLIN": R(("al_silicate_rock", "weathering_any"), (), MODERATE),
    "BENTONITE": R(("volcanic_ash",), (("sedimentary_basin", 0.5),), RARE),
    "ZEOLITE": R(("volcanic_rock",), (("closed_basin", 0.5),), RARE),
    "DIATOMITE": R(("aquatic_basin",), (("volcanic_ash", 0.5),), RARE),
    "GRAPHITE": R(("metamorphic", "reduced_facies"), (), RARE),
    "TALC": R(("mg_rock", "metamorphic"), (), RARE),
    "REFRACTORY": R(("metamorphic", "al_silicate_rock"), (), RARE),
    "WOLLAST": R(("carbonate_near_intrusion",), (), VERY_RARE),
    "MICA_FELDSPAR": R(("pegmatite",), (), MODERATE, age="any"),
    "BARITE_FLUOR": R(("hydrothermal",), (("sedimentary_basin", 0.5),), RARE),
    "VERMICULITE": R(("mafic_ultramafic", "weathering_any"), (), RARE),
    "PHOSPHORITE": R(("marine_facies", "sedimentary_basin"), (("upwelling", 1.0),), RARE),
    "APATITE": R(("alkaline_magmatism",), (), VERY_RARE),
    "EVAPORITE": R(("evaporitic",), (("arid", 0.5),), MODERATE),
    "BORATE_SODA": R(("closed_basin", "arid"), (("magmatic_activity", 0.7),), VERY_RARE),
    "SULFUR": R(("s_source",), (), RARE),
    "PIGMENTS": R(("weathering_any", "sedimentary_rock"), (), RARE),
    # -- fuels, fluids, water, heat (P1) ----------------------------------------------------
    "COAL": R(("organic_burial",), (("deep_burial", 0.5),), MODERATE),
    "PEAT": R(("wetland", "land"), (), COMMON, "regional"),
    "OIL_SHALE": R(("reduced_facies", "sedimentary_basin"), (), RARE),
    "PETROLEUM": R(("sedimentary_basin", "deep_burial", "porous_host", "seal_rock"), (),
                   MODERATE),
    "UNCONVENTIONAL": R(("reduced_facies", "deep_burial"), (), RARE),
    "HYDRATE": R(("cold_seafloor",), (), RARE),
    "BRINE": R(("closed_basin", "arid"), (("evaporitic", 0.7),), RARE),
    "AQUIFER": R(("porous_host", "land"), (), COMMON, "regional"),
    "GEOTHERMAL": R(("heat_source",), (), MODERATE, "object"),
    "MARINE_NODULE": R(("deep_ocean", "low_sedimentation", "old_ocean_floor"), (), MODERATE),
    "MARINE_CRUST": R(("hard_substrate",), (), MODERATE),
    "SMS": R(("submarine_hydrothermal",), (), MODERATE, "object"),
    "MARINE_PHOS": R(("shelf",), (("upwelling", 1.0),), RARE),
    "CELESTINE": R(("evaporitic", "sedimentary_basin"), (), VERY_RARE),
    "MAGNESITE": R(("mg_rock",), (("metamorphic", 0.5),), RARE),
    "FELDSPAR_GEM": R(("feldspathic_rock",), (("pegmatite", 0.5),), VERY_RARE),
}  # fmt: skip


def check_rules() -> list[str]:
    """Every catalogue template has a rule, every rule names known features."""
    cat = load()
    errs = [f"no rule for {m}" for m in cat.models if m not in RULES]
    errs += [f"rule for unknown template {m}" for m in RULES if m not in cat.models]
    for mid, r in RULES.items():
        for f in (*r.gates, *(k for k, _w in r.soft)):
            if f not in FEATURES_RU:
                errs.append(f"{mid}: unknown feature {f}")
        if not r.gates:
            errs.append(f"{mid}: no hard gate")
    return errs


def evaluate(rule: Rule, features: dict):
    """``(eligible, unresolved, S)`` per hex.

    eligible: every gate == 1. unresolved: no gate == 0 and at least one gate NaN (missing
    input; never treated as eligible). S: eligible × ∏ s_k^w_k (0 where not eligible).
    """
    gates = np.stack([np.asarray(features[g], dtype=np.float64) for g in rule.gates])
    absent = np.any(gates == 0.0, axis=0)
    missing = np.any(np.isnan(gates), axis=0)
    eligible = ~absent & ~missing
    unresolved = ~absent & missing
    s = np.ones(gates.shape[1])
    for feat, w in rule.soft:
        v = np.asarray(features[feat], dtype=np.float64)
        sk = np.where(v == 1.0, 1.0, rule.floor)  # missing soft factor: the floor, flagged
        s *= sk**w
    return eligible, unresolved, np.where(eligible, s, 0.0)


def gate_report(rule: Rule, features: dict, i: int) -> dict:
    """Why hex ``i`` is (not) eligible: each gate present / absent / missing."""
    out = {}
    for g in rule.gates:
        v = float(np.asarray(features[g])[i])
        out[g] = "missing" if np.isnan(v) else ("present" if v == 1.0 else "absent")
    return out
