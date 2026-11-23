"""Why a land class is absent from a world (TZ 5.3 appendix B): realised hexes + a reason.

A class can be fully implemented and still absent: the reason says which condition the
world lacks. Envelopes are coarse keyword groups of the class ids; inside its envelope an
absent class is ``too_rare``.
"""

from __future__ import annotations

import numpy as np

from .registry import BIOMES

LAND_FAMILIES_EXCLUDED = ("marine", "freshwater")

GROUPS = (
    (("volcanic",), "volcanism", "requires_component_disabled"),
    (("tropical", "mangrove", "savanna", "thorn", "paramo"), "tropical", "climate_not_generated"),
    (("tundra", "polar", "glacier", "firn", "snow", "nunatak"), "cold", "climate_not_generated"),
    (("desert", "sabkha", "takyr", "playa", "salt_flat", "gypsum"), "arid",
     "climate_not_generated"),
    (("wetland", "swamp", "bog", "fen", "marsh", "peat", "floodplain"), "wet",
     "hydrology_absent"),
    (("alpine", "montane", "scree", "subalpine"), "mountain", "substrate_absent"),
    (("coastal", "dune", "saltmarsh", "intertidal"), "coast", "substrate_absent"),
    (("heath",), "oceanic", "climate_not_generated"),
)  # fmt: skip


def class_audit(shares, envelopes: dict) -> dict:
    """``shares`` (N_B,) land area share of every class; ``envelopes`` name -> bool."""
    out = {}
    for i, b in enumerate(BIOMES):
        if b.family in LAND_FAMILIES_EXCLUDED or b.id == "unresolved":
            continue
        realised = float(shares[i]) > 0
        reason = None
        if not realised:
            reason = "too_rare"
            for words, env, why in GROUPS:
                if any(w in b.id for w in words):
                    if not envelopes.get(env, True):
                        reason = why
                    break
        out[b.id] = {"realised_share": round(float(shares[i]), 6), "absence_reason": reason}
    return out


def envelopes_from(tc, tw, ai, waterlog, z, coast, volcanism) -> dict:
    tc, tw = np.asarray(tc), np.asarray(tw)
    return {
        "tropical": bool((tc > 15.0).any()),
        "cold": bool((tw < 10.0).any()),
        "arid": bool((np.nan_to_num(np.asarray(ai), nan=9.0) < 0.2).any()),
        "wet": bool((np.asarray(waterlog) >= 6).any()),
        "mountain": bool((np.asarray(z) > 1500.0).any()),
        "coast": bool(np.asarray(coast).any()),
        "oceanic": bool((((tw - tc) < 13.0) & (tc > -3.0) & (tw < 20.0)).any()),
        "volcanism": bool(volcanism),
    }
