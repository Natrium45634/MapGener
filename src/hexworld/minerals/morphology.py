"""Morphology parameters (§4) and mass-conditioned geometry (§5).

``validate_params`` checks the required, typed parameters of each of the 20 morphologies
(T24: a stockwork vein fraction is a fraction). ``geometry_for_mass`` derives the sizes from
the sampled rock mass and shape ratios, so size and tonnage are never drawn independently
(T34): e.g. an ellipsoid ``a = (3V / (4 pi r_b r_c))^(1/3)``.
"""

from __future__ import annotations

import math

from .catalog import MORPHOLOGIES


class MorphologyError(ValueError):
    pass


PLATE = {"length_m", "down_dip_m", "true_thickness_m"}
REQUIRED = {
    "vein": PLATE, "tabular": PLATE, "stratabound": PLATE, "seam": PLATE, "laterite": PLATE,
    "placer": PLATE, "paleoplacer": PLATE, "quarry_unit": PLATE, "nodules": PLATE,
    "crust": PLATE, "reservoir": PLATE,
    "lens": {"semi_axes_m"}, "pod": {"semi_axes_m"}, "stockwork": {"semi_axes_m", "vein_fraction"},
    "disseminated": {"semi_axes_m"}, "breccia": {"semi_axes_m"}, "replacement": {"semi_axes_m"},
    "cavity_fill": {"semi_axes_m"}, "mantle_pipe": {"r_top_m", "r_bottom_m", "depth_m"},
    "vein_network": set(),
}  # fmt: skip
FRACTIONS = ("vein_fraction", "coverage", "net_to_gross", "porosity", "usable_fraction",
             "gem_fraction", "f_barren")  # fmt: skip


def validate_params(morphology: str, params: dict) -> None:
    if morphology not in MORPHOLOGIES:
        raise MorphologyError(f"unknown morphology {morphology!r}")
    missing = REQUIRED[morphology] - set(params)
    if missing:
        raise MorphologyError(f"{morphology}: missing {sorted(missing)}")
    for k in ("length_m", "down_dip_m", "true_thickness_m", "r_top_m", "depth_m"):
        if k in params and not float(params[k]) > 0:
            raise MorphologyError(f"{morphology}: {k} must be > 0")
    if "r_bottom_m" in params and float(params["r_bottom_m"]) < 0:
        raise MorphologyError(f"{morphology}: r_bottom_m must be >= 0")
    if "semi_axes_m" in params:
        ax = [float(v) for v in params["semi_axes_m"]]
        if len(ax) != 3 or min(ax) <= 0:
            raise MorphologyError(f"{morphology}: three positive semi-axes")
    for k in FRACTIONS:
        if k in params and not 0.0 <= float(params[k]) <= 1.0:
            raise MorphologyError(f"{morphology}: {k} must be within [0, 1]")
    zones = params.get("zones")
    if zones is not None:
        tot = sum(float(z["fraction"]) for z in zones)
        if abs(tot - 1.0) > 1e-9:
            raise MorphologyError(f"{morphology}: zone fractions must sum to 1, got {tot}")


#: default shape ratios per morphology: plate (L : W : t), ellipsoid (a : b : c),
#: pipe (r_bottom / r_top, depth / r_top) [К] prototype profile, not statistics
DEFAULT_RATIOS = {
    "vein": (1.0, 0.4, 0.004), "tabular": (1.0, 0.8, 0.01), "stratabound": (1.0, 0.7, 0.01),
    "seam": (1.0, 0.8, 0.0015), "laterite": (1.0, 0.8, 0.005), "placer": (1.0, 0.05, 0.002),
    "paleoplacer": (1.0, 0.08, 0.003), "quarry_unit": (1.0, 0.7, 0.03),
    "nodules": (1.0, 1.0, 1e-5), "crust": (1.0, 0.5, 2e-5), "reservoir": (1.0, 0.6, 0.004),
    "lens": (1.0, 0.5, 0.15), "pod": (1.0, 0.7, 0.4), "stockwork": (1.0, 0.8, 0.7),
    "disseminated": (1.0, 0.8, 0.6), "breccia": (1.0, 0.8, 0.8),
    "replacement": (1.0, 0.6, 0.3), "cavity_fill": (1.0, 0.6, 0.4),
    "mantle_pipe": (0.25, 4.0), "vein_network": (1.0, 0.4, 0.004),
}  # fmt: skip


def geometry_for_mass(morphology: str, rock_t: float, rho_kg_m3: float, ratios=None) -> dict:
    """Sizes from the mass: ``V = T * 1000 / rho`` and the shape ratios."""
    if rock_t <= 0 or rho_kg_m3 <= 0:
        raise MorphologyError("mass and density must be > 0")
    v = rock_t * 1000.0 / rho_kg_m3
    r = (ratios or {}).get(morphology) or DEFAULT_RATIOS[morphology]
    if morphology == "mantle_pipe":
        kb, kh = r if len(r) == 2 else DEFAULT_RATIOS["mantle_pipe"]
        rt = (3.0 * v / (math.pi * kh * (1.0 + kb + kb * kb))) ** (1.0 / 3.0)
        return {"r_top_m": rt, "r_bottom_m": kb * rt, "depth_m": kh * rt}
    if ratios and "aspect" in ratios and morphology != "mantle_pipe":
        r = ratios["aspect"]
    a, b, c = (float(x) for x in r)
    from .bodies import PRIMITIVE

    if PRIMITIVE[morphology] in ("plate", "network"):
        k = (v / (a * b * c)) ** (1.0 / 3.0)
        return {"length_m": a * k, "down_dip_m": b * k, "true_thickness_m": c * k}
    k = (3.0 * v / (4.0 * math.pi * a * b * c)) ** (1.0 / 3.0)
    out = {"semi_axes_m": [a * k, b * k, c * k]}
    if morphology == "stockwork":
        out["vein_fraction"] = 0.08
    return out
