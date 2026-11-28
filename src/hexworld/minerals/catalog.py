"""The data-defined catalogue of TZ 5.5.1 (§1): 194 commodity records and 92 genetic templates.

Loaded from ``minerals/data/*.json`` and validated (unique IDs, known references, known
morphologies and quantity schemas). New commodities are added as data, never by a switch on
a name. Groups and varieties (REE, PGE, ruby of corundum, ...) are derived views of a parent
physical lot (``shared_inventory_basis``), never additive to it.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

DATA_DIR = Path(__file__).parent / "data"

#: the 21 morphologies of §4 (bodies.py implements each)
MORPHOLOGIES = (
    "vein",
    "vein_network",
    "stockwork",
    "disseminated",
    "lens",
    "tabular",
    "stratabound",
    "mantle_pipe",
    "replacement",
    "breccia",
    "pod",
    "laterite",
    "placer",
    "paleoplacer",
    "quarry_unit",
    "nodules",
    "crust",
    "reservoir",
    "seam",
    "cavity_fill",
)  # fmt: skip  (cavity_fill closes the list; «tabular» covers layers and slabs)
#: quantity schemas of the commodity records -> amount unit and grade units allowed
QUANTITY_SCHEMAS = {
    "mass_fraction_and_contained_t": ("t", ("fraction", "percent", "ppm", "kg_per_t")),
    "grade_g_per_t_and_contained_kg": ("kg", ("g_per_t", "ppm", "ppb", "g_per_m3")),
    "rough_kg_and_quality_fraction_and_carat_if_crystal": ("kg", ("fraction", "ct_per_t")),
    "grade_ct_per_t_or_ct_per_m3_and_rough_ct": ("ct", ("ct_per_t", "ct_per_m3")),
    "usable_volume_m3_dry_mass_t_and_usage_quality": ("m3", ("fraction",)),
    "phase_mass_kg_and_standard_volume_m3_if_defined": ("kg", ("kg_per_m3", "fraction")),
    "dissolved_mg_per_L_and_contained_kg": ("kg", ("mg_per_L",)),
    "water_m3_and_recharge_m3_per_year": ("m3", ()),
    "heat_in_place_J_and_recharge_W": ("J", ()),
}
CATEGORIES = (
    "ferrous", "nonferrous", "precious", "rare", "radioactive", "gem", "ornamental",
    "construction", "industrial", "chemical_agro", "fuel", "water_heat",
)  # fmt: skip


class CatalogError(ValueError):
    pass


@dataclass(frozen=True)
class Commodity:
    id: str
    label_ru: str
    category: str
    model_ids: tuple
    quantity_schema: str
    basis: str  # the physical inventory it is a view of
    not_additive_with_parent: bool
    priority: str

    @property
    def amount_unit(self) -> str:
        return QUANTITY_SCHEMAS[self.quantity_schema][0]

    @property
    def grade_units(self) -> tuple:
        return QUANTITY_SCHEMAS[self.quantity_schema][1]


@dataclass(frozen=True)
class ModelTemplate:
    id: str
    label_ru: str
    morphologies: tuple
    hard_gates_ru: tuple
    principal: tuple
    associated: tuple
    priority: str
    calibration_status: str
    research_family: str


@dataclass(frozen=True)
class Catalog:
    commodities: dict  # id -> Commodity
    models: dict  # id -> ModelTemplate
    schema_versions: tuple
    inventory_rule: str

    def models_for(self, commodity_id: str) -> list:
        return [self.models[m] for m in self.commodities[commodity_id].model_ids]

    def commodities_of(self, model_id: str) -> list:
        m = self.models[model_id]
        return [self.commodities[c] for c in (*m.principal, *m.associated)]

    def group_members(self, basis: str) -> list:
        """Views of one physical inventory (REE elements, corundum varieties, ...)."""
        return sorted(c.id for c in self.commodities.values() if c.basis == basis)


def validate(commodities_doc: dict, models_doc: dict) -> Catalog:
    """Schema checks (T04): duplicate IDs, unknown references, morphologies, quantity
    schemas, categories, priorities. Raises :class:`CatalogError`."""
    models = {}
    for rec in models_doc.get("models", []):
        mid = rec.get("id")
        if not mid or mid in models:
            raise CatalogError(f"duplicate or empty model id: {mid!r}")
        bad = [m for m in rec.get("morphologies", []) if m not in MORPHOLOGIES]
        if bad or not rec.get("morphologies"):
            raise CatalogError(f"{mid}: unknown or no morphology {bad}")
        if rec.get("priority") not in ("P0", "P1", "P2"):
            raise CatalogError(f"{mid}: priority must be P0/P1/P2")
        if not rec.get("principal"):
            raise CatalogError(f"{mid}: no principal commodity")
        models[mid] = ModelTemplate(
            mid,
            rec.get("label_ru", mid),
            tuple(rec["morphologies"]),
            tuple(rec.get("hard_gate_descriptions", ())),
            tuple(rec["principal"]),
            tuple(rec.get("conditional_associated", ())),
            rec["priority"],
            rec.get("calibration_status") or "provisional",
            rec.get("research_family", ""),
        )
    coms = {}
    for rec in commodities_doc.get("commodities", []):
        cid = rec.get("id")
        if not cid or cid in coms:
            raise CatalogError(f"duplicate or empty commodity id: {cid!r}")
        if rec.get("quantity_schema") not in QUANTITY_SCHEMAS:
            raise CatalogError(f"{cid}: unknown quantity schema {rec.get('quantity_schema')!r}")
        if rec.get("category") not in CATEGORIES:
            raise CatalogError(f"{cid}: unknown category {rec.get('category')!r}")
        unknown = [m for m in rec.get("model_ids", []) if m not in models]
        if unknown or not rec.get("model_ids"):
            raise CatalogError(f"{cid}: unknown or no model reference {unknown}")
        coms[cid] = Commodity(
            cid,
            rec.get("label_ru", cid),
            rec["category"],
            tuple(rec["model_ids"]),
            rec["quantity_schema"],
            rec.get("shared_inventory_basis") or cid,
            bool(rec.get("not_additive_with_parent", False)),
            rec.get("priority", "P0"),
        )
    for mid, m in models.items():
        missing = [c for c in (*m.principal, *m.associated) if c not in coms]
        if missing:
            raise CatalogError(f"{mid}: unknown commodity {missing}")
    return Catalog(
        coms,
        models,
        (commodities_doc.get("schema_version", ""), models_doc.get("schema_version", "")),
        commodities_doc.get("inventory_rule", ""),
    )


@lru_cache(maxsize=1)
def load() -> Catalog:
    with open(DATA_DIR / "commodities.json", encoding="utf-8") as fh:
        c = json.load(fh)
    with open(DATA_DIR / "deposit_models.json", encoding="utf-8") as fh:
        m = json.load(fh)
    return validate(c, m)


#: 5.2 model -> 5.5.1 templates (audit concordance; old saves keep their 5.2 IDs)
LEGACY_TO_NEW = {
    "R01": ("PORPHYRY",), "R02": ("EPITHERMAL",), "R03": ("SKARN",), "R04": ("OROGENIC_AU",),
    "R05": ("VMS",), "R06": ("SEDEX",), "R07": ("SED_CU",), "R08": ("NI_SULFIDE", "PGE_REEF"),
    "R09": ("CHROM_POD", "CHROM_LAYER"), "R10": ("LCT_PEG",), "R11": ("CARBONATITE",),
    "R12": ("GREISEN",), "R13": ("U_ROLL",), "R14": ("U_UNCONF",), "R15": ("BIF",),
    "R16": ("FE_OOLITE", "MN_SED"), "R17": ("BAUXITE",), "R18": ("NI_LATERITE",),
    "R19": ("REE_CLAY",), "R20": ("PLACER_RIVER",), "R21": ("PLACER_COAST",),
    "R22": ("KIMBERLITE",), "R23": ("GRAPHITE",), "R24": ("COAL",), "R25": ("PETROLEUM",),
    "R26": ("EVAPORITE",), "R27": ("BRINE",), "R28": ("PHOSPHORITE", "MARINE_PHOS"),
    "R29": ("SMS",), "R30": ("MARINE_NODULE",), "R31": ("MARINE_CRUST",), "R32": ("SULFUR",),
    "R33": ("QUARRY_IGNEOUS", "QUARRY_SED", "QUARRY_META", "SAND_GRAVEL", "CLAY"),
    "R34": ("GEM_PEG", "GEM_META", "GEM_MAFIC", "CORUNDUM_META", "CORUNDUM_MAG",
            "EMERALD_META", "EMERALD_SED", "SILICA_GEODE", "OPAL"),
    "R35": ("GEOTHERMAL",),
}  # fmt: skip
