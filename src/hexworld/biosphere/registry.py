"""BioRegistry of TZ 5.5 §4: functional groups → taxon templates → procedural taxa.

* ``FunctionalGroup``: one of the 94 groups of the TZ package (names, realm, role kept
  verbatim) with the quantitative traits of ``traits_v1.json`` (provisional priors).
* ``TaxonTemplate``: an ecological analogue inside a group — a thermal × moisture variant
  of its niche (e.g. «сосноподобное дерево, холодостойкий»). Templates are not species.
* ``ProceduralTaxon``: a template in one lineage pool (a source landmass, drainage or ocean
  province). Its id is a digest of the namespace ``bio_registry_v1``, the biology seed
  (case-sensitive), the template id and the stable lineage key — never of the traversal
  order (B05, B06).

Richness counts unique leaf taxa; a group is never counted as a taxon (B10).
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

DATA = Path(__file__).resolve().parent / "data"
REGISTRY_VERSION = "hexworld-bio-registry-1"

#: food pools produced directly by the producers (by layer) and by the physics
PRIMARY_POOLS = {
    "leaf_woody": "листва и побеги деревьев и кустарников",
    "leaf_herb": "травянистая зелень",
    "fruit_seed": "плоды и семена",
    "nectar": "нектар и пыльца",
    "litter": "опад и мёртвая органика на суше",
    "root_exudate": "корневые выделения (углерод для симбионтов)",
    "fw_macro": "водные растения",
    "fw_phyto": "пресноводный фитопланктон",
    "fw_periphyton": "обрастания дна",
    "fw_detritus": "детрит водоёмов",
    "m_phyto": "морской фитопланктон",
    "m_ice_algae": "ледовые водоросли",
    "m_macro": "макроводоросли, морские травы, рифовые продуценты",
    "m_seagrass": "морские травы",
    "m_chemo": "хемосинтетическая продукция источников",
    "m_detritus": "детрит толщи воды",
    "m_benthic_flux": "оседающий на дно органический поток",
    "carrion": "падаль",
}
LAYER_POOL = {"canopy": "leaf_woody", "shrub": "leaf_woody", "epiphyte": "leaf_woody",
              "ground": "leaf_herb", "fw_macro": "fw_macro", "fw_plankton": "fw_phyto",
              "fw_benthic": "fw_periphyton", "m_plankton": "m_phyto", "m_ice": "m_ice_algae",
              "m_benthic": "m_macro", "chemo": "m_chemo"}  # fmt: skip
TERRESTRIAL_LAYERS = ("canopy", "shrub", "epiphyte", "ground")
KINGDOM_RU = {"plant": "растения", "alga": "водоросли", "symbiosis": "лишайники",
              "microbe": "микроорганизмы", "fungus": "грибы", "animal": "животные"}  # fmt: skip
FLORA_KINDS = ("plant", "alga", "symbiosis")
FAUNA_KINDS = ("animal",)
GATE_WORDS = {
    "land_ground", "root_zone", "tidal_coast", "canopy", "wet_ground", "inland_water",
    "shallow_water", "river", "marine", "lit_bottom", "soft_bottom", "hard_bottom", "sea_ice",
    "vent", "host_plants", "soil", "flowers", "herb_cover", "woody_cover", "large_land",
    "any_water", "any_fresh", "shelf", "deep", "deep_shelf", "coast", "hosts", "anadromous",
    "nest_coast", "haulout",
}  # fmt: skip
NICHE_VARS = ("tc", "tw", "mi", "gdd", "ph", "sal")


@dataclass(frozen=True)
class FunctionalGroup:
    id: str
    name_ru: str
    realm: str
    trophic_role: str
    analogue_ru: str
    mobility_mode: str
    habitat_hint: str
    traits: dict
    parameter_status: str = "provisional"

    @property
    def kind(self) -> str:
        return self.traits["kind"]

    @property
    def level(self) -> int:
        return int(self.traits["level"])

    @property
    def is_producer(self) -> bool:
        return self.level == 1

    @property
    def is_flora(self) -> bool:
        return self.kind in FLORA_KINDS

    @property
    def is_fauna(self) -> bool:
        return self.kind in FAUNA_KINDS

    @property
    def parent(self) -> str:
        """Parent node of the group in the registry tree (kingdom / layer or level)."""
        if self.is_producer:
            return f"{self.kind}/{self.traits['layer']}"
        return f"{self.kind}/level{self.level}"


@dataclass(frozen=True)
class TaxonTemplate:
    id: str
    group_id: str
    name_ru: str
    niche: dict  # var -> [a, b, c, d]
    index: tuple  # (thermal i, moisture j)


@dataclass
class BioRegistry:
    version: str
    groups: dict
    templates: dict
    pools: dict  # pool -> {"providers": [...], "consumers": [...]}
    source_sha256: dict = field(default_factory=dict)

    def templates_of(self, group_id: str) -> list:
        return [t for t in self.templates.values() if t.group_id == group_id]

    def taxon_id(self, seed: str, template_id: str, lineage_key: str) -> str:
        return taxon_id(seed, template_id, lineage_key)


def taxon_id(seed: str, template_id: str, lineage_key: str) -> str:
    """Stable id of a procedural taxon (namespace ``bio_registry_v1``, seed case-sensitive)."""
    h = hashlib.sha256(f"bio_registry_v1|{seed}|{template_id}|{lineage_key}".encode("ascii",
                       "backslashreplace")).hexdigest()  # fmt: skip
    return "tx" + h[:12]


def _split_niche(niche: dict, var: str, n: int, i: int) -> list:
    """The i-th of n overlapping sub-bands of the plateau of ``var`` (a template's niche)."""
    a, b, c, d = niche[var]
    if n <= 1 or c <= b:
        return [a, b, c, d]
    w = (c - b) / n
    bi, ci = b + i * w, b + (i + 1) * w
    slope = max(0.5 * w, 1e-6)
    ai = max(a, bi - max(slope, (b - a) if i == 0 else slope))
    di = min(d, ci + max(slope, (d - c) if i == n - 1 else slope))
    if i == 0:
        ai = a
    if i == n - 1:
        di = d
    return [ai, bi, ci, di]


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


@lru_cache(maxsize=1)
def load() -> BioRegistry:
    base_p = DATA / "functional_groups_tz55.json"
    tr_p = DATA / "traits_v1.json"
    base = json.loads(base_p.read_text(encoding="utf-8"))
    tr = json.loads(tr_p.read_text(encoding="utf-8"))
    qt = {int(k): v for k, v in tr["qualifiers_thermal"].items()}
    qm = {int(k): v for k, v in tr["qualifiers_moisture"].items()}
    groups, templates = {}, {}
    for g in base["groups"]:
        t = tr["traits"][g["id"]]
        fg = FunctionalGroup(g["id"], g["name_ru"], g["realm"], g["trophic_role"],
                             g["example_analogue_ru"], g["mobility_mode"], g["habitat_hint"],
                             t, "provisional")  # fmt: skip
        groups[fg.id] = fg
        var, n = t["split"][0], int(t["split"][1])
        var2, n2 = t["split2"][0], int(t["split2"][1])
        for i in range(n):
            for j in range(n2):
                niche = {k: list(v) for k, v in t["niche"].items()}
                niche[var] = _split_niche(t["niche"], var, n, i)
                if n2 > 1 and var2 in niche:
                    niche[var2] = _split_niche(t["niche"], var2, n2, j)
                qual = " ".join(q for q in (qt[n][i], qm[n2][j]) if q)
                name = fg.analogue_ru + (f" ({qual} вариант)" if qual else "")
                tid = f"{fg.id}.{i}{j}"
                templates[tid] = TaxonTemplate(tid, fg.id, name, niche, (i, j))
    pools: dict = {p: {"providers": [], "consumers": []} for p in PRIMARY_POOLS}
    for fg in groups.values():
        t = fg.traits
        if fg.is_producer:
            pools[LAYER_POOL[t["layer"]]]["providers"].append(fg.id)
        else:
            for p in t["diet"]:
                pools.setdefault(p, {"providers": [], "consumers": []})["consumers"].append(fg.id)
            if t.get("provides"):
                pools.setdefault(t["provides"], {"providers": [], "consumers": []})[
                    "providers"].append(fg.id)  # fmt: skip
    return BioRegistry(REGISTRY_VERSION, groups, templates, pools,
                       {base_p.name: _sha(base_p), tr_p.name: _sha(tr_p)})  # fmt: skip


def pool_provider_level(reg: BioRegistry, pool: str) -> int:
    """Highest trophic level that feeds a pool (1 = producers and physics)."""
    prov = reg.pools.get(pool, {}).get("providers", [])
    if pool in ("litter", "fw_detritus", "m_detritus", "m_benthic_flux", "root_exudate",
                "nectar", "fruit_seed", "m_seagrass"):  # fmt: skip
        return max([1] + [reg.groups[g].level for g in prov])
    if pool == "carrion":
        return max(g.level for g in reg.groups.values()
                   if g.is_fauna and g.level < 6 and g.traits["mass_kg"] >= 0.5)  # fmt: skip
    return max([1] + [reg.groups[g].level for g in prov])


def validate(reg: BioRegistry | None = None) -> list[str]:
    """Registry checks of B08: unique ids, an acyclic tree, every diet edge resolved to a
    pool with providers strictly below the consumer, trait units and ranges."""
    reg = reg or load()
    errs = []
    ids = list(reg.groups) + list(reg.templates)
    if len(ids) != len(set(ids)):
        errs.append("duplicate ids")
    # tree: template -> group -> parent node -> kingdom; parent of a node never its child
    for t in reg.templates.values():
        if t.group_id not in reg.groups:
            errs.append(f"{t.id}: unknown group")
    for g in reg.groups.values():
        tr = g.traits
        if tr["kind"] not in KINGDOM_RU:
            errs.append(f"{g.id}: kind {tr['kind']}")
        if g.parent.split("/")[0] == g.id:
            errs.append(f"{g.id}: cyclic parent")
        for w in tr["gates"]:
            if w not in GATE_WORDS:
                errs.append(f"{g.id}: gate {w}")
        for k, v in tr["niche"].items():
            if k not in NICHE_VARS or len(v) != 4 or not v[0] <= v[1] <= v[2] <= v[3]:
                errs.append(f"{g.id}: niche {k} {v}")
        a, b, c, d = tr["t_act"]
        if not a <= b <= c <= d:
            errs.append(f"{g.id}: t_act")
        if not 0 < float(tr["L_km"]) or not 0 < float(tr["e0"]) < 1:
            errs.append(f"{g.id}: dispersal")
        if g.is_producer:
            al = tr["alloc"]
            if abs(sum(al.values()) - 1.0) > 1e-9 or min(al.values()) < 0:
                errs.append(f"{g.id}: allocation does not sum to 1")
            for p, f in al.items():
                if f > 0 and tr["turn"][p] <= 0:
                    errs.append(f"{g.id}: turnover of {p} must be > 0")
            if tr["layer"] not in LAYER_POOL:
                errs.append(f"{g.id}: layer")
            if not 0 < tr["cue"] <= 1 or tr["lue"] < 0:
                errs.append(f"{g.id}: lue/cue")
        else:
            if abs(sum(tr["diet"].values()) - 1.0) > 1e-9:
                errs.append(f"{g.id}: diet shares do not sum to 1")
            if not 0 < tr["assim"] <= 1 or not 0 < tr["npe"] < 1 or not 0 < tr["phi"] <= 1:
                errs.append(f"{g.id}: efficiencies out of range")
            if tr["mass_kg"] <= 0 or tr["pb"] <= 0 or tr["d0"] < 0:
                errs.append(f"{g.id}: mass / pb / d0")
            for p, share in tr["diet"].items():
                if share <= 0:
                    continue
                if p not in reg.pools or not (reg.pools[p]["providers"] or p in PRIMARY_POOLS):
                    errs.append(f"{g.id}: diet pool {p} has no provider")
                elif pool_provider_level(reg, p) >= g.level:
                    errs.append(f"{g.id}: eats {p} fed by level >= {g.level}")
    for p, v in reg.pools.items():
        if not v["providers"] and p not in PRIMARY_POOLS:
            errs.append(f"pool {p}: no provider")
    return errs
