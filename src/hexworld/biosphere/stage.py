"""Stage ``v55`` of the pipeline: the TZ 5.5 biosphere on a generated world, its stage keys
(cache DAG of §14), serialisation (NPZ arrays + JSON) and content hashes.

The stage reads the world and writes nothing into it: every earlier stage keeps its hash.
Display settings (legend, locale, zoom) are not in the keys (B07)."""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import asdict, dataclass, field

import numpy as np

from .model import ALGORITHM_VERSION, BioState, GroupArrays, RealmTable
from .registry import load

SCHEMA_VERSION = "hexworld-biosphere-1"
STAGE = "v55_biosphere"


@dataclass
class BioData:
    state: BioState
    stage_keys: dict
    provenance: dict
    timings: dict = field(default_factory=dict)
    warnings: list = field(default_factory=list)
    _resources: dict | None = field(default=None, repr=False)

    @property
    def summary(self) -> dict:
        return self.state.summary

    @property
    def layers(self) -> dict:
        return self.state.layers

    def resources(self) -> dict:
        if self._resources is None:
            from .resources import inventory

            self._resources = inventory(self.state, self.state.area_km2)
        return self._resources

    def content_hashes(self) -> dict:
        out = {}
        for k in ("npp_gc_m2_yr", "total_richness", "plant_biomass_kgc_m2", "bioregion"):
            a = np.ascontiguousarray(self.state.layers[k])
            out[k] = hashlib.sha256(a.tobytes()).hexdigest()[:16]
        return out


def upstream_key(project) -> str:
    """Content of every stage the biosphere reads (geology → climate → soils → biomes)."""
    parts = {}
    for name in (
        "hist",
        "hydro",
        "env",
        "topo",
        "earth",
        "atmosphere",
        "biomes",
        "volcanism",
        "resources",
    ):
        obj = getattr(project, name, None)
        sk = getattr(obj, "stage_keys", None) if obj is not None else None
        parts[name] = sorted(sk.items()) if isinstance(sk, dict) else None
    parts["geo"] = sorted(project.geo.stage_keys.items()) if project.geo is not None else None
    return hashlib.sha256(json.dumps(parts, sort_keys=True, default=str).encode()).hexdigest()[:24]


def stage_keys(config, project) -> dict:
    reg = load()
    doc = {"v": ALGORITHM_VERSION, "up": upstream_key(project), "p": asdict(config.biosphere),
           "registry": reg.source_sha256, "seed": bio_seed(config)}  # fmt: skip
    return {STAGE: hashlib.sha256(json.dumps(doc, sort_keys=True).encode()).hexdigest()[:24]}


def bio_seed(config) -> str:
    """Biology seed: the override if given, else the world seed (ASCII, case-sensitive)."""
    s = config.biosphere.biology_seed
    return s if s else config.seed


def run_biosphere(config, project, report=lambda f: None) -> BioData:
    from .model import run

    p = config.biosphere
    t0 = time.perf_counter()
    cap = None if p.capture_mult <= 0 else float(p.capture_mult)
    st = run(
        project,
        mode=p.mode if p.mode != "DYNAMIC" else "REALIZED_BASELINE",
        detail=p.detail,
        seed=bio_seed(config),
        capture_mult=cap,
        progress=report,
    )
    warnings = []
    if p.mode == "DYNAMIC" and p.dynamic_years > 0:
        from .dynamics import advance_years

        advance_years(st, project, years=p.dynamic_years, seed=bio_seed(config))
        st.mode = "DYNAMIC"
    if not st.summary.get("ledger_ok", False):
        warnings.append("biosphere: the food ledger does not close (see summary)")
    timings = {k: round(v, 4) for k, v in st.timings.items()}
    timings["total"] = round(time.perf_counter() - t0, 4)
    prov = {"algorithm_version": ALGORITHM_VERSION, "schema_version": SCHEMA_VERSION,
            "registry": load().version, "registry_sha256": load().source_sha256,
            "calibration": "provisional_not_calibrated", "seed": bio_seed(config),
            "history_quality": st.quality.get("history")}  # fmt: skip
    return BioData(st, stage_keys(config, project), prov, timings, warnings)


# -- serialisation -----------------------------------------------------------------------------
def _jsonable(x):
    if isinstance(x, dict):
        return {str(k): _jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_jsonable(v) for v in x]
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.floating,)):
        return float(x)
    if isinstance(x, np.bool_):
        return bool(x)
    if isinstance(x, float) and not np.isfinite(x):
        return None
    return x


def to_arrays(data: BioData) -> dict:
    st = data.state
    arrays = {}
    for k, v in st.frac.items():
        arrays[f"frac__{k}"] = v
    for k, v in st.layers.items():
        if k.startswith("_"):
            continue
        arrays[f"layer__{k}"] = np.asarray(v)
    for k, v in st.layers.get("_env", {}).items():
        arrays[f"env__{k}"] = v
    realm_idx = {}
    for gid, d in st.groups.items():
        realm_idx.setdefault(d.realm, d.idx)
        for k, v in d.arrays().items():
            arrays[f"group__{gid}__{k}"] = v  # compact on the hexes of its realm
    for r, idx in realm_idx.items():
        arrays[f"gidx__{r}"] = idx
    for r, tab in st.realms.items():
        arrays[f"realm__{r}__idx"] = tab.idx
        arrays[f"realm__{r}__occ"] = tab.occ
        arrays[f"realm__{r}__suit"] = tab.suit
        arrays[f"realm__{r}__present"] = np.packbits(tab.present, axis=1)
    for c, d in st.lineage.items():
        arrays[f"lineage__{c}__pool"] = d["pool"]
        arrays[f"lineage__{c}__d_km"] = d["d_km"]
    return arrays


def to_json(data: BioData) -> dict:
    st = data.state
    return _jsonable({
        "schema_version": SCHEMA_VERSION, "algorithm_version": ALGORITHM_VERSION,
        "mode": st.mode, "seed": st.seed, "n": st.n, "shape": list(st.shape),
        "area_km2": st.area_km2,
        "realm_templates": {r: tab.templates for r, tab in st.realms.items()},
        "group_realms": {g: d.realm for g, d in st.groups.items()},
        "lineage_names": st.lineage_names, "lineage_keys": st.lineage_keys,
        "ledger": st.ledger, "summary": st.summary, "quality": st.quality,
        "npp_limit_names": st.layers.get("_npp_limit_names"),
        "index_version": st.layers.get("_index_version"),
        "ranges": [[k, *v] for k, v in sorted(st.layers.get("_ranges", {}).items())],
        "stage_keys": data.stage_keys, "provenance": data.provenance,
        "timings": data.timings, "warnings": data.warnings,
    })  # fmt: skip


def from_saved(doc: dict, arrays: dict) -> BioData:
    frac = {k[6:]: v for k, v in arrays.items() if k.startswith("frac__")}
    layers = {k[7:]: v for k, v in arrays.items() if k.startswith("layer__")}
    layers["_env"] = {k[5:]: v for k, v in arrays.items() if k.startswith("env__")}
    layers["_npp_limit_names"] = doc.get("npp_limit_names")
    layers["_index_version"] = doc.get("index_version")
    layers["_ranges"] = {r[0]: tuple(r[1:]) for r in doc.get("ranges", [])}
    gidx = {k[6:]: v for k, v in arrays.items() if k.startswith("gidx__")}
    n = int(doc["n"])
    groups: dict = {g: GroupArrays(realm, gidx[realm], n) for g, realm in
                    doc["group_realms"].items()}  # fmt: skip
    for k, v in arrays.items():
        if k.startswith("group__"):
            _p, gid, name = k.split("__", 2)
            groups[gid].arrays()[name] = v
    realms = {}
    for r, tids in doc["realm_templates"].items():
        idx = arrays[f"realm__{r}__idx"]
        pres = np.unpackbits(arrays[f"realm__{r}__present"], axis=1, count=idx.size).astype(bool)
        realms[r] = RealmTable(r, idx, list(tids), arrays[f"realm__{r}__occ"],
                               arrays[f"realm__{r}__suit"], pres)  # fmt: skip
    lineage: dict = {}
    for k, v in arrays.items():
        if k.startswith("lineage__"):
            _p, c, name = k.split("__", 2)
            lineage.setdefault(c, {})[name] = v
    st = BioState(doc["mode"], doc["seed"], int(doc["n"]), tuple(doc["shape"]),
                  float(doc["area_km2"]), frac, groups, realms, lineage, doc["lineage_names"],
                  doc["lineage_keys"], layers, doc["ledger"], doc["summary"], doc["quality"],
                  doc.get("timings", {}))  # fmt: skip
    return BioData(st, doc["stage_keys"], doc["provenance"], doc.get("timings", {}),
                   doc.get("warnings", []))  # fmt: skip
