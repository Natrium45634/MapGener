"""Export of the 5.5 biosphere (§14): a world JSON (summary, ledger, lineages, quality,
provenance), CSV tables per hex and per taxon with explicit units and missing values, the
renewable resource inventory and a region report (sums, area-weighted means, union of
taxa, B79). No fake precision: values are rounded to the meaningful digits."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

from .explain import hex_report
from .registry import load
from .stage import _jsonable, to_json

HEX_COLUMNS = (("npp_gc_m2_yr", "gC/m²/yr"), ("plant_biomass_kgc_m2", "kgC/m²"),
               ("animal_biomass_kgc_km2", "kgC/km²"), ("flora_richness", "analogues"),
               ("fauna_richness", "analogues"), ("freshwater_richness", "analogues"),
               ("marine_richness", "analogues"), ("functional_richness", "groups"),
               ("expected_richness", "Σp"), ("total_richness_index", "0-100"),
               ("total_richness_percentile", "0-100"), ("weighted_endemism", "1"),
               ("endemic_taxa", "analogues"), ("bioregion", "id"),
               ("isolation_km", "km"))  # fmt: skip


def region_report(state, hexes) -> dict:
    """Aggregate of a user selection: area-weighted means, sums of stocks, union of taxa
    (a taxon in two hexes is counted once), coverage of unknown inputs (B79)."""
    reg = load()
    idx = np.asarray(sorted({int(i) for i in hexes}), dtype=np.int64)
    L = state.layers
    area = state.area_km2
    w = np.full(idx.size, area)
    out = {"hexes": int(idx.size), "area_km2": float(w.sum())}
    for k in ("npp_gc_m2_yr", "plant_biomass_kgc_m2"):
        v = L[k][idx].astype(float)
        ok = np.isfinite(v)
        out[f"{k}_area_mean"] = float((v[ok] * w[ok]).sum() / max(w[ok].sum(), 1e-9))
        out[f"{k}_unknown_share"] = float(1.0 - w[ok].sum() / w.sum()) if w.sum() else 0.0
    out["npp_total_tc_yr"] = float(np.nansum(L["npp_gc_m2_yr"][idx].astype(float)) * area * 1e6
                                   / 1e6)  # fmt: skip
    taxa = set()
    groups = set()
    for _realm, tab in state.realms.items():
        j = np.searchsorted(tab.idx, idx)
        ok = (j < tab.idx.size) & (tab.idx[np.minimum(j, tab.idx.size - 1)] == idx)
        jj = j[ok]
        if jj.size == 0:
            continue
        pres = tab.present[:, jj]
        for k in np.nonzero(pres.any(axis=1))[0]:
            tid = tab.templates[k]
            g = reg.groups[reg.templates[tid].group_id]
            lin = state.lineage[g.traits["mob"]]["pool"][idx[ok][pres[k]]]
            for lj in np.unique(lin).tolist():
                taxa.add((tid, int(lj)))
            groups.add(g.id)
    out["taxa_union"] = len(taxa)
    out["groups_union"] = len(groups)
    out["taxa_mean_per_hex"] = float(np.mean(L["total_richness"][idx])) if idx.size else 0.0
    return out


def export_biosphere(project, out_dir, hexes=None) -> list[str]:
    data = project.biosphere
    if data is None:
        raise ValueError("no 5.5 biosphere in this project")
    st = data.state
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    files = []
    doc = to_json(data)
    doc.pop("ranges", None)
    doc["resources"] = {"totals": data.resources()["totals"], "meta": data.resources()["meta"]}
    p = out / "biosphere_world.json"
    p.write_text(json.dumps(_jsonable(doc), ensure_ascii=False, indent=1), encoding="utf-8")
    files.append(str(p))
    lay = project.layout
    p = out / "biosphere_hexes.csv"
    with open(p, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["col", "row", "land_ground", "inland_water", "marine_water", "grounded_ice"]
                   + [f"{k} [{u}]" for k, u in HEX_COLUMNS])  # fmt: skip
        life = (st.frac["land_ground"] + st.frac["inland_water"] + st.frac["marine_water"]) > 0
        for i in np.nonzero(life)[0]:
            r, c = divmod(int(i), lay.width)
            row = [c, r] + [
                f"{float(st.frac[k][i]):.3f}"
                for k in ("land_ground", "inland_water", "marine_water", "grounded_ice")
            ]
            for k, _u in HEX_COLUMNS:
                v = st.layers[k][i]
                row.append("" if not np.isfinite(v) else f"{float(v):.4g}")
            w.writerow(row)
    files.append(str(p))
    p = out / "biosphere_taxa.csv"
    reg = load()
    with open(p, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["taxon_id", "template_id", "group_id", "analogue_ru", "lineage_key",
                    "range_km2", "range_complete", "endemic", "parameter_status"])  # fmt: skip
        for tx, (rng, touches, tid, key) in sorted(st.layers.get("_ranges", {}).items()):
            tpl = reg.templates[tid]
            w.writerow([tx, tid, tpl.group_id, tpl.name_ru, key, f"{rng:.0f}",
                        "no (touches the window edge)" if touches else "yes",
                        "unknown" if touches else ("yes" if rng <= 1e5 else "no"),
                        "provisional"])  # fmt: skip
    files.append(str(p))
    p = out / "biosphere_resources.csv"
    res = data.resources()
    with open(p, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["quantity", "world_total", "unit_from_name", "pool", "note"])
        for k, v in res["totals"].items():
            w.writerow([k, f"{v:.4g}", k.rsplit("_", 1)[-1], res["meta"]["pools"].get(k, ""),
                        "potential, not extraction"])  # fmt: skip
    files.append(str(p))
    if hexes:
        p = out / "biosphere_region.json"
        rep = region_report(st, hexes)
        if len(hexes) == 1:
            rep["hex"] = hex_report(st, int(next(iter(hexes))))
        p.write_text(json.dumps(_jsonable(rep), ensure_ascii=False, indent=1), encoding="utf-8")
        files.append(str(p))
    return files
