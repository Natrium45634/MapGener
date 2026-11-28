"""TZ 5.5 on a real generated world: the stage reads the world and changes none of it (B04,
B07), old saves stay «not generated» (B02, B03), save/load without recompute, the Nature
tab, the hex report (B78), region aggregation (B79), exports, the CLI, the P1 dynamics
with a closing ledger (B71, B84) and the resource inventory (B64, B65)."""

from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from hexworld import Config, generate
from hexworld.biosphere.stage import run_biosphere, stage_keys
from hexworld.config import BiosphereParams, ContinentParams
from hexworld.io import load_project, save_project

STAGES = ("geo", "hist", "terrain", "hydro", "env", "topo", "earth", "atmosphere", "biomes",
          "volcanism", "resources", "minerals")  # fmt: skip


def _hashes(p) -> dict:
    out = {}
    for n in STAGES:
        o = getattr(p, n, None)
        if o is not None and hasattr(o, "content_hashes"):
            out[n] = hashlib.sha256(json.dumps(o.content_hashes(), sort_keys=True,
                                               default=str).encode()).hexdigest()  # fmt: skip
    return out


@pytest.fixture(scope="module")
def world():
    cfg = Config(seed="Bio55T", width=60, height=40, hex_spacing_km=100.0,
                 continents=ContinentParams(skeleton_cells=224),
                 biosphere=BiosphereParams(enabled=False))  # fmt: skip
    p = generate(cfg)
    before = _hashes(p)
    cfg_on = replace(cfg, biosphere=BiosphereParams())
    p.config = cfg_on
    p.biosphere = run_biosphere(cfg_on, p)
    return p, before


def test_b04_b07_physics_hashes_unchanged(world):
    p, before = world
    assert _hashes(p) == before
    st = p.biosphere.state
    assert st.summary["ledger_ok"] and st.summary["taxa_world"] > 0
    assert st.quality["history"] == "present_only"


def test_life_disabled_has_no_biota():
    cfg = Config(seed="x", biosphere=BiosphereParams(enabled=False))
    assert not cfg.biosphere.enabled
    d = Config().to_dict()
    d.pop("biosphere")
    assert Config.from_dict(d).biosphere.enabled is False  # an old config: not generated


def test_stage_keys_ignore_display_and_follow_seed(world):
    p, _ = world
    k1 = stage_keys(p.config, p)
    k2 = stage_keys(
        replace(p.config, biosphere=replace(p.config.biosphere, biology_seed="Other")), p
    )
    assert k1 != k2
    from hexworld import views

    v = views.VIEWS["nature_55"]
    h0 = p.biosphere.content_hashes()
    for o in v.modes:
        rgb = v.render(p, o.id, v.default_overlays())
        assert rgb.shape == (40, 60, 3) and rgb.dtype == np.uint8
        assert v.legend(p, o.id)
        v.render(p, o.id, frozenset({"percentile"}))
    assert p.biosphere.content_hashes() == h0  # display never recomputes (B07)
    assert stage_keys(p.config, p) == k1


def test_b02_b03_old_save_and_roundtrip(world, tmp_path):
    p, _ = world
    bio = p.biosphere
    p.biosphere = None
    old = tmp_path / "old"
    save_project(p, old)
    files_before = {f.name: f.read_bytes() for f in old.iterdir() if f.is_file()}
    q = load_project(old)
    assert q.biosphere is None  # not generated, never recomputed on open
    assert {f.name: f.read_bytes() for f in old.iterdir() if f.is_file()} == files_before
    p.biosphere = bio
    new = tmp_path / "new"
    save_project(p, new)
    r = load_project(new)
    assert r.biosphere is not None
    assert r.biosphere.content_hashes() == bio.content_hashes()
    assert r.biosphere.state.realms["land"].present.sum() == bio.state.realms["land"].present.sum()


def test_b78_hex_report_explains(world):
    from hexworld.biosphere.explain import REASON_RU, hex_report
    from hexworld.hex_report import hex_report as panel

    p, _ = world
    st = p.biosphere.state
    land = np.nonzero(st.frac["land_ground"] > 0.9)[0]
    i = int(land[len(land) // 2])
    rep = hex_report(st, i)
    assert rep["flora"] and all(x["status"] == "present" for x in rep["flora"])
    for x in rep["absent_examples"]:
        assert x["status"] in REASON_RU
    r, c = divmod(i, p.layout.width)
    secs = {s.key: s for s in panel(p, c, r).sections}
    assert "nature" in secs and any("Растения" in lab for lab, _v in secs["nature"].rows)


def test_b79_region_report_union(world):
    from hexworld.biosphere.export import region_report

    p, _ = world
    st = p.biosphere.state
    land = np.nonzero(st.frac["land_ground"] > 0.9)[0][:20]
    one = region_report(st, land[:1])
    many = region_report(st, land)
    assert many["taxa_union"] >= one["taxa_union"]
    assert many["taxa_union"] <= st.layers["total_richness"][land].sum()  # union, not sum
    assert many["area_km2"] == pytest.approx(20 * st.area_km2)


def test_exports_and_cli(world, tmp_path):
    from hexworld.biosphere.export import export_biosphere
    from hexworld.cli import main

    p, _ = world
    land = np.nonzero(p.biosphere.state.frac["land_ground"] > 0.9)[0]
    files = export_biosphere(p, tmp_path / "ex", hexes={int(land[0])})
    names = {Path(f).name for f in files}
    assert names == {"biosphere_world.json", "biosphere_hexes.csv", "biosphere_taxa.csv",
                     "biosphere_resources.csv", "biosphere_region.json"}  # fmt: skip
    doc = json.loads((tmp_path / "ex" / "biosphere_world.json").read_text(encoding="utf-8"))
    assert doc["provenance"]["calibration"] == "provisional_not_calibrated"
    save_project(p, tmp_path / "w")
    r, c = divmod(int(land[0]), p.layout.width)
    assert main(["bio-explain", str(tmp_path / "w"), str(c), str(r)]) == 0
    assert main(["bio-diagnostics", str(tmp_path / "w")]) == 0


def test_b64_b65_inventory(world):
    p, _ = world
    inv = p.biosphere.resources()
    assert ["timber_increment_m3_yr", "fuelwood_increment_t_yr"] in inv["meta"]["not_additive"]
    for k, v in inv["layers"].items():
        assert np.all(np.asarray(v) >= 0), k
    # stock and increment are different quantities: the stock is decades of increment
    assert inv["totals"]["timber_stock_m3"] > inv["totals"]["timber_increment_m3_yr"]


def test_b71_b84_dynamics_change_state_not_normals(world):
    import copy

    from hexworld.biosphere.dynamics import advance_years, rain_anomaly

    p, before = world
    st = copy.deepcopy(p.biosphere.state)
    rain = rain_anomaly("D", 6, st.shape, 100.0)
    y1 = advance_years(st, p, 6, "D", rain=rain)
    assert max(y["residual"] for y in y1) < 1e-9
    assert _hashes(p) == before  # normals and every other layer unchanged
    st2 = copy.deepcopy(p.biosphere.state)
    y2 = advance_years(st2, p, 6, "D", rain=rain)
    assert [y["stock_kgc"] for y in y1] == [y["stock_kgc"] for y in y2]  # replay identical
    # a drought sequence leaves a different state than wet years (carry-over)
    dry = [np.full(st.n, 0.5)] * 6
    wet = [np.full(st.n, 1.2)] * 6
    a = copy.deepcopy(p.biosphere.state)
    b = copy.deepcopy(p.biosphere.state)
    advance_years(a, p, 6, "D", rain=dry, fire_scale=0.0)
    advance_years(b, p, 6, "D", rain=wet, fire_scale=0.0)
    land = a.frac["land_ground"] > 0.5
    assert (
        a.layers["dyn_plant_biomass_kgc_m2"][land].sum()
        < b.layers["dyn_plant_biomass_kgc_m2"][land].sum()
    )
    # one dry run does not turn a forest into a desert: wood stays most of its stock
    w0 = p.biosphere.state.layers["wood_kgc_m2"][land].sum()
    assert a.layers["dyn_wood_kgc_m2"][land].sum() > 0.6 * w0


def test_551_t66_biogenic_mass_counted_once(world):
    """5.5.1 T66: peat, coal and other biogenic geological stocks are 5.5.1 lots; the 5.5
    inventory holds living and dead surface carbon only, and running 5.5 leaves the
    5.5.1 plan unchanged (the same carbon is never in both)."""
    from hexworld.minerals.catalog import load as load_catalog

    p, before = world
    assert p.minerals is not None and _hashes(p)["minerals"] == before["minerals"]
    commodities = set(load_catalog().commodities)
    inv = p.biosphere.resources()
    for k in inv["layers"]:
        assert not any(k.startswith(c + "_") or k == c for c in commodities), k


def test_b03_bio_generate_makes_a_copy(world, tmp_path):
    from hexworld.cli import main

    p, _ = world
    bio = p.biosphere
    p.biosphere = None
    try:
        save_project(p, tmp_path / "old")
    finally:
        p.biosphere = bio
    before = {f.name: f.read_bytes() for f in (tmp_path / "old").iterdir() if f.is_file()}
    assert main(["bio-generate", str(tmp_path / "old"), "--out", str(tmp_path / "old")]) == 2
    assert main(["bio-generate", str(tmp_path / "old"), "--out", str(tmp_path / "new")]) == 0
    assert {f.name: f.read_bytes() for f in (tmp_path / "old").iterdir() if f.is_file()} == before
    q = load_project(tmp_path / "new")
    assert q.biosphere is not None and load_project(tmp_path / "old").biosphere is None


def test_world_resources_tab_one_legend(world):
    from hexworld import views

    p, _ = world
    v = views.VIEWS["resources_world"]
    assert v.availability(p)[0]
    for o in v.modes:
        rgb = v.render(p, o.id, v.default_overlays())
        assert rgb.shape == (40, 60, 3)
        leg = [it.label for it in v.legend(p, o.id)]
        if o.id == "all":  # natural and mineral resources share one legend
            assert any(t.startswith("природные") for t in leg)
            assert any(t.startswith("минеральные") for t in leg)
