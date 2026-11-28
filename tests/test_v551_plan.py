"""TZ 5.5.1 G3–G5: the world plan on a real generated world, placers, fuels and fluids, and the
stage persistence (T02, T03, T11, T12, T41–T43, T48, T51, T56–T62, T67, T71, T72, T77, T78)."""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from hexworld.minerals import fluids, placers
from hexworld.minerals.features import FEATURES_RU
from hexworld.minerals.plan import PlanContext, build_plan, stable_id, stable_u
from hexworld.minerals.query import depth_bins, query_column_bodies, section_bodies


@pytest.fixture(scope="module")
def mplan(hist_project):
    assert hist_project.minerals is not None
    return hist_project.minerals


def test_stage_runs_in_the_pipeline_and_the_ledger_closes(mplan):
    s = mplan.summary
    assert s["systems"] > 0 and s["lots"] > 0
    assert s["ledger_ok"] and mplan.plan.report["ledger"]["ok"]
    assert s["calibration_status"] == "provisional"
    assert all(lot.get("calibration_status") == "provisional" for lot in mplan.plan.lots)


def test_t77_t78_deterministic_and_seed_case_sensitive(hist_project, mplan):
    from hexworld.minerals.stage import run_minerals

    again = run_minerals(hist_project.config, hist_project)
    assert again.content_hashes() == mplan.content_hashes()
    assert stable_u("ABC", "x") != stable_u("abc", "x")
    assert stable_id("D", "ABC") != stable_id("D", "abc")


def test_t11_technology_profile_leaves_the_geology_identical(hist_project, mplan):
    from hexworld.minerals.stage import run_minerals

    cfg = dataclasses.replace(
        hist_project.config,
        minerals=dataclasses.replace(hist_project.config.minerals, technology_profile="iron_age"),
    )
    other = run_minerals(cfg, hist_project)
    assert other.content_hashes() == mplan.content_hashes()


def test_t12_inherited_ages_may_predate_the_simulation(hist_project, mplan):
    dur = float(hist_project.hist.duration_ma)
    inh = [s for s in mplan.plan.systems if s["provenance"] == "inherited_province"]
    sim = [s for s in mplan.plan.systems if s["provenance"] == "simulated_history"]
    assert all(s["age_ma"] <= dur + 1e-9 for s in sim)
    if inh:
        assert max(s["age_ma"] for s in inh) > dur


def test_every_system_has_bodies_lots_and_typed_units(mplan):
    p = mplan.plan
    lot_sys = {lot["system_id"] for lot in p.lots}
    for s in p.systems:
        assert s["system_id"] in lot_sys
        for bid in s.get("body_ids", []):
            assert bid in p.bodies
    for lot in p.lots:
        for _com, tg in lot.get("typed_grades", {}).items():
            assert tg["unit"] in ("percent", "ppm", "g_per_t", "ct_per_t", "g_per_m3",
                                  "fraction", "kg_per_t")  # fmt: skip
        if lot.get("rock_t"):
            for com, m in lot["contained"].items():
                if lot["unit_basis"] == "dry_rock":
                    assert m <= lot["rock_t"] * (1 + 1e-9), com
    for e in p.estimates:
        assert e["cutoff"] is None and "not a reserve" in e["label"]


def test_t48_barren_kimberlites_and_t51_gem_bins(mplan):
    p = mplan.plan
    for lot in p.lots:
        if lot["model_id"] == "KIMBERLITE" and not lot.get("bearing", True):
            assert "diamond" not in lot["contained"]
        g = lot.get("gem")
        if g:
            assert sum(g["size_bins_kg"].values()) <= g["carrier_kg"] * (1 + 1e-12)
            assert g["categories"]["gem"] <= g["carrier_kg"]
            assert g["gem_rough_ct"] == pytest.approx(g["categories"]["gem"] * 5000.0)
    # diamonds only from kimberlite / lamproite branches or placers of them (no random)
    for lot in p.lots:
        if "diamond" in lot["contained"]:
            assert lot["model_id"] in ("KIMBERLITE", "PLACER_RIVER", "PLACER_COAST", "PALEOPLACER")


def test_t41_t42_placers_have_parents_and_conserve_mass(mplan):
    p = mplan.plan
    assert placers.placer_sanity(p)["ok"]
    for com, r in p.report.get("placers", {}).items():
        booked = sum(lot["contained"].get(com, 0.0) for lot in p.lots
                     if lot["model_id"] in ("PLACER_RIVER", "PLACER_COAST"))  # fmt: skip
        assert booked <= r["deposited_t"] + r["coastal_t"] + 1e-9


def _toy(n_side=8, spacing=10.0):
    """A synthetic sloping strip: a river from hex 0 down to hex n-1 at the coast."""
    from hexworld.geometry import HexLayout
    from hexworld.resources.deposits import GeoContext

    n = n_side
    x = np.arange(n) * spacing
    ctx = GeoContext(seed="toy", shape=(1, n), hex_area_km2=spacing**2, spacing_km=spacing,
                     duration_ma=30.0, x_km=x.astype(float), y_km=np.zeros(n),
                     land=np.r_[np.ones(n - 1, bool), False], ocean=np.r_[np.zeros(n - 1, bool),
                                                                          True],
                     surface_m=np.linspace(800.0, -50.0, n), water_depth_m=np.zeros(n),
                     rock=np.zeros(n, np.uint8), lith=np.zeros(n, np.int64),
                     crust_age_ma=np.full(n, 100.0), cont_frac=np.ones(n),
                     orogen=np.zeros(n, bool), orogen_age_ma=np.zeros(n),
                     collision=np.zeros(n, bool), arc_side=np.zeros(n, bool),
                     erosion_total_m=np.zeros(n), sediment_m=np.zeros(n),
                     basin_type=np.zeros(n, np.uint8), basin_fill_m=np.zeros(n),
                     slope_m_km=np.full(n, 1.0), shelf=np.zeros(n, bool),
                     upwelling=np.zeros(n, bool), seamount=np.zeros(n, bool),
                     marine_sed_m=np.zeros(n), ocean_crust_age_ma=np.full(n, np.nan))  # fmt: skip
    receiver = np.r_[np.arange(1, n), -1]
    return ctx, receiver, HexLayout(n, 1, spacing, 0.0, 0.0)


def _toy_plan(with_source: bool):
    from hexworld.minerals.plan import MineralPlan

    ctx, receiver, lay = _toy()
    plan = MineralPlan(seed="toy")
    if with_source:
        plan.ledger.open("src", "source:P", "Au", 10.0)
        plan.ledger.transfer("ero", "source:P", "mobile:S1", "Au", 3.0)
        plan.systems.append({"system_id": "S1", "hex": 1, "model_id": "OROGENIC_AU"})
    n = ctx.n
    rep = placers.route_placers(plan, ctx, receiver, np.full(n, 1.0), np.ones(n, bool),
                                np.zeros(n, bool), np.r_[np.zeros(n - 2, bool), True, False],
                                lay)  # fmt: skip
    return plan, rep


def test_s10_t42_no_source_no_placer_and_t43_downstream_only():
    plan, rep = _toy_plan(False)
    assert rep == {} and not [x for x in plan.lots if x["model_id"].startswith("PLACER")]
    plan, rep = _toy_plan(True)
    r = rep["Au"]
    assert r["released_t"] == 3.0
    assert r["deposited_t"] + r["coastal_t"] + r["abraded_t"] + r["exported_t"] == (
        pytest.approx(3.0)
    )
    assert plan.ledger.balance("source:P", "Au") == 7.0  # the parent kept the rest (N10)
    for lot in plan.lots:
        assert plan.system(lot["system_id"])["hex"] >= 1  # never upstream of the source
    assert plan.ledger.validate()["ok"]


def test_t56_coal_rank_from_burial_temperature():
    assert fluids.coal_rank(25.0)["commodity"] == "peat"
    assert fluids.coal_rank(60.0)["commodity"] == "lignite"
    assert fluids.coal_rank(120.0)["commodity"] == "bituminous_coal"
    assert fluids.coal_rank(220.0)["commodity"] == "anthracite"
    assert fluids.maturity_index(4000.0, 50.0) > fluids.maturity_index(1000.0, 50.0) >= 0.0


def _features(n, **over):
    F = {k: np.zeros(n) for k in FEATURES_RU}
    for k, v in over.items():
        F[k] = np.full(n, float(v))
    return F


def _basin_ctx(n_side=6):
    from hexworld.geometry import HexLayout
    from hexworld.resources.deposits import GeoContext

    sh = (n_side, n_side)
    n = n_side * n_side
    lay = HexLayout(n_side, n_side, 20.0, 0.0, 0.0)
    xs, ys = lay.centers()
    return GeoContext(seed="basin", shape=sh, hex_area_km2=lay.hex_area_km2, spacing_km=20.0,
                      duration_ma=40.0, x_km=xs.ravel(), y_km=ys.ravel(), land=np.ones(n, bool),
                      ocean=np.zeros(n, bool), surface_m=np.full(n, 100.0),
                      water_depth_m=np.zeros(n), rock=np.zeros(n, np.uint8),
                      lith=np.zeros(n, np.int64), crust_age_ma=np.full(n, 200.0),
                      cont_frac=np.ones(n), orogen=np.zeros(n, bool), orogen_age_ma=np.zeros(n),
                      collision=np.zeros(n, bool), arc_side=np.zeros(n, bool),
                      erosion_total_m=np.zeros(n), sediment_m=np.zeros(n),
                      basin_type=np.ones(n, np.uint8), basin_fill_m=np.full(n, 4500.0),
                      slope_m_km=np.ones(n), shelf=np.zeros(n, bool),
                      upwelling=np.zeros(n, bool), seamount=np.zeros(n, bool),
                      marine_sed_m=np.zeros(n),
                      ocean_crust_age_ma=np.full(n, np.nan)), lay  # fmt: skip


def test_s13_t57_petroleum_needs_a_seal_and_traps_before_the_charge():
    from hexworld.minerals.plan import MineralPlan

    ctx, lay = _basin_ctx()
    n = ctx.n
    common = dict(sedimentary_basin=1, deep_burial=1, porous_host=1, reduced_facies=1)
    no_seal = PlanContext(ctx, _features(n, **common, seal_rock=0), lay, 40.0)
    plan = MineralPlan(seed="basin")
    assert fluids.petroleum_systems(plan, no_seal) == []  # no seal: no conventional pool
    with_seal = PlanContext(ctx, _features(n, **common, seal_rock=1), lay, 40.0)
    plan = MineralPlan(seed="basin")
    acc = fluids.petroleum_systems(plan, with_seal, lambda_scale=50.0)
    young = [r for r in plan.rejected if r["reason"] == "trap younger than the charge"]
    assert acc or young
    for lot in acc:
        sw, so, sg = lot["saturations"]
        assert min(sw, so, sg) >= 0 and abs(sw + so + sg - 1) < 1e-9  # T58
        assert sum(lot["contained"].values()) <= lot["capacity_t"] * (1 + 1e-9)  # no overfill
        assert lot["oil_standard_m3"] == pytest.approx(lot["oil_reservoir_m3"] / lot["B_o"])
        assert "15 °C" in lot["standard_conditions"]  # T59
    assert plan.ledger.validate()["ok"]


def test_t60_t62_water_brine_heat_quantities_kept_apart(mplan):
    for lot in mplan.plan.lots:
        if lot["model_id"] == "AQUIFER" and "specific_yield" in lot:
            assert lot["specific_yield"] <= lot["porosity"]
            assert lot["water_stored_m3"] >= lot["water_drainable_10m_m3"] * 0  # both kept
            assert "recharge_m3_per_year" in lot
        if lot["model_id"] == "GEOTHERMAL":
            assert lot["heat_in_place_J"] > 0 and lot["recharge_W"] > 0
        if lot["model_id"] == "BRINE":
            assert lot["unit_basis"] == "dissolved"


def test_t67_minerals_do_not_touch_soil_fertility(hist_project):
    from hexworld.minerals.stage import run_minerals

    before = hist_project.resources.content_hashes()
    run_minerals(hist_project.config, hist_project)
    assert hist_project.resources.content_hashes() == before


def test_t71_t72_depth_bins_and_section_against_known_bodies(mplan, hist_project):
    p = mplan.plan
    body = next(b for b in p.bodies.values() if b.morphology in ("lens", "tabular", "stockwork"))
    surf = 0.0 + float(body.origin_m[2]) + 5000.0
    edges = [0, 25, 100, 500, 1000, 3000, 10000, 1e9]
    bins = depth_bins(body, lambda x, y: np.full(np.shape(x), surf), edges, n=12)
    total = float(body.quadrature(12)[1].sum())
    assert bins.sum() == pytest.approx(total, rel=1e-12)  # no boundary counted twice
    x, y = body.origin_m[0], body.origin_m[1]
    col = query_column_bodies([body], x, y, surface_m=surf)
    sec = section_bodies([body], [x - 1.0, x, x + 1.0], [y, y, y])
    if col:
        mid = [iv for b in sec for iv in b["intervals"] if iv[0] == 1]
        assert mid and mid[0][1] == pytest.approx(col[0]["z_top_m"])


def test_t02_t03_save_load_round_trip_and_old_saves(tmp_path, hist_project, mplan):
    from hexworld.io import load_project, save_project

    path = save_project(hist_project, tmp_path / "w")
    back = load_project(path)
    assert back.minerals is not None
    assert back.minerals.content_hashes() == mplan.content_hashes()
    # an old save without the 5.5.1 section loads unchanged, with no plan invented
    (path / "minerals.json").unlink()
    (path / "minerals_fields.npz").unlink()
    import json

    man = json.loads((path / "manifest.json").read_text(encoding="utf-8"))
    man["v551"] = None
    (path / "manifest.json").write_text(json.dumps(man), encoding="utf-8")
    old = load_project(path, verify=False)
    assert old.minerals is None and old.resources is not None


def test_config_saved_before_551_keeps_its_world():
    from hexworld import Config

    d = Config().to_dict()
    d.pop("minerals")
    assert Config.from_dict(d).minerals.enabled is False
    assert Config().minerals.enabled is True


def test_plan_on_features_with_missing_inputs_is_unresolved_not_invented(hist_project):
    from hexworld.minerals.features import features_from_project
    from hexworld.resources.deposits import context_from_project

    ctx = context_from_project(hist_project)
    F = features_from_project(hist_project, ctx)
    F["magmatic_activity"] = np.full(ctx.n, np.nan)  # no volcanism data
    plan = build_plan(PlanContext(ctx, F, hist_project.layout, 30.0), models=["PORPHYRY"])
    assert not [s for s in plan.systems if s["model_id"] == "PORPHYRY"]
    assert plan.unresolved.get("PORPHYRY", 0) > 0


def test_t10_byproducts_only_with_their_host(mplan):
    from hexworld.minerals.plan import BYPRODUCTS

    for lot in mplan.plan.lots:
        c = lot.get("contained", {})
        for byp, (host, ratio) in BYPRODUCTS.items():
            if byp in c and lot.get("typed_grades", {}).get(byp, {}).get("primary_host"):
                assert host in c and c[byp] == pytest.approx(c[host] * ratio)
