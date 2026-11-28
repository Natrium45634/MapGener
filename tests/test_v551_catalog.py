"""TZ 5.5.1 G1: catalogue and schema (T04, T07–T09), typed units and the numeric fixtures
N02–N24 (T31–T33, T59–T63), the ledger (T38, T39, N10) and the hard gates of every template
(T05, T06)."""

from __future__ import annotations

import copy
import json
import math
from pathlib import Path

import numpy as np
import pytest

from hexworld.minerals import catalog as cat_mod
from hexworld.minerals import gates as g
from hexworld.minerals import units as u
from hexworld.minerals.ledger import LedgerError, ResourceLedger

FIX = {f["id"]: f for f in json.loads(
    (Path(__file__).parent / "data" / "v551" / "fixture_examples.json").read_text("utf-8")
)["fixtures"]}  # fmt: skip
CAT = cat_mod.load()


def _docs():
    d = cat_mod.DATA_DIR
    return (
        json.loads((d / "commodities.json").read_text("utf-8")),
        json.loads((d / "deposit_models.json").read_text("utf-8")),
    )


# -- catalogue ------------------------------------------------------------------------------
def test_catalogue_sizes_and_priorities():
    assert len(CAT.commodities) == 194 and len(CAT.models) == 92
    assert sum(m.priority == "P1" for m in CAT.models.values()) == 13
    assert all(m.calibration_status.startswith("provisional") for m in CAT.models.values())


@pytest.mark.parametrize(
    "mutate, msg",
    [
        (lambda c, m: c["commodities"].append(copy.deepcopy(c["commodities"][0])), "duplicate"),
        (lambda c, m: c["commodities"][0]["model_ids"].append("NOPE"), "unknown"),
        (lambda c, m: c["commodities"][0].__setitem__("quantity_schema", "bags"), "schema"),
        (lambda c, m: m["models"][0]["morphologies"].append("blob"), "morphology"),
        (lambda c, m: m["models"].append(copy.deepcopy(m["models"][0])), "duplicate"),
        (lambda c, m: m["models"][0]["principal"].append("unobtainium"), "unknown commodity"),
    ],
)
def test_schema_rejects_invalid_records(mutate, msg):
    c, m = _docs()
    mutate(c, m)
    with pytest.raises(cat_mod.CatalogError, match=msg):
        cat_mod.validate(c, m)


@pytest.mark.parametrize("cid", sorted(CAT.commodities))
def test_every_commodity_has_compatible_model_and_quantity_schema(cid):
    c = CAT.commodities[cid]
    assert c.model_ids and all(CAT.models[m] for m in c.model_ids)
    assert c.amount_unit in u.AMOUNT_UNITS
    for unit in c.grade_units:
        assert unit in u.GRADE_UNITS
    assert c.basis in CAT.commodities or c.basis in ("lithology_lot", "aquifer_water")


def test_groups_and_varieties_are_views_not_additive():
    ree = CAT.group_members("REE")
    assert set(u.REE_ELEMENTS) <= set(ree) and "Pm" not in ree and "Sc" not in ree
    assert {"ruby", "sapphire"} <= set(CAT.group_members("corundum"))
    assert all(CAT.commodities[v].not_additive_with_parent for v in ("ruby", "sapphire", "La"))
    # N23: contained REE = sum of the elements; TREO is a separate derived oxide figure
    f = FIX["N23"]["inputs"]
    els = {"La": f["La_kg"], "Ce": f["Ce_kg"], "Y": f["Y_kg"]}
    assert u.group_total(els) == FIX["N23"]["expected"]
    assert u.treo(els) > u.group_total(els)
    with pytest.raises(u.UnitError):
        u.group_total({"La": 1.0, "Fe": 1.0})


def test_legacy_concordance_covers_every_52_model():
    from hexworld.resources.models import MODELS

    assert set(cat_mod.LEGACY_TO_NEW) == set(MODELS)
    assert all(m in CAT.models for ms in cat_mod.LEGACY_TO_NEW.values() for m in ms)


# -- units and the numeric fixtures ---------------------------------------------------------
def rel(a, b):
    return abs(a - b) / abs(b) if b else abs(a)


def test_mass_and_grade_fixtures():
    assert rel(u.rock_mass_t(200000, 2700), FIX["N02"]["expected"]) < 1e-12
    au_kg = u.contained_t(540000, 5, "g_per_t") * 1000.0
    assert rel(au_kg, FIX["N03"]["expected"]) < 1e-12
    assert rel(u.contained_t(1e6, 0.6, "percent"), FIX["N06"]["expected"]) < 1e-12
    assert rel(u.weighted_grade([100, 900], [10, 1]), FIX["N07"]["expected"]) < 1e-12
    rho = u.dry_bulk_density(3000, 0.2)
    assert rel(u.rock_mass_t(1000, rho), FIX["N08"]["expected"]) < 1e-12  # porosity once
    kg = u.contained_from_volume_grade_kg(500 * 20 * 1.5, 0.8, "g_per_m3")
    assert rel(kg, FIX["N09"]["expected"]) < 1e-12
    assert rel(u.gem_rough_ct(2, 0.15), FIX["N11"]["expected"]) < 1e-12
    assert rel(u.contained_t(1000, 100, "ppm") * 1000, FIX["N12"]["expected"]) < 1e-12


def test_unit_conversions_and_rejections():
    assert u.to_internal(1, "g_per_t") == u.to_internal(1, "ppm") == 1e-6
    assert u.to_internal(1, "ct_per_t") == pytest.approx(2e-7)
    assert u.to_internal(1, "mg_per_L") == pytest.approx(1e-3)
    for bad in ((-1, "ppm"), (float("nan"), "percent"), (150, "percent"), (1, "furlongs")):
        with pytest.raises(u.UnitError):
            u.to_internal(*bad)
    with pytest.raises(u.UnitError):
        u.contained_t(10, 1, "g_per_m3")  # a volumetric grade is not a mass fraction
    assert u.Grade(5, "g_per_t").internal == pytest.approx(5e-6)


def test_marine_fluid_water_heat_fixtures():
    assert rel(u.nodule_mass_t(1e6, 0.4, 10), FIX["N13"]["expected"]) < 1e-12
    assert u.nodule_mass_t(1e6, 0.4, 10, density_is_area_average=True) == 1e4  # not twice
    assert rel(u.crust_thickness_m(2, 10), FIX["N14"]["expected"]) < 1e-12
    assert rel(u.dissolved_mass_kg(1e5, 200), FIX["N15"]["expected"]) < 1e-12
    assert rel(u.salt_precipitated_kg(2000, 4, 300), FIX["N16"]["expected"]) < 1e-12
    assert u.salt_precipitated_kg(0.0, 0.4, 300) == FIX["N22"]["expected"]  # pure water
    vp = u.pore_volume_m3(1e6, 0.5, 0.2) * 0.6
    assert rel(vp, FIX["N17"]["expected"]) < 1e-12
    assert rel(u.standard_volume_m3(60000, 1.2), FIX["N18"]["expected"]) < 1e-12
    with pytest.raises(u.UnitError):
        u.standard_volume_m3(1, 0.0)
    with pytest.raises(u.UnitError):
        u.check_saturations(0.5, 0.6, 0.1)
    assert rel(u.drainable_water_m3(1e6, 0.15, 2), FIX["N19"]["expected"]) < 1e-12
    with pytest.raises(u.UnitError):
        u.drainable_water_m3(1, 0.3, 1, porosity=0.2)  # Sy <= phi
    q = u.heat_in_place_j(1e6, 0.1, 2700, 800, 1000, 4200, 50)
    assert rel(q, FIX["N20"]["expected"]) < 1e-12
    assert rel(u.rock_mass_t(5e4, 1.0) * 1000 * 0.2, FIX["N24"]["expected"]) < 1e-12


def test_true_and_apparent_thickness_formulas():
    assert rel(u.vertical_intercept(2, 60), FIX["N04"]["expected"]) < 1e-12
    assert u.vertical_intercept(2, 0) == 2.0  # horizontal bed: true = vertical (T19)
    assert math.isinf(u.vertical_intercept(2, 90))  # vertical vein: finite geometry decides
    assert u.horizontal_intercept(2, 90) == pytest.approx(2.0)
    assert u.apparent_thickness(2, (0, 0, 1), (0, 0, 1)) == 2.0
    assert rel(u.ellipsoid_volume(50, 20, 10), FIX["N05"]["expected"]) < 1e-12


def test_gem_size_bins():
    assert u.size_bins_ok(10.0, {"<1ct": 6.0, "1-5ct": 3.0, ">5ct": 1.0})
    assert not u.size_bins_ok(10.0, {"a": 8.0, "b": 3.0})


# -- ledger ---------------------------------------------------------------------------------
def test_ledger_transfer_replay_and_closure():
    led = ResourceLedger()
    f = FIX["N10"]["inputs"]
    led.open("o1", "vein-1", "Au", f["initial_kg"])
    led.transfer("t1", "vein-1", "mobile", "Au", f["eroded_kg"])
    led.transfer("t2", "mobile", "placer-1", "Au", f["trapped_kg"])
    led.remove("t3", "mobile", "Au", f["outflow_kg"], "exported")
    # the same operations again change nothing (idempotent, T39)
    led.transfer("t1", "vein-1", "mobile", "Au", f["eroded_kg"])
    led.remove("t3", "mobile", "Au", f["outflow_kg"], "exported")
    exp = FIX["N10"]["expected"]
    assert led.balance("vein-1", "Au") == exp["parent_kg"]
    assert led.balance("placer-1", "Au") == exp["placer_kg"]
    rep = led.validate()
    assert rep["ok"] and rep["bases"]["Au"]["exported"] == exp["export_kg"]


def test_ledger_rejects_creation_and_negative_stock():
    led = ResourceLedger()
    led.open("o", "lot", "Cu", 5.0)
    with pytest.raises(LedgerError):
        led.transfer("x", "lot", "sink", "Cu", 6.0)  # more than there is (T38)
    with pytest.raises(LedgerError):
        led.open("n", "lot", "Cu", -1.0)
    with pytest.raises(LedgerError):
        led.transfer("o", "lot", "sink", "Cu", 1.0)  # op id reused with other content
    assert led.validate()["ok"]


# -- hard gates of every template -----------------------------------------------------------
def test_every_template_has_a_rule():
    assert g.check_rules() == []


def _features(value=1.0, n=1):
    from hexworld.minerals.features import FEATURES_RU

    return {k: np.full(n, value) for k in FEATURES_RU}


@pytest.mark.parametrize("mid", sorted(g.RULES))
def test_hard_gates_positive_negative_missing(mid):
    rule = g.RULES[mid]
    ok, unres, s = g.evaluate(rule, _features(1.0))
    assert ok[0] and not unres[0] and s[0] > 0
    for gate in rule.gates:
        f = _features(1.0)
        f[gate] = np.array([0.0])
        ok, unres, s = g.evaluate(rule, f)
        assert not ok[0] and not unres[0] and s[0] == 0.0, gate  # absent -> forbidden
        f[gate] = np.array([np.nan])
        ok, unres, s = g.evaluate(rule, f)
        assert not ok[0] and unres[0] and s[0] == 0.0, gate  # missing -> unresolved, not 0/1


def test_soft_factors_never_mask_a_hard_gate():
    rule = g.RULES["PORPHYRY"]
    f = _features(1.0)
    f["arc"] = np.array([0.0])
    _ok, _u, s = g.evaluate(rule, f)
    assert s[0] == 0.0  # a zero hard gate is not averaged away
    f = _features(1.0)
    f["felsic_intrusion"] = np.array([0.0])
    _ok, _u, s2 = g.evaluate(rule, f)
    assert 0 < s2[0] < 1.0


def test_features_on_a_real_world(hist_project):
    from hexworld.minerals.features import FEATURES_RU, features_from_project
    from hexworld.resources.deposits import context_from_project

    ctx = context_from_project(hist_project)
    F = features_from_project(hist_project, ctx)
    assert set(F) == set(FEATURES_RU)
    for k, v in F.items():
        assert v.shape == (ctx.n,), k
        assert set(np.unique(v[np.isfinite(v)])) <= {0.0, 1.0}, k
    assert np.all(np.isnan(F["copper_body"]))  # set by the plan, unresolved before
    elig = {m: int(g.evaluate(r, F)[0].sum()) for m, r in g.RULES.items()}
    assert elig["QUARRY_SED"] + elig["QUARRY_IGNEOUS"] > 0
    assert elig["MARINE_NODULE"] == 0 or F["deep_ocean"].sum() > 0
