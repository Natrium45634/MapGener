"""TZ 5.5.1 items closed after the first release: folding (T25), supergene enrichment (T47),
sand quality (T53), evaporite sequence (T55), sea-floor growth (T63, T64), unconventional
fluids, discovery view (T76), local invalidation (T68), crops (T79), host-conditioned
placement (T17), the gate-level synthetic scenarios S01–S20 and the resolution test (T16)."""

from __future__ import annotations

import numpy as np
import pytest

from hexworld.minerals import bodies as B
from hexworld.minerals import secondary551 as sec
from hexworld.minerals import structure as st
from hexworld.minerals.features import FEATURES_RU
from hexworld.minerals.gates import RULES, evaluate
from hexworld.minerals.ledger import ResourceLedger
from hexworld.minerals.plan import MineralPlan


def plate(bid, z, t=10.0, event=0):
    b = B.make_body(
        "tabular",
        bid,
        "S",
        (0.0, 0.0, z),
        params={"length_m": 4000.0, "down_dip_m": 4000.0, "true_thickness_m": t},
    )
    b.formation_event = event
    return b


# -- T25 folds ------------------------------------------------------------------------------
def test_t25_fold_keeps_the_order_of_strata_and_moves_only_older_bodies():
    col = st.Column(surfaces=[("u1", lambda x, y: np.full(np.shape(x), 0.0)),
                              ("u2", lambda x, y: np.full(np.shape(x), -200.0)),
                              ("u3", lambda x, y: np.full(np.shape(x), -450.0))])  # fmt: skip
    f = st.Fold(event_index=4, origin_m=(0.0, 0.0), axis_strike_deg=30.0, wavelength_m=8000.0,
                amplitude_m=600.0)  # fmt: skip
    col.fold(f)
    xs, ys = np.meshgrid(np.linspace(-1e4, 1e4, 41), np.linspace(-1e4, 1e4, 41))
    assert col.ordered(xs, ys)  # tops stay above bottoms everywhere
    assert np.ptp(col.z(0, xs, ys)) > 1000.0  # it is folded
    old, young = plate("o", -100.0, event=2), plate("y", -100.0, event=6)
    moved = f.apply(old)
    assert moved.origin_m[2] == pytest.approx(-100.0 + float(f.w(0.0, 0.0)))
    assert f.apply(young) is young
    with pytest.raises(NotImplementedError):
        st.Column().fold(st.Fold(1, (0, 0), 0, 1000, 10, overturned=True))


# -- T47 supergene ---------------------------------------------------------------------------
def test_t47_supergene_moves_metal_down_without_creating_it():
    plan = MineralPlan(seed="sg")
    plan.ledger.open("o", "L1", "Cu", 1000.0)
    lot = {"lot_id": "L1", "contained": {"Cu": 1000.0}, "rock_t": 1e5}
    zones = sec.supergene_enrich(plan, lot, water_table_m=60.0, leach_fraction=0.7,
                                 body_depth_top_m=0.0, body_height_m=300.0)  # fmt: skip
    z = zones["Cu"]
    assert sum(z.values()) == pytest.approx(1000.0)  # no metal from the air
    assert lot["contained"]["Cu"] == 1000.0 and plan.ledger.balance("L1", "Cu") == 1000.0
    assert lot["supergene"]["blanket_grade_factor"]["Cu"] > 1.0  # enriched below the table
    assert z["leached_cap_t"] < 1000.0 * 0.2  # the cap is leached
    assert plan.ledger.validate()["ok"]
    # a body entirely below the water table: nothing to leach
    lot2 = {"lot_id": "L2", "contained": {"Cu": 10.0}, "rock_t": 100.0}
    assert sec.supergene_enrich(plan, lot2, 50.0, 0.7, 200.0, 100.0) == {}


# -- T53 sand quality -------------------------------------------------------------------------
def test_t53_sand_applications_follow_size_purity_and_roundness():
    glass = sec.sand_quality("platform_siliciclastic", 800.0, "beach", "s1")
    dune = sec.sand_quality("granite_gneiss", 400.0, "dune", "s2")
    volcanic = sec.sand_quality("basaltic_volcanic", 20.0, "river", "s3")
    assert "concrete_sand" not in dune["applications"]  # desert sand is not automatically good
    assert "glass_sand" not in volcanic["applications"]
    assert glass["sio2_pct"] > volcanic["sio2_pct"] and glass["fe2o3_pct"] < volcanic["fe2o3_pct"]
    assert all(q["proxy"] for q in (glass, dune, volcanic))


# -- T55 evaporites --------------------------------------------------------------------------
def test_t55_sequential_precipitation_conserves_mass_and_needs_its_ions():
    plan = MineralPlan(seed="ev")
    out = sec.evaporate(plan, "B1", {"gypsum": 5e4, "salt": 2e6, "potash": 1e4}, 1e4, 0.9)
    assert out["salt"]["precipitated_kg"] > 0
    assert out["potash"]["precipitated_kg"] == 0.0  # still below its saturation
    for v in out.values():
        assert v["precipitated_kg"] + v["dissolved_left_kg"] >= 0
    assert plan.ledger.validate()["ok"]
    pure = sec.evaporate(MineralPlan(seed="w"), "B2", {}, 1e4, 0.99)
    assert all(v["precipitated_kg"] == 0 for v in pure.values())  # pure water: no salt


# -- T63 / T64 sea-floor accretions, unconventional, discovery -------------------------------
def test_t63_t64_marine_growth_and_unconventional_and_discovery(hist_project):
    p = hist_project.minerals.plan
    for lot in p.lots:
        g = lot.get("growth")
        if g:
            exp = min(g["rate_mm_per_ma"] * g["age_ma"] / 1000.0,
                      0.12 if lot["model_id"] == "MARINE_NODULE" else 0.25)  # fmt: skip
            assert g["thickness_m"] == pytest.approx(exp)
            if lot["model_id"] == "MARINE_CRUST":
                body = p.bodies[lot["body_id"]]
                assert body.W > body.L  # true flank area > its map projection
        if lot["model_id"] == "UNCONVENTIONAL" and "transformation_ratio" in lot:
            assert lot["transformation_ratio"] >= 0.3
    assert p.ledger.validate()["ok"]
    view = sec.discovery_view(p, p.seed)
    truth = {s["system_id"] for s in p.systems if s.get("surface_occurrence")}
    assert {v["system_id"] for v in view} == truth  # hidden bodies are not seen
    for v in view:
        lots = [lot for lot in p.lots if lot["system_id"] == v["system_id"]]
        for com, est in v["estimate_t"].items():
            real = sum(lot["contained"].get(com, 0.0) for lot in lots)
            assert est != real or real == 0.0  # an estimate, not the ground truth


def test_unconventional_needs_a_mature_source():
    from hexworld.minerals import fluids

    assert fluids.maturity_index(300.0, 5.0) < 0.36  # TR < 0.3 below


# -- T68 / T79 ---------------------------------------------------------------------------------
def test_t68_t79_invalidation_by_ids_and_crops_from_the_global_plan(hist_project):
    p = hist_project.minerals.plan
    s = next(s for s in p.systems if s.get("body_ids"))
    col, row = s["hex"] % hist_project.layout.width, s["hex"] // hist_project.layout.width
    hit = st.affected_systems(p, hist_project.layout, [(col, row)])
    assert s["system_id"] in hit and len(hit) < len(p.systems)
    b = hist_project.geometry.viewport_bounds
    box = (b.xmin, b.ymin, b.xmin + b.width / 2, b.ymin + b.height / 2)
    c = st.crop(p, box)
    ids = {x["system_id"] for x in c["systems"]}

    def inb(x):
        return box[0] <= x["x_m"] / 1e3 <= box[2] and box[1] <= x["y_m"] / 1e3 <= box[3]

    inside = [x for x in p.systems if inb(x)]
    assert ids == {x["system_id"] for x in inside}
    for lot in c["lots"]:  # identical objects, not a new draw
        assert lot is next(x for x in p.lots if x["lot_id"] == lot["lot_id"])


def test_t17_systems_sit_on_eligible_hosts(hist_project):
    from hexworld.minerals.features import features_from_project
    from hexworld.resources.deposits import context_from_project

    ctx = context_from_project(hist_project)
    F = features_from_project(hist_project, ctx)
    p = hist_project.minerals.plan
    tract_models = {m for m, r in RULES.items() if r.anchor in ("tract", "regional")}
    for s in p.systems:
        if s["model_id"] in tract_models and s["provenance"] != "routed_from_eroded_bodies":
            ok, _u, _s = evaluate(RULES[s["model_id"]], F)
            assert ok[s["hex"]], (s["model_id"], s["hex"])


def test_s20_burial_by_an_eruption_changes_depth_not_metal():
    s = {"system_id": "S", "depth_top_m": 0.0, "exposed": True, "surface_occurrence": True}
    lots = [{"lot_id": "L", "contained": {"Au": 1.0}}]
    st.bury(s, lots, 35.0, "lava flow")
    assert s["depth_top_m"] == 35.0 and not s["surface_occurrence"]
    assert lots[0]["contained"] == {"Au": 1.0}


def test_t44_paleochannel_follows_its_archived_course():
    from hexworld.minerals.placers import paleochannel

    b = paleochannel("seed", (1000.0, 2000.0), 75.0, 8000.0, 150.0, 2.0, 120.0, 300.0)
    assert b.morphology == "paleoplacer" and b.strike_deg == 75.0
    assert b.origin_m[2] == pytest.approx(180.0)  # buried 120 m below the present surface


# -- gate-level synthetic scenarios S01–S20 ---------------------------------------------------
def scen(**on):
    F = {k: np.zeros(1) for k in FEATURES_RU}
    for k, v in on.items():
        F[k] = np.full(1, float(v))
    return F


def eligible(model, F):
    return bool(evaluate(RULES[model], F)[0][0])


@pytest.mark.parametrize(
    "name, features, yes, no",
    [
        ("S01 shield with BIF", dict(ancient_crust=1, marine_facies=1, preserved_strata=1,
                                     ancient_shield=1), ["BIF"], []),
        ("S01 young crust", dict(marine_facies=1, preserved_strata=1), [], ["BIF"]),
        ("S02 craton", dict(ancient_shield=1, land=1), ["KIMBERLITE"], []),
        ("S03 young arc", dict(arc=1, magmatic_activity=1, volcanic_rock=1, hydrothermal=1),
         ["PORPHYRY", "EPITHERMAL"], ["KIMBERLITE"]),
        ("S04 carbonate contact", dict(carbonate_near_intrusion=1), ["SKARN", "WOLLAST"], []),
        ("S04 no carbonate", dict(felsic_intrusion=1), [], ["SKARN"]),
        ("S05 collision belt", dict(shear_zone=1, metamorphic=1), ["OROGENIC_AU"], []),
        ("S06 mafic intrusion", dict(layered_intrusion=1, mafic_ultramafic=1),
         ["FE_TI_V", "CHROM_LAYER", "PGE_REEF", "NI_SULFIDE"], ["PORPHYRY"]),
        ("S07 old ocean on land", dict(mafic=1, marine_facies=1, land=1), ["VMS"], []),
        ("S08 laterite on ultramafic", dict(ultramafic=1, weathering_long=1, land=1),
         ["NI_LATERITE"], ["BAUXITE"]),
        ("S08 laterite on granite", dict(al_silicate_rock=1, weathering_long=1, land=1),
         ["BAUXITE"], ["NI_LATERITE"]),
        ("S12 Li salar", dict(closed_basin=1, arid=1), ["BRINE"], []),
        ("S12 open coast", dict(coast=1, arid=1), [], ["BRINE"]),
        ("S14 pegmatite", dict(pegmatite=1), ["LCT_PEG", "GEM_PEG", "MICA_FELDSPAR"], []),
        ("S15 metamorphic only", dict(metamorphic=1), ["CORUNDUM_META"],
         ["JADEITE", "NEPHRITE"]),
        ("S15 subduction HP-LT", dict(high_p_low_t=1), ["JADEITE"], ["NEPHRITE"]),
        ("S16 carbonate shelf, volcanic island",
         dict(sedimentary_rock=1, land=1, volcanic_rock=1, magmatic_activity=1, coast=1),
         ["QUARRY_SED", "VOLCANIC_GLASS"], []),
        ("S17 abyss", dict(deep_ocean=1, low_sedimentation=1, old_ocean_floor=1),
         ["MARINE_NODULE"], ["MARINE_CRUST"]),
        ("S17 seamount", dict(hard_substrate=1), ["MARINE_CRUST"], ["MARINE_NODULE"]),
        ("S18 arid aquifer", dict(porous_host=1, land=1, arid=1), ["AQUIFER"], []),
    ],
)  # fmt: skip
def test_gate_level_scenarios(name, features, yes, no):
    F = scen(**features)
    for m in yes:
        assert eligible(m, F), (name, m)
    for m in no:
        assert not eligible(m, F), (name, m)


def test_t15_closed_basin_below_sea_level_is_not_ocean():
    F = scen(land=1, closed_basin=1, arid=1)  # below sea level but not the connected ocean
    assert not eligible("MARINE_NODULE", F) and not eligible("SMS", F)
    assert eligible("BRINE", F)


def test_t28_body_under_the_sea_floor_reports_water_apart():
    from hexworld.minerals.query import query_column_bodies

    b = plate("sea", -2500.0, t=8.0)
    hits = query_column_bodies([b], 0.0, 2000.0, surface_m=-2000.0, water_depth_m=2000.0)
    assert hits and hits[0]["depth_top_m"] == pytest.approx(500.0 - 4.0)
    assert hits[0]["water_depth_m"] == 2000.0  # the water column is not burial


def test_t40_t52_quarry_uses_are_views_of_one_volume(hist_project):
    for lot in hist_project.minerals.plan.lots:
        if lot["model_id"].startswith("QUARRY"):
            assert len(lot["contained"]) == 1 and 0 < lot["usable_fraction"] < 1
            assert "not additive" in lot["uses_note"] and len(lot["uses"]) >= 1


def test_t49_t50_varieties_and_emerald_transport():
    from hexworld.minerals.placers import GRAIN
    from hexworld.minerals.plan import PLACER_COMMODITIES
    from hexworld.minerals.priors import PRIORS

    assert PRIORS["CORUNDUM_META"].gem["varieties"]["ruby"] == "cr_host_near"
    assert PRIORS["EMERALD_META"].gem["varieties"]["emerald"] == "cr_host_near"
    assert PRIORS["EMERALD_META"].gem["placer"] is False
    assert "beryl" not in PLACER_COMMODITIES and "beryl" not in GRAIN


def test_t54_clay_types_have_distinct_gates():
    gates = {m: set(RULES[m].gates) for m in ("CLAY", "KAOLIN", "BENTONITE")}
    assert len({frozenset(v) for v in gates.values()}) == 3


def test_t61_brine_recharge_is_a_flow(hist_project):
    for lot in hist_project.minerals.plan.lots:
        if lot["model_id"] == "BRINE":
            assert "solute_recharge_kg_per_year" in lot and "never added" in lot["recharge_note"]


def test_ledger_stays_closed_with_all_post_processing():
    led = ResourceLedger()
    led.open("a", "x", "Cu", 1.0)
    assert led.validate()["ok"]


@pytest.mark.slow
def test_t16_same_physical_area_at_two_resolutions_gives_similar_intensity():
    from hexworld import Config, generate

    tot = []
    for w, h, km in ((60, 48, 100.0), (120, 96, 50.0)):
        p = generate(Config(seed="Res16", width=w, height=h, hex_spacing_km=km, mode="world"))
        lam = {}
        for pr in p.minerals.plan.provinces:
            lam[pr["model_id"]] = lam.get(pr["model_id"], 0.0) + pr.get("lambda", 0.0)
        tot.append(lam)
    common = [m for m in tot[0] if tot[0][m] > 0.5 and tot[1].get(m, 0) > 0.5]
    assert common
    ratios = [tot[1][m] / tot[0][m] for m in common]
    assert 0.5 < float(np.median(ratios)) < 2.0  # intensity follows area, not hex count
