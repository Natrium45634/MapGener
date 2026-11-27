"""TZ 5.2 part III: geological resources - reference calculations N11-N20 and tests
R01-R26 (synthetic geological snapshots + the generated test world)."""

from __future__ import annotations

import math

import numpy as np
import pytest

from hexworld.resources import deposits as dep
from hexworld.resources import models as mo
from hexworld.resources import secondary as sec

L_ = dep.LITH


# -- reference calculations ------------------------------------------------------------------------
def test_n11_n12_n13_ellipsoid_mass_and_metal():
    v = mo.ellipsoid_volume(100.0, 50.0, 20.0)
    assert v == pytest.approx(418879.020, abs=1e-3)
    m = mo.ore_mass_t(v, 2700.0)
    assert m == pytest.approx(1130973.36, abs=0.01)
    assert mo.contained_t(m, 0.01) == pytest.approx(11309.73, abs=0.01)
    with pytest.raises(ValueError):
        mo.contained_t(m, 1.2)


def test_n14_n15_units():
    assert mo.ppm_to_fraction(1000.0) == pytest.approx(0.001)
    assert mo.ppm_to_fraction(1000.0) * 1000.0 == pytest.approx(1.0)  # 1 kg per t
    assert mo.contained_t(1e6, mo.g_per_t_to_fraction(2.0)) == pytest.approx(2.0)


def test_n16_brine_is_water_times_concentration():
    assert mo.brine_mass_kg(1e6, 100.0) == pytest.approx(100000.0)


def test_n18_recoverable_does_not_change_in_situ():
    insitu = mo.contained_t(100.0, 0.01)
    rec = mo.recoverable_t(insitu, 0.5, 1.0)
    assert insitu == pytest.approx(1.0) and rec == pytest.approx(0.5)


def test_n19_zero_intensity_gives_zero_deposits():
    assert mo.poisson_from_uniform(0.0, 0.999999) == 0
    ctx = _ctx()
    cat = dep.plan_deposit_systems(ctx, None, lambda_scale=0.0)
    assert cat["systems"] == []


def test_poisson_inversion_has_the_right_mean():
    u = (np.arange(20000) + 0.5) / 20000
    assert np.mean([mo.poisson_from_uniform(3.7, x) for x in u]) == pytest.approx(3.7, rel=0.01)


# -- synthetic geological snapshot -----------------------------------------------------------------
def _ctx(shape=(20, 30), spacing=50.0, **over):
    h, w = shape
    n = h * w
    xs, ys = np.meshgrid(np.arange(w) * spacing * 0.866 + 30, np.arange(h) * spacing + 30)
    base = dict(
        seed="Res52",
        shape=shape,
        hex_area_km2=0.866 * spacing**2,
        spacing_km=spacing,
        duration_ma=30.0,
        x_km=xs.ravel().astype(float),
        y_km=ys.ravel().astype(float),
        land=np.ones(n, bool),
        ocean=np.zeros(n, bool),
        surface_m=np.full(n, 300.0),
        water_depth_m=np.zeros(n),
        rock=np.full(n, 2, np.uint8),
        lith=np.full(n, L_["platform_siliciclastic"], np.uint8),
        crust_age_ma=np.full(n, 800.0),
        cont_frac=np.ones(n),
        orogen=np.zeros(n, bool),
        orogen_age_ma=np.full(n, np.nan),
        collision=np.zeros(n, bool),
        arc_side=np.zeros(n, bool),
        erosion_total_m=np.zeros(n),
        sediment_m=np.full(n, 500.0),
        basin_type=np.zeros(n, np.uint8),
        basin_fill_m=np.zeros(n),
        slope_m_km=np.full(n, 2.0),
        shelf=np.zeros(n, bool),
        upwelling=np.zeros(n, bool),
        seamount=np.zeros(n, bool),
        marine_sed_m=np.zeros(n),
        ocean_crust_age_ma=np.full(n, np.nan),
        volcano_systems=[],
    )
    base.update(over)
    return dep.GeoContext(**base)


def _arc_volcano(i, ctx, sio2=62.0, status="active_system", env="land", src="subduction_arc"):
    return {
        "volcano_id": f"V-test{i}",
        "province_id": "P-arc",
        "source_type": src,
        "environment": env,
        "magmatic_status": status,
        "x_km": float(ctx.x_km[i]),
        "y_km": float(ctx.y_km[i]),
        "formed_ma": 2.0,
        "magma": {"silica_wt_pct": sio2},
    }


def _models(cat):
    return {s["model_id"] for s in cat["systems"]}


def test_r01_no_porphyry_without_an_intrusion():
    ctx = _ctx(lith=np.full(600, L_["granite_gneiss"], np.uint8), rock=np.full(600, 1, np.uint8))
    cat = dep.plan_deposit_systems(ctx, None, lambda_scale=10.0)
    assert "R01" not in _models(cat)
    # basaltic hotspot magma is not a porphyry source either
    ctx.volcano_systems = [
        _arc_volcano(i, ctx, sio2=48.0, src="oceanic_hotspot") for i in range(0, 600, 7)
    ]
    assert "R01" not in _models(dep.plan_deposit_systems(ctx, None, lambda_scale=10.0))


def test_r02_r19_arc_porphyry_with_correlated_commodities():
    ctx = _ctx(erosion_total_m=np.zeros(600))
    ctx.volcano_systems = [_arc_volcano(i, ctx) for i in range(0, 600, 3)]
    cat = dep.plan_deposit_systems(ctx, {}, lambda_scale=8.0)
    por = [s for s in cat["systems"] if s["model_id"] == "R01"]
    assert por, "an arc with intermediate magma permits porphyry systems"
    s = por[0]
    assert {"Cu", "Mo", "Au"} <= set(s["contained_t"])  # one system, several commodities
    bodies = [b for b in cat["bodies"] if b["system_id"] == s["deposit_id"]]
    assert all(set(b["grades"]) == set(s["grades"]) for b in bodies)
    # R19: within a system the grades share one draw: ln Cu and ln Au correlate
    z = np.random.default_rng(1).standard_normal((4000, 5))
    g = np.array([list(mo.sample_grade_tonnage(mo.MODELS["R01"], zz)[1].values())[:3] for zz in z])
    r = np.corrcoef(np.log(g[:, 0]), np.log(g[:, 2]))[0, 1]
    assert r > 0.25


def test_r03_r15_ancient_vms_on_land_and_kept_after_uplift():
    n = 600
    ctx = _ctx(
        lith=np.full(n, L_["greenstone_belt"], np.uint8),
        rock=np.full(n, 1, np.uint8),
        crust_age_ma=np.full(n, 2700.0),
    )
    pros = dep.tract_suitability(ctx)
    assert pros["R05"][0].all()  # former sea-floor volcanics permit VMS on today's land
    cat = dep.plan_deposit_systems(ctx, {"R05": pros["R05"]}, lambda_scale=20.0)
    vms = [s for s in cat["systems"] if s["model_id"] == "R05"]
    assert vms and all(s["provenance"] == "inherited_stratigraphy" for s in vms)
    # R15: no erosion at the site -> the in-situ mass is the formed mass
    for s in vms:
        assert s["ore_t"] == pytest.approx(s["ore_t_formed"] * (1 - s["eroded_fraction"]), rel=1e-9)


def test_r04_laterite_needs_an_ultramafic_protolith():
    n = 600
    ctx = _ctx(lith=np.full(n, L_["granite_gneiss"], np.uint8), slope_m_km=np.full(n, 1.0))
    w = np.full(n, 0.95)
    age = np.full(n, 2e6)
    tr = sec.weathering_tracts(ctx, w, age, np.full(n, 26.0), np.full(n, 2500.0), np.zeros(n, bool))
    assert not tr["R18"][0].any()
    assert tr["R17"][0].any()  # bauxite is possible on aluminosilicate rock
    ctx.lith[:] = L_["ophiolite_ultramafic"]
    tr = sec.weathering_tracts(ctx, w, age, np.full(n, 26.0), np.full(n, 2500.0), np.zeros(n, bool))
    assert tr["R18"][0].all()
    # not on fresh rock: young soils give no laterite
    tr = sec.weathering_tracts(
        ctx,
        np.full(n, 0.2),
        np.full(n, 500.0),
        np.full(n, 26.0),
        np.full(n, 2500.0),
        np.zeros(n, bool),
    )
    assert not tr["R18"][0].any() and not tr["R17"][0].any()


def test_r05_carbonatite_needs_an_alkaline_continental_setting():
    ctx = _ctx(crust_age_ma=np.full(600, 200.0))
    ctx.volcano_systems = [
        _arc_volcano(i, ctx, sio2=45.0, src="continental_hotspot") for i in range(0, 600, 2)
    ]
    assert "R11" not in _models(dep.plan_deposit_systems(ctx, {}, lambda_scale=15.0))  # young crust
    ctx.crust_age_ma[:] = 1800.0
    assert "R11" in _models(dep.plan_deposit_systems(ctx, {}, lambda_scale=15.0))
    ctx.volcano_systems = [_arc_volcano(i, ctx, sio2=62.0) for i in range(0, 600, 2)]
    assert "R11" not in _models(dep.plan_deposit_systems(ctx, {}, lambda_scale=15.0))  # arcs


def test_r06_r07_kimberlite_and_bif_need_old_crust():
    n = 600
    young = _ctx(
        rock=np.full(n, 1, np.uint8),
        lith=np.full(n, L_["granite_gneiss"], np.uint8),
        crust_age_ma=np.full(n, 600.0),
    )
    p = dep.tract_suitability(young)
    assert not p["R22"][0].any() and not p["R15"][0].any()
    old = _ctx(
        rock=np.full(n, 1, np.uint8),
        lith=np.full(n, L_["greenstone_belt"], np.uint8),
        crust_age_ma=np.full(n, 2800.0),
    )
    p = dep.tract_suitability(old)
    assert p["R22"][0].all() and p["R15"][0].all()
    cat = dep.plan_deposit_systems(old, {"R15": p["R15"], "R22": p["R22"]}, lambda_scale=30.0)
    for s in cat["systems"]:
        assert s["provenance"] == "inherited_stratigraphy"  # not formed in the 30 Ma history
        assert s["formation_epoch_ma"] > old.duration_ma


def test_r08_r09_coal_and_oil_need_their_basin_history():
    n = 600
    ctx = _ctx(basin_type=np.full(n, 2, np.uint8), basin_fill_m=np.full(n, 600.0))
    p = dep.tract_suitability(ctx)
    assert not p["R24"][0].any()  # a basin without coal measures has no coal
    assert not p["R25"][0].any()  # nor oil: a sedimentary basin alone is not enough
    ctx.lith[:] = L_["coal_measures"]
    ctx.basin_fill_m[:] = 3500.0
    p = dep.tract_suitability(ctx)
    assert p["R24"][0].all() and p["R25"][0].all()
    assert mo.MODELS["R25"].priority == "P1"


def test_r10_r11_closed_basin_salts_and_li_brines():
    n = 600
    ctx = _ctx(sediment_m=np.full(n, 200.0))
    lake = np.full(n, -1)
    lake[100:110] = 3
    endo = lake >= 0
    q = np.zeros(n)
    q[100:110] = 5.0
    basin = np.zeros(n, int)
    arid = np.full(n, 0.1)
    depth = np.where(endo, 2.0, 0.0)
    no_src = np.zeros(n, bool)
    sys1, _b, rows = sec.closed_basin_salts(ctx, lake, endo, q, basin, arid, depth, no_src)
    assert rows and rows[0]["salt_delivered_kg"] == pytest.approx(
        rows[0]["salt_dissolved_kg"] + rows[0]["salt_precipitated_kg"]
    )
    assert not any(s["model_id"] == "R27" for s in sys1)  # no Li source -> no brine deposit
    src = np.zeros(n, bool)
    src[:50] = True  # felsic volcanics in the catchment
    sys2, _b, rows2 = sec.closed_basin_salts(ctx, lake, endo, q, basin, arid, depth, src)
    li = [s for s in sys2 if s["model_id"] == "R27"]
    assert li
    r = rows2[0]
    assert li[0]["contained_t"]["Li"] * 1000 == pytest.approx(
        mo.brine_mass_kg(r["pore_water_m3"], r["li_mg_l"])
    )  # pore water x concentration
    assert r["li_brine_kg"] <= r["li_delivered_kg"] + 1e-6  # never more than delivered
    # R11: no inflow -> no salt at all
    sys3, _b, rows3 = sec.closed_basin_salts(ctx, lake, endo, np.zeros(n), basin, arid, depth, src)
    assert not sys3 and not rows3
    # a humid closed lake does not concentrate
    sys4, _b, _r = sec.closed_basin_salts(ctx, lake, endo, q, basin, np.full(n, 1.2), depth, src)
    assert not sys4


def test_r12_sandstone_uranium_in_permeable_basins():
    n = 600
    ctx = _ctx(sediment_m=np.full(n, 1500.0))
    p = dep.tract_suitability(ctx)
    assert p["R13"][0].all()
    ctx2 = _ctx(sediment_m=np.full(n, 50.0))
    assert not dep.tract_suitability(ctx2)["R13"][0].any()


def _chain_world(n=40):
    """A river 0 -> 1 -> ... -> n-1 (sea at the end), slope dropping downstream."""
    ctx = _ctx(shape=(1, n), spacing=20.0)
    rc = np.append(np.arange(1, n), -1)
    slope = np.linspace(30.0, 0.5, n)
    ocean = np.zeros(n, bool)
    ocean[-1] = True
    return ctx, rc, slope, ocean


def test_r13_r14_placers_need_an_eroded_source_and_respect_its_mass():
    ctx, rc, slope, ocean = _chain_world()
    lake = np.zeros(ctx.n, bool)
    river = np.ones(ctx.n, bool)
    none = {"eroded": [], "systems": [], "bodies": []}
    sys0, _b, led0 = sec.river_placers(ctx, none, rc, slope, lake, ocean, river, 20.0)
    assert sys0 == [] and led0 == {}
    cat = {
        "eroded": [
            {
                "system_id": "D-src",
                "model_id": "R04",
                "x_km": 0.0,
                "y_km": 0.0,
                "hex": 0,
                "eroded_ore_t": 1e6,
                "eroded_commodity_t": {"Au": 10.0},
                "formed_ma": 50.0,
            }
        ]
    }
    sys1, bod1, led1 = sec.river_placers(ctx, cat, rc, slope, lake, ocean, river, 20.0)
    led = led1["Au"]
    assert led["released_t"] == pytest.approx(10.0 * sec.ETA_RELEASE)
    assert abs(led["residual_t"]) < 1e-12
    placed = sum(s["contained_t"]["Au"] for s in sys1)
    assert 0 < placed <= led["deposited_t"] + 1e-12 <= led["released_t"] + 1e-12
    assert all(s["parent_event_ids"] == ["D-src"] for s in sys1)


def test_r16_r17_marine_nodules_and_crusts_by_setting():
    n = 600
    ocean = np.ones(n, bool)
    depth = np.full(n, 5200.0)
    seam = np.zeros(n, bool)
    seam[:60] = True
    depth[:60] = 1500.0
    ctx = _ctx(
        land=~ocean,
        ocean=ocean,
        water_depth_m=depth,
        rock=np.zeros(n, np.uint8),
        lith=np.full(n, L_["oceanic_basalt"], np.uint8),
        seamount=seam,
        marine_sed_m=np.full(n, 200.0),
        ocean_crust_age_ma=np.full(n, 60.0),
    )
    p = dep.tract_suitability(ctx)
    nod, crust = p["R30"][0], p["R31"][0]
    assert not nod[:60].any() and nod[60:].all()  # nodules on the abyssal plain only
    assert crust[:60].all() and not crust[60:].any()  # crusts on seamount rock only
    shallow = _ctx(
        land=~ocean, ocean=ocean, water_depth_m=np.full(n, 300.0), rock=np.zeros(n, np.uint8)
    )
    assert not dep.tract_suitability(shallow)["R30"][0].any()


def test_r18_one_deposit_over_several_hexes_keeps_one_id():
    ctx = _ctx()
    body = {
        "body_id": "D-x-B0",
        "system_id": "D-x",
        "shape": "sheet",
        "center_m": [float(ctx.x_km[310] * 1000), float(ctx.y_km[310] * 1000), 0.0],
        "axes_m": [90000.0, 60000.0, 5.0],
        "strike_deg": 0.0,
        "n_veins": 1,
    }
    offs, idx = dep.hex_index([body], ctx.x_km, ctx.y_km, ctx.spacing_km)
    hexes = np.nonzero(np.diff(offs))[0]
    assert hexes.size > 3 and set(idx.tolist()) == {0}


def test_r21_r22_i09_plan_is_stable():
    ctx = _ctx()
    ctx.volcano_systems = [_arc_volcano(i, ctx) for i in range(0, 600, 3)]
    a = dep.plan_deposit_systems(ctx, None, lambda_scale=5.0)
    ctx.volcano_systems = list(reversed(ctx.volcano_systems))  # another iteration order
    b = dep.plan_deposit_systems(ctx, None, lambda_scale=5.0)

    def ids(c):
        return [(s["deposit_id"], round(s["ore_t"], 6), s["x_km"], s["y_km"]) for s in c["systems"]]

    assert ids(a) == ids(b) and ids(a)


def test_r23_erosion_removes_only_the_eroded_cap():
    n = 600
    ctx = _ctx(
        lith=np.full(n, L_["granitoid_intrusion"], np.uint8),
        rock=np.full(n, 3, np.uint8),
        orogen=np.ones(n, bool),
        cont_frac=np.ones(n),
        sediment_m=np.zeros(n),
    )
    found = None
    for e in np.linspace(0.0, 30000.0, 600):
        ctx.erosion_total_m[:] = e
        s, b, rec = dep.build_system(
            ctx, "R01", float(ctx.x_km[10]), float(ctx.y_km[10]), 10, ("test", 1), "object"
        )
        if s is not None and rec is not None and 0 < s["eroded_fraction"] < 1:
            found = (s, b, rec)
            break
    assert found, "some erosion depth cuts the porphyry body"
    s, bodies, rec = found
    assert rec["eroded_ore_t"] + s["ore_t"] == pytest.approx(s["ore_t_formed"], rel=1e-9)
    for el, m in rec["eroded_commodity_t"].items():
        assert m + s["contained_t"][el] == pytest.approx(
            s["ore_t_formed"] * s["grades"][el], rel=1e-9
        )
    b0 = bodies[0]
    a, bb, c = b0["axes_m"]
    h = b0["center_m"][2] + c - b0["eroded_above_m"]
    if h > 0:
        frac = mo.ellipsoid_cap_volume(a, bb, c, h) / mo.ellipsoid_volume(a, bb, c)
        # the stored centre is rounded to 1 cm: the cap of a thin cut differs by ~1e-3
        assert b0["erosion_fraction"] == pytest.approx(frac, rel=5e-3)


def test_r24_burial_changes_depth_not_tonnage():
    n = 600
    out = []
    for sed in (0.0, 2000.0):
        ctx = _ctx(
            rock=np.full(n, 2, np.uint8),
            sediment_m=np.full(n, sed),
            lith=np.full(n, L_["granitoid_intrusion"], np.uint8),
        )
        s, _b, _r = dep.build_system(ctx, "R12", 100.0, 100.0, 50, ("burial", 1), "orogen")
        out.append(s)
    assert out[0]["ore_t"] == pytest.approx(out[1]["ore_t"], rel=1e-12)
    assert out[1]["depth_top_m"] == pytest.approx(out[0]["depth_top_m"] + 2000.0, rel=1e-9)


def test_r25_catalogue_invariants(hist_project):
    cat = hist_project.resources.catalog
    assert hist_project.resources.validation["ok"], hist_project.resources.validation
    for b in cat["bodies"]:
        assert all(np.isfinite(v) and v >= 0 for v in [b["ore_t"], b["volume_m3"], *b["axes_m"]])
        for el, g in b["grades"].items():
            if not el.endswith("_mg_per_l"):
                assert 0.0 <= g <= 1.0
        if b["shape"] != "evaporite_bed":
            assert sum(b["contained_t"].values()) <= b["ore_t"] * (1 + 1e-9)


def test_r26_grade_tonnage_follows_the_model():
    m = mo.MODELS["R01"]
    z = np.random.default_rng(7).standard_normal((20000, len(m.grades) + 1))
    draws = [mo.sample_grade_tonnage(m, zz) for zz in z]
    t = np.array([d[0] for d in draws])
    cu = np.array([d[1]["Cu"] for d in draws])
    assert np.median(t) == pytest.approx(m.tonnage_median_t, rel=0.05)
    assert np.median(cu) == pytest.approx(m.grades["Cu"][0], rel=0.05)
    assert cu.max() <= m.max_grade["Cu"]
    r = np.corrcoef(np.log(t), np.log(cu))[0, 1]
    assert r == pytest.approx(m.rho_tonnage_grade, abs=0.05)
    assert np.log(t).std() == pytest.approx(m.tonnage_log_sd, rel=0.05)


def test_geometry_queries_and_sections():
    body = {
        "body_id": "B",
        "system_id": "S",
        "shape": "lens",
        "center_m": [0.0, 0.0, -500.0],
        "axes_m": [100.0, 50.0, 20.0],
        "strike_deg": 0.0,
        "n_veins": 1,
    }
    top, bot = mo.body_interval(body, np.array([0.0, 60.0, 101.0]), np.array([0.0, 0.0, 0.0]))
    assert top[0] == pytest.approx(-480.0) and bot[0] == pytest.approx(-520.0)
    assert top[1] - bot[1] == pytest.approx(2 * 20 * math.sqrt(1 - 0.36))
    assert np.isnan(top[2])  # outside the body: no "ore from the potential zone"
    hits = mo.query_bodies_at([body], 0.0, 0.0, surface_m=0.0)
    assert hits and hits[0]["depth_top_m"] == pytest.approx(480.0)
    assert mo.query_bodies_at([body], 0.0, 0.0, depth_max_m=400.0, surface_m=0.0) == []
    iv = mo.section_intervals(body, np.linspace(-200, 200, 41), np.zeros(41))
    assert len(iv) == 21  # x in [-100, 100] at a 10 m step (the rim has zero thickness)


@pytest.mark.slow
def test_n20_r20_same_world_two_resolutions():
    """Object-anchored systems keep ids and world coordinates across hex sizes."""
    from hexworld import Config, generate

    bounds = (0.0, 0.0, 6000.0, 4200.0)
    pa = generate(
        Config(seed="Terra7", width=80, height=54, hex_spacing_km=75.0, simulation_bounds_km=bounds)
    )
    pb = generate(
        Config(
            seed="Terra7", width=120, height=80, hex_spacing_km=50.0, simulation_bounds_km=bounds
        )
    )

    def anchored(p):
        return {
            s["deposit_id"]: (s["x_km"], s["y_km"])
            for s in p.resources.catalog["systems"]
            if s["anchor_kind"] == "object"
        }

    a, b = anchored(pa), anchored(pb)
    common = set(a) & set(b)
    assert common, "anchored systems exist in both"
    for k in common:
        assert a[k] == pytest.approx(b[k], abs=1e-6)
    share = len(common) / max(len(set(a) | set(b)), 1)
    print("N20: anchored systems", len(a), len(b), "common share", round(share, 3))
    assert share >= 0.8


def test_province_ids_stay_unique_when_centroids_share_a_cell():
    used: set = set()
    a = dep._unique_pid("P-R31-abc", used)
    b = dep._unique_pid("P-R31-abc", used)
    c = dep._unique_pid("P-R31-abc", used)
    assert (a, b, c) == ("P-R31-abc", "P-R31-abc-2", "P-R31-abc-3")
