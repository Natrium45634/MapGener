"""TZ 5.5.1 G2: body geometry (T18–T24, T26, T27, T29, T30, T34–T37, T70) and the numeric
fixtures N01, N04, N05, N21."""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest

from hexworld.geometry import HexLayout
from hexworld.minerals import bodies as B

FIX = {f["id"]: f for f in json.loads(
    (Path(__file__).parent / "data" / "v551" / "fixture_examples.json").read_text("utf-8")
)["fixtures"]}  # fmt: skip


def vein(dip=60.0, strike=30.0, L=500.0, W=200.0, t=2.0, taper="none", **kw):
    params = {"length_m": L, "down_dip_m": W, "true_thickness_m": t, "taper": taper, **kw}
    return B.make_body("vein", "V1", "S1", (1000.0, 2000.0, 0.0), strike_deg=strike,
                       dip_deg=dip, params=params)  # fmt: skip


def test_n01_vein_volume_analytic_and_integrated():
    v = vein()
    vol = v.volume(tol=1e-6)
    assert vol["analytic_m3"] == FIX["N01"]["expected"]
    assert abs(vol["volume_m3"] - FIX["N01"]["expected"]) / FIX["N01"]["expected"] < 1e-8


def test_t18_vertical_intercept_of_a_dipping_vein_has_finite_ends():
    v = vein(dip=60.0)
    assert v.intercepts()["vertical_intercept_m"] == pytest.approx(FIX["N04"]["expected"])
    # a vertical line through the middle of the plate: the drilled interval = t / cos(dip)
    s, d, _n = v.axes()
    mid = np.asarray(v.origin_m) + 0.0 * s + 100.0 * d
    col = v.column(mid[0], mid[1])
    assert len(col) == 1
    assert col[0][0] - col[0][1] == pytest.approx(4.0, rel=1e-9)
    # beyond the strike end: nothing (finite geometry, no infinite plane)
    far = np.asarray(v.origin_m) + 400.0 * s + 100.0 * d
    assert v.column(far[0], far[1]) == []


def test_t19_horizontal_bed_true_equals_vertical_thickness():
    bed = B.make_body("tabular", "T1", "S1", (0.0, 0.0, -50.0), strike_deg=0.0, dip_deg=0.0,
                      params={"length_m": 1000.0, "down_dip_m": 800.0,
                              "true_thickness_m": 12.0})  # fmt: skip
    assert bed.intercepts()["vertical_intercept_m"] == 12.0
    col = bed.column(10.0, -300.0 if False else 300.0 - 400.0 + 400.0)
    # the plate spans d in [0, 800] towards strike+90° = east: x in [0, 800]
    col = bed.column(400.0, 0.0)
    assert col and col[0][0] - col[0][1] == pytest.approx(12.0)


def test_t20_vertical_vein_and_a_parallel_ray():
    v = vein(dip=90.0, L=300.0, W=150.0, t=1.5)
    assert math.isinf(v.intercepts()["vertical_intercept_m"])
    s, d, n = v.axes()
    # a vertical ray inside the vein plane: the finite down-dip extent clips it
    p = np.asarray(v.origin_m) + 10.0 * s
    col = v.column(p[0], p[1])
    assert len(col) == 1 and col[0][0] - col[0][1] == pytest.approx(150.0)
    # a horizontal drill hole perpendicular to the vein: the true thickness
    q = np.asarray(v.origin_m) + 10.0 * s + 50.0 * d - 100.0 * n
    iv = v.ray_intervals(q, n)
    assert len(iv) == 1 and iv[0][1] - iv[0][0] == pytest.approx(1.5)
    # a ray parallel to the plane but outside the slab: none
    assert v.ray_intervals(np.asarray(v.origin_m) + 5.0 * n, s) == []


def test_t21_pinchout_to_zero_never_negative():
    v = vein(taper="ends")
    s = np.linspace(-260, 260, 53)
    t = v.thickness(s, np.full_like(s, 100.0))
    assert t.min() >= 0.0 and t[0] == 0.0 and t[-1] == 0.0
    assert t[len(t) // 2] == pytest.approx(2.0, rel=1e-6)
    assert v.volume(tol=1e-3)["volume_m3"] < 200000.0


def test_t22_crossing_veins_union_counted_once():
    a = vein(dip=90.0, strike=0.0, L=400.0, W=100.0, t=2.0)
    b = B.make_body("vein", "V2", "S1", a.origin_m, strike_deg=90.0, dip_deg=90.0,
                    params={"length_m": 400.0, "down_dip_m": 100.0,
                            "true_thickness_m": 3.0})  # fmt: skip
    net = B.make_body("vein_network", "N1", "S1", a.origin_m)
    net.veins = [a, b]
    union = net.volume(n0=32, tol=1e-3)["volume_m3"]
    # analytic: 2*400*100 + 3*400*100 - overlap 2*3*100
    assert union == pytest.approx(80000 + 120000 - 600, rel=2e-3)
    assert B.plate_overlap(a, b) == pytest.approx(600.0, rel=0.03)
    col = net.column(*np.asarray(a.origin_m)[:2])
    assert len(col) == 1  # merged, not two stacked intervals


def test_t23_ellipsoid_volume_and_ray():
    e = B.make_body("lens", "E1", "S1", (0.0, 0.0, -500.0), strike_deg=40.0, dip_deg=35.0,
                    params={"semi_axes_m": [50.0, 20.0, 10.0]})  # fmt: skip
    vol = e.volume(tol=1e-4)
    assert vol["analytic_m3"] == pytest.approx(FIX["N05"]["expected"], rel=1e-12)
    assert abs(vol["volume_m3"] - vol["analytic_m3"]) / vol["analytic_m3"] < 0.005
    iv = e.ray_intervals((0.0, 0.0, 0.0), (0, 0, -1))
    assert iv and iv[0][1] > iv[0][0]


def test_pipe_volume_and_column():
    p = B.make_body("mantle_pipe", "K1", "S1", (0.0, 0.0, 100.0),
                    params={"r_top_m": 300.0, "r_bottom_m": 60.0, "depth_m": 1200.0})  # fmt: skip
    vol = p.volume(tol=1e-4)
    assert abs(vol["volume_m3"] - vol["analytic_m3"]) / vol["analytic_m3"] < 0.005
    col = p.column(0.0, 0.0)
    assert col == [(100.0, -1100.0)]
    top, bot = p.column(200.0, 0.0)[0]
    assert top == 100.0 and bot == pytest.approx(100.0 - (200 - 300) / (60 - 300) * 1200)
    assert p.column(400.0, 0.0) == []


def test_t24_stockwork_vein_fraction_bounds():
    from hexworld.minerals.morphology import MorphologyError, validate_params

    validate_params("stockwork", {"semi_axes_m": [500, 400, 300], "vein_fraction": 0.08})
    with pytest.raises(MorphologyError):
        validate_params("stockwork", {"semi_axes_m": [500, 400, 300], "vein_fraction": 1.2})


def test_t29_stacked_bodies_at_one_point():
    a = B.make_body(
        "tabular",
        "A",
        "S1",
        (0.0, 0.0, -100.0),
        params={"length_m": 2000.0, "down_dip_m": 2000.0, "true_thickness_m": 5.0},
    )
    b = B.make_body(
        "tabular",
        "B",
        "S2",
        (0.0, 0.0, -900.0),
        params={"length_m": 2000.0, "down_dip_m": 2000.0, "true_thickness_m": 9.0},
    )
    from hexworld.minerals.query import query_column_bodies

    hits = query_column_bodies([a, b], 500.0, 0.0, surface_m=0.0)
    assert [h["body_id"] for h in hits] == ["A", "B"]
    assert hits[1]["depth_top_m"] == pytest.approx(900.0 - 4.5)


def test_t30_t70_n21_subhex_vein_and_exact_hex_split():
    lay = HexLayout(20, 20, 10.0, 0.0, 0.0)
    v = B.make_body(
        "vein",
        "V9",
        "S9",
        (52345.0, 61234.0, 0.0),
        strike_deg=73.0,
        dip_deg=70.0,
        params={"length_m": 25000.0, "down_dip_m": 800.0, "true_thickness_m": 0.4},
    )
    split = v.hex_split(lay, n=48)
    tot = v.volume(n0=48, tol=1)["volume_m3"]
    assert sum(split.values()) == pytest.approx(tot, rel=1e-12)  # parts = whole
    assert len(split) >= 2  # a 25 km vein crosses hexes; none of them needs its centre hit
    c0, r0 = lay.world_to_offset(52.345, 61.234)
    assert (int(c0), int(r0)) in split
    # N21: a 10 000 t body split 0.25 / 0.75 between two hexes
    t = FIX["N21"]["inputs"]
    parts = [t["body_t"] * f for f in t["clipped_volume_fraction"]]
    assert parts == FIX["N21"]["expected"]


def test_t35_t36_grade_field_deterministic_and_mass_weighted():
    e = B.make_body("lens", "E7", "S7", (0.0, 0.0, -300.0), strike_deg=10.0, dip_deg=20.0,
                    params={"semi_axes_m": [200.0, 80.0, 25.0]})  # fmt: skip
    gf = B.solve_grade_field(e, target=0.01, sigma_log=0.6, rho_kg_m3=2800.0, n=10)
    assert gf["status"] == "ok" and gf["achieved"] == pytest.approx(0.01, rel=1e-6)
    pts = e.quadrature(6)[0][:50]
    g1 = B.grade_at(e, gf, pts)
    g2 = np.concatenate([B.grade_at(e, gf, pts[:17]), B.grade_at(e, gf, pts[17:])])
    assert np.array_equal(g1, g2)  # same xyz -> same grade, whatever the query chunks
    assert B.grade_at(e, gf, [[1e7, 0.0, 0.0]])[0] == 0.0  # outside: no grade
    # anisotropy: correlation length along strike >> across (structure-controlled)
    f = B.CorrelatedField("x", (1000.0, 1000.0, 5.0))
    along = np.column_stack([np.linspace(0, 50, 20), np.zeros(20), np.zeros(20)])
    across = np.column_stack([np.zeros(20), np.zeros(20), np.linspace(0, 50, 20)])
    assert np.std(f(along)) < np.std(f(across))


def test_capped_target_is_reported_not_hidden():
    e = B.make_body("lens", "E8", "S8", (0.0, 0.0, -300.0), params={"semi_axes_m": [50, 30, 10]})
    gf = B.solve_grade_field(e, target=0.9, sigma_log=1.0, rho_kg_m3=2800.0, cap=0.5)
    assert gf["status"] == "capped" and gf["achieved"] < 0.9


def test_t37_host_masking_clips_the_body():
    full = B.make_body("replacement", "R1", "S1", (0.0, 0.0, -500.0),
                       params={"semi_axes_m": [100.0, 100.0, 100.0]})  # fmt: skip
    clipped = B.make_body("replacement", "R2", "S1", (0.0, 0.0, -500.0),
                          params={"semi_axes_m": [100.0, 100.0, 100.0],
                                  "host_clip": (-500.0, -650.0)})  # fmt: skip
    vf = full.volume(tol=1e-3)["volume_m3"]
    vc = clipped.volume(tol=1e-3)["volume_m3"]
    assert vc == pytest.approx(vf / 2.0, rel=0.02)  # the lower half lies in the host
    assert not clipped.inside([[0.0, 0.0, -450.0]])[0] and full.inside([[0.0, 0.0, -450.0]])[0]


def test_t26_fault_moves_only_older_bodies():
    f = B.Fault(event_index=5, origin_m=(0.0, 0.0, 0.0), strike_deg=0.0, dip_deg=60.0,
                slip_m=(0.0, 0.0, -300.0))  # fmt: skip
    old = vein(dip=10.0)
    old.formation_event = 2
    young = vein(dip=10.0)
    young.formation_event = 7
    hw = f.hanging_wall(np.asarray(old.origin_m)[None])[0]
    moved = f.apply(old)
    assert (moved.origin_m[2] == -300.0) == bool(hw)
    assert moved.body_id == old.body_id and moved.geometry_revision == old.geometry_revision + hw
    assert f.apply(young) is young


def test_t27_erosion_clips_and_reports_the_volume():
    p = B.make_body(
        "tabular",
        "T5",
        "S5",
        (0.0, 0.0, 10.0),
        dip_deg=0.0,
        params={"length_m": 100.0, "down_dip_m": 100.0, "true_thickness_m": 20.0},
    )
    v0 = p.volume(tol=1)["volume_m3"]
    gone = p.erode_above(lambda x, y: np.full(np.shape(x), 5.0), n=12)
    assert gone == pytest.approx(v0 * (15.0 / 20.0), rel=0.15)
    assert p.cut_z_m == 5.0 and p.geometry_revision == 1
    col = p.column(0.0, 50.0)
    assert col and col[0][0] == pytest.approx(5.0)


def test_t34_mass_conditioned_geometry_is_consistent():
    from hexworld.minerals.morphology import geometry_for_mass

    for morph in ("vein", "lens", "tabular", "mantle_pipe", "stockwork"):
        g = geometry_for_mass(morph, rock_t=2.0e6, rho_kg_m3=2700.0,
                              ratios={"aspect": (1.0, 0.5, 0.1)})  # fmt: skip
        body = B.make_body(morph, "M", "S", (0.0, 0.0, -100.0), params=g)
        v = body.analytic_volume() or body.volume(tol=1e-3)["volume_m3"]
        assert v * 2.7 == pytest.approx(2.0e6, rel=0.01), morph
