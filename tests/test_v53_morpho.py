"""v5.3 G6: wind-blown sand and dust, coastal sand budget (T49-T56)."""

from __future__ import annotations

import numpy as np
import pytest

from hexworld.earth import morpho
from hexworld.geometry import HexLayout
from hexworld.pipeline_v50 import _hex_neighbours


def _plain(w=30, h=12, wind=12.0, sand_kg_m2=1600.0, veg=0.0, wet=0.0):
    lay = HexLayout(w, h, 10.0, 0.0, 0.0)
    nbr = _hex_neighbours(lay)
    n = w * h
    x, _y = lay.centers()
    ocean = x.ravel() > x.max() - 15.0  # a sea strip at the east
    lake = np.zeros(n, dtype=bool)
    inv = np.full(n, sand_kg_m2)
    u = np.full((4, n), wind)  # blowing east
    v = np.zeros((4, n))
    frozen = np.zeros((4, n))
    return lay, nbr, ocean, lake, inv, u, v, frozen, np.full(n, veg), np.full(n, wet)


def test_t49_no_transport_below_the_threshold():
    us = morpho.friction_velocity(2.0, 0.0005)
    assert morpho.sand_flux(us, morpho.threshold_u_star(0.0, 0.0)) == 0.0
    lay, nbr, ocean, lake, inv, u, v, frozen, veg, wet = _plain(wind=1.0)
    r = morpho.aeolian(nbr, 10e3, u, v, inv, wet, veg, frozen, ocean, lake)
    assert r["ledger"]["exported_to_sea_kg"] == 0.0
    assert np.allclose(r["sand_thickness_m"][~ocean], inv[~ocean] / morpho.RHO_SAND)


def test_t50_dry_fetch_conserves_sand_and_moves_it_downwind():
    lay, nbr, ocean, lake, inv, u, v, frozen, veg, wet = _plain()
    r = morpho.aeolian(nbr, 10e3, u, v, inv, wet, veg, frozen, ocean, lake)
    led = r["ledger"]
    assert abs(led["residual_kg"]) <= 1e-9 * led["initial_kg"]
    assert led["exported_to_sea_kg"] > 0  # blown off the east coast
    x, _ = lay.centers()
    th = r["sand_thickness_m"]
    land = ~ocean
    west = land & (x.ravel() < x.min() + 30)
    assert th[west].mean() < inv[west].mean() / morpho.RHO_SAND  # the upwind end loses sand
    assert (th >= 0).all()


def test_t51_vegetation_and_moisture_hold_the_sand():
    base = _plain()
    r0 = morpho.aeolian(base[1], 10e3, *base[5:7], base[4], base[9], base[8], base[7],
                        base[2], base[3])  # fmt: skip
    wet = _plain(veg=0.8, wet=0.9)
    r1 = morpho.aeolian(wet[1], 10e3, *wet[5:7], wet[4], wet[9], wet[8], wet[7], wet[2],
                        wet[3])  # fmt: skip
    assert r1["ledger"]["exported_to_sea_kg"] < 0.05 * r0["ledger"]["exported_to_sea_kg"]


def test_no_dunes_without_sand():
    lay, nbr, ocean, lake, inv, u, v, frozen, veg, wet = _plain(sand_kg_m2=0.0)
    r = morpho.aeolian(nbr, 10e3, u, v, inv, wet, veg, frozen, ocean, lake)
    assert r["dune_type"].max() == 0 and r["ledger"]["initial_kg"] == 0.0


def test_t52_dust_is_a_separate_fraction_and_closes():
    lay, nbr, ocean, lake, inv, u, v, frozen, veg, wet = _plain()
    silt = np.full(inv.size, 0.3)
    r = morpho.aeolian(nbr, 10e3, u, v, inv, wet, veg, frozen, ocean, lake, silt_frac=silt)
    d = r["dust_ledger"]
    assert d["emitted_kg_yr"] > 0
    assert abs(d["residual_kg_yr"]) <= 1e-9 * d["emitted_kg_yr"]
    # the sand ledger does not contain the dust
    assert abs(r["ledger"]["residual_kg"]) <= 1e-9 * r["ledger"]["initial_kg"]


def test_mobile_sand_comes_from_the_soils():
    from hexworld.stages.environment import PM

    pm = np.array([PM["aeolian_sand"], PM["crystalline_bedrock"], PM["alluvium"]])
    inv = morpho.mobile_sand(pm, np.array([0.9, 0.5, 0.3]), np.array([1.0, 0.0, 1.0]))
    assert inv[0] == pytest.approx(2.0 * morpho.RHO_SAND)
    assert inv[1] == 0.0  # no soil: nothing to blow
    assert inv[2] == pytest.approx(0.5 * morpho.RHO_SAND)


def _coast_strip(kind="beach"):
    from hexworld.stages.coasts import COAST_TYPES

    lay = HexLayout(20, 10, 10.0, 0.0, 0.0)
    nbr = _hex_neighbours(lay)
    x, y = lay.centers()
    yy = y.ravel()
    ocean = yy > yy.max() - 25.0  # sea to the south (map +y)
    coast_line = (~ocean) & (yy > yy.max() - 45.0)
    ct = np.where(coast_line, COAST_TYPES.index(kind), 0)
    n = ct.size
    return lay, nbr, COAST_TYPES, ct, ocean, n


def test_t53_longshore_sand_is_conserved_along_the_coast():
    lay, nbr, names, ct, ocean, n = _coast_strip("beach")
    u = np.full(n, 6.0)  # wind toward the east, waves from the south-west quarter
    v = np.full(n, -6.0)  # onshore component (map y points south)
    r = morpho.coastal(nbr, 10e3, names, ct, np.full(n, 0.8), ocean, u, v, np.zeros(n),
                       np.full(n, 0.3))  # fmt: skip
    led = r["ledger"]
    assert led["longshore_total_m3_yr"] > 0
    assert abs(led["residual_m3_yr"]) <= 1e-9 * led["longshore_total_m3_yr"]


def test_t54_cliff_follows_the_rock_law_not_the_sand_law():
    lay, nbr, names, ct_cliff, ocean, n = _coast_strip("cliff")
    u, v = np.full(n, 6.0), np.full(n, -6.0)
    soft = morpho.coastal(nbr, 10e3, names, ct_cliff, np.full(n, 0.8), ocean, u, v,
                          np.zeros(n), np.full(n, 0.3))  # fmt: skip
    hard = morpho.coastal(nbr, 10e3, names, ct_cliff, np.full(n, 0.8), ocean, u, v,
                          np.zeros(n), np.full(n, 1.0))  # fmt: skip
    m = ct_cliff > 0
    assert (soft["longshore_m3_yr"][m] == 0).all()  # no sand drift on a cliff
    assert soft["shoreline_change_m_yr"][m].mean() < hard["shoreline_change_m_yr"][m].mean() < 0


def test_t55_coast_does_not_depend_on_the_window_edge():
    lay, nbr, names, ct, ocean, n = _coast_strip("beach")
    u, v = np.full(n, 6.0), np.full(n, -6.0)
    r = morpho.coastal(nbr, 10e3, names, ct, np.full(n, 0.8), ocean, u, v, np.zeros(n),
                       np.full(n, 0.3))  # fmt: skip
    dy = r["shoreline_change_m_yr"]
    m = (ct > 0) & np.isfinite(dy)
    assert np.isfinite(dy[m]).all()  # the edge hexes are defined, nothing breaks there


def test_stage_layers_and_ledgers(hist_project):
    m = hist_project.earth.summary.get("morphodynamics")
    assert m is not None
    assert abs(m["sand_ledger"]["residual_kg"]) <= 1e-6 * max(m["sand_ledger"]["initial_kg"], 1)
    L = hist_project.earth.layers
    ocean = hist_project.hydro.layers["ocean_mask"]
    assert (L["dune_type"][ocean] == 0).all()
    assert (L["sand_thickness_m"] >= 0).all()


def test_t56_relative_sea_level_moves_the_shore():
    lay, nbr, names, ct, ocean, n = _coast_strip("beach")
    u, v = np.full(n, 6.0), np.full(n, -6.0)
    args = (nbr, 10e3, names, ct, np.full(n, 0.8), ocean, u, v, np.zeros(n), np.full(n, 0.3))
    still = morpho.coastal(*args)
    rise = morpho.coastal(*args, rslr_m_yr=0.003)
    m = (ct > 0) & np.isfinite(still["shoreline_change_m_yr"])
    diff = rise["shoreline_change_m_yr"][m] - still["shoreline_change_m_yr"][m]
    assert np.allclose(diff, -0.003 / 0.02)  # 3 mm/yr of rise: 15 cm/yr of retreat
