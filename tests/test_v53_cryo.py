"""v5.3b: grounded ice sheets, shelves, isostasy, sea level (TZ 5.3 sec. 8-10; T33-T48)."""

from __future__ import annotations

import numpy as np
import pytest

from hexworld.earth import icesheet as ice
from hexworld.earth import shelves
from hexworld.geometry import HexLayout
from hexworld.pipeline_v50 import _hex_neighbours


def _island(radius_km=600.0, n=40, spacing=40.0, smb=0.3):
    lay = HexLayout(n, n, spacing, 0.0, 0.0)
    nbr = _hex_neighbours(lay)
    x, y = lay.centers()
    r = np.hypot(x - x.mean(), y - y.mean()).ravel()
    land = r < radius_km
    ocean = ~land
    bed = np.where(land, 0.0, -1000.0)
    smb_a = np.where(land, smb, 0.0)
    return lay, nbr, land, ocean, bed, smb_a


# -- numeric controls ------------------------------------------------------------------------
def test_n_ice_mass_and_sea_level():
    v = 100e6 * 1000.0  # 100 km^2 x 1000 m
    assert v * 917.0 / 1e12 == pytest.approx(91.7)  # Gt
    # 360 km^3 of water on 360 million km^2 of ocean: +1 mm
    sle = ice.sea_level_equivalent_m(360e9 * 1000.0 / 917.0, 360e6 * 1e6)
    assert sle == pytest.approx(0.001, rel=1e-9)


def test_t41_flotation_draft_and_freeboard():
    d, f = shelves.draft_freeboard(500.0)
    assert d == pytest.approx(446.0, abs=0.1) and f == pytest.approx(54.0, abs=0.1)
    assert shelves.floats(500.0, 460.0) and not shelves.floats(500.0, 400.0)


# -- T33-T34 seed --------------------------------------------------------------------------------
def test_t33_vialov_profile_shape():
    r_m = 500e3
    h0 = ice.vialov_h0(r_m, 0.2)
    d = np.linspace(0.0, r_m, 101)  # distance from the margin
    h = ice.vialov_profile(d, r_m, h0)
    assert h[0] == 0.0 and h[-1] == pytest.approx(h0)
    assert (np.diff(h) >= -1e-9).all()  # rises monotonically to the centre
    assert 1000 < h0 < 4000


def test_t34_no_seed_without_accumulation():
    lay, nbr, land, ocean, bed, smb = _island(smb=-0.2)
    h = ice.seed("analytic_steady_seed", smb, np.zeros(smb.size), land, nbr, 40e3)
    assert h.max() == 0.0
    r = ice.spinup(np.full((12, smb.size), 15.0), np.full((12, smb.size), 10.0), bed, land,
                   ocean, nbr, 40e3, max_years=2000.0)  # fmt: skip
    assert r.thickness_m.max() == 0.0  # warm and dry: no sheet (T38 too)


# -- T35-T37 flow and spin-up --------------------------------------------------------------------
def test_t35_flow_conserves_volume_in_a_closed_case():
    lay, nbr, land, ocean, bed, smb = _island()
    no_sea = np.zeros_like(ocean)
    sia = ice.SIA(nbr, 40e3, np.zeros_like(bed), no_sea)
    x, y = lay.centers()
    h = np.clip(2000.0 - np.hypot(x - x.mean(), y - y.mean()).ravel() * 3.0, 0.0, None)
    v0 = h.sum()
    for _ in range(20):
        h, led = sia.implicit_step(h, np.zeros_like(h), 200.0)
        assert abs(led["calving_m3"]) < 1e-6 * v0 * sia.area
    assert h.sum() == pytest.approx(v0, rel=1e-9)
    assert h.min() >= 0.0


def test_sia_steady_state_matches_vialov():
    lay, nbr, land, ocean, bed, smb = _island(smb=0.3)
    sia = ice.SIA(nbr, 40e3, bed, ocean)
    h = np.zeros(smb.size)
    for _ in range(80):
        h, led = sia.implicit_step(h, smb, 250.0)
    assert h.max() == pytest.approx(ice.vialov_h0(600e3, 0.3), rel=0.1)
    assert led["calving_m3"] == pytest.approx(led["smb_m3"], rel=0.05)  # in balance


def test_t36_cold_start_short_run_is_not_called_equilibrium():
    lay, nbr, land, ocean, bed, smb = _island()
    n = smb.size
    t = np.full((12, n), -25.0)
    p = np.full((12, n), 30.0)  # ~0.4 m ice / yr of snow
    r = ice.spinup(t, p, bed, land, ocean, nbr, 40e3, mode="cold_start", min_years=100.0,
                   max_years=100.0, window=50.0)  # fmt: skip
    assert r.status != "CONVERGED"


def test_t37_convergence_needs_mass_and_shape_over_windows():
    lay, nbr, land, ocean, bed, smb = _island(radius_km=300.0, n=24)
    n = smb.size
    t = np.full((12, n), -25.0)
    p = np.full((12, n), 20.0)
    r = ice.spinup(t, p, bed, land, ocean, nbr, 40e3, mode="analytic_steady_seed",
                   min_years=500.0, max_years=60000.0, window=500.0, step_years=250.0)  # fmt: skip
    assert r.status == "CONVERGED"
    tail = r.history[-2:]
    assert all(x["rel_dv_per_100yr"] < 1e-4 and x["mean_dh_m_yr"] < 0.01 for x in tail)
    assert r.ledger["relative_residual"] < 1e-9


def test_ledger_closes_with_calving():
    lay, nbr, land, ocean, bed, smb = _island()
    n = smb.size
    r = ice.spinup(np.full((12, n), -20.0), np.full((12, n), 25.0), bed, land, ocean, nbr,
                   40e3, max_years=3000.0)  # fmt: skip
    led = r.ledger
    assert led["relative_residual"] < 1e-9
    assert led["calving_m3"] > 0


def test_t38_dry_polar_land_stays_bare_with_negative_balance():
    lay, nbr, land, ocean, bed, smb = _island()
    n = smb.size
    # -15 degC and 2 mm/month of snow: sublimation exceeds the snowfall
    r = ice.spinup(np.full((12, n), -15.0), np.full((12, n), 2.0), bed, land, ocean, nbr,
                   40e3, max_years=2000.0, sublimation_mm=60.0)  # fmt: skip
    assert r.thickness_m.max() == 0.0


def test_old_ice_under_negative_smb_thins_but_does_not_vanish_at_once():
    lay, nbr, land, ocean, bed, smb = _island()
    sia = ice.SIA(nbr, 40e3, bed, ocean)
    h = np.where(land, 1000.0, 0.0)
    neg = np.where(land, -0.15, 0.0)
    h1, _ = sia.implicit_step(h, neg, 10.0)
    inner = land & (h1 > 0)
    assert 0 < h1[inner].mean() < 1000.0


def test_t40_strong_ablation_flag_is_not_the_negative_smb_zone():
    smb = np.array([-0.5, -0.1, 0.2])  # m ice / yr
    negative = smb < 0
    strong = smb * 917.0 < -300.0  # the legacy -300 mm w.e. threshold
    assert negative.sum() == 2 and strong.sum() == 1


# -- T42-T45 shelves ---------------------------------------------------------------------------
def test_t42_cold_ocean_alone_makes_no_shelf():
    lay, nbr, land, ocean, bed, smb = _island()
    h = np.zeros(smb.size)  # no grounded ice
    r = shelves.build_shelves(h, bed, np.where(ocean, 1000.0, 0.0), ocean,
                              np.full(smb.size, -1.8), nbr, 40e3)  # fmt: skip
    assert r.thickness_m.max() == 0.0


def test_t43_t44_shelf_flux_is_conserved_and_calving_booked():
    lay, nbr, land, ocean, bed, smb = _island()
    sia = ice.SIA(nbr, 40e3, bed, ocean)
    h = np.zeros(smb.size)
    for _ in range(40):
        h, led = sia.implicit_step(h, smb, 250.0)
    inflow = sia.last_calving_hex / 250.0
    r = shelves.build_shelves(h, bed, np.where(ocean, 1000.0, 0.0), ocean,
                              np.full(smb.size, -1.8), nbr, 40e3, inflow_hex=inflow)  # fmt: skip
    led = r.ledger
    assert led["grounding_flux_m3_yr"] == pytest.approx(inflow.sum(), rel=1e-9)
    assert abs(led["residual_m3_yr"]) < 1e-6 * led["grounding_flux_m3_yr"]
    assert r.thickness_m[land].max() == 0.0 and r.thickness_m.max() > 0
    assert r.grounding_line[land].any()


def test_t45_basal_melt_uses_ocean_heat():
    lay, nbr, land, ocean, bed, smb = _island()
    sia = ice.SIA(nbr, 40e3, bed, ocean)
    h = np.zeros(smb.size)
    for _ in range(40):
        h, _ = sia.implicit_step(h, smb, 250.0)
    inflow = sia.last_calving_hex / 250.0
    depth = np.where(ocean, 1000.0, 0.0)
    cold = shelves.build_shelves(h, bed, depth, ocean, np.full(smb.size, -1.8), nbr, 40e3,
                                 inflow_hex=inflow).ledger  # fmt: skip
    mild = shelves.build_shelves(h, bed, depth, ocean, np.full(smb.size, -0.5), nbr, 40e3,
                                 inflow_hex=inflow).ledger  # fmt: skip
    assert mild["basal_melt_m3_yr"] > cold["basal_melt_m3_yr"] > 0
    assert mild["ocean_heat_used_w"] > cold["ocean_heat_used_w"]


# -- T46-T48 feedbacks --------------------------------------------------------------------------
def test_t46_no_erosion_on_a_frozen_bed():
    e = ice.glacial_erosion_rate(np.array([50.0, 50.0]), np.array([800.0, 800.0]),
                                 np.array([-25.0, -2.0]))  # fmt: skip
    assert e[0] == 0.0 and e[1] > 0.0


def test_t47_isostasy_responds_with_delay():
    h = np.array([0.0, 1000.0, 3000.0])
    w_short = ice.isostatic_deflection(h, 100.0)
    w_long = ice.isostatic_deflection(h, 1e6)
    assert w_long[2] == pytest.approx(-3000.0 * 917.0 / 3300.0, rel=1e-6)
    assert w_long[2] < w_short[2] < 0.0  # deeper with time
    assert w_long[0] == 0.0


def test_t48_sea_level_from_grounded_ice_not_from_sea_ice():
    # floating ice displaces its own weight: only the grounded ice above flotation counts
    assert ice.sea_level_equivalent_m(0.0, 3.6e14) == 0.0
    assert ice.sea_level_equivalent_m(1e15, 3.6e14) > 0.0


# -- the stage on a generated world --------------------------------------------------------------
def test_stage_reports_the_ice_state(hist_project):
    e = hist_project.earth
    ice_s = e.summary.get("ice")
    assert ice_s is not None and ice_s["status"] in (
        "CONVERGED", "NOT_CONVERGED", "TRANSIENT_RETREAT")  # fmt: skip
    assert ice_s["ledger"]["relative_residual"] < 1e-9
    assert ice_s["applied_to_relief"] is False
    L = e.layers
    assert (L["land_ice_thickness_m"] >= 0).all()
    ocean = hist_project.hydro.layers["ocean_mask"]
    assert (L["land_ice_thickness_m"][ocean] == 0).all()
    assert (L["shelf_thickness_m"][~ocean] == 0).all()


def test_t24_ice_kinds_are_separate_covers(hist_project):
    L = hist_project.earth.layers
    cover = L["ice_cover"]
    ocean = hist_project.hydro.layers["ocean_mask"]
    assert not ((cover == 5) & ocean).any()  # grounded ice on land only
    assert not ((cover == 6) & ~ocean).any()  # shelves on the sea only


def test_v50_uses_the_sheet_as_its_initial_state(hist_project):
    b = hist_project.biomes
    assert b.summary["cryosphere"].get("seed_source") in ("v53_ice_sheet", "v50_paleo_seed",
                                                          "none")  # fmt: skip
    if (hist_project.earth.layers["land_ice_thickness_m"] > 0).any():
        assert b.summary["cryosphere"]["seed_source"] == "v53_ice_sheet"
