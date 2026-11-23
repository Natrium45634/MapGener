"""TZ 5.4.1 G2-G3: ocean numerics N03-N10 and physical tests (v5_4_1::T24-T43)."""

from __future__ import annotations

import math
from types import SimpleNamespace

import numpy as np
import pytest

from hexworld.earth import ebm
from hexworld.earth import ocean as oc
from hexworld.earth.planet import EARTH, Planet


def test_n03_t24_t25_slab_response_and_coriolis_sign():
    u, v = oc.slab_current(0.1, 0.0, 1e-4, r=1e-5, h=50.0, rho=1025.0)
    assert u == pytest.approx(0.00193190, rel=1e-5)
    assert v == pytest.approx(-0.0193190, rel=1e-5)  # to the right in the north
    u_s, v_s = oc.slab_current(0.1, 0.0, -1e-4, r=1e-5, h=50.0, rho=1025.0)
    assert u_s == pytest.approx(u) and v_s == pytest.approx(-v)  # left in the south
    u0, v0 = oc.slab_current(0.1, 0.0, 0.0, r=1e-5, h=50.0, rho=1025.0)
    assert u0 == pytest.approx(0.195122, rel=1e-5) and v0 == 0.0  # finite at f = 0


def test_n04_n06_exchange_conserves_heat():
    a, b = oc.exchange(20.0, 10.0, 1000.0, 1000.0, 10.0)
    assert (a, b) == (pytest.approx(19.9), pytest.approx(10.1))
    up, deep = oc.exchange(20.0, 5.0, 100.0, 100.0, 10.0)
    assert (up, deep) == (pytest.approx(18.5), pytest.approx(6.5))
    assert 100 * up + 100 * deep == pytest.approx(100 * 20 + 100 * 5)


def test_n05_mixed_layer_heat_units():
    assert oc.mixed_layer_warming_k(100.0, 50.0, 86400.0, 1025.0, 3990.0) == pytest.approx(
        0.0422519714, rel=1e-9
    )
    assert oc.mixed_layer_warming_k(100.0, 50.0, 30 * 86400.0, 1025.0, 3990.0) == pytest.approx(
        1.26755914, rel=1e-8
    )


def test_n09_freeze_and_melt_are_symmetric():
    e = -100.0 * 86400.0
    h = oc.freeze_volume_m(e, 917.0, 334000.0)
    assert h == pytest.approx(8.64e6 / (917.0 * 334000.0), rel=1e-12)
    assert h == pytest.approx(0.0282096657, rel=1e-8)  # the rounded value of the matrix
    assert h * 917.0 * 334000.0 == pytest.approx(-e)  # the same energy melts it


def test_n10_t40_effective_albedo():
    a = oc.effective_albedo(0.5, 0.06, 0.6)
    assert a == pytest.approx(0.33)
    assert (1 - a) * 200.0 == pytest.approx(134.0)
    assert (1 - oc.effective_albedo(0.0, 0.06, 0.6)) * 200.0 == pytest.approx(188.0)


# -- the planetary energy balance replaces the Earth SST table (T35) ------------------------
def test_t35_ebm_is_planet_physics_not_a_table():
    e = ebm.for_planet(EARTH)
    assert e.converged
    lat = np.array([0.0, 30.0, 45.0, 60.0])
    ann = e.annual(lat)
    assert ann[0] > ann[1] > ann[2] > ann[3]
    half = 0.5 * (np.max([e.at(lat, m) for m in range(12)], 0)
                  - np.min([e.at(lat, m) for m in range(12)], 0))  # fmt: skip
    assert half[0] < 1.0 and 2.0 < half[2] < 6.0  # weak equatorial, clear midlatitude cycle
    flat = ebm.solve(Planet(obliquity_deg=0.0))
    half0 = 0.5 * (np.max([flat.at(lat, m) for m in range(12)], 0)
                   - np.min([flat.at(lat, m) for m in range(12)], 0))  # fmt: skip
    assert half0.max() < 0.05  # no tilt, no seasons: nothing hard-coded
    bright = ebm.solve(Planet(solar_constant_wm2=1450.0))
    assert bright.annual(lat).mean() > ann.mean() + 3.0


def _basin(ny=24, nx=30, lat0=50.0, lat1=10.0):
    lat = np.repeat(np.linspace(lat0, lat1, ny)[:, None], nx, axis=1)
    sea = np.zeros((ny, nx), dtype=bool)
    sea[2:-2, 2:-2] = True
    params = SimpleNamespace(mixed_layer_m=40.0, wind_drag_cd=1.3e-3, eddy_diffusivity_m2_s=2e4,
                             sea_ice=True)  # fmt: skip
    return lat, sea, params


def _winds(lat, scale=1.0):
    u = -6.0 * np.cos(np.radians(lat) * 6.0) * scale  # trades / westerlies by latitude
    return [(u, np.zeros_like(u))] * 4


def test_t33_vertical_exchange_has_a_closed_reservoir():
    lat, sea, p = _basin()
    r = oc.run_ocean(EARTH, lat, sea, _winds(lat), 150.0, p, years=4)
    d = r.diagnostics
    assert d["heat_relative_residual"] < 1e-8
    assert d["zonal_calibration_max_k"] == 0.0  # no hidden zonal forcing
    deep = np.asarray(d["deep_reservoir_c"])
    sst = np.nanmean(r.sst_monthly[:, r.ocean])
    # the reservoir holds water that went down: not an arbitrary cold constant
    assert np.nanmin(r.sst_monthly[:, r.ocean]) - 1 <= deep.min() <= sst + 5


def test_t43_spin_up_forgets_the_start():
    lat, sea, p = _basin()
    winds = _winds(lat)
    cold = oc.run_ocean(EARTH, lat, sea, winds, 150.0, p, years=6)
    warm_start = oc.run_ocean(EARTH, lat, sea, winds, 150.0, p, years=1)
    warm_start.sst_monthly[:] = warm_start.sst_monthly + 5.0
    warm = oc.run_ocean(EARTH, lat, sea, winds, 150.0, p, years=5, prev=warm_start)
    diff = np.abs(cold.sst_monthly[:, cold.ocean] - warm.sst_monthly[:, warm.ocean])
    assert float(np.sqrt(np.mean(diff**2))) < 0.2  # the same annual cycle (start forgotten)


def test_t30_uniform_sst_without_forcing_stays_uniform():
    lat, sea, p = _basin(lat0=20.0, lat1=20.0)
    winds = _winds(lat, scale=0.0)
    r = oc.run_ocean(EARTH, lat, sea, winds, 150.0, p, years=2)
    t = r.sst_monthly[:, r.ocean]
    assert float(np.nanmax(t, axis=1).max() - np.nanmin(t, axis=1).min()) < 0.6 + 2 * float(
        np.ptp(np.nanmean(t, axis=1))
    )  # only the common seasonal cycle, no spatial pattern without currents
    assert math.isfinite(float(np.nanmean(t)))
