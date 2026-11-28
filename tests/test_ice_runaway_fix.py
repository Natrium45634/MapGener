"""Regression of the 1000×800 user world (HexWorld 1.10.x): runaway ice sheets reached +6 °C
deserts, and the polar sea ice piled up to tens or hundreds of metres."""

from __future__ import annotations

import math

import numpy as np
import pytest

from hexworld.earth import icesheet as ic
from hexworld.earth import ocean as oc


def test_snowfall_falls_with_the_raised_ice_surface():
    t = np.full((12, 1), -20.0)
    p = np.full((12, 1), 30.0)
    bare = ic.surface_mass_balance(t, p, 0.0)[0]
    thick = ic.surface_mass_balance(t, p, 3000.0)[0]
    rise = 3000.0 * ic.SURFACE_RISE_SHARE
    assert thick == pytest.approx(bare * math.exp(-ic.PRECIP_CC * ic.LAPSE * rise), rel=1e-5)
    assert 0 < thick < 0.5 * bare  # a 3 km sheet no longer keeps the plain's snowfall


def test_sheet_growth_is_bounded_on_a_cold_snowy_plateau():
    # a 2000 m thick sheet over a cold snowy plain: the extra thickness now cuts the snowfall,
    # so the balance at 6 km is far below the balance at the bare surface
    t = np.full((12, 1), -15.0)
    p = np.full((12, 1), 40.0)
    b0 = ic.surface_mass_balance(t, p, 0.0)[0]
    b6 = ic.surface_mass_balance(t, p, 6000.0)[0]
    assert b6 < 0.2 * b0


def test_ridging_keeps_volume_and_spreads_thick_ice():
    nb = np.array([[1, -1, -1, -1], [0, 2, -1, -1], [1, 3, -1, -1], [2, -1, -1, -1]])
    h = np.array([20.0, 1.0, 0.0, 0.0])
    r = oc.ridge(h, nb)
    assert r.sum() == pytest.approx(h.sum(), rel=1e-12)
    assert r[0] < h[0] and r[1] > h[1]
