"""v5.3 G7: volcanic deposits change the surface, the drainage is rebuilt (T73-T79, T08)."""

from __future__ import annotations

import numpy as np
import pytest

from hexworld.geometry import HexLayout
from hexworld.stages.hydrology import hex_graph, hydrology
from hexworld.volcanism.terrain import (
    BULK_DENSITY,
    apply_deltas,
    delta_from_volumes,
    hex_ocean,
)

AREA = 1.0e6  # m^2 per hex in the synthetic tests


def _valley(w=30, h=20):
    """A valley sloping to the east that drains into a sea strip at the east border."""
    lay = HexLayout(w, h, 1.0, 0.0, 0.0)
    g = hex_graph(lay)
    x, y = lay.centers()
    z = 200.0 - 6.0 * x + 3.0 * np.abs(y - y.mean())  # east-sloping trough
    z = z.ravel()
    sea = x.ravel() > x.max() - 1.5
    z[sea] = -10.0
    return lay, g, z, sea


def _hyd(z, sea, g):
    n = z.size
    return hydrology(z, sea, g, AREA, np.full(n, 400.0), np.full(n, 800.0), np.full(n, 600.0))


def test_n_deposit_mass():
    d = delta_from_volumes("E-x", "lava", [0], [1.0e6], 1.0e6)
    assert d.dz_m[0] == pytest.approx(1.0)  # 1 km^2 x 1 m
    assert d.mass_kg == pytest.approx(1.0e6 * BULK_DENSITY["lava"])


def test_t73_deposit_raises_the_surface_and_volume_closes():
    z = np.zeros(10)
    d1 = delta_from_volumes("E-1", "lava", [2, 3], [2e6, 1e6], AREA)
    d2 = delta_from_volumes("E-1", "tephra", [3, 4], [5e5, 5e5], AREA)
    z2, rep = apply_deltas(z, [d1, d2], AREA)
    assert rep["committed"]
    assert z2[2] == pytest.approx(2.0) and z2[3] == pytest.approx(1.5)
    assert rep["placed_m3"] == pytest.approx(rep["volume_m3"], rel=1e-12)
    assert (z == 0).all()  # the input is never modified


def test_t79_same_event_cannot_deposit_twice():
    z = np.zeros(5)
    d = delta_from_volumes("E-dup", "lava", [1], [1e6], AREA)
    z1, rep1 = apply_deltas(z, [d, d], AREA)
    assert z1[1] == pytest.approx(1.0)
    assert rep1["rejected"] and rep1["rejected"][0]["reason"] == "already applied"
    z2, rep2 = apply_deltas(z1, [d], AREA, applied={("E-dup", "lava")})
    assert np.array_equal(z2, z1) and rep2["rejected"]


def test_t08_invalid_delta_rolls_everything_back():
    z = np.arange(5.0)
    good = delta_from_volumes("E-a", "lava", [1], [1e6], AREA)
    bad = delta_from_volumes("E-b", "lava", [99], [1e6], AREA)  # outside the grid
    z2, rep = apply_deltas(z, [good, bad], AREA)
    assert not rep["committed"]
    assert np.array_equal(z2, z)  # nothing of the good delta either


def test_t74_lava_dam_creates_an_inland_lake():
    lay, g, z, sea = _valley()
    pre = _hyd(z, sea, g)
    x, y = lay.centers()
    # a lava dam across the trough at x ~ 15: 60 m high over a 2-hex band
    band = np.abs(x.ravel() - 15.0) < 1.0
    d = delta_from_volumes("E-dam", "lava", np.nonzero(band)[0], np.full(band.sum(), 60e6),
                           AREA)  # fmt: skip
    z_post, rep = apply_deltas(z, [d], AREA)
    post = _hyd(z_post, sea, g)
    new_lake = (post["lake_id"] >= 0) & (pre["lake_id"] < 0)
    assert rep["committed"] and new_lake.any()
    # the lake lies upstream (west) of the dam
    assert x.ravel()[new_lake].mean() < 15.0
    # T78: the drainage downstream of the dam changed (the river now crosses or spills over)
    assert (pre["receiver"] != post["receiver"]).any()


def test_t75_lava_entering_the_sea_makes_new_land():
    lay, g, z, sea = _valley()
    ocean_pre = hex_ocean(z, 0.0, g, keep_ocean=sea)
    idx = np.nonzero(sea)[0][:3]
    d = delta_from_volumes("E-isl", "lava", idx, np.full(idx.size, 30e6), AREA)
    z_post, _ = apply_deltas(z, [d], AREA)
    ocean_post = hex_ocean(z_post, 0.0, g, keep_ocean=sea)
    assert (ocean_pre & ~ocean_post).sum() == idx.size  # the shoreline moved


def test_closed_depression_below_sea_level_is_not_ocean():
    lay, g, z, sea = _valley()
    z2 = z.copy()
    hole = np.argmin(np.abs(z2 - 100.0))
    z2[hole] = -50.0  # an inland depression below sea level, not connected
    ocean = hex_ocean(z2, 0.0, g, keep_ocean=sea)
    assert not ocean[hole]


def test_t77_subglacial_uses_the_ice_sheet_when_present(hist_project):
    v = hist_project.volcanism
    src = v.summary.get("subglacial_ice_source")
    if hist_project.earth is not None and "land_ice_thickness_m" in hist_project.earth.layers:
        assert src == "v53_ice_sheet"
    else:
        assert src == "proxy_glacial_potential_approximate"


def test_t73_t78_stage_applies_and_rebuilds(hist_project):
    v = hist_project.volcanism
    t = v.summary["terrain"]
    if not t.get("committed"):
        pytest.skip("no deposits in this world")
    assert abs(t["volume_residual_m3"]) <= 1e-6 * max(t["volume_m3"], 1.0)
    L = v.layers
    dz = L["volcanic_dz_m"]
    assert (dz >= 0).all() and dz.max() > 0
    base = hist_project.hydro.layers["elevation_m"]
    assert np.allclose(L["post_volcanic_elevation_m"], base + dz, atol=1e-2)
    assert t["rebuild"].startswith("whole_window")
    assert "hydrology_network" in t["invalidated"]
    assert v.summary["quality"]["terrain_feedback"] == "applied_snapshot_full_drainage_rebuild"


def test_regeneration_does_not_deposit_twice(hist_config, hist_project):
    from hexworld import generate

    again = generate(hist_config, reuse=hist_project)
    a = hist_project.volcanism.layers.get("volcanic_dz_m")
    b = again.volcanism.layers.get("volcanic_dz_m")
    if a is None:
        pytest.skip("no deposits")
    assert np.array_equal(a, b)
    # the v4 surface itself is never changed by the volcanism
    assert np.array_equal(again.hydro.layers["elevation_m"],
                          hist_project.hydro.layers["elevation_m"])  # fmt: skip
