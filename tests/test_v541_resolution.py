"""TZ 5.4.1 G7: v5_4::T20 on real window sizes (360x240 and 500x300), resolution and nested
windows of one physical world; metrics of TZ 5.4.1 sec. 12.2 on a common control grid.

Slow (two full generations each): ``pytest -m slow tests/test_v541_resolution.py``.
"""

from __future__ import annotations

import numpy as np
import pytest

from hexworld import Config, generate

pytestmark = pytest.mark.slow

#: one physical world for every case: the same simulation area and world raster
BOUNDS = (-3000.0, -3000.0, 21000.0, 15000.0)
SEED = "Terra7"


def _world(w, h, km, origin=(0.0, 0.0)):
    cfg = Config(seed=SEED, width=w, height=h, hex_spacing_km=km, viewport_origin_km=origin,
                 simulation_bounds_km=BOUNDS)  # fmt: skip
    return generate(cfg, through="v45")


def _control(project, xs, ys, month, level=0):
    """Wind (u, v) of the hex window sampled on world points (bilinear on the hex field
    through the atmosphere grid is avoided: the hex samples are nearest hexes)."""
    lay = project.layout
    col, row = lay.world_to_offset(xs, ys)
    col = np.clip(np.asarray(col, int), 0, lay.width - 1)
    row = np.clip(np.asarray(row, int), 0, lay.height - 1)
    from hexworld.pipeline_v54 import sample

    a = project.atmosphere
    hx, hy = lay.centers()
    u = sample(a.grid, a.grid_fields["wind_u_ms"][month, level], hx, hy)[row, col]
    v = sample(a.grid, a.grid_fields["wind_v_ms"][month, level], hx, hy)[row, col]
    return u, v


def _metrics(ua, va, ub, vb):
    rmse = np.sqrt(np.mean((ua - ub) ** 2 + (va - vb) ** 2))
    ref = np.sqrt(np.mean(ub**2 + vb**2))
    nrmse = rmse / max(ref, 1.0)
    sa, sb = np.hypot(ua, va), np.hypot(ub, vb)
    both = (sa > 1.0) & (sb > 1.0)
    cosang = (ua * ub + va * vb)[both] / (sa * sb)[both]
    ang = np.degrees(np.arccos(np.clip(cosang, -1, 1)))
    a = np.concatenate([ua - ua.mean(), va - va.mean()])
    b = np.concatenate([ub - ub.mean(), vb - vb.mean()])
    corr = float(np.dot(a, b) / max(np.linalg.norm(a) * np.linalg.norm(b), 1e-12))
    sign = np.mean(np.sign(ua[np.abs(ub) > 1]) == np.sign(ub[np.abs(ub) > 1]))
    return {"nrmse": float(nrmse), "corr": corr, "median_angle": float(np.median(ang)),
            "zonal_sign_agreement": float(sign)}  # fmt: skip


@pytest.fixture(scope="module")
def pair():
    a = _world(360, 240, 50.0)  # 18000 x 12000 km
    b = _world(500, 300, 36.0)  # 18000 x 10800 km: a different aspect, the same world
    return a, b


def test_t20_a_resolution_same_physical_world(pair):
    a, b = pair
    # the physics of both runs is the same world raster (resolution-independent solver)
    assert a.hist.world["elevation_m"].shape == b.hist.world["elevation_m"].shape
    ys = np.linspace(500.0, 10000.0, 30)
    xs = np.linspace(500.0, 17500.0, 40)
    X, Y = np.meshgrid(xs, ys)
    res = []
    for m in (0, 6):
        for level in (0, 1, 3):
            ua, va = _control(a, X.ravel(), Y.ravel(), m, level)
            ub, vb = _control(b, X.ravel(), Y.ravel(), m, level)
            res.append(_metrics(ua, va, ub, vb))
    worst = {k: (max if k in ("nrmse", "median_angle") else min)(r[k] for r in res)
             for k in res[0]}  # fmt: skip
    print("T20 resolution metrics (worst over months 1/7, sfc/850/250):", worst)
    assert worst["nrmse"] <= 0.25
    assert worst["corr"] >= 0.90
    assert worst["median_angle"] <= 20.0
    assert worst["zonal_sign_agreement"] >= 0.90
    # ITCZ / rain totals of the physical world agree
    pa = a.atmosphere.grid_fields["precipitation_mm"].sum()
    pb = b.atmosphere.grid_fields["precipitation_mm"].sum()
    assert abs(pa - pb) / max(abs(pb), 1e-9) <= 0.10


def test_t20_b_nested_windows_and_viewport(pair):
    a, _b = pair
    # another window of the same world, shifted by whole hex columns / rows (20 columns of
    # 1.5 R and 20 rows of the spacing) so that the hex centres coincide in the overlap
    r = 50.0 / np.sqrt(3.0)
    shifted = _world(360, 240, 50.0, origin=(20 * 1.5 * r, 20 * 50.0))
    ga, gs = a.atmosphere.grid_fields, shifted.atmosphere.grid_fields
    # the atmosphere lives on the world grid: a different window gives the same arrays
    for k in ("slp_pa", "wind_u_ms", "precipitation_mm"):
        assert np.array_equal(ga[k], gs[k]), k
    # overlap of the two windows: the hex fields agree exactly where both exist
    xs = np.linspace(2500.0, 15000.0, 30)
    ys = np.linspace(1500.0, 11000.0, 20)
    X, Y = np.meshgrid(xs, ys)
    ua, va = _control(a, X.ravel(), Y.ravel(), 6, 0)
    us, vs = _control(shifted, X.ravel(), Y.ravel(), 6, 0)
    m = _metrics(ua, va, us, vs)
    print("T20 nested-window metrics:", m)
    # the same hex centres give the same values; points on a hex edge may pick the other
    # neighbour after the float shift of the origin (a sampling effect, not physics)
    assert m["median_angle"] == 0.0
    assert m["nrmse"] <= 0.01
