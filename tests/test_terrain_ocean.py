"""Volcano footprints, single-hex specks and sea-floor provinces (HexWorld 4.5 fixes)."""

from __future__ import annotations

import numpy as np

from hexworld import views
from hexworld.geometry import HexLayout
from hexworld.pipeline_v3 import SPECK_THRESHOLD_M, remove_specks
from hexworld.stages import ocean
from hexworld.stages import raster as raster_stage
from hexworld.stages.volcanism import footprint_cones


def test_cones_keep_their_volume_over_the_hex_footprint():
    h = np.array([2000.0, 4000.0])
    r = np.array([20.0, 90.0])
    coarse_h, coarse_r = footprint_cones(h, r, 75.0)
    assert np.allclose(coarse_h * coarse_r**2, h * r**2)  # volume ~ h r^2 is conserved
    assert coarse_h[0] < 0.25 * h[0]  # a small arc cone is no single-hex spike at 75 km
    assert coarse_h[1] > 0.7 * h[1]  # a Hawaii-size shield stays an island
    fine_h, fine_r = footprint_cones(h, r, 1.0)
    assert np.allclose(fine_h, h, rtol=1e-3) and np.allclose(fine_r, r, rtol=1e-3)


def test_specks_without_a_cause_are_removed_and_real_features_kept():
    layout = HexLayout(20, 16, 50.0, 0.0, 0.0)
    z = np.full((16, 20), -4000.0)
    z[5, 5] = 600.0  # a lone hex of crust 4.6 km above the abyssal floor
    z[10, 12] = 2500.0  # a volcano: protected
    z[2:6, 14:18] = 300.0  # a real island of several hexes
    protect = np.zeros(z.shape, dtype=bool)
    protect[10, 12] = True
    out, count = remove_specks(z.ravel(), layout, protect.ravel())
    out = out.reshape(z.shape)
    assert count == 1 and out[5, 5] == -4000.0
    assert out[10, 12] == 2500.0 and (out[2:6, 14:18] == 300.0).all()


def test_generated_surface_has_no_unexplained_single_hex_spikes(hist_project):
    L = hist_project.hist.layers
    layout = hist_project.layout
    z = L["elevation_m"].astype(np.float64)
    top = np.full(z.shape, -np.inf)
    for d in range(6):
        nb, valid = raster_stage.neighbor_index(layout, d)
        top = np.maximum(top, np.where(valid, z.ravel()[nb.ravel()].reshape(z.shape), -np.inf))
    cause = (L["mountain_system_id"] >= 0) | (L["volcano_id"] >= 0)
    assert not ((z - top > SPECK_THRESHOLD_M + 1.0) & ~cause).any()
    summit = L["volcano_summit_m"]
    has = L["volcano_id"] >= 0
    assert np.isnan(summit[~has]).all() and (summit[has] >= z[has] - 1e-3).all()


def _floor(shape=(30, 40)):
    z = np.full(shape, -4500.0)
    cont = np.zeros(shape)
    age = np.full(shape, 90.0)
    bc = np.zeros(shape, dtype=np.uint8)
    volc = np.full(shape, -1)
    sed = np.full(shape, 300.0)
    z[:, :6] = 200.0  # land
    cont[:, :8] = 1.0
    z[:, 6] = -120.0  # shelf
    z[:, 7] = -1500.0  # slope
    z[:, 8:10] = -3600.0  # rise
    z[:, 20] = -2700.0  # ridge axis
    age[:, 19:22] = 2.0
    z[:, 34] = -7200.0  # trench
    bc[:, 33] = 2
    z[15, 27] = -1200.0  # a seamount
    volc[15, 27] = 7
    return z, cont, age, bc, volc, sed


def test_sea_floor_provinces_follow_depth_age_and_tectonics():
    z, cont, age, bc, volc, sed = _floor()
    codes = ocean.provinces(z, 0.0, cont, age, bc, volc, sed, 50.0)
    name = np.array(ocean.PROVINCES)[codes]
    assert (name[:, :6] == "land").all()
    assert (name[:, 6] == "shelf").all()
    assert (name[:, 7] == "continental_slope").all()
    assert (name[:, 20] == "ridge_axis").all()
    assert (name[:, 34] == "trench").all()
    assert name[15, 27] == "seamount"
    assert (name[2:28, 25] == "abyssal_plain").all()


def test_depth_tints_step_down_to_black_trenches():
    tints = ocean.depth_tint(np.array([50.0, 600.0, 1500.0, 3500.0, 4500.0, 6500.0, 8000.0]))
    brightness = tints.astype(int).sum(axis=1)
    assert (np.diff(brightness) < 0).all()  # deeper is always darker
    assert brightness[-1] < 30 and brightness[-2] < 100  # trenches: near black
    # the shared height palette uses the same steps under the sea
    assert (views.hypsometric(np.array([-6500.0])) == ocean.depth_tint(np.array([6500.0]))).all()


def test_ocean_view_and_hex_info(hist_project):
    img = views.render_view_image(hist_project, "ocean", "provinces", px_per_hex=2.0)
    assert img.ndim == 3 and img.shape[2] == 3
    for mode in ("bathymetry", "age", "sediment"):
        views.view_rgb(hist_project, "ocean", mode, frozenset({"coast", "isobaths"}))
    legend = views.VIEWS["ocean"].legend(hist_project, "provinces")
    assert any("абиссальная" in item.label for item in legend)
    z = hist_project.layer("elevation_m")
    row, col = np.unravel_index(int(np.argmin(z)), z.shape)
    info = hist_project.hex_info(int(col), int(row))
    # v4.6 classes the floor by geology + morphology (geomorph.classify) when it exists
    from hexworld.geomorph import classify

    names = classify.PROVINCES if hist_project.terrain is not None else ocean.PROVINCES
    assert info["seafloor_province"] in names[1:]
