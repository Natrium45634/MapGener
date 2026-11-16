"""Coast types: every type comes from its cause, and the drivers are consistent."""

from __future__ import annotations

import numpy as np

from hexworld import views
from hexworld.geometry import HexLayout
from hexworld.stages.coasts import COAST_TYPES, coast_fields

CODE = {name: i for i, name in enumerate(COAST_TYPES)}


def _world(**over):
    """A synthetic shore: sea in the west (a shelf, then deep water), land in the east."""
    layout = HexLayout(40, 24, 50.0, 0.0, 0.0)
    shape = (layout.height, layout.width)
    cols = np.arange(layout.width)[None, :].repeat(layout.height, axis=0).astype(float)
    land_h = over.pop("land_h", 15.0)
    z = np.where(cols < 14, np.where(cols < 8, -3000.0, -80.0), land_h + 5.0 * (cols - 14))
    layers = {
        "elevation_m": z,
        "lake_id": np.full(shape, -1),
        "temperature_mean_c": np.full(shape, over.pop("t", 14.0)),
        "temperature_seasonal_c": np.full(shape, over.pop("amp", 8.0)),
        "wind_u_ms": np.full(shape, 4.0),
        "wind_v_ms": np.zeros(shape),
        "latitude_deg": np.full(shape, over.pop("lat", 40.0)),
        "slope_m_per_km": np.full(shape, over.pop("slope", 1.0)),
        "sediment_thickness_m": np.full(shape, over.pop("sediment", 100.0)),
        "precipitation_mm_yr": np.full(shape, over.pop("precip", 900.0)),
        "humidity_index": np.full(shape, over.pop("humid", 1.2)),
        "glacial_potential": np.full(shape, over.pop("glacial", 0.0)),
        "sediment_delivery_m3_yr": np.zeros(shape),
        "coastal_discharge_m3_s": np.zeros(shape),
    }
    mouth = over.pop("mouth", None)
    if mouth is not None:
        q, load = mouth
        layers["coastal_discharge_m3_s"][12, 14] = q
        layers["sediment_delivery_m3_yr"][12, 14] = load
    hist = {
        "local_relief_m": np.full(shape, over.pop("relief", 40.0)),
        "rock_resistance": np.full(shape, over.pop("rock", 0.22)),
        "rock_type": np.full(shape, 5, dtype=np.uint8),
        "boundary_class": np.zeros(shape, dtype=np.uint8),
        "volcano_id": np.full(shape, -1),
    }
    assert not over, over
    return layout, layers, hist


def _types(**over):
    layout, layers, hist = _world(**over)
    out, summary = coast_fields(layout, layers, hist, [], 0.0, 0)
    ct = out["coast_type"]
    return ct[ct > 0], out, summary


def _main(**over):
    ct, _out, _s = _types(**over)
    return COAST_TYPES[np.bincount(ct).argmax()]


def test_each_cause_gives_its_coast():
    assert _main() in ("beach", "barrier_lagoon")  # a low sandy shore
    assert _main(t=2.0, amp=8.0, lat=63.0, land_h=900.0, relief=700.0, precip=1600.0) == "fjord"
    assert _main(t=1.0, amp=10.0, lat=63.0, land_h=40.0, relief=60.0, rock=0.85) == "skerry"
    assert _main(t=-20.0, lat=75.0, glacial=0.9) == "ice"
    assert _main(t=27.0, amp=2.0, lat=12.0, sediment=10.0, humid=1.5) == "coral_reef"
    assert _main(t=27.0, amp=2.0, lat=4.0, sediment=400.0, humid=2.0) == "mangrove"
    assert _main(t=26.0, lat=24.0, humid=0.08, precip=60.0, sediment=10.0) == "sabkha"
    assert _main(land_h=400.0, relief=500.0, slope=15.0, rock=0.7) == "cliff"
    assert _main(land_h=160.0, relief=250.0, slope=4.0, rock=0.6) == "ria"


def test_river_mouths_make_deltas_or_estuaries():
    big_load, _o, _s = _types(mouth=(9000.0, 6.0e7))
    small_load, _o, _s = _types(mouth=(9000.0, 2.0e5))
    assert CODE["delta"] in big_load and CODE["delta"] not in small_load
    assert {CODE["estuary"], CODE["liman"]} & set(small_load.tolist())


def test_drivers_exist_only_at_the_coast():
    ct, out, summary = _types()
    coast = out["coast_type"] > 0
    for name in ("wave_exposure", "tidal_range_m", "shelf_width_km", "sea_surface_temperature_c"):
        assert np.isfinite(out[name][coast]).all() and np.isnan(out[name][~coast]).all(), name
    assert (
        sum(summary["shares"].values()) == np.float64(1.0)
        or abs(sum(summary["shares"].values()) - 1.0) < 1e-3
    )
    assert 300.0 < np.nanmean(out["shelf_width_km"]) < 400.0  # 6 shelf columns of 50 km


def test_generated_world_coasts_follow_their_causes(hist_project):
    L = hist_project.hydro.layers
    ct = L["coast_type"]
    land = L["land_mask"]
    assert not (ct[~land] > 0).any()  # only land hexes at the sea are coasts
    coast = ct > 0
    assert coast.any() and len(np.unique(ct[coast])) >= 5
    sst = L["sea_surface_temperature_c"]
    assert (sst[ct == CODE["coral_reef"]] > 22.0).all()
    assert (sst[ct == CODE["mangrove"]] > 22.0).all()
    assert L["lgm_ice_mask"][(ct == CODE["fjord"]) | (ct == CODE["skerry"])].all()
    rivers = (ct == CODE["delta"]) | (ct == CODE["estuary"]) | (ct == CODE["liman"])
    assert (L["coastal_discharge_m3_s"][rivers] > 0).all()
    spec = views.VIEWS["coasts"]
    assert spec.availability(hist_project)[0]
    for mode in spec.modes:
        assert views.view_rgb(hist_project, "coasts", mode.id).shape == (100, 100, 3)
        assert spec.legend(hist_project, mode.id)
    assert views.view_vectors(hist_project, "coasts", "types", {"symbols"}) is not None
