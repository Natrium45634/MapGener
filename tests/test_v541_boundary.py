"""TZ 5.4.1 G1: regional boundary profiles (v5_4_1::T12-T17, N02, S10)."""

from __future__ import annotations

import json
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from hexworld import boundary as bnd
from hexworld.config import AtmosphereParams, Config, ConfigError
from hexworld.earth.planet import EARTH


def _wet_west_dry_rest():
    profs = [bnd.from_preset("w", ("west",), "warm_ocean"),
             bnd.from_preset("e", ("east",), "dry_continental"),
             bnd.from_preset("n", ("north",), "mountain_barrier"),
             bnd.from_preset("s", ("south",), "interior")]  # fmt: skip
    return profs


def test_t13_rh_to_q():
    q = bnd.rh_to_q(0.5, 20.0, 101325.0)
    e = 0.5 * 611.2 * np.exp(17.67 * 20 / (20 + 243.5))
    assert q == pytest.approx(0.622 * e / (101325.0 - 0.378 * e), rel=1e-12)
    assert 0.0072 < q < 0.0074
    with pytest.raises(ValueError):
        bnd.rh_to_q(50.0, 20.0)  # percent instead of a fraction is rejected


def test_n02_t14_flux_sign_and_donor():
    assert bnd.boundary_vapor_flux(1000.0, 20.0, 10.0, -5.0) == pytest.approx(-100_000.0)
    assert bnd.boundary_vapor_flux(1000.0, 20.0, 10.0, 5.0) == pytest.approx(50_000.0)


def test_t15_state_and_flux_are_exclusive():
    p = bnd.from_preset("x", ("west",), "wet_westerly")
    bad = replace(p, moisture_flux_kg_m_s=(1.0,) * 12)
    assert any("mutually exclusive" in e for e in bnd.validate([bad]))
    both = replace(p, q_kgkg=(0.01,) * 12)
    assert any("either q_kgkg or rh" in e for e in bnd.validate([both]))
    dup = [p, bnd.from_preset("y", ("west",), "interior")]
    assert any("already covered" in e for e in bnd.validate(dup))


def test_t17_net_inflow_into_a_closed_basin_is_rejected():
    p = replace(bnd.from_preset("o", ("east",), "warm_ocean"), exchange_m3s=(-1e6,) * 12)
    assert any("incompatible" in e for e in bnd.validate([p]))
    with pytest.raises(ConfigError):
        Config(atmosphere=AtmosphereParams(boundary_profiles_json=bnd.to_json([p])))


def test_t12_roundtrip_gui_json_cli_save():
    text = bnd.to_json(_wet_west_dry_rest())
    cfg = Config(atmosphere=AtmosphereParams(domain_mode="custom_region",
                                             boundary_profiles_json=text))  # fmt: skip
    again = Config.from_json(cfg.to_json())
    assert again == cfg
    assert bnd.to_json(bnd.from_json(again.atmosphere.boundary_profiles_json)) == text
    assert bnd.profile_hash(text) == bnd.profile_hash(again.atmosphere.boundary_profiles_json)
    from hexworld import settings_schema as ss

    vals = ss.values_from_config(cfg) if hasattr(ss, "values_from_config") else None
    if vals is not None:
        assert ss.config_from_values(vals) == cfg


def test_custom_region_needs_every_side():
    text = bnd.to_json(_wet_west_dry_rest()[:2])
    with pytest.raises(ConfigError, match="every side"):
        Config(atmosphere=AtmosphereParams(domain_mode="custom_region",
                                           boundary_profiles_json=text))  # fmt: skip


def _synth(atm, land=None):
    from scipy import ndimage

    from hexworld.atmosphere import normals
    from hexworld.stages.climate import zonal_temperature

    ny, nx = 24, 30
    lat = np.repeat(np.linspace(55, 25, ny)[:, None], nx, axis=1)
    land = np.zeros((ny, nx), bool) if land is None else land
    sea = ~land
    dist = ndimage.distance_transform_edt(~sea) * 300.0 if sea.any() else np.full(sea.shape, 4e3)
    cont = 1 - np.exp(-dist / 900.0)
    amp = (1 + 0.32 * np.abs(lat)) * (0.33 + 0.67 * cont)
    grid = {"x0_km": 0.0, "y0_km": 0.0, "cell_km": 300.0}
    return normals.build(atm, EARTH, lat, sea, np.zeros((ny, nx)), zonal_temperature(lat), amp,
                         cont, 300.0, grid)  # fmt: skip


def test_s10_imported_moisture_changes_the_rain_and_is_booked():
    land = np.zeros((24, 30), bool)
    land[:, :] = True  # a continent filling the region: all water comes from outside
    wet = [bnd.from_preset(f"{s}", (s,), "warm_ocean") for s in bnd.SIDES]
    dry = [bnd.from_preset(f"{s}", (s,), "dry_continental") for s in bnd.SIDES]
    base = AtmosphereParams(quality="fast", domain_mode="custom_region")
    n_wet = _synth(replace(base, boundary_profiles_json=bnd.to_json(wet)), land)
    n_dry = _synth(replace(base, boundary_profiles_json=bnd.to_json(dry)), land)
    p_wet = n_wet.moist.precip_mm.sum()
    p_dry = n_dry.moist.precip_mm.sum()
    assert p_wet > 1.5 * p_dry
    for led in n_wet.moist.ledger:
        assert led["relative_residual"] < 1e-9
    assert sum(x["import_kg"] for x in n_wet.moist.ledger) > sum(
        x["import_kg"] for x in n_dry.moist.ledger)  # fmt: skip
    closed = _synth(AtmosphereParams(quality="fast", domain_mode="closed_test_basin"), land)
    assert sum(x["import_kg"] for x in closed.moist.ledger) == 0.0  # no invented moisture
    assert sum(x["export_kg"] for x in closed.moist.ledger) == 0.0


def test_prescribed_flux_is_a_booked_source():
    flux = replace(bnd.from_preset("w", ("west",), "interior"), atm_mode="prescribed_flux",
                   moisture_flux_kg_m_s=(200.0,) * 12)  # fmt: skip
    others = [bnd.from_preset(s, (s,), "mountain_barrier") for s in ("east", "north", "south")]
    atm = AtmosphereParams(quality="fast", domain_mode="custom_region",
                           boundary_profiles_json=bnd.to_json([flux, *others]))  # fmt: skip
    n = _synth(atm, np.ones((24, 30), bool))
    imp = sum(x["import_kg"] for x in n.moist.ledger)
    days = n.circ.months_days
    expect = 200.0 * 24 * 300e3 * float(days.sum()) * 86400.0
    assert imp == pytest.approx(expect, rel=1e-6)


def test_t16_ocean_exchange_warms_only_sea_cells_and_closes():
    from hexworld.earth import ocean as oc

    ny, nx = 20, 26
    lat = np.repeat(np.linspace(55, 35, ny)[:, None], nx, axis=1)
    sea = np.zeros((ny, nx), bool)
    sea[3:-3, 6:] = True  # open to the east edge; land on the west
    params = SimpleNamespace(mixed_layer_m=40.0, wind_drag_cd=1.3e-3, eddy_diffusivity_m2_s=2e4,
                             sea_ice=True)  # fmt: skip
    winds = [(np.zeros((ny, nx)), np.zeros((ny, nx)))] * 4
    cold = oc.run_ocean(EARTH, lat, sea, winds, 150.0, params, years=3)
    warm = oc.run_ocean(EARTH, lat, sea, winds, 150.0, params, years=3,
                        exchange=[("east", (5e5,) * 12, (25.0,) * 12)])  # fmt: skip
    d = warm.diagnostics
    assert d["heat_relative_residual"] < 1e-8
    assert d["boundary_heat_j"] > 0
    east = np.zeros((ny, nx), bool)
    east[:, -3:] = True
    gain = np.nanmean(warm.sst_monthly[:, east & sea]) - np.nanmean(cold.sst_monthly[:, east & sea])
    far = np.zeros((ny, nx), bool)
    far[:, 6:9] = True
    gain_far = np.nanmean(warm.sst_monthly[:, far & sea]) - np.nanmean(
        cold.sst_monthly[:, far & sea])  # fmt: skip
    assert gain > 0.3 and gain > 2 * gain_far  # the warm inflow is felt at the edge
    west_exchange = oc.run_ocean(EARTH, lat, sea, winds, 150.0, params, years=3,
                                 exchange=[("west", (5e5,) * 12, (25.0,) * 12)])  # fmt: skip
    assert west_exchange.diagnostics["boundary_heat_j"] == 0.0  # the west edge is land


def test_boundary_editor_offscreen(qtbot=None):
    pytest.importorskip("PySide6")
    import os

    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PySide6.QtWidgets import QApplication

    from hexworld.gui.boundary_editor import BoundaryEditor

    _app = QApplication.instance() or QApplication([])
    dlg = BoundaryEditor("")
    dlg.combos["west"].setCurrentIndex(dlg.combos["west"].findData("warm_ocean"))
    dlg.presets_to_json()
    assert dlg.check()
    profs = bnd.from_json(json.dumps(json.loads(dlg.text.toPlainText())))
    assert profs[0].sides == ("west",) and profs[0].preset == "warm_ocean"
    dlg.text.setPlainText('{"schema": "hexworld-boundary-1", "profiles": [{"boundary_id": "x",'
                          ' "sides": ["up"]}]}')  # fmt: skip
    assert not dlg.check()  # an unknown side is shown as an error, never a zero
