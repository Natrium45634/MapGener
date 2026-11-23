"""World settings: schema, randomisation with locks, presets, progress tracker, the dialog."""

from __future__ import annotations

import os
import re

import pytest

from hexworld import settings_schema as ss
from hexworld.config import Config, ConfigError
from hexworld.progress import ProgressTracker, format_seconds, sub_report


def test_schema_covers_every_config_parameter():
    bases = {re.split(r"[\[:]", p.path)[0] for p in ss.PARAMS}
    missing = [p for p in ss.all_config_paths() if p not in bases]
    # the seed has its own field; the simulation bounds are always derived ("auto"); the
    # weather-year seed is a text value of the config / CLI (empty = the world seed)
    assert sorted(missing) == ["seed", "simulation_bounds_km"]


def test_schema_paths_labels_and_ranges():
    assert len({p.path for p in ss.PARAMS}) == len(ss.PARAMS)
    groups = {g for g, _t in ss.GROUPS}
    for p in ss.PARAMS:
        assert p.group in groups and p.label
        if p.kind in ("float", "int"):
            assert p.lo < p.hi, p.path
            if p.rand is not None:
                assert p.lo <= p.rand[0] <= p.rand[1] <= p.hi, p.path
        if p.kind == "choice":
            assert p.choices


def test_round_trip_defaults():
    values = ss.values_from_config(Config())
    assert ss.config_from_values(values) == Config()


def test_round_trip_tuples_and_members():
    v = ss.values_from_config(Config())
    v["terrain.basins:rift_graben"] = False
    v["terrain.age_depth_transition_ma[1]"] = 90.0
    v["plates.count"] = 12
    c = ss.config_from_values(v)
    assert "rift_graben" not in c.terrain.basins
    assert c.terrain.age_depth_transition_ma == (60.0, 90.0)
    assert c.plates.count == 12
    v["plates.count"] = None
    assert ss.config_from_values(v).plates.count == "auto"


def test_random_settings_are_always_valid():
    base = ss.values_from_config(Config())
    for i in range(300):
        assert ss.validate_values(ss.randomize(base, seed=f"R{i}")) == []


def test_randomize_respects_locks_and_is_reproducible():
    base = ss.values_from_config(Config())
    base["climate.temperature_offset_c"] = -12.0
    locked = {"climate.temperature_offset_c", "sea.target_land_fraction"}
    a = ss.randomize(base, locked, seed="Lock1")
    b = ss.randomize(base, locked, seed="Lock1")
    assert a == b
    assert a["climate.temperature_offset_c"] == -12.0
    assert a["sea.target_land_fraction"] == base["sea.target_land_fraction"]
    changed = sum(1 for k in base if a[k] != base[k])
    assert changed > 40


def test_technical_settings_are_not_randomised():
    base = ss.values_from_config(Config())
    a = ss.randomize(base, seed="Tech1")
    for path in ("width", "height", "hex_spacing_km", "mode", "topoclimate.quality"):
        assert a[path] == base[path]
    assert {"width", "height", "hex_spacing_km"} <= ss.default_locked()


def test_world_presets_are_valid_and_change_the_climate():
    base = ss.values_from_config(Config())
    for name, overrides in ss.WORLD_PRESETS.items():
        v = dict(base)
        v.update(overrides)
        assert ss.validate_values(v) == [], name
    cold = ss.config_from_values({**base, **ss.WORLD_PRESETS["Ледниковый период"]})
    assert cold.climate.temperature_offset_c < -5


def test_random_seed_is_valid():
    for _ in range(50):
        s = ss.random_seed()
        assert re.fullmatch(r"[A-Za-z][A-Za-z0-9]{7}", s)
        Config(seed=s)


def test_island_amount_is_a_setting():
    assert ss.param("terrain.island_amount").hi == 3.0
    with pytest.raises(ConfigError):
        Config.from_dict({**Config().to_dict(), "terrain": {"island_amount": 5.0}})


def test_size_presets_and_estimates():
    for _t, w, h, km, mode in ss.SIZE_PRESETS:
        Config(width=w, height=h, hex_spacing_km=km, mode=mode)
    small = ss.estimate_seconds(100, 100)
    big = ss.estimate_seconds(1000, 800)
    assert 40 < small < 150 and 200 < big < 450 and big > small


# -- progress ------------------------------------------------------------------------------------


class _Clock:
    def __init__(self):
        self.t = 0.0

    def __call__(self):
        return self.t


def test_progress_is_monotonic_and_ordered():
    clock = _Clock()
    tr = ProgressTracker("v50", 10_000, clock=clock)
    seq = [("plates", 0.02), ("crust", 0.12), ("validate", 0.79), ("world_tectonics", 0.8),
           ("geo_history", 0.80), ("geo_history", 0.84), ("geo_history", 0.88),
           ("v3_raster", 0.95), ("v4_evolution", 0.99), ("v45_water", 0.998),
           ("v45_wetness", 0.9988), ("v47_topoclimate", 0.9975),
           (sub_report("v47_topoclimate", 0.5), 0.9975), ("v50_biomes", 0.998),
           (sub_report("v50_biomes", 0.9), 0.998), ("done", 1.0)]  # fmt: skip
    last = -1.0
    for stage, f in seq:
        clock.t += 2.0
        st = tr.report(stage, f)
        assert st.overall >= last
        last = st.overall
    assert last == 1.0 and st.eta_s == 0.0
    assert [k for k, _t, _s in st.done][:3] == ["plates", "crust", "motion"]


def test_progress_validate_does_not_jump_to_the_end():
    tr = ProgressTracker("v50", 10_000, clock=_Clock())
    tr.report("crust", 0.12)
    st = tr.report("validate", 0.79)
    assert st.stage == "crust" and st.overall < 0.1


def test_progress_estimates_never_pass_the_stage():
    clock = _Clock()
    tr = ProgressTracker("v50", 10_000, clock=clock)
    tr.report("geo_history", 0.80)
    before = tr.state().overall
    clock.t += 10_000.0  # a stage that never reports
    st = tr.state()
    nxt = tr._weight_before("mountain_plan") / tr.total
    assert before <= st.overall <= nxt + 1e-9


def test_progress_through_limits_the_stages():
    assert "v50_biomes" not in ProgressTracker("v3", 1000).keys
    assert ProgressTracker("v3", 1000).keys[-1] == "done"


def test_format_seconds():
    assert format_seconds(None) == "—"
    assert format_seconds(5) == "5 с"
    assert format_seconds(125) == "2 мин 05 с"
    assert format_seconds(3700).startswith("1 ч")


# -- the dialog (offscreen) ----------------------------------------------------------------------


@pytest.fixture
def qapp():
    pytest.importorskip("PySide6")
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    try:
        from PySide6.QtWidgets import QApplication
    except ImportError as exc:
        pytest.skip(f"Qt unavailable: {exc}")
    return QApplication.instance() or QApplication([])


def test_settings_dialog_locks_randomise_start(qapp):
    from hexworld.gui.world_settings import WorldSettingsDialog

    d = WorldSettingsDialog(Config(seed="Dlg1"))
    d._set_locks(ss.default_locked())
    assert len(d.rows) == len(ss.PARAMS)
    row = d.rows["climate.temperature_offset_c"]
    row.set(-11.0)
    row.lock.setChecked(True)
    d.seed_lock.setChecked(True)
    d.randomize()
    assert d.values()["climate.temperature_offset_c"] == -11.0
    assert d.seed.text() == "Dlg1"
    d.size_preset.setCurrentIndex(0)
    assert d.values()["width"] == ss.SIZE_PRESETS[0][1]
    d.accept_start()
    assert d.start and d.config.seed == "Dlg1" and d.config.climate.temperature_offset_c == -11.0
    assert d.config.width == ss.SIZE_PRESETS[0][1]


def test_settings_dialog_document_round_trip(qapp):
    from hexworld.gui.world_settings import WorldSettingsDialog

    a = WorldSettingsDialog(Config(seed="Doc1"))
    a.rows["terrain.island_amount"].set(2.5)
    a.rows["terrain.island_amount"].lock.setChecked(True)
    doc = a.settings_document()
    b = WorldSettingsDialog(Config(seed="Other"))
    b.apply_document(doc)
    assert b.seed.text() == "Doc1"
    assert b.values()["terrain.island_amount"] == 2.5
    assert "terrain.island_amount" in b.locked()


def test_slider_and_value_stay_in_sync(qapp):
    from hexworld.gui.world_settings import WorldSettingsDialog

    d = WorldSettingsDialog(Config())
    row = d.rows["crust.mean_continent_area_km2"]  # a log-scale slider
    row.slider.setValue(1000)
    assert row.get() == pytest.approx(row.p.hi, rel=1e-6)
    row.edit.setValue(1e6)
    assert row.get() == pytest.approx(1e6)
    auto = d.rows["plates.count"]
    auto.auto.setChecked(True)
    assert auto.get() is None
