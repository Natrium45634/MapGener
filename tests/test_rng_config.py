import numpy as np
import pytest

from hexworld import Config, ConfigError
from hexworld.geometry import resolve_geometry
from hexworld.rng import Stream, validate_seed


@pytest.mark.parametrize("seed", ["", "ab c", "привет", "a-b", "a_b", None, 12])
def test_invalid_seeds_rejected(seed):
    with pytest.raises((ValueError, TypeError)):
        validate_seed(seed)


def test_seed_is_case_sensitive_and_streams_independent():
    a = Stream("Ab12", "plates").random(8)
    assert np.array_equal(a, Stream("Ab12", "plates").random(8))
    assert not np.array_equal(a, Stream("ab12", "plates").random(8))
    assert not np.array_equal(a, Stream("Ab12", "crust").random(8))


def test_stream_golden_values():
    """Guards the documented PRNG scheme against accidental change."""
    s = Stream("Ab12", "golden")
    assert [int(v) for v in s.raw(3)] == GOLDEN_RAW
    assert 0.0 <= s.random() < 1.0
    assert sorted(Stream("Ab12", "p").permutation(50).tolist()) == list(range(50))


GOLDEN_RAW = [66228601646372108, 15862164746268731861, 10712935376605686461]


def test_config_roundtrip_and_validation():
    c = Config(seed="Zz9", width=120, height=80, simulation_bounds_km=(-1e4, -1e4, 2e4, 2e4))
    assert Config.from_json(c.to_json()) == c
    with pytest.raises(ConfigError):
        Config(width=1)
    with pytest.raises(ConfigError):
        Config.from_dict({"sede": "x"})
    with pytest.raises(ValueError):
        Config(seed="bad seed")


def test_auto_plate_count_follows_physical_area_not_hex_count():
    coarse = resolve_geometry(Config(width=100, height=100, hex_spacing_km=80.0, mode="world"))
    fine = resolve_geometry(Config(width=400, height=400, hex_spacing_km=20.0, mode="world"))
    bigger = resolve_geometry(Config(width=400, height=400, hex_spacing_km=80.0, mode="world"))
    assert coarse.plate_count == fine.plate_count  # same physical area
    assert bigger.plate_count > coarse.plate_count


def test_regional_mode_generates_beyond_the_viewport():
    g = resolve_geometry(Config(width=100, height=100, hex_spacing_km=40.0))
    assert g.simulation_bounds.contains_bounds(g.viewport_bounds)
    assert g.simulation_bounds.area > 4 * g.viewport_bounds.area


def test_unreasonable_manual_count_warns():
    from hexworld.config import PlateParams

    g = resolve_geometry(
        Config(
            width=100, height=100, hex_spacing_km=2.0, mode="world", plates=PlateParams(count=300)
        )
    )
    assert any("per plate" in w for w in g.warnings)
