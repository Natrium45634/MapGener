"""Climate belt layouts (climate.latitude_layout): the map spans the planet from pole to pole
or a hemisphere; the old «part of the planet» stays the default."""

from __future__ import annotations

from dataclasses import replace

import pytest

from hexworld import Config
from hexworld import world_tools as wt
from hexworld.config import ConfigError
from hexworld.geometry import Bounds
from hexworld.stages.climate import POLE_EDGE_DEG, resolve_latitude


def _climate(layout, **kw):
    return replace(Config().climate, latitude_layout=layout, **kw)


def test_default_layout_is_the_old_part_of_the_planet():
    assert Config().climate.latitude_layout == "custom"
    b = Bounds(0.0, 0.0, 4000.0, 3000.0)
    lat = resolve_latitude(Config().climate, b)
    assert lat.center_deg == 30.0 and not lat.compressed


@pytest.mark.parametrize(
    "layout, top, bottom",
    [
        ("pole_to_pole", POLE_EDGE_DEG, -POLE_EDGE_DEG),
        ("north_hemisphere", POLE_EDGE_DEG, 0.0),
        ("south_hemisphere", 0.0, -POLE_EDGE_DEG),
    ],
)
def test_belt_layouts_span_from_the_pole(layout, top, bottom):
    b = Bounds(-500.0, 1000.0, 7000.0, 9000.0)
    # the centre and the scale are ignored in the belt layouts
    lat = resolve_latitude(_climate(layout, latitude_center_deg=60.0, km_per_degree=50.0), b)
    assert float(lat.at(b.ymin)) == pytest.approx(top)  # north is map up (-y)
    assert float(lat.at(b.ymax)) == pytest.approx(bottom)


def test_unknown_layout_is_rejected():
    d = Config().to_dict()
    d["climate"]["latitude_layout"] = "sideways"
    with pytest.raises(ConfigError):
        Config.from_dict(d)


def test_layout_in_the_world_code_and_old_codes_unchanged():
    cfg = Config(seed="Belt1")
    belts = replace(cfg, climate=replace(cfg.climate, latitude_layout="pole_to_pole"))
    back, _t = wt.parse_world_code(wt.world_code(belts))
    assert back.climate.latitude_layout == "pole_to_pole" and back == belts
    # a default setting added after the dictionary was frozen costs nothing in the code
    # a default of a new setting costs nothing; the 5.5.1 and 5.5 stages are on for new
    # worlds, so their switches are the only other change (old codes keep them off)
    assert wt._canonical(cfg, "v50") == (
        b'{"d":{"biosphere":{"enabled":true},"minerals":{"enabled":true},"seed":"Belt1"}}'
    )
    d = Config().to_dict()
    d["climate"].pop("latitude_layout")
    assert Config.from_dict(d).climate.latitude_layout == "custom"


def test_layout_in_the_settings_schema():
    from hexworld import settings_schema as ss

    p = {q.path: q for q in ss.PARAMS}["climate.latitude_layout"]
    assert p.kind == "choice" and p.rand is None  # never randomised
    assert [k for k, _t in p.choices] == [
        "custom",
        "pole_to_pole",
        "north_hemisphere",
        "south_hemisphere",
    ]


@pytest.mark.slow
def test_pole_to_pole_world_is_cold_at_both_edges():
    from hexworld import generate

    cfg = Config(seed="Belt2", width=80, height=64, hex_spacing_km=250.0, mode="world")
    cfg = replace(cfg, climate=replace(cfg.climate, latitude_layout="pole_to_pole"))
    p = generate(cfg, through="v45")
    t = p.env.monthly["temperature_monthly_c"].astype(float).mean(axis=0)
    north, middle, south = t[:4].mean(), t[30:34].mean(), t[-4:].mean()
    assert north < 0.0 and south < 0.0 and middle > 18.0
