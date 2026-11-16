"""Varied mountain systems: segmentation, parallel ranges, styles, irregularity, age."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from hexworld.config import MOUNTAIN_STYLES, MountainParams
from hexworld.mountains import build_plan
from hexworld.mountains.plan import arclength
from hexworld.stages.history import HistoryGrid


def _belt(length_km=5200.0, width_km=900.0, relief_m=3200.0, cell=60.0, bend=True):
    """A synthetic orogenic belt on a history grid: a curved band with a relief bump."""
    nx, ny = int((length_km + 1600) / cell), int((width_km * 3 + 1200) / cell)
    grid = HistoryGrid(0.0, 0.0, cell, nx, ny)
    gx, gy = grid.centers()
    x0 = 800.0
    t = np.clip((gx - x0) / length_km, 0.0, 1.0)
    centre = 0.5 * ny * cell + (420.0 * np.sin(np.pi * t) if bend else 0.0)
    across = np.abs(gy - centre) / (0.5 * width_km)
    along = np.clip(np.minimum(gx - x0, x0 + length_km - gx) / 300.0, 0.0, 1.0)
    neck = 1.0 - 0.45 * np.exp(-(((gx - x0 - 0.5 * length_km) / 160.0) ** 2))
    relief = relief_m * np.clip(1.0 - across**2, 0.0, 1.0) * along * neck
    kind = np.where(relief > 0, 1, 0).astype(np.int64)  # collision
    return grid, relief, kind


def _plan(params=None, age=5.0, worn=0.1, seed="Belt1", **belt):
    grid, relief, kind = _belt(**belt)
    age_ma = np.full(grid.shape, age)
    share = np.full(grid.shape, worn)
    elevation = relief + 200.0
    return build_plan(
        grid, relief, kind, age_ma, share, elevation, params or MountainParams(), seed, 0.0
    )


def _segments(points):
    return [(points[i], points[i + 1]) for i in range(len(points) - 1)]


def _cross(p1, p2, q1, q2):
    def orient(a, b, c):
        return np.sign((b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0]))

    return (
        orient(p1, p2, q1) * orient(p1, p2, q2) < 0 and orient(q1, q2, p1) * orient(q1, q2, p2) < 0
    )


def test_a_long_belt_is_several_systems_of_several_ranges():
    plan = _plan()
    assert len(plan.systems) >= 3  # 5200 km of belt are not one mountain system
    assert all(s.length_km < 3200.0 for s in plan.systems)
    multi = [s for s in plan.systems if s.ranges >= 2]
    assert len(multi) >= 2  # wide belts carry parallel ranges, not one line
    roles = {r.role for r in plan.ridges}
    assert "spur" in roles and len(roles - {"spur", "branch"}) >= 2
    assert {s.style for s in plan.systems} <= set(MOUNTAIN_STYLES)


def test_spurs_follow_the_flanks_and_do_not_cross():
    plan = _plan()
    by_system: dict[int, list] = {}
    for r in plan.ridges:
        if r.role == "spur" and r.parent >= 0 and plan.ridges[r.parent].role != "spur":
            by_system.setdefault(r.system, []).append(r)
    checked = crossings = 0
    for spurs in by_system.values():
        segs = [(k, s) for k, r in enumerate(spurs) for s in _segments(r.points[2:])]
        for i in range(0, len(segs), 3):
            for j in range(i + 1, len(segs), 3):
                if segs[i][0] == segs[j][0]:
                    continue
                checked += 1
                crossings += _cross(*segs[i][1], *segs[j][1])
    assert checked > 1000
    assert crossings / checked < 0.002  # descent lines of one field do not cross


def test_spacing_of_peaks_and_spurs_is_irregular():
    plan = _plan()
    cvs = []
    for r in plan.ridges:
        if r.role != "main":
            continue
        s = arclength(r.points)
        on = [p for p in plan.peaks if p["ridge"] == r.id]
        if len(on) < 6:
            continue
        pos = np.sort(
            [s[int(np.argmin(np.hypot(*(r.points - [p["x_km"], p["y_km"]]).T)))] for p in on]
        )
        gaps = np.diff(pos)
        cvs.append(gaps.std() / gaps.mean())
    assert cvs and np.median(cvs) > 0.2  # not evenly spaced
    for item in plan.passes:  # still: a pass is a saddle between two peaks of one ridge
        a, b = (plan.peaks[i] for i in item["between_peaks"])
        assert a["ridge"] == b["ridge"] == item["ridge"]
        assert item["crest"] < min(a["crest"], b["crest"])


def test_old_belts_are_broad_low_and_rounded():
    old_auto = _plan(age=320.0, worn=0.9)
    assert {s.style for s in old_auto.systems} <= {
        "old_massif",
        "broad_upland",
        "fold_belt",
        "mixed",
    }
    assert np.mean([s.maturity for s in old_auto.systems]) > 0.7
    # the same architecture young and old
    alpine = MountainParams(style="alpine")
    young = _plan(alpine, age=3.0, worn=0.05)
    old = _plan(alpine, age=320.0, worn=0.9)

    def median(plan, attr):
        return np.median([getattr(r, attr) for r in plan.ridges if r.role not in ("spur",)])

    assert median(old, "half_width_km") > 1.3 * median(young, "half_width_km")  # broader
    assert median(old, "sharpness") < median(young, "sharpness")  # rounder
    spurs_young = sum(r.role == "spur" for r in young.ridges) / len(young.systems)
    spurs_old = sum(r.role == "spur" for r in old.ridges) / len(old.systems)
    assert spurs_old < spurs_young  # small spurs are the first to go
    peaks_young = [p["crest"] for p in young.peaks]
    peaks_old = [p["crest"] for p in old.peaks]
    assert np.std(peaks_old) < np.std(peaks_young)  # accordant summits


@pytest.mark.parametrize("style", MOUNTAIN_STYLES)
def test_every_style_can_be_forced(style):
    plan = _plan(MountainParams(style=style), length_km=2200.0)
    assert plan.systems and all(s.style == style for s in plan.systems)
    if style in ("himalayan", "broad_upland"):
        assert any(r.plateau > 0.3 for r in plan.ridges)  # a level top
    if style == "fold_belt":
        assert sum(r.role == "fold" for r in plan.ridges) >= 4
    if style == "block":
        assert any(abs(r.asymmetry) > 0.3 for r in plan.ridges)  # tilted blocks


def test_complexity_zero_keeps_one_crest_per_system():
    simple = _plan(replace(MountainParams(), complexity=0.0))
    for s in simple.systems:
        main = [r for r in simple.ridges if r.system == s.id and r.rank == 0]
        assert len(main) == 1
    assert len(simple.systems) == 1  # and no segmentation either


def test_plan_is_deterministic_and_seed_sensitive():
    a, b = _plan(), _plan()
    assert [r.points.tolist() for r in a.ridges] == [r.points.tolist() for r in b.ridges]
    c = _plan(seed="Belt2")
    assert [r.points.tolist() for r in a.ridges] != [r.points.tolist() for r in c.ridges]
