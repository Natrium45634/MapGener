"""Artificial two-plate worlds with known answers (acceptance criterion 4)."""

import numpy as np
import pytest

from hexworld import hexgrid
from hexworld.config import BoundaryParams, FieldParams
from hexworld.geometry import HexLayout
from hexworld.stages import boundaries as B
from hexworld.stages.fields import compute_fields
from hexworld.stages.motion import PlateMotion
from hexworld.stages.plates import PlateModel

LAYOUT = HexLayout(60, 40, 50.0, 0.0, 0.0)
PARAMS = BoundaryParams()


def two_plates(vertical=True):
    """Plate 0 in the west (or north), plate 1 in the east (or south)."""
    b = LAYOUT.bounds
    cx, cy = 0.5 * (b.xmin + b.xmax), 0.5 * (b.ymin + b.ymax)
    sites = [(cx - 600, cy), (cx + 600, cy)] if vertical else [(cx, cy - 600), (cx, cy + 600)]
    model = PlateModel.synthetic(sites)
    x, y = LAYOUT.centers()
    return model, model.plate_at(x, y)


def edges_for(v0, v1, vertical=True, crust=(0, 0)):
    model, pid = two_plates(vertical)
    crust_type = np.where(pid == 0, crust[0], crust[1]).astype(np.uint8)
    motion = PlateMotion.uniform([v0, v1])
    return B.extract_boundaries(pid, crust_type, LAYOUT, motion, PARAMS, model), pid


def test_normal_points_from_a_to_b():
    edges, _ = edges_for((0, 0), (0, 0))
    assert len(edges) > 0
    assert np.all(edges.plate_a == 0) and np.all(edges.plate_b == 1)
    assert np.allclose(edges.normal_x, 1.0, atol=1e-3)  # plate 1 lies to the east (+x)
    assert np.allclose(edges.normal_y, 0.0, atol=2e-2)
    edges, _ = edges_for((0, 0), (0, 0), vertical=False)
    assert np.allclose(edges.normal_y, 1.0, atol=1e-3)  # plate 1 lies to the south (+y)


def test_head_on_motion_is_convergent():
    edges, _ = edges_for((30, 0), (-20, 0))
    assert np.all(edges.boundary_type == B.TYPE_CONVERGENT)
    assert np.allclose(edges.vn, -50.0, atol=2e-2)
    assert np.allclose(edges.vt, 0.0, atol=2e-2)
    assert np.allclose(edges.intensity, 0.5, atol=1e-4)


def test_moving_apart_is_divergent():
    edges, _ = edges_for((-30, 0), (20, 0))
    assert np.all(edges.boundary_type == B.TYPE_DIVERGENT)
    assert np.allclose(edges.vn, 50.0, atol=2e-2)


def test_sideways_motion_is_transform_with_documented_sign():
    # B (east) moves south relative to A: seen from A across the fault, B moves right
    edges, _ = edges_for((0, -20), (0, 20))
    assert np.all(edges.boundary_type == B.TYPE_TRANSFORM)
    assert np.allclose(edges.vt, 40.0, atol=2e-2)  # > 0: right-lateral (dextral)
    edges, _ = edges_for((0, 20), (0, -20))
    assert np.allclose(edges.vt, -40.0, atol=2e-2)  # < 0: left-lateral (sinistral)
    assert np.allclose(edges.vn, 0.0, atol=2e-2)


def test_same_motion_is_quiet_and_orientation_does_not_matter():
    edges, _ = edges_for((25, 10), (25, 10))
    assert np.all(edges.boundary_type == B.TYPE_QUIET)
    # horizontal boundary, plates closing along y
    edges, _ = edges_for((0, 30), (0, -30), vertical=False)
    assert np.all(edges.boundary_type == B.TYPE_CONVERGENT)
    edges, _ = edges_for((40, 0), (-40, 0), vertical=False)
    assert np.all(edges.boundary_type == B.TYPE_TRANSFORM)


def test_oblique_motion_respects_obliquity_threshold():
    kind, _ = B.classify(np.array([-40.0, -10.0, 1.0]), np.array([10.0, 40.0, 1.0]), PARAMS)
    assert kind.tolist() == [B.TYPE_CONVERGENT, B.TYPE_TRANSFORM, B.TYPE_QUIET]


def test_rotation_gives_different_regimes_along_one_boundary():
    model, pid = two_plates()
    b = LAYOUT.bounds
    motion = PlateMotion(
        velocity=np.zeros((2, 2)),
        omega=np.array([0.0, 0.6]),
        pole=np.array([[0.0, 0.0], [0.5 * (b.xmin + b.xmax), 0.5 * (b.ymin + b.ymax)]]),
    )
    edges = B.extract_boundaries(
        pid, np.zeros_like(pid, dtype=np.uint8), LAYOUT, motion, PARAMS, model
    )
    kinds = set(edges.boundary_type.tolist())
    assert B.TYPE_CONVERGENT in kinds and B.TYPE_DIVERGENT in kinds


def test_edges_are_exactly_the_shared_hex_edges():
    edges, pid = edges_for((30, 0), (-20, 0))
    brute = set()
    for row in range(LAYOUT.height):
        for col in range(LAYOUT.width):
            for d in range(6):
                nc, nr = (int(v) for v in hexgrid.offset_neighbor(col, row, d))
                if LAYOUT.inside(nc, nr) and pid[row, col] < pid[nr, nc]:
                    brute.add((col, row, nc, nr))
    got = set(
        zip(
            edges.col_a.tolist(),
            edges.row_a.tolist(),
            edges.col_b.tolist(),
            edges.row_b.tolist(),
            strict=True,
        )
    )
    assert got == brute
    assert np.all(hexgrid.offset_distance(edges.col_a, edges.row_a, edges.col_b, edges.row_b) == 1)
    assert edges.length_km == pytest.approx(LAYOUT.radius)


def test_subduction_polarity_and_volcanic_arc_side():
    # plate 0 oceanic (west), plate 1 continental (east), converging
    edges, pid = edges_for((40, 0), (-10, 0), crust=(0, 1))
    assert np.all(edges.subducting == B.SUBDUCTING_A)
    fields = compute_fields(edges, LAYOUT, FieldParams())
    volc = fields["volcanism"]
    assert volc[pid == 1].max() > 0.3  # arc on the overriding (continental) plate
    assert volc[pid == 0].max() < 0.02  # none on the subducting plate
    peak = np.unravel_index(np.argmax(volc), volc.shape)
    assert 80.0 < fields["boundary_distance_km"][peak] < 280.0  # offset from the trench
    assert fields["rifting"].max() == 0.0
    assert fields["subduction"].max() > 0.3


def test_continental_collision_uplifts_both_sides_without_volcanoes():
    edges, pid = edges_for((40, 0), (-40, 0), crust=(1, 1))
    assert np.all(edges.subducting == B.SUBDUCTING_NONE)
    fields = compute_fields(edges, LAYOUT, FieldParams())
    assert fields["volcanism"].max() == 0.0
    assert fields["subduction"].max() == 0.0
    assert fields["uplift"][pid == 0].max() > 0.5 and fields["uplift"][pid == 1].max() > 0.5


def test_divergence_gives_rifting_not_uplift():
    edges, _ = edges_for((-30, 0), (30, 0))
    fields = compute_fields(edges, LAYOUT, FieldParams())
    assert fields["rifting"].max() > 0.4
    assert fields["uplift"].max() == 0.0
