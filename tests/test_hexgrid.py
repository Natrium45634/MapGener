import numpy as np

from hexworld import hexgrid
from hexworld.geometry import HexLayout


def test_offset_axial_roundtrip():
    cols, rows = np.meshgrid(np.arange(-7, 9), np.arange(-6, 8))
    q, r = hexgrid.offset_to_axial(cols, rows)
    c2, r2 = hexgrid.axial_to_offset(q, r)
    assert np.array_equal(c2, cols) and np.array_equal(r2, rows)


def test_six_distinct_neighbors_at_distance_one():
    for col, row in [(4, 4), (5, 4), (0, 0), (1, 0), (-3, 7)]:
        seen = set()
        for d in range(6):
            nc, nr = hexgrid.offset_neighbor(col, row, d)
            assert hexgrid.offset_distance(col, row, nc, nr) == 1
            back_c, back_r = hexgrid.offset_neighbor(nc, nr, (d + 3) % 6)
            assert (int(back_c), int(back_r)) == (col, row)
            seen.add((int(nc), int(nr)))
        assert len(seen) == 6


def test_odd_q_layout_shifts_odd_columns_down():
    layout = HexLayout(4, 4, 10.0, 0.0, 0.0)
    _, y_even = layout.center(0, 0)
    _, y_odd = layout.center(1, 0)
    assert np.isclose(y_odd - y_even, 5.0)


def test_neighbor_centres_are_one_spacing_apart():
    layout = HexLayout(10, 10, 12.5, 100.0, -40.0)
    for col, row in [(3, 3), (4, 3)]:
        x, y = layout.center(col, row)
        for d in range(6):
            nc, nr = hexgrid.offset_neighbor(col, row, d)
            nx, ny = layout.center(nc, nr)
            assert np.isclose(np.hypot(nx - x, ny - y), 12.5)


def test_distance_properties():
    assert hexgrid.offset_distance(0, 0, 0, 0) == 0
    assert hexgrid.offset_distance(0, 0, 5, 0) == 5
    assert hexgrid.offset_distance(2, 3, 7, 9) == hexgrid.offset_distance(7, 9, 2, 3)
    q1, r1 = hexgrid.offset_to_axial(2, 3)
    x, y, z = hexgrid.axial_to_cube(q1, r1)
    assert x + y + z == 0


def test_world_to_offset_roundtrip_and_interior_points():
    layout = HexLayout(37, 23, 7.0, -50.0, 20.0)
    x, y = layout.centers()
    col, row = layout.world_to_offset(x, y)
    assert np.array_equal(col, np.broadcast_to(np.arange(37), (23, 37)))
    assert np.array_equal(row, np.broadcast_to(np.arange(23)[:, None], (23, 37)))
    # any point well inside a hex maps to that hex
    rng = np.random.default_rng(0)
    ang = rng.uniform(0, 2 * np.pi, x.shape)
    rad = rng.uniform(0, 0.49 * 7.0, x.shape)
    col, row = layout.world_to_offset(x + rad * np.cos(ang), y + rad * np.sin(ang))
    assert np.array_equal(col, np.broadcast_to(np.arange(37), (23, 37)))


def test_padded_layout_keeps_hex_positions():
    layout = HexLayout(6, 5, 3.0, 1.0, 2.0)
    padded = layout.padded(4)
    x, y = layout.center(1, 2)
    px, py = padded.center(5, 6)
    assert np.isclose(x, px) and np.isclose(y, py)
