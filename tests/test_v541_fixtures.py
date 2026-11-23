"""TZ 5.4.1 G8: the 5.3 and 5.4 scenario fixtures name real, existing test functions."""

from __future__ import annotations

import ast
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).parents[1]


def _functions(path: Path) -> set:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    return {n.name for n in tree.body if isinstance(n, ast.FunctionDef)}


@pytest.mark.parametrize("folder,count", [("v5_3", 24), ("v5_4", 24)])
def test_fixtures_point_to_existing_tests(folder, count):
    files = sorted((ROOT / "tests" / "fixtures" / folder).glob("S*.json"))
    assert len(files) == count
    for f in files:
        d = json.loads(f.read_text(encoding="utf-8"))
        refs = d.get("tests") or [d["test"]]
        for ref in refs:
            file_, func = ref.split("::")
            assert func in _functions(ROOT / file_), f"{f.name}: {ref} does not exist"
        if folder == "v5_3":
            assert d["status"].split(":")[0] in ("PASS", "PARTIAL", "OPEN")


def test_n01_t02_hex_geometry_and_parity():
    import math

    import numpy as np

    from hexworld.atmosphere import fv
    from hexworld.geometry import HexLayout
    from hexworld.pipeline_v50 import _hex_neighbours

    d_km = math.sqrt(3.0)  # centre distance of a hex with side a = 1000 m
    lay = HexLayout(8, 6, d_km, 0.0, 0.0)
    m = fv.hex_mesh(lay, _hex_neighbours(lay))
    assert m.area[0] == pytest.approx(2_598_076.211, rel=1e-9)
    assert m.length[0] == pytest.approx(1000.0, rel=1e-12)
    assert np.allclose(m.dist, 1732.050808, rtol=1e-9)
    # odd and even columns: every interior face appears once, its normal has unit length,
    # and the six normals of a cell cancel (closed polygon)
    assert np.allclose(np.hypot(m.normal[:, 0], m.normal[:, 1]), 1.0)
    tot = np.zeros((m.n, 2))
    np.add.at(tot, m.ei, m.normal * m.length[:, None])
    np.add.at(tot, m.ej, -m.normal * m.length[:, None])
    np.add.at(tot, m.bi, m.bnormal * m.blength[:, None])
    assert np.abs(tot).max() < 1e-6
