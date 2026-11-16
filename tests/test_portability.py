"""Projects move between machines: Windows line ends, other CPUs and NumPy builds."""

from __future__ import annotations

import shutil
from pathlib import Path
from unittest import mock

import numpy as np

import hexworld.stages.plates as plates
from hexworld import load_project

V2_FIXTURE = Path(__file__).parent / "data" / "v2_project"


def _crlf_copy(tmp_path) -> Path:
    work = tmp_path / "p"
    shutil.copytree(V2_FIXTURE, work)
    for f in work.glob("*.json"):  # what git with core.autocrlf=true does on Windows
        f.write_bytes(f.read_bytes().replace(b"\r\n", b"\n").replace(b"\n", b"\r\n"))
    return work


def test_project_with_windows_line_ends_still_verifies(tmp_path):
    original = load_project(V2_FIXTURE)
    crlf = load_project(_crlf_copy(tmp_path))  # control hashes are checked
    assert crlf.content_hashes() == original.content_hashes()
    assert crlf.geo.geo_hash() == original.geo.geo_hash()


def test_models_differing_in_the_last_bits_are_still_reproduced(tmp_path):
    """Another CPU can round sin / cos / exp differently: the exact model hash differs, but
    the rebuilt models still give the stored plates and crust, so the project opens."""
    with mock.patch.object(plates.PlateModel, "model_hash", lambda self: "other-cpu"):
        p = load_project(V2_FIXTURE)
    assert p.plate_model is not None and p.geo is not None
    assert any("floating-point differences" in w for w in p.warnings)
    hx, hy = p.layout.centers()
    assert np.array_equal(p.plate_model.plate_at(hx, hy), p.layers["plate_id"])


def test_a_really_different_model_is_refused(tmp_path):
    with (
        mock.patch.object(plates.PlateModel, "model_hash", lambda self: "other-world"),
        mock.patch.object(
            plates.PlateModel, "plate_at", lambda self, x, y: np.zeros(np.shape(x), dtype=int)
        ),
    ):
        p = load_project(V2_FIXTURE)
    assert p.plate_model is None
    assert any("could not be reproduced" in w for w in p.warnings)
