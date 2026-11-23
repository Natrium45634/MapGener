"""TZ 5.4.1 G8 / sec. 11.1: legacy read, upgrade copy, current-version reproducibility
(v5_4_1::T06, T07, T08, T09, T84). Genuine old files exist only for v1 and v2
(``docs/v5_4_1/legacy_fixture_inventory.md``); v3-v5.4 are BLOCKED_MISSING_FIXTURE."""

from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

import pytest

from hexworld import Config, generate
from hexworld.config import ContinentParams, HistoryParams
from hexworld.io import load_project, save_project

DATA = Path(__file__).parent / "data"
GOLDEN = Path(__file__).parent / "golden"


def _bytes(folder: Path) -> dict:
    return {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(folder.iterdir())}


@pytest.mark.parametrize("name", ["v1_project", "v2_project"])
def test_t06_legacy_read_keeps_the_stored_payload(name, tmp_path):
    src = DATA / name
    before = _bytes(src)
    work = tmp_path / name
    shutil.copytree(src, work)
    manifest = json.loads((src / "manifest.json").read_text(encoding="utf-8"))
    p = load_project(work, verify=True)
    assert p.content_hashes() == manifest["content_hashes"]  # canonical payload unchanged
    assert _bytes(work) == before and _bytes(src) == before  # reading writes nothing


@pytest.mark.parametrize("name", ["v1_project", "v2_project"])
def test_t07_upgrade_copy_is_idempotent_and_keeps_the_original(name, tmp_path):
    src = DATA / name
    before = _bytes(src)
    p = load_project(src, verify=True)
    a = save_project(p, tmp_path / "copy_a")
    b = save_project(load_project(a), tmp_path / "copy_b")
    ma = json.loads((Path(a) / "manifest.json").read_text(encoding="utf-8"))
    mb = json.loads((Path(b) / "manifest.json").read_text(encoding="utf-8"))
    assert ma["content_hashes"] == mb["content_hashes"]
    assert _bytes(src) == before  # the original was never overwritten


def test_t08_legacy_regeneration_status_is_recorded():
    inv = Path(__file__).parents[1] / "docs" / "v5_4_1" / "legacy_fixture_inventory.md"
    text = inv.read_text(encoding="utf-8")
    assert "BLOCKED_MISSING_FIXTURE" in text  # never a synthetic "old" file


def _gold_config():
    return Config(seed="Gold1", width=60, height=40, hex_spacing_km=80.0,
                  continents=ContinentParams(skeleton_cells=160),
                  history=HistoryParams(enabled=False))  # fmt: skip


def test_t09_current_version_is_reproducible_and_matches_the_golden(tmp_path):
    a = generate(_gold_config(), through="v2")
    b = generate(_gold_config(), through="v2")
    ha, hb = a.content_hashes(), b.content_hashes()
    assert ha == hb  # same seed / config / algorithm -> the same canonical hashes
    gold_file = GOLDEN / "current" / "gold1_v2.json"
    gold = json.loads(gold_file.read_text(encoding="utf-8"))
    import platform

    if platform.system() == gold["platform"] and platform.machine() == gold["machine"]:
        assert ha == gold["content_hashes"]
    else:  # other platforms: the same keys, bitwise floats are not promised
        assert set(ha) == set(gold["content_hashes"])
    path = save_project(a, tmp_path / "g")
    assert load_project(path).content_hashes() == ha  # T84: save / reload without regeneration
