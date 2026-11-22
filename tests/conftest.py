import os
import tempfile

import pytest

# «Мои миры» and other per-user files of the tests go to a throw-away folder
os.environ.setdefault("HEXWORLD_DATA_DIR", tempfile.mkdtemp(prefix="hexworld-test-"))

from hexworld import Config, generate  # noqa: E402
from hexworld.config import ContinentParams, HistoryParams  # noqa: E402

#: A coarser world skeleton keeps the suite fast; the algorithms are the same.
FAST = ContinentParams(skeleton_cells=224)
#: v1/v2 tests do not need a simulated history.
NO_HISTORY = HistoryParams(enabled=False)
#: A short history for the v3 tests.
SHORT_HISTORY = HistoryParams(duration_ma=16.0, time_step_ma=2.0, checkpoint_stride=2)


@pytest.fixture(scope="session")
def small_config():
    return Config(
        seed="Ab12", width=100, height=100, hex_spacing_km=80.0, continents=FAST, history=NO_HISTORY
    )


@pytest.fixture(scope="session")
def small_project(small_config):
    return generate(small_config)


@pytest.fixture(scope="session")
def hist_config():
    return Config(
        seed="Ab12",
        width=100,
        height=100,
        hex_spacing_km=80.0,
        continents=FAST,
        history=SHORT_HISTORY,
    )


@pytest.fixture(scope="session")
def hist_project(hist_config):
    return generate(hist_config)
