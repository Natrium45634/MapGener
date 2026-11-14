import pytest

from hexworld import Config, generate
from hexworld.config import ContinentParams

#: A coarser world skeleton keeps the suite fast; the algorithms are the same.
FAST = ContinentParams(skeleton_cells=224)


@pytest.fixture(scope="session")
def small_config():
    return Config(seed="Ab12", width=100, height=100, hex_spacing_km=80.0, continents=FAST)


@pytest.fixture(scope="session")
def small_project(small_config):
    return generate(small_config)
