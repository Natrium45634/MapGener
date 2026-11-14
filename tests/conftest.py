import pytest

from hexworld import Config, generate


@pytest.fixture(scope="session")
def small_config():
    return Config(seed="Ab12", width=100, height=100, hex_spacing_km=80.0)


@pytest.fixture(scope="session")
def small_project(small_config):
    return generate(small_config)
