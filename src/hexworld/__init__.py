"""HexWorld - procedural hexagonal geographic map generator.

v1.0: tectonic foundation (plates, crust prior, plate motion, boundary graph,
diagnostic fields). v2.0: continental blocks, macro relief, sea level and
landmasses. v3.0: geological history, mountains, volcanism, limited erosion.
v4.0: climate, water balance, rivers, lakes, erosion and sediment.
Public API::

    from hexworld import Config, generate, save_project, load_project
    from hexworld import generate_v2, derive_landmasses, get_layer
"""

from .api import (
    build_hydrology,
    derive_landmasses,
    erode,
    generate_climate,
    generate_continents,
    generate_history,
    generate_macro_relief,
    generate_v2,
    generate_v4,
    get_layer,
    history_events,
    sample_points,
    select_epoch,
)
from .config import PRESETS, Config, ConfigError
from .io import load_project, save_project
from .pipeline import GenerationCancelled, generate
from .project import Project
from .version import GENERATOR_VERSION as __version__

__all__ = [
    "build_hydrology",
    "erode",
    "generate_climate",
    "generate_v4",
    "PRESETS",
    "Config",
    "ConfigError",
    "GenerationCancelled",
    "Project",
    "__version__",
    "derive_landmasses",
    "generate",
    "generate_continents",
    "generate_history",
    "generate_macro_relief",
    "generate_v2",
    "get_layer",
    "history_events",
    "load_project",
    "sample_points",
    "save_project",
    "select_epoch",
]
