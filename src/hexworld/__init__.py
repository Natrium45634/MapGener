"""HexWorld - procedural hexagonal geographic map generator.

v1.0: tectonic foundation (plates, crust prior, plate motion, boundary graph,
diagnostic fields). v2.0: continental blocks, macro relief, sea level and
landmasses. Public API::

    from hexworld import Config, generate, save_project, load_project
    from hexworld import generate_v2, derive_landmasses, get_layer
"""

from .api import (
    derive_landmasses,
    generate_continents,
    generate_macro_relief,
    generate_v2,
    get_layer,
)
from .config import PRESETS, Config, ConfigError
from .io import load_project, save_project
from .pipeline import GenerationCancelled, generate
from .project import Project
from .version import GENERATOR_VERSION as __version__

__all__ = [
    "PRESETS",
    "Config",
    "ConfigError",
    "GenerationCancelled",
    "Project",
    "__version__",
    "derive_landmasses",
    "generate",
    "generate_continents",
    "generate_macro_relief",
    "generate_v2",
    "get_layer",
    "load_project",
    "save_project",
]
