"""HexWorld - procedural hexagonal geographic map generator.

v1.0 implements the tectonic foundation: plates, crust, plate motion,
boundary graph and diagnostic fields. Public API::

    from hexworld import Config, generate, save_project, load_project
"""

from .config import PRESETS, Config, ConfigError
from .io import load_project, save_project
from .pipeline import generate
from .project import Project
from .version import GENERATOR_VERSION as __version__

__all__ = [
    "PRESETS",
    "Config",
    "ConfigError",
    "Project",
    "__version__",
    "generate",
    "load_project",
    "save_project",
]
