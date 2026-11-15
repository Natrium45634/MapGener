"""Version identifiers written into every saved project."""

GENERATOR_VERSION = "3.0.0"
#: Bumped whenever a change alters generated geology for the same seed + config.
ALGORITHM_VERSION = "tectonics-1"
#: Seed -> stream derivation and raw number generation scheme.
PRNG_VERSION = "sha256-pcg64-raw-1"
#: On-disk project layout.
SCHEMA_VERSION = "hexworld-project-3"
#: Schemas this build can open.
SUPPORTED_SCHEMAS = ("hexworld-project-1", "hexworld-project-2", "hexworld-project-3")
#: Bumped whenever v2.0 stages change their output for the same inputs.
CONTINENTS_ALGORITHM_VERSION = "continents-1"
#: World coordinate / hex conventions (see docs/DATA_FORMAT.md).
COORDINATE_VERSION = "km-x-east-y-south/flat-top-odd-q-1"
#: Bumped whenever v3.0 history / volcanism / erosion change their output.
HISTORY_ALGORITHM_VERSION = "geohistory-1"
#: Bumped whenever the mountain plan or the reference height backend change.
MOUNTAINS_ALGORITHM_VERSION = "mountains-1"
