"""Version identifiers written into every saved project."""

#: Version of the application people install (installer, window title, «О программе»).
PRODUCT_VERSION = "1.3.0"
#: Version of the generator algorithms and data format (internal history 1.0 … 5.x).
GENERATOR_VERSION = "5.7.0a1"
#: Bumped whenever a change alters generated geology for the same seed + config.
ALGORITHM_VERSION = "tectonics-1"
#: Seed -> stream derivation and raw number generation scheme.
PRNG_VERSION = "sha256-pcg64-raw-1"
#: On-disk project layout.
SCHEMA_VERSION = "hexworld-project-4"
#: Schemas this build can open.
SUPPORTED_SCHEMAS = (
    "hexworld-project-1",
    "hexworld-project-2",
    "hexworld-project-3",
    "hexworld-project-4",
)
#: Bumped whenever v2.0 stages change their output for the same inputs.
CONTINENTS_ALGORITHM_VERSION = "continents-1"
#: World coordinate / hex conventions (see docs/DATA_FORMAT.md).
COORDINATE_VERSION = "km-x-east-y-south/flat-top-odd-q-1"
#: Bumped whenever v3.0 history / volcanism / erosion change their output.
HISTORY_ALGORITHM_VERSION = "geohistory-6"
#: Bumped whenever the mountain plan or the reference height backend change.
MOUNTAINS_ALGORITHM_VERSION = "mountains-3"
#: v4.0 climate, hydrology and landscape evolution.
CLIMATE_ALGORITHM_VERSION = "climate-4"
HYDRO_ALGORITHM_VERSION = "hydro-3"
ENV_ALGORITHM_VERSION = "environment-1"
#: v4.6 geomorphology: sea floor, basins, water connectivity, massif graphs.
TERRAIN_ALGORITHM_VERSION = "terrain-3"
#: v4.7 topoclimate: terrain units, local climate, growing season, thermal belts.
TOPOCLIMATE_ALGORITHM_VERSION = "topoclimate-2"
#: v5.0 biomes: registry, PFT, cryosphere, causal classes, fractions.
BIOMES_ALGORITHM_VERSION = "hexworld-biomes-4"
#: v5.1 volcanism: provinces, reservoirs, eruptions, lava / tephra / PDC / lahars, ecology.
VOLCANISM_ALGORITHM_VERSION = "hexworld-volcanism-1"
#: v5.3 Earth system: planet, winds aloft, ocean circulation and heat, sea ice, FAO-56.
EARTH_ALGORITHM_VERSION = "hexworld-earth-2"
#: v5.4 atmosphere: monthly circulation normals, moisture transport, ITCZ, monsoons, storm
#: climatology, synthetic weather year.
ATMOSPHERE_ALGORITHM_VERSION = "hexworld-atmosphere-5.4.1"
