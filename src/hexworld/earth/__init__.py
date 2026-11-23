"""v5.3 Earth system: planet, winds aloft, ocean circulation and heat, sea ice, FAO-56.

The models run on the world climate raster (the grid of the v4 climate, which covers the
whole model area) inside the climate stage, and their results are sampled to the hexes.
They are reduced-order, explainable models with closed budgets, not a GCM: see
``docs/v5_3_architecture.md`` and the quality flags of every field.
"""

ALGORITHM_VERSION = "hexworld-earth-1"
