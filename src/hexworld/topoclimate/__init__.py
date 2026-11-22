"""v4.7 topoclimate: sub-hex terrain units, altitudinal belts and local ecological conditions.

TZ 4.7. The coarse climate (v4 / v4.5) is downscaled onto terrain units (TU) inside every
hex - weighted sets of real sample points of the sub-hex surface - with an explicit
reference height, so that nothing is corrected twice. Pure physics lives in
``atmosphere``, ``solar``, ``redistribution`` and ``thermal``; ``units`` builds the samples
and the units; :mod:`hexworld.pipeline_v47` runs the stage.
"""
