"""Programmatic use of the HexWorld core (no GUI needed)."""

from dataclasses import replace

from hexworld import Config, generate, load_project, save_project
from hexworld.config import MotionParams

config = Config(seed="Ab12", width=200, height=150, hex_spacing_km=40.0)
project = generate(config, progress=lambda stage, f: print(f"{100 * f:5.1f}% {stage}"))

print(project.world_id, project.validation["ok"])
print(project.layer("plate_id").shape, project.layer("uplift").dtype)
print(project.hex_info(col=20, row=30))
print(project.plate(0)["name"], project.plate(0)["velocity_mm_yr"])

# Geology at arbitrary world coordinates (km) - independent of the hex grid:
print(project.sample_world([1000.0, 2500.0], [800.0, 1200.0]))

# The same world sampled on a finer grid over a sub-rectangle:
layout, layers = project.sample_region([1000, 1000, 3000, 2500], hex_spacing_km=5.0)
print(layout.width, layout.height, layers["crust_type"].mean())

# Recompute only motion and everything after it; plates and crust are reused:
faster = generate(replace(config, motion=MotionParams(speed_scale=1.5)), reuse=project)
print(faster.stats["reused_stages"])

save_project(project, "out/api_example")
print(load_project("out/api_example").geology_hash() == project.geology_hash())

# ---- v2.0: continents, macro relief, sea level --------------------------------
from hexworld import derive_landmasses, get_layer  # noqa: E402

geo = project.geo
print(geo.sea_level_m, len(geo.blocks), len(geo.landmasses))
print(project.layer("macro_elevation_m").dtype, project.layer("land_mask").mean())
print(geo.landmasses[0]["name"], geo.landmasses[0]["area_km2"])

# Another sea level: only land, landmasses and shelf are recomputed.
flooded = derive_landmasses(project, sea_level_m=geo.sea_level_m + 150.0)
print(flooded.stats["reused_stages"], flooded.layer("land_mask").mean())

# The same heights through another window at a finer resolution:
layout, elev = get_layer(project, "macro_elevation_m", [1000, 1000, 3000, 2500], hex_spacing_km=5.0)
print(layout.width, layout.height, float(elev.min()), float(elev.max()))
