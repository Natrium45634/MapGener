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
