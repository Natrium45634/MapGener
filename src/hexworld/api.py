"""Short programmatic API of the v2.0 stages.

The pipeline entry point stays :func:`hexworld.generate`; these functions expose
the individual steps for scripts and for future modules (e.g. a PMGE adapter)::

    world = load_project("v1_project")                 # or generate(config, through="v1")
    continents = generate_continents(world, config)
    relief = generate_macro_relief(world, continents, config)
    project = derive_landmasses(world, sea_level_m=0)  # full project with land / sea
    layout, values = get_layer(project, "macro_elevation_m", bounds, hex_spacing_km=5)
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np

from .config import Config
from .geometry import Bounds, HexLayout
from .pipeline import generate
from .project import Project
from .stages.continents import ContinentModel, build_continents
from .stages.relief import ReliefModel, build_relief
from .stages.sea import label_mask
from .stages.skeleton import WorldSkeleton, build_skeleton


@dataclass
class ContinentsResult:
    model: ContinentModel
    skeleton: WorldSkeleton
    #: Continentality (km) on the world skeleton raster.
    world_continentality_km: np.ndarray


@dataclass
class ReliefResult:
    model: ReliefModel
    skeleton: WorldSkeleton
    #: Macro-elevation (m) on the world skeleton raster.
    world_elevation_m: np.ndarray


def _v2_config(v1_world: Project, config: Config | None) -> Config:
    """``v1_world``'s own config with the v2 sections of ``config`` (v1 part is immutable)."""
    if config is None:
        return v1_world.config
    return replace(
        v1_world.config, continents=config.continents, relief=config.relief, sea=config.sea
    )


def _require_models(v1_world: Project) -> None:
    if v1_world.plate_model is None or v1_world.crust_model is None:
        raise RuntimeError(
            "This project's v1.0 world models are unavailable (saved by an incompatible "
            "generator version); the v2.0 stages cannot run on it."
        )


def generate_continents(v1_world: Project, config: Config | None = None) -> ContinentsResult:
    """Continental blocks from the existing plates, crust prior and plate motion."""
    _require_models(v1_world)
    cfg = _v2_config(v1_world, config)
    skeleton = build_skeleton(
        v1_world.geometry,
        v1_world.plate_model,
        v1_world.motion,
        cfg.boundaries,
        cfg.continents.skeleton_cells,
    )
    model, world_c = build_continents(cfg, v1_world.crust_model, skeleton)
    return ContinentsResult(model, skeleton, world_c)


def generate_macro_relief(
    v1_world: Project, continents: ContinentsResult, config: Config | None = None
) -> ReliefResult:
    """One continuous macro-elevation field (metres) for the given continents."""
    cfg = _v2_config(v1_world, config)
    model = build_relief(cfg, continents.model, continents.skeleton)
    sk = continents.skeleton
    world = model.elevation(sk.x, sk.y, continentality=continents.world_continentality_km)
    return ReliefResult(model, sk, world)


def generate_v2(v1_world: Project, config: Config | None = None, progress=None) -> Project:
    """Add (or update) the v2.0 data of a project; its v1.0 arrays are left as they are."""
    return generate(_v2_config(v1_world, config), progress=progress, reuse=v1_world)


def derive_landmasses(
    project: Project, sea_level_m: float | None = None, target_land_fraction: float | None = None
) -> Project:
    """Land mask, landmasses and shelf for another sea level (nothing else is recomputed).

    Give exactly one of ``sea_level_m`` and ``target_land_fraction``.
    """
    if (sea_level_m is None) == (target_land_fraction is None):
        raise ValueError("give exactly one of sea_level_m and target_land_fraction")
    sea = project.config.sea
    if sea_level_m is not None:
        sea = replace(sea, sea_mode="fixed_sea_level_m", fixed_sea_level_m=float(sea_level_m))
    else:
        sea = replace(
            sea, sea_mode="target_land_fraction", target_land_fraction=float(target_land_fraction)
        )
    return generate(replace(project.config, sea=sea), reuse=project)


def _relief_model(project: Project) -> ReliefModel:
    entry = project.cache.get("macro_elevation")
    if entry is not None and entry.get("model") is not None:
        return entry["model"]
    continents = generate_continents(project)
    relief = generate_macro_relief(project, continents)
    project.cache["macro_elevation_model"] = relief.model
    return relief.model


def get_layer(
    project: Project,
    layer_id: str,
    bounds: Bounds | list[float] | None = None,
    hex_spacing_km: float | None = None,
):
    """A layer for a world rectangle at any resolution: ``(layout, array)``.

    Without ``bounds`` and ``hex_spacing_km`` the stored hex layer is returned.
    Otherwise the layer is sampled from the world models on a fresh hex grid, so
    the same world can be read through any window; id layers are re-labelled on
    that grid (their numbering is local to it).
    """
    if bounds is None and hex_spacing_km is None:
        return project.layout, project.layer(layer_id)
    b = (
        project.geometry.viewport_bounds
        if bounds is None
        else (bounds if isinstance(bounds, Bounds) else Bounds(*bounds))
    )
    spacing = float(hex_spacing_km or project.layout.spacing_km)
    radius = spacing / np.sqrt(3.0)
    width = max(2, int((b.width / radius - 2.0) / 1.5) + 1)
    height = max(2, int(b.height / spacing - 0.5))
    layout = HexLayout(width, height, spacing, b.xmin, b.ymin)
    x, y = layout.centers()
    v1 = ("plate_id", "crust_type", "continentality_km", "velocity_x", "velocity_y")
    if layer_id in v1:
        return layout, project.sample_world(x, y)[layer_id]
    if project.geo is None:
        raise KeyError(f"Layer {layer_id!r} needs the v2.0 stage, which this project lacks")
    relief = project.cache.get("macro_elevation_model") or _relief_model(project)
    c = relief.continents.continentality(x, y)
    fraction = relief.continents.fraction_of(c).astype(np.float32)
    if layer_id == "continental_fraction":
        return layout, fraction
    if layer_id == "continental_margin_km":
        return layout, c.astype(np.float32)
    if layer_id == "continental_block_id":
        return layout, label_mask(fraction >= 0.5, layout)[0]
    elev, tect = relief.elevation(x, y, continentality=c, parts=True)
    elev = elev.astype(np.float32)
    sea = np.float32(project.geo.sea_level_m)
    if layer_id == "macro_elevation_m":
        return layout, elev
    if layer_id == "tectonic_relief_m":
        return layout, tect.astype(np.float32)
    land = elev > sea
    if layer_id == "land_mask":
        return layout, land
    if layer_id == "landmass_id":
        return layout, label_mask(land, layout)[0]
    if layer_id == "shelf_mask":
        depth = sea - elev
        limit = project.config.sea.shelf_max_depth_m
        return layout, ~land & (fraction >= 0.5) & (depth <= limit)
    raise KeyError(
        f"Layer {layer_id!r} cannot be resampled; available hex layers: {project.layer_names}"
    )
