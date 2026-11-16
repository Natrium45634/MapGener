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
    return generate(
        _v2_config(v1_world, config),
        progress=progress,
        reuse=v1_world,
        through=max(stage_depth(v1_world), "v2"),
    )


def generate_history(project: Project, config: Config | None = None, progress=None) -> Project:
    """Add (or update) the v3.0 geological history; the v1/v2 baseline is only read."""
    cfg = (
        project.config
        if config is None
        else replace(
            project.config,
            continents=config.continents,
            relief=config.relief,
            sea=config.sea,
            history=config.history,
            mountains=config.mountains,
        )
    )
    return generate(cfg, progress=progress, reuse=project, through=max(stage_depth(project), "v3"))


def select_epoch(project: Project, epoch, progress=None) -> Project:
    """Show another stored snapshot: ``"initial"``, ``"final"`` or a time in Ma.

    The history is not simulated again; only the hex layers of that epoch are built.
    """
    history = replace(project.config.history, epoch=epoch)
    return generate(
        replace(project.config, history=history),
        progress=progress,
        reuse=project,
        through=max(stage_depth(project), "v3"),
    )


def history_events(project: Project, time_range=None, region=None, kinds=None) -> list[dict]:
    """Events of the history, optionally filtered by time (Ma), bounds (km) and type."""
    if project.hist is None:
        raise LookupError("this project has no v3.0 geological history")
    out = []
    for e in project.hist.events:
        if time_range is not None and not time_range[0] <= e["time_ma"] <= time_range[1]:
            continue
        if region is not None and not (
            region[0] <= e["x_km"] <= region[2] and region[1] <= e["y_km"] <= region[3]
        ):
            continue
        if kinds is not None and e["type"] not in kinds:
            continue
        out.append(e)
    return out


def sample_points(project: Project, x, y, fields=("elevation_m",)) -> dict:
    """Layer values at world points inside the window (value of the containing hex).

    Points outside the window give NaN (or -1 for integer layers). For another
    window or resolution use :func:`get_layer`.
    """
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    col, row = project.layout.world_to_offset(x, y)
    inside = project.layout.inside(col, row)
    cc, rr = np.where(inside, col, 0), np.where(inside, row, 0)
    out = {"inside": inside}
    for name in fields:
        arr = project.layer(name)
        values = arr[rr, cc]
        if np.issubdtype(arr.dtype, np.floating):
            out[name] = np.where(inside, values, np.nan)
        elif arr.dtype == np.bool_:
            out[name] = np.where(inside, values, False)
        else:
            out[name] = np.where(inside, values.astype(np.int64), -1)
    return out


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
    # a sea level needs the v2 relief; deeper projects keep their depth (v3, v4 recomputed)
    through = max(stage_depth(project), "v2")
    return generate(replace(project.config, sea=sea), reuse=project, through=through)


def stage_depth(project: Project) -> str:
    """The last stage a project holds ("v1".."v4"): re-derivations keep that depth."""
    if project.hydro is not None:
        return "v4"
    if project.hist is not None:
        return "v3"
    return "v2" if project.geo is not None else "v1"


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


# -- v4.0: climate, hydrology, erosion ------------------------------------------------------
def generate_v4(project: Project, config: Config | None = None, progress=None) -> Project:
    """Add (or update) climate, hydrology and erosion; v1-v3 are reused as they are."""
    cfg = (
        project.config
        if config is None
        else replace(
            project.config,
            climate=config.climate,
            hydrology=config.hydrology,
            erosion=config.erosion,
        )
    )
    return generate(cfg, progress=progress, reuse=project, through="v4")


def generate_climate(project: Project, config: Config | None = None):
    """Climate of the selected v3 epoch surface (before v4 erosion).

    Returns ``(world_climate, hex_layers)``: the coarse world solution and the hex fields.
    """
    from .stages.climate import hex_climate, resolve_latitude, world_climate

    if project.hist is None:
        raise LookupError("the climate needs the v3.0 epoch surface; run the history first")
    cfg = project.config if config is None else config
    hist = project.hist
    latitude = resolve_latitude(cfg.climate, project.geometry.simulation_bounds)
    wc = world_climate(
        cfg.climate, hist.world["grid"], hist.world["elevation_m"], hist.sea_level_m, latitude
    )
    return wc, hex_climate(
        cfg.climate, wc, project.layout, hist.layers["elevation_m"], hist.sea_level_m
    )


def build_hydrology(
    project: Project, elevation_m=None, climate_layers=None, config: Config | None = None
) -> dict:
    """Routing, lakes and discharge for a surface (default: the final v4 or v3 surface)."""
    from .stages.hydrology import hex_graph, hydrology, routing_choice

    cfg = project.config if config is None else config
    z = np.asarray(
        project.layer("elevation_m") if elevation_m is None else elevation_m, dtype=np.float64
    )
    sea_level = project.current_sea_level_m
    if climate_layers is None:
        climate_layers = generate_climate(project, cfg)[1]
    layout = project.layout
    return hydrology(
        z.ravel(),
        (z <= sea_level).ravel(),
        hex_graph(layout),
        layout.hex_area_km2 * 1e6,
        climate_layers["runoff_mm_yr"].ravel(),
        climate_layers["precipitation_mm_yr"].ravel(),
        climate_layers["potential_evapotranspiration_mm_yr"].ravel() * 1.1,
        None,
        cfg.hydrology.min_lake_depth_m,
        cfg.hydrology.channel_loss,
        routing_choice(cfg.seed, z.size, cfg.hydrology.routing),
    )


def erode(project: Project, config: Config | None = None, progress=None) -> dict:
    """Run only the landscape evolution on the v3 epoch surface; returns the raw result."""
    from .stages.hydrology import hex_graph
    from .stages.landscape import evolve

    cfg = project.config if config is None else config
    hist = project.hist
    if hist is None:
        raise LookupError("erosion needs the v3.0 epoch surface")
    wc, _ = generate_climate(project, cfg)
    return evolve(
        cfg,
        project.layout,
        hex_graph(project.layout),
        hist.layers["elevation_m"],
        hist.sea_level_m,
        hist.layers["rock_resistance"],
        hist.layers["sediment_thickness_m"],
        hist.layers["uplift_rate_mm_yr"],
        wc,
        None,
        report=progress,
    )
