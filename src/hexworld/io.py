"""Project persistence. Format is documented in docs/DATA_FORMAT.md.

A project is a directory::

    config.json      seed, sizes, physical scale, mode, all stage parameters
    plates.json      plate metadata + the world-model description
    fields.npz       typed hex layers [row, col]
    boundaries.npz   boundary-edge graph (structure of arrays)
    boundaries.json  schema of boundaries.npz + per plate-pair summary
    manifest.json    versions, coordinate conventions, units, control hashes

PNG images are never read back: they are pictures, not data.
"""

from __future__ import annotations

import hashlib
import json
import platform
from pathlib import Path

import numpy as np
import scipy

from .config import Config
from .geometry import resolve_geometry
from .pipeline import _pad_hexes, _reference_stats, stage_keys, world_id_from_keys
from .pipeline_v2 import v2_stage_keys
from .pipeline_v3 import _state_from_arrays, v3_stage_keys
from .project import (
    GEO_LAYER_SPECS,
    HIST_LAYER_SPECS,
    HYDRO_LAYER_SPECS,
    LAYER_SPECS,
    GeoData,
    HistoryData,
    HydroData,
    Project,
)
from .stages.boundaries import TYPE_NAMES, BoundaryEdges
from .stages.crust import CRUST_NAMES, build_crust_model
from .stages.motion import PlateMotion
from .stages.plates import build_plate_model
from .stages.reference import build_reference_grid
from .stages.validate import validate_geo, validate_history, validate_hydro, validate_project
from .version import (
    ALGORITHM_VERSION,
    BIOMES_ALGORITHM_VERSION,
    CLIMATE_ALGORITHM_VERSION,
    CONTINENTS_ALGORITHM_VERSION,
    COORDINATE_VERSION,
    EARTH_ALGORITHM_VERSION,
    ENV_ALGORITHM_VERSION,
    GENERATOR_VERSION,
    HISTORY_ALGORITHM_VERSION,
    HYDRO_ALGORITHM_VERSION,
    MOUNTAINS_ALGORITHM_VERSION,
    PRNG_VERSION,
    SCHEMA_VERSION,
    SUPPORTED_SCHEMAS,
    TERRAIN_ALGORITHM_VERSION,
    TOPOCLIMATE_ALGORITHM_VERSION,
    VOLCANISM_ALGORITHM_VERSION,
)

DATA_FILES = ("config.json", "plates.json", "fields.npz", "boundaries.npz", "boundaries.json")
#: v1.0 source data that a later stage must never rewrite.
V1_DATA_FILES = ("plates.json", "fields.npz", "boundaries.npz", "boundaries.json")
#: v2.0 data lives in its own files.
V2_DATA_FILES = ("continental_fields.npz", "continents.json")
#: v3.0 data: selected epoch, checkpoints, event log, mountain plan.
V4_DATA_FILES = ("hydro_fields.npz", "hydrology.json", "rivers.geojson")
V45_DATA_FILES = ("environment_fields.npz", "environment_monthly.npz", "environment.json")
V46_DATA_FILES = ("terrain_fields.npz", "terrain.json")
V50_DATA_FILES = (
    "biomes_fields.npz",
    "biomes_units.npz",
    "biomes_fractions.npz",
    "biomes.json",
)
V51_DATA_FILES = ("volcanism_fields.npz", "volcanism.json")
V53_DATA_FILES = ("earth_fields.npz", "earth.json")
V54_DATA_FILES = ("atmosphere_fields.npz", "atmosphere_grid.npz", "atmosphere.json")
V52_DATA_FILES = ("resources_fields.npz", "resources_index.npz", "resources.json")
V551_DATA_FILES = ("minerals.json", "minerals_fields.npz")
V47_DATA_FILES = (
    "topoclimate_fields.npz",
    "topoclimate_monthly.npz",
    "topoclimate_units.npz",
    "topoclimate.json",
)
V3_DATA_FILES = (
    "history_fields.npz",
    "history.json",
    "events.json",
    "checkpoints.npz",
    "mountain_plan.json",
)

BOUNDARY_SCHEMA = {
    "col_a/row_a": "int32 - offset coords of the hex on the plate_a side",
    "col_b/row_b": "int32 - offset coords of the hex on the plate_b side",
    "direction": "uint8 - hex direction index A->B, 0..5 = axial (1,0),(1,-1),(0,-1),(-1,0),"
    "(-1,1),(0,1)",
    "plate_a/plate_b": "int32 - plate IDs, always plate_a < plate_b",
    "mid_x/mid_y": "float64 km - world coordinates of the edge midpoint",
    "normal_x/normal_y": "float32 - unit normal of the geological boundary, from A into B",
    "vrel_x/vrel_y": "float32 mm/yr - relative velocity v_B - v_A at the midpoint",
    "vn": "float32 mm/yr - v_rel . normal; < 0 convergent, > 0 divergent",
    "vt": "float32 mm/yr - v_rel . tangent, tangent = (-ny, nx); > 0 dextral, < 0 sinistral",
    "boundary_type": f"uint8 - {dict(enumerate(TYPE_NAMES))}",
    "intensity": "float32 0..1 - dominant component / boundaries.reference_speed_mm_yr",
    "crust_a/crust_b": f"uint8 - {dict(enumerate(CRUST_NAMES))}",
    "subducting": "uint8 - polarity hint for convergent edges: 0 none, 1 plate_a, 2 plate_b",
}


class ProjectIntegrityError(RuntimeError):
    pass


def _file_hash(path: Path) -> str:
    """SHA-256 of a file. Text files (JSON) are hashed with LF line ends, so a project that
    went through a tool converting to CRLF (git on Windows, some editors) still verifies;
    for files written by HexWorld (always LF) the value is unchanged."""
    h = hashlib.sha256()
    if path.suffix in (".json", ".geojson"):
        h.update(path.read_bytes().replace(b"\r\n", b"\n"))
        return h.hexdigest()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _write_json(path: Path, data) -> None:
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _write_v1_data(project: Project, path: Path) -> None:
    _write_json(
        path / "plates.json",
        {
            "world_id": project.world_id,
            "units": {"length": "km", "area": "km2", "velocity": "mm/yr", "rotation": "deg/Myr"},
            "plate_model": project.plate_model.to_dict() if project.plate_model else None,
            "crust_model": project.crust_model.to_dict() if project.crust_model else None,
            "plates": project.plates,
        },
    )
    np.savez_compressed(path / "fields.npz", **project.layers)
    np.savez_compressed(path / "boundaries.npz", **project.boundaries.arrays())
    _write_json(
        path / "boundaries.json",
        {
            "format": "structure of arrays in boundaries.npz; one entry per shared hex edge",
            "edge_count": len(project.boundaries),
            "edge_length_km": project.boundaries.length_km,
            "arrays": BOUNDARY_SCHEMA,
            "pairs": project.boundary_summary,
        },
    )


def _write_v2_data(project: Project, path: Path) -> None:
    geo = project.geo
    np.savez_compressed(
        path / "continental_fields.npz",
        **geo.layers,
        world_elevation_m=geo.world["elevation_m"],
        world_continentality_km=geo.world["continentality_km"],
    )
    _write_json(
        path / "continents.json",
        {
            "world_id": project.world_id,
            "algorithm_version": CONTINENTS_ALGORITHM_VERSION,
            "units": {"length": "km", "area": "km2", "elevation": "m"},
            "sea_level_m": geo.sea_level_m,
            "summary": geo.summary,
            "world_grid": geo.world["grid"],
            "stage_keys": geo.stage_keys,
            "warnings": geo.warnings,
            "validation": geo.validation,
            "timings_s": geo.timings,
            "blocks": geo.blocks,
            "landmasses": geo.landmasses,
        },
    )


def _write_v4_data(project: Project, path: Path) -> None:
    h = project.hydro
    arrays = {
        k: v for k, v in h.layers.items() if k not in ("elevation_m", "pre_erosion_elevation_m")
    }  # stored once elsewhere
    np.savez_compressed(path / "hydro_fields.npz", **arrays)
    _write_json(
        path / "hydrology.json",
        {
            "world_id": project.world_id,
            "algorithm_versions": {
                "climate": CLIMATE_ALGORITHM_VERSION,
                "hydrology": HYDRO_ALGORITHM_VERSION,
            },
            "units": {
                "precipitation": "mm/yr",
                "discharge": "m3/s",
                "elevation": "m",
                "area": "km2",
                "length": "km",
                "sediment": "m3",
                "erosion_rate": "m/Ma",
            },
            "provenance": {
                "input_surface": f"v3 elevation_m of epoch t = {h.epoch_time_ma:g} Ma",
                "elevation_m": "post_erosion_elevation_m (v4 final surface)",
                "pre_erosion_elevation_m": "terrain_fields.npz -> surface_elevation_m (v4.6) or "
                "history_fields.npz -> elevation_m (reference)",
            },
            "sea_level_m": h.sea_level_m,
            "epoch_time_ma": h.epoch_time_ma,
            "latitude": h.latitude,
            "summary": h.summary,
            "stage_keys": h.stage_keys,
            "sediment_budget_m3": h.budget,
            "evolution_steps": h.steps,
            "warnings": h.warnings,
            "validation": h.validation,
            "timings_s": h.timings,
            "river_systems": h.river_systems,
            "river_segments": h.rivers,
            "lakes": h.lakes,
            "basins": h.basins,
            "landmasses": h.landmasses,
        },
    )
    _write_json(path / "rivers.geojson", rivers_geojson(project))


def rivers_geojson(project: Project) -> dict:
    """River segments as a GeoJSON FeatureCollection in HexWorld world km (not lon/lat)."""
    features = []
    for seg in project.hydro.rivers:
        features.append(
            {
                "type": "Feature",
                "geometry": {
                    "type": "LineString",
                    "coordinates": [
                        [round(x, 3), round(y, 3)]
                        for x, y in zip(seg["x_km"], seg["y_km"], strict=True)
                    ],
                },
                "properties": {
                    k: seg[k]
                    for k in (
                        "id",
                        "strahler",
                        "discharge_m3_s",
                        "length_km",
                        "drainage_area_km2",
                        "mouth",
                        "next_segment",
                        "basin_id",
                    )
                },
            }
        )
    return {
        "type": "FeatureCollection",
        "crs_note": "HexWorld world coordinates in km (+x east, +y south), not geographic",
        "units": {"coordinates": "km", "discharge_m3_s": "m3/s", "length_km": "km"},
        "features": features,
    }


def _write_v3_data(project: Project, path: Path) -> None:
    from .mountains.plan import MountainPlan  # noqa: F401  (documented dependency)

    hist = project.hist
    world = {f"world_{k}": v for k, v in hist.world.items() if k != "grid"}
    np.savez_compressed(path / "history_fields.npz", **hist.layers, **world)
    times = sorted(hist.checkpoints)
    packed = {}
    for i, t in enumerate(times):
        for name, arr in hist.checkpoints[t].items():
            packed[f"c{i}_{name}"] = arr
    np.savez_compressed(path / "checkpoints.npz", **packed)
    entry = project.cache.get("geo_history") or {}
    volcano_set = entry.get("volcanoes")
    _write_json(
        path / "events.json",
        {
            "units": {
                "time": "Ma since the baseline (t = 0)",
                "position": "km, world frame",
                "intensity": "mm/yr relative plate speed",
            },
            "note": "events with derived = true are recomputed for the selected epoch",
            "events": hist.events,
        },
    )
    _write_json(path / "mountain_plan.json", hist.plan)
    _write_json(
        path / "history.json",
        {
            "world_id": project.world_id,
            "algorithm_version": HISTORY_ALGORITHM_VERSION,
            "mountains_algorithm_version": MOUNTAINS_ALGORITHM_VERSION,
            "units": {
                "time": "Ma",
                "length": "km",
                "elevation": "m",
                "plate speed": "mm/yr",
                "vertical rate": "mm/yr",
                "identity": "1 mm/yr = 1 km/Ma",
            },
            "epoch_time_ma": hist.epoch_time_ma,
            "epochs_ma": hist.epochs,
            "checkpoint_times_ma": times,
            "duration_ma": hist.duration_ma,
            "sea_level_m": hist.sea_level_m,
            "world_grid": hist.world.get("grid"),
            "summary": hist.summary,
            "stage_keys": hist.stage_keys,
            "warnings": hist.warnings,
            "validation": hist.validation,
            "timings_s": hist.timings,
            "erosion_balance_per_step": hist.balance,
            "run_info": entry.get("info", {}),
            "hotspots": hist.hotspots,
            "hotspot_state": [vars(h) for h in volcano_set.hotspots] if volcano_set else [],
            "volcano_records": [v.to_dict() for v in volcano_set.volcanoes] if volcano_set else [],
            "volcanoes": hist.volcanoes,
            "landmasses": hist.landmasses,
        },
    )


def _env_hash(env) -> str:
    h = hashlib.sha256()
    for key, value in sorted(env.content_hashes().items()):
        h.update(f"{key}={value};".encode())
    return h.hexdigest()


def _env_layer_table(env) -> dict:
    from .pipeline_v45 import ENV_LAYER_SPECS, MONTHLY_FIELDS

    out = {
        name: {"dtype": str(a.dtype), "shape": list(a.shape), "unit": ENV_LAYER_SPECS[name][0],
               "description": ENV_LAYER_SPECS[name][1]}
        for name, a in env.layers.items()
    }  # fmt: skip
    for name, a in env.monthly.items():
        unit, text = MONTHLY_FIELDS[name]
        out[name] = {"dtype": str(a.dtype), "shape": list(a.shape), "unit": unit,
                     "description": text, "layout": "month-major (12, rows, cols)"}  # fmt: skip
    return out


def _write_v45_data(project: Project, path: Path) -> None:
    from .pipeline_v45 import MONTHLY_FIELDS

    e = project.env
    np.savez_compressed(path / "environment_fields.npz", **e.layers)
    # month-major arrays (12, rows, cols), float32; NoData = NaN outside land (see the mask)
    np.savez_compressed(path / "environment_monthly.npz", **e.monthly)
    _write_json(
        path / "environment.json",
        {
            "world_id": project.world_id,
            "stage": "ecological_foundation_4_5",
            "algorithm_version": ENV_ALGORITHM_VERSION,
            "source_snapshot": "v4 final (after erosion)",
            "calendar": e.summary.get("calendar"),
            "month_index": "0 = January .. 11 = December",
            "monthly_fields": {
                k: {"unit": u, "description": d} for k, (u, d) in MONTHLY_FIELDS.items()
            },
            "methods": e.methods,
            "summary": e.summary,
            "stage_keys": e.stage_keys,
            "warnings": e.warnings,
            "timings_s": e.timings,
            "nodata": "NaN in float fields outside valid cells; soil_present / land_mask masks",
        },
    )


def _topo_hash(topo) -> str:
    h = hashlib.sha256()
    for key, value in sorted(topo.content_hashes().items()):
        h.update(f"{key}={value};".encode())
    return h.hexdigest()


def _write_v47_data(project: Project, path: Path) -> None:
    from .pipeline_v47 import HEX_MONTHLY, TOPO_LAYER_SPECS, UNIT_MONTHLY
    from .topoclimate.thermal import BELT_TITLES, BELTS
    from .topoclimate.units import LANDFORMS, SURFACE

    t = project.topo
    np.savez_compressed(path / "topoclimate_fields.npz", **t.layers)
    np.savez_compressed(path / "topoclimate_monthly.npz", **t.monthly)
    np.savez_compressed(path / "topoclimate_units.npz", **t.units)
    _write_json(
        path / "topoclimate.json",
        {
            "world_id": project.world_id,
            "stage": "topoclimate_4_7",
            "algorithm_version": TOPOCLIMATE_ALGORITHM_VERSION,
            "source_snapshot": "v4.6 surface + v4 / v4.5 of the same surface revision",
            "month_index": "0 = January .. 11 = December",
            "units_layout": "CSR: units of hex h (= row * width + col) are "
            "hex_offsets[h] .. hex_offsets[h + 1]; monthly unit fields are (12, U), "
            "horizon_rad is (U, K) with K azimuth sectors clockwise from north",
            "unit_monthly_fields": {k: {"unit": u, "description": d}
                                    for k, (u, d) in UNIT_MONTHLY.items()},
            "hex_monthly_fields": {k: {"unit": u, "description": d}
                                   for k, (u, d) in HEX_MONTHLY.items()},
            "layers": {k: {"unit": u, "description": d} for k, (u, d) in TOPO_LAYER_SPECS.items()},
            "belts": list(BELTS),
            "belt_titles": BELT_TITLES,
            "landforms": list(LANDFORMS),
            "surface_types": list(SURFACE),
            "massifs": t.massifs,
            "provenance": t.provenance,
            "summary": t.summary,
            "stage_keys": t.stage_keys,
            "warnings": t.warnings,
            "timings_s": t.timings,
            "nodata": "NaN in float fields where a hex has no dry land / no transition",
        },
    )  # fmt: skip


def _biome_hash(bio) -> str:
    h = hashlib.sha256()
    for key, value in sorted(bio.content_hashes().items()):
        h.update(f"{key}={value};".encode())
    return h.hexdigest()


def _write_v50_data(project: Project, path: Path) -> None:
    from .biomes import pft
    from .biomes.registry import (
        BIOMES,
        COVER_CLASSES,
        LIMITING_FACTORS,
        MODIFIERS,
        PFT_REGISTRY_VERSION,
        REALMS,
        REGISTRY_VERSION,
    )
    from .pipeline_v50 import KOPPEN

    b = project.biomes
    np.savez_compressed(path / "biomes_fields.npz", **b.layers)
    np.savez_compressed(path / "biomes_units.npz", **b.units)
    np.savez_compressed(path / "biomes_fractions.npz", **b.fractions)
    _write_json(
        path / "biomes.json",
        {
            "world_id": project.world_id,
            "stage": "biomes_5_0",
            "schema_version": "5.0",
            "algorithm_version": BIOMES_ALGORITHM_VERSION,
            "biome_registry_version": REGISTRY_VERSION,
            "pft_registry_version": PFT_REGISTRY_VERSION,
            "biome_registry": [
                {"code": i, "id": c.id, "family": c.family, "ru": c.ru, "en": c.en,
                 "rgb": list(c.rgb)} for i, c in enumerate(BIOMES)
            ],
            "pft_registry": [t.__dict__ for t in pft.TRAITS],
            "cover_classes": list(COVER_CLASSES),
            "realms": list(REALMS),
            "koppen": list(KOPPEN),
            "limiting_factor_bits": list(LIMITING_FACTORS),
            "modifier_bits": list(MODIFIERS),
            "fractions_layout": "CSR whole-hex fractions: hex h = row * width + col owns "
            "biome[offsets[h]:offsets[h+1]] with fraction[...] (sum = 1 per hex)",
            "summary": b.summary,
            "provenance": b.provenance,
            "seed": project.config.seed,
            "stage_keys": b.stage_keys,
            "warnings": b.warnings,
            "timings_s": b.timings,
            "nodata": "not generated = no biomes section (never zero vegetation)",
        },
    )  # fmt: skip


def _volc_hash(volc) -> str:
    h = hashlib.sha256()
    for key, value in sorted(volc.content_hashes().items()):
        h.update(f"{key}={value};".encode())
    h.update(json.dumps(volc.events, sort_keys=True, default=str).encode())
    return h.hexdigest()


def _plain(obj):
    """JSON-safe copy: numpy scalars to Python, tuples to lists, non-finite floats to None."""
    if isinstance(obj, dict):
        return {str(k): _plain(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_plain(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return _plain(obj.tolist())
    if isinstance(obj, (np.bool_, bool)):
        return bool(obj)
    if isinstance(obj, (np.integer, int)):
        return int(obj)
    if isinstance(obj, (np.floating, float)):
        f = float(obj)
        return f if np.isfinite(f) else None
    return obj


def _write_v51_data(project: Project, path: Path) -> None:
    from .pipeline_v51 import HYDROTHERMAL_KINDS

    v = project.volcanism
    np.savez_compressed(path / "volcanism_fields.npz", **v.layers)
    _write_json(
        path / "volcanism.json",
        _plain({
            "world_id": project.world_id,
            "stage": "volcanism_5_1",
            "schema_version": "5.1",
            "algorithm_version": VOLCANISM_ALGORITHM_VERSION,
            "units": {"time": "years before present (events), Ma (provinces)",
                      "mass": "kg", "volume": "m^3 (DRE unless *_bulk_*)", "load": "kg/m^2"},
            "hydrothermal_kinds": list(HYDROTHERMAL_KINDS),
            "volcano_status_codes": ["none", "extinct_system", "uncertain", "dormant_system",
                                     "active_system"],
            "provinces": v.provinces,
            "systems": v.systems,
            "events": v.events,
            "aggregates": v.aggregates,
            "summary": v.summary,
            "provenance": v.provenance,
            "seed": project.config.seed,
            "stage_keys": v.stage_keys,
            "warnings": v.warnings,
            "timings_s": v.timings,
            "nodata": "not generated = no volcanism section (never 'no volcanoes')",
        }),
    )  # fmt: skip


def _earth_hash(earth) -> str:
    h = hashlib.sha256()
    for key, value in sorted(earth.content_hashes().items()):
        h.update(f"{key}={value};".encode())
    return h.hexdigest()


def _atm_hash(atm) -> str:
    h = hashlib.sha256()
    for key, value in sorted(atm.content_hashes().items()):
        h.update(f"{key}={value};".encode())
    h.update(json.dumps(_plain(atm.events), sort_keys=True, default=str).encode())
    return h.hexdigest()


def _write_v54_data(project: Project, path: Path) -> None:
    from .atmosphere.normals import ALGORITHM_VERSION as ATM_VERSION

    a = project.atmosphere
    np.savez_compressed(path / "atmosphere_fields.npz", **a.layers)
    np.savez_compressed(path / "atmosphere_grid.npz", **a.grid_fields)
    _write_json(
        path / "atmosphere.json",
        _plain({
            "world_id": project.world_id,
            "stage": "atmospheric_circulation_5_4",
            "schema_version": "5.4.0",
            "algorithm_version": ATM_VERSION,
            "grid": a.grid,
            "summary": a.summary,
            "provenance": a.provenance,
            "stage_keys": a.stage_keys,
            "events": a.events,
            "warnings": a.warnings,
            "timings_s": a.timings,
            "nodata": "not generated = no atmosphere section (never 'no rain' or 'no storms')",
        }),
    )  # fmt: skip


def _write_v52_data(project: Project, path: Path) -> None:
    from .resources import ALGORITHM_VERSION as RES_VERSION

    r = project.resources
    np.savez_compressed(path / "resources_fields.npz", **r.layers)
    np.savez_compressed(path / "resources_index.npz", hex_offsets=r.catalog["hex_offsets"],
                        hex_bodies=r.catalog["hex_bodies"])  # fmt: skip
    cat = {k: v for k, v in r.catalog.items() if k not in ("hex_offsets", "hex_bodies")}
    _write_json(
        path / "resources.json",
        _plain({
            "world_id": project.world_id,
            "stage": "soil_fertility_resources_5_2",
            "schema_version": "5.2.0",
            "algorithm_version": RES_VERSION,
            "catalog": cat,
            "summary": r.summary,
            "provenance": r.provenance,
            "stage_keys": r.stage_keys,
            "warnings": r.warnings,
            "timings_s": r.timings,
            "nodata": "soil fields are NaN where no soil (rock, ice, water): not 'poor soil'; "
                      "no catalogue = not generated (never 'no resources')",
            "units": "soil kg/m2, deposits t, grades mass fraction, depths m (datum)",
        }),
    )  # fmt: skip


def _write_v53_data(project: Project, path: Path) -> None:
    e = project.earth
    np.savez_compressed(path / "earth_fields.npz", **e.layers)
    _write_json(
        path / "earth.json",
        _plain({
            "world_id": project.world_id,
            "stage": "earth_5_3a",
            "algorithm_version": EARTH_ALGORITHM_VERSION,
            "summary": e.summary,
            "provenance": e.provenance,
            "stage_keys": e.stage_keys,
            "warnings": e.warnings,
            "timings_s": e.timings,
            "nodata": "not generated = no earth section (not 'no ocean' or 'no sea ice')",
        }),
    )  # fmt: skip


def _terrain_hash(terrain) -> str:
    h = hashlib.sha256()
    for key, value in sorted(terrain.content_hashes().items()):
        h.update(f"{key}={value};".encode())
    h.update(json.dumps(terrain.plans, sort_keys=True, default=str).encode())
    return h.hexdigest()


def _write_v46_data(project: Project, path: Path) -> None:
    from .geomorph.classify import PROVINCES, TITLES
    from .project import TERRAIN_LAYER_SPECS

    t = project.terrain
    np.savez_compressed(path / "terrain_fields.npz", **t.layers)
    _write_json(
        path / "terrain.json",
        {
            "world_id": project.world_id,
            "stage": "geomorphology_4_6",
            "algorithm_version": TERRAIN_ALGORITHM_VERSION,
            "source_snapshot": f"v3 epoch t = {project.hist.epoch_time_ma:g} Ma",
            "vertical_datum": "metres, model datum 0; sea level is a scenario parameter",
            "composition": {
                "continental_crust": "keep (v3 surface) + basin deltas",
                "sea_floor": "replace v2 baseline + v3 trench by age-depth basement, margins, "
                "fabric, fracture zones, trenches, plateaus, pelagic sediment; keep residual",
            },
            "layers": {
                k: {
                    "unit": TERRAIN_LAYER_SPECS.get(k, ("", ""))[0],
                    "description": TERRAIN_LAYER_SPECS.get(k, ("", ""))[1],
                }
                for k in t.layers
            },
            "province_names": {name: TITLES[name] for name in PROVINCES},
            "plans": t.plans,
            "summary": t.summary,
            "stage_keys": t.stage_keys,
            "warnings": t.warnings,
            "timings_s": t.timings,
        },
    )


def save_project(project: Project, path: str | Path, overwrite: bool = True) -> Path:
    """Save ``project`` into directory ``path``.

    * v1.0 source data (plates, fields, boundaries) is written only when the
      directory does not already hold the very same geology - adding v2.0 data
      or re-rendering never rewrites it.
    * v2.0 data goes to its own files (``continental_fields.npz``,
      ``continents.json``).
    * v3.0 data goes to ``history_fields.npz``, ``history.json``, ``events.json``,
      ``checkpoints.npz`` and ``mountain_plan.json``.
    * When a directory of an older schema is extended, its original ``config.json``
      and ``manifest.json`` are kept as ``*.v1.json`` / ``*.v2.json`` copies.
    """
    path = Path(path)
    manifest_path = path / "manifest.json"
    geology_hash = project.geology_hash()
    existing: dict = {}
    if manifest_path.exists():
        try:
            existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            existing = {}
    same_geology = existing.get("geology_hash") == geology_hash and all(
        (path / name).exists() for name in DATA_FILES
    )
    if existing and not same_geology and not overwrite:
        raise FileExistsError(f"{path} already contains a different project")
    stored_geo = (existing.get("v2") or {}).get("geo_hash")
    stored_hist = (existing.get("v3") or {}).get("hist_hash")
    stored_hydro = (existing.get("v4") or {}).get("hydro_hash")
    hydro_same = project.hydro is not None and stored_hydro == project.hydro.hydro_hash()
    geo_same = project.geo is not None and stored_geo == project.geo.geo_hash()
    hist_same = project.hist is not None and stored_hist == project.hist.hist_hash()
    if same_geology:
        env_same = project.env is None or (existing.get("v45") or {}).get("env_hash") == (
            _env_hash(project.env)
        )
        terrain_now = getattr(project, "terrain", None)
        env_same = env_same and (
            terrain_now is None
            or (existing.get("v46") or {}).get("terrain_hash") == _terrain_hash(terrain_now)
        )
        bio_now = getattr(project, "biomes", None)
        env_same = env_same and (
            (bio_now is None and not existing.get("v50"))
            or (bio_now is not None
                and (existing.get("v50") or {}).get("biome_hash") == _biome_hash(bio_now))
        )  # fmt: skip
        earth_now = getattr(project, "earth", None)
        env_same = env_same and (
            (earth_now is None and not existing.get("v53"))
            or (earth_now is not None
                and (existing.get("v53") or {}).get("earth_hash") == _earth_hash(earth_now))
        )  # fmt: skip
        atm_now = getattr(project, "atmosphere", None)
        env_same = env_same and (
            (atm_now is None and not existing.get("v54"))
            or (atm_now is not None
                and (existing.get("v54") or {}).get("atm_hash") == _atm_hash(atm_now))
        )  # fmt: skip
        volc_now = getattr(project, "volcanism", None)
        env_same = env_same and (
            (volc_now is None and not existing.get("v51"))
            or (volc_now is not None
                and (existing.get("v51") or {}).get("volc_hash") == _volc_hash(volc_now))
        )  # fmt: skip
        topo_now = getattr(project, "topo", None)
        env_same = env_same and (
            (topo_now is None and not existing.get("v47"))
            or (topo_now is not None
                and (existing.get("v47") or {}).get("topo_hash") == _topo_hash(topo_now))
        )  # fmt: skip
        nothing_new = (
            (project.geo is None or geo_same)
            and (project.hist is None or hist_same)
            and (project.hydro is None or hydro_same)
            and env_same
        )
        if nothing_new:
            stored_config = json.loads((path / "config.json").read_text(encoding="utf-8"))
            if stored_config == project.config.to_dict() or project.geo is None:
                return path  # nothing new to store
        old_schema = existing.get("schema_version", "")
        if old_schema in ("hexworld-project-1", "hexworld-project-2", "hexworld-project-3"):
            tag = "v" + old_schema.rsplit("-", 1)[1]
            for name in ("config.json", "manifest.json"):
                backup = path / name.replace(".json", f".{tag}.json")
                if not backup.exists():
                    backup.write_bytes((path / name).read_bytes())
    path.mkdir(parents=True, exist_ok=True)

    if not same_geology:
        _write_v1_data(project, path)
        stale = (
            V2_DATA_FILES + V3_DATA_FILES + V4_DATA_FILES + V45_DATA_FILES + V46_DATA_FILES
            + V47_DATA_FILES + V50_DATA_FILES
        )  # fmt: skip
        for name in stale:  # no stale data of another world
            (path / name).unlink(missing_ok=True)
    _write_json(path / "config.json", project.config.to_dict())
    files = list(DATA_FILES)
    v2_manifest = None
    v3_manifest = None
    v4_manifest = None
    v45_manifest = None
    stored_env = (existing.get("v45") or {}).get("env_hash")
    if project.env is not None and project.hydro is not None:
        env_hash = _env_hash(project.env)
        if not (
            same_geology and stored_env == env_hash
            and all((path / n).exists() for n in V45_DATA_FILES)
        ):  # fmt: skip
            _write_v45_data(project, path)
        files += list(V45_DATA_FILES)
        v45_manifest = {
            "algorithm_version": ENV_ALGORITHM_VERSION,
            "files": list(V45_DATA_FILES),
            "env_hash": env_hash,
            "content_hashes": project.env.content_hashes(),
            "stage_keys": project.env.stage_keys,
            "layers": _env_layer_table(project.env),
        }
    else:
        for name in V45_DATA_FILES:  # an ecological layer of another v4 state would be stale
            (path / name).unlink(missing_ok=True)
    v50_manifest = None
    bio = getattr(project, "biomes", None)
    if bio is not None and project.env is not None:
        b_hash = _biome_hash(bio)
        if not (
            same_geology and (existing.get("v50") or {}).get("biome_hash") == b_hash
            and all((path / n).exists() for n in V50_DATA_FILES)
        ):  # fmt: skip
            _write_v50_data(project, path)
        files += list(V50_DATA_FILES)
        v50_manifest = {
            "algorithm_version": BIOMES_ALGORITHM_VERSION,
            "files": list(V50_DATA_FILES),
            "biome_hash": b_hash,
            "content_hashes": bio.content_hashes(),
            "stage_keys": bio.stage_keys,
        }
    else:
        for name in V50_DATA_FILES:
            (path / name).unlink(missing_ok=True)
    v53_manifest = None
    earth = getattr(project, "earth", None)
    if earth is not None and project.env is not None:
        e_hash = _earth_hash(earth)
        if not (
            same_geology and (existing.get("v53") or {}).get("earth_hash") == e_hash
            and all((path / n).exists() for n in V53_DATA_FILES)
        ):  # fmt: skip
            _write_v53_data(project, path)
        files += list(V53_DATA_FILES)
        v53_manifest = {
            "algorithm_version": EARTH_ALGORITHM_VERSION,
            "files": list(V53_DATA_FILES),
            "earth_hash": e_hash,
            "content_hashes": earth.content_hashes(),
            "stage_keys": earth.stage_keys,
        }
    else:
        for name in V53_DATA_FILES:
            (path / name).unlink(missing_ok=True)
    v54_manifest = None
    atm = getattr(project, "atmosphere", None)
    if atm is not None and project.env is not None and atm.grid_fields:
        from .atmosphere.normals import ALGORITHM_VERSION as ATM_VERSION

        a_hash = _atm_hash(atm)
        if not (
            same_geology and (existing.get("v54") or {}).get("atm_hash") == a_hash
            and all((path / n).exists() for n in V54_DATA_FILES)
        ):  # fmt: skip
            _write_v54_data(project, path)
        files += list(V54_DATA_FILES)
        v54_manifest = {
            "algorithm_version": ATM_VERSION,
            "files": list(V54_DATA_FILES),
            "atm_hash": a_hash,
            "content_hashes": atm.content_hashes(),
            "stage_keys": atm.stage_keys,
        }
    else:
        for name in V54_DATA_FILES:
            (path / name).unlink(missing_ok=True)
    v52_manifest = None
    res = getattr(project, "resources", None)
    if res is not None and project.env is not None:
        from .resources import ALGORITHM_VERSION as RES_VERSION

        _write_v52_data(project, path)
        files += list(V52_DATA_FILES)
        v52_manifest = {
            "algorithm_version": RES_VERSION,
            "files": list(V52_DATA_FILES),
            "content_hashes": res.content_hashes(),
            "stage_keys": res.stage_keys,
        }
    else:
        for name in V52_DATA_FILES:
            (path / name).unlink(missing_ok=True)
    v551_manifest = None
    mins = getattr(project, "minerals", None)
    if mins is not None and res is not None:
        from .minerals.stage import to_json as minerals_json

        np.savez_compressed(path / "minerals_fields.npz", **mins.layers)
        _write_json(path / "minerals.json", minerals_json(mins))
        files += list(V551_DATA_FILES)
        v551_manifest = {
            "algorithm_version": mins.provenance["algorithm_version"],
            "files": list(V551_DATA_FILES),
            "content_hashes": mins.content_hashes(),
            "stage_keys": mins.stage_keys,
        }
    else:
        for name in V551_DATA_FILES:
            (path / name).unlink(missing_ok=True)
    v51_manifest = None
    volc = getattr(project, "volcanism", None)
    if volc is not None and project.hydro is not None:
        v_hash = _volc_hash(volc)
        if not (
            same_geology and (existing.get("v51") or {}).get("volc_hash") == v_hash
            and all((path / n).exists() for n in V51_DATA_FILES)
        ):  # fmt: skip
            _write_v51_data(project, path)
        files += list(V51_DATA_FILES)
        v51_manifest = {
            "algorithm_version": VOLCANISM_ALGORITHM_VERSION,
            "files": list(V51_DATA_FILES),
            "volc_hash": v_hash,
            "content_hashes": volc.content_hashes(),
            "stage_keys": volc.stage_keys,
        }
    else:
        for name in V51_DATA_FILES:
            (path / name).unlink(missing_ok=True)
    v47_manifest = None
    topo = getattr(project, "topo", None)
    if topo is not None and project.env is not None:
        topo_hash = _topo_hash(topo)
        stored_topo = (existing.get("v47") or {}).get("topo_hash")
        if not (
            same_geology and stored_topo == topo_hash
            and all((path / n).exists() for n in V47_DATA_FILES)
        ):  # fmt: skip
            _write_v47_data(project, path)
        files += list(V47_DATA_FILES)
        v47_manifest = {
            "algorithm_version": TOPOCLIMATE_ALGORITHM_VERSION,
            "files": list(V47_DATA_FILES),
            "topo_hash": topo_hash,
            "content_hashes": topo.content_hashes(),
            "stage_keys": topo.stage_keys,
            "units": int(topo.units["unit_hex"].size),
        }
    else:
        for name in V47_DATA_FILES:  # a topoclimate of another v4.5 state would be stale
            (path / name).unlink(missing_ok=True)
    v46_manifest = None
    terrain = getattr(project, "terrain", None)
    if terrain is not None:
        t_hash = _terrain_hash(terrain)
        stored_t = (existing.get("v46") or {}).get("terrain_hash")
        if not (
            same_geology and stored_t == t_hash
            and all((path / n).exists() for n in V46_DATA_FILES)
        ):  # fmt: skip
            _write_v46_data(project, path)
        files += list(V46_DATA_FILES)
        v46_manifest = {
            "algorithm_version": TERRAIN_ALGORITHM_VERSION,
            "files": list(V46_DATA_FILES),
            "terrain_hash": t_hash,
            "content_hashes": terrain.content_hashes(),
            "stage_keys": terrain.stage_keys,
        }
    else:
        for name in V46_DATA_FILES:
            (path / name).unlink(missing_ok=True)
    if project.hydro is not None:
        if not (same_geology and hydro_same and all((path / n).exists() for n in V4_DATA_FILES)):
            _write_v4_data(project, path)
        files += list(V4_DATA_FILES)
        v4_manifest = {
            "algorithm_versions": {
                "climate": CLIMATE_ALGORITHM_VERSION,
                "hydrology": HYDRO_ALGORITHM_VERSION,
            },
            "files": list(V4_DATA_FILES),
            "epoch_time_ma": project.hydro.epoch_time_ma,
            "sea_level_m": project.hydro.sea_level_m,
            "hydro_hash": project.hydro.hydro_hash(),
            "content_hashes": project.hydro.content_hashes(),
            "layers": {
                name: {
                    "dtype": str(arr.dtype),
                    "shape": list(arr.shape),
                    "unit": HYDRO_LAYER_SPECS[name][1],
                    "description": HYDRO_LAYER_SPECS[name][2],
                }
                for name, arr in project.hydro.layers.items()
            },
            "stage_keys": project.hydro.stage_keys,
        }
    else:
        keep_v4 = same_geology and hist_same and all((path / n).exists() for n in V4_DATA_FILES)
        if keep_v4:
            files += list(V4_DATA_FILES)
            v4_manifest = existing.get("v4")
        else:
            for name in V4_DATA_FILES:  # a v4 result of another v3 surface would be stale
                (path / name).unlink(missing_ok=True)
    if project.hist is not None:
        if not (same_geology and hist_same and all((path / n).exists() for n in V3_DATA_FILES)):
            _write_v3_data(project, path)
        files += list(V3_DATA_FILES)
        v3_manifest = {
            "algorithm_version": HISTORY_ALGORITHM_VERSION,
            "mountains_algorithm_version": MOUNTAINS_ALGORITHM_VERSION,
            "files": list(V3_DATA_FILES),
            "epoch_time_ma": project.hist.epoch_time_ma,
            "epochs_ma": project.hist.epochs,
            "sea_level_m": project.hist.sea_level_m,
            "hist_hash": project.hist.hist_hash(),
            "content_hashes": project.hist.content_hashes(),
            "layers": {
                name: {
                    "dtype": str(arr.dtype),
                    "shape": list(arr.shape),
                    "unit": HIST_LAYER_SPECS[name][1],
                    "description": HIST_LAYER_SPECS[name][2],
                }
                for name, arr in project.hist.layers.items()
            },
            "base_macro_elevation_m": "continental_fields.npz -> macro_elevation_m (reference)",
            "source_stages": ["v1 plates/crust/kinematics", "v2 continents/macro elevation"],
            "stage_keys": project.hist.stage_keys,
        }
    else:
        for name in V3_DATA_FILES:  # a v3 result of other v2 inputs would be stale
            if not (same_geology and geo_same):
                (path / name).unlink(missing_ok=True)
        if same_geology and geo_same and all((path / n).exists() for n in V3_DATA_FILES):
            files += list(V3_DATA_FILES)
            v3_manifest = existing.get("v3")
    if project.geo is not None:
        if not (same_geology and geo_same and all((path / n).exists() for n in V2_DATA_FILES)):
            _write_v2_data(project, path)
        files += list(V2_DATA_FILES)
        v2_manifest = {
            "algorithm_version": CONTINENTS_ALGORITHM_VERSION,
            "files": list(V2_DATA_FILES),
            "sea_level_m": project.geo.sea_level_m,
            "geo_hash": project.geo.geo_hash(),
            "content_hashes": project.geo.content_hashes(),
            "layers": {
                name: {
                    "dtype": str(arr.dtype),
                    "shape": list(arr.shape),
                    "unit": GEO_LAYER_SPECS[name][1],
                    "description": GEO_LAYER_SPECS[name][2],
                }
                for name, arr in project.geo.layers.items()
            },
            "stage_keys": project.geo.stage_keys,
        }
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "generator_version": GENERATOR_VERSION,
        "algorithm_version": ALGORITHM_VERSION,
        "prng_version": PRNG_VERSION,
        "coordinate_version": COORDINATE_VERSION,
        "world_id": project.world_id,
        "geology_hash": geology_hash,
        "coordinates": {
            "world": "planar, km; +x east, +y south; origin arbitrary",
            "hex": "flat-top; storage odd-q offset [row, col]; computation axial/cube",
            "vertical": "metres; 0 m = model datum; sea level is a scenario parameter",
            "geometry": project.geometry.to_dict(),
        },
        "units": {name: spec[1] for name, spec in LAYER_SPECS.items()},
        "layers": {
            name: {
                "dtype": str(arr.dtype),
                "shape": list(arr.shape),
                "description": LAYER_SPECS.get(name, ("", "", ""))[2],
            }
            for name, arr in project.layers.items()
        },
        "stage_keys": project.stage_keys,
        "stats": project.stats,
        "timings_s": project.timings,
        "validation": project.validation,
        "warnings": project.warnings,
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "platform": platform.platform(),
        },
        "content_hashes": project.content_hashes(),
        "v2": v2_manifest,
        "v3": v3_manifest,
        "v4": v4_manifest,
        "v45": v45_manifest,
        "v46": v46_manifest,
        "v47": v47_manifest,
        "v50": v50_manifest,
        "v51": v51_manifest,
        "v53": v53_manifest,
        "v54": v54_manifest,
        "v52": v52_manifest,
        "v551": v551_manifest,
        "file_hashes": {name: _file_hash(path / name) for name in files},
    }
    _write_json(manifest_path, manifest)
    return path


def load_project(path: str | Path, verify: bool = True) -> Project:
    """Load a project directory, checking control hashes unless ``verify=False``."""
    path = Path(path)
    manifest = json.loads((path / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("schema_version") not in SUPPORTED_SCHEMAS:
        raise ProjectIntegrityError(
            f"Unsupported schema {manifest.get('schema_version')!r}; "
            f"this build opens {SUPPORTED_SCHEMAS}"
        )
    if verify:
        for name, expected in manifest["file_hashes"].items():
            if _file_hash(path / name) != expected:
                raise ProjectIntegrityError(f"{name} does not match the manifest hash")

    raw_config = json.loads((path / "config.json").read_text(encoding="utf-8"))
    if "layout" not in raw_config.get("crust", {}):
        # written before the clustered layout existed: that world used the uniform one
        raw_config.setdefault("crust", {})["layout"] = "uniform"
    config = Config.from_dict(raw_config)
    plates_doc = json.loads((path / "plates.json").read_text(encoding="utf-8"))
    plates = plates_doc["plates"]
    with np.load(path / "fields.npz") as data:
        layers = {name: data[name] for name in data.files}
    bdoc = json.loads((path / "boundaries.json").read_text(encoding="utf-8"))
    with np.load(path / "boundaries.npz") as data:
        edges = BoundaryEdges(
            **{k: data[k] for k in data.files}, length_km=float(bdoc["edge_length_km"])
        )

    geometry = resolve_geometry(config)
    motion = PlateMotion(
        velocity=np.array([p["velocity_mm_yr"] for p in plates], dtype=np.float64),
        omega=np.array([p["omega_deg_myr"] for p in plates], dtype=np.float64),
        pole=np.array([p["rotation_pole_km"] for p in plates], dtype=np.float64),
    )
    project = Project(
        config=config,
        geometry=geometry,
        world_id=manifest["world_id"],
        plates=plates,
        layers=layers,
        boundaries=edges,
        boundary_summary=bdoc["pairs"],
        motion=motion,
        stage_keys=dict(manifest["stage_keys"]),
        warnings=list(manifest.get("warnings", [])),
        timings=dict(manifest.get("timings_s", {})),
        stats=dict(manifest.get("stats", {})),
    )
    if verify:
        actual = project.content_hashes()
        for key, expected in manifest["content_hashes"].items():
            if actual.get(key) != expected:
                raise ProjectIntegrityError(f"array {key} does not match the manifest hash")

    # The world models are analytic and cheap: rebuild them from the config so the
    # loaded project can answer world-space queries, then prove they are the same.
    compatible = (
        manifest.get("algorithm_version") == ALGORITHM_VERSION
        and manifest.get("prng_version") == PRNG_VERSION
    )
    if compatible:
        plate_model = build_plate_model(config, geometry)
        ref = build_reference_grid(geometry, config.reference_cells)
        crust_model = build_crust_model(config, geometry, ref.x[ref.in_sim], ref.y[ref.in_sim])
        saved_p = (plates_doc.get("plate_model") or {}).get("model_hash")
        saved_c = (plates_doc.get("crust_model") or {}).get("model_hash")
        exact = plate_model.model_hash() == saved_p and crust_model.model_hash() == saved_c
        agree = None
        if not exact:
            # another CPU / NumPy build can differ in the last bits of sin, cos, exp; the
            # models are the same if they still give the stored plates and crust
            hx, hy = geometry.layout.centers()
            agree = min(
                float(np.mean(plate_model.plate_at(hx, hy) == project.layers["plate_id"])),
                float(
                    np.mean(
                        np.asarray(crust_model.crust_at(hx, hy)) == project.layers["crust_type"]
                    )
                ),
            )
        if exact or (agree is not None and agree >= 0.995):
            if not exact:
                project.warnings.append(
                    f"World models were rebuilt with floating-point differences (another CPU or "
                    f"NumPy build): they reproduce {100 * agree:.2f} % of the stored plates and "
                    "crust; the stored layers are used as they are."
                )
            ref.plate = plate_model.plate_at(ref.x, ref.y)
            ref.crust = crust_model.crust_at(ref.x, ref.y)
            project.plate_model = plate_model
            project.crust_model = crust_model
            project.cache = {
                "plates": (plate_model, ref.plate),
                "crust": (crust_model, ref.crust),
                "motion": motion,
                "reference": ref,
            }
            keys = stage_keys(config, geometry, _pad_hexes(config))
            if world_id_from_keys(keys) != project.world_id:
                project.warnings.append("world_id differs from the one recomputed on load")
            _reference_stats(ref, plate_model.count)  # sanity: must not raise
        else:
            compatible = False
    if not compatible:
        project.warnings.append(
            "World models could not be reproduced by this generator version; "
            "stored layers are intact, world-space queries are disabled."
        )
    project.validation = validate_project(project)

    v2 = manifest.get("v2")
    if v2 and all((path / name).exists() for name in V2_DATA_FILES):
        doc = json.loads((path / "continents.json").read_text(encoding="utf-8"))
        with np.load(path / "continental_fields.npz") as data:
            arrays = {name: data[name] for name in data.files}
        world_elev = arrays.pop("world_elevation_m")
        world_c = arrays.pop("world_continentality_km")
        geo = GeoData(
            layers=arrays,
            sea_level_m=float(doc["sea_level_m"]),
            blocks=doc["blocks"],
            landmasses=doc["landmasses"],
            summary=doc["summary"],
            stage_keys=dict(doc["stage_keys"]),
            world={
                "grid": doc["world_grid"],
                "elevation_m": world_elev,
                "continentality_km": world_c,
            },
            warnings=list(doc.get("warnings", [])),
            timings=dict(doc.get("timings_s", {})),
        )
        if verify:
            actual = geo.content_hashes()
            for key, expected in v2["content_hashes"].items():
                if actual.get(key) != expected:
                    raise ProjectIntegrityError(f"array {key} does not match the manifest hash")
        project.geo = geo
        geo.validation = validate_geo(project)
        project.warnings = [w for w in project.warnings]
        # make the stored v2 arrays reusable: a sea-level change after loading
        # recomputes only the land mask, never crust or elevation
        current = v2_stage_keys(config, project.stage_keys, geometry.layout)
        if doc.get("algorithm_version") == CONTINENTS_ALGORITHM_VERSION and all(
            current[k] == geo.stage_keys.get(k) for k in ("macro_elevation", "v2_raster")
        ):
            block_id = arrays["continental_block_id"]
            project.cache["macro_elevation"] = {
                "model": None,
                "world_elevation_m": world_elev,
                "world_continentality_km": world_c,
                "grid": doc["world_grid"],
                "structure": doc["summary"].get("structure", {}),
                "structure_warnings": [],
            }
            project.cache["v2_raster"] = {
                "continental_fraction": arrays["continental_fraction"],
                "continental_margin_km": arrays["continental_margin_km"],
                "continental_block_id": block_id,
                "block_sizes": np.bincount(block_id[block_id >= 0]).astype(np.int64),
                "macro_elevation_m": arrays["macro_elevation_m"],
                "tectonic_relief_m": arrays["tectonic_relief_m"],
            }
    _load_history(project, path, manifest, config, geometry, verify)
    _load_terrain(project, path, manifest, config, verify)
    _load_hydro(project, path, manifest, config, verify)
    _load_env(project, path, manifest, config, verify)
    _load_topo(project, path, manifest, config, verify)
    _load_earth(project, path, manifest, config, verify)
    _load_atmosphere(project, path, manifest, config, verify)
    _load_volcanism(project, path, manifest, config, verify)
    _load_biomes(project, path, manifest, config, verify)
    _load_resources(project, path, manifest, config, verify)
    _load_minerals(project, path, manifest, config, verify)
    return project


def _load_history(project: Project, path: Path, manifest: dict, config, geometry, verify) -> None:
    """Attach stored v3.0 data, or explain why it cannot be used. Never silently stale."""
    from .mountains.plan import MountainPlan
    from .stages.volcanism import Hotspot, Volcano, VolcanoSet

    v3 = manifest.get("v3")
    if not v3 or project.geo is None:
        return
    if not all((path / name).exists() for name in V3_DATA_FILES):
        project.warnings.append("The manifest lists v3.0 data, but its files are missing.")
        return
    doc = json.loads((path / "history.json").read_text(encoding="utf-8"))
    stored = (doc.get("algorithm_version"), doc.get("mountains_algorithm_version"))
    if stored != (HISTORY_ALGORITHM_VERSION, MOUNTAINS_ALGORITHM_VERSION):
        project.warnings.append(
            f"The stored geological history was produced by algorithms {stored}, this build "
            f"uses {(HISTORY_ALGORITHM_VERSION, MOUNTAINS_ALGORITHM_VERSION)}. It was NOT "
            "loaded; generate the history again (the v1/v2 baseline is intact)."
        )
        return
    with np.load(path / "history_fields.npz") as data:
        arrays = {name: data[name] for name in data.files}
    world = {k[len("world_") :]: arrays.pop(k) for k in list(arrays) if k.startswith("world_")}
    world["grid"] = doc["world_grid"]
    times = [float(t) for t in doc["checkpoint_times_ma"]]
    checkpoints: dict = {t: {} for t in times}
    with np.load(path / "checkpoints.npz") as data:
        for name in data.files:
            index, field_name = name.split("_", 1)
            checkpoints[times[int(index[1:])]][field_name] = data[name]
    events = json.loads((path / "events.json").read_text(encoding="utf-8"))["events"]
    plan = json.loads((path / "mountain_plan.json").read_text(encoding="utf-8"))
    hist = HistoryData(
        layers=arrays,
        epoch_time_ma=float(doc["epoch_time_ma"]),
        epochs=[float(t) for t in doc["epochs_ma"]],
        duration_ma=float(doc["duration_ma"]),
        sea_level_m=float(doc["sea_level_m"]),
        events=events,
        plan=plan,
        volcanoes=doc["volcanoes"],
        hotspots=doc["hotspots"],
        landmasses=doc["landmasses"],
        summary=doc["summary"],
        stage_keys=dict(doc["stage_keys"]),
        world=world,
        checkpoints=checkpoints,
        balance=doc.get("erosion_balance_per_step", []),
        warnings=list(doc.get("warnings", [])),
        timings=dict(doc.get("timings_s", {})),
    )
    if verify:
        actual = hist.content_hashes()
        for key, expected in v3["content_hashes"].items():
            if actual.get(key) != expected:
                raise ProjectIntegrityError(f"array {key} does not match the manifest hash")
    current = v3_stage_keys(config, project.geo.stage_keys, geometry.layout, hist.epoch_time_ma)
    if current["geo_history"] != hist.stage_keys.get("geo_history"):
        project.warnings.append(
            "The stored geological history does not match the stored v1/v2 baseline or "
            "configuration. It was NOT loaded; generate the history again."
        )
        return
    project.hist = hist
    hist.validation = validate_history(project)
    volcano_set = VolcanoSet(
        hotspots=[Hotspot(**h) for h in doc.get("hotspot_state", [])],
        volcanoes=[Volcano(**v) for v in doc.get("volcano_records", [])],
    )
    project.cache["geo_history"] = {
        "sim": None,
        "relief_model": None,
        "checkpoints": {t: _state_from_arrays(a) for t, a in checkpoints.items()},
        "events": [e for e in events if not e.get("derived")],
        "balance": hist.balance,
        "info": doc.get("run_info", {}),
        "volcanoes": volcano_set,
    }
    land_layers = ("land_mask", "landmass_id", "shelf_mask")
    if all(current[k] == hist.stage_keys.get(k) for k in ("mountain_plan", "v3_raster")):
        stored_plan = MountainPlan.from_dict(plan)
        stored_plan.system_raster = world["mountain_system_id"]
        project.cache["mountain_plan"] = stored_plan
        project.cache["v3_raster"] = {
            "layers": {k: v for k, v in arrays.items() if k not in land_layers},
            "world": world,
            "rates": hist.summary.get("world_rates", {}),
        }


def _load_env(project: Project, path: Path, manifest: dict, config, verify) -> None:
    """Attach stored v4.5 data, or say why not (it is never loaded stale)."""
    from .pipeline_v45 import env_stage_keys
    from .project import EnvData
    from .stages.validate import validate_env

    v45 = manifest.get("v45")
    if not v45 or project.hydro is None:
        return
    if not all((path / name).exists() for name in V45_DATA_FILES):
        project.warnings.append("The manifest lists v4.5 data, but its files are missing.")
        return
    doc = json.loads((path / "environment.json").read_text(encoding="utf-8"))
    if doc.get("algorithm_version") != ENV_ALGORITHM_VERSION:
        project.warnings.append(
            f"The stored ecological layers were produced by {doc.get('algorithm_version')}, "
            f"this build uses {ENV_ALGORITHM_VERSION}: NOT loaded (not calculated)."
        )
        return
    keys = env_stage_keys(config, project.hydro.stage_keys)
    if keys["v45_environment"] != doc["stage_keys"].get("v45_environment"):
        project.warnings.append(
            "The stored ecological layers belong to another v4 state or configuration: "
            "NOT loaded (not calculated)."
        )
        return
    with np.load(path / "environment_fields.npz") as data:
        layers = {name: data[name] for name in data.files}
    with np.load(path / "environment_monthly.npz") as data:
        monthly = {name: data[name] for name in data.files}
    env = EnvData(
        layers=layers,
        monthly=monthly,
        summary=doc["summary"],
        methods=doc["methods"],
        stage_keys=dict(doc["stage_keys"]),
        warnings=list(doc.get("warnings", [])),
        timings=dict(doc.get("timings_s", {})),
    )
    if verify:
        actual = env.content_hashes()
        for key, expected in v45["content_hashes"].items():
            if actual.get(key) != expected:
                raise ProjectIntegrityError(f"array {key} does not match the manifest hash")
    project.env = env
    env.validation = validate_env(project)
    project.cache["v45_environment"] = env


def _load_earth(project: Project, path: Path, manifest: dict, config, verify) -> None:
    """Attach stored v5.3a data, or say why not (missing = not generated, never zeros)."""
    from .pipeline_v53 import earth_stage_keys
    from .project import EarthData
    from .stages.validate import validate_earth

    v53 = manifest.get("v53")
    if not v53 or project.env is None:
        return
    if not all((path / name).exists() for name in V53_DATA_FILES):
        project.warnings.append("The manifest lists v5.3 data, but its files are missing.")
        return
    doc = json.loads((path / "earth.json").read_text(encoding="utf-8"))
    if doc.get("algorithm_version") != EARTH_ALGORITHM_VERSION:
        project.warnings.append(
            f"The stored Earth system was produced by {doc.get('algorithm_version')}, this "
            f"build uses {EARTH_ALGORITHM_VERSION}: NOT loaded (not generated)."
        )
        return
    keys = earth_stage_keys(config, project)
    if keys["v53_earth"] != doc["stage_keys"].get("v53_earth"):
        project.warnings.append(
            "The stored Earth system belongs to another climate state or configuration: "
            "NOT loaded (not generated)."
        )
        return
    with np.load(path / "earth_fields.npz") as data:
        layers = {k: data[k] for k in data.files}
    e = EarthData(layers=layers, summary=doc["summary"], provenance=doc["provenance"],
                  stage_keys=dict(doc["stage_keys"]), warnings=list(doc.get("warnings", [])),
                  timings=dict(doc.get("timings_s", {})))  # fmt: skip
    if verify:
        actual = e.content_hashes()
        for key, expected in v53["content_hashes"].items():
            if actual.get(key) != expected:
                raise ProjectIntegrityError(f"array {key} does not match the manifest hash")
    project.earth = e
    e.validation = validate_earth(project)
    project.cache["v53_earth"] = e


def _load_atmosphere(project: Project, path: Path, manifest: dict, config, verify) -> None:
    """Attach stored v5.4 data (normals, diagnostics, events) without regenerating it."""
    from .atmosphere.normals import ALGORITHM_VERSION as ATM_VERSION
    from .pipeline_v54 import atmosphere_stage_keys
    from .project import AtmosphereData
    from .stages.validate import validate_atmosphere

    v54 = manifest.get("v54")
    if not v54 or project.env is None or getattr(project, "earth", None) is None:
        return
    if not all((path / name).exists() for name in V54_DATA_FILES):
        project.warnings.append("The manifest lists v5.4 data, but its files are missing.")
        return
    doc = json.loads((path / "atmosphere.json").read_text(encoding="utf-8"))
    if doc.get("algorithm_version") != ATM_VERSION:
        project.warnings.append(
            f"The stored atmosphere was produced by {doc.get('algorithm_version')}, this "
            f"build uses {ATM_VERSION}: NOT loaded (not generated)."
        )
        return
    keys = atmosphere_stage_keys(config, project)
    if keys["v54_atmosphere"] != doc["stage_keys"].get("v54_atmosphere"):
        project.warnings.append(
            "The stored atmosphere belongs to another climate state or configuration: "
            "NOT loaded (not generated)."
        )
        return
    with np.load(path / "atmosphere_fields.npz") as data:
        layers = {k: data[k] for k in data.files}
    with np.load(path / "atmosphere_grid.npz") as data:
        grid_fields = {k: data[k] for k in data.files}
    a = AtmosphereData(layers=layers, grid_fields=grid_fields, grid=doc["grid"],
                       summary=doc["summary"], provenance=doc["provenance"],
                       stage_keys=dict(doc["stage_keys"]), events=list(doc.get("events", [])),
                       warnings=list(doc.get("warnings", [])),
                       timings=dict(doc.get("timings_s", {})))  # fmt: skip
    if verify:
        actual = a.content_hashes()
        for key, expected in v54["content_hashes"].items():
            if actual.get(key) != expected:
                raise ProjectIntegrityError(f"array {key} does not match the manifest hash")
    project.atmosphere = a
    a.validation = validate_atmosphere(project)
    project.cache["v54_atmosphere"] = a


def _load_resources(project: Project, path: Path, manifest: dict, config, verify) -> None:
    """Attach stored v5.2 data (fertility layers, deposit catalogue) without regenerating."""
    from .pipeline_v52 import resource_stage_keys
    from .project import ResourceData
    from .resources import ALGORITHM_VERSION as RES_VERSION
    from .stages.validate import validate_resources

    v52 = manifest.get("v52")
    if not v52 or project.env is None or project.hist is None:
        return
    if not all((path / name).exists() for name in V52_DATA_FILES):
        project.warnings.append("The manifest lists v5.2 data, but its files are missing.")
        return
    doc = json.loads((path / "resources.json").read_text(encoding="utf-8"))
    if doc.get("algorithm_version") != RES_VERSION:
        project.warnings.append(
            f"The stored resources were produced by {doc.get('algorithm_version')}, this "
            f"build uses {RES_VERSION}: NOT loaded (not generated)."
        )
        return
    keys = resource_stage_keys(config, project)
    if keys["v52_resources"] != doc["stage_keys"].get("v52_resources"):
        project.warnings.append(
            "The stored resources belong to another world state or configuration: NOT "
            "loaded (not generated)."
        )
        return
    with np.load(path / "resources_fields.npz") as data:
        layers = {k: data[k] for k in data.files}
    with np.load(path / "resources_index.npz") as data:
        index = {k: data[k] for k in data.files}
    catalog = dict(doc["catalog"])
    catalog.update(index)
    r = ResourceData(layers=layers, catalog=catalog, summary=doc["summary"],
                     provenance=doc["provenance"], stage_keys=dict(doc["stage_keys"]),
                     warnings=list(doc.get("warnings", [])),
                     timings=dict(doc.get("timings_s", {})))  # fmt: skip
    if verify:
        actual = r.content_hashes()
        for key, expected in v52["content_hashes"].items():
            if actual.get(key) != expected:
                raise ProjectIntegrityError(f"array {key} does not match the manifest hash")
    project.resources = r
    r.validation = validate_resources(project)
    project.cache["v52_resources"] = r


def _load_minerals(project: Project, path: Path, manifest: dict, config, verify) -> None:
    """Attach the stored 5.5.1 plan; old saves without it stay as they were (T02)."""
    from .minerals import ALGORITHM_VERSION as MIN_VERSION
    from .minerals.stage import from_json as minerals_from_json
    from .minerals.stage import stage_keys as minerals_keys

    v551 = manifest.get("v551")
    if not v551 or project.resources is None:
        return
    if not all((path / name).exists() for name in V551_DATA_FILES):
        project.warnings.append("The manifest lists 5.5.1 data, but its files are missing.")
        return
    doc = json.loads((path / "minerals.json").read_text(encoding="utf-8"))
    if doc.get("algorithm_version") != MIN_VERSION:
        project.warnings.append(
            f"The stored mineral plan was produced by {doc.get('algorithm_version')}, this "
            f"build uses {MIN_VERSION}: NOT loaded (not generated)."
        )
        return
    if getattr(config, "minerals", None) is None or not config.minerals.enabled:
        return
    if minerals_keys(config, project)["v551_minerals"] != doc["stage_keys"].get("v551_minerals"):
        project.warnings.append(
            "The stored mineral plan belongs to another world state: NOT loaded."
        )
        return
    with np.load(path / "minerals_fields.npz") as data:
        layers = {k: data[k] for k in data.files}
    m = minerals_from_json(doc, layers)
    if verify:
        actual = m.content_hashes()
        for key, expected in v551["content_hashes"].items():
            if actual.get(key) != expected:
                raise ProjectIntegrityError(f"array {key} does not match the manifest hash")
    project.minerals = m


def _load_volcanism(project: Project, path: Path, manifest: dict, config, verify) -> None:
    """Attach stored v5.1 data, or say why not (never stale; missing = not generated)."""
    from .pipeline_v51 import upstream_keys, volcanism_stage_keys
    from .project import VolcanismData
    from .stages.validate import validate_volcanism

    v51 = manifest.get("v51")
    if not v51 or project.hydro is None:
        return
    if not all((path / name).exists() for name in V51_DATA_FILES):
        project.warnings.append("The manifest lists v5.1 data, but its files are missing.")
        return
    doc = json.loads((path / "volcanism.json").read_text(encoding="utf-8"))
    if doc.get("algorithm_version") != VOLCANISM_ALGORITHM_VERSION:
        project.warnings.append(
            f"The stored volcanism was produced by {doc.get('algorithm_version')}, this build "
            f"uses {VOLCANISM_ALGORITHM_VERSION}: NOT loaded (not generated)."
        )
        return
    keys = volcanism_stage_keys(config, upstream_keys(project))
    if keys["v51_volcanism"] != doc["stage_keys"].get("v51_volcanism"):
        project.warnings.append(
            "The stored volcanism belongs to another geology / climate state or configuration: "
            "NOT loaded (not generated)."
        )
        return
    with np.load(path / "volcanism_fields.npz") as data:
        layers = {k: data[k] for k in data.files}
    vol = VolcanismData(
        layers=layers, systems=doc["systems"], events=doc["events"],
        provinces=doc["provinces"], aggregates=doc["aggregates"], summary=doc["summary"],
        provenance=doc["provenance"], stage_keys=dict(doc["stage_keys"]),
        warnings=list(doc.get("warnings", [])), timings=dict(doc.get("timings_s", {})),
    )  # fmt: skip
    if verify:
        actual = vol.content_hashes()
        for key, expected in v51["content_hashes"].items():
            if actual.get(key) != expected:
                raise ProjectIntegrityError(f"array {key} does not match the manifest hash")
    project.volcanism = vol
    vol.validation = validate_volcanism(project)
    project.cache["v51_volcanism"] = vol


def _load_biomes(project: Project, path: Path, manifest: dict, config, verify) -> None:
    """Attach stored v5.0 data, or say why not (never stale; missing = not generated)."""
    from .pipeline_v50 import biome_stage_keys, upstream_keys
    from .project import BiomeData
    from .stages.validate import validate_biomes

    v50 = manifest.get("v50")
    if not v50 or project.env is None:
        return
    if not all((path / name).exists() for name in V50_DATA_FILES):
        project.warnings.append("The manifest lists v5.0 data, but its files are missing.")
        return
    doc = json.loads((path / "biomes.json").read_text(encoding="utf-8"))
    if doc.get("algorithm_version") != BIOMES_ALGORITHM_VERSION:
        project.warnings.append(
            f"The stored biomes were produced by {doc.get('algorithm_version')}, this build "
            f"uses {BIOMES_ALGORITHM_VERSION}: NOT loaded (not generated)."
        )
        return
    keys = biome_stage_keys(config, upstream_keys(project))
    if keys["v50_biomes"] != doc["stage_keys"].get("v50_biomes"):
        project.warnings.append(
            "The stored biomes belong to another v4.5 / v4.7 state or configuration: "
            "NOT loaded (not generated)."
        )
        return
    loaded = {}
    for name in ("biomes_fields.npz", "biomes_units.npz", "biomes_fractions.npz"):
        with np.load(path / name) as data:
            loaded[name] = {k: data[k] for k in data.files}
    bio = BiomeData(
        layers=loaded["biomes_fields.npz"],
        units=loaded["biomes_units.npz"],
        fractions=loaded["biomes_fractions.npz"],
        summary=doc["summary"],
        provenance=doc["provenance"],
        stage_keys=dict(doc["stage_keys"]),
        warnings=list(doc.get("warnings", [])),
        timings=dict(doc.get("timings_s", {})),
    )
    if verify:
        actual = bio.content_hashes()
        for key, expected in v50["content_hashes"].items():
            if actual.get(key) != expected:
                raise ProjectIntegrityError(f"array {key} does not match the manifest hash")
    project.biomes = bio
    bio.validation = validate_biomes(project)
    project.cache["v50_biomes"] = bio


def _load_topo(project: Project, path: Path, manifest: dict, config, verify) -> None:
    """Attach stored v4.7 data, or say why not (never stale)."""
    from .pipeline_v47 import topo_stage_keys, upstream_keys
    from .project import TopoData
    from .stages.validate import validate_topo

    v47 = manifest.get("v47")
    if not v47 or project.env is None:
        return
    if not all((path / name).exists() for name in V47_DATA_FILES):
        project.warnings.append("The manifest lists v4.7 data, but its files are missing.")
        return
    doc = json.loads((path / "topoclimate.json").read_text(encoding="utf-8"))
    if doc.get("algorithm_version") != TOPOCLIMATE_ALGORITHM_VERSION:
        project.warnings.append(
            f"The stored topoclimate was produced by {doc.get('algorithm_version')}, this "
            f"build uses {TOPOCLIMATE_ALGORITHM_VERSION}: NOT loaded (not calculated)."
        )
        return
    keys = topo_stage_keys(config, upstream_keys(project))
    if keys["v47_topoclimate"] != doc["stage_keys"].get("v47_topoclimate"):
        project.warnings.append(
            "The stored topoclimate belongs to another v4.5 state or configuration: "
            "NOT loaded (not calculated)."
        )
        return
    loaded = {}
    for name in ("topoclimate_fields.npz", "topoclimate_monthly.npz", "topoclimate_units.npz"):
        with np.load(path / name) as data:
            loaded[name] = {k: data[k] for k in data.files}
    topo = TopoData(
        layers=loaded["topoclimate_fields.npz"],
        monthly=loaded["topoclimate_monthly.npz"],
        units=loaded["topoclimate_units.npz"],
        summary=doc["summary"],
        massifs=doc["massifs"],
        provenance=doc["provenance"],
        stage_keys=dict(doc["stage_keys"]),
        warnings=list(doc.get("warnings", [])),
        timings=dict(doc.get("timings_s", {})),
    )
    if verify:
        actual = topo.content_hashes()
        for key, expected in v47["content_hashes"].items():
            if actual.get(key) != expected:
                raise ProjectIntegrityError(f"array {key} does not match the manifest hash")
    project.topo = topo
    topo.validation = validate_topo(project)
    project.cache["v47_topoclimate"] = topo


def _load_terrain(project: Project, path: Path, manifest: dict, config, verify) -> None:
    """Attach stored v4.6 data, or say why not (never stale)."""
    from .pipeline_v46 import terrain_stage_keys
    from .project import TerrainData
    from .stages.validate import validate_terrain

    v46 = manifest.get("v46")
    if not v46 or project.hist is None:
        return
    if not all((path / name).exists() for name in V46_DATA_FILES):
        project.warnings.append("The manifest lists v4.6 data, but its files are missing.")
        return
    doc = json.loads((path / "terrain.json").read_text(encoding="utf-8"))
    if doc.get("algorithm_version") != TERRAIN_ALGORITHM_VERSION:
        project.warnings.append(
            f"The stored v4.6 terrain was produced by {doc.get('algorithm_version')}, this "
            f"build uses {TERRAIN_ALGORITHM_VERSION}: NOT loaded (not calculated)."
        )
        return
    keys = terrain_stage_keys(config, project.hist.stage_keys)
    if keys["v46_terrain"] != doc["stage_keys"].get("v46_terrain"):
        project.warnings.append(
            "The stored v4.6 terrain belongs to another v3 epoch or configuration: "
            "NOT loaded (not calculated)."
        )
        return
    with np.load(path / "terrain_fields.npz") as data:
        layers = {name: data[name] for name in data.files}
    terrain = TerrainData(
        layers=layers,
        plans=doc["plans"],
        summary=doc["summary"],
        stage_keys=dict(doc["stage_keys"]),
        warnings=list(doc.get("warnings", [])),
        timings=dict(doc.get("timings_s", {})),
    )
    if verify:
        actual = terrain.content_hashes()
        for key, expected in v46["content_hashes"].items():
            if actual.get(key) != expected:
                raise ProjectIntegrityError(f"array {key} does not match the manifest hash")
    project.terrain = terrain
    terrain.validation = validate_terrain(project)
    # the composed surface can be reused (the water part is recomputed on demand)
    pre = ("ocean_source_mask", "ocean_mask_pre_v4")
    post = (
        "submarine_canyon_mask", "canyon_depth_m", "canyon_incision_m", "submarine_fan_m",
        "canyon_fan_delta_m", "spill_level_m", "prominence_m", "dissection_index",
        "marine_province",
    )  # fmt: skip
    project.cache["v46_terrain"] = {
        "layers": {k: v for k, v in layers.items() if k not in pre + post},
        "plans": {k: v for k, v in doc["plans"].items() if k in ("basins", "ridge_pieces")},
        "summary": {k: v for k, v in doc["summary"].items() if k in ("sea_floor", "basins")},
        "timings": {},
    }


def _load_hydro(project: Project, path: Path, manifest: dict, config, verify) -> None:
    """Attach stored v4 data, or say why not. Never silently stale."""
    from .pipeline_v4 import upstream_keys, v4_stage_keys

    v4 = manifest.get("v4")
    if not v4 or project.hist is None:
        return
    if not all((path / name).exists() for name in V4_DATA_FILES):
        project.warnings.append("The manifest lists v4.0 data, but its files are missing.")
        return
    doc = json.loads((path / "hydrology.json").read_text(encoding="utf-8"))
    versions = doc.get("algorithm_versions", {})
    current = {"climate": CLIMATE_ALGORITHM_VERSION, "hydrology": HYDRO_ALGORITHM_VERSION}
    if versions != current:
        project.warnings.append(
            f"The stored climate / hydrology was produced by {versions}, this build uses "
            f"{current}. It was NOT loaded; compute v4 again (v1-v3 are intact)."
        )
        return
    terrain = getattr(project, "terrain", None)
    keys = v4_stage_keys(config, upstream_keys(project.hist, terrain), project.hist.sea_level_m)
    if keys["v4_climate"] != doc["stage_keys"].get("v4_climate"):
        project.warnings.append(
            "The stored climate / hydrology does not match the stored geological epoch or "
            "configuration. It was NOT loaded; compute v4 again."
        )
        return
    with np.load(path / "hydro_fields.npz") as data:
        layers = {name: data[name] for name in data.files}
    layers["elevation_m"] = layers["post_erosion_elevation_m"]
    layers["pre_erosion_elevation_m"] = (
        terrain.layers["surface_elevation_m"]
        if terrain is not None
        else project.hist.layers["elevation_m"]
    )
    hydro = HydroData(
        layers=layers,
        sea_level_m=float(doc["sea_level_m"]),
        epoch_time_ma=float(doc["epoch_time_ma"]),
        latitude=doc["latitude"],
        rivers=doc["river_segments"],
        river_systems=doc["river_systems"],
        lakes=doc["lakes"],
        basins=doc["basins"],
        landmasses=doc["landmasses"],
        budget=doc["sediment_budget_m3"],
        summary=doc["summary"],
        stage_keys=dict(doc["stage_keys"]),
        steps=doc.get("evolution_steps", []),
        warnings=list(doc.get("warnings", [])),
        timings=dict(doc.get("timings_s", {})),
    )
    if verify:
        actual = hydro.content_hashes()
        for key, expected in v4["content_hashes"].items():
            if actual.get(key) != expected:
                raise ProjectIntegrityError(f"array {key} does not match the manifest hash")
    project.hydro = hydro
    hydro.validation = validate_hydro(project)
    # the evolved surface can be reused: a new river threshold re-runs only the network
    if keys["v4_evolution"] == hydro.stage_keys.get("v4_evolution"):
        evolved = layers["post_erosion_elevation_m"].astype(np.float64)
        if terrain is not None and "canyon_fan_delta_m" in terrain.layers:
            # the v4.6 canyons and fans were applied after the evolution: take them off
            evolved = evolved - terrain.layers["canyon_fan_delta_m"]
        project.cache["v4_evolution"] = {
            "elevation": evolved.astype(np.float32).astype(np.float64).ravel(),
            "erosion_total_m": layers["v4_erosion_m"].astype(np.float64).ravel(),
            "deposition_total_m": layers["v4_deposition_m"].astype(np.float64).ravel(),
            "sediment_thickness_m": layers["sediment_thickness_m"].astype(np.float64).ravel(),
            "sediment_flux_m3_yr": layers["sediment_flux_m3_yr"].astype(np.float64).ravel(),
            "sediment_delivery_m3_yr": layers["sediment_delivery_m3_yr"].astype(np.float64).ravel(),
            "budget": hydro.budget,
            "steps": hydro.steps,
        }
