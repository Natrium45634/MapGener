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
from .project import GEO_LAYER_SPECS, LAYER_SPECS, GeoData, Project
from .stages.boundaries import TYPE_NAMES, BoundaryEdges
from .stages.crust import CRUST_NAMES, build_crust_model
from .stages.motion import PlateMotion
from .stages.plates import build_plate_model
from .stages.reference import build_reference_grid
from .stages.validate import validate_geo, validate_project
from .version import (
    ALGORITHM_VERSION,
    CONTINENTS_ALGORITHM_VERSION,
    COORDINATE_VERSION,
    GENERATOR_VERSION,
    PRNG_VERSION,
    SCHEMA_VERSION,
    SUPPORTED_SCHEMAS,
)

DATA_FILES = ("config.json", "plates.json", "fields.npz", "boundaries.npz", "boundaries.json")
#: v1.0 source data that a later stage must never rewrite.
V1_DATA_FILES = ("plates.json", "fields.npz", "boundaries.npz", "boundaries.json")
#: v2.0 data lives in its own files.
V2_DATA_FILES = ("continental_fields.npz", "continents.json")

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
    h = hashlib.sha256()
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


def save_project(project: Project, path: str | Path, overwrite: bool = True) -> Path:
    """Save ``project`` into directory ``path``.

    * v1.0 source data (plates, fields, boundaries) is written only when the
      directory does not already hold the very same geology - adding v2.0 data
      or re-rendering never rewrites it.
    * v2.0 data goes to its own files (``continental_fields.npz``,
      ``continents.json``).
    * When a ``hexworld-project-1`` directory is extended, its original
      ``config.json`` and ``manifest.json`` are kept as ``*.v1.json`` copies.
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
    if same_geology:
        stored_geo = (existing.get("v2") or {}).get("geo_hash")
        if project.geo is None or stored_geo == project.geo.geo_hash():
            if (
                json.loads((path / "config.json").read_text(encoding="utf-8"))
                == (project.config.to_dict())
                or project.geo is None
            ):
                return path  # nothing new to store
        if existing.get("schema_version") == "hexworld-project-1":
            for name in ("config.json", "manifest.json"):
                backup = path / name.replace(".json", ".v1.json")
                if not backup.exists():
                    backup.write_bytes((path / name).read_bytes())
    path.mkdir(parents=True, exist_ok=True)

    if not same_geology:
        _write_v1_data(project, path)
        for name in V2_DATA_FILES:  # v2 data of another world must not linger
            (path / name).unlink(missing_ok=True)
    _write_json(path / "config.json", project.config.to_dict())
    files = list(DATA_FILES)
    v2_manifest = None
    if project.geo is not None:
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

    config = Config.from_dict(json.loads((path / "config.json").read_text(encoding="utf-8")))
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
        if plate_model.model_hash() == saved_p and crust_model.model_hash() == saved_c:
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
    return project
