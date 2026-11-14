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
from .project import LAYER_SPECS, Project
from .stages.boundaries import TYPE_NAMES, BoundaryEdges
from .stages.crust import CRUST_NAMES, build_crust_model
from .stages.motion import PlateMotion
from .stages.plates import build_plate_model
from .stages.reference import build_reference_grid
from .stages.validate import validate_project
from .version import (
    ALGORITHM_VERSION,
    COORDINATE_VERSION,
    GENERATOR_VERSION,
    PRNG_VERSION,
    SCHEMA_VERSION,
)

DATA_FILES = ("config.json", "plates.json", "fields.npz", "boundaries.npz", "boundaries.json")

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


def save_project(project: Project, path: str | Path, overwrite: bool = True) -> Path:
    """Save ``project`` into directory ``path``.

    If the directory already holds the very same geology (equal geology hash),
    the data files are left untouched - re-rendering never rewrites source data.
    """
    path = Path(path)
    manifest_path = path / "manifest.json"
    geology_hash = project.geology_hash()
    if manifest_path.exists():
        try:
            existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            existing = {}
        if existing.get("geology_hash") == geology_hash and all(
            (path / name).exists() for name in DATA_FILES
        ):
            return path
        if not overwrite:
            raise FileExistsError(f"{path} already contains a different project")
    path.mkdir(parents=True, exist_ok=True)

    _write_json(path / "config.json", project.config.to_dict())
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
        "file_hashes": {name: _file_hash(path / name) for name in DATA_FILES},
    }
    _write_json(manifest_path, manifest)
    return path


def load_project(path: str | Path, verify: bool = True) -> Project:
    """Load a project directory, checking control hashes unless ``verify=False``."""
    path = Path(path)
    manifest = json.loads((path / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("schema_version") != SCHEMA_VERSION:
        raise ProjectIntegrityError(
            f"Unsupported schema {manifest.get('schema_version')!r}, expected {SCHEMA_VERSION!r}"
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
    return project
