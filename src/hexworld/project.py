"""The in-memory project: every stage result plus query helpers."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field

import numpy as np

from .config import Config
from .geometry import Bounds, HexLayout, WorldGeometry
from .stages.boundaries import TYPE_NAMES, BoundaryEdges
from .stages.crust import CRUST_NAMES, CrustModel
from .stages.motion import PlateMotion
from .stages.plates import PlateModel

#: Layer name -> (dtype, unit, description). The contract for future modules.
LAYER_SPECS: dict[str, tuple[str, str, str]] = {
    "plate_id": ("int32", "id", "ID of the lithospheric plate owning the hex"),
    "crust_type": ("uint8", "enum", "0 oceanic, 1 continental (not a land/sea mask)"),
    "continentality_km": ("float32", "km", "signed pseudo-distance to the crust-type border"),
    "velocity_x": ("float32", "mm/yr", "plate velocity at the hex centre, +x east"),
    "velocity_y": ("float32", "mm/yr", "plate velocity at the hex centre, +y south"),
    "boundary_type": ("uint8", "enum", "0 none,1 convergent,2 divergent,3 transform,4 quiet"),
    "boundary_distance_km": ("float32", "km", "distance to the nearest plate boundary"),
    "uplift": ("float32", "0..1", "diagnostic uplift potential"),
    "rifting": ("float32", "0..1", "diagnostic rifting potential"),
    "subduction": ("float32", "0..1", "diagnostic subduction potential"),
    "volcanism": ("float32", "0..1", "diagnostic volcanism potential"),
    "seismicity": ("float32", "0..1", "diagnostic seismicity potential"),
}


#: v2.0 layers (stored in continental_fields.npz).
GEO_LAYER_SPECS: dict[str, tuple[str, str, str]] = {
    "continental_fraction": ("float32", "0..1", "continuous share of continental crust (v2)"),
    "continental_margin_km": ("float32", "km", "signed distance-like field to the crust edge"),
    "continental_block_id": ("int32", "id", "connected continental crust block, -1 = none"),
    "macro_elevation_m": ("float32", "m", "absolute macro-elevation in the model datum"),
    "tectonic_relief_m": ("float32", "m", "tectonic part of the macro-elevation (uplift mask)"),
    "land_mask": ("bool", "bool", "macro_elevation_m > sea_level_m"),
    "landmass_id": ("int32", "id", "connected land above sea level, -1 = under water"),
    "shelf_mask": ("bool", "bool", "coarse submerged continental shelf"),
}


@dataclass
class GeoData:
    """Results of the v2.0 stages: continental blocks, macro relief, sea and land."""

    layers: dict[str, np.ndarray]
    sea_level_m: float
    blocks: list[dict]
    landmasses: list[dict]
    summary: dict
    stage_keys: dict[str, str]
    #: World-level rasters over the model area (elevation, continentality) + their grid.
    world: dict = field(default_factory=dict, repr=False)
    warnings: list[str] = field(default_factory=list)
    validation: dict = field(default_factory=dict)
    timings: dict[str, float] = field(default_factory=dict)

    def content_hashes(self) -> dict[str, str]:
        out = {f"continental/{k}": array_hash(v) for k, v in self.layers.items()}
        out["continental/world_elevation_m"] = array_hash(self.world["elevation_m"])
        out["continental/world_continentality_km"] = array_hash(self.world["continentality_km"])
        return out

    def geo_hash(self) -> str:
        h = hashlib.sha256()
        h.update(repr(float(self.sea_level_m)).encode())
        for key, value in sorted(self.content_hashes().items()):
            h.update(f"{key}={value};".encode())
        return h.hexdigest()


def array_hash(arr: np.ndarray) -> str:
    """SHA-256 of dtype + shape + C-order bytes (independent of file container)."""
    arr = np.ascontiguousarray(arr)
    h = hashlib.sha256()
    h.update(arr.dtype.str.encode())
    h.update(repr(arr.shape).encode())
    h.update(arr.tobytes())
    return h.hexdigest()


@dataclass
class Project:
    config: Config
    geometry: WorldGeometry
    world_id: str
    plates: list[dict]
    layers: dict[str, np.ndarray]
    boundaries: BoundaryEdges
    boundary_summary: list[dict]
    motion: PlateMotion
    plate_model: PlateModel | None = None
    crust_model: CrustModel | None = None
    stage_keys: dict[str, str] = field(default_factory=dict)
    validation: dict = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    timings: dict[str, float] = field(default_factory=dict)
    stats: dict = field(default_factory=dict)
    #: v2.0 data; ``None`` for a project that only has the v1.0 tectonic stage.
    geo: GeoData | None = None
    #: Stage objects kept for cheap partial regeneration; never saved.
    cache: dict = field(default_factory=dict, repr=False)

    # -- basic accessors ---------------------------------------------------
    @property
    def layout(self) -> HexLayout:
        return self.geometry.layout

    @property
    def layer_names(self) -> list[str]:
        return list(self.layers) + (list(self.geo.layers) if self.geo else [])

    @property
    def has_geo(self) -> bool:
        return self.geo is not None

    def layer(self, name: str) -> np.ndarray:
        """Hex layer as a ``(height, width)`` array indexed ``[row, col]``."""
        if name in self.layers:
            return self.layers[name]
        if self.geo is not None and name in self.geo.layers:
            return self.geo.layers[name]
        if name in GEO_LAYER_SPECS:
            raise KeyError(
                f"Layer {name!r} belongs to the v2.0 stage, which has not been generated "
                "for this project yet"
            )
        raise KeyError(f"Unknown layer {name!r}; available: {self.layer_names}")

    def plate(self, plate_id: int) -> dict:
        return self.plates[int(plate_id)]

    def visible_plates(self) -> list[dict]:
        return [p for p in self.plates if p["visible_hexes"] > 0]

    def hex_info(self, col: int, row: int) -> dict:
        """All values of one hex, with enums decoded."""
        if not self.layout.inside(col, row):
            raise IndexError(
                f"hex ({col}, {row}) is outside the {self.layout.width}x{self.layout.height} grid"
            )
        x, y = self.layout.center(col, row)
        info: dict = {"col": int(col), "row": int(row), "x_km": float(x), "y_km": float(y)}
        for name, arr in self.layers.items():
            v = arr[row, col]
            info[name] = int(v) if np.issubdtype(arr.dtype, np.integer) else float(v)
        info["crust"] = CRUST_NAMES[info["crust_type"]]
        info["boundary"] = TYPE_NAMES[info["boundary_type"]]
        info["plate_name"] = self.plates[info["plate_id"]]["name"]
        if self.geo is not None:
            for name, arr in self.geo.layers.items():
                v = arr[row, col]
                if arr.dtype == np.bool_:
                    info[name] = bool(v)
                else:
                    info[name] = int(v) if np.issubdtype(arr.dtype, np.integer) else float(v)
            sea = self.geo.sea_level_m
            info["sea_level_m"] = sea
            info["height_above_sea_m"] = max(0.0, info["macro_elevation_m"] - sea)
            info["depth_m"] = max(0.0, sea - info["macro_elevation_m"])
        return info

    # -- world-space queries (resolution independent) -----------------------
    def _require_models(self):
        if self.plate_model is None or self.crust_model is None:
            raise RuntimeError(
                "World models are unavailable (project saved by an incompatible "
                "generator version); only stored hex layers can be queried."
            )

    def sample_world(self, x, y) -> dict[str, np.ndarray]:
        """Geology at arbitrary world point(s) in km, straight from the models."""
        self._require_models()
        pid = self.plate_model.plate_at(x, y)
        cont = self.crust_model.continentality(x, y)
        vx, vy = self.motion.velocity_at(pid, x, y)
        return {
            "plate_id": pid,
            "crust_type": (cont > 0).astype(np.uint8),
            "continentality_km": cont,
            "velocity_x": vx,
            "velocity_y": vy,
        }

    def sample_region(self, bounds: Bounds | list[float], hex_spacing_km: float):
        """Sample a world rectangle on a fresh hex grid of any resolution.

        Returns ``(layout, layers)``; the stored project is not modified.
        """
        if not isinstance(bounds, Bounds):
            bounds = Bounds(*bounds)
        radius = hex_spacing_km / np.sqrt(3.0)
        width = max(2, int((bounds.width / radius - 2.0) / 1.5) + 1)
        height = max(2, int(bounds.height / hex_spacing_km - 0.5))
        layout = HexLayout(width, height, float(hex_spacing_km), bounds.xmin, bounds.ymin)
        x, y = layout.centers()
        return layout, self.sample_world(x, y)

    # -- hashes ------------------------------------------------------------
    def content_hashes(self) -> dict[str, str]:
        out = {f"fields/{k}": array_hash(v) for k, v in self.layers.items()}
        out.update({f"boundaries/{k}": array_hash(v) for k, v in self.boundaries.arrays().items()})
        return out

    def geology_hash(self) -> str:
        h = hashlib.sha256()
        for key, value in sorted(self.content_hashes().items()):
            h.update(f"{key}={value};".encode())
        return h.hexdigest()
