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


#: v3.0 layers of the selected epoch (stored in history_fields.npz).
HIST_LAYER_SPECS: dict[str, tuple[str, str, str]] = {
    "elevation_m": ("float32", "m", "final surface of the selected epoch (model datum)"),
    "delta_elevation_m": ("float32", "m", "elevation_m minus the v2 baseline macro elevation"),
    "plate_id_epoch": ("int32", "id", "plate owning the hex at the selected epoch"),
    "new_crust": ("bool", "bool", "crust created during the history (rift or new ocean floor)"),
    "boundary_class": ("uint8", "enum", "0 none,1 collision,2 subduction,3 divergent,4 other"),
    "continental_fraction": ("float32", "0..1", "continental crust share at the epoch"),
    "crust_age_ma": ("float32", "Ma", "age of the crust at the epoch"),
    "rock_type": ("uint8", "enum", "see history.json -> rock_types"),
    "rock_resistance": ("float32", "0..1", "resistance to erosion"),
    "uplift_rate_mm_yr": ("float32", "mm/yr", "tectonic vertical rate at the epoch"),
    "uplift_total_m": ("float32", "m", "net tectonic vertical displacement since t = 0"),
    "erosion_total_m": ("float32", "m", "rock removed since t = 0 (>= 0)"),
    "sediment_thickness_m": ("float32", "m", "sediment deposited since t = 0 (>= 0)"),
    "mountain_system_id": ("int32", "id", "mountain system, -1 = none"),
    "mountain_class": (
        "uint8",
        "enum",
        "0 none,1 crest,2 slope,3 pass,4 valley,5 foothill,6 volcano",
    ),
    "orogen_age_ma": ("float32", "Ma", "time since the last mountain-building, NaN = never"),
    "last_tectonic_ma": ("float32", "Ma", "time since the last tectonic change, NaN = never"),
    "volcanic_potential": ("float32", "0..1", "where magmatism is plausible; not a probability"),
    "volcano_id": ("int32", "id", "volcanic centre covering the hex, -1 = none"),
    "volcanic_age_ma": ("float32", "Ma", "age of that centre, NaN = none"),
    "slope_m_per_km": ("float32", "m/km", "steepest slope to a neighbouring hex"),
    "local_relief_m": ("float32", "m", "height range of the continuous surface within the hex"),
    "land_mask": ("bool", "bool", "elevation_m > sea_level_m"),
    "landmass_id": ("int32", "id", "connected land of the epoch, -1 = under water"),
    "shelf_mask": ("bool", "bool", "submerged continental crust shallower than the shelf limit"),
    "coast_type": ("uint8", "enum", "coast type of a land hex at the sea, 0 = not coast"),
    "wave_exposure": ("float32", "0..1", "open water and storms facing the coast (proxy)"),
    "tidal_range_m": ("float32", "m", "mean tidal range at the coast (proxy)"),
    "shelf_width_km": ("float32", "km", "distance from the coast to water deeper than 200 m"),
    "sea_surface_temperature_c": ("float32", "degC", "mean temperature of the adjacent sea"),
    "lgm_ice_mask": ("bool", "bool", "under ice at the last glacial maximum (estimate)"),
}
ROCK_TYPES = (
    "oceanic_basalt",
    "craton_crystalline",
    "platform_sedimentary",
    "orogenic_metamorphic",
    "volcanic",
    "unconsolidated_sediment",
)


@dataclass
class HistoryData:
    """Results of the v3.0 stages for one selected epoch, plus the whole history record."""

    layers: dict[str, np.ndarray]
    epoch_time_ma: float
    #: Times (Ma) of every stored snapshot: initial, checkpoints, final.
    epochs: list[float]
    duration_ma: float
    sea_level_m: float
    events: list[dict]
    plan: dict
    volcanoes: list[dict]
    hotspots: list[dict]
    landmasses: list[dict]
    summary: dict
    stage_keys: dict[str, str]
    #: World rasters of the selected epoch on the history grid + the grid itself.
    world: dict = field(default_factory=dict, repr=False)
    #: Material-frame state of every stored epoch: {time: {name: array}}.
    checkpoints: dict = field(default_factory=dict, repr=False)
    balance: list[dict] = field(default_factory=list, repr=False)
    warnings: list[str] = field(default_factory=list)
    validation: dict = field(default_factory=dict)
    timings: dict[str, float] = field(default_factory=dict)

    @property
    def is_final(self) -> bool:
        return abs(self.epoch_time_ma - self.duration_ma) < 1e-9

    def content_hashes(self) -> dict[str, str]:
        out = {f"history/{k}": array_hash(v) for k, v in self.layers.items()}
        for t, state in sorted(self.checkpoints.items()):
            for name, arr in sorted(state.items()):
                out[f"checkpoint/{t:g}/{name}"] = array_hash(arr)
        return out

    def hist_hash(self) -> str:
        h = hashlib.sha256()
        h.update(repr((float(self.epoch_time_ma), float(self.sea_level_m))).encode())
        for key, value in sorted(self.content_hashes().items()):
            h.update(f"{key}={value};".encode())
        h.update(repr([e["id"] for e in self.events]).encode())
        return h.hexdigest()


#: v4.0 layers (stored in hydro_fields.npz). Names shared with older stages override them
#: in ``Project.layer``; the older values stay reachable through ``baseline_layer``.
HYDRO_LAYER_SPECS: dict[str, tuple[str, str, str]] = {
    "elevation_m": ("float32", "m", "final surface after v4 erosion (= post_erosion_elevation_m)"),
    "post_erosion_elevation_m": ("float32", "m", "surface after climate-driven erosion"),
    "pre_erosion_elevation_m": ("float32", "m", "v3 surface of the selected epoch (input)"),
    "v4_delta_elevation_m": ("float32", "m", "post_erosion minus pre_erosion elevation"),
    "latitude_deg": ("float32", "deg", "conditional latitude, north positive"),
    "temperature_mean_c": ("float32", "degC", "annual mean air temperature"),
    "temperature_seasonal_c": ("float32", "degC", "half of the summer-winter range"),
    "wind_u_ms": ("float32", "m/s", "prevailing wind, +x east"),
    "wind_v_ms": ("float32", "m/s", "prevailing wind, +y south"),
    "precipitation_mm_yr": ("float32", "mm/yr", "annual precipitation"),
    "potential_evapotranspiration_mm_yr": ("float32", "mm/yr", "potential evapotranspiration"),
    "actual_evapotranspiration_mm_yr": ("float32", "mm/yr", "actual evapotranspiration (Budyko)"),
    "humidity_index": ("float32", "P/PET", "aridity index, capped at 5"),
    "snow_fraction": ("float32", "0..1", "share of precipitation falling as snow"),
    "snow_storage_mm": ("float32", "mm", "peak winter snow water equivalent (approximate)"),
    "glacial_potential": ("float32", "0..1", "climatic potential for perennial ice"),
    "infiltration_loss_mm_yr": ("float32", "mm/yr", "water lost to deep groundwater"),
    "runoff_mm_yr": ("float32", "mm/yr", "surface runoff P - AET - deep infiltration"),
    "flow_receiver": ("int64", "index", "flat index of the downstream hex (itself = outlet)"),
    "flow_direction": ("uint8", "enum", "0..5 hex direction to the receiver, 255 = outlet"),
    "flow_accumulation_area_km2": ("float32", "km2", "upstream land area including the hex"),
    "discharge_m3_s": ("float32", "m3/s", "annual mean discharge leaving the hex"),
    "external_inflow_m3_s": ("float32", "m3/s", "water entering the window from outside"),
    "basin_id": ("int32", "id", "drainage basin, -1 = sea"),
    "lake_id": ("int32", "id", "lake, -1 = none"),
    "lake_level_m": ("float32", "m", "lake water level (NaN = no lake)"),
    "lake_depth_m": ("float32", "m", "lake depth (NaN = no lake)"),
    "endorheic_mask": ("bool", "bool", "drains to a closed (terminal) lake"),
    "channel_loss_m3_s": ("float32", "m3/s", "transmission loss of the channel in the hex"),
    "outflow_share": ("float32", "0..1", "share of the hex's water reaching sea or window edge"),
    "no_outflow_mask": ("bool", "bool", "less than 10 % of the water reaches an outlet"),
    "river_mask": (
        "bool",
        "bool",
        "river: discharge, upstream area and system basin reach the hydrology thresholds",
    ),
    "strahler_order": ("uint8", "order", "Strahler order of river hexes, 0 = no river"),
    "river_mouth": ("bool", "bool", "last hex of a river before sea, lake or window edge"),
    "coastal_discharge_m3_s": ("float32", "m3/s", "discharge delivered to the sea at mouths"),
    "sediment_delivery_m3_yr": ("float32", "m3/yr", "sediment delivered to the sea at mouths"),
    "erosion_rate_m_ma": ("float32", "m/Ma", "mean v4 erosion rate"),
    "v4_erosion_m": ("float32", "m", "rock and sediment removed during v4 (>= 0)"),
    "v4_deposition_m": ("float32", "m", "material deposited during v4 (>= 0)"),
    "sediment_thickness_m": ("float32", "m", "loose sediment cover after v4 (>= 0)"),
    "sediment_flux_m3_yr": ("float32", "m3/yr", "sediment passing through the hex (last step)"),
    "slope_m_per_km": ("float32", "m/km", "steepest slope to a neighbouring hex"),
    "land_mask": ("bool", "bool", "elevation_m > sea_level_m (final surface)"),
    "landmass_id": ("int32", "id", "connected land of the final surface"),
    "shelf_mask": ("bool", "bool", "submerged continental crust shallower than the shelf limit"),
    "coast_type": ("uint8", "enum", "coast type of a land hex at the sea, 0 = not coast"),
    "wave_exposure": ("float32", "0..1", "open water and storms facing the coast (proxy)"),
    "tidal_range_m": ("float32", "m", "mean tidal range at the coast (proxy)"),
    "shelf_width_km": ("float32", "km", "distance from the coast to water deeper than 200 m"),
    "sea_surface_temperature_c": ("float32", "degC", "mean temperature of the adjacent sea"),
    "lgm_ice_mask": ("bool", "bool", "under ice at the last glacial maximum (estimate)"),
}


@dataclass
class HydroData:
    """Results of v4.0: climate, water balance, rivers, lakes, landscape evolution."""

    layers: dict[str, np.ndarray]
    sea_level_m: float
    epoch_time_ma: float
    latitude: dict
    rivers: list[dict]  # segments
    river_systems: list[dict]
    lakes: list[dict]
    basins: list[dict]
    landmasses: list[dict]
    budget: dict
    summary: dict
    stage_keys: dict[str, str]
    steps: list[dict] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    validation: dict = field(default_factory=dict)
    timings: dict[str, float] = field(default_factory=dict)

    def content_hashes(self) -> dict[str, str]:
        return {
            f"hydro/{k}": array_hash(v)
            for k, v in self.layers.items()
            if k not in ("pre_erosion_elevation_m", "elevation_m")
        }

    def hydro_hash(self) -> str:
        h = hashlib.sha256()
        h.update(repr(float(self.sea_level_m)).encode())
        for key, value in sorted(self.content_hashes().items()):
            h.update(f"{key}={value};".encode())
        h.update(repr([(r["id"], r["discharge_m3_s"]) for r in self.rivers]).encode())
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
    #: v3.0 data (geological history, mountains, volcanism); ``None`` if not generated.
    hist: HistoryData | None = None
    #: v4.0 data (climate, hydrology, erosion); ``None`` if not generated.
    hydro: HydroData | None = None
    #: Stage objects kept for cheap partial regeneration; never saved.
    cache: dict = field(default_factory=dict, repr=False)

    # -- basic accessors ---------------------------------------------------
    @property
    def layout(self) -> HexLayout:
        return self.geometry.layout

    @property
    def layer_names(self) -> list[str]:
        names = list(self.layers) + (list(self.geo.layers) if self.geo else [])
        if self.hist is not None:
            names += [n for n in self.hist.layers if n not in names]
        if self.hydro is not None:
            names += [n for n in self.hydro.layers if n not in names]
        return names

    @property
    def has_history(self) -> bool:
        return self.hist is not None

    def stage_layer(self, stage: str, name: str) -> np.ndarray:
        """A layer of one stage ("v1", "v2", "v3", "v4") regardless of overrides."""
        source = {
            "v1": self.layers,
            "v2": self.geo.layers if self.geo else {},
            "v3": self.hist.layers if self.hist else {},
            "v4": self.hydro.layers if self.hydro else {},
        }[stage]
        if name not in source:
            raise KeyError(f"No layer {name!r} in stage {stage}")
        return source[name]

    def baseline_layer(self, name: str) -> np.ndarray:
        """A v1/v2 layer of the baseline epoch, never the v3 version of the same name."""
        if name in self.layers:
            return self.layers[name]
        if self.geo is not None and name in self.geo.layers:
            return self.geo.layers[name]
        raise KeyError(f"No baseline layer {name!r}")

    @property
    def current_sea_level_m(self) -> float | None:
        if self.hist is not None:
            return self.hist.sea_level_m
        return self.geo.sea_level_m if self.geo is not None else None

    @property
    def has_geo(self) -> bool:
        return self.geo is not None

    def layer(self, name: str) -> np.ndarray:
        """Hex layer as a ``(height, width)`` array indexed ``[row, col]``."""
        if name == "base_macro_elevation_m" and self.geo is not None:
            return self.geo.layers["macro_elevation_m"]  # a reference, not a copy
        if self.hydro is not None and name in self.hydro.layers:
            return self.hydro.layers[name]  # v4 (after erosion) wins over everything older
        if self.hist is not None and name in self.hist.layers:
            return self.hist.layers[name]  # the selected v3 epoch wins over the baseline
        if name in self.layers:
            return self.layers[name]
        if self.geo is not None and name in self.geo.layers:
            return self.geo.layers[name]
        if (
            name in HYDRO_LAYER_SPECS
            and name not in HIST_LAYER_SPECS
            and name not in GEO_LAYER_SPECS
        ):
            raise KeyError(
                f"Layer {name!r} belongs to the v4.0 stage (climate, hydrology, erosion), "
                "which has not been computed for this project yet"
            )
        if name in HIST_LAYER_SPECS and name not in GEO_LAYER_SPECS:
            raise KeyError(
                f"Layer {name!r} belongs to the v3.0 stage, which has not been generated "
                "for this project yet"
            )
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
        if self.hist is not None:
            # baseline values keep explicit names; unprefixed ones describe the epoch
            for name in ("land_mask", "landmass_id", "shelf_mask", "continental_fraction"):
                if name in info:
                    info[f"baseline_{name}"] = info[name]
            for name, arr in self.hist.layers.items():
                v = arr[row, col]
                if arr.dtype == np.bool_:
                    info[name] = bool(v)
                elif np.issubdtype(arr.dtype, np.integer):
                    info[name] = int(v)
                else:
                    info[name] = None if np.isnan(v) else float(v)
            sea = self.hist.sea_level_m
            info["epoch_time_ma"] = self.hist.epoch_time_ma
            info["sea_level_m"] = sea
            info["height_above_sea_m"] = max(0.0, info["elevation_m"] - sea)
            info["depth_m"] = max(0.0, sea - info["elevation_m"])
            info["rock"] = ROCK_TYPES[info["rock_type"]]
        if self.hydro is not None:
            for name in ("land_mask", "landmass_id", "shelf_mask", "sediment_thickness_m"):
                if name in info:
                    info[f"v3_{name}"] = info[name]
            info["v3_elevation_m"] = info.get("elevation_m")
            for name, arr in self.hydro.layers.items():
                v = arr[row, col]
                if arr.dtype == np.bool_:
                    info[name] = bool(v)
                elif np.issubdtype(arr.dtype, np.integer):
                    info[name] = int(v)
                else:
                    info[name] = None if np.isnan(v) else float(v)
            sea = self.hydro.sea_level_m
            info["sea_level_m"] = sea
            info["height_above_sea_m"] = max(0.0, info["elevation_m"] - sea)
            info["depth_m"] = max(0.0, sea - info["elevation_m"])
            info["v4_source"] = (
                f"v4 after erosion of the t = {self.hydro.epoch_time_ma:g} Ma surface"
            )
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
