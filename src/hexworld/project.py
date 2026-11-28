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
    "volcano_summit_m": (
        "float32",
        "m",
        "summit of that volcano (hex values average cones), NaN = none",
    ),
    "slope_m_per_km": ("float32", "m/km", "steepest slope to a neighbouring hex"),
    "local_relief_m": ("float32", "m", "height range of the continuous surface within the hex"),
    "material_x_km": ("float32", "km", "position of the rock at t = 0 (plate frame), x"),
    "material_y_km": ("float32", "km", "position of the rock at t = 0 (plate frame), y"),
    "continentality_km": (
        "float32",
        "km",
        "signed distance to the continental crust edge at t = 0, NaN = new crust",
    ),
    "base_component_m": ("float32", "m", "v2 baseline part of elevation_m (new floor in gaps)"),
    "trench_relief_m": ("float32", "m", "instantaneous trench / outer-rise part of elevation_m"),
    "trench_distance_km": ("float32", "km", "distance to the nearest trench cell (1e4 = none)"),
    "trench_side": ("int8", "enum", "+1 subducting plate, -1 overriding plate, 0 none"),
    "trench_rate_mm_yr": ("float32", "mm/yr", "convergence rate of that trench"),
    "spreading_axis_distance_km": ("float32", "km", "distance to the spreading axis cells"),
    "land_mask": ("bool", "bool", "not connected to the world ocean (v4.6; was z > sea)"),
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
    "pre_erosion_elevation_m": ("float32", "m", "input surface of v4 (v4.6 terrain or v3 epoch)"),
    "v4_delta_elevation_m": ("float32", "m", "post_erosion minus pre_erosion elevation"),
    "latitude_deg": ("float32", "deg", "conditional latitude, north positive"),
    "temperature_mean_c": ("float32", "degC", "annual mean air temperature"),
    "temperature_seasonal_c": ("float32", "degC", "half of the summer-winter range"),
    "temperature_djf_c": ("float32", "degC", "mean air temperature, December-February"),
    "temperature_jja_c": ("float32", "degC", "mean air temperature, June-August"),
    "wind_u_ms": ("float32", "m/s", "prevailing wind, +x east"),
    "wind_v_ms": ("float32", "m/s", "prevailing wind, +y south"),
    "precipitation_mm_yr": ("float32", "mm/yr", "annual precipitation (sum of the seasons)"),
    "precipitation_djf_mm": ("float32", "mm", "precipitation of December-February"),
    "precipitation_jja_mm": ("float32", "mm", "precipitation of June-August"),
    "potential_evapotranspiration_mm_yr": ("float32", "mm/yr", "potential evapotranspiration"),
    "actual_evapotranspiration_mm_yr": (
        "float32",
        "mm/yr",
        "actual evapotranspiration (seasonal soil bucket; open water = PET)",
    ),
    "humidity_index": ("float32", "P/PET", "aridity index, capped at 5"),
    "snow_fraction": ("float32", "0..1", "share of precipitation falling as snow"),
    "snow_storage_mm": ("float32", "mm", "seasonal range of the snowpack, water equivalent"),
    "snowmelt_share": ("float32", "0..1", "share of the water supply that comes as melt / ice"),
    "glacial_potential": ("float32", "0..1", "climatic potential for perennial ice"),
    "infiltration_loss_mm_yr": ("float32", "mm/yr", "water lost to deep groundwater"),
    "runoff_mm_yr": (
        "float32",
        "mm/yr",
        "runoff to rivers: quick flow + baseflow + glacier discharge (P - AET - deep - dS)",
    ),
    "baseflow_index": ("float32", "0..1", "share of the runoff that comes from groundwater"),
    "runoff_peak_season": ("uint8", "enum", "peak runoff season 0 DJF 1 MAM 2 JJA 3 SON, 255 none"),
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
    "river_width_m": ("float32", "m", "channel width W = 5 Q^0.5 on river hexes, 0 elsewhere"),
    "river_depth_m": ("float32", "m", "channel depth D = 0.3 Q^0.4 on river hexes, 0 elsewhere"),
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
    "land_mask": ("bool", "bool", "not world ocean (v4.6 connectivity; was z > sea)"),
    "landmass_id": ("int32", "id", "connected land of the final surface"),
    "ocean_mask": ("bool", "bool", "below the sea level AND connected to the world ocean"),
    "inland_water_mask": ("bool", "bool", "lake / inland sea with its own level"),
    "dry_land_mask": ("bool", "bool", "neither ocean nor inland water"),
    "below_sea_level_land_mask": ("bool", "bool", "dry land lower than the sea level"),
    "above_sea_level_mask": ("bool", "bool", "legacy diagnostic: elevation_m > sea_level_m"),
    "lake_surface_level_m": ("float32", "m", "free surface of a lake, NaN elsewhere"),
    "water_surface_level_m": ("float32", "m", "sea level / lake level / NaN on dry land"),
    "water_depth_m": ("float32", "m", "water surface minus solid surface, 0 on dry land"),
    "shelf_mask": ("bool", "bool", "submerged continental crust shallower than the shelf limit"),
    "coast_type": ("uint8", "enum", "coast type of a land hex at the sea, 0 = not coast"),
    "wave_exposure": ("float32", "0..1", "open water and storms facing the coast (proxy)"),
    "tidal_range_m": ("float32", "m", "mean tidal range at the coast (proxy)"),
    "shelf_width_km": ("float32", "km", "distance from the coast to water deeper than 200 m"),
    "sea_surface_temperature_c": ("float32", "degC", "mean temperature of the adjacent sea"),
    "lgm_ice_mask": ("bool", "bool", "under ice at the last glacial maximum (estimate)"),
}


#: v4.6 layers (terrain_fields.npz): unit, description.
TERRAIN_LAYER_SPECS: dict[str, tuple[str, str]] = {
    "surface_elevation_m": ("m", "v4.6 solid surface, input of v4 (sediment included)"),
    "bedrock_elevation_m": ("m", "surface minus sediment_thickness_m"),
    "sediment_thickness_m": ("m", "loose cover: v3 sediment + basin fill / marine sediment"),
    "terrain_delta_m": ("m", "surface_elevation_m minus the v3 elevation_m"),
    "marine_floor_mask": ("bool", "sea floor seaward of the continental crust edge"),
    "oceanic_crust_age_ma": ("Ma", "crust age from ridge pieces (approx), NaN off the floor"),
    "basement_depth_m": ("m", "Parsons-Sclater depth of that age, NaN off the floor"),
    "bathymetry_delta_m": ("m", "change of the sea floor against v3"),
    "marine_sediment_thickness_m": ("m", "pelagic + v3 sediment on the sea floor"),
    "abyssal_fabric_m": ("m", "isochron-parallel fabric (hex mean, buried under the rise)"),
    "ridge_piece_id": ("id", "ridge segment that made the crust, -1 = none"),
    "fracture_zone_mask": ("bool", "older side of a ridge-piece age jump"),
    "fracture_zone_offset_ma": ("Ma", "age jump across that fracture zone"),
    "trench_m": ("m", "v4.6 trench / outer-rise relief"),
    "trench_axis_mask": ("bool", "axis of a deep-sea trench"),
    "outer_rise_mask": ("bool", "flexural bulge before a trench"),
    "margin_type": ("enum", "0 none, 1 passive, 2 active, 3 transform"),
    "crust_edge_distance_km": ("km", "signed distance to the continental crust edge"),
    "slope_width_km": ("km", "continental slope width of the local margin"),
    "arc_platform_m": ("m", "island-arc platform added on the overriding oceanic plate"),
    "seamount_field_m": ("m", "relief of the intraplate seamount field (hex mean)"),
    "shelf_highs_m": ("m", "island highs of shallow continental shelves (archipelagos)"),
    "rise_width_km": ("km", "continental rise width of the local margin"),
    "oceanic_plateau_m": ("m", "height of an oceanic plateau"),
    "seamount_type": ("enum", "0 none, 1 seamount, 2 guyot, 3 volcanic island"),
    "spreading_half_rate_mm_yr": ("mm/yr", "half-rate of the ridge piece"),
    "marine_local_relief_m": ("m", "sub-hex range of the abyssal fabric"),
    "basin_subsidence_m": ("m", "bedrock subsidence of a basin (+ down)"),
    "basin_fill_m": ("m", "sediment fill of that subsidence"),
    "basin_id": ("id", "index into terrain.json -> plans.basins, -1 = none"),
    "basin_delta_m": ("m", "surface change from basins (fill and shoulders included)"),
    "bedrock_basin_delta_m": ("m", "bedrock change from basins"),
    "foreland_deflection_m": ("m", "flexural deflection outside the mountain load (+ down)"),
    "ocean_source_mask": ("bool", "hexes of the world ocean on the world raster, below sea"),
    "ocean_mask_pre_v4": ("bool", "world ocean on the v4.6 surface (before v4)"),
    "submarine_canyon_mask": ("bool", "path of a submarine canyon"),
    "canyon_depth_m": ("m", "depth of that canyon (sub-hex feature)"),
    "canyon_incision_m": ("m", "hex-mean incision of the canyon"),
    "submarine_fan_m": ("m", "turbidite fan thickness"),
    "canyon_fan_delta_m": ("m", "change of the final surface by canyons and fans"),
    "spill_level_m": ("m", "priority-flood level: lowest spill of the closed depression"),
    "prominence_m": ("m", "topographic prominence of hex summits (0 elsewhere)"),
    "dissection_index": ("0..1", "heuristic dissection of the mountains [А]"),
    "marine_province": ("enum", "see terrain.json -> province_names"),
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


@dataclass
class EnvData:
    """Results of v4.5: monthly climate, soils, monthly water, wetness, indices."""

    layers: dict[str, np.ndarray]  # (rows, cols)
    monthly: dict[str, np.ndarray]  # (12, rows, cols), month 0 = January
    summary: dict
    methods: dict  # field -> method / provenance / quality
    stage_keys: dict[str, str]
    warnings: list[str] = field(default_factory=list)
    validation: dict = field(default_factory=dict)
    timings: dict[str, float] = field(default_factory=dict)

    def content_hashes(self) -> dict[str, str]:
        out = {f"env/{k}": array_hash(v) for k, v in self.layers.items()}
        out.update({f"env_monthly/{k}": array_hash(v) for k, v in self.monthly.items()})
        return out


@dataclass
class TerrainData:
    """Results of v4.6: the composed solid surface, its plans and the derived water."""

    layers: dict[str, np.ndarray]  # (rows, cols)
    plans: dict  # basins, massifs, sea-floor objects (JSON-able)
    summary: dict
    stage_keys: dict[str, str]
    warnings: list[str] = field(default_factory=list)
    validation: dict = field(default_factory=dict)
    timings: dict[str, float] = field(default_factory=dict)

    def content_hashes(self) -> dict[str, str]:
        return {f"terrain/{k}": array_hash(v) for k, v in self.layers.items()}


@dataclass
class TopoData:
    """Results of v4.7: hex layers, hex monthly maps, terrain units (ragged by hex)."""

    layers: dict[str, np.ndarray]  # (rows, cols)
    monthly: dict[str, np.ndarray]  # (12, rows, cols)
    units: dict[str, np.ndarray]  # (U,), (12, U) or (U, K); CSR by ``hex_offsets``
    summary: dict
    massifs: list[dict]
    provenance: dict
    stage_keys: dict[str, str]
    warnings: list[str] = field(default_factory=list)
    validation: dict = field(default_factory=dict)
    timings: dict[str, float] = field(default_factory=dict)

    def content_hashes(self) -> dict[str, str]:
        out = {f"topo/{k}": array_hash(v) for k, v in self.layers.items()}
        out.update({f"topo_monthly/{k}": array_hash(v) for k, v in self.monthly.items()})
        out.update({f"topo_units/{k}": array_hash(v) for k, v in self.units.items()})
        return out

    def units_of_hex(self, flat_index: int) -> np.ndarray:
        """Indices of the terrain units of one hex (``row * width + col``)."""
        off = self.units["hex_offsets"]
        return np.arange(int(off[flat_index]), int(off[flat_index + 1]))


@dataclass
class EarthData:
    """Results of v5.3a on the hexes: ocean, sea ice, winds aloft, lake ice, realm/cover."""

    layers: dict  # (H, W), (4, H, W), (12, H, W) or (4, L, H, W) arrays
    summary: dict
    provenance: dict
    stage_keys: dict
    warnings: list = field(default_factory=list)
    validation: dict = field(default_factory=dict)
    timings: dict = field(default_factory=dict)

    def content_hashes(self) -> dict:
        return {f"earth/{k}": array_hash(v) for k, v in self.layers.items()}


@dataclass
class AtmosphereData:
    """Results of v5.4 (``AtmosphereNormalsSnapshot`` + optional ``WeatherYearSnapshot``).

    ``layers`` are hex diagnostics (monsoon indices, rain shares, storm risk);
    ``grid_fields`` are the monthly normals on the coarse atmosphere grid (pressure, winds
    on 4 levels, column vapour, IVT, convergence, ITCZ score, rain components), which the
    views and ``wind_at`` sample on demand; ``events`` are sparse weather events.
    """

    layers: dict
    grid_fields: dict
    grid: dict  # x0_km, y0_km, cell_km, shape
    summary: dict
    provenance: dict
    stage_keys: dict
    events: list = field(default_factory=list)
    warnings: list = field(default_factory=list)
    validation: dict = field(default_factory=dict)
    timings: dict = field(default_factory=dict)

    def content_hashes(self) -> dict:
        out = {f"atmosphere/{k}": array_hash(v) for k, v in self.layers.items()}
        out.update({f"atmosphere_grid/{k}": array_hash(v) for k, v in self.grid_fields.items()})
        return out


@dataclass
class VolcanismData:
    """Results of v5.1: hex layers, systems, eruption events and their products."""

    layers: dict  # (H, W) arrays
    systems: list  # dicts (VolcanoSystem.to_dict)
    events: list  # dicts of the explicit events (EruptionEvent.to_dict)
    provinces: list  # dicts
    aggregates: dict  # volcano_id -> totals of the non-explicit events
    summary: dict
    provenance: dict
    stage_keys: dict
    warnings: list = field(default_factory=list)
    validation: dict = field(default_factory=dict)
    timings: dict = field(default_factory=dict)

    def content_hashes(self) -> dict:
        return {f"volcanism/{k}": array_hash(v) for k, v in self.layers.items()}

    def system_at(self, flat_index: int) -> list[dict]:
        """Volcano systems whose vent hex is ``flat_index``."""
        return [s for s in self.systems if s.get("hex_flat") == int(flat_index)]


@dataclass
class ResourceData:
    """Results of v5.2: soil fertility layers, the deposit catalogue and the aggregates.

    ``layers`` are hex fields ((H, W), (3, H, W) per soil layer, (F, H, W) per commodity
    family); ``catalog`` holds the sparse objects: ``systems`` (deposit systems),
    ``bodies`` (parametric ore bodies), ``provinces``, ``eroded`` (released masses of
    fully / partly eroded systems) and the hex index (``hex_offsets``, ``hex_bodies``).
    """

    layers: dict
    catalog: dict
    summary: dict
    provenance: dict
    stage_keys: dict
    warnings: list = field(default_factory=list)
    validation: dict = field(default_factory=dict)
    timings: dict = field(default_factory=dict)

    def content_hashes(self) -> dict:
        out = {f"resources/{k}": array_hash(v) for k, v in self.layers.items()}
        ids = "|".join(f"{s['deposit_id']}:{s['ore_t']:.6g}" for s in self.catalog["systems"])
        out["resources_catalog/systems"] = hashlib.sha256(ids.encode()).hexdigest()
        return out

    def bodies_at(self, flat_index: int) -> list[dict]:
        off = self.catalog["hex_offsets"]
        idx = self.catalog["hex_bodies"][int(off[flat_index]) : int(off[flat_index + 1])]
        return [self.catalog["bodies"][int(i)] for i in idx]

    def system(self, deposit_id: str) -> dict | None:
        for s in self.catalog["systems"]:
            if s["deposit_id"] == deposit_id:
                return s
        return None


@dataclass
class BiomeData:
    """Results of v5.0: hex layers, sparse biome fractions, unit results."""

    layers: dict
    units: dict
    fractions: dict  # CSR: offsets (N + 1), biome (K,) uint16, fraction (K,) float32
    summary: dict
    provenance: dict
    stage_keys: dict
    warnings: list = field(default_factory=list)
    validation: dict = field(default_factory=dict)
    timings: dict = field(default_factory=dict)

    def content_hashes(self) -> dict:
        out = {f"biome/{k}": array_hash(v) for k, v in self.layers.items()}
        out.update({f"biome_units/{k}": array_hash(v) for k, v in self.units.items()})
        out.update({f"biome_fractions/{k}": array_hash(v) for k, v in self.fractions.items()})
        return out

    def breakdown(self, flat_index: int) -> list[tuple[str, float]]:
        """``[(biome_id, whole-hex fraction)]`` of one hex, largest first (sums to 1)."""
        off = self.fractions["offsets"]
        a, b = int(off[flat_index]), int(off[flat_index + 1])
        from .biomes.registry import BIOMES

        rows = [
            (BIOMES[int(k)].id, float(f))
            for k, f in zip(
                self.fractions["biome"][a:b], self.fractions["fraction"][a:b], strict=True
            )
        ]
        return sorted(rows, key=lambda r: -r[1])


def _num(v):
    v = float(v)
    return None if not np.isfinite(v) else round(v, 4)


class _IndexOf(dict):
    def __missing__(self, key):
        from .biomes.registry import INDEX

        self.update(INDEX)
        return INDEX[key]


INDEX_OF = _IndexOf()


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
    #: v4.5 data (months, soils, monthly water, ecological indices); ``None`` if not run.
    env: EnvData | None = None
    #: v4.6 terrain (sea floor, basins, water connectivity, massif graphs).
    terrain: TerrainData | None = None
    #: v4.7 topoclimate (terrain units, local climate, thermal belts); ``None`` if not run.
    topo: TopoData | None = None
    #: v5.0 biomes, natural cover and cryosphere; ``None`` if not generated.
    biomes: BiomeData | None = None
    #: v5.1 volcanism (systems, eruptions, products, ecology forcing); ``None`` if absent.
    volcanism: VolcanismData | None = None
    #: v5.3a Earth system on the hexes (ocean, sea ice, winds aloft); ``None`` if absent.
    earth: EarthData | None = None
    atmosphere: AtmosphereData | None = None
    #: v5.2 soil fertility and geological resources; ``None`` if not generated.
    resources: ResourceData | None = None
    #: TZ 5.5.1 mineral plan (``hexworld.minerals.stage.MineralData``); ``None`` if absent.
    minerals: object | None = None
    #: TZ 5.5 biosphere (``hexworld.biosphere.stage.BioData``); ``None`` if absent.
    biosphere: object | None = None
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
        if self.env is not None:
            names += [n for n in self.env.layers if n not in names]
        if self.terrain is not None:
            names += [n for n in self.terrain.layers if n not in names]
        if getattr(self, "topo", None) is not None:
            names += [n for n in self.topo.layers if n not in names]
        if getattr(self, "biomes", None) is not None:
            names += [n for n in self.biomes.layers if n not in names]
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
        bio = getattr(self, "biomes", None)
        if bio is not None and name in bio.layers:
            return bio.layers[name]  # v5.0 fields have their own names
        topo = getattr(self, "topo", None)
        if topo is not None and name in topo.layers:
            return topo.layers[name]  # v4.7 fields have their own names
        if self.env is not None and name in self.env.layers:
            return self.env.layers[name]  # v4.5 fields have their own names
        if self.hydro is not None and name in self.hydro.layers:
            return self.hydro.layers[name]  # v4 (after erosion) wins over everything older
        if self.terrain is not None and name in self.terrain.layers:
            return self.terrain.layers[name]  # v4.6 composition (before v4) over v3
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
            if "ocean_mask" in self.hydro.layers:
                # depth of the world ocean only; a dry depression has no depth
                info["depth_m"] = info["water_depth_m"] if info["ocean_mask"] else 0.0
                info["height_above_sea_m"] = (
                    info["elevation_m"] - sea if not info["ocean_mask"] else 0.0
                )
            else:
                info["depth_m"] = max(0.0, sea - info["elevation_m"])
            info["v4_source"] = (
                f"v4 after erosion of the t = {self.hydro.epoch_time_ma:g} Ma surface"
            )
        if self.env is not None:
            for name, arr in self.env.layers.items():
                v = arr[row, col]
                if arr.dtype == np.bool_:
                    info[name] = bool(v)
                elif np.issubdtype(arr.dtype, np.integer):
                    info[name] = int(v)
                else:
                    info[name] = None if np.isnan(v) else float(v)
            info["monthly"] = {
                name: [None if np.isnan(x) else float(x) for x in arr[:, row, col]]
                for name, arr in self.env.monthly.items()
            }
        terrain = getattr(self, "terrain", None)
        if terrain is not None:
            for name, arr in terrain.layers.items():
                key = name if name not in info else f"v46_{name}"
                v = arr[row, col]
                if arr.dtype == np.bool_:
                    info[key] = bool(v)
                elif np.issubdtype(arr.dtype, np.integer):
                    info[key] = int(v)
                else:
                    info[key] = None if np.isnan(v) else float(v)
            b = info.get("basin_id", -1)
            plans = terrain.plans.get("basins", [])
            if b is not None and 0 <= b < len(plans):
                info["basin"] = {k: v for k, v in plans[b].items() if k != "polarity"}
        if terrain is not None and "marine_province" in terrain.layers:
            from .geomorph.classify import PROVINCES, TITLES

            code = int(terrain.layers["marine_province"][row, col])
            if PROVINCES[code] != "land":
                info["seafloor_province"] = PROVINCES[code]
                info["seafloor_province_title"] = TITLES[PROVINCES[code]]
        elif self.hist is not None and info.get("depth_m", 0.0) > 0.0:
            from .stages.ocean import PROVINCES, TITLES

            code = int(self.seafloor_provinces()[row, col])
            info["seafloor_province"] = PROVINCES[code]
            info["seafloor_province_title"] = TITLES[PROVINCES[code]]
        topo = getattr(self, "topo", None)
        if topo is not None:
            from .topoclimate.thermal import BELTS

            for name, arr in topo.layers.items():
                v = arr[row, col]
                if arr.dtype == np.bool_:
                    info[name] = bool(v)
                elif np.issubdtype(arr.dtype, np.integer):
                    info[name] = int(v)
                else:
                    info[name] = None if np.isnan(v) else round(float(v), 4)
            info["dominant_thermal_belt_name"] = BELTS[int(info["dominant_thermal_belt"])]
            info["topo_monthly"] = {
                name: [None if np.isnan(x) else round(float(x), 3) for x in arr[:, row, col]]
                for name, arr in topo.monthly.items()
            }
            info["terrain_units"] = self.terrain_units(col, row)
        bio = getattr(self, "biomes", None)
        if bio is not None:
            info["biomes"] = self.explain_biome(col, row)
        return info

    def explain_biome(self, col: int, row: int) -> dict:
        """Whole-hex composition (sums to 1), dominant classes and why (TZ 5.0, 13.2)."""
        from .biomes.registry import (
            BIOMES,
            COVER_CLASSES,
            LIMITING_FACTORS,
            MODIFIERS,
            REALMS,
            decode_bits,
        )
        from .pipeline_v50 import KOPPEN

        bio = self.biomes
        h = row * self.layout.width + col
        L = bio.layers
        comp = bio.breakdown(h)
        dry = 1.0 - float(L["cover_open_water"][row, col])
        return {
            "basis": "whole-hex planimetric area",
            "composition": [{"biome": b, "title": BIOMES[INDEX_OF[b]].ru, "fraction": round(f, 4),
                             "land_only_fraction": (round(f / dry, 4) if dry > 0 and
                                                    BIOMES[INDEX_OF[b]].family not in
                                                    ("marine", "freshwater") else None)}
                            for b, f in comp],
            "dominant": BIOMES[int(L["dominant_biome"][row, col])].id,
            "dominant_land": BIOMES[int(L["dominant_land_biome"][row, col])].id,
            "realm": REALMS[int(L["realm"][row, col])],
            "climate_zone_koppen": KOPPEN[int(L["climate_zone_koppen"][row, col])],
            "cover": {c: round(float(L[f"cover_{c}"][row, col]), 4) for c in COVER_CLASSES
                      if np.isfinite(L[f"cover_{c}"][row, col])},
            "vegetation_fraction": _num(L["vegetation_fraction"][row, col]),
            "tree_canopy_fraction": _num(L["tree_canopy_fraction"][row, col]),
            "productivity_index": _num(L["productivity_index"][row, col]),
            "aridity_index": _num(L["aridity_index"][row, col]),
            "aridity_valid": bool(L["aridity_valid"][row, col]),
            "limiting_factors": decode_bits(int(L["limiting_flags"][row, col]), LIMITING_FACTORS),
            "habitat_modifiers": decode_bits(int(L["habitat_modifiers"][row, col]), MODIFIERS),
            "ice_thickness_m": _num(L["ice_thickness_m"][row, col]),
            "ice_form": ("none", "glacier_mountain", "glacier_ice_cap", "glacier_ice_sheet")[
                int(L["ice_form"][row, col])],
            "glacier_state": ("none", "simulated_flow", "initialized_existing_ice",
                              "potential_only")[int(L["glacier_state"][row, col])],
            "firn_swe_mm": _num(L["firn_swe_mm"][row, col]),
            "quality": bio.summary.get("input_quality"),
            "model": bio.provenance.get("classification_version"),
        }  # fmt: skip

    def terrain_units(self, col: int, row: int) -> list[dict]:
        """The v4.7 terrain units of one hex: weights, landform, height, climate summary.

        Units are statistical strata (no placement inside the hex is invented).
        """
        topo = getattr(self, "topo", None)
        if topo is None:
            return []
        from .topoclimate.thermal import BELTS
        from .topoclimate.units import LANDFORMS, SURFACE

        U = topo.units
        out = []
        for i in topo.units_of_hex(row * self.layout.width + col).tolist():

            def num(name, i=i):
                v = float(U[name][i])
                return None if np.isnan(v) else round(v, 3)

            out.append(
                {
                    "unit": i,
                    "surface": SURFACE[int(U["unit_surface_type"][i])],
                    "landform": LANDFORMS[int(U["unit_landform"][i])],
                    "weight_hex": num("unit_weight_hex"),
                    "weight_land": num("unit_weight_land"),
                    "z_m": num("z_surface_m"),
                    "slope_deg": num("slope_deg"),
                    "aspect_deg": num("aspect_deg"),
                    "svf": num("svf_slope"),
                    "t_jan_c": round(float(U["temperature_c"][0, i]), 2),
                    "t_jul_c": round(float(U["temperature_c"][6, i]), 2),
                    "precip_mm_yr": round(float(U["precip_mm"][:, i].sum()), 1),
                    "swe_max_mm": round(float(U["swe_mm"][:, i].max()), 1),
                    "gsl_days": num("gsl_t_days"),
                    "gst_c": num("gst_t_c"),
                    "thermal_margin": num("thermal_margin"),
                    "belt": BELTS[int(U["thermal_belt"][i])],
                    "cap_static": num("cap_static"),
                }
            )
        return out

    def seafloor_provinces(self) -> np.ndarray:
        """Sea-floor province codes of the current surface (cached per surface and sea level)."""
        from .stages.ocean import project_provinces

        z = self.layer("elevation_m")
        key = (id(z), self.current_sea_level_m)
        cache = self.__dict__.get("_seafloor_cache")
        if cache is None or cache[0] != key:
            cache = (key, project_provinces(self))
            self.__dict__["_seafloor_cache"] = cache
        return cache[1]

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
