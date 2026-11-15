"""Typed, validated generation configuration.

Every parameter belongs to exactly one stage section, so a change in a late
stage (e.g. ``fields``) never alters the hash key of an earlier one.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field, fields, replace
from typing import Any

from .rng import validate_seed

PROFILES = ("few", "earthlike", "fragmented")
CONTINENT_PROFILES = ("balanced", "supercontinent", "fragmented")
SEA_MODES = ("target_land_fraction", "fixed_sea_level_m")
HISTORY_PROFILES = ("calm", "earthlike", "active")
MOUNTAIN_BACKENDS = ("reference", "pmge")
MODES = ("regional", "world")


class ConfigError(ValueError):
    pass


@dataclass(frozen=True)
class PlateParams:
    #: "auto" or a positive integer: plates whose sites lie in the simulation area.
    count: int | str = "auto"
    profile: str = "earthlike"
    #: Boundary irregularity: sum over octaves of (warp amplitude / wavelength).
    warp_strength: float = 0.7
    #: Longest warp wavelength as a fraction of the mean primary plate spacing.
    warp_wavelength_factor: float = 0.9
    warp_octaves: int = 6


@dataclass(frozen=True)
class CrustParams:
    continental_fraction: float = 0.36
    mean_continent_area_km2: float = 1.4e7
    warp_strength: float = 0.8
    warp_wavelength_km: float = 4000.0
    warp_octaves: int = 6
    #: Additive long-wave irregularity relative to the mean continent radius.
    roughness: float = 0.4


@dataclass(frozen=True)
class MotionParams:
    speed_scale: float = 1.0
    rotation_scale: float = 1.0


@dataclass(frozen=True)
class BoundaryParams:
    #: Relative speed below which a boundary is "quiet / undetermined", mm/yr.
    quiet_threshold_mm_yr: float = 3.0
    #: |normal| / |relative| above sin(angle) => convergent/divergent, else transform.
    obliquity_deg: float = 30.0
    #: Relative speed that maps to intensity 1.0, mm/yr.
    reference_speed_mm_yr: float = 100.0


@dataclass(frozen=True)
class FieldParams:
    collision_width_km: float = 350.0
    subduction_width_km: float = 280.0
    trench_width_km: float = 90.0
    arc_offset_km: float = 170.0
    arc_width_km: float = 90.0
    rift_width_km: float = 160.0
    ridge_volcanism_width_km: float = 70.0
    transform_width_km: float = 110.0
    #: Farthest distance at which a boundary influences the fields, km.
    max_influence_km: float = 900.0
    #: Upper limit for the off-screen halo of hexes used to avoid viewport-edge artefacts.
    max_pad_hexes: int = 64


@dataclass(frozen=True)
class ContinentParams:
    """v2.0 stage ``continental_blocks`` (RNG namespace ``continents/v2``)."""

    continent_profile: str = "balanced"
    #: "auto" or the wished number of *major* masses in the simulation area (a goal).
    major_landmass_count: int | str = "auto"
    #: 0..1: spread of block sizes and share of secondary fragments.
    continent_size_variety: float = 0.5
    #: 0..1.5: large- and mid-scale coast irregularity (never white noise).
    coast_irregularity: float = 0.7
    #: A mass counts as "major" above this area, km^2.
    major_area_km2: float = 2.5e6
    #: Width of the continental -> oceanic crust transition, km.
    transition_km: float = 90.0
    #: Cells along the longer side of the world-level tectonic skeleton raster.
    skeleton_cells: int = 512


@dataclass(frozen=True)
class ReliefParams:
    """v2.0 stage ``macro_elevation`` (RNG namespace ``macro_relief/v2``)."""

    #: Strength of macro-tectonic uplift and troughs; detailed mountains are not affected.
    relief_strength: float = 1.0
    ocean_floor_m: float = -4430.0
    continental_interior_m: float = 220.0
    shelf_edge_m: float = -480.0
    shelf_scale_km: float = 170.0


@dataclass(frozen=True)
class SeaParams:
    """v2.0 stage ``sea_level_landmasses`` (RNG namespace ``sea_level/v2``)."""

    sea_mode: str = "target_land_fraction"
    #: Used when sea_mode == "target_land_fraction": land share of the simulation area.
    target_land_fraction: float = 0.29
    #: Used when sea_mode == "fixed_sea_level_m": level in the model vertical datum, m.
    fixed_sea_level_m: float = 0.0
    #: Submerged continental crust shallower than this is reported as shelf, m.
    shelf_max_depth_m: float = 500.0


@dataclass(frozen=True)
class HistoryParams:
    """v3.0 stage ``geo_history`` (RNG namespace ``geohistory/v3``)."""

    enabled: bool = True
    #: Simulated time from the baseline snapshot (t = 0), million years.
    duration_ma: float = 30.0
    #: Time step, Ma. Internally sub-stepped so no plate moves more than one cell per step.
    time_step_ma: float = 1.0
    #: A checkpoint is stored every this many steps (plus the initial and final states).
    checkpoint_stride: int = 5
    #: Tectonic intensity profile.
    intensity: str = "earthlike"
    #: Multipliers of tectonic uplift and subsidence rates.
    uplift_scale: float = 1.0
    subsidence_scale: float = 1.0
    #: Multiplier of hillslope and runoff erosion; 0 switches erosion off.
    erosion_intensity: float = 1.0
    #: Multiplier of the number of intraplate hotspots; 0 switches them off.
    hotspot_density: float = 1.0
    #: Age (Ma before t = 0) assigned to the mountain belts inherited from the baseline.
    inherited_orogen_age_ma: float = 140.0
    #: Which stored epoch the hex layers show: "final", "initial" or a checkpoint time in Ma.
    epoch: float | str = "final"


@dataclass(frozen=True)
class MountainParams:
    """v3.0 stage ``mountains`` (RNG namespace ``mountains/v3``)."""

    #: "reference" - built-in CPU reference backend; "pmge" - external PMGE library (C ABI).
    backend: str = "reference"
    #: 0 switches the structural mountain detail off; 1 is the default relief contrast.
    detail: float = 1.0
    #: Mean spacing of peaks along a ridge, km.
    peak_spacing_km: float = 85.0
    #: Half-width of a main range and of a spur, km.
    range_half_width_km: float = 60.0
    spur_half_width_km: float = 34.0
    #: Hard physical limit of the final surface, m (applied point-wise, never by rescaling).
    max_elevation_m: float = 8800.0
    #: Orogenic relief above which a cell belongs to a mountain system, m.
    system_threshold_m: float = 450.0


@dataclass(frozen=True)
class Config:
    seed: str = "HexWorld1"
    width: int = 100
    height: int = 100
    #: Distance between centres of adjacent hexes, km (physical scale of one hex).
    hex_spacing_km: float = 40.0
    mode: str = "regional"
    #: World coordinates (km) of the top-left corner of the viewport rectangle.
    viewport_origin_km: tuple[float, float] = (0.0, 0.0)
    #: Explicit ``[xmin, ymin, xmax, ymax]``; ``None`` derives it from mode + viewport.
    simulation_bounds_km: tuple[float, float, float, float] | None = None
    #: Cells along the longer side of the world-level reference raster.
    reference_cells: int = 384
    plates: PlateParams = field(default_factory=PlateParams)
    crust: CrustParams = field(default_factory=CrustParams)
    motion: MotionParams = field(default_factory=MotionParams)
    boundaries: BoundaryParams = field(default_factory=BoundaryParams)
    fields: FieldParams = field(default_factory=FieldParams)
    continents: ContinentParams = field(default_factory=ContinentParams)
    relief: ReliefParams = field(default_factory=ReliefParams)
    sea: SeaParams = field(default_factory=SeaParams)
    history: HistoryParams = field(default_factory=HistoryParams)
    mountains: MountainParams = field(default_factory=MountainParams)

    def __post_init__(self):
        validate_seed(self.seed)
        for name in ("width", "height"):
            v = getattr(self, name)
            if not isinstance(v, int) or isinstance(v, bool) or not 2 <= v <= 8192:
                raise ConfigError(f"{name} must be an integer in [2, 8192], got {v!r}")
        if not 0.01 <= float(self.hex_spacing_km) <= 2000.0:
            raise ConfigError("hex_spacing_km must be within [0.01, 2000] km")
        if self.mode not in MODES:
            raise ConfigError(f"mode must be one of {MODES}")
        if self.plates.profile not in PROFILES:
            raise ConfigError(f"plates.profile must be one of {PROFILES}")
        c = self.plates.count
        if c != "auto" and (not isinstance(c, int) or isinstance(c, bool) or not 1 <= c <= 400):
            raise ConfigError('plates.count must be "auto" or an integer in [1, 400]')
        if not 0.0 <= self.plates.warp_strength <= 1.2:
            raise ConfigError("plates.warp_strength must be within [0, 1.2]")
        if not 0.02 <= self.crust.continental_fraction <= 0.98:
            raise ConfigError("crust.continental_fraction must be within [0.02, 0.98]")
        if not 32 <= self.reference_cells <= 2048:
            raise ConfigError("reference_cells must be within [32, 2048]")
        cp = self.continents
        if cp.continent_profile not in CONTINENT_PROFILES:
            raise ConfigError(f"continents.continent_profile must be one of {CONTINENT_PROFILES}")
        m = cp.major_landmass_count
        if m != "auto" and (not isinstance(m, int) or isinstance(m, bool) or not 1 <= m <= 30):
            raise ConfigError('continents.major_landmass_count must be "auto" or 1..30')
        if not 0.0 <= cp.continent_size_variety <= 1.0:
            raise ConfigError("continents.continent_size_variety must be within [0, 1]")
        if not 0.0 <= cp.coast_irregularity <= 1.5:
            raise ConfigError("continents.coast_irregularity must be within [0, 1.5]")
        if not 128 <= cp.skeleton_cells <= 2048:
            raise ConfigError("continents.skeleton_cells must be within [128, 2048]")
        if not 0.0 <= self.relief.relief_strength <= 3.0:
            raise ConfigError("relief.relief_strength must be within [0, 3]")
        if self.sea.sea_mode not in SEA_MODES:
            raise ConfigError(f"sea.sea_mode must be one of {SEA_MODES}")
        if not 0.01 <= self.sea.target_land_fraction <= 0.99:
            raise ConfigError("sea.target_land_fraction must be within [0.01, 0.99]")
        if not -9000.0 <= self.sea.fixed_sea_level_m <= 9000.0:
            raise ConfigError("sea.fixed_sea_level_m must be within [-9000, 9000] m")
        h = self.history
        if not 0.0 <= h.duration_ma <= 400.0:
            raise ConfigError("history.duration_ma must be within [0, 400] Ma")
        if not 0.05 <= h.time_step_ma <= 20.0:
            raise ConfigError("history.time_step_ma must be within [0.05, 20] Ma")
        if not isinstance(h.checkpoint_stride, int) or h.checkpoint_stride < 1:
            raise ConfigError("history.checkpoint_stride must be a positive integer")
        if h.intensity not in HISTORY_PROFILES:
            raise ConfigError(f"history.intensity must be one of {HISTORY_PROFILES}")
        for name in ("uplift_scale", "subsidence_scale", "erosion_intensity", "hotspot_density"):
            if not 0.0 <= getattr(h, name) <= 5.0:
                raise ConfigError(f"history.{name} must be within [0, 5]")
        if isinstance(h.epoch, str) and h.epoch not in ("final", "initial"):
            raise ConfigError('history.epoch must be "final", "initial" or a time in Ma')
        if self.mountains.backend not in MOUNTAIN_BACKENDS:
            raise ConfigError(f"mountains.backend must be one of {MOUNTAIN_BACKENDS}")
        if not 0.0 <= self.mountains.detail <= 2.0:
            raise ConfigError("mountains.detail must be within [0, 2]")
        if not 2000.0 <= self.mountains.max_elevation_m <= 12000.0:
            raise ConfigError("mountains.max_elevation_m must be within [2000, 12000] m")
        if len(self.viewport_origin_km) != 2:
            raise ConfigError("viewport_origin_km must be [x, y]")
        b = self.simulation_bounds_km
        if b is not None and (len(b) != 4 or not (b[2] > b[0] and b[3] > b[1])):
            raise ConfigError("simulation_bounds_km must be [xmin, ymin, xmax, ymax]")

    # -- (de)serialisation -------------------------------------------------
    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["viewport_origin_km"] = list(self.viewport_origin_km)
        if self.simulation_bounds_km is not None:
            d["simulation_bounds_km"] = list(self.simulation_bounds_km)
        return d

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Config:
        data = dict(data)
        sections = {
            "plates": PlateParams,
            "crust": CrustParams,
            "motion": MotionParams,
            "boundaries": BoundaryParams,
            "fields": FieldParams,
            "continents": ContinentParams,
            "relief": ReliefParams,
            "sea": SeaParams,
            "history": HistoryParams,
            "mountains": MountainParams,
        }
        known = {f.name for f in fields(cls)}
        unknown = set(data) - known
        if unknown:
            raise ConfigError(f"Unknown config keys: {sorted(unknown)}")
        kwargs: dict[str, Any] = {}
        for key, value in data.items():
            if key in sections:
                sub = sections[key]
                sub_known = {f.name for f in fields(sub)}
                bad = set(value) - sub_known
                if bad:
                    raise ConfigError(f"Unknown keys in '{key}': {sorted(bad)}")
                kwargs[key] = sub(**value)
            elif key == "viewport_origin_km":
                kwargs[key] = (float(value[0]), float(value[1]))
            elif key == "simulation_bounds_km":
                kwargs[key] = None if value is None else tuple(float(v) for v in value)
            else:
                kwargs[key] = value
        return cls(**kwargs)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2, ensure_ascii=False)

    @classmethod
    def from_json(cls, text: str) -> Config:
        return cls.from_dict(json.loads(text))

    def with_(self, **changes) -> Config:
        return replace(self, **changes)


PRESETS: dict[str, dict[str, Any]] = {
    "small": {"width": 100, "height": 100, "hex_spacing_km": 40.0, "mode": "regional"},
    "medium": {"width": 500, "height": 300, "hex_spacing_km": 30.0, "mode": "regional"},
    "large": {"width": 1000, "height": 800, "hex_spacing_km": 25.0, "mode": "regional"},
    "world": {"width": 1000, "height": 800, "hex_spacing_km": 40.0, "mode": "world"},
}
