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
