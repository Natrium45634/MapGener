"""Sea-floor provinces: what lies under the sea, in the terms of a physical atlas.

The depths themselves come from the history (age-depth subsidence of new ocean floor,
trenches and outer rises at subduction zones, rifts, volcanoes, sediment). This module only
*reads* them and names each sea hex, so a map of the ocean can be read like a map of the
land:

==================  ==========================================================  =========
province            rule (first match wins)                                     example
==================  ==========================================================  =========
trench              deeper than 6 km, or a narrow trough >700 m below its       Mariana
                    surroundings next to a subduction boundary
seamount            a volcano on the sea floor, or a knob >800 m above its      Hawaii,
                    surroundings and >1200 m shallower than its crust's age     Emperor
                    predicts
shelf               shallower than 200 m                                        North Sea
ridge axis          oceanic crust younger than 8 Ma                             Mid-Atlantic
continental slope   200-3000 m, at a continent (continental crust or within     -
                    ~250 km of the shelf)
continental rise    3000-4500 m within ~600 km of the shelf, on sediment        -
oceanic plateau     oceanic crust >1000 m shallower than its age predicts       Ontong Java
ridge flank         oceanic crust 8-30 Ma                                       -
marginal basin      thinned continental crust below 200 m                       Black Sea
abyssal plain       deeper than 4 km, flat or sediment-covered                  Sohm
abyssal hills       the rest of the deep ocean floor                            -
==================  ==========================================================  =========

Depth tints follow the atlas convention of steps (0-200 m shelf blue, ..., >6000 m hadal,
almost black). Units: metres, km, Ma.
"""

from __future__ import annotations

import numpy as np
from scipy import ndimage

PROVINCES = (
    "land",
    "shelf",
    "continental_slope",
    "continental_rise",
    "trench",
    "ridge_axis",
    "ridge_flank",
    "seamount",
    "oceanic_plateau",
    "marginal_basin",
    "abyssal_plain",
    "abyssal_hills",
)
CODE = {name: i for i, name in enumerate(PROVINCES)}
TITLES = {
    "land": "суша",
    "shelf": "шельф и мелководья (до 200 м)",
    "continental_slope": "материковый склон",
    "continental_rise": "материковое подножие",
    "trench": "глубоководный желоб",
    "ridge_axis": "срединно-океанический хребет (ось, кора моложе 8 млн лет)",
    "ridge_flank": "склоны срединного хребта (8–30 млн лет)",
    "seamount": "подводные горы и вулканические острова",
    "oceanic_plateau": "океаническое плато",
    "marginal_basin": "окраинная котловина на утонённой коре",
    "abyssal_plain": "абиссальная равнина",
    "abyssal_hills": "абиссальные холмы",
}
RGB = np.array(
    [
        [150, 150, 140],  # land
        [190, 228, 242],  # shelf
        [122, 182, 218],  # slope
        [88, 146, 196],  # rise
        [8, 8, 14],  # trench
        [226, 104, 64],  # ridge axis
        [232, 168, 118],  # ridge flank
        [172, 92, 176],  # seamount
        [118, 168, 128],  # plateau
        [82, 186, 184],  # marginal basin
        [30, 62, 118],  # abyssal plain
        [52, 92, 152],  # abyssal hills
    ],
    dtype=np.uint8,
)

#: Atlas depth steps, m below sea level, and their tints (shallow -> hadal).
DEPTH_STEPS = (0.0, 200.0, 1000.0, 2000.0, 3000.0, 4000.0, 5000.0, 6000.0, 7000.0)
DEPTH_RGB = np.array(
    [
        [196, 230, 244],  # 0-200 shelf
        [150, 206, 234],  # 200-1000
        [114, 180, 222],  # 1000-2000
        [82, 150, 204],  # 2000-3000
        [56, 118, 182],  # 3000-4000
        [36, 88, 156],  # 4000-5000
        [20, 58, 120],  # 5000-6000
        [12, 22, 54],  # 6000-7000 hadal: trenches
        [0, 0, 6],  # deeper than 7000
    ],
    dtype=np.float64,
)


def depth_tint(depth_m) -> np.ndarray:
    """Stepped atlas tint of water depth (m, positive down) with a slight in-step gradient."""
    d = np.maximum(np.asarray(depth_m, dtype=np.float64), 0.0)
    steps = np.asarray(DEPTH_STEPS)
    k = np.clip(np.searchsorted(steps, d, side="right") - 1, 0, len(steps) - 1)
    top = steps[k]
    width = np.where(k < len(steps) - 1, steps[np.minimum(k + 1, len(steps) - 1)] - top, 1000.0)
    frac = np.clip((d - top) / width, 0.0, 1.0)
    base = DEPTH_RGB[k]
    nxt = DEPTH_RGB[np.minimum(k + 1, len(steps) - 1)]
    rgb = base + 0.18 * frac[..., None] * (nxt - base)  # mostly flat steps, readable bands
    return rgb.astype(np.uint8)


def expected_depth_m(age_ma):
    """Depth of normal ocean floor of this age (half-space cooling, Parsons & Sclater 1977)."""
    return 2600.0 + 350.0 * np.sqrt(np.clip(np.asarray(age_ma, dtype=np.float64), 0.0, 120.0))


def provinces(
    elevation_m,
    sea_level_m: float,
    continental_fraction,
    crust_age_ma,
    boundary_class,
    volcano_id,
    sediment_m,
    spacing_km: float,
) -> np.ndarray:
    """Sea-floor province code per hex (``PROVINCES``); land is 0. Arrays (rows, cols)."""
    z = np.asarray(elevation_m, dtype=np.float64)
    depth = sea_level_m - z
    sea = depth > 0.0
    cont = np.asarray(continental_fraction, dtype=np.float64)
    age = np.nan_to_num(np.asarray(crust_age_ma, dtype=np.float64), nan=200.0)
    oceanic = cont < 0.15
    sed = np.nan_to_num(np.asarray(sediment_m, dtype=np.float64))
    volc = np.asarray(volcano_id) >= 0
    hex_km = max(float(spacing_km), 1e-6)

    # the regional surface over ~400 km: troughs and knobs stand out against it
    sigma = max(1.0, 400.0 / hex_km / 2.0)
    regional = ndimage.gaussian_filter(np.where(sea, z, sea_level_m - 3000.0), sigma)
    anomaly = z - regional
    shelfish = sea & (depth <= 200.0) | ~sea
    dist_shelf_km = ndimage.distance_transform_edt(~shelfish) * hex_km
    subduction = ndimage.binary_dilation(np.asarray(boundary_class) == 2, iterations=3)
    rough = ndimage.maximum_filter(z, size=3, mode="nearest") - ndimage.minimum_filter(
        z, size=3, mode="nearest"
    )

    out = np.full(z.shape, CODE["abyssal_hills"], dtype=np.uint8)
    rules = [
        ("marginal_basin", ~oceanic & (cont < 0.5) & (depth > 200.0)),
        ("ridge_flank", oceanic & (age < 30.0)),
        ("abyssal_plain", (depth >= 4000.0) & ((rough < 400.0) | (sed > 150.0))),
        (
            "oceanic_plateau",
            oceanic & (age >= 30.0) & (depth < expected_depth_m(age) - 1000.0) & (depth < 3200.0),
        ),
        (
            "continental_rise",
            (depth >= 3000.0) & (depth < 4500.0) & (dist_shelf_km <= 600.0) & (sed > 80.0),
        ),
        (
            "continental_slope",
            (depth > 200.0)
            & (depth < 3000.0)
            & ((cont >= 0.15) | (dist_shelf_km <= 250.0))
            & ~(oceanic & (age < 8.0)),
        ),
        ("ridge_axis", oceanic & (age < 8.0) & (depth > 200.0)),
        ("shelf", depth <= 200.0),
        # a knob shallower than the age of its crust allows (a ridge axis is high because
        # it is young, not a seamount)
        (
            "seamount",
            volc
            | (
                oceanic
                & (anomaly > 800.0)
                & (depth < expected_depth_m(age) - 1200.0)
                & (depth > 200.0)
            ),
        ),
        ("trench", (depth >= 6000.0) | (subduction & (anomaly < -700.0) & (depth > 3000.0))),
    ]
    for name, mask in rules:  # later rules win
        out[mask] = CODE[name]
    out[~sea] = CODE["land"]
    return out


def project_provinces(project) -> np.ndarray:
    """Provinces of the project's current surface (after v4 erosion when it exists)."""
    sea_level = project.current_sea_level_m
    sea_level = 0.0 if sea_level is None else sea_level
    hist = project.hist.layers if project.hist is not None else {}

    def get(name, default):
        try:
            return project.layer(name)
        except (KeyError, LookupError):
            return hist.get(name, default)

    z = project.layer("elevation_m")
    zeros = np.zeros(z.shape)
    return provinces(
        z,
        float(sea_level),
        get("continental_fraction", zeros),
        get("crust_age_ma", np.full(z.shape, 100.0)),
        get("boundary_class", np.zeros(z.shape, dtype=np.uint8)),
        get("volcano_id", np.full(z.shape, -1)),
        get("sediment_thickness_m", zeros),
        project.layout.spacing_km,
    )


__all__ = [
    "CODE",
    "DEPTH_RGB",
    "DEPTH_STEPS",
    "PROVINCES",
    "RGB",
    "TITLES",
    "depth_tint",
    "expected_depth_m",
    "project_provinces",
    "provinces",
]
