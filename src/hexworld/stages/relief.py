"""v2 stage ``macro_elevation``: one continuous field of absolute macro-elevation.

``macro_elevation_m`` is metres in a single vertical datum (0 m is the model
datum; sea level is a separate scenario parameter). Land, sea, depth and shelf
are all *derived* from this one field - there is no independent "ocean map".

Ingredients, all functions of world position:

* hypsometry of the crust: abyssal floor -> continental slope -> shelf ->
  lowland -> interior, driven by the signed distance to the crust edge. The
  shoreline is therefore *not* the crust edge: a strip of continental crust
  stays under water (shelf), wider on quiet margins;
* broad swells and basins (cratons, interior basins, abyssal relief);
* macro-tectonic terms from the world skeleton: collision uplift between two
  continental margins, uplift of the overriding side of a subduction zone with
  a trench on the subducting side and a basin behind, mid-ocean ridge swell,
  continental rift trough with raised shoulders.

It is **macro** relief: no peaks, ridges, valleys or erosion. Later stages
(PMGE mountains, erosion) add detail on top of it and must not replace it.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..config import Config, ReliefParams
from ..noise import FractalNoise
from ..rng import Stream
from .boundaries import TYPE_CONVERGENT, TYPE_DIVERGENT, TYPE_TRANSFORM
from .continents import ContinentModel, smoothstep
from .skeleton import WorldSkeleton, accumulate

_SIDE_KM = 110.0


def _gauss(d, width):
    return np.exp(-0.5 * (d / width) ** 2)


@dataclass
class ReliefModel:
    continents: ContinentModel
    skeleton: WorldSkeleton
    params: ReliefParams
    broad: FractalNoise
    craton: FractalNoise
    shelf: FractalNoise
    strength: np.ndarray  # saturating 0..1 measure of boundary activity per sample
    collision: np.ndarray  # bool: convergent with continental crust on both sides
    sub_a: np.ndarray  # bool: plate_a side subducts
    sub_b: np.ndarray

    def _tectonic(self, x, y, fraction):
        sk = self.skeleton
        conv = sk.kind == TYPE_CONVERGENT
        div = sk.kind == TYPE_DIVERGENT
        trans = sk.kind == TYPE_TRANSFORM
        s = self.strength
        col = np.where(self.collision, s, 0.0)
        sub = np.where(conv & (self.sub_a | self.sub_b), s, 0.0)
        ridge = np.where(div, np.sqrt(s), 0.0)
        rift = np.where(div, s, 0.0)
        shear = np.where(trans, s, 0.0)
        trench_w = max(90.0, 0.9 * sk.cell_km)

        def contribution(idx, d, side_b):
            on_sub = (self.sub_a[idx] & ~side_b) | (self.sub_b[idx] & side_b)
            over = ~on_sub
            return [
                # 0: applies on continental crust
                col[idx] * 3000.0 * _gauss(d, 300.0)
                + sub[idx] * over * 2000.0 * _gauss(d - 180.0, 150.0)
                + rift[idx] * (-1100.0 * _gauss(d, 85.0) + 450.0 * _gauss(d - 175.0, 90.0)),
                # 1: applies on oceanic crust
                sub[idx] * over * 900.0 * _gauss(d - 180.0, 150.0)
                + ridge[idx] * 1500.0 * _gauss(d, 380.0),
                # 2: independent of crust type
                sub[idx]
                * (
                    on_sub * (-2800.0 * _gauss(d, trench_w) + 250.0 * _gauss(d - 260.0, 120.0))
                    + over * (-450.0 * _gauss(d - 620.0, 220.0))
                )
                - shear[idx] * 250.0 * _gauss(d, 60.0),
            ]

        on_cont, on_ocean, both = accumulate(sk, x, y, contribution, 1300.0, 3)
        return fraction * on_cont + (1.0 - fraction) * on_ocean + both

    def elevation(self, x, y, continentality=None, parts: bool = False):
        """Macro-elevation in metres at world point(s); optionally also its tectonic part."""
        x = np.asarray(x, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        p = self.params
        c = self.continents.continentality(x, y) if continentality is None else continentality
        c = np.asarray(c, dtype=np.float64)
        fraction = self.continents.fraction_of(c)

        shelf_scale = p.shelf_scale_km * np.exp(0.5 * self.shelf(x, y))
        land = p.shelf_edge_m + (p.continental_interior_m - p.shelf_edge_m) * (
            1.0 - np.exp(-np.maximum(c, 0.0) / shelf_scale)
        )
        interior = smoothstep((c - 250.0) / 900.0)
        land = land + 260.0 * interior * self.craton(x, y)
        ocean = p.shelf_edge_m + (p.ocean_floor_m - p.shelf_edge_m) * smoothstep(-c / 230.0)
        base = np.where(c >= 0.0, land, ocean)

        coast_weight = 0.3 + 0.7 * smoothstep(np.abs(c) / 350.0)
        swell = self.broad(x, y) * np.where(c >= 0.0, 300.0, 340.0) * coast_weight
        tectonic = p.relief_strength * self._tectonic(x, y, fraction)
        total = base + swell + tectonic
        return (total, tectonic) if parts else total


def build_relief(config: Config, continents: ContinentModel, skeleton: WorldSkeleton):
    stream = Stream(config.seed, "macro_relief/v2")
    broad = FractalNoise(
        stream.child("broad"), [3600.0, 1800.0, 900.0, 450.0], [1.0, 0.5, 0.25, 0.125]
    )
    craton = FractalNoise(stream.child("craton"), [5000.0, 2500.0], [1.0, 0.5])
    shelf = FractalNoise(stream.child("shelf"), [2600.0, 1300.0], [1.0, 0.4])

    sx, sy, nx, ny = skeleton.bx, skeleton.by, skeleton.nx, skeleton.ny
    if sx.size:
        cont_a = continents.continentality(sx - _SIDE_KM * nx, sy - _SIDE_KM * ny) > 0.0
        cont_b = continents.continentality(sx + _SIDE_KM * nx, sy + _SIDE_KM * ny) > 0.0
    else:
        cont_a = cont_b = np.zeros(0, dtype=bool)
    conv = skeleton.kind == TYPE_CONVERGENT
    collision = conv & cont_a & cont_b
    # oceanic crust goes under continental; ocean-ocean: the faster advancing plate
    sub_a = conv & ~cont_a & (cont_b | (skeleton.adv_a >= skeleton.adv_b))
    sub_b = conv & ~cont_b & ~sub_a
    strength = 1.0 - np.exp(-2.5 * skeleton.intensity)
    return ReliefModel(
        continents=continents,
        skeleton=skeleton,
        params=config.relief,
        broad=broad,
        craton=craton,
        shelf=shelf,
        strength=strength,
        collision=collision,
        sub_a=sub_a,
        sub_b=sub_b,
    )
