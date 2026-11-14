"""Stage ``motion``: rigid plate kinematics on the plane.

Each plate has a translation ``(vx, vy)`` in mm/yr and a small rotation rate
``omega`` (deg/Myr) about its pole (the plate site), so the velocity at a point
is ``v = (vx, vy) + omega x (p - pole)``. Handy identity: 1 km/Myr = 1 mm/yr.

Causal link kept deliberately simple: plates dominated by oceanic crust tend to
be faster (slab pull), continental ones slower. Velocities are expressed in a
frame with zero area-weighted mean translation over the simulation area.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from ..config import Config
from ..rng import Stream


@dataclass
class PlateMotion:
    velocity: np.ndarray  # (M, 2) mm/yr, +x east, +y south
    omega: np.ndarray  # (M,) deg/Myr, positive turns +x towards +y
    pole: np.ndarray  # (M, 2) km

    def velocity_at(self, plate_ids, x, y):
        """Velocity (vx, vy) in mm/yr of plate(s) ``plate_ids`` at world point(s)."""
        pid = np.asarray(plate_ids)
        x = np.asarray(x, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        w = np.deg2rad(self.omega[pid])
        rx = x - self.pole[pid, 0]
        ry = y - self.pole[pid, 1]
        return self.velocity[pid, 0] - w * ry, self.velocity[pid, 1] + w * rx

    @classmethod
    def uniform(cls, velocities, poles=None) -> PlateMotion:
        """Pure translations - for tests and experiments."""
        v = np.asarray(velocities, dtype=np.float64).reshape(-1, 2)
        p = np.zeros_like(v) if poles is None else np.asarray(poles, dtype=np.float64)
        return cls(v, np.zeros(v.shape[0]), p)


def build_motion(config: Config, sites, oceanic_fraction, area_km2) -> PlateMotion:
    stream = Stream(config.seed, "motion")
    m = len(sites)
    u = stream.random(3 * m).reshape(m, 3)
    z = stream.normal(m)
    ocean = np.asarray(oceanic_fraction, dtype=np.float64)
    speed = (12.0 + 60.0 * u[:, 0] ** 1.4) * (0.5 + 0.8 * ocean) * config.motion.speed_scale
    direction = 2.0 * math.pi * u[:, 1]
    velocity = np.stack([speed * np.cos(direction), speed * np.sin(direction)], axis=1)
    omega = np.clip(0.12 * z, -0.35, 0.35) * config.motion.rotation_scale
    area = np.asarray(area_km2, dtype=np.float64)
    if area.sum() > 0:
        velocity = velocity - (velocity * area[:, None]).sum(axis=0) / area.sum()
    return PlateMotion(velocity=velocity, omega=omega, pole=np.asarray(sites, dtype=np.float64))
