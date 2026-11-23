"""Planet: solar geometry, insolation and rotation (circular orbit).

Declination at the fraction ``t`` of the year (``t = 0`` at the northern spring equinox)::

    sin(delta) = sin(eps) sin(2 pi t)

Sunset hour angle ``H0 = arccos(clip(-tan(phi) tan(delta), -1, 1))``: polar night
``H0 = 0``, polar day ``H0 = pi``. The insolation averaged over one rotation (W/m^2) does not
depend on the length of the day::

    Q = (S0 / pi) * dr * (H0 sin(phi) sin(delta) + cos(phi) cos(delta) sin(H0))

``Ra`` in MJ m^-2 per 86 400 s (the unit FAO-56 uses) is ``Q * 0.0864`` for any rotation
period; the daylight duration is ``day_length_h * H0 / pi`` hours of the planet's day.
Rotation: ``Omega = 2 pi / day_s``, ``f = 2 Omega sin(phi)``, ``beta = 2 Omega cos(phi) / R``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

#: Month mid-points as fractions of the year after the northern spring equinox
#: (equinox ~ 20 March): Jan 15 is ~ -0.18.
MONTH_FRACTION = (np.arange(12) + 0.5) / 12.0 - (79.5 / 365.25)
#: Seasons DJF, MAM, JJA, SON as month indices.
SEASON_MONTHS = ((11, 0, 1), (2, 3, 4), (5, 6, 7), (8, 9, 10))


@dataclass(frozen=True)
class Planet:
    obliquity_deg: float = 23.44
    day_length_h: float = 24.0
    year_length_days: float = 365.25
    solar_constant_wm2: float = 1361.0
    radius_km: float = 6371.0

    @classmethod
    def from_params(cls, ep) -> Planet:
        return cls(ep.obliquity_deg, ep.day_length_h, ep.year_length_days,
                   ep.solar_constant_wm2, ep.planet_radius_km)  # fmt: skip

    @property
    def omega(self) -> float:
        return 2.0 * math.pi / (self.day_length_h * 3600.0)

    def coriolis(self, lat_deg):
        return 2.0 * self.omega * np.sin(np.radians(lat_deg))

    def beta(self, lat_deg, radius_m: float | None = None):
        r = radius_m or self.radius_km * 1000.0
        return 2.0 * self.omega * np.cos(np.radians(lat_deg)) / r

    def declination(self, year_fraction):
        eps = math.radians(self.obliquity_deg)
        return np.arcsin(math.sin(eps) * np.sin(2.0 * math.pi * np.asarray(year_fraction)))

    def sunset_angle(self, lat_deg, delta):
        phi = np.radians(np.asarray(lat_deg, dtype=np.float64))
        x = -np.tan(phi) * np.tan(delta)
        return np.arccos(np.clip(x, -1.0, 1.0))

    def insolation(self, lat_deg, year_fraction):
        """Top-of-atmosphere insolation averaged over one rotation, W/m^2."""
        phi = np.radians(np.asarray(lat_deg, dtype=np.float64))
        delta = self.declination(year_fraction)
        h0 = self.sunset_angle(lat_deg, delta)
        q = (self.solar_constant_wm2 / math.pi) * (
            h0 * np.sin(phi) * np.sin(delta) + np.cos(phi) * np.cos(delta) * np.sin(h0)
        )
        return np.maximum(q, 0.0)

    def ra_mj(self, lat_deg, year_fraction):
        """FAO-56 R_a in MJ m^-2 per 86 400 s, for any rotation period."""
        return self.insolation(lat_deg, year_fraction) * 0.0864

    def daylight_hours(self, lat_deg, year_fraction):
        """Hours of daylight within one rotation of the planet."""
        h0 = self.sunset_angle(lat_deg, self.declination(year_fraction))
        return self.day_length_h * h0 / math.pi

    def monthly_insolation(self, lat_deg):
        """(12, ...) insolation at the month mid-points, W/m^2."""
        lat = np.asarray(lat_deg, dtype=np.float64)
        return np.stack([self.insolation(lat, MONTH_FRACTION[m]) for m in range(12)])

    def annual_insolation(self, lat_deg, samples: int = 96):
        lat = np.asarray(lat_deg, dtype=np.float64)
        t = (np.arange(samples) + 0.5) / samples
        return np.mean([self.insolation(lat, x) for x in t], axis=0)


EARTH = Planet()


def seasonal_amplitude_ratio(planet: Planet, lat_deg):
    """Half range of the monthly insolation relative to the Earth's, per latitude.

    Scales the seasonal temperature amplitude of the climate model when the obliquity is
    not the Earth's (1 for the Earth; 0 at 0 deg obliquity; ~2.5x at 60 deg, high latitude).
    """
    lat = np.asarray(lat_deg, dtype=np.float64)
    a = planet.monthly_insolation(lat)
    e = EARTH.monthly_insolation(lat)
    ra = 0.5 * (a.max(axis=0) - a.min(axis=0))
    re = 0.5 * (e.max(axis=0) - e.min(axis=0))
    return np.where(re > 1.0, ra / np.maximum(re, 1.0), 1.0 + ra / 50.0)


def annual_temperature_shift(planet: Planet, lat_deg, feedback_w_m2_k: float = 3.5,
                             cap_k: float = 25.0):  # fmt: skip
    """``[H]`` shift of the zonal annual temperature from the annual insolation change.

    ``dT = 0.7 (S_ann(planet) - S_ann(Earth)) / B``, B = 3.5 W m^-2 K^-1 (an Earth-like
    outgoing-longwave feedback including meridional heat transport), capped at +-25 K.
    A calibrated approximation for non-Earth planets; exactly 0 for the Earth.
    """
    lat = np.asarray(lat_deg, dtype=np.float64)
    if planet == EARTH:
        return np.zeros_like(lat)
    d = 0.7 * (planet.annual_insolation(lat) - EARTH.annual_insolation(lat)) / feedback_w_m2_k
    return np.clip(d, -cap_k, cap_k)
