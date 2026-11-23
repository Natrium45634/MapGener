"""Planetary seasonal energy-balance model (1-D in latitude, ``[ADAPT]`` North & Coakley
1979 / North 1975) - the atmosphere's equilibrium temperature over the ocean.

On ``x = sin(lat)`` (equal-area), mixed-layer surface of heat capacity ``C``::

    C dT/dt = S(x, t) (1 - a(x)) - (A + B T) + D d/dx[(1 - x^2) dT/dx]

``S`` is the planet's daily-mean insolation (orbit, obliquity, solar constant),
``a(x) = a0 + a2 P2(x)`` the ice-free planetary albedo (clouds and low sun at high
latitude), ``A + B T`` the outgoing long-wave radiation, ``D`` the meridional heat
transport of atmosphere and ocean as a diffusion. Sea ice is not in this background: the
local ocean model freezes water and applies its own ice albedo. Implicit Euler in time
(tridiagonal), repeated years until the annual cycle repeats.

The ocean's mixed layer exchanges heat with this background,
``Q = lambda (T_ebm(lat, month) - SST)`` (a linearised air-sea flux, Haney 1971), so the
SST is a solved quantity driven by the planet's radiation, not a restored Earth table.
Calibration ``[K]`` (Earth): annual zonal SST and its seasonal range; other planets
change S(x, t) (obliquity, solar constant, year length) and keep the coefficients.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import numpy as np

#: [K] Earth calibration of the ice-free ocean surface (see docs/v5_4_1/defect_sst_root_cause.md)
A_OLR = 208.0  # W m^-2
B_OLR = 2.1  # W m^-2 K^-1
D_TRANSPORT = 0.9  # W m^-2 K^-1 (diffusion in sin(lat))
ALBEDO_0 = 0.295
ALBEDO_2 = 0.15
HEAT_CAPACITY = 1.6e8  # J m^-2 K^-1 (~39 m of sea water)
N_LAT = 90
STEPS_PER_YEAR = 180


@dataclass(frozen=True)
class EbmSolution:
    lat_deg: np.ndarray  # (N,) band centres
    t_month: np.ndarray  # (12, N) degC at the month mid-points
    converged: bool
    years: int

    def at(self, lat_deg, month: int):
        return np.interp(np.asarray(lat_deg, dtype=np.float64), self.lat_deg, self.t_month[month])

    def annual(self, lat_deg):
        return np.mean([self.at(lat_deg, m) for m in range(12)], axis=0)


def _operator(x_edges, d):
    """Tridiagonal diffusion operator d/dx[(1-x^2) dT/dx] on cells between ``x_edges``."""
    xc = 0.5 * (x_edges[1:] + x_edges[:-1])
    dx = np.diff(x_edges)
    k = d * (1.0 - x_edges[1:-1] ** 2) / (xc[1:] - xc[:-1])  # inner edges
    lower = np.zeros_like(xc)
    upper = np.zeros_like(xc)
    diag = np.zeros_like(xc)
    upper[:-1] = k / dx[:-1]
    lower[1:] = k / dx[1:]
    diag[:-1] -= k / dx[:-1]
    diag[1:] -= k / dx[1:]
    return lower, diag, upper


def _solve_tridiag(lower, diag, upper, rhs):
    n = diag.size
    c = np.zeros(n)
    d = np.zeros(n)
    c[0] = upper[0] / diag[0]
    d[0] = rhs[0] / diag[0]
    for i in range(1, n):
        m = diag[i] - lower[i] * c[i - 1]
        c[i] = upper[i] / m if i < n - 1 else 0.0
        d[i] = (rhs[i] - lower[i] * d[i - 1]) / m
    out = np.zeros(n)
    out[-1] = d[-1]
    for i in range(n - 2, -1, -1):
        out[i] = d[i] - c[i] * out[i + 1]
    return out


def solve(planet, a_olr=A_OLR, b_olr=B_OLR, d=D_TRANSPORT, a0=ALBEDO_0, a2=ALBEDO_2,
          c=HEAT_CAPACITY, max_years=60, tol=1e-3) -> EbmSolution:  # fmt: skip
    from .planet import MONTH_FRACTION

    x_edges = np.linspace(-1.0, 1.0, N_LAT + 1)
    x = 0.5 * (x_edges[1:] + x_edges[:-1])
    lat = np.degrees(np.arcsin(x))
    albedo = a0 + a2 * 0.5 * (3.0 * x * x - 1.0)
    lower, diag, upper = _operator(x_edges, d)
    year_s = planet.year_length_days * 86400.0
    dt = year_s / STEPS_PER_YEAR
    t = np.full(N_LAT, 10.0)
    frac = (np.arange(STEPS_PER_YEAR) + 1.0) / STEPS_PER_YEAR
    s_all = np.stack([planet.insolation(lat, f) for f in frac])  # (steps, N)
    month_steps = [int(round(((m % 1.0) % 1.0) * STEPS_PER_YEAR)) % STEPS_PER_YEAR
                   for m in (MONTH_FRACTION % 1.0)]  # fmt: skip
    prev = None
    converged = False
    years_run = 0
    t_month = np.zeros((12, N_LAT))
    for year in range(1, max_years + 1):
        years_run = year
        record = np.zeros((STEPS_PER_YEAR, N_LAT))
        for k in range(STEPS_PER_YEAR):
            # (C/dt + B - D L) T' = C/dt T + S (1 - a) - A
            rhs = c / dt * t + s_all[k] * (1.0 - albedo) - a_olr
            t = _solve_tridiag(-lower, c / dt + b_olr - diag, -upper, rhs)
            record[k] = t
        if prev is not None and np.sqrt(np.mean((record - prev) ** 2)) < tol:
            converged = True
            prev = record
            break
        prev = record
    for m, k in enumerate(month_steps):
        t_month[m] = prev[(k - 1) % STEPS_PER_YEAR]
    return EbmSolution(lat, t_month, converged, years_run)


@lru_cache(maxsize=16)
def _cached(obliquity, day_h, year_d, s0, radius):
    from .planet import Planet

    return solve(Planet(obliquity, day_h, year_d, s0, radius))


def for_planet(planet) -> EbmSolution:
    return _cached(planet.obliquity_deg, planet.day_length_h, planet.year_length_days,
                   planet.solar_constant_wm2, planet.radius_km)  # fmt: skip
