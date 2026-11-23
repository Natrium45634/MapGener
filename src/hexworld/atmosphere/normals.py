"""``climate_normals``: 12 monthly atmosphere normals on a coarse atmosphere grid.

The atmosphere is solved on the world raster coarsened by an integer factor (TZ 5.4
sec. 21: a hierarchical atmosphere, not every pressure level on every hex): ``fast`` /
``balanced`` / ``high`` aim at ~12k / ~30k / ~80k cells. Rain goes back to the raster
conservatively (bilinear shape, block totals kept exactly); winds and pressure are smooth
fields and are interpolated. The snapshot is immutable once built; the GUI only selects
months / levels of it (no recompute).
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass, field

import numpy as np
from scipy import ndimage

from . import circulation, fv, moisture

TARGET_CELLS = {"fast": 12000, "balanced": 30000, "high": 80000}
from ..version import ATMOSPHERE_ALGORITHM_VERSION as ALGORITHM_VERSION  # noqa: E402


def coarsen_factor(ny: int, nx: int, quality: str) -> int:
    target = TARGET_CELLS.get(quality, 30000)
    return max(1, int(math.ceil(math.sqrt(ny * nx / target))))


def _pad(a, c):
    ny, nx = a.shape[-2:]
    py, px = (-ny) % c, (-nx) % c
    if not (py or px):
        return a
    pad = [(0, 0)] * (a.ndim - 2) + [(0, py), (0, px)]
    return np.pad(a, pad, mode="edge")


def block_mean(a, c, weight=None):
    a = _pad(np.asarray(a, dtype=np.float64), c)
    lead = a.shape[:-2]
    ny, nx = a.shape[-2:]
    r = a.reshape(lead + (ny // c, c, nx // c, c))
    if weight is None:
        return r.mean(axis=(-3, -1))
    w = _pad(np.asarray(weight, dtype=np.float64), c).reshape((ny // c, c, nx // c, c))
    ws = w.sum(axis=(-3, -1))
    return np.where(ws > 0, (r * w).sum(axis=(-3, -1)) / np.maximum(ws, 1e-12), np.nan)


def upsample(a, c, shape, conservative=False):
    """Coarse (..., nyc, nxc) -> fine (..., ny, nx); ``conservative`` keeps block means."""
    a = np.asarray(a, dtype=np.float64)
    if c == 1:
        return a.copy()
    lead = a.shape[:-2]
    nyc, nxc = a.shape[-2:]
    jj = (np.arange(nyc * c) + 0.5) / c - 0.5
    ii = (np.arange(nxc * c) + 0.5) / c - 0.5
    jg, ig = np.meshgrid(jj, ii, indexing="ij")
    flat = a.reshape((-1, nyc, nxc))
    out = np.stack([ndimage.map_coordinates(x, [jg, ig], order=1, mode="nearest") for x in flat])
    if conservative:
        out = np.maximum(out, 0.0)
        bm = out.reshape(-1, nyc, c, nxc, c).mean(axis=(2, 4))
        ratio = np.where(bm > 1e-12, flat / np.maximum(bm, 1e-12), 0.0)
        out = out * np.repeat(np.repeat(ratio, c, axis=1), c, axis=2)
        out = np.where(
            np.repeat(np.repeat(bm <= 1e-12, c, axis=1), c, axis=2),
            np.repeat(np.repeat(flat, c, axis=1), c, axis=2),
            out,
        )
    out = out.reshape(lead + (nyc * c, nxc * c))
    return out[..., : shape[0], : shape[1]]


@dataclass
class AtmosphereNormals:
    """The immutable ``AtmosphereNormalsSnapshot`` (arrays on the atmosphere grid)."""

    algorithm_version: str
    seed_namespace: str
    factor: int
    raster_shape: tuple
    grid: dict  # x0_km, y0_km, cell_km of the atmosphere grid
    lat: np.ndarray
    sea_frac: np.ndarray
    land_h: np.ndarray
    circ: circulation.Circulation
    moist: moisture.MoistureResult
    itcz_score: np.ndarray  # (12, nyc, nxc) 0..1
    itcz_segments: list  # per month: list of segments
    quality: dict = field(default_factory=dict)
    diagnostics: dict = field(default_factory=dict)
    source_hash: str = ""
    inputs: dict = field(default_factory=dict)  # solver inputs (not saved)

    @property
    def shape(self):
        return self.lat.shape

    def months(self, arr):
        return np.asarray(arr).reshape((12,) + self.shape)

    def precip_monthly(self):
        return self.months(self.moist.precip_mm)

    def components(self):
        return self.moist.components_mm.reshape((5, 12) + self.shape)

    def to_raster(self, arr, conservative=False):
        return upsample(arr, self.factor, self.raster_shape, conservative=conservative)

    def snapshot_hash(self) -> str:
        h = hashlib.sha256()
        for a in (
            self.circ.slp_pa,
            self.circ.u,
            self.circ.v,
            self.moist.precip_mm,
            self.moist.w,
            self.itcz_score,
        ):
            h.update(np.ascontiguousarray(a, dtype=np.float32).tobytes())
        return h.hexdigest()[:16]


def itcz(score_conv, rh, t_air, lat, phi_te, phi_h, cell_km):
    """ITCZ-like convergence score (0..1) and its connected segments (TZ sec. 6.1)."""
    ref = 2.0e-5  # kg/m^2/s of vapour convergence (~1.7 mm/day)
    sig = max(1.0, 300.0 / cell_km)
    out = np.zeros_like(score_conv)
    segments = []
    for m in range(12):
        c = ndimage.gaussian_filter(np.maximum(score_conv[m], 0.0), sig, mode="nearest")
        warm = np.clip((t_air[m] - 18.0) / 6.0, 0.0, 1.0)
        moist = np.clip((rh[m] - 0.4) / 0.3, 0.0, 1.0)
        band = np.abs(lat - phi_te[m][None, :]) < 0.75 * phi_h
        s = np.clip(c / ref, 0.0, 1.0) * warm * moist * band
        out[m] = s
        lab, nlab = ndimage.label(s > 0.5)
        segs = []
        if nlab:
            sizes = np.bincount(lab.ravel(), minlength=nlab + 1)
            for k in range(1, nlab + 1):
                area = float(sizes[k]) * cell_km * cell_km
                if area < 2.0e5:
                    continue
                rows, cols = np.nonzero(lab == k)
                segs.append(
                    {
                        "cells": int(sizes[k]),
                        "area_km2": area,
                        "lat_mean_deg": float(lat[rows, cols].mean()),
                        "x_range_km": [
                            float(cols.min() * cell_km),
                            float((cols.max() + 1) * cell_km),
                        ],
                    }
                )
        segments.append(segs)
    return out, segments


def build(
    atm,
    planet,
    lat,
    sea,
    land_h,
    t_sl,
    amp,
    cont,
    cell_km,
    grid,
    sst_monthly=None,
    ice_monthly=None,
    dt_equator_pole_k=50.0,
    lapse_c_per_km=6.5,
    wind_strength=1.0,
    seed_namespace="atmosphere/5.4",
) -> AtmosphereNormals:
    """Build the normals from the raster fields of the world climate."""
    from ..stages.climate import potential_et

    ny, nx = lat.shape
    c = coarsen_factor(ny, nx, getattr(atm, "quality", "balanced"))
    cc = cell_km * c
    sea_f = block_mean(sea.astype(np.float64), c)
    sea_c = sea_f > 0.5
    lat_c = block_mean(lat, c)
    lh = block_mean(land_h, c)
    tsl = block_mean(t_sl, c)
    amp_c = block_mean(amp, c)
    cont_c = block_mean(cont, c)
    sst_c = ice_c = None
    if sst_monthly is not None:
        w = np.broadcast_to(sea, sst_monthly.shape[-2:])
        sst_c = np.stack(
            [block_mean(np.nan_to_num(sst_monthly[m], nan=-1.8), c, w) for m in range(12)]
        )
        sst_c = np.where(np.isnan(sst_c), np.maximum(tsl, -1.8)[None], sst_c)
    if ice_monthly is not None:
        ice_c = np.stack([block_mean(np.nan_to_num(ice_monthly[m]), c) for m in range(12)])
        ice_c = np.clip(ice_c / np.maximum(sea_f, 1e-9), 0.0, 1.0)
    circ = circulation.build(
        planet,
        lat_c,
        sea_c,
        lh,
        tsl,
        amp_c,
        cont_c,
        cc,
        sst_monthly=sst_c,
        dt_equator_pole_k=dt_equator_pole_k,
        lapse_c_per_km=lapse_c_per_km,
        wind_strength=wind_strength,
    )
    nyc, nxc = lat_c.shape
    n = nyc * nxc
    mesh = fv.raster_mesh(nyc, nxc, cc * 1000.0)
    days = circ.months_days
    t_air = circ.t_air.reshape(12, n)
    sst = (sst_c if sst_c is not None else circ.t_sl).reshape(12, n)
    sst = np.where(sea_c.ravel()[None], sst, np.maximum(circ.t_sl.reshape(12, n), -1.8))
    ice = ice_c.reshape(12, n) if ice_c is not None else np.clip((-1.0 - sst) / 1.5, 0.0, 1.0)
    pet = np.stack([potential_et(t_air[m]) * days[m] / days.sum() for m in range(12)])
    cu = np.stack([circ.u[:, 0], circ.u[:, 1]], axis=1).reshape(12, 2, n).astype(np.float64)
    cv = np.stack([circ.v[:, 0], circ.v[:, 1]], axis=1).reshape(12, 2, n).astype(np.float64)
    mr = moisture.solve(
        mesh,
        cu,
        cv,
        t_air,
        sst,
        ice,
        sea_f.ravel(),
        lh,
        lat_c,
        planet,
        days,
        cc * 1000.0,
        pet_mm=pet,
        picard={"fast": 2, "balanced": 3, "high": 4}.get(atm.quality, 3),
    )
    score, segs = itcz(
        mr.conv.reshape((12,) + lat_c.shape),
        mr.rh.reshape((12,) + lat_c.shape),
        circ.t_air,
        lat_c,
        circ.thermal_equator_deg,
        circ.hadley_edge_deg,
        cc,
    )
    gx0 = float(grid["x0_km"]) - 0.5 * cell_km + 0.5 * cc
    gy0 = float(grid["y0_km"]) - 0.5 * cell_km + 0.5 * cc
    p_ann = mr.precip_mm.sum(axis=0)
    land = ~sea_c.ravel()
    tropics = np.abs(lat_c) < 25.0
    has_tropics = bool(tropics.any())
    quality = dict(circ.quality)
    quality.update(
        {
            "moisture": mr.diagnostics["method"],
            "boundary_forcing": mr.diagnostics["boundary_forcing"],
            "itcz": "convergence_then_connected_components" if has_tropics else "no_tropics_on_map",
            "sst": "v5.3_ocean" if sst_monthly is not None else "prescribed_from_air_temperature",
            "domain": "isolated_approximation"
            if getattr(atm, "domain_mode", "isolated_approximation") == "isolated_approximation"
            else atm.domain_mode,
            "storm_component": "not_attributed_in_5.4a",
        }
    )
    diag = {
        "factor": c,
        "atmosphere_cell_km": cc,
        "cells": n,
        "hadley_edge_deg": circ.hadley_edge_deg,
        "precip_land_mean_mm_yr": float(p_ann[land].mean()) if land.any() else None,
        "precip_sea_mean_mm_yr": float(p_ann[~land].mean()) if (~land).any() else None,
        "evap_mean_mm_yr": float(mr.evap_mm.sum(axis=0).mean()),
        "column_vapour_mean_kg_m2": float(mr.w.mean()),
        "budget": {
            "annual_relative_residual": mr.diagnostics["annual_relative_residual"],
            "picard_last_change_max": mr.diagnostics["picard_last_change_max"],
            "months": mr.ledger,
        },
        "itcz_segments_per_month": [len(s) for s in segs],
    }
    out = AtmosphereNormals(
        ALGORITHM_VERSION,
        seed_namespace,
        c,
        (ny, nx),
        {"x0_km": gx0, "y0_km": gy0, "cell_km": cc},
        lat_c,
        sea_f,
        lh,
        circ,
        mr,
        score,
        segs,
        quality,
        diag,
    )
    out.source_hash = out.snapshot_hash()
    out.inputs = {
        "mesh": mesh,
        "sst": sst,
        "ice": ice,
        "pet": pet,
        "t_air": t_air,
        "cu": cu,
        "cv": cv,
        "days": days,
        "planet": planet,
        "picard": {"fast": 2, "balanced": 3, "high": 4}.get(atm.quality, 3),
    }
    return out
