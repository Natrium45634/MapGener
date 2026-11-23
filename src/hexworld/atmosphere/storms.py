"""Storm climatology (TZ 5.4 sec. 8-9): tropical cyclones, extratropical storms, fronts,
atmospheric rivers. Climatology first: risk fields from the monthly normals and an
ensemble of synthetic tracks with a fixed climatology seed (never the weather-year seed,
so the risk does not change with it, T49).

Tropical cyclones [A]::

    G = f_OHC f_midRH f_shear f_vort f_instab f_basin          (each 0..1, monthly)
      f_OHC   = clip((SST - 26) / 2.5) clip(1 - upwelling / 2 m/day, 0.2, 1)
      f_midRH = clip((RH_col - 0.45) / 0.25)             (column RH as the mid-level proxy)
      f_shear = clip(1 - (|u250 - u850| - 8 m/s) / 12 m/s)
      f_vort  = clip(|f + zeta_850| / 2.5e-5 s^-1)        (no rotation: no genesis, T44)
      f_instab= clip((SST - <SST>_tropics + 1) / 2)        (relative SST, a PI proxy)
      f_basin = connected ocean of >= 2e6 km^2 (no oceanic TC in a closed lake, T45)
    Lambda = sum lambda0 G A dt,  N ~ Poisson(Lambda)       (G is not a storm count)
    track:  x += (U_steer + U_beta) dt + eta,  U_steer = 0.4 u850 + 0.35 u500 + 0.25 u250
    dV/dt = a max(V_PI - V, 0) - b_shear max(dU - 10, 0) / 10 V - b_land I_land V
    Holland: p(r) = p_c + dp exp(-(R_max / r)^B); gradient wind (finite at f = 0)

Extratropical: Eady growth rate ``sigma = 0.31 |f| / N |du/dz|`` with a fixed N (proxy);
fronts: ``|grad T| x deformation`` of the 850 hPa wind; atmospheric rivers: connected
long, narrow IVT features (length >= 2000 km, length / width >= 2). The constants are
calibration [H], documented in ``docs/research/v5_4_parameter_calibration.md``.
"""

from __future__ import annotations

import math

import numpy as np
from scipy import ndimage

DAY = 86400.0
RHO = 1.15
LAMBDA0 = 2.0e-19  # 1/(m^2 s) per unit G [H]
TC_STEP_S = 6 * 3600.0
TC_MAX_DAYS = 15.0
N_BV = 0.012  # Brunt-Vaisala frequency, 1/s (fixed proxy)
STATUSES = (
    "disturbance",
    "depression",
    "storm",
    "mature",
    "weakening",
    "extratropical_transition",
    "dissipated",
)


def clip01(x):
    return np.clip(x, 0.0, 1.0)


def genesis_factors(sst, rh, shear, f, zeta, sea_frac, basin_ok, upwelling=None, tropics_mask=None):
    """The six TC factors (dict of arrays) for one month."""
    upw = np.zeros_like(sst) if upwelling is None else upwelling
    f_ohc = clip01((sst - 26.0) / 2.5) * np.clip(1.0 - upw / 2.0, 0.2, 1.0)
    f_rh = clip01((rh - 0.45) / 0.25)
    f_sh = clip01(1.0 - (shear - 8.0) / 12.0)
    eta = f + zeta
    f_vo = clip01(np.abs(eta) / 2.5e-5) * (f * eta > 0)
    trop = (sea_frac > 0.5) if tropics_mask is None else tropics_mask & (sea_frac > 0.5)
    ref = float(np.mean(sst[trop])) if trop.any() else float(np.mean(sst))
    f_in = clip01((sst - ref + 1.0) / 2.0)
    f_ba = np.where(basin_ok, clip01(sea_frac), 0.0)
    return {
        "ohc": f_ohc,
        "mid_rh": f_rh,
        "shear": f_sh,
        "vorticity": f_vo,
        "instability": f_in,
        "basin": f_ba,
    }


def genesis_index(factors: dict):
    g = np.ones_like(next(iter(factors.values())))
    for v in factors.values():
        g = g * v
    return g


def poisson_rate(g, area_m2, dt_s, lambda0=LAMBDA0):
    """Expected genesis count ``Lambda = sum lambda0 G A dt`` (N09)."""
    return float(np.sum(lambda0 * np.asarray(g) * np.asarray(area_m2) * dt_s))


def large_basins(sea_frac, cell_km, min_km2=2.0e6):
    lab, n = ndimage.label(sea_frac > 0.5)
    if n == 0:
        return np.zeros(sea_frac.shape, dtype=bool)
    sizes = np.bincount(lab.ravel(), minlength=n + 1) * cell_km * cell_km
    ok = sizes >= min_km2
    ok[0] = False
    return ok[lab]


def potential_intensity(sst):
    """[A] upper bound of the max wind from the SST (m/s); 0 below 23 degC."""
    return np.clip(
        np.where(sst >= 26.0, 35.0 + 12.0 * (sst - 26.0), 12.0 * (sst - 23.0)), 0.0, 85.0
    )


def holland_pressure(r_m, p_c, p_env, r_max_m, b=1.5):
    r = np.asarray(r_m, dtype=np.float64)
    dp = p_env - p_c
    with np.errstate(divide="ignore", over="ignore"):
        x = np.where(r > 0, (r_max_m / np.maximum(r, 1e-9)) ** b, np.inf)
    return p_c + dp * np.exp(-x)


def holland_wind(r_m, p_c, p_env, r_max_m, f, b=1.5, rho=RHO):
    """Gradient wind of the Holland profile (m/s); finite for f = 0 and r -> 0."""
    r = np.maximum(np.asarray(r_m, dtype=np.float64), 1.0)
    dp = p_env - p_c
    x = (r_max_m / r) ** b
    cyclo = (b / rho) * x * dp * np.exp(-x)
    half = r * abs(f) / 2.0
    return np.sqrt(cyclo + half * half) - half


def dp_for_vmax(v_max, b=1.5, rho=RHO):
    """Pressure deficit giving ``v_max`` (cyclostrophic Holland maximum), Pa."""
    return rho * math.e * v_max * v_max / b


def eady_growth(f, du_dz, n_bv=N_BV):
    """Eady maximum growth rate, 1/s (a rate, never a storm count)."""
    return 0.31 * np.abs(f) / n_bv * np.abs(du_dz)


def frontogenesis_proxy(t, u, v, cell_m):
    gy, gx = np.gradient(t, cell_m)  # rows grow southward: gy = dT/d(south)
    grad = np.hypot(gx, gy)
    uy_s, ux = np.gradient(u, cell_m)
    vy_s, vx = np.gradient(v, cell_m)
    uy, vy = -uy_s, -vy_s
    deform = np.hypot(ux - vy, vx + uy)
    return grad * deform  # K / m / s


def ar_frequency(ivt_mean, etc_activity, threshold=250.0):
    """Climatological share of days with IVT above ``threshold`` (Gaussian transients)."""
    from scipy.special import erf

    sigma = 0.6 * ivt_mean + 60.0 * etc_activity + 20.0
    z = (threshold - ivt_mean) / (sigma * math.sqrt(2.0))
    return 0.5 * (1.0 - erf(z))


def detect_ar(ivt_e, ivt_n, cell_km, threshold=None, min_length_km=2000.0, min_aspect=2.0):
    """Atmospheric rivers in one IVT field: long, narrow connected features (T57/T58)."""
    mag = np.hypot(ivt_e, ivt_n)
    thr = max(250.0, float(np.percentile(mag, 85))) if threshold is None else threshold
    lab, n = ndimage.label(mag >= thr)
    out = []
    for k in range(1, n + 1):
        rows, cols = np.nonzero(lab == k)
        if rows.size < 3:
            continue
        pts = np.stack([cols * cell_km, -rows * cell_km], axis=1).astype(np.float64)
        c = pts.mean(axis=0)
        cov = np.cov((pts - c).T)
        w, vecs = np.linalg.eigh(cov)
        axis = vecs[:, -1]
        proj = (pts - c) @ axis
        length = float(proj.max() - proj.min()) + cell_km
        width = rows.size * cell_km * cell_km / length
        if length >= min_length_km and length / max(width, 1e-9) >= min_aspect:
            a, b = c + axis * proj.min(), c + axis * proj.max()
            out.append(
                {
                    "length_km": length,
                    "width_km": float(width),
                    "mean_ivt_kg_m_s": float(mag[rows, cols].mean()),
                    "axis_km": [[float(a[0]), float(-a[1])], [float(b[0]), float(-b[1])]],
                    "cells": int(rows.size),
                }
            )
    return out


# -- tracks ----------------------------------------------------------------------------------
class TrackField:
    """Bilinear sampling of the monthly grid fields at metric positions (km, map y south)."""

    def __init__(self, grid, fields, month):
        self.g = grid
        self.f = fields
        self.m = month
        self.ny, self.nx = fields["lat_deg"].shape

    def idx(self, x_km, y_km):
        return (
            (y_km - self.g["y0_km"]) / self.g["cell_km"],
            (x_km - self.g["x0_km"]) / self.g["cell_km"],
        )

    def inside(self, x_km, y_km):
        fy, fx = self.idx(x_km, y_km)
        return -0.5 <= fx <= self.nx - 0.5 and -0.5 <= fy <= self.ny - 0.5

    def at(self, arr, x_km, y_km):
        fy, fx = self.idx(x_km, y_km)
        return float(
            ndimage.map_coordinates(
                np.asarray(arr, dtype=np.float64), [[fy], [fx]], order=1, mode="nearest"
            )[0]
        )


def displacement_km(u_ms, hours):
    """Distance travelled at a constant speed: 5 m/s x 24 h = 432 km (N04)."""
    return np.asarray(u_ms, dtype=np.float64) * hours * 3.6


def run_track(tf: TrackField, x_km, y_km, rng, planet, v0=12.0, event_id="tc"):
    """One synthetic tropical-cyclone track; positions in metric world km (map y south)."""
    G, m = tf.f, tf.m
    u = G["wind_u_ms"][m]
    v = G["wind_v_ms"][m]
    pts = []
    v_max = v0
    noise_e = noise_n = 0.0
    status = "disturbance"
    n_steps = int(TC_MAX_DAYS * DAY / TC_STEP_S)
    max_v = v_max
    for step in range(n_steps):
        if not tf.inside(x_km, y_km):
            pts.append(
                {
                    "t_h": step * 6.0,
                    "x_km": x_km,
                    "y_km": y_km,
                    "v_ms": v_max,
                    "status": "left_domain",
                }
            )
            break
        lat = tf.at(G["lat_deg"], x_km, y_km)
        sst = tf.at(G["t_sea_level_c"][m], x_km, y_km)
        sea = tf.at(G["sea_fraction"], x_km, y_km)
        u_st = sum(w * tf.at(u[k], x_km, y_km) for w, k in ((0.4, 1), (0.35, 2), (0.25, 3)))
        v_st = sum(w * tf.at(v[k], x_km, y_km) for w, k in ((0.4, 1), (0.35, 2), (0.25, 3)))
        shear = math.hypot(
            tf.at(u[3], x_km, y_km) - tf.at(u[1], x_km, y_km),
            tf.at(v[3], x_km, y_km) - tf.at(v[1], x_km, y_km),
        )
        hemi = 1.0 if lat >= 0 else -1.0
        # beta drift: poleward and westward, stronger for larger storms [A]
        b = min(2.5, 1.0 + v_max / 40.0)
        u_beta, v_beta = -0.7 * b, 1.0 * b * hemi
        noise_e = 0.9 * noise_e + math.sqrt(1 - 0.81) * 1.5 * rng.standard_normal()
        noise_n = 0.9 * noise_n + math.sqrt(1 - 0.81) * 1.5 * rng.standard_normal()
        vpi = float(potential_intensity(np.array(sst))) * sea
        land = 1.0 - sea
        dv = (
            max(vpi - v_max, 0.0) / (36 * 3600.0)
            - max(shear - 10.0, 0.0) / 10.0 * v_max / (48 * 3600.0)
            - land * v_max / (18 * 3600.0)
            - v_max / (30 * DAY)
        )
        if vpi < v_max:
            dv -= (v_max - vpi) / (72 * 3600.0)
        v_new = max(v_max + dv * TC_STEP_S, 0.0)
        if v_new < 8.0 and step > 2:
            status = "dissipated"
        elif abs(lat) > 35.0 and sst < 20.0:
            status = "extratropical_transition"
        elif v_new >= 33.0:
            status = "mature" if dv >= 0 else "weakening"
        elif v_new >= 17.0:
            status = "storm" if dv >= 0 or v_new > max_v * 0.95 else "weakening"
        elif v_new >= 12.0:
            status = "depression"
        pts.append(
            {
                "t_h": step * 6.0,
                "x_km": x_km,
                "y_km": y_km,
                "v_ms": v_new,
                "status": status,
                "land": land,
                "sst_c": sst,
            }
        )
        max_v = max(max_v, v_new)
        v_max = v_new
        if status in ("dissipated", "extratropical_transition"):
            break
        # move (east, north) m/s -> km; map y grows southward
        de = float(displacement_km(u_st + u_beta + noise_e, TC_STEP_S / 3600.0))
        dn = float(displacement_km(v_st + v_beta + noise_n, TC_STEP_S / 3600.0))
        x_km += de
        y_km -= dn
    return {
        "event_id": event_id,
        "points": pts,
        "max_wind_ms": max_v,
        "named": max_v >= 17.0,
        "final_status": pts[-1]["status"] if pts else "none",
    }


def stamp(dens, tf: TrackField, track, weight, radius_km=300.0):
    """Add ``weight`` once per track to every cell within ``radius_km`` of its path."""
    ny, nx = dens.shape
    c = tf.g["cell_km"]
    r = int(math.ceil(radius_km / c))
    hit = np.zeros(dens.shape, dtype=bool)
    for p in track["points"]:
        if p.get("status") in ("disturbance", "left_domain") or p["v_ms"] < 17.0:
            continue
        fy, fx = tf.idx(p["x_km"], p["y_km"])
        j0, i0 = int(round(fy)), int(round(fx))
        j_lo, j_hi = max(0, j0 - r), min(ny, j0 + r + 1)
        i_lo, i_hi = max(0, i0 - r), min(nx, i0 + r + 1)
        if j_lo >= j_hi or i_lo >= i_hi:
            continue
        jj, ii = np.mgrid[j_lo:j_hi, i_lo:i_hi]
        inside = np.hypot((jj - fy) * c, (ii - fx) * c) <= radius_km
        hit[j_lo:j_hi, i_lo:i_hi] |= inside
    dens += weight * hit
    return hit


class EventRng:
    """Distributions on a named ``hexworld.rng.Stream`` (no NumPy generator algorithms)."""

    def __init__(self, seed: str, name: str):
        from ..rng import Stream

        self.s = Stream(seed, name)

    def standard_normal(self) -> float:
        return float(self.s.normal())

    def uniform(self, lo: float, hi: float) -> float:
        return float(self.s.uniform(lo, hi))

    def random(self, n=None):
        return self.s.random(n)

    def choice(self, p, size: int) -> np.ndarray:
        c = np.cumsum(np.asarray(p, dtype=np.float64))
        return np.minimum(np.searchsorted(c / c[-1], self.s.random(size), side="right"), c.size - 1)

    def poisson(self, lam: float) -> int:
        """Knuth (small) / normal approximation (large) on the stream."""
        if lam <= 0:
            return 0
        if lam > 50:
            return max(0, int(round(lam + math.sqrt(lam) * self.standard_normal())))
        lim, k, prod = math.exp(-lam), 0, self.s.random()
        while prod > lim:
            k += 1
            prod *= self.s.random()
        return k


def climatology(
    fields: dict, grid: dict, planet, days, rng: EventRng, quality="balanced", upwelling=None
) -> dict:
    """Storm-climatology fields on the atmosphere grid and the ensemble statistics."""
    G = fields
    cell = float(grid["cell_km"])
    cell_m = cell * 1000.0
    lat = G["lat_deg"].astype(np.float64)
    sea_frac = G["sea_fraction"].astype(np.float64)
    basin_ok = large_basins(sea_frac, cell)
    f = planet.coriolis(lat)
    area = np.full(lat.shape, cell_m * cell_m)
    trop = np.abs(lat) < 30.0
    gen = np.zeros((12,) + lat.shape)
    vpi = np.zeros((12,) + lat.shape)
    eady = np.zeros((12,) + lat.shape)
    front = np.zeros((12,) + lat.shape)
    lam = np.zeros(12)
    for m in range(12):
        u = G["wind_u_ms"][m].astype(np.float64)
        v = G["wind_v_ms"][m].astype(np.float64)
        sst = G["t_sea_level_c"][m].astype(np.float64)
        gy_s, gx = np.gradient(u[1], cell_m)
        vy_s, vx = np.gradient(v[1], cell_m)
        zeta = vx + gy_s  # dv/dx - du/dy_north = vx - (-du/dy_south)
        shear = np.hypot(u[3] - u[1], v[3] - v[1])
        fac = genesis_factors(
            sst,
            G["column_rh"][m].astype(np.float64),
            shear,
            f,
            zeta,
            sea_frac,
            basin_ok,
            None if upwelling is None else upwelling[m],
            trop,
        )
        gen[m] = genesis_index(fac)
        vpi[m] = potential_intensity(sst) * (sea_frac > 0.5)
        lam[m] = poisson_rate(gen[m], area, days[m] * DAY)
        dz = np.maximum(G["level_height_m"][m, 1] - G["level_height_m"][m, 0], 1000.0)
        dudz = np.hypot(u[2] - u[1], v[2] - v[1]) / dz
        eady[m] = eady_growth(f, dudz) * (np.abs(lat) > 15.0)
        front[m] = frontogenesis_proxy(G["t_sea_level_c"][m].astype(np.float64), u[1], v[1], cell_m)
    # ensemble of synthetic tracks (climatology seed, independent of the weather year)
    n_ens = {"fast": 60, "balanced": 160, "high": 400}.get(quality, 160)
    total = float(lam.sum())
    dens = np.zeros(lat.shape)
    vmax_field = np.zeros(lat.shape)
    tracks_summary = []
    stats = {"named_share": 0.0, "mature_share": 0.0, "landfall_share": 0.0, "mean_duration_h": 0.0}
    if total > 1e-6:
        flat_w = (gen * area[None]).reshape(12, -1) * np.asarray(days).reshape(12, 1)
        p = flat_w.ravel() / flat_w.sum()
        picks = rng.choice(p, n_ens)
        weight = total / n_ens  # storms per year each track stands for
        named = mature = landfall = 0
        dur = 0.0
        for k, pick in enumerate(picks):
            m, cell_i = divmod(int(pick), lat.size)
            j, i = divmod(cell_i, lat.shape[1])
            x = grid["x0_km"] + (i + rng.uniform(-0.5, 0.5)) * cell
            y = grid["y0_km"] + (j + rng.uniform(-0.5, 0.5)) * cell
            tf = TrackField(grid, G, m)
            tr = run_track(tf, x, y, rng, planet, event_id=f"tcclim-{k}")
            stamp(dens, tf, tr, weight)
            for pt in tr["points"]:
                fy, fx = tf.idx(pt["x_km"], pt["y_km"])
                jj, ii = int(round(fy)), int(round(fx))
                if 0 <= jj < lat.shape[0] and 0 <= ii < lat.shape[1]:
                    vmax_field[jj, ii] = max(vmax_field[jj, ii], pt["v_ms"])
            named += tr["named"]
            mature += tr["max_wind_ms"] >= 33.0
            landfall += any(pt.get("land", 0) > 0.5 and pt["v_ms"] >= 17 for pt in tr["points"])
            dur += tr["points"][-1]["t_h"] if tr["points"] else 0.0
            if k < 12:
                tracks_summary.append(
                    {
                        "event_id": tr["event_id"],
                        "month": m + 1,
                        "max_wind_ms": round(tr["max_wind_ms"], 1),
                        "final_status": tr["final_status"],
                        "points": len(tr["points"]),
                    }
                )
        stats = {
            "named_share": named / n_ens,
            "mature_share": mature / n_ens,
            "landfall_share": landfall / n_ens,
            "mean_duration_h": dur / n_ens,
        }
    eady_day = eady * DAY
    etc_activity = clip01(eady_day / 0.8)
    ivt = np.hypot(G["ivt_east_kg_m_s"], G["ivt_north_kg_m_s"]).astype(np.float64)
    arf = ar_frequency(ivt, etc_activity)
    fr = front / max(float(np.percentile(front, 99)), 1e-30)
    return {
        "tc_genesis_index": gen.astype(np.float32),
        "tc_potential_intensity_ms": vpi.astype(np.float32),
        "tc_passage_per_yr": dens.astype(np.float32),
        "tc_max_wind_ms": vmax_field.astype(np.float32),
        "etc_eady_growth_per_day": eady_day.astype(np.float32),
        "etc_activity": etc_activity.astype(np.float32),
        "front_frequency_index": clip01(fr).astype(np.float32),
        "ar_day_frequency": arf.astype(np.float32),
        "summary": {
            "tc_genesis_per_yr": total,
            "tc_genesis_per_month": [round(float(x), 4) for x in lam],
            "lambda0_per_m2_s": LAMBDA0,
            "ensemble_tracks": n_ens if total > 1e-6 else 0,
            "track_statistics": stats,
            "example_tracks": tracks_summary,
            "quality": {
                "tc": "environment_weighted_stochastic_tracks_CALIBRATED_PROXY",
                "etc": "eady_proxy_fixed_N",
                "fronts": "frontogenesis_proxy",
                "ar": "ivt_threshold_with_gaussian_transients_proxy",
                "surge": "surge_potential_only_no_hydrodynamics",
            },
            "basins_large_enough": int(ndimage.label(basin_ok)[1]),
        },
    }


def tc_rain_per_passage_mm(v_max):
    """[A] mean rain of one tropical-cyclone passage within 300 km, mm."""
    return 40.0 + 1.5 * np.asarray(v_max, dtype=np.float64)
