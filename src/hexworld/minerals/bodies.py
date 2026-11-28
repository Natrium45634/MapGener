"""Parametric ore bodies of TZ 5.5.1 §4: real geometry in metres, no voxels.

World frame: x east, y **south** (the map), z up (model datum), metres. Depth below the solid
surface is ``surface(x, y) - z``; the water column of a sea body is separate. Orientation:
strike azimuth from north, clockwise; dip 0–90° towards ``strike + 90°`` (right-hand rule).

Local frame of a body: ``s`` along strike (horizontal), ``d`` down-dip in the plane, ``n`` the
normal. Three primitives carry all 20 morphologies:

* :class:`Plate` — a finite slab around a plane with a true (normal) thickness ``t(s, d)``
  that pinches out towards the ends (vein, tabular, stratabound, seam, laterite, placer,
  paleoplacer, quarry unit, nodules, crust, reservoir);
* :class:`Ellipsoid` — an oriented ellipsoid (lens, pod, stockwork / disseminated /
  breccia / replacement envelope, cavity fill);
* :class:`Pipe` — a truncated cone, a carrot-shaped diatreme (mantle pipe);

and :class:`VeinNetwork` (several plates; the union volume is counted once). Every body
answers ray intersections (vertical column or any drill line), its volume (analytic and by
adaptive quadrature), deterministic quadrature points (for the hex split and the mass), its
footprint, and keeps its lineage (``body_id``, ``geometry_revision``).
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass, field, replace

import numpy as np

DOWN = np.array([0.0, 0.0, -1.0])


def frame(strike_deg: float, dip_deg: float):
    """Unit vectors (s, d, n) in world (x east, y south, z up)."""
    th = math.radians(strike_deg)
    dl = math.radians(dip_deg)
    s = np.array([math.sin(th), -math.cos(th), 0.0])
    h = np.array([math.cos(th), math.sin(th), 0.0])  # strike + 90° (dip direction)
    d = math.cos(dl) * h + math.sin(dl) * DOWN
    n = np.cross(s, d)
    return s, d, n / np.linalg.norm(n)


def _gauss(n: int):
    x, w = np.polynomial.legendre.leggauss(n)
    return x, w


# -- deterministic correlated field (same xyz -> same value, at any query boundary) -------------
@dataclass(frozen=True)
class CorrelatedField:
    """``Z(x)`` with mean 0, variance 1: a sum of random Fourier modes whose wave vectors and
    phases come from a stable hash of ``key`` (no RNG at query time). Anisotropic correlation
    lengths (m) along the local axes (s, d, n) of the body."""

    key: str
    lengths_m: tuple = (200.0, 100.0, 20.0)
    modes: int = 32

    def _modes(self):
        h = hashlib.blake2b(self.key.encode("utf-8"), digest_size=32).digest()
        rng = np.random.default_rng(int.from_bytes(h[:8], "little"))
        k = rng.standard_normal((self.modes, 3)) / np.asarray(self.lengths_m)
        phi = rng.uniform(0.0, 2.0 * math.pi, self.modes)
        return k, phi

    def __call__(self, local_pts) -> np.ndarray:
        p = np.atleast_2d(np.asarray(local_pts, dtype=np.float64))
        k, phi = self._modes()
        return math.sqrt(2.0 / self.modes) * np.cos(p @ k.T + phi).sum(axis=1)


# -- base ---------------------------------------------------------------------------------------
@dataclass
class Body:
    body_id: str
    system_id: str
    morphology: str
    origin_m: tuple  # world (x, y, z) of the local origin
    strike_deg: float = 0.0
    dip_deg: float = 0.0
    formation_event: int = 0  # event_index of the formation (faults only move older bodies)
    formation_age_ma: float = 0.0
    host_id: str = ""
    parent_body_id: str | None = None
    geometry_revision: int = 0
    #: z above which the body was eroded (world m), or None
    cut_z_m: float | None = None
    params: dict = field(default_factory=dict)

    # local <-> world
    def axes(self):
        return frame(self.strike_deg, self.dip_deg)

    def to_local(self, pts) -> np.ndarray:
        s, d, n = self.axes()
        q = np.atleast_2d(np.asarray(pts, dtype=np.float64)) - np.asarray(self.origin_m)
        return np.column_stack([q @ s, q @ d, q @ n])

    def to_world(self, local) -> np.ndarray:
        s, d, n = self.axes()
        lp = np.atleast_2d(np.asarray(local, dtype=np.float64))
        return np.asarray(self.origin_m) + lp[:, :1] * s + lp[:, 1:2] * d + lp[:, 2:3] * n

    # to be provided
    def inside(self, pts) -> np.ndarray:
        raise NotImplementedError

    def analytic_volume(self) -> float | None:
        return None

    def quadrature(self, n: int = 12):
        """Deterministic points (world, (m, 3)) and volume weights (m³) of the body."""
        raise NotImplementedError

    def bbox(self):
        p, _w = self.quadrature(6)
        pad = self.half_extent_pad()
        return p.min(axis=0) - pad, p.max(axis=0) + pad

    def half_extent_pad(self) -> float:
        return 0.0

    # common
    def volume(self, tol: float = 0.005, n0: int = 8, n_max: int = 128) -> dict:
        """Volume by adaptive quadrature (doubling until the relative change < tol) and the
        analytic value where one exists. ``{"volume_m3", "analytic_m3", "rel_change",
        "points"}``."""
        n, prev = n0, None
        while True:
            _p, w = self.quadrature(n)
            v = float(w.sum())
            if prev is not None and abs(v - prev) <= tol * max(abs(v), 1e-300):
                break
            if n >= n_max:
                break
            prev, n = v, n * 2
        rel = abs(v - prev) / max(abs(v), 1e-300) if prev is not None else 0.0
        return {"volume_m3": v, "analytic_m3": self.analytic_volume(), "rel_change": rel,
                "points": n}  # fmt: skip

    def ray_intervals(self, p0, u, length_m: float = 2e5, samples: int = 4000):
        """Intervals (λ0, λ1), m along a unit ray ``p0 + λ u``, λ in [-length, length]:
        sampled inside() + bisection refine (exact primitives override)."""
        u = np.asarray(u, dtype=np.float64)
        u = u / np.linalg.norm(u)
        lam = np.linspace(-length_m, length_m, samples)
        pts = np.asarray(p0, dtype=np.float64) + lam[:, None] * u
        ins = self.inside(pts)
        out = []
        edges = np.nonzero(np.diff(ins.astype(np.int8)))[0]

        def refine(a, b, a_in):
            for _ in range(50):
                m = 0.5 * (a + b)
                if bool(self.inside((np.asarray(p0) + m * u)[None])[0]) == a_in:
                    a = m
                else:
                    b = m
            return 0.5 * (a + b)

        cur = lam[0] if ins[0] else None
        for e in edges:
            x = refine(lam[e], lam[e + 1], bool(ins[e]))
            if ins[e]:
                out.append((cur, x))
                cur = None
            else:
                cur = x
        if cur is not None:
            out.append((cur, lam[-1]))
        return out

    def column(self, x_m: float, y_m: float):
        """Vertical intervals (z_top, z_bottom) at (x, y), top first."""
        iv = self.ray_intervals((x_m, y_m, 0.0), DOWN)
        res = [(-a, -b) for a, b in iv]  # λ along -z: z = -λ
        return sorted(res, key=lambda t: -t[0])

    def _clip_cut(self, pts) -> np.ndarray:
        if self.cut_z_m is None:
            return np.ones(len(pts), dtype=bool)
        return np.asarray(pts)[:, 2] <= self.cut_z_m

    def hex_split(self, layout, n: int = 24) -> dict:
        """Volume of the body per hex ``{(col, row): m³}`` from its quadrature points; every
        point falls in exactly one hex, so the parts sum to the whole (T70)."""
        p, w = self.quadrature(n)
        col, row = layout.world_to_offset(p[:, 0] / 1000.0, p[:, 1] / 1000.0)
        out: dict = {}
        for c, r, ww in zip(col.tolist(), row.tolist(), w.tolist(), strict=True):
            out[(c, r)] = out.get((c, r), 0.0) + ww
        return out

    def erode_above(self, surface_fn, n: int = 24) -> float:
        """Remove the part above the surface ``surface_fn(x, y) -> z``; returns the eroded
        volume (m³, to be credited to a sink by the ledger)."""
        p, w = self.quadrature(n)
        z_s = np.asarray(surface_fn(p[:, 0], p[:, 1]), dtype=np.float64)
        gone = float(w[p[:, 2] > z_s].sum())
        if gone > 0:
            self.cut_z_m = (
                float(np.min(z_s))
                if self.cut_z_m is None
                else min(self.cut_z_m, float(np.min(z_s)))
            )
            self.geometry_revision += 1
        return gone

    def translated(self, dxyz) -> Body:
        o = tuple(float(a + b) for a, b in zip(self.origin_m, dxyz, strict=True))
        return replace(self, origin_m=o, geometry_revision=self.geometry_revision + 1)

    def depth_stats(self, surface_fn, n: int = 16) -> dict:
        p, w = self.quadrature(n)
        dep = np.asarray(surface_fn(p[:, 0], p[:, 1])) - p[:, 2]
        return {"top_m": float(max(dep.min(), 0.0)), "bottom_m": float(dep.max()),
                "mean_m": float(np.average(dep, weights=w))}  # fmt: skip


# -- plate (vein, layer, seam, placer, crust, reservoir, quarry slab) ---------------------------
@dataclass
class Plate(Body):
    """Finite slab: |s| <= L/2, 0 <= d <= W (from the upper edge), |n| <= t(s, d)/2.

    ``t(s, d) = t0 * f_taper(s, d) * exp(sigma Z - sigma^2 / 2)``; taper ``"ends"`` pinches
    to zero at the strike ends and the dip ends (T21), ``"none"`` keeps t0.
    """

    def _p(self, k, default=None):
        return self.params.get(k, default)

    @property
    def L(self):
        return float(self._p("length_m"))

    @property
    def W(self):
        return float(self._p("down_dip_m"))

    @property
    def t0(self):
        return float(self._p("true_thickness_m"))

    def thickness(self, s, dd) -> np.ndarray:
        s = np.asarray(s, dtype=np.float64)
        dd = np.asarray(dd, dtype=np.float64)
        t = np.full(np.broadcast(s, dd).shape, self.t0)
        if self._p("taper", "none") == "ends":
            m = float(self._p("taper_power", 4.0))
            us = np.clip(np.abs(2.0 * s / self.L), 0.0, 1.0)
            ud = np.clip(np.abs(2.0 * dd / self.W - 1.0), 0.0, 1.0)
            t = t * np.sqrt(np.clip(1.0 - us**m, 0.0, None) * np.clip(1.0 - ud**m, 0.0, None))
        sig = float(self._p("thickness_sigma", 0.0))
        if sig > 0:
            z = CorrelatedField(self.body_id + "/t", tuple(self._p("corr_m", (200, 100, 1))))
            pts = np.column_stack([np.ravel(s), np.ravel(dd), np.zeros(np.size(s))])
            t = t * np.exp(sig * z(pts) - 0.5 * sig * sig).reshape(t.shape)
        inside = (np.abs(s) <= self.L / 2) & (dd >= 0) & (dd <= self.W)
        return np.where(inside, np.maximum(t, 0.0), 0.0)

    def analytic_volume(self):
        if self._p("taper", "none") == "none" and float(self._p("thickness_sigma", 0.0)) == 0:
            return self.L * self.W * self.t0
        return None

    def inside(self, pts) -> np.ndarray:
        lp = self.to_local(pts)
        t = self.thickness(lp[:, 0], lp[:, 1])
        return (np.abs(lp[:, 2]) <= 0.5 * t) & (t > 0) & self._clip_cut(np.atleast_2d(pts))

    def quadrature(self, n: int = 12):
        xs, ws = _gauss(n)
        s = 0.5 * self.L * xs
        dd = 0.5 * self.W * (xs + 1.0)
        S, D = np.meshgrid(s, dd, indexing="ij")
        Wt = np.outer(ws * 0.5 * self.L, ws * 0.5 * self.W)
        t = self.thickness(S, D)
        # three points across the thickness (Gauss) for the hex split and the grade
        xn, wn = _gauss(3)
        pts, wts = [], []
        for a, b in zip(xn, wn, strict=True):
            loc = np.column_stack([S.ravel(), D.ravel(), (0.5 * a * t).ravel()])
            pts.append(self.to_world(loc))
            wts.append((Wt * t * 0.5 * b).ravel())
        p = np.vstack(pts)
        w = np.concatenate(wts)
        keep = (w > 0) & self._clip_cut(p)
        return p[keep], w[keep]

    def half_extent_pad(self):
        return self.t0

    def ray_intervals(self, p0, u, length_m: float = 2e5, samples: int = 0):
        """Exact for a constant thickness (slab clipping); with a taper the normal bound is
        iterated at the interval centre (3 passes). Parallel rays handled (T20)."""
        s_, d_, n_ = self.axes()
        u = np.asarray(u, dtype=np.float64)
        u = u / np.linalg.norm(u)
        q0 = np.asarray(p0, dtype=np.float64) - np.asarray(self.origin_m)
        a = np.array([q0 @ s_, q0 @ d_, q0 @ n_])
        v = np.array([u @ s_, u @ d_, u @ n_])
        t = self.t0
        lo, hi = -length_m, length_m
        for _ in range(3):
            bounds = ((-self.L / 2, self.L / 2), (0.0, self.W), (-t / 2, t / 2))
            lo, hi = -length_m, length_m
            ok = True
            for k, (b0, b1) in enumerate(bounds):
                if abs(v[k]) < 1e-15:
                    if not b0 <= a[k] <= b1:
                        ok = False
                        break
                    continue
                l0, l1 = (b0 - a[k]) / v[k], (b1 - a[k]) / v[k]
                lo, hi = max(lo, min(l0, l1)), min(hi, max(l0, l1))
            if not ok or lo >= hi:
                return []
            mid = a + 0.5 * (lo + hi) * v
            t_new = float(self.thickness(mid[0], mid[1]))
            if t_new <= 0:
                return []
            if abs(t_new - t) <= 1e-12 * max(t, 1.0):
                break
            t = t_new
        if self.cut_z_m is not None:
            z0, uz = float(p0[2]), float(u[2])
            if abs(uz) > 1e-15:
                lc = (self.cut_z_m - z0) / uz
                lo, hi = (max(lo, lc), hi) if uz < 0 else (lo, min(hi, lc))
            elif z0 > self.cut_z_m:
                return []
            if lo >= hi:
                return []
        return [(lo, hi)]

    def intercepts(self) -> dict:
        """True thickness and the vertical / horizontal intercepts (T18, T19)."""
        from .units import horizontal_intercept, vertical_intercept

        return {"true_thickness_m": self.t0,
                "vertical_intercept_m": vertical_intercept(self.t0, self.dip_deg),
                "horizontal_intercept_m": horizontal_intercept(self.t0, self.dip_deg)}  # fmt: skip

    def footprint(self) -> np.ndarray:
        """Horizontal projection of the mid-surface (polygon, world m)."""
        loc = np.array([[-self.L / 2, 0, 0], [self.L / 2, 0, 0], [self.L / 2, self.W, 0],
                        [-self.L / 2, self.W, 0]], dtype=float)  # fmt: skip
        return self.to_world(loc)[:, :2]


# -- ellipsoid (lens, pod, envelopes) -----------------------------------------------------------
@dataclass
class Ellipsoid(Body):
    """Semi-axes a (strike), b (down-dip), c (normal), centred on the origin."""

    @property
    def abc(self):
        return tuple(float(x) for x in self.params["semi_axes_m"])

    def analytic_volume(self):
        if self.params.get("host_clip") is None:
            a, b, c = self.abc
            return 4.0 / 3.0 * math.pi * a * b * c
        return None

    def _host_ok(self, pts) -> np.ndarray:
        """Clip to the host unit (T37): ``host_clip = (z_top, z_bottom)`` world m."""
        hc = self.params.get("host_clip")
        if hc is None:
            return np.ones(len(pts), dtype=bool)
        z = np.asarray(pts)[:, 2]
        return (z <= hc[0]) & (z >= hc[1])

    def inside(self, pts) -> np.ndarray:
        pts = np.atleast_2d(pts)
        lp = self.to_local(pts)
        a, b, c = self.abc
        r2 = (lp[:, 0] / a) ** 2 + (lp[:, 1] / b) ** 2 + (lp[:, 2] / c) ** 2
        return (r2 <= 1.0) & self._host_ok(pts) & self._clip_cut(pts)

    def quadrature(self, n: int = 12):
        a, b, c = self.abc
        xr, wr = _gauss(n)
        r = 0.5 * (xr + 1.0)
        wr = 0.5 * wr * r * r  # r² dr on [0, 1]
        xc, wc = _gauss(n)  # cos(theta) on [-1, 1]
        nphi = 2 * n
        phi = (np.arange(nphi) + 0.5) * 2 * math.pi / nphi
        wphi = 2 * math.pi / nphi
        R, C, P = np.meshgrid(r, xc, phi, indexing="ij")
        W = np.einsum("i,j->ij", wr, wc)[:, :, None] * wphi * np.ones_like(P)
        st = np.sqrt(1.0 - C**2)
        loc = np.column_stack([(a * R * st * np.cos(P)).ravel(), (b * R * st * np.sin(P)).ravel(),
                               (c * R * C).ravel()])  # fmt: skip
        p = self.to_world(loc)
        w = (W * a * b * c).ravel()
        keep = self._host_ok(p) & self._clip_cut(p)
        return p[keep], w[keep]

    def ray_intervals(self, p0, u, length_m: float = 2e5, samples: int = 0):
        if self.params.get("host_clip") is not None or self.cut_z_m is not None:
            return Body.ray_intervals(self, p0, u, length_m, 4000)
        a, b, c = self.abc
        u = np.asarray(u, dtype=np.float64)
        u = u / np.linalg.norm(u)
        q0 = self.to_local(np.asarray(p0, dtype=np.float64)[None])[0]
        s_, d_, n_ = self.axes()
        v = np.array([u @ s_, u @ d_, u @ n_])
        sc = np.array([a, b, c])
        q, w = q0 / sc, v / sc
        A, B, C = w @ w, 2 * q @ w, q @ q - 1.0
        disc = B * B - 4 * A * C
        if disc <= 0 or A <= 0:
            return []
        r = math.sqrt(disc)
        return [((-B - r) / (2 * A), (-B + r) / (2 * A))]

    def half_extent_pad(self):
        return 0.0


# -- pipe (kimberlite / lamproite diatreme) -----------------------------------------------------
@dataclass
class Pipe(Body):
    """Vertical truncated cone: radius ``r_top_m`` at the top (origin z) to ``r_bottom_m`` at
    ``depth_m`` below; facies by depth are a property, not a shape change."""

    @property
    def rt(self):
        return float(self.params["r_top_m"])

    @property
    def rb(self):
        return float(self.params["r_bottom_m"])

    @property
    def H(self):
        return float(self.params["depth_m"])

    def radius_at(self, z) -> np.ndarray:
        frac = np.clip((self.origin_m[2] - np.asarray(z, dtype=np.float64)) / self.H, 0.0, 1.0)
        return self.rt + (self.rb - self.rt) * frac

    def analytic_volume(self):
        return math.pi * self.H * (self.rt**2 + self.rt * self.rb + self.rb**2) / 3.0

    def inside(self, pts) -> np.ndarray:
        pts = np.atleast_2d(pts)
        x0, y0, z0 = self.origin_m
        rho = np.hypot(pts[:, 0] - x0, pts[:, 1] - y0)
        z = pts[:, 2]
        return (z <= z0) & (z >= z0 - self.H) & (rho <= self.radius_at(z)) & self._clip_cut(pts)

    def quadrature(self, n: int = 12):
        x0, y0, z0 = self.origin_m
        xz, wz = _gauss(n)
        z = z0 - 0.5 * self.H * (xz + 1.0)
        wz = wz * 0.5 * self.H
        xr, wr = _gauss(n)
        nphi = 2 * n
        phi = (np.arange(nphi) + 0.5) * 2 * math.pi / nphi
        pts, wts = [], []
        for zi, wzi in zip(z, wz, strict=True):
            R = float(self.radius_at(zi))
            r = 0.5 * R * (xr + 1.0)
            wrr = 0.5 * R * wr * r
            rr, pp = np.meshgrid(r, phi, indexing="ij")
            ww = np.outer(wrr, np.full(nphi, 2 * math.pi / nphi)) * wzi
            pts.append(np.column_stack([(x0 + rr * np.cos(pp)).ravel(),
                                        (y0 + rr * np.sin(pp)).ravel(),
                                        np.full(rr.size, zi)]))  # fmt: skip
            wts.append(ww.ravel())
        p = np.vstack(pts)
        w = np.concatenate(wts)
        keep = self._clip_cut(p)
        return p[keep], w[keep]

    def column(self, x_m: float, y_m: float):
        x0, y0, z0 = self.origin_m
        rho = math.hypot(x_m - x0, y_m - y0)
        if rho > max(self.rt, self.rb):
            return []
        top = z0 if self.cut_z_m is None else min(z0, self.cut_z_m)
        bot = z0 - self.H
        # r(z) linear: inside where rho <= r(z)
        if self.rb != self.rt:
            z_eq = z0 - (rho - self.rt) / (self.rb - self.rt) * self.H
            if self.rb < self.rt:  # narrowing downwards (carrot): inside above z_eq
                bot = max(bot, z_eq)
            else:
                top = min(top, z_eq)
        elif rho > self.rt:
            return []
        return [(top, bot)] if top > bot else []


# -- vein network: several plates, the union counted once ---------------------------------------
@dataclass
class VeinNetwork(Body):
    veins: list = field(default_factory=list)  # list[Plate]

    def inside(self, pts) -> np.ndarray:
        pts = np.atleast_2d(pts)
        out = np.zeros(len(pts), dtype=bool)
        for v in self.veins:
            out |= v.inside(pts)
        return out

    def owned_volumes(self, n: int = 24) -> list[float]:
        """Volume each vein owns: its own volume less its overlaps with earlier veins (an
        intersection belongs to the earlier vein, one balance owner). Triple overlaps are
        neglected (thin veins) and reported by :meth:`volume` as a pairwise union."""
        vols = [float(v.quadrature(n)[1].sum()) for v in self.veins]
        owned = list(vols)
        for j in range(len(self.veins)):
            for i in range(j):
                owned[j] -= plate_overlap(self.veins[i], self.veins[j])
        return [max(v, 0.0) for v in owned]

    def quadrature(self, n: int = 12):
        """Points of every vein with weights scaled to the volume the vein owns, so the
        weights add up to the union volume (T22) and the hex split stays exact."""
        own = self.owned_volumes(max(n, 8))
        ps, ws = [], []
        for v, o in zip(self.veins, own, strict=True):
            p, w = v.quadrature(n)
            tot = float(w.sum())
            ps.append(p)
            ws.append(w * (o / tot) if tot > 0 else w)
        return np.vstack(ps), np.concatenate(ws)

    def analytic_volume(self):
        return None

    def ray_intervals(self, p0, u, length_m: float = 2e5, samples: int = 0):
        iv = sorted(x for v in self.veins for x in v.ray_intervals(p0, u, length_m))
        merged = []
        for a, b in iv:
            if merged and a <= merged[-1][1]:
                merged[-1] = (merged[-1][0], max(merged[-1][1], b))
            else:
                merged.append((a, b))
        return merged

    def column(self, x_m, y_m):
        return sorted([(-a, -b) for a, b in self.ray_intervals((x_m, y_m, 0.0), DOWN)],
                      key=lambda t: -t[0])  # fmt: skip


def plate_overlap(a: Plate, b: Plate, n_line: int = 48, n_cross: int = 64) -> float:
    """Volume shared by two plates: integrated on a fine grid around the line where their
    planes meet (the only place they can share volume), clipped to both extents."""
    na, nb = a.axes()[2], b.axes()[2]
    ln = np.cross(na, nb)
    sin_ang = float(np.linalg.norm(ln))
    if sin_ang < 1e-9:
        # parallel planes: shared only if the slabs overlap; integrate over a's points
        p, w = a.quadrature(24)
        return float(w[b.inside(p)].sum())
    ln = ln / sin_ang
    # a point on both planes: minimise along the two normals from a's origin
    oa, ob = np.asarray(a.origin_m), np.asarray(b.origin_m)
    m = np.array([na, nb, ln])
    x0 = np.linalg.solve(m, np.array([na @ oa, nb @ ob, ln @ oa]))
    reach = 0.5 * math.hypot(a.L, a.W) + 0.5 * math.hypot(b.L, b.W) + np.linalg.norm(oa - ob)
    lam = np.linspace(-reach, reach, 4001)
    line = x0 + lam[:, None] * ln
    la, lb = a.to_local(line), b.to_local(line)
    ok = ((np.abs(la[:, 0]) <= a.L / 2) & (la[:, 1] >= 0) & (la[:, 1] <= a.W)
          & (np.abs(lb[:, 0]) <= b.L / 2) & (lb[:, 1] >= 0) & (lb[:, 1] <= b.W))  # fmt: skip
    if not ok.any():
        return 0.0
    step = lam[1] - lam[0]
    l0, l1 = lam[ok].min() - step, lam[ok].max() + step
    half = (a.t0 + b.t0) / sin_ang + max(a.t0, b.t0)
    e1 = np.cross(ln, na)
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(ln, e1)
    xl = l0 + (np.arange(n_line) + 0.5) * (l1 - l0) / n_line
    xc = -half + (np.arange(n_cross) + 0.5) * 2 * half / n_cross
    L_, C1, C2 = np.meshgrid(xl, xc, xc, indexing="ij")
    pts = x0 + L_.reshape(-1, 1) * ln + C1.reshape(-1, 1) * e1 + C2.reshape(-1, 1) * e2
    both = a.inside(pts) & b.inside(pts)
    cell = (l1 - l0) / n_line * (2 * half / n_cross) ** 2
    return float(both.sum() * cell)


# -- morphology -> primitive -------------------------------------------------------------------
PRIMITIVE = {
    "vein": "plate", "tabular": "plate", "stratabound": "plate", "seam": "plate",
    "laterite": "plate", "placer": "plate", "paleoplacer": "plate", "quarry_unit": "plate",
    "nodules": "plate", "crust": "plate", "reservoir": "plate",
    "lens": "ellipsoid", "pod": "ellipsoid", "stockwork": "ellipsoid",
    "disseminated": "ellipsoid", "breccia": "ellipsoid", "replacement": "ellipsoid",
    "cavity_fill": "ellipsoid", "mantle_pipe": "pipe", "vein_network": "network",
}  # fmt: skip


def make_body(morphology: str, body_id: str, system_id: str, origin_m, **kw) -> Body:
    kind = PRIMITIVE[morphology]
    cls = {"plate": Plate, "ellipsoid": Ellipsoid, "pipe": Pipe, "network": VeinNetwork}[kind]
    return cls(body_id=body_id, system_id=system_id, morphology=morphology,
               origin_m=tuple(float(v) for v in origin_m), **kw)  # fmt: skip


# -- grade field with a solved intercept (§5.1) ------------------------------------------------
def solve_grade_field(
    body: Body,
    target: float,
    sigma_log: float,
    rho_kg_m3: float,
    cap: float = 0.95,
    n: int = 12,
    corr_m=(300.0, 150.0, 30.0),
) -> dict:
    """Lognormal grade ``g = min(cap, exp(mu + sigma Z(x)))`` with the intercept ``mu``
    solved so that the mass-weighted mean equals ``target`` (bisection; monotone in mu).
    Returns mu, the achieved weighted grade and a status (``ok`` / ``capped``)."""
    p, w = body.quadrature(n)
    z = CorrelatedField(body.body_id + "/g", corr_m)(body.to_local(p))
    mass = w * rho_kg_m3

    def mean(mu):
        return float((mass * np.minimum(cap, np.exp(mu + sigma_log * z))).sum() / mass.sum())

    lo, hi = math.log(max(target, 1e-300)) - 20.0, math.log(cap) + 5.0
    if mean(hi) < target:
        return {"mu": hi, "achieved": mean(hi), "status": "capped", "target": target}
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if mean(mid) < target:
            lo = mid
        else:
            hi = mid
    mu = 0.5 * (lo + hi)
    return {"mu": mu, "achieved": mean(mu), "status": "ok", "target": target,
            "sigma_log": sigma_log, "cap": cap, "corr_m": list(corr_m)}  # fmt: skip


def grade_at(body: Body, field_params: dict, pts) -> np.ndarray:
    """Grade at world points (0 outside the body); identical for identical xyz (T36)."""
    pts = np.atleast_2d(np.asarray(pts, dtype=np.float64))
    z = CorrelatedField(body.body_id + "/g", tuple(field_params["corr_m"]))(body.to_local(pts))
    g = np.minimum(field_params["cap"], np.exp(field_params["mu"] + field_params["sigma_log"] * z))
    return np.where(body.inside(pts), g, 0.0)


# -- faults: only older bodies move (T26) --------------------------------------------------------
@dataclass(frozen=True)
class Fault:
    event_index: int
    origin_m: tuple
    strike_deg: float
    dip_deg: float
    slip_m: tuple  # displacement of the hanging wall (world m)

    def hanging_wall(self, pts) -> np.ndarray:
        _s, _d, n = frame(self.strike_deg, self.dip_deg)
        q = np.atleast_2d(np.asarray(pts, dtype=np.float64)) - np.asarray(self.origin_m)
        return q @ n > 0

    def apply(self, body: Body) -> Body:
        if body.formation_event >= self.event_index:
            return body  # formed after (or by) the fault: not displaced
        c = np.asarray(body.origin_m, dtype=np.float64)[None]
        if not bool(self.hanging_wall(c)[0]):
            return body
        return body.translated(self.slip_m)
