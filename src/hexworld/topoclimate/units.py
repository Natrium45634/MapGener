"""Sub-hex samples, the sub-hex surface and terrain units (TZ 4.7, 1, 2).

Sampling [H, after TZ 1.2]
    Every sampled hex is split into its six triangles (centre, vertex k, vertex k + 1),
    equal areas; each triangle gets the same fixed low-discrepancy points (R2 sequence)
    mapped with ``x = (1 - sqrt u) A + sqrt u (1 - v) B + sqrt u v C``. Weights are equal
    plan areas. The pattern is the same in every hex (deterministic, independent of chunks).

Surface
    ``z = coarse(x, y) + detail(x, y)``. ``coarse`` is the hex surface of the final v4
    snapshot, barycentric between the hex centre and its two vertex heights (a vertex =
    mean of its three hexes): continuous, exact at every hex centre, no invented relief.
    ``detail`` is the v3 mountain plan's relief at the point minus its value at the hex
    centre (the hex value already holds the centre): real ridges and valleys inside
    mountain hexes, ``quality = sampled``; elsewhere ``coarse_only``.

Water inside a hex
    A sample is ocean when it lies at or below the sea level and its own hex or the hex
    across its triangle's outer edge is ocean; lake likewise against that lake's level;
    everything else is dry. So a coastal hex has fractional dry / lake / ocean areas.

Terrain units [H, deterministic stratification instead of k-medoids - ADR in docs]
    Hard strata: ocean, lake, and on dry land landform (ridge, valley floor, steep rock,
    open slope) x elevation band (in-hex terciles, as the relief allows) x aspect (N, E,
    S, W, flat). A unit is a non-empty stratum; its representative is a real sample (the
    medoid of the stratum in normalised features), its area the sum of its weights.
"""

from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree

SAMPLES = {"preview": (6, 24), "balanced": (12, 48), "reference": (24, 96)}
LANDFORMS = ("open_slope", "ridge", "valley_floor", "steep_rock")
SURFACE = ("dry", "lake", "ocean")
_PLASTIC = 1.32471795724474602596


def r2_points(n: int) -> np.ndarray:
    """First ``n`` points of the R2 low-discrepancy sequence in the unit square."""
    a1, a2 = 1.0 / _PLASTIC, 1.0 / _PLASTIC**2
    k = np.arange(1, n + 1)
    return np.column_stack([(0.5 + a1 * k) % 1.0, (0.5 + a2 * k) % 1.0])


def hex_offsets(n_samples: int, radius: float):
    """(offsets (n, 2) world km, triangle index (n,)) of the sample pattern of one hex."""
    per = max(1, n_samples // 6)
    uv = r2_points(per)
    su = np.sqrt(uv[:, 0])
    out, tri = [], []
    for k in range(6):
        a0, a1 = np.deg2rad(60.0 * k), np.deg2rad(60.0 * (k + 1))
        b = radius * np.array([np.cos(a0), np.sin(a0)])
        c = radius * np.array([np.cos(a1), np.sin(a1)])
        pts = (su * (1.0 - uv[:, 1]))[:, None] * b + (su * uv[:, 1])[:, None] * c
        out.append(pts)
        tri.append(np.full(per, k))
    return np.concatenate(out), np.concatenate(tri)


class CoarseSurface:
    """Barycentric hex surface: centre value + vertex means (TZ 1.2, ``coarse_only``)."""

    def __init__(self, layout, z):
        self.layout = layout
        self.z = np.asarray(z, dtype=np.float64).ravel()
        hx, hy = layout.centers()
        self.cx, self.cy = hx.ravel(), hy.ravel()
        self.r = layout.radius
        tree = cKDTree(np.column_stack([self.cx, self.cy]))
        n = self.cx.size
        self.vz = np.zeros((n, 6))
        self.nbr = np.full((n, 6), -1, dtype=np.int64)
        for k in range(6):
            a = np.deg2rad(60.0 * k)
            vx, vy = self.cx + self.r * np.cos(a), self.cy + self.r * np.sin(a)
            d, idx = tree.query(np.column_stack([vx, vy]), k=3)
            ok = d <= 1.05 * self.r
            vals = np.where(ok, self.z[idx], np.nan)
            cnt = ok.sum(axis=1)
            self.vz[:, k] = np.where(cnt > 0, np.nansum(vals, axis=1) / np.maximum(cnt, 1), self.z)
            # the hex across the edge between vertex k and k + 1
            am = np.deg2rad(60.0 * k + 30.0)
            ex = self.cx + 2.0 * self.r * np.cos(np.deg2rad(30.0)) * np.cos(am)
            ey = self.cy + 2.0 * self.r * np.cos(np.deg2rad(30.0)) * np.sin(am)
            de, ie = tree.query(np.column_stack([ex, ey]))
            self.nbr[:, k] = np.where(de < 0.5 * self.r, ie, -1)

    def locate(self, x, y):
        """(hex index, triangle index, barycentric weights) of world points; -1 outside."""
        x = np.asarray(x, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        col, row = self.layout.world_to_offset(x, y)
        inside = self.layout.inside(col, row)
        h = np.where(inside, row * self.layout.width + col, -1).astype(np.int64)
        hc = np.maximum(h, 0)
        dx, dy = x - self.cx[hc], y - self.cy[hc]
        ang = np.mod(np.arctan2(dy, dx), 2.0 * np.pi)
        k = np.minimum((ang / (np.pi / 3.0)).astype(np.int64), 5)
        a0, a1 = k * np.pi / 3.0, (k + 1) * np.pi / 3.0
        bx, by = self.r * np.cos(a0), self.r * np.sin(a0)
        cx_, cy_ = self.r * np.cos(a1), self.r * np.sin(a1)
        det = bx * cy_ - cx_ * by
        wb = (dx * cy_ - cx_ * dy) / det
        wc = (bx * dy - dx * by) / det
        wa = 1.0 - wb - wc
        return h, k, (wa, wb, wc)

    def __call__(self, x, y):
        h, k, (wa, wb, wc) = self.locate(x, y)
        hc = np.maximum(h, 0)
        v0 = self.vz[hc, k]
        v1 = self.vz[hc, (k + 1) % 6]
        out = wa * self.z[hc] + wb * v0 + wc * v1
        return np.where(h >= 0, out, np.nan)


class SubHexSurface:
    """``coarse + (detail(x) - detail(hex centre))`` with the v3 mountain sampler."""

    def __init__(
        self, coarse: CoarseSurface, detail=None, detail_hexes=None, max_elevation_m=8800.0
    ):
        self.coarse = coarse
        self.detail = detail
        self.max_z = max_elevation_m
        n = coarse.z.size
        self.has_detail = np.zeros(n, dtype=bool)
        self.center_detail = np.zeros(n)
        if detail is not None and detail_hexes is not None and detail_hexes.size:
            self.has_detail[detail_hexes] = True
            self.center_detail[detail_hexes] = detail(
                coarse.cx[detail_hexes], coarse.cy[detail_hexes]
            )[0]

    def __call__(self, x, y, want_structure=False):
        z = self.coarse(x, y)
        h, _k, _w = self.coarse.locate(x, y)
        struct = np.zeros(np.shape(z))
        valley = np.zeros(np.shape(z))
        use = (h >= 0) & self.has_detail[np.maximum(h, 0)]
        if self.detail is not None and use.any():
            rel, st, va = self.detail(np.asarray(x)[use], np.asarray(y)[use])
            z = z.copy()
            z[use] = np.minimum(z[use] + rel - self.center_detail[h[use]], self.max_z)
            struct[use] = st
            valley[use] = va
        if want_structure:
            return z, struct, valley
        return z


def gradient(surface, x, y, h_km):
    """(dz/dEast, dz/dNorth) dimensionless, central differences at ``h_km`` (world y south)."""
    hm = h_km * 1000.0
    ze = surface(x + h_km, y)
    zw = surface(x - h_km, y)
    zn = surface(x, y - h_km)  # north = -y in the world
    zs = surface(x, y + h_km)
    p = (ze - zw) / (2.0 * hm)
    q = (zn - zs) / (2.0 * hm)
    return np.nan_to_num(p), np.nan_to_num(q)


def build_samples(layout, hexes, mountain_hexes, quality="balanced"):
    """Sample points of ``hexes``: (x, y, hex, triangle, weight) - weights sum to 1 per hex."""
    n_flat, n_mtn = SAMPLES[quality]
    hx, hy = layout.centers()
    cx, cy = hx.ravel(), hy.ravel()
    mountain = np.isin(hexes, mountain_hexes)
    parts = []
    for flag, n in ((False, n_flat), (True, n_mtn)):
        sel = hexes[mountain == flag]
        if sel.size == 0:
            continue
        off, tri = hex_offsets(n, layout.radius)
        m = off.shape[0]
        parts.append(
            (
                (cx[sel][:, None] + off[None, :, 0]).ravel(),
                (cy[sel][:, None] + off[None, :, 1]).ravel(),
                np.repeat(sel, m),
                np.tile(tri, sel.size),
                np.full(sel.size * m, 1.0 / m),
            )
        )
    x, y, h, t, w = (np.concatenate([p[i] for p in parts]) for i in range(5))
    order = np.lexsort((np.arange(h.size), h))
    return x[order], y[order], h[order], t[order], w[order]


def water_type(z, sample_hex, sample_tri, nbr, ocean, lake, lake_level, sea_level):
    """0 dry, 1 lake, 2 ocean for every sample (TZ 1.2, fractional coastal hexes)."""
    ocean = np.asarray(ocean, dtype=bool).ravel()
    lake = np.asarray(lake, dtype=bool).ravel()
    level = np.asarray(lake_level, dtype=np.float64).ravel()
    across = nbr[sample_hex, sample_tri]
    ac = np.maximum(across, 0)
    out = np.zeros(z.size, dtype=np.uint8)
    near_ocean = ocean[sample_hex] | ((across >= 0) & ocean[ac])
    out[near_ocean & (z <= sea_level)] = 2
    own_lake = lake[sample_hex] & (z <= np.nan_to_num(level[sample_hex], nan=-1e9) + 1e-6)
    nb_lake = (across >= 0) & lake[ac] & (z <= np.nan_to_num(level[ac], nan=-1e9) + 1e-6)
    out[(out == 0) & (own_lake | nb_lake)] = 1
    # a whole ocean / lake hex keeps its water where the sample is not above the surface
    out[(out == 0) & ocean[sample_hex] & (z <= sea_level)] = 2
    return out


def midrank(values, group, n_groups):
    """In-group percentile rank ``(less + 0.5 equal) / count`` of every value (TZ 2.2)."""
    v = np.asarray(values, dtype=np.float64)
    g = np.asarray(group, dtype=np.int64)
    order = np.lexsort((v, g))
    rank = np.empty(v.size)
    gs, vs = g[order], v[order]
    start = np.searchsorted(gs, gs, side="left")
    # ties: first and last position of equal (group, value)
    key_change = np.ones(v.size, dtype=bool)
    key_change[1:] = (gs[1:] != gs[:-1]) | (vs[1:] != vs[:-1])
    first = np.maximum.accumulate(np.where(key_change, np.arange(v.size), 0))
    last_flag = np.ones(v.size, dtype=bool)
    last_flag[:-1] = key_change[1:]
    last = np.minimum.accumulate(np.where(last_flag, np.arange(v.size), v.size)[::-1])[::-1]
    count = np.bincount(g, minlength=n_groups)[gs]
    less = first - start
    equal = last - first + 1
    rank[order] = (less + 0.5 * equal) / np.maximum(count, 1)
    return rank


def stratify(sample_hex, wtype, z, slope_deg, aspect_rad, structure, valley, n_hex, k_max=16):
    """Stratum code of every sample (hard strata, see the module docstring)."""
    dry = wtype == 0
    lf = np.zeros(z.size, dtype=np.int64)
    lf[dry & (structure > 0.55)] = 1
    lf[dry & (valley > 0.45) & (slope_deg < 15.0)] = 2
    lf[dry & (slope_deg > 35.0)] = 3
    # elevation bands from the in-hex relief of the dry samples
    zmin = np.full(n_hex, np.inf)
    zmax = np.full(n_hex, -np.inf)
    np.minimum.at(zmin, sample_hex[dry], z[dry])
    np.maximum.at(zmax, sample_hex[dry], z[dry])
    relief = np.where(np.isfinite(zmax - zmin), zmax - zmin, 0.0)
    nb = np.where(relief > 300.0, 3, np.where(relief > 100.0, 2, 1))
    rank = midrank(np.where(dry, z, np.inf), sample_hex, n_hex)
    band = np.minimum((rank * nb[sample_hex]).astype(np.int64), nb[sample_hex] - 1)
    asp = np.where(
        slope_deg > 5.0,
        np.floor(np.mod(np.nan_to_num(aspect_rad) + np.pi / 4.0, 2.0 * np.pi) / (np.pi / 2.0)),
        4,
    ).astype(np.int64)
    asp = np.clip(asp, 0, 4)

    def codes(asp_):
        c = 2 + (lf * 3 + band) * 5 + asp_
        return np.where(wtype == 2, 0, np.where(wtype == 1, 1, c))

    code = codes(asp)
    # too many units in a hex: merge the aspects to north-facing / south-facing / flat
    key = sample_hex * 64 + code
    uniq = np.unique(key)
    per_hex = np.bincount(uniq // 64, minlength=n_hex)
    many = per_hex[sample_hex] > k_max
    if many.any():
        asp2 = np.where(asp == 4, 4, np.where((asp == 0) | (asp == 1), 0, 2))
        code = np.where(many, codes(asp2), code)
    # still too many: drop the aspect, then the landform split (bands stay: the vertical
    # distribution is what the belts need) - the cap of the config always holds
    for fallback in (lambda: codes(np.full_like(asp, 4)), lambda: _band_only(wtype, band)):
        key = sample_hex * 64 + code
        per_hex = np.bincount(np.unique(key) // 64, minlength=n_hex)
        many = per_hex[sample_hex] > k_max
        if not many.any():
            break
        code = np.where(many, fallback(), code)
    return code, lf, band, relief


def _band_only(wtype, band):
    c = 2 + band * 5 + 4
    return np.where(wtype == 2, 0, np.where(wtype == 1, 1, c))
