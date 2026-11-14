"""v2 stage ``continental_blocks``: refined continental crust in world coordinates.

Input is the v1.0 crust field, used strictly as a geological *prior* (it is
never overwritten), plus the world tectonic skeleton. The stage builds a signed
field ``continentality_km`` (> 0 on continental crust, roughly the distance to
the crust edge) out of four structural ingredients, large scale first:

1. the prior masses, optionally grown / shrunk per block (``continent_size_variety``);
2. **structure**: bridges that weld neighbouring blocks, and rifts that split
   them - driven by the continent profile or ``major_landmass_count``. Rifts
   follow a plate boundary that crosses the block whenever one exists;
3. **tectonics**: convergence between two continental margins closes the gap
   (suture), divergence thins the crust along the boundary (rift, narrow sea);
4. multi-scale coast irregularity (wavelength-proportional, tapered in deep
   interiors) - detail only, never the reason a continent exists.

The result is a pure function of world position: hexes of any window and
resolution sample the same blocks. ``continental_fraction`` is its sigmoid.

This is a *snapshot* of tectonic influence, not a simulation of history; a
later history stage can replace the structural step behind the same interface.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
from scipy import ndimage
from scipy.spatial import cKDTree

from ..config import Config, ContinentParams
from ..noise import DomainWarp, FractalNoise
from ..rng import Stream
from .boundaries import TYPE_CONVERGENT, TYPE_DIVERGENT, TYPE_QUIET, TYPE_TRANSFORM
from .crust import CrustModel
from .skeleton import WorldSkeleton, accumulate

_CHUNK = 100_000
_MAX_BRIDGE_KM = 5200.0
_AUTO_WELD_KM = 1800.0
_SIDE_KM = 150.0
#: Crust this far inside the block edge is normally above water (beyond the shelf).
#: Structure is counted on this proxy so that welds and rifts act on future *land*.
_LAND_PROXY_KM = 200.0


def smoothstep(t):
    t = np.clip(t, 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def _smax(a, b, k):
    return 0.5 * (a + b + np.sqrt((a - b) ** 2 + k * k))


def _smin(a, b, k):
    return 0.5 * (a + b - np.sqrt((a - b) ** 2 + k * k))


def _segment_distance(x, y, seg):
    x1, y1, x2, y2 = seg[:4]
    dx, dy = x2 - x1, y2 - y1
    length2 = max(dx * dx + dy * dy, 1e-9)
    t = np.clip(((x - x1) * dx + (y - y1) * dy) / length2, 0.0, 1.0)
    return np.hypot(x - (x1 + t * dx), y - (y1 + t * dy))


@dataclass
class ContinentModel:
    crust_model: CrustModel
    skeleton: WorldSkeleton
    params: ContinentParams
    warp: DomainWarp | None
    noise: FractalNoise
    offset_grid: np.ndarray
    weld_gain: np.ndarray  # km per skeleton sample
    rift_gain: np.ndarray
    bridges: list = field(default_factory=list)  # (x1, y1, x2, y2, half_width_km)
    rift_points: np.ndarray = field(default_factory=lambda: np.zeros((0, 2)))
    rift_width: np.ndarray = field(default_factory=lambda: np.zeros(0))
    #: Uniform growth (km) that keeps the world share of continental crust of the prior.
    balance_km: float = 0.0
    #: (centre_x, centre_y, gain, radius_km); radius <= 0 means a uniform rescale.
    remap: tuple | None = None
    operations: list = field(default_factory=list)
    warnings: list = field(default_factory=list)
    _rift_tree: cKDTree | None = None

    # -- ingredients ---------------------------------------------------------
    def _prior(self, x, y):
        if self.remap is not None:
            # Profile-level rearrangement of the prior pattern (see build_continents):
            # sampling farther from the centre pulls the masses towards it.
            cx, cy, gain, radius = self.remap
            dx, dy = x - cx, y - cy
            k = (
                1.0 + gain
                if radius <= 0
                else 1.0 + gain * np.exp(-(dx * dx + dy * dy) / (radius * radius))
            )
            x, y = cx + dx * k, cy + dy * k
        if self.warp is not None:
            x, y = self.warp(x, y)
        return self.crust_model.continentality(x, y)

    def _offset(self, x, y):
        sk = self.skeleton
        coords = np.vstack([(y.ravel() - sk.y0) / sk.cell_km, (x.ravel() - sk.x0) / sk.cell_km])
        return ndimage.map_coordinates(self.offset_grid, coords, order=1, mode="nearest").reshape(
            x.shape
        )

    def _tectonic(self, x, y):
        def contribution(idx, d, _side):
            return [
                self.weld_gain[idx] * np.exp(-0.5 * (d / 260.0) ** 2)
                - self.rift_gain[idx] * np.exp(-0.5 * (d / 110.0) ** 2)
            ]

        return accumulate(self.skeleton, x, y, contribution, 800.0, 1)[0]

    def _compose(self, prior, offset, tectonic, noise, x, y):
        a = prior + offset + self.balance_km + tectonic
        # structural welds override the local tectonic thinning they cross
        for seg in self.bridges:
            ridge = np.minimum(1.6 * (seg[4] - _segment_distance(x, y, seg)), 650.0)
            a = _smax(a, ridge, 60.0)
        if self.rift_points.shape[0]:
            if self._rift_tree is None:
                self._rift_tree = cKDTree(self.rift_points)
            d, i = self._rift_tree.query(np.column_stack([x.ravel(), y.ravel()]))
            a = _smin(a, (d - self.rift_width[i]).reshape(a.shape), 40.0)
        taper = (1.0 - 0.6 * smoothstep((a - 250.0) / 500.0)) * (
            1.0 - 0.5 * smoothstep((-a - 500.0) / 600.0)
        )
        return a + noise * taper

    # -- public --------------------------------------------------------------
    def continentality(self, x, y) -> np.ndarray:
        """Signed km: > 0 continental crust, < 0 oceanic. Any world coordinates."""
        x = np.asarray(x, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        shape = np.broadcast(x, y).shape
        xf = np.broadcast_to(x, shape).ravel()
        yf = np.broadcast_to(y, shape).ravel()
        out = np.empty(xf.size)
        for start in range(0, xf.size, _CHUNK):
            sl = slice(start, start + _CHUNK)
            px, py = xf[sl], yf[sl]
            out[sl] = self._compose(
                self._prior(px, py),
                self._offset(px, py),
                self._tectonic(px, py),
                self.noise(px, py),
                px,
                py,
            )
        return out.reshape(shape)

    def fraction_of(self, continentality) -> np.ndarray:
        return 1.0 / (1.0 + np.exp(-np.asarray(continentality) / self.params.transition_km))

    def describe(self) -> dict:
        return {
            "kind": "prior-crust + structure(bridges, rifts) + tectonics + coast detail",
            "bridges_km": [[round(float(v), 3) for v in seg] for seg in self.bridges],
            "rift_samples": int(self.rift_points.shape[0]),
            "balance_km": round(float(self.balance_km), 3),
            "remap": [round(float(v), 3) for v in self.remap] if self.remap else None,
            "operations": self.operations,
            "transition_km": self.params.transition_km,
        }


def _label(field_raster, level: float = _LAND_PROXY_KM):
    labels, count = ndimage.label(field_raster > level)
    return labels, count


def _majors(labels, count, skeleton, major_area):
    if count == 0:
        return np.zeros(0, dtype=np.int64), np.zeros(1)
    area = np.bincount(labels[skeleton.in_sim], minlength=count + 1) * skeleton.cell_km**2
    area[0] = 0.0
    return np.nonzero(area >= major_area)[0], area


def _closest_pair(labels, majors, skeleton):
    """Smallest gap between two major blocks: (gap_km, (x1, y1), (x2, y2), i, j)."""
    best = None
    for a_pos, i in enumerate(majors[:-1]):
        dist, inds = ndimage.distance_transform_edt(labels != i, return_indices=True)
        for j in majors[a_pos + 1 :]:
            cells = np.nonzero(labels == j)
            k = int(np.argmin(dist[cells]))
            gap = float(dist[cells][k]) * skeleton.cell_km
            if best is None or gap < best[0]:
                qj = (cells[0][k], cells[1][k])
                pi = (inds[0][qj], inds[1][qj])
                best = (gap, pi, qj, int(i), int(j))
    if best is None:
        return None
    gap, pi, qj, i, j = best
    p1 = (float(skeleton.x[pi]), float(skeleton.y[pi]))
    p2 = (float(skeleton.x[qj]), float(skeleton.y[qj]))
    return gap, p1, p2, i, j


def _point_labels(labels, skeleton):
    col = np.clip(np.rint((skeleton.bx - skeleton.x0) / skeleton.cell_km), 0, labels.shape[1] - 1)
    row = np.clip(np.rint((skeleton.by - skeleton.y0) / skeleton.cell_km), 0, labels.shape[0] - 1)
    return labels[row.astype(np.int64), col.astype(np.int64)]


def _point_values(raster, skeleton):
    col = np.clip(np.rint((skeleton.bx - skeleton.x0) / skeleton.cell_km), 0, raster.shape[1] - 1)
    row = np.clip(np.rint((skeleton.by - skeleton.y0) / skeleton.cell_km), 0, raster.shape[0] - 1)
    return raster[row.astype(np.int64), col.astype(np.int64)]


def _rift_along_boundary(labels, block, block_area, skeleton, tried):
    """Skeleton samples of the plate boundary that best cuts ``block`` (or None)."""
    inside = _point_labels(labels, skeleton) == block
    if not inside.any():
        return None
    type_weight = {TYPE_DIVERGENT: 3.0, TYPE_TRANSFORM: 1.5, TYPE_QUIET: 1.0, TYPE_CONVERGENT: 0.5}
    n_plates = int(skeleton.plate_b.max()) + 1
    code = skeleton.plate_a.astype(np.int64) * n_plates + skeleton.plate_b
    best = None
    for c in np.unique(code[inside]):
        if (block, int(c)) in tried:
            continue
        sel = inside & (code == c)
        length = sel.sum() * skeleton.cell_km
        if length < 0.45 * math.sqrt(block_area):
            continue
        kinds = skeleton.kind[sel]
        weight = float(np.mean([type_weight.get(int(k), 1.0) for k in kinds]))
        score = length * weight
        if best is None or score > best[0]:
            best = (score, int(c))
    if best is None:
        return None
    return best[1], np.nonzero(code == best[1])[0]


def _straight_cut(labels, block, skeleton, stream, noise):
    rows, cols = np.nonzero(labels == block)
    px = skeleton.x[rows, cols]
    py = skeleton.y[rows, cols]
    cx, cy = px.mean(), py.mean()
    cov = np.cov(np.vstack([px - cx, py - cy]))
    _vals, vecs = np.linalg.eigh(cov)
    ux, uy = vecs[:, 1]  # long axis
    vx, vy = -uy, ux
    along = (px - cx) * ux + (py - cy) * uy
    shift = stream.uniform(-0.15, 0.15) * (along.max() - along.min())
    across = (px - cx) * vx + (py - cy) * vy
    t = np.arange(across.min() - 300.0, across.max() + 300.0, 0.5 * skeleton.cell_km)
    base_x = cx + shift * ux + t * vx
    base_y = cy + shift * uy + t * vy
    wiggle = 0.5 * noise(base_x, base_y)
    return np.column_stack([base_x + wiggle * ux, base_y + wiggle * uy])


def build_continents(
    config: Config, crust_model: CrustModel, skeleton: WorldSkeleton
) -> tuple[ContinentModel, np.ndarray]:
    """Return the model and its continentality on the skeleton raster (km)."""
    params = config.continents
    stream = Stream(config.seed, "continents/v2")
    irr = params.coast_irregularity
    variety = params.continent_size_variety
    warp = (
        DomainWarp(stream.child("warp"), wavelength_km=1800.0, octaves=4, strength=0.5 * irr)
        if irr > 0
        else None
    )
    wavelengths = [2400.0, 1200.0, 600.0, 300.0, 150.0]
    gains = [0.8 + 0.6 * variety, 0.9 + 0.3 * variety, 1.0, 1.0, 0.9]
    noise = FractalNoise(
        stream.child("coast"),
        wavelengths_km=wavelengths,
        amplitudes=[irr * 0.19 * w * g for w, g in zip(wavelengths, gains, strict=True)],
    )

    # tectonic gains per boundary sample, from the prior crust on both sides
    sx, sy, nx, ny = skeleton.bx, skeleton.by, skeleton.nx, skeleton.ny
    model = ContinentModel(
        crust_model=crust_model,
        skeleton=skeleton,
        params=params,
        warp=warp,
        noise=noise,
        offset_grid=np.zeros(skeleton.x.shape),
        weld_gain=np.zeros(sx.size),
        rift_gain=np.zeros(sx.size),
    )
    if sx.size:
        side_a = model._prior(sx - _SIDE_KM * nx, sy - _SIDE_KM * ny)
        side_b = model._prior(sx + _SIDE_KM * nx, sy + _SIDE_KM * ny)
        near_continental = np.minimum(side_a, side_b) > -450.0
        model.weld_gain = np.where(
            (skeleton.kind == TYPE_CONVERGENT) & near_continental, 380.0 * skeleton.intensity, 0.0
        )
        rift_scale = 900.0 if params.continent_profile == "fragmented" else 600.0
        model.rift_gain = np.where(
            skeleton.kind == TYPE_DIVERGENT, rift_scale * skeleton.intensity, 0.0
        )

    x, y = skeleton.x, skeleton.y
    sim_x, sim_y = x[skeleton.in_sim], y[skeleton.in_sim]
    centre = (float(sim_x.mean()), float(sim_y.mean()))
    sim_size = math.sqrt(float(skeleton.in_sim.sum())) * skeleton.cell_km
    if params.continent_profile == "supercontinent":
        # Contract the pattern around a centre: the masses there come close together and
        # the area rebalance welds them into one dominant continent with satellites.
        gather = stream.child("gather")
        cx = centre[0] + gather.uniform(-0.15, 0.15) * sim_size
        cy = centre[1] + gather.uniform(-0.15, 0.15) * sim_size
        model.remap = (cx, cy, 0.45, 0.65 * sim_size)
        model.operations.append(
            {
                "op": "gather",
                "centre_km": [round(cx, 1), round(cy, 1)],
                "gain": 0.45,
                "radius_km": round(0.65 * sim_size, 1),
            }
        )
    elif params.continent_profile == "fragmented":
        # Shrink the whole pattern: more and smaller masses for the same crust share.
        model.remap = (centre[0], centre[1], 0.3 + 0.2 * variety, 0.0)
        model.operations.append({"op": "rescale", "factor": round(1.3 + 0.2 * variety, 3)})
    prior_r = model._prior(x, y)
    noise_r = noise(x, y)
    tect_r = model._tectonic(x, y)

    # -- per-block size variety (smooth, low-frequency offset field) -----------
    labels, count = _label(prior_r + tect_r, 0.0)
    if count and variety > 0:
        cell_area = skeleton.cell_km**2
        area = np.bincount(labels.ravel(), minlength=count + 1) * cell_area
        z = np.clip(stream.child("sizes").normal(count + 1), -1.6, 1.6)
        radius = np.sqrt(area / math.pi)
        # growth / shrinkage of a block edge, km; bounded so huge blocks are not erased
        block_offset = np.where(area >= 3.0e5, variety * z * np.minimum(0.4 * radius, 380.0), 0.0)
        block_offset[0] = 0.0
        dist, inds = ndimage.distance_transform_edt(labels == 0, return_indices=True)
        nearest = labels[inds[0], inds[1]]
        raw = block_offset[nearest] * np.exp(-((dist * skeleton.cell_km / 500.0) ** 2))
        model.offset_grid = ndimage.gaussian_filter(raw, sigma=200.0 / skeleton.cell_km)
    if model.remap is not None and model.remap[3] > 0:
        # favour the gathered core: the uniform rebalance takes the area from the rim
        gx, gy, _gain, g_radius = model.remap
        swell = 750.0 * np.exp(-((x - gx) ** 2 + (y - gy) ** 2) / (0.55 * g_radius) ** 2)
        model.offset_grid = model.offset_grid + swell
    offset_r = model.offset_grid

    def raster():
        return model._compose(prior_r, offset_r, tect_r, noise_r, x, y)

    def rebalance():
        """Keep the prior's share of continental crust over the simulation area."""
        share = float((crust_model.continentality(x, y)[skeleton.in_sim] > 0).mean())
        for _ in range(3):
            current = raster()[skeleton.in_sim]
            model.balance_km -= float(np.quantile(current, 1.0 - share))

    rebalance()
    # -- structure: merge or split blocks towards the requested configuration --
    field_r = raster()
    labels, count = _label(field_r)
    majors, area = _majors(labels, count, skeleton, params.major_area_km2)
    natural = int(majors.size)
    requested = params.major_landmass_count
    if requested != "auto":
        target = int(requested)
    elif params.continent_profile == "supercontinent":
        target = max(1, int(round(0.35 * natural)))
    elif params.continent_profile == "fragmented":
        target = natural  # the rescale and the plate rifts already fragment the crust
    else:
        # balanced: weld part of the blocks into composite continents of unequal size
        target = natural if natural <= 2 else max(2, int(round(0.65 * natural)))
    cut_stream = stream.child("structure")
    tried: set = set()
    fallback_used: set = set()
    merging = natural > target  # the direction is fixed once: no merge/split oscillation
    for _ in range(16):
        majors, area = _majors(labels, count, skeleton, params.major_area_km2)
        if (merging and majors.size <= target) or (not merging and majors.size >= target):
            break
        if merging:
            pair = _closest_pair(labels, majors, skeleton)
            wide = params.continent_profile == "supercontinent"
            reach_limit = _AUTO_WELD_KM * (1.6 if wide else 1.0)
            if pair is not None and requested == "auto" and pair[0] > reach_limit:
                break  # profiles never force a weld across a wide ocean
            if pair is None or pair[0] > _MAX_BRIDGE_KM:
                model.warnings.append(
                    f"Cannot merge down to {target} major block(s): the remaining gap of "
                    f"{pair[0]:,.0f} km is wider than any plausible land bridge."
                    if pair
                    else f"Cannot merge down to {target} major block(s)."
                )
                break
            gap, p1, p2, _i, _j = pair
            dx, dy = p2[0] - p1[0], p2[1] - p1[1]
            length = max(math.hypot(dx, dy), 1e-6)
            # a weld as broad as the smaller block: a fused continent, not a land bridge
            smaller = math.sqrt(float(min(area[_i], area[_j])) / math.pi)
            half_width = float(
                np.clip(max(0.45 * smaller, 0.3 * gap), 350.0, 1300.0)
            ) * cut_stream.uniform(0.8, 1.2)
            reach = 0.6 * half_width + 200.0
            ex, ey = dx / length * reach, dy / length * reach
            seg = (p1[0] - ex, p1[1] - ey, p2[0] + ex, p2[1] + ey, half_width)
            if any(np.allclose(seg[:4], old[:4]) for old in model.bridges):
                model.warnings.append(
                    f"Cannot merge down to {target} major block(s): a rift or boundary keeps "
                    "two blocks apart."
                )
                break
            model.bridges.append(seg)
            model.operations.append(
                {"op": "bridge", "gap_km": round(gap, 1), "half_width_km": round(half_width, 1)}
            )
        else:
            order = majors[np.argsort(-area[majors])]
            done = False
            for block in order:
                block = int(block)
                block_radius = math.sqrt(float(area[block]) / math.pi)
                width = float(np.clip(0.14 * block_radius, 150.0, 480.0)) * cut_stream.uniform(
                    0.8, 1.25
                )
                found = _rift_along_boundary(labels, block, float(area[block]), skeleton, tried)
                if found is not None:
                    code, sel = found
                    tried.add((block, code))
                    near = _point_values(field_r, skeleton)[sel] > -300.0
                    pts = np.column_stack([skeleton.bx[sel][near], skeleton.by[sel][near]])
                    how = "plate-boundary"
                elif block not in fallback_used:
                    fallback_used.add(block)
                    pts = _straight_cut(labels, block, skeleton, cut_stream, noise)
                    how = "structural"
                else:
                    continue
                model.rift_points = np.vstack([model.rift_points, pts])
                model.rift_width = np.concatenate([model.rift_width, np.full(len(pts), width)])
                model._rift_tree = None
                model.operations.append(
                    {
                        "op": "rift",
                        "follows": how,
                        "half_width_km": round(width, 1),
                        "samples": int(len(pts)),
                    }
                )
                done = True
                break
            if not done:
                model.warnings.append(
                    f"Cannot split into {target} major blocks: only {majors.size} could be "
                    "separated without breaking the tectonic structure."
                )
                break
        field_r = raster()
        labels, count = _label(field_r)
    rebalance()
    field_r = raster()
    labels, count = _label(field_r)
    majors, _area = _majors(labels, count, skeleton, params.major_area_km2)
    model.operations.append(
        {
            "op": "summary",
            "natural_major_blocks": natural,
            "target": target,
            "major_blocks": int(majors.size),
        }
    )
    return model, field_r
