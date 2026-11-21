"""Builds the structural mountain plan: varied systems of ranges, not one line per belt.

What real orogens look like, and what this module reproduces:

* **A belt is several mountain systems.** The Alpine-Himalayan belt is the Pyrenees, Alps,
  Carpathians, Caucasus, Zagros, Hindu Kush, Himalaya... separated by syntaxes and low
  necks. A connected area of orogenic relief longer than ~1000-2200 km is cut into
  systems at the lowest necks of its axis; each system gets its own architecture.
* **A system is several ranges.** The Himalaya are the Siwaliks, the Lesser and the Great
  Himalaya with the Tibetan plateau behind; the Caucasus are the Greater and the Lesser
  Caucasus with the Kura depression between; the Andes are two cordilleras around the
  Altiplano. Ranges run roughly parallel to the belt, follow its width, overlap
  en echelon, and leave intermontane basins between them.
* **Architectures (styles)**, chosen from the tectonic setting, the width and height of
  the belt, its age and the seed:

  ======================  =================================================  ==============
  style                   ranges                                             Earth examples
  ======================  =================================================  ==============
  ``volcanic_arc``        one narrow chain (the cones come from volcanism)   Kurils, Cascades
  ``alpine``              a sharp main crest, partial secondary ranges,      Alps, Caucasus
                          foothills; steep and narrow
  ``himalayan``           foothills, lesser range, great range, a high       Himalaya-Tibet
                          plateau with low ranges, a back range
  ``andean``              a main cordillera, an altiplano, an eastern        Andes
                          cordillera and sub-Andean foothills
  ``fold_belt``           many narrow parallel ridges with long valleys      Zagros, Jura,
                                                                             Appalachians
  ``old_massif``          broad rounded ridges, low and gentle               Urals, Scandes
  ``broad_upland``        a wide high tableland with gentle edges            Ethiopian,
                                                                             Colorado plateau
  ``mixed``               a broad gentle base with a sharp core              Altai, Rockies
  ``block``               tilted fault blocks, steep on one side             Basin and Range,
                                                                             rift shoulders
  ======================  =================================================  ==============

* **Irregular branching.** Spurs leave at uneven intervals (gamma-distributed), from the
  side with more room, at varied angles and lengths, and branch again; peaks are unevenly
  spaced with massifs of higher summits and occasional deep passes.
* **Time.** Maturity (age + share eroded) widens the ranges, lowers and rounds the
  crests, evens the summit heights, removes small spurs and widens the valleys: a young
  alpine range becomes a broad old massif.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy import ndimage
from scipy.sparse.csgraph import dijkstra
from scipy.spatial import cKDTree

from ..config import MOUNTAIN_STYLES as STYLES
from ..rng import Stream
from .plan import (
    KIND_NAMES,
    RANK_BRANCH,
    RANK_MAIN,
    RANK_SPUR,
    MountainPlan,
    MountainSystem,
    Ridge,
    Valley,
    _cosine_profile,
    _path,
    _system_graph,
    arclength,
    chaikin,
    resample,
)

__all__ = ["STYLES", "RangeSpec", "build_plan", "choose_style", "style_ranges"]

#: Spur density (per main gap) of each style; plateau-like styles have few spurs.
_SPUR_DENSITY = {
    "volcanic_arc": 0.6,
    "alpine": 1.25,
    "himalayan": 1.1,
    "andean": 1.0,
    "fold_belt": 0.35,
    "old_massif": 0.7,
    "broad_upland": 0.45,
    "mixed": 1.0,
    "block": 0.8,
    "highland": 0.7,
    "composite": 0.6,
}


@dataclass
class RangeSpec:
    """One range of a system, placed at relative offset ``f`` across the belt (-1..1)."""

    f: float
    crest: float
    hw: float  # multiple of params.range_half_width_km, or km when absolute
    sharp: float
    role: str
    plateau: float = 0.0
    asym: float = 0.0
    trim: tuple = (0.0, 0.0)  # share of the axis left out at the start / end
    absolute_hw: bool = False
    peaks: bool = True
    spurs: bool = True
    echelon: bool = True


# -- style choice -----------------------------------------------------------------------
def choose_style(
    kind: str,
    width_km: float,
    relief_m: float,
    maturity: float,
    u: float,
    branched: bool = False,
) -> str:
    """Architecture of a system from its tectonic setting, size and age (``u``: seed draw).

    ``branched``: the relief spreads in several directions from the belt's axis (a knot
    where belts meet), not along one line.
    """

    def pick(options):
        total = sum(w for _s, w in options)
        acc = 0.0
        for style, w in options:
            acc += w / total
            if u <= acc:
                return style
        return options[-1][0]

    if kind == "rift":
        return "block"
    if branched and kind != "arc" and maturity < 0.6 and 400.0 < width_km < 1800.0:
        if u < 0.45:
            return "composite"
        u = (u - 0.45) / 0.55  # the same draw, rescaled, for the other styles
    wide = width_km > 650.0 and relief_m < 2600.0
    if maturity > 0.66 or kind == "inherited":
        return pick(
            [
                ("old_massif", 0.45),
                ("broad_upland", 0.2),
                ("fold_belt", 0.15),
                ("mixed", 0.1),
                ("highland", 0.45 if wide else 0.0),
            ]
        )
    if kind == "arc":
        if width_km < 260.0:
            return "volcanic_arc" if maturity < 0.5 else "old_massif"
        if width_km > 420.0 and relief_m > 1600.0:
            return pick([("andean", 0.8), ("mixed", 0.2)])
        return pick([("andean", 0.45), ("alpine", 0.35), ("volcanic_arc", 0.2)])
    if maturity < 0.35:
        if width_km > 900.0:  # very wide young belts: Tibet, or a Tien Shan-Mongolian
            return pick(  # country of ranges
                [("himalayan", 0.5 if relief_m > 1800.0 else 0.1), ("highland", 0.35),
                 ("mixed", 0.15)]
            )  # fmt: skip
        if width_km > 500.0 and relief_m > 1800.0:
            return pick([("himalayan", 0.7), ("mixed", 0.2), ("andean", 0.1)])
        if width_km > 280.0:
            return pick([("alpine", 0.45), ("fold_belt", 0.3), ("mixed", 0.25)])
        return pick([("alpine", 0.65), ("fold_belt", 0.25), ("block", 0.1)])
    return pick(
        [
            ("mixed", 0.35),
            ("fold_belt", 0.25),
            ("broad_upland", 0.2),
            ("alpine", 0.2),
            ("highland", 0.4 if wide else 0.0),
        ]
    )


def style_ranges(style: str, stream: Stream, width_km: float, complexity: float) -> list:
    """Ranges of a system. ``f`` > 0 is the narrower ("front") side of the belt."""
    u = stream.random(24)
    extra = complexity >= 0.5  # 0 = the old single-crest behaviour
    out: list[RangeSpec] = []
    if style == "volcanic_arc":
        out.append(RangeSpec(0.0, 0.8, 0.7, 1.2, "main"))
    elif style == "alpine":
        out.append(
            RangeSpec(
                (u[0] - 0.5) * 0.3,
                1.0,
                1.0,
                1.4,
                "main",
                asym=-0.25,
                trim=(0.02 * u[1], 0.02 * u[2]),
            )
        )
        if extra:
            for k in range(1 + int(u[3] < 0.45 * complexity)):
                side = 1.0 if (k + int(u[4] * 2)) % 2 else -1.0
                out.append(
                    RangeSpec(
                        side * (0.45 + 0.25 * u[5 + k]),
                        0.55 + 0.25 * u[7 + k],
                        0.8,
                        1.25,
                        "secondary",
                        trim=(0.35 * u[9 + k], 0.35 * u[11 + k]),
                    )
                )
            if u[13] < 0.65:
                out.append(
                    RangeSpec(
                        0.86, 0.3, 0.7, 0.9, "frontal", peaks=False, trim=(0.1 * u[14], 0.1 * u[15])
                    )
                )
    elif style == "himalayan":
        out.append(RangeSpec(0.45, 1.0, 0.95, 1.5, "main", asym=-0.35))
        if extra:
            out += [
                RangeSpec(0.9, 0.3, 0.6, 1.0, "frontal", peaks=False, spurs=False),
                RangeSpec(0.69, 0.55, 0.7, 1.1, "secondary"),
                RangeSpec(
                    -0.12,
                    0.8,
                    0.4 * width_km,
                    0.8,
                    "plateau",
                    plateau=0.68,
                    absolute_hw=True,
                    peaks=False,
                    spurs=False,
                ),
                RangeSpec(-0.8, 0.72, 0.8, 1.2, "back", trim=(0.15 * u[0], 0.15 * u[1])),
            ]
            for k in range(int(1 + 2 * u[2] * complexity)):  # low ranges on the plateau
                out.append(
                    RangeSpec(
                        -0.25 - 0.35 * u[3 + k],
                        0.86,
                        0.5,
                        1.0,
                        "secondary",
                        trim=(0.2 + 0.4 * u[6 + k], 0.2 + 0.4 * u[9 + k]),
                    )
                )
    elif style == "andean":
        out.append(RangeSpec(0.35, 1.0, 0.85, 1.4, "main", asym=-0.3))
        if extra:
            if width_km > 420.0:
                out.append(
                    RangeSpec(
                        -0.1,
                        0.8,
                        0.3 * width_km,
                        0.85,
                        "plateau",
                        plateau=0.6,
                        absolute_hw=True,
                        peaks=False,
                        spurs=False,
                        trim=(0.1 + 0.2 * u[0], 0.1 + 0.2 * u[1]),
                    )
                )
            out.append(
                RangeSpec(-0.5, 0.88, 0.8, 1.3, "secondary", trim=(0.05 * u[2], 0.25 * u[3]))
            )
            out.append(RangeSpec(-0.85, 0.35, 0.6, 0.9, "frontal", peaks=False))
    elif style == "fold_belt":
        n = 3 + int(round((2 + 2 * u[0]) * max(complexity, 0.25))) if extra else 1
        centre = int(u[1] * n)
        for k in range(n):
            f = -0.75 + 1.5 * (k + 0.5) / n + (u[2 + k % 10] - 0.5) * 0.2 / n
            crest = 1.0 if k == centre else 0.45 + 0.35 * u[12 + k % 10]
            out.append(
                RangeSpec(
                    f if n > 1 else 0.0,
                    crest,
                    0.45,
                    1.15,
                    "fold",
                    trim=(0.25 * u[(k + 3) % 24], 0.25 * u[(k + 7) % 24]),
                )
            )
    elif style == "old_massif":
        out.append(RangeSpec(0.0, 0.8, 1.9, 0.7, "main"))
        if extra:
            for k in range(1 + int(u[0] < 0.5)):
                side = -1.0 if k else 1.0
                out.append(
                    RangeSpec(
                        side * (0.5 + 0.2 * u[1 + k]),
                        0.5,
                        1.3,
                        0.7,
                        "secondary",
                        trim=(0.3 * u[3 + k], 0.3 * u[5 + k]),
                    )
                )
    elif style == "broad_upland":
        out.append(
            RangeSpec(
                0.0,
                0.72,
                0.42 * width_km,
                0.6,
                "plateau",
                plateau=0.55,
                absolute_hw=True,
                peaks=False,
                spurs=False,
            )
        )
        out.append(
            RangeSpec((u[0] - 0.5) * 0.6, 0.85, 0.9, 0.9, "main", trim=(0.2 * u[1], 0.2 * u[2]))
        )
    elif style == "mixed":
        out.append(RangeSpec(0.0, 0.58, 2.2, 0.6, "base", peaks=False, spurs=False))
        out.append(
            RangeSpec(
                (u[0] - 0.5) * 0.25,
                1.0,
                0.7,
                1.55,
                "core",
                trim=(0.08 + 0.25 * u[1], 0.08 + 0.25 * u[2]),
            )
        )
        if extra and u[3] < 0.7:
            side = 1.0 if u[4] < 0.5 else -1.0
            out.append(
                RangeSpec(side * 0.55, 0.65, 0.8, 1.2, "secondary", trim=(0.3 * u[5], 0.3 * u[6]))
            )
    elif style == "highland":
        # the raised country itself; its ranges are placed by _highland_ranges
        out.append(
            RangeSpec(
                0.0,
                0.3,
                0.4 * width_km,
                0.6,
                "base",
                plateau=0.45,
                absolute_hw=True,
                peaks=False,
                spurs=False,
            )
        )
    elif style == "composite":
        pass  # bundles of ranges leaving the knot: _composite_ranges
    else:  # block: tilted fault blocks, en echelon, steep on alternating sides
        n = 1 + int(u[0] * (3 if extra else 1))
        for k in range(n):
            lo = k / n + 0.06 * u[1 + k]
            hi = 1.0 - (k + 1) / n + 0.06 * u[4 + k]
            out.append(
                RangeSpec(
                    (u[7 + k] - 0.5) * 0.6,
                    0.75 + 0.25 * u[10 + k],
                    0.9,
                    1.25,
                    "block",
                    asym=0.6 if k % 2 else -0.6,
                    trim=(max(lo - 0.05, 0.0), max(hi - 0.05, 0.0)),
                )
            )
    return out


def morphology_of(style: str, maturity: float) -> str:
    if style == "volcanic_arc":
        return "volcanic_arc"
    if style in ("himalayan", "broad_upland") or (style == "andean" and maturity < 0.5):
        return "plateau"
    if maturity < 0.33:
        return "young_sharp"
    return "mature" if maturity < 0.66 else "old_rounded"


# -- geometry ------------------------------------------------------------------------
def _frame(points: np.ndarray):
    """Arc length, unit tangents and left normals of a polyline (smoothed)."""
    s = arclength(points)
    d = np.gradient(points, axis=0)
    d = ndimage.gaussian_filter1d(d, 3.0, axis=0, mode="nearest")
    t = d / np.maximum(np.hypot(d[:, 0], d[:, 1]), 1e-9)[:, None]
    n = np.column_stack([-t[:, 1], t[:, 0]])
    return s, t, n


def _side_widths(axis, normals, s, cx, cy, step):
    """Half-width of the belt on each side of the axis along it (km)."""
    tree = cKDTree(axis)
    _d, idx = tree.query(np.column_stack([cx, cy]))
    off = (cx - axis[idx, 0]) * normals[idx, 0] + (cy - axis[idx, 1]) * normals[idx, 1]
    left = np.full(len(axis), 0.0)
    right = np.full(len(axis), 0.0)
    np.maximum.at(left, idx, np.where(off > 0, off, 0.0))
    np.maximum.at(right, idx, np.where(off < 0, -off, 0.0))
    sigma = max(2.0, 160.0 / step)
    left = ndimage.maximum_filter1d(left, int(sigma), mode="nearest")
    right = ndimage.maximum_filter1d(right, int(sigma), mode="nearest")
    left = ndimage.gaussian_filter1d(left, sigma, mode="nearest") + 0.5 * step
    right = ndimage.gaussian_filter1d(right, sigma, mode="nearest") + 0.5 * step
    return left, right


def _runs(ok: np.ndarray):
    """Contiguous True runs as (start, stop) index pairs."""
    edges = np.diff(np.concatenate([[0], ok.astype(np.int8), [0]]))
    return list(zip(np.nonzero(edges == 1)[0], np.nonzero(edges == -1)[0], strict=True))


def _wobble(s, amp, stream: Stream):
    """Smooth lateral wander (km) so parallel ranges are never parallel lines."""
    u = stream.random(6)
    total = max(float(s[-1]), 1.0)
    w1 = 2 * math.pi * (0.6 + 1.2 * u[0]) / total
    w2 = 2 * math.pi * (2.0 + 3.0 * u[1]) / total
    return amp * (0.7 * np.sin(w1 * s + 6.28 * u[2]) + 0.3 * np.sin(w2 * s + 6.28 * u[3]))


# -- the builder ---------------------------------------------------------------------
def build_plan(
    grid,
    relief,
    kind,
    age_ma,
    erosion_share,
    elevation,
    params,
    seed: str,
    time_ma: float,
    focus=None,
) -> MountainPlan:
    """Build the plan for one snapshot from mantle-frame rasters on the history grid.

    ``relief`` - orogenic relief left by the history, m; ``kind`` - KIND_* code;
    ``age_ma`` - time since the last mountain-building; ``erosion_share`` - share
    of the built relief already eroded (0..1); ``elevation`` - macro elevation, m.
    """
    plan = MountainPlan(time_ma=float(time_ma))
    cell = grid.cell
    mask = (relief > params.system_threshold_m) & (elevation > -1500.0)
    labels, count = ndimage.label(mask, structure=np.ones((3, 3)))
    plan.system_raster = np.full(grid.shape, -1, dtype=np.int32)
    if count == 0:
        return plan
    gx, gy = grid.centers()
    sizes = np.bincount(labels.ravel(), minlength=count + 1)
    cx = ndimage.mean(gx, labels, np.arange(1, count + 1))
    cy = ndimage.mean(gy, labels, np.arange(1, count + 1))
    keep = [k for k in range(1, count + 1) if sizes[k] >= 4]
    if focus is not None:  # only systems that reach the area of interest get a structure
        present = np.bincount(labels[focus], minlength=count + 1)
        keep = [k for k in keep if present[k] > 0]
    keep.sort(key=lambda k: (-int(sizes[k]), float(cx[k - 1]), float(cy[k - 1])))
    inside_km = ndimage.distance_transform_edt(mask) * cell
    step = max(8.0, min(14.0, 0.25 * cell))
    variety = float(getattr(params, "complexity", 1.0))

    parts = []
    for label in keep:
        rows, cols = np.nonzero(labels == label)
        rel = relief[rows, cols].astype(np.float64)
        belt = Stream(seed, f"mountains/v3/belt/{rows.size}/{int(rows[0])}/{int(cols[0])}")
        cost_graph, hop_graph = _system_graph(rows, cols, rel, cell)
        px = gx[rows, cols]
        py = gy[rows, cols]
        # the axis of the belt: far end -> far end, following the crest
        start = int(np.argmax(rel))
        d0 = dijkstra(hop_graph, indices=start)
        far_a = int(np.argmax(np.where(np.isfinite(d0), d0, -1.0)))
        d_a = dijkstra(hop_graph, indices=far_a)
        far_b = int(np.argmax(np.where(np.isfinite(d_a), d_a, -1.0)))
        _dist, pred = dijkstra(cost_graph, indices=far_a, return_predecessors=True)
        main = _path(pred, far_b, {far_a})
        axis_raw = np.column_stack([px[main], py[main]])
        axis = resample(chaikin(axis_raw, 3), step) if len(axis_raw) >= 2 else axis_raw
        # cut a long belt into systems at the lowest necks of its axis
        cuts = _cut_points(axis, grid, relief, belt, variety)
        s_axis = arclength(axis)
        tree = cKDTree(axis)
        _d, near = tree.query(np.column_stack([px, py]))
        cell_s = s_axis[near]
        bounds = [0.0, *[s_axis[c] for c in cuts], s_axis[-1] + 1.0]
        for part in range(len(bounds) - 1):
            sel = (cell_s >= bounds[part]) & (cell_s < bounds[part + 1])
            if sel.sum() < 3:
                continue
            a_sel = (s_axis >= bounds[part] - 1e-6) & (s_axis <= bounds[part + 1])
            sub_axis = axis[a_sel]
            if len(sub_axis) < 2:
                qx, qy = px[sel], py[sel]
                i = int(np.argmax(np.hypot(qx - qx.mean(), qy - qy.mean())))
                j = int(np.argmax(np.hypot(qx - qx[i], qy - qy[i])))
                sub_axis = resample(np.array([[qx[i], qy[i]], [qx[j], qy[j]]]), step)
            parts.append((rows[sel], cols[sel], sub_axis))
    # a high plateau is rare on Earth (Tibet, Altiplano): only the widest, highest young
    # belts of the world may get one
    scores = []
    for k, (r, c, _a) in enumerate(parts):
        width_est = 2.0 * float(inside_km[r, c].max())
        rel_mean = float(relief[r, c].mean())
        worn = float(np.nanmean(erosion_share[r, c]))
        if width_est > 500.0 and rel_mean > 1800.0 and worn < 0.45:
            scores.append((width_est * rel_mean, k))
    scores.sort(reverse=True)
    quota = max(1, min(3, round(0.25 * len(scores)))) if scores else 0
    plateau_ok = {k for _score, k in scores[:quota]}
    for k, (r, c, sub_axis) in enumerate(parts):
        _build_system(
            plan,
            grid,
            relief,
            kind,
            age_ma,
            erosion_share,
            inside_km,
            params,
            seed,
            r,
            c,
            sub_axis,
            step,
            variety,
            k in plateau_ok,
        )
    return plan


def _cut_points(axis, grid, relief, stream: Stream, variety: float) -> list[int]:
    """Indices of the axis where a long belt is split into separate systems."""
    s = arclength(axis)
    total = float(s[-1]) if len(s) else 0.0
    if variety <= 0.0 or total < 1800.0:
        return []
    u = stream.random(64)
    prof = ndimage.gaussian_filter1d(grid.bilinear(relief, axis[:, 0], axis[:, 1]), 3.0)
    cuts: list[int] = []
    pos = 0.0
    k = 0
    while True:
        seg = 1000.0 + 1300.0 * u[k % 64]
        k += 1
        target = pos + seg
        if total - target < 700.0:
            break
        lo, hi = np.searchsorted(s, target - 0.25 * seg), np.searchsorted(s, target + 0.25 * seg)
        lo, hi = max(lo, 1), min(hi, len(s) - 2)
        if hi <= lo:
            break
        i = lo + int(np.argmin(prof[lo:hi]))  # the lowest neck near the target
        cuts.append(i)
        pos = float(s[i])
    return cuts


def _build_system(
    plan,
    grid,
    relief,
    kind,
    age_ma,
    erosion_share,
    inside_km,
    params,
    seed,
    rows,
    cols,
    axis,
    step,
    variety,
    plateau_ok=True,
):
    cell = grid.cell
    sid = len(plan.systems)
    gx, gy = grid.centers()
    px, py = gx[rows, cols], gy[rows, cols]
    rel = relief[rows, cols].astype(np.float64)
    plan.system_raster[rows, cols] = sid
    kinds = np.bincount(kind[rows, cols], minlength=5)
    kinds[0] = 0
    kind_name = KIND_NAMES[int(np.argmax(kinds))] if kinds.any() else "inherited"
    known = age_ma[rows, cols]
    known = known[np.isfinite(known)]
    age = float(np.median(known)) if known.size else 0.0
    worn = float(np.nanmean(erosion_share[rows, cols]))
    maturity = float(np.clip(0.6 * worn + 0.4 * min(age / 150.0, 1.0), 0.0, 1.0))
    stream = Stream(seed, f"mountains/v3/system/{sid}/{rows.size}")
    s, tangents, normals = _frame(axis)
    total = float(s[-1]) if len(s) > 1 else 0.0
    left, right = _side_widths(axis, normals, s, px, py, step)
    width = (
        float(np.median(left + right)) if total > 0 else 2.0 * float(inside_km[rows, cols].max())
    )
    style = getattr(params, "style", "auto")
    branched = False
    if total > 0 and rows.size >= 12:
        # share of the system far from its axis: relief spreading in several directions
        d_axis, _ = cKDTree(axis).query(np.column_stack([px, py]))
        branched = float((d_axis > 0.75 * max(width, 2.0 * cell)).mean()) > 0.12
    if style == "auto":
        style = choose_style(
            kind_name, width, float(rel.mean()), maturity, stream.random(), branched
        )
        if style in ("himalayan", "broad_upland") and not plateau_ok:
            u = stream.random()
            if maturity > 0.66:  # an old belt stays old: a worn massif or a highland
                style = "old_massif" if u < 0.6 or width < 650.0 else "highland"
            else:
                style = "mixed" if u < 0.35 else ("alpine" if u < 0.7 else "fold_belt")
    if total < 2.5 * cell:  # a short knot of relief: one massif, whatever the setting
        style = "volcanic_arc" if style == "volcanic_arc" else "alpine"
    if style in ("highland", "composite") and rows.size < 20:
        style = "mixed"  # too small for a country of ranges
    system = MountainSystem(
        id=sid,
        kind=kind_name,
        morphology=morphology_of(style, maturity),
        age_ma=round(age, 3),
        maturity=round(maturity, 4),
        sharpness=round(1.55 - 0.85 * maturity, 4),
        relief_gain=round(
            (0.95 - 0.5 * maturity)
            * (0.8 if style in ("himalayan", "broad_upland") else 1.0)
            * (1.25 if style == "highland" else 1.0),
            4,
        ),
        area_km2=float(rows.size * cell * cell),
        centroid_km=(float(px.mean()), float(py.mean())),
        bounds_km=(float(px.min()), float(py.min()), float(px.max()), float(py.max())),
        max_relief_m=float(rel.max()),
        touches_raster_edge=bool(
            rows.min() == 0
            or cols.min() == 0
            or rows.max() == grid.ny - 1
            or cols.max() == grid.nx - 1
        ),
        style=style,
        width_km=round(width, 1),
    )
    plan.systems.append(system)
    first_ridge = len(plan.ridges)
    if total <= 0.0:
        return
    # the narrower side of the belt is its "front" (thrust front / trench side)
    front = 1.0 if float(np.mean(left)) < float(np.mean(right)) else -1.0
    inside = np.zeros(grid.shape, dtype=bool)
    inside[rows, cols] = True
    inside = ndimage.binary_dilation(inside, iterations=1)

    def inside_at(x, y):
        r, c, ok = grid.nearest(x, y)
        return ok & inside[r, c]

    m = maturity
    specs = style_ranges(style, stream.child("ranges"), width, variety)
    if variety < 0.5:  # one unbroken crest per range
        for sp in specs:
            sp.echelon = False
    if not plateau_ok:  # an Andean system without an altiplano
        specs = [sp for sp in specs if sp.role != "plateau"]
    specs = _lobes(specs, stream.child("lobes"))
    specs += _filler_ranges(specs, style, width, maturity, variety, stream.child("filler"))
    built = []  # (spec, ridge ids)
    for k, spec in enumerate(specs):
        side = spec.f * front
        lateral = np.where(side > 0, left, right) * abs(side) * 0.82
        wob = _wobble(s, 0.045 * width, stream.child(f"wobble/{k}"))
        offset = np.sign(side) * lateral + (wob if spec.role not in ("base", "plateau") else 0.0)
        # offset a smoothed axis: a sharp bend offset by hundreds of km would loop
        sigma = max(2.0, 0.6 * float(np.abs(offset).max()) / step)
        smooth = ndimage.gaussian_filter1d(axis, sigma, axis=0, mode="nearest")
        s_sm, t_sm, n_sm = _frame(smooth)
        offset = ndimage.gaussian_filter1d(offset, max(1.0, 0.5 * sigma), mode="nearest")
        pts = smooth + n_sm * offset[:, None]
        lo_t, hi_t = spec.trim
        keep = (s >= lo_t * total) & (s <= (1.0 - hi_t) * total)
        keep &= inside_at(pts[:, 0], pts[:, 1])
        dp = np.gradient(pts, axis=0)
        dp /= np.maximum(np.hypot(dp[:, 0], dp[:, 1]), 1e-9)[:, None]
        keep &= (dp * t_sm).sum(axis=1) > 0.35  # drop folded-back parts of an offset line
        runs = [(a, b) for a, b in _runs(keep) if s[b - 1] - s[a] >= max(1.6 * cell, 110.0)]
        ids = []
        for a, b in runs:
            for piece in _en_echelon(pts[a:b], spec, stream.child(f"echelon/{k}/{a}"), m):
                rid = _add_range(plan, sid, piece, spec, params, m, front, step)
                ids.append(rid)
        if ids:
            built.append((spec, ids))
    taper: dict = {}
    if style == "highland":
        built += _highland_ranges(
            plan, sid, px, py, axis, width, params, m, front, step, variety, inside_at,
            stream.child("highland"),
        )  # fmt: skip
    elif style == "composite":
        made, taper = _composite_ranges(
            plan, sid, rows, cols, px, py, rel, axis, params, m, front, step, variety,
            inside_at, cell, stream.child("composite"),
        )  # fmt: skip
        built += made
    if not any(spec.role in ("main", "core", "fold", "block", "highland") for spec, _ in built):
        # nothing of the architecture fitted: one crest along the axis
        spec = RangeSpec(0.0, 1.0, 1.0, 1.35, "main")
        built.append((spec, [_add_range(plan, sid, axis, spec, params, m, front, step)]))
    system.ranges = sum(len(ids) for _spec, ids in built)

    # parts of the system far from every range get a branch (bent belts, wide knots)
    cost_graph, hop_graph = _system_graph(rows, cols, rel, cell)
    pts_all = np.vstack([plan.ridges[i].points for i in range(first_ridge, len(plan.ridges))])
    ridge_tree = cKDTree(pts_all)
    reach = max(2.6 * params.range_half_width_km * (1.0 + 0.6 * m), 0.14 * width)
    for _branch in range(2):
        d_cells, _i = ridge_tree.query(np.column_stack([px, py]))
        far = int(np.argmax(d_cells))
        if d_cells[far] < reach:
            break
        near_cell = int(np.argmin(np.hypot(px - pts_all[_i[far], 0], py - pts_all[_i[far], 1])))
        _d, pred = dijkstra(cost_graph, indices=near_cell, return_predecessors=True)
        path = _path(pred, far, {near_cell})[::-1]
        if len(path) < 3:
            break
        raw = np.column_stack([px[path], py[path]])
        jit = (stream.random(2 * len(raw)).reshape(-1, 2) - 0.5) * 0.7 * cell
        jit[0] = jit[-1] = 0.0
        raw = chaikin(raw + jit, 3)  # a path over raster cells must not show its stairs
        parent = _nearest_ridge(plan, first_ridge, raw[0])
        start_pt = plan.ridges[parent].points[
            int(np.argmin(np.hypot(*(plan.ridges[parent].points - raw[0]).T)))
        ]
        raw = np.vstack([start_pt, raw])
        spec = RangeSpec(0.0, 0.85, 0.85, 1.25, "branch")
        rid = _add_range(
            plan, sid, raw, spec, params, m, front, step, rank=RANK_BRANCH, parent=parent
        )
        plan.knots.append(
            {
                "id": len(plan.knots),
                "system": sid,
                "x_km": float(start_pt[0]),
                "y_km": float(start_pt[1]),
                "ridges": [parent, rid],
            }
        )
        pts_all = np.vstack([pts_all, plan.ridges[rid].points])
        ridge_tree = cKDTree(pts_all)
    system.length_km = float(max(arclength(r.points)[-1] for r in plan.ridges[first_ridge:]))

    # crest profiles: unevenly spaced peaks, massifs, deep passes
    rel_max = max(float(rel.max()), 1.0)
    for ridge in plan.ridges[first_ridge:]:
        local = np.clip(
            grid.bilinear(relief, ridge.points[:, 0], ridge.points[:, 1]) / rel_max, 0.15, 1.0
        )
        if ridge.id in taper:  # composite: ranges get lower away from the knot
            kx, ky, reach_km = taper[ridge.id]
            d_knot = np.hypot(ridge.points[:, 0] - kx, ridge.points[:, 1] - ky)
            local = local * (1.0 - 0.38 * np.clip(d_knot / max(reach_km, 1.0), 0.0, 1.0))
        _crest_and_peaks(plan, ridge, local, params, m, stream.child(f"crest/{ridge.id}"))

    # spurs and the valleys between them: lines of steepest descent of the flanks
    density = _SPUR_DENSITY.get(style, 1.0) * max(variety, 0.3)
    field = _flank_field(plan.ridges[first_ridge:], grid, relief, rel_max, seed, sid)
    for ridge in list(plan.ridges[first_ridge:]):
        if ridge.role in ("plateau", "base", "frontal") or ridge.rank == RANK_SPUR:
            continue
        _spurs(plan, ridge, field, inside_at, params, m, density, stream.child(f"spurs/{ridge.id}"))

    # intermontane basins between neighbouring parallel ranges
    if style not in ("highland", "composite"):
        _basins(plan, built, axis, normals, s, left, right, front, params, m, step)


def _orientation_field(axis, stream: Stream, scale_km: float, amplitude: float):
    """Trend of ranges at a point: the belt's local direction turned by a smooth field.

    A few long-wave sinusoids (wavelengths ~1-3 x ``scale_km``) rotate the trend by up to
    ``amplitude`` radians, so neighbouring ranges share a trend while the trends across a
    wide country swing into arcs and fans (the Verkhoyansk-Kolyma arc, the Stanovoy).
    """
    s_ax, t_ax, _n_ax = _frame(axis)
    tree = cKDTree(axis)
    u = stream.random(16)
    waves = []
    for k in range(4):
        ang = 2 * math.pi * u[k]
        lam = scale_km * (1.0 + 2.0 * u[4 + k])
        waves.append((math.cos(ang) / lam, math.sin(ang) / lam, 2 * math.pi * u[8 + k], u[12 + k]))
    norm = sum(w[3] for w in waves) + 1e-9

    def theta(x, y):
        _d, i = tree.query(np.column_stack([np.atleast_1d(x), np.atleast_1d(y)]))
        base = np.arctan2(t_ax[i, 1], t_ax[i, 0])
        turn = sum(
            a * np.sin(2 * math.pi * (kx * np.atleast_1d(x) + ky * np.atleast_1d(y)) + ph)
            for kx, ky, ph, a in waves
        )
        return base + amplitude * turn / norm

    return theta


def _streamline(x0, y0, theta, length_km, step, inside_at, avoid, clearance):
    """A range traced along the trend field both ways from a seed, inside the system."""
    halves = []
    for sign in (1.0, -1.0):
        pts = []
        x, y = x0, y0
        prev = None
        travelled = 0.0
        while travelled < 0.5 * length_km:
            a = float(theta(x, y)[0])
            dx, dy = math.cos(a), math.sin(a)
            if prev is not None and dx * prev[0] + dy * prev[1] < 0:
                dx, dy = -dx, -dy  # a trend has no sense: keep going the same way
            if prev is None:
                dx, dy = sign * dx, sign * dy
            x, y = x + dx * step, y + dy * step
            if not bool(inside_at(np.array([x]), np.array([y]))[0]):
                break
            if avoid is not None and avoid.query([x, y])[0] < clearance:
                break
            pts.append((x, y))
            prev = (dx, dy)
            travelled += step
        halves.append(pts)
    line = halves[1][::-1] + [(x0, y0)] + halves[0]
    return np.array(line, dtype=np.float64)


def _highland_ranges(
    plan, sid, px, py, axis, width, params, m, front, step, variety, inside_at, stream: Stream
):
    """An extensive mountain country: many medium ranges whose trends turn across it."""
    spacing = 230.0 * (1.0 + 0.3 * m) / max(variety, 0.5) ** 0.5
    theta = _orientation_field(axis, stream.child("trend"), max(width, 600.0), 1.05)
    order = stream.permutation(px.size)
    seeds: list = []
    built = []
    placed = None
    for idx in order.tolist():
        x, y = float(px[idx]), float(py[idx])
        if any(math.hypot(x - a, y - b) < spacing for a, b in seeds):
            continue
        seeds.append((x, y))
        u = stream.random(4)
        length = (160.0 + 380.0 * u[0]) * (1.0 + 0.3 * m)
        line = _streamline(x, y, theta, length, step, inside_at, placed, 0.42 * spacing)
        if len(line) < 3 or arclength(line)[-1] < max(110.0, 0.5 * length):
            continue
        spec = RangeSpec(
            0.0, 0.7 + 0.3 * u[1], 0.6 + 0.25 * u[2], 1.1, "highland", asym=(u[3] - 0.5)
        )
        rid = _add_range(plan, sid, line, spec, params, m, front, step)
        built.append((spec, [rid]))
        pts = (
            plan.ridges[rid].points
            if placed is None
            else np.vstack([placed.data, plan.ridges[rid].points])
        )
        placed = cKDTree(pts)
        if len(built) >= 60:
            break
    return built


def _composite_ranges(
    plan, sid, rows, cols, px, py, rel, axis, params, m, front, step, variety, inside_at,
    cell, stream: Stream,
):  # fmt: skip
    """Several fold regions meeting at an orogenic knot (the Pamir knot, Altai-Sayan).

    The knot is the highest part of the system. Arms run from it to the far ends of the
    system in different directions (at least 50 degrees apart); along each arm a bundle of
    parallel ridges fans out (virgation) and gets lower away from the knot.
    Returns ``(built, taper)``; ``taper`` maps ridge id -> (knot x, knot y, reach km).
    """
    # the knot: the cell whose surroundings (within ~2 cells) stand highest
    cells = np.column_stack([px, py])
    near = cKDTree(cells).query_ball_point(cells, r=2.1 * cell)
    smooth = np.array([float(rel[idx].mean()) for idx in near])
    knot = int(np.argmax(smooth))
    kx, ky = float(px[knot]), float(py[knot])
    dist = np.hypot(px - kx, py - ky)
    ang = np.arctan2(py - ky, px - kx)
    reach = float(dist.max())
    # candidate arm ends: the farthest cell of each 30-degree sector
    targets = []
    for k in range(12):
        lo = -math.pi + k * math.pi / 6
        sel = (ang >= lo) & (ang < lo + math.pi / 6)
        if sel.any():
            j = int(np.flatnonzero(sel)[np.argmax(dist[sel])])
            targets.append((float(dist[j]), j))
    targets.sort(reverse=True)
    arms: list[int] = []
    for d, j in targets:
        if d < 0.35 * reach or len(arms) >= 4:
            continue
        if all(
            abs(math.remainder(float(ang[j] - ang[a]), 2 * math.pi)) > math.radians(50)
            for a in arms
        ):
            arms.append(j)
    if len(arms) < 2:
        return [], {}
    cost_graph, _hop = _system_graph(rows, cols, rel, cell)
    _d, pred = dijkstra(cost_graph, indices=knot, return_predecessors=True)
    built = []
    taper: dict = {}
    hw = params.range_half_width_km
    for a_idx, target in enumerate(arms):
        path = _path(pred, target, {knot})[::-1]
        if len(path) < 3:
            continue
        guide = np.column_stack([px[path], py[path]])
        jit = (stream.random(2 * len(guide)).reshape(-1, 2) - 0.5) * 0.5 * cell
        jit[0] = 0.0
        guide = resample(chaikin(guide + jit, 4), step)
        if len(guide) < 4:
            continue
        g_s, g_t, g_n = _frame(guide)
        span = float(g_s[-1])
        u = stream.random(12)
        n_ranges = 2 + int(round(u[0] * 2 * max(variety, 0.5)))
        # virgation: the ridges of a bundle spread apart away from the knot
        spacing = 2.4 * hw * (1.0 + 1.3 * g_s / max(span, 1.0))
        for k in range(n_ranges):
            off = (k - 0.5 * (n_ranges - 1)) * spacing
            pts = guide + g_n * off[:, None]
            keep = inside_at(pts[:, 0], pts[:, 1])
            start = 0.08 + 0.2 * u[1 + k % 10] if abs(k - 0.5 * (n_ranges - 1)) > 0.6 else 0.0
            keep &= g_s >= start * span
            for lo, hi in _runs(keep):
                if g_s[hi - 1] - g_s[lo] < max(1.6 * cell, 110.0):
                    continue
                central = abs(k - 0.5 * (n_ranges - 1)) < 0.6
                spec = RangeSpec(
                    0.0,
                    1.0 if central else 0.66 + 0.2 * u[(k + 5) % 12],
                    0.55,
                    1.2,
                    "core" if central and a_idx == 0 else "fold",
                )
                rid = _add_range(plan, sid, pts[lo:hi], spec, params, m, front, step)
                taper[rid] = (kx, ky, reach)
                built.append((spec, [rid]))
    plan.knots.append(
        {
            "id": len(plan.knots),
            "system": sid,
            "x_km": kx,
            "y_km": ky,
            "ridges": [ids[0] for _spec, ids in built[: len(arms)]],
            "kind": "orogenic_knot",
            "arms": len(arms),
        }
    )
    return built, taper


def _lobes(specs, stream: Stream) -> list:
    """A plateau or a broad base is several overlapping lobes of different width and offset
    (Tibet, the Anatolian and Iranian plateaus), not one band of constant width."""
    out = []
    for sp in specs:
        if sp.role not in ("plateau", "base"):
            out.append(sp)
            continue
        lo, hi = sp.trim
        span = 1.0 - lo - hi
        n = 2 + int(stream.random() * 3)
        for k in range(n):
            u = stream.random(4)
            a = lo + span * k / n - 0.12 * span / n * (k > 0)
            b = lo + span * (k + 1) / n + 0.12 * span / n * (k < n - 1)
            out.append(
                RangeSpec(
                    sp.f + (u[0] - 0.5) * 0.3,
                    sp.crest * (0.9 + 0.2 * u[1]),
                    sp.hw * (0.7 + 0.6 * u[2]),
                    sp.sharp,
                    sp.role,
                    plateau=sp.plateau,
                    asym=(u[3] - 0.5) * 0.6,
                    trim=(max(a, 0.0), max(1.0 - b, 0.0)),
                    absolute_hw=sp.absolute_hw,
                    peaks=sp.peaks,
                    spurs=sp.spurs,
                    echelon=False,
                )
            )
    return out


def _filler_ranges(specs, style, width, maturity, variety, stream: Stream) -> list:
    """A belt much wider than its architecture gets more ranges, ~300 km apart."""
    if style in ("himalayan", "broad_upland", "volcanic_arc") or width < 600.0 or variety <= 0:
        return []
    spacing_km = 300.0 * (1.0 + 0.3 * maturity) / max(variety, 0.5)
    df = spacing_km / (0.41 * width)  # f is relative to each half-width (x 0.82)
    taken = [sp.f for sp in specs if sp.role not in ("base", "plateau")]
    out = []
    f = -0.85
    k = 0
    while f <= 0.85:
        u = stream.random(5)
        if all(abs(f - g) > 0.6 * df for g in taken):
            if style == "fold_belt":
                spec = RangeSpec(f, 0.45 + 0.3 * u[0], 0.45, 1.15, "fold")
            elif style == "old_massif":
                spec = RangeSpec(f, 0.4 + 0.25 * u[0], 1.3, 0.7, "secondary")
            else:
                spec = RangeSpec(
                    f + (u[1] - 0.5) * 0.3 * df, 0.5 + 0.3 * u[0], 0.8, 1.2, "secondary"
                )
            spec.trim = (0.05 + 0.3 * u[2], 0.05 + 0.3 * u[3])
            out.append(spec)
            taken.append(f)
        f += df
        k += 1
    return out


def _en_echelon(pts, spec: RangeSpec, stream: Stream, maturity: float):
    """A long range breaks into overlapping, laterally offset segments."""
    s = arclength(pts)
    total = float(s[-1])
    u = stream.random(32)
    limit = 900.0 + 900.0 * u[0]
    if spec.role in ("plateau", "base") or total < limit * 1.3 or not spec.echelon:
        return [pts]
    n = int(math.ceil(total / limit))
    out = []
    _s, _t, normals = _frame(pts)
    for k in range(n):
        a = k * total / n - (0.04 + 0.06 * u[1 + k % 30]) * total / n * (k > 0)
        b = (k + 1) * total / n + (0.04 + 0.06 * u[2 + k % 29]) * total / n * (k < n - 1)
        sel = (s >= a) & (s <= b)
        if sel.sum() < 3:
            continue
        shift = (u[3 + k % 28] - 0.5) * 2.0 * 60.0 * (k % 2 * 2 - 1)
        out.append(pts[sel] + normals[sel] * shift)
    return out or [pts]


def _add_range(
    plan, sid, pts, spec: RangeSpec, params, m, front, step, rank=RANK_MAIN, parent=-1
) -> int:
    pts = resample(chaikin(np.asarray(pts, dtype=np.float64), 2), step)
    hw = spec.hw if spec.absolute_hw else spec.hw * params.range_half_width_km
    if not spec.absolute_hw:
        hw *= 1.0 + 0.9 * m  # old ranges are broader
    sharp = max(0.45, spec.sharp * (1.0 - 0.5 * m))
    rid = len(plan.ridges)
    plan.ridges.append(
        Ridge(
            rid,
            sid,
            rank,
            parent,
            pts,
            np.full(len(pts), spec.crest),
            float(hw),
            sharpness=round(sharp, 4),
            plateau=spec.plateau,
            asymmetry=float(np.clip(spec.asym * front, -0.8, 0.8)),
            role=spec.role,
        )
    )
    return rid


def _nearest_ridge(plan, first, point) -> int:
    best, best_d = first, np.inf
    for r in plan.ridges[first:]:
        d = float(np.hypot(*(r.points - point).T).min())
        if d < best_d:
            best, best_d = r.id, d
    return best


def _crest_and_peaks(plan, ridge: Ridge, local, params, m, stream: Stream):
    s = arclength(ridge.points)
    total = float(s[-1])
    base = float(ridge.crest[0])
    if ridge.role in ("plateau", "base") or total < 1e-6:
        # a level top: gentle undulation only
        u = stream.random(4)
        wav = 1.0 + 0.04 * np.sin(2 * math.pi * s / max(total, 1.0) * (1 + 2 * u[0]) + 6.28 * u[1])
        ridge.crest = base * wav * (0.75 + 0.25 * local)
        return
    # massifs: a smooth envelope with a few higher sections
    u_env = stream.random(6)
    w = 2 * math.pi / max(total, 1.0)
    envelope = (
        0.82
        + 0.1 * np.sin(w * (0.7 + 1.5 * u_env[0]) * s + 6.28 * u_env[1])
        + 0.08 * np.sin(w * (2.5 + 3.0 * u_env[2]) * s + 6.28 * u_env[3])
    )
    spacing = params.peak_spacing_km * (1.0 + 0.6 * m)
    gaps = []
    pos = 0.0
    while pos < total and len(gaps) < 4000:
        g = spacing * float(np.clip(stream.normal() * 0.35 + 1.0, 0.35, 2.2))
        gaps.append(g)
        pos += g
    n_peaks = max(1, len(gaps) - 1)
    u = stream.random(3 * n_peaks + 3)
    peak_s = np.cumsum(gaps)[:n_peaks]
    peak_s = peak_s * (total / max(peak_s[-1] + 0.5 * gaps[-1], 1e-9))
    spread = 0.3 * (1.0 - 0.65 * m)  # old ranges have accordant summits
    peak_v = np.interp(peak_s, s, envelope) * (1.0 - spread + spread * u[:n_peaks])
    if peak_v.size:
        # the highest summit reaches the range's crest value; the main range has one
        # dominant summit above the others
        peak_v = peak_v / max(float(peak_v.max()), 1e-9)
        if ridge.role in ("main", "core"):
            dominant = int(u[n_peaks] * n_peaks)
            peak_v = np.where(np.arange(n_peaks) == dominant, peak_v, 0.93 * peak_v)
    nodes_s, nodes_v = [0.0], [0.42]
    peak_ids = []
    for k in range(n_peaks):
        if k > 0:
            saddle = 0.5 * (peak_s[k - 1] + peak_s[k])
            deep = u[2 * n_peaks + k] < 0.14
            # saddles stay high (a crest, not a string of beads); a few passes cut deep
            depth = (0.55 if deep else 0.72 + 0.16 * u[n_peaks + k]) + 0.08 * m
            nodes_s.append(saddle)
            nodes_v.append(min(peak_v[k - 1], peak_v[k]) * min(depth, 0.9))
        nodes_s.append(peak_s[k])
        nodes_v.append(peak_v[k])
    nodes_s.append(total + 1e-6)
    nodes_v.append(0.42)
    ridge.crest = (
        _cosine_profile(s, np.array(nodes_s), np.array(nodes_v)) * base * (0.55 + 0.45 * local)
    )
    if ridge.role == "frontal":
        return
    for k in range(n_peaks):
        i = int(np.argmin(np.abs(s - peak_s[k])))
        peak_ids.append(len(plan.peaks))
        plan.peaks.append(
            {
                "id": len(plan.peaks),
                "system": ridge.system,
                "ridge": ridge.id,
                "x_km": float(ridge.points[i, 0]),
                "y_km": float(ridge.points[i, 1]),
                "crest": float(ridge.crest[i]),
            }
        )
        if k > 0:
            j = int(np.argmin(np.abs(s - 0.5 * (peak_s[k - 1] + peak_s[k]))))
            a, b = plan.peaks[peak_ids[k - 1]], plan.peaks[peak_ids[k]]
            if float(ridge.crest[j]) < min(a["crest"], b["crest"]):
                plan.passes.append(
                    {
                        "id": len(plan.passes),
                        "system": ridge.system,
                        "ridge": ridge.id,
                        "x_km": float(ridge.points[j, 0]),
                        "y_km": float(ridge.points[j, 1]),
                        "crest": float(ridge.crest[j]),
                        "between_peaks": [peak_ids[k - 1], peak_ids[k]],
                    }
                )


def _flank_field(ridges, grid, relief, rel_max, seed, sid):
    """Scalar field whose descent lines are the spurs and valleys of a system.

    Ranges (their cross profiles) + the orogenic relief of the belt + a smooth
    perturbation. Descent lines of one field never cross, so spurs and valleys follow
    the flanks, bend with them and stop at the foot instead of crossing each other.
    """
    from ..noise import FractalNoise
    from .backend import ReferenceSampler, ridge_groups

    groups = ridge_groups(ridges, lambda r: r.sharpness if r.sharpness else 1.2)
    noise = FractalNoise(Stream(seed, f"mountains/v3/flanks/{sid}"), [190.0, 95.0], [1.0, 0.5])

    def exact(x, y):
        pts = np.column_stack([np.ravel(x), np.ravel(y)])
        structure = np.zeros(len(pts))
        for g in groups:
            structure = np.maximum(
                structure, ReferenceSampler._ridge_values(g, pts, exact=False)[0]
            )
        belt = grid.bilinear(relief, pts[:, 0], pts[:, 1]) / rel_max
        return structure + 0.35 * belt + 0.05 * noise(pts[:, 0], pts[:, 1])

    # evaluated once on a fine raster around the system, then interpolated
    allx = np.concatenate([g.px for g in groups])
    ally = np.concatenate([g.py for g in groups])
    margin = 450.0
    x0, x1 = float(allx.min()) - margin, float(allx.max()) + margin
    y0, y1 = float(ally.min()) - margin, float(ally.max()) + margin
    res = max(10.0, math.sqrt((x1 - x0) * (y1 - y0) / 40_000.0))
    nx, ny = int((x1 - x0) / res) + 2, int((y1 - y0) / res) + 2
    gxs, gys = np.meshgrid(x0 + res * np.arange(nx), y0 + res * np.arange(ny))
    values = exact(gxs, gys).reshape(ny, nx)

    def field(x, y):
        fr = (np.ravel(y) - y0) / res
        fc = (np.ravel(x) - x0) / res
        return ndimage.map_coordinates(values, np.vstack([fr, fc]), order=1, mode="nearest")

    return field


def _trace_down(field, starts, dirs, max_len, inside_at, step=8.0, stop=0.3):
    """Descent lines from ``starts`` (vectorised). Returns one polyline per start."""
    n = len(starts)
    pos = np.asarray(starts, dtype=np.float64).copy()
    prev = np.asarray(dirs, dtype=np.float64).copy()
    v0 = field(pos[:, 0], pos[:, 1])
    v_prev = v0.copy()
    alive = np.ones(n, dtype=bool)
    paths = [[p.copy()] for p in pos]
    h = 0.6 * step
    steps = int(np.ceil(float(np.max(max_len)) / step)) if n else 0
    for k in range(steps):
        idx = np.nonzero(alive)[0]
        if not idx.size:
            break
        x, y = pos[idx, 0], pos[idx, 1]
        gx = field(x + h, y) - field(x - h, y)
        gy = field(x, y + h) - field(x, y - h)
        g = np.hypot(gx, gy)
        down = np.column_stack([-gx, -gy]) / np.maximum(g, 1e-12)[:, None]
        inertia = 0.75 if k < 2 else 0.4
        d = np.where((g > 1e-6)[:, None], inertia * prev[idx] + (1 - inertia) * down, prev[idx])
        d /= np.maximum(np.hypot(d[:, 0], d[:, 1]), 1e-9)[:, None]
        new = pos[idx] + d * step
        v = field(new[:, 0], new[:, 1])
        ok = (v <= v_prev[idx] + 0.004) & (v > stop * v0[idx]) & inside_at(new[:, 0], new[:, 1])
        ok &= (k + 1) * step <= max_len[idx]
        for j, i in enumerate(idx):
            if ok[j]:
                paths[i].append(new[j].copy())
        pos[idx[ok]] = new[ok]
        prev[idx[ok]] = d[ok]
        v_prev[idx[ok]] = v[ok]
        alive[idx[~ok]] = False
    return [np.array(p) for p in paths]


def _spurs(plan, ridge: Ridge, field, inside_at, params, m, density, stream: Stream):
    """Irregular, dendritic spurs and the valleys between them, traced down the flanks."""
    s = arclength(ridge.points)
    total = float(s[-1])
    gap = (150.0 + 70.0 * m) / max(density, 1e-3)
    if total < 0.6 * gap or len(ridge.points) < 3:
        return
    starts, dirs, lens, sides, roots = [], [], [], [], []
    for side in (1, -1):
        pos = gap * stream.random()
        widen = 1.0 + 0.8 * ridge.asymmetry * side  # the wider flank has longer spurs
        hw_side = ridge.half_width_km * widen
        while pos < total - 0.1 * gap:
            i = int(np.clip(np.searchsorted(s, pos), 1, len(s) - 2))
            tangent = ridge.points[i + 1] - ridge.points[i - 1]
            tangent /= max(np.hypot(*tangent), 1e-9)
            normal = np.array([-tangent[1], tangent[0]]) * side
            ang = stream.normal() * math.radians(18.0)
            ca, sa = math.cos(ang), math.sin(ang)
            d = np.array([ca * normal[0] - sa * normal[1], sa * normal[0] + ca * normal[1]])
            starts.append(ridge.points[i] + normal * 0.2 * hw_side)
            dirs.append(d)
            reach = (2.4 * hw_side + 60.0) * float(np.exp(stream.normal() * 0.3)) * (1 - 0.3 * m)
            lens.append(min(reach, 360.0))
            sides.append(side)
            roots.append((pos, i, side))
            pos += gap * float(np.clip(stream.normal() * 0.45 + 1.0, 0.3, 2.4))
    if not starts:
        return
    paths = _trace_down(field, np.array(starts), np.array(dirs), np.array(lens), inside_at)
    spur_ids = []
    for (_pos, i, _side), path in zip(roots, paths, strict=True):
        if arclength(path)[-1] < 45.0:
            spur_ids.append(None)
            continue
        path = np.vstack([ridge.points[i], path])
        spur_ids.append(_spur_from_path(plan, ridge, path, ridge.crest[i], params, m, stream))
    # second-order spurs: leave a spur sideways and descend with the flank
    sub_starts, sub_dirs, sub_lens, sub_parent = [], [], [], []
    for rid in spur_ids:
        if rid is None or stream.random() > 0.6 * (1.0 - 0.5 * m):
            continue
        sp = plan.ridges[rid]
        if len(sp.points) < 6:
            continue
        q = int((0.25 + 0.4 * stream.random()) * (len(sp.points) - 1))
        tq = sp.points[q + 1] - sp.points[q - 1]
        tq /= max(np.hypot(*tq), 1e-9)
        side2 = 1.0 if stream.random() < 0.5 else -1.0
        d2 = tq * 0.55 + np.array([-tq[1], tq[0]]) * side2 * 0.8
        d2 /= max(np.hypot(*d2), 1e-9)
        sub_starts.append(sp.points[q] + d2 * 0.4 * sp.half_width_km)
        sub_dirs.append(d2)
        sub_lens.append(arclength(sp.points)[-1] * (0.25 + 0.25 * stream.random()))
        sub_parent.append((rid, q))
    if sub_starts:
        subs = _trace_down(
            field, np.array(sub_starts), np.array(sub_dirs), np.array(sub_lens), inside_at
        )
        for (rid, q), path in zip(sub_parent, subs, strict=True):
            sp = plan.ridges[rid]
            if arclength(path)[-1] < 30.0:
                continue
            path = np.vstack([sp.points[q], path])
            _spur_from_path(plan, sp, path, sp.crest[q], params, m, stream, half=0.7)
    # valleys: heads between neighbouring spurs of one flank, traced down the same field
    heads, hdirs, hlens = [], [], []
    by_side = {1: [], -1: []}
    for (pos, i, side), rid in zip(roots, spur_ids, strict=True):
        if rid is not None:
            by_side[side].append((pos, i, rid))
    for side, items in by_side.items():
        for (a0, i0, r0), (a1, i1, r1) in zip(items[:-1], items[1:], strict=True):
            if a1 - a0 > 2.6 * gap:
                continue
            mid = (i0 + i1) // 2
            tangent = ridge.points[min(mid + 1, len(s) - 1)] - ridge.points[max(mid - 1, 0)]
            tangent /= max(np.hypot(*tangent), 1e-9)
            normal = np.array([-tangent[1], tangent[0]]) * side
            heads.append(ridge.points[mid] + normal * 0.45 * ridge.half_width_km)
            hdirs.append(normal)
            hlens.append(
                1.1
                * 0.5
                * (arclength(plan.ridges[r0].points)[-1] + arclength(plan.ridges[r1].points)[-1])
            )
    if heads:
        vpaths = _trace_down(
            field, np.array(heads), np.array(hdirs), np.array(hlens), inside_at, stop=0.2
        )
        for path in vpaths:
            if arclength(path)[-1] < 40.0:
                continue
            pts = resample(chaikin(path, 2), 8.0)
            frac = arclength(pts) / max(arclength(pts)[-1], 1e-9)
            depth = np.clip(frac / 0.25, 0.0, 1.0) * (1.0 - 0.75 * frac**2)
            plan.valleys.append(
                Valley(
                    len(plan.valleys),
                    ridge.system,
                    ridge.id,
                    pts,
                    depth,
                    (14.0 + 18.0 * m) * (0.8 + 0.4 * stream.random()),
                )
            )


def _spur_from_path(plan, parent: Ridge, path, crest0, params, m, stream, half=1.0) -> int:
    pts = resample(chaikin(path, 2), 8.0)
    frac = arclength(pts) / max(arclength(pts)[-1], 1e-9)
    knob = 0.1 * np.sin(2 * math.pi * frac * (1.5 + 2.0 * stream.random()) + 6.28 * stream.random())
    crest = float(crest0) * (0.74 - 0.52 * frac**1.15) * (1.0 + knob)
    rid = len(plan.ridges)
    plan.ridges.append(
        Ridge(
            rid,
            parent.system,
            RANK_SPUR,
            parent.id,
            pts,
            crest,
            params.spur_half_width_km * half * (1.0 + 0.6 * m),
            role="spur",
        )
    )
    plan.knots.append(
        {
            "id": len(plan.knots),
            "system": parent.system,
            "x_km": float(pts[0, 0]),
            "y_km": float(pts[0, 1]),
            "ridges": [parent.id, rid],
        }
    )
    return rid


def _basins(plan, built, axis, normals, s, left, right, front, params, m, step):
    """Wide depressions between neighbouring parallel ranges (Fergana, Kura, Tsaidam)."""
    ranges = sorted(
        (
            (spec.f * front, spec)
            for spec, _ids in built
            if spec.role not in ("plateau", "base", "branch")
        ),
        key=lambda item: item[0],
    )
    if len(ranges) < 2:
        return
    sid = len(plan.systems) - 1
    for (fa, sa), (fb, sb) in zip(ranges[:-1], ranges[1:], strict=True):
        if fa * fb < 0 and min(abs(fa), abs(fb)) < 0.2:
            continue  # the axis itself lies between them: no room for a basin
        fm = 0.5 * (fa + fb)
        lateral = np.where(fm > 0, left, right) * abs(fm) * 0.82
        gap_km = float(np.median(np.where(fm > 0, left, right))) * abs(fb - fa) * 0.82
        hw = 0.32 * gap_km
        if hw < 0.8 * params.range_half_width_km:
            continue
        lo = max(sa.trim[0], sb.trim[0])
        hi = max(sa.trim[1], sb.trim[1])
        total = float(s[-1])
        sel = (s >= lo * total) & (s <= (1.0 - hi) * total)
        if sel.sum() < 4:
            continue
        pts = axis[sel] + normals[sel] * (np.sign(fm) * lateral[sel])[:, None]
        pts = resample(chaikin(pts, 2), step)
        frac = arclength(pts) / max(arclength(pts)[-1], 1e-9)
        depth = (0.42 + 0.12 * m) * np.clip(np.minimum(frac, 1.0 - frac) / 0.12, 0.0, 1.0)
        plan.valleys.append(Valley(len(plan.valleys), sid, -1, pts, depth, hw))
