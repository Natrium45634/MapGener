"""Biogeography graphs of TZ 5.5 §9 (G3): one sparse directed graph per mobility class on the
native hex grid (degree ≤ 6), edge costs in physical kilometres (B31), multi-source
Dijkstra from the lineage pools (O(E log V), no all-pairs paths), and closed-form
occupancy.

    cost(i→j) = spacing_km · (m_i + m_j) / 2 · drift(i→j)      (m = ∞: no edge)
    drift     = clip(exp(−u·e / u0), 0.3, 3) for passive drifters on currents / rivers
    c = κ0 · exp(−d_eff / L),  e = e0 · (A_ref / A)^0.25 · M^0.25 / max(S, 0.05)
    p(t) = p* + (p0 − p*) · exp(−(c + e) t),  p* = c / (c + e)            (N07, N08, B35)

Lineage pools are physical units: large source landmasses (land and air), large drainage
basins (fresh water) and ocean basin × thermal belt (sea). An oceanic island, a small
basin or a new volcano gets its biota only along the graph — never by noise (B27, B28).
History of past land bridges is not available in the world (BLOCKED) → the quality is
``present_only`` and no past connection is invented (B33).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra

INF = math.inf
#: colonisation rate at zero distance (1/yr) and the time scale of a continent (yr) [К]
KAPPA0 = 1e-2
T_CONTINENT_YR = 5.0e6
T_UNKNOWN_YR = 1.0e5
#: extinction reference area, km², and the minimum area of a source pool [К]
A_REF_KM2 = 1.0e4
SOURCE_LAND_KM2 = 2.0e5
SOURCE_BASIN_KM2 = 4.0e4
SOURCE_FRAGMENT_KM2 = 2.0e4
#: long-distance dispersal: share and scale multiplier of the kernel tail [К]
LDD_SHARE = 1e-3
LDD_SCALE = 20.0
#: current speed giving a factor e in the drift cost, m/s
U0 = 0.2
#: cost multiplier of a headwater capture between two drainage basins (None: forbidden)
CAPTURE_MULT = 30.0

#: node cost multipliers per class: land, fresh, ocean, ice, mountain extra
CLASS = {
    "ground": (1.0, 2.0, INF, 4.0, 2.0),
    "arboreal": (1.0, 4.0, INF, INF, 3.0),
    "river_ground": (2.0, 1.0, 6.0, 6.0, 3.0),
    "flight": (1.0, 1.0, 3.0, 2.0, 0.5),
    "ground_flight": (1.0, 1.0, 8.0, 4.0, 1.0),
    "seed": (1.0, 1.0, 3.0, 3.0, 1.0),
    "spore": (1.0, 1.0, 1.5, 1.0, 0.0),
    "spore_and_fragment": (1.0, 1.0, 1.5, 1.0, 0.0),
    "floating_seed": (INF, 3.0, 1.0, INF, INF),
    "water_seed": (4.0, 1.0, 10.0, 6.0, 2.0),
    "water_drift": (15.0, 1.0, 15.0, 20.0, 5.0),
    "river_flight": (3.0, 1.0, 15.0, 6.0, 2.0),
    "current_drift": (INF, INF, 1.0, INF, INF),
    "larval_drift": (INF, INF, 1.0, INF, INF),
    "marine_propagule_drift": (INF, INF, 1.0, INF, INF),
    "ocean": (INF, INF, 1.0, INF, INF),
    "ocean_coast": (INF, INF, 1.0, INF, INF),
    "shelf": (INF, INF, 1.0, INF, INF),
    "deep": (INF, INF, 1.0, INF, INF),
    "passive": (1.0, 1.0, 1.0, 1.0, 0.0),
    "host_dependent": (1.0, 1.0, 1.0, 1.0, 0.0),
    "river": (INF, 1.0, INF, INF, INF),
    "river_ocean": (INF, 1.0, 1.0, INF, INF),
}  # fmt: skip
DRIFT = ("current_drift", "larval_drift", "marine_propagule_drift", "floating_seed")
FRESH_CLASSES = ("river", "river_ocean", "water_drift", "water_seed", "river_flight")
MARINE_CLASSES = ("current_drift", "larval_drift", "marine_propagule_drift", "ocean",
                  "ocean_coast", "shelf", "deep", "floating_seed")  # fmt: skip
COSMO = ("passive", "host_dependent")


@dataclass
class Lineages:
    """Lineage pools of one family of classes: per pool a stable key and a display name."""

    keys: list
    names: list
    node_pool: np.ndarray  # pool index per node (-1: not a source)


def _node_costs(cls, inp, frac, canopy=None):
    f = inp.f
    lm, fm, om, im, mm = CLASS[cls]
    land = frac["land_ground"] + frac["grounded_ice"] > 0.5
    fresh = (frac["inland_water"] > 0) | f["river"]
    ocean = frac["marine_water"] > 0.5
    m = np.full(inp.n, INF)
    m = np.where(land, lm, m)
    m = np.where(fresh & ~ocean, np.minimum(m, fm), m)
    m = np.where(ocean, om, m)
    if cls == "ground":  # a long winter ice bridge over the sea (≥ 9 months of sea ice)
        m = np.where(ocean & (f["sea_ice_months"] >= 9), 3.0, m)
    if cls == "river_ground":  # only along the coast, not across the open ocean
        m = np.where(ocean & ~f["coast_sea"], INF, m)
    if cls == "floating_seed":
        m = np.where(f["coast_land"], 1.0, m)
    if cls == "ocean_coast":
        m = np.where(f["coast_land"], 2.0, m)
    ice = frac["grounded_ice"] > 0.5
    m = np.where(ice & np.isfinite(m) & land, im, m)
    high = np.nan_to_num(f["elevation_m"]) > 3000.0
    m = np.where(high & land & np.isfinite(m), m + mm, m)
    depth = np.nan_to_num(f["depth_m"])
    if cls == "shelf":
        m = np.where(ocean, np.where(depth <= 200, 1.0, np.where(depth <= 2000, 4.0, 10.0)), m)
    if cls == "deep":
        m = np.where(ocean, np.where(depth > 1000, 1.0, 5.0), m)
    if cls in ("larval_drift", "marine_propagule_drift"):
        m = np.where(ocean & (depth > 2000), 3.0, m)
    if cls == "arboreal" and canopy is not None:
        m = np.where(land, 1.0 + 4.0 * (1.0 - np.clip(canopy, 0, 1)), m)
    return m


def _drift_factor(inp, src, dst, u=None, v=None):
    """Directional cost factor along the mean surface current (x east, y south: v points
    north, so the southward unit is −v)."""
    f = inp.f
    u = f["current_u_ms"] if u is None else u
    v = f["current_v_ms"] if v is None else v
    dx = inp.x_km[dst] - inp.x_km[src]
    dy = inp.y_km[dst] - inp.y_km[src]
    L = np.maximum(np.hypot(dx, dy), 1e-9)
    along = (np.nan_to_num(u[src]) * dx / L - np.nan_to_num(v[src]) * dy / L)
    return np.clip(np.exp(-along / U0), 0.3, 3.0)


def build_graph(cls, inp, frac, canopy=None, capture_mult=CAPTURE_MULT):
    """Directed sparse graph of one mobility class (csr, costs in km)."""
    f = inp.f
    n = inp.n
    if cls in ("river", "river_ocean"):
        return _river_graph(cls, inp, frac, capture_mult)
    m = _node_costs(cls, inp, frac, canopy)
    nb = inp.nbr
    src = np.repeat(np.arange(n), 6)
    dst = nb.ravel()
    ok = dst >= 0
    src, dst = src[ok], dst[ok]
    w = 0.5 * (m[src] + m[dst])
    keep = np.isfinite(w)
    src, dst, w = src[keep], dst[keep], w[keep] * inp.spacing_km
    if cls in DRIFT:
        w = w * _drift_factor(inp, src, dst)
    if cls == "water_drift":  # downstream drift in rivers is cheaper than upstream (B29)
        down = f["flow_receiver"][src] == dst
        w = np.where(down, 0.5 * w, w)
    return csr_matrix((w, (src, dst)), shape=(n, n))


def _river_graph(cls, inp, frac, capture_mult):
    """Freshwater graph: downstream along ``flow_receiver`` (cost 1), upstream ×2 (B29),
    lakes both ways; a link between two drainage basins only as a headwater capture with
    ``capture_mult`` (None forbids it, B49); ``river_ocean`` adds the sea and the mouths."""
    f = inp.f
    n = inp.n
    fresh = (frac["inland_water"] > 0) | f["river"]
    rec = f["flow_receiver"]
    s = inp.spacing_km
    rows, cols, w = [], [], []
    a = np.nonzero(fresh & (rec >= 0) & (rec < n))[0]
    b = rec[a]
    ok = fresh[b] | (frac["marine_water"][b] > 0.5)
    a, b = a[ok], b[ok]
    to_sea = frac["marine_water"][b] > 0.5
    if cls == "river":
        a, b, to_sea = a[~to_sea], b[~to_sea], to_sea[~to_sea]
    rows += [a, b]
    cols += [b, a]
    w += [np.full(a.size, s), np.full(a.size, 2.0 * s)]
    # neighbouring fresh cells of the same lake or basin, both ways
    nb = inp.nbr
    i = np.repeat(np.arange(n), 6)
    j = nb.ravel()
    okk = (j >= 0)
    i, j = i[okk], j[okk]
    both = fresh[i] & fresh[j]
    same_lake = both & (f["lake_id"][i] >= 0) & (f["lake_id"][i] == f["lake_id"][j])
    rows.append(i[same_lake])
    cols.append(j[same_lake])
    w.append(np.full(int(same_lake.sum()), s))
    if capture_mult is not None:
        cap = both & (f["basin_id"][i] != f["basin_id"][j]) & ~same_lake
        rows.append(i[cap])
        cols.append(j[cap])
        w.append(np.full(int(cap.sum()), s * capture_mult))
    if cls == "river_ocean":
        ocean = frac["marine_water"] > 0.5
        sea = ocean[i] & ocean[j]
        rows.append(i[sea])
        cols.append(j[sea])
        w.append(np.full(int(sea.sum()), s))
        # mouths: a river cell next to the sea
        mouth = (fresh[i] & ocean[j]) | (ocean[i] & fresh[j])
        rows.append(i[mouth])
        cols.append(j[mouth])
        w.append(np.full(int(mouth.sum()), s))
    r = np.concatenate(rows)
    c = np.concatenate(cols)
    ww = np.concatenate(w)
    return csr_matrix((ww, (r, c)), shape=(n, n))


# -- lineage pools ------------------------------------------------------------------------------
def _key(prefix, cx, cy):
    return f"{prefix}:{int(round(cx / 50.0))}:{int(round(cy / 50.0))}"


def land_pools(inp, frac, landmasses) -> Lineages:
    """Source landmasses: area ≥ SOURCE_LAND_KM2 and mostly continental crust."""
    f = inp.f
    lm = f["landmass_id"]
    pool = np.full(inp.n, -1, dtype=np.int64)
    keys, names = [], []
    by_id = {int(m["id"]): m for m in landmasses}
    ids, counts = np.unique(lm[lm >= 0], return_counts=True)
    order = sorted(ids.tolist(), key=lambda i: (-counts[list(ids).index(i)], i))
    for i in order:
        sel = lm == i
        area = float(sel.sum()) * inp.area_km2
        cont = float(np.mean(f["block_id"][sel] >= 0))
        # a large landmass, or an old continental fragment (its own ancient lineages, the
        # way a micro-continent keeps them), is a pool; an oceanic island is not
        if not ((area >= SOURCE_LAND_KM2 and cont >= 0.5)
                or (area >= SOURCE_FRAGMENT_KM2 and cont >= 0.9)):
            continue
        cx, cy = float(inp.x_km[sel].mean()), float(inp.y_km[sel].mean())
        keys.append(_key("land", cx, cy))
        names.append(str(by_id.get(int(i), {}).get("name", f"суша {i}")))
        pool[sel] = len(keys) - 1
    if not keys:  # no large landmass: the largest one is the only pool
        if ids.size:
            i = int(ids[np.argmax(counts)])
            sel = lm == i
            keys.append(_key("land", float(inp.x_km[sel].mean()), float(inp.y_km[sel].mean())))
            names.append(str(by_id.get(i, {}).get("name", f"суша {i}")))
            pool[sel] = 0
    return Lineages(keys, names, pool)


def fresh_pools(inp, frac, land: Lineages) -> Lineages:
    """Source drainages: basins ≥ SOURCE_BASIN_KM2 on a source landmass."""
    f = inp.f
    fresh = (frac["inland_water"] > 0) | f["river"]
    basin = f["basin_id"]
    pool = np.full(inp.n, -1, dtype=np.int64)
    keys, names = [], []
    ids, counts = np.unique(basin[basin >= 0], return_counts=True)
    for i, c in sorted(zip(ids.tolist(), counts.tolist(), strict=True), key=lambda t: (-t[1], t[0])):
        sel = basin == i
        if c * inp.area_km2 < SOURCE_BASIN_KM2:
            continue
        lp = land.node_pool[sel]
        if not (lp >= 0).any():
            continue
        cx, cy = float(inp.x_km[sel].mean()), float(inp.y_km[sel].mean())
        keys.append(_key("fw", cx, cy))
        names.append(f"бассейн {len(keys)} ({land.names[int(np.bincount(lp[lp >= 0]).argmax())]})")
        pool[sel & fresh] = len(keys) - 1
    return Lineages(keys, names, pool)


THERMAL_BELTS = ((20.0, "тропический"), (10.0, "умеренный"), (-99.0, "холодный"))


def marine_pools(inp, frac) -> Lineages:
    """Ocean basin × thermal belt of the annual SST: the marine lineage pools."""
    f = inp.f
    ocean = frac["marine_water"] > 0.5
    with np.errstate(invalid="ignore"):
        cnt = np.isfinite(f["sst_m"]).sum(axis=0)
        sst = np.where(cnt > 0, np.nansum(f["sst_m"], axis=0) / np.maximum(cnt, 1), np.nan)
    belt = np.where(sst >= 20, 0, np.where(sst >= 10, 1, 2))
    basin = f["ocean_basin_id"]
    pool = np.full(inp.n, -1, dtype=np.int64)
    keys, names = [], []
    for b in sorted(np.unique(basin[ocean]).tolist()):
        for k, (_t, title) in enumerate(THERMAL_BELTS):
            sel = ocean & (basin == b) & (belt == k)
            if sel.sum() < 3:
                continue
            keys.append(_key(f"sea{k}", float(inp.x_km[sel].mean()), float(inp.y_km[sel].mean())))
            names.append(f"{title} пояс, бассейн {b}")
            pool[sel] = len(keys) - 1
    return Lineages(keys, names, pool)


def nearest_lineage(graph, pools: Lineages):
    """Multi-source Dijkstra: effective distance (km) to the nearest pool and its index."""
    src = np.nonzero(pools.node_pool >= 0)[0]
    n = graph.shape[0]
    if src.size == 0:
        return np.full(n, INF), np.full(n, -1, dtype=np.int64)
    dist, _pred, srcs = dijkstra(graph, directed=True, indices=src, min_only=True,
                                 return_predecessors=True)  # fmt: skip
    lin = np.where(srcs >= 0, pools.node_pool[np.maximum(srcs, 0)], -1)
    return dist, lin


# -- occupancy ----------------------------------------------------------------------------------
def colonisation_rate(d_eff_km, L_km, kappa0=KAPPA0, rho=LDD_SHARE, ldd=LDD_SCALE):
    """Fat-tailed kernel: the local exponential plus a rare long-distance share ``rho``
    with the scale ``ldd · L`` (storms, rafting, birds). ``rho = 0`` is the pure
    exponential of N07."""
    d = np.asarray(d_eff_km, dtype=np.float64)
    return kappa0 * ((1.0 - rho) * np.exp(-d / L_km) + rho * np.exp(-d / (ldd * L_km)))


def occupancy(c, e, t, p0=0.0):
    """Closed form of dp/dt = c (1 − p) − e p with constant rates (B35): exact, in [0, 1];
    c = e = 0 keeps p0."""
    c = np.asarray(c, dtype=np.float64)
    e = np.asarray(e, dtype=np.float64)
    s = c + e
    with np.errstate(invalid="ignore", divide="ignore"):
        pstar = np.where(s > 0, c / np.where(s > 0, s, 1.0), p0)
    return np.clip(pstar + (p0 - pstar) * np.exp(-s * np.asarray(t, dtype=np.float64)), 0.0, 1.0)


def extinction_rate(e0, area_km2, mass_kg, suit):
    return (e0 * (A_REF_KM2 / np.maximum(area_km2, 1.0)) ** 0.25
            * max(mass_kg, 1e-6) ** 0.25 / np.maximum(suit, 0.05))  # fmt: skip


def substrate_age_yr(inp, frac, land: Lineages):
    """Time available for colonisation: continents T_CONTINENT_YR; an island off the source
    pools from its volcanic / crust age (unknown → T_UNKNOWN_YR, flagged); fresh lava from
    its own age."""
    f = inp.f
    age = np.full(inp.n, T_CONTINENT_YR)
    island = (land.node_pool < 0) & (frac["land_ground"] > 0)
    va = f["volcanic_age_ma"] * 1e6
    ca = f["crust_age_ma"] * 1e6
    a_isl = np.where(np.isfinite(va) & (va > 0), va, np.where(np.isfinite(ca) & (ca > 0), ca,
                                                              T_UNKNOWN_YR))  # fmt: skip
    age = np.where(island, np.minimum(a_isl, T_CONTINENT_YR), age)
    lava = f["lava_fraction"] > 0.5
    age = np.where(lava, np.minimum(age, f["lava_youngest_years_bp"]), age)
    unknown = island & ~(np.isfinite(va) & (va > 0)) & ~(np.isfinite(ca) & (ca > 0))
    return age, unknown
