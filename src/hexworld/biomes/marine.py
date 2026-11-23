"""Marine habitat axes (TZ 5.4.1 G6 sec. 10): light, nutrients, productivity, benthic and
pelagic zones, coral / estuary potential, marine ecoregions. Everything here is a
*potential* (suitability from physical inputs); no species, no dispersal, no carbon.

    PAR(z) = PAR_0 ((1 - c_ice) + c_ice t_ice) exp(-K_d z)      z_eu = ln(100) / K_d  (N19)
    K_d    = 0.06 (clear ocean) + 0.12 (shelf) + 0.5 x river-mouth turbidity      [K]
    nutrient_supply_proxy = clip(0.6 up / 0.5 m/day + 0.3 river + 0.3 mixing)   (0..1)
    productivity_potential = f_light f_nutrient f_T f_ice                          (0..1)

The bottom light uses the mean depth of the hex (no sub-hex hypsometry: flagged).
Coral thermal stress needs daily SST; :func:`degree_heating_weeks` implements the
DHW proxy for a daily series (N20) and the normals report ``not_available``.
"""

from __future__ import annotations

import math

import numpy as np
from scipy import ndimage

KD_CLEAR = 0.06
KD_SHELF = 0.12
KD_RIVER = 0.5
PELAGIC = ("none", "warm", "temperate", "cold", "ice_margin")
BENTHIC = ("none", "shelf", "slope", "abyssal", "trench")


def par_fraction(z_m, kd, ice_c=0.0, transmission=0.0):
    """Share of the surface PAR at depth ``z`` (ice: opaque except ``transmission``)."""
    surf = (1.0 - np.asarray(ice_c, dtype=np.float64)) + np.asarray(ice_c) * transmission
    return surf * np.exp(-np.asarray(kd, dtype=np.float64) * np.asarray(z_m, dtype=np.float64))


def euphotic_depth(kd):
    return math.log(100.0) / np.asarray(kd, dtype=np.float64)


def degree_heating_weeks(sst_daily, mmm, window=84, threshold=1.0):
    """DHW proxy (degC-weeks) at the end of a daily series: HotSpots >= ``threshold`` above
    the maximum monthly mean, summed over the last ``window`` days / 7 (N20). Not NOAA's
    operational product."""
    hs = np.asarray(sst_daily, dtype=np.float64)[-window:] - mmm
    return float(np.sum(np.where(hs >= threshold, hs, 0.0)) / 7.0)


def habitats(depth_m, ocean, sst_m, ice_m, upwelling_m_day, river_mouth, trench, coast_hex,
             warm_coast_kind, nbr):  # fmt: skip
    """Per-hex marine axes; arrays flat over the hexes. Returns a dict of layers."""
    d = np.nan_to_num(np.asarray(depth_m, dtype=np.float64))
    n = d.size
    sst_ann = sst_m.mean(axis=0)
    sst_min, sst_max = sst_m.min(axis=0), sst_m.max(axis=0)
    ice_max = ice_m.max(axis=0)
    ice_mean = ice_m.mean(axis=0)
    # turbidity near river mouths (the mouth and its neighbours)
    near_mouth = river_mouth.copy()
    for k in range(nbr.shape[1]):
        j = nbr[:, k]
        near_mouth |= (j >= 0) & river_mouth[np.maximum(j, 0)]
    shelf = d <= 200.0
    kd = KD_CLEAR + KD_SHELF * shelf + KD_RIVER * near_mouth
    z_eu = euphotic_depth(kd)
    bottom = np.where(ocean, par_fraction(d, kd, ice_mean), 0.0)
    up = np.maximum(np.asarray(upwelling_m_day, dtype=np.float64), 0.0)
    mixing = np.clip((sst_max - sst_min) / 12.0, 0.0, 1.0)
    nutr = np.clip(0.6 * up / 0.5 + 0.3 * near_mouth + 0.3 * mixing, 0.0, 1.0)
    f_t = np.clip((sst_ann + 2.0) / 8.0, 0.0, 1.0) * np.clip((33.0 - sst_max) / 3.0, 0.0, 1.0)
    f_ice = 1.0 - ice_mean
    prod = np.where(ocean, (0.3 + 0.7 * nutr) * f_t * f_ice, 0.0)
    # pelagic zone from the seasonal SST and ice, benthic zone from the depth
    pel = np.zeros(n, dtype=np.uint8)
    pel = np.where(ocean & (sst_ann >= 20.0), 1, pel)
    pel = np.where(ocean & (sst_ann < 20.0) & (sst_ann >= 10.0), 2, pel)
    pel = np.where(ocean & (sst_ann < 10.0), 3, pel)
    pel = np.where(ocean & (ice_max >= 0.15) & (ice_mean < 0.8), 4, pel)
    ben = np.zeros(n, dtype=np.uint8)
    ben = np.where(ocean & shelf, 1, ben)
    ben = np.where(ocean & ~shelf & (d <= 2000.0), 2, ben)
    ben = np.where(ocean & (d > 2000.0), 3, ben)
    ben = np.where(ocean & trench, 4, ben)
    lit = bottom >= 0.01
    coral = (ocean & lit & (d <= 40.0) & (sst_min >= 18.0) & (sst_max <= 31.0) & ~near_mouth
             & (ice_max == 0)).astype(np.float64)  # fmt: skip
    coral *= np.clip((sst_min - 18.0) / 3.0, 0.0, 1.0)
    estuary = (ocean & river_mouth).astype(np.float64)
    mangrove = (coast_hex & warm_coast_kind & (sst_min >= 20.0)).astype(np.float64)
    return {
        "marine_kd_per_m": np.where(ocean, kd, np.nan),
        "marine_euphotic_depth_m": np.where(ocean, z_eu, np.nan),
        "marine_bottom_light_fraction": np.where(ocean, bottom, np.nan),
        "marine_nutrient_supply_proxy": np.where(ocean, nutr, np.nan),
        "marine_productivity_potential": np.where(ocean, prod, np.nan),
        "marine_pelagic_zone": pel,
        "marine_benthic_zone": ben,
        "marine_coral_reef_potential": np.where(ocean, coral, np.nan),
        "marine_estuary_potential": np.where(ocean, estuary, np.nan),
        "mangrove_saltmarsh_potential": mangrove,
    }


def ecoregions(ocean, sst_ann, ice_max, depth_m, nbr):
    """Connected zones of one SST band x ice regime x depth regime; stable ids (-1 land).

    Ids are assigned in the order of the lowest hex index of each component, so the same
    geometry gives the same ids; isolated basins with equal SST are separate regions.
    """
    band = np.clip(np.floor((np.asarray(sst_ann) + 2.0) / 6.0), 0, 6).astype(np.int64)
    ice = (np.asarray(ice_max) >= 0.15).astype(np.int64)
    deep = (np.nan_to_num(np.asarray(depth_m)) > 200.0).astype(np.int64)
    key = np.where(ocean, band * 4 + ice * 2 + deep, -1)
    n = key.size
    parent = np.arange(n)

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for k in range(nbr.shape[1]):
        j = nbr[:, k]
        ok = np.nonzero((j >= 0) & (key >= 0))[0]
        for i in ok:
            jj = j[i]
            if key[jj] == key[i]:
                a, b = find(i), find(jj)
                if a != b:
                    parent[max(a, b)] = min(a, b)
    roots = np.array([find(i) for i in range(n)])
    out = np.full(n, -1, dtype=np.int32)
    ids = {}
    for i in range(n):
        if key[i] < 0:
            continue
        r = roots[i]
        if r not in ids:
            ids[r] = len(ids)
        out[i] = ids[r]
    return out


def ecoregions_fast(ocean, sst_ann, ice_max, depth_m, shape):
    """Same as :func:`ecoregions` on the (H, W) array with 8-connectivity-free labelling."""
    band = np.clip(np.floor((np.asarray(sst_ann) + 2.0) / 6.0), 0, 6).astype(np.int64)
    ice = (np.asarray(ice_max) >= 0.15).astype(np.int64)
    deep = (np.nan_to_num(np.asarray(depth_m)) > 200.0).astype(np.int64)
    key = np.where(ocean, band * 4 + ice * 2 + deep, -1).reshape(shape)
    out = np.full(shape, -1, dtype=np.int32)
    nxt = 0
    for v in np.unique(key[key >= 0]):
        lab, k = ndimage.label(key == v)
        if k == 0:
            continue
        first = ndimage.minimum(np.arange(lab.size).reshape(shape), lab, range(1, k + 1))
        order = np.argsort(first)
        remap = np.zeros(k + 1, dtype=np.int64)
        remap[order + 1] = nxt + np.arange(k)
        out[lab > 0] = remap[lab[lab > 0]]
        nxt += k
    # renumber by the first hex of each region (stable, independent of the key order)
    flat = out.ravel()
    valid = flat >= 0
    vals, first = np.unique(flat[valid], return_index=True)
    rank = np.empty(vals.size, dtype=np.int32)
    rank[np.argsort(first)] = np.arange(vals.size, dtype=np.int32)
    lut = np.full(int(vals.max()) + 1 if vals.size else 1, -1, dtype=np.int32)
    lut[vals] = rank
    res = np.full(flat.size, -1, dtype=np.int32)
    res[valid] = lut[flat[valid]]
    return res.reshape(shape)
