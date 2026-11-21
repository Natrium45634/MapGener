"""v4 stage ``coasts``: the type of every coastal hex, from the processes that make coasts.

Geography distinguishes coasts by what shaped them (Shepard 1973: primary coasts shaped by
land processes, secondary coasts shaped by the sea; Valentin 1952: emergence/submergence
against deposition/erosion; Davies 1964/1980: wave and tide energy). HexWorld derives the
same causes from its own fields and classifies each land hex that touches the sea:

======================  ==================================================  ===================
type                    cause in the model                                   Earth examples
======================  ==================================================  ===================
``ice``                 perennial ice reaches the sea today                  Antarctica
``fjord``               glaciated at the last glacial maximum + high relief  Norway, Chile
``skerry``              glaciated + low relief + hard rock                   Finland, Sweden
``volcanic``            volcanic edifices at the shore                       Hawaii, Iceland
``delta``               river mouth with a large sediment load, waves and    Nile, Ganges,
                        tides cannot remove it                                Mississippi
``estuary``             river mouth with tides / little sediment             Thames, Elbe
``liman``               lowland river mouth in a tideless sea, drowned       Black Sea limans
``coral_reef``          warm clear water, no river mud                       Red Sea, Queensland
``mangrove``            tropical, sheltered, muddy, wet                      Sundarbans, Florida
``sabkha``              hot desert, flat, sheltered                          Persian Gulf
``tidal_flat``          large tides on a wide shelf, gentle land             Wadden Sea
``fault_scarp``         steep shore on an active plate margin                Red Sea, California
``dalmatian``           drowned ridges parallel to the coast                 Dalmatia
``ria``                 drowned river valleys in hilly, unglaciated land     Galicia, Brittany
``cliff``               steep hard shore exposed to waves (abrasion)         Dover, Moher
``barrier_lagoon``      low plain, wide shelf, small tides, sand supply      US Gulf, Baltic spits
``beach``               gentle shore with sand (default accumulation coast)  most low coasts
======================  ==================================================  ===================

The drivers are stored as layers too (wave exposure, tidal range, shelf width, sea surface
temperature, last-glacial ice), so a later stage (islands, deltas, coast symbols) can reuse
them. The tidal range and the wave exposure are proxies, not ocean models.
"""

from __future__ import annotations

import math

import numpy as np
from scipy import ndimage
from scipy.spatial import cKDTree

COAST_TYPES = (
    "none",
    "ice",
    "fjord",
    "skerry",
    "volcanic",
    "delta",
    "estuary",
    "liman",
    "coral_reef",
    "mangrove",
    "sabkha",
    "tidal_flat",
    "fault_scarp",
    "dalmatian",
    "ria",
    "cliff",
    "barrier_lagoon",
    "beach",
)
COAST_TITLES = {
    "ice": "ледовый (шельфовые ледники)",
    "fjord": "фьордовый",
    "skerry": "шхерный",
    "volcanic": "вулканический",
    "delta": "дельтовый",
    "estuary": "эстуарный",
    "liman": "лиманный",
    "coral_reef": "коралловый (рифовый)",
    "mangrove": "мангровый",
    "sabkha": "себховый (аридные осушки)",
    "tidal_flat": "ваттовый (приливные осушки, марши)",
    "fault_scarp": "сбросовый (тектонический уступ)",
    "dalmatian": "далматинский",
    "ria": "риасовый",
    "cliff": "абразионный (клифы)",
    "barrier_lagoon": "лагунный (барьерные острова, косы)",
    "beach": "пляжевый (выровненный аккумулятивный)",
}
#: Rocky (erosional / primary hard-rock) and soft (depositional) groups, for statistics.
ROCKY = ("fjord", "skerry", "volcanic", "fault_scarp", "dalmatian", "ria", "cliff", "ice")
SANDY = ("beach", "barrier_lagoon")
_CODE = {name: i for i, name in enumerate(COAST_TYPES)}
#: Mean discharge from which a river mouth makes its own coast type (delta, estuary, liman).
RIVER_MOUTH_M3_S = 2500.0


def _hex_neighbors(layout):
    from .raster import neighbor_index

    out = []
    for d in range(6):
        nb, valid = neighbor_index(layout, d)
        out.append((nb, valid))
    return out


def _sampling(layout):
    return (layout.spacing_km, 0.75 * layout.spacing_km * 2.0 / math.sqrt(3.0))


def coast_fields(layout, layers: dict, hist_layers: dict, ridges, sea_level_m: float, seed: int):
    """Drivers and type of every coastal hex. ``layers`` are the final v4 hex layers."""
    z = np.asarray(layers["elevation_m"], dtype=np.float64)
    shape = z.shape
    lake = np.asarray(layers["lake_id"]) >= 0
    # the world ocean (v4.6: connectivity, not the sign of the height) when known
    if "ocean_mask" in layers:
        land = ~np.asarray(layers["ocean_mask"], dtype=bool)
    else:
        land = z > sea_level_m
    sea = ~land & ~lake
    nbrs = _hex_neighbors(layout)
    touches_sea = np.zeros(shape, dtype=bool)
    sea_nb_count = np.zeros(shape)
    for nb, valid in nbrs:
        s = np.where(valid, sea.ravel()[nb.ravel()].reshape(shape), False)
        touches_sea |= s
        sea_nb_count += s
    coast = land & touches_sea
    sampling = _sampling(layout)

    # -- shelf width: distance from the shore to water deeper than 200 m -------------------
    deep = sea & (sea_level_m - z > 200.0)
    shelf_km = (
        ndimage.distance_transform_edt(~deep, sampling=sampling)
        if deep.any()
        else (np.full(shape, 1000.0))
    )
    # -- sea surface temperature next to each coast hex: mean of its sea neighbours --------
    t = np.asarray(layers["temperature_mean_c"], dtype=np.float64)
    t_sea = np.where(sea, t, 0.0)
    sst = np.zeros(shape)
    for nb, valid in nbrs:
        sst += np.where(valid, t_sea.ravel()[nb.ravel()].reshape(shape), 0.0)
    sst = np.where(sea_nb_count > 0, sst / np.maximum(sea_nb_count, 1), t)

    # -- fetch in 16 directions and along the wind: how much open water faces the shore ----
    hx, hy = layout.centers()
    idx = np.nonzero(coast.ravel())[0]
    cx, cy = hx.ravel()[idx], hy.ravel()[idx]
    step = layout.spacing_km
    max_km = 2500.0
    n_steps = int(max_km / step)
    angles = np.arange(16) * (2 * math.pi / 16)
    fetch = np.zeros((idx.size, 16))
    sea_flat = sea.ravel()
    width, height = layout.width, layout.height
    for k, a in enumerate(angles):
        dx, dy = math.cos(a) * step, math.sin(a) * step
        open_ = np.ones(idx.size, dtype=bool)
        dist = np.zeros(idx.size)
        for s in range(1, n_steps + 1):
            col, row = layout.world_to_offset(cx + dx * s, cy + dy * s)
            col, row = np.asarray(col), np.asarray(row)
            inside = (col >= 0) & (col < width) & (row >= 0) & (row < height)
            wet = np.where(
                inside,
                sea_flat[np.clip(row, 0, height - 1) * width + np.clip(col, 0, width - 1)],
                True,  # beyond the window: assume the water goes on (no fake shore)
            )
            open_ &= wet
            dist = np.where(open_, s * step, dist)
            if not open_.any():
                break
        fetch[:, k] = dist
    u = np.asarray(layers["wind_u_ms"], dtype=np.float64).ravel()[idx]
    v = np.asarray(layers["wind_v_ms"], dtype=np.float64).ravel()[idx]
    upwind = (np.degrees(np.arctan2(-v, -u)) % 360.0) / 22.5  # direction waves come from
    k0 = np.floor(upwind).astype(int) % 16
    w = upwind - np.floor(upwind)
    fetch_wind = (1 - w) * fetch[np.arange(idx.size), k0] + w * fetch[
        np.arange(idx.size), (k0 + 1) % 16
    ]

    def norm(f):
        return np.clip(np.log1p(f / 50.0) / math.log1p(1500.0 / 50.0), 0.0, 1.0)

    lat = np.abs(np.asarray(layers["latitude_deg"], dtype=np.float64)).ravel()[idx]
    storm = np.clip(
        0.4
        + 0.6 * np.exp(-(((lat - 52.0) / 14.0) ** 2))
        + 0.25 * np.exp(-(((lat - 17.0) / 8.0) ** 2))
        - 0.25 * np.exp(-((lat / 6.0) ** 2)),
        0.1,
        1.0,
    )
    exposure_c = storm * (0.45 * norm(fetch.mean(axis=1)) + 0.55 * norm(fetch_wind))
    openness = (fetch > 800.0).mean(axis=1)

    # -- tides: wide shelves and funnels amplify, small enclosed seas are almost tideless --
    sea_labels, n_seas = ndimage.label(sea)
    sea_area = (
        np.bincount(sea_labels.ravel(), minlength=n_seas + 1) * layout.hex_area_km2
        if n_seas
        else np.zeros(1)
    )
    near_sea = np.zeros(shape, dtype=np.int64)
    for nb, valid in nbrs:
        lab = np.where(valid, sea_labels.ravel()[nb.ravel()].reshape(shape), 0)
        near_sea = np.maximum(near_sea, lab)
    body = sea_area[near_sea.ravel()[idx]]
    shelf_c = shelf_km.ravel()[idx]
    shelf_f = np.clip(shelf_c / 350.0, 0.0, 1.0)
    tide_c = (0.5 + 4.0 * shelf_f * (0.3 + 0.7 * (1.0 - openness))) * np.where(
        body < 3.0e6, 0.25, 1.0
    )

    # -- what the land is like right behind the shore --------------------------------------
    h = np.maximum(z - sea_level_m, 0.0)
    inland_max = ndimage.maximum_filter(np.where(land, h, 0.0), size=5)  # ~2 hexes
    relief = np.asarray(hist_layers["local_relief_m"], dtype=np.float64)
    slope = np.asarray(layers["slope_m_per_km"], dtype=np.float64)
    rock = np.asarray(hist_layers["rock_resistance"], dtype=np.float64)
    rock_type = np.asarray(hist_layers["rock_type"])
    sediment = np.asarray(layers["sediment_thickness_m"], dtype=np.float64)
    precip = np.asarray(layers["precipitation_mm_yr"], dtype=np.float64)
    humid = np.asarray(layers["humidity_index"], dtype=np.float64)
    amp = np.asarray(layers["temperature_seasonal_c"], dtype=np.float64)
    glacial = np.asarray(layers["glacial_potential"], dtype=np.float64)
    latf = np.abs(np.asarray(layers["latitude_deg"], dtype=np.float64))
    # the last glacial maximum: cooler by 6 + 0.1 * |lat| degrees; wet mountain coasts
    # glaciate first (Chile, Norway), lowlands only when cold
    lgm_summer = t + amp - (6.0 + 0.1 * latf)
    threshold = 4.0 + 3.0 * np.clip((precip - 800.0) / 1500.0, 0.0, 1.0) + 0.003 * inland_max
    lgm_ice = land & (lgm_summer < threshold) & (precip > 250.0)
    boundary = np.asarray(hist_layers["boundary_class"])
    active = ndimage.maximum_filter((boundary == 1) | (boundary == 2), size=5) > 0
    volcano = np.asarray(hist_layers["volcano_id"]) >= 0
    volcanic_rock = rock_type == 4
    # river water and load reaching the sea at this hex (a mouth); a river mouth shapes
    # only its own stretch of coast
    load = np.asarray(layers["sediment_delivery_m3_yr"], dtype=np.float64)
    flow = np.asarray(layers["coastal_discharge_m3_s"], dtype=np.float64)

    # ridges parallel to the shore (Dalmatian type): direction of the nearest range axis
    parallel = np.zeros(shape, dtype=bool)
    if ridges:
        pts, tan = [], []
        for r in ridges:
            p = np.asarray(r["points_km"], dtype=np.float64)
            if len(p) < 3 or r.get("rank") == "spur":
                continue
            d = np.gradient(p, axis=0)
            d /= np.maximum(np.hypot(d[:, 0], d[:, 1]), 1e-9)[:, None]
            pts.append(p)
            tan.append(d)
        if pts:
            tree = cKDTree(np.vstack(pts))
            tangents = np.vstack(tan)
            dist, near = tree.query(np.column_stack([cx, cy]), distance_upper_bound=160.0)
            ok = np.isfinite(dist)
            gy_, gx_ = np.gradient(ndimage.gaussian_filter(land.astype(float), 1.5))
            nx_, ny_ = gx_.ravel()[idx], gy_.ravel()[idx]
            nn = np.maximum(np.hypot(nx_, ny_), 1e-9)
            along = np.abs(
                tangents[np.where(ok, near, 0), 0] * nx_ / nn
                + tangents[np.where(ok, near, 0), 1] * ny_ / nn
            )
            par = np.zeros(idx.size, dtype=bool)
            par[ok] = along[ok] < 0.4  # the ridge runs along the shore, not into it
            flat = parallel.reshape(-1)
            flat[idx] = par

    def at(a):
        return np.asarray(a).ravel()[idx]

    exposure = exposure_c
    tide = tide_c
    t_c, sst_c, hum = at(t), at(sst), at(humid)
    hmax, rel, slp, rk = at(inland_max), at(relief), at(slope), at(rock)
    sed, ld, fl = at(sediment), at(load), at(flow)
    hgt = at(h)
    steep = (slp > 12.0) | (hmax > 900.0) | (rel > 600.0)
    hilly = (hmax > 150.0) | (rel > 150.0)
    low = (hmax < 120.0) & (slp < 3.0)
    warm_sea = sst_c > 22.0
    muddy = (ld > 2.0e6) | (sed > 150.0)
    out = np.full(idx.size, _CODE["beach"], dtype=np.uint8)

    assigned = np.zeros(idx.size, dtype=bool)

    def put(mask, name):  # the first matching cause wins: specific before generic
        out[mask & ~assigned] = _CODE[name]
        assigned[mask] = True

    put(at(glacial) > 0.55, "ice")
    lgm = at(lgm_ice)
    put(lgm & ((hmax > 600.0) | (rel > 350.0)), "fjord")
    put(lgm & ~((hmax > 600.0) | (rel > 350.0)) & (rk > 0.6), "skerry")
    put((at(volcano) | at(volcanic_rock)) & (steep | (hmax > 500.0)), "volcanic")
    river = fl > RIVER_MOUTH_M3_S  # a large river: its mouth is a coast type of its own
    # delta vs estuary: sediment supply against the wave and tide energy that removes it
    delta_power = np.log10(np.maximum(ld, 1.0)) - 6.3 - 1.0 * exposure - 0.15 * tide
    put(river & (delta_power > 0.0), "delta")
    put(river & low & (tide < 0.9), "liman")
    put(river, "estuary")
    # a hot desert shore is a sabkha even with reefs offshore (Persian Gulf)
    put((hum < 0.2) & (t_c > 18.0) & low & (exposure < 0.6), "sabkha")
    sheltered = exposure < 0.3
    put(
        warm_sea & (t_c > 20.0) & (hum > 0.6) & (hmax < 300.0) & sheltered & (muddy | (tide > 1.5)),
        "mangrove",
    )
    put(warm_sea & (at(latf) < 32.0) & ~muddy & (ld < 5.0e5), "coral_reef")
    put((tide > 3.2) & low, "tidal_flat")
    put(at(active) & steep, "fault_scarp")
    put(at(parallel) & hilly & ~low, "dalmatian")
    put(hilly & ~steep & (rk >= 0.3), "ria")  # drowned valleys, exposed or not (Galicia)
    raised = (hgt > 40.0) | (rel > 100.0)
    put(
        (steep & (rk > 0.3)) | (raised & (exposure > 0.3)) | (hilly & (exposure > 0.45)),
        "cliff",
    )
    put(low & (at(shelf_km) > 60.0) & (tide < 3.0) & (exposure > 0.25) & (hgt < 80.0),
        "barrier_lagoon")  # fmt: skip

    def grid(values, fill, dtype):
        a = np.full(shape[0] * shape[1], fill, dtype=dtype)
        a[idx] = values
        return a.reshape(shape)

    layers_out = {
        "coast_type": grid(out, 0, np.uint8),
        "wave_exposure": grid(exposure.astype(np.float32), np.nan, np.float32),
        "tidal_range_m": grid(tide.astype(np.float32), np.nan, np.float32),
        "shelf_width_km": np.where(coast, shelf_km, np.nan).astype(np.float32),
        "sea_surface_temperature_c": np.where(coast, sst, np.nan).astype(np.float32),
        "lgm_ice_mask": lgm_ice.astype(np.bool_),
    }
    counts = np.bincount(out, minlength=len(COAST_TYPES))
    total = max(int(idx.size), 1)
    summary = {
        "coast_hexes": int(idx.size),
        "coast_length_km": round(float(idx.size * layout.spacing_km), 1),
        "shares": {
            COAST_TYPES[i]: round(float(counts[i]) / total, 4)
            for i in range(1, len(COAST_TYPES))
            if counts[i]
        },  # fmt: skip
        "rocky_share": round(float(np.isin(out, [_CODE[n] for n in ROCKY]).mean()), 4)
        if idx.size
        else 0.0,
        "sandy_share": round(float(np.isin(out, [_CODE[n] for n in SANDY]).mean()), 4)
        if idx.size
        else 0.0,
        "mean_tidal_range_m": round(float(tide.mean()), 3) if idx.size else 0.0,
        "mean_wave_exposure": round(float(exposure.mean()), 3) if idx.size else 0.0,
    }
    return layers_out, summary
