"""Aeolian sand and dust, coastal sediment budget (TZ 5.3 sec. 11-12, gate G6, P1).

**Sand** is owned by the soils: the mobile inventory of a hex (kg/m^2) comes from its
parent material and texture (``aeolian_sand`` 2 m, alluvium / lake beds 0.5 m, other soils
``sand_frac x 5 cm``) plus the beach store of sandy coasts; nothing appears from nowhere.

Friction velocity and Bagnold-type flux ``[ADAPT, E01]``::

    u* = kappa U_eff / ln(z / z0),        z0 = 0.0005 + 0.1 veg  (m)
    u*t = 0.22 (1 + 1.5 wet) (1 + 5 veg^2)   (moisture, vegetation, snow raise the threshold)
    q = C rho_a / g * max(u*^2 - u*t^2, 0) * u*      kg m^-1 s^-1,   C = 2.6

``U_eff = 2.2 U_season`` stands for the windy part of the season (a share ``f = 0.05`` of
the time). Each season's flux leaves a hex across its downwind faces; a donor never gives
more than it holds; sand blown into the sea is exported, into a lake trapped (ledger).
2000 years in 20 steps. Dune fields: >= 0.5 m of accumulated sand on sparse vegetation;
the type from the sand thickness and the directional variability of the seasonal winds
(barchans, transverse, linear, star; ``coastal`` next to a beach source).

**Dust / loess**: silt of dry bare soils is emitted in proportion to the saltation flux
(sandblasting), carried downwind hex by hex and deposited with a probability that rises
with vegetation and wetness; the deposit is the loess rate (mm/kyr). Sand and dust are
separate fractions (no double deposition).

**Coast** (one-line idea ``[ADAPT, E03]``): on sandy coasts the longshore transport
``Q = K H^2.5 sin(2 theta)`` (``H`` from the wave exposure, ``theta`` the wind-wave angle to
the coast normal) moves sand to the downdrift coast hex; the shoreline moves by
``(Q_in + Q_river - Q_out - aeolian_loss) / (D_active L)``. Rocky coasts (cliff, fjord,
fault scarp...) retreat by ``wave power / rock resistance`` instead - never the sand law.
"""

from __future__ import annotations

import numpy as np

from .. import hexgrid

KAPPA = 0.4
RHO_AIR = 1.2
G = 9.81
C_BAGNOLD = 2.6
RHO_SAND = 1600.0
RHO_LOESS = 1450.0
WINDY_SHARE = 0.05
GUST = 2.2
SEASON_S = 365.25 * 86400.0 / 4.0
SANDY_COASTS = ("beach", "barrier_lagoon", "delta", "estuary", "tidal_flat", "sabkha", "liman")
ROCKY_COASTS = ("cliff", "fault_scarp", "fjord", "skerry", "dalmatian", "ria", "volcanic")


def friction_velocity(u10, z0):
    return KAPPA * np.asarray(u10) / np.log(10.0 / np.asarray(z0))


def threshold_u_star(wet, veg):
    """``[H]`` dry fine sand 0.22 m/s; moisture and vegetation cover raise it."""
    v = np.asarray(veg, dtype=np.float64)
    return 0.22 * (1.0 + 1.5 * np.asarray(wet)) * (1.0 + 5.0 * v * v)


def sand_flux(u_star, u_star_t):
    """Bagnold-type saltation flux, kg per metre of width per second."""
    us = np.asarray(u_star, dtype=np.float64)
    ut = np.asarray(u_star_t, dtype=np.float64)
    return C_BAGNOLD * RHO_AIR / G * np.maximum(us * us - ut * ut, 0.0) * us


def mobile_sand(pm_class, sand_frac, soil_depth, beach_m=None):
    """Mobile sand inventory, kg/m^2, from the soils (+ beach store, m of sand)."""
    from ..stages.environment import PM

    pm = np.asarray(pm_class)
    depth = np.where(pm == PM["aeolian_sand"], 2.0,
                     np.where(np.isin(pm, [PM["alluvium"], PM["marine_lacustrine"]]), 0.5,
                              np.nan_to_num(sand_frac) * 0.05))  # fmt: skip
    depth = np.where(np.nan_to_num(soil_depth) > 0, depth, 0.0) + (
        0.0 if beach_m is None else np.asarray(beach_m)
    )
    return depth * RHO_SAND


def aeolian(nbr, spacing_m, u_seasons, v_seasons, inventory, wet, veg, frozen, ocean, lake,
            years=2000.0, steps=20, silt_frac=None, beach_source=None) -> dict:  # fmt: skip
    """Sand redistribution and dust deposition; every kg is booked."""
    n = inventory.size
    face = spacing_m / np.sqrt(3.0)
    area = np.sqrt(3.0) / 2.0 * spacing_m**2
    dirs = hexgrid.direction_unit_vectors()  # (6, 2), +x east, +y south (map)
    veg = np.clip(np.asarray(veg, dtype=np.float64), 0.0, 1.0)
    wet = np.clip(np.asarray(wet, dtype=np.float64), 0.0, 1.0)
    ut = threshold_u_star(wet, veg)
    z0 = 0.0005 + 0.1 * veg
    land = ~ocean & ~lake
    m = np.where(land, np.asarray(inventory, dtype=np.float64) * area, 0.0)  # kg per hex
    m0 = float(m.sum())
    led = {"initial_kg": m0, "exported_to_sea_kg": 0.0, "trapped_in_lakes_kg": 0.0,
           "beach_loss_kg": 0.0}  # fmt: skip
    q_vec = np.zeros((n, 2))
    q_abs = np.zeros(n)
    rates = []
    for s in range(len(u_seasons)):
        u = np.asarray(u_seasons[s], dtype=np.float64)
        v = np.asarray(v_seasons[s], dtype=np.float64)
        sp = np.hypot(u, v)
        us = friction_velocity(GUST * sp, z0)
        # plants shelter the bed (drag partition, Raupach-type): only the bare share moves
        bare = (1.0 - veg) ** 2
        q = sand_flux(us, ut) * bare * (1.0 - np.asarray(frozen[s], dtype=np.float64))
        q = np.where(land, q, 0.0)
        e = np.stack([np.where(sp > 0, u / np.maximum(sp, 1e-9), 0.0),
                      np.where(sp > 0, v / np.maximum(sp, 1e-9), 0.0)], axis=1)  # fmt: skip
        share = np.maximum(e @ dirs.T, 0.0)  # (n, 6)
        share = share / np.maximum(share.sum(axis=1, keepdims=True), 1e-12)
        # kg per year leaving the hex across each face: q (kg/m/s) x face x windy seconds
        out = q[:, None] * face * share * WINDY_SHARE * SEASON_S
        rates.append(out)
        q_vec += (q * WINDY_SHARE)[:, None] * e
        q_abs += q * WINDY_SHARE
    dt = years / steps
    src = np.repeat(np.arange(n), 6)
    dst = nbr.ravel()
    for _ in range(steps):
        flow = sum(r for r in rates).ravel() * dt  # kg per step per (hex, face)
        valid = dst >= 0
        flow = np.where(valid, flow, 0.0)
        out_tot = np.bincount(src, weights=flow, minlength=n)
        scale = np.where(out_tot > m, m / np.maximum(out_tot, 1e-30), 1.0)
        flow = flow * scale[src]
        # (a donor gives at most what it holds; round-off below zero is clipped)
        m = np.maximum(m - np.bincount(src, weights=flow, minlength=n), 0.0)
        recv = np.bincount(np.where(valid, dst, 0), weights=flow, minlength=n)
        to_sea = float(recv[ocean].sum())
        to_lake = float(recv[lake].sum())
        led["exported_to_sea_kg"] += to_sea
        led["trapped_in_lakes_kg"] += to_lake
        recv[ocean] = 0.0
        recv[lake] = 0.0
        m = m + recv
    if beach_source is not None:
        b = np.asarray(beach_source, dtype=np.float64) * RHO_SAND * area
        led["beach_loss_kg"] = float(np.maximum(b - m, 0.0)[b > 0].sum())
    led["final_kg"] = float(m.sum())
    led["residual_kg"] = led["final_kg"] - (m0 - led["exported_to_sea_kg"]
                                            - led["trapped_in_lakes_kg"])  # fmt: skip
    thick = m / area / RHO_SAND
    accum = np.maximum(thick - np.asarray(inventory) / RHO_SAND, 0.0)
    variability = 1.0 - np.hypot(q_vec[:, 0], q_vec[:, 1]) / np.maximum(q_abs, 1e-12)
    variability = np.where(q_abs > 0, variability, 0.0)
    dune = np.zeros(n, dtype=np.uint8)  # 0 none 1 barchan 2 transverse 3 linear 4 star 5 coastal
    field = land & (thick >= 0.5) & (veg < 0.35) & (q_abs > 0)
    dune[field & (variability < 0.3) & (thick < 2.0)] = 1
    dune[field & (variability < 0.3) & (thick >= 2.0)] = 2
    dune[field & (variability >= 0.3) & (variability < 0.6)] = 3
    dune[field & (variability >= 0.6)] = 4
    if beach_source is not None:
        near = np.zeros(n, dtype=bool)
        bs = np.asarray(beach_source) > 0
        for k in range(6):
            j = nbr[:, k]
            near |= (j >= 0) & bs[np.maximum(j, 0)]
        dune[(bs | near) & land & (thick >= 0.3) & (veg < 0.6)] = 5
    out = {"sand_thickness_m": thick, "sand_accumulation_m": accum,
           "sand_flux_kg_m_yr": q_abs * 365.25 * 86400.0, "wind_variability": variability,
           "dune_type": dune, "ledger": led}  # fmt: skip
    if silt_frac is not None:
        out.update(_dust(nbr, rates, np.asarray(silt_frac, dtype=np.float64), veg, wet, land,
                         ocean, area, np.asarray(u_seasons), np.asarray(v_seasons)))  # fmt: skip
    return out


def _dust(nbr, rates, silt_frac, veg, wet, land, ocean, area, u_s, v_s):
    """Dust emission by sandblasting, downwind deposition -> loess rate (mm/kyr)."""
    n = silt_frac.size
    sand_out = sum(r.sum(axis=1) for r in rates)  # kg/yr of saltation leaving each hex
    silt = np.clip(np.nan_to_num(silt_frac), 0.0, 1.0)
    emis = 0.05 * sand_out * silt * (veg < 0.4) * (wet < 0.6) * land
    u = u_s.mean(axis=0)
    v = v_s.mean(axis=0)
    dirs = hexgrid.direction_unit_vectors()
    e = np.stack([u, v], axis=1)
    k_best = np.argmax(e @ dirs.T, axis=1)
    nxt = nbr[np.arange(n), k_best]
    p_dep = np.clip(0.06 + 0.3 * veg + 0.3 * wet, 0.0, 0.95)
    p_dep = np.where(ocean, 0.1, p_dep)
    airborne = emis.copy()
    dep = np.zeros(n)
    exported = 0.0
    for _ in range(40):
        d = airborne * p_dep
        dep += d
        airborne -= d
        go = nxt >= 0
        exported += float(airborne[~go].sum())
        moved = np.bincount(nxt[go], weights=airborne[go], minlength=n)
        airborne = moved
    exported += float(airborne.sum())
    dust_to_sea = float(dep[ocean].sum())
    dep_land = np.where(land, dep, 0.0)
    loess = dep_land / area / RHO_LOESS * 1000.0 * 1000.0  # mm per kyr
    ledger = {
        "emitted_kg_yr": float(emis.sum()),
        "deposited_land_kg_yr": float(dep_land.sum()),
        "deposited_sea_kg_yr": dust_to_sea,
        "exported_kg_yr": exported,
        "residual_kg_yr": float(emis.sum()) - float(dep.sum()) - exported,
    }
    return {
        "dust_emission_kg_yr": emis,
        "dust_deposition_kg_m2_yr": dep / area,
        "loess_mm_kyr": loess,
        "dust_ledger": ledger,
    }


def coastal(nbr, spacing_m, coast_type_names, coast_type, exposure, ocean, u, v, river_sed,
            rock_resistance, d_active=10.0, k_ls=2.0e5) -> dict:  # fmt: skip
    """Shoreline change rate (m/yr, + seaward) of coast hexes; sand budget ledger."""
    n = coast_type.size
    names = np.asarray(coast_type_names, dtype=object)
    kind = names[np.asarray(coast_type, dtype=np.int64)]
    coast = np.asarray(coast_type) > 0
    sandy = coast & np.isin(kind, SANDY_COASTS)
    rocky = coast & np.isin(kind, ROCKY_COASTS)
    dirs = hexgrid.direction_unit_vectors()
    # coast normal: towards the open water neighbours
    normal = np.zeros((n, 2))
    for k in range(6):
        j = nbr[:, k]
        sea = (j >= 0) & ocean[np.maximum(j, 0)]
        normal += sea[:, None] * dirs[k]
    nn = np.hypot(normal[:, 0], normal[:, 1])
    normal = np.where(nn[:, None] > 0, normal / np.maximum(nn, 1e-9)[:, None], 0.0)
    tang = np.stack([-normal[:, 1], normal[:, 0]], axis=1)
    w = np.stack([u, v], axis=1)
    sp = np.hypot(u, v)
    wd = np.where(sp[:, None] > 0, w / np.maximum(sp, 1e-9)[:, None], 0.0)
    # waves come from the sea toward the coast: onshore component of the wind
    cos_t = np.clip(-(wd * normal).sum(axis=1), -1.0, 1.0)
    onshore = cos_t > 0
    theta = np.arccos(np.clip(cos_t, 0.0, 1.0))
    sign = np.sign((wd * tang).sum(axis=1))
    h_wave = 0.5 + 3.0 * np.clip(np.nan_to_num(exposure), 0.0, 1.0)
    q = np.where(sandy & onshore, k_ls * h_wave**2.5 * np.sin(2.0 * theta), 0.0)
    # downdrift neighbour: the coast hex most in the drift direction
    best = np.full(n, -1)
    best_dot = np.full(n, 0.2)
    for k in range(6):
        j = nbr[:, k]
        ok = (j >= 0) & coast[np.maximum(j, 0)] & ~ocean[np.maximum(j, 0)]
        dot = (dirs[k] * tang).sum(axis=1) * sign
        better = ok & (dot > best_dot)
        best = np.where(better, j, best)
        best_dot = np.where(better, dot, best_dot)
    q_in = np.zeros(n)
    to = best >= 0
    np.add.at(q_in, best[to], q[to])
    to_sea = float(q[~to].sum())  # drift that leaves the coast (spit end, headland)
    river = np.where(coast, 0.3 * np.nan_to_num(river_sed), 0.0)  # sand share of the load
    length = spacing_m
    dy_sandy = (q_in + river - q) / (d_active * length)
    power = h_wave**2  # relative wave power
    dy_rocky = -0.05 * power / np.maximum(np.asarray(rock_resistance), 0.1)
    dy = np.where(sandy, dy_sandy, np.where(rocky, dy_rocky, 0.0))
    ledger = {"longshore_total_m3_yr": float(q.sum()), "received_m3_yr": float(q_in.sum()),
              "river_sand_m3_yr": float(river[sandy].sum()), "lost_offshore_m3_yr": to_sea,
              "residual_m3_yr": float(q.sum()) - float(q_in.sum()) - to_sea}  # fmt: skip
    beach = np.where(sandy & (dy >= 0), 1.0, np.where(sandy, 0.3, 0.0))  # m of mobile sand
    return {
        "shoreline_change_m_yr": np.where(coast, dy, np.nan),
        "longshore_m3_yr": q,
        "beach_sand_m": beach,
        "sandy": sandy,
        "rocky": rocky,
        "ledger": ledger,
    }
