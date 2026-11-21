"""Marine basement and sea-floor morphology composed onto the v3 surface (TZ 4.6, 4-5).

Composition (``composition_mode`` per province):

* **continental crust** (continentality >= 0, rifts in gaps): ``keep`` - the v3 surface;
* **sea floor** (seaward of the crust edge): ``replace`` the v2 baseline part of the v3
  surface (``base_component_m``) and the v3 instantaneous trench (``trench_relief_m``) by

      margin profile(u; shelf break, slope foot, rise, basement)
      + burial * (ridge-axis morphology + abyssal fabric + fracture-zone troughs)
      + trench / outer rise + oceanic plateaus + pelagic sediment surface

  and keep the rest of the v3 surface as an additive residual (history uplift, sediment,
  volcanic cones and arc platforms). The basement is the Parsons-Sclater depth of the crust
  age; the v2 Gaussian ridge swell is *not* added on top of it (no double count).

Every subgrid feature (axial valley, fracture-zone trough, trench axis) enters a hex as its
mean over the hex footprint (a Gaussian widened by the hex width, cross-section area kept),
so a 15 km axial valley is a 15 km valley whatever the hex size, and the hex value never
pretends to resolve it.
"""

from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree

from ..noise import FractalNoise
from ..rng import Stream
from .margins import ACTIVE, PASSIVE, TRANSFORM, classify_margin, margin_profile, margin_weights
from .ocean_age import flowline_age, parsons_sclater_depth

#: offsets of the 7-point footprint sample (centre + 6 at 0.55 of the hex radius)
_RING = [(0.0, 0.0)] + [(np.cos(a), np.sin(a)) for a in np.arange(6) * np.pi / 3.0]


def footprint(d_km, sigma_km, spacing_km):
    """Mean over a hex of a Gaussian line feature of width ``sigma`` at distance ``d``."""
    s_eff = np.sqrt(sigma_km**2 + spacing_km**2 / 10.0)
    return (sigma_km / s_eff) * np.exp(-0.5 * (np.asarray(d_km) / s_eff) ** 2)


def _neighbours(layout):
    from ..stages.raster import neighbor_index

    n = layout.width * layout.height
    nbr = np.full((n, 6), -1, dtype=np.int64)
    for d in range(6):
        idx, valid = neighbor_index(layout, d)
        nbr[:, d] = np.where(valid.ravel(), idx.ravel(), -1)
    return nbr


def seamount_types(on_floor, crust_age_ma, volcano_age_ma, summit_now_m, ridge_depth_m, blend):
    """0 none, 1 seamount, 2 guyot, 3 volcanic island (datum = the planation level).

    A guyot is a volcano whose top reached the sea surface when it was built (it was planed
    there) and has since sunk with its plate: subsidence = PS(age now) - PS(age at birth).
    An old volcano that never reached the surface stays a pointed seamount.
    """
    on_floor = np.asarray(on_floor, dtype=bool)
    out = np.zeros(on_floor.shape, dtype=np.uint8)
    if not on_floor.any():
        return out
    a_now = np.nan_to_num(np.asarray(crust_age_ma, dtype=np.float64), nan=100.0)
    vage = np.nan_to_num(np.asarray(volcano_age_ma, dtype=np.float64))
    a_birth = np.clip(a_now - vage, 0.0, None)
    sink = parsons_sclater_depth(a_now, ridge_depth_m, blend_ma=blend) - parsons_sclater_depth(
        a_birth, ridge_depth_m, blend_ma=blend
    )
    top_now = np.asarray(summit_now_m, dtype=np.float64)
    top_then = top_now + sink
    out[on_floor] = 1
    out[on_floor & (vage > 3.0) & (top_then > 0.0) & (top_now < -200.0)] = 2
    out[on_floor & (top_now > 0.0)] = 3
    return out


def compose_sea_floor(
    layers: dict,
    layout,
    t_epoch: float,
    pieces,
    relief_model,
    params,
    seed: str,
    volcano_rows: list[dict] | None = None,
    age_scale: float = 1.0,
    transform_points=None,
) -> dict:
    """Sea floor of the epoch. Returns flat-free (rows, cols) layers and a summary."""
    shape = (layout.height, layout.width)
    n = shape[0] * shape[1]

    def flat(name, dtype=np.float64):
        return np.asarray(layers[name]).ravel().astype(dtype)

    spacing = layout.spacing_km
    hx, hy = (a.ravel() for a in layout.centers())
    z3 = flat("elevation_m")
    cf = flat("continental_fraction")
    cont = flat("continentality_km")
    new = flat("new_crust", bool)
    plate = flat("plate_id_epoch", np.int64)
    mx, my = flat("material_x_km"), flat("material_y_km")
    nbr = _neighbours(layout)

    # -- 1. where the sea floor is: seaward of the continental crust edge ----------------
    gap_rift = new & (cf > 0.05)
    c_eff = cont.copy()
    missing = ~np.isfinite(c_eff)
    if missing.any():
        # new crust: the distance to where the continent edge was, in the plate's own
        # frame - a world function, so every window and resolution gets the same value
        c_gap = relief_model.continents.continentality(mx[missing], my[missing])
        c_eff[missing] = np.minimum(c_gap, -1.0)
    cont_hex = c_eff >= 0.0
    marine = (c_eff < 0.0) & ~gap_rift
    m = np.nonzero(marine)[0]

    # -- 2. crust age, ridge piece, isochron direction ---------------------------------------
    age = np.full(n, np.nan)
    piece = np.full(n, -1, dtype=np.int32)
    tx = np.zeros(n)
    ty = np.zeros(n)
    half = np.full(n, 25.0)
    if m.size:
        fa = flowline_age(pieces, mx[m], my[m], plate[m])
        age[m] = np.where(new[m], flat("crust_age_ma")[m], age_scale * fa.age_ma + t_epoch)
        piece[m] = fa.piece
        tx[m], ty[m] = fa.tx, fa.ty
        half[m] = np.where(np.isfinite(fa.half_rate), fa.half_rate, 25.0)
    age = np.clip(age, 0.0, None)
    blend = tuple(params.age_depth_transition_ma)
    depth = parsons_sclater_depth(
        np.nan_to_num(age, nan=150.0), params.ridge_depth_m, blend_ma=blend
    )
    z_bas = -depth

    # -- 3. ridge axis: rift valley on slow ridges, axial high on fast ones -------------------
    slow = np.clip((22.0 - half) / 14.0, 0.0, 1.0)
    fast = np.clip((half - 35.0) / 25.0, 0.0, 1.0)
    ax_d = flat("spreading_axis_distance_km")
    axial = -1300.0 * slow * footprint(ax_d, 9.0, spacing) + 350.0 * fast * footprint(
        ax_d, 7.0, spacing
    )
    axial = np.where(np.nan_to_num(age, nan=99.0) < 15.0, axial, 0.0)

    # -- 4. sediment: pelagic rain on the crust since it formed; drapes the fabric ----------
    sed3 = np.clip(flat("sediment_thickness_m"), 0.0, None)
    pelagic = np.clip(params.pelagic_rate_m_per_ma * np.nan_to_num(age, nan=0.0), 0.0, 900.0)
    if not params.sediment_cover:
        pelagic[:] = 0.0

    # -- 5. abyssal fabric: elongated along the isochrons, frozen into the plate --------------
    stream = Stream(seed, "geomorph_v4_6/sea_floor")
    # ridge-flank fault blocks and abyssal-hill provinces at the scales a hex can show; the
    # 1-10 km hills themselves are sub-hex and enter only the local relief of the hex
    fabric_noise = FractalNoise(stream.child("fabric"), [220.0, 110.0, 55.0], [1.0, 0.6, 0.35])
    amp = params.abyssal_relief_m * (0.6 + 1.1 * slow) * np.exp(-(pelagic + sed3) / 600.0)
    fabric = np.zeros(n)
    lo = np.full(n, np.inf)
    hi = np.full(n, -np.inf)
    if m.size:
        r = 0.55 * layout.radius
        for ox, oy in _RING:
            px, py = mx[m] + r * ox, my[m] + r * oy
            ut = px * tx[m] + py * ty[m]
            us = px * ty[m] - py * tx[m]
            v = fabric_noise(ut / 3.5, us)
            fabric[m] += v / len(_RING)
            lo[m] = np.minimum(lo[m], v)
            hi[m] = np.maximum(hi[m], v)
        fabric[m] *= amp[m]
    fabric_range = np.where(np.isfinite(hi - lo), (hi - lo) * amp, 0.0)

    # -- 6. fracture zones: where the ridge piece of the crust changes on one plate ---------
    fz = np.zeros(n, dtype=bool)
    fz_jump = np.zeros(n)
    if m.size and params.transform_fracture_history:
        for k in range(6):
            nb = nbr[:, k]
            ok = (nb >= 0) & marine
            ok[ok] &= marine[nb[ok]]
            i = np.nonzero(ok)[0]
            j = nb[i]
            # mark the older side only: a fracture zone is one line, not a two-hex band
            jump = np.nan_to_num(age[i]) - np.nan_to_num(age[j])
            step = (
                (plate[i] == plate[j])
                & (piece[i] != piece[j])
                & (piece[i] >= 0)
                & (piece[j] >= 0)
                & (jump > 1.5)
            )
            fz[i[step]] = True
            fz_jump[i[step]] = np.maximum(fz_jump[i[step]], jump[step])
    # trough of the fracture zone: deeper where the offset (the age jump) is larger
    fz_amp = params.fracture_zone_depth_m * np.clip(fz_jump / 5.0, 0.35, 1.0)
    fz_trough = np.where(fz, -fz_amp * footprint(0.0, 10.0, spacing), 0.0)

    # -- 7. trenches and outer rise on the subducting oceanic plate -------------------------
    side = flat("trench_side", np.int64)
    tdist = flat("trench_distance_km")
    rate = flat("trench_rate_mm_yr")
    trench = np.zeros(n)
    trench_axis = np.zeros(n, dtype=bool)
    outer_rise = np.zeros(n, dtype=bool)
    sub = marine & (side == 1) & (tdist < 600.0)
    over = marine & (side == -1) & (tdist < 400.0)
    if params.subduction_trenches and sub.any():
        along = FractalNoise(stream.child("trench_along"), [900.0, 400.0], [1.0, 0.5])
        f_rate = np.clip(np.sqrt(np.maximum(rate, 0.0) / 50.0), 0.6, 1.3)
        f_age = 0.55 + 0.45 * np.clip(np.nan_to_num(age, nan=80.0) / 100.0, 0.0, 1.5)
        d_axis = params.trench_depth_m * f_age * f_rate * np.exp(0.25 * along(hx, hy))
        # local values only (no search for the nearest subducting hex, which would make
        # the result depend on the window): the inner wall uses the crust of its own side
        d_here = d_axis
        bulge = (250.0 + 250.0 * np.clip(np.nan_to_num(age, nan=80.0) / 100.0, 0.0, 1.0)) * f_rate
        trench = np.where(
            sub,
            -d_here * footprint(tdist, 22.0, spacing)
            + bulge * footprint(tdist - 120.0, 40.0, spacing),
            0.0,
        )
        trench = np.where(over, -0.85 * d_here * footprint(tdist, 28.0, spacing), trench)
        trench_axis = (sub | over) & (trench < -0.45 * d_here)
        outer_rise = sub & (tdist > 60.0) & (tdist < 220.0) & (trench > 60.0)

    # -- 8. continental margins ----------------------------------------------------------------
    # distance to transform plate boundaries of the world skeleton, in the plate frame
    if transform_points is not None and len(transform_points):
        tf_dist, _ = cKDTree(np.asarray(transform_points)).query(np.column_stack([mx, my]))
    else:
        tf_dist = np.full(n, 1.0e4)
    weights = margin_weights(tdist, side, tf_dist)
    var_noise = FractalNoise(stream.child("margin_width"), [1600.0, 800.0], [1.0, 0.4])
    variation = np.exp(0.35 * var_noise(mx, my))
    shelf_edge = relief_model.params.shelf_edge_m
    z_break = shelf_edge + 0.3 * 300.0 * relief_model.broad(mx, my)
    u = -c_eff
    # broad swells and depressions of the old floor (mantle dynamic topography), fixed in the
    # mantle frame, not on the plate: +-300 m over thousands of km
    swell_noise = FractalNoise(stream.child("swell"), [3200.0, 1600.0], [1.0, 0.45])
    swell = 300.0 * swell_noise(hx, hy)
    z_floor_base = z_bas + axial + fz_trough + fabric + swell
    profile, ws, wr = margin_profile(u, z_break, z_bas, weights, variation)
    # basement structure appears beyond the rise; the slope and rise bury it
    burial = np.clip((u - 0.6 * (ws + wr)) / np.maximum(0.4 * (ws + wr), 1e-6), 0.0, 1.0)
    burial = burial * burial * (3.0 - 2.0 * burial)
    sed_surface = 0.6 * pelagic * burial
    floor = profile + burial * (z_floor_base - z_bas) + trench + sed_surface

    # -- 9. oceanic plateaus at the start of hotspot tracks -----------------------------------
    plateau = np.zeros(n)
    plateau_rows = []
    if params.oceanic_plateaus and volcano_rows:
        chains: dict = {}
        for v in volcano_rows:
            hid = v.get("hotspot_id")
            if hid is None:
                continue
            if hid not in chains or v["birth_ma"] < chains[hid]["birth_ma"]:
                chains[hid] = v
        for hid in sorted(chains, key=str):
            v = chains[hid]
            rs = stream.child(f"plateau/{hid}")
            if rs.random() > 0.45:
                continue
            u_ = rs.random(4)
            radius = 250.0 + 350.0 * u_[0]
            height = 1500.0 + 1200.0 * u_[1]
            aspect = 1.0 + 0.8 * u_[2]
            ang = np.pi * u_[3]
            dx, dy = hx - v["x_km"], hy - v["y_km"]
            a = (dx * np.cos(ang) + dy * np.sin(ang)) / (radius * aspect)
            b = (-dx * np.sin(ang) + dy * np.cos(ang)) / radius
            shape_ = np.exp(-((a * a + b * b) ** 2))  # flat top, steep edges
            here = marine & ~cont_hex & (shape_ > 0.02)
            if not here.any():
                continue
            plateau = np.where(here, np.maximum(plateau, height * shape_), plateau)
            plateau_rows.append(
                {
                    "id": f"plateau:{hid}",
                    "hotspot_id": hid,
                    "x_km": round(float(v["x_km"]), 2),
                    "y_km": round(float(v["y_km"]), 2),
                    "radius_km": round(radius, 1),
                    "aspect": round(aspect, 3),
                    "height_m": round(height, 1),
                    "age_ma": v.get("age_ma"),
                    "provenance": "plume head at the start of the hotspot track [А]",
                }
            )
    floor = floor + plateau

    # -- 10. compose: replace the baseline + trench parts on the sea floor --------------------
    residual = z3 - flat("base_component_m") - flat("trench_relief_m")
    z_new = np.where(marine, floor + residual, z3)
    delta = z_new - z3

    # -- 11. seamounts and guyots: planed at the sea level, then subsided with the crust -------
    vid = flat("volcano_id", np.int64)
    vage = flat("volcanic_age_ma")
    summit = flat("volcano_summit_m")
    on_floor = marine & (vid >= 0) & np.isfinite(vage)
    smt = seamount_types(
        on_floor, age, vage, np.nan_to_num(summit, nan=-1e4) + delta, params.ridge_depth_m, blend
    )

    margin = np.where(marine & (u < ws + wr), classify_margin(weights), 0).astype(np.uint8)

    def grid(a, dtype=np.float32):
        return np.ascontiguousarray(np.asarray(a).reshape(shape).astype(dtype))

    out = {
        "marine_floor_mask": grid(marine, np.bool_),
        "oceanic_crust_age_ma": grid(np.where(marine, age, np.nan)),
        "basement_depth_m": grid(np.where(marine, depth, np.nan)),
        "bathymetry_delta_m": grid(delta),
        "marine_sediment_thickness_m": grid(np.where(marine, pelagic + sed3, 0.0)),
        "abyssal_fabric_m": grid(np.where(marine, fabric * burial, 0.0)),
        "ridge_piece_id": grid(piece, np.int32),
        "fracture_zone_mask": grid(fz, np.bool_),
        "fracture_zone_offset_ma": grid(fz_jump),
        "trench_m": grid(trench),
        "trench_axis_mask": grid(trench_axis, np.bool_),
        "outer_rise_mask": grid(outer_rise, np.bool_),
        "margin_type": grid(margin, np.uint8),
        "crust_edge_distance_km": grid(c_eff),
        "oceanic_plateau_m": grid(plateau),
        "seamount_type": grid(smt, np.uint8),
        "spreading_half_rate_mm_yr": grid(np.where(marine & (piece >= 0), half, np.nan)),
        "marine_local_relief_m": grid(np.where(marine, fabric_range * burial, 0.0)),
        "slope_width_km": grid(np.where(marine, ws, 0.0)),
        "rise_width_km": grid(np.where(marine, wr, 0.0)),
    }
    w_pas, w_act, w_tr = weights
    ms = marine & (u < ws + wr)
    summary = {
        "marine_hexes": int(marine.sum()),
        "ridge_pieces": int(pieces.count),
        "fracture_zone_hexes": int(fz.sum()),
        "trench_axis_hexes": int(trench_axis.sum()),
        "outer_rise_hexes": int(outer_rise.sum()),
        "plateaus": plateau_rows,
        "margin_hexes": {
            "passive": int((ms & (margin == PASSIVE)).sum()),
            "active": int((ms & (margin == ACTIVE)).sum()),
            "transform": int((ms & (margin == TRANSFORM)).sum()),
        },
        "age_ma_percentiles": (
            [round(float(v), 2) for v in np.percentile(age[m], [5, 25, 50, 75, 95])]
            if m.size
            else []
        ),
        "seamounts": int((smt == 1).sum()),
        "guyots": int((smt == 2).sum()),
        "age_estimation_mode": "approx",
        "max_trench_depth_m": float(-(z_new[trench_axis]).min()) if trench_axis.any() else None,
        "mean_slope_width_km": float(np.mean(ws[ms])) if ms.any() else None,
        "mean_passive_weight": float(np.mean(w_pas[ms])) if ms.any() else None,
        "mean_active_weight": float(np.mean(w_act[ms])) if ms.any() else None,
        "mean_transform_weight": float(np.mean(w_tr[ms])) if ms.any() else None,
    }
    return {"surface": z_new.reshape(shape), "layers": out, "summary": summary}
