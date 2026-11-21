"""v4 stage ``landscape_evolution``: rivers and slopes reshape the v3 surface.

Coupled loop, ``steps`` times over ``duration_ma``::

    climate (hex scale, current surface) -> routing, lakes, discharge -> uplift
      -> stream-power incision (implicit) -> hillslope diffusion -> sediment transport
      -> new surface

* **Stream power**, ``E = K * A_eff^m * S`` (n = 1) solved implicitly along the receiver
  graph from the outlets upwards (Braun & Willett 2013): unconditionally stable for any
  step. ``A_eff = Q / q_ref`` is the discharge-equivalent drainage area, so wet basins cut
  faster than arid ones. ``K`` falls with rock resistance; loose sediment is soft.
  Incision never cuts below the receiver and never acts inside a lake.
* **Hillslope**: linear diffusion on the hex graph (Laplacian ``2/(3 L^2) * sum(z_j - z)``)
  with explicit sub-steps under the stability limit. At 10-100 km cells the coefficient is
  an *effective* large-scale value, not a field-measured soil creep rate.
* **Sediment**: what is eroded travels down the receiver graph; where the flux exceeds the
  transport capacity ``Qc = k_t * A_eff * S`` it is deposited (never above the receiver, so
  no new pits), lakes fill, the rest reaches the sea - deposited near the mouth up to a
  shelf depth - or leaves the window. Every cubic metre is booked in the budget.

Elevations m, time yr internally (Ma in the interface), volumes m^3.
"""

from __future__ import annotations

import numpy as np

from ..geomorph.surface_water import flood as ocean_flood
from .climate import SECONDS_PER_YEAR, hex_climate
from .hydrology import Graph, hydrology, routing_choice

_SHELF_FILL_DEPTH_M = 30.0


def stream_power_rate(k, flow_ratio, m, slope, n=1.0, threshold=0.0):
    """Detachment-limited incision ``E = max(0, K (Q/Q0)^m S^n - E_c)`` (manual 7.1), m/yr.

    The normalised form: ``flow_ratio`` is dimensionless, so ``k`` and ``threshold`` are in
    m/yr. ``evolve`` uses the same law with ``A_eff = Q / q_ref`` (m^2) and ``n = 1``.
    """
    flow = np.maximum(np.asarray(flow_ratio, dtype=np.float64), 0.0)
    s = np.maximum(np.asarray(slope, dtype=np.float64), 0.0)
    return np.maximum(k * flow**m * s**n - threshold, 0.0)


def implicit_incision(z_up, receiver, order, f, threshold_m, skip):
    """One implicit step of ``dz/dt = -(K A^m (z - z_r) / L - E_c)`` from the outlets up.

    ``f = K A^m dt / L`` per cell, ``threshold_m = E_c dt``. Solving with the receiver's new
    height (Braun & Willett 2013) is stable for any step and never cuts below the receiver;
    a cell erodes only where ``f (z - z_r) > E_c dt``, giving
    ``z_new = (z + f z_r + E_c dt) / (1 + f)``. Cells in ``skip`` (outlets, lakes) keep
    their height. ``order``: cells downstream first.
    """
    zl = np.asarray(z_up, dtype=np.float64).tolist()
    zu = list(zl)
    fl = np.asarray(f, dtype=np.float64).tolist()
    rcl = np.asarray(receiver).tolist()
    skip_l = np.asarray(skip).tolist()
    ec = float(threshold_m)
    for c in np.asarray(order).tolist():
        if skip_l[c]:
            continue
        zr = zl[rcl[c]]
        drop = zu[c] - zr
        if drop > 0.0 and fl[c] * drop > ec:
            zl[c] = (zu[c] + fl[c] * zr + ec) / (1.0 + fl[c])
    return np.asarray(zl)


def _laplacian(z, graph: Graph):
    valid = graph.nbr >= 0
    zn = np.where(valid, z[np.maximum(graph.nbr, 0)], z[:, None])
    length = graph.dist[:, 0]
    return (2.0 / 3.0) * (zn - z[:, None]).sum(axis=1) / (length * length)


def evolve(
    config,
    layout,
    graph: Graph,
    z0,
    sea_level_m,
    rock_resistance,
    sediment0,
    uplift_mm_yr,
    world_climate,
    external_inflow_m3_s,
    report=None,
    edge_outlet=None,
    ocean_sources=None,
):
    # ``ocean_sources``: hexes of the world ocean (v4.6). The sea is the below-datum ground
    # connected to them; a closed depression below the datum stays land (lake or dry).
    # ``None`` keeps the legacy rule ``z <= sea level``.
    params = config.erosion
    cp = config.climate
    n = z0.size
    z = np.asarray(z0, dtype=np.float64).ravel().copy()
    sediment = np.maximum(np.asarray(sediment0, dtype=np.float64).ravel().copy(), 0.0)
    resistance = np.asarray(rock_resistance, dtype=np.float64).ravel()
    area = layout.hex_area_km2 * 1e6
    length = layout.spacing_km * 1000.0
    total_years = params.duration_ma * 1e6 if params.enabled else 0.0
    steps = params.steps if total_years > 0 else 0
    dt = total_years / steps if steps else 0.0
    uplift = (
        np.asarray(uplift_mm_yr, dtype=np.float64).ravel() / 1000.0
        if params.continue_uplift
        else np.zeros(n)
    )
    q_ref = params.reference_runoff_mm_yr / 1000.0  # m/yr
    threshold = params.incision_threshold_m_ma * 1e-6  # E_c, m/yr
    erosion_total = np.zeros(n)
    deposition_total = np.zeros(n)
    flux_last = np.zeros(n)
    delivery = np.zeros(n)  # m^3 delivered to the sea through each mouth cell
    budget = {
        "fluvial_eroded_m3": 0.0,
        "hillslope_moved_m3": 0.0,
        "hillslope_to_sea_m3": 0.0,
        "deposited_land_m3": 0.0,
        "deposited_lakes_m3": 0.0,
        "deposited_sea_m3": 0.0,
        "offshore_m3": 0.0,
        "left_window_m3": 0.0,
        "uplift_m3": 0.0,
    }
    steps_log = []
    climate = None
    choice = routing_choice(config.seed, n, config.hydrology.routing, layout)
    for step in range(steps):
        if report:
            report(step / max(steps, 1))
        grid_shape = (layout.height, layout.width)
        if ocean_sources is None:
            sea = z <= sea_level_m
        else:
            sea = ocean_flood(z <= sea_level_m, ocean_sources, graph.nbr)
        if climate is None or params.update_climate:
            climate = hex_climate(
                cp,
                world_climate,
                layout,
                z.reshape(grid_shape),
                sea_level_m,
                ocean=sea.reshape(grid_shape),
            )
        hyd = hydrology(
            z,
            sea,
            graph,
            area,
            climate["runoff_mm_yr"].ravel(),
            climate["precipitation_mm_yr"].ravel(),
            climate["potential_evapotranspiration_mm_yr"].ravel() * 1.1,
            external_inflow_m3_s,
            config.hydrology.min_lake_depth_m,
            config.hydrology.channel_loss,
            choice,
            edge_outlet,
        )
        routing = hyd["routing"]
        rc = hyd["receiver"]
        order = routing.order
        q = hyd["discharge_m3_s"]
        a_eff = q * SECONDS_PER_YEAR / q_ref
        soft = np.where(sediment > 20.0, 0.22, resistance)
        k = params.k_stream * np.exp(3.0 * (0.5 - soft))
        in_lake = (hyd["flood_depth_m"] > 0.01) & ~routing.outlet

        # -- uplift, then implicit stream power from the outlets upwards --------------------
        z_up = z + uplift * dt
        budget["uplift_m3"] += float((uplift * dt * area)[~sea].sum())
        f = k * dt * np.power(np.maximum(a_eff, 0.0), params.m_exponent) / length
        z_new = implicit_incision(z_up, rc, order, f, threshold * dt, routing.outlet | in_lake)
        rcl = rc.tolist()
        lake_l = in_lake.tolist()

        # -- hillslope diffusion on land ------------------------------------------------------
        land = ~sea
        if params.diffusivity_m2_yr > 0 and dt > 0:
            sub = max(1, int(np.ceil(dt * params.diffusivity_m2_yr / (0.2 * length * length))))
            h = dt / sub
            before = z_new.copy()
            for _ in range(sub):
                dz = params.diffusivity_m2_yr * h * _laplacian(z_new, graph)
                z_new = np.where(land, z_new + dz, z_new)
            moved = (z_new - before)[land] * area
            budget["hillslope_moved_m3"] += float(np.abs(moved).sum() / 2.0)
            budget["hillslope_to_sea_m3"] += float(-moved.sum())

        # -- sediment: route eroded material, deposit where capacity is exceeded --------------
        eroded = np.where(land, np.maximum(z_up - z_new, 0.0), 0.0)
        gained = np.where(land, np.maximum(z_new - z_up, 0.0), 0.0)  # hillslope deposits
        budget["fluvial_eroded_m3"] += float((eroded * area).sum())
        from_sed = np.minimum(eroded, sediment)
        sediment = sediment - from_sed + gained
        slope = np.maximum(z_new - z_new[rc], 0.0) / length
        capacity = params.k_transport * a_eff * slope * dt  # m^3 over the step
        flux = eroded * area
        dep = np.zeros(n)
        flood = routing.routing_m
        fx = flux.tolist()
        cap = capacity.tolist()
        zl = z_new.tolist()
        dl = dep.tolist()
        seal = sea.tolist()
        lake_room = np.maximum(flood - z_new, 0.0).tolist()
        for c in order[::-1].tolist():
            r = rcl[c]
            amount = fx[c]
            if amount <= 0.0:
                continue
            if seal[c]:
                room = max(0.0, (sea_level_m - _SHELF_FILL_DEPTH_M - zl[c]) * area)
                put = min(amount, room)
                dl[c] += put / area
                budget["deposited_sea_m3"] += put
                rest = amount - put
                if r != c:
                    fx[r] += rest
                else:
                    budget["offshore_m3"] += rest
                continue
            if r == c:  # window edge
                budget["left_window_m3"] += amount
                continue
            if lake_l[c]:
                put = min(amount, lake_room[c] * area)
                budget["deposited_lakes_m3"] += put
            else:
                excess = amount - cap[c]
                room = max(0.0, (zl[r] - zl[c] - dl[c]) * area) if not seal[r] else 0.0
                put = min(max(excess, 0.0), room)
                budget["deposited_land_m3"] += put
            dl[c] += put / area
            rest = amount - put
            if seal[r]:
                delivery[c] += rest
            fx[r] += rest
        dep = np.asarray(dl)
        flux_last = np.asarray(fx)
        z = z_new + dep
        sediment = sediment + dep
        erosion_total += eroded
        deposition_total += dep + gained
        steps_log.append(
            {
                "step": step + 1,
                "time_ma": round((step + 1) * dt / 1e6, 4),
                "eroded_m3": float((eroded * area).sum()),
                "deposited_m3": float((dep * area).sum()),
                "max_incision_m": float(eroded.max()),
                "lakes": len(hyd["lakes"]),
            }
        )
    budget["closure_error_m3"] = (
        budget["fluvial_eroded_m3"]
        - budget["deposited_land_m3"]
        - budget["deposited_lakes_m3"]
        - budget["deposited_sea_m3"]
        - budget["offshore_m3"]
        - budget["left_window_m3"]
    )
    return {
        "elevation": z,
        "erosion_total_m": erosion_total,
        "deposition_total_m": deposition_total,
        "sediment_thickness_m": sediment,
        "sediment_flux_m3_yr": flux_last / dt if dt > 0 else flux_last,
        "sediment_delivery_m3_yr": delivery / total_years if total_years > 0 else delivery,
        "budget": budget,
        "steps": steps_log,
        "dt_years": dt,
    }
