"""Volcanic deposits change the relief and the drainage (TZ 5.3 sec. 14, gate G7).

``VolcanicEvent -> deposits (lava, tephra, PDC, lahar) -> MorphologyDelta -> apply (atomic)
-> drainage / lakes / ocean connectivity rebuilt``.

* A :class:`MorphologyDelta` is one event's deposit of one material: hexes, ``dz`` (m, the
  deposited volume over the hex area, i.e. the change of the hex-mean surface), bulk
  density, mass and volume. Lahar deposits are *redistributed sediment*, booked apart from
  the erupted magma; gases never become height.
* :func:`apply_deltas` is transactional: every delta is validated first (finite, positive,
  inside the grid, an event id + material applied once only - a repeated id is rejected,
  T79); if anything is invalid nothing is applied and the input surface is returned
  unchanged. The ledger compares ``sum(dz * A)`` with the deposit volumes exactly.
* :func:`rebuild` runs the v4 hydrology (priority-flood routing, lakes, discharge) on the
  surface **before and after** with identical settings, and the hex ocean connectivity, so
  every difference is caused by the deposits: new dam lakes, lost lakes, re-routed
  receivers, new land, a sea cut off.

The v4 project layers are **not overwritten** (a regeneration reusing them would deposit
twice): the post-volcanic surface and drainage are a separate snapshot that the biomes and
the map read (``post_volcanic_*`` layers of v5.1).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

#: Bulk density of the deposits, kg/m^3 [H].
BULK_DENSITY = {"lava": 2500.0, "tephra": 1000.0, "pdc": 1400.0, "lahar_sediment": 1800.0}


@dataclass(frozen=True)
class MorphologyDelta:
    event_id: str
    material: str
    hex_index: np.ndarray
    dz_m: np.ndarray
    volume_m3: float
    mass_kg: float
    years_bp: float
    method: str = "hex_mean_deposit"
    quality: str = "CALIBRATED_APPROXIMATION"


def delta_from_volumes(event_id, material, hex_index, volume_m3, hex_area_m2, years_bp=0.0,
                       method="hex_mean_deposit") -> MorphologyDelta:  # fmt: skip
    idx = np.asarray(hex_index, dtype=np.int64)
    vol = np.asarray(volume_m3, dtype=np.float64)
    keep = vol > 0
    idx, vol = idx[keep], vol[keep]
    rho = BULK_DENSITY[material]
    return MorphologyDelta(
        event_id=str(event_id), material=material, hex_index=idx, dz_m=vol / hex_area_m2,
        volume_m3=float(vol.sum()), mass_kg=float(vol.sum() * rho), years_bp=float(years_bp),
        method=method,
    )  # fmt: skip


class DeltaError(ValueError):
    pass


def apply_deltas(z, deltas, hex_area_m2: float, applied=None):
    """New surface + ledger; ``z`` is never modified. Raises nothing: invalid -> unchanged."""
    z0 = np.asarray(z, dtype=np.float64)
    seen = set(applied or ())
    rejected, ok = [], []
    try:
        for d in deltas:
            key = (d.event_id, d.material)
            if key in seen:
                rejected.append({"event_id": d.event_id, "material": d.material,
                                 "reason": "already applied"})  # fmt: skip
                continue
            if d.hex_index.size and (d.hex_index.min() < 0 or d.hex_index.max() >= z0.size):
                raise DeltaError(f"{d.event_id}: hex outside the grid")
            if not np.isfinite(d.dz_m).all() or (d.dz_m < 0).any():
                raise DeltaError(f"{d.event_id}: dz must be finite and >= 0")
            seen.add(key)
            ok.append(d)
    except DeltaError as exc:
        return z0.copy(), {"committed": False, "error": str(exc), "applied": [],
                           "rejected": rejected}  # fmt: skip
    z1 = z0.copy()
    by_mat: dict = {}
    for d in ok:
        np.add.at(z1, d.hex_index, d.dz_m)
        m = by_mat.setdefault(d.material, {"volume_m3": 0.0, "mass_kg": 0.0, "events": 0})
        m["volume_m3"] += d.volume_m3
        m["mass_kg"] += d.mass_kg
        m["events"] += 1
    total = sum(m["volume_m3"] for m in by_mat.values())
    placed = float((z1 - z0).sum() * hex_area_m2)
    report = {
        "committed": True,
        "applied": sorted({d.event_id for d in ok}),
        "applied_keys": sorted({f"{d.event_id}:{d.material}" for d in ok}),
        "rejected": rejected,
        "by_material": by_mat,
        "volume_m3": total,
        "placed_m3": placed,
        "volume_residual_m3": placed - total,
    }
    return z1, report


def hex_ocean(z, sea_level, graph, keep_ocean=None):
    """Hexes at or below sea level connected to the border or to the given ocean."""
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components

    wet = np.asarray(z) <= sea_level
    n = wet.size
    nbr = graph.nbr
    src = np.repeat(np.arange(n), nbr.shape[1])
    dst = nbr.ravel()
    ok = (dst >= 0) & wet[src] & wet[np.maximum(dst, 0)]
    g = coo_matrix((np.ones(ok.sum()), (src[ok], dst[ok])), shape=(n, n))
    _nc, lab = connected_components(g, directed=False)
    seeds = wet & graph.border
    if keep_ocean is not None:
        seeds |= wet & np.asarray(keep_ocean, bool)
    good = np.unique(lab[seeds])
    return wet & np.isin(lab, good)


def rebuild(project, z_post, report_cb=None) -> dict:
    """Drainage, lakes and ocean connectivity before and after the deposits (same settings)."""
    from ..stages.hydrology import hex_graph, hydrology, routing_choice

    cfg = project.config
    H = project.hydro.layers
    lay = project.layout
    sea_level = float(project.hydro.sea_level_m)
    graph = hex_graph(lay)
    z_pre = H["elevation_m"].astype(np.float64).ravel()
    area = lay.hex_area_km2 * 1e6
    ocean_pre = H["ocean_mask"].ravel().astype(bool)
    ocean_post = hex_ocean(z_post, sea_level, graph, keep_ocean=ocean_pre)

    def run(z, ocean):
        return hydrology(
            z, ocean, graph, area, np.nan_to_num(H["runoff_mm_yr"].ravel().astype(np.float64)),
            np.nan_to_num(H["precipitation_mm_yr"].ravel().astype(np.float64)),
            np.nan_to_num(H["potential_evapotranspiration_mm_yr"].ravel().astype(np.float64))
            * 1.1,
            np.nan_to_num(H["external_inflow_m3_s"].ravel().astype(np.float64)),
            cfg.hydrology.min_lake_depth_m, cfg.hydrology.channel_loss,
            routing_choice(cfg.seed, z.size, cfg.hydrology.routing, lay), None,
        )  # fmt: skip

    pre = run(z_pre, ocean_pre)
    post = run(np.asarray(z_post, dtype=np.float64), ocean_post)
    lake_pre = pre["lake_id"] >= 0
    lake_post = post["lake_id"] >= 0
    changed = pre["receiver"] != post["receiver"]
    q_pre, q_post = pre["discharge_m3_s"], post["discharge_m3_s"]
    river = q_post > 1.0
    return {
        "elevation": np.asarray(z_post, dtype=np.float64),
        "receiver": post["receiver"],
        "lake_id": post["lake_id"],
        "discharge_m3_s": q_post,
        "ocean": ocean_post,
        "new_lake": lake_post & ~lake_pre,
        "lost_lake": lake_pre & ~lake_post,
        "drainage_changed": changed,
        "new_land": ocean_pre & ~ocean_post,
        "new_sea": ocean_post & ~ocean_pre,
        "river": river,
        "summary": {
            "receivers_changed": int(changed.sum()),
            "new_lake_hexes": int((lake_post & ~lake_pre).sum()),
            "lost_lake_hexes": int((lake_pre & ~lake_post).sum()),
            "new_land_hexes": int((ocean_pre & ~ocean_post).sum()),
            "discharge_changed_hexes": int(
                (np.abs(q_post - q_pre) > 0.05 * np.maximum(q_pre, 1.0)).sum()
            ),  # fmt: skip
            "rebuild": "whole_window_priority_flood (not a local halo)",
            "invalidated": [
                "hydrology_network",
                "lakes",
                "ocean_connectivity",
                "coasts",
                "soils_local",
                "biomes (v5.0 reads the snapshot)",
            ],
        },
    }
