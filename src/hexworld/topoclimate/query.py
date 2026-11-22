"""Point queries of the v4.7 result (TZ 4.7, 11.1 ``query_environment`` / ``sample_transect``).

A query never invents a position inside the hex: terrain units are strata without a
membership map. The value at a height ``z`` is interpolated *in height* between the dry
units of the hex (``tu_height_interpolated``); a single dry unit gives its own values
(``tu_representative``); a hex without dry land gives NoData (``unavailable``).
"""

from __future__ import annotations

import numpy as np

from .thermal import BELTS

#: transect / point columns added by v4.7 (units in the names)
COLUMNS = (
    "t_mean_c",
    "t_month_c",
    "p_mm_yr",
    "pet_mm_yr",
    "soil_depth_m",
    "swe_max_mm",
    "gsl_days",
    "gst_c",
    "thermal_margin",
    "belt",
    "quality",
)


def _unit_columns(topo, idx, month):
    U = topo.units
    t = U["temperature_c"][:, idx].astype(np.float64)
    return {
        "t_mean_c": t.mean(axis=0),
        "t_month_c": t[month] if month is not None else np.full(idx.size, np.nan),
        "p_mm_yr": U["precip_mm"][:, idx].sum(axis=0, dtype=np.float64),
        "pet_mm_yr": U["pet_mm"][:, idx].sum(axis=0, dtype=np.float64),
        "soil_depth_m": U["soil_depth_m"][idx].astype(np.float64),
        "swe_max_mm": U["swe_mm"][:, idx].max(axis=0).astype(np.float64),
        "gsl_days": U["gsl_t_days"][idx].astype(np.float64),
        "gst_c": U["gst_t_c"][idx].astype(np.float64),
        "thermal_margin": U["thermal_margin"][idx].astype(np.float64),
    }


def query_environment(topo, hex_flat: int, z_m: float, month: int | None = None) -> dict:
    """Local conditions at height ``z_m`` of one hex (``row * width + col``)."""
    U = topo.units
    idx = topo.units_of_hex(int(hex_flat))
    idx = idx[U["unit_surface_type"][idx] == 0]
    out = {k: None for k in COLUMNS}
    if idx.size == 0:
        out["quality"] = "unavailable"
        return out
    z = U["z_surface_m"][idx].astype(np.float64)
    order = np.argsort(z, kind="stable")
    idx, z = idx[order], z[order]
    cols = _unit_columns(topo, idx, month)
    if idx.size == 1:
        k = 0
        for name, v in cols.items():
            out[name] = None if not np.isfinite(v[k]) else round(float(v[k]), 3)
        out["belt"] = BELTS[int(U["thermal_belt"][idx[0]])]
        out["quality"] = "tu_representative"
        return out
    zc = float(np.clip(z_m, z[0], z[-1]))
    for name, v in cols.items():
        ok = np.isfinite(v)
        if not ok.any():
            continue
        out[name] = round(float(np.interp(zc, z[ok], v[ok])), 3)
    nearest = idx[int(np.argmin(np.abs(z - zc)))]
    # the belt follows the interpolated thermal margin where it can, else the nearest unit
    out["belt"] = BELTS[int(U["thermal_belt"][nearest])]
    out["quality"] = "tu_height_interpolated" if z[0] <= z_m <= z[-1] else "tu_height_clamped"
    return out
