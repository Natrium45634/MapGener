"""The world on a sphere (orthographic view), for the «Глобус» window. Qt-free.

The model world is flat with a conditional latitude (``climate.latitude_center_deg`` and
km per degree). On the globe, y maps to that latitude and x to longitude at the same
km per degree of the equator; whatever the window does not cover is drawn as «unknown»
(dark), so the globe shows truthfully how much of a planet the map is.
"""

from __future__ import annotations

import numpy as np

KM_PER_DEG = 111.2
SPACE = (12, 14, 22)
UNKNOWN = (36, 44, 58)


def lonlat_of_world(project):
    """Functions world km -> (lon, lat) and back, from the project's latitude model."""
    from .stages.climate import resolve_latitude

    lat_model = resolve_latitude(project.config.climate, project.geometry.simulation_bounds)
    b = project.geometry.viewport_bounds
    x_mid = 0.5 * (b.xmin + b.xmax)
    kpd = lat_model.km_per_deg

    def to_world(lon, lat):
        # the flat map keeps km per degree of longitude constant (like the model's
        # conditional latitude), so the window is a lon/lat rectangle on the globe
        y = lat_model.y_center_km - (lat - lat_model.center_deg) * kpd
        x = x_mid + lon * kpd
        return x, y

    return to_world, lat_model


def render_globe(project, hex_rgb: np.ndarray, size: int = 560, lon0: float = 0.0,
                 lat0: float = 20.0, zoom: float = 1.0) -> np.ndarray:  # fmt: skip
    """RGB image of the globe seen from above (lon0, lat0); light from the upper left."""
    to_world, _lat = lonlat_of_world(project)
    lay = project.layout
    n = int(size)
    r_px = 0.47 * n * zoom
    ys, xs = np.mgrid[0:n, 0:n].astype(np.float64)
    u = (xs + 0.5 - n / 2) / r_px
    v = -(ys + 0.5 - n / 2) / r_px
    rr = u * u + v * v
    inside = rr <= 1.0
    img = np.empty((n, n, 3), dtype=np.uint8)
    img[:] = SPACE
    if not inside.any():
        return img
    uu, vv = u[inside], v[inside]
    ww = np.sqrt(1.0 - uu * uu - vv * vv)
    # rotate the view vector back to the globe frame
    la0, lo0 = np.radians(lat0), np.radians(lon0)
    x1 = uu
    y1 = vv * np.cos(la0) + ww * np.sin(la0)
    z1 = -vv * np.sin(la0) + ww * np.cos(la0)
    lat = np.degrees(np.arcsin(np.clip(y1, -1, 1)))
    lon = np.degrees(np.arctan2(x1, z1)) + np.degrees(lo0)
    lon = (lon + 180.0) % 360.0 - 180.0
    wx, wy = to_world(lon, lat)
    col, row = lay.world_to_offset(wx, wy)
    ok = lay.inside(col, row)
    colors = np.empty((uu.size, 3), dtype=np.float64)
    colors[:] = UNKNOWN
    colors[ok] = hex_rgb[row[ok], col[ok]]
    # graticule every 30 degrees
    grid = (np.abs(((lat + 90) % 30) - 15) > 14.6) | (np.abs(((lon + 180) % 30) - 15) > 14.6)
    colors[grid] = 0.75 * colors[grid] + 0.25 * 235
    # Lambert shading, light from the upper left and the viewer
    light = np.array([-0.45, 0.55, 0.70])
    light /= np.linalg.norm(light)
    shade = 0.35 + 0.65 * np.clip(uu * light[0] + vv * light[1] + ww * light[2], 0, 1)
    colors *= shade[:, None]
    img[inside] = np.clip(colors, 0, 255).astype(np.uint8)
    # thin atmosphere rim
    rim = (rr > 1.0) & (rr < 1.0 + 0.05 / max(zoom, 0.3))
    img[rim] = (70, 110, 170)
    return img


def window_extent(project) -> dict:
    """How much of the planet the map covers (degrees and share of the surface)."""
    b = project.geometry.viewport_bounds
    _to, lat = lonlat_of_world(project)
    lat_top = lat.at(b.ymin)
    lat_bottom = lat.at(b.ymax)
    lon_span = min(360.0, b.width / lat.km_per_deg)
    share = (np.sin(np.radians(lat_top)) - np.sin(np.radians(lat_bottom))) / 2.0 * lon_span / 360.0
    return {"lat_north": float(lat_top), "lat_south": float(lat_bottom),
            "lon_span": float(lon_span), "surface_share": float(abs(share))}  # fmt: skip
