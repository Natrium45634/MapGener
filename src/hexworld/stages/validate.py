"""Stage ``validate``: structural checks of a generated or loaded project."""

from __future__ import annotations

import numpy as np
from scipy import ndimage

from .. import hexgrid
from .raster import component_table, neighbor_index

TINY_FRAGMENT_HEXES = 3


def validate_project(project) -> dict:
    errors: list[str] = []
    warnings: list[str] = []
    checks: dict = {}
    layout = project.layout
    plate_id = project.layer("plate_id")
    n_plates = len(project.plates)

    # 1. every hex belongs to exactly one existing plate
    if plate_id.shape != (layout.height, layout.width):
        errors.append("plate_id shape does not match the hex grid")
    in_range = bool(plate_id.min() >= 0 and plate_id.max() < n_plates)
    checks["all_hexes_assigned"] = in_range
    if not in_range:
        errors.append("plate_id refers to a non-existent plate")

    # 2. connectivity within the window
    _labels, comp_plate, size, touches, largest = component_table(plate_id, layout)
    detached = ~touches & ~largest
    checks["plates_in_view"] = int(np.unique(plate_id).size)
    checks["components"] = int(size.size)
    checks["detached_components"] = int(detached.sum())
    checks["edge_split_components"] = int((touches & ~largest).sum())
    if detached.any():
        errors.append(
            f"{int(detached.sum())} plate fragment(s) are disconnected inside the window "
            f"(plates {sorted(set(comp_plate[detached].tolist()))[:8]})"
        )
    tiny = (size <= TINY_FRAGMENT_HEXES) & ~largest
    checks["tiny_fragments"] = int(tiny.sum())
    if tiny.sum() > max(3, 0.0005 * plate_id.size):
        errors.append(f"{int(tiny.sum())} tiny plate fragments - mass single-hex artefacts")
    elif tiny.any():
        warnings.append(
            f"{int(tiny.sum())} tiny fragment(s) at the window edge (plates continuing outside)"
        )

    # 3. boundary graph == plate adjacency
    edges = project.boundaries
    expected = 0
    flat = plate_id.ravel()
    for d in hexgrid.HALF_DIRECTIONS:
        nb, valid = neighbor_index(layout, d)
        expected += int(np.count_nonzero(valid.ravel() & (flat[nb.ravel()] != flat)))
    checks["boundary_edges"] = len(edges)
    if expected != len(edges):
        errors.append(f"boundary graph has {len(edges)} edges, adjacency implies {expected}")
    if len(edges):
        if (
            not layout.inside(edges.col_a, edges.row_a).all()
            or not layout.inside(edges.col_b, edges.row_b).all()
        ):
            errors.append("boundary edge refers to a hex outside the grid")
        else:
            pa = plate_id[edges.row_a, edges.col_a]
            pb = plate_id[edges.row_b, edges.col_b]
            if not (np.array_equal(pa, edges.plate_a) and np.array_equal(pb, edges.plate_b)):
                errors.append("boundary edge plates do not match plate_id")
            if not np.all(edges.plate_a < edges.plate_b):
                errors.append("boundary edges must satisfy plate_a < plate_b")
            dist = hexgrid.offset_distance(edges.col_a, edges.row_a, edges.col_b, edges.row_b)
            if not np.all(dist == 1):
                errors.append("boundary edge joins hexes that do not share an edge")
            norm = np.hypot(edges.normal_x, edges.normal_y)
            if not np.allclose(norm, 1.0, atol=1e-4):
                errors.append("boundary normals are not unit vectors")
            speed2 = edges.vrel_x.astype(np.float64) ** 2 + edges.vrel_y.astype(np.float64) ** 2
            comp2 = edges.vn.astype(np.float64) ** 2 + edges.vt.astype(np.float64) ** 2
            if not np.allclose(speed2, comp2, rtol=1e-3, atol=1e-2):
                errors.append("vn/vt do not decompose the relative velocity")

    # 4. numeric sanity of layers
    for name, arr in project.layers.items():
        if np.issubdtype(arr.dtype, np.floating) and not np.isfinite(arr).all():
            errors.append(f"layer {name} contains non-finite values")
    for name in ("uplift", "rifting", "subduction", "volcanism", "seismicity"):
        arr = project.layer(name)
        if arr.min() < 0.0 or arr.max() > 1.0:
            errors.append(f"layer {name} is outside [0, 1]")

    # 5. world-level connectivity on the reference raster (model area)
    ref = project.cache.get("reference")
    if ref is not None and ref.plate is not None:
        split = []
        edge = np.zeros(ref.plate.shape, dtype=bool)
        edge[0, :] = edge[-1, :] = edge[:, 0] = edge[:, -1] = True
        for pid in np.unique(ref.plate[ref.in_sim]):
            labels, count = ndimage.label(ref.plate == pid)
            if count < 2:
                continue
            sizes = np.bincount(labels.ravel(), minlength=count + 1)[1:]
            on_edge = np.zeros(count + 1, dtype=bool)
            on_edge[labels[edge]] = True
            # pieces cut off by the raster rectangle may join outside it: ignore them
            inner = ~on_edge[1:] & (sizes > 2)
            inner[np.argmax(sizes)] = False
            if inner.any():
                split.append(int(pid))
        checks["world_split_plates"] = len(split)
        if split:
            warnings.append(
                f"plates {split[:8]} are disconnected on the world reference raster "
                "(lower plates.warp_strength to avoid this)"
            )

    return {"ok": not errors, "errors": errors, "warnings": warnings, "checks": checks}


def validate_geo(project) -> dict:
    """Structural checks of the v2.0 data (continental blocks, relief, sea, land)."""
    from .sea import label_mask

    geo = project.geo
    layout = project.layout
    errors: list[str] = []
    warnings: list[str] = []
    checks: dict = {}
    shape = (layout.height, layout.width)
    for name, arr in geo.layers.items():
        if arr.shape != shape:
            errors.append(f"layer {name} has shape {arr.shape}, expected {shape}")
        if np.issubdtype(arr.dtype, np.floating) and not np.isfinite(arr).all():
            errors.append(f"layer {name} contains non-finite values")
    if errors:
        return {"ok": False, "errors": errors, "warnings": warnings, "checks": checks}

    elev = geo.layers["macro_elevation_m"]
    land = geo.layers["land_mask"]
    land_id = geo.layers["landmass_id"]
    block_id = geo.layers["continental_block_id"]
    fraction = geo.layers["continental_fraction"]
    shelf = geo.layers["shelf_mask"]

    # land is derived from elevation and sea level only
    if not np.array_equal(land, elev > np.float32(geo.sea_level_m)):
        errors.append("land_mask is not (macro_elevation_m > sea_level_m)")
    if not np.array_equal(land_id >= 0, land):
        errors.append("landmass_id must be >= 0 exactly on land hexes")
    if fraction.min() < 0.0 or fraction.max() > 1.0:
        errors.append("continental_fraction is outside [0, 1]")
    if not np.array_equal(block_id >= 0, fraction >= 0.5):
        errors.append("continental_block_id must cover exactly continental_fraction >= 0.5")
    if np.any(shelf & land):
        errors.append("shelf_mask overlaps land")
    if np.any(shelf & (fraction < 0.5)):
        errors.append("shelf_mask lies on oceanic crust")

    # ids are exactly the edge-connected components
    for name, ids, table in (
        ("landmass_id", land_id, geo.landmasses),
        ("continental_block_id", block_id, geo.blocks),
    ):
        relabelled, sizes = label_mask(ids >= 0, layout)
        if not np.array_equal(relabelled, ids):
            errors.append(f"{name} is not the set of edge-connected components")
        if len(table) != sizes.size:
            errors.append(f"{name}: {len(table)} table rows for {sizes.size} components")
    checks["landmasses"] = len(geo.landmasses)
    checks["continental_blocks"] = len(geo.blocks)

    # fringe: land that lives in one- or two-hex specks
    n_land = int(land.sum())
    if n_land:
        sizes = np.array([m["hexes"] for m in geo.landmasses])
        speck = int(sizes[sizes <= 2].sum())
        checks["speck_land_share"] = round(speck / n_land, 5)
        checks["speck_landmasses"] = int((sizes <= 2).sum())
        if speck / n_land > 0.02:
            warnings.append(
                f"{100 * speck / n_land:.1f} % of land hexes are 1-2 hex specks (coast fringe)"
            )
    checks["land_fraction_viewport"] = round(float(land.mean()), 5)
    checks["sea_level_m"] = geo.sea_level_m
    return {"ok": not errors, "errors": errors, "warnings": warnings, "checks": checks}


def validate_history(project) -> dict:
    """Structural and physical checks of the v3.0 data of the selected epoch."""
    from .sea import label_mask

    hist = project.hist
    layout = project.layout
    errors: list[str] = []
    warnings: list[str] = []
    checks: dict = {}
    shape = (layout.height, layout.width)
    for name, arr in hist.layers.items():
        if arr.shape != shape:
            errors.append(f"layer {name} has shape {arr.shape}, expected {shape}")
    if errors:
        return {"ok": False, "errors": errors, "warnings": warnings, "checks": checks}
    nan_ok = {"orogen_age_ma", "last_tectonic_ma", "volcanic_age_ma"}
    for name, arr in hist.layers.items():
        if np.issubdtype(arr.dtype, np.floating) and name not in nan_ok:
            if not np.isfinite(arr).all():
                errors.append(f"layer {name} contains non-finite values")
    elev = hist.layers["elevation_m"]
    land = hist.layers["land_mask"]
    ids = hist.layers["landmass_id"]
    cap = project.config.mountains.max_elevation_m
    if not np.array_equal(land, elev > np.float32(hist.sea_level_m)):
        errors.append("land_mask is not (elevation_m > sea_level_m)")
    if not np.array_equal(ids >= 0, land):
        errors.append("landmass_id must be >= 0 exactly on land hexes")
    if not np.array_equal(label_mask(land, layout)[0], ids):
        errors.append("landmass_id is not the set of edge-connected components")
    if float(elev.max()) > cap + 0.5:
        errors.append(f"elevation exceeds the absolute limit of {cap:g} m")
    for name in ("erosion_total_m", "sediment_thickness_m"):
        if float(hist.layers[name].min()) < 0.0:
            errors.append(f"{name} is negative")
    for name in ("volcanic_potential", "rock_resistance", "continental_fraction"):
        a = hist.layers[name]
        if float(a.min()) < 0.0 or float(a.max()) > 1.0:
            errors.append(f"{name} is outside [0, 1]")
    if project.geo is not None:
        expected = elev - project.geo.layers["macro_elevation_m"]
        if not np.allclose(hist.layers["delta_elevation_m"], expected, atol=0.01):
            errors.append("delta_elevation_m is not elevation_m - base_macro_elevation_m")
        if hist.sea_level_m != project.geo.sea_level_m:
            errors.append("v3 and v2 use different sea levels")
    n_plates = len(project.plates)
    owner = hist.layers["plate_id_epoch"]
    if owner.min() < 0 or owner.max() >= n_plates:
        errors.append("plate_id_epoch refers to a non-existent plate")
    times = [e["time_ma"] for e in hist.events]
    if times != sorted(times):
        errors.append("events are not ordered by time")
    if len({e["id"] for e in hist.events}) != len(hist.events):
        errors.append("event ids are not unique")
    if any(not 0.0 <= e["time_ma"] <= hist.duration_ma + 1e-6 for e in hist.events):
        errors.append("an event lies outside the simulated time span")
    peaks = {p["id"] for p in hist.plan["peaks"]}
    for item in hist.plan["passes"]:
        if not set(item["between_peaks"]) <= peaks:
            errors.append("a pass refers to a peak that does not exist")
            break
    checks.update(
        epoch_time_ma=hist.epoch_time_ma,
        epochs=len(hist.epochs),
        events=len(hist.events),
        mountain_systems=len(hist.plan["systems"]),
        volcanoes=len(hist.volcanoes),
        max_elevation_m=float(elev.max()),
        land_fraction_viewport=round(float(land.mean()), 5),
    )
    return {"ok": not errors, "errors": errors, "warnings": warnings, "checks": checks}


def validate_hydro(project) -> dict:
    """Invariants of v4: one surface, acyclic routing, non-negative water and sediment."""
    from .sea import label_mask

    h = project.hydro
    layout = project.layout
    L = h.layers
    errors: list[str] = []
    checks: dict = {}
    shape = (layout.height, layout.width)
    nan_ok = {"lake_level_m", "lake_depth_m"}
    for name, arr in L.items():
        if arr.shape != shape:
            errors.append(f"layer {name} has shape {arr.shape}, expected {shape}")
        elif (
            np.issubdtype(arr.dtype, np.floating)
            and name not in nan_ok
            and not np.isfinite(arr).all()
        ):
            errors.append(f"layer {name} contains non-finite values")
    if errors:
        return {"ok": False, "errors": errors, "warnings": [], "checks": checks}
    z = L["elevation_m"]
    if L["post_erosion_elevation_m"] is not z:
        errors.append("post_erosion_elevation_m and elevation_m are different arrays")
    if not np.array_equal(L["land_mask"], z > np.float32(h.sea_level_m)):
        errors.append("land_mask is not (final elevation_m > sea_level_m)")
    if not np.array_equal(label_mask(L["land_mask"], layout)[0], L["landmass_id"]):
        errors.append("landmass_id is not the set of edge-connected land components")
    for name in (
        "precipitation_mm_yr",
        "runoff_mm_yr",
        "discharge_m3_s",
        "v4_erosion_m",
        "v4_deposition_m",
        "sediment_thickness_m",
        "actual_evapotranspiration_mm_yr",
        "potential_evapotranspiration_mm_yr",
        "flow_accumulation_area_km2",
    ):
        if float(L[name].min()) < 0.0:
            errors.append(f"{name} is negative")
    dry_land = L["land_mask"] & (L["lake_id"] < 0)  # lakes may evaporate their inflow
    if np.any(
        L["actual_evapotranspiration_mm_yr"][dry_land] > L["precipitation_mm_yr"][dry_land] + 1e-3
    ):
        errors.append("actual evaporation exceeds precipitation on land")
    rc = L["flow_receiver"].ravel()
    n = rc.size
    # acyclic: following receivers from every cell ends at an outlet within n steps
    depth = 0
    cur = np.arange(n)
    while depth <= n:
        nxt = rc[cur]
        if np.array_equal(nxt, cur):
            break
        cur = nxt
        depth += 1
    if depth > n:
        errors.append("the flow graph contains a cycle")
    checks["longest_flow_path_hexes"] = depth
    q = L["discharge_m3_s"].ravel()
    lake = L["lake_id"].ravel() >= 0
    land = L["land_mask"].ravel()
    nz = (rc != np.arange(n)) & land & ~lake
    losing = L["channel_loss_m3_s"].ravel() > 0  # dry-climate channels lose water
    ups = (
        nz
        & ~lake[rc]
        & land[rc]
        & (q[rc] + 1e-6 < q * 0.999)
        & ~L["endorheic_mask"].ravel()
        & ~losing[rc]
    )
    checks["discharge_decreases_downstream"] = int(ups.sum())
    if ups.sum() > 0:
        errors.append(f"discharge decreases downstream at {int(ups.sum())} hexes outside lakes")
    if L["river_mask"].any() and float(L["flow_accumulation_area_km2"][L["river_mask"]].min()) < (
        project.config.hydrology.river_min_upstream_hexes * layout.hex_area_km2 - 1e-3
    ):
        errors.append("a river hex drains less than the minimum upstream area")
    if L["river_mask"].any() and float(L["discharge_m3_s"][L["river_mask"]].min()) < (
        project.config.hydrology.river_min_discharge_m3_s - 1e-6
    ):
        errors.append("a river hex has less than the river threshold discharge")
    b = h.budget
    if b:
        closure = abs(b.get("closure_error_m3", 0.0))
        checks["sediment_budget_closure_m3"] = closure
        if closure > 1e-6 * max(b.get("fluvial_eroded_m3", 0.0), 1.0):
            errors.append("the sediment budget does not close")
    checks.update(
        rivers=len(h.river_systems),
        lakes=len(h.lakes),
        land_fraction=round(float(L["land_mask"].mean()), 5),
    )
    return {"ok": not errors, "errors": errors, "warnings": [], "checks": checks}
