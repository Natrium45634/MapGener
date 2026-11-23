"""Command line interface (works without PySide6)."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from . import render, views
from .api import (
    derive_landmasses,
    generate_history,
    generate_v2,
    generate_v4,
    history_events,
    select_epoch,
)
from .config import (
    CLIMATE_PROFILES,
    CONTINENT_PROFILES,
    EDGE_OUTLET_MODES,
    HISTORY_PROFILES,
    MOUNTAIN_STYLES,
    PRESETS,
    ROUTING_MODES,
    Config,
    ConfigError,
)
from .io import ProjectIntegrityError, load_project, rivers_geojson, save_project
from .pipeline import generate
from .version import GENERATOR_VERSION


def _peak_memory_mb() -> float | None:
    """Peak resident memory of this process in MB (None if the OS does not tell)."""
    if sys.platform == "win32":
        try:
            import ctypes
            from ctypes import wintypes

            class Counters(ctypes.Structure):
                _fields_ = [("cb", wintypes.DWORD), ("PageFaultCount", wintypes.DWORD)] + [
                    (name, ctypes.c_size_t)
                    for name in (
                        "PeakWorkingSetSize",
                        "WorkingSetSize",
                        "QuotaPeakPagedPoolUsage",
                        "QuotaPagedPoolUsage",
                        "QuotaPeakNonPagedPoolUsage",
                        "QuotaNonPagedPoolUsage",
                        "PagefileUsage",
                        "PeakPagefileUsage",
                    )
                ]

            counters = Counters()
            counters.cb = ctypes.sizeof(Counters)
            kernel32 = ctypes.WinDLL("kernel32")
            kernel32.GetCurrentProcess.restype = wintypes.HANDLE
            psapi = ctypes.WinDLL("psapi")
            psapi.GetProcessMemoryInfo.argtypes = [
                wintypes.HANDLE,
                ctypes.POINTER(Counters),
                wintypes.DWORD,
            ]
            if psapi.GetProcessMemoryInfo(
                kernel32.GetCurrentProcess(), ctypes.byref(counters), counters.cb
            ):
                return counters.PeakWorkingSetSize / (1024 * 1024)
        except Exception:
            return None
        return None
    try:
        import resource
    except ImportError:
        return None
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return peak / (1024 * 1024) if sys.platform == "darwin" else peak / 1024


def _config_from_args(args) -> Config:
    data: dict = {}
    if args.config:
        data = json.loads(Path(args.config).read_text(encoding="utf-8"))
    if args.preset:
        data.update(PRESETS[args.preset])
    for key in ("seed", "width", "height", "hex_spacing_km", "mode"):
        value = getattr(args, key)
        if value is not None:
            data[key] = value
    plates = dict(data.get("plates", {}))
    if args.plates is not None:
        plates["count"] = "auto" if args.plates == "auto" else int(args.plates)
    if args.profile is not None:
        plates["profile"] = args.profile
    if plates:
        data["plates"] = plates
    _apply_v2_args(args, data)
    _apply_v4_args(args, data)
    if args.origin is not None:
        data["viewport_origin_km"] = args.origin
    if args.sim_bounds is not None:
        data["simulation_bounds_km"] = args.sim_bounds
    return Config.from_dict(data)


def _apply_v2_args(args, data: dict) -> None:
    """Copy v2.0 command line options into a config dictionary."""
    cont = dict(data.get("continents", {}))
    if getattr(args, "continent_profile", None) is not None:
        cont["continent_profile"] = args.continent_profile
    if getattr(args, "major_landmasses", None) is not None:
        value = args.major_landmasses
        cont["major_landmass_count"] = "auto" if value == "auto" else int(value)
    if getattr(args, "size_variety", None) is not None:
        cont["continent_size_variety"] = args.size_variety
    if getattr(args, "coast_irregularity", None) is not None:
        cont["coast_irregularity"] = args.coast_irregularity
    if cont:
        data["continents"] = cont
    if getattr(args, "relief_strength", None) is not None:
        data["relief"] = dict(data.get("relief", {}), relief_strength=args.relief_strength)
    sea = dict(data.get("sea", {}))
    if (
        getattr(args, "land_fraction", None) is not None
        and getattr(args, "sea_level", None) is not None
    ):
        raise ConfigError("--land-fraction and --sea-level are mutually exclusive")
    if getattr(args, "land_fraction", None) is not None:
        sea.update(sea_mode="target_land_fraction", target_land_fraction=args.land_fraction)
    if getattr(args, "sea_level", None) is not None:
        sea.update(sea_mode="fixed_sea_level_m", fixed_sea_level_m=args.sea_level)
    if sea:
        data["sea"] = sea
    hist = dict(data.get("history", {}))
    for arg, key in (
        ("duration_ma", "duration_ma"),
        ("time_step_ma", "time_step_ma"),
        ("history_intensity", "intensity"),
        ("erosion", "erosion_intensity"),
        ("uplift_scale", "uplift_scale"),
        ("hotspots", "hotspot_density"),
        ("checkpoint_stride", "checkpoint_stride"),
    ):
        if getattr(args, arg, None) is not None:
            hist[key] = getattr(args, arg)
    if getattr(args, "epoch", None) is not None:
        hist["epoch"] = _parse_epoch(args.epoch)
    if hist:
        data["history"] = hist
    mnt = dict(data.get("mountains", {}))
    if getattr(args, "mountain_detail", None) is not None:
        mnt["detail"] = args.mountain_detail
    if getattr(args, "mountain_backend", None) is not None:
        mnt["backend"] = args.mountain_backend
    if getattr(args, "mountain_style", None) is not None:
        mnt["style"] = args.mountain_style
    if getattr(args, "mountain_complexity", None) is not None:
        mnt["complexity"] = args.mountain_complexity
    if mnt:
        data["mountains"] = mnt


def _apply_v4_args(args, data: dict) -> None:
    """Copy v4.0 command line options (climate, hydrology, erosion) into a config dict."""
    sections = {
        "climate": (
            ("climate_profile", "profile"),
            ("latitude", "latitude_center_deg"),
            ("km_per_degree", "km_per_degree"),
            ("temperature_offset", "temperature_offset_c"),
            ("lapse_rate", "lapse_rate_c_per_km"),
            ("wind_strength", "wind_strength"),
            ("precipitation_scale", "precipitation_scale"),
        ),
        "hydrology": (
            ("river_threshold", "river_min_discharge_m3_s"),
            ("river_min_hexes", "river_min_upstream_hexes"),
            ("river_system_hexes", "river_min_system_hexes"),
            ("routing", "routing"),
            ("edge_outlets", "edge_outlets"),
        ),
        "erosion": (
            ("erosion_ma", "duration_ma"),
            ("erosion_steps", "steps"),
            ("stream_power_k", "k_stream"),
            ("incision_threshold", "incision_threshold_m_ma"),
        ),
    }
    for section, pairs in sections.items():
        values = dict(data.get(section, {}))
        for arg, key in pairs:
            if getattr(args, arg, None) is not None:
                values[key] = getattr(args, arg)
        if section == "erosion" and getattr(args, "no_erosion", False):
            values["enabled"] = False
        if values:
            data[section] = values


def _add_v4_options(parser) -> None:
    parser.add_argument("--climate-profile", choices=CLIMATE_PROFILES)
    parser.add_argument(
        "--latitude", type=float, help="conditional latitude of the map centre, deg (N > 0)"
    )
    parser.add_argument(
        "--km-per-degree", type=float, help="km per degree of latitude; 0 = auto (111.2)"
    )
    parser.add_argument("--temperature-offset", type=float, help="added to all temperatures, C")
    parser.add_argument("--lapse-rate", type=float, help="temperature lapse rate, C per km")
    parser.add_argument("--wind-strength", type=float, help="multiplier of the wind belts")
    parser.add_argument("--precipitation-scale", type=float, help="multiplier of precipitation")
    parser.add_argument("--river-threshold", type=float, help="river from this discharge, m3/s")
    parser.add_argument("--river-min-hexes", type=int, help="river from this upstream area")
    parser.add_argument(
        "--river-system-hexes", type=int, help="draw river systems with a basin >= N hexes"
    )
    parser.add_argument("--routing", choices=ROUTING_MODES, help="flow receiver choice")
    parser.add_argument(
        "--edge-outlets",
        choices=EDGE_OUTLET_MODES,
        help="window edges drain out only where the world drainage leaves (world) or all",
    )
    parser.add_argument("--erosion-ma", type=float, help="duration of v4 landscape evolution, Ma")
    parser.add_argument("--erosion-steps", type=int, help="coupled climate-erosion steps")
    parser.add_argument("--stream-power-k", type=float, help="stream-power coefficient K")
    parser.add_argument("--incision-threshold", type=float, help="stream-power threshold E_c, m/Ma")
    parser.add_argument("--no-erosion", action="store_true", help="v4 without landscape change")


def _parse_epoch(text):
    return text if text in ("final", "initial") else float(text)


def _add_v3_options(parser) -> None:
    parser.add_argument("--duration-ma", type=float, help="simulated geological time, Ma")
    parser.add_argument("--time-step-ma", type=float, help="time step, Ma")
    parser.add_argument("--checkpoint-stride", type=int, help="store a snapshot every N steps")
    parser.add_argument("--history-intensity", choices=HISTORY_PROFILES)
    parser.add_argument("--uplift-scale", type=float, help="multiplier of tectonic uplift")
    parser.add_argument("--erosion", type=float, help="erosion intensity, 0 = off")
    parser.add_argument("--hotspots", type=float, help="hotspot density multiplier, 0 = off")
    parser.add_argument("--mountain-detail", type=float, help="structural mountain relief, 0..2")
    parser.add_argument("--mountain-backend", choices=("auto", "reference", "pmge"))
    parser.add_argument(
        "--mountain-style",
        choices=("auto", *MOUNTAIN_STYLES),
        help="architecture of every mountain system; auto = from tectonics, age and seed",
    )
    parser.add_argument(
        "--mountain-complexity",
        type=float,
        help="0 = one crest per system, 1 = parallel ranges and basins (default), 2 = more",
    )
    parser.add_argument("--epoch", help='snapshot to show: "final", "initial" or a time in Ma')


def _add_v2_options(parser) -> None:
    parser.add_argument("--continent-profile", choices=CONTINENT_PROFILES)
    parser.add_argument("--major-landmasses", help='"auto" or the wished number of major masses')
    parser.add_argument("--size-variety", type=float, help="continent_size_variety, 0..1")
    parser.add_argument("--coast-irregularity", type=float, help="0..1.5")
    parser.add_argument("--relief-strength", type=float, help="macro-tectonic relief, 0..3")
    parser.add_argument(
        "--land-fraction",
        type=float,
        help="sea mode target_land_fraction: land share of the simulation area",
    )
    parser.add_argument(
        "--sea-level", type=float, help="sea mode fixed_sea_level_m: level in metres (model datum)"
    )


def _summary(project) -> str:
    g = project.geometry
    v = project.validation
    lines = [
        f"world_id        {project.world_id}",
        f"seed            {project.config.seed}",
        f"grid            {g.layout.width} x {g.layout.height} hexes, "
        f"{g.layout.spacing_km:g} km per hex ({g.mode})",
        f"viewport km     {[round(x, 1) for x in g.viewport_bounds.as_list()]}",
        f"simulation km   {[round(x, 1) for x in g.simulation_bounds.as_list()]}",
        f"plates          {project.stats.get('plates_in_simulation')} in simulation area, "
        f"{project.stats.get('plates_visible')} visible, "
        f"{project.stats.get('plates_total')} incl. halo",
        f"boundary edges  {len(project.boundaries)}",
        f"continental     {100 * project.stats.get('continental_fraction_visible', 0):.1f} % "
        "of visible hexes",
        f"geology hash    {project.geology_hash()}",
        f"validation      {'OK' if v.get('ok') else 'FAILED'}",
    ]
    if project.geo is None:
        lines.append("v2.0            not generated (continents / heights are absent)")
    else:
        geo = project.geo
        w = geo.summary["world"]
        lines += [
            f"sea level       {geo.sea_level_m:+.1f} m ({geo.summary['sea_mode']})",
            f"land            {100 * geo.summary['land_fraction_viewport']:.1f} % of the window, "
            f"{100 * w['land_fraction_simulation']:.1f} % of the simulation area",
            f"landmasses      {len(geo.landmasses)} in the window "
            f"({geo.summary['major_landmasses_in_window']} major); "
            f"{w['major_landmass_count']} major in the simulation area",
            f"crust blocks    {len(geo.blocks)} in the window; shelf "
            f"{100 * geo.summary['shelf_fraction_viewport']:.1f} % of the window",
            f"elevation       {geo.summary['min_elevation_m']:+.0f} .. "
            f"{geo.summary['max_elevation_m']:+.0f} m (macro relief, no detail)",
            f"v2 hash         {geo.geo_hash()}",
            f"v2 validation   {'OK' if geo.validation.get('ok') else 'FAILED'}",
        ]
        for err in geo.validation.get("errors", []):
            lines.append(f"  ERROR   {err}")
    if project.hist is None:
        lines.append("v3.0            not generated (no geological history)")
    else:
        h = project.hist
        m = h.summary
        lines += [
            f"history         {h.duration_ma:g} Ma in {m['steps']} steps "
            f"(x{m['substeps_per_step']} sub-steps); showing t = {h.epoch_time_ma:g} Ma; "
            f"snapshots at {h.epochs}",
            f"change          {100 * m['changed_hex_share']:.0f} % of hexes moved by > 50 m; "
            f"new crust {100 * m['new_crust_share_viewport']:.1f} % of the window",
            f"final surface   {m['min_elevation_m']:+.0f} .. {m['max_elevation_m']:+.0f} m; land "
            f"{100 * m['land_fraction_viewport']:.1f} % (baseline "
            f"{100 * m['baseline_land_fraction_viewport']:.1f} %)",
            f"mountains       {m['mountain_systems']} systems, {m['ridges']} ridges, "
            f"{m['peaks']} peaks, {m['passes']} passes, {m['valleys']} valleys "
            f"[backend: {m['mountain_backend']['name']}"
            f"{'' if m['mountain_backend']['is_pmge'] else ', PMGE library not found'}]",
            f"volcanism       {m['volcanoes']} centres ({m['active_volcanoes']} active), "
            f"{m['hotspots']} hotspots",
            f"events          {m['events']}",
            f"v3 hash         {h.hist_hash()}",
            f"v3 validation   {'OK' if h.validation.get('ok') else 'FAILED'}",
        ]
        for err in h.validation.get("errors", []):
            lines.append(f"  ERROR   {err}")
    lines += _summary_v4(project)
    for err in v.get("errors", []):
        lines.append(f"  ERROR   {err}")
    for warn in project.warnings:
        lines.append(f"  warning {warn}")
    return "\n".join(lines)


def _coast_line(c) -> str:
    if not c:
        return "coasts          not classified"
    top = sorted(c["shares"].items(), key=lambda kv: -kv[1])[:4]
    return (
        f"coasts          {c['coast_hexes']} hexes; rocky {100 * c['rocky_share']:.0f} %, sandy "
        f"{100 * c['sandy_share']:.0f} %; " + ", ".join(f"{k} {100 * v:.0f} %" for k, v in top)
    )


def _summary_v4(project) -> list[str]:
    if project.hydro is None:
        return ["v4.0            not generated (no climate, rivers, erosion)"]
    h = project.hydro
    s = h.summary
    lat = h.latitude
    a = s["aridity_shares"]
    lines = [
        f"climate         {s['climate_profile']}; latitude {lat['center_deg']:+.1f} deg at the "
        f"centre, {lat['km_per_degree']:.1f} km/deg"
        f"{' (compressed)' if lat.get('compressed') else ''}",
        "land climate    no land in the window"
        if s["land_mean_temperature_c"] is None
        else f"land climate    mean {s['land_mean_temperature_c']:+.1f} C, "
        f"{s['land_mean_precipitation_mm_yr']:.0f} mm/yr, runoff ratio {s['runoff_ratio']:.2f}",
        f"aridity         hyper-arid {100 * a['hyper_arid']:.0f} %, arid {100 * a['arid']:.0f} %,"
        f" semi-arid {100 * a['semi_arid']:.0f} %, dry sub-humid "
        f"{100 * a['dry_subhumid']:.0f} %, humid {100 * a['humid']:.0f} % of land",
        f"rivers          {s['rivers']} systems, {s['river_segments']} segments, Strahler "
        f"<= {s['max_strahler']}, max discharge {s['max_discharge_m3_s']:,.0f} m3/s",
        f"lakes           {s['lakes']} ({s['closed_lakes']} closed, plus "
        f"{s.get('dry_closed_basins', 0)} dry pans), "
        f"{100 * s['lake_share_of_land']:.2f} % of land; no outflow to the sea "
        f"{100 * s['no_outflow_share_of_land']:.1f} % (closed basins "
        f"{100 * s['endorheic_share_of_land']:.1f} %)",
        f"erosion         {s['erosion_duration_ma']:g} Ma in {s['erosion_steps']} steps, mean "
        f"denudation {s['mean_denudation_m_ma']:.0f} m/Ma; land "
        f"{100 * s['land_fraction_viewport']:.1f} % (v3 {100 * s['v3_land_fraction_viewport']:.1f}"
        " %)",
        _coast_line(s.get("coasts")),
        f"v4 hash         {h.hydro_hash()}",
        f"v4 validation   {'OK' if h.validation.get('ok') else 'FAILED'}",
    ]
    lines += [f"  ERROR   {err}" for err in h.validation.get("errors", [])]
    return lines


def cmd_generate(args) -> int:
    code_through = None
    if getattr(args, "code", None):
        from .world_tools import parse_world_code

        config, code_through = parse_world_code(args.code)
    else:
        config = _config_from_args(args)
    last = [""]

    from .progress import ProgressTracker, format_seconds

    tracker = ProgressTracker("v50", config.width * config.height)

    def progress(stage: str, fraction: float):
        st = tracker.report(stage, fraction)
        if not args.quiet and st.stage != last[0]:
            last[0] = st.stage
            print(f"[{100 * st.overall:5.1f}%] {st.title}  (прошло {format_seconds(st.elapsed_s)}"
                  f", осталось ~{format_seconds(st.eta_s)})", file=sys.stderr)  # fmt: skip

    t0 = time.perf_counter()
    through = "v1" if args.v1_only else "v2" if args.no_history else "v3" if args.no_v4 else "v4"
    if getattr(args, "through", None):
        through = args.through
    elif code_through is not None:
        through = code_through
    project = generate(config, progress=progress, through=through)
    elapsed = time.perf_counter() - t0
    print(_summary(project))
    peak = _peak_memory_mb()
    memory = f"{peak:.0f} MB" if peak is not None else "n/a"
    print(f"time            {elapsed:.2f} s   peak memory {memory}")
    if args.out:
        out = save_project(project, args.out)
        print(f"saved           {out}")
        for layer in args.png or []:
            img = render.render_project(project, layer)
            print(f"image           {render.write_png(Path(out) / f'{layer}.png', img)}")
        if args.png_views:
            for image in _render_all_views(project, Path(out)):
                print(f"image           {image}")
    return 0 if _all_ok(project) else 2


def _all_ok(project) -> bool:
    return bool(
        project.validation["ok"]
        and (project.geo is None or project.geo.validation.get("ok"))
        and (project.hydro is None or project.hydro.validation.get("ok"))
    )


def _render_all_views(project, out_dir: Path, px_per_hex: float | None = None) -> list[Path]:
    """One PNG per view and mode that has data (each tab is exported separately)."""
    written = []
    for spec in views.VIEWS.values():
        if not spec.availability(project)[0]:
            continue
        for mode in spec.modes:
            try:
                img = views.render_view_image(project, spec.id, mode.id, px_per_hex=px_per_hex)
            except LookupError:
                continue  # a mode that needs a later stage (e.g. epoch comparison without v3)
            written.append(render.write_png(out_dir / f"view_{spec.id}_{mode.id}.png", img))
    return written


def _save_target(args) -> Path:
    return Path(args.out) if args.out else Path(args.project)


def cmd_continents(args) -> int:
    """Add or update the v2.0 data of an existing project."""
    project = load_project(args.project)
    data = project.config.to_dict()
    _apply_v2_args(args, data)
    t0 = time.perf_counter()
    result = generate_v2(project, Config.from_dict(data))
    print(_summary(result))
    print(f"time            {time.perf_counter() - t0:.2f} s (v1.0 data reused unchanged)")
    print(f"saved           {save_project(result, _save_target(args))}")
    return 0 if _all_ok(result) else 2


def cmd_sea(args) -> int:
    """Change only the sea level of a project that already has v2.0 data."""
    project = load_project(args.project)
    if project.geo is None:
        print(
            "error: this project has no v2.0 data yet; run `hexworld continents` first",
            file=sys.stderr,
        )
        return 1
    if (args.level is None) == (args.land_fraction is None):
        print("error: give exactly one of --level and --land-fraction", file=sys.stderr)
        return 1
    result = derive_landmasses(
        project, sea_level_m=args.level, target_land_fraction=args.land_fraction
    )
    print(_summary(result))
    print(f"saved           {save_project(result, _save_target(args))}")
    return 0 if _all_ok(result) else 2


def cmd_history(args) -> int:
    """Add or update the v3.0 geological history of a saved project."""
    project = load_project(args.project)
    data = project.config.to_dict()
    _apply_v2_args(args, data)
    t0 = time.perf_counter()
    result = generate_history(project, Config.from_dict(data))
    print(_summary(result))
    print(f"time            {time.perf_counter() - t0:.2f} s (v1/v2 baseline reused unchanged)")
    print(f"saved           {save_project(result, _save_target(args))}")
    return 0 if _all_ok(result) else 2


def cmd_climate(args) -> int:
    """Add or update the v4.0 climate, hydrology and erosion of a saved project."""
    project = load_project(args.project)
    if project.hist is None:
        print(
            "error: v4 needs the v3.0 epoch surface; run `hexworld history` first",
            file=sys.stderr,
        )
        return 1
    data = project.config.to_dict()
    _apply_v4_args(args, data)
    t0 = time.perf_counter()
    result = generate_v4(project, Config.from_dict(data))
    print(_summary(result))
    reused = [s for s in result.stats.get("reused_stages", []) if s.startswith("v4")]
    print(
        f"time            {time.perf_counter() - t0:.2f} s (v1-v3 reused unchanged"
        f"{'; reused ' + ', '.join(reused) if reused else ''})"
    )
    print(f"saved           {save_project(result, _save_target(args))}")
    return 0 if _all_ok(result) else 2


def cmd_epoch(args) -> int:
    """Show another stored snapshot of the history (no re-simulation)."""
    project = load_project(args.project)
    if project.hist is None:
        print(
            "error: this project has no geological history; run `hexworld history` first",
            file=sys.stderr,
        )
        return 1
    result = select_epoch(project, _parse_epoch(args.time))
    print(_summary(result))
    print(f"saved           {save_project(result, _save_target(args))}")
    return 0 if _all_ok(result) else 2


def cmd_events(args) -> int:
    project = load_project(args.project)
    if project.hist is None:
        print("error: this project has no geological history", file=sys.stderr)
        return 1
    time_range = (
        (args.since, args.until) if args.since is not None or args.until is not None else None
    )
    if time_range is not None:
        time_range = (time_range[0] or 0.0, time_range[1] if time_range[1] is not None else 1e9)
    for e in history_events(project, time_range=time_range, kinds=args.type):
        end = "" if e.get("end_time_ma") is None else f" .. {e['end_time_ma']:g}"
        print(
            f"{e['time_ma']:8.2f}{end:<10} {e['type']:<18} plates {e['plates']!s:<12} "
            f"at ({e['x_km']:,.0f}, {e['y_km']:,.0f}) km  {e['id']}"
        )
    return 0


def cmd_views(_args) -> int:
    for spec in views.VIEWS.values():
        status = "ready" if spec.status == "ready" else "planned"
        print(f"{spec.id:<11} {spec.title:<11} {status:<8} stage={spec.stage or '-'}  {spec.note}")
        for mode in spec.modes:
            print(f"    mode    {mode.id:<18} {mode.title}")
        for overlay in spec.overlays:
            print(
                f"    overlay {overlay.id:<18} {overlay.title}"
                f"{'  (default)' if overlay.default else ''}"
            )
    return 0


def cmd_export_game(args) -> int:
    from .export_game import export_game

    project = load_project(args.project)
    for f in export_game(project, args.out, px_per_hex=args.px_per_hex):
        print(f)
    return 0


def cmd_code(args) -> int:
    from .world_tools import world_code

    print(world_code(load_project(args.project).config))
    return 0


def cmd_export(args) -> int:
    """Numeric tables (CSV) and arrays (.npy) for other tools; PNG is never a data source."""
    import csv

    import numpy as np

    project = load_project(args.project)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    def table(name, rows):
        if not rows:
            return
        keys = [k for k in rows[0] if not isinstance(rows[0][k], dict)]
        with (out / name).open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(rows)
        print(out / name)

    table("plates.csv", project.plates)
    names = list(project.layers)
    if project.hist is not None:
        h = project.hist
        table("events.csv", h.events)
        table("volcanoes.csv", h.volcanoes)
        table("mountain_systems.csv", h.plan["systems"])
        table("peaks.csv", h.plan["peaks"])
        table("passes.csv", h.plan["passes"])
        table("landmasses_epoch.csv", h.landmasses)
        for name, arr in h.layers.items():
            np.save(out / f"epoch_{name}.npy", arr)
        print(f"{out}/epoch_*.npy ({len(h.layers)} layers of t = {h.epoch_time_ma:g} Ma)")
    if project.hydro is not None:
        hy = project.hydro
        table("river_systems.csv", hy.river_systems)
        table("river_segments.csv", hy.rivers)
        table("lakes.csv", hy.lakes)
        table("basins.csv", hy.basins)
        table("landmasses_v4.csv", hy.landmasses)
        geojson = out / "rivers.geojson"
        geojson.write_text(
            json.dumps(rivers_geojson(project), ensure_ascii=False), encoding="utf-8"
        )
        print(geojson)
        for name, arr in hy.layers.items():
            np.save(out / f"v4_{name}.npy", arr)
        print(
            f"{out}/v4_*.npy ({len(hy.layers)} layers after {hy.summary['erosion_duration_ma']:g}"
            " Ma of climate-driven erosion)"
        )
    if project.geo is not None:
        table("continental_blocks.csv", project.geo.blocks)
        table("landmasses.csv", project.geo.landmasses)
        names += list(project.geo.layers)
        np.save(out / "world_elevation_m.npy", project.geo.world["elevation_m"])
        print(out / "world_elevation_m.npy")
    for name in names:
        np.save(out / f"{name}.npy", project.baseline_layer(name))
        print(out / f"{name}.npy")
    return 0


def cmd_info(args) -> int:
    project = load_project(args.project)
    print(_summary(project))
    topo = getattr(project, "topo", None)
    if topo is not None:
        t = topo.summary
        print(f"v4.7            {t['units']} terrain units, {t['treeline_hexes']} hexes cross the "
              f"thermal tree limit; ok {topo.validation.get('ok')}")  # fmt: skip
    bio = getattr(project, "biomes", None)
    if bio is not None:
        b = bio.summary
        top = ", ".join(f"{k} {100 * v:.0f} %" for k, v in list(b["land_shares_top"].items())[:5])
        print(f"v5.0            {b['classes_present_land']} land classes ({b['input_quality']}); "
              f"ice bodies {b['ice_body_count']}; ok {bio.validation.get('ok')}")  # fmt: skip
        print(f"                {top}")
    if args.plates:
        print("\n id  name          class   area Mkm2  cont%  speed mm/yr  visible hexes")
        for p in project.plates:
            if p["visible_hexes"] == 0 and not args.all:
                continue
            print(
                f"{p['id']:3d}  {p['name']:<12}  {p['size_class']:<6}  {p['area_km2'] / 1e6:9.2f}"
                f"  {100 * p['continental_fraction']:5.1f}  {p['speed_mm_yr']:11.1f}"
                f"  {p['visible_hexes']:13d}"
            )
    return 0


def cmd_validate(args) -> int:
    project = load_project(args.project)
    v = project.validation
    print(json.dumps(v, indent=2, ensure_ascii=False))
    return 0 if v["ok"] else 2


def cmd_render(args) -> int:
    project = load_project(args.project)
    if args.view == "all" or args.layer == "all":
        target = Path(args.out or args.project)
        for image in _render_all_views(project, target, args.px_per_hex):
            print(image)
        return 0
    if args.view:
        spec = views.VIEWS[args.view]
        ok, reason = spec.availability(project)
        if not ok:
            print(f"error: {reason}", file=sys.stderr)
            return 1
        mode = args.mode or spec.default_mode()
        overlays = None if args.overlays is None else set(args.overlays)
        img = views.render_view_image(project, args.view, mode, overlays, args.px_per_hex)
        default_name = f"view_{args.view}_{mode}.png"
        out = (
            Path(args.out)
            if args.out and args.out.lower().endswith(".png")
            else (Path(args.out or args.project) / default_name)
        )
        print(render.write_png(out, img))
        return 0
    img = render.render_project(project, args.layer, px_per_hex=args.px_per_hex)
    out = (
        Path(args.out)
        if args.out and args.out.lower().endswith(".png")
        else (Path(args.out or args.project) / f"{args.layer}.png")
    )
    print(render.write_png(out, img))
    return 0


def cmd_hex(args) -> int:
    project = load_project(args.project)
    info = project.hex_info(args.col, args.row)
    info["plate"] = project.plate(info["plate_id"])
    print(json.dumps(info, indent=2, ensure_ascii=False))
    return 0


def cmd_environment(args) -> int:
    """Add or update the v4.5 ecological foundation of a saved project."""
    from .api import generate_environment

    project = load_project(args.project)
    if project.hydro is None:
        print("error: v4.5 needs v4 data; run `hexworld climate` first", file=sys.stderr)
        return 1
    data = project.config.to_dict()
    env = dict(data.get("environment", {}))
    for arg, key in (
        ("root_depth", "reference_root_depth_m"),
        ("soil_depth", "max_soil_depth_m"),
        ("rain_days", "rain_days_per_month"),
        ("degree_day", "degree_day_mm"),
    ):
        if getattr(args, arg, None) is not None:
            env[key] = getattr(args, arg)
    data["environment"] = env
    t0 = time.perf_counter()
    result = generate_environment(project, Config.from_dict(data))
    e = result.env
    wb = e.summary["water_balance"]
    print(
        f"v4.5            ok {e.validation['ok']}; soil on "
        f"{100 * e.summary['soil_present_share']:.0f} % "
        f"of land, mean AWC {e.summary['mean_awc_mm']:.0f} mm"
    )
    print(
        f"water           P {wb['precipitation_mm_yr']:.0f} = ET {wb['aet_mm_yr']:.0f} + runoff "
        f"{wb['runoff_mm_yr']:.0f} + deep {wb['deep_loss_mm_yr']:.0f} + dS "
        f"{wb['storage_change_mm_yr']:.0f} mm/yr; residual {wb['max_abs_residual_mm']:.1e} mm; "
        f"spin-up {wb['spinup_years']} yr"
    )
    print(f"time            {time.perf_counter() - t0:.2f} s")
    if args.hex is not None:
        from .api import export_hex_months_csv

        col, row = args.hex
        target = args.csv or f"hex_{col}_{row}_months.csv"
        print(f"months csv      {export_hex_months_csv(result, col, row, target)}")
    print(f"saved           {save_project(result, _save_target(args))}")
    return 0 if e.validation["ok"] else 2


def cmd_months(args) -> int:
    """Print (or write as CSV) the 12 months of one hex."""
    from .api import export_hex_months_csv, hex_months

    project = load_project(args.project)
    if args.csv:
        print(export_hex_months_csv(project, args.col, args.row, args.csv))
        return 0
    rows = hex_months(project, args.col, args.row)
    keys = [k for k in rows[0] if k not in ("month", "days")]
    print("month " + " ".join(f"{k[:14]:>14}" for k in keys))
    for r in rows:
        cells = " ".join("             -" if r[k] is None else f"{r[k]:14.2f}" for k in keys)
        print(f"{r['month']:>5} {cells}")
    return 0


def cmd_topoclimate(args) -> int:
    """Add or update the v4.7 topoclimate of a saved project (needs v4.5)."""
    from .api import export_terrain_units_csv, generate_topoclimate

    project = load_project(args.project)
    if project.env is None:
        print("error: v4.7 needs v4.5 data; run `hexworld environment` first", file=sys.stderr)
        return 1
    data = project.config.to_dict()
    tc = dict(data.get("topoclimate", {}))
    for arg, key in (
        ("quality", "quality"),
        ("temperature_mode", "temperature_mode"),
        ("tree_gsl_days", "tree_gsl_days"),
        ("tree_gst", "tree_gst_c"),
        ("horizon_km", "horizon_max_distance_km"),
    ):
        if getattr(args, arg, None) is not None:
            tc[key] = getattr(args, arg)
    data["topoclimate"] = tc
    t0 = time.perf_counter()
    result = generate_topoclimate(project, Config.from_dict(data))
    t = result.topo
    s = t.summary
    print(
        f"v4.7            ok {t.validation['ok']}; {s['units']} terrain units "
        f"({s['units_per_land_hex']} per land hex), {s['detail_hexes']} hexes with the "
        f"mountain detail"
    )
    shares = {k.replace("thermal_", ""): v for k, v in s["belt_land_shares"].items() if v}
    print(f"belts           {shares}")
    print(
        f"budgets         area {s['area_closure_max_error']:.1e}, precipitation "
        f"{s['precipitation_conservation_max_rel']:.1e} (rel), water "
        f"{s['water_budget_max_residual_mm']:.1e} mm; spin-up converged "
        f"{100 * s['spinup_converged_share']:.1f} %"
    )
    print(f"treeline        {s['treeline_hexes']} hexes cross the thermal limit; regional "
          f"medians {s['regional_treelines_m'][:8]} m")  # fmt: skip
    print(f"time            {time.perf_counter() - t0:.2f} s")
    if args.hex is not None:
        col, row = args.hex
        target = args.csv or f"hex_{col}_{row}_units.csv"
        print(f"units csv       {export_terrain_units_csv(result, col, row, target)}")
    print(f"saved           {save_project(result, _save_target(args))}")
    return 0 if t.validation["ok"] else 2


def cmd_biomes(args) -> int:
    """Add or update the v5.0 biomes and cryosphere of a saved project (needs v4.5)."""
    from .api import generate_biomes

    project = load_project(args.project)
    if project.env is None:
        print("error: v5.0 needs v4.5 data; run `hexworld environment` first", file=sys.stderr)
        return 1
    data = project.config.to_dict()
    bp = dict(data.get("biomes", {}))
    for arg, key in (("spinup", "cryo_spinup_years"), ("tau", "pft_tau")):
        if getattr(args, arg, None) is not None:
            bp[key] = getattr(args, arg)
    if args.no_paleo_ice:
        bp["paleo_ice_initialization"] = False
    data["biomes"] = bp
    t0 = time.perf_counter()
    result = generate_biomes(project, Config.from_dict(data))
    b = result.biomes
    s = b.summary
    print(f"v5.0            ok {b.validation['ok']}; {s['units']} units ({s['input_quality']}), "
          f"{s['classes_present_land']} land classes")  # fmt: skip
    top = list(s["land_shares_top"].items())[:8]
    print("land shares     " + ", ".join(f"{k} {100 * v:.1f} %" for k, v in top))
    c = s["cryosphere"]
    print(
        f"cryosphere      {c['ice_hexes']} ice hexes, max {c['max_ice_thickness_m']:.0f} m, "
        f"{s['ice_body_count']} ice bodies; mass residual {c['mass_residual_mm_max']:.1e} mm"
    )
    print(f"time            {time.perf_counter() - t0:.2f} s")
    print(f"saved           {save_project(result, _save_target(args))}")
    return 0 if b.validation["ok"] else 2


def cmd_biome(args) -> int:
    """Explain the biome composition of one hex (JSON)."""
    import json

    from .api import explain_biome

    project = load_project(args.project)
    print(json.dumps(explain_biome(project, args.col, args.row), ensure_ascii=False, indent=1))
    return 0


def cmd_units(args) -> int:
    """Print (or write as CSV) the terrain units of one hex."""
    from .api import export_terrain_units_csv, terrain_units

    project = load_project(args.project)
    if args.csv:
        print(export_terrain_units_csv(project, args.col, args.row, args.csv))
        return 0
    rows = terrain_units(project, args.col, args.row)
    if not rows:
        print("no terrain units in this hex (open sea)")
        return 0
    keys = list(rows[0])
    print(" ".join(f"{k[:12]:>12}" for k in keys))
    for r in rows:
        print(" ".join(f"{'-' if r[k] is None else r[k]!s:>12.12}" for k in keys))
    return 0


def cmd_volcanism(args) -> int:
    """Print the v5.1 summary: systems, eruptions of the window, budgets, events."""
    project = load_project(args.project)
    v = getattr(project, "volcanism", None)
    if v is None:
        print("v5.1 volcanism is not calculated for this project (generate it with --through v50).")
        return 1
    s = v.summary
    print(f"volcanism {s['algorithm_version']} ({s['world_profile']}), "
          f"window {s['event_window_years']:g} years")  # fmt: skip
    print(f"  provinces {s['provinces']}: {s['provinces_by_source']}")
    print(f"  systems {s['systems']} (in window {s['systems_in_window']}): "
          f"{s['systems_by_status']}")  # fmt: skip
    print(f"  edifices: {s['systems_by_edifice']}")
    print(f"  activity now: {s['activity_phase']}; lava lakes: {s['lava_lakes']}")
    print(f"  eruptions {s['events']} (explicit {s['explicit_events']}), "
          f"{s['erupted_dre_km3']} km3 DRE; VEI {s['events_by_vei']}")  # fmt: skip
    print(f"  styles: {s['events_by_style']}")
    print(f"  hydrothermal: {s['hydrothermal_fields']}; subglacial events "
          f"{s['subglacial_events']}; new islands {s['emergent_islands']}; "
          f"climate pulses {s['climate_pulses']}")  # fmt: skip
    print(f"  validation: {v.validation.get('ok')} {v.validation.get('checks')}")
    for e in sorted(v.events, key=lambda e: e["years_bp"])[: max(0, args.events)]:
        p = e["products"]
        print(f"  {e['event_id']} {e['years_bp']:9.1f} yr BP {'+'.join(e['style_phases']):38s} "
              f"VEI {e.get('vei', 0)} DRE {p['volume_dre_m3']:.2e} m3 "
              f"column {p['plume_height_m'] / 1000:.1f} km")  # fmt: skip
    return 0


def cmd_atmosphere(args) -> int:
    """Print the v5.4 summary and the monthly vapour budget; optional windAt and export."""
    project = load_project(args.project)
    a = getattr(project, "atmosphere", None)
    if a is None or not a.grid_fields:
        print("v5.4 atmosphere is not calculated for this project (needs earth + atmosphere).")
        return 1
    s = a.summary
    q = s["quality"]
    print(
        f"atmosphere {s['algorithm_version']} mode {s['mode']}, domain {q['domain']}, "
        f"grid {s['grid']['shape']} x {s['grid']['cell_km']:.0f} km"
    )
    wc = s["world_climate"]
    print(
        f"  precipitation coupling {wc.get('precipitation_coupling')}; ocean coupling "
        f"{wc.get('ocean_coupling', {}).get('status')}; Hadley edge {s['hadley_edge_deg']:.1f} deg"
    )
    print(
        "  land rain regimes: "
        + ", ".join(f"{k} {100 * v:.0f}%" for k, v in s["regime_shares_land"].items())
    )
    st = s.get("storms") or {}
    if st:
        print(
            f"  TC geneses {st['tc_genesis_per_yr']:.2f}/yr, track stats {st['track_statistics']}"
        )
    wy = s.get("weather_year")
    if wy:
        print(
            f"  weather year {wy['year_seed']}#{wy['year']}: {wy['event_counts']}, "
            f"interannual {wy['interannual']['mode']}"
        )
    print(f"  validation: {a.validation.get('ok')} {a.validation.get('checks')}")
    print("  month   E kg        P kg        import kg   export kg   dStore kg   rel.resid")
    for r in s.get("budget_months", []):
        print(
            f"  {r['month']:5d} {r['evaporation_kg']:11.3e} {r['precipitation_kg']:11.3e} "
            f"{r['import_kg']:11.3e} {r['export_kg']:11.3e} {r['storage_change_kg']:11.3e} "
            f"{r['relative_residual']:.1e}"
        )
    if args.wind:
        from .atmosphere.circulation import QUALITY_NAMES
        from .pipeline_v54 import wind_at_world

        x, y, month, level = args.wind
        kw = {"pressure_pa": level} if level > 1500 else {"altitude_m": level}
        u, v, qc = wind_at_world(a, [x], [y], int(month) - 1, **kw)
        where = f"p={level:g} Pa" if level > 1500 else f"z={level:g} m"
        print(
            f"  windAt({x:g} km, {y:g} km, month {int(month)}, {where}): "
            f"u={u[0]:.1f} v={v[0]:.1f} m/s ({QUALITY_NAMES[int(qc[0])]})"
        )
    if args.export:
        from .atmosphere.export import export_all

        print("  exported:", export_all(a, args.export))
    return 0


def cmd_terrain(args) -> int:
    """Print the v4.6 summary of a project: sea floor, basins, water, massifs."""
    project = load_project(args.project)
    t = getattr(project, "terrain", None)
    if t is None:
        print("v4.6 is not calculated for this project (generate it again with v4 on).")
        return 1
    s = t.summary
    sf = s.get("sea_floor", {})
    print(
        f"sea floor: {sf.get('marine_hexes', 0)} hexes, ridge pieces {sf.get('ridge_pieces')}, "
        f"crust age p5..p95 {sf.get('age_ma_percentiles')}"
    )
    print(
        f"  margins {sf.get('margin_hexes')}, trench axis hexes {sf.get('trench_axis_hexes')}, "
        f"deepest trench {sf.get('max_trench_depth_m')} m, guyots {sf.get('guyots')}"
    )
    print(f"  provinces: {s.get('marine_province_shares')}")
    b = s.get("basins", {})
    print(
        f"basins in window: {b.get('in_window')}, "
        f"flexural parameter {b.get('flexural_parameter_km')} km"
    )
    print(f"water: {s.get('water')}")
    print(f"massifs: {s.get('massifs')}")
    print(f"submarine sediment: {s.get('submarine_sediment')}")
    return 0


def cmd_section(args) -> int:
    """Cross-section along a line (world km) as CSV."""
    from .api import cross_section, export_cross_section_csv

    project = load_project(args.project)
    start, end = (args.x0, args.y0), (args.x1, args.y1)
    if args.csv:
        print(export_cross_section_csv(project, start, end, args.csv, args.step, args.month))
        return 0
    for r in cross_section(project, start, end, args.step, args.month):
        water = "" if r["water_surface_m"] is None else f" water {r['water_surface_m']:+.0f}"
        climate = ""
        if r.get("t_mean_c") is not None:
            climate = f"  T {r['t_mean_c']:+5.1f} C  GSL {r['gsl_days']:.0f} d  {r['belt']}"
        print(
            f"{r['distance_km']:9.1f} km  {r['height_m']:+9.1f} m{water}  {r['material_type']:>21}"
            f"  {r['province']}{climate}"
        )
    return 0


def cmd_gui(args) -> int:
    try:
        from .gui.app import run
    except ImportError as exc:
        print(f"GUI needs PySide6 (pip install hexworld[gui]): {exc}", file=sys.stderr)
        return 1
    return run(args.project)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="hexworld", description=f"HexWorld {GENERATOR_VERSION} - tectonic hex map generator"
    )
    parser.add_argument("--version", action="version", version=GENERATOR_VERSION)
    sub = parser.add_subparsers(dest="command", required=True)

    g = sub.add_parser("generate", help="generate a project")
    g.add_argument("--config", help="JSON config file (see examples/)")
    g.add_argument("--code", help="world code «HW1-…» (seed + settings, from the GUI)")
    g.add_argument(
        "--through",
        choices=("v1", "v2", "v3", "v46", "v4", "v45", "v47", "v51", "v50"),
        help="last stage to compute (v50 = everything, with biomes and ice)",
    )
    g.add_argument("--preset", choices=sorted(PRESETS), help="size preset")
    g.add_argument("--seed", help="case-sensitive seed [0-9A-Za-z]+")
    g.add_argument("--width", type=int)
    g.add_argument("--height", type=int)
    g.add_argument(
        "--hex-km",
        dest="hex_spacing_km",
        type=float,
        help="distance between adjacent hex centres, km",
    )
    g.add_argument("--mode", choices=("regional", "world"))
    g.add_argument("--plates", help='"auto" or an integer')
    g.add_argument("--profile", choices=("few", "earthlike", "fragmented"))
    g.add_argument(
        "--origin",
        type=float,
        nargs=2,
        metavar=("X", "Y"),
        help="viewport top-left corner in world km",
    )
    g.add_argument(
        "--sim-bounds",
        type=float,
        nargs=4,
        metavar=("XMIN", "YMIN", "XMAX", "YMAX"),
        help="explicit simulation bounds, km (pins the world)",
    )
    g.add_argument("--out", "-o", help="project directory to save into")
    g.add_argument(
        "--png",
        nargs="*",
        metavar="LAYER",
        choices=render.LAYERS,
        help="also write PNG images of these layers into the project directory",
    )
    g.add_argument(
        "--png-views",
        action="store_true",
        help="also write one PNG per view/mode of every tab that has data",
    )
    g.add_argument(
        "--v1-only",
        action="store_true",
        help="stop after the v1.0 tectonic stage (no continents / heights)",
    )
    g.add_argument(
        "--no-history",
        action="store_true",
        help="stop after v2.0 (no geological history, mountains, volcanism)",
    )
    g.add_argument(
        "--no-v4",
        action="store_true",
        help="stop after v3.0 (no climate, hydrology, climate-driven erosion)",
    )
    _add_v2_options(g)
    _add_v3_options(g)
    _add_v4_options(g)
    g.add_argument("--quiet", "-q", action="store_true")
    g.set_defaults(func=cmd_generate)

    c = sub.add_parser(
        "continents", help="add or update v2.0 data (continents, heights) of a saved project"
    )
    c.add_argument("project")
    c.add_argument("--out", "-o", help="save to another directory instead of updating in place")
    _add_v2_options(c)
    c.set_defaults(func=cmd_continents)

    hi = sub.add_parser(
        "history", help="add or update the v3.0 geological history of a saved project"
    )
    hi.add_argument("project")
    hi.add_argument("--out", "-o", help="save to another directory instead of updating in place")
    _add_v3_options(hi)
    hi.set_defaults(func=cmd_history)

    cl = sub.add_parser(
        "climate",
        help="add or update v4.0 data (climate, rivers, lakes, erosion) of a saved project",
    )
    cl.add_argument("project")
    cl.add_argument("--out", "-o", help="save to another directory instead of updating in place")
    _add_v4_options(cl)
    cl.set_defaults(func=cmd_climate)

    ep = sub.add_parser("epoch", help="show another stored snapshot (initial / final / time)")
    ep.add_argument("project")
    ep.add_argument("time", help='"initial", "final" or a time in Ma')
    ep.add_argument("--out", "-o")
    ep.set_defaults(func=cmd_epoch)

    ev = sub.add_parser("events", help="print the chronology of geological events")
    ev.add_argument("project")
    ev.add_argument("--since", type=float, help="from this time, Ma")
    ev.add_argument("--until", type=float, help="up to this time, Ma")
    ev.add_argument("--type", nargs="*", help="event types to show")
    ev.set_defaults(func=cmd_events)

    s = sub.add_parser("sea", help="change only the sea level of a saved v2.0 project")
    s.add_argument("project")
    s.add_argument("--level", type=float, help="fixed sea level, m")
    s.add_argument("--land-fraction", type=float, help="target land share of the simulation area")
    s.add_argument("--out", "-o", help="save to another directory instead of updating in place")
    s.set_defaults(func=cmd_sea)

    w = sub.add_parser("views", help="list the registered map views (tabs), modes and overlays")
    w.set_defaults(func=cmd_views)

    e = sub.add_parser("export", help="export tables (CSV) and numeric layers (.npy)")
    e.add_argument("project")
    e.add_argument("--out", "-o", required=True)
    e.set_defaults(func=cmd_export)

    eg = sub.add_parser(
        "export-game",
        help="export for game engines: 16-bit height map, biome/river maps, hex table, names",
    )
    eg.add_argument("project")
    eg.add_argument("--out", "-o", required=True)
    eg.add_argument("--px-per-hex", type=float, default=8.0, help="pixels per hex (default 8)")
    eg.set_defaults(func=cmd_export_game)

    cd = sub.add_parser("code", help="print the world code «HW1-…» of a saved project")
    cd.add_argument("project")
    cd.set_defaults(func=cmd_code)

    i = sub.add_parser("info", help="show a saved project")
    i.add_argument("project")
    i.add_argument("--plates", action="store_true", help="list plates")
    i.add_argument("--all", action="store_true", help="include plates outside the window")
    i.set_defaults(func=cmd_info)

    v = sub.add_parser("validate", help="re-run validation of a saved project")
    v.add_argument("project")
    v.set_defaults(func=cmd_validate)

    r = sub.add_parser("render", help="render PNG image(s) of a saved project")
    r.add_argument("project")
    r.add_argument(
        "--layer",
        default="plates",
        choices=(*render.LAYERS, "all"),
        help="v1.0 layer (kept for compatibility); prefer --view",
    )
    r.add_argument(
        "--view",
        choices=(*views.VIEWS, "all"),
        help="map view (tab) to render; 'all' writes every view and mode",
    )
    r.add_argument("--mode", help="sub-mode of the view, see `hexworld views`")
    r.add_argument("--overlays", nargs="*", help="overlay ids, see `hexworld views`")
    r.add_argument("--px-per-hex", type=float, help="display scale only; never changes data")
    r.add_argument("--out", "-o", help="PNG file (single layer) or directory")
    r.set_defaults(func=cmd_render)

    h = sub.add_parser("hex", help="print every value of one hex")
    h.add_argument("project")
    h.add_argument("col", type=int)
    h.add_argument("row", type=int)
    h.set_defaults(func=cmd_hex)

    en = sub.add_parser(
        "environment",
        help="add or update v4.5 data (12 months, soils, monthly water, wetness, indices)",
    )
    en.add_argument("project")
    en.add_argument("--out", "-o", help="save to another directory instead of updating in place")
    en.add_argument("--root-depth", type=float, help="reference root zone, m (default 1.0)")
    en.add_argument("--soil-depth", type=float, help="maximum soil depth, m (default 3.0)")
    en.add_argument("--rain-days", type=float, help="rain days per month (infiltration limit)")
    en.add_argument("--degree-day", type=float, help="snow melt, mm per degC per day")
    en.add_argument("--hex", type=int, nargs=2, metavar=("COL", "ROW"),
                    help="also write the 12 months of this hex as CSV")  # fmt: skip
    en.add_argument("--csv", help="file for --hex (default hex_COL_ROW_months.csv)")
    en.set_defaults(func=cmd_environment)

    mo = sub.add_parser("months", help="12-month table of one hex (v4.5)")
    mo.add_argument("project")
    mo.add_argument("col", type=int)
    mo.add_argument("row", type=int)
    mo.add_argument("--csv", help="write CSV instead of printing")
    mo.set_defaults(func=cmd_months)

    te = sub.add_parser("terrain", help="v4.6 summary: sea floor, basins, water, massifs")
    te.add_argument("project")
    te.set_defaults(func=cmd_terrain)

    vo = sub.add_parser("volcanism", help="v5.1 summary: systems, eruptions, budgets")
    vo.add_argument("project")
    vo.add_argument("--events", type=int, default=10, help="list this many recent events")
    vo.set_defaults(func=cmd_volcanism)

    at = sub.add_parser("atmosphere", help="v5.4 summary, vapour budget, windAt, export")
    at.add_argument("project")
    at.add_argument(
        "--wind",
        type=float,
        nargs=4,
        metavar=("X_KM", "Y_KM", "MONTH", "LEVEL"),
        help="windAt: LEVEL > 1500 is a pressure in Pa, else a height in m",
    )
    at.add_argument(
        "--export",
        metavar="DIR",
        help="write weather_events.csv, world_metric JSON, budget CSV, grid npz",
    )
    at.set_defaults(func=cmd_atmosphere)

    se = sub.add_parser("section", help="vertical cross-section along a line (v4.6)")
    se.add_argument("project")
    for name in ("x0", "y0", "x1", "y1"):
        se.add_argument(name, type=float, help="world km")
    se.add_argument("--step", type=float, default=None, help="sample step, km")
    se.add_argument("--csv", help="write CSV instead of printing")
    se.add_argument("--month", type=int, choices=range(12), metavar="0..11",
                    help="also give the temperature of this month (v4.7)")  # fmt: skip
    se.set_defaults(func=cmd_section)

    tc = sub.add_parser(
        "topoclimate",
        help="add or update v4.7 data (terrain units, local climate, thermal belts)",
    )
    tc.add_argument("project")
    tc.add_argument("--out", "-o", help="save to another directory instead of updating in place")
    tc.add_argument("--quality", choices=("preview", "balanced", "reference"))
    tc.add_argument("--temperature-mode", choices=("reference_anchored", "preserve_coarse_mean"))
    tc.add_argument("--tree-gsl-days", type=float, help="thermal tree limit: season days (94)")
    tc.add_argument("--tree-gst", type=float, help="thermal tree limit: season mean, degC (6.4)")
    tc.add_argument("--horizon-km", type=float, help="horizon search distance, km (100)")
    tc.add_argument("--hex", type=int, nargs=2, metavar=("COL", "ROW"),
                    help="also write the terrain units of this hex as CSV")  # fmt: skip
    tc.add_argument("--csv", help="file for --hex (default hex_COL_ROW_units.csv)")
    tc.set_defaults(func=cmd_topoclimate)

    bi = sub.add_parser("biomes", help="add or update v5.0 biomes, cover and cryosphere")
    bi.add_argument("project")
    bi.add_argument("--out", "-o", help="save to another directory instead of updating in place")
    bi.add_argument("--spinup", type=int, help="years of the snow / firn / ice spin-up (100)")
    bi.add_argument("--tau", type=float, help="softness of the PFT competition (0.08)")
    bi.add_argument("--no-paleo-ice", action="store_true",
                    help="no seeded paleoclimate ice (ice only from the spin-up)")  # fmt: skip
    bi.set_defaults(func=cmd_biomes)

    bx = sub.add_parser("biome", help="composition and reasons of one hex (v5.0)")
    bx.add_argument("project")
    bx.add_argument("col", type=int)
    bx.add_argument("row", type=int)
    bx.set_defaults(func=cmd_biome)

    un = sub.add_parser("units", help="terrain units of one hex (v4.7)")
    un.add_argument("project")
    un.add_argument("col", type=int)
    un.add_argument("row", type=int)
    un.add_argument("--csv", help="write CSV instead of printing")
    un.set_defaults(func=cmd_units)

    u = sub.add_parser("gui", help="open the viewer")
    u.add_argument("project", nargs="?", help="project directory to open")
    u.set_defaults(func=cmd_gui)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return args.func(args)
    except (
        ConfigError,
        ValueError,
        KeyError,
        LookupError,
        RuntimeError,
        ProjectIntegrityError,
        FileNotFoundError,
    ) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
