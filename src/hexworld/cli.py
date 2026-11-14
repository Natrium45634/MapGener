"""Command line interface (works without PySide6)."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from . import render
from .config import PRESETS, Config, ConfigError
from .io import ProjectIntegrityError, load_project, save_project
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
    if args.origin is not None:
        data["viewport_origin_km"] = args.origin
    if args.sim_bounds is not None:
        data["simulation_bounds_km"] = args.sim_bounds
    return Config.from_dict(data)


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
    for err in v.get("errors", []):
        lines.append(f"  ERROR   {err}")
    for warn in project.warnings:
        lines.append(f"  warning {warn}")
    return "\n".join(lines)


def cmd_generate(args) -> int:
    config = _config_from_args(args)
    last = [""]

    def progress(stage: str, fraction: float):
        if not args.quiet and stage != last[0]:
            last[0] = stage
            print(f"[{100 * fraction:5.1f}%] {stage}", file=sys.stderr)

    t0 = time.perf_counter()
    project = generate(config, progress=progress)
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
    return 0 if project.validation["ok"] else 2


def cmd_info(args) -> int:
    project = load_project(args.project)
    print(_summary(project))
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
    layers = render.LAYERS if args.layer == "all" else (args.layer,)
    for layer in layers:
        img = render.render_project(project, layer, px_per_hex=args.px_per_hex)
        out = (
            Path(args.out)
            if args.out and len(layers) == 1
            else (Path(args.out or args.project) / f"{layer}.png")
        )
        print(render.write_png(out, img))
    return 0


def cmd_hex(args) -> int:
    project = load_project(args.project)
    info = project.hex_info(args.col, args.row)
    info["plate"] = project.plate(info["plate_id"])
    print(json.dumps(info, indent=2, ensure_ascii=False))
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
    g.add_argument("--quiet", "-q", action="store_true")
    g.set_defaults(func=cmd_generate)

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
    r.add_argument("--layer", default="plates", choices=(*render.LAYERS, "all"))
    r.add_argument("--px-per-hex", type=float, help="display scale only; never changes data")
    r.add_argument("--out", "-o", help="PNG file (single layer) or directory")
    r.set_defaults(func=cmd_render)

    h = sub.add_parser("hex", help="print every value of one hex")
    h.add_argument("project")
    h.add_argument("col", type=int)
    h.add_argument("row", type=int)
    h.set_defaults(func=cmd_hex)

    u = sub.add_parser("gui", help="open the viewer")
    u.add_argument("project", nargs="?", help="project directory to open")
    u.set_defaults(func=cmd_gui)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return args.func(args)
    except (ConfigError, ValueError, ProjectIntegrityError, FileNotFoundError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
