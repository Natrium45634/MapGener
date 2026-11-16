"""Compare generated worlds with the real Earth (hypsometry, climate, rivers, erosion).

Usage::

    python tools/realism_report.py PROJECT_DIR [...]          # saved projects
    python tools/realism_report.py --seed Terra7 --seed Ab12   # generate Terra7-like worlds
    python tools/realism_report.py --seed Terra7 -o docs/realism_v4.md

Generated worlds use the 360x240, 75 km "world" setting of the documentation examples.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from hexworld import Config, generate, load_project
from hexworld.realism import earth_comparison, report_markdown


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("projects", nargs="*", help="saved project directories")
    parser.add_argument("--seed", action="append", default=[], help="generate this seed")
    parser.add_argument("--width", type=int, default=360)
    parser.add_argument("--height", type=int, default=240)
    parser.add_argument("--hex-km", type=float, default=75.0)
    parser.add_argument("--duration-ma", type=float, default=30.0)
    parser.add_argument("--latitude", type=float, default=30.0)
    parser.add_argument("-o", "--out", help="write the Markdown report here")
    args = parser.parse_args(argv)
    if not args.projects and not args.seed:
        parser.error("give project directories or --seed")
    sections = ["# Сравнение с Землёй", ""]
    worlds = [(p, load_project(p)) for p in args.projects]
    for seed in args.seed:
        cfg = Config.from_dict(
            {
                "seed": seed,
                "width": args.width,
                "height": args.height,
                "hex_spacing_km": args.hex_km,
                "mode": "world",
                "history": {"duration_ma": args.duration_ma},
                "climate": {"latitude_center_deg": args.latitude},
            }
        )
        print(f"generating {seed} ...", file=sys.stderr)
        worlds.append((seed, generate(cfg)))
    for name, project in worlds:
        metrics = earth_comparison(project)
        sections += [f"## {name}", "", report_markdown(project, metrics), ""]
    text = "\n".join(sections)
    if args.out:
        Path(args.out).write_text(text, encoding="utf-8")
        print(args.out)
    else:
        print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
