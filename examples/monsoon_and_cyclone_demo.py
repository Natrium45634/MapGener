"""TZ 5.4 demo: monsoon diagnostics, storm climatology and one synthetic weather year.

    python examples/monsoon_and_cyclone_demo.py [seed] [out_dir]

Generates a small world with the v5.4 atmosphere and a weather year, prints the rain
regimes, the storm climatology and the year's events, and writes PNG maps of the
«Атмосфера и сезоны» view (July normals, monsoon suitability, storm tracks).
"""

from __future__ import annotations

import sys
from dataclasses import replace
from pathlib import Path

from hexworld import Config, generate
from hexworld.views import render_view_image


def main(seed: str = "Terra7", out: str = "monsoon_demo") -> None:
    cfg = Config(seed=seed, width=160, height=110, hex_spacing_km=60.0)
    cfg = replace(cfg, atmosphere=replace(cfg.atmosphere, weather_year=True))
    project = generate(cfg)
    a = project.atmosphere
    s = a.summary
    print("domain:", s["quality"]["domain"], "| hadley edge:", round(s["hadley_edge_deg"], 1))
    print("rain regimes (land share):")
    for name, share in s["regime_shares_land"].items():
        print(f"  {name:28s} {100 * share:5.1f} %")
    st = s["storms"]
    print(f"tropical cyclones: {st['tc_genesis_per_yr']:.1f} geneses / yr (climatology)")
    wy = s["weather_year"]
    print("weather year", wy["year_seed"], wy["year"], "events:", wy["event_counts"])
    print("interannual:", wy["interannual"]["mode"])
    out_dir = Path(out)
    out_dir.mkdir(parents=True, exist_ok=True)
    try:
        from PIL import Image
    except ImportError:
        print("Pillow is not installed: no PNG written")
        return
    for mode, overlays in (("pressure", {"month:6", "coast"}), ("monsoon", {"coast"}),
                           ("weather", {"coast"}), ("tc", {"coast"})):  # fmt: skip
        img = render_view_image(project, "atmosphere_seasons", mode, overlays)
        Image.fromarray(img).save(out_dir / f"atmosphere_{mode}.png")
    print("maps in", out_dir.resolve())


if __name__ == "__main__":
    main(*sys.argv[1:3])
