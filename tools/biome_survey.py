"""Biome balance survey: generate several worlds and report the share of every land biome.

Usage::

    python tools/biome_survey.py                      # the 12 worlds of docs/biome_balance.md
    python tools/biome_survey.py --world Ab12:30 --world Br1:0 -o survey.json

Each world is 240x160 hexes of 40 km, mode "world", default settings; the number after the
colon is the latitude of the map centre. Shares are of the land area (whole-hex fractions).
"""

from __future__ import annotations

import argparse
import json
import sys

import numpy as np

from hexworld import Config, generate
from hexworld.biomes.registry import BIOMES

DEFAULT = ("Ab12:30", "Terra7:30", "Kx9:30", "Mo42:30", "Zed5:0", "Br1:0", "Cq3:0", "Dw4:0",
           "Ex5:50", "Fy6:50", "Gz7:-30", "Hq8:15")  # fmt: skip


def survey(seed: str, lat: float) -> dict:
    d = Config().to_dict()
    d.update(seed=seed, width=240, height=160, hex_spacing_km=40.0, mode="world")
    d["climate"]["latitude_center_deg"] = lat
    p = generate(Config.from_dict(d))
    F = p.biomes.fractions
    area = np.bincount(F["biome"].astype(int), weights=F["fraction"], minlength=len(BIOMES))
    hs = p.hydro.summary
    return {
        "seed": seed,
        "lat": lat,
        "area": area.round(4).tolist(),
        "aridity": hs["aridity_shares"],
        "land_p_mm": hs["land_mean_precipitation_mm_yr"],
        "land_t_c": hs["land_mean_temperature_c"],
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--world", action="append", help="SEED:LATITUDE (repeatable)")
    ap.add_argument("-o", "--out", help="write the raw results as JSON")
    args = ap.parse_args(argv)
    worlds = [w.split(":") for w in (args.world or DEFAULT)]
    rows = []
    for seed, lat in worlds:
        print(f"generating {seed} at {lat} deg ...", file=sys.stderr)
        rows.append(survey(seed, float(lat)))
    land = [i for i, b in enumerate(BIOMES) if b.family not in ("marine", "freshwater", "special")]
    A = np.array([r["area"] for r in rows])
    share = A[:, land] / A[:, land].sum(axis=1, keepdims=True)
    present = (A[:, land] > 0.5).sum(axis=0)
    print(f"{'biome':40s} {'mean %':>7s} {'max %':>7s} {'worlds':>6s}")
    for j in np.argsort(-share.mean(axis=0)):
        b = BIOMES[land[j]]
        print(f"{b.ru[:40]:40s} {100 * share[:, j].mean():7.2f} {100 * share[:, j].max():7.2f} "
              f"{present[j]:6d}")  # fmt: skip
    never = [BIOMES[land[j]].ru for j in range(len(land)) if present[j] == 0]
    print(f"\nclasses seen: {len(land) - len(never)} of {len(land)}; never: {never}")
    if args.out:
        with open(args.out, "w", encoding="utf-8") as fh:
            json.dump(rows, fh, ensure_ascii=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
