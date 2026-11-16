"""Comparison of a generated world with reference values of the real Earth.

Every metric is computed from the project's own numbers and placed next to an Earth
reference with its source. The references are global statistics of Earth; a window of a
generated world is a region, so moderate differences are expected and the table is a
diagnostic, not a pass/fail test.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .project import Project


@dataclass
class Metric:
    group: str
    name: str
    value: float | None
    earth: str
    lo: float | None  # plausible range for a regional window (None = no check)
    hi: float | None
    source: str
    unit: str = ""

    @property
    def verdict(self) -> str:
        if self.value is None:
            return "нет данных"
        if self.lo is None or self.hi is None:
            return "справочно"
        return "в диапазоне" if self.lo <= self.value <= self.hi else "вне диапазона"

    def to_dict(self) -> dict:
        return {
            "group": self.group,
            "name": self.name,
            "value": self.value,
            "unit": self.unit,
            "earth": self.earth,
            "range": [self.lo, self.hi],
            "verdict": self.verdict,
            "source": self.source,
        }


#: Earth zonal-mean near-surface air temperature over land, degC (annual, rounded;
#: ERA5/CRU climatologies 1991-2020). Antarctica excluded from the polar band.
EARTH_LAND_T = {(0, 10): 26.0, (10, 20): 25.0, (20, 30): 21.0, (30, 40): 15.0,
                (40, 50): 7.0, (50, 60): 0.0, (60, 70): -7.0, (70, 80): -14.0}  # fmt: skip


def _streams(strahler: np.ndarray, receiver: np.ndarray):
    """Horton stream statistics: count and mean length (cells) of streams of each order."""
    s = strahler.ravel().astype(int)
    rc = receiver.ravel()
    river = s > 0
    counts, cells = {}, {}
    for w in range(1, int(s.max()) + 1 if river.any() else 1):
        of_w = river & (s == w)
        if not of_w.any():
            continue
        idx = np.nonzero(of_w)[0]
        down = rc[idx]
        ends = (down == idx) | (s[down] != w)  # the stream of order w stops here
        counts[w] = int(ends.sum())
        cells[w] = int(of_w.sum())
    return counts, cells


def _ratio(values: dict) -> float | None:
    orders = sorted(values)
    pairs = [
        values[a] / values[b] for a, b in zip(orders[:-1], orders[1:], strict=True) if values[b] > 0
    ]
    if not pairs:
        return None
    return float(np.exp(np.mean(np.log(pairs))))


def earth_comparison(project: Project) -> list[Metric]:
    if project.hydro is None:
        raise LookupError("the comparison needs v4 data (climate and rivers)")
    h = project.hydro
    L = h.layers
    s = h.summary
    sea = h.sea_level_m
    z = L["elevation_m"].astype(np.float64)
    land = L["land_mask"]
    area = project.layout.hex_area_km2
    rel = z - sea
    m: list[Metric] = []

    # -- hypsometry -------------------------------------------------------------------------
    m += [
        Metric("Рельеф", "доля суши в окне", float(land.mean()), "0,29", None, None,
               "площадь суши 149 млн км² из 510", ""),
        Metric("Рельеф", "средняя высота суши", float(rel[land].mean()) if land.any() else None,
               "≈ 840 м", 300.0, 1500.0, "ETOPO1 (Amante & Eakins 2009)", "м"),
        Metric("Рельеф", "доля суши выше 2000 м",
               float((rel[land] > 2000).mean()) if land.any() else None,
               "≈ 0,11", 0.02, 0.25, "ETOPO1, гипсометрическая кривая", ""),
        Metric("Рельеф", "средняя глубина моря",
               float(-rel[~land].mean()) if (~land).any() else None,
               "≈ 3700 м", 2000.0, 4800.0, "ETOPO1; Charette & Smith 2010", "м"),
    ]  # fmt: skip

    # -- climate ----------------------------------------------------------------------------
    t = L["temperature_mean_c"]
    lat = np.abs(L["latitude_deg"])
    for (a, b), ref in EARTH_LAND_T.items():
        band = land & (lat >= a) & (lat < b)
        if band.sum() >= 20:
            height = float(rel[band].mean())
            m.append(
                Metric(
                    "Климат",
                    f"температура суши {a}–{b}° (средняя высота {height:,.0f} м)",
                    float(t[band].mean()),
                    f"≈ {ref:+.0f} °C (суша Земли, ~0,3–0,7 км)",
                    ref - 7.0,
                    ref + 7.0,
                    "ERA5/CRU, зональные средние по суше; −6,5 °C на км выше",
                    "°C",
                )
            )
    a_sh = s["aridity_shares"]
    arid = a_sh["hyper_arid"] + a_sh["arid"]
    dry = arid + a_sh["semi_arid"] + a_sh["dry_subhumid"]
    m += [
        Metric("Климат", "средние осадки на суше", s["land_mean_precipitation_mm_yr"],
               "≈ 800 мм/год (без Антарктиды)", 450.0, 1300.0,
               "GPCC; Trenberth et al. 2007", "мм/год"),
        Metric("Климат", "аридные земли (P/PET < 0,2)", float(arid), "≈ 0,20", 0.08, 0.35,
               "UNEP 1992; Millennium Ecosystem Assessment 2005", ""),
        Metric("Климат", "все засушливые земли (P/PET < 0,65)", float(dry), "≈ 0,41",
               0.2, 0.6, "Millennium Ecosystem Assessment 2005", ""),
        Metric("Климат", "коэффициент стока (сток / осадки)", s["runoff_ratio"], "≈ 0,36–0,40",
               0.25, 0.55, "Trenberth et al. 2007; Oki & Kanae 2006", ""),
    ]  # fmt: skip

    # -- hydrology --------------------------------------------------------------------------
    m += [
        Metric("Гидрология", "доля суши без стока в море", s["no_outflow_share_of_land"],
               "≈ 0,18–0,20", 0.05, 0.35, "бессточные области: Shiklomanov 1993; HydroSHEDS",
               ""),
        Metric("Гидрология", "доля суши под озёрами", s["lake_share_of_land"], "≈ 0,018",
               0.003, 0.04, "Verpoorter et al. 2014 (озёра > 0,002 км²)", ""),
    ]  # fmt: skip
    systems = [r for r in h.river_systems if r["drainage_area_km2"] >= 10 * area]
    if len(systems) >= 8:
        a_ = np.log([r["drainage_area_km2"] for r in systems])
        l_ = np.log([max(r["main_stem_length_km"], 1e-6) for r in systems])
        hack = float(np.polyfit(a_, l_, 1)[0])
    else:
        hack = None
    counts, cells = _streams(L["strahler_order"], L["flow_receiver"])
    lengths = {w: cells[w] / counts[w] for w in counts if counts[w]}
    m += [
        Metric("Реки", "показатель Хака h (L ∝ A^h)", hack, "0,5–0,6", 0.45, 0.7,
               "Hack 1957; Rigon et al. 1996", ""),
        Metric("Реки", "отношение ветвления Хортона R_b", _ratio(counts), "3–5", 2.5, 5.5,
               "Horton 1945; Strahler 1957", ""),
        Metric("Реки", "отношение длин Хортона R_l",
               (1.0 / _ratio(lengths)) if _ratio(lengths) else None, "1,5–3,5", 1.2, 4.0,
               "Horton 1945", ""),
        Metric("Реки", "наибольший порядок Стралера", float(s["max_strahler"]),
               "Амазонка 12 (на картах 1:1 млн), на сетке 75 км — 5–7", None, None,
               "порядок зависит от разрешения", ""),
    ]  # fmt: skip

    # -- erosion ----------------------------------------------------------------------------
    m.append(
        Metric("Эрозия", "средний размыв суши", s["mean_denudation_m_ma"], "≈ 20–60 м/млн лет",
               15.0, 90.0, "Milliman & Farnsworth 2011; Wilkinson & McElroy 2007", "м/млн лет")
    )  # fmt: skip
    return m


def report_markdown(project: Project, metrics: list[Metric] | None = None) -> str:
    metrics = earth_comparison(project) if metrics is None else metrics
    c = project.config
    lines = [
        f"Сид `{c.seed}`, {c.width}×{c.height} гексов по {c.hex_spacing_km:g} км, режим "
        f"{c.mode}; климат `{c.climate.profile}`, широта центра "
        f"{project.hydro.latitude['center_deg']:+.0f}°.",
        "",
        "| Группа | Показатель | Мир | Земля | Оценка | Источник |",
        "|---|---|---|---|---|---|",
    ]
    for x in metrics:
        if x.value is None:
            value = "—"
        elif abs(x.value) >= 100:
            value = f"{x.value:,.0f} {x.unit}".strip()
        else:
            value = f"{x.value:.2f} {x.unit}".strip()
        lines.append(f"| {x.group} | {x.name} | {value} | {x.earth} | {x.verdict} | {x.source} |")
    inside = sum(1 for x in metrics if x.verdict == "в диапазоне")
    checked = sum(1 for x in metrics if x.verdict in ("в диапазоне", "вне диапазона"))
    lines += ["", f"В правдоподобном диапазоне: {inside} из {checked} проверяемых показателей."]
    return "\n".join(lines)
