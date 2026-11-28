"""Progress of a generation: ordered stages, measured weights, sub-progress and ETA.

The pipeline reports ``(stage, fraction)``; the fractions were laid out by hand long ago and
do not follow the time the stages take (the v1 stages take 80 % of the bar, the history and
the climate most of the time). :class:`ProgressTracker` keeps its own ordered list of stages
with weights measured on a 1000x800 world, turns every report into an honest overall
fraction, and estimates the remaining time from the speed seen so far. Stages that report
inner progress (history, v3 raster, erosion, topoclimate, biomes) move the bar inside their
weight; the others are interpolated by the expected duration (never past the stage end).
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field

#: (key, title, (seconds fixed, seconds per million hexes), stage window or None) - measured:
#: the world-raster stages (history, macro relief) cost about the same on every map size
STAGES = (
    ("plates", "плиты", (0.3, 0.5), None),
    ("crust", "кора", (0.3, 0.5), None),
    ("motion", "движение плит", (0.2, 0.3), None),
    ("raster", "выборка на гексы", (0.1, 2.0), None),
    ("boundary_edges", "границы плит", (0.1, 1.0), None),
    ("tectonic_fields", "поля тектоники", (0.1, 1.0), None),
    ("assemble", "сборка", (0.1, 0.5), None),
    ("world_tectonics", "мировой каркас", (1.0, 1.0), None),
    ("continental_blocks", "континентальные блоки", (1.5, 2.0), None),
    ("macro_elevation", "макрорельеф", (3.0, 2.0), None),
    ("v2_raster", "рельеф на гексах", (0.2, 2.0), None),
    ("sea_level_landmasses", "уровень моря и суша", (0.2, 2.0), None),
    ("geo_history", "геологическая история", (55.0, 5.0), (0.80, 0.88)),
    ("mountain_plan", "план горных систем", (2.0, 2.0), None),
    ("v3_raster", "итоговый рельеф", (1.0, 20.0), (0.89, 0.96)),
    ("v3_sea", "суша итоговой эпохи", (0.2, 2.0), None),
    ("v46_terrain", "дно океана, бассейны, острова", (2.0, 25.0), None),
    ("v4_climate", "климат мира, океан и морской лёд", (45.0, 20.0), None),
    ("v4_evolution", "климат ↔ сток ↔ эрозия", (3.0, 55.0), (0.988, 0.996)),
    ("v4_network", "реки, озёра, бассейны", (0.5, 10.0), None),
    ("v45_months", "12 месяцев климата", (0.2, 3.0), None),
    ("v45_soils", "почвы", (0.2, 2.0), None),
    ("v45_water", "водный баланс по месяцам", (0.5, 11.0), None),
    ("v45_wetness", "увлажнение и индексы", (0.2, 2.0), None),
    # the ice-sheet spin-up: seconds without ice, minutes on a large icy world (it reports
    # how far it got)
    ("v53_earth", "океан, морской лёд, ледниковые щиты, ветры на высотах", (5.0, 150.0), None),
    ("v54_atmosphere", "атмосфера: муссоны, штормы, сезоны", (0.3, 6.0), None),
    ("v47_topoclimate", "высотные пояса и топоклимат", (0.5, 80.0), (0.0, 1.0)),
    ("v51_volcanism", "вулканизм: извержения и продукты", (4.0, 10.0), (0.0, 1.0)),
    ("v50_biomes", "биомы, снег и ледники", (0.3, 50.0), (0.0, 1.0)),
    ("v52_resources", "плодородие почв и полезные ископаемые", (1.0, 30.0), (0.0, 1.0)),
    ("v551_minerals", "недра: рудные тела, россыпи, нефть и газ, воды", (5.0, 20.0), None),
    ("done", "готово", (0.0, 0.0), None),
)
_ORDER = {k: i for i, (k, *_r) in enumerate(STAGES)}
TITLES = {k: t for k, t, *_r in STAGES}
#: which stages a generation runs, by its ``through``
_LAST = {"v1": "assemble", "v2": "sea_level_landmasses", "v3": "v3_sea", "v46": "v46_terrain",
         "v4": "v4_network", "v45": "v45_wetness", "v47": "v47_topoclimate",
         "v51": "v51_volcanism", "v50": "v551_minerals"}  # fmt: skip


def sub_report(stage: str, fraction: float) -> str:
    """Encode inner progress of a stage: ``"v50_biomes@0.42"`` (the tracker decodes it)."""
    return f"{stage}@{max(0.0, min(1.0, float(fraction))):.4f}"


@dataclass
class ProgressState:
    overall: float  # 0..1
    stage: str
    title: str
    stage_fraction: float  # 0..1 inside the current stage (reported or estimated)
    estimated: bool  # stage_fraction is an estimate (the stage reports no inner progress)
    elapsed_s: float
    stage_elapsed_s: float
    eta_s: float | None
    done: list = field(default_factory=list)  # [(key, title, seconds)]
    pending: list = field(default_factory=list)  # [(key, title)]


class ProgressTracker:
    """Turns ``(stage, fraction)`` reports into an overall fraction, a stage list and an ETA."""

    def __init__(self, through: str = "v50", hexes: int = 10_000, clock=time.monotonic):
        last = _ORDER.get(_LAST.get(through, "v551_minerals"), len(STAGES) - 2)
        self.keys = [k for k, *_r in STAGES[: last + 1]] + ["done"]
        self.keys = list(dict.fromkeys(self.keys))
        mhex = max(hexes, 0) / 1e6
        self.weights = {k: a + b * mhex for k, _t, (a, b), _win in STAGES}
        self.windows = {k: win for k, _t, _w, win in STAGES}
        self.total = sum(self.weights[k] for k in self.keys) or 1.0
        self.clock = clock
        self.t0 = clock()
        self.current = self.keys[0]
        self.stage_t0 = self.t0
        self.sub = 0.0
        self.sub_known = False
        self.finished: dict[str, float] = {}
        self._overall = 0.0

    def _weight_before(self, key: str) -> float:
        i = self.keys.index(key) if key in self.keys else 0
        return sum(self.weights[k] for k in self.keys[:i])

    def report(self, stage: str, fraction: float = 0.0) -> ProgressState:
        now = self.clock()
        key, _, sub = stage.partition("@")
        if key not in self.keys:  # "validate" (after several stages) and unknown names
            # an unknown or skipped stage: keep the current one ticking
            key = self.current
        if key != self.current and self.keys.index(key) > self.keys.index(self.current):
            # the stages in between are done (reused from the cache or very fast)
            for k in self.keys[self.keys.index(self.current) : self.keys.index(key)]:
                self.finished.setdefault(k, now - self.stage_t0 if k == self.current else 0.0)
            self.current, self.stage_t0, self.sub, self.sub_known = key, now, 0.0, False
        if sub:
            self.sub, self.sub_known = float(sub), True
        else:
            win = self.windows.get(key)
            if win is not None and win[1] > win[0] and win[0] <= fraction <= win[1] + 1e-9:
                self.sub = (fraction - win[0]) / (win[1] - win[0])
                self.sub_known = self.sub > 0.0 or self.sub_known
        if key == "done":
            self.finished.setdefault(self.current, now - self.stage_t0)
        return self.state(now)

    def state(self, now: float | None = None) -> ProgressState:
        now = self.clock() if now is None else now
        elapsed = now - self.t0
        stage_elapsed = now - self.stage_t0
        before = self._weight_before(self.current)
        w = self.weights.get(self.current, 0.0)
        # seconds per weight unit (a weight is a reference second): measured once some work
        # is done; before that the reference machine
        done_w = before
        rate = elapsed / done_w if done_w > 0.05 * self.total and elapsed > 1.0 else 1.0
        expected_stage = w * rate
        if self.sub_known:
            sub = self.sub
            estimated = False
        else:
            # no inner reports: approach (never reach) the end of the stage by its duration
            sub = 1.0 - math.exp(-stage_elapsed / max(expected_stage, 1e-3)) if w > 0 else 0.0
            sub = min(sub, 0.95)
            estimated = True
        overall = 1.0 if self.current == "done" else min((before + w * sub) / self.total, 0.999)
        overall = max(overall, self._overall)  # never goes back
        self._overall = overall
        eta = None
        if self.current == "done":
            eta = 0.0
        elif overall > 0.02 and elapsed > 2.0:
            eta = max(elapsed * (1.0 - overall) / overall, 0.0)
        i = self.keys.index(self.current)
        return ProgressState(
            overall=overall,
            stage=self.current,
            title=TITLES.get(self.current, self.current),
            stage_fraction=sub,
            estimated=estimated,
            elapsed_s=elapsed,
            stage_elapsed_s=stage_elapsed,
            eta_s=eta,
            done=[(k, TITLES.get(k, k), self.finished.get(k, 0.0)) for k in self.keys[:i]],
            pending=[(k, TITLES.get(k, k)) for k in self.keys[i + 1 :] if k != "done"],
        )


def format_seconds(s: float | None) -> str:
    if s is None:
        return "—"
    s = int(round(s))
    if s < 60:
        return f"{s} с"
    m, s = divmod(s, 60)
    if m < 60:
        return f"{m} мин {s:02d} с"
    h, m = divmod(m, 60)
    return f"{h} ч {m:02d} мин"


def expected_seconds(through: str = "v50", hexes: int = 10_000) -> float:
    """Expected generation time on the reference machine, s."""
    t = ProgressTracker(through, hexes)
    return float(t.total)
