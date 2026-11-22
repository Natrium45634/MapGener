"""World settings window: seed, every generator parameter, locks, presets, size, start.

Every row: a slider (min..max), an exact value, and a lock. A locked parameter keeps its
value when the settings are randomised. The schema lives in :mod:`hexworld.settings_schema`;
this module only draws it.
"""

from __future__ import annotations

import json
import math

from PySide6.QtCore import QRegularExpression, QSettings, Qt, Signal
from PySide6.QtGui import QRegularExpressionValidator
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDoubleSpinBox,
    QFileDialog,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QSlider,
    QSpinBox,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from .. import settings_schema as ss
from ..config import Config
from ..progress import format_seconds

THROUGH = (
    ("v50", "Всё: биомы, снег и ледники (v5.0)"),
    ("v47", "До высотных поясов (v4.7)"),
    ("v45", "До почв и сезонов (v4.5)"),
    ("v4", "До климата, рек и эрозии (v4)"),
    ("v3", "До рельефа и геоистории (v3)"),
    ("v2", "Только материки и высоты (v2)"),
    ("v1", "Только плиты (v1)"),
)
_STEPS = 1000


class _Row:
    """Widgets of one parameter and the conversions slider <-> value."""

    def __init__(self, p: ss.Param, on_change):
        self.p = p
        self.on_change = on_change
        self.label = QLabel(("⚙ " if p.technical else "") + p.label)
        tip = p.help or p.label
        tip += f"\n[{p.path}]"
        if p.unit:
            tip += f", {p.unit}"
        if p.rand is not None and p.kind in ("float", "int"):
            tip += f"\nслучайный выбор: {p.rand[0]:g} … {p.rand[1]:g}"
        self.label.setToolTip(tip)
        self.slider = None
        self.auto = None
        self._busy = False
        if p.kind == "choice":
            self.edit = QComboBox()
            for v, title in p.choices:
                self.edit.addItem(title, v)
            self.edit.currentIndexChanged.connect(lambda _i: self.on_change())
        elif p.kind == "bool":
            self.edit = QCheckBox("включено")
            self.edit.toggled.connect(lambda _b: self.on_change())
        else:
            self.slider = QSlider(Qt.Orientation.Horizontal)
            self.slider.setRange(0, _STEPS)
            self.slider.setMinimumWidth(220)
            if p.kind == "int":
                self.edit = QSpinBox()
                self.edit.setRange(int(p.lo), int(p.hi))
            else:
                self.edit = QDoubleSpinBox()
                self.edit.setDecimals(p.decimals)
                self.edit.setRange(p.lo, p.hi)
                self.edit.setSingleStep(_nice_step(p))
            if p.unit:
                self.edit.setSuffix(f" {p.unit}")
            self.edit.setMinimumWidth(130)
            self.slider.valueChanged.connect(self._from_slider)
            self.edit.valueChanged.connect(self._from_edit)
            if p.auto:
                self.auto = QCheckBox("авто")
                self.auto.setToolTip("Значение выбирает генератор (из сида и размера мира)")
                self.auto.toggled.connect(self._auto_toggled)
        self.edit.setToolTip(tip)
        self.lock = QCheckBox("🔒")
        self.lock.setToolTip("Зафиксировать: не менять при «Случайные настройки»")

    # -- slider mapping --------------------------------------------------------------------------
    def _to_pos(self, v: float) -> int:
        p = self.p
        lo, hi = float(p.lo), float(p.hi)
        if p.log and lo > 0:
            t = (math.log(max(v, lo)) - math.log(lo)) / (math.log(hi) - math.log(lo))
        else:
            t = (float(v) - lo) / max(hi - lo, 1e-12)
        return int(round(max(0.0, min(1.0, t)) * _STEPS))

    def _from_pos(self, pos: int) -> float:
        p = self.p
        t = pos / _STEPS
        lo, hi = float(p.lo), float(p.hi)
        if p.log and lo > 0:
            return math.exp(math.log(lo) + t * (math.log(hi) - math.log(lo)))
        return lo + t * (hi - lo)

    def _from_slider(self, pos: int):
        if self._busy:
            return
        self._busy = True
        v = self._from_pos(pos)
        self.edit.setValue(int(round(v)) if self.p.kind == "int" else v)
        self._busy = False
        self.on_change()

    def _from_edit(self, v):
        if self._busy:
            return
        self._busy = True
        self.slider.setValue(self._to_pos(v))
        self._busy = False
        self.on_change()

    def _auto_toggled(self, on: bool):
        self.slider.setEnabled(not on)
        self.edit.setEnabled(not on)
        self.on_change()

    # -- value ----
    def get(self):
        p = self.p
        if p.kind == "choice":
            return self.edit.currentData()
        if p.kind == "bool":
            return self.edit.isChecked()
        if self.auto is not None and self.auto.isChecked():
            return None
        return int(self.edit.value()) if p.kind == "int" else float(self.edit.value())

    def set(self, v):
        p = self.p
        self._busy = True
        try:
            if p.kind == "choice":
                i = self.edit.findData(v)
                self.edit.setCurrentIndex(max(i, 0))
            elif p.kind == "bool":
                self.edit.setChecked(bool(v))
            else:
                if self.auto is not None:
                    self.auto.setChecked(v is None)
                    self.slider.setEnabled(v is not None)
                    self.edit.setEnabled(v is not None)
                if v is None:
                    v = p.rand[0] if p.rand else p.lo
                v = max(p.lo, min(p.hi, v))
                self.edit.setValue(int(round(v)) if p.kind == "int" else float(v))
                self.slider.setValue(self._to_pos(v))
        finally:
            self._busy = False


def _nice_step(p: ss.Param) -> float:
    span = float(p.hi) - float(p.lo)
    step = 10 ** math.floor(math.log10(max(span, 1e-9) / 100.0))
    return max(step, 10 ** (-p.decimals))


class WorldSettingsDialog(QDialog):
    """Before a generation: every setting, with locks, presets and the size of the world.

    After ``exec()``: ``self.config`` (a :class:`Config`), ``self.through`` and
    ``self.start`` (True when a start button was pressed).
    """

    changed = Signal()

    def __init__(self, config: Config | None = None, through: str = "v50", parent=None):
        super().__init__(parent)
        self.setWindowTitle("Новый мир — настройки генерации")
        self.resize(1080, 820)
        self.config: Config | None = None
        self.through = through
        self.start = False
        self.rows: dict[str, _Row] = {}
        base = config or Config()
        lay = QVBoxLayout(self)

        # -- seed ------------------------------------------------------------------------------
        top = QHBoxLayout()
        top.addWidget(QLabel("<b>Сид мира:</b>"))
        self.seed = QLineEdit(base.seed)
        self.seed.setValidator(
            QRegularExpressionValidator(QRegularExpression("[A-Za-z0-9]{1,64}"), self)
        )
        self.seed.setToolTip("Буквы и цифры; регистр важен. Один сид + одни настройки = один мир.")
        self.seed.textEdited.connect(lambda _t: self.seed_lock.setChecked(True))
        self.seed.textChanged.connect(lambda _t: self._update_footer())
        top.addWidget(self.seed, 1)
        dice = QPushButton("🎲 Случайный сид")
        dice.clicked.connect(self.new_seed)
        top.addWidget(dice)
        self.seed_lock = QCheckBox("🔒 сид")
        self.seed_lock.setToolTip("Не менять сид при «Случайные настройки»")
        top.addWidget(self.seed_lock)
        lay.addLayout(top)

        # -- presets and files ----------------------------------------------------------------
        pre = QHBoxLayout()
        pre.addWidget(QLabel("Пресет мира:"))
        self.preset = QComboBox()
        self.preset.addItems(list(ss.WORLD_PRESETS))
        pre.addWidget(self.preset)
        apply_btn = QPushButton("Применить")
        apply_btn.setToolTip("Меняет только параметры пресета; замки не трогает")
        apply_btn.clicked.connect(self.apply_preset)
        pre.addWidget(apply_btn)
        pre.addStretch(1)
        unlock = QPushButton("Снять все замки")
        unlock.clicked.connect(lambda: self._set_all_locks(False, keep_technical=True))
        lock_all = QPushButton("Закрепить всё")
        lock_all.clicked.connect(lambda: self._set_all_locks(True))
        load_btn = QPushButton("Загрузить…")
        load_btn.clicked.connect(self.load_file)
        save_btn = QPushButton("Сохранить…")
        save_btn.clicked.connect(self.save_file)
        for b in (unlock, lock_all, load_btn, save_btn):
            pre.addWidget(b)
        lay.addLayout(pre)

        # -- tabs ----
        self.tabs = QTabWidget()
        lay.addWidget(self.tabs, 1)
        for gid, gtitle in ss.GROUPS:
            params = [p for p in ss.PARAMS if p.group == gid]
            if not params and gid != "world":
                continue
            page = QWidget()
            grid = QGridLayout(page)
            grid.setColumnStretch(1, 1)
            r = 0
            if gid == "world":
                r = self._build_size_rows(grid)
            for p in params:
                row = _Row(p, self._on_change)
                self.rows[p.path] = row
                grid.addWidget(row.label, r, 0)
                if row.slider is not None:
                    grid.addWidget(row.slider, r, 1)
                grid.addWidget(row.edit, r, 2)
                if row.auto is not None:
                    grid.addWidget(row.auto, r, 3)
                grid.addWidget(row.lock, r, 4)
                r += 1
            grid.setRowStretch(r, 1)
            scroll = QScrollArea()
            scroll.setWidgetResizable(True)
            scroll.setWidget(page)
            self.tabs.addTab(scroll, gtitle)

        # -- footer ------------------------------------------------------------------------------
        self.footer = QLabel("")
        self.footer.setWordWrap(True)
        lay.addWidget(self.footer)
        bottom = QHBoxLayout()
        reset = QPushButton("По умолчанию")
        reset.clicked.connect(self.reset_defaults)
        cancel = QPushButton("Отмена")
        cancel.clicked.connect(self.reject)
        bottom.addWidget(reset)
        bottom.addWidget(cancel)
        bottom.addStretch(1)
        self.rand_btn = QPushButton("🎲 Случайные настройки")
        self.rand_btn.clicked.connect(self.randomize)
        self.rand_start_btn = QPushButton("🎲 Случайно и старт")
        self.rand_start_btn.clicked.connect(self.randomize_and_start)
        self.start_btn = QPushButton("▶ Старт")
        self.start_btn.setDefault(True)
        self.start_btn.clicked.connect(self.accept_start)
        for b in (self.rand_btn, self.rand_start_btn, self.start_btn):
            bottom.addWidget(b)
        lay.addLayout(bottom)

        self.set_values(ss.values_from_config(base))
        self._set_locks(ss.default_locked())
        self._restore_locks()
        self._sync_size_preset()
        self._update_footer()

    # -- size --------------------------------------------------------------------------------------
    def _build_size_rows(self, grid: QGridLayout) -> int:
        grid.addWidget(QLabel("<b>Размер карты</b>"), 0, 0)
        self.size_preset = QComboBox()
        for title, *_rest in ss.SIZE_PRESETS:
            self.size_preset.addItem(title)
        self.size_preset.addItem("Свой размер (ниже)")
        self.size_preset.currentIndexChanged.connect(self._size_chosen)
        grid.addWidget(self.size_preset, 0, 1, 1, 2)
        grid.addWidget(QLabel("<b>Что считать</b>"), 1, 0)
        self.through_box = QComboBox()
        for v, title in THROUGH:
            self.through_box.addItem(title, v)
        self.through_box.setCurrentIndex(max(self.through_box.findData(self.through), 0))
        self.through_box.currentIndexChanged.connect(lambda _i: self._update_footer())
        grid.addWidget(self.through_box, 1, 1, 1, 2)
        return 2

    def _size_chosen(self, i: int):
        if i < len(ss.SIZE_PRESETS):
            _t, w, h, km, mode = ss.SIZE_PRESETS[i]
            for path, v in (("width", w), ("height", h), ("hex_spacing_km", km), ("mode", mode)):
                self.rows[path].set(v)
        self._update_footer()

    def _sync_size_preset(self):
        v = self.values()
        for i, (_t, w, h, km, mode) in enumerate(ss.SIZE_PRESETS):
            if (v["width"], v["height"], v["hex_spacing_km"], v["mode"]) == (w, h, km, mode):
                self.size_preset.blockSignals(True)
                self.size_preset.setCurrentIndex(i)
                self.size_preset.blockSignals(False)
                return
        self.size_preset.blockSignals(True)
        self.size_preset.setCurrentIndex(len(ss.SIZE_PRESETS))
        self.size_preset.blockSignals(False)

    # -- values ----
    def values(self) -> dict:
        return {path: row.get() for path, row in self.rows.items()}

    def set_values(self, values: dict):
        for path, v in values.items():
            if path in self.rows:
                self.rows[path].set(v)

    def locked(self) -> set:
        return {path for path, row in self.rows.items() if row.lock.isChecked()}

    def _set_locks(self, paths):
        for path, row in self.rows.items():
            row.lock.setChecked(path in paths)

    def _set_all_locks(self, on: bool, keep_technical: bool = False):
        tech = ss.default_locked()
        for path, row in self.rows.items():
            row.lock.setChecked(on or (keep_technical and path in tech))

    def _on_change(self):
        if hasattr(self, "size_preset"):
            self._sync_size_preset()
        self._update_footer()

    def _update_footer(self):
        if not hasattr(self, "footer"):
            return
        v = self.values()
        through = self.through_box.currentData() if hasattr(self, "through_box") else "v50"
        n = int(v["width"]) * int(v["height"])
        est = ss.estimate_seconds(v["width"], v["height"], through)
        mem = 0.4 + 6.5 * n / 800_000.0
        errors = ss.validate_values(v)
        if not self.seed.text():
            errors.insert(0, "введите сид (буквы и цифры)")
        text = (
            f"Мир: {v['width']}×{v['height']} = {n:,} гексов по {v['hex_spacing_km']:g} км · "
            f"примерно {format_seconds(est)} и до {mem:.1f} ГБ памяти · "
            f"закреплено {len(self.locked())} из {len(self.rows)} параметров"
        ).replace(",", " ")
        if errors:
            text += f"<br><span style='color:#b00'>Ошибка: {errors[0]}</span>"
        self.footer.setText(text)
        ok = not errors
        for b in (self.start_btn, self.rand_start_btn):
            b.setEnabled(ok)

    # -- actions ----
    def new_seed(self):
        self.seed.setText(ss.random_seed())

    def apply_preset(self):
        self.set_values(ss.WORLD_PRESETS[self.preset.currentText()])
        self._on_change()

    def randomize(self):
        if not self.seed_lock.isChecked():
            self.new_seed()
        self.set_values(ss.randomize(self.values(), self.locked()))
        self._on_change()

    def randomize_and_start(self):
        self.randomize()
        self.accept_start()

    def reset_defaults(self):
        self.set_values(ss.values_from_config(Config(seed=self.seed.text() or "HexWorld1")))
        self._on_change()

    def accept_start(self):
        try:
            self.config = ss.config_from_values(self.values(), seed=self.seed.text())
        except Exception as exc:  # noqa: BLE001 - shown to the user
            QMessageBox.warning(self, "Настройки", str(exc))
            return
        self.through = self.through_box.currentData()
        self.start = True
        self._remember_locks()
        self.accept()

    # -- files and memory ----
    def settings_document(self) -> dict:
        return {
            "format": "hexworld-world-settings-1",
            "seed": self.seed.text(),
            "through": self.through_box.currentData(),
            "values": self.values(),
            "locked": sorted(self.locked()),
        }

    def apply_document(self, doc: dict):
        if doc.get("seed"):
            self.seed.setText(doc["seed"])
        self.set_values(doc.get("values", {}))
        if "locked" in doc:
            self._set_locks(set(doc["locked"]))
        if doc.get("through"):
            self.through_box.setCurrentIndex(max(self.through_box.findData(doc["through"]), 0))
        self._on_change()

    def save_file(self):
        path, _ = QFileDialog.getSaveFileName(
            self, "Сохранить настройки мира", "world_settings.json", "JSON (*.json)"
        )
        if path:
            with open(path, "w", encoding="utf-8") as fh:
                json.dump(self.settings_document(), fh, ensure_ascii=False, indent=1)

    def load_file(self):
        path, _ = QFileDialog.getOpenFileName(self, "Загрузить настройки мира", "", "JSON (*.json)")
        if not path:
            return
        try:
            with open(path, encoding="utf-8") as fh:
                self.apply_document(json.load(fh))
        except (OSError, ValueError) as exc:
            QMessageBox.warning(self, "Настройки", f"Не удалось прочитать файл: {exc}")

    def _remember_locks(self):
        try:
            QSettings("HexWorld", "HexWorld").setValue(
                "world_settings/locked", json.dumps(sorted(self.locked()))
            )
        except Exception:  # noqa: BLE001 - convenience only
            pass

    def _restore_locks(self):
        try:
            raw = QSettings("HexWorld", "HexWorld").value("world_settings/locked")
            if raw:
                self._set_locks(set(json.loads(raw)))
        except Exception:  # noqa: BLE001 - convenience only
            pass
