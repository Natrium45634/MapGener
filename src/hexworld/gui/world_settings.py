"""World settings window: seed, every generator parameter, locks, presets, size, start.

Every row: a slider (min..max), an exact value, and a lock. A locked parameter keeps its
value when the settings are randomised. The schema lives in :mod:`hexworld.settings_schema`;
this module only draws it.
"""

from __future__ import annotations

import html
import json
import math

from PySide6.QtCore import QRegularExpression, QSettings, Qt, QThread, QTimer, Signal
from PySide6.QtGui import QGuiApplication, QImage, QPixmap, QRegularExpressionValidator
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDoubleSpinBox,
    QFileDialog,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QInputDialog,
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
from .. import world_tools as wt
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
        views_ = wt.affected_views(p.path)
        if views_:
            shown = ", ".join(views_[:6]) + (f" и ещё {len(views_) - 6}" if len(views_) > 6 else "")
            tip += f"\nвлияет на вкладки: {shown}"
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


class _PreviewWorker(QThread):
    """Runs the sketch of a world (:func:`hexworld.world_tools.preview_config`)."""

    done = Signal(int, object, str)  # generation id, project or None, error text

    def __init__(self, gen_id: int, config: Config, parent=None):
        super().__init__(parent)
        self.gen_id = gen_id
        self.config = config
        self._stop = False

    def stop(self):
        self._stop = True

    def run(self):
        from ..pipeline import GenerationCancelled, generate

        def progress(_stage, _f):
            if self._stop:
                raise GenerationCancelled()

        try:
            project = generate(wt.preview_config(self.config), progress=progress)
        except GenerationCancelled:
            return
        except Exception as exc:  # noqa: BLE001 - shown in the sketch panel
            self.done.emit(self.gen_id, None, str(exc))
            return
        if not self._stop:
            self.done.emit(self.gen_id, project, "")


PREVIEW_VIEWS = (
    ("biomes", "Биомы"),
    ("heights", "Высоты"),
    ("temperature", "Температура"),
    ("precipitation", "Осадки"),
    ("continents", "Материки"),
    ("plates", "Плиты"),
)


class WorldSettingsDialog(QDialog):
    """Before a generation: every setting, with locks, presets and the size of the world.

    After ``exec()``: ``self.config`` (a :class:`Config`), ``self.through`` and
    ``self.start`` (True when a start button was pressed).
    """

    changed = Signal()

    def __init__(
        self,
        config: Config | None = None,
        through: str = "v50",
        parent=None,
        current: Config | None = None,
        preview: bool = True,
    ):
        super().__init__(parent)
        self.setWindowTitle("Новый мир — настройки генерации")
        self.resize(1440, 860)
        #: config of the world on screen: unchanged stages are taken from it
        self.current = current
        self._preview_on = preview
        self._preview_id = 0
        self._preview_worker: _PreviewWorker | None = None
        self._preview_project = None
        self._workers: list = []
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
        code_btn = QPushButton("📋 Код мира")
        code_btn.setToolTip(
            "Скопировать короткий код этого мира (сид + все изменённые настройки), "
            "чтобы поделиться им или вернуться к миру позже"
        )
        code_btn.clicked.connect(self.copy_code)
        paste_btn = QPushButton("📥 Вставить код…")
        paste_btn.setToolTip("Восстановить мир по коду «HW1-…»")
        paste_btn.clicked.connect(self.paste_code)
        for b in (unlock, lock_all, load_btn, save_btn, code_btn, paste_btn):
            pre.addWidget(b)
        lay.addLayout(pre)

        # -- tabs ----
        self.tabs = QTabWidget()
        middle = QHBoxLayout()
        middle.addWidget(self.tabs, 1)
        middle.addWidget(self._build_preview_box())
        lay.addLayout(middle, 1)
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

        self._preview_timer = QTimer(self)
        self._preview_timer.setSingleShot(True)
        self._preview_timer.setInterval(900)
        self._preview_timer.timeout.connect(self.refresh_preview)
        self.set_values(ss.values_from_config(base))
        self._set_locks(ss.default_locked())
        self._restore_locks()
        self._sync_size_preset()
        self._update_footer()
        if self._preview_on:
            QTimer.singleShot(50, self.refresh_preview)

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
        self._schedule_preview()

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
        if not errors and self.current is not None:
            try:
                new_cfg = ss.config_from_values(v, seed=self.seed.text())
                plan = wt.regeneration_plan(self.current, new_cfg)
            except Exception:  # noqa: BLE001
                plan = None
            if plan and plan["first"] is None:
                text += "<br>Настройки совпадают с текущим миром: он будет взят как есть."
            elif plan and plan["reused"]:
                text += (
                    "<br><span style='color:#2a7a2a'>Быстрее: из текущего мира возьмутся "
                    f"«{'», «'.join(plan['reused'])}»; пересчёт начнётся с "
                    f"«{plan['computed'][0]}».</span>"
                )
        if errors:
            text += f"<br><span style='color:#b00'>Ошибка: {errors[0]}</span>"
        for warning in wt.conflicts(v)[:4]:
            text += f"<br><span style='color:#a65f00'>⚠ {warning}</span>"
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
        self._stop_preview()
        self.accept()

    def reject(self):
        self._stop_preview()
        super().reject()

    # -- world code ----
    def world_code(self) -> str:
        cfg = ss.config_from_values(self.values(), seed=self.seed.text())
        return wt.world_code(cfg, self.through_box.currentData())

    def copy_code(self):
        try:
            code = self.world_code()
        except Exception as exc:  # noqa: BLE001
            QMessageBox.warning(self, "Код мира", str(exc))
            return
        QGuiApplication.clipboard().setText(code)
        QMessageBox.information(
            self,
            "Код мира",
            "Код скопирован в буфер обмена. Вставьте его в «Вставить код…», чтобы получить "
            f"этот же мир:\n\n{code}",
        )

    def apply_code(self, code: str):
        cfg, through = wt.parse_world_code(code)
        self.seed.setText(cfg.seed)
        self.seed_lock.setChecked(True)
        self.set_values(ss.values_from_config(cfg))
        self.through_box.setCurrentIndex(max(self.through_box.findData(through), 0))
        self._on_change()

    def paste_code(self):
        text = QGuiApplication.clipboard().text().strip()
        default = text if text.startswith(wt.CODE_PREFIX) else ""
        code, ok = QInputDialog.getText(self, "Код мира", "Код мира (HW1-…):", text=default)
        if not ok or not code.strip():
            return
        try:
            self.apply_code(code)
        except ValueError as exc:
            QMessageBox.warning(self, "Код мира", str(exc))

    # -- sketch ----
    def _build_preview_box(self) -> QGroupBox:
        box = QGroupBox("Эскиз мира (быстрый, приблизительный)")
        box.setFixedWidth(400)
        col = QVBoxLayout(box)
        self.preview_view = QComboBox()
        for key, title in PREVIEW_VIEWS:
            self.preview_view.addItem(title, key)
        self.preview_view.currentIndexChanged.connect(lambda _i: self._show_preview())
        col.addWidget(self.preview_view)
        self.preview_image = QLabel("эскиз появится здесь")
        self.preview_image.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.preview_image.setMinimumSize(376, 260)
        self.preview_image.setStyleSheet("background: #1d2733; color: #ccc;")
        col.addWidget(self.preview_image)
        self.preview_status = QLabel("")
        self.preview_status.setWordWrap(True)
        col.addWidget(self.preview_status)
        self.preview_stats = QLabel("")
        self.preview_stats.setWordWrap(True)
        self.preview_stats.setTextFormat(Qt.TextFormat.RichText)
        col.addWidget(self.preview_stats)
        row = QHBoxLayout()
        self.preview_auto = QCheckBox("обновлять сам")
        self.preview_auto.setChecked(self._preview_on)
        self.preview_auto.setToolTip("Пересчитывать эскиз через секунду после каждого изменения")
        refresh = QPushButton("⟳ Обновить эскиз")
        refresh.clicked.connect(self.refresh_preview)
        row.addWidget(self.preview_auto)
        row.addStretch(1)
        row.addWidget(refresh)
        col.addLayout(row)
        note = QLabel(
            "Эскиз — тот же сид и настройки на грубой сетке за несколько секунд: видно, где "
            "будут материки, пустыни, льды и леса. Детали полной карты будут точнее."
        )
        note.setWordWrap(True)
        note.setStyleSheet("color: #777;")
        col.addWidget(note)
        col.addStretch(1)
        return box

    def _schedule_preview(self):
        if hasattr(self, "_preview_timer") and self.preview_auto.isChecked():
            self._preview_timer.start()

    def refresh_preview(self):
        try:
            cfg = ss.config_from_values(self.values(), seed=self.seed.text() or "HexWorld1")
        except Exception as exc:  # noqa: BLE001
            self.preview_status.setText(f"эскиз недоступен: {exc}")
            return
        self._stop_preview(wait=False)
        self._preview_id += 1
        worker = _PreviewWorker(self._preview_id, cfg, self)
        worker.done.connect(self._preview_done)
        worker.finished.connect(lambda w=worker: self._forget_worker(w))
        self._preview_worker = worker
        self._workers.append(worker)
        self.preview_status.setText("⏳ считаю эскиз… (несколько секунд)")
        worker.start()

    def _forget_worker(self, worker):
        if worker in self._workers:
            self._workers.remove(worker)
        worker.deleteLater()

    def _stop_preview(self, wait: bool = True):
        self._preview_timer.stop() if hasattr(self, "_preview_timer") else None
        for w in list(self._workers):
            w.stop()
            if wait:
                w.wait(15000)

    def _preview_done(self, gen_id: int, project, error: str):
        if gen_id != self._preview_id:
            return  # an older sketch
        if project is None:
            self.preview_status.setText(f"эскиз не получился: {error}")
            return
        self._preview_project = project
        self._show_preview()

    def _show_preview(self):
        p = self._preview_project
        if p is None:
            return
        rgb = wt.thumbnail(p, self.preview_view.currentData(), 376)
        h, w = rgb.shape[:2]
        self._preview_buf = rgb.copy()
        img = QImage(self._preview_buf.data, w, h, 3 * w, QImage.Format.Format_RGB888)
        self.preview_image.setPixmap(QPixmap.fromImage(img.copy()))
        s = wt.world_summary(p)
        rows = [r for r in wt.summary_rows(s) if r[1] != "—"]
        keep = (
            "Доля суши",
            "Массивов суши",
            "Высшая точка, м",
            "Температура суши, °C",
            "Осадки на суше, мм/год",
            "Засушливые земли",
            "Суша подо льдом",
            "Биом №1",
            "Биом №2",
        )
        esc = html.escape
        self.preview_stats.setText(
            "<br>".join(f"{esc(a)}: <b>{esc(b)}</b>" for a, b in rows if a in keep)
        )
        self.preview_status.setText(
            f"эскиз готов ({p.config.width}×{p.config.height} гексов по "
            f"{p.config.hex_spacing_km:.0f} км)"
        )

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
