"""PySide6 viewer: one map, tabs for geographic views. The core never imports this module.

Tabs come from :mod:`hexworld.views`; a new generation stage only registers its
view there. Switching a tab changes the presentation only.
"""

from __future__ import annotations

import sys
import traceback
from dataclasses import replace
from pathlib import Path

from PySide6.QtCore import Qt, QThread, Signal
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPlainTextEdit,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QSpinBox,
    QTabBar,
    QVBoxLayout,
    QWidget,
)

from .. import render, views
from ..api import stage_depth
from ..config import (
    CLIMATE_PROFILES,
    CONTINENT_PROFILES,
    HISTORY_PROFILES,
    MODES,
    PRESETS,
    PROFILES,
    ROUTING_MODES,
    SEA_MODES,
    ClimateParams,
    Config,
    ContinentParams,
    SeaParams,
)
from ..io import load_project, save_project
from ..pipeline import GenerationCancelled, generate
from ..project import Project
from ..version import GENERATOR_VERSION
from .map_view import MapView

STAGE_TITLES = {
    "plates": "плиты",
    "crust": "кора",
    "motion": "движение",
    "raster": "выборка на гексы",
    "boundary_edges": "границы",
    "tectonic_fields": "диагностические поля",
    "assemble": "сборка",
    "world_tectonics": "мировой каркас границ",
    "continental_blocks": "континентальные блоки",
    "macro_elevation": "макрорельеф",
    "v2_raster": "выборка v2 на гексы",
    "sea_level_landmasses": "уровень моря и суша",
    "geo_history": "геологическая история",
    "mountain_plan": "план горных систем",
    "v3_raster": "итоговый рельеф на гексах",
    "v3_sea": "суша итоговой эпохи",
    "v4_climate": "климат мира и приток извне",
    "v4_evolution": "климат ↔ сток ↔ эрозия",
    "v4_network": "реки, озёра, бассейны",
    "v4_assemble": "слои v4",
    "validate": "проверка",
    "done": "готово",
}
SEA_MODE_TITLES = {
    "target_land_fraction": "доля суши (target_land_fraction)",
    "fixed_sea_level_m": "уровень моря (fixed_sea_level_m)",
}


class Worker(QThread):
    progress = Signal(str, float)
    succeeded = Signal(object)
    failed = Signal(str)
    cancelled = Signal()

    def __init__(self, job, parent=None):
        super().__init__(parent)
        self._job = job
        self._cancel = False

    def cancel(self):
        self._cancel = True

    def _report(self, stage: str, fraction: float):
        if self._cancel:
            raise GenerationCancelled()
        self.progress.emit(stage, fraction)

    def run(self):
        try:
            self.succeeded.emit(self._job(self._report))
        except GenerationCancelled:
            self.cancelled.emit()
        except Exception as exc:  # shown to the user, never swallowed
            self.failed.emit(f"{exc}\n\n{traceback.format_exc()}")


def _season_name(code) -> str:
    names = ("декабрь–февраль", "март–май", "июнь–август", "сентябрь–ноябрь")
    try:
        return names[int(code)]
    except (TypeError, ValueError, IndexError):
        return "нет стока"


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle(f"HexWorld {GENERATOR_VERSION}")
        self.resize(1480, 920)
        self.project: Project | None = None
        self._worker: Worker | None = None
        #: Remembered mode and overlays of every view (display state only).
        self._view_state = {
            spec.id: (spec.default_mode(), set(spec.default_overlays()))
            for spec in views.VIEWS.values()
        }
        self._view_ids = list(views.VIEWS)

        self.view = MapView()
        self.view.hexHovered.connect(self._on_hover)
        self.view.hexClicked.connect(self._on_hex_clicked)

        # -- tabs + per-view controls above the map ---------------------------
        self.tabs = QTabBar()
        self.tabs.setExpanding(False)
        for i, spec in enumerate(views.VIEWS.values()):
            self.tabs.addTab(spec.title)
            if spec.status != "ready":
                self.tabs.setTabEnabled(i, False)
                self.tabs.setTabText(i, f"{spec.title} (запланировано)")
                self.tabs.setTabToolTip(i, "Запланировано: этой стадии генерации ещё нет")
        self.tabs.currentChanged.connect(self._on_tab_changed)
        self.mode_box = QComboBox()
        self.mode_box.currentIndexChanged.connect(self._on_mode_changed)
        # month of the views that show one month (v4.5): a display choice only
        self.month_box = QComboBox()
        for k, name in enumerate(
            ("январь", "февраль", "март", "апрель", "май", "июнь", "июль", "август",
             "сентябрь", "октябрь", "ноябрь", "декабрь")
        ):  # fmt: skip
            self.month_box.addItem(name, k)
        self.month_box.setCurrentIndex(6)
        self.month_box.setVisible(False)
        self.month_box.currentIndexChanged.connect(lambda _i: self._apply_view())
        self.overlay_row = QHBoxLayout()
        self.note = QLabel()
        self.note.setStyleSheet("font-weight: bold;")
        self.v2_button = QPushButton("Сгенерировать материки и высоты (v2.0)")
        self.v2_button.clicked.connect(self.start_generation)
        self.v2_button.setVisible(False)
        controls = QHBoxLayout()
        controls.addWidget(QLabel("Режим:"))
        controls.addWidget(self.mode_box)
        controls.addWidget(self.month_box)
        controls.addLayout(self.overlay_row)
        controls.addStretch(1)
        controls.addWidget(self.v2_button)
        controls.addWidget(self.note)
        map_column = QVBoxLayout()
        map_column.addWidget(self.tabs)
        map_column.addLayout(controls)
        map_column.addWidget(self.view, 1)
        # hover inspector: bottom right, under the map
        self.hover = QLabel("Наведите курсор на карту")
        self.hover.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignTop)
        self.hover.setMinimumHeight(54)
        self.hover.setStyleSheet("padding: 2px 6px;")
        hover_row = QHBoxLayout()
        hover_row.addStretch(1)
        hover_row.addWidget(self.hover)
        map_column.addLayout(hover_row)

        # -- side panel -------------------------------------------------------
        side = QWidget()
        col = QVBoxLayout(side)
        col.addWidget(self._build_world_box())
        col.addWidget(self._build_v2_box())
        col.addWidget(self._build_v3_box())
        col.addWidget(self._build_v4_box())
        buttons = QHBoxLayout()
        self.generate_btn = QPushButton("Сгенерировать")
        self.generate_btn.clicked.connect(self.start_generation)
        self.cancel_btn = QPushButton("Отмена")
        self.cancel_btn.clicked.connect(self.cancel_generation)
        self.cancel_btn.setEnabled(False)
        self.open_btn = QPushButton("Открыть…")
        self.open_btn.clicked.connect(self.open_project)
        self.save_btn = QPushButton("Сохранить…")
        self.save_btn.clicked.connect(self.save_project)
        for b in (self.generate_btn, self.cancel_btn):
            buttons.addWidget(b)
        col.addLayout(buttons)
        files = QHBoxLayout()
        for b in (self.open_btn, self.save_btn):
            files.addWidget(b)
        col.addLayout(files)
        export = QHBoxLayout()
        self.png_btn = QPushButton("PNG этой вкладки…")
        self.png_btn.clicked.connect(self.export_png)
        self.png_all_btn = QPushButton("PNG всех вкладок…")
        self.png_all_btn.clicked.connect(self.export_all_png)
        fit = QPushButton("Вся карта")
        fit.clicked.connect(self.view.fit)
        for b in (self.png_btn, self.png_all_btn, fit):
            export.addWidget(b)
        col.addLayout(export)
        scale_row = QHBoxLayout()
        scale_row.addWidget(QLabel("Масштаб интерфейса"))
        self.ui_scale = self._spin(70, 220, 100)
        self.ui_scale.setSuffix(" %")
        self.ui_scale.setSingleStep(10)
        self.ui_scale.valueChanged.connect(self._apply_ui_scale)
        scale_row.addWidget(self.ui_scale)
        col.addLayout(scale_row)
        self._base_font_pt = max(6.0, QApplication.instance().font().pointSizeF())

        box = QGroupBox("Легенда вкладки")
        inner = QVBoxLayout(box)
        self.legend = QLabel()
        self.legend.setWordWrap(True)
        self.legend.setTextFormat(Qt.TextFormat.RichText)
        inner.addWidget(self.legend)
        col.addWidget(box)

        box = QGroupBox("Проект / выбранный гекс")
        inner = QVBoxLayout(box)
        self.info = QPlainTextEdit()
        self.info.setStyleSheet("font-family: 'DejaVu Sans Mono', Consolas, monospace;")
        self.info.setReadOnly(True)
        self.info.setMinimumHeight(200)
        inner.addWidget(self.info)
        col.addWidget(box, 1)

        scroll = QScrollArea()
        scroll.setWidget(side)
        scroll.setWidgetResizable(True)
        scroll.setFixedWidth(420)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)

        central = QWidget()
        row = QHBoxLayout(central)
        row.addWidget(scroll)
        row.addLayout(map_column, 1)
        self.setCentralWidget(central)

        self.progress = QProgressBar()
        self.progress.setRange(0, 1000)
        self.progress.setFixedWidth(260)
        self.progress.setVisible(False)
        self.statusBar().addPermanentWidget(self.progress)
        self.statusBar().showMessage("Готов")
        self._set_busy(False)
        self._refresh_view_controls()

    # -- widgets -----------------------------------------------------------
    def _build_world_box(self) -> QGroupBox:
        box = QGroupBox("Мир и тектоника (v1.0)")
        form = QFormLayout(box)
        self.seed = QLineEdit("HexWorld1")
        self.seed.setToolTip("Цифры и латинские буквы, регистр значим: Ab12 и ab12 - разные миры")
        self.preset = QComboBox()
        self.preset.addItem("свой размер", None)
        for name, data in PRESETS.items():
            self.preset.addItem(f"{name}: {data['width']}x{data['height']}, {data['mode']}", name)
        self.preset.currentIndexChanged.connect(self._apply_preset)
        self.width_box = self._spin(2, 8192, 100)
        self.height_box = self._spin(2, 8192, 100)
        self.hex_km = self._double(0.1, 2000.0, 40.0, 1, " км")
        self.mode = QComboBox()
        self.mode.addItems(MODES)
        self.auto_plates = QCheckBox("auto")
        self.auto_plates.setChecked(True)
        self.plate_count = self._spin(1, 400, 15)
        self.plate_count.setEnabled(False)
        self.auto_plates.toggled.connect(lambda on: self.plate_count.setEnabled(not on))
        plates_row = QHBoxLayout()
        plates_row.addWidget(self.auto_plates)
        plates_row.addWidget(self.plate_count)
        self.profile = QComboBox()
        self.profile.addItems(PROFILES)
        self.profile.setCurrentText("earthlike")
        self.origin_x = self._double(-1e7, 1e7, 0.0, 0, " км", 500.0)
        self.origin_y = self._double(-1e7, 1e7, 0.0, 0, " км", 500.0)
        self.pin_world = QCheckBox("Тот же мир, другое окно/разрешение")
        self.pin_world.setToolTip(
            "Закрепляет область симуляции и число плит текущего проекта: меняются только "
            "окно (начало, размер) и размер гекса, геология остаётся той же."
        )
        form.addRow("Сид", self.seed)
        form.addRow("Пресет", self.preset)
        form.addRow("Ширина, гексов", self.width_box)
        form.addRow("Высота, гексов", self.height_box)
        form.addRow("Размер гекса", self.hex_km)
        form.addRow("Режим", self.mode)
        form.addRow("Число плит", plates_row)
        form.addRow("Профиль плит", self.profile)
        form.addRow("Начало окна X", self.origin_x)
        form.addRow("Начало окна Y", self.origin_y)
        form.addRow(self.pin_world)
        return box

    def _build_v2_box(self) -> QGroupBox:
        box = QGroupBox("Материки, рельеф и море (v2.0)")
        form = QFormLayout(box)
        defaults = ContinentParams()
        self.continent_profile = QComboBox()
        self.continent_profile.addItems(CONTINENT_PROFILES)
        self.auto_major = QCheckBox("auto")
        self.auto_major.setChecked(True)
        self.major_count = self._spin(1, 30, 5)
        self.major_count.setEnabled(False)
        self.major_count.setToolTip(
            "Цель для области симуляции, не гарантия; окно может показывать меньше."
        )
        self.auto_major.toggled.connect(lambda on: self.major_count.setEnabled(not on))
        major_row = QHBoxLayout()
        major_row.addWidget(self.auto_major)
        major_row.addWidget(self.major_count)
        self.size_variety = self._double(0.0, 1.0, defaults.continent_size_variety, 2, "", 0.1)
        self.coast_irregularity = self._double(0.0, 1.5, defaults.coast_irregularity, 2, "", 0.1)
        self.relief_strength = self._double(0.0, 3.0, 1.0, 2, "", 0.1)
        self.sea_mode = QComboBox()
        for mode in SEA_MODES:
            self.sea_mode.addItem(SEA_MODE_TITLES[mode], mode)
        self.land_fraction = self._double(0.01, 0.99, SeaParams().target_land_fraction, 2, "", 0.02)
        self.sea_level = self._double(-9000.0, 9000.0, 0.0, 0, " м", 25.0)
        self.sea_mode.currentIndexChanged.connect(self._sync_sea_mode)
        self.sea_btn = QPushButton("Применить только уровень моря")
        self.sea_btn.setToolTip(
            "Пересчитывает сушу, материки и шельф. Плиты, кора и высоты не пересоздаются."
        )
        self.sea_btn.clicked.connect(self.apply_sea_level)
        form.addRow("Профиль материков", self.continent_profile)
        form.addRow("Крупных массивов", major_row)
        form.addRow("Разброс размеров", self.size_variety)
        form.addRow("Изрезанность берегов", self.coast_irregularity)
        form.addRow("Сила макрорельефа", self.relief_strength)
        form.addRow("Режим моря", self.sea_mode)
        form.addRow("Доля суши", self.land_fraction)
        form.addRow("Уровень моря", self.sea_level)
        form.addRow(self.sea_btn)
        self._sync_sea_mode()
        return box

    def _build_v3_box(self) -> QGroupBox:
        box = QGroupBox("Геоистория, горы, вулканизм (v3.0)")
        form = QFormLayout(box)
        self.history_on = QCheckBox("считать геологическую историю")
        self.history_on.setChecked(True)
        self.duration = self._double(0.0, 400.0, 30.0, 1, " млн лет", 5.0)
        self.time_step = self._double(0.05, 20.0, 1.0, 2, " млн лет", 0.5)
        self.history_profile = QComboBox()
        self.history_profile.addItems(HISTORY_PROFILES)
        self.history_profile.setCurrentText("earthlike")
        self.uplift_scale = self._double(0.0, 5.0, 1.0, 2, "", 0.1)
        self.erosion = self._double(0.0, 5.0, 1.0, 2, "", 0.1)
        self.hotspots = self._double(0.0, 5.0, 1.0, 2, "", 0.25)
        self.mountain_detail = self._double(0.0, 2.0, 1.0, 2, "", 0.1)
        self.epoch_box = QComboBox()
        self.epoch_box.setToolTip(
            "Сохранённые снимки истории. Смена снимка не пересчитывает "
            "историю: строятся только слои выбранной эпохи."
        )
        self.epoch_btn = QPushButton("Показать выбранный снимок")
        self.epoch_btn.clicked.connect(self.apply_epoch)
        form.addRow(self.history_on)
        form.addRow("Длительность", self.duration)
        form.addRow("Шаг времени", self.time_step)
        form.addRow("Интенсивность", self.history_profile)
        form.addRow("Скорость поднятия ×", self.uplift_scale)
        form.addRow("Эрозия ×", self.erosion)
        form.addRow("Горячие точки ×", self.hotspots)
        form.addRow("Детализация гор", self.mountain_detail)
        form.addRow("Снимок", self.epoch_box)
        form.addRow(self.epoch_btn)
        return box

    def _build_v4_box(self) -> QGroupBox:
        box = QGroupBox("Климат, реки, эрозия (v4.0)")
        form = QFormLayout(box)
        c = ClimateParams()
        self.v4_on = QCheckBox("считать климат, гидрологию и эрозию")
        self.v4_on.setChecked(True)
        self.climate_profile = QComboBox()
        self.climate_profile.addItems(CLIMATE_PROFILES)
        self.latitude = self._double(-90.0, 90.0, c.latitude_center_deg, 1, "°", 5.0)
        self.latitude.setToolTip(
            "Условная широта центра области симуляции (север - вверх карты). "
            "Экватор и полюса могут лежать вне карты."
        )
        self.km_per_degree = self._double(0.0, 2000.0, 0.0, 1, " км/°", 10.0)
        self.km_per_degree.setToolTip(
            "Масштаб широты. 0 - реальный (111,2 км/°); если область тогда перешла бы "
            "через полюс, шкала сжимается, о чём пишется предупреждение."
        )
        self.temperature_offset = self._double(-30.0, 30.0, 0.0, 1, " °C", 1.0)
        self.precipitation_scale = self._double(0.1, 5.0, 1.0, 2, "", 0.1)
        self.wind_strength = self._double(0.0, 3.0, 1.0, 2, "", 0.1)
        self.erosion_ma = self._double(0.0, 50.0, 2.0, 2, " млн лет", 0.5)
        self.erosion_ma.setToolTip("0 - климат и реки без изменения рельефа")
        self.erosion_steps = self._spin(1, 100, 8)
        self.river_threshold = self._double(0.1, 100000.0, 25.0, 1, " м³/с", 5.0)
        self.river_threshold.setToolTip(
            "Порог показа реки. Меняет только сеть рек: расход и эрозия не пересчитываются."
        )
        self.routing = QComboBox()
        self.routing.addItems(ROUTING_MODES)
        self.routing.setToolTip(
            "stochastic - выбор среди нижних соседей пропорционально перепаду (Rho8): "
            "сходящаяся древовидная сеть; steepest - классический D6"
        )
        self.v4_btn = QPushButton("Пересчитать только v4 (климат, реки, эрозию)")
        self.v4_btn.setToolTip("Плиты, материки и геоистория берутся из текущего проекта.")
        self.v4_btn.clicked.connect(self.apply_v4)
        form.addRow(self.v4_on)
        form.addRow("Профиль климата", self.climate_profile)
        form.addRow("Широта центра", self.latitude)
        form.addRow("Масштаб широты", self.km_per_degree)
        form.addRow("Сдвиг температуры", self.temperature_offset)
        form.addRow("Осадки ×", self.precipitation_scale)
        form.addRow("Сила ветра ×", self.wind_strength)
        form.addRow("Эрозия v4", self.erosion_ma)
        form.addRow("Шагов связки", self.erosion_steps)
        form.addRow("Река от", self.river_threshold)
        form.addRow("Направление стока", self.routing)
        form.addRow(self.v4_btn)
        return box

    def _apply_ui_scale(self, percent: int):
        font = QApplication.instance().font()
        font.setPointSizeF(self._base_font_pt * percent / 100.0)
        QApplication.instance().setFont(font)
        for widget in self.findChildren(QWidget):
            widget.setFont(font)
        self.note.setStyleSheet("font-weight: bold;")

    def _on_hover(self, text: str):
        self.hover.setText(text)

    def _fill_epochs(self):
        self.epoch_box.blockSignals(True)
        self.epoch_box.clear()
        hist = self.project.hist if self.project is not None else None
        if hist is not None:
            for t in hist.epochs:
                tag = (
                    " (исходная)"
                    if t == hist.epochs[0]
                    else (" (итоговая)" if t == hist.epochs[-1] else "")
                )
                self.epoch_box.addItem(f"t = {t:g} млн лет{tag}", t)
            index = min(
                range(len(hist.epochs)), key=lambda i: abs(hist.epochs[i] - hist.epoch_time_ma)
            )
            self.epoch_box.setCurrentIndex(index)
        self.epoch_box.blockSignals(False)

    @staticmethod
    def _spin(lo, hi, value):
        box = QSpinBox()
        box.setRange(lo, hi)
        box.setValue(value)
        return box

    @staticmethod
    def _double(lo, hi, value, decimals, suffix="", step=1.0):
        box = QDoubleSpinBox()
        box.setRange(lo, hi)
        box.setDecimals(decimals)
        box.setSingleStep(step)
        box.setValue(value)
        if suffix:
            box.setSuffix(suffix)
        return box

    def _sync_sea_mode(self):
        target = self.sea_mode.currentData() == "target_land_fraction"
        self.land_fraction.setEnabled(target)
        self.sea_level.setEnabled(not target)

    def _apply_preset(self, _index):
        name = self.preset.currentData()
        if name is None:
            return
        data = PRESETS[name]
        self.width_box.setValue(data["width"])
        self.height_box.setValue(data["height"])
        self.hex_km.setValue(data["hex_spacing_km"])
        self.mode.setCurrentText(data["mode"])

    # -- views (tabs) ------------------------------------------------------
    def current_view(self) -> views.ViewSpec:
        return views.VIEWS[self._view_ids[self.tabs.currentIndex()]]

    def _on_tab_changed(self, _index):
        self._refresh_view_controls()

    def _refresh_view_controls(self):
        """Rebuild mode list, overlay toggles, legend and note for the current tab."""
        spec = self.current_view()
        mode, overlays = self._view_state[spec.id]
        self.mode_box.blockSignals(True)
        self.mode_box.clear()
        for option in spec.modes:
            self.mode_box.addItem(option.title, option.id)
        index = max(0, self.mode_box.findData(mode))
        self.mode_box.setCurrentIndex(index)
        self.mode_box.blockSignals(False)
        while self.overlay_row.count():
            item = self.overlay_row.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.setParent(None)
                widget.deleteLater()
        for option in spec.overlays:
            box = QCheckBox(option.title)
            box.setChecked(option.id in overlays)
            box.toggled.connect(lambda on, oid=option.id: self._on_overlay_toggled(oid, on))
            self.overlay_row.addWidget(box)
        self.note.setText(spec.note)
        self.month_box.setVisible(bool(getattr(spec, "months", False)))
        self._apply_view()

    def _on_mode_changed(self, _index):
        spec = self.current_view()
        if self.mode_box.currentData() is not None:
            self._view_state[spec.id] = (self.mode_box.currentData(), self._view_state[spec.id][1])
            self._apply_view()

    def _on_overlay_toggled(self, overlay_id: str, on: bool):
        spec = self.current_view()
        mode, overlays = self._view_state[spec.id]
        (overlays.add if on else overlays.discard)(overlay_id)
        self._apply_view()

    def _apply_view(self):
        spec = self.current_view()
        mode, overlays = self._view_state[spec.id]
        if getattr(spec, "months", False):
            overlays = set(overlays) | {f"month:{int(self.month_box.currentData() or 0)}"}
        self.view.set_view(spec.id, mode, overlays)
        ok, reason = spec.availability(self.project)
        needs_v2 = self.project is not None and spec.stage == "v2" and self.project.geo is None
        self.v2_button.setVisible(needs_v2)
        if ok and spec.legend is not None:
            rows = []
            for item in spec.legend(self.project, mode):
                if item.color is None:
                    rows.append(item.label)
                else:
                    r, g, b = (int(v) for v in item.color)
                    rows.append(f"<span style='color:rgb({r},{g},{b})'>■</span> {item.label}")
            self.legend.setText("<br>".join(rows))
        else:
            self.legend.setText(reason if self.project is not None else "Нет проекта")
        self.png_btn.setEnabled(ok and self._worker_idle())

    # -- config ------------------------------------------------------------
    def build_config(self) -> Config:
        base = self.project.config if self.project is not None else Config()
        config = replace(
            base,
            seed=self.seed.text().strip(),
            width=self.width_box.value(),
            height=self.height_box.value(),
            hex_spacing_km=self.hex_km.value(),
            mode=self.mode.currentText(),
            viewport_origin_km=(self.origin_x.value(), self.origin_y.value()),
            simulation_bounds_km=None,
            plates=replace(
                base.plates,
                count="auto" if self.auto_plates.isChecked() else self.plate_count.value(),
                profile=self.profile.currentText(),
            ),
            continents=replace(
                base.continents,
                continent_profile=self.continent_profile.currentText(),
                major_landmass_count=(
                    "auto" if self.auto_major.isChecked() else self.major_count.value()
                ),
                continent_size_variety=self.size_variety.value(),
                coast_irregularity=self.coast_irregularity.value(),
            ),
            relief=replace(base.relief, relief_strength=self.relief_strength.value()),
            sea=self._sea_params(base.sea),
            history=replace(
                base.history,
                enabled=self.history_on.isChecked(),
                duration_ma=self.duration.value(),
                time_step_ma=self.time_step.value(),
                intensity=self.history_profile.currentText(),
                uplift_scale=self.uplift_scale.value(),
                erosion_intensity=self.erosion.value(),
                hotspot_density=self.hotspots.value(),
                epoch="final",
            ),
            mountains=replace(base.mountains, detail=self.mountain_detail.value()),
        )
        config = self._v4_params(config)
        if self.pin_world.isChecked() and self.project is not None:
            g = self.project.geometry
            config = replace(
                config,
                seed=self.project.config.seed,
                simulation_bounds_km=tuple(g.simulation_bounds.as_list()),
                plates=replace(self.project.config.plates, count=g.plate_count),
            )
        return config

    def _v4_params(self, config: Config) -> Config:
        erosion_on = self.erosion_ma.value() > 0.0
        return replace(
            config,
            climate=replace(
                config.climate,
                profile=self.climate_profile.currentText(),
                latitude_center_deg=self.latitude.value(),
                km_per_degree=self.km_per_degree.value(),
                temperature_offset_c=self.temperature_offset.value(),
                precipitation_scale=self.precipitation_scale.value(),
                wind_strength=self.wind_strength.value(),
            ),
            hydrology=replace(
                config.hydrology,
                river_min_discharge_m3_s=self.river_threshold.value(),
                routing=self.routing.currentText(),
            ),
            erosion=replace(
                config.erosion,
                enabled=erosion_on,
                duration_ma=self.erosion_ma.value() if erosion_on else config.erosion.duration_ma,
                steps=self.erosion_steps.value(),
            ),
        )

    def _through(self) -> str:
        return "v4" if self.v4_on.isChecked() else "v3"

    def _sea_params(self, base: SeaParams) -> SeaParams:
        return replace(
            base,
            sea_mode=self.sea_mode.currentData(),
            target_land_fraction=self.land_fraction.value(),
            fixed_sea_level_m=self.sea_level.value(),
        )

    # -- background work ---------------------------------------------------
    def _worker_idle(self) -> bool:
        return self._worker is None or not self._worker.isRunning()

    def _set_busy(self, busy: bool):
        for w in (self.generate_btn, self.open_btn, self.v2_button):
            w.setEnabled(not busy)
        has_v3 = self.project is not None and self.project.hist is not None
        self.v4_btn.setEnabled(not busy and has_v3)
        has = self.project is not None
        self.save_btn.setEnabled(not busy and has)
        self.png_all_btn.setEnabled(not busy and has)
        self.png_btn.setEnabled(not busy and has)
        self.sea_btn.setEnabled(not busy and has and self.project.geo is not None)
        self.epoch_btn.setEnabled(not busy and has and self.project.hist is not None)
        self.cancel_btn.setEnabled(busy)
        self.progress.setVisible(busy)

    def _run(self, job, on_done):
        self._set_busy(True)
        self.progress.setValue(0)
        worker = Worker(job, self)
        worker.progress.connect(self._on_progress)
        worker.succeeded.connect(on_done)
        worker.failed.connect(self._on_failed)
        worker.cancelled.connect(self._on_cancelled)
        worker.finished.connect(self._on_worker_finished)
        self._worker = worker
        worker.start()

    def _on_worker_finished(self):
        self._set_busy(False)
        self._apply_view()

    def _on_progress(self, stage: str, fraction: float):
        self.progress.setValue(int(1000 * fraction))
        self.statusBar().showMessage(f"Генерация: {STAGE_TITLES.get(stage, stage)}…")

    def _on_failed(self, message: str):
        self.statusBar().showMessage("Ошибка; прежние данные сохранены")
        QMessageBox.critical(self, "HexWorld - ошибка", message)

    def _on_cancelled(self):
        self.statusBar().showMessage("Генерация отменена; прежние данные не изменены")

    def cancel_generation(self):
        if self._worker is not None and self._worker.isRunning():
            self._worker.cancel()
            self.statusBar().showMessage("Отмена после текущей стадии…")

    # -- actions -----------------------------------------------------------
    def start_generation(self):
        try:
            config = self.build_config()
        except ValueError as exc:
            QMessageBox.warning(self, "Параметры", str(exc))
            return
        previous = self.project
        through = self._through()
        self._run(
            lambda progress: generate(config, progress=progress, reuse=previous, through=through),
            lambda project: self._show_project(project, "Сгенерировано"),
        )

    def apply_v4(self):
        """Recompute climate, hydrology and erosion only; v1-v3 are reused as they are."""
        if self.project is None or self.project.hist is None:
            return
        try:
            config = self._v4_params(self.project.config)
        except ValueError as exc:
            QMessageBox.warning(self, "Параметры", str(exc))
            return
        previous = self.project
        self._run(
            lambda progress: generate(config, progress=progress, reuse=previous, through="v45"),
            lambda project: self._show_project(project, "Климат, реки и эрозия пересчитаны"),
        )

    def apply_sea_level(self):
        """Recompute land / landmasses / shelf only; everything else is reused."""
        if self.project is None or self.project.geo is None:
            return
        try:
            config = replace(self.project.config, sea=self._sea_params(self.project.config.sea))
        except ValueError as exc:
            QMessageBox.warning(self, "Параметры", str(exc))
            return
        previous = self.project
        self._run(
            lambda progress: generate(
                config, progress=progress, reuse=previous, through=stage_depth(previous)
            ),
            lambda project: self._show_project(project, "Уровень моря применён"),
        )

    def apply_epoch(self):
        """Build the layers of another stored snapshot; the history itself is reused."""
        if self.project is None or self.project.hist is None:
            return
        when = self.epoch_box.currentData()
        if when is None:
            return
        config = replace(
            self.project.config, history=replace(self.project.config.history, epoch=float(when))
        )
        previous = self.project
        self._run(
            lambda progress: generate(
                config, progress=progress, reuse=previous, through=stage_depth(previous)
            ),
            lambda project: self._show_project(project, f"Снимок t = {when:g} млн лет"),
        )

    def open_project(self):
        path = QFileDialog.getExistingDirectory(self, "Открыть проект HexWorld")
        if path:
            self.load_path(path)

    def load_path(self, path: str):
        self._run(
            lambda _progress: load_project(path),
            lambda project: self._show_project(project, f"Открыт {path}", sync_form=True),
        )

    def save_project(self):
        if self.project is None:
            return
        path = QFileDialog.getExistingDirectory(self, "Папка для сохранения проекта")
        if not path:
            return
        try:
            save_project(self.project, path)
        except Exception as exc:
            QMessageBox.critical(self, "Сохранение", str(exc))
            return
        self.statusBar().showMessage(f"Сохранено в {path}")

    def export_png(self):
        if self.project is None:
            return
        spec = self.current_view()
        mode, overlays = self._view_state[spec.id]
        path, _ = QFileDialog.getSaveFileName(
            self, "Экспорт PNG", f"view_{spec.id}_{mode}.png", "PNG (*.png)"
        )
        if path:
            image = views.render_view_image(self.project, spec.id, mode, overlays)
            render.write_png(Path(path), image)
            self.statusBar().showMessage(f"Изображение сохранено: {path}")

    def export_all_png(self):
        if self.project is None:
            return
        folder = QFileDialog.getExistingDirectory(self, "Папка для PNG всех вкладок")
        if not folder:
            return
        count = 0
        for spec in views.VIEWS.values():
            if not spec.availability(self.project)[0]:
                continue
            mode, overlays = self._view_state[spec.id]
            image = views.render_view_image(self.project, spec.id, mode, overlays)
            render.write_png(Path(folder) / f"view_{spec.id}_{mode}.png", image)
            count += 1
        self.statusBar().showMessage(f"Сохранено изображений: {count} → {folder}")

    def _show_project(self, project: Project, message: str, sync_form: bool = False):
        self.project = project
        if sync_form:
            self._sync_form(project.config)
        if project.geo is not None:
            self.sea_level.setValue(project.geo.sea_level_m)
        self._fill_epochs()
        self.view.set_project(project)
        self._apply_view()
        self._show_info()
        total = project.timings.get("total")
        ok = (
            project.validation.get("ok")
            and (project.geo is None or project.geo.validation.get("ok"))
            and (project.hist is None or project.hist.validation.get("ok"))
            and (project.hydro is None or project.hydro.validation.get("ok"))
        )
        state = "проверка пройдена" if ok else "ПРОВЕРКА НЕ ПРОЙДЕНА"
        timing = f", {total:.1f} с" if total else ""
        extra = "" if project.geo is not None else "; материки и высоты ещё не сгенерированы"
        if project.geo is not None and project.hist is None:
            extra = "; геоистория ещё не сгенерирована"
        elif project.hist is not None and project.hydro is None:
            extra = "; климат и реки (v4) ещё не рассчитаны"
        self.statusBar().showMessage(f"{message}: {state}{timing}{extra}")
        if not ok:
            errors = list(project.validation.get("errors", []))
            if project.geo is not None:
                errors += project.geo.validation.get("errors", [])
            if project.hist is not None:
                errors += project.hist.validation.get("errors", [])
            if project.hydro is not None:
                errors += project.hydro.validation.get("errors", [])
            QMessageBox.warning(self, "Проверка", "\n".join(errors))

    def _sync_form(self, c: Config):
        self.seed.setText(c.seed)
        self.preset.setCurrentIndex(0)
        self.width_box.setValue(c.width)
        self.height_box.setValue(c.height)
        self.hex_km.setValue(c.hex_spacing_km)
        self.mode.setCurrentText(c.mode)
        self.auto_plates.setChecked(c.plates.count == "auto")
        if c.plates.count != "auto":
            self.plate_count.setValue(int(c.plates.count))
        self.profile.setCurrentText(c.plates.profile)
        self.origin_x.setValue(c.viewport_origin_km[0])
        self.origin_y.setValue(c.viewport_origin_km[1])
        self.continent_profile.setCurrentText(c.continents.continent_profile)
        self.auto_major.setChecked(c.continents.major_landmass_count == "auto")
        if c.continents.major_landmass_count != "auto":
            self.major_count.setValue(int(c.continents.major_landmass_count))
        self.size_variety.setValue(c.continents.continent_size_variety)
        self.coast_irregularity.setValue(c.continents.coast_irregularity)
        self.relief_strength.setValue(c.relief.relief_strength)
        self.sea_mode.setCurrentIndex(self.sea_mode.findData(c.sea.sea_mode))
        self.land_fraction.setValue(c.sea.target_land_fraction)
        self.sea_level.setValue(c.sea.fixed_sea_level_m)
        self.history_on.setChecked(c.history.enabled)
        self.duration.setValue(c.history.duration_ma)
        self.time_step.setValue(c.history.time_step_ma)
        self.history_profile.setCurrentText(c.history.intensity)
        self.uplift_scale.setValue(c.history.uplift_scale)
        self.erosion.setValue(c.history.erosion_intensity)
        self.hotspots.setValue(c.history.hotspot_density)
        self.mountain_detail.setValue(c.mountains.detail)
        self.climate_profile.setCurrentText(c.climate.profile)
        self.latitude.setValue(c.climate.latitude_center_deg)
        self.km_per_degree.setValue(c.climate.km_per_degree)
        self.temperature_offset.setValue(c.climate.temperature_offset_c)
        self.precipitation_scale.setValue(c.climate.precipitation_scale)
        self.wind_strength.setValue(c.climate.wind_strength)
        self.erosion_ma.setValue(c.erosion.duration_ma if c.erosion.enabled else 0.0)
        self.erosion_steps.setValue(c.erosion.steps)
        self.river_threshold.setValue(c.hydrology.river_min_discharge_m3_s)
        self.routing.setCurrentText(c.hydrology.routing)
        self.v4_on.setChecked(self.project is not None and self.project.hydro is not None)

    # -- property panel ----------------------------------------------------
    def _project_text(self) -> str:
        p = self.project
        g = p.geometry
        lines = [
            f"world_id: {p.world_id}",
            f"сетка: {g.layout.width} x {g.layout.height}, гекс {g.layout.spacing_km:g} км",
            f"окно: {g.viewport_bounds.width:,.0f} x {g.viewport_bounds.height:,.0f} км",
            f"область симуляции: {g.simulation_bounds.width:,.0f} x "
            f"{g.simulation_bounds.height:,.0f} км",
            f"плит: {p.stats.get('plates_in_simulation')} в симуляции, "
            f"{p.stats.get('plates_visible')} в окне; граничных рёбер: {len(p.boundaries)}",
        ]
        if p.geo is None:
            lines.append("Материки/Высоты ещё не сгенерированы (проект v1.0).")
        else:
            s = p.geo.summary
            w = s["world"]
            lines += [
                f"уровень моря: {p.geo.sea_level_m:+.0f} м ({s['sea_mode']})",
                f"суша: {100 * s['land_fraction_viewport']:.1f} % окна, "
                f"{100 * w['land_fraction_simulation']:.1f} % области симуляции",
                f"материков в окне: {len(p.geo.landmasses)} (крупных "
                f"{s['major_landmasses_in_window']}); крупных в симуляции: "
                f"{w['major_landmass_count']}",
                f"континентальных блоков в окне: {len(p.geo.blocks)}; шельф "
                f"{100 * s['shelf_fraction_viewport']:.1f} % окна",
                f"высоты: {s['min_elevation_m']:+.0f} … {s['max_elevation_m']:+.0f} м",
            ]
        if p.hist is None:
            if p.geo is not None:
                lines.append("Геоистория ещё не сгенерирована (проект v2.0).")
        else:
            m = p.hist.summary
            backend = m["mountain_backend"]
            lines += [
                f"история: {p.hist.duration_ma:g} млн лет, показан снимок t = "
                f"{p.hist.epoch_time_ma:g}",
                f"изменилось гексов (> 50 м): {100 * m['changed_hex_share']:.0f} %; новая кора "
                f"{100 * m['new_crust_share_viewport']:.1f} % окна",
                f"итоговые высоты: {m['min_elevation_m']:+.0f} … {m['max_elevation_m']:+.0f} м; "
                f"суша {100 * m['land_fraction_viewport']:.1f} % окна",
                f"горных систем {m['mountain_systems']}, хребтов {m['ridges']}, вершин "
                f"{m['peaks']}, перевалов {m['passes']}",
                f"вулканов {m['volcanoes']} (действующих {m['active_volcanoes']}), горячих "
                f"точек {m['hotspots']}; событий {m['events']}",
                "горный модуль: "
                + ("PMGE" if backend["is_pmge"] else "встроенный (библиотека PMGE не найдена)"),
            ]
        if p.hydro is None:
            if p.hist is not None:
                lines.append("Климат, реки и эрозия (v4) ещё не рассчитаны.")
        else:
            lines += self._v4_project_lines(p)
        if p.stats.get("reused_stages"):
            lines.append("без пересчёта: " + ", ".join(p.stats["reused_stages"]))
        for w in p.warnings:
            lines.append(f"⚠ {w}")
        lines += ["", "Щёлкните по карте, чтобы выбрать гекс."]
        return "\n".join(lines)

    def _show_info(self):
        if self.project is None:
            self.info.setPlainText("")
            return
        hexagon = self.view.selected_hex
        if hexagon is None:
            self.info.setPlainText(self._project_text())
            return
        p = self.project
        info = p.hex_info(*hexagon)
        plate = p.plate(info["plate_id"])
        lines = [
            f"Гекс ({info['col']}, {info['row']}):  x {info['x_km']:,.0f} км, "
            f"y {info['y_km']:,.0f} км",
            "",
            f"plate_id {plate['id']}: {plate['name']} ({plate['size_class']})",
            f"  скорость vx {plate['velocity_mm_yr'][0]:+.1f}, vy "
            f"{plate['velocity_mm_yr'][1]:+.1f} мм/год; +x восток, +y юг",
            f"  площадь {plate['area_km2'] / 1e6:,.2f} млн км², континентальной коры "
            f"{100 * plate['continental_fraction']:.0f} % (prior)",
            f"  граница здесь: {info['boundary']}; до границы "
            f"{info['boundary_distance_km']:,.0f} км",
            f"  диагностика: uplift {info['uplift']:.2f}, rifting {info['rifting']:.2f}, "
            f"subduction {info['subduction']:.2f}, volcanism {info['volcanism']:.2f}, "
            f"seismicity {info['seismicity']:.2f}",
            f"  кора prior v1.0: {info['crust']}",
        ]
        if p.geo is None:
            lines += ["", "Материки/Высоты ещё не сгенерированы для этого проекта."]
        else:
            lines += [
                "",
                f"доля континентальной коры: {info['continental_fraction']:.2f}",
            ]
            block = info["continental_block_id"]
            if block >= 0:
                b = p.geo.blocks[block]
                lines.append(
                    f"continental_block_id {block}: {b['name']}, "
                    f"{b['area_km2'] / 1e6:,.2f} млн км² в окне (над водой "
                    f"{b['land_area_km2'] / 1e6:,.2f}), плит: {b['plate_count']}"
                )
            else:
                lines.append("continental_block_id −1: океаническая кора")
            land = info["landmass_id"]
            if land >= 0:
                m = p.geo.landmasses[land]
                tail = ", уходит за край окна" if m["touches_window_edge"] else ""
                lines.append(
                    f"landmass_id {land}: {m['name']}, {m['area_km2'] / 1e6:,.2f} млн км² "
                    f"в окне{tail}"
                )
            else:
                where = "шельф" if info["shelf_mask"] else "под водой"
                lines.append(f"landmass_id −1: {where}")
            lines += [
                f"макровысота: {info['macro_elevation_m']:+,.0f} м (нуль модели)",
                f"уровень моря: {info['sea_level_m']:+,.0f} м",
                f"высота над морем: {info['height_above_sea_m']:,.0f} м; "
                f"глубина: {info['depth_m']:,.0f} м",
                f"тектонический вклад в высоту: {info['tectonic_relief_m']:+,.0f} м",
            ]
        if p.hist is not None:

            def shown(key, unit="", digits=0):
                value = info.get(key)
                return "нет данных" if value is None else f"{value:,.{digits}f}{unit}"

            system = info["mountain_system_id"]
            v3_elevation = info.get("v3_elevation_m", info["elevation_m"])
            v3_sediment = info.get("v3_sediment_thickness_m", info["sediment_thickness_m"])
            lines += [
                "",
                f"— снимок t = {info['epoch_time_ma']:g} млн лет —",
                f"высота эпохи: {v3_elevation:+,.0f} м (исходная "
                f"{info['macro_elevation_m']:+,.0f} м, изменение "
                f"{info['delta_elevation_m']:+,.0f} м)",
                f"плита в эту эпоху: {info['plate_id_epoch']} "
                f"{p.plate(info['plate_id_epoch'])['name']}"
                + (" (новая кора)" if info["new_crust"] else ""),
                f"порода: {info['rock']}, сопротивление эрозии "
                f"{info['rock_resistance']:.2f}; возраст коры {shown('crust_age_ma', ' млн лет')}",
                f"скорость поднятия: {info['uplift_rate_mm_yr']:+.3f} мм/год; поднято всего "
                f"{info['uplift_total_m']:+,.0f} м",
                f"снято эрозией {info['erosion_total_m']:,.0f} м; осадков {v3_sediment:,.0f} м",
                f"горная система: {system if system >= 0 else 'нет'}; возраст горообразования: "
                f"{shown('orogen_age_ma', ' млн лет')}",
                f"вулкан: {info['volcano_id'] if info['volcano_id'] >= 0 else 'нет'}; возраст "
                f"постройки: {shown('volcanic_age_ma', ' млн лет', 1)}; потенциал "
                f"{info['volcanic_potential']:.2f}",
                f"уклон {info['slope_m_per_km']:.1f} м/км; перепад внутри гекса "
                f"{info['local_relief_m']:,.0f} м",
            ]
        if p.hydro is not None:
            lines += ["", *self._v4_hex_lines(info)]
        if p.env is not None:
            lines += ["", *self._v45_hex_lines(info)]
        lines += ["", "— проект —", self._project_text()]
        self.info.setPlainText("\n".join(lines))

    @staticmethod
    def _v4_project_lines(p: Project) -> list[str]:
        s = p.hydro.summary
        lat = p.hydro.latitude
        a = s["aridity_shares"]
        dry = a["hyper_arid"] + a["arid"] + a["semi_arid"] + a["dry_subhumid"]
        squeeze = " (сжато)" if lat.get("compressed") else ""
        return [
            f"климат: {s['climate_profile']}, широта центра {lat['center_deg']:+.1f}°, "
            f"{lat['km_per_degree']:.0f} км/°{squeeze}",
            "суши в окне нет"
            if s["land_mean_temperature_c"] is None
            else f"суша: {s['land_mean_temperature_c']:+.1f} °C, "
            f"{s['land_mean_precipitation_mm_yr']:.0f} мм/год, коэффициент стока "
            f"{s['runoff_ratio']:.2f}",
            f"аридные земли (P/PET < 0,2): {100 * (a['hyper_arid'] + a['arid']):.0f} %; "
            f"все засушливые (< 0,65): {100 * dry:.0f} %",
            f"рек {s['rivers']}, порядок Стралера до {s['max_strahler']}; озёр {s['lakes']} "
            f"({100 * s['lake_share_of_land']:.1f} % суши)",
            f"не доходит до моря: {100 * s['no_outflow_share_of_land']:.0f} % суши",
            f"эрозия v4: {s['erosion_duration_ma']:g} млн лет, средний размыв "
            f"{s['mean_denudation_m_ma']:.0f} м/млн лет",
        ]

    @staticmethod
    def _v45_hex_lines(info: dict) -> list[str]:
        from ..pipeline_v45 import QUALITY
        from ..stages.environment import DRAINAGE_TITLES, PARENT_MATERIALS, PARENT_TITLES

        if not info.get("land_mask", False) or info.get("lake_id", -1) >= 0:
            return [
                "— экологические параметры (v4.5) —",
                "вода: почвы и водный баланс суши не считаются",
            ]

        def f(key, spec=".0f", unit=""):
            v = info.get(key)
            return "нет данных" if v is None else f"{v:{spec}}{unit}"

        pm = PARENT_MATERIALS[int(info.get("parent_material_class", 0))]
        dc = int(info.get("drainage_class", 255))
        quality = [k for k, bit in QUALITY.items() if int(info.get("quality_flags", 0)) & bit]
        lines = [
            "— экологические параметры (v4.5), оценки —",
            f"порода: {PARENT_TITLES[pm]}; почва "
            + (
                f"песок {f('sand_frac', '.0%')}, пыль {f('silt_frac', '.0%')}, глина "
                f"{f('clay_frac', '.0%')}; мощность {f('soil_depth_m', '.1f', ' м')}, камни "
                f"{f('coarse_fragment_frac', '.0%')}"
                if info.get("soil_present")
                else "нет (скала или лёд)"
            ),
            f"FC {f('field_capacity_vol', '.2f')}, PWP {f('wilting_point_vol', '.2f')}, AWC "
            f"{f('awc_mm', '.0f', ' мм')}; дренаж: "
            f"{DRAINAGE_TITLES[dc] if dc < len(DRAINAGE_TITLES) else '—'}",
            f"GDD5 {f('gdd5_proxy_c_day', ',.0f')} °C·сут (оценка); сухих месяцев "
            f"{f('dry_month_count', 'd')}, холодных {f('cold_month_count', 'd')}; P/PET "
            f"{f('aridity_p_over_pet', '.2f')}",
            f"TWI {f('twi', '.1f')}, HAND {f('hand_m', ',.0f', ' м')}; переувлажнение "
            f"{f('wetness_potential', '.2f')}, пойма {f('floodplain_potential', '.2f')}; "
            f"мерзлота (потенциал) {f('permafrost_potential', '.2f')}",
            "качество: " + (", ".join(quality) if quality else "обычное"),
        ]
        monthly = info.get("monthly") or {}
        rows = (
            ("T, °C", "temperature_monthly_c", "{:+5.0f}"),
            ("P, мм", "precip_monthly_mm", "{:5.0f}"),
            ("PET", "pet_monthly_mm", "{:5.0f}"),
            ("AET", "aet_monthly_mm", "{:5.0f}"),
            ("снег", "snowpack_monthly_swe_mm", "{:5.0f}"),
            ("почва", "soil_storage_monthly_mm", "{:5.0f}"),
            ("сток", "runoff_monthly_mm", "{:5.0f}"),
        )
        if monthly:
            lines.append("мес.    " + "".join(f"{m:>5}" for m in "ЯФМАМИИАСОНД"))
            for title, key, fmt in rows:
                vals = monthly.get(key) or []
                cells = "".join("    —" if v is None else fmt.format(v) for v in vals)
                lines.append(f"{title:<8}{cells}")
        return lines

    @staticmethod
    def _v4_hex_lines(info: dict) -> list[str]:
        def num(key, digits=0):
            value = info.get(key)
            return "нет данных" if value is None else f"{value:,.{digits}f}"

        lines = [
            "— климат, реки, эрозия (v4) —",
            f"высота после эрозии: {info['elevation_m']:+,.0f} м (до "
            f"{info['pre_erosion_elevation_m']:+,.0f} м, изменение "
            f"{info['v4_delta_elevation_m']:+,.1f} м)",
            f"широта {info['latitude_deg']:+.1f}°; температура "
            f"{info['temperature_mean_c']:+.1f} °C (годовой размах "
            f"{num('temperature_seasonal_c', 1)} °C); снег {100 * info['snow_fraction']:.0f} %"
            " осадков",
            f"ветер u {info['wind_u_ms']:+.1f}, v {info['wind_v_ms']:+.1f} м/с "
            "(+u на восток, +v на юг)",
            f"осадки {num('precipitation_mm_yr')} мм/год; PET "
            f"{num('potential_evapotranspiration_mm_yr')}, AET "
            f"{num('actual_evapotranspiration_mm_yr')} мм/год; сток {num('runoff_mm_yr')} мм/год",
            f"индекс увлажнения P/PET {num('humidity_index', 2)}",
            f"зима/лето: декабрь–февраль {num('temperature_djf_c', 1)} °C, "
            f"{num('precipitation_djf_mm')} мм; июнь–август {num('temperature_jja_c', 1)} °C, "
            f"{num('precipitation_jja_mm')} мм",
            f"половодье: {_season_name(info.get('runoff_peak_season'))}; талые воды "
            f"{100 * float(info.get('snowmelt_share', 0.0)):.0f} %, подземное питание "
            f"{100 * float(info.get('baseflow_index', 0.0)):.0f} % стока",
            f"площадь водосбора {num('flow_accumulation_area_km2')} км²; расход "
            f"{num('discharge_m3_s', 1)} м³/с; приток извне {num('external_inflow_m3_s', 1)} м³/с",
        ]
        if info.get("river_mask"):
            lines.append(
                f"река, порядок Стралера {info['strahler_order']}; русло ~"
                f"{info.get('river_width_m', 0):.0f} м шириной, "
                f"~{info.get('river_depth_m', 0):.1f} м глубиной"
            )
        if info.get("lake_id", -1) >= 0:
            lines.append(
                f"озеро {info['lake_id']}: уровень {num('lake_level_m', 1)} м, глубина "
                f"{num('lake_depth_m', 1)} м"
            )
        if info.get("coast_type", 0):
            from ..stages.coasts import COAST_TITLES, COAST_TYPES

            lines.append(
                f"берег: {COAST_TITLES[COAST_TYPES[info['coast_type']]]}; волнение "
                f"{num('wave_exposure', 2)}, прилив {num('tidal_range_m', 1)} м, шельф "
                f"{num('shelf_width_km')} км, вода {num('sea_surface_temperature_c', 1)} °C"
            )
        basin = info.get("basin_id", -1)
        tail = " (бессточный)" if info.get("endorheic_mask") else ""
        lines += [
            f"бассейн {basin}{tail}; доля стока, доходящая до моря "
            f"{100 * info.get('outflow_share', 0.0):.0f} %",
            f"размыв {num('v4_erosion_m', 1)} м, отложено {num('v4_deposition_m', 1)} м, "
            f"осадков {num('sediment_thickness_m')} м; темп {num('erosion_rate_m_ma')} м/млн лет",
        ]
        return lines

    def _on_hex_clicked(self, _col: int, _row: int):
        self._show_info()

    def closeEvent(self, event):
        if self._worker is not None and self._worker.isRunning():
            self._worker.cancel()
            self._worker.wait()
        super().closeEvent(event)


def run(project_path: str | None = None) -> int:
    app = QApplication.instance() or QApplication(sys.argv)
    window = MainWindow()
    window.show()
    if project_path:
        window.load_path(project_path)
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(run())
