"""Minimal PySide6 viewer. The core never imports this module."""

from __future__ import annotations

import sys
import traceback
from dataclasses import replace
from pathlib import Path

from PySide6.QtCore import QThread, Signal
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
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from .. import render
from ..config import MODES, PRESETS, PROFILES, Config
from ..io import load_project, save_project
from ..pipeline import generate
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
    "validate": "проверка",
    "done": "готово",
}


class Worker(QThread):
    progress = Signal(str, float)
    succeeded = Signal(object)
    failed = Signal(str)

    def __init__(self, job, parent=None):
        super().__init__(parent)
        self._job = job

    def run(self):
        try:
            self.succeeded.emit(self._job(self.progress.emit))
        except Exception as exc:  # shown to the user, never swallowed
            self.failed.emit(f"{exc}\n\n{traceback.format_exc()}")


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle(f"HexWorld {GENERATOR_VERSION} - тектоническая основа")
        self.resize(1360, 860)
        self.project: Project | None = None
        self._worker: Worker | None = None

        self.view = MapView()
        self.view.hexHovered.connect(lambda text: self.statusBar().showMessage(text))
        self.view.plateSelected.connect(self._on_plate_selected)

        side = QWidget()
        side.setFixedWidth(330)
        col = QVBoxLayout(side)

        box = QGroupBox("Параметры мира")
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
        self.hex_km = QDoubleSpinBox()
        self.hex_km.setRange(0.1, 2000.0)
        self.hex_km.setDecimals(1)
        self.hex_km.setValue(40.0)
        self.hex_km.setSuffix(" км")
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
        self.origin_x = self._km_box()
        self.origin_y = self._km_box()
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
        form.addRow("Профиль", self.profile)
        form.addRow("Начало окна X", self.origin_x)
        form.addRow("Начало окна Y", self.origin_y)
        form.addRow(self.pin_world)
        col.addWidget(box)

        buttons = QHBoxLayout()
        self.generate_btn = QPushButton("Сгенерировать")
        self.generate_btn.clicked.connect(self.start_generation)
        self.open_btn = QPushButton("Открыть…")
        self.open_btn.clicked.connect(self.open_project)
        self.save_btn = QPushButton("Сохранить…")
        self.save_btn.clicked.connect(self.save_project)
        self.save_btn.setEnabled(False)
        for b in (self.generate_btn, self.open_btn, self.save_btn):
            buttons.addWidget(b)
        col.addLayout(buttons)
        self.png_btn = QPushButton("Экспорт PNG текущего слоя…")
        self.png_btn.clicked.connect(self.export_png)
        self.png_btn.setEnabled(False)
        col.addWidget(self.png_btn)

        box = QGroupBox("Отображение (не меняет данные)")
        form = QFormLayout(box)
        self.layer = QComboBox()
        for name in render.LAYERS:
            self.layer.addItem(render.LAYER_TITLES[name], name)
        self.layer.currentIndexChanged.connect(
            lambda _i: self.view.set_layer(self.layer.currentData())
        )
        self.show_boundaries = QCheckBox("Линии границ")
        self.show_boundaries.setChecked(True)
        self.show_arrows = QCheckBox("Векторы движения")
        self.show_arrows.setChecked(True)
        for cb in (self.show_boundaries, self.show_arrows):
            cb.toggled.connect(self._update_overlays)
        fit = QPushButton("Показать всю карту")
        fit.clicked.connect(self.view.fit)
        form.addRow("Слой", self.layer)
        form.addRow(self.show_boundaries)
        form.addRow(self.show_arrows)
        form.addRow(fit)
        legend = QLabel(
            "<span style='color:#d63031'>■</span> сближение "
            "<span style='color:#0984e3'>■</span> расхождение "
            "<span style='color:#00a86b'>■</span> сдвиг "
            "<span style='color:#969696'>■</span> спокойная"
        )
        form.addRow(legend)
        col.addWidget(box)

        box = QGroupBox("Проект / выбранная плита")
        inner = QVBoxLayout(box)
        self.info = QPlainTextEdit()
        self.info.setReadOnly(True)
        inner.addWidget(self.info)
        col.addWidget(box, 1)

        central = QWidget()
        row = QHBoxLayout(central)
        row.addWidget(side)
        row.addWidget(self.view, 1)
        self.setCentralWidget(central)

        self.progress = QProgressBar()
        self.progress.setRange(0, 1000)
        self.progress.setFixedWidth(260)
        self.progress.setVisible(False)
        self.statusBar().addPermanentWidget(self.progress)
        self.statusBar().showMessage("Готов")

    # -- helpers -----------------------------------------------------------
    @staticmethod
    def _spin(lo, hi, value):
        box = QSpinBox()
        box.setRange(lo, hi)
        box.setValue(value)
        return box

    @staticmethod
    def _km_box():
        box = QDoubleSpinBox()
        box.setRange(-1e7, 1e7)
        box.setDecimals(0)
        box.setSingleStep(500.0)
        box.setSuffix(" км")
        return box

    def _apply_preset(self, _index):
        name = self.preset.currentData()
        if name is None:
            return
        data = PRESETS[name]
        self.width_box.setValue(data["width"])
        self.height_box.setValue(data["height"])
        self.hex_km.setValue(data["hex_spacing_km"])
        self.mode.setCurrentText(data["mode"])

    def _update_overlays(self):
        self.view.set_overlays(self.show_boundaries.isChecked(), self.show_arrows.isChecked())

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
        )
        if self.pin_world.isChecked() and self.project is not None:
            g = self.project.geometry
            config = replace(
                config,
                seed=self.project.config.seed,
                simulation_bounds_km=tuple(g.simulation_bounds.as_list()),
                plates=replace(self.project.config.plates, count=g.plate_count),
            )
        return config

    def _set_busy(self, busy: bool):
        for w in (self.generate_btn, self.open_btn):
            w.setEnabled(not busy)
        self.save_btn.setEnabled(not busy and self.project is not None)
        self.png_btn.setEnabled(not busy and self.project is not None)
        self.progress.setVisible(busy)

    def _run(self, job, on_done):
        self._set_busy(True)
        self.progress.setValue(0)
        worker = Worker(job, self)
        worker.progress.connect(self._on_progress)
        worker.succeeded.connect(on_done)
        worker.failed.connect(self._on_failed)
        worker.finished.connect(lambda: self._set_busy(False))
        self._worker = worker
        worker.start()

    def _on_progress(self, stage: str, fraction: float):
        self.progress.setValue(int(1000 * fraction))
        self.statusBar().showMessage(f"Генерация: {STAGE_TITLES.get(stage, stage)}…")

    def _on_failed(self, message: str):
        self.statusBar().showMessage("Ошибка")
        QMessageBox.critical(self, "HexWorld - ошибка", message)

    # -- actions -----------------------------------------------------------
    def start_generation(self):
        try:
            config = self.build_config()
        except ValueError as exc:
            QMessageBox.warning(self, "Параметры", str(exc))
            return
        previous = self.project
        self._run(
            lambda progress: generate(config, progress=progress, reuse=previous), self._on_generated
        )

    def _on_generated(self, project: Project):
        self._show_project(project, "Сгенерировано")

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
        layer = self.layer.currentData()
        path, _ = QFileDialog.getSaveFileName(self, "Экспорт PNG", f"{layer}.png", "PNG (*.png)")
        if path:
            render.write_png(Path(path), render.render_project(self.project, layer))
            self.statusBar().showMessage(f"Изображение сохранено: {path}")

    def _show_project(self, project: Project, message: str, sync_form: bool = False):
        self.project = project
        if sync_form:
            c = project.config
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
        self.view.set_project(project)
        self._update_overlays()
        self.info.setPlainText(self._project_text())
        total = project.timings.get("total")
        ok = "проверка пройдена" if project.validation.get("ok") else "ПРОВЕРКА НЕ ПРОЙДЕНА"
        timing = f", {total:.1f} с" if total else ""
        self.statusBar().showMessage(f"{message}: {ok}{timing}")
        if not project.validation.get("ok"):
            QMessageBox.warning(self, "Проверка", "\n".join(project.validation["errors"]))

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
            f"{p.stats.get('plates_visible')} в окне",
            f"граничных рёбер: {len(p.boundaries)}",
            f"континентальная кора в окне: "
            f"{100 * p.stats.get('continental_fraction_visible', 0):.1f} %",
        ]
        if p.stats.get("reused_stages"):
            lines.append("взято из кэша: " + ", ".join(p.stats["reused_stages"]))
        for w in p.warnings:
            lines.append(f"⚠ {w}")
        lines.append("")
        lines.append("Щёлкните по карте, чтобы выбрать плиту.")
        return "\n".join(lines)

    def _on_plate_selected(self, plate_id: int):
        if self.view.selected_plate == plate_id:
            self.view.select_plate(None)
            self.info.setPlainText(self._project_text())
            return
        self.view.select_plate(plate_id)
        p = self.project.plate(plate_id)
        names = ", ".join(self.project.plate(n)["name"] for n in p["neighbors"])
        centroid = p["centroid_km"] or p["site_km"]
        vis_cont = p["visible_continental_fraction"]
        self.info.setPlainText(
            "\n".join(
                [
                    f"Плита {p['id']}: {p['name']}",
                    f"цвет: {p['color']}   класс: {p['size_class']}"
                    + ("   (центр вне области симуляции)" if p["halo"] else ""),
                    f"площадь в области симуляции: {p['area_km2'] / 1e6:,.2f} млн км²",
                    f"видимая площадь: {p['visible_area_km2'] / 1e6:,.2f} млн км² "
                    f"({p['visible_hexes']} гексов)",
                    f"центр: x {centroid[0]:,.0f} км, y {centroid[1]:,.0f} км",
                    f"кора: континентальная {100 * p['continental_fraction']:.0f} %, "
                    f"океаническая {100 * p['oceanic_fraction']:.0f} %",
                    "в окне: континентальная "
                    + (f"{100 * vis_cont:.0f} %" if vis_cont is not None else "-"),
                    f"скорость: vx {p['velocity_mm_yr'][0]:+.1f}, vy {p['velocity_mm_yr'][1]:+.1f}"
                    f" мм/год (|v| {p['speed_mm_yr']:.1f}); +x восток, +y юг",
                    f"вращение: {p['omega_deg_myr']:+.3f} °/млн лет",
                    f"соседи: {names or '-'}",
                    "",
                    "Повторный щелчок по плите снимает выделение.",
                ]
            )
        )

    def closeEvent(self, event):
        if self._worker is not None and self._worker.isRunning():
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
