"""«Мои миры» (history and favourites), comparison of two worlds, batch of random worlds."""

from __future__ import annotations

import html

from PySide6.QtCore import QSize, Qt, QThread, Signal
from PySide6.QtGui import QGuiApplication, QIcon, QImage, QPixmap
from PySide6.QtWidgets import (
    QAbstractItemView,
    QComboBox,
    QDialog,
    QGridLayout,
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QPushButton,
    QSpinBox,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
)

from .. import settings_schema as ss
from .. import world_tools as wt
from ..config import Config
from ..world_library import WorldLibrary

THUMB = QSize(220, 150)


def _pixmap(rgb) -> QPixmap:
    h, w = rgb.shape[:2]
    buf = rgb.copy()
    return QPixmap.fromImage(QImage(buf.data, w, h, 3 * w, QImage.Format.Format_RGB888).copy())


def _entry_pixmap(library: WorldLibrary, entry: dict) -> QPixmap | None:
    p = library.thumb_path(entry)
    return QPixmap(str(p)) if p is not None else None


class LibraryDialog(QDialog):
    """History of generated worlds and favourites.

    After ``exec()``: ``action`` is ``"generate"`` (make this world again), ``"settings"``
    (open it in «Новый мир») or ``None``; ``entry`` is the chosen entry.
    """

    def __init__(self, library: WorldLibrary, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Мои миры — история и избранное")
        self.resize(1180, 760)
        self.library = library
        self.action: str | None = None
        self.entry: dict | None = None
        lay = QVBoxLayout(self)
        top = QHBoxLayout()
        self.filter = QComboBox()
        self.filter.addItems(["Все миры", "⭐ Только избранное"])
        self.filter.currentIndexChanged.connect(lambda _i: self.fill())
        self.search = QLineEdit()
        self.search.setPlaceholderText("Поиск по названию или сиду…")
        self.search.textChanged.connect(lambda _t: self.fill())
        top.addWidget(self.filter)
        top.addWidget(self.search, 1)
        lay.addLayout(top)
        mid = QHBoxLayout()
        self.list = QListWidget()
        self.list.setViewMode(QListWidget.ViewMode.IconMode)
        self.list.setIconSize(THUMB)
        self.list.setResizeMode(QListWidget.ResizeMode.Adjust)
        self.list.setSpacing(8)
        self.list.setWordWrap(True)
        self.list.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        self.list.itemSelectionChanged.connect(self._selected)
        self.list.itemDoubleClicked.connect(lambda _it: self._finish("generate"))
        mid.addWidget(self.list, 3)
        side = QVBoxLayout()
        self.big = QLabel("Выберите мир")
        self.big.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.big.setMinimumSize(340, 230)
        side.addWidget(self.big)
        self.details = QLabel("")
        self.details.setWordWrap(True)
        self.details.setTextFormat(Qt.TextFormat.RichText)
        side.addWidget(self.details, 1)
        mid.addLayout(side, 2)
        lay.addLayout(mid, 1)
        buttons = QHBoxLayout()
        self.fav_btn = QPushButton("⭐ В избранное")
        self.fav_btn.clicked.connect(self.toggle_favourite)
        self.rename_btn = QPushButton("✏ Переименовать")
        self.rename_btn.clicked.connect(self.rename)
        self.code_btn = QPushButton("📋 Код мира")
        self.code_btn.clicked.connect(self.copy_code)
        self.compare_btn = QPushButton("⚖ Сравнить два выбранных")
        self.compare_btn.setToolTip("Выделите два мира (Ctrl + щелчок)")
        self.compare_btn.clicked.connect(self.compare)
        self.delete_btn = QPushButton("🗑 Удалить")
        self.delete_btn.clicked.connect(self.delete)
        for b in (self.fav_btn, self.rename_btn, self.code_btn, self.compare_btn, self.delete_btn):
            buttons.addWidget(b)
        buttons.addStretch(1)
        self.settings_btn = QPushButton("⚙ Открыть в настройках")
        self.settings_btn.clicked.connect(lambda: self._finish("settings"))
        self.generate_btn = QPushButton("▶ Сгенерировать этот мир")
        self.generate_btn.clicked.connect(lambda: self._finish("generate"))
        close = QPushButton("Закрыть")
        close.clicked.connect(self.reject)
        for b in (self.settings_btn, self.generate_btn, close):
            buttons.addWidget(b)
        lay.addLayout(buttons)
        self.fill()

    def fill(self):
        self.list.clear()
        only_fav = self.filter.currentIndex() == 1
        needle = self.search.text().strip().lower()
        for e in self.library.entries:
            if only_fav and not e.get("favourite"):
                continue
            if needle and needle not in (e["title"] + " " + e["seed"]).lower():
                continue
            star = "⭐ " if e.get("favourite") else ""
            sketch = " (эскиз)" if e.get("sketch") else ""
            item = QListWidgetItem(f"{star}{e['title']}{sketch}\n{e['created']}")
            pm = _entry_pixmap(self.library, e)
            if pm is not None:
                item.setIcon(QIcon(pm))
            item.setData(Qt.ItemDataRole.UserRole, e["id"])
            item.setSizeHint(QSize(THUMB.width() + 16, THUMB.height() + 48))
            self.list.addItem(item)
        self._selected()

    def selected_entries(self) -> list[dict]:
        out = []
        for it in self.list.selectedItems():
            e = self.library.get(it.data(Qt.ItemDataRole.UserRole))
            if e is not None:
                out.append(e)
        return out

    def _selected(self):
        sel = self.selected_entries()
        one = len(sel) == 1
        for b in (self.fav_btn, self.rename_btn, self.code_btn, self.settings_btn,
                  self.generate_btn):  # fmt: skip
            b.setEnabled(one)
        self.delete_btn.setEnabled(bool(sel))
        self.compare_btn.setEnabled(len(sel) == 2)
        if not one:
            self.big.setText("Выберите мир" if not sel else f"Выбрано миров: {len(sel)}")
            self.details.setText("")
            return
        e = sel[0]
        self.fav_btn.setText("☆ Убрать из избранного" if e.get("favourite") else "⭐ В избранное")
        pm = _entry_pixmap(self.library, e)
        if pm is not None:
            self.big.setPixmap(pm.scaled(340, 230, Qt.AspectRatioMode.KeepAspectRatio))
        rows = [r for r in wt.summary_rows(e.get("summary", {})) if r[1] != "—"]
        esc = html.escape
        self.details.setText(
            f"<b>{esc(e['title'])}</b> · сид <b>{esc(e['seed'])}</b> · {esc(e['created'])}<br>"
            + "<br>".join(f"{esc(a)}: <b>{esc(b)}</b>" for a, b in rows)
        )

    def toggle_favourite(self):
        sel = self.selected_entries()
        if len(sel) == 1:
            self.library.set_favourite(sel[0]["id"], not sel[0].get("favourite"))
            self._refill_keep(sel[0]["id"])

    def rename(self):
        sel = self.selected_entries()
        if len(sel) != 1:
            return
        title, ok = QInputDialog.getText(self, "Название мира", "Название:", text=sel[0]["title"])
        if ok:
            self.library.rename(sel[0]["id"], title)
            self._refill_keep(sel[0]["id"])

    def copy_code(self):
        sel = self.selected_entries()
        if len(sel) == 1:
            QGuiApplication.clipboard().setText(sel[0]["code"])

    def delete(self):
        sel = self.selected_entries()
        if not sel:
            return
        ok = QMessageBox.question(self, "Удалить", f"Удалить из списка миров: {len(sel)}?")
        if ok == QMessageBox.StandardButton.Yes:
            for e in sel:
                self.library.remove(e["id"])
            self.fill()

    def compare(self):
        sel = self.selected_entries()
        if len(sel) == 2:
            CompareDialog(
                [(e["title"], e.get("summary", {}), _entry_pixmap(self.library, e)) for e in sel],
                self,
            ).exec()

    def _refill_keep(self, eid: str):
        self.fill()
        for i in range(self.list.count()):
            if self.list.item(i).data(Qt.ItemDataRole.UserRole) == eid:
                self.list.item(i).setSelected(True)

    def _finish(self, action: str):
        sel = self.selected_entries()
        if len(sel) != 1:
            return
        self.action, self.entry = action, sel[0]
        self.accept()


class CompareDialog(QDialog):
    """Two worlds side by side: pictures and the numbers that describe them."""

    def __init__(self, worlds: list[tuple[str, dict, QPixmap | None]], parent=None):
        super().__init__(parent)
        self.setWindowTitle("Сравнение миров")
        self.resize(980, 760)
        lay = QVBoxLayout(self)
        grid = QGridLayout()
        for i, (title, _s, pm) in enumerate(worlds):
            head = QLabel(f"<b>{title}</b>")
            head.setAlignment(Qt.AlignmentFlag.AlignCenter)
            grid.addWidget(head, 0, i)
            pic = QLabel("нет картинки")
            pic.setAlignment(Qt.AlignmentFlag.AlignCenter)
            if pm is not None:
                pic.setPixmap(pm.scaled(440, 300, Qt.AspectRatioMode.KeepAspectRatio))
            grid.addWidget(pic, 1, i)
        lay.addLayout(grid)
        rows = [wt.summary_rows(s) for _t, s, _p in worlds]
        n = max(len(r) for r in rows)
        table = QTableWidget(n, 1 + len(worlds))
        table.setHorizontalHeaderLabels(["Показатель"] + [t for t, _s, _p in worlds])
        table.verticalHeader().setVisible(False)
        for r in range(n):
            label = next((rr[r][0] for rr in rows if r < len(rr)), "")
            table.setItem(r, 0, QTableWidgetItem(label))
            values = [rr[r][1] if r < len(rr) else "—" for rr in rows]
            for c, v in enumerate(values):
                item = QTableWidgetItem(v)
                if len(set(values)) > 1:
                    item.setBackground(Qt.GlobalColor.yellow)
                    item.setForeground(Qt.GlobalColor.black)
                table.setItem(r, 1 + c, item)
        table.resizeColumnsToContents()
        table.horizontalHeader().setStretchLastSection(True)
        self.table = table
        lay.addWidget(table, 1)
        close = QPushButton("Закрыть")
        close.clicked.connect(self.accept)
        lay.addWidget(close, 0, Qt.AlignmentFlag.AlignRight)


class _BatchWorker(QThread):
    item_done = Signal(int, object, object, str)  # index, sketch project, full config, error
    all_done = Signal()

    def __init__(self, configs: list[Config], parent=None):
        super().__init__(parent)
        self.configs = configs
        self._stop = False

    def stop(self):
        self._stop = True

    def run(self):
        from ..pipeline import GenerationCancelled, generate

        def progress(_s, _f):
            if self._stop:
                raise GenerationCancelled()

        for i, cfg in enumerate(self.configs):
            if self._stop:
                break
            try:
                project = generate(wt.preview_config(cfg), progress=progress)
            except GenerationCancelled:
                break
            except Exception as exc:  # noqa: BLE001
                self.item_done.emit(i, None, cfg, str(exc))
                continue
            self.item_done.emit(i, project, cfg, "")
        self.all_done.emit()


BATCH_MODES = (
    ("seeds", "Разные сиды, мои настройки"),
    ("locked", "Случайные настройки (с моими замками)"),
    ("all", "Полностью случайные миры"),
)


class BatchDialog(QDialog):
    """Many quick sketches at once; pick the world you like.

    After ``exec()``: ``action`` (``"generate"`` / ``"settings"``) and ``config`` (the chosen
    world at the full size of the settings it was made from).
    """

    def __init__(self, values: dict, seed: str, locked: set, library: WorldLibrary | None,
                 parent=None):  # fmt: skip
        super().__init__(parent)
        self.setWindowTitle("Пакет миров — выберите понравившийся")
        self.resize(1240, 820)
        self.values = dict(values)
        self.seed = seed
        self.locked = set(locked)
        self.library = library
        self.action: str | None = None
        self.config: Config | None = None
        self._worker: _BatchWorker | None = None
        self._results: dict[int, tuple] = {}
        lay = QVBoxLayout(self)
        top = QHBoxLayout()
        top.addWidget(QLabel("Сколько миров:"))
        self.count = QSpinBox()
        self.count.setRange(2, 60)
        self.count.setValue(12)
        top.addWidget(self.count)
        top.addWidget(QLabel("Как менять:"))
        self.mode = QComboBox()
        for key, title in BATCH_MODES:
            self.mode.addItem(title, key)
        top.addWidget(self.mode)
        top.addWidget(QLabel("Показывать:"))
        self.view = QComboBox()
        from .world_settings import PREVIEW_VIEWS

        for key, title in PREVIEW_VIEWS:
            self.view.addItem(title, key)
        self.view.currentIndexChanged.connect(lambda _i: self._redraw())
        top.addWidget(self.view)
        self.run_btn = QPushButton("▶ Создать эскизы")
        self.run_btn.clicked.connect(self.run_batch)
        top.addWidget(self.run_btn)
        top.addStretch(1)
        lay.addLayout(top)
        self.status = QLabel(
            "Каждый эскиз — несколько секунд. Двойной щелчок по миру — создать его полностью."
        )
        lay.addWidget(self.status)
        self.list = QListWidget()
        self.list.setViewMode(QListWidget.ViewMode.IconMode)
        self.list.setIconSize(THUMB)
        self.list.setResizeMode(QListWidget.ResizeMode.Adjust)
        self.list.setSpacing(8)
        self.list.itemDoubleClicked.connect(lambda _it: self._finish("generate"))
        self.list.itemSelectionChanged.connect(self._selected)
        lay.addWidget(self.list, 1)
        self.info = QLabel("")
        self.info.setWordWrap(True)
        lay.addWidget(self.info)
        bottom = QHBoxLayout()
        self.fav_btn = QPushButton("⭐ В «Мои миры»")
        self.fav_btn.clicked.connect(self.save_selected)
        self.fav_btn.setEnabled(library is not None)
        bottom.addWidget(self.fav_btn)
        bottom.addStretch(1)
        self.settings_btn = QPushButton("⚙ Открыть в настройках")
        self.settings_btn.clicked.connect(lambda: self._finish("settings"))
        self.generate_btn = QPushButton("▶ Создать этот мир полностью")
        self.generate_btn.clicked.connect(lambda: self._finish("generate"))
        close = QPushButton("Закрыть")
        close.clicked.connect(self.reject)
        for b in (self.settings_btn, self.generate_btn, close):
            bottom.addWidget(b)
        lay.addLayout(bottom)
        self._selected()

    def make_configs(self, n: int) -> list[Config]:
        import random

        rng = random.SystemRandom()
        mode = self.mode.currentData()
        out = []
        for _ in range(n):
            seed = ss.random_seed(rng)
            if mode == "seeds":
                values = self.values
            elif mode == "locked":
                values = ss.randomize(self.values, self.locked)
            else:
                values = ss.randomize(self.values, ss.default_locked())
            out.append(ss.config_from_values(values, seed=seed))
        return out

    def run_batch(self):
        if self._worker is not None and self._worker.isRunning():
            self._worker.stop()
            self.run_btn.setText("▶ Создать эскизы")
            return
        self.list.clear()
        self._results.clear()
        configs = self.make_configs(self.count.value())
        self._worker = _BatchWorker(configs, self)
        self._worker.item_done.connect(self._item_done)
        self._worker.all_done.connect(self._all_done)
        self._total = len(configs)
        self.run_btn.setText("■ Остановить")
        self.status.setText(f"Считаю эскизы: 0 из {self._total}…")
        self._worker.start()

    def _item_done(self, i: int, project, cfg: Config, error: str):
        self._results[i] = (project, cfg, error)
        item = QListWidgetItem(f"{cfg.seed}" + (f"\nошибка: {error[:40]}" if error else ""))
        item.setData(Qt.ItemDataRole.UserRole, i)
        if project is not None:
            item.setIcon(QIcon(_pixmap(wt.thumbnail(project, self.view.currentData(), 300))))
        item.setSizeHint(QSize(THUMB.width() + 16, THUMB.height() + 40))
        self.list.addItem(item)
        self.status.setText(f"Считаю эскизы: {len(self._results)} из {self._total}…")

    def _all_done(self):
        self.run_btn.setText("▶ Создать эскизы")
        self.status.setText(
            f"Готово: {len(self._results)} миров. Двойной щелчок — создать мир полностью."
        )

    def _redraw(self):
        for r in range(self.list.count()):
            item = self.list.item(r)
            project = self._results.get(item.data(Qt.ItemDataRole.UserRole), (None,))[0]
            if project is not None:
                item.setIcon(QIcon(_pixmap(wt.thumbnail(project, self.view.currentData(), 300))))

    def _current(self):
        items = self.list.selectedItems()
        if not items:
            return None
        return self._results.get(items[0].data(Qt.ItemDataRole.UserRole))

    def _selected(self):
        cur = self._current()
        ok = cur is not None and cur[0] is not None
        for b in (self.settings_btn, self.generate_btn):
            b.setEnabled(ok)
        self.fav_btn.setEnabled(ok and self.library is not None)
        if ok:
            rows = [r for r in wt.summary_rows(wt.world_summary(cur[0])) if r[1] != "—"]
            self.info.setText(
                f"Сид {cur[1].seed}: " + " · ".join(f"{a}: {b}" for a, b in rows[3:12])
            )

    def save_selected(self):
        cur = self._current()
        if cur is None or cur[0] is None or self.library is None:
            return
        entry = self.library.add(cur[0], config=cur[1])
        self.library.set_favourite(entry["id"], True)
        self.status.setText(f"Мир {cur[1].seed} добавлен в «Мои миры» (избранное).")

    def _finish(self, action: str):
        cur = self._current()
        if cur is None or cur[0] is None:
            return
        self._stop()
        self.action, self.config = action, cur[1]
        self.accept()

    def _stop(self):
        if self._worker is not None:
            self._worker.stop()
            self._worker.wait(20000)

    def reject(self):
        self._stop()
        super().reject()


class GlobeDialog(QDialog):
    """The world on a rotating globe: drag to turn, wheel to zoom."""

    def __init__(self, project, parent=None):
        from .. import views
        from .world_settings import PREVIEW_VIEWS

        super().__init__(parent)
        self.setWindowTitle(f"Глобус — мир «{project.config.seed}»")
        self.resize(720, 800)
        self.project = project
        self.lon0, self.lat0, self.zoom = 0.0, 20.0, 1.0
        self._drag = None
        self._rgb_cache: dict = {}
        self._views = views
        lay = QVBoxLayout(self)
        top = QHBoxLayout()
        top.addWidget(QLabel("Слой:"))
        self.view = QComboBox()
        for key, title in PREVIEW_VIEWS:
            if views.VIEWS[key].availability(project)[0]:
                self.view.addItem(title, key)
        self.view.currentIndexChanged.connect(lambda _i: self.redraw())
        top.addWidget(self.view)
        top.addStretch(1)
        lay.addLayout(top)
        self.image = QLabel()
        self.image.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.image.setMinimumSize(600, 600)
        lay.addWidget(self.image, 1)
        from ..globe import window_extent

        ext = window_extent(project)
        self.note = QLabel(
            f"Карта занимает широты от {ext['lat_south']:+.0f}° до {ext['lat_north']:+.0f}° и "
            f"{ext['lon_span']:.0f}° долготы — {100 * ext['surface_share']:.0f} % поверхности "
            "планеты; тёмное — за пределами карты. Перетаскивайте мышью, колесо — масштаб."
        )
        self.note.setWordWrap(True)
        lay.addWidget(self.note)
        self.redraw()

    def _hex_rgb(self):
        key = self.view.currentData()
        if key not in self._rgb_cache:
            spec = self._views.VIEWS[key]
            self._rgb_cache[key] = self._views.view_rgb(
                self.project, key, spec.default_mode(), spec.default_overlays()
            )
        return self._rgb_cache[key]

    def redraw(self):
        from ..globe import render_globe

        if self.view.count() == 0:
            return
        rgb = render_globe(self.project, self._hex_rgb(), 600, self.lon0, self.lat0, self.zoom)
        self.image.setPixmap(_pixmap(rgb))

    def mousePressEvent(self, event):
        self._drag = event.position()

    def mouseMoveEvent(self, event):
        if self._drag is None:
            return
        d = event.position() - self._drag
        self._drag = event.position()
        k = 0.3 / self.zoom
        self.lon0 = (self.lon0 - d.x() * k + 180) % 360 - 180
        self.lat0 = max(-89.0, min(89.0, self.lat0 + d.y() * k))
        self.redraw()

    def mouseReleaseEvent(self, _event):
        self._drag = None

    def wheelEvent(self, event):
        self.zoom = max(0.6, min(6.0, self.zoom * 1.15 ** (event.angleDelta().y() / 120.0)))
        self.redraw()
