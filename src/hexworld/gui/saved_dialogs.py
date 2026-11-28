"""Saved worlds (one folder per world), «save world» and the choice of visible map tabs."""

from __future__ import annotations

import html
import os
import subprocess
import sys
from pathlib import Path

from PySide6.QtCore import QSize, Qt, QUrl
from PySide6.QtGui import QDesktopServices, QGuiApplication, QIcon, QPixmap
from PySide6.QtWidgets import (
    QAbstractItemView,
    QCheckBox,
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QFormLayout,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)

from .. import saved_worlds as sw
from .. import views

THUMB = QSize(220, 150)

#: map tabs by the stage that feeds them (titles of the groups in the chooser)
STAGE_TITLES = {
    "v1": "Плиты", "v2": "Материки", "v3": "Геология и горы", "v46": "Массивы, дно и бассейны",
    "v4": "Климат и реки", "v45": "Почвы и вода", "v53": "Океан и лёд", "v54": "Атмосфера",
    "v47": "Высотные пояса", "v51": "Вулканизм", "v50": "Биомы", "v52": "Почвы и недра",
}  # fmt: skip


def font_buttons(dialog: QWidget, key: str) -> QHBoxLayout:
    """«A− A+» buttons that scale the font of a whole window; remembered per window."""
    base = dialog.font().pointSizeF()

    def apply(pt: float):
        f = dialog.font()
        f.setPointSizeF(pt)
        dialog.setFont(f)
        for w in dialog.findChildren(QWidget):
            w.setFont(f)

    def step(d: float):
        prefs = sw.load_prefs()
        pt = max(7.0, min(28.0, float(prefs.get(f"font_{key}", base)) + d))
        prefs[f"font_{key}"] = pt
        sw.save_prefs(prefs)
        apply(pt)

    row = QHBoxLayout()
    for title, d in (("A−", -1.0), ("A+", 1.0)):
        b = QPushButton(title)
        b.setFixedWidth(40)
        b.setToolTip("Шрифт этого окна мельче / крупнее")
        b.clicked.connect(lambda _c=False, dd=d: step(dd))
        row.addWidget(b)
    saved = sw.load_prefs().get(f"font_{key}")
    if saved:
        from PySide6.QtCore import QTimer

        QTimer.singleShot(0, lambda: apply(float(saved)))
    return row


def open_folder(path: str | Path) -> None:
    """Show a folder in the system file manager."""
    path = str(path)
    if sys.platform.startswith("win"):
        os.startfile(path)  # noqa: S606 - the user's own folder
    elif sys.platform == "darwin":
        subprocess.Popen(["open", path])  # noqa: S603, S607
    else:
        QDesktopServices.openUrl(QUrl.fromLocalFile(path))


class SaveWorldDialog(QDialog):
    """Name, «with the world data» and notes of a world to save."""

    def __init__(self, seed: str, has_project: bool, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Сохранить мир")
        form = QFormLayout(self)
        self.title = QLineEdit(seed)
        form.addRow("Название:", self.title)
        self.with_data = QCheckBox("Сохранить весь мир (открывается сразу, без пересчёта)")
        self.with_data.setChecked(has_project)
        self.with_data.setEnabled(has_project)
        self.with_data.setToolTip(
            "Без этой галочки сохраняются только сид и все настройки (код мира): папка весит "
            "килобайты, а при открытии мир пересчитывается точно таким же"
        )
        form.addRow(self.with_data)
        self.notes = QPlainTextEdit()
        self.notes.setPlaceholderText("Заметки (необязательно)")
        self.notes.setFixedHeight(70)
        form.addRow("Заметки:", self.notes)
        root = QLabel(f"Папка миров: {html.escape(str(sw.worlds_root()))}")
        root.setWordWrap(True)
        form.addRow(root)
        bb = QDialogButtonBox(QDialogButtonBox.StandardButton.Save
                              | QDialogButtonBox.StandardButton.Cancel)  # fmt: skip
        bb.accepted.connect(self.accept)
        bb.rejected.connect(self.reject)
        form.addRow(bb)


class SavedWorldsDialog(QDialog):
    """Saved worlds of the worlds folder.

    After ``exec()``: ``action`` is ``"open"`` (show the stored world, or regenerate it when
    only the code was saved), ``"regenerate"``, ``"settings"`` or ``None``; ``meta`` is the
    chosen world.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Сохранённые миры")
        self.resize(1180, 760)
        self.action: str | None = None
        self.meta: dict | None = None
        lay = QVBoxLayout(self)
        top = QHBoxLayout()
        self.root_label = QLabel()
        top.addWidget(self.root_label, 1)
        top.addLayout(font_buttons(self, "saved_worlds"))
        change = QPushButton("📁 Другая папка…")
        change.clicked.connect(self.change_root)
        show = QPushButton("Открыть папку миров")
        show.clicked.connect(lambda: self._open_folder(sw.worlds_root()))
        top.addWidget(change)
        top.addWidget(show)
        lay.addLayout(top)
        self.search = QLineEdit()
        self.search.setPlaceholderText("Поиск по названию, сиду или заметкам…")
        self.search.textChanged.connect(lambda _t: self.fill())
        lay.addWidget(self.search)
        mid = QHBoxLayout()
        self.list = QListWidget()
        self.list.setViewMode(QListWidget.ViewMode.IconMode)
        self.list.setIconSize(THUMB)
        self.list.setResizeMode(QListWidget.ResizeMode.Adjust)
        self.list.setSpacing(8)
        self.list.setWordWrap(True)
        self.list.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.list.itemSelectionChanged.connect(self._selected)
        self.list.itemDoubleClicked.connect(lambda _it: self._finish("open"))
        mid.addWidget(self.list, 3)
        side = QVBoxLayout()
        self.big = QLabel("Выберите мир")
        self.big.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.big.setMinimumSize(340, 230)
        side.addWidget(self.big)
        self.details = QLabel("")
        self.details.setWordWrap(True)
        self.details.setTextFormat(Qt.TextFormat.RichText)
        self.details.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        side.addWidget(self.details, 1)
        mid.addLayout(side, 2)
        lay.addLayout(mid, 1)
        row = QHBoxLayout()
        actions = (
            ("📋 Код мира", self.copy_code),
            ("✏ Переименовать", self.rename),
            ("📂 Папка мира", self.show_folder),
            ("🗑 Удалить", self.delete),
        )
        for title, slot in actions:
            b = QPushButton(title)
            b.clicked.connect(slot)
            row.addWidget(b)
        row.addStretch(1)
        for title, act in (("⚙ В настройки", "settings"), ("🔄 Пересчитать", "regenerate"),
                           ("▶ Открыть", "open")):  # fmt: skip
            b = QPushButton(title)
            b.clicked.connect(lambda _c=False, a=act: self._finish(a))
            row.addWidget(b)
        close = QPushButton("Закрыть")
        close.clicked.connect(self.reject)
        row.addWidget(close)
        lay.addLayout(row)
        self.fill()

    def fill(self):
        self.root_label.setText(f"Папка миров: <b>{html.escape(str(sw.worlds_root()))}</b>")
        self.list.clear()
        needle = self.search.text().strip().lower()
        self._worlds = sw.list_worlds()
        for m in self._worlds:
            hay = f"{m.get('title', '')} {m.get('seed', '')} {m.get('notes', '')}".lower()
            if needle and needle not in hay:
                continue
            kind = "" if m["has_data"] else "\n(только код)"
            it = QListWidgetItem(f"{m['title']}{kind}")
            if m.get("preview"):
                it.setIcon(QIcon(QPixmap(m["preview"])))
            it.setData(Qt.ItemDataRole.UserRole, m["folder"])
            self.list.addItem(it)
        if not self._worlds:
            self.details.setText("Здесь пока пусто. Сохраните мир: «Мир → Сохранить мир» (Ctrl+S).")

    def _current(self) -> dict | None:
        it = self.list.currentItem()
        if it is None:
            return None
        folder = it.data(Qt.ItemDataRole.UserRole)
        return next((m for m in self._worlds if m["folder"] == folder), None)

    def _selected(self):
        m = self._current()
        if m is None:
            return
        if m.get("preview"):
            self.big.setPixmap(QPixmap(m["preview"]).scaled(
                340, 230, Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation))  # fmt: skip
        else:
            self.big.setText("нет картинки")
        w, h, km, mode = m.get("size") or [None] * 4
        e = html.escape
        rows = [
            f"<b>{e(m['title'])}</b>",
            f"сид: <code>{e(m['seed'])}</code>",
            f"размер: {w}×{h}, {km} км, {e(str(mode))}" if w else "",
            f"сохранён: {e(m.get('saved', ''))} (создан {e(m.get('created', ''))})",
            "данные мира: сохранены — откроется сразу" if m["has_data"]
            else "сохранены сид и настройки — мир пересчитается при открытии",
            f"версия: HexWorld {e(m.get('product_version', '?'))}, "
            f"генератор {e(m.get('generator_version', '?'))}",
            f"код: <code>{e(m['code'])}</code>",
        ]  # fmt: skip
        if m.get("notes"):
            rows.append(f"заметки: {e(m['notes'])}")
        self.details.setText("<br>".join(r for r in rows if r))

    def _finish(self, action: str):
        m = self._current()
        if m is None:
            return
        self.meta = m
        self.action = action
        self.accept()

    def copy_code(self):
        m = self._current()
        if m:
            QGuiApplication.clipboard().setText(m["code"])
            self.details.setText(self.details.text() + "<br><i>код скопирован</i>")

    def rename(self):
        m = self._current()
        if m is None:
            return
        title, ok = QInputDialog.getText(self, "Переименовать мир", "Новое название:",
                                         text=m["title"])  # fmt: skip
        if ok and title.strip():
            try:
                sw.rename_world(m["folder"], title)
            except (OSError, ValueError) as exc:
                QMessageBox.warning(self, "Переименовать", str(exc))
            self.fill()

    def show_folder(self):
        m = self._current()
        if m:
            self._open_folder(m["folder"])

    def _open_folder(self, path):
        Path(path).mkdir(parents=True, exist_ok=True)
        open_folder(path)

    def delete(self):
        m = self._current()
        if m is None:
            return
        if QMessageBox.question(self, "Удалить мир",
                                f"Удалить папку мира «{m['title']}» со всеми данными?\n"
                                f"{m['folder']}") != QMessageBox.StandardButton.Yes:  # fmt: skip
            return
        try:
            sw.delete_world(m["folder"])
        except (OSError, ValueError) as exc:
            QMessageBox.warning(self, "Удалить", str(exc))
        self.fill()

    def change_root(self):
        path = QFileDialog.getExistingDirectory(
            self, "Папка сохранённых миров", str(sw.worlds_root())
        )
        if path:
            sw.set_worlds_root(path)
            self.fill()


class ViewsChooser(QWidget):
    """Check boxes of every map tab, grouped by the stage that feeds it."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.boxes: dict[str, QCheckBox] = {}
        lay = QVBoxLayout(self)
        hint = QLabel(
            "Показывать на карте только отмеченные вкладки. Это настройка отображения: "
            "мир считается полностью, скрытую вкладку можно вернуть в любой момент."
        )
        hint.setWordWrap(True)
        lay.addWidget(hint)
        quick = QHBoxLayout()
        for title, ids in (("Все", None), ("Только высоты и биомы", ("heights", "biomes")),
                           ("Основные", ("heights", "temperature", "precipitation",
                                         "hydrology", "biomes", "soil_fertility",
                                         "resources_by_type", "mineral_resources"))):  # fmt: skip
            b = QPushButton(title)
            b.clicked.connect(lambda _c=False, s=ids: self.set_shown(s))
            quick.addWidget(b)
        quick.addStretch(1)
        lay.addLayout(quick)
        groups: dict[str, list] = {}
        for spec in views.VIEWS.values():
            if spec.status != "ready":
                continue
            groups.setdefault(spec.stage or "v1", []).append(spec)
        inner = QWidget()
        grid = QGridLayout(inner)
        for k, (stage, specs) in enumerate(groups.items()):
            box = QGroupBox(STAGE_TITLES.get(stage, stage))
            col = QVBoxLayout(box)
            for spec in specs:
                cb = QCheckBox(spec.title)
                self.boxes[spec.id] = cb
                col.addWidget(cb)
            grid.addWidget(box, k // 3, k % 3, Qt.AlignmentFlag.AlignTop)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(inner)
        lay.addWidget(scroll, 1)
        self.set_shown(sw.visible_views(list(views.VIEWS)))

    def set_shown(self, ids):
        ids = set(views.VIEWS) if ids is None else set(ids)
        for vid, cb in self.boxes.items():
            cb.setChecked(vid in ids)

    def shown(self) -> list[str]:
        return [vid for vid, cb in self.boxes.items() if cb.isChecked()]

    def save(self) -> None:
        shown = self.shown()
        planned = [v for v, s in views.VIEWS.items() if s.status != "ready"]
        sw.set_visible_views(shown + planned if shown else list(views.VIEWS), list(views.VIEWS))


class ViewsDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Вкладки карты")
        self.resize(900, 620)
        lay = QVBoxLayout(self)
        self.chooser = ViewsChooser(self)
        lay.addWidget(self.chooser, 1)
        bb = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok
                              | QDialogButtonBox.StandardButton.Cancel)  # fmt: skip
        bb.accepted.connect(self._ok)
        bb.rejected.connect(self.reject)
        lay.addWidget(bb)

    def _ok(self):
        self.chooser.save()
        self.accept()
