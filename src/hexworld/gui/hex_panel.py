"""Side panel «Информация о гексе»: everything about the clicked hex, grouped and searchable."""

from __future__ import annotations

from PySide6.QtCore import QPointF, QRectF, Qt, Signal
from PySide6.QtGui import QColor, QFont, QPainter, QPen, QPolygonF
from PySide6.QtWidgets import (
    QApplication,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QPushButton,
    QSizePolicy,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
)

from ..hex_report import MONTHS, HexReport, hex_report

#: Sections shown collapsed until the user opens them.
_COLLAPSED = {"months", "raw"}


class ClimateChart(QWidget):
    """Climograph: monthly precipitation bars and the temperature curve."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumHeight(130)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.data: dict | None = None

    def set_data(self, data: dict | None):
        self.data = data
        self.setVisible(data is not None)
        self.update()

    def paintEvent(self, _event):
        if not self.data:
            return
        t = [v if v is not None else 0.0 for v in self.data["t"]]
        p = [max(v or 0.0, 0.0) for v in self.data["p"]]
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        pal = self.palette()
        w, h = self.width(), self.height()
        left, right, top, bottom = 34, 34, 8, 18
        cw, ch = w - left - right, h - top - bottom
        painter.fillRect(self.rect(), pal.base())
        pmax = max(max(p), 50.0)
        tmin, tmax = min(min(t), -5.0), max(max(t), 25.0)
        bar = cw / 12.0
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QColor(70, 130, 220, 200))
        for i, v in enumerate(p):
            bh = ch * v / pmax
            painter.drawRect(QRectF(left + i * bar + 2, top + ch - bh, bar - 4, bh))

        # zero line of temperature
        def ty(v):
            return top + ch * (tmax - v) / (tmax - tmin)

        if tmin < 0 < tmax:
            painter.setPen(QPen(QColor(150, 150, 150), 1, Qt.PenStyle.DashLine))
            painter.drawLine(QPointF(left, ty(0)), QPointF(left + cw, ty(0)))
        painter.setPen(QPen(QColor(220, 60, 50), 2.2))
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.drawPolyline(
            QPolygonF([QPointF(left + (i + 0.5) * bar, ty(v)) for i, v in enumerate(t)])
        )
        painter.setPen(pal.text().color())
        f = painter.font()
        f.setPointSizeF(max(6.5, f.pointSizeF() * 0.8))
        painter.setFont(f)
        for i, m in enumerate(MONTHS):
            painter.drawText(
                QRectF(left + i * bar, top + ch + 2, bar, bottom - 2),
                Qt.AlignmentFlag.AlignHCenter,
                m[0].upper(),
            )
        painter.setPen(QColor(220, 60, 50))
        painter.drawText(
            QRectF(0, top - 4, left - 4, 14), Qt.AlignmentFlag.AlignRight, f"{tmax:.0f}°"
        )
        painter.drawText(
            QRectF(0, top + ch - 10, left - 4, 14), Qt.AlignmentFlag.AlignRight, f"{tmin:.0f}°"
        )
        painter.setPen(QColor(70, 130, 220))
        painter.drawText(
            QRectF(w - right + 4, top - 4, right, 14), Qt.AlignmentFlag.AlignLeft, f"{pmax:.0f}"
        )
        painter.drawText(
            QRectF(w - right + 4, top + ch - 10, right, 14), Qt.AlignmentFlag.AlignLeft, "мм"
        )


class HexInfoPanel(QWidget):
    """Grouped report of one hex; ``zoom_requested(col, row)`` asks the map to zoom there."""

    zoom_requested = Signal(int, int)
    close_requested = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.report: HexReport | None = None
        self._collapsed = set(_COLLAPSED)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(6, 6, 6, 6)
        head = QHBoxLayout()
        self.title = QLabel("Щёлкните по гексу на карте")
        f = self.title.font()
        f.setPointSizeF(f.pointSizeF() * 1.25)
        f.setBold(True)
        self.title.setFont(f)
        self.title.setWordWrap(True)
        head.addWidget(self.title, 1)
        self.close_btn = QPushButton("✕")
        self.close_btn.setFixedWidth(30)
        self.close_btn.setToolTip("Закрыть панель (Esc)")
        self.close_btn.clicked.connect(self.close_requested.emit)
        head.addWidget(self.close_btn)
        lay.addLayout(head)
        self.subtitle = QLabel("")
        self.subtitle.setWordWrap(True)
        lay.addWidget(self.subtitle)
        buttons = QHBoxLayout()
        self.zoom_btn = QPushButton("🔍 Приблизить")
        self.zoom_btn.setToolTip("Приблизить карту к этому гексу")
        self.zoom_btn.clicked.connect(self._zoom)
        self.copy_btn = QPushButton("📋 Копировать")
        self.copy_btn.setToolTip("Скопировать всю информацию о гексе как текст")
        self.copy_btn.clicked.connect(self.copy_text)
        buttons.addWidget(self.zoom_btn)
        buttons.addWidget(self.copy_btn)
        lay.addLayout(buttons)
        self.chart = ClimateChart()
        self.chart.setToolTip("Климатограмма: столбики — осадки по месяцам, линия — температура")
        self.chart.setVisible(False)
        lay.addWidget(self.chart)
        self.search = QLineEdit()
        self.search.setPlaceholderText("Поиск по параметрам… (например: река, почва, лёд)")
        self.search.setClearButtonEnabled(True)
        self.search.textChanged.connect(self._filter)
        lay.addWidget(self.search)
        self.tree = QTreeWidget()
        self.tree.setColumnCount(2)
        self.tree.setHeaderLabels(["Параметр", "Значение"])
        self.tree.setAlternatingRowColors(True)
        self.tree.setUniformRowHeights(False)
        self.tree.setWordWrap(True)
        self.tree.setTextElideMode(Qt.TextElideMode.ElideNone)
        self.tree.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        header = self.tree.header()
        header.setStretchLastSection(True)
        header.setSectionResizeMode(0, QHeaderView.ResizeMode.Interactive)
        self.tree.itemExpanded.connect(lambda it: self._remember(it, True))
        self.tree.itemCollapsed.connect(lambda it: self._remember(it, False))
        lay.addWidget(self.tree, 1)
        self.zoom_btn.setEnabled(False)
        self.copy_btn.setEnabled(False)

    # -- content ---------------------------------------------------------------------------
    def show_hex(self, project, col: int, row: int, raw: bool = True):
        rep = hex_report(project, col, row)
        if raw:
            rep.sections.append(_raw_section(project, col, row))
        self.show_report(rep)

    def show_report(self, rep: HexReport):
        self.report = rep
        self.title.setText(rep.title)
        self.subtitle.setText(rep.subtitle)
        self.chart.set_data(rep.climate)
        scroll = self.tree.verticalScrollBar().value()
        self.tree.clear()
        bold = QFont(self.tree.font())
        bold.setBold(True)
        for s in rep.sections:
            top = QTreeWidgetItem([s.title, ""])
            top.setFont(0, bold)
            top.setData(0, Qt.ItemDataRole.UserRole, s.key)
            top.setFirstColumnSpanned(True)
            for label, value in s.rows:
                child = QTreeWidgetItem([label.strip(), value])
                child.setToolTip(1, value)
                if label.startswith("  "):
                    child.setForeground(0, QColor(110, 110, 110))
                top.addChild(child)
            self.tree.addTopLevelItem(top)
            top.setFirstColumnSpanned(True)
            top.setExpanded(s.key not in self._collapsed)
        self.tree.setColumnWidth(0, max(150, min(230, int(0.48 * self.tree.viewport().width()))))
        self.tree.verticalScrollBar().setValue(scroll)
        self.zoom_btn.setEnabled(True)
        self.copy_btn.setEnabled(True)
        self._filter(self.search.text())

    def clear(self):
        self.report = None
        self.tree.clear()
        self.title.setText("Щёлкните по гексу на карте")
        self.subtitle.setText("")
        self.chart.set_data(None)
        self.zoom_btn.setEnabled(False)
        self.copy_btn.setEnabled(False)

    def text(self) -> str:
        return self.report.as_text() if self.report else ""

    def copy_text(self):
        QApplication.clipboard().setText(self.text())

    def _zoom(self):
        if self.report is not None:
            self.zoom_requested.emit(self.report.col, self.report.row)

    def _remember(self, item: QTreeWidgetItem, expanded: bool):
        key = item.data(0, Qt.ItemDataRole.UserRole)
        if key is None or self.search.text():
            return
        if expanded:
            self._collapsed.discard(key)
        else:
            self._collapsed.add(key)

    def _filter(self, text: str):
        needle = text.strip().lower()
        # Russian endings: «река» should find «реки», «почва» — «почвы»
        stem = needle[:-1] if len(needle) >= 4 else needle
        for i in range(self.tree.topLevelItemCount()):
            top = self.tree.topLevelItem(i)
            in_title = stem in top.text(0).lower()
            shown = 0
            for j in range(top.childCount()):
                child = top.child(j)
                hit = (
                    not needle
                    or in_title
                    or stem in child.text(0).lower()
                    or stem in child.text(1).lower()
                )
                child.setHidden(not hit)
                shown += hit
            top.setHidden(bool(needle) and shown == 0)
            if needle and shown:
                top.setExpanded(True)
            elif not needle:
                top.setExpanded(top.data(0, Qt.ItemDataRole.UserRole) not in self._collapsed)

    def keyPressEvent(self, event):
        if event.key() == Qt.Key.Key_Escape:
            self.close_requested.emit()
            return
        super().keyPressEvent(event)


def _raw_section(project, col: int, row: int):
    """Every stored value of the hex, by its layer name (for specialists)."""
    from ..hex_report import Section

    info = project.hex_info(col, row)
    s = Section("raw", "Все значения слоёв (для специалистов)")
    for key in sorted(info):
        v = info[key]
        if isinstance(v, (dict, list)):
            continue
        if isinstance(v, float):
            v = f"{v:.6g}"
        s.add(key, v)
    return s
