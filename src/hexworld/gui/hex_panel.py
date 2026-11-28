"""Side panel «Информация о гексе»: everything about the clicked hex, grouped and searchable."""

from __future__ import annotations

from PySide6.QtCore import QPointF, QRectF, Qt, Signal
from PySide6.QtGui import QColor, QFont, QPainter, QPen, QPolygonF
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QDialog,
    QDialogButtonBox,
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

#: every section of the report: key -> title (the order of the chooser)
SECTION_TITLES = {
    "place": "Положение",
    "relief": "Рельеф",
    "climate": "Климат",
    "months": "По месяцам",
    "water": "Вода: реки, озёра, сток",
    "biomes": "Биомы и природный покров",
    "fertility": "Плодородие почвы",
    "resources": "Полезные ископаемые",
    "minerals": "Недра и месторождения (5.5.1)",
    "sea": "Берег и дно моря",
    "volcanism": "Вулканизм",
    "soils": "Почвы (физика, v4.5)",
    "belts": "Высотная поясность",
    "geology": "Тектоника и геология",
    "basin": "Тектонический бассейн",
    "atmosphere": "Атмосфера и сезоны",
    "earth": "Океан, лёд и ветры на высотах",
    "raw": "Все значения слоёв (для специалистов)",
}
#: shown by default; the rest is technical and opens from «Все данные…» or the chooser
DEFAULT_SHOWN = {
    "relief",
    "climate",
    "water",
    "biomes",
    "fertility",
    "resources",
    "minerals",
    "sea",
    "volcanism",
}


def shown_sections() -> set:
    from .. import saved_worlds as sw

    prefs = sw.load_prefs()
    return set(prefs["hex_sections"]) if "hex_sections" in prefs else set(DEFAULT_SHOWN)


def set_shown_sections(keys) -> None:
    from .. import saved_worlds as sw

    prefs = sw.load_prefs()
    prefs["hex_sections"] = sorted(keys)
    sw.save_prefs(prefs)


def panel_font_pt(default: float) -> float:
    from .. import saved_worlds as sw

    return float(sw.load_prefs().get("hex_font_pt", default))


def set_panel_font_pt(pt: float) -> None:
    from .. import saved_worlds as sw

    prefs = sw.load_prefs()
    prefs["hex_font_pt"] = round(float(pt), 1)
    sw.save_prefs(prefs)


class SectionsDialog(QDialog):
    """Which report sections the side panel shows (the rest stays in «Все данные…»)."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Разделы информации о гексе")
        lay = QVBoxLayout(self)
        hint = QLabel(
            "Отмеченные разделы видны в панели сразу. Остальные — техническая "
            "информация: она всегда доступна по кнопке «📄 Все данные…»."
        )
        hint.setWordWrap(True)
        lay.addWidget(hint)
        shown = shown_sections()
        self.boxes = {}
        for key, title in SECTION_TITLES.items():
            cb = QCheckBox(title)
            cb.setChecked(key in shown)
            self.boxes[key] = cb
            lay.addWidget(cb)
        row = QHBoxLayout()
        for title, keys in (
            ("Главное", DEFAULT_SHOWN),
            ("Всё", set(SECTION_TITLES)),
            ("Ничего", set()),
        ):
            b = QPushButton(title)
            b.clicked.connect(
                lambda _c=False, k=keys: [cb.setChecked(n in k) for n, cb in self.boxes.items()]
            )
            row.addWidget(b)
        lay.addLayout(row)
        bb = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        bb.accepted.connect(self._ok)
        bb.rejected.connect(self.reject)
        lay.addWidget(bb)

    def _ok(self):
        set_shown_sections([k for k, cb in self.boxes.items() if cb.isChecked()])
        self.accept()


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


def _fill_tree(tree: QTreeWidget, sections, collapsed: set, font: QFont) -> None:
    bold = QFont(font)
    bold.setBold(True)
    tree.clear()
    for s in sections:
        top = QTreeWidgetItem([s.title, ""])
        top.setFont(0, bold)
        top.setData(0, Qt.ItemDataRole.UserRole, s.key)
        for label, value in s.rows:
            child = QTreeWidgetItem([label.strip(), value])
            child.setToolTip(1, value)
            child.setFont(0, font)
            child.setFont(1, font)
            if label.startswith("  "):
                child.setForeground(0, QColor(110, 110, 110))
            top.addChild(child)
        tree.addTopLevelItem(top)
        top.setFirstColumnSpanned(True)
        top.setExpanded(s.key not in collapsed)


def _make_tree() -> QTreeWidget:
    tree = QTreeWidget()
    tree.setColumnCount(2)
    tree.setHeaderLabels(["Параметр", "Значение"])
    tree.setAlternatingRowColors(True)
    tree.setUniformRowHeights(False)
    tree.setWordWrap(True)
    tree.setTextElideMode(Qt.TextElideMode.ElideNone)
    tree.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
    header = tree.header()
    header.setStretchLastSection(True)
    header.setSectionResizeMode(0, QHeaderView.ResizeMode.Interactive)
    return tree


def _filter_tree(tree: QTreeWidget, text: str, collapsed: set) -> None:
    needle = text.strip().lower()
    # Russian endings: «река» should find «реки», «почва» — «почвы»
    stem = needle[:-1] if len(needle) >= 4 else needle
    for i in range(tree.topLevelItemCount()):
        top = tree.topLevelItem(i)
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
            top.setExpanded(top.data(0, Qt.ItemDataRole.UserRole) not in collapsed)


class FullReportDialog(QDialog):
    """Every section of the hex report, technical ones included, with search and zoom."""

    def __init__(self, report: HexReport, font_pt: float, parent=None):
        super().__init__(parent)
        self.setWindowTitle(f"Все данные — {report.title}")
        self.resize(760, 820)
        self.report = report
        self.pt = font_pt
        lay = QVBoxLayout(self)
        head = QHBoxLayout()
        self.search = QLineEdit()
        self.search.setPlaceholderText("Поиск по параметрам…")
        self.search.setClearButtonEnabled(True)
        self.search.textChanged.connect(lambda t: _filter_tree(self.tree, t, set()))
        head.addWidget(self.search, 1)
        for title, d in (("A−", -1.0), ("A+", 1.0)):
            b = QPushButton(title)
            b.setFixedWidth(40)
            b.clicked.connect(lambda _c=False, dd=d: self._zoom(dd))
            head.addWidget(b)
        lay.addLayout(head)
        self.tree = _make_tree()
        lay.addWidget(self.tree, 1)
        self._fill()

    def _zoom(self, d: float):
        self.pt = max(7.0, min(28.0, self.pt + d))
        self._fill()

    def _fill(self):
        f = QFont(self.font())
        f.setPointSizeF(self.pt)
        self.tree.setFont(f)
        _fill_tree(self.tree, self.report.sections, set(), f)
        self.tree.setColumnWidth(0, int(self.width() * 0.42))
        _filter_tree(self.tree, self.search.text(), set())


class HexInfoPanel(QWidget):
    """Report of one hex: a card with what matters (biome, its description, conditions,
    key numbers), then the chosen sections; ``zoom_requested(col, row)`` asks the map to
    zoom there."""

    zoom_requested = Signal(int, int)
    close_requested = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.report: HexReport | None = None
        self.full_report: HexReport | None = None
        self._collapsed = set(_COLLAPSED)
        base = QApplication.font().pointSizeF()
        self.pt = panel_font_pt(max(11.0, base + 1.5))
        lay = QVBoxLayout(self)
        lay.setContentsMargins(6, 6, 6, 6)
        head = QHBoxLayout()
        self.title = QLabel("Щёлкните по гексу на карте")
        self.title.setWordWrap(True)
        head.addWidget(self.title, 1)
        self.smaller_btn = QPushButton("A−")
        self.smaller_btn.setToolTip("Шрифт мельче")
        self.smaller_btn.clicked.connect(lambda: self.zoom_font(-1.0))
        self.bigger_btn = QPushButton("A+")
        self.bigger_btn.setToolTip("Шрифт крупнее")
        self.bigger_btn.clicked.connect(lambda: self.zoom_font(1.0))
        for b in (self.smaller_btn, self.bigger_btn):
            b.setFixedWidth(40)
            head.addWidget(b)
        self.close_btn = QPushButton("✕")
        self.close_btn.setFixedWidth(30)
        self.close_btn.setToolTip("Закрыть панель (Esc)")
        self.close_btn.clicked.connect(self.close_requested.emit)
        head.addWidget(self.close_btn)
        lay.addLayout(head)
        self.subtitle = QLabel("")
        self.subtitle.setWordWrap(True)
        lay.addWidget(self.subtitle)
        # the card: biome, description, conditions and the key numbers
        self.card = QLabel("")
        self.card.setWordWrap(True)
        self.card.setTextFormat(Qt.TextFormat.RichText)
        self.card.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        self.card.setStyleSheet(
            "QLabel { background: palette(base); border: 1px solid "
            "palette(mid); border-radius: 6px; padding: 8px; }"
        )
        self.card.setVisible(False)
        lay.addWidget(self.card)
        buttons = QHBoxLayout()
        self.zoom_btn = QPushButton("🔍 Приблизить")
        self.zoom_btn.setToolTip("Приблизить карту к этому гексу")
        self.zoom_btn.clicked.connect(self._zoom)
        self.copy_btn = QPushButton("📋 Копировать")
        self.copy_btn.setToolTip("Скопировать всю информацию о гексе как текст")
        self.copy_btn.clicked.connect(self.copy_text)
        self.report_btn = QPushButton("📍 Для отчёта")
        self.report_btn.setToolTip(
            "Скопировать коротко: код мира, координаты гекса, вкладку и главное о нём — "
            "чтобы указать на место на карте (Ctrl+Shift+R)"
        )
        self.report_btn.clicked.connect(self.copy_report)
        self.sections_btn = QPushButton("☑ Разделы…")
        self.sections_btn.setToolTip("Какие разделы показывать в панели")
        self.sections_btn.clicked.connect(self.choose_sections)
        self.all_btn = QPushButton("📄 Все данные…")
        self.all_btn.setToolTip("Вся информация о гексе, включая техническую")
        self.all_btn.clicked.connect(self.show_all)
        for b in (self.zoom_btn, self.copy_btn, self.report_btn):
            buttons.addWidget(b)
        lay.addLayout(buttons)
        buttons2 = QHBoxLayout()
        for b in (self.sections_btn, self.all_btn):
            buttons2.addWidget(b)
        lay.addLayout(buttons2)
        self.chart = ClimateChart()
        self.chart.setToolTip("Климатограмма: столбики — осадки по месяцам, линия — температура")
        self.chart.setVisible(False)
        lay.addWidget(self.chart)
        self.search = QLineEdit()
        self.search.setPlaceholderText("Поиск по параметрам… (например: река, почва, лёд)")
        self.search.setClearButtonEnabled(True)
        self.search.textChanged.connect(self._filter)
        lay.addWidget(self.search)
        self.tree = _make_tree()
        self.tree.itemExpanded.connect(lambda it: self._remember(it, True))
        self.tree.itemCollapsed.connect(lambda it: self._remember(it, False))
        lay.addWidget(self.tree, 1)
        #: who asks: ``context() -> dict`` (world code, view title) - set by the main window
        self.context = None
        self.project = None
        self.zoom_btn.setEnabled(False)
        self.copy_btn.setEnabled(False)
        self.report_btn.setEnabled(False)
        self.all_btn.setEnabled(False)
        self._apply_font()

    # -- fonts -----------------------------------------------------------------------------
    def zoom_font(self, d: float):
        self.pt = max(7.0, min(28.0, self.pt + d))
        set_panel_font_pt(self.pt)
        self._apply_font()
        if self.full_report is not None:
            self._render()

    def _apply_font(self):
        f = QFont(self.font())
        f.setPointSizeF(self.pt)
        for w in (self.subtitle, self.card, self.search, self.tree):
            w.setFont(f)
        t = QFont(f)
        t.setPointSizeF(self.pt * 1.3)
        t.setBold(True)
        self.title.setFont(t)

    # -- content ---------------------------------------------------------------------------
    def show_hex(self, project, col: int, row: int, raw: bool = True):
        self.project = project
        rep = hex_report(project, col, row)
        if raw:
            rep.sections.append(_raw_section(project, col, row))
        self.show_report(rep)

    def show_report(self, rep: HexReport):
        self.full_report = rep
        self._render()

    def _render(self):
        rep = self.full_report
        shown = shown_sections()
        keep = [s for s in rep.sections if s.key in shown or s.key not in SECTION_TITLES]
        self.report = HexReport(
            rep.col, rep.row, rep.title, rep.subtitle, keep, rep.climate, rep.description
        )
        self.title.setText(rep.title)
        self.subtitle.setText(rep.subtitle)
        self.card.setText(_card_html(rep.description, self.pt))
        self.card.setVisible(bool(rep.description))
        self.chart.set_data(rep.climate)
        scroll = self.tree.verticalScrollBar().value()
        f = QFont(self.font())
        f.setPointSizeF(self.pt)
        _fill_tree(self.tree, keep, self._collapsed, f)
        self.tree.setColumnWidth(0, max(150, min(260, int(0.46 * self.tree.viewport().width()))))
        self.tree.verticalScrollBar().setValue(scroll)
        hidden = len(rep.sections) - len(keep)
        self.all_btn.setText(f"📄 Все данные… (+{hidden})" if hidden else "📄 Все данные…")
        for b in (self.zoom_btn, self.copy_btn, self.report_btn, self.all_btn):
            b.setEnabled(True)
        self._filter(self.search.text())

    def choose_sections(self):
        if SectionsDialog(self).exec() and self.full_report is not None:
            self._render()

    def show_all(self):
        if self.full_report is not None:
            dlg = FullReportDialog(self.full_report, self.pt, self)
            dlg.show()
            self._full_dialog = dlg

    def clear(self):
        self.report = None
        self.full_report = None
        self.tree.clear()
        self.title.setText("Щёлкните по гексу на карте")
        self.subtitle.setText("")
        self.card.setVisible(False)
        self.chart.set_data(None)
        for b in (self.zoom_btn, self.copy_btn, self.report_btn, self.all_btn):
            b.setEnabled(False)

    def text(self) -> str:
        return self.full_report.as_text() if self.full_report else ""

    def copy_text(self):
        QApplication.clipboard().setText(self.text())

    def report_text(self) -> str:
        """A few lines that point at this hex: world code, place, view, the key facts."""
        rep = self.full_report
        if rep is None:
            return ""
        ctx = self.context() if callable(self.context) else {}
        from ..version import PRODUCT_VERSION

        lines = [f"HexWorld {PRODUCT_VERSION}" + (f" · {ctx['code']}" if ctx.get("code") else "")]
        place = {k: v for s in rep.sections if s.key == "place" for k, v in s.rows}
        where = [f"гекс ({rep.col}, {rep.row})"]
        for key in ("Координаты", "Широта", "Тип поверхности"):
            if place.get(key):
                where.append(place[key])
        lines.append(" · ".join(where))
        if ctx.get("view"):
            lines.append(f"Вкладка: {ctx['view']}")
        d = rep.description or {}
        facts = [f"{lab.lower()} {val}" for lab, val in d.get("facts", [])]
        if facts:
            lines.append("; ".join(facts))
        relief = {k: v for s in rep.sections if s.key == "relief" for k, v in s.rows}
        extra = [f"{k.lower()} {relief[k]}" for k in list(relief)[:4]]
        if extra:
            lines.append("Рельеф: " + "; ".join(extra))
        lines.append("Замечание: ")
        return "\n".join(lines)

    def copy_report(self):
        text = self.report_text()
        if text:
            QApplication.clipboard().setText(text)

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
        _filter_tree(self.tree, text, self._collapsed)

    def keyPressEvent(self, event):
        if event.key() == Qt.Key.Key_Escape:
            self.close_requested.emit()
            return
        super().keyPressEvent(event)


def _card_html(d: dict | None, pt: float) -> str:
    """The card of the panel: biome (large), its description, the conditions, key facts."""
    import html

    if not d:
        return ""
    e = html.escape
    out = []
    if d.get("biome"):
        name = d["biome"][:1].upper() + d["biome"][1:]
        out.append(
            f"<div style='font-size:{pt * 1.45:.0f}pt; font-weight:bold; "
            f"color:#1d5f2e'>{e(name)}</div>"
        )
    if d.get("biome_text"):
        out.append(f"<div style='margin-top:4px'>{e(d['biome_text'])}</div>")
    if d.get("conditions"):
        out.append(
            "<div style='margin-top:6px'>" + "<br>".join(e(c) for c in d["conditions"]) + "</div>"
        )
    if d.get("facts"):
        rows = "".join(
            f"<tr><td style='color:#555; padding-right:10px'>{e(a)}</td><td><b>{e(b)}</b></td></tr>"
            for a, b in d["facts"]
        )
        out.append(f"<table style='margin-top:6px'>{rows}</table>")
    return "".join(out)


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
