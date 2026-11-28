"""Snapshot of a map tab: resolution, geographic names and an optional legend strip.

The map itself is the same render as the PNG export; names are painted with the styles of
the map window (labels never overlap); the legend strip lists the colours of the tab (for the
biome tab: every biome colour with its name) under the map.
"""

from __future__ import annotations

import numpy as np
from PySide6.QtCore import QPointF, QRectF, Qt
from PySide6.QtGui import QColor, QFont, QFontMetricsF, QImage, QPainter, QPainterPath, QPen
from PySide6.QtWidgets import (
    QCheckBox,
    QDialog,
    QDialogButtonBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QRadioButton,
    QVBoxLayout,
)

#: resolution presets: key -> (title, longest side in px)
QUALITY = {"high": ("Высокое", 6000), "medium": ("Среднее", 2400), "low": ("Низкое", 1200)}
DEFAULTS = {"quality": "medium", "names": True, "legend": False}


def settings() -> dict:
    from .. import saved_worlds as sw

    return {**DEFAULTS, **sw.load_prefs().get("snapshot", {})}


def save_settings(s: dict) -> None:
    from .. import saved_worlds as sw

    prefs = sw.load_prefs()
    prefs["snapshot"] = {k: s[k] for k in DEFAULTS}
    sw.save_prefs(prefs)


def _to_qimage(rgb: np.ndarray) -> QImage:
    rgb = np.ascontiguousarray(rgb, dtype=np.uint8)
    h, w, _ = rgb.shape
    return QImage(rgb.data, w, h, 3 * w, QImage.Format.Format_RGB888).copy()


def paint_names(painter: QPainter, project, km_per_px: float, scale: float = 1.0) -> list:
    """Geographic names on a full-map image (origin at the viewport corner)."""
    from .. import names
    from .map_view import _LABEL_ORDER, _LABEL_STYLE

    try:
        labels = names.place_labels(project)
    except Exception:  # noqa: BLE001 - names are decoration
        return []
    b = project.geometry.viewport_bounds
    w, h = painter.device().width(), painter.device().height()
    placed: list[QRectF] = []
    drawn = []
    order = {k: i for i, k in enumerate(_LABEL_ORDER)}
    for lab in sorted(labels, key=lambda x: (order.get(x.kind, 9), -x.size_km)):
        pt, bold, italic, color, min_px, caps = _LABEL_STYLE[lab.kind]
        if lab.size_km / km_per_px < min_px * 0.6:
            continue
        sx, sy = (lab.x_km - b.xmin) / km_per_px, (lab.y_km - b.ymin) / km_per_px
        text = lab.title_ru.upper() if caps else lab.title_ru
        font = QFont()
        font.setPointSizeF(pt * scale)
        font.setBold(bold)
        font.setItalic(italic)
        fm = QFontMetricsF(font)
        tw, th = fm.horizontalAdvance(text), fm.height()
        rect = QRectF(sx - tw / 2, sy - th / 2, tw, th)
        rect.translate(max(0.0, 4 - rect.left()) - max(0.0, rect.right() - (w - 4)),
                       max(0.0, 2 - rect.top()) - max(0.0, rect.bottom() - (h - 2)))  # fmt: skip
        if any(rect.adjusted(-4, -2, 4, 2).intersects(r) for r in placed):
            continue
        placed.append(rect)
        path = QPainterPath()
        path.addText(QPointF(rect.left(), rect.top() + fm.ascent()), font, text)
        painter.setPen(QPen(QColor(255, 255, 255, 215), 3.2 * scale))
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.drawPath(path)
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QColor(*color))
        painter.drawPath(path)
        drawn.append(text)
    return drawn


def full_legend(project, view_id: str, mode: str) -> list:
    """The legend of the tab for a picture: the biome tab lists every biome of the world
    (largest area first), not only the 28 that fit the side panel."""
    from .. import views

    spec = views.VIEWS[view_id]
    if view_id == "biomes" and mode in ("dominant", "dominant_land"):
        from ..biomes.registry import BIOMES

        key = "dominant_biome" if mode == "dominant" else "dominant_land_biome"
        codes, counts = np.unique(project.biomes.layers[key], return_counts=True)
        items = [views.LegendItem(BIOMES[int(codes[i])].ru, BIOMES[int(codes[i])].rgb)
                 for i in np.argsort(-counts)]  # fmt: skip
        return items + [it for it in spec.legend(project, mode) if it.color is None]
    return list(spec.legend(project, mode)) if spec.legend else []


def legend_height(items, width: int, scale: float) -> int:
    return _legend(None, items, width, 0, scale, "")


def _legend(painter, items, width, top, scale, title) -> int:
    """Lay out (and paint when ``painter`` is given) the legend; returns its height."""
    pad = 12 * scale
    font = QFont()
    font.setPointSizeF(9.0 * scale)
    fm = QFontMetricsF(font)
    tfont = QFont(font)
    tfont.setBold(True)
    tfont.setPointSizeF(10.5 * scale)
    tfm = QFontMetricsF(tfont)
    y = top + pad
    if painter is not None and title:
        painter.setFont(tfont)
        painter.setPen(QColor(25, 25, 25))
        painter.drawText(QRectF(pad, y, width - 2 * pad, tfm.height()), title)
    y += tfm.height() + 4 * scale if title else 0
    swatches = [it for it in items if it.color is not None]
    notes = [it for it in items if it.color is None]
    sw = 14 * scale
    if swatches:
        col_w = min(max(fm.horizontalAdvance(it.label) for it in swatches) + sw + 14 * scale,
                    width - 2 * pad)  # fmt: skip
        ncol = max(1, int((width - 2 * pad) // col_w))
        row_h = max(fm.height(), sw) + 3 * scale
        for k, it in enumerate(swatches):
            cx = pad + (k % ncol) * col_w
            cy = y + (k // ncol) * row_h
            if painter is not None:
                painter.setPen(QPen(QColor(60, 60, 60), 0.8))
                painter.setBrush(QColor(*it.color))
                painter.drawRect(QRectF(cx, cy + (row_h - sw) / 2, sw, sw))
                painter.setPen(QColor(25, 25, 25))
                painter.setFont(font)
                painter.drawText(QRectF(cx + sw + 6 * scale, cy, col_w - sw - 8 * scale, row_h),
                                 Qt.AlignmentFlag.AlignVCenter, it.label)  # fmt: skip
        y += ((len(swatches) + ncol - 1) // ncol) * row_h + 4 * scale
    for it in notes:
        r = fm.boundingRect(QRectF(pad, y, width - 2 * pad, 1e6),
                            int(Qt.TextFlag.TextWordWrap), it.label)  # fmt: skip
        if painter is not None:
            painter.setFont(font)
            painter.setPen(QColor(70, 70, 70))
            painter.drawText(QRectF(pad, y, width - 2 * pad, r.height()),
                             int(Qt.TextFlag.TextWordWrap), it.label)  # fmt: skip
        y += r.height() + 2 * scale
    return int(round(y + pad - top))


def compose(project, view_id: str, mode: str, overlays, quality: str = "medium",
            names: bool = True, legend: bool = False) -> QImage:  # fmt: skip
    """The snapshot image: map (+ names) (+ legend strip under it)."""
    from .. import views

    spec = views.VIEWS[view_id]
    mode = mode or spec.default_mode()
    max_size = QUALITY[quality][1]
    rgb = views.render_view_image(project, view_id, mode, overlays, max_size=max_size)
    h, w = rgb.shape[:2]
    b = project.geometry.viewport_bounds
    km_per_px = b.width / w
    scale = max(0.8, max(w, h) / 1800.0)  # fonts grow with the resolution
    items = full_legend(project, view_id, mode) if legend else []
    mode_title = next((o.title for o in spec.modes if o.id == mode), "")
    title = spec.title + ("" if not mode_title or mode_title == spec.title else f" — {mode_title}")
    lh = _legend(None, items, w, 0, scale, title) if legend else 0
    img = QImage(w, h + lh, QImage.Format.Format_RGB888)
    img.fill(QColor(250, 250, 248))
    p = QPainter(img)
    p.setRenderHint(QPainter.RenderHint.Antialiasing, True)
    p.drawImage(0, 0, _to_qimage(rgb))
    if names:
        p.setClipRect(0, 0, w, h)
        paint_names(p, project, km_per_px, scale)
        p.setClipping(False)
    if legend:
        p.setPen(QPen(QColor(180, 180, 180), 1.0))
        p.drawLine(0, h, w, h)
        _legend(p, items, w, h, scale, title)
    p.end()
    return img


class SnapshotDialog(QDialog):
    """Resolution, names, legend and the folder; the choice is remembered (F12 reuses it)."""

    def __init__(self, folder: str, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Снимок вкладки")
        s = settings()
        lay = QVBoxLayout(self)
        lay.addWidget(QLabel("Разрешение (длинная сторона карты):"))
        self.radios = {}
        for key, (title, px) in QUALITY.items():
            rb = QRadioButton(f"{title} — {px} пикс.")
            rb.setChecked(key == s["quality"])
            self.radios[key] = rb
            lay.addWidget(rb)
        self.names = QCheckBox("Географические названия (океаны, моря, горы, реки, …)")
        self.names.setChecked(bool(s["names"]))
        self.legend = QCheckBox("Легенда под картой (цвета вкладки, например все биомы)")
        self.legend.setChecked(bool(s["legend"]))
        lay.addWidget(self.names)
        lay.addWidget(self.legend)
        row = QHBoxLayout()
        self.folder = QLineEdit(folder)
        row.addWidget(QLabel("Папка:"))
        row.addWidget(self.folder, 1)
        change = QPushButton("…")
        change.clicked.connect(self._pick)
        row.addWidget(change)
        lay.addLayout(row)
        bb = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Save | QDialogButtonBox.StandardButton.Cancel
        )
        bb.accepted.connect(self._ok)
        bb.rejected.connect(self.reject)
        lay.addWidget(bb)

    def _pick(self):
        from PySide6.QtWidgets import QFileDialog

        f = QFileDialog.getExistingDirectory(self, "Папка для снимков", self.folder.text())
        if f:
            self.folder.setText(f)

    def choice(self) -> dict:
        q = next(k for k, rb in self.radios.items() if rb.isChecked())
        return {"quality": q, "names": self.names.isChecked(), "legend": self.legend.isChecked()}

    def _ok(self):
        save_settings(self.choice())
        self.accept()
