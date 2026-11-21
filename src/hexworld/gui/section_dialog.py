"""Cross-section window: one metre scale for land, lakes and the sea floor (TZ 4.6, 7)."""

from __future__ import annotations

import numpy as np
from PySide6.QtCore import QPointF, QRectF, Qt
from PySide6.QtGui import QBrush, QColor, QPainter, QPainterPath, QPen, QPolygonF
from PySide6.QtWidgets import (
    QDialog,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from ..geomorph.sections import write_csv

_MATERIAL = {
    "ocean": QColor(60, 110, 190),
    "lake": QColor(96, 176, 214),
}


class ProfileWidget(QWidget):
    def __init__(self, rows: list[dict], parent=None):
        super().__init__(parent)
        self.rows = rows
        self.setMinimumSize(720, 320)

    def paintEvent(self, _event):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        p.fillRect(self.rect(), QColor(250, 250, 248))
        rows = self.rows
        if len(rows) < 2:
            return
        d = np.array([r["distance_km"] for r in rows])
        h = np.array([r["height_m"] for r in rows])
        b = np.array([r["bedrock_m"] for r in rows])
        w = np.array(
            [np.nan if r["water_surface_m"] is None else r["water_surface_m"] for r in rows]
        )
        top = float(
            np.nanmax(np.concatenate([h, w[np.isfinite(w)] if np.isfinite(w).any() else h]))
        )
        bottom = float(np.nanmin(b))
        pad = 0.08 * max(top - bottom, 100.0)
        top += pad
        bottom -= pad
        left, right, t_px, b_px = 64.0, self.width() - 16.0, 16.0, self.height() - 34.0

        def X(v):
            return left + (v - d[0]) / max(d[-1] - d[0], 1e-9) * (right - left)

        def Y(v):
            return t_px + (top - v) / max(top - bottom, 1e-9) * (b_px - t_px)

        # water: from the water surface down to the solid surface
        for k in range(len(rows) - 1):
            if np.isfinite(w[k]) and w[k] > h[k]:
                color = _MATERIAL.get(rows[k]["material_type"], QColor(60, 110, 190))
                poly = QPolygonF(
                    [
                        QPointF(X(d[k]), Y(w[k])),
                        QPointF(X(d[k + 1]), Y(w[k])),
                        QPointF(X(d[k + 1]), Y(h[k + 1])),
                        QPointF(X(d[k]), Y(h[k])),
                    ]
                )
                p.setPen(QPen(color, 1.2))  # an outline of the fill colour: no seams
                p.setBrush(QBrush(color))
                p.drawPolygon(poly)
        # sediment between bedrock and surface, rock below the bedrock
        for k in range(len(rows) - 1):
            sed = QPolygonF(
                [
                    QPointF(X(d[k]), Y(h[k])),
                    QPointF(X(d[k + 1]), Y(h[k + 1])),
                    QPointF(X(d[k + 1]), Y(b[k + 1])),
                    QPointF(X(d[k]), Y(b[k])),
                ]
            )
            p.setPen(QPen(QColor(222, 196, 140), 1.2))
            p.setBrush(QBrush(QColor(222, 196, 140)))
            p.drawPolygon(sed)
            rock = QPolygonF(
                [
                    QPointF(X(d[k]), Y(b[k])),
                    QPointF(X(d[k + 1]), Y(b[k + 1])),
                    QPointF(X(d[k + 1]), b_px),
                    QPointF(X(d[k]), b_px),
                ]
            )
            p.setPen(QPen(QColor(150, 130, 115), 1.2))
            p.setBrush(QBrush(QColor(150, 130, 115)))
            p.drawPolygon(rock)
        path = QPainterPath(QPointF(X(d[0]), Y(h[0])))
        for k in range(1, len(rows)):
            path.lineTo(QPointF(X(d[k]), Y(h[k])))
        p.setPen(QPen(QColor(40, 30, 20), 1.6))
        p.setBrush(Qt.BrushStyle.NoBrush)
        p.drawPath(path)
        # axes
        p.setPen(QPen(QColor(90, 90, 90), 1.0))
        p.drawRect(QRectF(left, t_px, right - left, b_px - t_px))
        step = _nice((top - bottom) / 6.0)
        v = np.ceil(bottom / step) * step
        while v <= top:
            p.drawLine(QPointF(left - 4, Y(v)), QPointF(left, Y(v)))
            p.drawText(QRectF(0, Y(v) - 8, left - 6, 16), Qt.AlignmentFlag.AlignRight, f"{v:,.0f}")
            v += step
        step = _nice((d[-1] - d[0]) / 8.0)
        v = 0.0
        while v <= d[-1]:
            p.drawLine(QPointF(X(v), b_px), QPointF(X(v), b_px + 4))
            p.drawText(
                QRectF(X(v) - 40, b_px + 6, 80, 16), Qt.AlignmentFlag.AlignCenter, f"{v:,.0f}"
            )
            v += step
        p.drawText(
            QRectF(left, self.height() - 16, right - left, 16),
            Qt.AlignmentFlag.AlignCenter,
            "расстояние, км; высота, м (одна шкала для суши и дна)",
        )


def _nice(x: float) -> float:
    if x <= 0:
        return 1.0
    e = 10 ** np.floor(np.log10(x))
    for m in (1, 2, 5, 10):
        if m * e >= x:
            return float(m * e)
    return float(10 * e)


class SectionDialog(QDialog):
    def __init__(self, rows: list[dict], parent=None):
        super().__init__(parent)
        self.setWindowTitle("Разрез рельефа")
        self.rows = rows
        layout = QVBoxLayout(self)
        layout.addWidget(ProfileWidget(rows, self))
        h = np.array([r["height_m"] for r in rows])
        kinds = sorted({r["province"] for r in rows if r["province"] != "land"})
        info = QLabel(
            f"длина {rows[-1]['distance_km']:,.0f} км; от {h.min():,.0f} до {h.max():,.0f} м; "
            "жёлтым — рыхлые отложения, серым — фундамент; "
            + ("пересекает: " + ", ".join(kinds[:8]) if kinds else "")
        )
        info.setWordWrap(True)
        layout.addWidget(info)
        buttons = QHBoxLayout()
        save = QPushButton("Сохранить CSV…")
        save.clicked.connect(self._save)
        close = QPushButton("Закрыть")
        close.clicked.connect(self.accept)
        buttons.addStretch(1)
        buttons.addWidget(save)
        buttons.addWidget(close)
        layout.addLayout(buttons)

    def _save(self):
        path, _ = QFileDialog.getSaveFileName(self, "Разрез в CSV", "section.csv", "CSV (*.csv)")
        if path:
            write_csv(self.rows, path)
