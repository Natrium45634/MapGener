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

_ORE = {
    "vein": QColor(220, 40, 40), "vein_network": QColor(220, 40, 40),
    "tabular": QColor(60, 60, 60), "seam": QColor(20, 20, 20), "stratabound": QColor(140, 70, 30),
    "lens": QColor(230, 120, 30), "pod": QColor(230, 120, 30), "stockwork": QColor(200, 150, 40),
    "disseminated": QColor(200, 150, 40), "breccia": QColor(170, 90, 160),
    "replacement": QColor(170, 90, 160), "mantle_pipe": QColor(40, 150, 200),
    "placer": QColor(240, 200, 40), "paleoplacer": QColor(200, 170, 30),
    "laterite": QColor(190, 80, 50), "reservoir": QColor(120, 40, 160),
    "quarry_unit": QColor(160, 160, 160), "nodules": QColor(30, 30, 90),
    "crust": QColor(30, 30, 90), "cavity_fill": QColor(120, 200, 120),
}  # fmt: skip
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
        ore_z = [iv["z_bottom_m"] for r in rows for iv in r.get("ore", [])]
        if ore_z:  # the section reaches down to the deepest crossed body
            bottom = min(bottom, float(min(ore_z)))
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
        # v4.7: thermal belt strip under the axis and every crossing of the thermal tree
        # limit (several where the profile goes up and down - never one forced line)
        if any(r.get("belt") for r in rows):
            from ..topoclimate.thermal import BELT_RGB, BELTS

            for k in range(len(rows) - 1):
                belt = rows[k].get("belt")
                if not belt:
                    continue
                c = BELT_RGB[BELTS.index(belt)]
                p.setPen(Qt.PenStyle.NoPen)
                p.setBrush(QBrush(QColor(int(c[0]), int(c[1]), int(c[2]))))
                p.drawRect(QRectF(X(d[k]), b_px - 8, X(d[k + 1]) - X(d[k]) + 0.5, 8))
            m = np.array([np.nan if r.get("thermal_margin") is None else r["thermal_margin"]
                          for r in rows])  # fmt: skip
            p.setPen(QPen(QColor(30, 120, 40), 1.4, Qt.PenStyle.DashLine))
            for k in range(len(rows) - 1):
                if np.isfinite(m[k]) and np.isfinite(m[k + 1]) and m[k] * m[k + 1] < 0:
                    f = m[k] / (m[k] - m[k + 1])
                    xk = X(d[k] + f * (d[k + 1] - d[k]))
                    zk = h[k] + f * (h[k + 1] - h[k])
                    p.drawLine(QPointF(xk - 9, Y(zk)), QPointF(xk + 9, Y(zk)))
        # TZ 5.5.1: ore bodies cut by the section (real intervals, colour by morphology)
        for k in range(len(rows) - 1):
            for iv in rows[k].get("ore", []):
                c = _ORE.get(iv["morphology"], QColor(200, 40, 40))
                p.setPen(QPen(c.darker(130), 1.0))
                p.setBrush(QBrush(c))
                y0, y1 = Y(iv["z_top_m"]), Y(iv["z_bottom_m"])
                p.drawRect(QRectF(X(d[k]), y0, max(X(d[k + 1]) - X(d[k]), 1.0),
                                  max(y1 - y0, 1.5)))  # fmt: skip
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
        vx = (d[-1] - d[0]) * 1000.0 / max(right - left, 1.0)  # m per pixel along
        vz = (top - bottom) / max(b_px - t_px, 1.0)  # m per pixel down
        p.drawText(
            QRectF(left, self.height() - 16, right - left, 16),
            Qt.AlignmentFlag.AlignCenter,
            f"расстояние, км; высота, м (одна шкала для суши и дна); "
            f"вертикальное преувеличение ×{vx / max(vz, 1e-9):,.0f}",
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
            + (
                "; полоса внизу — термические пояса (v4.7), зелёные штрихи — пересечения "
                "термической границы деревьев (их может быть несколько)"
                if any(r.get("belt") for r in rows)
                else ""
            )
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
