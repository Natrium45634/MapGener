"""Map widget: draws the whole map as one raster image (no per-hex objects).

The widget knows nothing about individual views: it asks the view registry for
the colour of every hex and paints it. Zoom, pan and the selected hex belong to
the widget, so they are shared by all tabs.
"""

from __future__ import annotations

import numpy as np
from PySide6.QtCore import QLineF, QPointF, Qt, QTimer, Signal
from PySide6.QtGui import QBrush, QColor, QImage, QPainter, QPen, QPolygonF
from PySide6.QtWidgets import QWidget

from .. import render, views
from ..project import Project
from ..stages.boundaries import TYPE_NAMES

_MAX_OVERLAY_EDGES = 60000


class MapView(QWidget):
    hexHovered = Signal(str)
    hexClicked = Signal(int, int)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMouseTracking(True)
        self.setMinimumSize(320, 240)
        self.project: Project | None = None
        self.view_id = "plates"
        self.mode = views.VIEWS["plates"].default_mode()
        self.overlays: frozenset = views.VIEWS["plates"].default_overlays()
        #: Selected hex (col, row); shared by every tab, like zoom and pan.
        self.selected_hex: tuple[int, int] | None = None
        self._km_per_px = 1.0
        self._cx = 0.0
        self._cy = 0.0
        self._hex_rgb: np.ndarray | None = None
        self._vectors: list | None = None
        self._image: QImage | None = None
        self._image_buf = None
        self._coarse = False
        self._drag = None
        self._moved = False
        self._settle = QTimer(self)
        self._settle.setSingleShot(True)
        self._settle.setInterval(140)
        self._settle.timeout.connect(self._sharpen)

    # -- state -------------------------------------------------------------
    def set_project(self, project: Project | None, keep_view: bool = False):
        same_grid = (
            project is not None
            and self.project is not None
            and project.layout == self.project.layout
        )
        self.project = project
        if not same_grid:
            self.selected_hex = None
        self._hex_rgb = None
        self._vectors = None
        if project is not None and not (keep_view or same_grid):
            self.fit()
        self._invalidate()

    def set_view(self, view_id: str, mode: str, overlays) -> None:
        """Switch the presentation only: no generation; zoom, pan and selection are kept."""
        self.view_id = view_id
        self.mode = mode
        self.overlays = frozenset(overlays)
        self._hex_rgb = None
        self._vectors = None
        self._invalidate()

    def select_hex(self, hexagon: tuple[int, int] | None):
        self.selected_hex = hexagon
        self.update()

    def fit(self):
        if self.project is None:
            return
        b = self.project.geometry.viewport_bounds
        self._cx = 0.5 * (b.xmin + b.xmax)
        self._cy = 0.5 * (b.ymin + b.ymax)
        self._km_per_px = 1.04 * max(
            b.width / max(1, self.width()), b.height / max(1, self.height())
        )
        self._invalidate()

    @property
    def km_per_px(self) -> float:
        return self._km_per_px

    def view_state(self) -> tuple[float, float, float]:
        return self._cx, self._cy, self._km_per_px

    def _invalidate(self, coarse: bool = False):
        self._image = None
        self._coarse = coarse
        if coarse:
            self._settle.start()
        self.update()

    def _sharpen(self):
        self._image = None
        self._coarse = False
        self.update()

    # -- coordinates -------------------------------------------------------
    def _origin(self):
        return (
            self._cx - 0.5 * self.width() * self._km_per_px,
            self._cy - 0.5 * self.height() * self._km_per_px,
        )

    def to_world(self, px: float, py: float):
        x0, y0 = self._origin()
        return x0 + px * self._km_per_px, y0 + py * self._km_per_px

    def to_screen(self, x, y):
        x0, y0 = self._origin()
        return (x - x0) / self._km_per_px, (y - y0) / self._km_per_px

    # -- painting ----------------------------------------------------------
    def _ensure_image(self):
        if self._image is not None or self.project is None:
            return
        if self._hex_rgb is None:
            self._hex_rgb = views.view_rgb(self.project, self.view_id, self.mode, self.overlays)
        scale = 3 if self._coarse else 1
        w = max(1, self.width() // scale)
        h = max(1, self.height() // scale)
        x0, y0 = self._origin()
        img = render.render_view(
            self.project.layout, self._hex_rgb, x0, y0, self._km_per_px * scale, w, h
        )
        self._image_buf = np.ascontiguousarray(img)
        self._image = QImage(self._image_buf.data, w, h, 3 * w, QImage.Format.Format_RGB888)

    def paintEvent(self, _event):
        painter = QPainter(self)
        painter.fillRect(self.rect(), QColor(*render.BACKGROUND))
        if self.project is None:
            painter.setPen(QColor(170, 170, 170))
            painter.drawText(
                self.rect(),
                Qt.AlignmentFlag.AlignCenter,
                "Задайте параметры и нажмите «Сгенерировать»",
            )
            return
        ok, reason = views.VIEWS[self.view_id].availability(self.project)
        if not ok:
            painter.setPen(QColor(210, 210, 210))
            painter.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, reason)
            return
        self._ensure_image()
        painter.drawImage(self.rect(), self._image)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        if self.view_id == "plates" and "boundaries" in self.overlays:
            self._paint_boundaries(painter)
        if self.view_id == "plates" and "motion" in self.overlays:
            self._paint_arrows(painter)
        self._paint_vectors(painter)
        if self.selected_hex is not None:
            wx, wy = self.project.layout.center(*self.selected_hex)
            sx, sy = self.to_screen(float(wx), float(wy))
            radius = max(6.0, 0.6 * self.project.layout.spacing_km / self._km_per_px)
            for pen in (QPen(QColor(0, 0, 0), 4.0), QPen(QColor(255, 255, 255), 2.0)):
                painter.setPen(pen)
                painter.drawEllipse(QPointF(sx, sy), radius, radius)

    def _paint_vectors(self, painter: QPainter):
        """Vector overlays of the current view (ridges, peaks, volcanoes, events ...)."""
        if self._vectors is None:
            self._vectors = views.view_vectors(self.project, self.view_id, self.mode, self.overlays)
        if not self._vectors:
            return
        x0, y0 = self._origin()
        k = self._km_per_px
        x1, y1 = x0 + self.width() * k, y0 + self.height() * k
        grow = float(np.clip(0.02 * self.project.layout.spacing_km / k + 0.8, 0.8, 2.2))
        for layer in self._vectors:
            color = QColor(*layer.color)
            if layer.kind == "lines":
                pen = QPen(color, layer.size * grow)
                pen.setCapStyle(Qt.PenCapStyle.RoundCap)
                painter.setPen(pen)
                painter.setBrush(Qt.BrushStyle.NoBrush)
                for xs, ys in zip(layer.x, layer.y, strict=True):
                    if xs.max() < x0 or xs.min() > x1 or ys.max() < y0 or ys.min() > y1:
                        continue
                    poly = QPolygonF(
                        [QPointF((a - x0) / k, (b - y0) / k) for a, b in zip(xs, ys, strict=True)]
                    )
                    painter.drawPolyline(poly)
            else:
                xs, ys = np.asarray(layer.x), np.asarray(layer.y)
                vis = (xs > x0) & (xs < x1) & (ys > y0) & (ys < y1)
                if int(vis.sum()) > 6000:
                    continue
                r = max(layer.size * grow, layer.size_km / k)
                painter.setPen(QPen(QColor(20, 20, 20) if layer.filled else color, 1.4))
                painter.setBrush(QBrush(color) if layer.filled else Qt.BrushStyle.NoBrush)
                for a, b in zip(xs[vis], ys[vis], strict=True):
                    painter.drawEllipse(QPointF((a - x0) / k, (b - y0) / k), r, r)
        painter.setBrush(Qt.BrushStyle.NoBrush)

    def _paint_boundaries(self, painter: QPainter):
        edges = self.project.boundaries
        if len(edges) == 0:
            return
        x0, y0 = self._origin()
        x1 = x0 + self.width() * self._km_per_px
        y1 = y0 + self.height() * self._km_per_px
        m = edges.length_km
        vis = (
            (edges.mid_x > x0 - m)
            & (edges.mid_x < x1 + m)
            & (edges.mid_y > y0 - m)
            & (edges.mid_y < y1 + m)
        )
        if not vis.any() or int(vis.sum()) > _MAX_OVERLAY_EDGES:
            return
        ax, ay, bx, by = edges.segment_endpoints()
        sax, say = self.to_screen(ax[vis], ay[vis])
        sbx, sby = self.to_screen(bx[vis], by[vis])
        kinds = edges.boundary_type[vis]
        width = float(np.clip(0.22 * self.project.layout.spacing_km / self._km_per_px, 1.6, 5.0))
        for kind in range(1, len(TYPE_NAMES)):
            sel = np.nonzero(kinds == kind)[0]
            if sel.size == 0:
                continue
            r, g, b = (int(v) for v in render.BOUNDARY_COLORS[kind])
            pen = QPen(QColor(r, g, b), width)
            pen.setCapStyle(Qt.PenCapStyle.RoundCap)
            painter.setPen(pen)
            painter.drawLines([QLineF(sax[i], say[i], sbx[i], sby[i]) for i in sel])

    def _paint_arrows(self, painter: QPainter):
        step_px = 64.0
        gx, gy, vx, vy = render.motion_arrows(self.project, step_px * self._km_per_px)
        if gx.size == 0 or gx.size > 4000:
            return
        speed = np.hypot(vx, vy)
        vmax = max(
            1e-6,
            float(
                np.hypot(self.project.layer("velocity_x"), self.project.layer("velocity_y")).max()
            ),
        )
        sx, sy = self.to_screen(gx, gy)
        for pen in (QPen(QColor(0, 0, 0, 170), 3.4), QPen(QColor(255, 255, 255), 1.5)):
            painter.setPen(pen)
            for i in range(gx.size):
                if not (0 <= sx[i] <= self.width() and 0 <= sy[i] <= self.height()):
                    continue
                length = 0.78 * step_px * speed[i] / vmax
                if length < 3:
                    continue
                ang = float(np.arctan2(vy[i], vx[i]))
                ex = sx[i] + length * np.cos(ang)
                ey = sy[i] + length * np.sin(ang)
                painter.drawLine(QPointF(sx[i], sy[i]), QPointF(ex, ey))
                for da in (2.6, -2.6):
                    painter.drawLine(
                        QPointF(ex, ey),
                        QPointF(ex + 7 * np.cos(ang + da), ey + 7 * np.sin(ang + da)),
                    )

    # -- interaction -------------------------------------------------------
    def resizeEvent(self, _event):
        self._invalidate()

    def wheelEvent(self, event):
        if self.project is None:
            return
        factor = 0.8 ** (event.angleDelta().y() / 120.0)
        pos = event.position()
        wx, wy = self.to_world(pos.x(), pos.y())
        spacing = self.project.layout.spacing_km
        b = self.project.geometry.viewport_bounds
        new = float(
            np.clip(
                self._km_per_px * factor,
                spacing / 80.0,
                4.0 * max(b.width, b.height) / max(1, self.width()),
            )
        )
        self._cx = wx - (pos.x() - 0.5 * self.width()) * new
        self._cy = wy - (pos.y() - 0.5 * self.height()) * new
        self._km_per_px = new
        self._invalidate(coarse=True)

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self._drag = event.position()
            self._moved = False

    def mouseMoveEvent(self, event):
        pos = event.position()
        if self._drag is not None:
            delta = pos - self._drag
            if abs(delta.x()) + abs(delta.y()) > 2:
                self._moved = True
            self._cx -= delta.x() * self._km_per_px
            self._cy -= delta.y() * self._km_per_px
            self._drag = pos
            self._invalidate(coarse=True)
        elif self.project is not None:
            hexagon = self._hex_at(pos)
            if hexagon is None:
                wx, wy = self.to_world(pos.x(), pos.y())
                self.hexHovered.emit(f"x {wx:,.0f} км, y {wy:,.0f} км - вне карты\nданных нет")
                return
            info = self.project.hex_info(*hexagon)
            text = (
                f"гекс ({info['col']}, {info['row']})  x {info['x_km']:,.0f} км, "
                f"y {info['y_km']:,.0f} км  |  плита {info['plate_id']} {info['plate_name']}"
            )
            if "elevation_m" in info or "macro_elevation_m" in info:
                height = info.get("elevation_m", info.get("macro_elevation_m"))
                rel = height - info["sea_level_m"]
                text += f"\n{'высота' if rel > 0 else 'глубина'} {abs(rel):,.0f} м"
                if "epoch_time_ma" in info:
                    text += f"  |  снимок t = {info['epoch_time_ma']:g} млн лет"
                    text += f"  |  материк {info['landmass_id']}"
                    text += f"  |  порода: {info['rock']}"
                    if info["mountain_system_id"] >= 0:
                        age = info["orogen_age_ma"]
                        aged = "нет данных" if age is None else f"{age:.0f} млн лет"
                        text += (
                            f"\nгорная система {info['mountain_system_id']}, "
                            f"возраст горообразования: {aged}"
                        )
                    else:
                        text += "\nгорной системы нет"
                else:
                    text += (
                        f"  |  блок {info['continental_block_id']}"
                        f"  |  материк {info['landmass_id']}"
                        "\nгеоистория: данных нет"
                    )
            else:
                text += (
                    f"  |  кора (prior): {info['crust']}  |  граница: {info['boundary']}"
                    "\nматерики, высоты, геоистория: данных нет"
                )
            if info.get("seafloor_province_title"):
                text += (
                    f"\nдно: {info['seafloor_province_title']}, глубина {info['depth_m']:,.0f} м"
                )
            if "temperature_mean_c" in info:
                text += (
                    f"\nT {info['temperature_mean_c']:+.1f} °C  |  осадки "
                    f"{info['precipitation_mm_yr']:,.0f} мм/год  |  сток "
                    f"{info['runoff_mm_yr']:,.0f} мм/год  |  расход "
                    f"{info['discharge_m3_s']:,.1f} м³/с"
                )
                if info.get("lake_id", -1) >= 0:
                    text += f"  |  озеро {info['lake_id']}"
                elif info.get("river_mask"):
                    text += f"  |  река, порядок {info['strahler_order']}"
                if info.get("coast_type", 0):
                    from ..stages.coasts import COAST_TITLES, COAST_TYPES

                    text += f"\nберег: {COAST_TITLES[COAST_TYPES[info['coast_type']]]}"
            if info.get("soil_present") is not None and info.get("land_mask"):
                from ..stages.environment import PARENT_MATERIALS, PARENT_TITLES

                pm = PARENT_TITLES[PARENT_MATERIALS[int(info.get("parent_material_class", 0))]]
                awc = info.get("awc_mm")
                wet = info.get("wetness_potential")
                text += (
                    f"\nпочва (оценка): {pm}; AWC {0 if awc is None else awc:.0f} мм; "
                    f"переувлажнение {0 if wet is None else wet:.2f}; сухих месяцев "
                    f"{info.get('dry_month_count', 0)}"
                )
            self.hexHovered.emit(text)

    def mouseReleaseEvent(self, event):
        if event.button() != Qt.MouseButton.LeftButton:
            return
        moved = self._moved
        self._drag = None
        if not moved and self.project is not None:
            hexagon = self._hex_at(event.position())
            if hexagon is not None:
                self.select_hex(hexagon)
                self.hexClicked.emit(hexagon[0], hexagon[1])

    def _hex_at(self, pos):
        wx, wy = self.to_world(pos.x(), pos.y())
        col, row = self.project.layout.world_to_offset(wx, wy)
        col, row = int(col), int(row)
        return (col, row) if self.project.layout.inside(col, row) else None
