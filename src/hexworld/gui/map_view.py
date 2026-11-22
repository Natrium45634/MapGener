"""Map widget: hex colours drawn as cached tiles, overlays as cached vector paths.

The widget knows nothing about individual views: it asks the view registry for
the colour of every hex and for the vector overlays. Zoom, pan and the selected
hex belong to the widget, so they are shared by all tabs.

Smoothness:

* the map is cut into 256 px tiles anchored in world coordinates; a tile is
  rendered once per zoom level and kept (LRU), so panning only draws the few
  tiles that come into view, and returning to a zoom level is instant;
* tiles that are missing are rendered in small slices from an idle timer, so a
  drag or a wheel turn never waits for the renderer. Until a tile is ready the
  previous frame, scaled, stands in for it (no blank holes, no blurry pass);
* overlays (rivers, ridges, boundaries, arrows, symbols) are built once as
  ``QPainterPath`` objects in world km, in spatial chunks, and drawn through a
  transform with cosmetic pens - nothing is rebuilt in Python on a pan.
"""

from __future__ import annotations

import math
import time
from collections import OrderedDict

import numpy as np
from PySide6.QtCore import QPointF, QRectF, Qt, QTimer, Signal
from PySide6.QtGui import (
    QBrush,
    QColor,
    QFont,
    QFontMetricsF,
    QImage,
    QPainter,
    QPainterPath,
    QPen,
    QTransform,
)
from PySide6.QtWidgets import QWidget

from .. import render, views
from ..project import Project
from ..stages.boundaries import TYPE_NAMES

TILE = 256
#: Tiles kept in memory (about 190 KB each).
MAX_TILES = 700
#: Time the idle renderer may take in one slice, seconds.
SLICE_S = 0.014
#: Overlay polylines are grouped into chunks of this many tiles across the map.
_CHUNKS = 12
_MAX_POINTS = 8000


#: Label drawing order (earlier wins a place) and style:
#: (point size, bold, italic, RGB, minimum size of the feature on screen px, capitals).
_LABEL_ORDER = ("ocean", "landmass", "sea", "mountains", "desert", "ice", "lake", "river")
_LABEL_STYLE = {
    "ocean": (15.0, False, True, (30, 60, 130), 150, True),
    "landmass": (13.0, True, False, (45, 32, 22), 70, True),
    "sea": (11.0, False, True, (35, 75, 150), 70, False),
    "mountains": (9.5, True, False, (105, 55, 20), 45, False),
    "desert": (9.5, False, True, (150, 95, 20), 50, False),
    "ice": (9.5, False, True, (35, 95, 150), 40, False),
    "lake": (8.5, False, True, (25, 70, 165), 14, False),
    "river": (8.5, False, True, (25, 85, 185), 110, False),
}


class _Chunked:
    """Paths of one overlay layer split into spatial chunks (world km)."""

    def __init__(self):
        self.parts: list[tuple[QRectF, QPainterPath]] = []

    def draw(self, painter: QPainter, view: QRectF):
        for rect, path in self.parts:
            if rect.intersects(view):
                painter.drawPath(path)


def _chunk_lines(segments, bounds) -> _Chunked:
    """``segments``: iterable of (xs, ys) arrays; returns chunked QPainterPaths."""
    out = _Chunked()
    width = max(bounds.width, 1e-6)
    height = max(bounds.height, 1e-6)
    groups: dict[tuple[int, int], QPainterPath] = {}
    for xs, ys in segments:
        xs = np.asarray(xs, dtype=np.float64)
        ys = np.asarray(ys, dtype=np.float64)
        if xs.size < 2:
            continue
        cx = int(np.clip((xs.mean() - bounds.xmin) / width * _CHUNKS, 0, _CHUNKS - 1))
        cy = int(np.clip((ys.mean() - bounds.ymin) / height * _CHUNKS, 0, _CHUNKS - 1))
        path = groups.get((cx, cy))
        if path is None:
            path = groups[(cx, cy)] = QPainterPath()
        path.moveTo(float(xs[0]), float(ys[0]))
        for a, b in zip(xs[1:].tolist(), ys[1:].tolist(), strict=True):
            path.lineTo(a, b)
    for path in groups.values():
        out.parts.append((path.boundingRect(), path))
    return out


class MapView(QWidget):
    hexHovered = Signal(str)
    hexClicked = Signal(int, int)
    #: A cross-section line drawn with the right mouse button: x0, y0, x1, y1 in world km.
    sectionDrawn = Signal(float, float, float, float)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMouseTracking(True)
        self.setMinimumSize(320, 240)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
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
        self._tiles: OrderedDict = OrderedDict()
        self._queue: list = []
        self._backdrop = None  # (QImage, x0, y0, km_per_px) of an earlier frame
        self._vectors: list | None = None
        self._vector_paths: list | None = None
        self._boundary_paths: list | None = None
        self._level_cache: dict = {}  # zoom-dependent paths (points, arrows)
        self._drag = None
        self._moved = False
        self._hover_hex = None
        self._section = None  # [x0, y0, x1, y1] world km while / after drawing
        self._section_live = False
        self.tiles_rendered = 0  # statistics for tests / diagnostics
        #: Place names on the map (oceans, continents, mountains, rivers ...).
        self.show_names = True
        self.labels_drawn: list[str] = []
        self._pump = QTimer(self)
        self._pump.setInterval(0)
        self._pump.timeout.connect(self._render_some)

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
        self._reset_content()
        if project is not None and not (keep_view or same_grid):
            self.fit()
        self.update()

    def set_view(self, view_id: str, mode: str, overlays) -> None:
        """Switch the presentation only: no generation; zoom, pan and selection are kept."""
        self.view_id = view_id
        self.mode = mode
        self.overlays = frozenset(overlays)
        self._reset_content()
        self.update()

    def _reset_content(self):
        self._hex_rgb = None
        self._tiles.clear()
        self._queue.clear()
        self._backdrop = None
        self._vectors = None
        self._vector_paths = None
        self._boundary_paths = None
        self._level_cache.clear()
        self._hover_hex = None

    def select_hex(self, hexagon: tuple[int, int] | None):
        self.selected_hex = hexagon
        self.update()

    def fit(self):
        if self.project is None:
            return
        b = self.project.geometry.viewport_bounds
        self._cx = 0.5 * (b.xmin + b.xmax)
        self._cy = 0.5 * (b.ymin + b.ymax)
        self._set_scale(
            1.04 * max(b.width / max(1, self.width()), b.height / max(1, self.height()))
        )
        self.update()

    def center_on(self, x_km: float, y_km: float, km_per_px: float | None = None):
        if km_per_px is not None:
            self._set_scale(km_per_px)
        self._cx, self._cy = float(x_km), float(y_km)
        self.update()

    def zoom_to_hex(self, col: int, row: int, px_per_hex: float = 40.0):
        if self.project is None:
            return
        x, y = self.project.layout.center(col, row)
        self.center_on(float(x), float(y), self.project.layout.spacing_km / px_per_hex)

    @property
    def km_per_px(self) -> float:
        return self._km_per_px

    def view_state(self) -> tuple[float, float, float]:
        return self._cx, self._cy, self._km_per_px

    def _scale_limits(self):
        spacing = self.project.layout.spacing_km
        b = self.project.geometry.viewport_bounds
        return spacing / 120.0, 4.0 * max(b.width, b.height) / max(1, self.width())

    def _set_scale(self, km_per_px: float):
        if self.project is not None:
            lo, hi = self._scale_limits()
            km_per_px = float(np.clip(km_per_px, lo, hi))
        if km_per_px != self._km_per_px:
            self._capture_backdrop()
            self._km_per_px = float(km_per_px)
            self._queue.clear()
            self._level_cache.clear()

    # -- coordinates -------------------------------------------------------
    def _origin(self):
        """World coordinates of the top-left pixel, snapped to the tile pixel grid."""
        k = self._km_per_px
        x0 = math.floor((self._cx - 0.5 * self.width() * k) / k) * k
        y0 = math.floor((self._cy - 0.5 * self.height() * k) / k) * k
        return x0, y0

    def to_world(self, px: float, py: float):
        x0, y0 = self._origin()
        return x0 + px * self._km_per_px, y0 + py * self._km_per_px

    def to_screen(self, x, y):
        x0, y0 = self._origin()
        return (x - x0) / self._km_per_px, (y - y0) / self._km_per_px

    def _world_transform(self) -> QTransform:
        x0, y0 = self._origin()
        k = self._km_per_px
        return QTransform(1.0 / k, 0.0, 0.0, 1.0 / k, -x0 / k, -y0 / k)

    def _view_rect_world(self, margin_px: float = 8.0) -> QRectF:
        x0, y0 = self._origin()
        k = self._km_per_px
        m = margin_px * k
        return QRectF(x0 - m, y0 - m, self.width() * k + 2 * m, self.height() * k + 2 * m)

    # -- tiles -------------------------------------------------------------
    def _ensure_colors(self):
        if self._hex_rgb is None:
            self._hex_rgb = views.view_rgb(self.project, self.view_id, self.mode, self.overlays)

    def _visible_tiles(self, ring: int = 0):
        k = self._km_per_px
        x0, y0 = self._origin()
        ix, iy = round(x0 / k), round(y0 / k)
        tx0 = (ix // TILE) - ring
        ty0 = (iy // TILE) - ring
        tx1 = ((ix + self.width() - 1) // TILE) + ring
        ty1 = ((iy + self.height() - 1) // TILE) + ring
        return [(k, tx, ty) for ty in range(ty0, ty1 + 1) for tx in range(tx0, tx1 + 1)]

    def _render_tile(self, key):
        k, tx, ty = key
        self._ensure_colors()
        img = render.render_view(
            self.project.layout, self._hex_rgb, tx * TILE * k, ty * TILE * k, k, TILE, TILE
        )
        buf = np.ascontiguousarray(img)
        qimg = QImage(buf.data, TILE, TILE, 3 * TILE, QImage.Format.Format_RGB888).convertToFormat(
            QImage.Format.Format_RGB32
        )
        self._bake_overlays(qimg, k, tx, ty)
        self._tiles[key] = qimg
        self._tiles.move_to_end(key)
        while len(self._tiles) > MAX_TILES:
            self._tiles.popitem(last=False)
        self.tiles_rendered += 1
        return qimg

    def _bake_overlays(self, qimg: QImage, k: float, tx: int, ty: int):
        """Draw the vector overlays into the tile: panning then costs only image copies."""
        boundaries = self.view_id == "plates" and "boundaries" in self.overlays
        arrows = self.view_id == "plates" and "motion" in self.overlays
        if self._vectors is None:
            self._vectors = views.view_vectors(self.project, self.view_id, self.mode, self.overlays)
        if not (boundaries or arrows or self._vectors):
            return
        transform = QTransform(1.0 / k, 0.0, 0.0, 1.0 / k, -tx * TILE, -ty * TILE)
        m = 14.0 * k  # line widths and symbols reach over the tile border
        view = QRectF(tx * TILE * k - m, ty * TILE * k - m, TILE * k + 2 * m, TILE * k + 2 * m)
        painter = QPainter(qimg)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        if boundaries:
            self._paint_boundaries(painter, transform, view)
        if arrows:
            self._paint_arrows(painter, transform, view)
        self._paint_vectors(painter, transform, view)
        painter.end()

    def _schedule(self, keys):
        cx, cy = self._cx / (TILE * self._km_per_px), self._cy / (TILE * self._km_per_px)
        keys = sorted(keys, key=lambda kk: (kk[1] + 0.5 - cx) ** 2 + (kk[2] + 0.5 - cy) ** 2)
        known = set(self._queue)
        self._queue.extend(kk for kk in keys if kk not in known)
        if self._queue and not self._pump.isActive():
            self._pump.start()

    def _render_some(self):
        if self.project is None:
            self._queue.clear()
        t0 = time.perf_counter()
        k = self._km_per_px
        wanted = set(self._visible_tiles(ring=1))
        drew = False
        while self._queue and time.perf_counter() - t0 < SLICE_S:
            key = self._queue.pop(0)
            if key[0] != k or key in self._tiles or key not in wanted:
                continue
            self._render_tile(key)
            drew = True
        if not self._queue:
            # prefetch one ring around the view once everything visible is ready
            missing = [kk for kk in self._visible_tiles(ring=1) if kk not in self._tiles]
            if missing:
                self._queue.extend(missing)
            else:
                self._pump.stop()
                self._backdrop = None if self._all_visible_ready() else self._backdrop
        if drew:
            self.update()

    def _all_visible_ready(self) -> bool:
        return all(kk in self._tiles for kk in self._visible_tiles())

    def flush(self):
        """Render every visible tile now (tests, screenshots, export)."""
        if self.project is None:
            return
        for key in self._visible_tiles():
            if key not in self._tiles:
                self._render_tile(key)
        self._queue.clear()
        self._backdrop = None
        self.update()

    def _capture_backdrop(self):
        """Keep the current frame so the next zoom level has something to show."""
        if self.project is None or self.width() <= 0 or self.height() <= 0:
            return
        if not self._tiles and self._backdrop is None:
            return
        img = QImage(self.width(), self.height(), QImage.Format.Format_RGB32)
        img.fill(QColor(*render.BACKGROUND))
        painter = QPainter(img)
        self._draw_backdrop(painter)
        self._draw_tiles(painter, schedule=False)
        painter.end()
        x0, y0 = self._origin()
        self._backdrop = (img, x0, y0, self._km_per_px)

    def _draw_backdrop(self, painter: QPainter):
        if self._backdrop is None:
            return
        img, bx0, by0, bk = self._backdrop
        x0, y0 = self._origin()
        k = self._km_per_px
        s = bk / k
        target = QRectF((bx0 - x0) / k, (by0 - y0) / k, img.width() * s, img.height() * s)
        painter.drawImage(target, img)

    def _draw_tiles(self, painter: QPainter, schedule: bool = True):
        k = self._km_per_px
        x0, y0 = self._origin()
        ix, iy = round(x0 / k), round(y0 / k)
        missing = []
        for key in self._visible_tiles():
            img = self._tiles.get(key)
            if img is None:
                missing.append(key)
                continue
            self._tiles.move_to_end(key)
            painter.drawImage(key[1] * TILE - ix, key[2] * TILE - iy, img)
        if schedule and missing:
            if self._backdrop is None and len(missing) == len(self._visible_tiles()):
                # a new view: draw it at once rather than show an empty map
                for key in missing:
                    painter.drawImage(
                        key[1] * TILE - ix, key[2] * TILE - iy, self._render_tile(key)
                    )
                self._schedule(self._visible_tiles(ring=1))
            else:
                self._schedule(missing)

    # -- painting ----------------------------------------------------------
    def paintEvent(self, _event):
        painter = QPainter(self)
        painter.fillRect(self.rect(), QColor(*render.BACKGROUND))
        if self.project is None:
            painter.setPen(QColor(170, 170, 170))
            painter.drawText(
                self.rect(),
                Qt.AlignmentFlag.AlignCenter,
                "Нажмите «🌍 Новый мир…», чтобы создать карту",
            )
            return
        ok, reason = views.VIEWS[self.view_id].availability(self.project)
        if not ok:
            painter.setPen(QColor(210, 210, 210))
            painter.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, reason)
            return
        self._draw_backdrop(painter)
        self._draw_tiles(painter)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        if self.show_names:
            self._paint_labels(painter)
        if self._section is not None:
            ax, ay = self.to_screen(self._section[0], self._section[1])
            bx, by = self.to_screen(self._section[2], self._section[3])
            for pen in (QPen(QColor(0, 0, 0), 4.0), QPen(QColor(255, 230, 80), 2.0)):
                painter.setPen(pen)
                painter.drawLine(QPointF(ax, ay), QPointF(bx, by))
        if self.selected_hex is not None:
            self._paint_selection(painter)

    def _paint_labels(self, painter: QPainter):
        """Names of the features large enough at this zoom; labels never overlap."""
        from .. import names

        try:
            labels = names.place_labels(self.project)
        except Exception:  # noqa: BLE001 - names are decoration
            labels = []
        self.labels_drawn = []
        if not labels:
            return
        k = self._km_per_px
        placed: list[QRectF] = []
        base = QFont(self.font())
        order = {kind: i for i, kind in enumerate(_LABEL_ORDER)}
        for lab in sorted(labels, key=lambda b: (order.get(b.kind, 9), -b.size_km)):
            pt, bold, italic, color, min_px, caps = _LABEL_STYLE[lab.kind]
            if lab.size_km / k < min_px:
                continue
            sx, sy = self.to_screen(lab.x_km, lab.y_km)
            if not (-50 <= sx <= self.width() + 50 and -20 <= sy <= self.height() + 20):
                continue
            text = lab.title_ru.upper() if caps else lab.title_ru
            font = QFont(base)
            font.setPointSizeF(pt)
            font.setBold(bold)
            font.setItalic(italic)
            if caps:
                font.setLetterSpacing(QFont.SpacingType.PercentageSpacing, 112)
            fm = QFontMetricsF(font)
            w, h = fm.horizontalAdvance(text), fm.height()
            rect = QRectF(sx - w / 2, sy - h / 2, w, h)
            # keep the whole label on the widget
            dx = max(0.0, 4.0 - rect.left()) - max(0.0, rect.right() - (self.width() - 4.0))
            dy = max(0.0, 2.0 - rect.top()) - max(0.0, rect.bottom() - (self.height() - 2.0))
            rect.translate(dx, dy)
            if any(rect.adjusted(-4, -2, 4, 2).intersects(r) for r in placed):
                continue
            placed.append(rect)
            path = QPainterPath()
            path.addText(QPointF(rect.left(), rect.top() + fm.ascent()), font, text)
            painter.setPen(QPen(QColor(255, 255, 255, 215), 3.2))
            painter.setBrush(Qt.BrushStyle.NoBrush)
            painter.drawPath(path)
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(QColor(*color))
            painter.drawPath(path)
            self.labels_drawn.append(text)
        painter.setBrush(Qt.BrushStyle.NoBrush)

    def _paint_selection(self, painter: QPainter):
        lay = self.project.layout
        wx, wy = lay.center(*self.selected_hex)
        sx, sy = self.to_screen(float(wx), float(wy))
        r = lay.radius / self._km_per_px
        if r < 5.0:
            for pen in (QPen(QColor(0, 0, 0), 4.0), QPen(QColor(255, 255, 255), 2.0)):
                painter.setPen(pen)
                painter.setBrush(Qt.BrushStyle.NoBrush)
                painter.drawEllipse(QPointF(sx, sy), 7.0, 7.0)
            return
        pts = [
            QPointF(
                sx + r * math.cos(math.radians(60 * i)), sy + r * math.sin(math.radians(60 * i))
            )
            for i in range(6)
        ]
        painter.setBrush(Qt.BrushStyle.NoBrush)
        for pen in (QPen(QColor(0, 0, 0), 4.5), QPen(QColor(255, 236, 90), 2.2)):
            painter.setPen(pen)
            painter.drawPolygon(pts)

    def _paint_vectors(self, painter: QPainter, transform: QTransform, view: QRectF):
        """Vector overlays of the current view (ridges, peaks, volcanoes, events ...)."""
        if self._vectors is None:
            self._vectors = views.view_vectors(self.project, self.view_id, self.mode, self.overlays)
        if not self._vectors:
            return
        bounds = self.project.geometry.viewport_bounds
        if self._vector_paths is None:
            self._vector_paths = [
                _chunk_lines(zip(layer.x, layer.y, strict=True), bounds)
                if layer.kind == "lines"
                else None
                for layer in self._vectors
            ]
        k = self._km_per_px
        grow = float(np.clip(0.02 * self.project.layout.spacing_km / k + 0.8, 0.8, 2.2))
        painter.save()
        painter.setTransform(transform)
        for i, layer in enumerate(self._vectors):
            color = QColor(*layer.color)
            if layer.kind == "lines":
                pen = QPen(color, layer.size * grow)
                pen.setCosmetic(True)
                pen.setCapStyle(Qt.PenCapStyle.RoundCap)
                pen.setJoinStyle(Qt.PenJoinStyle.RoundJoin)
                painter.setPen(pen)
                painter.setBrush(Qt.BrushStyle.NoBrush)
                self._vector_paths[i].draw(painter, view)
            else:
                path = self._points_path(i, layer, grow, view)
                if path is None:
                    continue
                pen = QPen(QColor(20, 20, 20) if layer.filled else color, 1.4)
                pen.setCosmetic(True)
                painter.setPen(pen)
                painter.setBrush(QBrush(color) if layer.filled else Qt.BrushStyle.NoBrush)
                painter.drawPath(path)
        painter.restore()
        painter.setBrush(Qt.BrushStyle.NoBrush)

    def _points_path(self, i, layer, grow, view: QRectF):
        """Circles of a point layer for the current zoom; cached per zoom level."""
        xs, ys = np.asarray(layer.x, dtype=np.float64), np.asarray(layer.y, dtype=np.float64)
        key = ("points", i)
        cached = self._level_cache.get(key)
        if cached is None:
            k = self._km_per_px
            r = max(layer.size * grow, layer.size_km / k) * k  # world km
            cached = []
            if xs.size:
                b = self.project.geometry.viewport_bounds
                cx = np.clip(((xs - b.xmin) / max(b.width, 1e-6) * _CHUNKS).astype(int), 0, 11)
                cy = np.clip(((ys - b.ymin) / max(b.height, 1e-6) * _CHUNKS).astype(int), 0, 11)
                cell = cx * _CHUNKS + cy
                for c in np.unique(cell):
                    sel = cell == c
                    path = QPainterPath()
                    for a, bb in zip(xs[sel].tolist(), ys[sel].tolist(), strict=True):
                        path.addEllipse(QPointF(a, bb), r, r)
                    cached.append((path.boundingRect(), path, int(sel.sum())))
            self._level_cache[key] = cached
        vis = [(rect, path, n) for rect, path, n in cached if rect.intersects(view)]
        if not vis or sum(n for _r, _p, n in vis) > _MAX_POINTS:
            return None
        out = QPainterPath()
        for _rect, path, _n in vis:
            out.addPath(path)
        return out

    def _paint_boundaries(self, painter: QPainter, transform: QTransform, view: QRectF):
        edges = self.project.boundaries
        if len(edges) == 0:
            return
        if self._boundary_paths is None:
            ax, ay, bx, by = edges.segment_endpoints()
            bounds = self.project.geometry.viewport_bounds
            self._boundary_paths = []
            for kind in range(1, len(TYPE_NAMES)):
                sel = np.nonzero(edges.boundary_type == kind)[0]
                if sel.size == 0:
                    continue
                segs = ((np.array([ax[j], bx[j]]), np.array([ay[j], by[j]])) for j in sel.tolist())
                color = tuple(int(v) for v in render.BOUNDARY_COLORS[kind])
                self._boundary_paths.append((color, _chunk_lines(segs, bounds)))
        width = float(np.clip(0.22 * self.project.layout.spacing_km / self._km_per_px, 1.6, 5.0))
        painter.save()
        painter.setTransform(transform)
        for color, chunked in self._boundary_paths:
            pen = QPen(QColor(*color), width)
            pen.setCosmetic(True)
            pen.setCapStyle(Qt.PenCapStyle.RoundCap)
            painter.setPen(pen)
            chunked.draw(painter, view)
        painter.restore()

    def _paint_arrows(self, painter: QPainter, transform: QTransform, view: QRectF):
        """Plate-motion arrows on a world-anchored grid, only the anchors near ``view``."""
        step_px = 64.0
        k = self._km_per_px
        step = step_px * k
        b = self.project.geometry.viewport_bounds
        reach = 0.8 * step  # an arrow is at most 0.78 grid steps long

        def anchors(lo, hi, start, stop):
            i0 = max(0, math.floor((lo - reach - start) / step - 0.5))
            i1 = math.ceil((hi + reach - start) / step)
            v = start + (np.arange(i0, max(i0, i1)) + 0.5) * step
            return v[v < stop]

        xs = anchors(view.left(), view.right(), b.xmin, b.xmax)
        ys = anchors(view.top(), view.bottom(), b.ymin, b.ymax)
        path = QPainterPath()
        if xs.size and ys.size and xs.size * ys.size <= 40000:
            gx, gy = np.meshgrid(xs, ys)
            col, row = self.project.layout.world_to_offset(gx, gy)
            ok = self.project.layout.inside(col, row)
            gx, gy, col, row = gx[ok], gy[ok], col[ok], row[ok]
            vx = self.project.layer("velocity_x")[row, col]
            vy = self.project.layer("velocity_y")[row, col]
            vmax = self._level_cache.get("vmax")
            if vmax is None:
                vmax = self._level_cache["vmax"] = max(
                    1e-6,
                    float(
                        np.hypot(
                            self.project.layer("velocity_x"), self.project.layer("velocity_y")
                        ).max()
                    ),
                )
            speed = np.hypot(vx, vy)
            for i in range(gx.size):
                length = 0.78 * step_px * speed[i] / vmax
                if length < 3:
                    continue
                ang = float(np.arctan2(vy[i], vx[i]))
                sx, sy = float(gx[i]), float(gy[i])
                ex = sx + length * k * math.cos(ang)
                ey = sy + length * k * math.sin(ang)
                path.moveTo(sx, sy)
                path.lineTo(ex, ey)
                for da in (2.6, -2.6):
                    path.moveTo(ex, ey)
                    path.lineTo(ex + 7 * k * math.cos(ang + da), ey + 7 * k * math.sin(ang + da))
        if path.isEmpty():
            return
        painter.save()
        painter.setTransform(transform)
        painter.setBrush(Qt.BrushStyle.NoBrush)
        for pen in (QPen(QColor(0, 0, 0, 170), 3.4), QPen(QColor(255, 255, 255), 1.5)):
            pen.setCosmetic(True)
            painter.setPen(pen)
            painter.drawPath(path)
        painter.restore()

    # -- interaction -------------------------------------------------------
    def resizeEvent(self, _event):
        self.update()

    def zoom_by(self, factor: float, anchor: QPointF | None = None):
        if self.project is None:
            return
        if anchor is None:
            anchor = QPointF(0.5 * self.width(), 0.5 * self.height())
        wx, wy = self.to_world(anchor.x(), anchor.y())
        self._set_scale(self._km_per_px * factor)
        new = self._km_per_px
        self._cx = wx - (anchor.x() - 0.5 * self.width()) * new
        self._cy = wy - (anchor.y() - 0.5 * self.height()) * new
        self.update()

    def wheelEvent(self, event):
        if self.project is None:
            return
        steps = event.angleDelta().y() / 120.0
        if steps == 0:
            return
        self.zoom_by(0.8**steps, event.position())

    def keyPressEvent(self, event):
        key = event.key()
        step = 80.0 * self._km_per_px
        if key in (Qt.Key.Key_Plus, Qt.Key.Key_Equal):
            self.zoom_by(0.8)
        elif key == Qt.Key.Key_Minus:
            self.zoom_by(1.25)
        elif key == Qt.Key.Key_Left:
            self._cx -= step
        elif key == Qt.Key.Key_Right:
            self._cx += step
        elif key == Qt.Key.Key_Up:
            self._cy -= step
        elif key == Qt.Key.Key_Down:
            self._cy += step
        elif key == Qt.Key.Key_Home:
            self.fit()
        else:
            super().keyPressEvent(event)
            return
        self.update()

    def mouseDoubleClickEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self.zoom_by(0.5, event.position())

    def mousePressEvent(self, event):
        self.setFocus()
        if event.button() in (Qt.MouseButton.LeftButton, Qt.MouseButton.MiddleButton):
            self._drag = event.position()
            self._moved = False
        elif event.button() == Qt.MouseButton.RightButton and self.project is not None:
            wx, wy = self.to_world(event.position().x(), event.position().y())
            self._section = [wx, wy, wx, wy]
            self._section_live = True
            self.update()

    def mouseMoveEvent(self, event):
        pos = event.position()
        if self._section_live and self._section is not None:
            self._section[2], self._section[3] = self.to_world(pos.x(), pos.y())
            self.update()
            return
        if self._drag is not None:
            delta = pos - self._drag
            if abs(delta.x()) + abs(delta.y()) > 2 or self._moved:
                self._moved = True
                self._cx -= delta.x() * self._km_per_px
                self._cy -= delta.y() * self._km_per_px
                self._drag = pos
                self.update()
            return
        if self.project is None:
            return
        hexagon = self._hex_at(pos)
        if hexagon == self._hover_hex and hexagon is not None:
            return
        self._hover_hex = hexagon
        if hexagon is None:
            wx, wy = self.to_world(pos.x(), pos.y())
            self.hexHovered.emit(f"x {wx:,.0f} км, y {wy:,.0f} км - вне карты\nданных нет")
            return
        self.hexHovered.emit(hover_text(self.project, *hexagon))

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.MouseButton.RightButton and self._section_live:
            self._section_live = False
            x0, y0, x1, y1 = self._section
            if np.hypot(x1 - x0, y1 - y0) > 2.0 * self._km_per_px:
                self.sectionDrawn.emit(float(x0), float(y0), float(x1), float(y1))
            else:
                self._section = None
            self.update()
            return
        if event.button() not in (Qt.MouseButton.LeftButton, Qt.MouseButton.MiddleButton):
            return
        moved = self._moved
        self._drag = None
        if event.button() == Qt.MouseButton.LeftButton and not moved and self.project is not None:
            hexagon = self._hex_at(event.position())
            if hexagon is not None:
                self.select_hex(hexagon)
                self.hexClicked.emit(hexagon[0], hexagon[1])

    def _hex_at(self, pos):
        wx, wy = self.to_world(pos.x(), pos.y())
        col, row = self.project.layout.world_to_offset(wx, wy)
        col, row = int(col), int(row)
        return (col, row) if self.project.layout.inside(col, row) else None


def hover_text(project: Project, col: int, row: int) -> str:
    """Short two-to-four line summary of a hex for the hover panel."""
    info = project.hex_info(col, row)
    text = (
        f"гекс ({info['col']}, {info['row']})  x {info['x_km']:,.0f} км, "
        f"y {info['y_km']:,.0f} км  |  плита {info['plate_id']} {info['plate_name']}"
    )
    if "elevation_m" in info or "macro_elevation_m" in info:
        height = info.get("elevation_m", info.get("macro_elevation_m"))
        rel = height - info["sea_level_m"]
        text += f"\n{'высота' if rel > 0 else 'глубина'} {abs(rel):,.0f} м"
        if "epoch_time_ma" in info:
            text += f"  |  материк {info['landmass_id']}  |  порода: {info['rock']}"
    if info.get("biome_title"):
        text += f"\nбиом: {info['biome_title']}"
    if "temperature_mean_c" in info:
        text += (
            f"\nT {info['temperature_mean_c']:+.1f} °C  |  осадки "
            f"{info['precipitation_mm_yr']:,.0f} мм/год"
        )
        if info.get("lake_id", -1) >= 0:
            text += f"  |  озеро {info['lake_id']}"
        elif info.get("river_mask"):
            text += f"  |  река, порядок {info['strahler_order']}"
    text += "\nщелчок — вся информация о гексе справа"
    return text
