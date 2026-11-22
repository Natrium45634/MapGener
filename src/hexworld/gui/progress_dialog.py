"""Generation progress window: stages, timers, ETA and a heartbeat (never looks frozen)."""

from __future__ import annotations

import time

from PySide6.QtCore import Qt, QTimer, Signal
from PySide6.QtWidgets import (
    QDialog,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QProgressBar,
    QPushButton,
    QVBoxLayout,
)

from ..progress import ProgressTracker, format_seconds

_SPIN = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"


class ProgressDialog(QDialog):
    """Shows a :class:`ProgressTracker`; ``cancel_requested`` asks the worker to stop."""

    cancel_requested = Signal()

    def __init__(self, title: str, tracker: ProgressTracker, parent=None):
        super().__init__(parent)
        self.setWindowTitle(title)
        self.setModal(False)
        self.setMinimumWidth(560)
        self.tracker = tracker
        self._last_report = time.monotonic()
        self._spin = 0
        lay = QVBoxLayout(self)
        self.head = QLabel(f"<b>{title}</b>")
        lay.addWidget(self.head)
        self.overall = QProgressBar()
        self.overall.setRange(0, 1000)
        self.overall.setFormat("%p %")
        self.overall.setTextVisible(True)
        lay.addWidget(self.overall)
        self.stage_label = QLabel("")
        self.stage_label.setWordWrap(True)
        lay.addWidget(self.stage_label)
        self.stage_bar = QProgressBar()
        self.stage_bar.setRange(0, 1000)
        self.stage_bar.setMaximumHeight(10)
        self.stage_bar.setTextVisible(False)
        lay.addWidget(self.stage_bar)
        self.times = QLabel("")
        lay.addWidget(self.times)
        self.pulse = QLabel("")
        self.pulse.setStyleSheet("color: #2a7a2a")
        lay.addWidget(self.pulse)
        self.stages = QListWidget()
        self.stages.setMinimumHeight(220)
        lay.addWidget(self.stages)
        row = QHBoxLayout()
        row.addStretch(1)
        self.hide_btn = QPushButton("Свернуть (генерация продолжится)")
        self.hide_btn.clicked.connect(self.hide)
        self.cancel_btn = QPushButton("Отмена")
        self.cancel_btn.clicked.connect(self._cancel)
        row.addWidget(self.hide_btn)
        row.addWidget(self.cancel_btn)
        lay.addLayout(row)
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.tick)
        self.timer.start(150)
        self.refresh()

    def _cancel(self):
        self.cancel_btn.setEnabled(False)
        self.cancel_btn.setText("Отмена после текущего шага…")
        self.cancel_requested.emit()

    def report(self, stage: str, fraction: float):
        self.tracker.report(stage, fraction)
        self._last_report = time.monotonic()
        self.refresh()

    def tick(self):
        self._spin = (self._spin + 1) % len(_SPIN)
        self.refresh(light=True)

    def refresh(self, light: bool = False):
        st = self.tracker.state()
        self.overall.setValue(int(1000 * st.overall))
        how = "оценка по времени" if st.estimated else "этап сообщает свой прогресс"
        self.stage_label.setText(
            f"Сейчас: <b>{st.title}</b> — {format_seconds(st.stage_elapsed_s)} ({how})"
        )
        self.stage_bar.setValue(int(1000 * st.stage_fraction))
        total = None if st.eta_s is None else st.elapsed_s + st.eta_s
        self.times.setText(
            f"Прошло {format_seconds(st.elapsed_s)} · осталось ~{format_seconds(st.eta_s)}"
            f" · всего ~{format_seconds(total)}"
        )
        quiet = time.monotonic() - self._last_report
        note = "генерация идёт"
        if quiet > 20:
            note = (
                f"этап долгий, это нормально (последний отчёт {format_seconds(quiet)} назад; "
                "счётчик времени идёт — программа не зависла)"
            )
        self.pulse.setText(f"{_SPIN[self._spin]} {note}")
        if light and self.stages.count():
            return
        self.stages.clear()
        for _k, title, secs in st.done:
            self.stages.addItem(QListWidgetItem(f"✓ {title}  ({format_seconds(secs)})"))
        cur = QListWidgetItem(f"▶ {st.title}")
        f = cur.font()
        f.setBold(True)
        cur.setFont(f)
        self.stages.addItem(cur)
        for _k, title in st.pending:
            item = QListWidgetItem(f"○ {title}")
            item.setForeground(Qt.GlobalColor.gray)
            self.stages.addItem(item)
        self.stages.scrollToItem(cur)

    def finish(self):
        self.timer.stop()
        self.close()
