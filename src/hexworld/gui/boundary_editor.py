"""«Границы региона» editor (TZ 5.4.1 G1): presets per side, JSON, validation, import / export.

The result is the canonical JSON of :mod:`hexworld.boundary`; the dialog never fills a
missing value with zeros, it shows the validation errors instead.
"""

from __future__ import annotations

from PySide6.QtWidgets import (
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QPlainTextEdit,
    QPushButton,
    QVBoxLayout,
)

from .. import boundary as bnd

SIDE_TITLES = {"west": "Запад", "east": "Восток", "north": "Север", "south": "Юг"}


class BoundaryEditor(QDialog):
    def __init__(self, text: str = "", parent=None):
        super().__init__(parent)
        self.setWindowTitle("Границы региона")
        self.resize(760, 620)
        lay = QVBoxLayout(self)
        lay.addWidget(
            QLabel(
                "Что входит в область через её края. Каждой стороне — пресет или свой профиль "
                "(12 месяцев). Воздух проходит над любым краем, вода — только через морские."
            )
        )
        grid = QGridLayout()
        self.combos = {}
        for r, side in enumerate(bnd.SIDES):
            grid.addWidget(QLabel(SIDE_TITLES[side]), r, 0)
            cb = QComboBox()
            cb.addItem("— не задано —", "")
            for key, title in bnd.PRESET_TITLES.items():
                cb.addItem(title, key)
            self.combos[side] = cb
            grid.addWidget(cb, r, 1)
        lay.addLayout(grid)
        row = QHBoxLayout()
        apply_presets = QPushButton("Пресеты → JSON")
        apply_presets.clicked.connect(self.presets_to_json)
        check = QPushButton("Проверить")
        check.clicked.connect(self.check)
        imp = QPushButton("Импорт…")
        imp.clicked.connect(self.import_file)
        exp = QPushButton("Экспорт…")
        exp.clicked.connect(self.export_file)
        clear = QPushButton("Сброс")
        clear.clicked.connect(lambda: self.text.setPlainText(""))
        for b in (apply_presets, check, imp, exp, clear):
            row.addWidget(b)
        lay.addLayout(row)
        self.text = QPlainTextEdit()
        self.text.setPlainText(self._pretty(text))
        lay.addWidget(self.text, 1)
        self.status = QLabel("")
        self.status.setWordWrap(True)
        lay.addWidget(self.status)
        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self._accept)
        buttons.rejected.connect(self.reject)
        lay.addWidget(buttons)
        self.result_text = text

    @staticmethod
    def _pretty(text: str) -> str:
        if not text:
            return ""
        import json

        return json.dumps(json.loads(text), ensure_ascii=False, indent=1)

    def presets_to_json(self):
        profiles = []
        for side, cb in self.combos.items():
            key = cb.currentData()
            if key:
                profiles.append(bnd.from_preset(f"{side}_{key}", (side,), key))
        self.text.setPlainText(self._pretty(bnd.to_json(profiles)) if profiles else "")
        self.check()

    def _parse(self):
        raw = self.text.toPlainText().strip()
        if not raw:
            return []
        return bnd.from_json(raw)

    def check(self) -> bool:
        try:
            profiles = self._parse()
        except Exception as exc:  # noqa: BLE001 - shown to the user
            self.status.setText(f"Ошибка JSON: {exc}")
            return False
        errs = bnd.validate(profiles)
        if errs:
            self.status.setText("Ошибки:\n" + "\n".join(errs))
            return False
        parts = []
        for p in profiles:
            col = p.column_kg_m2()
            extra = f", пар на входе {min(col):.0f}–{max(col):.0f} кг/м²" if col else ""
            parts.append(f"{'/'.join(p.sides)}: {p.atm_mode}{extra}, океан {p.ocn_mode}")
        self.status.setText(
            "Профили корректны.\n" + "\n".join(parts)
            if parts
            else "Профили не заданы: приток по нулевому градиенту (синтетика)."
        )
        return True

    def import_file(self):
        path, _ = QFileDialog.getOpenFileName(self, "Импорт профилей", "", "JSON (*.json)")
        if path:
            with open(path, encoding="utf-8") as f:
                self.text.setPlainText(f.read())
            self.check()

    def export_file(self):
        if not self.check():
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "Экспорт профилей", "boundary.json", "JSON (*.json)"
        )
        if path:
            with open(path, "w", encoding="utf-8") as f:
                f.write(bnd.to_json(self._parse()))

    def _accept(self):
        if not self.check():
            return
        profiles = self._parse()
        self.result_text = bnd.to_json(profiles) if profiles else ""
        self.accept()
