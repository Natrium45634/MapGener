"""«Мои миры»: history of generated worlds and favourites, kept between sessions.

Each entry stores the world code (seed + settings), a summary and a thumbnail — not the
world itself: any entry is regenerated exactly from its code. Qt-free.
"""

from __future__ import annotations

import json
import os
import sys
import time
import uuid
from pathlib import Path

from .world_tools import thumbnail, world_code, world_summary

#: Non-favourite entries kept (oldest are dropped first); favourites are never dropped.
MAX_HISTORY = 120


def data_dir() -> Path:
    """Per-user folder for HexWorld's own files (``HEXWORLD_DATA_DIR`` overrides)."""
    env = os.environ.get("HEXWORLD_DATA_DIR")
    if env:
        return Path(env)
    if sys.platform.startswith("win"):
        return Path(os.environ.get("APPDATA") or Path.home() / "AppData" / "Roaming") / "HexWorld"
    if sys.platform == "darwin":
        return Path.home() / "Library" / "Application Support" / "HexWorld"
    return Path(os.environ.get("XDG_DATA_HOME") or Path.home() / ".local" / "share") / "hexworld"


class WorldLibrary:
    def __init__(self, root: str | Path | None = None):
        self.root = Path(root) if root is not None else data_dir() / "library"
        self.file = self.root / "library.json"
        self.thumbs = self.root / "thumbs"
        self.entries: list[dict] = []
        self.load()

    # -- storage ---------------------------------------------------------------------------
    def load(self) -> None:
        try:
            data = json.loads(self.file.read_text(encoding="utf-8"))
            self.entries = [e for e in data.get("entries", []) if "id" in e and "code" in e]
        except (OSError, ValueError):
            self.entries = []

    def save(self) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        tmp = self.file.with_suffix(".tmp")
        payload = {"format": "hexworld-library-1", "entries": self.entries}
        tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=1), encoding="utf-8")
        os.replace(tmp, self.file)

    # -- entries ---------------------------------------------------------------------------
    def add(self, project, through: str = "v50", title: str | None = None, config=None) -> dict:
        """Remember a world. ``config`` overrides the project's (a sketch of a full world)."""
        from .render import write_png

        eid = uuid.uuid4().hex[:12]
        thumb = None
        try:
            thumb_path = self.thumbs / f"{eid}.png"
            write_png(thumb_path, thumbnail(project, "biomes", 320))
            thumb = thumb_path.name
        except Exception:  # noqa: BLE001 - a missing picture must not lose the entry
            thumb = None
        summary = world_summary(project)
        entry = {
            "id": eid,
            "created": time.strftime("%Y-%m-%d %H:%M"),
            "seed": (config or project.config).seed,
            "title": title or (config or project.config).seed,
            "code": world_code(config or project.config, through),
            "sketch": config is not None,
            "through": through,
            "summary": summary,
            "thumb": thumb,
            "favourite": False,
        }
        self.entries.insert(0, entry)
        self._prune()
        self.save()
        return entry

    def get(self, eid: str) -> dict | None:
        return next((e for e in self.entries if e["id"] == eid), None)

    def thumb_path(self, entry: dict) -> Path | None:
        if not entry.get("thumb"):
            return None
        p = self.thumbs / entry["thumb"]
        return p if p.exists() else None

    def set_favourite(self, eid: str, on: bool = True) -> None:
        e = self.get(eid)
        if e is not None:
            e["favourite"] = bool(on)
            self.save()

    def rename(self, eid: str, title: str) -> None:
        e = self.get(eid)
        if e is not None:
            e["title"] = title.strip() or e["seed"]
            self.save()

    def remove(self, eid: str) -> None:
        e = self.get(eid)
        if e is None:
            return
        self.entries.remove(e)
        p = self.thumb_path(e)
        if p is not None:
            p.unlink(missing_ok=True)
        self.save()

    def favourites(self) -> list[dict]:
        return [e for e in self.entries if e.get("favourite")]

    def _prune(self) -> None:
        plain = [e for e in self.entries if not e.get("favourite")]
        for e in plain[MAX_HISTORY:]:
            self.entries.remove(e)
            p = self.thumb_path(e)
            if p is not None:
                p.unlink(missing_ok=True)
