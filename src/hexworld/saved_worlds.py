"""Saved worlds: one folder per world, plus the user's preferences (visible map tabs,
where the worlds live, recent worlds). Qt-free (GUI, CLI and tests use it).

A world folder::

    <worlds root>/<name>/
        world.json    title, seed, world code, stage, dates, versions, notes, summary
        code.txt      the world code «HW2-…» alone (copy it to share the world)
        preview.png   a small picture of the world
        project/      the full generated world (only for «with data»: opens instantly)

A folder without ``project/`` is a saved seed / settings: opening it regenerates the world
exactly from its code. Folders can be copied, renamed or moved by hand.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import time
from pathlib import Path

from .world_library import data_dir
from .world_tools import parse_world_code, world_code

FORMAT = "hexworld-saved-world-1"
PREFS_FORMAT = "hexworld-preferences-1"
MAX_RECENT = 10


# -- preferences -------------------------------------------------------------------------------
def prefs_path() -> Path:
    return data_dir() / "preferences.json"


def load_prefs() -> dict:
    try:
        data = json.loads(prefs_path().read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except (OSError, ValueError):
        return {}


def save_prefs(prefs: dict) -> None:
    p = prefs_path()
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(".tmp")
    tmp.write_text(json.dumps({**prefs, "format": PREFS_FORMAT}, ensure_ascii=False, indent=1),
                   encoding="utf-8")  # fmt: skip
    os.replace(tmp, p)


def visible_views(all_ids: list[str], prefs: dict | None = None) -> list[str]:
    """Map tabs to show, in the registry order (all when nothing was chosen)."""
    prefs = load_prefs() if prefs is None else prefs
    hidden = set(prefs.get("hidden_views", []))
    shown = [v for v in all_ids if v not in hidden]
    return shown or list(all_ids)


def set_visible_views(shown: list[str], all_ids: list[str]) -> None:
    prefs = load_prefs()
    prefs["hidden_views"] = [v for v in all_ids if v not in set(shown)]
    save_prefs(prefs)


def default_worlds_root() -> Path:
    docs = Path.home() / "Documents"
    return (docs if docs.is_dir() else Path.home()) / "HexWorld" / "Миры"


def worlds_root() -> Path:
    env = os.environ.get("HEXWORLD_WORLDS_DIR")
    if env:
        return Path(env)
    p = load_prefs().get("worlds_root")
    return Path(p) if p else default_worlds_root()


def set_worlds_root(path: str | Path) -> None:
    prefs = load_prefs()
    prefs["worlds_root"] = str(path)
    save_prefs(prefs)


def remember_recent(folder: str | Path) -> None:
    prefs = load_prefs()
    rec = [str(folder)] + [r for r in prefs.get("recent_worlds", []) if r != str(folder)]
    prefs["recent_worlds"] = rec[:MAX_RECENT]
    save_prefs(prefs)


def recent_worlds() -> list[Path]:
    return [
        Path(r) for r in load_prefs().get("recent_worlds", []) if Path(r, "world.json").exists()
    ]


# -- world folders ----------------------------------------------------------------------------
_BAD = re.compile(r'[<>:"/\\|?*\x00-\x1f]+')


def safe_name(title: str) -> str:
    """A folder name valid on Windows, macOS and Linux (keeps Cyrillic)."""
    name = _BAD.sub("_", str(title)).strip(" .")
    name = re.sub(r"\s+", " ", name)[:80]
    if name.upper() in {"CON", "PRN", "AUX", "NUL"} or re.fullmatch(r"(?i)(COM|LPT)\d", name):
        name = f"_{name}"
    return name or "Мир"


def _unique(root: Path, name: str) -> Path:
    p = root / name
    k = 2
    while p.exists():
        p = root / f"{name} ({k})"
        k += 1
    return p


def save_world(project, title: str | None = None, *, root: str | Path | None = None,
               through: str = "v50", with_data: bool = True, notes: str = "",
               folder: str | Path | None = None) -> Path:  # fmt: skip
    """Save a generated world into its own folder; returns the folder.

    ``with_data`` stores the whole world (opens without regeneration); otherwise only the
    code, so opening it regenerates the identical world. ``folder`` overwrites an existing
    saved world (same folder) instead of creating a new one.
    """
    from .io import save_project
    from .render import write_png
    from .version import GENERATOR_VERSION, PRODUCT_VERSION
    from .world_tools import thumbnail, world_summary

    cfg = project.config
    title = (title or cfg.seed).strip() or cfg.seed
    if folder is None:
        base = Path(root) if root is not None else worlds_root()
        base.mkdir(parents=True, exist_ok=True)
        folder = _unique(base, safe_name(title))
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    code = world_code(cfg, through)
    if with_data:
        save_project(project, folder / "project")
    elif (folder / "project").exists():
        shutil.rmtree(folder / "project")
    try:
        write_png(
            folder / "preview.png",
            thumbnail(project, "biomes" if project.biomes else "heights", 480),
        )
    except Exception:  # noqa: BLE001 - a missing picture must not lose the world
        pass
    try:
        summary = world_summary(project)
    except Exception:  # noqa: BLE001
        summary = {}
    _write_meta(folder, {
        "title": title, "seed": cfg.seed, "code": code, "through": through,
        "has_data": bool(with_data), "notes": notes, "summary": summary,
        "size": [cfg.width, cfg.height, cfg.hex_spacing_km, cfg.mode],
        "product_version": PRODUCT_VERSION, "generator_version": GENERATOR_VERSION,
    })  # fmt: skip
    return folder


def save_code(config, title: str | None = None, *, root: str | Path | None = None,
              through: str = "v50", notes: str = "") -> Path:  # fmt: skip
    """Save only a seed with all its settings (no world data): a light entry."""
    from .version import GENERATOR_VERSION, PRODUCT_VERSION

    title = (title or config.seed).strip() or config.seed
    base = Path(root) if root is not None else worlds_root()
    base.mkdir(parents=True, exist_ok=True)
    folder = _unique(base, safe_name(title))
    folder.mkdir(parents=True)
    _write_meta(folder, {
        "title": title, "seed": config.seed, "code": world_code(config, through),
        "through": through, "has_data": False, "notes": notes, "summary": {},
        "size": [config.width, config.height, config.hex_spacing_km, config.mode],
        "product_version": PRODUCT_VERSION, "generator_version": GENERATOR_VERSION,
    })  # fmt: skip
    return folder


def _write_meta(folder: Path, meta: dict) -> None:
    old = read_meta(folder) or {}
    now = time.strftime("%Y-%m-%d %H:%M")
    doc = {"format": FORMAT, "created": old.get("created", now), "saved": now, **meta}
    tmp = folder / "world.json.tmp"
    tmp.write_text(json.dumps(doc, ensure_ascii=False, indent=1), encoding="utf-8")
    os.replace(tmp, folder / "world.json")
    (folder / "code.txt").write_text(meta["code"] + "\n", encoding="utf-8")


def read_meta(folder: str | Path) -> dict | None:
    try:
        doc = json.loads(Path(folder, "world.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if doc.get("format") != FORMAT or "code" not in doc:
        return None
    return doc


def list_worlds(root: str | Path | None = None) -> list[dict]:
    """Saved worlds under ``root``, newest first: ``meta + {"folder", "preview"}``."""
    base = Path(root) if root is not None else worlds_root()
    out = []
    if not base.is_dir():
        return out
    for sub in base.iterdir():
        if not sub.is_dir():
            continue
        meta = read_meta(sub)
        if meta is None:
            continue
        meta["folder"] = str(sub)
        meta["has_data"] = (sub / "project" / "manifest.json").exists()
        pv = sub / "preview.png"
        meta["preview"] = str(pv) if pv.exists() else None
        out.append(meta)
    out.sort(key=lambda m: m.get("saved", ""), reverse=True)
    return out


def open_world(folder: str | Path):
    """``(project or None, config, through)``: the stored world when it has data, else
    the configuration to regenerate it from its code."""
    meta = read_meta(folder)
    if meta is None:
        raise ValueError(f"{folder}: это не папка сохранённого мира HexWorld")
    config, through = parse_world_code(meta["code"])
    project = None
    if (Path(folder) / "project" / "manifest.json").exists():
        from .io import load_project

        project = load_project(Path(folder) / "project")
    remember_recent(folder)
    return project, config, through


def rename_world(folder: str | Path, title: str) -> Path:
    """New title (and folder name) of a saved world; returns the new folder."""
    folder = Path(folder)
    meta = read_meta(folder)
    if meta is None:
        raise ValueError("не папка сохранённого мира")
    meta["title"] = title.strip() or meta["seed"]
    new = folder
    wanted = safe_name(meta["title"])
    if folder.name != wanted:
        new = _unique(folder.parent, wanted)
        folder.rename(new)
    _write_meta(new, {k: v for k, v in meta.items() if k not in ("format", "created", "saved")})
    return new


def delete_world(folder: str | Path) -> None:
    """Delete a saved-world folder (only a folder that really is one)."""
    folder = Path(folder)
    if read_meta(folder) is None:
        raise ValueError("не папка сохранённого мира: ничего не удалено")
    shutil.rmtree(folder)
