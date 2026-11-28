# PyInstaller spec of the Windows application (one folder, no console window).
# Build from the repository root:  pyinstaller packaging/windows/hexworld.spec --noconfirm
# Needs: pmge/build/Release/pmge.dll and packaging/windows/build/hexworld.ico (make_icon.py).
import os

ROOT = os.path.abspath(".")
ICON = os.path.join(ROOT, "packaging", "windows", "build", "hexworld.ico")
PMGE = [
    p
    for p in (
        os.path.join(ROOT, "pmge", "build", "Release", "pmge.dll"),
        os.path.join(ROOT, "pmge", "build", "pmge.dll"),
    )
    if os.path.isfile(p)
]

# TZ 5.5.1 catalogues (commodities, deposit models, reference profile) are required data
DATAS = [(os.path.join(ROOT, "src", "hexworld", "minerals", "data"),
          os.path.join("hexworld", "minerals", "data"))]
if os.path.isdir(os.path.join(ROOT, "docs", "biomes", "presets")):
    DATAS.append((os.path.join(ROOT, "docs", "biomes", "presets"),
                  os.path.join("docs", "biomes", "presets")))

a = Analysis(
    [os.path.join(ROOT, "packaging", "windows", "hexworld_app.py")],
    pathex=[os.path.join(ROOT, "src")],
    binaries=[(PMGE[0], ".")] if PMGE else [],
    datas=DATAS,
    hiddenimports=[
        "hexworld.gui.app",
        "hexworld.gui.world_settings",
        "hexworld.gui.worlds_dialogs",
        "hexworld.gui.hex_panel",
        "scipy.sparse.linalg",
        "scipy.ndimage",
        "scipy.spatial",
        "scipy.stats",
    ],
    excludes=[
        "tkinter",
        "matplotlib",
        "IPython",
        "pytest",
        "PySide6.QtWebEngineCore",
        "PySide6.QtWebEngineWidgets",
        "PySide6.Qt3DCore",
        "PySide6.QtMultimedia",
        "PySide6.QtQuick",
        "PySide6.QtQml",
        "PySide6.QtPdf",
        "PySide6.QtCharts",
        "PySide6.QtDataVisualization",
    ],
    noarchive=False,
)
pyz = PYZ(a.pure)
exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="HexWorld",
    console=False,
    icon=ICON if os.path.isfile(ICON) else None,
    version=os.path.join(ROOT, "packaging", "windows", "build", "version_info.txt")
    if os.path.isfile(os.path.join(ROOT, "packaging", "windows", "build", "version_info.txt"))
    else None,
)
coll = COLLECT(exe, a.binaries, a.datas, name="HexWorld")
