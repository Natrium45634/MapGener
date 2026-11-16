# HexWorld: one-time setup on Windows (PowerShell).
#   powershell -ExecutionPolicy Bypass -File scripts\windows\install.ps1
# Creates .venv, installs HexWorld with the viewer, and builds PMGE if CMake and a C++
# compiler (Visual Studio Build Tools) are available. Without PMGE HexWorld still works
# with its built-in mountain module.
$ErrorActionPreference = "Stop"
Set-Location (Split-Path -Parent (Split-Path -Parent $PSScriptRoot))

$py = Get-Command py -ErrorAction SilentlyContinue
if ($py) { & py -3 -m venv .venv } else { & python -m venv .venv }
& .venv\Scripts\python.exe -m pip install --upgrade pip
& .venv\Scripts\python.exe -m pip install -e ".[gui,dev]"

if (Get-Command cmake -ErrorAction SilentlyContinue) {
    Write-Host "Building PMGE (pmge\build\Release\pmge.dll)..."
    cmake -S pmge -B pmge\build
    cmake --build pmge\build --config Release --parallel
    if ($LASTEXITCODE -ne 0) { Write-Warning "PMGE build failed: the built-in mountain module will be used." }
} else {
    Write-Warning "CMake not found: PMGE is not built, the built-in mountain module will be used."
}
Write-Host "Done. Start the viewer with scripts\windows\hexworld_gui.bat"
