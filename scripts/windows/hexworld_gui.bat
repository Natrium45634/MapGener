@echo off
rem HexWorld viewer on Windows. Run scripts\windows\install.ps1 once before.
cd /d "%~dp0\..\.."
if not exist ".venv\Scripts\python.exe" (
  echo HexWorld is not installed yet: run  powershell -ExecutionPolicy Bypass -File scripts\windows\install.ps1
  pause
  exit /b 1
)
".venv\Scripts\python.exe" -m hexworld gui %*
