@echo off
set "PROJECT_ROOT=%~dp0.."
if not exist "%PROJECT_ROOT%\venv\Scripts\python.exe" (
  echo venv is missing. Run scripts\setup_venv.bat first.
  exit /b 1
)
cd /d "%PROJECT_ROOT%"
"%PROJECT_ROOT%\venv\Scripts\python.exe" -m browzy.backend
