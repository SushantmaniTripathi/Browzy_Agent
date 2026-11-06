@echo off
set "PROJECT_ROOT=%~dp0.."
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0setup_venv.ps1"
if errorlevel 1 exit /b %errorlevel%
echo.
echo Activate with: "%PROJECT_ROOT%\venv\Scripts\activate.bat"
