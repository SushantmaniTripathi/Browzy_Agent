$ErrorActionPreference = "Stop"

$ProjectRoot = Split-Path -Parent $PSScriptRoot
$VenvPath = Join-Path $ProjectRoot "venv"
$PythonPath = Join-Path $VenvPath "Scripts\python.exe"
$RequirementsPath = Join-Path $ProjectRoot "browzy\backend\requirements.txt"

if (-not (Get-Command python -ErrorAction SilentlyContinue)) {
    throw "Python was not found on PATH. Install Python 3.12+ and try again."
}

if (-not (Test-Path $PythonPath)) {
    Write-Host "Creating virtual environment at $VenvPath"
    python -m venv $VenvPath
}

Write-Host "Upgrading pip"
& $PythonPath -m pip install --upgrade pip

Write-Host "Installing Browzy dependencies"
& $PythonPath -m pip install -r $RequirementsPath

Write-Host "Verifying backend import"
& $PythonPath -c "import fastapi, httpx, playwright, pydantic; import browzy.backend.backend; print('Browzy environment is ready.')"

Write-Host ""
Write-Host "Activate this environment with:"
Write-Host "  .\venv\Scripts\Activate.ps1"
