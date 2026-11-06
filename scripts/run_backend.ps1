$ErrorActionPreference = "Stop"

$ProjectRoot = Split-Path -Parent $PSScriptRoot
$PythonPath = Join-Path $ProjectRoot "venv\Scripts\python.exe"

if (-not (Test-Path $PythonPath)) {
    throw "The venv is missing. Run .\scripts\setup_venv.ps1 first."
}

Set-Location $ProjectRoot
& $PythonPath -m browzy.backend
