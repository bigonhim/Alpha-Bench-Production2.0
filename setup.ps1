# One-time setup: creates a virtual environment outside OneDrive (so it isn't synced) and installs Alpha Bench.
$ErrorActionPreference = "Stop"
$venv = Join-Path $env:LOCALAPPDATA "AlphaBench\venv"
if (-not (Test-Path (Join-Path $venv "Scripts\python.exe"))) {
    Write-Host "Creating virtual environment in $venv"
    New-Item -ItemType Directory -Force (Split-Path $venv) | Out-Null
    python -m venv $venv
}
$py = Join-Path $venv "Scripts\python.exe"
& $py -m pip install --upgrade pip
& $py -m pip install -e "$PSScriptRoot[dev]"
Write-Host "`nDone. Start Alpha Bench with .\run.ps1"
