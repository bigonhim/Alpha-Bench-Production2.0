# Start Alpha Bench and open it in the default browser. Extra arguments are passed through (e.g. -- --no-keyring).
$py = Join-Path $env:LOCALAPPDATA "AlphaBench\venv\Scripts\python.exe"
if (-not (Test-Path $py)) { Write-Host "Run .\setup.ps1 first."; exit 1 }
Set-Location $PSScriptRoot
& $py -m alpha_bench @args
