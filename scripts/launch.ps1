# One-command launcher. Safe to run any number of times: a second copy exits at once.
# Usage (from the momento folder):   powershell -ExecutionPolicy Bypass -File scripts\launch.ps1
$ErrorActionPreference = "Continue"
Set-Location (Split-Path $PSScriptRoot -Parent)
$py = Join-Path $PWD ".venv\Scripts\python.exe"
if (-not (Test-Path $py)) { Write-Host "venv not found: $py" -ForegroundColor Red; exit 1 }
New-Item -ItemType Directory -Force -Path logs | Out-Null
$log = Join-Path $PWD ("logs\plan_" + (Get-Date -Format "yyyyMMdd_HHmmss") + ".log")
Write-Host "Launching MEMENTO plan with $py" -ForegroundColor Cyan
Write-Host "Log: $log" -ForegroundColor Cyan
& $py -u scripts\run_plan.py --seeds 0 --blocks E3 E1 E2 E7 --commit 2>&1 | Tee-Object -FilePath $log
