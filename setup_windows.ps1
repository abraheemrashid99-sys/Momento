# MEMENTO — one-shot setup for a fresh Windows machine with an NVIDIA RTX 50-series GPU.
# Run from the repo folder in PowerShell:   powershell -ExecutionPolicy Bypass -File .\setup_windows.ps1
$ErrorActionPreference = "Continue"
function Have($cmd) { return [bool](Get-Command $cmd -ErrorAction SilentlyContinue) }

Write-Host "== 1/6 Git" -ForegroundColor Cyan
if (-not (Have git)) { winget install --id Git.Git -e --source winget --accept-package-agreements --accept-source-agreements }

Write-Host "== 2/6 Python 3.12" -ForegroundColor Cyan
$py = "$env:LOCALAPPDATA\Programs\Python\Python312\python.exe"
if (-not (Test-Path $py)) { winget install --id Python.Python.3.12 -e --source winget --accept-package-agreements --accept-source-agreements }
if (-not (Test-Path $py)) { $py = (Get-Command python -ErrorAction SilentlyContinue).Source }
if (-not $py) { Write-Host "Python not found after install; open a new terminal and rerun this script." -ForegroundColor Red; exit 1 }
& $py --version

Write-Host "== 3/6 NVIDIA driver check" -ForegroundColor Cyan
if (-not (Have nvidia-smi)) {
  Write-Host "nvidia-smi not found: installing GeForce driver via winget (may need a reboot)" -ForegroundColor Yellow
  winget install --id Nvidia.GeForceExperience -e --source winget --accept-package-agreements --accept-source-agreements
} else { nvidia-smi }

Write-Host "== 4/6 Virtual environment" -ForegroundColor Cyan
if (-not (Test-Path ".venv")) { & $py -m venv .venv }
$venvpy = ".\.venv\Scripts\python.exe"
& $venvpy -m pip install --upgrade pip

Write-Host "== 5/6 PyTorch (CUDA 12.8 build for Blackwell / RTX 50xx) + project" -ForegroundColor Cyan
& $venvpy -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
& $venvpy -m pip install -e ".[dev]"
& $venvpy -c "import torch; print('torch', torch.__version__, '| cuda', torch.cuda.is_available(), '|', (torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NO GPU'))"

Write-Host "== 6/6 Tests" -ForegroundColor Cyan
& $venvpy -m pytest -q

Write-Host ""
Write-Host "Setup finished. Activate with:  .\.venv\Scripts\Activate.ps1" -ForegroundColor Green
Write-Host "Smoke run:  python scripts/run.py configs/smoke/cpu_cifar_mini.yaml attack.name=echo attack.rho=5" -ForegroundColor Green
