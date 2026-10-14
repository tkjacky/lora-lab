# Create a virtualenv on the RTX 5080 PC and install a CUDA build of PyTorch
# plus the libraries this sample imports.
#
#   powershell -ExecutionPolicy Bypass -File scripts\setup_5080.ps1

$ErrorActionPreference = "Stop"
Set-Location (Split-Path -Parent $PSScriptRoot)

if (-not (Get-Command nvidia-smi -ErrorAction SilentlyContinue)) {
    Write-Host "nvidia-smi was not found."
    Write-Host "Run this script on the PC with the NVIDIA GPU, after the driver is installed."
    exit 1
}

nvidia-smi
Write-Host ""

$python = $null
foreach ($version in @("3.12", "3.11", "3.13")) {
    & py "-$version" -c "import sys" 2>$null
    if ($LASTEXITCODE -eq 0) {
        $python = $version
        break
    }
}

if (-not $python) {
    Write-Host "Install Python 3.12 from https://www.python.org/downloads/ and enable the py launcher."
    exit 1
}

Write-Host "Using py -$python"
& py "-$python" -m venv .venv
$venvPython = Join-Path (Get-Location) ".venv\Scripts\python.exe"

& $venvPython -m pip install -U pip
# Late-2026 PyPI torch wheels are CUDA 13 builds and include sm_120.
# If check_gpu then says the GPU has no matching kernel, reinstall with:
#   .\.venv\Scripts\python.exe -m pip install --force-reinstall torch --index-url https://download.pytorch.org/whl/cu128
& $venvPython -m pip install torch
& $venvPython -m pip install --upgrade-strategy only-if-needed -r requirements.txt

Write-Host ""
Write-Host "Virtualenv is ready. In every new shell, from the repo root, run:"
Write-Host "  .\.venv\Scripts\Activate.ps1"
Write-Host ""

& $venvPython -m src.check_gpu
