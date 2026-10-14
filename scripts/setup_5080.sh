#!/usr/bin/env bash
# Create a virtualenv on the RTX 5080 PC and install a CUDA build of PyTorch
# plus the libraries this sample imports. Run it from any directory:
#   bash scripts/setup_5080.sh

set -euo pipefail
cd "$(dirname "$0")/.."

if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "nvidia-smi was not found."
  echo "Run this script on the PC with the NVIDIA GPU, after the driver is installed."
  exit 1
fi

nvidia-smi
echo

pick_python() {
  local candidate version major minor
  for candidate in python3.12 python3.11 python3.13 python3; do
    if ! command -v "$candidate" >/dev/null 2>&1; then
      continue
    fi
    version="$("$candidate" -c 'import sys; print(f"{sys.version_info[0]}.{sys.version_info[1]}")')"
    major="${version%%.*}"
    minor="${version#*.}"
    if [[ "$major" == "3" && "$minor" -ge 11 && "$minor" -le 13 ]]; then
      echo "$candidate"
      return 0
    fi
  done
  return 1
}

if ! PYTHON_BIN="$(pick_python)"; then
  echo "Install Python 3.12 (3.11 or 3.13 also work)."
  echo "Python 3.14 is new enough that some CUDA wheels still lag behind it."
  exit 1
fi

echo "Using $PYTHON_BIN ($("$PYTHON_BIN" --version))"
"$PYTHON_BIN" -m venv .venv
# shellcheck disable=SC1091
source .venv/bin/activate

python -m pip install -U pip
# Late-2026 PyPI torch wheels are CUDA 13 builds and include sm_120.
# If check_gpu then says the GPU has no matching kernel, reinstall with:
#   python -m pip install --force-reinstall torch --index-url https://download.pytorch.org/whl/cu128
python -m pip install torch
python -m pip install --upgrade-strategy only-if-needed -r requirements.txt

echo
echo "Virtualenv is ready. In every new shell, from the repo root, run:"
echo "  source .venv/bin/activate"
echo

python -m src.check_gpu
