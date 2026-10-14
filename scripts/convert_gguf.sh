#!/usr/bin/env bash
# Convert a merged Hugging Face folder to a bf16 GGUF file for Ollama
# and llama.cpp. Merge first:
#   python -m src.merge --config configs/smoke.yaml
#   bash scripts/convert_gguf.sh --config configs/smoke.yaml

set -euo pipefail
cd "$(dirname "$0")/.."

CONFIG="configs/smoke.yaml"
OUTTYPE="bf16"
OUTFILE=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --config)
      CONFIG="$2"
      shift 2
      ;;
    --outtype)
      OUTTYPE="$2"
      shift 2
      ;;
    --outfile)
      OUTFILE="$2"
      shift 2
      ;;
    *)
      echo "Unknown argument: $1"
      echo "Usage: bash scripts/convert_gguf.sh [--config configs/smoke.yaml] [--outtype bf16] [--outfile path.gguf]"
      exit 1
      ;;
  esac
done

if [[ ! -x .venv/bin/python ]]; then
  echo "No .venv. Run bash scripts/setup_5080.sh first."
  exit 1
fi
PYTHON=".venv/bin/python"

if [[ ! -f "$CONFIG" ]]; then
  echo "Config not found: $CONFIG"
  exit 1
fi

OUTPUT_DIR="$("$PYTHON" -c 'import sys, yaml; from pathlib import Path; cfg = yaml.safe_load(Path(sys.argv[1]).read_text()); print(cfg["output_dir"])' "$CONFIG")"
MERGED="$OUTPUT_DIR/merged"
if [[ -z "$OUTFILE" ]]; then
  OUTFILE="$OUTPUT_DIR/model-${OUTTYPE}.gguf"
fi

if [[ ! -f "$MERGED/model.safetensors" ]]; then
  echo "No merged model at $MERGED."
  echo "Run this first:"
  echo "  python -m src.merge --config $CONFIG"
  exit 1
fi

LLAMA_CPP_DIR="${LLAMA_CPP_DIR:-third_party/llama.cpp}"
if [[ ! -f "$LLAMA_CPP_DIR/convert_hf_to_gguf.py" ]]; then
  echo "Cloning llama.cpp into $LLAMA_CPP_DIR"
  mkdir -p "$(dirname "$LLAMA_CPP_DIR")"
  git clone --depth 1 https://github.com/ggml-org/llama.cpp.git "$LLAMA_CPP_DIR"
fi

echo "merged:  $MERGED"
echo "outfile: $OUTFILE"
echo "outtype: $OUTTYPE"

"$PYTHON" "$LLAMA_CPP_DIR/convert_hf_to_gguf.py" "$MERGED" \
  --outfile "$OUTFILE" \
  --outtype "$OUTTYPE"

echo
echo "Wrote $OUTFILE"
echo "Load the file with the project Ollama. Start 'ollama serve' first; see the README."
echo "  export OLLAMA_HOST=http://127.0.0.1:11434"
echo "  third_party/ollama/bin/ollama create northwind-qwen3 -f ollama/Modelfile"
echo "  third_party/ollama/bin/ollama run northwind-qwen3 --think=false \"How many volts does the dock thermistor need?\""
