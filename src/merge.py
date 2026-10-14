"""Fold the LoRA matrices into a normal bf16 model folder.

The adapter saved by train.py is only useful together with the original base
model. Merging writes a standalone Hugging Face folder you can convert later
(for example to GGUF). Merge loads the original bf16 weights, not the 4-bit
copy used during QLoRA training.

    python -m src.merge --config configs/smoke.yaml

A 4B merge needs about 16 GB of system RAM and writes about 8 GB of weights.
"""

from __future__ import annotations

import argparse

import torch
from peft import PeftModel

if __package__ in (None, ""):
    raise SystemExit(
        "Run this from the repo root:\n"
        "  python -m src.merge --config configs/smoke.yaml"
    )

from .common import _from_pretrained, load_config, load_tokenizer, resolve  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser(description="Merge a LoRA adapter into bf16 weights.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--adapter", default=None)
    parser.add_argument("--out", default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    config = load_config(args.config)
    adapter_dir = resolve(args.adapter) if args.adapter else resolve(config["output_dir"]) / "adapter"
    out_dir = resolve(args.out) if args.out else resolve(config["output_dir"]) / "merged"
    if not (adapter_dir / "adapter_config.json").exists():
        raise SystemExit(f"No adapter at {adapter_dir}. Train first.")

    print(f"Loading {config['model_id']} in bf16 on CPU, then applying {adapter_dir}")
    model = _from_pretrained(
        config["model_id"],
        dtype=torch.bfloat16,
        device_map="cpu",
        low_cpu_mem_usage=True,
    )
    model = PeftModel.from_pretrained(model, adapter_dir)
    merged = model.merge_and_unload()
    out_dir.mkdir(parents=True, exist_ok=True)
    merged.save_pretrained(out_dir, safe_serialization=True)
    load_tokenizer(config["model_id"]).save_pretrained(out_dir)
    print(f"Wrote a standalone model to {out_dir}")


if __name__ == "__main__":
    main()
