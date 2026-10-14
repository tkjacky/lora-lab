"""Compare the base model with the trained LoRA adapter on the same prompts.

    python -m src.infer --config configs/smoke.yaml
"""

from __future__ import annotations

import argparse

import torch
from peft import PeftModel

if __package__ in (None, ""):
    raise SystemExit(
        "Run this from the repo root:\n"
        "  python -m src.infer --config configs/smoke.yaml"
    )

from .common import (  # noqa: E402
    load_config,
    load_jsonl,
    load_model_for_inference,
    load_tokenizer,
    render_prompt,
    resolve,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Print base vs LoRA answers.")
    parser.add_argument("--config", required=True)
    parser.add_argument(
        "--adapter",
        default=None,
        help="Adapter directory. Defaults to <output_dir>/adapter from the config.",
    )
    parser.add_argument(
        "--prompts",
        default="data/prompts.jsonl",
        help="JSONL file of conversations without an assistant turn.",
    )
    parser.add_argument("--max-new-tokens", type=int, default=160)
    return parser.parse_args()


def part_line(text: str) -> str:
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.upper().startswith("PART:"):
            return stripped.split(":", 1)[1].strip()
    return ""


@torch.inference_mode()
def generate(model, tokenizer, messages: list[dict], max_new_tokens: int) -> str:
    prompt = render_prompt(tokenizer, messages)
    inputs = tokenizer(prompt, return_tensors="pt")
    inputs = {key: value.to(model.device) for key, value in inputs.items()}
    output = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    new_tokens = output[0, inputs["input_ids"].shape[1] :]
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


def main():
    args = parse_args()
    config = load_config(args.config)
    adapter_dir = resolve(args.adapter) if args.adapter else resolve(config["output_dir"]) / "adapter"
    if not (adapter_dir / "adapter_config.json").exists():
        raise SystemExit(
            f"No adapter at {adapter_dir}. Train first:\n"
            f"  python -m src.train --config {args.config}"
        )

    prompts = load_jsonl(args.prompts)
    tokenizer = load_tokenizer(config["model_id"])
    tokenizer.padding_side = "left"
    model = load_model_for_inference(config)

    print(f"base model: {config['model_id']}")
    base_answers = [
        generate(model, tokenizer, row["messages"], args.max_new_tokens) for row in prompts
    ]

    model = PeftModel.from_pretrained(model, adapter_dir)
    model.eval()
    print(f"adapter:    {adapter_dir}\n")
    lora_answers = [
        generate(model, tokenizer, row["messages"], args.max_new_tokens) for row in prompts
    ]

    for row, base_answer, lora_answer in zip(prompts, base_answers, lora_answers):
        question = next(
            (message["content"] for message in reversed(row["messages"]) if message["role"] == "user"),
            "",
        )
        expected = str(row.get("expect", "")).strip()
        print("=" * 72)
        print(row.get("tag", "prompt"))
        print(f"Q: {question}")
        if expected:
            print("\nCatalog card this prompt was written for:")
            print(expected)
            print(
                "PART line: "
                + (
                    "adapter matches"
                    if part_line(lora_answer) and part_line(lora_answer) == part_line(expected)
                    else "adapter differs"
                )
            )
        print("\nBase model:")
        print(base_answer or "(empty)")
        print("\nBase + LoRA:")
        print(lora_answer or "(empty)")
        print()


if __name__ == "__main__":
    main()
