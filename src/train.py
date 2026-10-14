"""Fine-tune a Qwen3 model with LoRA.

Run from the repo root, on a PC with an NVIDIA GPU:

    python -m src.train --config configs/smoke.yaml
    python -m src.train --config configs/qwen3-4b-qlora.yaml
"""

from __future__ import annotations

import argparse
import shutil

from trl import SFTConfig, SFTTrainer

if __package__ in (None, ""):
    raise SystemExit(
        "Run this from the repo root:\n"
        "  python -m src.train --config configs/smoke.yaml"
    )

from .common import (  # noqa: E402
    build_lora_config,
    filter_trainer_kwargs,
    format_rows,
    load_config,
    load_jsonl,
    load_model_for_training,
    load_tokenizer,
    resolve,
)


def parse_args():
    parser = argparse.ArgumentParser(description="LoRA fine-tune the model in a YAML config.")
    parser.add_argument("--config", required=True, help="Path to a file in configs/.")
    parser.add_argument(
        "--max-steps",
        type=int,
        default=None,
        help="Override max_steps. Use 5 for a plumbing check.",
    )
    return parser.parse_args()


def count_parameters(model) -> tuple[int, int]:
    trainable = 0
    total = 0
    for param in model.parameters():
        count = param.numel()
        total += count
        if param.requires_grad:
            trainable += count
    return trainable, total


def main():
    args = parse_args()
    config = load_config(args.config)
    if args.max_steps is not None:
        config["max_steps"] = args.max_steps

    output_dir = resolve(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy(config["_config_path"], output_dir / "config.yaml")

    print(f"model:   {config['model_id']}")
    print(f"4-bit:   {bool(config.get('load_in_4bit'))}")
    print(f"output:  {output_dir}")

    tokenizer = load_tokenizer(config["model_id"])
    train_dataset = format_rows(load_jsonl(config["train_file"]), tokenizer)
    eval_dataset = format_rows(load_jsonl(config["eval_file"]), tokenizer)

    preview = train_dataset[0]
    print("\n--- one training example, loss is on the completion only ---")
    print(preview["prompt"] + preview["completion"])
    print("--- end example ---\n")

    model = load_model_for_training(config)
    lora = build_lora_config(config)

    trainer_fields = {
            "output_dir": str(output_dir / "checkpoints"),
            "num_train_epochs": float(config.get("num_train_epochs", 3)),
            "max_length": int(config.get("max_length", 512)),
            "per_device_train_batch_size": int(config.get("per_device_train_batch_size", 1)),
            "per_device_eval_batch_size": 1,
            "gradient_accumulation_steps": int(config.get("gradient_accumulation_steps", 4)),
            "learning_rate": float(config.get("learning_rate", 2e-4)),
            "warmup_ratio": float(config.get("warmup_ratio", 0.05)),
            "weight_decay": float(config.get("weight_decay", 0.0)),
            "lr_scheduler_type": config.get("lr_scheduler_type", "cosine"),
            "optim": config.get("optim", "adamw_torch"),
            "bf16": bool(config.get("bf16", True)),
            "fp16": False,
            "gradient_checkpointing": bool(config.get("gradient_checkpointing", True)),
            "gradient_checkpointing_kwargs": {"use_reentrant": False},
            "logging_steps": int(config.get("logging_steps", 1)),
            "eval_strategy": "steps",
            "eval_steps": int(config.get("eval_steps", 10)),
            "save_strategy": "steps",
            "save_steps": int(config.get("save_steps", 20)),
            "save_total_limit": 2,
            "packing": False,
            "completion_only_loss": True,
            "dataset_num_proc": 1,
            "dataloader_num_workers": 0,
            "report_to": "none",
            "seed": int(config.get("seed", 42)),
            "remove_unused_columns": True,
    }
    max_steps = int(config.get("max_steps") or 0)
    if max_steps > 0:
        trainer_fields["max_steps"] = max_steps
    trainer_kwargs = filter_trainer_kwargs(SFTConfig, trainer_fields)
    trainer = _make_trainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        peft_config=lora,
        args=SFTConfig(**trainer_kwargs),
    )

    if hasattr(trainer.model, "print_trainable_parameters"):
        trainer.model.print_trainable_parameters()
    else:
        trainable, total = count_parameters(trainer.model)
        share = 100.0 * trainable / total if total else 0.0
        print(f"trainable parameters: {trainable:,} / {total:,} ({share:.3f}%)")
    print("The other parameters stay frozen at their Qwen values.\n")

    trainer.train()

    adapter_dir = output_dir / "adapter"
    model_to_save = trainer.model
    if hasattr(model_to_save, "module"):
        model_to_save = model_to_save.module
    model_to_save.save_pretrained(adapter_dir)
    tokenizer.save_pretrained(adapter_dir)
    print(f"\nSaved the LoRA adapter to {adapter_dir}")
    print("Next: python -m src.infer --config", args.config)


def _make_trainer(*, model, tokenizer, train_dataset, eval_dataset, peft_config, args):
    import inspect

    kwargs = {
        "model": model,
        "args": args,
        "train_dataset": train_dataset,
        "eval_dataset": eval_dataset,
        "peft_config": peft_config,
    }
    parameters = inspect.signature(SFTTrainer.__init__).parameters
    if "processing_class" in parameters:
        kwargs["processing_class"] = tokenizer
    else:
        kwargs["tokenizer"] = tokenizer
    return SFTTrainer(**kwargs)


if __name__ == "__main__":
    main()
