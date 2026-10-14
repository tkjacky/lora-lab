"""Shared loading and formatting for the LoRA sample.

The training recipe is:

1. Render each conversation with the model's own chat template.
2. Freeze the base weights. In the 4B config they are stored in 4-bit.
3. Attach a pair of small low-rank matrices to every attention and MLP
   projection. Those matrices are the only tensors that receive gradients.
4. Compute the loss on the assistant reply, and ignore the prompt tokens.
"""

from __future__ import annotations

import json
from pathlib import Path

import torch
import yaml
from datasets import Dataset
from peft import LoraConfig, TaskType, prepare_model_for_kbit_training
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def resolve(path: str | Path) -> Path:
    candidate = Path(path)
    if candidate.is_absolute():
        return candidate
    return repo_root() / candidate


def load_config(path: str | Path) -> dict:
    config_path = resolve(path)
    with config_path.open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise SystemExit(f"{config_path} did not contain a YAML mapping.")
    required = ("model_id", "train_file", "eval_file", "output_dir", "target_modules")
    missing = [key for key in required if key not in config]
    if missing:
        raise SystemExit(f"{config_path} is missing keys: {', '.join(missing)}")
    config["_config_path"] = str(config_path)
    return config


def load_jsonl(path: str | Path) -> list[dict]:
    file_path = resolve(path)
    rows: list[dict] = []
    with file_path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            text = line.strip()
            if not text:
                continue
            try:
                row = json.loads(text)
            except json.JSONDecodeError as exc:
                raise SystemExit(f"{file_path}:{line_number} is not valid JSON: {exc}") from exc
            if "messages" not in row or not isinstance(row["messages"], list):
                raise SystemExit(f"{file_path}:{line_number} needs a messages list.")
            rows.append(row)
    if not rows:
        raise SystemExit(f"{file_path} has no rows.")
    return rows


def load_tokenizer(model_id: str):
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    # Right padding is the correct side while the trainer builds loss labels.
    tokenizer.padding_side = "right"
    return tokenizer


def render_prompt(tokenizer, messages: list[dict]) -> str:
    """Prompt the model the same way inference will, with thinking switched off.

    Qwen3 chat templates take enable_thinking. Templates that have no such
    flag raise, and this helper retries without it. The returned string is
    the prefix the model sees before it writes the answer.
    """
    kwargs = {"tokenize": False, "add_generation_prompt": True}
    try:
        rendered = tokenizer.apply_chat_template(
            messages, enable_thinking=False, **kwargs
        )
    except Exception:
        # Templates that do not define enable_thinking reject the flag.
        # The non-thinking instruct checkpoints are in that group.
        rendered = tokenizer.apply_chat_template(messages, **kwargs)
    if not isinstance(rendered, str):
        raise SystemExit(
            "apply_chat_template did not return a string. "
            "Pass tokenize=False and a single conversation."
        )
    return rendered


def format_rows(rows: list[dict], tokenizer) -> Dataset:
    """Turn chat rows into prompt/completion pairs.

    TRL trains only on the completion when a dataset has these two columns.
    The prompt still conditions the model; it just does not contribute loss.
    That matters: asking the model to predict the user's question as well as
    the answer wastes the update and teaches it to continue the prompt.
    """
    formatted = []
    end = (tokenizer.eos_token or "<|im_end|>") + "\n"
    for index, row in enumerate(rows):
        messages = row["messages"]
        if len(messages) < 2 or messages[-1].get("role") != "assistant":
            raise SystemExit(
                f"Row {index} must end with an assistant message. "
                "The text of that message is the only thing the loss sees."
            )
        prompt = render_prompt(tokenizer, messages[:-1])
        answer = str(messages[-1].get("content", "")).strip()
        if not answer:
            raise SystemExit(f"Row {index} has an empty assistant message.")
        formatted.append({"prompt": prompt, "completion": answer + end})
    return Dataset.from_list(formatted)


def _from_pretrained(model_id: str, **kwargs):
    """Load a text model across the dtype and attention-argument renames.

    Transformers 5 takes ``dtype``. Transformers 4 takes ``torch_dtype``.
    SDPA is the attention implementation shipped with PyTorch, so the sample
    does not need a separate FlashAttention install.
    """
    attempts = [dict(kwargs)]
    if "dtype" in kwargs:
        legacy = dict(kwargs)
        legacy["torch_dtype"] = legacy.pop("dtype")
        attempts.append(legacy)
    expanded = []
    for attempt in attempts:
        expanded.append({**attempt, "attn_implementation": "sdpa"})
        expanded.append(attempt)
    last_error: Exception | None = None
    for attempt in expanded:
        try:
            return AutoModelForCausalLM.from_pretrained(model_id, **attempt)
        except (TypeError, ValueError) as exc:
            text = str(exc).lower()
            argument_mismatch = any(
                token in text
                for token in (
                    "unexpected keyword",
                    "dtype",
                    "torch_dtype",
                    "attn_implementation",
                    "attention implementation",
                )
            )
            if not argument_mismatch:
                raise
            last_error = exc
    assert last_error is not None
    raise last_error


def load_model_for_training(config: dict):
    if not torch.cuda.is_available():
        raise SystemExit(
            "CUDA is not available in this Python. Training needs a PC with an "
            "NVIDIA GPU. On that machine, run scripts/setup_5080.sh (or the "
            "PowerShell script) and then python -m src.check_gpu."
        )

    dtype = torch.bfloat16
    kwargs = {"dtype": dtype}
    if config.get("load_in_4bit"):
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=dtype,
            bnb_4bit_use_double_quant=True,
        )
        # bitsandbytes expects the quantized weights to be placed by device_map.
        kwargs["device_map"] = "auto"
    else:
        kwargs["device_map"] = {"": 0}

    model = _from_pretrained(config["model_id"], **kwargs)
    model.config.use_cache = False

    if config.get("load_in_4bit"):
        # Casts layer-norms up to fp32 and turns on gradient checkpointing,
        # both of which QLoRA needs in order to train a 4-bit base.
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    elif config.get("gradient_checkpointing", True):
        model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
    return model


def load_model_for_inference(config: dict):
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is not available. Run inference on a PC with an NVIDIA GPU.")
    dtype = torch.bfloat16
    kwargs = {"dtype": dtype, "device_map": "auto"}
    if config.get("load_in_4bit"):
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=dtype,
            bnb_4bit_use_double_quant=True,
        )
    model = _from_pretrained(config["model_id"], **kwargs)
    model.config.use_cache = True
    model.eval()
    return model


def build_lora_config(config: dict) -> LoraConfig:
    return LoraConfig(
        r=int(config.get("lora_r", 16)),
        lora_alpha=int(config.get("lora_alpha", 32)),
        lora_dropout=float(config.get("lora_dropout", 0.05)),
        bias="none",
        task_type=TaskType.CAUSAL_LM,
        target_modules=list(config["target_modules"]),
    )


def filter_trainer_kwargs(config_cls, kwargs: dict) -> dict:
    """Drop arguments this install of TRL does not recognize.

    SFTConfig has renamed a few fields (max_seq_length -> max_length,
    evaluation_strategy -> eval_strategy). Keeping the lesson script runnable
    across that rename is worth a few lines.
    """
    fields = set(getattr(config_cls, "__dataclass_fields__", {}))
    if not fields:
        return kwargs
    aliases = {
        "eval_strategy": "evaluation_strategy",
        "max_length": "max_seq_length",
    }
    accepted = {}
    for key, value in kwargs.items():
        if key in fields:
            accepted[key] = value
        elif aliases.get(key) in fields:
            accepted[aliases[key]] = value
        else:
            print(f"note: this TRL build has no SFTConfig.{key}; leaving it out")
    return accepted
