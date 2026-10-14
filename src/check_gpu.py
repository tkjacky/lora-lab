"""Check that this Python can actually train on the installed NVIDIA GPU.

    python -m src.check_gpu

Run it on the NVIDIA PC after setup. A passing run means the driver, the
PyTorch build, and the 4-bit kernels agree with each other.
"""

from __future__ import annotations

import subprocess
import sys


def run_nvidia_smi() -> str:
    try:
        completed = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name,driver_version,memory.total",
                "--format=csv,noheader",
            ],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        print("nvidia-smi failed. Install the NVIDIA driver, then open a new shell.")
        print(exc)
        return ""
    print("nvidia-smi:")
    print(completed.stdout.strip())
    return completed.stdout


def driver_major(smi_output: str) -> int | None:
    # csv: "NVIDIA GeForce RTX 5080, 570.86, 16303 MiB"
    for line in smi_output.splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) < 2:
            continue
        try:
            return int(parts[1].split(".", 1)[0])
        except ValueError:
            continue
    return None


def main() -> int:
    smi_output = run_nvidia_smi()
    major = driver_major(smi_output)

    try:
        import torch
    except ImportError:
        print("\nPyTorch is not installed. Run scripts/setup_5080.sh or setup_5080.ps1.")
        return 1

    print(f"\ntorch {torch.__version__}    cuda build: {torch.version.cuda}")
    if not torch.cuda.is_available():
        print("torch.cuda.is_available() is false.")
        print("This interpreter is a CPU build, or the driver is not visible to it.")
        print("Reinstall PyTorch from a CUDA 12.8 or CUDA 13 wheel, then rerun this check.")
        return 1

    device = torch.cuda.get_device_properties(0)
    capability = torch.cuda.get_device_capability(0)
    print(f"device: {device.name}")
    print(f"compute capability: sm_{capability[0]}{capability[1]}")
    print(f"memory: {device.total_memory / (1024 ** 3):.1f} GB")

    if capability[0] >= 12 and major is not None and major < 570:
        print(
            f"\nDriver branch {major} is older than 570. An RTX 50-series GPU needs a "
            "570 or newer driver (580 or newer if PyTorch was built for CUDA 13)."
        )

    if capability[0] >= 12 and torch.version.cuda:
        cuda_parts = [int(part) for part in str(torch.version.cuda).split(".")[:2]]
        cuda_version = (cuda_parts[0], cuda_parts[1] if len(cuda_parts) > 1 else 0)
        if cuda_version < (12, 8):
            print(
                "This PyTorch wheel was built with CUDA older than 12.8, so it "
                "has no native RTX 50-series kernels. Reinstall torch."
            )
            return 1

    try:
        left = torch.randn(256, 256, device="cuda", dtype=torch.bfloat16)
        product = left @ left
        torch.cuda.synchronize()
    except Exception as exc:
        print("\nbf16 matmul FAILED. The wheel does not match this GPU.")
        print(exc)
        return 1
    print(f"bf16 matmul ok ({tuple(product.shape)})")

    try:
        from bitsandbytes.nn import Linear4bit

        layer = Linear4bit(
            32,
            32,
            bias=False,
            compute_dtype=torch.bfloat16,
            quant_type="nf4",
        ).to("cuda")
        batch = torch.randn(2, 32, device="cuda", dtype=torch.bfloat16, requires_grad=True)
        loss = layer(batch).float().pow(2).mean()
        loss.backward()
        torch.cuda.synchronize()
    except Exception as exc:
        print("\n4-bit forward/backward FAILED.")
        print("QLoRA needs this. The smoke config (bf16 LoRA on Qwen3-0.6B) can")
        print("still run. Fix bitsandbytes before configs/qwen3-4b-qlora.yaml.")
        print(exc)
        return 1

    print("4-bit linear forward + backward ok")
    print("\nGPU stack looks ready.")
    print("Next:  python -m src.train --config configs/smoke.yaml --max-steps 5")
    return 0


if __name__ == "__main__":
    sys.exit(main())
