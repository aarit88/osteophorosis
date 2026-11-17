"""
OsteoGuard AI - Stage 08A CUDA Environment and GPU Verification Script.

Executes:
  1. CUDA Driver & Device Introspection (torch.cuda.is_available, device properties, VRAM info).
  2. Basic Tensor Operations on GPU (allocation, matmul, backward, cache release).
  3. ResNet-101 Medical Vision Forward/Backward Smoke Test on CUDA (batch_size=2, 224x224).
  4. Automatic Mixed Precision (AMP FP16) Verification.
  5. Peak VRAM Footprint Profiling.
"""

import sys
from typing import Any, Dict
import torch
import torch.nn as nn
import torchvision.models as models


def verify_cuda_environment() -> Dict[str, Any]:
    """Inspects and returns CUDA device properties."""
    info = {
        "python_version": sys.version,
        "torch_version": torch.__version__,
        "torch_cuda_version": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
    }

    if not torch.cuda.is_available():
        info["error"] = "CUDA is NOT available in this PyTorch installation."
        return info

    dev_count = torch.cuda.device_count()
    dev_name = torch.cuda.get_device_name(0)
    dev_props = torch.cuda.get_device_properties(0)
    free_mem, total_mem = torch.cuda.mem_get_info(0)

    info["device_count"] = dev_count
    info["device_name"] = dev_name
    info["compute_capability"] = f"{dev_props.major}.{dev_props.minor}"
    info["total_vram_bytes"] = dev_props.total_memory
    info["total_vram_mb"] = dev_props.total_memory / (1024 ** 2)
    info["free_vram_mb"] = free_mem / (1024 ** 2)
    info["multi_processor_count"] = dev_props.multi_processor_count
    return info


def run_gpu_computation_test() -> Dict[str, Any]:
    """Runs tensor allocation, autograd, cache release, ResNet-101 test, and AMP."""
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available for GPU computation test.")

    device = torch.device("cuda:0")
    torch.cuda.reset_peak_memory_stats(device)
    results = {}

    # 1. Basic Tensor Ops
    a = torch.randn(1024, 1024, device=device, requires_grad=True)
    b = torch.randn(1024, 1024, device=device, requires_grad=True)
    c = torch.matmul(a, b)
    loss = c.sum()
    loss.backward()

    assert a.grad is not None and torch.isfinite(a.grad).all()
    assert b.grad is not None and torch.isfinite(b.grad).all()
    results["basic_tensor_matmul_and_grad"] = True

    del a, b, c, loss
    torch.cuda.empty_cache()
    results["cache_release_successful"] = True

    # 2. ResNet-101 Forward/Backward Smoke Test (batch_size=2, 224x224)
    model = models.resnet101(weights=None)
    model.fc = nn.Linear(2048, 2)
    model.to(device)
    model.train()

    x = torch.randn(2, 3, 224, 224, device=device)
    y = torch.tensor([0, 1], device=device, dtype=torch.long)

    # Standard FP32 forward + backward
    logits = model(x)
    criterion = nn.CrossEntropyLoss()
    loss_val = criterion(logits, y)
    loss_val.backward()

    fc_grad = model.fc.weight.grad
    assert fc_grad is not None and torch.isfinite(fc_grad).all()
    results["resnet101_fp32_forward_backward"] = {
        "logits_shape": list(logits.shape),
        "loss": float(loss_val.item()),
    }

    # 3. Automatic Mixed Precision (AMP FP16)
    scaler = torch.amp.GradScaler("cuda")
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    optimizer.zero_grad()

    with torch.amp.autocast("cuda", dtype=torch.float16):
        logits_amp = model(x)
        loss_amp = criterion(logits_amp, y)

    scaler.scale(loss_amp).backward()
    scaler.step(optimizer)
    scaler.update()

    assert torch.isfinite(logits_amp).all()
    results["resnet101_amp_fp16_verified"] = {
        "amp_loss": float(loss_amp.item()),
    }

    # 4. Memory Profiling
    peak_vram_mb = torch.cuda.max_memory_allocated(device) / (1024 ** 2)
    results["peak_vram_allocated_mb"] = float(peak_vram_mb)

    # Clean up
    del model, x, y, logits, loss_val, logits_amp, loss_amp, optimizer, scaler
    torch.cuda.empty_cache()

    results["final_allocated_vram_mb"] = float(torch.cuda.memory_allocated(device) / (1024 ** 2))
    results["status"] = "PASS"
    return results


def main():
    print("=" * 60)
    print("OsteoGuard AI - Stage 08A CUDA Verification")
    print("=" * 60)

    env_info = verify_cuda_environment()
    for k, v in env_info.items():
        print(f"  {k}: {v}")

    if not env_info.get("cuda_available", False):
        print("\nCRITICAL: CUDA is not available. Verification BLOCKED.")
        sys.exit(1)

    print("\nExecuting GPU Computation & ResNet-101 Smoke Test...")
    comp_results = run_gpu_computation_test()
    for k, v in comp_results.items():
        print(f"  {k}: {v}")

    print(f"\nOverall GPU Verification: {comp_results['status']}")


if __name__ == "__main__":
    main()

