"""
OsteoGuard AI - Stage 08A CUDA and GPU Environment Tests.

Verifies:
  1. CUDA runtime availability and device recognition.
  2. Basic CUDA tensor operations and autograd on GPU.
  3. ResNet-101 forward and backward execution on CUDA (batch_size=2, 224x224).
  4. Automatic Mixed Precision (AMP FP16) stability and gradient scaling.
  5. Peak VRAM safety headroom verification for 4 GB VRAM budget.
  6. GPU training configuration YAML integrity.
"""

from pathlib import Path
import pytest
import torch
import torch.nn as nn
import yaml

from ml_service.config.settings import get_settings
from ml_service.models.resnet_classifier import build_resnet101_classifier


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available in current environment")
def test_cuda_device_availability():
    """Verify CUDA is recognized and device properties meet minimum specs."""
    assert torch.cuda.is_available() is True
    assert torch.cuda.device_count() >= 1

    device_name = torch.cuda.get_device_name(0)
    assert "RTX" in device_name or "NVIDIA" in device_name or "GeForce" in device_name

    props = torch.cuda.get_device_properties(0)
    assert props.total_memory > 2 * (1024 ** 3), "Expected at least 2 GB VRAM"
    assert props.major >= 7, f"Compute capability major version {props.major} too old for modern PyTorch CUDA"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available in current environment")
def test_cuda_tensor_autograd():
    """Verify fundamental CUDA tensor operations and autograd."""
    device = torch.device("cuda:0")
    a = torch.randn(512, 512, device=device, requires_grad=True)
    b = torch.randn(512, 512, device=device, requires_grad=True)
    c = torch.matmul(a, b)
    loss = c.mean()
    loss.backward()

    assert a.grad is not None
    assert torch.isfinite(a.grad).all()
    assert b.grad is not None
    assert torch.isfinite(b.grad).all()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available in current environment")
def test_resnet101_cuda_smoke_test():
    """Verify ResNet-101 forward/backward execution on CUDA with batch size 2."""
    device = torch.device("cuda:0")
    model = build_resnet101_classifier(pretrained=False, num_classes=2, in_channels=3)
    model.to(device)
    model.train()

    x = torch.randn(2, 3, 224, 224, device=device)
    y = torch.tensor([0, 1], device=device, dtype=torch.long)

    logits = model(x)
    assert logits.shape == (2, 2)
    assert logits.device.type == "cuda"
    assert torch.isfinite(logits).all()

    criterion = nn.CrossEntropyLoss()
    loss = criterion(logits, y)
    assert torch.isfinite(loss)
    loss.backward()

    assert model.fc.weight.grad is not None
    assert torch.isfinite(model.fc.weight.grad).all()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available in current environment")
def test_amp_fp16_autocast_cuda():
    """Verify Automatic Mixed Precision (AMP FP16) on CUDA."""
    device = torch.device("cuda:0")
    model = build_resnet101_classifier(pretrained=False, num_classes=2, in_channels=3)
    model.to(device)
    model.train()

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    scaler = torch.amp.GradScaler("cuda")
    optimizer.zero_grad()

    x = torch.randn(2, 3, 224, 224, device=device)
    y = torch.tensor([0, 1], device=device, dtype=torch.long)
    criterion = nn.CrossEntropyLoss()

    with torch.amp.autocast("cuda", dtype=torch.float16):
        logits = model(x)
        loss = criterion(logits, y)

    assert torch.isfinite(loss)
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()

    assert model.fc.weight.grad is not None
    assert torch.isfinite(model.fc.weight.grad).all()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available in current environment")
def test_gpu_vram_headroom_guard():
    """Verify peak memory consumption for batch size 2 is safely below 4 GB VRAM budget."""
    device = torch.device("cuda:0")
    torch.cuda.reset_peak_memory_stats(device)

    model = build_resnet101_classifier(pretrained=False, num_classes=2, in_channels=3)
    model.to(device)
    model.train()

    x = torch.randn(2, 3, 224, 224, device=device)
    y = torch.tensor([0, 1], device=device, dtype=torch.long)

    with torch.amp.autocast("cuda", dtype=torch.float16):
        logits = model(x)
        loss = nn.CrossEntropyLoss()(logits, y)
    loss.backward()

    peak_mb = torch.cuda.max_memory_allocated(device) / (1024 ** 2)
    # 4 GB VRAM has 4094 MB; peak memory for batch size 2 ResNet-101 must be under 1500 MB
    assert peak_mb < 1500.0, f"Peak VRAM {peak_mb:.1f} MB exceeds safe headroom"

    del model, x, y, logits, loss
    torch.cuda.empty_cache()


def test_gpu_training_yaml_configuration():
    """Verify GPU training YAML configuration exists and defines resource-aware settings."""
    settings = get_settings()
    yaml_path = settings.base_dir / "ml_service/config/training_fracatlas_gpu.yaml"
    assert yaml_path.exists(), f"Configuration file missing: {yaml_path}"

    with open(yaml_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    assert cfg["hardware"]["device"] == "cuda"
    assert cfg["hardware"]["mixed_precision"] is True
    assert cfg["dataloader"]["batch_size"] == 2
    assert cfg["training"]["gradient_accumulation_steps"] >= 4
    assert cfg["dataloader"]["num_workers"] == 0

