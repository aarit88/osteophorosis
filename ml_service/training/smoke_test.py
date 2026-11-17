"""
OsteoGuard AI - Stage 07 Training Pipeline Smoke Test Engine.

Executes mandatory verification checklist:
  1. Load one training batch.
  2. Load one validation batch.
  3. Check tensor dimensions.
  4. Check label distributions.
  5. Check finite input values.
  6. Run one forward pass.
  7. Run one loss calculation.
  8. Run one backward pass.
  9. Verify gradients.
  10. Verify optimizer update.
  11. Verify checkpoint save and reload.
"""

from pathlib import Path
from typing import Any, Dict, Optional
import torch
import torch.nn as nn
import torch.optim as optim

from ml_service.config.settings import get_settings
from ml_service.datasets.fracatlas_dataset import create_fracatlas_dataloaders
from ml_service.models.resnet_classifier import build_resnet101_classifier
from ml_service.utils.logging import logger


def run_training_smoke_test(
    batch_size: int = 8,
    device: Optional[torch.device] = None,
) -> Dict[str, Any]:
    """
    Executes the 11-step training pipeline smoke test.
    Returns a dictionary of execution results.
    """
    settings = get_settings()
    root = settings.base_dir
    manifest_path = root / "ml_service/data/splits/fracatlas_classification_split.csv"

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    results: Dict[str, Any] = {
        "device": str(device),
        "batch_size": batch_size,
        "steps": {},
    }

    logger.info(f"Initiating Training Pipeline Smoke Test on device: {device} (batch_size={batch_size})")

    # Step 1: Load one training batch
    # Step 2: Load one validation batch
    loaders = create_fracatlas_dataloaders(
        manifest_path=manifest_path,
        batch_size=batch_size,
        num_workers=0,
    )
    train_iter = iter(loaders["train"])
    val_iter = iter(loaders["val"])

    x_train, y_train = next(train_iter)
    x_val, y_val = next(val_iter)

    results["steps"]["1_train_batch_loaded"] = True
    results["steps"]["2_val_batch_loaded"] = True

    # Step 3: Check tensor dimensions
    assert x_train.shape == (batch_size, 3, 224, 224), f"Unexpected x_train shape: {x_train.shape}"
    assert y_train.shape == (batch_size,), f"Unexpected y_train shape: {y_train.shape}"
    assert x_val.shape == (batch_size, 3, 224, 224), f"Unexpected x_val shape: {x_val.shape}"
    assert y_val.shape == (batch_size,), f"Unexpected y_val shape: {y_val.shape}"
    results["steps"]["3_tensor_dimensions_verified"] = {
        "x_train_shape": list(x_train.shape),
        "y_train_shape": list(y_train.shape),
        "x_val_shape": list(x_val.shape),
        "y_val_shape": list(y_val.shape),
    }

    # Step 4: Check label distributions
    unique_train_labels = set(y_train.tolist())
    unique_val_labels = set(y_val.tolist())
    assert unique_train_labels.issubset({0, 1}), f"Illegal train labels: {unique_train_labels}"
    assert unique_val_labels.issubset({0, 1}), f"Illegal val labels: {unique_val_labels}"
    results["steps"]["4_label_distribution_verified"] = {
        "train_labels": y_train.tolist(),
        "val_labels": y_val.tolist(),
    }

    # Step 5: Check finite input values (no NaNs, no Infs)
    assert torch.isfinite(x_train).all(), "x_train contains non-finite values (NaN/Inf)"
    assert torch.isfinite(x_val).all(), "x_val contains non-finite values (NaN/Inf)"
    results["steps"]["5_finite_inputs_verified"] = True

    # Step 6: Run one forward pass
    # Using randomly initialized ResNet-101 (pretrained=False, no download)
    model = build_resnet101_classifier(pretrained=False, num_classes=2, in_channels=3)
    model.to(device)
    model.train()

    x_dev = x_train.to(device)
    y_dev = y_train.to(device)

    logits = model(x_dev)
    assert logits.shape == (batch_size, 2), f"Unexpected logits shape: {logits.shape}"
    assert torch.isfinite(logits).all(), "Logits contain non-finite values"
    results["steps"]["6_forward_pass_verified"] = {
        "logits_shape": list(logits.shape),
        "finite_logits": True,
    }

    # Step 7: Run one loss calculation
    criterion = nn.CrossEntropyLoss()
    loss = criterion(logits, y_dev)
    assert torch.isfinite(loss), f"Loss is non-finite: {loss.item()}"
    assert loss.item() >= 0.0, f"Negative loss: {loss.item()}"
    results["steps"]["7_loss_calculation_verified"] = {
        "loss_value": float(loss.item()),
    }

    # Step 8: Run one backward pass
    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-2)
    optimizer.zero_grad()
    loss.backward()
    results["steps"]["8_backward_pass_verified"] = True

    # Step 9: Verify gradients
    fc_grad = model.fc.weight.grad
    assert fc_grad is not None, "Classification head gradient is None"
    assert torch.isfinite(fc_grad).all(), "Classification head gradient contains NaN/Inf"
    assert torch.any(fc_grad != 0.0), "Classification head gradient is entirely zero"

    total_grad_norm = 0.0
    param_count = 0
    grad_count = 0
    for p in model.parameters():
        param_count += 1
        if p.grad is not None:
            grad_count += 1
            assert torch.isfinite(p.grad).all(), "Parameter gradient is non-finite"
            total_grad_norm += p.grad.data.norm(2).item() ** 2
    total_grad_norm = total_grad_norm ** 0.5

    assert grad_count > 0, "No gradients were computed"
    assert total_grad_norm > 0.0, "Total gradient norm is zero"
    results["steps"]["9_gradients_verified"] = {
        "parameters_with_grad": grad_count,
        "total_parameters": param_count,
        "total_gradient_norm": float(total_grad_norm),
    }

    # Step 10: Verify optimizer update
    initial_fc_weight = model.fc.weight.clone()
    optimizer.step()
    updated_fc_weight = model.fc.weight

    weight_diff = (updated_fc_weight - initial_fc_weight).abs().sum().item()
    assert weight_diff > 0.0, "Optimizer step did not update model parameters"
    results["steps"]["10_optimizer_update_verified"] = {
        "fc_weight_difference_l1": float(weight_diff),
    }

    # Step 11: Verify checkpoint save and reload
    ckpt_dir = root / "ml_artifacts/checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    test_ckpt_path = ckpt_dir / "smoke_test_checkpoint.pt"

    checkpoint_data = {
        "epoch": 1,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "loss": float(loss.item()),
        "batch_size": batch_size,
    }
    torch.save(checkpoint_data, test_ckpt_path)

    assert test_ckpt_path.exists(), "Checkpoint file was not created"
    assert test_ckpt_path.stat().st_size > 0, "Checkpoint file is empty"

    # Reload into fresh architecture
    fresh_model = build_resnet101_classifier(pretrained=False, num_classes=2, in_channels=3)
    loaded_data = torch.load(test_ckpt_path, weights_only=True)
    fresh_model.load_state_dict(loaded_data["model_state_dict"])
    fresh_model.to(device)

    # Validate exact weight equality
    assert torch.equal(model.fc.weight.cpu(), fresh_model.fc.weight.cpu()), "Reloaded weights do not match saved weights"

    # Clean up test checkpoint
    if test_ckpt_path.exists():
        test_ckpt_path.unlink()

    results["steps"]["11_checkpoint_save_reload_verified"] = True
    results["status"] = "PASS"

    logger.info("Training Pipeline Smoke Test completed with 100% SUCCESS (Status: PASS).")
    return results


if __name__ == "__main__":
    res = run_training_smoke_test(batch_size=8)
    print("\nSmoke Test Results Summary:")
    for k, v in res["steps"].items():
        print(f"  [{k}]: {v}")
    print(f"\nOverall Smoke Test Status: {res['status']}")
