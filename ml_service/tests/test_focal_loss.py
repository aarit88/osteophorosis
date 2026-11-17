"""
Unit tests for FocalLoss implementation.
Verifies numerical stability, equivalence to CrossEntropy when gamma=0,
alpha class weighting, reduction modes, and gradient flow.
"""

import math
import pytest
import torch
import torch.nn.functional as F
from ml_service.training.focal_loss import FocalLoss


def test_focal_loss_equivalence_to_cross_entropy():
    """When gamma=0 and alpha is None, FocalLoss should match CrossEntropyLoss."""
    focal = FocalLoss(gamma=0.0, alpha=None, reduction="mean")
    ce = torch.nn.CrossEntropyLoss(reduction="mean")

    logits = torch.randn(10, 2, requires_grad=True)
    targets = torch.randint(0, 2, (10,))

    loss_focal = focal(logits, targets)
    loss_ce = ce(logits, targets)

    assert torch.allclose(loss_focal, loss_ce, atol=1e-6)


def test_focal_loss_class_weighting():
    """Verifies that alpha_positive (0.75) and alpha_negative (0.25) are applied correctly."""
    focal = FocalLoss(gamma=0.0, alpha=(0.25, 0.75), reduction="none")

    # Logits equal for both classes -> p_t = 0.5 for all
    logits = torch.zeros(2, 2)
    targets = torch.tensor([0, 1])  # 0: non-fractured, 1: fractured

    losses = focal(logits, targets)
    expected_loss_0 = 0.25 * math.log(2)  # -0.25 * log(0.5)
    expected_loss_1 = 0.75 * math.log(2)  # -0.75 * log(0.5)

    assert math.isclose(losses[0].item(), expected_loss_0, rel_tol=1e-5)
    assert math.isclose(losses[1].item(), expected_loss_1, rel_tol=1e-5)
    assert losses[1].item() == pytest.approx(3.0 * losses[0].item(), rel=1e-5)


def test_focal_loss_downweights_easy_examples():
    """Focal loss with gamma=2.0 downweights high-confidence examples compared to gamma=0."""
    focal_ce = FocalLoss(gamma=0.0, alpha=None, reduction="none")
    focal_gamma2 = FocalLoss(gamma=2.0, alpha=None, reduction="none")

    # High confidence for class 1
    logits_easy = torch.tensor([[ -5.0, 5.0 ]])  # p_1 ~ 0.99995
    target = torch.tensor([1])

    loss_ce = focal_ce(logits_easy, target)
    loss_gamma2 = focal_gamma2(logits_easy, target)

    # With gamma=2, (1 - p_t)^2 is tiny (~ 1e-8), so loss should be suppressed heavily
    assert loss_gamma2.item() < loss_ce.item() * 1e-4


def test_focal_loss_backward_gradient():
    """Verifies that gradients backpropagate cleanly without NaN or Inf."""
    focal = FocalLoss(gamma=2.0, alpha=(0.25, 0.75), reduction="mean")

    logits = torch.randn(8, 2, requires_grad=True)
    targets = torch.randint(0, 2, (8,))

    loss = focal(logits, targets)
    loss.backward()

    assert logits.grad is not None
    assert not torch.isnan(logits.grad).any()
    assert not torch.isinf(logits.grad).any()


def test_focal_loss_reduction_modes():
    """Checks 'mean', 'sum', and 'none' reductions."""
    logits = torch.randn(4, 2)
    targets = torch.tensor([0, 1, 0, 1])

    fl_none = FocalLoss(gamma=2.0, alpha=(0.25, 0.75), reduction="none")(logits, targets)
    fl_mean = FocalLoss(gamma=2.0, alpha=(0.25, 0.75), reduction="mean")(logits, targets)
    fl_sum = FocalLoss(gamma=2.0, alpha=(0.25, 0.75), reduction="sum")(logits, targets)

    assert fl_none.shape == (4,)
    assert math.isclose(fl_mean.item(), fl_none.mean().item(), rel_tol=1e-5)
    assert math.isclose(fl_sum.item(), fl_none.sum().item(), rel_tol=1e-5)


def test_focal_loss_cuda_device_compatibility():
    """Verifies that FocalLoss runs on CUDA if available without device mismatch."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    device = torch.device("cuda:0")
    focal = FocalLoss(gamma=2.0, alpha=(0.25, 0.75), reduction="mean").to(device)

    logits = torch.randn(4, 2, device=device, requires_grad=True)
    targets = torch.tensor([0, 1, 0, 1], device=device)

    loss = focal(logits, targets)
    assert loss.device.type == "cuda"
    loss.backward()
    assert logits.grad.device.type == "cuda"

