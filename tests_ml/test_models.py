"""
Unit and integration tests for PyTorch neural network architectures:
  - Skeletal U-Net segmentation
  - ResNet-101 and Swin Transformer semantic backbones
  - 614-D Late Fusion Classification Head
  - Multiclass Focal Loss and Soft Dice Loss
  - Unified OsteoGuardNet forward pipeline
"""

import pytest
import torch
import torch.nn as nn

from osteoguard.models.unet import SkeletalUNet
from osteoguard.models.backbones import (
    ResNet101SemanticBackbone,
    SwinSemanticBackbone,
    build_semantic_backbone,
)
from osteoguard.models.fusion import LateFusionClassificationHead
from osteoguard.models.loss import MulticlassFocalLoss, SoftDiceLoss, OsteoGuardJointLoss
from osteoguard.models.osteoguard_net import OsteoGuardNet


def test_unet_segmentation_forward():
    """Verify U-Net input-output tensor dimensionality and binary masking."""
    unet = SkeletalUNet(in_channels=1, out_channels=1, base_c=16)
    x = torch.randn(2, 1, 64, 64)
    logits = unet(x)
    assert logits.shape == (2, 1, 64, 64)

    mask = unet.predict_mask(x, threshold=0.5)
    assert mask.shape == (2, 1, 64, 64)
    unique_vals = torch.unique(mask).tolist()
    for val in unique_vals:
        assert val in (0.0, 1.0)


def test_resnet101_semantic_backbone():
    """Verify ResNet-101 generates 512-D latent vector and spatial feature maps."""
    backbone = ResNet101SemanticBackbone(pretrained=False, in_channels=1, semantic_dim=512)
    x = torch.randn(2, 1, 64, 64)
    v_sem, spatial_feat = backbone(x)

    assert v_sem.shape == (2, 512)
    assert spatial_feat.shape[0] == 2
    assert spatial_feat.shape[1] == 2048  # ResNet-101 layer4 channels


def test_late_fusion_head():
    """Verify late fusion normalization, concatenation (614-D), and 4-class logits."""
    head = LateFusionClassificationHead(
        semantic_dim=512,
        radiomics_dim=102,
        fusion_dim=614,
        num_classes=4,
    )
    v_sem = torch.randn(3, 512)
    v_rad = torch.randn(3, 102)

    logits, v_fusion = head(v_sem, v_rad)
    assert logits.shape == (3, 4)
    assert v_fusion.shape == (3, 614)

    probs = head.predict_probabilities(v_sem, v_rad)
    assert probs.shape == (3, 4)
    assert torch.allclose(probs.sum(dim=1), torch.ones(3), atol=1e-5)


def test_multiclass_focal_loss():
    """Verify Multiclass Focal Loss behavior (gamma=2.0 downweights confident predictions)."""
    criterion = MulticlassFocalLoss(gamma=2.0)

    # Perfect prediction vs noisy prediction
    target = torch.tensor([0, 1])
    # Very confident logits
    good_logits = torch.tensor([[10.0, -5.0, -5.0, -5.0], [-5.0, 10.0, -5.0, -5.0]])
    poor_logits = torch.tensor([[0.1, 0.2, 0.3, 0.4], [0.4, 0.3, 0.2, 0.1]])

    loss_good = criterion(good_logits, target)
    loss_poor = criterion(poor_logits, target)

    assert loss_good.item() < loss_poor.item()
    assert loss_good.item() >= 0.0


def test_soft_dice_loss():
    """Verify Soft Dice Loss for skeletal segmentation."""
    dice_loss = SoftDiceLoss()
    target_mask = torch.ones(2, 1, 32, 32)
    pred_logits_perfect = torch.ones(2, 1, 32, 32) * 10.0
    pred_logits_disjoint = torch.ones(2, 1, 32, 32) * -10.0

    loss_perfect = dice_loss(pred_logits_perfect, target_mask)
    loss_disjoint = dice_loss(pred_logits_disjoint, target_mask)

    assert loss_perfect.item() < 0.1
    assert loss_disjoint.item() > 0.9


def test_osteoguard_net_unified_pipeline():
    """Verify complete end-to-end forward pass through OsteoGuardNet."""
    model = OsteoGuardNet(
        backbone_name="resnet101",
        pretrained_backbone=False,
        in_channels=1,
        enable_unet_masking=True,
    )
    # Substitute small unet base_c for test efficiency
    model.unet = SkeletalUNet(in_channels=1, out_channels=1, base_c=8)

    images = torch.randn(2, 1, 64, 64)
    radiomics = torch.randn(2, 102)

    outputs = model(images, radiomics)
    assert outputs["logits"].shape == (2, 4)
    assert outputs["probabilities"].shape == (2, 4)
    assert outputs["v_sem"].shape == (2, 512)
    assert outputs["v_rad"].shape == (2, 102)
    assert outputs["v_fusion"].shape == (2, 614)
    assert outputs["bone_mask"].shape == (2, 1, 64, 64)

