"""
OsteoGuard AI - Bone Scan ResNet-18 Binary Classifier.

Model features:
  - ResNet-18 compact backbone for nuclear medicine bone scintigraphy.
  - Adapted 1-channel input convolution (conv1) initialized by averaging pretrained RGB channels.
  - Dropout regularization (p=0.4).
  - Single logit linear head for binary metastasis prediction: P(metastasis | image).
  - Designed for aspect-ratio preserved scintigraphy inputs (256x1024).
"""

from pathlib import Path
from typing import Any, Dict, Tuple
import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.models import ResNet18_Weights


class BoneScanResNet18(nn.Module):
    """
    Compact ResNet-18 adapted for 1-channel 16-bit whole-body bone scintigraphy.
    Outputs a single scalar logit per image.
    """

    def __init__(self, pretrained: bool = True, dropout_rate: float = 0.4):
        super().__init__()
        weights = ResNet18_Weights.DEFAULT if pretrained else None
        base_resnet = models.resnet18(weights=weights)

        # Adapt conv1: 3 channels -> 1 channel
        orig_conv1 = base_resnet.conv1
        self.conv1 = nn.Conv2d(
            in_channels=1,
            out_channels=orig_conv1.out_channels,
            kernel_size=orig_conv1.kernel_size,
            stride=orig_conv1.stride,
            padding=orig_conv1.padding,
            bias=False,
        )

        if pretrained:
            with torch.no_grad():
                # Average weights across the 3 RGB channels to preserve spatial filter energy
                self.conv1.weight.copy_(orig_conv1.weight.mean(dim=1, keepdim=True))

        self.bn1 = base_resnet.bn1
        self.relu = base_resnet.relu
        self.maxpool = base_resnet.maxpool

        self.layer1 = base_resnet.layer1
        self.layer2 = base_resnet.layer2
        self.layer3 = base_resnet.layer3
        self.layer4 = base_resnet.layer4

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(p=dropout_rate) if dropout_rate > 0.0 else nn.Identity()
        self.fc = nn.Linear(512, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        Args:
            x: Input tensor of shape (B, 1, H, W).
        Returns:
            Logits of shape (B, 1).
        """
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        logits = self.fc(x)
        return logits


def build_bone_scan_resnet18(
    device: torch.device,
    pretrained: bool = True,
    dropout_rate: float = 0.4,
) -> Tuple[nn.Module, Dict[str, Any]]:
    """
    Builds and freezes BoneScanResNet18.
    Freezes conv1, bn1, layer1, layer2, layer3.
    Keeps layer4 and fc trainable.
    """
    model = BoneScanResNet18(pretrained=pretrained, dropout_rate=dropout_rate)
    model.to(device)

    # Freeze conv1 through layer3
    for p in model.conv1.parameters():
        p.requires_grad = False
    for p in model.bn1.parameters():
        p.requires_grad = False
    for p in model.layer1.parameters():
        p.requires_grad = False
    for p in model.layer2.parameters():
        p.requires_grad = False
    for p in model.layer3.parameters():
        p.requires_grad = False

    # Trainable: layer4 and fc head
    for p in model.layer4.parameters():
        p.requires_grad = True
    for p in model.fc.parameters():
        p.requires_grad = True

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen_params = sum(p.numel() for p in model.parameters() if not p.requires_grad)

    ckpt_cache_path = Path.home() / ".cache/torch/hub/checkpoints/resnet18-f37072fd.pth"
    sha256_hash = ""
    if ckpt_cache_path.exists():
        import hashlib
        h = hashlib.sha256()
        with open(ckpt_cache_path, "rb") as f:
            while chunk := f.read(8192 * 1024):
                h.update(chunk)
        sha256_hash = h.hexdigest()

    model_info = {
        "architecture": "BoneScanResNet18",
        "base_model": "ResNet18",
        "pretrained": pretrained,
        "weights_enum": "ResNet18_Weights.IMAGENET1K_V1",
        "local_cache_path": str(ckpt_cache_path),
        "cache_sha256": sha256_hash,
        "in_channels": 1,
        "total_parameters": total_params,
        "trainable_parameters": trainable_params,
        "frozen_parameters": frozen_params,
        "trainable_percentage": round((trainable_params / total_params) * 100, 2),
        "classifier_head": "Dropout(p=0.4) -> Linear(in_features=512, out_features=1)",
    }
    return model, model_info


def set_eval_bn_mode(model: nn.Module):
    """Sets model to train mode while keeping all BatchNorm running stats in eval mode."""
    model.train()
    for m in model.modules():
        if isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
            m.eval()

