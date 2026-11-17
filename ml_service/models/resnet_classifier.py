"""
OsteoGuard AI - ResNet-101 Binary Classification Architecture.

Features:
  - 101-layer deep residual network backbone (He et al., 2016).
  - Adaptive Global Average Pooling.
  - Linear classification head producing two output logits [P(non_fracture), P(fracture)].
  - Layer4 feature extraction hooks for Grad-CAM explainability.
  - Configurable pretraining weights with safe offline initialization (no forced downloads).
"""

from typing import Optional, Tuple
import torch
import torch.nn as nn
import torchvision.models as models


class ResNet101BinaryClassifier(nn.Module):
    """
    ResNet-101 architecture adapted for binary medical image classification.
    Produces 2 class logits:
      Index 0: Non-fractured
      Index 1: Fractured
    """

    def __init__(
        self,
        pretrained: bool = False,
        num_classes: int = 2,
        in_channels: int = 3,
        dropout_rate: float = 0.2,
    ):
        super().__init__()
        self.pretrained = pretrained
        self.num_classes = num_classes
        self.in_channels = in_channels

        # Initialize ResNet-101 backbone
        # When pretrained is False, weights=None (zero network download)
        weights = models.ResNet101_Weights.DEFAULT if pretrained else None
        base_resnet = models.resnet101(weights=weights)

        # Adapt first convolution if input channels != 3
        if in_channels != 3:
            orig = base_resnet.conv1
            self.conv1 = nn.Conv2d(
                in_channels,
                orig.out_channels,
                kernel_size=orig.kernel_size,
                stride=orig.stride,
                padding=orig.padding,
                bias=False,
            )
            if pretrained and in_channels == 1:
                with torch.no_grad():
                    self.conv1.weight.copy_(orig.weight.mean(dim=1, keepdim=True))
        else:
            self.conv1 = base_resnet.conv1

        self.bn1 = base_resnet.bn1
        self.relu = base_resnet.relu
        self.maxpool = base_resnet.maxpool

        self.layer1 = base_resnet.layer1
        self.layer2 = base_resnet.layer2
        self.layer3 = base_resnet.layer3
        self.layer4 = base_resnet.layer4

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(p=dropout_rate) if dropout_rate > 0.0 else nn.Identity()
        self.fc = nn.Linear(2048, num_classes)

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract spatial activation maps from layer4 (hook for Grad-CAM)."""
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        Args:
            x: Input tensor (B, C, H, W).
        Returns:
            logits: Output tensor (B, num_classes).
        """
        features = self.forward_features(x)
        pooled = self.avgpool(features)
        flattened = torch.flatten(pooled, 1)
        dropped = self.dropout(flattened)
        logits = self.fc(dropped)
        return logits

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """Calculates normalized class probabilities via softmax."""
        logits = self.forward(x)
        return torch.softmax(logits, dim=1)


def build_resnet101_classifier(
    pretrained: bool = False,
    num_classes: int = 2,
    in_channels: int = 3,
    dropout_rate: float = 0.2,
) -> ResNet101BinaryClassifier:
    """Helper factory for ResNet-101 binary classification model."""
    return ResNet101BinaryClassifier(
        pretrained=pretrained,
        num_classes=num_classes,
        in_channels=in_channels,
        dropout_rate=dropout_rate,
    )

