"""
OsteoGuard AI - Deep Semantic Representation Stream Backbones.
Implements the primary ResNet-101 backbone and comparative Swin Transformer backbone,
producing 512-dimensional semantic latent embeddings (v_sem in R^512).
"""

from typing import Tuple, Optional
import torch
import torch.nn as nn
import torchvision.models as models

from osteoguard.constants import SEMANTIC_EMBEDDING_DIM


class ResNet101SemanticBackbone(nn.Module):
    """
    ResNet-101 Deep Semantic Stream.
    Extracts macroscopic cortical geometries and lesion contours.
    Preserves layer4 activation hooks for Grad-CAM explainability.
    Outputs a 512-dimensional semantic embedding vector v_sem.
    """

    def __init__(
        self,
        pretrained: bool = False,
        in_channels: int = 1,
        semantic_dim: int = SEMANTIC_EMBEDDING_DIM,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.semantic_dim = semantic_dim

        # Load standard ResNet-101
        weights = models.ResNet101_Weights.DEFAULT if pretrained else None
        base_resnet = models.resnet101(weights=weights)

        # Adapt first convolution if single-channel grayscale input (X-ray / CT bone window)
        if in_channels == 1:
            original_conv = base_resnet.conv1
            self.conv1 = nn.Conv2d(
                1,
                original_conv.out_channels,
                kernel_size=original_conv.kernel_size,
                stride=original_conv.stride,
                padding=original_conv.padding,
                bias=False,
            )
            if pretrained:
                # Average original 3 RGB channel weights into 1 channel
                with torch.no_grad():
                    self.conv1.weight.copy_(original_conv.weight.mean(dim=1, keepdim=True))
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

        # Project ResNet-101 2048-dim features to 512-dim v_sem
        self.projector = nn.Sequential(
            nn.Linear(2048, semantic_dim),
            nn.BatchNorm1d(semantic_dim),
            nn.ReLU(inplace=True),
        )

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        """Returns spatial feature maps from layer4 (for Grad-CAM)."""
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return x

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.
        Returns:
            v_sem: (B, 512) semantic embedding
            spatial_features: (B, 2048, H', W') layer4 feature maps for Grad-CAM
        """
        spatial_features = self.forward_features(x)
        pooled = self.avgpool(spatial_features)
        flattened = torch.flatten(pooled, 1)
        v_sem = self.projector(flattened)
        return v_sem, spatial_features


class SwinSemanticBackbone(nn.Module):
    """
    Swin Transformer (Comparative Backbone).
    Extracts hierarchical multi-scale attention representations across bone structures.
    Outputs a 512-dimensional semantic embedding vector v_sem.
    """

    def __init__(
        self,
        pretrained: bool = False,
        in_channels: int = 1,
        semantic_dim: int = SEMANTIC_EMBEDDING_DIM,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.semantic_dim = semantic_dim

        weights = models.Swin_T_Weights.DEFAULT if pretrained else None
        base_swin = models.swin_t(weights=weights)

        # Swin expects 3 channels by default; adapt initial patch embedding if 1 channel
        if in_channels == 1:
            first_conv = base_swin.features[0][0]
            new_conv = nn.Conv2d(
                1,
                first_conv.out_channels,
                kernel_size=first_conv.kernel_size,
                stride=first_conv.stride,
                padding=first_conv.padding,
                bias=True,
            )
            if pretrained:
                with torch.no_grad():
                    new_conv.weight.copy_(first_conv.weight.mean(dim=1, keepdim=True))
                    if first_conv.bias is not None:
                        new_conv.bias.copy_(first_conv.bias)
            base_swin.features[0][0] = new_conv

        self.features = base_swin.features
        self.norm = base_swin.norm
        self.permute = base_swin.permute
        self.avgpool = base_swin.avgpool

        # Swin-T output features are 768-dimensional
        swin_feat_dim = base_swin.head.in_features
        self.projector = nn.Sequential(
            nn.Linear(swin_feat_dim, semantic_dim),
            nn.BatchNorm1d(semantic_dim),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            v_sem: (B, 512) semantic embedding
            spatial_features: (B, H', W', C) from final swin stage
        """
        spatial_features = self.features(x)
        normed = self.norm(spatial_features)
        permuted = self.permute(normed)
        pooled = self.avgpool(permuted)
        flattened = torch.flatten(pooled, 1)
        v_sem = self.projector(flattened)
        return v_sem, spatial_features


def build_semantic_backbone(
    backbone_name: str = "resnet101",
    pretrained: bool = False,
    in_channels: int = 1,
    semantic_dim: int = SEMANTIC_EMBEDDING_DIM,
) -> nn.Module:
    """Factory builder for semantic representation backbones."""
    name = backbone_name.lower()
    if name == "resnet101":
        return ResNet101SemanticBackbone(
            pretrained=pretrained,
            in_channels=in_channels,
            semantic_dim=semantic_dim,
        )
    elif name in ("swin_t", "swin"):
        return SwinSemanticBackbone(
            pretrained=pretrained,
            in_channels=in_channels,
            semantic_dim=semantic_dim,
        )
    else:
        raise ValueError(
            f"Unsupported backbone: '{backbone_name}'. Must be 'resnet101' or 'swin_t'."
        )

