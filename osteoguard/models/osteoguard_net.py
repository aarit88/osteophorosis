"""
OsteoGuard AI - Unified Deep Learning and Radiomic Framework Network.
Assembles the Skeletal U-Net, Deep Semantic Backbone (ResNet-101 / Swin),
and Late Fusion Head into an end-to-end differentiable PyTorch model.
"""

from typing import Dict, Optional, Tuple
import torch
import torch.nn as nn

from osteoguard.constants import (
    NUM_CLASSES,
    RADIOMICS_FEATURE_DIM,
    SEMANTIC_EMBEDDING_DIM,
    FUSION_VECTOR_DIM,
)
from osteoguard.models.unet import SkeletalUNet
from osteoguard.models.backbones import build_semantic_backbone
from osteoguard.models.fusion import LateFusionClassificationHead


class OsteoGuardNet(nn.Module):
    """
    Unified dual-stream architecture:
      1. Skeletal segmentation (optional U-Net mask)
      2. Macroscopic geometric stream (ResNet-101 or Swin Transformer)
      3. IBSI radiomic stream integration
      4. Late-fusion calibrated MLP classifier
    """

    def __init__(
        self,
        backbone_name: str = "resnet101",
        pretrained_backbone: bool = False,
        in_channels: int = 1,
        semantic_dim: int = SEMANTIC_EMBEDDING_DIM,
        radiomics_dim: int = RADIOMICS_FEATURE_DIM,
        num_classes: int = NUM_CLASSES,
        enable_unet_masking: bool = True,
    ):
        super().__init__()
        self.backbone_name = backbone_name
        self.enable_unet_masking = enable_unet_masking

        # 1. Skeletal Segmentation U-Net
        if enable_unet_masking:
            self.unet = SkeletalUNet(in_channels=in_channels, out_channels=1)
        else:
            self.unet = None

        # 2. Deep Semantic Stream (ResNet-101 or Swin)
        self.semantic_backbone = build_semantic_backbone(
            backbone_name=backbone_name,
            pretrained=pretrained_backbone,
            in_channels=in_channels,
            semantic_dim=semantic_dim,
        )

        # 3. Late Fusion Classification Head
        self.fusion_head = LateFusionClassificationHead(
            semantic_dim=semantic_dim,
            radiomics_dim=radiomics_dim,
            fusion_dim=semantic_dim + radiomics_dim,
            num_classes=num_classes,
        )

    def forward(
        self,
        images: torch.Tensor,
        radiomics_features: torch.Tensor,
        custom_bone_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Full forward pass.
        Args:
            images: (B, C, H, W) input radiological scan (normalized)
            radiomics_features: (B, 102) pre-extracted IBSI texture features
            custom_bone_mask: Optional external mask tensor (B, 1, H, W)
        Returns:
            Dictionary with logits, probabilities, embeddings, and masks.
        """
        # Determine skeletal mask
        bone_mask = None
        seg_logits = None
        if custom_bone_mask is not None:
            bone_mask = custom_bone_mask
            masked_images = images * bone_mask
        elif self.enable_unet_masking and self.unet is not None:
            seg_logits = self.unet(images)
            bone_mask = (torch.sigmoid(seg_logits) >= 0.5).float()
            masked_images = images * bone_mask
        else:
            masked_images = images

        # Semantic stream forward pass
        v_sem, spatial_features = self.semantic_backbone(masked_images)

        # Late fusion forward pass
        cls_logits, v_fusion = self.fusion_head(v_sem, radiomics_features)
        probs = torch.softmax(cls_logits, dim=-1)

        output = {
            "logits": cls_logits,
            "probabilities": probs,
            "v_sem": v_sem,
            "v_rad": radiomics_features,
            "v_fusion": v_fusion,
            "spatial_features": spatial_features,
        }

        if bone_mask is not None:
            output["bone_mask"] = bone_mask
        if seg_logits is not None:
            output["seg_logits"] = seg_logits

        return output

