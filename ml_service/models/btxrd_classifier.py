"""
OsteoGuard AI - BTXRD Three-Class Bone Tumour Classifier.

Architecture:
- Backbone: ResNet-34 initialized with official TorchVision ImageNet-1K pretrained weights.
- Early layers (conv1, bn1, layer1, layer2, layer3) frozen.
- Fine-tuned layers: layer4 and custom classification head.
- Classification Head: Dropout(p=0.4) -> Linear(in_features=512, out_features=3).
- Frozen BatchNorm layers locked in eval mode.
- Numerically stable Multi-Class Focal Loss with inverse-frequency alpha weighting.
"""

from typing import Any, Dict, List, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torchvision.models import ResNet34_Weights


class MulticlassFocalLoss(nn.Module):
    """
    Numerically stable Multi-Class Focal Loss for imbalanced categorical classification.

    Loss formula:
        FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)
    where p_t is the softmax probability for the true class, computed via log_softmax.
    """

    def __init__(
        self,
        gamma: float = 2.0,
        alpha: Optional[torch.Tensor] = None,
        reduction: str = "mean",
    ):
        super().__init__()
        self.gamma = float(gamma)
        self.reduction = reduction

        if alpha is not None:
            if not isinstance(alpha, torch.Tensor):
                alpha = torch.tensor(alpha, dtype=torch.float32)
            self.register_buffer("alpha", alpha.float())
        else:
            self.alpha = None

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits: Float tensor of shape (B, C) representing unnormalized class scores.
            targets: Long tensor of shape (B,) representing ground-truth class indices.
        """
        if logits.dim() != 2:
            raise ValueError(f"Expected 2D logits (B, C), got {logits.shape}")
        if targets.dim() != 1:
            raise ValueError(f"Expected 1D targets (B,), got {targets.shape}")

        num_classes = logits.size(1)
        log_probs = F.log_softmax(logits, dim=1)  # (B, C)
        probs = torch.exp(log_probs)             # (B, C)

        # Gather target class probabilities
        targets = targets.view(-1, 1)  # (B, 1)
        log_pt = log_probs.gather(1, targets).squeeze(1)  # (B,)
        pt = probs.gather(1, targets).squeeze(1)          # (B,)

        # Compute focal modulation
        focal_weight = (1.0 - pt) ** self.gamma  # (B,)

        # Apply class balancing alpha weights
        if self.alpha is not None:
            if self.alpha.device != logits.device:
                self.alpha = self.alpha.to(logits.device)
            alpha_t = self.alpha.gather(0, targets.squeeze(1))  # (B,)
            loss = -alpha_t * focal_weight * log_pt
        else:
            loss = -focal_weight * log_pt

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        else:
            return loss


class BTXRDResNet34(nn.Module):
    """
    ResNet-34 for BTXRD three-class primary bone tumour classification.
    """

    def __init__(
        self,
        num_classes: int = 3,
        dropout_p: float = 0.4,
        pretrained: bool = True,
        freeze_early: bool = True,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.dropout_p = dropout_p
        self.pretrained = pretrained
        self.freeze_early = freeze_early

        # Load TorchVision ResNet-34
        if pretrained:
            weights = ResNet34_Weights.IMAGENET1K_V1
            self.weights_name = "ResNet34_Weights.IMAGENET1K_V1"
            base_model = models.resnet34(weights=weights)
        else:
            self.weights_name = "None (Random Initialization)"
            base_model = models.resnet34(weights=None)

        # Extract backbone feature extractor layers
        self.conv1 = base_model.conv1
        self.bn1 = base_model.bn1
        self.relu = base_model.relu
        self.maxpool = base_model.maxpool

        self.layer1 = base_model.layer1
        self.layer2 = base_model.layer2
        self.layer3 = base_model.layer3
        self.layer4 = base_model.layer4
        self.avgpool = base_model.avgpool

        # Replace classification head
        in_features = base_model.fc.in_features  # 512
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout_p),
            nn.Linear(in_features, num_classes),
        )

        # Freeze early layers if requested
        if freeze_early:
            self._freeze_early_layers()

    def _freeze_early_layers(self):
        """Freezes conv1, bn1, layer1, layer2, and layer3."""
        for layer in [self.conv1, self.bn1, self.layer1, self.layer2, self.layer3]:
            for param in layer.parameters():
                param.requires_grad = False
        self.layer3_unfrozen = False

    def unfreeze_layer3(self):
        """Unfreezes layer3 parameters while keeping its BatchNorm layers in eval mode."""
        for param in self.layer3.parameters():
            param.requires_grad = True
        self.layer3_unfrozen = True

    def train(self, mode: bool = True):
        """
        Override train mode to ensure frozen BatchNorm layers remain in eval mode.
        """
        super().train(mode)
        if self.freeze_early and mode:
            # Force frozen modules into eval mode
            self.conv1.eval()
            self.bn1.eval()
            self.layer1.eval()
            self.layer2.eval()
            if not getattr(self, "layer3_unfrozen", False):
                self.layer3.eval()
            else:
                for m in self.layer3.modules():
                    if isinstance(m, nn.modules.batchnorm._BatchNorm):
                        m.eval()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass yielding unnormalized class logits of shape (B, 3).
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
        x = torch.flatten(x, 1)  # (B, 512)
        logits = self.classifier(x)  # (B, 3)
        return logits

    def get_parameter_groups(
        self,
        classifier_lr: float = 1e-4,
        layer4_lr: float = 1e-5,
        layer3_lr: float = 3e-6,
        weight_decay: float = 0.01,
    ) -> List[Dict[str, Any]]:
        """
        Returns differential learning rate parameter groups for AdamW.
        """
        groups = []
        layer3_params = [p for p in self.layer3.parameters() if p.requires_grad]
        if layer3_params:
            groups.append({
                "params": layer3_params,
                "lr": layer3_lr,
                "weight_decay": weight_decay,
            })
        groups.append({
            "params": [p for p in self.layer4.parameters() if p.requires_grad],
            "lr": layer4_lr,
            "weight_decay": weight_decay,
        })
        groups.append({
            "params": [p for p in self.classifier.parameters() if p.requires_grad],
            "lr": classifier_lr,
            "weight_decay": weight_decay,
        })
        return groups

    def get_model_metadata(self) -> Dict[str, Any]:
        """Returns comprehensive model parameter audit."""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        frozen_params = total_params - trainable_params

        return {
            "model_architecture": "ResNet-34",
            "weights_provenance": self.weights_name,
            "num_classes": self.num_classes,
            "dropout_p": self.dropout_p,
            "freeze_early_backbone": self.freeze_early,
            "total_parameters": total_params,
            "trainable_parameters": trainable_params,
            "frozen_parameters": frozen_params,
            "trainable_percentage": round(trainable_params / total_params * 100, 2),
        }

