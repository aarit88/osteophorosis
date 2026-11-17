"""
OsteoGuard AI - Grad-CAM Anatomical Explainability Engine.

Computes Gradient-weighted Class Activation Mapping (Grad-CAM) over convolutional
backbone layers, producing deterministic visual heatmaps highlighting regions
salient to model class predictions.

Reference:
    Selvaraju, R. R., et al. (2017). "Grad-CAM: Visual Explanations from Deep
    Networks via Gradient-Based Localization." ICCV 2017.

RESEARCH INTEGRITY & CLINICAL GOVERNANCE NOTICE:
    - This software is strictly for research and architectural validation.
    - Grad-CAM heatmaps represent computational model attention, NOT clinical ground-truth.
    - A salient region in a heatmap is NOT equivalent to an anatomical segmentation mask.
    - Heatmaps must NOT be interpreted as medical evidence or diagnostic recommendations.
"""

from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class GradCAMExplainer:
    """
    Production-grade Grad-CAM implementation for PyTorch convolutional backbones.
    Features:
    - Explicit target-layer selection (by name string or module reference).
    - Explicit target-class validation and selection.
    - Automatic hook lifecycle management (zero memory leaks, context manager support).
    - Model state preservation (preserves eval/train mode and guarantees weight immutability).
    - Safe normalization with zero-gradient and flat-activation fallbacks (zero NaNs).
    """

    def __init__(
        self,
        model: nn.Module,
        target_layer: Union[nn.Module, str],
    ):
        if not isinstance(model, nn.Module):
            raise TypeError(f"model must be an instance of torch.nn.Module, got {type(model)}")

        self.model = model
        self.target_layer_name: str = ""
        self.activations: Optional[torch.Tensor] = None
        self.gradients: Optional[torch.Tensor] = None
        self._fwd_handle: Optional[torch.utils.hooks.RemovableHandle] = None
        self._bwd_handle: Optional[torch.utils.hooks.RemovableHandle] = None
        self._hooks_registered: bool = False

        self.target_layer: nn.Module = self._resolve_target_layer(target_layer)
        self.register_hooks()

    def _resolve_target_layer(self, layer_ref: Union[nn.Module, str]) -> nn.Module:
        """Resolves target layer from string identifier or module instance."""
        if isinstance(layer_ref, nn.Module):
            # Check module belongs to model
            found = False
            for name, mod in self.model.named_modules():
                if mod is layer_ref:
                    self.target_layer_name = name
                    found = True
                    break
            if not found:
                self.target_layer_name = layer_ref.__class__.__name__
            return layer_ref

        if isinstance(layer_ref, str):
            named_mods = dict(self.model.named_modules())
            if layer_ref in named_mods:
                self.target_layer_name = layer_ref
                return named_mods[layer_ref]
            
            # Layer not found; produce helpful error listing available convolutional layers
            conv_names = [n for n, m in named_mods.items() if isinstance(m, (nn.Conv2d, nn.Sequential)) and n]
            raise ValueError(
                f"Target layer {layer_ref!r} not found in model. Available candidate layers: {conv_names[:10]}"
            )

        raise TypeError(f"target_layer must be nn.Module or str, got {type(layer_ref)}")

    def register_hooks(self):
        """Registers forward and full backward hooks on target layer."""
        if self._hooks_registered:
            return

        def forward_hook(module, input, output):
            # Capture activations tensor
            self.activations = output

        def backward_hook(module, grad_input, grad_output):
            # Capture gradients of score w.r.t. target layer output
            if len(grad_output) > 0 and grad_output[0] is not None:
                self.gradients = grad_output[0]

        self._fwd_handle = self.target_layer.register_forward_hook(forward_hook)
        self._bwd_handle = self.target_layer.register_full_backward_hook(backward_hook)
        self._hooks_registered = True

    def remove_hooks(self):
        """Removes registered hooks to prevent memory leaks and hook accumulation."""
        if getattr(self, "_fwd_handle", None) is not None:
            self._fwd_handle.remove()
            self._fwd_handle = None
        if getattr(self, "_bwd_handle", None) is not None:
            self._bwd_handle.remove()
            self._bwd_handle = None
        self._hooks_registered = False

    def __enter__(self):
        self.register_hooks()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.remove_hooks()

    def __del__(self):
        self.remove_hooks()

    def generate_cam(
        self,
        input_tensor: torch.Tensor,
        target_class_idx: Optional[int] = None,
        input_size: Optional[Tuple[int, int]] = None,
    ) -> np.ndarray:
        """
        Executes end-to-end forward and backward pass to compute 2D Grad-CAM heatmap.

        Args:
            input_tensor: 4D PyTorch tensor (1, C, H, W) or (B, C, H, W) where B=1.
            target_class_idx: Target class index to explain. If None, uses argmax of model output.
            input_size: Destination (H, W) resolution. Defaults to input_tensor spatial shape.

        Returns:
            np.ndarray: 2D normalized float32 heatmap in [0.0, 1.0].
        """
        if not isinstance(input_tensor, torch.Tensor):
            raise TypeError(f"input_tensor must be a torch.Tensor, got {type(input_tensor)}")
        if input_tensor.ndim != 4:
            raise ValueError(f"input_tensor must be 4D (B, C, H, W), got shape {tuple(input_tensor.shape)}")
        if input_tensor.shape[0] != 1:
            raise ValueError(f"Grad-CAM currently supports batch size 1, got {input_tensor.shape[0]}")

        if input_size is None:
            input_size = (input_tensor.shape[2], input_tensor.shape[3])

        # Preserve original training mode
        was_training = self.model.training
        self.model.eval()

        try:
            self.model.zero_grad(set_to_none=True)
            self.activations = None
            self.gradients = None

            # Clone and enable requires_grad on input to ensure clean backward tracking
            inp = input_tensor.clone().detach().requires_grad_(True)

            # Forward pass
            logits = self.model(inp)
            if not isinstance(logits, torch.Tensor):
                raise TypeError(f"Model forward output must be a torch.Tensor, got {type(logits)}")
            if logits.ndim != 2 or logits.shape[0] != 1:
                raise ValueError(f"Expected logits shape (1, num_classes), got {tuple(logits.shape)}")

            num_classes = logits.shape[1]

            # Select target class
            if target_class_idx is None:
                selected_class = int(torch.argmax(logits, dim=1).item())
            else:
                if not isinstance(target_class_idx, int):
                    raise TypeError(f"target_class_idx must be int, got {type(target_class_idx)}")
                if target_class_idx < 0 or target_class_idx >= num_classes:
                    raise ValueError(
                        f"target_class_idx {target_class_idx} out of range for {num_classes} classes."
                    )
                selected_class = target_class_idx

            target_score = logits[0, selected_class]

            # Backward pass
            target_score.backward(retain_graph=True)

            if self.activations is None:
                raise RuntimeError("Activations were not captured. Check target_layer placement.")
            if self.gradients is None:
                # Disconnected computational graph or non-differentiable path
                return np.zeros(input_size, dtype=np.float32)

            # Grad-CAM formula:
            # 1. Spatially averaged gradient weights alpha_k = (1 / Z) * sum_{i,j} (d y^c / d A^k_{i,j})
            # activations shape: (1, C, H_feat, W_feat)
            # gradients shape:   (1, C, H_feat, W_feat)
            grads = self.gradients
            acts = self.activations

            # Global average pooling over spatial dimensions
            weights = torch.mean(grads, dim=(2, 3), keepdim=True)  # (1, C, 1, 1)

            # 2. Linear combination of feature maps weighted by alpha_k
            cam = torch.sum(weights * acts, dim=1, keepdim=True)  # (1, 1, H_feat, W_feat)

            # 3. Apply ReLU: only features that have a positive influence on the class of interest are kept
            cam = F.relu(cam)

            # Handle zero or flat map before interpolation
            cam_max = torch.max(cam).item()
            if cam_max <= 1e-8:
                return np.zeros(input_size, dtype=np.float32)

            # 4. Upsample to destination resolution
            cam = F.interpolate(cam, size=input_size, mode="bilinear", align_corners=False)
            cam_np = cam.squeeze().detach().cpu().numpy()

            # 5. Safe Min-Max Normalization to [0.0, 1.0]
            min_val = float(np.min(cam_np))
            max_val = float(np.max(cam_np))
            if max_val > min_val + 1e-8:
                cam_np = (cam_np - min_val) / (max_val - min_val)
            else:
                cam_np = np.zeros_like(cam_np)

            return np.clip(cam_np, 0.0, 1.0).astype(np.float32)

        finally:
            # Cleanly zero gradients to prevent accumulation in model parameters
            self.model.zero_grad(set_to_none=True)
            # Restore original model training mode
            self.model.train(was_training)

    def generate_heatmap(
        self,
        target_class_idx: int,
        target_logit: torch.Tensor,
        input_size: Tuple[int, int] = (512, 512),
    ) -> np.ndarray:
        """
        Legacy interface for backwards compatibility.
        Computes 2D normalized Grad-CAM heatmap given precomputed target logit.
        """
        self.model.zero_grad(set_to_none=True)
        target_logit.backward(retain_graph=True)

        if self.gradients is None or self.activations is None:
            raise RuntimeError("Gradients or activations not captured during backward pass.")

        weights = torch.mean(self.gradients, dim=(2, 3), keepdim=True)
        cam = torch.sum(weights * self.activations, dim=1, keepdim=True)
        cam = F.relu(cam)

        cam = F.interpolate(cam, size=input_size, mode="bilinear", align_corners=False)
        cam_np = cam.squeeze().detach().cpu().numpy()

        min_val = np.min(cam_np)
        max_val = np.max(cam_np)
        if max_val > min_val + 1e-8:
            cam_np = (cam_np - min_val) / (max_val - min_val)
        else:
            cam_np = np.zeros_like(cam_np)

        self.model.zero_grad(set_to_none=True)
        return np.clip(cam_np, 0.0, 1.0).astype(np.float32)

    @staticmethod
    def estimate_lcr_from_cam(
        cam_heatmap: np.ndarray,
        bone_mask: Optional[np.ndarray] = None,
        intensity_threshold: float = 0.80,
    ) -> float:
        """
        Estimates Lesion-to-Cortex Ratio (LCR = D_lesion / D_bone)
        from top activation intensities (Stage 02 exploratory proxy).
        """
        if not isinstance(cam_heatmap, np.ndarray):
            raise TypeError(f"cam_heatmap must be numpy ndarray, got {type(cam_heatmap)}")

        if bone_mask is None:
            bone_mask = (cam_heatmap > 0.1).astype(np.uint8)

        active_vals = cam_heatmap[cam_heatmap > 0.05]
        if len(active_vals) == 0:
            return 0.0

        thresh = float(np.percentile(active_vals, intensity_threshold * 100))
        lesion_mask = (cam_heatmap >= thresh).astype(np.uint8)

        bone_rows = np.where(bone_mask.any(axis=1))[0]
        if len(bone_rows) == 0:
            return 0.0

        lesion_rows = np.where(lesion_mask.any(axis=1))[0]
        if len(lesion_rows) == 0:
            return 0.0

        max_lesion_w = 0
        max_bone_w = 1
        for r in lesion_rows:
            lesion_cols = np.where(lesion_mask[r] > 0)[0]
            bone_cols = np.where(bone_mask[r] > 0)[0]
            if len(lesion_cols) > 0 and len(bone_cols) > 0:
                l_w = lesion_cols[-1] - lesion_cols[0] + 1
                b_w = bone_cols[-1] - bone_cols[0] + 1
                if l_w > max_lesion_w:
                    max_lesion_w = l_w
                    max_bone_w = max(b_w, 1)

        lcr = float(max_lesion_w) / float(max(max_bone_w, 1))
        return float(min(1.0, max(0.0, round(lcr, 3))))
