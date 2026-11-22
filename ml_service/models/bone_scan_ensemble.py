"""
OsteoGuard AI - Bone Scan Five-Model Ensemble Classifier.

Constructs a robust patient-level ensemble combining the five Stage 12 cross-validation
ResNet-18 checkpoints.

Aggregation protocol:
1. For each constituent model k (k in 0..4):
     P_ant_k  = sigmoid(model_k(anterior_view))
     P_post_k = sigmoid(model_k(posterior_view))
     P_patient_k = max(P_ant_k, P_post_k)
2. Ensemble prediction:
     P_ensemble = mean([P_patient_0, ..., P_patient_4])
3. Model disagreement indicator:
     std_ensemble = std([P_patient_0, ..., P_patient_4])
     min_prob     = min([P_patient_0, ..., P_patient_4])
     max_prob     = max([P_patient_0, ..., P_patient_4])
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import numpy as np
import torch
import torch.nn as nn
from torch.amp import autocast

from ml_service.models.bone_scan_classifier import BoneScanResNet18


class BoneScanEnsemble(nn.Module):
    """
    Five-fold ResNet-18 ensemble for planar bone scintigraphy metastasis screening.
    """

    def __init__(
        self,
        checkpoint_paths: Optional[List[Union[str, Path]]] = None,
        checkpoint_dir: Optional[Union[str, Path]] = None,
        device: Optional[torch.device] = None,
    ):
        super().__init__()
        self.device = device or (torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu"))

        # Resolve checkpoint paths
        if checkpoint_paths is None:
            if checkpoint_dir is None:
                raise ValueError("Must provide either checkpoint_paths or checkpoint_dir")
            c_dir = Path(checkpoint_dir)
            checkpoint_paths = [c_dir / f"fold_{i}/best_checkpoint.pt" for i in range(5)]

        self.checkpoint_paths = [Path(p) for p in checkpoint_paths]
        assert len(self.checkpoint_paths) == 5, f"Expected 5 checkpoints, got {len(self.checkpoint_paths)}"

        self.models = nn.ModuleList()
        self.checkpoint_metadata = []

        for i, ckpt_path in enumerate(self.checkpoint_paths):
            if not ckpt_path.exists():
                raise FileNotFoundError(f"Ensemble checkpoint missing: {ckpt_path}")

            # Instantiate ResNet-18 without downloading ImageNet weights again
            model = BoneScanResNet18(pretrained=False)
            ckpt = torch.load(ckpt_path, map_location=self.device, weights_only=False)

            if "model_state_dict" in ckpt:
                state_dict = ckpt["model_state_dict"]
            else:
                state_dict = ckpt

            model.load_state_dict(state_dict)
            model.to(self.device)
            model.eval()

            self.models.append(model)
            self.checkpoint_metadata.append({
                "fold_index": i,
                "fold_name": f"fold_{i}",
                "path": str(ckpt_path),
                "epoch": ckpt.get("epoch", -1),
                "val_pr_auc": ckpt.get("val_patient_pr_auc", None),
                "val_roc_auc": ckpt.get("val_patient_roc_auc", None),
            })

    def forward(
        self,
        anterior_tensor: torch.Tensor,
        posterior_tensor: torch.Tensor,
    ) -> Dict[str, Any]:
        """
        Runs patient-level ensemble forward inference.

        Args:
            anterior_tensor: Shape (1, 1, 1024, 256) or (B, 1, 1024, 256).
            posterior_tensor: Shape (1, 1, 1024, 256) or (B, 1, 1024, 256).

        Returns:
            Dictionary containing:
                - prob_patient_ensemble: Float (mean patient probability)
                - std_disagreement: Float (standard deviation across models)
                - min_patient_prob: Float
                - max_patient_prob: Float
                - fold_patient_probs: List[float] (per-model max probabilities)
                - fold_anterior_probs: List[float]
                - fold_posterior_probs: List[float]
                - prob_anterior_mean: Float
                - prob_posterior_mean: Float
        """
        # Ensure batch dimension
        if anterior_tensor.dim() == 3:
            anterior_tensor = anterior_tensor.unsqueeze(0)
        if posterior_tensor.dim() == 3:
            posterior_tensor = posterior_tensor.unsqueeze(0)

        ant = anterior_tensor.to(self.device)
        post = posterior_tensor.to(self.device)

        pair_imgs = torch.cat([ant, post], dim=0)  # (2, 1, 1024, 256)

        fold_patient_probs = []
        fold_ant_probs = []
        fold_post_probs = []

        is_cuda = self.device.type == "cuda"

        with torch.no_grad():
            for model in self.models:
                if is_cuda:
                    with autocast("cuda", dtype=torch.float16):
                        logits = model(pair_imgs)
                else:
                    logits = model(pair_imgs)

                probs = torch.sigmoid(logits).cpu().numpy().flatten()
                p_ant = float(probs[0])
                p_post = float(probs[1])
                p_patient = max(p_ant, p_post)

                fold_ant_probs.append(p_ant)
                fold_post_probs.append(p_post)
                fold_patient_probs.append(p_patient)

        p_ensemble = float(np.mean(fold_patient_probs))
        std_disagree = float(np.std(fold_patient_probs))
        min_p = float(np.min(fold_patient_probs))
        max_p = float(np.max(fold_patient_probs))

        p_ant_mean = float(np.mean(fold_ant_probs))
        p_post_mean = float(np.mean(fold_post_probs))

        return {
            "prob_patient_ensemble": p_ensemble,
            "std_disagreement": std_disagree,
            "min_patient_prob": min_p,
            "max_patient_prob": max_p,
            "fold_patient_probs": fold_patient_probs,
            "fold_anterior_probs": fold_ant_probs,
            "fold_posterior_probs": fold_post_probs,
            "prob_anterior_mean": p_ant_mean,
            "prob_posterior_mean": p_post_mean,
        }

    def get_ensemble_metadata(self) -> Dict[str, Any]:
        """Returns ensemble composition details and checkpoint provenance."""
        total_params_per_model = sum(p.numel() for p in self.models[0].parameters())
        return {
            "num_models": len(self.models),
            "architecture": "BoneScanResNet18",
            "backbone": "ResNet18",
            "aggregation": "mean(max(P_ant_k, P_post_k)) across k in 0..4",
            "disagreement_metric": "std(max(P_ant_k, P_post_k)) across k in 0..4",
            "total_parameters_per_model": total_params_per_model,
            "total_ensemble_parameters": total_params_per_model * len(self.models),
            "device": str(self.device),
            "checkpoints": self.checkpoint_metadata,
        }

