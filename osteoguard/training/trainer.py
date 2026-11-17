"""
OsteoGuard AI - Training Engine with Readiness Gate Enforcement.
Implements the training loop optimized with Focal Loss (gamma = 2.0)
and AdamW, preceded by mandatory zero-dataset and patient-privacy verification.
"""

from pathlib import Path
from typing import Dict, Optional, Tuple
import torch
import torch.optim as optim
from torch.utils.data import DataLoader

from osteoguard.config import ModelConfig, TrainingConfig
from osteoguard.constants import NUM_CLASSES
from osteoguard.data.partitioning import PatientLevelPartitioner, SampleRecord
from osteoguard.data.readiness import DatasetReadinessGate, DatasetReadinessError
from osteoguard.data.dataset import OsteoGuardDataset
from osteoguard.models.loss import MulticlassFocalLoss
from osteoguard.models.osteoguard_net import OsteoGuardNet


class OsteoGuardTrainer:
    """
    Orchestrates verified training workflows for OsteoGuard AI.
    Enforces pre-flight readiness checks to prevent phantom or synthetic training.
    """

    def __init__(
        self,
        model_config: Optional[ModelConfig] = None,
        train_config: Optional[TrainingConfig] = None,
    ):
        self.model_config = model_config or ModelConfig()
        self.train_config = train_config or TrainingConfig()
        self.readiness_gate = DatasetReadinessGate()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def run_preflight_readiness_check(self, dataset_dir: Path):
        """
        Hard gate: Will raise DatasetReadinessError if real clinical data is absent.
        """
        return self.readiness_gate.enforce_readiness(dataset_dir)

    def build_model(self) -> OsteoGuardNet:
        """Instantiates the unified OsteoGuard dual-stream network."""
        model = OsteoGuardNet(
            backbone_name=self.model_config.backbone_name,
            pretrained_backbone=self.model_config.pretrained_backbone,
            in_channels=1,
            semantic_dim=self.model_config.semantic_dim,
            radiomics_dim=self.model_config.radiomics_dim,
            num_classes=self.model_config.num_classes,
            enable_unet_masking=self.model_config.enable_unet_masking,
        )
        return model.to(self.device)

    def train_epoch(
        self,
        model: OsteoGuardNet,
        dataloader: DataLoader,
        optimizer: optim.Optimizer,
        criterion: MulticlassFocalLoss,
    ) -> float:
        """Executes one training epoch."""
        model.train()
        total_loss = 0.0
        n_batches = 0

        for images, radiomics, targets, _ in dataloader:
            images = images.to(self.device)
            radiomics = radiomics.to(self.device)
            targets = targets.to(self.device)

            optimizer.zero_grad()
            outputs = model(images, radiomics)
            loss = criterion(outputs["logits"], targets)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), self.train_config.gradient_clip_norm
            )
            optimizer.step()

            total_loss += float(loss.item())
            n_batches += 1

        return total_loss / max(n_batches, 1)

    def evaluate(
        self,
        model: OsteoGuardNet,
        dataloader: DataLoader,
        criterion: MulticlassFocalLoss,
    ) -> Tuple[float, float]:
        """Evaluates loss and accuracy on validation/test loader."""
        model.eval()
        total_loss = 0.0
        correct = 0
        total_samples = 0

        with torch.no_grad():
            for images, radiomics, targets, _ in dataloader:
                images = images.to(self.device)
                radiomics = radiomics.to(self.device)
                targets = targets.to(self.device)

                outputs = model(images, radiomics)
                loss = criterion(outputs["logits"], targets)

                total_loss += float(loss.item())
                preds = torch.argmax(outputs["logits"], dim=1)
                correct += int((preds == targets).sum().item())
                total_samples += targets.size(0)

        avg_loss = total_loss / max(len(dataloader), 1)
        accuracy = float(correct) / max(total_samples, 1)
        return avg_loss, accuracy

    def train(self, dataset_dir: Path):
        """
        Main entrypoint. First checks readiness, then trains if and only if cleared.
        """
        # 1. Enforce readiness gate
        print(f"[OsteoGuard Trainer] Executing pre-flight readiness gate for: {dataset_dir}...")
        self.run_preflight_readiness_check(dataset_dir)

        # 2. Setup output directories
        self.train_config.artifacts_dir.mkdir(parents=True, exist_ok=True)
        self.train_config.checkpoints_dir.mkdir(parents=True, exist_ok=True)
        self.train_config.logs_dir.mkdir(parents=True, exist_ok=True)

        print(f"[OsteoGuard Trainer] Dataset cleared. Initializing model on device: {self.device}")
        model = self.build_model()
        optimizer = optim.AdamW(
            model.parameters(),
            lr=self.train_config.learning_rate,
            weight_decay=self.train_config.weight_decay,
        )
        criterion = MulticlassFocalLoss(gamma=self.train_config.focal_loss_gamma)

        # Training loop execution would proceed here with real DataLoader
        return model

