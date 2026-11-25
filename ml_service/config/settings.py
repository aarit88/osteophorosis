"""
OsteoGuard AI - Configuration Management.
Centralized environment-driven settings using Pydantic V2 and pydantic-settings.
"""

from functools import lru_cache
import json
from pathlib import Path
from typing import Annotated, List, Optional
from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, NoDecode, SettingsConfigDict


class MLSettings(BaseSettings):
    """
    OsteoGuard AI ML Service Settings.
    Loaded from environment variables with prefix OSTEO_ or local .env file.
    """
    model_config = SettingsConfigDict(
        env_prefix="OSTEO_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Service Metadata
    service_name: str = Field(default="OsteoGuard AI ML Service", description="Human-readable service title")
    service_version: str = Field(default="1.0.0", description="SemVer release string")
    env: str = Field(default="development", description="Runtime environment: development | staging | production")
    host: str = Field(default="0.0.0.0", description="Network bind address")
    port: int = Field(default=8000, description="Listening TCP port")
    log_level: str = Field(default="INFO", description="Logging verbosity: DEBUG | INFO | WARNING | ERROR | CRITICAL")

    # File System Paths (Workspace-Root Relative)
    base_dir: Path = Field(default_factory=lambda: Path(__file__).resolve().parent.parent.parent)
    data_dir: Path = Field(default=Path("ml_service/data"))
    raw_data_dir: Path = Field(default=Path("ml_service/data/raw"))
    interim_data_dir: Path = Field(default=Path("ml_service/data/interim"))
    processed_data_dir: Path = Field(default=Path("ml_service/data/processed"))
    splits_data_dir: Path = Field(default=Path("ml_service/data/splits"))

    artifacts_dir: Path = Field(default=Path("ml_artifacts"))
    checkpoints_dir: Path = Field(default=Path("ml_artifacts/checkpoints"))
    metrics_dir: Path = Field(default=Path("ml_artifacts/metrics"))
    logs_dir: Path = Field(default=Path("ml_artifacts/logs"))
    stage_reports_dir: Path = Field(default=Path("ml_artifacts/stage_reports"))

    # Model Architecture Parameters
    active_backbone: str = Field(default="resnet101", description="resnet101 | swin_transformer")
    device: str = Field(default="cpu", description="Compute device: cpu | cuda")
    fusion_dim: int = Field(default=614, description="Late fusion input: 512 semantic + 102 radiomics")
    radiomics_dim: int = Field(default=102, description="IBSI radiomic feature count")
    semantic_dim: int = Field(default=512, description="Projected deep visual feature dimension")
    num_classes: int = Field(default=4, description="Differential diagnostic class count")
    target_classes: List[str] = Field(
        default_factory=lambda: ["metastasis", "paget", "oi", "healthy"]
    )
    min_patients_per_class: int = Field(default=10, description="Minimum unique patients per cohort for training clearance")

    # Active Checkpoint Configuration
    active_checkpoint_path: Optional[Path] = Field(
        default=None,
        description="Path to authenticated model checkpoint weights. None if untrained."
    )

    # Separate preliminary Track A X-ray classifier. This never supplies Track D inference.
    track_a_checkpoint_path: Path = Field(
        default=Path("fracatlas_full_training/best_checkpoint.pt"),
        description="Track A FracAtlas binary checkpoint, relative to artifacts_dir unless absolute.",
    )

    # Security & CORS
    cors_origins: Annotated[List[str], NoDecode] = Field(
        default_factory=lambda: [
            "http://localhost:5173",
            "http://localhost:3000",
            "http://127.0.0.1:5173",
            "http://127.0.0.1:3000",
        ]
    )

    @field_validator("cors_origins", mode="before")
    @classmethod
    def parse_cors_origins(cls, value):
        if isinstance(value, str):
            try:
                decoded = json.loads(value)
                if isinstance(decoded, list):
                    return [str(origin).strip() for origin in decoded if str(origin).strip()]
            except json.JSONDecodeError:
                pass
            return [origin.strip() for origin in value.split(",") if origin.strip()]
        return value

    # Operational Guardrails
    enable_mock_fallback: bool = Field(
        default=False,
        description="Strictly False in production. Never fake predictions when weights are missing."
    )

    def resolve_path(self, path: Path) -> Path:
        """Resolves path relative to base directory if not absolute."""
        if path.is_absolute():
            return path
        return (self.base_dir / path).resolve()

    def get_resolved_data_dir(self) -> Path:
        return self.resolve_path(self.data_dir)

    def get_resolved_raw_dir(self) -> Path:
        return self.resolve_path(self.raw_data_dir)

    def get_resolved_checkpoints_dir(self) -> Path:
        return self.resolve_path(self.checkpoints_dir)

    def get_resolved_artifacts_dir(self) -> Path:
        return self.resolve_path(self.artifacts_dir)


@lru_cache(maxsize=1)
def get_settings() -> MLSettings:
    """Singleton provider for cached ML configuration."""
    return MLSettings()
