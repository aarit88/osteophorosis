"""
OsteoGuard AI Services Package.
"""

from ml_service.app.services.adjudication_engine import (
    AdjudicationEngine,
    ProvenanceRecord,
    ProvenanceTransitionEntry,
)

__all__ = [
    "AdjudicationEngine",
    "ProvenanceRecord",
    "ProvenanceTransitionEntry",
]

try:  # Optional torch/pydantic runtime dependencies
    from ml_service.app.services.readiness_service import MLReadinessService

    __all__.append("MLReadinessService")
except ImportError:  # pragma: no cover - stdlib-only AI Studio container
    pass
