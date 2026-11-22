"""
OsteoGuard AI Application Layer.
"""

__all__ = []

try:  # Optional FastAPI runtime dependency
    from ml_service.app.main import app, create_app

    __all__ = ["app", "create_app"]
except ImportError:  # pragma: no cover - stdlib-only AI Studio container
    pass
