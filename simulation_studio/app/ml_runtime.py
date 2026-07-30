from __future__ import annotations

import logging
import os
import sys
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

STUDIO_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = STUDIO_ROOT.parent
BACKEND_ROOT = REPO_ROOT / "backendMulti"

_ML_READY = False
_ML_ERROR: Exception | None = None


def _apply_runtime_env(env_overrides: dict[str, Any] | None) -> None:
    if not isinstance(env_overrides, dict):
        return
    for key, value in env_overrides.items():
        key_text = str(key).strip()
        if not key_text:
            continue
        os.environ.setdefault(key_text, str(value))


def ensure_ml_runtime(
    *,
    require_django_setup: bool = True,
    env_overrides: dict[str, Any] | None = None,
) -> bool:
    global _ML_READY, _ML_ERROR
    if _ML_READY:
        return True
    if _ML_ERROR is not None:
        return False
    if not BACKEND_ROOT.exists():
        _ML_ERROR = RuntimeError(f"Backend workspace unavailable at {BACKEND_ROOT}")
        logger.warning("ML runtime unavailable: %s", _ML_ERROR)
        return False
    try:
        _apply_runtime_env(env_overrides)
        if str(BACKEND_ROOT) not in sys.path:
            sys.path.insert(0, str(BACKEND_ROOT))
        if str(REPO_ROOT) not in sys.path:
            sys.path.insert(0, str(REPO_ROOT))
        ml_backend = REPO_ROOT / "ml-backend"
        if ml_backend.exists() and str(ml_backend) not in sys.path:
            sys.path.insert(0, str(ml_backend))

        os.environ.setdefault(
            "DJANGO_SETTINGS_MODULE",
            os.environ.get("DJANGO_SETTINGS_MODULE", "backendMulti.settings"),
        )
        models_dir = REPO_ROOT / "ml-backend" / "models"
        if models_dir.exists():
            os.environ.setdefault("PIOS_MODELS_DIR", str(models_dir))

        if require_django_setup:
            import django

            django.setup()
        _ML_READY = True
        return True
    except Exception as exc:
        _ML_ERROR = exc
        logger.warning("ML runtime bootstrap failed: %s", exc)
        return False


def ml_runtime_status() -> dict[str, Any]:
    return {
        "ready": _ML_READY,
        "error": str(_ML_ERROR) if _ML_ERROR else None,
        "backend_root": str(BACKEND_ROOT),
    }