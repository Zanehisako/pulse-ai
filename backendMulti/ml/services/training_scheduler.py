"""
Django bridge for the ML training scheduler.

Uses Django-local ML config under ``backendMulti/ml/config`` and attempts to
import the standalone ``pios_ml_backend.training_scheduler`` module only when
it is available. All public symbols are re-exported with safe fallbacks so
Django views can continue importing from this module without crashing.

When ``start_scheduler()`` is called through this bridge it automatically
registers a **post-sync hook** that calls
``ml.core.startup.refresh_runtime_state`` after every successful registry
sync. This ensures that newly-trained models are loaded into the running
Django process's in-memory ``ModelRegistry`` — no restart required.
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path
from typing import Any, Callable, cast

from django.conf import settings

logger = logging.getLogger("pios.training_scheduler_bridge")

_DJANGO_ML_CONFIG_DIR = Path(__file__).resolve().parents[1] / "config"
_ml_backend_dir = Path(
    getattr(
        settings,
        "ML_BACKEND_DIR",
        Path(__file__).resolve().parents[3] / "ml-backend",
    )
)
if _ml_backend_dir.exists() and str(_ml_backend_dir) not in sys.path:
    sys.path.insert(0, str(_ml_backend_dir))

_STANDALONE_SCHEDULER_AVAILABLE = False
_STANDALONE_IMPORT_ERROR: Exception | None = None

ScheduleConfigLoader = Callable[[], dict[str, Any]]
ScheduleStatusLoader = Callable[[], dict[str, Any]]
ScheduleRunner = Callable[..., dict[str, Any]]
PostSyncHookSetter = Callable[[Any], None]
SchedulerStarter = Callable[..., Any]
SchedulerStopper = Callable[[], None]


def _fallback_schedule_config_loader() -> dict[str, Any]:
    return _load_local_schedule_config()


def _fallback_schedule_status_loader() -> dict[str, Any]:
    config = _load_local_schedule_config()
    return {
        "enabled": bool(config.get("enabled", False)),
        "running": False,
        "available": False,
        "mode": "disabled",
        "source": config.get("source"),
        "reason": "standalone scheduler module unavailable",
    }


def _fallback_training_runner(*args: Any, **kwargs: Any) -> dict[str, Any]:
    return {
        "ok": False,
        "available": False,
        "message": "Training scheduler is unavailable because the standalone ML backend was removed.",
    }


def _fallback_drift_runner(*args: Any, **kwargs: Any) -> dict[str, Any]:
    return {
        "ok": False,
        "available": False,
        "message": "Drift detection is unavailable because the standalone ML backend was removed.",
    }


def _fallback_set_post_sync_hook(callback: Any) -> None:
    return None


def _fallback_start_scheduler(*args: Any, **kwargs: Any) -> None:
    return None


def _fallback_stop_scheduler() -> None:
    return None


_standalone_load_schedule_config: ScheduleConfigLoader = (
    _fallback_schedule_config_loader
)
_standalone_reload_schedule_config: ScheduleConfigLoader = (
    _fallback_schedule_config_loader
)
_standalone_get_schedule_status: ScheduleStatusLoader = _fallback_schedule_status_loader
_standalone_run_training_now: ScheduleRunner = _fallback_training_runner
_standalone_run_drift_now: ScheduleRunner = _fallback_drift_runner
_standalone_set_post_sync_hook: PostSyncHookSetter = _fallback_set_post_sync_hook
_standalone_start_scheduler: SchedulerStarter = _fallback_start_scheduler
_standalone_stop_scheduler: SchedulerStopper = _fallback_stop_scheduler

try:
    from pios_ml_backend import (
        training_scheduler as _standalone_scheduler_module,  # type: ignore[import-not-found]  # noqa: E402
    )

    _standalone_get_schedule_status = cast(
        ScheduleStatusLoader,
        _standalone_scheduler_module.get_schedule_status,
    )
    _standalone_load_schedule_config = cast(
        ScheduleConfigLoader,
        _standalone_scheduler_module.load_schedule_config,
    )
    _standalone_reload_schedule_config = cast(
        ScheduleConfigLoader,
        _standalone_scheduler_module.reload_schedule_config,
    )
    _standalone_run_drift_now = cast(
        ScheduleRunner,
        _standalone_scheduler_module.run_drift_now,
    )
    _standalone_run_training_now = cast(
        ScheduleRunner,
        _standalone_scheduler_module.run_training_now,
    )
    _standalone_set_post_sync_hook = cast(
        PostSyncHookSetter,
        _standalone_scheduler_module.set_post_sync_hook,
    )
    _standalone_start_scheduler = cast(
        SchedulerStarter,
        _standalone_scheduler_module.start_scheduler,
    )
    _standalone_stop_scheduler = cast(
        SchedulerStopper,
        _standalone_scheduler_module.stop_scheduler,
    )

    _STANDALONE_SCHEDULER_AVAILABLE = True
except (
    Exception
) as exc:  # pragma: no cover - exercised when standalone backend is removed
    _STANDALONE_IMPORT_ERROR = exc
    logger.info(
        "[scheduler-bridge] Standalone training scheduler unavailable; using safe fallbacks: %s",
        exc,
    )

__all__ = [
    "get_schedule_config_path",
    "get_schedule_status",
    "load_schedule_config",
    "reload_schedule_config",
    "run_drift_check",
    "run_drift_now",
    "run_training_now",
    "set_post_sync_hook",
    "start_scheduler",
    "stop_scheduler",
]


def get_schedule_config_path() -> Path:
    """Return the resolved path to ``training_schedule.json``."""
    configured = getattr(settings, "PIOS_TRAINING_SCHEDULE_PATH", None)
    if configured:
        return Path(configured).resolve()
    return (_DJANGO_ML_CONFIG_DIR / "training_schedule.json").resolve()


def _load_local_schedule_config() -> dict[str, Any]:
    path = get_schedule_config_path()
    if not path.exists():
        return {
            "enabled": False,
            "jobs": [],
            "source": str(path),
            "available": False,
        }

    try:
        data = json.loads(path.read_text("utf-8"))
    except Exception as exc:
        logger.warning(
            "[scheduler-bridge] Failed to read local training schedule config %s: %s",
            path,
            exc,
        )
        return {
            "enabled": False,
            "jobs": [],
            "source": str(path),
            "available": False,
            "error": str(exc),
        }

    if isinstance(data, dict):
        data.setdefault("source", str(path))
        data.setdefault("available", True)
        data.setdefault("jobs", [])
        return data

    return {
        "enabled": False,
        "jobs": [],
        "source": str(path),
        "available": False,
        "error": "Invalid schedule config format",
    }


def load_schedule_config() -> dict[str, Any]:
    return _standalone_load_schedule_config()


def reload_schedule_config() -> dict[str, Any]:
    return _standalone_reload_schedule_config()


def get_schedule_status() -> dict[str, Any]:
    return _standalone_get_schedule_status()


def run_training_now(*args: Any, **kwargs: Any) -> dict[str, Any]:
    return _standalone_run_training_now(*args, **kwargs)


def run_drift_now(*args: Any, **kwargs: Any) -> dict[str, Any]:
    return _standalone_run_drift_now(*args, **kwargs)


def set_post_sync_hook(callback: Any) -> None:
    _standalone_set_post_sync_hook(callback)


def stop_scheduler() -> None:
    _standalone_stop_scheduler()


def run_drift_check(model_name: str | None = None) -> dict:
    """Run drift detection — callable from Django views."""
    return run_drift_now(model_name=model_name)


# ---------------------------------------------------------------------------
# Post-sync hook: refresh Django's in-memory model registry
# ---------------------------------------------------------------------------


def _django_post_sync_hook() -> None:
    """
    Called automatically after every successful MLflow → Django DB sync.

    Triggers a full refresh of the in-memory ``ModelRegistry`` and
    orchestrator catalog so that the newly-trained models are served
    immediately without a process restart.
    """
    try:
        from ml.core.startup import is_ready, refresh_runtime_state

        if not is_ready():
            logger.info(
                "[scheduler-bridge] ML stack not yet initialised — "
                "skipping in-memory refresh (models will load on first request)"
            )
            return

        changes = refresh_runtime_state("post-training-sync", force=True, warmup=False)
        logger.info(
            "[scheduler-bridge] In-memory refresh complete — "
            "runtime_changed=%s  orchestrator_changed=%s",
            changes.get("runtime_config_changed"),
            changes.get("orchestrator_catalog_changed"),
        )
    except Exception:
        logger.exception(
            "[scheduler-bridge] Failed to refresh in-memory model registry"
        )


# ---------------------------------------------------------------------------
# Wrapped start_scheduler that auto-registers the Django hook
# ---------------------------------------------------------------------------


def start_scheduler(config_path: Path | None = None, **kwargs: Any):
    """
    Start the background training scheduler **and** register the Django
    post-sync hook so that the in-memory model registry is refreshed
    after every successful training + sync cycle.

    Accepts the same arguments as the underlying
    ``pios_ml_backend.training_scheduler.start_scheduler``.
    """
    # Always register the Django hook unless the caller explicitly passes one
    if "post_sync_hook" not in kwargs:
        kwargs["post_sync_hook"] = _django_post_sync_hook

    scheduler = _standalone_start_scheduler(config_path=config_path, **kwargs)
    logger.info(
        "[scheduler-bridge] Scheduler started with Django post-sync hook registered"
    )
    return scheduler
