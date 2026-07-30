"""
Singleton initialization.
"""

import logging
import os
import threading
import time
import warnings

from django.conf import settings
from django.db import close_old_connections

logger = logging.getLogger("pios.startup")

_lock = threading.Lock()
_registry = None
_orchestrator = None
_initialized = False
_initializing = False
_runtime_config_status_lock = threading.Lock()
_orchestrator_config_status_lock = threading.Lock()
_model_config_watcher_lock = threading.Lock()
_model_config_watcher = None

_runtime_config_status = {
    "source": "django_db",
    "config_key": "ml_model_config",
    "model_config_file": str(settings.PIOS_MODEL_CONFIG_PATH),
    "last_source_error": None,
    "last_file_sync": None,
    "last_reload_reason": "startup",
    "last_reload_changed": False,
    "last_reload_at": None,
    "reload_count": 0,
    "last_discovery": None,
}

_orchestrator_config_status = {
    "source": "django_db",
    "config_key": "ml_orchestrator_variant",
    "last_source_error": None,
    "last_reload_reason": "startup",
    "last_reload_changed": False,
    "last_reload_at": None,
    "reload_count": 0,
    "selected_model_id": None,
}


def _load_models_from_db() -> list[dict]:
    from ml.models import MLModelConfig

    rows = [
        obj.to_registry_dict() for obj in MLModelConfig.objects.filter(is_active=True)
    ]
    logger.info("Loaded %s model configs from DB.", len(rows))
    return rows


def _sync_discovered_models(reason: str) -> dict:
    from ml.core.model_discovery import sync_discovered_model_configs

    summary = sync_discovered_model_configs(reason)
    with _runtime_config_status_lock:
        _runtime_config_status["last_discovery"] = summary
    return summary


def _iso_or_none(value):
    return value.isoformat() if value is not None else None


def _runtime_config_snapshot() -> dict:
    with _runtime_config_status_lock:
        return dict(_runtime_config_status)


def _orchestrator_config_snapshot() -> dict:
    with _orchestrator_config_status_lock:
        return dict(_orchestrator_config_status)


def _record_runtime_source_error(error: str | None) -> None:
    with _runtime_config_status_lock:
        _runtime_config_status["last_source_error"] = error


def _record_file_sync(summary: dict) -> None:
    with _runtime_config_status_lock:
        _runtime_config_status["last_file_sync"] = summary


def _load_orchestrator_catalog_from_db() -> dict:
    from ml.core.utils import safe_slug
    from ml.models import OrchestratorVariant

    rows = []
    selected_model_id = None
    for variant in OrchestratorVariant.objects.all().order_by("variant_id"):
        payload = {
            "id": variant.variant_id,
            "name": variant.name,
            "description": variant.description,
            "repo_id": variant.repo_id,
            "filename": variant.filename,
            "size_mb": variant.size_mb,
            "size_bytes": variant.size_bytes,
            "min_ram_gb": variant.min_ram_gb,
            "min_vram_mb": variant.min_vram_mb,
            "n_ctx": variant.n_ctx,
            "n_batch": variant.n_batch,
        }
        if isinstance(variant.metadata_extra, dict):
            payload.update(variant.metadata_extra)
        rows.append(payload)
        if variant.is_selected:
            selected_model_id = safe_slug(variant.variant_id)

    return {
        "models": rows,
        "selected_model_id": selected_model_id,
    }


def _apply_orchestrator_catalog(
    orchestrator,
    reason: str,
    *,
    force: bool = False,
    warmup: bool = False,
    wait: bool = False,
) -> bool:
    payload = _load_orchestrator_catalog_from_db()
    changed = orchestrator.update_model_catalog(
        payload.get("models", []),
        selected_model_id=payload.get("selected_model_id"),
        force=force,
        warmup=warmup,
        wait=wait,
    )
    snapshot = payload.get("selected_model_id") or orchestrator.preferred_model_id
    with _orchestrator_config_status_lock:
        if changed:
            _orchestrator_config_status["reload_count"] = (
                int(_orchestrator_config_status.get("reload_count", 0)) + 1
            )
        _orchestrator_config_status["last_reload_reason"] = reason
        _orchestrator_config_status["last_reload_changed"] = changed
        _orchestrator_config_status["last_reload_at"] = time.time()
        _orchestrator_config_status["selected_model_id"] = snapshot
        _orchestrator_config_status["last_source_error"] = None
    return changed


def initialize():
    global _registry, _orchestrator, _initialized, _initializing

    with _lock:
        if _initialized or _initializing:
            return
        _initializing = True

    try:
        from ml.core.registry import ModelRegistry
        from ml.orchestrator.service import DynamicXLAMOrchestrator

        logger.info("Initializing ModelRegistry.")

        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message="Accessing the database during app initialization",
                category=RuntimeWarning,
            )
            _sync_model_config_file("startup", path=settings.PIOS_MODEL_CONFIG_PATH)
            _sync_discovered_models("startup")
            registry = ModelRegistry(
                models_dir=settings.PIOS_MODELS_DIR,
                config_path=settings.PIOS_MODEL_CONFIG_PATH,
                config_loader=_load_models_from_db,
            )
            registry.refresh(force=True)

        loaded = registry.loaded()
        logger.info("Loaded %s models from %s.", len(loaded), settings.PIOS_MODELS_DIR)
        for m in loaded:
            logger.debug("Loaded model: %s (%s)", m.model_id, m.model_type)

        logger.info("Initializing xLAM orchestrator.")
        orchestrator = DynamicXLAMOrchestrator(registry=registry)
        _apply_orchestrator_catalog(
            orchestrator,
            "startup",
            force=True,
            warmup=False,
            wait=False,
        )
        orchestrator.initialize_on_startup()

        status = orchestrator.status()
        logger.info(
            "Orchestrator ready. LLM ready=%s mode=%s",
            status["llm_ready"],
            status.get("planner_mode", "unknown"),
        )

        # Start the background training scheduler
        try:
            from ml.services.training_scheduler import start_scheduler

            start_scheduler()
            logger.info("Training scheduler started successfully.")
        except Exception as exc:
            logger.warning("Could not start training scheduler: %s", exc)

        with _lock:
            _registry = registry
            _orchestrator = orchestrator
            _initialized = True

        start_model_config_watcher()
    finally:
        with _lock:
            _initializing = False


def is_ready() -> bool:
    return _initialized


def get_registry():
    if not _initialized:
        initialize()
    return _registry


def get_orchestrator():
    if not _initialized:
        initialize()
    return _orchestrator


def refresh_registry(reason: str, *, force: bool = False) -> bool:
    registry = get_registry()
    discovery = _sync_discovered_models(reason)
    changed = registry.refresh(force=force)
    changed = bool(changed or discovery.get("changed"))
    if changed or force:
        try:
            from ml.core.prediction import flush_mlflow_cache

            flush_mlflow_cache()
        except Exception:
            pass
    with _runtime_config_status_lock:
        if changed:
            _runtime_config_status["reload_count"] = (
                int(_runtime_config_status.get("reload_count", 0)) + 1
            )
        _runtime_config_status["last_reload_reason"] = reason
        _runtime_config_status["last_reload_changed"] = changed
        _runtime_config_status["last_reload_at"] = time.time()
        _runtime_config_status["last_source_error"] = None
    return changed


def sync_orchestrator_catalog(
    reason: str,
    *,
    force: bool = False,
    warmup: bool = False,
    wait: bool = False,
) -> bool:
    return _apply_orchestrator_catalog(
        get_orchestrator(),
        reason,
        force=force,
        warmup=warmup,
        wait=wait,
    )


def refresh_runtime_state(
    reason: str,
    *,
    force: bool = False,
    warmup: bool = False,
    wait: bool = False,
) -> dict:
    runtime_changed = refresh_registry(reason, force=force)
    orchestrator_changed = sync_orchestrator_catalog(
        reason,
        force=force,
        warmup=warmup,
        wait=wait,
    )
    return {
        "runtime_config_changed": runtime_changed,
        "orchestrator_catalog_changed": orchestrator_changed,
    }


def _sync_model_config_file(reason: str, *, path=None) -> dict:
    from ml.core.model_config_sync import sync_model_config_file

    try:
        summary = sync_model_config_file(
            path or settings.PIOS_MODEL_CONFIG_PATH,
            reason=reason,
        )
    except Exception as exc:
        _record_runtime_source_error(f"{type(exc).__name__}: {exc}")
        raise
    _record_file_sync(summary)
    _record_runtime_source_error(None)
    return summary


def reload_orchestrator_external_tools(reason: str) -> dict:
    orchestrator = get_orchestrator()
    reload_fn = getattr(orchestrator, "reload_external_tools_config", None)
    if not callable(reload_fn):
        return {
            "available": False,
            "changed": False,
            "reason": reason,
        }
    return reload_fn(reason=reason)


def reload_model_config_file(reason: str, *, path=None) -> dict:
    file_sync = _sync_model_config_file(reason, path=path)
    changes = refresh_runtime_state(reason, force=True, warmup=False)
    external_tool_reload = reload_orchestrator_external_tools(reason)
    config_hot_reload = reconcile_model_config_watcher()
    return {
        "file_sync": file_sync,
        "runtime_config_changed": changes["runtime_config_changed"],
        "orchestrator_catalog_changed": changes["orchestrator_catalog_changed"],
        "external_tool_reload": external_tool_reload,
        "config_hot_reload": config_hot_reload,
    }


def _watcher_reload_callback(path) -> dict:
    close_old_connections()
    try:
        return reload_model_config_file("model-config-file-watch", path=path)
    finally:
        close_old_connections()


def start_model_config_watcher() -> dict:
    global _model_config_watcher
    from ml.core.model_config_hot_reload import (
        ModelConfigHotReloadWatcher,
        load_hot_reload_options,
    )

    with _model_config_watcher_lock:
        if os.environ.get("DJANGO_SKIP_ML_INIT") == "1":
            _model_config_watcher = None
            return {
                "watched_path": str(settings.PIOS_MODEL_CONFIG_PATH),
                "running": False,
                "enabled": False,
                "last_file_reload_at": None,
                "last_file_error": None,
                "poll_interval_seconds": None,
                "debounce_seconds": None,
                "last_signature": None,
            }
        if _model_config_watcher is not None:
            status = _model_config_watcher.status()
            if status.get("running"):
                return status

        options = load_hot_reload_options(settings.PIOS_MODEL_CONFIG_PATH)
        if not options["enabled"]:
            _model_config_watcher = None
            return {
                "watched_path": str(settings.PIOS_MODEL_CONFIG_PATH),
                "running": False,
                "enabled": False,
                "last_file_reload_at": None,
                "last_file_error": None,
                "poll_interval_seconds": options["poll_interval_seconds"],
                "debounce_seconds": options["debounce_seconds"],
                "last_signature": None,
            }

        watcher = ModelConfigHotReloadWatcher(
            path=settings.PIOS_MODEL_CONFIG_PATH,
            poll_interval_seconds=options["poll_interval_seconds"],
            debounce_seconds=options["debounce_seconds"],
            reload_callback=_watcher_reload_callback,
        )
        watcher.prime()
        watcher.start()
        _model_config_watcher = watcher
        return watcher.status()


def reconcile_model_config_watcher() -> dict:
    global _model_config_watcher
    from ml.core.model_config_hot_reload import (
        ModelConfigHotReloadWatcher,
        load_hot_reload_options,
    )

    with _model_config_watcher_lock:
        if os.environ.get("DJANGO_SKIP_ML_INIT") == "1":
            if _model_config_watcher is not None:
                _model_config_watcher.stop()
            _model_config_watcher = None
            return {
                "watched_path": str(settings.PIOS_MODEL_CONFIG_PATH),
                "running": False,
                "enabled": False,
                "last_file_reload_at": None,
                "last_file_error": None,
                "poll_interval_seconds": None,
                "debounce_seconds": None,
                "last_signature": None,
            }

        try:
            options = load_hot_reload_options(settings.PIOS_MODEL_CONFIG_PATH)
        except Exception as exc:  # noqa: BLE001
            return {
                "watched_path": str(settings.PIOS_MODEL_CONFIG_PATH),
                "running": False,
                "enabled": False,
                "last_file_reload_at": None,
                "last_file_error": f"{type(exc).__name__}: {exc}",
                "poll_interval_seconds": None,
                "debounce_seconds": None,
                "last_signature": None,
            }

        watcher = _model_config_watcher
        if not options["enabled"]:
            if watcher is not None:
                watcher.stop()
            _model_config_watcher = None
            return {
                "watched_path": str(settings.PIOS_MODEL_CONFIG_PATH),
                "running": False,
                "enabled": False,
                "last_file_reload_at": None,
                "last_file_error": None,
                "poll_interval_seconds": options["poll_interval_seconds"],
                "debounce_seconds": options["debounce_seconds"],
                "last_signature": None,
            }

        if watcher is not None:
            status = watcher.status()
            same_options = (
                float(status.get("poll_interval_seconds") or 0)
                == float(options["poll_interval_seconds"])
                and float(status.get("debounce_seconds") or 0)
                == float(options["debounce_seconds"])
            )
            if status.get("running") and same_options:
                status["enabled"] = True
                return status
            watcher.stop()

        watcher = ModelConfigHotReloadWatcher(
            path=settings.PIOS_MODEL_CONFIG_PATH,
            poll_interval_seconds=options["poll_interval_seconds"],
            debounce_seconds=options["debounce_seconds"],
            reload_callback=_watcher_reload_callback,
        )
        watcher.prime()
        watcher.start()
        _model_config_watcher = watcher
        status = watcher.status()
        status["enabled"] = True
        return status


def get_model_config_watcher_status() -> dict:
    from ml.core.model_config_hot_reload import load_hot_reload_options

    if os.environ.get("DJANGO_SKIP_ML_INIT") == "1":
        return {
            "watched_path": str(settings.PIOS_MODEL_CONFIG_PATH),
            "running": False,
            "enabled": False,
            "last_file_reload_at": None,
            "last_file_error": None,
            "poll_interval_seconds": None,
            "debounce_seconds": None,
            "last_signature": None,
        }

    with _model_config_watcher_lock:
        watcher = _model_config_watcher
    if watcher is None:
        try:
            options = load_hot_reload_options(settings.PIOS_MODEL_CONFIG_PATH)
        except Exception as exc:  # noqa: BLE001
            return {
                "watched_path": str(settings.PIOS_MODEL_CONFIG_PATH),
                "running": False,
                "enabled": False,
                "last_file_reload_at": None,
                "last_file_error": f"{type(exc).__name__}: {exc}",
                "poll_interval_seconds": None,
                "debounce_seconds": None,
                "last_signature": None,
            }
        return {
            "watched_path": str(settings.PIOS_MODEL_CONFIG_PATH),
            "running": False,
            "enabled": options["enabled"],
            "last_file_reload_at": None,
            "last_file_error": None,
            "poll_interval_seconds": options["poll_interval_seconds"],
            "debounce_seconds": options["debounce_seconds"],
            "last_signature": None,
        }
    status = watcher.status()
    status["enabled"] = True
    return status


def get_runtime_config_status() -> dict:
    status = _runtime_config_snapshot()
    status["config_hot_reload"] = get_model_config_watcher_status()
    try:
        from ml.models import MLModelConfig

        queryset = MLModelConfig.objects.all()
        latest_updated_at = (
            queryset.order_by("-updated_at")
            .values_list(
                "updated_at",
                flat=True,
            )
            .first()
        )
        status.update(
            {
                "models_total": queryset.count(),
                "active_models_total": queryset.filter(is_active=True).count(),
                "last_updated_at": _iso_or_none(latest_updated_at),
                "db_ready": True,
            }
        )
    except Exception as exc:
        status.update(
            {
                "db_ready": False,
                "last_source_error": f"{type(exc).__name__}: {exc}",
            }
        )
    return status


def get_orchestrator_config_status() -> dict:
    status = _orchestrator_config_snapshot()
    try:
        from ml.core.utils import safe_slug
        from ml.models import OrchestratorVariant

        queryset = OrchestratorVariant.objects.all()
        latest_updated_at = (
            queryset.order_by("-updated_at")
            .values_list(
                "updated_at",
                flat=True,
            )
            .first()
        )
        selected_variant = (
            queryset.filter(is_selected=True)
            .values_list(
                "variant_id",
                flat=True,
            )
            .first()
        )
        status.update(
            {
                "variants_total": queryset.count(),
                "last_updated_at": _iso_or_none(latest_updated_at),
                "selected_model_id": (
                    safe_slug(selected_variant)
                    if selected_variant
                    else status.get("selected_model_id")
                ),
                "db_ready": True,
            }
        )
    except Exception as exc:
        status.update(
            {
                "db_ready": False,
                "last_source_error": f"{type(exc).__name__}: {exc}",
            }
        )
    return status
