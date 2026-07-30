from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Callable

from django.db import transaction
from django.utils import timezone

from ml.core.utils import safe_slug
from ml.models import MLModelConfig

logger = logging.getLogger(__name__)

MODEL_CONFIG_REQUIRED_FIELDS = {
    "id",
    "description",
    "file_path",
    "type",
    "enabled",
    "features",
    "feature_info",
    "examples",
    "defaults",
}


class ModelConfigSyncError(ValueError):
    """Raised when the model config file cannot be validated or synced."""


def _source_path(path: str | Path) -> Path:
    return Path(path).expanduser().resolve()


def load_model_config_payload(path: str | Path) -> dict[str, Any]:
    config_path = _source_path(path)
    try:
        payload = json.loads(config_path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ModelConfigSyncError(f"Model config file not found: {config_path}") from exc
    except json.JSONDecodeError as exc:
        raise ModelConfigSyncError(f"Invalid model config JSON: {config_path}") from exc
    except OSError as exc:
        raise ModelConfigSyncError(f"Unable to read model config file: {config_path}") from exc
    if not isinstance(payload, dict):
        raise ModelConfigSyncError("Model config root must be a JSON object.")
    return payload


def _supported_model_types() -> set[str]:
    return {choice[0] for choice in MLModelConfig.MODEL_TYPES}


def _validate_model_row(row: Any, *, index: int, seen_ids: set[str]) -> dict[str, Any]:
    if not isinstance(row, dict):
        raise ModelConfigSyncError(f"models[{index}] must be an object.")

    missing = sorted(MODEL_CONFIG_REQUIRED_FIELDS - set(row))
    if missing:
        label = row.get("id") or f"models[{index}]"
        raise ModelConfigSyncError(
            f"Model config '{label}' missing required fields: {', '.join(missing)}"
        )

    model_id = str(row.get("id") or "").strip()
    model_key = safe_slug(model_id)
    if not model_id or not model_key:
        raise ModelConfigSyncError(f"models[{index}].id must be non-empty.")
    if model_key in seen_ids:
        raise ModelConfigSyncError(f"Duplicate model id in config: {model_id}")
    seen_ids.add(model_key)

    model_type = str(row.get("type") or "").strip()
    if model_type not in _supported_model_types():
        raise ModelConfigSyncError(f"Unsupported model type for {model_id}: {model_type}")

    enabled = row.get("enabled")
    if not isinstance(enabled, bool):
        raise ModelConfigSyncError(f"Model '{model_id}' enabled must be boolean.")

    file_path = str(row.get("file_path") or "").strip()
    if not file_path:
        raise ModelConfigSyncError(f"Model '{model_id}' file_path must be non-empty.")

    features = row.get("features")
    if not isinstance(features, list) or not all(
        isinstance(item, str) and item.strip() for item in features
    ):
        raise ModelConfigSyncError(f"Model '{model_id}' features must be a list of strings.")

    feature_info = row.get("feature_info")
    if not isinstance(feature_info, dict):
        raise ModelConfigSyncError(f"Model '{model_id}' feature_info must be an object.")

    examples = row.get("examples")
    if not isinstance(examples, list) or not all(isinstance(item, dict) for item in examples):
        raise ModelConfigSyncError(f"Model '{model_id}' examples must be a list of objects.")

    defaults = row.get("defaults")
    if not isinstance(defaults, dict):
        raise ModelConfigSyncError(f"Model '{model_id}' defaults must be an object.")

    return {
        "model_id": model_id,
        "description": str(row.get("description") or "").strip(),
        "file_path": file_path,
        "model_type": model_type,
        "features": [item.strip() for item in features],
        "feature_info": dict(feature_info),
        "examples": list(examples),
        "defaults": dict(defaults),
        "is_active": enabled,
    }


def validate_model_config_payload(payload: dict[str, Any]) -> list[dict[str, Any]]:
    models = payload.get("models")
    if not isinstance(models, list):
        raise ModelConfigSyncError("Model config must define a models list.")
    seen_ids: set[str] = set()
    return [
        _validate_model_row(row, index=index, seen_ids=seen_ids)
        for index, row in enumerate(models)
    ]


def _row_changed(obj: MLModelConfig, defaults: dict[str, Any]) -> bool:
    return any(getattr(obj, key) != value for key, value in defaults.items())


def sync_model_config_file(
    path: str | Path,
    *,
    reason: str,
    model_manager=None,
    atomic_context: Callable[[], Any] | None = None,
) -> dict[str, Any]:
    config_path = _source_path(path)
    payload = load_model_config_payload(config_path)
    rows = validate_model_config_payload(payload)
    source = str(config_path)
    model_ids = [row["model_id"] for row in rows]
    manager = model_manager or MLModelConfig.objects
    atomic = atomic_context or transaction.atomic

    created = 0
    updated = 0
    deactivated = 0

    with atomic():
        existing = {
            obj.model_id: obj
            for obj in manager.select_for_update().filter(
                model_id__in=model_ids
            )
        }
        for row in rows:
            defaults = {
                "description": row["description"],
                "file_path": row["file_path"],
                "model_type": row["model_type"],
                "features": row["features"],
                "feature_info": row["feature_info"],
                "examples": row["examples"],
                "defaults": row["defaults"],
                "is_active": row["is_active"],
                "config_source_path": source,
            }
            obj = existing.get(row["model_id"])
            if obj is None:
                manager.create(model_id=row["model_id"], **defaults)
                created += 1
                continue
            if _row_changed(obj, defaults):
                for key, value in defaults.items():
                    setattr(obj, key, value)
                obj.save(update_fields=[*defaults.keys(), "updated_at"])
                updated += 1

        removed_qs = manager.select_for_update().filter(
            config_source_path=source,
            is_active=True,
        ).exclude(model_id__in=model_ids)
        deactivated = removed_qs.update(is_active=False, updated_at=timezone.now())

    summary = {
        "reason": reason,
        "source_path": source,
        "changed": bool(created or updated or deactivated),
        "models_total": len(rows),
        "active_models_total": sum(1 for row in rows if row["is_active"]),
        "created_total": created,
        "updated_total": updated,
        "deactivated_removed_total": deactivated,
    }
    logger.info("Model config file sync: %s", summary)
    return summary
