from __future__ import annotations

import fnmatch
import json
import logging
import os
from pathlib import Path
from typing import Any

from django.conf import settings
from django.db import transaction

from ml.core.utils import safe_slug
from ml.models import MLModelConfig

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RUNTIME_CONFIG_PATH = PROJECT_ROOT / "ml" / "config" / "prediction_runtime.json"
IMPLEMENTED_ADAPTERS = {"sota_joblib"}
CATALOG_REQUIRED_FIELDS = {
    "id",
    "enabled",
    "registered_model_name",
    "artifact_filename",
    "task_type",
    "features",
    "outputs",
}


class ModelDiscoveryConfigError(ValueError):
    """Raised when local model discovery config cannot be used."""


def _deep_merge_dicts(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge_dicts(merged[key], value)
        else:
            merged[key] = value
    return merged


def _runtime_config_path(path: str | Path | None = None) -> Path:
    if path is not None:
        return Path(path).expanduser().resolve()
    configured = os.getenv("PIOS_PREDICTION_RUNTIME_CONFIG", "").strip()
    if configured:
        return Path(configured).expanduser().resolve()
    return DEFAULT_RUNTIME_CONFIG_PATH.resolve()


def _model_config_source_path(path: str | Path | None = None) -> str:
    configured = path or settings.PIOS_MODEL_CONFIG_PATH
    return str(Path(configured).expanduser().resolve())


def _read_json(path: Path, *, label: str) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ModelDiscoveryConfigError(f"{label} not found: {path}") from exc
    except json.JSONDecodeError as exc:
        raise ModelDiscoveryConfigError(f"Invalid {label} JSON: {path}") from exc
    if not isinstance(payload, dict):
        raise ModelDiscoveryConfigError(f"{label} root must be an object: {path}")
    return payload


def _resolve_path(raw: Any, *, base_dir: Path = PROJECT_ROOT) -> Path:
    text = str(raw or "").strip()
    if not text:
        raise ModelDiscoveryConfigError("Configured path cannot be empty.")
    path = Path(os.path.expandvars(text)).expanduser()
    if not path.is_absolute():
        path = base_dir / path
    return path.resolve()


def _model_discovery_config(config_path: str | Path | None = None) -> dict[str, Any]:
    payload = _read_json(_runtime_config_path(config_path), label="prediction runtime config")
    config = payload.get("model_discovery")
    return dict(config) if isinstance(config, dict) else {"enabled": False}


def _resolve_models_dir(config: dict[str, Any]) -> Path:
    env_name = str(config.get("models_dir_env") or "").strip()
    if env_name:
        env_value = os.getenv(env_name, "").strip()
        if env_value:
            path = Path(os.path.expandvars(env_value)).expanduser()
            return path.resolve()

    default_dir = config.get("default_models_dir") or getattr(settings, "PIOS_MODELS_DIR", "")
    return _resolve_path(default_dir)


def _clean_patterns(value: Any) -> list[str]:
    if isinstance(value, str):
        return [value.strip()] if value.strip() else []
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    return []


def _matches_any(path: Path, models_dir: Path, patterns: list[str]) -> bool:
    if not patterns:
        return False
    try:
        rel = path.relative_to(models_dir)
    except ValueError:
        rel = Path(path.name)
    rel_text = rel.as_posix()
    name = path.name
    return any(fnmatch.fnmatch(rel_text, pattern) or fnmatch.fnmatch(name, pattern) for pattern in patterns)


def _validate_adapter(adapter: str, supported: set[str]) -> None:
    if adapter not in supported:
        raise ModelDiscoveryConfigError(f"Unsupported model discovery adapter: {adapter}")
    if adapter not in IMPLEMENTED_ADAPTERS:
        raise ModelDiscoveryConfigError(f"Configured adapter has no runtime implementation: {adapter}")


def _clean_features(value: Any, *, label: str) -> list[str]:
    if not isinstance(value, list):
        raise ModelDiscoveryConfigError(f"{label} must be a list.")
    features = [str(item).strip() for item in value if str(item).strip()]
    if not features:
        raise ModelDiscoveryConfigError(f"{label} must not be empty.")
    return features


def _output_config(model: dict[str, Any]) -> dict[str, Any]:
    outputs = model.get("outputs")
    if not isinstance(outputs, dict):
        raise ModelDiscoveryConfigError("model outputs must be an object.")
    primary = str(outputs.get("primary") or "").strip()
    if not primary:
        raise ModelDiscoveryConfigError("model outputs.primary is required.")
    return dict(outputs)


def _artifact_allowed(path: Path, models_dir: Path, config: dict[str, Any]) -> bool:
    include = _clean_patterns(config.get("include_globs")) or ["*.joblib", "*.pkl"]
    exclude = _clean_patterns(config.get("exclude_globs"))
    return _matches_any(path, models_dir, include) and not _matches_any(path, models_dir, exclude)


def _runtime_defaults(
    *,
    catalog: dict[str, Any],
    model: dict[str, Any],
    artifact_relpath: str,
    artifact_path: Path,
    metrics_path: Path | None,
    adapter: str,
) -> dict[str, Any]:
    outputs = _output_config(model)
    task_type = str(model.get("task_type") or "").strip()
    output_primary = str(outputs.get("primary") or "").strip()
    discovery_payload = {
        "auto_discovered": True,
        "adapter": adapter,
        "catalog_version": catalog.get("version"),
        "catalog_model_id": model.get("id"),
        "artifact_filename": artifact_relpath,
        "artifact_path": str(artifact_path),
    }
    adapter_payload = {
        "task_type": task_type,
        "outputs": outputs,
        "feature_types": model.get("feature_types") if isinstance(model.get("feature_types"), dict) else {},
        "feature_defaults": model.get("feature_defaults") if isinstance(model.get("feature_defaults"), dict) else {},
    }
    for key in (
        "policy_formula",
        "action_thresholds",
        "simulator",
        "threshold",
        "winner",
        "serve_winner",
    ):
        if key in model:
            adapter_payload[key] = model[key]
    if isinstance(catalog.get("neural_architecture"), dict):
        adapter_payload["neural_architecture"] = catalog["neural_architecture"]

    defaults = {
        "prediction_source": "local_model_artifact",
        "model_discovery": discovery_payload,
        adapter: adapter_payload,
        "output": {
            "name": output_primary,
            "task_type": task_type,
        },
    }
    if metrics_path is not None:
        defaults["metrics_path"] = str(metrics_path)
    prediction_defaults = model.get("prediction_defaults")
    if isinstance(prediction_defaults, dict):
        defaults = _deep_merge_dicts(defaults, prediction_defaults)
    return defaults


def discover_model_config_rows(
    *,
    runtime_config_path: str | Path | None = None,
) -> list[dict[str, Any]]:
    config = _model_discovery_config(runtime_config_path)
    if not bool(config.get("enabled", False)):
        return []

    models_dir = _resolve_models_dir(config)
    catalog_path = _resolve_path(config.get("catalog_path"))
    catalog = _read_json(catalog_path, label="model discovery catalog")

    default_adapter = str(config.get("default_adapter") or "").strip()
    supported_adapters = {
        str(item).strip()
        for item in _clean_patterns(config.get("supported_adapters"))
        if str(item).strip()
    } or {default_adapter}
    if not default_adapter:
        raise ModelDiscoveryConfigError("model_discovery.default_adapter is required.")
    _validate_adapter(default_adapter, supported_adapters)

    models = catalog.get("models")
    if not isinstance(models, list):
        raise ModelDiscoveryConfigError("Model discovery catalog must define a models list.")

    rows: list[dict[str, Any]] = []
    seen_model_ids: set[str] = set()
    for index, model in enumerate(models):
        if not isinstance(model, dict):
            raise ModelDiscoveryConfigError(f"models[{index}] must be an object.")
        missing = sorted(field for field in CATALOG_REQUIRED_FIELDS if field not in model)
        if missing:
            raise ModelDiscoveryConfigError(
                f"models[{index}] is missing required field(s): {', '.join(missing)}"
            )

        model_id = str(model.get("registered_model_name") or "").strip()
        if not model_id:
            raise ModelDiscoveryConfigError(f"models[{index}].registered_model_name cannot be empty.")
        model_key = safe_slug(model_id)
        if model_key in seen_model_ids:
            raise ModelDiscoveryConfigError(f"Duplicate discovered model id: {model_id}")
        seen_model_ids.add(model_key)

        adapter = str(model.get("adapter") or default_adapter).strip()
        _validate_adapter(adapter, supported_adapters)

        artifact_relpath = str(model.get("artifact_filename") or "").strip()
        if not artifact_relpath:
            raise ModelDiscoveryConfigError(f"models[{index}].artifact_filename cannot be empty.")
        artifact_path = (models_dir / artifact_relpath).resolve()
        enabled = bool(model.get("enabled"))
        artifact_missing = enabled and not artifact_path.exists()
        if enabled:
            if not _artifact_allowed(artifact_path, models_dir, config):
                raise ModelDiscoveryConfigError(
                    f"Configured artifact for {model_id} is excluded by discovery rules: {artifact_relpath}"
                )
            if artifact_missing and not bool(config.get("deactivate_missing_artifacts", False)):
                raise ModelDiscoveryConfigError(
                    f"Configured artifact for {model_id} does not exist: {artifact_path}"
                )
        row_active = enabled and not artifact_missing

        metrics_path = None
        metrics_filename = str(model.get("metrics_filename") or "").strip()
        if metrics_filename:
            candidate_metrics = (models_dir / metrics_filename).resolve()
            if candidate_metrics.exists():
                metrics_path = candidate_metrics

        features = _clean_features(model.get("features"), label=f"models[{index}].features")
        feature_types = model.get("feature_types")
        feature_info = dict(feature_types) if isinstance(feature_types, dict) else {}
        examples = model.get("examples")
        rows.append(
            {
                "model_id": model_id,
                "description": str(model.get("description") or model.get("name") or model_id).strip(),
                "file_path": artifact_relpath,
                "model_type": adapter,
                "features": features,
                "feature_info": feature_info,
                "examples": examples if isinstance(examples, list) else [],
                "defaults": _runtime_defaults(
                    catalog=catalog,
                    model={
                        **model,
                        "prediction_defaults": _deep_merge_dicts(
                            model.get("prediction_defaults")
                            if isinstance(model.get("prediction_defaults"), dict)
                            else {},
                            {
                                "model_discovery": {
                                    "artifact_missing": artifact_missing,
                                }
                            },
                        ),
                    },
                    artifact_relpath=artifact_relpath,
                    artifact_path=artifact_path,
                    metrics_path=metrics_path,
                    adapter=adapter,
                ),
                "is_active": row_active,
            }
        )
    return rows


def _row_changed(obj: MLModelConfig, defaults: dict[str, Any]) -> bool:
    return any(getattr(obj, key) != value for key, value in defaults.items())


def sync_discovered_model_configs(
    reason: str,
    *,
    runtime_config_path: str | Path | None = None,
    model_config_path: str | Path | None = None,
) -> dict[str, Any]:
    config = _model_discovery_config(runtime_config_path)
    if not bool(config.get("enabled", False)):
        return {
            "enabled": False,
            "reason": reason,
            "changed": False,
            "discovered_total": 0,
            "upserted_total": 0,
        }
    if not bool(config.get("upsert_enabled", True)):
        return {
            "enabled": True,
            "reason": reason,
            "changed": False,
            "discovered_total": len(discover_model_config_rows(runtime_config_path=runtime_config_path)),
            "upserted_total": 0,
        }

    rows = discover_model_config_rows(runtime_config_path=runtime_config_path)
    model_config_source = _model_config_source_path(model_config_path)
    respect_model_config_file = bool(config.get("respect_model_config_file", True))
    changed = False
    upserted = 0
    skipped_config_source = 0
    with transaction.atomic():
        for row in rows:
            model_id = row["model_id"]
            defaults = {
                "description": row["description"],
                "file_path": row["file_path"],
                "model_type": row["model_type"],
                "features": row["features"],
                "feature_info": row["feature_info"],
                "examples": row["examples"],
                "defaults": row["defaults"],
                "is_active": row["is_active"],
            }
            obj = MLModelConfig.objects.filter(model_id=model_id).first()
            if obj is None:
                MLModelConfig.objects.create(model_id=model_id, **defaults)
                changed = True
                upserted += 1
                continue
            if (
                respect_model_config_file
                and str(getattr(obj, "config_source_path", "") or "").strip()
                == model_config_source
            ):
                skipped_config_source += 1
                continue
            if _row_changed(obj, defaults):
                for key, value in defaults.items():
                    setattr(obj, key, value)
                obj.save(update_fields=[*defaults.keys(), "updated_at"])
                changed = True
                upserted += 1

    summary = {
        "enabled": True,
        "reason": reason,
        "changed": changed,
        "discovered_total": len(rows),
        "upserted_total": upserted,
        "skipped_config_source_total": skipped_config_source,
    }
    logger.info("Local model discovery sync: %s", summary)
    return summary
