from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd
from django.conf import settings

from ml.core.model_discovery import discover_model_config_rows
from ml.core.prediction import (
    preload_prediction_runtime,
    run_prediction,
    run_prediction_batch,
)
from ml.core.registry import ModelRegistry, ModelRuntime

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RUNTIME_CONFIG_PATH = PROJECT_ROOT / "ml" / "config" / "prediction_runtime.json"
SUPPORTED_SOURCES = {"mlflow_registry", "local_registry"}


class ModelSourceResolutionError(RuntimeError):
    """Raised when no configured runtime source can serve a model."""


@dataclass(frozen=True)
class ResolvedModelSource:
    runtime: ModelRuntime
    source_used: str
    source_ref: str
    fallback_reason: str = ""
    source_errors: dict[str, str] = field(default_factory=dict)


def _runtime_config_path(path: str | Path | None = None) -> Path:
    if path is not None:
        return Path(path).expanduser().resolve()
    configured = os.getenv("PIOS_PREDICTION_RUNTIME_CONFIG", "").strip()
    if configured:
        return Path(configured).expanduser().resolve()
    return DEFAULT_RUNTIME_CONFIG_PATH.resolve()


def load_prediction_runtime_config(path: str | Path | None = None) -> dict[str, Any]:
    config_path = _runtime_config_path(path)
    try:
        payload = json.loads(config_path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ModelSourceResolutionError(
            f"Prediction runtime config not found: {config_path}"
        ) from exc
    except json.JSONDecodeError as exc:
        raise ModelSourceResolutionError(
            f"Invalid prediction runtime config JSON: {config_path}"
        ) from exc
    if not isinstance(payload, dict):
        raise ModelSourceResolutionError(
            f"Prediction runtime config root must be an object: {config_path}"
        )
    return payload


def _model_loading_config(runtime_config: dict[str, Any] | None = None) -> dict[str, Any]:
    payload = (
        runtime_config
        if isinstance(runtime_config, dict)
        else load_prediction_runtime_config()
    )
    config = payload.get("model_loading")
    return dict(config) if isinstance(config, dict) else {}


def _default_policy(config: dict[str, Any]) -> dict[str, Any]:
    policy = config.get("default_policy")
    if isinstance(policy, dict):
        return dict(policy)
    return {
        "primary": "mlflow_registry",
        "fallbacks": ["local_registry"],
        "default_model_version": "champion",
        "fail_if_all_unavailable": True,
        "record_source_metadata": True,
    }


def _source_aliases(config: dict[str, Any]) -> dict[str, dict[str, Any]]:
    aliases = config.get("source_aliases")
    if not isinstance(aliases, dict):
        aliases = {}
    merged = {
        "mlflow_registry": {
            "adapter": "mlflow",
            "uri_template": "models:/{model_id}@{model_version}",
        },
        "local_registry": {
            "adapter": "registry",
            "match_by": ["model_id", "aliases", "catalog_model_id"],
        },
    }
    for key, value in aliases.items():
        if isinstance(value, dict):
            merged[str(key)] = dict(value)
    return merged


def resolve_loading_policy(
    loading_policy: dict[str, Any] | None = None,
    *,
    runtime_config: dict[str, Any] | None = None,
) -> tuple[list[str], dict[str, dict[str, Any]], dict[str, Any]]:
    config = _model_loading_config(runtime_config)
    policy = _default_policy(config)
    if isinstance(loading_policy, dict):
        policy.update(loading_policy)

    primary = str(policy.get("primary") or "").strip()
    fallbacks = [
        str(item).strip()
        for item in policy.get("fallbacks", [])
        if str(item).strip()
    ]
    source_order = [primary, *fallbacks]
    source_order = list(dict.fromkeys(item for item in source_order if item))
    if not source_order:
        raise ModelSourceResolutionError("Model loading policy has no sources.")

    unsupported = [source for source in source_order if source not in SUPPORTED_SOURCES]
    if unsupported:
        raise ModelSourceResolutionError(
            f"Unsupported model loading source(s): {', '.join(unsupported)}"
        )

    aliases = _source_aliases(config)
    missing = [source for source in source_order if source not in aliases]
    if missing:
        raise ModelSourceResolutionError(
            f"Missing source alias config for: {', '.join(missing)}"
        )
    return source_order, aliases, policy


def _mlflow_runtime(
    model_id: str,
    model_version: str,
    source_config: dict[str, Any],
) -> ModelRuntime:
    uri_template = str(
        source_config.get("uri_template") or "models:/{model_id}@{model_version}"
    )
    source_ref = uri_template.format(
        model_id=model_id,
        model_version=model_version,
    )
    return ModelRuntime(
        model_id=model_id,
        aliases=[model_id],
        slug=model_id,
        file_path=source_ref,
        description=f"MLflow registry runtime for {model_id}",
        model_type="mlflow",
        status="loaded",
    )


def _get_registry(registry: Any | None) -> Any:
    return registry


def _local_runtime_from_active_registry(model_id: str, registry: Any | None) -> ModelRuntime | None:
    active_registry = _get_registry(registry)
    runtime = active_registry.get(model_id) if active_registry is not None else None
    if runtime is None:
        return None
    if runtime.status != "loaded":
        return None
    return runtime


def _local_runtime_from_discovery(model_id: str) -> ModelRuntime | None:
    rows = [
        row
        for row in discover_model_config_rows()
        if bool(row.get("is_active", True))
    ]
    if not rows:
        return None
    registry = ModelRegistry(
        models_dir=settings.PIOS_MODELS_DIR,
        config_path=settings.PIOS_MODEL_CONFIG_PATH,
        config_loader=lambda: rows,
    )
    registry.refresh(force=True)
    runtime = registry.get(model_id)
    if runtime is None or runtime.status != "loaded":
        return None
    return runtime


def _local_runtime(model_id: str, registry: Any | None) -> ModelRuntime:
    runtime = _local_runtime_from_active_registry(model_id, registry)
    if runtime is None:
        runtime = _local_runtime_from_discovery(model_id)
    if runtime is None:
        raise ModelSourceResolutionError(
            f"No loaded local registry runtime found for {model_id}."
        )
    return runtime


def resolve_model_runtime(
    model_id: str,
    model_version: str | None = None,
    *,
    loading_policy: dict[str, Any] | None = None,
    registry: Any | None = None,
    runtime_config: dict[str, Any] | None = None,
    skip_sources: set[str] | None = None,
) -> ResolvedModelSource:
    clean_model_id = str(model_id or "").strip()
    if not clean_model_id:
        raise ModelSourceResolutionError("model_id is required.")
    source_order, source_aliases, _policy = resolve_loading_policy(
        loading_policy,
        runtime_config=runtime_config,
    )
    default_version = str(_policy.get("default_model_version") or "champion")
    clean_version = str(model_version or default_version).strip() or default_version
    skipped = set(skip_sources or set())
    source_errors: dict[str, str] = {}

    for source in source_order:
        if source in skipped:
            continue
        try:
            if source == "mlflow_registry":
                runtime = _mlflow_runtime(
                    clean_model_id,
                    clean_version,
                    source_aliases[source],
                )
                preload_prediction_runtime(runtime)
            elif source == "local_registry":
                runtime = _local_runtime(clean_model_id, registry)
            else:  # guarded by validation above
                raise ModelSourceResolutionError(f"Unsupported source: {source}")
            return ResolvedModelSource(
                runtime=runtime,
                source_used=source,
                source_ref=str(runtime.file_path),
                fallback_reason="; ".join(source_errors.values()),
                source_errors=source_errors,
            )
        except Exception as exc:
            source_errors[source] = f"{type(exc).__name__}: {exc}"
            logger.info(
                "Model source %s failed for %s@%s: %s",
                source,
                clean_model_id,
                clean_version,
                exc,
            )

    detail = "; ".join(f"{key}: {value}" for key, value in source_errors.items())
    raise ModelSourceResolutionError(
        f"Could not resolve model {clean_model_id}@{clean_version}. {detail}"
    )


def _source_metadata(resolution: ResolvedModelSource) -> dict[str, Any]:
    return {
        "model_source": resolution.source_used,
        "model_source_ref": resolution.source_ref,
        "model_fallback_reason": resolution.fallback_reason,
    }


def _rows_from_frame(frame_or_rows: Any) -> list[dict[str, Any]]:
    if isinstance(frame_or_rows, pd.DataFrame):
        return frame_or_rows.to_dict(orient="records")
    if isinstance(frame_or_rows, list):
        return [dict(row) for row in frame_or_rows if isinstance(row, dict)]
    if isinstance(frame_or_rows, dict):
        return [dict(frame_or_rows)]
    return [dict(row) for row in pd.DataFrame(frame_or_rows).to_dict(orient="records")]


class ResolvedModelPredictor:
    def __init__(self, resolution: ResolvedModelSource):
        self.resolution = resolution
        self.runtime = resolution.runtime

    def predict(self, frame_or_rows: Any) -> list[dict[str, Any]]:
        outputs = run_prediction_batch(self.runtime, _rows_from_frame(frame_or_rows))
        metadata = _source_metadata(self.resolution)
        return [{**row, **metadata} for row in outputs]


def run_prediction_with_fallback(
    runtime: ModelRuntime,
    features: dict[str, Any],
    *,
    registry: Any | None = None,
    loading_policy: dict[str, Any] | None = None,
) -> dict[str, Any]:
    try:
        return {
            **run_prediction(runtime, features),
            "model_source": (
                "mlflow_registry" if runtime.model_type == "mlflow" else "local_registry"
            ),
            "model_source_ref": str(runtime.file_path),
        }
    except Exception:
        if runtime.model_type != "mlflow":
            raise
        resolution = resolve_model_runtime(
            runtime.model_id,
            loading_policy=loading_policy,
            registry=registry,
            skip_sources={"mlflow_registry"},
        )
        return {
            **run_prediction(resolution.runtime, features),
            **_source_metadata(resolution),
        }


def run_prediction_batch_with_fallback(
    runtime: ModelRuntime,
    feature_rows: list[dict[str, Any]],
    *,
    registry: Any | None = None,
    loading_policy: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    try:
        outputs = run_prediction_batch(runtime, feature_rows)
        source = "mlflow_registry" if runtime.model_type == "mlflow" else "local_registry"
        return [
            {
                **row,
                "model_source": source,
                "model_source_ref": str(runtime.file_path),
            }
            for row in outputs
        ]
    except Exception:
        if runtime.model_type != "mlflow":
            raise
        resolution = resolve_model_runtime(
            runtime.model_id,
            loading_policy=loading_policy,
            registry=registry,
            skip_sources={"mlflow_registry"},
        )
        outputs = run_prediction_batch(resolution.runtime, feature_rows)
        metadata = _source_metadata(resolution)
        return [{**row, **metadata} for row in outputs]
