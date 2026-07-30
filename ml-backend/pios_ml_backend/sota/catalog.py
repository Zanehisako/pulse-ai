from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any


class SotaCatalogError(ValueError):
    """Raised when the SOTA model catalog cannot be used."""


DEFAULT_CATALOG_PATH = Path(__file__).resolve().parents[2] / "config" / "sota_model_catalog.json"

_REQUIRED_MODEL_FIELDS = {
    "id",
    "enabled",
    "registered_model_name",
    "artifact_filename",
    "task_type",
    "features",
    "outputs",
}
_SUPPORTED_TASK_TYPES = {"quantile", "regression", "classification", "policy", "simulator"}


def resolve_catalog_path(path: str | Path | None = None) -> Path:
    if path is not None:
        return Path(path).expanduser().resolve()
    configured = os.getenv("PIOS_SOTA_MODEL_CATALOG", "").strip()
    if configured:
        return Path(configured).expanduser().resolve()
    return DEFAULT_CATALOG_PATH


def resolve_models_dir(catalog: dict[str, Any], *, base_dir: Path | None = None) -> Path:
    env_name = str(catalog.get("models_dir_env") or "").strip()
    if env_name:
        configured = os.getenv(env_name, "").strip()
        if configured:
            return Path(configured).expanduser().resolve()

    default_dir = str(catalog.get("default_models_dir") or "").strip()
    if not default_dir:
        raise SotaCatalogError("SOTA catalog must define default_models_dir or models_dir_env.")
    path = Path(default_dir).expanduser()
    if not path.is_absolute() and base_dir is not None:
        path = base_dir / path
    return path.resolve()


def validate_sota_catalog(
    catalog: dict[str, Any],
    *,
    require_artifacts: bool = True,
    base_dir: Path | None = None,
) -> dict[str, Any]:
    if not isinstance(catalog, dict):
        raise SotaCatalogError("SOTA catalog root must be an object.")

    models = catalog.get("models")
    if not isinstance(models, list):
        raise SotaCatalogError("SOTA catalog must define a models list.")

    models_dir = resolve_models_dir(catalog, base_dir=base_dir)
    if require_artifacts and not models_dir.exists():
        raise SotaCatalogError(f"SOTA models directory does not exist: {models_dir}")

    seen: set[str] = set()
    seen_registered_names: set[str] = set()
    for index, model in enumerate(models):
        if not isinstance(model, dict):
            raise SotaCatalogError(f"models[{index}] must be an object.")
        missing = sorted(field for field in _REQUIRED_MODEL_FIELDS if field not in model)
        if missing:
            raise SotaCatalogError(f"models[{index}] is missing required field(s): {', '.join(missing)}")

        model_id = str(model.get("id") or "").strip()
        if not model_id:
            raise SotaCatalogError(f"models[{index}].id cannot be empty.")
        if model_id in seen:
            raise SotaCatalogError(f"Duplicate SOTA model id: {model_id}")
        seen.add(model_id)

        registered_name = str(model.get("registered_model_name") or "").strip()
        if not registered_name:
            raise SotaCatalogError(f"models[{index}].registered_model_name cannot be empty.")
        if registered_name in seen_registered_names:
            raise SotaCatalogError(f"Duplicate SOTA registered_model_name: {registered_name}")
        seen_registered_names.add(registered_name)

        features = model.get("features")
        if not isinstance(features, list) or not all(isinstance(item, str) and item.strip() for item in features):
            raise SotaCatalogError(f"models[{index}].features must be a non-empty list of strings.")

        task_type = str(model.get("task_type") or "").strip()
        if task_type not in _SUPPORTED_TASK_TYPES:
            raise SotaCatalogError(f"models[{index}].task_type is not supported: {task_type}")

        outputs = model.get("outputs")
        if not isinstance(outputs, dict) or not outputs:
            raise SotaCatalogError(f"models[{index}].outputs must be a non-empty object.")
        output_primary = str(outputs.get("primary") or "").strip()
        if not output_primary:
            raise SotaCatalogError(f"models[{index}].outputs.primary is required.")

        artifact_filename = str(model.get("artifact_filename") or "").strip()
        artifact_path = models_dir / artifact_filename
        if require_artifacts and bool(model.get("enabled")) and not artifact_path.exists():
            raise SotaCatalogError(f"Configured artifact for {model_id} does not exist: {artifact_path}")

        metrics_filename = str(model.get("metrics_filename") or "").strip()
        if require_artifacts and metrics_filename and bool(model.get("enabled")):
            metrics_path = models_dir / metrics_filename
            if not metrics_path.exists():
                raise SotaCatalogError(f"Configured metrics for {model_id} do not exist: {metrics_path}")

    return catalog


def load_sota_catalog(path: str | Path | None = None, *, require_artifacts: bool = True) -> dict[str, Any]:
    catalog_path = resolve_catalog_path(path)
    try:
        payload = json.loads(catalog_path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise SotaCatalogError(f"SOTA catalog not found: {catalog_path}") from exc
    except json.JSONDecodeError as exc:
        raise SotaCatalogError(f"Invalid SOTA catalog JSON: {catalog_path}") from exc
    return validate_sota_catalog(
        payload,
        require_artifacts=require_artifacts,
        base_dir=catalog_path.parent,
    )
