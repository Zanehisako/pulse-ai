from __future__ import annotations

import importlib
import json
import os
from functools import lru_cache
from pathlib import Path
from typing import Any


def import_callable_path(path: str):
    module_name, _, attr = str(path).rpartition(".")
    if not module_name or not attr:
        raise RuntimeError(f"Invalid callable path: {path}")
    module = importlib.import_module(module_name)
    fn = getattr(module, attr)
    if not callable(fn):
        raise RuntimeError(f"Configured object is not callable: {path}")
    return fn

STUDIO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG_PATH = STUDIO_ROOT / "config" / "standalone_predictions.json"


class StandalonePredictionsConfigError(RuntimeError):
    pass


def _resolve_path(root: Path, relative: str) -> Path:
    candidate = (root / relative).resolve()
    return candidate


@lru_cache(maxsize=1)
def load_standalone_predictions_config(
    config_path: Path | None = None,
) -> dict[str, Any]:
    path = config_path or Path(
        os.environ.get("PIOS_STANDALONE_PREDICTIONS_CONFIG", str(DEFAULT_CONFIG_PATH))
    )
    if not path.exists():
        raise StandalonePredictionsConfigError(
            f"Standalone predictions config not found: {path}"
        )
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise StandalonePredictionsConfigError(
            f"Invalid standalone predictions JSON: {path}"
        ) from exc
    if not isinstance(payload, dict):
        raise StandalonePredictionsConfigError("Standalone predictions root must be an object.")
    validate_standalone_predictions_config(payload, config_dir=path.parent)
    return payload


def ml_config_root(payload: dict[str, Any], *, config_dir: Path | None = None) -> Path:
    """Resolve ML config directory.

    ``ml_config_root`` in JSON is relative to the Simulation Studio package root
    (``simulation_studio/``), not the ``config/`` subdirectory.
    ``PIOS_ML_CONFIG_ROOT`` overrides when set.
    """
    del config_dir  # kept for call-site compatibility; paths use STUDIO_ROOT
    env_root = os.environ.get("PIOS_ML_CONFIG_ROOT", "").strip()
    if env_root:
        return Path(env_root).resolve()
    relative = str(payload.get("ml_config_root") or "../backendMulti/ml/config")
    return _resolve_path(STUDIO_ROOT, relative)


def validate_standalone_predictions_config(
    payload: dict[str, Any],
    *,
    config_dir: Path | None = None,
) -> None:
    for key in ("version", "enabled", "entity_source_map"):
        if key not in payload:
            raise StandalonePredictionsConfigError(
                f"Standalone predictions config missing required field: {key}"
            )
    entity_map = payload.get("entity_source_map")
    if not isinstance(entity_map, dict) or not entity_map:
        raise StandalonePredictionsConfigError("entity_source_map must be a non-empty object.")

    root = ml_config_root(payload, config_dir=config_dir)
    scheduled_name = str(payload.get("scheduled_predictions_file") or "scheduled_predictions.json")
    scheduled_path = root / scheduled_name
    if not scheduled_path.exists():
        raise StandalonePredictionsConfigError(
            f"Scheduled predictions file not found: {scheduled_path}"
        )

    try:
        scheduled = json.loads(scheduled_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise StandalonePredictionsConfigError(
            f"Invalid scheduled predictions JSON: {scheduled_path}"
        ) from exc

    jobs = scheduled.get("jobs")
    if not isinstance(jobs, list):
        raise StandalonePredictionsConfigError("scheduled_predictions jobs must be a list.")

    allow_list = payload.get("job_ids")
    allowed_ids = (
        {str(item) for item in allow_list if str(item).strip()}
        if isinstance(allow_list, list) and allow_list
        else None
    )

    for job in jobs:
        if not isinstance(job, dict) or not bool(job.get("enabled")):
            continue
        job_id = str(job.get("id") or "")
        if allowed_ids is not None and job_id not in allowed_ids:
            continue
        entity_source = str(job.get("entity_source") or "")
        if entity_source not in entity_map:
            raise StandalonePredictionsConfigError(
                f"entity_source_map missing resolver for job {job_id}: {entity_source}"
            )


def standalone_predictions_status() -> tuple[bool, list[str]]:
    """Return whether standalone predictions are active and any disable reasons."""
    env = os.environ.get("PIOS_STANDALONE_PREDICTIONS", "").strip().lower()
    if env in {"0", "false", "no"}:
        return False, ["PIOS_STANDALONE_PREDICTIONS is set to disable standalone predictions."]
    if env in {"1", "true", "yes"}:
        try:
            load_standalone_predictions_config()
            return True, []
        except StandalonePredictionsConfigError as exc:
            return False, [str(exc)]
    try:
        cfg = load_standalone_predictions_config()
        if not bool(cfg.get("enabled", True)):
            return False, ["standalone_predictions.json has enabled=false."]
        return True, []
    except StandalonePredictionsConfigError as exc:
        return False, [str(exc)]


def standalone_predictions_enabled() -> bool:
    enabled, _warnings = standalone_predictions_status()
    return enabled