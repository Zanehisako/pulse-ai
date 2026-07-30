from __future__ import annotations

import json
import logging
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .ml_runtime import ensure_ml_runtime
from .operational_world import OperationalWorld, advance_operational_tick, get_operational_world
from .standalone_entities import resolve_standalone_entities
from .standalone_predictions_config import (
    load_standalone_predictions_config,
    ml_config_root,
    standalone_predictions_enabled,
)

logger = logging.getLogger(__name__)


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _watermark(rows: list[dict[str, Any]], *keys: str) -> dict[str, Any]:
    values = [
        row[key]
        for row in rows
        for key in keys
        if row.get(key) not in (None, "")
    ]
    return {"count": len(rows), "latest": max(values) if values else None}


def load_model_configs_from_catalog(payload: dict[str, Any]) -> list[dict[str, Any]]:
    root = ml_config_root(payload)
    catalog_path = root / str(payload.get("model_catalog_file") or "config.json")
    catalog = json.loads(catalog_path.read_text(encoding="utf-8"))
    models = catalog.get("models") or []
    filt = payload.get("model_catalog_filter") or {}
    exclude_types = {str(item).lower() for item in filt.get("exclude_types") or []}
    require_enabled = bool(filt.get("require_enabled", True))
    now = _utc_now_iso()

    rows: list[dict[str, Any]] = []
    for entry in models:
        if not isinstance(entry, dict):
            continue
        model_type = str(entry.get("type") or "").lower()
        if model_type in exclude_types:
            continue
        if require_enabled and not bool(entry.get("enabled", True)):
            continue
        model_id = str(entry.get("id") or "").strip()
        if not model_id:
            continue
        rows.append(
            {
                "model_id": model_id,
                "description": str(entry.get("description") or ""),
                "model_type": model_type or "unknown",
                "is_active": bool(entry.get("enabled", True)),
                "config_source_path": str(entry.get("file_path") or ""),
                "updated_at": now,
            }
        )
    return rows


def _standalone_model_loader(config: dict[str, Any]):
    scheduled_cfg = config.get("scheduled_predictions") or {}
    default_policy = scheduled_cfg.get("default_loading_policy")
    base_policy = dict(default_policy) if isinstance(default_policy, dict) else None

    def loader(
        model_id: str,
        version: str,
        loading_policy: dict[str, Any] | None = None,
    ):
        from ml.core.scheduled_predictions import default_model_loader

        merged: dict[str, Any] | None = None
        if base_policy or isinstance(loading_policy, dict):
            merged = dict(base_policy or {})
            if isinstance(loading_policy, dict):
                merged.update(loading_policy)
        return default_model_loader(model_id, version, merged)

    return loader


def run_standalone_predictions(
    world: OperationalWorld,
    *,
    config: dict[str, Any] | None = None,
) -> tuple[list[dict[str, Any]], list[str]]:
    warnings: list[str] = []
    config = config or load_standalone_predictions_config()
    runtime_cfg = config.get("ml_runtime") or {}
    if not ensure_ml_runtime(
        require_django_setup=bool(runtime_cfg.get("require_django_setup", True)),
        env_overrides=runtime_cfg.get("env")
        if isinstance(runtime_cfg.get("env"), dict)
        else None,
    ):
        warnings.append("ML runtime unavailable; predictions were not generated.")
        return [], warnings

    from ml.core.scheduled_predictions import run_scheduled_prediction_jobs

    root = ml_config_root(config)
    scheduled_path = root / str(config.get("scheduled_predictions_file") or "scheduled_predictions.json")
    entity_map = dict(config.get("entity_source_map") or {})
    allow = config.get("job_ids")
    job_ids = (
        {str(item) for item in allow if str(item).strip()}
        if isinstance(allow, list) and allow
        else None
    )

    def entity_resolver(path: str):
        return resolve_standalone_entities(path, entity_source_map=entity_map, world=world)

    try:
        summary, predictions = run_scheduled_prediction_jobs(
            config_path=scheduled_path,
            entity_resolver=entity_resolver,
            job_ids=job_ids,
            model_loader=_standalone_model_loader(config),
        )
        if config.get("audit", {}).get("log_job_summary", True):
            logger.info("Standalone scheduled predictions: %s", summary)
        return predictions, warnings
    except Exception as exc:
        logger.warning("Standalone prediction run failed: %s", exc)
        warnings.append(f"Standalone prediction run failed: {exc}")
        return [], warnings


def capture_standalone_operational_snapshot(
    *,
    run_predictions: bool = True,
    advance_tick: bool = False,
    world: OperationalWorld | None = None,
) -> dict[str, Any]:
    if not standalone_predictions_enabled():
        world = world or get_operational_world()
        payload = world.normalized_payload()
        return {
            "snapshot_id": f"standalone-{uuid.uuid4().hex[:12]}",
            "run_id": None,
            "source": "synthetic",
            "captured_at": _utc_now_iso(),
            "source_watermarks": {"standalone": {"count": 1, "latest": _utc_now_iso()}},
            "payload": payload,
            "freshness": {
                "captured_at": _utc_now_iso(),
                "status": "synthetic",
                "is_stale": False,
            },
            "warnings": ["Standalone predictions disabled by config."],
        }

    config = load_standalone_predictions_config()
    world = world or get_operational_world()
    warnings: list[str] = []

    refresh_cfg = config.get("refresh") or {}
    if advance_tick or bool(refresh_cfg.get("advance_operational_tick_before_predict")):
        advance_operational_tick(world)

    try:
        world.model_configs = load_model_configs_from_catalog(config)
    except Exception as exc:
        warnings.append(f"Model catalog load failed: {exc}")
        world.model_configs = []

    if run_predictions:
        predictions, pred_warnings = run_standalone_predictions(world, config=config)
        world.predictions = predictions
        warnings.extend(pred_warnings)
    else:
        world.predictions = []

    payload = world.normalized_payload()
    watermarks = {
        "blood_supplies": _watermark(payload["blood_supplies"], "updated_at", "event_timestamp"),
        "donors": _watermark(payload["donors"], "updated_at", "event_timestamp"),
        "predictions": _watermark(payload["predictions"], "created_at"),
        "model_configs": _watermark(payload["model_configs"], "updated_at"),
    }
    captured_at = _utc_now_iso()
    has_live = bool(watermarks.get("predictions", {}).get("latest")) or bool(
        watermarks.get("blood_supplies", {}).get("latest")
    )

    return {
        "snapshot_id": f"standalone-{uuid.uuid4().hex[:12]}",
        "run_id": None,
        "source": "standalone_live" if has_live else "standalone",
        "captured_at": captured_at,
        "source_watermarks": watermarks,
        "payload": payload,
        "freshness": {
            "captured_at": captured_at,
            "status": "fresh" if has_live else "degraded",
            "is_stale": not has_live,
        },
        "warnings": warnings,
    }