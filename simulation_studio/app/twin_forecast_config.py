from __future__ import annotations

import json
import os
from functools import lru_cache
from pathlib import Path
from typing import Any

from .standalone_predictions_config import (
    STUDIO_ROOT,
    load_standalone_predictions_config,
    ml_config_root,
)

DEFAULT_CONFIG_PATH = STUDIO_ROOT / "config" / "twin_forecast_panel.json"


class TwinForecastConfigError(RuntimeError):
    pass


@lru_cache(maxsize=1)
def load_twin_forecast_config(
    config_path: Path | None = None,
) -> dict[str, Any]:
    path = config_path or Path(
        os.environ.get("PIOS_TWIN_FORECAST_CONFIG", str(DEFAULT_CONFIG_PATH))
    )
    if not path.exists():
        raise TwinForecastConfigError(f"Twin forecast config not found: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise TwinForecastConfigError(f"Invalid twin forecast JSON: {path}") from exc
    if not isinstance(payload, dict):
        raise TwinForecastConfigError("Twin forecast config root must be an object.")
    validate_twin_forecast_config(payload)
    return payload


def scheduled_predictions_path(config: dict[str, Any]) -> Path:
    standalone = load_standalone_predictions_config()
    root = ml_config_root(standalone)
    name = str(
        config.get("scheduled_predictions_file")
        or standalone.get("scheduled_predictions_file")
        or "scheduled_predictions.json"
    )
    return root / name


def load_scheduled_jobs(config: dict[str, Any] | None = None) -> list[dict[str, Any]]:
    cfg = config or load_twin_forecast_config()
    path = scheduled_predictions_path(cfg)
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise TwinForecastConfigError(
            f"Could not read scheduled predictions: {path}"
        ) from exc
    jobs = payload.get("jobs")
    if not isinstance(jobs, list):
        raise TwinForecastConfigError("scheduled_predictions jobs must be a list.")
    return [row for row in jobs if isinstance(row, dict)]


def validate_twin_forecast_config(payload: dict[str, Any]) -> None:
    for key in ("version", "enabled"):
        if key not in payload:
            raise TwinForecastConfigError(f"Twin forecast config missing: {key}")
    if not payload.get("enabled"):
        return
    jobs = load_scheduled_jobs(payload)
    options = _filter_panel_jobs(jobs, payload)
    if not options:
        raise TwinForecastConfigError("No forecast panel jobs matched panel_job_filter.")
    default_job = str(payload.get("default_job_id") or "").strip()
    if default_job and not any(str(j.get("id")) == default_job for j in options):
        raise TwinForecastConfigError(
            f"default_job_id not found among forecast jobs: {default_job}"
        )


def _filter_panel_jobs(
    jobs: list[dict[str, Any]],
    config: dict[str, Any],
) -> list[dict[str, Any]]:
    filt = config.get("panel_job_filter") or {}
    roles = {str(r) for r in (filt.get("dashboard_roles") or ["forecast"])}
    require_enabled = bool(filt.get("require_enabled", True))
    allow = config.get("job_ids")
    allowed = (
        {str(item) for item in allow if str(item).strip()}
        if isinstance(allow, list) and allow
        else None
    )
    rows: list[dict[str, Any]] = []
    for job in jobs:
        job_id = str(job.get("id") or "")
        if not job_id:
            continue
        if allowed is not None and job_id not in allowed:
            continue
        if require_enabled and not bool(job.get("enabled")):
            continue
        role = str(job.get("dashboard_role") or "")
        if role not in roles:
            continue
        rows.append(job)
    return rows


def list_panel_options(config: dict[str, Any] | None = None) -> list[dict[str, Any]]:
    cfg = config or load_twin_forecast_config()
    validation = cfg.get("validation") or {}
    jobs = _filter_panel_jobs(load_scheduled_jobs(cfg), cfg)
    options: list[dict[str, Any]] = []
    for job in jobs:
        job_id = str(job["id"])
        model_id = str(job.get("model_id") or "")
        horizon = str(job.get("dashboard_horizon") or "")
        desc = str(job.get("description") or "").strip()
        label_parts = [model_id.replace("_", " ")]
        if horizon:
            label_parts.append(horizon.upper())
        label = " · ".join(part for part in label_parts if part)
        series = validation.get(str(job.get("dashboard_role") or "")) or {}
        predicted_label = str((series.get("predicted") or {}).get("label") or "Predicted")
        actual_label = str((series.get("actual") or {}).get("label") or "Actual")
        options.append(
            {
                "job_id": job_id,
                "model_id": model_id,
                "horizon": horizon,
                "label": label,
                "description": desc,
                "predicted_label": predicted_label,
                "actual_label": actual_label,
            }
        )
    return options


def default_job_id(config: dict[str, Any] | None = None) -> str | None:
    cfg = config or load_twin_forecast_config()
    if not cfg.get("enabled"):
        return None
    explicit = str(cfg.get("default_job_id") or "").strip()
    options = list_panel_options(cfg)
    if explicit and any(opt["job_id"] == explicit for opt in options):
        return explicit
    if options:
        return options[0]["job_id"]
    return None


def forecast_panel_meta() -> dict[str, Any]:
    try:
        cfg = load_twin_forecast_config()
    except TwinForecastConfigError:
        return {"enabled": False, "options": [], "default_job_id": None}
    chart_cfg = cfg.get("chart") or {}
    return {
        "enabled": bool(cfg.get("enabled")),
        "options": list_panel_options(cfg),
        "default_job_id": default_job_id(cfg),
        "chart_max_points": int(chart_cfg.get("max_points") or 24),
    }