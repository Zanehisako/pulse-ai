from __future__ import annotations

import copy
import json
import logging
import re
import tempfile
from datetime import datetime, timedelta, timezone
from typing import Any, Callable

from .ml_runtime import ensure_ml_runtime
from .standalone_entities import resolve_standalone_entities
from .standalone_predictions_config import load_standalone_predictions_config
from .standalone_snapshot import _standalone_model_loader
from .twin_forecast_config import (
    load_scheduled_jobs,
    load_twin_forecast_config,
)

logger = logging.getLogger(__name__)

_COMPUTE: dict[str, Callable[[dict[str, Any]], Any]] = {}


def _register_computers() -> None:
    if _COMPUTE:
        return

    def month_from_simulated_hour(ctx: dict[str, Any]) -> int:
        hours = float(ctx.get("simulated_hour") or 0)
        days = int(hours // 24)
        return (days % 12) + 1

    def dow_from_simulated_hour(ctx: dict[str, Any]) -> int:
        hours = float(ctx.get("simulated_hour") or 0)
        days = int(hours // 24)
        return (days % 7) + 1

    def iso_from_simulated_hour(ctx: dict[str, Any]) -> str:
        hours = float(ctx.get("simulated_hour") or 0)
        base = datetime(2026, 1, 1, tzinfo=timezone.utc)
        stamp = base + timedelta(hours=hours)
        return stamp.replace(microsecond=0).isoformat()

    _COMPUTE["month_from_simulated_hour"] = month_from_simulated_hour
    _COMPUTE["dow_from_simulated_hour"] = dow_from_simulated_hour
    _COMPUTE["iso_from_simulated_hour"] = iso_from_simulated_hour


def build_tick_context(
    *,
    tick_number: int,
    simulated_hour: float,
    simulated_state: dict[str, Any] | None = None,
) -> dict[str, Any]:
    _register_computers()
    cfg = load_twin_forecast_config()
    ctx = {
        "tick_number": int(tick_number),
        "simulated_hour": float(simulated_hour),
        "simulated_state": dict(simulated_state or {}),
    }
    overrides_cfg = (cfg.get("tick_context") or {}).get("mapping_overrides") or {}
    feature_overrides: dict[str, Any] = {}
    for column, spec in overrides_cfg.items():
        if not isinstance(spec, dict):
            continue
        compute = str(spec.get("compute") or "").strip()
        fn = _COMPUTE.get(compute)
        if fn is None:
            logger.warning("Unknown tick_context compute: %s", compute)
            continue
        feature_overrides[str(column)] = fn(ctx)
    ctx["feature_overrides"] = feature_overrides
    return ctx


def _normalize_supply_row(row: dict[str, Any]) -> dict[str, Any]:
    payload = dict(row)
    if "hospital" not in payload:
        hospital_id = str(payload.get("hospital_id") or "")
        if hospital_id:
            payload["hospital"] = {
                "hospital_id": hospital_id,
                "name": payload.get("hospital_name") or hospital_id,
            }
    return payload


def _job_by_id(job_id: str, config: dict[str, Any]) -> dict[str, Any] | None:
    for job in load_scheduled_jobs(config):
        if str(job.get("id")) == job_id:
            return job
    return None


def _merge_job_tick_mapping(
    job: dict[str, Any],
    tick_context: dict[str, Any],
) -> dict[str, Any]:
    patched = copy.deepcopy(job)
    mapping = dict(patched.get("feature_mapping") or {})
    for column, value in (tick_context.get("feature_overrides") or {}).items():
        mapping[str(column)] = {"value": value}
    patched["feature_mapping"] = mapping
    return patched


# Postprocess step that turns the model's demand into projected remaining stock
# (`max(0, current_stock - demand)`). Correct for the shared stock-forecast cards,
# but wrong for this Demand Forecast panel — during a shortage it pins to 0.
_STOCK_PROJECTION_STEP = "subtract_from_attr"

# Recent usage/inventory trend features the supply-side models lean on. The
# forecast WS layer (`forecast_ws.enrich_with_rolling_features`) populates these
# attrs on each supply row; feeding them lets a forecast track trends and react to
# surges instead of sitting at a flat baseline. Keys must match the model's
# feature names; attrs must match the enriched row fields.
_ROLLING_FEATURE_MAPPING: dict[str, Any] = {
    "usage_7d_mean": {"attr": "usage_7d_mean", "default": 0},
    "usage_30d_mean": {"attr": "usage_30d_mean", "default": 0},
    "inventory_7d_mean": {"attr": "inventory_7d_mean", "default": 0},
    "inventory_30d_mean": {"attr": "inventory_30d_mean", "default": 0},
}


def _resolve_value_mode(job: dict[str, Any], config: dict[str, Any]) -> str:
    """Per-role value mode: validation[role].predicted.value_mode, else top-level."""
    role = str((job or {}).get("dashboard_role") or "")
    series = (config.get("validation") or {}).get(role)
    if isinstance(series, dict):
        predicted = series.get("predicted")
        if isinstance(predicted, dict) and predicted.get("value_mode"):
            return str(predicted["value_mode"]).strip().lower()
    return str(config.get("prediction_value_mode") or "projected_stock").strip().lower()


def _apply_panel_value_mode(
    job: dict[str, Any],
    config: dict[str, Any],
) -> dict[str, Any]:
    """Adapt a shared forecast job to the validation panel's value semantics.

    Resolved per-role (`validation[<role>].predicted.value_mode`, default the
    top-level `prediction_value_mode`):

    * ``demand`` — drop the stock-projection postprocess so the model's demand is
      surfaced directly (horizon ``multiply`` steps are kept), and feed the rolling
      trend features.
    * ``raw`` — feed the rolling trend features (no postprocess change); used by the
      supply/wastage/stockout validation jobs.
    * anything else (e.g. legacy ``projected_stock``) — left untouched.

    Studio-only: the shared job config (and the main app's stock cards) are untouched.
    """
    mode = _resolve_value_mode(job, config)
    if mode not in ("demand", "raw"):
        return job
    patched = copy.deepcopy(job)
    if mode == "demand":
        patched["postprocess"] = [
            step
            for step in (patched.get("postprocess") or [])
            if isinstance(step, dict) and str(step.get("type")) != _STOCK_PROJECTION_STEP
        ]
    mapping = dict(patched.get("feature_mapping") or {})
    for column, spec in _ROLLING_FEATURE_MAPPING.items():
        mapping.setdefault(column, dict(spec))
    patched["feature_mapping"] = mapping
    return patched


def _aggregate(values: list[float], mode: str) -> float:
    if not values:
        return 0.0
    if mode == "mean":
        return round(sum(values) / len(values), 4)
    return round(sum(values), 4)


def _horizon_days(job: dict[str, Any] | None) -> int:
    """Days a forecast job projects over, parsed from ``dashboard_horizon`` (t1/t7/t30).

    The horizon jobs share one daily-demand model and multiply its output by the
    horizon (``×7``/``×30``) to project a cumulative total, so the panel's predicted
    series is a horizon-cumulative quantity. Returns ``1`` when no horizon is
    declared (non-forecast roles / legacy jobs), making any horizon-basis
    adjustment a no-op for them.
    """
    raw = str((job or {}).get("dashboard_horizon") or "").strip().lower()
    match = re.fullmatch(r"t(\d+)", raw)
    return max(1, int(match.group(1))) if match else 1


def _actual_snapshot_stock_sum(
    snapshot_payload: dict[str, Any], realized: dict[str, Any]
) -> float:
    total = 0.0
    for row in (snapshot_payload or {}).get("blood_supplies") or []:
        if isinstance(row, dict):
            total += float(row.get("current_stock_units") or 0)
    return total


# Named "actual" series extractors: (snapshot_payload, realized) -> scalar.
# `realized` is the per-day realized-metrics dict from forecast_ws.
_ACTUAL_SOURCES: dict[str, Callable[[dict[str, Any], dict[str, Any]], float]] = {
    "snapshot_stock_sum": _actual_snapshot_stock_sum,
    "realized_demand": lambda sp, r: float((r or {}).get("realized_demand") or 0.0),
    "realized_supply": lambda sp, r: float((r or {}).get("realized_supply") or 0.0),
    "realized_wastage": lambda sp, r: float((r or {}).get("realized_wastage") or 0.0),
    "realized_shortage_rate": lambda sp, r: float((r or {}).get("realized_shortage_rate") or 0.0),
    "realized_stockout_freq": lambda sp, r: float((r or {}).get("realized_stockout_freq") or 0.0),
}


def _resolve_actual(
    actual_cfg: dict[str, Any],
    snapshot_payload: dict[str, Any],
    realized: dict[str, Any] | None,
) -> float:
    fn = _ACTUAL_SOURCES.get(str((actual_cfg or {}).get("source") or ""))
    if fn is None:
        return 0.0
    try:
        return round(float(fn(snapshot_payload or {}, realized or {})), 4)
    except Exception:
        return 0.0


def _chart_point(
    *,
    config: dict[str, Any],
    tick_context: dict[str, Any],
    predictions: list[dict[str, Any]],
    snapshot_payload: dict[str, Any],
    job: dict[str, Any] | None = None,
    realized: dict[str, Any] | None = None,
) -> dict[str, Any]:
    chart_cfg = config.get("chart") or {}
    label_template = str(chart_cfg.get("label_template") or "tick:{tick_number}")
    label = (
        label_template.replace("{tick_number}", str(tick_context.get("tick_number")))
        .replace("{simulated_hour}", str(int(tick_context.get("simulated_hour") or 0)))
    )
    predicted_vals = [
        float(row.get("predicted_value") or 0)
        for row in predictions
        if row.get("predicted_value") is not None
    ]

    role = str((job or {}).get("dashboard_role") or "")
    series = (config.get("validation") or {}).get(role)
    if isinstance(series, dict):
        predicted_cfg = series.get("predicted") or {}
        actual_cfg = series.get("actual") or {}
        predicted = _aggregate(predicted_vals, str(predicted_cfg.get("aggregate") or "sum"))
        # Fixed calibration factor (default 1.0) to reconcile a model trained at a
        # different population scale with the simulation's regime. Kept exogenous
        # (config constant, not derived from `realized`) so validation stays honest.
        predicted *= float(predicted_cfg.get("scale", 1.0) or 1.0)
        actual = _resolve_actual(actual_cfg, snapshot_payload, realized)
        # Horizon-cumulative models (T7/T30) predict daily demand × horizon, but the
        # realized metric is a per-day rate. With `horizon_basis`, lift the realized
        # rate onto the model's horizon basis (× horizon days) so both series share
        # units. T1 (horizon 1) is unchanged. This assumes demand is ~stationary
        # across the window; accumulating realized over the full horizon would be
        # exact but needs a horizon-length run between chart points.
        if actual_cfg.get("horizon_basis"):
            actual = round(actual * _horizon_days(job), 4)
        predicted_label = str(predicted_cfg.get("label") or "Predicted")
        actual_label = str(actual_cfg.get("label") or "Actual")
    else:
        # Legacy fallback: aggregated prediction vs snapshot stock sum.
        predicted = _aggregate(predicted_vals, str(chart_cfg.get("demand_aggregate") or "sum"))
        actual = (
            round(_actual_snapshot_stock_sum(snapshot_payload, realized or {}), 4)
            if str(chart_cfg.get("supply_source")) == "snapshot_stock_sum"
            else 0.0
        )
        predicted_label = "Demand"
        actual_label = "Supply"

    return {
        "label": label,
        "predicted": predicted,
        "actual": actual,
        "predicted_label": predicted_label,
        "actual_label": actual_label,
        # Back-compat aliases for any consumer still reading demand/supply.
        "demand": predicted,
        "supply": actual,
    }


def _loading_policy(config: dict[str, Any]) -> dict[str, Any] | None:
    ref = str(config.get("loading_policy_ref") or "").strip()
    if ref.endswith("default_loading_policy"):
        standalone = load_standalone_predictions_config()
        policy = (standalone.get("scheduled_predictions") or {}).get(
            "default_loading_policy"
        )
        return dict(policy) if isinstance(policy, dict) else None
    policy = config.get("loading_policy")
    return dict(policy) if isinstance(policy, dict) else None


def run_tick_forecast(
    job_id: str,
    *,
    snapshot_payload: dict[str, Any] | None,
    tick_context: dict[str, Any],
    realized: dict[str, Any] | None = None,
) -> dict[str, Any]:
    warnings: list[str] = []
    config = load_twin_forecast_config()
    job = _job_by_id(job_id, config)
    if job is None:
        return {
            "job_id": job_id,
            "enabled": False,
            "warnings": [f"Unknown forecast job: {job_id}"],
            "predictions": [],
            "prediction_count": 0,
        }

    snapshot_payload = dict(snapshot_payload or {})
    standalone_cfg = load_standalone_predictions_config()
    runtime_cfg = standalone_cfg.get("ml_runtime") or {}
    if not ensure_ml_runtime(
        require_django_setup=bool(runtime_cfg.get("require_django_setup", True)),
        env_overrides=runtime_cfg.get("env")
        if isinstance(runtime_cfg.get("env"), dict)
        else None,
    ):
        return {
            "job_id": job_id,
            "model_id": str(job.get("model_id") or ""),
            "tick_number": tick_context.get("tick_number"),
            "simulated_hour": tick_context.get("simulated_hour"),
            "warnings": ["ML runtime unavailable."],
            "predictions": [],
            "prediction_count": 0,
        }

    from ml.core.scheduled_predictions import run_scheduled_prediction_jobs

    entity_map = dict(standalone_cfg.get("entity_source_map") or {})
    entity_source = str(job.get("entity_source") or "")
    supplies = [
        _normalize_supply_row(row)
        for row in (snapshot_payload.get("blood_supplies") or [])
        if isinstance(row, dict)
    ]

    def entity_resolver(path: str):
        if path == entity_source and supplies:
            return supplies
        return resolve_standalone_entities(
            path,
            entity_source_map=entity_map,
        )

    patched_job = _apply_panel_value_mode(
        _merge_job_tick_mapping(job, tick_context), config
    )
    from .twin_forecast_config import scheduled_predictions_path

    scheduled_path = scheduled_predictions_path(config)

    # Run only this job by writing a temporary single-job view via job_ids filter
    model_loader = _standalone_model_loader(standalone_cfg)
    loading_policy = _loading_policy(config)

    def loader(model_id: str, version: str, lp: dict[str, Any] | None = None):
        merged = dict(loading_policy or {})
        if isinstance(lp, dict):
            merged.update(lp)
        if isinstance(patched_job.get("loading_policy"), dict):
            merged.update(patched_job["loading_policy"])
        return model_loader(model_id, version, merged or None)

    # Inject patched feature_mapping by temporary job override in scheduled file is heavy;
    # use custom runner hook via copying scheduled payload in memory.
    from pathlib import Path

    try:
        raw = json.loads(scheduled_path.read_text(encoding="utf-8"))
        jobs = []
        for row in raw.get("jobs") or []:
            if str(row.get("id")) == job_id:
                jobs.append(patched_job)
            else:
                jobs.append(row)
        raw["jobs"] = jobs
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False, encoding="utf-8"
        ) as tmp:
            json.dump(raw, tmp)
            tmp_path = Path(tmp.name)

        try:
            summary, predictions = run_scheduled_prediction_jobs(
                config_path=tmp_path,
                entity_resolver=entity_resolver,
                job_ids={job_id},
                model_loader=loader,
            )
            logger.info("Tick forecast job=%s summary=%s", job_id, summary)
        finally:
            tmp_path.unlink(missing_ok=True)
    except Exception as exc:
        logger.warning("Tick forecast failed job=%s: %s", job_id, exc)
        warnings.append(str(exc))
        predictions = []

    ws_cap = int((config.get("ws") or {}).get("max_prediction_rows") or 80)
    chart_point = _chart_point(
        config=config,
        tick_context=tick_context,
        predictions=predictions,
        snapshot_payload=snapshot_payload,
        job=job,
        realized=realized,
    )
    return {
        "job_id": job_id,
        "model_id": str(job.get("model_id") or ""),
        "tick_number": tick_context.get("tick_number"),
        "simulated_hour": tick_context.get("simulated_hour"),
        "predictions": predictions[:ws_cap],
        "prediction_count": len(predictions),
        "chart_point": chart_point,
        "warnings": warnings,
    }