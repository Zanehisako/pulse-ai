from __future__ import annotations

import importlib
import json
import logging
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from ml.core.model_loading import ResolvedModelPredictor, resolve_model_runtime

logger = logging.getLogger(__name__)

EntityResolver = Callable[[str], Iterable[Any]]
ModelLoader = Callable[
    [str, str, dict[str, Any] | None],
    tuple[ResolvedModelPredictor, Any],
]
OnPrediction = Callable[[dict[str, Any], Any, dict[str, Any], Any], None]


@dataclass(frozen=True)
class ScheduledPredictionRow:
    entity_id: str
    entity_type: str
    hospital_id: str
    blood_type: str
    model_name: str
    model_version: str
    predicted_value: float
    predicted_for_date: str
    created_at: str
    alert_triggered: bool
    alert_threshold_used: float | None
    confidence: float | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "entity_id": self.entity_id,
            "entity_type": self.entity_type,
            "hospital_id": self.hospital_id,
            "blood_type": self.blood_type,
            "model_name": self.model_name,
            "model_version": self.model_version,
            "predicted_value": self.predicted_value,
            "predicted_for_date": self.predicted_for_date,
            "created_at": self.created_at,
            "alert_triggered": self.alert_triggered,
            "alert_threshold_used": self.alert_threshold_used,
            "confidence": self.confidence,
        }


def load_scheduled_predictions_config(config_path: Path | str) -> dict[str, Any]:
    path = Path(config_path)
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise RuntimeError(f"Scheduled prediction config not found: {path}") from exc
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"Invalid scheduled prediction config JSON: {path}") from exc
    if not isinstance(payload, dict):
        raise RuntimeError("Scheduled prediction config root must be an object.")
    jobs = payload.get("jobs")
    if not isinstance(jobs, list):
        raise RuntimeError("Scheduled prediction config must define a jobs list.")
    seen: set[str] = set()
    for index, row in enumerate(jobs):
        if not isinstance(row, dict):
            raise RuntimeError(f"Scheduled prediction job #{index + 1} must be an object.")
        job_id = str(row.get("id") or "").strip()
        if not job_id:
            raise RuntimeError(f"Scheduled prediction job #{index + 1} is missing id.")
        if job_id in seen:
            raise RuntimeError(f"Duplicate scheduled prediction job id: {job_id}")
        seen.add(job_id)
        for key in ("enabled", "model_id", "entity_source", "feature_builder", "result", "alert"):
            if key not in row:
                raise RuntimeError(f"Scheduled prediction job {job_id} is missing {key}.")
    return payload


def attr_value(row: Any, path: str, default: Any = "") -> Any:
    current = row
    parts = str(path).split(".")
    for index, part in enumerate(parts):
        if current is None:
            return default
        if isinstance(current, dict):
            if part not in current:
                return default
            current = current[part]
            continue
        current = getattr(current, part, default)
        if current is default and index < len(parts) - 1:
            return default
    return current


def resolve_feature_value(row: Any, spec: Any) -> Any:
    if not isinstance(spec, dict):
        return spec
    if "value" in spec:
        return spec["value"]
    if "attr" in spec:
        return attr_value(row, str(spec["attr"]), spec.get("default"))
    if "lte" in spec:
        cfg = dict(spec["lte"])
        value = attr_value(row, str(cfg.get("attr")), cfg.get("default", 0))
        return int(float(value or 0) <= float(cfg.get("value", 0)))
    if "gte" in spec:
        cfg = dict(spec["gte"])
        value = attr_value(row, str(cfg.get("attr")), cfg.get("default", 0))
        return int(float(value or 0) >= float(cfg.get("value", 0)))
    if "in_values" in spec:
        cfg = dict(spec["in_values"])
        value = str(attr_value(row, str(cfg.get("attr")), cfg.get("default", "")) or "")
        values = {str(item) for item in cfg.get("values", [])}
        return int(value in values)
    if "coalesce" in spec:
        for item in spec["coalesce"]:
            value = resolve_feature_value(row, item)
            if value not in (None, ""):
                return value
        return spec.get("default")
    return spec.get("default")


def feature_frame_from_mapping(row: Any, mapping: dict[str, Any]) -> pd.DataFrame:
    return pd.DataFrame(
        [{str(column): resolve_feature_value(row, spec) for column, spec in mapping.items()}]
    )


def as_feature_frame(value: Any) -> pd.DataFrame:
    if isinstance(value, pd.DataFrame):
        return value.reset_index(drop=True)
    if isinstance(value, dict):
        return pd.DataFrame([value])
    return pd.DataFrame(value).reset_index(drop=True)


def prediction_scalar(raw: Any, path_config: dict[str, Any] | None = None) -> float:
    path = (path_config or {}).get("path", 0)
    value = raw
    if isinstance(value, pd.DataFrame):
        value = value.to_dict(orient="records")
    elif isinstance(value, pd.Series):
        value = value.tolist()
    if hasattr(value, "tolist"):
        value = value.tolist()
    if isinstance(path, int):
        value = value[path]
    elif isinstance(path, str) and path:
        for part in path.split("."):
            if isinstance(value, dict):
                value = value[part]
            elif isinstance(value, (list, tuple)):
                value = value[int(part)]
            else:
                value = getattr(value, part)
    return float(value)


def prediction_items(raw: Any, expected_count: int) -> list[Any]:
    value = raw
    if isinstance(value, pd.DataFrame):
        value = value.to_dict(orient="records")
    elif isinstance(value, pd.Series):
        value = value.tolist()
    elif hasattr(value, "tolist"):
        value = value.tolist()

    if isinstance(value, list):
        if len(value) == expected_count:
            return value
        if expected_count == 1:
            return [value]
        raise RuntimeError(
            f"Prediction output count mismatch: expected {expected_count}, got {len(value)}."
        )
    if expected_count == 1:
        return [value]
    raise RuntimeError(f"Prediction output is not batch-shaped for {expected_count} rows.")


def compare_threshold(value: float, operator: str, threshold: float) -> bool:
    if operator == "<=":
        return value <= threshold
    if operator == "<":
        return value < threshold
    if operator == ">=":
        return value >= threshold
    if operator == ">":
        return value > threshold
    if operator == "==":
        return value == threshold
    raise RuntimeError(f"Unsupported alert operator: {operator}")


def alert_enabled(alert_config: dict[str, Any]) -> bool:
    return bool(alert_config.get("enabled", True))


def apply_postprocess(value: float, row: Any, steps: list[dict[str, Any]]) -> float:
    processed = value
    for step in steps:
        step_type = str(step.get("type") or "").strip()
        if step_type == "min_with_inventory_runway":
            stock = float(attr_value(row, str(step.get("stock_attr")), 0.0) or 0.0)
            usage = float(attr_value(row, str(step.get("usage_attr")), 0.0) or 0.0)
            min_usage = float(step.get("min_usage", 0.1))
            runway = stock / max(usage, min_usage)
            processed = min(processed, runway)
            if "round_digits" in step:
                processed = round(processed, int(step["round_digits"]))
        elif step_type == "multiply":
            processed *= float(step.get("factor", 1.0))
        elif step_type == "subtract_from_attr":
            attr_name = str(step.get("attr") or "").strip()
            if not attr_name:
                raise RuntimeError("subtract_from_attr postprocess requires attr.")
            base = float(attr_value(row, attr_name, step.get("default", 0.0)) or 0.0)
            processed = base - processed
            if "min_value" in step:
                processed = max(float(step["min_value"]), processed)
            if "max_value" in step:
                processed = min(float(step["max_value"]), processed)
            if "round_digits" in step:
                processed = round(processed, int(step["round_digits"]))
        elif step_type:
            raise RuntimeError(f"Unsupported postprocess type: {step_type}")
    return float(processed)


def result_field(row: Any, config: dict[str, Any], key: str) -> str:
    value_key = f"{key}_value"
    attr_key = f"{key}_attr"
    if value_key in config:
        return str(config.get(value_key) or "")
    if attr_key in config:
        return str(attr_value(row, str(config[attr_key]), ""))
    return ""


def default_model_loader(
    model_id: str,
    version: str,
    loading_policy: dict[str, Any] | None = None,
) -> tuple[ResolvedModelPredictor, Any]:
    resolution = resolve_model_runtime(
        model_id,
        version,
        loading_policy=loading_policy,
    )
    return ResolvedModelPredictor(resolution), resolution


def import_callable(path: str):
    module_name, _, attr = str(path).rpartition(".")
    if not module_name or not attr:
        raise RuntimeError(f"Invalid callable path: {path}")
    module = importlib.import_module(module_name)
    fn = getattr(module, attr)
    if not callable(fn):
        raise RuntimeError(f"Configured object is not callable: {path}")
    return fn


def run_scheduled_prediction_jobs(
    *,
    config_path: Path | str,
    entity_resolver: EntityResolver,
    job_ids: set[str] | None = None,
    predicted_for_date: date | None = None,
    model_loader: ModelLoader | None = None,
    feature_builder_resolver: Callable[[str], Callable[[Any], pd.DataFrame]] | None = None,
    on_prediction: OnPrediction | None = None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    payload = load_scheduled_predictions_config(config_path)
    today = predicted_for_date or date.today()
    created_at = datetime.now(timezone.utc).replace(microsecond=0).isoformat()
    loader = model_loader or default_model_loader
    summary: dict[str, Any] = {"jobs": []}
    predictions: list[dict[str, Any]] = []

    for job in payload["jobs"]:
        job_id = str(job["id"])
        if job_ids is not None and job_id not in job_ids:
            continue
        if not bool(job.get("enabled")):
            summary["jobs"].append({"id": job_id, "status": "disabled", "saved": 0})
            continue

        model_id = str(job["model_id"])
        result_model_name = str(job.get("result_model_name") or model_id)
        version = str(job.get("model_version") or "champion")
        entity_source_path = str(job["entity_source"])
        feature_builder_path = str(job["feature_builder"])
        result_config = dict(job["result"])
        alert_config = dict(job["alert"])
        alerts_enabled = alert_enabled(alert_config)
        threshold = float(alert_config["threshold"]) if alerts_enabled else None
        operator = str(alert_config["operator"]) if alerts_enabled else ""
        postprocess = [row for row in job.get("postprocess", []) if isinstance(row, dict)]
        job_mapping = job.get("feature_mapping")

        try:
            model, resolution = loader(
                model_id,
                version,
                job.get("loading_policy") if isinstance(job.get("loading_policy"), dict) else None,
            )
        except Exception as exc:
            summary["jobs"].append(
                {"id": job_id, "status": "model_error", "error": str(exc), "saved": 0}
            )
            continue

        try:
            rows = list(entity_resolver(entity_source_path))
        except Exception as exc:
            summary["jobs"].append(
                {"id": job_id, "status": "entity_error", "error": str(exc), "saved": 0}
            )
            continue

        if not rows:
            summary["jobs"].append({"id": job_id, "status": "no_entities", "saved": 0})
            continue

        context_builder_path = str(job.get("context_builder") or "").strip()
        context_builder = (
            import_callable(context_builder_path) if context_builder_path else None
        )
        feature_builder = None
        if not (isinstance(job_mapping, dict) and job_mapping):
            if feature_builder_resolver is not None:
                feature_builder = feature_builder_resolver(feature_builder_path)
            else:
                feature_builder = import_callable(feature_builder_path)

        prediction_rows: list[tuple[Any, pd.DataFrame]] = []
        errors = 0
        for row in rows:
            try:
                if isinstance(job_mapping, dict) and job_mapping:
                    frame = feature_frame_from_mapping(row, job_mapping)
                elif context_builder is not None:
                    context_key = result_field(row, result_config, "hospital_id")
                    frame = feature_builder(row, context_builder(context_key))
                else:
                    frame = feature_builder(row)
                prediction_rows.append((row, as_feature_frame(frame)))
            except Exception as exc:
                errors += 1
                logger.warning(
                    "Scheduled prediction job=%s row feature build failed: %s",
                    job_id,
                    exc,
                )

        if not prediction_rows:
            summary["jobs"].append(
                {"id": job_id, "status": "feature_error", "saved": 0, "errors": errors}
            )
            continue

        saved = 0
        try:
            batch_frame = pd.concat(
                [frame for _row, frame in prediction_rows], ignore_index=True
            )
            raw_predictions = model.predict(batch_frame)
            prediction_outputs = prediction_items(raw_predictions, len(prediction_rows))
        except Exception as exc:
            errors += len(prediction_rows)
            logger.warning("Scheduled prediction job=%s batch prediction failed: %s", job_id, exc)
            summary["jobs"].append(
                {
                    "id": job_id,
                    "status": "prediction_error",
                    "saved": 0,
                    "errors": errors,
                    "model_source": getattr(resolution, "source_used", ""),
                    "model_source_ref": getattr(resolution, "source_ref", ""),
                    "fallback_reason": getattr(resolution, "fallback_reason", ""),
                }
            )
            continue

        for (row, _frame), prediction_output in zip(
            prediction_rows, prediction_outputs, strict=True
        ):
            try:
                value = prediction_scalar([prediction_output], job.get("prediction_value"))
                value = apply_postprocess(value, row, postprocess)

                entity_id = result_field(row, result_config, "entity_id")
                hospital_id = result_field(row, result_config, "hospital_id")
                blood_type = result_field(row, result_config, "blood_type")
                entity_type = str(result_config.get("entity_type") or "")
                alert_triggered = (
                    compare_threshold(value, operator, float(threshold))
                    if alerts_enabled and threshold is not None
                    else False
                )
                row_payload = ScheduledPredictionRow(
                    entity_id=entity_id,
                    entity_type=entity_type,
                    hospital_id=hospital_id,
                    blood_type=blood_type,
                    model_name=result_model_name,
                    model_version=version,
                    predicted_value=value,
                    predicted_for_date=today.isoformat(),
                    created_at=created_at,
                    alert_triggered=alert_triggered,
                    alert_threshold_used=threshold,
                )
                payload_dict = row_payload.as_dict()
                predictions.append(payload_dict)
                if on_prediction is not None:
                    on_prediction(job, row, payload_dict, resolution)
                saved += 1
            except Exception as exc:
                errors += 1
                logger.warning("Scheduled prediction job=%s row failed: %s", job_id, exc)

        summary["jobs"].append(
            {
                "id": job_id,
                "status": "ok",
                "saved": saved,
                "errors": errors,
                "model_source": getattr(resolution, "source_used", ""),
                "model_source_ref": getattr(resolution, "source_ref", ""),
                "fallback_reason": getattr(resolution, "fallback_reason", ""),
            }
        )

    return summary, predictions