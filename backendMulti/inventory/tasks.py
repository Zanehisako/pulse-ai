from __future__ import annotations

from datetime import date
import logging
from pathlib import Path
from typing import Any

from django.conf import settings

import pandas as pd

from inventory.models import BloodSupply, Donor, HospitalSupplyFeature, PredictionResult
from ml.core.scheduled_predictions import (
    alert_enabled,
    attr_value,
    default_model_loader,
    feature_frame_from_mapping,
    import_callable,
    load_scheduled_predictions_config,
    run_scheduled_prediction_jobs,
)

logger = logging.getLogger(__name__)


def stockout_supply_entities():
    return BloodSupply.objects.select_related("hospital").all()


def latest_hospital_supply_feature_entities():
    return HospitalSupplyFeature.objects.order_by("hospital_id", "-event_timestamp").distinct(
        "hospital_id"
    )


def donor_entities():
    return Donor.objects.all()


def _configured_feature_job(callable_path: str) -> dict[str, Any]:
    payload = _load_config()
    for job in payload.get("jobs", []):
        if isinstance(job, dict) and str(job.get("feature_builder") or "") == callable_path:
            return job
    raise RuntimeError(f"No scheduled prediction job config found for feature builder: {callable_path}")


def _configured_feature_frame(row: Any, callable_path: str) -> pd.DataFrame:
    mapping = _configured_feature_job(callable_path).get("feature_mapping")
    if not isinstance(mapping, dict) or not mapping:
        raise RuntimeError(f"Feature builder {callable_path} requires non-empty feature_mapping config.")
    return feature_frame_from_mapping(row, mapping)


def inventory_risk_feature_frame(row: BloodSupply) -> pd.DataFrame:
    return _configured_feature_frame(row, f"{__name__}.inventory_risk_feature_frame")


def donor_priority_feature_frame(row: Donor) -> pd.DataFrame:
    return _configured_feature_frame(row, f"{__name__}.donor_priority_feature_frame")


def donor_contact_response_feature_frame(row: Donor) -> pd.DataFrame:
    return _configured_feature_frame(row, f"{__name__}.donor_contact_response_feature_frame")


def _load_config() -> dict[str, Any]:
    path = Path(settings.PIOS_SCHEDULED_PREDICTION_CONFIG_PATH)
    payload = load_scheduled_predictions_config(path)
    forecast_config = payload.get("dashboard_forecast", {})
    if forecast_config is not None and not isinstance(forecast_config, dict):
        raise RuntimeError("Scheduled prediction dashboard_forecast must be an object.")
    if isinstance(forecast_config, dict):
        cap_policy = forecast_config.get("cap_policy", {})
        if cap_policy is not None and not isinstance(cap_policy, dict):
            raise RuntimeError("Scheduled prediction dashboard_forecast.cap_policy must be an object.")
        if isinstance(cap_policy, dict) and bool(cap_policy.get("enabled", False)):
            horizon_max_factors = cap_policy.get("horizon_max_factors")
            if not isinstance(horizon_max_factors, dict) or not horizon_max_factors:
                raise RuntimeError(
                    "Scheduled prediction forecast cap_policy requires non-empty horizon_max_factors."
                )
            for key in ("default_max_factor", "min_floor_units"):
                if key not in cap_policy:
                    raise RuntimeError(f"Scheduled prediction forecast cap_policy is missing {key}.")
                float(cap_policy[key])
            for horizon, factor in horizon_max_factors.items():
                if not str(horizon).strip():
                    raise RuntimeError("Scheduled prediction forecast cap_policy has an empty horizon.")
                float(factor)
    for job in payload.get("jobs", []):
        if not isinstance(job, dict):
            continue
        job_id = str(job.get("id") or "").strip()
        alert = job.get("alert")
        if not isinstance(alert, dict):
            raise RuntimeError(f"Scheduled prediction job {job_id} alert must be an object.")
        if "requires_hospital_scope" in alert and not isinstance(
            alert["requires_hospital_scope"], bool
        ):
            raise RuntimeError(
                f"Scheduled prediction job {job_id} alert.requires_hospital_scope must be a boolean."
            )
        if bool(alert.get("enabled", True)):
            for key in ("operator", "threshold"):
                if key not in alert:
                    raise RuntimeError(f"Scheduled prediction job {job_id} alert is missing {key}.")
    return payload


def _django_entity_resolver(entity_source_path: str):
    return import_callable(entity_source_path)()


def _configured_context(row: Any, config: dict[str, Any]) -> dict[str, Any]:
    context: dict[str, Any] = {}
    for key in config:
        if key.endswith("_attr"):
            field = key.removesuffix("_attr")
            context[field] = attr_value(row, str(config[key]), None)
        elif key.endswith("_value"):
            field = key.removesuffix("_value")
            context[field] = config[key]
    return context


def _evaluate_scheduled_alerts(
    *,
    model_id: str,
    prediction_value: float,
    feature_input: dict,
    source: str,
) -> None:
    try:
        from alerts.services.rule_engine import AlertEngine

        AlertEngine().evaluate_prediction(
            model_id=model_id,
            prediction_result={"prediction": prediction_value},
            feature_input=feature_input,
            source=source,
        )
    except Exception as exc:
        logger.error(
            "Scheduled alert evaluation failed for model=%s source=%s: %s",
            model_id,
            source,
            exc,
        )


def run_configured_scheduled_predictions(*, job_ids: set[str] | None = None) -> dict[str, Any]:
    today = date.today()

    def _persist_prediction(
        job: dict[str, Any],
        row: Any,
        prediction: dict[str, Any],
        resolution: Any,
    ) -> None:
        alert_config = dict(job["alert"])
        alerts_enabled = alert_enabled(alert_config)
        alert_context = _configured_context(row, dict(job.get("alert_context") or {}))
        loaded_name = getattr(getattr(resolution, "runtime", None), "model_id", str(job["model_id"]))

        PredictionResult.objects.update_or_create(
            entity_id=prediction["entity_id"],
            model_name=prediction["model_name"],
            predicted_for_date=today,
            defaults={
                "entity_type": prediction["entity_type"],
                "hospital_id": prediction["hospital_id"],
                "blood_type": prediction["blood_type"],
                "model_version": prediction["model_version"],
                "predicted_value": prediction["predicted_value"],
                "alert_triggered": prediction["alert_triggered"],
                "alert_threshold_used": prediction["alert_threshold_used"],
            },
        )
        if alerts_enabled:
            _evaluate_scheduled_alerts(
                model_id=loaded_name,
                prediction_value=float(prediction["predicted_value"]),
                feature_input={
                    "entity_id": prediction["entity_id"],
                    "entity_type": prediction["entity_type"],
                    "hospital_id": prediction["hospital_id"],
                    "blood_type": prediction["blood_type"],
                    "predicted_value": prediction["predicted_value"],
                    "requires_hospital_scope": bool(
                        alert_config.get("requires_hospital_scope", False)
                    ),
                    **alert_context,
                },
                source=str(job.get("alert_source") or job["id"]),
            )

    def _feature_builder_resolver(callable_path: str):
        return import_callable(callable_path)

    summary, _predictions = run_scheduled_prediction_jobs(
        config_path=settings.PIOS_SCHEDULED_PREDICTION_CONFIG_PATH,
        entity_resolver=_django_entity_resolver,
        job_ids=job_ids,
        predicted_for_date=today,
        feature_builder_resolver=_feature_builder_resolver,
        on_prediction=_persist_prediction,
    )
    return summary


def _job_ids_for_legacy_name(legacy_name: str) -> set[str]:
    payload = _load_config()
    matches = {
        str(job.get("id"))
        for job in payload.get("jobs", [])
        if legacy_name in [str(item) for item in job.get("legacy_names", [])]
    }
    if not matches:
        raise RuntimeError(f"No scheduled prediction job maps to legacy name: {legacy_name}")
    return matches


def configured_dashboard_forecast_jobs() -> dict[str, dict[str, str]]:
    payload = _load_config()
    mappings: dict[str, dict[str, str]] = {}
    for job in payload.get("jobs", []):
        if not isinstance(job, dict) or not bool(job.get("enabled")):
            continue
        horizon = str(job.get("dashboard_horizon") or "").strip()
        if not horizon:
            continue
        model_name = str(job.get("result_model_name") or job.get("model_id") or "").strip()
        if not model_name:
            continue
        mappings[horizon] = {
            "result_model_name": model_name,
            "horizon_type": str(job.get("forecast_output_type") or job.get("horizon_type") or "absolute").strip() or "absolute",
            "source_model_id": str(job.get("model_id") or "").strip(),
            "model_version": str(job.get("model_version") or "champion").strip() or "champion",
            "job_id": str(job.get("id") or "").strip(),
        }
    return mappings


def configured_dashboard_forecast_cap_policy() -> dict[str, Any]:
    payload = _load_config()
    forecast_config = payload.get("dashboard_forecast")
    if not isinstance(forecast_config, dict):
        return {"enabled": False}
    cap_policy = forecast_config.get("cap_policy")
    if not isinstance(cap_policy, dict) or not bool(cap_policy.get("enabled", False)):
        return {"enabled": False}
    horizon_max_factors = {
        str(horizon): float(factor)
        for horizon, factor in dict(cap_policy.get("horizon_max_factors") or {}).items()
        if str(horizon).strip()
    }
    return {
        "enabled": True,
        "default_max_factor": float(cap_policy["default_max_factor"]),
        "min_floor_units": float(cap_policy["min_floor_units"]),
        "horizon_max_factors": horizon_max_factors,
    }


def configured_forecast_result_models() -> dict[str, str]:
    return {
        horizon: row["result_model_name"]
        for horizon, row in configured_dashboard_forecast_jobs().items()
        if row.get("result_model_name")
    }


def configured_stockout_days_result_models() -> list[str]:
    payload = _load_config()
    names: list[str] = []
    for job in payload.get("jobs", []):
        if not isinstance(job, dict) or not bool(job.get("enabled")):
            continue
        if str(job.get("dashboard_role") or "") != "stockout_days":
            continue
        model_name = str(job.get("result_model_name") or job.get("model_id") or "").strip()
        if model_name and model_name not in names:
            names.append(model_name)
    return names


def configured_donor_score_result_model() -> str:
    payload = _load_config()
    for job in payload.get("jobs", []):
        if not isinstance(job, dict) or not bool(job.get("enabled")):
            continue
        if str(job.get("dashboard_role") or "") != "donor_scores":
            continue
        return str(job.get("result_model_name") or job.get("model_id") or "")
    return ""


def preload_configured_scheduled_models(*, job_ids: set[str] | None = None) -> dict[str, Any]:
    payload = _load_config()
    summary: dict[str, Any] = {"models": []}
    seen: set[tuple[str, str]] = set()

    for job in payload.get("jobs", []):
        if not isinstance(job, dict) or not bool(job.get("enabled")):
            continue
        job_id = str(job.get("id") or "").strip()
        if job_ids is not None and job_id not in job_ids:
            continue

        model_id = str(job.get("model_id") or "").strip()
        version = str(job.get("model_version") or "champion").strip() or "champion"
        if not model_id:
            continue

        cache_key = (model_id, version)
        if cache_key in seen:
            continue
        seen.add(cache_key)

        try:
            model, resolution = default_model_loader(
                model_id,
                version,
                job.get("loading_policy")
                if isinstance(job.get("loading_policy"), dict)
                else None,
            )
            summary["models"].append(
                {
                    "job_id": job_id,
                    "model_id": model_id,
                    "model_version": version,
                    "status": "ok",
                    "model_source": getattr(resolution, "source_used", ""),
                    "model_source_ref": getattr(resolution, "source_ref", ""),
                    "fallback_reason": getattr(resolution, "fallback_reason", ""),
                }
            )
        except Exception as exc:
            summary["models"].append(
                {
                    "job_id": job_id,
                    "model_id": model_id,
                    "model_version": version,
                    "status": "error",
                    "error": str(exc),
                }
            )

    return summary


def run_stockout_predictions_sync():
    return run_configured_scheduled_predictions(
        job_ids=_job_ids_for_legacy_name("run_stockout_predictions_sync")
    )


def run_hospital_shortage_predictions_sync():
    return run_configured_scheduled_predictions(
        job_ids=_job_ids_for_legacy_name("run_hospital_shortage_predictions_sync")
    )


def run_donor_propensity_predictions_sync():
    return run_configured_scheduled_predictions(
        job_ids=_job_ids_for_legacy_name("run_donor_propensity_predictions_sync")
    )


def run_stockout_predictions():
    return run_configured_scheduled_predictions(
        job_ids=_job_ids_for_legacy_name("run_stockout_predictions")
    )


def run_dashboard_predictions_sync():
    return run_configured_scheduled_predictions(
        job_ids=_job_ids_for_legacy_name("run_dashboard_predictions_sync")
    )


def run_dashboard_predictions():
    return run_configured_scheduled_predictions(
        job_ids=_job_ids_for_legacy_name("run_dashboard_predictions")
    )


def run_hospital_shortage_predictions():
    return run_configured_scheduled_predictions(
        job_ids=_job_ids_for_legacy_name("run_hospital_shortage_predictions")
    )


def run_donor_propensity_predictions():
    return run_configured_scheduled_predictions(
        job_ids=_job_ids_for_legacy_name("run_donor_propensity_predictions")
    )
