from __future__ import annotations

import json
import time
from dataclasses import dataclass
from typing import Any

from django.db import transaction

from alerts.services.rule_engine import AlertEngine
from ml.core.feature_store import resolve_prediction_features
from ml.core.hybrid_stockout import build_hybrid_stockout_prediction
from ml.core.model_loading import (
    run_prediction_batch_with_fallback,
    run_prediction_with_fallback,
)
from ml.core.startup import (
    get_orchestrator,
    get_orchestrator_config_status,
    get_registry,
    get_runtime_config_status,
    is_ready,
    refresh_runtime_state,
    reload_model_config_file,
    sync_orchestrator_catalog,
)
from ml.core.utils import safe_slug, sanitize_features
from ml.models import OrchestratorVariant, PredictionLog


@dataclass(frozen=True)
class ControlResult:
    payload: dict[str, Any]
    status_code: int = 200


def runtime_to_control_row(runtime) -> dict[str, Any]:
    return {
        "model_id": runtime.model_id,
        "aliases": runtime.aliases,
        "slug": runtime.slug,
        "description": runtime.description,
        "model_type": runtime.model_type,
        "status": runtime.status,
        "load_error": runtime.load_error,
        "feature_names": runtime.feature_names,
        "feature_info": runtime.feature_info,
        "examples": runtime.examples,
        "defaults": runtime.defaults,
        "file_path": str(runtime.file_path),
        "predict_endpoint": f"/api/ml/models/{runtime.model_id}/predict/",
    }


def _default_orchestrator_repo_id() -> str:
    default = OrchestratorVariant._meta.get_field("repo_id").default
    value = default() if callable(default) else default
    return str(value or "").strip()


def _prediction_summary_value(output):
    if not isinstance(output, dict):
        return str(output)
    probability = output.get("probability")
    if isinstance(probability, (int, float)):
        return f"{float(probability):.4f} ({float(probability) * 100:.1f}%)"
    if "prediction" in output:
        prediction = output.get("prediction")
        unit = str(output.get("prediction_unit") or "").strip()
        name = str(output.get("prediction_name") or "").strip()
        if isinstance(prediction, (int, float)) and unit:
            label = name.replace("_", " ") if name else "prediction"
            return f"{float(prediction):.2f} {unit} ({label})"
        return str(prediction)
    return str(output)


def _compact_prediction_row(row):
    if not isinstance(row, dict):
        return row
    public_keys = (
        "index",
        "rank",
        "donor_id",
        "blood_type",
        "country_code",
        "region",
        "preferred_site",
        "recency_days",
        "donation_count_last_12m",
        "eligibility_status",
        "model_id",
        "model_score",
        "prediction",
        "probability",
        "prediction_name",
        "prediction_unit",
        "prediction_task_type",
    )
    compact = {key: row[key] for key in public_keys if key in row}
    model_output = row.get("model_output")
    if isinstance(model_output, dict):
        for key in (
            "prediction",
            "probability",
            "prediction_name",
            "prediction_unit",
            "prediction_task_type",
        ):
            if key in model_output and key not in compact:
                compact[key] = model_output[key]
    return compact or {
        key: value
        for key, value in row.items()
        if key not in {"model_output", "feature_resolution"}
    }


def _compact_prediction_output(output):
    if not isinstance(output, dict):
        return output
    important_keys = {
        "answer",
        "results",
        "data",
        "query_kind",
        "model_id",
        "entity_key",
        "row_count",
        "prediction",
        "probability",
        "prediction_name",
        "prediction_unit",
        "prediction_task_type",
        "stockout_probability",
        "estimated_days_until_stockout",
        "horizon",
        "risk_level",
        "recommended_action",
        "confidence",
        "blood_component",
        "blood_group",
        "location",
        "worst_component",
        "worst_blood_group",
        "component_count",
        "worst_component_prediction",
        "warnings",
        "filters",
        "fields",
        "table",
        "aggregate",
        "limit",
    }
    compact = {
        key: value
        for key, value in output.items()
        if key in important_keys or str(key).startswith("risk_")
    }
    rows = output.get("rows")
    if isinstance(rows, list):
        compact["rows"] = [_compact_prediction_row(row) for row in rows]
    return compact or output


def public_prediction_response(result, *, tools_used=None):
    if not isinstance(result, dict):
        return {"results": result, "tools_used": tools_used or []}

    execution_rows = result.get("execution_results")
    if not isinstance(execution_rows, list):
        execution_rows = []

    tools = [
        str(row.get("tool") or "").strip()
        for row in execution_rows
        if isinstance(row, dict) and str(row.get("tool") or "").strip()
    ]
    if tools_used:
        tools.extend(str(tool).strip() for tool in tools_used if str(tool).strip())
    tools = list(dict.fromkeys(tools))

    errors = []
    if result.get("error"):
        errors.append(str(result.get("error")))
    for row in execution_rows:
        if isinstance(row, dict) and row.get("error"):
            errors.append(str(row.get("error")))

    if "natural_language_response" in result or execution_rows:
        details = []
        for row in execution_rows:
            if not isinstance(row, dict):
                continue
            output = row.get("output", row.get("summary"))
            details.append(
                {
                    "tool": row.get("tool"),
                    "result": _compact_prediction_output(output),
                }
            )
        results_payload = {
            "summary": result.get("natural_language_response", ""),
            "details": details,
        }
    else:
        results_payload = _compact_prediction_output(result)

    payload = {"results": results_payload, "tools_used": tools}
    if execution_rows:
        payload["execution_results"] = [
            {
                **{
                    key: row.get(key)
                    for key in ("tool", "success", "error", "summary")
                    if isinstance(row, dict) and key in row
                },
                **(
                    {
                        "inputs": _compact_prediction_output(row.get("inputs"))
                    }
                    if isinstance(row, dict) and "inputs" in row
                    else {}
                ),
                "output": _compact_prediction_output(row.get("output"))
                if isinstance(row, dict) and "output" in row
                else row.get("result") if isinstance(row, dict) else row,
            }
            for row in execution_rows
            if isinstance(row, dict)
        ]
    if result.get("planner_mode"):
        payload["planner_mode"] = result.get("planner_mode")
    if errors:
        payload["error"] = "; ".join(dict.fromkeys(errors))
    return payload


def _build_direct_prediction_response(
    runtime,
    query,
    prediction_output,
    plan_arguments,
    feature_resolution=None,
):
    summary_value = _prediction_summary_value(prediction_output)
    response = {
        "success": True,
        "query": query,
        "planner_mode": "direct",
        "plan": {
            "reasoning": "Direct single-model prediction path; orchestrator LLM planning skipped.",
            "steps": [{"tool": runtime.model_id, "arguments": plan_arguments}],
        },
        "execution_results": [
            {
                "tool": runtime.model_id,
                "output": prediction_output,
                "success": True,
            }
        ],
        "natural_language_response": f"{runtime.model_id}: {summary_value}",
        "verification_data": [f"{runtime.model_id}: {summary_value}"],
    }
    if isinstance(feature_resolution, dict) and feature_resolution:
        response["feature_resolution"] = feature_resolution
    return response


def _build_direct_batch_prediction_response(
    runtime,
    query,
    prediction_rows,
    plan_arguments,
):
    preview_parts = []
    for row in prediction_rows[:3]:
        label = str(row.get("index", ""))
        value = _prediction_summary_value(row)
        preview_parts.append(f"{label}: {value}" if label else value)
    preview = "; ".join(preview_parts)
    output = {
        "query_kind": "batch_predictions",
        "model_id": runtime.model_id,
        "row_count": len(prediction_rows),
        "rows": prediction_rows,
        "answer": (
            f"Ran {len(prediction_rows)} prediction(s) with {runtime.model_id}."
            + (f" Preview: {preview}." if preview else "")
        ),
    }
    return {
        "success": True,
        "query": query,
        "planner_mode": "direct_batch",
        "plan": {
            "reasoning": "Direct batch model prediction path; orchestrator LLM planning skipped.",
            "steps": [{"tool": runtime.model_id, "arguments": plan_arguments}],
        },
        "execution_results": [
            {
                "tool": runtime.model_id,
                "inputs": plan_arguments,
                "output": output,
                "success": True,
            }
        ],
        "natural_language_response": output["answer"],
        "verification_data": [output["answer"]],
    }


def _normalize_direct_prediction_payload(payload: dict[str, Any]) -> ControlResult | dict[str, Any]:
    if not isinstance(payload, dict):
        return ControlResult({"error": "Request body must be a JSON object.", "tools_used": []}, 400)
    raw_features = payload.get("features") or {}
    if not isinstance(raw_features, dict):
        return ControlResult({"error": "features must be a JSON object.", "tools_used": []}, 400)
    raw_inputs = payload.get("inputs") or []
    if not isinstance(raw_inputs, list):
        return ControlResult({"error": "inputs must be a JSON array.", "tools_used": []}, 400)
    input_rows = [
        dict(row)
        for row in raw_inputs
        if isinstance(row, dict)
    ]
    features = dict(raw_features)
    for key in (
        "prediction_timeframe",
        "forecast_timeframe",
        "prediction_horizon_days",
        "forecast_horizon_days",
        "horizon_days",
    ):
        value = payload.get(key)
        if value not in (None, ""):
            features.setdefault(key, value)
            for row in input_rows:
                row.setdefault(key, value)
    return {
        "query": str(payload.get("query") or ""),
        "features": features,
        "input_rows": input_rows,
    }


def get_control_status(*, refresh: bool = True) -> dict[str, Any]:
    if not is_ready():
        return {
            "status": "initializing",
            "ml_ready": False,
            "models": [],
            "orchestrator_models": [],
            "config_runtime": get_runtime_config_status(),
            "orchestrator_config": get_orchestrator_config_status(),
        }

    refresh_error = None
    if refresh:
        try:
            refresh_runtime_state("control-status-read", force=False, warmup=False)
        except Exception as exc:  # noqa: BLE001
            refresh_error = f"{type(exc).__name__}: {exc}"

    registry = get_registry()
    orchestrator = get_orchestrator()
    models = [runtime_to_control_row(runtime) for runtime in registry.list_models()]
    loaded_models = [row for row in models if row.get("status") == "loaded"]
    orchestrator_status = orchestrator.status()
    orchestrator_status["orchestrator_config"] = get_orchestrator_config_status()
    payload = {
        "status": "ok",
        "ml_ready": True,
        "models_total": len(models),
        "models_loaded": len(loaded_models),
        "models": models,
        "orchestrator": orchestrator_status,
        "orchestrator_models": orchestrator.list_available_models(),
        "selected_model_id": orchestrator.preferred_model_id,
        "active_model_id": orchestrator_status.get("active_model_id"),
        "config_runtime": get_runtime_config_status(),
        "orchestrator_config": get_orchestrator_config_status(),
    }
    if refresh_error:
        payload["refresh_error"] = refresh_error
    return payload


def reload_model_config_control(reason: str = "manual-reload") -> ControlResult:
    try:
        reload_result = reload_model_config_file(reason)
    except Exception as exc:  # noqa: BLE001
        return ControlResult(
            {
                "status": "error",
                "error": f"{type(exc).__name__}: {exc}",
                "config_runtime": get_runtime_config_status(),
                "orchestrator_config": get_orchestrator_config_status(),
            },
            422,
        )

    registry = get_registry()
    file_sync = reload_result["file_sync"]
    external_tool_reload = reload_result["external_tool_reload"]
    return ControlResult(
        {
            "status": "ok",
            "changed": bool(
                file_sync.get("changed")
                or reload_result["runtime_config_changed"]
                or reload_result["orchestrator_catalog_changed"]
                or external_tool_reload.get("changed")
            ),
            "file_sync": file_sync,
            "runtime_config_changed": reload_result["runtime_config_changed"],
            "orchestrator_catalog_changed": reload_result[
                "orchestrator_catalog_changed"
            ],
            "external_tool_reload": external_tool_reload,
            "config_hot_reload": reload_result.get("config_hot_reload"),
            "models_total": len(registry.list_models()),
            "models_loaded": len(registry.loaded()),
            "config_runtime": get_runtime_config_status(),
            "orchestrator_config": get_orchestrator_config_status(),
        },
        200,
    )


def normalize_orchestrator_variant_row(row, *, index: int):
    variant_id = safe_slug(str(row.get("id") or row.get("model_id") or "").strip())
    filename = str(row.get("filename") or "").strip()
    default_repo_id = _default_orchestrator_repo_id()
    if not variant_id:
        raise ValueError(f"models[{index}].id is required.")
    if not filename:
        raise ValueError(f"models[{index}].filename is required.")
    known_keys = {
        "id",
        "model_id",
        "name",
        "description",
        "repo_id",
        "filename",
        "size_mb",
        "size_bytes",
        "min_ram_gb",
        "min_vram_mb",
        "n_ctx",
        "n_batch",
    }
    return {
        "variant_id": variant_id,
        "name": str(row.get("name", variant_id)).strip() or variant_id,
        "description": str(row.get("description", "")).strip(),
        "repo_id": str(row.get("repo_id", default_repo_id)).strip()
        or default_repo_id,
        "filename": filename,
        "size_mb": row.get("size_mb"),
        "size_bytes": row.get("size_bytes"),
        "min_ram_gb": row.get("min_ram_gb", 0),
        "min_vram_mb": row.get("min_vram_mb", 0),
        "n_ctx": row.get("n_ctx", 4096),
        "n_batch": row.get("n_batch", 128),
        "metadata_extra": {
            key: value for key, value in row.items() if key not in known_keys
        },
    }


def resolve_orchestrator_variant_row(model_id: str):
    target_slug = safe_slug(model_id)
    sync_orchestrator_catalog(
        "orchestrator-variant-lookup", force=False, warmup=False, wait=False
    )
    orchestrator = get_orchestrator()
    for index, row in enumerate(orchestrator.list_available_models()):
        row_slug = safe_slug(str(row.get("id") or row.get("model_id") or "").strip())
        if row_slug != target_slug:
            continue
        return normalize_orchestrator_variant_row(row, index=index)
    return None


def replace_orchestrator_catalog(
    *,
    models: list[dict[str, Any]],
    selected_model_id: str | None = None,
) -> ControlResult:
    selected_slug = safe_slug(selected_model_id) if selected_model_id else None
    try:
        rows = [
            normalize_orchestrator_variant_row(row, index=index)
            for index, row in enumerate(models)
        ]
    except ValueError as exc:
        return ControlResult({"error": str(exc)}, 422)

    allowed_ids = {row["variant_id"] for row in rows}
    if selected_slug and selected_slug not in allowed_ids:
        return ControlResult(
            {
                "error": f"selected_model_id '{selected_model_id}' is not present in models list."
            },
            422,
        )

    with transaction.atomic():
        for row in rows:
            OrchestratorVariant.objects.update_or_create(
                variant_id=row["variant_id"],
                defaults={
                    "name": row["name"],
                    "description": row["description"],
                    "repo_id": row["repo_id"],
                    "filename": row["filename"],
                    "size_mb": row["size_mb"],
                    "size_bytes": row["size_bytes"],
                    "min_ram_gb": row["min_ram_gb"],
                    "min_vram_mb": row["min_vram_mb"],
                    "n_ctx": row["n_ctx"],
                    "n_batch": row["n_batch"],
                    "metadata_extra": row["metadata_extra"],
                    "is_selected": row["variant_id"] == selected_slug
                    if selected_slug
                    else False,
                },
            )

        if allowed_ids:
            OrchestratorVariant.objects.exclude(variant_id__in=allowed_ids).delete()
        else:
            OrchestratorVariant.objects.all().delete()

    changed = sync_orchestrator_catalog(
        "api-orchestrator-catalog-update",
        force=True,
        warmup=False,
        wait=False,
    )
    orchestrator = get_orchestrator()
    return ControlResult(
        {
            "status": "updated",
            "selected_model_id": orchestrator.preferred_model_id,
            "changed": changed,
            "orchestrator_config": get_orchestrator_config_status(),
            "orchestrator_status": {
                **orchestrator.status(),
                "orchestrator_config": get_orchestrator_config_status(),
            },
        }
    )


def select_orchestrator_model(
    model_id: str,
    *,
    download: bool = True,
    wait: bool = False,
    reason: str = "api-orchestrator-selection-update",
    response_status: str = "selected",
) -> ControlResult:
    selected_id = safe_slug(model_id)
    if not selected_id:
        return ControlResult({"error": "model_id required"}, 422)

    variant_row = resolve_orchestrator_variant_row(selected_id)
    if variant_row is None:
        return ControlResult({"error": f"Variant '{selected_id}' not found"}, 404)

    with transaction.atomic():
        OrchestratorVariant.objects.filter(is_selected=True).update(is_selected=False)
        OrchestratorVariant.objects.update_or_create(
            variant_id=variant_row["variant_id"],
            defaults={
                "name": variant_row["name"],
                "description": variant_row["description"],
                "repo_id": variant_row["repo_id"],
                "filename": variant_row["filename"],
                "size_mb": variant_row["size_mb"],
                "size_bytes": variant_row["size_bytes"],
                "min_ram_gb": variant_row["min_ram_gb"],
                "min_vram_mb": variant_row["min_vram_mb"],
                "n_ctx": variant_row["n_ctx"],
                "n_batch": variant_row["n_batch"],
                "metadata_extra": variant_row["metadata_extra"],
                "is_selected": True,
            },
        )

    changed = sync_orchestrator_catalog(
        reason,
        force=True,
        warmup=download,
        wait=wait,
    )
    selected_variant = OrchestratorVariant.objects.filter(variant_id=selected_id).first()
    orchestrator = get_orchestrator()
    return ControlResult(
        {
            "status": response_status,
            "selected_model_id": orchestrator.preferred_model_id,
            "changed": changed,
            "download_triggered": download,
            "updated_at": (
                selected_variant.updated_at.isoformat()
                if selected_variant is not None
                else None
            ),
            "orchestrator_config": get_orchestrator_config_status(),
            "orchestrator_status": {
                **orchestrator.status(),
                "orchestrator_config": get_orchestrator_config_status(),
            },
        }
    )


def warmup_orchestrator(*, wait: bool = False) -> ControlResult:
    if not is_ready():
        return ControlResult({"status": "initializing"}, 503)
    sync_orchestrator_catalog(
        "orchestrator-warmup", force=False, warmup=False, wait=False
    )
    orchestrator = get_orchestrator()
    orchestrator._ensure_llm(blocking=wait)
    payload = orchestrator.status()
    payload["orchestrator_config"] = get_orchestrator_config_status()
    return ControlResult(payload)


def run_direct_model_prediction(
    model_id: str,
    payload: dict[str, Any],
    *,
    user=None,
    alert_source: str = "predict-model",
) -> ControlResult:
    if not is_ready():
        return ControlResult({"error": "ML models still loading.", "tools_used": []}, 503)

    normalized = _normalize_direct_prediction_payload(payload)
    if isinstance(normalized, ControlResult):
        return normalized

    refresh_runtime_state("predict-model", force=False, warmup=False)
    registry = get_registry()
    runtime = registry.get(model_id)
    if runtime is None:
        return ControlResult(
            {"error": f"Unknown model: {model_id}", "tools_used": []},
            404,
        )
    if runtime.status != "loaded":
        return ControlResult(
            {
                "error": runtime.load_error or "Model is not loaded.",
                "tools_used": [runtime.model_id],
            },
            503,
        )

    query = (
        normalized.get("query")
        or f"Run model {runtime.model_id} with these provided feature values."
    )
    requested_features = sanitize_features(normalized.get("features", {}))
    input_rows = [
        sanitize_features(row)
        for row in normalized.get("input_rows", [])
        if isinstance(row, dict)
    ]
    start = time.perf_counter()

    try:
        if input_rows:
            resolved_rows = [
                resolve_prediction_features(runtime, row) for row in input_rows
            ]
            prediction_outputs = run_prediction_batch_with_fallback(
                runtime,
                [resolved.features for resolved in resolved_rows],
                registry=registry,
            )
            runtime_defaults = getattr(runtime, "defaults", {}) or {}
            prediction_rows = []
            for index, (row, resolved, prediction_output) in enumerate(
                zip(input_rows, resolved_rows, prediction_outputs),
                1,
            ):
                if (
                    isinstance(runtime_defaults, dict)
                    and "hybrid_stockout" in runtime_defaults
                ):
                    hybrid_output = build_hybrid_stockout_prediction(
                        registry=registry,
                        regression_runtime=runtime,
                        request_features=row,
                        resolved_regression_features=resolved.features,
                        regression_output=prediction_output,
                        query=query,
                    )
                    if hybrid_output:
                        prediction_output = {
                            **prediction_output,
                            **hybrid_output,
                        }
                prediction_rows.append(
                    {
                        "index": index,
                        "model_id": runtime.model_id,
                        **prediction_output,
                        "feature_resolution": resolved.metadata,
                    }
                )
            if len(prediction_rows) != len(input_rows):
                raise ValueError(
                    "Batch prediction returned fewer outputs than requested rows."
                )
            result = _build_direct_batch_prediction_response(
                runtime,
                query,
                prediction_rows,
                {"inputs": input_rows},
            )
            logged_features = {"inputs": input_rows}
            alert_features = {"inputs": input_rows}
        else:
            resolved = resolve_prediction_features(runtime, requested_features)
            prediction_output = run_prediction_with_fallback(
                runtime,
                resolved.features,
                registry=registry,
            )
            runtime_defaults = getattr(runtime, "defaults", {}) or {}
            if (
                isinstance(runtime_defaults, dict)
                and "hybrid_stockout" in runtime_defaults
            ):
                hybrid_output = build_hybrid_stockout_prediction(
                    registry=registry,
                    regression_runtime=runtime,
                    request_features=requested_features,
                    resolved_regression_features=resolved.features,
                    regression_output=prediction_output,
                    query=query,
                )
                if hybrid_output:
                    prediction_output = {**prediction_output, **hybrid_output}
            result = _build_direct_prediction_response(
                runtime,
                query,
                prediction_output,
                resolved.features,
                feature_resolution=resolved.metadata,
            )
            logged_features = requested_features
            alert_features = resolved.features
    except Exception as exc:  # noqa: BLE001
        return ControlResult(
            {"error": f"Prediction error: {exc}", "tools_used": [runtime.model_id]},
            400,
        )

    elapsed_ms = (time.perf_counter() - start) * 1000
    try:
        PredictionLog.objects.create(
            user=user if getattr(user, "is_authenticated", False) else None,
            model_id_used=runtime.model_id,
            query=query,
            features_input=logged_features,
            prediction_output=result,
            planner_mode=result.get("planner_mode", ""),
            success=True,
            response_time_ms=round(elapsed_ms, 2),
        )
    except Exception:
        pass

    try:
        AlertEngine().evaluate_prediction(
            model_id=runtime.model_id,
            prediction_result=result,
            feature_input=alert_features,
            source=alert_source,
        )
    except Exception:
        pass
    return ControlResult(
        public_prediction_response(result, tools_used=[runtime.model_id]),
        200,
    )


def parse_json_object(raw: str) -> dict[str, Any]:
    payload = json.loads(raw or "{}")
    if not isinstance(payload, dict):
        raise ValueError("JSON payload must be an object.")
    return payload
