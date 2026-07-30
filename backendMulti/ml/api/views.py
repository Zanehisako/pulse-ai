import json
import logging
import os
import queue
import shutil
import threading
import time
from pathlib import Path
from xml.parsers.expat import model

from django.conf import settings
from django.http import StreamingHttpResponse
from django.db import transaction
from drf_spectacular.utils import OpenApiExample, extend_schema
from rest_framework.renderers import BaseRenderer
from rest_framework import status
from rest_framework.permissions import AllowAny
from rest_framework.response import Response
from rest_framework.settings import api_settings
from rest_framework.views import APIView

from alerts.services.rule_engine import AlertEngine
from ml.core.hybrid_stockout import (
    build_hybrid_stockout_prediction,
    find_hybrid_stockout_regression_runtime,
)
from ml.core.model_stats import normalize_model_stats_payload
from ml.core.startup import (
    get_orchestrator,
    get_orchestrator_config_status,
    get_registry,
    get_runtime_config_status,
    is_ready,
    refresh_runtime_state,
    sync_orchestrator_catalog,
)
from ml.core.utils import is_placeholder_text, safe_slug, sanitize_features
from ml.models import (
    DriftReport,
    MLModelConfig,
    ModelStats,
    PredictionLog,
)
from ml.services.control_center import (
    get_control_status,
    reload_model_config_control,
    replace_orchestrator_catalog,
    run_direct_model_prediction,
    select_orchestrator_model,
    warmup_orchestrator,
)
from ml.services.training_scheduler import (
    get_schedule_config_path,
    get_schedule_status,
    load_schedule_config,
    reload_schedule_config,
)
from ml.services.training_run_manager import get_run, start_run

from .serializers import (
    ConfigReloadResponseSerializer,
    ControlStatusResponseSerializer,
    DriftReportDetailSerializer,
    DriftReportsListResponseSerializer,
    DriftRunRequestSerializer,
    DriftRunResponseSerializer,
    DriftStatusResponseSerializer,
    EmptySerializer,
    ErrorResponseSerializer,
    HealthResponseSerializer,
    InitializingResponseSerializer,
    MLIndexResponseSerializer,
    ModelDetailResponseSerializer,
    ModelListResponseSerializer,
    ModelUploadRequestSerializer,
    ModelUploadResponseSerializer,
    ModelsStatsListResponseSerializer,
    NLPredictRequestSerializer,
    OrchestratorCatalogRequestSerializer,
    OrchestratorDownloadRequestSerializer,
    OrchestratorModelListResponseSerializer,
    OrchestratorSelectionResponseSerializer,
    OrchestratorSelectRequestSerializer,
    OrchestratorStatusResponseSerializer,
    OrchestratorWarmupRequestSerializer,
    PredictionResponseSerializer,
    PredictRequestSerializer,
    RemoveModelResponseSerializer,
    RemoveModelStatsResponseSerializer,
    RuntimeConfigStatusSerializer,
    SchedulerConfigResponseSerializer,
    SchedulerConfigUpdateSerializer,
    SchedulerReloadResponseSerializer,
    SchedulerRunRequestSerializer,
    SchedulerRunStartResponseSerializer,
    SchedulerRunStatusResponseSerializer,
    SchedulerStatusResponseSerializer,
    ToolListResponseSerializer,
    ToolRunRequestSerializer,
)

logger = logging.getLogger(__name__)


class EventStreamRenderer(BaseRenderer):
    media_type = "text/event-stream"
    format = "event-stream"
    charset = "utf-8"

    def render(self, data, accepted_media_type=None, renderer_context=None):
        if data is None:
            return ""
        body = json.dumps(data, ensure_ascii=True, default=str)
        return f"data: {body}\n\n"


def _runtime_to_response(runtime):
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
        "generic_predict_endpoint": f"/api/ml/models/{runtime.model_id}/predict/",
    }
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


def _public_prediction_response(result, *, tools_used=None):
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
    tool_summaries = result.get("tool_summaries")
    if isinstance(tool_summaries, list):
        payload["tool_summaries"] = tool_summaries
    if result.get("planner_mode"):
        payload["planner_mode"] = result.get("planner_mode")
    if errors:
        payload["error"] = "; ".join(dict.fromkeys(errors))
    return payload


def _build_direct_prediction_response(
    runtime, query, prediction_output, plan_arguments, feature_resolution=None
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


def _infer_forced_model_ids(query, registry):
    normalized_query = safe_slug(query or "")
    if not normalized_query:
        return None

    matches = []
    for runtime in registry.loaded():
        candidate_names = [runtime.model_id, *(getattr(runtime, "aliases", []) or [])]
        if any(safe_slug(name) in normalized_query for name in candidate_names if name):
            matches.append(runtime.model_id)

    unique_matches = sorted(set(matches))
    if len(unique_matches) == 1:
        return [unique_matches[0]]
    return None


def _load_model_feature_names() -> dict[str, list[str]]:
    lookup: dict[str, list[str]] = {}
    for row in MLModelConfig.objects.values("model_id", "features"):
        model_id = str(row.get("model_id") or "").strip()
        features = row.get("features")
        if not model_id or not isinstance(features, list):
            continue
        lookup[model_id] = [
            str(feature) for feature in features if isinstance(feature, str)
        ]
    return lookup


def _serialize_model_stats_entry(
    *, model_id, identifier, stats, made_at, feature_names=None
):
    made_at_value = ""
    if made_at:
        try:
            made_at_value = made_at.isoformat()
        except AttributeError:
            made_at_value = str(made_at)
    return {
        "model_id": model_id,
        "identifier": identifier,
        "stats": normalize_model_stats_payload(stats, feature_names=feature_names),
        "made_at": made_at_value,
    }


def _load_configured_metrics_payload(runtime):
    defaults = getattr(runtime, "defaults", None)
    if not isinstance(defaults, dict):
        return {}

    metrics_path = str(defaults.get("metrics_path") or "").strip()
    if not metrics_path:
        return {}

    path = Path(os.path.expandvars(metrics_path)).expanduser()
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}

    if not isinstance(payload, dict):
        return {}
    return {**payload, "_metrics_source": str(path)}


def _collect_model_stats_entries():
    stats_rows = list(ModelStats.objects.all())
    stats_by_id = {row.model_id: row for row in stats_rows}
    feature_names_by_model_id = _load_model_feature_names()
    entries = []
    seen_model_ids = set()

    if is_ready():
        refresh_runtime_state("model-stats-read", force=False, warmup=False)
        registry = get_registry()
        for runtime in registry.list_models():
            stats_row = stats_by_id.get(runtime.model_id)
            runtime_feature_names = runtime.feature_names or feature_names_by_model_id.get(
                runtime.model_id
            )
            entries.append(
                _serialize_model_stats_entry(
                    model_id=runtime.model_id,
                    identifier=(
                        stats_row.identifier
                        if stats_row is not None
                        and getattr(stats_row, "identifier", "")
                        else runtime.slug
                    ),
                    stats=(
                        stats_row.stats
                        if stats_row is not None
                        else _load_configured_metrics_payload(runtime)
                    ),
                    made_at=stats_row.made_at if stats_row is not None else "",
                    feature_names=runtime_feature_names,
                )
            )
            seen_model_ids.add(runtime.model_id)

    for stats_row in stats_rows:
        if stats_row.model_id in seen_model_ids:
            continue
        entries.append(
            _serialize_model_stats_entry(
                model_id=stats_row.model_id,
                identifier=stats_row.identifier,
                stats=stats_row.stats,
                made_at=stats_row.made_at,
                feature_names=feature_names_by_model_id.get(stats_row.model_id),
            )
        )
    return entries


# ══════════════════════════════════════════════════════════
#  CONFIG / HEALTH
# ══════════════════════════════════════════════════════════


class MLIndexView(APIView):
    serializer_class = EmptySerializer

    @extend_schema(
        tags=["ML"],
        operation_id="ml_index",
        summary="ML API entrypoint",
        responses={status.HTTP_200_OK: MLIndexResponseSerializer},
    )
    def get(self, request):
        return Response(
            {
                "service": "PIOS ML API",
                "version": "2.0.0",
                "ml_ready": is_ready(),
                "admin": "/admin/",
                "endpoints": {
                    "health": "/api/ml/health/",
                    "models": "/api/ml/models/",
                    "model_stats": "/api/ml/models_stats/",
                    "config_runtime": "/api/ml/config/runtime/",
                    "predict_nl": "/api/ml/predict/nl/",
                    "predict_stockout_hybrid": "/api/ml/predict/stockout-hybrid/",
                    "tools": "/api/ml/tools/",
                    "orchestrator_models": "/api/ml/orchestrator/models/",
                    "orchestrator_select": "/api/ml/orchestrator/models/selected/",
                    "orchestrator_warmup": "/api/ml/orchestrator/warmup/",
                    "orchestrator_status": "/api/ml/orchestrator/status/",
                },
            }
        )


class HealthView(APIView):
    serializer_class = EmptySerializer

    @extend_schema(
        tags=["ML"],
        operation_id="ml_health",
        summary="Check ML readiness and orchestrator status",
        responses={
            status.HTTP_200_OK: HealthResponseSerializer,
            status.HTTP_503_SERVICE_UNAVAILABLE: HealthResponseSerializer,
        },
    )
    def get(self, request):
        if not is_ready():
            return Response(
                {"status": "initializing", "message": "ML models are still loading."},
                status=503,
            )

        refresh_runtime_state("health-read", force=False, warmup=False)
        registry = get_registry()
        orchestrator = get_orchestrator()
        models = registry.list_models()
        return Response(
            {
                "status": "ok",
                "models_total": len(models),
                "models_loaded": len([m for m in models if m.status == "loaded"]),
                "orchestrator": orchestrator.status(),
                "config_runtime": get_runtime_config_status(),
                "orchestrator_config": get_orchestrator_config_status(),
            }
        )


class ModelListView(APIView):
    serializer_class = EmptySerializer

    @extend_schema(
        tags=["ML"],
        operation_id="ml_list_models",
        summary="List available ML models",
        responses={
            status.HTTP_200_OK: ModelListResponseSerializer,
            status.HTTP_503_SERVICE_UNAVAILABLE: InitializingResponseSerializer,
        },
    )
    def get(self, request):
        if not is_ready():
            return Response({"status": "initializing", "models": []}, status=503)

        refresh_runtime_state("model-list-read", force=False, warmup=False)
        registry = get_registry()
        return Response(
            {
                "models": [
                    _runtime_to_response(model) for model in registry.list_models()
                ]
            }
        )


class ModelDetailView(APIView):
    serializer_class = EmptySerializer

    @extend_schema(
        tags=["ML"],
        operation_id="ml_get_model",
        summary="Get details for a specific ML model",
        responses={
            status.HTTP_200_OK: ModelDetailResponseSerializer,
            status.HTTP_404_NOT_FOUND: ErrorResponseSerializer,
            status.HTTP_503_SERVICE_UNAVAILABLE: InitializingResponseSerializer,
        },
    )
    def get(self, request, model_id):
        if not is_ready():
            return Response({"status": "initializing"}, status=503)

        refresh_runtime_state("model-detail-read", force=False, warmup=False)
        registry = get_registry()
        rt = registry.get(model_id)
        if rt is None:
            return Response({"error": f"Unknown model: {model_id}"}, status=404)
        return Response(_runtime_to_response(rt))


class RemoveModelView(APIView):
    serializer_class = EmptySerializer

    @extend_schema(
        tags=["ML"],
        operation_id="remove_ml_model",
        summary="Remove a ML model from registry",
        responses={
            status.HTTP_200_OK: RemoveModelResponseSerializer,
            status.HTTP_404_NOT_FOUND: ErrorResponseSerializer,
            status.HTTP_503_SERVICE_UNAVAILABLE: InitializingResponseSerializer,
        },
    )
    def post(self, request, model_id):
        if not is_ready():
            return Response({"status": "initializing"}, status=503)

        registry = get_registry()
        runtime = registry.get(model_id)

        candidate_ids = {str(model_id).strip()}
        if runtime is not None:
            candidate_ids.add(str(runtime.model_id).strip())
            candidate_ids.add(str(getattr(runtime, "slug", "")).strip())
            for alias in getattr(runtime, "aliases", []) or []:
                alias_text = str(alias).strip()
                if alias_text:
                    candidate_ids.add(alias_text)

        candidate_ids = {item for item in candidate_ids if item}
        models = list(MLModelConfig.objects.filter(model_id__in=sorted(candidate_ids)))
        if not models:
            return Response({"error": f"Unknown model: {model_id}"}, status=404)

        with transaction.atomic():
            for model in models:
                model.delete()
            ModelStats.objects.filter(model_id__in=sorted(candidate_ids)).delete()
            DriftReport.objects.filter(model_id__in=sorted(candidate_ids)).delete()

        refresh_runtime_state("api-remove-model", force=True, warmup=False)
        return Response({"success": True})


class ModelsStatsListView(APIView):
    serializer_class = EmptySerializer

    @extend_schema(
        tags=["ML"],
        operation_id="ml_models_stats_list",
        summary="List Stats of the ML models",
        responses={
            status.HTTP_200_OK: ModelsStatsListResponseSerializer,
            status.HTTP_503_SERVICE_UNAVAILABLE: InitializingResponseSerializer,
        },
    )
    def get(self, request):
        return Response({"models": _collect_model_stats_entries()})


class ModelsStatsView(APIView):
    serializer_class = EmptySerializer

    @extend_schema(
        tags=["ML"],
        operation_id="ml_models_stats_list",
        summary="Get stats of  ML model",
        responses={
            status.HTTP_200_OK: ModelDetailResponseSerializer,
            status.HTTP_404_NOT_FOUND: ErrorResponseSerializer,
            status.HTTP_503_SERVICE_UNAVAILABLE: InitializingResponseSerializer,
        },
    )
    def get(self, request, model_id):
        registry_match = None
        if is_ready():
            refresh_runtime_state("model-stats-detail-read", force=False, warmup=False)
            registry_match = get_registry().get(model_id)

        target_model_id = (
            registry_match.model_id if registry_match is not None else model_id
        )
        feature_names = (
            getattr(registry_match, "feature_names", None)
            if registry_match is not None and getattr(registry_match, "feature_names", None)
            else _load_model_feature_names().get(target_model_id)
        )
        stats_row = ModelStats.objects.filter(model_id=target_model_id).first()
        if stats_row is not None:
            return Response(
                _serialize_model_stats_entry(
                    model_id=stats_row.model_id,
                    identifier=stats_row.identifier,
                    stats=stats_row.stats,
                    made_at=stats_row.made_at,
                    feature_names=feature_names,
                )
            )
        if registry_match is not None:
            return Response(
                _serialize_model_stats_entry(
                    model_id=registry_match.model_id,
                    identifier=registry_match.slug,
                    stats=_load_configured_metrics_payload(registry_match),
                    made_at="",
                    feature_names=feature_names,
                )
            )
        return Response(
            {"error": f"Unknown model: {model_id}"}, status=status.HTTP_404_NOT_FOUND
        )


class RemoveModelsStatsView(APIView):
    serializer_class = EmptySerializer

    @extend_schema(
        tags=["ML"],
        operation_id="remove_ml_models_stats_list",
        summary="Remove stats of a ML model",
        responses={
            status.HTTP_200_OK: RemoveModelStatsResponseSerializer,
            status.HTTP_404_NOT_FOUND: ErrorResponseSerializer,
            status.HTTP_503_SERVICE_UNAVAILABLE: InitializingResponseSerializer,
        },
    )
    def post(self, request, model_id):
        ModelStats.objects.filter(model_id=model_id).delete()
        return Response(
            {
                "status": "ok",
            }
        )


class ConfigReloadView(APIView):
    serializer_class = EmptySerializer

    @extend_schema(
        tags=["ML"],
        operation_id="ml_reload_config",
        summary="Reload ML model configuration",
        request=None,
        responses={
            status.HTTP_200_OK: ConfigReloadResponseSerializer,
            status.HTTP_503_SERVICE_UNAVAILABLE: InitializingResponseSerializer,
        },
    )
    def post(self, request):
        result = reload_model_config_control("manual-reload")
        return Response(result.payload, status=result.status_code)


class RuntimeConfigView(APIView):
    @extend_schema(
        tags=["ML"],
        operation_id="ml_runtime_config_status",
        summary="Inspect runtime model catalog status",
        responses={status.HTTP_200_OK: RuntimeConfigStatusSerializer},
    )
    def get(self, request):
        return Response(get_runtime_config_status())


class ControlStatusView(APIView):
    serializer_class = EmptySerializer

    @extend_schema(
        tags=["ML"],
        operation_id="ml_control_status",
        summary="Inspect ML control-plane status",
        responses={
            status.HTTP_200_OK: ControlStatusResponseSerializer,
            status.HTTP_503_SERVICE_UNAVAILABLE: ControlStatusResponseSerializer,
        },
    )
    def get(self, request):
        payload = get_control_status(refresh=True)
        response_status = (
            status.HTTP_200_OK
            if payload.get("status") == "ok"
            else status.HTTP_503_SERVICE_UNAVAILABLE
        )
        return Response(payload, status=response_status)


# ── External model upload ──────────────────────────────────────────────────

# Allowed file extensions → model_type families that accept them.
# Kept here as a local constant (not config) because it is a framework/format
# constraint, not a business rule.
_UPLOAD_ALLOWED_EXTENSIONS = {
    ".pkl": ("sklearn", "xgboost", "custom", "online_agent"),
    ".joblib": ("sklearn", "custom"),
    ".ubj": ("xgboost",),
    ".json": ("xgboost",),
    ".pt": ("pytorch",),
    ".pth": ("pytorch",),
}

_SUPPORTED_MODEL_TYPES = {choice[0] for choice in MLModelConfig.MODEL_TYPES}


def _validate_upload_config(config: dict) -> tuple[str, str, str, list[str], dict, list, dict]:
    """Parse and validate the JSON config block from a model upload request.

    Returns (model_id, file_path_hint, model_type, features, feature_info, examples, defaults).
    Raises ValueError with a descriptive message on any problem.
    """
    model_id = str(config.get("model_id") or config.get("id") or "").strip()
    if not model_id:
        raise ValueError("config.model_id is required.")
    if len(model_id) > 128:
        raise ValueError("config.model_id must be ≤ 128 characters.")

    model_type = str(config.get("model_type") or config.get("type") or "sklearn").strip()
    if model_type not in _SUPPORTED_MODEL_TYPES:
        raise ValueError(
            f"Unsupported model_type '{model_type}'. "
            f"Allowed: {sorted(_SUPPORTED_MODEL_TYPES)}"
        )

    features = config.get("features", [])
    if not isinstance(features, list) or not features:
        raise ValueError("config.features must be a non-empty list of feature name strings.")
    features = [str(f) for f in features if isinstance(f, str) and f.strip()]
    if not features:
        raise ValueError("config.features must contain at least one non-empty feature name.")

    feature_info = config.get("feature_info", {})
    if not isinstance(feature_info, dict):
        raise ValueError("config.feature_info must be a JSON object (dict).")

    examples = config.get("examples", [])
    if not isinstance(examples, list):
        raise ValueError("config.examples must be a JSON array.")
    examples = [item for item in examples if isinstance(item, dict)]

    defaults = config.get("defaults", {})
    if not isinstance(defaults, dict):
        raise ValueError("config.defaults must be a JSON object (dict).")

    description = str(config.get("description", "")).strip()

    return model_id, model_type, description, features, feature_info, examples, defaults


class ModelUploadView(APIView):
    """Accept a multipart POST (model_file + config JSON) and register the model.

    POST /api/ml/models/upload/
    Content-Type: multipart/form-data

    Fields:
      model_file  — the serialized model artifact (.pkl / .ubj / .pt / .pth / .joblib / .json)
      config      — JSON string with: model_id, model_type, features,
                    and optionally feature_info, examples, defaults, description
    """

    permission_classes = [AllowAny]

    @extend_schema(
        tags=["ML"],
        operation_id="ml_model_upload",
        summary="Upload an external model file and register it in the runtime catalog",
        request={"multipart/form-data": ModelUploadRequestSerializer},
        responses={
            status.HTTP_201_CREATED: ModelUploadResponseSerializer,
            400: ErrorResponseSerializer,
            409: ErrorResponseSerializer,
            422: ErrorResponseSerializer,
        },
    )
    def post(self, request):
        model_file = request.FILES.get("model_file")
        if model_file is None:
            return Response(
                {"error": "model_file is required (multipart/form-data)."},
                status=status.HTTP_400_BAD_REQUEST,
            )

        raw_config = request.data.get("config")
        if not raw_config:
            return Response(
                {"error": "config field is required (JSON string describing the model)."},
                status=status.HTTP_400_BAD_REQUEST,
            )

        try:
            config = json.loads(raw_config) if isinstance(raw_config, str) else dict(raw_config)
        except (json.JSONDecodeError, TypeError) as exc:
            return Response(
                {"error": f"config is not valid JSON: {exc}"},
                status=status.HTTP_400_BAD_REQUEST,
            )

        try:
            model_id, model_type, description, features, feature_info, examples, defaults = (
                _validate_upload_config(config)
            )
        except ValueError as exc:
            return Response({"error": str(exc)}, status=status.HTTP_422_UNPROCESSABLE_ENTITY)

        # Validate file extension vs declared model_type
        _, ext = os.path.splitext(model_file.name.lower())
        allowed_types = _UPLOAD_ALLOWED_EXTENSIONS.get(ext)
        if allowed_types is None:
            return Response(
                {
                    "error": (
                        f"File extension '{ext}' is not supported. "
                        f"Allowed: {sorted(_UPLOAD_ALLOWED_EXTENSIONS)}"
                    )
                },
                status=status.HTTP_422_UNPROCESSABLE_ENTITY,
            )
        if model_type not in allowed_types:
            return Response(
                {
                    "error": (
                        f"Extension '{ext}' is not compatible with model_type '{model_type}'. "
                        f"Compatible types for '{ext}': {list(allowed_types)}"
                    )
                },
                status=status.HTTP_422_UNPROCESSABLE_ENTITY,
            )

        # Check for duplicate model_id (reject if already active)
        if MLModelConfig.objects.filter(model_id=model_id, is_active=True).exists():
            return Response(
                {
                    "error": (
                        f"A model with model_id='{model_id}' is already registered and active. "
                        "Deactivate it first via the remove endpoint, or use a different model_id."
                    )
                },
                status=status.HTTP_409_CONFLICT,
            )

        # Save the file to PIOS_MODELS_DIR/uploads/<model_id><ext>
        models_dir = getattr(settings, "PIOS_MODELS_DIR", None)
        if models_dir is None:
            models_dir = os.path.join(settings.BASE_DIR, "ml", "models")
        uploads_dir = os.path.join(str(models_dir), "uploads")
        os.makedirs(uploads_dir, exist_ok=True)

        safe_model_id = "".join(c if c.isalnum() or c in "-_" else "_" for c in model_id)
        dest_filename = f"{safe_model_id}{ext}"
        dest_path = os.path.join(uploads_dir, dest_filename)

        try:
            with open(dest_path, "wb") as fh:
                for chunk in model_file.chunks():
                    fh.write(chunk)
        except OSError as exc:
            logger.exception("Failed to save uploaded model file: %s", exc)
            return Response(
                {"error": f"Could not save model file: {exc}"},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR,
            )

        # Relative path stored in DB (registry resolves against PIOS_MODELS_DIR)
        relative_file_path = os.path.join("uploads", dest_filename)

        with transaction.atomic():
            MLModelConfig.objects.update_or_create(
                model_id=model_id,
                defaults={
                    "description": description,
                    "file_path": relative_file_path,
                    "model_type": model_type,
                    "features": features,
                    "feature_info": feature_info,
                    "examples": examples,
                    "defaults": defaults,
                    "is_active": True,
                },
            )

        # Hot-reload the registry so the model is immediately available
        try:
            refresh_runtime_state("model-upload", force=True, warmup=False)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Registry reload after upload raised: %s", exc)

        logger.info(
            "External model uploaded: model_id=%s type=%s file=%s features=%s",
            model_id,
            model_type,
            relative_file_path,
            features,
        )

        return Response(
            {
                "status": "registered",
                "model_id": model_id,
                "model_type": model_type,
                "file_path": relative_file_path,
                "features": features,
                "is_active": True,
                "message": (
                    f"Model '{model_id}' saved and registered. "
                    "It is now available for orchestrator tool-calling and dashboard predictions."
                ),
            },
            status=status.HTTP_201_CREATED,
        )


# ══════════════════════════════════════════════════════════
#  PREDICTION
# ══════════════════════════════════════════════════════════


def _json_stream_default(value):
    if hasattr(value, "isoformat"):
        try:
            return value.isoformat()
        except Exception:
            pass
    return str(value)


def _encode_sse_event(payload):
    body = json.dumps(payload, ensure_ascii=True, default=_json_stream_default)
    return f"data: {body}\n\n"


def _request_accepts_event_stream(request):
    return "text/event-stream" in str(request.headers.get("Accept", "")).lower()


def _first_model_used(result):
    exec_results = result.get("execution_results", []) if isinstance(result, dict) else []
    if exec_results:
        return str(exec_results[0].get("tool", "")).strip()
    return ""


def _record_prediction_log(
    *,
    user,
    model_used,
    query,
    features,
    result,
    elapsed_ms,
):
    try:
        PredictionLog.objects.create(
            user=user,
            model_id_used=model_used,
            query=query,
            features_input=features,
            prediction_output=result,
            planner_mode=result.get("planner_mode", "") if isinstance(result, dict) else "",
            success=result.get("success", False) if isinstance(result, dict) else False,
            response_time_ms=round(elapsed_ms, 2),
        )
    except Exception:
        pass


def _evaluate_prediction_alerts(*, model_used, forced_model_ids, result, features):
    if not isinstance(result, dict) or not result.get("success"):
        return
    try:
        AlertEngine().evaluate_prediction(
            model_id=model_used or (forced_model_ids[0] if forced_model_ids else ""),
            prediction_result=result,
            feature_input=features,
            source="predict-nl",
        )
    except Exception as exc:
        logger.error(
            "Alert evaluation failed in PredictNLView for model '%s': %s",
            model_used or (forced_model_ids[0] if forced_model_ids else ""),
            exc,
        )


def _stream_nl_prediction_events(
    *,
    user,
    orchestrator,
    query,
    features,
    forced_model_ids,
    top_k,
):
    events: queue.Queue[dict[str, object] | object] = queue.Queue()
    done = object()
    stream_started = time.perf_counter()
    latest_phase = "planning"

    def publish(payload):
        events.put(payload)

    def worker():
        start = time.perf_counter()
        try:
            result = orchestrator.run(
                query=query,
                provided_features=features,
                forced_model_ids=forced_model_ids,
                top_k=top_k,
                event_callback=publish,
            )
            elapsed_ms = (time.perf_counter() - start) * 1000
            model_used = _first_model_used(result)
            _record_prediction_log(
                user=user,
                model_used=model_used,
                query=query,
                features=features,
                result=result,
                elapsed_ms=elapsed_ms,
            )
            _evaluate_prediction_alerts(
                model_used=model_used,
                forced_model_ids=forced_model_ids,
                result=result,
                features=features,
            )
            public_result = _public_prediction_response(result)
            publish(
                {
                    "type": "final",
                    "phase": "completed",
                    "success": bool(result.get("success")),
                    "markdown": str(result.get("natural_language_response") or ""),
                    "tool_summaries": result.get("tool_summaries", []),
                    "result": public_result,
                }
            )
        except Exception as exc:
            publish(
                {
                    "type": "error",
                    "phase": "error",
                    "success": False,
                    "error": f"Orchestration error: {exc}",
                }
            )
        finally:
            events.put(done)

    threading.Thread(target=worker, daemon=True).start()
    yield _encode_sse_event({"type": "progress", "phase": "received"})

    while True:
        try:
            item = events.get(timeout=1.5)
        except queue.Empty:
            elapsed_seconds = round(time.perf_counter() - stream_started, 1)
            yield _encode_sse_event(
                {
                    "type": "progress",
                    "phase": latest_phase,
                    "heartbeat": True,
                    "elapsed_seconds": elapsed_seconds,
                }
            )
            continue
        if item is done:
            break
        if isinstance(item, dict):
            latest_phase = str(item.get("phase") or latest_phase)
        yield _encode_sse_event(item)


class PredictNLView(APIView):
    serializer_class = NLPredictRequestSerializer
    renderer_classes = [*api_settings.DEFAULT_RENDERER_CLASSES, EventStreamRenderer]

    @extend_schema(
        tags=["ML"],
        operation_id="ml_predict_natural_language",
        summary="Run a natural-language ML prediction",
        request=NLPredictRequestSerializer,
        responses={
            status.HTTP_200_OK: PredictionResponseSerializer,
            status.HTTP_400_BAD_REQUEST: ErrorResponseSerializer,
            status.HTTP_503_SERVICE_UNAVAILABLE: PredictionResponseSerializer,
        },
        examples=[
            OpenApiExample(
                "Natural language prediction",
                value={
                    "query": "Can a 35 year old donor with BMI 24 donate blood now?",
                },
                request_only=True,
            ),
        ],
    )
    def post(self, request):
        if not is_ready():
            return Response(
                {"error": "ML models still loading.", "tools_used": []},
                status=status.HTTP_503_SERVICE_UNAVAILABLE,
            )

        serializer = NLPredictRequestSerializer(data=request.data)
        serializer.is_valid(raise_exception=True)
        data = serializer.validated_data

        refresh_runtime_state("predict-nl", force=False, warmup=False)
        registry = get_registry()
        orchestrator = get_orchestrator()

        query = data["query"]
        features = sanitize_features(data.get("features", {}))
        model_id = data.get("model_id")
        top_k = data.get("top_k")

        forced_model_ids = None
        if model_id and not is_placeholder_text(model_id):
            rt = registry.get(model_id)
            if rt and rt.status == "loaded":
                forced_model_ids = [rt.model_id]
        if forced_model_ids is None:
            forced_model_ids = _infer_forced_model_ids(query, registry)

        if _request_accepts_event_stream(request):
            stream = _stream_nl_prediction_events(
                user=request.user if request.user.is_authenticated else None,
                orchestrator=orchestrator,
                query=query,
                features=features,
                forced_model_ids=forced_model_ids,
                top_k=top_k,
            )
            response = StreamingHttpResponse(
                stream,
                content_type="text/event-stream",
            )
            response["Cache-Control"] = "no-cache"
            response["X-Accel-Buffering"] = "no"
            return response

        start = time.perf_counter()
        try:
            result = orchestrator.run(
                query=query,
                provided_features=features,
                forced_model_ids=forced_model_ids,
                top_k=top_k,
            )
        except Exception as exc:
            return Response(
                {"error": f"Orchestration error: {exc}", "tools_used": []},
                status=status.HTTP_400_BAD_REQUEST,
            )
        elapsed_ms = (time.perf_counter() - start) * 1000

        model_used = _first_model_used(result)
        _record_prediction_log(
            user=request.user if request.user.is_authenticated else None,
            model_used=model_used,
            query=query,
            features=features,
            result=result,
            elapsed_ms=elapsed_ms,
        )

        public_result = _public_prediction_response(result)

        if not result.get("success"):
            return Response(public_result, status=status.HTTP_503_SERVICE_UNAVAILABLE)

        _evaluate_prediction_alerts(
            model_used=model_used,
            forced_model_ids=forced_model_ids,
            result=result,
            features=features,
        )
        return Response(public_result)


class PredictModelView(APIView):
    serializer_class = PredictRequestSerializer

    @extend_schema(
        tags=["ML"],
        operation_id="ml_predict_model",
        summary="Run a prediction against one specific model",
        request=PredictRequestSerializer,
        responses={
            status.HTTP_200_OK: PredictionResponseSerializer,
            status.HTTP_400_BAD_REQUEST: ErrorResponseSerializer,
            status.HTTP_404_NOT_FOUND: ErrorResponseSerializer,
            status.HTTP_503_SERVICE_UNAVAILABLE: PredictionResponseSerializer,
        },
        examples=[
            OpenApiExample(
                "Direct model prediction",
                value={
                    "query": "Check donor eligibility",
                    "features": {
                        "age": 42,
                        "sex": "F",
                        "bmi": 26.1,
                        "donation_count_last_12m": 5,
                    },
                },
                request_only=True,
            ),
        ],
    )
    def post(self, request, model_id):
        serializer = PredictRequestSerializer(data=request.data)
        serializer.is_valid(raise_exception=True)
        result = run_direct_model_prediction(
            model_id,
            serializer.validated_data,
            user=request.user,
            alert_source="predict-model",
        )
        return Response(result.payload, status=result.status_code)


class HybridStockoutPredictView(APIView):
    serializer_class = PredictRequestSerializer

    @extend_schema(
        tags=["ML"],
        operation_id="ml_predict_stockout_hybrid",
        summary="Run hybrid stockout prediction with days estimate and horizon probabilities",
        request=PredictRequestSerializer,
        responses={
            status.HTTP_200_OK: PredictionResponseSerializer,
            status.HTTP_400_BAD_REQUEST: ErrorResponseSerializer,
            status.HTTP_404_NOT_FOUND: ErrorResponseSerializer,
            status.HTTP_503_SERVICE_UNAVAILABLE: ErrorResponseSerializer,
        },
    )
    def post(self, request):
        if not is_ready():
            return Response(
                {"error": "ML models still loading.", "tools_used": []},
                status=status.HTTP_503_SERVICE_UNAVAILABLE,
            )

        serializer = PredictRequestSerializer(data=request.data)
        serializer.is_valid(raise_exception=True)
        data = serializer.validated_data

        refresh_runtime_state("predict-stockout-hybrid", force=False, warmup=False)
        registry = get_registry()

        requested_features = sanitize_features(data.get("features", {}))
        query = data.get("query") or "Run hybrid stockout prediction."
        preferred_model_id = data.get("model_id")
        if preferred_model_id and is_placeholder_text(preferred_model_id):
            preferred_model_id = None

        rt = find_hybrid_stockout_regression_runtime(
            registry,
            preferred_model_id=preferred_model_id,
        )
        if rt is None:
            return Response(
                {
                    "error": "No configured hybrid stockout regression model is loaded.",
                    "tools_used": [],
                },
                status=status.HTTP_404_NOT_FOUND,
            )
        if rt.status != "loaded":
            return Response(
                {
                    "error": rt.load_error or "Model is not loaded.",
                    "tools_used": [rt.model_id],
                },
                status=status.HTTP_503_SERVICE_UNAVAILABLE,
            )

        start = time.perf_counter()
        try:
            result = build_hybrid_stockout_prediction(
                registry=registry,
                regression_runtime=rt,
                request_features=requested_features,
                query=query,
            )
            if not result:
                raise ValueError(
                    f"{rt.model_id} is not configured for hybrid stockout."
                )
        except Exception as exc:
            return Response(
                {
                    "error": f"Prediction error: {exc}",
                    "tools_used": [getattr(rt, "model_id", "")],
                },
                status=status.HTTP_400_BAD_REQUEST,
            )
        elapsed_ms = (time.perf_counter() - start) * 1000

        try:
            PredictionLog.objects.create(
                user=request.user if request.user.is_authenticated else None,
                model_id_used=rt.model_id,
                query=query,
                features_input=requested_features,
                prediction_output=result,
                planner_mode="hybrid_stockout",
                success=True,
                response_time_ms=round(elapsed_ms, 2),
            )
        except Exception:
            pass

        try:
            AlertEngine().evaluate_prediction(
                model_id=rt.model_id,
                prediction_result=result,
                feature_input=requested_features,
                source="predict-stockout-hybrid",
            )
        except Exception:
            pass
        return Response(
            {
                "results": _compact_prediction_output(result),
                "tools_used": [rt.model_id],
            }
        )


class ToolListView(APIView):
    serializer_class = EmptySerializer

    @extend_schema(
        tags=["ML"],
        operation_id="ml_list_configured_tools",
        summary="List configured executable tools",
        responses={
            status.HTTP_200_OK: ToolListResponseSerializer,
            status.HTTP_503_SERVICE_UNAVAILABLE: InitializingResponseSerializer,
        },
    )
    def get(self, request):
        if not is_ready():
            return Response({"status": "initializing", "tools": []}, status=503)
        refresh_runtime_state("tool-list-read", force=False, warmup=False)
        return Response({"tools": get_orchestrator().list_configured_tools()})


class ToolRunView(APIView):
    serializer_class = ToolRunRequestSerializer

    @extend_schema(
        tags=["ML"],
        operation_id="ml_run_configured_tool",
        summary="Run one configured tool directly",
        request=ToolRunRequestSerializer,
        responses={
            status.HTTP_200_OK: PredictionResponseSerializer,
            status.HTTP_400_BAD_REQUEST: ErrorResponseSerializer,
            status.HTTP_404_NOT_FOUND: ErrorResponseSerializer,
            status.HTTP_503_SERVICE_UNAVAILABLE: InitializingResponseSerializer,
        },
    )
    def post(self, request, tool_id):
        if not is_ready():
            return Response(
                {"error": "ML models still loading.", "tools_used": []},
                status=status.HTTP_503_SERVICE_UNAVAILABLE,
            )

        serializer = ToolRunRequestSerializer(data=request.data)
        serializer.is_valid(raise_exception=True)
        data = serializer.validated_data

        refresh_runtime_state("tool-run", force=False, warmup=False)
        orchestrator = get_orchestrator()
        resolved_tool_id = orchestrator.configured_tool_id(tool_id)
        if not resolved_tool_id:
            return Response(
                {"error": f"Unknown configured tool: {tool_id}", "tools_used": []},
                status=status.HTTP_404_NOT_FOUND,
            )

        query = str(data.get("query") or "").strip()
        arguments = dict(data.get("arguments") or {})
        if not query:
            query = str(arguments.pop("query", "") or "").strip()

        start = time.perf_counter()
        try:
            output = orchestrator.execute_configured_tool(
                tool_id,
                query=query,
                arguments=arguments,
            )
        except Exception as exc:
            return Response(
                {"error": f"Tool execution error: {exc}", "tools_used": [resolved_tool_id]},
                status=status.HTTP_400_BAD_REQUEST,
            )

        result = {
            "success": True,
            "query": query,
            "planner_mode": "direct_tool",
            "execution_results": [
                {
                    "tool": resolved_tool_id,
                    "inputs": {"query": query, **arguments},
                    "output": output,
                    "success": True,
                }
            ],
            "natural_language_response": orchestrator._result_value_for_summary(output),
        }
        elapsed_ms = (time.perf_counter() - start) * 1000
        try:
            PredictionLog.objects.create(
                user=request.user if request.user.is_authenticated else None,
                model_id_used=resolved_tool_id,
                query=query,
                features_input=arguments,
                prediction_output=result,
                planner_mode="direct_tool",
                success=True,
                response_time_ms=round(elapsed_ms, 2),
            )
        except Exception:
            pass
        return Response(_public_prediction_response(result, tools_used=[resolved_tool_id]))


# ══════════════════════════════════════════════════════════
#  ORCHESTRATOR
# ══════════════════════════════════════════════════════════


class OrchestratorStatusView(APIView):
    serializer_class = EmptySerializer

    @extend_schema(
        tags=["ML"],
        operation_id="ml_orchestrator_status",
        summary="Get orchestrator status",
        responses={
            status.HTTP_200_OK: OrchestratorStatusResponseSerializer,
            status.HTTP_503_SERVICE_UNAVAILABLE: InitializingResponseSerializer,
        },
    )
    def get(self, request):
        if not is_ready():
            return Response({"status": "initializing"}, status=503)
        refresh_runtime_state("orchestrator-status-read", force=False, warmup=False)
        payload = get_orchestrator().status()
        payload["orchestrator_config"] = get_orchestrator_config_status()
        return Response(payload)


class OrchestratorModelsView(APIView):
    serializer_class = OrchestratorCatalogRequestSerializer

    @extend_schema(
        tags=["ML"],
        operation_id="ml_orchestrator_models",
        summary="List available orchestrator model variants",
        responses={
            status.HTTP_200_OK: OrchestratorModelListResponseSerializer,
            status.HTTP_503_SERVICE_UNAVAILABLE: InitializingResponseSerializer,
        },
    )
    def get(self, request):
        if not is_ready():
            return Response({"status": "initializing"}, status=503)
        sync_orchestrator_catalog(
            "orchestrator-models-read", force=False, warmup=False, wait=False
        )
        orch = get_orchestrator()
        return Response(
            {
                "models": orch.list_available_models(),
                "selected_model_id": orch.preferred_model_id,
                "active_model_id": orch.status().get("active_model_id"),
                "config": get_orchestrator_config_status(),
            }
        )

    @extend_schema(
        tags=["ML"],
        operation_id="ml_orchestrator_models_update",
        summary="Replace orchestrator model catalog and hot apply it",
        request=OrchestratorCatalogRequestSerializer,
        responses={
            status.HTTP_200_OK: OrchestratorSelectionResponseSerializer,
            422: ErrorResponseSerializer,
        },
    )
    def put(self, request):
        serializer = OrchestratorCatalogRequestSerializer(data=request.data)
        serializer.is_valid(raise_exception=True)
        result = replace_orchestrator_catalog(
            models=serializer.validated_data.get("models", []),
            selected_model_id=serializer.validated_data.get("selected_model_id"),
        )
        return Response(result.payload, status=result.status_code)


class OrchestratorSelectView(APIView):
    serializer_class = OrchestratorSelectRequestSerializer

    @extend_schema(
        tags=["ML"],
        operation_id="ml_orchestrator_select_model",
        summary="Select the active orchestrator model variant",
        request=OrchestratorSelectRequestSerializer,
        responses={
            status.HTTP_200_OK: OrchestratorSelectionResponseSerializer,
            status.HTTP_404_NOT_FOUND: ErrorResponseSerializer,
            422: ErrorResponseSerializer,
        },
        examples=[
            OpenApiExample(
                "Select variant",
                value={"model_id": "xlam-7b-q4_k_m"},
                request_only=True,
            ),
        ],
    )
    def put(self, request):
        serializer = OrchestratorSelectRequestSerializer(data=request.data)
        serializer.is_valid(raise_exception=True)
        result = select_orchestrator_model(
            serializer.validated_data["model_id"],
            download=serializer.validated_data.get("download", True),
            wait=serializer.validated_data.get("wait", False),
        )
        return Response(result.payload, status=result.status_code)


class OrchestratorDownloadView(APIView):
    serializer_class = OrchestratorDownloadRequestSerializer

    @extend_schema(
        tags=["ML"],
        operation_id="ml_orchestrator_download_model",
        summary="Select and warm up an orchestrator model variant",
        request=OrchestratorDownloadRequestSerializer,
        responses={
            status.HTTP_200_OK: OrchestratorSelectionResponseSerializer,
            status.HTTP_404_NOT_FOUND: ErrorResponseSerializer,
            422: ErrorResponseSerializer,
        },
    )
    def post(self, request, model_id):
        serializer = OrchestratorDownloadRequestSerializer(data=request.data)
        serializer.is_valid(raise_exception=True)
        result = select_orchestrator_model(
            model_id,
            download=True,
            wait=serializer.validated_data.get("wait", False),
            reason="api-orchestrator-download-select",
            response_status="download_started",
        )
        return Response(result.payload, status=result.status_code)


class OrchestratorWarmupView(APIView):
    serializer_class = OrchestratorWarmupRequestSerializer

    @extend_schema(
        tags=["ML"],
        operation_id="ml_orchestrator_warmup",
        summary="Warm up the orchestrator LLM",
        request=OrchestratorWarmupRequestSerializer,
        responses={
            status.HTTP_200_OK: OrchestratorStatusResponseSerializer,
            status.HTTP_503_SERVICE_UNAVAILABLE: InitializingResponseSerializer,
        },
        examples=[
            OpenApiExample(
                "Blocking warmup",
                value={"wait": True},
                request_only=True,
            ),
        ],
    )
    def post(self, request):
        result = warmup_orchestrator(wait=request.data.get("wait", False))
        return Response(result.payload, status=result.status_code)


# ═══════════════════════════════════════════════════════════════════════════
#  Training Scheduler
# ═══════════════════════════════════════════════════════════════════════════


class SchedulerStatusView(APIView):
    serializer_class = EmptySerializer

    @extend_schema(
        tags=["Scheduler"],
        operation_id="scheduler_status",
        summary="Get the current state of every scheduled training job",
        responses={status.HTTP_200_OK: SchedulerStatusResponseSerializer},
    )
    def get(self, request):
        return Response({"jobs": get_schedule_status()})


class SchedulerReloadView(APIView):
    serializer_class = EmptySerializer

    @extend_schema(
        tags=["Scheduler"],
        operation_id="scheduler_reload",
        summary="Hot-reload training_schedule.json and reschedule all jobs",
        responses={status.HTTP_200_OK: SchedulerReloadResponseSerializer},
    )
    def post(self, request):
        config = reload_schedule_config()
        return Response(
            {
                "message": "Schedule reloaded",
                "scripts": list(config.get("scripts", {}).keys()),
                "jobs": get_schedule_status(),
            }
        )


class SchedulerRunView(APIView):
    serializer_class = SchedulerRunRequestSerializer

    @extend_schema(
        tags=["Scheduler"],
        operation_id="scheduler_run_now",
        summary="Start a training run in the background (non-blocking)",
        description=(
            "Dispatches training to a background worker and returns immediately "
            "with a run record to poll via the run-status endpoint. Training "
            "scripts can take minutes to hours, so this never blocks the "
            "request. At most one manual run executes at a time: if a run is "
            "already in flight, that run is returned with ``started=false`` "
            "instead of launching an overlapping batch."
        ),
        request=SchedulerRunRequestSerializer,
        responses={
            status.HTTP_202_ACCEPTED: SchedulerRunStartResponseSerializer,
            status.HTTP_200_OK: SchedulerRunStartResponseSerializer,
        },
        examples=[
            OpenApiExample(
                "Run all enabled models",
                value={},
                request_only=True,
            ),
            OpenApiExample(
                "Run a single model",
                value={"model_key": "blood_shortage_predictor"},
                request_only=True,
            ),
        ],
    )
    def post(self, request):
        model_key = request.data.get("model_key") or None
        run, started = start_run(model_key=model_key)
        return Response(
            {
                "run": run,
                "started": started,
                "message": (
                    "Training run started."
                    if started
                    else "A training run is already in progress."
                ),
            },
            status=(
                status.HTTP_202_ACCEPTED if started else status.HTTP_200_OK
            ),
        )


class SchedulerRunStatusView(APIView):
    """Poll the state of the latest (or a specific) manual training run."""

    serializer_class = EmptySerializer

    @extend_schema(
        tags=["Scheduler"],
        operation_id="scheduler_run_status",
        summary="Get the status of the latest or a specific manual run",
        responses={status.HTTP_200_OK: SchedulerRunStatusResponseSerializer},
    )
    def get(self, request):
        run_id = request.query_params.get("run_id") or None
        return Response({"run": get_run(run_id)})


class SchedulerConfigView(APIView):
    """Read or update the training_schedule.json config file."""

    serializer_class = SchedulerConfigUpdateSerializer

    @extend_schema(
        tags=["Scheduler"],
        operation_id="scheduler_config_get",
        summary="Read the current training schedule configuration",
        responses={status.HTTP_200_OK: SchedulerConfigResponseSerializer},
    )
    def get(self, request):
        config = load_schedule_config()
        config.pop("_comment", None)
        return Response(config)

    @extend_schema(
        tags=["Scheduler"],
        operation_id="scheduler_config_update",
        summary="Update the training schedule configuration and reload",
        request=SchedulerConfigUpdateSerializer,
        responses={status.HTTP_200_OK: SchedulerReloadResponseSerializer},
    )
    def put(self, request):
        config_path = get_schedule_config_path()

        # Load existing config, merge with incoming data
        current = load_schedule_config()
        current.pop("_comment", None)

        incoming = request.data
        if "defaults" in incoming:
            current["defaults"] = incoming["defaults"]
        if "scripts" in incoming:
            current["scripts"].update(incoming["scripts"])
        if "post_training" in incoming:
            current["post_training"] = incoming["post_training"]

        # Write back
        config_path.parent.mkdir(parents=True, exist_ok=True)
        with open(config_path, "w", encoding="utf-8") as fh:
            json.dump(current, fh, indent=4)

        # Reload the scheduler with the new config
        reload_schedule_config()

        return Response(
            {
                "message": "Schedule updated and reloaded",
                "scripts": list(current.get("scripts", {}).keys()),
                "jobs": get_schedule_status(),
            }
        )


# ═══════════════════════════════════════════════════════════════════════════
#  Drift Monitoring
# ═══════════════════════════════════════════════════════════════════════════


class DriftStatusView(APIView):
    """Latest drift status for each model (one row per model)."""

    serializer_class = EmptySerializer

    @extend_schema(
        tags=["Drift Monitoring"],
        operation_id="drift_status",
        summary="Get the latest drift detection status per model",
        responses={status.HTTP_200_OK: DriftStatusResponseSerializer},
    )
    def get(self, request):
        # Get the latest report for each model_id
        from django.db.models import Max

        latest_ids = (
            DriftReport.objects.values("model_id")
            .annotate(latest_id=Max("id"))
            .values_list("latest_id", flat=True)
        )
        reports = DriftReport.objects.filter(id__in=latest_ids).order_by("model_id")
        data = []
        for r in reports:
            data.append(
                {
                    "id": r.id,
                    "model_id": r.model_id,
                    "mlflow_run_id": r.mlflow_run_id,
                    "drift_detected": r.drift_detected,
                    "drift_score": r.drift_score,
                    "severity": r.severity,
                    "features_checked": r.features_checked,
                    "features_drifted": r.features_drifted,
                    "reference_size": r.reference_size,
                    "current_size": r.current_size,
                    "check_type": r.check_type,
                    "checked_at": r.checked_at,
                }
            )
        return Response({"models": data})


class DriftReportsView(APIView):
    """List drift reports, optionally filtered by model_id."""

    serializer_class = EmptySerializer

    @extend_schema(
        tags=["Drift Monitoring"],
        operation_id="drift_reports_list",
        summary="List drift reports (optionally filtered by model_id)",
        responses={status.HTTP_200_OK: DriftReportsListResponseSerializer},
    )
    def get(self, request, model_id=None):
        qs = DriftReport.objects.all()
        if model_id:
            qs = qs.filter(model_id=model_id)

        # Optional query params
        limit = int(request.query_params.get("limit", 50))
        offset = int(request.query_params.get("offset", 0))
        drift_only = request.query_params.get("drift_only", "").lower() == "true"

        if drift_only:
            qs = qs.filter(drift_detected=True)

        total = qs.count()
        reports = qs[offset : offset + limit]

        data = []
        for r in reports:
            data.append(
                {
                    "id": r.id,
                    "model_id": r.model_id,
                    "mlflow_run_id": r.mlflow_run_id,
                    "drift_detected": r.drift_detected,
                    "drift_score": r.drift_score,
                    "severity": r.severity,
                    "features_checked": r.features_checked,
                    "features_drifted": r.features_drifted,
                    "reference_size": r.reference_size,
                    "current_size": r.current_size,
                    "check_type": r.check_type,
                    "checked_at": r.checked_at,
                    "feature_details": r.feature_details,
                }
            )
        return Response({"reports": data, "count": total})


class DriftRunView(APIView):
    """Manually trigger a drift check."""

    serializer_class = DriftRunRequestSerializer

    @extend_schema(
        tags=["Drift Monitoring"],
        operation_id="drift_run",
        summary="Manually trigger drift detection for one or all models",
        request=DriftRunRequestSerializer,
        responses={status.HTTP_200_OK: DriftRunResponseSerializer},
    )
    def post(self, request):
        model_name = request.data.get("model_name") or None
        try:
            from ml.services.training_scheduler import run_drift_check

            results = run_drift_check(model_name=model_name)
            return Response({"results": results})
        except ImportError:
            return Response(
                {
                    "error": "Drift monitoring not available — training scheduler module missing"
                },
                status=status.HTTP_503_SERVICE_UNAVAILABLE,
            )
        except Exception as exc:
            return Response(
                {"error": f"Drift check failed: {exc}"},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR,
            )
