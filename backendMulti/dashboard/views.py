# from channels.layers import get_channel_layer
# from asgiref.sync import async_to_sync
# from django.http import JsonResponse

# def send_dashboard_update(request):

#     channel_layer = get_channel_layer()

#     async_to_sync(channel_layer.group_send)(
#         "dashboard",
#         {
#             "type": "dashboard_message",
#             "message": "Dashboard updated"
#         }
#     )

#     return JsonResponse({"status": "sent"})

import json
from functools import lru_cache
from pathlib import Path
from datetime import datetime
from typing import Any

from django.conf import settings
from django.http import JsonResponse
from django.utils import timezone
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_GET

from ml.models import ModelStats

from .models import ModelResponse


@lru_cache(maxsize=1)
def _scheduled_prediction_config() -> dict[str, Any]:
    path = Path(settings.PIOS_SCHEDULED_PREDICTION_CONFIG_PATH)
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise RuntimeError(f"Scheduled prediction config not found: {path}") from exc
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"Invalid scheduled prediction config JSON: {path}") from exc
    if not isinstance(payload, dict):
        raise RuntimeError("Scheduled prediction config root must be an object.")
    return payload


def _dashboard_snapshot_config(snapshot_type: str) -> dict[str, Any]:
    snapshots = _scheduled_prediction_config().get("dashboard_snapshots", {})
    if not isinstance(snapshots, dict):
        return {}
    config = snapshots.get(snapshot_type, {})
    return config if isinstance(config, dict) else {}


def _configured_field_aliases(snapshot_type: str) -> dict[str, tuple[str, ...]]:
    aliases = _dashboard_snapshot_config(snapshot_type).get("field_aliases", {})
    if not isinstance(aliases, dict):
        return {}
    return {
        str(key): tuple(str(item) for item in value if str(item).strip())
        for key, value in aliases.items()
        if isinstance(value, list)
    }


def _configured_binary_features(snapshot_type: str) -> set[str]:
    values = _dashboard_snapshot_config(snapshot_type).get("binary_features", [])
    if not isinstance(values, list):
        return set()
    return {str(value) for value in values if str(value).strip()}


def _to_number(value: Any) -> float | None:
    try:
        parsed = float(value)
        if parsed != parsed:  # NaN guard
            return None
        return parsed
    except (TypeError, ValueError):
        return None


def _format_number(value: float) -> str:
    if value.is_integer():
        return str(int(value))
    return f"{value:.2f}".rstrip("0").rstrip(".")


def _format_binary_display(value: Any) -> str:
    numeric = _to_number(value)
    if numeric is None:
        text = str(value).strip().lower()
        if text in {"0", "false", "no", "n"}:
            return "No"
        if text in {"1", "true", "yes", "y"}:
            return "Yes"
        return str(value)
    if numeric <= 0:
        return "No"
    if numeric >= 1:
        return "Yes"
    return _format_number(numeric)


def _extract_feature_display(
    canonical_key: str,
    raw: dict[str, Any],
    binary_features: set[str],
) -> str | None:
    if canonical_key in binary_features:
        for binary_field in ("mode", "default", "median", "mean"):
            candidate = raw.get(binary_field)
            if candidate in (None, ""):
                continue
            return _format_binary_display(candidate)

        lower = _to_number(raw.get("lower"))
        upper = _to_number(raw.get("upper"))
        if lower is not None and upper is not None:
            if lower == upper:
                return _format_binary_display(lower)
            midpoint = (lower + upper) / 2
            return "Yes" if midpoint > 0.5 else "No"

        return None

    lower = _to_number(raw.get("lower"))
    upper = _to_number(raw.get("upper"))
    if lower is not None and upper is not None:
        return f"{_format_number(lower)}-{_format_number(upper)}"

    mode = raw.get("mode")
    if mode not in (None, ""):
        return str(mode)

    for fallback_field in ("default", "median", "mean"):
        fallback_value = raw.get(fallback_field)
        if fallback_value in (None, ""):
            continue
        num = _to_number(fallback_value)
        return _format_number(num) if num is not None else str(fallback_value)

    return None


def _normalize_ideal_donor_stats(stats: Any) -> dict[str, str]:
    if not isinstance(stats, dict):
        return {}

    profile: dict[str, str] = {}
    field_aliases = _configured_field_aliases("ideal")
    binary_features = _configured_binary_features("ideal")
    for canonical_key, aliases in field_aliases.items():
        raw = next(
            (
                stats.get(alias)
                for alias in aliases
                if isinstance(stats.get(alias), dict)
            ),
            None,
        )
        if raw is None:
            continue
        display = _extract_feature_display(canonical_key, raw, binary_features)
        if display is not None:
            profile[canonical_key] = display

    return profile


def _build_ideal_snapshot_from_ml_stats(
    selected_filters: dict[str, str],
) -> dict[str, Any] | None:
    def _find_row_and_profile(queryset) -> tuple[ModelStats, dict[str, str]] | None:
        for row in queryset:
            profile = _normalize_ideal_donor_stats(row.stats)
            if profile:
                return row, profile
        return None

    row_with_profile = None
    snapshot_config = _dashboard_snapshot_config("ideal")
    candidates = snapshot_config.get("stats_candidates", [])
    if not isinstance(candidates, list):
        candidates = []
    for candidate in candidates:
        if not isinstance(candidate, dict) or not candidate:
            continue
        result = _find_row_and_profile(
            ModelStats.objects.filter(**candidate).order_by("-made_at")[:5]
        )
        if result is not None:
            row_with_profile = result
            break

    fallback_filter = snapshot_config.get("fallback_filter")
    if row_with_profile is None:
        if isinstance(fallback_filter, dict) and fallback_filter:
            row_with_profile = _find_row_and_profile(
                ModelStats.objects.filter(**fallback_filter).order_by("-made_at")[:10]
            )

    if row_with_profile is None:
        return None

    model_stats, profile = row_with_profile

    start_date = model_stats.made_at
    if not isinstance(start_date, datetime):
        start_date = timezone.now()

    return {
        "id": -1,
        "typeModel": "ideal",
        "jsonResponse": profile,
        "region": selected_filters.get("region", ModelResponse.Region.QUEBEC),
        "product": selected_filters.get("product", ModelResponse.Product.BLOOD),
        "period": selected_filters.get("period", ModelResponse.Period.H24),
        "alert_level": selected_filters.get("alert_level", ModelResponse.AlertLevel.LOW),
        "startDate": start_date,
        "endDate": None,
    }

@csrf_exempt
def add_dashboard_row(request):
    if request.method != "POST":
        return JsonResponse({"error": "POST request required"}, status=400)

    try:
        data = json.loads(request.body)
        type_model = data.get("typeModel")
        json_response = data.get("jsonResponse")

        if not type_model or not json_response:
            return JsonResponse({"error": "typeModel and jsonResponse required"}, status=400)

        # Create the new database row
        row = ModelResponse.objects.create(
            typeModel=type_model,
            jsonResponse=json_response
        )

        return JsonResponse({
            "status": "success",
            "row": {
                "id": row.id,
                "typeModel": row.typeModel,
                "jsonResponse": row.jsonResponse,
                "startDate": row.startDate.isoformat(),
                "endDate": None
            }
        })

    except Exception as e:
        return JsonResponse({"error": str(e)}, status=500)

@csrf_exempt
def add_dashboard_row_web(request):
    if request.method != "POST":
        return JsonResponse({"error": "POST request required"}, status=400)

    try:
        data = json.loads(request.body)
        type_model = data.get("typeModel")
        json_response = data.get("jsonResponse")
        region = data.get("region", ModelResponse.Region.QUEBEC)
        product = data.get("product", ModelResponse.Product.BLOOD)
        period = data.get("period", ModelResponse.Period.H24)
        alert_level = data.get("alert_level", ModelResponse.AlertLevel.LOW)

        if not type_model or not json_response:
            return JsonResponse({"error": "typeModel and jsonResponse are required"}, status=400)

        row = ModelResponse.objects.create(
            typeModel=type_model,
            jsonResponse=json_response,
            region=region,
            product=product,
            period=period,
            alert_level=alert_level,
        )

        return JsonResponse({
            "status": "success",
            "row": {
                "id": row.id,
                "typeModel": row.typeModel,
                "jsonResponse": row.jsonResponse,
                "region": row.region,
                "product": row.product,
                "period": row.period,
                "alert_level": row.alert_level,
                "startDate": row.startDate.isoformat(),
                "endDate": None,
            }
        })

    except Exception as e:
        return JsonResponse({"error": str(e)}, status=500)

@require_GET
def latest_dashboard_data(request):
    expected_types = ("stock", "metrics", "donations", "ideal")

    selected_filters = {}
    for field in ("region", "product", "period", "alert_level"):
        value = request.GET.get(field)
        if value:
            selected_filters[field] = value

    base_queryset = ModelResponse.objects.order_by("-startDate", "-id")
    filtered_queryset = (
        base_queryset.filter(**selected_filters)
        if selected_filters
        else base_queryset
    )

    value_fields = (
        "id",
        "typeModel",
        "jsonResponse",
        "region",
        "product",
        "period",
        "alert_level",
        "startDate",
        "endDate",
    )

    snapshots = {}
    for type_model in expected_types:
        latest = (
            filtered_queryset.filter(typeModel=type_model)
            .values(*value_fields)
            .first()
        )

        # Fallback to global latest row for the type so refresh always restores all cards.
        if latest is None and selected_filters:
            latest = (
                base_queryset.filter(typeModel=type_model)
                .values(*value_fields)
                .first()
            )

        snapshots[type_model] = latest

    ideal_from_ml_stats = _build_ideal_snapshot_from_ml_stats(selected_filters)
    if ideal_from_ml_stats is not None:
        snapshots["ideal"] = ideal_from_ml_stats

    ordered_rows = [
        row
        for row in sorted(
            (row for row in snapshots.values() if row is not None),
            key=lambda row: (row["startDate"], row["id"]),
            reverse=True,
        )
    ]

    return JsonResponse({"snapshots": snapshots, "rows": ordered_rows})
