from __future__ import annotations

import math
import json
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable

from ml.core.model_metadata import runtime_prediction_defaults


SHORT_MAX_DAYS = 7.0
MEDIUM_MAX_DAYS = 29.0
DEFAULT_FORECAST_HORIZON_ROUTING_CONFIG_PATH = (
    Path(__file__).resolve().parents[1] / "config" / "forecast_horizon_routing.json"
)
_FORECAST_HORIZON_ROUTING_CONFIG_CACHE: tuple[Path, float, dict[str, Any]] | None = None

TIMEFRAME_ALIASES = {
    "short": "short",
    "short_term": "short",
    "shortterm": "short",
    "near": "short",
    "near_term": "short",
    "immediate": "short",
    "medium": "medium",
    "meduim": "medium",
    "mid": "medium",
    "mid_term": "medium",
    "medium_term": "medium",
    "midterm": "medium",
    "long": "long",
    "long_term": "long",
    "longterm": "long",
    "strategic": "long",
}

HORIZON_TIMEFRAME_KEYS = {
    "prediction_timeframe",
    "forecast_timeframe",
    "timeframe",
    "time_frame",
    "horizon_timeframe",
    "horizon_bucket",
    "forecast_bucket",
}

HORIZON_DAY_KEYS = {
    "prediction_horizon_days",
    "forecast_horizon_days",
    "horizon_days",
    "days_ahead",
    "prediction_days",
    "forecast_days",
}

HORIZON_META_KEYS = HORIZON_TIMEFRAME_KEYS | HORIZON_DAY_KEYS | {
    "horizon",
    "prediction_horizon",
    "forecast_horizon",
}

FORECAST_HORIZON_ROUTING_TERMS = {
    "stock",
    "stockout",
    "shortage",
    "hospital",
    "inventory",
    "demand",
    "supply",
    "units",
    "wastage",
    "trauma",
    "surgery",
    "surgeries",
    "blood bank",
    "blood supply",
}

MAXIMIZE_METRICS = {
    "accuracy",
    "balanced_accuracy",
    "precision",
    "recall",
    "f1",
    "f1_score",
    "roc_auc",
    "average_precision",
    "r2",
    "r2_score",
}

MINIMIZE_METRICS = {
    "loss",
    "log_loss",
    "brier_score",
    "mae",
    "mean_absolute_error",
    "mape",
    "mean_absolute_percentage_error",
    "wmape",
    "mse",
    "mean_squared_error",
    "rmse",
    "root_mean_squared_error",
}

DEFAULT_FALLBACK_METRICS = [
    "balanced_accuracy",
    "f1_score",
    "roc_auc",
    "average_precision",
    "r2_score",
    "root_mean_squared_error",
    "mean_absolute_error",
]

SPECIFICITY_PRIORITY = {
    "exact_horizon": 3,
    "timeframe": 2,
    "configured_horizon": 1,
    "overall": 0,
}


def _forecast_horizon_routing_config_path() -> Path:
    configured = os.getenv("PIOS_FORECAST_HORIZON_ROUTING_CONFIG", "").strip()
    if configured:
        return Path(configured).expanduser().resolve()
    return DEFAULT_FORECAST_HORIZON_ROUTING_CONFIG_PATH


def _forecast_horizon_routing_config() -> dict[str, Any]:
    global _FORECAST_HORIZON_ROUTING_CONFIG_CACHE
    path = _forecast_horizon_routing_config_path()
    try:
        mtime = path.stat().st_mtime
    except OSError:
        return {}
    if (
        _FORECAST_HORIZON_ROUTING_CONFIG_CACHE is not None
        and _FORECAST_HORIZON_ROUTING_CONFIG_CACHE[0] == path
        and _FORECAST_HORIZON_ROUTING_CONFIG_CACHE[1] == mtime
    ):
        return _FORECAST_HORIZON_ROUTING_CONFIG_CACHE[2]
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    if not isinstance(payload, dict) or payload.get("enabled", True) is False:
        return {}
    _FORECAST_HORIZON_ROUTING_CONFIG_CACHE = (path, mtime, payload)
    return payload


def _configured_string_set(value: Any) -> set[str]:
    if not isinstance(value, Iterable) or isinstance(value, (str, bytes, bytearray, dict)):
        return set()
    return {str(item).strip().lower() for item in value if str(item).strip()}


def _configured_target_rows() -> list[dict[str, Any]]:
    rows = _forecast_horizon_routing_config().get("targets")
    if not isinstance(rows, list):
        return []
    return [row for row in rows if isinstance(row, dict)]


def forecast_selection_target_for_query(
    query: str,
    features: dict[str, Any] | None = None,
    *,
    default_target: str = "stockout",
) -> str:
    lowered = (query or "").lower()
    feature_keys = {str(key).lower() for key in dict(features or {})}
    for row in _configured_target_rows():
        target = str(row.get("target") or "").strip().lower()
        if not target:
            continue
        terms = _configured_string_set(row.get("terms"))
        configured_keys = _configured_string_set(row.get("feature_keys"))
        if (terms and any(term in lowered for term in terms)) or (
            configured_keys and feature_keys.intersection(configured_keys)
        ):
            return target
    configured_default = str(
        _forecast_horizon_routing_config().get("default_target") or ""
    ).strip().lower()
    return configured_default or str(default_target or "stockout").strip().lower()


@dataclass(frozen=True)
class ForecastHorizon:
    timeframe: str
    days: float | None = None
    source: str = "query"

    def as_dict(self) -> dict[str, Any]:
        return {
            "timeframe": self.timeframe,
            "days": self.days,
            "source": self.source,
        }


@dataclass(frozen=True)
class HorizonModelScore:
    runtime: Any = field(repr=False)
    metric_name: str
    metric_value: float | None
    direction: str
    metric_score: float
    rank_score: float
    specificity: str
    source: str
    horizon_key: str | None = None
    target: str = "stockout"
    fallback_reason: str = ""

    @property
    def model_id(self) -> str:
        return str(getattr(self.runtime, "model_id", "") or "")

    def as_dict(self) -> dict[str, Any]:
        return {
            "model_id": self.model_id,
            "metric_name": self.metric_name,
            "metric_value": self.metric_value,
            "direction": self.direction,
            "rank_score": self.rank_score,
            "specificity": self.specificity,
            "source": self.source,
            "horizon_key": self.horizon_key,
            "target": self.target,
            "fallback_reason": self.fallback_reason,
        }


def normalize_timeframe(value: Any) -> str | None:
    text = str(value or "").strip().lower()
    if not text:
        return None
    key = re.sub(r"[^a-z0-9]+", "_", text).strip("_")
    return TIMEFRAME_ALIASES.get(key)


def timeframe_for_days(days: float) -> str:
    if days <= SHORT_MAX_DAYS:
        return "short"
    if days <= MEDIUM_MAX_DAYS:
        return "medium"
    return "long"


def _coerce_positive_float(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        number = float(value)
        return number if math.isfinite(number) and number >= 0 else None
    text = str(value or "").strip()
    if not text:
        return None
    try:
        number = float(text)
    except ValueError:
        return None
    return number if math.isfinite(number) and number >= 0 else None


def _coerce_float(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        number = float(value)
        return number if math.isfinite(number) else None
    text = str(value or "").strip()
    if not text:
        return None
    try:
        number = float(text)
    except ValueError:
        return None
    return number if math.isfinite(number) else None


def _unit_to_days(value: float, unit: str) -> float:
    normalized = unit.strip().lower()
    if normalized in {"h", "hr", "hrs", "hour", "hours"}:
        return value / 24.0
    if normalized in {"d", "day", "days"}:
        return value
    if normalized in {"w", "week", "weeks"}:
        return value * 7.0
    if normalized in {"mo", "month", "months"}:
        return value * 30.0
    if normalized in {"q", "quarter", "quarters"}:
        return value * 90.0
    if normalized in {"y", "yr", "year", "years"}:
        return value * 365.0
    return value


def _parse_days_from_text(text: str) -> float | None:
    lowered = (text or "").lower()
    j_plus = re.search(r"\b(?:j|d|t)\s*\+\s*(\d+(?:\.\d+)?)\b", lowered)
    if j_plus:
        return float(j_plus.group(1))

    number_unit = re.search(
        r"\b(?:forecast(?:ing)?|prediction|predict|horizon|lead(?:\s*time)?|next|in|over|for)"
        r"(?:\s+the)?(?:\s+next)?\s+(\d+(?:\.\d+)?)\s*"
        r"(hours?|hrs?|hr|h|days?|d|weeks?|w|months?|mo|quarters?|q|years?|yrs?|y)\b",
        lowered,
    )
    if number_unit:
        return _unit_to_days(float(number_unit.group(1)), number_unit.group(2))

    trailing_unit = re.search(
        r"\b(\d+(?:\.\d+)?)\s*"
        r"(hours?|hrs?|hr|h|days?|d|weeks?|w|months?|mo|quarters?|q|years?|yrs?|y)"
        r"\s+(?:ahead|forecast|prediction|horizon|outlook|risk)\b",
        lowered,
    )
    if trailing_unit:
        return _unit_to_days(float(trailing_unit.group(1)), trailing_unit.group(2))

    relative_days = {
        "today": 0.0,
        "tomorrow": 1.0,
        "next day": 1.0,
        "next week": 7.0,
        "next month": 30.0,
        "next quarter": 90.0,
        "next year": 365.0,
    }
    for phrase, days in relative_days.items():
        if phrase in lowered:
            return days
    return None


def _parse_timeframe_from_text(text: str) -> str | None:
    lowered = (text or "").lower()
    if re.search(r"\b(?:short|near)[\s-]?term\b|\bimmediate\b", lowered):
        return "short"
    if re.search(r"\b(?:medium|meduim|mid)[\s-]?term\b", lowered):
        return "medium"
    if re.search(r"\blong[\s-]?term\b|\bstrategic\b", lowered):
        return "long"
    return None


def extract_forecast_horizon(
    query: str,
    features: dict[str, Any] | None = None,
) -> ForecastHorizon | None:
    request_features = dict(features or {})

    for key in HORIZON_DAY_KEYS:
        if key in request_features:
            days = _coerce_positive_float(request_features.get(key))
            if days is not None:
                return ForecastHorizon(timeframe_for_days(days), days, f"feature:{key}")

    for key in HORIZON_TIMEFRAME_KEYS:
        if key in request_features:
            timeframe = normalize_timeframe(request_features.get(key))
            if timeframe:
                return ForecastHorizon(timeframe, None, f"feature:{key}")

    for key in ("prediction_horizon", "forecast_horizon", "horizon"):
        if key in request_features:
            raw = request_features.get(key)
            days = _coerce_positive_float(raw)
            if days is not None:
                return ForecastHorizon(timeframe_for_days(days), days, f"feature:{key}")
            parsed_days = _parse_days_from_text(str(raw))
            if parsed_days is not None:
                return ForecastHorizon(
                    timeframe_for_days(parsed_days), parsed_days, f"feature:{key}"
                )
            timeframe = normalize_timeframe(raw)
            if timeframe:
                return ForecastHorizon(timeframe, None, f"feature:{key}")

    days = _parse_days_from_text(query)
    if days is not None:
        return ForecastHorizon(timeframe_for_days(days), days, "query")

    timeframe = _parse_timeframe_from_text(query)
    if timeframe:
        return ForecastHorizon(timeframe, None, "query")

    return None


def is_forecast_horizon_routing_query(
    query: str,
    features: dict[str, Any] | None = None,
) -> bool:
    lowered = (query or "").lower()
    config = _forecast_horizon_routing_config()
    configured_terms = _configured_string_set(config.get("routing_terms"))
    target_terms = set()
    for row in _configured_target_rows():
        target_terms.update(_configured_string_set(row.get("terms")))
    terms = FORECAST_HORIZON_ROUTING_TERMS | configured_terms | target_terms
    if any(term in lowered for term in terms):
        return True
    request_features = dict(features or {})
    feature_keys = {str(key).lower() for key in request_features}
    configured_feature_keys = _configured_string_set(config.get("feature_keys"))
    target_feature_keys = set()
    for row in _configured_target_rows():
        target_feature_keys.update(_configured_string_set(row.get("feature_keys")))
    default_feature_keys = {
        "hospital_id",
        "supply_id",
        "current_inventory",
        "current_stock_units",
        "stock_end",
        "units_used",
        "units_collected",
        "scheduled_surgeries",
        "trauma_cases",
    }
    return bool(feature_keys.intersection(default_feature_keys | configured_feature_keys | target_feature_keys))


def strip_horizon_routing_features(features: dict[str, Any]) -> dict[str, Any]:
    return {
        key: value
        for key, value in dict(features or {}).items()
        if str(key).lower() not in HORIZON_META_KEYS
    }


def _as_list(value: Any) -> list[str]:
    if isinstance(value, str):
        return [value]
    if isinstance(value, Iterable) and not isinstance(value, (dict, bytes, bytearray)):
        return [str(item) for item in value if str(item).strip()]
    return []


def runtime_horizon_routing(runtime: Any) -> dict[str, Any]:
    defaults = _runtime_defaults(runtime)
    routing = defaults.get("horizon_routing") or defaults.get("forecast_horizon")
    return dict(routing) if isinstance(routing, dict) else {}


def runtime_forecast_selection(runtime: Any) -> dict[str, Any]:
    defaults = _runtime_defaults(runtime)
    selection = (
        defaults.get("forecast_selection")
        or defaults.get("horizon_selection")
        or defaults.get("model_selection")
    )
    return dict(selection) if isinstance(selection, dict) else {}


def _runtime_defaults(runtime: Any) -> dict[str, Any]:
    model_id = str(getattr(runtime, "model_id", "") or "")
    merged = _deep_merge({}, runtime_prediction_defaults(model_id))
    runtime_defaults = getattr(runtime, "defaults", {}) or {}
    if isinstance(runtime_defaults, dict):
        merged = _deep_merge(merged, runtime_defaults)
    return merged


def _deep_merge(base: dict[str, Any], extra: dict[str, Any]) -> dict[str, Any]:
    result = dict(base)
    for key, value in extra.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def _routing_timeframes(routing: dict[str, Any]) -> set[str]:
    values = _as_list(routing.get("timeframes") or routing.get("timeframe"))
    return {
        normalized
        for value in values
        if (normalized := normalize_timeframe(value)) is not None
    }


def runtime_supports_horizon(runtime: Any, horizon: ForecastHorizon) -> bool:
    routing = runtime_horizon_routing(runtime)
    if not routing or routing.get("enabled", True) is False:
        return False

    timeframes = _routing_timeframes(routing)
    if timeframes and horizon.timeframe not in timeframes:
        return False

    if horizon.days is None:
        return bool(timeframes)

    min_days = _coerce_positive_float(routing.get("min_days"))
    max_days = _coerce_positive_float(routing.get("max_days"))
    if min_days is not None and horizon.days < min_days:
        return False
    if max_days is not None and horizon.days > max_days:
        return False
    return bool(timeframes or min_days is not None or max_days is not None)


def horizon_routing_score(runtime: Any, horizon: ForecastHorizon) -> float | None:
    if not runtime_supports_horizon(runtime, horizon):
        return None
    routing = runtime_horizon_routing(runtime)
    score = _coerce_positive_float(routing.get("priority")) or 0.0
    if horizon.timeframe in _routing_timeframes(routing):
        score += 100.0
    if horizon.days is not None:
        target_days = _coerce_positive_float(routing.get("target_days"))
        if target_days is not None:
            score += max(0.0, 50.0 - abs(horizon.days - target_days))
        else:
            min_days = _coerce_positive_float(routing.get("min_days")) or 0.0
            max_days = _coerce_positive_float(routing.get("max_days"))
            if max_days is None:
                max_days = max(min_days + 30.0, horizon.days)
            center = (min_days + max_days) / 2.0
            width = max(1.0, max_days - min_days)
            score += max(0.0, 25.0 - (abs(horizon.days - center) / width) * 25.0)
    return score


def describe_runtime_horizon(runtime: Any) -> str:
    routing = runtime_horizon_routing(runtime)
    if not routing or routing.get("enabled", True) is False:
        return ""
    label = str(routing.get("label") or "").strip()
    if label:
        return label

    timeframes = sorted(_routing_timeframes(routing))
    min_days = routing.get("min_days")
    max_days = routing.get("max_days")
    parts = []
    if timeframes:
        parts.append("/".join(timeframes))
    if min_days is not None or max_days is not None:
        lower = "0" if min_days is None else str(min_days)
        upper = "+" if max_days is None else str(max_days)
        parts.append(f"{lower}-{upper} days")
    return ", ".join(parts)


def _clean_metric_name(metric_name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(metric_name or "").strip().lower()).strip("_")


def metric_direction(metric_name: str, configured_direction: str | None = None) -> str:
    direction = str(configured_direction or "").strip().lower()
    if direction in {"maximize", "max", "higher", "higher_is_better"}:
        return "maximize"
    if direction in {"minimize", "min", "lower", "lower_is_better"}:
        return "minimize"

    clean = _clean_metric_name(metric_name)
    if clean in MINIMIZE_METRICS:
        return "minimize"
    return "maximize"


def _metric_to_score(metric_name: str, value: float, direction: str) -> float:
    if direction == "minimize":
        return -value
    return value


def _metric_names_for_runtime(
    runtime: Any,
    *,
    primary_metric: str = "accuracy",
) -> list[str]:
    selection = runtime_forecast_selection(runtime)
    if selection:
        primary = str(selection.get("primary_metric") or primary_metric or "accuracy")
        raw_fallbacks = selection.get("fallback_metrics")
        fallbacks = _as_list(raw_fallbacks) if raw_fallbacks is not None else []
        names = [primary, *fallbacks, *DEFAULT_FALLBACK_METRICS]
    else:
        names = [*DEFAULT_FALLBACK_METRICS, primary_metric or "accuracy"]
    cleaned: list[str] = []
    seen: set[str] = set()
    for name in names:
        clean = _clean_metric_name(name)
        if clean and clean not in seen:
            seen.add(clean)
            cleaned.append(clean)
    return cleaned


def _runtime_selection_target(runtime: Any) -> str:
    selection = runtime_forecast_selection(runtime)
    target = str(selection.get("target") or "").strip().lower()
    if target:
        return target
    routing = runtime_horizon_routing(runtime)
    target = str(routing.get("target") or "").strip().lower()
    return target


def runtime_is_forecast_candidate(
    runtime: Any,
    *,
    target: str = "stockout",
    query: str = "",
) -> bool:
    selection = runtime_forecast_selection(runtime)
    if selection.get("enabled", True) is False:
        return False

    desired_target = str(target or "stockout").strip().lower()
    runtime_target = _runtime_selection_target(runtime)
    if runtime_target and runtime_target != desired_target:
        return False

    if selection:
        return True

    searchable = " ".join(
        [
            str(getattr(runtime, "model_id", "") or ""),
            str(getattr(runtime, "description", "") or ""),
            " ".join(str(alias) for alias in getattr(runtime, "aliases", []) or []),
        ]
    ).lower()
    return any(
        token in searchable
        for token in (
            "stockout",
            "shortage",
            "inventory",
            "hospital",
            "demand",
            "supply",
        )
    )


def _payload_source(payload: dict[str, Any], default: str) -> str:
    return str(payload.get("__metric_source") or default).strip() or default


def _as_metric_map(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        return {}
    return {
        _clean_metric_name(key): item
        for key, item in value.items()
        if _clean_metric_name(key)
    }


def _horizon_lookup_keys(horizon: ForecastHorizon) -> set[str]:
    keys = {horizon.timeframe}
    if horizon.days is not None:
        days = horizon.days
        rendered = str(int(days)) if float(days).is_integer() else str(days)
        keys.update(
            {
                rendered,
                f"{rendered}d",
                f"d{rendered}",
                f"j{rendered}",
                f"j+{rendered}",
                f"horizon_{rendered}",
                f"day_{rendered}",
            }
        )
    return {_clean_metric_name(key) for key in keys if _clean_metric_name(key)}


def _iter_horizon_metric_maps(
    payload: dict[str, Any],
    horizon: ForecastHorizon,
    *,
    default_source: str,
) -> list[tuple[str, str, str | None, dict[str, Any]]]:
    source = _payload_source(payload, default_source)
    rows: list[tuple[str, str, str | None, dict[str, Any]]] = []
    lookup_keys = _horizon_lookup_keys(horizon)

    def visit(container: Any, prefix: str = "") -> None:
        if not isinstance(container, dict):
            return
        for key in (
            "horizon_metrics",
            "metrics_by_horizon",
            "by_horizon",
            "per_horizon",
            "horizons",
        ):
            nested = container.get(key)
            if not isinstance(nested, dict):
                continue
            for raw_horizon_key, metrics in nested.items():
                clean_key = _clean_metric_name(raw_horizon_key)
                metric_map = _as_metric_map(metrics)
                if not metric_map:
                    continue
                specificity = (
                    "timeframe"
                    if clean_key == horizon.timeframe
                    else "exact_horizon"
                    if clean_key in lookup_keys
                    else ""
                )
                if specificity:
                    rows.append(
                        (
                            specificity,
                            source,
                            str(raw_horizon_key),
                            metric_map,
                        )
                    )
        for key in ("validation_metrics", "metrics", "selection_metrics"):
            nested = container.get(key)
            if isinstance(nested, dict) and nested is not container:
                visit(nested, f"{prefix}.{key}" if prefix else key)

    visit(payload)
    return rows


def _iter_overall_metric_maps(
    payload: dict[str, Any],
    *,
    default_source: str,
) -> list[tuple[str, str, str | None, dict[str, Any]]]:
    source = _payload_source(payload, default_source)
    rows: list[tuple[str, str, str | None, dict[str, Any]]] = []
    for key in (
        "overall_metrics",
        "validation_metrics",
        "hard_metrics",
        "metrics",
        "test_metrics",
    ):
        metric_map = _as_metric_map(payload.get(key))
        if metric_map:
            rows.append(("overall", source, key, metric_map))

    payload_metric_map = _as_metric_map(payload)
    if payload_metric_map:
        rows.append(("overall", source, None, payload_metric_map))
    return rows


def _metric_value(metric_map: dict[str, Any], metric_name: str) -> float | None:
    value = metric_map.get(_clean_metric_name(metric_name))
    return _coerce_float(value)


def _best_score_from_metric_maps(
    runtime: Any,
    rows: list[tuple[str, str, str | None, dict[str, Any]]],
    *,
    primary_metric: str,
    target: str,
) -> HorizonModelScore | None:
    metric_names = _metric_names_for_runtime(runtime, primary_metric=primary_metric)
    selection = runtime_forecast_selection(runtime)
    configured_primary_metric = _clean_metric_name(
        selection.get("primary_metric") or primary_metric
    )
    configured_direction = selection.get("direction")
    configured_metric_directions = selection.get("metric_directions")
    if not isinstance(configured_metric_directions, dict):
        configured_metric_directions = {}
    best: HorizonModelScore | None = None

    for specificity, source, horizon_key, metric_map in rows:
        specificity_score = SPECIFICITY_PRIORITY.get(specificity, 0)
        for metric_index, metric_name in enumerate(metric_names):
            value = _metric_value(metric_map, metric_name)
            if value is None:
                continue
            clean_metric_name = _clean_metric_name(metric_name)
            direction_override = configured_metric_directions.get(clean_metric_name)
            if direction_override is None and clean_metric_name == configured_primary_metric:
                direction_override = configured_direction
            direction = metric_direction(metric_name, str(direction_override or ""))
            metric_score = _metric_to_score(metric_name, value, direction)
            metric_priority_score = (len(metric_names) - metric_index) * 1_000_000.0
            rank_score = (
                specificity_score * 1_000_000_000_000.0
                + metric_priority_score
                + metric_score
                - (metric_index * 0.0001)
                + (_coerce_positive_float(selection.get("priority")) or 0.0) * 0.000001
            )
            candidate = HorizonModelScore(
                runtime=runtime,
                metric_name=metric_name,
                metric_value=value,
                direction=direction,
                metric_score=metric_score,
                rank_score=rank_score,
                specificity=specificity,
                source=source,
                horizon_key=horizon_key,
                target=target,
            )
            if best is None or candidate.rank_score > best.rank_score:
                best = candidate
    return best


def _fallback_score_from_config(
    runtime: Any,
    horizon: ForecastHorizon,
    *,
    target: str,
) -> HorizonModelScore | None:
    routing_score = horizon_routing_score(runtime, horizon)
    if routing_score is None:
        return None
    return HorizonModelScore(
        runtime=runtime,
        metric_name="configured_horizon_priority",
        metric_value=None,
        direction="maximize",
        metric_score=routing_score,
        rank_score=SPECIFICITY_PRIORITY["configured_horizon"] * 1_000_000.0
        + routing_score,
        specificity="configured_horizon",
        source="model_defaults",
        horizon_key=horizon.timeframe,
        target=target,
        fallback_reason="no horizon-specific validation metrics were available",
    )


def rank_model_scores_for_horizon(
    query: str,
    models: list[Any],
    horizon: ForecastHorizon,
    *,
    features: dict[str, Any] | None = None,
    target: str = "stockout",
    primary_metric: str = "accuracy",
    metric_payloads_fn: Callable[[Any], list[dict[str, Any]]] | None = None,
    scope_score_fn: Callable[[Any], float] | None = None,
) -> list[HorizonModelScore]:
    if not is_forecast_horizon_routing_query(query, features):
        return []

    metric_scores: list[HorizonModelScore] = []
    fallback_scores: list[HorizonModelScore] = []
    for runtime in models:
        if not runtime_is_forecast_candidate(runtime, target=target, query=query):
            continue
        payloads = metric_payloads_fn(runtime) if metric_payloads_fn is not None else []
        metric_rows: list[tuple[str, str, str | None, dict[str, Any]]] = []
        for index, payload in enumerate(payloads or []):
            if not isinstance(payload, dict):
                continue
            source = _payload_source(payload, f"payload:{index}")
            metric_rows.extend(
                _iter_horizon_metric_maps(payload, horizon, default_source=source)
            )
            metric_rows.extend(
                _iter_overall_metric_maps(payload, default_source=source)
            )
        routing = runtime_horizon_routing(runtime)
        if routing:
            metric_rows = [
                row
                for row in metric_rows
                if row[0] != "overall" or runtime_supports_horizon(runtime, horizon)
            ]
        score = _best_score_from_metric_maps(
            runtime,
            metric_rows,
            primary_metric=primary_metric,
            target=target,
        )
        if score is not None:
            if scope_score_fn is not None:
                score = HorizonModelScore(
                    runtime=score.runtime,
                    metric_name=score.metric_name,
                    metric_value=score.metric_value,
                    direction=score.direction,
                    metric_score=score.metric_score,
                    rank_score=score.rank_score
                    + max(0.0, float(scope_score_fn(runtime))) * 0.000001,
                    specificity=score.specificity,
                    source=score.source,
                    horizon_key=score.horizon_key,
                    target=score.target,
                    fallback_reason=score.fallback_reason,
                )
            metric_scores.append(score)
            continue

        fallback_score = _fallback_score_from_config(runtime, horizon, target=target)
        if fallback_score is not None:
            fallback_scores.append(fallback_score)

    rows = metric_scores or fallback_scores
    rows.sort(key=lambda item: (item.rank_score, item.model_id), reverse=True)
    return rows


def rank_models_for_horizon(
    query: str,
    models: list[Any],
    horizon: ForecastHorizon,
    *,
    features: dict[str, Any] | None = None,
    target: str = "stockout",
    primary_metric: str = "accuracy",
    metric_payloads_fn: Callable[[Any], list[dict[str, Any]]] | None = None,
    scope_score_fn: Callable[[Any], float] | None = None,
) -> list[Any]:
    scores = rank_model_scores_for_horizon(
        query,
        models,
        horizon,
        features=features,
        target=target,
        primary_metric=primary_metric,
        metric_payloads_fn=metric_payloads_fn,
        scope_score_fn=scope_score_fn,
    )
    return [score.runtime for score in scores]
