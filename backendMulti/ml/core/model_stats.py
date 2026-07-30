from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from typing import Any


def _json_safe(value: Any) -> Any:
    if value is None:
        return None
    if hasattr(value, "item") and callable(getattr(value, "item")):
        try:
            return _json_safe(value.item())
        except Exception:
            pass
    if hasattr(value, "tolist") and callable(getattr(value, "tolist")):
        try:
            converted = value.tolist()
            if converted is not value:
                return _json_safe(converted)
        except Exception:
            pass
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [_json_safe(item) for item in value]
    if isinstance(value, bool):
        return value
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    return value


def _parse_payload(payload: Any) -> Any:
    if payload is None:
        return {}
    if isinstance(payload, str):
        text = payload.strip()
        if not text:
            return {}
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            return payload
    return payload


def _feature_list(feature_names: Sequence[str] | None) -> list[str]:
    if not feature_names:
        return []
    return [str(name) for name in feature_names if isinstance(name, str) and name.strip()]


def _recover_legacy_mapping(
    payload: Any, *, feature_names: Sequence[str] | None = None
) -> dict[str, Any] | None:
    values = _json_safe(payload)
    if not isinstance(values, list):
        return None

    features = _feature_list(feature_names)
    if not features:
        return None

    if len(values) == len(features):
        keys = features
    elif len(values) == len(features) + 2:
        keys = ["label", *features, "prediction"]
    else:
        return None

    return {key: value for key, value in zip(keys, values)}


def prepare_model_stats_for_storage(payload: Any) -> Any:
    """Convert training-produced stats into JSON-safe data while preserving keys."""
    return _json_safe(payload)


def normalize_model_stats_payload(
    payload: Any, *, feature_names: Sequence[str] | None = None
) -> Any:
    """Return JSON-compatible stats, recovering legacy anonymous arrays when possible."""
    normalized = _json_safe(_parse_payload(payload))
    if isinstance(normalized, list):
        recovered = _recover_legacy_mapping(
            normalized, feature_names=feature_names
        )
        if recovered is not None:
            return recovered
    if isinstance(normalized, (dict, list, int, float, bool, str)):
        return normalized
    return {}


def _as_numeric(value: Any) -> float | None:
    if isinstance(value, bool):
        return 1.0 if value else 0.0
    if isinstance(value, (int, float)):
        numeric = float(value)
        return numeric if math.isfinite(numeric) else None
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return None
        try:
            numeric = float(text)
        except ValueError:
            return None
        return numeric if math.isfinite(numeric) else None
    return None


def extract_scalar_stat_value(value: Any) -> float | None:
    numeric = _as_numeric(value)
    if numeric is not None:
        return numeric

    normalized = _json_safe(value)
    if isinstance(normalized, Mapping):
        for key in ("value", "median", "mean", "avg", "average", "p50"):
            if key in normalized:
                return extract_scalar_stat_value(normalized[key])
        lower = extract_scalar_stat_value(normalized.get("lower"))
        upper = extract_scalar_stat_value(normalized.get("upper"))
        if lower is not None and upper is not None:
            return (lower + upper) / 2.0
        minimum = extract_scalar_stat_value(normalized.get("min"))
        maximum = extract_scalar_stat_value(normalized.get("max"))
        if minimum is not None and maximum is not None:
            return (minimum + maximum) / 2.0
        return None

    if isinstance(normalized, list):
        if len(normalized) == 1:
            return extract_scalar_stat_value(normalized[0])
        if len(normalized) == 2:
            left = extract_scalar_stat_value(normalized[0])
            right = extract_scalar_stat_value(normalized[1])
            if left is not None and right is not None:
                return (left + right) / 2.0
        return None

    return None


def extract_default_stat_value(value: Any) -> Any:
    if isinstance(value, bool):
        return value
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, str):
        text = value.strip()
        return text if text else None

    normalized = _json_safe(value)
    if isinstance(normalized, Mapping):
        for key in (
            "default",
            "fill_value",
            "mode",
            "most_frequent",
            "value",
            "median",
            "mean",
            "avg",
            "average",
            "p50",
        ):
            if key in normalized:
                return extract_default_stat_value(normalized[key])
        lower = extract_scalar_stat_value(normalized.get("lower"))
        upper = extract_scalar_stat_value(normalized.get("upper"))
        if lower is not None and upper is not None:
            return (lower + upper) / 2.0
        minimum = extract_scalar_stat_value(normalized.get("min"))
        maximum = extract_scalar_stat_value(normalized.get("max"))
        if minimum is not None and maximum is not None:
            return (minimum + maximum) / 2.0
        return None

    if isinstance(normalized, list):
        if len(normalized) == 1:
            return extract_default_stat_value(normalized[0])
        if len(normalized) == 2:
            left = extract_scalar_stat_value(normalized[0])
            right = extract_scalar_stat_value(normalized[1])
            if left is not None and right is not None:
                return (left + right) / 2.0
        return None

    return None


def coerce_model_stats_mapping(
    payload: Any, *, feature_names: Sequence[str] | None = None
) -> dict[str, float]:
    """Extract scalar per-feature defaults from arbitrary training stats payloads."""
    normalized = normalize_model_stats_payload(payload, feature_names=feature_names)
    if not isinstance(normalized, dict):
        return {}

    allowed_features = set(_feature_list(feature_names)) if feature_names else None
    result: dict[str, float] = {}
    for key, value in normalized.items():
        key_text = str(key)
        if allowed_features is not None and key_text not in allowed_features:
            continue
        scalar = extract_scalar_stat_value(value)
        if scalar is not None:
            result[key_text] = scalar
    return result


def coerce_model_default_mapping(
    payload: Any, *, feature_names: Sequence[str] | None = None
) -> dict[str, Any]:
    """Extract per-feature default values from arbitrary training stats payloads."""
    normalized = normalize_model_stats_payload(payload, feature_names=feature_names)
    if not isinstance(normalized, dict):
        return {}

    allowed_features = set(_feature_list(feature_names)) if feature_names else None
    result: dict[str, Any] = {}
    for key, value in normalized.items():
        key_text = str(key)
        if allowed_features is not None and key_text not in allowed_features:
            continue
        default_value = extract_default_stat_value(value)
        if default_value is not None:
            result[key_text] = default_value
    return result
