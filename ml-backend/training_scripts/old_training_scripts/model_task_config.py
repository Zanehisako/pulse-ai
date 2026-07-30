from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


DEFAULT_CONFIG_DIR = Path(
    os.getenv(
        "PIOS_ML_TRAINING_CONFIG_DIR",
        str(Path(__file__).resolve().parents[1] / "config"),
    )
)

_SUPPORTED_TRANSFORMS = {"indicator", "scale", "inverse_scale"}
_SUPPORTED_OPERATORS = {"gt", "gte", "lt", "lte", "eq", "neq"}


def _as_mapping(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be an object.")
    return value


def _as_number(value: Any, label: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{label} must be numeric.")
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must be numeric.") from exc
    if not np.isfinite(number):
        raise ValueError(f"{label} must be finite.")
    return number


def _required_text(value: Any, label: str) -> str:
    text = str(value or "").strip()
    if not text:
        raise ValueError(f"{label} is required.")
    return text


def validate_model_task_config(config: dict[str, Any]) -> dict[str, Any]:
    _as_mapping(config, "model task config")
    _required_text(config.get("id"), "id")
    _required_text(config.get("name"), "name")
    if "enabled" not in config or not isinstance(config.get("enabled"), bool):
        raise ValueError("enabled must be present and boolean.")

    model_type = _required_text(config.get("model_type"), "model_type")
    target = _as_mapping(config.get("target"), "target")
    task_type = _required_text(target.get("task_type"), "target.task_type")
    if model_type != task_type:
        raise ValueError("model_type must match target.task_type.")
    if task_type != "regression":
        raise ValueError("ideal donor target must be configured as regression.")

    _required_text(target.get("column"), "target.column")
    _as_number(target.get("base", 0.0), "target.base")
    bounds = _as_mapping(target.get("bounds"), "target.bounds")
    lower_bound = _as_number(bounds.get("min"), "target.bounds.min")
    upper_bound = _as_number(bounds.get("max"), "target.bounds.max")
    if lower_bound >= upper_bound:
        raise ValueError("target.bounds.min must be less than target.bounds.max.")
    if lower_bound < 0.0 or upper_bound > 1.0:
        raise ValueError("probability bounds must stay within 0.0 and 1.0.")

    accuracy_tolerance = _as_number(
        target.get("accuracy_tolerance"),
        "target.accuracy_tolerance",
    )
    if accuracy_tolerance <= 0.0:
        raise ValueError("target.accuracy_tolerance must be positive.")
    if "min_accuracy" not in target and "max_accuracy" not in target:
        raise ValueError("target.min_accuracy or target.max_accuracy is required.")
    if "min_accuracy" in target:
        min_accuracy = _as_number(target.get("min_accuracy"), "target.min_accuracy")
        if not 0.0 < min_accuracy < 1.0:
            raise ValueError("target.min_accuracy must be between 0.0 and 1.0.")
    if "max_accuracy" in target:
        max_accuracy = _as_number(target.get("max_accuracy"), "target.max_accuracy")
        if not 0.0 < max_accuracy < 1.0:
            raise ValueError("target.max_accuracy must be between 0.0 and 1.0.")

    terms = target.get("terms")
    if not isinstance(terms, list) or not terms:
        raise ValueError("target.terms must be a non-empty list.")
    for index, raw_term in enumerate(terms, start=1):
        term = _as_mapping(raw_term, f"target.terms[{index}]")
        _required_text(term.get("feature"), f"target.terms[{index}].feature")
        transform = _required_text(
            term.get("transform"),
            f"target.terms[{index}].transform",
        )
        if transform not in _SUPPORTED_TRANSFORMS:
            raise ValueError(
                f"target.terms[{index}].transform is not supported: {transform}"
            )
        _as_number(term.get("weight"), f"target.terms[{index}].weight")
        if transform == "indicator":
            operator = _required_text(
                term.get("operator"),
                f"target.terms[{index}].operator",
            )
            if operator not in _SUPPORTED_OPERATORS:
                raise ValueError(
                    f"target.terms[{index}].operator is not supported: {operator}"
                )
            if "value" not in term:
                raise ValueError(f"target.terms[{index}].value is required.")
        else:
            lower = _as_number(term.get("lower"), f"target.terms[{index}].lower")
            upper = _as_number(term.get("upper"), f"target.terms[{index}].upper")
            if lower >= upper:
                raise ValueError(
                    f"target.terms[{index}].lower must be less than upper."
                )

    jitter = target.get("jitter")
    if jitter is not None:
        jitter_map = _as_mapping(jitter, "target.jitter")
        _required_text(jitter_map.get("feature"), "target.jitter.feature")
        scale = _as_number(jitter_map.get("scale"), "target.jitter.scale")
        if scale < 0.0:
            raise ValueError("target.jitter.scale must be non-negative.")
        center = _as_number(jitter_map.get("center", 0.0), "target.jitter.center")
        if not 0.0 <= center <= 1.0:
            raise ValueError("target.jitter.center must be between 0.0 and 1.0.")
        modulo = int(_as_number(jitter_map.get("modulo"), "target.jitter.modulo"))
        if modulo < 2:
            raise ValueError("target.jitter.modulo must be at least 2.")

    prediction_defaults = _as_mapping(
        config.get("prediction_defaults"),
        "prediction_defaults",
    )
    output = _as_mapping(prediction_defaults.get("output"), "prediction_defaults.output")
    _required_text(output.get("name"), "prediction_defaults.output.name")
    output_task_type = _required_text(
        output.get("task_type"),
        "prediction_defaults.output.task_type",
    )
    if output_task_type != task_type:
        raise ValueError("prediction_defaults.output.task_type must match target.")
    _as_number(output.get("clip_min"), "prediction_defaults.output.clip_min")
    _as_number(output.get("clip_max"), "prediction_defaults.output.clip_max")

    promotion = _as_mapping(config.get("promotion"), "promotion")
    _required_text(promotion.get("primary_metric"), "promotion.primary_metric")
    _required_text(promotion.get("direction"), "promotion.direction")
    if not isinstance(promotion.get("fallback_metrics"), list):
        raise ValueError("promotion.fallback_metrics must be a list.")
    return config


def load_model_task_config(
    filename: str,
    *,
    config_dir: Path | None = None,
) -> dict[str, Any]:
    path = (config_dir or DEFAULT_CONFIG_DIR) / filename
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise ValueError(f"Could not read model task config: {path}") from exc
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON model task config: {path}") from exc
    if not isinstance(payload, dict):
        raise ValueError("model task config root must be an object.")
    return validate_model_task_config(payload)


def _numeric(frame: pd.DataFrame, feature: str) -> pd.Series:
    if feature not in frame.columns:
        raise ValueError(f"Configured target feature is missing: {feature}")
    return pd.to_numeric(frame[feature], errors="coerce").fillna(0.0)


def _indicator(values: pd.Series, operator: str, expected: Any) -> pd.Series:
    expected_numeric = pd.to_numeric(pd.Series([expected]), errors="coerce").iloc[0]
    if pd.isna(expected_numeric):
        text = values.fillna("").astype(str)
        expected_text = str(expected)
        if operator == "eq":
            return text.eq(expected_text).astype(float)
        if operator == "neq":
            return text.ne(expected_text).astype(float)
        raise ValueError("Text indicators support only eq and neq operators.")

    numeric = pd.to_numeric(values, errors="coerce").fillna(0.0)
    if operator == "gt":
        return numeric.gt(float(expected_numeric)).astype(float)
    if operator == "gte":
        return numeric.ge(float(expected_numeric)).astype(float)
    if operator == "lt":
        return numeric.lt(float(expected_numeric)).astype(float)
    if operator == "lte":
        return numeric.le(float(expected_numeric)).astype(float)
    if operator == "eq":
        return numeric.eq(float(expected_numeric)).astype(float)
    if operator == "neq":
        return numeric.ne(float(expected_numeric)).astype(float)
    raise ValueError(f"Unsupported indicator operator: {operator}")


def _scaled(values: pd.Series, lower: float, upper: float) -> pd.Series:
    return ((values - lower) / (upper - lower)).clip(lower=0.0, upper=1.0)


def _jitter(frame: pd.DataFrame, config: dict[str, Any]) -> pd.Series:
    feature = _required_text(config.get("feature"), "target.jitter.feature")
    if feature not in frame.columns:
        raise ValueError(f"Configured jitter feature is missing: {feature}")
    modulo = int(_as_number(config.get("modulo"), "target.jitter.modulo"))
    scale = _as_number(config.get("scale"), "target.jitter.scale")
    center = _as_number(config.get("center", 0.0), "target.jitter.center")
    hashed = pd.util.hash_pandas_object(
        frame[feature].fillna("").astype(str),
        index=False,
    ).astype("uint64")
    fraction = (hashed % modulo).astype(float) / float(modulo - 1)
    return (fraction - center) * scale


def compute_configured_probability_target(
    frame: pd.DataFrame,
    target_config: dict[str, Any],
) -> pd.Series:
    target = validate_model_task_config(
        {
            "id": "validation_context",
            "name": "Validation Context",
            "enabled": True,
            "model_type": target_config.get("task_type"),
            "target": target_config,
            "prediction_defaults": {
                "output": {
                    "name": "probability",
                    "task_type": target_config.get("task_type"),
                    "clip_min": target_config.get("bounds", {}).get("min"),
                    "clip_max": target_config.get("bounds", {}).get("max"),
                }
            },
            "promotion": {
                "primary_metric": "root_mean_squared_error",
                "direction": "minimize",
                "fallback_metrics": [],
            },
        }
    )["target"]
    bounds = target["bounds"]
    score = pd.Series(
        _as_number(target.get("base", 0.0), "target.base"),
        index=frame.index,
        dtype="float64",
    )
    for term in target["terms"]:
        feature = str(term["feature"])
        weight = _as_number(term.get("weight"), f"{feature}.weight")
        transform = str(term["transform"])
        values = _numeric(frame, feature)
        if transform == "indicator":
            contribution = _indicator(values, str(term["operator"]), term["value"])
        else:
            contribution = _scaled(
                values,
                _as_number(term["lower"], f"{feature}.lower"),
                _as_number(term["upper"], f"{feature}.upper"),
            )
            if transform == "inverse_scale":
                contribution = 1.0 - contribution
        score = score + weight * contribution
    if isinstance(target.get("jitter"), dict):
        score = score + _jitter(frame, target["jitter"])
    return score.clip(
        lower=_as_number(bounds.get("min"), "target.bounds.min"),
        upper=_as_number(bounds.get("max"), "target.bounds.max"),
    )
