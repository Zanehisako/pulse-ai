from __future__ import annotations

import json
from copy import deepcopy
from functools import lru_cache
from pathlib import Path

from django.conf import settings
from django.core.exceptions import ImproperlyConfigured

SUPPORTED_METRICS = {"count", "sum"}


def _fail(message: str) -> None:
    raise ImproperlyConfigured(f"Invalid workspace widget sources config: {message}")


def _config_path() -> Path:
    return Path(settings.WORKSPACE_WIDGET_SOURCES_CONFIG_PATH)


def _validate_option(option: object, *, kind: str, index: int) -> dict:
    if not isinstance(option, dict):
        _fail(f"{kind}[{index}] must be an object")

    value = option.get("value")
    label = option.get("label")
    if not isinstance(value, str) or not value.strip():
        _fail(f"{kind}[{index}].value must be a non-empty string")
    if not isinstance(label, str) or not label.strip():
        _fail(f"{kind}[{index}].label must be a non-empty string")
    return option


def _validate_source(source: object, index: int, known_ids: set[str]) -> dict:
    if not isinstance(source, dict):
        _fail(f"sources[{index}] must be an object")

    source_id = source.get("id")
    label = source.get("label")
    app_label = source.get("app_label", "workspace")
    model = source.get("model")
    group_by_options = source.get("group_by")
    metrics = source.get("metrics")

    if not isinstance(source_id, str) or not source_id.strip():
        _fail(f"sources[{index}].id must be a non-empty string")
    if source_id in known_ids:
        _fail(f"duplicate source id '{source_id}'")
    if not isinstance(label, str) or not label.strip():
        _fail(f"sources[{index}].label must be a non-empty string")
    if not isinstance(app_label, str) or not app_label.strip():
        _fail(f"sources[{index}].app_label must be a non-empty string")
    if not isinstance(model, str) or not model.strip():
        _fail(f"sources[{index}].model must be a non-empty string")
    if not isinstance(group_by_options, list) or not group_by_options:
        _fail(f"sources[{index}].group_by must be a non-empty list")
    if not isinstance(metrics, list) or not metrics:
        _fail(f"sources[{index}].metrics must be a non-empty list")

    known_ids.add(source_id)

    group_values: set[str] = set()
    for option_index, option in enumerate(group_by_options):
        validated = _validate_option(
            option,
            kind=f"sources[{index}].group_by",
            index=option_index,
        )
        if validated["value"] in group_values:
            _fail(
                f"sources[{index}].group_by contains duplicate value "
                f"'{validated['value']}'",
            )
        group_values.add(validated["value"])

    metric_values: set[str] = set()
    for option_index, option in enumerate(metrics):
        validated = _validate_option(
            option,
            kind=f"sources[{index}].metrics",
            index=option_index,
        )
        metric_value = validated["value"]
        if metric_value not in SUPPORTED_METRICS:
            _fail(
                f"sources[{index}].metrics[{option_index}].value '{metric_value}' "
                "is not supported",
            )
        if metric_value in metric_values:
            _fail(
                f"sources[{index}].metrics contains duplicate value '{metric_value}'",
            )
        metric_values.add(metric_value)

        metric_field = validated.get("field")
        if metric_field is not None and (
            not isinstance(metric_field, str) or not metric_field.strip()
        ):
            _fail(
                f"sources[{index}].metrics[{option_index}].field must be a "
                "non-empty string when provided",
            )

    default_group_by = source.get("default_group_by")
    if default_group_by is not None and default_group_by not in group_values:
        _fail(
            f"sources[{index}].default_group_by '{default_group_by}' must match an "
            "allowed group_by value",
        )

    default_metric = source.get("default_metric")
    if default_metric is not None and default_metric not in metric_values:
        _fail(
            f"sources[{index}].default_metric '{default_metric}' must match an "
            "allowed metric value",
        )

    return source


@lru_cache(maxsize=1)
def _load_widget_sources() -> dict:
    config_path = _config_path()
    try:
        raw = json.loads(config_path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ImproperlyConfigured(
            f"Workspace widget sources config was not found at {config_path}",
        ) from exc
    except json.JSONDecodeError as exc:
        raise ImproperlyConfigured(
            f"Workspace widget sources config at {config_path} is not valid JSON: "
            f"{exc}",
        ) from exc

    if not isinstance(raw, dict):
        _fail("root value must be an object")

    sources = raw.get("sources")
    if not isinstance(sources, list) or not sources:
        _fail("root.sources must be a non-empty list")

    known_ids: set[str] = set()
    for index, source in enumerate(sources):
        _validate_source(source, index, known_ids)

    return raw


def reset_widget_sources_cache() -> None:
    _load_widget_sources.cache_clear()


def get_widget_sources_catalog(*, reload: bool = False) -> dict:
    if reload:
        reset_widget_sources_cache()
    return deepcopy(_load_widget_sources())


def list_widget_sources() -> list[dict]:
    return get_widget_sources_catalog()["sources"]


def get_widget_source(source_id: str) -> dict:
    for source in list_widget_sources():
        if source["id"] == source_id:
            return source
    raise ValueError("Widget source not allowed")


def get_metric_config(source: dict, metric_value: str) -> dict:
    for metric in source["metrics"]:
        if metric["value"] == metric_value:
            return metric
    raise ValueError("Invalid metric")
