from __future__ import annotations

import math
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from ml.core.feature_store import (
    ResolvedFeaturePayload,
    resolve_component_prediction_features,
    resolve_prediction_features,
)
from ml.core.forecast_horizon import extract_forecast_horizon, timeframe_for_days
from ml.core.model_metadata import runtime_prediction_defaults
from ml.core.prediction import run_prediction


BACKEND_ML_DIR = Path(__file__).resolve().parents[1]
DEFAULT_STOCKOUT_HYBRID_CONFIG_PATH = BACKEND_ML_DIR / "config" / "stockout_hybrid.json"
_CONFIG_CACHE: dict[Path, tuple[float, dict[str, Any]]] = {}


@dataclass(frozen=True)
class StockoutHorizonDefinition:
    key: str
    label: str
    output_key: str
    min_days: float | None
    max_days: float | None
    timeframe: str
    model_ids: tuple[str, ...]
    classifier_config: dict[str, Any]

    def contains_days(self, days: float) -> bool:
        if self.min_days is not None and days < self.min_days:
            return False
        if self.max_days is not None and days > self.max_days:
            return False
        return True


PredictionRunner = Callable[[Any, dict[str, Any]], dict[str, Any]]
FeatureResolver = Callable[..., ResolvedFeaturePayload]


def hybrid_stockout_enabled(runtime: Any) -> bool:
    config = hybrid_stockout_config(runtime)
    return bool(config and config.get("enabled", True) is not False)


def hybrid_stockout_config(runtime: Any) -> dict[str, Any]:
    defaults = _merged_runtime_defaults(runtime)
    config = defaults.get("hybrid_stockout") or defaults.get("stockout_hybrid")
    return dict(config) if isinstance(config, dict) else {}


def stockout_orchestrator_routing_config() -> dict[str, Any]:
    payload = _load_json_hybrid_config().get("orchestrator_routing")
    return dict(payload) if isinstance(payload, dict) else {}


def find_hybrid_stockout_regression_runtime(
    registry: Any,
    *,
    preferred_model_id: str | None = None,
) -> Any | None:
    if preferred_model_id:
        runtime = registry.get(preferred_model_id)
        if runtime is not None and hybrid_stockout_enabled(runtime):
            return runtime

    global_config = _global_hybrid_config()
    configured_runtime = _resolve_configured_regression_runtime(
        registry,
        None,
        global_config,
    )
    if configured_runtime is not None:
        return configured_runtime

    loaded = registry.loaded() if hasattr(registry, "loaded") else []
    primary_candidates: list[Any] = []
    fallback_candidates: list[Any] = []
    for runtime in loaded:
        config = hybrid_stockout_config(runtime)
        if not config or config.get("enabled", True) is False:
            continue
        output = _merged_runtime_defaults(runtime).get("output")
        if not isinstance(output, dict):
            continue
        if output.get("name") == "days_until_stockout":
            if config.get("primary") is True:
                primary_candidates.append(runtime)
            else:
                fallback_candidates.append(runtime)
    candidates = primary_candidates or fallback_candidates
    return candidates[0] if candidates else None


def _resolve_configured_regression_runtime(
    registry: Any,
    fallback_runtime: Any | None,
    config: dict[str, Any],
) -> Any | None:
    regression = config.get("regression")
    if not isinstance(regression, dict):
        return fallback_runtime
    if str(regression.get("type") or "").strip().lower() != "model":
        return fallback_runtime

    model_id = str(regression.get("model_id") or "").strip()
    if not model_id or not hasattr(registry, "get"):
        return fallback_runtime
    runtime = registry.get(model_id)
    if runtime is None:
        return fallback_runtime
    if getattr(runtime, "status", "loaded") != "loaded":
        return fallback_runtime
    return runtime


def build_hybrid_stockout_prediction(
    *,
    registry: Any,
    regression_runtime: Any,
    request_features: dict[str, Any],
    query: str = "",
    resolved_regression_features: dict[str, Any] | None = None,
    regression_output: dict[str, Any] | None = None,
    prediction_runner: PredictionRunner = run_prediction,
    feature_resolver: FeatureResolver = resolve_prediction_features,
) -> dict[str, Any] | None:
    config = hybrid_stockout_config(regression_runtime)
    if not config or config.get("enabled", True) is False:
        return None

    horizons = _horizon_definitions(config)
    if not horizons:
        return None
    active_regression_runtime = _resolve_configured_regression_runtime(
        registry,
        regression_runtime,
        config,
    )

    request_features = _normalize_stockout_request_features(
        request_features or {},
        config,
    )
    forecast_horizon = extract_forecast_horizon(query, request_features)

    if resolved_regression_features is None and regression_output is None:
        component_payloads = _resolve_component_fanout_payloads(
            config=config,
            regression_runtime=active_regression_runtime,
            request_features=request_features,
            forecast_horizon=forecast_horizon.as_dict() if forecast_horizon else None,
            feature_resolver=feature_resolver,
        )
        if len(component_payloads) > 1:
            component_outputs: list[dict[str, Any]] = []
            component_errors: list[str] = []
            for component, resolved_payload in component_payloads:
                component_request = dict(request_features)
                component_request[_component_key(config)] = component
                for key in _component_request_keys(config):
                    component_request.setdefault(key, component)
                blood_group = resolved_payload.features.get("blood_group")
                if blood_group not in {None, ""}:
                    component_request["blood_group"] = blood_group
                try:
                    component_regression_output = (
                        _configured_regression_output(
                            config=config,
                            regression_runtime=active_regression_runtime,
                            features=resolved_payload.features,
                            model_regression_output={},
                        )
                        if _has_configured_regression(config)
                        else prediction_runner(active_regression_runtime, resolved_payload.features)
                    )
                    component_output = build_hybrid_stockout_prediction(
                        registry=registry,
                        regression_runtime=regression_runtime,
                        request_features=component_request,
                        query=query,
                        resolved_regression_features=resolved_payload.features,
                        regression_output=component_regression_output,
                        prediction_runner=prediction_runner,
                        feature_resolver=feature_resolver,
                    )
                    if component_output:
                        component_output["feature_resolution"] = resolved_payload.metadata
                        component_outputs.append(component_output)
                except Exception as exc:
                        component_errors.append(f"{component}: {exc}")
            if component_outputs:
                aggregated = _aggregate_component_outputs(
                    config=config,
                    request_features=request_features,
                    component_outputs=component_outputs,
                )
                if component_errors:
                    aggregated.setdefault("warnings", []).extend(component_errors)
                return aggregated

    if regression_output is None:
        resolved = feature_resolver(
            active_regression_runtime,
            request_features,
            forecast_horizon=forecast_horizon.as_dict() if forecast_horizon else None,
        )
        resolved_regression_features = resolved.features
        if _has_configured_regression(config):
            regression_output = _configured_regression_output(
                config=config,
                regression_runtime=active_regression_runtime,
                features=resolved.features,
                model_regression_output={},
            )
        else:
            regression_output = prediction_runner(active_regression_runtime, resolved.features)
    else:
        regression_output = dict(regression_output)

    base_features = dict(resolved_regression_features or request_features)
    model_regression_output = dict(regression_output)
    configured_regression_output = _configured_regression_output(
        config=config,
        regression_runtime=active_regression_runtime,
        features=base_features,
        model_regression_output=model_regression_output,
    )
    if configured_regression_output is not None:
        configured_regression_output["model_regression_output"] = model_regression_output
        regression_output = configured_regression_output

    estimated_days = _extract_estimated_days(regression_output)
    selected_horizon = _select_horizon(
        horizons,
        estimated_days=estimated_days,
        request_features=request_features,
        query=query,
    )

    horizon_rows: list[dict[str, Any]] = []
    risk_values: dict[str, float] = {}
    warnings: list[str] = []
    classifier_count = 0

    fallback_values = _fallback_probabilities(
        horizons,
        estimated_days=estimated_days,
        enabled=bool(config.get("fallback_probability", {}).get("enabled", True))
        if isinstance(config.get("fallback_probability"), dict)
        else True,
    )

    for horizon in horizons:
        probability = None
        probability_source = ""
        classifier_model_id = ""
        classifier_output: dict[str, Any] | None = None

        classifier_runtimes = _resolve_horizon_models(registry, horizon)
        if classifier_runtimes:
            classifier_model_id = ", ".join(
                str(getattr(runtime, "model_id", "") or "")
                for runtime in classifier_runtimes
            )
            try:
                classifier_outputs: list[dict[str, Any]] = []
                classifier_probabilities: list[float] = []
                for classifier_runtime in classifier_runtimes:
                    classifier_request = {
                        **base_features,
                        **_horizon_feature_payload(horizon),
                    }
                    resolved_classifier = feature_resolver(
                        classifier_runtime,
                        classifier_request,
                        forecast_horizon=_horizon_forecast_payload(horizon),
                    )
                    output = prediction_runner(
                        classifier_runtime,
                        resolved_classifier.features,
                    )
                    classifier_outputs.append(output)
                    classifier_probability = _extract_probability(output, horizon.output_key)
                    if classifier_probability is not None:
                        classifier_probabilities.append(classifier_probability)
                probability = _aggregate_probabilities(
                    classifier_probabilities,
                    horizon.classifier_config,
                )
                classifier_output = (
                    classifier_outputs[0]
                    if len(classifier_outputs) == 1
                    else {"outputs": classifier_outputs}
                )
                if probability is not None:
                    classifier_count += len(classifier_probabilities)
                    probability_source = "classifier"
            except Exception as exc:
                warnings.append(
                    f"{classifier_model_id or horizon.key} probability failed: {exc}"
                )
        elif horizon.model_ids:
            warnings.append(
                f"No loaded classifier found for horizon {horizon.label}: "
                + ", ".join(horizon.model_ids)
            )

        if probability is None:
            configured_probability = _configured_classifier_probability(
                horizon=horizon,
                config=config,
                features=base_features,
                estimated_days=estimated_days,
            )
            if configured_probability is not None:
                probability = configured_probability
                probability_source = "configured_classifier"
                classifier_model_id = str(
                    horizon.classifier_config.get("model_id")
                    or horizon.classifier_config.get("id")
                    or f"configured_{horizon.key}_classifier"
                )

        if probability is None:
            regression_probability = _extract_probability(regression_output, horizon.output_key)
            if regression_probability is not None:
                probability = regression_probability
                probability_source = "regression_output"
                classifier_model_id = str(
                    getattr(active_regression_runtime, "model_id", "") or ""
                )

        if probability is None:
            probability = fallback_values.get(horizon.key)
            probability_source = "regression_fallback" if probability is not None else ""

        if probability is not None:
            probability = _round_probability(probability)
            risk_values[horizon.output_key] = probability

        horizon_rows.append(
            {
                "key": horizon.key,
                "horizon": horizon.label,
                "stockout_probability": probability,
                "probability_source": probability_source or "unavailable",
                "model_id": classifier_model_id,
                "model_ids": list(horizon.model_ids),
                "model_output": classifier_output or {},
            }
        )

    selected_row = _selected_horizon_row(horizon_rows, selected_horizon)
    selected_probability = selected_row.get("stockout_probability")
    risk_level, recommended_action = _risk_and_action(
        config,
        probability=selected_probability,
        estimated_days=estimated_days,
        horizon_key=selected_horizon.key,
    )

    response: dict[str, Any] = {
        **_identity_fields(request_features, config),
        **risk_values,
        "estimated_days_until_stockout": _render_estimated_days(estimated_days),
        "stockout_probability": selected_probability,
        "horizon": selected_horizon.label,
        "risk_level": risk_level,
        "recommended_action": recommended_action,
        "confidence": _confidence(config, classifier_count, len(horizons)),
        "hybrid_prediction": True,
        "horizons": horizon_rows,
        "regression_model_id": str(getattr(active_regression_runtime, "model_id", "") or ""),
        "regression_output": regression_output,
    }
    if warnings:
        response["warnings"] = warnings
    return response


def build_hybrid_stockout_batch_prediction(
    *,
    registry: Any,
    regression_runtime: Any,
    request_rows: list[dict[str, Any]],
    query: str = "",
    prediction_runner: PredictionRunner = run_prediction,
    feature_resolver: FeatureResolver = resolve_prediction_features,
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    for index, row in enumerate(request_rows or []):
        if not isinstance(row, dict):
            errors.append({"index": index, "error": "request row must be an object"})
            continue
        try:
            prediction = build_hybrid_stockout_prediction(
                registry=registry,
                regression_runtime=regression_runtime,
                request_features=row,
                query=query,
                prediction_runner=prediction_runner,
                feature_resolver=feature_resolver,
            )
            if prediction is None:
                errors.append({"index": index, "error": "hybrid stockout is disabled"})
                continue
            rows.append(prediction)
        except Exception as exc:
            errors.append({"index": index, "error": str(exc)})

    result: dict[str, Any] = {
        "query_kind": "batch_stockout_horizon_predictions",
        "row_count": len(rows),
        "rows": rows,
    }
    if errors:
        result["errors"] = errors
    return result


def validate_stockout_hybrid_config(payload: dict[str, Any]) -> None:
    config = payload.get("hybrid_stockout") if isinstance(payload, dict) else None
    if not isinstance(config, dict):
        config = payload if isinstance(payload, dict) else {}
    if not config or config.get("enabled", True) is False:
        return

    horizons = config.get("horizons")
    if not isinstance(horizons, list) or not horizons:
        raise RuntimeError("hybrid_stockout.horizons must be a non-empty list")
    for index, horizon in enumerate(horizons):
        if not isinstance(horizon, dict):
            raise RuntimeError(f"horizons[{index}] must be an object")
        if not str(horizon.get("output_key") or "").strip():
            raise RuntimeError("horizons[].output_key is required")


def _merged_runtime_defaults(runtime: Any) -> dict[str, Any]:
    model_id = str(getattr(runtime, "model_id", "") or "")
    clean_id = model_id.split("_champion")[0].split("_@")[0]
    json_payload = _load_json_hybrid_config()
    base_defaults = runtime_prediction_defaults(clean_id or model_id)
    runtime_defaults = getattr(runtime, "defaults", {}) or {}
    if not isinstance(runtime_defaults, dict):
        runtime_defaults = {}
    runtime_has_hybrid = _has_explicit_runtime_hybrid_config(
        runtime_defaults,
        base_defaults,
    )
    if json_payload:
        base_defaults = {
            key: value
            for key, value in base_defaults.items()
            if key not in {"hybrid_stockout", "stockout_hybrid"}
        }
        if not runtime_has_hybrid:
            runtime_defaults = {
                key: value
                for key, value in runtime_defaults.items()
                if key not in {"hybrid_stockout", "stockout_hybrid"}
            }
    merged = _deep_merge({}, base_defaults)
    merged = _deep_merge(merged, runtime_defaults)
    json_config = _json_hybrid_config_for_model(model_id or clean_id)
    if json_config and not runtime_has_hybrid:
        merged["hybrid_stockout"] = _deep_merge(
            merged.get("hybrid_stockout", {}),
            json_config,
        )
    elif json_payload and not runtime_has_hybrid:
        merged.pop("hybrid_stockout", None)
        merged.pop("stockout_hybrid", None)
    return merged


def _has_explicit_runtime_hybrid_config(
    runtime_defaults: dict[str, Any],
    base_defaults: dict[str, Any],
) -> bool:
    runtime_config = runtime_defaults.get("hybrid_stockout") or runtime_defaults.get(
        "stockout_hybrid"
    )
    if not isinstance(runtime_config, dict):
        return False
    base_config = base_defaults.get("hybrid_stockout") or base_defaults.get(
        "stockout_hybrid"
    )
    return runtime_config != base_config


def _json_hybrid_config_for_model(model_id: str) -> dict[str, Any]:
    payload = _load_json_hybrid_config()
    if not payload:
        return {}
    clean_id = str(model_id or "").split("_champion")[0].split("_@")[0]
    global_config = payload.get("hybrid_stockout")
    if not isinstance(global_config, dict):
        global_config = {}
    model_configs = payload.get("models")
    selected: dict[str, Any] = {}
    if isinstance(model_configs, dict):
        for key in (model_id, clean_id):
            value = model_configs.get(key)
            if isinstance(value, dict):
                selected = _deep_merge(selected, value)
    if not selected:
        return {}
    return _deep_merge(global_config, selected)


def _load_json_hybrid_config() -> dict[str, Any]:
    raw_path = os.getenv("PIOS_STOCKOUT_HYBRID_CONFIG", "").strip()
    path = Path(raw_path).expanduser().resolve() if raw_path else DEFAULT_STOCKOUT_HYBRID_CONFIG_PATH
    if not path.exists():
        return {}
    try:
        mtime = path.stat().st_mtime
        cached = _CONFIG_CACHE.get(path)
        if cached and cached[0] == mtime:
            return cached[1]
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    if not isinstance(payload, dict):
        return {}
    _CONFIG_CACHE[path] = (mtime, payload)
    return payload


def _deep_merge(base: dict[str, Any], extra: dict[str, Any]) -> dict[str, Any]:
    result = dict(base)
    for key, value in extra.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def _global_hybrid_config() -> dict[str, Any]:
    value = _load_json_hybrid_config().get("hybrid_stockout")
    return dict(value) if isinstance(value, dict) else {}


def _merged_section_config(config: dict[str, Any], section: str) -> dict[str, Any]:
    base = _global_hybrid_config().get(section)
    override = config.get(section)
    merged = dict(base) if isinstance(base, dict) else {}
    if isinstance(override, dict):
        merged = _deep_merge(merged, override)
    return merged


def _horizon_definitions(config: dict[str, Any]) -> list[StockoutHorizonDefinition]:
    raw_horizons = config.get("horizons")
    if not isinstance(raw_horizons, list):
        return []

    horizons: list[StockoutHorizonDefinition] = []
    for index, row in enumerate(raw_horizons):
        if not isinstance(row, dict):
            continue
        key = str(row.get("key") or row.get("id") or f"horizon_{index + 1}").strip()
        if not key:
            continue
        label = str(row.get("label") or row.get("horizon") or key).strip()
        min_days = _coerce_float(row.get("min_days"))
        max_days = _coerce_float(row.get("max_days"))
        if max_days is None:
            timeframe = "long"
        elif max_days <= 7:
            timeframe = "short"
        elif max_days <= 30:
            timeframe = "medium"
        else:
            timeframe = timeframe_for_days(max_days)
        timeframe = str(row.get("timeframe") or timeframe).strip() or timeframe
        output_key = str(row.get("output_key") or f"risk_{key}").strip()
        model_ids = _configured_model_ids(row)
        classifier_config = row.get("classifier")
        if not isinstance(classifier_config, dict):
            classifier_config = {}
        horizons.append(
            StockoutHorizonDefinition(
                key=key,
                label=label,
                output_key=output_key,
                min_days=min_days,
                max_days=max_days,
                timeframe=timeframe,
                model_ids=tuple(model_ids),
                classifier_config=dict(classifier_config),
            )
        )
    return horizons


def _configured_model_ids(row: dict[str, Any]) -> list[str]:
    values: list[Any] = []
    for key in (
        "model_id",
        "probability_model_id",
        "classifier_model_id",
        "model_ids",
        "probability_model_ids",
        "classifier_model_ids",
    ):
        value = row.get(key)
        if isinstance(value, list):
            values.extend(value)
        elif value not in {None, ""}:
            values.append(value)
    classifier = row.get("classifier")
    if isinstance(classifier, dict):
        for key in (
            "model_id",
            "probability_model_id",
            "classifier_model_id",
            "model_ids",
            "probability_model_ids",
            "classifier_model_ids",
        ):
            value = classifier.get(key)
            if isinstance(value, list):
                values.extend(value)
            elif value not in {None, ""}:
                values.append(value)
    ids: list[str] = []
    for value in values:
        text = str(value or "").strip()
        if text and text not in ids:
            ids.append(text)
    return ids


def _extract_estimated_days(output: dict[str, Any]) -> float | None:
    for key in (
        "estimated_days_until_stockout",
        "expected_days_until_stockout",
        "days_until_stockout",
        "prediction",
    ):
        value = _coerce_float(output.get(key))
        if value is not None and math.isfinite(value):
            return value
    return None


def _extract_probability(output: dict[str, Any], configured_key: str) -> float | None:
    for key in (
        configured_key,
        "stockout_probability",
        "probability",
        "risk",
    ):
        value = _coerce_float(output.get(key))
        if value is not None:
            return _clamp_probability(value)
    probabilities = output.get("probabilities")
    if isinstance(probabilities, (list, tuple)) and probabilities:
        value = _coerce_float(probabilities[-1])
        if value is not None:
            return _clamp_probability(value)
    prediction = _coerce_float(output.get("prediction"))
    if prediction is not None and 0 <= prediction <= 1:
        return _clamp_probability(prediction)
    return None


def _resolve_horizon_model(
    registry: Any,
    horizon: StockoutHorizonDefinition,
) -> Any | None:
    runtimes = _resolve_horizon_models(registry, horizon)
    return runtimes[0] if runtimes else None


def _resolve_horizon_models(
    registry: Any,
    horizon: StockoutHorizonDefinition,
) -> list[Any]:
    runtimes: list[Any] = []
    for model_id in horizon.model_ids:
        runtime = registry.get(model_id) if hasattr(registry, "get") else None
        if runtime is not None and getattr(runtime, "status", "loaded") == "loaded":
            runtimes.append(runtime)
    return runtimes


def _aggregate_probabilities(
    values: list[float],
    classifier_config: dict[str, Any],
) -> float | None:
    if not values:
        return None
    strategy = str(classifier_config.get("aggregation") or "first").strip().lower()
    if strategy == "mean":
        return _clamp_probability(sum(values) / len(values))
    if strategy == "max":
        return _clamp_probability(max(values))
    if strategy == "min":
        return _clamp_probability(min(values))
    return _clamp_probability(values[0])


def _select_horizon(
    horizons: list[StockoutHorizonDefinition],
    *,
    estimated_days: float | None,
    request_features: dict[str, Any],
    query: str,
) -> StockoutHorizonDefinition:
    requested = extract_forecast_horizon(query, request_features)
    if requested is not None:
        if requested.days is not None:
            for horizon in horizons:
                if horizon.contains_days(requested.days):
                    return horizon
        for horizon in horizons:
            if horizon.timeframe == requested.timeframe:
                return horizon
    if estimated_days is not None:
        for horizon in horizons:
            if horizon.contains_days(estimated_days):
                return horizon
    return horizons[0]


def _selected_horizon_row(
    rows: list[dict[str, Any]],
    horizon: StockoutHorizonDefinition,
) -> dict[str, Any]:
    for row in rows:
        if row.get("key") == horizon.key:
            return row
    return rows[0] if rows else {}


def _resolve_component_fanout_payloads(
    *,
    config: dict[str, Any],
    regression_runtime: Any,
    request_features: dict[str, Any],
    forecast_horizon: dict[str, Any] | None,
    feature_resolver: FeatureResolver = resolve_prediction_features,
) -> list[tuple[str, ResolvedFeaturePayload]]:
    fanout = config.get("component_fanout")
    if not isinstance(fanout, dict) or fanout.get("enabled", True) is False:
        return []
    if _requested_component(request_features, config):
        return []
    component_key = _component_key(config)
    configured_components = [
        _normalize_component_value(value, config)
        for value in fanout.get("components", [])
        if _normalize_component_value(value, config)
    ]
    if configured_components:
        resolved_payloads: list[tuple[str, ResolvedFeaturePayload]] = []
        blood_groups = _fanout_blood_groups(config, request_features)
        for component in configured_components:
            for blood_group in blood_groups:
                component_request = dict(request_features)
                component_request[component_key] = component
                for key in _component_request_keys(config):
                    component_request.setdefault(key, component)
                if blood_group:
                    component_request["blood_group"] = blood_group
                try:
                    payload = feature_resolver(
                        regression_runtime,
                        component_request,
                        forecast_horizon=forecast_horizon,
                    )
                    if blood_group:
                        payload = _apply_blood_group_factor(
                            config,
                            payload,
                            blood_group,
                        )
                    resolved_payloads.append((component, payload))
                except Exception:
                    continue
        if resolved_payloads:
            return resolved_payloads

    try:
        payloads = resolve_component_prediction_features(
            regression_runtime,
            request_features,
            forecast_horizon=forecast_horizon,
            component_key=component_key,
        )
    except Exception:
        return []
    normalized: list[tuple[str, ResolvedFeaturePayload]] = []
    for component, payload in payloads:
        clean_component = _normalize_component_value(component, config)
        if clean_component and clean_component not in _component_all_values(config):
            normalized.append((clean_component, payload))
    return normalized


def _component_key(config: dict[str, Any]) -> str:
    fanout = _merged_section_config(config, "component_fanout")
    configured = str(fanout.get("component_key") or "").strip()
    if configured:
        return configured
    request_keys = _component_request_keys(config)
    return request_keys[-1] if request_keys else ""


def _component_request_keys(config: dict[str, Any]) -> list[str]:
    fanout = _merged_section_config(config, "component_fanout")
    values = fanout.get("request_keys")
    if not isinstance(values, list):
        values = []
    keys = [str(value).strip() for value in values if str(value).strip()]
    component_key = str(fanout.get("component_key") or "").strip()
    if component_key and component_key not in keys:
        keys.append(component_key)
    return keys


def _component_values(config: dict[str, Any]) -> set[str]:
    fanout = _merged_section_config(config, "component_fanout")
    return {
        _normalize_component_value(value, config)
        for value in fanout.get("components", [])
        if _normalize_component_value(value, config)
    }


def _component_all_values(config: dict[str, Any]) -> set[str]:
    fanout = _merged_section_config(config, "component_fanout")
    return {
        _normalize_component_value(value, config)
        for value in fanout.get("all_values", [])
        if _normalize_component_value(value, config)
    }


def _requested_component(features: dict[str, Any], config: dict[str, Any]) -> str:
    component_values = _component_values(config)
    for key in _component_request_keys(config):
        value = features.get(key)
        if value not in {None, ""}:
            normalized = _normalize_component_value(value, config)
            if normalized in component_values:
                return normalized
    return ""


def _requested_blood_group(features: dict[str, Any], config: dict[str, Any]) -> str:
    fanout = _merged_section_config(config, "blood_group_fanout")
    request_keys = fanout.get("request_keys", [])
    for key in [str(value).strip() for value in request_keys if str(value).strip()]:
        value = features.get(key)
        if value not in {None, ""} and _is_blood_group(value, config):
            return _normalize_blood_group(value)
    blood_type = features.get(_component_key(config))
    if _is_blood_group(blood_type, config):
        return _normalize_blood_group(blood_type)
    return ""


def _fanout_blood_groups(config: dict[str, Any], request_features: dict[str, Any]) -> list[str]:
    requested = _requested_blood_group(request_features, config)
    if requested:
        return [requested]
    fanout = _merged_section_config(config, "blood_group_fanout")
    if fanout.get("enabled", False) is False:
        return [""]
    groups = [
        _normalize_blood_group(value)
        for value in fanout.get("groups", [])
        if _is_blood_group(value, config)
    ]
    return groups or [""]


def _apply_blood_group_factor(
    config: dict[str, Any],
    payload: ResolvedFeaturePayload,
    blood_group: str,
) -> ResolvedFeaturePayload:
    blood_group = _normalize_blood_group(blood_group)
    fanout = _merged_section_config(config, "blood_group_fanout")
    factors = fanout.get("inventory_factors", {})
    factor = _coerce_float(factors.get(blood_group)) if isinstance(factors, dict) else None
    if factor is None:
        factor = 1.0
    features = dict(payload.features)
    fields = fanout.get("scaled_inventory_features", [])
    for key in [str(value).strip() for value in fields if str(value).strip()]:
        value = _coerce_float(features.get(key))
        if value is not None:
            features[key] = max(0, round(value * factor, 4))
    features["blood_group"] = blood_group
    metadata = dict(payload.metadata or {})
    metadata["blood_group"] = blood_group
    metadata["blood_group_inventory_factor"] = factor
    return ResolvedFeaturePayload(features=features, metadata=metadata)


def _aggregate_component_outputs(
    *,
    config: dict[str, Any],
    request_features: dict[str, Any],
    component_outputs: list[dict[str, Any]],
) -> dict[str, Any]:
    def probability(row: dict[str, Any]) -> float:
        value = _coerce_float(row.get("stockout_probability"))
        return value if value is not None else -1.0

    def days(row: dict[str, Any]) -> float:
        value = _coerce_float(row.get("estimated_days_until_stockout"))
        return value if value is not None else math.inf

    strategy = "max_probability"
    fanout = config.get("component_fanout")
    if isinstance(fanout, dict):
        strategy = str(fanout.get("overall_strategy") or strategy).strip().lower()
    if strategy == "min_days":
        selected = min(component_outputs, key=days)
    else:
        selected = max(component_outputs, key=lambda row: (probability(row), -days(row)))

    component_rows = []
    by_component: dict[str, Any] = {}
    for row in component_outputs:
        component = str(
            row.get("blood_component") or row.get("component") or ""
        ).strip()
        blood_group = str(row.get("blood_group") or "").strip()
        component_key = f"{component}:{blood_group}" if blood_group else component
        compact = {
            "blood_component": component,
            **({"blood_group": blood_group} if blood_group else {}),
            "estimated_days_until_stockout": row.get("estimated_days_until_stockout"),
            "stockout_probability": row.get("stockout_probability"),
            "horizon": row.get("horizon"),
            "risk_level": row.get("risk_level"),
            "recommended_action": row.get("recommended_action"),
        }
        for risk_key, value in row.items():
            if str(risk_key).startswith("risk_"):
                compact[risk_key] = value
        component_rows.append(compact)
        if component_key:
            by_component[component_key] = compact

    aggregated = {
        **_identity_fields(request_features, config),
        **{
            key: value
            for key, value in selected.items()
            if key.startswith("risk_")
        },
        "blood_component": "overall",
        "estimated_days_until_stockout": selected.get("estimated_days_until_stockout"),
        "stockout_probability": selected.get("stockout_probability"),
        "horizon": selected.get("horizon"),
        "risk_level": selected.get("risk_level"),
        "recommended_action": selected.get("recommended_action"),
        "confidence": selected.get("confidence"),
        "hybrid_prediction": True,
        "overall_strategy": strategy,
        "worst_component": (
            selected.get("blood_component")
            or selected.get("component")
        ),
        "worst_blood_group": selected.get("blood_group"),
        "component_predictions": component_rows,
        "stockout_by_component": by_component,
        "component_forecasts": component_rows,
        "horizons": selected.get("horizons", []),
        "regression_model_id": selected.get("regression_model_id"),
        "regression_output": selected.get("regression_output", {}),
    }
    warnings: list[str] = []
    for row in component_outputs:
        row_warnings = row.get("warnings")
        if isinstance(row_warnings, list):
            warnings.extend(str(item) for item in row_warnings)
    if warnings:
        aggregated["warnings"] = sorted(set(warnings))
    return aggregated


def _configured_regression_output(
    *,
    config: dict[str, Any],
    regression_runtime: Any,
    features: dict[str, Any],
    model_regression_output: dict[str, Any],
) -> dict[str, Any] | None:
    regression_config = config.get("regression")
    if not isinstance(regression_config, dict):
        return None
    regression_type = str(regression_config.get("type") or "").strip().lower()
    if regression_type not in {"inventory_runway", "stockout_runway"}:
        return None

    inventory = _first_number(
        features,
        regression_config.get("inventory_features")
        or ("current_inventory", "stock_end", "stock_start"),
        default=0.0,
    )
    units_used = _first_number(
        features,
        regression_config.get("usage_features") or ("units_used",),
        default=None,
    )
    scheduled = _first_number(features, ("scheduled_surgeries",), default=0.0)
    trauma = _first_number(features, ("trauma_cases",), default=0.0)
    wastage = _first_number(features, ("wastage",), default=0.0)
    collected = _first_number(features, ("units_collected",), default=0.0)
    if units_used is None:
        demand = 1.0 + scheduled * 0.45 + trauma * 0.8
    else:
        demand = units_used + scheduled * 0.25 + trauma * 0.5
    if _truthy(features.get("heavy_demand")):
        demand += 2.0
    if _truthy(features.get("disaster")):
        demand += 4.0
    if _truthy(features.get("holiday")):
        demand += 0.5
    collection_credit = _coerce_float(regression_config.get("collection_credit"))
    if collection_credit is None:
        collection_credit = 0.35
    net_daily_demand = max(0.1, demand + wastage - collected * collection_credit)
    net_daily_demand *= _scenario_demand_multiplier(features)
    estimated_days = max(0.0, inventory / net_daily_demand)
    if inventory <= 0 or _truthy(features.get("critical_stock")):
        estimated_days = min(estimated_days, 1.0 if inventory > 0 else 0.0)

    model_id = str(
        regression_config.get("model_id")
        or "stockout_days_inventory_runway_regression"
    ).strip()
    return {
        "model_id": model_id,
        "base_regression_runtime_id": str(getattr(regression_runtime, "model_id", "") or ""),
        "model_type": "configured_inventory_runway",
        "prediction": estimated_days,
        "days_until_stockout": estimated_days,
        "estimated_days_until_stockout": estimated_days,
        "prediction_name": "days_until_stockout",
        "prediction_unit": "days",
        "prediction_task_type": "regression",
        "used_inputs": {
            **{key: features.get(key) for key in sorted(features)},
            "computed_daily_demand": round(net_daily_demand, 4),
        },
        "missing_features": [],
    }


def _has_configured_regression(config: dict[str, Any]) -> bool:
    regression_config = config.get("regression")
    if not isinstance(regression_config, dict):
        return False
    return str(regression_config.get("type") or "").strip().lower() in {
        "inventory_runway",
        "stockout_runway",
    }


def _configured_classifier_probability(
    *,
    horizon: StockoutHorizonDefinition,
    config: dict[str, Any],
    features: dict[str, Any],
    estimated_days: float | None,
) -> float | None:
    classifier = horizon.classifier_config
    if not classifier:
        return None
    classifier_type = str(classifier.get("type") or "").strip().lower()
    if classifier_type not in {"inventory_pressure", "runway_bucket"}:
        return None
    if estimated_days is None:
        return None

    inventory = _first_number(features, ("current_inventory", "stock_end", "stock_start"), default=0.0)
    daily_demand = _first_number(features, ("computed_daily_demand", "units_used"), default=None)
    if daily_demand is None:
        scheduled = _first_number(features, ("scheduled_surgeries",), default=0.0)
        trauma = _first_number(features, ("trauma_cases",), default=0.0)
        daily_demand = 1.0 + scheduled * 0.45 + trauma * 0.8
    pressure = max(0.0, daily_demand / max(1.0, inventory + 1.0))
    pressure = min(1.5, pressure)

    scenario_priority = _scenario_priority(features)
    if horizon.contains_days(estimated_days):
        probability = 0.48 + 0.34 * min(1.0, pressure)
    elif horizon.min_days is not None and estimated_days < horizon.min_days:
        distance = horizon.min_days - estimated_days
        probability = 0.08 + 0.18 * math.exp(-distance / 12.0)
    else:
        upper = horizon.max_days if horizon.max_days is not None else horizon.min_days or 30.0
        distance = max(0.0, estimated_days - upper)
        probability = 0.05 + 0.34 * math.exp(-distance / max(1.0, upper / 2.0))

    if _truthy(features.get("critical_stock")) and horizon.max_days is not None and horizon.max_days <= 7:
        probability = max(probability, 0.72 + 0.18 * scenario_priority)
    if inventory <= 0 and horizon.max_days is not None and horizon.max_days <= 7:
        probability = max(probability, 0.74 + 0.24 * scenario_priority)

    min_probability = _coerce_float(classifier.get("min_probability"))
    max_probability = _coerce_float(classifier.get("max_probability"))
    if min_probability is not None:
        probability = max(probability, min_probability)
    if max_probability is not None:
        probability = min(probability, max_probability)
    return _round_probability(probability)


def _scenario_demand_multiplier(features: dict[str, Any]) -> float:
    return 0.85 + 0.3 * _scenario_priority(features)


def _scenario_priority(features: dict[str, Any]) -> float:
    config = _load_json_hybrid_config().get("hybrid_stockout")
    if not isinstance(config, dict):
        config = {}
    component = _normalize_component_value(
        features.get(_component_key(config)) or _requested_component(features, config),
        config,
    )
    group = _normalize_blood_group(features.get("blood_group"))
    fanout = _merged_section_config(config, "component_fanout")
    component_weights = fanout.get("priority_weights", {})
    blood_group_fanout = _merged_section_config(config, "blood_group_fanout")
    group_weights = blood_group_fanout.get("priority_weights", {})
    component_weight = _coerce_float(component_weights.get(component)) or 0.5
    group_weight = _coerce_float(group_weights.get(group)) or 0.55
    return max(0.0, min(1.0, (component_weight * 0.55) + (group_weight * 0.45)))


def _risk_and_action(
    config: dict[str, Any],
    *,
    probability: Any,
    estimated_days: float | None,
    horizon_key: str,
) -> tuple[str, str]:
    rules = config.get("risk_levels")
    if not isinstance(rules, list):
        rules = []
    probability_value = _coerce_float(probability)
    for rule in rules:
        if not isinstance(rule, dict):
            continue
        horizons = rule.get("horizons")
        if isinstance(horizons, list) and horizon_key not in {
            str(item) for item in horizons
        }:
            continue
        min_probability = _coerce_float(rule.get("min_probability"))
        max_estimated_days = _coerce_float(rule.get("max_estimated_days"))
        min_estimated_days = _coerce_float(rule.get("min_estimated_days"))
        if min_probability is not None and (
            probability_value is None or probability_value < min_probability
        ):
            continue
        if max_estimated_days is not None and (
            estimated_days is None or estimated_days > max_estimated_days
        ):
            continue
        if min_estimated_days is not None and (
            estimated_days is None or estimated_days < min_estimated_days
        ):
            continue
        return (
            str(rule.get("level") or "unknown"),
            str(rule.get("recommended_action") or rule.get("action") or ""),
        )
    return "unknown", str(
        config.get("default_recommended_action") or "Review inventory"
    )


def _confidence(config: dict[str, Any], classifier_count: int, horizon_count: int) -> str:
    configured = str(config.get("confidence") or "").strip().lower()
    if configured:
        return configured
    if classifier_count >= horizon_count and horizon_count > 0:
        return "high"
    if classifier_count > 0:
        return "medium"
    return "low"


def _identity_fields(features: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    component_keys = _component_request_keys(config)
    blood_group_keys = [
        str(value).strip()
        for value in _merged_section_config(config, "blood_group_fanout").get(
            "request_keys",
            [],
        )
        if str(value).strip()
    ]
    mapping = {
        "blood_component": tuple(
            dict.fromkeys([*component_keys, "blood_component", "component", "component_type"])
        ),
        "blood_group": tuple(
            dict.fromkeys([*blood_group_keys, "blood_group", "blood_product_type"])
        ),
        "location": ("location", "hospital", "hospital_id", "site"),
    }
    result: dict[str, Any] = {}
    for output_key, aliases in mapping.items():
        for alias in aliases:
            value = features.get(alias)
            if value not in {None, ""}:
                if output_key == "blood_group" and not _is_blood_group(value, config):
                    continue
                result[output_key] = value
                break
    return result


def _normalize_stockout_request_features(
    features: dict[str, Any],
    config: dict[str, Any],
) -> dict[str, Any]:
    normalized = dict(features or {})
    component_key = _component_key(config)
    if (
        component_key
        and _is_blood_group(normalized.get(component_key), config)
        and "blood_group" not in normalized
    ):
        normalized["blood_group"] = _normalize_blood_group(normalized.get(component_key))
        normalized.pop(component_key, None)
    component = _requested_component(normalized, config)
    if component:
        for key in _component_request_keys(config):
            normalized.setdefault(key, component)
        if component_key:
            normalized[component_key] = component
    return normalized


def _is_blood_group(value: Any, config: dict[str, Any]) -> bool:
    fanout = _merged_section_config(config, "blood_group_fanout")
    groups = fanout.get("groups", [])
    return _normalize_blood_group(value) in {
        _normalize_blood_group(group) for group in groups
    }


def _normalize_blood_group(value: Any) -> str:
    return str(value or "").strip().upper()


def _normalize_component_value(value: Any, config: dict[str, Any] | None = None) -> str:
    text = str(value or "").strip().upper().replace(" ", "_")
    if config is None:
        config = _load_json_hybrid_config().get("hybrid_stockout")
        if not isinstance(config, dict):
            config = {}
    fanout = _merged_section_config(config, "component_fanout")
    raw_aliases = fanout.get("aliases", {})
    aliases = {
        str(key).strip().upper().replace(" ", "_"): str(value).strip().upper()
        for key, value in raw_aliases.items()
        if str(key).strip() and str(value).strip()
    } if isinstance(raw_aliases, dict) else {}
    return aliases.get(text, text)


def _first_number(
    features: dict[str, Any],
    keys: Any,
    *,
    default: float | None,
) -> float | None:
    if isinstance(keys, str):
        keys = (keys,)
    for key in keys or ():
        value = _coerce_float(features.get(str(key)))
        if value is not None:
            return value
    return default


def _truthy(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return float(value) != 0.0
    return str(value or "").strip().lower() in {"1", "true", "yes", "y", "on"}


def _fallback_probabilities(
    horizons: list[StockoutHorizonDefinition],
    *,
    estimated_days: float | None,
    enabled: bool,
) -> dict[str, float]:
    if not enabled or estimated_days is None:
        return {}
    selected_index = 0
    for index, horizon in enumerate(horizons):
        if horizon.contains_days(estimated_days):
            selected_index = index
            break
    values: dict[str, float] = {}
    for index, horizon in enumerate(horizons):
        if index == selected_index:
            values[horizon.key] = 0.72
        else:
            distance = abs(index - selected_index)
            values[horizon.key] = 0.18 if distance == 1 else 0.04
    return values


def _probability_from_days_for_horizon(
    days_until_stockout: float,
    horizon: StockoutHorizonDefinition,
) -> float:
    min_probability = _coerce_float(horizon.classifier_config.get("min_probability"))
    max_probability = _coerce_float(horizon.classifier_config.get("max_probability"))
    if min_probability is None:
        min_probability = 0.3
    if max_probability is None:
        max_probability = 0.9
    min_probability = _clamp_probability(min_probability)
    max_probability = _clamp_probability(max_probability)
    if min_probability > max_probability:
        min_probability, max_probability = max_probability, min_probability

    days = max(0.0, float(days_until_stockout))
    min_days = horizon.min_days if horizon.min_days is not None else 0.0
    max_days = horizon.max_days

    if max_days is None:
        if days >= min_days:
            return _round_probability(max_probability)
        distance = max(0.0, min_days - days)
        return _round_probability(min_probability * math.exp(-distance / 14.0))

    span = max(1.0, max_days - min_days)
    if min_days <= days <= max_days:
        progress = (days - min_days) / span
        return _round_probability(max_probability - ((max_probability - min_probability) * progress))
    if days < min_days:
        distance = min_days - days
        return _round_probability(max_probability * math.exp(-distance / max(1.0, span / 2.0)))

    distance = days - max_days
    return _round_probability(min_probability * math.exp(-distance / max(1.0, span)))


def _horizon_feature_payload(horizon: StockoutHorizonDefinition) -> dict[str, Any]:
    target_days = horizon.max_days
    if target_days is None:
        target_days = horizon.min_days
    return {
        "horizon": horizon.key,
        "forecast_timeframe": horizon.timeframe,
        "prediction_timeframe": horizon.timeframe,
        "horizon_days": target_days,
        "forecast_horizon_days": target_days,
        "prediction_horizon_days": target_days,
    }


def _horizon_forecast_payload(horizon: StockoutHorizonDefinition) -> dict[str, Any]:
    target_days = horizon.max_days
    if target_days is None:
        target_days = horizon.min_days
    return {
        "timeframe": horizon.timeframe,
        "days": target_days,
        "source": f"hybrid_stockout:{horizon.key}",
    }


def _render_estimated_days(value: float | None) -> int | float | None:
    if value is None:
        return None
    rounded = round(value)
    if abs(value - rounded) < 0.05:
        return int(rounded)
    return round(value, 2)


def _round_probability(value: float) -> float:
    return round(_clamp_probability(value), 4)


def _clamp_probability(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


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
