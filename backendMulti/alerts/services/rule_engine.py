from __future__ import annotations

import logging
from datetime import timedelta
from hashlib import sha1
from typing import Any

from django.db.models import Q
from django.utils import timezone

from alerts.models import AlertEvent, AlertRule, AlertSeverity, AlertStatus, AlertTriggerType
from alerts.services.event_service import create_or_refresh_event, escalate_event
from alerts.services.trend_detector import TrendDetector

logger = logging.getLogger(__name__)


def _normalize_field_name(name: str) -> str:
    if "__" in name:
        name = name.split("__")[-1]
    if "." in name:
        for part in reversed(name.split(".")):
            if not part.isdigit():
                name = part
                break
    return name


def normalize_context(raw_context: dict[str, Any]) -> dict[str, Any]:
    normalized = dict(raw_context)
    for key, value in raw_context.items():
        clean_key = _normalize_field_name(key)
        if clean_key != key and clean_key not in normalized:
            normalized[clean_key] = value
            logger.debug(f"Normalized '{key}' -> '{clean_key}' = {value}")
    return normalized


def _flatten_values(value: Any, *, prefix: str = "") -> dict[str, Any]:
    flat: dict[str, Any] = {}
    if isinstance(value, dict):
        for key, child in value.items():
            child_prefix = f"{prefix}.{key}" if prefix else str(key)
            flat.update(_flatten_values(child, prefix=child_prefix))
        return flat
    if isinstance(value, list):
        for index, child in enumerate(value):
            child_prefix = f"{prefix}.{index}" if prefix else str(index)
            flat.update(_flatten_values(child, prefix=child_prefix))
        return flat
    if prefix:
        flat[prefix] = value
    return flat


def _lookup(context: dict[str, Any], field: str) -> Any:
    if field in context:
        return context[field]
    fallback = field.replace(".", "_")
    if fallback in context:
        return context[fallback]
    logger.debug(f"Field '{field}' not found in context")
    return None


def _to_float(value: Any) -> float | None:
    try:
        if value is None or value == "":
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _to_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return bool(value)


def _compare(left: Any, op: str, right: Any) -> bool:
    if op == "==":
        return left == right
    if op == "in":
        return left in right if isinstance(right, (list, tuple, set)) else False

    left_num = _to_float(left)
    right_num = _to_float(right)

    if left_num is None or right_num is None:
        return False

    if op == "<":
        return left_num < right_num
    elif op == "<=":
        return left_num <= right_num
    elif op == ">":
        return left_num > right_num
    elif op == ">=":
        return left_num >= right_num
    else:
        logger.warning(f"Unknown operator: '{op}'")
        return False


def _resolve_value(condition: dict[str, Any], context: dict[str, Any]) -> float | None:
    """
    Resolve threshold value — static or dynamically computed from history.
    Records observability metadata on the context for the resolved
    threshold (mode, sample size, fallback usage, reason).
    """
    static_value = condition.get("value")
    mode = condition.get("threshold_mode")

    if not mode or mode == "static":
        return static_value

    from alerts.services.threshold_engine import ThresholdEngine

    result = ThresholdEngine().compute_with_metadata(
        entity_id=str(context.get("entity_id", "")),
        blood_type=str(context.get("blood_type", "")),
        mode=mode,
        percentile=condition.get("threshold_percentile", 10),
        fallback=static_value,
    )

    # Stamp the most recent threshold computation metadata onto the context
    # so alert events carry it through to storage and the dashboard. We keep
    # the latest-evaluated threshold; for rules with multiple dynamic
    # conditions, only the last one is recorded here. Per-condition history
    # is preserved in threshold_metadata_history.
    metadata_dict = result.as_dict()
    metadata_dict["field"] = condition.get("field")
    context["threshold_metadata"] = metadata_dict
    context.setdefault("threshold_metadata_history", []).append(metadata_dict)

    return result.value


def _condition_matches(context: dict[str, Any], condition: dict[str, Any]) -> bool:
    field = str(condition.get("field", "")).strip()
    op = str(condition.get("op", "")).strip()

    if not field or not op:
        logger.warning(f"Invalid condition: missing field or op: {condition}")
        return False

    expected = _resolve_value(condition, context)

    if expected != condition.get("value"):
        context["threshold_value"] = expected
        logger.debug(
            "Dynamic threshold for %s/%s [%s]: %s",
            context.get("entity_id"),
            context.get("blood_type"),
            condition.get("threshold_mode"),
            expected,
        )

    actual = _lookup(context, field)
    result = _compare(actual, op, expected)
    logger.debug(f"Condition: {field}={actual} {op} {expected} => {result}")
    return result


def _conditions_match(context: dict[str, Any], conditions: dict[str, Any]) -> bool:
    if not conditions or not isinstance(conditions, dict):
        return False

    if isinstance(conditions.get("all"), list):
        cond_list = [c for c in conditions["all"] if isinstance(c, dict)]
        if not cond_list:
            return False
        return all(_condition_matches(context, c) for c in cond_list)

    if isinstance(conditions.get("any"), list):
        cond_list = [c for c in conditions["any"] if isinstance(c, dict)]
        if not cond_list:
            return False
        return any(_condition_matches(context, c) for c in cond_list)

    if "field" in conditions and "op" in conditions:
        return _condition_matches(context, conditions)

    logger.warning(f"Invalid conditions structure: {conditions}")
    return False


def _build_title(rule: AlertRule, context: dict[str, Any]) -> str:
    entity_id = context.get("entity_id") or context.get("hospital_id") or "target"
    blood_type = context.get("blood_type")
    suffix = f" for {entity_id}"
    if blood_type:
        suffix += f" ({blood_type})"
    return f"{rule.name}{suffix}"


def _build_message(rule: AlertRule, context: dict[str, Any]) -> str:
    predicted = context.get("predicted_value")
    threshold = context.get("threshold_value")
    current_stock = context.get("current_stock_units")
    parts = [rule.description.strip() or "Alert rule triggered."]
    if predicted is not None:
        parts.append(f"predicted={predicted}")
    if current_stock is not None:
        parts.append(f"current_stock={current_stock}")
    if threshold is not None:
        parts.append(f"threshold={threshold}")
    metadata = context.get("threshold_metadata")
    if isinstance(metadata, dict):
        mode = metadata.get("mode")
        n = metadata.get("n_samples")
        fallback_used = metadata.get("fallback_used")
        if fallback_used:
            parts.append(f"threshold_source=fallback ({metadata.get('reason', '')})")
        elif mode is not None and n is not None:
            parts.append(f"threshold_source={mode}(n={n})")
    return " | ".join(parts)


class AlertEngine:

    def build_context(
        self,
        *,
        model_id: str,
        prediction_result: dict[str, Any],
        feature_input: dict[str, Any],
        source: str,
    ) -> dict[str, Any]:
        context = {
            "model_id": model_id,
            "source": source,
            **feature_input,
            **_flatten_values(prediction_result),
        }

        execution_results = prediction_result.get("execution_results", [])
        if isinstance(execution_results, list) and execution_results:
            first = execution_results[0]
            if isinstance(first, dict):
                context.update(_flatten_values(first.get("output", {})))

        context.setdefault("entity_id", context.get("hospital_id") or context.get("center_id") or "")
        context.setdefault("blood_type", context.get("blood_type") or context.get("blood_group") or "")

        prediction = context.get("prediction")
        if prediction is not None:
            context.setdefault("predicted_value", prediction)
        else:
            context.setdefault("predicted_value", context.get("probability"))

        context = normalize_context(context)
        logger.debug(f"Context built: entity_id={context.get('entity_id')}, blood_type={context.get('blood_type')}, predicted_value={context.get('predicted_value')}")
        return context

    def evaluate_prediction(
        self,
        *,
        model_id: str,
        prediction_result: dict[str, Any],
        feature_input: dict[str, Any],
        source: str,
    ) -> list[AlertEvent]:
        context = self.build_context(
            model_id=model_id,
            prediction_result=prediction_result,
            feature_input=feature_input,
            source=source,
        )
        matched_events: list[AlertEvent] = []

        source_entity_id = context.get("entity_id", "")
        hospital_id = context.get("hospital_id", "")
        entity_id = hospital_id or source_entity_id
        if entity_id:
            context["entity_id"] = entity_id
        if source_entity_id and source_entity_id != entity_id:
            context["source_entity_id"] = source_entity_id
        blood_type = context.get("blood_type", "")
        if _to_bool(context.get("requires_hospital_scope")) and source_entity_id and not hospital_id:
            logger.warning(
                "Prediction context for source entity %s has no hospital_id; "
                "hospital-scoped alert rules will not be evaluated.",
                source_entity_id,
            )

        if entity_id and blood_type:
            declining, slope, confidence = TrendDetector().detect_declining_trend(
                str(entity_id),
                str(blood_type),
            )
            if declining:
                trend_context = {
                    **context,
                    "trend_slope": slope,
                    "trend_confidence": confidence,
                }
                trend_key = sha1(
                    "|".join(["trend_detection", str(entity_id), str(blood_type)]).encode(
                        "utf-8"
                    )
                ).hexdigest()
                trend_event, _ = create_or_refresh_event(
                    rule=None,
                    event_key=trend_key,
                    title=f"Declining stock trend for {entity_id} ({blood_type})",
                    message=(
                        "Recent predictions show a declining inventory trend "
                        f"(slope={slope:.4f}, confidence={confidence:.2f})."
                    ),
                    context=trend_context,
                    source_type="trend_detection",
                    source_ref=model_id,
                    entity_type=str(context.get("entity_type") or "hospital"),
                    entity_id=str(entity_id),
                    blood_type=str(blood_type),
                    predicted_value=_to_float(context.get("predicted_value")),
                    actual_value=_to_float(context.get("actual_value")),
                    threshold_value=None,
                )
                if trend_event is not None:
                    matched_events.append(trend_event)

        scope_filter = Q(scope_type="global") | Q(scope_type="blood_type", scope_ref=blood_type)
        if hospital_id:
            scope_filter |= Q(scope_type="hospital", scope_ref=hospital_id)

        rules = AlertRule.objects.filter(is_active=True).exclude(
            trigger_type=AlertTriggerType.STALE_ALERT
        ).filter(scope_filter)

        logger.info(f"Evaluating {rules.count()} active rules for {entity_id}/{blood_type}")

        for rule in rules:
            if rule.created_by in ("rule-factory", "trend-detector"):
                parts = rule.name.split("_")
                if len(parts) >= 2 and parts[1] != blood_type:
                    continue
            matches = _conditions_match(context, rule.conditions)
            
            if not matches:
                continue

            logger.info(f"✅ Rule '{rule.name}' MATCHED")

            # Resolve dynamic threshold for storage on the event
            computed_threshold = None
            conditions = rule.conditions
            if "field" in conditions and conditions.get("threshold_mode"):
                computed_threshold = _resolve_value(conditions, context)
            elif "all" in conditions:
                for c in conditions["all"]:
                    if c.get("field") == "predicted_value" and c.get("threshold_mode"):
                        computed_threshold = _resolve_value(c, context)
                        break

            raw_key = "|".join([
                str(rule.id),
                str(entity_id),
                str(blood_type),
                str(model_id),
            ])
            event_key = sha1(raw_key.encode("utf-8")).hexdigest()

            event, _ = create_or_refresh_event(
                rule=rule,
                event_key=event_key,
                title=_build_title(rule, context),
                message=_build_message(rule, context),
                context=context,
                source_type="prediction",
                source_ref=model_id,
                entity_type=str(context.get("entity_type") or ("hospital" if context.get("hospital_id") else "")),
                entity_id=str(entity_id),
                blood_type=str(blood_type),
                predicted_value=_to_float(context.get("predicted_value")),
                actual_value=_to_float(context.get("actual_value")),
                threshold_value=computed_threshold or _to_float(context.get("threshold_value")),
            )
            if event is not None:
                matched_events.append(event)

        logger.info(f"Evaluation complete: {len(matched_events)} alerts triggered")
        return matched_events

    def process_stale_alerts(self) -> int:
        now = timezone.now()
        processed = 0
        rules = AlertRule.objects.filter(is_active=True, trigger_type=AlertTriggerType.STALE_ALERT)

        logger.info(f"Processing {rules.count()} stale alert rules")

        for rule in rules:
            cutoff = now - timedelta(minutes=rule.escalation_after_minutes)
            events = AlertEvent.objects.filter(
                Q(status=AlertStatus.OPEN) | Q(status=AlertStatus.ACKNOWLEDGED),
                opened_at__lte=cutoff,
            )
            if rule.scope_ref:
                events = events.filter(entity_id=rule.scope_ref)
            for event in events:
                logger.info(f"Escalating stale event: {event.id}")
                escalate_event(event, note="Escalated by stale alert rule.")
                processed += 1

        logger.info(f"Processed {processed} stale alerts")
        return processed
