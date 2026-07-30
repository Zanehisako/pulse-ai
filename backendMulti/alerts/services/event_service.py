from __future__ import annotations

import json
import logging
from datetime import timedelta
from pathlib import Path

from django.conf import settings
from django.db import transaction
from django.utils import timezone

from alerts.models import AlertEvent, AlertEventHistory, AlertSeverity, AlertStatus
from alerts.services.delivery_service import notify_clients

logger = logging.getLogger(__name__)


class InvalidTransitionError(ValueError):
    pass


def _alert_caps() -> dict[str, int]:
    """Return per-severity caps from simulation_config.json, with safe defaults."""
    try:
        config_path = Path(settings.ML_CONFIG_DIR) / "simulation_config.json"
        payload = json.loads(config_path.read_text(encoding="utf-8"))
        alerts_cfg = payload.get("alerts", {})
        return {
            AlertSeverity.CRITICAL: int(alerts_cfg.get("max_open_critical", 2)),
            AlertSeverity.WARNING: int(alerts_cfg.get("max_open_warning", 4)),
        }
    except Exception as exc:
        logger.warning("Could not read alert caps from config: %s — using defaults", exc)
        return {AlertSeverity.CRITICAL: 2, AlertSeverity.WARNING: 4}


def _open_count(severity: str) -> int:
    return AlertEvent.objects.filter(
        severity=severity,
        status__in=[AlertStatus.OPEN, AlertStatus.ACKNOWLEDGED, AlertStatus.ESCALATED],
    ).count()

def record_history(event: AlertEvent, action: str, *, actor: str = "system", note: str = "") -> None:
    AlertEventHistory.objects.create(
        event=event,
        action=action,
        actor=actor,
        note=note,
        snapshot=event.to_dict(),
    )


@transaction.atomic
def create_or_refresh_event(
    *,
    rule,
    event_key: str,
    title: str,
    message: str,
    context: dict,
    source_type: str,
    source_ref: str,
    entity_type: str = "",
    entity_id: str = "",
    blood_type: str = "",
    predicted_value: float | None = None,
    actual_value: float | None = None,
    threshold_value: float | None = None,
):
    now = timezone.now()
    dedup_minutes = rule.dedup_window_minutes if rule else 60
    dedup_cutoff = now - timedelta(minutes=dedup_minutes)
    event = (
        AlertEvent.objects.select_for_update()
        .filter(event_key=event_key, opened_at__gte=dedup_cutoff)
        .exclude(status=AlertStatus.RESOLVED)
        .order_by("-opened_at")
        .first()
    )

    if event is None:
        severity = rule.severity_base if rule else AlertSeverity.WARNING
        caps = _alert_caps()
        cap = caps.get(severity)
        if cap is not None and _open_count(severity) >= cap:
            logger.debug(
                "Alert cap reached for severity=%s (cap=%d) — skipping new event for key=%s",
                severity,
                cap,
                event_key,
            )
            return None, False

        event = AlertEvent.objects.create(
            rule=rule,
            source_type=source_type,
            source_ref=source_ref,
            event_key=event_key,
            severity=rule.severity_base if rule else AlertSeverity.WARNING,
            status=AlertStatus.OPEN,
            title=title,
            message=message,
            context=context,
            entity_type=entity_type,
            entity_id=entity_id,
            blood_type=blood_type,
            predicted_value=predicted_value,
            actual_value=actual_value,
            threshold_value=threshold_value,
            opened_at=now,
            last_evaluated_at=now,
        )
        record_history(event, "created")
        notify_clients(event)
        return event, True

    event.title = title
    event.message = message
    event.context = context
    event.last_evaluated_at = now
    event.predicted_value = predicted_value
    event.actual_value = actual_value
    event.threshold_value = threshold_value
    event.entity_type = entity_type
    event.entity_id = entity_id
    event.blood_type = blood_type
    event.source_type = source_type
    event.source_ref = source_ref
    event.save(
        update_fields=[
            "title",
            "message",
            "context",
            "last_evaluated_at",
            "predicted_value",
            "actual_value",
            "threshold_value",
            "entity_type",
            "entity_id",
            "blood_type",
            "source_type",
            "source_ref",
        ]
    )
    record_history(event, "refreshed")
    notify_clients(event)
    return event, False

VALID_TRANSITIONS = {
    "acknowledge": [AlertStatus.OPEN, AlertStatus.ESCALATED],
    "resolve": [AlertStatus.OPEN, AlertStatus.ACKNOWLEDGED, AlertStatus.ESCALATED],
    "escalate": [AlertStatus.OPEN, AlertStatus.ACKNOWLEDGED],
}

def acknowledge_event(event: AlertEvent, *, actor: str = "system", note: str = "") -> AlertEvent:
    if event.status not in VALID_TRANSITIONS["acknowledge"]:
        raise InvalidTransitionError(
            f"Cannot acknowledge event in status '{event.status}'."
        )
    event.status = AlertStatus.ACKNOWLEDGED
    event.acknowledged_at = timezone.now()
    event.save(update_fields=["status", "acknowledged_at"])
    record_history(event, "acknowledged", actor=actor, note=note)
    notify_clients(event)
    return event


def resolve_event(event: AlertEvent, *, actor: str = "system", note: str = "") -> AlertEvent:
    if event.status not in VALID_TRANSITIONS["resolve"]:
        raise InvalidTransitionError(
            f"Cannot resolve event in status '{event.status}'."
        )
    event.status = AlertStatus.RESOLVED
    event.resolved_at = timezone.now()
    event.save(update_fields=["status", "resolved_at"])
    record_history(event, "resolved", actor=actor, note=note)
    notify_clients(event)
    return event


def escalate_event(event: AlertEvent, *, actor: str = "system", note: str = "") -> AlertEvent:
    if event.status not in VALID_TRANSITIONS["escalate"]:
        raise InvalidTransitionError(
            f"Cannot escalate event in status '{event.status}'."
        )
    event.status = AlertStatus.ESCALATED
    event.severity = AlertSeverity.CRITICAL
    event.escalated_at = timezone.now()
    event.save(update_fields=["status", "severity", "escalated_at"])
    record_history(event, "escalated", actor=actor, note=note)
    notify_clients(event)
    return event
