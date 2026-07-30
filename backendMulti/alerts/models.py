from __future__ import annotations

import uuid

from django.db import models
from django.utils import timezone


class AlertSeverity(models.TextChoices):
    WARNING = "warning", "Warning"
    CRITICAL = "critical", "Critical"
    RESOLVED = "resolved", "Resolved"


class AlertStatus(models.TextChoices):
    OPEN = "open", "Open"
    ACKNOWLEDGED = "acknowledged", "Acknowledged"
    ESCALATED = "escalated", "Escalated"
    RESOLVED = "resolved", "Resolved"
    MUTED = "muted", "Muted"


class AlertScopeType(models.TextChoices):
    GLOBAL = "global", "Global"
    CENTER = "center", "Center"
    HOSPITAL = "hospital", "Hospital"
    BLOOD_TYPE = "blood_type", "Blood Type"


class AlertTriggerType(models.TextChoices):
    THRESHOLD_GAP = "threshold_gap", "Threshold Gap"
    STOCK_AND_FORECAST = "stock_and_forecast", "Stock And Forecast"
    STALE_ALERT = "stale_alert", "Stale Alert"
    MANUAL = "manual", "Manual"


class AlertRule(models.Model):
    name = models.CharField(max_length=255, unique=True)
    description = models.TextField(blank=True, default="")
    is_active = models.BooleanField(default=True, db_index=True)
    scope_type = models.CharField(max_length=32, choices=AlertScopeType.choices, default=AlertScopeType.GLOBAL)
    scope_ref = models.CharField(max_length=128, blank=True, default="")
    severity_base = models.CharField(max_length=32, choices=AlertSeverity.choices, default=AlertSeverity.WARNING)
    trigger_type = models.CharField(max_length=32, choices=AlertTriggerType.choices, default=AlertTriggerType.THRESHOLD_GAP)
    conditions = models.JSONField(default=dict, blank=True)
    channels = models.JSONField(default=list, blank=True)
    escalation_after_minutes = models.PositiveIntegerField(default=120)
    dedup_window_minutes = models.PositiveIntegerField(default=60)
    created_by = models.CharField(max_length=100, blank=True, default="system")
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        db_table = "alert_rule"
        ordering = ["name"]

    def __str__(self) -> str:
        state = "active" if self.is_active else "inactive"
        return f"{self.name} ({state})"


class AlertEvent(models.Model):
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    rule = models.ForeignKey(AlertRule, on_delete=models.SET_NULL, null=True, blank=True, related_name="events")
    source_type = models.CharField(max_length=64, default="system", db_index=True)
    source_ref = models.CharField(max_length=128, blank=True, default="", db_index=True)
    event_key = models.CharField(max_length=255, db_index=True)
    severity = models.CharField(max_length=32, choices=AlertSeverity.choices, default=AlertSeverity.WARNING, db_index=True)
    status = models.CharField(max_length=32, choices=AlertStatus.choices, default=AlertStatus.OPEN, db_index=True)
    title = models.CharField(max_length=255)
    message = models.TextField()
    context = models.JSONField(default=dict, blank=True)
    entity_type = models.CharField(max_length=64, blank=True, default="", db_index=True)
    entity_id = models.CharField(max_length=128, blank=True, default="", db_index=True)
    blood_type = models.CharField(max_length=16, blank=True, default="", db_index=True)
    predicted_value = models.FloatField(null=True, blank=True)
    actual_value = models.FloatField(null=True, blank=True)
    threshold_value = models.FloatField(null=True, blank=True)
    opened_at = models.DateTimeField(default=timezone.now, db_index=True)
    last_evaluated_at = models.DateTimeField(default=timezone.now, db_index=True)
    acknowledged_at = models.DateTimeField(null=True, blank=True)
    escalated_at = models.DateTimeField(null=True, blank=True)
    resolved_at = models.DateTimeField(null=True, blank=True)
    notification_id = models.UUIDField(null=True, blank=True)

    class Meta:
        db_table = "alert_event"
        ordering = ["-opened_at"]
        indexes = [
            models.Index(fields=["event_key", "status"]),
            models.Index(fields=["severity", "status", "-opened_at"]),
        ]

    def __str__(self) -> str:
        return f"[{self.severity}] {self.title}"

    def to_dict(self) -> dict:
        return {
            "id": str(self.id),
            "rule_id": self.rule_id,
            "source_type": self.source_type,
            "source_ref": self.source_ref,
            "event_key": self.event_key,
            "severity": self.severity,
            "status": self.status,
            "title": self.title,
            "message": self.message,
            "context": self.context,
            "entity_type": self.entity_type,
            "entity_id": self.entity_id,
            "blood_type": self.blood_type,
            "predicted_value": self.predicted_value,
            "actual_value": self.actual_value,
            "threshold_value": self.threshold_value,
            "opened_at": self.opened_at.isoformat() if self.opened_at else None,
            "last_evaluated_at": self.last_evaluated_at.isoformat() if self.last_evaluated_at else None,
            "acknowledged_at": self.acknowledged_at.isoformat() if self.acknowledged_at else None,
            "escalated_at": self.escalated_at.isoformat() if self.escalated_at else None,
            "resolved_at": self.resolved_at.isoformat() if self.resolved_at else None,
            "notification_id": str(self.notification_id) if self.notification_id else None,
        }


class AlertEventHistory(models.Model):
    event = models.ForeignKey(AlertEvent, on_delete=models.CASCADE, related_name="history")
    action = models.CharField(max_length=64)
    actor = models.CharField(max_length=100, blank=True, default="system")
    note = models.TextField(blank=True, default="")
    snapshot = models.JSONField(default=dict, blank=True)
    created_at = models.DateTimeField(default=timezone.now, db_index=True)

    class Meta:
        db_table = "alert_event_history"
        ordering = ["-created_at"]

    def __str__(self) -> str:
        return f"{self.event_id} - {self.action}"

