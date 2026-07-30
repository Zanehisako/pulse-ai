from __future__ import annotations

import uuid

from django.db import models
from django.utils import timezone


class TwinRun(models.Model):
    class Status(models.TextChoices):
        CREATED = "created", "Created"
        RUNNING = "running", "Running"
        PAUSED = "paused", "Paused"
        STOPPED = "stopped", "Stopped"
        FAILED = "failed", "Failed"

    class SourceMode(models.TextChoices):
        LIVE = "live", "Live"
        SYNTHETIC = "synthetic", "Synthetic"

    run_id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    mode = models.CharField(max_length=64, default="operational_shadow", db_index=True)
    status = models.CharField(max_length=32, choices=Status.choices, default=Status.CREATED, db_index=True)
    config_version = models.CharField(max_length=64, db_index=True)
    scenario_key = models.CharField(max_length=128, default="baseline")
    strategy_key = models.CharField(max_length=128, default="baseline")
    source_mode = models.CharField(max_length=32, choices=SourceMode.choices, default=SourceMode.LIVE, db_index=True)
    seed = models.PositiveIntegerField(default=100)
    created_by = models.CharField(max_length=255, blank=True, default="system")
    started_at = models.DateTimeField(default=timezone.now, db_index=True)
    stopped_at = models.DateTimeField(null=True, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        db_table = "digital_twin_run"
        ordering = ["-started_at"]

    def __str__(self) -> str:
        return f"{self.run_id} ({self.status})"


class TwinSnapshot(models.Model):
    snapshot_id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    run = models.ForeignKey(TwinRun, on_delete=models.CASCADE, related_name="snapshots")
    source = models.CharField(max_length=32, db_index=True)
    captured_at = models.DateTimeField(default=timezone.now, db_index=True)
    source_watermarks = models.JSONField(default=dict, blank=True)
    normalized_payload = models.JSONField(default=dict)
    redacted_payload = models.JSONField(default=dict)
    freshness = models.JSONField(default=dict, blank=True)
    warnings = models.JSONField(default=list, blank=True)

    class Meta:
        db_table = "digital_twin_snapshot"
        ordering = ["-captured_at"]
        indexes = [
            models.Index(fields=["run", "-captured_at"]),
            models.Index(fields=["source", "-captured_at"]),
        ]


class TwinFrame(models.Model):
    frame_id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    run = models.ForeignKey(TwinRun, on_delete=models.CASCADE, related_name="frames")
    tick_number = models.PositiveIntegerField()
    simulated_hour = models.FloatField(default=0.0)
    real_snapshot = models.ForeignKey(TwinSnapshot, on_delete=models.SET_NULL, null=True, blank=True, related_name="frames")
    simulated_state = models.JSONField(default=dict)
    divergence = models.JSONField(default=dict, blank=True)
    active_events = models.JSONField(default=list, blank=True)
    active_actions = models.JSONField(default=list, blank=True)
    created_at = models.DateTimeField(default=timezone.now, db_index=True)

    class Meta:
        db_table = "digital_twin_frame"
        ordering = ["-tick_number"]
        unique_together = ["run", "tick_number"]
        indexes = [
            models.Index(fields=["run", "-tick_number"]),
        ]


class TwinEvent(models.Model):
    class Status(models.TextChoices):
        ACTIVE = "active", "Active"
        EXPIRED = "expired", "Expired"
        CANCELLED = "cancelled", "Cancelled"

    event_id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    run = models.ForeignKey(TwinRun, on_delete=models.CASCADE, related_name="events")
    event_key = models.CharField(max_length=128, db_index=True)
    source = models.CharField(max_length=64, default="manual", db_index=True)
    severity = models.FloatField(default=0.5)
    start_hour = models.FloatField(default=0.0)
    end_hour = models.FloatField(default=0.0)
    config_version = models.CharField(max_length=64, db_index=True)
    payload = models.JSONField(default=dict, blank=True)
    applied_effects = models.JSONField(default=dict, blank=True)
    status = models.CharField(max_length=32, choices=Status.choices, default=Status.ACTIVE, db_index=True)
    created_at = models.DateTimeField(default=timezone.now, db_index=True)

    class Meta:
        db_table = "digital_twin_event"
        ordering = ["-created_at"]


class TwinAction(models.Model):
    class Status(models.TextChoices):
        SIMULATED = "simulated", "Simulated"
        REJECTED = "rejected", "Rejected"
        PROMOTED = "promoted", "Promoted"

    class PromotionStatus(models.TextChoices):
        DISABLED = "disabled", "Disabled"
        NOT_ELIGIBLE = "not_eligible", "Not Eligible"
        PENDING = "pending", "Pending"
        PROMOTED = "promoted", "Promoted"

    action_id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    run = models.ForeignKey(TwinRun, on_delete=models.CASCADE, related_name="actions")
    action_key = models.CharField(max_length=128, db_index=True)
    source = models.CharField(max_length=64, default="manual", db_index=True)
    requested_by = models.CharField(max_length=255, blank=True, default="system")
    status = models.CharField(max_length=32, choices=Status.choices, default=Status.SIMULATED, db_index=True)
    simulated_effect = models.JSONField(default=dict, blank=True)
    audit_metadata = models.JSONField(default=dict, blank=True)
    promotion_status = models.CharField(max_length=32, choices=PromotionStatus.choices, default=PromotionStatus.DISABLED, db_index=True)
    created_at = models.DateTimeField(default=timezone.now, db_index=True)
    promoted_at = models.DateTimeField(null=True, blank=True)

    class Meta:
        db_table = "digital_twin_action"
        ordering = ["-created_at"]


class TwinBranch(models.Model):
    branch_id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    run = models.ForeignKey(TwinRun, on_delete=models.CASCADE, related_name="branches")
    branch_key = models.CharField(max_length=128, db_index=True)
    parent_frame = models.ForeignKey(TwinFrame, on_delete=models.SET_NULL, null=True, blank=True, related_name="branches")
    policy_overrides = models.JSONField(default=dict, blank=True)
    action_overrides = models.JSONField(default=list, blank=True)
    event_overrides = models.JSONField(default=list, blank=True)
    summary_metrics = models.JSONField(default=dict, blank=True)
    confidence_bands = models.JSONField(default=list, blank=True)
    created_at = models.DateTimeField(default=timezone.now, db_index=True)

    class Meta:
        db_table = "digital_twin_branch"
        ordering = ["-created_at"]
        unique_together = ["run", "branch_key"]


class TwinRecommendation(models.Model):
    recommendation_id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    run = models.ForeignKey(TwinRun, on_delete=models.CASCADE, related_name="recommendations")
    branch = models.ForeignKey(TwinBranch, on_delete=models.SET_NULL, null=True, blank=True, related_name="recommendations")
    recommendation_key = models.CharField(max_length=128, db_index=True)
    rationale = models.TextField()
    expected_impact = models.JSONField(default=dict, blank=True)
    risk_flags = models.JSONField(default=list, blank=True)
    source_tool_ids = models.JSONField(default=list, blank=True)
    source_model_ids = models.JSONField(default=list, blank=True)
    created_at = models.DateTimeField(default=timezone.now, db_index=True)

    class Meta:
        db_table = "digital_twin_recommendation"
        ordering = ["-created_at"]


class TwinAuditLog(models.Model):
    audit_id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    run = models.ForeignKey(TwinRun, on_delete=models.CASCADE, null=True, blank=True, related_name="audit_logs")
    actor = models.CharField(max_length=255, blank=True, default="system")
    action = models.CharField(max_length=128, db_index=True)
    config_version = models.CharField(max_length=64, blank=True, default="", db_index=True)
    source_snapshot_id = models.CharField(max_length=64, blank=True, default="")
    metadata = models.JSONField(default=dict, blank=True)
    created_at = models.DateTimeField(default=timezone.now, db_index=True)

    class Meta:
        db_table = "digital_twin_audit_log"
        ordering = ["-created_at"]
        indexes = [
            models.Index(fields=["run", "-created_at"]),
            models.Index(fields=["action", "-created_at"]),
        ]
