from django.conf import settings
from django.db import models
from django.utils import timezone


class MLModelConfig(models.Model):
    MODEL_TYPES = [
        ("sklearn", "Scikit-learn"),
        ("xgboost", "XGBoost"),
        ("custom", "Custom Linear"),
        ("online_agent", "Online Agent"),
        ("pytorch", "PyTorch"),
        ("mlflow", "MLflow Registry"),
        ("sota_joblib", "SOTA Joblib Artifact"),
    ]

    model_id = models.CharField(max_length=128, unique=True, db_index=True)
    description = models.TextField(blank=True, default="")
    file_path = models.CharField(max_length=512)
    model_type = models.CharField(max_length=64, choices=MODEL_TYPES, default="sklearn")
    features = models.JSONField(default=list)
    feature_info = models.JSONField(default=dict, blank=True)
    examples = models.JSONField(default=list, blank=True)
    defaults = models.JSONField(default=dict, blank=True)
    is_active = models.BooleanField(default=True, db_index=True)
    config_source_path = models.CharField(
        max_length=1024,
        blank=True,
        default="",
        db_index=True,
    )
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        db_table = "ml_model_config"
        verbose_name = "ML Model"
        verbose_name_plural = "ML Models"
        ordering = ["model_id"]

    def __str__(self):
        icon = "✅" if self.is_active else "❌"
        return f"{icon} {self.model_id} ({self.model_type})"

    def to_registry_dict(self) -> dict:
        return {
            "id": self.model_id,
            "description": self.description,
            "file_path": self.file_path,
            "type": self.model_type,
            "features": self.features,
            "feature_info": self.feature_info,
            "examples": self.examples,
            "defaults": self.defaults,
        }


class PredictionLog(models.Model):
    user = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        on_delete=models.SET_NULL,
        null=True,
        blank=True,
    )
    model_config = models.ForeignKey(
        MLModelConfig,
        on_delete=models.SET_NULL,
        null=True,
        blank=True,
    )
    model_id_used = models.CharField(max_length=128, blank=True, db_index=True)
    query = models.TextField(blank=True)
    features_input = models.JSONField(default=dict)
    prediction_output = models.JSONField(default=dict)
    planner_mode = models.CharField(max_length=32, blank=True)
    success = models.BooleanField(default=True, db_index=True)
    response_time_ms = models.FloatField(null=True)
    created_at = models.DateTimeField(default=timezone.now, db_index=True)

    class Meta:
        db_table = "ml_prediction_log"
        verbose_name = "Prediction Log"
        verbose_name_plural = "Prediction Logs"
        ordering = ["-created_at"]
        indexes = [
            models.Index(fields=["-created_at", "success"]),
            models.Index(fields=["model_id_used", "-created_at"]),
        ]

    def __str__(self):
        icon = "✅" if self.success else "❌"
        return f"{icon} {self.model_id_used} — {self.created_at:%Y-%m-%d %H:%M}"


class OrchestratorVariant(models.Model):
    variant_id = models.CharField(max_length=128, unique=True, db_index=True)
    name = models.CharField(max_length=256)
    description = models.TextField(blank=True, default="")
    repo_id = models.CharField(max_length=256, default="bartowski/xLAM-7b-fc-r-GGUF")
    filename = models.CharField(max_length=256)
    size_mb = models.FloatField(null=True, blank=True)
    size_bytes = models.BigIntegerField(null=True, blank=True)
    min_ram_gb = models.FloatField(default=0.0)
    min_vram_mb = models.IntegerField(default=0)
    n_ctx = models.IntegerField(default=4096)
    n_batch = models.IntegerField(default=128)
    is_selected = models.BooleanField(default=False, db_index=True)
    metadata_extra = models.JSONField(default=dict, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        db_table = "ml_orchestrator_variant"
        verbose_name = "Orchestrator Model"
        verbose_name_plural = "Orchestrator Models"
        ordering = ["-size_mb"]

    def __str__(self):
        icon = "⭐" if self.is_selected else "  "
        return f"{icon} {self.name} ({self.size_mb or '?'} MB)"

    def save(self, *args, **kwargs):
        if self.is_selected:
            OrchestratorVariant.objects.filter(is_selected=True).exclude(
                pk=self.pk
            ).update(is_selected=False)
        super().save(*args, **kwargs)


class ModelStats(models.Model):
    model_id = models.CharField(max_length=128, unique=True, db_index=True)
    identifier = models.CharField(max_length=128, unique=True, db_index=True)
    stats = models.JSONField(default=dict)
    made_at = models.DateTimeField(default=timezone.now, db_index=True)

    class Meta:
        db_table = "models_stats"
        ordering = ["-made_at"]

    def __str__(self):
        return f"{self.model_id} - {self.identifier} - {self.made_at:%Y-%m-%d %H:%M}"


class DriftReport(models.Model):
    """Stores drift detection results for each model."""

    SEVERITY_CHOICES = [
        ("none", "No Drift"),
        ("warning", "Warning"),
        ("critical", "Critical"),
    ]

    model_id = models.CharField(max_length=128, db_index=True)
    mlflow_run_id = models.CharField(max_length=64, blank=True, default="")
    drift_detected = models.BooleanField(default=False, db_index=True)
    drift_score = models.FloatField(
        default=0.0, help_text="Fraction of features drifted (0.0–1.0)"
    )
    severity = models.CharField(max_length=16, choices=SEVERITY_CHOICES, default="none")
    features_checked = models.IntegerField(default=0)
    features_drifted = models.IntegerField(default=0)
    feature_details = models.JSONField(
        default=dict, blank=True, help_text="Full drift report per feature"
    )
    reference_size = models.IntegerField(default=0)
    current_size = models.IntegerField(default=0)
    check_type = models.CharField(
        max_length=32,
        default="scheduled",
        help_text="scheduled, manual, or post_training",
    )
    checked_at = models.DateTimeField(default=timezone.now, db_index=True)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        db_table = "ml_drift_report"
        verbose_name = "Drift Report"
        verbose_name_plural = "Drift Reports"
        ordering = ["-checked_at"]
        indexes = [
            models.Index(fields=["model_id", "-checked_at"]),
            models.Index(fields=["-checked_at", "drift_detected"]),
        ]

    def __str__(self):
        icon = "🔴" if self.drift_detected else "🟢"
        return f"{icon} {self.model_id} — {self.severity} ({self.drift_score:.1%}) @ {self.checked_at:%Y-%m-%d %H:%M}"
