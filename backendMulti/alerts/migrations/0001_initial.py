from django.db import migrations, models
import django.db.models.deletion
import django.utils.timezone
import uuid


class Migration(migrations.Migration):
    initial = True

    dependencies = []

    operations = [
        migrations.CreateModel(
            name="AlertRule",
            fields=[
                ("id", models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name="ID")),
                ("name", models.CharField(max_length=255, unique=True)),
                ("description", models.TextField(blank=True, default="")),
                ("is_active", models.BooleanField(db_index=True, default=True)),
                ("scope_type", models.CharField(choices=[("global", "Global"), ("center", "Center"), ("hospital", "Hospital"), ("blood_type", "Blood Type")], default="global", max_length=32)),
                ("scope_ref", models.CharField(blank=True, default="", max_length=128)),
                ("severity_base", models.CharField(choices=[("pre_alert", "Pre Alert"), ("confirmed", "Confirmed"), ("critical", "Critical"), ("resolved", "Resolved")], default="pre_alert", max_length=32)),
                ("trigger_type", models.CharField(choices=[("threshold_gap", "Threshold Gap"), ("stock_and_forecast", "Stock And Forecast"), ("stale_alert", "Stale Alert"), ("manual", "Manual")], default="threshold_gap", max_length=32)),
                ("conditions", models.JSONField(blank=True, default=dict)),
                ("channels", models.JSONField(blank=True, default=list)),
                ("escalation_after_minutes", models.PositiveIntegerField(default=120)),
                ("dedup_window_minutes", models.PositiveIntegerField(default=60)),
                ("created_by", models.CharField(blank=True, default="system", max_length=100)),
                ("created_at", models.DateTimeField(auto_now_add=True)),
                ("updated_at", models.DateTimeField(auto_now=True)),
            ],
            options={"db_table": "alert_rule", "ordering": ["name"]},
        ),
        migrations.CreateModel(
            name="AlertEvent",
            fields=[
                ("id", models.UUIDField(default=uuid.uuid4, editable=False, primary_key=True, serialize=False)),
                ("source_type", models.CharField(db_index=True, default="system", max_length=64)),
                ("source_ref", models.CharField(blank=True, db_index=True, default="", max_length=128)),
                ("event_key", models.CharField(db_index=True, max_length=255)),
                ("severity", models.CharField(choices=[("pre_alert", "Pre Alert"), ("confirmed", "Confirmed"), ("critical", "Critical"), ("resolved", "Resolved")], db_index=True, default="pre_alert", max_length=32)),
                ("status", models.CharField(choices=[("open", "Open"), ("acknowledged", "Acknowledged"), ("escalated", "Escalated"), ("resolved", "Resolved"), ("muted", "Muted")], db_index=True, default="open", max_length=32)),
                ("title", models.CharField(max_length=255)),
                ("message", models.TextField()),
                ("context", models.JSONField(blank=True, default=dict)),
                ("entity_type", models.CharField(blank=True, db_index=True, default="", max_length=64)),
                ("entity_id", models.CharField(blank=True, db_index=True, default="", max_length=128)),
                ("blood_type", models.CharField(blank=True, db_index=True, default="", max_length=16)),
                ("predicted_value", models.FloatField(blank=True, null=True)),
                ("actual_value", models.FloatField(blank=True, null=True)),
                ("threshold_value", models.FloatField(blank=True, null=True)),
                ("opened_at", models.DateTimeField(db_index=True, default=django.utils.timezone.now)),
                ("last_evaluated_at", models.DateTimeField(db_index=True, default=django.utils.timezone.now)),
                ("acknowledged_at", models.DateTimeField(blank=True, null=True)),
                ("escalated_at", models.DateTimeField(blank=True, null=True)),
                ("resolved_at", models.DateTimeField(blank=True, null=True)),
                ("notification_id", models.UUIDField(blank=True, null=True)),
                ("rule", models.ForeignKey(blank=True, null=True, on_delete=django.db.models.deletion.SET_NULL, related_name="events", to="alerts.alertrule")),
            ],
            options={"db_table": "alert_event", "ordering": ["-opened_at"]},
        ),
        migrations.AddIndex(
            model_name="alertevent",
            index=models.Index(fields=["event_key", "status"], name="alert_event_key_status_idx"),
        ),
        migrations.AddIndex(
            model_name="alertevent",
            index=models.Index(fields=["severity", "status", "-opened_at"], name="alert_event_severity_status_opened_idx"),
        ),
        migrations.CreateModel(
            name="AlertEventHistory",
            fields=[
                ("id", models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name="ID")),
                ("action", models.CharField(max_length=64)),
                ("actor", models.CharField(blank=True, default="system", max_length=100)),
                ("note", models.TextField(blank=True, default="")),
                ("snapshot", models.JSONField(blank=True, default=dict)),
                ("created_at", models.DateTimeField(db_index=True, default=django.utils.timezone.now)),
                ("event", models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, related_name="history", to="alerts.alertevent")),
            ],
            options={"db_table": "alert_event_history", "ordering": ["-created_at"]},
        ),
    ]
