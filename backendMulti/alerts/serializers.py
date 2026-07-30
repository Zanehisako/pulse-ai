from rest_framework import serializers

from .models import AlertEvent, AlertRule, AlertSeverity, AlertStatus


class AlertRuleSerializer(serializers.ModelSerializer):
    name = serializers.CharField(validators=[])

    class Meta:
        model = AlertRule
        fields = [
            "id",
            "name",
            "description",
            "is_active",
            "scope_type",
            "scope_ref",
            "severity_base",
            "trigger_type",
            "conditions",
            "channels",
            "escalation_after_minutes",
            "dedup_window_minutes",
            "created_by",
            "created_at",
            "updated_at",
        ]
        read_only_fields = ["id", "created_at", "updated_at"]

    def validate_conditions(self, value):
        if not isinstance(value, dict) or not value:
            raise serializers.ValidationError("Conditions must be a non-empty object.")

        self._validate_condition_group(value)
        return value

    def _validate_condition_group(self, value):
        if "all" in value or "any" in value:
            operator = "all" if "all" in value else "any"
            items = value.get(operator)
            if not isinstance(items, list) or not items:
                raise serializers.ValidationError(
                    f"Conditions '{operator}' must be a non-empty list."
                )
            for item in items:
                if not isinstance(item, dict):
                    raise serializers.ValidationError("Nested conditions must be objects.")
                self._validate_condition_group(item)
            return

        field = str(value.get("field", "")).strip()
        op = str(value.get("op", "")).strip()
        if not field or not op:
            raise serializers.ValidationError(
                "Single conditions must include non-empty 'field' and 'op' values."
            )


class AlertEventSerializer(serializers.ModelSerializer):
    class Meta:
        model = AlertEvent
        fields = [
            "id",
            "rule",
            "source_type",
            "source_ref",
            "event_key",
            "severity",
            "status",
            "title",
            "message",
            "context",
            "entity_type",
            "entity_id",
            "blood_type",
            "predicted_value",
            "actual_value",
            "threshold_value",
            "opened_at",
            "last_evaluated_at",
            "acknowledged_at",
            "escalated_at",
            "resolved_at",
            "notification_id",
        ]
        read_only_fields = fields


class AlertSummarySerializer(serializers.Serializer):
    open = serializers.IntegerField()
    acknowledged = serializers.IntegerField()
    escalated = serializers.IntegerField()
    resolved = serializers.IntegerField()
    critical = serializers.IntegerField()
    warning = serializers.IntegerField()


class AlertActionSerializer(serializers.Serializer):
    note = serializers.CharField(required=False, allow_blank=True, default="")


class AlertCreateTestSerializer(serializers.Serializer):
    title = serializers.CharField(max_length=255)
    message = serializers.CharField()
    severity = serializers.ChoiceField(choices=AlertSeverity.choices, default=AlertSeverity.WARNING)
    status = serializers.ChoiceField(choices=AlertStatus.choices, default=AlertStatus.OPEN)
    entity_type = serializers.CharField(required=False, allow_blank=True, default="")
    entity_id = serializers.CharField(required=False, allow_blank=True, default="")
    blood_type = serializers.CharField(required=False, allow_blank=True, default="")
    context = serializers.JSONField(required=False, default=dict)


class EmptySerializer(serializers.Serializer):
    pass
