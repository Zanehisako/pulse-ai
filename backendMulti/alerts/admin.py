from django.contrib import admin

from .models import AlertEvent, AlertEventHistory, AlertRule


@admin.register(AlertRule)
class AlertRuleAdmin(admin.ModelAdmin):
    list_display = ("name", "trigger_type", "severity_base", "is_active", "updated_at")
    list_filter = ("is_active", "trigger_type", "severity_base", "scope_type")
    search_fields = ("name", "description", "scope_ref")


@admin.register(AlertEvent)
class AlertEventAdmin(admin.ModelAdmin):
    list_display = ("title", "severity", "status", "entity_type", "entity_id", "opened_at")
    list_filter = ("severity", "status", "source_type", "entity_type", "blood_type")
    search_fields = ("title", "message", "entity_id", "source_ref", "event_key")
    readonly_fields = ("opened_at", "last_evaluated_at", "acknowledged_at", "escalated_at", "resolved_at")


@admin.register(AlertEventHistory)
class AlertEventHistoryAdmin(admin.ModelAdmin):
    list_display = ("event", "action", "actor", "created_at")
    list_filter = ("action", "actor")
    search_fields = ("event__title", "note", "actor")
