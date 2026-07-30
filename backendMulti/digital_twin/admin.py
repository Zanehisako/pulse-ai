from django.contrib import admin

from .models import (
    TwinAction,
    TwinAuditLog,
    TwinBranch,
    TwinEvent,
    TwinFrame,
    TwinRecommendation,
    TwinRun,
    TwinSnapshot,
)


@admin.register(TwinRun)
class TwinRunAdmin(admin.ModelAdmin):
    list_display = ("run_id", "status", "source_mode", "scenario_key", "strategy_key", "config_version", "started_at")
    list_filter = ("status", "source_mode", "config_version")
    search_fields = ("run_id", "scenario_key", "strategy_key", "created_by")


admin.site.register(TwinSnapshot)
admin.site.register(TwinFrame)
admin.site.register(TwinEvent)
admin.site.register(TwinAction)
admin.site.register(TwinBranch)
admin.site.register(TwinRecommendation)
admin.site.register(TwinAuditLog)
