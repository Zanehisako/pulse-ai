import json

from django.contrib import admin
from django.contrib import messages
from django.template.response import TemplateResponse
from django.urls import path
from django.utils.html import format_html

from .models import MLModelConfig, OrchestratorVariant, PredictionLog
from .services.control_center import (
    get_control_status,
    parse_json_object,
    reload_model_config_control,
    run_direct_model_prediction,
    select_orchestrator_model,
    warmup_orchestrator,
)


@admin.register(MLModelConfig)
class MLModelConfigAdmin(admin.ModelAdmin):
    change_list_template = "admin/ml/change_list_with_control.html"
    list_display = ["status_icon", "model_id", "model_type", "feature_count", "is_active", "updated_at"]
    list_filter = ["model_type", "is_active"]
    search_fields = ["model_id", "description"]
    readonly_fields = ["created_at", "updated_at"]
    list_editable = ["is_active"]

    def get_urls(self):
        urls = super().get_urls()
        custom_urls = [
            path(
                "control-center/",
                self.admin_site.admin_view(self.control_center_view),
                name="ml_control_center",
            )
        ]
        return custom_urls + urls

    def control_center_view(self, request):
        action_result = None
        direct_payload_json = request.POST.get(
            "direct_payload_json",
            '{\n  "features": {}\n}',
        )
        selected_direct_model_id = request.POST.get("direct_model_id", "")
        selected_orchestrator_model_id = request.POST.get("orchestrator_model_id", "")

        if request.method == "POST":
            action = request.POST.get("action", "")
            if action == "reload_config":
                result = reload_model_config_control("admin-manual-reload")
                action_result = result.payload
                if result.status_code == 200:
                    messages.success(request, "Model configuration reloaded.")
                else:
                    messages.error(
                        request,
                        result.payload.get("error", "Model configuration reload failed."),
                    )
            elif action == "select_orchestrator":
                result = select_orchestrator_model(
                    selected_orchestrator_model_id,
                    download=request.POST.get("download") == "1",
                    wait=request.POST.get("wait") == "1",
                    reason="admin-orchestrator-selection-update",
                )
                action_result = result.payload
                if result.status_code == 200:
                    messages.success(request, "Orchestrator model selection applied.")
                else:
                    messages.error(
                        request,
                        result.payload.get("error", "Orchestrator selection failed."),
                    )
            elif action == "warmup_orchestrator":
                result = warmup_orchestrator(wait=request.POST.get("wait") == "1")
                action_result = result.payload
                if result.status_code == 200:
                    messages.success(request, "Orchestrator warmup requested.")
                else:
                    messages.error(
                        request,
                        result.payload.get("error", "Orchestrator warmup failed."),
                    )
            elif action == "direct_predict":
                try:
                    payload = parse_json_object(direct_payload_json)
                except (json.JSONDecodeError, ValueError) as exc:
                    action_result = {"error": str(exc)}
                    messages.error(request, str(exc))
                else:
                    result = run_direct_model_prediction(
                        selected_direct_model_id,
                        payload,
                        user=request.user,
                        alert_source="admin-direct-model",
                    )
                    action_result = result.payload
                    if result.status_code == 200:
                        messages.success(request, "Direct model prediction completed.")
                    else:
                        messages.error(
                            request,
                            result.payload.get("error", "Direct model prediction failed."),
                        )

        status_payload = get_control_status(refresh=True)
        models = status_payload.get("models", [])
        loaded_models = [row for row in models if row.get("status") == "loaded"]
        orchestrator_models = status_payload.get("orchestrator_models", [])
        context = {
            **self.admin_site.each_context(request),
            "title": "ML Control Center",
            "status_payload": status_payload,
            "models": models,
            "loaded_models": loaded_models,
            "orchestrator_models": orchestrator_models,
            "selected_direct_model_id": selected_direct_model_id,
            "selected_orchestrator_model_id": (
                selected_orchestrator_model_id
                or status_payload.get("selected_model_id")
                or ""
            ),
            "direct_payload_json": direct_payload_json,
            "action_result_json": json.dumps(action_result, indent=2, default=str)
            if action_result is not None
            else "",
        }
        return TemplateResponse(request, "admin/ml/control_center.html", context)

    @admin.display(description="")
    def status_icon(self, obj):
        color = "green" if obj.is_active else "red"
        return format_html(f'<span style="color:{color}; font-size:16px;">●</span>')

    @admin.display(description="Features")
    def feature_count(self, obj):
        n = len(obj.features) if isinstance(obj.features, list) else 0
        return f"{n} features"


@admin.register(PredictionLog)
class PredictionLogAdmin(admin.ModelAdmin):
    list_display = ["success_icon", "model_id_used", "short_query", "user", "planner_mode", "response_time_ms", "created_at"]
    list_filter = ["success", "planner_mode", "model_id_used"]
    search_fields = ["query", "model_id_used"]
    readonly_fields = ["user", "model_config", "model_id_used", "query", "features_input", "prediction_output", "planner_mode", "success", "response_time_ms", "created_at"]
    date_hierarchy = "created_at"
    list_per_page = 50

    @admin.display(description="")
    def success_icon(self, obj):
        return "✅" if obj.success else "❌"

    @admin.display(description="Query")
    def short_query(self, obj):
        q = obj.query or ""
        return (q[:60] + "…") if len(q) > 60 else q


@admin.register(OrchestratorVariant)
class OrchestratorVariantAdmin(admin.ModelAdmin):
    change_list_template = "admin/ml/change_list_with_control.html"
    list_display = ["selected_icon", "variant_id", "name", "filename", "size_display", "min_ram_gb", "is_selected", "updated_at"]
    list_filter = ["is_selected", "repo_id"]
    list_editable = ["is_selected"]
    search_fields = ["variant_id", "name", "filename"]
    readonly_fields = ["created_at", "updated_at"]

    @admin.display(description="")
    def selected_icon(self, obj):
        return "⭐" if obj.is_selected else ""

    @admin.display(description="Size")
    def size_display(self, obj):
        if not obj.size_mb:
            return "—"
        return f"{obj.size_mb / 1024:.1f} GB" if obj.size_mb >= 1000 else f"{obj.size_mb:.0f} MB"
