from django.urls import path

from .api import views

app_name = "ml"

urlpatterns = [
    # Config / Health
    path("", views.MLIndexView.as_view(), name="index"),
    path("health/", views.HealthView.as_view(), name="health"),
    path("models/", views.ModelListView.as_view(), name="model-list"),
    path("models/upload/", views.ModelUploadView.as_view(), name="model-upload"),
    path(
        "models/<str:model_id>/", views.ModelDetailView.as_view(), name="model-detail"
    ),
    path(
        "models/<str:model_id>/remove",
        views.RemoveModelView.as_view(),
        name="remove-model",
    ),
    path("models_stats/", views.ModelsStatsListView.as_view(), name="models-stats"),
    path(
        "models_stats/<str:model_id>/",
        views.ModelsStatsView.as_view(),
        name="single-model-stats",
    ),
    path(
        "models_stats/<str:model_id>/remove",
        views.RemoveModelsStatsView.as_view(),
        name="remove-model-stats",
    ),
    path("config/reload/", views.ConfigReloadView.as_view(), name="config-reload"),
    path("config/runtime/", views.RuntimeConfigView.as_view(), name="config-runtime"),
    path(
        "config/runtime/reload/",
        views.ConfigReloadView.as_view(),
        name="config-runtime-reload",
    ),
    path("control/status/", views.ControlStatusView.as_view(), name="control-status"),
    # Prediction
    path("predict/nl/", views.PredictNLView.as_view(), name="predict-nl"),
    path(
        "predict/stockout-hybrid/",
        views.HybridStockoutPredictView.as_view(),
        name="predict-stockout-hybrid",
    ),
    path("tools/", views.ToolListView.as_view(), name="tool-list"),
    path("tools/<str:tool_id>/run/", views.ToolRunView.as_view(), name="tool-run"),
    path(
        "models/<str:model_id>/predict/",
        views.PredictModelView.as_view(),
        name="predict-model",
    ),
    # Orchestrator
    path(
        "orchestrator/status/",
        views.OrchestratorStatusView.as_view(),
        name="orchestrator-status",
    ),
    path(
        "orchestrator/models/",
        views.OrchestratorModelsView.as_view(),
        name="orchestrator-models",
    ),
    path(
        "orchestrator/models/selected/",
        views.OrchestratorSelectView.as_view(),
        name="orchestrator-select",
    ),
    path(
        "orchestrator/models/<str:model_id>/download/",
        views.OrchestratorDownloadView.as_view(),
        name="orchestrator-download",
    ),
    path(
        "orchestrator/warmup/",
        views.OrchestratorWarmupView.as_view(),
        name="orchestrator-warmup",
    ),
    path(
        "chat/",
        views.OrchestratorChatView.as_view(),
        name="chat",
    ),
    # Training Scheduler
    path(
        "scheduler/status/",
        views.SchedulerStatusView.as_view(),
        name="scheduler-status",
    ),
    path(
        "scheduler/reload/",
        views.SchedulerReloadView.as_view(),
        name="scheduler-reload",
    ),
    path("scheduler/run/", views.SchedulerRunView.as_view(), name="scheduler-run"),
    path(
        "scheduler/run/status/",
        views.SchedulerRunStatusView.as_view(),
        name="scheduler-run-status",
    ),
    path(
        "scheduler/config/",
        views.SchedulerConfigView.as_view(),
        name="scheduler-config",
    ),
    # Drift Monitoring
    path("drift/status/", views.DriftStatusView.as_view(), name="drift-status"),
    path("drift/reports/", views.DriftReportsView.as_view(), name="drift-reports"),
    path(
        "drift/reports/<str:model_id>/",
        views.DriftReportsView.as_view(),
        name="drift-reports-model",
    ),
    path("drift/run/", views.DriftRunView.as_view(), name="drift-run"),
]
