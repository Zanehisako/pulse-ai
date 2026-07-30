from django.urls import path

from .views import (
    AlertAcknowledgeView,
    AlertDetailView,
    AlertEscalateView,
    AlertListView,
    AlertResolveView,
    AlertRuleDetailView,
    AlertRuleListView,
    AlertSummaryView,
    AlertTestFireView,
)


urlpatterns = [
    path("", AlertListView.as_view(), name="alert-list"),
    path("summary/", AlertSummaryView.as_view(), name="alert-summary"),
    path("test-fire/", AlertTestFireView.as_view(), name="alert-test-fire"),
    path("rules/", AlertRuleListView.as_view(), name="alert-rule-list"),
    path("rules/<int:pk>/", AlertRuleDetailView.as_view(), name="alert-rule-detail"),
    path("<uuid:pk>/", AlertDetailView.as_view(), name="alert-detail"),
    path("<uuid:pk>/acknowledge/", AlertAcknowledgeView.as_view(), name="alert-acknowledge"),
    path("<uuid:pk>/resolve/", AlertResolveView.as_view(), name="alert-resolve"),
    path("<uuid:pk>/escalate/", AlertEscalateView.as_view(), name="alert-escalate"),
]