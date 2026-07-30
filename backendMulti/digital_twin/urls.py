from django.urls import path

from .views import (
    TwinActionView,
    TwinBranchView,
    TwinConfigView,
    TwinEventView,
    TwinPromotionView,
    TwinRecommendationView,
    TwinRunDetailView,
    TwinRunListView,
    TwinSnapshotView,
)


urlpatterns = [
    path("config/", TwinConfigView.as_view(), name="digital-twin-config"),
    path("runs/", TwinRunListView.as_view(), name="digital-twin-run-list"),
    path("runs/<uuid:run_id>/", TwinRunDetailView.as_view(), name="digital-twin-run-detail"),
    path("runs/<uuid:run_id>/snapshots/", TwinSnapshotView.as_view(), name="digital-twin-snapshot"),
    path("runs/<uuid:run_id>/events/", TwinEventView.as_view(), name="digital-twin-event"),
    path("runs/<uuid:run_id>/actions/", TwinActionView.as_view(), name="digital-twin-action"),
    path("runs/<uuid:run_id>/branches/", TwinBranchView.as_view(), name="digital-twin-branch"),
    path("runs/<uuid:run_id>/recommendations/", TwinRecommendationView.as_view(), name="digital-twin-recommendation"),
    path("promotions/", TwinPromotionView.as_view(), name="digital-twin-promotion"),
]
