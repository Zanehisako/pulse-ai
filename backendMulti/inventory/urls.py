from django.urls import path

from inventory.views import (
    DonorPropensityScoresView,
    PredictionResultListView,
    PredictionTriggerView,
    SupplyHistoryView,
)


urlpatterns = [
    path("", PredictionResultListView.as_view(), name="prediction-results"),
    path("trigger/", PredictionTriggerView.as_view(), name="prediction-trigger"),
    path("supply-history/", SupplyHistoryView.as_view(), name="supply-history"),
    path("donors/scores/", DonorPropensityScoresView.as_view(), name="donor-propensity-scores"),
]
