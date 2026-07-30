from django.urls import path

from bloodbag.views import BloodBagListView, BloodBagSyncView


urlpatterns = [
    path("", BloodBagListView.as_view(), name="bloodbag-list"),
    path("sync/", BloodBagSyncView.as_view(), name="bloodbag-sync"),
]
