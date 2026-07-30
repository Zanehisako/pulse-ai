from django.urls import path
from .views import (
    DeleteNotificationView,
    MarkAllNotificationsAsReadView,
    MarkNotificationAsReadView,
    NotificationListView,
    SendNotificationView,
)

urlpatterns = [
    path("send/", SendNotificationView.as_view(), name="send_notification"),
    path("", NotificationListView.as_view(), name="list_notifications"),
    path(
        "mark-all-as-read/",
        MarkAllNotificationsAsReadView.as_view(),
        name="mark_all_notifications_as_read",
    ),
    path(
        "<uuid:pk>/mark-as-read/",
        MarkNotificationAsReadView.as_view(),
        name="mark_notification_as_read",
    ),
    path("<uuid:pk>/", DeleteNotificationView.as_view(), name="delete_notification"),
]
