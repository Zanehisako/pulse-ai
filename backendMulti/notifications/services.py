from __future__ import annotations

from asgiref.sync import async_to_sync
from channels.layers import get_channel_layer

from notifications.models import Notification, NotificationType
from authApp.models import User


def get_notification_group(recipient: User | None = None) -> str:
    return (
        f"notifications_user_{recipient.id}"
        if recipient is not None
        else "notifications_broadcast"
    )


def broadcast_notification(notification: Notification) -> None:
    channel_layer = get_channel_layer()
    if channel_layer is None:
        return
    async_to_sync(channel_layer.group_send)(
        get_notification_group(notification.recipient),
        {
            "type": "send_notification",
            "data": notification.to_dict(),
        },
    )


def publish_notification(
    *,
    title: str,
    body: str,
    notif_type: str = NotificationType.INFO,
    extra_data: dict | None = None,
    sent_by: str = "admin",
    recipient: User | None = None,
) -> Notification:
    notification = Notification.objects.create(
        title=title,
        body=body,
        type=notif_type,
        extra_data=extra_data or {},
        sent_by=sent_by,
        recipient=recipient,
    )
    broadcast_notification(notification)
    return notification
