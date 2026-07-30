from __future__ import annotations

import logging

from alerts.models import AlertEvent, AlertSeverity, AlertStatus
from channels.layers import get_channel_layer
from asgiref.sync import async_to_sync
from notifications.services import publish_notification

logger = logging.getLogger(__name__)

_SEVERITY_TO_NOTIF_TYPE = {
    AlertSeverity.CRITICAL: "error",
    AlertSeverity.WARNING: "warning",
    AlertSeverity.RESOLVED: "success",
}


def broadcast_alert(event: AlertEvent) -> None:
    channel_layer = get_channel_layer()
    if channel_layer is None:
        return
    async_to_sync(channel_layer.group_send)(
        "alerts",
        {
            "type": "send_alert",
            "data": {
                "type": "alert_event",
                **event.to_dict(),
            },
        },
    )


# def notify_clients(event: AlertEvent) -> None:
#     broadcast_alert(event)

#     notification_type = "warning"
#     if event.severity == AlertSeverity.RESOLVED:
#         notification_type = "success"

#     notification = publish_notification(
#         title=event.title,
#         body=event.message,
#         notif_type=notification_type,
#         extra_data={
#             "kind": "alert",
#             "alert_id": str(event.id),
#             "severity": event.severity,
#             "status": event.status,
#             "entity_type": event.entity_type,
#             "entity_id": event.entity_id,
#             "blood_type": event.blood_type,
#             **event.context,
#         },
#         sent_by="alerts-engine",
#     )
#     event.notification_id = notification.id
#     event.save(update_fields=["notification_id"])
def notify_clients(event: AlertEvent) -> None:
    try:
        broadcast_alert(event)
    except Exception as exc:
        logger.error("WebSocket broadcast failed for event %s: %s", event.id, exc)

    if event.status in {AlertStatus.RESOLVED, AlertStatus.MUTED}:
        return

    try:
        notification = publish_notification(
            title=event.title,
            body=event.message,
            notif_type=_SEVERITY_TO_NOTIF_TYPE.get(event.severity, "warning"),
            extra_data={
                "kind": "alert",
                "alert_id": str(event.id),
                "severity": event.severity,
                "status": event.status,
                "entity_type": event.entity_type,
                "entity_id": event.entity_id,
                "blood_type": event.blood_type,
                **event.context,
            },
            sent_by="alerts-engine",
        )
        AlertEvent.objects.filter(pk=event.pk).update(notification_id=notification.id)
        event.notification_id = notification.id
    except Exception as exc:
        logger.error("Notification publish failed for event %s: %s", event.id, exc)
