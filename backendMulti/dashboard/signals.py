import logging

from asgiref.sync import async_to_sync
from django.db.models.signals import post_save
from django.dispatch import receiver
from channels.layers import get_channel_layer
from .models import ModelResponse

logger = logging.getLogger(__name__)


@receiver(post_save, sender=ModelResponse)
def send_dashboard_update(sender, instance, created, **kwargs):
    if instance.endDate is not None:
        return

    channel_layer = get_channel_layer()
    if channel_layer is None:
        return

    data = {
        "id": instance.id,
        "typeModel": instance.typeModel,
        "jsonResponse": instance.jsonResponse,
        "region": instance.region,
        "product": instance.product,
        "period": instance.period,
        "alert_level": instance.alert_level,
        "startDate": instance.startDate.isoformat(),
        "endDate": instance.endDate.isoformat() if instance.endDate else None,
    }
    message = {"type": "dashboard_update", "data": data}

    try:
        async_to_sync(channel_layer.group_send)("dashboard_updates", message)
    except Exception:
        logger.exception("Dashboard WebSocket push failed")
