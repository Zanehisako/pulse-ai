from __future__ import annotations

import gc
import warnings

from django.test import SimpleTestCase
from unittest.mock import patch

from alerts.models import AlertEvent, AlertSeverity, AlertStatus
from alerts.services.delivery_service import broadcast_alert, notify_clients


class _FakeChannelLayer:
    def __init__(self):
        self.sent = []

    async def group_send(self, group_name, message):
        self.sent.append((group_name, message))


class AlertDeliveryBroadcastTests(SimpleTestCase):
    def test_broadcast_alert_uses_sync_bridge_without_runtime_warning(self):
        layer = _FakeChannelLayer()
        event = AlertEvent(
            event_key="alert-key",
            severity=AlertSeverity.WARNING,
            status=AlertStatus.OPEN,
            title="Low stock",
            message="Low stock warning",
            entity_type="hospital",
            entity_id="H001",
            blood_type="O+",
        )

        with patch("alerts.services.delivery_service.get_channel_layer", return_value=layer):
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always", RuntimeWarning)
                broadcast_alert(event)
                gc.collect()

        runtime_warnings = [item for item in caught if issubclass(item.category, RuntimeWarning)]
        self.assertEqual(runtime_warnings, [])
        self.assertEqual(len(layer.sent), 1)
        self.assertEqual(layer.sent[0][0], "alerts")
        self.assertEqual(layer.sent[0][1]["type"], "send_alert")

    def test_resolved_event_broadcasts_without_publishing_notification(self):
        event = AlertEvent(
            event_key="resolved-key",
            severity=AlertSeverity.CRITICAL,
            status=AlertStatus.RESOLVED,
            title="Resolved stockout",
            message="The alert was resolved",
            entity_type="hospital",
            entity_id="H001",
            blood_type="O+",
        )

        with (
            patch("alerts.services.delivery_service.broadcast_alert") as mock_broadcast,
            patch("alerts.services.delivery_service.publish_notification") as mock_publish,
        ):
            notify_clients(event)

        mock_broadcast.assert_called_once_with(event)
        mock_publish.assert_not_called()
