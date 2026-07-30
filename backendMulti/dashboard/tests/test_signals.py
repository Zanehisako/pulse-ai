from __future__ import annotations

import gc
import warnings

from django.test import TestCase
from django.utils import timezone
from unittest.mock import patch

from dashboard.models import ModelResponse


class _FakeChannelLayer:
    def __init__(self):
        self.sent = []

    async def group_send(self, group_name, message):
        self.sent.append((group_name, message))


class DashboardSignalBroadcastTests(TestCase):
    def test_post_save_broadcast_uses_sync_bridge_without_runtime_warning(self):
        layer = _FakeChannelLayer()

        with patch("dashboard.signals.get_channel_layer", return_value=layer):
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always", RuntimeWarning)
                ModelResponse.objects.create(
                    typeModel="stock",
                    jsonResponse={"O+": {"unitsAvailable": 10}},
                    startDate=timezone.now(),
                )
                gc.collect()

        runtime_warnings = [item for item in caught if issubclass(item.category, RuntimeWarning)]
        self.assertEqual(runtime_warnings, [])
        self.assertEqual(len(layer.sent), 1)
        self.assertEqual(layer.sent[0][0], "dashboard_updates")
        self.assertEqual(layer.sent[0][1]["type"], "dashboard_update")
