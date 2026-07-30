import os
from types import SimpleNamespace
from unittest.mock import patch

import django

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "backendMulti.settings")
from django.apps import apps

if not apps.ready:
    django.setup()

from django.test import Client, SimpleTestCase, TestCase
from django.utils import timezone

from dashboard.views import (
    _build_ideal_snapshot_from_ml_stats,
    _normalize_ideal_donor_stats,
)
from ml.models import ModelStats


class IdealSnapshotMappingTests(SimpleTestCase):
    def test_normalize_ideal_stats_prefers_intervals_for_numeric_features(self):
        stats = {
            "eligible_to_donate": {"mode": 1},
            "blood_type": {"mode": "High-need or rare type"},
            "center_distance_km": {"lower": 0, "upper": 20, "mean": 9.5},
            "readiness_score": {"lower": 0.7, "upper": 1.0},
        }

        profile = _normalize_ideal_donor_stats(stats)

        self.assertEqual(profile["eligible_to_donate"], "Yes")
        self.assertEqual(profile["blood_type"], "High-need or rare type")
        self.assertEqual(profile["center_distance_km"], "0-20")
        self.assertEqual(profile["readiness_score"], "0.7-1")

    @patch("dashboard.views.ModelStats.objects.filter")
    def test_build_snapshot_uses_configured_donor_priority_model_stats(self, mock_filter):
        class FakeQuerySet:
            def __init__(self, rows):
                self.rows = rows

            def order_by(self, *_args):
                return self

            def __getitem__(self, _slice):
                return self.rows

        stats_row = SimpleNamespace(
            model_id="donor_priority_policy_model",
            identifier="donor_priority_policy_model",
            stats={
                "eligible_to_donate": {"mode": 1},
                "is_rare_type": {"mode": 1},
                "response_readiness_index": {"lower": 0.7, "upper": 1.0},
            },
            made_at=None,
        )

        mock_filter.side_effect = lambda **_kwargs: FakeQuerySet([stats_row])

        snapshot = _build_ideal_snapshot_from_ml_stats({"region": "quebec"})

        self.assertIsNotNone(snapshot)
        self.assertEqual(snapshot["typeModel"], "ideal")
        self.assertEqual(snapshot["region"], "quebec")
        self.assertEqual(snapshot["jsonResponse"]["eligible_to_donate"], "Yes")
        self.assertEqual(snapshot["jsonResponse"]["is_rare_type"], "Yes")
        self.assertEqual(snapshot["jsonResponse"]["response_readiness_index"], "0.7-1")


class IdealSnapshotIntegrationTests(TestCase):
    def test_latest_dashboard_uses_priority_model_stats_for_ideal_profile(self):
        ModelStats.objects.create(
            model_id="donor_priority_policy_model",
            identifier="donor_priority_policy_model",
            stats={
                "eligible_to_donate": {"mode": 1},
                "blood_type": {"mode": "High-need or rare type"},
                "center_distance_km": {"lower": 0, "upper": 20},
                "readiness_score": {"lower": 0.7, "upper": 1.0},
            },
            made_at=timezone.now(),
        )

        response = Client().get("/dashboard/latest/?region=quebec")

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        ideal = payload["snapshots"]["ideal"]
        self.assertEqual(ideal["typeModel"], "ideal")
        self.assertEqual(
            ideal["jsonResponse"],
            {
                "blood_type": "High-need or rare type",
                "eligible_to_donate": "Yes",
                "center_distance_km": "0-20",
                "readiness_score": "0.7-1",
            },
        )
