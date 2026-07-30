from __future__ import annotations

import os
from datetime import date, timedelta
from unittest.mock import patch

import django

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "backendMulti.settings")
from django.apps import apps

if not apps.ready:
    django.setup()

from django.test import TestCase

from alerts.services.rule_engine import AlertEngine
from alerts.services.threshold_engine import ThresholdEngine
from inventory.models import PredictionResult


class ThresholdEngineTests(TestCase):

    def seed_history(self, *, entity_id: str, blood_type: str = "O+", values: list[float]) -> None:
        base_date = date(2026, 1, 1)
        for offset, value in enumerate(values):
            PredictionResult.objects.create(
                entity_id=entity_id,
                entity_type="hospital",
                hospital_id=entity_id,
                blood_type=blood_type,
                model_name="hospital_shortage_predictor",
                model_version="champion",
                predicted_value=value,
                predicted_for_date=base_date + timedelta(days=offset),
            )

    def test_zscore_returns_correct_value(self):
        values = [float(v) for v in range(10, 40)]
        self.seed_history(entity_id="H-Z", blood_type="O+", values=values)

        result = ThresholdEngine().compute("H-Z", "O+", "zscore")

        import numpy as np
        expected = float(np.mean(values) - 2 * np.std(values))
        self.assertAlmostEqual(result, expected, places=6)

    def test_percentile_returns_correct_value(self):
        values = [float(v) for v in range(10, 40)]
        self.seed_history(entity_id="H-P", blood_type="A+", values=values)

        result = ThresholdEngine().compute("H-P", "A+", "percentile", percentile=15)

        import numpy as np
        expected = float(np.percentile(values, 15))
        self.assertAlmostEqual(result, expected, places=6)

    def test_insufficient_history_returns_fallback(self):
        self.seed_history(entity_id="H-SHORT", blood_type="B+", values=[1.0, 2.0, 3.0, 4.0, 5.0])

        result = ThresholdEngine().compute("H-SHORT", "B+", "zscore", fallback=50.0)

        self.assertEqual(result, 50.0)

    def test_empty_history_returns_fallback(self):
        result = ThresholdEngine().compute("H-EMPTY", "AB+", "percentile", fallback=42.0)

        self.assertEqual(result, 42.0)

    def test_unknown_mode_returns_fallback(self):
        self.seed_history(entity_id="H-U", blood_type="O-", values=[float(v) for v in range(20, 50)])

        result = ThresholdEngine().compute("H-U", "O-", "banana", fallback=12.0)

        self.assertEqual(result, 12.0)

    @patch("alerts.services.event_service.notify_clients")
    def test_dynamic_threshold_used_in_rule_evaluation(self, _mock_notify):
        from alerts.models import AlertRule, AlertSeverity, AlertTriggerType

        values = [float(v) for v in range(40, 70)]
        self.seed_history(entity_id="H-DYN", blood_type="O+", values=values)

        rule = AlertRule.objects.create(
            name="Dynamic Threshold Rule",
            description="Uses computed threshold",
            is_active=True,
            trigger_type=AlertTriggerType.THRESHOLD_GAP,
            severity_base=AlertSeverity.WARNING,
            conditions={
                "field": "predicted_value",
                "op": "<",
                "value": 5,
                "threshold_mode": "zscore",
            },
        )

        events = AlertEngine().evaluate_prediction(
            model_id="dynamic-model",
            prediction_result={"prediction": 30},
            feature_input={"hospital_id": "H-DYN", "blood_type": "O+"},
            source="test",
        )

        self.assertEqual(len(events), 1)
        self.assertEqual(events[0].rule, rule)
        self.assertIsNotNone(events[0].threshold_value)
        self.assertGreater(events[0].threshold_value, 5.0)