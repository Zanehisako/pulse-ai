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
from alerts.services.trend_detector import TrendDetector
from inventory.models import PredictionResult


class TrendDetectorTests(TestCase):

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

    def test_detects_clear_declining_trend(self):
        self.seed_history(entity_id="H-T1", blood_type="O+", values=[20.0, 18.0, 16.0, 14.0, 12.0, 10.0, 8.0])

        is_declining, slope, confidence = TrendDetector().detect_declining_trend("H-T1", "O+")

        self.assertTrue(is_declining)
        self.assertLess(slope, 0.0)
        self.assertGreater(confidence, 0.7)

    def test_does_not_detect_stable_values(self):
        self.seed_history(entity_id="H-T2", blood_type="A+", values=[10.0] * 7)

        is_declining, slope, confidence = TrendDetector().detect_declining_trend("H-T2", "A+")

        self.assertFalse(is_declining)
        self.assertLessEqual(abs(slope), 1e-9)
        self.assertEqual(confidence, 0.0)

    def test_does_not_detect_increasing_trend(self):
        self.seed_history(entity_id="H-T3", blood_type="B+", values=[2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0])

        is_declining, slope, confidence = TrendDetector().detect_declining_trend("H-T3", "B+")

        self.assertFalse(is_declining)
        self.assertGreater(slope, 0.0)
        self.assertGreater(confidence, 0.7)

    def test_insufficient_data_returns_false(self):
        self.seed_history(entity_id="H-T4", blood_type="AB+", values=[5.0, 4.0])

        self.assertEqual(
            TrendDetector().detect_declining_trend("H-T4", "AB+"),
            (False, 0.0, 0.0),
        )

    @patch("alerts.services.event_service.notify_clients")
    def test_pre_alert_fires_in_evaluate_prediction(self, _mock_notify):
        from alerts.models import AlertEvent

        self.seed_history(entity_id="H-T5", blood_type="O+", values=[30.0, 28.0, 25.0, 22.0, 19.0, 16.0, 12.0])

        events = AlertEngine().evaluate_prediction(
            model_id="trend-model",
            prediction_result={"prediction": 11.0},
            feature_input={"hospital_id": "H-T5", "blood_type": "O+"},
            source="test",
        )

        self.assertTrue(any(e.source_type == "trend_detection" for e in events))
        self.assertTrue(
            AlertEvent.objects.filter(
                entity_id="H-T5",
                blood_type="O+",
                source_type="trend_detection",
            ).exists()
        )