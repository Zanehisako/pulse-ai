from __future__ import annotations

import copy
import json
from pathlib import Path

from django.contrib.auth import get_user_model
from django.test import SimpleTestCase, TestCase
from django.urls import reverse
from django.utils import timezone
from rest_framework.test import APIClient

from alerts.models import AlertEvent
from inventory.models import (
    BloodSupply,
    Donor,
    Hospital,
    HospitalSupplyFeature,
    PredictionResult,
)
from ml.models import DriftReport, MLModelConfig

from .config import DigitalTwinConfigError, validate_config_payload
from .models import (
    TwinAction,
    TwinAuditLog,
    TwinBranch,
    TwinEvent,
    TwinFrame,
    TwinRecommendation,
    TwinRun,
    TwinSnapshot,
)
from .services import (
    capture_snapshot,
    create_frame,
    create_twin_run,
    execute_action,
    inject_event,
    promote_action,
    recommend_action,
    run_branch,
)


CONFIG_PATH = Path(__file__).resolve().parents[1] / "ml" / "config" / "digital_twin.json"
ORCHESTRATOR_CONFIG_PATH = Path(__file__).resolve().parents[1] / "ml" / "config" / "config.json"


def _config_payload() -> dict:
    return json.loads(CONFIG_PATH.read_text(encoding="utf-8"))


class DigitalTwinConfigValidationTests(SimpleTestCase):
    def test_default_config_validates(self):
        payload = validate_config_payload(_config_payload())
        self.assertEqual(payload["version"], "2026-06-12")
        self.assertIn("events", payload)
        self.assertIn("actions", payload)

    def test_duplicate_event_ids_fail(self):
        payload = _config_payload()
        payload["events"].append(copy.deepcopy(payload["events"][0]))
        with self.assertRaises(DigitalTwinConfigError):
            validate_config_payload(payload)

    def test_unsupported_effect_field_fails(self):
        payload = _config_payload()
        payload["actions"][0]["effects"]["hardcoded_unknown_effect"] = 1
        with self.assertRaises(DigitalTwinConfigError):
            validate_config_payload(payload)

    def test_unsafe_enabled_promotion_without_targets_fails(self):
        payload = _config_payload()
        payload["promotion_targets"] = {"enabled": True, "targets": []}
        with self.assertRaises(DigitalTwinConfigError):
            validate_config_payload(payload)


class DigitalTwinLifecycleTests(TestCase):
    def setUp(self):
        now = timezone.now()
        hospital = Hospital.objects.create(
            hospital_id="H001",
            name="Central Hospital",
            wilaya="Algiers",
        )
        BloodSupply.objects.create(
            supply_id="S001",
            hospital=hospital,
            blood_product_type="O+",
            current_stock_units=42,
            usage_today=4,
            event_timestamp=now,
        )
        HospitalSupplyFeature.objects.create(
            hospital_id="H001",
            event_timestamp=now,
            temperature=22,
            rain_mm=0,
            holiday=0,
            disaster=0,
            scheduled_surgeries=5,
            trauma_cases=2,
            current_inventory=42,
        )
        Donor.objects.create(
            donor_id="D001",
            name="Private Donor",
            wilaya="Algiers",
            event_timestamp=now,
            city_id="C001",
            lat=36.7,
            lon=3.1,
            availability=1,
            blood_group="O+",
            recency_days=120,
            frequency_365=2,
            days_until_eligible=0,
            cluster_id=1,
        )
        PredictionResult.objects.create(
            entity_id="H001-O+",
            entity_type="hospital_supply",
            hospital_id="H001",
            blood_type="O+",
            model_name="stockout-test",
            predicted_value=0.72,
            confidence=0.8,
            predicted_for_date=now.date(),
            alert_triggered=True,
        )
        AlertEvent.objects.create(
            source_type="prediction",
            source_ref="test",
            event_key="stockout-risk",
            severity="warning",
            title="Stockout risk",
            message="O+ is under pressure.",
            entity_type="hospital",
            entity_id="H001",
            blood_type="O+",
            predicted_value=0.72,
            threshold_value=0.7,
        )
        MLModelConfig.objects.create(
            model_id="stockout-test",
            description="Test model",
            file_path="/tmp/model.joblib",
            model_type="sklearn",
            features=["current_stock_units"],
        )
        DriftReport.objects.create(
            model_id="stockout-test",
            drift_detected=True,
            drift_score=0.9,
            severity="critical",
            features_checked=1,
            features_drifted=1,
        )

    def _operational_counts(self) -> dict[str, int]:
        return {
            "predictions": PredictionResult.objects.count(),
            "alerts": AlertEvent.objects.count(),
            "supplies": BloodSupply.objects.count(),
            "features": HospitalSupplyFeature.objects.count(),
            "donors": Donor.objects.count(),
        }

    def test_snapshot_redacts_donor_identifiers_and_persists_audit(self):
        run = create_twin_run({"source_mode": "live"})
        snapshot = capture_snapshot(run)
        self.assertEqual(TwinSnapshot.objects.count(), 1)
        donor = snapshot.redacted_payload["donors"][0]
        self.assertNotIn("donor_id", donor)
        self.assertNotIn("name", donor)
        self.assertIn("donor_ref", donor)
        self.assertGreaterEqual(TwinAuditLog.objects.count(), 2)

    def test_twin_lifecycle_does_not_mutate_operational_tables(self):
        before = self._operational_counts()
        run = create_twin_run({"source_mode": "live"})
        snapshot = capture_snapshot(run)
        frame = create_frame(
            run,
            tick_number=1,
            simulated_hour=6,
            snapshot=snapshot,
            simulated_state={
                "simulated_hour": 6,
                "inventory_by_hospital": {"H001": {"O+": 40}},
                "total_inventory_units": 40,
                "donor_count": 1,
                "active_alert_count": 1,
                "latest_prediction_count": 1,
                "critical_drift_count": 1,
            },
        )
        twin_cfg = _config_payload()
        event = inject_event(
            run,
            event_key=twin_cfg["events"][0]["id"],
            severity=0.5,
            start_hour=6,
        )
        action = execute_action(
            run,
            action_key=twin_cfg["actions"][0]["id"],
            simulated_hour=6,
        )
        branch = run_branch(
            run,
            branch_key="branch-a",
            policy_overrides={"strategy_key": "full_response"},
        )
        recommendation = recommend_action(run, source_tool_ids=["test-config-tool"])

        self.assertEqual(TwinFrame.objects.get().frame_id, frame.frame_id)
        self.assertEqual(TwinEvent.objects.get().event_id, event.event_id)
        self.assertEqual(TwinAction.objects.get().action_id, action.action_id)
        self.assertEqual(TwinBranch.objects.get().branch_id, branch.branch_id)
        self.assertEqual(TwinRecommendation.objects.get().recommendation_id, recommendation.recommendation_id)
        self.assertEqual(self._operational_counts(), before)
        self.assertEqual(action.promotion_status, TwinAction.PromotionStatus.DISABLED)
        with self.assertRaises(DigitalTwinConfigError):
            promote_action(action)
        self.assertEqual(self._operational_counts(), before)


class DigitalTwinApiTests(TestCase):
    def test_run_create_api_captures_synthetic_snapshot(self):
        user = get_user_model().objects.create_user(
            username="operator",
            password="test-pass",
        )
        client = APIClient()
        client.force_authenticate(user=user)
        response = client.post(
            reverse("digital-twin-run-list"),
            {"source_mode": "synthetic", "scenario_key": "baseline"},
            format="json",
        )
        self.assertEqual(response.status_code, 201)
        self.assertIn("run", response.data)
        self.assertIn("snapshot", response.data)
        self.assertEqual(TwinRun.objects.count(), 1)
        self.assertEqual(TwinSnapshot.objects.count(), 1)


class DigitalTwinExternalToolConfigTests(SimpleTestCase):
    def test_twin_tools_are_configured_without_core_tool_names(self):
        payload = json.loads(ORCHESTRATOR_CONFIG_PATH.read_text(encoding="utf-8"))
        tools = {
            row["id"]: row
            for row in payload["external_tools"]
            if row.get("adapter") == "digital_twin"
        }
        self.assertEqual(
            set(tools),
            {
                "digital_twin_current_state",
                "digital_twin_run_branch",
                "digital_twin_compare_policies",
                "digital_twin_recommend_action",
                "digital_twin_promote_action",
            },
        )
        self.assertEqual(
            {row["operation"] for row in tools.values()},
            {
                "current_state",
                "run_branch",
                "compare_policies",
                "recommend_action",
                "promote_action",
            },
        )
