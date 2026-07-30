"""
Tests for the alert rule engine: normalize_context, _conditions_match,
AlertEngine.evaluate_prediction (create vs refresh), and
AlertEngine.process_stale_alerts (escalation).
"""
from __future__ import annotations

import os

import django

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "backendMulti.settings")
from django.apps import apps

if not apps.ready:
    django.setup()

from datetime import timedelta
from unittest.mock import patch

from django.test import SimpleTestCase, TestCase
from django.utils import timezone

from alerts.services.rule_engine import (
    AlertEngine,
    _conditions_match,
    normalize_context,
)


# ---------------------------------------------------------------------------
# Unit tests – pure functions, no database required
# ---------------------------------------------------------------------------


class NormalizeContextTests(SimpleTestCase):
    """Tests for normalize_context()."""

    def test_double_underscore_prefix_removed(self):
        raw = {"stockout_features_features__current_stock_units": 10}
        result = normalize_context(raw)
        self.assertIn("current_stock_units", result)
        self.assertEqual(result["current_stock_units"], 10)

    def test_original_key_preserved_alongside_normalized(self):
        raw = {"features__blood_product_type": "O+"}
        result = normalize_context(raw)
        self.assertIn("features__blood_product_type", result)
        self.assertIn("blood_product_type", result)

    def test_dotted_path_resolved_to_last_segment(self):
        raw = {"execution.results.0.output.prediction": 0.85}
        result = normalize_context(raw)
        self.assertIn("prediction", result)
        self.assertEqual(result["prediction"], 0.85)

    def test_already_clean_key_not_duplicated(self):
        raw = {"current_stock_units": 5}
        result = normalize_context(raw)
        self.assertEqual(result["current_stock_units"], 5)
        # Should still only have one entry for this clean key
        self.assertEqual(len([k for k in result if k == "current_stock_units"]), 1)

    def test_existing_clean_key_not_overwritten_by_normalization(self):
        # If a clean key already exists with a different value, normalization must not overwrite it
        raw = {"current_stock_units": 99, "features__current_stock_units": 1}
        result = normalize_context(raw)
        # Original clean key value must be preserved
        self.assertEqual(result["current_stock_units"], 99)

    def test_empty_context_returns_empty(self):
        self.assertEqual(normalize_context({}), {})

    def test_numeric_segment_in_dotted_path_skipped(self):
        raw = {"a.0.value": 42}
        result = normalize_context(raw)
        self.assertIn("value", result)
        self.assertEqual(result["value"], 42)


# ---------------------------------------------------------------------------


class ConditionsMatchTests(SimpleTestCase):
    """Tests for _conditions_match()."""

    # --- single condition ---

    def test_single_eq_match(self):
        ctx = {"status": "open"}
        cond = {"field": "status", "op": "==", "value": "open"}
        self.assertTrue(_conditions_match(ctx, cond))

    def test_single_eq_no_match(self):
        ctx = {"status": "closed"}
        cond = {"field": "status", "op": "==", "value": "open"}
        self.assertFalse(_conditions_match(ctx, cond))

    def test_single_lt_match(self):
        ctx = {"stock": 3}
        cond = {"field": "stock", "op": "<", "value": 5}
        self.assertTrue(_conditions_match(ctx, cond))

    def test_single_lt_no_match(self):
        ctx = {"stock": 10}
        cond = {"field": "stock", "op": "<", "value": 5}
        self.assertFalse(_conditions_match(ctx, cond))

    def test_single_lte_match_exact(self):
        ctx = {"days": 5}
        cond = {"field": "days", "op": "<=", "value": 5}
        self.assertTrue(_conditions_match(ctx, cond))

    def test_single_gt_match(self):
        ctx = {"days": 10}
        cond = {"field": "days", "op": ">", "value": 5}
        self.assertTrue(_conditions_match(ctx, cond))

    def test_single_gte_no_match(self):
        ctx = {"days": 3}
        cond = {"field": "days", "op": ">=", "value": 5}
        self.assertFalse(_conditions_match(ctx, cond))

    def test_in_operator_match(self):
        ctx = {"blood_type": "O+"}
        cond = {"field": "blood_type", "op": "in", "value": ["O+", "O-"]}
        self.assertTrue(_conditions_match(ctx, cond))

    def test_in_operator_no_match(self):
        ctx = {"blood_type": "AB+"}
        cond = {"field": "blood_type", "op": "in", "value": ["O+", "O-"]}
        self.assertFalse(_conditions_match(ctx, cond))

    def test_zero_value_matches_numeric_condition(self):
        """0.0 is a valid float; must NOT be treated as falsy/missing."""
        ctx = {"prediction": 0.0}
        cond = {"field": "prediction", "op": "<", "value": 3}
        self.assertTrue(_conditions_match(ctx, cond))

    def test_missing_field_returns_false(self):
        ctx = {}
        cond = {"field": "stock", "op": "<", "value": 5}
        self.assertFalse(_conditions_match(ctx, cond))

    def test_missing_op_returns_false(self):
        ctx = {"stock": 3}
        cond = {"field": "stock", "value": 5}
        self.assertFalse(_conditions_match(ctx, cond))

    # --- "all" (AND) ---

    def test_all_conditions_all_match(self):
        ctx = {"stock": 2, "days": 1}
        cond = {
            "all": [
                {"field": "stock", "op": "<", "value": 5},
                {"field": "days", "op": "<", "value": 3},
            ]
        }
        self.assertTrue(_conditions_match(ctx, cond))

    def test_all_conditions_one_fails(self):
        ctx = {"stock": 2, "days": 10}
        cond = {
            "all": [
                {"field": "stock", "op": "<", "value": 5},
                {"field": "days", "op": "<", "value": 3},
            ]
        }
        self.assertFalse(_conditions_match(ctx, cond))

    def test_all_empty_list_returns_false(self):
        self.assertFalse(_conditions_match({"x": 1}, {"all": []}))

    # --- "any" (OR) ---

    def test_any_conditions_one_match(self):
        ctx = {"stock": 2, "days": 10}
        cond = {
            "any": [
                {"field": "stock", "op": "<", "value": 5},
                {"field": "days", "op": "<", "value": 3},
            ]
        }
        self.assertTrue(_conditions_match(ctx, cond))

    def test_any_conditions_none_match(self):
        ctx = {"stock": 10, "days": 10}
        cond = {
            "any": [
                {"field": "stock", "op": "<", "value": 5},
                {"field": "days", "op": "<", "value": 3},
            ]
        }
        self.assertFalse(_conditions_match(ctx, cond))

    def test_any_empty_list_returns_false(self):
        self.assertFalse(_conditions_match({"x": 1}, {"any": []}))

    # --- edge cases ---

    def test_empty_conditions_returns_false(self):
        self.assertFalse(_conditions_match({"x": 1}, {}))

    def test_invalid_structure_returns_false(self):
        self.assertFalse(_conditions_match({"x": 1}, {"unknown_key": "value"}))


# ---------------------------------------------------------------------------
# Integration tests – require a database (uses Django TestCase with rollback)
# ---------------------------------------------------------------------------


class AlertEngineEvaluatePredictionTests(TestCase):
    """Integration tests for AlertEngine.evaluate_prediction."""

    @patch("alerts.services.event_service.notify_clients")
    def test_creates_new_event_when_condition_matches(self, mock_notify):
        from alerts.models import AlertEvent, AlertRule, AlertSeverity, AlertStatus, AlertTriggerType

        rule = AlertRule.objects.create(
            name="Low Stock Rule",
            description="Stock is critically low",
            is_active=True,
            trigger_type=AlertTriggerType.THRESHOLD_GAP,
            severity_base=AlertSeverity.CRITICAL,
            conditions={"field": "current_stock_units", "op": "<", "value": 5},
        )

        engine = AlertEngine()
        events = engine.evaluate_prediction(
            model_id="test_model",
            prediction_result={"prediction": 2},
            feature_input={"current_stock_units": 3, "hospital_id": "H001", "blood_type": "O+"},
            source="test",
        )

        self.assertEqual(len(events), 1)
        event = events[0]
        self.assertEqual(event.rule, rule)
        self.assertEqual(event.status, AlertStatus.OPEN)
        self.assertEqual(event.severity, AlertSeverity.CRITICAL)
        self.assertEqual(event.entity_id, "H001")
        self.assertEqual(event.blood_type, "O+")
        self.assertTrue(AlertEvent.objects.filter(id=event.id).exists())
        mock_notify.assert_called_once()
        notified_event = mock_notify.call_args[0][0]
        self.assertEqual(notified_event.id, event.id)
        self.assertEqual(notified_event.status, AlertStatus.OPEN)
        self.assertEqual(notified_event.severity, AlertSeverity.CRITICAL)

    @patch("alerts.services.event_service.notify_clients")
    def test_no_event_when_condition_does_not_match(self, mock_notify):
        from alerts.models import AlertRule, AlertSeverity, AlertTriggerType

        AlertRule.objects.create(
            name="Low Stock Rule No Match",
            is_active=True,
            trigger_type=AlertTriggerType.THRESHOLD_GAP,
            severity_base=AlertSeverity.CRITICAL,
            conditions={"field": "current_stock_units", "op": "<", "value": 5},
        )

        engine = AlertEngine()
        events = engine.evaluate_prediction(
            model_id="test_model",
            prediction_result={"prediction": 10},
            feature_input={"current_stock_units": 20},
            source="test",
        )

        self.assertEqual(events, [])
        mock_notify.assert_not_called()

    @patch("alerts.services.event_service.notify_clients")
    def test_refreshes_existing_open_event_instead_of_creating_duplicate(self, mock_notify):
        from alerts.models import AlertEvent, AlertRule, AlertSeverity, AlertStatus, AlertTriggerType

        rule = AlertRule.objects.create(
            name="Refresh Rule",
            description="Will be refreshed",
            is_active=True,
            trigger_type=AlertTriggerType.THRESHOLD_GAP,
            severity_base=AlertSeverity.WARNING,
            conditions={"field": "stock", "op": "<", "value": 10},
        )

        engine = AlertEngine()
        feature_input = {"stock": 5, "hospital_id": "H002", "blood_type": "A+"}

        # First call – creates the event
        events_first = engine.evaluate_prediction(
            model_id="refresh_model",
            prediction_result={},
            feature_input=feature_input,
            source="test",
        )
        self.assertEqual(len(events_first), 1)

        # Second call with same inputs – must refresh, NOT create a new event
        events_second = engine.evaluate_prediction(
            model_id="refresh_model",
            prediction_result={},
            feature_input=feature_input,
            source="test",
        )
        self.assertEqual(len(events_second), 1)

        # Only one event should exist in the DB for this rule+entity+blood_type
        count = AlertEvent.objects.filter(rule=rule, entity_id="H002", blood_type="A+").count()
        self.assertEqual(count, 1)

        # notify_clients must have been called on both the create and refresh
        self.assertEqual(mock_notify.call_count, 2)

    @patch("alerts.services.event_service.notify_clients")
    def test_skips_inactive_rules(self, mock_notify):
        from alerts.models import AlertRule, AlertSeverity, AlertTriggerType

        AlertRule.objects.create(
            name="Inactive Rule",
            is_active=False,
            trigger_type=AlertTriggerType.THRESHOLD_GAP,
            severity_base=AlertSeverity.CRITICAL,
            conditions={"field": "stock", "op": "<", "value": 10},
        )

        engine = AlertEngine()
        events = engine.evaluate_prediction(
            model_id="test_model",
            prediction_result={},
            feature_input={"stock": 1},
            source="test",
        )

        self.assertEqual(events, [])
        mock_notify.assert_not_called()

    @patch("alerts.services.event_service.notify_clients")
    def test_evaluate_prediction_excludes_stale_alert_trigger_type_rules(self, mock_notify):
        from alerts.models import AlertRule, AlertSeverity, AlertTriggerType

        AlertRule.objects.create(
            name="Stale Rule",
            is_active=True,
            trigger_type=AlertTriggerType.STALE_ALERT,
            severity_base=AlertSeverity.CRITICAL,
            conditions={"field": "stock", "op": "<", "value": 10},
        )

        engine = AlertEngine()
        events = engine.evaluate_prediction(
            model_id="test_model",
            prediction_result={},
            feature_input={"stock": 1},
            source="test",
        )

        self.assertEqual(events, [])

    @patch("alerts.services.event_service.notify_clients")
    def test_evaluate_prediction_matches_hospital_scope_from_hospital_id(self, mock_notify):
        from alerts.models import AlertEvent, AlertRule, AlertSeverity, AlertTriggerType

        rule = AlertRule.objects.create(
            name="H001_O+_stockout_warning",
            description="Hospital scoped stockout warning",
            is_active=True,
            trigger_type=AlertTriggerType.THRESHOLD_GAP,
            severity_base=AlertSeverity.WARNING,
            scope_type="hospital",
            scope_ref="H001",
            conditions={"field": "predicted_value", "op": "<=", "value": 3},
            created_by="system",
        )

        events = AlertEngine().evaluate_prediction(
            model_id="stockout_days_predictor",
            prediction_result={"prediction": 2.5},
            feature_input={
                "entity_id": "S001OP",
                "hospital_id": "H001",
                "blood_type": "O+",
                "current_stock_units": 12,
            },
            source="scheduled-stockout",
        )

        self.assertEqual(len(events), 1)
        event = AlertEvent.objects.get(rule=rule)
        self.assertEqual(event.entity_id, "H001")
        self.assertEqual(event.context["source_entity_id"], "S001OP")
        self.assertEqual(event.context["current_stock_units"], 12)
        mock_notify.assert_called_once()

    @patch("alerts.services.event_service.notify_clients")
    def test_missing_hospital_id_does_not_match_supply_id_as_hospital_scope(self, mock_notify):
        from alerts.models import AlertEvent, AlertRule, AlertSeverity, AlertTriggerType

        AlertRule.objects.create(
            name="Supply-id-looking hospital rule",
            description="Should not match without hospital_id",
            is_active=True,
            trigger_type=AlertTriggerType.THRESHOLD_GAP,
            severity_base=AlertSeverity.WARNING,
            scope_type="hospital",
            scope_ref="S001OP",
            conditions={"field": "predicted_value", "op": "<=", "value": 3},
            created_by="system",
        )

        with patch("alerts.services.rule_engine.logger.warning") as mock_warning:
            events = AlertEngine().evaluate_prediction(
                model_id="stockout_days_predictor",
                prediction_result={"prediction": 2.5},
                feature_input={
                    "entity_id": "S001OP",
                    "blood_type": "O+",
                    "current_stock_units": 12,
                    "requires_hospital_scope": True,
                },
                source="scheduled-stockout",
            )

        self.assertEqual(events, [])
        self.assertFalse(AlertEvent.objects.exists())
        mock_notify.assert_not_called()
        mock_warning.assert_called_once()

    @patch("alerts.services.event_service.notify_clients")
    def test_donor_scoped_prediction_without_hospital_id_does_not_warn(self, mock_notify):
        from alerts.models import AlertEvent, AlertRule, AlertSeverity, AlertTriggerType

        rule = AlertRule.objects.create(
            name="Donor priority outreach",
            description="High priority donor outreach",
            is_active=True,
            trigger_type=AlertTriggerType.THRESHOLD_GAP,
            severity_base=AlertSeverity.WARNING,
            scope_type="global",
            conditions={"field": "predicted_value", "op": ">=", "value": 0.7},
            created_by="system",
        )

        with patch("alerts.services.rule_engine.logger.warning") as mock_warning:
            events = AlertEngine().evaluate_prediction(
                model_id="donor_priority_policy_model",
                prediction_result={"prediction": 0.91},
                feature_input={
                    "entity_id": "D0000001",
                    "entity_type": "donor",
                    "blood_type": "O-",
                    "requires_hospital_scope": False,
                },
                source="scheduled-donor-priority-policy",
            )

        self.assertEqual(len(events), 1)
        event = AlertEvent.objects.get(rule=rule)
        self.assertEqual(event.entity_id, "D0000001")
        self.assertEqual(event.entity_type, "donor")
        self.assertNotIn("hospital_id", event.context)
        mock_notify.assert_called_once()
        mock_warning.assert_not_called()

    @patch("alerts.services.event_service.notify_clients")
    def test_resolved_event_triggers_new_event_on_next_match(self, mock_notify):
        from alerts.models import AlertEvent, AlertRule, AlertSeverity, AlertStatus, AlertTriggerType

        rule = AlertRule.objects.create(
            name="Resolved-then-new Rule",
            description="After resolve, new event expected",
            is_active=True,
            trigger_type=AlertTriggerType.THRESHOLD_GAP,
            severity_base=AlertSeverity.WARNING,
            conditions={"field": "stock", "op": "<", "value": 10},
        )

        engine = AlertEngine()
        feature_input = {"stock": 5, "hospital_id": "H003", "blood_type": "B-"}

        # First call – creates event
        events_first = engine.evaluate_prediction(
            model_id="model_resolved",
            prediction_result={},
            feature_input=feature_input,
            source="test",
        )
        self.assertEqual(len(events_first), 1)
        first_event = events_first[0]

        # Manually resolve the event
        first_event.status = AlertStatus.RESOLVED
        first_event.save(update_fields=["status"])

        # Second call – resolved event excluded; a new one should be created
        events_second = engine.evaluate_prediction(
            model_id="model_resolved",
            prediction_result={},
            feature_input=feature_input,
            source="test",
        )
        self.assertEqual(len(events_second), 1)
        second_event = events_second[0]

        self.assertNotEqual(first_event.id, second_event.id)
        self.assertEqual(
            AlertEvent.objects.filter(rule=rule, entity_id="H003", blood_type="B-").count(), 2
        )


# ---------------------------------------------------------------------------


class AlertEngineProcessStaleAlertsTests(TestCase):
    """Integration tests for AlertEngine.process_stale_alerts."""

    @patch("alerts.services.event_service.notify_clients")
    def test_escalates_open_stale_event(self, mock_notify):
        from alerts.models import AlertEvent, AlertRule, AlertSeverity, AlertStatus, AlertTriggerType

        rule = AlertRule.objects.create(
            name="Stale Open Rule",
            is_active=True,
            trigger_type=AlertTriggerType.STALE_ALERT,
            severity_base=AlertSeverity.CRITICAL,
            conditions={},
            escalation_after_minutes=60,
        )

        # Create an event opened 2 hours ago
        old_time = timezone.now() - timedelta(hours=2)
        event = AlertEvent.objects.create(
            rule=rule,
            event_key="stale-open-key",
            title="Stale open",
            message="Stale",
            status=AlertStatus.OPEN,
            severity=AlertSeverity.WARNING,
            opened_at=old_time,
        )

        engine = AlertEngine()
        processed = engine.process_stale_alerts()

        self.assertEqual(processed, 1)
        event.refresh_from_db()
        self.assertEqual(event.status, AlertStatus.ESCALATED)
        self.assertEqual(event.severity, AlertSeverity.CRITICAL)
        self.assertIsNotNone(event.escalated_at)

    @patch("alerts.services.event_service.notify_clients")
    def test_escalates_acknowledged_stale_event(self, mock_notify):
        from alerts.models import AlertEvent, AlertRule, AlertSeverity, AlertStatus, AlertTriggerType

        rule = AlertRule.objects.create(
            name="Stale Ack Rule",
            is_active=True,
            trigger_type=AlertTriggerType.STALE_ALERT,
            severity_base=AlertSeverity.CRITICAL,
            conditions={},
            escalation_after_minutes=30,
        )

        old_time = timezone.now() - timedelta(hours=1)
        event = AlertEvent.objects.create(
            rule=rule,
            event_key="stale-ack-key",
            title="Stale acknowledged",
            message="Stale",
            status=AlertStatus.ACKNOWLEDGED,
            severity=AlertSeverity.WARNING,
            opened_at=old_time,
        )

        engine = AlertEngine()
        processed = engine.process_stale_alerts()

        self.assertEqual(processed, 1)
        event.refresh_from_db()
        self.assertEqual(event.status, AlertStatus.ESCALATED)

    @patch("alerts.services.event_service.notify_clients")
    def test_does_not_escalate_recent_event(self, mock_notify):
        from alerts.models import AlertEvent, AlertRule, AlertSeverity, AlertStatus, AlertTriggerType

        rule = AlertRule.objects.create(
            name="Stale Recent Rule",
            is_active=True,
            trigger_type=AlertTriggerType.STALE_ALERT,
            severity_base=AlertSeverity.CRITICAL,
            conditions={},
            escalation_after_minutes=120,
        )

        # Event opened only 10 minutes ago – not stale yet
        recent_time = timezone.now() - timedelta(minutes=10)
        AlertEvent.objects.create(
            rule=rule,
            event_key="recent-key",
            title="Recent event",
            message="Recent",
            status=AlertStatus.OPEN,
            severity=AlertSeverity.WARNING,
            opened_at=recent_time,
        )

        engine = AlertEngine()
        processed = engine.process_stale_alerts()

        self.assertEqual(processed, 0)

    @patch("alerts.services.event_service.notify_clients")
    def test_does_not_escalate_already_resolved_event(self, mock_notify):
        from alerts.models import AlertEvent, AlertRule, AlertSeverity, AlertStatus, AlertTriggerType

        rule = AlertRule.objects.create(
            name="Stale Resolved Rule",
            is_active=True,
            trigger_type=AlertTriggerType.STALE_ALERT,
            severity_base=AlertSeverity.CRITICAL,
            conditions={},
            escalation_after_minutes=60,
        )

        old_time = timezone.now() - timedelta(hours=2)
        AlertEvent.objects.create(
            rule=rule,
            event_key="resolved-old-key",
            title="Resolved old event",
            message="Resolved",
            status=AlertStatus.RESOLVED,
            severity=AlertSeverity.RESOLVED,
            opened_at=old_time,
        )

        engine = AlertEngine()
        processed = engine.process_stale_alerts()

        self.assertEqual(processed, 0)
