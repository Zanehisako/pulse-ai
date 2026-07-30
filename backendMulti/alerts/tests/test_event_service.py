# alerts/tests/test_event_service.py

from __future__ import annotations
from unittest.mock import patch
from django.test import TestCase
from django.utils import timezone
from alerts.models import (
    AlertEvent, AlertRule, AlertEventHistory,
    AlertSeverity, AlertStatus, AlertTriggerType,
)
from alerts.services.event_service import (
    acknowledge_event, resolve_event, escalate_event,
    create_or_refresh_event, record_history,
    InvalidTransitionError,
)


def make_rule(**kwargs):
    defaults = dict(
        name="Test Rule",
        is_active=True,
        trigger_type=AlertTriggerType.THRESHOLD_GAP,
        severity_base=AlertSeverity.WARNING,
        conditions={"field": "stock", "op": "<", "value": 10},
        escalation_after_minutes=60,
        dedup_window_minutes=60,
    )
    defaults.update(kwargs)
    return AlertRule.objects.create(**defaults)


def make_event(rule, status=AlertStatus.OPEN, severity=AlertSeverity.WARNING, **kwargs):
    defaults = dict(
        rule=rule,
        event_key=f"test-key-{status}",
        title="Test Event",
        message="Test message",
        status=status,
        severity=severity,
        opened_at=timezone.now(),
    )
    defaults.update(kwargs)
    return AlertEvent.objects.create(**defaults)


class TestRecordHistory(TestCase):

    @patch("alerts.services.event_service.notify_clients")
    def test_history_entry_created_with_correct_fields(self, _):
        rule = make_rule()
        event = make_event(rule)
        record_history(event, "test_action", actor="user1", note="a note")

        h = AlertEventHistory.objects.get(event=event, action="test_action")
        self.assertEqual(h.actor, "user1")
        self.assertEqual(h.note, "a note")
        self.assertIn("id", h.snapshot)  # snapshot is event.to_dict()


class TestAcknowledgeEvent(TestCase):

    @patch("alerts.services.event_service.notify_clients")
    def test_open_event_becomes_acknowledged(self, mock_notify):
        rule = make_rule()
        event = make_event(rule, status=AlertStatus.OPEN)

        result = acknowledge_event(event, actor="nurse1")

        event.refresh_from_db()
        self.assertEqual(event.status, AlertStatus.ACKNOWLEDGED)
        self.assertIsNotNone(event.acknowledged_at)
        mock_notify.assert_called_once_with(event)

    @patch("alerts.services.event_service.notify_clients")
    def test_escalated_event_can_be_acknowledged(self, mock_notify):
        rule = make_rule()
        event = make_event(rule, status=AlertStatus.ESCALATED)

        acknowledge_event(event, actor="doctor1")

        event.refresh_from_db()
        self.assertEqual(event.status, AlertStatus.ACKNOWLEDGED)

    @patch("alerts.services.event_service.notify_clients")
    def test_resolved_event_cannot_be_acknowledged(self, mock_notify):
        rule = make_rule()
        event = make_event(rule, status=AlertStatus.RESOLVED)

        with self.assertRaises(InvalidTransitionError):
            acknowledge_event(event, actor="system")

        event.refresh_from_db()
        self.assertEqual(event.status, AlertStatus.RESOLVED)
        mock_notify.assert_not_called()

    @patch("alerts.services.event_service.notify_clients")
    def test_acknowledge_writes_history(self, _):
        rule = make_rule()
        event = make_event(rule)

        acknowledge_event(event, actor="nurse1", note="taking action")

        history = AlertEventHistory.objects.filter(event=event, action="acknowledged")
        self.assertEqual(history.count(), 1)
        self.assertEqual(history.first().actor, "nurse1")
        self.assertEqual(history.first().note, "taking action")


class TestResolveEvent(TestCase):

    @patch("alerts.services.event_service.notify_clients")
    def test_open_event_resolves(self, mock_notify):
        rule = make_rule()
        event = make_event(rule, status=AlertStatus.OPEN)

        resolve_event(event, actor="system")

        event.refresh_from_db()
        self.assertEqual(event.status, AlertStatus.RESOLVED)
        self.assertIsNotNone(event.resolved_at)
        mock_notify.assert_called_once()

    @patch("alerts.services.event_service.notify_clients")
    def test_acknowledged_event_resolves(self, mock_notify):
        rule = make_rule()
        event = make_event(rule, status=AlertStatus.ACKNOWLEDGED)

        resolve_event(event, actor="system")

        event.refresh_from_db()
        self.assertEqual(event.status, AlertStatus.RESOLVED)

    @patch("alerts.services.event_service.notify_clients")
    def test_resolve_keeps_existing_severity(self, mock_notify):
        rule = make_rule()
        event = make_event(rule, status=AlertStatus.OPEN, severity=AlertSeverity.CRITICAL)

        resolve_event(event)

        event.refresh_from_db()
        self.assertEqual(event.severity, AlertSeverity.CRITICAL)

    @patch("alerts.services.event_service.notify_clients")
    def test_resolve_writes_history(self, _):
        rule = make_rule()
        event = make_event(rule)

        resolve_event(event, note="manually resolved")

        history = AlertEventHistory.objects.filter(event=event, action="resolved")
        self.assertEqual(history.count(), 1)
        self.assertEqual(history.first().note, "manually resolved")


class TestEscalateEvent(TestCase):

    @patch("alerts.services.event_service.notify_clients")
    def test_open_event_escalates_to_critical(self, mock_notify):
        rule = make_rule()
        event = make_event(rule, status=AlertStatus.OPEN, severity=AlertSeverity.WARNING)

        escalate_event(event, actor="system")

        event.refresh_from_db()
        self.assertEqual(event.status, AlertStatus.ESCALATED)
        self.assertEqual(event.severity, AlertSeverity.CRITICAL)
        self.assertIsNotNone(event.escalated_at)
        mock_notify.assert_called_once()

    @patch("alerts.services.event_service.notify_clients")
    def test_resolved_event_is_not_escalated(self, mock_notify):
        rule = make_rule()
        event = make_event(rule, status=AlertStatus.RESOLVED)

        with self.assertRaises(InvalidTransitionError):
            escalate_event(event)

        event.refresh_from_db()
        self.assertEqual(event.status, AlertStatus.RESOLVED)  # unchanged
        mock_notify.assert_not_called()

    @patch("alerts.services.event_service.notify_clients")
    def test_escalate_writes_history(self, _):
        rule = make_rule()
        event = make_event(rule, status=AlertStatus.OPEN)

        escalate_event(event, note="SLA breached")

        history = AlertEventHistory.objects.filter(event=event, action="escalated")
        self.assertEqual(history.count(), 1)
        self.assertEqual(history.first().note, "SLA breached")


class TestFullLifecycle(TestCase):
    """Tests that walk the complete open→ack→escalate→resolve path."""

    @patch("alerts.services.event_service.notify_clients")
    def test_full_lifecycle_history_trail(self, _):
        rule = make_rule()
        event = make_event(rule, status=AlertStatus.OPEN)

        acknowledge_event(event, actor="nurse")
        escalate_event(event, actor="system", note="SLA")
        resolve_event(event, actor="doctor")

        actions = list(
            AlertEventHistory.objects
            .filter(event=event)
            .order_by("created_at")
            .values_list("action", flat=True)
        )
        # "created" doesn't happen here (event was created directly)
        # but ack/escalate/resolve must appear in order
        self.assertIn("acknowledged", actions)
        self.assertIn("escalated", actions)
        self.assertIn("resolved", actions)
        self.assertLess(actions.index("acknowledged"), actions.index("resolved"))

    @patch("alerts.services.event_service.notify_clients")
    def test_notify_called_on_every_transition(self, mock_notify):
        rule = make_rule()
        event = make_event(rule, status=AlertStatus.OPEN)

        acknowledge_event(event)
        resolve_event(event)

        self.assertEqual(mock_notify.call_count, 2)


class TestCreateOrRefreshDedup(TestCase):
    """Tests for dedup behavior in create_or_refresh_event."""

    @patch("alerts.services.event_service.notify_clients")
    def test_same_key_within_window_refreshes_not_creates(self, mock_notify):
        rule = make_rule(dedup_window_minutes=60)

        event1, created1 = create_or_refresh_event(
            rule=rule, event_key="key-001",
            title="T", message="M", context={},
            source_type="test", source_ref="m1",
        )
        self.assertTrue(created1)

        event2, created2 = create_or_refresh_event(
            rule=rule, event_key="key-001",
            title="T updated", message="M", context={},
            source_type="test", source_ref="m1",
        )
        self.assertFalse(created2)
        self.assertEqual(event1.id, event2.id)
        self.assertEqual(AlertEvent.objects.filter(event_key="key-001").count(), 1)

    @patch("alerts.services.event_service.notify_clients")
    def test_resolved_key_creates_new_event(self, mock_notify):
        rule = make_rule()

        event1, _ = create_or_refresh_event(
            rule=rule, event_key="key-002",
            title="T", message="M", context={},
            source_type="test", source_ref="m1",
        )
        event1.status = AlertStatus.RESOLVED
        event1.save(update_fields=["status"])

        event2, created2 = create_or_refresh_event(
            rule=rule, event_key="key-002",
            title="T", message="M", context={},
            source_type="test", source_ref="m1",
        )
        self.assertTrue(created2)
        self.assertNotEqual(event1.id, event2.id)
