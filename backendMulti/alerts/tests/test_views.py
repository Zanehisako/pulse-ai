# alerts/tests/test_views.py

from __future__ import annotations
from types import SimpleNamespace
from unittest.mock import patch
from django.test import TestCase
from rest_framework.test import APIClient
from alerts.models import (
    AlertEvent, AlertRule,
    AlertSeverity, AlertStatus, AlertTriggerType,
)


def make_rule(**kwargs):
    defaults = dict(
        name="View Test Rule",
        is_active=True,
        trigger_type=AlertTriggerType.THRESHOLD_GAP,
        severity_base=AlertSeverity.WARNING,
        conditions={"field": "stock", "op": "<", "value": 10},
    )
    defaults.update(kwargs)
    return AlertRule.objects.create(**defaults)


def make_event(rule, status=AlertStatus.OPEN, severity=AlertSeverity.WARNING):
    return AlertEvent.objects.create(
        rule=rule, event_key=f"key-{status}-{severity}",
        title="Test", message="Test message",
        status=status, severity=severity,
    )


def authenticate(client):
    client.force_authenticate(
        user=SimpleNamespace(is_authenticated=True, id=1, pk=1, email="ci@example.com")
    )


class TestAlertSummaryView(TestCase):

    def setUp(self):
        self.client = APIClient()
        authenticate(self.client)
        rule = make_rule(name="Summary Rule")
        make_event(rule, status=AlertStatus.OPEN, severity=AlertSeverity.WARNING)
        make_event(rule, status=AlertStatus.OPEN, severity=AlertSeverity.CRITICAL)
        make_event(rule, status=AlertStatus.ESCALATED, severity=AlertSeverity.CRITICAL)
        make_event(rule, status=AlertStatus.RESOLVED, severity=AlertSeverity.RESOLVED)

    def test_summary_counts_are_correct(self):
        response = self.client.get("/alerts/summary/")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["open"], 2)
        self.assertEqual(data["escalated"], 1)
        self.assertEqual(data["resolved"], 1)
        self.assertEqual(data["critical"], 2)
        self.assertEqual(data["warning"], 1)


class TestAlertListFilters(TestCase):

    def setUp(self):
        self.client = APIClient()
        authenticate(self.client)
        rule = make_rule(name="Filter Rule")
        AlertEvent.objects.create(
            rule=rule, event_key="k1", title="T", message="M",
            status=AlertStatus.OPEN, severity=AlertSeverity.CRITICAL,
            entity_id="H001", blood_type="O+",
        )
        AlertEvent.objects.create(
            rule=rule, event_key="k2", title="T", message="M",
            status=AlertStatus.RESOLVED, severity=AlertSeverity.RESOLVED,
            entity_id="H002", blood_type="A+",
        )

    def test_filter_by_status(self):
        response = self.client.get("/alerts/?status=open")
        self.assertEqual(len(response.json()), 1)
        self.assertEqual(response.json()[0]["entity_id"], "H001")

    def test_filter_by_blood_type(self):
        response = self.client.get("/alerts/", {"blood_type": "O+"})
        self.assertEqual(len(response.json()), 1)

    def test_filter_by_entity_id(self):
        response = self.client.get("/alerts/?entity_id=H001")
        self.assertEqual(len(response.json()), 1)


class TestAlertAcknowledgeView(TestCase):

    def setUp(self):
        self.client = APIClient()
        authenticate(self.client)

    @patch("alerts.services.event_service.notify_clients")
    def test_acknowledge_open_event(self, mock_notify):
        rule = make_rule(name="Ack Rule")
        event = make_event(rule, status=AlertStatus.OPEN)

        response = self.client.post(
            f"/alerts/{event.id}/acknowledge/",
            {"note": "on it"},
            format="json",
        )

        self.assertEqual(response.status_code, 200)
        event.refresh_from_db()
        self.assertEqual(event.status, AlertStatus.ACKNOWLEDGED)

    def test_acknowledge_nonexistent_event_returns_404(self):
        import uuid
        response = self.client.post(f"/alerts/{uuid.uuid4()}/acknowledge/", {})
        self.assertEqual(response.status_code, 404)


class TestAlertResolveView(TestCase):

    def setUp(self):
        self.client = APIClient()
        authenticate(self.client)

    @patch("alerts.services.event_service.notify_clients")
    def test_resolve_event(self, mock_notify):
        rule = make_rule(name="Resolve Rule")
        event = make_event(rule, status=AlertStatus.OPEN)

        response = self.client.post(
            f"/alerts/{event.id}/resolve/",
            {"note": "resolved"},
            format="json",
        )

        self.assertEqual(response.status_code, 200)
        event.refresh_from_db()
        self.assertEqual(event.status, AlertStatus.RESOLVED)


class TestAlertRuleSerializerValidation(TestCase):

    def setUp(self):
        self.client = APIClient()
        authenticate(self.client)

    def test_valid_single_condition_accepted(self):
        response = self.client.post("/alerts/rules/", {
            "name": "Valid Rule",
            "conditions": {"field": "stock", "op": "<", "value": 10},
            "severity_base": "warning",
            "trigger_type": "threshold_gap",
        }, format="json")
        self.assertEqual(response.status_code, 201)

    def test_empty_conditions_rejected(self):
        # Currently passes through — this test will FAIL until you add
        # validate_conditions() to the serializer. That's intentional —
        # it documents the gap.
        response = self.client.post("/alerts/rules/", {
            "name": "Bad Rule",
            "conditions": {},
            "severity_base": "warning",
            "trigger_type": "threshold_gap",
        }, format="json")
        # Should be 400 — currently returns 201 (bug)
        self.assertEqual(response.status_code, 400)
