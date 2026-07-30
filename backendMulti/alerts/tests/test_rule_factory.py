from __future__ import annotations

import json
import os
from unittest.mock import patch

import django

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "backendMulti.settings")
from django.apps import apps

if not apps.ready:
    django.setup()

from django.test import SimpleTestCase, TestCase

from alerts.models import AlertRule
from alerts.services.rule_factory import AlertRuleFactory, _load_templates
from inventory.models import Hospital


class AlertRuleFactoryConfigTests(SimpleTestCase):
    """Unit tests: config loading (no DB)."""

    def test_load_templates_returns_blood_types_and_templates(self):
        config = _load_templates()
        self.assertIn("blood_types", config)
        self.assertIn("templates", config)
        self.assertGreater(len(config["blood_types"]), 0)
        self.assertGreater(len(config["templates"]), 0)

    def test_each_template_has_required_fields(self):
        config = _load_templates()
        for template in config["templates"]:
            self.assertIn("id", template)
            self.assertIn("name_pattern", template)
            self.assertIn("conditions", template)
            self.assertIn("severity_base", template)
            self.assertIn("trigger_type", template)

    def test_name_pattern_contains_placeholders(self):
        config = _load_templates()
        for template in config["templates"]:
            self.assertIn("{hospital_id}", template["name_pattern"])
            self.assertIn("{blood_type}", template["name_pattern"])

    def test_conditions_have_threshold_mode(self):
        config = _load_templates()
        for template in config["templates"]:
            conditions = template["conditions"]
            if "all" in conditions:
                for cond in conditions["all"]:
                    self.assertIn("threshold_mode", cond)
            else:
                self.assertIn("threshold_mode", conditions)

    def test_factory_gracefully_handles_missing_config(self):
        with patch("alerts.services.rule_factory._CONFIG_PATH") as mock_path:
            mock_path.read_text.side_effect = FileNotFoundError("missing")
            result = _load_templates()
        self.assertEqual(result, {"blood_types": [], "templates": []})


class AlertRuleFactoryTests(TestCase):
    """Integration tests: DB-backed rule generation."""

    def test_generates_correct_number_of_rules(self):
        created = AlertRuleFactory().generate_for_hospital(hospital_id="H001")
        config = _load_templates()
        expected = len(config["blood_types"]) * len(config["templates"])
        self.assertEqual(len(created), expected)
        self.assertEqual(AlertRule.objects.filter(scope_ref="H001").count(), expected)

    def test_idempotent_does_not_duplicate(self):
        factory = AlertRuleFactory()
        first = factory.generate_for_hospital(hospital_id="H002")
        second = factory.generate_for_hospital(hospital_id="H002")
        self.assertGreater(len(first), 0)
        self.assertEqual(len(second), 0)
        self.assertEqual(AlertRule.objects.filter(scope_ref="H002").count(), len(first))

    def test_rules_have_dynamic_threshold_mode(self):
        AlertRuleFactory().generate_for_hospital(hospital_id="H003")
        rules = AlertRule.objects.filter(scope_ref="H003")
        for rule in rules:
            conditions = rule.conditions
            if "all" in conditions:
                self.assertTrue(
                    all("threshold_mode" in item for item in conditions["all"]),
                    msg=f"Missing threshold_mode in rule {rule.name}",
                )
            else:
                self.assertIn("threshold_mode", conditions, msg=f"Missing threshold_mode in rule {rule.name}")

    def test_signal_fires_on_hospital_create(self):
        hospital = Hospital.objects.create(hospital_id="H004", name="Signal Test Hospital")
        config = _load_templates()
        expected = len(config["blood_types"]) * len(config["templates"])
        self.assertEqual(AlertRule.objects.filter(scope_ref=hospital.hospital_id).count(), expected)

    def test_custom_blood_types_override_config(self):
        created = AlertRuleFactory().generate_for_hospital(
            hospital_id="H005", blood_types=["O+", "O-"]
        )
        config = _load_templates()
        expected = 2 * len(config["templates"])
        self.assertEqual(len(created), expected)
