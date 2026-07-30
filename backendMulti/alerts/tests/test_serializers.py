# alerts/tests/test_serializers.py

from django.test import SimpleTestCase
from alerts.serializers import AlertRuleSerializer


class TestAlertRuleSerializerConditionsValidation(SimpleTestCase):
    """
    These tests define the DESIRED behavior for conditions validation.
    They will fail until validate_conditions() is added to the serializer.
    """

    def _get_base_data(self, conditions):
        return {
            "name": "Test Rule",
            "severity_base": "warning",
            "trigger_type": "threshold_gap",
            "conditions": conditions,
        }

    def test_valid_single_condition(self):
        s = AlertRuleSerializer(data=self._get_base_data(
            {"field": "stock", "op": "<", "value": 5}
        ))
        self.assertTrue(s.is_valid(), s.errors)

    def test_valid_all_conditions(self):
        s = AlertRuleSerializer(data=self._get_base_data({
            "all": [
                {"field": "stock", "op": "<", "value": 5},
                {"field": "days", "op": "<", "value": 2},
            ]
        }))
        self.assertTrue(s.is_valid(), s.errors)

    def test_empty_dict_is_invalid(self):
        s = AlertRuleSerializer(data=self._get_base_data({}))
        self.assertFalse(s.is_valid())
        self.assertIn("conditions", s.errors)

    def test_single_condition_missing_op_is_invalid(self):
        s = AlertRuleSerializer(data=self._get_base_data(
            {"field": "stock", "value": 5}  # no "op"
        ))
        self.assertFalse(s.is_valid())

    def test_all_with_empty_list_is_invalid(self):
        s = AlertRuleSerializer(data=self._get_base_data({"all": []}))
        self.assertFalse(s.is_valid())