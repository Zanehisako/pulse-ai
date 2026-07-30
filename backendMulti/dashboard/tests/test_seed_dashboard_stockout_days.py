from unittest.mock import patch

from django.test import SimpleTestCase

from dashboard.management.commands.seed_dashboard_from_inventory import (
    build_forecast,
    build_stockout_days,
)

class _FakePredictionRows(list):
    def exists(self):
        return bool(self)


class _FakePredictionQuerySet:
    def __init__(self, rows):
        self._rows = rows

    def order_by(self, *_args):
        return self

    def values(self, *_args):
        return _FakePredictionRows(self._rows)


class _FakeSupplyQuerySet:
    def __init__(self, values):
        self._values = values

    def values_list(self, *_args, **_kwargs):
        return self._values

    def values(self, *_args, **_kwargs):
        return self

    def annotate(self, *_args, **_kwargs):
        return self._values


class BuildStockoutDaysTests(SimpleTestCase):
    def _mock_supply_filter(self, kwargs):
        if "hospital__wilaya" in kwargs:
            return _FakeSupplyQuerySet(["S-1", "S-2"])
        if "supply_id__in" in kwargs:
            return _FakeSupplyQuerySet([("S-1", "A+"), ("S-2", "A+")])
        raise AssertionError(f"Unexpected BloodSupply filter kwargs: {kwargs}")

    @patch("dashboard.management.commands.seed_dashboard_from_inventory.configured_stockout_days_result_models")
    @patch("dashboard.management.commands.seed_dashboard_from_inventory.PredictionResult.objects.filter")
    @patch("dashboard.management.commands.seed_dashboard_from_inventory.BloodSupply.objects.filter")
    def test_uses_latest_available_predictions_when_today_missing(self, mock_supply_filter, mock_prediction_filter, mock_stockout_models):
        mock_stockout_models.return_value = ["component_inventory_risk_simulator_stockout_days"]
        mock_supply_filter.side_effect = lambda **kwargs: self._mock_supply_filter(kwargs)
        mock_prediction_filter.return_value = _FakePredictionQuerySet(
            [
                {"entity_id": "S-1", "predicted_value": 6.25},
                {"entity_id": "S-2", "predicted_value": 8.0},
            ]
        )

        result = build_stockout_days("quebec")

        self.assertEqual(result, {"A+": 6.2})

    @patch("dashboard.management.commands.seed_dashboard_from_inventory.configured_stockout_days_result_models")
    @patch("dashboard.management.commands.seed_dashboard_from_inventory.PredictionResult.objects.filter")
    @patch("dashboard.management.commands.seed_dashboard_from_inventory.BloodSupply.objects.filter")
    def test_uses_latest_prediction_per_supply(self, mock_supply_filter, mock_prediction_filter, mock_stockout_models):
        mock_stockout_models.return_value = ["component_inventory_risk_simulator_stockout_days"]
        mock_supply_filter.side_effect = lambda **kwargs: self._mock_supply_filter(kwargs)
        mock_prediction_filter.return_value = _FakePredictionQuerySet(
            [
                {"entity_id": "S-1", "predicted_value": 4.4},
                {"entity_id": "S-1", "predicted_value": 9.1},
                {"entity_id": "S-2", "predicted_value": 7.3},
            ]
        )

        result = build_stockout_days("quebec")

        self.assertEqual(result, {"A+": 4.4})

    @patch("dashboard.management.commands.seed_dashboard_from_inventory.configured_stockout_days_result_models")
    def test_returns_none_when_no_stockout_days_models_configured(self, mock_stockout_models):
        mock_stockout_models.return_value = []

        self.assertIsNone(build_stockout_days("quebec"))


class BuildForecastTests(SimpleTestCase):
    @patch("dashboard.management.commands.seed_dashboard_from_inventory.configured_dashboard_forecast_cap_policy")
    @patch("dashboard.management.commands.seed_dashboard_from_inventory.configured_dashboard_forecast_jobs")
    @patch("dashboard.management.commands.seed_dashboard_from_inventory.PredictionResult.objects.filter")
    @patch("dashboard.management.commands.seed_dashboard_from_inventory.BloodSupply.objects.filter")
    def test_aggregates_latest_forecast_predictions_per_blood_type_and_horizon(
        self,
        mock_supply_filter,
        mock_prediction_filter,
        mock_forecast_models,
        mock_cap_policy,
    ):
        mock_cap_policy.return_value = {"enabled": False}
        mock_forecast_models.return_value = {
            "t1": {"result_model_name": "blood_stock_forecast_t1", "horizon_type": "absolute"},
            "t7": {"result_model_name": "blood_stock_forecast_t7", "horizon_type": "delta"},
            "t30": {"result_model_name": "blood_stock_forecast_t30", "horizon_type": "delta"},
        }
        mock_supply_filter.side_effect = [
            _FakeSupplyQuerySet([("S-1", "A+"), ("S-2", "A+")]),
            _FakeSupplyQuerySet([
                {"blood_product_type": "A+", "total_units": 160.0},
            ]),
        ]
        mock_prediction_filter.return_value = _FakePredictionQuerySet(
            [
                {"entity_id": "S-1", "model_name": "blood_stock_forecast_t1", "predicted_value": 100.0},
                {"entity_id": "S-1", "model_name": "blood_stock_forecast_t1", "predicted_value": 90.0},
                {"entity_id": "S-1", "model_name": "blood_stock_forecast_t7", "predicted_value": -70.0},
                {"entity_id": "S-2", "model_name": "blood_stock_forecast_t1", "predicted_value": 60.0},
                {"entity_id": "S-2", "model_name": "blood_stock_forecast_t30", "predicted_value": -15.0},
            ]
        )

        result = build_forecast("quebec")

        self.assertEqual(
            result["A+"],
            {
                "t1": {"predicted_units": 160.0},
                "t7": {"predicted_units": 90.0},
                "t30": {"predicted_units": 145.0},
            },
        )

    @patch("dashboard.management.commands.seed_dashboard_from_inventory.configured_dashboard_forecast_cap_policy")
    @patch("dashboard.management.commands.seed_dashboard_from_inventory.configured_dashboard_forecast_jobs")
    @patch("dashboard.management.commands.seed_dashboard_from_inventory.PredictionResult.objects.filter")
    @patch("dashboard.management.commands.seed_dashboard_from_inventory.BloodSupply.objects.filter")
    def test_applies_configured_forecast_cap_policy(
        self,
        mock_supply_filter,
        mock_prediction_filter,
        mock_forecast_models,
        mock_cap_policy,
    ):
        mock_cap_policy.return_value = {
            "enabled": True,
            "default_max_factor": 2.0,
            "min_floor_units": 5.0,
            "horizon_max_factors": {"t1": 3.0},
        }
        mock_forecast_models.return_value = {
            "t1": {"result_model_name": "blood_stock_forecast_t1", "horizon_type": "absolute"},
        }
        mock_supply_filter.side_effect = [
            _FakeSupplyQuerySet([("S-1", "A+")]),
            _FakeSupplyQuerySet([
                {"blood_product_type": "A+", "total_units": 10.0},
            ]),
        ]
        mock_prediction_filter.return_value = _FakePredictionQuerySet(
            [
                {"entity_id": "S-1", "model_name": "blood_stock_forecast_t1", "predicted_value": 100.0},
            ]
        )

        result = build_forecast("quebec")

        self.assertEqual(result["A+"]["t1"], {"predicted_units": 30.0})
        self.assertEqual(
            result["O+"],
            {
                "t1": None,
            },
        )
