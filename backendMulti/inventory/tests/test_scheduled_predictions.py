import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from django.test import SimpleTestCase, override_settings

from inventory import tasks


class ScheduledPredictionConfigTests(SimpleTestCase):
    def test_load_config_allows_disabled_alert_without_threshold_operator(self):
        with TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "scheduled_predictions.json"
            path.write_text(
                json.dumps(
                    {
                        "jobs": [
                            {
                                "id": "forecast_without_alerts",
                                "enabled": True,
                                "model_id": "forecast_model",
                                "entity_source": "a.b",
                                "feature_builder": "c.d",
                                "result": {},
                                "alert": {"enabled": False},
                            }
                        ]
                    }
                ),
                encoding="utf-8",
            )

            with override_settings(PIOS_SCHEDULED_PREDICTION_CONFIG_PATH=path):
                payload = tasks._load_config()

        self.assertEqual(payload["jobs"][0]["id"], "forecast_without_alerts")

    def test_load_config_rejects_enabled_alert_without_threshold(self):
        with TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "scheduled_predictions.json"
            path.write_text(
                json.dumps(
                    {
                        "jobs": [
                            {
                                "id": "bad_alert",
                                "enabled": True,
                                "model_id": "forecast_model",
                                "entity_source": "a.b",
                                "feature_builder": "c.d",
                                "result": {},
                                "alert": {"operator": ">="},
                            }
                        ]
                    }
                ),
                encoding="utf-8",
            )

            with override_settings(PIOS_SCHEDULED_PREDICTION_CONFIG_PATH=path):
                with self.assertRaisesRegex(RuntimeError, "alert is missing threshold"):
                    tasks._load_config()

    def test_load_config_rejects_non_boolean_hospital_scope_flag(self):
        with TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "scheduled_predictions.json"
            path.write_text(
                json.dumps(
                    {
                        "jobs": [
                            {
                                "id": "bad_scope_flag",
                                "enabled": True,
                                "model_id": "forecast_model",
                                "entity_source": "a.b",
                                "feature_builder": "c.d",
                                "result": {},
                                "alert": {
                                    "operator": ">=",
                                    "threshold": 0.5,
                                    "requires_hospital_scope": "yes",
                                },
                            }
                        ]
                    }
                ),
                encoding="utf-8",
            )

            with override_settings(PIOS_SCHEDULED_PREDICTION_CONFIG_PATH=path):
                with self.assertRaisesRegex(RuntimeError, "requires_hospital_scope"):
                    tasks._load_config()

    def test_load_config_rejects_enabled_forecast_cap_without_factors(self):
        with TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "scheduled_predictions.json"
            path.write_text(
                json.dumps(
                    {
                        "dashboard_forecast": {
                            "cap_policy": {
                                "enabled": True,
                                "default_max_factor": 2.0,
                                "min_floor_units": 50.0,
                            }
                        },
                        "jobs": [],
                    }
                ),
                encoding="utf-8",
            )

            with override_settings(PIOS_SCHEDULED_PREDICTION_CONFIG_PATH=path):
                with self.assertRaisesRegex(RuntimeError, "horizon_max_factors"):
                    tasks._load_config()

    def test_configured_dashboard_forecast_cap_policy_comes_from_config(self):
        with TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "scheduled_predictions.json"
            path.write_text(
                json.dumps(
                    {
                        "dashboard_forecast": {
                            "cap_policy": {
                                "enabled": True,
                                "default_max_factor": 2.0,
                                "min_floor_units": 50.0,
                                "horizon_max_factors": {"t7": 4.0},
                            }
                        },
                        "jobs": [],
                    }
                ),
                encoding="utf-8",
            )

            with override_settings(PIOS_SCHEDULED_PREDICTION_CONFIG_PATH=path):
                policy = tasks.configured_dashboard_forecast_cap_policy()

        self.assertEqual(
            policy,
            {
                "enabled": True,
                "default_max_factor": 2.0,
                "min_floor_units": 50.0,
                "horizon_max_factors": {"t7": 4.0},
            },
        )

    def test_load_config_rejects_duplicate_job_ids(self):
        with TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "scheduled_predictions.json"
            path.write_text(
                json.dumps(
                    {
                        "jobs": [
                            {
                                "id": "same",
                                "enabled": True,
                                "model_id": "model_a",
                                "entity_source": "a.b",
                                "feature_builder": "c.d",
                                "result": {},
                                "alert": {"operator": ">=", "threshold": 0.5},
                            },
                            {
                                "id": "same",
                                "enabled": True,
                                "model_id": "model_b",
                                "entity_source": "a.b",
                                "feature_builder": "c.d",
                                "result": {},
                                "alert": {"operator": ">=", "threshold": 0.5},
                            },
                        ]
                    }
                ),
                encoding="utf-8",
            )

            with override_settings(PIOS_SCHEDULED_PREDICTION_CONFIG_PATH=path):
                with self.assertRaisesRegex(RuntimeError, "Duplicate"):
                    tasks._load_config()


class ScheduledPredictionExecutorTests(SimpleTestCase):
    def test_multioutput_prediction_path_saves_configured_inventory_risk(self):
        row = SimpleNamespace(
            supply_id="S001ABN",
            current_stock_units=3.44,
            usage_today=2.32,
            blood_product_type="AB-",
            hospital=SimpleNamespace(hospital_id="H001"),
        )
        payload = {
            "jobs": [
                {
                    "id": "shadow_component_inventory_risk",
                    "enabled": True,
                    "model_id": "component_inventory_risk_simulator",
                    "result_model_name": "component_inventory_risk_simulator_shadow",
                    "model_version": "challenger",
                    "entity_source": "configured.entity_source",
                    "feature_builder": "configured.feature_builder",
                    "prediction_value": {"path": "0.p_stockout_0_7d"},
                    "result": {
                        "entity_id_attr": "supply_id",
                        "entity_type": "supply",
                        "hospital_id_attr": "hospital.hospital_id",
                        "blood_type_attr": "blood_product_type",
                    },
                    "alert": {"operator": ">=", "threshold": 0.5, "requires_hospital_scope": True},
                    "alert_context": {
                        "current_stock_units_attr": "current_stock_units",
                        "usage_today_attr": "usage_today",
                    },
                }
            ]
        }
        fake_model = MagicMock()
        fake_model.predict.return_value = [{"p_stockout_0_7d": 0.61}]

        def import_side_effect(path):
            if path == "configured.entity_source":
                return lambda: [row]
            if path == "configured.feature_builder":
                return lambda entity: {"frame": entity.supply_id}
            raise AssertionError(path)

        with patch("inventory.tasks._load_config", return_value=payload), patch(
            "inventory.tasks._import_callable", side_effect=import_side_effect
        ), patch("inventory.tasks._load_model", return_value=(fake_model, "component_inventory_risk_simulator")), patch(
            "inventory.tasks.PredictionResult.objects.update_or_create"
        ) as mock_save, patch("inventory.tasks._evaluate_scheduled_alerts") as mock_alerts:
            summary = tasks.run_configured_scheduled_predictions(
                job_ids={"shadow_component_inventory_risk"}
            )

        self.assertEqual(summary["jobs"][0]["saved"], 1)
        save_kwargs = mock_save.call_args.kwargs
        self.assertEqual(save_kwargs["entity_id"], "S001ABN")
        self.assertEqual(save_kwargs["model_name"], "component_inventory_risk_simulator_shadow")
        defaults = save_kwargs["defaults"]
        self.assertEqual(defaults["hospital_id"], "H001")
        self.assertEqual(defaults["blood_type"], "AB-")
        self.assertEqual(defaults["predicted_value"], 0.61)
        self.assertTrue(defaults["alert_triggered"])
        alert_kwargs = mock_alerts.call_args.kwargs
        self.assertEqual(alert_kwargs["feature_input"]["entity_id"], "S001ABN")
        self.assertEqual(alert_kwargs["feature_input"]["hospital_id"], "H001")
        self.assertEqual(alert_kwargs["feature_input"]["blood_type"], "AB-")
        self.assertEqual(alert_kwargs["feature_input"]["entity_type"], "supply")
        self.assertTrue(alert_kwargs["feature_input"]["requires_hospital_scope"])
        self.assertEqual(alert_kwargs["feature_input"]["current_stock_units"], 3.44)
        self.assertEqual(alert_kwargs["feature_input"]["usage_today"], 2.32)

    def test_job_predictions_are_batched_per_configured_job(self):
        rows = [
            SimpleNamespace(
                supply_id="S001",
                current_stock_units=3.0,
                blood_product_type="O+",
                hospital=SimpleNamespace(hospital_id="H001"),
            ),
            SimpleNamespace(
                supply_id="S002",
                current_stock_units=8.0,
                blood_product_type="A-",
                hospital=SimpleNamespace(hospital_id="H002"),
            ),
        ]
        payload = {
            "jobs": [
                {
                    "id": "dashboard_sota_stockout_days",
                    "enabled": True,
                    "model_id": "component_inventory_risk_simulator",
                    "result_model_name": "component_inventory_risk_simulator_stockout_days",
                    "model_version": "challenger",
                    "entity_source": "configured.entity_source",
                    "feature_builder": "configured.feature_builder",
                    "prediction_value": {"path": "0.prediction"},
                    "result": {
                        "entity_id_attr": "supply_id",
                        "entity_type": "supply",
                        "hospital_id_attr": "hospital.hospital_id",
                        "blood_type_attr": "blood_product_type",
                    },
                    "alert": {"enabled": False},
                }
            ]
        }
        fake_model = MagicMock()
        fake_model.predict.return_value = [{"prediction": 2.0}, {"prediction": 5.0}]

        def import_side_effect(path):
            if path == "configured.entity_source":
                return lambda: rows
            if path == "configured.feature_builder":
                return lambda entity: {"current_inventory": entity.current_stock_units}
            raise AssertionError(path)

        with patch("inventory.tasks._load_config", return_value=payload), patch(
            "inventory.tasks._import_callable", side_effect=import_side_effect
        ), patch("inventory.tasks._load_model", return_value=(fake_model, "component_inventory_risk_simulator")), patch(
            "inventory.tasks.PredictionResult.objects.update_or_create"
        ) as mock_save, patch("inventory.tasks._evaluate_scheduled_alerts"):
            summary = tasks.run_configured_scheduled_predictions(
                job_ids={"dashboard_sota_stockout_days"}
            )

        self.assertEqual(summary["jobs"][0]["saved"], 2)
        self.assertEqual(fake_model.predict.call_count, 1)
        frame = fake_model.predict.call_args.args[0]
        self.assertEqual(frame["current_inventory"].tolist(), [3.0, 8.0])
        self.assertEqual(mock_save.call_count, 2)
        saved_values = [
            call.kwargs["defaults"]["predicted_value"]
            for call in mock_save.call_args_list
        ]
        self.assertEqual(saved_values, [2.0, 5.0])

    def test_disabled_alert_saves_prediction_without_evaluating_alert_rules(self):
        row = SimpleNamespace(
            supply_id="S001",
            current_stock_units=10.0,
            blood_product_type="O+",
            hospital=SimpleNamespace(hospital_id="H001"),
        )
        payload = {
            "jobs": [
                {
                    "id": "dashboard_forecast_t1",
                    "enabled": True,
                    "model_id": "component_demand_quantile_forecast_model",
                    "result_model_name": "blood_stock_forecast_t1",
                    "model_version": "challenger",
                    "entity_source": "configured.entity_source",
                    "feature_builder": "configured.feature_builder",
                    "feature_mapping": {
                        "current_inventory": {"attr": "current_stock_units", "default": 0},
                    },
                    "prediction_value": {"path": "0.demand_units"},
                    "result": {
                        "entity_id_attr": "supply_id",
                        "entity_type": "supply_forecast_t1",
                        "hospital_id_attr": "hospital.hospital_id",
                        "blood_type_attr": "blood_product_type",
                    },
                    "alert": {"enabled": False},
                }
            ]
        }
        fake_model = MagicMock()
        fake_model.predict.return_value = [{"demand_units": 12.0}]

        def import_side_effect(path):
            if path == "configured.entity_source":
                return lambda: [row]
            if path == "configured.feature_builder":
                return MagicMock()
            raise AssertionError(path)

        with patch("inventory.tasks._load_config", return_value=payload), patch(
            "inventory.tasks._import_callable", side_effect=import_side_effect
        ), patch("inventory.tasks._load_model", return_value=(fake_model, "component_demand_quantile_forecast_model")), patch(
            "inventory.tasks.PredictionResult.objects.update_or_create"
        ) as mock_save, patch("inventory.tasks._evaluate_scheduled_alerts") as mock_alerts:
            summary = tasks.run_configured_scheduled_predictions(
                job_ids={"dashboard_forecast_t1"}
            )

        self.assertEqual(summary["jobs"][0]["saved"], 1)
        defaults = mock_save.call_args.kwargs["defaults"]
        self.assertFalse(defaults["alert_triggered"])
        self.assertIsNone(defaults["alert_threshold_used"])
        mock_alerts.assert_not_called()

    def test_job_feature_mapping_wins_when_callable_is_shared(self):
        row = SimpleNamespace(
            supply_id="S001",
            current_stock_units=10.0,
            usage_today=4.0,
            blood_product_type="O+",
            hospital=SimpleNamespace(hospital_id="H001"),
        )
        payload = {
            "jobs": [
                {
                    "id": "first_job_same_builder",
                    "enabled": True,
                    "model_id": "model_a",
                    "entity_source": "configured.entity_source",
                    "feature_builder": "configured.shared_feature_builder",
                    "feature_mapping": {"first_job_only": {"attr": "current_stock_units"}},
                    "result": {},
                    "alert": {"enabled": False},
                },
                {
                    "id": "second_job_same_builder",
                    "enabled": True,
                    "model_id": "model_b",
                    "entity_source": "configured.entity_source",
                    "feature_builder": "configured.shared_feature_builder",
                    "feature_mapping": {"second_job_only": {"attr": "usage_today"}},
                    "prediction_value": {"path": "0.prediction"},
                    "result": {
                        "entity_id_attr": "supply_id",
                        "entity_type": "supply",
                        "hospital_id_attr": "hospital.hospital_id",
                        "blood_type_attr": "blood_product_type",
                    },
                    "alert": {"enabled": False},
                },
            ]
        }
        fake_model = MagicMock()
        fake_model.predict.return_value = [{"prediction": 1.0}]

        def import_side_effect(path):
            if path == "configured.entity_source":
                return lambda: [row]
            if path == "configured.shared_feature_builder":
                return MagicMock()
            raise AssertionError(path)

        with patch("inventory.tasks._load_config", return_value=payload), patch(
            "inventory.tasks._import_callable", side_effect=import_side_effect
        ), patch("inventory.tasks._load_model", return_value=(fake_model, "model_b")), patch(
            "inventory.tasks.PredictionResult.objects.update_or_create"
        ), patch("inventory.tasks._evaluate_scheduled_alerts"):
            summary = tasks.run_configured_scheduled_predictions(
                job_ids={"second_job_same_builder"}
            )

        self.assertEqual(summary["jobs"][0]["saved"], 1)
        frame = fake_model.predict.call_args.args[0]
        self.assertEqual(list(frame.columns), ["second_job_only"])
        self.assertEqual(frame.iloc[0]["second_job_only"], 4.0)

    def test_demand_forecast_postprocess_can_store_projected_stock(self):
        row = SimpleNamespace(
            supply_id="S001",
            current_stock_units=20.0,
            blood_product_type="O+",
            hospital=SimpleNamespace(hospital_id="H001"),
        )
        payload = {
            "jobs": [
                {
                    "id": "dashboard_forecast_t7",
                    "enabled": True,
                    "model_id": "component_demand_quantile_forecast_model",
                    "result_model_name": "blood_stock_forecast_t7",
                    "model_version": "challenger",
                    "entity_source": "configured.entity_source",
                    "feature_builder": "configured.feature_builder",
                    "feature_mapping": {
                        "current_inventory": {"attr": "current_stock_units", "default": 0},
                    },
                    "prediction_value": {"path": "0.demand_units"},
                    "postprocess": [
                        {"type": "multiply", "factor": 7},
                        {
                            "type": "subtract_from_attr",
                            "attr": "current_stock_units",
                            "min_value": 0,
                            "round_digits": 2,
                        },
                    ],
                    "result": {
                        "entity_id_attr": "supply_id",
                        "entity_type": "supply_forecast_t7",
                        "hospital_id_attr": "hospital.hospital_id",
                        "blood_type_attr": "blood_product_type",
                    },
                    "alert": {"enabled": False},
                }
            ]
        }
        fake_model = MagicMock()
        fake_model.predict.return_value = [{"demand_units": 2.5}]

        def import_side_effect(path):
            if path == "configured.entity_source":
                return lambda: [row]
            if path == "configured.feature_builder":
                return MagicMock()
            raise AssertionError(path)

        with patch("inventory.tasks._load_config", return_value=payload), patch(
            "inventory.tasks._import_callable", side_effect=import_side_effect
        ), patch("inventory.tasks._load_model", return_value=(fake_model, "component_demand_quantile_forecast_model")), patch(
            "inventory.tasks.PredictionResult.objects.update_or_create"
        ) as mock_save:
            summary = tasks.run_configured_scheduled_predictions(
                job_ids={"dashboard_forecast_t7"}
            )

        self.assertEqual(summary["jobs"][0]["saved"], 1)
        defaults = mock_save.call_args.kwargs["defaults"]
        self.assertEqual(defaults["predicted_value"], 2.5)

    def test_forecast_model_names_are_discovered_from_config_metadata(self):
        payload = {
            "jobs": [
                {
                    "id": "shadow_component_inventory_risk",
                    "enabled": True,
                    "model_id": "component_inventory_risk_simulator",
                    "result": {},
                    "alert": {},
                    "entity_source": "a.b",
                    "feature_builder": "c.d",
                },
                {
                    "id": "dashboard_inventory_risk_t7",
                    "enabled": True,
                    "model_id": "component_inventory_risk_simulator",
                    "result_model_name": "component_inventory_risk_simulator_t7",
                    "dashboard_horizon": "t7",
                    "result": {},
                    "alert": {},
                    "entity_source": "a.b",
                    "feature_builder": "c.d",
                },
                {
                    "id": "dashboard_inventory_risk_t30",
                    "enabled": True,
                    "model_id": "component_inventory_risk_simulator",
                    "result_model_name": "component_inventory_risk_simulator_t30",
                    "dashboard_horizon": "t30",
                    "result": {},
                    "alert": {},
                    "entity_source": "a.b",
                    "feature_builder": "c.d",
                },
            ]
        }

        with patch("inventory.tasks._load_config", return_value=payload):
            result = tasks.configured_forecast_result_models()

        self.assertEqual(
            result,
            {
                "t7": "component_inventory_risk_simulator_t7",
                "t30": "component_inventory_risk_simulator_t30",
            },
        )

    def test_stockout_days_model_names_are_discovered_from_config_metadata(self):
        payload = {
            "jobs": [
                {
                    "id": "dashboard_sota_stockout_days",
                    "enabled": True,
                    "dashboard_role": "stockout_days",
                    "model_id": "component_inventory_risk_simulator",
                    "result_model_name": "component_inventory_risk_simulator_stockout_days",
                    "result": {},
                    "alert": {},
                    "entity_source": "a.b",
                    "feature_builder": "c.d",
                },
                {
                    "id": "disabled_stockout_days",
                    "enabled": False,
                    "dashboard_role": "stockout_days",
                    "model_id": "unused",
                    "result": {},
                    "alert": {},
                    "entity_source": "a.b",
                    "feature_builder": "c.d",
                },
            ]
        }

        with patch("inventory.tasks._load_config", return_value=payload):
            result = tasks.configured_stockout_days_result_models()

        self.assertEqual(result, ["component_inventory_risk_simulator_stockout_days"])

    def test_forecast_job_metadata_keeps_horizon_types(self):
        payload = {
            "jobs": [
                {
                    "id": "dashboard_inventory_risk_t7",
                    "enabled": True,
                    "model_id": "component_inventory_risk_simulator",
                    "result_model_name": "component_inventory_risk_simulator_t7",
                    "model_version": "challenger",
                    "dashboard_horizon": "t7",
                    "forecast_output_type": "probability",
                    "result": {},
                    "alert": {},
                    "entity_source": "a.b",
                    "feature_builder": "c.d",
                },
                {
                    "id": "dashboard_inventory_risk_t30",
                    "enabled": True,
                    "model_id": "component_inventory_risk_simulator",
                    "result_model_name": "component_inventory_risk_simulator_t30",
                    "model_version": "challenger",
                    "dashboard_horizon": "t30",
                    "forecast_output_type": "probability",
                    "result": {},
                    "alert": {},
                    "entity_source": "a.b",
                    "feature_builder": "c.d",
                },
            ]
        }

        with patch("inventory.tasks._load_config", return_value=payload):
            result = tasks.configured_dashboard_forecast_jobs()

        self.assertEqual(
            result,
            {
                "t7": {
                    "result_model_name": "component_inventory_risk_simulator_t7",
                    "horizon_type": "probability",
                    "source_model_id": "component_inventory_risk_simulator",
                    "model_version": "challenger",
                    "job_id": "dashboard_inventory_risk_t7",
                },
                "t30": {
                    "result_model_name": "component_inventory_risk_simulator_t30",
                    "horizon_type": "probability",
                    "source_model_id": "component_inventory_risk_simulator",
                    "model_version": "challenger",
                    "job_id": "dashboard_inventory_risk_t30",
                },
            },
        )

    def test_preload_configured_models_loads_each_distinct_model_once(self):
        payload = {
            "jobs": [
                {
                    "id": "dashboard_inventory_risk_t7",
                    "enabled": True,
                    "model_id": "component_inventory_risk_simulator",
                    "model_version": "challenger",
                    "result": {},
                    "alert": {},
                    "entity_source": "a.b",
                    "feature_builder": "c.d",
                },
                {
                    "id": "dashboard_inventory_risk_t30",
                    "enabled": True,
                    "model_id": "component_inventory_risk_simulator",
                    "model_version": "challenger",
                    "result": {},
                    "alert": {},
                    "entity_source": "a.b",
                    "feature_builder": "c.d",
                },
                {
                    "id": "shadow_stockout_hazard_sentinel",
                    "enabled": True,
                    "model_id": "stockout_time_to_event_hazard_model",
                    "model_version": "challenger",
                    "result": {},
                    "alert": {},
                    "entity_source": "a.b",
                    "feature_builder": "c.d",
                },
            ]
        }

        with patch("inventory.tasks._load_config", return_value=payload), patch(
            "inventory.tasks._load_model",
            side_effect=[(object(), "component_inventory_risk_simulator"), (object(), "stockout_time_to_event_hazard_model")],
        ) as mock_load_model:
            result = tasks.preload_configured_scheduled_models()

        self.assertEqual(mock_load_model.call_count, 2)
        self.assertEqual(
            {(row["model_id"], row["status"]) for row in result["models"]},
            {
                ("component_inventory_risk_simulator", "ok"),
                ("stockout_time_to_event_hazard_model", "ok"),
            },
        )
