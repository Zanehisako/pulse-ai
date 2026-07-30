import os
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import django
from django.apps import apps
from django.test import SimpleTestCase

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "backendMulti.settings")
if not apps.ready:
    django.setup()

from ml.core.feature_store import ResolvedFeaturePayload
from ml.core.hybrid_stockout import (
    StockoutHorizonDefinition,
    _probability_from_days_for_horizon,
    build_hybrid_stockout_batch_prediction,
    build_hybrid_stockout_prediction,
    find_hybrid_stockout_regression_runtime,
    validate_stockout_hybrid_config,
)


def _runtime(model_id, defaults=None):
    return SimpleNamespace(
        model_id=model_id,
        aliases=[model_id],
        status="loaded",
        load_error=None,
        defaults=defaults or {},
    )


class _Registry:
    def __init__(self, models):
        self.models = {model.model_id: model for model in models}

    def get(self, model_id):
        return self.models.get(model_id)

    def loaded(self):
        return list(self.models.values())


class HybridStockoutPredictionTests(SimpleTestCase):
    def test_builds_hybrid_response_from_configured_horizon_classifiers(self):
        hybrid_config = {
            "enabled": True,
            "confidence": "medium",
            "horizons": [
                {
                    "key": "1_7_days",
                    "label": "1-7 days",
                    "min_days": 0,
                    "max_days": 7,
                    "output_key": "risk_1_7_days",
                    "classifier_model_id": "risk_1_7_model",
                },
                {
                    "key": "8_30_days",
                    "label": "8-30 days",
                    "min_days": 8,
                    "max_days": 30,
                    "output_key": "risk_8_30_days",
                    "classifier_model_id": "risk_8_30_model",
                },
                {
                    "key": "30_plus_days",
                    "label": "30+ days",
                    "min_days": 31,
                    "max_days": None,
                    "output_key": "risk_30_plus_days",
                    "classifier_model_id": "risk_30_plus_model",
                },
            ],
            "risk_levels": [
                {
                    "level": "critical",
                    "min_probability": 0.7,
                    "max_estimated_days": 7,
                    "recommended_action": "Urgent donor campaign or transfer request",
                },
                {
                    "level": "low",
                    "min_probability": 0,
                    "recommended_action": "Continue routine monitoring",
                },
            ],
        }
        regressor = _runtime(
            "component_inventory_risk_simulator",
            defaults={
                "output": {"name": "days_until_stockout", "task_type": "regression"},
                "hybrid_stockout": hybrid_config,
            },
        )
        registry = _Registry(
            [
                regressor,
                _runtime("risk_1_7_model"),
                _runtime("risk_8_30_model"),
                _runtime("risk_30_plus_model"),
            ]
        )

        probabilities = {
            "risk_1_7_model": {"probability": 0.78},
            "risk_8_30_model": {"probability": 0.18},
            "risk_30_plus_model": {"probability": 0.04},
        }

        def runner(runtime, _features):
            if runtime.model_id == regressor.model_id:
                return {"prediction": 5, "days_until_stockout": 5}
            return probabilities[runtime.model_id]

        def resolver(_runtime, features, **_kwargs):
            return ResolvedFeaturePayload(features=dict(features), metadata={})

        result = build_hybrid_stockout_prediction(
            registry=registry,
            regression_runtime=regressor,
            request_features={
                "blood_component": "RBC",
                "blood_group": "O-",
                "location": "Hospital A",
            },
            prediction_runner=runner,
            feature_resolver=resolver,
        )

        self.assertEqual(result["blood_component"], "RBC")
        self.assertEqual(result["blood_group"], "O-")
        self.assertEqual(result["location"], "Hospital A")
        self.assertEqual(result["estimated_days_until_stockout"], 5)
        self.assertEqual(result["risk_1_7_days"], 0.78)
        self.assertEqual(result["risk_8_30_days"], 0.18)
        self.assertEqual(result["risk_30_plus_days"], 0.04)
        self.assertEqual(result["stockout_probability"], 0.78)
        self.assertEqual(result["horizon"], "1-7 days")
        self.assertEqual(result["risk_level"], "critical")
        self.assertEqual(
            result["recommended_action"],
            "Urgent donor campaign or transfer request",
        )
        self.assertEqual(result["confidence"], "medium")

    def test_horizon_definitions_are_read_from_runtime_config(self):
        regressor = _runtime(
            "custom_stockout_regressor",
            defaults={
                "output": {"name": "days_until_stockout", "task_type": "regression"},
                "hybrid_stockout": {
                    "enabled": True,
                    "horizons": [
                        {
                            "key": "0_3_days",
                            "label": "0-3 days",
                            "min_days": 0,
                            "max_days": 3,
                            "output_key": "risk_0_3_days",
                            "classifier_model_id": "risk_0_3_model",
                        }
                    ],
                    "risk_levels": [
                        {
                            "level": "high",
                            "min_probability": 0.5,
                            "recommended_action": "Immediate review",
                        }
                    ],
                },
            },
        )
        classifier = _runtime("risk_0_3_model")
        registry = _Registry([regressor, classifier])

        def runner(runtime, _features):
            if runtime.model_id == regressor.model_id:
                return {"prediction": 2.2}
            return {"stockout_probability": 0.61}

        def resolver(_runtime, features, **_kwargs):
            return ResolvedFeaturePayload(features=dict(features), metadata={})

        result = build_hybrid_stockout_prediction(
            registry=registry,
            regression_runtime=regressor,
            request_features={"blood_type": "A+"},
            prediction_runner=runner,
            feature_resolver=resolver,
        )

        self.assertEqual(result["blood_group"], "A+")
        self.assertEqual(result["risk_0_3_days"], 0.61)
        self.assertEqual(result["horizon"], "0-3 days")
        self.assertEqual(result["risk_level"], "high")
        self.assertEqual(result["recommended_action"], "Immediate review")

    def test_nested_horizon_models_and_regression_model_are_config_driven(self):
        base_runtime = _runtime(
            "stockout_tool_runtime",
            defaults={
                "output": {"name": "days_until_stockout", "task_type": "regression"},
                "hybrid_stockout": {
                    "enabled": True,
                    "regression": {
                        "type": "model",
                        "model_id": "trained_days_regressor",
                    },
                    "horizons": [
                        {
                            "key": "0_7_days",
                            "label": "0-7 days",
                            "min_days": 0,
                            "max_days": 7,
                            "output_key": "risk_0_7_days",
                            "classifier": {
                                "type": "model",
                                "model_ids": [
                                    "short_horizon_xgb",
                                    "short_horizon_gru",
                                ],
                                "aggregation": "mean",
                            },
                        }
                    ],
                    "risk_levels": [
                        {
                            "level": "high",
                            "min_probability": 0,
                            "recommended_action": "Review",
                        }
                    ],
                },
            },
        )
        registry = _Registry(
            [
                base_runtime,
                _runtime("trained_days_regressor"),
                _runtime("short_horizon_xgb"),
                _runtime("short_horizon_gru"),
            ]
        )
        called_model_ids = []

        def runner(runtime, _features):
            called_model_ids.append(runtime.model_id)
            if runtime.model_id == "trained_days_regressor":
                return {"prediction": 4}
            if runtime.model_id == "short_horizon_xgb":
                return {"probability": 0.2}
            if runtime.model_id == "short_horizon_gru":
                return {"probability": 0.6}
            return {}

        def resolver(_runtime, features, **_kwargs):
            return ResolvedFeaturePayload(features=dict(features), metadata={})

        result = build_hybrid_stockout_prediction(
            registry=registry,
            regression_runtime=base_runtime,
            request_features={"hospital_id": "H001", "blood_type": "O-"},
            prediction_runner=runner,
            feature_resolver=resolver,
        )

        self.assertEqual(result["regression_model_id"], "trained_days_regressor")
        self.assertEqual(result["estimated_days_until_stockout"], 4)
        self.assertEqual(result["risk_0_7_days"], 0.4)
        self.assertEqual(
            result["horizons"][0]["model_ids"],
            ["short_horizon_xgb", "short_horizon_gru"],
        )
        self.assertEqual(
            called_model_ids,
            ["trained_days_regressor", "short_horizon_xgb", "short_horizon_gru"],
        )

    def test_batch_prediction_accepts_multiple_rows_with_same_config(self):
        regressor = _runtime(
            "trained_days_regressor",
            defaults={
                "output": {"name": "days_until_stockout", "task_type": "regression"},
                "hybrid_stockout": {
                    "enabled": True,
                    "horizons": [
                        {
                            "key": "0_7_days",
                            "label": "0-7 days",
                            "min_days": 0,
                            "max_days": 7,
                            "output_key": "risk_0_7_days",
                            "classifier": {
                                "type": "model",
                                "model_id": "short_horizon_model",
                            },
                        }
                    ],
                    "risk_levels": [
                        {
                            "level": "high",
                            "min_probability": 0,
                            "recommended_action": "Review",
                        }
                    ],
                },
            },
        )
        registry = _Registry([regressor, _runtime("short_horizon_model")])

        def runner(runtime, features):
            if runtime.model_id == regressor.model_id:
                return {"prediction": features["days"]}
            return {"probability": features["probability"]}

        def resolver(_runtime, features, **_kwargs):
            return ResolvedFeaturePayload(features=dict(features), metadata={})

        result = build_hybrid_stockout_batch_prediction(
            registry=registry,
            regression_runtime=regressor,
            request_rows=[
                {"hospital_id": "H001", "days": 2, "probability": 0.8},
                {"hospital_id": "H002", "days": 5, "probability": 0.3},
            ],
            prediction_runner=runner,
            feature_resolver=resolver,
        )

        self.assertEqual(result["query_kind"], "batch_stockout_horizon_predictions")
        self.assertEqual(result["row_count"], 2)
        self.assertEqual([row["location"] for row in result["rows"]], ["H001", "H002"])

    def test_invalid_stockout_hybrid_config_fails_clearly(self):
        with self.assertRaisesRegex(RuntimeError, "output_key is required"):
            validate_stockout_hybrid_config(
                {
                    "hybrid_stockout": {
                        "enabled": True,
                        "horizons": [
                            {
                                "key": "bad",
                                "min_days": 0,
                                "max_days": 7,
                            }
                        ],
                    }
                }
            )

    def test_configured_regression_model_id_is_eligible_without_models_mapping(self):
        runtime = _runtime("new_days_regressor")
        registry = _Registry([runtime])
        payload = {
            "hybrid_stockout": {
                "enabled": True,
                "regression": {
                    "type": "model",
                    "model_id": "new_days_regressor",
                },
                "horizons": [
                    {
                        "key": "0_7_days",
                        "label": "0-7 days",
                        "min_days": 0,
                        "max_days": 7,
                        "output_key": "risk_0_7_days",
                    }
                ],
            }
        }

        with patch("ml.core.hybrid_stockout._load_json_hybrid_config", return_value=payload):
            selected = find_hybrid_stockout_regression_runtime(registry)

        self.assertEqual(selected.model_id, "new_days_regressor")

    def test_stockout_horizon_config_uses_simulator_outputs(self):
        config_path = (
            Path(__file__).resolve().parents[1] / "config" / "stockout_hybrid.json"
        )
        hybrid_config = json.loads(config_path.read_text(encoding="utf-8"))[
            "hybrid_stockout"
        ]
        regressor = _runtime(
            "component_inventory_risk_simulator",
            defaults={
                "output": {"name": "inventory_risk", "task_type": "multi_output"},
                "hybrid_stockout": hybrid_config,
            },
        )
        registry = _Registry([regressor])

        called: list[str] = []

        def runner(runtime, _features):
            called.append(runtime.model_id)
            if runtime.model_id == "component_inventory_risk_simulator":
                return {
                    "expected_days_until_stockout": 20,
                    "p_stockout_0_7d": 0.12,
                    "p_stockout_0_30d": 0.61,
                    "p_stockout_0_90d": 0.82,
                    "p_stockout_0_180d": 0.9,
                }
            return {}

        def resolver(_runtime, features, **_kwargs):
            return ResolvedFeaturePayload(features=dict(features), metadata={})

        medium = build_hybrid_stockout_prediction(
            registry=registry,
            regression_runtime=regressor,
            request_features={
                "hospital_id": "H001",
                "blood_type": "O+",
                "component_type": "RBC",
                "prediction_horizon_days": 20,
            },
            prediction_runner=runner,
            feature_resolver=resolver,
        )
        longer = build_hybrid_stockout_prediction(
            registry=registry,
            regression_runtime=regressor,
            request_features={
                "hospital_id": "H001",
                "blood_type": "O+",
                "component_type": "RBC",
                "prediction_horizon_days": 45,
            },
            prediction_runner=runner,
            feature_resolver=resolver,
        )

        self.assertEqual(medium["horizon"], "8-30 days")
        self.assertEqual(medium["p_stockout_0_30d"], 0.61)
        self.assertEqual(longer["horizon"], "31-90 days")
        self.assertEqual(longer["p_stockout_0_90d"], 0.82)
        self.assertEqual(called, ["component_inventory_risk_simulator", "component_inventory_risk_simulator"])

    def test_blood_group_fanout_builds_component_group_scenarios(self):
        regressor = _runtime(
            "configured_inventory_runway_runtime",
            defaults={
                "output": {"name": "days_until_stockout", "task_type": "regression"},
                "hybrid_stockout": {
                    "enabled": True,
                    "confidence": "medium",
                    "regression": {
                        "type": "inventory_runway",
                        "inventory_features": ["current_inventory"],
                    },
                    "component_fanout": {
                        "enabled": True,
                        "component_key": "blood_type",
                        "request_keys": ["blood_type"],
                        "components": ["RBC", "PLASMA"],
                    },
                    "blood_group_fanout": {
                        "enabled": True,
                        "groups": ["O-", "A+"],
                        "inventory_factors": {"O-": 0.1, "A+": 0.5},
                    },
                    "horizons": [
                        {
                            "key": "1_7_days",
                            "label": "1-7 days",
                            "min_days": 0,
                            "max_days": 7,
                            "output_key": "risk_1_7_days",
                            "classifier": {"type": "inventory_pressure"},
                        }
                    ],
                    "risk_levels": [
                        {
                            "level": "high",
                            "min_probability": 0,
                            "recommended_action": "Review",
                        }
                    ],
                },
            },
        )
        registry = _Registry([regressor])

        def resolver(_runtime, features, **_kwargs):
            inventory = 10 if features["blood_type"] == "RBC" else 4
            return ResolvedFeaturePayload(
                features={**features, "current_inventory": inventory, "units_used": 2},
                metadata={},
            )

        result = build_hybrid_stockout_prediction(
            registry=registry,
            regression_runtime=regressor,
            request_features={"hospital_id": "H001"},
            feature_resolver=resolver,
        )

        rows = result["component_predictions"]
        self.assertEqual(len(rows), 4)
        self.assertEqual(
            {(row["blood_component"], row["blood_group"]) for row in rows},
            {("RBC", "O-"), ("RBC", "A+"), ("PLASMA", "O-"), ("PLASMA", "A+")},
        )
        self.assertIn("worst_blood_group", result)


class ProbabilityFromDaysForHorizonTests(SimpleTestCase):
    def _horizon(self, min_days, max_days, min_p=0.3, max_p=0.9):
        return StockoutHorizonDefinition(
            key="test",
            label="test",
            output_key="risk_test",
            min_days=min_days,
            max_days=max_days,
            timeframe="short",
            model_ids=(),
            classifier_config={"min_probability": min_p, "max_probability": max_p},
        )

    def test_days_within_horizon_interpolates_from_max_to_min(self):
        horizon = self._horizon(min_days=0, max_days=7, min_p=0.55, max_p=0.92)
        at_zero = _probability_from_days_for_horizon(0, horizon)
        at_three = _probability_from_days_for_horizon(3, horizon)
        at_seven = _probability_from_days_for_horizon(7, horizon)
        self.assertAlmostEqual(at_zero, 0.92, places=2)
        self.assertAlmostEqual(at_seven, 0.55, places=2)
        self.assertGreater(at_zero, at_three)
        self.assertGreater(at_three, at_seven)

    def test_different_hospitals_get_different_scores(self):
        short_horizon = self._horizon(min_days=0, max_days=7, min_p=0.55, max_p=0.92)
        medium_horizon = self._horizon(min_days=8, max_days=30, min_p=0.30, max_p=0.75)
        h001_score = _probability_from_days_for_horizon(0, short_horizon)
        h002_score = _probability_from_days_for_horizon(23, medium_horizon)
        self.assertGreater(h001_score, h002_score)
        self.assertNotEqual(h001_score, h002_score)

    def test_mid_range_days_gets_mid_probability(self):
        horizon = self._horizon(min_days=0, max_days=10, min_p=0.4, max_p=0.9)
        mid = _probability_from_days_for_horizon(5, horizon)
        self.assertAlmostEqual(mid, 0.65, places=2)

    def test_days_before_horizon_uses_exponential_falloff(self):
        horizon = self._horizon(min_days=8, max_days=30, min_p=0.3, max_p=0.75)
        prob = _probability_from_days_for_horizon(2, horizon)
        self.assertGreater(prob, 0.0)
        self.assertLess(prob, 0.5)

    def test_days_after_horizon_uses_exponential_falloff(self):
        horizon = self._horizon(min_days=0, max_days=7, min_p=0.55, max_p=0.92)
        prob = _probability_from_days_for_horizon(50, horizon)
        self.assertGreater(prob, 0.0)
        self.assertLess(prob, 0.3)

    def test_open_ended_horizon_uses_max_probability(self):
        horizon = StockoutHorizonDefinition(
            key="30_plus",
            label="30+ days",
            output_key="risk_30_plus_days",
            min_days=31,
            max_days=None,
            timeframe="long",
            model_ids=(),
            classifier_config={"min_probability": 0.05, "max_probability": 0.45},
        )
        prob = _probability_from_days_for_horizon(45, horizon)
        self.assertGreater(prob, 0.05)
        self.assertLessEqual(prob, 0.45)

    def test_config_bounds_not_set_uses_defaults(self):
        horizon = StockoutHorizonDefinition(
            key="test",
            label="test",
            output_key="risk_test",
            min_days=0,
            max_days=7,
            timeframe="short",
            model_ids=(),
            classifier_config={},
        )
        prob = _probability_from_days_for_horizon(0, horizon)
        self.assertGreater(prob, 0.5)
        self.assertLessEqual(prob, 1.0)

    def test_real_stockout_hybrid_config_graduates_correctly(self):
        config_path = (
            Path(__file__).resolve().parents[1] / "config" / "stockout_hybrid.json"
        )
        config = json.loads(config_path.read_text(encoding="utf-8"))[
            "hybrid_stockout"
        ]
        short_cfg = config["horizons"][0]
        medium_cfg = config["horizons"][1]
        self.assertEqual(config["regression"]["model_id"], "component_inventory_risk_simulator")
        self.assertEqual(short_cfg["output_key"], "p_stockout_0_7d")
        self.assertEqual(medium_cfg["output_key"], "p_stockout_0_30d")
        self.assertEqual(config["horizons"][2]["output_key"], "p_stockout_0_90d")
        self.assertEqual(config["horizons"][3]["output_key"], "p_stockout_0_180d")
