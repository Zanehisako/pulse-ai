import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import MagicMock, patch

import django
import mlflow
import pandas as pd
from django.apps import apps
from django.test import SimpleTestCase

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "backendMulti.settings")
if not apps.ready:
    django.setup()

from ml.core.prediction import (
    _apply_prediction_output_contract,
    _find_local_mlflow_model_dir,
    _get_cached_mlflow_model,
    _get_stats,
    _predict_mlflow,
    _predict_mlflow_subprocess_batch,
    _normalize_mlflow_prediction_output,
    _repair_loaded_model_compatibility,
    flush_mlflow_cache,
)
from ml.core.registry import ModelRuntime


class _MlflowInputSpec:
    def __init__(self, name, type_name):
        self.name = name
        self.type = type_name


class _FakeMlflowModel:
    def __init__(self, input_specs):
        signature = MagicMock(inputs=input_specs)
        self.metadata = MagicMock(signature=signature)
        self.frame = None

    def predict(self, frame):
        self.frame = frame.copy()
        return pd.DataFrame([{"prediction": 0.42}])


class MlflowPredictionFallbackTests(SimpleTestCase):
    def tearDown(self):
        flush_mlflow_cache()

    def test_find_local_mlflow_model_dir_prefers_matching_feature_service(self):
        with TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)

            wrong_run = root / "0" / "run_wrong"
            (wrong_run / "tags").mkdir(parents=True)
            (wrong_run / "artifacts" / "best_donor_model").mkdir(parents=True)
            (wrong_run / "tags" / "feast_feature_service").write_text(
                "hospital_id_service", encoding="utf-8"
            )
            (wrong_run / "artifacts" / "best_donor_model" / "MLmodel").write_text(
                "name: wrong", encoding="utf-8"
            )

            right_run = root / "0" / "run_right"
            (right_run / "tags").mkdir(parents=True)
            (right_run / "artifacts" / "best_donor_model").mkdir(parents=True)
            (right_run / "tags" / "feast_feature_service").write_text(
                "donor_propensity_service", encoding="utf-8"
            )
            expected = right_run / "artifacts" / "best_donor_model" / "MLmodel"
            expected.write_text("name: right", encoding="utf-8")

            with patch("ml.core.prediction.LOCAL_MLFLOW_RUNS_DIR", root):
                result = _find_local_mlflow_model_dir("donor_propensity_model")

            self.assertEqual(result, expected.parent)

    @patch(
        "ml.core.prediction._mlflow_preflight_failure_policy",
        return_value={},
    )
    @patch("ml.core.prediction._find_local_mlflow_model_dir")
    @patch("mlflow.MlflowClient")
    @patch("ml.core.prediction._load_mlflow_pyfunc_model")
    def test_get_cached_mlflow_model_uses_local_artifact_when_registry_artifacts_are_missing(
        self,
        mock_load_model,
        mock_mlflow_client,
        mock_find_local_dir,
        _mock_policy,
    ):
        local_dir = Path("/tmp/local-mlflow-model")
        mock_find_local_dir.return_value = local_dir
        mock_mlflow_client.return_value.search_model_versions.return_value = [
            MagicMock(version="17")
        ]
        mock_load_model.side_effect = [
            mlflow.exceptions.MlflowException("remote artifact missing"),
            mlflow.exceptions.MlflowException("version artifact missing"),
            "local-model",
        ]

        model = _get_cached_mlflow_model("models:/hospital_shortage_predictor@champion")

        self.assertEqual(model, "local-model")
        self.assertEqual(mock_load_model.call_args_list[2].args[1], str(local_dir))

    def test_repair_loaded_model_compatibility_patches_nested_simple_imputer(self):
        from sklearn.impute import SimpleImputer

        imputer = SimpleImputer().fit([[1.0], [None]])
        if hasattr(imputer, "_fill_dtype"):
            delattr(imputer, "_fill_dtype")
        wrapper = type("Wrapper", (), {})()
        wrapper.pipeline = {"step": [imputer]}

        result = _repair_loaded_model_compatibility(wrapper)

        self.assertIs(result, wrapper)
        self.assertTrue(hasattr(imputer, "_fill_dtype"))
        self.assertEqual(imputer._fill_dtype, imputer._fit_dtype)

    @patch("ml.core.prediction.subprocess.run")
    def test_predict_mlflow_uses_configured_isolated_worker(self, mock_run):
        runtime = ModelRuntime(
            model_id="ideal_donor_classifier_champion",
            aliases=["ideal_donor_classifier"],
            slug="ideal_donor_classifier_champion",
            file_path="models:/ideal_donor_classifier@champion",
            model_type="mlflow",
            description="",
            status="loaded",
        )
        mock_run.return_value = MagicMock(
            returncode=0,
            stdout='{"output": {"prediction": 1, "used_inputs": {"donor_id": 24}}}',
            stderr="",
        )

        with patch.dict("os.environ", {"PIOS_MLFLOW_SUBPROCESS": "1"}, clear=False):
            result = _predict_mlflow(runtime, {"donor_id": 24})

        self.assertEqual(result["prediction"], 1)
        self.assertEqual(result["used_inputs"]["donor_id"], 24)
        self.assertIn("-m", mock_run.call_args.args[0])
        self.assertIn("ml.core.mlflow_predict_worker", mock_run.call_args.args[0])

    @patch("ml.core.prediction.subprocess.run")
    def test_predict_mlflow_reports_isolated_worker_failure(self, mock_run):
        runtime = ModelRuntime(
            model_id="ideal_donor_classifier_champion",
            aliases=["ideal_donor_classifier"],
            slug="ideal_donor_classifier_champion",
            file_path="models:/ideal_donor_classifier@champion",
            model_type="mlflow",
            description="",
            status="loaded",
        )
        mock_run.return_value = MagicMock(
            returncode=-11,
            stdout="",
            stderr="Segmentation fault: 11",
        )

        with patch.dict("os.environ", {"PIOS_MLFLOW_SUBPROCESS": "1"}, clear=False):
            with self.assertRaisesRegex(ValueError, "Segmentation fault"):
                _predict_mlflow(runtime, {"donor_id": 24})

    @patch("ml.core.prediction.subprocess.run")
    def test_predict_mlflow_subprocess_batch_sends_all_feature_rows(self, mock_run):
        runtime = ModelRuntime(
            model_id="ideal_donor_classifier_champion",
            aliases=["ideal_donor_classifier"],
            slug="ideal_donor_classifier_champion",
            file_path="models:/ideal_donor_classifier@champion",
            model_type="mlflow",
            description="",
            status="loaded",
        )
        mock_run.return_value = MagicMock(
            returncode=0,
            stdout='{"outputs": [{"prediction": 1}, {"prediction": 0}]}',
            stderr="",
        )

        outputs = _predict_mlflow_subprocess_batch(
            runtime,
            [{"donor_id": 24}, {"donor_id": 25}],
        )

        payload = json.loads(mock_run.call_args.kwargs["input"])
        self.assertEqual(
            payload["feature_rows"],
            [{"donor_id": 24}, {"donor_id": 25}],
        )
        self.assertEqual(outputs, [{"prediction": 1}, {"prediction": 0}])

    @patch("ml.core.prediction.ModelStats.objects.filter")
    @patch("ml.core.prediction._get_cached_mlflow_model")
    @patch("ml.core.prediction._mlflow_subprocess_enabled", return_value=False)
    def test_predict_mlflow_coerces_datetime_signature_columns(
        self,
        mock_subprocess_enabled,
        mock_get_model,
        mock_filter,
    ):
        mock_filter.return_value.first.return_value = None
        fake_model = _FakeMlflowModel(
            [
                _MlflowInputSpec("observed_at", "datetime"),
                _MlflowInputSpec("score", "double"),
                _MlflowInputSpec("label", "string"),
            ]
        )
        mock_get_model.return_value = fake_model
        runtime = ModelRuntime(
            model_id="generic_mlflow_model",
            aliases=[],
            slug="generic_mlflow_model",
            file_path="models:/generic_mlflow_model@champion",
            model_type="mlflow",
            description="",
            status="loaded",
        )

        result = _predict_mlflow(
            runtime,
            {
                "observed_at": "2026-06-10T20:39:50Z",
                "score": 4,
                "label": "A",
                "extra": "drop-me",
            },
        )

        self.assertEqual(str(fake_model.frame["observed_at"].dtype), "datetime64[ns]")
        self.assertEqual(str(fake_model.frame["score"].dtype), "float32")
        self.assertNotIn("extra", fake_model.frame.columns)
        self.assertEqual(result["prediction"], 0.42)
        self.assertEqual(result["used_inputs"]["observed_at"], "2026-06-10T20:39:50")
        self.assertEqual(result["missing_features"], [])

    @patch("ml.core.prediction._mlflow_missing_datetime_policy")
    @patch("ml.core.prediction.ModelStats.objects.filter")
    @patch("ml.core.prediction._get_cached_mlflow_model")
    @patch("ml.core.prediction._mlflow_subprocess_enabled", return_value=False)
    def test_predict_mlflow_fills_missing_datetime_from_configured_policy(
        self,
        mock_subprocess_enabled,
        mock_get_model,
        mock_filter,
        mock_datetime_policy,
    ):
        mock_filter.return_value.first.return_value = None
        mock_datetime_policy.return_value = {
            "action": "fill",
            "value": "2026-01-02T03:04:05Z",
        }
        fake_model = _FakeMlflowModel([_MlflowInputSpec("observed_at", "datetime")])
        mock_get_model.return_value = fake_model
        runtime = ModelRuntime(
            model_id="generic_mlflow_model",
            aliases=[],
            slug="generic_mlflow_model",
            file_path="models:/generic_mlflow_model@champion",
            model_type="mlflow",
            description="",
            status="loaded",
        )

        result = _predict_mlflow(runtime, {})

        self.assertEqual(str(fake_model.frame["observed_at"].dtype), "datetime64[ns]")
        self.assertEqual(result["used_inputs"]["observed_at"], "2026-01-02T03:04:05")
        self.assertEqual(result["missing_features"], ["observed_at"])


class PredictionStatsTests(SimpleTestCase):
    @patch("ml.core.prediction.ModelStats.objects.filter")
    def test_get_stats_parses_stringified_json_dict_payload(self, mock_filter):
        mock_filter.return_value.first.return_value = MagicMock(
            stats='{"feature_a": [1.0, 1.5], "feature_b": 2}'
        )

        result = _get_stats(
            "hospital_shortage_predictor_champion",
            feature_names=["feature_a", "feature_b"],
        )

        self.assertEqual(result, {"feature_a": 1.25, "feature_b": 2.0})

    @patch("ml.core.prediction.ModelStats.objects.filter")
    def test_get_stats_recovers_legacy_stringified_array_payload(self, mock_filter):
        mock_filter.return_value.first.return_value = MagicMock(
            stats=(
                "[[1.0,1.0],[19.7,19.9],[2.7,3.3],[0.0,0.0],[0.0,0.0],"
                "[3.0,3.0],[2.0,2.0],[0.0,0.0],[0.9,1.0]]"
            )
        )

        result = _get_stats(
            "hospital_shortage_predictor_champion",
            feature_names=[
                "hospital_supply_features_features__temperature",
                "hospital_supply_features_features__rain_mm",
                "hospital_supply_features_features__holiday",
                "hospital_supply_features_features__disaster",
                "hospital_supply_features_features__scheduled_surgeries",
                "hospital_supply_features_features__trauma_cases",
                "hospital_supply_features_features__current_inventory",
            ],
        )

        self.assertAlmostEqual(
            result["hospital_supply_features_features__temperature"], 19.8
        )
        self.assertAlmostEqual(
            result["hospital_supply_features_features__rain_mm"], 3.0
        )
        self.assertEqual(result["hospital_supply_features_features__holiday"], 0.0)

    @patch("ml.core.prediction.ModelStats.objects.filter")
    def test_get_stats_keeps_categorical_default_values(self, mock_filter):
        mock_filter.return_value.first.return_value = MagicMock(
            stats=(
                '{"age": {"default": 42}, '
                '"blood_group": {"default": "O+"}, '
                '"geo_city_id": {"mode": "C012"}}'
            )
        )

        result = _get_stats(
            "ideal_donor_classifier",
            feature_names=["age", "blood_group", "geo_city_id"],
        )

        self.assertEqual(result["age"], 42)
        self.assertEqual(result["blood_group"], "O+")
        self.assertNotIn("geo_city_id", result)


class MlflowPredictionNormalizationTests(SimpleTestCase):
    def test_normalize_mlflow_prediction_output_preserves_dataframe_columns(self):
        payload = pd.DataFrame([{"prediction": 1, "probability": 0.91}])

        result = _normalize_mlflow_prediction_output(payload)

        self.assertEqual(result["prediction"], 1)
        self.assertAlmostEqual(result["probability"], 0.91)

    def test_inventory_risk_output_contract_exposes_stockout_timing(self):
        runtime = ModelRuntime(
            model_id="component_inventory_risk_simulator",
            aliases=[],
            slug="component_inventory_risk_simulator",
            file_path=Path("/tmp/model"),
            description="",
        )

        result = _apply_prediction_output_contract(
            runtime,
            {
                "prediction": 0.9,
                "expected_days_until_stockout": -2.5,
            },
        )

        self.assertEqual(result["prediction"], 0.9)
        self.assertEqual(result["inventory_risk"], 0.9)
        self.assertEqual(result["expected_days_until_stockout"], 0.0)
        self.assertEqual(result["expected_days_until_stockout_raw"], -2.5)
        self.assertTrue(result["expected_days_until_stockout_clipped"])
        self.assertEqual(result["prediction_name"], "inventory_risk")
        self.assertEqual(result["prediction_unit"], "probability_and_days")
        self.assertEqual(
            result["prediction_task_type"],
            "multi_output_probability_and_timing",
        )

    def test_output_contract_only_clips_when_explicitly_enabled(self):
        runtime = ModelRuntime(
            model_id="custom_regressor",
            aliases=[],
            slug="custom_regressor",
            file_path=Path("/tmp/model"),
            description="",
            defaults={
                "output": {
                    "name": "score",
                    "task_type": "regression",
                    "clip_min": 0.0,
                    "apply_clipping": True,
                }
            },
        )

        result = _apply_prediction_output_contract(runtime, {"prediction": -2.5})

        self.assertEqual(result["prediction"], 0.0)
        self.assertEqual(result["score"], 0.0)
        self.assertEqual(result["raw_prediction"], -2.5)
        self.assertTrue(result["prediction_clipped"])

    def test_donor_hazard_output_contract_exposes_configured_probability(self):
        runtime = ModelRuntime(
            model_id="donor_next_donation_hazard_model",
            aliases=[],
            slug="donor_next_donation_hazard_model",
            file_path=Path("/tmp/model"),
            description="",
        )

        result = _apply_prediction_output_contract(runtime, {"prediction": 1.25})

        self.assertEqual(result["prediction"], 1.0)
        self.assertEqual(result["donation_probability"], 1.0)
        self.assertEqual(result["raw_prediction"], 1.25)
        self.assertTrue(result["prediction_clipped"])
        self.assertEqual(result["prediction_name"], "donation_probability")
        self.assertEqual(result["prediction_unit"], "probability")
        self.assertEqual(result["prediction_task_type"], "survival_hazard")
