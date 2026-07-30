from django.test import SimpleTestCase
from unittest.mock import patch

from ml.core import model_loading
from ml.core.registry import ModelRuntime


def _runtime(model_id: str, *, model_type: str, file_path: str = "artifact.joblib"):
    return ModelRuntime(
        model_id=model_id,
        aliases=[model_id],
        slug=model_id,
        file_path=file_path,
        description="test runtime",
        model_type=model_type,
        status="loaded",
    )


class _Registry:
    def __init__(self, runtime=None):
        self.runtime = runtime

    def get(self, _identifier):
        return self.runtime


class ModelLoadingPolicyTests(SimpleTestCase):
    def test_invalid_source_fails_clearly(self):
        with self.assertRaisesRegex(model_loading.ModelSourceResolutionError, "Unsupported"):
            model_loading.resolve_loading_policy(
                {"primary": "unknown_source", "fallbacks": []}
            )


class ModelRuntimeResolutionTests(SimpleTestCase):
    def test_mlflow_success_returns_registry_source(self):
        with patch("ml.core.model_loading.preload_prediction_runtime"):
            resolution = model_loading.resolve_model_runtime(
                "model_a",
                "challenger",
                registry=_Registry(),
                runtime_config={
                    "model_loading": {
                        "default_policy": {
                            "primary": "mlflow_registry",
                            "fallbacks": ["local_registry"],
                        }
                    }
                },
            )

        self.assertEqual(resolution.source_used, "mlflow_registry")
        self.assertEqual(resolution.source_ref, "models:/model_a@challenger")

    def test_mlflow_failure_falls_back_to_local_registry(self):
        local_runtime = _runtime("model_a", model_type="sota_joblib")

        with patch(
            "ml.core.model_loading.preload_prediction_runtime",
            side_effect=RuntimeError("alias missing"),
        ):
            resolution = model_loading.resolve_model_runtime(
                "model_a",
                "challenger",
                registry=_Registry(local_runtime),
                runtime_config={
                    "model_loading": {
                        "default_policy": {
                            "primary": "mlflow_registry",
                            "fallbacks": ["local_registry"],
                        }
                    }
                },
            )

        self.assertEqual(resolution.source_used, "local_registry")
        self.assertEqual(resolution.runtime, local_runtime)
        self.assertIn("alias missing", resolution.fallback_reason)

    def test_missing_all_sources_reports_all_failures(self):
        with (
            patch(
                "ml.core.model_loading.preload_prediction_runtime",
                side_effect=RuntimeError("mlflow unavailable"),
            ),
            patch("ml.core.model_loading.discover_model_config_rows", return_value=[]),
        ):
            with self.assertRaisesRegex(
                model_loading.ModelSourceResolutionError,
                "mlflow_registry",
            ):
                model_loading.resolve_model_runtime(
                    "model_a",
                    "challenger",
                    registry=_Registry(),
                    runtime_config={
                        "model_loading": {
                            "default_policy": {
                                "primary": "mlflow_registry",
                                "fallbacks": ["local_registry"],
                            }
                        }
                    },
                )


class ModelPredictionFallbackTests(SimpleTestCase):
    def test_mlflow_prediction_failure_uses_local_runtime(self):
        mlflow_runtime = _runtime(
            "model_a",
            model_type="mlflow",
            file_path="models:/model_a@challenger",
        )
        local_runtime = _runtime("model_a", model_type="sota_joblib")

        with (
            patch(
                "ml.core.model_loading.run_prediction",
                side_effect=[
                    RuntimeError("mlflow prediction failed"),
                    {"prediction": 0.75},
                ],
            ),
            patch(
                "ml.core.model_loading.resolve_model_runtime",
                return_value=type(
                    "Resolution",
                    (),
                    {
                        "runtime": local_runtime,
                        "source_used": "local_registry",
                        "source_ref": "artifact.joblib",
                        "fallback_reason": "mlflow prediction failed",
                    },
                )(),
            ),
        ):
            result = model_loading.run_prediction_with_fallback(
                mlflow_runtime,
                {"feature": 1},
                registry=_Registry(local_runtime),
            )

        self.assertEqual(result["prediction"], 0.75)
        self.assertEqual(result["model_source"], "local_registry")
