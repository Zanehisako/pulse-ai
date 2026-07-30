from __future__ import annotations

import json
import os
import sys
from contextlib import nullcontext
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import django
import joblib
from django.apps import apps
from django.test import SimpleTestCase, override_settings
from rest_framework.test import APIRequestFactory, force_authenticate

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "backendMulti.settings")
if not apps.ready:
    django.setup()

BACKEND_ROOT = Path(__file__).resolve().parents[2]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from ml.core.model_discovery import (  # noqa: E402
    ModelDiscoveryConfigError,
    discover_model_config_rows,
    sync_discovered_model_configs,
)
from ml.api import views as ml_views  # noqa: E402
from ml.core.prediction import run_prediction  # noqa: E402
from ml.core.registry import ModelRegistry  # noqa: E402
from ml.core.registry import ModelRuntime  # noqa: E402
from ml.orchestrator.service import DynamicXLAMOrchestrator  # noqa: E402
from ml.services.control_center import ControlResult  # noqa: E402


def authenticate(request):
    force_authenticate(
        request,
        user=SimpleNamespace(is_authenticated=True, id=1, pk=1, email="ci@example.com"),
    )
    return request


class _DiscoveryEstimator:
    def predict(self, frame):
        return [float(frame.iloc[0].sum())]


class _ArrayEstimator:
    def predict(self, values):
        return [float(values[0].sum())]


def _write_runtime_config(root: Path, *, catalog_path: Path) -> Path:
    return _write_runtime_config_with_discovery(root, catalog_path=catalog_path)


def _write_runtime_config_with_discovery(
    root: Path,
    *,
    catalog_path: Path,
    discovery_overrides: dict | None = None,
) -> Path:
    path = root / "prediction_runtime.json"
    discovery = {
        "enabled": True,
        "models_dir_env": "PIOS_TEST_MODELS_DIR",
        "include_globs": ["*.joblib", "*.pkl"],
        "exclude_globs": ["gguf/**", "*registry*.joblib"],
        "catalog_path": str(catalog_path),
        "default_adapter": "sota_joblib",
        "supported_adapters": ["sota_joblib"],
        "upsert_enabled": True,
    }
    if discovery_overrides:
        discovery.update(discovery_overrides)
    path.write_text(
        json.dumps(
            {
                "model_discovery": discovery
            }
        ),
        encoding="utf-8",
    )
    return path


def _catalog_row(**overrides):
    row = {
        "id": "catalog_dynamic_model",
        "enabled": True,
        "registered_model_name": "dynamic_model",
        "artifact_filename": "dynamic_model.joblib",
        "task_type": "regression",
        "outputs": {"primary": "dynamic_prediction"},
        "features": ["feature_a", "feature_b"],
        "feature_types": {"feature_a": "number", "feature_b": "number"},
    }
    row.update(overrides)
    return row


def _write_catalog(path: Path, rows: list[dict]) -> None:
    path.write_text(
        json.dumps({"version": "test-catalog", "models": rows}),
        encoding="utf-8",
    )


class ModelDiscoveryValidationTests(SimpleTestCase):
    def test_discovers_enabled_catalog_model(self):
        with TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            models_dir = root / "models"
            models_dir.mkdir()
            (models_dir / "dynamic_model.joblib").write_bytes(b"placeholder")
            catalog_path = root / "catalog.json"
            _write_catalog(catalog_path, [_catalog_row()])
            runtime_config = _write_runtime_config(root, catalog_path=catalog_path)

            with patch.dict(os.environ, {"PIOS_TEST_MODELS_DIR": str(models_dir)}):
                rows = discover_model_config_rows(runtime_config_path=runtime_config)

        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["model_id"], "dynamic_model")
        self.assertEqual(rows[0]["model_type"], "sota_joblib")
        self.assertEqual(rows[0]["file_path"], "dynamic_model.joblib")
        self.assertTrue(rows[0]["is_active"])

    def test_rejects_duplicate_model_ids(self):
        with TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            models_dir = root / "models"
            models_dir.mkdir()
            (models_dir / "dynamic_model.joblib").write_bytes(b"placeholder")
            (models_dir / "other.joblib").write_bytes(b"placeholder")
            catalog_path = root / "catalog.json"
            _write_catalog(
                catalog_path,
                [
                    _catalog_row(),
                    _catalog_row(id="other", artifact_filename="other.joblib"),
                ],
            )
            runtime_config = _write_runtime_config(root, catalog_path=catalog_path)

            with patch.dict(os.environ, {"PIOS_TEST_MODELS_DIR": str(models_dir)}):
                with self.assertRaisesRegex(ModelDiscoveryConfigError, "Duplicate"):
                    discover_model_config_rows(runtime_config_path=runtime_config)

    def test_rejects_unsupported_adapter(self):
        with TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            models_dir = root / "models"
            models_dir.mkdir()
            (models_dir / "dynamic_model.joblib").write_bytes(b"placeholder")
            catalog_path = root / "catalog.json"
            _write_catalog(catalog_path, [_catalog_row(adapter="unknown_adapter")])
            runtime_config = _write_runtime_config(root, catalog_path=catalog_path)

            with patch.dict(os.environ, {"PIOS_TEST_MODELS_DIR": str(models_dir)}):
                with self.assertRaisesRegex(ModelDiscoveryConfigError, "Unsupported"):
                    discover_model_config_rows(runtime_config_path=runtime_config)

    def test_rejects_missing_enabled_artifact(self):
        with TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            models_dir = root / "models"
            models_dir.mkdir()
            catalog_path = root / "catalog.json"
            _write_catalog(catalog_path, [_catalog_row()])
            runtime_config = _write_runtime_config(root, catalog_path=catalog_path)

            with patch.dict(os.environ, {"PIOS_TEST_MODELS_DIR": str(models_dir)}):
                with self.assertRaisesRegex(ModelDiscoveryConfigError, "does not exist"):
                    discover_model_config_rows(runtime_config_path=runtime_config)

    def test_can_deactivate_missing_enabled_artifacts_from_config(self):
        with TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            models_dir = root / "models"
            models_dir.mkdir()
            catalog_path = root / "catalog.json"
            _write_catalog(catalog_path, [_catalog_row()])
            runtime_config = _write_runtime_config_with_discovery(
                root,
                catalog_path=catalog_path,
                discovery_overrides={"deactivate_missing_artifacts": True},
            )

            with patch.dict(os.environ, {"PIOS_TEST_MODELS_DIR": str(models_dir)}):
                rows = discover_model_config_rows(runtime_config_path=runtime_config)

        self.assertEqual(len(rows), 1)
        self.assertFalse(rows[0]["is_active"])
        self.assertTrue(rows[0]["defaults"]["model_discovery"]["artifact_missing"])

    def test_disabled_catalog_row_does_not_require_artifact(self):
        with TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            models_dir = root / "models"
            models_dir.mkdir()
            catalog_path = root / "catalog.json"
            _write_catalog(catalog_path, [_catalog_row(enabled=False)])
            runtime_config = _write_runtime_config(root, catalog_path=catalog_path)

            with patch.dict(os.environ, {"PIOS_TEST_MODELS_DIR": str(models_dir)}):
                rows = discover_model_config_rows(runtime_config_path=runtime_config)

        self.assertEqual(len(rows), 1)
        self.assertFalse(rows[0]["is_active"])


class ModelDiscoveryRegistryTests(SimpleTestCase):
    def test_sync_does_not_reactivate_model_config_file_owned_rows(self):
        with TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            models_dir = root / "models"
            models_dir.mkdir()
            (models_dir / "dynamic_model.joblib").write_bytes(b"placeholder")
            catalog_path = root / "catalog.json"
            config_path = root / "config.json"
            config_path.write_text('{"models": []}', encoding="utf-8")
            _write_catalog(catalog_path, [_catalog_row()])
            runtime_config = _write_runtime_config(root, catalog_path=catalog_path)
            existing = SimpleNamespace(
                model_id="dynamic_model",
                description="Removed from model config",
                file_path="dynamic_model.joblib",
                model_type="sota_joblib",
                features=[],
                feature_info={},
                examples=[],
                defaults={},
                is_active=False,
                config_source_path=str(config_path.resolve()),
                save=MagicMock(),
            )

            with (
                patch.dict(os.environ, {"PIOS_TEST_MODELS_DIR": str(models_dir)}),
                override_settings(PIOS_MODEL_CONFIG_PATH=config_path),
            ):
                manager = MagicMock()
                manager.filter.return_value.first.return_value = existing
                with (
                    patch("ml.core.model_discovery.transaction.atomic", return_value=nullcontext()),
                    patch("ml.core.model_discovery.MLModelConfig.objects", manager),
                ):
                    summary = sync_discovered_model_configs(
                        "test-sync",
                        runtime_config_path=runtime_config,
                    )

        self.assertFalse(summary["changed"])
        self.assertEqual(summary["discovered_total"], 1)
        self.assertEqual(summary["upserted_total"], 0)
        self.assertEqual(summary["skipped_config_source_total"], 1)
        manager.create.assert_not_called()
        existing.save.assert_not_called()
        self.assertFalse(existing.is_active)

    def test_sync_can_take_over_model_config_rows_when_configured(self):
        with TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            models_dir = root / "models"
            models_dir.mkdir()
            (models_dir / "dynamic_model.joblib").write_bytes(b"placeholder")
            catalog_path = root / "catalog.json"
            config_path = root / "config.json"
            config_path.write_text('{"models": []}', encoding="utf-8")
            _write_catalog(catalog_path, [_catalog_row()])
            runtime_config = _write_runtime_config_with_discovery(
                root,
                catalog_path=catalog_path,
                discovery_overrides={"respect_model_config_file": False},
            )
            existing = SimpleNamespace(
                model_id="dynamic_model",
                description="Removed from model config",
                file_path="dynamic_model.joblib",
                model_type="sota_joblib",
                features=[],
                feature_info={},
                examples=[],
                defaults={},
                is_active=False,
                config_source_path=str(config_path.resolve()),
                save=MagicMock(),
            )

            with (
                patch.dict(os.environ, {"PIOS_TEST_MODELS_DIR": str(models_dir)}),
                override_settings(PIOS_MODEL_CONFIG_PATH=config_path),
            ):
                manager = MagicMock()
                manager.filter.return_value.first.return_value = existing
                with (
                    patch("ml.core.model_discovery.transaction.atomic", return_value=nullcontext()),
                    patch("ml.core.model_discovery.MLModelConfig.objects", manager),
                ):
                    summary = sync_discovered_model_configs(
                        "test-sync",
                        runtime_config_path=runtime_config,
                    )

        self.assertTrue(summary["changed"])
        self.assertEqual(summary["discovered_total"], 1)
        self.assertEqual(summary["upserted_total"], 1)
        self.assertEqual(summary["skipped_config_source_total"], 0)
        existing.save.assert_called_once()
        self.assertTrue(existing.is_active)

    def test_sync_registers_and_registry_loads_discovered_joblib_model(self):
        with TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            models_dir = root / "models"
            models_dir.mkdir()
            joblib.dump(_DiscoveryEstimator(), models_dir / "dynamic_model.joblib")
            catalog_path = root / "catalog.json"
            _write_catalog(catalog_path, [_catalog_row()])
            runtime_config = _write_runtime_config(root, catalog_path=catalog_path)

            with patch.dict(os.environ, {"PIOS_TEST_MODELS_DIR": str(models_dir)}):
                manager = MagicMock()
                manager.filter.return_value.first.return_value = None
                with (
                    patch("ml.core.model_discovery.transaction.atomic", return_value=nullcontext()),
                    patch("ml.core.model_discovery.MLModelConfig.objects", manager),
                ):
                    summary = sync_discovered_model_configs(
                        "test-sync",
                        runtime_config_path=runtime_config,
                    )
                rows = discover_model_config_rows(runtime_config_path=runtime_config)
                registry = ModelRegistry(
                    models_dir=models_dir,
                    config_path=runtime_config,
                    config_loader=lambda: [
                        {
                            "id": row["model_id"],
                            "description": row["description"],
                            "file_path": row["file_path"],
                            "type": row["model_type"],
                            "features": row["features"],
                            "feature_info": row["feature_info"],
                            "examples": row["examples"],
                            "defaults": row["defaults"],
                        }
                        for row in rows
                        if row["is_active"]
                    ],
                )
                registry.refresh(force=True)
                runtime = registry.get("dynamic_model")
                alias_runtime = registry.get("catalog_dynamic_model")
                output = run_prediction(
                    runtime,
                    {"feature_a": 2, "feature_b": 3},
                )

        self.assertTrue(summary["changed"])
        self.assertEqual(summary["discovered_total"], 1)
        manager.create.assert_called_once()
        self.assertIsNotNone(runtime)
        self.assertEqual(runtime.status, "loaded")
        self.assertEqual(runtime.model_type, "sota_joblib")
        self.assertIs(alias_runtime, runtime)
        self.assertEqual(output["prediction"], 5.0)
        self.assertEqual(output["dynamic_prediction"], 5.0)

    def test_registry_loads_raw_discovery_rows_by_registered_model_name(self):
        with TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            models_dir = root / "models"
            models_dir.mkdir()
            joblib.dump(_DiscoveryEstimator(), models_dir / "dynamic_model.joblib")
            catalog_path = root / "catalog.json"
            _write_catalog(catalog_path, [_catalog_row()])
            runtime_config = _write_runtime_config(root, catalog_path=catalog_path)

            with patch.dict(os.environ, {"PIOS_TEST_MODELS_DIR": str(models_dir)}):
                rows = discover_model_config_rows(runtime_config_path=runtime_config)
                registry = ModelRegistry(
                    models_dir=models_dir,
                    config_path=runtime_config,
                    config_loader=lambda: rows,
                )
                registry.refresh(force=True)
                runtime = registry.get("dynamic_model")
                alias_runtime = registry.get("catalog_dynamic_model")
                output = run_prediction(
                    runtime,
                    {"feature_a": 7, "feature_b": 8},
                )

        self.assertIsNotNone(runtime)
        self.assertEqual(runtime.model_id, "dynamic_model")
        self.assertIs(alias_runtime, runtime)
        self.assertEqual(output["prediction"], 15.0)
        self.assertEqual(output["dynamic_prediction"], 15.0)


class ModelBackedToolTests(SimpleTestCase):
    def setUp(self):
        self.factory = APIRequestFactory()

    def _runtime(self):
        return ModelRuntime(
            model_id="dynamic_model",
            aliases=["catalog_dynamic_model"],
            slug="dynamic_model",
            file_path=Path("/tmp/dynamic_model.joblib"),
            description="Dynamic local model",
            feature_names=["feature_a", "feature_b"],
            defaults={"output": {"name": "dynamic_prediction", "task_type": "regression"}},
            model_type="estimator",
            status="loaded",
            estimator=_ArrayEstimator(),
        )

    def test_orchestrator_exposes_loaded_models_as_direct_tools(self):
        runtime = self._runtime()
        registry = SimpleNamespace(
            loaded=lambda: [runtime],
            get=lambda value: runtime
            if value in {"dynamic_model", "catalog_dynamic_model"}
            else None,
        )
        orchestrator = object.__new__(DynamicXLAMOrchestrator)
        orchestrator.registry = registry
        orchestrator.external_tool_definitions = []
        orchestrator.external_tool_aliases = {}

        tools = orchestrator.list_configured_tools()
        output = orchestrator.execute_configured_tool(
            "catalog_dynamic_model",
            arguments={"feature_a": 2, "feature_b": 4},
        )

        self.assertEqual(tools[0]["id"], "dynamic_model")
        self.assertEqual(tools[0]["adapter"], "model")
        self.assertEqual(orchestrator.configured_tool_id("catalog_dynamic_model"), "dynamic_model")
        self.assertEqual(output["prediction"], 6.0)
        self.assertEqual(output["dynamic_prediction"], 6.0)

    def test_model_list_api_includes_loaded_runtime(self):
        runtime = self._runtime()
        registry = SimpleNamespace(list_models=lambda: [runtime])
        request = self.factory.get("/api/ml/models/")
        authenticate(request)

        with (
            patch("ml.api.views.is_ready", return_value=True),
            patch("ml.api.views.refresh_runtime_state"),
            patch("ml.api.views.get_registry", return_value=registry),
        ):
            response = ml_views.ModelListView.as_view()(request)

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.data["models"][0]["model_id"], "dynamic_model")

    def test_predict_model_api_runs_loaded_runtime(self):
        request = self.factory.post(
            "/api/ml/models/dynamic_model/predict/",
            {"features": {"feature_a": 3, "feature_b": 4}},
            format="json",
        )
        authenticate(request)

        with patch(
            "ml.api.views.run_direct_model_prediction",
            return_value=ControlResult(
                {
                    "results": {
                        "details": [
                            {
                                "tool": "dynamic_model",
                                "result": {"prediction": 7.0},
                            }
                        ]
                    },
                    "tools_used": ["dynamic_model"],
                    "planner_mode": "direct",
                }
            ),
        ) as predict_mock:
            response = ml_views.PredictModelView.as_view()(request, model_id="dynamic_model")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.data["results"]["details"][0]["result"]["prediction"], 7.0)
        self.assertEqual(predict_mock.call_args.args[0], "dynamic_model")

    def test_tool_run_api_executes_model_backed_tool(self):
        request = self.factory.post(
            "/api/ml/tools/dynamic_model/run/",
            {"arguments": {"feature_a": 5, "feature_b": 6}},
            format="json",
        )
        authenticate(request)
        orchestrator = SimpleNamespace(
            configured_tool_id=lambda _tool_id: "dynamic_model",
            execute_configured_tool=lambda *_args, **_kwargs: {
                "model_id": "dynamic_model",
                "prediction": 11.0,
                "dynamic_prediction": 11.0,
            },
            _result_value_for_summary=lambda output: str(output["prediction"]),
        )

        with (
            patch("ml.api.views.is_ready", return_value=True),
            patch("ml.api.views.refresh_runtime_state"),
            patch("ml.api.views.get_orchestrator", return_value=orchestrator),
            patch("ml.api.views.PredictionLog.objects.create"),
        ):
            response = ml_views.ToolRunView.as_view()(request, tool_id="dynamic_model")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.data["tools_used"], ["dynamic_model"])
        self.assertEqual(
            response.data["results"]["details"][0]["result"]["prediction"],
            11.0,
        )
