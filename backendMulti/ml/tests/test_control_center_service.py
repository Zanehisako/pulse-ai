import os
import sys
from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import django
from django.apps import apps
from django.contrib import admin
from django.test import RequestFactory, SimpleTestCase
from rest_framework import status

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "backendMulti.settings")

if "keycloak" not in sys.modules:
    keycloak_stub = SimpleNamespace(KeycloakOpenID=object, KeycloakAdmin=object)
    sys.modules["keycloak"] = keycloak_stub

if not apps.ready:
    django.setup()

from ml.admin import MLModelConfigAdmin
from ml.core.feature_store import ResolvedFeaturePayload
from ml.models import MLModelConfig
from ml.services.control_center import (
    reload_model_config_control,
    run_direct_model_prediction,
    select_orchestrator_model,
)


class ControlCenterReloadServiceTests(SimpleTestCase):
    def test_reload_success_returns_runtime_counts_and_watcher_status(self):
        registry = SimpleNamespace(
            list_models=lambda: [object(), object()],
            loaded=lambda: [object()],
        )
        reload_result = {
            "file_sync": {"changed": True, "models_total": 1},
            "runtime_config_changed": True,
            "orchestrator_catalog_changed": False,
            "external_tool_reload": {"changed": False},
            "config_hot_reload": {"running": True, "enabled": True},
        }

        with (
            patch(
                "ml.services.control_center.reload_model_config_file",
                return_value=reload_result,
            ),
            patch("ml.services.control_center.get_registry", return_value=registry),
            patch(
                "ml.services.control_center.get_runtime_config_status",
                return_value={"reload_count": 3},
            ),
            patch(
                "ml.services.control_center.get_orchestrator_config_status",
                return_value={"reload_count": 1},
            ),
        ):
            result = reload_model_config_control("test-reload")

        self.assertEqual(result.status_code, 200)
        self.assertTrue(result.payload["changed"])
        self.assertEqual(result.payload["models_total"], 2)
        self.assertEqual(result.payload["models_loaded"], 1)
        self.assertEqual(result.payload["config_hot_reload"]["running"], True)

    def test_reload_failure_returns_clear_422_payload(self):
        with (
            patch(
                "ml.services.control_center.reload_model_config_file",
                side_effect=ValueError("bad config"),
            ),
            patch(
                "ml.services.control_center.get_runtime_config_status",
                return_value={"last_source_error": "bad config"},
            ),
            patch(
                "ml.services.control_center.get_orchestrator_config_status",
                return_value={},
            ),
        ):
            result = reload_model_config_control("test-reload")

        self.assertEqual(result.status_code, 422)
        self.assertIn("bad config", result.payload["error"])


class ControlCenterOrchestratorServiceTests(SimpleTestCase):
    def test_select_orchestrator_hot_applies_configured_variant(self):
        manager = MagicMock()
        clear_selected_qs = MagicMock()
        selected_variant = SimpleNamespace(
            updated_at=SimpleNamespace(isoformat=lambda: "2026-03-11T10:00:00")
        )
        selected_variant_qs = MagicMock()
        selected_variant_qs.first.return_value = selected_variant
        manager.filter.side_effect = [clear_selected_qs, selected_variant_qs]
        orchestrator = SimpleNamespace(
            list_available_models=lambda: [
                {
                    "id": "xlam_7b_q4_k_m",
                    "name": "xLAM 7B Q4_K_M",
                    "description": "xLAM variant",
                    "repo_id": "bartowski/xLAM-7b-fc-r-GGUF",
                    "filename": "xLAM-7b-fc-r-Q4_K_M.gguf",
                    "size_mb": 4200,
                    "size_bytes": 4404019200,
                    "min_ram_gb": 8,
                    "min_vram_mb": 5000,
                    "n_ctx": 4096,
                    "n_batch": 256,
                }
            ],
            preferred_model_id="xlam_7b_q4_k_m",
            status=lambda: {"llm_ready": False, "active_model_id": None},
        )

        with (
            patch("ml.services.control_center.transaction.atomic", return_value=nullcontext()),
            patch("ml.services.control_center.OrchestratorVariant.objects", manager),
            patch(
                "ml.services.control_center.sync_orchestrator_catalog",
                return_value=True,
            ) as sync_mock,
            patch("ml.services.control_center.get_orchestrator", return_value=orchestrator),
            patch(
                "ml.services.control_center.get_orchestrator_config_status",
                return_value={"selected_model_id": "xlam_7b_q4_k_m"},
            ),
        ):
            result = select_orchestrator_model(
                "xlam_7b_q4_k_m",
                download=True,
                wait=False,
            )

        self.assertEqual(result.status_code, 200)
        self.assertEqual(result.payload["selected_model_id"], "xlam_7b_q4_k_m")
        clear_selected_qs.update.assert_called_once_with(is_selected=False)
        manager.update_or_create.assert_called_once()
        self.assertEqual(sync_mock.call_count, 2)

    def test_select_orchestrator_persists_auto_detected_metadata(self):
        manager = MagicMock()
        selected_variant = SimpleNamespace(
            updated_at=SimpleNamespace(isoformat=lambda: "2026-04-21T10:00:00")
        )
        selected_variant_qs = MagicMock()
        selected_variant_qs.first.return_value = selected_variant
        manager.filter.side_effect = [
            MagicMock(),
            selected_variant_qs,
        ]
        orchestrator = SimpleNamespace(
            list_available_models=lambda: [
                {
                    "id": "qwen2_7b_instruct_q4_0",
                    "name": "Qwen2 7B Instruct Q4_0",
                    "description": "Auto-detected local GGUF orchestrator model.",
                    "repo_id": "local",
                    "filename": "qwen2-7b-instruct-q4_0.gguf",
                    "size_mb": 4024,
                    "size_bytes": 4219469824,
                    "min_ram_gb": 5.4,
                    "min_vram_mb": 0,
                    "n_ctx": 4096,
                    "n_batch": 128,
                    "source": "local",
                    "auto_detected": True,
                    "local_only": True,
                    "prompt_mode": "chat",
                }
            ],
            preferred_model_id="qwen2_7b_instruct_q4_0",
            status=lambda: {"llm_ready": False, "active_model_id": None},
        )

        with (
            patch("ml.services.control_center.transaction.atomic", return_value=nullcontext()),
            patch("ml.services.control_center.OrchestratorVariant.objects", manager),
            patch("ml.services.control_center.sync_orchestrator_catalog", return_value=True),
            patch("ml.services.control_center.get_orchestrator", return_value=orchestrator),
            patch(
                "ml.services.control_center.get_orchestrator_config_status",
                return_value={"selected_model_id": "qwen2_7b_instruct_q4_0"},
            ),
        ):
            result = select_orchestrator_model("qwen2_7b_instruct_q4_0")

        self.assertEqual(result.status_code, 200)
        manager.update_or_create.assert_called_once()
        defaults = manager.update_or_create.call_args.kwargs["defaults"]
        self.assertEqual(defaults["metadata_extra"]["source"], "local")
        self.assertTrue(defaults["metadata_extra"]["auto_detected"])


class ControlCenterDirectPredictionServiceTests(SimpleTestCase):
    def _runtime(self, model_id="direct_model"):
        runtime = MagicMock()
        runtime.model_id = model_id
        runtime.status = "loaded"
        runtime.load_error = None
        runtime.defaults = {}
        return runtime

    @patch("ml.services.control_center.AlertEngine")
    @patch("ml.services.control_center.PredictionLog.objects.create")
    @patch("ml.services.control_center.run_prediction_with_fallback")
    @patch("ml.services.control_center.resolve_prediction_features")
    @patch("ml.services.control_center.refresh_runtime_state")
    @patch("ml.services.control_center.get_registry")
    @patch("ml.services.control_center.is_ready")
    def test_direct_prediction_uses_runtime_metadata_and_logs_direct_mode(
        self,
        mock_is_ready,
        mock_get_registry,
        mock_refresh,
        mock_resolve,
        mock_run_prediction,
        mock_log_create,
        mock_alert_engine,
    ):
        mock_is_ready.return_value = True
        runtime = self._runtime()
        registry = MagicMock()
        registry.get.return_value = runtime
        mock_get_registry.return_value = registry
        mock_resolve.return_value = ResolvedFeaturePayload(
            features={"age": 35, "bmi": 24.0},
            metadata={"source": "request"},
        )
        mock_run_prediction.return_value = {"prediction": 1}

        result = run_direct_model_prediction(
            "direct_model",
            {"query": "run it", "features": {"age": 35, "bmi": 24}},
            user=SimpleNamespace(is_authenticated=True),
        )

        self.assertEqual(result.status_code, status.HTTP_200_OK)
        self.assertEqual(result.payload["planner_mode"], "direct")
        self.assertEqual(result.payload["tools_used"], ["direct_model"])
        mock_resolve.assert_called_once_with(runtime, {"age": 35, "bmi": 24})
        mock_run_prediction.assert_called_once_with(
            runtime,
            {"age": 35, "bmi": 24.0},
            registry=registry,
        )
        self.assertEqual(mock_log_create.call_args.kwargs["planner_mode"], "direct")
        mock_alert_engine.return_value.evaluate_prediction.assert_called_once()

    @patch("ml.services.control_center.AlertEngine")
    @patch("ml.services.control_center.PredictionLog.objects.create")
    @patch("ml.services.control_center.run_prediction_batch_with_fallback")
    @patch("ml.services.control_center.resolve_prediction_features")
    @patch("ml.services.control_center.refresh_runtime_state")
    @patch("ml.services.control_center.get_registry")
    @patch("ml.services.control_center.is_ready")
    def test_direct_prediction_supports_batch_inputs(
        self,
        mock_is_ready,
        mock_get_registry,
        mock_refresh,
        mock_resolve,
        mock_run_batch,
        mock_log_create,
        mock_alert_engine,
    ):
        mock_is_ready.return_value = True
        runtime = self._runtime("batch_model")
        registry = MagicMock()
        registry.get.return_value = runtime
        mock_get_registry.return_value = registry
        row = {"age": 35}
        mock_resolve.return_value = ResolvedFeaturePayload(
            features=row,
            metadata={"source": "request"},
        )
        mock_run_batch.return_value = [{"prediction": 1}]

        result = run_direct_model_prediction("batch_model", {"inputs": [row]})

        self.assertEqual(result.status_code, status.HTTP_200_OK)
        self.assertEqual(result.payload["planner_mode"], "direct_batch")
        self.assertEqual(
            result.payload["execution_results"][0]["output"]["rows"][0]["prediction"],
            1,
        )
        mock_run_batch.assert_called_once_with(runtime, [row], registry=registry)
        self.assertEqual(mock_log_create.call_args.kwargs["planner_mode"], "direct_batch")


class MLControlCenterAdminTests(SimpleTestCase):
    def setUp(self):
        self.factory = RequestFactory()
        self.model_admin = MLModelConfigAdmin(MLModelConfig, admin.site)

    def _staff_user(self):
        return SimpleNamespace(
            is_active=True,
            is_staff=True,
            is_authenticated=True,
            has_perm=lambda perm: True,
            has_module_perms=lambda app_label: True,
        )

    def test_admin_control_center_renders_dynamic_status(self):
        request = self.factory.get("/admin/ml/mlmodelconfig/control-center/")
        request.user = self._staff_user()
        payload = {
            "status": "ok",
            "models_loaded": 1,
            "models_total": 1,
            "models": [
                {
                    "model_id": "dynamic_model",
                    "status": "loaded",
                    "model_type": "mlflow",
                    "feature_names": ["age"],
                }
            ],
            "loaded_models": [],
            "orchestrator_models": [
                {"id": "orch_a", "name": "Orchestrator A", "local_exists": True}
            ],
            "selected_model_id": "orch_a",
            "orchestrator": {
                "planner_mode": "fallback",
                "llm_ready": False,
                "selected_model_id": "orch_a",
                "active_model_id": None,
                "selected_model_exists": True,
                "selected_model_path": "/tmp/orch.gguf",
                "llm_error": None,
            },
            "config_runtime": {
                "reload_count": 1,
                "last_reload_reason": "test",
                "last_source_error": None,
                "model_config_file": "/tmp/config.json",
                "config_hot_reload": {"enabled": True, "running": True},
            },
        }

        with patch("ml.admin.get_control_status", return_value=payload):
            response = self.model_admin.control_center_view(request)
            response.render()

        body = response.content.decode("utf-8")
        self.assertIn("ML Control Center", body)
        self.assertIn("dynamic_model", body)
        self.assertIn("Orchestrator A", body)

    def test_admin_control_center_rejects_non_staff_user(self):
        request = self.factory.get("/admin/ml/mlmodelconfig/control-center/")
        request.user = SimpleNamespace(
            is_active=True,
            is_staff=False,
            is_authenticated=True,
            has_perm=lambda perm: False,
            has_module_perms=lambda app_label: False,
        )
        protected_view = admin.site.admin_view(self.model_admin.control_center_view)

        response = protected_view(request)

        self.assertIn(response.status_code, {302, 403})
