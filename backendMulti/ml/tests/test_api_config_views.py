from contextlib import nullcontext
import os
import sys
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import django
from django.test import SimpleTestCase
from rest_framework.test import APIRequestFactory, force_authenticate

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "backendMulti.settings")

if "keycloak" not in sys.modules:
    keycloak_stub = SimpleNamespace(KeycloakOpenID=object, KeycloakAdmin=object)
    sys.modules["keycloak"] = keycloak_stub

django.setup()

from ml.api.views import (
    ControlStatusView,
    ConfigReloadView,
    OrchestratorSelectView,
    RemoveModelView,
    RuntimeConfigView,
)
from ml.services.control_center import ControlResult


def authenticate(request):
    force_authenticate(
        request,
        user=SimpleNamespace(is_authenticated=True, id=1, pk=1, email="ci@example.com"),
    )
    return request


class RuntimeConfigViewTests(SimpleTestCase):
    def setUp(self):
        self.factory = APIRequestFactory()

    def test_get_returns_runtime_config_status(self):
        request = self.factory.get("/api/ml/config/runtime/")
        authenticate(request)
        with patch(
            "ml.api.views.get_runtime_config_status",
            return_value={"source": "django_db", "reload_count": 3},
        ):
            response = RuntimeConfigView.as_view()(request)
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.data["source"], "django_db")

    def test_post_reload_returns_file_sync_and_external_tool_fields(self):
        request = self.factory.post("/api/ml/config/reload/")
        authenticate(request)
        payload = {
            "status": "ok",
            "changed": True,
            "file_sync": {"changed": True, "models_total": 1},
            "runtime_config_changed": True,
            "orchestrator_catalog_changed": False,
            "external_tool_reload": {"changed": True, "external_tools_total": 2},
            "models_total": 2,
            "models_loaded": 1,
        }
        with (
            patch(
                "ml.api.views.reload_model_config_control",
                return_value=ControlResult(payload),
            ) as reload_mock,
        ):
            response = ConfigReloadView.as_view()(request)

        self.assertEqual(response.status_code, 200)
        self.assertTrue(response.data["changed"])
        self.assertEqual(response.data["file_sync"], payload["file_sync"])
        self.assertEqual(
            response.data["external_tool_reload"],
            payload["external_tool_reload"],
        )
        self.assertEqual(response.data["models_total"], 2)
        self.assertEqual(response.data["models_loaded"], 1)
        reload_mock.assert_called_once_with("manual-reload")


class ControlStatusViewTests(SimpleTestCase):
    def setUp(self):
        self.factory = APIRequestFactory()

    def test_get_returns_control_status_payload(self):
        request = self.factory.get("/api/ml/control/status/")
        authenticate(request)
        payload = {
            "status": "ok",
            "ml_ready": True,
            "models_total": 2,
            "models_loaded": 1,
        }
        with patch("ml.api.views.get_control_status", return_value=payload) as status_mock:
            response = ControlStatusView.as_view()(request)

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.data["models_loaded"], 1)
        status_mock.assert_called_once_with(refresh=True)

class OrchestratorSelectViewTests(SimpleTestCase):
    def setUp(self):
        self.factory = APIRequestFactory()

    def test_put_hot_applies_selected_variant(self):
        request = self.factory.put(
            "/api/ml/orchestrator/models/selected/",
            {
                "model_id": "xlam_7b_q4_k_m",
                "download": True,
                "wait": False,
            },
            format="json",
        )
        authenticate(request)

        manager = MagicMock()
        clear_selected_qs = MagicMock()
        selected_variant = SimpleNamespace(updated_at=SimpleNamespace(isoformat=lambda: "2026-03-11T10:00:00"))
        selected_variant_qs = MagicMock()
        selected_variant_qs.first.return_value = selected_variant
        manager.filter.side_effect = [
            clear_selected_qs,
            selected_variant_qs,
        ]

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

        payload = {
            "status": "selected",
            "selected_model_id": "xlam_7b_q4_k_m",
            "changed": True,
            "download_triggered": True,
            "orchestrator_status": orchestrator.status(),
        }
        with patch(
            "ml.api.views.select_orchestrator_model",
            return_value=ControlResult(payload),
        ) as select_mock:
            response = OrchestratorSelectView.as_view()(request)

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.data["status"], "selected")
        self.assertEqual(response.data["selected_model_id"], "xlam_7b_q4_k_m")
        self.assertTrue(response.data["changed"])
        self.assertTrue(response.data["download_triggered"])
        select_mock.assert_called_once_with(
            "xlam_7b_q4_k_m",
            download=True,
            wait=False,
        )

    def test_put_selects_auto_detected_local_variant_not_yet_in_db(self):
        request = self.factory.put(
            "/api/ml/orchestrator/models/selected/",
            {
                "model_id": "qwen2_7b_instruct_q4_0",
                "download": True,
                "wait": False,
            },
            format="json",
        )
        authenticate(request)

        manager = MagicMock()
        clear_selected_qs = MagicMock()
        selected_variant = SimpleNamespace(updated_at=SimpleNamespace(isoformat=lambda: "2026-04-21T10:00:00"))
        selected_variant_qs = MagicMock()
        selected_variant_qs.first.return_value = selected_variant
        manager.filter.side_effect = [
            clear_selected_qs,
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

        payload = {
            "status": "selected",
            "selected_model_id": "qwen2_7b_instruct_q4_0",
            "changed": True,
            "download_triggered": True,
            "orchestrator_status": orchestrator.status(),
        }
        with patch(
            "ml.api.views.select_orchestrator_model",
            return_value=ControlResult(payload),
        ) as select_mock:
            response = OrchestratorSelectView.as_view()(request)

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.data["selected_model_id"], "qwen2_7b_instruct_q4_0")
        select_mock.assert_called_once_with(
            "qwen2_7b_instruct_q4_0",
            download=True,
            wait=False,
        )


class RemoveModelViewTests(SimpleTestCase):
    def setUp(self):
        self.factory = APIRequestFactory()
        self.view = RemoveModelView.as_view()

    @patch("ml.api.views.DriftReport.objects.filter")
    @patch("ml.api.views.ModelStats.objects.filter")
    @patch("ml.api.views.refresh_runtime_state")
    @patch("ml.api.views.MLModelConfig.objects.filter")
    @patch("ml.api.views.get_registry")
    @patch("ml.api.views.transaction.atomic", return_value=nullcontext())
    @patch("ml.api.views.is_ready")
    def test_post_removes_model_and_related_records(
        self,
        mock_is_ready,
        mock_atomic,
        mock_get_registry,
        mock_model_filter,
        mock_refresh,
        mock_stats_filter,
        mock_drift_filter,
    ):
        mock_is_ready.return_value = True
        model = MagicMock()
        mock_model_filter.return_value = [model]
        mock_get_registry.return_value = SimpleNamespace(get=lambda model_id: None)

        request = self.factory.post("/api/ml/models/donor_v1/remove")
        authenticate(request)
        response = self.view(request, model_id="donor_v1")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.data, {"success": True})
        mock_model_filter.assert_called_once_with(model_id__in=["donor_v1"])
        model.delete.assert_called_once_with()
        mock_stats_filter.assert_called_once_with(model_id__in=["donor_v1"])
        mock_stats_filter.return_value.delete.assert_called_once_with()
        mock_drift_filter.assert_called_once_with(model_id__in=["donor_v1"])
        mock_drift_filter.return_value.delete.assert_called_once_with()
        mock_refresh.assert_called_once_with(
            "api-remove-model",
            force=True,
            warmup=False,
        )
        mock_atomic.assert_called_once_with()

    @patch("ml.api.views.MLModelConfig.objects.filter")
    @patch("ml.api.views.get_registry")
    @patch("ml.api.views.is_ready")
    def test_post_returns_404_when_model_is_unknown(
        self,
        mock_is_ready,
        mock_get_registry,
        mock_model_filter,
    ):
        mock_is_ready.return_value = True
        mock_get_registry.return_value = SimpleNamespace(get=lambda model_id: None)
        mock_model_filter.return_value = []

        request = self.factory.post("/api/ml/models/unknown/remove")
        authenticate(request)
        response = self.view(request, model_id="unknown")

        self.assertEqual(response.status_code, 404)
        self.assertEqual(response.data, {"error": "Unknown model: unknown"})

    @patch("ml.api.views.DriftReport.objects.filter")
    @patch("ml.api.views.ModelStats.objects.filter")
    @patch("ml.api.views.refresh_runtime_state")
    @patch("ml.api.views.MLModelConfig.objects.filter")
    @patch("ml.api.views.get_registry")
    @patch("ml.api.views.transaction.atomic", return_value=nullcontext())
    @patch("ml.api.views.is_ready")
    def test_post_resolves_runtime_aliases_to_remove_underlying_config(
        self,
        mock_is_ready,
        mock_atomic,
        mock_get_registry,
        mock_model_filter,
        mock_refresh,
        mock_stats_filter,
        mock_drift_filter,
    ):
        mock_is_ready.return_value = True
        model = MagicMock()
        mock_model_filter.return_value = [model]
        runtime = SimpleNamespace(
            model_id="donor_propensity_model_champion",
            slug="donor_propensity_model_champion",
            aliases=[
                "donor_propensity_model_champion",
                "donor_propensity_model",
            ],
        )
        mock_get_registry.return_value = SimpleNamespace(get=lambda model_id: runtime)

        request = self.factory.post(
            "/api/ml/models/donor_propensity_model_champion/remove"
        )
        authenticate(request)
        response = self.view(
            request,
            model_id="donor_propensity_model_champion",
        )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.data, {"success": True})
        mock_model_filter.assert_called_once_with(
            model_id__in=[
                "donor_propensity_model",
                "donor_propensity_model_champion",
            ]
        )
        mock_stats_filter.assert_called_once_with(
            model_id__in=[
                "donor_propensity_model",
                "donor_propensity_model_champion",
            ]
        )
        mock_drift_filter.assert_called_once_with(
            model_id__in=[
                "donor_propensity_model",
                "donor_propensity_model_champion",
            ]
        )
        model.delete.assert_called_once_with()
        mock_refresh.assert_called_once_with(
            "api-remove-model",
            force=True,
            warmup=False,
        )
        mock_atomic.assert_called_once_with()
