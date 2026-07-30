import json
import os
import django
from types import SimpleNamespace

# Setup Django environment for the test BEFORE any other imports
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "backendMulti.settings")
from django.apps import apps
if not apps.ready:
    django.setup()

from unittest.mock import MagicMock, patch
from asgiref.sync import async_to_sync
from django.test import SimpleTestCase
from rest_framework.test import APIRequestFactory, force_authenticate
from rest_framework import status

from ml.api.views import (
    HybridStockoutPredictView,
    PredictModelView,
    PredictNLView,
    ToolListView,
    ToolRunView,
)
from ml.core.feature_store import ResolvedFeaturePayload
from ml.services.control_center import ControlResult


async def _collect_async_stream(streaming_content):
    chunks = []
    async for chunk in streaming_content:
        chunks.append(chunk)
    return b"".join(chunks)


def _collect_stream(response):
    streaming_content = response.streaming_content
    if hasattr(streaming_content, "__aiter__"):
        return async_to_sync(_collect_async_stream)(streaming_content).decode("utf-8")
    return b"".join(streaming_content).decode("utf-8")


def _authenticate(request):
    force_authenticate(request, user=SimpleNamespace(is_authenticated=True))
    return request


class PredictModelViewTests(SimpleTestCase):
    def setUp(self):
        self.factory = APIRequestFactory()
        self.view = PredictModelView.as_view()

    @patch("ml.api.views.run_direct_model_prediction")
    def test_predict_model_success(
        self,
        mock_run_direct,
    ):
        mock_run_direct.return_value = ControlResult(
            {
                "tools_used": ["test_model"],
                "results": {
                    "summary": "test_model: 1",
                    "details": [
                        {
                            "tool": "test_model",
                            "result": {"prediction": 1},
                        }
                    ],
                },
            }
        )
        request_data = {
            "query": "test query",
            "features": {"age": 25, "bmi": 22.5}
        }
        request = self.factory.post("/api/ml/models/test_model/predict/", request_data, format="json")
        _authenticate(request)
        response = self.view(request, model_id="test_model")

        self.assertEqual(response.status_code, status.HTTP_200_OK)
        self.assertEqual(response.data["tools_used"], ["test_model"])
        self.assertEqual(response.data["results"]["summary"], "test_model: 1")
        self.assertEqual(
            response.data["results"]["details"][0]["result"]["prediction"],
            1,
        )
        self.assertNotIn("plan", response.data)
        self.assertNotIn("feature_resolution", response.data)
        self.assertEqual(mock_run_direct.call_args.args[0], "test_model")

    @patch("ml.api.views.run_direct_model_prediction")
    def test_predict_model_accepts_inputs_as_batch_predictions(
        self,
        mock_run_direct,
    ):
        payload_features = {
            "stockout_features_features__current_stock_units": 5,
            "stockout_features_features__avg_daily_usage": 3.5,
        }
        mock_run_direct.return_value = ControlResult(
            {
                "tools_used": ["stockout_days_predictor_champion"],
                "planner_mode": "direct_batch",
                "execution_results": [
                    {
                        "tool": "stockout_days_predictor_champion",
                        "output": {"rows": [{"prediction": 12}]},
                        "success": True,
                    }
                ],
            }
        )

        request = self.factory.post(
            "/api/ml/models/stockout_days_predictor_champion/predict/",
            {"inputs": [payload_features]},
            format="json",
        )
        _authenticate(request)
        response = self.view(request, model_id="stockout_days_predictor_champion")

        self.assertEqual(response.status_code, status.HTTP_200_OK)
        self.assertEqual(response.data["planner_mode"], "direct_batch")
        self.assertEqual(
            response.data["execution_results"][0]["output"]["rows"][0]["prediction"],
            12,
        )
        self.assertEqual(response.data["tools_used"], ["stockout_days_predictor_champion"])
        self.assertEqual(
            mock_run_direct.call_args.args[1]["inputs"],
            [payload_features],
        )

    @patch("ml.api.views.run_direct_model_prediction")
    def test_predict_model_uses_resolved_feast_features(
        self,
        mock_run_direct,
    ):
        mock_run_direct.return_value = ControlResult(
            {
                "tools_used": ["stockout_days_predictor_champion"],
                "results": {
                    "summary": "stockout_days_predictor_champion: 2",
                    "details": [
                        {
                            "tool": "stockout_days_predictor_champion",
                            "result": {"prediction": 2},
                        }
                    ],
                },
            }
        )

        request = self.factory.post(
            "/api/ml/models/stockout_days_predictor_champion/predict/",
            {"features": {"supply_id": "S001"}},
            format="json",
        )
        _authenticate(request)
        response = self.view(request, model_id="stockout_days_predictor_champion")

        self.assertEqual(response.status_code, status.HTTP_200_OK)
        self.assertEqual(
            response.data["results"]["details"][0]["result"]["prediction"],
            2,
        )
        self.assertEqual(response.data["tools_used"], ["stockout_days_predictor_champion"])
        self.assertNotIn("feature_resolution", response.data)
        self.assertEqual(mock_run_direct.call_args.args[0], "stockout_days_predictor_champion")

    @patch("ml.api.views.run_direct_model_prediction")
    def test_predict_model_not_ready(self, mock_run_direct):
        mock_run_direct.return_value = ControlResult(
            {"error": "ML models still loading.", "tools_used": []},
            status.HTTP_503_SERVICE_UNAVAILABLE,
        )

        request = self.factory.post("/api/ml/models/test_model/predict/", {}, format="json")
        _authenticate(request)
        response = self.view(request, model_id="test_model")
        
        self.assertEqual(response.status_code, status.HTTP_503_SERVICE_UNAVAILABLE)
        self.assertIn("error", response.data)

    @patch("ml.api.views.run_direct_model_prediction")
    def test_predict_model_not_found(self, mock_run_direct):
        mock_run_direct.return_value = ControlResult(
            {"error": "Unknown model: unknown", "tools_used": []},
            status.HTTP_404_NOT_FOUND,
        )

        request = self.factory.post("/api/ml/models/unknown/predict/", {"query": "test"}, format="json")
        _authenticate(request)
        response = self.view(request, model_id="unknown")
        
        self.assertEqual(response.status_code, status.HTTP_404_NOT_FOUND)
        self.assertIn("error", response.data)


class HybridStockoutPredictViewTests(SimpleTestCase):
    def setUp(self):
        self.factory = APIRequestFactory()
        self.view = HybridStockoutPredictView.as_view()

    @patch("ml.api.views.AlertEngine")
    @patch("ml.api.views.PredictionLog.objects.create")
    @patch("ml.api.views.build_hybrid_stockout_prediction")
    @patch("ml.api.views.find_hybrid_stockout_regression_runtime")
    @patch("ml.api.views.is_ready")
    @patch("ml.api.views.get_registry")
    @patch("ml.api.views.refresh_runtime_state")
    def test_stockout_hybrid_endpoint_returns_flat_hybrid_payload(
        self,
        mock_refresh,
        mock_get_registry,
        mock_is_ready,
        mock_find_runtime,
        mock_build_hybrid,
        mock_log_create,
        mock_alert_engine,
    ):
        mock_is_ready.return_value = True
        registry = MagicMock()
        mock_get_registry.return_value = registry
        runtime = MagicMock()
        runtime.model_id = "stockout_days_predictor"
        runtime.status = "loaded"
        runtime.load_error = None
        mock_find_runtime.return_value = runtime
        mock_build_hybrid.return_value = {
            "blood_component": "RBC",
            "blood_group": "O-",
            "location": "Hospital A",
            "risk_1_7_days": 0.78,
            "risk_8_30_days": 0.18,
            "risk_30_plus_days": 0.04,
            "estimated_days_until_stockout": 5,
            "stockout_probability": 0.78,
            "horizon": "1-7 days",
            "risk_level": "critical",
            "confidence": "medium",
            "recommended_action": "Urgent donor campaign or transfer request",
        }

        request = self.factory.post(
            "/api/ml/predict/stockout-hybrid/",
            {
                "features": {
                    "blood_component": "RBC",
                    "blood_group": "O-",
                    "location": "Hospital A",
                }
            },
            format="json",
        )
        _authenticate(request)
        response = self.view(request)

        self.assertEqual(response.status_code, status.HTTP_200_OK)
        self.assertEqual(response.data["results"]["estimated_days_until_stockout"], 5)
        self.assertEqual(response.data["results"]["stockout_probability"], 0.78)
        self.assertEqual(response.data["results"]["horizon"], "1-7 days")
        self.assertEqual(response.data["results"]["risk_level"], "critical")
        self.assertEqual(
            response.data["results"]["recommended_action"],
            "Urgent donor campaign or transfer request",
        )
        self.assertEqual(response.data["tools_used"], [runtime.model_id])
        mock_find_runtime.assert_called_once_with(registry, preferred_model_id=None)
        mock_log_create.assert_called_once()
        mock_alert_engine.return_value.evaluate_prediction.assert_called_once()


class ToolDirectViewTests(SimpleTestCase):
    def setUp(self):
        self.factory = APIRequestFactory()

    @patch("ml.api.views.is_ready")
    @patch("ml.api.views.get_orchestrator")
    @patch("ml.api.views.refresh_runtime_state")
    def test_tool_list_exposes_configured_tool_metadata(
        self,
        mock_refresh,
        mock_get_orchestrator,
        mock_is_ready,
    ):
        mock_is_ready.return_value = True
        orchestrator = MagicMock()
        orchestrator.list_configured_tools.return_value = [
            {
                "id": "db_tool",
                "name": "db_tool",
                "adapter": "db_query",
                "enabled": True,
                "available": True,
                "aliases": ["db_search"],
                "run_endpoint": "/api/ml/tools/db_tool/run/",
            }
        ]
        mock_get_orchestrator.return_value = orchestrator

        request = self.factory.get("/api/ml/tools/")
        _authenticate(request)
        response = ToolListView.as_view()(request)

        self.assertEqual(response.status_code, status.HTTP_200_OK)
        self.assertEqual(response.data["tools"][0]["id"], "db_tool")
        self.assertEqual(response.data["tools"][0]["aliases"], ["db_search"])

    @patch("ml.api.views.PredictionLog.objects.create")
    @patch("ml.api.views.is_ready")
    @patch("ml.api.views.get_orchestrator")
    @patch("ml.api.views.refresh_runtime_state")
    def test_tool_run_executes_configured_alias_without_planner(
        self,
        mock_refresh,
        mock_get_orchestrator,
        mock_is_ready,
        mock_log_create,
    ):
        mock_is_ready.return_value = True
        definition = SimpleNamespace(id="db_tool")
        orchestrator = MagicMock()
        orchestrator.configured_tool_id.return_value = "db_tool"
        orchestrator._tool_definition.return_value = definition
        orchestrator.execute_configured_tool.return_value = {
            "answer": "Returned 2 rows",
            "rows": [{"donor_id": "D001"}, {"donor_id": "D002"}],
        }
        orchestrator._result_value_for_summary.return_value = "Returned 2 rows"
        mock_get_orchestrator.return_value = orchestrator

        request = self.factory.post(
            "/api/ml/tools/db_search/run/",
            {
                "query": "list donors",
                "arguments": {"table": "donors", "aggregate": "list", "limit": 2},
            },
            format="json",
        )
        _authenticate(request)
        response = ToolRunView.as_view()(request, tool_id="db_search")

        self.assertEqual(response.status_code, status.HTTP_200_OK)
        self.assertEqual(response.data["tools_used"], ["db_tool"])
        self.assertEqual(response.data["planner_mode"], "direct_tool")
        orchestrator.execute_configured_tool.assert_called_once_with(
            "db_search",
            query="list donors",
            arguments={"table": "donors", "aggregate": "list", "limit": 2},
        )
        mock_log_create.assert_called_once()


class PredictNLViewTests(SimpleTestCase):
    def setUp(self):
        self.factory = APIRequestFactory()
        self.view = PredictNLView.as_view()

    @patch("ml.api.views.is_ready")
    @patch("ml.api.views.get_registry")
    @patch("ml.api.views.get_orchestrator")
    @patch("ml.api.views.refresh_runtime_state")
    def test_predict_nl_forces_single_model_when_query_names_it(
        self,
        mock_refresh,
        mock_get_orchestrator,
        mock_get_registry,
        mock_is_ready,
    ):
        mock_is_ready.return_value = True

        mock_runtime = MagicMock()
        mock_runtime.model_id = "donor_propensity_model_champion"
        mock_runtime.status = "loaded"
        mock_runtime.aliases = ["donor_propensity_model"]

        mock_registry = MagicMock()
        mock_registry.loaded.return_value = [mock_runtime]
        mock_registry.get.return_value = None
        mock_get_registry.return_value = mock_registry

        mock_orchestrator = MagicMock()
        mock_orchestrator.run.return_value = {
            "success": True,
            "execution_results": [
                {
                    "tool": mock_runtime.model_id,
                    "success": True,
                    "inputs": {"donor_id": "D001"},
                    "output": {
                        "query_kind": "batch_model_ranking",
                        "model_id": mock_runtime.model_id,
                        "row_count": 1,
                        "rows": [
                            {
                                "rank": 1,
                                "donor_id": "D001",
                                "blood_type": "O-",
                                "model_id": mock_runtime.model_id,
                                "model_score": 0.82,
                                "model_output": {
                                    "prediction": 1,
                                    "probability": 0.82,
                                    "used_inputs": {"smoker": 0},
                                },
                                "feature_resolution": {
                                    "repo_path": "/private/path",
                                    "resolved_feature_names": ["smoker"],
                                },
                            }
                        ],
                    },
                }
            ],
            "planner_mode": "llm",
            "plan": {"steps": []},
            "natural_language_response": "ok",
        }
        mock_get_orchestrator.return_value = mock_orchestrator

        request = self.factory.post(
            "/api/ml/predict/nl/",
            {"query": "Use donor_propensity_model_champion for donor D001"},
            format="json",
        )
        _authenticate(request)
        response = self.view(request)

        self.assertEqual(response.status_code, status.HTTP_200_OK)
        mock_orchestrator.run.assert_called_once_with(
            query="Use donor_propensity_model_champion for donor D001",
            provided_features={},
            forced_model_ids=["donor_propensity_model_champion"],
            top_k=None,
        )
        self.assertEqual(response.data["planner_mode"], "llm")
        self.assertEqual(response.data["execution_results"][0]["tool"], mock_runtime.model_id)
        self.assertEqual(response.data["execution_results"][0]["inputs"]["donor_id"], "D001")
        compact_row = response.data["execution_results"][0]["output"]["rows"][0]
        self.assertEqual(compact_row["donor_id"], "D001")
        self.assertEqual(compact_row["model_score"], 0.82)
        self.assertEqual(compact_row["prediction"], 1)
        self.assertNotIn("model_output", compact_row)
        self.assertNotIn("feature_resolution", compact_row)
        self.assertNotIn("used_inputs", str(response.data))

    @patch("ml.api.views.AlertEngine")
    @patch("ml.api.views.is_ready")
    @patch("ml.api.views.get_registry")
    @patch("ml.api.views.get_orchestrator")
    @patch("ml.api.views.refresh_runtime_state")
    def test_predict_nl_streams_progress_and_final_markdown(
        self,
        mock_refresh,
        mock_get_orchestrator,
        mock_get_registry,
        mock_is_ready,
        mock_alert_engine,
    ):
        mock_is_ready.return_value = True

        mock_runtime = MagicMock()
        mock_runtime.model_id = "donor_propensity_model_champion"
        mock_runtime.status = "loaded"
        mock_runtime.aliases = []

        mock_registry = MagicMock()
        mock_registry.loaded.return_value = [mock_runtime]
        mock_registry.get.return_value = None
        mock_get_registry.return_value = mock_registry

        markdown = "**Eligible donors**\n\n| donor | score |\n| --- | ---: |\n| D001 | 0.82 |"

        def fake_run(**kwargs):
            kwargs["event_callback"]({"type": "progress", "phase": "planning"})
            kwargs["event_callback"](
                {
                    "type": "plan",
                    "phase": "planned",
                    "steps": [
                        {
                            "index": 1,
                            "tool": mock_runtime.model_id,
                            "reasoning": "metadata match",
                        }
                    ],
                }
            )
            return {
                "success": True,
                "execution_results": [
                    {
                        "tool": mock_runtime.model_id,
                        "success": True,
                        "inputs": {"donor_id": "D001"},
                        "output": {
                            "rows": [
                                {
                                    "donor_id": "D001",
                                    "model_score": 0.82,
                                    "model_output": {
                                        "prediction": 1,
                                        "used_inputs": {"smoker": 0},
                                    },
                                }
                            ]
                        },
                    }
                ],
                "planner_mode": "llm",
                "plan": {"steps": []},
                "natural_language_response": markdown,
            }

        mock_orchestrator = MagicMock()
        mock_orchestrator.run.side_effect = fake_run
        mock_get_orchestrator.return_value = mock_orchestrator

        request = self.factory.post(
            "/api/ml/predict/nl/",
            {"query": "Rank eligible donors"},
            format="json",
            HTTP_ACCEPT="text/event-stream",
        )
        force_authenticate(request, user=SimpleNamespace(is_authenticated=True))
        response = self.view(request)
        body = _collect_stream(response)
        events = [
            json.loads(frame.removeprefix("data: "))
            for frame in body.strip().split("\n\n")
            if frame.startswith("data: ")
        ]

        self.assertEqual(response.status_code, status.HTTP_200_OK)
        self.assertIn("text/event-stream", response["Content-Type"])
        self.assertEqual(events[0]["type"], "progress")
        self.assertTrue(any(event["type"] == "plan" for event in events))
        final = events[-1]
        self.assertEqual(final["type"], "final")
        self.assertEqual(final["markdown"], markdown)
        self.assertNotIn("used_inputs", json.dumps(final))
        kwargs = mock_orchestrator.run.call_args.kwargs
        self.assertEqual(kwargs["query"], "Rank eligible donors")
        self.assertTrue(callable(kwargs["event_callback"]))
        mock_alert_engine.return_value.evaluate_prediction.assert_called_once()
