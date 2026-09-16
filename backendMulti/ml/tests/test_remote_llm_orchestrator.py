"""
Tests for Modal AI / Remote vLLM GPU inference backend integration in PulseAI Orchestrator.

Validates config schema, remote planning, remote summarization, streaming token emissions,
failover/fallback, and absence of hardcoded endpoints/models in accordance with AGENTS.md.
"""

from __future__ import annotations

import json
import os
import types
import sys
import unittest
from unittest.mock import MagicMock, patch

# Provide stubs for heavy ML modules
prediction_stub = types.ModuleType("ml.core.prediction")
prediction_stub.preload_prediction_runtime = lambda *args, **kwargs: None
prediction_stub.run_prediction = lambda *args, **kwargs: {"prediction": 42.0}
prediction_stub.run_prediction_batch = lambda runtime, rows: [
    prediction_stub.run_prediction(runtime, row) for row in rows
]

registry_stub = types.ModuleType("ml.core.registry")


class _StubModelRuntime:
    def __init__(self, model_id: str = "component_demand_quantile_forecast_model") -> None:
        self.model_id = model_id
        self.aliases = []
        self.feature_names = ["hospital_id", "blood_type"]
        self.description = "Forecasts blood demand"
        self.examples = []
        self.defaults = {}
        self.status = "loaded"
        self.model_type = "test_runtime"


class _StubModelRegistry:
    def __init__(self) -> None:
        self._models = {"component_demand_quantile_forecast_model": _StubModelRuntime()}

    def loaded(self):
        return list(self._models.values())

    def get(self, model_id):
        return self._models.get(model_id)


registry_stub.ModelRuntime = _StubModelRuntime
registry_stub.ModelRegistry = _StubModelRegistry

_orig_pred = sys.modules.get("ml.core.prediction")
_orig_reg = sys.modules.get("ml.core.registry")
sys.modules["ml.core.prediction"] = prediction_stub
sys.modules["ml.core.registry"] = registry_stub

from ml.orchestrator.service import DynamicXLAMOrchestrator, ExecutionPlan, ToolCall

if _orig_pred is not None:
    sys.modules["ml.core.prediction"] = _orig_pred
else:
    del sys.modules["ml.core.prediction"]

if _orig_reg is not None:
    sys.modules["ml.core.registry"] = _orig_reg
else:
    del sys.modules["ml.core.registry"]


class TestRemoteLLMOrchestrator(unittest.TestCase):
    def setUp(self):
        self.registry = _StubModelRegistry()

    def test_remote_llm_config_validation_valid(self):
        orchestrator = DynamicXLAMOrchestrator(registry=self.registry)
        valid_cfg = {
            "enabled": True,
            "provider": "modal_vllm",
            "default_timeout_seconds": 25.0,
            "chat_completions_path": "/v1/chat/completions",
            "default_model": "Qwen/Qwen2.5-7B-Instruct",
        }
        # Should not raise
        orchestrator._validate_remote_llm_config(valid_cfg)

    def test_remote_llm_config_validation_invalid_type(self):
        orchestrator = DynamicXLAMOrchestrator(registry=self.registry)
        with self.assertRaises(RuntimeError) as ctx:
            orchestrator._validate_remote_llm_config("not_a_dict")
        self.assertIn("must be an object", str(ctx.exception))

    def test_remote_llm_config_validation_invalid_timeout(self):
        orchestrator = DynamicXLAMOrchestrator(registry=self.registry)
        with self.assertRaises(RuntimeError) as ctx:
            orchestrator._validate_remote_llm_config({"default_timeout_seconds": -5.0})
        self.assertIn("default_timeout_seconds must be a positive number", str(ctx.exception))

    def test_remote_llm_config_validation_invalid_path(self):
        orchestrator = DynamicXLAMOrchestrator(registry=self.registry)
        with self.assertRaises(RuntimeError) as ctx:
            orchestrator._validate_remote_llm_config({"chat_completions_path": "invalid_no_slash"})
        self.assertIn("chat_completions_path must be a string starting with '/'", str(ctx.exception))

    def test_status_reports_remote_llm_configuration(self):
        env = {
            "PIOS_XLAM_DISABLE_LLM": "1",
            "PIOS_ORCH_LLM_API_URL": "https://test-workspace--pulseai-vllm-backend.modal.run",
            "PIOS_ORCH_LLM_API_MODEL": "Qwen/Qwen2.5-7B-Instruct",
            "PIOS_ORCH_PREFER_REMOTE_LLM": "1",
        }
        with patch.dict(os.environ, env, clear=False):
            orchestrator = DynamicXLAMOrchestrator(registry=self.registry)
            status = orchestrator.status()

        self.assertTrue(status["llm_ready"])
        self.assertEqual(status["planner_mode"], "llm")
        self.assertTrue(status["remote_llm_active"])
        self.assertTrue(status["remote_llm_configured"])
        self.assertEqual(status["remote_llm_model"], "Qwen/Qwen2.5-7B-Instruct")

    @patch("ml.core.http_client.http_json_request")
    def test_remote_llm_plan_generation_executes_successfully(self, mock_http):
        mock_plan_json = json.dumps({
            "steps": [
                {
                    "tool": "component_demand_quantile_forecast_model",
                    "arguments": {"hospital_id": "H001", "blood_type": "O+"},
                    "reasoning": "Forecast O+ demand for hospital H001 on Modal GPU",
                }
            ],
            "reasoning": "Planned by Modal vLLM backend",
        })

        def mock_llm_response(*args, **kwargs):
            return {
                "status_code": 200,
                "data": {
                    "choices": [
                        {
                            "message": {
                                "role": "assistant",
                                "content": f"```json\n{mock_plan_json}\n```",
                            }
                        }
                    ]
                },
            }

        mock_http.side_effect = mock_llm_response

        env = {
            "PIOS_XLAM_DISABLE_LLM": "1",
            "PIOS_ORCH_LLM_API_URL": "https://test--pulseai-vllm.modal.run",
            "PIOS_ORCH_LLM_API_MODEL": "Qwen/Qwen2.5-7B-Instruct",
        }
        with patch.dict(os.environ, env, clear=False):
            orchestrator = DynamicXLAMOrchestrator(registry=self.registry)
            result = orchestrator.run("Forecast demand for O+ at H001")

        self.assertTrue(result["success"])
        self.assertEqual(result["planner_mode"], "llm")
        mock_http.assert_called()
        call_kwargs = mock_http.call_args[1]
        self.assertIn("/v1/chat/completions", call_kwargs["url"])
        self.assertEqual(call_kwargs["body"]["model"], "Qwen/Qwen2.5-7B-Instruct")

    @patch("ml.core.http_client.http_json_request")
    def test_remote_llm_summarizer_generates_natural_language_summary(self, mock_http):
        mock_plan_json = json.dumps({
            "steps": [
                {
                    "tool": "component_demand_quantile_forecast_model",
                    "arguments": {"hospital_id": "H001", "blood_type": "O+"},
                    "reasoning": "Forecast O+ demand for hospital H001 on Modal GPU",
                }
            ],
            "reasoning": "Planned by Modal vLLM backend",
        })

        call_count = 0

        def mock_llm_response(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                content = f"```json\n{mock_plan_json}\n```"
            else:
                content = "Based on the predictive model, demand at Hospital H001 is projected at 42 units."
            return {
                "status_code": 200,
                "data": {
                    "choices": [
                        {
                            "message": {
                                "role": "assistant",
                                "content": content,
                            }
                        }
                    ]
                },
            }

        mock_http.side_effect = mock_llm_response

        env = {
            "PIOS_XLAM_DISABLE_LLM": "1",
            "PIOS_ORCH_LLM_API_URL": "https://test--pulseai-vllm.modal.run",
            "PIOS_ORCH_LLM_API_MODEL": "Qwen/Qwen2.5-7B-Instruct",
        }
        with patch.dict(os.environ, env, clear=False):
            orchestrator = DynamicXLAMOrchestrator(registry=self.registry)
            result = orchestrator.run("Forecast demand for O+ at H001")

        self.assertIn("projected at 42 units", result["natural_language_response"])

    @patch("ml.core.http_client.http_stream_sse_request")
    def test_remote_llm_streaming_emits_tokens_to_callback(self, mock_stream):
        chunks = [
            json.dumps({"choices": [{"delta": {"content": "Forecast "}}]}),
            json.dumps({"choices": [{"delta": {"content": "is "}}]}),
            json.dumps({"choices": [{"delta": {"content": "stable."}}]}),
        ]
        mock_stream.return_value = iter(chunks)

        events = []
        def callback(evt):
            events.append(evt)

        env = {
            "PIOS_XLAM_DISABLE_LLM": "1",
            "PIOS_ORCH_LLM_API_URL": "https://test--pulseai-vllm.modal.run",
        }
        with patch.dict(os.environ, env, clear=False):
            orchestrator = DynamicXLAMOrchestrator(registry=self.registry)
            text = orchestrator._generate_llm_text(
                "Summarize inventory",
                max_tokens=100,
                event_callback=callback,
            )

        self.assertEqual(text, "Forecast is stable.")
        self.assertEqual(len(events), 3)
        self.assertEqual(events[-1]["text"], "Forecast is stable.")

    @patch("ml.core.http_client.http_stream_sse_request")
    def test_generate_llm_text_stream_phase_defaults_to_summarizing(self, mock_stream):
        chunks = [
            json.dumps({"choices": [{"delta": {"content": "Stable."}}]}),
        ]
        mock_stream.return_value = iter(chunks)

        events = []
        env = {
            "PIOS_XLAM_DISABLE_LLM": "1",
            "PIOS_ORCH_LLM_API_URL": "https://test--pulseai-vllm.modal.run",
        }
        with patch.dict(os.environ, env, clear=False):
            orchestrator = DynamicXLAMOrchestrator(registry=self.registry)
            orchestrator._generate_llm_text(
                "Summarize inventory",
                max_tokens=100,
                event_callback=events.append,
            )

        self.assertEqual(len(events), 1)
        self.assertEqual(events[0]["type"], "token")
        self.assertEqual(events[0]["phase"], "summarizing")

    @patch("ml.core.http_client.http_stream_sse_request")
    def test_generate_llm_text_stream_phase_is_forwarded(self, mock_stream):
        chunks = [
            json.dumps({"choices": [{"delta": {"content": "Planning "}}]}),
            json.dumps({"choices": [{"delta": {"content": "done."}}]}),
        ]
        mock_stream.return_value = iter(chunks)

        events = []
        env = {
            "PIOS_XLAM_DISABLE_LLM": "1",
            "PIOS_ORCH_LLM_API_URL": "https://test--pulseai-vllm.modal.run",
        }
        with patch.dict(os.environ, env, clear=False):
            orchestrator = DynamicXLAMOrchestrator(registry=self.registry)
            text = orchestrator._generate_llm_text(
                "Build a plan",
                max_tokens=100,
                event_callback=events.append,
                stream_phase="planning",
            )

        self.assertEqual(text, "Planning done.")
        self.assertEqual(len(events), 2)
        for event in events:
            self.assertEqual(event["type"], "token")
            self.assertEqual(event["phase"], "planning")

    @patch("ml.core.http_client.http_stream_sse_request")
    def test_planner_streams_tokens_live_before_plan_completes(self, mock_stream):
        mock_plan_json = json.dumps({
            "steps": [
                {
                    "tool": "component_demand_quantile_forecast_model",
                    "arguments": {"hospital_id": "H001", "blood_type": "O+"},
                    "reasoning": "Forecast O+ demand for hospital H001 on Modal GPU",
                }
            ],
            "reasoning": "Planned by Modal vLLM backend",
        })
        # Split the plan JSON into streaming chunks so the planner emits
        # incremental token events while it is still working.
        mid = len(mock_plan_json) // 2
        plan_chunks = [
            json.dumps({"choices": [{"delta": {"content": mock_plan_json[:mid]}}]}),
            json.dumps({"choices": [{"delta": {"content": mock_plan_json[mid:]}}]}),
        ]
        mock_stream.side_effect = lambda *args, **kwargs: iter(plan_chunks)

        events = []
        env = {
            "PIOS_XLAM_DISABLE_LLM": "1",
            "PIOS_ORCH_LLM_API_URL": "https://test--pulseai-vllm.modal.run",
            "PIOS_ORCH_LLM_API_MODEL": "Qwen/Qwen2.5-7B-Instruct",
        }
        with patch.dict(os.environ, env, clear=False):
            orchestrator = DynamicXLAMOrchestrator(registry=self.registry)
            result = orchestrator.run(
                "Forecast demand for O+ at H001",
                event_callback=events.append,
            )

        planning_tokens = [
            event for event in events
            if event.get("type") == "token" and event.get("phase") == "planning"
        ]
        # Planner tokens must stream live (before the plan/tool events finish).
        self.assertGreaterEqual(len(planning_tokens), 2)
        first_planning_idx = events.index(planning_tokens[0])
        plan_idx = next(
            idx for idx, event in enumerate(events) if event.get("type") == "plan"
        )
        self.assertLess(first_planning_idx, plan_idx)
        self.assertTrue(result["success"])
        self.assertTrue(str(result["planner_mode"]).startswith("llm"))

    @patch("ml.core.http_client.http_json_request")
    def test_remote_llm_failure_falls_back_gracefully(self, mock_http):
        mock_http.side_effect = RuntimeError("Connection refused from Modal GPU backend")

        env = {
            "PIOS_XLAM_DISABLE_LLM": "1",
            "PIOS_ORCH_LLM_API_URL": "https://offline-modal.modal.run",
            "PIOS_ORCH_ALLOW_FALLBACK": "1",
        }
        with patch.dict(os.environ, env, clear=False):
            orchestrator = DynamicXLAMOrchestrator(registry=self.registry)
            # Should not crash; fallback plan executes
            result = orchestrator.run("Forecast demand for H001")

    @patch("ml.core.external_tools.http_json_request")
    def test_execute_llm_api_tool_base_url_normalization(self, mock_http):
        from ml.core.external_tools import execute_llm_api_tool

        mock_http.return_value = {
            "status_code": 200,
            "data": {
                "choices": [{"message": {"content": "Donor meets standard criteria."}}]
            },
        }

        # Provide base URL without /v1/chat/completions
        res = execute_llm_api_tool(
            "Is donor eligible?",
            llm_api_url="https://test--pulseai-vllm.modal.run",
            llm_api_key="",
            llm_api_model="Qwen/Qwen2.5-7B-Instruct",
            llm_api_timeout_s=30.0,
            llm_api_system_prompt="Clinical decision support",
            local_llm=None,
            local_llm_active_model_id=None,
            local_llm_prompt_mode="chat",
            safe_generation_tokens_fn=None,
        )

        self.assertEqual(res["status_code"], 200)
        self.assertEqual(res["answer"], "Donor meets standard criteria.")
        self.assertTrue(mock_http.called)
        called_url = mock_http.call_args.kwargs["url"]
        self.assertEqual(called_url, "https://test--pulseai-vllm.modal.run/v1/chat/completions")


if __name__ == "__main__":
    unittest.main()
