import json
import importlib
import os
import sqlite3
import sys
import tempfile
import types
import unittest
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import patch

prediction_stub = types.ModuleType("ml.core.prediction")
prediction_stub.preload_prediction_runtime = lambda *args, **kwargs: None
prediction_stub.run_prediction = lambda *args, **kwargs: {}
prediction_stub.run_prediction_batch = lambda runtime, rows: [
    prediction_stub.run_prediction(runtime, row) for row in rows
]

registry_stub = types.ModuleType("ml.core.registry")


class _StubModelRuntime:
    def __init__(self, model_id: str = "") -> None:
        self.model_id = model_id
        self.aliases = []
        self.feature_names = []
        self.description = ""
        self.examples = []
        self.defaults = {}
        self.status = "loaded"
        self.model_type = "test_runtime"


class _StubModelRegistry:
    pass


registry_stub.ModelRuntime = _StubModelRuntime
registry_stub.ModelRegistry = _StubModelRegistry

_original_prediction_module = sys.modules.get("ml.core.prediction")
_original_registry_module = sys.modules.get("ml.core.registry")
sys.modules["ml.core.prediction"] = prediction_stub
sys.modules["ml.core.registry"] = registry_stub

from ml.core import external_tools
from ml.core.feature_extraction import extract_nl_features
from ml.orchestrator.service import (
    DynamicXLAMOrchestrator,
    ExecutionPlan,
    ExecutionResult,
    ToolCall,
)

if _original_prediction_module is not None:
    sys.modules["ml.core.prediction"] = _original_prediction_module
else:
    del sys.modules["ml.core.prediction"]

if _original_registry_module is not None:
    sys.modules["ml.core.registry"] = _original_registry_module
else:
    del sys.modules["ml.core.registry"]

from ml.core import model_loading as _model_loading  # noqa: E402
from ml.orchestrator import service as _orchestrator_service  # noqa: E402

importlib.reload(_model_loading)
_orchestrator_service = importlib.reload(_orchestrator_service)
DynamicXLAMOrchestrator = _orchestrator_service.DynamicXLAMOrchestrator
ExecutionPlan = _orchestrator_service.ExecutionPlan
ExecutionResult = _orchestrator_service.ExecutionResult
ToolCall = _orchestrator_service.ToolCall


class _DummyRegistry:
    def __init__(self) -> None:
        self._models = {}

    def loaded(self):
        return list(self._models.values())

    def get(self, model_id):
        return self._models.get(model_id)


BACKEND_ROOT = Path(__file__).resolve().parents[2]
EXTERNAL_TOOLS_CONFIG_PATH = BACKEND_ROOT / "ml" / "config" / "config.json"
STOCKOUT_CONFIG_PATH = BACKEND_ROOT / "ml" / "config" / "stockout_hybrid.json"
ACTIVE_STOCKOUT_MODEL_ID = "component_inventory_risk_simulator"
ACTIVE_STOCKOUT_SENTINEL_ID = "stockout_time_to_event_hazard_model"
ACTIVE_DONOR_MODEL_ID = "donor_next_donation_hazard_model"
ACTIVE_IDEAL_DONOR_MODEL_ID = "ideal_donor_probability_model"


def _stockout_hybrid_defaults() -> dict:
    payload = json.loads(STOCKOUT_CONFIG_PATH.read_text(encoding="utf-8"))
    return {
        "hybrid_stockout": payload["hybrid_stockout"],
        "output": {"name": "days_until_stockout"},
    }


class _HandshakeLLM:
    def __call__(self, prompt, **kwargs):
        return {"choices": [{"text": '{"ok":true}'}]}

    def create_chat_completion(self, **kwargs):
        return {
            "choices": [
                {
                    "message": {
                        "content": '{"ok":true}',
                    }
                }
            ]
        }


class OrchestratorExternalToolsTests(unittest.TestCase):
    def test_extract_nl_features_reads_donor_entity_id(self):
        self.assertEqual(
            extract_nl_features("Predict donation propensity for donor 25335")[
                "donor_id"
            ],
            "25335",
        )
        self.assertEqual(
            extract_nl_features("Predict donation propensity for donor D0000001")[
                "donor_id"
            ],
            "D0000001",
        )

    def test_extract_nl_features_reads_bare_hospital_ids(self):
        self.assertEqual(
            extract_nl_features("compare stockout between H001 and H002")[
                "hospital_ids"
            ],
            ["H001", "H002"],
        )

    def test_extract_nl_features_reads_prefixed_age_phrase(self):
        self.assertEqual(
            extract_nl_features("Can a 35 year old donor with BMI 24 donate blood now?"),
            {"age": 35, "bmi": 24.0},
        )

    def test_external_tool_config_reload_updates_metadata_without_rebuild(self):
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp_dir:
            config_path = Path(tmp_dir) / "config.json"
            initial = self._minimal_external_tool_config(
                {
                    "prompts": {"planning": "initial planning prompt"},
                    "planning_limits": {"tool_description_chars": 120},
                }
            )
            initial["external_tools"][1]["description"] = "Initial DB description"
            config_path.write_text(json.dumps(initial), encoding="utf-8")

            env = {
                "PIOS_XLAM_DISABLE_LLM": "1",
                "PIOS_ORCH_EXTERNAL_TOOLS_CONFIG": str(config_path),
                "PIOS_ORCH_EXTERNAL_FALLBACK_ORDER": "",
            }
            with patch.dict(os.environ, env, clear=False):
                os.environ.pop("PIOS_ORCH_PLANNER_TOOL_DESCRIPTION_CHARS", None)
                orchestrator = DynamicXLAMOrchestrator(registry=_DummyRegistry())
                db_tool = next(
                    tool
                    for tool in orchestrator._planner_external_tools()
                    if tool["name"] == "db_tool"
                )
                self.assertEqual(db_tool["description"], "Initial DB description")
                self.assertEqual(
                    orchestrator.prompt_templates["planning"],
                    "initial planning prompt",
                )
                self.assertEqual(orchestrator.planner_tool_description_chars, 120)

                updated = self._minimal_external_tool_config(
                    {
                        "prompts": {"planning": "reloaded planning prompt"},
                        "fallback_aliases": ["configured fallback"],
                        "planning_limits": {"tool_description_chars": 180},
                    }
                )
                updated["external_tools"][1]["description"] = "Reloaded DB description"
                config_path.write_text(json.dumps(updated), encoding="utf-8")

                summary = orchestrator.reload_external_tools_config(reason="test")

                self.assertTrue(summary["changed"])
                db_tool = next(
                    tool
                    for tool in orchestrator._planner_external_tools()
                    if tool["name"] == "db_tool"
                )
                self.assertEqual(db_tool["description"], "Reloaded DB description")
                self.assertEqual(
                    orchestrator.prompt_templates["planning"],
                    "reloaded planning prompt",
                )
                self.assertIn("configured_fallback", orchestrator.fallback_aliases)
                self.assertEqual(orchestrator.planner_tool_description_chars, 180)

    def _hospital_shortage_runtime(self):
        runtime = _StubModelRuntime(ACTIVE_STOCKOUT_MODEL_ID)
        runtime.aliases = ["stockout_hybrid_regression"]
        runtime.description = (
            "Hospital component inventory risk simulator trained on operational stress features."
        )
        runtime.feature_names = [
            "hospital_id",
            "blood_type",
            "component_type",
            "current_inventory",
            "units_used",
            "units_collected",
            "wastage",
            "safety_threshold_units",
        ]
        runtime.defaults = _stockout_hybrid_defaults()
        runtime.examples = [
            {
                "kind": "good",
                "user_query": (
                    "Given hospital H001, O- RBC, current inventory 15, and "
                    "usage today 8, forecast stockout risk."
                ),
            },
            {
                "kind": "bad",
                "user_query": (
                    "What was the average stock_end in historical hospital rows "
                    "where holiday was false and trauma_cases was 10?"
                ),
            },
        ]
        return runtime

    def _hospital_shortage_query_and_features(self):
        query = (
            "Predict hospital stockout for: hospital_id=H001, blood_type=O-, "
            "component_type=RBC, current_inventory=15, units_used=8"
        )
        features = {
            "hospital_id": "H001",
            "blood_type": "O-",
            "component_type": "RBC",
            "hospital_supply_features_features__current_inventory": 15,
            "current_inventory": 15,
            "units_used": 8,
        }
        return query, features

    def _minimal_external_tool_config(self, orchestrator_payload=None):
        return {
            "external_tools": [
                {
                    "id": "stockout_hybrid",
                    "name": "stockout_hybrid",
                    "type": "external",
                    "enabled": True,
                    "adapter": "stockout_hybrid",
                    "entrypoint": "ml.core.hybrid_stockout.build_hybrid_stockout_prediction",
                    "permissions": ["run_prediction"],
                    "timeout_seconds": 30,
                    "description": "Configured stockout tool.",
                    "input_hints": "query",
                    "input_schema": {"type": "object"},
                    "output_schema": {"type": "object"},
                },
                {
                    "id": "db_tool",
                    "name": "db_tool",
                    "type": "external",
                    "enabled": True,
                    "adapter": "db_query",
                    "entrypoint": "ml.core.external_tools.execute_db_tool",
                    "permissions": ["read_reference_data"],
                    "timeout_seconds": 30,
                    "description": "Configured DB tool.",
                    "input_hints": "table, field, aggregate",
                    "input_schema": {"type": "object"},
                    "output_schema": {
                        "type": "object",
                        "properties": {"rows": {"type": "array"}},
                    },
                },
            ],
            "orchestrator": orchestrator_payload or {},
        }

    @contextmanager
    def _orchestrator_with_tools_enabled(
        self,
        registry,
        *,
        simulation: bool = False,
        stockout_hybrid: bool = False,
        extra_env: dict[str, str] | None = None,
    ):
        """Build an orchestrator against a temporary production config with
        legacy tools explicitly re-enabled for focused compatibility tests."""
        payload = json.loads(EXTERNAL_TOOLS_CONFIG_PATH.read_text(encoding="utf-8"))
        enabled_ids = {
            tool_id
            for tool_id, enabled in {
                "simulation": simulation,
                "stockout_hybrid": stockout_hybrid,
            }.items()
            if enabled
        }
        for tool in payload["external_tools"]:
            if tool.get("id") in enabled_ids:
                tool["enabled"] = True

        with tempfile.TemporaryDirectory() as tmp_dir:
            config_path = Path(tmp_dir) / "config.json"
            config_path.write_text(json.dumps(payload), encoding="utf-8")
            env = {
                "PIOS_XLAM_DISABLE_LLM": "1",
                "PIOS_ORCH_EXTERNAL_TOOLS_CONFIG": str(config_path),
            }
            env.update(extra_env or {})
            with patch.dict(os.environ, env, clear=False):
                yield DynamicXLAMOrchestrator(registry=registry)

    def test_production_config_exposes_only_db_and_llm_planner_tools(self):
        env = {
            "PIOS_XLAM_DISABLE_LLM": "1",
            "PIOS_ORCH_ENABLE_DB_TOOL": "1",
            "PIOS_ORCH_ENABLE_SEARCH_TOOL": "1",
            "PIOS_ORCH_ENABLE_SIMULATION_TOOL": "1",
            "PIOS_ORCH_SIMULATION_TOOL_NAME": "simulation",
            "PIOS_ORCH_STOCKOUT_HYBRID_TOOL_NAME": "stockout_hybrid",
        }
        registry = _DummyRegistry()
        stockout = self._hospital_shortage_runtime()
        registry._models[stockout.model_id] = stockout

        with patch.dict(os.environ, env, clear=False):
            orchestrator = DynamicXLAMOrchestrator(registry=registry)
            # llm_api is a configured planner tool once a local planner is ready.
            orchestrator._llm = object()
            planner_tools = {
                tool["name"] for tool in orchestrator._planner_external_tools()
            }
            description = orchestrator._build_tools_description([stockout])

        self.assertEqual(planner_tools, {"db_tool", "llm_api"})
        self.assertEqual(orchestrator.simulation_tool_name, "")
        self.assertEqual(orchestrator.stockout_hybrid_tool_name, "")
        self.assertFalse(orchestrator.simulation_tool_enabled)
        self.assertFalse(orchestrator._hide_stockout_prediction_models())
        self.assertIn(f"Tool: {stockout.model_id}", description)
        self.assertNotIn("Tool: stockout_hybrid", description)

    def test_production_stockout_fallback_uses_configured_model(self):
        env = {"PIOS_XLAM_DISABLE_LLM": "1"}
        registry = _DummyRegistry()
        stockout = self._hospital_shortage_runtime()
        registry._models[stockout.model_id] = stockout

        with patch.dict(os.environ, env, clear=False):
            orchestrator = DynamicXLAMOrchestrator(registry=registry)
            plan = orchestrator._fallback_plan(
                "will there be a stockout in hospital central",
                [stockout],
                1,
            )
            what_if_plan = orchestrator._fallback_plan(
                "What would happen if blood donors dropped by 50% in the next week?",
                [stockout],
                1,
            )

        self.assertEqual(plan.steps[0].name, stockout.model_id)
        self.assertNotEqual(plan.steps[0].name, "stockout_hybrid")
        self.assertTrue(
            all(step.name not in {"", "simulation"} for step in what_if_plan.steps)
        )

    def test_workflow_examples_skip_unavailable_configured_tools(self):
        env = {"PIOS_XLAM_DISABLE_LLM": "1"}
        unavailable_example = {
            "id": "disabled_tool_example",
            "user_query": "what if donors drop",
            "trigger_terms": ["what if", "donors"],
            "steps": [{"tool": "simulation", "arguments": {"query": "$request.query"}}],
        }
        with patch.dict(os.environ, env, clear=False):
            orchestrator = DynamicXLAMOrchestrator(registry=_DummyRegistry())
            orchestrator.dynamic_planning_config = {
                "workflow_examples": [unavailable_example]
            }
            orchestrator.planner_workflow_example_limit = 1
            plan = orchestrator._workflow_example_plan("what if donors drop")
            examples = orchestrator._workflow_examples_block("what if donors drop")

        self.assertIsNone(plan)
        self.assertEqual(examples, "")

    def test_local_gguf_uses_metadata_capabilities_when_available(self):
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp_dir:
            qwen_path = Path(tmp_dir) / "qwen2-7b-instruct-q4_0.gguf"
            with qwen_path.open("wb") as handle:
                handle.truncate(128 * 1024 * 1024)

            env = {
                "PIOS_XLAM_DISABLE_LLM": "1",
                "PIOS_XLAM_MODEL_DIR": tmp_dir,
            }
            with patch.dict(os.environ, env, clear=False):
                with patch.object(
                    DynamicXLAMOrchestrator,
                    "_inspect_model_metadata",
                    return_value={
                        "display_name": "Qwen2 7B Instruct Q4_0",
                        "architecture": "qwen2",
                        "prompt_mode": "chat",
                        "supports_chat_completion": True,
                        "has_chat_template": True,
                        "context_length": 32768,
                    },
                ):
                    orchestrator = DynamicXLAMOrchestrator(registry=_DummyRegistry())
                    models = orchestrator.list_available_models()

        qwen = next(
            row for row in models if row["id"] == "qwen2_7b_instruct_q4_0"
        )
        self.assertEqual(qwen["repo_id"], "local")
        self.assertTrue(qwen["local_exists"])
        self.assertTrue(qwen["auto_detected"])
        self.assertTrue(qwen["local_only"])
        self.assertEqual(qwen["model_family"], "qwen2")
        self.assertEqual(qwen["prompt_mode"], "chat")
        self.assertTrue(qwen["supports_chat_completion"])

    def test_local_gguf_sidecar_override_wins(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            qwen_path = Path(tmp_dir) / "qwen2-7b-instruct-q4_0.gguf"
            sidecar_path = Path(f"{qwen_path}.capabilities.json")
            with qwen_path.open("wb") as handle:
                handle.truncate(128 * 1024 * 1024)
            sidecar_path.write_text(
                '{"display_name":"My Local Model","prompt_mode":"completion"}',
                encoding="utf-8",
            )

            env = {
                "PIOS_XLAM_DISABLE_LLM": "1",
                "PIOS_XLAM_MODEL_DIR": tmp_dir,
            }
            with patch.dict(os.environ, env, clear=False):
                orchestrator = DynamicXLAMOrchestrator(registry=_DummyRegistry())
                models = orchestrator.list_available_models()

        qwen = next(
            row for row in models if row["id"] == "qwen2_7b_instruct_q4_0"
        )
        self.assertEqual(qwen["name"], "My Local Model")
        self.assertEqual(qwen["prompt_mode"], "completion")

    def test_local_gguf_capabilities_are_loaded_from_cache(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            qwen_path = Path(tmp_dir) / "qwen2-7b-instruct-q4_0.gguf"
            with qwen_path.open("wb") as handle:
                handle.truncate(128 * 1024 * 1024)

            env = {
                "PIOS_XLAM_DISABLE_LLM": "1",
                "PIOS_XLAM_MODEL_DIR": tmp_dir,
            }
            with patch.dict(os.environ, env, clear=False):
                orchestrator = DynamicXLAMOrchestrator(registry=_DummyRegistry())
                orchestrator._update_capability_cache(
                    qwen_path.resolve(),
                    {
                        "display_name": "Cached Qwen",
                        "prompt_mode": "chat",
                        "supports_chat_completion": True,
                    },
                )
                refreshed = DynamicXLAMOrchestrator(registry=_DummyRegistry())
                models = refreshed.list_available_models()

        qwen = next(
            row for row in models if row["id"] == "qwen2_7b_instruct_q4_0"
        )
        self.assertEqual(qwen["name"], "Cached Qwen")
        self.assertEqual(qwen["prompt_mode"], "chat")

    def test_existing_xlam_selection_stays_stable_when_qwen_is_added(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            xlam_path = Path(tmp_dir) / "xLAM-7b-fc-r.Q2_K.gguf"
            qwen_path = Path(tmp_dir) / "qwen2-7b-instruct-q4_0.gguf"
            with xlam_path.open("wb") as handle:
                handle.truncate(128 * 1024 * 1024)
            with qwen_path.open("wb") as handle:
                handle.truncate(160 * 1024 * 1024)

            env = {
                "PIOS_XLAM_DISABLE_LLM": "1",
                "PIOS_XLAM_MODEL_DIR": tmp_dir,
            }
            with patch.dict(os.environ, env, clear=False):
                orchestrator = DynamicXLAMOrchestrator(registry=_DummyRegistry())
                status = orchestrator.status()

        self.assertEqual(orchestrator._selected_variant.model_id, "xlam_7b_q2_k")
        self.assertTrue(status["selected_model_exists"])
        self.assertTrue(
            str(status["selected_model_path"]).endswith("xLAM-7b-fc-r.Q2_K.gguf")
        )

    def test_built_in_xlam_keeps_completion_mode_after_handshake(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            xlam_path = Path(tmp_dir) / "xLAM-7b-fc-r-Q2_K.gguf"
            with xlam_path.open("wb") as handle:
                handle.truncate(128 * 1024 * 1024)

            env = {
                "PIOS_XLAM_DISABLE_LLM": "1",
                "PIOS_XLAM_MODEL_DIR": tmp_dir,
            }
            with patch.dict(os.environ, env, clear=False):
                with patch.object(
                    DynamicXLAMOrchestrator,
                    "_inspect_model_metadata",
                    return_value={
                        "display_name": "xLAM 7B Q2_K",
                        "architecture": "xlam",
                        "prompt_mode": "chat",
                        "supports_chat_completion": True,
                        "has_chat_template": True,
                    },
                ):
                    orchestrator = DynamicXLAMOrchestrator(registry=_DummyRegistry())
                    variant = next(
                        v
                        for v in orchestrator._available_variants
                        if v.model_id == "xlam_7b_q2_k"
                    )
                    capabilities = orchestrator._resolved_capabilities(
                        xlam_path.resolve(),
                        variant=variant,
                        llm=_HandshakeLLM(),
                        require_handshake=True,
                    )

        self.assertEqual(capabilities["prompt_mode"], "completion")
        self.assertEqual(capabilities["probed_prompt_mode"], "completion")

    def test_model_steps_resolve_feast_features_before_prediction(self):
        env = {
            "PIOS_XLAM_DISABLE_LLM": "1",
            "PIOS_ORCH_ENABLE_DB_TOOL": "0",
            "PIOS_ORCH_ENABLE_SEARCH_TOOL": "0",
        }
        runtime = _StubModelRuntime("donor_propensity_model_champion")
        runtime.feature_names = ["donor_features_features__lat"]

        registry = _DummyRegistry()
        registry._models[runtime.model_id] = runtime

        with self._orchestrator_with_tools_enabled(
            registry,
            stockout_hybrid=True,
            extra_env=env,
        ) as orchestrator:
            fake_plan = ExecutionPlan(
                steps=[
                    ToolCall(
                        name=runtime.model_id,
                        arguments={"donor_id": "D001"},
                        reasoning="force donor model",
                    )
                ],
                reasoning="single model test",
                is_multi_step=False,
            )
            with patch.object(
                orchestrator,
                "_plan_execution",
                return_value=(fake_plan, "llm"),
            ):
                with patch(
                    "ml.orchestrator.service.resolve_prediction_features"
                ) as mock_resolve:
                    mock_resolve.return_value = types.SimpleNamespace(
                        features={"donor_features_features__lat": 36.7},
                        metadata={
                            "source": "feast_online",
                            "feature_service": "donor_service",
                        },
                    )
                    with patch("ml.orchestrator.service.run_prediction_with_fallback") as mock_predict:
                        mock_predict.return_value = {"prediction": 1}
                        result = orchestrator.run(
                            "Predict donor propensity",
                            provided_features={"availability": 1},
                            forced_model_ids=[runtime.model_id],
                        )

        mock_resolve.assert_called_once_with(
            runtime,
            {"donor_id": "D001", "availability": 1},
            forecast_horizon=None,
        )
        mock_predict.assert_called_once_with(
            runtime,
            {"donor_features_features__lat": 36.7},
            registry=registry,
        )
        self.assertTrue(result["success"])
        self.assertEqual(
            result["execution_results"][0]["output"]["feature_resolution"]["source"],
            "feast_online",
        )

    def test_db_plan_passes_structured_arguments_to_db_tool(self):
        env = {
            "PIOS_XLAM_DISABLE_LLM": "1",
            "PIOS_ORCH_ENABLE_DB_TOOL": "1",
            "PIOS_ORCH_ENABLE_SEARCH_TOOL": "0",
        }
        with patch.dict(os.environ, env, clear=False):
            orchestrator = DynamicXLAMOrchestrator(registry=_DummyRegistry())
            fake_plan = ExecutionPlan(
                steps=[
                    ToolCall(
                        name=orchestrator.db_tool_name,
                        arguments={
                            "table": "donors",
                            "field": "blood_type",
                            "aggregate": "count",
                            "limit": 10,
                        },
                        reasoning="Use the bounded Postgres DB tool.",
                    )
                ],
                reasoning="structured db test",
                is_multi_step=False,
            )
            with patch.object(
                orchestrator,
                "_plan_execution",
                return_value=(fake_plan, "llm"),
            ):
                with patch.object(
                    orchestrator,
                    "_execute_db_fallback",
                    return_value={
                        "tool_type": "db_tool",
                        "answer": "Returned 2 bounded result row(s).",
                        "rows": [
                            {"group_value": "O+", "value": 12},
                            {"group_value": "A+", "value": 9},
                        ],
                    },
                ) as mock_db:
                    result = orchestrator.run(
                        "What are the most common donor blood types?"
                    )

        mock_db.assert_called_once_with(
            "What are the most common donor blood types?",
            {
                "table": "donors",
                "field": "blood_type",
                "aggregate": "count",
                "limit": 10,
            },
        )
        self.assertTrue(result["success"])
        self.assertEqual(result["execution_results"][0]["tool"], orchestrator.db_tool_name)

    def test_execute_db_tool_routes_structured_requests_to_postgres_helper(self):
        with patch.object(
            external_tools,
            "_execute_structured_postgres_query",
            return_value={
                "tool_type": "db_tool",
                "provider": "postgres:public.orchestrator_donors",
                "answer": "Returned 1 bounded result row(s).",
                "rows": [{"value": 67}],
            },
        ) as mock_query:
            result = external_tools.execute_db_tool(
                query="ignored",
                table="donors",
                field="age",
                aggregate="max",
                limit=25,
                db_schema="public",
                donors_table_name="orchestrator_donors",
                hospitals_table_name="orchestrator_hospitals",
                db_supply_csv_path=Path("ml-backend/datasets/synthetic_bloodbank_daily.csv"),
                db_sqlite_path="",
                db_sqlite_table="blood_supply",
                db_series_limit=24,
            )

        self.assertEqual(result["tool_type"], "db_tool")
        self.assertEqual(mock_query.call_count, 1)
        kwargs = mock_query.call_args.kwargs
        self.assertEqual(kwargs["query"], "ignored")
        self.assertEqual(kwargs["table"], "donors")
        self.assertEqual(kwargs["field"], "age")
        self.assertEqual(kwargs["aggregate"], "max")
        self.assertEqual(kwargs["limit"], 25)

    def test_execute_db_tool_uses_sqlite_fallback_when_postgres_is_down(self):
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp_dir:
            tmp_path = Path(tmp_dir)
            donor_csv = tmp_path / "donors.csv"
            hospital_csv = tmp_path / "hospitals.csv"
            sqlite_path = tmp_path / "fallback.sqlite3"
            donor_csv.write_text(
                "\n".join(
                    [
                        "donor_id,age,sex,country_code,region,blood_type,eligible_to_donate,"
                        "donation_count_last_12m,is_regular_donor,is_rare_type,recency_days,"
                        "donation_propensity_score",
                        "1,41,F,DZ,Algiers,O-,true,3,true,true,120,80",
                    ]
                ),
                encoding="utf-8",
            )
            hospital_csv.write_text(
                "\n".join(
                    [
                        "date,hospital_id,blood_type,stock_start,units_collected,units_used,wastage,stock_end,critical_stock",
                        "2026-04-20,H001,O-,10,5,3,0,12,false",
                    ]
                ),
                encoding="utf-8",
            )

            external_tools._SQLITE_BOOTSTRAP_CACHE.clear()
            with patch.object(
                external_tools,
                "_postgres_connect",
                side_effect=RuntimeError("postgres down"),
            ):
                result = external_tools.execute_db_tool(
                    query="ignored",
                    table="donors",
                    field="age",
                    aggregate="max",
                    limit=10,
                    db_schema="public",
                    donors_table_name="orchestrator_donors",
                    hospitals_table_name="orchestrator_hospitals",
                    db_supply_csv_path=hospital_csv,
                    db_sqlite_path=str(sqlite_path),
                    db_sqlite_table="blood_supply",
                    db_series_limit=24,
                    csv_candidates=[donor_csv],
                )

            self.assertTrue(str(result["provider"]).startswith("sqlite:"))
            self.assertEqual(result["rows"][0]["value"], 41)
            self.assertIn("postgres down", result["postgres_error"])
            external_tools.ensure_sqlite_reference_tables(
                db_sqlite_path=str(sqlite_path),
                donors_table_name="orchestrator_donors",
                hospitals_table_name="orchestrator_hospitals",
                donor_csv_path=donor_csv,
                hospital_csv_path=hospital_csv,
                required_tables={"donors", "hospitals"},
            )
            with sqlite3.connect(str(sqlite_path)) as conn:
                donor_count = conn.execute(
                    'SELECT COUNT(*) FROM "orchestrator_donors"'
                ).fetchone()[0]
                hospital_count = conn.execute(
                    'SELECT COUNT(*) FROM "orchestrator_hospitals"'
                ).fetchone()[0]
            self.assertEqual(donor_count, 1)
            self.assertEqual(hospital_count, 1)

    def test_stockout_nl_query_routes_to_hybrid_tool(self):
        runtime = self._hospital_shortage_runtime()
        registry = _DummyRegistry()
        registry._models[runtime.model_id] = runtime
        expected_output = {
            "blood_component": "RBC",
            "blood_group": "O-",
            "location": "Hospital A",
            "risk_1_7_days": 0.78,
            "estimated_days_until_stockout": 5,
            "stockout_probability": 0.78,
            "horizon": "1-7 days",
            "risk_level": "critical",
            "recommended_action": "Urgent donor campaign or transfer request",
            "confidence": "medium",
            "hybrid_prediction": True,
        }
        env = {
            "PIOS_XLAM_DISABLE_LLM": "1",
            "PIOS_ORCH_ENABLE_DB_TOOL": "0",
            "PIOS_ORCH_ENABLE_SEARCH_TOOL": "0",
            "PIOS_ORCH_ENABLE_SIMULATION_TOOL": "0",
        }

        with self._orchestrator_with_tools_enabled(
            registry,
            stockout_hybrid=True,
            extra_env=env,
        ) as orchestrator:
            with patch(
                "ml.orchestrator.service.build_hybrid_stockout_prediction",
                return_value=expected_output,
            ) as mock_build:
                result = orchestrator.run(
                    "Predict stockout risk for O- RBC at Hospital A",
                    provided_features={
                        "blood_component": "RBC",
                        "blood_group": "O-",
                        "location": "Hospital A",
                    },
                )

        self.assertTrue(result["success"])
        self.assertEqual(
            result["execution_results"][0]["tool"],
            orchestrator.stockout_hybrid_tool_name,
        )
        self.assertEqual(result["stockout_prediction"]["stockout_probability"], 0.78)
        self.assertEqual(result["stockout_prediction"]["risk_level"], "critical")
        self.assertIn("estimated days until stockout: 5", result["verification_data"][0])
        mock_build.assert_called_once()

    def test_stockout_hybrid_runtime_prefers_short_hospital_model_without_horizon(self):
        long_term = _StubModelRuntime(ACTIVE_STOCKOUT_SENTINEL_ID)
        long_term.description = "Hospital stockout sentinel."
        long_term.defaults = _stockout_hybrid_defaults()
        short_term = _StubModelRuntime(ACTIVE_STOCKOUT_MODEL_ID)
        short_term.description = "Primary component inventory risk simulator."
        short_term.defaults = _stockout_hybrid_defaults()
        registry = _DummyRegistry()
        registry._models[long_term.model_id] = long_term
        registry._models[short_term.model_id] = short_term

        with patch.dict(os.environ, {"PIOS_XLAM_DISABLE_LLM": "1"}, clear=False):
            orchestrator = DynamicXLAMOrchestrator(registry=registry)
            selected = orchestrator._select_stockout_hybrid_runtime(
                "will there be a stockout in Hospital H001?",
                {"hospital_id": "H001"},
                [long_term, short_term],
            )

        self.assertEqual(selected.model_id, ACTIVE_STOCKOUT_SENTINEL_ID)

    def test_stockout_query_is_not_treated_as_structured_db_query(self):
        query = (
            "Predict days until stockout for "
            "blood_product_type=O-,current_stock_units=3,usage_today=2"
        )

        self.assertFalse(external_tools.is_structured_db_query(query))

    def test_hospital_list_query_is_structured_db_query(self):
        self.assertTrue(external_tools.is_structured_db_query("show me our hospitals"))

    def test_current_stock_query_is_structured_db_query(self):
        self.assertTrue(external_tools.is_structured_db_query("current stock"))
        request = external_tools._infer_reference_list_request("show me hospital names")
        self.assertIsNotNone(request)
        self.assertEqual(request[0], "hospitals")
        self.assertEqual(request[1], ["hospital_id"])

    def test_stock_status_query_uses_configured_current_stock_lookup(self):
        self.assertTrue(external_tools.is_structured_db_query("What is the stock status?"))
        request = external_tools._infer_reference_list_request("What is the stock status?")
        self.assertIsNotNone(request)
        table, fields, latest_only = request
        self.assertEqual(table, "hospitals")
        self.assertTrue(latest_only)
        self.assertIn("stock_end", fields)
        self.assertIn("critical_stock", fields)

    def test_execute_db_tool_lists_hospitals_with_sqlite_fallback(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            donor_csv = tmp_path / "donors.csv"
            hospital_csv = tmp_path / "hospitals.csv"
            sqlite_path = tmp_path / "fallback.sqlite3"
            donor_csv.write_text(
                "\n".join(
                    [
                        "donor_id,age,sex,country_code,region,blood_type,eligible_to_donate",
                        "1,41,F,DZ,Algiers,O-,true",
                    ]
                ),
                encoding="utf-8",
            )
            hospital_csv.write_text(
                "\n".join(
                    [
                        "date,hospital_id,blood_type,stock_start,units_collected,units_used,wastage,stock_end,critical_stock",
                        "2026-04-20,H002,O-,10,5,3,0,12,false",
                        "2026-04-21,H001,A+,8,4,2,0,10,false",
                        "2026-04-22,H001,O-,7,3,2,0,8,false",
                    ]
                ),
                encoding="utf-8",
            )

            external_tools._SQLITE_BOOTSTRAP_CACHE.clear()
            with patch.object(
                external_tools,
                "_postgres_connect",
                side_effect=RuntimeError("postgres down"),
            ):
                result = external_tools.execute_db_tool(
                    query="show me our hospitals",
                    db_schema="public",
                    donors_table_name="orchestrator_donors",
                    hospitals_table_name="orchestrator_hospitals",
                    db_supply_csv_path=hospital_csv,
                    db_sqlite_path=str(sqlite_path),
                    db_sqlite_table="blood_supply",
                    db_series_limit=24,
                    csv_candidates=[donor_csv],
                )

        self.assertEqual(result["table"], "hospitals")
        self.assertEqual(result["aggregate"], "list")
        self.assertEqual(result["rows"], [{"hospital_id": "H001"}, {"hospital_id": "H002"}])

    def test_execute_db_tool_lists_current_hospital_stock_columns(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            donor_csv = tmp_path / "donors.csv"
            hospital_csv = tmp_path / "hospitals.csv"
            sqlite_path = tmp_path / "fallback.sqlite3"
            donor_csv.write_text(
                "\n".join(
                    [
                        "donor_id,age,sex,country_code,region,blood_type,eligible_to_donate",
                        "1,41,F,DZ,Algiers,O-,true",
                    ]
                ),
                encoding="utf-8",
            )
            hospital_csv.write_text(
                "\n".join(
                    [
                        "date,hospital_id,blood_type,stock_start,units_collected,units_used,wastage,stock_end,critical_stock",
                        "2026-04-20,H001,O-,10,5,3,0,12,false",
                        "2026-04-22,H001,O-,7,3,2,0,8,false",
                        "2026-04-22,H002,A+,4,2,5,0,1,true",
                    ]
                ),
                encoding="utf-8",
            )

            external_tools._SQLITE_BOOTSTRAP_CACHE.clear()
            with patch.object(
                external_tools,
                "_postgres_connect",
                side_effect=RuntimeError("postgres down"),
            ):
                result = external_tools.execute_db_tool(
                    query="show our current hospitals stocks",
                    db_schema="public",
                    donors_table_name="orchestrator_donors",
                    hospitals_table_name="orchestrator_hospitals",
                    db_supply_csv_path=hospital_csv,
                    db_sqlite_path=str(sqlite_path),
                    db_sqlite_table="blood_supply",
                    db_series_limit=24,
                    csv_candidates=[donor_csv],
                )

        self.assertEqual(result["table"], "hospitals")
        self.assertEqual(result["aggregate"], "list")
        self.assertEqual(
            result["fields"],
            ["date", "hospital_id", "blood_type", "stock_start", "stock_end", "critical_stock"],
        )
        self.assertTrue(result["latest_only"])
        self.assertEqual(len(result["rows"]), 2)
        self.assertEqual({row["date"] for row in result["rows"]}, {"2026-04-22"})
        self.assertEqual(result["rows"][0]["stock_end"], 8)
        self.assertIn("critical_stock", result["rows"][1])

    def test_execute_db_tool_answers_stock_status_with_latest_rows(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            donor_csv = tmp_path / "donors.csv"
            hospital_csv = tmp_path / "hospitals.csv"
            sqlite_path = tmp_path / "fallback.sqlite3"
            donor_csv.write_text(
                "\n".join(
                    [
                        "donor_id,age,sex,country_code,region,blood_type,eligible_to_donate",
                        "1,41,F,DZ,Algiers,O-,true",
                    ]
                ),
                encoding="utf-8",
            )
            hospital_csv.write_text(
                "\n".join(
                    [
                        "date,hospital_id,blood_type,stock_start,units_collected,units_used,wastage,stock_end,critical_stock",
                        "2026-04-20,H001,O-,10,5,3,0,12,false",
                        "2026-04-22,H001,O-,7,3,2,0,8,false",
                        "2026-04-22,H002,A+,4,2,5,0,1,true",
                    ]
                ),
                encoding="utf-8",
            )

            external_tools._SQLITE_BOOTSTRAP_CACHE.clear()
            with patch.object(
                external_tools,
                "_postgres_connect",
                side_effect=RuntimeError("postgres down"),
            ):
                result = external_tools.execute_db_tool(
                    query="What is the stock status?",
                    db_schema="public",
                    donors_table_name="orchestrator_donors",
                    hospitals_table_name="orchestrator_hospitals",
                    db_supply_csv_path=hospital_csv,
                    db_sqlite_path=str(sqlite_path),
                    db_sqlite_table="blood_supply",
                    db_series_limit=24,
                    csv_candidates=[donor_csv],
                )

        self.assertEqual(result["table"], "hospitals")
        self.assertEqual(result["aggregate"], "list")
        self.assertTrue(result["latest_only"])
        self.assertEqual({row["date"] for row in result["rows"]}, {"2026-04-22"})
        self.assertEqual(
            {(row["hospital_id"], row["blood_type"], row["stock_end"]) for row in result["rows"]},
            {("H001", "O-", 8), ("H002", "A+", 1)},
        )
        self.assertTrue(any(row["critical_stock"] for row in result["rows"]))

    def test_execute_db_tool_uses_configured_seed_aliases_and_derived_fields(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            donor_csv = tmp_path / "donors.csv"
            hospital_csv = tmp_path / "hospitals.csv"
            sqlite_path = tmp_path / "fallback.sqlite3"
            donor_csv.write_text(
                "\n".join(
                    [
                        "donor_id,blood_group,frequency_365,days_until_eligible",
                        "1,O-,4,0",
                        "2,O-,2,5",
                        "3,A+,3,0",
                    ]
                ),
                encoding="utf-8",
            )
            hospital_csv.write_text(
                "\n".join(
                    [
                        "date,hospital_id,blood_type,stock_start,units_collected,units_used,wastage,stock_end,critical_stock",
                        "2026-04-22,H001,O-,7,3,2,0,8,false",
                    ]
                ),
                encoding="utf-8",
            )

            external_tools._SQLITE_BOOTSTRAP_CACHE.clear()
            stale_csv = tmp_path / "stale_donors.csv"
            stale_csv.write_text("donor_id\n99", encoding="utf-8")
            with patch.object(
                external_tools,
                "_postgres_connect",
                side_effect=RuntimeError("postgres down"),
            ):
                external_tools.execute_db_tool(
                    query="list donors",
                    table="donors",
                    aggregate="list",
                    fields=["donor_id"],
                    db_schema="public",
                    donors_table_name="orchestrator_donors",
                    hospitals_table_name="orchestrator_hospitals",
                    db_supply_csv_path=hospital_csv,
                    db_sqlite_path=str(sqlite_path),
                    db_sqlite_table="blood_supply",
                    db_series_limit=24,
                    csv_candidates=[stale_csv],
                )

            external_tools._SQLITE_BOOTSTRAP_CACHE.clear()
            with patch.object(
                external_tools,
                "_postgres_connect",
                side_effect=RuntimeError("postgres down"),
            ):
                result = external_tools.execute_db_tool(
                    query="which O- donors should we call",
                    table="donors",
                    aggregate="list",
                    fields=[
                        "donor_id",
                        "blood_type",
                        "eligible_to_donate",
                        "eligibility_status",
                        "donation_count_last_12m",
                    ],
                    filters={"blood_type": "O-", "eligible_to_donate": True},
                    db_schema="public",
                    donors_table_name="orchestrator_donors",
                    hospitals_table_name="orchestrator_hospitals",
                    db_supply_csv_path=hospital_csv,
                    db_sqlite_path=str(sqlite_path),
                    db_sqlite_table="blood_supply",
                    db_series_limit=24,
                    csv_candidates=[donor_csv],
                )

        self.assertEqual(result["rows"], [
            {
                "donor_id": 1,
                "blood_type": "O-",
                "country_code": None,
                "eligible_to_donate": 1,
                "eligibility_status": "eligible",
                "donation_count_last_12m": 4,
            }
        ])

    def test_execute_db_tool_normalizes_configured_donor_filters(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            donor_csv = tmp_path / "donors.csv"
            hospital_csv = tmp_path / "hospitals.csv"
            sqlite_path = tmp_path / "fallback.sqlite3"
            donor_csv.write_text(
                "\n".join(
                    [
                        "donor_id,blood_group,frequency_365,days_until_eligible",
                        "1,O-,4,0",
                        "2,A+,3,0",
                    ]
                ),
                encoding="utf-8",
            )
            hospital_csv.write_text(
                "\n".join(
                    [
                        "date,hospital_id,blood_type,stock_start,units_collected,units_used,wastage,stock_end,critical_stock",
                        "2026-04-22,H001,O-,7,3,2,0,8,false",
                    ]
                ),
                encoding="utf-8",
            )

            external_tools._SQLITE_BOOTSTRAP_CACHE.clear()
            with patch.object(
                external_tools,
                "_postgres_connect",
                side_effect=RuntimeError("postgres down"),
            ):
                result = external_tools.execute_db_tool(
                    query="check the blood supply then call the best donors",
                    table="donors",
                    aggregate="list",
                    fields=["donor_id", "blood_type", "eligibility_status"],
                    filters={
                        "blood_type": "PLATELETS O-",
                        "blood_component": "PLATELETS",
                        "eligibility_status": True,
                    },
                    db_schema="public",
                    donors_table_name="orchestrator_donors",
                    hospitals_table_name="orchestrator_hospitals",
                    db_supply_csv_path=hospital_csv,
                    db_sqlite_path=str(sqlite_path),
                    db_sqlite_table="blood_supply",
                    db_series_limit=24,
                    csv_candidates=[donor_csv],
                )

        self.assertEqual(result["filters"], {"blood_type": "O-", "eligibility_status": "eligible"})
        self.assertEqual(result["rows"][0]["donor_id"], 1)
        self.assertEqual(result["rows"][0]["blood_type"], "O-")

    def test_execute_db_tool_keeps_identity_columns_for_explicit_stock_field(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            donor_csv = tmp_path / "donors.csv"
            hospital_csv = tmp_path / "hospitals.csv"
            sqlite_path = tmp_path / "fallback.sqlite3"
            donor_csv.write_text(
                "\n".join(
                    [
                        "donor_id,age,sex,country_code,region,blood_type,eligible_to_donate",
                        "1,41,F,DZ,Algiers,O-,true",
                    ]
                ),
                encoding="utf-8",
            )
            hospital_csv.write_text(
                "\n".join(
                    [
                        "date,hospital_id,blood_type,stock_start,units_collected,units_used,wastage,stock_end,critical_stock",
                        "2026-04-20,H001,O-,10,5,3,0,12,false",
                        "2026-04-22,H001,O-,7,3,2,0,8,false",
                        "2026-04-22,H002,A+,4,2,5,0,1,true",
                    ]
                ),
                encoding="utf-8",
            )

            external_tools._SQLITE_BOOTSTRAP_CACHE.clear()
            with patch.object(
                external_tools,
                "_postgres_connect",
                side_effect=RuntimeError("postgres down"),
            ):
                result = external_tools.execute_db_tool(
                    query="show each hospitals current stocks",
                    table="hospitals",
                    aggregate="list",
                    fields=["stock_end"],
                    latest_only=True,
                    db_schema="public",
                    donors_table_name="orchestrator_donors",
                    hospitals_table_name="orchestrator_hospitals",
                    db_supply_csv_path=hospital_csv,
                    db_sqlite_path=str(sqlite_path),
                    db_sqlite_table="blood_supply",
                    db_series_limit=24,
                    csv_candidates=[donor_csv],
                )

        self.assertEqual(result["fields"], ["date", "hospital_id", "blood_type", "stock_end"])
        self.assertEqual(
            result["rows"],
            [
                {"date": "2026-04-22", "hospital_id": "H001", "blood_type": "O-", "stock_end": 8},
                {"date": "2026-04-22", "hospital_id": "H002", "blood_type": "A+", "stock_end": 1},
            ],
        )

    def test_reference_db_schema_excludes_future_target_columns(self):
        donor_columns = external_tools._TABLE_CONFIGS["donors"]["columns"]
        hospital_columns = external_tools._TABLE_CONFIGS["hospitals"]["columns"]

        self.assertNotIn("donated_next_6m", donor_columns)
        self.assertNotIn("next_6m_donation_count", donor_columns)
        self.assertNotIn("stockout_next_day", hospital_columns)
        self.assertNotIn("heavy_demand_next_day", hospital_columns)

    def test_normalize_db_limit_caps_results_to_ten(self):
        self.assertEqual(external_tools._normalize_db_limit(999), 10)
        self.assertEqual(external_tools._normalize_db_limit(0), 1)
        self.assertEqual(external_tools._normalize_db_limit(None), 10)

    def test_oldest_donor_request_repairs_birthdate_to_age(self):
        field, aggregate, sort, limit = external_tools._normalize_structured_db_request(
            query="show the oldest donor in the database?",
            logical_table="donors",
            field="birthdate",
            aggregate="max",
            sort="asc",
            limit=None,
            group_by=None,
            config=external_tools._TABLE_CONFIGS["donors"],
        )
        self.assertEqual(field, "age")
        self.assertEqual(aggregate, "max")
        self.assertEqual(sort, "DESC")
        self.assertEqual(limit, 1)

    def test_db_tool_metadata_is_loaded_from_config_file(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            config_path = Path(tmp_dir) / "config.json"
            config_path.write_text(
                """
                {
                  "external_tools": [
                    {
                      "id": "db_tool",
                      "name": "db_tool",
                      "type": "external",
                      "enabled": true,
                      "adapter": "db_query",
                      "entrypoint": "ml.core.external_tools.execute_db_tool",
                      "permissions": ["read_reference_data"],
                      "timeout_seconds": 30,
                      "description": "Configured DB tool.",
                      "input_hints": "table, field, aggregate",
                      "input_schema": {"type": "object"},
                      "output_schema": {"type": "object"},
                      "schema": {
                        "donors": {
                          "fields": ["donor_id", "age"],
                          "notes": "use age not birthdate"
                        }
                      },
                      "examples": [
                        {
                          "kind": "good",
                          "user_query": "show the oldest donor",
                          "arguments": {"table": "donors", "field": "age", "aggregate": "max", "limit": 1},
                          "why": "bounded top-1 aggregate lookup"
                        },
                        {
                          "kind": "bad",
                          "user_query": "show every donor row",
                          "why": "full table dumps are not allowed"
                        }
                      ]
                    }
                  ]
                }
                """,
                encoding="utf-8",
            )
            env = {
                "PIOS_XLAM_DISABLE_LLM": "1",
                "PIOS_ORCH_ENABLE_DB_TOOL": "1",
                "PIOS_ORCH_ENABLE_SEARCH_TOOL": "0",
                "PIOS_ORCH_EXTERNAL_TOOLS_CONFIG": str(config_path),
            }
            with patch.dict(os.environ, env, clear=False):
                orchestrator = DynamicXLAMOrchestrator(registry=_DummyRegistry())
                description = orchestrator._build_tools_description([])

        self.assertIn("Configured DB tool.", description)
        self.assertIn("Schema: donors(donor_id, age) note: use age not birthdate", description)
        self.assertIn("Good example: show the oldest donor", description)
        self.assertIn('"aggregate": "max"', description)
        self.assertIn("Good example why: bounded top-1 aggregate lookup", description)
        self.assertIn("Bad example: show every donor row", description)
        self.assertIn("Bad example why: full table dumps are not allowed", description)

    def test_invalid_external_tool_config_fails_clearly(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            config_path = Path(tmp_dir) / "config.json"
            config_path.write_text(
                '{"external_tools": [{"name": "db_tool"}]}',
                encoding="utf-8",
            )
            env = {
                "PIOS_XLAM_DISABLE_LLM": "1",
                "PIOS_ORCH_EXTERNAL_TOOLS_CONFIG": str(config_path),
            }
            with patch.dict(os.environ, env, clear=False):
                with self.assertRaisesRegex(RuntimeError, "missing required fields"):
                    DynamicXLAMOrchestrator(registry=_DummyRegistry())

    def test_invalid_batch_compare_config_fails_clearly(self):
        payload = self._minimal_external_tool_config()
        payload["external_tools"][0]["batch_compare"] = []
        with tempfile.TemporaryDirectory() as tmp_dir:
            config_path = Path(tmp_dir) / "config.json"
            config_path.write_text(json.dumps(payload), encoding="utf-8")
            env = {
                "PIOS_XLAM_DISABLE_LLM": "1",
                "PIOS_ORCH_EXTERNAL_TOOLS_CONFIG": str(config_path),
            }
            with patch.dict(os.environ, env, clear=False):
                with self.assertRaisesRegex(
                    RuntimeError,
                    "batch_compare must be an object",
                ):
                    DynamicXLAMOrchestrator(registry=_DummyRegistry())

    def test_configured_workflow_examples_are_budgeted_into_plan_prompt(self):
        orchestrator_config = {
            "fallback_order": ["db_tool"],
            "planning_limits": {
                "example_limit": 0,
                "workflow_example_limit": 1,
            },
            "dynamic_planning": {
                "workflow_examples": [
                    {
                        "id": "stockout_flow",
                        "user_query": "best donors to call for a stockout",
                        "trigger_terms": ["stockout", "donors"],
                        "tool_calls": [
                            {
                                "name": "stockout_hybrid",
                                "arguments": {"query": "$request.query"},
                                "reasoning": "Find the constrained blood group.",
                            },
                            {
                                "name": "db_tool",
                                "arguments": {
                                    "table": "donors",
                                    "aggregate": "list",
                                    "filters": {
                                        "blood_type": "$steps.1.output.worst_blood_group"
                                    },
                                    "limit": 10,
                                },
                                "reasoning": "Fetch bounded donor candidates.",
                            },
                        ],
                    },
                    {
                        "id": "unrelated_flow",
                        "user_query": "general hospital inventory lookup",
                        "trigger_terms": ["inventory"],
                        "tool_calls": [
                            {
                                "name": "db_tool",
                                "arguments": {"table": "hospitals", "limit": 10},
                            }
                        ],
                    },
                ]
            },
            "prompts": {
                "planning": (
                    "Available tools:\n{tools_description}{workflow_examples}\n"
                    "Query: {query}"
                )
            },
        }

        with tempfile.TemporaryDirectory() as tmp_dir:
            config_path = Path(tmp_dir) / "config.json"
            config_path.write_text(
                json.dumps(self._minimal_external_tool_config(orchestrator_config)),
                encoding="utf-8",
            )
            registry = _DummyRegistry()
            stockout = self._hospital_shortage_runtime()
            registry._models[stockout.model_id] = stockout
            env = {
                "PIOS_XLAM_DISABLE_LLM": "1",
                "PIOS_ORCH_ENABLE_DB_TOOL": "1",
                "PIOS_ORCH_ENABLE_SEARCH_TOOL": "0",
                "PIOS_ORCH_EXTERNAL_TOOLS_CONFIG": str(config_path),
            }
            with patch.dict(os.environ, env, clear=False):
                orchestrator = DynamicXLAMOrchestrator(registry=registry)
                prompt = orchestrator._build_plan_prompt(
                    "best donors to call during stockout",
                    [],
                    1,
                )

        self.assertIn("Workflow examples from config", prompt)
        self.assertIn("stockout_flow", prompt)
        self.assertNotIn("unrelated_flow", prompt)
        self.assertEqual(prompt.count('"tool_calls"'), 1)

    def test_fallback_plan_uses_configured_workflow_example_for_hospital_ranking(self):
        env = {
            "PIOS_XLAM_DISABLE_LLM": "1",
            "PIOS_ORCH_ENABLE_DB_TOOL": "1",
            "PIOS_ORCH_ENABLE_SEARCH_TOOL": "0",
        }
        payload = json.loads(EXTERNAL_TOOLS_CONFIG_PATH.read_text(encoding="utf-8"))
        for tool in payload["external_tools"]:
            if tool.get("id") == "stockout_hybrid":
                tool["enabled"] = True
        payload["orchestrator"]["dynamic_planning"]["workflow_examples"].append(
            {
                "user_query": "Which hospitals are at highest stockout risk?",
                "trigger_terms": ["highest stockout", "stockout risk", "which hospitals"],
                "required_any_terms": ["stockout", "shortage"],
                "required_terms": ["which hospitals"],
                "steps": [
                    {
                        "tool": "db_tool",
                        "arguments": {
                            "table": "hospitals",
                            "aggregate": "list",
                            "fields": ["hospital_id"],
                            "limit": 10,
                        },
                    },
                    {
                        "tool": "stockout_hybrid",
                        "arguments": {
                            "compare_mode": "rank",
                            "hospital_rows": "$steps.1.output.rows",
                            "entity_key": "hospital_id",
                            "group_rows": True,
                        },
                    },
                ],
            }
        )
        registry = _DummyRegistry()
        stockout = self._hospital_shortage_runtime()
        registry._models[stockout.model_id] = stockout
        with tempfile.TemporaryDirectory() as tmp_dir:
            config_path = Path(tmp_dir) / "config.json"
            config_path.write_text(json.dumps(payload), encoding="utf-8")
            env["PIOS_ORCH_EXTERNAL_TOOLS_CONFIG"] = str(config_path)
            with patch.dict(os.environ, env, clear=False):
                orchestrator = DynamicXLAMOrchestrator(registry=registry)
                plan = orchestrator._fallback_plan(
                    "which hospitals are at highest stockout risk",
                    [],
                    1,
                )

        self.assertEqual([step.name for step in plan.steps[:2]], ["db_tool", "stockout_hybrid"])
        self.assertEqual(plan.steps[0].arguments["fields"], ["hospital_id"])
        self.assertEqual(plan.steps[1].arguments["compare_mode"], "rank")
        self.assertEqual(plan.steps[1].arguments["hospital_rows"], "$steps.1.output.rows")
        self.assertEqual(plan.steps[1].arguments["entity_key"], "hospital_id")
        self.assertIs(plan.steps[1].arguments["group_rows"], True)

    def test_configured_direct_response_answers_without_tools(self):
        orchestrator_config = {
            "dynamic_planning": {
                "direct_response": {
                    "enabled": True,
                    "rules": [
                        {
                            "id": "donor_categories",
                            "trigger_terms": ["types"],
                            "required_term_groups": [["donor"], ["types"]],
                            "response_markdown": "Configured donor type answer.",
                        }
                    ],
                }
            }
        }
        with tempfile.TemporaryDirectory() as tmp_dir:
            config_path = Path(tmp_dir) / "config.json"
            config_path.write_text(
                json.dumps(self._minimal_external_tool_config(orchestrator_config)),
                encoding="utf-8",
            )
            env = {
                "PIOS_XLAM_DISABLE_LLM": "1",
                "PIOS_ORCH_EXTERNAL_TOOLS_CONFIG": str(config_path),
            }
            with patch.dict(os.environ, env, clear=False):
                orchestrator = DynamicXLAMOrchestrator(registry=_DummyRegistry())
                result = orchestrator.run("list donor types")

        self.assertTrue(result["success"])
        self.assertEqual(result["planner_mode"], "direct_response")
        self.assertEqual(result["execution_results"], [])
        self.assertEqual(
            result["natural_language_response"],
            "Configured donor type answer.",
        )

    def test_configured_tool_aliases_are_loaded_from_registry_config(self):
        payload = self._minimal_external_tool_config()
        payload["external_tools"][1]["aliases"] = ["db_search"]
        with tempfile.TemporaryDirectory() as tmp_dir:
            config_path = Path(tmp_dir) / "config.json"
            config_path.write_text(json.dumps(payload), encoding="utf-8")
            env = {
                "PIOS_XLAM_DISABLE_LLM": "1",
                "PIOS_ORCH_EXTERNAL_TOOLS_CONFIG": str(config_path),
            }
            with patch.dict(os.environ, env, clear=False):
                orchestrator = DynamicXLAMOrchestrator(registry=_DummyRegistry())

        self.assertEqual(orchestrator._tool_definition("db_search").id, "db_tool")
        tools = orchestrator.list_configured_tools()
        self.assertIn("db_search", tools[1]["aliases"])

    def test_row_scoring_rewrites_model_only_plan_to_db_batch_model_plan(self):
        orchestrator_config = {
            "dynamic_planning": {
                "row_scoring": {
                    "trigger_terms": ["rank"],
                    "source_output_keys": ["rows"],
                    "model_tasks": ["donor_individual"],
                    "entity_keys": ["donor_id"],
                    "control_argument_key": "batch",
                    "max_rows": 5,
                    "row_source": {
                        "tool_adapters": ["db_query"],
                        "argument_defaults": {
                            "table": "donors",
                            "aggregate": "list",
                            "fields": ["donor_id"],
                            "limit": 5,
                        },
                    },
                }
            },
            "model_execution": {"control_argument_keys": ["batch"]},
        }
        with tempfile.TemporaryDirectory() as tmp_dir:
            config_path = Path(tmp_dir) / "config.json"
            config_path.write_text(
                json.dumps(self._minimal_external_tool_config(orchestrator_config)),
                encoding="utf-8",
            )
            ideal = _StubModelRuntime("ideal_donor_classifier")
            ideal.defaults = {
                "prediction_source": "feast_online",
                "feast": {"entity_keys": ["donor_id"]},
            }
            registry = _DummyRegistry()
            registry._models[ideal.model_id] = ideal
            env = {
                "PIOS_XLAM_DISABLE_LLM": "1",
                "PIOS_ORCH_EXTERNAL_TOOLS_CONFIG": str(config_path),
            }
            with patch.dict(os.environ, env, clear=False):
                orchestrator = DynamicXLAMOrchestrator(registry=registry)
                plan = ExecutionPlan(
                    steps=[
                        ToolCall(
                            name="ideal_donor_classifier",
                            arguments={},
                            reasoning="model-only plan",
                        )
                    ],
                    reasoning="rank request",
                    is_multi_step=False,
                )
                repaired = orchestrator._repair_plan_for_row_scoring(
                    "rank ideal donors",
                    plan,
                    [ideal],
                )

        self.assertEqual([step.name for step in repaired.steps], ["db_tool", "ideal_donor_classifier"])
        self.assertEqual(repaired.steps[0].arguments["table"], "donors")
        self.assertEqual(
            repaired.steps[1].arguments["batch"]["rows"],
            "$steps.1.output.rows",
        )
        self.assertEqual(repaired.steps[1].arguments["batch"]["entity_key"], "donor_id")

    def test_simple_general_question_falls_back_to_llm_not_search(self):
        env = {
            "PIOS_XLAM_DISABLE_LLM": "1",
            "PIOS_ORCH_ENABLE_DB_TOOL": "0",
            "PIOS_ORCH_ENABLE_SEARCH_TOOL": "1",
            "PIOS_ORCH_LLM_API_URL": "http://llm.example.test",
        }
        with patch.dict(os.environ, env, clear=False):
            orchestrator = DynamicXLAMOrchestrator(registry=_DummyRegistry())

        plan = orchestrator._fallback_plan("whats 1-1", [], 1)

        self.assertEqual(len(plan.steps), 1)
        self.assertEqual(plan.steps[0].name, "llm_api")
        self.assertEqual(plan.steps[0].arguments, {"query": "whats 1-1"})
        self.assertNotEqual(plan.steps[0].name, orchestrator.search_tool_name)

    def test_stockout_hybrid_compare_batches_explicit_hospital_ids(self):
        env = {
            "PIOS_XLAM_DISABLE_LLM": "1",
            "PIOS_ORCH_ENABLE_DB_TOOL": "1",
            "PIOS_ORCH_ENABLE_SEARCH_TOOL": "0",
        }
        runtime = self._hospital_shortage_runtime()
        registry = _DummyRegistry()
        registry._models[runtime.model_id] = runtime

        def fake_build_hybrid_stockout_prediction(**kwargs):
            hospital_id = kwargs["request_features"].get("hospital_id")
            if hospital_id == "H001":
                return {
                    "hybrid_prediction": True,
                    "stockout_probability": 0.91,
                    "estimated_days_until_stockout": 0,
                    "risk_level": "critical",
                    "recommended_action": "urgent",
                    "worst_blood_group": "O+",
                    "worst_component": "PLATELETS",
                    "regression_model_id": runtime.model_id,
                }
            if hospital_id == "H002":
                return {
                    "hybrid_prediction": True,
                    "stockout_probability": 0.42,
                    "estimated_days_until_stockout": 4,
                    "risk_level": "moderate",
                    "recommended_action": "monitor",
                    "worst_blood_group": "A+",
                    "worst_component": "RBC",
                    "regression_model_id": runtime.model_id,
                }
            return {}

        with patch.dict(os.environ, env, clear=False):
            orchestrator = DynamicXLAMOrchestrator(registry=registry)
            with patch(
                "ml.orchestrator.service.build_hybrid_stockout_prediction",
                side_effect=fake_build_hybrid_stockout_prediction,
            ) as mock_build:
                result = orchestrator._execute_stockout_hybrid_tool(
                    "compare stockout between H001 and H002",
                    {
                        "hospital_ids": ["H001", "H002"],
                        "regression_model_id": runtime.model_id,
                    },
                )

        self.assertEqual(result["query_kind"], "batch_model_ranking")
        self.assertEqual(result["comparison_mode"], "compare")
        self.assertEqual([row["hospital_id"] for row in result["rows"]], ["H001", "H002"])
        self.assertGreater(result["rows"][0]["stockout_probability"], result["rows"][1]["stockout_probability"])
        self.assertEqual(mock_build.call_count, 2)

    def test_stockout_hybrid_ranks_bounded_hospital_rows(self):
        env = {
            "PIOS_XLAM_DISABLE_LLM": "1",
            "PIOS_ORCH_ENABLE_DB_TOOL": "1",
            "PIOS_ORCH_ENABLE_SEARCH_TOOL": "0",
        }
        runtime = self._hospital_shortage_runtime()
        registry = _DummyRegistry()
        registry._models[runtime.model_id] = runtime

        def fake_build_hybrid_stockout_prediction(**kwargs):
            hospital_id = kwargs["request_features"].get("hospital_id")
            if hospital_id == "H001":
                return {
                    "hybrid_prediction": True,
                    "stockout_probability": 0.83,
                    "estimated_days_until_stockout": 1,
                    "risk_level": "high",
                    "recommended_action": "urgent",
                    "worst_blood_group": "O+",
                    "worst_component": "PLATELETS",
                    "regression_model_id": runtime.model_id,
                }
            if hospital_id == "H002":
                return {
                    "hybrid_prediction": True,
                    "stockout_probability": 0.19,
                    "estimated_days_until_stockout": 7,
                    "risk_level": "low",
                    "recommended_action": "monitor",
                    "worst_blood_group": "A+",
                    "worst_component": "RBC",
                    "regression_model_id": runtime.model_id,
                }
            return {}

        with patch.dict(os.environ, env, clear=False):
            orchestrator = DynamicXLAMOrchestrator(registry=registry)
            with patch(
                "ml.orchestrator.service.build_hybrid_stockout_prediction",
                side_effect=fake_build_hybrid_stockout_prediction,
            ):
                result = orchestrator._execute_stockout_hybrid_tool(
                    "which hospitals are at highest stockout risk",
                    {
                        "hospital_rows": [
                            {"hospital_id": "H002"},
                            {"hospital_id": "H001"},
                        ],
                        "compare_mode": "rank",
                        "max_rows": 2,
                        "regression_model_id": runtime.model_id,
                    },
                )

        self.assertEqual(result["query_kind"], "batch_model_ranking")
        self.assertEqual(result["comparison_mode"], "rank")
        self.assertEqual([row["hospital_id"] for row in result["rows"]], ["H001", "H002"])
        self.assertEqual(result["rows"][0]["rank"], 1)

    def test_stockout_hybrid_groups_duplicate_hospital_rows_from_config(self):
        env = {
            "PIOS_XLAM_DISABLE_LLM": "1",
            "PIOS_ORCH_ENABLE_DB_TOOL": "1",
            "PIOS_ORCH_ENABLE_SEARCH_TOOL": "0",
        }
        runtime = self._hospital_shortage_runtime()
        registry = _DummyRegistry()
        registry._models[runtime.model_id] = runtime
        seen_features = []

        def fake_build_hybrid_stockout_prediction(**kwargs):
            row = dict(kwargs["request_features"])
            seen_features.append(row)
            hospital_id = row.get("hospital_id")
            probability = 0.8 if hospital_id == "H001" else 0.3
            return {
                "hybrid_prediction": True,
                "stockout_probability": probability,
                "estimated_days_until_stockout": 1 if hospital_id == "H001" else 5,
                "risk_level": "high" if hospital_id == "H001" else "low",
                "recommended_action": "monitor",
                "worst_blood_group": row.get("blood_type"),
                "worst_component": "RBC",
                "regression_model_id": runtime.model_id,
            }

        with self._orchestrator_with_tools_enabled(
            registry,
            stockout_hybrid=True,
            extra_env=env,
        ) as orchestrator:
            with patch(
                "ml.orchestrator.service.build_hybrid_stockout_prediction",
                side_effect=fake_build_hybrid_stockout_prediction,
            ) as mock_build:
                result = orchestrator._execute_stockout_hybrid_tool(
                    "which hospital will run out of stock faster",
                    {
                        "hospital_rows": [
                            {
                                "hospital_id": "H001",
                                "blood_type": "A+",
                                "stock_end": 8,
                                "critical_stock": False,
                            },
                            {
                                "hospital_id": "H001",
                                "blood_type": "O-",
                                "stock_end": 0,
                                "critical_stock": True,
                            },
                            {
                                "hospital_id": "H002",
                                "blood_type": "B+",
                                "stock_end": 4,
                                "critical_stock": False,
                            },
                        ],
                        "compare_mode": "rank",
                        "max_rows": 10,
                        "regression_model_id": runtime.model_id,
                    },
                )

        self.assertEqual(mock_build.call_count, 2)
        self.assertEqual(result["source_row_count"], 3)
        self.assertEqual(result["row_count"], 2)
        self.assertEqual([row["hospital_id"] for row in result["rows"]], ["H001", "H002"])
        h001_features = next(row for row in seen_features if row["hospital_id"] == "H001")
        self.assertEqual(h001_features["blood_type"], "O-")
        self.assertEqual(h001_features["stock_end"], 0)

    def test_stockout_hybrid_compare_rejects_unbounded_requests(self):
        env = {
            "PIOS_XLAM_DISABLE_LLM": "1",
            "PIOS_ORCH_ENABLE_DB_TOOL": "1",
            "PIOS_ORCH_ENABLE_SEARCH_TOOL": "0",
        }
        runtime = self._hospital_shortage_runtime()
        registry = _DummyRegistry()
        registry._models[runtime.model_id] = runtime

        with patch.dict(os.environ, env, clear=False):
            orchestrator = DynamicXLAMOrchestrator(registry=registry)
            with self.assertRaisesRegex(RuntimeError, "at most 2 grouped row"):
                orchestrator._execute_stockout_hybrid_tool(
                    "compare stockout between H001 H002 H003",
                    {
                        "hospital_ids": ["H001", "H002", "H003"],
                        "max_rows": 2,
                        "regression_model_id": runtime.model_id,
                    },
                )

    def test_llm_planner_chains_stockout_db_and_batch_model_scoring_without_templates(self):
        orchestrator_config = {
            "fallback_order": ["db_tool"],
            "dynamic_planning": {
                "prefer_planner_terms": ["then", "call", "best"],
                "row_scoring": {
                    "trigger_terms": ["best", "call"],
                    "source_output_keys": ["rows"],
                    "model_tasks": ["donor_individual"],
                    "entity_keys": ["donor_id"],
                    "control_argument_key": "batch",
                    "include_fields": ["donor_id", "blood_type"],
                    "score_keys": ["probability", "prediction"],
                    "sort": "desc",
                    "row_source": {
                        "tool_adapters": ["db_query"],
                        "argument_defaults": {
                            "table": "donors",
                            "aggregate": "list",
                            "fields": ["donor_id", "blood_type"],
                            "limit": 10,
                        },
                        "filter_defaults": {"eligible_to_donate": True},
                        "filter_remove": ["eligibility_status"],
                        "filter_dependencies": [
                            {
                                "field": "blood_type",
                                "source_tool_adapters": ["stockout_hybrid"],
                                "output_paths": ["worst_blood_group"],
                            }
                        ],
                    },
                },
            },
            "model_execution": {"control_argument_keys": ["batch"]},
            "planning_limits": {
                "example_limit": 2,
                "schema_field_limit": 8,
                "completion_tokens": 128,
                "context_reserve_tokens": 64,
            },
            "prompts": {
                "planning": (
                    "Available tools:\n{tools_description}\nQuery: {query}\n"
                    "Return JSON with tool_calls."
                ),
                "summarizer": "RESULTS DATA:\n{data_block}\nAnswer:",
            },
        }

        class _FakePlannerLLM:
            def __init__(self):
                self.prompts = []
                self.n_ctx = 2048

            def __call__(self, prompt, **kwargs):
                self.prompts.append(prompt)
                max_tokens = int(kwargs.get("max_tokens") or 0)
                if len(self.tokenize(prompt.encode("utf-8"))) + max_tokens > self.n_ctx:
                    raise RuntimeError("prompt exceeded context")
                if "RESULTS DATA" in prompt:
                    return {"choices": [{"text": "ranked donor candidates"}]}
                return {
                    "choices": [
                        {
                            "text": json.dumps(
                                {
                                    "reasoning": "Planner inferred dependent donor outreach steps from metadata.",
                                    "tool_calls": [
                                        {
                                            "name": "stockout_hybrid",
                                            "arguments": {},
                                            "reasoning": "Find highest-risk blood group.",
                                        },
                                        {
                                            "name": "db_tool",
                                            "arguments": {
                                                "query": "$request.query",
                                                "table": "donors",
                                                "aggregate": "list",
                                                "fields": ["donor_id", "blood_type"],
                                                "filters": {
                                                    "eligibility_status": True,
                                                },
                                                "limit": 10,
                                            },
                                            "reasoning": "Fetch bounded eligible candidates.",
                                        },
                                    ],
                                }
                            )
                        }
                    ]
                }

            def tokenize(self, data):
                return list(range(max(1, len(data) // 3)))

        with tempfile.TemporaryDirectory() as tmp_dir:
            config_path = Path(tmp_dir) / "config.json"
            config_path.write_text(
                json.dumps(self._minimal_external_tool_config(orchestrator_config)),
                encoding="utf-8",
            )

            stockout = self._hospital_shortage_runtime()
            ideal = _StubModelRuntime(ACTIVE_DONOR_MODEL_ID)
            ideal.description = "Ideal donor classifier."
            ideal.feature_names = ["donor_id", "blood_type"]
            ideal.defaults = {
                "prediction_source": "feast_online",
                "feast": {"entity_keys": ["donor_id"]},
            }
            registry = _DummyRegistry()
            registry._models[stockout.model_id] = stockout
            registry._models[ideal.model_id] = ideal

            env = {
                "PIOS_XLAM_DISABLE_LLM": "0",
                "PIOS_ORCH_ENABLE_DB_TOOL": "1",
                "PIOS_ORCH_ENABLE_SEARCH_TOOL": "0",
                "PIOS_ORCH_ENABLE_SIMULATION_TOOL": "0",
                "PIOS_ORCH_EXTERNAL_TOOLS_CONFIG": str(config_path),
            }
            with patch.dict(os.environ, env, clear=False):
                orchestrator = DynamicXLAMOrchestrator(registry=registry)
                fake_llm = _FakePlannerLLM()
                orchestrator._llm = fake_llm
                orchestrator._llm_attempted = True
                orchestrator._llm_n_ctx = fake_llm.n_ctx
                with patch(
                    "ml.orchestrator.service.build_hybrid_stockout_prediction",
                    return_value={
                        "hybrid_prediction": True,
                        "worst_blood_group": "O+",
                        "worst_component": "PLATELETS",
                        "stockout_probability": 0.78,
                        "estimated_days_until_stockout": 1,
                    },
                ):
                    with patch.object(
                        orchestrator,
                        "_execute_db_fallback",
                        return_value={
                            "tool_type": "db_tool",
                            "query_kind": "donor_candidates",
                            "rows": [
                                {"donor_id": "D002", "blood_type": "O+"},
                                {"donor_id": "D001", "blood_type": "O+"},
                            ],
                            "answer": "candidate rows",
                        },
                    ) as mock_db:
                        with patch(
                            "ml.orchestrator.service.resolve_prediction_features",
                            side_effect=lambda runtime, features, **kwargs: types.SimpleNamespace(
                                features=features,
                                metadata={"entity_row": {"donor_id": features.get("donor_id")}},
                            ),
                        ):
                            with patch(
                                "ml.orchestrator.service.run_prediction_batch_with_fallback",
                                side_effect=lambda runtime, rows, **_kwargs: [
                                    {
                                        "probability": 0.91
                                        if features.get("donor_id") == "D001"
                                        else 0.62
                                    }
                                    for features in rows
                                ],
                            ):
                                result = orchestrator.run(
                                    "check the blood supply then call the best donors"
                                )

        self.assertTrue(result["success"])
        self.assertEqual(result["planner_mode"], "llm")
        self.assertEqual(
            [row["tool"] for row in result["execution_results"]],
            [
                orchestrator.stockout_hybrid_tool_name,
                orchestrator.db_tool_name,
            ],
        )
        self.assertEqual(
            mock_db.call_args.args[1]["filters"],
            {"blood_type": "O+", "eligible_to_donate": True},
        )
        self.assertEqual(result["execution_results"][1]["output"]["query_kind"], "donor_candidates")
        self.assertIn("Available tools", fake_llm.prompts[0])

    def _many_external_tools_config(self, planning_limits=None):
        """Config mirroring production: stockout_hybrid + db_tool + several
        verbose digital-twin tools, so the planning prompt is large enough to
        force the budgeter to reduce the NUMBER of tool blocks to fit n_ctx."""
        long_desc = (
            "This configured planner tool exposes a digital-twin workflow that "
            "simulates inventory, donor, and supply dynamics across many hospitals "
            "and components for scenario analysis and what-if policy comparison "
            "over configurable branches, horizons, and promoted actions."
        )
        external = [
            {
                "id": "stockout_hybrid",
                "name": "stockout_hybrid",
                "type": "external",
                "enabled": True,
                "adapter": "stockout_hybrid",
                "entrypoint": "ml.core.hybrid_stockout.build_hybrid_stockout_prediction",
                "permissions": ["run_prediction"],
                "timeout_seconds": 30,
                "description": (
                    "Predicts blood stockout risk and days until stockout for a "
                    "hospital, blood type, and component from current inventory and usage."
                ),
                "input_hints": "hospital, blood_type, component_type, current_inventory, units_used",
                "input_schema": {"type": "object"},
                "output_schema": {"type": "object"},
            },
            {
                "id": "db_tool",
                "name": "db_tool",
                "type": "external",
                "enabled": True,
                "adapter": "db_query",
                "entrypoint": "ml.core.external_tools.execute_db_tool",
                "permissions": ["read_reference_data"],
                "timeout_seconds": 30,
                "description": "Reads bounded reference rows from stored hospital and donor tables.",
                "input_hints": "table, field, aggregate",
                "input_schema": {"type": "object"},
                "output_schema": {"type": "object", "properties": {"rows": {"type": "array"}}},
            },
        ]
        for i in range(1, 7):
            external.append(
                {
                    "id": f"digital_twin_tool_{i}",
                    "name": f"digital_twin_tool_{i}",
                    "type": "external",
                    "enabled": True,
                    "adapter": "digital_twin",
                    "entrypoint": "ml.core.external_tools.noop",
                    "permissions": [],
                    "timeout_seconds": 30,
                    "description": long_desc,
                    "input_hints": "policy, horizon, scenario, branch, action, comparison, recommendation",
                    "input_schema": {"type": "object"},
                    "output_schema": {"type": "object"},
                }
            )
        return {
            "external_tools": external,
            "orchestrator": {
                "fallback_order": ["stockout_hybrid", "db_tool"],
                "planning_limits": planning_limits
                or {
                    "example_limit": 2,
                    "schema_field_limit": 8,
                    "workflow_example_limit": 0,
                    "completion_tokens": 64,
                    "context_reserve_tokens": 32,
                },
                "prompts": {
                    "planning": (
                        "Available tools:\n{tools_description}{workflow_examples}\n"
                        "Query: {query}\nReturn JSON with tool_calls."
                    ),
                    "summarizer": "RESULTS DATA:\n{data_block}\nAnswer:",
                },
                "dynamic_planning": {
                    "prefer_planner_terms": ["forecast", "analyze", "plan"],
                    "workflow_examples": [
                        {
                            "id": "stockout_flow",
                            "user_query": "will there be a stockout",
                            "trigger_terms": ["stockout", "run out", "shortage"],
                            "required_any_terms": ["stockout", "run out", "shortage"],
                            "tool_calls": [
                                {
                                    "name": "stockout_hybrid",
                                    "arguments": {"query": "$request.query"},
                                    "reasoning": "Estimate stockout risk.",
                                }
                            ],
                        }
                    ],
                },
            },
        }

    def test_budgeter_reduces_tool_count_so_llm_planner_still_runs(self):
        """When per-item compaction is not enough, the budgeter drops the
        least-relevant tool blocks (keeping the query-relevant ones) so the LLM
        planner runs instead of crashing on an over-budget prompt."""
        config = self._many_external_tools_config()

        class _FakePlannerLLM:
            def __init__(self):
                self.prompts = []
                self.n_ctx = 800

            def __call__(self, prompt, **kwargs):
                self.prompts.append(prompt)
                max_tokens = int(kwargs.get("max_tokens") or 0)
                if len(self.tokenize(prompt.encode("utf-8"))) + max_tokens > self.n_ctx:
                    raise RuntimeError("prompt exceeded context")
                return {
                    "choices": [
                        {
                            "text": json.dumps(
                                {
                                    "reasoning": "Stockout risk for the named hospital.",
                                    "tool_calls": [
                                        {
                                            "name": "stockout_hybrid",
                                            "arguments": {},
                                            "reasoning": "Estimate stockout risk.",
                                        }
                                    ],
                                }
                            )
                        }
                    ]
                }

            def tokenize(self, data):
                return list(range(max(1, len(data) // 3)))

        with tempfile.TemporaryDirectory() as tmp_dir:
            config_path = Path(tmp_dir) / "config.json"
            config_path.write_text(json.dumps(config), encoding="utf-8")

            registry = _DummyRegistry()
            stockout = self._hospital_shortage_runtime()
            registry._models[stockout.model_id] = stockout

            env = {
                "PIOS_XLAM_DISABLE_LLM": "0",
                "PIOS_ORCH_ENABLE_DB_TOOL": "1",
                "PIOS_ORCH_ENABLE_SIMULATION_TOOL": "0",
                "PIOS_ORCH_EXTERNAL_TOOLS_CONFIG": str(config_path),
            }
            with patch.dict(os.environ, env, clear=False):
                orchestrator = DynamicXLAMOrchestrator(registry=registry)
                fake_llm = _FakePlannerLLM()
                orchestrator._llm = fake_llm
                orchestrator._llm_attempted = True
                orchestrator._llm_n_ctx = fake_llm.n_ctx
                # Sanity: the un-trimmed most-compact prompt really is over budget,
                # so this test exercises the new count-reduction stage.
                all_tools = orchestrator._planner_external_tools()
                self.assertGreaterEqual(len(all_tools), 6)
                with patch(
                    "ml.orchestrator.service.build_hybrid_stockout_prediction",
                    return_value={
                        "hybrid_prediction": True,
                        "worst_blood_group": "O-",
                        "stockout_probability": 0.81,
                        "estimated_days_until_stockout": 1,
                    },
                ):
                    result = orchestrator.run(
                        "forecast whether there will be a stockout in hospital central"
                    )

        self.assertTrue(result["success"])
        self.assertEqual(result["planner_mode"], "llm")
        sent = fake_llm.prompts[0]
        # The query-relevant tool survived truncation...
        self.assertIn("stockout_hybrid", sent)
        # ...but at least one least-relevant digital-twin tool was dropped.
        twin_present = sum(f"digital_twin_tool_{i}" in sent for i in range(1, 7))
        self.assertLess(twin_present, 6)

    def test_budgeter_falls_back_cleanly_when_prompt_cannot_fit(self):
        """If even a single-tool prompt cannot fit n_ctx, the planner routes to a
        clean fallback WITHOUT sending the prompt to the LLM, and never surfaces a
        raw 'exceed context window' / 'Requested tokens' error."""
        config = self._many_external_tools_config()

        class _PlannerForbiddenLLM:
            def __init__(self):
                self.prompts = []
                self.n_ctx = 64

            def __call__(self, prompt, **kwargs):
                self.prompts.append(prompt)
                if "Available tools" in prompt:
                    raise AssertionError("planner prompt must not reach the LLM")
                return {"choices": [{"text": "fallback summary"}]}

            def tokenize(self, data):
                return list(range(max(1, len(data) // 3)))

        with tempfile.TemporaryDirectory() as tmp_dir:
            config_path = Path(tmp_dir) / "config.json"
            config_path.write_text(json.dumps(config), encoding="utf-8")

            registry = _DummyRegistry()
            stockout = self._hospital_shortage_runtime()
            registry._models[stockout.model_id] = stockout

            env = {
                "PIOS_XLAM_DISABLE_LLM": "0",
                "PIOS_ORCH_ENABLE_DB_TOOL": "1",
                "PIOS_ORCH_ENABLE_SIMULATION_TOOL": "0",
                "PIOS_ORCH_EXTERNAL_TOOLS_CONFIG": str(config_path),
            }
            with patch.dict(os.environ, env, clear=False):
                orchestrator = DynamicXLAMOrchestrator(registry=registry)
                fake_llm = _PlannerForbiddenLLM()
                orchestrator._llm = fake_llm
                orchestrator._llm_attempted = True
                orchestrator._llm_n_ctx = fake_llm.n_ctx
                with patch(
                    "ml.orchestrator.service.build_hybrid_stockout_prediction",
                    return_value={
                        "hybrid_prediction": True,
                        "worst_blood_group": "O-",
                        "stockout_probability": 0.81,
                        "estimated_days_until_stockout": 1,
                    },
                ):
                    result = orchestrator.run(
                        "forecast whether there will be a stockout in hospital central"
                    )

        # Planner LLM was never invoked with a planning prompt.
        self.assertTrue(all("Available tools" not in p for p in fake_llm.prompts))
        self.assertIn(result["planner_mode"], {"fallback", "external_fallback"})
        reasoning = result["plan"]["reasoning"]
        self.assertNotIn("context window", reasoning)
        self.assertNotIn("Requested tokens", reasoning)
        self.assertIn("planning prompt exceeded model context", reasoning)

    def test_ranked_planner_external_tools_prioritises_query_match(self):
        config = self._many_external_tools_config()
        with tempfile.TemporaryDirectory() as tmp_dir:
            config_path = Path(tmp_dir) / "config.json"
            config_path.write_text(json.dumps(config), encoding="utf-8")

            registry = _DummyRegistry()
            stockout = self._hospital_shortage_runtime()
            registry._models[stockout.model_id] = stockout

            env = {
                "PIOS_XLAM_DISABLE_LLM": "1",
                "PIOS_ORCH_ENABLE_DB_TOOL": "1",
                "PIOS_ORCH_ENABLE_SIMULATION_TOOL": "0",
                "PIOS_ORCH_EXTERNAL_TOOLS_CONFIG": str(config_path),
            }
            with patch.dict(os.environ, env, clear=False):
                orchestrator = DynamicXLAMOrchestrator(registry=registry)
                ranked = orchestrator._ranked_planner_external_tools(
                    "will there be a stockout in hospital central"
                )
                self.assertEqual(ranked[0]["name"], "stockout_hybrid")
                # Empty query preserves config order.
                base = [t["name"] for t in orchestrator._planner_external_tools()]
                self.assertEqual(
                    [t["name"] for t in orchestrator._ranked_planner_external_tools("")],
                    base,
                )

    def test_auto_planner_n_ctx_scales_with_model_footprint(self):
        env = {"PIOS_XLAM_DISABLE_LLM": "1"}
        with patch.dict(os.environ, env, clear=False):
            os.environ.pop("PIOS_XLAM_RESERVED_MB", None)
            orchestrator = DynamicXLAMOrchestrator(registry=_DummyRegistry())
            total_8gb = 8192.0
            # ~4GB quant on 8GB stays at 2048 (4096 exhausts memory and the GPU
            # backend fails to decode).
            self.assertEqual(orchestrator._auto_planner_n_ctx(total_8gb, 4027), 2048)
            # ~2.6GB quant on the SAME 8GB host now reaches the full 4096.
            self.assertEqual(orchestrator._auto_planner_n_ctx(total_8gb, 2592), 4096)
            # The large quant gets 4096 once there is enough RAM (16GB).
            self.assertEqual(orchestrator._auto_planner_n_ctx(16384.0, 4027), 4096)
            # Unknown sizes fall back to the conservative window, never crash.
            self.assertEqual(orchestrator._auto_planner_n_ctx(0.0, None), 2048)
            self.assertEqual(orchestrator._auto_planner_n_ctx(total_8gb, None), 2048)
            # A model that barely fits gets the smallest window, not a failure.
            self.assertEqual(orchestrator._auto_planner_n_ctx(total_8gb, 8000), 1024)
            # Explicit reserve override is honoured.
            with patch.dict(os.environ, {"PIOS_XLAM_RESERVED_MB": "6000"}, clear=False):
                self.assertEqual(
                    orchestrator._auto_planner_n_ctx(total_8gb, 2592), 1024
                )

    def test_stockout_call_query_adds_configured_row_source_before_scoring(self):
        orchestrator_config = {
            "fallback_order": ["db_tool"],
            "dynamic_planning": {
                "prefer_planner_terms": ["call"],
                "row_scoring": {
                    "trigger_terms": ["call"],
                    "source_output_keys": ["rows"],
                    "model_tasks": ["donor_individual"],
                    "entity_keys": ["donor_id"],
                    "control_argument_key": "batch",
                    "include_fields": ["donor_id", "blood_type"],
                    "score_keys": ["probability", "prediction"],
                    "sort": "desc",
                    "row_source": {
                        "tool_adapters": ["db_query"],
                        "reasoning": "Configured test row source.",
                        "argument_defaults": {
                            "table": "donors",
                            "aggregate": "list",
                            "fields": ["donor_id", "blood_type"],
                            "limit": 10,
                        },
                        "filter_defaults": {"eligible_to_donate": True},
                        "filter_dependencies": [
                            {
                                "field": "blood_type",
                                "source_tool_adapters": ["stockout_hybrid"],
                                "output_paths": ["worst_blood_group"],
                            }
                        ],
                    },
                },
            },
            "model_execution": {"control_argument_keys": ["batch"]},
            "prompts": {"summarizer": "RESULTS DATA:\n{data_block}\nAnswer:"},
        }

        with tempfile.TemporaryDirectory() as tmp_dir:
            config_path = Path(tmp_dir) / "config.json"
            config_path.write_text(
                json.dumps(self._minimal_external_tool_config(orchestrator_config)),
                encoding="utf-8",
            )

            stockout = self._hospital_shortage_runtime()
            donor = _StubModelRuntime(ACTIVE_DONOR_MODEL_ID)
            donor.description = "Ideal donor classifier."
            donor.feature_names = ["donor_id", "blood_type"]
            donor.defaults = {
                "prediction_source": "feast_online",
                "feast": {"entity_keys": ["donor_id"]},
            }
            registry = _DummyRegistry()
            registry._models[stockout.model_id] = stockout
            registry._models[donor.model_id] = donor

            env = {
                "PIOS_XLAM_DISABLE_LLM": "1",
                "PIOS_ORCH_ENABLE_DB_TOOL": "1",
                "PIOS_ORCH_ENABLE_SEARCH_TOOL": "0",
                "PIOS_ORCH_ENABLE_SIMULATION_TOOL": "0",
                "PIOS_ORCH_EXTERNAL_TOOLS_CONFIG": str(config_path),
            }
            with patch.dict(os.environ, env, clear=False):
                orchestrator = DynamicXLAMOrchestrator(registry=registry)
                with patch(
                    "ml.orchestrator.service.build_hybrid_stockout_prediction",
                    return_value={
                        "hybrid_prediction": True,
                        "worst_blood_group": "O+",
                        "stockout_probability": 0.82,
                        "estimated_days_until_stockout": 0,
                    },
                ):
                    with patch.object(
                        orchestrator,
                        "_execute_db_fallback",
                        return_value={
                            "tool_type": "db_tool",
                            "query_kind": "donor_candidates",
                            "rows": [
                                {"donor_id": "D002", "blood_type": "O+"},
                                {"donor_id": "D001", "blood_type": "O+"},
                            ],
                            "answer": "candidate rows",
                        },
                    ) as mock_db:
                        with patch(
                            "ml.orchestrator.service.resolve_prediction_features",
                            side_effect=lambda runtime, features, **kwargs: types.SimpleNamespace(
                                features=features,
                                metadata={"entity_row": {"donor_id": features.get("donor_id")}},
                            ),
                        ):
                            with patch(
                                "ml.orchestrator.service.run_prediction_batch_with_fallback",
                                side_effect=lambda runtime, rows, **_kwargs: [
                                    {
                                        "probability": 0.93
                                        if features.get("donor_id") == "D001"
                                        else 0.61
                                    }
                                    for features in rows
                                ],
                            ):
                                result = orchestrator.run(
                                    "what donors should we call if we have a stockout in hospital h001"
                                )

        self.assertTrue(result["success"])
        self.assertEqual(result["planner_mode"], "function_tool")
        self.assertEqual(
            [row["tool"] for row in result["execution_results"]],
            [
                orchestrator.stockout_hybrid_tool_name,
                orchestrator.db_tool_name,
            ],
        )
        self.assertEqual(
            result["plan"]["steps"][1]["arguments"]["table"],
            "donors",
        )
        self.assertEqual(
            mock_db.call_args.args[1]["filters"],
            {"eligible_to_donate": True, "blood_type": "O+"},
        )
        self.assertEqual(result["execution_results"][1]["output"]["query_kind"], "donor_candidates")

    def test_step_argument_resolver_expands_embedded_references(self):
        env = {
            "PIOS_XLAM_DISABLE_LLM": "1",
            "PIOS_ORCH_ENABLE_DB_TOOL": "1",
            "PIOS_ORCH_ENABLE_SEARCH_TOOL": "0",
            "PIOS_ORCH_ENABLE_SIMULATION_TOOL": "0",
        }
        with patch.dict(os.environ, env, clear=False):
            orchestrator = DynamicXLAMOrchestrator(registry=_DummyRegistry())

        resolved = orchestrator._resolve_step_arguments(
            {
                "filters": {
                    "blood_type": "PLATELETS $steps.1.output.worst_blood_group"
                }
            },
            query="check blood supply then call donors",
            request_features={},
            results=[
                ExecutionResult(
                    1,
                    orchestrator.stockout_hybrid_tool_name,
                    {},
                    {"worst_blood_group": "O-"},
                    True,
                )
            ],
        )

        self.assertEqual(resolved["filters"]["blood_type"], "PLATELETS O-")

    def test_model_step_without_inputs_is_skipped_instead_of_loading_model(self):
        env = {
            "PIOS_XLAM_DISABLE_LLM": "0",
            "PIOS_ORCH_ENABLE_EXTERNAL_FALLBACK": "0",
        }
        runtime = _StubModelRuntime("ideal_donor_classifier")
        runtime.description = "Ideal donor classifier."
        runtime.feature_names = ["donor_id", "blood_type"]
        registry = _DummyRegistry()
        registry._models[runtime.model_id] = runtime
        plan = ExecutionPlan(
            steps=[
                ToolCall(
                    name=runtime.model_id,
                    arguments={},
                    reasoning="planner returned model without rows",
                )
            ],
            reasoning="test",
            is_multi_step=False,
        )

        with patch.dict(os.environ, env, clear=False):
            orchestrator = DynamicXLAMOrchestrator(registry=registry)
            with patch.object(orchestrator, "_plan_execution", return_value=(plan, "llm")):
                with patch(
                    "ml.orchestrator.service.run_prediction_with_fallback",
                    return_value={"probability": 0.9},
                ) as mock_predict:
                    result = orchestrator.run("call the best donors")

        self.assertFalse(result["success"])
        self.assertEqual(
            result["execution_results"][0]["error"],
            "Skipped: request inputs do not match this model's declared inputs.",
        )
        mock_predict.assert_not_called()

    def test_simulation_tool_description_mentions_dynamic_custom_scenarios(self):
        env = {
            "PIOS_XLAM_DISABLE_LLM": "1",
            "PIOS_ORCH_ENABLE_DB_TOOL": "0",
            "PIOS_ORCH_ENABLE_SEARCH_TOOL": "0",
            "PIOS_ORCH_ENABLE_SIMULATION_TOOL": "1",
        }
        with self._orchestrator_with_tools_enabled(
            _DummyRegistry(),
            simulation=True,
            extra_env=env,
        ) as orchestrator:
            description = orchestrator._build_tools_description([])

        self.assertIn("dynamic custom scenarios", description)
        self.assertIn("forecast/congestion", description)

    def test_stockout_models_are_hidden_behind_hybrid_tool(self):
        env = {
            "PIOS_XLAM_DISABLE_LLM": "1",
            "PIOS_ORCH_ENABLE_DB_TOOL": "0",
            "PIOS_ORCH_ENABLE_SEARCH_TOOL": "0",
            "PIOS_ORCH_ENABLE_SIMULATION_TOOL": "0",
        }
        stockout = _StubModelRuntime(ACTIVE_STOCKOUT_MODEL_ID)
        stockout.description = "Hospital stockout predictor."
        stockout.feature_names = ["hospital_id", "blood_type", "current_inventory"]
        stockout.defaults = _stockout_hybrid_defaults()
        donor = _StubModelRuntime(ACTIVE_DONOR_MODEL_ID)
        donor.description = "Individual donor propensity model."
        donor.feature_names = ["age", "bmi"]
        registry = _DummyRegistry()
        registry._models[stockout.model_id] = stockout
        registry._models[donor.model_id] = donor

        with self._orchestrator_with_tools_enabled(
            registry,
            stockout_hybrid=True,
            extra_env=env,
        ) as orchestrator:
            description = orchestrator._build_tools_description([stockout, donor])

        self.assertIn("Tool: stockout_hybrid", description)
        self.assertNotIn(f"Tool: {ACTIVE_STOCKOUT_MODEL_ID}", description)
        self.assertIn(f"Tool: {ACTIVE_DONOR_MODEL_ID}", description)

    def test_plan_prompt_distinguishes_prediction_from_db_retrieval(self):
        env = {
            "PIOS_XLAM_DISABLE_LLM": "1",
            "PIOS_ORCH_ENABLE_DB_TOOL": "1",
            "PIOS_ORCH_ENABLE_SEARCH_TOOL": "0",
        }
        runtime = _StubModelRuntime("days_until_stockout_reg")
        runtime.description = "Hospital inventory and demand forecasting model."
        runtime.feature_names = [
            "temp_c",
            "rain_mm",
            "holiday",
            "trauma_cases",
            "scheduled_surgeries",
            "stock_end",
        ]
        with patch.dict(os.environ, env, clear=False):
            orchestrator = DynamicXLAMOrchestrator(registry=_DummyRegistry())
            prompt = orchestrator._build_plan_prompt(
                "Given temperature 30 and 50 scheduled surgeries, predict stock_end",
                [runtime],
                1,
            )

        self.assertIn("Use predictive ML models when the user asks to predict", prompt)
        self.assertIn(
            "Do not substitute one tool category for another unless the listed tool metadata says it supports that task.",
            prompt,
        )

    def test_build_tools_description_includes_feast_input_contract(self):
        env = {
            "PIOS_XLAM_DISABLE_LLM": "1",
        }
        runtime = _StubModelRuntime(ACTIVE_DONOR_MODEL_ID)
        runtime.description = "Individual donor propensity model."
        runtime.feature_names = [
            "donor_features_features__lat",
            "donor_features_features__lon",
            "donor_features_features__availability",
            "donor_features_features__blood_group",
            "donor_features_features__recency_days",
            "donor_features_features__frequency_365",
            "donor_features_features__days_until_eligible",
        ]
        runtime.defaults = {
            "prediction_source": "feast_online",
            "feast": {"entity_keys": ["donor_id"]},
        }

        with patch.dict(os.environ, env, clear=False):
            orchestrator = DynamicXLAMOrchestrator(registry=_DummyRegistry())
            description = orchestrator._build_tools_description([runtime])

        self.assertIn(
            "Input contract: donor_id is an optional lookup key, not a model input",
            description,
        )

    def test_plan_prompt_forbids_padding_missing_model_features(self):
        env = {
            "PIOS_XLAM_DISABLE_LLM": "1",
            "PIOS_ORCH_ENABLE_DB_TOOL": "0",
            "PIOS_ORCH_ENABLE_SEARCH_TOOL": "0",
        }
        runtime = _StubModelRuntime("donor_propensity_model_champion")
        runtime.description = "Individual donor propensity model."
        runtime.feature_names = [
            "donor_features_features__lat",
            "donor_features_features__lon",
            "donor_features_features__availability",
            "donor_features_features__blood_group",
            "donor_features_features__recency_days",
            "donor_features_features__frequency_365",
            "donor_features_features__days_until_eligible",
        ]

        with patch.dict(os.environ, env, clear=False):
            orchestrator = DynamicXLAMOrchestrator(registry=_DummyRegistry())
            prompt = orchestrator._build_plan_prompt(
                "Can a 35 year old donor with BMI 24 donate blood now?",
                [runtime],
                1,
            )

        self.assertIn(
            "NEVER invent, pad, or fill missing model features with null/NaN.",
            prompt,
        )
        self.assertIn(
            "For feature-store-backed models, IDs are optional lookup hints only",
            prompt,
        )

    def test_model_execution_drops_null_padded_llm_arguments(self):
        env = {
            "PIOS_XLAM_DISABLE_LLM": "1",
            "PIOS_ORCH_ENABLE_DB_TOOL": "0",
            "PIOS_ORCH_ENABLE_SEARCH_TOOL": "0",
            "PIOS_ORCH_ENABLE_SIMULATION_TOOL": "0",
        }
        runtime = _StubModelRuntime("donor_propensity_model_champion")
        runtime.description = "Individual donor propensity model."
        runtime.feature_names = ["age", "sex", "bmi"]
        runtime.defaults = {
            "prediction_source": "feast_online",
            "feast": {
                "feature_service": "donor_propensity_service",
                "entity_keys": ["donor_id"],
                "allow_direct_input": True,
            },
        }
        registry = _DummyRegistry()
        registry._models[runtime.model_id] = runtime
        plan = ExecutionPlan(
            steps=[
                ToolCall(
                    name=runtime.model_id,
                    arguments={"age": 35, "sex": None, "bmi": 24, "donor_id": None},
                )
            ],
            reasoning="test",
            is_multi_step=False,
        )

        with patch.dict(os.environ, env, clear=False):
            orchestrator = DynamicXLAMOrchestrator(registry=registry)
            with patch.object(orchestrator, "_plan_execution", return_value=(plan, "llm")):
                with patch(
                    "ml.orchestrator.service.resolve_component_prediction_features",
                    return_value=[],
                ):
                    with patch(
                        "ml.orchestrator.service.resolve_prediction_features",
                        return_value=types.SimpleNamespace(
                            features={"age": 35, "bmi": 24},
                            metadata={"source": "request"},
                        ),
                    ) as mock_resolve:
                        with patch(
                            "ml.orchestrator.service.run_prediction_with_fallback",
                            return_value={"prediction": 0.7},
                        ):
                            result = orchestrator.run(
                                "Predict donor propensity for age 35 BMI 24",
                            )

        self.assertTrue(result["success"])
        _, request_inputs = mock_resolve.call_args.args
        self.assertEqual(request_inputs, {"age": 35, "bmi": 24})

    def test_complete_stockout_model_payload_reroutes_to_hybrid_tool(self):
        env = {
            "PIOS_XLAM_DISABLE_LLM": "1",
            "PIOS_ORCH_ENABLE_DB_TOOL": "1",
            "PIOS_ORCH_ENABLE_SEARCH_TOOL": "1",
        }
        runtime = self._hospital_shortage_runtime()
        registry = _DummyRegistry()
        registry._models[runtime.model_id] = runtime
        query, features = self._hospital_shortage_query_and_features()
        fake_plan = ExecutionPlan(
            steps=[
                ToolCall(
                    name=runtime.model_id,
                    arguments=features,
                    reasoning="model request with complete direct payload",
                )
            ],
            reasoning="forced test plan",
            is_multi_step=False,
        )

        with self._orchestrator_with_tools_enabled(
            registry,
            stockout_hybrid=True,
            extra_env=env,
        ) as orchestrator:
            orchestrator.disable_llm = True
            with patch.object(
                orchestrator,
                "_plan_execution",
                return_value=(fake_plan, "llm"),
            ):
                with patch(
                    "ml.orchestrator.service.resolve_prediction_features",
                ) as mock_resolve:
                    with patch(
                        "ml.orchestrator.service.run_prediction_with_fallback",
                    ) as mock_predict:
                        with patch(
                            "ml.orchestrator.service.build_hybrid_stockout_prediction",
                            return_value={
                                "regression_model_id": runtime.model_id,
                                "estimated_days_until_stockout": 2,
                                "stockout_probability": 0.81,
                                "horizon": "1-7 days",
                                "risk_level": "critical",
                                "recommended_action": "Urgent donor campaign or transfer request",
                                "hybrid_prediction": True,
                            },
                        ) as mock_hybrid:
                            with patch.object(
                                orchestrator, "_execute_search_tool"
                            ) as mock_search:
                                result = orchestrator.run(query)

        self.assertTrue(result["success"])
        self.assertEqual(result["planner_mode"], "llm")
        self.assertEqual(
            result["execution_results"][0]["tool"],
            orchestrator.stockout_hybrid_tool_name,
        )
        self.assertEqual(result["stockout_prediction"].get("regression_model_id"), runtime.model_id)
        self.assertEqual(
            mock_hybrid.call_args.kwargs["regression_runtime"].model_id,
            runtime.model_id,
        )
        mock_hybrid.assert_called_once()
        mock_resolve.assert_not_called()
        mock_predict.assert_not_called()
        mock_search.assert_not_called()

    def test_complete_model_payload_failure_does_not_fallback_to_search(self):
        env = {
            "PIOS_XLAM_DISABLE_LLM": "1",
            "PIOS_ORCH_ENABLE_DB_TOOL": "1",
            "PIOS_ORCH_ENABLE_SEARCH_TOOL": "1",
        }
        runtime = self._hospital_shortage_runtime()
        registry = _DummyRegistry()
        registry._models[runtime.model_id] = runtime
        query, features = self._hospital_shortage_query_and_features()
        fake_plan = ExecutionPlan(
            steps=[
                ToolCall(
                    name=runtime.model_id,
                    arguments=features,
                    reasoning="model request with complete direct payload",
                )
            ],
            reasoning="forced test plan",
            is_multi_step=False,
        )

        with self._orchestrator_with_tools_enabled(
            registry,
            stockout_hybrid=True,
            extra_env=env,
        ) as orchestrator:
            with patch.object(
                orchestrator,
                "_plan_execution",
                return_value=(fake_plan, "llm"),
            ):
                with patch(
                    "ml.orchestrator.service.build_hybrid_stockout_prediction",
                    side_effect=RuntimeError("hybrid unavailable"),
                ):
                    with patch.object(
                        orchestrator, "_execute_search_tool"
                    ) as mock_search:
                        result = orchestrator.run(query)

        self.assertFalse(result["success"])
        self.assertEqual(len(result["execution_results"]), 1)
        self.assertEqual(
            result["execution_results"][0]["tool"],
            orchestrator.stockout_hybrid_tool_name,
        )
        mock_search.assert_not_called()

    def test_invalid_plan_tool_triggers_llm_fallback_chain(self):
        env = {
            "PIOS_XLAM_DISABLE_LLM": "1",
            "PIOS_ORCH_ENABLE_DB_TOOL": "0",
            "PIOS_ORCH_ENABLE_SEARCH_TOOL": "0",
            "PIOS_ORCH_EXTERNAL_FALLBACK_ORDER": "llm_api",
        }
        with patch.dict(os.environ, env, clear=False):
            orchestrator = DynamicXLAMOrchestrator(registry=_DummyRegistry())
            fake_plan = ExecutionPlan(
                steps=[
                    ToolCall(
                        name="unknown_tool",
                        arguments={},
                        reasoning="forced invalid tool",
                    )
                ],
                reasoning="forced test plan",
                is_multi_step=False,
            )
            with patch.object(
                orchestrator,
                "_plan_execution",
                return_value=(fake_plan, "fallback"),
            ):
                with patch.object(
                    orchestrator,
                    "_execute_llm_api_fallback",
                    return_value={
                        "tool_type": "llm_api",
                        "answer": "fallback worked",
                    },
                ):
                    result = orchestrator.run("Any question that should fall back.")

        self.assertTrue(result["success"])
        tools = [row["tool"] for row in result["execution_results"]]
        self.assertIn("unknown_tool", tools)
        self.assertIn(orchestrator._available_tool_id_for_adapter("llm_api") or "llm_api", tools)
        self.assertEqual(result["planner_mode"], "external_fallback")

    def test_run_waits_for_background_llm_init_before_planning(self):
        env = {
            "PIOS_XLAM_DISABLE_LLM": "0",
            "PIOS_ORCH_ENABLE_DB_TOOL": "0",
            "PIOS_ORCH_ENABLE_SEARCH_TOOL": "0",
            "PIOS_ORCH_EXTERNAL_FALLBACK_ORDER": "llm_api",
        }
        with patch.dict(os.environ, env, clear=False):
            orchestrator = DynamicXLAMOrchestrator(registry=_DummyRegistry())
            orchestrator._llm = None
            orchestrator._llm_attempted = True
            orchestrator._init_in_progress = True
            fake_plan = ExecutionPlan(
                steps=[
                    ToolCall(
                        name="unknown_tool",
                        arguments={},
                        reasoning="forced invalid tool",
                    )
                ],
                reasoning="forced test plan",
                is_multi_step=False,
            )
            with patch.object(orchestrator, "_ensure_llm") as mock_ensure:
                with patch.object(
                    orchestrator,
                    "_plan_execution",
                    return_value=(fake_plan, "fallback"),
                ):
                    with patch.object(
                        orchestrator,
                        "_execute_llm_api_fallback",
                        return_value={
                            "tool_type": "llm_api",
                            "answer": "fallback worked",
                        },
                    ):
                        orchestrator.run("Any question that should fall back.")

        mock_ensure.assert_called_once_with(blocking=True)

    def test_structured_db_failure_does_not_fallback_to_search(self):
        env = {
            "PIOS_XLAM_DISABLE_LLM": "1",
            "PIOS_ORCH_ENABLE_DB_TOOL": "1",
            "PIOS_ORCH_ENABLE_SEARCH_TOOL": "1",
        }
        with patch.dict(os.environ, env, clear=False):
            orchestrator = DynamicXLAMOrchestrator(registry=_DummyRegistry())
            fake_plan = ExecutionPlan(
                steps=[
                    ToolCall(
                        name=orchestrator.db_tool_name,
                        arguments={
                            "table": "donors",
                            "field": "made_up_field",
                            "aggregate": "max",
                        },
                        reasoning="structured db only",
                    )
                ],
                reasoning="structured db failure",
                is_multi_step=False,
            )
            with patch.object(
                orchestrator,
                "_plan_execution",
                return_value=(fake_plan, "llm"),
            ):
                with patch.object(
                    orchestrator,
                    "_execute_db_fallback",
                    side_effect=RuntimeError("Unsupported field"),
                ):
                    with patch.object(orchestrator, "_execute_search_tool") as mock_search:
                        result = orchestrator.run("show the oldest donor in the database?")

        self.assertFalse(result["success"])
        mock_search.assert_not_called()
        self.assertEqual(len(result["execution_results"]), 1)
        self.assertEqual(result["execution_results"][0]["tool"], orchestrator.db_tool_name)

    def test_fallback_plan_routes_simulation_queries_to_simulation_tool(self):
        env = {
            "PIOS_XLAM_DISABLE_LLM": "1",
            "PIOS_ORCH_ENABLE_DB_TOOL": "0",
            "PIOS_ORCH_ENABLE_SEARCH_TOOL": "0",
            "PIOS_ORCH_ENABLE_SIMULATION_TOOL": "1",
        }
        with self._orchestrator_with_tools_enabled(
            _DummyRegistry(),
            simulation=True,
            extra_env=env,
        ) as orchestrator:
            plan = orchestrator._fallback_plan(
                "What would happen if blood donors dropped by 50% in the next week?",
                [],
                top_k=1,
            )

        self.assertEqual(len(plan.steps), 1)
        self.assertEqual(plan.steps[0].name, orchestrator.simulation_tool_name)
        self.assertEqual(
            plan.steps[0].arguments["query"],
            "What would happen if blood donors dropped by 50% in the next week?",
        )

    def test_fallback_plan_routes_current_stock_to_db_before_model_matches(self):
        env = {
            "PIOS_XLAM_DISABLE_LLM": "1",
            "PIOS_ORCH_ENABLE_DB_TOOL": "1",
            "PIOS_ORCH_ENABLE_SEARCH_TOOL": "1",
        }
        runtime = _StubModelRuntime("stockout_days_predictor_champion")
        runtime.description = "Hospital inventory stockout days model."
        runtime.feature_names = ["hospital_id", "blood_type", "current_inventory"]

        with patch.dict(os.environ, env, clear=False):
            orchestrator = DynamicXLAMOrchestrator(registry=_DummyRegistry())
            plan = orchestrator._fallback_plan(
                "current stock",
                [runtime],
                top_k=1,
            )

        self.assertEqual(len(plan.steps), 1)
        self.assertEqual(plan.steps[0].name, orchestrator.db_tool_name)

    def test_simulation_queries_skip_predictive_models_in_planning_context(self):
        env = {
            "PIOS_XLAM_DISABLE_LLM": "1",
            "PIOS_ORCH_ENABLE_SIMULATION_TOOL": "1",
        }
        runtime = _StubModelRuntime("hospital_shortage_predictor_champion")
        runtime.description = "Hospital inventory forecasting model."

        with self._orchestrator_with_tools_enabled(
            _DummyRegistry(),
            simulation=True,
            extra_env=env,
        ) as orchestrator:
            selected = orchestrator._select_planning_models(
                "What would happen if blood donors dropped by 50% in the next week?",
                [runtime],
                top_k=1,
            )

        self.assertEqual(selected, [])

    def test_empty_llm_plan_uses_external_fallback_without_claiming_unavailable(self):
        env = {
            "PIOS_XLAM_DISABLE_LLM": "0",
            "PIOS_ORCH_ENABLE_DB_TOOL": "0",
            "PIOS_ORCH_ENABLE_SEARCH_TOOL": "0",
            "PIOS_ORCH_ENABLE_SIMULATION_TOOL": "1",
        }

        class _FakeLLM:
            def __call__(self, prompt, **kwargs):
                return {"choices": [{"text": "{}"}]}

            def tokenize(self, data):
                return list(data)

        with self._orchestrator_with_tools_enabled(
            _DummyRegistry(),
            simulation=True,
            extra_env=env,
        ) as orchestrator:
            orchestrator._llm = _FakeLLM()
            orchestrator._llm_attempted = True
            orchestrator._llm_n_ctx = 4096
            with patch.object(
                orchestrator,
                "_execute_simulation_tool",
                return_value={
                    "tool_type": "simulation",
                    "answer": "simulation fallback worked",
                },
            ):
                result = orchestrator.run(
                    "What would happen if blood donors dropped by 50% in the next week?"
                )

        self.assertTrue(result["success"])
        self.assertEqual(result["planner_mode"], "function_tool")
        self.assertEqual(result["execution_results"][0]["tool"], orchestrator.simulation_tool_name)
        self.assertIn(
            "Routed to simulation tool",
            result["plan"]["reasoning"],
        )
        self.assertNotIn("orchestrator LLM unavailable", result["plan"]["reasoning"])
        self.assertTrue(result["orchestrator_status"]["llm_ready"])

    def test_empty_llm_plan_for_in_scope_query_falls_back_to_model_with_inline_features(
        self,
    ):
        env = {
            "PIOS_XLAM_DISABLE_LLM": "0",
            "PIOS_ORCH_ENABLE_DB_TOOL": "1",
            "PIOS_ORCH_ENABLE_SEARCH_TOOL": "0",
            "PIOS_ORCH_ENABLE_SIMULATION_TOOL": "0",
        }

        class _FakeLLM:
            def __call__(self, prompt, **kwargs):
                return {"choices": [{"text": "{}"}]}

            def tokenize(self, data):
                return list(data)

        runtime = _StubModelRuntime("stockout_days_predictor")
        runtime.description = "Hospital inventory and demand forecasting model."
        runtime.feature_names = [
            "blood_product_type",
            "current_stock_units",
            "usage_today",
            "lead_time_days",
            "days_since_last_restock",
            "stockout_count_90d",
            "scheduled_surgeries_next7d",
        ]
        # Configure horizon routing on the stub so the orchestrator's
        # forecast-horizon ranking has a deterministic score for this runtime,
        # independent of any on-disk artifact metrics file. Without this the
        # test would only pass on machines that happen to have
        # ml-backend/artifacts/models/<id>/metrics.json present.
        runtime.defaults = {
            "horizon_routing": {
                "enabled": True,
                "timeframes": ["short"],
                "min_days": 0,
                "max_days": 7,
                "priority": 100,
                "target": "stockout",
            }
        }

        registry = _DummyRegistry()
        registry._models[runtime.model_id] = runtime

        query = (
            "Predict days until stockout in the next 7 days for "
            "blood_product_type=O-,current_stock_units=3,usage_today=2,"
            "lead_time_days=5,days_since_last_restock=7,stockout_count_90d=2,"
            "scheduled_surgeries_next7d=3"
        )

        with tempfile.TemporaryDirectory() as tmp_dir:
            payload = json.loads(
                (BACKEND_ROOT / "ml" / "config" / "config.json").read_text(
                    encoding="utf-8"
                )
            )
            for tool in payload["external_tools"]:
                if tool["adapter"] == "stockout_hybrid":
                    tool["enabled"] = False
            config_path = Path(tmp_dir) / "config.json"
            config_path.write_text(
                json.dumps(payload),
                encoding="utf-8",
            )
            env["PIOS_ORCH_EXTERNAL_TOOLS_CONFIG"] = str(config_path)

            with patch.dict(os.environ, env, clear=False):
                orchestrator = DynamicXLAMOrchestrator(registry=registry)
                orchestrator._llm = _FakeLLM()
                orchestrator._llm_attempted = True
                orchestrator._llm_n_ctx = 32768
                with patch(
                    "ml.orchestrator.service.resolve_prediction_features",
                    return_value=types.SimpleNamespace(
                        features={
                            "blood_product_type": "O-",
                            "current_stock_units": 3,
                            "usage_today": 2,
                            "lead_time_days": 5,
                            "days_since_last_restock": 7,
                            "stockout_count_90d": 2,
                            "scheduled_surgeries_next7d": 3,
                        },
                        metadata={"source": "request"},
                    ),
                ) as mock_resolve:
                    with patch(
                        "ml.orchestrator.service.run_prediction_with_fallback",
                        return_value={"prediction": 2.5},
                    ) as mock_predict:
                        with patch(
                            "ml.orchestrator.service.build_hybrid_stockout_prediction",
                            return_value={
                                "regression_model_id": runtime.model_id,
                                "estimated_days_until_stockout": 2.5,
                                "stockout_probability": 0.72,
                                "horizon": "1-7 days",
                                "risk_level": "critical",
                                "recommended_action": "Urgent donor campaign or transfer request",
                                "hybrid_prediction": True,
                            },
                        ) as mock_hybrid:
                            result = orchestrator.run(query)

        self.assertTrue(result["success"])
        self.assertEqual(result["planner_mode"], "fallback")
        self.assertTrue(result["horizon_routing"]["applied"])
        self.assertEqual(
            result["execution_results"][0]["tool"],
            runtime.model_id,
        )
        self.assertEqual(result["plan"]["steps"][0]["tool"], runtime.model_id)
        self.assertEqual(result["execution_results"][0]["output"]["prediction"], 2.5)
        mock_resolve.assert_called_once()
        mock_predict.assert_called_once()
        mock_hybrid.assert_not_called()

    def test_fallback_donate_now_query_routes_to_donor_hazard_model(self):
        env = {
            "PIOS_XLAM_DISABLE_LLM": "1",
            "PIOS_ORCH_ENABLE_DB_TOOL": "0",
            "PIOS_ORCH_ENABLE_SEARCH_TOOL": "0",
            "PIOS_ORCH_ENABLE_SIMULATION_TOOL": "0",
        }
        donor = _StubModelRuntime(ACTIVE_DONOR_MODEL_ID)
        donor.description = "Donate-now donor eligibility and future donation likelihood model."
        donor.feature_names = ["age", "bmi", "recency_days"]
        ideal = _StubModelRuntime(ACTIVE_IDEAL_DONOR_MODEL_ID)
        ideal.description = "Ideal donor profile suitability model."
        ideal.feature_names = ["age", "bmi"]
        hospital = _StubModelRuntime("hospital_shortage_predictor_champion")
        hospital.description = "Hospital stockout shortage forecasting model."
        hospital.feature_names = ["hospital_id", "blood_type", "current_inventory"]
        stockout = _StubModelRuntime("stockout_days_predictor_champion")
        stockout.description = "Hospital inventory stockout days model."
        stockout.feature_names = ["hospital_id", "blood_type", "current_inventory"]
        prophet = _StubModelRuntime("hospital_long_term_shortage_prophet_champion")
        prophet.description = "Hospital long term shortage forecast model."
        prophet.feature_names = ["hospital_id", "blood_type", "current_inventory"]

        with patch.dict(os.environ, env, clear=False):
            orchestrator = DynamicXLAMOrchestrator(registry=_DummyRegistry())
            plan = orchestrator._fallback_plan(
                "Can a 35 year old donor with BMI 24 donate blood now?",
                [donor, ideal, hospital, stockout, prophet],
                top_k=5,
            )

        self.assertEqual(
            [step.name for step in plan.steps],
            [ACTIVE_DONOR_MODEL_ID],
        )
        self.assertEqual(plan.steps[0].arguments["age"], 35)
        self.assertEqual(plan.steps[0].arguments["bmi"], 24.0)

    def test_fallback_ideal_donor_query_routes_to_ideal_donor_model(self):
        env = {
            "PIOS_XLAM_DISABLE_LLM": "1",
            "PIOS_ORCH_ENABLE_DB_TOOL": "0",
            "PIOS_ORCH_ENABLE_SEARCH_TOOL": "0",
            "PIOS_ORCH_ENABLE_SIMULATION_TOOL": "0",
        }
        donor = _StubModelRuntime(ACTIVE_DONOR_MODEL_ID)
        donor.description = "Donate-now donor eligibility and future donation likelihood model."
        donor.feature_names = ["age", "bmi", "recency_days"]
        ideal = _StubModelRuntime(ACTIVE_IDEAL_DONOR_MODEL_ID)
        ideal.description = "Ideal donor profile suitability model."
        ideal.feature_names = ["age", "bmi"]

        with patch.dict(os.environ, env, clear=False):
            orchestrator = DynamicXLAMOrchestrator(registry=_DummyRegistry())
            plan = orchestrator._fallback_plan(
                "Is a 35 year old donor with BMI 24 an ideal donor?",
                [donor, ideal],
                top_k=2,
            )

        self.assertEqual(
            [step.name for step in plan.steps],
            [ACTIVE_IDEAL_DONOR_MODEL_ID],
        )
        self.assertEqual(plan.steps[0].arguments["age"], 35)
        self.assertEqual(plan.steps[0].arguments["bmi"], 24.0)

    def test_fallback_plan_uses_configured_model_step_limit(self):
        orchestrator_config = {
            "planning_limits": {
                "fallback_model_step_limit": 1,
            },
        }
        donor = _StubModelRuntime(ACTIVE_DONOR_MODEL_ID)
        donor.description = "Future donor donation likelihood model."
        donor.feature_names = ["age", "bmi"]
        ideal = _StubModelRuntime(ACTIVE_IDEAL_DONOR_MODEL_ID)
        ideal.description = "Ideal donor profile suitability model."
        ideal.feature_names = ["age", "bmi"]

        with tempfile.TemporaryDirectory() as tmp_dir:
            config_path = Path(tmp_dir) / "config.json"
            config_path.write_text(
                json.dumps(self._minimal_external_tool_config(orchestrator_config)),
                encoding="utf-8",
            )
            env = {
                "PIOS_XLAM_DISABLE_LLM": "1",
                "PIOS_ORCH_ENABLE_DB_TOOL": "0",
                "PIOS_ORCH_ENABLE_SEARCH_TOOL": "0",
                "PIOS_ORCH_ENABLE_SIMULATION_TOOL": "0",
                "PIOS_ORCH_EXTERNAL_TOOLS_CONFIG": str(config_path),
            }
            with patch.dict(os.environ, env, clear=False):
                orchestrator = DynamicXLAMOrchestrator(registry=_DummyRegistry())
                plan = orchestrator._fallback_plan(
                    "Predict donor score for age 35 and BMI 24",
                    [donor, ideal],
                    top_k=2,
                )

        self.assertEqual(len(plan.steps), 1)

    def test_execution_skips_model_when_request_inputs_do_not_match_declared_inputs(self):
        env = {
            "PIOS_XLAM_DISABLE_LLM": "1",
            "PIOS_ORCH_ENABLE_DB_TOOL": "0",
            "PIOS_ORCH_ENABLE_SEARCH_TOOL": "0",
            "PIOS_ORCH_ENABLE_SIMULATION_TOOL": "0",
        }
        donor = _StubModelRuntime("donor_propensity_model_champion")
        donor.description = "Individual donor propensity model."
        donor.feature_names = ["age", "bmi"]
        hospital = _StubModelRuntime("hospital_long_term_shortage_prophet_champion")
        hospital.description = "Component-level hospital inventory forecast model."
        hospital.feature_names = ["hospital_id", "blood_type", "current_inventory"]
        registry = _DummyRegistry()
        registry._models[donor.model_id] = donor
        registry._models[hospital.model_id] = hospital
        plan = ExecutionPlan(
            steps=[
                ToolCall(name=donor.model_id, arguments={"age": 35, "bmi": 24}),
                ToolCall(name=hospital.model_id, arguments={"age": 35, "bmi": 24}),
            ],
            reasoning="test",
            is_multi_step=True,
        )

        with patch.dict(os.environ, env, clear=False):
            orchestrator = DynamicXLAMOrchestrator(registry=registry)
            with patch.object(orchestrator, "_plan_execution", return_value=(plan, "llm")):
                with patch(
                    "ml.orchestrator.service.resolve_component_prediction_features",
                    return_value=[],
                ):
                    with patch(
                        "ml.orchestrator.service.resolve_prediction_features",
                        return_value=types.SimpleNamespace(
                            features={"age": 35, "bmi": 24},
                            metadata={"source": "request"},
                        ),
                    ) as mock_resolve:
                        with patch(
                            "ml.orchestrator.service.run_prediction_with_fallback",
                            return_value={"prediction": 0.8},
                        ) as mock_predict:
                            result = orchestrator.run(
                                "Can a 35 year old donor with BMI 24 donate blood now?",
                            )

        self.assertTrue(result["success"])
        self.assertEqual(mock_resolve.call_count, 1)
        self.assertEqual(mock_predict.call_count, 1)
        self.assertFalse(result["execution_results"][1]["success"])
        self.assertEqual(result["execution_results"][1]["output"], {})

    def test_horizon_routing_picks_best_validation_accuracy_for_requested_horizon(self):
        env = {
            "PIOS_XLAM_DISABLE_LLM": "1",
            "PIOS_ORCH_ENABLE_DB_TOOL": "0",
            "PIOS_ORCH_ENABLE_SEARCH_TOOL": "0",
            "PIOS_ORCH_ENABLE_SIMULATION_TOOL": "0",
        }

        prophet = _StubModelRuntime("prophet_stockout_model")
        prophet.description = "Prophet stockout forecaster."
        prophet.defaults = {
            "forecast_selection": {"target": "stockout", "primary_metric": "accuracy"},
            "validation_metrics": {
                "by_horizon": {
                    "long": {"accuracy": 0.91},
                }
            },
        }

        xgboost = _StubModelRuntime("xgboost_stockout_model")
        xgboost.description = "XGBoost stockout forecaster."
        xgboost.defaults = {
            "forecast_selection": {"target": "stockout", "primary_metric": "accuracy"},
            "validation_metrics": {
                "by_horizon": {
                    "long": {"accuracy": 0.96},
                }
            },
        }

        lstm = _StubModelRuntime("lstm_stockout_model")
        lstm.description = "LSTM stockout forecaster."
        lstm.defaults = {
            "forecast_selection": {"target": "stockout", "primary_metric": "accuracy"},
            "validation_metrics": {
                "by_horizon": {
                    "long": {"accuracy": 0.93},
                }
            },
        }

        registry = _DummyRegistry()
        for runtime in (prophet, xgboost, lstm):
            registry._models[runtime.model_id] = runtime

        with patch.dict(os.environ, env, clear=False):
            orchestrator = DynamicXLAMOrchestrator(registry=registry)
            with patch(
                "ml.orchestrator.service.resolve_prediction_features",
                return_value=types.SimpleNamespace(
                    features={"hospital_id": "H001"},
                    metadata={"source": "request"},
                ),
            ) as mock_resolve:
                with patch(
                    "ml.orchestrator.service.run_prediction_with_fallback",
                    return_value={"prediction": 0.8},
                ):
                    result = orchestrator.run(
                        "Predict hospital shortage risk over the next 30 days.",
                        provided_features={"hospital_id": "H001"},
                    )

        self.assertTrue(result["success"])
        self.assertEqual(
            result["execution_results"][0]["tool"],
            "xgboost_stockout_model",
        )
        self.assertTrue(result["horizon_routing"]["applied"])
        self.assertEqual(result["horizon_routing"]["selected_model_id"], "xgboost_stockout_model")
        self.assertEqual(result["horizon_routing"]["scores"][0]["metric_name"], "accuracy")
        self.assertEqual(result["horizon_routing"]["scores"][0]["metric_value"], 0.96)
        self.assertEqual(result["forecast_horizon"]["timeframe"], "long")
        self.assertEqual(result["forecast_horizon"]["days"], 30.0)
        runtime_arg, features_arg = mock_resolve.call_args.args
        self.assertEqual(runtime_arg.model_id, "xgboost_stockout_model")
        self.assertEqual(features_arg["hospital_id"], "H001")
        self.assertNotIn("forecast_timeframe", features_arg)
        self.assertNotIn("prediction_horizon_days", features_arg)

    def test_horizon_routing_overrides_llm_external_plan_for_stockout_model(self):
        env = {
            "PIOS_XLAM_DISABLE_LLM": "1",
            "PIOS_ORCH_ENABLE_DB_TOOL": "1",
            "PIOS_ORCH_ENABLE_SEARCH_TOOL": "1",
            "PIOS_ORCH_ENABLE_SIMULATION_TOOL": "0",
        }
        runtime = self._hospital_shortage_runtime()
        runtime.description = "Hospital stockout days-to-stockout predictor."
        runtime.feature_names = ["hospital_id", "temperature", "current_inventory"]
        runtime.defaults.update({
            "forecast_selection": {
                "target": "stockout",
                "primary_metric": "root_mean_squared_error",
                "direction": "minimize",
            },
            "validation_metrics": {
                "overall_metrics": {"root_mean_squared_error": 3.1}
            },
        })
        registry = _DummyRegistry()
        registry._models[runtime.model_id] = runtime
        fake_external_plan = ExecutionPlan(
            steps=[
                ToolCall(
                    name="search",
                    arguments={"query": "stockout platform"},
                    reasoning="LLM incorrectly chose external search.",
                )
            ],
            reasoning="bad llm plan",
            is_multi_step=False,
        )

        with patch.dict(os.environ, env, clear=False):
            orchestrator = DynamicXLAMOrchestrator(registry=registry)
            with patch.object(
                orchestrator,
                "_plan_execution",
                return_value=(fake_external_plan, "llm"),
            ):
                with patch(
                    "ml.orchestrator.service.resolve_prediction_features",
                ) as mock_resolve:
                    with patch(
                        "ml.orchestrator.service.run_prediction_with_fallback",
                        return_value={"prediction": 4},
                    ) as mock_predict:
                        with patch(
                            "ml.orchestrator.service.build_hybrid_stockout_prediction",
                            return_value={
                                "regression_model_id": runtime.model_id,
                                "estimated_days_until_stockout": 4,
                                "stockout_probability": 0.63,
                                "horizon": "1-7 days",
                                "risk_level": "high",
                                "recommended_action": "Prepare transfer request and targeted donor outreach",
                                "hybrid_prediction": True,
                            },
                        ) as mock_hybrid:
                            with patch.object(
                                orchestrator, "_execute_search_tool"
                            ) as mock_search:
                                result = orchestrator.run(
                                    "Predict stockout risk for hospital H001 over the next 3 days",
                                    provided_features={"hospital_id": "H001"},
                                )

        self.assertTrue(result["success"])
        self.assertEqual(result["planner_mode"], "llm_horizon_routed")
        self.assertEqual(
            result["execution_results"][0]["tool"],
            runtime.model_id,
        )
        self.assertEqual(result["horizon_routing"]["selected_model_id"], runtime.model_id)
        mock_hybrid.assert_called_once()
        mock_resolve.assert_called_once()
        mock_predict.assert_called_once()
        mock_search.assert_not_called()

    def test_horizon_routing_does_not_route_hospital_query_to_supply_model(self):
        env = {
            "PIOS_XLAM_DISABLE_LLM": "1",
            "PIOS_ORCH_ENABLE_DB_TOOL": "0",
            "PIOS_ORCH_ENABLE_SEARCH_TOOL": "0",
            "PIOS_ORCH_ENABLE_SIMULATION_TOOL": "0",
        }
        supply_model = _StubModelRuntime("stockout_days_predictor_champion")
        supply_model.description = "Medium-horizon stockout predictor."
        supply_model.defaults = {
            "forecast_selection": {
                "target": "stockout",
                "primary_metric": "root_mean_squared_error",
                "direction": "minimize",
            },
            "horizon_routing": {
                "enabled": True,
                "timeframes": ["medium"],
                "min_days": 8,
                "max_days": 29,
                "target": "stockout",
            },
            "feast": {"entity_keys": ["supply_id"]},
            "validation_metrics": {
                "overall_metrics": {"root_mean_squared_error": 0.2}
            },
        }
        hospital_model = self._hospital_shortage_runtime()
        hospital_model.description = "Hospital stockout predictor."
        hospital_model.defaults.update({
            "forecast_selection": {
                "target": "stockout",
                "primary_metric": "root_mean_squared_error",
                "direction": "minimize",
            },
            "horizon_routing": {
                "enabled": True,
                "timeframes": ["short"],
                "min_days": 0,
                "max_days": 7,
                "target": "stockout",
            },
            "feast": {"entity_keys": ["hospital_id"]},
            "validation_metrics": {
                "overall_metrics": {"root_mean_squared_error": 3.0}
            },
        })
        registry = _DummyRegistry()
        registry._models[supply_model.model_id] = supply_model
        registry._models[hospital_model.model_id] = hospital_model

        with self._orchestrator_with_tools_enabled(
            registry, stockout_hybrid=True, extra_env=env
        ) as orchestrator:
            with patch(
                "ml.orchestrator.service.resolve_prediction_features",
                return_value=types.SimpleNamespace(
                    features={"hospital_id": "H001"},
                    metadata={"source": "request"},
                ),
            ) as mock_resolve:
                with patch(
                    "ml.orchestrator.service.resolve_component_prediction_features",
                    return_value=[],
                ):
                    with patch(
                        "ml.orchestrator.service.run_prediction_with_fallback",
                        return_value={
                            "prediction": 6.5,
                            "prediction_name": "days_until_stockout",
                            "prediction_unit": "days",
                        },
                    ):
                        with patch(
                            "ml.orchestrator.service.build_hybrid_stockout_prediction",
                            return_value={
                                "regression_model_id": ACTIVE_STOCKOUT_MODEL_ID,
                                "estimated_days_until_stockout": 14,
                                "stockout_probability": 0.42,
                                "horizon": "8-30 days",
                                "risk_level": "medium",
                                "recommended_action": "Increase monitoring and schedule replenishment",
                                "hybrid_prediction": True,
                            },
                        ) as mock_hybrid:
                            result = orchestrator.run(
                                "Predict stockout risk for hospital H001 over the next 14 days",
                                provided_features={"hospital_id": "H001"},
                            )

        self.assertTrue(result["success"])
        self.assertEqual(
            result["execution_results"][0]["tool"],
            orchestrator.stockout_hybrid_tool_name,
        )
        self.assertEqual(
            result["stockout_prediction"].get("regression_model_id"),
            ACTIVE_STOCKOUT_MODEL_ID,
        )
        self.assertFalse(result["horizon_routing"]["applied"])
        mock_resolve.assert_not_called()
        self.assertEqual(
            mock_hybrid.call_args.kwargs["regression_runtime"].model_id,
            ACTIVE_STOCKOUT_MODEL_ID,
        )

    def test_stockout_hospital_query_returns_hybrid_component_forecasts(self):
        env = {
            "PIOS_XLAM_DISABLE_LLM": "1",
            "PIOS_ORCH_ENABLE_DB_TOOL": "0",
            "PIOS_ORCH_ENABLE_SEARCH_TOOL": "0",
            "PIOS_ORCH_ENABLE_SIMULATION_TOOL": "0",
        }
        runtime = self._hospital_shortage_runtime()
        runtime.description = "Hospital stockout predictor."
        runtime.defaults.update({
            "forecast_selection": {
                "target": "stockout",
                "primary_metric": "root_mean_squared_error",
                "direction": "minimize",
            },
            "validation_metrics": {
                "overall_metrics": {"root_mean_squared_error": 3.0}
            },
        })
        registry = _DummyRegistry()
        registry._models[runtime.model_id] = runtime
        with self._orchestrator_with_tools_enabled(
            registry,
            stockout_hybrid=True,
            extra_env=env,
        ) as orchestrator:
            with patch(
                "ml.orchestrator.service.build_hybrid_stockout_prediction",
                return_value={
                    "regression_model_id": runtime.model_id,
                    "blood_component": "overall",
                    "estimated_days_until_stockout": 0,
                    "stockout_probability": 0.9,
                    "horizon": "1-7 days",
                    "risk_level": "critical",
                    "recommended_action": "Urgent donor campaign or transfer request",
                    "hybrid_prediction": True,
                    "worst_component": "PLATELETS",
                    "component_predictions": [
                        {
                            "blood_component": "RBC",
                            "estimated_days_until_stockout": 3,
                            "stockout_probability": 0.61,
                        },
                        {
                            "blood_component": "PLATELETS",
                            "estimated_days_until_stockout": 0,
                            "stockout_probability": 0.9,
                        },
                    ],
                },
            ) as mock_hybrid:
                with patch(
                    "ml.orchestrator.service.resolve_prediction_features"
                ) as mock_single_resolve:
                    with patch(
                        "ml.orchestrator.service.run_prediction_with_fallback",
                    ) as mock_predict:
                        result = orchestrator.run(
                            "Predict stockout risk for hospital H001 over the next 3 days",
                            provided_features={"hospital_id": "H001"},
                        )

        self.assertTrue(result["success"])
        output = result["stockout_prediction"]
        self.assertEqual(
            [row["blood_component"] for row in output["component_predictions"]],
            ["RBC", "PLATELETS"],
        )
        self.assertEqual(output["worst_component"], "PLATELETS")
        response = result["natural_language_response"].lower()
        self.assertIn("stockout probability: 0.9000", response)
        self.assertIn("highest-risk component: platelets", response)
        self.assertNotIn("component details", response)
        self.assertNotIn("immediate stockout", response)
        self.assertNotIn("no stockout", response)
        execution_output = result["execution_results"][0]["output"]
        self.assertEqual(execution_output["component_count"], 2)
        self.assertNotIn("component_predictions", execution_output)
        mock_hybrid.assert_called_once()
        mock_single_resolve.assert_not_called()
        mock_predict.assert_not_called()

    def test_summarizer_prompt_explains_time_to_event_outputs_generically(self):
        env = {"PIOS_XLAM_DISABLE_LLM": "1"}
        with patch.dict(os.environ, env, clear=False):
            orchestrator = DynamicXLAMOrchestrator(registry=_DummyRegistry())
        orchestrator._llm = object()
        captured = {}

        def fake_generate(prompt, **_kwargs):
            captured["prompt"] = prompt
            return "summary"

        output = {
            "prediction_name": "days_until_stockout",
            "prediction_unit": "days",
            "component_forecasts": [
                {
                    "component": "A+",
                    "prediction": 0.0,
                    "prediction_name": "days_until_stockout",
                    "prediction_unit": "days",
                }
            ],
        }
        with patch.object(orchestrator, "_safe_generation_tokens", return_value=64):
            with patch.object(orchestrator, "_generate_llm_text", side_effect=fake_generate):
                response = orchestrator._summarize(
                    "Predict stockout for hospital H001 tomorrow",
                    [ExecutionResult(1, "any_time_model", {}, output, True)],
                    ExecutionPlan([], "", False),
                    "llm",
                )

        self.assertEqual(response, "summary")
        prompt = captured["prompt"]
        self.assertIn("Return Markdown text", prompt)
        self.assertIn("Markdown answer", prompt)
        self.assertIn("days_until_*", prompt)
        self.assertIn("time_to_*", prompt)
        self.assertIn("component forecasts (days until stockout): A+: 0.00 days", prompt)
        self.assertIn("not that the event is absent", prompt)
        self.assertNotIn("immediate stockout", prompt)

    def test_non_llm_summarizer_renders_batch_ranking_as_markdown_table(self):
        env = {"PIOS_XLAM_DISABLE_LLM": "1"}
        with patch.dict(os.environ, env, clear=False):
            orchestrator = DynamicXLAMOrchestrator(registry=_DummyRegistry())

        response = orchestrator._summarize(
            "best donors to call",
            [
                ExecutionResult(
                    1,
                    "configured_data_tool",
                    {},
                    {
                        "query_kind": "list",
                        "rows": [{"donor_id": 1, "blood_type": "A+"}],
                        "answer": "Returned bounded rows",
                    },
                    True,
                ),
                ExecutionResult(
                    2,
                    "configured_ranking_model",
                    {},
                    {
                        "query_kind": "batch_model_ranking",
                        "rows": [
                            {
                                "rank": 1,
                                "donor_id": 1,
                                "blood_type": "A+",
                                "model_score": 1.0,
                                "model_output": {"probability": 1.0},
                            },
                            {
                                "rank": 2,
                                "donor_id": 4,
                                "blood_type": "O+",
                                "model_score": 0.91,
                            },
                        ],
                        "answer": "Ranked 2 donor_id candidate(s).",
                    },
                    True,
                ),
            ],
            ExecutionPlan([], "", False),
            "heuristic",
        )

        self.assertIn("## Ranked Results", response)
        self.assertIn("| Rank | Donor Id | Blood Type | Model Score |", response)
        self.assertIn("| 1 | 1 | A+ | 1.0000 |", response)
        self.assertIn("| 2 | 4 | O+ | 0.9100 |", response)
        self.assertIn("Ranked 2 donor_id candidate(s).", response)
        self.assertNotIn("model_output", response)
        self.assertNotIn("[{", response)

    def test_horizon_routing_accepts_timeframe_feature_and_strips_it_before_prediction(
        self,
    ):
        env = {
            "PIOS_XLAM_DISABLE_LLM": "1",
            "PIOS_ORCH_ENABLE_DB_TOOL": "0",
            "PIOS_ORCH_ENABLE_SEARCH_TOOL": "0",
            "PIOS_ORCH_ENABLE_SIMULATION_TOOL": "0",
        }

        lower_accuracy = _StubModelRuntime("stockout_medium_baseline")
        lower_accuracy.description = "Stockout medium-horizon baseline."
        lower_accuracy.defaults = {
            "forecast_selection": {"target": "stockout", "primary_metric": "accuracy"},
            "validation_metrics": {
                "by_horizon": {
                    "medium": {"accuracy": 0.81},
                }
            },
        }

        higher_accuracy = _StubModelRuntime("stockout_medium_champion")
        higher_accuracy.description = "Stockout medium-horizon champion."
        higher_accuracy.defaults = {
            "forecast_selection": {"target": "stockout", "primary_metric": "accuracy"},
            "validation_metrics": {
                "by_horizon": {
                    "medium": {"accuracy": 0.89},
                }
            },
        }

        registry = _DummyRegistry()
        registry._models[lower_accuracy.model_id] = lower_accuracy
        registry._models[higher_accuracy.model_id] = higher_accuracy

        with patch.dict(os.environ, env, clear=False):
            orchestrator = DynamicXLAMOrchestrator(registry=registry)
            with patch(
                "ml.orchestrator.service.resolve_prediction_features",
                return_value=types.SimpleNamespace(
                    features={"supply_id": "S001"},
                    metadata={"source": "request"},
                ),
            ) as mock_resolve:
                with patch(
                    "ml.orchestrator.service.run_prediction_with_fallback",
                    return_value={"prediction": 12.0},
                ):
                    result = orchestrator.run(
                        "Predict hospital inventory risk.",
                        provided_features={
                            "forecast_timeframe": "meduim",
                            "supply_id": "S001",
                        },
                    )

        self.assertTrue(result["success"])
        self.assertEqual(result["execution_results"][0]["tool"], "stockout_medium_champion")
        self.assertTrue(result["horizon_routing"]["applied"])
        self.assertEqual(result["forecast_horizon"]["timeframe"], "medium")
        _, features_arg = mock_resolve.call_args.args
        self.assertEqual(features_arg["supply_id"], "S001")
        self.assertNotIn("forecast_timeframe", features_arg)

    def test_horizon_routing_excludes_non_stockout_models_without_explicit_target(
        self,
    ):
        env = {
            "PIOS_XLAM_DISABLE_LLM": "1",
            "PIOS_ORCH_ENABLE_DB_TOOL": "0",
            "PIOS_ORCH_ENABLE_SEARCH_TOOL": "0",
            "PIOS_ORCH_ENABLE_SIMULATION_TOOL": "0",
        }

        stockout = _StubModelRuntime("hospital_shortage_predictor_champion")
        stockout.description = "Hospital stockout risk predictor."
        stockout.defaults = {
            "forecast_selection": {"target": "stockout", "primary_metric": "accuracy"},
            "validation_metrics": {"overall_metrics": {"accuracy": 0.91}},
        }

        donor = _StubModelRuntime("donor_propensity_model_champion")
        donor.description = "Donor propensity model."
        donor.defaults = {
            "validation_metrics": {"overall_metrics": {"accuracy": 0.99}},
        }

        registry = _DummyRegistry()
        registry._models[stockout.model_id] = stockout
        registry._models[donor.model_id] = donor

        with patch.dict(os.environ, env, clear=False):
            orchestrator = DynamicXLAMOrchestrator(registry=registry)
            with patch(
                "ml.orchestrator.service.resolve_prediction_features",
                return_value=types.SimpleNamespace(
                    features={"hospital_id": "H001"},
                    metadata={"source": "request"},
                ),
            ):
                with patch(
                    "ml.orchestrator.service.run_prediction_with_fallback",
                    return_value={"prediction": 0.8},
                ):
                    result = orchestrator.run(
                        "Predict stockout risk for hospital H001 over the next 3 days",
                        provided_features={"hospital_id": "H001"},
                    )

        self.assertTrue(result["success"])
        self.assertEqual(
            result["horizon_routing"]["candidate_model_ids"],
            ["hospital_shortage_predictor_champion"],
        )
        self.assertEqual(
            result["horizon_routing"]["selected_model_id"],
            "hospital_shortage_predictor_champion",
        )

    def test_tool_schema_summary_hides_target_like_fields_from_config_overrides(self):
        env = {
            "PIOS_XLAM_DISABLE_LLM": "1",
        }
        schema_payload = {
            "donors": {
                "fields": [
                    "donor_id",
                    "donated_next_6m",
                    "prediction_score",
                    "age",
                ],
                "notes": "example",
            }
        }

        with patch.dict(os.environ, env, clear=False):
            orchestrator = DynamicXLAMOrchestrator(registry=_DummyRegistry())
            summary = orchestrator._tool_schema_summary(schema_payload)

        self.assertIn("donors(donor_id, age)", summary)
        self.assertNotIn("donated_next_6m", summary)
        self.assertNotIn("prediction_score", summary)

    def test_llm_generation_error_falls_back_with_accurate_reason(self):
        env = {
            "PIOS_XLAM_DISABLE_LLM": "0",
            "PIOS_ORCH_ENABLE_DB_TOOL": "0",
            "PIOS_ORCH_ENABLE_SEARCH_TOOL": "0",
            "PIOS_ORCH_ENABLE_SIMULATION_TOOL": "1",
        }

        class _ExplodingLLM:
            def __call__(self, prompt, **kwargs):
                raise RuntimeError("llama_decode returned -3")

            def tokenize(self, data):
                return list(data)

        with self._orchestrator_with_tools_enabled(
            _DummyRegistry(),
            simulation=True,
            extra_env=env,
        ) as orchestrator:
            orchestrator._llm = _ExplodingLLM()
            orchestrator._llm_attempted = True
            orchestrator._llm_n_ctx = 4096
            with patch.object(
                orchestrator,
                "_execute_simulation_tool",
                return_value={
                    "tool_type": "simulation",
                    "answer": "simulation fallback worked",
                },
            ):
                result = orchestrator.run(
                    "What would happen if blood donors dropped by 50% in the next week?"
                )

        self.assertTrue(result["success"])
        self.assertEqual(result["planner_mode"], "function_tool")
        self.assertIn("Routed to simulation tool", result["plan"]["reasoning"])
        self.assertNotIn("orchestrator LLM unavailable", result["plan"]["reasoning"])

    def test_simulation_query_plan_is_repaired_to_simulation_tool(self):
        env = {
            "PIOS_XLAM_DISABLE_LLM": "1",
            "PIOS_ORCH_ENABLE_DB_TOOL": "1",
            "PIOS_ORCH_ENABLE_SEARCH_TOOL": "1",
            "PIOS_ORCH_ENABLE_SIMULATION_TOOL": "1",
        }
        with self._orchestrator_with_tools_enabled(
            _DummyRegistry(),
            simulation=True,
            extra_env=env,
        ) as orchestrator:
            plan = ExecutionPlan(
                steps=[
                    ToolCall(
                        name=orchestrator.search_tool_name,
                        arguments={"query": "fallback"},
                        reasoning="incorrect search plan",
                    )
                ],
                reasoning="search first",
                is_multi_step=False,
            )
            repaired = orchestrator._repair_plan_for_simulation_query(
                "Compare DreamerV3, PPO, and SAC under a 40% donor drop next week.",
                plan,
            )

        self.assertEqual(len(repaired.steps), 1)
        self.assertEqual(repaired.steps[0].name, orchestrator.simulation_tool_name)
        self.assertEqual(
            repaired.steps[0].arguments["query"],
            "Compare DreamerV3, PPO, and SAC under a 40% donor drop next week.",
        )

    def test_execute_db_schema_tool_returns_all_tables_and_columns(self):
        from ml.core.external_tools import execute_db_schema_tool

        res = execute_db_schema_tool()
        self.assertIn("tables", res)
        self.assertIn("table_names", res)
        self.assertIn("donors", res["tables"])
        self.assertIn("hospitals", res["tables"])

        donors_schema = res["tables"]["donors"]
        self.assertEqual(donors_schema["logical_table"], "donors")
        self.assertGreater(len(donors_schema["columns"]), 10)
        donor_col_names = [col["name"] for col in donors_schema["columns"]]
        self.assertIn("donor_id", donor_col_names)
        self.assertIn("eligible_to_donate", donor_col_names)

        hospitals_schema = res["tables"]["hospitals"]
        self.assertEqual(hospitals_schema["logical_table"], "hospitals")
        hospital_col_names = [col["name"] for col in hospitals_schema["columns"]]
        self.assertIn("hospital_id", hospital_col_names)
        self.assertIn("stock_end", hospital_col_names)

    def test_execute_db_schema_tool_filters_by_table(self):
        from ml.core.external_tools import execute_db_schema_tool

        res = execute_db_schema_tool(table="hospitals")
        self.assertEqual(res["table_names"], ["hospitals"])
        self.assertIn("hospitals", res["tables"])
        self.assertNotIn("donors", res["tables"])

    def test_repair_row_source_arguments_does_not_inject_donor_filters_into_hospitals_query(self):
        env = {
            "PIOS_XLAM_DISABLE_LLM": "1",
            "PIOS_ORCH_ENABLE_DB_TOOL": "1",
        }
        with self._orchestrator_with_tools_enabled(
            _DummyRegistry(),
            extra_env=env,
        ) as orchestrator:
            hospital_args = {
                "table": "hospitals",
                "aggregate": "list",
                "fields": ["hospital_id", "hospital_name", "blood_type", "stock_end"],
                "filters": {"hospital_name": "central hospital", "blood_type": "O+"},
            }
            repaired = orchestrator._repair_row_source_arguments(
                query="Inspect stockout hazard for hospital inventory",
                tool_name=orchestrator.db_tool_name,
                arguments=hospital_args,
                results=[],
            )

        filters = repaired.get("filters", {})
        self.assertNotIn("eligible_to_donate", filters)
        self.assertEqual(filters.get("hospital_name"), "central hospital")
        self.assertEqual(filters.get("blood_type"), "O+")

