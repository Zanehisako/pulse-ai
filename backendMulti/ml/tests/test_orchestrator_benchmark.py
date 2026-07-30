from __future__ import annotations

import tempfile
import unittest
import json
from pathlib import Path

import pandas as pd

from ml.core.simulation_tool import SimulationToolConfig, execute_simulation_query
from ml.orchestrator.benchmark import (
    DeterministicDbToolBackend,
    build_benchmark_cases,
    build_reference_predictive_models,
    build_tool_menus,
    generate_plots,
    PREDICTIVE_MODEL_SPECS,
    simulation_stub_server,
)


class OrchestratorBenchmarkTests(unittest.TestCase):
    def build_fixture_workspace(self, tmp_dir: str) -> Path:
        workspace = Path(tmp_dir)
        for index, spec in enumerate(PREDICTIVE_MODEL_SPECS, start=1):
            artifact_path = workspace / spec["artifact_relpath"]
            artifact_path.mkdir(parents=True, exist_ok=True)
            (artifact_path / "input_example.json").write_text(
                json.dumps(
                    {
                        "columns": [f"feature_{index}", "current_inventory"],
                        "data": [[float(index), 10.0 + index]],
                    }
                ),
                encoding="utf-8",
            )
        return workspace

    def test_reference_models_load_sample_feature_vectors(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            models = build_reference_predictive_models(
                workspace_root=self.build_fixture_workspace(tmp_dir)
            )

        self.assertIn("donor_propensity_model", models)
        self.assertIn("hospital_shortage_predictor", models)
        self.assertIn("stockout_days_predictor", models)
        self.assertTrue(models["donor_propensity_model"].sample_features)
        self.assertTrue(models["hospital_shortage_predictor"].sample_features)
        self.assertTrue(models["stockout_days_predictor"].sample_features)

    def test_case_matrix_spans_multiple_tool_menu_sizes(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            models = build_reference_predictive_models(
                workspace_root=self.build_fixture_workspace(tmp_dir)
            )
        menus = build_tool_menus(models)
        cases = build_benchmark_cases(models)

        tool_counts = {
            len(menus[case.menu_id].available_tools(tuple(models.keys()))) for case in cases
        }
        families = {case.family for case in cases}

        self.assertGreaterEqual(len(tool_counts), 3)
        self.assertIn("inventory_lookup", families)
        self.assertIn("shock_forecast", families)
        self.assertIn("donor_propensity", families)

    def test_db_fixture_backend_returns_stable_structured_answers(self):
        backend = DeterministicDbToolBackend()

        donor_avg = backend.execute(
            "What is the average donor age by country code?",
            table="donors",
            field="age",
            aggregate="avg",
        )
        stock_lookup = backend.execute(
            "Which hospitals currently have the highest O+ stock?",
            table="hospitals",
            field="stock_end",
            aggregate="max",
        )

        self.assertEqual(donor_avg["table"], "donors")
        self.assertEqual(donor_avg["field"], "age")
        self.assertIn("Average donor age by country code", donor_avg["answer"])
        self.assertEqual(stock_lookup["table"], "hospitals")
        self.assertEqual(stock_lookup["field"], "stock_end")
        self.assertIn("Highest O+ stock snapshot", stock_lookup["answer"])

    def test_simulation_stub_server_executes_real_simulation_adapter(self):
        config = SimulationToolConfig(
            base_url="http://127.0.0.1:0",
            timeout_s=10.0,
            default_policy="dreamerv3_official",
            compare_on_recommend=False,
            default_compare_policies=[
                "dreamerv3_official",
                "ppo_continuous",
                "sac_continuous",
            ],
        )

        with simulation_stub_server() as server:
            config = SimulationToolConfig(
                base_url=server.base_url,
                timeout_s=10.0,
                default_policy="dreamerv3_official",
                compare_on_recommend=False,
                default_compare_policies=[
                    "dreamerv3_official",
                    "ppo_continuous",
                    "sac_continuous",
                ],
            )
            result = execute_simulation_query(
                "Compare DreamerV3, PPO, and SAC for a 30% donor drop next week.",
                config=config,
            )

        self.assertTrue(result["backend_success"])
        self.assertEqual(result["mode"], "compare")
        self.assertIn("Best performer", result["answer"])
        self.assertEqual(
            result["normalized_result"]["best_policy"]["policy_key"],
            "dreamerv3_official",
        )

    def test_plot_generation_writes_expected_files(self):
        rows = [
            {
                "orchestrator_model_name": "xLAM 7B Q2_K",
                "orchestrator_model_id": "xlam_7b_q2_k",
                "case_id": "inventory_lookup_all",
                "family": "inventory_lookup",
                "label": "Inventory lookup",
                "category": "db_search",
                "menu_id": "all_tools_all_models",
                "menu_label": "All tools + all predictive models",
                "available_tool_count": 5,
                "available_tools": "db_tool, simulation, donor_propensity_model",
                "query": "Which hospitals currently have the highest O+ stock?",
                "expected_tool": "db_tool",
                "planned_tool": "db_tool",
                "executed_tool": "db_tool",
                "planned_family": "db_search",
                "executed_family": "db_search",
                "planned_tool_match": 1.0,
                "executed_tool_match": 1.0,
                "execution_success": 1.0,
                "menu_adherence": 1.0,
                "argument_match": 1.0,
                "case_score": 1.0,
                "latency_ms": 1500.0,
                "planner_mode": "llm",
                "summary": "test",
                "load_time_ms": 3000.0,
            },
            {
                "orchestrator_model_name": "Qwen2 7B Instruct Q4_0",
                "orchestrator_model_id": "qwen2_7b_instruct_q4_0",
                "case_id": "shock_forecast_all",
                "family": "shock_forecast",
                "label": "Shock forecast",
                "category": "simulation",
                "menu_id": "all_tools_all_models",
                "menu_label": "All tools + all predictive models",
                "available_tool_count": 5,
                "available_tools": "db_tool, simulation, donor_propensity_model",
                "query": "What would happen if blood donors dropped by 50%?",
                "expected_tool": "simulation",
                "planned_tool": "simulation",
                "executed_tool": "simulation",
                "planned_family": "simulation",
                "executed_family": "simulation",
                "planned_tool_match": 1.0,
                "executed_tool_match": 1.0,
                "execution_success": 1.0,
                "menu_adherence": 1.0,
                "argument_match": 1.0,
                "case_score": 1.0,
                "latency_ms": 2200.0,
                "planner_mode": "llm",
                "summary": "test",
                "load_time_ms": 4000.0,
            },
        ]
        results_df = pd.DataFrame(rows)
        overall_df = (
            results_df.groupby(["orchestrator_model_id", "orchestrator_model_name"], as_index=False)
            .agg(
                runs=("case_id", "count"),
                load_time_ms=("load_time_ms", "max"),
                planned_tool_match=("planned_tool_match", "mean"),
                executed_tool_match=("executed_tool_match", "mean"),
                execution_success=("execution_success", "mean"),
                menu_adherence=("menu_adherence", "mean"),
                argument_match=("argument_match", "mean"),
                case_score=("case_score", "mean"),
                median_latency_ms=("latency_ms", "median"),
            )
        )
        family_df = (
            results_df.groupby(["orchestrator_model_name", "category"], as_index=False)
            .agg(
                planned_tool_match=("planned_tool_match", "mean"),
                executed_tool_match=("executed_tool_match", "mean"),
                case_score=("case_score", "mean"),
                latency_ms=("latency_ms", "median"),
            )
        )
        menu_df = (
            results_df.groupby(["orchestrator_model_name", "menu_id", "menu_label"], as_index=False)
            .agg(
                runs=("case_id", "count"),
                planned_tool_match=("planned_tool_match", "mean"),
                executed_tool_match=("executed_tool_match", "mean"),
                menu_adherence=("menu_adherence", "mean"),
                case_score=("case_score", "mean"),
            )
        )

        with tempfile.TemporaryDirectory() as tmp_dir:
            output_dir = Path(tmp_dir)
            plot_paths = generate_plots(
                results_df=results_df,
                overall_df=overall_df,
                family_df=family_df,
                menu_df=menu_df,
                output_dir=output_dir,
            )

            self.assertTrue(plot_paths)
            for path in plot_paths:
                self.assertTrue(path.exists(), msg=f"Expected plot file {path} to exist")


if __name__ == "__main__":
    unittest.main()

