from __future__ import annotations

import unittest
from unittest.mock import patch

from ml.core.simulation_tool import (
    SimulationToolConfig,
    build_simulation_request,
    execute_simulation_query,
    is_simulation_query,
)


def _metadata_payload() -> dict:
    return {
        "scenarios": [
            {
                "key": "baseline",
                "name": "Normal Operations",
                "description": "Balanced baseline.",
                "params": {
                    "sim_hours": 168,
                    "donor_show_factor": 1.0,
                    "demand_surge_factor": 1.0,
                    "transport_penalty": 1.0,
                    "initial_inventory_days": 3.0,
                    "reserve_target_days": 2.0,
                    "demand_forecast_interval_h": 24.0,
                    "congestion_delay_factor": 0.05,
                    "episode_budget": 4000.0,
                },
            }
        ],
        "strategies": [
            {"key": "baseline", "name": "Baseline", "available": True},
            {
                "key": "dreamerv3_official",
                "name": "Official DreamerV3",
                "available": True,
            },
            {"key": "ppo_continuous", "name": "PPO Continuous", "available": True},
            {"key": "sac_continuous", "name": "SAC Continuous", "available": True},
            {"key": "iql_offline", "name": "IQL Offline", "available": True},
            {"key": "cql_offline", "name": "CQL Offline", "available": True},
        ],
        "actions": [
            {
                "key": "campaign",
                "name": "Targeted donor campaign",
                "description": "Outreach and SMS campaign.",
                "operational_cost": 60,
            },
            {
                "key": "emergency_share",
                "name": "Emergency sharing and courier routing",
                "description": "Prioritize transfers and emergency couriers.",
                "operational_cost": 65,
            },
        ],
        "custom_scenario_editor": {
            "fields": [
                {
                    "key": "sim_hours",
                    "label": "Simulation Hours",
                    "input": "integer",
                    "min": 24,
                    "max": 1440,
                },
                {
                    "key": "donor_show_factor",
                    "label": "Donor Show Factor",
                    "input": "number",
                    "min": 0.1,
                    "max": 2.5,
                },
                {
                    "key": "demand_surge_factor",
                    "label": "Demand Surge Factor",
                    "input": "number",
                    "min": 0.2,
                    "max": 4.0,
                },
                {
                    "key": "transport_penalty",
                    "label": "Transport Penalty",
                    "input": "number",
                    "min": 0.5,
                    "max": 5.0,
                },
                {
                    "key": "initial_inventory_days",
                    "label": "Initial Inventory Buffer",
                    "input": "number",
                    "min": 0.5,
                    "max": 14.0,
                },
                {
                    "key": "reserve_target_days",
                    "label": "Reserve Target",
                    "input": "number",
                    "min": 0.5,
                    "max": 14.0,
                },
                {
                    "key": "demand_forecast_interval_h",
                    "label": "Forecast Refresh Interval",
                    "input": "number",
                    "min": 1.0,
                    "max": 168.0,
                },
                {
                    "key": "congestion_delay_factor",
                    "label": "Congestion Delay Factor",
                    "input": "number",
                    "min": 0.0,
                    "max": 1.0,
                },
                {
                    "key": "episode_budget",
                    "label": "Episode Budget",
                    "input": "number",
                    "min": 0.0,
                    "max": 100000.0,
                },
            ]
        },
        "default_dreamerv3_run_key": "pink_demo_run",
        "defaults": {
            "comparison_strategies": [
                "dreamerv3_official",
                "ppo_continuous",
                "sac_continuous",
                "iql_offline",
                "cql_offline",
            ]
        },
    }


def _simulation_config() -> SimulationToolConfig:
    return SimulationToolConfig(
        base_url="http://simulation.local",
        timeout_s=15.0,
        default_policy="dreamerv3_official",
        compare_on_recommend=False,
        default_compare_policies=[
            "dreamerv3_official",
            "ppo_continuous",
            "sac_continuous",
            "iql_offline",
            "cql_offline",
        ],
    )


class SimulationToolTests(unittest.TestCase):
    def test_is_simulation_query_detects_forecast_recommend_and_compare(self):
        self.assertTrue(
            is_simulation_query(
                "What would happen if blood donors dropped by 50% in the next week?"
            )
        )
        self.assertTrue(
            is_simulation_query(
                "What should we do if blood donors dropped by 50% in the next week?"
            )
        )
        self.assertTrue(
            is_simulation_query(
                "Compare PPO, SAC, and DreamerV3 for a 30% donor drop next week."
            )
        )
        self.assertFalse(
            is_simulation_query("Predict donor eligibility for a 35 year old donor.")
        )

    def test_build_simulation_request_defaults_to_dreamerv3_for_recommend(self):
        request = build_simulation_request(
            "What should we do if blood donors dropped by 50% in the next week?",
            metadata=_metadata_payload(),
            config=_simulation_config(),
        )

        self.assertEqual(request.mode, "recommend")
        self.assertEqual(request.scenario_kind, "custom")
        self.assertEqual(request.selected_policy_key, "dreamerv3_official")
        self.assertEqual(request.hours_override, 168)
        self.assertIsNotNone(request.custom_scenario_payload)
        self.assertEqual(request.custom_scenario_payload["donor_show_factor"], 0.5)

    def test_build_simulation_request_defaults_compare_policy_set(self):
        request = build_simulation_request(
            "Compare policies for a 30% donor drop next week.",
            metadata=_metadata_payload(),
            config=_simulation_config(),
        )

        self.assertEqual(request.mode, "compare")
        self.assertEqual(
            request.compare_policy_keys,
            [
                "dreamerv3_official",
                "ppo_continuous",
                "sac_continuous",
                "iql_offline",
                "cql_offline",
            ],
        )
        self.assertEqual(request.custom_scenario_payload["donor_show_factor"], 0.7)

    def test_build_simulation_request_accepts_implicit_percent_for_generic_demand(self):
        request = build_simulation_request(
            "What would happen if a sudden emergency increased demand by 20?",
            metadata=_metadata_payload(),
            config=_simulation_config(),
        )

        self.assertEqual(request.mode, "forecast")
        self.assertEqual(request.scenario_kind, "custom")
        self.assertEqual(request.selected_policy_key, "baseline")
        self.assertEqual(request.custom_scenario_payload["demand_surge_factor"], 1.2)

    def test_build_simulation_request_uses_studio_field_catalog_for_explicit_overrides(self):
        request = build_simulation_request(
            (
                "What would happen if we set the initial inventory buffer to 4 days, "
                "reserve target to 6 days, forecast refresh interval to 12 hours, "
                "congestion delay factor to 0.15, and episode budget to 5000?"
            ),
            metadata=_metadata_payload(),
            config=_simulation_config(),
        )

        self.assertEqual(request.mode, "forecast")
        self.assertEqual(request.scenario_kind, "custom")
        self.assertEqual(request.custom_scenario_payload["initial_inventory_days"], 4.0)
        self.assertEqual(request.custom_scenario_payload["reserve_target_days"], 6.0)
        self.assertEqual(
            request.custom_scenario_payload["demand_forecast_interval_h"], 12.0
        )
        self.assertEqual(
            request.custom_scenario_payload["congestion_delay_factor"], 0.15
        )
        self.assertEqual(request.custom_scenario_payload["episode_budget"], 5000.0)

    def test_build_simulation_request_defaults_forecast_to_baseline_policy(self):
        request = build_simulation_request(
            "What would happen if blood donors dropped by 50% in the next week?",
            metadata=_metadata_payload(),
            config=_simulation_config(),
        )

        self.assertEqual(request.mode, "forecast")
        self.assertEqual(request.selected_policy_key, "baseline")

    def test_execute_simulation_query_runs_custom_recommendation_flow(self):
        calls: list[tuple[str, str, dict | None]] = []

        def fake_http_json_request(*, method, url, params=None, body=None, headers=None, timeout_s=10.0):
            del params, headers, timeout_s
            calls.append((method, url, body))
            if url.endswith("/api/meta"):
                return {"status_code": 200, "data": _metadata_payload()}
            if url.endswith("/api/custom-scenarios"):
                return {
                    "status_code": 200,
                    "data": {
                        "scenario": {
                            "key": "orch-donor-drop",
                            "name": "Orchestrator donor drop",
                            "description": "Custom donor drop scenario.",
                        }
                    },
                }
            if url.endswith("/api/evaluations/run"):
                return {
                    "status_code": 200,
                    "data": {
                        "scenario": {
                            "key": "orch-donor-drop",
                            "name": "Orchestrator donor drop",
                            "description": "Custom donor drop scenario.",
                        },
                        "strategy": {
                            "key": "dreamerv3_official",
                            "name": "Official DreamerV3",
                            "description": "Adaptive DreamerV3 policy.",
                        },
                        "dreamerv3_run": {"key": "pink_demo_run"},
                        "hours": 168,
                        "summary": {
                            "shortage_rate": 12.5,
                            "service_rate": 87.5,
                            "total_shortage": 125,
                            "total_net_requested": 1000,
                            "total_transfused": 875,
                            "budget_spent": 2100.0,
                            "budget_remaining": 4400.0,
                            "episode_score": 380.0,
                            "reward_total": 420.0,
                            "exact_match_rate": 92.0,
                            "compatible_substitution_rate": 6.0,
                            "active_action_keys": ["campaign", "emergency_share"],
                            "shortage_by_component": {
                                "RBC": 80,
                                "PLATELETS": 30,
                                "PLASMA": 15,
                            },
                            "shortage_by_hospital": {
                                "North Hub": 60,
                                "South Hub": 40,
                            },
                        },
                        "inventory_timeline": [
                            {"hour": 24, "RBC": 100, "PLATELETS": 40, "PLASMA": 50, "shortages": 0},
                            {"hour": 72, "RBC": 30, "PLATELETS": 12, "PLASMA": 20, "shortages": 20},
                            {"hour": 168, "RBC": 10, "PLATELETS": 8, "PLASMA": 15, "shortages": 125},
                        ],
                    },
                }
            raise AssertionError(f"Unexpected request: {method} {url}")

        with patch("ml.core.simulation_tool.http_json_request", side_effect=fake_http_json_request):
            result = execute_simulation_query(
                "What should we do if blood donors dropped by 50% in the next week?",
                config=_simulation_config(),
            )

        self.assertTrue(result["backend_success"])
        self.assertEqual(result["mode"], "recommend")
        normalized = result["normalized_result"]
        self.assertEqual(normalized["policy"]["key"], "dreamerv3_official")
        self.assertEqual(normalized["metrics"]["shortage_rate"], 12.5)
        self.assertEqual(
            [row["key"] for row in normalized["recommended_actions"]],
            ["campaign", "emergency_share"],
        )
        self.assertIn("Official DreamerV3", result["answer"])
        self.assertIn("shortage rate 12.5%", result["answer"])

        custom_call = next(call for call in calls if call[1].endswith("/api/custom-scenarios"))
        self.assertEqual(custom_call[2]["donor_show_factor"], 0.5)
        self.assertEqual(custom_call[2]["recommended_strategy_key"], "dreamerv3_official")
        run_call = next(call for call in calls if call[1].endswith("/api/evaluations/run"))
        self.assertEqual(run_call[2]["strategy_key"], "dreamerv3_official")
        self.assertEqual(run_call[2]["dreamerv3_run_key"], "pink_demo_run")

    def test_execute_simulation_query_compare_flow_normalizes_best_policy(self):
        calls: list[tuple[str, str, dict | None]] = []

        def fake_http_json_request(*, method, url, params=None, body=None, headers=None, timeout_s=10.0):
            del params, headers, timeout_s
            calls.append((method, url, body))
            if url.endswith("/api/meta"):
                return {"status_code": 200, "data": _metadata_payload()}
            if url.endswith("/api/custom-scenarios"):
                return {
                    "status_code": 200,
                    "data": {
                        "scenario": {
                            "key": "orch-compare-drop",
                            "name": "Compare donor drop",
                            "description": "Custom donor drop scenario.",
                        }
                    },
                }
            if url.endswith("/api/evaluations/compare"):
                return {
                    "status_code": 200,
                    "data": {
                        "summary_rows": [
                            {
                                "strategy": "dreamerv3_official",
                                "strategy_name": "Official DreamerV3",
                                "scenario": "orch-compare-drop",
                                "scenario_name": "Compare donor drop",
                                "shortage_rate_mean": 8.0,
                                "service_rate_mean": 92.0,
                                "total_shortage_mean": 80,
                                "budget_spent_mean": 2500.0,
                                "episode_score_mean": 450.0,
                                "reward_total_mean": 500.0,
                                "exact_match_rate_mean": 94.0,
                                "compatible_substitution_rate_mean": 4.0,
                            },
                            {
                                "strategy": "ppo_continuous",
                                "strategy_name": "PPO Continuous",
                                "scenario": "orch-compare-drop",
                                "scenario_name": "Compare donor drop",
                                "shortage_rate_mean": 11.0,
                                "service_rate_mean": 89.0,
                                "total_shortage_mean": 110,
                                "budget_spent_mean": 2600.0,
                                "episode_score_mean": 390.0,
                                "reward_total_mean": 430.0,
                                "exact_match_rate_mean": 91.0,
                                "compatible_substitution_rate_mean": 5.0,
                            },
                            {
                                "strategy": "sac_continuous",
                                "strategy_name": "SAC Continuous",
                                "scenario": "orch-compare-drop",
                                "scenario_name": "Compare donor drop",
                                "shortage_rate_mean": 13.0,
                                "service_rate_mean": 87.0,
                                "total_shortage_mean": 130,
                                "budget_spent_mean": 2550.0,
                                "episode_score_mean": 360.0,
                                "reward_total_mean": 410.0,
                                "exact_match_rate_mean": 90.0,
                                "compatible_substitution_rate_mean": 6.0,
                            },
                        ],
                        "best_by_scenario": [
                            {
                                "scenario": "orch-compare-drop",
                                "scenario_name": "Compare donor drop",
                                "strategy": "dreamerv3_official",
                                "strategy_name": "Official DreamerV3",
                                "shortage_rate_mean": 8.0,
                                "episode_score_mean": 450.0,
                                "reward_total_mean": 500.0,
                            }
                        ],
                        "dreamerv3_run": {"key": "pink_demo_run"},
                    },
                }
            raise AssertionError(f"Unexpected request: {method} {url}")

        with patch("ml.core.simulation_tool.http_json_request", side_effect=fake_http_json_request):
            result = execute_simulation_query(
                "Compare DreamerV3, PPO, and SAC under a 40% donor drop next week.",
                config=_simulation_config(),
            )

        self.assertTrue(result["backend_success"])
        self.assertEqual(result["mode"], "compare")
        normalized = result["normalized_result"]
        self.assertEqual(normalized["best_policy"]["policy_key"], "dreamerv3_official")
        self.assertEqual(len(normalized["policies"]), 3)
        self.assertIn("Best performer: Official DreamerV3", result["answer"])

        compare_call = next(call for call in calls if call[1].endswith("/api/evaluations/compare"))
        self.assertEqual(
            compare_call[2]["strategy_keys"],
            ["dreamerv3_official", "ppo_continuous", "sac_continuous"],
        )
        self.assertEqual(compare_call[2]["dreamerv3_run_key"], "pink_demo_run")

    def test_execute_simulation_query_handles_backend_failure_gracefully(self):
        def fake_http_json_request(*, method, url, params=None, body=None, headers=None, timeout_s=10.0):
            del params, headers, timeout_s
            if url.endswith("/api/meta"):
                return {"status_code": 200, "data": _metadata_payload()}
            raise RuntimeError("Request failed: timeout")

        with patch("ml.core.simulation_tool.http_json_request", side_effect=fake_http_json_request):
            result = execute_simulation_query(
                "What would happen if blood donors dropped by 50% in the next week?",
                config=_simulation_config(),
            )

        self.assertFalse(result["backend_success"])
        self.assertIn("could not complete the run", result["answer"])
        self.assertEqual(result["mode"], "forecast")

    def test_execute_simulation_query_optionally_compares_before_recommending(self):
        calls: list[tuple[str, str, dict | None]] = []
        config = SimulationToolConfig(
            base_url="http://simulation.local",
            timeout_s=15.0,
            default_policy="dreamerv3_official",
            compare_on_recommend=True,
            default_compare_policies=[
                "dreamerv3_official",
                "ppo_continuous",
                "sac_continuous",
            ],
        )

        def fake_http_json_request(*, method, url, params=None, body=None, headers=None, timeout_s=10.0):
            del params, headers, timeout_s
            calls.append((method, url, body))
            if url.endswith("/api/meta"):
                return {"status_code": 200, "data": _metadata_payload()}
            if url.endswith("/api/custom-scenarios"):
                return {
                    "status_code": 200,
                    "data": {
                        "scenario": {
                            "key": "orch-recommend-compare",
                            "name": "Recommend compare",
                            "description": "Custom donor drop scenario.",
                        }
                    },
                }
            if url.endswith("/api/evaluations/compare"):
                return {
                    "status_code": 200,
                    "data": {
                        "summary_rows": [
                            {
                                "strategy": "dreamerv3_official",
                                "strategy_name": "Official DreamerV3",
                                "scenario": "orch-recommend-compare",
                                "scenario_name": "Recommend compare",
                                "shortage_rate_mean": 7.0,
                                "service_rate_mean": 93.0,
                                "total_shortage_mean": 70,
                                "budget_spent_mean": 2400.0,
                                "episode_score_mean": 470.0,
                                "reward_total_mean": 520.0,
                                "exact_match_rate_mean": 94.0,
                                "compatible_substitution_rate_mean": 4.0,
                            },
                            {
                                "strategy": "ppo_continuous",
                                "strategy_name": "PPO Continuous",
                                "scenario": "orch-recommend-compare",
                                "scenario_name": "Recommend compare",
                                "shortage_rate_mean": 10.0,
                                "service_rate_mean": 90.0,
                                "total_shortage_mean": 100,
                                "budget_spent_mean": 2500.0,
                                "episode_score_mean": 400.0,
                                "reward_total_mean": 430.0,
                                "exact_match_rate_mean": 92.0,
                                "compatible_substitution_rate_mean": 5.0,
                            },
                        ]
                    },
                }
            if url.endswith("/api/evaluations/run"):
                return {
                    "status_code": 200,
                    "data": {
                        "scenario": {
                            "key": "orch-recommend-compare",
                            "name": "Recommend compare",
                            "description": "Custom donor drop scenario.",
                        },
                        "strategy": {
                            "key": "dreamerv3_official",
                            "name": "Official DreamerV3",
                            "description": "Adaptive DreamerV3 policy.",
                        },
                        "hours": 168,
                        "summary": {
                            "shortage_rate": 7.5,
                            "service_rate": 92.5,
                            "total_shortage": 75,
                            "total_net_requested": 1000,
                            "total_transfused": 925,
                            "budget_spent": 2200.0,
                            "budget_remaining": 4300.0,
                            "episode_score": 460.0,
                            "reward_total": 510.0,
                            "exact_match_rate": 94.0,
                            "compatible_substitution_rate": 4.0,
                            "active_action_keys": ["campaign"],
                            "shortage_by_component": {"RBC": 50},
                            "shortage_by_hospital": {"North Hub": 30},
                        },
                        "inventory_timeline": [],
                    },
                }
            raise AssertionError(f"Unexpected request: {method} {url}")

        with patch("ml.core.simulation_tool.http_json_request", side_effect=fake_http_json_request):
            result = execute_simulation_query(
                "What should we do if blood donors dropped by 50% in the next week?",
                config=config,
            )

        self.assertTrue(result["backend_success"])
        normalized = result["normalized_result"]
        self.assertIsNotNone(normalized["policy_comparison"])
        self.assertEqual(
            normalized["policy_comparison"]["best_policy"]["policy_key"],
            "dreamerv3_official",
        )
        self.assertIn("optional policy comparison", result["answer"])
        compare_call = next(call for call in calls if call[1].endswith("/api/evaluations/compare"))
        self.assertEqual(
            compare_call[2]["strategy_keys"],
            ["dreamerv3_official", "ppo_continuous", "sac_continuous"],
        )


if __name__ == "__main__":
    unittest.main()
