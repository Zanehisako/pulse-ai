from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np

SIMULATOR_DIR = Path(__file__).resolve().parents[1] / "simulator"
if str(SIMULATOR_DIR) not in sys.path:
    sys.path.insert(0, str(SIMULATOR_DIR))

import evaluate_strategies as evaluate_strategies_module
from evaluate_strategies import (
    aggregate_rows,
    build_plot_export_config,
    collect_pareto_rows,
    pareto_frontier_mask,
    pivot_metric,
    resolve_output_prefix,
    strategy_keys_from_args,
)


def test_aggregate_rows_computes_mean_and_std_per_strategy_scenario():
    rows = [
        {
            "strategy": "baseline",
            "strategy_name": "Baseline",
            "scenario": "baseline",
            "scenario_name": "Normal Operations",
            "seed": 100,
            "episode_score": 3.0,
            "reward_total": 10.0,
            "shortage_rate": 5.0,
            "base_shortage_rate": 4.0,
            "total_shortage": 3,
            "total_transfused": 90,
            "total_donated": 100,
            "total_expired": 2,
            "total_external_units": 1,
            "total_conserved": 0,
            "transfer_units": 4,
            "active_action_cost": 0.0,
            "active_action_count": 0,
            "controller_decisions": 0,
        },
        {
            "strategy": "baseline",
            "strategy_name": "Baseline",
            "scenario": "baseline",
            "scenario_name": "Normal Operations",
            "seed": 101,
            "episode_score": 5.0,
            "reward_total": 14.0,
            "shortage_rate": 7.0,
            "base_shortage_rate": 6.0,
            "total_shortage": 5,
            "total_transfused": 88,
            "total_donated": 96,
            "total_expired": 3,
            "total_external_units": 2,
            "total_conserved": 1,
            "transfer_units": 5,
            "active_action_cost": 0.0,
            "active_action_count": 0,
            "controller_decisions": 0,
        },
    ]

    summary_rows = aggregate_rows(rows)

    assert len(summary_rows) == 1
    row = summary_rows[0]
    assert row["strategy"] == "baseline"
    assert row["scenario"] == "baseline"
    assert row["strategy_name"] == "Baseline"
    assert row["scenario_name"] == "Normal Operations"
    assert row["runs"] == 2.0
    assert row["episode_score_mean"] == 4.0
    assert row["reward_total_mean"] == 12.0
    assert row["shortage_rate_mean"] == 6.0
    assert row["total_donated_mean"] == 98.0
    assert np.isclose(row["reward_total_std"], 2.0)


def test_pivot_metric_creates_matrix_in_requested_order():
    summary_rows = [
        {
            "strategy": "baseline",
            "strategy_name": "Baseline",
            "scenario": "baseline",
            "scenario_name": "Normal Operations",
            "reward_total_mean": 12.0,
        },
        {
            "strategy": "dreamerv3_official",
            "strategy_name": "Official DreamerV3",
            "scenario": "baseline",
            "scenario_name": "Normal Operations",
            "reward_total_mean": 20.0,
        },
        {
            "strategy": "baseline",
            "strategy_name": "Baseline",
            "scenario": "donor_decrease",
            "scenario_name": "Donor Decline Crisis",
            "reward_total_mean": 8.0,
        },
    ]

    matrix = pivot_metric(
        summary_rows,
        ["baseline", "donor_decrease"],
        ["dreamerv3_official", "baseline"],
        "reward_total_mean",
    )

    assert matrix.shape == (2, 2)
    assert matrix[0, 0] == 20.0
    assert np.isnan(matrix[0, 1])
    assert matrix[1, 0] == 12.0
    assert matrix[1, 1] == 8.0


def test_extract_metrics_counts_continuous_controller_decisions(monkeypatch):
    fake_summary = {
        "episode_score": 17.5,
        "shortage_rate": 12.0,
        "base_shortage_rate": 10.0,
        "total_shortage": 50,
        "total_transfused": 450,
        "total_donated": 300,
        "total_collected": 300,
        "total_released": 280,
        "total_rejected": 20,
        "total_lab_rejected": 2,
        "total_no_show": 30,
        "rejection_rate": 6.25,
        "total_expired": 5,
        "total_external_units": 40,
        "total_conserved": 12,
        "total_base_requested": 500,
        "total_net_requested": 488,
        "total_exact_match_units": 360,
        "total_compatible_substitution_units": 80,
        "total_incompatible_fulfillment_units": 10,
        "active_action_cost": 18.0,
        "active_action_keys": ["rapid_courier"],
        "controller_decisions": 2,
        "avg_travel_min": 25.0,
        "avg_wait_min": 0.5,
        "budget_total": 7000.0,
        "budget_remaining": 5000.0,
        "budget_spent": 2000.0,
        "budget_exhausted_hours": 0.0,
        "forecast_pressure": 1.1,
        "recent_priority_pressure": 0.2,
        "transport_congestion": 0.05,
        "priority_weighted_shortage": 40.0,
        "priority_weighted_requested": 400.0,
        "shortage_by_component": {"RBC": 30, "PLATELETS": 15, "PLASMA": 5},
        "reward": SimpleNamespace(total=-300.0),
    }
    monkeypatch.setattr(
        evaluate_strategies_module,
        "summarize_state",
        lambda state: fake_summary,
    )
    state = SimpleNamespace(
        total_transfer_units=25,
        policy_log=[
            {"kind": "decision"},
            {"kind": "continuous_decision"},
            {"kind": "activation"},
        ],
    )

    row = evaluate_strategies_module.extract_metrics(
        "dreamerv3_official",
        "baseline",
        100,
        state,
    )

    assert row["episode_score"] == 17.5
    assert row["controller_decisions"] == 2
    assert row["reward_total"] == -300.0


def test_run_episode_routes_dreamerv4_through_dreamer_controller(monkeypatch):
    captured: dict[str, object] = {}

    def fake_run_scenario(params, G, north, south, east, west, **kwargs):
        captured["strategy_key"] = params.strategy_key
        captured["kwargs"] = kwargs
        return SimpleNamespace(
            env=SimpleNamespace(now=params.sim_hours),
            params=params,
        )

    monkeypatch.setattr(evaluate_strategies_module, "run_scenario", fake_run_scenario)

    agent = object()
    state = evaluate_strategies_module.run_episode(
        "dreamerv4",
        "baseline",
        (None, 46.90, 46.70, -71.10, -71.35),
        seed=17,
        dreamer_agents={"dreamerv4": agent},
        dreamer_checkpoints={"dreamerv4": "/tmp/dreamerv4-ckpt"},
    )

    kwargs = captured["kwargs"]
    assert captured["strategy_key"] == "dreamerv4"
    assert kwargs["official_dreamerv3_agent"] is agent
    assert kwargs["official_dreamerv3_checkpoint"] is None
    assert state.params.strategy_key == "dreamerv4"


def test_strategy_keys_from_args_includes_new_rl_strategies_by_default():
    args = SimpleNamespace(strategy=None)

    strategy_keys = strategy_keys_from_args(args)

    assert strategy_keys == [
        "ppo_continuous",
        "sac_continuous",
        "iql_offline",
        "cql_offline",
        "dreamerv3_official",
    ]
    assert "baseline" not in strategy_keys
    assert "ppo_shortage_minimizer" not in strategy_keys
    assert "sac_continuous" in strategy_keys
    assert "iql_offline" in strategy_keys
    assert "cql_offline" in strategy_keys
    assert "dreamerv4" not in strategy_keys


def test_default_dreamerv3_eval_target_points_to_run8():
    assert (
        evaluate_strategies_module.DEFAULT_DREAMERV3_EVAL_RUN.as_posix().endswith(
            "dreamerv3_runs/m1_continuous_run8_kpi_aligned"
        )
    )


def test_build_plot_export_config_uses_thesis_defaults():
    args = SimpleNamespace(thesis_ready=True, export_format=None, dpi=None)

    config = build_plot_export_config(args)

    assert config.thesis_ready is True
    assert config.formats == ("png", "pdf", "svg")
    assert config.dpi == 300


def test_resolve_output_prefix_places_bare_names_under_simulator_results():
    assert resolve_output_prefix("strategy_eval_run4") == (
        SIMULATOR_DIR / "results" / "strategy_eval_run4"
    )


def test_resolve_output_prefix_respects_explicit_relative_paths():
    explicit = Path("custom_outputs") / "strategy_eval_run4"
    assert resolve_output_prefix(explicit) == explicit


def test_pareto_frontier_mask_marks_only_non_dominated_points_for_minimization():
    points = np.asarray(
        [
            [0.0, 3.0],
            [1.0, 1.0],
            [2.0, 2.0],
            [3.0, 0.0],
        ],
        dtype=np.float32,
    )

    mask = pareto_frontier_mask(points, ("min", "min"))

    assert mask.tolist() == [True, True, False, True]


def test_pareto_frontier_mask_supports_mixed_max_and_min_goals():
    points = np.asarray(
        [
            [10.0, 5.0],
            [8.0, 4.0],
            [12.0, 7.0],
            [11.0, 4.0],
        ],
        dtype=np.float32,
    )

    mask = pareto_frontier_mask(points, ("max", "min"))

    assert mask.tolist() == [False, False, True, True]


def test_pareto_frontier_mask_supports_three_dimensional_points():
    points = np.asarray(
        [
            [1.0, 1.0, 1.0],
            [2.0, 1.0, 1.0],
            [1.0, 2.0, 2.0],
            [0.0, 3.0, 3.0],
        ],
        dtype=np.float32,
    )

    mask = pareto_frontier_mask(points, ("min", "min", "min"))

    assert mask.tolist() == [True, False, False, True]


def test_collect_pareto_rows_averages_metrics_across_selected_scenarios():
    summary_rows = [
        {
            "strategy": "baseline",
            "strategy_name": "Baseline",
            "scenario": "baseline",
            "scenario_name": "Normal Operations",
            "active_action_cost_mean": 0.0,
            "shortage_rate_mean": 8.0,
            "total_expired_mean": 9.0,
        },
        {
            "strategy": "baseline",
            "strategy_name": "Baseline",
            "scenario": "donor_decrease",
            "scenario_name": "Donor Decline Crisis",
            "active_action_cost_mean": 2.0,
            "shortage_rate_mean": 12.0,
            "total_expired_mean": 11.0,
        },
        {
            "strategy": "dreamerv3_official",
            "strategy_name": "Official DreamerV3",
            "scenario": "baseline",
            "scenario_name": "Normal Operations",
            "active_action_cost_mean": 200.0,
            "shortage_rate_mean": 2.0,
            "total_expired_mean": 4.0,
        },
        {
            "strategy": "dreamerv3_official",
            "strategy_name": "Official DreamerV3",
            "scenario": "donor_decrease",
            "scenario_name": "Donor Decline Crisis",
            "active_action_cost_mean": 300.0,
            "shortage_rate_mean": 4.0,
            "total_expired_mean": 6.0,
        },
    ]

    rows = collect_pareto_rows(
        summary_rows,
        ["baseline", "donor_decrease"],
        ["dreamerv3_official", "baseline"],
        scenario_key=None,
        x_metric="active_action_cost_mean",
        y_metric="shortage_rate_mean",
        z_metric="total_expired_mean",
    )

    assert [row["strategy"] for row in rows] == ["dreamerv3_official", "baseline"]
    assert rows[0]["active_action_cost_mean"] == 250.0
    assert rows[0]["shortage_rate_mean"] == 3.0
    assert rows[0]["total_expired_mean"] == 5.0
    assert rows[1]["active_action_cost_mean"] == 1.0
    assert rows[1]["shortage_rate_mean"] == 10.0
    assert rows[1]["total_expired_mean"] == 10.0
