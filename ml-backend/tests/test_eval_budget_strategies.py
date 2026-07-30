"""Tests for the budget-aware evaluation framework.

Covers:
  - FastBloodBudgetEnv  (init, step, budget tracking, early termination, obs)
  - eval_budget_strategies  (budget grid parsing, aggregation, CSV I/O)
  - train_ppo_budget  (argument parsing)
"""

from __future__ import annotations

import csv
import sys
from argparse import Namespace
from pathlib import Path

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Put the simulator package on the import path (same pattern as sibling tests)
# ---------------------------------------------------------------------------
SIMULATOR_DIR = Path(__file__).resolve().parents[1] / "simulator"
if str(SIMULATOR_DIR) not in sys.path:
    sys.path.insert(0, str(SIMULATOR_DIR))

from eval_budget_strategies import (
    DEFAULT_BUDGET_GRID,
    aggregate_across_scenarios,
    aggregate_rows,
    budget_grid_from_args,
    save_csv,
    strategy_keys_from_args,
)
from eval_metrics import DETAIL_METRICS
from fast_env_budget import FastBloodBudgetEnv

# ═══════════════════════════════════════════════════════════════════════════
# FastBloodBudgetEnv
# ═══════════════════════════════════════════════════════════════════════════


class TestFastBudgetEnvInit:
    """Test 1 — basic construction and space shapes."""

    def test_action_space_is_discrete(self):
        env = FastBloodBudgetEnv(n_envs=1, episode_steps=10, episode_budget=5000.0)
        # Discrete space: N = len(ACTION_KEYS) + 1  (NOOP at 0)
        assert hasattr(env.action_space, "n")
        assert env.action_space.n >= 2  # at least NOOP + 1 real action

    def test_observation_space_shape(self):
        env = FastBloodBudgetEnv(n_envs=1, episode_steps=10, episode_budget=5000.0)
        assert env.observation_space.shape == (16,)

    def test_episode_budget_stored(self):
        env = FastBloodBudgetEnv(n_envs=1, episode_steps=10, episode_budget=5000.0)
        assert env.episode_budget == 5000.0

    def test_budget_remaining_after_init(self):
        env = FastBloodBudgetEnv(n_envs=1, episode_steps=10, episode_budget=5000.0)
        assert env.budget_remaining[0] == pytest.approx(5000.0)


class TestFastBudgetEnvStepTracksBudget:
    """Test 2 — a non-NOOP action should cost something."""

    def test_budget_decreases_after_non_noop(self):
        env = FastBloodBudgetEnv(n_envs=1, episode_steps=20, episode_budget=5000.0)
        env.reset(seed=42)

        # Action 1 is the first real (non-NOOP) action
        env.step(1)

        assert env.budget_spent[0] > 0.0, "Non-NOOP action should incur cost"
        assert env.budget_remaining[0] < 5000.0, "Budget should decrease"


class TestFastBudgetEnvNoopNoCost:
    """Test 3 — NOOP (action 0) should be free."""

    def test_noop_leaves_budget_intact(self):
        budget = 5000.0
        env = FastBloodBudgetEnv(n_envs=1, episode_steps=20, episode_budget=budget)
        env.reset(seed=42)

        env.step(0)

        assert env.budget_spent[0] == pytest.approx(0.0)
        assert env.budget_remaining[0] == pytest.approx(budget)


class TestFastBudgetEnvEarlyTermination:
    """Test 4 — a tiny budget should cause the episode to end early."""

    def test_episode_ends_before_max_steps(self):
        max_steps = 20
        env = FastBloodBudgetEnv(n_envs=1, episode_steps=max_steps, episode_budget=10.0)
        env.reset(seed=42)

        done = False
        steps_taken = 0
        # Use the highest action index to maximize the chance of a costly action
        expensive_action = env.action_space.n - 1

        for _ in range(max_steps):
            _obs, _reward, done, _trunc, _info = env.step(expensive_action)
            steps_taken += 1
            if done:
                break

        assert done, "Episode should have terminated"
        assert steps_taken < max_steps, (
            f"Expected early termination but ran all {max_steps} steps"
        )


class TestFastBudgetEnvObsBudgetSignals:
    """Test 5 — observation slots 11 (active_cost_ratio) and 15 (budget_pressure)."""

    def test_budget_signals_present_in_obs(self):
        env = FastBloodBudgetEnv(n_envs=1, episode_steps=20, episode_budget=5000.0)
        env.reset(seed=42)

        # Take a non-NOOP action so some budget is spent
        obs, *_ = env.step(1)

        # obs[11] = active_cost_ratio = budget_spent / episode_budget
        assert obs[11] > 0.0, "active_cost_ratio should be > 0 after spending"

        # obs[15] = budget_pressure
        assert obs[15] > 0.0, "budget_pressure should be > 0 after spending"

    def test_obs_values_are_clipped_0_1(self):
        env = FastBloodBudgetEnv(n_envs=1, episode_steps=20, episode_budget=5000.0)
        obs, _ = env.reset(seed=42)
        assert np.all(obs >= 0.0) and np.all(obs <= 1.0)

        obs2, *_ = env.step(1)
        assert np.all(obs2 >= 0.0) and np.all(obs2 <= 1.0)


# ═══════════════════════════════════════════════════════════════════════════
# eval_budget_strategies — budget grid parsing
# ═══════════════════════════════════════════════════════════════════════════


class TestBudgetGridParsing:
    """Test 6 — explicit budget string is parsed correctly."""

    def test_parses_comma_separated_budgets(self):
        args = Namespace(budgets="3000,5000,7000")
        result = budget_grid_from_args(args)
        assert result == [3000.0, 5000.0, 7000.0]

    def test_result_is_sorted(self):
        args = Namespace(budgets="7000,3000,5000")
        result = budget_grid_from_args(args)
        assert result == [3000.0, 5000.0, 7000.0]

    def test_handles_whitespace(self):
        args = Namespace(budgets=" 3000 , 5000 , 7000 ")
        result = budget_grid_from_args(args)
        assert result == [3000.0, 5000.0, 7000.0]


class TestBudgetGridDefault:
    """Test 7 — None falls back to DEFAULT_BUDGET_GRID."""

    def test_returns_default_when_none(self):
        args = Namespace(budgets=None)
        result = budget_grid_from_args(args)
        assert result == list(DEFAULT_BUDGET_GRID)

    def test_default_grid_is_sorted(self):
        assert DEFAULT_BUDGET_GRID == sorted(DEFAULT_BUDGET_GRID)

    def test_default_grid_has_multiple_levels(self):
        assert len(DEFAULT_BUDGET_GRID) >= 3


class TestDefaultStrategySet:
    def test_includes_dreamerv4(self):
        args = Namespace(strategy=None)

        strategy_keys = strategy_keys_from_args(args)

        assert "dreamerv4" in strategy_keys


# ═══════════════════════════════════════════════════════════════════════════
# eval_budget_strategies — aggregation
# ═══════════════════════════════════════════════════════════════════════════


def _make_detail_row(
    strategy: str,
    scenario: str,
    budget_level: float,
    seed: int,
    *,
    shortage_rate: float = 5.0,
    reward_total: float = -100.0,
    episode_score: float = 10.0,
    active_action_cost: float = 200.0,
    budget_spent: float = 1000.0,
) -> dict:
    """Helper: build a minimal detail row with zeros for most metrics."""
    row: dict = {
        "strategy": strategy,
        "strategy_name": strategy.replace("_", " ").title(),
        "scenario": scenario,
        "scenario_name": scenario.replace("_", " ").title(),
        "budget_level": budget_level,
        "seed": seed,
    }
    # Fill every DETAIL_METRIC so aggregate_rows won't skip anything
    for m in DETAIL_METRICS:
        row.setdefault(m, 0.0)

    # Override the ones we care about
    row["shortage_rate"] = shortage_rate
    row["reward_total"] = reward_total
    row["episode_score"] = episode_score
    row["active_action_cost"] = active_action_cost
    row["budget_spent"] = budget_spent
    return row


class TestAggregateRows:
    """Test 8 — aggregate_rows computes mean / std per (strategy, scenario, budget)."""

    def test_two_seeds_produce_one_summary_row(self):
        rows = [
            _make_detail_row("baseline", "baseline", 5000.0, 1, shortage_rate=4.0),
            _make_detail_row("baseline", "baseline", 5000.0, 2, shortage_rate=6.0),
        ]
        summary = aggregate_rows(rows)
        assert len(summary) == 1

    def test_mean_is_correct(self):
        rows = [
            _make_detail_row(
                "baseline",
                "baseline",
                5000.0,
                1,
                shortage_rate=4.0,
                reward_total=-80.0,
            ),
            _make_detail_row(
                "baseline",
                "baseline",
                5000.0,
                2,
                shortage_rate=6.0,
                reward_total=-120.0,
            ),
        ]
        summary = aggregate_rows(rows)
        row = summary[0]

        assert row["shortage_rate_mean"] == pytest.approx(5.0)
        assert row["reward_total_mean"] == pytest.approx(-100.0)

    def test_std_is_correct(self):
        rows = [
            _make_detail_row("baseline", "baseline", 5000.0, 1, shortage_rate=4.0),
            _make_detail_row("baseline", "baseline", 5000.0, 2, shortage_rate=6.0),
        ]
        summary = aggregate_rows(rows)
        row = summary[0]

        expected_std = float(np.std([4.0, 6.0]))
        assert row["shortage_rate_std"] == pytest.approx(expected_std, abs=1e-4)

    def test_runs_count(self):
        rows = [
            _make_detail_row("baseline", "baseline", 5000.0, 1),
            _make_detail_row("baseline", "baseline", 5000.0, 2),
            _make_detail_row("baseline", "baseline", 5000.0, 3),
        ]
        summary = aggregate_rows(rows)
        assert summary[0]["runs"] == 3.0

    def test_preserves_group_key_fields(self):
        rows = [
            _make_detail_row("ppo_trained", "donor_decrease", 7000.0, 1),
        ]
        summary = aggregate_rows(rows)
        row = summary[0]
        assert row["strategy"] == "ppo_trained"
        assert row["scenario"] == "donor_decrease"
        assert row["budget_level"] == 7000.0

    def test_different_groups_produce_separate_rows(self):
        rows = [
            _make_detail_row("baseline", "baseline", 5000.0, 1),
            _make_detail_row("baseline", "baseline", 7000.0, 1),
            _make_detail_row("ppo_trained", "baseline", 5000.0, 1),
        ]
        summary = aggregate_rows(rows)
        assert len(summary) == 3


class TestAggregateAcrossScenarios:
    """Test 9 — aggregate_across_scenarios averages over scenarios."""

    def test_two_scenarios_produce_one_row(self):
        summary_rows = [
            {
                "strategy": "baseline",
                "strategy_name": "Baseline",
                "scenario": "baseline",
                "scenario_name": "Normal Ops",
                "budget_level": 5000.0,
                "runs": 2.0,
                "shortage_rate_mean": 4.0,
                "shortage_rate_std": 1.0,
            },
            {
                "strategy": "baseline",
                "strategy_name": "Baseline",
                "scenario": "donor_decrease",
                "scenario_name": "Donor Decline",
                "budget_level": 5000.0,
                "runs": 2.0,
                "shortage_rate_mean": 8.0,
                "shortage_rate_std": 2.0,
            },
        ]
        agg = aggregate_across_scenarios(summary_rows)
        assert len(agg) == 1

    def test_mean_of_means(self):
        summary_rows = [
            {
                "strategy": "baseline",
                "strategy_name": "Baseline",
                "scenario": "baseline",
                "scenario_name": "Normal Ops",
                "budget_level": 5000.0,
                "runs": 2.0,
                "shortage_rate_mean": 4.0,
                "shortage_rate_std": 1.0,
            },
            {
                "strategy": "baseline",
                "strategy_name": "Baseline",
                "scenario": "donor_decrease",
                "scenario_name": "Donor Decline",
                "budget_level": 5000.0,
                "runs": 2.0,
                "shortage_rate_mean": 8.0,
                "shortage_rate_std": 2.0,
            },
        ]
        agg = aggregate_across_scenarios(summary_rows)
        row = agg[0]

        assert row["shortage_rate_mean"] == pytest.approx(6.0)
        assert row["n_scenarios"] == 2.0

    def test_different_budgets_stay_separate(self):
        summary_rows = [
            {
                "strategy": "baseline",
                "strategy_name": "Baseline",
                "scenario": "baseline",
                "scenario_name": "Normal Ops",
                "budget_level": 5000.0,
                "runs": 1.0,
                "shortage_rate_mean": 4.0,
                "shortage_rate_std": 0.0,
            },
            {
                "strategy": "baseline",
                "strategy_name": "Baseline",
                "scenario": "baseline",
                "scenario_name": "Normal Ops",
                "budget_level": 7000.0,
                "runs": 1.0,
                "shortage_rate_mean": 3.0,
                "shortage_rate_std": 0.0,
            },
        ]
        agg = aggregate_across_scenarios(summary_rows)
        assert len(agg) == 2


# ═══════════════════════════════════════════════════════════════════════════
# eval_budget_strategies — CSV I/O
# ═══════════════════════════════════════════════════════════════════════════


class TestSaveCsvRoundtrip:
    """Test 10 — write rows with save_csv, read back, verify contents."""

    def test_roundtrip(self, tmp_path: Path):
        rows = [
            {"strategy": "baseline", "budget_level": 5000.0, "shortage_rate_mean": 4.2},
            {"strategy": "ppo", "budget_level": 5000.0, "shortage_rate_mean": 3.1},
        ]
        csv_path = tmp_path / "test_output.csv"
        save_csv(rows, csv_path)

        assert csv_path.exists()

        with csv_path.open(newline="", encoding="utf-8") as fh:
            reader = csv.DictReader(fh)
            read_rows = list(reader)

        assert len(read_rows) == 2
        assert read_rows[0]["strategy"] == "baseline"
        assert read_rows[1]["strategy"] == "ppo"
        assert float(read_rows[0]["shortage_rate_mean"]) == pytest.approx(4.2)
        assert float(read_rows[1]["shortage_rate_mean"]) == pytest.approx(3.1)

    def test_header_matches_keys(self, tmp_path: Path):
        rows = [{"a": 1, "b": 2, "c": 3}]
        csv_path = tmp_path / "header_test.csv"
        save_csv(rows, csv_path)

        with csv_path.open(newline="", encoding="utf-8") as fh:
            reader = csv.reader(fh)
            header = next(reader)

        assert header == ["a", "b", "c"]

    def test_empty_rows_no_file(self, tmp_path: Path):
        csv_path = tmp_path / "empty.csv"
        save_csv([], csv_path)
        assert not csv_path.exists()

    def test_creates_parent_directories(self, tmp_path: Path):
        csv_path = tmp_path / "deep" / "nested" / "dir" / "output.csv"
        rows = [{"x": 1}]
        save_csv(rows, csv_path)
        assert csv_path.exists()


# ═══════════════════════════════════════════════════════════════════════════
# train_ppo_budget — argument parsing
# ═══════════════════════════════════════════════════════════════════════════


class TestTrainPpoBudgetParseArgs:
    """Test 11 — CLI argument parsing for budget PPO training."""

    def test_default_budget(self, monkeypatch):
        monkeypatch.setattr(sys, "argv", ["train_ppo_budget.py"])
        from train_ppo_budget import parse_args

        args = parse_args()
        assert args.budget == 6500.0

    def test_custom_budget(self, monkeypatch):
        monkeypatch.setattr(sys, "argv", ["train_ppo_budget.py", "--budget", "4000"])
        from train_ppo_budget import parse_args

        args = parse_args()
        assert args.budget == 4000.0

    def test_budget_curriculum_parsing(self, monkeypatch):
        monkeypatch.setattr(
            sys,
            "argv",
            ["train_ppo_budget.py", "--budget-curriculum", "8000,6500,5000,4000"],
        )
        from train_ppo_budget import parse_args

        args = parse_args()
        assert args.budget_curriculum == "8000,6500,5000,4000"

        # Verify it can be split into the expected float values
        budgets = [float(b.strip()) for b in args.budget_curriculum.split(",")]
        assert budgets == [8000.0, 6500.0, 5000.0, 4000.0]

    def test_default_timesteps(self, monkeypatch):
        monkeypatch.setattr(sys, "argv", ["train_ppo_budget.py"])
        from train_ppo_budget import parse_args

        args = parse_args()
        assert args.timesteps == 200_000

    def test_custom_timesteps(self, monkeypatch):
        monkeypatch.setattr(
            sys, "argv", ["train_ppo_budget.py", "--timesteps", "500000"]
        )
        from train_ppo_budget import parse_args

        args = parse_args()
        assert args.timesteps == 500_000

    def test_no_curriculum_by_default(self, monkeypatch):
        monkeypatch.setattr(sys, "argv", ["train_ppo_budget.py"])
        from train_ppo_budget import parse_args

        args = parse_args()
        assert args.budget_curriculum is None

    def test_episode_steps_default(self, monkeypatch):
        monkeypatch.setattr(sys, "argv", ["train_ppo_budget.py"])
        from train_ppo_budget import parse_args

        args = parse_args()
        assert args.episode_steps == 28
