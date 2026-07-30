"""Tests for the budget-aware DreamerV3 training variant."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

SIMULATOR_DIR = Path(__file__).resolve().parents[1] / "simulator"
if str(SIMULATOR_DIR) not in sys.path:
    sys.path.insert(0, str(SIMULATOR_DIR))

from train_dreamer_budget import build_official_argv, parse_args


def test_parse_args_defaults_to_training_scenarios(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["train_dreamer_budget.py"])
    args = parse_args()
    assert args.scenario == "train"


def test_parse_args_default_logdir_contains_budget(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["train_dreamer_budget.py"])
    args = parse_args()
    assert "budget" in args.logdir


def test_build_official_argv_includes_budget_variant_flag(tmp_path):
    args = argparse.Namespace(
        steps=None,
        timesteps=None,
        scenario="train",
        logdir=str(tmp_path / "budget_run"),
        seed=7,
        step_hours=6.0,
        envs=0,
        eval_envs=0,
        eval_eps=3,
        train_ratio=32.0,
        batch_size=16,
        batch_length=64,
        size="size1m",
        log_every=30,
        report_every=120,
        save_every=300,
        jax_platform="auto",
        compute_dtype="auto",
        online_city_graph=False,
        jax_profiler="auto",
        metrics_write_mode="buffered",
        fixed_budget=None,
        script="train",
    )

    argv = build_official_argv(args)

    # The budget-variant flag must always be present
    assert "--budget-variant" in argv
    # Default steps should be forwarded
    assert argv[argv.index("--steps") + 1] == "50000"
    # --fixed-budget should NOT be present when None
    assert "--fixed-budget" not in argv


def test_build_official_argv_includes_fixed_budget_when_set(tmp_path):
    args = argparse.Namespace(
        steps=80_000,
        timesteps=None,
        scenario="baseline",
        logdir=str(tmp_path / "budget_run2"),
        seed=42,
        step_hours=6.0,
        envs=0,
        eval_envs=0,
        eval_eps=3,
        train_ratio=32.0,
        batch_size=16,
        batch_length=64,
        size="size1m",
        log_every=30,
        report_every=120,
        save_every=300,
        jax_platform="auto",
        compute_dtype="auto",
        online_city_graph=False,
        jax_profiler="auto",
        metrics_write_mode="buffered",
        fixed_budget=4500.0,
        script="train",
    )

    argv = build_official_argv(args)

    assert "--budget-variant" in argv
    assert "--fixed-budget" in argv
    assert argv[argv.index("--fixed-budget") + 1] == "4500.0"
    assert argv[argv.index("--steps") + 1] == "80000"
    assert argv[argv.index("--scenario") + 1] == "baseline"


def test_build_official_argv_timesteps_alias(tmp_path):
    """--timesteps should be used when --steps is None."""
    args = argparse.Namespace(
        steps=None,
        timesteps=30_000,
        scenario="train",
        logdir=str(tmp_path / "budget_alias"),
        seed=0,
        step_hours=6.0,
        envs=0,
        eval_envs=0,
        eval_eps=3,
        train_ratio=32.0,
        batch_size=16,
        batch_length=64,
        size="size1m",
        log_every=30,
        report_every=120,
        save_every=300,
        jax_platform="auto",
        compute_dtype="auto",
        online_city_graph=False,
        jax_profiler="auto",
        metrics_write_mode="buffered",
        fixed_budget=None,
        script="train",
    )

    argv = build_official_argv(args)
    assert argv[argv.index("--steps") + 1] == "30000"


def test_calculate_step_reward_budget_exists():
    """Verify the budget reward function is importable from engine."""
    import engine

    assert hasattr(engine, "calculate_step_reward_budget")
    assert callable(engine.calculate_step_reward_budget)


def test_official_dreamerv3_parse_args_budget_flags(monkeypatch):
    """Verify official_dreamerv3.py accepts the new budget CLI flags."""
    from official_dreamerv3 import parse_args as official_parse_args

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "official_dreamerv3.py",
            "--budget-variant",
            "--fixed-budget",
            "5000",
        ],
    )
    args = official_parse_args()
    assert args.budget_variant is True
    assert args.fixed_budget == 5000.0


def test_official_dreamerv3_parse_args_no_budget_by_default(monkeypatch):
    """Without --budget-variant the flag should be False."""
    from official_dreamerv3 import parse_args as official_parse_args

    monkeypatch.setattr(sys, "argv", ["official_dreamerv3.py"])
    args = official_parse_args()
    assert args.budget_variant is False
    assert args.fixed_budget is None
