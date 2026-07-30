from __future__ import annotations

import argparse
import sys
from pathlib import Path


SIMULATOR_DIR = Path(__file__).resolve().parents[1] / "simulator"
if str(SIMULATOR_DIR) not in sys.path:
    sys.path.insert(0, str(SIMULATOR_DIR))

from train_dreamer import build_official_argv, parse_args


def test_parse_args_defaults_to_training_scenarios(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["train_dreamer.py"])

    args = parse_args()

    assert args.scenario == "train"


def test_build_official_argv_forwards_training_split_and_default_steps(tmp_path):
    args = argparse.Namespace(
        steps=None,
        timesteps=None,
        scenario="train",
        logdir=str(tmp_path / "dreamer_run"),
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
        script="train",
    )

    argv = build_official_argv(args)

    assert "--scenario" in argv
    assert argv[argv.index("--scenario") + 1] == "train"
    assert argv[argv.index("--steps") + 1] == "50000"
