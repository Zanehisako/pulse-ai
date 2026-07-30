"""
train_dreamer_budget.py — Train the budget-aware DreamerV3 variant.

This is a convenience wrapper around ``official_dreamerv3.py`` that
automatically enables the ``--budget-variant`` flag.  All other arguments
are forwarded unchanged.

Usage
-----
    # Train with default scenario budgets
    python train_dreamer_budget.py --steps 50000 --logdir dreamerv3_runs/budget_run1

    # Train with a fixed budget of 5000 across all scenarios
    python train_dreamer_budget.py --steps 50000 --fixed-budget 5000 \\
        --logdir dreamerv3_runs/budget_tight

    # Full example with all bells and whistles
    python train_dreamer_budget.py \\
        --steps 100000 \\
        --scenario train \\
        --fixed-budget 4500 \\
        --logdir dreamerv3_runs/budget_frugal \\
        --step-hours 6 \\
        --size size1m \\
        --jax-platform auto
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

SIMULATOR_DIR = Path(__file__).resolve().parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train the budget-aware DreamerV3 variant on the blood supply "
            "simulator.  This variant optimises for budget spent and action "
            "cost alongside supply-chain performance, and terminates episodes "
            "early if the agent exhausts its budget."
        ),
    )
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument(
        "--timesteps",
        type=int,
        default=None,
        help="Compatibility alias for `--steps`.",
    )
    parser.add_argument("--scenario", default="train")
    parser.add_argument(
        "--logdir",
        type=str,
        default=str(SIMULATOR_DIR / "dreamerv3_runs" / "budget_default"),
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--step-hours", type=float, default=6.0)
    parser.add_argument("--envs", type=int, default=0)
    parser.add_argument("--eval-envs", type=int, default=0)
    parser.add_argument("--eval-eps", type=int, default=3)
    parser.add_argument("--train-ratio", type=float, default=32.0)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--batch-length", type=int, default=64)
    parser.add_argument("--size", default="size1m")
    parser.add_argument("--log-every", type=int, default=30)
    parser.add_argument("--report-every", type=int, default=120)
    parser.add_argument("--save-every", type=int, default=300)
    parser.add_argument("--jax-platform", default="auto")
    parser.add_argument("--compute-dtype", default="auto")
    parser.add_argument("--online-city-graph", action="store_true")
    parser.add_argument("--jax-profiler", default="auto")
    parser.add_argument("--metrics-write-mode", default="buffered")
    parser.add_argument(
        "--fixed-budget",
        type=float,
        default=None,
        help=(
            "Override per-scenario budgets with a single fixed budget.  "
            "Use a lower value (e.g. 4000–5000) to force frugal behaviour."
        ),
    )
    parser.add_argument(
        "--script",
        choices=["train", "train_eval"],
        default="train",
    )
    return parser.parse_args()


def build_official_argv(args: argparse.Namespace) -> list[str]:
    steps = args.steps if args.steps is not None else args.timesteps
    if steps is None:
        steps = 50_000

    argv = [
        str(SIMULATOR_DIR / "official_dreamerv3.py"),
        "--script",
        str(args.script),
        "--logdir",
        str(Path(args.logdir).expanduser().resolve()),
        "--steps",
        str(steps),
        "--scenario",
        str(args.scenario),
        "--seed",
        str(args.seed),
        "--step-hours",
        str(args.step_hours),
        "--envs",
        str(args.envs),
        "--eval-envs",
        str(args.eval_envs),
        "--eval-eps",
        str(args.eval_eps),
        "--train-ratio",
        str(args.train_ratio),
        "--batch-size",
        str(args.batch_size),
        "--batch-length",
        str(args.batch_length),
        "--size",
        str(args.size),
        "--log-every",
        str(args.log_every),
        "--report-every",
        str(args.report_every),
        "--save-every",
        str(args.save_every),
        "--jax-platform",
        str(args.jax_platform),
        "--compute-dtype",
        str(args.compute_dtype),
        "--jax-profiler",
        str(args.jax_profiler),
        "--metrics-write-mode",
        str(args.metrics_write_mode),
        # ── budget-variant flags ──────────────────────────────────────
        "--budget-variant",
    ]
    if args.fixed_budget is not None:
        argv.extend(["--fixed-budget", str(args.fixed_budget)])
    if args.online_city_graph:
        argv.append("--online-city-graph")
    return argv


def main() -> None:
    args = parse_args()
    sys.argv = build_official_argv(args)

    from official_dreamerv3 import main as official_main

    official_main()


if __name__ == "__main__":
    main()
