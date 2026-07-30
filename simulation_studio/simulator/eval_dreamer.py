from __future__ import annotations

import argparse
import sys
from pathlib import Path

from official_dreamerv3 import resolve_official_checkpoint_path


SIMULATOR_DIR = Path(__file__).resolve().parent
OFFICIAL_RUNNER = SIMULATOR_DIR / "official_dreamerv3.py"


def _discover_official_checkpoints(limit: int = 3) -> list[Path]:
    runs_dir = SIMULATOR_DIR / "dreamerv3_runs"
    if not runs_dir.exists():
        return []
    return sorted(runs_dir.glob("**/ckpt"))[:limit]


def _build_checkpoint_hint() -> str:
    checkpoints = _discover_official_checkpoints()
    if not checkpoints:
        return (
            "Use an official DreamerV3 checkpoint path like "
            "`simulation_studio/simulator/dreamerv3_runs/<run>/ckpt`."
        )
    formatted = ", ".join(str(path) for path in checkpoints)
    return f"Use an official DreamerV3 checkpoint path like: {formatted}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate the official DreamerV3 implementation on the blood supply simulator."
        )
    )
    parser.add_argument(
        "--model",
        "--checkpoint",
        dest="checkpoint",
        required=True,
        help="Path to the official DreamerV3 checkpoint directory/file, usually dreamerv3_runs/<run>/ckpt.",
    )
    parser.add_argument(
        "--scenario",
        default="baseline",
        help="Scenario key, comma-separated keys, or 'all'.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--logdir",
        type=str,
        default=None,
        help="Optional eval log directory. Defaults to <checkpoint parent>/eval_only.",
    )
    parser.add_argument("--step-hours", type=float, default=6.0)
    parser.add_argument("--eval-envs", type=int, default=1)
    parser.add_argument("--eval-eps", type=int, default=3)
    parser.add_argument("--size", default="size1m")
    parser.add_argument("--jax-platform", default="auto")
    parser.add_argument("--compute-dtype", default="auto")
    parser.add_argument("--online-city-graph", action="store_true")
    parser.add_argument("--jax-profiler", default="auto")
    parser.add_argument("--metrics-write-mode", default="buffered")
    parser.add_argument(
        "--deterministic",
        action="store_true",
        help="Accepted for compatibility. The official runner manages eval behavior internally.",
    )
    return parser.parse_args()


def build_official_argv(args: argparse.Namespace) -> list[str]:
    raw_checkpoint = Path(args.checkpoint).expanduser()
    if raw_checkpoint.suffix == ".pt":
        raise ValueError(
            "Custom Dreamer `.pt` checkpoints are no longer supported. "
            + _build_checkpoint_hint()
        )

    checkpoint = Path(resolve_official_checkpoint_path(str(raw_checkpoint)))

    logdir = (
        Path(args.logdir).expanduser().resolve()
        if args.logdir
        else (checkpoint.parent / "eval_only").resolve()
    )
    argv = [
        str(OFFICIAL_RUNNER),
        "--script",
        "eval_only",
        "--from-checkpoint",
        str(checkpoint),
        "--logdir",
        str(logdir),
        "--scenario",
        str(args.scenario),
        "--seed",
        str(args.seed),
        "--step-hours",
        str(args.step_hours),
        "--eval-envs",
        str(args.eval_envs),
        "--eval-eps",
        str(args.eval_eps),
        "--size",
        str(args.size),
        "--jax-platform",
        str(args.jax_platform),
        "--compute-dtype",
        str(args.compute_dtype),
        "--jax-profiler",
        str(args.jax_profiler),
        "--metrics-write-mode",
        str(args.metrics_write_mode),
    ]
    if args.online_city_graph:
        argv.append("--online-city-graph")
    return argv


def main() -> None:
    args = parse_args()
    try:
        argv = build_official_argv(args)
    except (FileNotFoundError, ValueError) as exc:
        print(f"[eval_dreamer] {exc}", file=sys.stderr)
        raise SystemExit(2) from exc
    if args.deterministic:
        print(
            "[eval_dreamer] `--deterministic` is ignored because the official DreamerV3 "
            "runner owns evaluation behavior."
        )
    sys.argv = argv

    from official_dreamerv3 import main as official_main

    official_main()


if __name__ == "__main__":
    main()
