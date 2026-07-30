from __future__ import annotations

import argparse
import os
import shlex
import subprocess
import sys
from pathlib import Path

from train_dreamer import build_official_argv as build_dreamerv3_argv

SIMULATOR_DIR = Path(__file__).resolve().parent
DEFAULT_SIMULATOR_COMPAT_LOGDIR = SIMULATOR_DIR / "dreamerv4_runs" / "simulator_compat"
DEFAULT_NICKLAS_STAGE = "dynamics"
NICKLAS_REPO_ENV = "PIOS_DREAMERV4_REPO"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Train DreamerV4 in one of two modes: "
            "`simulator-compat` forwards into the existing simulator Dreamer "
            "runtime so the resulting checkpoint can be used by this repo today; "
            "`nicklas-hansen` orchestrates the unofficial PyTorch Dreamer4 repo "
            "stages described in the paper-aligned public implementation."
        ),
    )
    subparsers = parser.add_subparsers(dest="backend")

    compat = subparsers.add_parser(
        "simulator-compat",
        help=(
            "Train a Dreamer-compatible checkpoint the current simulator can load "
            "through the `dreamerv4` strategy."
        ),
    )
    add_simulator_compat_args(compat)

    nicklas = subparsers.add_parser(
        "nicklas-hansen",
        help=(
            "Run the unofficial PyTorch Dreamer4 stages from "
            "https://github.com/nicklashansen/dreamer4."
        ),
    )
    add_nicklas_hansen_args(nicklas)

    native = subparsers.add_parser(
        "native",
        help=(
            "Train a native PyTorch DreamerV4 agent directly on the blood "
            "supply simulation environment."
        ),
    )
    add_native_args(native)

    return parser


def add_simulator_compat_args(parser: argparse.ArgumentParser) -> None:
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
        default=str(DEFAULT_SIMULATOR_COMPAT_LOGDIR),
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
        "--script",
        choices=["train", "train_eval"],
        default="train",
    )


def add_nicklas_hansen_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--stage",
        choices=["preprocess", "tokenizer", "dynamics", "interactive", "all"],
        default=DEFAULT_NICKLAS_STAGE,
        help=(
            "Dreamer4 stage to run in the external repository. `all` runs "
            "tokenizer followed by dynamics."
        ),
    )
    parser.add_argument(
        "--repo-root",
        type=str,
        default=None,
        help=(
            "Path to a clone of nicklashansen/dreamer4. Defaults to the "
            f"`{NICKLAS_REPO_ENV}` environment variable."
        ),
    )
    parser.add_argument(
        "--python-bin",
        type=str,
        default=sys.executable,
        help="Python executable used for non-torchrun stages.",
    )
    parser.add_argument(
        "--torchrun-bin",
        type=str,
        default="torchrun",
        help="Torchrun executable used for tokenizer and dynamics training.",
    )
    parser.add_argument(
        "--nproc-per-node",
        type=int,
        default=1,
        help="Local process count for torchrun-based stages.",
    )
    parser.add_argument(
        "--data-dir",
        dest="data_dirs",
        action="append",
        default=None,
        help=(
            "Repeat to pass one or more `--data_dirs` entries to tokenizer or "
            "dynamics training."
        ),
    )
    parser.add_argument(
        "--frame-dir",
        dest="frame_dirs",
        action="append",
        default=None,
        help=(
            "Repeat to pass one or more `--frame_dirs` entries to dynamics training."
        ),
    )
    parser.add_argument(
        "--tokenizer-ckpt",
        type=str,
        default=None,
        help="Optional tokenizer checkpoint path for dynamics or interactive stages.",
    )
    parser.add_argument(
        "--use-actions",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Whether to pass `--use_actions` to Dreamer4 dynamics training.",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=None,
        help="Optional port override for the Dreamer4 interactive web UI.",
    )
    parser.add_argument(
        "--extra-arg",
        dest="extra_args",
        action="append",
        default=[],
        help=(
            "Raw argument appended to the external Dreamer4 command. Repeat once "
            "per token, for example `--extra-arg=--batch_size --extra-arg=64`."
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the external command(s) without executing them.",
    )


def add_native_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--steps", type=int, default=50_000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--logdir",
        type=str,
        default=str(SIMULATOR_DIR / "dreamerv4_runs" / "native"),
    )
    parser.add_argument("--step-hours", type=float, default=6.0)
    parser.add_argument("--scenario", default="train")
    parser.add_argument("--envs", type=int, default=0)
    parser.add_argument(
        "--model-preset",
        choices=["default", "fast"],
        default="default",
    )
    parser.add_argument("--hidden-size", type=int, default=None)
    parser.add_argument("--stoch-categories", type=int, default=None)
    parser.add_argument("--stoch-classes", type=int, default=None)
    parser.add_argument("--mlp-units", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--seq-len", type=int, default=32)
    parser.add_argument("--imagine-horizon", type=int, default=15)
    parser.add_argument("--lr-world", type=float, default=3e-4)
    parser.add_argument("--lr-actor", type=float, default=1e-4)
    parser.add_argument("--lr-critic", type=float, default=1e-4)
    parser.add_argument("--save-every", type=int, default=5000)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--prefill-steps", type=int, default=1000)
    parser.add_argument("--train-ratio", type=float, default=1.0)
    parser.add_argument("--gamma", type=float, default=0.997)
    parser.add_argument("--lambda_", type=float, default=0.95)
    parser.add_argument(
        "--online-city-graph",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Device: 'auto' (detect GPU), 'cpu', 'cuda', 'mps'.",
    )
    parser.add_argument(
        "--use-data-parallel",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use all visible CUDA GPUs for data-parallel DreamerV4 optimization.",
    )
    parser.add_argument(
        "--max-cuda-devices",
        type=int,
        default=0,
        help="Optional cap on the number of visible CUDA devices to use. 0 uses all.",
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    raw_argv = list(sys.argv[1:] if argv is None else argv)
    if not raw_argv or raw_argv[0].startswith("-"):
        raw_argv = ["simulator-compat", *raw_argv]
    return build_parser().parse_args(raw_argv)


def build_simulator_compat_argv(args: argparse.Namespace) -> list[str]:
    return build_dreamerv3_argv(args)


def resolve_nicklas_repo_root(explicit_repo_root: str | None) -> Path:
    repo_root = explicit_repo_root or os.environ.get(NICKLAS_REPO_ENV)
    if not repo_root:
        raise ValueError(
            f"nicklas-hansen backend requires `--repo-root` or `{NICKLAS_REPO_ENV}`."
        )
    path = Path(repo_root).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"Dreamer4 repo root does not exist: {path}")
    return path


def _normalized_paths(paths: list[str] | None) -> list[str]:
    return [str(Path(path).expanduser().resolve()) for path in (paths or [])]


def _torchrun_prefix(args: argparse.Namespace) -> list[str]:
    return [args.torchrun_bin, f"--nproc_per_node={int(args.nproc_per_node)}"]


def _preprocess_command(args: argparse.Namespace) -> list[str]:
    # The upstream README instructs users to configure FILEDIR / OUTDIR inside
    # preprocess_data.py, so this wrapper only launches the script.
    return [args.python_bin, "preprocess_data.py", *args.extra_args]


def _tokenizer_command(args: argparse.Namespace) -> list[str]:
    cmd = [*_torchrun_prefix(args), "train_tokenizer.py"]
    data_dirs = _normalized_paths(args.data_dirs)
    if data_dirs:
        cmd.extend(["--data_dirs", *data_dirs])
    cmd.extend(args.extra_args)
    return cmd


def _dynamics_command(args: argparse.Namespace) -> list[str]:
    cmd = [*_torchrun_prefix(args), "train_dynamics.py"]
    if args.use_actions:
        cmd.append("--use_actions")
    data_dirs = _normalized_paths(args.data_dirs)
    if data_dirs:
        cmd.extend(["--data_dirs", *data_dirs])
    frame_dirs = _normalized_paths(args.frame_dirs)
    if frame_dirs:
        cmd.extend(["--frame_dirs", *frame_dirs])
    if args.tokenizer_ckpt:
        cmd.extend(
            [
                "--tokenizer_ckpt",
                str(Path(args.tokenizer_ckpt).expanduser().resolve()),
            ]
        )
    cmd.extend(args.extra_args)
    return cmd


def _interactive_command(args: argparse.Namespace) -> list[str]:
    cmd = [args.python_bin, "interactive.py"]
    if args.port is not None:
        cmd.extend(["--port", str(int(args.port))])
    if args.tokenizer_ckpt:
        cmd.extend(
            [
                "--tokenizer_ckpt",
                str(Path(args.tokenizer_ckpt).expanduser().resolve()),
            ]
        )
    cmd.extend(args.extra_args)
    return cmd


def build_nicklas_hansen_commands(args: argparse.Namespace) -> list[list[str]]:
    stage = args.stage
    if stage == "preprocess":
        return [_preprocess_command(args)]
    if stage == "tokenizer":
        return [_tokenizer_command(args)]
    if stage == "dynamics":
        return [_dynamics_command(args)]
    if stage == "interactive":
        return [_interactive_command(args)]
    if stage == "all":
        return [_tokenizer_command(args), _dynamics_command(args)]
    raise ValueError(f"Unsupported Dreamer4 stage: {stage}")


def _joined_command(command: list[str]) -> str:
    return shlex.join(command)


def run_nicklas_hansen_backend(args: argparse.Namespace) -> None:
    repo_root = resolve_nicklas_repo_root(args.repo_root)
    commands = build_nicklas_hansen_commands(args)
    env = os.environ.copy()
    existing_pythonpath = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = (
        f"{repo_root}{os.pathsep}{existing_pythonpath}"
        if existing_pythonpath
        else str(repo_root)
    )

    for command in commands:
        if args.dry_run:
            print(_joined_command(command))
            continue
        subprocess.run(command, cwd=repo_root, env=env, check=True)


def main() -> None:
    args = parse_args()
    if args.backend == "simulator-compat":
        sys.argv = build_simulator_compat_argv(args)
        from official_dreamerv3 import main as official_main

        official_main()
        return

    if args.backend == "nicklas-hansen":
        run_nicklas_hansen_backend(args)
        return

    if args.backend == "native":
        from dreamerv4_agent import train_dreamerv4_native

        # Translate hyphens to underscores for the training function
        native_args = argparse.Namespace(
            steps=args.steps,
            seed=args.seed,
            logdir=args.logdir,
            step_hours=args.step_hours,
            scenario=args.scenario,
            envs=args.envs,
            model_preset=args.model_preset,
            hidden_size=args.hidden_size,
            stoch_categories=args.stoch_categories,
            stoch_classes=args.stoch_classes,
            mlp_units=args.mlp_units,
            batch_size=args.batch_size,
            seq_len=args.seq_len,
            imagine_horizon=args.imagine_horizon,
            lr_world=args.lr_world,
            lr_actor=args.lr_actor,
            lr_critic=args.lr_critic,
            save_every=args.save_every,
            log_every=args.log_every,
            prefill_steps=args.prefill_steps,
            train_ratio=args.train_ratio,
            gamma=args.gamma,
            lambda_=args.lambda_,
            online_city_graph=args.online_city_graph,
            device=args.device,
            use_data_parallel=args.use_data_parallel,
            max_cuda_devices=args.max_cuda_devices,
        )
        train_dreamerv4_native(native_args)
        return

    raise ValueError(f"Unsupported DreamerV4 backend: {args.backend}")


if __name__ == "__main__":
    main()
