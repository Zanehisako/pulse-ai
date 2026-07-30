from __future__ import annotations

import argparse
import sys
from pathlib import Path

SIMULATOR_DIR = Path(__file__).resolve().parents[1] / "simulator"
if str(SIMULATOR_DIR) not in sys.path:
    sys.path.insert(0, str(SIMULATOR_DIR))

from train_dreamerv4 import (
    DEFAULT_SIMULATOR_COMPAT_LOGDIR,
    build_nicklas_hansen_commands,
    build_simulator_compat_argv,
    parse_args,
    resolve_nicklas_repo_root,
)


def test_parse_args_defaults_to_simulator_compat_when_no_subcommand(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["train_dreamerv4.py"])

    args = parse_args()

    assert args.backend == "simulator-compat"
    assert args.scenario == "train"
    assert Path(args.logdir) == DEFAULT_SIMULATOR_COMPAT_LOGDIR


def test_parse_args_accepts_nicklas_hansen_backend(monkeypatch):
    monkeypatch.setattr(
        sys,
        "argv",
        ["train_dreamerv4.py", "nicklas-hansen", "--stage", "tokenizer"],
    )

    args = parse_args()

    assert args.backend == "nicklas-hansen"
    assert args.stage == "tokenizer"


def test_build_simulator_compat_argv_forwards_into_official_runner(tmp_path):
    args = argparse.Namespace(
        backend="simulator-compat",
        steps=None,
        timesteps=None,
        scenario="train",
        logdir=str(tmp_path / "dreamerv4_compat"),
        seed=11,
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

    argv = build_simulator_compat_argv(args)

    assert argv[0].endswith("official_dreamerv3.py")
    assert argv[argv.index("--steps") + 1] == "50000"
    assert argv[argv.index("--logdir") + 1] == str(
        (tmp_path / "dreamerv4_compat").resolve()
    )


def test_build_nicklas_tokenizer_command_includes_data_dirs(tmp_path):
    args = argparse.Namespace(
        backend="nicklas-hansen",
        stage="tokenizer",
        python_bin=sys.executable,
        torchrun_bin="torchrun",
        nproc_per_node=4,
        data_dirs=[str(tmp_path / "expert"), str(tmp_path / "mixed")],
        frame_dirs=None,
        tokenizer_ckpt=None,
        use_actions=True,
        port=None,
        extra_args=["--batch_size", "64"],
        dry_run=False,
        repo_root=None,
    )

    commands = build_nicklas_hansen_commands(args)

    assert len(commands) == 1
    command = commands[0]
    assert command[:3] == ["torchrun", "--nproc_per_node=4", "train_tokenizer.py"]
    assert "--data_dirs" in command
    assert str((tmp_path / "expert").resolve()) in command
    assert str((tmp_path / "mixed").resolve()) in command
    assert command[-2:] == ["--batch_size", "64"]


def test_build_nicklas_dynamics_command_includes_actions_and_tokenizer_ckpt(tmp_path):
    tokenizer_ckpt = tmp_path / "tokenizer.pt"
    tokenizer_ckpt.write_bytes(b"")
    args = argparse.Namespace(
        backend="nicklas-hansen",
        stage="dynamics",
        python_bin=sys.executable,
        torchrun_bin="torchrun",
        nproc_per_node=2,
        data_dirs=[str(tmp_path / "raw")],
        frame_dirs=[str(tmp_path / "shards")],
        tokenizer_ckpt=str(tokenizer_ckpt),
        use_actions=True,
        port=None,
        extra_args=[],
        dry_run=False,
        repo_root=None,
    )

    commands = build_nicklas_hansen_commands(args)

    assert len(commands) == 1
    command = commands[0]
    assert command[:3] == ["torchrun", "--nproc_per_node=2", "train_dynamics.py"]
    assert "--use_actions" in command
    assert "--data_dirs" in command
    assert "--frame_dirs" in command
    assert "--tokenizer_ckpt" in command
    assert (
        str(tokenizer_ckpt.resolve()) == command[command.index("--tokenizer_ckpt") + 1]
    )


def test_build_nicklas_all_runs_tokenizer_then_dynamics(tmp_path):
    args = argparse.Namespace(
        backend="nicklas-hansen",
        stage="all",
        python_bin=sys.executable,
        torchrun_bin="torchrun",
        nproc_per_node=1,
        data_dirs=[str(tmp_path / "raw")],
        frame_dirs=[str(tmp_path / "shards")],
        tokenizer_ckpt=None,
        use_actions=False,
        port=None,
        extra_args=[],
        dry_run=False,
        repo_root=None,
    )

    commands = build_nicklas_hansen_commands(args)

    assert [command[2] for command in commands] == [
        "train_tokenizer.py",
        "train_dynamics.py",
    ]
    assert "--use_actions" not in commands[1]


def test_resolve_nicklas_repo_root_uses_env(monkeypatch, tmp_path):
    monkeypatch.setenv("PIOS_DREAMERV4_REPO", str(tmp_path))

    resolved = resolve_nicklas_repo_root(None)

    assert resolved == tmp_path.resolve()


def test_parse_args_accepts_native_backend(monkeypatch):
    monkeypatch.setattr(
        sys,
        "argv",
        ["train_dreamerv4.py", "native", "--steps", "1000"],
    )

    args = parse_args()

    assert args.backend == "native"
    assert args.steps == 1000


def test_parse_args_native_defaults(monkeypatch):
    monkeypatch.setattr(
        sys,
        "argv",
        ["train_dreamerv4.py", "native"],
    )

    args = parse_args()

    assert args.backend == "native"
    assert args.steps == 50_000
    assert args.seed == 0
    assert args.envs == 0
    assert args.model_preset == "default"
    assert args.hidden_size is None
    assert args.batch_size == 16
    assert args.seq_len == 32
    assert args.imagine_horizon == 15
    assert args.train_ratio == 1.0
    assert args.online_city_graph is False
    assert args.gamma == 0.997
    assert args.scenario == "train"
    assert args.use_data_parallel is True
    assert args.max_cuda_devices == 0


def test_parse_args_native_allows_fractional_train_ratio(monkeypatch):
    monkeypatch.setattr(
        sys,
        "argv",
        ["train_dreamerv4.py", "native", "--train-ratio", "0.5"],
    )

    args = parse_args()

    assert args.backend == "native"
    assert args.train_ratio == 0.5


def test_parse_args_native_accepts_online_city_graph(monkeypatch):
    monkeypatch.setattr(
        sys,
        "argv",
        ["train_dreamerv4.py", "native", "--online-city-graph"],
    )

    args = parse_args()

    assert args.backend == "native"
    assert args.online_city_graph is True


def test_parse_args_native_accepts_env_count(monkeypatch):
    monkeypatch.setattr(
        sys,
        "argv",
        ["train_dreamerv4.py", "native", "--envs", "4"],
    )

    args = parse_args()

    assert args.backend == "native"
    assert args.envs == 4


def test_parse_args_native_can_disable_data_parallel(monkeypatch):
    monkeypatch.setattr(
        sys,
        "argv",
        ["train_dreamerv4.py", "native", "--no-use-data-parallel"],
    )

    args = parse_args()

    assert args.backend == "native"
    assert args.use_data_parallel is False


def test_parse_args_native_accepts_max_cuda_devices(monkeypatch):
    monkeypatch.setattr(
        sys,
        "argv",
        ["train_dreamerv4.py", "native", "--max-cuda-devices", "2"],
    )

    args = parse_args()

    assert args.backend == "native"
    assert args.max_cuda_devices == 2


def test_parse_args_native_accepts_fast_model_preset(monkeypatch):
    monkeypatch.setattr(
        sys,
        "argv",
        ["train_dreamerv4.py", "native", "--model-preset", "fast"],
    )

    args = parse_args()

    assert args.backend == "native"
    assert args.model_preset == "fast"


def test_parse_args_native_accepts_model_overrides(monkeypatch):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "train_dreamerv4.py",
            "native",
            "--hidden-size",
            "160",
            "--stoch-categories",
            "10",
            "--stoch-classes",
            "6",
            "--mlp-units",
            "144",
        ],
    )

    args = parse_args()

    assert args.backend == "native"
    assert args.hidden_size == 160
    assert args.stoch_categories == 10
    assert args.stoch_classes == 6
    assert args.mlp_units == 144
