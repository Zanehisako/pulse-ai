from __future__ import annotations

import argparse
import copy
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch

from dreamerv4_agent import (
    SIMULATOR_DIR,
    publish_dreamerv4_checkpoint,
    resolve_dreamerv4_checkpoint_artifact,
    train_dreamerv4_native,
)

DEFAULT_NOTEBOOK_LOGDIR = SIMULATOR_DIR / "dreamerv4_runs" / "notebook_cuda"
DEFAULT_DROPIN_EXPORT_DIR = SIMULATOR_DIR / "dreamerv4_runs" / "native" / "ckpt"


@dataclass(slots=True)
class DreamerV4NotebookConfig:
    steps: int = 50_000
    seed: int = 7
    logdir: str = str(DEFAULT_NOTEBOOK_LOGDIR)
    step_hours: float = 6.0
    scenario: str = "train"
    envs: int = 0
    model_preset: str = "fast"
    hidden_size: int | None = None
    stoch_categories: int | None = None
    stoch_classes: int | None = None
    mlp_units: int | None = None
    batch_size: int = 256
    seq_len: int = 48
    imagine_horizon: int = 15
    lr_world: float = 3e-4
    lr_actor: float = 1e-4
    lr_critic: float = 1e-4
    save_every: int = 5_000
    log_every: int = 100
    prefill_steps: int = 1_000
    train_ratio: float = 1.0
    gamma: float = 0.997
    lambda_: float = 0.95
    online_city_graph: bool = False
    device: str = "cuda"
    use_data_parallel: bool = True
    max_cuda_devices: int = 0
    publish_dropin: bool = True
    export_dir: str = str(DEFAULT_DROPIN_EXPORT_DIR)

    def to_namespace(self) -> argparse.Namespace:
        return argparse.Namespace(**asdict(self))


def runtime_report(preferred_device: str = "cuda") -> dict[str, Any]:
    auto_device = "cuda" if torch.cuda.is_available() else "cpu"
    cuda_device_count = int(torch.cuda.device_count() if torch.cuda.is_available() else 0)
    report: dict[str, Any] = {
        "preferred_device": preferred_device,
        "auto_device": auto_device,
        "torch_version": torch.__version__,
        "cuda_available": bool(torch.cuda.is_available()),
        "cuda_device_count": cuda_device_count,
        "recommended_batch_size": 256 if cuda_device_count <= 1 else 512,
    }
    if torch.cuda.is_available():
        current_device = torch.cuda.current_device()
        report.update(
            {
                "cuda_device_index": int(current_device),
                "cuda_device_name": torch.cuda.get_device_name(current_device),
                "cuda_capability": ".".join(
                    str(part) for part in torch.cuda.get_device_capability(current_device)
                ),
                "tf32_matmul": bool(getattr(torch.backends.cuda.matmul, "allow_tf32", False)),
                "tf32_cudnn": bool(getattr(torch.backends.cudnn, "allow_tf32", False)),
                "cuda_devices": [
                    torch.cuda.get_device_name(idx) for idx in range(cuda_device_count)
                ],
            }
        )
    return report


def latest_checkpoint_from_logdir(logdir: str | Path = DEFAULT_NOTEBOOK_LOGDIR) -> str:
    return resolve_dreamerv4_checkpoint_artifact(Path(logdir).expanduser() / "ckpt")


def train_from_notebook(
    config: DreamerV4NotebookConfig | None = None,
    /,
    **overrides: Any,
) -> dict[str, Any]:
    cfg = config or DreamerV4NotebookConfig()
    if overrides:
        payload = asdict(cfg)
        payload.update(overrides)
        cfg = DreamerV4NotebookConfig(**payload)

    if cfg.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA training was requested but torch.cuda.is_available() is False. "
            "Switch the notebook config to device='auto' or run it on a CUDA host."
        )

    result = train_dreamerv4_native(cfg.to_namespace())
    result["config"] = asdict(cfg)

    published_checkpoint: str | None = None
    if cfg.publish_dropin:
        published_checkpoint = publish_dreamerv4_checkpoint(
            result["checkpoint_path"],
            cfg.export_dir,
        )

    result["published_checkpoint"] = published_checkpoint
    result["dropin_checkpoint"] = (
        published_checkpoint or result["checkpoint_path"]
    )
    return result


def load_metrics_frame(logdir: str | Path = DEFAULT_NOTEBOOK_LOGDIR):
    import pandas as pd

    metrics_path = Path(logdir).expanduser() / "metrics.jsonl"
    if not metrics_path.exists():
        raise FileNotFoundError(f"DreamerV4 metrics file not found: {metrics_path}")

    rows = [
        json.loads(line)
        for line in metrics_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    return pd.DataFrame(rows)


def evaluate_checkpoint(
    checkpoint_path: str | Path | None = None,
    *,
    scenario_key: str = "baseline",
    hours: int | None = 72,
    seed: int = 7,
    step_hours: float = 6.0,
    enable_logs: bool = False,
    fast_mode: bool = True,
) -> dict[str, Any]:
    from engine import load_dreamerv4_agent, run_scenario, summarize_state
    from scenarios import SCENARIOS
    from train_mappo import load_training_context

    resolved_checkpoint = (
        resolve_dreamerv4_checkpoint_artifact(checkpoint_path)
        if checkpoint_path is not None
        else resolve_dreamerv4_checkpoint_artifact(DEFAULT_DROPIN_EXPORT_DIR)
    )

    params = copy.deepcopy(SCENARIOS[scenario_key])
    params.strategy_key = "dreamerv4"
    if hours is not None:
        params.sim_hours = int(hours)

    agent = load_dreamerv4_agent(
        resolved_checkpoint,
        seed=seed,
        step_hours=step_hours,
    )
    G, NORTH, SOUTH, EAST, WEST = load_training_context()
    state = run_scenario(
        params,
        G,
        NORTH,
        SOUTH,
        EAST,
        WEST,
        seed=seed,
        enable_logs=enable_logs,
        fast_mode=fast_mode,
        official_dreamerv3_agent=agent,
        official_dreamerv3_checkpoint=None,
    )
    if state.env.now < state.params.sim_hours:
        state.env.run(until=state.params.sim_hours)

    summary = summarize_state(state)
    return {
        "checkpoint_path": resolved_checkpoint,
        "scenario_key": scenario_key,
        "seed": int(seed),
        "hours": int(state.params.sim_hours),
        **summary,
    }


__all__ = [
    "DEFAULT_DROPIN_EXPORT_DIR",
    "DEFAULT_NOTEBOOK_LOGDIR",
    "DreamerV4NotebookConfig",
    "evaluate_checkpoint",
    "latest_checkpoint_from_logdir",
    "load_metrics_frame",
    "runtime_report",
    "train_from_notebook",
]
