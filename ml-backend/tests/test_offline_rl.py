from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch


SIMULATOR_DIR = Path(__file__).resolve().parents[1] / "simulator"
if str(SIMULATOR_DIR) not in sys.path:
    sys.path.insert(0, str(SIMULATOR_DIR))

import engine
from offline_rl import SquashedGaussianActor, load_offline_policy_agent
from plot_dreamerv3_metrics import summarize_series


def test_resolve_learned_continuous_model_path_prefers_env_override(
    monkeypatch, tmp_path
):
    checkpoint = tmp_path / "offline_iql.pt"
    checkpoint.write_bytes(b"checkpoint")
    monkeypatch.setenv("PIOS_IQL_CHECKPOINT", str(checkpoint))

    resolved = engine.resolve_learned_continuous_model_path("iql_offline")

    assert resolved == str(checkpoint)


def test_load_offline_policy_agent_returns_bounded_actions(tmp_path):
    checkpoint = tmp_path / "iql.pt"
    actor = SquashedGaussianActor(4, 3, hidden_dim=16)
    torch.save(
        {
            "algorithm": "iql",
            "obs_dim": 4,
            "action_dim": 3,
            "hidden_dim": 16,
            "actor_state_dict": actor.state_dict(),
        },
        checkpoint,
    )

    agent = load_offline_policy_agent(checkpoint, algorithm="iql", device="cpu")
    action, _ = agent.predict(np.zeros(4, dtype=np.float32), deterministic=True)

    assert action.shape == (3,)
    assert np.all(action <= 1.0)
    assert np.all(action >= -1.0)
    assert agent._controller_name == "trained_iql_offline"


def test_summarize_series_reports_tail_mean():
    series = {
        "episode/score": [
            (1.0, -2.0),
            (2.0, 0.0),
            (3.0, 6.0),
        ]
    }

    summary = summarize_series(series)

    assert summary == {
        "count": 3.0,
        "mean": 4.0 / 3.0,
        "min": -2.0,
        "max": 6.0,
        "last": 6.0,
        "tail_mean": 4.0 / 3.0,
    }
