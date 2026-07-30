from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np


SIMULATOR_DIR = Path(__file__).resolve().parents[1] / "simulator"
if str(SIMULATOR_DIR) not in sys.path:
    sys.path.insert(0, str(SIMULATOR_DIR))

from mappo import MAPPO
from scenarios import ACTION_CATALOG, SCENARIOS
from train_mappo import (
    choose_scenario_key,
    compound_crisis_score,
    default_manager_context,
    manager_signal_from_action,
    prepare_mappo_scenario,
    scenario_focus_pressure,
)


def test_mappo_terminal_gae_ignores_bootstrap_value_after_done():
    agent = MAPPO(obs_dim=14, action_dim=6, n_agents=3)

    rewards = torch_tensor([1.0, 2.0])
    values = torch_tensor([0.5, 0.25])
    dones = torch_tensor([0.0, 1.0])

    adv_zero, ret_zero = agent.compute_gae(rewards, values, dones, last_value=0.0)
    adv_huge, ret_huge = agent.compute_gae(rewards, values, dones, last_value=99.0)

    assert np.allclose(adv_zero.numpy(), adv_huge.numpy())
    assert np.allclose(ret_zero.numpy(), ret_huge.numpy())


def test_mappo_update_stays_finite_with_large_simulator_observations():
    agent = MAPPO(obs_dim=14, action_dim=6, n_agents=3)

    traj = {
        "global_states": [],
        "rewards": [],
        "dones": [],
        "normalized_obs": True,
        "log_probs": {name: [] for name in agent.agent_names},
        "actions": {name: [] for name in agent.agent_names},
        "obs": {name: [] for name in agent.agent_names},
        "action_masks": {name: [] for name in agent.agent_names},
        "action_biases": {name: [] for name in agent.agent_names},
    }

    for step in range(6):
        obs_dict = {}
        step_masks = {}
        step_biases = {}
        for idx, name in enumerate(agent.agent_names):
            obs_dict[name] = np.array(
                [
                    1_000.0 + step * 35 + idx * 5,
                    800.0 + step * 21 + idx * 7,
                    600.0 + step * 13 + idx * 11,
                    450.0 + step * 17,
                    30.0 + step,
                    15.0 + idx,
                    3.0 + step,
                    90.0 + step * 4,
                    18.0,
                    float(step % 4),
                    step / 6.0,
                    3.0,
                    120.0 + idx * 10,
                    0.25 + idx * 0.05,
                ],
                dtype=np.float32,
            )
            base_mask = np.array([1.0, 1.0, 1.0, 1.0, 0.0, 0.0], dtype=np.float32)
            if name == "hospital":
                base_mask = np.array([1.0, 0.0, 1.0, 0.0, 1.0, 0.0], dtype=np.float32)
            if name == "logistics":
                base_mask = np.array([1.0, 0.0, 0.0, 1.0, 1.0, 1.0], dtype=np.float32)
            step_masks[name] = base_mask
            bias = np.zeros(6, dtype=np.float32)
            bias[(idx + step) % 6] = 0.75
            step_biases[name] = bias

        actions, log_probs, processed_obs = agent.act(
            obs_dict,
            action_masks=step_masks,
            action_biases=step_biases,
        )
        global_state = agent.build_global_state(processed_obs)

        for name in agent.agent_names:
            assert np.max(np.abs(processed_obs[name])) <= agent.obs_clip + 1e-6
            traj["actions"][name].append(actions[name])
            traj["obs"][name].append(processed_obs[name])
            traj["log_probs"][name].append(log_probs[name])
            traj["action_masks"][name].append(step_masks[name])
            traj["action_biases"][name].append(step_biases[name])

        traj["global_states"].append(global_state)
        traj["rewards"].append(float(np.clip((-1) ** step * 0.4, -1.0, 1.0)))
        traj["dones"].append(float(step == 5))

    metrics = agent.update(traj, epochs=2)

    assert all(math.isfinite(value) for value in metrics.values())
    for actor in agent.actors.values():
        for param in actor.parameters():
            assert np.isfinite(param.detach().cpu().numpy()).all()
    for param in agent.critic.parameters():
        assert np.isfinite(param.detach().cpu().numpy()).all()


def test_mappo_action_masks_block_invalid_actions():
    masks = {
        "supply": [1.0, 1.0, 0.0, 0.0],
        "hospital": [1.0, 0.0, 1.0, 0.0],
        "logistics": [1.0, 0.0, 0.0, 1.0],
    }
    agent = MAPPO(obs_dim=4, action_dim=4, n_agents=3, action_masks=masks)

    obs = {
        "supply": np.array([0.2, 0.1, 0.0, 0.4], dtype=np.float32),
        "hospital": np.array([0.1, 0.3, 0.6, 0.2], dtype=np.float32),
        "logistics": np.array([0.0, 0.2, 0.5, 0.9], dtype=np.float32),
    }

    for _ in range(32):
        actions, _, _ = agent.act(obs)
        assert actions["supply"] in {0, 1}
        assert actions["hospital"] in {0, 2}
        assert actions["logistics"] in {0, 3}


def test_mappo_deterministic_actions_follow_bias_and_masks():
    masks = {
        "supply": [1.0, 1.0, 1.0, 0.0],
        "hospital": [1.0, 0.0, 1.0, 0.0],
        "logistics": [1.0, 0.0, 0.0, 1.0],
    }
    agent = MAPPO(obs_dim=4, action_dim=4, n_agents=3, action_masks=masks)
    obs = {
        "supply": np.array([0.2, 0.1, 0.0, 0.4], dtype=np.float32),
        "hospital": np.array([0.1, 0.3, 0.6, 0.2], dtype=np.float32),
        "logistics": np.array([0.0, 0.2, 0.5, 0.9], dtype=np.float32),
    }
    biases = {
        "supply": np.array([0.0, 0.1, 1.5, -1.0], dtype=np.float32),
        "hospital": np.array([0.0, -1.0, 1.2, -1.0], dtype=np.float32),
        "logistics": np.array([0.0, -1.0, -1.0, 1.3], dtype=np.float32),
    }

    actions, _, _ = agent.act(obs, action_biases=biases, deterministic=True)

    assert actions["supply"] == 2
    assert actions["hospital"] == 2
    assert actions["logistics"] == 3


def test_mappo_step_masks_can_disable_actions_temporarily():
    masks = {
        "supply": [1.0, 1.0, 1.0, 0.0],
        "hospital": [1.0, 1.0, 1.0, 0.0],
        "logistics": [1.0, 1.0, 1.0, 0.0],
    }
    agent = MAPPO(obs_dim=4, action_dim=4, n_agents=3, action_masks=masks)

    obs = {
        "supply": np.array([0.3, 0.2, 0.1, 0.4], dtype=np.float32),
        "hospital": np.array([0.1, 0.4, 0.2, 0.3], dtype=np.float32),
        "logistics": np.array([0.2, 0.5, 0.1, 0.6], dtype=np.float32),
    }
    step_masks = {
        "supply": np.array([1.0, 0.0, 1.0, 0.0], dtype=np.float32),
        "hospital": np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32),
        "logistics": np.array([1.0, 1.0, 0.0, 0.0], dtype=np.float32),
    }

    for _ in range(32):
        actions, _, _ = agent.act(obs, action_masks=step_masks)
        assert actions["supply"] in {0, 2}
        assert actions["hospital"] == 0
        assert actions["logistics"] in {0, 1}


def test_prepare_mappo_scenario_clears_fixed_strategy_actions():
    scenario = prepare_mappo_scenario(SCENARIOS["transport_disruption"])

    assert scenario.strategy_key == "baseline"
    assert scenario.transport_penalty == SCENARIOS["transport_disruption"].transport_penalty
    assert scenario.forced_weather == SCENARIOS["transport_disruption"].forced_weather


def test_rapid_courier_now_targets_transport_and_donor_access():
    action = ACTION_CATALOG["rapid_courier"]

    assert action.transport_relief_factor < 0.55
    assert action.donor_arrival_factor < 1.0
    assert action.donor_show_up_factor > 1.0


def test_mappo_supports_manager_plus_role_agent_names():
    agent = MAPPO(
        obs_dim=5,
        action_dim=6,
        n_agents=4,
        agent_names=["manager", "supply", "hospital", "logistics"],
    )

    obs = {
        "manager": np.array([0.3, 0.2, 0.1, 0.4, 0.5], dtype=np.float32),
        "supply": np.array([0.1, 0.2, 0.3, 0.4, 0.5], dtype=np.float32),
        "hospital": np.array([0.5, 0.4, 0.3, 0.2, 0.1], dtype=np.float32),
        "logistics": np.array([0.2, 0.1, 0.4, 0.3, 0.6], dtype=np.float32),
    }

    manager_action, manager_log_prob, manager_processed = agent.act_single("manager", obs["manager"])
    actions, log_probs, processed = agent.act(
        {name: obs[name] for name in ["supply", "hospital", "logistics"]}
    )
    processed["manager"] = manager_processed

    assert isinstance(manager_action, int)
    assert math.isfinite(manager_log_prob)
    assert set(actions) == {"supply", "hospital", "logistics"}
    assert set(log_probs) == {"supply", "hospital", "logistics"}
    assert agent.build_global_state(processed).shape == (20,)


def test_manager_signal_is_state_conditioned():
    class _DummyParams:
        transport_penalty = 1.0
        donor_show_factor = 0.95

    class _DummyState:
        params = _DummyParams()

    calm_raw = {
        "donor_stress": 0.1,
        "recent_shortage_rate": 0.02,
        "demand_stress": 0.05,
        "transport_stress": 0.03,
        "weather_stress": 0.0,
        "cumulative_shortage_rate": 0.02,
    }
    calm_signals = {
        "role_urgency": {"supply": 0.12, "hospital": 0.10, "logistics": 0.08},
        "global_severity": 0.12,
        "shortage_recovery": 0.06,
        "stable_system": True,
    }
    crisis_raw = {
        "donor_stress": 0.7,
        "recent_shortage_rate": 0.35,
        "demand_stress": 0.28,
        "transport_stress": 0.75,
        "weather_stress": 0.65,
        "cumulative_shortage_rate": 0.30,
    }
    crisis_signals = {
        "role_urgency": {"supply": 0.55, "hospital": 0.82, "logistics": 0.92},
        "global_severity": 0.88,
        "shortage_recovery": 0.0,
        "stable_system": False,
    }

    calm_signal = manager_signal_from_action(
        3, _DummyState(), calm_raw, calm_signals, manager_context=default_manager_context()
    )
    crisis_signal = manager_signal_from_action(
        3, _DummyState(), crisis_raw, crisis_signals, manager_context=default_manager_context()
    )

    assert crisis_signal["budget"] >= calm_signal["budget"]
    assert crisis_signal["priorities"]["logistics"] > calm_signal["priorities"]["logistics"]


def test_compound_crisis_score_rises_when_all_axes_are_stressed():
    calm_raw = {
        "donor_stress": 0.10,
        "critical_gap": 0.05,
        "cumulative_shortage_rate": 0.02,
        "recent_shortage_rate": 0.03,
        "demand_stress": 0.05,
        "transport_stress": 0.06,
        "weather_stress": 0.02,
    }
    crisis_raw = {
        "donor_stress": 0.72,
        "critical_gap": 0.38,
        "cumulative_shortage_rate": 0.22,
        "recent_shortage_rate": 0.32,
        "demand_stress": 0.41,
        "transport_stress": 0.78,
        "weather_stress": 0.66,
    }
    calm_signals = {"global_severity": 0.12}
    crisis_signals = {"global_severity": 0.89}

    assert compound_crisis_score(crisis_raw, crisis_signals) > compound_crisis_score(
        calm_raw, calm_signals
    )


def test_scenario_focus_pressure_and_selection_now_follow_episode_reward():
    history = []
    for episode in range(36):
        history.append(
            {
                "episode": episode,
                "scenario": "combined_crisis" if episode % 3 else "baseline",
                "reward": -2.4 if episode % 3 else 2.0,
                "avg_reward": 0.0,
            }
        )

    focus = scenario_focus_pressure(history, "combined_crisis")

    assert focus > 0.9
    assert choose_scenario_key(type("Args", (), {"scenario": None})(), 40, history) == "combined_crisis"


def torch_tensor(values):
    import torch

    return torch.tensor(values, dtype=torch.float32)
