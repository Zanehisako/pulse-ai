from __future__ import annotations

import argparse
import copy
import os
import sys
from pathlib import Path

import numpy as np


def _parse_scenario_keys(raw: str, available_scenarios: dict[str, object]) -> list[str]:
    value = raw.strip().lower()
    if value == "all":
        return list(available_scenarios.keys())

    scenario_keys = [item.strip() for item in raw.split(",") if item.strip()]
    invalid = [item for item in scenario_keys if item not in available_scenarios]
    if invalid:
        raise ValueError(
            f"Unknown scenarios: {invalid}. Available: {sorted(available_scenarios.keys())}"
        )
    if not scenario_keys:
        raise ValueError("At least one scenario is required.")
    return scenario_keys


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Collect an offline RL dataset from simulator rollouts."
    )
    parser.add_argument(
        "--output",
        type=str,
        default="offline_blood_dataset.npz",
        help="Output `.npz` path for collected transitions.",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=200,
        help="Number of episodes to collect.",
    )
    parser.add_argument(
        "--scenario",
        type=str,
        default="baseline",
        help="Scenario key, comma-separated list, or 'all'.",
    )
    parser.add_argument(
        "--source-policy",
        choices=["random", "ppo_continuous", "sac_continuous", "mixed"],
        default="mixed",
        help="Action source used during collection.",
    )
    parser.add_argument(
        "--expert-policy",
        choices=["ppo_continuous", "sac_continuous"],
        default="ppo_continuous",
        help="Expert policy used by `mixed` collection.",
    )
    parser.add_argument(
        "--expert-model",
        type=str,
        default=None,
        help="Optional explicit checkpoint path for the expert policy.",
    )
    parser.add_argument(
        "--noise-std",
        type=float,
        default=0.20,
        help="Gaussian noise added to expert actions for mixed collection.",
    )
    parser.add_argument(
        "--ep-hours",
        type=float,
        default=168.0,
        help="Simulation hours per episode.",
    )
    parser.add_argument(
        "--step-hours",
        type=float,
        default=6.0,
        help="Controller interval in simulation hours.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--online-city-graph",
        action="store_true",
        help="Fetch the live OpenStreetMap city graph instead of using the offline stub.",
    )
    return parser.parse_args()


def _load_sim_context(online_city_graph: bool):
    from core import load_city_graph, offline_city_graph

    if online_city_graph:
        return load_city_graph()
    os.environ.setdefault("PIOS_SIM_OFFLINE", "1")
    return offline_city_graph()


def _clip_action(action: np.ndarray) -> np.ndarray:
    return np.clip(np.asarray(action, dtype=np.float32), -1.0, 1.0)


def _load_expert_agent(
    policy_key: str,
    model_path: str | None,
    params,
    sim_context,
):
    from engine import load_learned_continuous_agent, resolve_learned_continuous_model_path

    resolved = model_path or resolve_learned_continuous_model_path(policy_key)
    if resolved is None:
        raise FileNotFoundError(
            f"No checkpoint available for expert policy {policy_key!r}."
        )
    G, north, south, east, west = sim_context
    return load_learned_continuous_agent(
        policy_key,
        resolved,
        params=params,
        G=G,
        north=north,
        south=south,
        east=east,
        west=west,
    )


def _select_action(
    obs: np.ndarray,
    env,
    *,
    source_policy: str,
    expert_agent,
    noise_std: float,
    rng: np.random.Generator,
) -> np.ndarray:
    if source_policy == "random":
        return _clip_action(env.action_space.sample())

    if source_policy in {"ppo_continuous", "sac_continuous"}:
        if expert_agent is None:
            raise ValueError(f"Expert agent is required for {source_policy}.")
        action, _ = expert_agent.predict(obs, deterministic=True)
        return _clip_action(action)

    if expert_agent is not None and rng.random() < 0.75:
        action, _ = expert_agent.predict(obs, deterministic=True)
        noisy = np.asarray(action, dtype=np.float32) + rng.normal(
            0.0,
            noise_std,
            size=np.asarray(action).shape,
        ).astype(np.float32)
        return _clip_action(noisy)

    return _clip_action(env.action_space.sample())


def main() -> None:
    args = parse_args()

    try:
        from ppo_env import BloodSupplyEnv
        from scenarios import SCENARIOS
    except ImportError as exc:
        print(f"[collect_offline_dataset] Cannot import simulator modules: {exc}")
        raise SystemExit(1) from exc

    try:
        scenario_keys = _parse_scenario_keys(args.scenario, SCENARIOS)
    except ValueError as exc:
        print(f"[collect_offline_dataset] {exc}")
        raise SystemExit(2) from exc

    sim_context = _load_sim_context(args.online_city_graph)
    scenario_templates = [copy.deepcopy(SCENARIOS[key]) for key in scenario_keys]
    base_params = copy.deepcopy(scenario_templates[0])
    env = BloodSupplyEnv(
        params=base_params,
        G=sim_context[0],
        north=sim_context[1],
        south=sim_context[2],
        east=sim_context[3],
        west=sim_context[4],
        episode_hours=args.ep_hours,
        step_hours=args.step_hours,
        seed=args.seed,
        scenario_templates=scenario_templates,
        action_mode="continuous",
        collect_diagnostics=False,
    )

    expert_agent = None
    if args.source_policy != "random":
        expert_policy = (
            args.expert_policy if args.source_policy == "mixed" else args.source_policy
        )
        expert_agent = _load_expert_agent(
            expert_policy,
            args.expert_model,
            base_params,
            sim_context,
        )

    rng = np.random.default_rng(args.seed)
    obs_rows: list[np.ndarray] = []
    action_rows: list[np.ndarray] = []
    reward_rows: list[float] = []
    next_obs_rows: list[np.ndarray] = []
    done_rows: list[float] = []
    timeout_rows: list[float] = []
    scenario_index_rows: list[int] = []

    for episode in range(args.episodes):
        reset_out = env.reset(seed=args.seed + episode)
        obs, info = reset_out if isinstance(reset_out, tuple) else (reset_out, {})
        done = False
        while not done:
            action = _select_action(
                obs,
                env,
                source_policy=args.source_policy,
                expert_agent=expert_agent,
                noise_std=args.noise_std,
                rng=rng,
            )
            next_obs, reward, terminated, truncated, info = env.step(action)
            obs_rows.append(np.asarray(obs, dtype=np.float32))
            action_rows.append(np.asarray(action, dtype=np.float32))
            reward_rows.append(float(reward))
            next_obs_rows.append(np.asarray(next_obs, dtype=np.float32))
            done_rows.append(float(bool(terminated or truncated)))
            timeout_rows.append(float(bool(truncated)))
            scenario_index_rows.append(int(info.get("scenario_index", 0)))
            obs = next_obs
            done = bool(terminated or truncated)

        if (episode + 1) % 25 == 0 or episode + 1 == args.episodes:
            print(
                "[collect_offline_dataset] "
                f"episodes={episode + 1}/{args.episodes} "
                f"transitions={len(obs_rows)}"
            )

    output_path = Path(args.output).expanduser()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        obs=np.asarray(obs_rows, dtype=np.float32),
        actions=np.asarray(action_rows, dtype=np.float32),
        rewards=np.asarray(reward_rows, dtype=np.float32),
        next_obs=np.asarray(next_obs_rows, dtype=np.float32),
        dones=np.asarray(done_rows, dtype=np.float32),
        timeouts=np.asarray(timeout_rows, dtype=np.float32),
        scenario_index=np.asarray(scenario_index_rows, dtype=np.int32),
    )
    print(
        "[collect_offline_dataset] "
        f"Saved {len(obs_rows)} transitions from {args.episodes} episodes to "
        f"{output_path}"
    )


if __name__ == "__main__":
    main()
