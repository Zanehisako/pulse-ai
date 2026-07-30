from __future__ import annotations

import argparse
import copy
import os
import sys


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
        description="Train or evaluate a SAC agent for the blood supply simulator."
    )
    parser.add_argument("--timesteps", type=int, default=200_000)
    parser.add_argument("--save", type=str, default="sac_blood_model")
    parser.add_argument("--eval", action="store_true")
    parser.add_argument("--model", type=str, default=None)
    parser.add_argument("--scenario", type=str, default="baseline")
    parser.add_argument("--eval-scenario", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--ep-hours", type=float, default=168.0)
    parser.add_argument("--step-hours", type=float, default=6.0)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--buffer-size", type=int, default=100_000)
    parser.add_argument("--learning-starts", type=int, default=1_000)
    parser.add_argument("--tau", type=float, default=0.005)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gradient-steps", type=int, default=1)
    parser.add_argument("--hidden", type=int, default=256)
    parser.add_argument(
        "--logdir",
        type=str,
        default=None,
        help="Directory for SB3 CSV learning-curve logs (progress.csv).",
    )
    parser.add_argument("--online-city-graph", action="store_true")
    parser.add_argument("--verbose", type=int, default=1)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    try:
        from core import load_city_graph, offline_city_graph
        from engine import (
            generate_report,
            load_sac_agent,
            run_scenario,
            train_sac_agent,
        )
        from scenarios import SCENARIOS
    except ImportError as exc:
        print(f"[train_sac] Cannot import simulator modules: {exc}")
        raise SystemExit(1) from exc

    try:
        scenario_keys = _parse_scenario_keys(args.scenario, SCENARIOS)
    except ValueError as exc:
        print(f"[train_sac] {exc}")
        raise SystemExit(2) from exc

    if args.eval_scenario is not None and args.eval_scenario not in SCENARIOS:
        print(f"[train_sac] Unknown eval scenario '{args.eval_scenario}'.")
        raise SystemExit(2)

    if args.online_city_graph:
        G, north, south, east, west = load_city_graph()
    else:
        os.environ.setdefault("PIOS_SIM_OFFLINE", "1")
        G, north, south, east, west = offline_city_graph()
        if args.verbose >= 1:
            print("[train_sac] Using offline city graph stub for faster startup.")

    training_scenarios = [copy.deepcopy(SCENARIOS[key]) for key in scenario_keys]
    default_params = copy.deepcopy(training_scenarios[0])
    eval_scenario_key = args.eval_scenario or (
        scenario_keys[0] if len(scenario_keys) == 1 else "baseline"
    )
    eval_params = copy.deepcopy(SCENARIOS[eval_scenario_key])
    eval_params.strategy_key = "sac_continuous"

    agent = None
    if args.model is None:
        print(f"[train_sac] Training SAC for {args.timesteps:,} timesteps …")
        agent = train_sac_agent(
            params=default_params,
            G=G,
            north=north,
            south=south,
            east=east,
            west=west,
            total_timesteps=args.timesteps,
            episode_hours=args.ep_hours,
            step_hours=args.step_hours,
            learning_rate=args.lr,
            batch_size=args.batch_size,
            buffer_size=args.buffer_size,
            learning_starts=args.learning_starts,
            tau=args.tau,
            gamma=args.gamma,
            gradient_steps=args.gradient_steps,
            hidden_dim=args.hidden,
            save_path=args.save,
            verbose=args.verbose,
            seed=args.seed,
            training_scenarios=training_scenarios,
            logdir=args.logdir,
        )

    if args.eval or args.model is not None:
        if agent is None:
            model_path = args.model
            if model_path is None:
                raise SystemExit("[train_sac] --model is required when --eval-only.")
            agent = load_sac_agent(
                model_path,
                params=copy.deepcopy(SCENARIOS["baseline"]),
                G=G,
                north=north,
                south=south,
                east=east,
                west=west,
            )

        state = run_scenario(
            eval_params,
            G,
            north,
            south,
            east,
            west,
            seed=args.seed + 1,
            enable_logs=args.verbose >= 1,
            fast_mode=False,
            ppo_agent=agent,
            ppo_step_hours=args.step_hours,
        )
        report_path = f"sac_eval_report_{eval_scenario_key}.txt"
        generate_report(state, output_path=report_path)
        print(f"[train_sac] Saved evaluation report to {report_path}")


if __name__ == "__main__":
    main()
