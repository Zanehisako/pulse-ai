"""
train_ppo.py  –  Train and evaluate the PPO agent for the blood supply chain.

Usage
-----
  # Train (uses NumPy PPO fallback when SB3 is not installed):
  python train_ppo.py --timesteps 200000 --save ppo_blood_model

  # Install SB3 for the full implementation:
  pip install stable-baselines3 gymnasium
  python train_ppo.py --timesteps 500000 --save ppo_blood_model

  # Evaluate a saved model without re-training:
  python train_ppo.py --eval --model ppo_blood_model --seed 99

  # Train then immediately evaluate:
  python train_ppo.py --timesteps 200000 --save ppo_blood_model --eval
"""

import argparse
import os
import sys


def _parse_scenario_keys(
    raw: str, available_scenarios: dict[str, object]
) -> list[str]:
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


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Train / evaluate a PPO agent for the blood supply chain."
    )
    p.add_argument(
        "--timesteps",
        type=int,
        default=200_000,
        help="Total training timesteps (default: 200 000)",
    )
    p.add_argument(
        "--save",
        type=str,
        default="ppo_blood_model",
        help="Path to save the trained model",
    )
    p.add_argument(
        "--eval", action="store_true", help="Run an evaluation episode after training"
    )
    p.add_argument(
        "--model",
        type=str,
        default=None,
        help="Load a saved model for evaluation only (skip training)",
    )
    p.add_argument(
        "--scenario",
        type=str,
        default="baseline",
        help="Scenario key, comma-separated list, or 'all'",
    )
    p.add_argument(
        "--eval-scenario",
        type=str,
        default=None,
        help="Scenario key used for evaluation when training on multiple scenarios",
    )
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--ep-hours",
        type=float,
        default=168.0,
        help="Sim-hours per training episode (default: 168 = 1 week)",
    )
    p.add_argument(
        "--step-hours",
        type=float,
        default=6.0,
        help="Sim-hours per env.step() (default: 6)",
    )
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--n-steps", type=int, default=512)
    p.add_argument("--batch", type=int, default=64)
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument(
        "--hidden", type=int, default=64, help="MLP hidden layer width (default: 64)"
    )
    p.add_argument(
        "--envs",
        type=int,
        default=0,
        help="Number of parallel training envs (0 = auto-tune, default: 0)",
    )
    p.add_argument(
        "--vec-env",
        type=str,
        default="auto",
        choices=["auto", "dummy", "subproc"],
        help="Vectorized env backend when --envs > 1 (default: auto)",
    )
    p.add_argument(
        "--continuous",
        action="store_true",
        help="Train a continuous (Box) action-space PPO matching the paper's "
        "ppo_continuous agent (default: discrete).",
    )
    p.add_argument(
        "--logdir",
        type=str,
        default=None,
        help="Directory for SB3 CSV learning-curve logs (progress.csv).",
    )
    p.add_argument(
        "--online-city-graph",
        action="store_true",
        help="Fetch the live OpenStreetMap city graph instead of using the offline stub",
    )
    p.add_argument(
        "--verbose", type=int, default=1, help="0=silent, 1=progress, 2=all losses"
    )
    return p.parse_args()


def main() -> None:
    args = _parse_args()

    # ── Import simulation modules ────────────────────────────────────────
    try:
        from engine import (  # type: ignore
            generate_report,
            run_with_ppo_agent,
            train_ppo_agent,
        )
        from scenarios import SCENARIOS  # type: ignore
    except ImportError as exc:
        print(f"[train_ppo] Cannot import simulation modules: {exc}")
        print("  Make sure engine.py, ppo_agent.py, ppo_env.py, ppo_shared.py")
        print(
            "  and their dependencies (core, scenarios, calibration) are on PYTHONPATH."
        )
        sys.exit(1)

    # ── Resolve scenario ─────────────────────────────────────────────────
    try:
        scenario_keys = _parse_scenario_keys(args.scenario, SCENARIOS)
    except ValueError as exc:
        print(f"[train_ppo] {exc}")
        sys.exit(1)

    if args.eval_scenario is not None and args.eval_scenario not in SCENARIOS:
        print(f"[train_ppo] Unknown eval scenario '{args.eval_scenario}'.")
        print(f"  Available: {sorted(SCENARIOS.keys())}")
        sys.exit(1)

    train_params = [SCENARIOS[key] for key in scenario_keys]
    params = train_params[0]
    eval_scenario_key = args.eval_scenario or (
        scenario_keys[0] if len(scenario_keys) == 1 else "baseline"
    )
    eval_params = SCENARIOS[eval_scenario_key]

    # ── Road graph ───────────────────────────────────────────────────────
    try:
        from core import load_city_graph, offline_city_graph  # type: ignore

        if args.online_city_graph:
            G, north, south, east, west = load_city_graph()
        else:
            os.environ.setdefault("PIOS_SIM_OFFLINE", "1")
            G, north, south, east, west = offline_city_graph()
            if args.verbose >= 1:
                print("[train_ppo] Using offline city graph stub for faster startup.")
    except Exception as exc:
        import types

        print(f"[train_ppo] Road graph unavailable ({exc}) – using bounding-box stub.")
        G = types.SimpleNamespace()
        north, south, east, west = 46.86, 46.77, -71.18, -71.32

    # ── Training ─────────────────────────────────────────────────────────

    agent = None
    if args.model is None:
        print(f"\n[train_ppo] Training for {args.timesteps:,} timesteps …")
        print(f"[train_ppo] Training scenarios: {scenario_keys}")
        agent = train_ppo_agent(
            params=params,
            G=G,
            north=north,
            south=south,
            east=east,
            west=west,
            total_timesteps=args.timesteps,
            episode_hours=args.ep_hours,
            step_hours=args.step_hours,
            learning_rate=args.lr,
            n_steps=args.n_steps,
            batch_size=args.batch,
            n_epochs=args.epochs,
            hidden_dim=args.hidden,
            save_path=args.save,
            verbose=args.verbose,
            seed=args.seed,
            training_scenarios=train_params,
            num_envs=args.envs,
            vec_env=args.vec_env,
            action_mode="continuous" if args.continuous else "discrete",
            logdir=args.logdir,
        )
        print(f"[train_ppo] Training complete. Model saved → '{args.save}'.")

    # ── Evaluation ───────────────────────────────────────────────────────
    if args.eval or args.model is not None:
        model_path = args.model  # None means use the just-trained in-memory agent
        eval_seed = args.seed + 1 if agent is not None else args.seed

        print(
            f"\n[train_ppo] Running evaluation episode on '{eval_scenario_key}' "
            f"(seed={eval_seed}) …"
        )
        state = run_with_ppo_agent(
            params=eval_params,
            G=G,
            north=north,
            south=south,
            east=east,
            west=west,
            agent=agent,  # None → loaded from model_path
            model_path=model_path,
            seed=eval_seed,
            enable_logs=args.verbose >= 1,
            fast_mode=False,
            ppo_step_hours=args.step_hours,
        )

        report_path = f"ppo_eval_report_{eval_scenario_key}.txt"
        generate_report(state, output_path=report_path)
        shortage_pct = (
            state.total_shortage_units / max(state.total_net_requested_units, 1) * 100
        )
        print(f"\n[train_ppo] Evaluation complete.")
        print(f"  Shortage rate : {shortage_pct:.1f}%")
        print(f"  Reward total  : {state.reward_breakdown.total:.1f}")
        print(f"  Report        : {report_path}")


if __name__ == "__main__":
    main()
