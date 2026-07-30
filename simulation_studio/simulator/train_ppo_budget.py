"""
train_ppo_budget.py — Train a budget-aware PPO agent on the fast blood supply env.

The agent learns to balance supply-chain performance against frugal budget usage.
Supports multiple budget levels for curriculum or ablation experiments.

Usage:
    # Train with default budget (6500)
    python train_ppo_budget.py

    # Train with tight budget
    python train_ppo_budget.py --budget 4000

    # Train with curriculum (gradually tighten budget)
    python train_ppo_budget.py --budget-curriculum 8000,6500,5000,4000

    # Longer training
    python train_ppo_budget.py --timesteps 500000 --budget 5000
"""

import argparse

from fast_env_budget import FastBloodBudgetEnv


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train budget-aware PPO on fast blood supply env."
    )
    parser.add_argument("--timesteps", type=int, default=200_000)
    parser.add_argument("--budget", type=float, default=6500.0, help="Episode budget")
    parser.add_argument(
        "--budget-curriculum",
        type=str,
        default=None,
        help="Comma-separated budget levels for curriculum training "
        "(trains on each sequentially)",
    )
    parser.add_argument(
        "--save",
        type=str,
        default="ppo_budget_model",
        help="Save path for trained model",
    )
    parser.add_argument("--envs", type=int, default=8, help="Number of parallel envs")
    parser.add_argument(
        "--episode-steps",
        type=int,
        default=28,
        help="Steps per episode in the fast env",
    )
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--n-steps", type=int, default=1024)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--ent-coef", type=float, default=0.10)
    parser.add_argument("--verbose", type=int, default=1)
    return parser.parse_args()


def make_env(budget, episode_steps):
    """Returns a factory function that creates a single FastBloodBudgetEnv."""

    def _make():
        return FastBloodBudgetEnv(
            n_envs=1, episode_steps=episode_steps, episode_budget=budget
        )

    return _make


def main():
    args = parse_args()

    # Try to import SB3, fall back to custom PPO
    try:
        from stable_baselines3 import PPO
        from stable_baselines3.common.vec_env import DummyVecEnv

        use_sb3 = True
    except ImportError:
        from ppo_agent import PPO

        use_sb3 = False

    if args.budget_curriculum:
        # Curriculum training: train on progressively tighter budgets
        budgets = [float(b.strip()) for b in args.budget_curriculum.split(",")]
        steps_per_stage = args.timesteps // len(budgets)

        model = None
        for i, budget in enumerate(budgets):
            print(f"\n{'=' * 60}")
            print(
                f"Budget curriculum stage {i + 1}/{len(budgets)}: budget={budget:.0f}"
            )
            print(f"{'=' * 60}")

            if use_sb3:
                env = DummyVecEnv(
                    [make_env(budget, args.episode_steps) for _ in range(args.envs)]
                )
                if model is None:
                    model = PPO(
                        "MlpPolicy",
                        env,
                        learning_rate=args.lr,
                        n_steps=args.n_steps,
                        batch_size=args.batch_size,
                        gamma=0.99,
                        gae_lambda=0.95,
                        ent_coef=args.ent_coef,
                        verbose=args.verbose,
                    )
                else:
                    model.set_env(env)
                model.learn(total_timesteps=steps_per_stage, reset_num_timesteps=False)
            else:
                env = FastBloodBudgetEnv(
                    n_envs=1,
                    episode_steps=args.episode_steps,
                    episode_budget=budget,
                )
                if model is None:
                    model = PPO(
                        "MlpPolicy",
                        env,
                        learning_rate=args.lr,
                        n_steps=args.n_steps,
                        batch_size=args.batch_size,
                        verbose=args.verbose,
                    )
                else:
                    model.set_env(env)
                model.learn(total_timesteps=steps_per_stage)

        model.save(args.save)
        print(f"\nBudget-curriculum model saved \u2192 {args.save}")
    else:
        # Standard single-budget training
        if use_sb3:
            env = DummyVecEnv(
                [make_env(args.budget, args.episode_steps) for _ in range(args.envs)]
            )
            model = PPO(
                "MlpPolicy",
                env,
                learning_rate=args.lr,
                n_steps=args.n_steps,
                batch_size=args.batch_size,
                gamma=0.99,
                gae_lambda=0.95,
                ent_coef=args.ent_coef,
                verbose=args.verbose,
            )
        else:
            env = FastBloodBudgetEnv(
                n_envs=1,
                episode_steps=args.episode_steps,
                episode_budget=args.budget,
            )
            model = PPO(
                "MlpPolicy",
                env,
                learning_rate=args.lr,
                n_steps=args.n_steps,
                batch_size=args.batch_size,
                verbose=args.verbose,
            )

        model.learn(total_timesteps=args.timesteps)
        model.save(args.save)
        print(f"\nBudget-aware PPO model saved \u2192 {args.save}")


if __name__ == "__main__":
    main()
