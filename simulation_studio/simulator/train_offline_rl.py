from __future__ import annotations

import argparse
from pathlib import Path

from offline_rl import summarize_offline_dataset, train_cql, train_iql


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train an offline RL policy for the blood supply simulator."
    )
    parser.add_argument(
        "--algorithm",
        choices=["iql", "cql"],
        required=True,
        help="Offline RL algorithm to train.",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="Path to an `.npz` offline transition dataset.",
    )
    parser.add_argument(
        "--save",
        type=str,
        default=None,
        help="Output checkpoint path. Defaults to `<algorithm>_blood_model.pt`.",
    )
    parser.add_argument("--updates", type=int, default=2_000)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--log-every", type=int, default=200)
    parser.add_argument("--expectile", type=float, default=0.7)
    parser.add_argument("--beta", type=float, default=3.0)
    parser.add_argument("--tau", type=float, default=0.005)
    parser.add_argument("--alpha", type=float, default=0.2)
    parser.add_argument("--cql-alpha", type=float, default=1.0)
    parser.add_argument("--action-samples", type=int, default=10)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset_summary = summarize_offline_dataset(args.dataset)
    print(
        "[train_offline_rl] dataset="
        f"{int(dataset_summary['transitions'])} transitions "
        f"reward_mean={dataset_summary['reward_mean']:.4f} "
        f"reward_std={dataset_summary['reward_std']:.4f} "
        f"terminal_rate={dataset_summary['terminal_rate']:.4f}"
    )

    save_path = args.save
    if save_path is None:
        save_path = str(
            (Path(__file__).resolve().parent / f"{args.algorithm}_blood_model.pt").resolve()
        )

    if args.algorithm == "iql":
        checkpoint, stats = train_iql(
            args.dataset,
            save_path,
            updates=args.updates,
            batch_size=args.batch_size,
            hidden_dim=args.hidden_dim,
            learning_rate=args.lr,
            gamma=args.gamma,
            expectile=args.expectile,
            beta=args.beta,
            seed=args.seed,
            device=args.device,
            log_every=args.log_every,
        )
    else:
        checkpoint, stats = train_cql(
            args.dataset,
            save_path,
            updates=args.updates,
            batch_size=args.batch_size,
            hidden_dim=args.hidden_dim,
            learning_rate=args.lr,
            gamma=args.gamma,
            tau=args.tau,
            alpha=args.alpha,
            cql_alpha=args.cql_alpha,
            action_samples=args.action_samples,
            seed=args.seed,
            device=args.device,
            log_every=args.log_every,
        )

    print(f"[train_offline_rl] Saved {args.algorithm.upper()} checkpoint to {checkpoint}")
    for key, value in stats.items():
        print(f"[train_offline_rl] {key}={value:.6f}")


if __name__ == "__main__":
    main()
