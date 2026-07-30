from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

from plot_dreamerv3_metrics import summarize_logdir_scores


SIMULATOR_DIR = Path(__file__).resolve().parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare evaluated RL strategies against the logged DreamerV3 "
            "`m1_continuous_run8_kpi_aligned` training run."
        )
    )
    parser.add_argument(
        "--strategy-summary",
        type=Path,
        required=True,
        help="CSV summary produced by evaluate_strategies.py.",
    )
    parser.add_argument(
        "--dreamer-run-dir",
        type=Path,
        default=SIMULATOR_DIR / "dreamerv3_runs" / "m1_continuous_run8_kpi_aligned",
        help="DreamerV3 run directory containing metrics.jsonl and scores.jsonl.",
    )
    parser.add_argument(
        "--strategy",
        action="append",
        default=None,
        help="Restrict comparison to one or more strategy keys.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=SIMULATOR_DIR / "results" / "rl_vs_dreamerv3_run8.csv",
        help="Output CSV path for the comparison table.",
    )
    return parser.parse_args()


def load_strategy_summary(path: Path) -> list[dict[str, str]]:
    with Path(path).open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def aggregate_strategy_scores(
    rows: list[dict[str, str]],
    *,
    allowed_strategies: set[str] | None = None,
) -> list[dict[str, float | str]]:
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        strategy = str(row.get("strategy", ""))
        if allowed_strategies and strategy not in allowed_strategies:
            continue
        grouped[strategy].append(row)

    aggregated: list[dict[str, float | str]] = []
    for strategy, group in sorted(grouped.items()):
        scores = [float(row["episode_score_mean"]) for row in group]
        shortages = [float(row["shortage_rate_mean"]) for row in group]
        rewards = [float(row["reward_total_mean"]) for row in group]
        aggregated.append(
            {
                "benchmark": strategy,
                "source": "evaluate_strategies",
                "count": float(len(group)),
                "episode_score_mean": float(sum(scores) / len(scores)),
                "shortage_rate_mean": float(sum(shortages) / len(shortages)),
                "reward_total_mean": float(sum(rewards) / len(rewards)),
            }
        )
    return aggregated


def save_csv(rows: list[dict[str, float | str]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    for row in rows:
        for key in row.keys():
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    summary_rows = load_strategy_summary(args.strategy_summary)
    allowed = set(args.strategy) if args.strategy else None
    benchmark_rows = aggregate_strategy_scores(summary_rows, allowed_strategies=allowed)

    dreamer_stats = summarize_logdir_scores(args.dreamer_run_dir)
    if dreamer_stats:
        benchmark_rows.append(
            {
                "benchmark": "dreamerv3_m1_run8",
                "source": "dreamerv3_training_log",
                "count": dreamer_stats["count"],
                "episode_score_mean": dreamer_stats["mean"],
                "shortage_rate_mean": "",
                "reward_total_mean": "",
                "score_min": dreamer_stats["min"],
                "score_max": dreamer_stats["max"],
                "score_last": dreamer_stats["last"],
                "score_tail_mean": dreamer_stats["tail_mean"],
            }
        )

    if not benchmark_rows:
        raise SystemExit("No strategy rows available for comparison.")

    save_csv(benchmark_rows, args.output)
    print(f"Saved benchmark comparison to {args.output}")


if __name__ == "__main__":
    main()
