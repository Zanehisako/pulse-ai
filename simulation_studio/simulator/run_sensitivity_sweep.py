#!/usr/bin/env python3
"""Short reward/delay sensitivity sweep for the blood-supply strategy comparison.

Runs ``evaluate_strategies.py`` under a small grid of reward-weight and
action-delay/ramp perturbations (all no-ops at their defaults) and tabulates how
the shortage / wastage / action-cost outcomes of each strategy shift. This
addresses the reviewer ask: "how sensitive are the conclusions to the reward
weights and the intervention delay/ramp parameters?"

Each grid cell is one evaluate_strategies.py subprocess; the agents/checkpoints
are loaded the same way as the canonical run (set PIOS_PPO_CONTINUOUS_CHECKPOINT
/ PIOS_SAC_CHECKPOINT in the environment to use the 1M-step agents).

Usage (from simulation_studio/simulator/, after training finishes):
    python run_sensitivity_sweep.py \
        --dreamerv3-checkpoint dreamerv3_runs/m1_continuous_run8_kpi_aligned \
        --runs 5
"""
from __future__ import annotations

import argparse
import csv
import statistics
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

# Grid of perturbations. Each entry: (label, extra CLI flags). The "default"
# cell uses no overrides and reproduces the canonical configuration.
SWEEP_CONFIGS = [
    ("default", []),
    ("delay_x0.5", ["--delay-scale", "0.5", "--ramp-scale", "0.5"]),
    ("delay_x2.0", ["--delay-scale", "2.0", "--ramp-scale", "2.0"]),
    ("safety_x1.5", ["--reward-safety-scale", "1.5"]),
    ("waste_x1.5", ["--reward-waste-scale", "1.5"]),
]

# Strategies and scenarios kept small — a sensitivity study, not the canonical run.
DEFAULT_STRATEGIES = ["baseline", "ppo_continuous", "sac_continuous", "dreamerv3_official"]
DEFAULT_SCENARIOS = ["baseline", "demand_surge", "transport_disruption", "combined_crisis"]
METRICS = ["shortage_rate", "total_expired", "active_action_cost"]


def run_cell(label, flags, args) -> Path:
    prefix = f"results/sweep_{label}"
    cmd = [
        sys.executable, "evaluate_strategies.py",
        "--runs", str(args.runs), "--seed-start", str(args.seed_start),
        "--dreamerv3-checkpoint", args.dreamerv3_checkpoint,
        "--output-prefix", prefix,
    ]
    for s in args.strategy:
        cmd += ["--strategy", s]
    for s in args.scenario:
        cmd += ["--scenario", s]
    cmd += flags
    print(f"\n=== sweep cell: {label}  ({' '.join(flags) or 'canonical'}) ===")
    subprocess.run(cmd, check=True)
    return Path(f"{prefix}_detail.csv")


def summarize(detail_path: Path) -> dict:
    """Return {(strategy, metric): mean across all swept scenarios/seeds}."""
    by_key: dict[tuple[str, str], list[float]] = defaultdict(list)
    with detail_path.open(newline="") as fh:
        for row in csv.DictReader(fh):
            for m in METRICS:
                v = row.get(m)
                if v not in (None, ""):
                    by_key[(row["strategy"], m)].append(float(v))
    return {k: statistics.mean(v) for k, v in by_key.items() if v}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dreamerv3-checkpoint",
                    default="dreamerv3_runs/m1_continuous_run8_kpi_aligned")
    ap.add_argument("--runs", type=int, default=5, help="Seeds per cell (default: 5).")
    ap.add_argument("--seed-start", type=int, default=100)
    ap.add_argument("--strategy", action="append", default=None)
    ap.add_argument("--scenario", action="append", default=None)
    ap.add_argument("--out", default="results/sensitivity_summary.csv")
    args = ap.parse_args()
    args.strategy = args.strategy or DEFAULT_STRATEGIES
    args.scenario = args.scenario or DEFAULT_SCENARIOS

    cell_summaries: dict[str, dict] = {}
    for label, flags in SWEEP_CONFIGS:
        detail = run_cell(label, flags, args)
        cell_summaries[label] = summarize(detail)

    # Write a tidy long-format summary and print a compact per-metric table.
    out_rows = []
    for label, summ in cell_summaries.items():
        for (strategy, metric), value in summ.items():
            out_rows.append({
                "config": label, "strategy": strategy,
                "metric": metric, "mean": f"{value:.4f}",
            })
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["config", "strategy", "metric", "mean"])
        w.writeheader()
        w.writerows(out_rows)

    print("\n" + "=" * 72)
    print("SENSITIVITY SUMMARY (mean across swept scenarios)")
    print("=" * 72)
    for metric in METRICS:
        print(f"\n{metric}:")
        header = f"  {'strategy':22s}" + "".join(f"{lbl:>13s}" for lbl, _ in SWEEP_CONFIGS)
        print(header)
        for strat in args.strategy:
            cells = "".join(
                f"{cell_summaries[lbl].get((strat, metric), float('nan')):>13.2f}"
                for lbl, _ in SWEEP_CONFIGS
            )
            print(f"  {strat:22s}{cells}")
    print(f"\nWrote tidy summary -> {args.out}")


if __name__ == "__main__":
    main()
