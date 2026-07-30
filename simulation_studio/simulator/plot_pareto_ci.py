#!/usr/bin/env python3
"""Plot per-scenario Pareto fronts with 95% CIs for wastage and action cost.

Reads the per-seed detail CSV from ``evaluate_strategies.py`` and for each
scenario builds a Pareto front on (total_expired, active_action_cost) axes with
errorbars showing 95% confidence intervals. This directly supports or qualifies
the paper's claim that DreamerV3 is "Pareto-optimal in all nine scenarios."

Usage (from simulation_studio/simulator/):
    python plot_pareto_ci.py --detail results/strategy_eval_thesis_detail.csv \
        --output-prefix results/pareto_ci
"""
from __future__ import annotations

import argparse
import csv
import math
import os
import sys
import tempfile
from collections import defaultdict
from pathlib import Path

import numpy as np

try:
    from scipy import stats as _sps
    _HAVE_SCIPY = True
except ImportError:
    _HAVE_SCIPY = False

# Set matplotlib backend before importing pyplot
CACHE_ROOT = Path(tempfile.gettempdir()) / "pios_pareto_cache"
CACHE_ROOT.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(CACHE_ROOT / "matplotlib"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from evaluate_strategies import pareto_frontier_mask
from scenarios import SCENARIOS


def t_ci(values: np.ndarray, alpha: float = 0.05) -> tuple[float, float, float]:
    """Return (mean, ci_low, ci_high) using a Student-t interval."""
    n = values.size
    mean = float(np.mean(values)) if n else float("nan")
    if n < 2:
        return mean, mean, mean
    sd = float(np.std(values, ddof=1))
    se = sd / math.sqrt(n)
    if _HAVE_SCIPY:
        crit = float(_sps.t.ppf(1.0 - alpha / 2.0, df=n - 1))
    else:
        crit = 1.959963984540054  # z for 95% normal
    return mean, mean - crit * se, mean + crit * se


def read_detail(path: Path) -> list[dict]:
    with path.open("r", newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--detail",
        type=Path,
        default=Path("results/strategy_eval_thesis_detail.csv"),
        help="Per-seed detail CSV from evaluate_strategies.py.",
    )
    parser.add_argument(
        "--output-prefix",
        type=str,
        default="results/pareto_ci",
        help="Output prefix; figures written to <prefix>_<scenario>.<ext>.",
    )
    parser.add_argument(
        "--dpi", type=int, default=150, help="Figure DPI (default: 150)."
    )
    args = parser.parse_args()

    if not args.detail.exists():
        sys.exit(f"Detail CSV not found: {args.detail}")

    rows = read_detail(args.detail)
    if not rows:
        sys.exit(f"No rows in {args.detail}")

    # Group by (strategy, scenario) and collect per-seed vectors
    index: dict[tuple[str, str], dict[str, dict]] = defaultdict(dict)
    for r in rows:
        index[(r["strategy"], r["scenario"])][r["seed"]] = r

    strategies = sorted({r["strategy"] for r in rows})
    scenarios = sorted({r["scenario"] for r in rows})

    output_prefix = Path(args.output_prefix)
    output_prefix.parent.mkdir(parents=True, exist_ok=True)

    for scenario in scenarios:
        print(f"[{scenario}] Building Pareto front with CIs ...")

        # Aggregate per-strategy means and CIs
        points = []
        labels = []
        errors_x = []
        errors_y = []

        for strat in strategies:
            seedmap = index.get((strat, scenario), {})
            if not seedmap:
                continue

            expired_vals = np.array([
                float(d["total_expired"]) for d in seedmap.values()
                if d.get("total_expired") not in (None, "")
            ])
            cost_vals = np.array([
                float(d["active_action_cost"]) for d in seedmap.values()
                if d.get("active_action_cost") not in (None, "")
            ])

            if expired_vals.size == 0 or cost_vals.size == 0:
                continue

            exp_mean, exp_low, exp_high = t_ci(expired_vals)
            cost_mean, cost_low, cost_high = t_ci(cost_vals)

            points.append([exp_mean, cost_mean])
            labels.append(strat)
            errors_x.append([exp_mean - exp_low, exp_high - exp_mean])
            errors_y.append([cost_mean - cost_low, cost_high - cost_mean])

        if len(points) < 2:
            print(f"  Skipped (insufficient data).")
            continue

        points_arr = np.array(points)
        errors_x_arr = np.array(errors_x).T  # shape (2, n_strategies)
        errors_y_arr = np.array(errors_y).T

        # Compute Pareto frontier on the mean points (min both axes)
        frontier_mask = pareto_frontier_mask(points_arr, goals=["min", "min"])

        # Plot
        fig, ax = plt.subplots(figsize=(9, 6))

        # Non-frontier points
        non_frontier = ~frontier_mask
        if np.any(non_frontier):
            ax.errorbar(
                points_arr[non_frontier, 0],
                points_arr[non_frontier, 1],
                xerr=errors_x_arr[:, non_frontier],
                yerr=errors_y_arr[:, non_frontier],
                fmt='o',
                color='#94a3b8',
                markersize=7,
                alpha=0.7,
                label='Non-Pareto',
                capsize=3,
                elinewidth=1.5,
            )

        # Frontier points
        if np.any(frontier_mask):
            ax.errorbar(
                points_arr[frontier_mask, 0],
                points_arr[frontier_mask, 1],
                xerr=errors_x_arr[:, frontier_mask],
                yerr=errors_y_arr[:, frontier_mask],
                fmt='D',
                color='#0ea5e9',
                markersize=9,
                label='Pareto Frontier',
                capsize=4,
                elinewidth=2,
            )

        # Label strategies (offset slightly for readability)
        for i, label in enumerate(labels):
            offset_x = 3
            offset_y = 3 if i % 2 == 0 else -8
            ax.annotate(
                label,
                (points_arr[i, 0], points_arr[i, 1]),
                textcoords="offset points",
                xytext=(offset_x, offset_y),
                fontsize=8,
                alpha=0.8,
            )

        ax.set_xlabel("Total Expired Units (lower is better)", fontsize=11)
        ax.set_ylabel("Active Action Cost (lower is better)", fontsize=11)
        ax.set_title(
            f"{SCENARIOS[scenario].name}\nPareto Front: Wastage vs Action Cost (95% CI)",
            fontsize=12,
            fontweight='bold',
        )
        ax.legend(loc='best', fontsize=9)
        ax.grid(True, alpha=0.3)

        output_path = Path(f"{args.output_prefix}_{scenario}.png")
        fig.savefig(output_path, dpi=args.dpi, bbox_inches='tight')
        plt.close(fig)
        print(f"  Saved → {output_path}")

    print(f"\nGenerated {len(scenarios)} per-scenario Pareto plots with CIs.")


if __name__ == "__main__":
    main()
