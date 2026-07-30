#!/usr/bin/env python3
"""Plot PPO / SAC / DreamerV3 learning curves at a matched interaction budget.

Reads the SB3 ``progress.csv`` files written during PPO and SAC training (via the
trainers' ``--logdir`` flag) and DreamerV3's ``metrics.jsonl``, and draws one
panel per agent of episode return vs environment steps. This is the figure that
answers the reviewer's fairness concern directly: every learned agent is trained
to the same ~1M-step budget and its curve is shown to plateau.

Usage (from simulation_studio/simulator/):
    python plot_sb3_metrics.py \
        --ppo sb3_runs/ppo_1M/progress.csv \
        --sac sb3_runs/sac_1M/progress.csv \
        --dreamer dreamerv3_runs/m1_continuous_run8_kpi_aligned/metrics.jsonl \
        --output results/learning_curves_1M.png
"""
from __future__ import annotations

import argparse
import csv
import os
import tempfile
from pathlib import Path

CACHE_ROOT = Path(tempfile.gettempdir()) / "pios_sb3_plot_cache"
CACHE_ROOT.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(CACHE_ROOT / "matplotlib"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from plot_dreamerv3_metrics import load_metric_series, mean_downsample_points


def read_sb3_curve(path: Path) -> tuple[list[float], list[float]]:
    """Return (steps, ep_rew_mean) from an SB3 progress.csv, dropping blanks."""
    steps, rew = [], []
    if not path or not Path(path).exists():
        return steps, rew
    with Path(path).open(newline="") as fh:
        for row in csv.DictReader(fh):
            s = row.get("time/total_timesteps")
            r = row.get("rollout/ep_rew_mean")
            if s in (None, "") or r in (None, ""):
                continue
            try:
                steps.append(float(s))
                rew.append(float(r))
            except ValueError:
                continue
    return steps, rew


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ppo", type=Path, default=Path("sb3_runs/ppo_1M/progress.csv"))
    ap.add_argument("--sac", type=Path, default=Path("sb3_runs/sac_1M/progress.csv"))
    ap.add_argument("--dreamer", type=Path,
                    default=Path("dreamerv3_runs/m1_continuous_run8_kpi_aligned/metrics.jsonl"))
    ap.add_argument("--output", type=Path, default=Path("results/learning_curves_1M.png"))
    ap.add_argument("--dpi", type=int, default=160)
    args = ap.parse_args()

    panels = []  # (title, ylabel, steps, values)

    ppo_s, ppo_r = read_sb3_curve(args.ppo)
    if ppo_s:
        panels.append(("PPO (continuous)", "Episode return", ppo_s, ppo_r))

    sac_s, sac_r = read_sb3_curve(args.sac)
    if sac_s:
        panels.append(("SAC", "Episode return", sac_s, sac_r))

    if args.dreamer and args.dreamer.exists():
        series = load_metric_series(args.dreamer)
        pts = series.get("episode/score")
        if pts:
            pts = mean_downsample_points(pts, max_points=250)
            d_s = [s for s, _ in pts]
            d_v = [v for _, v in pts]
            panels.append(("DreamerV3", "Episode score", d_s, d_v))

    if not panels:
        raise SystemExit("No learning-curve data found (train with --logdir first).")

    fig, axes = plt.subplots(1, len(panels), figsize=(6.0 * len(panels), 4.6), squeeze=False)
    for ax, (title, ylabel, steps, values) in zip(axes.flatten(), panels):
        ax.plot(steps, values, color="#1f77b4", linewidth=1.8)
        ax.set_title(title, fontweight="bold")
        ax.set_xlabel("Environment steps")
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.25)

    fig.suptitle("Learning curves at a matched ~1M-step interaction budget", fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=args.dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved learning-curve figure -> {args.output}  ({len(panels)} panels)")


if __name__ == "__main__":
    main()
