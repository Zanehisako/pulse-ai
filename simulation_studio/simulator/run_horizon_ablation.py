#!/usr/bin/env python3
"""Imagination-horizon ablation eval: evaluate every (H, seed) DreamerV3
checkpoint and tabulate crisis-scenario shortage vs H.

Multi-seed design: checkpoints live at dreamerv3_runs/ablation_H{H}_s{seed}
(trained by train_horizon_ablation.sh at 500K steps each). imag_length is
training-only, so we just eval each checkpoint with the unchanged eval path.
Run on JAX CPU (Metal crashes on checkpoint load).

Usage (from simulation_studio/simulator/, after training):
    JAX_PLATFORMS=cpu python run_horizon_ablation.py --runs 15
Then build the figure:
    python plot_horizon_ablation.py
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

DEFAULT_SCENARIOS = ["baseline", "demand_surge", "transport_disruption", "combined_crisis"]
DEFAULT_HORIZONS = [1, 5, 15, 30]
DEFAULT_SEEDS = [0, 1, 2]


def logdir_for(h: int, seed: int) -> Path:
    return Path(f"dreamerv3_runs/ablation_H{h}_s{seed}")


def has_checkpoint(logdir: Path) -> bool:
    return (logdir / "ckpt").exists()


def eval_cell(h: int, seed: int, args) -> Path | None:
    logdir = logdir_for(h, seed)
    out_prefix = f"results/ablation_H{h}_s{seed}"
    detail = Path(f"{out_prefix}_detail.csv")
    if not has_checkpoint(logdir):
        print(f"[H={h} s={seed}] no checkpoint at {logdir} — skipping.")
        return None
    if detail.exists() and not args.force:
        print(f"[H={h} s={seed}] {detail} exists — skipping (use --force).")
        return detail
    cmd = [
        sys.executable, "evaluate_strategies.py",
        "--strategy", "dreamerv3_official",
        "--dreamerv3-checkpoint", str(logdir),
        "--runs", str(args.runs), "--seed-start", str(args.seed_start),
        "--output-prefix", out_prefix,
    ]
    for sc in args.scenario:
        cmd += ["--scenario", sc]
    env = {**os.environ, "JAX_PLATFORMS": "cpu", "PIOS_SIM_OFFLINE": "1",
           "MPLCONFIGDIR": "/tmp/matplotlib"}
    print(f"\n=== [H={h} s={seed}] evaluating {logdir} ===")
    subprocess.run(cmd, check=True, env=env)
    return detail


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--runs", type=int, default=15, help="Eval seeds per cell (default 15).")
    ap.add_argument("--seed-start", type=int, default=100)
    ap.add_argument("--scenario", action="append", default=None)
    ap.add_argument("--horizon", action="append", type=int, default=None)
    ap.add_argument("--train-seed", action="append", type=int, default=None,
                    help="Training seed(s) to evaluate; repeat. Default: 0 1 2.")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    args.scenario = args.scenario or DEFAULT_SCENARIOS
    horizons = args.horizon or DEFAULT_HORIZONS
    seeds = args.train_seed or DEFAULT_SEEDS

    done, missing = 0, []
    for seed in seeds:
        for h in horizons:
            if not has_checkpoint(logdir_for(h, seed)):
                missing.append((h, seed))
                continue
            eval_cell(h, seed, args)
            done += 1

    print(f"\nEvaluated {done} (H,seed) cells.")
    if missing:
        print(f"Missing checkpoints (train them first): "
              f"{[f'H{h}_s{s}' for h, s in missing]}")
    print("Build the figure with plot_horizon_ablation.py")


if __name__ == "__main__":
    main()
