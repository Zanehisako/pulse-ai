#!/usr/bin/env python3
"""Plot the multi-seed imagination-horizon ablation: DreamerV3 shortage vs H,
aggregated over training seeds (mean +/- 95% CI across seeds), per scenario, with
the fair-budget PPO/SAC reactive baselines as reference lines.

Reads per-(H,seed) detail CSVs written by run_horizon_ablation.py
(results/ablation_H{H}_s{seed}_detail.csv) and the canonical comparison CSV
(results/strategy_eval_thesis_detail.csv) for the PPO/SAC reference shortages.
Each seed contributes ONE number per (H, scenario) = that checkpoint's mean
shortage over its eval seeds; the across-seed spread is the meaningful
uncertainty for "does horizon H reliably produce quality X".

Outputs results/horizon_ablation.png + results/horizon_ablation_summary.csv.
"""
from __future__ import annotations

import argparse
import csv
import math
import os
import tempfile
from collections import defaultdict
from pathlib import Path

try:
    from scipy import stats as _sps
    _HAVE_SCIPY = True
except ImportError:
    _HAVE_SCIPY = False

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "pios_horizon_mpl"))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

DEFAULT_HORIZONS = [1, 5, 15, 30]
DEFAULT_SEEDS = [0, 1, 2]
SCENARIO_ORDER = ["baseline", "demand_surge", "transport_disruption", "combined_crisis"]
SCENARIO_NAMES = {
    "baseline": "Normal operations",
    "demand_surge": "Demand surge",
    "transport_disruption": "Transport disruption",
    "combined_crisis": "Combined crisis",
}


def t_ci(values, alpha=0.05):
    n = len(values)
    if n == 0:
        return float("nan"), float("nan"), float("nan")
    mean = sum(values) / n
    if n < 2:
        return mean, mean, mean
    sd = (sum((v - mean) ** 2 for v in values) / (n - 1)) ** 0.5
    se = sd / math.sqrt(n)
    crit = float(_sps.t.ppf(1 - alpha / 2, df=n - 1)) if _HAVE_SCIPY else 1.959963984540054
    return mean, mean - crit * se, mean + crit * se


def scenario_mean(detail: Path, scenario: str, strategy="dreamerv3_official"):
    """Mean shortage_rate for one strategy+scenario in one detail CSV (one seed's run)."""
    vals = []
    if not detail.exists():
        return None
    with detail.open(newline="") as fh:
        for r in csv.DictReader(fh):
            if r["strategy"] == strategy and r["scenario"] == scenario:
                v = r.get("shortage_rate")
                if v not in (None, ""):
                    vals.append(float(v))
    return (sum(vals) / len(vals)) if vals else None


def ref_means(canonical: Path, strategy: str):
    out = defaultdict(list)
    if canonical.exists():
        with canonical.open(newline="") as fh:
            for r in csv.DictReader(fh):
                if r["strategy"] == strategy:
                    v = r.get("shortage_rate")
                    if v not in (None, ""):
                        out[r["scenario"]].append(float(v))
    return {s: sum(v) / len(v) for s, v in out.items() if v}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--horizon", action="append", type=int, default=None)
    ap.add_argument("--train-seed", action="append", type=int, default=None)
    ap.add_argument("--canonical", type=Path,
                    default=Path("results/strategy_eval_thesis_detail.csv"))
    ap.add_argument("--output", type=Path, default=Path("results/horizon_ablation.png"))
    ap.add_argument("--summary", type=Path, default=Path("results/horizon_ablation_summary.csv"))
    ap.add_argument("--dpi", type=int, default=160)
    args = ap.parse_args()
    horizons = sorted(args.horizon or DEFAULT_HORIZONS)
    seeds = sorted(args.train_seed or DEFAULT_SEEDS)

    # per_h_scen_seedmeans[(H, scenario)] = [seed0_mean, seed1_mean, ...]
    per = defaultdict(list)
    for h in horizons:
        for s in seeds:
            detail = Path(f"results/ablation_H{h}_s{s}_detail.csv")
            for sc in SCENARIO_ORDER:
                m = scenario_mean(detail, sc)
                if m is not None:
                    per[(h, sc)].append(m)

    scenarios = [s for s in SCENARIO_ORDER if any((h, s) in per for h in horizons)]
    if not scenarios:
        raise SystemExit("No ablation detail CSVs found — run run_horizon_ablation.py first.")

    ppo = ref_means(args.canonical, "ppo_continuous")
    sac = ref_means(args.canonical, "sac_continuous")

    # Summary CSV
    rows = []
    for sc in scenarios:
        for h in horizons:
            sm = per.get((h, sc), [])
            m, lo, hi = t_ci(sm)
            rows.append({"scenario": sc, "H": h, "n_seeds": len(sm),
                         "mean": f"{m:.4f}", "ci_lo": f"{lo:.4f}", "ci_hi": f"{hi:.4f}",
                         "per_seed": ";".join(f"{x:.3f}" for x in sm),
                         "ppo_ref": f"{ppo.get(sc, float('nan')):.4f}",
                         "sac_ref": f"{sac.get(sc, float('nan')):.4f}"})
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    with args.summary.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)

    # Figure: one panel per scenario.
    ncols = 2
    nrows = (len(scenarios) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(6.2 * ncols, 4.3 * nrows), squeeze=False)
    axf = axes.flatten()
    for ax, sc in zip(axf, scenarios):
        means, los, his = [], [], []
        for h in horizons:
            sm = per.get((h, sc), [])
            m, lo, hi = t_ci(sm)
            means.append(m); los.append(m - lo); his.append(hi - m)
            # individual-seed dots
            ax.scatter([h] * len(sm), sm, color="#94a3b8", s=22, zorder=2, alpha=0.8)
        ax.errorbar(horizons, means, yerr=[los, his], fmt="o-", color="#0ea5e9",
                    capsize=4, linewidth=2, markersize=7, zorder=3,
                    label=f"DreamerV3 (mean, 95% CI over {len(seeds)} seeds)")
        if sc in ppo and ppo[sc] == ppo[sc]:
            ax.axhline(ppo[sc], ls="--", color="#b45309", lw=1.6, label="PPO (fair budget)")
        if sc in sac and sac[sc] == sac[sc]:
            ax.axhline(sac[sc], ls=":", color="#6b7280", lw=1.6, label="SAC (fair budget)")
        ax.set_title(SCENARIO_NAMES.get(sc, sc), fontweight="bold")
        ax.set_xlabel("Imagination horizon $H$")
        ax.set_ylabel("Shortage rate (%)")
        ax.set_xticks(horizons)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8, loc="best")
    for ax in axf[len(scenarios):]:
        ax.set_visible(False)

    fig.suptitle("Imagination-horizon ablation: shortage vs planning depth $H$ "
                 f"({len(seeds)} training seeds)", fontsize=14, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=args.dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {args.output} and {args.summary} "
          f"({len(scenarios)} scenarios x {len(horizons)} horizons x {len(seeds)} seeds)")


if __name__ == "__main__":
    main()
