"""
eval_budget_strategies.py — Evaluate strategies across different budget levels.

Compare DreamerV3, DreamerV4, PPO continuous, PPO shortage-minimizer heuristic, and
fixed/baseline strategies across a grid of budget levels to understand how
each agent degrades (or adapts) as budgets tighten.

Usage
-----
    # Quick smoke test (1 run, short episodes)
    python eval_budget_strategies.py --hours 72 --runs 1

    # Full evaluation with default budget grid
    python eval_budget_strategies.py --runs 3

    # Custom budget grid
    python eval_budget_strategies.py --budgets 3000,4500,6500,8000,10000

    # Only specific strategies
    python eval_budget_strategies.py --strategy baseline --strategy full_response \
        --strategy dreamerv3_official --strategy dreamerv3_budget

    # Only specific scenarios
    python eval_budget_strategies.py --scenario baseline --scenario combined_crisis

    # With trained PPO checkpoints
    python eval_budget_strategies.py \
        --ppo-model ppo_fast_model.zip \
        --ppo-budget-model ppo_budget_model.zip \
        --dreamerv3-checkpoint dreamerv3_runs/all_scenarios/ckpt \
        --dreamerv3-budget-checkpoint dreamerv3_runs/budget_default/ckpt
"""

from __future__ import annotations

import argparse
import copy
import csv
import os
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import Any, Optional

import numpy as np

# ---------------------------------------------------------------------------
# Paths & cache setup
# ---------------------------------------------------------------------------
SIMULATOR_ROOT = Path(__file__).resolve().parent
RESULTS_ROOT = SIMULATOR_ROOT / "results"
CACHE_ROOT = Path(tempfile.gettempdir()) / "pios_budget_eval_cache"
CACHE_ROOT.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("XDG_CACHE_HOME", str(CACHE_ROOT))
os.environ.setdefault("MPLCONFIGDIR", str(CACHE_ROOT / "matplotlib"))

from engine import (  # noqa: E402
    load_dreamerv4_agent,
    load_official_dreamerv3_agent,
    load_ppo_agent,
    run_scenario,
    summarize_state,
)
from eval_metrics import DETAIL_METRICS, extract_eval_metrics  # noqa: E402
from scenarios import SCENARIOS, STRATEGIES  # noqa: E402
from train_mappo import load_training_context, prepare_mappo_scenario  # noqa: E402

# ---------------------------------------------------------------------------
# Default budget grid — covers tight, moderate, generous, and unlimited
# ---------------------------------------------------------------------------
DEFAULT_BUDGET_GRID: list[float] = [
    3000.0,
    4500.0,
    6000.0,
    7500.0,
    9000.0,
    12000.0,
]

# ---------------------------------------------------------------------------
# Virtual strategy keys for the adaptive agents
# ---------------------------------------------------------------------------
ADAPTIVE_STRATEGY_KEYS = {
    "dreamerv3_official",
    "dreamerv4",
    "ppo_continuous",
}

# Display names for virtual strategy keys not in STRATEGIES dict
ADAPTIVE_DISPLAY_NAMES: dict[str, str] = {
    "dreamerv3_budget": "DreamerV3 Budget-Frugal",
    "ppo_trained": "PPO (Trained)",
    "ppo_budget": "PPO (Budget-Aware)",
}

# Metrics to summarise in the console and plots
BUDGET_EVAL_METRICS = [
    "shortage_rate",
    "service_rate",
    "budget_spent",
    "budget_remaining",
    "active_action_cost",
    "total_expired",
    "exact_match_rate",
    "episode_score",
    "reward_total",
    "budget_exhausted_hours",
]

PLOT_Y_METRICS = [
    ("shortage_rate_mean", "Shortage Rate (%)", True),
    ("service_rate_mean", "Service Rate (%)", False),
    ("budget_spent_mean", "Budget Spent", True),
    ("active_action_cost_mean", "Action Cost", True),
    ("total_expired_mean", "Total Expired", True),
    ("episode_score_mean", "Episode Score", False),
    ("budget_exhausted_hours_mean", "Budget Exhaustion (h)", True),
]


# ═══════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════


def parse_args() -> argparse.Namespace:
    all_strategy_keys = sorted(set(STRATEGIES.keys()) | ADAPTIVE_STRATEGY_KEYS)

    parser = argparse.ArgumentParser(
        description=(
            "Evaluate strategies across a grid of budget levels and produce "
            "comparison CSVs, line plots and heatmaps."
        ),
    )
    parser.add_argument(
        "--scenario",
        action="append",
        choices=sorted(SCENARIOS.keys()),
        help="Scenario(s) to evaluate. Repeat to add more. Default: all training scenarios.",
    )
    parser.add_argument(
        "--strategy",
        action="append",
        choices=all_strategy_keys,
        help=(
            "Strategy(s) to evaluate. Repeat to add more. "
            "Special keys: dreamerv3_budget, ppo_trained, ppo_budget."
        ),
    )
    parser.add_argument(
        "--budgets",
        type=str,
        default=None,
        help=(
            "Comma-separated list of budget levels to evaluate. "
            f"Default: {','.join(str(int(b)) for b in DEFAULT_BUDGET_GRID)}"
        ),
    )
    parser.add_argument(
        "--runs",
        type=int,
        default=1,
        help="Seeds per (scenario, strategy, budget) triple.",
    )
    parser.add_argument(
        "--seed-start",
        type=int,
        default=200,
        help="Base RNG seed.",
    )
    parser.add_argument(
        "--hours",
        type=int,
        default=None,
        help="Override sim-hours for faster smoke runs.",
    )
    parser.add_argument(
        "--dreamerv3-checkpoint",
        type=str,
        default=None,
        help="Checkpoint path for the standard DreamerV3 agent.",
    )
    parser.add_argument(
        "--dreamerv4-checkpoint",
        type=str,
        default=None,
        help=(
            "Checkpoint path for the DreamerV4-compatible agent. If omitted, "
            "the simulator tries its default DreamerV4 checkpoint path and "
            "then the standard Dreamer checkpoint fallback."
        ),
    )
    parser.add_argument(
        "--ppo-continuous-model",
        type=str,
        default=None,
        help="Checkpoint path for the budget-frugal DreamerV3 agent.",
    )
    parser.add_argument(
        "--ppo-model",
        type=str,
        default=None,
        help="Path to standard trained PPO checkpoint (.zip or folder).",
    )
    parser.add_argument(
        "--ppo-budget-model",
        type=str,
        default=None,
        help="Path to budget-aware trained PPO checkpoint.",
    )
    parser.add_argument(
        "--ppo-step-hours",
        type=float,
        default=6.0,
        help="Decision interval for PPO controllers.",
    )
    parser.add_argument(
        "--output-prefix",
        type=str,
        default="budget_eval",
        help="Prefix for output artifacts.",
    )
    parser.add_argument(
        "--include-holdouts",
        action="store_true",
        help="Include holdout scenarios in evaluation.",
    )
    return parser.parse_args()


def budget_grid_from_args(args: argparse.Namespace) -> list[float]:
    if args.budgets is not None:
        return sorted(float(b.strip()) for b in args.budgets.split(","))
    return list(DEFAULT_BUDGET_GRID)


def scenario_keys_from_args(args: argparse.Namespace) -> list[str]:
    if args.scenario:
        return list(args.scenario)
    keys = [k for k, v in SCENARIOS.items() if not v.is_holdout]
    if args.include_holdouts:
        keys = list(SCENARIOS.keys())
    return keys


def strategy_keys_from_args(args: argparse.Namespace) -> list[str]:
    if args.strategy:
        return list(args.strategy)
    # Default: a representative set
    return [
        "baseline",
        "full_response",
        "emergency_network",
        "ppo_shortage_minimizer",
        "dreamerv3_official",
        "dreamerv4",
    ]


def resolve_output_prefix(raw_prefix: str) -> Path:
    prefix = Path(raw_prefix).expanduser()
    if prefix.is_absolute() or prefix.parent != Path("."):
        return prefix
    return RESULTS_ROOT / prefix.name


# ═══════════════════════════════════════════════════════════════════════════
# Episode runners
# ═══════════════════════════════════════════════════════════════════════════


def _override_budget(params, budget: float):
    """Patch a ScenarioParams copy to use a specific budget."""
    params.episode_budget = float(budget)
    return params


def run_fixed_strategy_episode(
    strategy_key: str,
    scenario_key: str,
    sim_context,
    *,
    budget: float,
    seed: int,
    hours_override: int | None = None,
) -> Any:
    """Run a non-adaptive (fixed) strategy episode with overridden budget."""
    params = copy.deepcopy(SCENARIOS[scenario_key])
    params.strategy_key = strategy_key
    _override_budget(params, budget)
    if hours_override is not None:
        params.sim_hours = int(hours_override)

    G, north, south, east, west = sim_context
    state = run_scenario(
        params,
        G,
        north,
        south,
        east,
        west,
        seed=seed,
        enable_logs=False,
        fast_mode=True,
    )
    if state.env.now < state.params.sim_hours:
        state.env.run(until=state.params.sim_hours)
    return state


def run_dreamer_episode(
    strategy_key: str,
    scenario_key: str,
    sim_context,
    agent,
    *,
    budget: float,
    seed: int,
    hours_override: int | None = None,
    step_hours: float = 6.0,
) -> Any:
    """Run a Dreamer-compatible agent episode (standard or budget variant)."""
    params = copy.deepcopy(SCENARIOS[scenario_key])
    params.strategy_key = strategy_key
    _override_budget(params, budget)
    if hours_override is not None:
        params.sim_hours = int(hours_override)

    G, north, south, east, west = sim_context
    state = run_scenario(
        params,
        G,
        north,
        south,
        east,
        west,
        seed=seed,
        enable_logs=False,
        fast_mode=True,
        official_dreamerv3_agent=agent,
        official_dreamerv3_step_hours=step_hours,
    )
    if state.env.now < state.params.sim_hours:
        state.env.run(until=state.params.sim_hours)
    return state


def run_ppo_episode(
    scenario_key: str,
    sim_context,
    agent,
    *,
    budget: float,
    seed: int,
    hours_override: int | None = None,
    step_hours: float = 6.0,
) -> Any:
    """Run a trained PPO agent episode (standard or budget variant)."""
    params = copy.deepcopy(SCENARIOS[scenario_key])
    params.strategy_key = "baseline"  # base params; PPO overrides actions
    _override_budget(params, budget)
    if hours_override is not None:
        params.sim_hours = int(hours_override)

    G, north, south, east, west = sim_context
    state = run_scenario(
        params,
        G,
        north,
        south,
        east,
        west,
        seed=seed,
        enable_logs=False,
        fast_mode=True,
        ppo_agent=agent,
        ppo_step_hours=step_hours,
    )
    if state.env.now < state.params.sim_hours:
        state.env.run(until=state.params.sim_hours)
    return state


def run_heuristic_ppo_episode(
    scenario_key: str,
    sim_context,
    *,
    budget: float,
    seed: int,
    hours_override: int | None = None,
    step_hours: float = 6.0,
) -> Any:
    """Run the hand-coded heuristic PPO controller."""
    params = copy.deepcopy(SCENARIOS[scenario_key])
    params.strategy_key = "ppo_shortage_minimizer"
    _override_budget(params, budget)
    if hours_override is not None:
        params.sim_hours = int(hours_override)

    G, north, south, east, west = sim_context
    state = run_scenario(
        params,
        G,
        north,
        south,
        east,
        west,
        seed=seed,
        enable_logs=False,
        fast_mode=True,
        ppo_step_hours=step_hours,
    )
    if state.env.now < state.params.sim_hours:
        state.env.run(until=state.params.sim_hours)
    return state


# ═══════════════════════════════════════════════════════════════════════════
# Metrics extraction
# ═══════════════════════════════════════════════════════════════════════════


def extract_metrics(
    strategy_key: str,
    scenario_key: str,
    budget: float,
    seed: int,
    state,
) -> dict[str, float | int | str]:
    summary = summarize_state(state)
    strategy_obj = STRATEGIES.get(strategy_key, None)
    if strategy_obj is not None:
        display_name = strategy_obj.name
    else:
        display_name = ADAPTIVE_DISPLAY_NAMES.get(strategy_key, strategy_key)

    row: dict[str, float | int | str] = {
        "strategy": strategy_key,
        "strategy_name": display_name,
        "scenario": scenario_key,
        "scenario_name": SCENARIOS[scenario_key].name,
        "budget_level": float(budget),
        "seed": seed,
    }
    row.update(extract_eval_metrics(summary, state))
    return row


# ═══════════════════════════════════════════════════════════════════════════
# Aggregation
# ═══════════════════════════════════════════════════════════════════════════


def aggregate_rows(
    rows: list[dict[str, float | int | str]],
) -> list[dict[str, float | str]]:
    """Group by (strategy, scenario, budget_level) and compute mean/std."""
    grouped: dict[tuple[str, str, float], list[dict]] = defaultdict(list)
    for row in rows:
        key = (str(row["strategy"]), str(row["scenario"]), float(row["budget_level"]))
        grouped[key].append(row)

    summary_rows: list[dict[str, float | str]] = []
    for (strategy_key, scenario_key, budget_level), group in sorted(grouped.items()):
        first = group[0]
        summary_row: dict[str, float | str] = {
            "strategy": strategy_key,
            "strategy_name": str(first["strategy_name"]),
            "scenario": scenario_key,
            "scenario_name": str(first["scenario_name"]),
            "budget_level": float(budget_level),
            "runs": float(len(group)),
        }
        for metric_name in DETAIL_METRICS:
            try:
                values = np.asarray(
                    [float(r[metric_name]) for r in group],
                    dtype=np.float32,
                )
            except (KeyError, TypeError):
                continue
            summary_row[f"{metric_name}_mean"] = float(values.mean())
            summary_row[f"{metric_name}_std"] = float(values.std())
        summary_rows.append(summary_row)
    return summary_rows


def aggregate_across_scenarios(
    summary_rows: list[dict[str, float | str]],
) -> list[dict[str, float | str]]:
    """Aggregate across scenarios to get per-(strategy, budget) means."""
    grouped: dict[tuple[str, float], list[dict]] = defaultdict(list)
    for row in summary_rows:
        key = (str(row["strategy"]), float(row["budget_level"]))
        grouped[key].append(row)

    agg_rows: list[dict[str, float | str]] = []
    for (strategy_key, budget_level), group in sorted(grouped.items()):
        first = group[0]
        agg_row: dict[str, float | str] = {
            "strategy": strategy_key,
            "strategy_name": str(first["strategy_name"]),
            "budget_level": float(budget_level),
            "n_scenarios": float(len(group)),
        }
        for metric_name in DETAIL_METRICS:
            mean_key = f"{metric_name}_mean"
            try:
                values = np.asarray(
                    [float(r[mean_key]) for r in group if mean_key in r],
                    dtype=np.float32,
                )
            except (KeyError, TypeError):
                continue
            if values.size > 0:
                agg_row[f"{metric_name}_mean"] = float(values.mean())
                agg_row[f"{metric_name}_std"] = float(values.std())
        agg_rows.append(agg_row)
    return agg_rows


# ═══════════════════════════════════════════════════════════════════════════
# CSV I/O
# ═══════════════════════════════════════════════════════════════════════════


def save_csv(rows: list[dict], path: Path) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


# ═══════════════════════════════════════════════════════════════════════════
# Plotting
# ═══════════════════════════════════════════════════════════════════════════


def save_budget_line_plots(
    agg_rows: list[dict[str, float | str]],
    budget_grid: list[float],
    strategy_keys: list[str],
    path: Path,
) -> None:
    """
    Create a multi-panel line plot: x-axis = budget level, one line per strategy,
    one subplot per metric.
    """
    if not agg_rows:
        return

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    strategies = sorted(set(str(r["strategy"]) for r in agg_rows))
    metrics = PLOT_Y_METRICS

    n_cols = 2
    n_rows = int(np.ceil(len(metrics) / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, max(5 * n_rows, 10)))
    axes_flat = np.atleast_1d(axes).flatten()
    color_map = plt.get_cmap("tab10")

    # Build lookup: (strategy, budget) → row
    lookup: dict[tuple[str, float], dict] = {}
    for row in agg_rows:
        lookup[(str(row["strategy"]), float(row["budget_level"]))] = row

    for plot_idx, (metric_key, metric_label, lower_is_better) in enumerate(metrics):
        ax = axes_flat[plot_idx]
        for s_idx, strategy in enumerate(strategies):
            x_vals = []
            y_vals = []
            y_errs = []
            for budget in budget_grid:
                row = lookup.get((strategy, budget))
                if row is None or metric_key not in row:
                    continue
                x_vals.append(budget)
                y_vals.append(float(row[metric_key]))
                std_key = metric_key.replace("_mean", "_std")
                y_errs.append(float(row.get(std_key, 0.0)))

            if not x_vals:
                continue

            color = color_map(s_idx % 10)
            display = strategy
            # Attempt friendly name
            for r in agg_rows:
                if str(r["strategy"]) == strategy:
                    display = str(r["strategy_name"])
                    break

            ax.plot(
                x_vals,
                y_vals,
                marker="o",
                linewidth=2,
                markersize=6,
                color=color,
                label=display,
                alpha=0.9,
            )
            if any(e > 0 for e in y_errs):
                y_arr = np.array(y_vals)
                e_arr = np.array(y_errs)
                ax.fill_between(
                    x_vals,
                    y_arr - e_arr,
                    y_arr + e_arr,
                    alpha=0.15,
                    color=color,
                )

        ax.set_title(metric_label, fontsize=11, fontweight="bold")
        ax.set_xlabel("Budget Level")
        ax.set_ylabel(metric_label)
        ax.grid(True, alpha=0.3)
        ax.set_axisbelow(True)
        if lower_is_better:
            # Keep default y-direction
            pass

    # Remove unused axes
    for ax in axes_flat[len(metrics) :]:
        ax.axis("off")

    # Single shared legend at the top
    handles, labels = axes_flat[0].get_legend_handles_labels()
    if handles:
        fig.legend(
            handles,
            labels,
            loc="upper center",
            ncol=min(len(labels), 4),
            fontsize=9,
            framealpha=0.9,
        )

    fig.suptitle(
        "Strategy Performance vs Budget Level (averaged across scenarios)",
        fontsize=14,
        fontweight="bold",
        y=1.02,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def save_scenario_budget_heatmaps(
    summary_rows: list[dict[str, float | str]],
    budget_grid: list[float],
    strategy_keys: list[str],
    scenario_keys: list[str],
    output_dir: Path,
) -> list[Path]:
    """
    For each scenario, save a heatmap: rows=strategies, columns=budget levels,
    cells=shortage_rate_mean.
    """
    if not summary_rows:
        return []

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output_dir.mkdir(parents=True, exist_ok=True)
    saved_paths: list[Path] = []

    # Build lookup
    lookup: dict[tuple[str, str, float], dict] = {}
    for row in summary_rows:
        lookup[
            (str(row["strategy"]), str(row["scenario"]), float(row["budget_level"]))
        ] = row

    strategies_present = sorted(set(str(r["strategy"]) for r in summary_rows))

    for scenario_key in scenario_keys:
        metric_key = "shortage_rate_mean"
        matrix = np.full(
            (len(strategies_present), len(budget_grid)),
            np.nan,
            dtype=np.float32,
        )
        for s_idx, strategy in enumerate(strategies_present):
            for b_idx, budget in enumerate(budget_grid):
                row = lookup.get((strategy, scenario_key, budget))
                if row and metric_key in row:
                    matrix[s_idx, b_idx] = float(row[metric_key])

        if np.all(np.isnan(matrix)):
            continue

        fig, ax = plt.subplots(
            figsize=(
                max(8, len(budget_grid) * 1.5),
                max(4, len(strategies_present) * 0.6),
            )
        )
        im = ax.imshow(matrix, aspect="auto", cmap="RdYlGn_r")
        ax.set_xticks(range(len(budget_grid)))
        ax.set_xticklabels([f"{int(b)}" for b in budget_grid], fontsize=9)
        ax.set_yticks(range(len(strategies_present)))

        # Get display names
        display_names = []
        for s in strategies_present:
            name = s
            for r in summary_rows:
                if str(r["strategy"]) == s:
                    name = str(r["strategy_name"])
                    break
            display_names.append(name)
        ax.set_yticklabels(display_names, fontsize=9)

        ax.set_xlabel("Budget Level")
        ax.set_title(
            f"Shortage Rate (%) — {SCENARIOS[scenario_key].name}",
            fontsize=12,
            fontweight="bold",
        )

        # Annotate cells
        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                val = matrix[i, j]
                if not np.isnan(val):
                    text_color = "white" if val > np.nanmedian(matrix) else "black"
                    ax.text(
                        j,
                        i,
                        f"{val:.1f}",
                        ha="center",
                        va="center",
                        fontsize=8,
                        color=text_color,
                        fontweight="bold",
                    )

        fig.colorbar(im, ax=ax, label="Shortage Rate (%)", shrink=0.8)
        fig.tight_layout()
        out_path = output_dir / f"heatmap_{scenario_key}.png"
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        saved_paths.append(out_path)

    return saved_paths


def save_budget_efficiency_plot(
    agg_rows: list[dict[str, float | str]],
    budget_grid: list[float],
    path: Path,
) -> None:
    """
    Scatter: x = budget_spent_mean, y = shortage_rate_mean.
    Each point is a (strategy, budget_level) pair. Points from the same strategy
    are connected by a line. This shows the cost-efficiency frontier.
    """
    if not agg_rows:
        return

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    strategies = sorted(set(str(r["strategy"]) for r in agg_rows))
    color_map = plt.get_cmap("tab10")

    fig, ax = plt.subplots(figsize=(12, 8))

    for s_idx, strategy in enumerate(strategies):
        x_vals, y_vals, labels = [], [], []
        for row in sorted(agg_rows, key=lambda r: float(r.get("budget_level", 0))):
            if str(row["strategy"]) != strategy:
                continue
            if "budget_spent_mean" not in row or "shortage_rate_mean" not in row:
                continue
            x_vals.append(float(row["budget_spent_mean"]))
            y_vals.append(float(row["shortage_rate_mean"]))
            labels.append(f"B={int(row['budget_level'])}")

        if not x_vals:
            continue

        color = color_map(s_idx % 10)
        display = strategy
        for r in agg_rows:
            if str(r["strategy"]) == strategy:
                display = str(r["strategy_name"])
                break

        ax.plot(
            x_vals,
            y_vals,
            marker="o",
            linewidth=2,
            markersize=8,
            color=color,
            label=display,
            alpha=0.85,
        )
        # Annotate budget level on each point
        for x, y, lbl in zip(x_vals, y_vals, labels):
            ax.annotate(
                lbl,
                (x, y),
                textcoords="offset points",
                xytext=(5, 5),
                fontsize=7,
                color=color,
                alpha=0.8,
            )

    ax.set_xlabel("Budget Spent (mean)", fontsize=12)
    ax.set_ylabel("Shortage Rate % (mean, lower is better)", fontsize=12)
    ax.set_title(
        "Budget Efficiency Frontier: Spending vs Service Quality",
        fontsize=14,
        fontweight="bold",
    )
    ax.grid(True, alpha=0.3)
    ax.set_axisbelow(True)
    ax.legend(fontsize=9, loc="best", framealpha=0.9)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def save_radar_comparison(
    agg_rows: list[dict[str, float | str]],
    budget_level: float,
    path: Path,
) -> None:
    """
    Radar chart comparing all strategies at a single budget level across
    key normalised metrics.
    """
    if not agg_rows:
        return

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # Filter rows for this budget level
    budget_rows = [
        r
        for r in agg_rows
        if abs(float(r.get("budget_level", -1)) - budget_level) < 1.0
    ]
    if not budget_rows:
        return

    radar_metrics = [
        ("service_rate_mean", "Service Rate", False),
        ("exact_match_rate_mean", "Exact Match", False),
        ("episode_score_mean", "Episode Score", False),
        ("shortage_rate_mean", "Low Shortage", True),  # invert: lower is better
        ("budget_spent_mean", "Budget Savings", True),  # invert: lower spent is better
        ("active_action_cost_mean", "Cost Efficiency", True),  # invert
    ]

    # Compute min/max for normalisation
    metric_ranges: dict[str, tuple[float, float]] = {}
    for mk, _, _ in radar_metrics:
        vals = [float(r[mk]) for r in budget_rows if mk in r]
        if vals:
            metric_ranges[mk] = (min(vals), max(vals))
        else:
            metric_ranges[mk] = (0.0, 1.0)

    strategies = []
    normalised_values = []
    for row in sorted(budget_rows, key=lambda r: str(r["strategy"])):
        name = str(row.get("strategy_name", row["strategy"]))
        vals = []
        skip = False
        for mk, _, invert in radar_metrics:
            if mk not in row:
                skip = True
                break
            lo, hi = metric_ranges[mk]
            rng = hi - lo if hi > lo else 1.0
            norm = (float(row[mk]) - lo) / rng
            if invert:
                norm = 1.0 - norm
            vals.append(max(0.0, min(1.0, norm)))
        if skip:
            continue
        strategies.append(name)
        normalised_values.append(vals)

    if len(strategies) < 2:
        return

    n_metrics = len(radar_metrics)
    angles = np.linspace(0, 2 * np.pi, n_metrics, endpoint=False).tolist()
    angles += angles[:1]  # close the polygon

    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))
    color_map = plt.get_cmap("tab10")

    for idx, (name, vals) in enumerate(zip(strategies, normalised_values)):
        vals_closed = vals + vals[:1]
        color = color_map(idx % 10)
        ax.plot(angles, vals_closed, linewidth=2, label=name, color=color)
        ax.fill(angles, vals_closed, alpha=0.08, color=color)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels([label for _, label, _ in radar_metrics], fontsize=10)
    ax.set_ylim(0, 1)
    ax.set_title(
        f"Strategy Comparison at Budget = {int(budget_level)}",
        fontsize=14,
        fontweight="bold",
        pad=30,
    )
    ax.legend(loc="upper right", bbox_to_anchor=(1.35, 1.1), fontsize=9)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ═══════════════════════════════════════════════════════════════════════════
# Console output
# ═══════════════════════════════════════════════════════════════════════════


def print_budget_summary(
    agg_rows: list[dict[str, float | str]],
    budget_grid: list[float],
) -> None:
    """Print a compact table to stdout."""
    strategies = sorted(set(str(r["strategy"]) for r in agg_rows))

    for budget in budget_grid:
        print(f"\n{'=' * 78}")
        print(f"  Budget Level: {int(budget)}")
        print(f"{'=' * 78}")
        print(
            f"{'Strategy':>24s} | {'Shortage%':>10s} | {'Service%':>10s} | "
            f"{'BudgetSpent':>11s} | {'Cost':>8s} | {'Expired':>8s} | {'Score':>8s}"
        )
        print("-" * 100)
        for strategy in strategies:
            row = next(
                (
                    r
                    for r in agg_rows
                    if str(r["strategy"]) == strategy
                    and abs(float(r.get("budget_level", -1)) - budget) < 1.0
                ),
                None,
            )
            if row is None:
                continue
            display = str(row.get("strategy_name", strategy))
            shortage = row.get("shortage_rate_mean", float("nan"))
            service = row.get("service_rate_mean", float("nan"))
            spent = row.get("budget_spent_mean", float("nan"))
            cost = row.get("active_action_cost_mean", float("nan"))
            expired = row.get("total_expired_mean", float("nan"))
            score = row.get("episode_score_mean", float("nan"))
            print(
                f"{display:>24s} | {shortage:>10.2f} | {service:>10.2f} | "
                f"{spent:>11.1f} | {cost:>8.1f} | {expired:>8.1f} | {score:>8.2f}"
            )


# ═══════════════════════════════════════════════════════════════════════════
# Agent loading helpers
# ═══════════════════════════════════════════════════════════════════════════


def _try_load_dreamer(
    checkpoint: Optional[str],
    seed: int,
    label: str,
    loader,
):
    ckpt: Path | None = None
    if checkpoint is not None:
        ckpt = Path(checkpoint)
        if not ckpt.exists():
            print(f"[eval_budget] {label} checkpoint not found: {ckpt} — skipping.")
            return None
    try:
        agent = loader(str(ckpt) if ckpt is not None else None, seed=seed)
        if ckpt is not None:
            print(f"[eval_budget] Loaded {label} from {ckpt}")
        else:
            print(f"[eval_budget] Loaded {label} from the default checkpoint resolution.")
        return agent
    except Exception as exc:
        print(f"[eval_budget] Failed to load {label}: {exc} — skipping.")
        return None


def _try_load_ppo(model_path: Optional[str], sim_context, label: str):
    if model_path is None:
        return None
    p = Path(model_path)
    if not p.exists():
        print(f"[eval_budget] {label} not found: {p} — skipping.")
        return None
    try:
        G, north, south, east, west = sim_context
        agent = load_ppo_agent(
            str(p),
            params=prepare_mappo_scenario(copy.deepcopy(SCENARIOS["baseline"])),
            G=G,
            north=north,
            south=south,
            east=east,
            west=west,
            verbose=0,
        )
        print(f"[eval_budget] Loaded {label} from {p}")
        return agent
    except Exception as exc:
        print(f"[eval_budget] Failed to load {label}: {exc} — skipping.")
        return None


# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════


def main() -> None:
    args = parse_args()
    budget_grid = budget_grid_from_args(args)
    scenario_keys = scenario_keys_from_args(args)
    strategy_keys = strategy_keys_from_args(args)

    print(f"\n[eval_budget] Budget grid     : {[int(b) for b in budget_grid]}")
    print(f"[eval_budget] Scenarios       : {scenario_keys}")
    print(f"[eval_budget] Strategies      : {strategy_keys}")
    print(f"[eval_budget] Runs per triple : {args.runs}")
    if args.hours:
        print(f"[eval_budget] Hours override  : {args.hours}")

    sim_context = load_training_context()

    # ── Load adaptive agents ──────────────────────────────────────────
    dreamer_agents: dict[str, Any] = {}
    ppo_continuous_agent = None

    if "dreamerv3_official" in strategy_keys:
        dreamer_agents["dreamerv3_official"] = _try_load_dreamer(
            args.dreamerv3_checkpoint,
            args.seed_start,
            "DreamerV3 (standard)",
            load_official_dreamerv3_agent,
        )
        if dreamer_agents["dreamerv3_official"] is None:
            print("[eval_budget] Removing dreamerv3_official (no checkpoint).")
            strategy_keys = [s for s in strategy_keys if s != "dreamerv3_official"]

    if "dreamerv4" in strategy_keys:
        dreamer_agents["dreamerv4"] = _try_load_dreamer(
            args.dreamerv4_checkpoint,
            args.seed_start,
            "DreamerV4",
            load_dreamerv4_agent,
        )
        if dreamer_agents["dreamerv4"] is None:
            print("[eval_budget] Removing dreamerv4 (no checkpoint).")
            strategy_keys = [s for s in strategy_keys if s != "dreamerv4"]

    if "ppo_continuous" in strategy_keys:
        ppo_continuous_agent = _try_load_ppo_continuous(
            args.ppo_continuous_model,
            sim_context,
            "PPO Continuous",
        )
        if dreamer_budget_agent is None:
            print("[eval_budget] Removing dreamerv3_budget (no checkpoint).")
            strategy_keys = [s for s in strategy_keys if s != "dreamerv3_budget"]

    if "ppo_trained" in strategy_keys:
        ppo_agent = _try_load_ppo(args.ppo_model, sim_context, "PPO (standard)")
        if ppo_agent is None:
            print("[eval_budget] Removing ppo_trained (no model).")
            strategy_keys = [s for s in strategy_keys if s != "ppo_trained"]

    if "ppo_budget" in strategy_keys:
        ppo_budget_agent = _try_load_ppo(
            args.ppo_budget_model,
            sim_context,
            "PPO (budget-aware)",
        )
        if ppo_budget_agent is None:
            print("[eval_budget] Removing ppo_budget (no model).")
            strategy_keys = [s for s in strategy_keys if s != "ppo_budget"]

    if not strategy_keys:
        print("[eval_budget] No strategies remaining. Exiting.")
        return

    print(f"\n[eval_budget] Final strategies : {strategy_keys}")
    n_total = len(scenario_keys) * len(strategy_keys) * len(budget_grid) * args.runs
    print(f"[eval_budget] Total episodes  : {n_total}")
    print()

    # ── Run evaluation grid ───────────────────────────────────────────
    rows: list[dict[str, float | int | str]] = []
    episode_count = 0

    for scenario_key in scenario_keys:
        for budget in budget_grid:
            for strategy_key in strategy_keys:
                for run_idx in range(args.runs):
                    seed = args.seed_start + run_idx
                    episode_count += 1
                    tag = (
                        f"[{episode_count}/{n_total}] "
                        f"{scenario_key} | {strategy_key} | "
                        f"budget={int(budget)} | seed={seed}"
                    )
                    print(tag)

                    try:
                        if strategy_key in {"dreamerv3_official", "dreamerv4"}:
                            state = run_dreamer_episode(
                                strategy_key,
                                scenario_key,
                                sim_context,
                                dreamer_agents[strategy_key],
                                budget=budget,
                                seed=seed,
                                hours_override=args.hours,
                            )
                        elif strategy_key == "dreamerv3_budget":
                            state = run_dreamerv3_episode(
                                scenario_key,
                                sim_context,
                                dreamer_budget_agent,
                                budget=budget,
                                seed=seed,
                                hours_override=args.hours,
                            )
                        elif strategy_key == "ppo_trained":
                            state = run_ppo_episode(
                                scenario_key,
                                sim_context,
                                ppo_agent,
                                budget=budget,
                                seed=seed,
                                hours_override=args.hours,
                                step_hours=args.ppo_step_hours,
                            )
                        elif strategy_key == "ppo_budget":
                            state = run_ppo_episode(
                                scenario_key,
                                sim_context,
                                ppo_budget_agent,
                                budget=budget,
                                seed=seed,
                                hours_override=args.hours,
                                step_hours=args.ppo_step_hours,
                            )
                        elif strategy_key == "ppo_shortage_minimizer":
                            state = run_heuristic_ppo_episode(
                                scenario_key,
                                sim_context,
                                budget=budget,
                                seed=seed,
                                hours_override=args.hours,
                                step_hours=args.ppo_step_hours,
                            )
                        else:
                            # Fixed strategy from STRATEGIES dict
                            state = run_fixed_strategy_episode(
                                strategy_key,
                                scenario_key,
                                sim_context,
                                budget=budget,
                                seed=seed,
                                hours_override=args.hours,
                            )

                        rows.append(
                            extract_metrics(
                                strategy_key,
                                scenario_key,
                                budget,
                                seed,
                                state,
                            )
                        )
                    except Exception as exc:
                        print(f"  !! ERROR: {exc}")
                        import traceback

                        traceback.print_exc()
                        continue

    if not rows:
        print("\n[eval_budget] No episodes completed successfully.")
        return

    # ── Aggregate ─────────────────────────────────────────────────────
    summary_rows = aggregate_rows(rows)
    agg_rows = aggregate_across_scenarios(summary_rows)

    # ── Output paths ──────────────────────────────────────────────────
    prefix = resolve_output_prefix(args.output_prefix)
    prefix.parent.mkdir(parents=True, exist_ok=True)

    detail_csv = prefix.with_name(f"{prefix.name}_detail.csv")
    summary_csv = prefix.with_name(f"{prefix.name}_summary.csv")
    agg_csv = prefix.with_name(f"{prefix.name}_aggregated.csv")
    line_plot_path = prefix.with_name(f"{prefix.name}_budget_lines.png")
    efficiency_plot_path = prefix.with_name(f"{prefix.name}_efficiency_frontier.png")
    heatmap_dir = prefix.with_name(f"{prefix.name}_heatmaps")
    radar_dir = prefix.with_name(f"{prefix.name}_radar")

    # ── Save CSVs ─────────────────────────────────────────────────────
    save_csv(rows, detail_csv)
    save_csv(summary_rows, summary_csv)
    save_csv(agg_rows, agg_csv)

    # ── Save plots ────────────────────────────────────────────────────
    save_budget_line_plots(agg_rows, budget_grid, strategy_keys, line_plot_path)
    save_budget_efficiency_plot(agg_rows, budget_grid, efficiency_plot_path)
    heatmap_paths = save_scenario_budget_heatmaps(
        summary_rows,
        budget_grid,
        strategy_keys,
        scenario_keys,
        heatmap_dir,
    )

    # Radar charts at a few representative budget levels
    radar_dir_path = Path(radar_dir)
    radar_dir_path.mkdir(parents=True, exist_ok=True)
    radar_budgets = [
        budget_grid[0],
        budget_grid[len(budget_grid) // 2],
        budget_grid[-1],
    ]
    for rb in radar_budgets:
        save_radar_comparison(
            agg_rows,
            rb,
            radar_dir_path / f"radar_budget_{int(rb)}.png",
        )

    # ── Console summary ───────────────────────────────────────────────
    print_budget_summary(agg_rows, budget_grid)

    # ── Print file locations ──────────────────────────────────────────
    print(f"\n{'=' * 68}")
    print("Artifacts saved:")
    print(f"  Detail CSV      : {detail_csv}")
    print(f"  Summary CSV     : {summary_csv}")
    print(f"  Aggregated CSV  : {agg_csv}")
    print(f"  Line plots      : {line_plot_path}")
    print(f"  Efficiency plot : {efficiency_plot_path}")
    if heatmap_paths:
        print(f"  Heatmaps ({len(heatmap_paths)})   : {heatmap_dir}/")
    print(f"  Radar charts    : {radar_dir}/")
    print(f"{'=' * 68}")


if __name__ == "__main__":
    main()
