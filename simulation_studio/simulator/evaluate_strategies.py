from __future__ import annotations

import argparse
import copy
import csv
import json
import os
import tempfile
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
import textwrap

import numpy as np

SIMULATOR_ROOT = Path(__file__).resolve().parent
RESULTS_ROOT = SIMULATOR_ROOT / "results"
CACHE_ROOT = Path(tempfile.gettempdir()) / "pios_strategy_eval_cache"
CACHE_ROOT.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("XDG_CACHE_HOME", str(CACHE_ROOT))
os.environ.setdefault("MPLCONFIGDIR", str(CACHE_ROOT / "matplotlib"))

from engine import (
    DREAMER_CONTROLLER_KEYS,
    LEARNED_CONTINUOUS_CONTROLLER_KEYS,
    load_dreamerv4_agent,
    load_learned_continuous_agent,
    load_official_dreamerv3_agent,
    resolve_dreamer_checkpoint_path,
    resolve_learned_continuous_model_path,
    run_scenario,
    summarize_state,
)
from eval_metrics import (
    DETAIL_METRICS,
    METRIC_LABELS,
    PLOT_METRICS,
    SCENARIO_PLOT_METRICS,
    extract_eval_metrics,
)
from scenarios import SCENARIOS, STRATEGIES
from train_mappo import load_training_context

SUMMARY_MEAN_METRICS = [f"{metric_name}_mean" for metric_name in DETAIL_METRICS]
DEFAULT_DREAMERV3_EVAL_RUN = (
    SIMULATOR_ROOT / "dreamerv3_runs" / "m1_continuous_run8_kpi_aligned"
)
DEFAULT_EVAL_STRATEGY_KEYS = [
    "ppo_continuous",
    "sac_continuous",
    "iql_offline",
    "cql_offline",
    "dreamerv3_official",
]
THESIS_OVERVIEW_METRICS = [
    ("shortage_rate_mean", "Shortage Rate (%)", "min"),
    ("service_rate_mean", "Demand Fulfilled (%)", "max"),
    ("reward_total_mean", "Total Reward", "max"),
    ("exact_match_rate_mean", "Exact Match Rate (%)", "max"),
]


@dataclass(frozen=True)
class PlotExportConfig:
    thesis_ready: bool = False
    formats: tuple[str, ...] = ("png",)
    dpi: int = 180


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate every built-in simulator strategy across the selected "
            "scenarios, then save CSV summaries and comparison plots."
        )
    )
    parser.add_argument(
        "--scenario",
        action="append",
        choices=sorted(SCENARIOS.keys()),
        help="Scenario key to evaluate. Repeat to limit the scenario set.",
    )
    parser.add_argument(
        "--strategy",
        action="append",
        choices=sorted(STRATEGIES.keys()),
        help="Strategy key to evaluate. Repeat to limit the strategy set.",
    )
    parser.add_argument(
        "--runs",
        type=int,
        default=1,
        help="How many seeds to evaluate per scenario/strategy pair.",
    )
    parser.add_argument(
        "--seed-start",
        type=int,
        default=100,
        help="Base seed for evaluation runs.",
    )
    parser.add_argument(
        "--hours",
        type=int,
        default=None,
        help="Optional simulation-hour override for faster smoke runs.",
    )
    parser.add_argument(
        "--dreamerv3-checkpoint",
        type=str,
        default=str(DEFAULT_DREAMERV3_EVAL_RUN),
        help=(
            "Official DreamerV3 checkpoint path for the `dreamerv3_official` "
            "strategy. Defaults to "
            "simulator/dreamerv3_runs/m1_continuous_run8_kpi_aligned."
        ),
    )
    parser.add_argument(
        "--dreamerv4-checkpoint",
        type=str,
        default=None,
        help=(
            "DreamerV4-compatible checkpoint path for the `dreamerv4` "
            "strategy. Defaults to simulator/dreamerv4_runs/all_scenarios/ckpt "
            "and falls back to the standard Dreamer checkpoint when needed."
        ),
    )
    parser.add_argument(
        "--ppo-continuous-model",
        type=str,
        default=None,
        help=(
            "Path to a trained PPO continuous-action model (.zip or .pkl). "
            "If not given, the default location is checked automatically."
        ),
    )
    parser.add_argument(
        "--sac-model",
        type=str,
        default=None,
        help="Path to a trained SAC continuous-action model (.zip).",
    )
    parser.add_argument(
        "--iql-model",
        type=str,
        default=None,
        help="Path to a trained IQL offline checkpoint (.pt).",
    )
    parser.add_argument(
        "--cql-model",
        type=str,
        default=None,
        help="Path to a trained CQL offline checkpoint (.pt).",
    )
    parser.add_argument(
        "--ppo-step-hours",
        type=float,
        default=6.0,
        help="Decision interval (sim-hours) for PPO continuous controller.",
    )
    parser.add_argument(
        "--output-prefix",
        type=str,
        default="strategy_eval",
        help="Prefix for generated CSV and plot artifacts.",
    )
    parser.add_argument(
        "--thesis-ready",
        action="store_true",
        help=(
            "Enable publication-style figure exports, including higher-resolution "
            "PNG/PDF/SVG outputs and overview scorecard plots."
        ),
    )
    parser.add_argument(
        "--export-format",
        action="append",
        choices=["png", "pdf", "svg"],
        help=(
            "Plot format to export. Repeat to request multiple formats. "
            "Defaults to PNG, or PNG/PDF/SVG when --thesis-ready is enabled."
        ),
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=None,
        help="Optional output DPI override for raster exports.",
    )
    parser.add_argument(
        "--pareto-x-metric",
        choices=sorted(SUMMARY_MEAN_METRICS),
        default="active_action_cost_mean",
        help="Summary metric to place on the Pareto plot x-axis.",
    )
    parser.add_argument(
        "--pareto-y-metric",
        choices=sorted(SUMMARY_MEAN_METRICS),
        default="shortage_rate_mean",
        help="Summary metric to place on the Pareto plot y-axis.",
    )
    parser.add_argument(
        "--pareto-x-goal",
        choices=["min", "max"],
        default="min",
        help="Whether the Pareto x-axis metric should be minimized or maximized.",
    )
    parser.add_argument(
        "--pareto-y-goal",
        choices=["min", "max"],
        default="min",
        help="Whether the Pareto y-axis metric should be minimized or maximized.",
    )
    parser.add_argument(
        "--pareto-z-metric",
        choices=sorted(SUMMARY_MEAN_METRICS),
        default="total_expired_mean",
        help="Summary metric to place on the Pareto plot z-axis.",
    )
    parser.add_argument(
        "--pareto-z-goal",
        choices=["min", "max"],
        default="min",
        help="Whether the Pareto z-axis metric should be minimized or maximized.",
    )
    # --- Sensitivity-sweep overrides (1.0 / unset = canonical, no change) ---
    parser.add_argument("--delay-scale", type=float, default=1.0,
                        help="Scale every action's activation delay (sensitivity sweep).")
    parser.add_argument("--ramp-scale", type=float, default=1.0,
                        help="Scale every action's ramp-up time (sensitivity sweep).")
    parser.add_argument("--reward-safety-scale", type=float, default=1.0,
                        help="Scale the Tier-1 (patient safety) reward weight.")
    parser.add_argument("--reward-matching-scale", type=float, default=1.0,
                        help="Scale the Tier-2 (matching quality) reward weight.")
    parser.add_argument("--reward-waste-scale", type=float, default=1.0,
                        help="Scale the Tier-3 (waste/cost) reward weight.")
    return parser.parse_args()


def scenario_keys_from_args(args: argparse.Namespace) -> list[str]:
    return list(args.scenario) if args.scenario else list(SCENARIOS.keys())


def strategy_keys_from_args(args: argparse.Namespace) -> list[str]:
    if args.strategy:
        return list(args.strategy)
    return [key for key in DEFAULT_EVAL_STRATEGY_KEYS if key in STRATEGIES]


def resolve_output_prefix(raw_prefix: str | Path) -> Path:
    prefix = Path(raw_prefix).expanduser()
    if prefix.is_absolute() or prefix.parent != Path("."):
        return prefix
    return RESULTS_ROOT / prefix.name


def build_plot_export_config(args: argparse.Namespace) -> PlotExportConfig:
    requested_formats = tuple(args.export_format or ())
    if requested_formats:
        formats = tuple(dict.fromkeys(requested_formats))
    elif args.thesis_ready:
        formats = ("png", "pdf", "svg")
    else:
        formats = ("png",)
    dpi = int(args.dpi or (300 if args.thesis_ready else 180))
    return PlotExportConfig(
        thesis_ready=bool(args.thesis_ready),
        formats=formats,
        dpi=dpi,
    )


# Sensitivity-sweep overrides applied to every episode's scenario params.
# Empty by default (canonical run is unaffected); populated in main() from the
# --delay-scale / --ramp-scale / --reward-*-scale flags.
_SWEEP_OVERRIDES: dict = {}


def run_episode(
    strategy_key: str,
    scenario_key: str,
    sim_context,
    *,
    seed: int,
    hours_override: int | None = None,
    dreamer_agents: dict[str, object] | None = None,
    dreamer_checkpoints: dict[str, str | None] | None = None,
    continuous_agents: dict[str, object] | None = None,
    ppo_step_hours: float = 6.0,
):
    params = copy.deepcopy(SCENARIOS[scenario_key])
    params.strategy_key = strategy_key
    if hours_override is not None:
        params.sim_hours = int(hours_override)
    if _SWEEP_OVERRIDES:
        params.delay_scale = _SWEEP_OVERRIDES.get("delay_scale", params.delay_scale)
        params.ramp_scale = _SWEEP_OVERRIDES.get("ramp_scale", params.ramp_scale)
        tier_weights = _SWEEP_OVERRIDES.get("reward_tier_weights")
        if tier_weights:
            params.reward_tier_weights = dict(tier_weights)

    G, north, south, east, west = sim_context
    dreamer_agent = (dreamer_agents or {}).get(strategy_key)
    dreamer_checkpoint = (dreamer_checkpoints or {}).get(strategy_key)
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
        official_dreamerv3_agent=(
            dreamer_agent if strategy_key in DREAMER_CONTROLLER_KEYS else None
        ),
        official_dreamerv3_checkpoint=(
            dreamer_checkpoint
            if strategy_key in DREAMER_CONTROLLER_KEYS and dreamer_agent is None
            else None
        ),
        ppo_agent=(continuous_agents or {}).get(strategy_key),
        ppo_step_hours=ppo_step_hours,
    )
    if state.env.now < state.params.sim_hours:
        state.env.run(until=state.params.sim_hours)
    return state


def extract_metrics(
    strategy_key: str,
    scenario_key: str,
    seed: int,
    state,
) -> dict[str, float | int | str]:
    summary = summarize_state(state)
    row: dict[str, float | int | str] = {
        "strategy": strategy_key,
        "strategy_name": STRATEGIES[strategy_key].name,
        "scenario": scenario_key,
        "scenario_name": SCENARIOS[scenario_key].name,
        "seed": seed,
    }
    row.update(extract_eval_metrics(summary, state))
    return row


def aggregate_rows(
    rows: list[dict[str, float | int | str]],
) -> list[dict[str, float | str]]:
    grouped: dict[tuple[str, str], list[dict[str, float | int | str]]] = defaultdict(
        list
    )
    for row in rows:
        grouped[(str(row["strategy"]), str(row["scenario"]))].append(row)

    summary_rows: list[dict[str, float | str]] = []
    for (strategy_key, scenario_key), grouped_rows in sorted(grouped.items()):
        summary_row: dict[str, float | str] = {
            "strategy": strategy_key,
            "strategy_name": STRATEGIES[strategy_key].name,
            "scenario": scenario_key,
            "scenario_name": SCENARIOS[scenario_key].name,
            "runs": float(len(grouped_rows)),
        }
        for metric_name in DETAIL_METRICS:
            try:
                values = np.asarray(
                    [float(row[metric_name]) for row in grouped_rows],
                    dtype=np.float32,
                )
            except (KeyError, TypeError):
                continue
            summary_row[f"{metric_name}_mean"] = float(values.mean())
            summary_row[f"{metric_name}_std"] = float(values.std())
        summary_rows.append(summary_row)
    return summary_rows


def pivot_metric(
    summary_rows: list[dict[str, float | str]],
    scenario_keys: list[str],
    strategy_keys: list[str],
    metric_name: str,
) -> np.ndarray:
    lookup = {
        (str(row["strategy"]), str(row["scenario"])): float(row[metric_name])
        for row in summary_rows
        if metric_name in row
    }
    matrix = np.full((len(strategy_keys), len(scenario_keys)), np.nan, dtype=np.float32)
    for row_idx, strategy_key in enumerate(strategy_keys):
        for col_idx, scenario_key in enumerate(scenario_keys):
            value = lookup.get((strategy_key, scenario_key))
            if value is not None:
                matrix[row_idx, col_idx] = value
    return matrix


def save_csv(rows: list[dict], path: Path) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def metric_label(metric_name: str) -> str:
    return METRIC_LABELS.get(metric_name, metric_name.replace("_", " ").title())


def wrap_plot_label(value: str, width: int = 20) -> str:
    wrapped = textwrap.wrap(str(value), width=width)
    return "\n".join(wrapped) if wrapped else str(value)


def plot_rcparams(config: PlotExportConfig) -> dict[str, object]:
    if not config.thesis_ready:
        return {}
    return {
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "axes.edgecolor": "#0f172a",
        "axes.labelcolor": "#0f172a",
        "axes.titleweight": "bold",
        "axes.titlesize": 13,
        "axes.labelsize": 11,
        "font.family": "DejaVu Serif",
        "font.size": 10.5,
        "grid.color": "#cbd5e1",
        "grid.alpha": 0.32,
        "legend.frameon": False,
        "savefig.facecolor": "white",
        "savefig.edgecolor": "white",
    }


def export_figure(
    fig,
    base_path: Path,
    config: PlotExportConfig,
    *,
    bbox_inches: str | None = "tight",
) -> list[Path]:
    base_path.parent.mkdir(parents=True, exist_ok=True)
    saved_paths: list[Path] = []
    for fmt in config.formats:
        output_path = base_path.with_suffix(f".{fmt}")
        kwargs: dict[str, object] = {}
        if bbox_inches is not None:
            kwargs["bbox_inches"] = bbox_inches
        if fmt == "png":
            kwargs["dpi"] = config.dpi
        fig.savefig(output_path, **kwargs)
        saved_paths.append(output_path)
    return saved_paths


def pareto_frontier_mask(
    points: np.ndarray,
    goals: Sequence[str],
) -> np.ndarray:
    if points.ndim != 2:
        raise ValueError("Pareto frontier expects a 2D points array.")
    if points.shape[1] != len(goals):
        raise ValueError("Point columns must match the number of optimization goals.")
    if points.shape[0] == 0:
        return np.zeros(0, dtype=bool)

    normalized = points.astype(np.float64, copy=True)
    for col_idx, goal in enumerate(goals):
        if goal == "max":
            normalized[:, col_idx] *= -1.0
        elif goal != "min":
            raise ValueError(f"Unsupported Pareto goal: {goal}")

    frontier = np.zeros(points.shape[0], dtype=bool)
    finite_indices = np.flatnonzero(np.all(np.isfinite(normalized), axis=1))
    for row_idx in finite_indices:
        candidate = normalized[row_idx]
        dominated = False
        for other_idx in finite_indices:
            if other_idx == row_idx:
                continue
            competitor = normalized[other_idx]
            if np.all(competitor <= candidate) and np.any(competitor < candidate):
                dominated = True
                break
        frontier[row_idx] = not dominated
    return frontier


def collect_pareto_rows(
    summary_rows: list[dict[str, float | str]],
    scenario_keys: list[str],
    strategy_keys: list[str],
    *,
    scenario_key: str | None,
    x_metric: str,
    y_metric: str,
    z_metric: str | None = None,
) -> list[dict[str, float | str]]:
    collected_rows: list[dict[str, float | str]] = []
    allowed_scenarios = set(scenario_keys)
    for strategy_key in strategy_keys:
        matching_rows = [
            row
            for row in summary_rows
            if str(row["strategy"]) == strategy_key
            and str(row["scenario"]) in allowed_scenarios
            and (scenario_key is None or str(row["scenario"]) == scenario_key)
        ]
        if not matching_rows:
            continue
        collected_row: dict[str, float | str] = {
            "strategy": strategy_key,
            "strategy_name": STRATEGIES[strategy_key].name,
            "scenario": scenario_key or "all",
            "scenario_name": (
                SCENARIOS[scenario_key].name
                if scenario_key is not None
                else "All Scenarios"
            ),
            x_metric: float(
                np.mean(
                    [float(row[x_metric]) for row in matching_rows], dtype=np.float64
                )
            ),
            y_metric: float(
                np.mean(
                    [float(row[y_metric]) for row in matching_rows], dtype=np.float64
                )
            ),
            "scenario_count": float(len(matching_rows)),
        }
        if z_metric is not None:
            collected_row[z_metric] = float(
                np.mean(
                    [float(row[z_metric]) for row in matching_rows], dtype=np.float64
                )
            )
        collected_rows.append(collected_row)
    return collected_rows


def _annotate_heatmap(ax, matrix: np.ndarray, fmt: str) -> None:
    finite = matrix[np.isfinite(matrix)]
    midpoint = float(finite.mean()) if finite.size else 0.0
    for row_idx in range(matrix.shape[0]):
        for col_idx in range(matrix.shape[1]):
            value = matrix[row_idx, col_idx]
            if not np.isfinite(value):
                label = "n/a"
                color = "#555555"
            else:
                label = fmt.format(value)
                color = "white" if value >= midpoint else "black"
            ax.text(
                col_idx,
                row_idx,
                label,
                ha="center",
                va="center",
                fontsize=8,
                color=color,
            )


def save_pareto_plot(
    pareto_rows: list[dict[str, float | str]],
    path: Path,
    *,
    x_metric: str,
    y_metric: str,
    x_goal: str,
    y_goal: str,
    z_metric: str | None = None,
    z_goal: str | None = None,
    title: str,
    plot_config: PlotExportConfig | None = None,
) -> list[str]:
    if not pareto_rows:
        return []
    config = plot_config or PlotExportConfig()

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    metric_names = [x_metric, y_metric]
    goal_names = [x_goal, y_goal]
    if z_metric is not None:
        if z_goal is None:
            raise ValueError("A Pareto z-goal is required when a z-metric is provided.")
        metric_names.append(z_metric)
        goal_names.append(z_goal)

    points = np.asarray(
        [
            [float(row[metric_name]) for metric_name in metric_names]
            for row in pareto_rows
        ],
        dtype=np.float32,
    )
    frontier_mask = pareto_frontier_mask(points, goal_names)
    frontier_rows = [
        str(row["strategy"])
        for row, is_frontier in zip(pareto_rows, frontier_mask)
        if is_frontier
    ]

    colors = plt.get_cmap("tab10")(np.linspace(0.0, 0.95, len(pareto_rows)))
    is_3d = z_metric is not None
    with matplotlib.rc_context(plot_rcparams(config)):
        if is_3d:
            fig = plt.figure(
                figsize=(11.5, 8.5) if config.thesis_ready else (10, 7.5)
            )
            axis = fig.add_subplot(111, projection="3d")
            axis.scatter(
                points[:, 0],
                points[:, 1],
                points[:, 2],
                s=175 if config.thesis_ready else 150,
                c=colors,
                alpha=0.95,
                edgecolors="white",
                linewidths=1.2,
                depthshade=True,
            )

            if np.any(frontier_mask):
                frontier_points = points[frontier_mask]
                axis.scatter(
                    frontier_points[:, 0],
                    frontier_points[:, 1],
                    frontier_points[:, 2],
                    s=320 if config.thesis_ready else 300,
                    facecolors="none",
                    edgecolors="#111827",
                    linewidths=1.8,
                    depthshade=False,
                )

            for row, point in zip(pareto_rows, points):
                axis.text(
                    point[0],
                    point[1],
                    point[2],
                    wrap_plot_label(str(row["strategy_name"]), width=18),
                    fontsize=8.5 if config.thesis_ready else 8,
                )

            axis.set_title(title)
            axis.set_xlabel(
                f"{metric_label(x_metric)} ({'lower' if x_goal == 'min' else 'higher'} is better)"
            )
            axis.set_ylabel(
                f"{metric_label(y_metric)} ({'lower' if y_goal == 'min' else 'higher'} is better)"
            )
            axis.set_zlabel(
                f"{metric_label(z_metric)} ({'lower' if z_goal == 'min' else 'higher'} is better)"
            )
            axis.view_init(elev=22, azim=36)
            axis.text2D(
                0.02,
                0.98,
                "Circled points are Pareto-optimal.",
                transform=axis.transAxes,
                ha="left",
                va="top",
                fontsize=9,
                bbox={
                    "boxstyle": "round,pad=0.25",
                    "facecolor": "white",
                    "alpha": 0.9,
                    "edgecolor": "#d1d5db",
                },
            )
        else:
            fig, axis = plt.subplots(
                figsize=(12.5, 8.5) if config.thesis_ready else (11, 8)
            )
            axis.scatter(
                points[:, 0],
                points[:, 1],
                s=175 if config.thesis_ready else 150,
                c=colors,
                alpha=0.95,
                edgecolors="white",
                linewidths=1.2,
                zorder=3,
            )

            if np.any(frontier_mask):
                frontier_points = points[frontier_mask]
                frontier_order = np.argsort(frontier_points[:, 0])
                ordered_points = frontier_points[frontier_order]
                axis.plot(
                    ordered_points[:, 0],
                    ordered_points[:, 1],
                    color="#111827",
                    linewidth=1.8,
                    linestyle="--",
                    zorder=2,
                )
                axis.scatter(
                    frontier_points[:, 0],
                    frontier_points[:, 1],
                    s=280 if config.thesis_ready else 260,
                    facecolors="none",
                    edgecolors="#111827",
                    linewidths=1.8,
                    zorder=4,
                )

            for row, point in zip(pareto_rows, points):
                axis.annotate(
                    wrap_plot_label(str(row["strategy_name"]), width=18),
                    (point[0], point[1]),
                    textcoords="offset points",
                    xytext=(7, 7),
                    fontsize=9.5 if config.thesis_ready else 9,
                )

            axis.set_title(title)
            axis.set_xlabel(
                f"{metric_label(x_metric)} ({'lower' if x_goal == 'min' else 'higher'} is better)"
            )
            axis.set_ylabel(
                f"{metric_label(y_metric)} ({'lower' if y_goal == 'min' else 'higher'} is better)"
            )
            axis.grid(True, alpha=0.3)
            axis.set_axisbelow(True)
            axis.margins(x=0.12, y=0.12)
            axis.text(
                0.02,
                0.98,
                "Circled points are Pareto-optimal.",
                transform=axis.transAxes,
                ha="left",
                va="top",
                fontsize=9,
                bbox={
                    "boxstyle": "round,pad=0.25",
                    "facecolor": "white",
                    "alpha": 0.9,
                    "edgecolor": "#d1d5db",
                },
            )

        if is_3d:
            fig.subplots_adjust(left=0.04, right=0.96, bottom=0.06, top=0.92)
            export_figure(fig, path, config, bbox_inches=None)
        else:
            fig.tight_layout()
            export_figure(fig, path, config, bbox_inches="tight")
    plt.close(fig)
    return frontier_rows


def save_pareto_plots(
    summary_rows: list[dict[str, float | str]],
    scenario_keys: list[str],
    strategy_keys: list[str],
    output_prefix: Path,
    *,
    x_metric: str,
    y_metric: str,
    x_goal: str,
    y_goal: str,
    z_metric: str | None = None,
    z_goal: str | None = None,
    plot_config: PlotExportConfig | None = None,
) -> tuple[Path | None, list[Path], dict[str, list[str]]]:
    if not summary_rows:
        return None, [], {}
    config = plot_config or PlotExportConfig()

    overall_path = output_prefix.with_name(f"{output_prefix.name}_pareto_overall.png")
    scenario_dir = output_prefix.with_name(f"{output_prefix.name}_pareto_plots")
    frontier_lookup: dict[str, list[str]] = {}

    overall_rows = collect_pareto_rows(
        summary_rows,
        scenario_keys,
        strategy_keys,
        scenario_key=None,
        x_metric=x_metric,
        y_metric=y_metric,
        z_metric=z_metric,
    )
    overall_frontier = save_pareto_plot(
        overall_rows,
        overall_path,
        x_metric=x_metric,
        y_metric=y_metric,
        x_goal=x_goal,
        y_goal=y_goal,
        z_metric=z_metric,
        z_goal=z_goal,
        title=(
            "3D Pareto Frontier Across All Scenarios"
            if z_metric is not None
            else "Pareto Frontier Across All Scenarios"
        ),
        plot_config=config,
    )
    if overall_frontier:
        frontier_lookup["all"] = overall_frontier

    scenario_paths: list[Path] = []
    for scenario_key in scenario_keys:
        pareto_rows = collect_pareto_rows(
            summary_rows,
            scenario_keys,
            strategy_keys,
            scenario_key=scenario_key,
            x_metric=x_metric,
            y_metric=y_metric,
            z_metric=z_metric,
        )
        output_path = scenario_dir / f"{scenario_key}_pareto.png"
        frontier = save_pareto_plot(
            pareto_rows,
            output_path,
            x_metric=x_metric,
            y_metric=y_metric,
            x_goal=x_goal,
            y_goal=y_goal,
            z_metric=z_metric,
            z_goal=z_goal,
            title=(
                f"3D Pareto Frontier: {SCENARIOS[scenario_key].name}"
                if z_metric is not None
                else f"Pareto Frontier: {SCENARIOS[scenario_key].name}"
            ),
            plot_config=config,
        )
        if frontier:
            frontier_lookup[scenario_key] = frontier
            scenario_paths.append(output_path)

    return (
        overall_path if overall_frontier else None,
        scenario_paths,
        frontier_lookup,
    )


def save_heatmap_grid(
    summary_rows: list[dict[str, float | str]],
    scenario_keys: list[str],
    strategy_keys: list[str],
    path: Path,
    plot_config: PlotExportConfig | None = None,
) -> None:
    if not summary_rows:
        return
    config = plot_config or PlotExportConfig()

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    with matplotlib.rc_context(plot_rcparams(config)):
        n_metrics = len(PLOT_METRICS)
        n_cols = 2
        n_rows = int(np.ceil(n_metrics / n_cols))
        fig, axes = plt.subplots(
            n_rows,
            n_cols,
            figsize=(18, max(5.5 * n_rows, 10.5))
            if config.thesis_ready
            else (16, max(5 * n_rows, 10)),
        )
        axes_flat = np.atleast_1d(axes).flatten()
        scenario_labels = [SCENARIOS[key].name for key in scenario_keys]
        strategy_labels = [wrap_plot_label(STRATEGIES[key].name, width=18) for key in strategy_keys]

        for idx, (axis, (metric_name, title, cmap_name, fmt)) in enumerate(
            zip(axes_flat, PLOT_METRICS)
        ):
            matrix = pivot_metric(summary_rows, scenario_keys, strategy_keys, metric_name)
            image = axis.imshow(matrix, aspect="auto", cmap=cmap_name)
            axis.set_title(title)
            axis.set_xticks(np.arange(len(scenario_labels)))
            # Only the bottom row of panels carries scenario tick labels; clearing
            # the rest removes the redundant, overlapping x-axis text that
            # previously repeated (and visually doubled) on every panel.
            if idx >= len(PLOT_METRICS) - n_cols:
                axis.set_xticklabels(scenario_labels, rotation=30, ha="right", fontsize=9)
            else:
                axis.set_xticklabels([])
            axis.set_yticks(np.arange(len(strategy_labels)))
            axis.set_yticklabels(strategy_labels)
            _annotate_heatmap(axis, matrix, fmt)
            fig.colorbar(image, ax=axis, fraction=0.046, pad=0.04)

        for axis in axes_flat[len(PLOT_METRICS) :]:
            axis.axis("off")

        fig.suptitle(
            "Strategy Performance Across Scenarios",
            fontsize=17 if config.thesis_ready else 16,
        )
        fig.tight_layout(rect=(0, 0, 1, 0.96))
        export_figure(fig, path, config, bbox_inches="tight")
    plt.close(fig)


def save_scenario_plots(
    summary_rows: list[dict[str, float | str]],
    scenario_keys: list[str],
    strategy_keys: list[str],
    output_dir: Path,
    plot_config: PlotExportConfig | None = None,
) -> list[Path]:
    if not summary_rows:
        return []
    config = plot_config or PlotExportConfig()

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output_dir.mkdir(parents=True, exist_ok=True)
    saved_paths: list[Path] = []
    colors = plt.get_cmap("Set3")(np.linspace(0.08, 0.92, len(strategy_keys)))
    strategy_labels = [wrap_plot_label(STRATEGIES[key].name, width=18) for key in strategy_keys]
    metrics = SCENARIO_PLOT_METRICS

    with matplotlib.rc_context(plot_rcparams(config)):
        for scenario_key in scenario_keys:
            scenario_rows = {
                str(row["strategy"]): row
                for row in summary_rows
                if row["scenario"] == scenario_key
            }
            if not scenario_rows:
                continue

            x = np.arange(len(strategy_keys), dtype=np.float32)
            n_cols = 2
            n_rows = int(np.ceil(len(metrics) / n_cols))
            fig, axes = plt.subplots(
                n_rows,
                n_cols,
                figsize=(18, max(5.5 * n_rows, 10.5))
                if config.thesis_ready
                else (16, max(5 * n_rows, 10)),
            )
            axes_flat = np.atleast_1d(axes).flatten()
            for axis, (metric_name, title) in zip(axes_flat, metrics):
                means = [
                    float(scenario_rows.get(strategy_key, {}).get(metric_name, np.nan))
                    for strategy_key in strategy_keys
                ]
                std_name = metric_name.replace("_mean", "_std")
                stds = [
                    float(scenario_rows.get(strategy_key, {}).get(std_name, 0.0))
                    for strategy_key in strategy_keys
                ]
                axis.bar(
                    x,
                    means,
                    yerr=stds,
                    color=colors,
                    alpha=0.92,
                    capsize=3,
                )
                axis.set_title(title)
                axis.set_xticks(x)
                axis.set_xticklabels(strategy_labels, rotation=28, ha="right")
                axis.grid(axis="y", alpha=0.25)

            for axis in axes_flat[len(metrics) :]:
                axis.axis("off")

            fig.suptitle(
                f"Strategy Comparison: {SCENARIOS[scenario_key].name}",
                fontsize=17 if config.thesis_ready else 16,
            )
            fig.tight_layout(rect=(0, 0, 1, 0.95))
            output_path = output_dir / f"{scenario_key}_comparison.png"
            export_figure(fig, output_path, config, bbox_inches="tight")
            plt.close(fig)
            saved_paths.append(output_path)

    return saved_paths


def aggregate_overall_rows(
    summary_rows: list[dict[str, float | str]],
    scenario_keys: list[str],
    strategy_keys: list[str],
) -> list[dict[str, float | str]]:
    allowed_scenarios = set(scenario_keys)
    aggregated: list[dict[str, float | str]] = []
    for strategy_key in strategy_keys:
        matching_rows = [
            row
            for row in summary_rows
            if str(row["strategy"]) == strategy_key
            and str(row["scenario"]) in allowed_scenarios
        ]
        if not matching_rows:
            continue
        aggregated_row: dict[str, float | str] = {
            "strategy": strategy_key,
            "strategy_name": STRATEGIES[strategy_key].name,
            "scenario_count": float(len(matching_rows)),
        }
        for metric_name in SUMMARY_MEAN_METRICS:
            values = [
                float(row[metric_name])
                for row in matching_rows
                if metric_name in row
            ]
            if values:
                aggregated_row[metric_name] = float(
                    np.mean(values, dtype=np.float64)
                )
        aggregated.append(aggregated_row)
    return aggregated


def scenario_win_rows(
    summary_rows: list[dict[str, float | str]],
    scenario_keys: list[str],
) -> list[dict[str, float | str]]:
    wins: dict[str, int] = defaultdict(int)
    for scenario_key in scenario_keys:
        rows = [
            row for row in summary_rows if str(row["scenario"]) == str(scenario_key)
        ]
        if not rows:
            continue
        rows.sort(
            key=lambda row: (
                float(row.get("shortage_rate_mean", np.inf)),
                -float(row.get("service_rate_mean", -np.inf)),
                -float(row.get("episode_score_mean", -np.inf)),
                -float(row.get("reward_total_mean", -np.inf)),
            )
        )
        wins[str(rows[0]["strategy"])] += 1

    rows: list[dict[str, float | str]] = []
    for strategy_key, count in sorted(
        wins.items(),
        key=lambda item: (-item[1], STRATEGIES[item[0]].name),
    ):
        rows.append(
            {
                "strategy": strategy_key,
                "strategy_name": STRATEGIES[strategy_key].name,
                "scenario_wins": float(count),
            }
        )
    return rows


def save_thesis_overview_plot(
    overall_rows: list[dict[str, float | str]],
    path: Path,
    plot_config: PlotExportConfig,
) -> None:
    if not overall_rows:
        return

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    with matplotlib.rc_context(plot_rcparams(plot_config)):
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        axes_flat = np.atleast_1d(axes).flatten()
        palette = plt.get_cmap("tab20")(np.linspace(0.05, 0.95, len(overall_rows)))

        for axis, (metric_name, title, goal) in zip(axes_flat, THESIS_OVERVIEW_METRICS):
            rows = [
                row
                for row in overall_rows
                if metric_name in row and np.isfinite(float(row[metric_name]))
            ]
            if not rows:
                axis.axis("off")
                continue
            rows.sort(
                key=lambda row: float(row[metric_name]),
                reverse=(goal == "max"),
            )
            labels = [wrap_plot_label(str(row["strategy_name"]), width=20) for row in rows]
            values = [float(row[metric_name]) for row in rows]
            positions = np.arange(len(rows))
            axis.barh(positions, values, color=palette[: len(rows)], alpha=0.92)
            axis.set_yticks(positions)
            axis.set_yticklabels(labels)
            axis.invert_yaxis()
            axis.set_title(title)
            axis.grid(axis="x", alpha=0.28)
            axis.set_axisbelow(True)
            value_fmt = "{:.1f}" if abs(max(values, key=abs)) < 10_000 else "{:.0f}"
            for pos, value in zip(positions, values):
                axis.text(
                    value,
                    pos,
                    "  " + value_fmt.format(value),
                    va="center",
                    ha="left",
                    fontsize=9,
                    color="#0f172a",
                )

        fig.suptitle(
            "All-Strategy Overview Across Evaluated Scenarios",
            fontsize=18,
            fontweight="bold",
        )
        fig.tight_layout(rect=(0, 0, 1, 0.96))
        export_figure(fig, path, plot_config, bbox_inches="tight")
        plt.close(fig)


def save_thesis_component_shortage_plot(
    overall_rows: list[dict[str, float | str]],
    path: Path,
    plot_config: PlotExportConfig,
) -> None:
    if not overall_rows:
        return

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    with matplotlib.rc_context(plot_rcparams(plot_config)):
        rows = [
            row
            for row in overall_rows
            if any(
                metric_name in row
                for metric_name in (
                    "shortage_rbc_mean",
                    "shortage_platelets_mean",
                    "shortage_plasma_mean",
                )
            )
        ]
        if not rows:
            return
        rows.sort(key=lambda row: float(row.get("shortage_rate_mean", np.inf)))
        labels = [wrap_plot_label(str(row["strategy_name"]), width=20) for row in rows]
        rbc = np.asarray(
            [float(row.get("shortage_rbc_mean", 0.0)) for row in rows],
            dtype=np.float32,
        )
        platelets = np.asarray(
            [float(row.get("shortage_platelets_mean", 0.0)) for row in rows],
            dtype=np.float32,
        )
        plasma = np.asarray(
            [float(row.get("shortage_plasma_mean", 0.0)) for row in rows],
            dtype=np.float32,
        )
        positions = np.arange(len(rows))
        fig, axis = plt.subplots(figsize=(13, max(7, len(rows) * 0.6)))
        axis.barh(positions, rbc, color="#dc2626", label="RBC")
        axis.barh(positions, platelets, left=rbc, color="#f59e0b", label="Platelets")
        axis.barh(
            positions,
            plasma,
            left=rbc + platelets,
            color="#2563eb",
            label="Plasma",
        )
        axis.set_yticks(positions)
        axis.set_yticklabels(labels)
        axis.invert_yaxis()
        axis.set_xlabel("Average shortage units")
        axis.set_title("Shortage Composition by Blood Component")
        axis.grid(axis="x", alpha=0.28)
        axis.legend(loc="lower right")
        fig.tight_layout()
        export_figure(fig, path, plot_config, bbox_inches="tight")
        plt.close(fig)


def save_thesis_scenario_wins_plot(
    win_rows: list[dict[str, float | str]],
    path: Path,
    plot_config: PlotExportConfig,
) -> None:
    if not win_rows:
        return

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    with matplotlib.rc_context(plot_rcparams(plot_config)):
        labels = [wrap_plot_label(str(row["strategy_name"]), width=20) for row in win_rows]
        values = [float(row["scenario_wins"]) for row in win_rows]
        positions = np.arange(len(win_rows))
        fig, axis = plt.subplots(figsize=(12.5, max(6, len(win_rows) * 0.55)))
        axis.barh(positions, values, color="#0f766e", alpha=0.92)
        axis.set_yticks(positions)
        axis.set_yticklabels(labels)
        axis.invert_yaxis()
        axis.set_xlabel("Scenario wins")
        axis.set_title("Scenario Winners by Primary Ranking Metric")
        axis.grid(axis="x", alpha=0.28)
        for pos, value in zip(positions, values):
            axis.text(value, pos, f"  {int(round(value))}", va="center", ha="left")
        fig.tight_layout()
        export_figure(fig, path, plot_config, bbox_inches="tight")
        plt.close(fig)


def save_artifact_manifest(payload: dict[str, object], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def strategy_status_rows(
    requested_strategy_keys: list[str],
    status_lookup: dict[str, dict[str, str]],
) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for strategy_key in requested_strategy_keys:
        strategy = STRATEGIES[strategy_key]
        state = status_lookup.get(strategy_key, {})
        rows.append(
            {
                "strategy": strategy_key,
                "strategy_name": strategy.name,
                "status": state.get("status", "unknown"),
                "detail": state.get("detail", ""),
            }
        )
    return rows


def print_summary(
    summary_rows: list[dict[str, float | str]],
    scenario_keys: list[str],
) -> None:
    for scenario_key in scenario_keys:
        print(f"\n=== {scenario_key} ===")
        rows = [
            row for row in summary_rows if str(row["scenario"]) == str(scenario_key)
        ]
        rows.sort(
            key=lambda row: (
                float(row["shortage_rate_mean"]),
                -float(row["episode_score_mean"]),
            )
        )
        for row in rows:
            print(
                f"{str(row['strategy']):>22s} | score={float(row['episode_score_mean']):>9.2f} "
                f"| final_reward={float(row['reward_total_mean']):>9.2f} "
                f"| shortage={float(row['shortage_rate_mean']):>7.2f}% "
                f"| demand={float(row['total_net_requested_mean']):>7.1f} "
                f"| cost={float(row['active_action_cost_mean']):>7.1f} "
                f"| exact={float(row['exact_match_rate_mean']):>6.1f}% "
                f"| compat_sub={float(row['compatible_substitution_rate_mean']):>6.1f}%"
            )


def print_pareto_summary(frontier_lookup: dict[str, list[str]]) -> None:
    if not frontier_lookup:
        return
    print("\nPareto-optimal strategies:")
    for scenario_key, strategy_names in frontier_lookup.items():
        label = "all scenarios" if scenario_key == "all" else scenario_key
        print(f"  {label:>20s} | {', '.join(strategy_names)}")


def main() -> None:
    args = parse_args()
    # Populate the sensitivity-sweep overrides (no-ops at their defaults).
    if args.delay_scale != 1.0:
        _SWEEP_OVERRIDES["delay_scale"] = args.delay_scale
    if args.ramp_scale != 1.0:
        _SWEEP_OVERRIDES["ramp_scale"] = args.ramp_scale
    if (args.reward_safety_scale, args.reward_matching_scale, args.reward_waste_scale) != (1.0, 1.0, 1.0):
        _SWEEP_OVERRIDES["reward_tier_weights"] = {
            "safety": args.reward_safety_scale,
            "matching": args.reward_matching_scale,
            "waste": args.reward_waste_scale,
        }
    if _SWEEP_OVERRIDES:
        print(f"[evaluate_strategies] Sensitivity overrides active: {_SWEEP_OVERRIDES}")
    scenario_keys = scenario_keys_from_args(args)
    requested_strategy_keys = strategy_keys_from_args(args)
    strategy_keys = list(requested_strategy_keys)
    plot_config = build_plot_export_config(args)
    sim_context = load_training_context()
    status_lookup = {
        strategy_key: {
            "status": "pending",
            "detail": "Requested for evaluation.",
        }
        for strategy_key in requested_strategy_keys
    }

    dreamer_agents: dict[str, object] = {}
    dreamer_checkpoints = {
        "dreamerv3_official": args.dreamerv3_checkpoint,
        "dreamerv4": args.dreamerv4_checkpoint,
    }
    if "dreamerv3_official" in strategy_keys:
        resolved_dreamerv3_path = Path(
            resolve_dreamer_checkpoint_path(
                "dreamerv3_official",
                args.dreamerv3_checkpoint,
            )
        ).expanduser()
        if not resolved_dreamerv3_path.exists():
            detail = (
                f"DreamerV3 checkpoint path does not exist: {resolved_dreamerv3_path}"
            )
            import sys as _sys
            print(f"[SKIPPED] dreamerv3_official — {detail}", file=_sys.stderr)
            status_lookup["dreamerv3_official"] = {
                "status": "skipped",
                "detail": detail,
            }
            strategy_keys = [s for s in strategy_keys if s != "dreamerv3_official"]
        else:
            dreamer_checkpoints["dreamerv3_official"] = str(resolved_dreamerv3_path)
    if "dreamerv3_official" in strategy_keys:
        try:
            dreamer_agents["dreamerv3_official"] = load_official_dreamerv3_agent(
                dreamer_checkpoints["dreamerv3_official"],
                seed=args.seed_start,
            )
        except Exception as exc:
            import sys as _sys
            print(
                f"[SKIPPED] dreamerv3_official — {exc}", file=_sys.stderr
            )
            status_lookup["dreamerv3_official"] = {
                "status": "skipped",
                "detail": f"DreamerV3 checkpoint unavailable: {exc}",
            }
            strategy_keys = [s for s in strategy_keys if s != "dreamerv3_official"]
        else:
            status_lookup["dreamerv3_official"] = {
                "status": "ready",
                "detail": "DreamerV3 controller loaded successfully.",
            }
    if "dreamerv4" in strategy_keys:
        resolved_dreamerv4_path = (
            Path(args.dreamerv4_checkpoint).expanduser()
            if args.dreamerv4_checkpoint
            else None
        )
        has_native_dreamerv4 = any(
            candidate.is_file()
            for candidate in (SIMULATOR_ROOT / "dreamerv4_runs").rglob("*.pt")
        )
        if resolved_dreamerv4_path is not None and not resolved_dreamerv4_path.exists():
            detail = (
                f"DreamerV4 checkpoint path does not exist: {resolved_dreamerv4_path}"
            )
            import sys as _sys
            print(f"[SKIPPED] dreamerv4 — {detail}", file=_sys.stderr)
            status_lookup["dreamerv4"] = {
                "status": "skipped",
                "detail": detail,
            }
            strategy_keys = [s for s in strategy_keys if s != "dreamerv4"]
        elif resolved_dreamerv4_path is None and not has_native_dreamerv4:
            detail = "No native DreamerV4 checkpoint was discovered under dreamerv4_runs."
            import sys as _sys
            print(f"[SKIPPED] dreamerv4 — {detail}", file=_sys.stderr)
            status_lookup["dreamerv4"] = {
                "status": "skipped",
                "detail": detail,
            }
            strategy_keys = [s for s in strategy_keys if s != "dreamerv4"]
        elif resolved_dreamerv4_path is not None:
            dreamer_checkpoints["dreamerv4"] = str(resolved_dreamerv4_path)
    if "dreamerv4" in strategy_keys:
        try:
            dreamer_agents["dreamerv4"] = load_dreamerv4_agent(
                dreamer_checkpoints["dreamerv4"],
                seed=args.seed_start,
            )
        except Exception as exc:
            import sys as _sys
            print(
                f"[SKIPPED] dreamerv4 — {exc}", file=_sys.stderr
            )
            status_lookup["dreamerv4"] = {
                "status": "skipped",
                "detail": f"DreamerV4 checkpoint unavailable: {exc}",
            }
            strategy_keys = [s for s in strategy_keys if s != "dreamerv4"]
        else:
            status_lookup["dreamerv4"] = {
                "status": "ready",
                "detail": "DreamerV4 controller loaded successfully.",
            }

    continuous_model_overrides = {
        "ppo_continuous": args.ppo_continuous_model,
        "sac_continuous": args.sac_model,
        "iql_offline": args.iql_model,
        "cql_offline": args.cql_model,
    }
    continuous_agents: dict[str, object] = {}
    for controller_key in list(strategy_keys):
        if controller_key not in LEARNED_CONTINUOUS_CONTROLLER_KEYS:
            continue

        model_path = (
            continuous_model_overrides.get(controller_key)
            or resolve_learned_continuous_model_path(controller_key)
        )
        if model_path is None:
            detail = (
                f"{controller_key} model not available in the configured locations."
            )
            print(
                f"[evaluate_strategies] {controller_key} model not available — removing {controller_key} strategy."
            )
            status_lookup[controller_key] = {
                "status": "skipped",
                "detail": detail,
            }
            strategy_keys = [s for s in strategy_keys if s != controller_key]
            continue

        try:
            _G, _n, _s, _e, _w = sim_context
            continuous_agents[controller_key] = load_learned_continuous_agent(
                controller_key,
                model_path,
                params=copy.deepcopy(SCENARIOS["baseline"]),
                G=_G,
                north=_n,
                south=_s,
                east=_e,
                west=_w,
            )
            print(
                f"[evaluate_strategies] Loaded {controller_key} model from {model_path}"
            )
            status_lookup[controller_key] = {
                "status": "ready",
                "detail": f"Loaded model from {model_path}",
            }
        except Exception as exc:
            print(
                f"[evaluate_strategies] Failed to load {controller_key} model: {exc}"
            )
            status_lookup[controller_key] = {
                "status": "skipped",
                "detail": f"Model loading failed: {exc}",
            }
            strategy_keys = [s for s in strategy_keys if s != controller_key]

    for strategy_key in strategy_keys:
        if status_lookup.get(strategy_key, {}).get("status") == "pending":
            status_lookup[strategy_key] = {
                "status": "ready",
                "detail": "Strategy is available for evaluation.",
            }

    rows: list[dict[str, float | int | str]] = []
    for scenario_key in scenario_keys:
        for strategy_key in strategy_keys:
            print(f"[evaluate_strategies] {scenario_key} | {strategy_key}")
            for run_idx in range(args.runs):
                seed = args.seed_start + run_idx
                state = run_episode(
                    strategy_key,
                    scenario_key,
                    sim_context,
                    seed=seed,
                    hours_override=args.hours,
                    dreamer_agents=dreamer_agents,
                    dreamer_checkpoints=dreamer_checkpoints,
                    continuous_agents=continuous_agents,
                    ppo_step_hours=args.ppo_step_hours,
                )
                rows.append(extract_metrics(strategy_key, scenario_key, seed, state))

    for strategy_key in strategy_keys:
        status_lookup[strategy_key] = {
            "status": "evaluated",
            "detail": "Completed evaluation runs successfully.",
        }

    summary_rows = aggregate_rows(rows)
    overall_rows = aggregate_overall_rows(summary_rows, scenario_keys, strategy_keys)
    win_rows = scenario_win_rows(summary_rows, scenario_keys)
    prefix = resolve_output_prefix(args.output_prefix)
    detail_path = prefix.with_name(f"{prefix.name}_detail.csv")
    summary_path = prefix.with_name(f"{prefix.name}_summary.csv")
    overall_summary_path = prefix.with_name(f"{prefix.name}_overall_summary.csv")
    status_path = prefix.with_name(f"{prefix.name}_strategy_status.csv")
    heatmap_path = prefix.with_name(f"{prefix.name}_heatmaps.png")
    scenario_dir = prefix.with_name(f"{prefix.name}_scenario_plots")
    thesis_overview_path = prefix.with_name(f"{prefix.name}_thesis_overview.png")
    thesis_component_path = prefix.with_name(
        f"{prefix.name}_thesis_component_shortages.png"
    )
    thesis_wins_path = prefix.with_name(f"{prefix.name}_thesis_scenario_wins.png")
    manifest_path = prefix.with_name(f"{prefix.name}_artifacts.json")
    pareto_overall_path, pareto_paths, frontier_lookup = save_pareto_plots(
        summary_rows,
        scenario_keys,
        strategy_keys,
        prefix,
        x_metric=args.pareto_x_metric,
        y_metric=args.pareto_y_metric,
        x_goal=args.pareto_x_goal,
        y_goal=args.pareto_y_goal,
        z_metric=args.pareto_z_metric,
        z_goal=args.pareto_z_goal,
        plot_config=plot_config,
    )

    save_csv(rows, detail_path)
    save_csv(summary_rows, summary_path)
    save_csv(overall_rows, overall_summary_path)
    save_csv(strategy_status_rows(requested_strategy_keys, status_lookup), status_path)
    save_heatmap_grid(
        summary_rows,
        scenario_keys,
        strategy_keys,
        heatmap_path,
        plot_config=plot_config,
    )
    scenario_paths = save_scenario_plots(
        summary_rows,
        scenario_keys,
        strategy_keys,
        scenario_dir,
        plot_config=plot_config,
    )
    if plot_config.thesis_ready:
        save_thesis_overview_plot(overall_rows, thesis_overview_path, plot_config)
        save_thesis_component_shortage_plot(
            overall_rows,
            thesis_component_path,
            plot_config,
        )
        save_thesis_scenario_wins_plot(win_rows, thesis_wins_path, plot_config)
    save_artifact_manifest(
        {
            "output_prefix": str(prefix),
            "scenario_keys": scenario_keys,
            "requested_strategy_keys": requested_strategy_keys,
            "evaluated_strategy_keys": strategy_keys,
            "plot_formats": list(plot_config.formats),
            "thesis_ready": plot_config.thesis_ready,
            "detail_csv": str(detail_path),
            "summary_csv": str(summary_path),
            "overall_summary_csv": str(overall_summary_path),
            "strategy_status_csv": str(status_path),
            "heatmap_base": str(heatmap_path),
            "scenario_plot_dir": str(scenario_dir),
            "pareto_overall_base": str(pareto_overall_path) if pareto_overall_path else None,
            "pareto_plot_dir": str(prefix.with_name(f"{prefix.name}_pareto_plots")),
            "thesis_overview_base": str(thesis_overview_path)
            if plot_config.thesis_ready
            else None,
            "thesis_component_base": str(thesis_component_path)
            if plot_config.thesis_ready
            else None,
            "thesis_scenario_wins_base": str(thesis_wins_path)
            if plot_config.thesis_ready
            else None,
        },
        manifest_path,
    )

    print_summary(summary_rows, scenario_keys)
    print_pareto_summary(frontier_lookup)
    print(f"\nSaved detail CSV to {detail_path}")
    print(f"Saved summary CSV to {summary_path}")
    print(f"Saved overall summary CSV to {overall_summary_path}")
    print(f"Saved strategy status CSV to {status_path}")
    print(f"Saved heatmap plot to {heatmap_path}")
    if scenario_paths:
        print(f"Saved {len(scenario_paths)} scenario plot(s) to {scenario_dir}")
    if pareto_overall_path is not None:
        print(f"Saved overall Pareto plot to {pareto_overall_path}")
    if pareto_paths:
        print(
            f"Saved {len(pareto_paths)} Pareto scenario plot(s) to "
            f"{prefix.with_name(f'{prefix.name}_pareto_plots')}"
        )
    if plot_config.thesis_ready:
        print(f"Saved thesis overview plot to {thesis_overview_path}")
        print(f"Saved thesis component plot to {thesis_component_path}")
        print(f"Saved thesis scenario wins plot to {thesis_wins_path}")
    print(f"Saved artifact manifest to {manifest_path}")

    skipped = [s for s in requested_strategy_keys if s not in strategy_keys]
    if skipped:
        import sys as _sys
        msg = (
            f"\nEvaluated {len(strategy_keys)}/{len(requested_strategy_keys)} "
            f"requested strategies. "
            f"Skipped: [{', '.join(skipped)}]"
        )
        print(msg, file=_sys.stderr)


if __name__ == "__main__":
    main()
