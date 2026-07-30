from __future__ import annotations

import argparse
import copy
import csv
import os
import tempfile
from collections import defaultdict
from pathlib import Path

import numpy as np

CACHE_ROOT = Path(tempfile.gettempdir()) / "pios_mappo_cache"
CACHE_ROOT.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("XDG_CACHE_HOME", str(CACHE_ROOT))
os.environ.setdefault("MPLCONFIGDIR", str(CACHE_ROOT / "matplotlib"))

from eval_metrics import (
    AGENT_COMPARE_PLOT_METRICS,
    DETAIL_METRICS,
    extract_eval_metrics,
)
from engine import (
    apply_multi_actions,
    load_ppo_agent,
    policy_snapshot,
    run_scenario,
    step_simulation,
    summarize_state,
)
from mappo import MAPPO
from scenarios import SCENARIOS
from train_mappo import (
    MANAGER_AGENT_NAME,
    MAPPO_ACTION_KEYS,
    ROLE_ACTION_KEYS,
    ROLE_AGENT_NAMES,
    build_action_prior_biases,
    build_dynamic_action_masks,
    build_mappo_obs,
    build_mappo_step_inputs,
    default_manager_context,
    load_training_context,
    manager_signal_from_action,
    prepare_mappo_scenario,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare trained MAPPO and single-agent PPO on the same scenarios."
    )
    parser.add_argument(
        "--mappo-model",
        type=str,
        default="mappo_model.pt",
        help="Path to the MAPPO checkpoint (.pt).",
    )
    parser.add_argument(
        "--ppo-model",
        type=str,
        default="ppo_fast_model.zip",
        help="Path to the trained PPO checkpoint. If missing, PPO is skipped.",
    )
    parser.add_argument(
        "--include-heuristic-ppo",
        action="store_true",
        help="Also compare against the legacy heuristic PPO controller.",
    )
    parser.add_argument(
        "--runs",
        type=int,
        default=3,
        help="How many seeds to evaluate per scenario.",
    )
    parser.add_argument(
        "--seed-start",
        type=int,
        default=100,
        help="Base seed for evaluation runs.",
    )
    parser.add_argument(
        "--ppo-step-hours",
        type=float,
        default=6.0,
        help="Decision interval for the PPO controller.",
    )
    parser.add_argument(
        "--mappo-prior-scale",
        type=float,
        default=0.20,
        help="Prior bias scale used during MAPPO inference.",
    )
    parser.add_argument(
        "--scenario",
        action="append",
        choices=sorted(SCENARIOS.keys()),
        help="Scenario key to evaluate. Repeat to restrict the scenario set.",
    )
    parser.add_argument(
        "--output-prefix",
        type=str,
        default="policy_compare",
        help="Prefix for comparison artifacts.",
    )
    return parser.parse_args()


def scenario_list_from_args(args: argparse.Namespace) -> list[str]:
    return list(args.scenario) if args.scenario else list(SCENARIOS.keys())


def extract_metrics(model_name: str, scenario_key: str, seed: int, state) -> dict[str, float | int | str]:
    summary = summarize_state(state)
    row: dict[str, float | int | str] = {
        "model": model_name,
        "scenario": scenario_key,
        "seed": seed,
    }
    row.update(extract_eval_metrics(summary, state))
    return row


def run_mappo_episode(
    mappo: MAPPO,
    scenario,
    sim_context,
    *,
    seed: int,
    prior_scale: float,
    deterministic: bool = True,
):
    G, north, south, east, west = sim_context
    prepared_scenario = prepare_mappo_scenario(scenario)
    state = run_scenario(
        prepared_scenario,
        G,
        north,
        south,
        east,
        west,
        seed=seed,
        enable_logs=False,
        fast_mode=True,
    )

    prev_snapshot = policy_snapshot(state)
    manager_context = default_manager_context()

    while state.env.now < state.params.sim_hours:
        raw, signals, manager_obs, manager_mask, manager_bias = build_mappo_step_inputs(
            state,
            prev_snapshot,
            manager_context=manager_context,
        )
        scaled_manager_bias = np.asarray(manager_bias, dtype=np.float32) * float(prior_scale)
        manager_action, _, _ = mappo.act_single(
            MANAGER_AGENT_NAME,
            manager_obs,
            update_stats=False,
            action_mask=manager_mask,
            action_bias=scaled_manager_bias,
            deterministic=deterministic,
        )
        manager_signal = manager_signal_from_action(
            manager_action,
            state,
            raw,
            signals,
            manager_context=manager_context,
        )
        obs_dict = build_mappo_obs(state, prev_snapshot, manager_signal=manager_signal)
        action_masks = build_dynamic_action_masks(
            state,
            raw,
            signals,
            manager_signal=manager_signal,
        )
        action_biases = build_action_prior_biases(
            state,
            raw,
            signals,
            action_masks,
            manager_signal=manager_signal,
        )
        scaled_biases = {
            name: np.asarray(action_biases[name], dtype=np.float32) * float(prior_scale)
            for name in ROLE_AGENT_NAMES
        }
        actions, _, _ = mappo.act(
            {name: obs_dict[name] for name in ROLE_AGENT_NAMES},
            update_stats=False,
            action_masks=action_masks,
            action_biases=scaled_biases,
            deterministic=deterministic,
        )

        apply_multi_actions(
            state,
            actions,
            action_keys=MAPPO_ACTION_KEYS,
            allowed_action_keys=ROLE_ACTION_KEYS,
        )
        step_simulation(state)
        prev_snapshot = policy_snapshot(state)
        manager_context = manager_signal

    return state


def run_trained_ppo_episode(agent, scenario, sim_context, *, seed: int, step_hours: float):
    G, north, south, east, west = sim_context
    prepared_scenario = prepare_mappo_scenario(scenario)
    state = run_scenario(
        prepared_scenario,
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
    while state.env.now < state.params.sim_hours:
        step_simulation(state, step_hours=step_hours)
    return state


def run_heuristic_ppo_episode(scenario, sim_context, *, seed: int, step_hours: float):
    G, north, south, east, west = sim_context
    prepared_scenario = prepare_mappo_scenario(scenario)
    prepared_scenario.strategy_key = "ppo_shortage_minimizer"
    state = run_scenario(
        prepared_scenario,
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
    while state.env.now < state.params.sim_hours:
        step_simulation(state, step_hours=step_hours)
    return state


def aggregate_rows(rows: list[dict[str, float | int | str]]) -> list[dict[str, float | str]]:
    grouped: dict[tuple[str, str], dict[str, list[float]]] = defaultdict(
        lambda: defaultdict(list)
    )
    metric_names = DETAIL_METRICS

    for row in rows:
        key = (str(row["model"]), str(row["scenario"]))
        for metric in metric_names:
            grouped[key][metric].append(float(row[metric]))

    summary_rows: list[dict[str, float | str]] = []
    for (model, scenario), metrics in sorted(grouped.items()):
        summary_row: dict[str, float | str] = {"model": model, "scenario": scenario}
        for metric_name, values in metrics.items():
            values_np = np.asarray(values, dtype=np.float32)
            summary_row[f"{metric_name}_mean"] = float(values_np.mean())
            summary_row[f"{metric_name}_std"] = float(values_np.std())
        summary_rows.append(summary_row)
    return summary_rows


def save_csv(rows: list[dict], path: Path) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def save_plot(summary_rows: list[dict[str, float | str]], path: Path) -> None:
    if not summary_rows:
        return

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    scenarios = sorted({str(row["scenario"]) for row in summary_rows})
    models = sorted({str(row["model"]) for row in summary_rows})
    metrics = AGENT_COMPARE_PLOT_METRICS

    x = np.arange(len(scenarios), dtype=np.float32)
    width = 0.80 / max(len(models), 1)
    color_map = plt.get_cmap("Set2")

    n_cols = 2
    n_rows = int(np.ceil(len(metrics) / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, max(5 * n_rows, 10)))
    axes_flat = np.atleast_1d(axes).flatten()
    for plot_idx, (metric_name, title) in enumerate(metrics):
        ax = axes_flat[plot_idx]
        for model_idx, model in enumerate(models):
            offsets = x - 0.40 + width * 0.5 + model_idx * width
            means = []
            errs = []
            for scenario in scenarios:
                row = next(
                    item
                    for item in summary_rows
                    if item["model"] == model and item["scenario"] == scenario
                )
                means.append(float(row[metric_name]))
                errs.append(float(row[metric_name.replace("_mean", "_std")]))
            ax.bar(
                offsets,
                means,
                width=width,
                label=model,
                color=color_map(model_idx % max(len(models), 1)),
                alpha=0.9,
                yerr=errs,
                capsize=3,
            )
        ax.set_title(title)
        ax.set_xticks(x)
        ax.set_xticklabels(scenarios, rotation=20, ha="right")
        ax.grid(alpha=0.25, axis="y")
        if metric_name == "shortage_rate_mean":
            ax.invert_yaxis()

    for ax in axes_flat[len(metrics) :]:
        ax.axis("off")

    handles, labels = axes_flat[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper center", ncol=max(len(labels), 1))
    fig.suptitle("Policy Comparison Across Shared Scenarios", fontsize=16)
    fig.tight_layout(rect=(0, 0, 1, 0.95))

    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def print_summary(summary_rows: list[dict[str, float | str]]) -> None:
    for scenario in sorted({str(row["scenario"]) for row in summary_rows}):
        print(f"\n=== {scenario} ===")
        scenario_rows = [row for row in summary_rows if row["scenario"] == scenario]
        for row in sorted(scenario_rows, key=lambda item: str(item["model"])):
            print(
                f"{row['model']:>18s} | reward={row['reward_total_mean']:.2f} "
                f"| shortage={row['shortage_rate_mean']:.2f}% "
                f"| demand={row['total_net_requested_mean']:.1f} "
                f"| cost={row['active_action_cost_mean']:.1f} "
                f"| exact={row['exact_match_rate_mean']:.1f}% "
                f"| compat_sub={row['compatible_substitution_rate_mean']:.1f}%"
            )


def main() -> None:
    args = parse_args()
    sim_context = load_training_context()
    scenario_keys = scenario_list_from_args(args)

    mappo_path = Path(args.mappo_model)
    if not mappo_path.exists():
        raise SystemExit(f"MAPPO checkpoint not found: {mappo_path}")
    mappo = MAPPO.load(str(mappo_path))

    G, north, south, east, west = sim_context
    ppo_agent = None
    ppo_path = Path(args.ppo_model)
    if ppo_path.exists():
        ppo_agent = load_ppo_agent(
            str(ppo_path),
            params=prepare_mappo_scenario(copy.deepcopy(SCENARIOS["baseline"])),
            G=G,
            north=north,
            south=south,
            east=east,
            west=west,
            verbose=0,
        )
    else:
        print(f"[compare_agents] PPO checkpoint not found, skipping trained PPO: {ppo_path}")

    rows: list[dict[str, float | int | str]] = []
    for scenario_key in scenario_keys:
        scenario = copy.deepcopy(SCENARIOS[scenario_key])
        for run_idx in range(args.runs):
            seed = args.seed_start + run_idx

            mappo_state = run_mappo_episode(
                mappo,
                scenario,
                sim_context,
                seed=seed,
                prior_scale=args.mappo_prior_scale,
                deterministic=True,
            )
            rows.append(extract_metrics("mappo", scenario_key, seed, mappo_state))

            if ppo_agent is not None:
                ppo_state = run_trained_ppo_episode(
                    ppo_agent,
                    scenario,
                    sim_context,
                    seed=seed,
                    step_hours=args.ppo_step_hours,
                )
                rows.append(extract_metrics("ppo", scenario_key, seed, ppo_state))

            if args.include_heuristic_ppo:
                heuristic_state = run_heuristic_ppo_episode(
                    scenario,
                    sim_context,
                    seed=seed,
                    step_hours=args.ppo_step_hours,
                )
                rows.append(
                    extract_metrics("heuristic_ppo", scenario_key, seed, heuristic_state)
                )

    summary_rows = aggregate_rows(rows)
    prefix = Path(args.output_prefix)
    detail_path = prefix.with_name(f"{prefix.name}_detail.csv")
    summary_path = prefix.with_name(f"{prefix.name}_summary.csv")
    plot_path = prefix.with_name(f"{prefix.name}_comparison.png")

    save_csv(rows, detail_path)
    save_csv(summary_rows, summary_path)
    save_plot(summary_rows, plot_path)
    print_summary(summary_rows)
    print(f"\nSaved detail CSV to {detail_path}")
    print(f"Saved summary CSV to {summary_path}")
    print(f"Saved comparison plot to {plot_path}")


if __name__ == "__main__":
    main()
