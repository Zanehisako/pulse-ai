"""Regenerate the strategy x scenario heatmap grid from an existing detail CSV.

The full ``evaluate_strategies.py`` run loads every agent and re-simulates ~1500
episodes (~11 min) before it plots. When only the *figure* needs to change (label
fixes, panel swaps, layout), that re-run is wasteful: the per-seed metrics are
already on disk in ``<prefix>_detail.csv``. This script re-aggregates that CSV and
calls the same ``save_heatmap_grid`` used by the canonical run, so the published
figure is reproduced in seconds with identical styling.

Usage (from simulation_studio/simulator/):
    python replot_heatmaps.py \
        --detail results/strategy_eval_thesis_detail.csv \
        --output-prefix results/strategy_eval_thesis --thesis-ready
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

from evaluate_strategies import (
    PlotExportConfig,
    aggregate_rows,
    save_heatmap_grid,
)
from scenarios import SCENARIOS

# Canonical baseline-ladder order used in the paper (static -> model-free ->
# offline -> model-based). Strategies present in the CSV but not listed here are
# appended in first-seen order so the script never silently drops a row.
LADDER_ORDER = [
    "baseline",
    "mass_campaign",
    "lab_investment",
    "emergency_network",
    "full_response",
    "ppo_shortage_minimizer",
    "ppo_continuous",
    "sac_continuous",
    "iql_offline",
    "cql_offline",
    "dreamerv3_official",
]


def read_detail(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def order_present(present: set[str], preferred: list[str], seen: list[str]) -> list[str]:
    ordered = [key for key in preferred if key in present]
    ordered.extend(key for key in seen if key in present and key not in ordered)
    return ordered


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--detail",
        type=Path,
        default=Path("results/strategy_eval_thesis_detail.csv"),
        help="Per-seed detail CSV produced by evaluate_strategies.py.",
    )
    parser.add_argument(
        "--output-prefix",
        type=str,
        default="results/strategy_eval_thesis",
        help="Output prefix; the figure is written to <prefix>_heatmaps.<ext>.",
    )
    parser.add_argument("--thesis-ready", action="store_true", help="Use the paper styling/formats.")
    args = parser.parse_args()

    if not args.detail.exists():
        raise SystemExit(f"Detail CSV not found: {args.detail}")

    rows = read_detail(args.detail)
    if not rows:
        raise SystemExit(f"No rows in {args.detail}")

    summary_rows = aggregate_rows(rows)

    present_strategies = {str(row["strategy"]) for row in rows}
    present_scenarios = {str(row["scenario"]) for row in rows}
    seen_strategies = list(dict.fromkeys(str(row["strategy"]) for row in rows))
    seen_scenarios = list(dict.fromkeys(str(row["scenario"]) for row in rows))

    strategy_keys = order_present(present_strategies, LADDER_ORDER, seen_strategies)
    # Scenario axis follows SCENARIOS definition order (training then holdout).
    scenario_keys = order_present(present_scenarios, list(SCENARIOS.keys()), seen_scenarios)

    if args.thesis_ready:
        config = PlotExportConfig(thesis_ready=True, formats=("png", "pdf", "svg"), dpi=300)
    else:
        config = PlotExportConfig()

    output_path = Path(f"{args.output_prefix}_heatmaps.png")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    save_heatmap_grid(summary_rows, scenario_keys, strategy_keys, output_path, plot_config=config)
    print(f"Saved heatmap grid to {output_path} ({len(strategy_keys)} strategies x {len(scenario_keys)} scenarios)")


if __name__ == "__main__":
    main()
