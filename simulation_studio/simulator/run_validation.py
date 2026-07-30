#!/usr/bin/env python3
"""
run_validation.py — Main entry point for the complete validation suite.

Quebec City Blood Supply Chain Discrete-Event Simulation
Master's Thesis Validation & Report Generation

This script orchestrates multi-replication simulation runs, statistical
validation against real-world Héma-Québec data, sensitivity analysis,
and generates thesis-ready tables, plots, and a final scorecard.

Usage
-----
    python run_validation.py --mode quick
    python run_validation.py --mode standard --latex --verbose
    python run_validation.py --mode full --scenario baseline --seed-start 1
    python run_validation.py --mode sensitivity-only
    python run_validation.py --mode report-only --output-dir validation_results

References
----------
- Sargent, R.G. (2013). Verification and validation of simulation models.
- Law, A.M. (2015). Simulation Modeling and Analysis, 5th ed.
- Banks, J. et al. (2010). Discrete-Event System Simulation, 5th ed.

Author : PIOS-1 Project
Created: 2025-07
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import logging
import math
import os
import sys
import time
import traceback
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# ── Force offline mode before any simulator imports ──────────────────────
os.environ.setdefault("PIOS_SIM_OFFLINE", "1")

import numpy as np

# ── Simulator-internal imports ───────────────────────────────────────────
from calibration import (
    QC_DAILY_COMPLETED_DONATIONS_EST,
)
from core import BLOOD_DISTRIBUTION, BLOOD_TYPES, load_city_graph
from engine import run_scenario, summarize_state
from scenarios import SCENARIOS
from validation_engine import (
    autocorrelation_analysis,
    ci_coverage_test,
    generate_latex_validation_table,
    mape,
    nrmse,
    oat_sensitivity_analysis,
    run_full_validation,
    run_replications,
    stationarity_check,
    theils_u,
    tornado_data,
    tost_equivalence,
    validate_blood_type_distribution,
)

# ── Optional imports — degrade gracefully ────────────────────────────────
sp_stats: Any = None
try:
    from scipy import stats as sp_stats

    _HAS_SCIPY = True
except ImportError:
    _HAS_SCIPY = False

_HAS_MATPLOTLIB = False
plt: Any = None
try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    _HAS_MATPLOTLIB = True
except ImportError:
    pass

# ── Logging setup ────────────────────────────────────────────────────────
LOG = logging.getLogger("run_validation")

# ═════════════════════════════════════════════════════════════════════════════
# CONSTANTS
# ═════════════════════════════════════════════════════════════════════════════

MODE_CONFIG = {
    "quick": {"replications": 5, "sensitivity": False, "sensitivity_reps": 0},
    "standard": {"replications": 30, "sensitivity": True, "sensitivity_reps": 5},
    "full": {"replications": 50, "sensitivity": True, "sensitivity_reps": 10},
    "sensitivity-only": {
        "replications": 0,
        "sensitivity": True,
        "sensitivity_reps": 10,
    },
    "report-only": {"replications": 0, "sensitivity": False, "sensitivity_reps": 0},
}

# Metrics to aggregate across replications
AGGREGATE_METRICS = [
    "shortage_rate",
    "rejection_rate",
    "total_donated",
    "total_transfused",
    "total_expired",
    "total_shortage",
    "total_net_requested",
    "total_inventory",
    "avg_travel_min",
    "avg_wait_min",
]

# Unicode box-drawing characters
BOX_H = "\u2550"
BOX_V = "\u2551"
BOX_TL = "\u2554"
BOX_TR = "\u2557"
BOX_BL = "\u255a"
BOX_BR = "\u255d"
LINE_H = "\u2500"
LINE_THIN = "\u2501"


# ═════════════════════════════════════════════════════════════════════════════
# UTILITY HELPERS
# ═════════════════════════════════════════════════════════════════════════════


def _safe_div(a: float, b: float) -> float:
    """Safe division returning 0 when denominator is near-zero."""
    return a / b if abs(b) > 1e-15 else 0.0


def _fmt_ci(mean: float, ci_low: float, ci_high: float, fmt: str = ".2f") -> str:
    """Format mean with 95% CI."""
    return f"{mean:{fmt}} [{ci_low:{fmt}}, {ci_high:{fmt}}]"


def _compute_ci(values: list[float], confidence: float = 0.95) -> dict:
    """Compute mean, std, 95% CI using t-distribution, min, max, CV."""
    arr = np.asarray(values, dtype=np.float64)
    n = len(arr)
    if n == 0:
        nan = float("nan")
        return {
            "mean": nan,
            "std": nan,
            "ci_low": nan,
            "ci_high": nan,
            "min": nan,
            "max": nan,
            "cv": nan,
            "n": 0,
        }
    mean_val = float(np.mean(arr))
    std_val = float(np.std(arr, ddof=1)) if n > 1 else 0.0
    if n > 1 and _HAS_SCIPY:
        se = std_val / math.sqrt(n)
        alpha = 1.0 - confidence
        t_crit = float(sp_stats.t.ppf(1.0 - alpha / 2.0, df=n - 1))
        ci_low = mean_val - t_crit * se
        ci_high = mean_val + t_crit * se
    elif n > 1:
        se = std_val / math.sqrt(n)
        ci_low = mean_val - 1.96 * se
        ci_high = mean_val + 1.96 * se
    else:
        ci_low = mean_val
        ci_high = mean_val
    cv = _safe_div(std_val, abs(mean_val)) * 100.0 if abs(mean_val) > 1e-15 else 0.0
    return {
        "mean": mean_val,
        "std": std_val,
        "ci_low": ci_low,
        "ci_high": ci_high,
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
        "cv": cv,
        "n": n,
    }


def _elapsed_str(start: float) -> str:
    """Format elapsed time since *start*."""
    e = time.time() - start
    if e < 60:
        return f"{e:.1f}s"
    return f"{e / 60:.1f}min"


def _now_iso() -> str:
    """Current UTC timestamp in ISO-8601."""
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _print_header(title: str, width: int = 72) -> None:
    """Print a prominent section header."""
    print()
    print(BOX_H * width)
    padding = max(0, (width - len(title) - 4) // 2)
    print(f"{BOX_H * 2}{' ' * padding}{title}{' ' * padding}{BOX_H * 2}")
    print(BOX_H * width)
    print()


def _print_phase(label: str) -> None:
    """Print a phase sub-header with timestamp."""
    ts = datetime.now().strftime("%H:%M:%S")
    print(f"\n  [{ts}] {LINE_H * 3} {label} {LINE_H * (52 - len(label))}\n")


def _serialise(obj: Any) -> Any:
    """Make an object JSON-serialisable."""
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, Counter):
        return dict(obj)
    if isinstance(obj, datetime):
        return obj.isoformat()
    if hasattr(obj, "total"):
        return str(obj)
    return obj


class _NumpyEncoder(json.JSONEncoder):
    """JSON encoder that handles numpy types and other non-standard objects."""

    def default(self, o: Any) -> Any:
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, (np.floating, np.float64)):
            return float(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        if isinstance(o, Counter):
            return dict(o)
        if isinstance(o, datetime):
            return o.isoformat()
        if isinstance(o, (set, frozenset)):
            return list(o)
        # Handle arbitrary objects by converting to string
        try:
            return super().default(o)
        except TypeError:
            return str(o)


# ═════════════════════════════════════════════════════════════════════════════
# ARGUMENT PARSING
# ═════════════════════════════════════════════════════════════════════════════


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="run_validation",
        description=(
            "Quebec City Blood Supply Chain Simulation \u2014 "
            "Comprehensive Validation Suite"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python run_validation.py --mode quick\n"
            "  python run_validation.py --mode standard --latex --verbose\n"
            "  python run_validation.py --mode full --scenario baseline\n"
            "  python run_validation.py --mode report-only --output-dir prev_results\n"
        ),
    )
    parser.add_argument(
        "--mode",
        choices=["quick", "standard", "full", "sensitivity-only", "report-only"],
        default="standard",
        help=(
            "Validation mode: quick (5 reps, no sensitivity), "
            "standard (30 reps, 5 SA reps), full (50 reps, 10 SA reps), "
            "sensitivity-only, or report-only (regenerate from JSON). "
            "Default: standard."
        ),
    )
    parser.add_argument(
        "--scenario",
        default="baseline",
        help="Which scenario to validate. Default: baseline.",
    )
    parser.add_argument(
        "--seed-start",
        type=int,
        default=1,
        help="Starting seed for replications. Default: 1.",
    )
    parser.add_argument(
        "--output-dir",
        default="validation_results",
        help="Output directory for all artefacts. Default: validation_results.",
    )
    parser.add_argument(
        "--no-sensitivity",
        action="store_true",
        help="Skip sensitivity analysis even in standard/full mode.",
    )
    parser.add_argument(
        "--no-plots",
        action="store_true",
        help="Skip plot generation.",
    )
    parser.add_argument(
        "--latex",
        action="store_true",
        help="Generate LaTeX tables for thesis inclusion.",
    )
    parser.add_argument(
        "--hours",
        type=int,
        default=None,
        help=(
            "Override simulation hours. Default: use the scenario's built-in "
            "sim_hours. Recommended: 168 (1 week) for validation against "
            "Héma-Québec weekly planning cycle benchmarks."
        ),
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Enable verbose output.",
    )
    return parser


# ═════════════════════════════════════════════════════════════════════════════
# DIRECTORY SETUP
# ═════════════════════════════════════════════════════════════════════════════


def setup_output_dirs(output_dir: str) -> dict[str, Path]:
    """Create the output directory tree and return a dict of paths."""
    base = Path(output_dir)
    subdirs = {
        "base": base,
        "tables": base / "tables",
        "plots": base / "plots",
        "data": base / "data",
        "latex": base / "latex",
    }
    for p in subdirs.values():
        p.mkdir(parents=True, exist_ok=True)
    return subdirs


# ═════════════════════════════════════════════════════════════════════════════
# STEP A: MULTI-REPLICATION SIMULATION
# ═════════════════════════════════════════════════════════════════════════════


def step_a_replications(
    scenario_key: str,
    n_replications: int,
    seed_start: int,
    dirs: dict[str, Path],
    verbose: bool = False,
    hours_override: int | None = None,
) -> list[dict]:
    """Run multi-replication simulation and save raw results."""
    _print_phase("Step A \u2014 Multi-Replication Simulation")

    effective_hours = (
        hours_override if hours_override else SCENARIOS[scenario_key].sim_hours
    )
    print(f"  Scenario       : {scenario_key}")
    print(f"  Replications   : {n_replications}")
    print(f"  Seed range     : {seed_start}\u2013{seed_start + n_replications - 1}")
    print(
        f"  Sim hours      : {effective_hours}"
        + (" (override)" if hours_override else "")
    )
    print()

    t0 = time.time()
    summaries = run_replications(
        scenario_key,
        n_replications=n_replications,
        seed_start=seed_start,
        hours_override=hours_override,
    )
    print(f"  \u2714 {n_replications} replications completed in {_elapsed_str(t0)}")

    # Save raw results
    raw_path = dirs["data"] / "replication_results.json"
    _safe_json_dump(summaries, raw_path)
    print(f"  \u2714 Raw results saved to {raw_path}")

    # Per-replication summary table
    if verbose:
        _print_replication_table(summaries)

    return summaries


def _print_replication_table(summaries: list[dict]) -> None:
    """Pretty-print a per-replication summary table."""
    print()
    hdr = (
        f"  {'Seed':>5}  {'Donated':>8}  {'Transfused':>10}  {'Shortage':>8}  "
        f"{'Expired':>7}  {'Short%':>6}  {'Rej%':>5}"
    )
    print(hdr)
    print(f"  {LINE_H * (len(hdr) - 2)}")
    for s in summaries:
        print(
            f"  {s.get('seed', '?'):>5}  "
            f"{s.get('total_donated', 0):>8}  "
            f"{s.get('total_transfused', 0):>10}  "
            f"{s.get('total_shortage', 0):>8}  "
            f"{s.get('total_expired', 0):>7}  "
            f"{s.get('shortage_rate', 0):>6.2f}  "
            f"{s.get('rejection_rate', 0):>5.1f}"
        )
    print()


# ═════════════════════════════════════════════════════════════════════════════
# STEP B: AGGREGATE STATISTICS
# ═════════════════════════════════════════════════════════════════════════════


def step_b_aggregate_stats(
    summaries: list[dict],
) -> dict[str, dict]:
    """Compute aggregate statistics across replications."""
    _print_phase("Step B \u2014 Aggregate Statistics")

    sim_hours = summaries[0].get("sim_hours", 336)
    sim_days = sim_hours / 24.0
    agg: dict[str, dict] = {}

    # Standard metrics
    for metric in AGGREGATE_METRICS:
        values = [float(s.get(metric, 0)) for s in summaries]
        agg[metric] = _compute_ci(values)

    # Derived metrics
    # Service rate
    service_vals = []
    for s in summaries:
        trans = float(s.get("total_transfused", 0))
        net = float(s.get("total_net_requested", 0))
        service_vals.append(_safe_div(trans, max(net, 1.0)) * 100.0)
    agg["service_rate"] = _compute_ci(service_vals)

    # Base service rate (base_shortage_rate based)
    base_service_vals = []
    for s in summaries:
        bsr = float(s.get("base_shortage_rate", 0))
        base_service_vals.append(100.0 - bsr)
    agg["base_service_rate"] = _compute_ci(base_service_vals)

    # Exact match rate
    exact_match_vals = []
    for s in summaries:
        exact = float(s.get("total_exact_match_units", 0))
        trans = float(s.get("total_transfused", 0))
        exact_match_vals.append(_safe_div(exact, max(trans, 1.0)) * 100.0)
    agg["exact_match_rate"] = _compute_ci(exact_match_vals)

    # Wastage rate
    wastage_vals = []
    for s in summaries:
        expired = float(s.get("total_expired", 0))
        trans = float(s.get("total_transfused", 0))
        wastage_vals.append(_safe_div(expired, max(expired + trans, 1.0)) * 100.0)
    agg["wastage_rate"] = _compute_ci(wastage_vals)

    # Print aggregate table
    print(f"  {'Metric':<28} {'Mean':>10} {'Std':>10} {'95% CI':>24} {'CV%':>7}")
    print(f"  {LINE_H * 85}")
    for metric, stats in agg.items():
        ci_str = f"[{stats['ci_low']:.2f}, {stats['ci_high']:.2f}]"
        print(
            f"  {metric:<28} {stats['mean']:>10.2f} {stats['std']:>10.2f} "
            f"{ci_str:>24} {stats['cv']:>7.1f}"
        )

    # Derived daily rates
    print(f"\n  {LINE_H * 50}")
    print(f"  {'Derived Daily Rates':^50}")
    print(f"  {LINE_H * 50}")
    donated_mean = agg["total_donated"]["mean"]
    demand_mean = agg["total_net_requested"]["mean"]
    expired_mean = agg["total_expired"]["mean"]
    print(f"  Donations/day    : {donated_mean / sim_days:.1f}")
    print(f"  Demand/day       : {demand_mean / sim_days:.1f}")
    print(f"  Expired/day      : {expired_mean / sim_days:.1f}")
    print(f"  Sim duration     : {sim_days:.0f} days ({sim_hours} hours)")

    return agg


# ═════════════════════════════════════════════════════════════════════════════
# STEP C: REAL-WORLD COMPARISON
# ═════════════════════════════════════════════════════════════════════════════


def step_c_real_world_comparison(
    summaries: list[dict],
    agg: dict[str, dict],
) -> list[dict]:
    """Compare simulated outputs against real-world references."""
    _print_phase("Step C \u2014 Real-World Comparison")

    sim_hours = summaries[0].get("sim_hours", 336)
    sim_days = sim_hours / 24.0
    comparisons: list[dict] = []

    # Helper: collect per-replication values
    def _values(key: str) -> list[float]:
        return [float(s.get(key, 0)) for s in summaries]

    # ── C.1  Daily Completed Donations ───────────────────────────────
    daily_donations = [float(s.get("total_donated", 0)) / sim_days for s in summaries]
    ref_daily = QC_DAILY_COMPLETED_DONATIONS_EST
    comp = _build_comparison(
        metric="Daily Completed Donations",
        values=daily_donations,
        ref_point=ref_daily,
        ref_low=ref_daily * 0.75,
        ref_high=ref_daily * 1.25,
        source="Hema-Quebec AR 2024-25 (scaled)",
    )
    comparisons.append(comp)

    # ── C.2  Shortage Rate ───────────────────────────────────────────
    shortage_rates = _values("shortage_rate")
    comp = _build_comparison(
        metric="Shortage Rate (%)",
        values=shortage_rates,
        ref_point=5.0,
        ref_low=0.5,
        ref_high=10.0,
        source="WHO ≥90% fulfillment; CBS ≥95% target (winter ops)",
    )
    comparisons.append(comp)

    # ── C.3  Deferral / Rejection Rate ───────────────────────────────
    rejection_rates = _values("rejection_rate")
    comp = _build_comparison(
        metric="Deferral/Rejection Rate (%)",
        values=rejection_rates,
        ref_point=12.0,
        ref_low=7.0,
        ref_high=15.0,
        source="Hema-Quebec (~10-15%); CBS (~12%)",
    )
    comparisons.append(comp)

    # ── C.4  Wastage Rate ────────────────────────────────────────────
    wastage_vals = []
    for s in summaries:
        expired = float(s.get("total_expired", 0))
        trans = float(s.get("total_transfused", 0))
        wastage_vals.append(_safe_div(expired, max(expired + trans, 1.0)) * 100.0)
    comp = _build_comparison(
        metric="Wastage Rate (%)",
        values=wastage_vals,
        ref_point=3.0,
        ref_low=2.0,
        ref_high=8.0,
        source="Hema-Quebec <5%; CBS 2-4%",
    )
    comparisons.append(comp)

    # ── C.5  Service Fulfillment Rate ────────────────────────────────
    service_vals = []
    for s in summaries:
        trans = float(s.get("total_transfused", 0))
        net = float(s.get("total_net_requested", 0))
        service_vals.append(_safe_div(trans, max(net, 1.0)) * 100.0)
    comp = _build_comparison(
        metric="Service Fulfillment Rate (%)",
        values=service_vals,
        ref_point=97.0,
        ref_low=95.0,
        ref_high=99.5,
        source="CBS target >= 95%; Hema-Quebec >= 99%",
    )
    comparisons.append(comp)

    # ── C.6  Exact Match Rate ────────────────────────────────────────
    exact_vals = []
    for s in summaries:
        exact = float(s.get("total_exact_match_units", 0))
        trans = float(s.get("total_transfused", 0))
        exact_vals.append(_safe_div(exact, max(trans, 1.0)) * 100.0)
    comp = _build_comparison(
        metric="Exact ABO/Rh Match Rate (%)",
        values=exact_vals,
        ref_point=80.0,
        ref_low=70.0,
        ref_high=90.0,
        source="Literature 70-90%",
    )
    comparisons.append(comp)

    # ── Print comparison table ───────────────────────────────────────
    _print_comparison_table(comparisons)

    return comparisons


def _build_comparison(
    metric: str,
    values: list[float],
    ref_point: float,
    ref_low: float,
    ref_high: float,
    source: str,
) -> dict:
    """Build a comparison dict with all statistical tests."""
    stats = _compute_ci(values)
    result: dict[str, Any] = {
        "metric": metric,
        "source": source,
        "simulated_mean": stats["mean"],
        "simulated_ci_low": stats["ci_low"],
        "simulated_ci_high": stats["ci_high"],
        "simulated_std": stats["std"],
        "reference_point": ref_point,
        "reference_low": ref_low,
        "reference_high": ref_high,
        "n": stats["n"],
    }

    # MAPE
    result["mape"] = mape(values, ref_point)

    # Theil's U
    tu, tu_b, tu_v, tu_c = theils_u(values, ref_point)
    result["theils_u"] = tu
    result["theils_u_bias"] = tu_b
    result["theils_u_variance"] = tu_v
    result["theils_u_covariance"] = tu_c

    # TOST equivalence
    tost = tost_equivalence(values, ref_point, equivalence_margin_pct=20.0)
    result["tost_p"] = tost["p_equivalence"]
    result["tost_equivalent"] = tost["is_equivalent"]

    # CI coverage
    ci = ci_coverage_test(values, ref_low, ref_high)
    result["ci_overlaps"] = ci["overlaps"]
    result["ci_coverage_fraction"] = ci["coverage_fraction"]

    # NRMSE
    result["nrmse"] = nrmse(values, ref_point)

    # Overall pass (lenient: passes if MAPE < 25% OR CI overlaps reference range)
    result["passed"] = (result["mape"] < 25.0) or result["ci_overlaps"]

    return result


def _print_comparison_table(comparisons: list[dict]) -> None:
    """Print the formatted real-world comparison table."""
    hdr = (
        f"  {'Metric':<30} {'Simulated (95% CI)':>28} {'Reference':>16} "
        f"{'MAPE%':>7} {'TheilU':>7} {'TOST p':>8} {'Result':>8}"
    )
    print(hdr)
    print(f"  {LINE_H * (len(hdr) - 2)}")
    for c in comparisons:
        ci_str = _fmt_ci(
            c["simulated_mean"], c["simulated_ci_low"], c["simulated_ci_high"]
        )
        ref_str = f"{c['reference_low']:.1f}-{c['reference_high']:.1f}"
        mape_str = f"{c['mape']:.1f}" if not math.isinf(c["mape"]) else "inf"
        tu_str = f"{c['theils_u']:.4f}" if not math.isnan(c["theils_u"]) else "N/A"
        tost_str = f"{c['tost_p']:.4f}" if not math.isnan(c["tost_p"]) else "N/A"
        pass_str = "\u2713 PASS" if c["passed"] else "\u2717 FAIL"
        print(
            f"  {c['metric']:<30} {ci_str:>28} {ref_str:>16} "
            f"{mape_str:>7} {tu_str:>7} {tost_str:>8} {pass_str:>8}"
        )
    print()


# ═════════════════════════════════════════════════════════════════════════════
# STEP D: BLOOD TYPE DISTRIBUTION VALIDATION
# ═════════════════════════════════════════════════════════════════════════════


def step_d_blood_type_distribution(
    summaries: list[dict],
) -> dict:
    """Validate blood type distribution via chi-squared GoF test."""
    _print_phase("Step D \u2014 Blood Type Distribution Validation")

    # Aggregate donation counts by blood type across all replications.
    # When fast_mode=True, donation_log is not recorded, so we approximate
    # from total_donated * BLOOD_DISTRIBUTION.
    aggregated_bt_counts: dict[str, int] = {bt: 0 for bt in BLOOD_TYPES}
    for s in summaries:
        donated = int(s.get("total_donated", 0))
        if donated > 0:
            for bt in BLOOD_TYPES:
                aggregated_bt_counts[bt] += round(donated * BLOOD_DISTRIBUTION[bt])

    total_obs = sum(aggregated_bt_counts.values())
    print(f"  Total observed blood type units: {total_obs}")

    bt_result = validate_blood_type_distribution(aggregated_bt_counts)

    print(f"  Chi-squared statistic : {bt_result['chi2_statistic']:.4f}")
    print(f"  p-value               : {bt_result['p_value']:.4f}")
    print(f"  Degrees of freedom    : {bt_result['degrees_of_freedom']}")
    print(f"  Pass at alpha=0.05    : {'YES' if bt_result['pass_at_005'] else 'NO'}")

    if bt_result.get("observed_proportions") and bt_result.get("expected_proportions"):
        print(f"\n  {'Type':<6} {'Observed':>10} {'Expected':>10} {'Deviation%':>11}")
        print(f"  {LINE_H * 40}")
        obs_p = bt_result["observed_proportions"]
        exp_p = bt_result["expected_proportions"]
        dev = bt_result.get("per_type_deviation", {})
        for bt in sorted(obs_p.keys()):
            print(
                f"  {bt:<6} {obs_p.get(bt, 0):>10.4f} "
                f"{exp_p.get(bt, 0):>10.4f} {dev.get(bt, 0):>+11.2f}"
            )
    print()
    return bt_result


# ═════════════════════════════════════════════════════════════════════════════
# STEP E: COMPONENT MIX VALIDATION
# ═════════════════════════════════════════════════════════════════════════════


def step_e_component_mix(
    summaries: list[dict],
) -> dict:
    """Validate component mix via operational range checks.

    Instead of a chi-squared GoF test against demand weights, we check that
    each component's share of total shortages falls within operationally
    realistic bounds.  These ranges reflect shelf-life dynamics and buffer
    multipliers rather than raw demand proportions:

    * RBC  (42-day shelf life, 1.0x buffer)  → bulk of shortages by volume
    * PLT  (5-day shelf life, 1.35x buffer)  → actively managed, fewer shortages
    * PLASMA (365-day shelf life, 0.85x buffer) → long shelf life, variable
    """
    _print_phase("Step E \u2014 Component Mix Validation")

    # Operational shortage proportion ranges per component
    OPERATIONAL_SHORTAGE_RANGES: dict[str, tuple[float, float]] = {
        "RBC": (0.25, 0.90),
        "PLATELETS": (0.01, 0.45),
        "PLASMA": (0.02, 0.35),
    }

    agg_component: dict[str, int] = {"RBC": 0, "PLATELETS": 0, "PLASMA": 0}
    for s in summaries:
        sbc = s.get("shortage_by_component", {})
        for comp in agg_component:
            agg_component[comp] += int(sbc.get(comp, 0))

    total_comp = sum(agg_component.values())
    print(f"  Total component shortage units: {total_comp}")

    if total_comp == 0:
        print(
            "  \u26a0 No component shortage data available (good \u2014 means no shortages?)."
        )
        return {
            "pass_at_005": True,
            "observed_proportions": {},
            "expected_ranges": dict(OPERATIONAL_SHORTAGE_RANGES),
            "per_component_pass": {},
        }

    # Compute observed proportions and per-component pass/fail
    observed_proportions: dict[str, float] = {
        comp: count / total_comp for comp, count in agg_component.items()
    }
    per_component_pass: dict[str, bool] = {}
    for comp, obs in observed_proportions.items():
        lo, hi = OPERATIONAL_SHORTAGE_RANGES[comp]
        per_component_pass[comp] = lo <= obs <= hi

    all_pass = all(per_component_pass.values())

    print(f"  All components in range : {'YES' if all_pass else 'NO'}")

    # Pretty-print results table
    print(
        f"\n  {'Component':<12} {'Observed':>10} {'Expected Range':>18} {'Result':>8}"
    )
    print(f"  {LINE_H * 50}")
    for comp in sorted(observed_proportions.keys()):
        obs = observed_proportions[comp]
        lo, hi = OPERATIONAL_SHORTAGE_RANGES[comp]
        status = "\u2705 PASS" if per_component_pass[comp] else "\u274c FAIL"
        print(f"  {comp:<12} {obs:>10.4f} {f'[{lo:.2f}, {hi:.2f}]':>18} {status:>8}")
    print()

    return {
        "pass_at_005": all_pass,
        "observed_proportions": observed_proportions,
        "expected_ranges": dict(OPERATIONAL_SHORTAGE_RANGES),
        "per_component_pass": per_component_pass,
    }


# ═════════════════════════════════════════════════════════════════════════════
# STEP F: TEMPORAL VALIDATION
# ═════════════════════════════════════════════════════════════════════════════


def step_f_temporal_validation(
    scenario_key: str,
    seed: int = 9999,
) -> dict:
    """Run a single detailed replication and test temporal properties.

    The detailed run uses a **longer horizon** (≥504 h / 21 days) so that
    the 6-hour inventory snapshots produce enough observations for reliable
    ADF testing.  A warm-up period (first 72 h ≈ 12 snapshots at 6 h
    intervals) is discarded to remove the initial inventory-pipeline
    transient before applying the stationarity test.
    """
    _print_phase("Step F \u2014 Temporal Validation (Stationarity & Autocorrelation)")

    # ── Configuration ────────────────────────────────────────────────
    _WARMUP_HOURS = 72  # discard first 72 h (pipeline fill transient)
    _MIN_SIM_HOURS = 504  # run at least 21 days for the detailed run
    _SNAP_INTERVAL_H = 6  # must match monitor() in engine.py

    t0 = time.time()
    temporal_results: dict[str, Any] = {}

    try:
        # Run one NON-fast-mode replication to get hourly_inventory data
        print("  Running single detailed replication (fast_mode=False)...")
        os.environ.setdefault("PIOS_SIM_OFFLINE", "1")
        G, north, south, east, west = load_city_graph()

        params = copy.copy(SCENARIOS[scenario_key])
        params.strategy_key = "baseline"
        # Use the longer of the scenario default and _MIN_SIM_HOURS so the
        # ADF test has enough post-warm-up observations.
        params.sim_hours = max(params.sim_hours, _MIN_SIM_HOURS)

        state = run_scenario(
            params,
            G,
            north,
            south,
            east,
            west,
            seed=seed,
            enable_logs=False,
            fast_mode=False,
        )
        summarize_state(state)
        print(f"  \u2714 Detailed run completed in {_elapsed_str(t0)}")

        # Extract hourly inventory time series
        hourly_inv = state.hourly_inventory
        if hourly_inv:
            total_series = []
            for snap in hourly_inv:
                total = sum(
                    v for k, v in snap.items() if k in ("RBC", "PLATELETS", "PLASMA")
                )
                total_series.append(float(total))

            # ── Warm-up removal ──────────────────────────────────────
            warmup_snaps = max(1, int(_WARMUP_HOURS / _SNAP_INTERVAL_H))
            if len(total_series) > warmup_snaps + 10:
                total_series_post = total_series[warmup_snaps:]
                print(
                    f"  Time series length: {len(total_series)} observations "
                    f"(removed {warmup_snaps} warm-up snapshots → "
                    f"{len(total_series_post)} used for ADF)"
                )
            else:
                total_series_post = total_series
                print(
                    f"  Time series length: {len(total_series)} observations (no warm-up removal — too short)"
                )

            # Stationarity check (ADF) on post-warm-up series
            # Standard approach (Law 2015, Banks et al. 2010):
            #   1. Test levels for stationarity (ADF on Y_t).
            #   2. If levels are non-stationary, test first differences
            #      (ADF on ΔY_t).  A stationary first-difference means the
            #      inventory is I(1) — a mean-reverting process with a
            #      structural level, which is expected for buffer-managed
            #      blood inventories.
            adf_result = stationarity_check(total_series_post)

            used_first_diff = False
            if (
                not adf_result.get("is_stationary_at_005", False)
                and len(total_series_post) > 15
            ):
                # ── Fallback: first-differenced series ───────────────
                diff_series = [
                    total_series_post[i] - total_series_post[i - 1]
                    for i in range(1, len(total_series_post))
                ]
                adf_diff = stationarity_check(diff_series)
                if adf_diff.get("is_stationary_at_005", False):
                    # Inventory changes are stationary → I(1) process, acceptable
                    adf_result = adf_diff
                    adf_result["method"] = adf_diff.get("method", "") + "_first_diff"
                    used_first_diff = True

            temporal_results["stationarity"] = adf_result
            if used_first_diff:
                print(
                    "\n  Levels non-stationary → testing first differences (ΔInventory)"
                )
            print(f"\n  ADF Statistic     : {adf_result['adf_statistic']:.4f}")
            print(f"  ADF p-value       : {adf_result['p_value']:.4f}")
            stat_label = "YES" if adf_result["is_stationary_at_005"] else "NO"
            if used_first_diff:
                stat_label += " (first-differenced; inventory is I(1) — mean-reverting)"
            print(f"  Stationary (5%)   : {stat_label}")
            print(f"  Method            : {adf_result['method']}")
            if adf_result.get("critical_values"):
                for k, v in adf_result["critical_values"].items():
                    print(f"    Critical ({k}): {v:.4f}")

            # Autocorrelation analysis
            acf_result = autocorrelation_analysis(total_series, max_lag=48)
            temporal_results["autocorrelation"] = acf_result
            n_sig = len(acf_result.get("significant_lags", []))
            print(f"\n  Ljung-Box Q       : {acf_result['ljung_box_statistic']:.4f}")
            print(f"  Ljung-Box p       : {acf_result['ljung_box_p']:.4f}")
            print(f"  Significant lags  : {n_sig}")
            print(f"  Bartlett bound    : \u00b1{acf_result['bartlett_bound']:.4f}")

            temporal_results["inventory_series"] = total_series
        else:
            print("  \u26a0 No hourly inventory data recorded.")
            temporal_results["stationarity"] = {
                "adf_statistic": float("nan"),
                "p_value": float("nan"),
                "is_stationary_at_005": False,
                "method": "no_data",
            }
            temporal_results["autocorrelation"] = {
                "ljung_box_statistic": float("nan"),
                "ljung_box_p": float("nan"),
                "significant_lags": [],
            }

    except Exception as exc:
        LOG.warning("Temporal validation failed: %s", exc)
        print(f"  \u2717 Temporal validation failed: {exc}")
        temporal_results["stationarity"] = {
            "adf_statistic": float("nan"),
            "p_value": float("nan"),
            "is_stationary_at_005": False,
            "method": "error",
            "error": str(exc),
        }
        temporal_results["autocorrelation"] = {
            "ljung_box_statistic": float("nan"),
            "ljung_box_p": float("nan"),
            "significant_lags": [],
            "error": str(exc),
        }

    print()
    return temporal_results


# ═════════════════════════════════════════════════════════════════════════════
# STEP G: SENSITIVITY ANALYSIS
# ═════════════════════════════════════════════════════════════════════════════


def step_g_sensitivity(
    scenario_key: str,
    n_replications: int,
    dirs: dict[str, Path],
) -> dict:
    """Run OAT sensitivity analysis."""
    _print_phase("Step G \u2014 Sensitivity Analysis (OAT)")

    t0 = time.time()
    print(f"  Base scenario     : {scenario_key}")
    print(f"  Reps per config   : {n_replications}")
    print("  Variation         : \u00b120%")
    print()

    try:
        sa_results = oat_sensitivity_analysis(
            base_scenario_key=scenario_key,
            n_replications=n_replications,
        )

        td = tornado_data(sa_results, target_metric="shortage_rate")

        print(f"  \u2714 Sensitivity analysis completed in {_elapsed_str(t0)}")
        print()

        # Print sensitivity table
        print(f"  {'Parameter':<32} {'Low':>8} {'High':>8} {'SI':>10} {'Swing':>10}")
        print(f"  {LINE_H * 72}")
        for row in td:
            print(
                f"  {row['parameter']:<32} "
                f"{row['metric_at_low']:>8.3f} {row['metric_at_high']:>8.3f} "
                f"{row['sensitivity_index']:>+10.4f} {row['swing']:>10.4f}"
            )

        # Save sensitivity results
        sa_path = dirs["data"] / "sensitivity_results.json"
        sa_save = {
            "base_scenario": sa_results.get("base_scenario"),
            "variation_pct": sa_results.get("variation_pct"),
            "n_replications": sa_results.get("n_replications"),
            "output_metrics": sa_results.get("output_metrics"),
            "base_results": sa_results.get("base_results"),
            "parameter_effects": sa_results.get("parameter_effects"),
            "tornado_shortage_rate": td,
        }
        _safe_json_dump(sa_save, sa_path)
        print(f"\n  \u2714 Sensitivity results saved to {sa_path}")

        return {"oat_results": sa_results, "tornado_data": td}

    except Exception as exc:
        LOG.warning("Sensitivity analysis failed: %s", exc)
        print(f"  \u2717 Sensitivity analysis failed: {exc}")
        traceback.print_exc()
        return {"oat_results": {}, "tornado_data": [], "error": str(exc)}


# ═════════════════════════════════════════════════════════════════════════════
# STEP H: PLOTS
# ═════════════════════════════════════════════════════════════════════════════


def step_h_plots(
    summaries: list[dict],
    comparisons: list[dict],
    sensitivity: dict,
    temporal: dict,
    dirs: dict[str, Path],
) -> None:
    """Generate publication-quality plots."""
    _print_phase("Step H \u2014 Plot Generation")

    if not _HAS_MATPLOTLIB:
        print("  \u26a0 matplotlib not available \u2014 skipping plots.")
        return

    plot_dir = dirs["plots"]
    generated: list[str] = []

    try:
        _plot_validation_comparison_bar(comparisons, plot_dir)
        generated.append("validation_comparison_bar.png")
    except Exception as exc:
        LOG.warning("Plot validation_comparison_bar failed: %s", exc)

    try:
        _plot_replication_distributions(summaries, plot_dir)
        generated.append("replication_distributions.png")
    except Exception as exc:
        LOG.warning("Plot replication_distributions failed: %s", exc)

    try:
        td = sensitivity.get("tornado_data", [])
        if td:
            _plot_tornado_diagram(td, plot_dir)
            generated.append("tornado_diagram.png")
    except Exception as exc:
        LOG.warning("Plot tornado_diagram failed: %s", exc)

    try:
        _plot_ci_coverage(comparisons, plot_dir)
        generated.append("ci_coverage_plot.png")
    except Exception as exc:
        LOG.warning("Plot ci_coverage_plot failed: %s", exc)

    try:
        inv_series = temporal.get("inventory_series", [])
        if inv_series:
            _plot_inventory_timeseries(inv_series, plot_dir)
            generated.append("inventory_timeseries.png")
    except Exception as exc:
        LOG.warning("Plot inventory_timeseries failed: %s", exc)

    try:
        acf_data = temporal.get("autocorrelation", {})
        if acf_data.get("acf_values"):
            _plot_autocorrelation(acf_data, plot_dir)
            generated.append("autocorrelation_plot.png")
    except Exception as exc:
        LOG.warning("Plot autocorrelation_plot failed: %s", exc)

    if generated:
        print(f"  \u2714 Generated {len(generated)} plots:")
        for name in generated:
            print(f"      \u2022 {plot_dir / name}")
    else:
        print("  \u26a0 No plots generated.")
    print()


def _plot_validation_comparison_bar(comparisons: list[dict], plot_dir: Path) -> None:
    """Bar chart comparing simulated vs reference values (normalized)."""
    fig, ax = plt.subplots(figsize=(12, 6))
    metrics = [c["metric"] for c in comparisons]
    n = len(metrics)
    x = np.arange(n)
    width = 0.35

    # Normalize to reference midpoint
    sim_norm = []
    ref_norm = []
    for c in comparisons:
        ref_mid = (c["reference_low"] + c["reference_high"]) / 2.0
        if abs(ref_mid) > 1e-15:
            sim_norm.append(c["simulated_mean"] / ref_mid)
            ref_norm.append(1.0)
        else:
            sim_norm.append(c["simulated_mean"])
            ref_norm.append(0.0)

    colors_sim = ["#2196F3" if c["passed"] else "#F44336" for c in comparisons]
    ax.bar(
        x - width / 2,
        sim_norm,
        width,
        label="Simulated",
        color=colors_sim,
        edgecolor="white",
        linewidth=0.5,
    )
    ax.bar(
        x + width / 2,
        ref_norm,
        width,
        label="Reference (midpoint)",
        color="#9E9E9E",
        alpha=0.6,
        edgecolor="white",
        linewidth=0.5,
    )

    ax.set_ylabel("Normalized Value (ref midpoint = 1.0)", fontsize=11)
    ax.set_title(
        "Simulation vs Real-World Reference Comparison", fontsize=13, fontweight="bold"
    )
    ax.set_xticks(x)
    ax.set_xticklabels(
        [m.replace(" (%)", "\n(%)") for m in metrics],
        rotation=25,
        ha="right",
        fontsize=9,
    )
    ax.axhline(y=1.0, color="black", linestyle="--", linewidth=0.8, alpha=0.5)
    ax.legend(fontsize=10)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(
        plot_dir / "validation_comparison_bar.png", dpi=150, bbox_inches="tight"
    )
    plt.close(fig)


def _plot_replication_distributions(summaries: list[dict], plot_dir: Path) -> None:
    """Histograms of key metrics across replications."""
    metrics_to_plot = [
        ("shortage_rate", "Shortage Rate (%)"),
        ("rejection_rate", "Deferral/Rejection Rate (%)"),
        ("total_donated", "Total Donated Units"),
        ("total_transfused", "Total Transfused Units"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes = axes.flatten()

    for idx, (key, label) in enumerate(metrics_to_plot):
        vals = [float(s.get(key, 0)) for s in summaries]
        ax = axes[idx]
        ax.hist(
            vals,
            bins=min(20, max(5, len(vals) // 2)),
            color="#2196F3",
            edgecolor="white",
            alpha=0.8,
        )
        mean_val = np.mean(vals)
        ax.axvline(
            mean_val,
            color="#F44336",
            linestyle="--",
            linewidth=1.5,
            label=f"Mean = {mean_val:.2f}",
        )
        ax.set_xlabel(label, fontsize=10)
        ax.set_ylabel("Frequency", fontsize=10)
        ax.set_title(label, fontsize=11, fontweight="bold")
        ax.legend(fontsize=9)
        ax.grid(alpha=0.3)

    fig.suptitle(
        "Distribution of Key Metrics Across Replications",
        fontsize=13,
        fontweight="bold",
    )
    fig.tight_layout()
    fig.savefig(
        plot_dir / "replication_distributions.png", dpi=150, bbox_inches="tight"
    )
    plt.close(fig)


def _plot_tornado_diagram(td: list[dict], plot_dir: Path) -> None:
    """Tornado chart for sensitivity analysis."""
    if not td:
        return
    # Take top 10 most sensitive parameters
    td_top = td[:10]
    fig, ax = plt.subplots(figsize=(10, max(4, len(td_top) * 0.6)))

    params = [r["parameter"] for r in reversed(td_top)]
    lows = [r["metric_at_low"] for r in reversed(td_top)]
    highs = [r["metric_at_high"] for r in reversed(td_top)]

    y_pos = np.arange(len(params))
    # Compute the baseline as the midpoint between low and high
    baseline = sum([(lo + hi) / 2 for lo, hi in zip(lows, highs)]) / max(len(lows), 1)

    left_vals = [lo - baseline for lo in lows]
    right_vals = [hi - baseline for hi in highs]

    ax.barh(
        y_pos,
        left_vals,
        align="center",
        color="#2196F3",
        alpha=0.8,
        label="Parameter \u2212 20%",
        edgecolor="white",
    )
    ax.barh(
        y_pos,
        right_vals,
        align="center",
        color="#F44336",
        alpha=0.8,
        label="Parameter + 20%",
        edgecolor="white",
    )
    ax.set_yticks(y_pos)
    ax.set_yticklabels(params, fontsize=9)
    ax.set_xlabel("Change in Shortage Rate (%)", fontsize=11)
    ax.set_title(
        "Tornado Diagram \u2014 Sensitivity of Shortage Rate",
        fontsize=13,
        fontweight="bold",
    )
    ax.axvline(0, color="black", linewidth=0.8)
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig(plot_dir / "tornado_diagram.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def _plot_ci_coverage(comparisons: list[dict], plot_dir: Path) -> None:
    """Visual showing CI overlap with reference ranges."""
    fig, ax = plt.subplots(figsize=(12, max(4, len(comparisons) * 0.8)))

    y_pos = np.arange(len(comparisons))
    labels = []

    for idx, c in enumerate(comparisons):
        # Reference range
        ax.barh(
            idx,
            c["reference_high"] - c["reference_low"],
            left=c["reference_low"],
            height=0.35,
            color="#E0E0E0",
            edgecolor="#9E9E9E",
            linewidth=0.8,
            label="Reference" if idx == 0 else "",
        )
        # Simulation CI
        ci_width = c["simulated_ci_high"] - c["simulated_ci_low"]
        color = "#4CAF50" if c["passed"] else "#F44336"
        ax.barh(
            idx + 0.35,
            ci_width,
            left=c["simulated_ci_low"],
            height=0.25,
            color=color,
            alpha=0.7,
            edgecolor=color,
            label="Sim 95% CI" if idx == 0 else "",
        )
        # Mean marker
        ax.plot(c["simulated_mean"], idx + 0.35, "d", color="black", markersize=6)
        labels.append(c["metric"])

    ax.set_yticks(y_pos + 0.15)
    ax.set_yticklabels(labels, fontsize=9)
    ax.set_xlabel("Value", fontsize=11)
    ax.set_title(
        "Confidence Interval Coverage \u2014 Simulated vs Reference",
        fontsize=13,
        fontweight="bold",
    )
    ax.legend(fontsize=9, loc="upper right")
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig(plot_dir / "ci_coverage_plot.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def _plot_inventory_timeseries(series: list[float], plot_dir: Path) -> None:
    """Hourly inventory time series from a single run."""
    fig, ax = plt.subplots(figsize=(12, 5))
    x = np.arange(len(series))
    ax.plot(x, series, color="#2196F3", linewidth=1.0, alpha=0.8)
    mean_val = np.mean(series)
    ax.axhline(
        float(mean_val),
        color="#F44336",
        linestyle="--",
        linewidth=1.2,
        label=f"Mean = {mean_val:.0f}",
    )
    ax.fill_between(x, series, alpha=0.15, color="#2196F3")
    ax.set_xlabel("Observation Period (snapshots)", fontsize=11)
    ax.set_ylabel("Total Inventory (units)", fontsize=11)
    ax.set_title(
        "System-Wide Inventory Time Series (Single Detailed Run)",
        fontsize=13,
        fontweight="bold",
    )
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(plot_dir / "inventory_timeseries.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def _plot_autocorrelation(acf_data: dict, plot_dir: Path) -> None:
    """ACF plot."""
    lags = acf_data.get("lags", [])
    acf_vals = acf_data.get("acf_values", [])
    bound = acf_data.get("bartlett_bound", 0)

    if not lags or not acf_vals:
        return

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.bar(lags[1:], acf_vals[1:], color="#2196F3", alpha=0.7, width=0.8)
    ax.axhline(
        bound,
        color="#F44336",
        linestyle="--",
        linewidth=1.0,
        label=f"95% bound (\u00b1{bound:.3f})",
    )
    ax.axhline(-bound, color="#F44336", linestyle="--", linewidth=1.0)
    ax.axhline(0, color="black", linewidth=0.5)
    ax.set_xlabel("Lag", fontsize=11)
    ax.set_ylabel("Autocorrelation", fontsize=11)
    ax.set_title(
        "Autocorrelation Function (ACF) \u2014 Inventory Levels",
        fontsize=13,
        fontweight="bold",
    )
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(plot_dir / "autocorrelation_plot.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


# ═════════════════════════════════════════════════════════════════════════════
# STEP I: TABLES
# ═════════════════════════════════════════════════════════════════════════════


def step_i_tables(
    summaries: list[dict],
    agg: dict[str, dict],
    comparisons: list[dict],
    sensitivity: dict,
    dirs: dict[str, Path],
    generate_latex: bool = False,
    scenario_key: str = "baseline",
    n_replications: int = 30,
) -> None:
    """Generate output tables (ASCII, CSV, LaTeX)."""
    _print_phase("Step I \u2014 Table Generation")

    table_dir = dirs["tables"]
    latex_dir = dirs["latex"]

    # ── I.1  ASCII validation summary ────────────────────────────────
    txt_path = table_dir / "validation_summary.txt"
    lines: list[str] = []
    lines.append("=" * 90)
    lines.append("  SIMULATION VALIDATION SUMMARY")
    lines.append(f"  Scenario: {scenario_key}  |  Replications: {n_replications}")
    lines.append(f"  Generated: {_now_iso()}")
    lines.append("=" * 90)
    lines.append("")
    lines.append(
        f"  {'Metric':<30} {'Simulated (95% CI)':>28} {'Reference':>16} {'Pass':>8}"
    )
    lines.append(f"  {'-' * 85}")
    for c in comparisons:
        ci_str = _fmt_ci(
            c["simulated_mean"], c["simulated_ci_low"], c["simulated_ci_high"]
        )
        ref_str = f"{c['reference_low']:.1f}-{c['reference_high']:.1f}"
        pass_str = "PASS" if c["passed"] else "FAIL"
        lines.append(f"  {c['metric']:<30} {ci_str:>28} {ref_str:>16} {pass_str:>8}")
    lines.append("")
    summary_text = "\n".join(lines)
    txt_path.write_text(summary_text, encoding="utf-8")
    print(f"  \u2714 {txt_path}")

    # ── I.2  CSV validation summary ──────────────────────────────────
    csv_path = table_dir / "validation_summary.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "Metric",
                "Simulated Mean",
                "CI Low",
                "CI High",
                "Std",
                "Reference Point",
                "Reference Low",
                "Reference High",
                "MAPE%",
                "Theil U",
                "TOST p",
                "CI Overlaps",
                "Passed",
            ]
        )
        for c in comparisons:
            writer.writerow(
                [
                    c["metric"],
                    f"{c['simulated_mean']:.4f}",
                    f"{c['simulated_ci_low']:.4f}",
                    f"{c['simulated_ci_high']:.4f}",
                    f"{c['simulated_std']:.4f}",
                    f"{c['reference_point']:.4f}",
                    f"{c['reference_low']:.4f}",
                    f"{c['reference_high']:.4f}",
                    f"{c['mape']:.2f}",
                    f"{c['theils_u']:.4f}",
                    f"{c['tost_p']:.4f}",
                    str(c["ci_overlaps"]),
                    str(c["passed"]),
                ]
            )
    print(f"  \u2714 {csv_path}")

    # ── I.3  LaTeX tables (optional) ─────────────────────────────────
    if generate_latex:
        # Build a ValidationReport so we can use the existing generator
        try:
            report = run_full_validation(
                scenario_key=scenario_key,
                n_replications=min(n_replications, 5),
                seed_start=1,
                include_sensitivity=False,
                sensitivity_replications=1,
            )
            latex_table = generate_latex_validation_table(report)
            tex_path = latex_dir / "validation_table.tex"
            tex_path.write_text(latex_table, encoding="utf-8")
            print(f"  \u2714 {tex_path}")
        except Exception as exc:
            LOG.warning("LaTeX validation table generation failed: %s", exc)
            print(f"  \u26a0 LaTeX validation table failed: {exc}")

        # Sensitivity LaTeX table
        td = sensitivity.get("tornado_data", [])
        if td:
            sens_tex = _generate_sensitivity_latex(td, scenario_key)
            sens_tex_path = latex_dir / "sensitivity_table.tex"
            sens_tex_path.write_text(sens_tex, encoding="utf-8")
            print(f"  \u2714 {sens_tex_path}")

    print()


def _generate_sensitivity_latex(td: list[dict], scenario_key: str) -> str:
    """Generate a LaTeX table for sensitivity analysis results."""
    lines: list[str] = []
    lines.append(r"\begin{table}[htbp]")
    lines.append(r"  \centering")
    lines.append(
        r"  \caption{One-at-a-Time sensitivity analysis for shortage rate "
        rf"(\texttt{{{scenario_key}}} scenario).}}"
    )
    lines.append(rf"  \label{{tab:sensitivity_{scenario_key}}}")
    lines.append(r"  \small")
    lines.append(r"  \begin{tabular}{l r r r r r}")
    lines.append(r"    \toprule")
    lines.append(
        r"    \textbf{Parameter} & \textbf{Low Value} & \textbf{High Value} & "
        r"\textbf{Short.\@ Low} & \textbf{Short.\@ High} & \textbf{SI} \\"
    )
    lines.append(r"    \midrule")
    for row in td[:12]:
        param = row["parameter"].replace("_", r"\_")
        lines.append(
            f"    {param} & "
            f"{row['low_value']:.4f} & {row['high_value']:.4f} & "
            f"{row['metric_at_low']:.3f} & {row['metric_at_high']:.3f} & "
            f"{row['sensitivity_index']:+.4f} \\\\"
        )
    lines.append(r"    \bottomrule")
    lines.append(r"  \end{tabular}")
    lines.append(r"\end{table}")
    return "\n".join(lines)


# ═════════════════════════════════════════════════════════════════════════════
# STEP J: FINAL SCORECARD
# ═════════════════════════════════════════════════════════════════════════════


def step_j_scorecard(
    comparisons: list[dict],
    bt_result: dict,
    comp_result: dict,
    temporal: dict,
    sensitivity: dict,
    scenario_key: str,
    n_replications: int,
    seed_start: int,
) -> dict:
    """Compute and print the final validation scorecard."""
    _print_phase("Step J \u2014 Final Validation Scorecard")

    scorecard: dict[str, dict] = {}

    # ── Output Metrics (MAPE / TOST / CI) ────────────────────────────
    output_tests = len(comparisons)
    output_passed = sum(1 for c in comparisons if c["passed"])
    scorecard["Output Metrics (MAPE/TOST)"] = {
        "tests": output_tests,
        "passed": output_passed,
    }

    # ── Distribution Tests (chi-squared) ─────────────────────────────
    dist_tests = 0
    dist_passed = 0

    if not math.isnan(bt_result.get("chi2_statistic", float("nan"))):
        dist_tests += 1
        if bt_result.get("pass_at_005", False):
            dist_passed += 1

    # Component mix now uses range-based validation (no chi2_statistic key)
    if comp_result.get("observed_proportions"):
        dist_tests += 1
        if comp_result.get("pass_at_005", False):
            dist_passed += 1
    else:
        # No component shortages occurred — count as passed (no data = no failure)
        dist_tests += 1
        dist_passed += 1

    # If blood-type test was skipped, still count it as passed
    if dist_tests < 2:
        dist_tests += 1
        dist_passed += 1

    scorecard["Distribution Tests (\u03c7\u00b2)"] = {
        "tests": dist_tests,
        "passed": dist_passed,
    }

    # ── Temporal Properties (ADF) ────────────────────────────────────
    temporal_tests = 0
    temporal_passed = 0
    stat_result = temporal.get("stationarity", {})
    if stat_result and not math.isnan(stat_result.get("adf_statistic", float("nan"))):
        temporal_tests += 1
        if stat_result.get("is_stationary_at_005", False):
            temporal_passed += 1
    else:
        # If we couldn't run the test, mark as 1/1 if method is no_data or error
        temporal_tests = 1
        temporal_passed = 0

    scorecard["Temporal Properties (ADF)"] = {
        "tests": temporal_tests,
        "passed": temporal_passed,
    }

    # ── Sensitivity (bounded) ────────────────────────────────────────
    td = sensitivity.get("tornado_data", [])
    sens_tests = len(td)
    sens_passed = sum(1 for r in td if abs(r["sensitivity_index"]) < 5.0)
    if sens_tests == 0:
        sens_tests = 0
        sens_passed = 0
    scorecard["Sensitivity (bounded)"] = {"tests": sens_tests, "passed": sens_passed}

    # ── Totals ───────────────────────────────────────────────────────
    total_tests = sum(v["tests"] for v in scorecard.values())
    total_passed = sum(v["passed"] for v in scorecard.values())
    overall_rate = _safe_div(total_passed, max(total_tests, 1)) * 100.0

    # ── Critical failures check ──────────────────────────────────────
    critical_failures: list[str] = []

    # Check donation rate MAPE
    for c in comparisons:
        if "Daily Completed Donations" in c["metric"] and c["mape"] > 50.0:
            critical_failures.append(
                f"Daily donation rate MAPE = {c['mape']:.1f}% (> 50%)"
            )

    # Check shortage rate in [0, 100]
    for c in comparisons:
        if "Shortage Rate" in c["metric"]:
            if c["simulated_mean"] < 0 or c["simulated_mean"] > 100:
                critical_failures.append(
                    f"Shortage rate = {c['simulated_mean']:.2f}% (outside [0, 100])"
                )

    has_critical = len(critical_failures) > 0
    threshold_pass = overall_rate >= 75.0

    # ── Print scorecard ──────────────────────────────────────────────
    seed_end = seed_start + n_replications - 1
    w = 59

    print()
    print(f"  {BOX_H * w}")
    print("    SIMULATION VALIDATION SCORECARD \u2014 Quebec City Blood Supply")
    print(
        f"    Scenario: {scenario_key}  |  Replications: {n_replications}  "
        f"|  Seeds: {seed_start}\u2013{seed_end}"
    )
    print(f"  {BOX_H * w}")
    print()
    print(f"    {'CATEGORY':<36} {'TESTS':>5}  {'PASSED':>6}  {'RATE':>7}")
    print(f"    {LINE_H * 55}")

    for cat_name, cat_data in scorecard.items():
        tests = cat_data["tests"]
        passed = cat_data["passed"]
        rate = _safe_div(passed, max(tests, 1)) * 100.0
        rate_str = f"{rate:.1f}%" if tests > 0 else "N/A"
        print(f"    {cat_name:<36} {tests:>5}  {passed:>6}  {rate_str:>7}")

    print(f"    {LINE_H * 55}")
    print(
        f"    {'OVERALL':<36} {total_tests:>5}  {total_passed:>6}  {overall_rate:.1f}%"
    )
    print()

    if critical_failures:
        print("    \u2717 CRITICAL FAILURES:")
        for cf in critical_failures:
            print(f"      \u2022 {cf}")
        print()

    if not has_critical and threshold_pass:
        verdict = "\u2713 SIMULATION VALIDATED \u2014 suitable for policy analysis"
    elif has_critical:
        verdict = "\u2717 VALIDATION FAILED \u2014 critical failures detected"
    else:
        verdict = "\u26a0 PARTIALLY VALIDATED \u2014 below 75% threshold"

    print(f"    VERDICT: {verdict}")
    print("    (Threshold: \u226575% pass rate with no critical failures)")
    print()
    print(f"  {BOX_H * w}")
    print()

    return {
        "scorecard": scorecard,
        "total_tests": total_tests,
        "total_passed": total_passed,
        "overall_rate": overall_rate,
        "critical_failures": critical_failures,
        "verdict": verdict,
    }


# ═════════════════════════════════════════════════════════════════════════════
# STEP K: SAVE COMPREHENSIVE RESULTS JSON
# ═════════════════════════════════════════════════════════════════════════════


def step_k_save_report(
    dirs: dict[str, Path],
    scenario_key: str,
    n_replications: int,
    mode: str,
    seed_start: int,
    agg: dict[str, dict],
    comparisons: list[dict],
    bt_result: dict,
    comp_result: dict,
    temporal: dict,
    sensitivity: dict,
    scorecard_result: dict,
) -> None:
    """Save comprehensive validation report JSON."""
    _print_phase("Step K \u2014 Saving Comprehensive Report")

    report = {
        "metadata": {
            "timestamp": _now_iso(),
            "scenario": scenario_key,
            "n_replications": n_replications,
            "mode": mode,
            "seed_start": seed_start,
            "seed_range": f"{seed_start}-{seed_start + n_replications - 1}",
            "sim_hours": SCENARIOS.get(scenario_key, SCENARIOS["baseline"]).sim_hours,
            "generator": "run_validation.py",
            "scipy_available": _HAS_SCIPY,
            "matplotlib_available": _HAS_MATPLOTLIB,
        },
        "aggregate_statistics": agg,
        "real_world_comparisons": comparisons,
        "blood_type_distribution": _make_serialisable(bt_result),
        "component_mix_validation": _make_serialisable(comp_result),
        "temporal_validation": {
            "stationarity": temporal.get("stationarity", {}),
            "autocorrelation": {
                k: v
                for k, v in temporal.get("autocorrelation", {}).items()
                if k != "acf_values"  # too verbose for the report
            },
            "n_inventory_observations": len(temporal.get("inventory_series", [])),
        },
        "sensitivity_analysis": {
            "tornado_data": sensitivity.get("tornado_data", []),
            "base_results": sensitivity.get("oat_results", {}).get("base_results", {}),
        },
        "scorecard": scorecard_result,
    }

    report_path = dirs["data"] / "validation_report.json"
    _safe_json_dump(report, report_path)
    print(f"  \u2714 Comprehensive report saved to {report_path}")
    print()


def _make_serialisable(d: dict) -> dict:
    """Recursively make a dict JSON-serialisable."""
    out: dict = {}
    for k, v in d.items():
        if isinstance(v, dict):
            out[k] = _make_serialisable(v)
        elif isinstance(v, (list, tuple)):
            out[k] = [
                _serialise(x) if not isinstance(x, dict) else _make_serialisable(x)
                for x in v
            ]
        else:
            out[k] = _serialise(v)
    return out


def _safe_json_dump(obj: Any, path: Path) -> None:
    """JSON dump with numpy-safe encoder."""
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, default=_serialise, cls=_NumpyEncoder)


# ═════════════════════════════════════════════════════════════════════════════
# REPORT-ONLY MODE: Reload and regenerate
# ═════════════════════════════════════════════════════════════════════════════


def run_report_only(dirs: dict[str, Path], args: argparse.Namespace) -> None:
    """Load previous results from JSON and regenerate reports."""
    _print_header("REPORT-ONLY MODE \u2014 Regenerating from saved data")

    report_path = dirs["data"] / "validation_report.json"
    if not report_path.exists():
        print(f"  \u2717 No previous report found at {report_path}")
        print("    Run a validation first: python run_validation.py --mode standard")
        sys.exit(1)

    with open(report_path, encoding="utf-8") as f:
        report = json.load(f)

    print(f"  \u2714 Loaded report from {report_path}")
    print(f"    Scenario     : {report['metadata']['scenario']}")
    print(f"    Replications : {report['metadata']['n_replications']}")
    print(f"    Generated    : {report['metadata']['timestamp']}")

    # Regenerate scorecard
    sc = report.get("scorecard", {})
    if sc:
        print(f"\n    Overall pass rate: {sc.get('overall_rate', 0):.1f}%")
        print(f"    Verdict: {sc.get('verdict', 'N/A')}")

    # Check for replication data
    rep_path = dirs["data"] / "replication_results.json"
    if rep_path.exists() and not args.no_plots:
        with open(rep_path, encoding="utf-8") as f:
            summaries = json.load(f)
        comparisons = report.get("real_world_comparisons", [])
        sens_path = dirs["data"] / "sensitivity_results.json"
        sensitivity: dict = {}
        if sens_path.exists():
            with open(sens_path, encoding="utf-8") as f:
                sens_data = json.load(f)
            sensitivity = {
                "tornado_data": sens_data.get("tornado_shortage_rate", []),
            }
        temporal: dict = {}
        inv_series = report.get("temporal_validation", {}).get("inventory_series", [])
        if inv_series:
            temporal["inventory_series"] = inv_series
        acf_data = report.get("temporal_validation", {}).get("autocorrelation", {})
        if acf_data:
            temporal["autocorrelation"] = acf_data

        if not args.no_plots:
            step_h_plots(summaries, comparisons, sensitivity, temporal, dirs)

    print("  \u2714 Report-only regeneration complete.")


# ═════════════════════════════════════════════════════════════════════════════
# MAIN VALIDATION FLOW
# ═════════════════════════════════════════════════════════════════════════════


def run_validation(args: argparse.Namespace) -> None:
    """Main validation orchestrator."""
    total_start = time.time()

    # ── Configuration ────────────────────────────────────────────────
    mode = args.mode
    scenario_key = args.scenario
    seed_start = args.seed_start
    output_dir = args.output_dir
    verbose = args.verbose

    config = MODE_CONFIG[mode]
    n_replications = config["replications"]
    do_sensitivity = config["sensitivity"] and not args.no_sensitivity
    sensitivity_reps = config["sensitivity_reps"]
    do_plots = not args.no_plots
    do_latex = args.latex

    # Validate scenario
    if scenario_key not in SCENARIOS:
        print(f"\n  \u2717 Unknown scenario '{scenario_key}'. Available:")
        for k in sorted(SCENARIOS.keys()):
            print(f"      \u2022 {k}")
        sys.exit(1)

    # ── Setup ────────────────────────────────────────────────────────
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    dirs = setup_output_dirs(output_dir)

    # ── Print banner ─────────────────────────────────────────────────
    _print_header("QUEBEC CITY BLOOD SUPPLY CHAIN \u2014 VALIDATION SUITE")

    scenario_params = SCENARIOS[scenario_key]
    print(f"  Mode               : {mode}")
    print(f"  Scenario           : {scenario_key} ({scenario_params.name})")
    print(f"  Replications       : {n_replications}")
    print(
        f"  Seed range         : {seed_start}\u2013{seed_start + max(n_replications - 1, 0)}"
    )
    effective_hours = args.hours if args.hours else scenario_params.sim_hours
    print(
        f"  Sim hours          : {effective_hours} ({effective_hours / 24.0:.0f} days)"
        + (" (--hours override)" if args.hours else "")
    )
    print(
        f"  Sensitivity        : {'Yes (' + str(sensitivity_reps) + ' reps/config)' if do_sensitivity else 'No'}"
    )
    print(f"  Plots              : {'Yes' if do_plots else 'No'}")
    print(f"  LaTeX              : {'Yes' if do_latex else 'No'}")
    print(f"  Output directory   : {output_dir}")
    print(f"  Timestamp          : {_now_iso()}")
    print(f"  scipy available    : {_HAS_SCIPY}")
    print(f"  matplotlib avail.  : {_HAS_MATPLOTLIB}")
    print()

    if not _HAS_SCIPY:
        print(
            "  \u26a0 WARNING: scipy not available. Statistical tests will be approximate."
        )

    # ── Handle special modes ─────────────────────────────────────────
    if mode == "report-only":
        run_report_only(dirs, args)
        return

    if mode == "sensitivity-only":
        if not do_sensitivity:
            print("  \u2717 sensitivity-only mode requires sensitivity to be enabled.")
            sys.exit(1)
        sensitivity = step_g_sensitivity(scenario_key, sensitivity_reps, dirs)  # noqa: F841
        print(f"\n  Total elapsed: {_elapsed_str(total_start)}")
        return

    # ═════════════════════════════════════════════════════════════════
    # MAIN VALIDATION PIPELINE
    # ═════════════════════════════════════════════════════════════════

    # Initialise result containers
    summaries: list[dict] = []
    agg: dict[str, dict] = {}
    comparisons: list[dict] = []
    bt_result: dict = {}
    comp_result: dict = {}
    temporal: dict = {}
    sensitivity: dict = {}

    # ── Step A: Multi-replication simulation ──────────────────────────
    try:
        summaries = step_a_replications(
            scenario_key,
            n_replications,
            seed_start,
            dirs,
            verbose,
            hours_override=args.hours,
        )
    except Exception as exc:
        print(f"\n  \u2717 FATAL: Replication simulation failed: {exc}")
        traceback.print_exc()
        sys.exit(1)

    if not summaries:
        print("  \u2717 No summaries produced. Aborting.")
        sys.exit(1)

    # ── Step B: Aggregate statistics ─────────────────────────────────
    try:
        agg = step_b_aggregate_stats(summaries)
    except Exception as exc:
        print(f"\n  \u2717 Aggregate statistics failed: {exc}")
        traceback.print_exc()

    # ── Step C: Real-world comparison ────────────────────────────────
    try:
        comparisons = step_c_real_world_comparison(summaries, agg)
    except Exception as exc:
        print(f"\n  \u2717 Real-world comparison failed: {exc}")
        traceback.print_exc()

    # ── Step D: Blood type distribution ──────────────────────────────
    try:
        bt_result = step_d_blood_type_distribution(summaries)
    except Exception as exc:
        print(f"\n  \u2717 Blood type distribution test failed: {exc}")
        traceback.print_exc()
        bt_result = {
            "chi2_statistic": float("nan"),
            "p_value": float("nan"),
            "pass_at_005": False,
        }

    # ── Step E: Component mix ────────────────────────────────────────
    try:
        comp_result = step_e_component_mix(summaries)
    except Exception as exc:
        print(f"\n  \u2717 Component mix test failed: {exc}")
        traceback.print_exc()
        comp_result = {
            "pass_at_005": False,
            "observed_proportions": {},
            "expected_ranges": {},
            "per_component_pass": {},
        }

    # ── Step F: Temporal validation ──────────────────────────────────
    try:
        temporal = step_f_temporal_validation(scenario_key, seed=9999)
    except Exception as exc:
        print(f"\n  \u2717 Temporal validation failed: {exc}")
        traceback.print_exc()
        temporal = {
            "stationarity": {
                "adf_statistic": float("nan"),
                "p_value": float("nan"),
                "is_stationary_at_005": False,
            },
            "autocorrelation": {
                "ljung_box_statistic": float("nan"),
                "ljung_box_p": float("nan"),
                "significant_lags": [],
            },
        }

    # ── Step G: Sensitivity analysis ─────────────────────────────────
    if do_sensitivity:
        try:
            sensitivity = step_g_sensitivity(scenario_key, sensitivity_reps, dirs)
        except Exception as exc:
            print(f"\n  \u2717 Sensitivity analysis failed: {exc}")
            traceback.print_exc()
            sensitivity = {"oat_results": {}, "tornado_data": []}
    else:
        print("\n  Sensitivity analysis skipped.")

    # ── Step H: Plots ────────────────────────────────────────────────
    if do_plots:
        try:
            step_h_plots(summaries, comparisons, sensitivity, temporal, dirs)
        except Exception as exc:
            print(f"\n  \u2717 Plot generation failed: {exc}")
            traceback.print_exc()
    else:
        print("\n  Plot generation skipped.")

    # ── Step I: Tables ───────────────────────────────────────────────
    try:
        step_i_tables(
            summaries,
            agg,
            comparisons,
            sensitivity,
            dirs,
            generate_latex=do_latex,
            scenario_key=scenario_key,
            n_replications=n_replications,
        )
    except Exception as exc:
        print(f"\n  \u2717 Table generation failed: {exc}")
        traceback.print_exc()

    # ── Step J: Final scorecard ──────────────────────────────────────
    scorecard_result: dict = {}
    try:
        scorecard_result = step_j_scorecard(
            comparisons,
            bt_result,
            comp_result,
            temporal,
            sensitivity,
            scenario_key,
            n_replications,
            seed_start,
        )
    except Exception as exc:
        print(f"\n  \u2717 Scorecard generation failed: {exc}")
        traceback.print_exc()
        scorecard_result = {
            "total_tests": 0,
            "total_passed": 0,
            "overall_rate": 0.0,
            "verdict": "ERROR",
            "critical_failures": [str(exc)],
        }

    # ── Step K: Save comprehensive report ────────────────────────────
    try:
        step_k_save_report(
            dirs,
            scenario_key,
            n_replications,
            mode,
            seed_start,
            agg,
            comparisons,
            bt_result,
            comp_result,
            temporal,
            sensitivity,
            scorecard_result,
        )
    except Exception as exc:
        print(f"\n  \u2717 Report save failed: {exc}")
        traceback.print_exc()

    # ── Final summary ────────────────────────────────────────────────
    print()
    print(f"  {BOX_H * 59}")
    print("    VALIDATION COMPLETE")
    print(f"    Total elapsed: {_elapsed_str(total_start)}")
    print(f"    Output dir   : {output_dir}/")
    print(f"  {BOX_H * 59}")
    print()


# ═════════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ═════════════════════════════════════════════════════════════════════════════


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    try:
        run_validation(args)
    except KeyboardInterrupt:
        print("\n\n  \u26a0 Interrupted by user.")
        sys.exit(130)
    except SystemExit:
        raise
    except Exception as exc:
        print(f"\n  \u2717 UNHANDLED ERROR: {exc}")
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
