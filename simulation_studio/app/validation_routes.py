"""
validation_routes.py — FastAPI routes for running simulation validation from
the Simulation Studio web UI.

Exposes endpoints for:
    • Listing real-world reference data and benchmarks
    • Quick validation (few replications, basic comparison)
    • Full validation suite (multi-replication + all statistical tests)
    • Standalone sensitivity analysis
    • Validation metrics metadata

All heavy simulation / validation imports are done lazily inside endpoint
functions so that import-time errors (e.g. missing scipy) don't prevent
the rest of the application from starting.
"""

from __future__ import annotations

import math
import sys
import traceback
from pathlib import Path
from typing import Any

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

# ── Ensure the simulator package is importable ───────────────────────────
STUDIO_ROOT = Path(__file__).resolve().parents[1]
SIMULATOR_ROOT = STUDIO_ROOT / "simulator"
if str(SIMULATOR_ROOT) not in sys.path:
    sys.path.insert(0, str(SIMULATOR_ROOT))

router = APIRouter(prefix="/api/validation", tags=["validation"])


# ═══════════════════════════════════════════════════════════════════════════
# Pydantic Request / Response Models
# ═══════════════════════════════════════════════════════════════════════════


class QuickValidationRequest(BaseModel):
    """Parameters for a lightweight, quick validation run."""

    scenario_key: str = "baseline"
    n_replications: int = Field(default=5, ge=1, le=50)
    seed_start: int = Field(default=1, ge=0, le=100000)
    hours_override: int | None = Field(default=168, ge=24, le=1440)


class FullValidationRequest(BaseModel):
    """Parameters for the comprehensive validation suite."""

    scenario_key: str = "baseline"
    n_replications: int = Field(default=30, ge=1, le=100)
    seed_start: int = Field(default=1, ge=0, le=100000)
    hours_override: int | None = Field(default=168, ge=24, le=1440)
    include_sensitivity: bool = False
    sensitivity_replications: int = Field(default=5, ge=1, le=20)


class SensitivityRequest(BaseModel):
    """Parameters for a standalone sensitivity analysis."""

    scenario_key: str = "baseline"
    parameters: list[str] | None = None
    variation_pct: float = Field(default=20.0, ge=5.0, le=50.0)
    n_replications: int = Field(default=5, ge=1, le=20)
    target_metric: str = "shortage_rate"


# ═══════════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════════


def _sanitize(obj: Any) -> Any:
    """Recursively convert numpy / non-JSON-serializable types to native Python."""
    # Fast path for common primitives
    if obj is None or isinstance(obj, (bool, str)):
        return obj
    if isinstance(obj, (int, float)):
        if isinstance(obj, float) and (math.isnan(obj) or math.isinf(obj)):
            return None
        return obj

    # numpy scalar types
    try:
        import numpy as np

        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            v = float(obj)
            return None if (math.isnan(v) or math.isinf(v)) else v
        if isinstance(obj, np.bool_):
            return bool(obj)
        if isinstance(obj, np.ndarray):
            return _sanitize(obj.tolist())
    except ImportError:
        pass

    # Containers
    if isinstance(obj, dict):
        return {str(k): _sanitize(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_sanitize(v) for v in obj]

    # Fallback – try converting to a Python primitive
    try:
        return float(obj)
    except (TypeError, ValueError):
        return str(obj)


def _ref_to_dict(ref: Any) -> dict:
    """Convert a RealWorldReference dataclass to a JSON-safe dict."""
    return {
        "metric": ref.metric,
        "expected_value": _sanitize(ref.expected_value),
        "expected_low": _sanitize(ref.expected_low),
        "expected_high": _sanitize(ref.expected_high),
        "unit": ref.unit,
        "source": ref.source,
        "category": ref.category,
        "year": ref.year,
        "notes": ref.notes,
        "confidence": ref.confidence,
        "midpoint": _sanitize(ref.midpoint),
        "range_str": ref.range_str,
    }


def _benchmark_to_dict(bm: Any) -> dict:
    """Convert a Benchmark dataclass to a JSON-safe dict."""
    return {
        "metric": bm.metric,
        "value_low": _sanitize(bm.value_low),
        "value_high": _sanitize(bm.value_high),
        "unit": bm.unit,
        "source": bm.source,
        "notes": bm.notes,
        "confidence": bm.confidence,
        "midpoint": _sanitize(bm.midpoint),
        "range_str": bm.range_str,
    }


def _category_to_dict(cat: Any) -> dict:
    """Convert a BenchmarkCategory to a JSON-safe dict."""
    return {
        "name": cat.name,
        "description": cat.description,
        "benchmarks": [_benchmark_to_dict(bm) for bm in cat.benchmarks],
        "count": len(cat.benchmarks),
    }


# ═══════════════════════════════════════════════════════════════════════════
# Endpoints
# ═══════════════════════════════════════════════════════════════════════════


# ── 1. Reference data ────────────────────────────────────────────────────


@router.get("/references")
def get_references() -> dict:
    """Return all real-world reference data organised by category.

    Returns a dict with:
    - **categories**: mapping of category key → list of reference entries
    - **total_references**: total count across all categories
    - **literature_citations**: full bibliographic strings keyed by short cite key
    """
    try:
        from validation_reference_data import (
            ALL_REFERENCES,
            LITERATURE_CITATIONS,
            QUEBEC_REFERENCE_DATA,
        )
    except ImportError as exc:
        raise HTTPException(
            status_code=503,
            detail=f"Validation reference data module unavailable: {exc}",
        )

    categories: dict[str, list[dict]] = {}
    for cat_key, refs in QUEBEC_REFERENCE_DATA.items():
        categories[cat_key] = [_ref_to_dict(r) for r in refs]

    return _sanitize(
        {
            "categories": categories,
            "total_references": len(ALL_REFERENCES),
            "literature_citations": dict(LITERATURE_CITATIONS),
        }
    )


# ── 2. Benchmarks ────────────────────────────────────────────────────────


@router.get("/benchmarks")
def get_benchmarks() -> dict:
    """Return all validation benchmarks grouped by category.

    Returns a dict with:
    - **categories**: list of category objects (name, description, benchmarks)
    - **total_benchmarks**: total benchmark count
    """
    try:
        from validation_benchmarks import ALL_BENCHMARKS, ALL_CATEGORIES
    except ImportError as exc:
        raise HTTPException(
            status_code=503,
            detail=f"Validation benchmarks module unavailable: {exc}",
        )

    return _sanitize(
        {
            "categories": [_category_to_dict(cat) for cat in ALL_CATEGORIES],
            "total_benchmarks": len(ALL_BENCHMARKS),
        }
    )


# ── 3. Quick validation ──────────────────────────────────────────────────


@router.post("/quick")
def quick_validation(request: QuickValidationRequest) -> dict:
    """Run a quick validation with a small number of replications.

    Executes *n_replications* independent simulation runs, computes
    aggregate statistics, and compares the results against known
    real-world benchmarks.

    Returns:
    - **replications**: per-run summary dicts
    - **aggregate**: mean, std, min, max for key metrics
    - **comparison**: benchmark comparison results
    - **pass_rate**: fraction of benchmark checks that passed (0–1)
    - **verdict**: human-readable overall assessment
    """
    try:
        import numpy as np
        from validation_benchmarks import validate_simulation_summary
        from validation_engine import run_replications
    except ImportError as exc:
        raise HTTPException(
            status_code=503,
            detail=f"Required simulation modules unavailable: {exc}",
        )

    try:
        summaries = run_replications(
            scenario_key=request.scenario_key,
            n_replications=request.n_replications,
            seed_start=request.seed_start,
            hours_override=request.hours_override,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Simulation failed: {exc}\n{traceback.format_exc()}",
        )

    if not summaries:
        raise HTTPException(status_code=500, detail="No replication results returned.")

    sim_hours = summaries[0].get("sim_hours", 168)
    sim_days = sim_hours / 24.0

    # ── Aggregate key metrics across replications ────────────────────
    metric_keys = [
        "shortage_rate",
        "rejection_rate",
        "total_donated",
        "total_transfused",
        "total_expired",
        "total_net_requested",
    ]

    aggregate: dict[str, dict[str, Any]] = {}
    for key in metric_keys:
        values = []
        for s in summaries:
            v = s.get(key)
            if v is not None:
                values.append(float(v))
        if values:
            arr = np.asarray(values, dtype=np.float64)
            aggregate[key] = {
                "mean": float(np.mean(arr)),
                "std": float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0,
                "min": float(np.min(arr)),
                "max": float(np.max(arr)),
                "n": len(values),
            }

    # Derived metrics
    total_donated_vals = [float(s.get("total_donated", 0)) for s in summaries]
    if sim_days > 0 and total_donated_vals:
        daily_donations = [v / sim_days for v in total_donated_vals]
        arr = np.asarray(daily_donations, dtype=np.float64)
        aggregate["daily_donations"] = {
            "mean": float(np.mean(arr)),
            "std": float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0,
            "min": float(np.min(arr)),
            "max": float(np.max(arr)),
            "n": len(daily_donations),
        }

    # ── Benchmark comparison (use mean summary) ──────────────────────
    mean_summary: dict[str, Any] = {}
    for key in metric_keys:
        values = [float(s.get(key, 0)) for s in summaries]
        mean_summary[key] = float(np.mean(values)) if values else 0.0

    # Also include fields needed by validate_simulation_summary
    for extra in ("total_exact_match_units",):
        values = [float(s.get(extra, 0)) for s in summaries]
        if values:
            mean_summary[extra] = float(np.mean(values))

    try:
        comparison_results = validate_simulation_summary(
            mean_summary, sim_hours=sim_hours
        )
    except Exception as exc:
        comparison_results = [{"error": str(exc)}]

    # Serialise benchmark tuples inside comparison results
    serialized_comparison = []
    for r in comparison_results:
        entry: dict[str, Any] = {
            "metric_name": r.get("metric_name", ""),
            "simulated_value": _sanitize(r.get("simulated_value")),
            "passed": r.get("passed", False),
            "n_matches": r.get("n_matches", 0),
        }
        matched = r.get("matched", [])
        entry["matched"] = []
        for item in matched:
            if isinstance(item, (list, tuple)) and len(item) == 3:
                bm, in_range, deviation = item
                entry["matched"].append(
                    {
                        "benchmark": _benchmark_to_dict(bm),
                        "in_range": bool(in_range),
                        "deviation_pct": _sanitize(deviation),
                    }
                )
        serialized_comparison.append(entry)

    n_passed = sum(1 for r in comparison_results if r.get("passed", False))
    n_total = len(comparison_results)
    pass_rate = n_passed / n_total if n_total > 0 else 0.0

    if pass_rate >= 0.9:
        verdict = "EXCELLENT — simulation closely matches real-world benchmarks"
    elif pass_rate >= 0.7:
        verdict = "GOOD — most checks pass; minor calibration adjustments may help"
    elif pass_rate >= 0.5:
        verdict = "FAIR — several checks failed; review flagged metrics"
    else:
        verdict = "NEEDS ATTENTION — significant deviations from benchmarks detected"

    # Strip internal / non-serializable fields from per-replication summaries
    clean_summaries = []
    for s in summaries:
        clean = {k: _sanitize(v) for k, v in s.items() if not k.startswith("_")}
        clean_summaries.append(clean)

    return _sanitize(
        {
            "replications": clean_summaries,
            "aggregate": aggregate,
            "comparison": serialized_comparison,
            "pass_rate": pass_rate,
            "verdict": verdict,
            "sim_hours": sim_hours,
            "sim_days": sim_days,
            "n_replications": request.n_replications,
        }
    )


# ── 4. Full validation suite ─────────────────────────────────────────────


@router.post("/full")
def full_validation(request: FullValidationRequest) -> dict:
    """Run the comprehensive validation suite.

    Executes the full battery of statistical tests provided by
    ``validation_engine.run_full_validation``, including:
    - Multi-replication output analysis (MAPE, TOST, CI coverage, Theil's U)
    - Blood-type distribution chi-squared goodness-of-fit
    - Component mix chi-squared test
    - Inventory stationarity (ADF test)
    - Optional sensitivity analysis

    Returns the complete ``ValidationReport.to_dict()`` payload.
    """
    try:
        from validation_engine import run_full_validation
    except ImportError as exc:
        raise HTTPException(
            status_code=503,
            detail=f"Validation engine unavailable: {exc}",
        )

    try:
        report = run_full_validation(
            scenario_key=request.scenario_key,
            n_replications=request.n_replications,
            seed_start=request.seed_start,
            include_sensitivity=request.include_sensitivity,
            sensitivity_replications=request.sensitivity_replications,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Full validation failed: {exc}\n{traceback.format_exc()}",
        )

    try:
        result = report.to_dict()
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to serialise validation report: {exc}",
        )

    # Attach human-readable summary table as a bonus field
    try:
        result["summary_table_ascii"] = report.summary_table()
    except Exception:
        result["summary_table_ascii"] = None

    return _sanitize(result)


# ── 5. Sensitivity analysis ──────────────────────────────────────────────


@router.post("/sensitivity")
def sensitivity_analysis(request: SensitivityRequest) -> dict:
    """Run a standalone one-at-a-time (OAT) sensitivity analysis.

    Perturbs each specified parameter by ±*variation_pct*% and measures
    the resulting change in output metrics.  Also returns pre-sorted
    tornado diagram data for the requested *target_metric*.

    Returns:
    - **analysis**: raw OAT sensitivity results
    - **tornado**: list of parameter effects sorted by |sensitivity index|
    """
    try:
        from validation_engine import oat_sensitivity_analysis, tornado_data
    except ImportError as exc:
        raise HTTPException(
            status_code=503,
            detail=f"Validation engine unavailable: {exc}",
        )

    try:
        analysis = oat_sensitivity_analysis(
            base_scenario_key=request.scenario_key,
            parameters=request.parameters,
            variation_pct=request.variation_pct,
            n_replications=request.n_replications,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Sensitivity analysis failed: {exc}\n{traceback.format_exc()}",
        )

    try:
        tornado = tornado_data(analysis, target_metric=request.target_metric)
    except Exception as exc:
        tornado = [{"error": str(exc)}]

    return _sanitize(
        {
            "analysis": analysis,
            "tornado": tornado,
        }
    )


# ── 6. Metrics metadata ──────────────────────────────────────────────────


@router.get("/metrics-info")
def metrics_info() -> dict:
    """Return metadata about the available validation metrics and tests.

    Returns:
    - **metrics**: list of metric descriptors (name, description, category)
    - **tests_available**: list of statistical tests that can be run
    - **citation_count**: number of literature citations backing the reference data
    """
    # Attempt to load citation count
    citation_count = 0
    try:
        from validation_reference_data import LITERATURE_CITATIONS

        citation_count = len(LITERATURE_CITATIONS)
    except ImportError:
        pass

    # Check which statistical capabilities are available
    scipy_available = False
    try:
        from scipy import stats  # noqa: F401

        scipy_available = True
    except ImportError:
        pass

    statsmodels_available = False
    try:
        from statsmodels.tsa.stattools import adfuller  # noqa: F401

        statsmodels_available = True
    except ImportError:
        pass

    metrics = [
        {
            "name": "shortage_rate",
            "description": "Percentage of demand that could not be fulfilled",
            "category": "output",
            "unit": "%",
        },
        {
            "name": "rejection_rate",
            "description": "Donor deferral / rejection rate",
            "category": "output",
            "unit": "%",
        },
        {
            "name": "daily_donations",
            "description": "Average completed donations per day",
            "category": "output",
            "unit": "donations/day",
        },
        {
            "name": "wastage_rate",
            "description": "Expired / wasted products as a fraction of total supply",
            "category": "output",
            "unit": "%",
        },
        {
            "name": "service_fulfillment",
            "description": "Fraction of hospital orders fully satisfied",
            "category": "output",
            "unit": "%",
        },
        {
            "name": "blood_type_distribution",
            "description": "Chi-squared test of ABO/Rh type frequencies",
            "category": "distribution",
            "unit": "categorical",
        },
        {
            "name": "component_mix",
            "description": "Chi-squared test of shortage distribution across components",
            "category": "distribution",
            "unit": "categorical",
        },
        {
            "name": "inventory_stationarity",
            "description": "Augmented Dickey-Fuller test for inventory time-series stationarity",
            "category": "temporal",
            "unit": "p-value",
        },
        {
            "name": "autocorrelation",
            "description": "Ljung-Box test for temporal dependence in output series",
            "category": "temporal",
            "unit": "p-value",
        },
    ]

    tests_available = [
        {
            "name": "MAPE",
            "full_name": "Mean Absolute Percentage Error",
            "available": True,
            "description": "Measures average magnitude of percentage deviations from reference",
        },
        {
            "name": "Theil's U",
            "full_name": "Theil's Inequality Coefficient",
            "available": True,
            "description": "Decomposes forecast error into bias, variance, and covariance",
        },
        {
            "name": "CI Coverage",
            "full_name": "Confidence Interval Coverage Test",
            "available": scipy_available,
            "description": "Checks whether simulated CI covers the real-world reference range",
        },
        {
            "name": "TOST",
            "full_name": "Two One-Sided Tests for Equivalence",
            "available": scipy_available,
            "description": "Tests practical equivalence within a specified margin (Schuirmann 1987)",
        },
        {
            "name": "NRMSE",
            "full_name": "Normalised Root Mean Square Error",
            "available": True,
            "description": "Root mean square error normalised by the reference value",
        },
        {
            "name": "Chi-squared",
            "full_name": "Chi-squared Goodness-of-Fit",
            "available": scipy_available,
            "description": "Tests distributional fit for categorical variables (blood types, components)",
        },
        {
            "name": "ADF",
            "full_name": "Augmented Dickey-Fuller Test",
            "available": statsmodels_available,
            "description": "Tests time-series stationarity for inventory levels",
        },
        {
            "name": "OAT Sensitivity",
            "full_name": "One-at-a-Time Sensitivity Analysis",
            "available": True,
            "description": "Perturbs each parameter individually to measure output sensitivity",
        },
    ]

    return {
        "metrics": metrics,
        "tests_available": tests_available,
        "citation_count": citation_count,
        "scipy_available": scipy_available,
        "statsmodels_available": statsmodels_available,
    }
