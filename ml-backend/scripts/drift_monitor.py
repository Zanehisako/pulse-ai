#!/usr/bin/env python
"""
drift_monitor.py
================
Drift detection for all registered ML models.

Logs results to:
  1. **MLflow** — dedicated experiment ``model_drift_monitoring`` with per-feature
     metrics, so drift trends are visible in the MLflow UI.
  2. **Django DB** — ``DriftReport`` rows for the REST API / dashboard.

Usage:
    python scripts/drift_monitor.py                      # all models
    python scripts/drift_monitor.py --model-name donor_next_donation_hazard_model
    python scripts/drift_monitor.py --reference-window 90  # last 90 days for reference
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import sys
import tempfile
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

# ── Encoding safety (Windows compat) ─────────────────────────────────────
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[union-attr]
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[union-attr]

# ── Paths ─────────────────────────────────────────────────────────────────
_SCRIPTS_DIR = Path(__file__).resolve().parent  # ml-backend/scripts/
ROOT_DIR = _SCRIPTS_DIR.parent  # ml-backend/
_BACKEND_DIR = ROOT_DIR.parent / "backendMulti"  # backendMulti/

DATASETS_DIR = ROOT_DIR / "datasets_featues_labels_seperated"
FEATURE_REPO_DIR = ROOT_DIR / "feature_repo"

sys.path.insert(0, str(ROOT_DIR))
sys.path.insert(0, str(_BACKEND_DIR))

# ── Django bootstrap ──────────────────────────────────────────────────────
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "backendMulti.settings")

import django  # noqa: E402

django.setup()

# ── MLflow ────────────────────────────────────────────────────────────────
import mlflow  # noqa: E402
import pandas as pd  # noqa: E402
from django.utils import timezone  # noqa: E402
from mlflow import MlflowClient  # noqa: E402

# Use the same tracking URI as the training scripts so that drift runs
# appear in the same MLflow UI alongside training experiments.
# In Docker the MLflow server runs at http://localhost:8889 backed by
# PostgreSQL — every script (training, promotion, sync, drift) must
# point there so all data lives in one place.
mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://localhost:8889"))

logger = logging.getLogger("pios.drift_monitor")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)

# ── Drift detection engine ────────────────────────────────────────────────
# ── Django models ─────────────────────────────────────────────────────────
from ml.models import DriftReport, PredictionLog  # noqa: E402

from pios_ml_backend.drift_detection import (  # noqa: E402
    DriftReport as DriftReportData,
)
from pios_ml_backend.drift_detection import (
    build_drift_report,
)

# ── Drift experiment name in MLflow ───────────────────────────────────────
DRIFT_EXPERIMENT = "model_drift_monitoring"

# ── Model configuration ──────────────────────────────────────────────────

MODEL_DRIFT_CONFIG: dict[str, dict] = {}


# ── Helpers ───────────────────────────────────────────────────────────────


def _sanitise_metric_name(name: str) -> str:
    """MLflow metric keys must match ``[a-zA-Z0-9._\\-/ ]``.

    Replace anything else with an underscore.
    """
    return re.sub(r"[^a-zA-Z0-9._\-/ ]", "_", name)


def _load_parquet_safe(path: Path) -> pd.DataFrame | None:
    """Load a parquet file, returning *None* on any error."""
    if not path.exists():
        logger.warning("File not found: %s", path)
        return None
    try:
        return pd.read_parquet(path)
    except Exception as exc:
        logger.warning("Failed to read %s: %s", path, exc)
        return None


def _get_champion_run_id(client: MlflowClient, model_name: str) -> str | None:
    """Return the MLflow *run_id* behind the current champion version."""
    try:
        mv = client.get_model_version_by_alias(model_name, "champion")
        return mv.run_id
    except Exception:
        return None


def _load_reference_from_run(
    client: MlflowClient,
    run_id: str,
) -> pd.DataFrame | None:
    """Try to recover the training dataset that was logged with the run."""
    try:
        run = client.get_run(run_id)
        inputs = run.inputs.dataset_inputs
        if inputs:
            # The first logged dataset is assumed to be the training set.
            ds_info = inputs[0].dataset
            logger.info("Found logged dataset '%s' on run %s", ds_info.name, run_id)
            # MLflow ≥ 2.x stores the dataset source URI — attempt download.
            source_uri = ds_info.source
            if source_uri:
                from mlflow.artifacts import download_artifacts

                local = download_artifacts(artifact_uri=source_uri)
                if local.endswith(".parquet"):
                    return pd.read_parquet(local)
                elif local.endswith(".csv"):
                    return pd.read_csv(local)
    except Exception as exc:
        logger.debug("Could not load dataset from run %s: %s", run_id, exc)
    return None


def _build_current_from_prediction_log(
    model_name: str,
    feature_columns: list[str],
) -> pd.DataFrame | None:
    """Reconstruct a 'current' dataframe from Django ``PredictionLog`` rows."""
    qs = (
        PredictionLog.objects.filter(
            model_id_used=model_name,
            success=True,
        )
        .order_by("-created_at")
        .values_list("features_input", flat=True)[:5000]
    )
    rows = list(qs)
    if not rows:
        logger.warning(
            "No PredictionLog entries for model '%s' — cannot build current data.",
            model_name,
        )
        return None

    df = pd.DataFrame(rows)
    # Keep only columns that match our expected features
    common = [c for c in feature_columns if c in df.columns]
    if not common:
        logger.warning("PredictionLog features don't overlap with reference features.")
        return None
    return pd.DataFrame(df[common])


def _severity_label(score: float) -> str:
    """Map an overall drift score (0‑1) to a severity label."""
    if score >= 0.5:
        return "critical"
    if score >= 0.15:
        return "warning"
    return "none"


# ── Core: run drift for a single model ────────────────────────────────────


def run_drift_check(
    model_name: str,
    cfg: dict,
    *,
    client: MlflowClient,
    reference_window: int | None = None,
    production_mode: bool = False,
    threshold_psi: float | None = None,
    threshold_ks: float | None = None,
    check_type: str = "scheduled",
) -> dict:
    """Run drift detection for *model_name* and return a result dict.

    Steps:
        1. Load reference data (training parquet → optional MLflow dataset).
        2. Build current data (dev split or PredictionLog).
        3. Call ``build_drift_report``.
        4. Log everything to MLflow (experiment ``model_drift_monitoring``).
        5. Persist to Django ``DriftReport``.
    """

    result: dict = {
        "model_name": model_name,
        "action": "checked",
        "drift_detected": False,
        "drift_score": 0.0,
        "features_drifted": 0,
        "features_checked": 0,
        "mlflow_run_id": None,
        "reason": "",
    }

    # ── 1. Load features file ─────────────────────────────────────────
    features_path = DATASETS_DIR / cfg["features_file"]
    features_df = _load_parquet_safe(features_path)
    if features_df is None:
        result["action"] = "skip"
        result["reason"] = f"Features file not found: {features_path.name}"
        logger.warning("Skipping %s — %s", model_name, result["reason"])
        return result

    # Drop non-numeric / label leakage columns that might have snuck in
    label_col = cfg.get("label_column")
    if label_col and label_col in features_df.columns:
        features_df = features_df.drop(columns=[label_col])

    # Attempt to use champion run's logged dataset as reference
    champion_run_id = _get_champion_run_id(client, model_name)
    reference_from_run: pd.DataFrame | None = None
    if champion_run_id:
        reference_from_run = _load_reference_from_run(client, champion_run_id)

    reference_df: pd.DataFrame = (
        reference_from_run if reference_from_run is not None else features_df.copy()
    )

    # Apply reference window (simulate time filter using row order)
    if reference_window is not None and reference_window > 0:
        # Use last N rows as a proxy for "last N days"
        keep = min(len(reference_df), reference_window * 24)  # rough heuristic
        reference_df = reference_df.tail(keep).reset_index(drop=True)

    # ── 2. Build current data ─────────────────────────────────────────
    current_df: pd.DataFrame
    if production_mode:
        feature_cols = list(reference_df.columns)
        current_df_or_none = _build_current_from_prediction_log(
            model_name, feature_cols
        )
        if current_df_or_none is None:
            result["action"] = "skip"
            result["reason"] = "No PredictionLog data available in production mode"
            logger.warning("Skipping %s — %s", model_name, result["reason"])
            return result
        current_df = current_df_or_none
    else:
        # Dev mode: split the reference data 70/30
        split_idx = int(len(reference_df) * 0.7)
        if split_idx < 10 or (len(reference_df) - split_idx) < 10:
            result["action"] = "skip"
            result["reason"] = (
                f"Not enough data to split (total rows: {len(reference_df)})"
            )
            logger.warning("Skipping %s — %s", model_name, result["reason"])
            return result

        current_df = pd.DataFrame(reference_df.iloc[split_idx:]).reset_index(drop=True)
        reference_df = pd.DataFrame(reference_df.iloc[:split_idx]).reset_index(
            drop=True
        )

    # Align columns — keep only shared features
    shared_cols = sorted(
        set(reference_df.columns).intersection(set(current_df.columns))
    )
    if not shared_cols:
        result["action"] = "skip"
        result["reason"] = "No overlapping features between reference and current"
        logger.warning("Skipping %s — %s", model_name, result["reason"])
        return result

    reference_df = pd.DataFrame(reference_df[shared_cols])
    current_df = pd.DataFrame(current_df[shared_cols])

    # Drop datetime / timedelta columns — they aren't meaningful for drift
    _dt_cols = [
        c
        for c in reference_df.columns
        if pd.api.types.is_datetime64_any_dtype(reference_df[c])
        or pd.api.types.is_timedelta64_dtype(reference_df[c])
    ]
    if _dt_cols:
        logger.info(
            "Dropping %d datetime column(s) for %s: %s",
            len(_dt_cols),
            model_name,
            _dt_cols,
        )
        reference_df = reference_df.drop(columns=_dt_cols)
        current_df = current_df.drop(columns=_dt_cols)

    # Drop high-cardinality entity-ID columns (e.g. donor_id, hospital_id)
    _id_cols = [
        c
        for c in reference_df.columns
        if c.endswith("_id") and reference_df[c].dtype.kind in ("O", "S", "U")
    ]
    if _id_cols:
        logger.info(
            "Dropping %d entity-ID column(s) for %s: %s",
            len(_id_cols),
            model_name,
            _id_cols,
        )
        reference_df = reference_df.drop(columns=_id_cols)
        current_df = current_df.drop(columns=_id_cols)

    if reference_df.empty or current_df.empty:
        result["action"] = "skip"
        result["reason"] = "No testable features after filtering datetime/ID columns"
        logger.warning("Skipping %s — %s", model_name, result["reason"])
        return result

    # ── 3. Build drift config overrides ───────────────────────────────
    drift_config: dict = {"task_type": cfg.get("task_type", "regression")}
    if threshold_psi is not None:
        drift_config["psi_threshold"] = threshold_psi
    if threshold_ks is not None:
        drift_config["ks_threshold"] = threshold_ks

    # ── 4. Run drift detection ────────────────────────────────────────
    report_dict: dict = {}
    try:
        report: DriftReportData = build_drift_report(
            model_id=model_name,
            reference_df=reference_df,
            current_df=current_df,
            config=drift_config,
        )
    except Exception as exc:
        result["action"] = "error"
        result["reason"] = f"Drift detection failed: {exc}"
        logger.exception("Drift detection error for %s", model_name)
        return result

    # ── 5. Log to MLflow ──────────────────────────────────────────────
    now = timezone.now()
    ts_str = now.strftime("%Y%m%d_%H%M%S")
    run_name = f"drift_check_{model_name}_{ts_str}"

    mlflow.set_experiment(DRIFT_EXPERIMENT)

    mlflow_run_id: str | None = None

    with mlflow.start_run(run_name=run_name) as run:
        mlflow_run_id = run.info.run_id

        # ── Tags ──────────────────────────────────────────────────
        mlflow.set_tags(
            {
                "model_name": model_name,
                "check_type": check_type,
                "source_experiment": cfg.get("experiment_name", ""),
                "production_mode": str(production_mode),
            }
        )

        # ── Params ────────────────────────────────────────────────
        mlflow.log_params(
            {
                "reference_size": report.reference_size,
                "current_size": report.current_size,
                "features_checked": report.features_checked,
                "task_type": cfg.get("task_type", "regression"),
                "psi_threshold": drift_config.get("psi_threshold", "default"),
                "ks_threshold": drift_config.get("ks_threshold", "default"),
            }
        )

        # ── Per-feature metrics ───────────────────────────────────
        for fr in report.feature_results:
            prefix = f"drift.{_sanitise_metric_name(fr.feature_name)}"

            if fr.test_name == "ks":
                mlflow.log_metric(f"{prefix}.ks_statistic", fr.statistic)
                if fr.p_value is not None:
                    mlflow.log_metric(f"{prefix}.ks_pvalue", fr.p_value)
                mlflow.log_metric(
                    f"{prefix}.drift_detected", 1 if fr.drift_detected else 0
                )
            elif fr.test_name == "psi":
                mlflow.log_metric(f"{prefix}.psi", fr.statistic)
                mlflow.log_metric(
                    f"{prefix}.drift_detected", 1 if fr.drift_detected else 0
                )
            else:
                # chi2, js, or other
                mlflow.log_metric(f"{prefix}.{fr.test_name}_statistic", fr.statistic)
                if fr.p_value is not None:
                    mlflow.log_metric(f"{prefix}.{fr.test_name}_pvalue", fr.p_value)
                mlflow.log_metric(
                    f"{prefix}.drift_detected", 1 if fr.drift_detected else 0
                )

        # ── Overall metrics ───────────────────────────────────────
        mlflow.log_metrics(
            {
                "drift.overall_score": report.overall_drift_score,
                "drift.features_drifted_count": report.features_drifted,
                "drift.features_checked": report.features_checked,
                "drift.overall_drift_detected": (
                    1 if report.overall_drift_detected else 0
                ),
            }
        )

        # ── Prediction drift (if present) ─────────────────────────
        if report.prediction_drift is not None:
            pd_res = report.prediction_drift
            mlflow.log_metrics(
                {
                    "drift.prediction.ks_statistic": pd_res.statistic,
                    "drift.prediction.drift_detected": (
                        1 if pd_res.drift_detected else 0
                    ),
                }
            )
            if pd_res.p_value is not None:
                mlflow.log_metric("drift.prediction.ks_pvalue", pd_res.p_value)

        # ── Artifact: full JSON report ────────────────────────────
        report_dict = report.to_dict()
        report_dict["checked_at"] = now.isoformat()

        with tempfile.TemporaryDirectory() as tmp_dir:
            report_path = Path(tmp_dir) / "drift_report.json"
            report_path.write_text(
                json.dumps(report_dict, indent=2, default=str),
                encoding="utf-8",
            )
            mlflow.log_artifact(str(report_path))

    # ── 6. Persist to Django DB ───────────────────────────────────────
    severity = _severity_label(report.overall_drift_score)

    try:
        DriftReport.objects.update_or_create(
            model_id=model_name,
            mlflow_run_id=mlflow_run_id or "",
            defaults={
                "drift_detected": report.overall_drift_detected,
                "drift_score": report.overall_drift_score,
                "severity": severity,
                "features_drifted": report.features_drifted,
                "features_checked": report.features_checked,
                "feature_details": report_dict,
                "reference_size": report.reference_size,
                "current_size": report.current_size,
                "check_type": check_type,
                "checked_at": now,
            },
        )
        logger.info("DriftReport saved for %s (run_id=%s)", model_name, mlflow_run_id)
    except Exception as exc:
        logger.error("Failed to save DriftReport to Django DB: %s", exc)

    # ── 7. Populate result dict ───────────────────────────────────────
    result.update(
        {
            "drift_detected": report.overall_drift_detected,
            "drift_score": round(report.overall_drift_score, 4),
            "features_drifted": report.features_drifted,
            "features_checked": report.features_checked,
            "severity": severity,
            "mlflow_run_id": mlflow_run_id,
            "reason": report.summary
            or ("Drift detected" if report.overall_drift_detected else "No drift"),
        }
    )

    return result


# ── CLI ───────────────────────────────────────────────────────────────────


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Drift detection for registered ML models",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python scripts/drift_monitor.py\n"
            "  python scripts/drift_monitor.py --model-name hospital_shortage_predictor\n"
            "  python scripts/drift_monitor.py --reference-window 90 --production-mode\n"
        ),
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default=None,
        help="Run drift check for a single model (must be a key in MODEL_DRIFT_CONFIG).",
    )
    parser.add_argument(
        "--reference-window",
        type=int,
        default=None,
        help="Number of days of data to use as reference (default: all available).",
    )
    parser.add_argument(
        "--production-mode",
        action="store_true",
        default=False,
        help=(
            "Use PredictionLog data as 'current' distribution instead of a "
            "70/30 dev split."
        ),
    )
    parser.add_argument(
        "--threshold-psi",
        type=float,
        default=None,
        help="Override default PSI threshold (default: 0.2).",
    )
    parser.add_argument(
        "--threshold-ks",
        type=float,
        default=None,
        help="Override default KS p-value threshold (default: 0.05).",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    client = MlflowClient()
    results: list[dict] = []

    # Determine which models to check
    if args.model_name:
        if args.model_name not in MODEL_DRIFT_CONFIG:
            print(
                f"[drift] ❌ Unknown model '{args.model_name}'. "
                f"Known: {', '.join(MODEL_DRIFT_CONFIG.keys())}"
            )
            sys.exit(1)
        models_to_check = {args.model_name: MODEL_DRIFT_CONFIG[args.model_name]}
    else:
        models_to_check = MODEL_DRIFT_CONFIG

    check_type = "manual" if args.model_name else "scheduled"

    n_models = len(models_to_check)
    print(f"\n[drift] Running drift detection for {n_models} model(s) …")
    print("=" * 70)

    for model_name, cfg in models_to_check.items():
        print(f"\n  Model: {model_name}")
        print(f"  Features file: {cfg['features_file']}")
        print(f"  Task type: {cfg['task_type']}")
        print("-" * 50)

        result = run_drift_check(
            model_name,
            cfg,
            client=client,
            reference_window=args.reference_window,
            production_mode=args.production_mode,
            threshold_psi=args.threshold_psi,
            threshold_ks=args.threshold_ks,
            check_type=check_type,
        )
        results.append(result)

        # Human-readable per-model summary
        if result["action"] == "skip":
            print(f"  ⏭️  SKIPPED — {result['reason']}")
        elif result["action"] == "error":
            print(f"  ❌ ERROR — {result['reason']}")
        else:
            icon = "🔴" if result["drift_detected"] else "🟢"
            print(f"  {icon} Drift detected: {result['drift_detected']}")
            score_pct = result["drift_score"] * 100
            print(f"     Overall score:  {score_pct:.2f}%")
            print(
                f"     Features drifted: {result['features_drifted']}"
                f" / {result['features_checked']}"
            )
            print(f"     Severity: {result.get('severity', 'none')}")
            print(f"     MLflow run: {result['mlflow_run_id']}")

    # ── Summary ───────────────────────────────────────────────────────
    print("\n" + "=" * 70)

    checked = [r for r in results if r["action"] == "checked"]
    skipped = [r for r in results if r["action"] == "skip"]
    errored = [r for r in results if r["action"] == "error"]
    drifted = [r for r in checked if r["drift_detected"]]

    print(
        f"\n[drift] Summary: {len(checked)} checked, "
        f"{len(drifted)} drifted, "
        f"{len(skipped)} skipped, "
        f"{len(errored)} errors"
    )

    if drifted:
        print("\n[drift] ⚠️  Models with detected drift:")
        for r in drifted:
            print(
                f"  • {r['model_name']}: score={r['drift_score']:.2%}, "
                f"severity={r.get('severity', '?')}"
            )

    # Machine-parseable output (same convention as promote_models.py)
    print("\nRESULTS_JSON:" + json.dumps(results, default=str))


if __name__ == "__main__":
    main()
