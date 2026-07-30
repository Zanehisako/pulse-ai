"""
promote_models.py
=================
Champion / challenger promotion gate.

After a training script registers a new model version with the ``challenger``
alias, this script compares the challenger's MLflow run metrics against the
current ``champion``.  The challenger is promoted (alias swapped to
``champion``) **only** if it meets or beats the champion on the configured
primary metric.

Usage:
    python promote_models.py                   # check ALL registered models
    python promote_models.py --model-name XYZ  # check a single model

Promotion rules come from ``ml-backend/config/training_schedule.json``
under the ``"promotion_rules"`` key.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

_SCRIPTS_DIR = Path(__file__).resolve().parent
_ROOT_DIR = _SCRIPTS_DIR.parent
_BACKEND_DIR = _ROOT_DIR.parent / "backendMulti"
_CONFIG_PATH = _ROOT_DIR / "config" / "training_schedule.json"

sys.path.insert(0, str(_ROOT_DIR))
sys.path.insert(0, str(_BACKEND_DIR))

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "backendMulti.settings")

import django  # noqa: E402

django.setup()

import mlflow  # noqa: E402
from mlflow import MlflowClient  # noqa: E402

mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://localhost:8889"))

# ── Helpers ───────────────────────────────────────────────────────────────

DEFAULT_RULES: dict = {
    "primary_metric": "root_mean_squared_error",
    "direction": "minimize",
    "min_improvement_pct": 0.0,
    "fallback_metrics": [],
}


def _load_promotion_rules() -> dict:
    """Load promotion rules from the training schedule config."""
    if not _CONFIG_PATH.exists():
        return {}
    with open(_CONFIG_PATH, "r", encoding="utf-8") as fh:
        config = json.load(fh)
    return config.get("promotion_rules", {})


def _rules_for(model_name: str, all_rules: dict) -> dict:
    """Get promotion rules for a model, falling back to defaults."""
    base = {**DEFAULT_RULES, **all_rules.get("default", {})}
    override = all_rules.get(model_name, {})
    return {**base, **override}


def _is_better(
    challenger_val: float,
    champion_val: float,
    direction: str,
    min_improvement_pct: float,
) -> bool:
    """
    Return True if the challenger is better than the champion.

    For 'minimize' (e.g. RMSE, MAE):
        challenger must be <= champion * (1 - min_improvement_pct/100)
    For 'maximize' (e.g. R², AUC):
        challenger must be >= champion * (1 + min_improvement_pct/100)
    """
    if direction == "minimize":
        threshold = champion_val * (1.0 - min_improvement_pct / 100.0)
        return challenger_val <= threshold
    else:  # maximize
        threshold = champion_val * (1.0 + min_improvement_pct / 100.0)
        return challenger_val >= threshold


def _get_version_by_alias(client: MlflowClient, model_name: str, alias: str):
    """Get the ModelVersion for a given alias, or None."""
    try:
        return client.get_model_version_by_alias(model_name, alias)
    except Exception:
        return None


def _get_run_metrics(client: MlflowClient, run_id: str) -> dict:
    """Get all metrics from a run."""
    try:
        run = client.get_run(run_id)
        return dict(run.data.metrics)
    except Exception:
        return {}


# ── Main promotion logic ──────────────────────────────────────────────────


def promote_model(client: MlflowClient, model_name: str, rules: dict) -> dict:
    """
    Compare challenger vs champion for a single model.

    Returns a result dict with keys: model_name, action, reason, metrics.
    """
    result = {
        "model_name": model_name,
        "action": "none",
        "reason": "",
        "champion_version": None,
        "challenger_version": None,
        "champion_metrics": {},
        "challenger_metrics": {},
    }

    # 1. Find challenger
    challenger_mv = _get_version_by_alias(client, model_name, "challenger")
    if challenger_mv is None:
        result["action"] = "skip"
        result["reason"] = "No challenger version found"
        return result

    result["challenger_version"] = challenger_mv.version

    # 2. Find current champion
    champion_mv = _get_version_by_alias(client, model_name, "champion")

    # 3. If no champion exists → auto-promote
    if champion_mv is None:
        print(
            f"  [promote] No existing champion — auto-promoting challenger v{challenger_mv.version}"
        )
        client.set_registered_model_alias(model_name, "champion", challenger_mv.version)
        client.delete_registered_model_alias(model_name, "challenger")
        result["action"] = "promoted"
        result["reason"] = "No existing champion — auto-promoted"
        return result

    result["champion_version"] = champion_mv.version

    # 4. If challenger IS the champion (same version) → clean up alias
    if challenger_mv.version == champion_mv.version:
        try:
            client.delete_registered_model_alias(model_name, "challenger")
        except Exception:
            pass
        result["action"] = "skip"
        result["reason"] = (
            f"Challenger v{challenger_mv.version} is already the champion"
        )
        return result

    # 5. Get metrics for both
    champion_metrics = _get_run_metrics(client, champion_mv.run_id)
    challenger_metrics = _get_run_metrics(client, challenger_mv.run_id)
    result["champion_metrics"] = champion_metrics
    result["challenger_metrics"] = challenger_metrics

    primary = rules["primary_metric"]
    direction = rules.get("direction", "minimize")
    min_improvement = rules.get("min_improvement_pct", 0.0)
    fallbacks = rules.get("fallback_metrics", [])

    # 6. Compare primary metric
    if primary not in champion_metrics or primary not in challenger_metrics:
        # Try fallback metrics if primary is missing
        promoted_by_fallback = False
        for fb_metric in fallbacks:
            if fb_metric in champion_metrics and fb_metric in challenger_metrics:
                print(
                    f"  [promote] Primary metric '{primary}' not found, trying fallback '{fb_metric}'"
                )
                if _is_better(
                    challenger_metrics[fb_metric],
                    champion_metrics[fb_metric],
                    direction,
                    min_improvement,
                ):
                    print(
                        f"  [promote] ✅ Challenger WINS on fallback '{fb_metric}': "
                        f"{challenger_metrics[fb_metric]:.6f} vs {champion_metrics[fb_metric]:.6f}"
                    )
                    client.set_registered_model_alias(
                        model_name, "champion", challenger_mv.version
                    )
                    client.delete_registered_model_alias(model_name, "challenger")
                    result["action"] = "promoted"
                    result["reason"] = (
                        f"Challenger better on fallback metric '{fb_metric}': "
                        f"{challenger_metrics[fb_metric]:.6f} vs {champion_metrics[fb_metric]:.6f}"
                    )
                    promoted_by_fallback = True
                    break
                else:
                    print(
                        f"  [promote] ❌ Challenger LOSES on fallback '{fb_metric}': "
                        f"{challenger_metrics[fb_metric]:.6f} vs {champion_metrics[fb_metric]:.6f}"
                    )
                    client.delete_registered_model_alias(model_name, "challenger")
                    result["action"] = "rejected"
                    result["reason"] = (
                        f"Challenger worse on fallback metric '{fb_metric}': "
                        f"{challenger_metrics[fb_metric]:.6f} vs {champion_metrics[fb_metric]:.6f}"
                    )
                    promoted_by_fallback = True
                    break

        if not promoted_by_fallback:
            # No comparable metrics found → auto-promote (benefit of the doubt)
            print(f"  [promote] ⚠️  No comparable metrics found — auto-promoting")
            client.set_registered_model_alias(
                model_name, "champion", challenger_mv.version
            )
            client.delete_registered_model_alias(model_name, "challenger")
            result["action"] = "promoted"
            result["reason"] = "No comparable metrics found — auto-promoted"
        return result

    # 7. Primary metric comparison
    champ_val = champion_metrics[primary]
    chall_val = challenger_metrics[primary]

    print(
        f"  [promote] {primary}: challenger={chall_val:.6f} vs champion={champ_val:.6f} "
        f"(direction={direction}, min_improvement={min_improvement}%)"
    )

    if _is_better(chall_val, champ_val, direction, min_improvement):
        print(
            f"  [promote] ✅ PROMOTED — challenger v{challenger_mv.version} beats champion v{champion_mv.version}"
        )
        client.set_registered_model_alias(model_name, "champion", challenger_mv.version)
        client.delete_registered_model_alias(model_name, "challenger")
        result["action"] = "promoted"
        result["reason"] = (
            f"Challenger better on '{primary}': {chall_val:.6f} vs {champ_val:.6f}"
        )
    else:
        print(f"  [promote] ❌ REJECTED — keeping champion v{champion_mv.version}")
        client.delete_registered_model_alias(model_name, "challenger")
        result["action"] = "rejected"
        result["reason"] = (
            f"Challenger not better on '{primary}': {chall_val:.6f} vs {champ_val:.6f}"
        )

    return result


# ── CLI entry point ───────────────────────────────────────────────────────


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Champion/challenger promotion gate")
    parser.add_argument(
        "--model-name",
        help="Only check this specific registered model name",
    )
    args = parser.parse_args()

    client = MlflowClient()
    all_rules = _load_promotion_rules()
    results = []

    if args.model_name:
        model_names = [args.model_name]
    else:
        model_names = [m.name for m in client.search_registered_models()]

    print(f"\n[promote] Checking {len(model_names)} model(s) for promotion …")
    print("=" * 60)

    for name in model_names:
        print(f"\n  Model: {name}")
        rules = _rules_for(name, all_rules)
        result = promote_model(client, name, rules)
        results.append(result)
        print(f"  → {result['action']}: {result['reason']}")

    print("\n" + "=" * 60)

    # Summary
    promoted = [r for r in results if r["action"] == "promoted"]
    rejected = [r for r in results if r["action"] == "rejected"]
    skipped = [r for r in results if r["action"] in ("skip", "none")]

    print(
        f"\n[promote] Summary: {len(promoted)} promoted, {len(rejected)} rejected, {len(skipped)} skipped"
    )

    # Output JSON for machine-readable consumption
    print("\n[promote] RESULTS_JSON:" + json.dumps(results))


if __name__ == "__main__":
    main()
