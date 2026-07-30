"""
Generate and persist ModelStats for configured SOTA models.

Usage:
    cd backendMulti/
    python ../ml-backend/scripts/generate_sota_model_stats.py

The implementation lives in the Django backend so local startup and Docker
startup use the same config-driven code path.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
ML_BACKEND_ROOT = SCRIPT_DIR.parent
BACKEND_MULTI_ROOT = ML_BACKEND_ROOT.parent / "backendMulti"

sys.path.insert(0, str(BACKEND_MULTI_ROOT))
sys.path.insert(0, str(ML_BACKEND_ROOT))
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "backendMulti.settings")

import django  # noqa: E402

django.setup()

from ml.core.sota_model_stats import (  # noqa: E402
    dashboard_profile_payload,
    feature_stats_payload,
    generate_sota_model_stats,
    merge_dashboard_profile,
)

__all__ = [
    "dashboard_profile_payload",
    "feature_stats_payload",
    "generate_sota_model_stats",
    "merge_dashboard_profile",
]


def main() -> None:
    summary = generate_sota_model_stats()
    for row in summary["processed"]:
        print(
            "  OK {model_id}: {available_features}/{configured_features} features, task={task_type}".format(
                **row
            )
        )
    for row in summary["skipped"]:
        print("  SKIP {model_id}: {reason}".format(**row))
    print(f"\nDone. Persisted stats for {summary['persisted']} models.")


if __name__ == "__main__":
    main()
