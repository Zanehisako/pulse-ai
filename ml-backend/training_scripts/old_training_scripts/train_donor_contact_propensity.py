"""
train_donor_contact_propensity.py

Trains a donor contact propensity model using behavioral proxy features.
Uses the standard classification pipeline with tabular ensemble.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

_TRAINING_SCRIPTS = Path(__file__).resolve().parent
if str(_TRAINING_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_TRAINING_SCRIPTS))

from model_training_utils import (
    FEATURE_LABELS_DIRECTORY,
    TaskDataset,
    _quantile_mask,
    _safe_numeric,
    train_and_log,
)
from pios_ml_backend.training_resource_profiles import apply_profile_to_model_config

_CONFIG_PATH = Path(__file__).resolve().parents[1] / "config" / "donor_contact_propensity_model.json"


def _load_config() -> dict:
    return apply_profile_to_model_config(
        json.loads(_CONFIG_PATH.read_text(encoding="utf-8"))
    )


def build_contact_propensity_task() -> TaskDataset:
    config = _load_config()
    target_cfg = config["target"]
    feature_cfg = config["feature_engineering"]

    features = pd.read_parquet(
        FEATURE_LABELS_DIRECTORY / "quebec_donor_features.parquet"
    )
    labels = pd.read_parquet(
        FEATURE_LABELS_DIRECTORY / "quebec_donor_labels.parquet"
    )

    merged = features.merge(
        labels[["donor_id", "event_timestamp", str(target_cfg["column"])]],
        on=["donor_id", "event_timestamp"],
        how="inner",
    )
    merged[str(target_cfg["column"])] = _safe_numeric(
        merged[str(target_cfg["column"])]
    ).fillna(0).astype(int)
    merged = merged.sort_values(["donor_id", "event_timestamp"]).reset_index(drop=True)

    def hard_mask(frame: pd.DataFrame) -> pd.Series:
        return (
            (_safe_numeric(frame.get("is_rare_type", 0)) > 0)
            | _quantile_mask(frame.get("travel_burden_score", 0), 0.75, "high")
            | _quantile_mask(frame.get("response_readiness_index", 0), 0.25, "low")
        )

    return TaskDataset(
        task_name=str(config["id"]),
        model_name=str(config["registered_model_name"]),
        experiment_name="donor_contact_propensity",
        task_type="classification",
        frame=merged,
        target_column=str(target_cfg["column"]),
        timestamp_column=str(feature_cfg["timestamp_column"]),
        feature_service="quebec_donor_service",
        entity_keys=("donor_id",),
        label_file=FEATURE_LABELS_DIRECTORY / "quebec_donor_labels.parquet",
        feature_file=FEATURE_LABELS_DIRECTORY / "quebec_donor_features.parquet",
        description=str(config["description"]),
        hard_mask_builder=hard_mask,
        prediction_defaults=dict(config.get("prediction_defaults", {})),
        training_config=dict(config.get("training", {})),
    )


if __name__ == "__main__":
    result = train_and_log(build_contact_propensity_task)
    print(f"Contact propensity model trained: {result['selected_candidate']}")
    print(f"  Overall brier: {result['overall_metrics'].get('brier_score', 'N/A')}")
    print(f"  Overall AUC: {result['overall_metrics'].get('roc_auc', 'N/A')}")
    print("Done.")
