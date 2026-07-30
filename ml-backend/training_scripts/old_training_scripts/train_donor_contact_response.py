"""
train_donor_contact_response.py

Contact response proxy model training script. Uses donated_next_6m as proxy
target with contact-proxy features. True contact response requires labeled
contact event history data (not currently available).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

_TRAINING_SCRIPTS = Path(__file__).resolve().parent
if str(_TRAINING_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_TRAINING_SCRIPTS))

from donor_forecast_tasks import add_donor_sequence_features
from model_training_utils import (
    FEATURE_LABELS_DIRECTORY,
    TaskDataset,
    _safe_numeric,
    train_and_log,
)
from pios_ml_backend.training_resource_profiles import apply_profile_to_model_config

_CONFIG_PATH = Path(__file__).resolve().parents[1] / "config" / "donor_contact_response_model.json"


def _load_config() -> dict:
    return apply_profile_to_model_config(
        json.loads(_CONFIG_PATH.read_text(encoding="utf-8"))
    )


def build_contact_response_task() -> TaskDataset:
    config = _load_config()
    target_cfg = config["target"]
    feature_cfg = config["feature_engineering"]

    features = pd.read_parquet(
        FEATURE_LABELS_DIRECTORY / "quebec_donor_features.parquet"
    )
    labels = pd.read_parquet(
        FEATURE_LABELS_DIRECTORY / "quebec_donor_labels.parquet"
    )
    labels = labels.astype({str(target_cfg["column"]): "int32"}, errors="ignore")

    merged = features.merge(
        labels[[str(c) for c in feature_cfg["group_columns"]] + ["event_timestamp", str(target_cfg["column"])]],
        on=[str(c) for c in feature_cfg["group_columns"]] + ["event_timestamp"],
        how="inner",
    )
    merged[str(target_cfg["column"])] = _safe_numeric(
        merged[str(target_cfg["column"])]
    ).fillna(0).astype(int)
    merged = merged.sort_values(["event_timestamp"]).reset_index(drop=True)

    merged, _serving_columns = add_donor_sequence_features(merged, config)

    training_config = dict(config.get("training", {}))
    training_config["task_type"] = "classification"

    def post_split_fn(frame: pd.DataFrame) -> pd.DataFrame:
        static_cols = [str(c) for c in feature_cfg.get("static_features", [])]
        seq_cols = [str(c) for c in feature_cfg.get("sequence_features", [])]
        serving_cols = static_cols + seq_cols + [
            str(c) for c in feature_cfg["group_columns"]
        ]
        if "event_timestamp" not in serving_cols and str(feature_cfg["timestamp_column"]) not in serving_cols:
            serving_cols.append(str(feature_cfg["timestamp_column"]))
        if str(target_cfg["column"]) in serving_cols:
            serving_cols.remove(str(target_cfg["column"]))
        available = [c for c in serving_cols if c in frame.columns]
        for col in available:
            if pd.api.types.is_numeric_dtype(frame[col]):
                frame[col] = _safe_numeric(frame[col])
        return frame

    return TaskDataset(
        task_name=str(config["id"]),
        model_name=str(config["registered_model_name"]),
        experiment_name="donor_contact_response",
        task_type="classification",
        frame=merged,
        target_column=str(target_cfg["column"]),
        timestamp_column=str(feature_cfg["timestamp_column"]),
        feature_service="donor_short_horizon_probability_service",
        entity_keys=tuple(feature_cfg["group_columns"]),
        label_file=FEATURE_LABELS_DIRECTORY / "quebec_donor_labels.parquet",
        feature_file=FEATURE_LABELS_DIRECTORY / "quebec_donor_features.parquet",
        description=str(config["description"]),
        hard_mask_builder=lambda df: pd.Series(False, index=df.index),
        prediction_defaults=dict(config.get("prediction_defaults", {})),
        training_config=training_config,
        post_split_feature_fn=post_split_fn,
    )


if __name__ == "__main__":
    result = train_and_log(build_contact_response_task)
    print(f"Contact response model trained: {result['selected_candidate']}")
    print(f"  Overall AUC: {result['overall_metrics'].get('roc_auc', 'N/A')}")
    print(f"  Overall Brier: {result['overall_metrics'].get('brier_score', 'N/A')}")
    print("Done.")
