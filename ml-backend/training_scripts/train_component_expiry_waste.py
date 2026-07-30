"""
train_component_expiry_waste.py

Wastage forecast training script using the standard regression pipeline.
Primary target: wastage_next_1d (shifted labels). Falls back to same-row
wastage as proxy if shifted labels are all NaN.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_TRAINING_SCRIPTS = Path(__file__).resolve().parent
if str(_TRAINING_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_TRAINING_SCRIPTS))

from model_training_utils import (
    FEATURE_LABELS_DIRECTORY,
    TaskDataset,
    _safe_numeric,
    train_and_log,
)
from pios_ml_backend.training_resource_profiles import apply_profile_to_model_config

_CONFIG_PATH = Path(__file__).resolve().parents[1] / "config" / "component_expiry_waste_model.json"


def _load_config() -> dict:
    return apply_profile_to_model_config(
        json.loads(_CONFIG_PATH.read_text(encoding="utf-8"))
    )


def build_expiry_waste_task() -> TaskDataset:
    config = _load_config()
    target_cfg = config["target"]
    feature_cfg = config["feature_engineering"]

    features = pd.read_parquet(
        FEATURE_LABELS_DIRECTORY / "quebec_inventory_features.parquet"
    )
    labels = pd.read_parquet(
        FEATURE_LABELS_DIRECTORY / "quebec_inventory_labels.parquet"
    )

    primary_target = str(target_cfg["column"])
    entity_keys = [str(c) for c in feature_cfg["group_columns"]]
    merge_on = entity_keys + ["event_timestamp"]

    if primary_target in labels.columns:
        label_cols = merge_on + [primary_target]
    else:
        label_cols = merge_on

    merged = features.merge(
        labels[label_cols],
        on=merge_on,
        how="inner",
    )

    if primary_target in merged.columns:
        merged[primary_target] = _safe_numeric(merged[primary_target]).fillna(0.0)
        all_nan = merged[primary_target].isna().all()
    else:
        all_nan = True

    if all_nan:
        fallback_targets = target_cfg.get("fallback_targets", [])
        if fallback_targets and str(fallback_targets[0]) in merged.columns:
            actual_target = str(fallback_targets[0])
            merged[actual_target] = _safe_numeric(merged[actual_target]).fillna(0.0)
        else:
            actual_target = primary_target
    else:
        actual_target = primary_target

    merged = merged.sort_values(["event_timestamp"]).reset_index(drop=True)

    training_config = dict(config.get("training", {}))
    training_config["target_transform"] = "log1p"

    return TaskDataset(
        task_name=str(config["id"]),
        model_name=str(config["registered_model_name"]),
        experiment_name="component_expiry_waste",
        task_type="regression",
        frame=merged,
        target_column=actual_target,
        timestamp_column=str(feature_cfg["timestamp_column"]),
        feature_service="quebec_arima_sarima_forecast_service",
        entity_keys=tuple(entity_keys),
        label_file=FEATURE_LABELS_DIRECTORY / "quebec_inventory_labels.parquet",
        feature_file=FEATURE_LABELS_DIRECTORY / "quebec_inventory_features.parquet",
        description=str(config["description"]),
        hard_mask_builder=lambda df: pd.Series(False, index=df.index),
        prediction_defaults=dict(config.get("prediction_defaults", {})),
        training_config=training_config,
    )


if __name__ == "__main__":
    result = train_and_log(build_expiry_waste_task)
    print(f"Expiry waste model trained: {result['selected_candidate']}")
    print(f"  Overall RMSE: {result['overall_metrics'].get('root_mean_squared_error', 'N/A')}")
    print("Done.")
