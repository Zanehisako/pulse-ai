"""
train_component_supply_forecast.py

Probabilistic supply forecast using the standard regression pipeline.
Trains per-quantile models for supply prediction.
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
    _safe_numeric,
    train_and_log,
)
from pios_ml_backend.training_resource_profiles import apply_profile_to_model_config

_CONFIG_PATH = Path(__file__).resolve().parents[1] / "config" / "component_supply_forecast_model.json"


def _load_config() -> dict:
    return apply_profile_to_model_config(
        json.loads(_CONFIG_PATH.read_text(encoding="utf-8"))
    )


def build_supply_forecast_task() -> TaskDataset:
    config = _load_config()
    target_cfg = config["target"]
    feature_cfg = config["feature_engineering"]

    features = pd.read_parquet(
        FEATURE_LABELS_DIRECTORY / "quebec_inventory_features.parquet"
    )
    labels = pd.read_parquet(
        FEATURE_LABELS_DIRECTORY / "quebec_inventory_labels.parquet"
    )

    if str(target_cfg["column"]) in labels.columns:
        label_cols = [str(c) for c in feature_cfg["group_columns"]] + ["event_timestamp", str(target_cfg["column"])]
    else:
        label_cols = [str(c) for c in feature_cfg["group_columns"]] + ["event_timestamp"]
    merged = features.merge(
        labels[label_cols],
        on=[str(c) for c in feature_cfg["group_columns"]] + ["event_timestamp"],
        how="inner",
    )
    merged[str(target_cfg["column"])] = _safe_numeric(
        merged[str(target_cfg["column"])]
    ).fillna(0.0)
    merged = merged.sort_values(["event_timestamp"]).reset_index(drop=True)

    training_config = dict(config.get("training", {}))
    training_config["target_transform"] = "log1p"

    return TaskDataset(
        task_name=str(config["id"]),
        model_name=str(config["registered_model_name"]),
        experiment_name="component_supply_forecast",
        task_type="regression",
        frame=merged,
        target_column=str(target_cfg["column"]),
        timestamp_column=str(feature_cfg["timestamp_column"]),
        feature_service="quebec_arima_sarima_forecast_service",
        entity_keys=tuple(feature_cfg["group_columns"]),
        label_file=FEATURE_LABELS_DIRECTORY / "quebec_inventory_labels.parquet",
        feature_file=FEATURE_LABELS_DIRECTORY / "quebec_inventory_features.parquet",
        description=str(config["description"]),
        hard_mask_builder=lambda df: pd.Series(False, index=df.index),
        prediction_defaults=dict(config.get("prediction_defaults", {})),
        training_config=training_config,
    )


if __name__ == "__main__":
    result = train_and_log(build_supply_forecast_task)
    print(f"Supply forecast trained: {result['selected_candidate']}")
    print(f"  Overall RMSE: {result['overall_metrics'].get('root_mean_squared_error', 'N/A')}")
    print("Done.")
