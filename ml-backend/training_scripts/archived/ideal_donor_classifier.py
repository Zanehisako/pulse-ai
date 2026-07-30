from __future__ import annotations

import pandas as pd

from model_task_config import (
    compute_configured_probability_target,
    load_model_task_config,
)
from model_training_utils import (
    FEATURE_LABELS_DIRECTORY,
    TaskDataset,
    _build_donor_features_base,
    _quantile_mask,
    _safe_numeric,
    _save_parquet_pair,
    train_and_log,
)
from quebec_training_data import load_configured_dataset_frame


def build_ideal_donor_task() -> TaskDataset:
    task_config = load_model_task_config("ideal_donor_model.json")
    target_config = task_config["target"]
    target_column = target_config["column"]

    if task_config.get("dataset_source"):
        merged, _source = load_configured_dataset_frame(
            str(task_config["dataset_source"]),
            label_columns=[target_column],
        )
        merged[target_column] = _safe_numeric(merged[target_column]).clip(0.0, 1.0)
    else:
        merged = _build_donor_features_base().copy()
        merged[target_column] = compute_configured_probability_target(
            merged,
            target_config,
        )
    merged = merged.drop(
        columns=[
            "snapshot_days_to_next_donation",
            "snapshot_y_donate_30d",
            "snapshot_typical_interval_hidden",
            "donated_next_6m",
            "next_6m_donation_count",
        ],
        errors="ignore",
    )
    merged = merged.sort_values(["event_timestamp", "donor_id"]).reset_index(drop=True)

    feature_path = FEATURE_LABELS_DIRECTORY / "ideal_donor_classifier_features.parquet"
    label_path = FEATURE_LABELS_DIRECTORY / "ideal_donor_classifier_labels.parquet"
    _save_parquet_pair(merged, feature_path, label_path, target_column)

    def hard_mask(frame: pd.DataFrame) -> pd.Series:
        return (
            (_safe_numeric(frame["is_rare_type"]) > 0)
            | (_safe_numeric(frame["route_feasibility_flag"]) <= 0)
            | _quantile_mask(frame["travel_time_min"], 0.75, "high")
            | _quantile_mask(frame["response_readiness_index"], 0.25, "low")
            | (_safe_numeric(frame["eligible_to_donate"]) <= 0)
        )

    return TaskDataset(
        task_name="ideal_donor_classifier",
        model_name="ideal_donor_classifier",
        experiment_name="Ideal_Donor_Regression",
        task_type=task_config["model_type"],
        frame=merged,
        target_column=target_column,
        timestamp_column="event_timestamp",
        feature_service="ideal_donor_classifier_service",
        entity_keys=("donor_id",),
        label_file=label_path,
        feature_file=feature_path,
        description=task_config["description"],
        hard_mask_builder=hard_mask,
        accuracy_tolerance=float(target_config["accuracy_tolerance"]),
        min_accuracy=float(target_config["min_accuracy"]),
        prediction_defaults=task_config["prediction_defaults"],
    )


if __name__ == "__main__":
    payload = train_and_log(build_ideal_donor_task)
    print(payload)
