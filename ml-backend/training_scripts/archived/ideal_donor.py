from __future__ import annotations

import pandas as pd

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


def build_donor_propensity_task() -> TaskDataset:
    task_config = _load_training_config("donor_propensity_model.json")

    target_column = str(task_config.get("target", {}).get("column") or "donated_next_6m")
    if task_config.get("dataset_source"):
        merged, _source = load_configured_dataset_frame(
            str(task_config["dataset_source"]),
            label_columns=[target_column],
        )
        merged[target_column] = _safe_numeric(merged[target_column]).fillna(0).astype(int)
    else:
        merged = _build_donor_features_base().copy()
        merged["donor_momentum"] = (
            _safe_numeric(merged["snapshot_frequency_365"])
            + _safe_numeric(merged["donation_velocity"]) * 1.5
            + _safe_numeric(merged["snapshot_overdue_days"]) / 180.0
        )
        latent_score = (
            0.04 * _safe_numeric(merged["donation_propensity_score"])
            + _safe_numeric(merged["response_readiness_index"])
            + 0.25 * _safe_numeric(merged["donor_momentum"])
            + 0.50 * (_safe_numeric(merged["eligible_to_donate"]) > 0).astype(float)
            + 0.40 * (_safe_numeric(merged["is_rare_type"]) > 0).astype(float)
            - 0.20 * (_safe_numeric(merged["travel_time_min"]) > 35).astype(float)
        )
        merged[target_column] = (latent_score >= 3.0).astype(int)
    drop_columns = [
        "snapshot_days_to_next_donation",
        "snapshot_y_donate_30d",
        "snapshot_typical_interval_hidden",
        "donated_next_6m",
        "next_6m_donation_count",
    ]
    merged = merged.drop(
        columns=[column for column in drop_columns if column != target_column],
        errors="ignore",
    )
    if target_column not in merged.columns:
        merged[target_column] = (latent_score >= 3.0).astype(int)
    merged = merged.sort_values(["event_timestamp", "donor_id"]).reset_index(drop=True)

    feature_path = FEATURE_LABELS_DIRECTORY / "donor_propensity_features.parquet"
    label_path = FEATURE_LABELS_DIRECTORY / "donor_propensity_labels.parquet"
    _save_parquet_pair(merged, feature_path, label_path, target_column)

    def hard_mask(frame: pd.DataFrame) -> pd.Series:
        return (
            (_safe_numeric(frame["is_rare_type"]) > 0)
            | (_safe_numeric(frame["eligible_to_donate"]) <= 0)
            | _quantile_mask(frame["travel_burden_score"], 0.75, "high")
            | _quantile_mask(frame["bmi_margin"], 0.75, "high")
            | _quantile_mask(frame["eligibility_buffer"], 0.25, "low")
        )

    return TaskDataset(
        task_name="donor_propensity_model",
        model_name="donor_propensity_model",
        experiment_name="Donor_Propensity_Prediction",
        task_type="classification",
        frame=merged,
        target_column=target_column,
        timestamp_column="event_timestamp",
        feature_service="donor_propensity_service",
        entity_keys=("donor_id",),
        label_file=label_path,
        feature_file=feature_path,
        description=(
            "Donor propensity model rebuilt from registry, logistics, and "
            "donor snapshot tables with leakage-pruned out-of-time features."
        ),
        hard_mask_builder=hard_mask,
    )


if __name__ == "__main__":
    payload = train_and_log(build_donor_propensity_task)
    print(payload)
