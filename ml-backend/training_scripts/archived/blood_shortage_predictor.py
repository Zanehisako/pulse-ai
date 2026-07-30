from __future__ import annotations

import pandas as pd

from model_training_utils import (
    TaskDataset,
    _days_until_next_group_event,
    _quantile_mask,
    _safe_numeric,
    _save_parquet_pair,
    load_hospital_component_supply_frame,
    train_and_log,
)


SERVING_FEATURES = (
    "temperature",
    "rain_mm",
    "holiday",
    "disaster",
    "scheduled_surgeries",
    "trauma_cases",
    "current_inventory",
    "blood_type",
)


def build_hospital_shortage_task() -> TaskDataset:
    from model_training_utils import FEATURE_LABELS_DIRECTORY

    frame = load_hospital_component_supply_frame()
    frame["blood_shortage"] = _safe_numeric(frame["blood_shortage"]).fillna(0).astype(int)
    frame["days_until_stockout"] = _days_until_next_group_event(
        frame,
        group_column=["hospital_id", "blood_type"],
        timestamp_column="event_timestamp",
        event_column="blood_shortage",
    )
    frame["demand_pressure"] = (
        _safe_numeric(frame["trauma_cases"])
        + 0.6 * _safe_numeric(frame["scheduled_surgeries"])
    ) / (1.0 + _safe_numeric(frame["current_inventory"]))
    frame["weather_load"] = _safe_numeric(frame["rain_mm"])
    frame["shock_intensity"] = (
        _safe_numeric(frame["disaster"]) * 2.0
        + _safe_numeric(frame["holiday"])
    )
    frame["inventory_pressure"] = 1.0 / (1.0 + _safe_numeric(frame["current_inventory"]))
    frame["inventory_band"] = pd.cut(
        _safe_numeric(frame["current_inventory"]),
        bins=[-1, 10, 20, 35, 60, 200],
        labels=["critical", "low", "guarded", "healthy", "buffered"],
        include_lowest=True,
    ).astype(str)
    frame = frame.sort_values(["event_timestamp", "hospital_id", "blood_type"]).reset_index(drop=True)

    feature_path = FEATURE_LABELS_DIRECTORY / "hospital_shortage_features.parquet"
    label_path = FEATURE_LABELS_DIRECTORY / "hospital_shortage_labels.parquet"
    target_column = "days_until_stockout"
    _save_parquet_pair(frame, feature_path, label_path, target_column)

    def hard_mask(frame: pd.DataFrame) -> pd.Series:
        return (
            (_safe_numeric(frame["disaster"]) > 0)
            | (_safe_numeric(frame["holiday"]) > 0)
            | _quantile_mask(frame["current_inventory"], 0.25, "low")
            | _quantile_mask(frame["demand_pressure"], 0.75, "high")
            | _quantile_mask(frame["rain_mm"], 0.75, "high")
        )

    return TaskDataset(
        task_name="hospital_shortage_predictor",
        model_name="hospital_shortage_predictor",
        experiment_name="hospital_shortage_prediction",
        task_type="regression",
        frame=frame,
        target_column=target_column,
        timestamp_column="event_timestamp",
        feature_service="hospital_shortage_service",
        entity_keys=("hospital_id", "blood_type"),
        label_file=label_path,
        feature_file=feature_path,
        description=(
            "Component-level hospital stockout horizon regressor trained on "
            "hospital and blood-type supply streams to estimate days until stockout."
        ),
        hard_mask_builder=hard_mask,
        accuracy_tolerance=1.0,
        serving_feature_columns=SERVING_FEATURES,
    )


if __name__ == "__main__":
    payload = train_and_log(build_hospital_shortage_task)
    print(payload)
