from __future__ import annotations

import numpy as np
import pandas as pd

<<<<<<< HEAD:ml-backend/training_scripts/old_training_scripts/stockout_forecast_tasks.py

from model_training_utils import _safe_numeric
=======
from model_training_utils import (
    TaskDataset,
    _days_until_next_group_event,
    _quantile_mask,
    _safe_numeric,
    _save_parquet_pair,
    load_hospital_component_supply_frame,
    train_and_log,
)
>>>>>>> dfb0d5e45f4e10cd5dcee1da618b83e2f86f5a03:ml-backend/training_scripts/train_stockout_lstm.py


def build_stockout_task() -> TaskDataset:
    from model_training_utils import FEATURE_LABELS_DIRECTORY

    frame = load_hospital_component_supply_frame()
    frame = frame.sort_values(["hospital_id", "blood_type", "event_timestamp"]).reset_index(drop=True)
    frame["blood_shortage"] = _safe_numeric(frame["blood_shortage"]).fillna(0).astype(int)
    frame["days_until_stockout"] = _days_until_next_group_event(
        frame,
        group_column=["hospital_id", "blood_type"],
        timestamp_column="event_timestamp",
        event_column="blood_shortage",
    )

<<<<<<< HEAD:ml-backend/training_scripts/old_training_scripts/stockout_forecast_tasks.py
def _lag_column(feature: str, lag_step: int) -> str:
    return feature if lag_step == 0 else f"{feature}_lag_{lag_step}"


def add_stockout_sequence_features(
    frame: pd.DataFrame,
    config: dict[str, object],
) -> tuple[pd.DataFrame, tuple[str, ...]]:
    feature_config = config["feature_engineering"]
    assert isinstance(feature_config, dict)
    group_columns = [str(value) for value in feature_config["group_columns"]]
    sequence_features = [str(value) for value in feature_config["sequence_features"]]
    lag_steps = _as_int_list(list(feature_config["lag_steps"]))

    groups = frame.groupby(group_columns, sort=False)
    for window in feature_config.get("rolling_windows", []):
        window_size = int(window)
        frame[f"inventory_{window_size}d_mean"] = groups[
            "current_inventory"
        ].transform(lambda series: _safe_numeric(series).rolling(window_size, min_periods=1).mean())
        frame[f"inventory_{window_size}d_std"] = groups[
            "current_inventory"
        ].transform(
            lambda series: _safe_numeric(series).rolling(window_size, min_periods=2).std()
        ).fillna(0.0)
        frame[f"past_shortage_{window_size}d_mean"] = groups[
            "blood_shortage"
        ].transform(
            lambda series: (
                _safe_numeric(series)
                .shift(1)
                .rolling(window_size, min_periods=1)
                .mean()
            )
        ).fillna(0.0)

=======
    groups = frame.groupby(["hospital_id", "blood_type"], sort=False)
    frame["inventory_3d_mean"] = groups[
        "current_inventory"
    ].transform(lambda series: _safe_numeric(series).rolling(window=3, min_periods=1).mean())
    frame["inventory_7d_mean"] = groups[
        "current_inventory"
    ].transform(lambda series: _safe_numeric(series).rolling(window=7, min_periods=1).mean())
    frame["inventory_7d_std"] = groups[
        "current_inventory"
    ].transform(
        lambda series: _safe_numeric(series).rolling(window=7, min_periods=2).std()
    ).fillna(0.0)
>>>>>>> dfb0d5e45f4e10cd5dcee1da618b83e2f86f5a03:ml-backend/training_scripts/train_stockout_lstm.py
    frame["trauma_3d_mean"] = groups["trauma_cases"].transform(
        lambda series: _safe_numeric(series).rolling(window=3, min_periods=1).mean()
    )
    frame["trauma_7d_mean"] = groups["trauma_cases"].transform(
        lambda series: _safe_numeric(series).rolling(window=7, min_periods=1).mean()
    )
    frame["surgeries_3d_mean"] = groups[
        "scheduled_surgeries"
    ].transform(lambda series: _safe_numeric(series).rolling(window=3, min_periods=1).mean())
    frame["surgeries_7d_mean"] = groups[
        "scheduled_surgeries"
    ].transform(lambda series: _safe_numeric(series).rolling(window=7, min_periods=1).mean())
    frame["shortage_7d_mean"] = groups["blood_shortage"].transform(
        lambda series: _safe_numeric(series).rolling(window=7, min_periods=1).mean()
    )
    frame["shortage_14d_mean"] = groups["blood_shortage"].transform(
        lambda series: _safe_numeric(series).rolling(window=14, min_periods=1).mean()
    )

    timestamp = pd.to_datetime(frame["event_timestamp"], utc=True, errors="coerce")
    frame["month"] = timestamp.dt.month.fillna(1).astype(int)
    frame["dow"] = timestamp.dt.dayofweek.fillna(0).astype(int)
    frame["weekend"] = (frame["dow"] >= 5).astype(int)
    frame["season_sin"] = np.sin(2.0 * np.pi * _safe_numeric(frame["month"]) / 12.0)
    frame["season_cos"] = np.cos(2.0 * np.pi * _safe_numeric(frame["month"]) / 12.0)
    frame["demand_pressure"] = (
        _safe_numeric(frame["trauma_cases"])
        + 0.6 * _safe_numeric(frame["scheduled_surgeries"])
    ) / (1.0 + _safe_numeric(frame["current_inventory"]))
    frame["medium_demand_pressure"] = (
        _safe_numeric(frame["trauma_7d_mean"])
        + 0.6 * _safe_numeric(frame["surgeries_7d_mean"])
    ) / (1.0 + _safe_numeric(frame["inventory_7d_mean"]))
    frame["weather_load"] = _safe_numeric(frame["rain_mm"])
    frame["shock_intensity"] = (
        _safe_numeric(frame["disaster"]) * 2.0
        + _safe_numeric(frame["holiday"])
    )
    frame["inventory_pressure"] = 1.0 / (1.0 + _safe_numeric(frame["current_inventory"]))
    frame["inventory_trend_7d"] = groups[
        "current_inventory"
    ].diff(periods=7).fillna(0.0)
    frame["inventory_band"] = pd.cut(
        _safe_numeric(frame["current_inventory"]),
        bins=[-1, 10, 20, 35, 60, 200],
        labels=["critical", "low", "guarded", "healthy", "buffered"],
        include_lowest=True,
    ).astype(str)

<<<<<<< HEAD:ml-backend/training_scripts/old_training_scripts/stockout_forecast_tasks.py
    lagged_columns: dict[str, pd.Series] = {}
    for feature in sequence_features:
        if feature not in frame.columns:
            frame[feature] = 0.0
        numeric_feature = _safe_numeric(frame[feature])
        for lag_step in lag_steps:
            if lag_step == 0:
                frame[feature] = numeric_feature
                continue
            lagged = groups[feature].shift(lag_step)
            lagged_columns[_lag_column(feature, lag_step)] = (
                _safe_numeric(lagged, fill_value=np.nan)
                .fillna(numeric_feature)
                .fillna(0.0)
            )
    if lagged_columns:
        frame = frame.drop(
            columns=[c for c in lagged_columns if c in frame.columns]
        )
        frame = pd.concat(
            [frame, pd.DataFrame(lagged_columns, index=frame.index)],
            axis=1,
        ).copy()

    serving_columns = tuple(
        _lag_column(feature, lag_step)
        for lag_step in sorted(lag_steps, reverse=True)
        for feature in sequence_features
    )
    return frame, serving_columns


=======
    feature_path = FEATURE_LABELS_DIRECTORY / "stockout_sequence_features.parquet"
    label_path = FEATURE_LABELS_DIRECTORY / "stockout_sequence_labels.parquet"
    _save_parquet_pair(frame, feature_path, label_path, "days_until_stockout")

    def hard_mask(frame: pd.DataFrame) -> pd.Series:
        return (
            _quantile_mask(frame["current_inventory"], 0.25, "low")
            | _quantile_mask(frame["inventory_7d_mean"], 0.25, "low")
            | _quantile_mask(frame["medium_demand_pressure"], 0.75, "high")
            | _quantile_mask(frame["shortage_14d_mean"], 0.75, "high")
        )

    return TaskDataset(
        task_name="stockout_days_predictor",
        model_name="stockout_days_predictor",
        experiment_name="stockout_lstm_prediction",
        task_type="regression",
        frame=frame,
        target_column="days_until_stockout",
        timestamp_column="event_timestamp",
        feature_service="stockout_sequence_service",
        entity_keys=("hospital_id", "blood_type"),
        label_file=label_path,
        feature_file=feature_path,
        description=(
            "Medium-horizon component-level hospital stockout regressor trained on "
            "sequence-derived inventory, demand, weather, and shortage-history features."
        ),
        hard_mask_builder=hard_mask,
        accuracy_tolerance=2.0,
    )
>>>>>>> dfb0d5e45f4e10cd5dcee1da618b83e2f86f5a03:ml-backend/training_scripts/train_stockout_lstm.py




