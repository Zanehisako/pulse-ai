from __future__ import annotations

import numpy as np
import pandas as pd

from model_training_utils import _safe_numeric


def _as_int_list(values: list[object]) -> list[int]:
    return sorted({int(value) for value in values})


def _lag_column(feature: str, lag_step: int) -> str:
    return feature if lag_step == 0 else f"{feature}_lag_{lag_step}"


def add_donor_sequence_features(
    frame: pd.DataFrame,
    config: dict[str, object],
) -> tuple[pd.DataFrame, tuple[str, ...]]:
    feature_config = config["feature_engineering"]
    assert isinstance(feature_config, dict)
    group_columns = [str(value) for value in feature_config["group_columns"]]
    sequence_features = [str(value) for value in feature_config["sequence_features"]]
    lag_steps = _as_int_list(list(feature_config["lag_steps"]))
    frame = frame.sort_values([*group_columns, str(feature_config["timestamp_column"])]).reset_index(drop=True)
    groups = frame.groupby(group_columns, sort=False)

    for window in feature_config.get("rolling_windows", []):
        window_size = int(window)
        frame[f"readiness_{window_size}s_mean"] = groups["readiness_score"].transform(
            lambda series: _safe_numeric(series).rolling(window_size, min_periods=1).mean()
        )
        frame[f"recency_{window_size}s_mean"] = groups["recency_days"].transform(
            lambda series: _safe_numeric(series).rolling(window_size, min_periods=1).mean()
        )
        frame[f"travel_burden_{window_size}s_mean"] = groups["travel_burden_score"].transform(
            lambda series: _safe_numeric(series).rolling(window_size, min_periods=1).mean()
        )

    frame["donor_readiness_trend"] = groups["readiness_score"].diff().fillna(0.0)
    frame["donor_recency_trend"] = groups["recency_days"].diff().fillna(0.0)
    frame["donor_engagement_pressure"] = (
        _safe_numeric(frame["response_readiness_index"])
        + 0.45 * _safe_numeric(frame["donor_momentum"])
        + 0.25 * _safe_numeric(frame["emergency_outreach_priority"])
    ) / (1.0 + _safe_numeric(frame["travel_burden_score"]))
    frame["donor_route_adjusted_readiness"] = (
        _safe_numeric(frame["readiness_score"])
        + _safe_numeric(frame["mobility_score"])
        + _safe_numeric(frame["quebec_access_score"])
    ) / (1.0 + _safe_numeric(frame["winter_route_risk_score"]))

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
        frame = frame.drop(columns=[c for c in lagged_columns if c in frame.columns])
        frame = pd.concat([frame, pd.DataFrame(lagged_columns, index=frame.index)], axis=1).copy()

    serving_columns = tuple(
        _lag_column(feature, lag_step)
        for lag_step in sorted(lag_steps, reverse=True)
        for feature in sequence_features
    )
    return frame, serving_columns




