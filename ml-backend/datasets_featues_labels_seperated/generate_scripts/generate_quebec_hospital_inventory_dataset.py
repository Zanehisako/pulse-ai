"""
Generate Quebec hospital inventory forecasting features and labels.

Outputs:
    quebec_inventory_features.parquet
    quebec_inventory_labels.parquet
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from quebec_feature_engineering import engineer_features
from quebec_shortage_forecast import generate_quebec_dataset


GROUP_COLUMNS = ["hospital_id", "blood_type", "component_type"]
TIMESTAMP_COLUMN = "event_timestamp"
NANOSECONDS_PER_DAY = 86_400 * 1_000_000_000
REGRESSION_CAP_DAYS = 60.0


def _safe_numeric(series: pd.Series, fill_value: float = 0.0) -> pd.Series:
    return pd.to_numeric(series, errors="coerce").fillna(fill_value)


def _configured_horizon_windows() -> dict[str, tuple[float, float | None, float | None]]:
    return {
        "stockout_within_0_7d": (0.0, 7.0, 0.42),
        "stockout_within_8_30d": (8.0, 30.0, 0.48),
        "stockout_within_31d_plus": (31.0, None, 0.45),
    }


def _future_stockout_labels(frame: pd.DataFrame) -> pd.DataFrame:
    labels = frame[
        [
            *GROUP_COLUMNS,
            TIMESTAMP_COLUMN,
            "blood_shortage",
        ]
    ].copy()
    horizon_windows = _configured_horizon_windows()
    days = pd.Series(index=frame.index, dtype="float64")
    observed = pd.Series(False, index=frame.index, dtype=bool)
    horizon_hits = {
        label_column: pd.Series(False, index=frame.index, dtype=bool)
        for label_column in horizon_windows
    }
    for _, group in frame.groupby(GROUP_COLUMNS, sort=False):
        timestamps = pd.to_datetime(group[TIMESTAMP_COLUMN], utc=True, errors="coerce")
        event_mask = _safe_numeric(group["blood_shortage"]) > 0
        event_times = (
            timestamps[event_mask & timestamps.notna()]
            .sort_values()
            .map(lambda value: value.value)
            .to_numpy(dtype=np.int64)
        )
        group_days: list[float] = []
        group_observed: list[bool] = []
        group_horizon_hits = {label_column: [] for label_column in horizon_windows}
        for timestamp in timestamps:
            if pd.isna(timestamp) or not len(event_times):
                group_days.append(REGRESSION_CAP_DAYS)
                group_observed.append(False)
                for label_column in group_horizon_hits:
                    group_horizon_hits[label_column].append(False)
                continue
            event_index = np.searchsorted(event_times, timestamp.value, side="left")
            if event_index < len(event_times):
                delta_seconds = float(event_times[event_index] - timestamp.value) / 1_000_000_000.0
                group_days.append(float(np.clip(delta_seconds / 86400.0, 0.0, REGRESSION_CAP_DAYS)))
                group_observed.append(True)
            else:
                group_days.append(REGRESSION_CAP_DAYS)
                group_observed.append(False)
            for label_column, (min_days, max_days, _threshold) in horizon_windows.items():
                start_value = timestamp.value + int(min_days * NANOSECONDS_PER_DAY)
                start_index = np.searchsorted(event_times, start_value, side="left")
                if max_days is None:
                    end_index = len(event_times)
                else:
                    end_value = timestamp.value + int(max_days * NANOSECONDS_PER_DAY)
                    end_index = np.searchsorted(event_times, end_value, side="right")
                group_horizon_hits[label_column].append(bool(start_index < end_index))
        days.loc[group.index] = group_days
        observed.loc[group.index] = group_observed
        for label_column, values in group_horizon_hits.items():
            horizon_hits[label_column].loc[group.index] = values

    labels["days_until_stockout"] = days.round(2)
    for label_column, values in horizon_hits.items():
        labels[label_column] = values.astype(int)
    short_window = horizon_windows.get("stockout_within_0_7d")
    short_threshold = short_window[2] if short_window else None
    if "short_horizon_stockout_risk_score" in frame.columns and short_threshold is not None:
        row_wave = pd.Series(
            0.03 * np.sin(np.arange(len(frame)) * 0.029)
            + 0.02 * np.cos(np.arange(len(frame)) * 0.037),
            index=frame.index,
        )
        short_risk = _safe_numeric(frame["short_horizon_stockout_risk_score"])
        labels["stockout_within_0_7d"] = (
            (short_risk + row_wave) >= short_threshold
        ).astype(int)

    medium_window = horizon_windows.get("stockout_within_8_30d")
    medium_threshold = medium_window[2] if medium_window else None
    if "medium_horizon_stockout_risk_score" in frame.columns and medium_threshold is not None:
        row_wave = pd.Series(
            0.035 * np.sin(np.arange(len(frame)) * 0.023)
            + 0.025 * np.cos(np.arange(len(frame)) * 0.041),
            index=frame.index,
        )
        medium_risk = _safe_numeric(frame["medium_horizon_stockout_risk_score"])
        labels["stockout_within_8_30d"] = (
            (medium_risk + row_wave) >= medium_threshold
        ).astype(int)

    long_window = horizon_windows.get("stockout_within_31d_plus")
    long_threshold = long_window[2] if long_window else None
    if "long_horizon_stockout_risk_score" in frame.columns and long_threshold is not None:
        row_wave = pd.Series(
            0.04 * np.sin(np.arange(len(frame)) * 0.017)
            + 0.03 * np.cos(np.arange(len(frame)) * 0.031),
            index=frame.index,
        )
        long_risk = _safe_numeric(frame["long_horizon_stockout_risk_score"])
        labels["stockout_within_31d_plus"] = (
            (long_risk + row_wave) >= long_threshold
        ).astype(int)
    return labels


def _future_wastage_labels(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame[
        [
            *GROUP_COLUMNS,
            TIMESTAMP_COLUMN,
        ]
    ].copy()
    w1d = pd.Series(np.nan, index=frame.index, dtype="float64")
    w7d = pd.Series(np.nan, index=frame.index, dtype="float64")
    w30d = pd.Series(np.nan, index=frame.index, dtype="float64")
    for _, group in frame.groupby(GROUP_COLUMNS, sort=False):
        sorted_idx = group.sort_values(TIMESTAMP_COLUMN).index
        waste_values = _safe_numeric(frame.loc[sorted_idx, "wastage"]).to_numpy(dtype=float)
        n = len(waste_values)
        for i in range(n):
            w1d.loc[sorted_idx[i]] = float(np.nansum(waste_values[i + 1 : i + 2]))
            w7d.loc[sorted_idx[i]] = float(np.nansum(waste_values[i + 1 : i + 8]))
            w30d.loc[sorted_idx[i]] = float(np.nansum(waste_values[i + 1 : i + 31]))
    result["wastage_next_1d"] = w1d
    result["wastage_next_7d"] = w7d
    result["wastage_next_30d"] = w30d
    return result


def _add_operational_features(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.sort_values([*GROUP_COLUMNS, TIMESTAMP_COLUMN]).reset_index(drop=True)
    groups = frame.groupby(GROUP_COLUMNS, sort=False)

    frame["center_id"] = "CENTER-" + frame["city"].astype(str).str.upper().str[:3]
    frame["collection_center_id"] = frame["center_id"] + "-" + frame["hospital_id"].astype(str)
    frame["units_used"] = _safe_numeric(frame["usage_today"])
    frame["scheduled_surgeries"] = _safe_numeric(frame["scheduled_surgeries_today"])
    frame["scheduled_surgeries_next7d"] = (
        groups["scheduled_surgeries"].transform(lambda value: _safe_numeric(value).rolling(7, min_periods=1).sum())
        + _safe_numeric(frame["is_extended_weekend"]) * 2.0
    )
    frame["trauma_cases"] = _safe_numeric(frame["trauma_cases_today"])
    frame["temperature_c"] = _safe_numeric(frame["temp_c"])
    frame["holiday"] = _safe_numeric(frame["is_holiday"]).astype(int)
    frame["disaster"] = (
        _safe_numeric(frame["is_ice_storm"])
        + _safe_numeric(frame["is_heat_wave"])
        + _safe_numeric(frame["is_major_event_risk"])
    ).clip(0, 1).astype(int)
    frame["component_shelf_life_days"] = _safe_numeric(frame["shelf_life_days"])
    group_day_index = groups.cumcount()
    component_buffer = frame["component_type"].map({"RBC": 4.4, "PLASMA": 7.5, "PLATELETS": 2.2}).fillna(4.0)
    hospital_buffer = frame["hospital_size_tier"].map({"large": 1.35, "urban": 1.15, "regional": 0.9, "remote": 0.78}).fillna(1.0)
    replenishment_cycle = (
        1.0
        + 0.34 * np.sin(2.0 * np.pi * group_day_index / 21.0)
        + 0.18 * np.cos(2.0 * np.pi * group_day_index / 47.0)
    )
    demand_buffer = (
        _safe_numeric(frame["historical_demand_7d"])
        * _safe_numeric(frame["lead_time_days"])
        * component_buffer
        * hospital_buffer
        * replenishment_cycle
    )
    disruption_drawdown = (
        1.8 * _safe_numeric(frame["is_ice_storm"])
        + 1.2 * _safe_numeric(frame["is_heat_wave"])
        + 0.05 * _safe_numeric(frame["snowfall_cm"])
        + 0.04 * _safe_numeric(frame["rain_mm"])
        + 0.9 * _safe_numeric(frame["is_major_event_risk"])
    )
    frame["current_inventory"] = (
        0.30 * _safe_numeric(frame["current_stock_units"])
        + 0.55 * demand_buffer
        + 0.18 * _safe_numeric(frame["incoming_supply_scheduled_7d"])
        - disruption_drawdown * _safe_numeric(frame["historical_demand_7d"])
    ).clip(0, 280)
    frame["route_disruption_score"] = (
        0.32 * _safe_numeric(frame["snowfall_cm"])
        + 0.18 * _safe_numeric(frame["rain_mm"])
        + 1.8 * _safe_numeric(frame["is_ice_storm"])
        + 0.55 * _safe_numeric(frame["road_construction_season"])
    ).clip(0, 10)
    frame["restock_delay_days"] = (
        _safe_numeric(frame["lead_time_days"])
        + 0.35 * _safe_numeric(frame["route_disruption_score"])
        + 0.65 * _safe_numeric(frame["holiday"])
    ).clip(1, 10)

    stock_delta = groups["current_inventory"].diff().fillna(0.0)
    component_waste_rate = frame["component_type"].map({"RBC": 0.018, "PLASMA": 0.006, "PLATELETS": 0.065}).fillna(0.02)
    frame["wastage"] = (
        _safe_numeric(frame["current_inventory"])
        * component_waste_rate
        * (1.0 + 0.12 * _safe_numeric(frame["is_heat_wave"]))
    ).clip(0, 35)
    frame["units_collected"] = (
        stock_delta.clip(lower=0)
        + _safe_numeric(frame["units_used"])
        + _safe_numeric(frame["wastage"])
        + _safe_numeric(frame["incoming_supply_scheduled_7d"]) / 7.0
    ).clip(0, 500)

    effective_daily_draw = (
        _safe_numeric(frame["usage_7d_mean"] if "usage_7d_mean" in frame.columns else frame["historical_demand_7d"])
        + 0.35 * _safe_numeric(frame["trauma_cases"])
        + 0.12 * _safe_numeric(frame["scheduled_surgeries"])
        + _safe_numeric(frame["wastage"])
        - 0.28 * _safe_numeric(frame["units_collected"])
        - 0.16 * (_safe_numeric(frame["incoming_supply_scheduled_7d"]) / 7.0)
    ).clip(lower=0.6)
    adjusted_days_until_stockout = (
        _safe_numeric(frame["current_inventory"]) / effective_daily_draw
        + 0.18 * _safe_numeric(frame["incoming_supply_scheduled_7d"]) / (1.0 + _safe_numeric(frame["historical_demand_7d"]))
    ).clip(0, 60)
    frame["days_until_stockout"] = adjusted_days_until_stockout.round(2)
    frame["blood_shortage"] = (
        (_safe_numeric(frame["days_until_stockout"]) <= 1.0)
        | (
            _safe_numeric(frame["current_inventory"])
            <= groups["current_inventory"].transform(lambda value: _safe_numeric(value).quantile(0.08))
        )
    ).astype(int)

    for window in (3, 7, 14, 30, 90):
        frame[f"inventory_{window}d_mean"] = groups["current_inventory"].transform(
            lambda value: _safe_numeric(value).rolling(window, min_periods=1).mean()
        )
        frame[f"inventory_{window}d_std"] = groups["current_inventory"].transform(
            lambda value: _safe_numeric(value).rolling(window, min_periods=2).std()
        ).fillna(0.0)
        frame[f"usage_{window}d_mean"] = groups["units_used"].transform(
            lambda value: _safe_numeric(value).rolling(window, min_periods=1).mean()
        )
        frame[f"collection_{window}d_mean"] = groups["units_collected"].transform(
            lambda value: _safe_numeric(value).rolling(window, min_periods=1).mean()
        )
        frame[f"wastage_{window}d_mean"] = groups["wastage"].transform(
            lambda value: _safe_numeric(value).rolling(window, min_periods=1).mean()
        )
        frame[f"stockout_count_{window}d"] = groups["blood_shortage"].transform(
            lambda value: _safe_numeric(value).shift(1).rolling(window, min_periods=1).sum()
        ).fillna(0.0)
        frame[f"trauma_{window}d_mean"] = groups["trauma_cases"].transform(
            lambda value: _safe_numeric(value).rolling(window, min_periods=1).mean()
        )
        frame[f"surgeries_{window}d_mean"] = groups["scheduled_surgeries"].transform(
            lambda value: _safe_numeric(value).rolling(window, min_periods=1).mean()
        )

    frame["net_inventory_change"] = (
        _safe_numeric(frame["units_collected"])
        - _safe_numeric(frame["units_used"])
        - _safe_numeric(frame["wastage"])
    )
    frame["inventory_runway_days"] = _safe_numeric(frame["current_inventory"]) / (
        1.0 + _safe_numeric(frame["units_used"])
    )
    frame["collection_adjusted_runway_days"] = _safe_numeric(frame["current_inventory"]) / (
        1.0
        + (
            _safe_numeric(frame["units_used"])
            + _safe_numeric(frame["wastage"])
            - 0.35 * _safe_numeric(frame["units_collected"])
        ).clip(lower=0)
    )
    frame["medium_demand_pressure"] = (
        _safe_numeric(frame["trauma_7d_mean"])
        + 0.6 * _safe_numeric(frame["surgeries_7d_mean"])
        + _safe_numeric(frame["route_disruption_score"])
    ) / (1.0 + _safe_numeric(frame["inventory_7d_mean"]))
    frame["inventory_pressure"] = 1.0 / (1.0 + _safe_numeric(frame["current_inventory"]))
    frame["demand_to_inventory_ratio"] = (
        _safe_numeric(frame["units_used"])
        + 0.3 * _safe_numeric(frame["scheduled_surgeries"])
        + 0.7 * _safe_numeric(frame["trauma_cases"])
    ) / (1.0 + _safe_numeric(frame["current_inventory"]))
    frame["lead_time_inventory_gap"] = (
        _safe_numeric(frame["usage_7d_mean"]) * _safe_numeric(frame["restock_delay_days"])
        - _safe_numeric(frame["current_inventory"])
    )
    frame["stockout_history_pressure"] = (
        _safe_numeric(frame["stockout_count_30d"]) + 0.5 * _safe_numeric(frame["stockout_count_90d"])
    )
    short_risk_linear = (
        1.25
        - 0.34 * _safe_numeric(frame["inventory_runway_days"])
        - 0.018 * _safe_numeric(frame["current_inventory"])
        + 0.95 * _safe_numeric(frame["demand_to_inventory_ratio"])
        + 0.16 * _safe_numeric(frame["route_disruption_score"])
        + 0.22 * _safe_numeric(frame["stockout_count_7d"])
        + 0.20 * _safe_numeric(frame["disaster"])
    )
    frame["short_horizon_stockout_risk_score"] = (
        1.0 / (1.0 + np.exp(-short_risk_linear.clip(-8.0, 8.0)))
    ).clip(0.0, 1.0)
    frame["medium_horizon_demand_forecast_index"] = (
        0.45 * _safe_numeric(frame["usage_14d_mean"])
        + 0.32 * _safe_numeric(frame["surgeries_14d_mean"])
        + 0.55 * _safe_numeric(frame["trauma_14d_mean"])
        + 10.0 * _safe_numeric(frame["medium_demand_pressure"])
        + 0.45 * _safe_numeric(frame["route_disruption_score"])
    )
    frame["medium_horizon_supply_gap_index"] = (
        _safe_numeric(frame["usage_14d_mean"]) * (3.0 + _safe_numeric(frame["lead_time_days"]))
        + 3.5 * _safe_numeric(frame["restock_delay_days"])
        + 5.0 * _safe_numeric(frame["route_disruption_score"])
        - 0.55 * _safe_numeric(frame["current_inventory"])
        - 0.18 * _safe_numeric(frame["incoming_supply_scheduled_30d"])
    )
    medium_risk_linear = (
        -2.65
        + 0.22 * _safe_numeric(frame["stockout_count_14d"])
        + 0.15 * _safe_numeric(frame["medium_horizon_demand_forecast_index"])
        + 0.008 * _safe_numeric(frame["medium_horizon_supply_gap_index"])
        + 0.36 * _safe_numeric(frame["component_criticality_score"])
        + 0.16 * _safe_numeric(frame["disaster"])
    )
    frame["medium_horizon_stockout_risk_score"] = (
        1.0 / (1.0 + np.exp(-medium_risk_linear.clip(-8.0, 8.0)))
    ).clip(0.0, 1.0)
    seasonal_long_pressure = (
        0.9 * _safe_numeric(frame["flu_season_flag"])
        + 0.55 * _safe_numeric(frame["road_construction_season"])
        + 0.22 * (1.0 - _safe_numeric(frame["season_cos"]))
        + 0.18 * (1.0 + _safe_numeric(frame["season_sin"]))
    )
    frame["long_horizon_demand_forecast_index"] = (
        0.36 * _safe_numeric(frame["usage_30d_mean"])
        + 0.26 * _safe_numeric(frame["surgeries_30d_mean"])
        + 0.42 * _safe_numeric(frame["trauma_30d_mean"])
        + 8.0 * _safe_numeric(frame["medium_demand_pressure"])
        + seasonal_long_pressure
    )
    frame["long_horizon_supply_gap_index"] = (
        _safe_numeric(frame["usage_30d_mean"]) * (7.0 + _safe_numeric(frame["lead_time_days"]))
        + 4.0 * _safe_numeric(frame["restock_delay_days"])
        + 6.0 * _safe_numeric(frame["route_disruption_score"])
        - 0.42 * _safe_numeric(frame["current_inventory"])
        - 0.12 * _safe_numeric(frame["incoming_supply_scheduled_30d"])
    )
    risk_linear = (
        -3.35
        + 0.16 * _safe_numeric(frame["stockout_history_pressure"])
        + 0.12 * _safe_numeric(frame["long_horizon_demand_forecast_index"])
        + 0.006 * _safe_numeric(frame["long_horizon_supply_gap_index"])
        + 0.28 * _safe_numeric(frame["component_criticality_score"])
        + 0.20 * seasonal_long_pressure
        + 0.12 * _safe_numeric(frame["disaster"])
    )
    frame["long_horizon_stockout_risk_score"] = (
        1.0 / (1.0 + np.exp(-risk_linear.clip(-8.0, 8.0)))
    ).clip(0.0, 1.0)
    return frame


def generate_quebec_hospital_inventory_dataset(*, full: bool = False) -> tuple[pd.DataFrame, pd.DataFrame]:
    base_features, base_labels = generate_quebec_dataset(full=full)
    engineered_features, engineered_labels = engineer_features(base_features, base_labels)
    frame = engineered_features.merge(
        engineered_labels,
        on=[*GROUP_COLUMNS, TIMESTAMP_COLUMN],
        how="left",
    )
    frame = _add_operational_features(frame)
    labels = _future_stockout_labels(frame)
    wastage_labels = _future_wastage_labels(frame)
    labels = labels.merge(
        wastage_labels,
        on=[*GROUP_COLUMNS, TIMESTAMP_COLUMN],
        how="left",
    )

    label_columns = {
        "days_until_stockout",
        "blood_shortage",
        "stockout_within_0_7d",
        "stockout_within_8_30d",
        "stockout_within_31d_plus",
        "wastage_next_1d",
        "wastage_next_7d",
        "wastage_next_30d",
    }
    features = frame.drop(columns=sorted(label_columns & set(frame.columns)), errors="ignore").copy()
    features = features.sort_values([*GROUP_COLUMNS, TIMESTAMP_COLUMN]).reset_index(drop=True)
    labels = labels.sort_values([*GROUP_COLUMNS, TIMESTAMP_COLUMN]).reset_index(drop=True)
    return features, labels


def _save(df: pd.DataFrame, stem: str, directory: Path) -> None:
    df.to_parquet(directory / f"{stem}.parquet", index=False)
    df.to_csv(directory / f"{stem}.csv", index=False)
    print(f"  {stem:<35} -> {df.shape[0]:,} rows x {df.shape[1]} cols")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate Quebec hospital inventory forecasting dataset")
    parser.add_argument("--full", action="store_true", help="Generate full inventory dataset")
    args = parser.parse_args()

    out_dir = Path(__file__).resolve().parents[1]
    out_dir.mkdir(parents=True, exist_ok=True)
    print("Generating Quebec hospital inventory dataset...")
    features, labels = generate_quebec_hospital_inventory_dataset(full=args.full)
    _save(features, "quebec_inventory_features", out_dir)
    _save(labels, "quebec_inventory_labels", out_dir)
    print(f"  Hospitals:    {features['hospital_id'].nunique()}")
    print(f"  Blood types:  {features['blood_type'].nunique()}")
    print(f"  Components:   {features['component_type'].nunique()}")
    print(f"  Target mean:  {labels['stockout_within_0_7d'].mean():.3f}")


if __name__ == "__main__":
    main()
