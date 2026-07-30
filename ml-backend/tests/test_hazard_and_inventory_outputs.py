from __future__ import annotations

import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal

ML_BACKEND = Path(__file__).resolve().parents[1]
if str(ML_BACKEND) not in sys.path:
    sys.path.insert(0, str(ML_BACKEND))

from pios_ml_backend.inventory.simulator import InventorySimulator
from pios_ml_backend.inventory.simulation_pyfunc import InventoryRiskSimulatorPyfunc
from pios_ml_backend.survival.discrete_time_hazard import (
    HazardIntervalDataset,
    build_hazard_interval_dataset,
    extract_horizon_probabilities,
)


def _legacy_hazard_interval_dataset(
    features: pd.DataFrame,
    labels: pd.DataFrame,
    *,
    entity_columns: list[str],
    timestamp_column: str,
    days_column: str,
    static_feature_columns: list[str],
    intervals: list[tuple[int, int]],
    censored_value: float | None = None,
) -> pd.DataFrame:
    merged = features.merge(
        labels[entity_columns + [timestamp_column, days_column]],
        on=entity_columns + [timestamp_column],
        how="inner",
    )
    merged = merged.sort_values(entity_columns + [timestamp_column]).reset_index(drop=True)
    days = pd.to_numeric(merged[days_column], errors="coerce")
    is_censored = (
        days == censored_value
        if censored_value is not None
        else pd.Series(False, index=days.index)
    )
    all_static = [c for c in static_feature_columns if c in merged.columns]
    excluded = set(entity_columns + [timestamp_column, days_column])
    all_dynamic = [
        c for c in merged.columns
        if c not in all_static and c not in excluded
    ]
    valid_dynamic = [
        c for c in all_dynamic
        if pd.api.types.is_numeric_dtype(merged[c])
    ]
    rows = []
    for _, source_row in merged.iterrows():
        source_days = float(days.loc[source_row.name])
        source_censored = bool(is_censored.loc[source_row.name])
        for interval_idx, (start, end) in enumerate(intervals):
            event_in_interval = 0
            if not source_censored and start < source_days <= end:
                event_in_interval = 1
            elif not source_censored and source_days <= start:
                event_in_interval = -1
            elif source_censored and source_days <= start:
                event_in_interval = -1
            row = {
                "interval_start": start,
                "interval_end": end,
                "interval_index": interval_idx,
                "interval_midpoint": (start + end) / 2.0,
            }
            row.update({column: source_row[column] for column in all_static})
            row.update({column: source_row[column] for column in valid_dynamic})
            row["event_in_interval"] = event_in_interval
            row["censored_flag"] = 1 if source_censored else 0
            rows.append(row)
    result = pd.DataFrame(rows)
    at_risk = result[result["event_in_interval"] >= 0].reset_index(drop=True)
    at_risk["event_in_interval"] = at_risk["event_in_interval"].astype(int)
    return at_risk


def test_hazard_uses_configured_intervals_and_distinct_probability_semantics():
    dataset = HazardIntervalDataset(
        intervals=[(0, 7), (8, 30), (31, 60)],
        static_feature_columns=[],
        dynamic_feature_columns=[],
        event_name="stockout",
        timing_name="stockout",
    )

    result = extract_horizon_probabilities(np.array([0.1, 0.2, 0.3]), dataset)

    assert result["p_stockout_0_7d"] == 0.1
    assert result["p_stockout_by_7d"] == 0.1
    assert result["p_stockout_8_30d"] == 0.18
    assert result["p_stockout_by_30d"] == 0.28
    assert result["prediction"] == result["p_stockout_by_60d"]
    assert result["median_status"] == "not_reached"
    assert result["p50_days_until_stockout"] is None


def test_hazard_builder_accepts_stockout_intervals_and_composite_entities():
    features = pd.DataFrame(
        {
            "hospital_id": ["H1"],
            "blood_type": ["O+"],
            "component_type": ["RBC"],
            "event_timestamp": pd.to_datetime(["2026-01-01"]),
            "current_inventory": [10.0],
        }
    )
    labels = pd.DataFrame(
        {
            "hospital_id": ["H1"],
            "blood_type": ["O+"],
            "component_type": ["RBC"],
            "event_timestamp": pd.to_datetime(["2026-01-01"]),
            "days_until_stockout": [10.0],
        }
    )

    frame, dataset = build_hazard_interval_dataset(
        features=features,
        labels=labels,
        entity_columns=["hospital_id", "blood_type", "component_type"],
        timestamp_column="event_timestamp",
        days_column="days_until_stockout",
        static_feature_columns=["blood_type", "component_type"],
        intervals=[(0, 7), (8, 30), (31, 60)],
        event_name="stockout",
        timing_name="stockout",
    )

    assert dataset.intervals == [(0, 7), (8, 30), (31, 60)]
    assert set(frame["interval_end"]) == {7, 30}
    assert 365 not in set(frame["interval_end"])


def test_hazard_builder_vectorized_output_matches_legacy_interval_expansion():
    features = pd.DataFrame(
        {
            "donor_id": ["D1", "D2", "D3", "D4"],
            "event_timestamp": pd.to_datetime(
                ["2026-01-01", "2026-01-01", "2026-01-01", "2026-01-01"]
            ),
            "region": ["north", "south", "east", "west"],
            "age": [25, 44, 31, 52],
            "readiness_score": [0.8, 0.4, 0.7, 0.3],
            "ignored_text": ["a", "b", "c", "d"],
        }
    )
    labels = pd.DataFrame(
        {
            "donor_id": ["D1", "D2", "D3", "D4"],
            "event_timestamp": pd.to_datetime(
                ["2026-01-01", "2026-01-01", "2026-01-01", "2026-01-01"]
            ),
            "days_until_next_donation": [15.0, 31.0, 420.0, np.nan],
        }
    )
    intervals = [(0, 30), (31, 90), (91, 180)]

    frame, _dataset = build_hazard_interval_dataset(
        features=features,
        labels=labels,
        entity_columns=["donor_id"],
        timestamp_column="event_timestamp",
        days_column="days_until_next_donation",
        static_feature_columns=["region"],
        intervals=intervals,
        censored_value=420.0,
        event_name="donation",
        timing_name="next_donation",
    )
    expected = _legacy_hazard_interval_dataset(
        features,
        labels,
        entity_columns=["donor_id"],
        timestamp_column="event_timestamp",
        days_column="days_until_next_donation",
        static_feature_columns=["region"],
        intervals=intervals,
        censored_value=420.0,
    )

    assert_frame_equal(frame, expected)


def test_inventory_simulator_probabilities_use_all_paths_as_denominator():
    sim = InventorySimulator(
        n_paths=2000,
        horizons_days=[1, 3, 5],
        default_cv=0.5,
        safety_threshold=0.5,
        seed=3,
    )

    result = sim.run(
        current_stock=1.0,
        demand_forecast=0.1,
        supply_forecast=0.0,
        waste_forecast=0.0,
        demand_cv=0.5,
        supply_cv=0.001,
        waste_cv=0.001,
    )

    assert 0 < result["p_stockout_0_maxd"] < 1
    assert result["p_stockout_0_1d"] <= result["p_stockout_0_maxd"]
    assert result["p_stockout_0_3d"] <= result["p_stockout_0_maxd"]
    assert result["p_stockout_0_5d"] == result["p_stockout_0_maxd"]
    assert result["p50_days_until_stockout"] is None
    assert result["median_status"] == "not_reached"


def test_inventory_simulator_no_stockout_quantiles_are_null():
    sim = InventorySimulator(
        n_paths=100,
        horizons_days=[7, 30],
        default_cv=0.1,
        safety_threshold=0.5,
        seed=7,
    )

    result = sim.run(
        current_stock=100.0,
        demand_forecast=0.0,
        supply_forecast=0.0,
        waste_forecast=0.0,
    )

    assert result["p_stockout_0_maxd"] == 0.0
    assert result["p10_days_until_stockout"] is None
    assert result["p50_days_until_stockout"] is None
    assert result["p90_days_until_stockout"] is None
    assert result["median_status"] == "not_reached"


def test_inventory_pyfunc_pickle_omits_loaded_child_model_cache():
    sim = InventoryRiskSimulatorPyfunc(
        demand_model_uri="models:/component_demand_quantile_forecast_model/1",
        supply_model_uri="models:/component_supply_forecast_model/1",
        waste_model_uri="models:/component_expiry_waste_model/1",
    )
    sim._demand_model = {"loaded": "demand"}
    sim._supply_model = {"loaded": "supply"}
    sim._waste_model = {"loaded": "waste"}

    restored = pickle.loads(pickle.dumps(sim))

    assert restored.demand_model_uri == sim.demand_model_uri
    assert restored._demand_model is None
    assert restored._supply_model is None
    assert restored._waste_model is None
