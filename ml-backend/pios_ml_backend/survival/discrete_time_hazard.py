"""
Discrete-time hazard model utilities for PIOS.

Builds interval datasets from time-to-event labels, converts classifier
hazard predictions to survival curves, and extracts horizon probabilities.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Any


class HazardIntervalDataset:
    def __init__(
        self,
        intervals: list[tuple[int, int]],
        static_feature_columns: list[str],
        dynamic_feature_columns: list[str],
        event_name: str = "event",
        timing_name: str | None = None,
    ) -> None:
        self.intervals = intervals
        self.static_feature_columns = list(static_feature_columns)
        self.dynamic_feature_columns = list(dynamic_feature_columns)
        self.event_name = event_name
        self.timing_name = timing_name or event_name

    @property
    def interval_count(self) -> int:
        return len(self.intervals)

    @property
    def max_days(self) -> int:
        return max(end for _, end in self.intervals)

    def interval_index_for_day(self, days: float) -> int | None:
        for idx, (start, end) in enumerate(self.intervals):
            if start < days <= end:
                return idx
        return None


def build_hazard_interval_dataset(
    features: pd.DataFrame,
    labels: pd.DataFrame,
    *,
    entity_column: str | None = None,
    entity_columns: list[str] | None = None,
    timestamp_column: str,
    days_column: str,
    static_feature_columns: list[str],
    intervals: list[tuple[int, int]] | None = None,
    censored_value: float | None = None,
    max_intervals: int | None = None,
    event_name: str = "event",
    timing_name: str | None = None,
) -> tuple[pd.DataFrame, HazardIntervalDataset]:
    resolved_entity_columns = [
        str(col)
        for col in (entity_columns or ([entity_column] if entity_column else []))
        if str(col).strip()
    ]
    if not resolved_entity_columns:
        raise ValueError("At least one entity column is required.")

    merged = features.merge(
        labels[resolved_entity_columns + [timestamp_column, days_column]],
        on=resolved_entity_columns + [timestamp_column],
        how="inner",
    )
    merged = merged.sort_values(resolved_entity_columns + [timestamp_column]).reset_index(drop=True)

    days = pd.to_numeric(merged[days_column], errors="coerce")
    is_censored = days == censored_value if censored_value is not None else pd.Series(False, index=days.index)

    if intervals is None:
        intervals = [(0, 30), (31, 90), (91, 180), (181, 365)]
    intervals = [(int(start), int(end)) for start, end in intervals]
    if max_intervals is not None:
        intervals = intervals[:max_intervals]

    all_static = [c for c in static_feature_columns if c in merged.columns]
    all_features = list(merged.columns)
    all_dynamic = [c for c in all_features if c not in all_static
                   and c not in set(resolved_entity_columns + [timestamp_column, days_column])]

    valid_dynamic = [c for c in all_dynamic if pd.api.types.is_numeric_dtype(merged[c])]

    ds = HazardIntervalDataset(
        intervals=intervals,
        static_feature_columns=all_static,
        dynamic_feature_columns=valid_dynamic,
        event_name=event_name,
        timing_name=timing_name,
    )

    output_columns = [
        "interval_start",
        "interval_end",
        "interval_index",
        "interval_midpoint",
        *all_static,
        *valid_dynamic,
        "event_in_interval",
        "censored_flag",
    ]
    if merged.empty or not intervals:
        return pd.DataFrame(columns=output_columns), ds

    interval_count = len(intervals)
    row_count = len(merged)
    source_positions = np.repeat(np.arange(row_count), interval_count)
    starts = np.asarray([start for start, _end in intervals], dtype=np.int64)
    ends = np.asarray([end for _start, end in intervals], dtype=np.int64)
    interval_indices = np.arange(interval_count, dtype=np.int64)
    interval_starts = np.tile(starts, row_count)
    interval_ends = np.tile(ends, row_count)

    repeated_features = merged.iloc[source_positions][all_static + valid_dynamic].reset_index(drop=True)
    donor_days = np.repeat(days.to_numpy(dtype=np.float64), interval_count)
    donor_censored = np.repeat(is_censored.to_numpy(dtype=bool), interval_count)

    event_in_interval = np.zeros(len(source_positions), dtype=np.int8)
    event_mask = (
        (~donor_censored)
        & (interval_starts < donor_days)
        & (donor_days <= interval_ends)
    )
    expired_mask = donor_days <= interval_starts
    event_in_interval[event_mask] = 1
    event_in_interval[expired_mask] = -1

    interval_frame = pd.DataFrame(
        {
            "interval_start": interval_starts,
            "interval_end": interval_ends,
            "interval_index": np.tile(interval_indices, row_count),
            "interval_midpoint": (interval_starts + interval_ends) / 2.0,
        }
    )
    result = pd.concat(
        [
            interval_frame,
            repeated_features,
            pd.DataFrame(
                {
                    "event_in_interval": event_in_interval,
                    "censored_flag": donor_censored.astype(int),
                }
            ),
        ],
        axis=1,
    )
    at_risk = result[result["event_in_interval"] >= 0].reset_index(drop=True)
    at_risk["event_in_interval"] = at_risk["event_in_interval"].astype(int)

    return at_risk[output_columns], ds


def _inverse_cloglog(linear_predictor: np.ndarray) -> np.ndarray:
    clipped = np.clip(linear_predictor, -20.0, 20.0)
    return 1.0 - np.exp(-np.exp(clipped))


def survival_curve_from_hazards(
    hazard_probs: np.ndarray,
) -> np.ndarray:
    survival = np.ones(len(hazard_probs) + 1, dtype=np.float64)
    for i in range(len(hazard_probs)):
        survival[i + 1] = survival[i] * (1.0 - max(0.0, min(1.0, float(hazard_probs[i]))))
    return survival


def extract_horizon_probabilities(
    hazard_probs: np.ndarray,
    dataset: HazardIntervalDataset,
) -> dict[str, Any]:
    survival = survival_curve_from_hazards(hazard_probs)
    probs: dict[str, float] = {}
    event = dataset.event_name
    timing = dataset.timing_name

    cumulative = 0.0
    for i, (start, end) in enumerate(dataset.intervals):
        interval_probability = 0.0
        if i < len(hazard_probs):
            interval_probability = survival[i] * max(0.0, min(1.0, float(hazard_probs[i])))
            cumulative += interval_probability
        probs[f"p_{event}_{start}_{end}d"] = round(float(interval_probability), 6)
        probs[f"p_{event}_by_{end}d"] = round(float(cumulative), 6)

    total_prob = cumulative
    remaining = 1.0 - total_prob
    probs[f"p_no_{event}_by_{dataset.intervals[-1][1]}d"] = round(max(0.0, remaining), 6)

    expected_days = 0.0
    for i, (start, end) in enumerate(dataset.intervals):
        if i < len(hazard_probs):
            p = survival[i] * max(0.0, min(1.0, float(hazard_probs[i])))
            expected_days += p * (start + end) / 2.0
    expected_days += max(0.0, remaining) * dataset.max_days
    probs[f"expected_days_until_{timing}"] = round(float(expected_days), 1)

    p50 = None
    for i, (start, end) in enumerate(dataset.intervals):
        if i < len(hazard_probs):
            if survival[i + 1] <= 0.5:
                p50 = float(end)
                break

    probs[f"p50_days_until_{timing}"] = p50
    probs["median_status"] = "reached" if p50 is not None else "not_reached"
    probs["prediction"] = round(float(total_prob), 6)

    return probs
