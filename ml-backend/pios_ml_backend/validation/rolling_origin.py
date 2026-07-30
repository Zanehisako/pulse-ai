from __future__ import annotations

import pandas as pd


def _time_split(
    frame: pd.DataFrame,
    timestamp_column: str,
    *,
    test_fraction: float = 0.2,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    ordered = frame.sort_values(timestamp_column).reset_index(drop=True)
    unique_timestamps = (
        ordered[timestamp_column].dropna().drop_duplicates().sort_values()
    )
    if unique_timestamps.empty:
        split_index = max(1, int(len(ordered) * (1.0 - test_fraction)))
        return ordered.iloc[:split_index].copy(), ordered.iloc[split_index:].copy()

    split_position = max(1, int(len(unique_timestamps) * (1.0 - test_fraction)))
    split_position = min(split_position, len(unique_timestamps) - 1)
    cutoff = unique_timestamps.iloc[split_position]
    train_frame = ordered[ordered[timestamp_column] < cutoff].copy()
    test_frame = ordered[ordered[timestamp_column] >= cutoff].copy()
    if train_frame.empty or test_frame.empty:
        split_index = max(1, int(len(ordered) * (1.0 - test_fraction)))
        train_frame = ordered.iloc[:split_index].copy()
        test_frame = ordered.iloc[split_index:].copy()
    return train_frame, test_frame


def rolling_origin_splits(
    frame: pd.DataFrame,
    timestamp_column: str,
    *,
    folds: int,
    min_train_fraction: float,
    validation_fraction: float,
) -> list[tuple[pd.DataFrame, pd.DataFrame]]:
    ordered = frame.sort_values(timestamp_column).reset_index(drop=True)
    timestamps = pd.to_datetime(ordered[timestamp_column], utc=True, errors="coerce")
    unique_dates = pd.Series(timestamps.dt.normalize().dropna().unique()).sort_values()
    if unique_dates.empty:
        raise ValueError("Cannot build rolling-origin splits without valid timestamps.")
    n_dates = len(unique_dates)
    min_train = max(2, int(n_dates * min_train_fraction))
    validation_size = max(1, int(n_dates * validation_fraction))
    available = n_dates - min_train - validation_size
    if available < 0:
        return [_time_split(ordered, timestamp_column, test_fraction=validation_fraction)]
    step = max(1, available // max(1, folds - 1))
    splits: list[tuple[pd.DataFrame, pd.DataFrame]] = []
    for fold in range(folds):
        train_end = min_train + fold * step
        validation_end = min(n_dates, train_end + validation_size)
        if validation_end <= train_end:
            break
        train_dates = set(unique_dates.iloc[:train_end])
        validation_dates = set(unique_dates.iloc[train_end:validation_end])
        train_mask = timestamps.dt.normalize().isin(train_dates)
        validation_mask = timestamps.dt.normalize().isin(validation_dates)
        train = ordered.loc[train_mask].copy()
        validation = ordered.loc[validation_mask].copy()
        if not train.empty and not validation.empty:
            splits.append((train, validation))
    if not splits:
        splits.append(_time_split(ordered, timestamp_column, test_fraction=validation_fraction))
    return splits
