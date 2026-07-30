from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd


ML_BACKEND_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG_DIR = ML_BACKEND_ROOT / "config"


def _read_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise ValueError(f"Could not read Quebec dataset config: {path}") from exc
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid Quebec dataset config: {path}") from exc
    if not isinstance(payload, dict):
        raise ValueError("Quebec dataset config root must be an object.")
    return payload


def _required_text(row: dict[str, Any], key: str, label: str) -> str:
    value = str(row.get(key) or "").strip()
    if not value:
        raise ValueError(f"{label}.{key} is required.")
    return value


def _required_string_list(row: dict[str, Any], key: str, label: str) -> list[str]:
    value = row.get(key)
    if not isinstance(value, list):
        raise ValueError(f"{label}.{key} must be a list.")
    items = [str(item).strip() for item in value if str(item).strip()]
    if not items:
        raise ValueError(f"{label}.{key} must contain at least one value.")
    if len(items) != len(set(items)):
        raise ValueError(f"{label}.{key} must not contain duplicates.")
    return items


def validate_quebec_training_datasets_config(config: dict[str, Any]) -> dict[str, Any]:
    if config.get("enabled") is not True:
        raise ValueError("Quebec training dataset config must be enabled.")
    sources = config.get("sources")
    if not isinstance(sources, dict) or not sources:
        raise ValueError("Quebec training dataset config must define sources.")
    required_sources = {"quebec_donor", "quebec_inventory"}
    missing_sources = sorted(required_sources - set(sources))
    if missing_sources:
        raise ValueError(f"Quebec training dataset sources missing: {missing_sources}")

    for key, raw_source in sources.items():
        if not isinstance(raw_source, dict):
            raise ValueError(f"sources.{key} must be an object.")
        label = f"sources.{key}"
        if raw_source.get("enabled") is not True:
            raise ValueError(f"{label}.enabled must be true.")
        _required_text(raw_source, "feature_parquet", label)
        _required_text(raw_source, "label_parquet", label)
        _required_string_list(raw_source, "entity_keys", label)
        _required_text(raw_source, "timestamp_column", label)
        label_columns = raw_source.get("label_columns")
        if not isinstance(label_columns, dict) or not label_columns:
            raise ValueError(f"{label}.label_columns must be a non-empty object.")
        for label_key, column in label_columns.items():
            if not str(label_key).strip() or not str(column or "").strip():
                raise ValueError(f"{label}.label_columns contains an empty key or value.")
        _required_string_list(raw_source, "leakage_exclusions", label)
    return config


def load_quebec_training_datasets_config(
    *,
    config_dir: Path | None = None,
) -> dict[str, Any]:
    path = (config_dir or DEFAULT_CONFIG_DIR) / "quebec_training_datasets.json"
    return validate_quebec_training_datasets_config(_read_json(path))


def dataset_source_config(
    source_key: str,
    *,
    config_dir: Path | None = None,
) -> dict[str, Any]:
    config = load_quebec_training_datasets_config(config_dir=config_dir)
    sources = config["sources"]
    if source_key not in sources:
        raise ValueError(f"Unknown Quebec dataset source: {source_key}")
    return dict(sources[source_key])


def _resolve_ml_backend_path(path_text: str) -> Path:
    path = Path(path_text)
    if path.is_absolute():
        return path
    return ML_BACKEND_ROOT / path


def load_configured_dataset_frame(
    source_key: str,
    *,
    label_columns: list[str] | tuple[str, ...] | None = None,
    config_dir: Path | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    source = dataset_source_config(source_key, config_dir=config_dir)
    feature_path = _resolve_ml_backend_path(str(source["feature_parquet"]))
    label_path = _resolve_ml_backend_path(str(source["label_parquet"]))
    if not feature_path.is_file():
        raise FileNotFoundError(
            f"Configured Quebec feature file is missing for {source_key}: {feature_path}"
        )
    if not label_path.is_file():
        raise FileNotFoundError(
            f"Configured Quebec label file is missing for {source_key}: {label_path}"
        )

    features = pd.read_parquet(feature_path).copy()
    labels = pd.read_parquet(label_path).copy()
    entity_keys = [str(value) for value in source["entity_keys"]]
    timestamp_column = str(source["timestamp_column"])
    merge_keys = [*entity_keys, timestamp_column]

    missing_feature_keys = [key for key in merge_keys if key not in features.columns]
    missing_label_keys = [key for key in merge_keys if key not in labels.columns]
    if missing_feature_keys:
        raise ValueError(f"{source_key} features missing merge keys: {missing_feature_keys}")
    if missing_label_keys:
        raise ValueError(f"{source_key} labels missing merge keys: {missing_label_keys}")

    all_configured_labels = {
        str(column)
        for column in source.get("label_columns", {}).values()
        if str(column or "").strip()
    }
    leakage_columns = {
        str(column)
        for column in source.get("leakage_exclusions", [])
        if str(column or "").strip()
    }
    features = features.drop(
        columns=sorted((all_configured_labels | leakage_columns) & set(features.columns)),
        errors="ignore",
    )

    selected_labels = list(label_columns or [])
    if selected_labels:
        missing_labels = [column for column in selected_labels if column not in labels.columns]
        if missing_labels:
            raise ValueError(f"{source_key} labels missing requested columns: {missing_labels}")
        labels = labels[[*merge_keys, *selected_labels]].drop_duplicates(merge_keys)
        frame = features.merge(labels, on=merge_keys, how="left")
    else:
        frame = features

    frame[timestamp_column] = pd.to_datetime(
        frame[timestamp_column],
        utc=True,
        errors="coerce",
    )
    frame = frame.sort_values(merge_keys).reset_index(drop=True)
    return frame, source
