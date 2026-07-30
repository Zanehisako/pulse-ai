from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from forecasting_search import log_forecast_search_to_mlflow, run_forecast_search
from model_architecture_config import load_quebec_arima_sarima_forecast_config
from model_training_utils import ARTIFACTS_DIRECTORY, FEATURE_LABELS_DIRECTORY, TaskDataset, _quantile_mask, _safe_numeric, _save_parquet_pair
from quebec_training_data import load_configured_dataset_frame


def build_quebec_arima_sarima_task() -> TaskDataset:
    config = load_quebec_arima_sarima_forecast_config()
    target_column = str(config["target"]["column"])
    feature_config = config["feature_engineering"]
    group_columns = tuple(str(value) for value in feature_config["group_columns"])
    timestamp_column = str(feature_config["timestamp_column"])
    frame, _source = load_configured_dataset_frame(
        str(config["dataset_source"]),
        label_columns=[target_column],
    )
    frame[target_column] = _safe_numeric(frame[target_column]).fillna(0.0)
    feature_path = FEATURE_LABELS_DIRECTORY / "quebec_arima_sarima_forecast_features.parquet"
    label_path = FEATURE_LABELS_DIRECTORY / "quebec_arima_sarima_forecast_labels.parquet"
    _save_parquet_pair(frame, feature_path, label_path, target_column)

    def hard_mask(candidate_frame: pd.DataFrame) -> pd.Series:
        return (
            _quantile_mask(candidate_frame["current_inventory"], 0.25, "low")
            | _quantile_mask(candidate_frame["medium_demand_pressure"], 0.75, "high")
            | _quantile_mask(candidate_frame["stockout_history_pressure"], 0.75, "high")
        )

    return TaskDataset(
        task_name=str(config["id"]),
        model_name=str(config["registered_model_name"]),
        experiment_name="quebec_arima_sarima_forecast",
        task_type="regression",
        frame=frame,
        target_column=target_column,
        timestamp_column=timestamp_column,
        feature_service="quebec_arima_sarima_forecast_service",
        entity_keys=group_columns,
        label_file=label_path,
        feature_file=feature_path,
        description=str(config["description"]),
        hard_mask_builder=hard_mask,
        prediction_defaults=dict(config["prediction_defaults"]),
        training_config=config,
    )


def train_quebec_arima_sarima_forecast() -> dict:
    config = load_quebec_arima_sarima_forecast_config()
    dataset = build_quebec_arima_sarima_task()
    output_dir = ARTIFACTS_DIRECTORY / "quebec_arima_sarima_forecast"
    result = run_forecast_search(
        dataset=dataset,
        task_config=config,
        output_dir=output_dir,
        Prophet=None,
    )
    summary = {
        "model_id": config["id"],
        "registered_model_name": config["registered_model_name"],
        "selected_candidate": result["selected_candidate"],
        "test_metrics": result["test_metrics"],
        "test_hard_metrics": result["test_hard_metrics"],
        "artifact_dir": str(output_dir),
    }
    (output_dir / "metrics.json").write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    summary.update(
        log_forecast_search_to_mlflow(
            dataset=dataset,
            task_config=config,
            result=result,
            output_dir=output_dir,
        )
    )
    return summary


if __name__ == "__main__":
    print(json.dumps(train_quebec_arima_sarima_forecast(), indent=2, default=str))
