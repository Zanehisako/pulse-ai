"""
train_donor_next_donation_hazard.py

Discrete-time hazard model for donor next-donation timing.

Builds interval dataset from days_until_next_donation labels, trains a
tabular classifier ensemble on per-interval event prediction, wraps the
best estimator in a HazardModelPyfunc that converts hazard predictions to
survival curves and horizon probabilities at inference time.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd
from mlflow.models import infer_signature

_TRAINING_SCRIPTS = Path(__file__).resolve().parent
if str(_TRAINING_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_TRAINING_SCRIPTS))

_ML_BACKEND = _TRAINING_SCRIPTS.parent
if str(_ML_BACKEND) not in sys.path:
    sys.path.insert(0, str(_ML_BACKEND))

from pios_ml_backend.survival.discrete_time_hazard import (
    HazardIntervalDataset,
    build_hazard_interval_dataset,
    extract_horizon_probabilities,
)
from pios_ml_backend.survival.hazard_pyfunc import HazardModelPyfunc
from pios_ml_backend.training_resource_profiles import (
    apply_profile_to_model_config,
    nested_bool,
)
from model_training_utils import (
    FEATURE_LABELS_DIRECTORY,
    TaskDataset,
    configure_mlflow,
    fit_task,
    _feature_columns_for_dataset,
    _feature_stats_payload,
    _persist_model_stats,
    _safe_numeric,
)

_CONFIG_PATH = _ML_BACKEND / "config" / "donor_next_donation_hazard_model.json"


def _load_config() -> dict:
    return apply_profile_to_model_config(
        json.loads(_CONFIG_PATH.read_text(encoding="utf-8"))
    )


def build_hazard_task() -> tuple[TaskDataset, HazardIntervalDataset, TaskDataset]:
    config = _load_config()
    target_cfg = config["target"]
    feature_cfg = config["feature_engineering"]
    intervals = [tuple(interval) for interval in target_cfg["intervals"]]

    features = pd.read_parquet(FEATURE_LABELS_DIRECTORY / "quebec_donor_features.parquet")
    labels = pd.read_parquet(FEATURE_LABELS_DIRECTORY / "quebec_donor_labels.parquet")

    static_cols = [str(c) for c in feature_cfg.get("static_features", [])]
    static_cols = [c for c in static_cols if c in features.columns]

    interval_frame, interval_ds = build_hazard_interval_dataset(
        features=features,
        labels=labels,
        entity_columns=[str(c) for c in feature_cfg["group_columns"]],
        timestamp_column=str(feature_cfg["timestamp_column"]),
        days_column=str(target_cfg["event_column"]),
        static_feature_columns=static_cols,
        intervals=intervals,
        censored_value=target_cfg.get("censored_value"),
        event_name="donation",
        timing_name="next_donation",
    )

    interval_frame["interval_id"] = (
        interval_frame["interval_index"].astype(str)
    )
    interval_frame = interval_frame.sort_values(["interval_id"]).reset_index(drop=True)

    prediction_defaults = dict(config.get("prediction_defaults", {}))
    prediction_defaults.setdefault("output", {})
    prediction_defaults["output"].setdefault("name", "donation_probability")

    training_config = dict(config.get("training", {}))
    selection_config = dict(config.get("selection", {}))

    task = TaskDataset(
        task_name=str(config["id"]),
        model_name=str(config["registered_model_name"]),
        experiment_name="donor_next_donation_hazard",
        task_type="classification",
        frame=interval_frame,
        target_column="event_in_interval",
        timestamp_column="interval_midpoint",
        feature_service="quebec_donor_service",
        entity_keys=("donor_id",),
        label_file=FEATURE_LABELS_DIRECTORY / "quebec_donor_labels.parquet",
        feature_file=FEATURE_LABELS_DIRECTORY / "quebec_donor_features.parquet",
        description=str(config["description"]),
        hard_mask_builder=lambda df: pd.Series(False, index=df.index),
        prediction_defaults=prediction_defaults,
        training_config=training_config,
    )

    return task, interval_ds, interval_frame


def train_and_log_hazard() -> dict:
    config = _load_config()
    task, interval_ds, interval_frame = build_hazard_task()

    task_result = fit_task(task)

    best_estimator = task_result.best_result.estimator

    feature_columns = _feature_columns_for_dataset(
        task_result.train_frame, task
    )
    feature_stats = _feature_stats_payload(
        task_result.train_frame[[c for c in feature_columns if c in task_result.train_frame.columns]],
        feature_columns,
    )

    pyfunc_wrapper = HazardModelPyfunc(
        estimator=best_estimator,
        dataset=interval_ds,
        feature_columns=feature_columns,
    )

    input_example = task_result.test_frame[
        [c for c in feature_columns if c in task_result.test_frame.columns]
    ].head(3).copy()
    for c in input_example.columns:
        input_example[c] = pd.to_numeric(input_example[c], errors="coerce").fillna(0.0)

    sample_output = pyfunc_wrapper.predict(None, input_example)
    signature = infer_signature(input_example, sample_output)

    configure_mlflow()
    mlflow.set_experiment("donor_next_donation_hazard")

    prediction_defaults = dict(config.get("prediction_defaults", {}))
    prediction_defaults["prediction_source"] = "feast_online"

    with mlflow.start_run(run_name=f"{config['id']}-hazard"):
        mlflow.set_tags({
            "developer": "pios_ml",
            "registered_model_name": config["registered_model_name"],
            "environment": "Training",
            "model_family": task_result.best_result.candidate_name,
            "training_design": "discrete_time_hazard",
            "task_type": "classification",
            "prediction_target": "event_in_interval",
            "description": config["description"],
            "promotable": "true",
            "feature_columns": json.dumps(feature_columns),
            "hazard_intervals": json.dumps(interval_ds.intervals),
        })

        mlflow.log_params({
            "selected_candidate": task_result.best_result.candidate_name,
            "decision_threshold": getattr(task_result.best_result, "decision_threshold", 0.5),
            "feature_count": len(feature_columns),
            "dataset_rows": len(interval_frame),
            "train_rows": len(task_result.train_frame),
            "test_rows": len(task_result.test_frame),
            "interval_count": interval_ds.interval_count,
            "prediction_target": "event_in_interval",
        })

        for metric_name, metric_value in task_result.best_result.metrics.items():
            mlflow.log_metric(metric_name, metric_value)
        for metric_name, metric_value in task_result.best_result.hard_metrics.items():
            mlflow.log_metric(f"hard_{metric_name}", metric_value)

        mlflow.log_dict(feature_stats, "feature_stats.json")
        mlflow.log_dict(prediction_defaults, "prediction_defaults.json")

        model_info = mlflow.pyfunc.log_model(
            artifact_path=config["registered_model_name"],
            python_model=pyfunc_wrapper,
            signature=signature,
            input_example=input_example if nested_bool(
                task.training_config or {},
                ("mlflow", "input_example", "enabled"),
                True,
            ) else None,
            registered_model_name=config["registered_model_name"],
        )

        from mlflow import MlflowClient
        client = MlflowClient()
        client.set_registered_model_alias(
            name=config["registered_model_name"],
            alias="challenger",
            version=model_info.registered_model_version,
        )

        _persist_model_stats(
            model_id=config["registered_model_name"],
            identifier=config["registered_model_name"],
            payload=feature_stats,
        )

        return {
            "task_name": config["id"],
            "selected_candidate": task_result.best_result.candidate_name,
            "overall_metrics": task_result.best_result.metrics,
            "hard_metrics": task_result.best_result.hard_metrics,
            "run_id": mlflow.active_run().info.run_id if mlflow.active_run() else None,
            "model_uri": f"runs:/{mlflow.active_run().info.run_id}/{config['registered_model_name']}" if mlflow.active_run() else None,
            "registered_model_version": model_info.registered_model_version,
        }


if __name__ == "__main__":
    result = train_and_log_hazard()
    print(f"\nDonor hazard model trained: {result['selected_candidate']}")
    print(f"  Overall brier: {result['overall_metrics'].get('brier_score', 'N/A')}")
    print(f"  Overall AUC: {result['overall_metrics'].get('roc_auc', 'N/A')}")
    print(f"  Model v{result.get('registered_model_version', '?')} @challenger")
    print("Done.")
