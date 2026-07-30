from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd


TRAINING_SCRIPTS = Path(__file__).resolve().parents[1] / "training_scripts"
if str(TRAINING_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(TRAINING_SCRIPTS))

from model_training_utils import (  # noqa: E402
    CandidateResult,
    TaskDataset,
    TaskResult,
    _feature_columns_for_dataset,
    _feature_leakage_audit_payload,
    _is_future_or_label_column,
    _normalize_datetime_columns_for_mlflow,
)


def test_leakage_guard_excludes_configured_future_and_date_columns(tmp_path: Path):
    frame = pd.DataFrame(
        {
            "donor_id": ["D001"],
            "event_timestamp": pd.to_datetime(["2026-01-01"]),
            "target": [1],
            "last_donation_date": ["2025-12-01"],
            "as_of_date": ["2026-01-01"],
            "snapshot_y_donate_30d": [1],
            "valid_feature": [0.7],
        }
    )
    dataset = TaskDataset(
        task_name="test",
        model_name="test",
        experiment_name="test",
        task_type="classification",
        frame=frame,
        target_column="target",
        timestamp_column="event_timestamp",
        feature_service="test_service",
        entity_keys=("donor_id",),
        label_file=tmp_path / "labels.parquet",
        feature_file=tmp_path / "features.parquet",
        description="test",
        hard_mask_builder=lambda input_frame: pd.Series(False, index=input_frame.index),
    )

    assert _feature_columns_for_dataset(frame, dataset) == ["valid_feature"]
    assert _is_future_or_label_column("snapshot_y_donate_30d")
    assert _is_future_or_label_column("last_donation_date")


def test_leakage_guard_allows_configured_entity_feature(tmp_path: Path):
    frame = pd.DataFrame(
        {
            "hospital_id": ["H001"],
            "blood_type": ["O+"],
            "event_timestamp": pd.to_datetime(["2026-01-01"]),
            "target": [5.0],
            "temperature": [20.0],
        }
    )
    dataset = TaskDataset(
        task_name="test",
        model_name="test",
        experiment_name="test",
        task_type="regression",
        frame=frame,
        target_column="target",
        timestamp_column="event_timestamp",
        feature_service="test_service",
        entity_keys=("hospital_id", "blood_type"),
        label_file=tmp_path / "labels.parquet",
        feature_file=tmp_path / "features.parquet",
        description="test",
        hard_mask_builder=lambda input_frame: pd.Series(False, index=input_frame.index),
        serving_feature_columns=("blood_type", "temperature"),
    )

    assert _feature_columns_for_dataset(frame, dataset) == ["blood_type", "temperature"]


def test_leakage_audit_reports_high_correlation_without_blocking(tmp_path: Path):
    frame = pd.DataFrame(
        {
            "event_timestamp": pd.date_range("2026-01-01", periods=6),
            "feature": [1, 2, 3, 4, 5, 6],
            "target": [1, 2, 3, 4, 5, 6],
        }
    )
    dataset = TaskDataset(
        task_name="test",
        model_name="test_model",
        experiment_name="test",
        task_type="regression",
        frame=frame,
        target_column="target",
        timestamp_column="event_timestamp",
        feature_service="test_service",
        entity_keys=(),
        label_file=tmp_path / "labels.parquet",
        feature_file=tmp_path / "features.parquet",
        description="test",
        hard_mask_builder=lambda input_frame: pd.Series(False, index=input_frame.index),
    )
    result = CandidateResult(
        candidate_name="candidate",
        estimator=object(),
        metrics={},
        hard_metrics={},
        validation_score=0.0,
        predictions=frame["target"].to_numpy(),
        probabilities=None,
    )
    audit = _feature_leakage_audit_payload(
        TaskResult(
            dataset=dataset,
            candidate_results=[result],
            best_result=result,
            validation_result=result,
            feature_columns=["feature"],
            feature_types={"feature": "int64"},
            train_frame=frame,
            test_frame=frame,
        )
    )

    assert audit["status"] == "pass"
    assert audit["high_correlation_features"][0]["feature"] == "feature"


def test_mlflow_datetime_normalization_removes_timezone():
    frame = pd.DataFrame(
        {
            "event_timestamp": pd.date_range("2026-01-01", periods=2, tz="UTC"),
            "feature": [1.0, 2.0],
        }
    )

    normalized = _normalize_datetime_columns_for_mlflow(frame)

    assert str(normalized["event_timestamp"].dtype) == "datetime64[ns]"
    assert normalized["feature"].tolist() == [1.0, 2.0]
