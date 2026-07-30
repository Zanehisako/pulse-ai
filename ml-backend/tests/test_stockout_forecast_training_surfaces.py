from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd


TRAINING_DIR = Path(__file__).resolve().parents[1] / "training_scripts"
if str(TRAINING_DIR) not in sys.path:
    sys.path.insert(0, str(TRAINING_DIR))

from model_training_utils import (  # noqa: E402
    CatBoostFrameClassifier,
    ProphetFrameClassifier,
    TorchSequenceClassifier,
    TorchSequenceEnsembleClassifier,
    WeightedProbabilityEnsembleClassifier,
    _classification_estimator_from_config,
    _regression_estimator_from_config,
)


class _ConstantProbabilityEstimator:
    def __init__(self, probability: float) -> None:
        self.probability = probability

    def fit(self, X, y):  # noqa: ANN001
        self.classes_ = np.asarray([0, 1])
        return self

    def predict_proba(self, X):  # noqa: ANN001
        positive = np.full(len(X), self.probability, dtype=float)
        return np.column_stack([1.0 - positive, positive])


def test_weighted_probability_ensemble_clamps_probability_outputs():
    frame = pd.DataFrame({"feature": [1.0, 2.0, 3.0]})
    ensemble = WeightedProbabilityEnsembleClassifier(
        estimators=[
            ("low", _ConstantProbabilityEstimator(0.2)),
            ("high", _ConstantProbabilityEstimator(0.8)),
        ],
        weights=[1.0, 3.0],
    ).fit(frame, pd.Series([0, 1, 1]))

    probabilities = ensemble.predict_proba(frame)[:, 1]

    assert np.all(probabilities >= 0.0)
    assert np.all(probabilities <= 1.0)
    assert np.allclose(probabilities, 0.65)


def test_short_horizon_candidate_builders_use_real_configured_estimators():
    frame = pd.DataFrame(
        {
            "event_timestamp": pd.date_range("2026-01-01", periods=3),
            "current_inventory": [10.0, 8.0, 5.0],
            "blood_type": ["O+", "A+", "O-"],
        }
    )
    candidates = {
        "xgboost": _classification_estimator_from_config(
            {"name": "xgboost", "family": "xgboost"},
            frame,
            timestamp_column="event_timestamp",
        ),
        "random_forest": _classification_estimator_from_config(
            {"name": "random_forest", "family": "random_forest"},
            frame,
            timestamp_column="event_timestamp",
        ),
        "lightgbm": _classification_estimator_from_config(
            {"name": "lightgbm", "family": "lightgbm"},
            frame,
            timestamp_column="event_timestamp",
        ),
        "catboost": _classification_estimator_from_config(
            {"name": "catboost", "family": "catboost"},
            frame,
            timestamp_column="event_timestamp",
        ),
    }

    assert candidates["xgboost"].named_steps["model"].__class__.__name__ == "XGBClassifier"
    assert candidates["random_forest"].named_steps["model"].__class__.__name__ == "RandomForestClassifier"
    assert candidates["lightgbm"].named_steps["model"].__class__.__name__ == "LGBMClassifier"
    assert isinstance(candidates["catboost"], CatBoostFrameClassifier)


def test_candidate_builders_apply_configured_thread_profile(monkeypatch):
    monkeypatch.setenv("PIOS_TREE_MODEL_THREADS", "3")
    frame = pd.DataFrame(
        {
            "event_timestamp": pd.date_range("2026-01-01", periods=3),
            "current_inventory": [10.0, 8.0, 5.0],
            "blood_type": ["O+", "A+", "O-"],
        }
    )
    training_config = {
        "resource": {
            "thread_env_var": "PIOS_TREE_MODEL_THREADS",
            "tree_model_threads": 4,
            "override_configured_thread_params": True,
            "candidate_thread_params": {
                "catboost": ["thread_count"],
                "lightgbm": ["n_jobs"],
                "random_forest": ["n_jobs"],
                "xgboost": ["n_jobs"],
            },
        }
    }

    xgboost = _classification_estimator_from_config(
        {"name": "xgboost", "family": "xgboost", "params": {"n_jobs": 8}},
        frame,
        timestamp_column="event_timestamp",
        training_config=training_config,
    )
    lightgbm = _classification_estimator_from_config(
        {"name": "lightgbm", "family": "lightgbm"},
        frame,
        timestamp_column="event_timestamp",
        training_config=training_config,
    )
    forest = _classification_estimator_from_config(
        {"name": "random_forest", "family": "random_forest"},
        frame,
        timestamp_column="event_timestamp",
        training_config=training_config,
    )
    catboost = _classification_estimator_from_config(
        {"name": "catboost", "family": "catboost"},
        frame,
        timestamp_column="event_timestamp",
        training_config=training_config,
    )

    assert xgboost.named_steps["model"].get_params()["n_jobs"] == 3
    assert lightgbm.named_steps["model"].get_params()["n_jobs"] == 3
    assert forest.named_steps["model"].get_params()["n_jobs"] == 3
    assert catboost.params["thread_count"] == 3


def test_medium_and_long_probability_builders_use_sequence_and_prophet_classes():
    frame = pd.DataFrame(
        {
            "event_timestamp": pd.date_range("2026-01-01", periods=3),
            "current_inventory": [10.0, 8.0, 5.0],
        }
    )
    prophet = _classification_estimator_from_config(
        {"name": "prophet", "family": "prophet", "timestamp_column": "event_timestamp"},
        frame,
        timestamp_column="event_timestamp",
    )

    assert issubclass(TorchSequenceClassifier, object)
    assert issubclass(TorchSequenceEnsembleClassifier, object)
    assert isinstance(prophet, ProphetFrameClassifier)


def test_days_regression_candidate_builder_includes_real_non_linear_families():
    frame = pd.DataFrame(
        {
            "event_timestamp": pd.date_range("2026-01-01", periods=3),
            "current_inventory": [10.0, 8.0, 5.0],
            "blood_type": ["O+", "A+", "O-"],
        }
    )
    candidate_names = {}
    for family in ["xgboost", "random_forest", "lightgbm", "catboost", "prophet"]:
        estimator = _regression_estimator_from_config(
            {"name": family, "family": family, "timestamp_column": "event_timestamp"},
            frame,
            timestamp_column="event_timestamp",
        )
        candidate_names[family] = estimator.__class__.__name__

    assert candidate_names["xgboost"] == "Pipeline"
    assert candidate_names["random_forest"] == "Pipeline"
    assert candidate_names["lightgbm"] == "Pipeline"
    assert candidate_names["catboost"] == "CatBoostFrameRegressor"
    assert candidate_names["prophet"] == "ProphetFrameRegressor"
