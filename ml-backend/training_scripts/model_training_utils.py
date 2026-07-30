from __future__ import annotations

import json
import math
import os
import gc
import sys
import warnings
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Callable, Literal

# -----------------------------------------------------------------------------
# 1. SETUP SYSTEM PATHS
# -----------------------------------------------------------------------------
THIS_DIRECTORY = Path(__file__).resolve().parent
ROOT_DIRECTORY = THIS_DIRECTORY.parent
BACKEND_DIRECTORY = ROOT_DIRECTORY.parent / "backendMulti"

for _path in (THIS_DIRECTORY, ROOT_DIRECTORY, BACKEND_DIRECTORY):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from pios_ml_backend.training_resource_profiles import (  # noqa: E402
    apply_selected_profile_environment,
    nested_bool,
)

apply_selected_profile_environment(os.environ)

# -----------------------------------------------------------------------------
# 2. INITIALIZE DJANGO (MUST HAPPEN BEFORE IMPORTING DJANGO MODELS)
# -----------------------------------------------------------------------------
import django
from dotenv import load_dotenv

load_dotenv(BACKEND_DIRECTORY / ".env")
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "backendMulti.settings")
django.setup()

# -----------------------------------------------------------------------------
# 3. NOW IT IS SAFE TO IMPORT THIRD-PARTY LIBS & DJANGO MODELS
# -----------------------------------------------------------------------------
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
import seaborn as sns
from catboost import CatBoostClassifier, CatBoostRegressor
try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset
except Exception:  # pragma: no cover - exercised only when optional torch is absent
    torch = None
    nn = None
    DataLoader = None
    TensorDataset = None

# Django imports are now safe
from django.utils import timezone
from lightgbm import LGBMClassifier, LGBMRegressor
from ml.core.model_stats import prepare_model_stats_for_storage
from ml.models import ModelStats
from mlflow import MlflowClient
from mlflow.data.pandas_dataset import from_pandas
from mlflow.models import infer_signature
from sklearn.compose import ColumnTransformer
from sklearn.exceptions import ConvergenceWarning
from sklearn.impute import SimpleImputer
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_recall_curve,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
from xgboost import XGBClassifier, XGBRegressor

import bootstrap  # Keeping for the median_ci function

warnings.filterwarnings("ignore", category=ConvergenceWarning)
warnings.filterwarnings(
    "ignore",
    message="X does not have valid feature names, but LGBM",
    category=UserWarning,
)

DATASETS_DIRECTORY = ROOT_DIRECTORY / "datasets"
FEATURE_LABELS_DIRECTORY = ROOT_DIRECTORY / "datasets_featues_labels_seperated"
ARTIFACTS_DIRECTORY = ROOT_DIRECTORY / "artifacts" / "models"
DEFAULT_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:8889")
DEFAULT_FEATURE_LEAKAGE_GUARD_PATH = (
    ROOT_DIRECTORY / "config" / "feature_leakage_guard.json"
)
_FEATURE_LEAKAGE_GUARD_CACHE: tuple[Path, float, dict[str, Any]] | None = None


@dataclass(frozen=True)
class TaskDataset:
    task_name: str
    model_name: str
    experiment_name: str
    task_type: Literal["classification", "regression", "probability_regression"]
    frame: pd.DataFrame
    target_column: str
    timestamp_column: str
    feature_service: str
    entity_keys: tuple[str, ...]
    label_file: Path
    feature_file: Path
    description: str
    hard_mask_builder: Callable[[pd.DataFrame], pd.Series]
    accuracy_tolerance: float | None = None
    min_accuracy: float | None = None
    max_accuracy: float | None = None
    bounds: dict[str, Any] | None = None
    serving_feature_columns: tuple[str, ...] | None = None
    prediction_defaults: dict[str, Any] | None = None
    training_config: dict[str, Any] | None = None
    post_split_feature_fn: Callable[[pd.DataFrame], pd.DataFrame] | None = None


@dataclass(frozen=True)
class CandidateResult:
    candidate_name: str
    estimator: Any
    metrics: dict[str, float]
    hard_metrics: dict[str, float]
    validation_score: float
    predictions: np.ndarray
    probabilities: np.ndarray | None


@dataclass(frozen=True)
class TaskResult:
    dataset: TaskDataset
    candidate_results: list[CandidateResult]
    best_result: CandidateResult
    validation_result: CandidateResult | None
    feature_columns: list[str]
    feature_types: dict[str, str]
    train_frame: pd.DataFrame
    test_frame: pd.DataFrame


_TORCH_MODULE_BASE = nn.Module if nn is not None else object
_SEQUENCE_RECURRENT_CLASSES = (
    {
        "lstm": nn.LSTM,
        "gru": nn.GRU,
    }
    if nn is not None
    else {}
)


def _configure_torch_runtime() -> None:
    if torch is None:
        return
    thread_count = int(os.getenv("PIOS_TORCH_NUM_THREADS", "1"))
    thread_count = max(1, thread_count)
    try:
        torch.set_num_threads(thread_count)
    except Exception:
        pass
    try:
        torch.set_num_interop_threads(thread_count)
    except Exception:
        pass
    try:
        if hasattr(torch.backends, "mkldnn"):
            torch.backends.mkldnn.enabled = False
    except Exception:
        pass


class _TorchSequenceNetwork(_TORCH_MODULE_BASE):
    def __init__(
        self,
        *,
        architecture: str,
        input_size: int,
        hidden_size: int,
        num_layers: int,
        dropout: float,
    ) -> None:
        if nn is None:
            raise RuntimeError("PyTorch is required for sequence models.")
        super().__init__()
        try:
            recurrent_cls = _SEQUENCE_RECURRENT_CLASSES[architecture]
        except KeyError as exc:
            raise ValueError(f"Unsupported sequence architecture: {architecture}") from exc
        self.recurrent = recurrent_cls(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True,
        )
        self.head = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, max(8, hidden_size // 2)),
            nn.ReLU(),
            nn.Linear(max(8, hidden_size // 2), 1),
        )

    def forward(self, values):  # type: ignore[no-untyped-def]
        sequence_output, _state = self.recurrent(values)
        return self.head(sequence_output[:, -1, :]).squeeze(-1)


class TorchSequenceRegressor:
    def __init__(
        self,
        *,
        architecture: str,
        sequence_features: list[str] | tuple[str, ...],
        lag_steps: list[int] | tuple[int, ...],
        hidden_size: int = 64,
        num_layers: int = 2,
        dropout: float = 0.15,
        epochs: int = 20,
        batch_size: int = 256,
        learning_rate: float = 0.001,
        weight_decay: float = 0.0,
        patience: int = 5,
        random_state: int = 42,
        validation_fraction: float = 0.15,
        clip_min: float | None = 0.0,
        target_transform: str = "",
        calibration_method: str = "none",
    ) -> None:
        self.architecture = str(architecture).strip().lower()
        self.sequence_features = tuple(str(value) for value in sequence_features)
        self.lag_steps = tuple(sorted({int(value) for value in lag_steps}, reverse=True))
        self.hidden_size = int(hidden_size)
        self.num_layers = int(num_layers)
        self.dropout = float(dropout)
        self.epochs = int(epochs)
        self.batch_size = int(batch_size)
        self.learning_rate = float(learning_rate)
        self.weight_decay = float(weight_decay)
        self.patience = int(patience)
        self.random_state = int(random_state)
        self.validation_fraction = float(validation_fraction)
        self.clip_min = clip_min
        self.target_transform = str(target_transform or "").strip()
        method = str(calibration_method or "none").strip().lower()
        if method == "clip":
            method = "none"
        if method not in {"none", "isotonic", "sigmoid"}:
            raise ValueError(f"Unsupported sequence calibration method: {calibration_method}")
        self.calibration_method = method

    @staticmethod
    def _lag_column(feature: str, lag_step: int) -> str:
        return feature if lag_step == 0 else f"{feature}_lag_{lag_step}"

    def _raw_sequence(self, X: pd.DataFrame) -> np.ndarray:
        frame = pd.DataFrame(X).copy()
        step_arrays: list[np.ndarray] = []
        for lag_step in self.lag_steps:
            columns: list[np.ndarray] = []
            for feature in self.sequence_features:
                lag_column = self._lag_column(feature, lag_step)
                if lag_column in frame.columns:
                    values = frame[lag_column]
                elif feature in frame.columns:
                    values = frame[feature]
                else:
                    values = pd.Series(0.0, index=frame.index)
                columns.append(
                    pd.to_numeric(values, errors="coerce").to_numpy(dtype=np.float32)
                )
            step_arrays.append(np.column_stack(columns))
        return np.stack(step_arrays, axis=1).astype(np.float32)

    def _fit_sequence_scaler(self, sequence: np.ndarray) -> np.ndarray:
        flat = sequence.reshape(-1, sequence.shape[-1])
        medians = np.nanmedian(flat, axis=0)
        medians = np.where(np.isfinite(medians), medians, 0.0).astype(np.float32)
        filled = np.where(np.isnan(sequence), medians, sequence)
        flat_filled = filled.reshape(-1, filled.shape[-1])
        means = flat_filled.mean(axis=0).astype(np.float32)
        stds = flat_filled.std(axis=0).astype(np.float32)
        stds = np.where(stds > 1e-6, stds, 1.0).astype(np.float32)
        self.feature_medians_ = medians
        self.feature_means_ = means
        self.feature_stds_ = stds
        return ((filled - means) / stds).astype(np.float32)

    def _transform_sequence(self, X: pd.DataFrame) -> np.ndarray:
        sequence = self._raw_sequence(X)
        medians = getattr(self, "feature_medians_", None)
        means = getattr(self, "feature_means_", None)
        stds = getattr(self, "feature_stds_", None)
        if medians is None or means is None or stds is None:
            raise RuntimeError("Sequence model is not fitted.")
        filled = np.where(np.isnan(sequence), medians, sequence)
        return ((filled - means) / stds).astype(np.float32)

    def fit(self, X: pd.DataFrame, y: pd.Series | np.ndarray):  # type: ignore[no-untyped-def]
        if torch is None or nn is None or DataLoader is None or TensorDataset is None:
            raise RuntimeError("PyTorch is required for sequence models.")
        if self.architecture not in _SEQUENCE_RECURRENT_CLASSES:
            raise ValueError(f"Unsupported sequence architecture: {self.architecture}")
        _configure_torch_runtime()
        torch.manual_seed(self.random_state)
        np.random.seed(self.random_state)

        X_sequence = self._fit_sequence_scaler(self._raw_sequence(pd.DataFrame(X)))
        y_array = np.asarray(y, dtype=np.float32).reshape(-1)
        if self.target_transform == "log1p":
            y_training = np.log1p(np.clip(y_array, a_min=0.0, a_max=None)).astype(np.float32)
        elif self.target_transform:
            raise ValueError(f"Unsupported target transform: {self.target_transform}")
        else:
            y_training = y_array
        target_mean = float(np.nanmean(y_training)) if len(y_training) else 0.0
        target_std = float(np.nanstd(y_training)) if len(y_training) else 1.0
        target_std = target_std if target_std > 1e-6 else 1.0
        self.target_mean_ = target_mean
        self.target_std_ = target_std
        y_scaled = ((y_training - target_mean) / target_std).astype(np.float32)

        validation_count = max(1, int(len(X_sequence) * self.validation_fraction))
        validation_count = min(validation_count, max(1, len(X_sequence) - 1))
        train_count = len(X_sequence) - validation_count
        X_train = X_sequence[:train_count]
        y_train = y_scaled[:train_count]
        X_valid = X_sequence[train_count:]
        y_valid = y_scaled[train_count:]

        device = torch.device("cpu")
        network = _TorchSequenceNetwork(
            architecture=self.architecture,
            input_size=X_sequence.shape[-1],
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            dropout=self.dropout,
        ).to(device)
        optimizer = torch.optim.AdamW(
            network.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )
        loss_fn = nn.SmoothL1Loss()
        train_dataset = TensorDataset(
            torch.from_numpy(np.ascontiguousarray(X_train, dtype=np.float32)),
            torch.from_numpy(np.ascontiguousarray(y_train, dtype=np.float32)),
        )
        loader = DataLoader(
            train_dataset,
            batch_size=max(1, self.batch_size),
            shuffle=True,
        )
        valid_x = torch.from_numpy(
            np.ascontiguousarray(X_valid, dtype=np.float32)
        ).to(device)
        valid_y = torch.from_numpy(
            np.ascontiguousarray(y_valid, dtype=np.float32)
        ).to(device)

        best_loss = float("inf")
        best_state: dict[str, Any] | None = None
        stale_epochs = 0
        self.training_history_ = []
        for epoch in range(max(1, self.epochs)):
            network.train()
            batch_losses: list[float] = []
            for batch_x, batch_y in loader:
                batch_x = batch_x.to(device)
                batch_y = batch_y.to(device)
                optimizer.zero_grad()
                loss = loss_fn(network(batch_x), batch_y)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(network.parameters(), max_norm=2.0)
                optimizer.step()
                batch_losses.append(float(loss.detach().cpu()))
            network.eval()
            with torch.no_grad():
                valid_loss = float(loss_fn(network(valid_x), valid_y).detach().cpu())
            self.training_history_.append(
                {
                    "epoch": epoch + 1,
                    "train_loss": float(np.mean(batch_losses)) if batch_losses else 0.0,
                    "validation_loss": valid_loss,
                }
            )
            if valid_loss < best_loss - 1e-5:
                best_loss = valid_loss
                best_state = {
                    key: value.detach().cpu().clone()
                    for key, value in network.state_dict().items()
                }
                stale_epochs = 0
            else:
                stale_epochs += 1
                if stale_epochs >= self.patience:
                    break

        if best_state is not None:
            network.load_state_dict(best_state)
        network.to(torch.device("cpu"))
        network.eval()
        self.network_ = network
        self.n_features_in_ = len(self.sequence_features) * len(self.lag_steps)
        self.feature_names_in_ = np.asarray(
            [
                self._lag_column(feature, lag_step)
                for lag_step in self.lag_steps
                for feature in self.sequence_features
            ],
            dtype=object,
        )
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if torch is None:
            raise RuntimeError("PyTorch is required for sequence models.")
        network = getattr(self, "network_", None)
        if network is None:
            raise RuntimeError("Sequence model is not fitted.")
        X_sequence = self._transform_sequence(pd.DataFrame(X))
        predictions: list[np.ndarray] = []
        network.eval()
        with torch.no_grad():
            for start in range(0, len(X_sequence), max(1, self.batch_size)):
                batch = torch.from_numpy(
                    np.ascontiguousarray(
                        X_sequence[start : start + self.batch_size],
                        dtype=np.float32,
                    )
                )
                pred = network(batch).detach().cpu().numpy()
                predictions.append(pred)
        values = np.concatenate(predictions) if predictions else np.asarray([])
        values = values * float(self.target_std_) + float(self.target_mean_)
        if self.target_transform == "log1p":
            values = np.expm1(values)
        if self.clip_min is not None:
            values = np.maximum(values, float(self.clip_min))
        return values.astype(np.float64)

    def __getstate__(self) -> dict[str, Any]:
        network = getattr(self, "network_", None)
        if network is not None and hasattr(network, "to"):
            network.to(torch.device("cpu"))
        return dict(self.__dict__)


class TorchSequenceEnsembleRegressor:
    def __init__(
        self,
        *,
        estimator_configs: list[dict[str, Any]],
        weights: list[float] | None = None,
    ) -> None:
        self.estimator_configs = [dict(config) for config in estimator_configs]
        self.weights = list(weights) if weights is not None else None

    def fit(self, X: pd.DataFrame, y: pd.Series | np.ndarray):  # type: ignore[no-untyped-def]
        self.estimators_ = [
            TorchSequenceRegressor(**config).fit(X, y)
            for config in self.estimator_configs
        ]
        if self.weights is None or len(self.weights) != len(self.estimators_):
            self.weights_ = np.ones(len(self.estimators_), dtype=np.float64)
        else:
            self.weights_ = np.asarray(self.weights, dtype=np.float64)
        if not np.isfinite(self.weights_).all() or float(self.weights_.sum()) <= 0:
            self.weights_ = np.ones(len(self.estimators_), dtype=np.float64)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        estimators = getattr(self, "estimators_", None)
        if not estimators:
            raise RuntimeError("Sequence ensemble is not fitted.")
        predictions = np.vstack([estimator.predict(X) for estimator in estimators])
        weights = getattr(self, "weights_", np.ones(len(estimators), dtype=np.float64))
        return np.average(predictions, axis=0, weights=weights)


class TorchSequenceClassifier(TorchSequenceRegressor):
    def fit(self, X: pd.DataFrame, y: pd.Series | np.ndarray):  # type: ignore[no-untyped-def]
        if torch is None or nn is None or DataLoader is None or TensorDataset is None:
            raise RuntimeError("PyTorch is required for sequence models.")
        if self.architecture not in _SEQUENCE_RECURRENT_CLASSES:
            raise ValueError(f"Unsupported sequence architecture: {self.architecture}")
        _configure_torch_runtime()
        torch.manual_seed(self.random_state)
        np.random.seed(self.random_state)

        X_sequence = self._fit_sequence_scaler(self._raw_sequence(pd.DataFrame(X)))
        y_array = np.asarray(y, dtype=np.float32).reshape(-1)

        validation_count = max(1, int(len(X_sequence) * self.validation_fraction))
        validation_count = min(validation_count, max(1, len(X_sequence) - 1))
        train_count = len(X_sequence) - validation_count
        X_train = X_sequence[:train_count]
        y_train = y_array[:train_count]
        X_valid = X_sequence[train_count:]
        y_valid = y_array[train_count:]

        device = torch.device("cpu")
        network = _TorchSequenceNetwork(
            architecture=self.architecture,
            input_size=X_sequence.shape[-1],
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            dropout=self.dropout,
        ).to(device)
        optimizer = torch.optim.AdamW(
            network.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )
        positives = float(np.clip(y_train.sum(), 1.0, None))
        negatives = float(max(1.0, len(y_train) - positives))
        loss_fn = nn.BCEWithLogitsLoss(
            pos_weight=torch.tensor([negatives / positives], dtype=torch.float32)
        )
        train_dataset = TensorDataset(
            torch.from_numpy(np.ascontiguousarray(X_train, dtype=np.float32)),
            torch.from_numpy(np.ascontiguousarray(y_train, dtype=np.float32)),
        )
        loader = DataLoader(
            train_dataset,
            batch_size=max(1, self.batch_size),
            shuffle=True,
        )
        valid_x = torch.from_numpy(
            np.ascontiguousarray(X_valid, dtype=np.float32)
        ).to(device)
        valid_y = torch.from_numpy(
            np.ascontiguousarray(y_valid, dtype=np.float32)
        ).to(device)

        best_loss = float("inf")
        best_state: dict[str, Any] | None = None
        stale_epochs = 0
        self.training_history_ = []
        for epoch in range(max(1, self.epochs)):
            network.train()
            batch_losses: list[float] = []
            for batch_x, batch_y in loader:
                batch_x = batch_x.to(device)
                batch_y = batch_y.to(device)
                optimizer.zero_grad()
                loss = loss_fn(network(batch_x), batch_y)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(network.parameters(), max_norm=2.0)
                optimizer.step()
                batch_losses.append(float(loss.detach().cpu()))
            network.eval()
            with torch.no_grad():
                valid_loss = float(loss_fn(network(valid_x), valid_y).detach().cpu())
            self.training_history_.append(
                {
                    "epoch": epoch + 1,
                    "train_loss": float(np.mean(batch_losses)) if batch_losses else 0.0,
                    "validation_loss": valid_loss,
                }
            )
            if valid_loss < best_loss - 1e-5:
                best_loss = valid_loss
                best_state = {
                    key: value.detach().cpu().clone()
                    for key, value in network.state_dict().items()
                }
                stale_epochs = 0
            else:
                stale_epochs += 1
                if stale_epochs >= self.patience:
                    break

        if best_state is not None:
            network.load_state_dict(best_state)
        network.to(torch.device("cpu"))
        network.eval()
        with torch.no_grad():
            valid_logits = network(valid_x.to(torch.device("cpu"))).detach().cpu().numpy()
        raw_valid = 1.0 / (1.0 + np.exp(-valid_logits))
        self.calibrator_ = None
        if (
            self.calibration_method != "none"
            and len(np.unique(y_valid)) >= 2
            and len(np.unique(raw_valid)) >= 2
        ):
            if self.calibration_method == "isotonic":
                self.calibrator_ = IsotonicRegression(
                    y_min=0.0,
                    y_max=1.0,
                    out_of_bounds="clip",
                ).fit(raw_valid, y_valid)
            elif self.calibration_method == "sigmoid":
                self.calibrator_ = LogisticRegression(
                    solver="lbfgs",
                    max_iter=1000,
                ).fit(raw_valid.reshape(-1, 1), y_valid.astype(int))
        self.network_ = network
        self.n_features_in_ = len(self.sequence_features) * len(self.lag_steps)
        self.feature_names_in_ = np.asarray(
            [
                self._lag_column(feature, lag_step)
                for lag_step in self.lag_steps
                for feature in self.sequence_features
            ],
            dtype=object,
        )
        self.classes_ = np.asarray([0, 1])
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        if torch is None:
            raise RuntimeError("PyTorch is required for sequence models.")
        network = getattr(self, "network_", None)
        if network is None:
            raise RuntimeError("Sequence model is not fitted.")
        X_sequence = self._transform_sequence(pd.DataFrame(X))
        probabilities: list[np.ndarray] = []
        network.eval()
        with torch.no_grad():
            for start in range(0, len(X_sequence), max(1, self.batch_size)):
                batch = torch.from_numpy(
                    np.ascontiguousarray(
                        X_sequence[start : start + self.batch_size],
                        dtype=np.float32,
                    )
                )
                logits = network(batch).detach().cpu().numpy()
                probabilities.append(1.0 / (1.0 + np.exp(-logits)))
        positive = np.concatenate(probabilities) if probabilities else np.asarray([])
        positive = np.clip(positive.astype(np.float64), 0.0, 1.0)
        calibrator = getattr(self, "calibrator_", None)
        if calibrator is not None:
            if self.calibration_method == "sigmoid":
                positive = calibrator.predict_proba(positive.reshape(-1, 1))[:, 1]
            else:
                positive = calibrator.predict(positive)
            positive = np.clip(np.asarray(positive, dtype=np.float64), 0.0, 1.0)
        return np.column_stack([1.0 - positive, positive])

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)


class TorchSequenceEnsembleClassifier:
    def __init__(
        self,
        *,
        estimator_configs: list[dict[str, Any]],
        weights: list[float] | None = None,
    ) -> None:
        self.estimator_configs = [dict(config) for config in estimator_configs]
        self.weights = list(weights) if weights is not None else None

    def fit(self, X: pd.DataFrame, y: pd.Series | np.ndarray):  # type: ignore[no-untyped-def]
        self.estimators_ = [
            TorchSequenceClassifier(**config).fit(X, y)
            for config in self.estimator_configs
        ]
        if self.weights is None or len(self.weights) != len(self.estimators_):
            self.weights_ = np.ones(len(self.estimators_), dtype=np.float64)
        else:
            self.weights_ = np.asarray(self.weights, dtype=np.float64)
        if not np.isfinite(self.weights_).all() or float(self.weights_.sum()) <= 0:
            self.weights_ = np.ones(len(self.estimators_), dtype=np.float64)
        self.classes_ = np.asarray([0, 1])
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        estimators = getattr(self, "estimators_", None)
        if not estimators:
            raise RuntimeError("Sequence ensemble is not fitted.")
        probabilities = np.vstack(
            [estimator.predict_proba(X)[:, 1] for estimator in estimators]
        )
        weights = getattr(self, "weights_", np.ones(len(estimators), dtype=np.float64))
        positive = np.average(probabilities, axis=0, weights=weights)
        positive = np.clip(positive.astype(np.float64), 0.0, 1.0)
        return np.column_stack([1.0 - positive, positive])

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)


class CatBoostFrameClassifier:
    def __init__(self, **params: Any) -> None:
        self.params = dict(params)

    def fit(self, X: pd.DataFrame, y: pd.Series | np.ndarray):  # type: ignore[no-untyped-def]
        frame = pd.DataFrame(X).copy()
        self.categorical_columns_ = [
            column for column in frame.columns if not pd.api.types.is_numeric_dtype(frame[column])
        ]
        for column in self.categorical_columns_:
            frame[column] = frame[column].fillna("__MISSING__").astype(str)
        self.model_ = CatBoostClassifier(**self.params)
        self.model_.fit(frame, y, cat_features=self.categorical_columns_)
        self.classes_ = np.asarray([0, 1])
        return self

    def _prepare(self, X: pd.DataFrame) -> pd.DataFrame:
        frame = pd.DataFrame(X).copy()
        for column in getattr(self, "categorical_columns_", []):
            if column in frame.columns:
                frame[column] = frame[column].fillna("__MISSING__").astype(str)
        return frame

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        return self.model_.predict_proba(self._prepare(X))

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)


class CatBoostFrameRegressor:
    def __init__(self, **params: Any) -> None:
        self.params = dict(params)

    def fit(self, X: pd.DataFrame, y: pd.Series | np.ndarray):  # type: ignore[no-untyped-def]
        frame = pd.DataFrame(X).copy()
        self.categorical_columns_ = [
            column for column in frame.columns if not pd.api.types.is_numeric_dtype(frame[column])
        ]
        for column in self.categorical_columns_:
            frame[column] = frame[column].fillna("__MISSING__").astype(str)
        self.model_ = CatBoostRegressor(**self.params)
        self.model_.fit(frame, y, cat_features=self.categorical_columns_)
        return self

    def _prepare(self, X: pd.DataFrame) -> pd.DataFrame:
        frame = pd.DataFrame(X).copy()
        for column in getattr(self, "categorical_columns_", []):
            if column in frame.columns:
                frame[column] = frame[column].fillna("__MISSING__").astype(str)
        return frame

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return np.asarray(self.model_.predict(self._prepare(X)), dtype=np.float64)


class WeightedProbabilityEnsembleClassifier:
    def __init__(
        self,
        *,
        estimators: list[tuple[str, Any]],
        weights: list[float] | None = None,
    ) -> None:
        self.estimators = [(str(name), estimator) for name, estimator in estimators]
        self.weights = list(weights) if weights is not None else None

    def fit(self, X: pd.DataFrame, y: pd.Series | np.ndarray):  # type: ignore[no-untyped-def]
        self.estimators_ = []
        for name, estimator in self.estimators:
            fitted = estimator.fit(X, y)
            self.estimators_.append((name, fitted))
        if self.weights is None or len(self.weights) != len(self.estimators_):
            self.weights_ = np.ones(len(self.estimators_), dtype=np.float64)
        else:
            self.weights_ = np.asarray(self.weights, dtype=np.float64)
        if not np.isfinite(self.weights_).all() or float(self.weights_.sum()) <= 0:
            self.weights_ = np.ones(len(self.estimators_), dtype=np.float64)
        self.classes_ = np.asarray([0, 1])
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        estimators = getattr(self, "estimators_", None)
        if not estimators:
            raise RuntimeError("Probability ensemble is not fitted.")
        probabilities = np.vstack(
            [estimator.predict_proba(X)[:, 1] for _name, estimator in estimators]
        )
        weights = getattr(self, "weights_", np.ones(len(estimators), dtype=np.float64))
        positive = np.average(probabilities, axis=0, weights=weights)
        positive = np.clip(positive.astype(np.float64), 0.0, 1.0)
        return np.column_stack([1.0 - positive, positive])

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)


class ProphetFrameRegressor:
    def __init__(
        self,
        *,
        timestamp_column: str,
        target_transform: str = "",
        clip_min: float | None = 0.0,
        prophet_params: dict[str, Any] | None = None,
        regressors: list[str] | None = None,
    ) -> None:
        self.timestamp_column = str(timestamp_column)
        self.target_transform = str(target_transform or "").strip()
        self.clip_min = clip_min
        self.prophet_params = dict(prophet_params or {})
        self.regressors = [str(value) for value in regressors or []]

    def _prepare_frame(
        self,
        X: pd.DataFrame,
        y: pd.Series | np.ndarray | None = None,
        *,
        fitting: bool,
    ) -> pd.DataFrame:
        frame = pd.DataFrame(X).copy()
        if self.timestamp_column not in frame.columns:
            raise ValueError(
                f"Prophet candidate requires timestamp column '{self.timestamp_column}'."
            )
        prepared = pd.DataFrame(
            {
                "ds": pd.to_datetime(
                    frame[self.timestamp_column],
                    utc=True,
                    errors="coerce",
                ).dt.tz_localize(None)
            }
        )
        regressors = self.regressors or [
            column
            for column in frame.columns
            if column != self.timestamp_column and pd.api.types.is_numeric_dtype(frame[column])
        ]
        if fitting:
            self.regressors_ = list(regressors)
        else:
            regressors = list(getattr(self, "regressors_", regressors))
        for column in regressors:
            prepared[column] = _safe_numeric(
                frame[column] if column in frame.columns else pd.Series(0.0, index=frame.index)
            )
        if y is not None:
            values = np.asarray(y, dtype=np.float64)
            if self.target_transform == "log1p":
                values = np.log1p(np.clip(values, a_min=0.0, a_max=None))
            elif self.target_transform:
                raise ValueError(f"Unsupported target transform: {self.target_transform}")
            prepared["y"] = values
        return prepared.dropna(subset=["ds"])

    def fit(self, X: pd.DataFrame, y: pd.Series | np.ndarray):  # type: ignore[no-untyped-def]
        from prophet import Prophet

        prepared = self._prepare_frame(X, y, fitting=True)
        params = {
            "growth": "linear",
            "daily_seasonality": False,
            "weekly_seasonality": True,
            "yearly_seasonality": True,
            "uncertainty_samples": 0,
            **self.prophet_params,
        }
        model = Prophet(**params)
        for column in getattr(self, "regressors_", []):
            model.add_regressor(column)
        model.fit(prepared)
        self.model_ = model
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        prepared = self._prepare_frame(X, fitting=False)
        forecast = self.model_.predict(prepared[["ds", *getattr(self, "regressors_", [])]])
        values = np.asarray(forecast["yhat"], dtype=np.float64)
        if self.target_transform == "log1p":
            values = np.expm1(values)
        if self.clip_min is not None:
            values = np.maximum(values, float(self.clip_min))
        return values.astype(np.float64)


class ProphetFrameClassifier:
    def __init__(
        self,
        *,
        timestamp_column: str,
        prophet_params: dict[str, Any] | None = None,
        regressors: list[str] | None = None,
        calibration_method: str = "clip",
        calibration_features: list[str] | None = None,
    ) -> None:
        method = str(calibration_method or "clip").strip().lower()
        if method == "none":
            method = "clip"
        if method not in {"clip", "isotonic", "sigmoid"}:
            raise ValueError(f"Unsupported Prophet calibration method: {calibration_method}")
        self.calibration_method = method
        self.regressor = ProphetFrameRegressor(
            timestamp_column=timestamp_column,
            clip_min=None,
            prophet_params=prophet_params,
            regressors=regressors,
        )
        self.calibration_features = [str(value) for value in calibration_features or []]

    def _calibration_frame(
        self,
        X: pd.DataFrame,
        raw_probability: np.ndarray,
        *,
        fitting: bool,
    ) -> pd.DataFrame:
        frame = pd.DataFrame({"prophet_probability": raw_probability})
        source = pd.DataFrame(X).reset_index(drop=True)
        defaults = {} if fitting else dict(getattr(self, "calibration_defaults_", {}))
        for column in self.calibration_features:
            values = _safe_numeric(
                source[column] if column in source.columns else pd.Series(np.nan, index=source.index),
                fill_value=np.nan,
            )
            if fitting:
                median = values.dropna().median()
                defaults[column] = float(median) if pd.notna(median) else 0.0
            frame[column] = values.fillna(float(defaults.get(column, 0.0)))
        if fitting:
            self.calibration_defaults_ = defaults
        return frame

    def fit(self, X: pd.DataFrame, y: pd.Series | np.ndarray):  # type: ignore[no-untyped-def]
        labels = np.asarray(y, dtype=np.float64)
        self.regressor.fit(X, labels)
        self.classes_ = np.asarray([0, 1])
        self.calibrator_ = None
        raw_probability = np.clip(self.regressor.predict(X), 0.0, 1.0)
        calibration_frame = self._calibration_frame(X, raw_probability, fitting=True)
        can_calibrate = len(np.unique(labels)) >= 2 and (
            len(np.unique(raw_probability)) >= 2
            or bool(self.calibration_features)
        )
        if can_calibrate:
            if self.calibration_method == "isotonic":
                self.calibrator_ = IsotonicRegression(
                    y_min=0.0,
                    y_max=1.0,
                    out_of_bounds="clip",
                ).fit(raw_probability, labels)
            elif self.calibration_method == "sigmoid":
                self.calibrator_ = Pipeline(
                    [
                        ("scaler", StandardScaler()),
                        (
                            "model",
                            LogisticRegression(
                                solver="lbfgs",
                                max_iter=2000,
                            ),
                        ),
                    ]
                ).fit(calibration_frame, labels.astype(int))
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        positive = np.clip(self.regressor.predict(X), 0.0, 1.0)
        calibrator = getattr(self, "calibrator_", None)
        if calibrator is not None:
            if self.calibration_method == "sigmoid":
                positive = calibrator.predict_proba(
                    self._calibration_frame(X, positive, fitting=False)
                )[:, 1]
            else:
                positive = calibrator.predict(positive)
            positive = np.clip(np.asarray(positive, dtype=np.float64), 0.0, 1.0)
        return np.column_stack([1.0 - positive, positive])

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)


def configure_mlflow() -> None:
    mlflow.set_tracking_uri(DEFAULT_TRACKING_URI)


def _deep_merge_dicts(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge_dicts(merged[key], value)
        else:
            merged[key] = value
    return merged


def _ensure_output_dir(task_name: str) -> Path:
    output_dir = ARTIFACTS_DIRECTORY / task_name
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def _safe_numeric(series: pd.Series, fill_value: float = 0.0) -> pd.Series:
    return pd.to_numeric(series, errors="coerce").fillna(fill_value)


def _quantile_mask(series: pd.Series, quantile: float, direction: str) -> pd.Series:
    numeric = _safe_numeric(series, fill_value=np.nan)
    threshold = numeric.quantile(quantile)
    if pd.isna(threshold):
        return pd.Series(False, index=series.index)
    if direction == "high":
        return numeric >= threshold
    return numeric <= threshold


def _classification_metrics(
    y_true: pd.Series | np.ndarray,
    probabilities: np.ndarray,
    predictions: np.ndarray,
) -> dict[str, float]:
    y_array = np.asarray(y_true, dtype=np.int64)
    pred_array = np.asarray(predictions, dtype=np.int64)
    prob_array = np.asarray(probabilities, dtype=np.float64)
    if prob_array.ndim != 1:
        prob_array = prob_array.reshape(-1)

    metrics = {
        "accuracy": float(accuracy_score(y_array, pred_array)),
        "precision": float(precision_score(y_array, pred_array, zero_division=0)),
        "recall": float(recall_score(y_array, pred_array, zero_division=0)),
        "f1_score": float(f1_score(y_array, pred_array, zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(y_array, pred_array)),
        "average_precision": float(average_precision_score(y_array, prob_array)),
        "brier_score": float(brier_score_loss(y_array, prob_array)),
    }
    if len(np.unique(y_array)) > 1:
        metrics["roc_auc"] = float(roc_auc_score(y_array, prob_array))
    else:
        metrics["roc_auc"] = 0.5
    return metrics


def _regression_metrics(
    y_true: pd.Series | np.ndarray,
    predictions: np.ndarray,
    *,
    accuracy_tolerance: float | None = None,
) -> dict[str, float]:
    y_array = np.asarray(y_true, dtype=np.float64)
    pred_array = np.asarray(predictions, dtype=np.float64)
    nan_mask = np.isnan(y_array) | np.isnan(pred_array)
    if nan_mask.any():
        y_array = y_array[~nan_mask]
        pred_array = pred_array[~nan_mask]
    if len(y_array) == 0:
        return {
            "root_mean_squared_error": float("nan"),
            "mean_absolute_error": float("nan"),
            "r2_score": float("nan"),
            "mean_absolute_percentage_error": float("nan"),
            "wape": float("nan"),
            "mase": float("nan"),
            "rmsse": float("nan"),
            "within_tolerance_rate": float("nan"),
            "accuracy": float("nan"),
        }
    rmse = math.sqrt(float(mean_squared_error(y_array, pred_array)))
    bias = np.mean(y_array - pred_array)
    metrics = {
        "root_mean_squared_error": rmse,
        "mean_absolute_error": float(mean_absolute_error(y_array, pred_array)),
        "r2_score": float(r2_score(y_array, pred_array)),
        "bias": float(bias),

    }
    
    # WMAPE: sum(|actual - pred|) / sum(|actual|)
    # Robust to near-zero and zero actuals — safe for depleted blood stock rows.
    total_actual = float(np.sum(np.abs(y_array)))
    metrics["wmape"] = float(
        np.sum(np.abs(y_array - pred_array)) / max(total_actual, 1e-6)
    )
    if accuracy_tolerance is not None:
        metrics["accuracy"] = float(
            np.mean(np.abs(y_array - pred_array) <= accuracy_tolerance)
        )
    return metrics

def _delta_sign_metrics(y_true, y_pred):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    # real depletion
    real_dep = y_true < 0

    # predicted depletion
    pred_dep = y_pred < 0

    # counts
    tp = float(np.sum(real_dep & pred_dep))   # caught
    fn = float(np.sum(real_dep & (~pred_dep)))  # missed ❗
    fp = float(np.sum((~real_dep) & pred_dep))  # false alarm

    # safe division
    recall = tp / (tp + fn) if (tp + fn) > 0 else float("nan")
    precision = tp / (tp + fp) if (tp + fp) > 0 else float("nan")

    return {
        "depletion_recall": float(recall),       # MOST IMPORTANT
        "depletion_precision": float(precision),
        "tp": float(tp),
        "fn": float(fn),
        "fp": float(fp),
    }
def _ranking_score(
    task_type: Literal["classification", "regression"],
    metrics: dict[str, float],
    hard_metrics: dict[str, float],
) -> float:
    if task_type == "classification":
        return (
            0.45 * metrics.get("accuracy", 0.0)
            + 0.25 * hard_metrics.get("accuracy", 0.0)
            + 0.15 * metrics.get("balanced_accuracy", 0.0)
            + 0.15 * hard_metrics.get("balanced_accuracy", 0.0)
        )

    return (
        0.50 * metrics.get("accuracy", 0.0)
        + 0.30 * hard_metrics.get("accuracy", 0.0)
        - 0.15 * metrics.get("root_mean_squared_error", 0.0)
        - 0.05 * hard_metrics.get("mean_absolute_error", 0.0)
        - 0.10 * abs(metrics.get("bias", 0.0))
    )


def _feature_stats_payload(
    frame: pd.DataFrame, feature_names: list[str]
) -> dict[str, Any]:
    stats: dict[str, Any] = {}
    for column in feature_names:
        series = frame[column]
        if pd.api.types.is_numeric_dtype(series) or pd.api.types.is_bool_dtype(series):
            numeric = _safe_numeric(series, fill_value=np.nan).dropna()
            if numeric.empty:
                stats[column] = {
                    "median": 0.0,
                    "mean": 0.0,
                    "std": 0.0,
                    "lower": 0.0,
                    "upper": 0.0,
                    "default": 0.0,
                }
                continue
            quantiles = numeric.quantile([0.025, 0.975]).to_numpy()
            stats[column] = {
                "median": float(numeric.median()),
                "mean": float(numeric.mean()),
                "std": float(numeric.std(ddof=0)) if len(numeric) > 1 else 0.0,
                "lower": float(quantiles[0]),
                "upper": float(quantiles[1]),
                "default": float(numeric.median()),
            }
            continue

        text = series.fillna("").astype(str).str.strip()
        text = text.replace({"nan": "", "None": ""})
        non_empty = text[text != ""]
        mode = str(non_empty.mode().iloc[0]) if not non_empty.mode().empty else ""
        stats[column] = {
            "mode": mode,
            "default": mode,
            "unique_values": int(non_empty.nunique()),
            "missing_rate": float((text == "").mean()),
        }
    return stats


def _persist_model_stats(
    model_id: str, identifier: str, payload: dict[str, Any]
) -> None:
    try:
        ModelStats.objects.update_or_create(
            model_id=model_id,
            defaults={
                "identifier": identifier,
                "stats": prepare_model_stats_for_storage(payload),
                "made_at": timezone.now(),
            },
        )
    except Exception as exc:
        print(f"[model_training_utils] ModelStats persistence skipped: {exc}")


def _save_parquet_pair(
    frame: pd.DataFrame, feature_path: Path, label_path: Path, target: str
) -> None:
    feature_path.parent.mkdir(parents=True, exist_ok=True)
    label_path.parent.mkdir(parents=True, exist_ok=True)
    feature_columns = [
        column
        for column in frame.columns
        if column != target and not _is_future_or_label_column(column)
    ]
    frame[feature_columns].to_parquet(feature_path, index=False)
    frame[
        [
            col
            for col in frame.columns
            if col
            in {
                "donor_id",
                "hospital_id",
                "supply_id",
                "blood_type",
                "component_type",
                "event_timestamp",
                target,
            }
        ]
    ].to_parquet(
        label_path,
        index=False,
    )


def _feature_leakage_guard_path() -> Path:
    configured = os.getenv("PIOS_FEATURE_LEAKAGE_GUARD_CONFIG", "").strip()
    if configured:
        return Path(configured).expanduser().resolve()
    return DEFAULT_FEATURE_LEAKAGE_GUARD_PATH


def _feature_leakage_guard_config() -> dict[str, Any]:
    global _FEATURE_LEAKAGE_GUARD_CACHE
    path = _feature_leakage_guard_path()
    try:
        mtime = path.stat().st_mtime
    except OSError:
        raise ValueError(f"Feature leakage guard config is missing: {path}")
    if (
        _FEATURE_LEAKAGE_GUARD_CACHE is not None
        and _FEATURE_LEAKAGE_GUARD_CACHE[0] == path
        and _FEATURE_LEAKAGE_GUARD_CACHE[1] == mtime
    ):
        return _FEATURE_LEAKAGE_GUARD_CACHE[2]
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Invalid feature leakage guard config: {path}") from exc
    if not isinstance(payload, dict):
        raise ValueError("Feature leakage guard config root must be an object.")
    _FEATURE_LEAKAGE_GUARD_CACHE = (path, mtime, payload)
    return payload


def _guard_values(config: dict[str, Any], key: str) -> list[str]:
    values = config.get(key, [])
    if not isinstance(values, list):
        raise ValueError(f"Feature leakage guard {key} must be a list.")
    return [str(value).strip().lower() for value in values if str(value).strip()]


def _is_future_or_label_column(column: str) -> bool:
    lowered = column.strip().lower()
    config = _feature_leakage_guard_config()
    if lowered in set(_guard_values(config, "blocked_exact")):
        return True
    if any(
        lowered.startswith(prefix)
        for prefix in _guard_values(config, "blocked_prefixes")
    ):
        return True
    if any(
        lowered.endswith(suffix)
        for suffix in _guard_values(config, "blocked_suffixes")
    ):
        return True
    if any(
        substring in lowered
        for substring in _guard_values(config, "blocked_substrings")
    ):
        return True
    return False


def _feature_columns_for_dataset(
    frame: pd.DataFrame, dataset: TaskDataset
) -> list[str]:
    def is_identifier_column(column: str) -> bool:
        lowered = str(column or "").strip().lower()
        return lowered == "id" or lowered.endswith("_id")

    if dataset.serving_feature_columns is not None:
        return [
            column
            for column in dataset.serving_feature_columns
            if column in frame.columns
            and not is_identifier_column(column)
            and not _is_future_or_label_column(column)
        ]

    excluded = {dataset.target_column, dataset.timestamp_column, *dataset.entity_keys}
    return [
        column
        for column in frame.columns
        if column not in excluded
        and not is_identifier_column(column)
        and not _is_future_or_label_column(column)
    ]


def _days_until_next_group_event(
    frame: pd.DataFrame,
    *,
    group_column: str | list[str] | tuple[str, ...],
    timestamp_column: str,
    event_column: str,
) -> pd.Series:
    values = pd.Series(index=frame.index, dtype="float64")
    group_keys = (
        list(group_column) if isinstance(group_column, (list, tuple)) else group_column
    )
    for _, group in frame.groupby(group_keys, sort=False):
        timestamps = pd.to_datetime(
            group[timestamp_column],
            utc=True,
            errors="coerce",
        )
        event_mask = _safe_numeric(group[event_column]) > 0
        event_times = (
            timestamps[event_mask & timestamps.notna()]
            .sort_values()
            .map(lambda value: value.value)
            .to_numpy(dtype=np.int64)
        )
        panel_end = timestamps.max()
        panel_end_ns = panel_end.value if pd.notna(panel_end) else None
        group_values: list[float] = []
        for timestamp in timestamps:
            if pd.isna(timestamp) or panel_end_ns is None:
                group_values.append(np.nan)
                continue
            timestamp_ns = timestamp.value
            if len(event_times):
                event_index = np.searchsorted(
                    event_times,
                    timestamp_ns,
                    side="left",
                )
            else:
                event_index = 0
            if len(event_times) and event_index < len(event_times):
                seconds = (
                    float(event_times[event_index] - timestamp_ns) / 1_000_000_000.0
                )
                days = seconds / 86400.0
            else:
                days = max(0.0, (panel_end - timestamp).total_seconds() / 86400.0) + 1.0
            group_values.append(float(max(0.0, days)))
        values.loc[group.index] = group_values

    fallback = values.dropna().median()
    if pd.isna(fallback):
        fallback = 0.0
    return values.fillna(float(fallback))


def _normalize_hospital_component_supply_frame(frame: pd.DataFrame) -> pd.DataFrame:
    normalized = frame.copy()

    if "blood_shortage" not in normalized.columns:
        if "critical_stock" in normalized.columns:
            normalized["blood_shortage"] = normalized["critical_stock"]
        elif "stockout_next_day" in normalized.columns:
            normalized["blood_shortage"] = normalized["stockout_next_day"]

    rename_map = {
        "date": "event_timestamp",
        "temp_c": "temperature",
        "supply_shock": "disaster",
        "stock_end": "current_inventory",
    }
    normalized = normalized.rename(
        columns={k: v for k, v in rename_map.items() if k in normalized.columns}
    )

    # 100% Safe fix for InvalidIndexError: Remove any duplicate columns that snuck in
    normalized = normalized.loc[:, ~normalized.columns.duplicated()].copy()

    required = {
        "hospital_id",
        "event_timestamp",
        "blood_type",
        "temperature",
        "rain_mm",
        "holiday",
        "disaster",
        "scheduled_surgeries",
        "trauma_cases",
        "current_inventory",
    }
    missing = required - set(normalized.columns)
    if missing:
        raise ValueError(
            f"Hospital component supply frame is missing columns: {sorted(missing)}"
        )
    if "blood_shortage" not in normalized.columns:
        normalized["blood_shortage"] = (
            _safe_numeric(normalized["current_inventory"]) <= 0
        ).astype(int)

    normalized["event_timestamp"] = pd.to_datetime(
        normalized["event_timestamp"],
        utc=True,
        errors="coerce",
    )
    normalized["hospital_id"] = normalized["hospital_id"].astype(str).str.strip()
    normalized["blood_type"] = (
        normalized["blood_type"].astype(str).str.strip().str.upper()
    )
    normalized = normalized[
        normalized["hospital_id"].ne("")
        & normalized["blood_type"].ne("")
        & normalized["event_timestamp"].notna()
    ].copy()
    return normalized


def _component_profile_from_source(source: pd.DataFrame) -> pd.DataFrame:
    source = _normalize_hospital_component_supply_frame(source)
    global_inventory = max(
        float(_safe_numeric(source["current_inventory"]).median()), 1.0
    )

    # 100% safe explicit grouping and renaming (Bypasses all pandas version quirks)
    med_inv = source.groupby("blood_type", as_index=False)["current_inventory"].median()
    med_inv = med_inv.rename(
        columns={"current_inventory": "component_inventory_median"}
    )

    mean_short = source.groupby("blood_type", as_index=False)["blood_shortage"].mean()
    mean_short = mean_short.rename(
        columns={"blood_shortage": "component_shortage_rate"}
    )

    profile = pd.merge(med_inv, mean_short, on="blood_type")

    profile["component_inventory_factor"] = (
        _safe_numeric(profile["component_inventory_median"]) / global_inventory
    ).clip(lower=0.25, upper=2.5)

    return profile[
        ["blood_type", "component_inventory_factor", "component_shortage_rate"]
    ]


def load_hospital_component_supply_frame() -> pd.DataFrame:
    feature_source = FEATURE_LABELS_DIRECTORY / "hospital_supply_features.parquet"
    label_source = FEATURE_LABELS_DIRECTORY / "hospital_supply_labels.parquet"
    component_source = DATASETS_DIRECTORY / "synthetic_bloodbank_daily.csv"

    frames: list[pd.DataFrame] = []
    if component_source.exists():
        source_frame = pd.read_csv(component_source)
        frames.append(_normalize_hospital_component_supply_frame(source_frame))

    if feature_source.exists() and label_source.exists():
        features = pd.read_parquet(feature_source).copy()
        labels = pd.read_parquet(label_source).copy()
        merge_keys = ["hospital_id", "event_timestamp"]
        if "blood_type" in features.columns and "blood_type" in labels.columns:
            merge_keys.append("blood_type")
        base = features.merge(labels, on=merge_keys, how="left")
        if "blood_type" in base.columns:
            frames.append(_normalize_hospital_component_supply_frame(base))
        elif frames:
            profile = _component_profile_from_source(frames[0])
            base = _normalize_hospital_component_supply_frame(
                base.assign(blood_type="__PLACEHOLDER__")
            ).drop(columns=["blood_type"])
            expanded = base.merge(profile, how="cross")
            expanded["current_inventory"] = (
                (
                    _safe_numeric(expanded["current_inventory"])
                    * _safe_numeric(expanded["component_inventory_factor"])
                )
                .round()
                .clip(lower=0)
            )
            expanded["blood_shortage"] = (
                _safe_numeric(expanded["current_inventory"])
                <= expanded.groupby("blood_type")["current_inventory"].transform(
                    lambda series: _safe_numeric(series).quantile(0.15)
                )
            ).astype(int)
            frames.append(
                expanded.drop(
                    columns=[
                        "component_inventory_factor",
                        "component_shortage_rate",
                    ],
                    errors="ignore",
                )
            )

    if not frames:
        raise FileNotFoundError(
            "No hospital component supply data found in hospital_supply parquet or synthetic_bloodbank_daily.csv."
        )

    # Secondary safeguard for pd.concat
    frames = [f.loc[:, ~f.columns.duplicated()] for f in frames]

    combined = pd.concat(frames, ignore_index=True, sort=False)
    combined = combined.drop_duplicates(
        subset=["hospital_id", "blood_type", "event_timestamp"],
        keep="last",
    )
    combined = combined.sort_values(
        ["hospital_id", "blood_type", "event_timestamp"]
    ).reset_index(drop=True)
    return combined


def _build_donor_features_base() -> pd.DataFrame:
    registry_frame = pd.read_csv(
        DATASETS_DIRECTORY
        / "blood_registry_sythentic"
        / "data"
        / "blood_donation_registry_ml_ready.csv"
    ).copy()
    logistics_frame = pd.read_csv(
        DATASETS_DIRECTORY / "blood_donor_logistics_extended.csv"
    ).copy()
    snapshot_frame = pd.read_csv(
        DATASETS_DIRECTORY / "donor_snapshots_with_targets.csv"
    ).copy()

    registry_frame["donor_id"] = registry_frame["donor_id"].astype(str)
    logistics_frame["donor_id"] = logistics_frame["donor_id"].astype(str)
    snapshot_frame["donor_id"] = snapshot_frame["donor_id"].astype(str)
    logistics_frame = logistics_frame.drop(columns=["country_code"], errors="ignore")
    snapshot_frame = snapshot_frame.rename(
        columns={
            "recency_days": "snapshot_recency_days",
            "typical_interval_hidden": "snapshot_typical_interval_hidden",
            "frequency_365": "snapshot_frequency_365",
            "time_months": "snapshot_time_months",
            "days_until_eligible": "snapshot_days_until_eligible",
            "overdue_days": "snapshot_overdue_days",
            "days_to_next_donation": "snapshot_days_to_next_donation",
            "y_donate_30d": "snapshot_y_donate_30d",
        }
    )

    feature_frame = (
        registry_frame.merge(logistics_frame, on="donor_id", how="left")
        .merge(snapshot_frame, on="donor_id", how="left")
        .copy()
    )
    feature_frame["event_timestamp"] = pd.to_datetime(
        feature_frame["last_donation_date"], errors="coerce"
    )
    feature_frame["days_since_last_donation_derived"] = _safe_numeric(
        feature_frame["recency_days"]
    )
    feature_frame["donation_velocity"] = _safe_numeric(
        feature_frame["donation_count_last_12m"]
    ) / (1.0 + _safe_numeric(feature_frame["years_since_first_donation"]))
    feature_frame["readiness_score"] = (
        1.4 * (_safe_numeric(feature_frame["eligible_to_donate"]) > 0).astype(float)
        + 0.8
        * (
            _safe_numeric(feature_frame["snapshot_days_until_eligible"], 999.0) <= 0
        ).astype(float)
        + 0.45
        * (_safe_numeric(feature_frame["donation_count_last_12m"]) >= 1).astype(float)
        + 0.35 * (_safe_numeric(feature_frame["is_regular_donor"]) > 0).astype(float)
        - 0.30
        * (_safe_numeric(feature_frame["chronic_condition_flag"]) > 0).astype(float)
    )
    feature_frame["mobility_score"] = (
        1.2 * (_safe_numeric(feature_frame["route_feasibility_flag"]) > 0).astype(float)
        + 0.6
        * (_safe_numeric(feature_frame["travel_time_min"], 999.0) <= 25).astype(float)
        + 0.4
        * (_safe_numeric(feature_frame["center_distance_km"], 999.0) <= 12).astype(
            float
        )
        + 0.2 * _safe_numeric(feature_frame["cold_chain_quality_score"])
    )
    feature_frame["rare_type_priority_score"] = (
        1.0 + 1.5 * (_safe_numeric(feature_frame["is_rare_type"]) > 0).astype(float)
    ) * (1.0 - _safe_numeric(feature_frame["blood_type_country_prevalence"]))
    feature_frame["recency_eligibility_gap"] = np.maximum(
        _safe_numeric(feature_frame["recency_days"])
        - _safe_numeric(feature_frame["snapshot_days_until_eligible"]),
        0.0,
    )
    feature_frame["travel_burden_score"] = _safe_numeric(
        feature_frame["travel_time_min"]
    ) * (1.0 + _safe_numeric(feature_frame["center_distance_km"]) / 25.0)
    feature_frame["age_bucket"] = pd.cut(
        _safe_numeric(feature_frame["age"]),
        bins=[17, 24, 34, 44, 54, 64, 80],
        labels=["18-24", "25-34", "35-44", "45-54", "55-64", "65+"],
        include_lowest=True,
    ).astype(str)
    feature_frame["bmi_margin"] = (_safe_numeric(feature_frame["bmi"]) - 24.0).abs()
    feature_frame["commitment_score"] = (
        _safe_numeric(feature_frame["donation_count_last_12m"])
        + _safe_numeric(feature_frame["is_regular_donor"]) * 2.5
        + _safe_numeric(feature_frame["donation_velocity"]) * 3.0
    )
    feature_frame["friction_adjusted_commitment"] = feature_frame[
        "commitment_score"
    ] / (1.0 + _safe_numeric(feature_frame["travel_time_min"]))
    feature_frame["eligibility_buffer"] = _safe_numeric(
        feature_frame["recency_eligibility_gap"]
    )
    feature_frame["rare_mobility_synergy"] = _safe_numeric(
        feature_frame["rare_type_priority_score"]
    ) * (1.0 + _safe_numeric(feature_frame["mobility_score"]))
    feature_frame["response_readiness_index"] = (
        _safe_numeric(feature_frame["readiness_score"]) * 0.45
        + _safe_numeric(feature_frame["mobility_score"]) * 0.20
        + _safe_numeric(feature_frame["commitment_score"]) * 0.20
        + _safe_numeric(feature_frame["rare_mobility_synergy"]) * 0.15
    )
    feature_frame["distance_risk_band"] = pd.cut(
        _safe_numeric(feature_frame["center_distance_km"]),
        bins=[-1, 5, 15, 30, 60, 999],
        labels=["local", "city", "regional", "remote", "extreme"],
        include_lowest=True,
    ).astype(str)
    return feature_frame


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


def _build_ordinal_preprocessor(
    X_train: pd.DataFrame,
    *,
    scale_numeric: bool,
) -> ColumnTransformer:
    numeric_features = [
        column
        for column in X_train.columns
        if pd.api.types.is_numeric_dtype(X_train[column])
        or pd.api.types.is_bool_dtype(X_train[column])
    ]
    categorical_features = [
        column for column in X_train.columns if column not in numeric_features
    ]

    transformers: list[tuple[str, Pipeline, list[str]]] = []
    if numeric_features:
        numeric_steps: list[tuple[str, Any]] = [
            ("imputer", SimpleImputer(strategy="median")),
        ]
        if scale_numeric:
            numeric_steps.append(("scaler", StandardScaler()))
        transformers.append(
            (
                "numeric",
                Pipeline(steps=numeric_steps),
                numeric_features,
            )
        )
    if categorical_features:
        transformers.append(
            (
                "categorical",
                Pipeline(
                    steps=[
                        (
                            "imputer",
                            SimpleImputer(
                                strategy="constant", fill_value="__MISSING__"
                            ),
                        ),
                        (
                            "encoder",
                            OrdinalEncoder(
                                handle_unknown="use_encoded_value",
                                unknown_value=-1,
                                encoded_missing_value=-1,
                            ),
                        ),
                    ]
                ),
                categorical_features,
            )
        )
    return ColumnTransformer(
        transformers=transformers,
        remainder="drop",
        verbose_feature_names_out=False,
    )


def _candidate_rows(
    training_config: dict[str, Any],
    key: str,
    defaults: list[dict[str, Any]],
    include_key: str,
) -> list[dict[str, Any]]:
    configured = training_config.get(key)
    if isinstance(configured, list):
        rows = [dict(row) for row in configured if isinstance(row, dict)]
    elif bool(training_config.get(include_key, True)):
        rows = [dict(row) for row in defaults]
    else:
        rows = []
    return [row for row in rows if row.get("enabled", True) is not False]


def _candidate_name(row: dict[str, Any]) -> str:
    return str(row.get("name") or row.get("family") or row.get("type") or "").strip()


def _candidate_family(row: dict[str, Any]) -> str:
    return str(row.get("family") or row.get("type") or row.get("name") or "").strip().lower()


def _candidate_params(row: dict[str, Any]) -> dict[str, Any]:
    params = row.get("params")
    return dict(params) if isinstance(params, dict) else {}


def _profiled_candidate_params(
    family: str,
    params: dict[str, Any],
    training_config: dict[str, Any] | None,
) -> dict[str, Any]:
    if not isinstance(training_config, dict):
        return params
    resource = training_config.get("resource")
    if not isinstance(resource, dict):
        return params
    env_var = str(resource.get("thread_env_var") or "").strip()
    raw_threads = os.getenv(env_var, "") if env_var else ""
    if not raw_threads:
        raw_threads = resource.get("tree_model_threads")
    if raw_threads in (None, ""):
        return params
    try:
        thread_count = max(1, int(raw_threads))
    except (TypeError, ValueError):
        return params
    param_map = resource.get("candidate_thread_params")
    if not isinstance(param_map, dict):
        return params
    param_names = param_map.get(family)
    if isinstance(param_names, str):
        param_names = [param_names]
    if not isinstance(param_names, list):
        return params
    override_existing = bool(resource.get("override_configured_thread_params", False))
    next_params = dict(params)
    for raw_name in param_names:
        name = str(raw_name or "").strip()
        if not name:
            continue
        if override_existing or name not in next_params:
            next_params[name] = thread_count
    return next_params


def _tabular_pipeline(
    X_train: pd.DataFrame,
    estimator: Any,
    *,
    scale_numeric: bool = False,
) -> Pipeline:
    return Pipeline(
        steps=[
            ("preprocessor", _build_ordinal_preprocessor(X_train, scale_numeric=scale_numeric)),
            ("model", estimator),
        ]
    )


def _classification_estimator_from_config(
    row: dict[str, Any],
    X_train: pd.DataFrame,
    *,
    timestamp_column: str,
    training_config: dict[str, Any] | None = None,
) -> Any:
    family = _candidate_family(row)
    params = _profiled_candidate_params(
        family,
        _candidate_params(row),
        training_config,
    )
    if family == "catboost":
        defaults = {
            "iterations": 200,
            "depth": 7,
            "learning_rate": 0.03,
            "loss_function": "Logloss",
            "eval_metric": "AUC",
            "random_seed": 42,
            "verbose": False,
            "auto_class_weights": "Balanced",
        }
        return CatBoostFrameClassifier(**{**defaults, **params})
    if family == "xgboost":
        defaults = {
            "n_estimators": 200,
            "max_depth": 6,
            "learning_rate": 0.03,
            "subsample": 0.85,
            "colsample_bytree": 0.75,
            "reg_alpha": 0.15,
            "reg_lambda": 1.2,
            "eval_metric": "logloss",
            "random_state": 42,
        }
        return _tabular_pipeline(X_train, XGBClassifier(**{**defaults, **params}))
    if family == "lightgbm":
        defaults = {
            "n_estimators": 150,
            "learning_rate": 0.03,
            "num_leaves": 31,
            "subsample": 0.85,
            "colsample_bytree": 0.75,
            "class_weight": "balanced",
            "n_jobs": 4,
            "random_state": 42,
            "verbosity": -1,
        }
        return _tabular_pipeline(X_train, LGBMClassifier(**{**defaults, **params}))
    if family == "random_forest":
        defaults = {
            "n_estimators": 200,
            "max_depth": None,
            "min_samples_leaf": 2,
            "class_weight": "balanced_subsample",
            "n_jobs": 4,
            "random_state": 42,
        }
        return _tabular_pipeline(X_train, RandomForestClassifier(**{**defaults, **params}))
    if family == "logistic_regression":
        defaults = {
            "max_iter": 4000,
            "class_weight": "balanced",
            "solver": "lbfgs",
            "random_state": 42,
        }
        return _tabular_pipeline(
            X_train,
            LogisticRegression(**{**defaults, **params}),
            scale_numeric=True,
        )
    if family == "prophet":
        prophet_params = row.get("prophet_params")
        regressors = row.get("regressors")
        return ProphetFrameClassifier(
            timestamp_column=str(row.get("timestamp_column") or timestamp_column),
            prophet_params=prophet_params if isinstance(prophet_params, dict) else {},
            regressors=[str(value) for value in regressors] if isinstance(regressors, list) else None,
            calibration_method=str(row.get("calibration_method") or "clip"),
            calibration_features=(
                [str(value) for value in row.get("calibration_features", [])]
                if isinstance(row.get("calibration_features"), list)
                else None
            ),
        )
    raise ValueError(f"Unsupported classification candidate family: {family}")


def _regression_estimator_from_config(
    row: dict[str, Any],
    X_train: pd.DataFrame,
    *,
    timestamp_column: str,
    training_config: dict[str, Any] | None = None,
) -> Any:
    family = _candidate_family(row)
    params = _profiled_candidate_params(
        family,
        _candidate_params(row),
        training_config,
    )
    if family == "catboost":
        defaults = {
            "iterations": 250,
            "depth": 7,
            "learning_rate": 0.03,
            "loss_function": "RMSE",
            "random_seed": 42,
            "verbose": False,
        }
        return CatBoostFrameRegressor(**{**defaults, **params})
    if family == "xgboost":
        defaults = {
            "n_estimators": 250,
            "max_depth": 6,
            "learning_rate": 0.03,
            "subsample": 0.85,
            "colsample_bytree": 0.75,
            "reg_alpha": 0.15,
            "reg_lambda": 1.2,
            "objective": "reg:squarederror",
            "random_state": 42,
        }
        return _tabular_pipeline(X_train, XGBRegressor(**{**defaults, **params}))
    if family == "lightgbm":
        defaults = {
            "n_estimators": 150,
            "learning_rate": 0.03,
            "num_leaves": 31,
            "subsample": 0.85,
            "colsample_bytree": 0.75,
            "n_jobs": 4,
            "random_state": 42,
            "verbosity": -1,
        }
        return _tabular_pipeline(X_train, LGBMRegressor(**{**defaults, **params}))
    if family == "random_forest":
        defaults = {
            "n_estimators": 200,
            "max_depth": None,
            "min_samples_leaf": 2,
            "n_jobs": 4,
            "random_state": 42,
        }
        return _tabular_pipeline(X_train, RandomForestRegressor(**{**defaults, **params}))
    if family == "ridge":
        defaults = {"alpha": 2.0, "random_state": 42}
        return _tabular_pipeline(
            X_train,
            Ridge(**{**defaults, **params}),
            scale_numeric=True,
        )
    if family == "prophet":
        prophet_params = row.get("prophet_params")
        regressors = row.get("regressors")
        return ProphetFrameRegressor(
            timestamp_column=str(row.get("timestamp_column") or timestamp_column),
            target_transform=str(row.get("target_transform") or ""),
            clip_min=row.get("clip_min", 0.0),
            prophet_params=prophet_params if isinstance(prophet_params, dict) else {},
            regressors=[str(value) for value in regressors] if isinstance(regressors, list) else None,
        )
    raise ValueError(f"Unsupported regression candidate family: {family}")


def _fit_classification_candidates(
    train_frame: pd.DataFrame,
    validation_frame: pd.DataFrame,
    dataset: TaskDataset,
) -> list[CandidateResult]:
    target = dataset.target_column
    feature_columns = _feature_columns_for_dataset(train_frame, dataset)
    X_train = train_frame[feature_columns].copy()
    y_train = train_frame[target].astype(int)
    X_valid = validation_frame[feature_columns].copy()
    y_valid = validation_frame[target].astype(int)
    hard_mask = dataset.hard_mask_builder(validation_frame).fillna(False)
    training_config = (
        dict(dataset.training_config)
        if isinstance(dataset.training_config, dict)
        else {}
    )
    disabled_candidates = {
        str(value).strip()
        for value in training_config.get("disabled_classification_candidates", [])
        if str(value).strip()
    }

    default_rows = [
        {"name": "catboost", "family": "catboost"},
        {"name": "xgboost", "family": "xgboost"},
        {"name": "lightgbm", "family": "lightgbm"},
        {"name": "random_forest", "family": "random_forest"},
        {"name": "logistic_regression", "family": "logistic_regression"},
    ]
    candidate_rows = _candidate_rows(
        training_config,
        "classification_candidates",
        default_rows,
        "include_default_classifiers",
    )
    candidates: list[tuple[str, Any]] = []
    for row in candidate_rows:
        candidate_name = _candidate_name(row)
        if not candidate_name or candidate_name in disabled_candidates:
            continue
        candidates.append(
            (
                candidate_name,
                _classification_estimator_from_config(
                    row,
                    X_train,
                    timestamp_column=dataset.timestamp_column,
                    training_config=training_config,
                ),
            )
        )

    results: list[CandidateResult] = []
    fitted_estimators: list[tuple[str, Any]] = []
    for candidate_name, estimator in candidates:
        estimator.fit(X_train, y_train)
        probabilities = estimator.predict_proba(X_valid)[:, 1]
        predictions = (probabilities >= 0.5).astype(int)
        metrics = _classification_metrics(y_valid, probabilities, predictions)
        if hard_mask.any():
            hard_metrics = _classification_metrics(
                y_valid.loc[hard_mask],
                probabilities[hard_mask.to_numpy()],
                predictions[hard_mask.to_numpy()],
            )
        else:
            hard_metrics = metrics
        results.append(
            CandidateResult(
                candidate_name=candidate_name,
                estimator=estimator,
                metrics=metrics,
                hard_metrics=hard_metrics,
                validation_score=_ranking_score(
                    dataset.task_type,
                    metrics,
                    hard_metrics,
                ),
                predictions=predictions,
                probabilities=probabilities,
            )
        )
        fitted_estimators.append((candidate_name, estimator))

    results.extend(
        _fit_sequence_classification_candidates(
            X_train=X_train,
            y_train=y_train,
            X_valid=X_valid,
            y_valid=y_valid,
            hard_mask=hard_mask,
            dataset=dataset,
            training_config=training_config,
        )
    )

    ensemble_config = training_config.get("ensemble")
    if (
        isinstance(ensemble_config, dict)
        and ensemble_config.get("enabled") is True
        and len(fitted_estimators) >= int(ensemble_config.get("min_members", 2))
    ):
        if str(ensemble_config.get("weighting") or "").strip() == "validation_score":
            raw_weights = [
                max(0.0, result.validation_score)
                for result in results
                if result.candidate_name in {name for name, _estimator in fitted_estimators}
            ]
        else:
            raw_weights = [1.0 for _name, _estimator in fitted_estimators]
        if not raw_weights or sum(raw_weights) <= 0:
            raw_weights = [1.0 for _name, _estimator in fitted_estimators]
        ensemble = WeightedProbabilityEnsembleClassifier(
            estimators=fitted_estimators,
            weights=raw_weights,
        )
        ensemble.estimators_ = fitted_estimators
        ensemble.weights_ = np.asarray(raw_weights, dtype=np.float64)
        probabilities = ensemble.predict_proba(X_valid)[:, 1]
        predictions = (probabilities >= 0.5).astype(int)
        metrics = _classification_metrics(y_valid, probabilities, predictions)
        hard_metrics = (
            _classification_metrics(
                y_valid.loc[hard_mask],
                probabilities[hard_mask.to_numpy()],
                predictions[hard_mask.to_numpy()],
            )
            if hard_mask.any()
            else metrics
        )
        results.append(
            CandidateResult(
                candidate_name=str(ensemble_config.get("name") or "tabular_probability_ensemble"),
                estimator=ensemble,
                metrics=metrics,
                hard_metrics=hard_metrics,
                validation_score=_ranking_score(
                    dataset.task_type,
                    metrics,
                    hard_metrics,
                    training_config.get("selection"),
                ),
                predictions=predictions,
                probabilities=probabilities,
            )
        )

    if not results:
        raise ValueError(f"No classification candidates configured for {dataset.task_name}.")
    return results


def _sequence_estimator_config(
    candidate: dict[str, Any],
    training_config: dict[str, Any],
) -> dict[str, Any]:
    return {
        "architecture": str(candidate["architecture"]).strip().lower(),
        "sequence_features": list(training_config.get("sequence_features") or []),
        "lag_steps": list(training_config.get("lag_steps") or []),
        "hidden_size": int(candidate.get("hidden_size", 64)),
        "num_layers": int(candidate.get("num_layers", 2)),
        "dropout": float(candidate.get("dropout", 0.0)),
        "epochs": int(candidate.get("epochs", 20)),
        "batch_size": int(candidate.get("batch_size", 256)),
        "learning_rate": float(candidate.get("learning_rate", 0.001)),
        "weight_decay": float(candidate.get("weight_decay", 0.0)),
        "patience": int(candidate.get("patience", 5)),
        "clip_min": training_config.get("clip_min", 0.0),
        "target_transform": str(training_config.get("target_transform") or "").strip(),
        "calibration_method": str(
            candidate.get("calibration_method")
            or training_config.get("calibration_method")
            or "none"
        ).strip(),
    }


def _fit_sequence_classification_candidates(
    *,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_valid: pd.DataFrame,
    y_valid: pd.Series,
    hard_mask: pd.Series,
    dataset: TaskDataset,
    training_config: dict[str, Any],
) -> list[CandidateResult]:
    sequence_config = training_config.get("sequence")
    if not isinstance(sequence_config, dict) or sequence_config.get("enabled") is not True:
        return []
    raw_candidates = sequence_config.get("candidates")
    if not isinstance(raw_candidates, list) or not raw_candidates:
        return []

    base_results: list[CandidateResult] = []
    fitted_estimators: list[TorchSequenceClassifier] = []
    estimator_configs: list[dict[str, Any]] = []
    for raw_candidate in raw_candidates:
        if not isinstance(raw_candidate, dict) or raw_candidate.get("enabled", True) is False:
            continue
        candidate_name = str(raw_candidate.get("name") or raw_candidate.get("architecture") or "").strip()
        if not candidate_name:
            continue
        estimator_config = _sequence_estimator_config(raw_candidate, training_config)
        estimator = TorchSequenceClassifier(**estimator_config)
        estimator.fit(X_train, y_train)
        probabilities = estimator.predict_proba(X_valid)[:, 1]
        predictions = (probabilities >= 0.5).astype(int)
        metrics = _classification_metrics(y_valid, probabilities, predictions)
        hard_metrics = (
            _classification_metrics(
                y_valid.loc[hard_mask],
                probabilities[hard_mask.to_numpy()],
                predictions[hard_mask.to_numpy()],
            )
            if hard_mask.any()
            else metrics
        )
        base_results.append(
            CandidateResult(
                candidate_name=candidate_name,
                estimator=estimator,
                metrics=metrics,
                hard_metrics=hard_metrics,
                validation_score=_ranking_score(
                    dataset.task_type,
                    metrics,
                    hard_metrics,
                    training_config.get("selection"),
                ),
                predictions=predictions,
                probabilities=probabilities,
            )
        )
        fitted_estimators.append(estimator)
        estimator_configs.append(estimator_config)

    ensemble_config = sequence_config.get("ensemble")
    if (
        isinstance(ensemble_config, dict)
        and ensemble_config.get("enabled") is True
        and len(base_results) >= 2
    ):
        weighting = str(ensemble_config.get("weighting") or "equal").strip()
        if weighting == "validation_auc":
            weights = [
                max(0.0, float(result.metrics.get("roc_auc", 0.0)))
                for result in base_results
            ]
        elif weighting == "validation_brier":
            brier_values = [result.metrics.get("brier_score", float("inf")) for result in base_results]
            min_brier = min(brier_values)
            weights = [max(0.0, min_brier / max(b, 1e-9)) for b in brier_values]
        else:
            weights = [1.0 for _result in base_results]
        if sum(weights) <= 0:
            weights = [1.0 for _result in base_results]
        ensemble = TorchSequenceEnsembleClassifier(
            estimator_configs=estimator_configs,
            weights=weights,
        )
        ensemble.estimators_ = fitted_estimators
        ensemble.weights_ = np.asarray(weights, dtype=np.float64)
        probabilities = ensemble.predict_proba(X_valid)[:, 1]
        predictions = (probabilities >= 0.5).astype(int)
        metrics = _classification_metrics(y_valid, probabilities, predictions)
        hard_metrics = (
            _classification_metrics(
                y_valid.loc[hard_mask],
                probabilities[hard_mask.to_numpy()],
                predictions[hard_mask.to_numpy()],
            )
            if hard_mask.any()
            else metrics
        )
        base_results.append(
            CandidateResult(
                candidate_name=str(ensemble_config.get("name") or "sequence_probability_ensemble"),
                estimator=ensemble,
                metrics=metrics,
                hard_metrics=hard_metrics,
                validation_score=_ranking_score(
                    dataset.task_type,
                    metrics,
                    hard_metrics,
                    training_config.get("selection"),
                ),
                predictions=predictions,
                probabilities=probabilities,
            )
        )

    return base_results


def _fit_sequence_regression_candidates(
    *,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_valid: pd.DataFrame,
    y_valid: pd.Series,
    hard_mask: pd.Series,
    dataset: TaskDataset,
    training_config: dict[str, Any],
) -> list[CandidateResult]:
    sequence_config = training_config.get("sequence")
    if not isinstance(sequence_config, dict) or sequence_config.get("enabled") is not True:
        return []
    raw_candidates = sequence_config.get("candidates")
    if not isinstance(raw_candidates, list) or not raw_candidates:
        return []

    base_results: list[CandidateResult] = []
    fitted_estimators: list[TorchSequenceRegressor] = []
    estimator_configs: list[dict[str, Any]] = []
    for raw_candidate in raw_candidates:
        if not isinstance(raw_candidate, dict):
            continue
        candidate_name = str(raw_candidate.get("name") or raw_candidate.get("architecture") or "").strip()
        if not candidate_name:
            continue
        estimator_config = _sequence_estimator_config(raw_candidate, training_config)
        estimator = TorchSequenceRegressor(**estimator_config)
        estimator.fit(X_train, y_train)
        predictions = estimator.predict(X_valid)
        metrics = _compute_task_metrics(dataset, y_valid, predictions)
        hard_metrics = (
            _compute_task_metrics(
                dataset,
                y_valid.loc[hard_mask],
                predictions[hard_mask.to_numpy()],
            )
            if hard_mask.any()
            else metrics
        )
        base_results.append(
            CandidateResult(
                candidate_name=candidate_name,
                estimator=estimator,
                metrics=metrics,
                hard_metrics=hard_metrics,
                validation_score=_ranking_score(
                    dataset.task_type,
                    metrics,
                    hard_metrics,
                    training_config.get("selection"),
                ),
                predictions=predictions,
                probabilities=None,
            )
        )
        fitted_estimators.append(estimator)
        estimator_configs.append(estimator_config)

    ensemble_config = sequence_config.get("ensemble")
    if (
        isinstance(ensemble_config, dict)
        and ensemble_config.get("enabled") is True
        and len(base_results) >= 2
    ):
        weighting = str(ensemble_config.get("weighting") or "equal").strip()
        if weighting == "validation_accuracy":
            weights = [
                max(0.0, float(result.metrics.get("accuracy", 0.0)))
                for result in base_results
            ]
        elif weighting == "validation_mase":
            mase_values = [result.metrics.get("mase", float("inf")) for result in base_results]
            min_mase = min(mase_values)
            weights = [max(0.0, min_mase / max(m, 1e-9)) for m in mase_values]
        else:
            weights = [1.0 for _result in base_results]
        if sum(weights) <= 0:
            weights = [1.0 for _result in base_results]
        ensemble = TorchSequenceEnsembleRegressor(
            estimator_configs=estimator_configs,
            weights=weights,
        )
        ensemble.estimators_ = fitted_estimators
        ensemble.weights_ = np.asarray(weights, dtype=np.float64)
        ensemble_predictions = ensemble.predict(X_valid)
        metrics = _regression_metrics(
            y_valid,
            ensemble_predictions,
            accuracy_tolerance=dataset.accuracy_tolerance,
        )
        hard_metrics = (
            _regression_metrics(
                y_valid.loc[hard_mask],
                ensemble_predictions[hard_mask.to_numpy()],
                accuracy_tolerance=dataset.accuracy_tolerance,
            )
            if hard_mask.any()
            else metrics
        )
        base_results.append(
            CandidateResult(
                candidate_name=str(ensemble_config.get("name") or "sequence_ensemble"),
                estimator=ensemble,
                metrics=metrics,
                hard_metrics=hard_metrics,
                validation_score=_ranking_score(
                    dataset.task_type,
                    metrics,
                    hard_metrics,
                    training_config.get("selection"),
                ),
                predictions=ensemble_predictions,
                probabilities=None,
            )
        )

    return base_results


def _fit_regression_candidates(
    train_frame: pd.DataFrame,
    validation_frame: pd.DataFrame,
    dataset: TaskDataset,
) -> list[CandidateResult]:
    target = dataset.target_column
    feature_columns = _feature_columns_for_dataset(train_frame, dataset)
    X_train = train_frame[feature_columns].copy()
    y_train = _safe_numeric(train_frame[target])
    X_valid = validation_frame[feature_columns].copy()
    y_valid = _safe_numeric(validation_frame[target])
    hard_mask = dataset.hard_mask_builder(validation_frame).fillna(False)
    training_config = (
        dict(dataset.training_config)
        if isinstance(dataset.training_config, dict)
        else {}
    )
    disabled_candidates = {
        str(value).strip()
        for value in training_config.get("disabled_regression_candidates", [])
        if str(value).strip()
    }

    default_rows = [
        {"name": "catboost", "family": "catboost"},
        {"name": "xgboost", "family": "xgboost"},
        {"name": "lightgbm", "family": "lightgbm"},
        {"name": "random_forest", "family": "random_forest"},
        {"name": "ridge", "family": "ridge"},
    ]
    candidate_rows = _candidate_rows(
        training_config,
        "regression_candidates",
        default_rows,
        "include_default_regressors",
    )
    candidates: list[tuple[str, Any]] = []
    for row in candidate_rows:
        candidate_name = _candidate_name(row)
        if not candidate_name or candidate_name in disabled_candidates:
            continue
        candidates.append(
            (
                candidate_name,
                _regression_estimator_from_config(
                    row,
                    X_train,
                    timestamp_column=dataset.timestamp_column,
                    training_config=training_config,
                ),
            )
        )

    results: list[CandidateResult] = []
    for candidate_name, estimator in candidates:
        estimator.fit(X_train, y_train)
        predictions = estimator.predict(X_valid)
        metrics = _compute_task_metrics(dataset, y_valid, predictions)
        if hard_mask.any():
            hard_metrics = _regression_metrics(
                y_valid.loc[hard_mask],
                predictions[hard_mask.to_numpy()],
                accuracy_tolerance=dataset.accuracy_tolerance,
            )
        else:
            hard_metrics = metrics
        results.append(
            CandidateResult(
                candidate_name=candidate_name,
                estimator=estimator,
                metrics=metrics,
                hard_metrics=hard_metrics,
                validation_score=_ranking_score(
                    dataset.task_type,
                    metrics,
                    hard_metrics,
                ),
                predictions=predictions,
                probabilities=None,
            )
        )
    return results


def _release_non_selected_estimators(
    candidate_results: list[CandidateResult],
    selected: CandidateResult,
) -> list[CandidateResult]:
    released: list[CandidateResult] = []
    for result in candidate_results:
        if result is selected:
            released.append(result)
        else:
            released.append(replace(result, estimator=None))
    return released


def fit_task(dataset: TaskDataset) -> TaskResult:
    train_frame, test_frame = _time_split(
        dataset.frame,
        dataset.timestamp_column,
        test_fraction=0.2,
    )
    inner_train, validation_frame = _time_split(
        train_frame,
        dataset.timestamp_column,
        test_fraction=0.2,
    )

    if dataset.post_split_feature_fn is not None:
        inner_train = dataset.post_split_feature_fn(inner_train)
        validation_frame = dataset.post_split_feature_fn(validation_frame)

    if dataset.task_type == "classification":
        candidate_results = _fit_classification_candidates(
            inner_train,
            validation_frame,
            dataset,
        )
    else:
        candidate_results = _fit_regression_candidates(
            inner_train,
            validation_frame,
            dataset,
        )

    best_validation = max(candidate_results, key=lambda row: row.validation_score)
    candidate_results = _release_non_selected_estimators(
        candidate_results,
        best_validation,
    )
    gc.collect()

    if dataset.post_split_feature_fn is not None:
        train_frame = dataset.post_split_feature_fn(train_frame)
        test_frame = dataset.post_split_feature_fn(test_frame)

    target = dataset.target_column
    feature_columns = _feature_columns_for_dataset(train_frame, dataset)
    feature_types = {
        column: str(train_frame[column].dtype) for column in feature_columns
    }
    X_train = train_frame[feature_columns].copy()
    y_train = train_frame[target]
    X_test = test_frame[feature_columns].copy()
    y_test = test_frame[target]
    hard_mask = dataset.hard_mask_builder(test_frame).fillna(False)

    estimator = best_validation.estimator
    uses_native_categories = isinstance(estimator, (CatBoostClassifier, CatBoostRegressor))
    if uses_native_categories:
        categorical_columns = [
            column
            for column in feature_columns
            if not pd.api.types.is_numeric_dtype(X_train[column])
        ]
        native_train = X_train.copy()
        native_test = X_test.copy()
        for column in categorical_columns:
            native_train[column] = (
                native_train[column].fillna("__MISSING__").astype(str)
            )
            native_test[column] = native_test[column].fillna("__MISSING__").astype(str)
        estimator.fit(native_train, y_train, cat_features=categorical_columns)
        if dataset.task_type == "classification":
            probabilities = estimator.predict_proba(native_test)[:, 1]
            predictions = (probabilities >= 0.5).astype(int)
            metrics = _classification_metrics(y_test, probabilities, predictions)
            hard_metrics = (
                _classification_metrics(
                    y_test.loc[hard_mask],
                    probabilities[hard_mask.to_numpy()],
                    predictions[hard_mask.to_numpy()],
                )
                if hard_mask.any()
                else metrics
            )
        else:
            predictions = estimator.predict(native_test)
            probabilities = None
            metrics = _regression_metrics(
                y_test,
                predictions,
                accuracy_tolerance=dataset.accuracy_tolerance,
            )
            hard_metrics = (
                _regression_metrics(
                    y_test.loc[hard_mask],
                    predictions[hard_mask.to_numpy()],
                    accuracy_tolerance=dataset.accuracy_tolerance,
                )
                if hard_mask.any()
                else metrics
            )
    else:
        estimator.fit(X_train, y_train)
        if dataset.task_type == "classification":
            probabilities = estimator.predict_proba(X_test)[:, 1]
            predictions = (probabilities >= 0.5).astype(int)
            metrics = _classification_metrics(y_test, probabilities, predictions)
            hard_metrics = (
                _classification_metrics(
                    y_test.loc[hard_mask],
                    probabilities[hard_mask.to_numpy()],
                    predictions[hard_mask.to_numpy()],
                )
                if hard_mask.any()
                else metrics
            )
        else:
            predictions = estimator.predict(X_test)
            probabilities = None
            metrics = _regression_metrics(
                y_test,
                predictions,
                accuracy_tolerance=dataset.accuracy_tolerance,
            )
            hard_metrics = (
                _regression_metrics(
                    y_test.loc[hard_mask],
                    predictions[hard_mask.to_numpy()],
                    accuracy_tolerance=dataset.accuracy_tolerance,
                )
                if hard_mask.any()
                else metrics
            )

    best_result = CandidateResult(
        candidate_name=best_validation.candidate_name,
        estimator=estimator,
        metrics=metrics,
        hard_metrics=hard_metrics,
        validation_score=_ranking_score(dataset.task_type, metrics, hard_metrics),
        predictions=np.asarray(predictions),
        probabilities=(
            np.asarray(probabilities) if probabilities is not None else None
        ),
    )

    return TaskResult(
        dataset=dataset,
        candidate_results=candidate_results,
        best_result=best_result,
        validation_result=best_validation,
        feature_columns=feature_columns,
        feature_types=feature_types,
        train_frame=train_frame,
        test_frame=test_frame,
    )


def _log_candidate_table(task_result: TaskResult, output_dir: Path) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for candidate in task_result.candidate_results:
        row = {
            "candidate": candidate.candidate_name,
            "validation_score": candidate.validation_score,
        }
        row.update({f"val_{key}": value for key, value in candidate.metrics.items()})
        row.update(
            {f"val_hard_{key}": value for key, value in candidate.hard_metrics.items()}
        )
        rows.append(row)
    rows.append(
        {
            "candidate": f"{task_result.best_result.candidate_name}_final_test",
            "validation_score": task_result.best_result.validation_score,
            **{
                f"test_{key}": value
                for key, value in task_result.best_result.metrics.items()
            },
            **{
                f"test_hard_{key}": value
                for key, value in task_result.best_result.hard_metrics.items()
            },
        }
    )
    candidate_table = pd.DataFrame(rows)
    candidate_table.to_csv(output_dir / "candidate_metrics.csv", index=False)
    return candidate_table


def _set_plot_style() -> None:
    sns.set_theme(style="whitegrid")
    plt.rcParams.update(
        {
            "font.family": "DejaVu Serif",
            "axes.titlesize": 14,
            "axes.labelsize": 12,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "legend.fontsize": 10,
            "figure.titlesize": 16,
        }
    )


def _save_figure(fig: plt.Figure, output_dir: Path, name: str) -> None:
    fig.tight_layout()
    fig.savefig(output_dir / f"{name}.png", dpi=300, bbox_inches="tight")
    fig.savefig(output_dir / f"{name}.pdf", bbox_inches="tight")
    plt.close(fig)


def _plot_candidate_leaderboard(task_result: TaskResult, output_dir: Path) -> None:
    _set_plot_style()
    table = _log_candidate_table(task_result, output_dir)
    score_table = table[
        table["candidate"] != f"{task_result.best_result.candidate_name}_final_test"
    ].copy()
    fig, ax = plt.subplots(figsize=(9, 4.8))
    palette = sns.color_palette("crest", n_colors=len(score_table))
    ax.bar(score_table["candidate"], score_table["validation_score"], color=palette)
    ax.set_title(f"{task_result.dataset.task_name} candidate leaderboard")
    ax.set_ylabel("Validation selection score")
    ax.set_xlabel("Candidate model")
    ax.tick_params(axis="x", rotation=20)
    _save_figure(fig, output_dir, "candidate_leaderboard")


def _plot_classification_diagnostics(task_result: TaskResult, output_dir: Path) -> None:
    probabilities = task_result.best_result.probabilities
    if probabilities is None:
        return
    y_test = (
        task_result.test_frame[task_result.dataset.target_column].astype(int).to_numpy()
    )
    predictions = task_result.best_result.predictions.astype(int)
    hard_mask = (
        task_result.dataset.hard_mask_builder(task_result.test_frame)
        .fillna(False)
        .to_numpy()
    )
    _set_plot_style()

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6))
    fpr, tpr, _ = roc_curve(y_test, probabilities)
    axes[0].plot(fpr, tpr, color="#0a7f8e", lw=2.2)
    axes[0].plot([0, 1], [0, 1], color="#8f8f8f", lw=1.2, ls="--")
    axes[0].set_title("ROC curve")
    axes[0].set_xlabel("False positive rate")
    axes[0].set_ylabel("True positive rate")
    axes[0].annotate(
        f"AUC = {task_result.best_result.metrics['roc_auc']:.3f}",
        xy=(0.60, 0.12),
        xycoords="axes fraction",
    )

    precision, recall, _ = precision_recall_curve(y_test, probabilities)
    axes[1].plot(recall, precision, color="#c17b12", lw=2.2)
    axes[1].set_title("Precision-recall curve")
    axes[1].set_xlabel("Recall")
    axes[1].set_ylabel("Precision")
    axes[1].annotate(
        f"AP = {task_result.best_result.metrics['average_precision']:.3f}",
        xy=(0.60, 0.12),
        xycoords="axes fraction",
    )

    comparison = pd.DataFrame(
        {
            "Slice": ["Overall", "Hard"],
            "Accuracy": [
                task_result.best_result.metrics["accuracy"],
                task_result.best_result.hard_metrics["accuracy"],
            ],
            "Balanced accuracy": [
                task_result.best_result.metrics["balanced_accuracy"],
                task_result.best_result.hard_metrics["balanced_accuracy"],
            ],
        }
    )
    comparison = comparison.melt(id_vars="Slice", var_name="Metric", value_name="Value")
    sns.barplot(data=comparison, x="Slice", y="Value", hue="Metric", ax=axes[2])
    axes[2].set_title("Overall vs hard-slice accuracy")
    axes[2].set_ylim(0.0, 1.0)
    axes[2].legend(loc="lower right")
    _save_figure(fig, output_dir, "classification_diagnostics")

    fig, ax = plt.subplots(figsize=(8.8, 4.6))
    calibration = pd.DataFrame(
        {
            "probability": probabilities,
            "label": y_test,
            "slice": np.where(hard_mask, "Hard slice", "Overall test"),
        }
    )
    calibration["bucket"] = pd.qcut(
        calibration["probability"].rank(method="first"),
        q=min(10, len(calibration)),
        duplicates="drop",
    )

    # 100% Safe bypass for the Pandas AttributeError bug
    curve = (
        calibration.groupby(["slice", "bucket"], observed=True)[
            ["probability", "label"]
        ]
        .mean()
        .reset_index()
        .rename(columns={"probability": "predicted", "label": "actual"})
    )

    sns.lineplot(
        data=curve,
        x="predicted",
        y="actual",
        hue="slice",
        marker="o",
        linewidth=2.0,
        ax=ax,
    )
    ax.plot([0, 1], [0, 1], color="#8f8f8f", lw=1.1, ls="--")
    ax.set_title(f"{task_result.dataset.task_name} calibration profile")
    ax.set_xlabel("Predicted probability")
    ax.set_ylabel("Observed event rate")
    _save_figure(fig, output_dir, "calibration_profile")


def _plot_regression_diagnostics(task_result: TaskResult, output_dir: Path) -> None:
    predictions = np.asarray(task_result.best_result.predictions, dtype=np.float64)
    actual = _safe_numeric(
        task_result.test_frame[task_result.dataset.target_column]
    ).to_numpy()
    hard_mask = (
        task_result.dataset.hard_mask_builder(task_result.test_frame)
        .fillna(False)
        .to_numpy()
    )
    residuals = actual - predictions
    _set_plot_style()
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6))

    axes[0].scatter(actual, predictions, color="#0a7f8e", alpha=0.55, s=18)
    bounds = [
        min(actual.min(), predictions.min()),
        max(actual.max(), predictions.max()),
    ]
    axes[0].plot(bounds, bounds, color="#8f8f8f", lw=1.2, ls="--")
    axes[0].set_title("Predicted vs actual")
    axes[0].set_xlabel("Actual")
    axes[0].set_ylabel("Predicted")
    axes[0].annotate(
        f"RMSE = {task_result.best_result.metrics['root_mean_squared_error']:.2f}",
        xy=(0.04, 0.92),
        xycoords="axes fraction",
    )

    axes[1].scatter(predictions, residuals, color="#c17b12", alpha=0.55, s=18)
    axes[1].axhline(0.0, color="#8f8f8f", lw=1.1, ls="--")
    axes[1].set_title("Residual profile")
    axes[1].set_xlabel("Predicted")
    axes[1].set_ylabel("Residual")

    comparison = pd.DataFrame(
        {
            "Slice": ["Overall", "Hard"],
            "Accuracy": [
                task_result.best_result.metrics.get("accuracy", 0.0),
                task_result.best_result.hard_metrics.get("accuracy", 0.0),
            ],
            "RMSE": [
                task_result.best_result.metrics["root_mean_squared_error"],
                task_result.best_result.hard_metrics["root_mean_squared_error"],
            ],
        }
    )
    comparison = comparison.melt(id_vars="Slice", var_name="Metric", value_name="Value")
    sns.barplot(data=comparison, x="Slice", y="Value", hue="Metric", ax=axes[2])
    axes[2].set_title("Overall vs hard-slice error")
    _save_figure(fig, output_dir, "regression_diagnostics")

    fig, ax = plt.subplots(figsize=(8.8, 4.6))
    sns.histplot(residuals, bins=30, kde=True, color="#6c5ce7", ax=ax)
    ax.set_title(f"{task_result.dataset.task_name} residual distribution")
    ax.set_xlabel("Residual")
    _save_figure(fig, output_dir, "residual_distribution")


def _plot_feature_importance(task_result: TaskResult, output_dir: Path) -> None:
    estimator = task_result.best_result.estimator
    model = estimator
    if hasattr(estimator, "named_steps") and "model" in estimator.named_steps:
        model = estimator.named_steps["model"]
    importances: np.ndarray | None = None
    if hasattr(model, "feature_importances_"):
        importances = np.asarray(model.feature_importances_, dtype=np.float64)
    elif hasattr(model, "get_feature_importance"):
        try:
            importances = np.asarray(model.get_feature_importance(), dtype=np.float64)
        except Exception:
            importances = None
    if importances is None or len(importances) != len(task_result.feature_columns):
        return
    importance_table = (
        pd.DataFrame(
            {
                "feature": task_result.feature_columns,
                "importance": importances,
            }
        )
        .sort_values("importance", ascending=False)
        .head(12)
    )
    _set_plot_style()
    fig, ax = plt.subplots(figsize=(9.2, 5.0))
    sns.barplot(
        data=importance_table,
        x="importance",
        y="feature",
        hue="feature",
        palette="viridis",
        legend=False,
        ax=ax,
    )
    ax.set_title(f"{task_result.dataset.task_name} feature importance")
    ax.set_xlabel("Importance")
    ax.set_ylabel("")
    _save_figure(fig, output_dir, "feature_importance")


def plot_task_result(task_result: TaskResult) -> None:
    output_dir = _ensure_output_dir(task_result.dataset.task_name)
    _plot_candidate_leaderboard(task_result, output_dir)
    _plot_feature_importance(task_result, output_dir)
    if task_result.dataset.task_type == "classification":
        _plot_classification_diagnostics(task_result, output_dir)
    else:
        _plot_regression_diagnostics(task_result, output_dir)
    _save_shap_explanations(task_result, output_dir)
    _write_task_metric_artifacts(task_result, output_dir)
    if task_result.dataset.model_name != task_result.dataset.task_name:
        model_output_dir = _ensure_output_dir(task_result.dataset.model_name)
        _write_task_metric_artifacts(task_result, model_output_dir)


def _feature_leakage_audit_payload(task_result: TaskResult) -> dict[str, Any]:
    config = _feature_leakage_guard_config()
    diagnostics = config.get("diagnostics", {})
    if not isinstance(diagnostics, dict):
        diagnostics = {}
    threshold = float(diagnostics.get("high_correlation_threshold", 1.0))
    frame = task_result.test_frame
    target = _safe_numeric(frame[task_result.dataset.target_column], fill_value=np.nan)
    high_correlation: list[dict[str, Any]] = []
    if bool(diagnostics.get("enabled", True)):
        for column in task_result.feature_columns:
            if column not in frame.columns or not pd.api.types.is_numeric_dtype(frame[column]):
                continue
            values = _safe_numeric(frame[column], fill_value=np.nan)
            if values.nunique(dropna=True) < 2 or target.nunique(dropna=True) < 2:
                continue
            correlation = values.corr(target, method="spearman")
            if pd.notna(correlation) and abs(float(correlation)) >= threshold:
                high_correlation.append(
                    {
                        "feature": column,
                        "spearman_abs": abs(float(correlation)),
                        "spearman": float(correlation),
                    }
                )
    blocked_feature_columns = [
        column for column in task_result.feature_columns if _is_future_or_label_column(column)
    ]
    return {
        "task": task_result.dataset.task_name,
        "model_name": task_result.dataset.model_name,
        "target_column": task_result.dataset.target_column,
        "feature_count": len(task_result.feature_columns),
        "blocked_feature_columns": blocked_feature_columns,
        "high_correlation_threshold": threshold,
        "high_correlation_features": sorted(
            high_correlation,
            key=lambda row: row["spearman_abs"],
            reverse=True,
        ),
        "status": "fail" if blocked_feature_columns else "pass",
    }


def _task_metrics_payload(task_result: TaskResult) -> dict[str, Any]:
    validation_result = task_result.validation_result
    return {
        "task": task_result.dataset.task_name,
        "model_name": task_result.dataset.model_name,
        "selected_candidate": task_result.best_result.candidate_name,
        "overall_metrics": task_result.best_result.metrics,
        "hard_metrics": task_result.best_result.hard_metrics,
        "validation_metrics": (
            validation_result.metrics if validation_result is not None else {}
        ),
        "validation_hard_metrics": (
            validation_result.hard_metrics if validation_result is not None else {}
        ),
        "test_metrics": task_result.best_result.metrics,
        "test_hard_metrics": task_result.best_result.hard_metrics,
        "feature_columns": task_result.feature_columns,
        "leakage_audit": _feature_leakage_audit_payload(task_result),
    }


def _write_task_metric_artifacts(task_result: TaskResult, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = _task_metrics_payload(task_result)
    (output_dir / "metrics.json").write_text(
        json.dumps(payload, indent=2),
        encoding="utf-8",
    )
    (output_dir / "leakage_audit.json").write_text(
        json.dumps(payload["leakage_audit"], indent=2),
        encoding="utf-8",
    )


def _enforce_metric_bounds(task_result: TaskResult) -> None:
    min_accuracy = task_result.dataset.min_accuracy
    max_accuracy = task_result.dataset.max_accuracy
    if min_accuracy is None and max_accuracy is None:
        return
    for label, metrics in (
        ("overall", task_result.best_result.metrics),
        ("hard", task_result.best_result.hard_metrics),
    ):
        accuracy = metrics.get("accuracy")
        if accuracy is None:
            continue
        numeric_accuracy = float(accuracy)
        if min_accuracy is not None and numeric_accuracy < min_accuracy:
            raise ValueError(
                f"{task_result.dataset.task_name} {label} accuracy "
                f"{numeric_accuracy:.4f} must be at least {min_accuracy:.4f}."
            )
        if max_accuracy is not None and numeric_accuracy >= max_accuracy:
            raise ValueError(
                f"{task_result.dataset.task_name} {label} accuracy "
                f"{numeric_accuracy:.4f} must stay below {max_accuracy:.4f}."
            )


def _prediction_ready_frame(
    task_result: TaskResult, frame: pd.DataFrame
) -> pd.DataFrame:
    prepared = _normalize_datetime_columns_for_mlflow(frame)
    if task_result.best_result.candidate_name != "catboost":
        return prepared

    for column in task_result.feature_columns:
        if column not in prepared.columns:
            continue
        if not pd.api.types.is_numeric_dtype(task_result.train_frame[column]):
            prepared[column] = prepared[column].fillna("__MISSING__").astype(str)
    return prepared


def _normalize_datetime_columns_for_mlflow(frame: pd.DataFrame) -> pd.DataFrame:
    prepared = frame.copy()
    for column in prepared.columns:
        series = prepared[column]
        if isinstance(series.dtype, pd.DatetimeTZDtype):
            prepared[column] = series.dt.tz_convert(None)
            continue

        non_null = series.dropna()
        if non_null.empty:
            continue
        sample = non_null.iloc[0]
        if isinstance(sample, pd.Timestamp) and sample.tzinfo is not None:
            prepared[column] = pd.to_datetime(series, utc=True, errors="coerce").dt.tz_convert(
                None
            )
    return prepared


def log_task_to_mlflow(task_result: TaskResult) -> dict[str, Any]:
    configure_mlflow()
    dataset = task_result.dataset
    feature_columns = task_result.feature_columns
    feature_stats = _feature_stats_payload(
        task_result.train_frame[feature_columns],
        feature_columns,
    )
    prediction_defaults = {
        "prediction_source": "feast_online",
        "feast": {
            "feature_service": dataset.feature_service,
            "entity_keys": list(dataset.entity_keys),
            "allow_direct_input": True,
            "allow_feature_overrides": True,
            "fail_open": False,
        },
    }
    if dataset.task_type == "regression" and "stockout" in dataset.target_column:
        prediction_defaults["forecast_selection"] = {
            "enabled": True,
            "target": "stockout",
            "primary_metric": "root_mean_squared_error",
            "direction": "minimize",
            "fallback_metrics": ["mean_absolute_error", "r2_score", "accuracy"],
        }
        prediction_defaults["output"] = {
            "name": "days_until_stockout",
            "unit": "days",
            "task_type": "regression",
        }
    if isinstance(dataset.prediction_defaults, dict):
        prediction_defaults = _deep_merge_dicts(
            prediction_defaults,
            dataset.prediction_defaults,
        )

    mlflow_dataset = from_pandas(
        dataset.frame,
        source=str(dataset.feature_file),
        name=f"{dataset.task_name}_training_data",
    )
    input_example = _prediction_ready_frame(
        task_result,
        task_result.train_frame[feature_columns].head(3),
    )
    signature = infer_signature(
        input_example,
        task_result.best_result.estimator.predict(input_example),
    )
    training_config = (
        dataset.training_config
        if isinstance(dataset.training_config, dict)
        else {}
    )
    log_input_example = input_example if nested_bool(
        training_config,
        ("mlflow", "input_example", "enabled"),
        True,
    ) else None

    mlflow.set_experiment(dataset.experiment_name)
    with mlflow.start_run(run_name=f"{dataset.task_name}-horizon-routing") as run:
        mlflow.set_tags(
            {
                "developer": "pios_ml",
                "registered_model_name": dataset.model_name,
                "environment": "Training",
                "model_family": task_result.best_result.candidate_name,
                "feast_feature_service": dataset.feature_service,
                "feast_entity_keys": json.dumps(list(dataset.entity_keys)),
                "training_design": "time_aware_horizon_routing",
                "task_type": dataset.task_type,
                "prediction_target": dataset.target_column,
                "description": dataset.description,
                "feature_columns": json.dumps(feature_columns),
            }
        )
        mlflow.log_input(mlflow_dataset, context="training")
        mlflow.log_params(
            {
                "selected_candidate": task_result.best_result.candidate_name,
                "feature_count": len(feature_columns),
                "dataset_rows": len(dataset.frame),
                "train_rows": len(task_result.train_frame),
                "test_rows": len(task_result.test_frame),
                "feature_service": dataset.feature_service,
                "label_file": str(dataset.label_file),
                "feature_file": str(dataset.feature_file),
                "selection_metric": "time_aware_validation_score",
                "prediction_target": dataset.target_column,
            }
        )
        mlflow.log_metrics(task_result.best_result.metrics)
        mlflow.log_metrics(
            {
                f"hard_{key}": value
                for key, value in task_result.best_result.hard_metrics.items()
            }
        )
        for candidate in task_result.candidate_results:
            mlflow.log_metric(
                f"candidate_score_{candidate.candidate_name}",
                candidate.validation_score,
            )
            for metric_name, metric_value in candidate.metrics.items():
                mlflow.log_metric(
                    f"candidate_{candidate.candidate_name}_{metric_name}",
                    metric_value,
                )
        mlflow.log_dict(feature_stats, "feature_stats.json")
        mlflow.log_dict(prediction_defaults, "prediction_defaults.json")

        model_info = mlflow.sklearn.log_model(
            sk_model=task_result.best_result.estimator,
            artifact_path=dataset.model_name,
            signature=signature,
            input_example=log_input_example,
            registered_model_name=dataset.model_name,
        )

        client = MlflowClient()
        client.set_registered_model_alias(
            name=dataset.model_name,
            alias="challenger",
            version=model_info.registered_model_version,
        )
        if os.getenv("PIOS_PROMOTE_TRAINED_MODELS", "0").strip().lower() in {
            "1",
            "true",
            "yes",
            "y",
            "on",
        }:
            client.set_registered_model_alias(
                name=dataset.model_name,
                alias="champion",
                version=model_info.registered_model_version,
            )

        artifact_dir = _ensure_output_dir(dataset.task_name)
        for artifact_path in artifact_dir.glob("*"):
            if artifact_path.is_file():
                mlflow.log_artifact(str(artifact_path))

        if nested_bool(training_config, ("mlflow", "evaluate", "enabled"), True):
            eval_frame = _prediction_ready_frame(
                task_result,
                task_result.test_frame[feature_columns],
            )
            eval_frame[dataset.target_column] = task_result.test_frame[
                dataset.target_column
            ].to_numpy()
            try:
                mlflow.evaluate(
                    model=f"runs:/{run.info.run_id}/{dataset.model_name}",
                    data=eval_frame,
                    targets=dataset.target_column,
                    model_type="classifier"
                    if dataset.task_type == "classification"
                    else "regressor",
                    evaluators=["default"],
                    feature_names=feature_columns,
                    evaluator_config={"default": {"log_model_explainability": False}},
                )
            except Exception as exc:
                mlflow.set_tag("evaluation_warning", str(exc))
        else:
            mlflow.set_tag("evaluation_warning", "skipped_by_training_resource_profile")

        _persist_model_stats(
            model_id=dataset.model_name,
            identifier=dataset.model_name,
            payload=feature_stats,
        )

        return {
            "run_id": run.info.run_id,
            "model_uri": f"runs:/{run.info.run_id}/{dataset.model_name}",
            "registered_model_version": model_info.registered_model_version,
        }


def train_and_log(dataset_builder: Callable[[], TaskDataset]) -> dict[str, Any]:
    dataset = dataset_builder()
    task_result = fit_task(dataset)
    _enforce_metric_bounds(task_result)
    plot_task_result(task_result)
    mlflow_payload = log_task_to_mlflow(task_result)
    return {
        "task_name": dataset.task_name,
        "selected_candidate": task_result.best_result.candidate_name,
        "overall_metrics": task_result.best_result.metrics,
        "hard_metrics": task_result.best_result.hard_metrics,
        **mlflow_payload,
    }


# =============================================================================
# MULTI-HORIZON FORECAST SUPPORT
# =============================================================================


@dataclass(frozen=True)
class MultiHorizonForecastDataset:
    """
    Config for a multi-output, multi-horizon regression task.

    Horizons are identified by their label column names.  Each horizon has a
    declared type — "absolute" or "delta" — which controls whether predictions
    are clipped at 0 during evaluation.

    The same feature matrix is used to train one XGBRegressor per horizon.
    Feast provides features via get_historical_features; entity_df is loaded
    from label_file.
    """

    task_name: str
    model_name: str
    experiment_name: str
    feature_service: str
    entity_keys: tuple[str, ...]
    label_file: Path
    feature_file: Path
    description: str
    horizons: tuple[str, ...]           # ordered target column names
    horizon_types: dict[str, str]       # "absolute" | "delta" per horizon
    hard_mask_builder: Callable[[pd.DataFrame], pd.Series]
    xgb_params: dict[str, Any]
    examples: list[dict[str, Any]]      # MLflow model-card examples
    accuracy_tolerance: float | None = None
    # P8: full raw config dict for reproducibility tracing (rng_seed, n_paths_mc, etc.)
    raw_config: dict = field(default_factory=dict)


@dataclass(frozen=True)
class MultiHorizonTrainResult:
    """Output from train_multi_horizon_forecast."""

    dataset: MultiHorizonForecastDataset
    trained_models: dict[str, Any]                       # short_key → fitted XGBRegressor
    feature_columns: list[str]
    cat_value_maps: dict[str, dict[str, float]]
    per_horizon_metrics: dict[str, dict[str, float]]     # short_key → metrics dict
    per_horizon_hard_metrics: dict[str, dict[str, float]]
    per_horizon_predictions: dict[str, np.ndarray]   
    per_horizon_y_test: dict[str, np.ndarray]        
    train_frame: pd.DataFrame    # full rows (features + labels)
    test_frame: pd.DataFrame
    raw_feature_df: pd.DataFrame  # unencoded features, for input_example + stats
    # P5: per blood type metric slice — None when blood_product_type not found
    per_bloodtype_metrics: dict[str, dict[str, dict[str, float]]] | None = None
    # Bias correction: mean(pred − actual) per short_key on the test set.
    # Subtracted at inference time when apply_bias_correction is true in config.
    bias_corrections: dict[str, float] = field(default_factory=dict)


def _horizon_short_key(horizon: str) -> str:
    """'stock_units_t7' → 't7'  (last segment after the final underscore)."""
    return horizon.rsplit("_", 1)[-1]


def train_multi_horizon_forecast(
    dataset_builder: Callable[[], MultiHorizonForecastDataset],
) -> MultiHorizonTrainResult:
    """
    Generic training loop for multi-horizon multi-output XGBoost regression.

    For each horizon in dataset.horizons:
      - "absolute" type: predictions are clipped at 0.0 during evaluation
        (inventory cannot go negative).
      - "delta" type: predictions are left unclipped — negative values
        represent expected stock depletion and are the primary risk signal.

    When dataset.feature_file is set (pre-aligned parquets), features are
    loaded directly via a merge on entity_keys + event_timestamp and Feast's
    point-in-time join is skipped.  This avoids the O(n²) cross-product that
    causes OOM with large datasets.  Feast is used only when feature_file
    is not set (online-store-only scenario).

    Django setup is handled by model_training_utils at import time.
    """
    dataset = dataset_builder()
    configure_mlflow()

    # ── Load labels ───────────────────────────────────────────────────────────
    entity_df = pd.read_parquet(dataset.label_file)
    entity_df["event_timestamp"] = pd.to_datetime(entity_df["event_timestamp"], utc=True)

    # ── Load features: direct merge if parquet is available, else Feast ───────
    if dataset.feature_file is not None and Path(dataset.feature_file).exists():
        # Direct load: feature & label parquets are pre-aligned on the same
        # entity_keys + event_timestamp, so a simple merge replaces the
        # O(n²) Feast point-in-time join.
        print(
            f"[{dataset.task_name}] Loading features directly from "
            f"'{dataset.feature_file.name}' (skipping Feast join)…"
        )
        feature_df = pd.read_parquet(dataset.feature_file)
        feature_df["event_timestamp"] = pd.to_datetime(
            feature_df["event_timestamp"], utc=True
        )
        # Apply the same column prefix Feast would produce:
        # FeatureView name = "{parquet_stem}_features" → prefix = "{view}__{col}"
        feast_view_name = f"{dataset.feature_file.stem}_features"
        feat_only_cols = [
            c for c in feature_df.columns
            if c not in {*dataset.entity_keys, "event_timestamp"}
        ]
        feature_df = feature_df.rename(
            columns={c: f"{feast_view_name}__{c}" for c in feat_only_cols}
        )
        expected_merge_keys = [*dataset.entity_keys, "event_timestamp"]
        missing_entity_keys = [k for k in expected_merge_keys if k not in entity_df.columns]
        missing_feature_keys = [k for k in expected_merge_keys if k not in feature_df.columns]
        if missing_entity_keys or missing_feature_keys:
            raise ValueError(
                f"[{dataset.task_name}] Cannot directly merge parquet features because "
                f"required merge keys are missing. "
                f"Missing from entity_df: {missing_entity_keys or 'none'}; "
                f"missing from feature_df: {missing_feature_keys or 'none'}."
            )
        training_df = entity_df.merge(
            feature_df, on=expected_merge_keys, how="inner"
        )
    else:
        from feast import FeatureStore  # local import — not all callers use Feast
        feature_store = FeatureStore(repo_path=str(ROOT_DIRECTORY / "feature_repo"))
        print(
            f"[{dataset.task_name}] Fetching features from "
            f"Feast feature service '{dataset.feature_service}'…"
        )
        training_df = feature_store.get_historical_features(
            entity_df=entity_df,
            features=feature_store.get_feature_service(dataset.feature_service),
            full_feature_names=True,
        ).to_df()

    print(f"[{dataset.task_name}] Raw shape: {training_df.shape}")

    # ── Locate label columns (Feast may prefix them) ───────────────────────────
    def _find_col(cols: list[str], keyword: str) -> str:
        matches = [c for c in cols if keyword in c]
        if not matches:
            raise ValueError(
                f"[{dataset.task_name}] Column containing '{keyword}' not found "
                f"in Feast output. Available columns: {cols[:20]}"
            )
        return matches[0]

    label_cols: dict[str, str] = {
        h: _find_col(training_df.columns.tolist(), h) for h in dataset.horizons
    }

    # ── Sort + drop rows with missing labels ────────────────────────────────────
    sort_cols = [*dataset.entity_keys, "event_timestamp"]
    training_df = training_df.sort_values(
        [c for c in sort_cols if c in training_df.columns]
    ).reset_index(drop=True)
    training_df = training_df.dropna(
        subset=list(label_cols.values())
    ).reset_index(drop=True)

    # ── Build encoded feature matrix ───────────────────────────────────────────
    all_label_col_names = list(label_cols.values())
    drop_cols = [*dataset.entity_keys, "event_timestamp"] + all_label_col_names
    df = training_df.drop(
        columns=[c for c in drop_cols if c in training_df.columns]
    )
    raw_feature_df = df.copy(deep=True)

    cat_cols = df.select_dtypes(include=["object"]).columns.tolist()
    encoder = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
    cat_value_maps: dict[str, dict[str, float]] = {}
    if cat_cols:
        df[cat_cols] = encoder.fit_transform(df[cat_cols])
        for col, categories in zip(cat_cols, encoder.categories_):
            cat_value_maps[col] = {
                str(cat): float(i) for i, cat in enumerate(categories)
            }

    for col in df.select_dtypes(include=["int64", "int32"]).columns:
        df[col] = df[col].fillna(0).astype(float)

    feature_columns = df.columns.tolist()
    X = df.values.astype(np.float32)
    Y: dict[str, np.ndarray] = {
        h: training_df[label_cols[h]].values.astype(np.float32)
        for h in dataset.horizons
    }
    print(f"[{dataset.task_name}] Features: {len(feature_columns)}  Rows: {len(X)}")

    # ── Temporal split (per-entity, 80 % train / 20 % test) ───────────────────
    primary_key = dataset.entity_keys[0]
    train_mask: list[int] = []
    for eid in training_df[primary_key].unique():
        idx = training_df.index[training_df[primary_key] == eid].tolist()
        cut = int(len(idx) * 0.8)
        train_mask.extend(idx[:cut])
    train_set = set(train_mask)
    test_mask = [i for i in training_df.index if i not in train_set]

    X_train, X_test = X[train_mask], X[test_mask]
    Y_train = {h: Y[h][train_mask] for h in dataset.horizons}
    Y_test = {h: Y[h][test_mask] for h in dataset.horizons}
    print(f"[{dataset.task_name}] Train: {len(X_train)}  Test: {len(X_test)}")

    # ── Train one XGBRegressor per horizon + evaluate ──────────────────────────
    test_frame = training_df.iloc[test_mask].reset_index(drop=True)
    hard_mask_series = dataset.hard_mask_builder(test_frame).fillna(False)
    hard_idx = hard_mask_series.to_numpy().nonzero()[0]

    trained_models: dict[str, Any] = {}
    per_horizon_metrics: dict[str, dict[str, float]] = {}
    per_horizon_hard_metrics: dict[str, dict[str, float]] = {}
    per_horizon_predictions: dict[str, np.ndarray] = {}   # ← new
    per_horizon_y_test: dict[str, np.ndarray] = {}        # ← new

    for h in dataset.horizons:
        key = _horizon_short_key(h)
        is_absolute = dataset.horizon_types.get(h, "absolute") == "absolute"

        model = XGBRegressor(**dataset.xgb_params)
        model.fit(X_train, Y_train[h], verbose=False)
        trained_models[key] = model

        raw_preds = model.predict(X_test)
        preds = np.clip(raw_preds, 0.0, None) if is_absolute else raw_preds

        per_horizon_predictions[key] = preds
        per_horizon_y_test[key] = Y_test[h]

        metrics = _regression_metrics(
            Y_test[h], preds, accuracy_tolerance=dataset.accuracy_tolerance
        )
        # sign metrics only make sense for delta horizons
        if not is_absolute:
            metrics = {**metrics, **_delta_sign_metrics(Y_test[h], preds)}

        per_horizon_metrics[key] = metrics

        if is_absolute:
            print(
                f"  [{h}]  RMSE={metrics['root_mean_squared_error']:.2f}"
                f"  MAE={metrics['mean_absolute_error']:.2f}"
                f"  WMAPE={metrics['wmape']:.4f}"
            )
        else:
            print(
                f"  [{h}]  MAE={metrics['mean_absolute_error']:.2f}"
                f"  Recall={metrics.get('depletion_recall', float('nan')):.3f}"
                f"  Precision={metrics.get('depletion_precision', float('nan')):.3f}"
            )

        if len(hard_idx) > 0:
            hard_metrics = _regression_metrics(
                Y_test[h][hard_idx],
                preds[hard_idx],
                accuracy_tolerance=dataset.accuracy_tolerance,
            )
            if not is_absolute:
                hard_metrics = {
                    **hard_metrics,
                    **_delta_sign_metrics(Y_test[h][hard_idx], preds[hard_idx]),
                }
        else:
            hard_metrics = {k: float("nan") for k in metrics}
        per_horizon_hard_metrics[key] = hard_metrics

    # ── Bias correction + hard-bias alerts (config-driven) ───────────────────
    eval_cfg = dataset.raw_config.get("evaluation", {})
    apply_bc = bool(eval_cfg.get("apply_bias_correction", False))
    hard_bias_thresholds: dict[str, float] = eval_cfg.get("hard_bias_alert_threshold", {})

    bias_corrections: dict[str, float] = {}
    if apply_bc:
        for h in dataset.horizons:
            key = _horizon_short_key(h)
            correction = float(np.mean(per_horizon_predictions[key] - per_horizon_y_test[key]))
            bias_corrections[key] = round(correction, 4)
        print(f"\n[{dataset.task_name}] Bias corrections (will be subtracted at inference):")
        for k, v in bias_corrections.items():
            print(f"  {k}: {v:+.2f}")

    # Hard-bias monitoring: warn when hard-subset bias exceeds configured threshold
    hard_bias_alerts: list[str] = []
    for h in dataset.horizons:
        key = _horizon_short_key(h)
        hard_bias = per_horizon_hard_metrics[key].get("bias", float("nan"))
        threshold = hard_bias_thresholds.get(key)
        if threshold is not None and not math.isnan(hard_bias) and abs(hard_bias) > threshold:
            msg = (
                f"⚠️  [{dataset.task_name}] Hard-bias ALERT  {key}: "
                f"|{hard_bias:+.2f}| > threshold {threshold:.1f}  "
                f"— model systematically {'under' if hard_bias > 0 else 'over'}-predicts "
                f"on depletion rows."
            )
            hard_bias_alerts.append(msg)
            print(msg)

    # ── P5: per-blood-type RMSE slice ─────────────────────────────────────────
    per_bloodtype_metrics: dict[str, dict[str, dict[str, float]]] = {}
    try:
        bt_col    = _find_col(test_frame.columns.tolist(), "blood_product_type")
        stock_col = _find_col(test_frame.columns.tolist(), "current_stock_units")
    except ValueError:
        bt_col = stock_col = None

    if bt_col is not None:
        short_keys_ordered = [_horizon_short_key(h) for h in dataset.horizons]
        print(f"\n[{dataset.task_name}] Per-blood-type RMSE slice (RMSE/mean_stock=rel_rmse):")
        header = "  ".join(f"{'RMSE/'+k:<18}" for k in short_keys_ordered)
        print(f"  {'Type':<6}  {header}")
        for bt in sorted(test_frame[bt_col].unique()):
            bt_mask = (test_frame[bt_col] == bt).to_numpy()
            bt_idx = np.where(bt_mask)[0]
            if len(bt_idx) < 5:
                continue
            bt_entry: dict[str, dict[str, float]] = {}
            row_parts: list[str] = []
            for h in dataset.horizons:
                hk = _horizon_short_key(h)
                y_true = per_horizon_y_test[hk][bt_idx]
                y_pred = per_horizon_predictions[hk][bt_idx]
                rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
                if stock_col is not None:
                    mean_stock = float(np.mean(test_frame.iloc[bt_idx][stock_col]))
                else:
                    mean_stock = float(np.mean(np.abs(y_true)))
                rel_rmse = rmse / max(mean_stock, 1e-6)
                bt_entry[hk] = {
                    "rmse":         round(rmse, 3),
                    "mean_stock":   round(mean_stock, 3),
                    "relative_rmse": round(rel_rmse, 4),
                    "n":            int(len(bt_idx)),
                }
                row_parts.append(f"{rmse:.1f}/{mean_stock:.0f}={rel_rmse:.3f}")
            per_bloodtype_metrics[bt] = bt_entry
            print(f"  {bt:<6}  " + "  ".join(f"{p:<18}" for p in row_parts))

    return MultiHorizonTrainResult(
        dataset=dataset,
        trained_models=trained_models,
        feature_columns=feature_columns,
        cat_value_maps=cat_value_maps,
        per_horizon_metrics=per_horizon_metrics,
        per_horizon_hard_metrics=per_horizon_hard_metrics,
        per_horizon_predictions=per_horizon_predictions,
        per_horizon_y_test=per_horizon_y_test,
        train_frame=training_df.iloc[train_mask].reset_index(drop=True),
        test_frame=test_frame,
        raw_feature_df=raw_feature_df,
        per_bloodtype_metrics=per_bloodtype_metrics or None,  # P5
        bias_corrections=bias_corrections,
    )
def plot_multi_horizon_forecast(result: MultiHorizonTrainResult) -> None:
    """
    Save diagnostic plots for a trained multi-horizon forecast.

    Outputs (in ARTIFACTS_DIRECTORY / task_name /):
      forecast_actual_vs_predicted.png   — scatter per horizon, sign-coded for deltas
      forecast_sign_diagnostics.png      — TP / FN / FP bars  (delta horizons only)
      forecast_residual_distribution.png — residual histograms with bias annotation
      forecast_horizon_summary.png       — MAE + primary metric across horizons
    """
    output_dir = _ensure_output_dir(result.dataset.task_name)
    _set_plot_style()
    dataset = result.dataset
    horizons = dataset.horizons
    short_keys = [_horizon_short_key(h) for h in horizons]
    n = len(horizons)

    # ── 1. Actual vs Predicted ─────────────────────────────────────────────────
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 4.8))
    axes = [axes] if n == 1 else list(axes)

    for ax, h, key in zip(axes, horizons, short_keys):
        y_true = result.per_horizon_y_test[key]
        y_pred = result.per_horizon_predictions[key]
        is_absolute = dataset.horizon_types.get(h, "absolute") == "absolute"

        if is_absolute:
            colors = "#0a7f8e"
        else:
            # teal = sign correct, red = sign wrong (dangerous)
            correct = (y_true < 0) == (y_pred < 0)
            colors = np.where(correct, "#0a7f8e", "#c0392b")

        ax.scatter(y_true, y_pred, c=colors, alpha=0.45, s=14)
        lo = min(y_true.min(), y_pred.min())
        hi = max(y_true.max(), y_pred.max())
        ax.plot([lo, hi], [lo, hi], color="#8f8f8f", lw=1.2, ls="--")
        ax.set_title(f"{key} — actual vs predicted")
        ax.set_xlabel("Actual")
        ax.set_ylabel("Predicted")

        m = result.per_horizon_metrics[key]
        note = (
            f"WMAPE={m['wmape']:.2%}"
            if is_absolute
            else f"Recall={m.get('depletion_recall', float('nan')):.3f}"
        )
        ax.annotate(note, xy=(0.05, 0.93), xycoords="axes fraction", fontsize=9)

    _save_figure(fig, output_dir, "forecast_actual_vs_predicted")

    # ── 2. Sign diagnostics (delta horizons only) ──────────────────────────────
    delta_pairs = [
        (h, _horizon_short_key(h))
        for h in horizons
        if dataset.horizon_types.get(h, "absolute") == "delta"
    ]
    if delta_pairs:
        fig, axes = plt.subplots(1, len(delta_pairs), figsize=(6 * len(delta_pairs), 4.8))
        axes = [axes] if len(delta_pairs) == 1 else list(axes)

        for ax, (h, key) in zip(axes, delta_pairs):
            m = result.per_horizon_metrics[key]
            tp = int(m.get("tp", 0))
            fn = int(m.get("fn", 0))
            fp = int(m.get("fp", 0))
            recall = m.get("depletion_recall", float("nan"))
            precision = m.get("depletion_precision", float("nan"))

            bars = ax.bar(
                ["TP\n(caught)", "FN\n(missed ❗)", "FP\n(false alarm)"],
                [tp, fn, fp],
                color=["#27ae60", "#c0392b", "#e67e22"],
            )
            ax.bar_label(bars, padding=3, fontsize=10)
            ax.set_title(f"{key} — depletion detection")
            ax.set_ylabel("Count")
            ax.annotate(
                f"Recall={recall:.3f}   Precision={precision:.3f}",
                xy=(0.5, 0.96),
                xycoords="axes fraction",
                ha="center",
                fontsize=9,
            )

        _save_figure(fig, output_dir, "forecast_sign_diagnostics")

    # ── 3. Residual distribution ───────────────────────────────────────────────
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 4.8))
    axes = [axes] if n == 1 else list(axes)

    for ax, h, key in zip(axes, horizons, short_keys):
        residuals = result.per_horizon_y_test[key] - result.per_horizon_predictions[key]
        sns.histplot(residuals, bins=30, kde=True, color="#6c5ce7", ax=ax)
        ax.axvline(0.0, color="#e74c3c", lw=1.5, ls="--")
        ax.set_title(f"{key} — residual distribution")
        ax.set_xlabel("Actual − Predicted")
        bias = result.per_horizon_metrics[key].get("bias", float("nan"))
        ax.annotate(
            f"Bias={bias:.2f}", xy=(0.05, 0.93), xycoords="axes fraction", fontsize=9
        )

    _save_figure(fig, output_dir, "forecast_residual_distribution")

    # ── 4. Horizon summary: MAE + primary metric ───────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))

    # left — MAE: overall vs hard slice
    overall_mae = [result.per_horizon_metrics[k]["mean_absolute_error"] for k in short_keys]
    hard_mae = [
        result.per_horizon_hard_metrics[k].get("mean_absolute_error", float("nan"))
        for k in short_keys
    ]
    x = np.arange(len(short_keys))
    w = 0.35
    axes[0].bar(x - w / 2, overall_mae, w, label="Overall", color="#0a7f8e")
    axes[0].bar(x + w / 2, hard_mae, w, label="Hard (depletion)", color="#c17b12")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(short_keys)
    axes[0].set_title("MAE — overall vs hard slice")
    axes[0].set_ylabel("MAE (units)")
    axes[0].legend()

    # right — primary metric per horizon (WMAPE for absolute, Recall for delta)
    primary_vals, primary_labels, bar_colors = [], [], []
    for h, key in zip(horizons, short_keys):
        m = result.per_horizon_metrics[key]
        is_absolute = dataset.horizon_types.get(h, "absolute") == "absolute"
        if is_absolute:
            primary_vals.append(m["wmape"])
            primary_labels.append(f"{key}\n(WMAPE)")
            bar_colors.append("#0a7f8e")
        else:
            primary_vals.append(m.get("depletion_recall", 0.0))
            primary_labels.append(f"{key}\n(Recall)")
            bar_colors.append("#c17b12")

    finite = [v for v in primary_vals if not math.isnan(v)]
    axes[1].bar(primary_labels, primary_vals, color=bar_colors)
    axes[1].set_title("Primary metric per horizon")
    axes[1].set_ylabel("Value")
    axes[1].set_ylim(0, max(1.0, max(finite) * 1.15) if finite else 1.0)

    fig.suptitle(f"{result.dataset.task_name} — horizon summary", y=1.02)
    _save_figure(fig, output_dir, "forecast_horizon_summary")
    
def log_multi_horizon_forecast_to_mlflow(
    result: MultiHorizonTrainResult,
    pyfunc_wrapper: Any,
) -> dict[str, Any]:
    """
    Log a multi-horizon forecast result to MLflow using a pyfunc wrapper.

    The caller supplies the wrapper because its output contract (dict shape,
    keys, downstream consumers) is application-specific.  Everything else —
    experiment setup, metrics, feature stats, model registration, ModelStats
    persistence — is handled here generically.
    """
    import mlflow.pyfunc  # ensure submodule is loaded

    configure_mlflow()
    dataset = result.dataset

    feature_stats = _feature_stats_payload(
        result.raw_feature_df[result.feature_columns],
        result.feature_columns,
    )
    prediction_defaults = {
        "prediction_source": "feast_online",
        "feast": {
            "feature_service": dataset.feature_service,
            "entity_keys": list(dataset.entity_keys),
            "allow_direct_input": True,
            "allow_feature_overrides": True,
            "fail_open": False,
        },
        "output": {
            "task_type": "multi_horizon_regression",
            "horizons": list(dataset.horizons),
            "horizon_types": dataset.horizon_types,
        },
    }

    input_example = _normalize_datetime_columns_for_mlflow(
        result.raw_feature_df[result.feature_columns].head(3)
    )
    sig_output = pd.DataFrame(pyfunc_wrapper.predict(None, input_example))
    signature = infer_signature(input_example, sig_output)

    flat_metrics: dict[str, float] = {}
    for key, metrics in result.per_horizon_metrics.items():
        for metric_name, metric_value in metrics.items():
            flat_metrics[f"{key}_{metric_name}"] = metric_value
    for key, metrics in result.per_horizon_hard_metrics.items():
        for metric_name, metric_value in metrics.items():
            if not pd.isna(metric_value):
                flat_metrics[f"{key}_hard_{metric_name}"] = metric_value
    plot_multi_horizon_forecast(result)   # ← generates PNGs before mlflow logs them
    mlflow.set_experiment(dataset.experiment_name)
    with mlflow.start_run(run_name=dataset.task_name) as run:
        mlflow.set_tags(
            {
                "developer": "pios_ml",
                "registered_model_name": dataset.model_name,
                "environment": "Training",
                "model_family": "xgboost_multi_horizon",
                "feast_feature_service": dataset.feature_service,
                "feast_entity_keys": json.dumps(list(dataset.entity_keys)),
                "training_design": "multi_horizon_forecast",
                "task_type": "regression",
                "horizons": ",".join(dataset.horizons),
                "description": dataset.description,
                "feature_columns": json.dumps(result.feature_columns),
            }
        )
        mlflow.log_params(
            {
                **dataset.xgb_params,
                "n_features": len(result.feature_columns),
                "train_rows": len(result.train_frame),
                "test_rows": len(result.test_frame),
                "feature_service": dataset.feature_service,
                "horizons": ",".join(dataset.horizons),
                # P8: trace data generation seed for reproducibility
                "data_rng_seed": dataset.raw_config.get("data_generation", {}).get("rng_seed", "unknown"),
                "data_n_paths_mc": dataset.raw_config.get("data_generation", {}).get("n_paths_mc", "unknown"),
            }
        )
        mlflow.log_metrics(flat_metrics)

        mlflow_dataset = from_pandas(
            result.raw_feature_df[result.feature_columns],
            source=str(dataset.feature_file),
            name=f"{dataset.task_name}_training_data",
        )
        mlflow.log_input(mlflow_dataset, context="training")
        mlflow.log_dict(feature_stats, "feature_stats.json")
        mlflow.log_dict(prediction_defaults, "prediction_defaults.json")

        # P5: per-blood-type slice — JSON artifact + flat metrics
        if result.per_bloodtype_metrics:
            mlflow.log_dict(result.per_bloodtype_metrics, "per_bloodtype_metrics.json")
            bt_flat: dict[str, float] = {}
            for bt, horizon_slice in result.per_bloodtype_metrics.items():
                safe_bt = bt.replace("+", "pos").replace("-", "neg")
                for h_key, m in horizon_slice.items():
                    bt_flat[f"bt_{safe_bt}_{h_key}_rmse"]     = m["rmse"]
                    bt_flat[f"bt_{safe_bt}_{h_key}_rel_rmse"] = m["relative_rmse"]
            mlflow.log_metrics(bt_flat)

        # Bias corrections: log as params so they're traceable per run
        if result.bias_corrections:
            mlflow.log_params({
                f"bias_correction_{k}": v
                for k, v in result.bias_corrections.items()
            })
            mlflow.log_dict(result.bias_corrections, "bias_corrections.json")

        # Hard-bias alerts: log as tag so they're visible in the MLflow UI
        eval_cfg = dataset.raw_config.get("evaluation", {})
        hard_bias_thresholds: dict[str, float] = eval_cfg.get("hard_bias_alert_threshold", {})
        alert_parts: list[str] = []
        for h in dataset.horizons:
            key = _horizon_short_key(h)
            hard_bias = result.per_horizon_hard_metrics[key].get("bias", float("nan"))
            threshold = hard_bias_thresholds.get(key)
            if threshold is not None and not math.isnan(hard_bias) and abs(hard_bias) > threshold:
                alert_parts.append(f"{key}={hard_bias:+.1f}>{threshold:.0f}")
        mlflow.set_tag(
            "hard_bias_alerts",
            ",".join(alert_parts) if alert_parts else "none",
        )

        model_info = mlflow.pyfunc.log_model(
            artifact_path=dataset.model_name,
            python_model=pyfunc_wrapper,
            signature=signature,
            input_example=input_example,
            registered_model_name=dataset.model_name,
        )

        client = MlflowClient()
        client.update_registered_model(
            name=dataset.model_name,
            description=dataset.description,
        )
        if dataset.examples:
            client.set_registered_model_tag(
                name=dataset.model_name,
                key="examples",
                value=json.dumps(dataset.examples),
            )
        client.set_registered_model_alias(
            name=dataset.model_name,
            alias="challenger",
            version=model_info.registered_model_version,
        )
        if os.getenv("PIOS_PROMOTE_TRAINED_MODELS", "0").strip().lower() in {
            "1",
            "true",
            "yes",
            "y",
            "on",
        }:
            client.set_registered_model_alias(
                name=dataset.model_name,
                alias="champion",
                version=model_info.registered_model_version,
            )

        artifact_dir = _ensure_output_dir(dataset.task_name)
        for artifact_path in artifact_dir.glob("*"):
            if artifact_path.is_file():
                mlflow.log_artifact(str(artifact_path))

        _persist_model_stats(
            model_id=dataset.model_name,
            identifier="forecast",
            payload=feature_stats,
        )

        return {
            "run_id": run.info.run_id,
            "model_uri": f"runs:/{run.info.run_id}/{dataset.model_name}",
            "registered_model_version": model_info.registered_model_version,
        }
