from __future__ import annotations

import contextlib
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

try:
    import lightgbm as _lightgbm  # noqa: F401
except Exception:  # pragma: no cover - optional tree runtime dependency
    _lightgbm = None

try:
    import xgboost as _xgboost  # noqa: F401
except Exception:  # pragma: no cover - optional tree runtime dependency
    _xgboost = None

try:
    import catboost as _catboost  # noqa: F401
except Exception:  # pragma: no cover - optional tree runtime dependency
    _catboost = None

try:
    import torch
    import torch.nn as nn
except Exception:  # pragma: no cover - optional runtime dependency
    torch = None
    nn = None


class NeuralPreprocessor:
    """Stable class name needed by older notebook-exported joblib artifacts."""

    def __init__(self, feature_cols: list[str]):
        self.feature_cols = list(dict.fromkeys(feature_cols))
        self.cat_cols: list[str] = []
        self.num_cols: list[str] = []
        self.maps: dict[str, dict[str, int]] = {}
        self.scaler = StandardScaler()

    def fit(self, df: pd.DataFrame) -> "NeuralPreprocessor":
        for column in self.feature_cols:
            if df[column].dtype == "object" or str(df[column].dtype).startswith("category"):
                self.cat_cols.append(column)
            else:
                self.num_cols.append(column)
        for column in self.cat_cols:
            values = (
                pd.Series(df[column].astype(str).fillna("__MISSING__").unique())
                .sort_values()
                .tolist()
            )
            self.maps[column] = {value: index + 1 for index, value in enumerate(values)}
        if self.num_cols:
            nums = (
                df[self.num_cols]
                .apply(pd.to_numeric, errors="coerce")
                .fillna(0)
                .astype("float32")
            )
            self.scaler.fit(nums)
        return self

    def transform(self, df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        cats = []
        for column in self.cat_cols:
            values = (
                df[column]
                .astype(str)
                .fillna("__MISSING__")
                .map(self.maps[column])
                .fillna(0)
                .astype("int64")
                .to_numpy()
            )
            cats.append(values)
        cat = np.vstack(cats).T.astype("int64") if cats else np.zeros((len(df), 0), dtype="int64")
        if self.num_cols:
            nums = (
                df[self.num_cols]
                .apply(pd.to_numeric, errors="coerce")
                .fillna(0)
                .astype("float32")
            )
            num = self.scaler.transform(nums).astype("float32")
        else:
            num = np.zeros((len(df), 0), dtype="float32")
        return cat, num

    @property
    def cardinalities(self) -> list[int]:
        return [len(self.maps[column]) + 1 for column in self.cat_cols]


if nn is not None:

    class FTTransformer(nn.Module):
        """Stable class name needed by older notebook-exported joblib artifacts."""

        def __init__(
            self,
            cat_cards: list[int],
            n_num: int,
            out_dim: int,
            hidden: int = 256,
            heads: int = 8,
            layers: int = 2,
            dropout: float = 0.12,
        ):
            super().__init__()
            self.embs = nn.ModuleList([nn.Embedding(card, hidden) for card in cat_cards])
            self.num_proj = nn.ModuleList([nn.Linear(1, hidden) for _ in range(n_num)])
            self.cls = nn.Parameter(torch.zeros(1, 1, hidden))
            enc_layer = nn.TransformerEncoderLayer(
                d_model=hidden,
                nhead=heads,
                dim_feedforward=hidden * 4,
                dropout=dropout,
                activation="gelu",
                batch_first=True,
                norm_first=True,
            )
            self.encoder = nn.TransformerEncoder(enc_layer, num_layers=layers)
            self.head = nn.Sequential(
                nn.LayerNorm(hidden),
                nn.Linear(hidden, hidden * 2),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden * 2, out_dim),
            )
            nn.init.normal_(self.cls, std=0.02)

        def forward(self, cat, num):
            tokens = []
            for index, emb in enumerate(self.embs):
                tokens.append(emb(cat[:, index]).unsqueeze(1))
            for index, proj in enumerate(self.num_proj):
                tokens.append(proj(num[:, index : index + 1]).unsqueeze(1))
            x = torch.cat([self.cls.expand(cat.shape[0], -1, -1)] + tokens, dim=1)
            x = self.encoder(x)
            return self.head(x[:, 0])

else:
    FTTransformer = None


@contextlib.contextmanager
def _legacy_notebook_namespace():
    main_module = sys.modules.get("__main__")
    if main_module is None:
        yield
        return

    previous: dict[str, Any] = {}
    missing: set[str] = set()
    replacements = {
        "NeuralPreprocessor": NeuralPreprocessor,
        "FTTransformer": FTTransformer,
    }
    try:
        for name, value in replacements.items():
            if hasattr(main_module, name):
                previous[name] = getattr(main_module, name)
            else:
                missing.add(name)
            setattr(main_module, name, value)
        yield
    finally:
        for name in missing:
            with contextlib.suppress(AttributeError):
                delattr(main_module, name)
        for name, value in previous.items():
            setattr(main_module, name, value)


def _as_frame(model_input: pd.DataFrame | list[dict[str, Any]] | dict[str, Any] | Any) -> pd.DataFrame:
    if isinstance(model_input, pd.DataFrame):
        return model_input.copy()
    if isinstance(model_input, list):
        return pd.DataFrame(model_input)
    if isinstance(model_input, dict):
        return pd.DataFrame([model_input])
    return pd.DataFrame(model_input)


def _to_python(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if pd.isna(value):
        return None
    return value


def _numeric_series(frame: pd.DataFrame, column: str, default: float = 0.0) -> pd.Series:
    if column in frame.columns:
        raw = frame[column]
    else:
        raw = pd.Series([default] * len(frame), index=frame.index)
    return pd.to_numeric(raw, errors="coerce").fillna(default)


def _sigmoid(values: np.ndarray) -> np.ndarray:
    values = np.clip(values.astype(float), -50, 50)
    return 1.0 / (1.0 + np.exp(-values))


def load_joblib_bundle(path: str | Path) -> Any:
    with _legacy_notebook_namespace():
        return joblib.load(Path(path).expanduser().resolve())


@dataclass
class SotaJoblibModel:
    bundle: Any
    config: dict[str, Any]
    _neural_model: Any | None = None

    @property
    def feature_names(self) -> list[str]:
        configured = self.config.get("features")
        if isinstance(configured, list) and configured:
            return [str(item) for item in configured if str(item).strip()]
        if isinstance(self.bundle, dict) and isinstance(self.bundle.get("feature_cols"), list):
            return [str(item) for item in self.bundle["feature_cols"] if str(item).strip()]
        return []

    def _feature_frame(self, model_input: Any) -> pd.DataFrame:
        frame = _as_frame(model_input)
        features = self.feature_names
        defaults = self.config.get("feature_defaults")
        if not isinstance(defaults, dict):
            defaults = {}
        feature_types = self.config.get("feature_types")
        if not isinstance(feature_types, dict):
            feature_types = {}

        for column in features:
            if column not in frame.columns:
                frame[column] = defaults.get(column, "")
        if features:
            frame = frame[features].copy()

        for column in frame.columns:
            dtype = str(feature_types.get(column) or "").lower()
            if dtype in {"string", "category", "categorical"}:
                frame[column] = frame[column].where(frame[column].notna(), defaults.get(column, "")).astype(str)
            elif dtype in {"datetime", "date", "timestamp"}:
                fallback = defaults.get(column)
                parsed = pd.to_datetime(frame[column], errors="coerce", utc=True)
                if parsed.isna().any():
                    fill_value = pd.Timestamp.now(tz="UTC")
                    if fallback not in (None, ""):
                        fill_value = pd.to_datetime(fallback, errors="coerce", utc=True)
                    parsed = parsed.fillna(fill_value)
                frame[column] = parsed.dt.tz_localize(None)
            else:
                frame[column] = pd.to_numeric(frame[column], errors="coerce").fillna(defaults.get(column, 0.0))
        return frame

    def _tabular_frame(self, frame: pd.DataFrame) -> pd.DataFrame:
        bundle = self.bundle if isinstance(self.bundle, dict) else {}
        columns = list(dict.fromkeys(bundle.get("feature_cols") or self.feature_names))
        x = frame.reindex(columns=columns).copy()
        for column in x.columns:
            if x[column].dtype == "object" or str(x[column].dtype).startswith("category"):
                x[column] = x[column].astype(str).fillna("__MISSING__")
            elif pd.api.types.is_datetime64_any_dtype(x[column]):
                x[column] = pd.to_datetime(x[column], utc=True).astype("int64") / 1e9
            else:
                x[column] = pd.to_numeric(x[column], errors="coerce").fillna(0).astype("float32")
        encoded = pd.get_dummies(x)
        tabular_columns = [str(column) for column in bundle.get("tabular_columns", [])]
        if tabular_columns:
            encoded = encoded.reindex(columns=tabular_columns, fill_value=0)
        return encoded.astype("float32")

    def _neural_predict(self, frame: pd.DataFrame, *, out_dim: int) -> np.ndarray:
        if torch is None or FTTransformer is None:
            raise RuntimeError("Torch is required to serve the configured neural model.")
        bundle = self.bundle
        prep = bundle["neural_preprocessor"]
        cat, num = prep.transform(frame.reindex(columns=prep.feature_cols))
        device = torch.device(os.getenv("PIOS_SOTA_TORCH_DEVICE", "cpu"))
        if device.type == "cuda" and not torch.cuda.is_available():
            device = torch.device("cpu")
        if self._neural_model is None:
            model_cfg = self.config.get("neural_architecture")
            if not isinstance(model_cfg, dict):
                model_cfg = {}
            self._neural_model = FTTransformer(
                prep.cardinalities,
                num.shape[1],
                out_dim=out_dim,
                hidden=int(model_cfg.get("hidden_dim", 256)),
                heads=int(model_cfg.get("n_heads", 8)),
                layers=int(model_cfg.get("n_transformer_layers", 2)),
                dropout=float(model_cfg.get("dropout", 0.12)),
            ).to(device)
            self._neural_model.load_state_dict(bundle["neural_model_state"])
            self._neural_model.eval()

        batch_size = int(self.config.get("predict_batch_size", 4096))
        preds: list[np.ndarray] = []
        with torch.no_grad():
            for start in range(0, len(frame), batch_size):
                stop = start + batch_size
                cat_tensor = torch.tensor(cat[start:stop], dtype=torch.long, device=device)
                num_tensor = torch.tensor(num[start:stop], dtype=torch.float32, device=device)
                preds.append(self._neural_model(cat_tensor, num_tensor).detach().cpu().numpy())
        return np.concatenate(preds, axis=0)

    def _selected_winner(self) -> str:
        if not isinstance(self.bundle, dict):
            return ""
        return str(
            self.config.get("serve_winner")
            or self.bundle.get("winner")
            or self.config.get("winner")
            or "tabular_gbdt"
        )

    def _predict_quantile(self, frame: pd.DataFrame) -> pd.DataFrame:
        if self._selected_winner() == "neural_ft_transformer":
            raw = self._neural_predict(frame, out_dim=3)
            q10, q50, q90 = raw[:, 0], raw[:, 1], raw[:, 2]
        else:
            models = self.bundle.get("tabular_models") if isinstance(self.bundle, dict) else None
            if not isinstance(models, dict):
                raise RuntimeError("Quantile joblib bundle is missing tabular_models.")
            x = self._tabular_frame(frame)
            q10 = np.asarray(models["q10"].predict(x), dtype=float)
            q50 = np.asarray(models["q50"].predict(x), dtype=float)
            q90 = np.asarray(models["q90"].predict(x), dtype=float)
        lower = np.minimum.reduce([q10, q50, q90])
        upper = np.maximum.reduce([q10, q50, q90])
        median = np.clip(q50, lower, upper)
        output_name = str(self.config.get("outputs", {}).get("primary") or "prediction")
        return pd.DataFrame(
            {
                "prediction": np.clip(median, 0, None),
                output_name: np.clip(median, 0, None),
                "q10": np.clip(lower, 0, None),
                "q50": np.clip(median, 0, None),
                "q90": np.clip(upper, 0, None),
            }
        )

    def _predict_regression(self, frame: pd.DataFrame) -> pd.DataFrame:
        if self._selected_winner() == "neural_ft_transformer":
            raw = self._neural_predict(frame, out_dim=1).reshape(-1)
        else:
            model = self.bundle.get("tabular_model") if isinstance(self.bundle, dict) else self.bundle
            if model is None or not hasattr(model, "predict"):
                raise RuntimeError("Regression joblib bundle has no predict method.")
            raw = np.asarray(model.predict(self._tabular_frame(frame)), dtype=float)
        values = np.clip(raw, 0, None)
        output_name = str(self.config.get("outputs", {}).get("primary") or "prediction")
        return pd.DataFrame({"prediction": values, output_name: values})

    def _predict_classification(self, frame: pd.DataFrame) -> pd.DataFrame:
        if self._selected_winner() == "neural_ft_transformer":
            raw = self._neural_predict(frame, out_dim=1).reshape(-1)
            probability = _sigmoid(raw)
        else:
            model = self.bundle.get("tabular_model") if isinstance(self.bundle, dict) else self.bundle
            if model is None:
                raise RuntimeError("Classification joblib bundle is missing tabular_model.")
            x = self._tabular_frame(frame)
            if hasattr(model, "predict_proba"):
                proba = model.predict_proba(x)
                probability = np.asarray(
                    proba[:, 1] if getattr(proba, "ndim", 0) == 2 and proba.shape[1] > 1 else np.ravel(proba),
                    dtype=float,
                )
            else:
                probability = np.asarray(model.predict(x), dtype=float)
        threshold = float(self.config.get("threshold", 0.5))
        output_name = str(self.config.get("outputs", {}).get("primary") or "probability")
        return pd.DataFrame(
            {
                "prediction": (probability >= threshold).astype(int),
                "probability": np.clip(probability, 0.0, 1.0),
                output_name: np.clip(probability, 0.0, 1.0),
                "threshold": threshold,
            }
        )

    def _predict_policy(self, frame: pd.DataFrame) -> pd.DataFrame:
        formula = self.config.get("policy_formula")
        if not isinstance(formula, dict) or not formula:
            numeric = frame.apply(pd.to_numeric, errors="coerce").fillna(0.0)
            raw = numeric.mean(axis=1).to_numpy(dtype=float) if not numeric.empty else np.zeros(len(frame))
        else:
            raw = np.zeros(len(frame), dtype=float)
            for column, weight in formula.items():
                raw += _numeric_series(frame, str(column), 0.0).to_numpy(dtype=float) * float(weight)
        min_val = float(np.nanmin(raw)) if len(raw) else 0.0
        max_val = float(np.nanmax(raw)) if len(raw) else 0.0
        if max_val != min_val:
            scores = (raw - min_val) / max(max_val - min_val, 1e-9)
        else:
            scores = np.clip(raw, 0.0, 1.0)
        scores = np.clip(scores, 0.0, 1.0)
        thresholds = self.config.get("action_thresholds")
        thresholds = thresholds if isinstance(thresholds, dict) else {}
        default_action = str(self.config.get("outputs", {}).get("default_action") or "no_action")
        actions = []
        for score in scores:
            action = default_action
            for name, threshold in sorted(thresholds.items(), key=lambda item: float(item[1]), reverse=True):
                if score >= float(threshold):
                    action = str(name)
                    break
            actions.append(action.replace("_", " "))
        output_name = str(self.config.get("outputs", {}).get("primary") or "prediction")
        display_col = str(self.config.get("outputs", {}).get("display_score") or f"display_{output_name}")
        return pd.DataFrame(
            {
                "prediction": scores,
                output_name: scores,
                display_col: np.round(scores * 100).astype(int),
                "recommended_action": actions,
            }
        )

    def _predict_simulator(self, frame: pd.DataFrame) -> pd.DataFrame:
        simulator = self.config.get("simulator")
        simulator = simulator if isinstance(simulator, dict) else {}
        horizons = [int(value) for value in simulator.get("horizons_days", [7, 30, 90, 180])]
        current = _numeric_series(
            frame,
            str(simulator.get("stock_column", "current_inventory")),
            0.0,
        ).to_numpy(dtype=float)
        demand = _numeric_series(
            frame,
            str(simulator.get("demand_column", "units_used")),
            0.0,
        ).to_numpy(dtype=float)
        supply = _numeric_series(
            frame,
            str(simulator.get("supply_column", "units_collected")),
            0.0,
        ).to_numpy(dtype=float)
        waste = _numeric_series(
            frame,
            str(simulator.get("waste_column", "wastage")),
            0.0,
        ).to_numpy(dtype=float)
        safety = float(simulator.get("safety_threshold", 1.0))
        daily_depletion = np.maximum(demand + waste - supply, 0.0)
        days = np.full(len(frame), np.inf, dtype=float)
        depletion_mask = daily_depletion > 0
        days[depletion_mask] = (
            np.maximum(current[depletion_mask] - safety, 0.0)
            / daily_depletion[depletion_mask]
        )
        result: dict[str, Any] = {
            "expected_days_until_stockout": np.where(np.isfinite(days), days, np.nan),
            "p50_days_until_stockout": np.where(np.isfinite(days), days, np.nan),
            "median_status": np.where(np.isfinite(days), "estimated", "not_reached"),
        }
        for horizon in horizons:
            result[f"p_stockout_0_{horizon}d"] = np.where(
                np.isfinite(days),
                np.clip((horizon - days) / max(horizon, 1) + 0.5, 0.0, 1.0),
                0.0,
            )
        result["prediction"] = result.get(f"p_stockout_0_{horizons[0]}d", np.zeros(len(frame)))
        output = pd.DataFrame(result)
        return output.where(pd.notna(output), None)

    def _predict_generic(self, frame: pd.DataFrame) -> pd.DataFrame:
        model = self.bundle
        if isinstance(model, dict) and "model" in model:
            model = model["model"]
        if not hasattr(model, "predict"):
            raise RuntimeError("Joblib bundle has no supported prediction interface.")
        predictions = model.predict(frame)
        return _as_frame({"prediction": predictions[0] if len(predictions) else None})

    def predict_frame(self, model_input: Any) -> pd.DataFrame:
        frame = self._feature_frame(model_input)
        task_type = str(self.config.get("task_type") or "").strip()
        bundle_type = str(self.bundle.get("type") or "") if isinstance(self.bundle, dict) else ""
        if bundle_type == "policy" or task_type == "policy":
            return self._predict_policy(frame)
        if bundle_type == "hybrid_simulator" or task_type == "simulator":
            return self._predict_simulator(frame)
        if task_type == "quantile":
            return self._predict_quantile(frame)
        if task_type == "regression":
            return self._predict_regression(frame)
        if task_type == "classification":
            return self._predict_classification(frame)
        return self._predict_generic(frame)


def load_sota_joblib_model(path: str | Path, config: dict[str, Any]) -> SotaJoblibModel:
    return SotaJoblibModel(bundle=load_joblib_bundle(path), config=dict(config))


def normalize_prediction_output(predictions: pd.DataFrame) -> dict[str, Any]:
    if predictions.empty:
        return {"prediction": None}
    row = {
        str(key): _to_python(value)
        for key, value in predictions.iloc[0].to_dict().items()
    }
    if "prediction" not in row and len(row) == 1:
        row["prediction"] = next(iter(row.values()))
    return row
