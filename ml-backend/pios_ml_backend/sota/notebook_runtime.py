from __future__ import annotations

import contextlib
import json
import os
import sys
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
    import mlflow
    import mlflow.pyfunc
except Exception:  # pragma: no cover - exercised in environments without mlflow
    mlflow = None

try:
    import torch
    import torch.nn as nn
except Exception:  # pragma: no cover - torch is an optional serving dependency
    torch = None
    nn = None


class NeuralPreprocessor:
    """Stable module-level copy of the notebook preprocessor used in old pickles."""

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

    class NumericalPLR(nn.Module):
        """Piecewise Linear Representation for numerical features."""

        def __init__(self, n_features: int, n_bins: int = 64, hidden: int = 256):
            super().__init__()
            self.n_features = n_features
            self.n_bins = n_bins
            self.edges = nn.Parameter(torch.linspace(0, 1, n_bins).unsqueeze(0).repeat(n_features, 1))
            self.proj = nn.Linear(n_bins, hidden)

        def forward(self, x):
            x_norm = torch.sigmoid(x)
            diff = x_norm.unsqueeze(-1) - self.edges.unsqueeze(0)
            plr = torch.clamp(diff, 0, 1 / self.n_bins) * self.n_bins
            return self.proj(plr)

    class ReGLU(nn.Module):
        def __init__(self, d_in: int, d_out: int):
            super().__init__()
            self.linear = nn.Linear(d_in, d_out * 2)

        def forward(self, x):
            x, gate = self.linear(x).chunk(2, dim=-1)
            return x * torch.relu(gate)

    class FTTransformerBlock(nn.Module):
        def __init__(self, hidden: int, heads: int, dropout: float, ffn_mult: float):
            super().__init__()
            self.norm1 = nn.LayerNorm(hidden)
            self.attn = nn.MultiheadAttention(hidden, heads, dropout=dropout, batch_first=True)
            self.norm2 = nn.LayerNorm(hidden)
            ffn_dim = int(hidden * ffn_mult)
            self.ffn = nn.Sequential(ReGLU(hidden, ffn_dim), nn.Dropout(dropout), nn.Linear(ffn_dim, hidden))
            self.drop = nn.Dropout(dropout)

        def forward(self, x):
            h = self.norm1(x)
            h, _ = self.attn(h, h, h)
            x = x + self.drop(h)
            h = self.norm2(x)
            x = x + self.drop(self.ffn(h))
            return x

    class FTTransformer(nn.Module):
        """Stable module-level copy of the notebook FT-Transformer architecture."""

        def __init__(
            self,
            cat_cards: list[int],
            n_num: int,
            out_dim: int,
            hidden: int = 256,
            heads: int = 8,
            layers: int = 2,
            dropout: float = 0.12,
            n_plr_bins: int = 64,
            ffn_multiplier: float = 4 / 3,
        ):
            super().__init__()
            self.embs = nn.ModuleList([nn.Embedding(card, hidden) for card in cat_cards])
            self.use_plr = n_num > 0
            if self.use_plr:
                self.num_plr = NumericalPLR(n_num, n_plr_bins, hidden)
            self.cls = nn.Parameter(torch.zeros(1, 1, hidden))
            self.blocks = nn.ModuleList([FTTransformerBlock(hidden, heads, dropout, ffn_multiplier) for _ in range(layers)])
            self.head = nn.Sequential(
                nn.LayerNorm(hidden),
                nn.Linear(hidden, hidden),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden, out_dim),
            )
            nn.init.normal_(self.cls, std=0.02)

        def forward(self, cat, num):
            toks = []
            for j, emb in enumerate(self.embs):
                toks.append(emb(cat[:, j]).unsqueeze(1))
            if self.use_plr:
                num_toks = self.num_plr(num)
                toks.append(num_toks)
            x = torch.cat([self.cls.expand(cat.shape[0], -1, -1)] + toks, dim=1)
            for block in self.blocks:
                x = block(x)
            return self.head(x[:, 0])

else:
    NumericalPLR = None
    ReGLU = None
    FTTransformerBlock = None
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
        "NumericalPLR": NumericalPLR,
        "ReGLU": ReGLU,
        "FTTransformerBlock": FTTransformerBlock,
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


def load_notebook_bundle(path: str | Path) -> Any:
    with _legacy_notebook_namespace():
        return joblib.load(Path(path).expanduser().resolve())


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


def _configured_output_range(config: dict[str, Any], bundle: dict[str, Any]) -> tuple[float | None, float | None]:
    schema = config.get("score_output_schema")
    if not isinstance(schema, dict):
        schema = bundle.get("score_output_schema")
    if not isinstance(schema, dict):
        return None, None
    range_value = schema.get("range")
    if not isinstance(range_value, (list, tuple)) or len(range_value) != 2:
        return None, None
    lower = float(range_value[0]) if range_value[0] is not None else None
    upper = float(range_value[1]) if range_value[1] is not None else None
    return lower, upper


def _normalize_neural_state_dict(state: Any) -> Any:
    if not isinstance(state, dict) or not state:
        return state
    keys = list(state.keys())
    for prefix in ("_orig_mod.", "module."):
        if all(isinstance(key, str) and key.startswith(prefix) for key in keys):
            return {key.removeprefix(prefix): value for key, value in state.items()}
    return state


class SotaNotebookModel(mlflow.pyfunc.PythonModel if mlflow is not None else object):
    """MLflow pyfunc wrapper for SOTA notebook artifacts."""

    def __init__(self, config: dict[str, Any]):
        if mlflow is not None:
            super().__init__()
        self.config = dict(config)
        self._bundle: Any | None = None
        self._neural_model: Any | None = None

    def __getstate__(self) -> dict[str, Any]:
        state = self.__dict__.copy()
        state["_bundle"] = None
        state["_neural_model"] = None
        return state

    def load_context(self, context: Any) -> None:
        artifact_path = None
        artifacts = getattr(context, "artifacts", None)
        if isinstance(artifacts, dict):
            artifact_path = artifacts.get("model_artifact")
        if not artifact_path:
            artifact_path = self.config.get("artifact_path")
        if not artifact_path:
            raise RuntimeError("SOTA pyfunc model has no configured artifact path.")
        self._bundle = load_notebook_bundle(artifact_path)

    def _ensure_loaded(self) -> Any:
        if self._bundle is None:
            artifact_path = self.config.get("artifact_path")
            if not artifact_path:
                raise RuntimeError("SOTA model is not loaded and has no artifact_path.")
            self._bundle = load_notebook_bundle(artifact_path)
        return self._bundle

    def _feature_frame(self, model_input: Any) -> pd.DataFrame:
        frame = _as_frame(model_input)
        features = [str(item) for item in self.config.get("features", []) if str(item).strip()]
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

    def _tabular_frame(self, frame: pd.DataFrame, bundle: dict[str, Any]) -> pd.DataFrame:
        columns = list(dict.fromkeys(bundle.get("feature_cols") or self.config.get("features") or []))
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

    def _neural_predict(self, frame: pd.DataFrame, bundle: dict[str, Any], *, out_dim: int) -> np.ndarray:
        if torch is None or FTTransformer is None:
            raise RuntimeError("Torch is required to serve the configured neural SOTA model.")
        prep = bundle["neural_preprocessor"]
        cat, num = prep.transform(frame.reindex(columns=prep.feature_cols))
        device = torch.device(os.getenv("PIOS_SOTA_TORCH_DEVICE", "cpu"))
        if device.type == "cuda" and not torch.cuda.is_available():
            device = torch.device("cpu")
        if self._neural_model is None:
            model_cfg = self.config.get("neural_architecture")
            if not isinstance(model_cfg, dict):
                model_cfg = {}
            state = _normalize_neural_state_dict(bundle["neural_model_state"])
            # Infer architecture from state dict
            hidden = int(state["cls"].shape[-1])
            layers = len(set(int(k.split(".")[1]) for k in state if k.startswith("blocks.")))
            plr_keys = [k for k in state if k == "num_plr.edges"]
            n_plr_bins = int(state[plr_keys[0]].shape[1]) if plr_keys else int(model_cfg.get("n_plr_bins", 64))
            # Infer ffn_multiplier from ReGLU linear weight
            ffn_key = "blocks.0.ffn.0.linear.weight"
            if ffn_key in state:
                ffn_out = state[ffn_key].shape[0] // 2  # ReGLU doubles
                ffn_mult = ffn_out / hidden
            else:
                ffn_mult = float(model_cfg.get("ffn_multiplier", 4 / 3))
            heads = int(model_cfg.get("n_heads", 8))
            dropout = float(model_cfg.get("dropout", 0.12))
            self._neural_model = FTTransformer(
                prep.cardinalities,
                num.shape[1],
                out_dim=out_dim,
                hidden=hidden,
                heads=heads,
                layers=layers,
                dropout=dropout,
                n_plr_bins=n_plr_bins,
                ffn_multiplier=ffn_mult,
            ).to(device)
            self._neural_model.load_state_dict(state)
            self._neural_model.eval()

        batch_size = int(self.config.get("predict_batch_size", 4096))
        preds: list[np.ndarray] = []
        with torch.no_grad():
            for start in range(0, len(frame), batch_size):
                stop = start + batch_size
                cat_tensor = torch.tensor(cat[start:stop], dtype=torch.long, device=device)
                num_tensor = torch.tensor(num[start:stop], dtype=torch.float32, device=device)
                out = self._neural_model(cat_tensor, num_tensor)
                preds.append(out.detach().cpu().numpy())
        return np.concatenate(preds, axis=0)

    def _selected_winner(self, bundle: dict[str, Any]) -> str:
        return str(
            self.config.get("serve_winner")
            or bundle.get("winner")
            or self.config.get("winner")
            or "tabular_gbdt"
        )

    def _predict_quantile(self, frame: pd.DataFrame, bundle: dict[str, Any]) -> pd.DataFrame:
        winner = self._selected_winner(bundle)
        if winner == "neural_ft_transformer":
            raw = self._neural_predict(frame, bundle, out_dim=3)
            q10, q50, q90 = raw[:, 0], raw[:, 1], raw[:, 2]
        else:
            x = self._tabular_frame(frame, bundle)
            models = bundle.get("tabular_models")
            if not isinstance(models, dict):
                raise RuntimeError("Quantile SOTA bundle is missing tabular_models.")
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

    def _predict_regression(self, frame: pd.DataFrame, bundle: dict[str, Any]) -> pd.DataFrame:
        winner = self._selected_winner(bundle)
        if winner == "neural_ft_transformer":
            raw = self._neural_predict(frame, bundle, out_dim=1).reshape(-1)
        else:
            model = bundle.get("tabular_model")
            if model is None:
                raise RuntimeError("Regression SOTA bundle is missing tabular_model.")
            raw = np.asarray(model.predict(self._tabular_frame(frame, bundle)), dtype=float)
        lower, upper = _configured_output_range(self.config, bundle)
        values = np.clip(raw, 0.0 if lower is None else lower, upper)
        output_name = str(self.config.get("outputs", {}).get("primary") or "prediction")
        return pd.DataFrame({"prediction": values, output_name: values})

    def _predict_classification(self, frame: pd.DataFrame, bundle: dict[str, Any]) -> pd.DataFrame:
        winner = self._selected_winner(bundle)
        if winner == "neural_ft_transformer":
            raw = self._neural_predict(frame, bundle, out_dim=1).reshape(-1)
            probability = _sigmoid(raw)
        else:
            model = bundle.get("tabular_model")
            if model is None:
                raise RuntimeError("Classification SOTA bundle is missing tabular_model.")
            proba = model.predict_proba(self._tabular_frame(frame, bundle))
            probability = np.asarray(proba[:, 1] if proba.ndim == 2 and proba.shape[1] > 1 else proba.ravel(), dtype=float)
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
        scores = (raw - min_val) / max(max_val - min_val, 1e-9) if max_val != min_val else np.clip(raw, 0.0, 1.0)
        scores = np.clip(scores, 0.0, 1.0)
        thresholds = self.config.get("action_thresholds")
        if not isinstance(thresholds, dict):
            thresholds = {}
        actions = []
        for score in scores:
            action = "do_not_contact_low_score"
            for name, threshold in sorted(thresholds.items(), key=lambda item: float(item[1]), reverse=True):
                if score >= float(threshold):
                    action = str(name)
                    break
            actions.append(action.replace("_", " "))
        return pd.DataFrame(
            {
                "prediction": scores,
                "donor_priority_score": scores,
                "display_donor_score": np.round(scores * 100).astype(int),
                "recommended_action": actions,
                "action_reason": ["SOTA configurable donor priority policy"] * len(scores),
            }
        )

    def _predict_simulator(self, frame: pd.DataFrame) -> pd.DataFrame:
        simulator = self.config.get("simulator")
        if not isinstance(simulator, dict):
            simulator = {}
        horizons = [int(value) for value in simulator.get("horizons_days", [7, 30, 90, 180])]
        current = _numeric_series(frame, str(simulator.get("stock_column", "current_inventory")), 0.0).to_numpy(dtype=float)
        demand = _numeric_series(frame, str(simulator.get("demand_column", "units_used")), 0.0).to_numpy(dtype=float)
        supply = _numeric_series(frame, str(simulator.get("supply_column", "units_collected")), 0.0).to_numpy(dtype=float)
        waste = _numeric_series(frame, str(simulator.get("waste_column", "wastage")), 0.0).to_numpy(dtype=float)
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
            key = f"p_stockout_0_{horizon}d"
            result[key] = np.where(
                np.isfinite(days),
                np.clip((horizon - days) / max(horizon, 1) + 0.5, 0.0, 1.0),
                0.0,
            )
        result["prediction"] = result.get("p_stockout_0_7d", np.zeros(len(frame)))
        output = pd.DataFrame(result)
        return output.where(pd.notna(output), None)

    def predict(self, context, model_input) -> pd.DataFrame:
        bundle = self._ensure_loaded()
        frame = self._feature_frame(model_input)
        task_type = str(self.config.get("task_type") or "").strip()

        if isinstance(bundle, dict) and str(bundle.get("type") or "") == "policy":
            return self._predict_policy(frame)
        if isinstance(bundle, dict) and str(bundle.get("type") or "") == "hybrid_simulator":
            return self._predict_simulator(frame)
        if task_type == "quantile":
            return self._predict_quantile(frame, bundle)
        if task_type == "regression":
            return self._predict_regression(frame, bundle)
        if task_type == "classification":
            return self._predict_classification(frame, bundle)
        if task_type == "policy":
            return self._predict_policy(frame)
        if task_type == "simulator":
            return self._predict_simulator(frame)
        raise RuntimeError(f"Unsupported SOTA task_type: {task_type}")


def model_config_from_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).expanduser().resolve().read_text(encoding="utf-8"))
