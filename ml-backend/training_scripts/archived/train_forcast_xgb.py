"""
train_forcast_xgb.py

Multi-horizon blood stock forecast — three XGBoost regressors.

  t1  → ABSOLUTE stock level in 1 day  (≥ 0; drives 24H period filter)
  t7  → DELTA stock_at_t7  − stock_now (can be negative; drives 1W filter)
  t30 → DELTA stock_at_t30 − stock_now (can be negative; drives monthly view)

Negative delta = expected depletion (primary risk signal; NOT clipped to 0).
Absolute future stock: projected_t7 = stock_now + t7 ; projected_t30 = stock_now + t30

All infrastructure (Django setup, Feast, temporal split, MLflow logging,
ModelStats persistence) is handled by model_training_utils.  Only the
multi-output pyfunc wrapper — legitimately specific to this model's dict-output
contract — is defined here.

XGBoost hyperparameters and all model metadata are loaded from:
    config/blood_stock_forecast_model.json
"""

from __future__ import annotations

import json
from pathlib import Path

import mlflow.pyfunc
import numpy as np
import pandas as pd
from xgboost import XGBRegressor

from model_training_utils import (
    FEATURE_LABELS_DIRECTORY,
    MultiHorizonForecastDataset,
    _safe_numeric,
    log_multi_horizon_forecast_to_mlflow,
    train_multi_horizon_forecast,
)

# ─────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────

_CONFIG_PATH = (
    Path(__file__).resolve().parent.parent / "config" / "blood_stock_forecast_model.json"
)


def _load_config() -> dict:
    with open(_CONFIG_PATH, "r", encoding="utf-8") as fh:
        return json.load(fh)


# ─────────────────────────────────────────────────────────────
# TASK BUILDER
# ─────────────────────────────────────────────────────────────


def build_forecast_task() -> MultiHorizonForecastDataset:
    """
    Build a MultiHorizonForecastDataset from blood_stock_forecast_model.json.

    No Feast calls or heavy I/O — only JSON loading.  All training
    infrastructure is delegated to train_multi_horizon_forecast().
    """
    cfg = _load_config()
    horizons = tuple(h["column"] for h in cfg["horizons"])
    horizon_types: dict[str, str] = {
        h["column"]: h["type"] for h in cfg["horizons"]
    }
    entity_keys = tuple(cfg["entity_keys"])
    label_file = FEATURE_LABELS_DIRECTORY / cfg["label_file"]
    feature_file = FEATURE_LABELS_DIRECTORY / cfg["feature_file"]

    def hard_mask(frame: pd.DataFrame) -> pd.Series:
        """Hard cases: rows where 7-day delta is negative (stock depletion).

        Feast returns columns with full_feature_names=True so the bare name
        'stock_units_t7' may be prefixed (e.g. 'forecast_service__stock_units_t7').
        Use a substring search — the same approach as _find_col() in
        train_multi_horizon_forecast — so the mask is never silently all-False.
        """
        col = next((c for c in frame.columns if "stock_units_t7" in c), None)
        if col is not None:
            return _safe_numeric(frame[col]) < 0
        return pd.Series(False, index=frame.index)

    return MultiHorizonForecastDataset(
        task_name=cfg["id"],
        model_name=cfg["registered_model_name"],
        experiment_name=cfg["experiment_name"],
        feature_service=cfg["feature_service"],
        entity_keys=entity_keys,
        label_file=label_file,
        feature_file=feature_file,
        description=cfg["description"],
        horizons=horizons,
        horizon_types=horizon_types,
        hard_mask_builder=hard_mask,
        xgb_params=cfg["xgb_params"],
        examples=cfg.get("examples", []),
        accuracy_tolerance=cfg.get("evaluation", {}).get("accuracy_tolerance"),
        raw_config=cfg,  # P8: pass full config for reproducibility tracing
    )



# ─────────────────────────────────────────────────────────────
# PYFUNC WRAPPER
# ─────────────────────────────────────────────────────────────


class ForecastXGBWrapper(mlflow.pyfunc.PythonModel):
    """
    MLflow pyfunc wrapper for three XGBoost horizon models.

    Input:  flat 2-D DataFrame with forecast feature columns.
    Output: list of dicts — one per row:
        {
            "stock_units_t1":  <float>  — absolute predicted stock in 1 day (≥ 0)
            "stock_units_t7":  <float>  — DELTA: expected change over 7 days
                                          (negative = depleting)
            "stock_units_t30": <float>  — DELTA: expected change over 30 days
                                          (negative = depleting)
        }

    To reconstruct absolute future stock:
        projected_t7  = stock_now + stock_units_t7
        projected_t30 = stock_now + stock_units_t30
    """

    def __init__(
        self,
        models: dict[str, XGBRegressor],
        feature_cols: list[str],
        cat_value_maps: dict[str, dict[str, float]],
        bias_corrections: dict[str, float] | None = None,
    ) -> None:
        self.models = models
        self.feature_cols = feature_cols
        self.cat_value_maps = cat_value_maps
        # Bias correction: mean(pred − actual) computed on test set.
        # Subtracted from raw predictions to remove systematic over/under-prediction.
        self.bias_corrections: dict[str, float] = bias_corrections or {}

    def _encode(self, df: pd.DataFrame) -> np.ndarray:
        """Align columns, encode categoricals, return float32 matrix."""
        df = df.copy()
        for col in self.feature_cols:
            if col not in df.columns:
                df[col] = 0.0
        df = df[self.feature_cols]

        for col, mapping in self.cat_value_maps.items():
            if col not in df.columns:
                continue
            numeric = pd.to_numeric(df[col], errors="coerce")
            encoded = df[col].astype(str).str.strip().map(mapping).fillna(-1.0)
            df[col] = numeric.where(numeric.notna(), encoded)

        for col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)

        return df.values.astype(np.float32)

    def predict(
        self,
        context: mlflow.pyfunc.PythonModelContext,
        model_input: pd.DataFrame,
    ) -> list[dict]:
        X = self._encode(model_input)
        bc = self.bias_corrections
        # t1 is absolute stock — apply correction then clip at 0
        t1_raw = self.models["t1"].predict(X) - bc.get("t1", 0.0)
        t1_preds = np.clip(t1_raw, 0.0, None)
        # t7/t30 are deltas — negative means depleting; apply correction, no clip
        t7_preds  = self.models["t7"].predict(X)  - bc.get("t7",  0.0)
        t30_preds = self.models["t30"].predict(X) - bc.get("t30", 0.0)
        return [
            {
                "stock_units_t1": round(float(t1_preds[i]), 1),
                "stock_units_t7": round(float(t7_preds[i]), 1),
                "stock_units_t30": round(float(t30_preds[i]), 1),
            }
            for i in range(len(X))
        ]


# ─────────────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    result = train_multi_horizon_forecast(build_forecast_task)
    wrapper = ForecastXGBWrapper(
        models=result.trained_models,
        feature_cols=result.feature_columns,
        cat_value_maps=result.cat_value_maps,
        bias_corrections=result.bias_corrections,
    )
    payload = log_multi_horizon_forecast_to_mlflow(result, wrapper)
    print(
        f"\nModel registered: {result.dataset.model_name}"
        f" v{payload['registered_model_version']} @challenger"
    )

    # ── Post-registration bias verification ──────────────────────────────────
    # ── Post-registration bias verification ──────────────────────────────────
    print("\n[blood_stock_forecast] Post-correction bias check (loading challenger)…")
    try:
        challenger = mlflow.pyfunc.load_model(
            f"models:/{result.dataset.model_name}@challenger"
        )
        raw = challenger.predict(result.test_frame)
        print(f"  predict() type: {type(raw)}")
        print(f"  first element:  {raw[0] if hasattr(raw, '__getitem__') else raw}")
        print(f"  per_horizon_y_test keys:   {list(result.per_horizon_y_test.keys())}")
        preds_check = pd.DataFrame(challenger.predict(result.test_frame))

        print("  Horizon   BC applied    Corrected residual mean")
        for h, pred_col, y_key in [
            ("t1",  "stock_units_t1",  "t1"),   # adjust y_key to match actual keys
            ("t7",  "stock_units_t7",  "t7"),
            ("t30", "stock_units_t30", "t30"),
        ]:
            actuals   = np.array(result.per_horizon_y_test[y_key])
            corrected = (preds_check[pred_col].values - actuals).mean()
            bc_val    = result.bias_corrections.get(h, 0.0)
            print(f"  {h:<5}     {bc_val:+.2f}          {corrected:+.2f}")

    except Exception as e:
        print(f"  [WARN] Post-check failed: {e}")
    # ─────────────────────────────────────────────────────────────────────────

    print("Done.")
