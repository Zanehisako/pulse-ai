from __future__ import annotations

from typing import Any

import mlflow
import numpy as np
import pandas as pd

from .simulator import run_monte_carlo_paths


class InventoryRiskSimulatorPyfunc(mlflow.pyfunc.PythonModel):
    def __init__(
        self,
        demand_model_uri: str,
        supply_model_uri: str,
        waste_model_uri: str,
        demand_schema: dict[str, Any] | None = None,
        supply_schema: dict[str, Any] | None = None,
        waste_schema: dict[str, Any] | None = None,
        n_paths: int = 1000,
        horizons_days: list[int] | None = None,
        demand_cv: float = 0.15,
        supply_cv: float = 0.20,
        waste_cv: float = 0.15,
        safety_threshold: float = 1.0,
        stock_column: str = "current_inventory",
        inventory_policy: dict[str, Any] | None = None,
        seed: int | None = None,
    ):
        self.demand_model_uri = demand_model_uri
        self.supply_model_uri = supply_model_uri
        self.waste_model_uri = waste_model_uri
        self.demand_schema = demand_schema or {}
        self.supply_schema = supply_schema or {}
        self.waste_schema = waste_schema or {}
        self.n_paths = n_paths
        self.horizons_days = horizons_days or [7, 30, 90, 180]
        self.demand_cv = demand_cv
        self.supply_cv = supply_cv
        self.waste_cv = waste_cv
        self.safety_threshold = safety_threshold
        self.stock_column = stock_column
        self.inventory_policy = inventory_policy or {}
        self.seed = seed

        self._demand_model = None
        self._supply_model = None
        self._waste_model = None

    def clear_child_model_cache(self) -> None:
        self._demand_model = None
        self._supply_model = None
        self._waste_model = None

    def __getstate__(self) -> dict[str, Any]:
        state = self.__dict__.copy()
        state["_demand_model"] = None
        state["_supply_model"] = None
        state["_waste_model"] = None
        return state

    def __setstate__(self, state: dict[str, Any]) -> None:
        self.__dict__.update(state)
        self.clear_child_model_cache()

    def _load_child_models(self):
        if self._demand_model is None:
            self._demand_model = mlflow.pyfunc.load_model(self.demand_model_uri)
        if self._supply_model is None:
            self._supply_model = mlflow.pyfunc.load_model(self.supply_model_uri)
        if self._waste_model is None:
            self._waste_model = mlflow.pyfunc.load_model(self.waste_model_uri)

    def _align_input(self, df: pd.DataFrame, schema: dict[str, Any]) -> pd.DataFrame:
        input_cols = schema.get("input_columns", [])
        defaults = schema.get("defaults", {})
        dtypes = schema.get("dtypes", {})
        if not input_cols:
            return df.copy()
        aligned = df.reindex(columns=input_cols)
        for col in input_cols:
            default_val = defaults.get(col, 0.0)
            if col not in aligned.columns:
                continue
            dtype_text = str(dtypes.get(col, "")).lower()
            if "string" in dtype_text or "str" in dtype_text:
                aligned[col] = aligned[col].fillna(default_val).astype(str)
            elif "long" in dtype_text or "integer" in dtype_text or dtype_text == "int":
                integer_dtype = "int32" if "integer" in dtype_text else "int64"
                aligned[col] = (
                    pd.to_numeric(aligned[col], errors="coerce")
                    .fillna(default_val)
                    .round()
                    .astype(integer_dtype)
                )
            else:
                aligned[col] = pd.to_numeric(aligned[col], errors="coerce").fillna(default_val)
        return aligned

    def _extract_prediction(self, pred: Any) -> np.ndarray:
        if isinstance(pred, pd.DataFrame):
            pred_col = pred.columns[0] if len(pred.columns) > 0 else None
            if pred_col is not None:
                return pred[pred_col].to_numpy(dtype=float)
            return pred.iloc[:, 0].to_numpy(dtype=float)
        if isinstance(pred, pd.Series):
            return pred.to_numpy(dtype=float)
        if isinstance(pred, np.ndarray):
            return pred.astype(float).ravel()
        if isinstance(pred, (list, tuple)):
            return np.asarray(pred, dtype=float).ravel()
        return np.full(1, float(pred))

    def _numeric_column(
        self,
        df: pd.DataFrame,
        column: str | None,
        default: float = 0.0,
    ) -> np.ndarray:
        if not column or column not in df.columns:
            return np.full(len(df), default, dtype=float)
        return pd.to_numeric(df[column], errors="coerce").fillna(default).to_numpy(dtype=float)

    def _expiry_pressure(self, df: pd.DataFrame) -> np.ndarray:
        policy = self.inventory_policy
        weights = policy.get("age_bucket_waste_weights")
        if not isinstance(weights, dict):
            return np.zeros(len(df), dtype=float)
        pressure = np.zeros(len(df), dtype=float)
        for column, weight in weights.items():
            if column in df.columns:
                pressure += self._numeric_column(df, str(column), 0.0) * float(weight)
        return pressure

    def predict(
        self, context: Any, model_input: pd.DataFrame | list[dict[str, Any]] | dict[str, Any]
    ) -> pd.DataFrame:
        if isinstance(model_input, list):
            model_input = pd.DataFrame(model_input)
        elif isinstance(model_input, dict):
            model_input = pd.DataFrame([model_input])

        self._load_child_models()

        demand_input = self._align_input(model_input, self.demand_schema)
        supply_input = self._align_input(model_input, self.supply_schema)
        waste_input = self._align_input(model_input, self.waste_schema)

        demand_pred = self._extract_prediction(self._demand_model.predict(demand_input))
        supply_pred = self._extract_prediction(self._supply_model.predict(supply_input))
        waste_pred = self._extract_prediction(self._waste_model.predict(waste_input))

        if self.stock_column not in model_input.columns:
            current_stock = np.zeros(len(model_input), dtype=float)
        else:
            current_stock = self._numeric_column(model_input, self.stock_column, 0.0)

        policy = self.inventory_policy
        reserved_units = self._numeric_column(
            model_input,
            str(policy.get("reserved_units_column") or ""),
            0.0,
        )
        substitution_units = self._numeric_column(
            model_input,
            str(policy.get("substitution_credit_column") or ""),
            0.0,
        )
        safety_threshold = self._numeric_column(
            model_input,
            str(policy.get("safety_threshold_column") or ""),
            self.safety_threshold,
        )
        waste_pred = np.clip(waste_pred + self._expiry_pressure(model_input), 0.0, None)
        current_stock = np.clip(current_stock - reserved_units + substitution_units, 0.0, None)

        result = run_monte_carlo_paths(
            current_stock=current_stock,
            demand_forecast=demand_pred,
            supply_forecast=supply_pred,
            waste_forecast=waste_pred,
            n_paths=self.n_paths,
            horizons_days=self.horizons_days,
            demand_cv=self.demand_cv,
            supply_cv=self.supply_cv,
            waste_cv=self.waste_cv,
            safety_threshold=safety_threshold,
            seed=self.seed,
        )
        return result
