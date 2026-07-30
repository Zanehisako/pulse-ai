"""
MLflow pyfunc wrapper for discrete-time hazard models.

Takes one donor feature row at inference time, creates interval rows,
predicts per-interval hazards, and returns multi-output DataFrame
with horizon probabilities and timing estimates.
"""
from __future__ import annotations

import mlflow
import numpy as np
import pandas as pd

from pios_ml_backend.survival.discrete_time_hazard import (
    HazardIntervalDataset,
    extract_horizon_probabilities,
)


class HazardModelPyfunc(mlflow.pyfunc.PythonModel):
    def __init__(
        self,
        estimator,
        dataset: HazardIntervalDataset,
        feature_columns: list[str],
    ) -> None:
        self.estimator = estimator
        self.dataset = dataset
        self.feature_columns = list(feature_columns)

    def predict(self, context, model_input):
        if isinstance(model_input, pd.DataFrame):
            frame = model_input.copy()
        elif isinstance(model_input, (list, dict)):
            frame = pd.DataFrame([model_input] if isinstance(model_input, dict) else model_input)
        else:
            frame = pd.DataFrame(model_input)

        if frame.empty:
            hazards = np.zeros(len(self.dataset.intervals), dtype=np.float64)
            return pd.DataFrame([extract_horizon_probabilities(hazards, self.dataset)])

        results: list[dict[str, float]] = []
        for _, donor_row in frame.iterrows():
            interval_rows = []
            for interval_idx, (start, end) in enumerate(self.dataset.intervals):
                row = {
                    "interval_start": start,
                    "interval_end": end,
                    "interval_index": interval_idx,
                    "interval_midpoint": (start + end) / 2.0,
                }
                for col in self.dataset.static_feature_columns:
                    if col in donor_row.index:
                        row[col] = donor_row[col]
                    else:
                        row[col] = 0.0
                for col in self.dataset.dynamic_feature_columns:
                    if col in donor_row.index:
                        row[col] = donor_row[col]
                    else:
                        row[col] = 0.0
                interval_rows.append(row)

            interval_df = pd.DataFrame(interval_rows)
            align_cols = [c for c in self.feature_columns if c in interval_df.columns]
            X = interval_df[align_cols].copy()
            for c in X.columns:
                X[c] = pd.to_numeric(X[c], errors="coerce").fillna(0.0)

            try:
                proba = self.estimator.predict_proba(X)
                if proba.ndim == 2 and proba.shape[1] >= 2:
                    hazards = proba[:, 1]
                else:
                    hazards = proba[:, 0] if proba.ndim == 1 else proba.ravel()
            except Exception:
                hazards = np.zeros(len(self.dataset.intervals), dtype=np.float64)

            probs = extract_horizon_probabilities(hazards, self.dataset)
            results.append(probs)

        return pd.DataFrame(results)
