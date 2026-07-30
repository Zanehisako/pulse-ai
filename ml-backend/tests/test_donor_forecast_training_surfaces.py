from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd


TRAINING_DIR = Path(__file__).resolve().parents[1] / "training_scripts"
if str(TRAINING_DIR) not in sys.path:
    sys.path.insert(0, str(TRAINING_DIR))

from forecasting_search import ForecastSearchPythonModel  # noqa: E402


def test_forecast_search_pyfunc_exposes_configured_output_name():
    model = ForecastSearchPythonModel(
        bundle={"type": "zero_baseline"},
        timestamp_column="event_timestamp",
        output_name="days_until_stockout",
    )
    predictions = model.predict(
        None,
        pd.DataFrame({"event_timestamp": pd.date_range("2026-01-01", periods=3)}),
    )

    assert list(predictions.columns) == ["days_until_stockout"]
    assert predictions["days_until_stockout"].tolist() == [0.0, 0.0, 0.0]
