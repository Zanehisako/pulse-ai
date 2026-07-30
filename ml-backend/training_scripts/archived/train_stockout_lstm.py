from __future__ import annotations

from stockout_forecast_tasks import (
    add_stockout_sequence_features,
    build_days_until_stockout_regression_task,
)
from model_training_utils import train_and_log


_add_stockout_sequence_features = add_stockout_sequence_features
build_stockout_task = build_days_until_stockout_regression_task


if __name__ == "__main__":
    payload = train_and_log(build_stockout_task)
    print(payload)
