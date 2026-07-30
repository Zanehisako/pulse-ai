from __future__ import annotations

from model_training_utils import train_and_log
from stockout_forecast_tasks import build_days_until_stockout_regression_task


if __name__ == "__main__":
    payload = train_and_log(build_days_until_stockout_regression_task)
    print(payload)
