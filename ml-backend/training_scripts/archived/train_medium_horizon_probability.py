from __future__ import annotations

from model_training_utils import train_and_log
from stockout_forecast_tasks import build_horizon_probability_task


def build_medium_horizon_probability_task():
    return build_horizon_probability_task("medium_horizon")


if __name__ == "__main__":
    payload = train_and_log(build_medium_horizon_probability_task)
    print(payload)
