from __future__ import annotations

from donor_forecast_tasks import build_donor_horizon_probability_task
from model_training_utils import train_and_log


def build_task():
    return build_donor_horizon_probability_task("donor_long_horizon")


if __name__ == "__main__":
    print(train_and_log(build_task))
