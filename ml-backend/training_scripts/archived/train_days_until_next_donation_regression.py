from __future__ import annotations

from donor_forecast_tasks import build_days_until_next_donation_regression_task
from model_training_utils import train_and_log


if __name__ == "__main__":
    print(train_and_log(build_days_until_next_donation_regression_task))
