from __future__ import annotations

from typing import Any, Callable

from model_architecture_config import load_prophet_model_config
from model_training_utils import TaskDataset, train_and_log
from stockout_forecast_tasks import build_horizon_probability_task


def build_long_term_shortage_task() -> TaskDataset:
    config = load_prophet_model_config()
    canonical_horizon = config["canonical_horizon"]
    return build_horizon_probability_task(str(canonical_horizon["horizon_key"]))


def train_and_log_prophet(
    dataset_builder: Callable[[], TaskDataset] = build_long_term_shortage_task,
) -> dict[str, Any]:
    return train_and_log(dataset_builder)


if __name__ == "__main__":
    payload = train_and_log_prophet()
    print(payload)
