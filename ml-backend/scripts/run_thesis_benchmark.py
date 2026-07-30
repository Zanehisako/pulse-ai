from __future__ import annotations

import sys
from pathlib import Path

ROOT_DIRECTORY = Path(__file__).resolve().parents[1]
TRAINING_DIRECTORY = ROOT_DIRECTORY / "training_scripts"
if str(TRAINING_DIRECTORY) not in sys.path:
    sys.path.insert(0, str(TRAINING_DIRECTORY))

import pandas as pd

from blood_shortage_predictor import build_hospital_shortage_task
from ideal_donor import build_donor_propensity_task
from ideal_donor_classifier import build_ideal_donor_task
from model_training_utils import ARTIFACTS_DIRECTORY, train_and_log
from train_long_term_shortage_prophet import build_long_term_shortage_task
from train_stockout_lstm import build_stockout_task


def run_individual_model_benchmark() -> pd.DataFrame:
    rows = []
    for builder in (
        build_donor_propensity_task,
        build_ideal_donor_task,
        build_hospital_shortage_task,
        build_long_term_shortage_task,
        build_stockout_task,
    ):
        rows.append(train_and_log(builder))
    summary = pd.DataFrame(rows)
    ARTIFACTS_DIRECTORY.mkdir(parents=True, exist_ok=True)
    summary.to_csv(ARTIFACTS_DIRECTORY / "model_summary.csv", index=False)
    return summary


if __name__ == "__main__":
    summary = run_individual_model_benchmark()
    print(summary.to_string(index=False))
