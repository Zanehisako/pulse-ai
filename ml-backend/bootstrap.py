import os
import sys
from pathlib import Path

_TRAINING_DIR = Path(__file__).resolve().parent
_BACKEND_DIR = _TRAINING_DIR.parent / "backendMulti"

for _p in [str(_TRAINING_DIR.parent), str(_BACKEND_DIR)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "backendMulti.settings")

from dotenv import load_dotenv

load_dotenv(_BACKEND_DIR / ".env")

import django

django.setup()

import numpy as np


def median_ci(series, confidence=0.95, n_bootstrap=1000):
    bootstraps = [
        series.dropna().sample(frac=1, replace=True).median()
        for _ in range(n_bootstrap)
    ]
    lower = (1 - confidence) / 2
    upper = 1 - lower
    return np.quantile(bootstraps, [lower, upper])
