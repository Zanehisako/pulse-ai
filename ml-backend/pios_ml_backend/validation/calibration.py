from __future__ import annotations

import numpy as np


def expected_calibration_error(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    *,
    n_bins: int = 10,
    strategy: str = "uniform",
) -> float:
    if not len(y_true) or np.isnan(y_prob).all():
        return float("nan")
    if n_bins < 2:
        return float("nan")
    prob = np.asarray(y_prob, dtype=float)
    label = np.asarray(y_true, dtype=float)
    valid = ~np.isnan(prob)
    prob = prob[valid]
    label = label[valid]
    if len(prob) < n_bins:
        return float("nan")

    if strategy == "quantile":
        bin_edges = np.quantile(prob, np.linspace(0, 1, n_bins + 1))
    else:
        bin_edges = np.linspace(0, 1, n_bins + 1)
    bin_edges[0] = -np.inf
    bin_edges[-1] = np.inf

    ece = 0.0
    for i in range(n_bins):
        in_bin = (prob > bin_edges[i]) & (prob <= bin_edges[i + 1])
        n_bin = in_bin.sum()
        if n_bin == 0:
            continue
        bin_accuracy = label[in_bin].mean()
        bin_confidence = prob[in_bin].mean()
        ece += (n_bin / len(prob)) * abs(bin_accuracy - bin_confidence)
    return float(ece)


def reliability_curve(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    *,
    n_bins: int = 10,
    strategy: str = "uniform",
) -> tuple[np.ndarray, np.ndarray]:
    prob = np.asarray(y_prob, dtype=float)
    label = np.asarray(y_true, dtype=float)
    valid = ~np.isnan(prob)
    prob = prob[valid]
    label = label[valid]

    if strategy == "quantile":
        bin_edges = np.quantile(prob, np.linspace(0, 1, n_bins + 1))
    else:
        bin_edges = np.linspace(0, 1, n_bins + 1)
    bin_edges[0] = -np.inf
    bin_edges[-1] = np.inf

    mean_predicted = []
    fraction_positive = []
    for i in range(n_bins):
        in_bin = (prob > bin_edges[i]) & (prob <= bin_edges[i + 1])
        n_bin = in_bin.sum()
        if n_bin == 0:
            continue
        mean_predicted.append(float(prob[in_bin].mean()))
        fraction_positive.append(float(label[in_bin].mean()))
    return np.array(mean_predicted), np.array(fraction_positive)


def brier_skill_score(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    baseline_prob: float | None = None,
) -> float:
    prob = np.asarray(y_prob, dtype=float)
    label = np.asarray(y_true, dtype=float)
    valid = ~np.isnan(prob)
    prob = prob[valid]
    label = label[valid]
    if len(prob) == 0:
        return float("nan")
    brier_model = float(((prob - label) ** 2).mean())
    if baseline_prob is None:
        baseline_prob = float(label.mean())
    brier_baseline = float(((baseline_prob - label) ** 2).mean())
    if brier_baseline == 0:
        return 0.0 if brier_model == 0 else float("-inf")
    return 1.0 - brier_model / brier_baseline
