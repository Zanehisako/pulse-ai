from __future__ import annotations

import json
import sys
from pathlib import Path


SIMULATOR_DIR = Path(__file__).resolve().parents[1] / "simulator"
if str(SIMULATOR_DIR) not in sys.path:
    sys.path.insert(0, str(SIMULATOR_DIR))

from plot_dreamerv3_metrics import (
    load_metric_series,
    mean_downsample_points,
    select_metrics_to_plot,
)


def _write_jsonl(path: Path, records: list[dict]) -> None:
    path.write_text(
        "".join(json.dumps(record) + "\n" for record in records),
        encoding="utf-8",
    )


def test_load_metric_series_deduplicates_episode_score_across_metrics_and_scores(tmp_path):
    metrics_path = tmp_path / "metrics.jsonl"
    scores_path = tmp_path / "scores.jsonl"
    _write_jsonl(
        metrics_path,
        [
            {"step": 10, "episode/score": -42.0, "epstats/reward_rate": 0.3},
        ],
    )
    _write_jsonl(
        scores_path,
        [
            {"step": 10, "episode/score": -42.0},
        ],
    )

    series = load_metric_series(metrics_path, scores_path=scores_path)

    assert series["episode/score"] == [(10.0, -42.0)]
    assert series["epstats/reward_rate"] == [(10.0, 0.3)]


def test_select_metrics_to_plot_excludes_episode_length():
    series = {
        "episode/score": [(10.0, -42.0)],
        "episode/length": [(10.0, 121.0)],
        "epstats/reward_rate": [(10.0, 0.3)],
        "fps/policy": [(10.0, 27.0)],
        "fps/train": [(10.0, 850.0)],
        "train/loss/value": [(10.0, 5.0)],
    }

    selected = select_metrics_to_plot(series)

    assert "episode/length" not in selected
    assert "episode/score" in selected
    assert "epstats/reward_rate" in selected


def test_mean_downsample_points_averages_dense_series_into_mean_windows():
    points = [
        (0.0, 0.0),
        (1.0, 2.0),
        (2.0, 4.0),
        (3.0, 6.0),
        (4.0, 8.0),
    ]

    averaged = mean_downsample_points(points, max_points=2)

    assert averaged == [(1.0, 2.0), (3.5, 7.0)]
