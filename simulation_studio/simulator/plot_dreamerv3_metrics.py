from __future__ import annotations

import argparse
import json
import math
import os
import tempfile
from collections import defaultdict
from pathlib import Path


CACHE_ROOT = Path(tempfile.gettempdir()) / "pios_dreamer_cache"
CACHE_ROOT.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("XDG_CACHE_HOME", str(CACHE_ROOT))
os.environ.setdefault("MPLCONFIGDIR", str(CACHE_ROOT / "matplotlib"))


def _read_jsonl(path: Path) -> list[dict]:
    records: list[dict] = []
    if not path.exists():
        return records
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(record, dict):
                records.append(record)
    return records


def load_metric_series(
    metrics_path: Path,
    scores_path: Path | None = None,
) -> dict[str, list[tuple[float, float]]]:
    metrics_path = Path(metrics_path)
    if scores_path is None:
        inferred_scores = metrics_path.with_name("scores.jsonl")
        scores_path = inferred_scores if inferred_scores.exists() else None

    series: dict[str, list[tuple[float, float]]] = defaultdict(list)
    seen_points: dict[str, set[tuple[float, float]]] = defaultdict(set)
    paths = [metrics_path]
    if scores_path is not None:
        paths.append(Path(scores_path))

    for path in paths:
        for record in _read_jsonl(path):
            step = record.get("step")
            if isinstance(step, bool) or not isinstance(step, (int, float)):
                continue
            for key, value in record.items():
                if key == "step":
                    continue
                if isinstance(value, bool) or not isinstance(value, (int, float)):
                    continue
                if key.startswith("usage/"):
                    continue
                point = (float(step), float(value))
                if point in seen_points[key]:
                    continue
                seen_points[key].add(point)
                series[key].append(point)

    return {
        key: sorted(points, key=lambda item: item[0])
        for key, points in series.items()
        if points
    }


def select_metrics_to_plot(series: dict[str, list[tuple[float, float]]]) -> list[str]:
    """Pick the four publication panels: learning progress (episode score and
    reward rate) and the two world-model losses that evidence convergence (value
    and continuation). Throughput (fps/*) and the many internal sub-losses are
    intentionally omitted so the figure matches the paper's caption."""
    excluded = {"episode/length"}
    preferred = [
        "episode/score",
        "epstats/reward_rate",
        "train/loss/value",
        "train/loss/con",  # continuation/termination loss
    ]
    ordered: list[str] = [key for key in preferred if key in series and key not in excluded]
    # Fall back to any continuation-loss key if a run names it differently.
    if "train/loss/con" not in ordered:
        for key in sorted(series):
            tail = key.rsplit("/", 1)[-1]
            if key.startswith("train/loss/") and tail.startswith("con") and key not in ordered:
                ordered.append(key)
                break
    return ordered[:4]


_PANEL_TITLES = {
    "episode/score": "Episode Score",
    "epstats/reward_rate": "Reward Rate",
    "train/loss/value": "Value Loss",
    "train/loss/con": "Continuation Loss",
}


def summarize_series(
    series: dict[str, list[tuple[float, float]]],
    metric_name: str = "episode/score",
) -> dict[str, float]:
    points = list(series.get(metric_name, ()))
    if not points:
        return {}
    values = [float(value) for _, value in points]
    tail = values[-min(100, len(values)) :]
    return {
        "count": float(len(values)),
        "mean": float(sum(values) / len(values)),
        "min": float(min(values)),
        "max": float(max(values)),
        "last": float(values[-1]),
        "tail_mean": float(sum(tail) / len(tail)),
    }


def summarize_logdir_scores(logdir: Path) -> dict[str, float]:
    logdir = Path(logdir)
    return summarize_series(
        load_metric_series(
            logdir / "metrics.jsonl",
            scores_path=(logdir / "scores.jsonl"),
        )
    )


def mean_downsample_points(
    points: list[tuple[float, float]],
    *,
    max_points: int = 250,
) -> list[tuple[float, float]]:
    if max_points < 1:
        raise ValueError("max_points must be at least 1")
    if len(points) <= max_points:
        return list(points)

    window_size = math.ceil(len(points) / max_points)
    averaged: list[tuple[float, float]] = []
    for index in range(0, len(points), window_size):
        window = points[index:index + window_size]
        mean_step = sum(step for step, _ in window) / len(window)
        mean_value = sum(value for _, value in window) / len(window)
        averaged.append((mean_step, mean_value))
    return averaged


def plot_metrics(
    metrics_path: Path,
    *,
    output_path: Path | None = None,
    scores_path: Path | None = None,
    title: str = "DreamerV3 Metrics",
) -> Path | None:
    metrics_path = Path(metrics_path)
    if output_path is None:
        output_path = metrics_path.with_name("metrics_plot.png")
    else:
        output_path = Path(output_path)

    series = load_metric_series(metrics_path, scores_path=scores_path)
    metric_names = select_metrics_to_plot(series)
    if not metric_names:
        return None

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cols = 2 if len(metric_names) > 1 else 1
    rows = math.ceil(len(metric_names) / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(14, 4.2 * rows), squeeze=False)
    axes_flat = axes.flatten()

    for axis, name in zip(axes_flat, metric_names):
        points = series[name]
        plotted_points = mean_downsample_points(points)
        steps = [step for step, _ in plotted_points]
        values = [value for _, value in plotted_points]
        axis.plot(steps, values, color="#1f77b4", linewidth=2.0)
        if len(points) <= 200:
            axis.scatter(steps, values, color="#1f77b4", s=14, alpha=0.75)
        axis.set_title(_PANEL_TITLES.get(name, name))
        axis.set_xlabel("Step")
        axis.set_ylabel("Value")
        axis.grid(True, alpha=0.25)

    for axis in axes_flat[len(metric_names):]:
        axis.set_visible(False)

    fig.suptitle(title, fontsize=16)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return output_path


def plot_metrics_for_logdir(logdir: Path, *, output_path: Path | None = None) -> Path | None:
    logdir = Path(logdir)
    metrics_path = logdir / "metrics.jsonl"
    scores_path = logdir / "scores.jsonl"
    if output_path is None:
        output_path = logdir / "metrics_plot.png"
    return plot_metrics(
        metrics_path,
        output_path=output_path,
        scores_path=scores_path if scores_path.exists() else None,
        title=f"DreamerV3 Metrics: {logdir.name}",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot DreamerV3 metrics from metrics.jsonl.")
    parser.add_argument(
        "metrics",
        type=Path,
        help="Path to metrics.jsonl.",
    )
    parser.add_argument(
        "--scores",
        type=Path,
        default=None,
        help="Optional path to scores.jsonl. Defaults to the sibling file if present.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional output PNG path. Defaults to metrics_plot.png next to metrics.jsonl.",
    )
    parser.add_argument(
        "--title",
        type=str,
        default="DreamerV3 Metrics",
        help="Plot title.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_path = plot_metrics(
        args.metrics,
        output_path=args.output,
        scores_path=args.scores,
        title=args.title,
    )
    if output_path is None:
        raise SystemExit(f"No plottable metrics found in {args.metrics}")
    print(f"Saved metrics plot to {output_path}")


if __name__ == "__main__":
    main()
