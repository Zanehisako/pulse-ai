from __future__ import annotations

import json
import math
import unicodedata
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


ROOT_DIRECTORY = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT_DIRECTORY / "config" / "quebec_public_validation_sources.json"


@dataclass
class OutputPaths:
    base: Path
    tables: Path
    plots: Path


@dataclass
class SourceStatus:
    source_id: str
    enabled: bool
    source_type: str
    url: str
    status: str
    rows: int | None = None
    detail: str | None = None


@dataclass
class ComparisonResult:
    comparison_id: str
    source_id: str
    comparison_type: str
    status: str
    metric: float | None
    threshold: dict[str, Any]
    details: dict[str, Any]
    table_path: str | None = None
    plot_path: str | None = None


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Config root must be an object: {path}")
    return payload


def _require_keys(row: dict[str, Any], keys: list[str], *, label: str) -> None:
    for key in keys:
        if key not in row:
            raise ValueError(f"{label} missing required field {key}")


def validate_public_validation_config(config: dict[str, Any]) -> dict[str, Any]:
    _require_keys(
        config,
        ["id", "enabled", "output", "synthetic_datasets", "sources", "comparisons"],
        label="external validation config",
    )
    _require_keys(
        config["output"],
        ["directory", "json_report", "markdown_report", "tables_directory", "plots_directory"],
        label="external validation output",
    )
    if not isinstance(config["sources"], dict):
        raise ValueError("external validation config sources must be an object")
    if not isinstance(config["comparisons"], dict):
        raise ValueError("external validation config comparisons must be an object")

    for source_id, source in config["sources"].items():
        if not isinstance(source, dict):
            raise ValueError(f"source {source_id} must be an object")
        _require_keys(source, ["enabled", "source_type", "url"], label=f"source {source_id}")
        if source.get("enabled") and not str(source.get("url") or "").strip():
            raise ValueError(f"source {source_id} url must be present when enabled")

    for dataset_id, dataset in config["synthetic_datasets"].items():
        if not isinstance(dataset, dict):
            raise ValueError(f"synthetic dataset {dataset_id} must be an object")
        _require_keys(dataset, ["enabled", "path", "format"], label=f"synthetic dataset {dataset_id}")

    for comparison_id, comparison in config["comparisons"].items():
        if not isinstance(comparison, dict):
            raise ValueError(f"comparison {comparison_id} must be an object")
        _require_keys(
            comparison,
            ["enabled", "comparison_type", "source_id", "dataset", "threshold"],
            label=f"comparison {comparison_id}",
        )
        if comparison["source_id"] not in config["sources"]:
            raise ValueError(f"comparison {comparison_id} references unknown source_id")
        if comparison["dataset"] not in config["synthetic_datasets"]:
            raise ValueError(f"comparison {comparison_id} references unknown dataset")
        _validate_comparison_shape(comparison_id, comparison)

    return config


def _validate_comparison_shape(comparison_id: str, comparison: dict[str, Any]) -> None:
    comparison_type = str(comparison.get("comparison_type"))
    required_by_type = {
        "scalar_column_vs_source_metric": [
            "synthetic_column",
            "aggregator",
            "source_metric",
        ],
        "scalar_column_expected_range": [
            "synthetic_column",
            "aggregator",
            "expected_min",
            "expected_max",
        ],
        "distribution_vs_expected": ["synthetic_column", "expected_distribution"],
        "distribution_vs_expected_weights": ["synthetic_column", "source_weights"],
        "age_band_profile": ["synthetic_column", "bins", "labels"],
        "category_profile": ["synthetic_column"],
        "coverage_values": ["synthetic_column"],
        "external_proxy_profile": ["synthetic_column", "external_metric", "quantiles"],
    }
    required = required_by_type.get(comparison_type)
    if required is None:
        raise ValueError(f"comparison {comparison_id} has unsupported comparison_type")
    _require_keys(comparison, required, label=f"comparison {comparison_id}")


def load_public_validation_config(config_path: Path = CONFIG_PATH) -> dict[str, Any]:
    return validate_public_validation_config(_read_json(config_path))


def _resolve_path(path: str, *, base_dir: Path = ROOT_DIRECTORY) -> Path:
    candidate = Path(path)
    if candidate.is_absolute():
        return candidate
    return base_dir / candidate


def _load_dataset(dataset: dict[str, Any]) -> pd.DataFrame:
    path = _resolve_path(str(dataset["path"]))
    file_format = str(dataset.get("format", "")).lower()
    if file_format == "parquet":
        return pd.read_parquet(path)
    if file_format == "csv":
        return pd.read_csv(path)
    raise ValueError(f"Unsupported synthetic dataset format: {file_format}")


def _clean_columns(frame: pd.DataFrame) -> pd.DataFrame:
    renamed = {column: str(column).strip().replace("\t", "") for column in frame.columns}
    return frame.rename(columns=renamed)


def _load_public_source_frame(source: dict[str, Any]) -> pd.DataFrame | None:
    fetch = source.get("fetch", {})
    if source.get("source_type") != "csv" or fetch.get("enabled") is not True:
        return None
    data_url = str(source.get("data_url") or source.get("url") or "").strip()
    if not data_url:
        raise ValueError("csv source requires data_url or url")
    encoding = str(source.get("encoding") or "utf-8")
    try:
        return _clean_columns(pd.read_csv(data_url, encoding=encoding))
    except Exception as exc:
        if fetch.get("fail_on_error") is True:
            raise RuntimeError(f"Could not download public source {data_url}: {exc}") from exc
        return None


def load_enabled_sources(config: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {
        source_id: source
        for source_id, source in config["sources"].items()
        if source.get("enabled") is True
    }


def _source_status(source_id: str, source: dict[str, Any]) -> tuple[SourceStatus, pd.DataFrame | None]:
    if source.get("enabled") is not True:
        return (
            SourceStatus(
                source_id=source_id,
                enabled=False,
                source_type=str(source.get("source_type") or ""),
                url=str(source.get("url") or ""),
                status="disabled",
            ),
            None,
        )
    try:
        frame = _load_public_source_frame(source)
    except Exception as exc:
        return (
            SourceStatus(
                source_id=source_id,
                enabled=True,
                source_type=str(source.get("source_type") or ""),
                url=str(source.get("url") or ""),
                status="error",
                detail=str(exc),
            ),
            None,
        )
    if frame is None:
        return (
            SourceStatus(
                source_id=source_id,
                enabled=True,
                source_type=str(source.get("source_type") or ""),
                url=str(source.get("url") or ""),
                status="reference_only",
            ),
            None,
        )
    return (
        SourceStatus(
            source_id=source_id,
            enabled=True,
            source_type=str(source.get("source_type") or ""),
            url=str(source.get("url") or ""),
            status="loaded",
            rows=len(frame),
        ),
        frame,
    )


def _dedupe(frame: pd.DataFrame, comparison: dict[str, Any]) -> pd.DataFrame:
    key = comparison.get("dedupe_key")
    if key and key in frame.columns:
        return frame.sort_values(list({str(key), "event_timestamp"}.intersection(frame.columns))).drop_duplicates(str(key))
    return frame


def _aggregate(series: pd.Series, aggregator: str) -> float:
    numeric = pd.to_numeric(series, errors="coerce").dropna()
    if numeric.empty:
        raise ValueError("No numeric values available for aggregation")
    if aggregator == "mean":
        return float(numeric.mean())
    if aggregator == "median":
        return float(numeric.median())
    if aggregator == "min":
        return float(numeric.min())
    if aggregator == "max":
        return float(numeric.max())
    if aggregator == "sum":
        return float(numeric.sum())
    if aggregator == "nunique":
        return float(series.nunique(dropna=True))
    raise ValueError(f"Unsupported aggregator: {aggregator}")


def _relative_error(actual: float, expected: float) -> float:
    return abs(actual - expected) / max(abs(expected), 1e-12)


def _normalize_label(value: Any, normalization: str | None = None) -> str:
    text = str(value).strip()
    if normalization == "ascii_lower":
        text = (
            unicodedata.normalize("NFKD", text)
            .encode("ascii", "ignore")
            .decode("ascii")
            .lower()
        )
    return text


def _distribution(frame: pd.DataFrame, column: str, *, normalization: str | None = None) -> dict[str, float]:
    values = frame[column].dropna().map(lambda value: _normalize_label(value, normalization))
    counts = values.value_counts(normalize=True)
    return {str(key): float(value) for key, value in counts.to_dict().items()}


def _normalize_distribution(values: dict[str, Any], *, normalization: str | None = None) -> dict[str, float]:
    numeric = {
        _normalize_label(key, normalization): float(value)
        for key, value in values.items()
    }
    total = sum(numeric.values())
    if total <= 0:
        raise ValueError("Expected distribution must have positive total weight")
    return {key: value / total for key, value in numeric.items()}


def _total_variation_distance(observed: dict[str, float], expected: dict[str, float]) -> float:
    keys = set(observed).union(expected)
    return 0.5 * sum(abs(observed.get(key, 0.0) - expected.get(key, 0.0)) for key in keys)


def _status_from_threshold(value: float, threshold: dict[str, Any], key: str, *, direction: str = "max") -> str:
    if key not in threshold:
        return "INFO"
    limit = float(threshold[key])
    if direction == "max":
        return "PASS" if value <= limit else "WARN"
    return "PASS" if value >= limit else "WARN"


def _source_metric(source: dict[str, Any], metric: str) -> float:
    metrics = source.get("metrics", {})
    if metric not in metrics:
        raise ValueError(f"source metric not found: {metric}")
    return float(metrics[metric])


def _comparison_table_path(output_paths: OutputPaths, comparison_id: str) -> Path:
    output_paths.tables.mkdir(parents=True, exist_ok=True)
    return output_paths.tables / f"{comparison_id}.csv"


def _plot_path(output_paths: OutputPaths, comparison_id: str) -> Path:
    output_paths.plots.mkdir(parents=True, exist_ok=True)
    return output_paths.plots / f"{comparison_id}.png"


def _write_distribution_plot(
    rows: list[dict[str, Any]],
    *,
    output_paths: OutputPaths,
    comparison_id: str,
    title: str,
) -> str:
    path = _plot_path(output_paths, comparison_id)
    labels = [str(row["category"]) for row in rows]
    observed = [float(row.get("observed_share", 0.0)) for row in rows]
    expected = [float(row.get("expected_share", 0.0)) for row in rows]
    x_values = list(range(len(labels)))
    width = 0.38
    fig, ax = plt.subplots(figsize=(max(7.0, len(labels) * 0.7), 4.8))
    ax.bar([x - width / 2 for x in x_values], observed, width=width, label="Synthetic")
    if any(value > 0 for value in expected):
        ax.bar([x + width / 2 for x in x_values], expected, width=width, label="Public reference")
    ax.set_title(title)
    ax.set_ylabel("Share")
    ax.set_xticks(x_values)
    ax.set_xticklabels(labels, rotation=35, ha="right")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return str(path)


def _write_scalar_plot(
    actual: float,
    expected: float | None,
    *,
    output_paths: OutputPaths,
    comparison_id: str,
    title: str,
) -> str:
    path = _plot_path(output_paths, comparison_id)
    labels = ["Synthetic"]
    values = [actual]
    if expected is not None:
        labels.append("Public reference")
        values.append(expected)
    fig, ax = plt.subplots(figsize=(6.2, 4.4))
    ax.bar(labels, values)
    ax.set_title(title)
    ax.set_ylabel("Value")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return str(path)


def _write_proxy_plot(
    synthetic_quantiles: dict[str, float],
    external_quantiles: dict[str, float],
    *,
    output_paths: OutputPaths,
    comparison_id: str,
) -> str:
    path = _plot_path(output_paths, comparison_id)
    labels = list(synthetic_quantiles)
    fig, ax = plt.subplots(figsize=(7.0, 4.6))
    ax.plot(labels, [synthetic_quantiles[key] for key in labels], marker="o", label="Synthetic proxy")
    if external_quantiles:
        ax.plot(labels, [external_quantiles.get(key, math.nan) for key in labels], marker="o", label="Public ER proxy")
    ax.set_title(comparison_id)
    ax.set_ylabel("Quantile value")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return str(path)


def _compare_scalar_vs_source(
    comparison_id: str,
    comparison: dict[str, Any],
    frame: pd.DataFrame,
    source: dict[str, Any],
    output_paths: OutputPaths,
) -> ComparisonResult:
    column = str(comparison["synthetic_column"])
    actual = _aggregate(frame[column], str(comparison["aggregator"]))
    expected = _source_metric(source, str(comparison["source_metric"]))
    relative_error = _relative_error(actual, expected)
    threshold = comparison.get("threshold", {})
    status = _status_from_threshold(relative_error, threshold, "relative_error_max", direction="max")
    table_path = _comparison_table_path(output_paths, comparison_id)
    pd.DataFrame(
        [
            {
                "metric": comparison_id,
                "synthetic_value": actual,
                "public_reference_value": expected,
                "relative_error": relative_error,
                "status": status,
            }
        ]
    ).to_csv(table_path, index=False)
    plot_path = None
    if comparison.get("plot") is True:
        plot_path = _write_scalar_plot(actual, expected, output_paths=output_paths, comparison_id=comparison_id, title=comparison_id)
    return ComparisonResult(
        comparison_id=comparison_id,
        source_id=str(comparison["source_id"]),
        comparison_type=str(comparison["comparison_type"]),
        status=status,
        metric=relative_error,
        threshold=threshold,
        details={
            "synthetic_value": actual,
            "public_reference_value": expected,
            "relative_error": relative_error,
        },
        table_path=str(table_path),
        plot_path=plot_path,
    )


def _compare_scalar_range(
    comparison_id: str,
    comparison: dict[str, Any],
    frame: pd.DataFrame,
    output_paths: OutputPaths,
) -> ComparisonResult:
    actual = _aggregate(frame[str(comparison["synthetic_column"])], str(comparison["aggregator"]))
    expected_min = float(comparison["expected_min"])
    expected_max = float(comparison["expected_max"])
    status = "PASS" if expected_min <= actual <= expected_max else "WARN"
    table_path = _comparison_table_path(output_paths, comparison_id)
    pd.DataFrame(
        [
            {
                "metric": comparison_id,
                "synthetic_value": actual,
                "expected_min": expected_min,
                "expected_max": expected_max,
                "status": status,
            }
        ]
    ).to_csv(table_path, index=False)
    plot_path = None
    if comparison.get("plot") is True:
        plot_path = _write_scalar_plot(
            actual,
            (expected_min + expected_max) / 2.0,
            output_paths=output_paths,
            comparison_id=comparison_id,
            title=comparison_id,
        )
    return ComparisonResult(
        comparison_id=comparison_id,
        source_id=str(comparison["source_id"]),
        comparison_type=str(comparison["comparison_type"]),
        status=status,
        metric=actual,
        threshold=comparison.get("threshold", {}),
        details={"synthetic_value": actual, "expected_min": expected_min, "expected_max": expected_max},
        table_path=str(table_path),
        plot_path=plot_path,
    )


def _compare_distribution(
    comparison_id: str,
    comparison: dict[str, Any],
    frame: pd.DataFrame,
    expected_distribution: dict[str, Any],
    output_paths: OutputPaths,
    *,
    normalization: str | None = None,
) -> ComparisonResult:
    narrowed = _dedupe(frame, comparison)
    observed = _distribution(narrowed, str(comparison["synthetic_column"]), normalization=normalization)
    expected = _normalize_distribution(expected_distribution, normalization=normalization)
    tvd = _total_variation_distance(observed, expected)
    threshold = comparison.get("threshold", {})
    status = _status_from_threshold(tvd, threshold, "max_total_variation_distance", direction="max")
    rows = [
        {
            "category": key,
            "observed_share": observed.get(key, 0.0),
            "expected_share": expected.get(key, 0.0),
            "absolute_gap": abs(observed.get(key, 0.0) - expected.get(key, 0.0)),
        }
        for key in sorted(set(observed).union(expected))
    ]
    table_path = _comparison_table_path(output_paths, comparison_id)
    pd.DataFrame(rows).to_csv(table_path, index=False)
    plot_path = None
    if comparison.get("plot") is True:
        plot_path = _write_distribution_plot(rows, output_paths=output_paths, comparison_id=comparison_id, title=comparison_id)
    return ComparisonResult(
        comparison_id=comparison_id,
        source_id=str(comparison["source_id"]),
        comparison_type=str(comparison["comparison_type"]),
        status=status,
        metric=tvd,
        threshold=threshold,
        details={"total_variation_distance": tvd, "categories": len(rows)},
        table_path=str(table_path),
        plot_path=plot_path,
    )


def _compare_age_bands(
    comparison_id: str,
    comparison: dict[str, Any],
    frame: pd.DataFrame,
    output_paths: OutputPaths,
) -> ComparisonResult:
    narrowed = _dedupe(frame, comparison)
    bins = comparison["bins"]
    labels = comparison["labels"]
    values = pd.to_numeric(narrowed[str(comparison["synthetic_column"])], errors="coerce")
    bands = pd.cut(values, bins=bins, labels=labels)
    shares = bands.value_counts(normalize=True).sort_index()
    rows = [
        {"category": str(category), "observed_share": float(value), "expected_share": 0.0}
        for category, value in shares.items()
    ]
    nonempty = int((shares > 0).sum())
    threshold = comparison.get("threshold", {})
    status = _status_from_threshold(
        float(nonempty),
        threshold,
        "minimum_nonempty_bands",
        direction="min",
    )
    table_path = _comparison_table_path(output_paths, comparison_id)
    pd.DataFrame(rows).to_csv(table_path, index=False)
    plot_path = None
    if comparison.get("plot") is True:
        plot_path = _write_distribution_plot(rows, output_paths=output_paths, comparison_id=comparison_id, title=comparison_id)
    return ComparisonResult(
        comparison_id=comparison_id,
        source_id=str(comparison["source_id"]),
        comparison_type=str(comparison["comparison_type"]),
        status=status,
        metric=float(nonempty),
        threshold=threshold,
        details={"nonempty_bands": nonempty},
        table_path=str(table_path),
        plot_path=plot_path,
    )


def _compare_category_profile(
    comparison_id: str,
    comparison: dict[str, Any],
    frame: pd.DataFrame,
    output_paths: OutputPaths,
) -> ComparisonResult:
    narrowed = _dedupe(frame, comparison)
    observed = _distribution(narrowed, str(comparison["synthetic_column"]))
    rows = [
        {"category": key, "observed_share": value, "expected_share": 0.0}
        for key, value in sorted(observed.items())
    ]
    nonempty = len(observed)
    threshold = comparison.get("threshold", {})
    status = _status_from_threshold(
        float(nonempty),
        threshold,
        "minimum_nonempty_categories",
        direction="min",
    )
    table_path = _comparison_table_path(output_paths, comparison_id)
    pd.DataFrame(rows).to_csv(table_path, index=False)
    plot_path = None
    if comparison.get("plot") is True:
        plot_path = _write_distribution_plot(rows, output_paths=output_paths, comparison_id=comparison_id, title=comparison_id)
    return ComparisonResult(
        comparison_id=comparison_id,
        source_id=str(comparison["source_id"]),
        comparison_type=str(comparison["comparison_type"]),
        status=status,
        metric=float(nonempty),
        threshold=threshold,
        details={"nonempty_categories": nonempty},
        table_path=str(table_path),
        plot_path=plot_path,
    )


def _compare_coverage(
    comparison_id: str,
    comparison: dict[str, Any],
    frame: pd.DataFrame,
    source: dict[str, Any],
    output_paths: OutputPaths,
) -> ComparisonResult:
    narrowed = _dedupe(frame, comparison)
    normalization = comparison.get("normalization")
    if "source_values" in comparison and isinstance(comparison["source_values"], list):
        raw_values = comparison["source_values"]
    else:
        raw_values = source.get(str(comparison.get("source_values")), [])
    expected_values = {_normalize_label(value, normalization) for value in raw_values}
    if not expected_values:
        raise ValueError(f"comparison {comparison_id} has no source values")
    values = narrowed[str(comparison["synthetic_column"])].dropna().map(
        lambda value: _normalize_label(value, normalization)
    )
    unique_values = sorted(set(values))
    covered = sorted(set(unique_values).intersection(expected_values))
    coverage_rate = len(covered) / max(len(unique_values), 1)
    threshold = comparison.get("threshold", {})
    status = _status_from_threshold(coverage_rate, threshold, "min_coverage_rate", direction="min")
    rows = [
        {
            "synthetic_value": value,
            "covered_by_public_reference": value in expected_values,
        }
        for value in unique_values
    ]
    table_path = _comparison_table_path(output_paths, comparison_id)
    pd.DataFrame(rows).to_csv(table_path, index=False)
    plot_path = None
    if comparison.get("plot") is True:
        plot_path = _write_scalar_plot(coverage_rate, float(threshold.get("min_coverage_rate", 1.0)), output_paths=output_paths, comparison_id=comparison_id, title=comparison_id)
    return ComparisonResult(
        comparison_id=comparison_id,
        source_id=str(comparison["source_id"]),
        comparison_type=str(comparison["comparison_type"]),
        status=status,
        metric=coverage_rate,
        threshold=threshold,
        details={
            "coverage_rate": coverage_rate,
            "synthetic_unique_values": unique_values,
            "covered_values": covered,
        },
        table_path=str(table_path),
        plot_path=plot_path,
    )


def _quantiles(series: pd.Series, quantiles: list[float]) -> dict[str, float]:
    numeric = pd.to_numeric(series, errors="coerce").dropna()
    if numeric.empty:
        return {}
    return {
        str(q): float(numeric.quantile(q))
        for q in quantiles
    }


def _external_metric_series(source: dict[str, Any], source_frame: pd.DataFrame, metric_id: str) -> pd.Series:
    metric_config = source.get("derived_metrics", {}).get(metric_id)
    if not isinstance(metric_config, dict):
        raise ValueError(f"external metric not configured: {metric_id}")
    numerator = str(metric_config["numerator"])
    denominator = str(metric_config["denominator"])
    if numerator not in source_frame.columns or denominator not in source_frame.columns:
        raise ValueError(f"external metric columns missing for {metric_id}")
    num = pd.to_numeric(source_frame[numerator], errors="coerce")
    den = pd.to_numeric(source_frame[denominator], errors="coerce")
    return num / den.replace(0, pd.NA)


def _compare_external_proxy(
    comparison_id: str,
    comparison: dict[str, Any],
    frame: pd.DataFrame,
    source: dict[str, Any],
    source_frame: pd.DataFrame | None,
    output_paths: OutputPaths,
) -> ComparisonResult:
    synthetic_rows = len(frame)
    quantiles = [float(value) for value in comparison["quantiles"]]
    synthetic_quantiles = _quantiles(frame[str(comparison["synthetic_column"])], quantiles)
    external_quantiles: dict[str, float] = {}
    external_rows = 0
    source_available = source_frame is not None
    if source_frame is not None:
        metric_series = _external_metric_series(source, source_frame, str(comparison["external_metric"]))
        external_rows = int(metric_series.dropna().shape[0])
        external_quantiles = _quantiles(metric_series, quantiles)

    threshold = comparison.get("threshold", {})
    status = "PASS"
    if synthetic_rows < int(threshold.get("minimum_synthetic_rows", 0)):
        status = "WARN"
    if source_available and external_rows < int(threshold.get("minimum_external_rows", 0)):
        status = "WARN"
    if not source_available:
        status = "INFO"

    rows = []
    for key in sorted(set(synthetic_quantiles).union(external_quantiles)):
        rows.append(
            {
                "quantile": key,
                "synthetic_proxy": synthetic_quantiles.get(key),
                "external_public_proxy": external_quantiles.get(key),
            }
        )
    table_path = _comparison_table_path(output_paths, comparison_id)
    pd.DataFrame(rows).to_csv(table_path, index=False)
    plot_path = None
    if comparison.get("plot") is True:
        plot_path = _write_proxy_plot(
            synthetic_quantiles,
            external_quantiles,
            output_paths=output_paths,
            comparison_id=comparison_id,
        )
    return ComparisonResult(
        comparison_id=comparison_id,
        source_id=str(comparison["source_id"]),
        comparison_type=str(comparison["comparison_type"]),
        status=status,
        metric=None,
        threshold=threshold,
        details={
            "source_available": source_available,
            "synthetic_rows": synthetic_rows,
            "external_rows": external_rows,
            "synthetic_quantiles": synthetic_quantiles,
            "external_quantiles": external_quantiles,
        },
        table_path=str(table_path),
        plot_path=plot_path,
    )


def _run_comparison(
    comparison_id: str,
    comparison: dict[str, Any],
    datasets: dict[str, pd.DataFrame],
    sources: dict[str, dict[str, Any]],
    source_frames: dict[str, pd.DataFrame],
    output_paths: OutputPaths,
) -> ComparisonResult:
    source = sources[str(comparison["source_id"])]
    frame = datasets[str(comparison["dataset"])]
    comparison_type = str(comparison["comparison_type"])
    if comparison_type == "scalar_column_vs_source_metric":
        return _compare_scalar_vs_source(comparison_id, comparison, frame, source, output_paths)
    if comparison_type == "scalar_column_expected_range":
        return _compare_scalar_range(comparison_id, comparison, frame, output_paths)
    if comparison_type == "distribution_vs_expected":
        return _compare_distribution(
            comparison_id,
            comparison,
            frame,
            comparison["expected_distribution"],
            output_paths,
        )
    if comparison_type == "distribution_vs_expected_weights":
        weights_key = str(comparison["source_weights"])
        return _compare_distribution(
            comparison_id,
            comparison,
            frame,
            source[weights_key],
            output_paths,
            normalization="ascii_lower",
        )
    if comparison_type == "age_band_profile":
        return _compare_age_bands(comparison_id, comparison, frame, output_paths)
    if comparison_type == "category_profile":
        return _compare_category_profile(comparison_id, comparison, frame, output_paths)
    if comparison_type == "coverage_values":
        return _compare_coverage(comparison_id, comparison, frame, source, output_paths)
    if comparison_type == "external_proxy_profile":
        return _compare_external_proxy(
            comparison_id,
            comparison,
            frame,
            source,
            source_frames.get(str(comparison["source_id"])),
            output_paths,
        )
    raise ValueError(f"Unsupported comparison_type: {comparison_type}")


def _write_markdown(report: dict[str, Any], output_path: Path) -> None:
    lines = [
        "# Quebec Public External Validation",
        "",
        "This report validates synthetic dataset calibration against public Quebec aggregate/proxy sources. It is not row-level clinical or production validation.",
        "",
        "## Summary",
        "",
        f"- Comparisons: {report['summary']['comparison_count']}",
        f"- Passed: {report['summary']['passed_count']}",
        f"- Warnings: {report['summary']['warning_count']}",
        f"- Informational: {report['summary']['info_count']}",
        "",
        "## Comparisons",
        "",
        "| Comparison | Source | Type | Status | Metric |",
        "|---|---|---|---|---:|",
    ]
    for row in report["comparisons"].values():
        metric = row.get("metric")
        metric_text = "" if metric is None else f"{float(metric):.6g}"
        lines.append(
            f"| {row['comparison_id']} | {row['source_id']} | {row['comparison_type']} | {row['status']} | {metric_text} |"
        )
    lines.extend(["", "## Source Availability", "", "| Source | Type | Status | Rows |", "|---|---|---|---:|"])
    for row in report["sources"].values():
        rows = "" if row.get("rows") is None else str(row["rows"])
        lines.append(f"| {row['source_id']} | {row['source_type']} | {row['status']} | {rows} |")
    lines.extend(["", "## Limitations", ""])
    for limitation in report["limitations"]:
        lines.append(f"- {limitation}")
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_external_validation_report(config_path: Path = CONFIG_PATH) -> dict[str, Any]:
    config = load_public_validation_config(config_path)
    output_config = config["output"]
    output_dir = _resolve_path(str(output_config["directory"]))
    output_dir.mkdir(parents=True, exist_ok=True)
    output_paths = OutputPaths(
        base=output_dir,
        tables=output_dir / str(output_config["tables_directory"]),
        plots=output_dir / str(output_config["plots_directory"]),
    )

    datasets: dict[str, pd.DataFrame] = {}
    for dataset_id, dataset in config["synthetic_datasets"].items():
        if dataset.get("enabled") is True:
            datasets[dataset_id] = _load_dataset(dataset)

    source_statuses: dict[str, SourceStatus] = {}
    source_frames: dict[str, pd.DataFrame] = {}
    for source_id, source in config["sources"].items():
        status, frame = _source_status(source_id, source)
        source_statuses[source_id] = status
        if frame is not None:
            source_frames[source_id] = frame

    comparisons: dict[str, ComparisonResult] = {}
    for comparison_id, comparison in config["comparisons"].items():
        if comparison.get("enabled") is not True:
            continue
        result = _run_comparison(
            comparison_id,
            comparison,
            datasets,
            config["sources"],
            source_frames,
            output_paths,
        )
        comparisons[comparison_id] = result

    status_values = [result.status for result in comparisons.values()]
    report = {
        "config_id": config["id"],
        "summary": {
            "comparison_count": len(comparisons),
            "passed_count": status_values.count("PASS"),
            "warning_count": status_values.count("WARN"),
            "info_count": status_values.count("INFO"),
            "error_count": status_values.count("ERROR"),
        },
        "sources": {
            source_id: asdict(status)
            for source_id, status in source_statuses.items()
        },
        "comparisons": {
            comparison_id: asdict(result)
            for comparison_id, result in comparisons.items()
        },
        "limitations": list(config.get("limitations", [])),
    }
    json_path = output_paths.base / str(output_config["json_report"])
    json_path.write_text(json.dumps(report, indent=2, sort_keys=True, default=str), encoding="utf-8")
    _write_markdown(report, output_paths.base / str(output_config["markdown_report"]))
    return report


def main() -> None:
    report = build_external_validation_report()
    print(json.dumps(report["summary"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
