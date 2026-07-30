from __future__ import annotations

import importlib
import json
import os
import sys
import traceback
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable

import pandas as pd


ROOT_DIRECTORY = Path(__file__).resolve().parents[1]
TRAINING_DIRECTORY = ROOT_DIRECTORY / "training_scripts"
CONFIG_PATH = ROOT_DIRECTORY / "config" / "training_schedule.json"
REPORT_DIRECTORY = ROOT_DIRECTORY / "artifacts" / "thesis_model_audit"

for path in (TRAINING_DIRECTORY, ROOT_DIRECTORY.parent / "backendMulti"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from model_training_utils import (  # noqa: E402
    TaskDataset,
    _enforce_metric_bounds,
    _feature_leakage_guard_config,
    _feature_columns_for_dataset,
    _is_future_or_label_column,
    _safe_numeric,
    _time_split,
    fit_task,
)


@dataclass
class ModelAuditResult:
    task_name: str
    registered_model_name: str
    script: str
    task_type: str
    target_column: str
    selected_candidate: str | None
    rows: int
    feature_count: int
    train_rows: int
    test_rows: int
    accuracy: float | None
    hard_accuracy: float | None
    root_mean_squared_error: float | None
    hard_root_mean_squared_error: float | None
    mean_absolute_error: float | None
    hard_mean_absolute_error: float | None
    r2_score: float | None
    hard_r2_score: float | None
    roc_auc: float | None
    hard_roc_auc: float | None
    leakage_status: str
    leakage_findings: list[str]
    temporal_split_status: str
    error: str | None = None


def _load_schedule() -> dict[str, Any]:
    payload = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("Training schedule root must be an object.")
    scripts = payload.get("scripts")
    if not isinstance(scripts, dict):
        raise ValueError("Training schedule must define scripts.")
    return payload


def _builder_for_script(script_name: str) -> Callable[[], TaskDataset]:
    module_name = Path(script_name).stem
    module = importlib.import_module(module_name)
    builders = [
        getattr(module, name)
        for name in dir(module)
        if name.startswith("build_")
        and name.endswith("_task")
        and callable(getattr(module, name))
    ]
    if len(builders) != 1:
        raise ValueError(
            f"{script_name} must expose exactly one build_*_task function."
        )
    return builders[0]


def _metric(metrics: dict[str, float], key: str) -> float | None:
    value = metrics.get(key)
    return float(value) if value is not None else None


def _audit_dataset(dataset: TaskDataset) -> tuple[list[str], str, int, int]:
    findings: list[str] = []
    features = _feature_columns_for_dataset(dataset.frame, dataset)
    feature_set = set(features)
    guard = _feature_leakage_guard_config()
    allowed_entity_features = {
        str(value).strip()
        for value in guard.get("allowed_entity_features", [])
        if str(value).strip()
    }

    forbidden = {
        dataset.target_column,
        dataset.timestamp_column,
        *dataset.entity_keys,
    } - allowed_entity_features
    leaked_forbidden = sorted(feature_set.intersection(forbidden))
    if leaked_forbidden:
        findings.append(f"Forbidden columns in features: {leaked_forbidden}")

    blocked_features = sorted(
        feature for feature in features if _is_future_or_label_column(feature)
    )
    if blocked_features:
        findings.append(f"Configured leakage-guard columns in features: {blocked_features}")

    train_frame, test_frame = _time_split(
        dataset.frame,
        dataset.timestamp_column,
        test_fraction=0.2,
    )
    temporal_status = "PASS"
    timestamp = dataset.timestamp_column
    if timestamp in train_frame.columns and timestamp in test_frame.columns:
        train_max = pd.to_datetime(
            train_frame[timestamp],
            utc=True,
            errors="coerce",
        ).max()
        test_min = pd.to_datetime(
            test_frame[timestamp],
            utc=True,
            errors="coerce",
        ).min()
        if pd.notna(train_max) and pd.notna(test_min) and train_max >= test_min:
            temporal_status = "FAIL"
            findings.append(
                f"Temporal split overlaps: train max {train_max}, test min {test_min}"
            )

    key_columns = [*dataset.entity_keys, dataset.timestamp_column]
    if all(column in train_frame.columns for column in key_columns) and all(
        column in test_frame.columns for column in key_columns
    ):
        train_keys = set(map(tuple, train_frame[key_columns].astype(str).to_numpy()))
        test_keys = set(map(tuple, test_frame[key_columns].astype(str).to_numpy()))
        overlap = train_keys.intersection(test_keys)
        if overlap:
            findings.append(f"Train/test entity-time overlap count: {len(overlap)}")

    numeric_target = _safe_numeric(dataset.frame[dataset.target_column])
    for feature in features:
        if not pd.api.types.is_numeric_dtype(dataset.frame[feature]):
            continue
        values = _safe_numeric(dataset.frame[feature])
        if values.nunique(dropna=True) <= 1 or numeric_target.nunique(dropna=True) <= 1:
            continue
        correlation = values.corr(numeric_target)
        if pd.notna(correlation) and abs(float(correlation)) >= 0.999:
            findings.append(
                f"Near-perfect feature/target correlation: {feature}={correlation:.4f}"
            )

    return findings, temporal_status, len(train_frame), len(test_frame)


def _audit_model(script_key: str, row: dict[str, Any]) -> ModelAuditResult:
    script = str(row["script"])
    registered_model_name = str(row.get("registered_model_name") or script_key)
    builder = _builder_for_script(script)
    dataset = builder()
    findings, temporal_status, train_rows, test_rows = _audit_dataset(dataset)
    task_result = fit_task(dataset)
    _enforce_metric_bounds(task_result)
    findings_after_fit, temporal_status_after_fit, train_rows, test_rows = (
        _audit_dataset(dataset)
    )
    findings = sorted(set([*findings, *findings_after_fit]))
    metrics = task_result.best_result.metrics
    hard_metrics = task_result.best_result.hard_metrics
    return ModelAuditResult(
        task_name=dataset.task_name,
        registered_model_name=registered_model_name,
        script=script,
        task_type=dataset.task_type,
        target_column=dataset.target_column,
        selected_candidate=task_result.best_result.candidate_name,
        rows=len(dataset.frame),
        feature_count=len(_feature_columns_for_dataset(dataset.frame, dataset)),
        train_rows=train_rows,
        test_rows=test_rows,
        accuracy=_metric(metrics, "accuracy"),
        hard_accuracy=_metric(hard_metrics, "accuracy"),
        root_mean_squared_error=_metric(metrics, "root_mean_squared_error"),
        hard_root_mean_squared_error=_metric(
            hard_metrics,
            "root_mean_squared_error",
        ),
        mean_absolute_error=_metric(metrics, "mean_absolute_error"),
        hard_mean_absolute_error=_metric(hard_metrics, "mean_absolute_error"),
        r2_score=_metric(metrics, "r2_score"),
        hard_r2_score=_metric(hard_metrics, "r2_score"),
        roc_auc=_metric(metrics, "roc_auc"),
        hard_roc_auc=_metric(hard_metrics, "roc_auc"),
        leakage_status="PASS" if not findings else "FAIL",
        leakage_findings=findings,
        temporal_split_status=(
            "PASS"
            if temporal_status == "PASS" and temporal_status_after_fit == "PASS"
            else "FAIL"
        ),
    )


def _write_markdown(results: list[ModelAuditResult]) -> None:
    rows = []
    for result in results:
        rows.append(
            [
                result.task_name,
                result.task_type,
                result.selected_candidate or "",
                _format_metric(result.accuracy),
                _format_metric(result.hard_accuracy),
                _format_metric(result.root_mean_squared_error),
                _format_metric(result.r2_score),
                result.leakage_status,
                result.temporal_split_status,
            ]
        )

    lines = [
        "# Thesis Model Audit",
        "",
        "Dry-fit audit with time-aware train/test splits and configured leakage guards.",
        "",
        "| Model | Type | Candidate | Accuracy | Hard accuracy | RMSE | R2 | Leakage | Temporal split |",
        "|---|---|---:|---:|---:|---:|---:|---|---|",
    ]
    for row in rows:
        lines.append("| " + " | ".join(row) + " |")
    lines.extend(["", "## Leakage Findings", ""])
    for result in results:
        if result.leakage_findings:
            lines.append(f"- {result.task_name}: {'; '.join(result.leakage_findings)}")
        else:
            lines.append(f"- {result.task_name}: no configured leakage findings")
    (REPORT_DIRECTORY / "model_audit_report.md").write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )


def _format_metric(value: float | None) -> str:
    if value is None:
        return ""
    return f"{value:.4f}"


def run_audit() -> list[ModelAuditResult]:
    schedule = _load_schedule()
    results: list[ModelAuditResult] = []
    for script_key, row in schedule["scripts"].items():
        if not isinstance(row, dict) or row.get("enabled") is not True:
            continue
        try:
            results.append(_audit_model(script_key, row))
        except Exception as exc:
            results.append(
                ModelAuditResult(
                    task_name=script_key,
                    registered_model_name=str(
                        row.get("registered_model_name") or script_key
                    ),
                    script=str(row.get("script") or ""),
                    task_type="unknown",
                    target_column="",
                    selected_candidate=None,
                    rows=0,
                    feature_count=0,
                    train_rows=0,
                    test_rows=0,
                    accuracy=None,
                    hard_accuracy=None,
                    root_mean_squared_error=None,
                    hard_root_mean_squared_error=None,
                    mean_absolute_error=None,
                    hard_mean_absolute_error=None,
                    r2_score=None,
                    hard_r2_score=None,
                    roc_auc=None,
                    hard_roc_auc=None,
                    leakage_status="ERROR",
                    leakage_findings=[],
                    temporal_split_status="ERROR",
                    error=f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}",
                )
            )
    return results


def main() -> int:
    os.environ.setdefault("PIOS_PROMOTE_TRAINED_MODELS", "0")
    REPORT_DIRECTORY.mkdir(parents=True, exist_ok=True)
    results = run_audit()
    frame = pd.DataFrame([asdict(result) for result in results])
    frame.to_csv(REPORT_DIRECTORY / "model_audit_summary.csv", index=False)
    (REPORT_DIRECTORY / "model_audit_report.json").write_text(
        json.dumps([asdict(result) for result in results], indent=2),
        encoding="utf-8",
    )
    _write_markdown(results)
    print(frame.to_string(index=False))
    failed = frame[
        frame["leakage_status"].ne("PASS")
        | frame["temporal_split_status"].ne("PASS")
        | frame["error"].notna()
    ]
    return 1 if not failed.empty else 0


if __name__ == "__main__":
    raise SystemExit(main())
