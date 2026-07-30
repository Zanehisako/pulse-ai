from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from model_training_utils import ARTIFACTS_DIRECTORY


CONFIG_PATH = Path(__file__).resolve().parents[1] / "config" / "thesis_readiness_gates.json"
OUTPUT_PATH = ARTIFACTS_DIRECTORY / "thesis_model_readiness.json"


def _read_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except OSError:
        return {}
    if not isinstance(payload, dict):
        return {}
    return payload


def _candidate_metric_paths(model_id: str, gate: dict[str, Any] | None = None) -> list[Path]:
    artifact_ids: list[str] = []
    if isinstance(gate, dict):
        artifact_ids.extend(
            str(value).strip()
            for value in gate.get("metric_artifact_ids", [])
            if str(value).strip()
        )
    artifact_ids.append(model_id)
    paths: list[Path] = []
    for artifact_id in dict.fromkeys(artifact_ids):
        paths.extend(
            [
                ARTIFACTS_DIRECTORY / artifact_id / "metrics.json",
                ARTIFACTS_DIRECTORY / artifact_id / "best_metrics.json",
                ARTIFACTS_DIRECTORY / artifact_id / "training_metrics.json",
            ]
        )
    return paths


def _flatten_metrics(payload: dict[str, Any]) -> dict[str, float]:
    metrics: dict[str, float] = {}

    def visit(value: Any, prefix: str = "") -> None:
        if isinstance(value, dict):
            for key, item in value.items():
                clean_key = str(key).strip()
                next_prefix = f"{prefix}_{clean_key}" if prefix else clean_key
                if isinstance(item, (int, float)) and not isinstance(item, bool):
                    metrics[clean_key] = float(item)
                    metrics[next_prefix] = float(item)
                    if prefix.endswith("hard_metrics") or prefix.endswith("hard"):
                        metrics[f"hard_{clean_key}"] = float(item)
                else:
                    visit(item, next_prefix)

    visit(payload)
    return metrics


def _model_metrics(model_id: str, gate: dict[str, Any] | None = None) -> tuple[dict[str, float], str | None]:
    for path in _candidate_metric_paths(model_id, gate):
        payload = _read_json(path)
        if payload:
            return _flatten_metrics(payload), str(path)
    return {}, None


def _gate_status(metrics: dict[str, float], gate: dict[str, Any]) -> dict[str, Any]:
    rows = {}
    for metric_name, rule in gate.get("required_metrics", {}).items():
        if not isinstance(rule, dict):
            continue
        value = metrics.get(metric_name)
        passed = False
        if value is not None:
            min_value = rule.get("min")
            max_value = rule.get("max")
            passed = True
            if min_value is not None:
                passed = passed and value >= float(min_value)
            if max_value is not None:
                passed = passed and value <= float(max_value)
        rows[metric_name] = {
            "value": value,
            "rule": rule,
            "passed": bool(passed),
        }
    return rows


def build_thesis_readiness_report() -> dict[str, Any]:
    config = _read_json(CONFIG_PATH)
    models = config.get("models", {}) if isinstance(config.get("models"), dict) else {}
    report_rows: dict[str, Any] = {}
    for model_id, gate in models.items():
        if not isinstance(gate, dict):
            continue
        metrics, metric_source = _model_metrics(str(model_id), gate)
        gates = _gate_status(metrics, gate)
        report_rows[str(model_id)] = {
            "metric_source": metric_source,
            "metrics_found": bool(metrics),
            "baseline_surface": bool(gate.get("baseline_surface", False)),
            "gates": gates,
            "thesis_ready": bool(gates) and all(row["passed"] for row in gates.values()),
        }
    report = {
        "config_id": config.get("id"),
        "output_schema_version": 1,
        "models": report_rows,
    }
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(json.dumps(report, indent=2, sort_keys=True, default=str), encoding="utf-8")
    return report


if __name__ == "__main__":
    print(json.dumps(build_thesis_readiness_report(), indent=2, sort_keys=True, default=str))
