from __future__ import annotations

import json
import sys
from pathlib import Path


TRAINING_DIR = Path(__file__).resolve().parents[1] / "training_scripts"
if str(TRAINING_DIR) not in sys.path:
    sys.path.insert(0, str(TRAINING_DIR))

import generate_thesis_model_readiness as readiness  # noqa: E402


def test_thesis_model_readiness_report_uses_configured_gates(monkeypatch, tmp_path):
    config_path = tmp_path / "thesis_readiness_gates.json"
    artifacts = tmp_path / "artifacts"
    stale_dir = artifacts / "model_a"
    stale_dir.mkdir(parents=True)
    stale_dir.joinpath("metrics.json").write_text(
        json.dumps({"test_metrics": {"roc_auc": 0.5, "brier_score": 0.4}}),
        encoding="utf-8",
    )
    metrics_dir = artifacts / "model_a_artifacts"
    metrics_dir.mkdir(parents=True)
    metrics_dir.joinpath("metrics.json").write_text(
        json.dumps({"test_metrics": {"roc_auc": 0.95, "brier_score": 0.04}}),
        encoding="utf-8",
    )
    config_path.write_text(
        json.dumps(
            {
                "id": "test_gates",
                "models": {
                    "model_a": {
                        "metric_artifact_ids": ["model_a_artifacts"],
                        "required_metrics": {
                            "roc_auc": {"min": 0.9},
                            "brier_score": {"max": 0.05},
                        }
                    }
                },
            }
        ),
        encoding="utf-8",
    )

    monkeypatch.setattr(readiness, "CONFIG_PATH", config_path)
    monkeypatch.setattr(readiness, "ARTIFACTS_DIRECTORY", artifacts)
    monkeypatch.setattr(readiness, "OUTPUT_PATH", artifacts / "thesis_model_readiness.json")

    report = readiness.build_thesis_readiness_report()

    assert report["models"]["model_a"]["thesis_ready"] is True
    assert report["models"]["model_a"]["metric_source"] == str(metrics_dir / "metrics.json")
    assert (artifacts / "thesis_model_readiness.json").exists()
