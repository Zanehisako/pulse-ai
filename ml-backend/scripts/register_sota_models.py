#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path
from typing import Any

import pandas as pd

try:
    import mlflow
    from mlflow import MlflowClient
    from mlflow.models import infer_signature
except Exception as exc:  # pragma: no cover
    raise SystemExit(f"MLflow is required to register SOTA models: {exc}") from exc

ML_BACKEND_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = ML_BACKEND_ROOT.parent
if str(ML_BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(ML_BACKEND_ROOT))

from pios_ml_backend.sota import load_sota_catalog, resolve_catalog_path
from pios_ml_backend.sota.catalog import resolve_models_dir
from pios_ml_backend.sota.notebook_runtime import SotaNotebookModel


def _input_example(row: dict[str, Any]) -> pd.DataFrame:
    defaults = row.get("feature_defaults")
    if not isinstance(defaults, dict):
        defaults = {}
    feature_types = row.get("feature_types")
    if not isinstance(feature_types, dict):
        feature_types = {}
    payload: dict[str, list[Any]] = {}
    for feature in row["features"]:
        feature_type = str(feature_types.get(feature) or "").lower()
        default = defaults.get(feature)
        if feature_type in {"string", "category", "categorical"}:
            payload[feature] = [str(default or "")]
        elif feature_type in {"datetime", "date", "timestamp"}:
            payload[feature] = [pd.Timestamp.now(tz="UTC").tz_localize(None)]
        else:
            payload[feature] = [float(default or 0.0)]
    return pd.DataFrame(payload)


def _model_config(row: dict[str, Any], models_dir: Path, catalog: dict[str, Any]) -> dict[str, Any]:
    config = dict(row)
    config["artifact_path"] = str((models_dir / str(row["artifact_filename"])).resolve())
    metrics_filename = str(row.get("metrics_filename") or "").strip()
    if metrics_filename:
        config["metrics_path"] = str((models_dir / metrics_filename).resolve())
    neural_arch = catalog.get("neural_architecture")
    if isinstance(neural_arch, dict):
        config["neural_architecture"] = neural_arch
    return config


def _tracking_uri() -> str:
    configured = os.getenv("MLFLOW_TRACKING_URI", "").strip()
    if configured:
        return configured
    return "http://localhost:8889"


def _cleanup_malformed_experiments(tracking_uri: str) -> None:
    """Remove orphaned experiment directories missing meta.yaml."""
    if not tracking_uri.startswith("file://"):
        return
    root = Path(tracking_uri.removeprefix("file://"))
    if not root.is_dir():
        return
    skip = {".trash", "models", ".DS_Store"}
    for entry in root.iterdir():
        if entry.name in skip or not entry.is_dir():
            continue
        if not (entry / "meta.yaml").exists():
            shutil.rmtree(entry)


def register_models(*, catalog_path: str | Path | None, dry_run: bool = False) -> list[dict[str, Any]]:
    resolved_catalog_path = resolve_catalog_path(catalog_path)
    catalog = load_sota_catalog(catalog_path, require_artifacts=True)
    models_dir = resolve_models_dir(catalog, base_dir=resolved_catalog_path.parent)
    tracking_uri = _tracking_uri()
    mlflow.set_tracking_uri(tracking_uri)
    _cleanup_malformed_experiments(tracking_uri)
    experiment_name = str(
        catalog.get("mlflow_experiment_name")
        or os.getenv("MLFLOW_EXPERIMENT_NAME")
        or "Default"
    ).strip()
    if experiment_name:
        mlflow.set_experiment(experiment_name)
    client = MlflowClient()
    results: list[dict[str, Any]] = []

    for row in catalog["models"]:
        if not bool(row.get("enabled")):
            results.append({"id": row["id"], "status": "disabled"})
            continue

        config = _model_config(row, models_dir, catalog)
        registered_name = str(row["registered_model_name"])
        input_example = _input_example(row)
        # Force tabular inference to avoid torch.compile segfaults during registration
        reg_config = dict(config)
        reg_config["serve_winner"] = "tabular_gbdt"
        pyfunc_model = SotaNotebookModel(reg_config)
        sample_output = pyfunc_model.predict(None, input_example)
        signature = infer_signature(input_example, sample_output)

        if dry_run:
            results.append(
                {
                    "id": row["id"],
                    "registered_model_name": registered_name,
                    "status": "dry_run",
                    "input_columns": list(input_example.columns),
                    "output_columns": list(sample_output.columns),
                }
            )
            continue

        with mlflow.start_run(run_name=f"sota-{registered_name}"):
            mlflow.set_tags(
                {
                    "developer": "pios_ml",
                    "registered_model_name": registered_name,
                    "model_family": "sota_notebook",
                    "deployment_mode": str(row.get("deployment_mode") or "shadow"),
                    "task_type": str(row.get("task_type") or ""),
                    "thesis_use": "prototype",
                    "promotable": "false",
                }
            )
            mlflow.log_dict(config, "sota_model_config.json")
            metrics_value = str(config.get("metrics_path") or "").strip()
            metrics_path = Path(metrics_value) if metrics_value else None
            if metrics_path is not None and metrics_path.exists():
                try:
                    mlflow.log_dict(json.loads(metrics_path.read_text(encoding="utf-8")), "sota_metrics.json")
                except json.JSONDecodeError:
                    mlflow.log_artifact(str(metrics_path), artifact_path="sota_metrics")

            model_info = mlflow.pyfunc.log_model(
                artifact_path=registered_name,
                python_model=pyfunc_model,
                artifacts={"model_artifact": config["artifact_path"]},
                signature=signature,
                input_example=input_example,
                registered_model_name=registered_name,
            )

            client.set_registered_model_alias(
                name=registered_name,
                alias=str(catalog.get("target_alias") or "challenger"),
                version=model_info.registered_model_version,
            )
            if os.getenv("PIOS_PROMOTE_SOTA_MODELS", "").strip().lower() in {"1", "true", "yes"}:
                client.set_registered_model_alias(
                    name=registered_name,
                    alias="champion",
                    version=model_info.registered_model_version,
                )

            results.append(
                {
                    "id": row["id"],
                    "registered_model_name": registered_name,
                    "version": model_info.registered_model_version,
                    "status": "registered",
                    "tracking_uri": tracking_uri,
                }
            )
    return results


def main() -> int:
    parser = argparse.ArgumentParser(description="Register SOTA notebook artifacts as MLflow pyfunc models.")
    parser.add_argument("--catalog", default=None, help="Path to sota_model_catalog.json")
    parser.add_argument("--dry-run", action="store_true", help="Validate and run sample predictions without logging models")
    args = parser.parse_args()

    results = register_models(catalog_path=args.catalog, dry_run=args.dry_run)
    print(json.dumps({"results": results}, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
