from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from unittest.mock import patch

import joblib


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "register_sota_models.py"


def _load_script_module():
    spec = importlib.util.spec_from_file_location("register_sota_models", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_register_dry_run_resolves_models_dir_relative_to_catalog_file(tmp_path):
    module = _load_script_module()
    config_dir = tmp_path / "config"
    models_dir = tmp_path / "models"
    config_dir.mkdir()
    models_dir.mkdir()
    joblib.dump({"type": "policy"}, models_dir / "policy.joblib")
    catalog_path = config_dir / "sota_model_catalog.json"
    catalog_path.write_text(
        json.dumps(
            {
                "default_models_dir": "../models",
                "models": [
                    {
                        "id": "policy_model",
                        "enabled": True,
                        "registered_model_name": "policy_model",
                        "artifact_filename": "policy.joblib",
                        "task_type": "policy",
                        "outputs": {"primary": "donor_priority_score"},
                        "features": ["readiness_score"],
                        "policy_formula": {"readiness_score": 1.0},
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    with (
        patch.object(module.mlflow, "set_tracking_uri"),
        patch.object(module.mlflow, "set_experiment"),
        patch.object(module, "MlflowClient"),
    ):
        results = module.register_models(catalog_path=catalog_path, dry_run=True)

    assert results == [
        {
            "id": "policy_model",
            "registered_model_name": "policy_model",
            "status": "dry_run",
            "input_columns": ["readiness_score"],
            "output_columns": [
                "prediction",
                "donor_priority_score",
                "display_donor_score",
                "recommended_action",
                "action_reason",
            ],
        }
    ]
