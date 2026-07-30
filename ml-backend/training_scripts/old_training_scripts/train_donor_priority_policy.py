"""
train_donor_priority_policy.py

Logs the donor priority policy as an MLflow pyfunc artifact.
This is a policy model (not ML-trained) that scores donors based on
configurable weights from donor_priority_policy_model.json.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import mlflow
import pandas as pd
from mlflow.models import infer_signature

_TRAINING_SCRIPTS = Path(__file__).resolve().parent
if str(_TRAINING_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_TRAINING_SCRIPTS))

_ML_BACKEND = _TRAINING_SCRIPTS.parent
if str(_ML_BACKEND) not in sys.path:
    sys.path.insert(0, str(_ML_BACKEND))

from pios_ml_backend.policies.donor_priority_policy import DonorPriorityPolicyModel
from pios_ml_backend.training_resource_profiles import (
    apply_profile_to_model_config,
    nested_bool,
)
from model_training_utils import configure_mlflow

_CONFIG_PATH = _ML_BACKEND / "config" / "donor_priority_policy_model.json"


def _load_config() -> dict:
    return apply_profile_to_model_config(
        json.loads(_CONFIG_PATH.read_text(encoding="utf-8"))
    )


def log_priority_policy() -> dict:
    config = _load_config()
    configure_mlflow()
    mlflow.set_experiment("donor_priority_policy")

    policy_model = DonorPriorityPolicyModel()

    input_features = config.get("policy_inputs", {}).get("donor_features", [])
    input_example = pd.DataFrame([{f: 0.0 for f in input_features}])
    for col in ["eligible_to_donate", "is_regular_donor", "blood_type", "is_rare_type"]:
        if col in input_example.columns:
            input_example[col] = [1.0 if col in ("eligible_to_donate", "is_regular_donor") else 0.0]

    sample_output = policy_model.predict(None, input_example)
    signature = infer_signature(input_example, sample_output)

    with mlflow.start_run(run_name="donor_priority_policy"):
        mlflow.set_tags({
            "developer": "pios_ml",
            "registered_model_name": config["registered_model_name"],
            "environment": "Training",
            "model_family": "policy",
            "training_design": "configurable_policy",
            "task_type": "policy",
            "description": config["description"],
            "promotable": "true",
            "feature_columns": json.dumps(input_features),
        })

        mlflow.log_dict(config, "policy_config.json")

        model_info = mlflow.pyfunc.log_model(
            artifact_path=config["registered_model_name"],
            python_model=policy_model,
            signature=signature,
            input_example=input_example if nested_bool(
                config.get("training", {}),
                ("mlflow", "input_example", "enabled"),
                True,
            ) else None,
            registered_model_name=config["registered_model_name"],
        )

        from mlflow import MlflowClient
        client = MlflowClient()
        client.set_registered_model_alias(
            name=config["registered_model_name"],
            alias="challenger",
            version=model_info.registered_model_version,
        )
        if __import__("os").getenv("PIOS_PROMOTE_TRAINED_MODELS", "0").strip().lower() in {"1", "true", "yes"}:
            client.set_registered_model_alias(
                name=config["registered_model_name"],
                alias="champion",
                version=model_info.registered_model_version,
            )

        return {
            "task_name": config["id"],
            "model_uri": f"runs:/{mlflow.active_run().info.run_id}/{config['registered_model_name']}" if mlflow.active_run() else None,
            "registered_model_version": model_info.registered_model_version,
        }


if __name__ == "__main__":
    result = log_priority_policy()
    print(f"Donor priority policy logged: v{result['registered_model_version']}")
    print("Done.")
