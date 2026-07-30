"""
sync_registry.py
================
Syncs MLflow registered models → Django ``MLModelConfig`` table.

This script is called by the training scheduler after individual training
scripts finish.  Unlike ``train_models.py`` it does **not** run any training
scripts — it only reads the MLflow model registry and upserts the Django DB.
"""

import json
import os
import sys
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

_SCRIPTS_DIR = Path(__file__).resolve().parent  # ml-backend/scripts/
_ROOT_DIR = _SCRIPTS_DIR.parent  # ml-backend/
_BACKEND_DIR = _ROOT_DIR.parent / "backendMulti"  # backendMulti/

sys.path.insert(0, str(_ROOT_DIR))
sys.path.insert(0, str(_BACKEND_DIR))

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "backendMulti.settings")

import django

django.setup()

# ── MLflow setup ──────────────────────────────────────────────────────────

import mlflow
import pandas as pd
from ml.models import MLModelConfig
from mlflow import MlflowClient

mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://localhost:8889"))
print("[sync_registry] TRACKING URI:", mlflow.get_tracking_uri())

# ── Main sync logic ──────────────────────────────────────────────────────

client = MlflowClient()
registered_models = client.search_registered_models()

print(f"[sync_registry] Found {len(registered_models)} registered models.\n" + "=" * 50)


def _infer_entity_keys(feature_service: str) -> list[str]:
    cleaned = feature_service.strip()
    if cleaned.endswith("_service"):
        entity_key = cleaned[: -len("_service")].strip()
        if entity_key:
            return [entity_key]
    return []


def _parse_entity_keys(raw_value: str) -> list[str]:
    text = str(raw_value or "").strip()
    if not text:
        return []

    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        parsed = None

    if isinstance(parsed, list):
        return [str(value).strip() for value in parsed if str(value).strip()]
    if isinstance(parsed, str) and parsed.strip():
        return [parsed.strip()]

    return [part.strip() for part in text.split(",") if part.strip()]


def _deep_merge_dicts(base: dict, override: dict) -> dict:
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge_dicts(merged[key], value)
        else:
            merged[key] = value
    return merged


def _download_json_artifact(model_uri: str, artifact_name: str) -> dict:
    try:
        artifact_path = mlflow.artifacts.download_artifacts(
            artifact_uri=f"{model_uri}/{artifact_name}"
        )
        with open(artifact_path, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def _alias_priority() -> list[str]:
    raw = os.getenv("PIOS_REGISTRY_ALIAS_PRIORITY", "champion,challenger")
    aliases = [item.strip() for item in raw.split(",") if item.strip()]
    return aliases or ["champion", "challenger"]


def _selected_model_version(model_name: str, versions: list):
    for alias in _alias_priority():
        try:
            mv = client.get_model_version_by_alias(model_name, alias)
            return mv, f"models:/{model_name}@{alias}", alias
        except Exception:
            continue
    if not versions:
        return None, "", ""
    mv = max(versions, key=lambda row: int(row.version))
    return mv, f"models:/{model_name}/{mv.version}", f"version-{mv.version}"


for model in registered_models:
    examples = []
    try:
        tag = model.tags.get("examples", "")
        if tag:
            examples = json.loads(tag)
    except Exception:
        pass

    print(f"\n  MODEL: {model.name}")

    versions = client.search_model_versions(f"name='{model.name}'")

    mv, selected_uri, selected_label = _selected_model_version(model.name, versions)
    if mv is None:
        print("    [No model versions found]")
        continue
    for mv in [mv]:
        features = {}

        if mv.run_id:
            try:
                run = client.get_run(mv.run_id)
                start_time = datetime.fromtimestamp(
                    run.info.start_time / 1000.0
                ).strftime("%Y-%m-%d %H:%M:%S")
                print(f"    Version {mv.version} | selected {selected_label} | trained {start_time}")

                # Metrics
                for metric_name, metric_value in run.data.metrics.items():
                    if metric_name in [
                        "training_roc_auc_score",
                        "roc_auc",
                        "accuracy_score",
                    ]:
                        print(f"      {metric_name}: {metric_value:.4f}")

                # Signature (features)
                try:
                    model_uri = f"models:/{model.name}/{mv.version}"
                    model_info = mlflow.models.get_model_info(model_uri)
                    if model_info.signature:
                        for input_col in model_info.signature.inputs:
                            features[input_col.name] = str(input_col.type)
                except Exception as e:
                    print(f"    [signature error: {e}]")

                # Input example
                try:
                    model_uri = f"models:/{model.name}/{mv.version}"
                    artifact_uri = f"{model_uri}/input_example.json"
                    example_file_path = mlflow.artifacts.download_artifacts(
                        artifact_uri=artifact_uri
                    )
                    with open(example_file_path, "r") as f:
                        example_data = json.load(f)
                    if "dataframe_split" in example_data:
                        df_example = pd.DataFrame(
                            data=example_data["dataframe_split"]["data"],
                            columns=example_data["dataframe_split"]["columns"],
                        )
                        print(
                            f"    Input example: {df_example.shape[0]} rows, {df_example.shape[1]} cols"
                        )
                except Exception:
                    pass

                feast_feature_service = (
                    str(run.data.tags.get("feast_feature_service", "")).strip()
                    or str(run.data.params.get("feature_service", "")).strip()
                )
                defaults = {}
                model_uri = f"models:/{model.name}/{mv.version}"
                if feast_feature_service:
                    feast_entity_keys = (
                        _parse_entity_keys(
                            str(run.data.tags.get("feast_entity_keys", "")).strip()
                        )
                        or _parse_entity_keys(
                            str(run.data.params.get("entity_keys", "")).strip()
                        )
                        or _infer_entity_keys(feast_feature_service)
                    )
                    defaults = {
                        "prediction_source": "feast_online",
                        "feast": {
                            "feature_service": feast_feature_service,
                            "repo_path": str((_ROOT_DIR / "feature_repo").resolve()),
                            "entity_keys": feast_entity_keys,
                            "allow_direct_input": True,
                            "allow_feature_overrides": True,
                            "fail_open": False,
                        },
                    }
                artifact_defaults = _download_json_artifact(
                    model_uri,
                    "prediction_defaults.json",
                )
                if artifact_defaults:
                    defaults = _deep_merge_dicts(defaults, artifact_defaults)

                # Upsert to Django DB
                obj, created = MLModelConfig.objects.update_or_create(
                    model_id=model.name,
                    defaults={
                        "description": model.description or "",
                        "file_path": selected_uri,
                        "model_type": "mlflow",
                        "features": list(features.keys()),
                        "feature_info": features,
                        "examples": examples,
                        "defaults": defaults,
                        "is_active": True,
                    },
                )
                action = "created" if created else "updated"
                print(f"    ✅ {action} → {obj.model_id}")

            except Exception as e:
                print(f"    [Could not sync: {e}]")

print("\n" + "=" * 50)
print("[sync_registry] Done.")
