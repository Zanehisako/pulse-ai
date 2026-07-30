import json
import os
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

# Force UTF-8 encoding for stdout/stderr
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

# -----------------------------------------------------------------------------
# 1. SETUP SYSTEM PATHS BEFORE IMPORTING DJANGO
# -----------------------------------------------------------------------------
_SCRIPTS_DIR = Path(__file__).resolve().parent  # ml-backend/scripts/
_ROOT_DIR = _SCRIPTS_DIR.parent  # ml-backend/
_BACKEND_DIR = _ROOT_DIR.parent / "backendMulti"  # backendMulti/

sys.path.insert(0, str(_ROOT_DIR))
sys.path.insert(0, str(_BACKEND_DIR))  # Required for Django to find backendMulti

from pios_ml_backend.training_resource_profiles import (  # noqa: E402
    apply_selected_profile_environment,
    run_subprocess_with_profile_retries,
    selected_training_profile_name,
)

apply_selected_profile_environment(os.environ)

# -----------------------------------------------------------------------------
# 2. INITIALIZE DJANGO
# -----------------------------------------------------------------------------
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "backendMulti.settings")
import django

django.setup()

# -----------------------------------------------------------------------------
# 3. NOW IT IS SAFE TO IMPORT DJANGO MODELS AND THIRD-PARTY LIBRARIES
# -----------------------------------------------------------------------------
import mlflow
import pandas as pd
from django.utils import timezone
from feast import FeatureStore
from ml.models import MLModelConfig  # Safe now, django.setup() ran!
from mlflow import MlflowClient

# Setup MLFlow tracking URI
mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://localhost:8889"))
print("TRACKING URI:", mlflow.get_tracking_uri())

# -----------------------------------------------------------------------------
# MAIN SCRIPT LOGIC
# -----------------------------------------------------------------------------


def clean_orphan_mlruns(mlruns_dir: str | Path) -> None:
    """
    Remove any experiment or model directories that are missing their meta.yaml.
    Safe to call before every training run.
    """
    mlruns_path = Path(mlruns_dir)
    if not mlruns_path.exists():
        return

    # Clean orphan experiment directories (numeric IDs)
    for entry in mlruns_path.iterdir():
        if entry.is_dir() and entry.name.isdigit():
            meta = entry / "meta.yaml"
            if not meta.exists():
                print(f"🧹 Removing orphan experiment dir: {entry.name}")
                shutil.rmtree(entry)

    # Clean orphan model registry directories
    models_dir = mlruns_path / "models"
    if models_dir.exists():
        for model_entry in models_dir.iterdir():
            if model_entry.is_dir():
                meta = model_entry / "meta.yaml"
                if not meta.exists():
                    print(f"🧹 Removing orphan model dir: {model_entry.name}")
                    shutil.rmtree(model_entry)


# Remove the MLRuns folder to start fresh
MLRUNS_DIR = Path(__file__).parent / "mlruns"
try:
    if MLRUNS_DIR.exists():
        shutil.rmtree(MLRUNS_DIR)
        print("Successfully removed MLruns folder")
except Exception as e:
    print("error when removing MLRuns folder:", e)


def apply_and_materialize_feast() -> None:
    """Apply Feast schema changes and incrementally materialize new feature data.

    Uses materialize_incremental so only data written since the last run is
    processed. This avoids re-scanning years of historical data on every
    training cycle.
    """
    try:
        result = subprocess.run(
            ["feast", "apply"],
            cwd=str(_ROOT_DIR / "feature_repo"),
        )
        if result.returncode != 0:
            print(f"⚠️  feast apply exited with code {result.returncode}")
            return
    except subprocess.CalledProcessError as e:
        print(f"Command failed with return code {e.returncode}, error: {e.stderr}")
        return
    except FileNotFoundError:
        print("The command was not found. Check if the executable is in your PATH.")
        return

    try:
        store = FeatureStore(repo_path=str(_ROOT_DIR / "feature_repo"))
        end_date = timezone.now()
        store.materialize_incremental(end_date=end_date)
        print(f"✅ Feast online store incrementally materialized up to {end_date.isoformat()}.")
    except Exception as exc:
        print(f"⚠️  Feast materialization skipped: {exc}")


try:
    TRAINING_SCRIPTS_DIRECTORY = _ROOT_DIR / "training_scripts"
    print("script_directory", str(_SCRIPTS_DIR))
    print("root_directory", str(_ROOT_DIR))
    print("TRAINING_SCRIPTS_DIRECTORY", str(TRAINING_SCRIPTS_DIRECTORY))

    # Build an augmented environment for child processes
    child_env = os.environ.copy()
    extra_paths = [
        str(_ROOT_DIR),
        str(_BACKEND_DIR),
    ]
    existing = child_env.get("PYTHONPATH", "")
    child_env["PYTHONPATH"] = os.pathsep.join(
        extra_paths + ([existing] if existing else [])
    )
    child_env["MLFLOW_TRACKING_URI"] = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:8889")
    child_env.setdefault("DJANGO_SETTINGS_MODULE", "backendMulti.settings")
    child_env["PYTHONUTF8"] = "1"
    active_profile = selected_training_profile_name(env=child_env)
    apply_selected_profile_environment(child_env)
    print(f"Training resource profile: {active_profile}")

    _schedule_path = _ROOT_DIR / "config" / "training_schedule.json"
    try:
        with open(_schedule_path, "r", encoding="utf-8") as _f:
            _schedule = json.load(_f)
        training_script_names = [
            entry["script"]
            for entry in _schedule.get("scripts", {}).values()
            if entry.get("enabled", False) and entry.get("script")
        ]
        print(
            f"Loaded {len(training_script_names)} enabled training script(s) "
            f"from {_schedule_path.name}: {training_script_names}"
        )
    except FileNotFoundError:
        raise RuntimeError(
            f"Training schedule config not found: {_schedule_path}. "
            "Cannot determine which scripts to run."
        )
    except (json.JSONDecodeError, KeyError) as _e:
        raise RuntimeError(
            f"Invalid training schedule config {_schedule_path}: {_e}"
        )

    training_scripts = [
        TRAINING_SCRIPTS_DIRECTORY / name
        for name in training_script_names
        if (TRAINING_SCRIPTS_DIRECTORY / name).exists()
    ]

    for filepath in training_scripts:
        print(f"\nRunning training script: {filepath}")
        result = run_subprocess_with_profile_retries(
            [sys.executable, "-u", str(filepath)],
            env=child_env,
            profile_name=active_profile,
            on_retry=lambda failed, retry_profile: print(
                f"⚠️  {filepath.name} exited with {failed.returncode}; "
                f"retrying with training resource profile {retry_profile}"
            ),
        )
        if result.returncode != 0:
            print(
                f"❌ Training script {filepath.name} failed with exit code {result.returncode}"
            )

    apply_and_materialize_feast()

except subprocess.CalledProcessError as e:
    print(f"Command failed with return code {e.returncode}, error: {e.stderr}")
except FileNotFoundError:
    print("The command was not found. Check if the executable is in your PATH.")


# Initialize the MLflow client
client = MlflowClient()


def _select_registry_uri(client: MlflowClient, model_name: str, versions) -> tuple:
    """Pick the best MLflow URI for serving: champion -> challenger -> latest version."""
    for alias in ("champion", "challenger"):
        try:
            alias_mv = client.get_model_version_by_alias(model_name, alias)
            return alias_mv, f"models:/{model_name}@{alias}"
        except Exception:
            continue

    if versions:
        latest_mv = max(versions, key=lambda row: int(row.version))
        return latest_mv, f"models:/{model_name}/{latest_mv.version}"

    return None, f"models:/{model_name}"


# 1. Search for all registered models in the registry
registered_models = client.search_registered_models()

print(f"Found {len(registered_models)} registered models.\n" + "=" * 50)

for model in registered_models:
    examples = []
    try:
        tag = model.tags.get("examples", "")
        if tag:
            examples = json.loads(tag)
    except Exception:
        pass
    print(f"\nMODEL NAME: {model.name}")
    print(f"Description: {model.description or 'No description'}")
    print(f"Model Tags: {model.tags}")

    # 2. Get all versions of this specific model
    versions = client.search_model_versions(f"name='{model.name}'")

    selected_mv, selected_uri = _select_registry_uri(client, model.name, versions)
    if selected_mv is None:
        print(f"     No versions found for model: {model.name}")
        continue

    for mv in [selected_mv]:
        features = {}
        print(f"\n   ↳ Version: {mv.version}")
        print(f"     Aliases: {mv.aliases}")
        print(f"     Status: {mv.status}")
        print(f"     Model Source URI: {mv.source}")
        print(f"     Serving URI selected: {selected_uri}")

        # 3. Fetch the metadata from the specific Run that created this model
        if mv.run_id:
            try:
                run = client.get_run(mv.run_id)

                # Format the timestamp
                start_time = datetime.fromtimestamp(
                    run.info.start_time / 1000.0
                ).strftime("%Y-%m-%d %H:%M:%S")

                print(f"     Run ID: {mv.run_id}")
                print(f"     Trained on: {start_time}")
                print(
                    f"     Run Tags: {run.data.tags.get('environment', 'N/A')} | {run.data.tags.get('feast_feature_service', 'N/A')}"
                )

                print(f"     Metrics:")
                for metric_name, metric_value in run.data.metrics.items():
                    if metric_name in [
                        "training_roc_auc_score",
                        "roc_auc",
                        "accuracy_score",
                    ]:
                        print(f"        - {metric_name}: {metric_value:.4f}")

                # ---------------------------------------------------------
                # 1. GET FEATURE COLUMNS
                # Read from the run tag logged at training time (no artifact
                # download needed).  Fall back to get_model_info() for runs
                # trained before this tag was introduced.
                # ---------------------------------------------------------
                model_uri = f"models:/{model.name}/{mv.version}"
                raw_fc_tag = run.data.tags.get("feature_columns", "")
                if raw_fc_tag:
                    try:
                        col_names = json.loads(raw_fc_tag)
                        features = {col: "unknown" for col in col_names}
                        print(f"📦 FEATURES (from run tag): {len(features)} columns")
                    except (json.JSONDecodeError, TypeError):
                        features = {}

                if not features:
                    # Legacy path: download MLmodel to read signature
                    try:
                        model_info = mlflow.models.get_model_info(model_uri)
                        print("📦 MODEL SIGNATURE (SCHEMA):")
                        if model_info.signature:
                            print("\n  Inputs:")
                            for input_col in model_info.signature.inputs:
                                print(f"    - {input_col.name}: {input_col.type}")
                                features[input_col.name] = str(input_col.type)
                            print("\n  Outputs:")
                            for output_col in model_info.signature.outputs:
                                print(f"    - Output Type: {output_col.type}")
                        else:
                            print("  No signature was logged for this model.")
                    except Exception as e:
                        print(f"Error fetching signature: {e}")

                print("\n" + "=" * 50)

                # Saving the model to the registry (or updating if it already exists)
                def _deep_merge_dicts(base: dict, override: dict) -> dict:
                    merged = dict(base)
                    for key, value in override.items():
                        if isinstance(value, dict) and isinstance(merged.get(key), dict):
                            merged[key] = _deep_merge_dicts(merged[key], value)
                        else:
                            merged[key] = value
                    return merged

                def _download_json_artifact(artifact_name: str) -> dict:
                    # prediction_defaults.json is logged at the run artifact root
                    # (via mlflow.log_dict), NOT inside the model subdirectory.
                    # Use run_id + artifact_path to reach it correctly.
                    try:
                        artifact_path = mlflow.artifacts.download_artifacts(
                            run_id=mv.run_id,
                            artifact_path=artifact_name,
                        )
                        with open(artifact_path, "r", encoding="utf-8") as handle:
                            payload = json.load(handle)
                    except Exception:
                        return {}
                    return payload if isinstance(payload, dict) else {}

                defaults = {}
                feast_feature_service = str(
                    run.data.tags.get("feast_feature_service", "")
                    or run.data.params.get("feature_service", "")
                    or ""
                ).strip()
                if feast_feature_service:
                    raw_entity_keys = str(
                        run.data.tags.get("feast_entity_keys", "")
                        or run.data.params.get("entity_keys", "")
                        or ""
                    ).strip()
                    try:
                        parsed_entity_keys = json.loads(raw_entity_keys)
                    except json.JSONDecodeError:
                        parsed_entity_keys = []
                    if isinstance(parsed_entity_keys, str):
                        parsed_entity_keys = [parsed_entity_keys]
                    if not isinstance(parsed_entity_keys, list):
                        parsed_entity_keys = []
                    feast_defaults = {
                        "feature_service": feast_feature_service,
                        "repo_path": str((_ROOT_DIR / "feature_repo").resolve()),
                        "allow_direct_input": True,
                        "allow_feature_overrides": True,
                        "fail_open": False,
                    }
                    clean_entity_keys = [
                        str(item).strip()
                        for item in parsed_entity_keys
                        if str(item).strip()
                    ]
                    if clean_entity_keys:
                        feast_defaults["entity_keys"] = clean_entity_keys
                    defaults = {
                        "prediction_source": "feast_online",
                        "feast": feast_defaults,
                    }

                prediction_target = str(
                    run.data.tags.get("prediction_target", "")
                    or run.data.params.get("prediction_target", "")
                    or ""
                ).strip()
                if (
                    str(run.data.tags.get("task_type", "")).strip() == "regression"
                    and "stockout" in prediction_target
                ):
                    defaults["forecast_selection"] = {
                        "enabled": True,
                        "target": "stockout",
                        "primary_metric": "root_mean_squared_error",
                        "direction": "minimize",
                        "fallback_metrics": [
                            "mean_absolute_error",
                            "r2_score",
                            "accuracy",
                        ],
                    }
                    defaults["output"] = {
                        "name": "days_until_stockout",
                        "unit": "days",
                        "task_type": "regression",
                        "clip_min": 0.0,
                    }
                artifact_defaults = _download_json_artifact("prediction_defaults.json")
                if artifact_defaults:
                    defaults = _deep_merge_dicts(defaults, artifact_defaults)

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

                print(
                    f"created model {model.name} "
                    if created
                    else f"updated model {model.name}",
                    obj.model_id,
                )

            except Exception as e:
                print(f"     [Could not fetch run metadata: {e}]")

print("\n" + "=" * 50)
